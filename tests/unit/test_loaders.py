"""Loader plugins: registry, artifact routing, pickle policy, shared helpers.

Framework-specific fidelity / SHAP tests live in ``test_loaders_{lightgbm,xgboost,sklearn,onnx}.py``;
they import the helpers defined here (``make_binary_data``, ``assert_faithful``).
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from malvalid import registry
from malvalid.core import PickleRefused
from malvalid.loaders.base import (
    PICKLE_EXTENSIONS,
    ModelLoader,
    find_loader_for_native,
    is_pickle_artifact,
    load_model,
)
from malvalid.loaders.lightgbm_loader import (
    LightGBMLoader,
    ensemble_thresholds,
    pickle_claims,
    pickle_top_module,
    probe_matrix,
)
from malvalid.loaders.onnx_loader import ONNXLoader
from malvalid.loaders.sklearn_loader import SklearnLoader
from malvalid.loaders.trees import TreeEnsemble
from malvalid.loaders.xgboost_loader import XGBoostLoader

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

THREADS = 4

# --------------------------------------------------------------------------------------------------
# shared helpers (imported by the per-framework test modules)
# --------------------------------------------------------------------------------------------------


def make_binary_data(
    n: int = 600, d: int = 8, *, seed: int = 0, nan_frac: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """float32 features with heavy ties (few distinct values, many exact zeros) and a noisy target."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X[:, 1] = np.round(X[:, 1] * 2.0) / 2.0  # ~9 distinct values
    X[:, 2] = np.where(rng.random(n) < 0.5, 0.0, rng.integers(1, 4, n)).astype(np.float32)  # zeros
    logit = 1.5 * X[:, 0] - X[:, 1] + 0.8 * X[:, 2] + 0.7 * X[:, 3] * X[:, 4] + 0.4 * rng.normal(size=n)
    y = (logit > 0).astype(np.int64)
    if nan_frac > 0:
        X[rng.random(X.shape) < nan_frac] = np.nan
    return X, y


def eval_rows(te: TreeEnsemble, X: np.ndarray, *, nan: bool = True, n_probe: int = 400, seed: int = 1) -> np.ndarray:
    """Training-like rows plus probe rows sitting exactly on / next to every split threshold."""
    P = probe_matrix(ensemble_thresholds(te), X.shape[1], n_rows=n_probe, seed=seed, nan=nan)
    return np.vstack([X[:200].astype(np.float32), P])


def assert_faithful(
    loader: ModelLoader,
    native: Any,
    X: np.ndarray,
    *,
    nan: bool = True,
    tol: float = 1e-6,
    shap_rows: int = 80,
) -> TreeEnsemble:
    """TreeEnsemble == native probability (<= tol), SHAP additive in raw space, bytes round-trip."""
    import shap

    te = loader.tree_ensemble(native)
    Z = eval_rows(te, X, nan=nan)
    p_native = loader.predict_proba(native, Z)
    p_te = te.predict_proba(Z)
    assert p_native.shape == p_te.shape == (Z.shape[0],)
    err = float(np.max(np.abs(p_native - p_te)))
    assert err <= tol, f"{te.model_kind}: tree ensemble differs from native predict_proba by {err:.3g}"
    assert te.fidelity(Z, p_native) <= tol

    # SHAP TreeExplainer on the normalized model is exactly additive in raw (margin) space.
    rows = np.vstack([Z[:shap_rows // 2], Z[-(shap_rows // 2):]]).astype(np.float64)
    explainer = shap.TreeExplainer(te.to_shap_model())
    sv = np.asarray(explainer.shap_values(rows))
    assert sv.shape == rows.shape
    base = float(np.ravel(explainer.expected_value)[0])
    scale = te.sigmoid_scale if te.output_transform == "logistic" else 1.0
    np.testing.assert_allclose(sv.sum(axis=1) + base, scale * te.predict_raw(rows), rtol=0, atol=1e-6)

    # safe (pickle-free) serialization round trip
    te2 = TreeEnsemble.from_bytes(te.to_bytes())
    assert te2.n_trees == te.n_trees and te2.n_features == te.n_features
    assert te2.base_score == te.base_score and te2.output_transform == te.output_transform
    np.testing.assert_array_equal(te2.predict_raw(Z), te.predict_raw(Z))
    assert te2.meta == te.meta
    return te


# --------------------------------------------------------------------------------------------------
# fixtures: one small model per framework, saved in every format the loaders accept
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    return make_binary_data(400, 6, seed=3)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory, data: tuple[np.ndarray, np.ndarray]) -> dict[str, Path]:
    import joblib
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import GradientBoostingClassifier
    from skl2onnx import to_onnx

    X, y = data
    d = tmp_path_factory.mktemp("artifacts")
    out: dict[str, Path] = {}

    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 7, "num_threads": THREADS}, lgb.Dataset(X, y), 5
    )
    booster.save_model(str(d / "lgb.txt"))
    booster.save_model(str(d / "lgb_text.model"))
    out["lgb_txt"], out["lgb_model"] = d / "lgb.txt", d / "lgb_text.model"
    clf = lgb.LGBMClassifier(n_estimators=5, num_leaves=7, verbose=-1, n_jobs=THREADS).fit(X, y)
    joblib.dump(clf, d / "lgb_clf.joblib")
    with open(d / "lgb_booster.pkl", "wb") as f:
        pickle.dump(booster, f)
    out["lgb_joblib"], out["lgb_pkl"] = d / "lgb_clf.joblib", d / "lgb_booster.pkl"

    xb = xgb.train({"objective": "binary:logistic", "max_depth": 3, "nthread": THREADS}, xgb.DMatrix(X, y), 5)
    for name in ("xgb.json", "xgb.ubj", "xgb_ubj.model"):
        with warnings.catch_warnings():  # ".model" -> XGBoost's "UBJSON by default" notice (intended)
            warnings.filterwarnings("ignore", message=".*Saving model in the UBJSON format as default.*")
            xb.save_model(str(d / name))
    out["xgb_json"], out["xgb_ubj"], out["xgb_model"] = d / "xgb.json", d / "xgb.ubj", d / "xgb_ubj.model"
    xclf = xgb.XGBClassifier(n_estimators=5, max_depth=3, n_jobs=THREADS).fit(X, y)
    joblib.dump(xclf, d / "xgb_clf.pkl")
    out["xgb_pkl"] = d / "xgb_clf.pkl"

    gbc = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=0).fit(X, y)
    joblib.dump(gbc, d / "gbc.joblib")
    with open(d / "gbc.pkl", "wb") as f:
        pickle.dump(gbc, f)
    out["sk_joblib"], out["sk_pkl"] = d / "gbc.joblib", d / "gbc.pkl"

    onx = to_onnx(gbc, X[:1], target_opset={"": 17, "ai.onnx.ml": 3})
    (d / "gbc.onnx").write_bytes(onx.SerializeToString())
    out["onnx"] = d / "gbc.onnx"

    # A pickle hiding behind "safe" extensions: must be caught by content sniffing.
    for name in ("disguised.txt", "disguised.json", "disguised.onnx", "disguised.ubj"):
        with open(d / name, "wb") as f:
            pickle.dump({"not": "a model"}, f, protocol=4)
        out[name] = d / name
    return out


@pytest.fixture()
def no_unpickling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if anything tries to deserialize a pickle."""
    import joblib

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("a pickle was deserialized before the pickle policy was enforced")

    monkeypatch.setattr(joblib, "load", boom)
    monkeypatch.setattr(pickle, "load", boom)
    monkeypatch.setattr(pickle, "loads", boom)


# --------------------------------------------------------------------------------------------------
# registry / class attributes
# --------------------------------------------------------------------------------------------------

ALL_LOADERS = (LightGBMLoader, XGBoostLoader, SklearnLoader, ONNXLoader)


def test_registry_exposes_all_four_loaders() -> None:
    loaders = registry.model_loaders()
    assert {k: loaders[k] for k in ("lightgbm", "xgboost", "sklearn_gbdt", "onnx")} == {
        "lightgbm": LightGBMLoader,
        "xgboost": XGBoostLoader,
        "sklearn_gbdt": SklearnLoader,
        "onnx": ONNXLoader,
    }


@pytest.mark.parametrize("cls", ALL_LOADERS)
def test_loader_class_attributes(cls: type[ModelLoader]) -> None:
    ld = cls()
    assert ld.kind and ld.description
    assert all(e.startswith(".") and e == e.lower() for e in ld.extensions)
    assert set(ld.safe_extensions) <= set(ld.extensions)
    assert not set(ld.safe_extensions) & set(PICKLE_EXTENSIONS), "a pickle extension is marked safe"
    info = ld.info()
    assert info["kind"] == ld.kind and info["safe_extensions"] == list(ld.safe_extensions)


def test_sklearn_has_no_safe_format_and_onnx_is_safe() -> None:
    assert SklearnLoader.safe_extensions == ()
    assert ONNXLoader.safe_extensions == (".onnx",)
    assert ".txt" in LightGBMLoader.safe_extensions
    assert {".json", ".ubj"} <= set(XGBoostLoader.safe_extensions)


# --------------------------------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------------------------------

ROUTES = {
    "lgb_txt": {"lightgbm"},
    "lgb_model": {"lightgbm"},  # LightGBM text model saved as .model: not XGBoost's
    "xgb_json": {"xgboost"},
    "xgb_ubj": {"xgboost"},
    "xgb_model": {"xgboost"},  # XGBoost UBJSON saved as .model: not LightGBM's
    "onnx": {"onnx"},
    "lgb_joblib": {"lightgbm"},  # pickles are routed by the framework their opcodes reference
    "lgb_pkl": {"lightgbm"},
    "xgb_pkl": {"xgboost"},
    "sk_joblib": {"sklearn_gbdt"},
    "sk_pkl": {"sklearn_gbdt"},
}


@pytest.mark.parametrize("key", sorted(ROUTES))
def test_can_load_routes_each_artifact_to_one_loader(artifacts: dict[str, Path], key: str) -> None:
    claimed = {cls.kind for cls in ALL_LOADERS if cls().can_load(artifacts[key])}
    assert claimed == ROUTES[key]


def test_can_load_rejects_unknown_extensions(tmp_path: Path) -> None:
    p = tmp_path / "model.h5"
    p.write_bytes(b"\x89HDF\r\n")
    assert not any(cls().can_load(p) for cls in ALL_LOADERS)


def test_pickle_top_module_is_static(artifacts: dict[str, Path], no_unpickling: None) -> None:
    assert pickle_top_module(artifacts["lgb_joblib"]).startswith("lightgbm")
    assert pickle_top_module(artifacts["lgb_pkl"]).startswith("lightgbm")
    assert pickle_top_module(artifacts["xgb_pkl"]).startswith("xgboost")
    assert pickle_top_module(artifacts["sk_joblib"]).startswith("sklearn")
    assert pickle_top_module(artifacts["disguised.txt"]) is None  # only builtins inside
    assert pickle_claims(artifacts["disguised.txt"], ("sklearn",))  # undeterminable -> by extension
    assert not pickle_claims(artifacts["xgb_pkl"], ("sklearn",))


@pytest.mark.parametrize(
    ("key", "expected_type"),
    [
        ("lgb_txt", "Booster"),
        ("lgb_model", "Booster"),
        ("xgb_json", "Booster"),
        ("xgb_ubj", "Booster"),
        ("xgb_model", "Booster"),
        ("onnx", "ONNXModel"),
    ],
)
def test_load_model_autodetects_safe_formats(
    artifacts: dict[str, Path], data: tuple[np.ndarray, np.ndarray], key: str, expected_type: str, no_unpickling: None
) -> None:
    native = load_model(artifacts[key], allow_pickle=False)
    assert type(native).__name__ == expected_type
    ld = find_loader_for_native(native)
    assert ld is not None and ld.kind in ROUTES[key]
    p = ld.predict_proba(native, data[0][:50])
    assert p.shape == (50,) and np.all((p >= 0) & (p <= 1))


def test_load_model_with_explicit_kind(artifacts: dict[str, Path]) -> None:
    assert type(load_model(artifacts["xgb_json"], kind="xgboost")).__name__ == "Booster"
    with pytest.raises(KeyError, match="no model loader for kind"):
        load_model(artifacts["xgb_json"], kind="tensorflow")


# --------------------------------------------------------------------------------------------------
# pickle policy: refused BEFORE any deserialization unless allowed
# --------------------------------------------------------------------------------------------------

PICKLE_CASES = [
    (LightGBMLoader, "lgb_joblib"),
    (LightGBMLoader, "lgb_pkl"),
    (LightGBMLoader, "disguised.txt"),  # pickle bytes behind a safe extension
    (XGBoostLoader, "xgb_pkl"),
    (XGBoostLoader, "disguised.json"),
    (XGBoostLoader, "disguised.ubj"),
    (SklearnLoader, "sk_joblib"),
    (SklearnLoader, "sk_pkl"),
    (ONNXLoader, "disguised.onnx"),
]


@pytest.mark.parametrize(("cls", "key"), PICKLE_CASES)
def test_pickles_refused_without_allow_pickle(
    artifacts: dict[str, Path], cls: type[ModelLoader], key: str, no_unpickling: None
) -> None:
    ld = cls()
    assert is_pickle_artifact(artifacts[key])
    with pytest.raises(PickleRefused, match="allow-pickle"):
        ld.load(artifacts[key])
    with pytest.raises(PickleRefused):
        ld.load(artifacts[key], allow_pickle=False)


def test_sklearn_loader_treats_every_artifact_as_pickle(tmp_path: Path, no_unpickling: None) -> None:
    p = tmp_path / "model.sav"
    p.write_bytes(b"not obviously a pickle")
    assert SklearnLoader().is_pickle(p)
    with pytest.raises(PickleRefused):
        SklearnLoader().load(p)


@pytest.mark.parametrize(
    ("cls", "key", "type_name"),
    [
        (LightGBMLoader, "lgb_joblib", "LGBMClassifier"),
        (LightGBMLoader, "lgb_pkl", "Booster"),
        (XGBoostLoader, "xgb_pkl", "XGBClassifier"),
        (SklearnLoader, "sk_joblib", "GradientBoostingClassifier"),
        (SklearnLoader, "sk_pkl", "GradientBoostingClassifier"),
    ],
)
def test_pickles_load_with_allow_pickle(
    artifacts: dict[str, Path], data: tuple[np.ndarray, np.ndarray], cls: type[ModelLoader], key: str, type_name: str
) -> None:
    ld = cls()
    native = ld.load(artifacts[key], allow_pickle=True)
    assert type(native).__name__ == type_name
    assert ld.supports_native(native)
    p = ld.predict_proba(native, data[0][:30])
    assert p.shape == (30,) and np.all((p >= 0) & (p <= 1))
    assert ld.tree_ensemble(native).n_trees == 5


@pytest.mark.parametrize(("cls", "key"), [(LightGBMLoader, "disguised.txt"), (XGBoostLoader, "disguised.json")])
def test_allowed_pickle_of_wrong_type_is_rejected(artifacts: dict[str, Path], cls: type[ModelLoader], key: str) -> None:
    with pytest.raises(TypeError, match="not an? "):
        cls().load(artifacts[key], allow_pickle=True)


def test_onnx_loader_never_unpickles_even_when_allowed(artifacts: dict[str, Path], no_unpickling: None) -> None:
    with pytest.raises(ValueError, match="never unpickles"):
        ONNXLoader().load(artifacts["disguised.onnx"], allow_pickle=True)


def test_load_model_defers_to_sandbox_env(
    artifacts: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MALVALID_ALLOW_PICKLE", raising=False)
    with pytest.raises(PickleRefused):
        load_model(artifacts["sk_joblib"])
    monkeypatch.setenv("MALVALID_ALLOW_PICKLE", "0")
    with pytest.raises(PickleRefused):
        load_model(artifacts["sk_joblib"], kind="sklearn_gbdt")
    monkeypatch.setenv("MALVALID_ALLOW_PICKLE", "1")
    assert type(load_model(artifacts["sk_joblib"], kind="sklearn_gbdt")).__name__ == "GradientBoostingClassifier"


@pytest.mark.parametrize("cls", ALL_LOADERS)
def test_missing_file_is_a_clear_error(tmp_path: Path, cls: type[ModelLoader]) -> None:
    ext = cls.extensions[0]
    with pytest.raises(FileNotFoundError, match="not found"):
        cls().load(tmp_path / f"absent{ext}", allow_pickle=True)


def test_wrong_format_for_safe_extension(tmp_path: Path) -> None:
    bad = tmp_path / "model.txt"
    bad.write_text("hello\n")
    with pytest.raises(ValueError, match="not a LightGBM text model"):
        LightGBMLoader().load(bad)
    badj = tmp_path / "model.json"
    badj.write_text('{"hello": 1}')
    with pytest.raises(ValueError, match="Booster.load_model"):
        XGBoostLoader().load(badj)


# --------------------------------------------------------------------------------------------------
# native detection
# --------------------------------------------------------------------------------------------------


def test_find_loader_for_native_objects(artifacts: dict[str, Path]) -> None:
    import onnxruntime as ort

    assert find_loader_for_native(load_model(artifacts["lgb_txt"])).kind == "lightgbm"
    assert find_loader_for_native(load_model(artifacts["xgb_json"])).kind == "xgboost"
    assert find_loader_for_native(SklearnLoader().load(artifacts["sk_joblib"], allow_pickle=True)).kind == "sklearn_gbdt"
    assert find_loader_for_native(LightGBMLoader().load(artifacts["lgb_joblib"], allow_pickle=True)).kind == "lightgbm"
    assert find_loader_for_native(XGBoostLoader().load(artifacts["xgb_pkl"], allow_pickle=True)).kind == "xgboost"
    sess = ort.InferenceSession(str(artifacts["onnx"]), providers=["CPUExecutionProvider"])
    assert find_loader_for_native(sess).kind == "onnx"
    assert find_loader_for_native(str(artifacts["onnx"])).kind == "onnx"
    assert find_loader_for_native(object()) is None
    assert find_loader_for_native({"trees": []}) is None


@pytest.mark.parametrize("cls", ALL_LOADERS)
def test_supports_native_rejects_foreign_objects(cls: type[ModelLoader]) -> None:
    from sklearn.linear_model import LogisticRegression

    ld = cls()
    assert not ld.supports_native(object())
    assert not ld.supports_native(np.zeros(3))
    if cls is not SklearnLoader:
        assert not ld.supports_native(LogisticRegression())
