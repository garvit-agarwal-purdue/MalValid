"""XGBoost loader: strict '<' -> '<=' conversion, default_left, base_score margin (xgboost 3.x)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from malvalid.core import UnsupportedTreeError
from malvalid.loaders.lightgbm_loader import ensemble_thresholds
from malvalid.loaders.trees import LEAF, strict_lt_to_le
from malvalid.loaders.xgboost_loader import XGBoostLoader, _verify_and_set_base_margin, xgboost_tree_ensemble
from tests.unit.test_loaders import THREADS, assert_faithful, eval_rows, make_binary_data

xgb = pytest.importorskip("xgboost")


def _train(X: np.ndarray, y: np.ndarray, rounds: int = 25, **params: object):
    p: dict[str, object] = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.3, "nthread": THREADS}
    p.update(params)
    return xgb.train(p, xgb.DMatrix(X, y, missing=np.nan), rounds)


def _margins(booster, X: np.ndarray) -> np.ndarray:
    return booster.predict(xgb.DMatrix(X, missing=np.nan), output_margin=True).astype(np.float64)


@pytest.mark.parametrize("base_score", [None, 0.3, 0.85])
def test_booster_with_nan_is_exact(base_score: float | None) -> None:
    X, y = make_binary_data(700, 8, seed=1, nan_frac=0.06)
    params = {} if base_score is None else {"base_score": base_score}
    booster = _train(X, y, **params)
    te = assert_faithful(XGBoostLoader(), booster, X)
    assert te.model_kind == "xgboost" and te.n_trees == 25 and te.n_features == 8
    # base_score is stored in probability space; the ensemble works in margin space.
    stored = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
    b = float(np.float32(float(str(stored).strip("[]"))))
    assert te.base_score == pytest.approx(math.log(b / (1 - b)), abs=1e-12)
    assert te.meta["base_score_source"] == "logit(base_score)"
    if base_score is not None:
        assert b == pytest.approx(base_score, abs=1e-7)
    # raw margins agree with XGBoost's own output_margin=True (float32 accumulation in XGBoost)
    Z = eval_rows(te, X)
    np.testing.assert_allclose(te.predict_raw(Z), _margins(booster, Z), rtol=0, atol=2e-5)


def test_strict_less_than_ties_and_default_left() -> None:
    X, y = make_binary_data(600, 6, seed=2, nan_frac=0.1)
    booster = _train(X, y, 10)
    model = json.loads(bytes(booster.save_raw(raw_format="json")).decode())
    trees_json = model["learner"]["gradient_booster"]["model"]["trees"]
    te = XGBoostLoader().tree_ensemble(booster)
    for t_json, t in zip(trees_json, te.trees):
        left = np.asarray(t_json["left_children"])
        conds = np.asarray(t_json["split_conditions"], dtype=np.float32)[left >= 0]
        dl = np.asarray(t_json["default_left"], dtype=bool)[left >= 0]
        internal = t.children_left != LEAF
        # pre-order renumbering keeps the root first; compare as multisets per tree
        np.testing.assert_array_equal(np.sort(t.threshold[internal]), np.sort(strict_lt_to_le(conds)))
        assert int((t.children_default[internal] == t.children_left[internal]).sum()) == int(dl.sum())
    # Rows sitting exactly on every split value: XGBoost sends x == t right (x < t is False).
    Z = np.tile(X[:1], (len(conds), 1))
    feats = np.asarray(trees_json[-1]["split_indices"])[np.asarray(trees_json[-1]["left_children"]) >= 0]
    Z[np.arange(len(conds)), feats] = conds
    p = XGBoostLoader().predict_proba(booster, Z)
    np.testing.assert_allclose(te.predict_proba(Z), p, rtol=0, atol=1e-6)


def test_matches_trees_to_dataframe() -> None:
    pytest.importorskip("pandas")
    X, y = make_binary_data(500, 6, seed=3, nan_frac=0.05)
    booster = _train(X, y, 8)
    te = XGBoostLoader().tree_ensemble(booster)
    df = booster.trees_to_dataframe()
    for i, t in enumerate(te.trees):
        rows = df[(df["Tree"] == i) & (df["Feature"] != "Leaf")]
        internal = t.children_left != LEAF
        assert internal.sum() == len(rows)
        feat_idx = rows["Feature"].str.lstrip("f").astype(int).to_numpy()
        np.testing.assert_array_equal(np.sort(t.feature[internal]), np.sort(feat_idx))
        np.testing.assert_array_equal(
            np.sort(t.threshold[internal]), np.sort(strict_lt_to_le(rows["Split"].to_numpy(np.float32)))
        )
        leaves = df[(df["Tree"] == i) & (df["Feature"] == "Leaf")]["Gain"].to_numpy(np.float32)
        np.testing.assert_allclose(np.sort(t.value[~internal]), np.sort(leaves.astype(np.float64)), rtol=0, atol=0)
        missing_left = int((rows["Missing"] == rows["Yes"]).sum())
        assert int((t.children_default[internal] == t.children_left[internal]).sum()) == missing_left


@pytest.mark.parametrize("fmt", ["json", "ubj"])
def test_saved_model_round_trip(tmp_path: Path, fmt: str) -> None:
    X, y = make_binary_data(600, 7, seed=4, nan_frac=0.05)
    booster = _train(X, y, 15, base_score=0.4)
    path = tmp_path / f"detector.{fmt}"
    booster.save_model(str(path))
    loader = XGBoostLoader()
    assert loader.can_load(path)
    loaded = loader.load(path)
    p_ref = booster.predict(xgb.DMatrix(X, missing=np.nan))
    np.testing.assert_allclose(loader.predict_proba(loaded, X), p_ref, rtol=0, atol=1e-7)
    te = assert_faithful(loader, loaded, X)
    np.testing.assert_allclose(te.predict_raw(X), loader.tree_ensemble(booster).predict_raw(X), rtol=0, atol=0)


def test_classifier_with_early_stopping_uses_best_iteration() -> None:
    X, y = make_binary_data(900, 8, seed=5, nan_frac=0.03)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.5, n_jobs=THREADS, early_stopping_rounds=3)
    clf.fit(X[:600], y[:600], eval_set=[(X[600:], y[600:])], verbose=False)
    assert 0 < clf.best_iteration < 299
    loader = XGBoostLoader()
    np.testing.assert_allclose(loader.predict_proba(clf, X), clf.predict_proba(X)[:, 1], rtol=0, atol=1e-6)  # float64 sigmoid vs xgboost's float32
    te = assert_faithful(loader, clf, X)
    assert te.n_trees == clf.best_iteration + 1
    assert te.meta["n_trees_total"] > te.n_trees


@pytest.mark.parametrize(
    ("params", "transform"),
    [
        ({"objective": "binary:logitraw"}, "logistic"),
        ({"objective": "reg:logistic"}, "logistic"),
        ({"objective": "reg:squarederror"}, "identity"),
        ({"booster": "dart", "rate_drop": 0.3, "skip_drop": 0.0}, "logistic"),
        ({"num_parallel_tree": 3, "subsample": 0.8, "colsample_bynode": 0.8}, "logistic"),
    ],
)
def test_objective_and_booster_variants(params: dict[str, object], transform: str) -> None:
    X, y = make_binary_data(600, 6, seed=6, nan_frac=0.05)
    booster = _train(X, y.astype(float), 10, **params)
    te = assert_faithful(XGBoostLoader(), booster, X)
    assert te.output_transform == transform


@pytest.mark.filterwarnings("ignore:(?s).*are not used:UserWarning")  # gblinear + max_depth
def test_unsupported_models_are_rejected() -> None:
    X, y = make_binary_data(300, 5, seed=7)
    loader = XGBoostLoader()
    lin = _train(X, y, 3, booster="gblinear")
    with pytest.raises(UnsupportedTreeError, match="gblinear"):
        loader.tree_ensemble(lin)
    y3 = np.random.default_rng(0).integers(0, 3, X.shape[0])
    multi = _train(X, y3, 3, objective="multi:softprob", num_class=3)
    with pytest.raises(UnsupportedTreeError, match="multi"):
        loader.tree_ensemble(multi)
    with pytest.raises(ValueError, match="single malicious-class probability|binary"):
        loader.predict_proba(multi, X)
    with pytest.raises(UnsupportedTreeError, match="not fitted"):
        loader.tree_ensemble(xgb.XGBClassifier())


def test_verification_catches_a_misparsed_dump() -> None:
    X, y = make_binary_data(500, 6, seed=8)
    booster = _train(X, y, 10)
    te = xgboost_tree_ensemble(booster, verify=False)
    thresholds = ensemble_thresholds(te)
    _verify_and_set_base_margin(te, booster, [("base_score", te.base_score)], thresholds, None)  # intact: ok
    leaves = te.trees[3].children_left == LEAF
    te.trees[3].value[leaves] *= -1.0  # e.g. a mis-parsed sign / leaf layout
    with pytest.raises(UnsupportedTreeError, match="does not reproduce"):
        _verify_and_set_base_margin(te, booster, [("base_score", te.base_score)], thresholds, None)


# --------------------------------------------------------------------------------------------------
# float64 tail resolution: XGBoost's own probabilities are float32 and tie at 1.0 for margins > ~16.6
# --------------------------------------------------------------------------------------------------


def _extreme_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(3000, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def _saturating_booster() -> tuple[Any, np.ndarray, np.ndarray]:
    X, y = _extreme_data()
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 3, "eta": 1.0, "lambda": 0.0, "min_child_weight": 0.0,
         "nthread": THREADS},
        xgb.DMatrix(X, label=y), 40,
    )
    Xe = X.copy()
    Xe[:, :2] *= 6.0  # push far into both tails
    return booster, X, Xe


def test_saturated_margins_do_not_tie_booster() -> None:
    booster, X, Xe = _saturating_booster()
    dm = xgb.DMatrix(Xe, missing=np.nan)
    margin = booster.predict(dm, output_margin=True).astype(np.float64)
    native = booster.predict(dm).astype(np.float64)
    assert (margin > 17).sum() > 50 and np.unique(native[margin > 17]).size == 1  # the artifact is real
    p = XGBoostLoader().predict_proba(booster, Xe)
    assert p.dtype == np.float64
    hi = margin > 17
    assert np.unique(p[hi]).size > 20  # distinct margins stay distinct
    order = np.argsort(margin, kind="stable")
    assert np.all(np.diff(p[order]) >= 0)  # monotone in the margin
    lo = margin < -17
    assert lo.sum() > 50 and np.all(p[lo] > 0)
    ok = (np.abs(margin) < 10)
    np.testing.assert_allclose(p[ok], native[ok], rtol=0, atol=1e-6)


def test_saturated_margins_do_not_tie_sklearn_classifier() -> None:
    X, y = _extreme_data()
    clf = xgb.XGBClassifier(n_estimators=40, max_depth=3, learning_rate=1.0, reg_lambda=0.0,
                            min_child_weight=0.0, n_jobs=THREADS).fit(X, y)
    Xe = X.copy()
    Xe[:, :2] *= 6.0
    margin = clf.predict(Xe, output_margin=True).astype(np.float64)
    native = clf.predict_proba(Xe)[:, 1].astype(np.float64)
    p = XGBoostLoader().predict_proba(clf, Xe)
    hi = margin > 17
    assert hi.sum() > 50 and np.unique(native[hi]).size == 1 and np.unique(p[hi]).size > 20
    ok = np.abs(margin) < 10
    np.testing.assert_allclose(p[ok], native[ok], rtol=0, atol=1e-6)


@pytest.mark.parametrize("fmt,suffix", [("json", "json"), ("ubj", "ubj"), ("json", "model"), ("ubj", "bst")])
def test_saved_model_loads_from_non_ascii_path(tmp_path: Path, fmt: str, suffix: str) -> None:
    # XGBoost's own file open fails on non-ASCII Windows paths, so the loader always loads from bytes and
    # XGBoost detects JSON vs UBJSON from the content. The bytes come from save_raw, so the test does not
    # depend on XGBoost writing such a path either.
    X, y = make_binary_data(500, 6, seed=12, nan_frac=0.05)
    booster = _train(X, y, 10, base_score=0.35)
    d = tmp_path / "José_模型"
    d.mkdir()
    path = d / f"détecteur_模型.{suffix}"
    path.write_bytes(bytes(booster.save_raw(raw_format=fmt)))
    loader = XGBoostLoader()
    assert loader.can_load(path)
    loaded = loader.load(path)
    p_ref = booster.predict(xgb.DMatrix(X, missing=np.nan))
    np.testing.assert_allclose(loader.predict_proba(loaded, X), p_ref, rtol=0, atol=1e-7)
    np.testing.assert_allclose(loader.tree_ensemble(loaded).predict_raw(X), loader.tree_ensemble(booster).predict_raw(X),
                               rtol=0, atol=0)
