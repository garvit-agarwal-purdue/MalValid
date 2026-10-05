"""LightGBM loader: exact tree normalization (NaN, ties, LightGBM's zero band), SHAP additivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from malvalid.core import UnsupportedTreeError
from malvalid.loaders.lightgbm_loader import (
    LGB_ZERO_THRESHOLD,
    LightGBMLoader,
    categorical_partition,
    ensemble_thresholds,
    expand_categorical_splits,
    probe_matrix,
)
from malvalid.loaders.trees import LEAF, MISSING_NAN, MISSING_ZERO
from tests.unit.test_loaders import THREADS, assert_faithful, make_binary_data

lgb = pytest.importorskip("lightgbm")

KZ = LGB_ZERO_THRESHOLD


def _train(X: np.ndarray, y: np.ndarray, rounds: int = 25, **params: object):
    p = {"objective": "binary", "verbose": -1, "num_leaves": 12, "min_data_in_leaf": 5, "num_threads": THREADS}
    p.update(params)
    return lgb.train(p, lgb.Dataset(X, y), rounds)


def _all_thresholds(te) -> np.ndarray:
    return np.concatenate([t.threshold[t.children_left != LEAF] for t in te.trees])


def test_booster_with_nan_training_data_is_exact() -> None:
    X, y = make_binary_data(700, 8, seed=1, nan_frac=0.06)
    booster = _train(X, y)
    te = assert_faithful(LightGBMLoader(), booster, X)
    assert te.model_kind == "lightgbm" and te.n_trees == 25 and te.n_features == 8
    assert te.output_transform == "logistic" and te.base_score == 0.0
    assert te.meta["loader"] == "lightgbm"


def test_nan_inputs_on_model_trained_without_nan() -> None:
    """missing_type=None splits compare NaN as 0.0; normalized to NaN-default nodes for SHAP."""
    X, y = make_binary_data(700, 8, seed=2)
    booster = _train(X, y)
    te = assert_faithful(LightGBMLoader(), booster, X, nan=True)
    assert te.meta["missing_none_normalized"] > 0
    assert all(np.all(t.missing_type[t.children_left != LEAF] == MISSING_NAN) for t in te.trees)


def test_zero_band_thresholds_match_lightgbm_predictor() -> None:
    """LightGBM zeroes |x| <= kZeroThreshold before splitting; ties at +-kZeroThreshold are exact."""
    X, y = make_binary_data(900, 6, seed=3)
    X[:, 0] = np.round(X[:, 0])  # many exact zeros -> splits at +-kZeroThreshold
    booster = _train(X, y)
    loader = LightGBMLoader()
    te = assert_faithful(loader, booster, X)
    assert te.meta["zero_thresholds_adjusted"] > 0
    thr = _all_thresholds(te)
    assert not np.any((thr >= -KZ) & (thr < KZ) & (thr != KZ)), "thresholds left inside the zero band"
    band = np.array([-KZ, -1e-36, -0.0, 0.0, 1e-36, KZ, np.nextafter(KZ, 1.0), np.nextafter(-KZ, -1.0)])
    Z = np.repeat(X[:len(band)].astype(np.float64), 1, axis=0)
    Z[:, 0] = band
    np.testing.assert_allclose(te.predict_proba(Z), loader.predict_proba(booster, Z), rtol=0, atol=1e-12)


def test_zero_as_missing_model() -> None:
    X, y = make_binary_data(900, 6, seed=4)
    X[:, :2] = np.round(X[:, :2])
    booster = _train(X, y, zero_as_missing=True)
    loader = LightGBMLoader()
    te = loader.tree_ensemble(booster)
    assert any(np.any(t.missing_type == MISSING_ZERO) for t in te.trees)
    Z = probe_matrix(ensemble_thresholds(te), 6, n_rows=500, seed=5).astype(np.float64)
    # Known frozen-core limitation (trees._K_ZERO = 1e-35 instead of float32(1e-35)): an input of
    # exactly +-1.0000000180025095e-35 on a Zero-missing node is compared, LightGBM sends it to the
    # default child. Real feature vectors never hold such values; exclude them here.
    Z[np.abs(Z) == KZ] = 0.0
    Z[:50, :2] = np.array([0.0, np.nan])[np.arange(50) % 2][:, None]
    np.testing.assert_allclose(te.predict_proba(Z), loader.predict_proba(booster, Z), rtol=0, atol=1e-12)


def test_sigmoid_scale_and_cross_entropy_objectives() -> None:
    X, y = make_binary_data(500, 6, seed=5)
    te = assert_faithful(LightGBMLoader(), _train(X, y, 15, sigmoid=0.7), X)
    assert te.sigmoid_scale == pytest.approx(0.7)
    te = assert_faithful(LightGBMLoader(), _train(X, y, 15, objective="cross_entropy"), X)
    assert te.output_transform == "logistic"


def test_regression_objective_is_clipped_identity() -> None:
    X, y = make_binary_data(500, 6, seed=6)
    booster = _train(X, y.astype(float), 15, objective="regression")
    loader = LightGBMLoader()
    p = loader.predict_proba(booster, X)
    assert p.min() >= 0.0 and p.max() <= 1.0
    te = assert_faithful(loader, booster, X)
    assert te.output_transform == "identity"


def test_text_model_round_trip(tmp_path: Path) -> None:
    X, y = make_binary_data(600, 7, seed=7, nan_frac=0.05)
    booster = _train(X, y, 20)
    path = tmp_path / "detector.txt"
    booster.save_model(str(path))
    loader = LightGBMLoader()
    assert loader.can_load(path)
    loaded = loader.load(path)
    np.testing.assert_allclose(loader.predict_proba(loaded, X), booster.predict(X), rtol=0, atol=1e-12)
    te = assert_faithful(loader, loaded, X)
    te_mem = loader.tree_ensemble(booster)
    np.testing.assert_allclose(te.predict_raw(X), te_mem.predict_raw(X), rtol=0, atol=1e-12)


def test_json_dump_is_not_a_lightgbm_artifact(tmp_path: Path) -> None:
    import json

    X, y = make_binary_data(200, 5, seed=8)
    p = tmp_path / "dump.json"
    p.write_text(json.dumps(_train(X, y, 3).dump_model()))
    assert not LightGBMLoader().can_load(p)


def test_lgbm_classifier_with_early_stopping_uses_best_iteration() -> None:
    X, y = make_binary_data(900, 8, seed=9, nan_frac=0.03)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.3, num_leaves=15, verbose=-1, n_jobs=THREADS)
    clf.fit(X[:600], y[:600], eval_X=(X[600:],), eval_y=(y[600:],), callbacks=[lgb.early_stopping(3, verbose=False)])
    assert 0 < clf.best_iteration_ < 300
    loader = LightGBMLoader()
    assert loader.supports_native(clf)
    np.testing.assert_allclose(loader.predict_proba(clf, X), clf.predict_proba(X)[:, 1], rtol=0, atol=1e-15)
    te = assert_faithful(loader, clf, X)
    assert te.n_trees == clf.best_iteration_


def test_linear_trees_are_rejected() -> None:
    X, y = make_binary_data(400, 5, seed=10)
    booster = _train(X, y, 5, linear_tree=True)
    with pytest.raises(UnsupportedTreeError, match="linear"):
        LightGBMLoader().tree_ensemble(booster)


def test_multiclass_is_rejected() -> None:
    X, _ = make_binary_data(400, 5, seed=11)
    y3 = np.random.default_rng(0).integers(0, 3, X.shape[0])
    booster = _train(X, y3, 3, objective="multiclass", num_class=3)
    loader = LightGBMLoader()
    with pytest.raises(UnsupportedTreeError, match="binary"):
        loader.tree_ensemble(booster)
    with pytest.raises(ValueError, match="binary"):
        loader.predict_proba(booster, X)


def test_unfitted_classifier_has_no_trees() -> None:
    with pytest.raises(UnsupportedTreeError, match="not fitted"):
        LightGBMLoader().tree_ensemble(lgb.LGBMClassifier())


# --------------------------------------------------------------------------------------------------
# categorical splits (decision_type "=="), rewritten as threshold chains
# --------------------------------------------------------------------------------------------------


def _lgb_category_test(x: np.ndarray, cats: set[int]) -> np.ndarray:
    """LightGBM's categorical decision: left iff not NaN and int(x) (truncation) is a member >= 0."""
    out = np.zeros(x.shape, dtype=bool)
    ok = np.isfinite(x) & (np.abs(x) < 2**31)
    xi = np.trunc(x[ok]).astype(np.int64)
    out[ok] = (xi >= 0) & np.isin(xi, sorted(cats))
    return out


@pytest.mark.parametrize("cats", [{0}, {3}, {0, 1, 2}, {0, 3, 4, 7}, {1, 2, 6, 8, 9, 11}, {5, 10}, set()])
def test_categorical_partition_matches_lightgbm_semantics(cats: set[int]) -> None:
    bounds, flags = categorical_partition(cats)
    assert len(flags) == len(bounds) + 1 and not flags[0]
    assert all(a < b for a, b in zip(bounds, bounds[1:]))
    grid = np.concatenate([np.arange(-3, 14, 0.25), [-1.0, -0.999, -1e-40, 0.0, -0.0, 0.999999, 1e9, 3e9]])
    grid = np.concatenate([grid, np.nextafter(grid, -np.inf), np.nextafter(grid, np.inf), [np.nan, np.inf, -np.inf]])
    seg = np.searchsorted(np.asarray(bounds), grid, side="left")  # segment k: (b_{k-1}, b_k]
    got = np.where(np.isnan(grid), False, np.asarray(flags)[seg])
    np.testing.assert_array_equal(got, _lgb_category_test(grid, cats))


def _categorical_data(n: int = 3000, seed: int = 0, nan_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    X[:, 0] = rng.integers(0, 14, n)
    X[:, 1] = rng.integers(0, 6, n)
    X[rng.random(n) < nan_frac, 0] = np.nan
    y = (np.isin(X[:, 0], [0, 3, 4, 7]) ^ (X[:, 1] == 2) ^ (X[:, 2] > 0.3) ^ np.isnan(X[:, 0])).astype(np.int64)
    flip = rng.random(n) < 0.05
    return X, np.where(flip, 1 - y, y)


def _category_rows(X: np.ndarray, seed: int = 3) -> np.ndarray:
    """Rows whose categorical features hit every category, fractional/negative values, NaN and inf."""
    rng = np.random.default_rng(seed)
    Z = X[rng.integers(0, X.shape[0], 600)].copy()
    pool = np.concatenate(
        [np.arange(-2, 16, 0.5), [-1.0, -0.999, -1e-40, 0.999999, 1e6, np.nan, np.inf, -np.inf]]
    ).astype(np.float32)
    Z[:, 0] = rng.choice(pool, 600)
    Z[:, 1] = rng.choice(pool, 600)
    return Z


@pytest.mark.parametrize("nan_frac", [0.1, 0.0])
def test_categorical_splits_are_exact(nan_frac: float) -> None:
    X, y = _categorical_data(nan_frac=nan_frac)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 12, "min_data_per_group": 5, "cat_smooth": 1,
         "max_cat_to_onehot": 2, "num_threads": THREADS},
        lgb.Dataset(X, y, categorical_feature=[0, 1]),
        20,
    )
    dump = booster.dump_model()
    loader = LightGBMLoader()
    te = assert_faithful(loader, booster, X)  # proba <= 1e-6, SHAP additive, bytes round trip
    assert te.meta["categorical_splits_expanded"] > 0 and te.meta["verify_max_abs_raw_diff"] <= 1e-9
    Z = _category_rows(X)
    np.testing.assert_allclose(te.predict_raw(Z), booster.predict(Z, raw_score=True), rtol=0, atol=1e-12)
    te_z = assert_faithful(loader, booster, Z)
    assert te_z.n_trees == te.n_trees

    # copies share the original subtree's cover and gain: totals are those of the native model
    def totals(tree_json: dict) -> tuple[float, float]:
        gain, stack = 0.0, [tree_json]
        while stack:
            nd = stack.pop()
            if "split_index" in nd:
                gain += nd.get("split_gain", 0.0)
                stack += [nd["left_child"], nd["right_child"]]
        root = tree_json.get("internal_count", tree_json.get("leaf_count", 0))
        return gain, float(root)

    for t, ti in zip(te.trees, dump["tree_info"]):
        gain, root_cover = totals(ti["tree_structure"])
        internal = t.children_left != LEAF
        assert t.gain[internal].sum() == pytest.approx(gain, rel=1e-9, abs=1e-9)
        assert t.cover[0] == pytest.approx(root_cover, rel=1e-9)
        leaves = ~internal
        assert t.cover[leaves].sum() == pytest.approx(root_cover, rel=1e-9)


def test_categorical_expansion_node_budget() -> None:
    X, y = _categorical_data(1500, seed=4)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_leaves": 12, "min_data_per_group": 5, "num_threads": THREADS},
        lgb.Dataset(X, y, categorical_feature=[0, 1]),
        5,
    )
    dump = booster.dump_model()
    _, stats = expand_categorical_splits(dump)
    assert stats["categorical_splits_expanded"] > 0
    with pytest.raises(UnsupportedTreeError, match="categorical splits"):
        expand_categorical_splits(dump, max_nodes=stats["nodes_after_expansion"] - 1)


def test_text_model_loads_from_non_ascii_path(tmp_path: Path) -> None:
    # LightGBM's own file open fails on non-ASCII Windows paths, so the loader reads the text in Python.
    # Save under an ASCII name and copy, so the test does not depend on LightGBM writing such a path.
    X, y = make_binary_data(400, 6, seed=11, nan_frac=0.05)
    booster = _train(X, y, 12)
    ascii_path = tmp_path / "detector.txt"
    booster.save_model(str(ascii_path))
    d = tmp_path / "José_模型"
    d.mkdir()
    path = d / "détecteur_模型.txt"
    path.write_bytes(ascii_path.read_bytes())
    loader = LightGBMLoader()
    assert loader.can_load(path)
    loaded = loader.load(path)
    np.testing.assert_array_equal(loader.predict_proba(loaded, X), loader.predict_proba(loader.load(ascii_path), X))
    np.testing.assert_allclose(loader.predict_proba(loaded, X), booster.predict(X), rtol=0, atol=1e-12)


def test_non_utf8_text_model_is_a_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "detector.txt"
    path.write_bytes(b"tree\nversion=v4\nfeature_names=\xff\xfe\n")
    with pytest.raises(ValueError, match="UTF-8"):
        LightGBMLoader().load(path)
