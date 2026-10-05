"""scikit-learn loader: (Hist)GradientBoostingClassifier trees incl. init/baseline, learning rate,
HGB bin thresholds and missing_go_to_left; forests; pickle-only artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from malvalid.core import PickleRefused, UnsupportedTreeError
from malvalid.loaders.sklearn_loader import SklearnLoader
from tests.unit.test_loaders import THREADS, assert_faithful, make_binary_data

sk_ensemble = pytest.importorskip("sklearn.ensemble")
GradientBoostingClassifier = sk_ensemble.GradientBoostingClassifier
HistGradientBoostingClassifier = sk_ensemble.HistGradientBoostingClassifier


@pytest.mark.parametrize(
    "kw",
    [
        {"loss": "log_loss"},
        {"loss": "log_loss", "learning_rate": 0.37, "subsample": 0.8},
        {"loss": "exponential"},
        {"loss": "log_loss", "init": "zero"},
        {"loss": "exponential", "init": "zero", "max_depth": 4},
    ],
    ids=["log_loss", "lr_subsample", "exponential", "init_zero", "exp_init_zero"],
)
def test_gradient_boosting_classifier_is_exact(kw: dict[str, object]) -> None:
    X, y = make_binary_data(600, 8, seed=1)
    y[:40] = 1  # skew the prior so the init raw prediction is not ~0
    est = GradientBoostingClassifier(n_estimators=20, max_depth=kw.pop("max_depth", 3), random_state=0, **kw).fit(X, y)
    # GBC rejects NaN inputs, so the native comparison runs without NaN
    te = assert_faithful(SklearnLoader(), est, X, nan=False)
    assert te.output_transform == "logistic" and te.n_trees == 20
    raw_native = est.decision_function(X[:100].astype(np.float64))
    factor = 2.0 if est.loss == "exponential" else 1.0
    np.testing.assert_allclose(te.predict_raw(X[:100]), factor * raw_native, rtol=0, atol=1e-10)
    if kw.get("init") == "zero":
        assert te.base_score == 0.0


def test_gbc_with_sample_weights_uses_weighted_prior() -> None:
    X, y = make_binary_data(500, 6, seed=2)
    w = np.where(y == 1, 3.0, 1.0)
    est = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, y, sample_weight=w)
    te = assert_faithful(SklearnLoader(), est, X, nan=False)
    p = np.average(y, weights=w)
    assert te.base_score == pytest.approx(np.log(p / (1 - p)), abs=1e-12)


def test_gbc_custom_init_estimator_is_unsupported() -> None:
    from sklearn.linear_model import LogisticRegression

    X, y = make_binary_data(300, 5, seed=3)
    est = GradientBoostingClassifier(n_estimators=5, init=LogisticRegression()).fit(X, y)
    loader = SklearnLoader()
    assert loader.predict_proba(est, X).shape == (300,)  # scoring still works
    with pytest.raises(UnsupportedTreeError, match="init"):
        loader.tree_ensemble(est)


@pytest.mark.parametrize("nan_frac", [0.0, 0.08])
def test_hist_gradient_boosting_is_exact(nan_frac: float) -> None:
    """HGB compares raw values against real-valued bin thresholds; NaN follows missing_go_to_left
    (set for every split, also for features that had no NaN in training)."""
    X, y = make_binary_data(800, 8, seed=4, nan_frac=nan_frac)
    est = HistGradientBoostingClassifier(max_iter=30, learning_rate=0.2, max_leaf_nodes=15, random_state=0).fit(X, y)
    te = assert_faithful(SklearnLoader(), est, X, nan=True)
    assert te.n_trees == est.n_iter_
    np.testing.assert_allclose(te.predict_raw(X), est.decision_function(X), rtol=0, atol=1e-10)
    assert te.meta["estimator"] == "HistGradientBoostingClassifier"


def test_hist_gradient_boosting_with_early_stopping() -> None:
    X, y = make_binary_data(1500, 8, seed=5, nan_frac=0.02)
    est = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.5, early_stopping=True, n_iter_no_change=3, random_state=0
    ).fit(X, y)
    assert est.n_iter_ < 500
    te = assert_faithful(SklearnLoader(), est, X)
    assert te.n_trees == est.n_iter_


def test_hist_gradient_boosting_categorical_is_unsupported() -> None:
    X, y = make_binary_data(400, 5, seed=6)
    X[:, 1] = np.abs(np.round(X[:, 1] * 2))
    est = HistGradientBoostingClassifier(max_iter=5, categorical_features=[1]).fit(X, y)
    with pytest.raises(UnsupportedTreeError, match="categorical"):
        SklearnLoader().tree_ensemble(est)


@pytest.mark.parametrize("name", ["RandomForestClassifier", "ExtraTreesClassifier"])
def test_forests_are_exact(name: str) -> None:
    X, y = make_binary_data(500, 6, seed=7)
    est = getattr(sk_ensemble, name)(n_estimators=12, max_depth=5, random_state=0, n_jobs=THREADS).fit(X, y)
    te = assert_faithful(SklearnLoader(), est, X, nan=False)
    assert te.output_transform == "identity" and te.average_output


def test_forest_with_missing_values() -> None:
    X, y = make_binary_data(500, 6, seed=8, nan_frac=0.08)
    est = sk_ensemble.RandomForestClassifier(n_estimators=8, max_depth=6, random_state=0, n_jobs=THREADS).fit(X, y)
    assert_faithful(SklearnLoader(), est, X, nan=True)


def test_passthrough_pipeline_is_unwrapped_and_preprocessing_is_not() -> None:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = make_binary_data(400, 6, seed=9)
    pipe = Pipeline([("noop", "passthrough"), ("gbc", GradientBoostingClassifier(n_estimators=5))]).fit(X, y)
    loader = SklearnLoader()
    assert loader.supports_native(pipe)
    assert_faithful(loader, pipe, X, nan=False)
    scaled = Pipeline([("sc", StandardScaler()), ("gbc", GradientBoostingClassifier(n_estimators=5))]).fit(X, y)
    assert loader.predict_proba(scaled, X).shape == (400,)
    with pytest.raises(UnsupportedTreeError, match="preprocessing"):
        loader.tree_ensemble(scaled)


def test_non_tree_and_multiclass_models() -> None:
    from sklearn.linear_model import LogisticRegression

    X, y = make_binary_data(300, 5, seed=10)
    loader = SklearnLoader()
    lr = LogisticRegression().fit(X, y)
    assert loader.supports_native(lr)
    np.testing.assert_allclose(loader.predict_proba(lr, X), lr.predict_proba(X)[:, 1])
    with pytest.raises(UnsupportedTreeError, match="not a tree model"):
        loader.tree_ensemble(lr)
    y3 = np.random.default_rng(0).integers(0, 3, X.shape[0])
    multi = GradientBoostingClassifier(n_estimators=3).fit(X, y3)
    with pytest.raises(ValueError, match="binary"):
        loader.predict_proba(multi, X)
    with pytest.raises(UnsupportedTreeError, match="binary"):
        loader.tree_ensemble(multi)


def test_string_class_labels_use_second_sorted_class() -> None:
    X, y = make_binary_data(300, 5, seed=11)
    labels = np.where(y == 1, "malicious", "benign")
    est = GradientBoostingClassifier(n_estimators=5).fit(X, labels)
    assert list(est.classes_) == ["benign", "malicious"]
    np.testing.assert_allclose(SklearnLoader().predict_proba(est, X), est.predict_proba(X)[:, 1])


def test_joblib_artifact_requires_allow_pickle(tmp_path: Path) -> None:
    import joblib

    X, y = make_binary_data(400, 6, seed=12, nan_frac=0.05)
    est = HistGradientBoostingClassifier(max_iter=10, random_state=0).fit(X, y)
    path = tmp_path / "detector.joblib"
    joblib.dump(est, path, compress=3)  # zlib container: detected by content, not just extension
    loader = SklearnLoader()
    assert loader.can_load(path)
    with pytest.raises(PickleRefused):
        loader.load(path)
    loaded = loader.load(path, allow_pickle=True)
    np.testing.assert_array_equal(loader.predict_proba(loaded, X), est.predict_proba(X)[:, 1])
    assert_faithful(loader, loaded, X)
