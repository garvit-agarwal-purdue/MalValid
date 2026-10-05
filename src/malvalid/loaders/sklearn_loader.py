"""scikit-learn loader (``model_kind: sklearn_gbdt``).

scikit-learn estimators only exist as pickles (joblib/pickle), so loading **always** requires
``--allow-pickle`` (:meth:`SklearnLoader.is_pickle` is True for every artifact). Inside the sandbox
the pickle guards stay off only when the operator allowed pickles.

Scoring works for any fitted binary scikit-learn classifier with ``predict_proba`` (the
malicious-class column is the one for label 1, i.e. the second of the sorted ``classes_``).
Tree access (normalized :class:`TreeEnsemble`) is supported for:

* ``GradientBoostingClassifier`` — raw margin = init + learning_rate * sum(tree.predict(X)). The
  init estimator must be ``"zero"`` or the default prior ``DummyClassifier``; its raw prediction is
  the loss link of the float32-eps-clipped class prior (what ``_init_raw_predictions`` computes).
  Leaves are pre-multiplied by the learning rate. ``loss="exponential"`` predicts
  ``expit(2 * raw)``; the factor 2 is folded into leaves and base. GBC rejects NaN inputs; the
  ensemble routes NaN by the trees' ``missing_go_to_left``.
* ``HistGradientBoostingClassifier`` — raw = ``_baseline_prediction`` + sum of the
  ``_predictors`` leaf values (already shrunk by the learning rate). Splits compare the raw value
  against the real-valued bin threshold ``num_threshold`` (``x <= t`` goes left); NaN goes left
  iff ``missing_go_to_left`` (sklearn sets it for every split, including features that had no
  missing values in training). Categorical splits are rejected.
* ``RandomForestClassifier`` / ``ExtraTreesClassifier`` / ``DecisionTreeClassifier`` — the
  averaged per-tree class-1 fraction (identity output, ``average_output=True``).

A ``Pipeline`` whose steps before the estimator are all ``"passthrough"`` is unwrapped; any real
preprocessing step makes tree access unavailable (thresholds would be in the transformed space).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from malvalid.core import UnsupportedTreeError
from malvalid.loaders.base import ModelLoader
from malvalid.loaders.lightgbm_loader import (
    as_float_matrix,
    build_tree,
    describe_type,
    load_pickled,
    pickle_claims,
)
from malvalid.loaders.trees import Tree, TreeEnsemble

log = logging.getLogger("malvalid.loaders")


def _is_sklearn_obj(obj: Any) -> bool:
    return type(obj).__module__.split(".")[0] == "sklearn"


def _unwrap(obj: Any) -> Any:
    """Final estimator of a Pipeline whose other steps are all passthrough; else ``obj``."""
    steps = getattr(obj, "steps", None)
    if steps and type(obj).__name__ == "Pipeline":
        pre = [s for _, s in steps[:-1] if s not in (None, "passthrough")]
        if pre:
            raise UnsupportedTreeError(
                "sklearn Pipeline with preprocessing steps cannot expose trees over the raw features"
            )
        return steps[-1][1]
    return obj


def _feature_names(est: Any) -> list[str] | None:
    names = getattr(est, "feature_names_in_", None)
    return [str(f) for f in names] if names is not None else None


def _require_binary(est: Any) -> None:
    classes = getattr(est, "classes_", None)
    if classes is None or len(classes) != 2:
        raise UnsupportedTreeError(
            f"only binary classifiers are supported (classes_={None if classes is None else list(classes)})"
        )


def _sk_tree(dt: Any, idx: int, *, scale: float = 1.0, class_fraction: bool = False) -> Tree:
    """sklearn ``tree_`` -> normalized Tree. Regression trees: leaf = scale * value;
    classification trees (``class_fraction``): leaf = scale * P(class 1 | leaf)."""
    t = dt.tree_
    left = np.asarray(t.children_left, dtype=np.int64)
    right = np.asarray(t.children_right, dtype=np.int64)
    value = np.asarray(t.value, dtype=np.float64)
    if class_fraction:
        if value.ndim != 3 or value.shape[1] != 1 or value.shape[2] != 2:
            raise UnsupportedTreeError(f"unexpected sklearn classification tree value shape {value.shape}")
        tot = value[:, 0, :].sum(axis=1)
        leaf_value = scale * np.divide(value[:, 0, 1], tot, out=np.zeros_like(tot), where=tot > 0)
    else:
        if value.ndim != 3 or value.shape[1:] != (1, 1):
            raise UnsupportedTreeError(f"unexpected sklearn regression tree value shape {value.shape}")
        leaf_value = scale * value[:, 0, 0]
    mgl = getattr(t, "missing_go_to_left", None)
    # sklearn < 1.3 has no missing-value support: NaN <= t is False -> right child.
    default_left = np.zeros(left.shape[0], dtype=bool) if mgl is None else np.asarray(mgl).astype(bool)
    w = np.asarray(t.weighted_n_node_samples, dtype=np.float64)
    imp = np.asarray(t.impurity, dtype=np.float64)
    internal = np.flatnonzero(left >= 0)
    gain = np.zeros(left.shape[0])
    gain[internal] = (
        w[internal] * imp[internal] - w[left[internal]] * imp[left[internal]] - w[right[internal]] * imp[right[internal]]
    )
    return build_tree(
        left=left,
        right=right,
        feature=np.asarray(t.feature, dtype=np.int64),
        threshold=np.asarray(t.threshold, dtype=np.float64),
        default_left=default_left,
        leaf_value=leaf_value,
        cover=w,
        gain=gain,
        what=f"sklearn tree {idx}",
    )


def _gbc_ensemble(est: Any) -> TreeEnsemble:
    _require_binary(est)
    loss_name = type(getattr(est, "_loss", None)).__name__
    if loss_name == "ExponentialLoss" or getattr(est, "loss", None) == "exponential":
        factor = 2.0  # predict_proba = expit(2 * raw); raw init = 0.5 * logit(prior)
    elif loss_name in ("HalfBinomialLoss", "BinomialDeviance") or getattr(est, "loss", None) == "log_loss":
        factor = 1.0
    else:
        raise UnsupportedTreeError(f"GradientBoostingClassifier loss {loss_name!r} is not supported")
    init = est.init_
    if isinstance(init, str) and init == "zero":
        base_margin = 0.0
    else:
        from sklearn.dummy import DummyClassifier

        if not isinstance(init, DummyClassifier) or getattr(init, "strategy", "prior") != "prior":
            raise UnsupportedTreeError(
                f"GradientBoostingClassifier with a custom init estimator ({describe_type(init)}) is not supported"
            )
        p = float(np.asarray(init.class_prior_, dtype=np.float64)[1])
        eps = float(np.finfo(np.float32).eps)
        p = min(max(p, eps), 1.0 - eps)
        base_margin = math.log(p / (1.0 - p))  # == factor * link(p) for both supported losses
    lr = float(est.learning_rate)
    stages = np.asarray(est.estimators_)
    if stages.ndim != 2 or stages.shape[1] != 1:
        raise UnsupportedTreeError(f"unexpected GradientBoostingClassifier estimators_ shape {stages.shape}")
    trees = [_sk_tree(dt, i, scale=lr * factor) for i, dt in enumerate(stages[:, 0])]
    return TreeEnsemble(
        trees=trees,
        n_features=int(est.n_features_in_),
        base_score=float(base_margin),
        output_transform="logistic",
        model_kind="sklearn_gbdt",
        feature_names=_feature_names(est),
        meta={
            "loader": "sklearn_gbdt",
            "estimator": "GradientBoostingClassifier",
            "loss": loss_name,
            "learning_rate": lr,
            "init": "zero" if isinstance(init, str) else "prior",
            "source": "sklearn estimators_[i].tree_",
            "cover": "weighted_n_node_samples",
        },
    )


def _hgb_ensemble(est: Any) -> TreeEnsemble:
    _require_binary(est)
    if int(getattr(est, "n_trees_per_iteration_", 1)) != 1:
        raise UnsupportedTreeError("only binary HistGradientBoostingClassifier models are supported")
    is_cat = getattr(est, "is_categorical_", None)
    if getattr(est, "_preprocessor", None) is not None or (is_cat is not None and np.any(is_cat)):
        raise UnsupportedTreeError("HistGradientBoosting models with categorical features are not supported")
    base = np.ravel(np.asarray(est._baseline_prediction, dtype=np.float64))
    if base.shape != (1,):
        raise UnsupportedTreeError(f"unexpected HistGradientBoosting baseline shape {base.shape}")
    trees: list[Tree] = []
    for i, preds in enumerate(est._predictors):
        if len(preds) != 1:
            raise UnsupportedTreeError("only single-output HistGradientBoosting predictors are supported")
        nodes = preds[0].nodes
        if np.any(nodes["is_categorical"]):
            raise UnsupportedTreeError("HistGradientBoosting categorical splits are not supported")
        is_leaf = nodes["is_leaf"].astype(bool)
        trees.append(
            build_tree(
                left=np.where(is_leaf, -1, nodes["left"].astype(np.int64)),
                right=np.where(is_leaf, -1, nodes["right"].astype(np.int64)),
                feature=nodes["feature_idx"].astype(np.int64),
                threshold=nodes["num_threshold"].astype(np.float64),
                default_left=nodes["missing_go_to_left"].astype(bool),
                leaf_value=nodes["value"].astype(np.float64),
                cover=nodes["count"].astype(np.float64),
                gain=np.where(is_leaf, 0.0, nodes["gain"].astype(np.float64)),
                what=f"HistGradientBoosting tree {i}",
            )
        )
    return TreeEnsemble(
        trees=trees,
        n_features=int(est.n_features_in_),
        base_score=float(base[0]),
        output_transform="logistic",
        model_kind="sklearn_gbdt",
        feature_names=_feature_names(est),
        meta={
            "loader": "sklearn_gbdt",
            "estimator": "HistGradientBoostingClassifier",
            "learning_rate": float(est.learning_rate),
            "n_iter": int(getattr(est, "n_iter_", len(trees))),
            "source": "sklearn _predictors[i][0].nodes (num_threshold, missing_go_to_left)",
            "cover": "count",
        },
    )


def _forest_ensemble(est: Any) -> TreeEnsemble:
    from sklearn.tree import DecisionTreeClassifier

    _require_binary(est)
    members = [est] if isinstance(est, DecisionTreeClassifier) else list(est.estimators_)
    for m in members:
        if int(getattr(m, "n_outputs_", 1)) != 1:
            raise UnsupportedTreeError("multi-output tree classifiers are not supported")
    trees = [_sk_tree(m, i, class_fraction=True) for i, m in enumerate(members)]
    return TreeEnsemble(
        trees=trees,
        n_features=int(est.n_features_in_),
        base_score=0.0,
        output_transform="identity",
        average_output=True,
        model_kind="sklearn_gbdt",
        feature_names=_feature_names(est),
        meta={
            "loader": "sklearn_gbdt",
            "estimator": type(est).__name__,
            "source": "sklearn tree_ class fractions (averaged)",
            "cover": "weighted_n_node_samples",
        },
    )


class SklearnLoader(ModelLoader):
    kind: ClassVar[str] = "sklearn_gbdt"
    extensions: ClassVar[tuple[str, ...]] = (".pkl", ".pickle", ".joblib", ".jbl", ".sav")
    safe_extensions: ClassVar[tuple[str, ...]] = ()  # sklearn estimators are always pickles
    description: ClassVar[str] = (
        "scikit-learn binary classifier ((Hist)GradientBoosting, RandomForest/ExtraTrees) via "
        "joblib/pickle — always requires --allow-pickle"
    )

    def is_pickle(self, path: Path) -> bool:
        return True  # every sklearn artifact is unpickled, whatever its extension

    def can_load(self, path: Path) -> bool:
        p = Path(path)
        return p.suffix.lower() in self.extensions and pickle_claims(p, ("sklearn",))

    def load(self, path: Path, *, allow_pickle: bool = False) -> Any:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"scikit-learn model file not found: {p}")
        self.check_pickle_policy(p, allow_pickle)  # before anything is deserialized
        obj = load_pickled(p, allow_pickle=allow_pickle, what="scikit-learn")
        if not self.supports_native(obj):
            raise TypeError(
                f"{p} contains a {describe_type(obj)}, not a fitted scikit-learn classifier with "
                "predict_proba; set model_kind to match the model or load it in the adapter"
            )
        return obj

    def supports_native(self, obj: Any) -> bool:
        return _is_sklearn_obj(obj) and callable(getattr(obj, "predict_proba", None))

    def predict_proba(self, native: Any, X: np.ndarray) -> np.ndarray:
        X = as_float_matrix(X)
        classes = getattr(native, "classes_", None)
        if classes is None or len(classes) != 2:
            raise ValueError(
                f"{type(native).__name__} is not a binary classifier (classes_={classes}); malvalid "
                "evaluates binary detectors (label 1 = malicious)"
            )
        p = np.asarray(native.predict_proba(X), dtype=np.float64)
        if p.shape != (X.shape[0], 2):
            raise ValueError(f"{type(native).__name__}.predict_proba returned shape {p.shape}")
        return p[:, 1]

    def tree_ensemble(self, native: Any) -> TreeEnsemble:
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            GradientBoostingClassifier,
            HistGradientBoostingClassifier,
            RandomForestClassifier,
        )
        from sklearn.tree import DecisionTreeClassifier

        est = _unwrap(native)
        if isinstance(est, HistGradientBoostingClassifier):
            return _hgb_ensemble(est)
        if isinstance(est, GradientBoostingClassifier):
            return _gbc_ensemble(est)
        if isinstance(est, (RandomForestClassifier, ExtraTreesClassifier, DecisionTreeClassifier)):
            return _forest_ensemble(est)
        raise UnsupportedTreeError(
            f"{type(est).__name__} is not a tree model malvalid can normalize "
            "(supported: (Hist)GradientBoostingClassifier, RandomForest/ExtraTrees/DecisionTreeClassifier)"
        )

    def library_version(self) -> str | None:
        try:
            import sklearn

            return str(sklearn.__version__)
        except ImportError:  # pragma: no cover
            return None
