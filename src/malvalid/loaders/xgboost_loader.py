"""XGBoost loader (``model_kind: xgboost``).

Artifacts:

* ``.json`` / ``.ubj`` (and ``.model`` / ``.bst`` / ``.xgb``, which XGBoost >= 2 writes as UBJSON) —
  XGBoost's own model formats, safe and non-executable, loaded with ``Booster.load_model``.
* pickled ``xgboost.Booster`` / ``XGBClassifier`` (``.pkl``/``.joblib``/...) — only with
  ``--allow-pickle``.

Tree access parses the model JSON (``Booster.save_raw("json")``, the same content as
``trees_to_dataframe()`` but with every float stored exactly):

* XGBoost splits go left iff ``x < split_condition`` in float32. The normalized ensemble uses
  ``x <= threshold``, so thresholds become ``nextafter(float32(t), -inf)``
  (:func:`~malvalid.loaders.trees.strict_lt_to_le`) — exact for float32 feature vectors, which is
  what XGBoost itself compares (it converts inputs to float32).
* NaN follows ``default_left``.
* ``learner_model_param.base_score`` is stored in *probability* space for logistic objectives
  (XGBoost 3.x writes a one-element vector string such as ``"[3.44E-1]"``, possibly estimated from
  the training data) and becomes ``logit(base_score)`` in margin space. The conversion is always
  cross-checked against ``predict(output_margin=True)`` on probe rows built from the model's own
  split values (exact ties, both float32 neighbours, NaN); a dump that does not reproduce the
  model's margins raises :class:`UnsupportedTreeError` instead of returning wrong trees.
* ``binary:logistic`` / ``reg:logistic`` -> logistic; ``binary:logitraw`` -> logistic (the loader's
  probability is ``sigmoid(margin)``); squared-error style regression -> identity, clipped to [0, 1].
* DART: per-tree ``weight_drop`` scales the leaves. A fitted ``XGBClassifier`` that used early
  stopping predicts with trees up to ``best_iteration`` only, and so does its ensemble; a plain
  ``Booster`` always uses all trees (as ``Booster.predict`` does).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from malvalid.core import UnsupportedTreeError
from malvalid.loaders.base import ModelLoader
from malvalid.loaders.lightgbm_loader import (
    LIGHTGBM_TEXT_MAGIC,
    as_float_matrix,
    build_tree,
    describe_type,
    env_threads,
    load_pickled,
    pickle_claims,
    positive_column,
    probe_matrix,
    read_head,
)
from malvalid.loaders.trees import Tree, TreeEnsemble, strict_lt_to_le

log = logging.getLogger("malvalid.loaders")

LOGISTIC_OBJECTIVES = ("binary:logistic", "reg:logistic")
LOGITRAW_OBJECTIVES = ("binary:logitraw",)
HARD_LABEL_OBJECTIVES = ("binary:hinge",)
IDENTITY_OBJECTIVES = (
    "reg:squarederror",
    "reg:linear",
    "reg:pseudohubererror",
    "reg:absoluteerror",
    "reg:squaredlogerror",
)


def _is_xgb_obj(obj: Any) -> bool:
    return type(obj).__module__.split(".")[0] == "xgboost"


def _booster_of(obj: Any) -> Any:
    if hasattr(obj, "get_booster"):
        try:
            return obj.get_booster()
        except Exception as e:  # noqa: BLE001 - NotFittedError
            raise UnsupportedTreeError(f"XGBoost estimator is not fitted: {e}") from e
    return obj


def booster_objective(booster: Any) -> str:
    return str(json.loads(booster.save_config())["learner"]["objective"]["name"])


def _parse_scalar_param(v: Any, what: str) -> float:
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        parts = [x for x in s[1:-1].split(",") if x.strip()]
        if len(parts) != 1:
            raise UnsupportedTreeError(f"multi-target XGBoost {what} {s!r} is not supported")
        s = parts[0]
    return float(s)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable float64 logistic function (no overflow warnings, exact tails)."""
    from scipy.special import expit

    return expit(np.asarray(x, dtype=np.float64))


def _dmatrix(booster: Any, X: np.ndarray) -> Any:
    import xgboost as xgb

    kw: dict[str, Any] = {"missing": np.nan}
    if getattr(booster, "feature_names", None):
        kw["feature_names"] = booster.feature_names
    if getattr(booster, "feature_types", None):
        kw["feature_types"] = booster.feature_types
    nt = env_threads()
    if nt:
        kw["nthread"] = nt
    return xgb.DMatrix(X, **kw)


def _convert_tree(t: dict[str, Any], weight: float, idx: int) -> tuple[Tree, np.ndarray, np.ndarray]:
    """One XGBoost JSON tree -> normalized Tree; also returns (split features, raw split values)."""
    if int(t.get("tree_param", {}).get("size_leaf_vector", "1") or 1) > 1:
        raise UnsupportedTreeError("XGBoost multi-target (vector-leaf) trees are not supported")
    if any(int(s) != 0 for s in t.get("split_type", [])):
        raise UnsupportedTreeError("XGBoost categorical splits are not supported")
    left = np.asarray(t["left_children"], dtype=np.int64)
    right = np.asarray(t["right_children"], dtype=np.int64)
    n = left.shape[0]
    cond32 = np.asarray(t["split_conditions"], dtype=np.float64).astype(np.float32)
    is_leaf = left < 0
    feature = np.asarray(t["split_indices"], dtype=np.int64)
    tree = build_tree(
        left=left,
        right=right,
        feature=feature,
        threshold=np.where(is_leaf, 0.0, strict_lt_to_le(cond32)),
        default_left=np.asarray([bool(int(v)) for v in t["default_left"]], dtype=bool),
        leaf_value=cond32.astype(np.float64) * weight,
        cover=np.asarray(t.get("sum_hessian", np.ones(n)), dtype=np.float64),
        gain=np.asarray(t.get("loss_changes", np.zeros(n)), dtype=np.float64),
        what=f"XGBoost tree {idx}",
    )
    return tree, feature[~is_leaf], cond32[~is_leaf]


def xgboost_tree_ensemble(native: Any, *, verify: bool = True) -> TreeEnsemble:
    """Normalize an XGBoost ``Booster`` / fitted ``XGBModel`` into a :class:`TreeEnsemble`."""
    booster = _booster_of(native)
    model = json.loads(bytes(booster.save_raw(raw_format="json")).decode("utf-8"))
    learner = model["learner"]
    objective = str(learner["objective"]["name"])
    lmp = learner["learner_model_param"]
    if int(lmp.get("num_class", "0") or 0) > 1 or int(lmp.get("num_target", "1") or 1) > 1:
        raise UnsupportedTreeError("multi-class / multi-target XGBoost models are not supported")
    if objective in LOGISTIC_OBJECTIVES or objective in LOGITRAW_OBJECTIVES:
        transform = "logistic"
    elif objective in IDENTITY_OBJECTIVES:
        transform = "identity"
    else:
        raise UnsupportedTreeError(f"XGBoost objective {objective!r} is not supported for tree access")

    gb = learner["gradient_booster"]
    gname = gb.get("name")
    weights: list[float] | None = None
    if gname == "gbtree":
        gmodel = gb["model"]
    elif gname == "dart":
        gmodel = gb["gbtree"]["model"]
        weights = [float(w) for w in gb.get("weight_drop", [])]
    else:
        raise UnsupportedTreeError(f"XGBoost booster {gname!r} has no trees (gblinear?)")
    trees_json = list(gmodel["trees"])
    if any(int(v) != 0 for v in gmodel.get("tree_info", [])):
        raise UnsupportedTreeError("XGBoost model has trees for more than one output group")

    # Fitted sklearn wrappers with early stopping predict with iteration_range=(0, best_iteration + 1).
    n_use, best = len(trees_json), None
    if native is not booster:
        try:
            best = int(native.best_iteration)
        except (AttributeError, TypeError, ValueError):
            best = None
    if best is not None:
        indptr = gmodel.get("iteration_indptr")
        if indptr is not None and best + 1 < len(indptr):
            n_use = int(indptr[best + 1])
        else:
            npt = int(gmodel.get("gbtree_model_param", {}).get("num_parallel_tree", "1") or 1)
            n_use = min(len(trees_json), (best + 1) * npt)

    trees: list[Tree] = []
    split_vals: dict[int, list[np.ndarray]] = {}
    for i, t in enumerate(trees_json[:n_use]):
        w = weights[i] if weights is not None and i < len(weights) else 1.0
        tree, feats, conds = _convert_tree(t, w, i)
        trees.append(tree)
        for f in np.unique(feats):
            split_vals.setdefault(int(f), []).append(conds[feats == f])

    base_param = _parse_scalar_param(lmp.get("base_score", "0.5"), "base_score")
    base32 = float(np.float32(base_param))
    candidates: list[tuple[str, float]] = []
    if transform == "logistic":
        if 0.0 < base32 < 1.0:
            candidates.append(("logit(base_score)", math.log(base32 / (1.0 - base32))))
        candidates.append(("base_score", base32))
    else:
        candidates.append(("base_score", base32))

    n_features = int(lmp.get("num_feature", "0") or 0)
    te = TreeEnsemble(
        trees=trees,
        n_features=n_features,
        base_score=candidates[0][1],
        output_transform=transform,
        sigmoid_scale=1.0,
        average_output=False,
        model_kind="xgboost",
        feature_names=list(booster.feature_names) if booster.feature_names else None,
        meta={
            "loader": "xgboost",
            "source": "xgboost.Booster.save_raw(json)",
            "objective": objective,
            "booster": gname,
            "base_score_param": base_param,
            "n_trees_total": len(trees_json),
            "n_trees_used": n_use,
            "best_iteration": best,
            "xgboost_version": ".".join(str(v) for v in model.get("version", [])),
            "cover": "sum_hessian",
        },
    )
    if verify:
        thresholds = {f: np.unique(np.concatenate(v)) for f, v in split_vals.items()}
        _verify_and_set_base_margin(te, booster, candidates, thresholds, best)
    return te


def _verify_and_set_base_margin(
    te: TreeEnsemble,
    booster: Any,
    candidates: list[tuple[str, float]],
    thresholds: dict[int, np.ndarray],
    best: int | None,
) -> None:
    """Check the parsed trees against XGBoost's own margins and pin the base margin."""
    X = probe_matrix(thresholds, te.n_features, n_rows=256, seed=20240917)
    rng = (0, best + 1) if best is not None else (0, 0)
    native = np.asarray(
        booster.predict(_dmatrix(booster, X), output_margin=True, iteration_range=rng), dtype=np.float64
    ).reshape(-1)
    tree_sum = te.predict_raw(X) - te.base_score
    offset = native - tree_sum
    center = float(np.median(offset))
    spread = float(np.max(np.abs(offset - center)))
    # XGBoost accumulates margins in float32; allow for that, but not for a mis-parsed tree.
    tol = 1e-4 + 2e-6 * max(1, te.n_trees)
    if spread > tol:
        raise UnsupportedTreeError(
            f"XGBoost tree dump does not reproduce the model's margins (max error {spread:.3g}); "
            "tree-based analyses will fall back to black-box methods"
        )
    name, value = min(candidates, key=lambda c: abs(c[1] - center))
    if abs(value - center) > tol:
        log.warning(
            "XGBoost base margin %.9g does not match any known base_score convention %s; using the "
            "value measured from the model's own margins",
            center,
            candidates,
        )
        name, value = "measured", center
    te.base_score = float(value)
    te.meta.update(
        {"base_score_source": name, "base_margin": float(value), "verify_max_abs_margin_diff": spread}
    )


class XGBoostLoader(ModelLoader):
    kind: ClassVar[str] = "xgboost"
    extensions: ClassVar[tuple[str, ...]] = (
        ".json",
        ".ubj",
        ".model",
        ".bst",
        ".xgb",
        ".pkl",
        ".pickle",
        ".joblib",
        ".jbl",
        ".sav",
    )
    safe_extensions: ClassVar[tuple[str, ...]] = (".json", ".ubj", ".model", ".bst", ".xgb")
    description: ClassVar[str] = (
        "XGBoost Booster: .json/.ubj model (safe, Booster.load_model) or pickled Booster/XGBClassifier "
        "(--allow-pickle)"
    )

    def can_load(self, path: Path) -> bool:
        p = Path(path)
        suf = p.suffix.lower()
        if suf not in self.extensions:
            return False
        head = read_head(p, 256)
        if head is None:
            return True
        if self.is_pickle(p):
            return pickle_claims(p, ("xgboost",))
        if head.startswith(LIGHTGBM_TEXT_MAGIC):
            return False
        return b"learner" in head

    def load(self, path: Path, *, allow_pickle: bool = False) -> Any:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"XGBoost model file not found: {p}")
        self.check_pickle_policy(p, allow_pickle)  # before anything is deserialized
        nt = env_threads()
        if self.is_pickle(p):
            obj = load_pickled(p, allow_pickle=allow_pickle, what="XGBoost")
            if not self.supports_native(obj):
                raise TypeError(
                    f"{p} contains a {describe_type(obj)}, not an XGBoost Booster/XGBModel; "
                    "set model_kind to match the model or load it in the adapter"
                )
            if nt:
                if hasattr(obj, "set_params") and hasattr(obj, "get_booster"):
                    obj.set_params(n_jobs=nt)
                else:
                    obj.set_param({"nthread": nt})
            return obj
        import xgboost as xgb

        booster = xgb.Booster(params={"nthread": nt} if nt else None)
        try:
            # Always load from bytes: XGBoost detects JSON vs UBJSON from the content (not the name), and
            # its C++ file open fails on Windows paths with non-ASCII characters (C:\Users\José\...).
            booster.load_model(bytearray(p.read_bytes()))
        except xgb.core.XGBoostError as e:
            raise ValueError(
                f"{p} is not an XGBoost model that Booster.load_model can read ({str(e).splitlines()[0]}); "
                "save it with booster.save_model('model.json') or 'model.ubj'"
            ) from e
        log.debug("loaded XGBoost model %s (%d boosting rounds)", p, booster.num_boosted_rounds())
        return booster

    def supports_native(self, obj: Any) -> bool:
        if not _is_xgb_obj(obj):
            return False
        try:
            import xgboost as xgb
        except ImportError:  # pragma: no cover
            return False
        return isinstance(obj, (xgb.Booster, xgb.XGBModel))

    def predict_proba(self, native: Any, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        X = as_float_matrix(X)
        n = X.shape[0]
        booster = _booster_of(native)
        objective = booster_objective(booster)
        if isinstance(native, xgb.XGBModel) and hasattr(native, "predict_proba"):
            if objective in LOGISTIC_OBJECTIVES:
                # predict_proba returns float32 probabilities, which tie at 1.0 for margins above ~16.6 and
                # lose resolution near 0; sigmoid the raw margin in float64 instead (same iteration range).
                margin = positive_column(np.asarray(native.predict(X, output_margin=True)), n,
                                         f"{type(native).__name__}.predict(output_margin=True)")
                return _sigmoid(margin)
            return positive_column(native.predict_proba(X), n, f"{type(native).__name__}.predict_proba")
        if isinstance(native, xgb.XGBModel):  # regressor wrapper: its own iteration_range logic
            margin_or_p = lambda raw: np.asarray(native.predict(X, output_margin=raw))  # noqa: E731
        else:
            dm = _dmatrix(booster, X)
            margin_or_p = lambda raw: np.asarray(booster.predict(dm, output_margin=raw))  # noqa: E731
        if objective in LOGITRAW_OBJECTIVES or objective in LOGISTIC_OBJECTIVES:
            # float64 sigmoid of the raw margin: XGBoost's own probabilities are float32 and tie at 1.0
            return _sigmoid(positive_column(margin_or_p(True), n, "XGBoost predict"))
        if objective in HARD_LABEL_OBJECTIVES:
            return positive_column(margin_or_p(False), n, "XGBoost predict")
        if objective in IDENTITY_OBJECTIVES:
            return np.clip(positive_column(margin_or_p(False), n, "XGBoost predict"), 0.0, 1.0)
        raise ValueError(
            f"XGBoost objective {objective!r} does not produce a single malicious-class probability; "
            "wrap the model in an adapter whose predict_proba maps its output to [0, 1]"
        )

    def tree_ensemble(self, native: Any) -> TreeEnsemble:
        return xgboost_tree_ensemble(native)

    def library_version(self) -> str | None:
        try:
            import xgboost

            return str(xgboost.__version__)
        except ImportError:  # pragma: no cover
            return None
