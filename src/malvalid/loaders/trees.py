"""Normalized, framework-independent tree-ensemble representation.

Every tree-model loader converts its native model into a :class:`TreeEnsemble` so tree-structure
analyses (M5 backdoor screening, M7 importance/SHAP) work uniformly and — crucially — run in the
trusted parent process on plain numbers, never on the deserialized untrusted model object.

Split semantics are normalized to: go LEFT iff ``x[feature] <= threshold``. Loaders for frameworks
with strict ``<`` splits (XGBoost) must convert thresholds (see :func:`strict_lt_to_le`).
Leaf ``value`` is the raw-margin contribution *including* shrinkage/learning rate.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from malvalid.core import UnsupportedTreeError

LEAF = -1

# missing-value handling per node
MISSING_NAN = 0  # NaN -> default child; everything else compared normally
MISSING_NONE = 1  # NaN is treated as 0.0 and compared (LightGBM missing_type=None)
MISSING_ZERO = 2  # 0.0 (|x| <= 1e-35) and NaN -> default child (LightGBM missing_type=Zero)

_K_ZERO = float(np.float32(1e-35))  # LightGBM kZeroThreshold is the float literal 1e-35f


def strict_lt_to_le(threshold: np.ndarray | float) -> np.ndarray:
    """Convert float32 ``x < t`` splits into the equivalent ``x <= t'`` (for float32 inputs)."""
    t = np.asarray(threshold, dtype=np.float32)
    return np.nextafter(t, np.float32(-np.inf)).astype(np.float64)


@dataclass
class Tree:
    children_left: np.ndarray  # (m,) int32, LEAF for leaves
    children_right: np.ndarray  # (m,) int32
    children_default: np.ndarray  # (m,) int32 — child taken for missing values (LEAF at leaves)
    feature: np.ndarray  # (m,) int32, LEAF at leaves
    threshold: np.ndarray  # (m,) float64, go left iff x <= threshold
    value: np.ndarray  # (m,) float64; leaves: raw contribution; internal: cover-weighted mean
    cover: np.ndarray  # (m,) float64 training samples (or hessian) reaching the node
    gain: np.ndarray  # (m,) float64 split gain (0 at leaves)
    missing_type: np.ndarray  # (m,) int8 MISSING_*

    @property
    def n_nodes(self) -> int:
        return int(self.children_left.shape[0])

    def is_leaf(self) -> np.ndarray:
        return self.children_left == LEAF

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Leaf node id reached by every row of ``X``."""
        n = X.shape[0]
        node = np.zeros(n, dtype=np.int64)
        active = np.full(n, self.children_left[0] != LEAF)
        while active.any():
            rows = np.flatnonzero(active)
            nd = node[rows]
            x = np.asarray(X[rows, self.feature[nd]], dtype=np.float64)
            mt = self.missing_type[nd]
            isnan = np.isnan(x)
            x = np.where(isnan & (mt == MISSING_NONE), 0.0, x)
            go_default = (isnan & (mt != MISSING_NONE)) | (
                (mt == MISSING_ZERO) & (np.abs(x) <= _K_ZERO)
            )
            nxt = np.where(
                go_default,
                self.children_default[nd],
                np.where(x <= self.threshold[nd], self.children_left[nd], self.children_right[nd]),
            )
            node[rows] = nxt
            active[rows] = self.children_left[nxt] != LEAF
        return node

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.value[self.apply(X)]

    def iter_paths(self) -> Iterator[tuple[int, list[tuple[int, str, float]], float, float]]:
        """Yield ``(leaf_id, conditions, leaf_value, leaf_cover)`` for every root->leaf path.

        ``conditions`` is a list of ``(feature, op, threshold)`` with op ``"<="`` or ``">"``.
        Missing-value routing is ignored (feature vectors are dense).
        """
        stack: list[tuple[int, list[tuple[int, str, float]]]] = [(0, [])]
        while stack:
            nd, conds = stack.pop()
            if self.children_left[nd] == LEAF:
                yield nd, conds, float(self.value[nd]), float(self.cover[nd])
                continue
            f, t = int(self.feature[nd]), float(self.threshold[nd])
            stack.append((int(self.children_right[nd]), conds + [(f, ">", t)]))
            stack.append((int(self.children_left[nd]), conds + [(f, "<=", t)]))

    def fill_internal_values(self) -> None:
        """Set internal node values to the cover-weighted mean of their children (post-order)."""
        order: list[int] = []
        stack = [0]
        while stack:
            nd = stack.pop()
            order.append(nd)
            if self.children_left[nd] != LEAF:
                stack.append(int(self.children_left[nd]))
                stack.append(int(self.children_right[nd]))
        for nd in reversed(order):
            if self.children_left[nd] != LEAF:
                l, r = int(self.children_left[nd]), int(self.children_right[nd])
                cl, cr = self.cover[l], self.cover[r]
                tot = cl + cr
                self.value[nd] = (
                    (self.value[l] * cl + self.value[r] * cr) / tot
                    if tot > 0
                    else 0.5 * (self.value[l] + self.value[r])
                )
                if self.cover[nd] <= 0:
                    self.cover[nd] = tot


@dataclass
class TreeEnsemble:
    """An additive binary-classification tree ensemble.

    ``proba = sigmoid(sigmoid_scale * raw)`` for ``output_transform == "logistic"``,
    ``proba = clip(raw, 0, 1)`` for ``"identity"``, where
    ``raw = base_score + sum(tree raw)`` (or the mean over trees if ``average_output``).
    """

    trees: list[Tree]
    n_features: int
    base_score: float = 0.0
    output_transform: str = "logistic"  # "logistic" | "identity"
    sigmoid_scale: float = 1.0
    average_output: bool = False
    model_kind: str = "unknown"
    feature_names: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- prediction (reference implementation) -----------------------------------------------

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] < self.n_features:
            raise ValueError(f"X has shape {X.shape}; ensemble expects {self.n_features} features")
        acc = np.zeros(X.shape[0], dtype=np.float64)
        for t in self.trees:
            acc += t.predict_raw(X)
        if self.average_output and self.trees:
            acc /= len(self.trees)
        return acc + self.base_score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.output_transform == "logistic":
            return 1.0 / (1.0 + np.exp(-self.sigmoid_scale * raw))
        return np.clip(raw, 0.0, 1.0)

    def fidelity(self, X: np.ndarray, model_proba: np.ndarray) -> float:
        """Max |ensemble proba - model proba| on ``X`` (used to verify a dump is faithful)."""
        return float(np.max(np.abs(self.predict_proba(X) - np.asarray(model_proba, np.float64))))

    # ---- structure summaries -----------------------------------------------------------------

    @property
    def n_trees(self) -> int:
        return len(self.trees)

    def feature_importance(self, kind: str = "gain") -> np.ndarray:
        """(n_features,) importance: ``gain`` (sum split gain), ``split`` (count), ``cover``."""
        imp = np.zeros(self.n_features, dtype=np.float64)
        for t in self.trees:
            internal = t.children_left != LEAF
            f = t.feature[internal]
            if kind == "gain":
                np.add.at(imp, f, t.gain[internal])
            elif kind == "split":
                np.add.at(imp, f, 1.0)
            elif kind == "cover":
                np.add.at(imp, f, t.cover[internal])
            else:
                raise ValueError(f"unknown importance kind {kind!r}")
        return imp

    def iter_paths(self) -> Iterator[tuple[int, int, list[tuple[int, str, float]], float, float]]:
        """Yield ``(tree_index, leaf_id, conditions, leaf_value, leaf_cover)`` over all trees."""
        for ti, t in enumerate(self.trees):
            for leaf, conds, val, cov in t.iter_paths():
                yield ti, leaf, conds, val, cov

    # ---- SHAP ----------------------------------------------------------------------------------

    def to_shap_model(self) -> dict[str, Any]:
        """A dict model accepted by ``shap.TreeExplainer`` (raw/log-odds output space)."""
        scale = self.sigmoid_scale if self.output_transform == "logistic" else 1.0
        div = float(len(self.trees)) if (self.average_output and self.trees) else 1.0
        trees = []
        for t in self.trees:
            feat = t.feature.astype(np.int64).copy()
            feat[t.children_left == LEAF] = -2
            trees.append(
                {
                    "children_left": t.children_left.astype(np.int64),
                    "children_right": t.children_right.astype(np.int64),
                    "children_default": np.where(
                        t.children_left == LEAF, -1, t.children_default
                    ).astype(np.int64),
                    "features": feat,
                    "thresholds": t.threshold.astype(np.float64),
                    "values": (t.value * scale / div).reshape(-1, 1).astype(np.float64),
                    "node_sample_weight": t.cover.astype(np.float64),
                }
            )
        return {
            "trees": trees,
            "base_offset": float(self.base_score * scale),
            "tree_output": "log_odds" if self.output_transform == "logistic" else "raw_value",
            "objective": "binary_crossentropy" if self.output_transform == "logistic" else "squared_error",
            "input_dtype": np.float64,
            "internal_dtype": np.float64,
        }

    # ---- serialization (safe: npz without pickle) ----------------------------------------------

    _NODE_FIELDS = (
        "children_left",
        "children_right",
        "children_default",
        "feature",
        "threshold",
        "value",
        "cover",
        "gain",
        "missing_type",
    )

    def to_bytes(self) -> bytes:
        offsets = np.cumsum([0] + [t.n_nodes for t in self.trees]).astype(np.int64)
        arrays: dict[str, np.ndarray] = {"offsets": offsets}
        for fld in self._NODE_FIELDS:
            parts = [getattr(t, fld) for t in self.trees]
            arrays[fld] = np.concatenate(parts) if parts else np.empty(0)
        meta = {
            "n_features": self.n_features,
            "base_score": self.base_score,
            "output_transform": self.output_transform,
            "sigmoid_scale": self.sigmoid_scale,
            "average_output": self.average_output,
            "model_kind": self.model_kind,
            "feature_names": self.feature_names,
            "meta": self.meta,
        }
        arrays["meta_json"] = np.array(json.dumps(meta))
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "TreeEnsemble":
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            offsets = z["offsets"]
            cols = {f: z[f] for f in cls._NODE_FIELDS}
            meta = json.loads(str(z["meta_json"]))
        dtypes = {
            "children_left": np.int32,
            "children_right": np.int32,
            "children_default": np.int32,
            "feature": np.int32,
            "missing_type": np.int8,
        }
        trees = []
        for a, b in zip(offsets[:-1], offsets[1:]):
            kw = {
                f: np.ascontiguousarray(cols[f][a:b]).astype(dtypes.get(f, np.float64))
                for f in cls._NODE_FIELDS
            }
            trees.append(Tree(**kw))
        return cls(
            trees=trees,
            n_features=int(meta["n_features"]),
            base_score=float(meta["base_score"]),
            output_transform=meta["output_transform"],
            sigmoid_scale=float(meta["sigmoid_scale"]),
            average_output=bool(meta["average_output"]),
            model_kind=meta["model_kind"],
            feature_names=meta.get("feature_names"),
            meta=meta.get("meta", {}),
        )


# --------------------------------------------------------------------------------------------------
# LightGBM dump_model() -> TreeEnsemble (kept here so analyses can be tested without loaders)
# --------------------------------------------------------------------------------------------------

_LGB_MISSING = {"None": MISSING_NONE, "NaN": MISSING_NAN, "Zero": MISSING_ZERO}


def from_lightgbm_dump(dump: dict[str, Any]) -> TreeEnsemble:
    """Convert ``lightgbm.Booster.dump_model()`` output (binary objective) to a TreeEnsemble."""
    if int(dump.get("num_class", 1)) != 1 or int(dump.get("num_tree_per_iteration", 1)) != 1:
        raise UnsupportedTreeError("only binary (single-output) LightGBM models are supported")
    objective = str(dump.get("objective", "binary"))
    obj_name = objective.split()[0]
    if obj_name in ("binary", "cross_entropy", "xentropy"):
        transform, scale = "logistic", 1.0
        for tok in objective.split()[1:]:
            if tok.startswith("sigmoid:"):
                scale = float(tok.split(":", 1)[1])
    elif obj_name in ("regression", "regression_l2", "l2", "mse", "regression_l1", "huber"):
        transform, scale = "identity", 1.0
    else:
        raise UnsupportedTreeError(f"unsupported LightGBM objective {objective!r}")

    trees: list[Tree] = []
    for ti in dump["tree_info"]:
        nodes: list[dict[str, Any]] = []

        def visit(node: dict[str, Any]) -> int:
            my = len(nodes)
            rec: dict[str, Any] = {}
            nodes.append(rec)
            if "split_index" in node:
                if node.get("decision_type", "<=") != "<=":
                    raise UnsupportedTreeError(
                        f"LightGBM split decision_type {node.get('decision_type')!r} (categorical) unsupported"
                    )
                rec.update(
                    feature=int(node["split_feature"]),
                    threshold=float(node["threshold"]),
                    gain=float(node.get("split_gain", 0.0)),
                    cover=float(node.get("internal_count", 0.0)),
                    missing=_LGB_MISSING.get(str(node.get("missing_type", "None")), MISSING_NONE),
                    default_left=bool(node.get("default_left", True)),
                    value=float(node.get("internal_value", 0.0)),
                )
                rec["left"] = visit(node["left_child"])
                rec["right"] = visit(node["right_child"])
            else:
                rec.update(
                    feature=LEAF,
                    threshold=0.0,
                    gain=0.0,
                    cover=float(node.get("leaf_count", 0.0)),
                    missing=MISSING_NAN,
                    value=float(node["leaf_value"]),
                    left=LEAF,
                    right=LEAF,
                )
            return my

        visit(ti["tree_structure"])
        m = len(nodes)
        cl = np.array([r["left"] for r in nodes], dtype=np.int32)
        cr = np.array([r["right"] for r in nodes], dtype=np.int32)
        cd = np.array(
            [
                LEAF if r["left"] == LEAF else (r["left"] if r.get("default_left", True) else r["right"])
                for r in nodes
            ],
            dtype=np.int32,
        )
        t = Tree(
            children_left=cl,
            children_right=cr,
            children_default=cd,
            feature=np.array([r["feature"] for r in nodes], dtype=np.int32),
            threshold=np.array([r["threshold"] for r in nodes], dtype=np.float64),
            value=np.array([r["value"] for r in nodes], dtype=np.float64),
            cover=np.array([r["cover"] for r in nodes], dtype=np.float64),
            gain=np.array([r["gain"] for r in nodes], dtype=np.float64),
            missing_type=np.array([r["missing"] for r in nodes], dtype=np.int8),
        )
        if m > 1 and np.all(t.cover[~t.is_leaf()] == 0):
            t.fill_internal_values()
        trees.append(t)

    names = dump.get("feature_names")
    return TreeEnsemble(
        trees=trees,
        n_features=int(dump.get("max_feature_idx", -1)) + 1,
        base_score=0.0,
        output_transform=transform,
        sigmoid_scale=scale,
        average_output=bool(dump.get("average_output", False)),
        model_kind="lightgbm",
        feature_names=list(names) if names else None,
        meta={"objective": objective, "version": dump.get("version")},
    )
