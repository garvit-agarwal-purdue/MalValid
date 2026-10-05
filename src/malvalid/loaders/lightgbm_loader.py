"""LightGBM loader (``model_kind: lightgbm``).

Artifacts:

* ``.txt`` / ``.model`` / ``.lgb`` — LightGBM's text model (``Booster.save_model``), a safe,
  non-executable format. Read as UTF-8 text in Python and loaded with
  ``lightgbm.Booster(model_str=...)`` (works with non-ASCII paths on Windows). The file is recognised
  by its ``tree`` header, so a ``.model`` file is routed here only when it really is a LightGBM text
  model (XGBoost also uses ``.model``).
* pickled ``lightgbm.Booster`` / ``LGBMClassifier`` (``.pkl``/``.joblib``/...) — only with
  ``--allow-pickle``.

LightGBM cannot load its own ``dump_model()`` JSON, so ``.json`` is not a LightGBM artifact format
(``.json`` models are routed to the XGBoost loader). Re-export with ``booster.save_model("model.txt")``.

Tree access reuses the frozen core converter :func:`malvalid.loaders.trees.from_lightgbm_dump`,
then makes it exact under LightGBM's input handling (zero band, ``missing_type=None``). Categorical
splits (``decision_type "=="``, e.g. in the EMBER2024 reference model) are first rewritten as
equivalent chains of ``x <= t`` splits (see :func:`expand_categorical_splits`) and the result is
verified against ``Booster.predict(raw_score=True)``. Linear-leaf and multi-class models raise
:class:`~malvalid.core.UnsupportedTreeError`.

This module also hosts small helpers shared by malvalid's built-in loaders (thread count from the
sandbox environment, static pickle inspection, guarded unpickling, probe matrices for verifying
tree dumps). They are private to the loaders package; plugin authors should use
:mod:`malvalid.loaders.base`.
"""

from __future__ import annotations

import bz2
import logging
import lzma
import os
import pickletools
import zlib
from pathlib import Path
from typing import Any, ClassVar, Iterable

import numpy as np

from malvalid.core import PickleRefused, UnsupportedTreeError
from malvalid.loaders.base import PICKLE_EXTENSIONS, ModelLoader, sniff_format
from malvalid.loaders.trees import LEAF, MISSING_NAN, MISSING_NONE, Tree, TreeEnsemble, from_lightgbm_dump

log = logging.getLogger("malvalid.loaders")

LIGHTGBM_TEXT_MAGIC = (b"tree\n", b"tree\r\n")

# --------------------------------------------------------------------------------------------------
# Shared helpers for the built-in loaders
# --------------------------------------------------------------------------------------------------


def env_threads() -> int | None:
    """Thread count the sandbox asked for (``MALVALID_THREADS``, else ``OMP_NUM_THREADS``)."""
    for var in ("MALVALID_THREADS", "OMP_NUM_THREADS"):
        v = os.environ.get(var)
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                log.debug("ignoring non-integer %s=%r", var, v)
    return None


def describe_type(obj: Any) -> str:
    t = type(obj)
    return f"{t.__module__}.{t.__qualname__}"


def read_head(path: Path, n: int = 1 << 16) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return None


# Modules that appear in almost every pickle and say nothing about which framework produced it.
_GENERIC_PICKLE_PACKAGES = frozenset(
    {
        "copyreg",
        "copy_reg",
        "builtins",
        "__builtin__",
        "_codecs",
        "codecs",
        "collections",
        "functools",
        "operator",
        "numpy",
        "joblib",
        "types",
        "abc",
        "_collections_abc",
        "pickle",
        "_pickle",
        "array",
        "datetime",
        "decimal",
        "enum",
        "pathlib",
        "re",
        "scipy",
    }
)
_STR_PUSH_OPS = frozenset(
    {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING", "STRING"}
)
_NO_PUSH_OPS = frozenset(
    {"PROTO", "FRAME", "MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT", "MARK", "STOP", "POP", "POP_MARK"}
)


def _pickle_payload(path: Path, max_bytes: int) -> bytes | None:
    """The first ``max_bytes`` of the (decompressed) pickle stream — never unpickles."""
    fmt = sniff_format(path)
    head = read_head(path, max_bytes)
    if head is None:
        return None
    if fmt == "pickle":
        return head
    if fmt != "joblib-compressed":
        return None
    try:
        if head[:1] == b"\x78":
            return zlib.decompressobj().decompress(head, max_bytes)
        if head[:3] == b"\x1f\x8b\x08":
            return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(head, max_bytes)
        if head[:3] == b"BZh":
            return bz2.BZ2Decompressor().decompress(head, max_bytes)
        if head[:6] == b"\xfd7zXZ\x00":
            return lzma.LZMADecompressor().decompress(head, max_bytes)
    except (zlib.error, OSError, EOFError, lzma.LZMAError, ValueError):
        return None
    return None


def pickle_top_module(path: Path, *, max_bytes: int = 1 << 20) -> str | None:
    """Module of the first framework class referenced by a pickle / joblib artifact.

    Read statically with :func:`pickletools.genops` (opcodes are parsed, nothing is constructed or
    executed); joblib's zlib/gzip/bz2/xz containers are decompressed up to ``max_bytes``. Generic
    modules (``copyreg``, ``builtins``, ``numpy``, ``joblib``...) are skipped, so a pickled
    ``LGBMClassifier`` yields ``"lightgbm.sklearn"``. Returns None when undeterminable.
    """
    data = _pickle_payload(Path(path), max_bytes)
    if not data:
        return None
    recent: list[str | None] = []
    memo: dict[int, str | None] = {}

    def found(mod: str | None) -> str | None:
        if mod and mod.split(".")[0] not in _GENERIC_PICKLE_PACKAGES:
            return mod
        return None

    try:
        for op, arg, _pos in pickletools.genops(data):
            name = op.name
            if name in _STR_PUSH_OPS:
                recent.append(arg if isinstance(arg, str) else str(arg))
            elif name == "MEMOIZE":
                memo[len(memo)] = recent[-1] if recent else None
            elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
                memo[int(arg)] = recent[-1] if recent else None
            elif name in ("GET", "BINGET", "LONG_BINGET"):
                recent.append(memo.get(int(arg)))
            elif name in ("GLOBAL", "INST"):
                mod = found(str(arg).split(" ", 1)[0])
                if mod:
                    return mod
                recent.append(None)
            elif name == "STACK_GLOBAL":
                mod = found(recent[-2] if len(recent) >= 2 else None)
                if mod:
                    return mod
                recent.append(None)
            elif name not in _NO_PUSH_OPS:
                recent.append(None)
            if len(recent) > 64:
                del recent[:32]
    except Exception:  # truncated stream / joblib's raw array bytes: stop scanning  # noqa: BLE001
        return None
    return None


def pickle_claims(path: Path, packages: Iterable[str]) -> bool:
    """Routing helper for pickle-based artifacts: False only when the pickle provably holds an object
    from a package not in ``packages`` (undeterminable pickles are claimed by extension)."""
    mod = pickle_top_module(path)
    return mod is None or mod.split(".")[0] in set(packages)


def load_pickled(path: Path, *, allow_pickle: bool, what: str) -> Any:
    """Unpickle ``path`` (joblib or plain pickle) — only when the operator allowed pickles."""
    if not allow_pickle:  # callers run check_pickle_policy first; this is defence in depth
        raise PickleRefused(
            f"{path} is a pickle-based {what} artifact; pickles can execute code on load. Re-export "
            "the model in a non-pickle format or pass --allow-pickle to accept the risk."
        )
    log.warning(
        "loading pickle-based %s artifact %s (--allow-pickle): unpickling can execute arbitrary code; "
        "prefer a non-pickle export for production models",
        what,
        path,
    )
    import joblib

    return joblib.load(Path(path))


def as_float_matrix(X: Any) -> np.ndarray:
    """C-contiguous 2-D float32/float64 matrix (other dtypes become float64)."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {arr.shape}")
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float64)
    return np.ascontiguousarray(arr)


def positive_column(p: np.ndarray, n: int, what: str) -> np.ndarray:
    """(n,) malicious-class scores from a (n,), (n, 1) or binary (n, 2) prediction."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 2 and p.shape[1] == 2:
        p = p[:, 1]
    elif p.ndim == 2 and p.shape[1] == 1:
        p = p[:, 0]
    if p.shape != (n,):
        raise ValueError(
            f"{what} returned predictions of shape {p.shape} for {n} rows; malvalid evaluates binary "
            "detectors (one malicious-class probability per row). Multi-class models need an adapter "
            "that maps them to a single malicious score."
        )
    return p


def probe_matrix(
    thresholds: dict[int, np.ndarray], n_features: int, *, n_rows: int = 256, seed: int = 0, nan: bool = True
) -> np.ndarray:
    """float32 rows that exercise tree splits: values drawn from each feature's split thresholds and
    their float32 neighbours (ties and both sides), plus NaNs and random values. Deterministic."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, max(1, n_features))).astype(np.float32)
    for f, t in thresholds.items():
        if f >= n_features or t.size == 0:
            continue
        t32 = np.asarray(t, dtype=np.float32)
        pool = np.concatenate(
            [
                t32,
                np.nextafter(t32, np.float32(np.inf)),
                np.nextafter(t32, np.float32(-np.inf)),
            ]
        )
        pool = pool[np.isfinite(pool)]
        if pool.size:
            X[:, f] = rng.choice(pool, size=n_rows)
    if nan:
        X[rng.random(X.shape) < 0.08] = np.nan
    return X


def build_tree(
    *,
    left: np.ndarray,
    right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    default_left: np.ndarray,
    leaf_value: np.ndarray,
    cover: np.ndarray,
    gain: np.ndarray | None = None,
    root: int = 0,
    what: str = "tree",
) -> Tree:
    """A normalized :class:`Tree` from parallel per-node arrays indexed by native node id.

    ``left[i] < 0`` marks a leaf; thresholds must already be in ``x <= threshold`` form and
    ``leaf_value`` must already include shrinkage. Reachable nodes are renumbered in pre-order from
    ``root`` (unreachable/deleted nodes are dropped). Internal-node cover is recomputed as the sum
    of the children's cover so TreeSHAP on :meth:`TreeEnsemble.to_shap_model` is exactly additive;
    internal values become the cover-weighted mean of the children. NaN goes to the default child.
    """
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    n = left.shape[0]
    order: list[int] = []
    stack = [int(root)]
    while stack:
        nd = stack.pop()
        if nd < 0 or nd >= n or len(order) >= n:
            raise UnsupportedTreeError(f"malformed {what}: bad child id {nd} or a cycle")
        order.append(nd)
        if left[nd] >= 0:
            stack.append(int(right[nd]))
            stack.append(int(left[nd]))
    new_id = {old: i for i, old in enumerate(order)}
    if len(new_id) != len(order):
        raise UnsupportedTreeError(f"malformed {what}: a node is reachable twice")
    m = len(order)
    old = np.asarray(order, dtype=np.int64)
    is_leaf = left[old] < 0
    cl = np.full(m, LEAF, dtype=np.int32)
    cr = np.full(m, LEAF, dtype=np.int32)
    internal = np.flatnonzero(~is_leaf)
    cl[internal] = [new_id[int(v)] for v in left[old[internal]]]
    cr[internal] = [new_id[int(v)] for v in right[old[internal]]]
    dl = np.asarray(default_left, dtype=bool)[old]
    cd = np.where(is_leaf, LEAF, np.where(dl, cl, cr)).astype(np.int32)
    feat = np.where(is_leaf, LEAF, np.asarray(feature, dtype=np.int64)[old]).astype(np.int32)
    thr = np.where(is_leaf, 0.0, np.asarray(threshold, dtype=np.float64)[old])
    val = np.where(is_leaf, np.asarray(leaf_value, dtype=np.float64)[old], 0.0)
    cov = np.asarray(cover, dtype=np.float64)[old].copy()
    cov[is_leaf] = np.maximum(cov[is_leaf], 1e-12)  # TreeSHAP divides by cover
    g = np.zeros(m) if gain is None else np.where(is_leaf, 0.0, np.asarray(gain, dtype=np.float64)[old])
    for i in reversed(range(m)):  # pre-order reversed: children before parents
        if not is_leaf[i]:
            a, b = cl[i], cr[i]
            cov[i] = cov[a] + cov[b]
            val[i] = (val[a] * cov[a] + val[b] * cov[b]) / cov[i]
    return Tree(
        children_left=cl,
        children_right=cr,
        children_default=cd,
        feature=feat,
        threshold=thr,
        value=val,
        cover=cov,
        gain=g,
        missing_type=np.full(m, MISSING_NAN, dtype=np.int8),
    )


def ensemble_thresholds(te: TreeEnsemble) -> dict[int, np.ndarray]:
    """{feature: split thresholds} over all internal nodes of a normalized ensemble."""
    acc: dict[int, list[np.ndarray]] = {}
    for t in te.trees:
        internal = t.children_left != LEAF
        for f in np.unique(t.feature[internal]):
            acc.setdefault(int(f), []).append(t.threshold[internal & (t.feature == f)])
    return {f: np.unique(np.concatenate(v)) for f, v in acc.items()}


# --------------------------------------------------------------------------------------------------
# LightGBM loader
# --------------------------------------------------------------------------------------------------

_BINARY_OBJECTIVES = ("binary", "cross_entropy", "xentropy")
_REGRESSION_OBJECTIVES = ("regression", "regression_l2", "l2", "mse", "regression_l1", "l1", "huber")


def _is_lgb_obj(obj: Any) -> bool:
    return type(obj).__module__.split(".")[0] == "lightgbm"


def _booster_of(obj: Any) -> Any:
    """The ``lightgbm.Booster`` of a Booster or a fitted ``LGBMModel``."""
    if hasattr(obj, "dump_model"):  # already a Booster
        return obj
    try:
        return obj.booster_
    except Exception as e:  # noqa: BLE001 - LGBMNotFittedError (an AttributeError subclass) etc.
        raise UnsupportedTreeError(f"LightGBM estimator {type(obj).__name__} is not fitted: {e}") from e


def _objective_name(booster: Any) -> str:
    """Objective name without its arguments (``"binary sigmoid:1"`` -> ``"binary"``)."""
    obj = (getattr(booster, "params", None) or {}).get("objective")
    if obj is None or callable(obj):
        # In-memory boosters only carry the user's params; the model header always has it.
        # num_iteration=1 keeps the dump to the header plus one iteration's trees.
        obj = booster.dump_model(num_iteration=1).get("objective", "regression")
    return str(obj).split()[0]


# LightGBM's kZeroThreshold is the float literal 1e-35f stored in a double.
LGB_ZERO_THRESHOLD = float(np.float32(1e-35))


def _normalize_lightgbm(te: TreeEnsemble) -> dict[str, int]:
    """Make a ``from_lightgbm_dump`` ensemble exact under LightGBM's input handling (in place).

    * **Zero band.** LightGBM's predictor drops dense input values with ``|x| <= kZeroThreshold``
      (they become 0.0) before any split is evaluated, and its binning places split thresholds
      exactly at ``+-kZeroThreshold`` for features with many zeros. With a plain ``x <= t`` rule an
      input equal to ``-kZeroThreshold`` would go left at ``t = -kZeroThreshold`` while LightGBM
      sends it right (as 0). Moving ``t`` in ``[0, kz)`` to ``kz`` and ``t`` in ``[-kz, 0)`` to just
      below ``-kz`` reproduces LightGBM for every input.
    * **missing_type=None.** Such a node compares NaN as 0.0, i.e. it is a NaN-default node whose
      default child is the side 0.0 goes to. Rewriting it that way keeps ``Tree.apply`` identical
      and makes SHAP (which routes NaN via ``children_default``) agree with it.

    ``missing_type=Zero`` nodes are kept (their zero band has no SHAP equivalent).
    """
    kz = LGB_ZERO_THRESHOLD
    below = float(np.nextafter(-kz, -np.inf))
    n_thr = n_none = 0
    for t in te.trees:
        internal = t.children_left != LEAF
        th = t.threshold
        pos = internal & (th >= 0.0) & (th < kz)
        neg = internal & (th >= -kz) & (th < 0.0)
        th[pos] = kz
        th[neg] = below
        n_thr += int(pos.sum() + neg.sum())
        none = internal & (t.missing_type == MISSING_NONE)
        if none.any():
            t.children_default[none] = np.where(
                0.0 <= th[none], t.children_left[none], t.children_right[none]
            ).astype(t.children_default.dtype)
            t.missing_type[none] = MISSING_NAN
            n_none += int(none.sum())
    return {"zero_thresholds_adjusted": n_thr, "missing_none_normalized": n_none}


# ---- categorical splits ---------------------------------------------------------------------------
#
# LightGBM's categorical test (``decision_type == "=="``) sends a row LEFT iff x is not NaN and
# ``int(x)`` (C++ truncation toward zero) is a non-negative member of the split's category set;
# NaN, x <= -1, +-inf and values outside the int range go RIGHT (verified empirically for both
# missing types). On the real line that test is piecewise constant: category 0 covers (-1, 1),
# category c >= 1 covers [c, c + 1). Each categorical node is therefore rewritten as a balanced
# chain of ``x <= b`` splits over those segments, with a copy of the left/right subtree under every
# segment. The copies share the original subtree's cover and gain evenly, so total cover, gain
# importance and the TreeSHAP expected value are unchanged (split *counts* are inflated).

MAX_EXPANDED_NODES = 4_000_000  # across the ensemble; beyond that tree access is refused


def _parse_categories(threshold: Any) -> list[int]:
    out: list[int] = []
    for tok in str(threshold).split("||"):
        tok = tok.strip()
        if tok:
            out.append(int(float(tok)))
    return out


def categorical_partition(categories: Iterable[int]) -> tuple[list[float], list[bool]]:
    """``(boundaries, in_set)`` for LightGBM's ``int(x) in categories`` test.

    ``boundaries`` are strictly increasing ``x <= b`` split points; ``in_set[k]`` says whether
    segment ``k`` (``(-inf, b_0]``, ``(b_0, b_1]``, ..., ``(b_last, inf)``) goes left.
    Segment 0 always goes right (it holds x <= -1); NaN also goes right.
    """
    cats = sorted({int(c) for c in categories if int(c) >= 0})
    runs: list[list[int]] = []
    for c in cats:
        if runs and c == runs[-1][1]:
            runs[-1][1] = c + 1
        else:
            runs.append([c, c + 1])
    bounds: list[float] = []
    flags = [False]
    for lo, hi in runs:
        # int(x) in [lo, hi)  <=>  x in (-1, hi) if lo == 0 else x in [lo, hi)
        b_lo = -1.0 if lo == 0 else float(np.nextafter(float(lo), -np.inf))
        b_hi = float(np.nextafter(float(hi), -np.inf))
        bounds += [b_lo, b_hi]
        flags += [True, False]
    return bounds, flags


def _node_cover(node: dict[str, Any]) -> float:
    return float(node.get("internal_count" if "split_index" in node else "leaf_count", 0.0))


class _CategoricalExpander:
    def __init__(self, max_nodes: int) -> None:
        self.max_nodes = max_nodes
        self.n_nodes = 0
        self.n_categorical = 0

    def _take(self, k: int = 1) -> None:
        self.n_nodes += k
        if self.n_nodes > self.max_nodes:
            raise UnsupportedTreeError(
                f"rewriting the LightGBM model's categorical splits as threshold splits needs more than "
                f"{self.max_nodes:,} nodes; tree-based analyses fall back to black-box methods"
            )

    def expand(self, node: dict[str, Any], scale: float) -> dict[str, Any]:
        if "split_index" not in node:
            self._take()
            leaf = dict(node)
            leaf["leaf_count"] = float(node.get("leaf_count", 0.0)) * scale
            return leaf
        if node.get("decision_type", "<=") != "==":
            self._take()
            out = {k: v for k, v in node.items() if k not in ("left_child", "right_child")}
            out["split_gain"] = float(node.get("split_gain", 0.0)) * scale
            out["internal_count"] = float(node.get("internal_count", 0.0)) * scale
            out["left_child"] = self.expand(node["left_child"], scale)
            out["right_child"] = self.expand(node["right_child"], scale)
            return out
        self.n_categorical += 1
        bounds, flags = categorical_partition(_parse_categories(node["threshold"]))
        n_in = sum(flags)
        n_out = len(flags) - n_in
        if n_in == 0:  # empty category set: every row goes right
            return self.expand(node["right_child"], scale)
        segments = [
            self.expand(node["left_child"], scale / n_in) if f else self.expand(node["right_child"], scale / n_out)
            for f in flags
        ]
        return self._chain(node, bounds, segments, 0, len(segments) - 1, scale, root=True)

    def _chain(
        self,
        cat: dict[str, Any],
        bounds: list[float],
        segments: list[dict[str, Any]],
        i: int,
        j: int,
        scale: float,
        *,
        root: bool = False,
    ) -> dict[str, Any]:
        if i == j:
            return segments[i]
        self._take()
        mid = (i + j) // 2  # boundary ``mid`` separates segment mid (x <= b) from mid + 1
        left = self._chain(cat, bounds, segments, i, mid, scale)
        right = self._chain(cat, bounds, segments, mid + 1, j, scale)
        return {
            "split_index": cat.get("split_index", 0),
            "split_feature": int(cat["split_feature"]),
            "split_gain": float(cat.get("split_gain", 0.0)) * scale if root else 0.0,
            "threshold": float(bounds[mid]),
            "decision_type": "<=",
            "default_left": True,  # NaN goes to segment 0 (right of the categorical test)
            "missing_type": "NaN",
            "internal_value": float(cat.get("internal_value", 0.0)),
            "internal_count": _node_cover(left) + _node_cover(right),
            "left_child": left,
            "right_child": right,
        }


def count_categorical_splits(dump: dict[str, Any]) -> int:
    n = 0
    for ti in dump.get("tree_info", []):
        stack = [ti["tree_structure"]]
        while stack:
            nd = stack.pop()
            if "split_index" in nd:
                n += nd.get("decision_type", "<=") == "=="
                stack.extend((nd["left_child"], nd["right_child"]))
    return n


def expand_categorical_splits(
    dump: dict[str, Any], *, max_nodes: int = MAX_EXPANDED_NODES
) -> tuple[dict[str, Any], dict[str, int]]:
    """A copy of a ``dump_model()`` dict whose categorical splits are rewritten as ``<=`` chains."""
    ex = _CategoricalExpander(max_nodes)
    tree_info = [{**ti, "tree_structure": ex.expand(ti["tree_structure"], 1.0)} for ti in dump["tree_info"]]
    return {**dump, "tree_info": tree_info}, {
        "categorical_splits_expanded": ex.n_categorical,
        "nodes_after_expansion": ex.n_nodes,
    }


def _categorical_probe(te: TreeEnsemble, dump: dict[str, Any], seed: int = 17) -> np.ndarray:
    """Probe rows around every numeric threshold plus the category grid of categorical features."""
    X = probe_matrix(ensemble_thresholds(te), te.n_features, n_rows=512, seed=seed)
    cat_vals: dict[int, set[int]] = {}
    for ti in dump.get("tree_info", []):
        stack = [ti["tree_structure"]]
        while stack:
            nd = stack.pop()
            if "split_index" in nd:
                if nd.get("decision_type") == "==":
                    cat_vals.setdefault(int(nd["split_feature"]), set()).update(_parse_categories(nd["threshold"]))
                stack.extend((nd["left_child"], nd["right_child"]))
    rng = np.random.default_rng(seed + 1)
    special = np.array([np.nan, -1.0, -0.999, -0.5, 0.0, 0.5, 0.999, 1.0, 1e6, np.inf, -np.inf], dtype=np.float32)
    for f, cs in cat_vals.items():
        c = np.asarray(sorted(cs), dtype=np.float32)
        pool = np.concatenate([special, c, c + 0.5, c + 1.0, c - 1.0, np.nextafter(c, np.float32(-np.inf))])
        X[:, f] = rng.choice(pool.astype(np.float32), size=X.shape[0])
    return X


# --------------------------------------------------------------------------------------------------


class LightGBMLoader(ModelLoader):
    kind: ClassVar[str] = "lightgbm"
    extensions: ClassVar[tuple[str, ...]] = (
        ".txt",
        ".model",
        ".lgb",
        ".pkl",
        ".pickle",
        ".joblib",
        ".jbl",
        ".sav",
    )
    safe_extensions: ClassVar[tuple[str, ...]] = (".txt", ".model", ".lgb")
    description: ClassVar[str] = (
        "LightGBM Booster: text model (.txt/.model/.lgb, safe) or pickled Booster/LGBMClassifier "
        "(--allow-pickle)"
    )

    def can_load(self, path: Path) -> bool:
        p = Path(path)
        suf = p.suffix.lower()
        if suf not in self.extensions:
            return False
        head = read_head(p, 16)
        if head is None:  # missing/unreadable: route by extension only (load() reports the error)
            return suf != ".model"
        if self.is_pickle(p):
            return pickle_claims(p, ("lightgbm",))
        return head.startswith(LIGHTGBM_TEXT_MAGIC)

    def load(self, path: Path, *, allow_pickle: bool = False) -> Any:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"LightGBM model file not found: {p}")
        self.check_pickle_policy(p, allow_pickle)  # before anything is deserialized
        if self.is_pickle(p):
            obj = load_pickled(p, allow_pickle=allow_pickle, what="LightGBM")
            if not self.supports_native(obj):
                raise TypeError(
                    f"{p} contains a {describe_type(obj)}, not a LightGBM Booster/LGBMModel; "
                    "set model_kind to match the model or load it in the adapter"
                )
            return obj
        head = read_head(p, 16) or b""
        if not head.startswith(LIGHTGBM_TEXT_MAGIC):
            raise ValueError(
                f"{p} is not a LightGBM text model (it should start with 'tree'); save the model with "
                "booster.save_model('model.txt'). LightGBM cannot load dump_model() JSON."
            )
        import lightgbm as lgb

        # Read the text in Python and pass it as model_str: LightGBM's C++ file open uses the narrow ANSI
        # API on Windows and fails on paths with non-ASCII characters (e.g. C:\Users\José\model.txt).
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"{p} is not a UTF-8 LightGBM text model ({e})") from e
        booster = lgb.Booster(model_str=text)
        del text
        log.debug("loaded LightGBM text model %s (%d trees)", p, booster.num_trees())
        return booster

    def supports_native(self, obj: Any) -> bool:
        if not _is_lgb_obj(obj):
            return False
        try:
            import lightgbm as lgb
        except ImportError:  # pragma: no cover
            return False
        return isinstance(obj, (lgb.Booster, lgb.LGBMModel))

    def predict_proba(self, native: Any, X: np.ndarray) -> np.ndarray:
        X = as_float_matrix(X)
        kw: dict[str, Any] = {}
        nt = env_threads()
        if nt:
            kw["num_threads"] = nt
        if hasattr(native, "predict_proba"):  # LGBMClassifier
            p = native.predict_proba(X, **kw)
            return positive_column(p, X.shape[0], f"{type(native).__name__}.predict_proba")
        booster = _booster_of(native)
        obj = _objective_name(booster)
        p = positive_column(np.asarray(native.predict(X, **kw)), X.shape[0], "LightGBM predict")
        if obj in _BINARY_OBJECTIVES:
            return p
        if obj in _REGRESSION_OBJECTIVES:
            return np.clip(p, 0.0, 1.0)
        raise ValueError(
            f"LightGBM objective {obj!r} does not produce malicious-class probabilities; wrap the model "
            "in an adapter whose predict_proba maps its scores to [0, 1]"
        )

    def tree_ensemble(self, native: Any) -> TreeEnsemble:
        booster = _booster_of(native)
        dump = booster.dump_model()
        for ti in dump.get("tree_info", []):
            stack = [ti["tree_structure"]]
            while stack:
                nd = stack.pop()
                if "split_index" in nd:
                    stack.extend((nd["left_child"], nd["right_child"]))
                elif nd.get("leaf_coeff"):
                    raise UnsupportedTreeError(
                        "LightGBM linear_tree models (linear leaves) cannot be normalized to constant-leaf trees"
                    )
        cat_stats: dict[str, Any] = {}
        src = dump
        if count_categorical_splits(dump):
            src, cat_stats = expand_categorical_splits(dump)
        te = from_lightgbm_dump(src)
        stats = _normalize_lightgbm(te)
        if cat_stats:
            cat_stats["verify_max_abs_raw_diff"] = self._verify_raw(te, booster, dump)
            cat_stats["cover"] = "categorical subtree copies share the original cover evenly"
            log.info(
                "LightGBM model: %d categorical splits rewritten as threshold chains (%d nodes)",
                cat_stats["categorical_splits_expanded"],
                cat_stats["nodes_after_expansion"],
            )
        te.meta.update({"loader": self.kind, "source": "lightgbm.Booster.dump_model", **stats, **cat_stats})
        return te

    @staticmethod
    def _verify_raw(te: TreeEnsemble, booster: Any, dump: dict[str, Any]) -> float:
        """Max |raw margin| difference vs ``booster.predict(raw_score=True)`` on probe rows."""
        X = _categorical_probe(te, dump)
        kw: dict[str, Any] = {"raw_score": True}
        nt = env_threads()
        if nt:
            kw["num_threads"] = nt
        native = np.asarray(booster.predict(X, **kw), dtype=np.float64).reshape(-1)
        err = float(np.max(np.abs(te.predict_raw(X) - native)))
        if not err <= 1e-7 * max(1.0, float(te.n_trees)):
            raise UnsupportedTreeError(
                f"the rewritten LightGBM categorical splits do not reproduce the model's margins (max error "
                f"{err:.3g}); tree-based analyses fall back to black-box methods"
            )
        return err

    def library_version(self) -> str | None:
        try:
            import lightgbm

            return str(lightgbm.__version__)
        except ImportError:  # pragma: no cover
            return None


__all__ = [
    "LightGBMLoader",
    "PICKLE_EXTENSIONS",
    "env_threads",
    "pickle_top_module",
    "pickle_claims",
    "load_pickled",
    "probe_matrix",
    "build_tree",
    "ensemble_thresholds",
    "categorical_partition",
    "expand_categorical_splits",
]
