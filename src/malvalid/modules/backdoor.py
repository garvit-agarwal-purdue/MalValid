"""M5 — backdoor / poisoning screening (screening only, never a certification).

Question answered: *are there signs that the detector was trained to let specific, deliberately
marked malware through?* Two independent screens run, depending on what the run provides:

(a) **Operator trigger hypotheses** (needs the canonical corpus, ``feature_space``). The researcher
    supplies candidate triggers — ``{feature_name_or_index: value}`` maps — in ``params.triggers``
    or a YAML/JSON file (``params.triggers_path``). Each trigger's values are stamped onto the
    malicious eval vectors the model currently detects, and the detection rate is re-measured:
    ``drop = DR_before - DR_after`` (undetected vectors are left unchanged). A drop above
    ``max_trigger_drop`` is a red flag: that value combination switches detections off.

(b) **Static tree-structure anomaly scan** (needs ``tree_access``). The normalized ensemble's
    root->leaf paths are analysed as plain numbers — no samples are stamped or scored. The
    signature of a trigger-style backdoor in boosted trees is a *narrow* split (few training
    samples go that way) that routes to a *strongly benign* leaf, on a feature the model otherwise
    barely uses, with *low training cover*. Per feature ``f``:

    * ``gate`` branches: a split child whose training cover is at most ``RHO_GATE`` (0.5) of its
      parent's and whose subtree mean output is lower (more benign) than its sibling's;
    * ``gate_share[f]`` = sum over gate branches on ``f`` of
      ``gain x narrow(rho) x low_cover(phi)`` / total split gain of the ensemble, where ``rho`` is
      the branch's share of its parent's cover, ``phi`` its share of the tree's root cover and
      ``narrow`` / ``low_cover`` are log-ramps (1 at ``rho <= 0.02`` / ``phi <= 0.005``, 0 at
      ``rho >= 0.5`` / ``phi >= 0.25``);
    * ``low_importance[f]`` = ``clip((1 - pct) / (1 - q), 0, 1)`` where ``pct`` is the percentile of
      ``f``'s *non-gate* gain among the features the ensemble uses and ``q`` =
      ``low_importance_quantile``;
    * ``feature_suspicion[f]`` = ``low_importance[f] x clip((gate_share[f] - 0.002) / 0.008, 0, 1)``.

    Per root->leaf rule (leaf ``l`` of tree ``t``): ``benign`` = how far the leaf sits below the
    tree's cover-weighted mean output, relative to the tree's largest leaf deviation (0..1);
    ``narrowness`` = log-ramp of the product of the gate ``rho`` values on the path; ``low_cover``
    = log-ramp of the leaf's share of the tree's cover; the *dominant* feature is the gate feature
    on the path with the highest ``feature_suspicion``. Then
    ``suspicion = feature_suspicion[dominant] x sqrt(benign x max(narrowness, low_cover))``
    (0 for rules with no gate or a non-benign leaf). Rules with ``suspicion >= min_suspicion``
    are flagged.

    Subtree means are recomputed from the leaves (cover-weighted) because some LightGBM dumps store
    internal values without shrinkage. The thresholds above were calibrated on toy models and on
    LightGBM models (300 trees x 64 leaves) trained on 200k EMBER2018 feature vectors with and
    without a planted label-flip trigger on a low-importance import feature (0.5 % / 2 % of the
    malicious rows poisoned: top suspicion 0.84 / 1.00; the same model trained clean: 0.04; the
    public EMBER2018 and EMBER2024 reference models: 0.11 and 0.04).

Gate checks (warn gate): ``max_trigger_drop <= max_trigger_drop`` (anchors ideal 0.02, floor 0.8)
when triggers are supplied, and ``n_flagged_rules <= 0`` when the scan ran. The scan check's axis
score is 1.0 when nothing is flagged, otherwise ``0.75 x (1 - s) / (1 - min_suspicion)`` (kept just
below the 0.75 pass line) where ``s`` is the top rule suspicion.

ART's poisoning detectors (activation clustering, spectral signatures, provenance/RONI defences)
need neural-network activations and/or the training data; they do not apply to tree ensembles and
are recorded as not applicable. Absence of findings is **not** proof the model is backdoor-free.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from malvalid.core import ConfigError, GateCheck, GateMode, Module, ModuleResult, Requirement
from malvalid.loaders.trees import LEAF
from malvalid.modules.performance import batch_rows, graded_check, score_indices, unique_by_hash
from malvalid.scoring import PASS_LINE

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.context import RunContext
    from malvalid.loaders.trees import Tree, TreeEnsemble
    from malvalid.schemas.base import FeatureSchema

log = logging.getLogger("malvalid.modules.backdoor")

MODULE_ID = "backdoor_screen"

# ---- static scan calibration (see module docstring) ----------------------------------------------
RHO_GATE = 0.5  # a branch taking at most this share of its parent's cover counts as "narrow"
RHO_NARROW = 0.02  # narrowness ramp reaches 1 at this share
PHI_HIGH = 0.25  # low-cover ramp is 0 at/above this share of the tree's root cover
PHI_LOW = 0.005  # ... and 1 at/below this share
SHARE_LOW = 0.002  # gate-gain share (of total ensemble gain) where feature suspicion starts
SHARE_HIGH = 0.01  # ... and saturates

ABSENCE_NOTE = (
    "Screening only: absence of findings is NOT proof that the model is backdoor-free. Certifying "
    "backdoor-freedom requires training-time access (the training data, or retraining), which this "
    "gate does not have."
)
ART_NOTE_TREES = (
    "ART's poisoning/backdoor detectors (activation clustering, spectral signatures, provenance and "
    "RONI defences) target neural networks and/or need the training data; they are not applicable to "
    "tree ensembles and were not run."
)
ART_NOTE_OTHER = (
    "ART's poisoning/backdoor detectors (activation clustering, spectral signatures, provenance and "
    "RONI defences) need the model's internal activations and/or the training data, which a "
    "black-box pre-deployment gate does not have; they were not run."
)


# --------------------------------------------------------------------------------------------------
# Trigger hypotheses
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Trigger:
    """One operator-supplied trigger hypothesis: feature index -> value to stamp."""

    name: str
    features: dict[int, float]
    labels: dict[int, str] = field(default_factory=dict)  # index -> feature name (for display)


def _resolve_feature(key: Any, names: Sequence[str], index_of: dict[str, int], where: str) -> int:
    """Map a trigger key (feature name or index) to a feature index, with a helpful error."""
    dim = len(names)
    if isinstance(key, bool):
        raise ConfigError(f"{where}: feature key {key!r} is not a feature name or index")
    if isinstance(key, (int, np.integer)):
        i = int(key)
        if not 0 <= i < dim:
            raise ConfigError(f"{where}: feature index {i} is outside the schema's range [0, {dim})")
        return i
    if isinstance(key, str):
        k = key.strip()
        if k in index_of:
            return index_of[k]
        if k.lstrip("-").isdigit():
            return _resolve_feature(int(k), names, index_of, where)
        close = difflib.get_close_matches(k, names, n=3, cutoff=0.6)
        hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else (
            "; use a name from the schema's feature_names() (e.g. "
            f"{names[0]!r}) or an integer index in [0, {dim})"
        )
        raise ConfigError(f"{where}: unknown feature {k!r}{hint}")
    raise ConfigError(f"{where}: feature key {key!r} must be a feature name (str) or index (int)")


def parse_triggers(items: Any, schema: "FeatureSchema", *, source: str = "triggers") -> list[Trigger]:
    """Validate trigger specs ``[{name, features: {feature_name_or_index: value}}, ...]``.

    Raises :class:`ConfigError` with an actionable message for malformed specs, unknown feature
    names, out-of-range indices and non-finite values.
    """
    if items is None:
        return []
    if isinstance(items, dict) and "triggers" in items:
        items = items["triggers"]
    if not isinstance(items, (list, tuple)):
        raise ConfigError(
            f"{MODULE_ID}.{source} must be a list of {{name, features: {{feature: value}}}} entries, "
            f"got {type(items).__name__}"
        )
    names = list(schema.feature_names())
    index_of = {n: i for i, n in enumerate(names)}
    out: list[Trigger] = []
    seen: set[str] = set()
    for k, item in enumerate(items):
        where = f"{MODULE_ID}.{source}[{k}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where} must be a mapping with 'name' and 'features', got {type(item).__name__}")
        unknown = set(item) - {"name", "features", "description"}
        if unknown:
            raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; expected 'name' and 'features'")
        feats = item.get("features")
        if not isinstance(feats, dict) or not feats:
            raise ConfigError(f"{where}.features must be a non-empty mapping {{feature_name_or_index: value}}")
        name = str(item.get("name") or f"trigger_{k + 1}")
        if name in seen:
            raise ConfigError(f"{where}: duplicate trigger name {name!r}")
        seen.add(name)
        resolved: dict[int, float] = {}
        for key, val in feats.items():
            i = _resolve_feature(key, names, index_of, f"{where} ({name!r})")
            if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
                raise ConfigError(f"{where} ({name!r}): value for {names[i]!r} must be a number, got {val!r}")
            v = float(val)
            if not math.isfinite(v):
                raise ConfigError(f"{where} ({name!r}): value for {names[i]!r} must be finite, got {val!r}")
            if i in resolved and resolved[i] != v:
                raise ConfigError(f"{where} ({name!r}): feature {names[i]!r} is given twice with different values")
            resolved[i] = v
        out.append(Trigger(name=name, features=resolved, labels={i: names[i] for i in resolved}))
    return out


def load_triggers_file(path: str | Path, *, base_dir: Path | None = None) -> Any:
    """Read a YAML/JSON trigger file (a list, or a mapping with a ``triggers`` list)."""
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists() and base_dir is not None and (base_dir / p).exists():
        p = base_dir / p
    if not p.is_file():
        raise ConfigError(f"{MODULE_ID}.triggers_path: file not found: {path}")
    text = p.read_text(encoding="utf-8")
    try:
        if p.suffix.lower() == ".json":
            return json.loads(text)
        import yaml

        return yaml.safe_load(text)
    except Exception as e:  # yaml.YAMLError / json.JSONDecodeError
        raise ConfigError(f"{MODULE_ID}.triggers_path: cannot parse {p}: {e}") from e


# --------------------------------------------------------------------------------------------------
# Static tree-structure scan
# --------------------------------------------------------------------------------------------------


def _ramp_log(x: np.ndarray | float, hi: float, lo: float) -> np.ndarray:
    """1 at ``x <= lo``, 0 at ``x >= hi``, log-linear in between."""
    xv = np.maximum(np.asarray(x, dtype=np.float64), 1e-300)
    return np.clip((math.log(hi) - np.log(xv)) / (math.log(hi) - math.log(lo)), 0.0, 1.0)


@dataclass
class _TreeStats:
    """Per-node statistics of one tree (reachable nodes only)."""

    levels: list[np.ndarray]  # BFS levels of node ids (root first)
    parent: np.ndarray  # (m,) parent node id, -1 at the root / unreachable nodes
    cover: np.ndarray  # (m,) subtree cover (sum of leaf covers)
    mean: np.ndarray  # (m,) cover-weighted mean leaf value of the subtree
    edge_nodes: np.ndarray  # non-root reachable nodes (children)
    rho: np.ndarray  # (len(edge_nodes),) child's share of the parent's cover
    gate: np.ndarray  # (len(edge_nodes),) narrow AND more benign than its sibling


def _tree_stats(t: "Tree") -> _TreeStats | None:
    """BFS order, parents and subtree cover/mean of ``t``; ``None`` for a single-leaf tree or a tree
    without cover statistics."""
    cl, cr = t.children_left, t.children_right
    m = t.n_nodes
    if m <= 1 or cl[0] == LEAF:
        return None
    parent = np.full(m, -1, dtype=np.int64)
    levels = [np.array([0], dtype=np.int64)]
    seen = 1
    while True:
        fr = levels[-1]
        internal = fr[cl[fr] != LEAF]
        if internal.size == 0:
            break
        kids = np.concatenate([cl[internal], cr[internal]]).astype(np.int64)
        seen += kids.size
        if seen > m:  # malformed (cycle / shared child) — refuse rather than loop
            from malvalid.core import UnsupportedTreeError

            raise UnsupportedTreeError("tree structure is not a tree (a node is reachable twice)")
        parent[cl[internal]] = internal
        parent[cr[internal]] = internal
        levels.append(kids)
    is_leaf = cl == LEAF
    cover = np.where(is_leaf, np.maximum(np.nan_to_num(t.cover, nan=0.0), 0.0), 0.0).astype(np.float64)
    sval = np.where(is_leaf, t.value * cover, 0.0)
    for lv in reversed(levels):
        internal = lv[cl[lv] != LEAF]
        if internal.size:
            cover[internal] = cover[cl[internal]] + cover[cr[internal]]
            sval[internal] = sval[cl[internal]] + sval[cr[internal]]
    if cover[0] <= 0:
        return None
    mean = np.where(cover > 0, sval / np.maximum(cover, 1e-300), 0.0)
    edge_nodes = np.concatenate(levels[1:]) if len(levels) > 1 else np.empty(0, dtype=np.int64)
    p = parent[edge_nodes]
    sib = np.where(cl[p] == edge_nodes, cr[p], cl[p])
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(cover[p] > 0, cover[edge_nodes] / np.maximum(cover[p], 1e-300), 1.0)
    gate = (cover[p] > 0) & (rho <= RHO_GATE) & (mean[edge_nodes] < mean[sib])
    return _TreeStats(levels, parent, cover, mean, edge_nodes, rho, gate)


@dataclass
class TreeScan:
    """Output of :func:`static_tree_scan` (arrays are per feature unless noted)."""

    n_trees: int
    n_trees_scanned: int
    n_rules: int  # root->leaf paths in scanned trees
    importance_source: str  # "gain" | "cover"
    total_gain: np.ndarray
    gate_gain: np.ndarray
    gate_share: np.ndarray
    low_importance: np.ndarray
    feature_suspicion: np.ndarray
    rule_suspicion: np.ndarray  # suspicion of every rule with suspicion > 0
    rule_dominant: np.ndarray  # dominant feature of those rules
    top_rules: list[dict[str, Any]]
    seconds: float


def _fmt_num(x: float) -> str:
    return f"{x:.6g}"


def _merge_conditions(conds: list[tuple[int, str, float]], names: Sequence[str] | None) -> list[str]:
    """``[(f, op, t)]`` -> readable per-feature interval strings, in first-appearance order."""
    order: list[int] = []
    lo: dict[int, float] = {}
    hi: dict[int, float] = {}
    for f, op, thr in conds:
        if f not in lo and f not in hi:
            order.append(f)
        if op == "<=":
            hi[f] = min(hi.get(f, math.inf), thr)
        else:
            lo[f] = max(lo.get(f, -math.inf), thr)
    out = []
    for f in order:
        nm = names[f] if names is not None and 0 <= f < len(names) else f"f{f}"
        a, b = lo.get(f), hi.get(f)
        if a is not None and b is not None:
            out.append(f"{_fmt_num(a)} < {nm} <= {_fmt_num(b)}")
        elif a is not None:
            out.append(f"{nm} > {_fmt_num(a)}")
        else:
            out.append(f"{nm} <= {_fmt_num(b)}")  # type: ignore[arg-type]
    return out


def _path_conditions(t: "Tree", parent: np.ndarray, leaf: int) -> list[tuple[int, str, float]]:
    conds: list[tuple[int, str, float]] = []
    node = leaf
    while parent[node] >= 0:
        p = int(parent[node])
        op = "<=" if int(t.children_left[p]) == node else ">"
        conds.append((int(t.feature[p]), op, float(t.threshold[p])))
        node = p
    conds.reverse()
    return conds


def static_tree_scan(
    te: "TreeEnsemble",
    *,
    low_importance_quantile: float = 0.5,
    max_rules: int = 50,
    feature_names: Sequence[str] | None = None,
    check: Callable[[], None] | None = None,
) -> TreeScan:
    """Score every root->leaf rule of ``te`` for backdoor-like structure (see module docstring).

    Works on plain arrays only; two vectorized passes over the trees (per-feature statistics, then
    per-rule scores). ``check`` is called periodically (deadline check).
    """
    t0 = time.monotonic()
    nf = int(te.n_features)
    tick = check or (lambda: None)
    q = float(low_importance_quantile)

    # ---- pass 1: per-feature gate statistics ---------------------------------------------------
    tot = np.zeros(nf)
    gate_gain = np.zeros(nf)
    gate_w = np.zeros(nf)
    tot_c = np.zeros(nf)
    gate_c = np.zeros(nf)
    gate_wc = np.zeros(nf)
    n_scanned = 0
    n_rules = 0
    for ti, t in enumerate(te.trees):
        if ti % 50 == 0:
            tick()
        st = _tree_stats(t)
        if st is None:
            continue
        n_scanned += 1
        internal = np.concatenate([lv[t.children_left[lv] != LEAF] for lv in st.levels])
        n_rules += int(internal.size + 1)
        f_int = t.feature[internal].astype(np.int64)
        g_int = np.maximum(np.nan_to_num(t.gain[internal], nan=0.0), 0.0)
        c_int = st.cover[internal]
        np.add.at(tot, f_int, g_int)
        np.add.at(tot_c, f_int, c_int)
        if st.edge_nodes.size:
            gm = st.gate
            e = st.edge_nodes[gm]
            p = st.parent[e]
            fg = t.feature[p].astype(np.int64)
            w = _ramp_log(st.rho[gm], RHO_GATE, RHO_NARROW) * _ramp_log(
                st.cover[e] / st.cover[0], PHI_HIGH, PHI_LOW
            )
            gp = np.maximum(np.nan_to_num(t.gain[p], nan=0.0), 0.0)
            np.add.at(gate_gain, fg, gp)
            np.add.at(gate_w, fg, gp * w)
            np.add.at(gate_c, fg, st.cover[p])
            np.add.at(gate_wc, fg, st.cover[p] * w)
    source = "gain"
    if tot.sum() <= 0:  # loader without split gains: fall back to node cover as the weight
        source = "cover"
        tot, gate_gain, gate_w = tot_c, gate_c, gate_wc
    total = float(tot.sum())
    used = tot > 0
    other = tot - gate_gain
    low_imp = np.ones(nf)
    if used.any():
        ou = np.sort(other[used])
        pct = np.searchsorted(ou, other[used], side="right") / ou.size
        low_imp[used] = np.clip((1.0 - pct) / max(1.0 - q, 1e-12), 0.0, 1.0)
    share = gate_w / total if total > 0 else np.zeros(nf)
    fsusp = low_imp * np.clip((share - SHARE_LOW) / (SHARE_HIGH - SHARE_LOW), 0.0, 1.0)

    # ---- pass 2: per-rule suspicion ---------------------------------------------------------------
    r_s: list[np.ndarray] = []
    r_tree: list[np.ndarray] = []
    r_leaf: list[np.ndarray] = []
    r_dom: list[np.ndarray] = []
    r_extra: list[np.ndarray] = []
    if n_scanned and fsusp.max() > 0:
        for ti, t in enumerate(te.trees):
            if ti % 50 == 0:
                tick()
            st = _tree_stats(t)
            if st is None:
                continue
            m = t.n_nodes
            logrho = np.zeros(m)
            best = np.full(m, -1.0)
            dom = np.full(m, -1, dtype=np.int64)
            pos = np.full(m, -1, dtype=np.int64)
            pos[st.edge_nodes] = np.arange(st.edge_nodes.size)
            for lv in st.levels[1:]:
                p = st.parent[lv]
                k = pos[lv]
                g = st.gate[k]
                fpar = t.feature[p].astype(np.int64)
                logrho[lv] = logrho[p] + np.where(g, np.log(np.maximum(st.rho[k], 1e-300)), 0.0)
                cand = np.where(g, fsusp[fpar], -1.0)
                upd = cand > best[p]
                best[lv] = np.where(upd, cand, best[p])
                dom[lv] = np.where(upd, fpar, dom[p])
            leaves = np.flatnonzero((t.children_left == LEAF) & ((st.parent >= 0)))
            leaves = leaves[best[leaves] > 0]
            if leaves.size == 0:
                continue
            all_leaves = np.flatnonzero((t.children_left == LEAF) & (st.parent >= 0))
            mroot = st.mean[0]
            spread = float(np.max(np.abs(t.value[all_leaves] - mroot))) if all_leaves.size else 0.0
            if spread <= 0:
                continue
            benign = np.clip((mroot - t.value[leaves]) / spread, 0.0, 1.0)
            narrow = _ramp_log(np.exp(logrho[leaves]), RHO_GATE, RHO_NARROW)
            lowc = _ramp_log(st.cover[leaves] / st.cover[0], PHI_HIGH, PHI_LOW)
            s = best[leaves] * np.sqrt(benign * np.maximum(narrow, lowc))
            keep = s > 0
            if not keep.any():
                continue
            r_s.append(s[keep])
            r_tree.append(np.full(int(keep.sum()), ti, dtype=np.int64))
            r_leaf.append(leaves[keep])
            r_dom.append(dom[leaves[keep]])
            r_extra.append(np.stack([benign[keep], narrow[keep], lowc[keep],
                                     st.cover[leaves[keep]], st.cover[leaves[keep]] / st.cover[0]], axis=1))
    if r_s:
        rs = np.concatenate(r_s)
        rt = np.concatenate(r_tree)
        rl = np.concatenate(r_leaf)
        rd = np.concatenate(r_dom)
        rx = np.concatenate(r_extra)
    else:
        rs = np.empty(0)
        rt = rl = rd = np.empty(0, dtype=np.int64)
        rx = np.empty((0, 5))

    # ---- top rules with readable conditions --------------------------------------------------
    top: list[dict[str, Any]] = []
    if rs.size and max_rules > 0:
        order = np.lexsort((rl, rt, -rs))[:max_rules]  # deterministic tie-break (tree, leaf)
        parents: dict[int, np.ndarray] = {}
        for rank, j in enumerate(order, start=1):
            ti, leaf = int(rt[j]), int(rl[j])
            t = te.trees[ti]
            if ti not in parents:
                st = _tree_stats(t)
                parents[ti] = st.parent if st is not None else np.full(t.n_nodes, -1)
            conds = _path_conditions(t, parents[ti], leaf)
            d = int(rd[j])
            top.append(
                {
                    "rank": rank,
                    "tree": ti,
                    "leaf": leaf,
                    "suspicion": float(rs[j]),
                    "dominant_feature": feature_names[d] if feature_names is not None else f"f{d}",
                    "dominant_feature_index": d,
                    "conditions": _merge_conditions(conds, feature_names),
                    "depth": len(conds),
                    "leaf_value": float(t.value[leaf]),
                    "benign_strength": float(rx[j, 0]),
                    "narrowness": float(rx[j, 1]),
                    "low_cover": float(rx[j, 2]),
                    "cover": float(rx[j, 3]),
                    "cover_fraction": float(rx[j, 4]),
                }
            )
    return TreeScan(
        n_trees=te.n_trees,
        n_trees_scanned=n_scanned,
        n_rules=n_rules,
        importance_source=source,
        total_gain=tot,
        gate_gain=gate_gain,
        gate_share=share,
        low_importance=low_imp,
        feature_suspicion=fsusp,
        rule_suspicion=rs,
        rule_dominant=rd,
        top_rules=top,
        seconds=time.monotonic() - t0,
    )


def scan_check_score(n_flagged: int, top_suspicion: float, min_suspicion: float) -> float:
    """Axis score of the static-scan check: 1.0 with no flagged rules, else
    ``0.75 * (1 - s) / (1 - min_suspicion)`` kept strictly below the 0.75 pass line."""
    if n_flagged <= 0:
        return 1.0
    s = float(min(max(top_suspicion, 0.0), 1.0))
    base = PASS_LINE * (1.0 - s) / (1.0 - min_suspicion) if min_suspicion < 1.0 else 0.0
    return float(min(max(base, 0.0), PASS_LINE - 1e-6))


# --------------------------------------------------------------------------------------------------
# Parameter validation
# --------------------------------------------------------------------------------------------------


def _num(params: dict[str, Any], key: str, lo: float, hi: float, *, lo_open: bool = False,
         hi_open: bool = False) -> float:
    v = params.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(float(v)):
        raise ConfigError(f"{MODULE_ID}.{key} must be a number, got {v!r}")
    f = float(v)
    if (f <= lo if lo_open else f < lo) or (f >= hi if hi_open else f > hi):
        rng = f"{'(' if lo_open else '['}{lo:g}, {hi:g}{')' if hi_open else ']'}"
        raise ConfigError(f"{MODULE_ID}.{key} must be in {rng}, got {v!r}")
    return f


def _int(params: dict[str, Any], key: str, lo: int, *, allow_none: bool = False) -> int | None:
    v = params.get(key)
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < lo:
        extra = " or null" if allow_none else ""
        raise ConfigError(f"{MODULE_ID}.{key} must be an integer >= {lo}{extra}, got {v!r}")
    return int(v)


def _pp(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f} pp"


# --------------------------------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------------------------------


class BackdoorScreenModule(Module):
    id = MODULE_ID
    code = "M5"
    title = "Backdoor / poisoning screening"
    description = (
        "Screens for signs that the detector was trained to let deliberately marked malware through: "
        "stamps operator-supplied trigger hypotheses onto detected malicious vectors and measures the "
        "detection drop, and statically scans the tree ensemble for narrow, low-cover splits on otherwise "
        "low-importance features that route to strongly benign leaves. Screening only — absence of "
        "findings is not proof of a clean model."
    )
    requires = ()
    requires_any = (Requirement.TREE_ACCESS, Requirement.FEATURE_SPACE)
    screening = True
    default_gate = GateMode.WARN
    default_params = {
        "triggers": [],
        "triggers_path": None,
        "max_trigger_drop": 0.20,
        "n_samples": 2000,
        "max_rules_reported": 50,
        "low_importance_quantile": 0.5,
        "min_suspicion": 0.5,
    }

    # ---- params ------------------------------------------------------------------------------
    @staticmethod
    def _scalar_params(p: dict[str, Any]) -> dict[str, Any]:
        return {
            "max_trigger_drop": _num(p, "max_trigger_drop", 0.0, 1.0),
            "n_samples": _int(p, "n_samples", 1, allow_none=True),
            "max_rules_reported": _int(p, "max_rules_reported", 0),
            "low_importance_quantile": _num(p, "low_importance_quantile", 0.0, 1.0, hi_open=True),
            "min_suspicion": _num(p, "min_suspicion", 0.0, 1.0, lo_open=True),
        }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        """Scalar values and the trigger containers; trigger contents need the schema (run time)."""
        cls._scalar_params(params)
        raw = params.get("triggers")
        if raw is not None and not isinstance(raw, (list, tuple)):
            raise ConfigError(f"{MODULE_ID}.triggers must be a list of {{name, features: {{feature: value}}}} entries")
        tp = params.get("triggers_path")
        if tp not in (None, "") and not isinstance(tp, (str, Path)):
            raise ConfigError(f"{MODULE_ID}.triggers_path must be a file path or null, got {tp!r}")

    def _params(self, ctx: "RunContext") -> dict[str, Any]:
        p = ctx.params
        out = self._scalar_params(p)
        items: list[Any] = []
        raw = p.get("triggers")
        if raw is not None:
            if not isinstance(raw, (list, tuple)):
                raise ConfigError(
                    f"{MODULE_ID}.triggers must be a list of {{name, features: {{feature: value}}}} entries"
                )
            items.extend(raw)
        triggers = parse_triggers(items, ctx.schema, source="triggers")
        tp = p.get("triggers_path")
        if tp not in (None, ""):
            if not isinstance(tp, (str, Path)):
                raise ConfigError(f"{MODULE_ID}.triggers_path must be a file path or null, got {tp!r}")
            src = getattr(ctx.config, "source_path", None)
            base = Path(src).parent if src else None
            from_file = parse_triggers(load_triggers_file(tp, base_dir=base), ctx.schema, source="triggers_path")
            clash = {t.name for t in triggers} & {t.name for t in from_file}
            if clash:
                raise ConfigError(f"{MODULE_ID}: trigger name(s) {sorted(clash)} defined in both triggers and triggers_path")
            triggers.extend(from_file)
        out["triggers"] = triggers
        return out

    # ---- run -----------------------------------------------------------------------------------
    def run(self, ctx: "RunContext") -> ModuleResult:
        prm = self._params(ctx)
        triggers: list[Trigger] = prm["triggers"]
        has_trees = ctx.has(Requirement.TREE_ACCESS)
        has_fs = ctx.has(Requirement.FEATURE_SPACE) and ctx.corpus is not None
        te = ctx.model.tree_ensemble() if has_trees else None
        if te is None and not triggers:
            reason = (
                f"{ctx.missing_reason(Requirement.TREE_ACCESS)} (the static scan needs the trees) and no trigger "
                f"hypotheses were supplied ({MODULE_ID}.triggers / triggers_path)"
            )
            return self.skip(ctx, reason, notes=[ABSENCE_NOTE])

        notes: list[str] = []
        checks: list[GateCheck] = []
        metrics: dict[str, Any] = {"n_triggers": len(triggers), "scan_ran": False}
        details: dict[str, Any] = {
            "art_poisoning_detectors": {
                "applicable": False,
                "reason": ART_NOTE_TREES if te is not None else ART_NOTE_OTHER,
            },
        }
        finding_parts: list[str] = []

        # (b) static scan
        scan_sentence = self._run_scan(ctx, te, prm, checks, metrics, details, notes)
        if scan_sentence:
            finding_parts.append(scan_sentence)

        # (a) trigger hypotheses
        if triggers:
            trig_sentence = self._run_triggers(ctx, triggers, prm, has_fs, checks, metrics, details, notes)
            finding_parts.append(trig_sentence)
        else:
            notes.append(
                f"No trigger hypotheses supplied ({MODULE_ID}.triggers or triggers_path), so the targeted "
                "trigger test did not run. If you suspect a specific trigger (e.g. a string, import or header "
                "value), list it there to measure its effect on detection."
            )

        notes.append(details["art_poisoning_detectors"]["reason"])
        notes.insert(0, ABSENCE_NOTE)
        finding = " ".join(finding_parts) + " Screening only — absence of findings is not proof of a clean model."
        return self.result(ctx, finding=finding, checks=checks, metrics=metrics, details=details, notes=notes)

    # ---- (b) --------------------------------------------------------------------------------------
    def _run_scan(self, ctx: "RunContext", te: "TreeEnsemble | None", prm: dict[str, Any],
                  checks: list[GateCheck], metrics: dict[str, Any], details: dict[str, Any],
                  notes: list[str]) -> str:
        m = prm["min_suspicion"]
        if te is None:
            why = ctx.missing_reason(Requirement.TREE_ACCESS)
            notes.append(f"Static tree-structure scan not run: {why}.")
            details["scan"] = {"ran": False, "reason": why}
            return ""
        names = list(ctx.schema.feature_names())
        if te.n_features > len(names):
            names = names + [f"f{i}" for i in range(len(names), te.n_features)]
        scan = static_tree_scan(
            te,
            low_importance_quantile=prm["low_importance_quantile"],
            max_rules=prm["max_rules_reported"],
            feature_names=names,
            check=ctx.check_deadline,
        )
        log.info("static tree scan: %d trees, %d rules, %.2fs", scan.n_trees, scan.n_rules, scan.seconds)
        if scan.n_trees_scanned == 0:
            why = (
                "the tree ensemble carries no training-cover statistics (or only single-leaf trees), so "
                "branch narrowness and leaf cover cannot be assessed"
            )
            notes.append(f"Static tree-structure scan could not be evaluated: {why}.")
            details["scan"] = {"ran": False, "reason": why, "n_trees": scan.n_trees}
            checks.append(GateCheck.evaluate(
                "n_flagged_rules", None, "<=", 0,
                description=f"root->leaf rules with backdoor suspicion >= {m:g} (static tree scan)",
            ))
            return "The static tree scan could not be evaluated (no cover statistics in the trees)."
        rs = scan.rule_suspicion
        flagged_mask = rs >= m
        n_flagged = int(flagged_mask.sum())
        top_s = float(rs.max()) if rs.size else 0.0
        fs = scan.feature_suspicion
        top_f = int(np.argmax(fs)) if fs.size else -1
        flagged_feats = sorted({int(f) for f in scan.rule_dominant[flagged_mask]},
                               key=lambda f: (-fs[f], f))
        for r in scan.top_rules:
            r["flagged"] = bool(r["suspicion"] >= m)
        metrics.update(
            scan_ran=True,
            n_trees=scan.n_trees,
            n_trees_scanned=scan.n_trees_scanned,
            n_rules_scanned=scan.n_rules,
            n_candidate_rules=int(rs.size),
            n_flagged_rules=n_flagged,
            n_flagged_features=len(flagged_feats),
            top_suspicion=top_s,
            top_feature_suspicion=float(fs[top_f]) if top_f >= 0 else 0.0,
            top_suspicious_feature=names[top_f] if top_f >= 0 and fs[top_f] > 0 else None,
            scan_seconds=round(scan.seconds, 3),
        )
        feat_rows = self._feature_rows(scan, names, flagged_mask, limit=max(20, len(flagged_feats)))
        details["scan"] = {
            "ran": True,
            "importance_source": scan.importance_source,
            "flagged_features": [names[f] for f in flagged_feats],
            "top_rules": scan.top_rules,
            "features": feat_rows,
            "calibration": {
                "rho_gate": RHO_GATE, "rho_narrow": RHO_NARROW, "phi_high": PHI_HIGH, "phi_low": PHI_LOW,
                "share_low": SHARE_LOW, "share_high": SHARE_HIGH,
                "low_importance_quantile": prm["low_importance_quantile"], "min_suspicion": m,
            },
            "score_formula": "1.0 if no rule is flagged, else 0.75 * (1 - top_suspicion) / (1 - min_suspicion)",
        }
        if scan.importance_source == "cover":
            notes.append("The trees carry no split gains; node cover was used as the importance weight in the scan.")
        n_cat = int(te.meta.get("categorical_splits_expanded", 0) or 0)
        if n_cat:
            notes.append(
                f"{n_cat} categorical splits of this model were rewritten as threshold chains; the loader shares "
                "each categorical subtree's training cover evenly across its copies, so branch narrowness and leaf "
                "cover under those splits are approximate."
            )
            details["scan"]["categorical_splits_expanded"] = n_cat
        if str(te.meta.get("cover", "")) == "uniform":
            notes.append(
                "The tree ensemble carries no training cover (e.g. an ONNX export gives every leaf cover 1), so "
                "branch narrowness and leaf cover reflect leaf counts, not training samples; scan scores are "
                "less reliable for this model."
            )
            details["scan"]["uniform_cover"] = True
        chk = GateCheck.evaluate(
            "n_flagged_rules", n_flagged, "<=", 0,
            description=f"root->leaf rules with backdoor suspicion >= {m:g} (static tree scan)",
        )
        chk.score = scan_check_score(n_flagged, top_s, m)
        checks.append(chk)
        self._scan_artifacts(ctx, scan, feat_rows, m)
        if n_flagged:
            shown = ", ".join(names[f] for f in flagged_feats[:3]) + (" …" if len(flagged_feats) > 3 else "")
            return (
                f"The static tree scan flagged {n_flagged} of {scan.n_rules} root→leaf rules "
                f"(top suspicion {top_s:.2f} ≥ {m:.2f}) gating on {shown}: narrow, low-cover splits on otherwise "
                "low-importance features that route to strongly benign leaves — a common backdoor signature."
            )
        return (
            f"The static tree scan of {scan.n_trees} trees ({scan.n_rules} root→leaf rules) found no "
            f"backdoor-like rules (top suspicion {top_s:.2f} < {m:.2f})."
        )

    @staticmethod
    def _feature_rows(scan: TreeScan, names: Sequence[str], flagged_mask: np.ndarray, limit: int) -> list[dict[str, Any]]:
        fs = scan.feature_suspicion
        order = [int(f) for f in np.lexsort((np.arange(fs.size), -scan.gate_share, -fs)) if fs[f] > 0 or scan.gate_share[f] > 0]
        order = order[:limit]
        n_flag_by_feat = np.bincount(scan.rule_dominant[flagged_mask], minlength=fs.size) if flagged_mask.any() else np.zeros(fs.size, int)
        total = float(scan.total_gain.sum()) or 1.0
        rows = []
        for f in order:
            rows.append({
                "feature": names[f],
                "index": f,
                "suspicion": float(fs[f]),
                "low_importance": float(scan.low_importance[f]),
                "gate_share": float(scan.gate_share[f]),
                "importance_share": float(scan.total_gain[f] / total),
                "n_flagged_rules": int(n_flag_by_feat[f]),
            })
        return rows

    def _scan_artifacts(self, ctx: "RunContext", scan: TreeScan, feat_rows: list[dict[str, Any]], m: float) -> None:
        if feat_rows:
            top = feat_rows[:15]
            ctx.artifacts.add_chart(
                self.id, "feature_suspicion", kind="bar",
                title="Backdoor suspicion by feature (static tree scan)",
                series=[{"label": "feature suspicion", "x": [r["feature"] for r in top], "y": [r["suspicion"] for r in top]}],
                xlabel="feature", ylabel="suspicion (0..1)", ylim=(0.0, 1.0),
                reference_lines=[{"axis": "y", "value": m, "label": f"min_suspicion {m:g}"}],
            )
            ctx.artifacts.add_table(
                self.id, "suspicious_features", title="Features with backdoor-like gate structure",
                columns=["feature", "suspicion", "low importance", "gate gain share", "importance share", "flagged rules"],
                rows=[[r["feature"], round(r["suspicion"], 3), round(r["low_importance"], 3), round(r["gate_share"], 5),
                       round(r["importance_share"], 5), r["n_flagged_rules"]] for r in feat_rows],
            )
        if scan.top_rules:
            ctx.artifacts.add_table(
                self.id, "suspicious_rules", title="Most suspicious root→leaf rules (static tree scan)",
                columns=["rank", "suspicion", "flagged", "dominant feature", "conditions", "leaf value", "cover", "cover share", "tree", "leaf"],
                rows=[[r["rank"], round(r["suspicion"], 3), "yes" if r["suspicion"] >= m else "no", r["dominant_feature"],
                       " AND ".join(r["conditions"]), round(r["leaf_value"], 5), r["cover"], round(r["cover_fraction"], 5),
                       r["tree"], r["leaf"]] for r in scan.top_rules],
                note="Conditions are merged per feature; the leaf value is the tree's raw (margin) contribution.",
            )

    # ---- (a) --------------------------------------------------------------------------------------
    def _run_triggers(self, ctx: "RunContext", triggers: list[Trigger], prm: dict[str, Any], has_fs: bool,
                      checks: list[GateCheck], metrics: dict[str, Any], details: dict[str, Any],
                      notes: list[str]) -> str:
        t_max = prm["max_trigger_drop"]
        desc = "largest detection-rate drop caused by stamping an operator trigger onto detected malicious vectors"

        def unevaluated(why: str) -> str:
            notes.append(f"Trigger hypotheses were supplied but could not be tested: {why}.")
            details["triggers"] = {"ran": False, "reason": why, "names": [t.name for t in triggers]}
            checks.append(graded_check("max_trigger_drop", None, "<=", t_max, ideal=0.02, floor=0.8, description=desc))
            return f"The {len(triggers)} supplied trigger hypothesis(es) could not be tested ({why})."

        if not has_fs:
            return unevaluated(ctx.missing_reason(Requirement.FEATURE_SPACE))
        corpus = ctx.corpus
        assert corpus is not None
        idx_all = corpus.eval_indices(1, exclude_hashes=ctx.training_hashes)
        idx_u, n_dup = unique_by_hash(corpus, idx_all)
        idx = corpus.subsample(idx_u, prm["n_samples"], ctx.rng)
        if idx.size == 0:
            return unevaluated("the canonical corpus has no malicious eval rows outside the training manifest")
        dim = corpus.dim
        for tr in triggers:
            bad = [i for i in tr.features if i >= dim]
            if bad:
                raise ConfigError(f"{MODULE_ID}: trigger {tr.name!r} uses feature index {bad[0]} >= corpus dim {dim}")
        thr = ctx.threshold
        p0 = score_indices(ctx, idx)
        det = p0 >= thr
        n, n_det = int(idx.size), int(det.sum())
        dr_before = n_det / n
        X_det = corpus.take(idx[det])
        lo = X_det.min(axis=0) if n_det else np.zeros(dim)
        hi = X_det.max(axis=0) if n_det else np.zeros(dim)
        step = batch_rows(ctx)
        rows: list[dict[str, Any]] = []
        for tr in triggers:
            ctx.check_deadline()
            cols = np.fromiter(tr.features.keys(), dtype=np.int64)
            vals = np.fromiter(tr.features.values(), dtype=np.float64)
            still = 0
            shift_sum = 0.0
            for s in range(0, n_det, step):
                Xs = np.array(X_det[s : s + step], dtype=np.float32, copy=True)
                Xs[:, cols] = vals.astype(np.float32)
                pa = np.asarray(ctx.score(Xs), dtype=np.float64).reshape(-1)
                still += int((pa >= thr).sum())
                shift_sum += float((p0[det][s : s + step] - pa).sum())
            dr_after = still / n
            drop = dr_before - dr_after
            rows.append({
                "name": tr.name,
                "features": {tr.labels[i]: float(v) for i, v in tr.features.items()},
                "n_features": len(tr.features),
                "dr_before": dr_before,
                "dr_after": dr_after,
                "drop": drop,
                "evasion_rate": (n_det - still) / n_det if n_det else None,
                "mean_score_shift": shift_sum / n_det if n_det else None,
                "outside_observed_range": [tr.labels[i] for i, v in tr.features.items()
                                           if n_det and not (lo[i] <= v <= hi[i])],
                "flagged": bool(drop > t_max),
            })
        worst = max(rows, key=lambda r: (r["drop"], r["name"]))
        metrics.update(
            max_trigger_drop=worst["drop"],
            worst_trigger=worst["name"],
            n_flagged_triggers=int(sum(r["flagged"] for r in rows)),
            dr_before=dr_before,
            n_trigger_samples=n,
            n_trigger_detected=n_det,
        )
        details["triggers"] = {
            "ran": True,
            "results": rows,
            "n_malicious_eval": int(idx_all.size),
            "n_duplicates_removed": n_dup,
            "n_sampled": n,
            "subsampled": bool(idx_u.size > n),
            "operating_threshold": thr,
            "drop_definition": "DR_before - DR_after on the sampled malicious eval rows; the trigger is stamped "
                               "onto every detected row, undetected rows are unchanged",
        }
        if idx_u.size > n:
            notes.append(
                f"Trigger test used a seeded random sample of {n} of {idx_u.size} unique malicious eval rows "
                f"({MODULE_ID}.n_samples = {prm['n_samples']})."
            )
        if n_dup:
            notes.append(f"{n_dup} duplicate malicious eval rows (same sha256) were counted once.")
        oor = [r["name"] for r in rows if r["outside_observed_range"]]
        if oor:
            notes.append(
                "Trigger(s) " + ", ".join(repr(x) for x in oor) + " stamp values outside the range seen in the detected "
                "eval vectors; a drop there may reflect out-of-distribution inputs rather than a planted backdoor."
            )
        checks.append(graded_check("max_trigger_drop", worst["drop"], "<=", t_max, ideal=0.02, floor=0.8, description=desc))
        ctx.artifacts.add_table(
            self.id, "trigger_tests", title="Operator trigger hypotheses",
            columns=["trigger", "features stamped", "DR before", "DR after", "drop", "evasion of detected", "flagged"],
            rows=[[r["name"], ", ".join(f"{k}={v:g}" for k, v in r["features"].items()), round(r["dr_before"], 4),
                   round(r["dr_after"], 4), round(r["drop"], 4),
                   None if r["evasion_rate"] is None else round(r["evasion_rate"], 4), "yes" if r["flagged"] else "no"]
                  for r in rows],
            note=f"{n} malicious eval rows, {n_det} detected at threshold {thr:g}.",
        )
        if len(rows) > 1 or rows[0]["flagged"]:
            ctx.artifacts.add_chart(
                self.id, "trigger_drop", kind="bar", title="Detection-rate drop per trigger hypothesis",
                series=[{"label": "DR drop", "x": [r["name"] for r in rows], "y": [r["drop"] for r in rows]}],
                xlabel="trigger", ylabel="DR before − DR after", ylim=(0.0, 1.0),
                reference_lines=[{"axis": "y", "value": t_max, "label": f"max_trigger_drop {t_max:g}"}],
            )
        if worst["flagged"]:
            flagged = [r["name"] for r in rows if r["flagged"]]
            return (
                f"Trigger hypothesis {worst['name']!r} lowers detection by {_pp(worst['drop'])} "
                f"(DR {100 * dr_before:.1f}% → {100 * worst['dr_after']:.1f}%; limit {_pp(t_max)})"
                + (f"; {len(flagged)} of {len(rows)} triggers exceed the limit." if len(flagged) > 1 else ".")
            )
        return (
            f"None of the {len(rows)} trigger hypotheses lowers detection by more than {_pp(t_max)} "
            f"(worst: {worst['name']!r}, {_pp(worst['drop'])})."
        )
