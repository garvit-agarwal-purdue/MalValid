"""M7 — explanation & spurious-feature reliance (attribution only).

Question answered: *does the detector lean on features an attacker can set for free?* Real-world
bypasses of ML anti-virus engines appended benign strings or overlay bytes to malware and flipped
the verdict because the model had learned to trust those features. M7 measures where the model's
decision mass sits and cross-references it with the feature schema's per-feature controllability
(:meth:`FeatureSchema.feature_controllability`):

* **global importance** from the trees' split gain (tree access), and
* **mean |SHAP|** over eval rows — exact TreeSHAP (``shap.TreeExplainer`` on
  :meth:`TreeEnsemble.to_shap_model`, raw/log-odds space) when the trees are available, otherwise a
  model-agnostic permutation SHAP explainer on a small sample through ``ctx.score`` (probability
  space). Without a corpus (tree access only) the attribution falls back to split gain.

Metrics (on the attribution shares ``s_f = a_f / sum(a)``):

* ``controllable_share`` = ``sum_f w(c_f) s_f`` over CONTROLLABLE and APPEND_ONLY features with the
  weights of :data:`malvalid.schemas.base.CONTROLLABILITY_WEIGHT` (1.0 and 0.75);
* ``top_feature_share`` = ``max_f s_f``; ``effective_n_features`` = ``1 / sum_f s_f^2``
  (inverse Herfindahl index); ``top_k_share`` = mass of the ``top_k`` largest features;
* per-group and per-controllability-level shares, for SHAP and for gain.

Checks (warn gate): ``controllable_share <= max_controllable_share`` (anchors ideal 0.15, floor 0.9)
and ``top_feature_share <= max_top_feature_share`` (ideal 0.05, floor 0.6).

TreeSHAP cost grows with ensemble size (the EMBER2018 reference LightGBM — 1000 trees, ~1500 leaves
each — takes ~0.9 s per row single-threaded), so the number of explained rows is capped
deterministically from the ensemble's structure (see :func:`treeshap_cost`) to fit a fixed time
budget; the requested and used sample sizes are both recorded.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from malvalid.core import ConfigError, GateMode, Module, ModuleResult, Requirement
from malvalid.loaders.trees import LEAF
from malvalid.modules.performance import batch_rows, graded_check, unique_by_hash
from malvalid.schemas.base import CONTROLLABILITY_WEIGHT, Controllability

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.context import RunContext
    from malvalid.loaders.trees import TreeEnsemble

log = logging.getLogger("malvalid.modules.explanation")

MODULE_ID = "explanation"

# ---- TreeSHAP budget -------------------------------------------------------------------------------
SHAP_TIME_BUDGET_S = 600.0  # target wall time for TreeSHAP (well inside the 1800 s module limit)
SHAP_NS_PER_UNIT = 4.0  # ns per cost unit of treeshap_cost(); measured 2.1-3.1 ns on the build host
SHAP_MIN_SAMPLES = 20  # never explain fewer rows than this (unless fewer are available/requested)
SHAP_STOP_RESERVE_S = 90.0  # stop explaining more chunks when less than this is left of the module budget
INTERVENTIONAL_BACKGROUND = 50  # background rows for interventional TreeSHAP (trees without cover)

# ---- model-agnostic (permutation) SHAP budget --------------------------------------------------
AGNOSTIC_ROW_BUDGET = 4_000_000  # upper bound on model rows scored by the permutation explainer
AGNOSTIC_MAX_SAMPLES = 256
AGNOSTIC_MIN_SAMPLES = 8
AGNOSTIC_CHUNK = 4  # rows per explainer call; the deadline is checked between chunks

SOURCE_TREE_SHAP = "tree_shap"
SOURCE_TREE_SHAP_INTERVENTIONAL = "tree_shap_interventional"
SOURCE_AGNOSTIC = "permutation_shap"
SOURCE_GAIN = "tree_gain"

_SOURCE_LABEL = {
    SOURCE_TREE_SHAP: "mean |SHAP| (exact TreeSHAP)",
    SOURCE_TREE_SHAP_INTERVENTIONAL: "mean |SHAP| (interventional TreeSHAP)",
    SOURCE_AGNOSTIC: "mean |SHAP| (model-agnostic permutation SHAP)",
    SOURCE_GAIN: "split gain",
}

_ATTACKER_LEVELS = (Controllability.CONTROLLABLE, Controllability.APPEND_ONLY)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _node_depths(children_left: np.ndarray, children_right: np.ndarray) -> np.ndarray:
    """Depth of every reachable node (root = 0; unreachable nodes get 0)."""
    m = children_left.shape[0]
    depth = np.zeros(m, dtype=np.int64)
    frontier = np.array([0], dtype=np.int64)
    d = 0
    while frontier.size and d <= m:
        internal = frontier[children_left[frontier] != LEAF]
        if internal.size == 0:
            break
        d += 1
        frontier = np.concatenate([children_left[internal], children_right[internal]]).astype(np.int64)
        depth[frontier] = d
    return depth


def treeshap_cost(te: "TreeEnsemble") -> float:
    """Deterministic per-row cost proxy of exact TreeSHAP: ``sum_trees(sum_leaves(d^2 + d) + nodes)``.

    TreeSHAP extends the decision path at every node and unwinds it at every leaf, so its per-row
    work is ~``sum_leaves depth^2``; the proxy tracks measured wall time within ~1.5x across ensembles
    from 60 x 15 leaves to 1000 x 1500 leaves (2.1-3.1 ns per unit on the build host).
    """
    total = 0.0
    for t in te.trees:
        if t.n_nodes <= 1:
            total += 1.0
            continue
        dep = _node_depths(t.children_left, t.children_right).astype(np.float64)
        lv = t.children_left == LEAF
        total += float(np.sum(dep[lv] ** 2 + dep[lv])) + t.n_nodes
    return total


def _eval_sample(ctx: "RunContext", n_total: int, *, rng: np.random.Generator,
                 exclude: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """A class-balanced, seeded sample of unique eval rows (training members excluded)."""
    corpus = ctx.corpus
    assert corpus is not None
    ben, dup_b = unique_by_hash(corpus, corpus.eval_indices(0, exclude_hashes=ctx.training_hashes))
    mal, dup_m = unique_by_hash(corpus, corpus.eval_indices(1, exclude_hashes=ctx.training_hashes))
    if exclude is not None and exclude.size:
        ben = np.setdiff1d(ben, exclude)
        mal = np.setdiff1d(mal, exclude)
    n_mal = min(mal.size, (n_total + 1) // 2)
    n_ben = min(ben.size, n_total - n_mal)
    n_mal = min(mal.size, n_total - n_ben)
    idx = np.sort(np.concatenate([corpus.subsample(ben, n_ben, rng), corpus.subsample(mal, n_mal, rng)]))
    info = {"n_benign_available": int(ben.size), "n_malicious_available": int(mal.size),
            "n_benign": int(n_ben), "n_malicious": int(n_mal), "n_duplicates_removed": int(dup_b + dup_m)}
    return idx.astype(np.int64), info


def _class_interleaved(corpus: Any, idx: np.ndarray) -> np.ndarray:
    """``idx`` reordered benign, malicious, benign, ... so that any prefix stays class-balanced
    (matters only when explaining stops early on the time budget)."""
    lab = np.asarray(corpus.label)[idx]
    parts = [idx[lab == 0], idx[lab == 1]]
    n = max(p.size for p in parts) if idx.size else 0
    out = [p[i] for i in range(n) for p in parts if i < p.size]
    return np.asarray(out, dtype=np.int64)


def _shares(a: np.ndarray) -> np.ndarray:
    tot = float(a.sum())
    return a / tot if tot > 0 else np.zeros_like(a)


def concentration(shares: np.ndarray, top_k: int) -> dict[str, float]:
    """Top-feature share, top-k share and effective number of features (1 / Herfindahl)."""
    if shares.size == 0 or shares.sum() <= 0:
        return {"top_feature_share": math.nan, "top_k_share": math.nan, "effective_n_features": math.nan}
    srt = np.sort(shares)[::-1]
    return {
        "top_feature_share": float(srt[0]),
        "top_k_share": float(srt[:top_k].sum()),
        "effective_n_features": float(1.0 / np.sum(shares**2)),
    }


def controllable_share(shares: np.ndarray, levels: Sequence[Controllability]) -> float:
    """``sum w(c_f) s_f`` over CONTROLLABLE / APPEND_ONLY features (CONTROLLABILITY_WEIGHT)."""
    w = np.array([CONTROLLABILITY_WEIGHT[c] if c in _ATTACKER_LEVELS else 0.0 for c in levels])
    return float(np.sum(w * shares))


def controllability_reasons(schema: Any, dim: int) -> list[str] | None:
    """Per-feature justification of the controllability level, when the schema offers one
    (``controllability_reasons()``, e.g. ember_v2); ``None`` otherwise or if it is malformed."""
    fn = getattr(schema, "controllability_reasons", None)
    if not callable(fn):
        return None
    try:
        reasons = [str(r) for r in fn()]
    except Exception as e:  # optional schema extra: never fail the module over it
        log.info("schema.controllability_reasons() unavailable: %s", e)
        return None
    return reasons if len(reasons) == dim else None


def _int_param(params: dict[str, Any], key: str, lo: int) -> int:
    v = params.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < lo:
        raise ConfigError(f"{MODULE_ID}.{key} must be an integer >= {lo}, got {v!r}")
    return int(v)


def _share_param(params: dict[str, Any], key: str) -> float:
    v = params.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not (0.0 < float(v) <= 1.0):
        raise ConfigError(f"{MODULE_ID}.{key} must be a number in (0, 1], got {v!r}")
    return float(v)


def _pct(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100.0 * x:.1f}%"


# --------------------------------------------------------------------------------------------------
# Module
# --------------------------------------------------------------------------------------------------


class ExplanationModule(Module):
    id = MODULE_ID
    code = "M7"
    title = "Explanation & spurious-feature reliance"
    description = (
        "Global feature attribution (tree split gain and mean |SHAP| over eval rows) aggregated by feature "
        "group and by attacker controllability. Flags detectors whose decisions rest on features an attacker "
        "can set or append for free (strings, overlay/size, header fields) or on a single dominant feature — "
        "the failure mode behind real-world ML-AV bypasses."
    )
    requires = ()
    requires_any = (Requirement.TREE_ACCESS, Requirement.FEATURE_SPACE)
    default_gate = GateMode.WARN
    default_params = {
        "top_k": 20,
        "shap_samples": 2000,
        "max_controllable_share": 0.50,
        "max_top_feature_share": 0.30,
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        _int_param(params, "top_k", 1)
        _int_param(params, "shap_samples", 1)
        _share_param(params, "max_controllable_share")
        _share_param(params, "max_top_feature_share")

    def run(self, ctx: "RunContext") -> ModuleResult:
        top_k = _int_param(ctx.params, "top_k", 1)
        n_req = _int_param(ctx.params, "shap_samples", 1)
        max_ctrl = _share_param(ctx.params, "max_controllable_share")
        max_top = _share_param(ctx.params, "max_top_feature_share")

        schema = ctx.schema
        dim = int(schema.dim)
        names = list(schema.feature_names())
        levels = list(schema.feature_controllability())
        groups = schema.groups()
        gidx = schema.group_index()
        reasons = controllability_reasons(schema, dim)
        notes: list[str] = []
        details: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        te = ctx.model.tree_ensemble() if ctx.has(Requirement.TREE_ACCESS) else None
        has_fs = ctx.has(Requirement.FEATURE_SPACE) and ctx.corpus is not None

        # ---- tree gain ------------------------------------------------------------------------
        gain: np.ndarray | None = None
        if te is not None:
            gain = self._fit_dim(np.asarray(te.feature_importance("gain"), dtype=np.float64), dim, notes, "gain")
            if gain.sum() <= 0:
                gain = self._fit_dim(np.asarray(te.feature_importance("split"), dtype=np.float64), dim, notes, "split")
                notes.append("The trees carry no split gains; split counts are used as the tree importance.")

        # ---- SHAP attribution -----------------------------------------------------------------
        attribution: np.ndarray | None = None
        source = SOURCE_GAIN
        shap_info: dict[str, Any] = {"requested": n_req, "used": 0}
        if has_fs:
            try:
                if te is not None:
                    attribution, source, shap_info = self._tree_shap(ctx, te, n_req, dim, notes)
                else:
                    attribution, source, shap_info = self._agnostic_shap(ctx, n_req, dim, notes)
            except ImportError as e:  # shap missing: degrade honestly
                notes.append(f"SHAP is not importable ({e}); attribution falls back to tree split gain.")
                attribution = None
        elif te is not None:
            notes.append(
                f"SHAP needs eval rows but {ctx.missing_reason(Requirement.FEATURE_SPACE)}; attribution shares "
                "are computed from tree split gain only."
            )
        if attribution is None:
            if gain is None:
                return self.skip(
                    ctx,
                    "no attribution could be computed: model-agnostic SHAP needs eval rows from the canonical corpus "
                    f"and a working shap install, and tree attribution is unavailable "
                    f"({ctx.missing_reason(Requirement.TREE_ACCESS)})",
                    notes=notes,
                )
            attribution, source = gain, SOURCE_GAIN
        shap_info["source"] = source
        details["attribution"] = shap_info

        shares = _shares(attribution)
        gain_shares = _shares(gain) if gain is not None else None
        if shares.sum() <= 0:
            notes.append("All attributions are zero (the model output does not vary on the explained rows).")

        # ---- aggregate ------------------------------------------------------------------------------
        conc = concentration(shares, top_k)
        ctrl = controllable_share(shares, levels) if shares.sum() > 0 else math.nan
        lvl_arr = np.array([c.value for c in levels])
        by_level = {c.value: float(shares[lvl_arr == c.value].sum()) for c in Controllability}
        by_group = {g.name: float(shares[g.start : g.stop].sum()) for g in groups}
        order = [int(i) for i in np.lexsort((np.arange(dim), -shares))]
        k = min(top_k, dim)
        top = [i for i in order[:k] if shares[i] > 0]
        top_rows = [
            {
                "rank": r + 1,
                "feature": names[i],
                "index": i,
                "group": groups[int(gidx[i])].name,
                "controllability": levels[i].value,
                "attacker_controllable": levels[i] in _ATTACKER_LEVELS,
                "controllability_reason": None if reasons is None else reasons[i],
                "share": float(shares[i]),
                "gain_share": None if gain_shares is None else float(gain_shares[i]),
            }
            for r, i in enumerate(top)
        ]
        top_k_attacker = float(sum(row["share"] for row in top_rows if row["attacker_controllable"]))
        metrics.update(
            attribution_source=source,
            controllable_share=ctrl,
            top_feature_share=conc["top_feature_share"],
            top_feature=names[order[0]] if shares.sum() > 0 else None,
            top_feature_controllability=levels[order[0]].value if shares.sum() > 0 else None,
            effective_n_features=conc["effective_n_features"],
            top_k=k,
            top_k_share=conc["top_k_share"],
            top_k_attacker_controllable_share=top_k_attacker,
            n_top_k_attacker_controllable=int(sum(r["attacker_controllable"] for r in top_rows)),
            n_features_used=int(np.count_nonzero(shares)),
            shap_samples_requested=n_req,
            shap_samples_used=int(shap_info.get("used", 0)),
            **{f"share_{lv}": v for lv, v in by_level.items()},
        )
        details["top_features"] = top_rows
        details["by_group"] = by_group
        details["by_controllability"] = by_level
        details["controllability_weights"] = {c.value: CONTROLLABILITY_WEIGHT[c] for c in Controllability}
        if gain_shares is not None:
            gconc = concentration(gain_shares, top_k)
            metrics.update(
                gain_controllable_share=controllable_share(gain_shares, levels) if gain_shares.sum() > 0 else math.nan,
                gain_top_feature_share=gconc["top_feature_share"],
                gain_top_feature=names[int(np.argmax(gain_shares))] if gain_shares.sum() > 0 else None,
            )
            details["by_group_gain"] = {g.name: float(gain_shares[g.start : g.stop].sum()) for g in groups}
            details["by_controllability_gain"] = {
                c.value: float(gain_shares[lvl_arr == c.value].sum()) for c in Controllability
            }

        # ---- checks ---------------------------------------------------------------------------------
        checks = [
            graded_check(
                "controllable_share", ctrl, "<=", max_ctrl, ideal=0.15, floor=0.9,
                description="controllability-weighted share of attribution on CONTROLLABLE / APPEND_ONLY features",
            ),
            graded_check(
                "top_feature_share", conc["top_feature_share"], "<=", max_top, ideal=0.05, floor=0.6,
                description="share of attribution carried by the single most important feature",
            ),
        ]
        self._artifacts(ctx, top_rows, groups, by_group, details.get("by_group_gain"), source)
        finding = self._finding(ctrl, conc, metrics, max_ctrl, max_top, source, top_rows)
        if source == SOURCE_GAIN:
            notes.append("Split gain measures how much the trees used a feature while training, not how much it moves "
                         "individual decisions; treat the shares as indicative.")
        notes.append(
            "Attribution only: M7 shows which features the decisions rest on and whether an attacker could set "
            "them; it does not test whether changing them actually evades the model."
        )
        return self.result(ctx, finding=finding, checks=checks, metrics=metrics, details=details, notes=notes)

    # ---- attribution back-ends -------------------------------------------------------------------
    @staticmethod
    def _fit_dim(v: np.ndarray, dim: int, notes: list[str], what: str) -> np.ndarray:
        if v.size == dim:
            return v
        if v.size < dim:
            return np.concatenate([v, np.zeros(dim - v.size)])
        notes.append(f"The tree ensemble has {v.size} features but the schema {dim}; extra {what} entries ignored.")
        return v[:dim]

    def _tree_shap(self, ctx: "RunContext", te: "TreeEnsemble", n_req: int, dim: int,
                   notes: list[str]) -> tuple[np.ndarray | None, str, dict[str, Any]]:
        import shap

        corpus = ctx.corpus
        assert corpus is not None
        uniform_cover = str(te.meta.get("cover", "")) == "uniform"
        cost = treeshap_cost(te)
        n_bg = INTERVENTIONAL_BACKGROUND if uniform_cover else 0
        unit_cost = cost * max(n_bg, 1) / (10.0 if uniform_cover else 1.0)  # interventional ~ n_bg/10 x
        est_row_s = unit_cost * SHAP_NS_PER_UNIT * 1e-9
        budget_s = min(SHAP_TIME_BUDGET_S, 0.4 * ctx.time_left())
        affordable = int(budget_s / est_row_s) if est_row_s > 0 else n_req
        n_target = min(n_req, max(affordable, SHAP_MIN_SAMPLES))
        idx, sample_info = _eval_sample(ctx, n_target, rng=ctx.rng)
        info: dict[str, Any] = {
            "requested": n_req,
            "budget_cap": affordable,
            "estimated_seconds_per_row": est_row_s,
            "cost_units_per_row": cost,
            "time_budget_s": budget_s,
            "sample": sample_info,
            "output_space": "raw margin (log-odds)" if te.output_transform == "logistic" else "raw value",
        }
        if idx.size == 0:
            notes.append("The canonical corpus has no eval rows outside the training manifest; attribution falls back "
                         "to tree split gain.")
            info["used"] = 0
            return None, SOURCE_GAIN, info
        n_cat = int(te.meta.get("categorical_splits_expanded", 0) or 0)
        if n_cat:
            # The loader's threshold chains split each categorical subtree's cover evenly over its copies;
            # the cover-weighted expectations TreeSHAP uses are unchanged by that, so values stay exact.
            info["categorical_splits_expanded"] = n_cat
        model = te.to_shap_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if uniform_cover:
                bg_idx, _ = _eval_sample(ctx, n_bg, rng=ctx.rng, exclude=idx)
                if bg_idx.size == 0:
                    bg_idx = idx[: min(idx.size, n_bg)]
                bg = corpus.take(bg_idx).astype(np.float64)[:, : te.n_features]
                explainer = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
                source = SOURCE_TREE_SHAP_INTERVENTIONAL
                info["background_rows"] = int(bg.shape[0])
                notes.append(
                    "The tree ensemble carries no training cover (e.g. ONNX), so interventional TreeSHAP with "
                    f"{bg.shape[0]} background eval rows was used instead of path-dependent TreeSHAP."
                )
            else:
                explainer = shap.TreeExplainer(model)
                source = SOURCE_TREE_SHAP
        idx = _class_interleaved(corpus, idx)
        chunk = int(max(1, min(256, 30.0 / max(est_row_s, 1e-9))))
        acc = np.zeros(te.n_features, dtype=np.float64)
        done = 0
        stopped_early = False
        t0 = time.monotonic()
        for s in range(0, idx.size, chunk):
            ctx.check_deadline()
            if done >= SHAP_MIN_SAMPLES and ctx.time_left() < SHAP_STOP_RESERVE_S:
                stopped_early = True
                break
            X = corpus.take(idx[s : s + chunk]).astype(np.float64)[:, : te.n_features]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = explainer.shap_values(X, check_additivity=False)
            sv = np.asarray(sv[-1] if isinstance(sv, list) else sv, dtype=np.float64)
            if sv.ndim == 3:  # (n, d, outputs)
                sv = sv[..., -1]
            acc += np.abs(sv).sum(axis=0)
            done += X.shape[0]
        info["seconds"] = round(time.monotonic() - t0, 3)
        info["used"] = done
        if done < idx.size:
            info["stopped_early"] = stopped_early
            notes.append(f"TreeSHAP stopped after {done} of {idx.size} rows to stay within the module time budget.")
        if n_target < n_req:
            notes.append(
                f"TreeSHAP explained {done} eval rows instead of shap_samples = {n_req}: this ensemble costs "
                f"~{est_row_s:.2f} s per row, and M7 caps TreeSHAP at ~{budget_s:.0f} s."
            )
        elif done < n_req:
            notes.append(f"Only {done} unique eval rows were available for TreeSHAP (shap_samples = {n_req}).")
        log.info("TreeSHAP on %d rows took %.1fs (source=%s)", done, info["seconds"], source)
        return self._fit_dim(acc / max(done, 1), dim, notes, "SHAP"), source, info

    def _agnostic_shap(self, ctx: "RunContext", n_req: int, dim: int,
                       notes: list[str]) -> tuple[np.ndarray | None, str, dict[str, Any]]:
        import shap

        corpus = ctx.corpus
        assert corpus is not None
        n_bg = 32 if dim <= 512 else 16
        per_row = (2 * dim + 1) * n_bg
        n_target = int(min(n_req, AGNOSTIC_MAX_SAMPLES, max(AGNOSTIC_MIN_SAMPLES, AGNOSTIC_ROW_BUDGET // per_row)))
        idx, sample_info = _eval_sample(ctx, n_target, rng=ctx.rng)
        bg_idx, _ = _eval_sample(ctx, n_bg, rng=ctx.rng, exclude=idx)
        if bg_idx.size == 0:
            bg_idx = idx[:n_bg]
        info: dict[str, Any] = {
            "requested": n_req,
            "budget_cap": n_target,
            "background_rows": int(bg_idx.size),
            "max_model_rows": int(per_row * n_target),
            "sample": sample_info,
            "output_space": "probability (predict_proba)",
            "explainer": "shap.PermutationExplainer (one antithetic permutation per row, Independent masker)",
        }
        if idx.size == 0 or bg_idx.size == 0:
            notes.append("The canonical corpus has no eval rows outside the training manifest; no attribution possible.")
            info["used"] = 0
            return None, SOURCE_AGNOSTIC, info
        idx = _class_interleaved(corpus, idx)
        X = corpus.take(idx).astype(np.float64)
        bg = corpus.take(bg_idx).astype(np.float64)
        step = batch_rows(ctx)
        q0 = ctx.model.query_count

        def f(Z: np.ndarray) -> np.ndarray:
            Z = np.asarray(Z)
            out = np.empty(Z.shape[0], dtype=np.float64)
            for s in range(0, Z.shape[0], step):
                out[s : s + step] = np.asarray(ctx.score(Z[s : s + step]), dtype=np.float64).reshape(-1)
            return out

        t0 = time.monotonic()
        parts: list[np.ndarray] = []
        done = 0
        stopped_early = False
        last_chunk_s = 0.0
        state = np.random.get_state()  # the shap explainer seeds numpy's global RNG once, at construction
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                masker = shap.maskers.Independent(bg, max_samples=bg.shape[0])
                explainer = shap.PermutationExplainer(f, masker, seed=int(ctx.seed % (2**32)))
                # Chunks draw the same permutation sequence as one call; stopping early (a slow model
                # near the module deadline) only shortens the class-interleaved sample.
                for s in range(0, X.shape[0], AGNOSTIC_CHUNK):
                    ctx.check_deadline()
                    if done and ctx.time_left() < SHAP_STOP_RESERVE_S + 1.5 * last_chunk_s:
                        stopped_early = True
                        break
                    tc = time.monotonic()
                    exp = explainer(X[s : s + AGNOSTIC_CHUNK], max_evals=2 * dim + 1, silent=True)
                    v = np.asarray(exp.values, dtype=np.float64)
                    parts.append(v[..., -1] if v.ndim == 3 else v)
                    done += v.shape[0]
                    last_chunk_s = time.monotonic() - tc
        finally:
            np.random.set_state(state)
        sv = np.concatenate(parts, axis=0) if parts else np.zeros((0, dim))
        info["used"] = int(sv.shape[0])
        info["seconds"] = round(time.monotonic() - t0, 3)
        info["model_rows_scored"] = int(ctx.model.query_count - q0)
        if stopped_early:
            info["stopped_early"] = True
            notes.append(
                f"Model-agnostic SHAP stopped after {done} of {idx.size} rows to stay within the module time budget "
                "(the model scores slowly); the shares are noisier than usual."
            )
        notes.append(
            f"Tree access is unavailable ({ctx.missing_reason(Requirement.TREE_ACCESS)}), so attribution uses a "
            f"model-agnostic permutation SHAP explainer "
            f"on {sv.shape[0]} eval rows (background {bg.shape[0]} rows; shap_samples = {n_req} is capped at "
            f"{n_target} for this {dim}-feature schema to bound the number of model queries)."
        )
        log.info("permutation SHAP on %d rows took %.1fs (%d model rows)", done, info["seconds"], info["model_rows_scored"])
        return np.abs(sv).mean(axis=0), SOURCE_AGNOSTIC, info

    # ---- report ----------------------------------------------------------------------------------
    def _artifacts(self, ctx: "RunContext", top_rows: list[dict[str, Any]], groups: Sequence[Any],
                   by_group: dict[str, float], by_group_gain: dict[str, float] | None, source: str) -> None:
        label = _SOURCE_LABEL[source]
        if top_rows:
            series = [{"label": f"{label} share", "x": [r["feature"] for r in top_rows], "y": [r["share"] for r in top_rows]}]
            if source != SOURCE_GAIN and top_rows[0]["gain_share"] is not None:
                series.append({"label": "split-gain share", "x": [r["feature"] for r in top_rows],
                               "y": [r["gain_share"] for r in top_rows]})
            ctx.artifacts.add_chart(
                self.id, "top_features", kind="bar", title=f"Top {len(top_rows)} features by attribution",
                series=series, xlabel="feature", ylabel="share of attribution",
            )
            with_reason = any(r["controllability_reason"] for r in top_rows)
            columns = ["rank", "feature", "group", "controllability", "share", "gain share"]
            if with_reason:
                columns.append("why (schema)")
            rows = []
            for r in top_rows:
                row = [r["rank"], r["feature"], r["group"], r["controllability"], round(r["share"], 4),
                       None if r["gain_share"] is None else round(r["gain_share"], 4)]
                if with_reason:
                    row.append(r["controllability_reason"] or "")
                rows.append(row)
            ctx.artifacts.add_table(
                self.id, "top_features_table", title="Top features vs attacker controllability",
                columns=columns, rows=rows,
                note="controllability: controllable = settable for free; append_only = can be added to (strings, "
                     "imports, size); derived = moves as a side effect; fixed = tied to program semantics.",
            )
        gnames = [g.name for g in groups]
        gseries = [{"label": f"{label} share", "x": gnames, "y": [by_group[g] for g in gnames]}]
        if by_group_gain is not None and source != SOURCE_GAIN:
            gseries.append({"label": "split-gain share", "x": gnames, "y": [by_group_gain[g] for g in gnames]})
        ctx.artifacts.add_chart(
            self.id, "group_importance", kind="bar", title="Attribution by feature group",
            series=gseries, xlabel="feature group", ylabel="share of attribution", ylim=(0.0, 1.0),
        )

    @staticmethod
    def _finding(ctrl: float, conc: dict[str, float], metrics: dict[str, Any], max_ctrl: float, max_top: float,
                 source: str, top_rows: list[dict[str, Any]]) -> str:
        if not top_rows:
            return "No attribution mass: the model's output does not vary on the explained rows, so reliance could not be assessed."
        top = top_rows[0]
        ctrl_bad = not math.isnan(ctrl) and ctrl > max_ctrl
        top_bad = conc["top_feature_share"] > max_top
        s1 = (
            f"{_pct(ctrl)} of the model's attribution ({_SOURCE_LABEL[source]}, controllability-weighted) sits on "
            f"attacker-controllable features (limit {_pct(max_ctrl)})"
            + (" — the model leans on features an attacker can set or append for free." if ctrl_bad else ".")
        )
        s2 = (
            f"The largest single feature, {top['feature']} ({top['controllability']}), carries "
            f"{_pct(conc['top_feature_share'])} (limit {_pct(max_top)})"
            + (" — one feature dominates the decision." if top_bad else ".")
        )
        s3 = (
            f"The top {metrics['top_k']} features hold {_pct(conc['top_k_share'])} of the attribution "
            f"({metrics['n_top_k_attacker_controllable']} of them attacker-controllable); effective number of "
            f"features {conc['effective_n_features']:.1f}."
        )
        return " ".join([s1, s2, s3])
