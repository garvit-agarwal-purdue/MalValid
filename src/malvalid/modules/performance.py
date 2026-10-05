"""M1 — Performance & calibration at the declared operating threshold (hard gate).

The one axis that is hard by default: a detector with a high false-positive rate is unshippable
regardless of anything else. M1 scores the canonical corpus's held-out ``eval`` role (benign and
malicious), excluding any sample listed in the researcher's training manifest, and reports:

* detection rate (DR) and false-positive rate (FPR) at ``operating_threshold`` with 95% Wilson
  intervals — the two hard-gate conditions;
* threshold-free quality: AUROC, AUPRC (average precision), TPR at FPR 0.1% / 1%, and the
  threshold that would achieve ``max_fpr`` on the benign set;
* calibration: Brier score and expected calibration error (ECE, equal-width bins);
* detection rate on the corpus's ``challenge`` role (historically evasive malware), informational;
* when malvalid auto-calibrated the threshold (``--calibrate-fpr``), a bootstrap that re-draws the
  held-out calibration rows (re-fitting the threshold with the same rule) and the eval rows, so
  the FPR / detection-rate intervals include the threshold's own sampling noise (informational).

A sample is predicted malicious iff ``predict_proba(x) >= operating_threshold`` (the submission
contract requires ``predict`` to agree; M1 spot-checks that and notes any disagreement).

Helpers :func:`score_indices`, :func:`graded_check` and :func:`wilson_interval` are shared with
M2 (:mod:`malvalid.modules.drift`).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from malvalid.core import ConfigError, GateCheck, GateMode, Module, ModuleResult, Requirement

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.context import RunContext

log = logging.getLogger("malvalid.modules.performance")

Z95 = 1.959963984540054  # two-sided 95% normal quantile
DEFAULT_BATCH_ROWS = 20000
_PREDICT_SPOT_CHECK_ROWS = 1000
_ROC_POINTS = 300
_HIST_BINS = 50
DEFAULT_THRESHOLD_BOOTSTRAP = 1000


# --------------------------------------------------------------------------------------------------
# Shared helpers (also used by M2)
# --------------------------------------------------------------------------------------------------


def batch_rows(ctx: "RunContext") -> int:
    """Rows per scoring request: the runtime's ``chunk_rows`` (bounded memory per batch)."""
    runtime = getattr(ctx.config, "runtime", None)
    n = getattr(runtime, "chunk_rows", None) or DEFAULT_BATCH_ROWS
    return max(1, int(n))


def score_indices(ctx: "RunContext", idx: np.ndarray, *, batch: int | None = None) -> np.ndarray:
    """Score corpus rows ``idx`` through ``ctx.score`` in bounded batches (never row by row).

    Returns float64 scores aligned with ``idx``. Each batch is materialized from the (memmapped)
    corpus only while it is being scored, so memory stays at ``batch x dim`` floats.
    """
    corpus = ctx.corpus
    if corpus is None:
        raise ValueError("score_indices needs a loaded corpus")
    idx = np.asarray(idx, dtype=np.int64)
    out = np.empty(idx.size, dtype=np.float64)
    step = int(batch or batch_rows(ctx))
    for start in range(0, idx.size, step):
        ctx.check_deadline()
        sl = idx[start : start + step]
        p = np.asarray(ctx.score(corpus.take(sl)), dtype=np.float64).reshape(-1)
        if p.shape[0] != sl.size:
            raise ValueError(f"model returned {p.shape[0]} scores for {sl.size} rows")
        bad = ~np.isfinite(p)
        if bad.any():
            # the sandbox enforces this too; in-process handles may not
            raise ValueError(
                f"predict_proba returned {int(bad.sum())} non-finite score(s) (NaN/inf) for canonical "
                f"corpus rows (first at corpus row {int(sl[np.flatnonzero(bad)[0]])}); scores must be "
                "finite probabilities in [0, 1]"
            )
        out[start : start + sl.size] = p
    return out


def wilson_interval(k: int, n: int, z: float = Z95) -> list[float] | None:
    """Wilson score interval ``[lo, hi]`` for a binomial proportion ``k / n`` (None if ``n == 0``)."""
    if n <= 0:
        return None
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return [float(lo), float(hi)]


def unique_by_hash(corpus: Any, idx: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep one row per sha256 among ``idx`` (the first, in corpus order). Returns ``(idx, n_removed)``.

    A file that occurs several times in a corpus would otherwise be counted several times, which
    biases the metrics toward it and makes confidence intervals too narrow.
    """
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size < 2:
        return idx, 0
    _, first = np.unique(corpus.sha256[idx], return_index=True)
    if first.size == idx.size:
        return idx, 0
    keep = np.sort(idx[first])
    return keep, int(idx.size - keep.size)


def _safe_anchors(op: str, threshold: float, ideal: float, floor: float, scale: str) -> tuple[float, float]:
    """Nudge score anchors so that ``floor < threshold < ideal`` holds on the good-side axis.

    The anchor formulas in the build contract degenerate at extreme thresholds (e.g.
    ``min_detection = 1.0`` gives ``ideal == threshold``); grading must still work there.
    """
    t = float(threshold)
    higher_better = op in (">=", ">")
    if scale == "log":
        # all anchors must be positive in log space
        t = max(t, 1e-12)
        if higher_better:
            if not ideal > t:
                ideal = t * 10.0
            if not 0 < floor < t:
                floor = t / 10.0
        else:
            if not 0 < ideal < t:
                ideal = t / 10.0
            if not floor > t:
                floor = t * 10.0
        return float(ideal), float(floor)
    gap = max(abs(t) * 1e-3, 1e-6)
    if higher_better:
        if not ideal > t:
            ideal = t + gap
        if not floor < t:
            floor = t - gap
    else:
        if not ideal < t:
            ideal = t - gap
        if not floor > t:
            floor = t + gap
    return float(ideal), float(floor)


def graded_check(
    name: str,
    value: float | int | None,
    op: str,
    threshold: float,
    *,
    ideal: float,
    floor: float,
    scale: str = "linear",
    metric: str | None = None,
    description: str = "",
) -> GateCheck:
    """:meth:`GateCheck.evaluate` with anchors made valid for any configured threshold."""
    ideal, floor = _safe_anchors(op, threshold, ideal, floor, scale)
    return GateCheck.evaluate(
        name, value, op, threshold, metric=metric, description=description, ideal=ideal, floor=floor, scale=scale
    )


def _param_float(params: dict[str, Any], key: str, module: str, lo: float, hi: float, *,
                 lo_open: bool = False, hi_open: bool = False) -> float:
    v = params.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ConfigError(f"{module}.{key} must be a number, got {v!r}")
    f = float(v)
    ok_lo = f > lo if lo_open else f >= lo
    ok_hi = f < hi if hi_open else f <= hi
    if not (ok_lo and ok_hi):
        rng = f"{'(' if lo_open else '['}{lo:g}, {hi:g}{')' if hi_open else ']'}"
        raise ConfigError(f"{module}.{key} must be in {rng}, got {v!r}")
    return f


def _param_int(params: dict[str, Any], key: str, module: str, lo: int, *, allow_none: bool = False) -> int | None:
    v = params.get(key)
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or int(v) < lo:
        extra = " or null" if allow_none else ""
        raise ConfigError(f"{module}.{key} must be an integer >= {lo}{extra}, got {v!r}")
    return int(v)


def _param_bool(params: dict[str, Any], key: str, module: str) -> bool:
    v = params.get(key)
    if not isinstance(v, (bool, np.bool_)):
        raise ConfigError(f"{module}.{key} must be true or false, got {v!r}")
    return bool(v)


def _pct(x: float | None, digits: int = 2) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f}%"


def _fmt_t(t: float | None) -> str:
    return "n/a" if t is None else f"{t:.4g}"


# --------------------------------------------------------------------------------------------------
# Pure metric functions (vectorized; unit-tested against scikit-learn)
# --------------------------------------------------------------------------------------------------


def roc_arrays(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full ROC (no intermediate points dropped): ``fpr, tpr, thresholds`` with ``thresholds[0] = inf``.

    Classification rule is ``score >= threshold``.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    return fpr, tpr, thr


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target: float | np.ndarray) -> Any:
    """Highest TPR achievable by a threshold whose FPR is ``<= target`` (vectorized over target)."""
    i = np.searchsorted(fpr, target, side="right") - 1
    return tpr[np.clip(i, 0, tpr.size - 1)]


def threshold_for_fpr(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray, max_fpr: float) -> tuple[float | None, float, float]:
    """Lowest threshold whose benign FPR is ``<= max_fpr``: ``(threshold, fpr, tpr)`` at it.

    ``threshold`` is None when only rejecting everything satisfies ``max_fpr`` (e.g. more than
    ``max_fpr`` of benign samples share the maximum score).
    """
    i = int(np.searchsorted(fpr, max_fpr, side="right") - 1)
    i = max(i, 0)
    t = float(thr[i])
    return (t if math.isfinite(t) else None), float(fpr[i]), float(tpr[i])


def expected_calibration_error(y: np.ndarray, s: np.ndarray, bins: int) -> tuple[float, dict[str, Any]]:
    """ECE with ``bins`` equal-width bins on [0, 1] (same binning as sklearn's ``calibration_curve``).

    ``ECE = sum_b (n_b / N) * |mean(y in b) - mean(s in b)|``. Returns the ECE and the per-bin
    table used for the reliability diagram.
    """
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ids = np.searchsorted(edges[1:-1], s)
    cnt = np.bincount(ids, minlength=bins).astype(np.float64)
    sum_s = np.bincount(ids, weights=s, minlength=bins)
    sum_y = np.bincount(ids, weights=y, minlength=bins)
    n = float(s.size)
    ece = float(np.sum(np.abs(sum_y - sum_s)) / n) if n else float("nan")
    nz = cnt > 0
    mean_pred = np.where(nz, sum_s / np.maximum(cnt, 1), np.nan)
    frac_pos = np.where(nz, sum_y / np.maximum(cnt, 1), np.nan)
    table = {
        "bins": int(bins),
        "edges": edges.tolist(),
        "count": cnt.astype(np.int64).tolist(),
        "mean_predicted": mean_pred.tolist(),
        "fraction_malicious": frac_pos.tolist(),
    }
    return ece, table


def _ci95(v: np.ndarray) -> list[float]:
    lo, hi = np.quantile(v, [0.025, 0.975])
    return [float(lo), float(hi)]


def threshold_bootstrap(
    cal_scores: np.ndarray,
    target_fpr: float,
    s_ben: np.ndarray,
    s_mal: np.ndarray,
    *,
    max_fpr: float,
    min_dr: float,
    resamples: int,
    rng: np.random.Generator,
    check_deadline: Any = None,
) -> dict[str, Any]:
    """Bootstrap FPR / detection rate including the uncertainty of an auto-calibrated threshold.

    Each replicate (1) re-draws the benign calibration scores with replacement and re-fits the
    threshold with :func:`malvalid.submission.calibrate_threshold` (the rule the runner used), then
    (2) re-draws the benign and the malicious eval rows with replacement and counts the rows at or
    above that threshold. Step (2) is done exactly without materialising the resamples: for a fixed
    threshold, the number of rows of an ``n``-row with-replacement resample that score ``>= t`` is
    ``Binomial(n, q(t))``, where ``q(t)`` is the fraction of the original rows scoring ``>= t``
    (found by ``searchsorted`` on the sorted scores). Cost: ``resamples`` sorts of the calibration
    slice plus O(resamples) for the eval sets, independent of the eval-set size.

    Returns percentile 95% intervals of the threshold, FPR and detection rate, and the fraction of
    replicates in which each M1 gate condition (``fpr <= max_fpr``, ``detection_rate >= min_dr``)
    holds. A rate is ``None`` when its eval class is empty.
    """
    from malvalid.submission import CalibrationError, calibrate_threshold

    cal = np.asarray(cal_scores, dtype=np.float64).reshape(-1)
    n_c, B = int(cal.size), int(resamples)
    if n_c == 0 or B < 1:
        raise ValueError("threshold_bootstrap needs calibration scores and resamples >= 1")
    thr = np.empty(B, dtype=np.float64)
    above_one = float(np.nextafter(1.0, np.inf))  # an unachievable target flags nothing
    for b in range(B):
        if check_deadline is not None and b % 100 == 0:
            check_deadline()
        try:
            thr[b] = calibrate_threshold(cal[rng.integers(0, n_c, size=n_c)], target_fpr)[0]
        except CalibrationError:
            thr[b] = above_one

    def rate(scores: np.ndarray) -> np.ndarray | None:
        x = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
        n = x.size
        if n == 0:
            return None
        q = (n - np.searchsorted(x, thr, side="left")) / n  # fraction of the eval rows with score >= thr
        return rng.binomial(n, q) / n

    fpr_b = rate(s_ben)
    dr_b = rate(s_mal)
    fpr_ok = None if fpr_b is None else fpr_b <= max_fpr
    dr_ok = None if dr_b is None else dr_b >= min_dr
    both = None if fpr_ok is None or dr_ok is None else float(np.mean(fpr_ok & dr_ok))
    return {
        "resamples": B,
        "n_calibration": n_c,
        "target_fpr": float(target_fpr),
        "threshold_ci95": _ci95(thr),
        "fpr_ci95": None if fpr_b is None else _ci95(fpr_b),
        "detection_rate_ci95": None if dr_b is None else _ci95(dr_b),
        "pass_fraction": {
            "fpr": None if fpr_ok is None else float(np.mean(fpr_ok)),
            "detection_rate": None if dr_ok is None else float(np.mean(dr_ok)),
            "both": both,
        },
    }


def calibration_policy_note(cal: Any) -> tuple[dict[str, Any], str] | None:
    """``(details, note)`` describing which benign rows an auto-calibrated threshold was fit on
    (``RunContext.extras["threshold_calibration"]``), or None when the threshold was declared."""
    if not isinstance(cal, dict) or not cal.get("policy"):
        return None
    period = cal.get("period") or None
    after = cal.get("scored_benign_after_period")
    info = {"policy": cal["policy"], "period": period, "scored_benign_after_period": after}
    when = f" ({period[0]}..{period[1]})" if period else ""
    if cal["policy"] == "earliest":
        note = (f"Held threshold: it was auto-calibrated on the earliest benign rows of the evaluation period{when} "
                "and every test scores other rows only")
        if after is not None:
            note += f" ({after:.0%} of the scored benign rows are dated after that period)"
        note += (", so FPR here includes any drift of the benign data after calibration, as in deployment "
                 "(runtime.calibration_period: earliest).")
    else:
        note = (f"Matched threshold: it was auto-calibrated on benign rows sampled across the whole evaluation "
                f"period{when}, the same period it is scored on, so FPR here is close to the target by "
                "construction and hides drift after deployment (runtime.calibration_period: uniform; "
                "'earliest' fits it before the scored period instead).")
    return info, note


# --------------------------------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------------------------------


class PerformanceModule(Module):
    id = "performance"
    code = "M1"
    title = "Performance & calibration"
    description = (
        "Scores the canonical held-out benign and malicious corpus (excluding your training samples) "
        "at your declared operating threshold: false-positive rate and detection rate (the hard "
        "gate), plus ROC/PR curves, AUROC/AUPRC, TPR at low FPR, the threshold that would meet "
        "max_fpr, and calibration (Brier score, ECE, reliability diagram)."
    )
    requires = (Requirement.FEATURE_SPACE,)
    enhanced_by = (Requirement.TRAINING_HASHES,)
    default_gate = GateMode.HARD
    default_params = {
        "max_fpr": 0.01,
        "min_detection": 0.95,
        "max_samples_per_class": None,
        "calibration_bins": 15,
        "include_challenge": True,
        "threshold_bootstrap_resamples": DEFAULT_THRESHOLD_BOOTSTRAP,
    }

    # ---- parameters ------------------------------------------------------------------------

    @classmethod
    def _parse_params(cls, p: dict[str, Any]) -> dict[str, Any]:
        return {
            "max_fpr": _param_float(p, "max_fpr", cls.id, 0.0, 1.0, lo_open=True, hi_open=True),
            "min_detection": _param_float(p, "min_detection", cls.id, 0.0, 1.0),
            "max_samples_per_class": _param_int(p, "max_samples_per_class", cls.id, 1, allow_none=True),
            "calibration_bins": _param_int(p, "calibration_bins", cls.id, 2),
            "include_challenge": _param_bool(p, "include_challenge", cls.id),
            # optional key (added later): params saved by older configs/jobs may lack it
            "threshold_bootstrap_resamples": _param_int({**cls.default_params, **p}, "threshold_bootstrap_resamples",
                                                        cls.id, 0),
        }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        cls._parse_params(params)

    def _params(self, ctx: "RunContext") -> dict[str, Any]:
        return self._parse_params(ctx.params)

    # ---- run -------------------------------------------------------------------------------

    def run(self, ctx: "RunContext") -> ModuleResult:
        if ctx.corpus is None or not ctx.has(Requirement.FEATURE_SPACE):
            return self.skip(ctx, ctx.missing_reason(Requirement.FEATURE_SPACE))
        prm = self._params(ctx)
        corpus = ctx.corpus
        hashes = ctx.training_hashes or None
        t = ctx.threshold
        max_fpr, min_dr = prm["max_fpr"], prm["min_detection"]
        notes: list[str] = []
        details: dict[str, Any] = {"operating_threshold": t}

        # ---- eval sets: canonical held-out rows minus training members ---------------------
        ben_all = corpus.eval_indices(0)
        mal_all = corpus.eval_indices(1)
        ben_avail = corpus.eval_indices(0, exclude_hashes=hashes) if hashes else ben_all
        mal_avail = corpus.eval_indices(1, exclude_hashes=hashes) if hashes else mal_all
        excl = {"benign": int(ben_all.size - ben_avail.size), "malicious": int(mal_all.size - mal_avail.size)}
        ben_avail, dup_ben = unique_by_hash(corpus, ben_avail)
        mal_avail, dup_mal = unique_by_hash(corpus, mal_avail)
        cap = prm["max_samples_per_class"]
        ben_idx = corpus.subsample(ben_avail, cap, ctx.rng)
        mal_idx = corpus.subsample(mal_avail, cap, ctx.rng)
        n_ben, n_mal = int(ben_idx.size), int(mal_idx.size)
        subsampled = n_ben < ben_avail.size or n_mal < mal_avail.size
        details["eval_set"] = {
            "role": "eval",
            "splits": corpus.role_splits("eval"),
            "benign_available": int(ben_avail.size),
            "malicious_available": int(mal_avail.size),
            "benign_used": n_ben,
            "malicious_used": n_mal,
            "max_samples_per_class": cap,
            "subsampled": bool(subsampled),
            "duplicates_removed": {"benign": dup_ben, "malicious": dup_mal},
        }
        if dup_ben or dup_mal:
            notes.append(
                f"The canonical eval set repeats some files: {dup_ben} benign and {dup_mal} malicious duplicate "
                "rows (same sha256) were dropped so that each file counts once."
            )
        if hashes:
            details["excluded_training_members"] = dict(excl)
            n_ex = excl["benign"] + excl["malicious"]
            if n_ex:
                notes.append(
                    f"{n_ex} canonical eval samples ({excl['benign']} benign, {excl['malicious']} malicious) "
                    "are listed in your training manifest and were excluded; evaluating on them would "
                    "inflate the results."
                )
        else:
            details["excluded_training_members"] = None
            lead = ("The declared training manifest contains no sha256 hashes"
                    if (ctx.extras or {}).get("training_manifest_declared") else "No training manifest was declared")
            notes.append(
                f"{lead}, so malvalid cannot verify that the canonical eval "
                "samples were not used for training. If they were, these numbers are optimistic."
            )
        if subsampled:
            notes.append(
                f"Subsampled to at most {cap} samples per class (max_samples_per_class): "
                f"{n_ben} of {ben_avail.size} benign and {n_mal} of {mal_avail.size} malicious."
            )
        if corpus.synthetic:
            notes.append(
                f"The corpus '{corpus.name}' is synthetic: these numbers exercise the pipeline but say "
                "nothing about real-world performance."
            )

        if n_ben == 0 and n_mal == 0:
            finding = (
                "The canonical eval set has no usable labeled samples"
                + (" after excluding your training samples" if excl["benign"] + excl["malicious"] else "")
                + ", so FPR and detection rate could not be measured. The hard gate is not evaluated."
            )
            checks = self._checks(None, None, max_fpr, min_dr, t, n_ben, n_mal)
            metrics = {"n_benign": 0, "n_malicious": 0, "threshold": t}
            return self.result(ctx, finding=finding, checks=checks, metrics=metrics, details=details, notes=notes)

        # ---- score ----------------------------------------------------------------------------
        s_ben = score_indices(ctx, ben_idx)
        s_mal = score_indices(ctx, mal_idx)
        self._predict_spot_check(ctx, ben_idx, mal_idx, s_ben, s_mal, t, details, notes)

        fp = int(np.count_nonzero(s_ben >= t))
        tp = int(np.count_nonzero(s_mal >= t))
        fpr = fp / n_ben if n_ben else None
        dr = tp / n_mal if n_mal else None
        details["confusion"] = {"tp": tp, "fn": n_mal - tp, "fp": fp, "tn": n_ben - fp}
        metrics: dict[str, Any] = {
            "detection_rate": dr,
            "fpr": fpr,
            "n_benign": n_ben,
            "n_malicious": n_mal,
            "threshold": t,
            "auroc": None,
            "auprc": None,
            "brier": None,
            "ece": None,
            "tpr_at_fpr_0.001": None,
            "tpr_at_fpr_0.01": None,
            "threshold_at_max_fpr": None,
            "detection_rate_ci95": wilson_interval(tp, n_mal),
            "fpr_ci95": wilson_interval(fp, n_ben),
        }

        # ---- threshold-free metrics + charts (need both classes) ------------------------------
        y = np.concatenate([np.zeros(n_ben, dtype=np.int8), np.ones(n_mal, dtype=np.int8)])
        s = np.concatenate([s_ben, s_mal])
        retune: tuple[float | None, float, float] | None = None
        if n_ben and n_mal:
            from sklearn.metrics import average_precision_score

            f_arr, t_arr, thr_arr = roc_arrays(y, s)
            metrics["auroc"] = float(np.trapezoid(t_arr, f_arr))
            metrics["auprc"] = float(average_precision_score(y, s))
            metrics["tpr_at_fpr_0.001"] = float(tpr_at_fpr(f_arr, t_arr, 0.001))
            metrics["tpr_at_fpr_0.01"] = float(tpr_at_fpr(f_arr, t_arr, 0.01))
            retune = threshold_for_fpr(f_arr, t_arr, thr_arr, max_fpr)
            metrics["threshold_at_max_fpr"] = retune[0]
            details["at_threshold_at_max_fpr"] = {
                "threshold": retune[0], "fpr": retune[1], "detection_rate": retune[2],
            }
            if retune[0] is None:
                notes.append(
                    f"No finite threshold achieves FPR <= {_pct(max_fpr)} on the benign set: more than "
                    f"{_pct(max_fpr)} of benign samples share the model's highest score."
                )
            details["operating_points"] = self._operating_points(f_arr, t_arr, thr_arr, t, fpr, dr, max_fpr)
            self._roc_chart(ctx, f_arr, t_arr, n_ben, fpr, dr, t, max_fpr, min_dr)
            self._pr_chart(ctx, y, s, dr, tp, fp, t, n_ben, n_mal)
            ctx.artifacts.add_table(
                self.id, "operating_points", title="Operating points on the canonical eval set",
                columns=["operating point", "threshold", "FPR", "detection rate"],
                rows=[[r["label"], r["threshold"], r["fpr"], r["detection_rate"]] for r in details["operating_points"]],
                note="Classification rule: score >= threshold.",
            )
        else:
            missing = "benign" if not n_ben else "malicious"
            notes.append(f"The eval set has no {missing} samples; ROC/PR metrics are unavailable.")

        metrics["brier"] = float(np.mean((s - y) ** 2))
        ece, cal = expected_calibration_error(y, s, prm["calibration_bins"])
        metrics["ece"] = ece
        details["calibration"] = cal
        self._calibration_chart(ctx, cal)
        self._hist_chart(ctx, s_ben, s_mal, t)
        if n_ben and n_mal:
            notes.append(
                f"Calibration and precision are measured at the eval set's class balance "
                f"({_pct(n_mal / (n_ben + n_mal), 1)} malicious); real deployment prevalence is usually "
                "far lower, which lowers precision."
            )

        # ---- challenge split (informational) --------------------------------------------------
        if prm["include_challenge"]:
            self._challenge(ctx, corpus, hashes, cap, t, metrics, details, notes)

        checks = self._checks(fpr, dr, max_fpr, min_dr, t, n_ben, n_mal)
        self._ci_notes(metrics, fpr, dr, max_fpr, min_dr, notes)
        # after every other ctx.rng draw, so the eval subsample and spot check are unchanged
        self._threshold_noise(ctx, prm, s_ben, s_mal, checks, metrics, details, notes)
        finding = self._finding(t, fpr, dr, n_ben, n_mal, max_fpr, min_dr, retune)
        return self.result(ctx, finding=finding, checks=checks, metrics=metrics, details=details, notes=notes)

    # ---- pieces ----------------------------------------------------------------------------

    @staticmethod
    def _checks(fpr: float | None, dr: float | None, max_fpr: float, min_dr: float, t: float,
                n_ben: int, n_mal: int) -> list[GateCheck]:
        fpr_desc = (
            f"False-positive rate on {n_ben} canonical benign samples at operating threshold {_fmt_t(t)}"
            if n_ben else "False-positive rate: no benign eval samples available (not evaluated)"
        )
        dr_desc = (
            f"Detection rate on {n_mal} canonical malicious samples at operating threshold {_fmt_t(t)}"
            if n_mal else "Detection rate: no malicious eval samples available (not evaluated)"
        )
        return [
            graded_check(
                "fpr", fpr, "<=", max_fpr,
                ideal=max_fpr / 10.0, floor=min(0.5, 5.0 * max_fpr), scale="log", description=fpr_desc,
            ),
            graded_check(
                "detection_rate", dr, ">=", min_dr,
                ideal=min_dr + 0.9 * (1.0 - min_dr), floor=max(0.0, min_dr - 0.15), description=dr_desc,
            ),
        ]

    @staticmethod
    def _finding(t: float, fpr: float | None, dr: float | None, n_ben: int, n_mal: int, max_fpr: float,
                 min_dr: float, retune: tuple[float | None, float, float] | None) -> str:
        parts = []
        if fpr is not None and dr is not None:
            parts.append(
                f"At its declared operating threshold ({_fmt_t(t)}) the model flags {_pct(fpr)} of "
                f"{n_ben} canonical benign samples (FPR; max {_pct(max_fpr)}) and detects {_pct(dr, 1)} of "
                f"{n_mal} malicious samples (min {_pct(min_dr, 1)})."
            )
        elif fpr is not None:
            parts.append(f"At threshold {_fmt_t(t)} the FPR on {n_ben} benign samples is {_pct(fpr)} (max {_pct(max_fpr)}); no malicious eval samples were available.")
        elif dr is not None:
            parts.append(f"At threshold {_fmt_t(t)} the detection rate on {n_mal} malicious samples is {_pct(dr, 1)} (min {_pct(min_dr, 1)}); no benign eval samples were available.")
        fpr_fail = fpr is not None and fpr > max_fpr
        dr_fail = dr is not None and dr < min_dr
        if fpr_fail and dr_fail:
            parts.append("Both conditions fail: the false-positive rate is too high to ship and detection is below the minimum.")
        elif fpr_fail:
            parts.append("The false-positive rate exceeds the maximum: this detector would raise too many false alarms for production.")
        elif dr_fail:
            parts.append("The detection rate is below the minimum.")
        elif fpr is not None and dr is not None:
            parts.append("Both hard-gate conditions hold.")
        else:
            parts.append("The hard gate could not be fully evaluated.")
        if fpr_fail and retune is not None and retune[0] is not None:
            parts.append(
                f"On the same data a threshold of {_fmt_t(retune[0])} would meet max_fpr "
                f"(FPR {_pct(retune[1])}) at a detection rate of {_pct(retune[2], 1)}."
            )
        return " ".join(parts)

    @staticmethod
    def _ci_notes(metrics: dict[str, Any], fpr: float | None, dr: float | None, max_fpr: float,
                  min_dr: float, notes: list[str]) -> None:
        fci, dci = metrics.get("fpr_ci95"), metrics.get("detection_rate_ci95")
        if fpr is not None and fci and fpr <= max_fpr < fci[1]:
            notes.append(
                f"FPR passes on the point estimate, but its 95% interval reaches {_pct(fci[1])} (> max "
                f"{_pct(max_fpr)}): the estimate is too close to the limit for this many benign samples to "
                "show that the true FPR is within policy. Consider a slightly higher operating threshold."
            )
        if dr is not None and dci and dr >= min_dr > dci[0]:
            notes.append(
                f"Detection rate passes on the point estimate, but its 95% interval goes down to "
                f"{_pct(dci[0], 1)} (< min {_pct(min_dr, 1)}): the estimate is too close to the limit for "
                "this many malicious samples to show that the true detection rate meets policy."
            )

    @staticmethod
    def _threshold_noise(ctx: "RunContext", prm: dict[str, Any], s_ben: np.ndarray, s_mal: np.ndarray,
                         checks: list[GateCheck], metrics: dict[str, Any], details: dict[str, Any],
                         notes: list[str]) -> None:
        """Bootstrap the auto-calibrated threshold together with the eval rows (informational)."""
        import time

        cal = (ctx.extras or {}).get("threshold_calibration")
        B = int(prm["threshold_bootstrap_resamples"])
        if not isinstance(cal, dict) or cal.get("scores") is None:
            details["threshold_bootstrap"] = {
                "applicable": False,
                "reason": "the operating threshold was declared, not calibrated by malvalid, so it has no "
                          "calibration-sample noise; the Wilson intervals cover the eval-set noise",
            }
            return
        pol = calibration_policy_note(cal)
        if pol is not None:
            details["threshold_calibration_policy"] = pol[0]
            notes.append(pol[1])
        if B < 1:
            details["threshold_bootstrap"] = {"applicable": True, "ran": False,
                                              "reason": "threshold_bootstrap_resamples is 0"}
            return
        t0 = time.monotonic()
        try:
            bs = threshold_bootstrap(
                cal["scores"], float(cal["target_fpr"]), s_ben, s_mal, max_fpr=prm["max_fpr"],
                min_dr=prm["min_detection"], resamples=B, rng=ctx.rng, check_deadline=ctx.check_deadline,
            )
        except Exception as e:  # informational: never turn the hard gate's measured result into an error
            log.info("threshold bootstrap not completed: %s", e)
            details["threshold_bootstrap"] = {"applicable": True, "ran": False,
                                              "reason": f"not completed ({type(e).__name__}: {e})"}
            return
        bs = {"applicable": True, "ran": True, **bs,
              "method": "paired percentile bootstrap: each replicate re-draws the held-out benign calibration "
                        "rows and re-fits the threshold with the same rule, then re-draws the benign and "
                        "malicious eval rows",
              "duration_s": round(time.monotonic() - t0, 3)}
        metrics["threshold_ci95"] = bs["threshold_ci95"]
        metrics["fpr_ci95_with_threshold"] = bs["fpr_ci95"]
        metrics["detection_rate_ci95_with_threshold"] = bs["detection_rate_ci95"]
        within: list[str] = []
        labels = {"fpr": ("FPR", "max", _pct(prm["max_fpr"], 3), 3),
                  "detection_rate": ("Detection-rate", "min", _pct(prm["min_detection"], 2), 2)}
        for c in checks:
            frac = bs["pass_fraction"].get(c.name)
            if frac is None or c.passed is None or not 0.025 < frac < 0.975:
                continue
            within.append(c.name)
            name, side, limit, digits = labels[c.name]
            ci = bs["fpr_ci95" if c.name == "fpr" else "detection_rate_ci95"]
            how = "passes" if c.passed else "fails"
            other = "fails" if c.passed else "passes"
            notes.append(
                f"{name} gate decided within threshold noise: it {how} on the point estimate "
                f"({_pct(c.value, digits)}; {side} {limit}) but {other} in {_pct(1.0 - frac if c.passed else frac, 1)} "
                f"of {B} bootstrap replicates that re-draw the {bs['n_calibration']:,} held-out calibration rows "
                f"(re-fitting the auto-calibrated threshold, 95% interval {_fmt_t(bs['threshold_ci95'][0])}–"
                f"{_fmt_t(bs['threshold_ci95'][1])}) and the eval rows; 95% interval with threshold noise "
                f"{_pct(ci[0], digits)}–{_pct(ci[1], digits)}. The gate still uses the point estimate; declare "
                "the threshold you ship (--threshold) to remove this source of noise."
            )
        bs["decided_within_noise"] = within
        details["threshold_bootstrap"] = bs

    def _predict_spot_check(self, ctx: "RunContext", ben_idx: np.ndarray, mal_idx: np.ndarray,
                            s_ben: np.ndarray, s_mal: np.ndarray, t: float, details: dict[str, Any],
                            notes: list[str]) -> None:
        """Check on a small sample that ``predict`` agrees with ``predict_proba >= threshold``."""
        idx = np.concatenate([ben_idx, mal_idx])
        s = np.concatenate([s_ben, s_mal])
        if idx.size == 0:
            return
        k = min(_PREDICT_SPOT_CHECK_ROWS, idx.size)
        pos = np.sort(ctx.rng.choice(idx.size, size=k, replace=False))
        try:
            ctx.check_deadline()
            pred = np.asarray(ctx.model.predict(ctx.corpus.take(idx[pos]))).reshape(-1)  # type: ignore[union-attr]
        except Exception as e:  # informational only; the gate uses predict_proba
            log.info("predict() spot check failed: %s", e)
            details["predict_consistency"] = {"checked": 0, "error": f"{type(e).__name__}: {e}"}
            notes.append(f"predict() could not be spot-checked ({type(e).__name__}); M1 uses predict_proba >= operating_threshold.")
            return
        expected = (s[pos] >= t).astype(np.int8)
        n_bad = int(np.count_nonzero(pred.astype(np.int8) != expected))
        details["predict_consistency"] = {"checked": int(k), "disagreements": n_bad}
        if n_bad:
            notes.append(
                f"predict() disagrees with predict_proba() >= operating_threshold ({_fmt_t(t)}) on {n_bad} of "
                f"{k} spot-checked samples; M1 uses the latter. Make predict() consistent before deploying."
            )

    @staticmethod
    def _operating_points(f_arr: np.ndarray, t_arr: np.ndarray, thr_arr: np.ndarray, t: float,
                          fpr: float | None, dr: float | None, max_fpr: float) -> list[dict[str, Any]]:
        rows = [{"label": "declared operating threshold", "threshold": t, "fpr": fpr, "detection_rate": dr}]
        targets = [(f"FPR <= {max_fpr:g} (max_fpr)", max_fpr)]
        for tgt in (0.001, 0.01):
            if not math.isclose(tgt, max_fpr):
                targets.append((f"FPR <= {tgt:g}", tgt))
        for label, tgt in targets:
            th, fp_, tp_ = threshold_for_fpr(f_arr, t_arr, thr_arr, tgt)
            rows.append({"label": label, "threshold": th, "fpr": fp_, "detection_rate": tp_})
        return rows

    def _roc_chart(self, ctx: "RunContext", f_arr: np.ndarray, t_arr: np.ndarray, n_ben: int,
                   fpr: float | None, dr: float | None, t: float, max_fpr: float, min_dr: float) -> None:
        xmin = min(1e-4, 10.0 ** math.floor(math.log10(0.5 / max(n_ben, 1))))
        xmin = max(xmin, 1e-7)
        grid = np.logspace(math.log10(xmin), 0.0, _ROC_POINTS)
        series: list[dict[str, Any]] = [{"label": "ROC", "x": grid, "y": tpr_at_fpr(f_arr, t_arr, grid)}]
        if fpr is not None and dr is not None:
            series.append({"label": f"operating point (threshold {_fmt_t(t)})", "x": [max(fpr, xmin)], "y": [dr]})
        ctx.artifacts.add_chart(
            self.id, "roc", title="ROC on the canonical eval set", series=series, kind="line",
            xlabel="false-positive rate (log)", ylabel="detection rate (TPR)", xscale="log",
            xlim=(xmin, 1.0), ylim=(0.0, 1.0),
            reference_lines=[
                {"axis": "x", "value": max_fpr, "label": f"max_fpr {max_fpr:g}"},
                {"axis": "y", "value": min_dr, "label": f"min_detection {min_dr:g}"},
            ],
            note=f"An FPR of 0 is drawn at {xmin:g} on the log axis.",
        )

    def _pr_chart(self, ctx: "RunContext", y: np.ndarray, s: np.ndarray, dr: float | None, tp: int, fp: int,
                  t: float, n_ben: int, n_mal: int) -> None:
        from sklearn.metrics import precision_recall_curve

        prec, rec, _ = precision_recall_curve(y, s)
        series: list[dict[str, Any]] = [{"label": "PR", "x": rec[::-1], "y": prec[::-1]}]
        if dr is not None and tp + fp > 0:
            series.append({"label": f"operating point (threshold {_fmt_t(t)})", "x": [dr], "y": [tp / (tp + fp)]})
        ctx.artifacts.add_chart(
            self.id, "pr", title="Precision-recall on the canonical eval set", series=series, kind="line",
            xlabel="recall (detection rate)", ylabel="precision", xlim=(0.0, 1.0), ylim=(0.0, 1.0),
            note=f"Precision at the eval set's class balance ({n_mal} malicious : {n_ben} benign).",
        )

    def _calibration_chart(self, ctx: "RunContext", cal: dict[str, Any]) -> None:
        cnt = np.asarray(cal["count"])
        nz = cnt > 0
        mp = np.asarray(cal["mean_predicted"], dtype=np.float64)[nz]
        fr = np.asarray(cal["fraction_malicious"], dtype=np.float64)[nz]
        ctx.artifacts.add_chart(
            self.id, "calibration", title="Reliability diagram", kind="line",
            series=[{"label": "model", "x": mp, "y": fr}],
            xlabel="mean predicted score (per bin)", ylabel="observed fraction malicious",
            xlim=(0.0, 1.0), ylim=(0.0, 1.0), note="diagonal",
        )

    def _hist_chart(self, ctx: "RunContext", s_ben: np.ndarray, s_mal: np.ndarray, t: float) -> None:
        edges = np.linspace(0.0, 1.0, _HIST_BINS + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        series = []
        for label, sc in (("benign", s_ben), ("malicious", s_mal)):
            if sc.size:
                h, _ = np.histogram(np.clip(sc, 0.0, 1.0), bins=edges)
                series.append({"label": f"{label} (n={sc.size})", "x": centers, "y": h / sc.size})
        ctx.artifacts.add_chart(
            self.id, "score_hist", title="Score distribution by class", series=series, kind="step",
            xlabel="model score", ylabel="fraction of class", xlim=(0.0, 1.0),
            reference_lines=[{"axis": "x", "value": t, "label": f"operating threshold {_fmt_t(t)}"}],
        )

    def _challenge(self, ctx: "RunContext", corpus: Any, hashes: frozenset[str] | None, cap: int | None,
                   t: float, metrics: dict[str, Any], details: dict[str, Any], notes: list[str]) -> None:
        if not corpus.role_splits("challenge"):
            details["challenge"] = {"available": False}
            return
        all_idx = corpus.indices(role="challenge", label=1)
        avail = corpus.indices(role="challenge", label=1, exclude_hashes=hashes) if hashes else all_idx
        avail, n_dup = unique_by_hash(corpus, avail)
        if avail.size == 0:
            details["challenge"] = {"available": False, "excluded_training_members": int(all_idx.size - avail.size)}
            if all_idx.size:
                notes.append("Every challenge-set sample is in your training manifest; challenge detection rate not measured.")
            return
        idx = corpus.subsample(avail, cap, ctx.rng)
        sc = score_indices(ctx, idx)
        k = int(np.count_nonzero(sc >= t))
        cdr = k / idx.size
        metrics["challenge_detection_rate"] = cdr
        details["challenge"] = {
            "available": True,
            "splits": corpus.role_splits("challenge"),
            "n": int(idx.size),
            "n_available": int(avail.size),
            "detected": k,
            "detection_rate_ci95": wilson_interval(k, int(idx.size)),
            "duplicates_removed": n_dup,
            "excluded_training_members": int(all_idx.size - avail.size) if hashes else None,
        }
        notes.append(
            f"Challenge set (historically evasive malware, informational): {_pct(cdr, 1)} of {idx.size} "
            "detected at the operating threshold."
        )


__all__ = [
    "PerformanceModule",
    "score_indices",
    "graded_check",
    "wilson_interval",
    "unique_by_hash",
    "expected_calibration_error",
    "roc_arrays",
    "tpr_at_fpr",
    "threshold_for_fpr",
    "threshold_bootstrap",
    "calibration_policy_note",
]
