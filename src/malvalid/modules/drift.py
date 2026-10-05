"""M2 — Temporal drift: TESSERACT-style time-aware evaluation, summarized as AUT(F1).

A detector is trained on the past and deployed on the future. M2 measures how its F1 at the
declared operating threshold holds up on canonical samples that *post-date* its training data:

1. **Effective cutoff (C1).** Evaluation windows start after the declared ``training_cutoff``. If
   the training manifest is known and some of its samples are dated after the declared cutoff,
   the declaration is inconsistent; M2 notes it and uses the newest training-member timestamp as
   the effective cutoff so that no window overlaps training data.
2. **Windows.** Temporal-role rows dated after the effective cutoff, excluding training members,
   are partitioned with TESSERACT's ``time_aware_indexes`` into consecutive windows of one
   ``granularity`` unit each, starting the day after the cutoff. Windows smaller than
   ``min_window_samples`` or (with ``require_both_classes``) missing a class are dropped and listed.
3. **Per window** at the operating threshold: n, n_mal, n_ben, precision, recall, F1, FPR; plus the
   TESSERACT constraint diagnostics C2 (malware/goodware time alignment) and C3 (malware ratio).
4. **AUT(F1)** — area under the F1-over-time curve, unweighted (trapezoid / (N - 1), as in
   TESSERACT) and **sample-weighted** so a sparse window cannot count as much as a dense one. The
   gate is on the weighted value. Fewer than two usable windows => the check is unevaluable
   (status warn, never a pass).

TESSERACT (BSD-3) is used from the installed ``tesseract`` package when importable, else from the
vendored subset in :mod:`malvalid._vendor.tesseract_temporal`.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from malvalid.core import ConfigError, GateCheck, GateMode, Module, ModuleResult, Requirement
from malvalid.modules.performance import (
    calibration_policy_note,
    _param_bool,
    _param_float,
    _param_int,
    graded_check,
    score_indices,
    unique_by_hash,
    wilson_interval,
)

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.context import RunContext
    from malvalid.corpora.base import Corpus

log = logging.getLogger("malvalid.modules.drift")

GRANULARITIES = ("year", "quarter", "month", "week", "day")
AUTO = "auto"  # pick "week" when timestamps are day-level and every weekly window is big enough, else "month"
DEFAULT_MAX_FPR = 0.01  # M1's default budget, used when the performance module is not configured
_ADJECTIVE = {"year": "yearly", "quarter": "quarterly", "month": "monthly", "week": "weekly", "day": "daily"}
C2_MONTH_VARIANCE = 1  # TESSERACT's default month_variance for malware/goodware alignment
C3_RATIO_TOLERANCE = 0.10  # flag windows whose malware ratio is this far from the median window
_VENDORED_LABEL = "vendored TESSERACT helpers (malvalid._vendor.tesseract_temporal, BSD-3, tesseract-ml-release @05132c8)"


# --------------------------------------------------------------------------------------------------
# TESSERACT backend
# --------------------------------------------------------------------------------------------------


def load_temporal(prefer_installed: bool = True) -> tuple[ModuleType, str]:
    """Return ``(temporal_module, description)``: installed ``tesseract.temporal`` if importable,
    else malvalid's vendored BSD-3 copy of the helpers it needs."""
    if prefer_installed:
        try:
            from tesseract import temporal as tt  # type: ignore[import-not-found]

            for fn in ("time_aware_indexes", "get_relative_delta", "month_difference"):
                getattr(tt, fn)
            try:
                from importlib.metadata import version

                ver = version("tesseract")
            except Exception:  # pragma: no cover - metadata missing
                ver = "unknown"
            return tt, f"tesseract {ver} (installed)"
        except Exception as e:  # pragma: no cover - depends on environment
            log.debug("tesseract.temporal not importable (%s); using vendored helpers", e)
    from malvalid._vendor import tesseract_temporal as vt

    return vt, _VENDORED_LABEL


# --------------------------------------------------------------------------------------------------
# Pure functions (unit-tested)
# --------------------------------------------------------------------------------------------------


def aut_unweighted(f: Sequence[float]) -> float | None:
    """TESSERACT's AUT: trapezoid area under the metric-over-windows curve / (N - 1).

    None for fewer than two windows (a single window is not a time-aware evaluation).
    """
    a = np.asarray(f, dtype=np.float64)
    if a.size < 2:
        return None
    return float(np.sum((a[:-1] + a[1:]) / 2.0) / (a.size - 1))


def aut_sample_weighted(f: Sequence[float], n: Sequence[float]) -> float | None:
    """Sample-weighted AUT over consecutive windows ``k = 1..N-1``::

        sum_k ((n_k f_k + n_{k+1} f_{k+1}) / (n_k + n_{k+1})) * (n_k + n_{k+1}) / sum_j (n_j + n_{j+1})

    Each trapezoid segment's height is the size-weighted mean of its two windows and its width is
    proportional to the samples it covers, so a sparse window cannot swing AUT as much as a dense
    one. Equals :func:`aut_unweighted` when all windows have the same size.
    """
    fa = np.asarray(f, dtype=np.float64)
    na = np.asarray(n, dtype=np.float64)
    if fa.size != na.size:
        raise ValueError("f and n must have the same length")
    if fa.size < 2:
        return None
    if np.any(na < 0):
        raise ValueError("window sizes must be non-negative")
    pair_n = na[:-1] + na[1:]
    total = float(np.sum(pair_n))
    if total <= 0:
        return None
    seg = np.divide(na[:-1] * fa[:-1] + na[1:] * fa[1:], pair_n, out=np.zeros_like(pair_n), where=pair_n > 0)
    return float(np.sum(seg * pair_n / total))


def train_test_temporally_consistent(t_train: np.ndarray, t_test: np.ndarray) -> bool:
    """Vectorized TESSERACT C1 (``assert_train_test_temporal_consistency``): no training date is
    later than any test date, i.e. ``max(t_train) <= min(t_test)``. O(n) instead of O(n*m)."""
    a = np.asarray(t_train)
    b = np.asarray(t_test)
    if a.size == 0 or b.size == 0:
        return True
    return bool(a.max() <= b.min())


def binary_counts(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, Any]:
    """Confusion counts and precision/recall/F1/FPR at ``score >= threshold``.

    Undefined ratios follow scikit-learn's ``zero_division=0`` convention, except FPR, which is
    None when a window has no benign samples.
    """
    y = np.asarray(y)
    pred = np.asarray(s) >= threshold
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Rank-based (Mann-Whitney) AUROC with ties averaged; None if a class is missing."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    n1 = int(np.count_nonzero(y == 1))
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return None
    from scipy.stats import rankdata

    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def operating_point_stats(counts: dict[str, Any], y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    """Add the threshold-free and operating-point extras to :func:`binary_counts` output: AUROC,
    realised FPR and miss rate (FNR) with Wilson 95% intervals (None when the class is absent)."""
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    n_mal, n_ben = tp + fn, fp + tn
    return {
        "auroc": auroc(y, s),
        "fnr": (fn / n_mal) if n_mal else None,
        "fnr_ci95": wilson_interval(fn, n_mal),
        "fpr_ci95": wilson_interval(fp, n_ben),
    }


def fpr_budget_summary(windows: Sequence[dict[str, Any]], max_fpr: float, min_benign: int) -> dict[str, Any]:
    """Operating-point FPR drift across windows. ``windows`` items need ``label``, ``fp``, ``n_ben``.

    Windows with fewer than ``min_benign`` benign rows are *unevaluable* for FPR (listed, never
    counted as within budget). ``value`` is the largest Wilson 95% *lower* bound of window FPR: the
    check ``value <= max_fpr`` fails only when some window's FPR is statistically above the budget,
    so a point estimate that merely brushes the limit by sampling noise does not fail.
    """
    ev, unev = [], []
    for w in windows:
        (ev if w["n_ben"] >= max(1, min_benign) else unev).append(w)
    out: dict[str, Any] = {
        "max_fpr": max_fpr, "min_window_benign": min_benign, "n_evaluable": len(ev),
        "unevaluable_windows": [w["label"] for w in unev], "max_window_fpr": None, "max_window_fpr_label": None,
        "max_window_fpr_ci95": None, "value": None, "windows_over_budget": [], "windows_significantly_over_budget": [],
    }
    if not ev:
        return out
    fprs = [w["fp"] / w["n_ben"] for w in ev]
    i = int(np.argmax(fprs))
    out["max_window_fpr"] = float(fprs[i])
    out["max_window_fpr_label"] = ev[i]["label"]
    out["max_window_fpr_ci95"] = wilson_interval(ev[i]["fp"], ev[i]["n_ben"])
    los = [wilson_interval(w["fp"], w["n_ben"])[0] for w in ev]  # type: ignore[index]
    out["value"] = float(max(los))
    out["windows_over_budget"] = [w["label"] for w, f in zip(ev, fprs) if f > max_fpr]
    out["windows_significantly_over_budget"] = [w["label"] for w, lo in zip(ev, los) if lo > max_fpr]
    return out


def _to_date(x: np.datetime64) -> dt.date:
    return np.datetime64(x, "D").astype(object)


def _median_date(ts: np.ndarray) -> dt.date:
    days = np.asarray(ts, dtype="datetime64[D]").astype(np.int64)
    return _to_date(np.datetime64(int(round(float(np.median(days)))), "D"))


def _norm_granularity(g: Any) -> str:
    if not isinstance(g, str):
        raise ConfigError(f"drift.granularity must be one of {list(GRANULARITIES) + [AUTO]}, got {g!r}")
    s = g.strip().lower()
    if s == AUTO:
        return AUTO
    s = s[:-1] if s.endswith("s") and s[:-1] in GRANULARITIES else s
    if s not in GRANULARITIES:
        raise ConfigError(f"drift.granularity must be one of {list(GRANULARITIES) + [AUTO]}, got {g!r}")
    return s


def _ci_txt(ci: Sequence[float] | None) -> str | None:
    return None if not ci else f"{ci[0]:.4f}-{ci[1]:.4f}"


def _window_label(start: dt.date, end: dt.date, gran: str) -> str:
    if gran == "month" and start.day == 1:
        return f"{start.year:04d}-{start.month:02d}"
    if gran == "quarter" and start.day == 1 and start.month in (1, 4, 7, 10):
        return f"{start.year:04d}Q{(start.month - 1) // 3 + 1}"
    if gran == "year" and (start.month, start.day) == (1, 1):
        return f"{start.year:04d}"
    if gran == "day":
        return start.isoformat()
    return f"{start.isoformat()}..{end.isoformat()}"


# --------------------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------------------


@dataclass
class Window:
    k: int  # 1-based period number after the effective cutoff (empty periods count)
    start: dt.date
    end: dt.date  # inclusive
    label: str
    idx_available: np.ndarray
    idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    n_mal: int = 0
    n_ben: int = 0
    status: str = "used"
    metrics: dict[str, Any] = field(default_factory=dict)
    c2: dict[str, Any] = field(default_factory=dict)

    @property
    def used(self) -> bool:
        return self.status == "used"

    @property
    def n(self) -> int:
        return int(self.idx.size)

    def row(self) -> dict[str, Any]:
        m = self.metrics
        return {
            "window": self.label,
            "k": self.k,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "n_available": int(self.idx_available.size),
            "n": self.n,
            "n_mal": self.n_mal,
            "n_ben": self.n_ben,
            "malware_ratio": (self.n_mal / self.n) if self.n else None,
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "f1": m.get("f1"),
            "fpr": m.get("fpr"),
            "fnr": m.get("fnr"),
            "fpr_ci95": m.get("fpr_ci95"),
            "fnr_ci95": m.get("fnr_ci95"),
            "auroc": m.get("auroc"),
            "fp": m.get("fp"),
            "fn": m.get("fn"),
            "c2_gap_months": self.c2.get("gap_months"),
            "status": self.status,
        }


def partition_windows(
    tt: ModuleType, corpus: "Corpus", idx: np.ndarray, cutoff: dt.date, granularity: str
) -> list[Window]:
    """Split corpus rows ``idx`` (all dated after ``cutoff``) into consecutive TESSERACT windows of
    one ``granularity`` unit, the first starting the day after ``cutoff``. Empty periods are
    omitted but keep their period number ``k``."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return []
    ts = corpus.timestamp[idx].astype("datetime64[D]")
    t_obj = ts.astype("datetime64[us]").astype(object)  # datetime.datetime, as TESSERACT expects
    start = dt.datetime.combine(cutoff + dt.timedelta(days=1), dt.time())
    train, tests = tt.time_aware_indexes(t_obj, 0, 1, granularity, start_date=start)
    if len(train):  # cannot happen: every row is after the cutoff
        raise AssertionError("rows before the window start leaked into the drift partition")
    out: list[Window] = []
    lo = start + tt.get_relative_delta(0, granularity)
    for k, pos in enumerate(tests, start=1):
        hi = lo + tt.get_relative_delta(1, granularity)
        if len(pos):
            w_start = lo.date()
            w_end = hi.date() - dt.timedelta(days=1)
            out.append(
                Window(
                    k=k,
                    start=w_start,
                    end=w_end,
                    label=_window_label(w_start, w_end, granularity),
                    idx_available=np.sort(idx[np.asarray(pos, dtype=np.int64)]),
                )
            )
        lo = hi
    return out


# --------------------------------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------------------------------


class DriftModule(Module):
    id = "drift"
    code = "M2"
    title = "Temporal drift"
    description = (
        "TESSERACT-style time-aware evaluation: scores canonical samples dated after your declared "
        "training cutoff in consecutive time windows and reports per-window precision/recall/F1/FPR "
        "at your operating threshold, the decay curve, and AUT(F1) — sample-weighted so sparse "
        "windows cannot dominate. Enforces TESSERACT's temporal constraints (C1-C3)."
    )
    requires = (Requirement.FEATURE_SPACE, Requirement.TRAINING_CUTOFF)
    enhanced_by = (Requirement.TRAINING_HASHES,)
    default_gate = GateMode.WARN
    default_params = {
        "min_aut_f1": 0.70,
        "granularity": "month",  # year|quarter|month|week|day|auto (auto: week if day-level timestamps allow)
        "min_window_samples": 200,
        "min_window_benign": 100,  # fewer benign rows => the window's FPR is unevaluable
        "max_windows": 36,
        "max_samples_per_window": None,
        "require_both_classes": True,
    }

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        cls._parse_params(params)

    def _params(self, ctx: "RunContext") -> dict[str, Any]:
        return self._parse_params(ctx.params)

    @classmethod
    def _parse_params(cls, p: dict[str, Any]) -> dict[str, Any]:
        prm = {
            "min_aut_f1": _param_float(p, "min_aut_f1", cls.id, 0.0, 1.0),
            "granularity": _norm_granularity(p.get("granularity")),
            "min_window_samples": _param_int(p, "min_window_samples", cls.id, 1),
            "min_window_benign": _param_int(p, "min_window_benign", cls.id, 1),
            "max_windows": _param_int(p, "max_windows", cls.id, 2),
            "max_samples_per_window": _param_int(p, "max_samples_per_window", cls.id, 1, allow_none=True),
            "require_both_classes": _param_bool(p, "require_both_classes", cls.id),
        }
        cap = prm["max_samples_per_window"]
        if cap is not None and cap < prm["min_window_samples"]:
            raise ConfigError(
                f"drift.max_samples_per_window ({cap}) must be >= drift.min_window_samples "
                f"({prm['min_window_samples']}), otherwise every window would be dropped"
            )
        return prm

    # ---- run -------------------------------------------------------------------------------

    def run(self, ctx: "RunContext") -> ModuleResult:
        if ctx.corpus is None or not ctx.has(Requirement.FEATURE_SPACE):
            return self.skip(ctx, ctx.missing_reason(Requirement.FEATURE_SPACE))
        if ctx.training_cutoff is None:
            return self.skip(
                ctx,
                ctx.missing_reason(Requirement.TRAINING_CUTOFF)
                + "; declare training_cutoff = 'YYYY-MM-DD' (date of your newest training sample) on "
                "the adapter to enable the drift test",
            )
        prm = self._params(ctx)
        corpus = ctx.corpus
        if not corpus.role_splits("temporal"):
            return self.skip(ctx, f"the canonical corpus '{corpus.name}' defines no temporal split")
        temporal = corpus.indices(role="temporal", require_timestamp=True)
        if temporal.size == 0:
            return self.skip(ctx, f"the canonical corpus '{corpus.name}' has no timestamped labeled temporal samples")

        tt, backend = load_temporal()
        gran_requested = prm["granularity"]
        gran = gran_requested
        hashes = ctx.training_hashes or None
        declared = ctx.training_cutoff
        notes: list[str] = []
        details: dict[str, Any] = {"tesseract": backend, "granularity": gran, "granularity_requested": gran_requested}

        eff, c1, member_ts = self._effective_cutoff(corpus, declared, hashes, notes)
        corpus_end = _to_date(corpus.timestamp[temporal].max())
        corpus_start = _to_date(corpus.timestamp[temporal].min())
        details["corpus_temporal_range"] = [corpus_start.isoformat(), corpus_end.isoformat()]
        if not hashes:
            lead = ("The declared training manifest contains no sha256 hashes"
                    if (ctx.extras or {}).get("training_manifest_declared") else "No training manifest was declared")
            notes.append(
                f"{lead}, so malvalid cannot check that none of the evaluated "
                "post-cutoff samples were used for training; if some were, drift is understated."
            )
        if corpus.synthetic:
            notes.append(f"The corpus '{corpus.name}' is synthetic: drift numbers exercise the pipeline only.")
        pol = calibration_policy_note((ctx.extras or {}).get("threshold_calibration"))
        if pol is not None:
            details["threshold_calibration_policy"] = pol[0]
            notes.append(pol[1].replace("FPR here", "per-window FPR")
                         + (" Windows inside the calibration period have fewer (or no) benign rows."
                            if pol[0]["policy"] == "earliest" else ""))

        cand = corpus.indices(role="temporal", after=eff, exclude_hashes=hashes)
        cand, n_dup = unique_by_hash(corpus, cand)
        details["duplicates_removed"] = n_dup
        if n_dup:
            notes.append(
                f"{n_dup} duplicate post-cutoff rows (same sha256) were dropped so that each file counts once."
            )
        base_metrics: dict[str, Any] = {"effective_cutoff": eff.isoformat()}
        if cand.size == 0:
            if corpus_end <= eff:
                why = (
                    f"The effective training cutoff ({eff.isoformat()}) is on or after the newest dated sample in "
                    f"the canonical corpus's temporal split ({corpus_end.isoformat()}), so there are no future "
                    "samples to measure drift on."
                )
            else:
                why = (
                    f"All temporal samples dated after the effective training cutoff ({eff.isoformat()}) are in "
                    "your training manifest, so there are no unseen future samples to measure drift on."
                )
            details["temporal_constraints"] = {"c1": c1}
            return self._unevaluable(ctx, prm, why, base_metrics, details, notes, n_windows=0)

        if gran_requested == AUTO:
            gran, auto_info = self._auto_granularity(tt, corpus, cand, eff, prm)
            details["granularity"] = gran
            details["granularity_auto"] = auto_info
            notes.append(f"granularity=auto selected '{gran}': {auto_info['reason']}")
        windows = partition_windows(tt, corpus, cand, eff, gran)
        considered, beyond = windows[: prm["max_windows"]], windows[prm["max_windows"] :]
        if beyond:
            n_beyond = int(sum(w.idx_available.size for w in beyond))
            details["beyond_max_windows"] = {
                "windows": len(beyond), "samples": n_beyond, "first": beyond[0].label, "last": beyond[-1].label,
            }
            notes.append(
                f"Only the first {prm['max_windows']} non-empty windows after the cutoff were evaluated "
                f"(max_windows); {len(beyond)} later window(s) with {n_beyond} samples were not."
            )
        self._select(ctx, considered, prm)
        used = [w for w in considered if w.used]
        dropped = [w for w in considered if not w.used]
        details["dropped_windows"] = [{"window": w.label, "n": int(w.idx_available.size), "reason": w.status} for w in dropped]
        if dropped:
            notes.append(
                f"{len(dropped)} window(s) dropped: "
                + "; ".join(f"{w.label} ({w.status})" for w in dropped[:6])
                + ("; ..." if len(dropped) > 6 else "")
                + ". AUT is computed over the remaining windows in time order."
            )
        n_sub = sum(1 for w in used if w.n < w.idx_available.size)
        details["subsampling"] = {"max_samples_per_window": prm["max_samples_per_window"], "windows_subsampled": n_sub}
        if n_sub:
            notes.append(
                f"{n_sub} window(s) were subsampled to max_samples_per_window={prm['max_samples_per_window']}; "
                "the sample weights in AUT use the evaluated counts."
            )

        # ---- score used windows ----------------------------------------------------------------
        t = ctx.threshold
        if used:
            all_idx = np.concatenate([w.idx for w in used])
            order = np.argsort(all_idx, kind="stable")
            scores = np.empty(all_idx.size, dtype=np.float64)
            scores[order] = score_indices(ctx, all_idx[order])
            pos = 0
            for w in used:
                s = scores[pos : pos + w.n]
                pos += w.n
                yw = corpus.label[w.idx]
                w.metrics = binary_counts(yw, s, t)
                w.metrics.update(operating_point_stats(w.metrics, yw, s))
        details["n_samples_evaluated"] = int(sum(w.n for w in used))
        if used:  # C1, verified on what was actually evaluated (holds by construction)
            eval_ts = np.concatenate([corpus.timestamp[w.idx] for w in used])
            c1["holds"] = bool(eval_ts.min() > np.datetime64(eff, "D")) and train_test_temporally_consistent(
                member_ts, eval_ts
            )
            c1["earliest_evaluated_timestamp"] = _to_date(eval_ts.min()).isoformat()

        c2, c3 = self._constraints(tt, corpus, used, notes)
        details["temporal_constraints"] = {"c1": c1, "c2": c2, "c3": c3}
        rows = [w.row() for w in considered]
        details["windows"] = rows
        max_fpr = self._budget(ctx)
        fb = fpr_budget_summary(
            [{"label": w.label, "fp": w.metrics["fp"], "n_ben": w.n_ben} for w in used], max_fpr, prm["min_window_benign"]
        )
        details["operating_point"] = fb
        self._charts(ctx, considered, used, eff, gran, prm, max_fpr)

        if len(used) < 2:
            why = (
                f"Only {len(used)} usable {_ADJECTIVE[gran]} window(s) after the effective training cutoff "
                f"({eff.isoformat()}); AUT needs at least 2"
                + (f" ({len(dropped)} dropped as too small or single-class)" if dropped else "")
                + "."
            )
            m = dict(base_metrics)
            if used:
                m.update({"first_window": used[0].label, "last_window": used[0].label,
                          "f1_first": used[0].metrics["f1"], "f1_last": used[0].metrics["f1"]})
            return self._unevaluable(ctx, prm, why, m, details, notes, n_windows=len(used))

        f = [w.metrics["f1"] for w in used]
        n = [w.n for w in used]
        aut = aut_unweighted(f)
        aut_w = aut_sample_weighted(f, n)
        metrics = {
            "n_windows": len(used),
            "first_window": used[0].label,
            "last_window": used[-1].label,
            "f1_first": f[0],
            "f1_last": f[-1],
            "f1_drop": f[0] - f[-1],
            "aut_f1": aut,
            "aut_f1_weighted": aut_w,
            "effective_cutoff": eff.isoformat(),
        }
        metrics.update(self._operating_metrics(used, fb))
        check = self._check(aut_w, prm, len(used), gran, None)
        fpr_check = self._fpr_check(fb, max_fpr)
        if fb["unevaluable_windows"] and fb["n_evaluable"]:
            notes.append(
                f"FPR is unevaluable in {len(fb['unevaluable_windows'])} window(s) with fewer than "
                f"{prm['min_window_benign']} benign samples ({', '.join(fb['unevaluable_windows'][:6])}); "
                "they are excluded from the operating-point FPR check."
            )
        finding = self._finding(metrics, prm, eff, gran, check) + self._operating_headline(metrics, fb, max_fpr)
        return self.result(ctx, finding=finding, checks=[check, fpr_check], metrics=metrics, details=details, notes=notes)

    # ---- pieces ----------------------------------------------------------------------------

    @staticmethod
    def _budget(ctx: "RunContext") -> float:
        """M1's ``max_fpr`` from the gate config (the FPR the operator promised to ship at)."""
        try:
            v = ctx.config.module("performance").params().get("max_fpr")
        except Exception:  # pragma: no cover - config without a performance section
            v = None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 < float(v) < 1.0:
            return DEFAULT_MAX_FPR
        return float(v)

    @staticmethod
    def _operating_metrics(used: list[Window], fb: dict[str, Any]) -> dict[str, Any]:
        a, b = used[0].metrics, used[-1].metrics
        return {
            "auroc_first": a.get("auroc"), "auroc_last": b.get("auroc"),
            "fnr_first": a.get("fnr"), "fnr_last": b.get("fnr"),
            "fpr_first": a.get("fpr"), "fpr_last": b.get("fpr"),
            "max_window_fpr": fb["max_window_fpr"], "max_window_fpr_window": fb["max_window_fpr_label"],
            "max_window_fpr_ci95": fb["max_window_fpr_ci95"],
            "max_fpr_budget": fb["max_fpr"],
            "windows_over_fpr_budget": len(fb["windows_over_budget"]),
        }

    @staticmethod
    def _fpr_check(fb: dict[str, Any], max_fpr: float) -> GateCheck:
        """Operating-point drift: is any window's realised FPR significantly above the M1 budget?

        Deliberately *ungraded* (no ideal/floor => no axis score): it can flag the module (warn gate)
        but never moves the 0-100 score, which stays AUT(F1)-only.
        """
        v = fb["value"]
        if v is None:
            desc = (f"Realised FPR per window vs the M1 budget {max_fpr:g}: not evaluable — no window has "
                    f">= {fb['min_window_benign']} benign samples")
        else:
            desc = (
                f"Largest per-window 95% lower bound of the realised FPR at the frozen threshold over "
                f"{fb['n_evaluable']} window(s) (budget max_fpr {max_fpr:g}; worst point estimate "
                f"{fb['max_window_fpr']:.4f} in {fb['max_window_fpr_label']})"
            )
        return GateCheck.evaluate("window_fpr_within_budget", v, "<=", max_fpr, metric="window_fpr_lower_bound",
                                  description=desc)

    @staticmethod
    def _operating_headline(m: dict[str, Any], fb: dict[str, Any], max_fpr: float) -> str:
        def pc(x: float | None, d: int = 1) -> str:
            return "n/a" if x is None else f"{100 * x:.{d}f}%"

        def f3(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.3f}"

        s = (
            f" Operating point: AUROC {f3(m['auroc_first'])}\u2192{f3(m['auroc_last'])} while the miss rate at the shipped threshold "
            f"goes {pc(m['fnr_first'])}\u2192{pc(m['fnr_last'])} and the realised FPR {pc(m['fpr_first'], 2)}\u2192"
            f"{pc(m['fpr_last'], 2)} (first\u2192last window; FPR budget {pc(max_fpr, 2)})."
        )
        if fb["value"] is None:
            s += " Per-window FPR could not be evaluated (too few benign samples), so operating-point drift is unassessed."
        elif fb["windows_significantly_over_budget"]:
            s += (f" FPR is significantly above the budget in {len(fb['windows_significantly_over_budget'])} window(s) "
                  f"(worst {pc(fb['max_window_fpr'], 2)} in {fb['max_window_fpr_label']}): the frozen threshold no "
                  "longer meets the M1 false-positive budget on newer data.")
        elif fb["windows_over_budget"]:
            s += (f" FPR point estimates exceed the budget in {len(fb['windows_over_budget'])} window(s) but within "
                  "sampling noise (the 95% interval still includes the budget).")
        return s

    @staticmethod
    def _auto_granularity(tt: ModuleType, corpus: "Corpus", cand: np.ndarray, eff: dt.date,
                          prm: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """``granularity=auto``: weekly when the timestamps are day-level and weekly windows are
        large enough to be evaluated (``min_window_samples`` rows, ``min_window_benign`` benign, both
        classes) covering >= 95% of the rows, with at least 4 and at most ``max_windows`` windows;
        otherwise monthly (the conservative default)."""
        ts = corpus.timestamp[cand].astype("datetime64[D]")
        days = ts.astype("datetime64[D]").astype(object)
        day_level = len({d.day for d in days[:: max(1, len(days) // 20000)]}) > 1
        info: dict[str, Any] = {"day_level_timestamps": bool(day_level)}
        if not day_level:
            info["reason"] = "timestamps are month-level (all on the same day of the month), weekly windows would be meaningless"
            return "month", info
        wk = partition_windows(tt, corpus, cand, eff, "week")
        lab = corpus.label
        ok_rows = 0
        for w in wk:
            y = lab[w.idx_available]
            n_ben, n_mal = int(np.count_nonzero(y == 0)), int(np.count_nonzero(y == 1))
            if w.idx_available.size >= prm["min_window_samples"] and n_ben >= prm["min_window_benign"] and n_mal > 0:
                ok_rows += int(w.idx_available.size)
        cover = ok_rows / max(1, int(cand.size))
        info.update({"weekly_windows": len(wk), "weekly_rows_in_evaluable_windows": cover})
        if len(wk) < 4:
            info["reason"] = f"only {len(wk)} weekly window(s) (< 4); using month"
            return "month", info
        if len(wk) > prm["max_windows"]:
            info["reason"] = f"{len(wk)} weekly windows exceed max_windows {prm['max_windows']}; using month"
            return "month", info
        if cover < 0.95:
            info["reason"] = f"only {cover:.0%} of rows fall in weekly windows with enough samples; using month"
            return "month", info
        info["reason"] = f"day-level timestamps give {len(wk)} weekly windows, {cover:.0%} of rows in evaluable windows"
        return "week", info

    @staticmethod
    def _check(value: float | None, prm: dict[str, Any], n_windows: int, gran: str, why: str | None) -> GateCheck:
        t = prm["min_aut_f1"]
        desc = (
            f"Sample-weighted AUT(F1) over {n_windows} {_ADJECTIVE[gran]} windows after the training cutoff"
            if value is not None
            else f"Sample-weighted AUT(F1): not evaluable — {why}"
        )
        return graded_check(
            "aut_f1_weighted", value, ">=", t, ideal=min(1.0, t + 0.2), floor=max(0.0, t - 0.3), description=desc,
        )

    def _unevaluable(self, ctx: "RunContext", prm: dict[str, Any], why: str, metrics: dict[str, Any],
                     details: dict[str, Any], notes: list[str], *, n_windows: int) -> ModuleResult:
        m = {
            "n_windows": n_windows, "first_window": None, "last_window": None, "f1_first": None, "f1_last": None,
            "f1_drop": None, "aut_f1": None, "aut_f1_weighted": None,
        }
        m.update(metrics)
        details["unevaluable_reason"] = why
        finding = (
            why + " Temporal drift could not be measured, so this axis is unevaluated — not passed."
        )
        gran = details.get("granularity", prm["granularity"])
        check = self._check(None, prm, n_windows, gran if gran in _ADJECTIVE else "month", why)
        return self.result(ctx, finding=finding, checks=[check], metrics=m, details=details, notes=notes)

    @staticmethod
    def _effective_cutoff(corpus: "Corpus", declared: dt.date, hashes: frozenset[str] | None,
                          notes: list[str]) -> tuple[dt.date, dict[str, Any], np.ndarray]:
        """C1: the effective cutoff is the declared one, or the newest training-member timestamp
        if members post-date the declaration."""
        info: dict[str, Any] = {
            "rule": "every evaluation window post-dates the effective training cutoff",
            "declared_cutoff": declared.isoformat(),
            "effective_cutoff": declared.isoformat(),
            "members_in_corpus": None,
            "members_after_declared_cutoff": None,
            "latest_member_timestamp": None,
            "adjusted": False,
            "holds": True,
        }
        if not hashes:
            return declared, info, np.empty(0, dtype="datetime64[D]")
        m_idx = corpus.indices(include_hashes=hashes, labeled_only=False)
        ts = corpus.timestamp[m_idx]
        ts = ts[~np.isnat(ts)]
        info["members_in_corpus"] = int(m_idx.size)
        if ts.size == 0:
            info["members_after_declared_cutoff"] = 0
            return declared, info, ts
        latest = _to_date(ts.max())
        n_after = int(np.count_nonzero(ts > np.datetime64(declared, "D")))
        info["latest_member_timestamp"] = latest.isoformat()
        info["members_after_declared_cutoff"] = n_after
        if n_after:
            info["adjusted"] = True
            info["effective_cutoff"] = latest.isoformat()
            notes.append(
                f"Inconsistent training_cutoff: {n_after} of the {ts.size} dated training-manifest samples in the "
                f"corpus are dated after the declared cutoff {declared.isoformat()} (newest: {latest.isoformat()}). "
                f"Evaluating from the declared cutoff would overlap training data and inflate results, so M2 "
                f"uses {latest.isoformat()} as the effective cutoff. Fix training_cutoff in your adapter."
            )
            return latest, info, ts
        return declared, info, ts

    @staticmethod
    def _select(ctx: "RunContext", windows: list[Window], prm: dict[str, Any]) -> None:
        """Subsample and drop windows that are too small or lack a class (recording why)."""
        corpus = ctx.corpus
        assert corpus is not None
        for w in windows:
            ctx.check_deadline()
            n_av = int(w.idx_available.size)
            if n_av < prm["min_window_samples"]:
                w.status = f"only {n_av} samples < min_window_samples {prm['min_window_samples']}"
                w.idx = w.idx_available
            else:
                w.idx = corpus.subsample(w.idx_available, prm["max_samples_per_window"], ctx.rng)
            lab = corpus.label[w.idx]
            w.n_mal = int(np.count_nonzero(lab == 1))
            w.n_ben = int(np.count_nonzero(lab == 0))
            if w.status != "used":
                continue
            if w.n_mal == 0:
                w.status = "no malicious samples (F1 undefined)"
            elif prm["require_both_classes"] and w.n_ben == 0:
                w.status = "no benign samples (require_both_classes)"

    @staticmethod
    def _constraints(tt: ModuleType, corpus: "Corpus", used: list[Window],
                     notes: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
        """C2 (per-window malware/goodware time alignment) and C3 (per-window malware ratio)."""
        misaligned: list[str] = []
        for w in used:
            ts, lab = corpus.timestamp[w.idx], corpus.label[w.idx]
            tm, tb = ts[lab == 1], ts[lab == 0]
            if tm.size and tb.size:
                mm, mb = _median_date(tm), _median_date(tb)
                gap = abs(int(tt.month_difference(mm, mb)))
                w.c2 = {"median_malicious": mm.isoformat(), "median_benign": mb.isoformat(), "gap_months": gap,
                        "aligned": gap <= C2_MONTH_VARIANCE}
                if gap > C2_MONTH_VARIANCE:
                    misaligned.append(w.label)
            else:
                w.c2 = {"gap_months": None, "aligned": None}
        c2 = {
            "rule": (f"within each window, the median timestamps of malicious and benign samples are at most "
                     f"{C2_MONTH_VARIANCE} month apart (TESSERACT month_variance)"),
            "windows_misaligned": misaligned,
            "holds": not misaligned,
        }
        if misaligned:
            notes.append(
                f"C2: in {len(misaligned)} window(s) ({', '.join(misaligned[:5])}{', ...' if len(misaligned) > 5 else ''}) "
                "malicious and benign samples come from different sub-periods; F1 there may partly reflect "
                "sample age rather than maliciousness. Consider a finer granularity."
            )
        ratios = np.array([w.n_mal / w.n for w in used if w.n], dtype=np.float64)
        c3: dict[str, Any] = {
            "rule": f"per-window malware ratio within {C3_RATIO_TOLERANCE:.2f} of the median window",
            "median_malware_ratio": None, "min": None, "max": None, "windows_deviating": [], "holds": True,
        }
        if ratios.size:
            med = float(np.median(ratios))
            dev = [w.label for w in used if w.n and abs(w.n_mal / w.n - med) > C3_RATIO_TOLERANCE]
            c3.update({"median_malware_ratio": med, "min": float(ratios.min()), "max": float(ratios.max()),
                       "windows_deviating": dev, "holds": not dev})
            if dev:
                notes.append(
                    f"C3: malware ratio varies across windows ({ratios.min():.0%}-{ratios.max():.0%}, median "
                    f"{med:.0%}); F1 is prevalence-sensitive, so part of the change in {len(dev)} window(s) may "
                    "reflect class balance rather than model decay. Per-window recall and FPR are "
                    "prevalence-independent."
                )
            notes.append(
                f"F1 is measured at the corpus's class balance (median {med:.0%} malicious per window); "
                "at a lower production prevalence, precision and F1 would be lower."
            )
        return c2, c3

    def _charts(self, ctx: "RunContext", considered: list[Window], used: list[Window], eff: dt.date,
                gran: str, prm: dict[str, Any], max_fpr: float = DEFAULT_MAX_FPR) -> None:
        if used:
            x = [w.k for w in used]
            series = [
                {"label": "F1", "x": x, "y": [w.metrics["f1"] for w in used]},
                {"label": "precision", "x": x, "y": [w.metrics["precision"] for w in used]},
                {"label": "recall (detection rate)", "x": x, "y": [w.metrics["recall"] for w in used]},
            ]
            ctx.artifacts.add_chart(
                self.id, "decay", title="Performance decay after the training cutoff", series=series, kind="line",
                xlabel=f"{gran}s after the training cutoff ({eff.isoformat()})",
                ylabel="at the operating threshold", ylim=(0.0, 1.0),
                reference_lines=[{"axis": "y", "value": prm["min_aut_f1"], "label": f"min_aut_f1 {prm['min_aut_f1']:g}"}],
                note="; ".join(f"{w.k}={w.label}" for w in used[:48]),
            )
        if used:
            x = [w.k for w in used]
            xl = f"{gran}s after the training cutoff ({eff.isoformat()})"
            ev = [w for w in used if w.n_ben >= prm["min_window_benign"] and w.metrics.get("fpr_ci95")]
            if ev:
                xe = [w.k for w in ev]
                top = max([w.metrics["fpr_ci95"][1] for w in ev] + [max_fpr]) * 1.15
                ctx.artifacts.add_chart(
                    self.id, "window_fpr", title="Realised FPR at the frozen threshold (own scale)",
                    series=[
                        {"label": "FPR", "x": xe, "y": [w.metrics["fpr"] for w in ev]},
                        {"label": "FPR 95% CI upper", "x": xe, "y": [w.metrics["fpr_ci95"][1] for w in ev]},
                        {"label": "FPR 95% CI lower", "x": xe, "y": [w.metrics["fpr_ci95"][0] for w in ev]},
                    ],
                    kind="line", xlabel=xl, ylabel="false-positive rate", ylim=(0.0, float(top)),
                    reference_lines=[{"axis": "y", "value": max_fpr, "label": f"max_fpr {max_fpr:g} (M1 budget)"}],
                    note="; ".join(f"{w.k}={w.label}" for w in ev[:48]),
                )
            evm = [w for w in used if w.metrics.get("fnr_ci95")]
            if evm:
                xm = [w.k for w in evm]
                ctx.artifacts.add_chart(
                    self.id, "window_miss_rate", title="Miss rate (FNR) at the frozen threshold (own scale)",
                    series=[
                        {"label": "FNR (1 - recall)", "x": xm, "y": [w.metrics["fnr"] for w in evm]},
                        {"label": "FNR 95% CI upper", "x": xm, "y": [w.metrics["fnr_ci95"][1] for w in evm]},
                        {"label": "FNR 95% CI lower", "x": xm, "y": [w.metrics["fnr_ci95"][0] for w in evm]},
                    ],
                    kind="line", xlabel=xl, ylabel="miss rate",
                    ylim=(0.0, float(max(w.metrics["fnr_ci95"][1] for w in evm) * 1.15) or 1.0),
                    note="; ".join(f"{w.k}={w.label}" for w in evm[:48]),
                )
        if considered:
            labels = [w.label for w in considered]
            lab = ctx.corpus.label  # type: ignore[union-attr]
            mal = [int(np.count_nonzero(lab[w.idx_available] == 1)) for w in considered]
            ben = [int(np.count_nonzero(lab[w.idx_available] == 0)) for w in considered]
            dropped = [w.label for w in considered if not w.used]
            ctx.artifacts.add_chart(
                self.id, "window_sizes", title="Samples per window", kind="bar",
                series=[{"label": "malicious", "x": labels, "y": mal}, {"label": "benign", "x": labels, "y": ben}],
                xlabel="window", ylabel="samples",
                reference_lines=[{"axis": "y", "value": prm["min_window_samples"], "label": "min_window_samples"}],
                note=("dropped: " + ", ".join(dropped)) if dropped else "",
            )
            ctx.artifacts.add_table(
                self.id, "windows", title="Per-window results at the operating threshold",
                columns=["window", "n", "malicious", "benign", "malware ratio", "precision", "recall", "F1", "AUROC",
                         "FPR", "FPR 95% CI", "miss rate", "miss rate 95% CI", "status"],
                rows=[[r["window"], r["n"], r["n_mal"], r["n_ben"], r["malware_ratio"], r["precision"], r["recall"],
                       r["f1"], r["auroc"], r["fpr"], _ci_txt(r["fpr_ci95"]), r["fnr"], _ci_txt(r["fnr_ci95"]),
                       r["status"]] for r in (w.row() for w in considered)],
            )

    @staticmethod
    def _finding(m: dict[str, Any], prm: dict[str, Any], eff: dt.date, gran: str, check: GateCheck) -> str:
        t = prm["min_aut_f1"]
        drop = m["f1_drop"]
        trend = f"drops by {drop:.3f}" if drop > 0 else (f"rises by {-drop:.3f}" if drop < 0 else "is unchanged")
        s = (
            f"Over {m['n_windows']} {_ADJECTIVE[gran]} windows after the training cutoff ({eff.isoformat()}), "
            f"sample-weighted AUT(F1) is {m['aut_f1_weighted']:.3f} (min {t:.2f}; unweighted {m['aut_f1']:.3f}). "
            f"F1 at the operating threshold {trend} from {m['f1_first']:.3f} ({m['first_window']}) to "
            f"{m['f1_last']:.3f} ({m['last_window']})."
        )
        if check.passed and drop >= 0.05:
            s += (f" AUT meets the drift policy, but F1 still fell by {drop:.3f} over the evaluated period: budget "
                  "for periodic retraining.")
        elif check.passed:
            s += " Detection quality holds up on samples newer than the training data."
        else:
            s += (" Detection degrades on samples newer than the training data faster than policy allows: expect "
                  "to retrain often (and monitor drift) if this model is deployed.")
        return s


__all__ = [
    "DriftModule",
    "aut_unweighted",
    "aut_sample_weighted",
    "train_test_temporally_consistent",
    "binary_counts",
    "auroc",
    "operating_point_stats",
    "fpr_budget_summary",
    "partition_windows",
    "load_temporal",
    "Window",
]
