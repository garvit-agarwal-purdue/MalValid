"""Generate realistic ``report.json`` fixtures (schema ``malvalid-report/1``) for the HTML report tests.

The fixtures are built from malvalid's real result types (:class:`~malvalid.core.ModuleResult`,
:meth:`~malvalid.core.GateCheck.evaluate` with the §5 score anchors,
:class:`~malvalid.context.ArtifactStore`, :func:`~malvalid.verdict.compute_verdict`), so their shape
is exactly what the runner writes. The numbers are synthetic but plausible; nothing here is derived
from real malware.

Variants (``VARIANTS``):

* ``blocked``     – the kitchen sink written to ``sample_report.json``: M1 hard gate fails, M6
  errors (with a traceback and a partial chart), a third-party plugin is skipped, screening
  badges, every §5 chart, and long / HTML-special strings to exercise escaping.
* ``ready``       – everything passes.
* ``conditional`` – a warn gate fails and M4 is skipped (no training manifest).
* ``not_ready``   – no hard failure, but the weighted score is below the CONDITIONAL band.
* ``aborted``     – M0 finds a pickle without ``--allow-pickle``; every other module is skipped.

Usage::

    python tests/fixtures/make_sample_report.py            # writes sample_report*.json here
    python tests/fixtures/make_sample_report.py --out DIR
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from malvalid.config import load_config
from malvalid.context import ArtifactStore
from malvalid.core import GateCheck, GateMode, GateOutcome, ModuleResult, Status, derive_status
from malvalid.scoring import combine_axis
from malvalid.verdict import compute_verdict, exit_code_for

# Copy of malvalid.runner.EXIT_MEANINGS (kept literal so the fixtures stay byte-stable).
EXIT_MEANINGS = {
    0: "passed: no hard gate failed, no module errored, verdict above --fail-on",
    1: "gate failed: the run is BLOCKED (a hard gate failed / could not be evaluated / M0 aborted) "
    "or the verdict is at or below --fail-on",
    2: "error: a module errored or the model could not be loaded — the gate result is not trustworthy",
}

VARIANTS = ("blocked", "ready", "conditional", "not_ready", "aborted")
FIXTURE_DIR = Path(__file__).resolve().parent

# Strings that must come out of the renderer escaped (used by the ``blocked`` kitchen sink).
XSS_SCRIPT = '<script>alert("xss")</script>'
XSS_IMG = "<img src=x onerror=alert(1)>"
XSS_BREAKOUT = "</script><script>alert('breakout')</script>"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# --------------------------------------------------------------------------------------------------
# Result builder mirroring Module.result / Module.skip (no RunContext needed)
# --------------------------------------------------------------------------------------------------


def _result(
    module_id: str,
    code: str,
    title: str,
    gate: GateMode,
    *,
    finding: str,
    checks: list[GateCheck],
    metrics: dict[str, Any],
    store: ArtifactStore,
    params: dict[str, Any],
    details: dict[str, Any] | None = None,
    notes: list[str] | None = None,
    screening: bool = False,
    score: float | None = None,
    advisory_warn: bool = False,
    seed: int = 0,
    duration_s: float = 1.0,
) -> ModuleResult:
    status, outcome = derive_status(checks, gate, advisory_warn=advisory_warn)
    if score is None:
        score = combine_axis([c.score for c in checks], "min")
    return ModuleResult(
        module_id=module_id,
        code=code,
        title=title,
        status=status,
        gate=gate,
        gate_outcome=outcome,
        finding=finding,
        metrics=metrics,
        checks=checks,
        params=params,
        details=details or {},
        artifacts=store.keys_for(module_id),
        notes=notes or [],
        screening=screening,
        score=score,
        seed=seed,
        duration_s=duration_s,
    )


def _skipped(
    module_id: str, code: str, title: str, gate: GateMode, reason: str, params: dict[str, Any],
    *, screening: bool = False, seed: int | None = None,
) -> ModuleResult:
    return ModuleResult(
        module_id=module_id,
        code=code,
        title=title,
        status=Status.SKIPPED,
        gate=gate,
        gate_outcome=GateOutcome.NOT_EVALUATED,
        finding=f"Not run — {reason}",
        params=params,
        screening=screening,
        skip_reason=reason,
        seed=seed,
        duration_s=0.0,
    )


def _seed(module_id: str) -> int:
    from malvalid.context import module_seed

    return module_seed(0, module_id)


# --------------------------------------------------------------------------------------------------
# Module result factories
# --------------------------------------------------------------------------------------------------


def m0_file_safety(store: ArtifactStore, *, pickle_abort: bool = False, artifact: str = "") -> ModuleResult:
    params = {"fail_on": "HIGH", "scan_adapter_dir": True}
    fs_cols = ["file", "format", "pickle", "declared", "sha256 (prefix)", "modelscan", "worst finding"]
    if pickle_abort:
        store.add_table("file_safety", "artifacts", title="Scanned model artifacts", columns=fs_cols,
                        rows=[[Path(artifact).name, "Python pickle", "yes", "yes", _sha(artifact)[:16], "scanned", "MEDIUM"]])
        store.add_table("file_safety", "findings", title="File-safety findings",
                        columns=["file", "severity", "finding", "reported by"],
                        rows=[[Path(artifact).name, "MEDIUM", "pickle protocol 4 stream; loading executes code "
                               "(GLOBAL sklearn.ensemble._gb)", "modelscan, malvalid opcode scan"]])
        checks = [
            GateCheck.evaluate("n_critical", 0, "<=", 0, description="no CRITICAL scanner findings"),
            GateCheck.evaluate("pickle_refused", 1, "==", 0, description="no pickle artifact without --allow-pickle"),
        ]
        return _result(
            "file_safety", "M0", "Model file safety", GateMode.HARD,
            finding=(
                f"{Path(artifact).name} is a pickle and --allow-pickle was not given, so the model was never "
                "loaded. Re-export it in a non-executable format (LightGBM: booster.save_model('model.txt'); "
                "XGBoost: save_model('model.ubj'); sklearn: skl2onnx) and re-run."
            ),
            checks=checks,
            metrics={"n_artifacts": 1, "n_pickle": 1, "n_critical": 0, "n_high": 0, "n_medium": 0, "n_low": 0,
                     "scanner": "modelscan", "scanner_version": "0.8.8"},
            details={"abort": True, "artifacts": [{
                "path": artifact, "format": "pickle", "is_pickle": True, "scanned": True,
                "findings": [{"severity": "MEDIUM", "description": "pickle protocol 4 stream; loading executes code",
                              "operator": "GLOBAL", "module": "sklearn.ensemble._gb"}],
            }]},
            store=store, params=params, score=0.0, seed=_seed("file_safety"), duration_s=0.41,
        )
    store.add_table("file_safety", "artifacts", title="Scanned model artifacts", columns=fs_cols, rows=[
        ["ember_lgbm.txt", "LightGBM text model", "no", "yes", _sha("model/ember_lgbm.txt")[:16],
         "not scanned: modelscan has no scanner for LightGBM text dumps", "none"],
        ["requirements.lock", "text", "no", "no (found in adapter dir)", _sha("adapter/requirements.lock")[:16],
         "not scanned: not a model artifact", "none"],
    ])
    checks = [
        GateCheck.evaluate("n_critical", 0, "<=", 0, description="no CRITICAL scanner findings"),
        GateCheck.evaluate("n_at_or_above_fail_on", 0, "<=", 0, description="no findings at or above HIGH"),
    ]
    return _result(
        "file_safety", "M0", "Model file safety", GateMode.HARD,
        finding="2 artifacts scanned before load (modelscan 0.8.8 + opcode scan): no findings. No pickle artifacts.",
        checks=checks,
        metrics={"n_artifacts": 2, "n_pickle": 0, "n_critical": 0, "n_high": 0, "n_medium": 0, "n_low": 0,
                 "scanner": "modelscan", "scanner_version": "0.8.8"},
        details={"artifacts": [
            {"path": "model/ember_lgbm.txt", "format": "lightgbm_text", "is_pickle": False, "scanned": False,
             "reason": "modelscan has no scanner for LightGBM text dumps (plain text, not executable)", "findings": []},
            {"path": "adapter/requirements.lock", "format": "text", "is_pickle": False, "scanned": False,
             "reason": "not a model artifact", "findings": []},
        ]},
        store=store, params=params, score=1.0, seed=_seed("file_safety"), duration_s=0.83,
    )


def _roc_from_scores(rng: np.random.Generator, sep: float, n: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Mostly well-separated classes plus a "hard" tail (packed benign installers, evasive malware),
    # so the curves look like a real detector's rather than a perfect separator.
    hard = min(0.2, 0.06 / sep)
    ben = np.where(rng.random(n) < hard, rng.beta(2.0, 2.6, n), rng.beta(0.6, 9.0 * sep, n))
    mal = np.where(rng.random(n) < 1.4 * hard, rng.beta(2.2, 1.8, n), rng.beta(9.0 * sep, 0.7, n))
    y = np.r_[np.zeros(n), np.ones(n)]
    s = np.r_[ben, mal]
    return ben, mal, y, s


def m1_performance(store: ArtifactStore, rng: np.random.Generator, *, fpr: float, dr: float, sep: float,
                   threshold: float, challenge: float | None = None, hard: bool = True) -> ModuleResult:
    from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

    ben, mal, y, s = _roc_from_scores(rng, sep)
    f, t, thr = roc_curve(y, s)
    keep = f > 0
    auroc = float(roc_auc_score(y, s))
    mid = "performance"
    store.add_chart(
        mid, "roc", title="ROC curve (training members excluded)", kind="line",
        series=[{"label": f"ROC (AUROC {auroc:.4f})", "x": f[keep], "y": t[keep]},
                {"label": f"operating point (t = {threshold:g})", "x": [fpr], "y": [dr]}],
        xlabel="False-positive rate (log)", ylabel="Detection rate (TPR)", xscale="log",
        xlim=(1e-5, 1.0), ylim=(0.0, 1.0),
        reference_lines=[{"axis": "x", "value": 0.01, "label": "max_fpr 0.01"},
                         {"axis": "y", "value": 0.95, "label": "min_detection 0.95"}],
        note="An FPR of 0 is drawn at 1e-05 on the log axis.",
    )
    def _op(tgt: float) -> list[Any]:
        i = int(np.searchsorted(f, tgt, side="right")) - 1
        return [round(float(thr[max(i, 0)]), 4) if np.isfinite(thr[max(i, 0)]) else None, float(f[max(i, 0)]), float(t[max(i, 0)])]
    store.add_table(mid, "operating_points", title="Operating points on the canonical eval set",
                    columns=["operating point", "threshold", "FPR", "detection rate"],
                    rows=[["declared operating threshold", threshold, fpr, dr],
                          ["FPR <= 0.01 (max_fpr)", *_op(0.01)], ["FPR <= 0.001", *_op(0.001)]],
                    note="Classification rule: score >= threshold.")
    p, r, _ = precision_recall_curve(y, s)
    store.add_chart(mid, "pr", title="Precision–recall", kind="line",
                    series=[{"label": "PR", "x": r[::-1], "y": p[::-1]}],
                    xlabel="Recall", ylabel="Precision", xlim=(0.0, 1.0), ylim=(0.0, 1.02))
    bins = np.linspace(0, 1, 16)
    idx = np.clip(np.digitize(s, bins) - 1, 0, 14)
    mp = np.array([s[idx == k].mean() if np.any(idx == k) else np.nan for k in range(15)])
    fp = np.array([y[idx == k].mean() if np.any(idx == k) else np.nan for k in range(15)])
    ok = np.isfinite(mp)
    store.add_chart(mid, "calibration", title="Reliability diagram (15 bins)", kind="line",
                    series=[{"label": "model", "x": mp[ok], "y": fp[ok]}],
                    xlabel="Mean predicted score", ylabel="Fraction malicious", xlim=(0, 1), ylim=(0, 1),
                    note="diagonal")
    edges = np.linspace(0, 1, 51)
    hb, _ = np.histogram(ben, edges)
    hm, _ = np.histogram(mal, edges)
    store.add_chart(mid, "score_hist", title="Score distribution by class", kind="step",
                    series=[{"label": "benign", "x": edges[:-1], "y": np.maximum(hb, 0.5)},
                            {"label": "malicious", "x": edges[:-1], "y": np.maximum(hm, 0.5)}],
                    xlabel="Model score", ylabel="Samples per bin (log)", yscale="log", xlim=(0, 1),
                    reference_lines=[{"axis": "x", "value": threshold, "label": f"operating threshold {threshold:g}"}])
    brier = float(np.mean((s - y) ** 2))
    ece = float(np.nansum(np.abs(mp - fp) * np.bincount(idx, minlength=15) / s.size))
    n_ben, n_mal = 99_812, 99_945
    checks = [
        GateCheck.evaluate("fpr", fpr, "<=", 0.01, description="false-positive rate on benign eval rows at the operating threshold",
                           ideal=0.001, floor=0.05, scale="log"),
        GateCheck.evaluate("detection_rate", dr, ">=", 0.95, description="detection rate on malicious eval rows at the operating threshold",
                           ideal=0.95 + 0.9 * 0.05, floor=0.80),
    ]
    metrics: dict[str, Any] = {
        "detection_rate": dr, "fpr": fpr, "n_benign": n_ben, "n_malicious": n_mal, "threshold": threshold,
        "auroc": auroc, "auprc": float(np.trapezoid(p[::-1], r[::-1])), "brier": brier, "ece": ece,
        "tpr_at_fpr_0.001": float(np.interp(1e-3, f, t)), "tpr_at_fpr_0.01": float(np.interp(1e-2, f, t)),
        "threshold_at_max_fpr": round(threshold + 0.07, 4),
        "detection_rate_ci95": [round(dr - 0.0012, 4), round(dr + 0.0011, 4)],
        "fpr_ci95": [round(fpr * 0.93, 5), round(fpr * 1.07, 5)],
    }
    if challenge is not None:
        metrics["challenge_detection_rate"] = challenge
    status_word = "passes" if all(c.passed for c in checks) else "fails"
    finding = (
        f"At the declared operating threshold {threshold:g}, FPR is {fpr:.4f} (max {0.01:g}) and detection rate is "
        f"{dr:.4f} (min {0.95:g}) on {n_ben:,} benign and {n_mal:,} malicious held-out rows; the hard gate {status_word}."
    )
    notes = [f"1,204 training-manifest members were excluded from the evaluation sets."]
    if challenge is not None:
        notes.append(f"Challenge split (informational, not gated): detection rate {challenge:.3f} on 6,315 evasive samples.")
    return _result(mid, "M1", "Performance & calibration", GateMode.HARD if hard else GateMode.WARN,
                   finding=finding, checks=checks, metrics=metrics, store=store,
                   params={"max_fpr": 0.01, "min_detection": 0.95, "max_samples_per_class": None,
                           "calibration_bins": 15, "include_challenge": True},
                   notes=notes, seed=_seed(mid), duration_s=48.2)


def m2_drift(store: ArtifactStore, rng: np.random.Generator, *, start_f1: float, decay: float,
             months: list[str], cutoff: str, min_aut: float = 0.70) -> ModuleResult:
    mid = "drift"
    k = len(months)
    n_mal = rng.integers(900, 4200, k)
    n_ben = rng.integers(900, 4200, k)
    n_mal[2] = 240  # a sparse window: shows why sample weighting matters
    f1 = np.clip(start_f1 - decay * np.arange(k) + rng.normal(0, 0.012, k), 0, 1)
    prec = np.clip(f1 + rng.normal(0.02, 0.01, k), 0, 1)
    rec = np.clip(f1 * prec / np.maximum(2 * prec - f1, 1e-9), 0, 1)  # from F1 = 2pr / (p + r)
    x = np.arange(1, k + 1)
    store.add_chart(mid, "decay", title="Performance decay after the training cutoff", kind="line",
                    series=[{"label": "F1", "x": x, "y": f1}, {"label": "precision", "x": x, "y": prec},
                            {"label": "recall", "x": x, "y": rec}],
                    xlabel=f"months after the training cutoff ({cutoff})", ylabel="at the operating threshold", ylim=(0, 1),
                    reference_lines=[{"axis": "y", "value": min_aut, "label": f"min_aut_f1 {min_aut:g}"}],
                    note="; ".join(f"{i + 1}={m}" for i, m in enumerate(months)))
    store.add_chart(mid, "window_sizes", title="Samples per evaluation window", kind="bar",
                    series=[{"label": "malicious", "x": [*months, "2025-01"], "y": [*n_mal, 23]},
                            {"label": "benign", "x": [*months, "2025-01"], "y": [*n_ben, 38]}],
                    xlabel="window", ylabel="samples",
                    reference_lines=[{"axis": "y", "value": 200, "label": "min_window_samples"}],
                    note="dropped: 2025-01")
    fpr_w = np.clip(0.004 + 0.0009 * np.arange(k) + rng.normal(0, 0.0005, k), 0, 1)
    store.add_table(mid, "windows", title="Per-window metrics at the operating threshold",
                    columns=["window", "n", "malicious", "benign", "malware ratio", "precision", "recall", "F1", "FPR", "status"],
                    rows=[[months[i], int(n_mal[i] + n_ben[i]), int(n_mal[i]), int(n_ben[i]),
                           round(float(n_mal[i] / (n_mal[i] + n_ben[i])), 4), round(float(prec[i]), 4),
                           round(float(rec[i]), 4), round(float(f1[i]), 4), round(float(fpr_w[i]), 5), "used"] for i in range(k)]
                         + [["2025-01", 61, 23, 38, 0.377, None, None, None, None, "dropped: below min_window_samples (200)"]])
    n = n_mal + n_ben
    aut = float(np.trapezoid(f1) / (k - 1))
    pair_w = n[:-1] + n[1:]
    aut_w = float(np.sum((n[:-1] * f1[:-1] + n[1:] * f1[1:]) / pair_w * pair_w) / np.sum(pair_w))
    check = GateCheck.evaluate("aut_f1_weighted", aut_w, ">=", min_aut, description="sample-weighted AUT(F1) over post-cutoff windows",
                               ideal=min(1.0, min_aut + 0.2), floor=max(0.0, min_aut - 0.3))
    finding = (f"Sample-weighted AUT(F1) over {k} monthly windows after the {cutoff} cutoff is {aut_w:.3f} "
               f"(min {min_aut:g}); F1 falls from {f1[0]:.3f} to {f1[-1]:.3f}.")
    return _result(mid, "M2", "Temporal drift", GateMode.WARN, finding=finding, checks=[check],
                   metrics={"n_windows": k, "first_window": months[0], "last_window": months[-1],
                            "f1_first": float(f1[0]), "f1_last": float(f1[-1]), "f1_drop": float(f1[0] - f1[-1]),
                            "aut_f1": aut, "aut_f1_weighted": aut_w, "effective_cutoff": cutoff},
                   details={"dropped_windows": [{"window": "2025-01", "n": 61, "reason": "below min_window_samples (200)"}],
                            "temporal_constraints": {"C1_windows_after_cutoff": True, "C2_time_alignment": "ok",
                                                     "C3_malware_ratio": [round(float(v), 3) for v in n_mal / n]}},
                   store=store, params={"min_aut_f1": min_aut, "granularity": "month", "min_window_samples": 200,
                                        "max_windows": 36, "max_samples_per_window": None, "require_both_classes": True},
                   notes=["Windows with fewer than 200 samples or a missing class are dropped and listed under details."],
                   seed=_seed(mid), duration_s=31.7)


def m4_membership(store: ArtifactStore, rng: np.random.Generator, *, adv: float) -> ModuleResult:
    mid = "membership_inf"
    series = []
    rows = []
    for name, a in (("loss-threshold attack", adv * 0.7), ("ART black-box attack (random forest)", adv),
                    ("ART black-box attack (gradient boosting)", adv * 0.9)):
        fx = np.linspace(0, 1, 60)
        gamma = max(0.2, 1 - 4 * a)
        ty = fx ** gamma
        series.append({"label": name, "x": fx, "y": ty})
        rows.append([name, round(a, 4), round(a * 1.18, 4), round(0.5 + a * 1.1, 4), round(0.01 + a * 0.25, 4)])
    store.add_chart(mid, "attack_roc", title="Membership-inference attack ROC", kind="line", series=series,
                    xlabel="Attack FPR (non-members flagged)", ylabel="Attack TPR (members found)",
                    xlim=(0, 1), ylim=(0, 1), reference_lines=[{"axis": "x", "value": 0.01, "label": "1% FPR"}],
                    note="diagonal")
    edges = np.linspace(-8.0, 2.0, 41)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mem = np.clip(rng.normal(-3.4 - 4 * adv, 1.1, 5000), -8, 2)
    non = np.clip(rng.normal(-3.1, 1.2, 5000), -8, 2)
    store.add_chart(mid, "member_score_hist",
                    title="Model loss on the true label: training members vs matched non-members", kind="step",
                    series=[{"label": "training members", "x": centers, "y": np.histogram(mem, edges)[0] / 5000},
                            {"label": "non-members", "x": centers, "y": np.histogram(non, edges)[0] / 5000}],
                    xlabel="log10 cross-entropy loss on the true label (lower = more confident)",
                    ylabel="fraction of samples",
                    reference_lines=[{"axis": "x", "value": -3.9, "label": "loss-attack threshold"}])
    store.add_table(mid, "attacks", title="Per-attack results (held-out half)",
                    columns=["attack", "advantage", "best-threshold advantage", "attack AUROC", "TPR @ 1% FPR"], rows=rows)
    check = GateCheck.evaluate("advantage", adv, "<=", 0.10, description="worst-case membership advantage (TPR − FPR) over attacks",
                               ideal=0.01, floor=0.30)
    return _result(mid, "M4", "Membership inference", GateMode.WARN,
                   finding=(f"Worst-case membership advantage is {adv:.3f} (max 0.10) across 3 black-box attacks on "
                            "5,000 time-matched members and 5,000 non-members."),
                   checks=[check],
                   metrics={"advantage": adv, "best_threshold_advantage": adv * 1.18, "attack_auroc": 0.5 + adv * 1.1,
                            "tpr_at_1pct_fpr": 0.01 + adv * 0.25, "n_members": 5000, "n_nonmembers": 5000,
                            "time_matched": True},
                   store=store, params={"max_advantage": 0.10, "max_per_side": 5000, "min_members": 200,
                                        "test_fraction": 0.5, "time_matched": True},
                   notes=["Non-members are time-matched to members by month and label, so distribution shift does not inflate the advantage."],
                   seed=_seed(mid), duration_s=112.4)


def m5_backdoor(store: ArtifactStore, *, flagged: int, top_suspicion: float, escape: bool = False) -> ModuleResult:
    mid = "backdoor_screen"
    rules = []
    feats = ["header.coff.timestamp", "section.entry_name_hashed[41]", "strings.numstrings", "general.has_signature",
             "imports.functions_hashed[17]", "datadirectories.debug_size"]
    for i in range(8):
        susp = max(0.0, top_suspicion - 0.11 * i)
        cond = f"{feats[i % 6]} <= {0.5 + i:.1f} && {feats[(i + 2) % 6]} > {1.6e9 if i == 0 else 3.0:g}"
        if escape and i == 0:
            cond += f" {XSS_SCRIPT}"
        rules.append([f"tree {17 + 9 * i} / leaf {3 + i}", cond, round(-4.2 + 0.4 * i, 3), 3 + 5 * i, round(susp, 3),
                      "yes" if susp >= 0.5 else "no"])
    store.add_table(mid, "suspicious_rules", title="Most suspicious root→leaf paths (static scan)",
                    columns=["path", "conditions", "leaf value", "cover", "suspicion", "flagged"], rows=rows_sorted(rules),
                    note="Suspicion combines a strongly benign leaf, a narrow conjunction on low-importance features, and low training cover.")
    scan = GateCheck.evaluate("n_flagged_rules", flagged, "<=", 0,
                              description="root→leaf paths with suspicion ≥ min_suspicion (0.5)")
    scan.score = 1.0 if flagged == 0 else max(0.0, 0.74 * (1 - top_suspicion) / 0.5)
    checks = [scan]
    finding = (f"The static scan flagged {flagged} of 48,113 root→leaf paths as backdoor-like (top suspicion {top_suspicion:.2f}); "
               "no operator trigger hypotheses were supplied." if flagged else
               "The static scan flagged no backdoor-like root→leaf paths (top suspicion "
               f"{top_suspicion:.2f} < 0.5); no operator trigger hypotheses were supplied.")
    return _result(mid, "M5", "Backdoor / poisoning screen", GateMode.WARN, finding=finding, checks=checks,
                   metrics={"n_paths_scanned": 48113, "n_flagged_rules": flagged, "top_suspicion": top_suspicion,
                            "n_triggers_tested": 0},
                   details={"art_poisoning_detectors": "not applicable: ART's activation/spectral defences target neural networks, not tree ensembles",
                            "flagged_features": {"header.coff.timestamp": 2, "section.entry_name_hashed[41]": 1} if flagged else {}},
                   store=store, params={"triggers": [], "triggers_path": None, "max_trigger_drop": 0.2, "n_samples": 2000,
                                        "max_rules_reported": 50, "low_importance_quantile": 0.5, "min_suspicion": 0.5},
                   notes=["Absence of findings is not proof that the model is backdoor-free; certification needs training-time access."],
                   screening=True, seed=_seed(mid), duration_s=22.9)


def rows_sorted(rows: list[list[Any]]) -> list[list[Any]]:
    return sorted(rows, key=lambda r: -float(r[4]))


def m6_extraction_error(store: ArtifactStore) -> ModuleResult:
    mid = "extraction"
    budgets = np.array([100, 1000, 10000])
    store.add_chart(mid, "fidelity_vs_queries", title="Surrogate fidelity vs query budget", kind="line",
                    series=[{"label": "fidelity (agreement with the model)", "x": budgets, "y": [0.861, 0.934, 0.962]},
                            {"label": "surrogate accuracy on eval rows", "x": budgets, "y": [0.842, 0.917, 0.948]}],
                    xlabel="Queries (log)", ylabel="Rate", xscale="log", ylim=(0.5, 1.0),
                    reference_lines=[{"axis": "y", "value": 0.95, "label": "max_fidelity = 0.95"},
                                     {"axis": "x", "value": 10000, "label": "fidelity_budget"}])
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/home/user/malvalid/src/malvalid/runner.py", line 412, in _run_module\n'
        "    result = module.run(ctx)\n"
        '  File "/home/user/malvalid/src/malvalid/modules/extraction.py", line 188, in run\n'
        "    labels = ctx.score(Xq) >= ctx.threshold\n"
        '  File "/home/user/malvalid/src/malvalid/sandbox/host.py", line 301, in predict_proba\n'
        "    raise SandboxError(f\"worker exited with signal {sig} <SIGKILL>\\n{tail}\")\n"
        "malvalid.core.SandboxError: worker exited with signal 9 <SIGKILL> while scoring 50000 rows "
        "(RLIMIT_AS=32768 MB). Tail of worker.log:\n"
        "  MemoryError: Unable to allocate 1.14 GiB for an array with shape (50000, 2568) & dtype float64\n"
    )
    return ModuleResult(
        module_id=mid, code="M6", title="Model extraction susceptibility", status=Status.ERROR, gate=GateMode.WARN,
        gate_outcome=GateOutcome.NOT_EVALUATED,
        finding="module crashed: SandboxError: worker exited with signal 9 <SIGKILL> while scoring 50000 rows",
        params={"query_budgets": [100, 1000, 10000, 50000], "fidelity_budget": 10000, "max_fidelity": 0.95,
                "surrogate": "lightgbm", "n_eval": 5000},
        artifacts=[], error=tb, seed=_seed(mid), duration_s=402.6,
    )


def m6_extraction_ok(store: ArtifactStore, *, fid: float) -> ModuleResult:
    mid = "extraction"
    budgets = np.array([100, 1000, 10000, 50000])
    fids = np.array([fid - 0.11, fid - 0.04, fid, fid + 0.01])
    store.add_chart(mid, "fidelity_vs_queries", title="Surrogate fidelity vs query budget", kind="line",
                    series=[{"label": "fidelity (agreement with the model)", "x": budgets, "y": fids},
                            {"label": "surrogate accuracy (vs true labels)", "x": budgets, "y": fids - 0.02}],
                    xlabel="queries to the model (hard verdicts)", ylabel="agreement on held-out eval rows",
                    xscale="log", ylim=(0.0, 1.0),
                    reference_lines=[{"axis": "y", "value": 0.95, "label": "max_fidelity 0.95"},
                                     {"axis": "x", "value": 10000.0, "label": "fidelity_budget 10000"},
                                     {"axis": "y", "value": 0.981, "label": "your model's accuracy"}])
    check = GateCheck.evaluate("fidelity_at_budget", fid, "<=", 0.95, description="surrogate agreement at 10,000 queries",
                               ideal=0.80, floor=0.995)
    return _result(mid, "M6", "Model extraction susceptibility", GateMode.WARN,
                   finding=f"A LightGBM surrogate trained on 10,000 hard-label queries agrees with the model on {fid:.3f} of held-out rows (max 0.95).",
                   checks=[check], metrics={"fidelity_at_budget": fid, "fidelity_budget": 10000, "surrogate": "lightgbm",
                                             "extraction_api": "direct query-and-fit (ART CopycatCNN/KnockoffNets need a neural surrogate)"},
                   store=store, params={"query_budgets": [100, 1000, 10000, 50000], "fidelity_budget": 10000,
                                        "max_fidelity": 0.95, "surrogate": "lightgbm", "n_eval": 5000},
                   notes=["Only relevant if the model will be exposed as a queryable service."],
                   seed=_seed(mid), duration_s=88.0)


def m7_explanation(store: ArtifactStore, rng: np.random.Generator, *, ctrl: float, top: float,
                   escape: bool = False, dim_label: str = "ember_v2") -> ModuleResult:
    mid = "explanation"
    names = [
        "strings.printables", "general.size", "header.optional.sizeof_code", "byteentropy[113]",
        "imports.functions_hashed[17]", "section.entropy_hashed[3]", "header.coff.timestamp", "histogram[0]",
        "strings.numstrings", "general.imports", "datadirectories[6].size", "section.entry_name_hashed[41]",
        "header.optional.major_linker_version", "strings.entropy", "exports.hashed[88]", "general.has_signature",
        "byteentropy[240]", "imports.libraries_hashed[5]", "header.optional.subsystem", "strings.paths",
    ]
    ctl = ["controllable", "derived", "append_only", "derived", "controllable", "derived", "controllable",
           "append_only", "append_only", "derived", "derived", "controllable", "controllable", "derived", "controllable",
           "fixed", "derived", "controllable", "fixed", "append_only"]
    raw = np.sort(rng.pareto(1.6, len(names)) + 0.05)[::-1]
    rest = raw[1:] / raw[1:].sum() * max(0.05, 0.8 - top)  # top-20 carry ~80% of the |SHAP| mass
    shares = np.r_[top, np.minimum(rest, 0.95 * top)]
    if escape:
        names[7] = f"histogram[0] {XSS_SCRIPT}"
    groups = ["strings", "general", "header", "byteentropy", "imports", "section", "histogram", "datadirectories", "exports"]
    gshap = np.array([top + 0.05, 0.14, 0.13, 0.12, 0.09, 0.08, 0.05, 0.03, 0.01])
    gshap = gshap / gshap.sum()
    ggain = np.clip(gshap + rng.normal(0, 0.02, gshap.size), 0.002, None)
    ggain = ggain / ggain.sum()
    store.add_chart(mid, "top_features", title="Top 20 features by mean |SHAP| share", kind="bar",
                    series=[{"label": "share of |SHAP| mass", "x": names, "y": shares}],
                    xlabel="Feature", ylabel="Share of |SHAP| mass")
    store.add_chart(mid, "group_importance", title="Importance by feature group", kind="bar",
                    series=[{"label": "mean |SHAP| share", "x": groups, "y": gshap},
                            {"label": "tree gain share", "x": groups, "y": ggain}],
                    xlabel="Feature group", ylabel="Share")
    store.add_table(mid, "top_features_table", title="Top features and how freely a file author can change them",
                    columns=["rank", "feature", "group", "controllability", "|SHAP| share"],
                    rows=[[i + 1, n, n.split(".")[0].split("[")[0], c, round(float(s), 4)]
                          for i, (n, c, s) in enumerate(zip(names, ctl, shares))])
    herf = float(np.sum(shares ** 2))
    checks = [
        GateCheck.evaluate("controllable_share", ctrl, "<=", 0.50, description="|SHAP| mass on controllable + append-only features (weighted)",
                           ideal=0.15, floor=0.9),
        GateCheck.evaluate("top_feature_share", top, "<=", 0.30, description="largest single feature's share of |SHAP| mass",
                           ideal=0.05, floor=0.6),
    ]
    return _result(mid, "M7", "Explanation & spurious-feature reliance", GateMode.WARN,
                   finding=(f"{ctrl:.0%} of attribution mass sits on features a file author can change freely (max 50%); the single "
                            f"largest feature, {names[0]}, carries {top:.0%} (max 30%)."),
                   checks=checks,
                   metrics={"controllable_share": ctrl, "top_feature_share": top, "effective_n_features": 1.0 / herf,
                            "top_k_share": float(shares.sum()), "top_k": 20, "attribution": "TreeExplainer (exact, tree access)",
                            "shap_samples": 2000},
                   store=store, params={"top_k": 20, "shap_samples": 2000, "max_controllable_share": 0.5,
                                        "max_top_feature_share": 0.3},
                   notes=["Attribution only: this shows what the model relies on, not whether those features can be exploited."],
                   seed=_seed(mid), duration_s=64.3)


# --------------------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------------------


def _libraries() -> dict[str, str | None]:
    return {"numpy": "2.4.0", "scipy": "1.16.2", "sklearn": "1.9.0", "lightgbm": "4.7.0", "xgboost": "3.2.0",
            "shap": "0.51.0", "art": "1.20.0", "modelscan": "0.8.8", "onnxruntime": "1.23.0", "tesseract": "0.9",
            "lief": None, "pefile": None, "jinja2": "3.1.6", "pydantic": "2.11.9"}


def _corpus(name: str) -> dict[str, Any]:
    if name == "ember_v3_2024":
        return {"name": name, "version": "2024.1", "feature_version": "ember_v3", "content_hash": _sha("ember2024-corpus"),
                "n": 1_589_371, "dim": 2568, "synthetic": False,
                "splits": {"train": {"n": 1_260_000, "malicious": 630_000, "benign": 630_000, "unlabeled": 0},
                           "test": {"n": 323_056, "malicious": 161_528, "benign": 161_528, "unlabeled": 0},
                           "challenge": {"n": 6_315, "malicious": 6_315, "benign": 0, "unlabeled": 0}},
                "roles": {"eval": ["test"], "temporal": ["train", "test"], "challenge": ["challenge"], "pool": ["train"]},
                "time_range": ["2023-09-24", "2024-12-14"], "source": "EMBER2024 feature release (thrember, Apache-2.0)",
                "error": None, "excluded_training_members": 1204}
    return {"name": name, "version": "2018.2", "feature_version": "ember_v2", "content_hash": _sha("ember2018-corpus"),
            "n": 1_000_000, "dim": 2381, "synthetic": False,
            "splits": {"train": {"n": 800_000, "malicious": 300_000, "benign": 300_000, "unlabeled": 200_000},
                       "test": {"n": 200_000, "malicious": 100_000, "benign": 100_000, "unlabeled": 0}},
            "roles": {"eval": ["test"], "temporal": ["train", "test"], "pool": ["train"]},
            "time_range": ["2017-01-01", "2018-12-31"], "source": "EMBER2018 features (vectorized from the published raw JSONL)",
            "error": None, "excluded_training_members": 0}


def _assemble(variant: str, results: list[ModuleResult], store: ArtifactStore, *, corpus: str, fv: str,
              class_name: str, title: str | None, warnings: list[str], aborted: str | None = None,
              hashes: bool = True, cutoff: str | None = "2023-12-31", out_dir: str | None = None,
              extraction_enabled: bool = False) -> dict[str, Any]:
    overrides: dict[str, Any] = {"corpus": corpus, "report": {"title": title}}
    if extraction_enabled:
        overrides["modules"] = {"extraction": {"enabled": True}}
    cfg = load_config(None, overrides)
    vr = compute_verdict(results, cfg, aborted=aborted)
    code = exit_code_for(vr, results, "blocked")
    hard = [r for r in results if r.gate is GateMode.HARD]
    started = dt.datetime(2026, 9, 28, 14, 2, 11, tzinfo=dt.timezone.utc)
    dur = round(sum(r.duration_s or 0 for r in results) + 17.3, 1)
    out_dir = out_dir or f"/scratch/researcher/runs/{variant}"
    model_path = f"{Path(out_dir).parent}/adapter/model/" + (
        "ember_xgb.json" if class_name.startswith("XGB") else "ember_lgbm.txt")
    kind = "xgboost" if class_name.startswith("XGB") else "sklearn_gbdt" if class_name.startswith("Sklearn") else "lightgbm"
    decl = {
        "feature_version": fv, "model_kind": kind, "operating_threshold": 0.8,
        "training_hashes_path": "/scratch/researcher/adapter/train_hashes.txt" if hashes else None,
        "training_cutoff": cutoff, "model_paths": [model_path],
        "adapter_path": "/scratch/researcher/adapter/my_detector_adapter.py", "class_name": class_name,
        "has_featurize": False, "extras": {},
    }
    corpus_summary = _corpus(corpus)
    corpus_summary.update(available=True, used_for_evaluation=not aborted, provider=f"malvalid.corpora:{corpus}",
                          location=f"/scratch/researcher/malvalid_corpora/{corpus}",
                          path=f"/scratch/researcher/malvalid_corpora/{corpus}")
    report: dict[str, Any] = {
        "schema_version": "malvalid-report/1",
        "tool": {"name": "malvalid", "version": "0.1.0"},
        "title": title,
        "run": {"id": f"20260928T140211Z-{_sha(variant)[:6]}", "started_at": started.isoformat(),
                "finished_at": (started + dt.timedelta(seconds=dur)).isoformat(), "duration_s": dur,
                "command": f"malvalid run --adapter {decl['adapter_path']} --config gate.yaml --out {out_dir}",
                "seed": 0, "out_dir": out_dir, "report_json": f"{out_dir}/report.json",
                "report_html": f"{out_dir}/report.html", "run_log": f"{out_dir}/run.log", "only": None, "skip": [],
                "allow_pickle": False},
        "verdict": vr.to_dict(),
        "gate": {"exit_code": code, "exit_meaning": EXIT_MEANINGS.get(code, ""), "passed": code == 0,
                 "fail_on": "blocked",
                 "hard_gates": [{"module_id": r.module_id, "code": r.code, "title": r.title, "status": r.status.value,
                                 "gate_outcome": r.gate_outcome.value} for r in hard],
                 "failed_hard": [r.module_id for r in hard if r.gate_outcome is GateOutcome.FAILED],
                 "not_evaluated_hard": [r.module_id for r in hard if r.gate_outcome is GateOutcome.NOT_EVALUATED
                                        and r.status is not Status.ERROR],
                 "errored": [r.module_id for r in results if r.status is Status.ERROR],
                 "skipped": [r.module_id for r in results if r.status is Status.SKIPPED],
                 "aborted": aborted is not None, "abort_reason": aborted,
                 "load_error": "model was not loaded: the run was aborted by M0" if aborted else None,
                 "disabled": [] if extraction_enabled else ["extraction"], "deselected": []},
        "model": {**decl, "artifacts": [{"path": model_path, "sha256": _sha(variant + "model"), "size": 18_734_112,
                                         "format": {"lightgbm": "lightgbm_text", "xgboost": "xgboost_json"}.get(kind, "pickle"),
                                         "is_pickle": variant == "aborted"}],
                  "adapter_sha256": _sha(variant + "adapter"),
                  "tree_access": ({"available": False, "n_trees": None, "fidelity_max_abs_diff": None,
                                   "note": "model not loaded (run aborted by M0)"} if aborted else
                                  {"available": True, "n_trees": 1000, "fidelity_max_abs_diff": 3.1e-7, "note": None}),
                  "load_error": "model was not loaded: the run was aborted by M0" if aborted else None,
                  "query_count": None if aborted else 1_184_310},
        "corpus": corpus_summary,
        "schema": {"name": fv, "dim": corpus_summary["dim"], "featurize_available": False},
        "training_manifest": ({"path": decl["training_hashes_path"], "n_hashes": 600_000, "n_in_corpus": 598_870,
                               "declared": True, "cutoff": cutoff, "cutoff_parsed": cutoff, "warnings": []} if hashes else
                              {"path": None, "declared": False, "n_hashes": None, "n_in_corpus": None, "cutoff": cutoff,
                               "cutoff_parsed": cutoff, "warnings": []}),
        "capabilities": [] if aborted else ["feature_space", "query_only", "training_cutoff", "tree_access"]
        + (["training_hashes"] if hashes else []),
        "config": cfg.to_dict(),
        "environment": {"python": "3.11.9", "platform": "Linux-5.14.0-687.12.1.el9_8.x86_64-x86_64-with-glibc2.34",
                        "cpu_count": 28, "libraries": _libraries()},
        "sandbox": {"enabled": True, "backend": "bwrap", "network_isolated": True, "memory_mb": 32768, "threads": 8,
                    "allow_pickle": False, "detail": "bubblewrap 0.9.0: --unshare-net --unshare-pid --die-with-parent"},
        "modules": [r.to_dict() for r in results],
        "artifacts": store.to_dict(),
        "warnings": warnings,
        "disclaimers": [
            "malvalid is a pre-deployment screen for a researcher's own detector, not a certification.",
            "Screening modules (M5) can only find evidence of a problem; absence of findings is not proof of absence.",
            "Scores depend on the canonical corpus and on the thresholds in your gate configuration, which are policy choices.",
        ],
    }
    return report


def build_report(variant: str = "blocked") -> dict[str, Any]:
    """Build one fixture report. Deterministic for a given numpy version."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    rng = np.random.default_rng({"blocked": 7, "ready": 11, "conditional": 13, "not_ready": 17, "aborted": 19}[variant])
    store = ArtifactStore()
    months_24 = [f"2024-{m:02d}" for m in range(1, 13)]
    months_18 = [f"2018-{m:02d}" for m in range(1, 13)]

    if variant == "blocked":
        results = [
            m0_file_safety(store),
            m1_performance(store, rng, fpr=0.0231, dr=0.9812, sep=0.9, threshold=0.8, challenge=0.612),
            m2_drift(store, rng, start_f1=0.93, decay=0.024, months=months_24, cutoff="2023-12-31"),
            m4_membership(store, rng, adv=0.062),
            m5_backdoor(store, flagged=2, top_suspicion=0.83, escape=True),
            m6_extraction_error(store),
            m7_explanation(store, rng, ctrl=0.44, top=0.41, escape=True, dim_label="ember_v3"),
            _skipped("featurizer_parity", "P1", "Featurizer parity (third-party plugin)", GateMode.WARN,
                     "no sample corpus (sample_dir not set or missing)", {"max_mismatch": 0.0}),
        ]
        # A note carrying HTML/script payloads (must be escaped) and a very long unbroken token.
        results[1].notes.append(f"Adapter self-description: {XSS_BREAKOUT} & \"quoted\" 'single' <b>bold?</b>")
        results[2].notes.append("Long path check: /" + "/".join(["very_long_directory_name_segment"] * 9) + "/windows.parquet")
        results[1].finding += " " + XSS_IMG
        return _assemble(
            variant, results, store, corpus="ember_v3_2024", fv="ember_v3",
            class_name="EmberLGBMDetector<v2 & friends>",
            title='Q3 detector <b>candidate</b> & "rc-2"',
            warnings=[f"adapter printed to stdout during load(): {XSS_IMG}",
                      "tree access verified: 1,000 trees, max |Δ| vs predict_proba 3.1e-07"],
            out_dir="/scratch/researcher/experiments/2026-09-detector-bakeoff/ember2024_lgbm_candidate_rc2/"
                    "malvalid_runs/run_20260928T140211Z_with_a_rather_long_directory_name",
            extraction_enabled=True,
        )
    if variant == "ready":
        results = [
            m0_file_safety(store),
            m1_performance(store, rng, fpr=0.0042, dr=0.9721, sep=1.3, threshold=0.8),
            m2_drift(store, rng, start_f1=0.95, decay=0.006, months=months_18, cutoff="2017-12-31"),
            m4_membership(store, rng, adv=0.031),
            m5_backdoor(store, flagged=0, top_suspicion=0.31),
            m7_explanation(store, rng, ctrl=0.22, top=0.09),
        ]
        return _assemble(variant, results, store, corpus="ember_v2_2018", fv="ember_v2", class_name="EmberLGBMDetector",
                         title=None, warnings=[], cutoff="2017-12-31")
    if variant == "conditional":
        results = [
            m0_file_safety(store),
            m1_performance(store, rng, fpr=0.0071, dr=0.9614, sep=1.1, threshold=0.75),
            m2_drift(store, rng, start_f1=0.78, decay=0.02, months=months_18, cutoff="2017-12-31"),
            _skipped("membership_inf", "M4", "Membership inference", GateMode.WARN,
                     "no training manifest (training_hashes_path not declared)",
                     {"max_advantage": 0.10, "max_per_side": 5000, "min_members": 200, "test_fraction": 0.5,
                      "time_matched": True}, seed=_seed("membership_inf")),
            m5_backdoor(store, flagged=0, top_suspicion=0.44),
            m7_explanation(store, rng, ctrl=0.31, top=0.12),
        ]
        return _assemble(variant, results, store, corpus="ember_v2_2018", fv="ember_v2", class_name="XGBEmberDetector",
                         title="XGBoost baseline", warnings=[], hashes=False, cutoff="2017-12-31")
    if variant == "not_ready":
        results = [
            m0_file_safety(store),
            m1_performance(store, rng, fpr=0.0098, dr=0.9507, sep=0.95, threshold=0.5),
            m2_drift(store, rng, start_f1=0.74, decay=0.035, months=months_18, cutoff="2017-12-31"),
            m4_membership(store, rng, adv=0.27),
            m5_backdoor(store, flagged=5, top_suspicion=0.93),
            m7_explanation(store, rng, ctrl=0.81, top=0.52),
        ]
        return _assemble(variant, results, store, corpus="ember_v2_2018", fv="ember_v2", class_name="OverfitDetector",
                         title=None, warnings=["corpus verification used a cached sidecar (verified 2026-09-20)"],
                         cutoff="2017-12-31")
    # aborted
    pkl = "/scratch/researcher/adapter/model/gbdt.pkl"
    m0 = m0_file_safety(store, pickle_abort=True, artifact=pkl)
    reason = f"run aborted by M0: {Path(pkl).name} is a pickle and --allow-pickle was not given"
    others = [
        _skipped("performance", "M1", "Performance & calibration", GateMode.HARD, reason, {}),
        _skipped("drift", "M2", "Temporal drift", GateMode.WARN, reason, {}),
        _skipped("membership_inf", "M4", "Membership inference", GateMode.WARN, reason, {}),
        _skipped("backdoor_screen", "M5", "Backdoor / poisoning screen", GateMode.WARN, reason, {}, screening=True),
        _skipped("explanation", "M7", "Explanation & spurious-feature reliance", GateMode.WARN, reason, {}),
    ]
    rep = _assemble("aborted", [m0, *others], store, corpus="ember_v2_2018", fv="ember_v2", class_name="SklearnGBDetector",
                    title=None, warnings=["model was never loaded: M0 aborted the run"],
                    aborted=f"{Path(pkl).name} is a pickle and --allow-pickle was not given", cutoff="2017-12-31")
    rep["model"]["model_paths"] = [pkl]
    rep["model"]["artifacts"][0]["path"] = pkl
    rep["sandbox"] = {"enabled": True, "backend": "bwrap", "network_isolated": True, "note": "worker never started"}
    return rep


def write_all(out: Path = FIXTURE_DIR) -> list[Path]:
    """Write ``sample_report.json`` (the ``blocked`` kitchen sink) and ``sample_report_<variant>.json``."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for v in VARIANTS:
        rep = build_report(v)
        p = out / ("sample_report.json" if v == "blocked" else f"sample_report_{v}.json")
        p.write_text(json.dumps(rep, indent=1, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=FIXTURE_DIR)
    args = ap.parse_args()
    for p in write_all(args.out):
        rep = json.loads(p.read_text())
        v = rep["verdict"]
        print(f"{p.name}: {v['verdict']} score={v['score']} coverage={v['coverage']}")  # noqa: T201 (script)


if __name__ == "__main__":
    main()
