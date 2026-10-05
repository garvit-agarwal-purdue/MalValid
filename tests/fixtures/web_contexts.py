"""Example template contexts for the malvalid local web UI.

Used by ``tests/unit/test_web_templates.py`` and by the design previews. Each context is what the
backend passes to one template (docs/WEB_CONTRACT.md §5): realistic values (the real module and
corpus shapes, the sample reports in this folder), every verdict, a running job with progress, a
queued job, failed / cancelled / interrupted jobs, a run made on the command line (no job.json),
contexts where every field is None, and hostile strings that must come out escaped.

``CONTEXTS`` maps a case name to ``(template_name, context)``.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# Strings that must never reach the page unescaped.
HOSTILE = "<script>alert(1)</script>"
HOSTILE_ATTR = '" onmouseover="alert(1)" data-x="'
HOSTILE_IMG = "<img src=x onerror=alert(1)>"
HOSTILE_URL = "javascript:alert(1)"
HOSTILE_STRINGS = (HOSTILE, HOSTILE_ATTR, HOSTILE_IMG)

CSRF = "5f0c1a9e2b7d4c3a8e6f1b2d3c4a5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d"
RUNS_DIR = "/home/researcher/malvalid-runs"
PYTHON = "/home/researcher/.venvs/malvalid/bin/python"

COMMON: dict[str, Any] = {
    "app_version": "0.1.0",
    "csrf_token": CSRF,
    "bind": "127.0.0.1:8765",
    "runs_dir": RUNS_DIR,
    # a server started with --allow-path-mode --allow-no-sandbox: every control of the form is rendered
    # (the defaults, with these hidden, are covered by tests/unit/test_web_feature_flags.py)
    "allow_path_mode": True,
    "allow_no_sandbox": True,
}


def common(nav_active: str | None, request_path: str, **extra: Any) -> dict[str, Any]:
    ctx = dict(COMMON, nav_active=nav_active, request_path=request_path)
    ctx.update(extra)
    return ctx


# ------------------------------------------------------------------------------------------------
# Reports: the sample reports next to this file (fallback: a minimal report per verdict)
# ------------------------------------------------------------------------------------------------


def _minimal_report(verdict: str, score: float | None, label: str) -> dict[str, Any]:
    return {
        "schema_version": "malvalid-report/1",
        "tool": {"name": "malvalid", "version": "0.1.0"},
        "verdict": {"verdict": verdict, "label": label, "score": score, "raw_score": score, "capped": False,
                    "coverage": 1.0, "reasons": [], "blockers": [], "axes": [],
                    "bands": {"ready_min": 80.0, "conditional_min": 60.0, "min_coverage_ready": 0.75,
                              "blocked_score_cap": 49.0}, "summary": label},
        "gate": {"exit_code": 0, "fail_on": "blocked", "hard_gates": []},
        "modules": [],
    }


def _report(name: str, verdict: str, score: float | None, label: str) -> dict[str, Any]:
    try:
        return json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _minimal_report(verdict, score, label)


REPORT_READY = _report("sample_report_ready", "ready", 92.3, "Ready (meets the gate policy)")
REPORT_CONDITIONAL = _report("sample_report_conditional", "conditional", 81.4,
                             "Conditionally ready — review before promotion")
REPORT_NOT_READY = _report("sample_report_not_ready", "not_ready", 42.4, "Not ready")
REPORT_BLOCKED = _report("sample_report", "blocked", 49.0, "Blocked — a hard gate failed")  # module errored, exit 2
REPORT_ABORTED = _report("sample_report_aborted", "blocked", None, "Blocked — unsafe model artifact")


def _hostile_report() -> dict[str, Any]:
    r = copy.deepcopy(REPORT_CONDITIONAL)
    r["title"] = HOSTILE
    v = r.setdefault("verdict", {})
    v["label"] = HOSTILE_IMG
    v["summary"] = HOSTILE
    v["reasons"] = [HOSTILE, HOSTILE_ATTR]
    v["blockers"] = [HOSTILE_IMG]
    r.setdefault("gate", {})["exit_meaning"] = HOSTILE
    r["warnings"] = [HOSTILE, HOSTILE_IMG]
    model = r.setdefault("model", {})
    model.update(class_name=HOSTILE, adapter_path=HOSTILE_URL, model_kind=HOSTILE_ATTR, training_cutoff=HOSTILE)
    r.setdefault("corpus", {}).update(name=HOSTILE, error=HOSTILE_IMG)
    r.setdefault("run", {})["command"] = "malvalid run --adapter '" + HOSTILE + "'"
    mods = r.setdefault("modules", [])
    if mods:
        mods[0].update(title=HOSTILE, finding=HOSTILE_IMG, code=HOSTILE_ATTR)
        if mods[0].get("checks"):
            mods[0]["checks"][0].update(metric=HOSTILE, value=HOSTILE_IMG)
    mods.append({"module_id": HOSTILE_ATTR, "code": "X9", "title": HOSTILE, "status": "error",
                 "gate": "warn", "gate_outcome": "not_evaluated", "finding": HOSTILE,
                 "error": "Traceback (most recent call last):\n" + HOSTILE_IMG, "checks": [], "score": None})
    return r


REPORT_HOSTILE = _hostile_report()
REPORT_MALFORMED: dict[str, Any] = {
    "verdict": {"verdict": 42, "score": "not a number", "bands": "oops", "reasons": "not a list",
                "axes": [None, 3, {"module_id": None}]},
    "gate": None,
    "modules": [None, "x", {"status": None, "checks": [None, {"passed": "maybe"}]}],
    "model": [],
    "corpus": "nope",
    "warnings": None,
}

# ------------------------------------------------------------------------------------------------
# RunSummary dicts
# ------------------------------------------------------------------------------------------------

RUN_SUMMARY_KEYS = (
    "run_id", "source", "status", "created_at", "finished_at", "duration_s", "display_name", "adapter",
    "class_name", "model_kind", "feature_version", "operating_threshold", "corpus", "verdict",
    "verdict_label", "score", "coverage", "exit_code", "n_modules", "n_pass", "n_warn", "n_fail",
    "n_skipped", "n_error", "title",
)


def summary(run_id: str | None, **fields: Any) -> dict[str, Any]:
    """A RunSummary with every contract key present (None unless given)."""
    s = {k: None for k in RUN_SUMMARY_KEYS}
    s["run_id"] = run_id
    s.update(fields)
    return s


def _counts(report: dict[str, Any]) -> dict[str, int]:
    mods = [m for m in report.get("modules") or [] if isinstance(m, dict)]
    st = [str(m.get("status")) for m in mods]
    return {"n_modules": len(mods), "n_pass": st.count("pass"), "n_warn": st.count("warn"),
            "n_fail": st.count("fail"), "n_skipped": st.count("skipped"), "n_error": st.count("error")}


def _from_report(run_id: str, report: dict[str, Any], **fields: Any) -> dict[str, Any]:
    v = report.get("verdict") or {}
    m = report.get("model") or {}
    base = dict(
        verdict=v.get("verdict"), verdict_label=v.get("label"), score=v.get("score"), coverage=v.get("coverage"),
        class_name=m.get("class_name"), model_kind=m.get("model_kind"), feature_version=m.get("feature_version"),
        operating_threshold=m.get("operating_threshold"), corpus=(report.get("corpus") or {}).get("name"),
        exit_code=(report.get("gate") or {}).get("exit_code"), title=report.get("title"),
        display_name=m.get("class_name"), status="finished", source="web",
    )
    base.update(_counts(report))
    base.update(fields)
    return summary(run_id, **base)


RID = {
    "running": "20260929T151744Z-5c1e9a07",
    "queued": "20260929T152210Z-0b7d33e1",
    "ready": "20260929T140312Z-1a2b3c4d",
    "conditional": "20260928T221530Z",
    "not_ready": "20260928T113002Z-7f3e2a10",
    "blocked": "20260928T140211Z-69735dd2",
    "aborted": "20260927T090015Z-3d4e5f60",
    "failed": "20260927T081503Z-a0b1c2d3",
    "cancelled": "20260926T173000Z-e4f5a6b7",
    "interrupted": "20260926T101010Z-c8d9e0f1",
    "hostile": "20260925T120000Z-deadbeef",
    "sparse": "20260924T000000Z-00000000",
}

R_READY = _from_report(RID["ready"], REPORT_READY, title="LightGBM baseline on EMBER2018",
                       adapter=f"{RUNS_DIR}/submissions/{RID['ready']}/adapter.py",
                       created_at="2026-09-29T14:03:12Z", finished_at="2026-09-29T14:10:05Z", duration_s=412.7)
R_CONDITIONAL = _from_report(RID["conditional"], REPORT_CONDITIONAL, source="cli", display_name=None,
                             adapter="/home/researcher/detectors/xgb-baseline/adapter.py",
                             created_at="2026-09-28T22:15:30Z", finished_at="2026-09-28T22:31:02Z",
                             duration_s=931.9)
R_NOT_READY = _from_report(RID["not_ready"], REPORT_NOT_READY, title="Overfit ablation (no dropout)",
                           adapter=f"{RUNS_DIR}/submissions/{RID['not_ready']}/adapter.py",
                           created_at="2026-09-28T11:30:02Z", finished_at="2026-09-28T11:41:47Z", duration_s=705.3)
R_BLOCKED = _from_report(RID["blocked"], REPORT_BLOCKED, status="failed", exit_code=2,
                         adapter="/scratch/researcher/adapter/my_detector_adapter.py",
                         created_at="2026-09-28T14:02:11Z", finished_at="2026-09-28T14:13:51Z", duration_s=700.2)
R_ABORTED = _from_report(RID["aborted"], REPORT_ABORTED, title="sklearn GBDT (pickle)",
                         adapter=f"{RUNS_DIR}/submissions/{RID['aborted']}/adapter.py",
                         created_at="2026-09-27T09:00:15Z", finished_at="2026-09-27T09:00:19Z", duration_s=3.9)
R_RUNNING = summary(RID["running"], source="web", status="running", created_at="2026-09-29T15:17:44Z",
                    display_name="Ember2024LightGBM", title="EMBER2024 LightGBM, threshold tuned for 0.1% FPR",
                    adapter=f"{RUNS_DIR}/submissions/{RID['running']}/adapter.py", class_name="Ember2024LightGBM",
                    model_kind="lightgbm", feature_version="ember_v3", operating_threshold=0.9312,
                    corpus="ember_v3_2024")
R_QUEUED = summary(RID["queued"], source="web", status="queued", created_at="2026-09-29T15:22:10Z",
                   display_name="adapter.py", adapter="/home/researcher/detectors/onnx-mlp/adapter.py")
R_FAILED = summary(RID["failed"], source="web", status="failed", created_at="2026-09-27T08:15:03Z",
                   finished_at="2026-09-27T08:15:07Z", duration_s=4.2, exit_code=2, display_name="adapter.py",
                   title="MLP detector (ONNX)", adapter=f"{RUNS_DIR}/submissions/{RID['failed']}/adapter.py")
R_CANCELLED = summary(RID["cancelled"], source="web", status="cancelled", created_at="2026-09-26T17:30:00Z",
                      finished_at="2026-09-26T17:33:41Z", duration_s=221.0, display_name="XGBEmberDetector",
                      class_name="XGBEmberDetector", model_kind="xgboost", feature_version="ember_v2",
                      operating_threshold=0.5, corpus="ember_v2_2018")
R_INTERRUPTED = summary(RID["interrupted"], source="web", status="interrupted", created_at="2026-09-26T10:10:10Z",
                        display_name="Ember2018LightGBM", class_name="Ember2018LightGBM", model_kind="lightgbm",
                        feature_version="ember_v2", operating_threshold=0.8336, corpus="ember_v2_2018")
R_HOSTILE = summary(RID["hostile"], source="web", status=HOSTILE_IMG, created_at=HOSTILE, title=HOSTILE,
                    display_name=HOSTILE_ATTR, adapter=HOSTILE_URL, class_name=HOSTILE, model_kind=HOSTILE_IMG,
                    feature_version=HOSTILE_ATTR, corpus=HOSTILE, verdict="<b>ready</b>", verdict_label=HOSTILE,
                    score=HOSTILE, coverage=HOSTILE, n_pass="3<b>", duration_s=HOSTILE)
R_SPARSE = summary(RID["sparse"])

ALL_RUNS = [R_RUNNING, R_QUEUED, R_READY, R_CONDITIONAL, R_NOT_READY, R_BLOCKED, R_ABORTED, R_FAILED,
            R_CANCELLED, R_INTERRUPTED, R_HOSTILE, R_SPARSE]
ACTIVE_RUNS = [R_RUNNING, R_QUEUED]


def _count_runs(runs: list[dict[str, Any]]) -> dict[str, int]:
    c = {"ready": 0, "conditional": 0, "not_ready": 0, "blocked": 0, "running": 0, "failed": 0}
    for r in runs:
        if r.get("verdict") in c:
            c[r["verdict"]] += 1
        if r.get("status") in ("queued", "running"):
            c["running"] += 1
        if r.get("status") == "failed":
            c["failed"] += 1
    return c


# ------------------------------------------------------------------------------------------------
# Jobs and progress
# ------------------------------------------------------------------------------------------------


def rid_time(run_id: str, delta_s: float = 0.0) -> str:
    """The UTC time encoded in a web run id (``YYYYMMDDTHHMMSSZ-…``), shifted by ``delta_s``, as ISO."""
    t = dt.datetime.strptime(run_id[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    return (t + dt.timedelta(seconds=delta_s)).strftime("%Y-%m-%dT%H:%M:%SZ")


def job(run_id: str, status: str, *, mode: str = "upload", options: dict[str, Any] | None = None,
        **fields: Any) -> dict[str, Any]:
    sub = f"{run_id}-sub"
    adapter = (f"{RUNS_DIR}/submissions/{sub}/adapter.py" if mode == "upload"
               else "/home/researcher/detectors/lgbm-2024/adapter.py")
    argv = [PYTHON, "-m", "malvalid", "run", "--adapter", adapter, "--out", f"{RUNS_DIR}/{run_id}"]
    # The shape web/forms.py RunRequest.options() writes to job.json.
    opts = {"title": None, "class_name": None, "only": [], "skip": [], "corpus": None, "corpus_dir": None,
            "seed": 0, "fail_on": "blocked", "allow_pickle": False, "no_sandbox": False,
            "config": f"{RUNS_DIR}/submissions/{sub}/gate.yaml" if mode == "upload" else None,
            "config_source": "upload" if mode == "upload" else "default",
            "files": ({"adapter": "adapter.py", "models": ["model.txt"], "manifest": "train_sha256.txt",
                       "config": "gate.yaml"} if mode == "upload" else {})}
    opts.update(options or {})
    for flag, key in (("--corpus", "corpus"), ("--seed", "seed"), ("--skip", "skip"), ("--only", "only")):
        if opts.get(key) not in (None, "", []):
            for v in (opts[key] if isinstance(opts[key], list) else [opts[key]]):
                argv += [flag, str(v)]
    if opts.get("fail_on") not in (None, "", "blocked"):
        argv += ["--fail-on", str(opts["fail_on"])]
    j = {
        "schema": "malvalid-job/1", "run_id": run_id, "created_at": rid_time(run_id, -2),
        "started_at": rid_time(run_id), "finished_at": None, "status": status, "argv": argv,
        "mode": mode, "adapter": adapter, "display_name": "adapter.py", "submission_id": sub if mode == "upload" else None,
        "options": opts, "pid": 48213, "exit_code": None, "error": None,
    }
    j.update(fields)
    return j


_MODS = [("file_safety", "M0", "Model file safety"), ("performance", "M1", "Performance & calibration"),
         ("drift", "M2", "Temporal drift"), ("membership_inf", "M4", "Membership inference (training-data leakage)"),
         ("backdoor_screen", "M5", "Backdoor / poisoning screening"),
         ("explanation", "M7", "Explanation & spurious-feature reliance")]


def progress(run_id: str, stage: str, statuses: list[str], durations: list[float | None], *,
             message: str = "", current: int | None = None, verdict: dict[str, Any] | None = None) -> dict[str, Any]:
    mods = [{"id": i, "code": c, "title": t, "status": s, "duration_s": d}
            for (i, c, t), s, d in zip(_MODS, statuses, durations)]
    cur = None
    if current is not None:
        i, c, t = _MODS[current]
        cur = {"id": i, "code": c, "title": t}
    return {"schema": "malvalid-progress/1", "run_id": run_id, "stage": stage, "message": message, "module": cur,
            "modules": mods, "started_at": rid_time(run_id), "updated_at": rid_time(run_id, 198),
            "verdict": verdict}


PROGRESS_RUNNING = progress(
    RID["running"], "modules", ["pass", "pass", "running", "skipped", "pending", "pending"],
    [0.9, 48.2, None, 0.0, None, None],
    message="Scoring window 7 of 12 (2024-11) on 38,211 post-cutoff samples", current=2)
PROGRESS_LOADING = progress(RID["running"], "loading", ["pass"] + ["pending"] * 5, [1.2] + [None] * 5,
                            message="Starting the sandbox worker (bwrap)")
PROGRESS_DONE = progress(RID["ready"], "done", ["pass"] * 6, [0.8, 48.2, 31.7, 112.4, 22.9, 64.3],
                         verdict={"verdict": "ready", "label": "Ready (meets the gate policy)", "score": 92.3, "coverage": 1.0})

FILES_ALL = {"report_html": True, "report_json": True, "run_log": True, "console_log": True}
FILES_LOGS = {"report_html": False, "report_json": False, "run_log": True, "console_log": True}
FILES_NONE = {"report_html": False, "report_json": False, "run_log": False, "console_log": False}


def run_ctx(run: dict[str, Any], *, job_: dict[str, Any] | None, progress_: dict[str, Any] | None,
            report: dict[str, Any] | None, files: dict[str, Any] | None, path_id: str | None = None) -> dict[str, Any]:
    return common("dashboard", f"/runs/{path_id or run.get('run_id')}", run=run, job=job_, progress=progress_,
                  report=report, files=files)


# ------------------------------------------------------------------------------------------------
# Modules, corpora, system (shapes from `malvalid list-modules --json`, `corpus list --json`,
# `sandbox-check --json`)
# ------------------------------------------------------------------------------------------------

MODULES: list[dict[str, Any]] = [
    {"id": "file_safety", "code": "M0", "title": "Model file safety",
     "description": "Scans every model artifact with modelscan and malvalid's static pickle-opcode scan before "
     "anything is deserialized, and refuses pickle-based artifacts unless --allow-pickle is given. A CRITICAL "
     "finding (or a pickle without --allow-pickle) aborts the run before load.",
     "requires": [], "requires_any": [], "enhanced_by": [], "screening": False, "default_gate": "hard",
     "default_params": {"fail_on": "HIGH", "scan_adapter_dir": True}, "enabled_by_default": True, "weight": 1.0},
    {"id": "performance", "code": "M1", "title": "Performance & calibration",
     "description": "Scores the canonical held-out benign and malicious corpus (excluding your training samples) at "
     "your declared operating threshold: false-positive rate and detection rate (the hard gate), plus ROC/PR "
     "curves, AUROC/AUPRC, TPR at low FPR and calibration.",
     "requires": ["feature_space"], "requires_any": [], "enhanced_by": ["training_hashes"], "screening": False,
     "default_gate": "hard", "default_params": {"max_fpr": 0.01, "min_detection": 0.95, "max_samples_per_class": None,
                                                "calibration_bins": 15, "include_challenge": True},
     "enabled_by_default": True, "weight": 3.0},
    {"id": "drift", "code": "M2", "title": "Temporal drift",
     "description": "TESSERACT-style time-aware evaluation: scores canonical samples dated after your declared "
     "training cutoff in consecutive time windows and reports AUT(F1), sample-weighted so sparse windows cannot "
     "dominate.",
     "requires": ["feature_space", "training_cutoff"], "requires_any": [], "enhanced_by": ["training_hashes"],
     "screening": False, "default_gate": "warn",
     "default_params": {"min_aut_f1": 0.7, "granularity": "month", "min_window_samples": 200, "max_windows": 36},
     "enabled_by_default": True, "weight": 1.5},
    {"id": "membership_inf", "code": "M4", "title": "Membership inference (training-data leakage)",
     "description": "Black-box membership inference through predict_proba: how well an attacker who can only query "
     "the model can tell files that were in its training set from comparable unseen files.",
     "requires": ["feature_space", "training_hashes"], "requires_any": [], "enhanced_by": [], "screening": False,
     "default_gate": "warn", "default_params": {"max_advantage": 0.1, "max_per_side": 5000, "min_members": 200},
     "enabled_by_default": True, "weight": 1.0},
    {"id": "backdoor_screen", "code": "M5", "title": "Backdoor / poisoning screening",
     "description": "Screens for signs that the detector was trained to let deliberately marked malware through. "
     "Screening only: absence of findings is not proof of a clean model.",
     "requires": [], "requires_any": ["tree_access", "feature_space"], "enhanced_by": [], "screening": True,
     "default_gate": "warn", "default_params": {"triggers": [], "triggers_path": None, "max_trigger_drop": 0.2},
     "enabled_by_default": True, "weight": 1.0},
    {"id": "extraction", "code": "M6", "title": "Model-extraction susceptibility",
     "description": "How cheaply the model can be cloned through its verdicts. Only relevant if the model will be "
     "exposed as a queryable service.",
     "requires": ["query_only"], "requires_any": [], "enhanced_by": [], "screening": False, "default_gate": "warn",
     "default_params": {"query_budgets": [100, 1000, 10000, 50000], "max_fidelity": 0.95},
     "enabled_by_default": False, "weight": 0.5},
    {"id": "explanation", "code": "M7", "title": "Explanation & spurious-feature reliance",
     "description": "Global feature attribution aggregated by attacker controllability: flags detectors whose "
     "decisions rest on features a file's author can set for free.",
     "requires": [], "requires_any": ["tree_access", "feature_space"], "enhanced_by": [], "screening": False,
     "default_gate": "warn", "default_params": {"top_k": 20, "max_controllable_share": 0.5,
                                                "max_top_feature_share": 0.3},
     "enabled_by_default": True, "weight": 1.0},
]
MODULE_HOSTILE = {"id": HOSTILE_ATTR, "code": HOSTILE, "title": HOSTILE_IMG, "description": HOSTILE,
                  "requires": [HOSTILE], "requires_any": [HOSTILE_ATTR], "enhanced_by": [HOSTILE_IMG],
                  "screening": True, "default_gate": HOSTILE, "default_params": {HOSTILE: HOSTILE_IMG},
                  "enabled_by_default": None, "weight": HOSTILE}

CORPORA: list[dict[str, Any]] = [
    {"name": "ember_v2_2018", "feature_version": "ember_v2", "version": "2018.2-r1",
     "description": "EMBER2018 (feature version 2): 800k train rows and 200k test rows first seen Jan-Dec 2018, "
     "vectorised to ember_v2.", "available": True,
     "location": "/scratch/researcher/malvalid_data/corpora/ember_v2_2018", "n": 1000000,
     "content_hash": "b76ea441be1a62197c7d19801174309e988eea54740ff22f03fa75e66beb4b98", "synthetic": False,
     "hint": None},
    {"name": "ember_v3_2024", "feature_version": "ember_v3", "version": "2",
     "description": "EMBER2024 PE test set (Win32+Win64, 480k files first seen 2024-09-22..2024-12-14) and the "
     "4,868 PE rows of the EMBER2024 challenge set, in EMBER feature version 3.", "available": False,
     "location": "/scratch/researcher/malvalid_data/corpora/ember_v3_2024", "n": None,
     "content_hash": "4f275e15e00ee11cdc67f176bbda932b3feaa92ec6d8584420cb04714e5110cc", "synthetic": False,
     "hint": "corpus 'ember_v3_2024' not found at /scratch/researcher/malvalid_data/corpora/ember_v3_2024: build it "
     "with `malvalid corpus build ember_v3_2024 --source <dir with the EMBER2024 feature files>`"},
    {"name": "synthetic_v2", "feature_version": "ember_v2", "version": "1",
     "description": "SYNTHETIC EMBER v2-shaped corpus (2381 features, 36 monthly windows) for demos and CI - not "
     "evidence about any real detector.", "available": True,
     "location": "/scratch/researcher/malvalid_data/corpora/synthetic_v2", "n": 20000, "content_hash": None,
     "synthetic": True, "hint": None},
    {"name": "synthetic_v3", "feature_version": "ember_v3", "version": "1",
     "description": "SYNTHETIC EMBER v3-shaped corpus (2568 features) for demos and CI - not evidence about any "
     "real detector.", "available": True, "location": "/scratch/researcher/malvalid_data/corpora/synthetic_v3",
     "n": 20000, "content_hash": None, "synthetic": True, "hint": None},
]
CORPUS_HOSTILE = {"name": HOSTILE, "feature_version": HOSTILE_ATTR, "version": HOSTILE_IMG, "description": HOSTILE,
                  "available": False, "location": HOSTILE_URL, "n": HOSTILE, "content_hash": HOSTILE,
                  "synthetic": True, "hint": HOSTILE_IMG}
CORPUS_SPARSE = {k: None for k in ("name", "feature_version", "version", "description", "available", "location",
                                   "n", "content_hash", "synthetic", "hint")}


def _default_config_text() -> str:
    try:
        from malvalid.config import default_config_text

        return default_config_text()
    except Exception:  # pragma: no cover - the frozen config module is always importable in this repo
        return "# malvalid gate configuration\ncorpus: ember_v2_2018\n"


DEFAULT_CONFIG_TEXT = _default_config_text()

BACKENDS: dict[str, Any] = {
    "bwrap": {"available": True, "network_isolated": True,
              "detail": "network isolated, own PID namespace, read-only root, $HOME hidden", "path": "/bin/bwrap",
              "pid_namespace": True, "readonly_root": True, "home_hidden": True, "version": "bubblewrap 0.6.3"},
    "unshare": {"available": True, "network_isolated": True,
                "detail": "network isolated, own PID namespace, file system NOT restricted", "path": "/bin/unshare",
                "kill_child": True, "pid_namespace": True, "readonly_root": False, "home_hidden": False},
    "subprocess": {"available": True, "network_isolated": False,
                   "detail": "plain subprocess: NO network or file-system isolation (resource limits and pickle "
                   "guards only)", "pid_namespace": False, "readonly_root": False},
}
BACKENDS_NONE: dict[str, Any] = {
    "bwrap": {"available": False, "network_isolated": False, "detail": "bwrap not found on PATH"},
    "unshare": {"available": False, "network_isolated": False,
                "detail": "unshare: unprivileged user namespaces are disabled (kernel.unprivileged_userns_clone=0)"},
    "subprocess": dict(BACKENDS["subprocess"]),
}
ENVIRONMENT: dict[str, Any] = {
    "python": "3.11.9 (main, Apr 19 2024, 16:48:06) [GCC 11.2.0]",
    "platform": "Linux-5.14.0-687.12.1.el9_8.x86_64-x86_64-with-glibc2.34",
    "cpu_count": 28,
    "libraries": {"numpy": "2.4.1", "scipy": "1.16.2", "sklearn": "1.9.0", "lightgbm": "4.7.0", "xgboost": "3.2.0",
                  "shap": "0.51.0", "art": "1.20.1", "modelscan": "0.8.8", "onnx": "1.19.0", "onnxruntime": "1.23.0",
                  "tesseract": "0.9.0", "lief": None, "pefile": None, "jinja2": "3.1.6"},
    "env": {"OMP_NUM_THREADS": "4", "MALVALID_CORPUS_DIR": "/scratch/researcher/malvalid_data/corpora"},
}
SETTINGS: dict[str, Any] = {
    "host": "127.0.0.1", "port": 8765, "runs_dir": RUNS_DIR, "config": None, "max_concurrent": 1,
    "max_upload_mb": 4096, "allow_remote": False, "validate_timeout_s": 600,
    "token": "THIS-TOKEN-MUST-NEVER-BE-RENDERED", "csrf_secret": "THIS-SECRET-MUST-NEVER-BE-RENDERED",
}
SECRET_VALUES = (SETTINGS["token"], SETTINGS["csrf_secret"])


def _compare_rows(n: int) -> list[dict[str, Any]]:
    cell = lambda st, sc, out, km, kv, th: {"status": st, "score": sc, "gate_outcome": out,  # noqa: E731
                                            "key_metric": km, "key_value": kv, "threshold": th}
    rows = [
        {"module_id": "file_safety", "code": "M0", "title": "Model file safety",
         "cells": [cell("pass", None, "passed", "n_critical", 0, 0)] * n},
        {"module_id": "performance", "code": "M1", "title": "Performance & calibration",
         "cells": [cell("pass", 96.4, "passed", "fpr", 0.0041, 0.01), cell("pass", 88.1, "passed", "fpr", 0.0072, 0.01),
                   cell("fail", 36.0, "failed", "fpr", 0.0231, 0.01)][:n]},
        {"module_id": "drift", "code": "M2", "title": "Temporal drift",
         "cells": [cell("pass", 91.2, "passed", "aut_f1_weighted", 0.874, 0.7),
                   cell("warn", 61.7, "failed", "aut_f1_weighted", 0.662, 0.7),
                   cell("pass", 87.9, "passed", "aut_f1_weighted", 0.803, 0.7)][:n]},
        {"module_id": "membership_inf", "code": "M4", "title": "Membership inference",
         "cells": [cell("pass", 0.93, "passed", "advantage", 0.031, 0.1),  # 0..1 score: read as a fraction
                   cell("skipped", None, "not_evaluated", None, None, None),
                   cell("pass", 85.6, "passed", "advantage", 0.062, 0.1)][:n]},
        {"module_id": "extraction", "code": "M6", "title": "Model extraction susceptibility",
         "cells": [None, None, cell("error", None, "not_evaluated", None, None, None)][:n]},
    ]
    return rows


CONFIG_DIFF = [
    {"key": "modules.performance.max_fpr", "values": [0.01, 0.01, 0.005]},
    {"key": "runtime.seed", "values": [0, 1, 0]},
    {"key": "corpus_dir", "values": [None, "/data/corpora", None]},
    {"key": "modules.backdoor_screen.triggers", "values": [[], [{"feature": "strings.numstrings", "value": 1.0}], []]},
    {"key": "fail_on", "values": ["blocked", "not_ready", "blocked"]},
]

# ------------------------------------------------------------------------------------------------
# The contexts
# ------------------------------------------------------------------------------------------------

NEW_RUN_BASE = dict(modules=MODULES, corpora=CORPORA, default_config_text=DEFAULT_CONFIG_TEXT,
                    defaults={"corpus": "ember_v2_2018", "seed": 0, "fail_on": "blocked"}, max_upload_mb=4096,
                    errors=[], form={})

CONTEXTS: dict[str, tuple[str, dict[str, Any]]] = {
    # dashboard
    "dashboard": ("dashboard.html.j2", common("dashboard", "/", runs=ALL_RUNS, counts=_count_runs(ALL_RUNS),
                                              active=ACTIVE_RUNS)),
    "dashboard_idle": ("dashboard.html.j2", common(
        "dashboard", "/", runs=[R_READY, R_CONDITIONAL, R_NOT_READY, R_BLOCKED, R_FAILED],
        counts=_count_runs([R_READY, R_CONDITIONAL, R_NOT_READY, R_BLOCKED, R_FAILED]), active=[])),
    "dashboard_empty": ("dashboard.html.j2", common("dashboard", "/", runs=[],
                                                    counts=_count_runs([]), active=[])),
    "dashboard_none": ("dashboard.html.j2", common("dashboard", "/", runs=None, counts=None, active=None)),
    # new run
    "new_run": ("new_run.html.j2", common("new", "/runs/new", **NEW_RUN_BASE)),
    "new_run_errors": ("new_run.html.j2", common("new", "/runs", **dict(
        NEW_RUN_BASE,
        errors=["Model file 'gbdt.pkl' is pickle-based. Tick “Allow pickle-based model files” to accept it "
                "(pickles can run code when they are loaded).",
                "adapter_path: /home/researcher/detectors/missing/adapter.py does not exist",
                HOSTILE],
        form={"mode": "path", "adapter_path": HOSTILE_ATTR, "config_path": "/home/researcher/gate.yaml",
              "class_name": HOSTILE, "corpus": "ember_v3_2024", "corpus_dir": HOSTILE_IMG, "seed": "7",
              "only": "M1,M2", "skip": HOSTILE_ATTR, "fail_on": "not_ready", "allow_pickle": "on",
              "no_sandbox": "on", "confirm_no_sandbox": "", "title": HOSTILE}))),
    # after "Inspect model" (POST /runs/inspect): detected values pre-selected, the stored model reused
    "new_run_inspected": ("new_run.html.j2", common("new", "/runs/inspect", **dict(
        NEW_RUN_BASE,
        form={"mode": "model", "model_submission": "20260930T101500Z-1a2b3c4d", "model_kind": "lightgbm",
              "feature_version": "ember_v2", "threshold_mode": "declared", "threshold": "0.8336",
              "training_cutoff": "2018-10", "title": "LightGBM rc-2"},
        inspect={"ok": True, "model_submission": "20260930T101500Z-1a2b3c4d", "file_name": "lgbm_rc2.txt",
                 "size": 784536, "format": "lightgbm-text", "model_kind": "lightgbm", "n_features": 2381,
                 "feature_version": "ember_v2", "default_corpus": "ember_v2_2018", "is_pickle": False,
                 "notes": [], "errors": [], "error": None, "supported": {"2381": "ember_v2", "2568": "ember_v3"},
                 "details": {"lightgbm_version": "v4"}}))),
    "new_run_inspected_hostile": ("new_run.html.j2", common("new", "/runs/inspect", **dict(
        NEW_RUN_BASE,
        notice="Only the model file is kept: choose " + HOSTILE + " again.",
        form={"mode": "model", "model_submission": HOSTILE_ATTR, "model_kind": HOSTILE_ATTR,
              "feature_version": HOSTILE, "threshold_mode": HOSTILE, "threshold": HOSTILE_ATTR,
              "calibrate_fpr": HOSTILE_ATTR, "training_cutoff": HOSTILE_ATTR, "title": HOSTILE},
        inspect={"ok": False, "model_submission": HOSTILE_ATTR, "file_name": HOSTILE_IMG, "size": 12,
                 "format": HOSTILE, "model_kind": None, "n_features": 100, "feature_version": None,
                 "default_corpus": HOSTILE, "is_pickle": True, "notes": [HOSTILE], "errors": [HOSTILE_IMG],
                 "error": None, "supported": {}, "details": {}}))),
    "new_run_none": ("new_run.html.j2", common("new", "/runs/new", modules=None, corpora=None,
                                               default_config_text=None, defaults=None, max_upload_mb=None,
                                               errors=None, form=None)),
    # run detail: every verdict
    "run_ready": ("run_detail.html.j2", run_ctx(R_READY, job_=job(
        RID["ready"], "finished", exit_code=0, finished_at="2026-09-29T14:10:05Z",
        options={"title": "LightGBM baseline on EMBER2018", "corpus": "ember_v2_2018", "skip": "M6"}),
        progress_=PROGRESS_DONE, report=REPORT_READY, files=FILES_ALL)),
    "run_conditional_cli": ("run_detail.html.j2", run_ctx(R_CONDITIONAL, job_=None, progress_=None,
                                                          report=REPORT_CONDITIONAL, files=FILES_ALL)),
    "run_not_ready": ("run_detail.html.j2", run_ctx(R_NOT_READY, job_=job(
        RID["not_ready"], "finished", mode="path", exit_code=0, options={"fail_on": "blocked", "seed": "3"}),
        progress_=None, report=REPORT_NOT_READY, files=FILES_ALL)),
    "run_blocked_exit2": ("run_detail.html.j2", run_ctx(R_BLOCKED, job_=job(
        RID["blocked"], "failed", exit_code=2, error="malvalid run exited with code 2"),
        progress_=None, report=REPORT_BLOCKED, files=FILES_ALL)),
    "run_aborted": ("run_detail.html.j2", run_ctx(R_ABORTED, job_=job(
        RID["aborted"], "finished", exit_code=1, options={"allow_pickle": False}),
        progress_=None, report=REPORT_ABORTED, files={"report_html": True, "report_json": True, "run_log": True,
                                                      "console_log": False})),
    # run detail: live states
    "run_running": ("run_detail.html.j2", run_ctx(R_RUNNING, job_=job(
        RID["running"], "running", options={"corpus": "ember_v3_2024", "seed": "0",
                                            "title": R_RUNNING["title"]}),
        progress_=PROGRESS_RUNNING, report=None, files=FILES_LOGS)),
    "run_loading": ("run_detail.html.j2", run_ctx(R_RUNNING, job_=job(RID["running"], "running"),
                                                  progress_=PROGRESS_LOADING, report=None, files=FILES_LOGS)),
    "run_queued": ("run_detail.html.j2", run_ctx(R_QUEUED, job_=job(RID["queued"], "queued", mode="path",
                                                                    started_at=None, pid=None),
                                                 progress_=None, report=None, files=FILES_NONE)),
    # run detail: no report
    "run_failed": ("run_detail.html.j2", run_ctx(R_FAILED, job_=job(
        RID["failed"], "failed", exit_code=2, finished_at="2026-09-27T08:15:07Z",
        error="AdapterError: adapter.py: class MLPDetector does not declare operating_threshold "
              "(a float in (0, 1)); see the submission contract"),
        progress_=progress(RID["failed"], "failed", ["pending"] * 6, [None] * 6), report=None, files=FILES_LOGS)),
    "run_cancelled": ("run_detail.html.j2", run_ctx(R_CANCELLED, job_=job(RID["cancelled"], "cancelled",
                                                                          exit_code=-15),
                                                    progress_=None, report=None, files=FILES_LOGS)),
    "run_interrupted": ("run_detail.html.j2", run_ctx(R_INTERRUPTED, job_=job(RID["interrupted"], "interrupted"),
                                                      progress_=PROGRESS_LOADING, report=None, files=FILES_LOGS)),
    # run detail: robustness
    "run_hostile": ("run_detail.html.j2", run_ctx(R_HOSTILE, job_=job(
        RID["hostile"], "finished", options={"title": HOSTILE, "class_name": HOSTILE_ATTR, HOSTILE: HOSTILE_IMG},
        adapter=HOSTILE_URL, error=HOSTILE, argv=["python", "-m", "malvalid", "run", "--adapter", HOSTILE]),
        progress_=None, report=REPORT_HOSTILE, files=FILES_ALL)),
    "run_hostile_running": ("run_detail.html.j2", run_ctx(
        dict(R_HOSTILE, status="running"), job_=None,
        progress_={"stage": HOSTILE, "message": HOSTILE_IMG, "module": {"code": HOSTILE, "title": HOSTILE_ATTR},
                   "modules": [{"id": HOSTILE_ATTR, "code": HOSTILE, "title": HOSTILE_IMG, "status": HOSTILE,
                                "duration_s": HOSTILE}]},
        report=None, files=None)),
    "run_malformed": ("run_detail.html.j2", run_ctx(R_SPARSE, job_={"options": "nope", "argv": "x"},
                                                    progress_={"modules": "nope", "module": "x"},
                                                    report=REPORT_MALFORMED, files={})),
    "run_sparse": ("run_detail.html.j2", run_ctx(R_SPARSE, job_=None, progress_=None, report=None, files=None)),
    "run_none": ("run_detail.html.j2", common("dashboard", "/runs/x", run=None, job=None, progress=None,
                                              report=None, files=None)),
    # compare
    "compare": ("compare.html.j2", common(
        "compare", "/compare?ids=a,b,c", runs=[R_READY, R_CONDITIONAL, R_BLOCKED], rows=_compare_rows(3),
        config_diff=CONFIG_DIFF, best_run_id=R_READY["run_id"])),
    "compare_same_policy": ("compare.html.j2", common(
        "compare", "/compare?ids=a,b", runs=[R_READY, R_NOT_READY], rows=_compare_rows(2), config_diff=[],
        best_run_id=R_READY["run_id"])),
    "compare_hostile": ("compare.html.j2", common(
        "compare", "/compare", runs=[R_HOSTILE, R_SPARSE],
        rows=[{"module_id": HOSTILE, "code": HOSTILE_ATTR, "title": HOSTILE_IMG,
               "cells": [{"status": HOSTILE, "score": HOSTILE, "gate_outcome": HOSTILE_IMG, "key_metric": HOSTILE,
                          "key_value": HOSTILE_ATTR, "threshold": HOSTILE}, None]},
              {"module_id": "x", "code": None, "title": None, "cells": None}],
        config_diff=[{"key": HOSTILE, "values": [HOSTILE_IMG, {"k": HOSTILE}]}, {"key": None, "values": None}],
        best_run_id=RID["hostile"])),
    "compare_one": ("compare.html.j2", common("compare", "/compare?ids=a", runs=[R_READY], rows=[], config_diff=[],
                                              best_run_id=None)),
    "compare_empty": ("compare.html.j2", common("compare", "/compare", runs=[], rows=[], config_diff=[],
                                                best_run_id=None)),
    "compare_none": ("compare.html.j2", common("compare", "/compare", runs=None, rows=None, config_diff=None,
                                               best_run_id=None)),
    # corpora / modules / system
    "corpora": ("corpora.html.j2", common("corpora", "/corpora", corpora=CORPORA + [CORPUS_HOSTILE, CORPUS_SPARSE])),
    "corpora_empty": ("corpora.html.j2", common("corpora", "/corpora", corpora=[])),
    "modules": ("modules.html.j2", common("modules", "/modules", modules=MODULES,
                                          default_config_text=DEFAULT_CONFIG_TEXT)),
    "modules_hostile": ("modules.html.j2", common("modules", "/modules", modules=[MODULE_HOSTILE, {}],
                                                  default_config_text=HOSTILE)),
    "modules_none": ("modules.html.j2", common("modules", "/modules", modules=None, default_config_text=None)),
    "system": ("system.html.j2", common("system", "/system", backends=BACKENDS, environment=ENVIRONMENT,
                                        settings=SETTINGS)),
    "system_no_isolation": ("system.html.j2", common(
        "system", "/system", backends=BACKENDS_NONE, environment={"python": HOSTILE, "libraries": {HOSTILE: None}},
        settings=dict(SETTINGS, host="0.0.0.0", allow_remote=True, runs_dir=HOSTILE))),
    "system_none": ("system.html.j2", common("system", "/system", backends=None, environment=None, settings=None)),
    # errors
    "error_404": ("error.html.j2", common(None, "/runs/nope", status=404, title="Run not found",
                                          message="There is no run called 20260101T000000Z-ffffffff in "
                                                  f"{RUNS_DIR}.")),
    "error_401": ("error.html.j2", {"status": 401, "title": None, "message": None, "request_path": "/"}),
    "error_403": ("error.html.j2", common(None, "/runs", status=403, title=None,
                                          message="The form's security token is missing or wrong.")),
    "error_413": ("error.html.j2", common(None, "/runs", status=413, title=None,
                                          message="model.onnx is larger than the 4096 MB upload limit.")),
    "error_500": ("error.html.j2", common(None, "/runs/new", status=500, title=HOSTILE, message=HOSTILE_IMG)),
    "error_none": ("error.html.j2", {}),
}

#: The context keys each template receives (docs/WEB_CONTRACT.md §5) besides the common ones.
TEMPLATE_KEYS: dict[str, tuple[str, ...]] = {
    "dashboard.html.j2": ("runs", "counts", "active"),
    "new_run.html.j2": ("modules", "corpora", "default_config_text", "defaults", "max_upload_mb", "errors", "form",
                        "inspect"),
    "run_detail.html.j2": ("run", "job", "progress", "report", "files"),
    "compare.html.j2": ("runs", "rows", "config_diff", "best_run_id"),
    "corpora.html.j2": ("corpora",),
    "modules.html.j2": ("modules", "default_config_text"),
    "system.html.j2": ("backends", "environment", "settings"),
    "error.html.j2": ("status", "title", "message"),
}
COMMON_KEYS = ("app_version", "csrf_token", "nav_active", "bind", "runs_dir", "request_path")


def all_none(template: str) -> dict[str, Any]:
    """A context for ``template`` in which every contract key is present and None."""
    return {k: None for k in COMMON_KEYS + TEMPLATE_KEYS[template]}


def get(name: str) -> tuple[str, dict[str, Any]]:
    """A deep copy of one case, safe to mutate."""
    template, ctx = CONTEXTS[name]
    return template, copy.deepcopy(ctx)
