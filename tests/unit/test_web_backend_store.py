"""Run store of the local web UI (``docs/WEB_CONTRACT.md`` §2, §5): ``RunSummary`` built from
report.json / job.json / progress.json, tolerant of partial and malformed files and of runs that
``malvalid run`` wrote from a terminal (source ``cli``); plus the compare view's rows, policy diff
and best run."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")

from malvalid.web import compare  # noqa: E402
from malvalid.web.routes import dashboard_counts  # noqa: E402
from malvalid.web.store import (  # noqa: E402
    SUMMARY_FIELDS,
    RunNotFound,
    RunStore,
    build_summary,
    read_json_file,
    write_json_atomic,
)
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    FIXTURES,
    app,
    captured,
    cli_run_dir,
    client,
    fake_malvalid,
    settings,
)

pytestmark = pytest.mark.web

REPORTS = {
    "blocked": "sample_report.json",
    "ready": "sample_report_ready.json",
    "conditional": "sample_report_conditional.json",
    "not_ready": "sample_report_not_ready.json",
    "aborted": "sample_report_aborted.json",
}


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / REPORTS.get(name, name)).read_text())


def job(run_id: str, status: str = "finished", **kw: Any) -> dict[str, Any]:
    rec = {"schema": "malvalid-job/1", "run_id": run_id, "created_at": "2026-09-30T10:00:00Z",
           "started_at": "2026-09-30T10:00:01Z", "finished_at": "2026-09-30T10:00:31Z", "status": status,
           "argv": ["malvalid", "run"], "mode": "upload", "adapter": "/x/subm/my_adapter.py",
           "display_name": "My detector", "submission_id": None, "options": {"title": None, "corpus": "toy_v1_corpus"},
           "pid": None, "exit_code": 0, "error": None}
    rec.update(kw)
    return rec


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    s = RunStore(tmp_path / "runs")
    s.ensure_root()
    return s


# --------------------------------------------------------------------------------------------------
# RunSummary
# --------------------------------------------------------------------------------------------------


def test_cli_run_summary_from_report_json(store):
    cli_run_dir(store.root, "cli-ready", REPORTS["ready"])
    rep = fixture("ready")
    s = store.summary("cli-ready")
    assert tuple(s) == SUMMARY_FIELDS  # every contract field, in order
    assert s["run_id"] == "cli-ready" and s["source"] == "cli" and s["status"] == "finished"
    assert s["verdict"] == "ready" and s["score"] == rep["verdict"]["score"]
    assert s["verdict_label"] == rep["verdict"]["label"] and s["coverage"] == rep["verdict"]["coverage"]
    assert s["exit_code"] == 0
    assert s["class_name"] == rep["model"]["class_name"] and s["model_kind"] == rep["model"]["model_kind"]
    assert s["feature_version"] == rep["model"]["feature_version"]
    assert s["operating_threshold"] == rep["model"]["operating_threshold"]
    assert s["adapter"] == rep["model"]["adapter_path"]
    assert s["corpus"] == rep["corpus"]["name"]
    assert s["created_at"] == rep["run"]["started_at"] and s["finished_at"] == rep["run"]["finished_at"]
    assert s["duration_s"] == pytest.approx(rep["run"]["duration_s"], abs=1e-3)
    assert s["n_modules"] == len(rep["modules"])
    assert s["n_pass"] == sum(m["status"] == "pass" for m in rep["modules"])
    assert sum(s[f"n_{k}"] for k in ("pass", "warn", "fail", "skipped", "error")) == s["n_modules"]
    assert s["display_name"]  # never empty


def test_cli_run_with_exit_code_2_is_failed_but_keeps_its_verdict(store):
    cli_run_dir(store.root, "cli-blocked", REPORTS["blocked"])
    s = store.summary("cli-blocked")
    assert s["status"] == "failed" and s["exit_code"] == 2
    assert s["verdict"] == "blocked" and s["score"] == 49.0
    assert s["title"] == fixture("blocked")["title"]  # raw; the template escapes it
    assert s["n_error"] == 1 and s["n_skipped"] == 1


def test_web_run_summary_prefers_job_json(store):
    d = cli_run_dir(store.root, "20260930T100000Z-0000abcd", REPORTS["conditional"])
    write_json_atomic(d / "job.json", job("20260930T100000Z-0000abcd", exit_code=0))
    s = store.summary("20260930T100000Z-0000abcd")
    assert s["source"] == "web" and s["status"] == "finished"
    assert s["display_name"] == "My detector" and s["adapter"] == "/x/subm/my_adapter.py"
    assert s["created_at"] == "2026-09-30T10:00:00Z"
    assert s["verdict"] == "conditional"


def test_queued_web_run_without_any_run_output(store):
    d = store.root / "20260930T100000Z-00000001"
    d.mkdir()
    write_json_atomic(d / "job.json", job(d.name, status="queued", started_at=None, finished_at=None, exit_code=None,
                                          display_name=None, options={"title": "Nightly", "corpus": "c1"}))
    s = store.summary(d.name)
    assert s["status"] == "queued" and s["source"] == "web"
    assert s["verdict"] is None and s["score"] is None and s["duration_s"] is None
    assert s["title"] == "Nightly" and s["display_name"] == "Nightly" and s["corpus"] == "c1"
    assert s["n_modules"] == 0


def test_running_web_run_uses_progress(store):
    d = store.root / "20260930T100000Z-00000002"
    d.mkdir()
    write_json_atomic(d / "job.json", job(d.name, status="running", finished_at=None, exit_code=None,
                                          started_at="2026-09-30T10:00:00Z"))
    write_json_atomic(d / "progress.json", {
        "schema": "malvalid-progress/1", "stage": "modules", "started_at": "2026-09-30T10:00:00Z",
        "modules": [{"id": "file_safety", "status": "pass"}, {"id": "performance", "status": "running"},
                    {"id": "drift", "status": "pending"}],
        "verdict": None})
    s = store.summary(d.name)
    assert s["status"] == "running" and s["n_modules"] == 3 and s["n_pass"] == 1
    assert s["duration_s"] is not None and s["duration_s"] > 0  # time so far
    assert s["verdict"] is None


def test_cli_run_in_progress_or_interrupted(store):
    # A live process whose command line mentions malvalid: still running.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "malvalid", "run"])
    try:
        d = store.root / "cli-live"
        d.mkdir()
        write_json_atomic(d / "progress.json", {"schema": "malvalid-progress/1", "stage": "modules",
                                                "pid": proc.pid, "modules": [], "started_at": "2026-09-30T10:00:00Z"})
        assert store.summary("cli-live")["status"] == "running"
        assert store.summary("cli-live")["source"] == "cli"
    finally:
        proc.kill()
        proc.wait()
    assert store.summary("cli-live")["status"] == "interrupted"  # process gone, no report
    # An unrelated live process (pid reuse) does not count as the run.
    d2 = store.root / "cli-reused"
    d2.mkdir()
    write_json_atomic(d2 / "progress.json", {"stage": "modules", "pid": 1})
    assert store.summary("cli-reused")["status"] == "interrupted"


@pytest.mark.parametrize("stage", ["done", "failed"])
def test_cli_run_that_ended_without_a_report_is_failed(store, stage):
    d = store.root / "cli-ended"
    d.mkdir()
    write_json_atomic(d / "progress.json", {"stage": stage, "pid": 999999999, "updated_at": "2026-09-30T10:05:00Z",
                                            "started_at": "2026-09-30T10:00:00Z"})
    s = store.summary("cli-ended")
    assert s["status"] == "failed" and s["finished_at"] == "2026-09-30T10:05:00Z" and s["duration_s"] == 300.0


BAD_JSON = [b"", b"{", b"not json", b"[1, 2, 3]", b'"a string"', b"null", b"\xff\xfe\x00garbage",
            b"[" * 100000 + b"]" * 100000]


# Short ids: the default id embeds the 200 KB value, and pytest puts the test id in the
# PYTEST_CURRENT_TEST environment variable, which Windows limits to 32767 characters.
@pytest.mark.parametrize("content", BAD_JSON, ids=[f"bad{i}" for i in range(len(BAD_JSON))])
def test_malformed_files_are_treated_as_absent(store, content):
    d = cli_run_dir(store.root, "cli-good", REPORTS["ready"])
    (d / "progress.json").write_bytes(content)
    (d / "job.json").write_bytes(content)
    s = store.summary("cli-good")
    assert s["verdict"] == "ready" and s["source"] == "cli"
    # a dir holding only broken files is listed as an unreadable run instead of vanishing
    # (correctness-malformed-run-files-crash-or-hide-runs)
    bad = store.root / "cli-bad"
    bad.mkdir()
    (bad / "report.json").write_bytes(content)
    rec = store.load("cli-bad")
    assert rec.problem and "report.json" in rec.problem and rec.report is None
    assert rec.summary["status"] == "failed" and "unreadable" in rec.summary["display_name"]
    assert rec.files["report_html"] is False
    assert sorted(x["run_id"] for x in store.summaries()) == ["cli-bad", "cli-good"]
    empty = store.root / "cli-empty"
    empty.mkdir()
    with pytest.raises(RunNotFound):  # no run files at all: not a run
        store.load("cli-empty")


def test_wrong_typed_fields_become_none(store):
    rep = {"schema_version": "malvalid-report/1",
           "run": {"started_at": 12, "finished_at": ["x"], "duration_s": "long"},
           "verdict": {"verdict": "excellent", "label": {"x": 1}, "score": "high", "coverage": math.inf},
           "gate": {"exit_code": "zero"}, "model": "a string", "corpus": [1], "config": None,
           "modules": ["not a dict", {"status": "pass"}, {"status": 3}], "title": ""}
    d = store.root / "cli-weird"
    d.mkdir()
    (d / "report.json").write_text(json.dumps(rep).replace("Infinity", "1e999"))
    s = store.summary("cli-weird")
    assert s["verdict"] is None and s["verdict_label"] is None and s["score"] is None and s["coverage"] is None
    assert s["exit_code"] is None and s["class_name"] is None and s["corpus"] is None and s["title"] is None
    assert s["n_modules"] == 2 and s["n_pass"] == 1
    assert s["display_name"] == "cli-weird"
    assert s["created_at"] is not None  # falls back to the directory's mtime


def test_foreign_schemas_are_ignored(store):
    d = store.root / "other-tool"
    d.mkdir()
    (d / "report.json").write_text(json.dumps({"schema_version": "someone-else/9", "verdict": {"verdict": "ready"}}))
    (d / "job.json").write_text(json.dumps({"schema": "celery/1", "status": "running"}))
    with pytest.raises(RunNotFound):
        store.load("other-tool")


def test_listing_skips_non_runs_and_sorts_newest_first(store, tmp_path):
    for i, (rid, started) in enumerate([("b-old", "2026-01-01T00:00:00Z"), ("a-new", "2026-09-01T00:00:00Z"),
                                        ("c-mid", "2026-05-01T00:00:00Z")]):
        d = cli_run_dir(store.root, rid, REPORTS["ready"])
        rep = json.loads((d / "report.json").read_text())
        rep["run"]["started_at"] = started
        (d / "report.json").write_text(json.dumps(rep))
    (store.root / "empty-dir").mkdir()
    (store.root / "stray.json").write_text("{}")
    (store.root / ".hidden").mkdir()
    cli_run_dir(store.root / ".hidden", "x", REPORTS["ready"])
    assert [s["run_id"] for s in store.summaries()] == ["a-new", "c-mid", "b-old"]
    assert "submissions" not in store.list_run_ids()


def test_summary_cache_follows_file_changes(store):
    d = cli_run_dir(store.root, "cli-x", REPORTS["ready"])
    assert store.summaries()[0]["verdict"] == "ready"
    rep = fixture("not_ready")
    time.sleep(0.01)
    (d / "report.json").write_text(json.dumps(rep) + "\n")
    assert store.summaries()[0]["verdict"] == "not_ready"


def test_live_job_record_overrides_job_json(store):
    d = store.root / "20260930T100000Z-00000003"
    d.mkdir()
    write_json_atomic(d / "job.json", job(d.name, status="queued"))
    live = job(d.name, status="running")
    assert store.summaries({d.name: live})[0]["status"] == "running"


def test_read_json_file_and_atomic_write(tmp_path):
    p = tmp_path / "x.json"
    write_json_atomic(p, {"a": 1, "when": tmp_path})
    assert read_json_file(p) == {"a": 1, "when": str(tmp_path)}
    assert [q.name for q in tmp_path.iterdir()] == ["x.json"]  # no temp files left
    assert read_json_file(tmp_path / "missing.json") is None
    assert read_json_file(tmp_path) is None
    big = tmp_path / "big.json"
    big.write_text(json.dumps({"k": "v" * 1000}))
    assert read_json_file(big, max_bytes=100) is None
    with pytest.raises(ValueError):
        write_json_atomic(tmp_path / "nan.json", {"x": float("nan")})
    assert not (tmp_path / "nan.json").exists()


def test_build_summary_with_nothing_at_all(tmp_path):
    s = build_summary("r1", tmp_path, None, None, None)
    assert tuple(s) == SUMMARY_FIELDS
    assert s["status"] == "failed" and s["display_name"] == "r1" and s["source"] == "cli"


def test_dashboard_counts():
    runs = [{"status": "finished", "verdict": "ready"}, {"status": "finished", "verdict": "blocked"},
            {"status": "failed", "verdict": "blocked"}, {"status": "running", "verdict": None},
            {"status": "queued", "verdict": None}, {"status": "failed", "verdict": None},
            {"status": "interrupted", "verdict": None}, {"status": "cancelled", "verdict": None},
            {"status": "finished", "verdict": "conditional"}, {"status": "finished", "verdict": "not_ready"}]
    # an exit-2 ("failed") run carrying a BLOCKED verdict counts as failed, not as a blocked model
    assert dashboard_counts(runs) == {"ready": 1, "conditional": 1, "not_ready": 1, "blocked": 1, "running": 2,
                                      "failed": 3}


def test_dashboard_context(client, app, settings):
    for name in ("ready", "blocked", "aborted"):
        cli_run_dir(settings.runs_dir, f"cli-{name}", REPORTS[name])
    d = settings.runs_dir / "20260930T100000Z-00000004"
    d.mkdir()
    write_json_atomic(d / "job.json", job(d.name, status="queued"))
    r = client.get("/")
    assert r.status_code == 200
    ctx = captured(app, "dashboard.html.j2")
    assert ctx["nav_active"] == "dashboard"
    assert {s["run_id"] for s in ctx["runs"]} == {"cli-ready", "cli-blocked", "cli-aborted", d.name}
    assert [s["run_id"] for s in ctx["active"]] == [d.name]
    # cli-blocked exited 2 (not trustworthy): a failed run; cli-aborted (exit 1) is a blocked model
    assert ctx["counts"]["ready"] == 1 and ctx["counts"]["blocked"] == 1 and ctx["counts"]["running"] == 1
    assert ctx["counts"]["failed"] == 1
    for k in ("app_version", "csrf_token", "bind", "runs_dir", "request_path"):
        assert k in ctx
    assert ctx["bind"] == "127.0.0.1:8765" and ctx["request_path"] == "/"
    api = client.get("/api/runs").json()
    assert [s["run_id"] for s in api] == [s["run_id"] for s in ctx["runs"]]
    assert all(tuple(s) == SUMMARY_FIELDS for s in api)


def test_run_detail_context_for_a_cli_run(client, app, settings):
    d = cli_run_dir(settings.runs_dir, "cli-ready", REPORTS["ready"], run__log="x\n")
    r = client.get("/runs/cli-ready")
    assert r.status_code == 200
    ctx = captured(app, "run_detail.html.j2")
    assert ctx["run"]["run_id"] == "cli-ready" and ctx["job"] is None and ctx["progress"] is None
    assert ctx["report"]["run"]["id"] == fixture("ready")["run"]["id"]
    assert ctx["files"] == {"report_html": False, "report_json": True, "run_log": True, "console_log": False}
    api = client.get("/api/runs/cli-ready").json()
    assert set(api) >= {"summary", "job", "progress"} and api["summary"]["run_id"] == "cli-ready"
    assert d.exists()


def test_api_run_tolerates_nan_in_files(client, settings):
    d = cli_run_dir(settings.runs_dir, "cli-nan", REPORTS["ready"])
    (d / "progress.json").write_text('{"stage": "done", "verdict": {"score": NaN}, "modules": []}')
    r = client.get("/api/runs/cli-nan")
    assert r.status_code == 200
    assert r.json()["progress"]["verdict"]["score"] is None


# --------------------------------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------------------------------


def _row(rows: list[dict[str, Any]], mid: str) -> dict[str, Any]:
    return next(r for r in rows if r["module_id"] == mid)


def test_compare_rows_key_metric_per_module():
    reps = [fixture("blocked"), fixture("ready"), fixture("conditional")]
    rows = compare.build_rows(reps)
    assert [r["code"] for r in rows] == ["M0", "M1", "M2", "M4", "M5", "M6", "M7", "P1"]
    for r in rows:
        assert len(r["cells"]) == 3
        for c in r["cells"]:
            assert {"status", "score", "gate_outcome", "key_metric", "key_value", "threshold"} <= set(c)

    m1 = _row(rows, "performance")
    blocked = reps[0]["modules"][1]
    assert m1["cells"][0]["key_metric"] == "fpr"
    assert m1["cells"][0]["key_value"] == blocked["metrics"]["fpr"]
    assert m1["cells"][0]["threshold"] == blocked["params"]["max_fpr"]
    assert m1["cells"][0]["passed"] is False and m1["cells"][0]["status"] == "fail"
    assert [e["metric"] for e in m1["cells"][0]["metrics"]] == ["fpr", "detection_rate"]
    assert m1["cells"][0]["metrics"][1]["threshold"] == blocked["params"]["min_detection"]
    axes = {a["module_id"]: a["score"] for a in reps[0]["verdict"]["axes"]}
    assert m1["cells"][0]["score"] == axes["performance"]

    m2 = _row(rows, "drift")
    assert m2["cells"][2]["key_metric"] == "aut_f1_weighted"
    assert m2["cells"][2]["threshold"] == reps[2]["config"]["modules"]["drift"]["min_aut_f1"]
    assert m2["cells"][2]["status"] == "warn"

    m4 = _row(rows, "membership_inf")
    assert m4["cells"][0]["key_metric"] == "advantage"
    assert m4["cells"][0]["key_value"] == reps[0]["modules"][3]["metrics"]["advantage"]
    assert m4["cells"][2]["status"] == "skipped" and m4["cells"][2]["key_metric"] is None

    m5 = _row(rows, "backdoor_screen")
    assert m5["cells"][0]["key_metric"] == "n_flagged_rules" and m5["cells"][0]["key_value"] == 2
    assert m5["cells"][0]["threshold"] == 0

    m6 = _row(rows, "extraction")
    assert m6["cells"][0]["status"] == "error"
    assert m6["cells"][1]["status"] is None  # not in that report
    assert m6["cells"][1]["key_metric"] is None

    m7 = _row(rows, "explanation")
    assert m7["cells"][1]["key_metric"] == "controllable_share"
    assert m7["cells"][1]["threshold"] == 0.5

    m0 = _row(rows, "file_safety")
    assert m0["cells"][0]["key_metric"] == "n_critical" and m0["cells"][0]["key_value"] == 0


def _module(mid: str, code: str, checks: list[dict[str, Any]], metrics: dict[str, Any], params: dict[str, Any],
            status: str = "pass") -> dict[str, Any]:
    return {"module_id": mid, "code": code, "title": mid, "status": status, "gate_outcome": "passed",
            "checks": checks, "metrics": metrics, "params": params, "score": 0.8}


def _check(name: str, metric: str, value: float | None, op: str, threshold: float, passed: bool | None) -> dict:
    return {"name": name, "metric": metric, "value": value, "op": op, "threshold": threshold, "passed": passed}


def test_compare_key_metrics_as_the_modules_write_them():
    """Result keys of M5 (dynamic triggers), M6 and M7 as src/malvalid/modules/*.py produce them."""
    # M5 with only the trigger check (no tree access): max_trigger_drop.
    m5 = _module("backdoor_screen", "M5", [_check("max_trigger_drop", "max_trigger_drop", 0.31, "<=", 0.2, False)],
                 {"max_trigger_drop": 0.31, "n_triggers_tested": 3}, {"max_trigger_drop": 0.2}, "warn")
    # M5 with a static scan whose value is missing and a trigger check that has one.
    m5b = _module("backdoor_screen", "M5", [_check("n_flagged_rules", "n_flagged_rules", None, "<=", 0, None),
                                            _check("max_trigger_drop", "max_trigger_drop", 0.05, "<=", 0.2, True)],
                  {}, {"max_trigger_drop": 0.2})
    # M6: GateCheck("surrogate_fidelity", metric="fidelity") at fidelity_budget queries.
    m6 = _module("extraction", "M6", [_check("surrogate_fidelity", "fidelity", 0.97, "<=", 0.95, False)],
                 {"fidelity": 0.97, "fidelity_budget": 10000, "fidelity_budget_effective": 10000},
                 {"max_fidelity": 0.95, "fidelity_budget": 10000}, "warn")
    # M6 without a check (e.g. read from metrics only): threshold from params.
    m6b = _module("extraction", "M6", [], {"fidelity": 0.5}, {"max_fidelity": 0.9})
    m7 = _module("explanation", "M7", [_check("controllable_share", "controllable_share", 0.2, "<=", 0.5, True),
                                       _check("top_feature_share", "top_feature_share", 0.4, "<=", 0.3, False)],
                 {"controllable_share": 0.2, "top_feature_share": 0.4}, {"max_controllable_share": 0.5})
    m4 = _module("membership_inf", "M4", [_check("membership_advantage", "advantage", 0.12, "<=", 0.1, False)],
                 {"advantage": 0.12}, {"max_advantage": 0.1}, "warn")
    m0 = _module("file_safety", "M0", [_check("n_critical", "n_critical", 0, "<=", 0, True),
                                       _check("n_high", "n_high", 2, "<=", 0, False)], {"n_critical": 0}, {}, "fail")
    plugin = _module("my_plugin", "P9", [_check("a", "a", 1.0, ">=", 0.5, True), _check("b", "b", 0.1, ">=", 0.5, False)],
                     {}, {})

    def primary(mod):
        c = compare.build_cell(mod, None)
        return c["key_metric"], c["key_value"], c["threshold"], c["passed"]

    assert primary(m5) == ("max_trigger_drop", 0.31, 0.2, False)
    assert primary(m5b) == ("max_trigger_drop", 0.05, 0.2, True)
    assert primary(m6) == ("fidelity", 0.97, 0.95, False)
    assert primary(m6b) == ("fidelity", 0.5, 0.9, None)
    # A failed key check leads the cell (as on the run page): M7 failed on top_feature_share.
    assert primary(m7) == ("top_feature_share", 0.4, 0.3, False)
    assert [e["metric"] for e in compare.build_cell(m7, None)["metrics"]] == ["top_feature_share", "controllable_share"]
    m7ok = _module("explanation", "M7", [_check("controllable_share", "controllable_share", 0.2, "<=", 0.5, True),
                                         _check("top_feature_share", "top_feature_share", 0.1, "<=", 0.3, True)],
                   {"controllable_share": 0.2, "top_feature_share": 0.1}, {"max_controllable_share": 0.5})
    assert primary(m7ok) == ("controllable_share", 0.2, 0.5, True)  # all passed: the declared order
    # M1 failing on detection while FPR passes (the EMBER2018 XGBoost example): detection_rate leads.
    m1 = _module("performance", "M1", [_check("fpr", "fpr", 0.0089, "<=", 0.01, True),
                                       _check("detection_rate", "detection_rate", 0.925, ">=", 0.95, False)],
                 {"fpr": 0.0089, "detection_rate": 0.925}, {"max_fpr": 0.01, "min_detection": 0.95}, "fail")
    assert primary(m1) == ("detection_rate", 0.925, 0.95, False)
    assert primary(m4) == ("advantage", 0.12, 0.1, False)
    assert primary(m0) == ("n_high", 2, 0, False)  # M0: the failed check is what matters
    assert primary(plugin) == ("b", 0.1, 0.5, False)
    assert compare.build_cell(m7, None)["score"] == 80.0  # no verdict axes: the module score x 100
    assert compare.build_cell(m0, None)["score"] is None  # M0 is not scored


def test_compare_config_diff_and_best_run():
    a, b, c = fixture("ready"), fixture("ready"), fixture("conditional")
    b["config"]["modules"]["performance"]["max_fpr"] = 0.001
    b["gate"]["fail_on"] = "not_ready"
    diff = {d["key"]: d["values"] for d in compare.config_diff([a, b, c])}
    assert diff["modules.performance.max_fpr"] == [0.01, 0.001, 0.01]
    assert diff["fail_on"] == [a["gate"]["fail_on"], "not_ready", c["gate"]["fail_on"]]
    assert "source_path" not in diff
    assert compare.config_diff([a, copy.deepcopy(a)]) == []
    assert compare.config_diff([a, None]) == []

    runs = [{"run_id": "x", "verdict": "blocked", "score": 49.0}, {"run_id": "y", "verdict": "conditional", "score": 90.0},
            {"run_id": "z", "verdict": "ready", "score": 70.0}, {"run_id": "w", "verdict": None, "score": 99.0},
            {"run_id": "v", "verdict": "ready", "score": 71.0, "coverage": 0.5}]
    assert compare.best_run_id(runs) == "v"
    assert compare.best_run_id([{"run_id": "w", "verdict": None}]) is None


def test_compare_page(client, app, settings):
    for name in ("ready", "blocked", "conditional"):
        cli_run_dir(settings.runs_dir, f"cli-{name}", REPORTS[name])
    r = client.get("/compare?ids=cli-ready,cli-blocked&ids=cli-conditional")
    assert r.status_code == 200
    ctx = captured(app, "compare.html.j2")
    assert ctx["nav_active"] == "compare"
    assert [s["run_id"] for s in ctx["runs"]] == ["cli-ready", "cli-blocked", "cli-conditional"]
    assert ctx["best_run_id"] == "cli-ready"
    assert ctx["rows"] and all(len(row["cells"]) == 3 for row in ctx["rows"])
    assert any(d["key"] == "corpus.content_hash" for d in ctx["config_diff"])


def test_compare_page_needs_two_to_six_existing_runs(client, app, settings):
    for i in range(7):
        cli_run_dir(settings.runs_dir, f"cli-{i}", REPORTS["ready"])
    r = client.get("/compare?ids=cli-0")
    assert r.status_code == 400
    ctx = captured(app, "compare.html.j2")
    assert ctx["rows"] == [] and ctx["errors"]
    r = client.get("/compare?ids=cli-0,nope,../etc")
    assert r.status_code == 400
    assert any("nope" in e for e in captured(app, "compare.html.j2")["errors"])
    r = client.get("/compare?ids=" + ",".join(f"cli-{i}" for i in range(7)))
    assert r.status_code == 200
    ctx = captured(app, "compare.html.j2")
    assert len(ctx["runs"]) == 6 and any("at most 6" in e for e in ctx["errors"])
    r = client.get("/compare")  # the picker
    assert r.status_code == 200 and captured(app, "compare.html.j2")["runs"] == []


# ---- regressions from the review ------------------------------------------------------------------


def _report(name: str = "ready") -> dict[str, Any]:
    return json.loads((FIXTURES / REPORTS[name]).read_text())


def test_a_stale_report_in_a_reused_out_dir_is_ignored(store):
    """Regression (correctness-stale-report-in-reused-out-dir): a second `malvalid run` into the same --out
    dir that fails (or is still running) must not show the first run's verdict as current."""
    rep = _report("blocked")
    d = cli_run_dir(store.root, "cli-rerun", report__json=rep)
    write_json_atomic(d / "progress.json", {"schema": "malvalid-progress/1", "run_id": "second-run",
                                            "stage": "failed", "message": "run failed: importing the adapter "
                                            "adapter.py failed: SyntaxError: invalid syntax",
                                            "started_at": "2026-09-30T10:00:00Z",
                                            "updated_at": "2026-09-30T10:00:05Z", "pid": 999999999})
    rec = store.load("cli-rerun")
    assert rec.stale_report and rec.report is None
    assert rec.summary["status"] == "failed" and rec.summary["verdict"] is None and rec.summary["score"] is None
    assert rec.files["report_html"] is False and rec.files["report_json"] is False
    assert store.summary("cli-rerun")["verdict"] is None
    # the same run id in both: the report is this run's
    write_json_atomic(d / "progress.json", {"schema": "malvalid-progress/1", "run_id": rep["run"]["id"],
                                            "stage": "done"})
    rec = store.load("cli-rerun")
    assert not rec.stale_report and rec.summary["verdict"] == "blocked"


def test_malformed_fields_never_hide_a_run(store):
    """Regression (correctness-malformed-run-files-crash-or-hide-runs)."""
    good_progress = {"schema": "malvalid-progress/1", "run_id": "x", "stage": "modules",
                     "modules": [{"id": "performance", "code": "M1", "title": "P", "status": ["x"]}],
                     "pid": 10 ** 30}
    cli_run_dir(store.root, "pv-weird", progress__json=good_progress)
    rep = _report()
    rep["modules"][0]["status"] = ["x"]
    cli_run_dir(store.root, "rv-weird", report__json=rep)
    cli_run_dir(store.root, "empty-obj", report__json={})
    cli_run_dir(store.root, "foo-obj", report__json={"foo": 1})
    listed = {s["run_id"]: s for s in store.summaries()}
    assert set(listed) == {"pv-weird", "rv-weird", "empty-obj", "foo-obj"}
    assert listed["pv-weird"]["status"] == "interrupted"  # pid 10**30 is not alive (no OverflowError)
    for rid in ("empty-obj", "foo-obj"):  # the listing and the full read agree: unreadable, not "finished"
        assert listed[rid]["status"] == "failed" and store.load(rid).problem


def test_pid_alive_rejects_absurd_pids():
    from malvalid.web.store import pid_alive

    assert pid_alive(10 ** 30) is False and pid_alive(2 ** 31) is False and pid_alive(-5) is False


def test_malformed_run_pages_render(client, settings):
    good_progress = {"schema": "malvalid-progress/1", "run_id": "x", "stage": "modules",
                     "modules": [{"id": "performance", "code": "M1", "title": "P", "status": ["x"]}],
                     "module": ["not", "a", "dict"], "verdict": "nope", "pid": 10 ** 30}
    cli_run_dir(settings.runs_dir, "pv-weird", progress__json=good_progress)
    rep = _report()
    rep["modules"][0]["status"] = {"x": 1}
    rep["modules"][1]["checks"] = "nope"
    cli_run_dir(settings.runs_dir, "rv-weird", report__json=rep)
    cli_run_dir(settings.runs_dir, "empty-obj", report__json={})
    cli_run_dir(settings.runs_dir, "bad-json", report__json=b"{not json")
    listed = {s["run_id"] for s in client.get("/api/runs").json()}
    assert listed >= {"pv-weird", "rv-weird", "empty-obj", "bad-json"}
    for rid in ("pv-weird", "rv-weird", "empty-obj", "bad-json"):
        assert client.get(f"/runs/{rid}").status_code == 200, rid
        assert client.get(f"/api/runs/{rid}").status_code == 200, rid
    page = client.get("/runs/bad-json").text
    assert "could not be read" in page
    assert client.get("/compare?ids=rv-weird,empty-obj").status_code == 200


@pytest.mark.parametrize("name", ["my model v2", "résumé", "_under"])
def test_cli_run_dirs_with_other_names_are_listed(client, settings, name):
    """Regression (correctness-cli-run-dirs-unlisted-and-wrong-copy): listed under a URL-safe alias."""
    from malvalid.web.store import alias_for

    cli_run_dir(settings.runs_dir, name, REPORTS["ready"], run__log="log\n")
    runs = client.get("/api/runs").json()
    assert len(runs) == 1
    rid = runs[0]["run_id"]
    assert rid == alias_for(name) and rid.startswith("enc-")
    assert client.get(f"/runs/{rid}").status_code == 200
    assert client.get(f"/runs/{rid}/run.log").text == "log\n"
    r = client.post(f"/api/runs/{rid}/delete", headers={"x-csrf-token": settings.csrf_token,
                                                         "accept": "application/json"})
    assert r.status_code == 200 and not (settings.runs_dir / name).exists()


def test_aliases_cannot_escape_the_runs_dir(store):
    from malvalid.web.store import alias_for, name_for_alias

    for evil in ("..", ".", "../x", "a/b", ".hidden", "submissions"):
        alias = "enc-" + evil.encode().hex()
        assert alias_for(evil) is None or name_for_alias(alias) is None or "/" not in evil
        with pytest.raises(RunNotFound):
            store.run_dir(alias)
    assert name_for_alias("enc-" + "ok-name".encode().hex()) is None  # valid ids need no alias
    assert name_for_alias("enc-zz") is None


def test_same_second_submissions_keep_queue_order(store):
    for i, rid in enumerate(["q-b", "q-a", "q-d", "q-c"]):  # random-looking ids, submitted in this order
        d = store.root / rid
        d.mkdir()
        write_json_atomic(d / "job.json", {"schema": "malvalid-job/1", "run_id": rid, "status": "queued",
                                            "created_at": "2026-09-30T10:00:00Z", "seq": 1000 + i})
    assert [s["run_id"] for s in store.summaries()] == ["q-c", "q-d", "q-a", "q-b"]  # newest first
