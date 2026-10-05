"""Regressions for the web-UI review findings (one test per fixed finding where it is testable).

Correctness: crash-after-report messaging, CLI-run cancel button and failure/interrupt copy, queue
position and the elapsed clock of queued runs, dashboard paging and the active-only refresh poll.
UX: policy-mismatch warning on /compare, re-run with the same settings, readable failure messages,
no-model-file warning, onboarding (adapter skeleton, demo), synthetic/corpus caveats once, scorecard
check lists, the hidden X0 smoke-test module and focus under the sticky form bar.
"""

from __future__ import annotations

import copy
import html as htmllib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")

from malvalid.web.jobs import JobManager  # noqa: E402
from malvalid.web.settings import WebSettings  # noqa: E402
from malvalid.web.store import RunStore, write_json_atomic  # noqa: E402
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    FIXTURES,
    REPO,
    TOKEN,
    app,
    authed_client,
    captured,
    cli_run_dir,
    client,
    fake_adapter,
    fake_malvalid,
    job_json,
    make_app,
    post_run,
    run_id_from,
    settings,
    wait_for,
)

pytestmark = pytest.mark.web

STATIC = REPO / "src" / "malvalid" / "web" / "static"


def _report(name: str = "sample_report_ready.json") -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def _text(page: str) -> str:
    return " ".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", page)).split())


def _job(rid: str, status: str, **kw: Any) -> dict[str, Any]:
    rec = {"schema": "malvalid-job/1", "run_id": rid, "created_at": "2026-09-30T10:00:00Z",
           "started_at": "2026-09-30T10:00:01Z", "finished_at": "2026-09-30T10:05:00Z", "status": status,
           "argv": ["malvalid", "run", "--adapter", "/x/a.py"], "mode": "path", "adapter": "/x/a.py",
           "display_name": "x", "submission_id": None, "options": {}, "pid": None, "exit_code": None, "error": None}
    rec.update(kw)
    return rec


def _web_run(runs: Path, rid: str, job: dict[str, Any], report: dict[str, Any] | None = None) -> Path:
    d = runs / rid
    d.mkdir(parents=True)
    write_json_atomic(d / "job.json", job)
    if report is not None:
        (d / "report.json").write_text(json.dumps(report))
    return d


# ---- correctness-native-crash-after-report-mislabelled ---------------------------------------------


def test_a_crash_after_a_complete_report_names_the_signal(client, settings, tmp_path):
    rep = _report()
    jm = JobManager(settings, RunStore(settings.runs_dir))
    d = tmp_path / "rd"
    d.mkdir()
    reason = jm._failure_reason(-11, d, rep)
    assert reason == "MalValid crashed (SIGSEGV) after writing report.json"
    rid = "20260930T100000Z-000000c1"
    _web_run(settings.runs_dir, rid, _job(rid, "failed", exit_code=-11, error=reason), rep)
    t = _text(client.get(f"/runs/{rid}").text)
    assert "MalValid crashed (SIGSEGV) after writing this report" in t
    assert "exited with code 2" not in t


def test_an_exit_2_report_states_its_meaning_once(client, settings):
    rep = _report("sample_report.json")  # exit code 2
    meaning = rep["gate"]["exit_meaning"]
    rid = "20260930T100000Z-000000c2"
    _web_run(settings.runs_dir, rid, _job(rid, "failed", exit_code=2, error=f"exit code 2: {meaning}"), rep)
    t = _text(client.get(f"/runs/{rid}").text)
    assert "The run exited with code 2:" in t
    assert t.count(_text(meaning)) == 2  # once in the failure callout, once as the CI exit-code fact


# ---- correctness-cli-run-cancel-button-409 + cli copy ------------------------------------------------


@pytest.mark.posix
def test_a_running_cli_run_has_no_cancel_button(client, settings):
    d = settings.runs_dir / "cli-live"
    d.mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "malvalid", "run"],
                            start_new_session=True)
    try:
        write_json_atomic(d / "progress.json", {"schema": "malvalid-progress/1", "run_id": "r", "stage": "modules",
                                                "pid": proc.pid, "started_at": "2026-09-30T10:00:00Z",
                                                "modules": []})
        page = client.get("/runs/cli-live").text
        assert client.get("/api/runs/cli-live").json()["summary"]["status"] == "running"
        assert "/api/runs/cli-live/cancel" not in page
        assert "stop it in its terminal" in _text(page)
    finally:
        proc.kill()
        proc.wait()


def test_a_failed_cli_run_shows_its_cause_and_interrupted_copy_is_neutral(client, settings):
    d = settings.runs_dir / "cli-syntax"
    d.mkdir(parents=True)
    write_json_atomic(d / "progress.json", {"schema": "malvalid-progress/1", "run_id": "r", "stage": "failed",
                                            "message": "run failed: importing the adapter adapter.py failed: "
                                                       "SyntaxError: invalid syntax", "modules": []})
    assert "SyntaxError: invalid syntax" in _text(client.get("/runs/cli-syntax").text)
    d = settings.runs_dir / "cli-killed"
    d.mkdir()
    write_json_atomic(d / "progress.json", {"schema": "malvalid-progress/1", "run_id": "r", "stage": "modules",
                                            "pid": 999999999, "modules": []})
    t = _text(client.get("/runs/cli-killed").text)
    assert "Interrupted" in t and "malvalid server stopped" not in t and "process ended before it finished" in t


# ---- ux-4: queued runs ------------------------------------------------------------------------------


def test_a_queued_run_shows_submitted_time_and_queue_position(tmp_path, fake_malvalid):
    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=tmp_path / "runs", token=TOKEN, port=8765, cancel_grace_s=1.0, max_concurrent=1,
                    command_prefix=(sys.executable, str(fake_malvalid)))
    with authed_client(make_app(s)) as c:
        first = run_id_from(post_run(c, s, {"mode": "path", "adapter_path": str(fake_adapter(tmp_path, "sleep")),
                                            "title": "First run"}))
        wait_for(lambda: job_json(s, first)["status"] == "running", what="first run to start")
        second = run_id_from(post_run(c, s, {"mode": "path", "adapter_path": str(fake_adapter(tmp_path, "ok"))}))
        page = c.get(f"/runs/{second}").text
        assert re.search(r'data-started=""', page)  # the clock does not count queue time as run time
        t = _text(page)
        assert "Submitted" in t and "Queued for" in t
        assert "Waiting for 1 run ahead of it" in t and re.search(r"\(\s*First run\s+running\s*\)", t)
        c.post(f"/api/runs/{first}/cancel", headers={"x-csrf-token": s.csrf_token, "accept": "application/json"})


def test_the_elapsed_clock_restarts_when_a_queued_run_starts():
    js = (STATIC / "app.js").read_text()
    assert "data-live-elapsed-label" in js and "queuedSince" in js
    assert re.search(r"curStatus === \"queued\" \|\| isNaN\(started\)", js)


# ---- ux-5: failure messages and re-run ---------------------------------------------------------------


def test_failure_messages_wrap_and_split_into_items(client, settings):
    rid = "20260930T100000Z-000000c3"
    err = ("class Foo does not meet the submission contract: - training_hashes_path 'h.txt' does not exist "
           "- model_path 'm.txt' does not exist Traceback (most recent call last): File \"x.py\", line 1")
    _web_run(settings.runs_dir, rid, _job(rid, "failed", exit_code=2, error=err,
                                          options={"no_sandbox": True, "title": "T", "only": ["performance"],
                                                   "config": "/p/gate.yaml", "config_source": "path"}))
    page = client.get(f"/runs/{rid}").text
    items = re.findall(r'<ul class="error-items">(.*?)</ul>', page, re.S)
    assert items and items[0].count("<li") == 2
    assert "<summary>Traceback</summary>" in page
    assert f'href="/runs/new?from={rid}"' in page
    css = (STATIC / "app.css").read_text()
    assert re.search(r"pre\.code\.error-text[^{]*\{[^}]*white-space:\s*pre-wrap", css)


def test_rerun_prefills_the_new_run_form(client, app, settings):
    rid = "20260930T100000Z-000000c4"
    _web_run(settings.runs_dir, rid, _job(rid, "failed", exit_code=2, error="boom", mode="path",
                                          adapter="/home/me/adapter.py",
                                          options={"no_sandbox": True, "title": "T", "only": ["performance"],
                                                   "seed": 7, "class_name": "Det", "config": "/p/gate.yaml",
                                                   "config_source": "path", "fail_on": "not_ready"}))
    r = client.get(f"/runs/new?from={rid}")
    assert r.status_code == 200
    ctx = captured(app, "new_run.html.j2")
    f = ctx["form"]
    assert f["mode"] == "path" and f["adapter_path"] == "/home/me/adapter.py" and f["config_path"] == "/p/gate.yaml"
    assert f["no_sandbox"] == "on" and "confirm_no_sandbox" not in f  # the risk must be confirmed again
    assert f["only"] == "performance" and f["seed"] == "7" and f["class_name"] == "Det" and f["title"] == "T"
    assert "Settings copied from" in _text(r.text)
    up = "20260930T100000Z-000000c5"
    _web_run(settings.runs_dir, up, _job(up, "failed", mode="upload",
                                         options={"files": {"adapter": "ad.py", "models": ["m.txt"]}}))
    t = _text(client.get(f"/runs/new?from={up}").text)
    assert "choose them again (ad.py, m.txt)" in t
    assert client.get("/runs/new?from=no-such-run").status_code == 200


# ---- ux-6: upload without model files -----------------------------------------------------------------


def test_upload_without_model_files_is_flagged_before_submitting(client, settings, tmp_path):
    page = client.get("/runs/new").text
    assert "data-no-model-warning" in page
    js = (STATIC / "app.js").read_text()
    assert "missingModels()" in js and "win.confirm(" in js
    # the run title defaults to the detector class read from the adapter text, not just the file name
    ad = tmp_path / "ad.py"
    ad.write_text("from malvalid.adapter import BaseDetector\n\nclass MyLgbm(BaseDetector):\n    pass\n")
    r = post_run(client, settings, {"mode": "path", "adapter_path": str(ad)})
    assert job_json(settings, run_id_from(r))["display_name"] == "MyLgbm · ad.py"


# ---- ux-1: compare ---------------------------------------------------------------------------------


def test_compare_flags_different_policies_and_does_not_crown_a_winner(client, settings):
    a = _report()
    b = copy.deepcopy(a)
    b["run"]["id"] = "other"
    b["config"]["modules"]["performance"]["max_fpr"] = 0.05
    b["verdict"]["score"] = 99.0
    cli_run_dir(settings.runs_dir, "cmp-a", report__json=a)
    cli_run_dir(settings.runs_dir, "cmp-b", report__json=b)
    page = client.get("/compare?ids=cmp-a,cmp-b").text
    assert "data-policy-mismatch" in page and "different gate policies" in _text(page)
    assert "max_fpr" in page
    assert "Highest score" not in page and "col-best" not in page
    assert "data-verdict-summary" in page
    # same policy: the best run is marked and the summary says which is safer to ship
    c2 = copy.deepcopy(a)
    c2["run"]["id"] = "third"
    c2["verdict"]["score"] = 70.0
    cli_run_dir(settings.runs_dir, "cmp-c", report__json=c2)
    page = client.get("/compare?ids=cmp-a,cmp-c").text
    assert "data-policy-mismatch" not in page and "Highest score" in page and "Which is safer to ship" in page


def test_compare_explains_a_blocked_run(client, settings):
    cli_run_dir(settings.runs_dir, "cmp-ready", "sample_report_ready.json")
    cli_run_dir(settings.runs_dir, "cmp-blocked", "sample_report_aborted.json")
    t = _text(client.get("/compare?ids=cmp-ready,cmp-blocked").text)
    assert "because M0 model file safety aborted the run" in t


def test_compare_table_is_a_focusable_region():
    tpl = (REPO / "src" / "malvalid" / "web" / "templates" / "compare.html.j2").read_text()
    assert 'class="cmp-wrap" tabindex="0" role="region" aria-label=' in tpl
    css = (STATIC / "app.css").read_text()
    assert "table.cmp .rh-title" in css


# ---- ux-7 / ux-8 / ux-9: run page details ---------------------------------------------------------


def test_scorecard_lists_every_check_and_disabled_modules(client, settings):
    cli_run_dir(settings.runs_dir, "cli-ready", "sample_report_ready.json")
    page = client.get("/runs/cli-ready").text
    assert "+1 more" not in page and "All 2 checks" in page
    assert "Not run (off in this policy): M6" in _text(page)


def test_synthetic_and_corpus_caveats(client, settings):
    rep = _report()
    rep["corpus"]["synthetic"] = True
    rep["corpus"]["name"] = "synthetic_v2"
    cli_run_dir(settings.runs_dir, "cli-synth", report__json=rep)
    assert "data-synthetic-badge" in client.get("/runs/cli-synth").text
    assert "Synthetic data" in client.get("/").text

    err = ("no canonical corpus in the model's feature_version is loaded (corpus 'ember_v3_2024' is in "
           "feature space ember_v3)")
    rep = _report()
    rep["run"]["id"] = "mism"
    rep["corpus"]["error"] = err
    rep["warnings"] = [err, "another warning"]
    for m in rep["modules"]:
        if m["module_id"] in ("drift", "membership_inf"):
            m.update(status="skipped", skip_reason=err, checks=[])
    rep["verdict"]["verdict"] = "blocked"
    rep["gate"]["hard_gates"][1]["gate_outcome"] = "not_evaluated"
    cli_run_dir(settings.runs_dir, "cli-mism", report__json=rep)
    page = client.get("/runs/cli-mism").text
    t = _text(page)
    assert t.count("Skipped: no usable corpus (see above)") == 2
    assert t.count("is in feature space ember_v3") <= 2  # the corpus callout (+ a blocker line), not 4-6 times
    assert "another warning" in t
    assert "a hard gate could not be evaluated (M1 Performance & calibration)" in t
    assert "a hard gate failed, a test errored, or the run was aborted" not in t


def test_the_smoke_test_module_is_not_listed(client, settings):
    assert "Pipeline smoke test" not in client.get("/modules").text
    assert "Pipeline smoke test" not in client.get("/runs/new").text
    r = post_run(client, settings, {"mode": "path", "adapter_path": "/nope.py", "only": "M9"})
    assert r.status_code == 400 and "X0=dummy" not in r.text and "M9" in r.text


# ---- ux-3: onboarding --------------------------------------------------------------------------------


def test_onboarding_offers_an_adapter_skeleton_and_the_demo(client, settings):
    from malvalid.web.onboarding import ADAPTER_SKELETON, demo_submission

    assert "class MyDetector(BaseDetector)" in ADAPTER_SKELETON
    compile(ADAPTER_SKELETON, "skeleton.py", "exec")  # it is valid Python
    for path in ("/", "/runs/new"):
        page = client.get(path).text
        assert "data-adapter-skeleton" in page and "class MyDetector(BaseDetector)" in page
        if demo_submission() is not None:
            assert "data-demo-form" in page and "Run the synthetic demo" in page


def test_the_demo_button_submits_a_path_mode_run(client, settings):
    from malvalid.web.onboarding import demo_submission

    demo = demo_submission()
    if demo is None:
        pytest.skip("examples/synthetic_demo is not in this checkout")
    r = post_run(client, settings, {"mode": "path", **demo})
    job = job_json(settings, run_id_from(r))
    assert job["mode"] == "path" and job["adapter"].endswith("synthetic_demo/adapter.py")


# ---- ux-2: focus under the sticky bar ---------------------------------------------------------------


def test_focused_fields_are_not_hidden_under_the_sticky_form_bar():
    css = (STATIC / "app.css").read_text()
    assert re.search(r"html:has\(\.form-actions\)\s*\{\s*scroll-padding-bottom:\s*7rem", css)
    assert re.search(r"\.run-form :is\([^)]*input[^)]*\)\s*\{\s*scroll-margin-bottom", css)


# ---- correctness-no-pagination-o-n-scan ----------------------------------------------------------------


def test_dashboard_pages_and_the_refresh_poll_asks_only_for_active_runs(client, settings):
    from malvalid.web import routes

    src = FIXTURES / "sample_report_ready.json"
    for i in range(routes.DASHBOARD_LIMIT + 5):
        d = settings.runs_dir / f"cli-{i:04d}"
        d.mkdir(parents=True)
        (d / "report.json").write_bytes(src.read_bytes())
    page = client.get("/").text
    assert "data-show-all" in page and f"Show all {routes.DASHBOARD_LIMIT + 5}" in _text(page)
    assert page.count('<tr data-run-id="cli-') == routes.DASHBOARD_LIMIT
    assert client.get("/?all=1").text.count('<tr data-run-id="cli-') == routes.DASHBOARD_LIMIT + 5
    assert client.get("/api/runs?active=1").json() == []
    assert len(client.get("/api/runs").json()) == routes.DASHBOARD_LIMIT + 5
    assert '"/api/runs?active=1"' in (STATIC / "app.js").read_text()
