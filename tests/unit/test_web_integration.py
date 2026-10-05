"""Regressions found by driving the real `malvalid serve` end to end (web_integrator).

Each test pins one backend/frontend mismatch that the unit tests of either side missed: the frontend
formatting the job options the backend actually writes, the web run id reaching report.json, M0's
weight, integer settings, the compare page on runs from different corpora, the validation error
shape app.js must render, and a few layout rules seen in headless screenshots.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

pytest.importorskip("starlette")

from malvalid.web import filters as F  # noqa: E402
from malvalid.web.forms import RunRequest  # noqa: E402
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    FIXTURES,
    PORT,
    app,
    captured,
    cli_run_dir,
    client,
    fake_malvalid,
    settings,
)

STATIC = Path(F.__file__).resolve().parent / "static"


def _text(html: str) -> str:
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


# ---- job options: the keys web/forms.py writes -------------------------------------------------


def _job(req: RunRequest) -> dict:
    return {"schema": "malvalid-job/1", "run_id": "r1", "status": "finished", "mode": req.mode,
            "adapter": str(req.adapter), "options": req.options(), "argv": ["python", "-m", "malvalid", "run"]}


def test_job_view_formats_the_upload_options_the_backend_writes(tmp_path):
    sub = tmp_path / "submissions" / "s1"
    req = RunRequest(mode="upload", adapter=sub / "det.py", base_config=sub / "gate.yaml", config_source="upload",
                     uploaded={"adapter": "det.py", "models": ["model.txt", "extra.json"],
                               "manifest": "train_sha256.txt", "config": "gate.yaml"},
                     title="LightGBM rc-2", seed=3)
    rows = {r["label"]: r["value"] for r in F.job_view(_job(req))["options"]}
    assert rows["Uploaded files"] == "det.py, model.txt, extra.json, train_sha256.txt, gate.yaml"
    assert rows["Policy from"] == "an uploaded file"
    assert rows["Gate policy"] == str(sub / "gate.yaml")
    assert rows["Seed"] == "3" and rows["Run title"] == "LightGBM rc-2"
    assert not any(v.startswith(("{", "[")) for v in rows.values()), rows  # no raw JSON on the page


def test_job_view_path_mode_has_no_empty_files_row(tmp_path):
    req = RunRequest(mode="path", adapter=tmp_path / "det.py", base_config=None, config_source="default")
    rows = {r["label"]: r["value"] for r in F.job_view(_job(req))["options"]}
    assert "Uploaded files" not in rows and "Files" not in rows
    assert rows["Policy from"] == "the packaged default policy"
    assert "{}" not in rows.values()


# ---- the web run id reaches the run ------------------------------------------------------------


def test_run_argv_passes_the_web_run_id(tmp_path, settings):
    req = RunRequest(mode="path", adapter=tmp_path / "det.py", base_config=None, config_source="default")
    argv = req.run_argv(settings, None)
    assert ("--run-id", "{run_id}") in list(zip(argv, argv[1:]))
    assert ("--out", "{run_dir}") in list(zip(argv, argv[1:]))


def test_terminal_command_hides_the_internal_run_id():
    argv = ["/venv/bin/python", "-m", "malvalid", "run", "--adapter", "/a/det.py", "--out", "/r/x",
            "--run-id", "20260930T161255Z-7c4a5616", "--seed", "0"]
    assert F.cli_command(argv) == "malvalid run --adapter /a/det.py --out /r/x --seed 0"
    assert F.cli_command(argv[:-2] + ["--run-id=abc"]) == "malvalid run --adapter /a/det.py --out /r/x"


def test_submitted_job_records_its_own_run_id(client, settings):
    """POST /runs (fake malvalid): the stored argv carries --run-id <the run's directory name>."""
    from tests.unit.test_web_backend_support import fake_adapter, job_json, post_run, run_id_from

    ad = fake_adapter(settings.runs_dir.parent)
    rid = run_id_from(post_run(client, settings, {"mode": "path", "adapter_path": str(ad)}))
    argv = job_json(settings, rid)["argv"]
    assert argv[argv.index("--run-id") + 1] == rid
    assert "{run_id}" not in " ".join(argv)


# ---- pages -------------------------------------------------------------------------------------


def test_m0_is_a_precondition_without_a_weight(client, app):
    r = client.get("/modules")
    assert r.status_code == 200
    mods = {m["id"]: m for m in captured(app, "modules.html.j2")["modules"]}
    assert mods["file_safety"]["scored"] is False and mods["file_safety"]["weight"] is None
    assert mods["performance"]["scored"] is True and mods["performance"]["weight"] == 3.0
    m0_card = r.text.split('id="mod-file_safety"')[-1].split("</section>")[0] if 'id="mod-file_safety"' in r.text \
        else r.text[r.text.index("Model file safety"):][:3000]
    assert "precondition, not scored" in m0_card and "weight 1" not in m0_card


def test_system_page_shows_ports_without_thousands_separators(client):
    t = _text(client.get("/system").text)
    assert f"port {PORT}" in t and f"{PORT:,}" not in t
    assert "127.0.0.1:8765, [::1]:8765, localhost:8765" in t  # a list, not JSON


def test_compare_warns_when_corpora_differ(client, settings):
    runs = settings.runs_dir
    cli_run_dir(runs, "cli-ready", "sample_report_ready.json")
    rep = json.loads((FIXTURES / "sample_report_ready.json").read_text())
    other = copy.deepcopy(rep)
    other["corpus"]["name"] = "synthetic_v2"
    cli_run_dir(runs, "cli-synth", None, report__json=other)
    same = client.get("/compare?ids=cli-ready,cli-ready")
    assert same.status_code in (200, 400) and "data-corpus-mismatch" not in same.text
    r = client.get("/compare?ids=cli-ready,cli-synth")
    assert r.status_code == 200
    assert "data-corpus-mismatch" in r.text and "evaluated on different corpora" in _text(r.text)
    assert "synthetic_v2" in _text(r.text)


def test_401_page_points_at_this_server_without_markdown(app):
    from starlette.testclient import TestClient

    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as anon:
        r = anon.get("/")
        assert r.status_code == 401
        t = _text(r.text)
        assert f"http://127.0.0.1:{PORT}/?token=" in t and "`" not in t
        j = anon.get("/api/runs")
        assert j.status_code == 401 and "`" not in j.text


def test_new_run_seed_defaults_to_the_policy(client):
    html = client.get("/runs/new").text
    tag = re.search(r'<input[^>]*id="seed"[^>]*>', html).group(0)
    assert 'value=""' in tag and 'placeholder="Policy default (0)"' in tag


def test_dashboard_calls_queued_plus_running_in_progress(client, settings):
    cli_run_dir(settings.runs_dir, "cli-ready", "sample_report_ready.json")
    html = client.get("/").text
    tiles = html[:html.index("All runs")] if "All runs" in html else html
    assert "In progress" in _text(tiles)  # the tile counts queued + running runs


# ---- static assets -----------------------------------------------------------------------------


def test_app_js_renders_validation_errors_that_come_with_no_checks():
    """/api/validate failures are {ok: false, checks: [], errors|error}: an empty checks list is the
    error branch (it used to fall through to "0 checks failed" and hide the messages)."""
    js = (STATIC / "app.js").read_text()
    assert "!d.checks.length" in js
    body = js[js.index("function renderValidation"):]
    assert body.index("!d.checks.length") < body.index("nFail")


def test_css_layout_rules_seen_in_screenshots():
    css = (STATIC / "app.css").read_text()
    assert ".grid-cards:not(:last-child) { margin-bottom: 1rem; }" in css  # corpora: gap before the next card
    assert ".validation { scroll-margin-bottom:" in css  # result not hidden under the sticky action bar
    assert ".fields-2 > .field:last-child, .fields-3 > .field:last-child { margin-bottom: 1rem; }" in css


def test_running_page_shows_the_live_console_tail():
    """A long module (TreeSHAP on EMBER: ~7 min) left the running page unchanged; the page now shows the
    console_tail that GET /api/runs/<id> returns, written with textContent (never as HTML)."""
    import jinja2

    from tests.fixtures import web_contexts as W

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(STATIC.parent / "templates")),
                             autoescape=True)
    env.filters.update(F.FILTERS)
    template, ctx = W.get("run_running")
    html = env.get_template(template).render(**ctx)
    assert "data-live-console" in html and 'class="more js-only live-console"' in html
    js = (STATIC / "app.js").read_text()
    body = js[js.index("function showConsole"):]
    assert "consoleNode.textContent = s" in body and "innerHTML" not in body[:body.index("\n    }\n")]
    assert "showConsole(d.console_tail)" in js


def test_compare_shows_why_runs_were_dropped(client, settings):
    """The route collects `errors` (unknown id, more than 6, fewer than 2) but the template used to
    ignore them: ticking 7 runs silently lost one and /compare?ids=a was a bare 400."""
    r = client.get("/compare?ids=nope")
    assert r.status_code == 400 and "data-compare-errors" in r.text
    assert "There is no run" in _text(r.text) and "Pick at least 2 runs to compare." in _text(r.text)
    for i in range(7):
        cli_run_dir(settings.runs_dir, f"cli-{i}", "sample_report_ready.json")
    r = client.get("/compare?ids=" + ",".join(f"cli-{i}" for i in range(7)))
    assert r.status_code == 200
    assert "Compare at most 6 runs at a time; showing the first 6." in _text(r.text)
    assert "cli-6" not in r.text.split("data-compare-errors", 1)[1].split("</table>", 1)[0]
    ok = client.get("/compare?ids=cli-0,cli-1")
    assert ok.status_code == 200 and "data-compare-errors" not in ok.text


def test_dashboard_pending_note_sits_on_its_own_line():
    html = (STATIC.parent / "templates" / "dashboard.html.j2").read_text()
    assert 'class="small muted cell-note">verdict pending' in html
    assert ".cell-note { display: block;" in (STATIC / "app.css").read_text()


def test_validate_names_the_users_policy_not_the_merged_copy():
    """validate-adapter reports the --config it got, which is the server's merged .malvalid-gate.yaml
    (policy + form options); the UI must name the policy the user actually chose."""
    from malvalid.web.api import _relabel_effective_config
    from malvalid.web.forms import EFFECTIVE_CONFIG_NAME

    def res():
        return {"checks": [{"name": "config", "detail": f"{EFFECTIVE_CONFIG_NAME} is valid for `malvalid run`"},
                           {"name": "load", "detail": f"/runs/submissions/x/{EFFECTIVE_CONFIG_NAME}: bad key"},
                           {"name": "other", "detail": "untouched"}, "not-a-dict"]}

    r = res()
    _relabel_effective_config(r, Path("/home/me/gate.yaml"))
    assert r["checks"][0]["detail"] == "gate.yaml (with this form's options) is valid for `malvalid run`"
    assert r["checks"][1]["detail"] == "gate.yaml (with this form's options): bad key"
    assert r["checks"][2]["detail"] == "untouched"
    r = res()
    _relabel_effective_config(r, None)
    assert r["checks"][0]["detail"].startswith("The default gate policy (with this form's options) is valid")
    assert EFFECTIVE_CONFIG_NAME not in json.dumps(r)


def test_scrollbars_follow_the_theme():
    """Headless Firefox drew a white scrollbar under the compare table in dark mode."""
    assert "html { scrollbar-color: var(--line-strong) transparent; }" in (STATIC / "app.css").read_text()
