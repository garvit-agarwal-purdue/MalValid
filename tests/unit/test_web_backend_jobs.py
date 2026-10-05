"""Job lifecycle of the local web UI (``docs/WEB_CONTRACT.md`` §2, §4): FIFO queue + worker threads,
``malvalid run`` in its own session with output in console.log, job.json state
(``queued → running → finished | failed | cancelled | interrupted``), cancel via the process group
(SIGTERM, then SIGKILL after the grace period), start-up reconciliation, delete, and one real
``malvalid run --no-sandbox`` on the toy adapter (its progress.json stages included)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import pytest
import yaml

pytest.importorskip("starlette")

import malvalid.testing  # noqa: E402,F401 - registers toy_v1 / toy_v1_corpus in this process
from malvalid.web.jobs import JobError, JobManager, JobSpec  # noqa: E402
from malvalid.web.settings import WebSettings  # noqa: E402
from malvalid.web.store import RunStore, pid_alive, write_json_atomic  # noqa: E402
from tests.fixtures.adapters.builders import make_adapter_dir  # noqa: E402
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    REPO,
    TOKEN,
    app,
    authed_client,
    captured,
    cli_run_dir,
    client,
    fake_adapter,
    fake_malvalid,
    fake_record,
    job_json,
    make_app,
    post_run,
    read_record,
    run_id_from,
    settings,
    toy_python_path,
    wait_for,
)

pytestmark = pytest.mark.web

JSON = {"accept": "application/json"}
TERMINAL = ("finished", "failed", "cancelled", "interrupted")


def submit(client, settings, tmp_path, mode: str = "ok", **data: Any) -> str:
    ad = fake_adapter(tmp_path, mode)
    return run_id_from(post_run(client, settings, {"mode": "path", "adapter_path": str(ad), **data}))


def status(settings: WebSettings, rid: str) -> str:
    return job_json(settings, rid)["status"]


def wait_status(settings: WebSettings, rid: str, *want: str, timeout: float = 30.0) -> dict[str, Any]:
    want = want or TERMINAL

    def check():
        try:
            j = job_json(settings, rid)
        except (OSError, ValueError):
            return None
        return j if j["status"] in want else None

    return wait_for(check, timeout=timeout, what=f"run {rid} to become {'/'.join(want)}")


def wait_running(settings: WebSettings, rid: str, timeout: float = 30.0) -> dict[str, Any]:
    """Wait until the run's process has started (job.json says ``running`` a moment before its pid is known)."""

    def check():
        try:
            j = job_json(settings, rid)
        except (OSError, ValueError):
            return None
        return j if j["status"] == "running" and isinstance(j.get("pid"), int) else None

    return wait_for(check, timeout=timeout, what=f"run {rid} to start its process")


def cancel(client, settings, rid: str):
    return client.post(f"/api/runs/{rid}/cancel", headers={"x-csrf-token": settings.csrf_token, **JSON})


def delete(client, settings, rid: str):
    return client.post(f"/api/runs/{rid}/delete", headers={"x-csrf-token": settings.csrf_token, **JSON})


def group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


# --------------------------------------------------------------------------------------------------
# Lifecycle with the fake malvalid
# --------------------------------------------------------------------------------------------------


def test_a_run_goes_from_queued_to_finished(client, app, settings, tmp_path, fake_record):
    rid = submit(client, settings, tmp_path, "ok", title="Fake run")
    assert rid[:8].isdigit() and rid[8] == "T" and rid[15] == "Z" and len(rid.split("-")[1]) == 8
    job = wait_status(settings, rid)
    assert job["status"] == "finished" and job["exit_code"] == 0 and job["error"] is None
    assert job["schema"] == "malvalid-job/1" and job["run_id"] == rid
    assert job["created_at"] <= job["started_at"] <= job["finished_at"]
    assert isinstance(job["pid"], int)
    for k in ("argv", "mode", "adapter", "display_name", "submission_id", "options"):
        assert k in job
    run_dir = settings.runs_dir / rid
    console = (run_dir / "console.log").read_text()
    assert console.startswith("$ ") and "fake malvalid run: done" in console
    call = [c for c in read_record(fake_record) if c["cmd"] == "run"][-1]
    # The directory malvalid serve was started in (like `malvalid run` typed there), never the run dir.
    assert call["cwd"] == str(settings.launch_dir) and call["cwd"] != str(run_dir)
    assert call["PYTHONSAFEPATH"] == "1"
    # the pages see it
    r = client.get(f"/runs/{rid}")
    assert r.status_code == 200
    ctx = captured(app, "run_detail.html.j2")
    assert ctx["run"]["status"] == "finished" and ctx["run"]["source"] == "web"
    assert ctx["run"]["verdict"] == "ready" and ctx["run"]["score"] == 91.5
    assert ctx["job"]["status"] == "finished" and ctx["progress"]["stage"] == "done"
    assert ctx["report"]["verdict"]["verdict"] == "ready"
    assert ctx["files"] == {"report_html": True, "report_json": True, "run_log": False, "console_log": True}
    api = client.get(f"/api/runs/{rid}").json()
    assert api["summary"]["status"] == "finished" and api["job"]["exit_code"] == 0
    assert "fake malvalid run: done" in api["console_tail"]
    assert client.get(f"/runs/{rid}/console.log").text == console


def test_extra_env_and_inherited_env_reach_the_run(tmp_path, fake_malvalid, fake_record, monkeypatch):
    monkeypatch.setenv("MALVALID_CORPUS_DIR", str(tmp_path / "corpora"))
    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=tmp_path / "runs", token=TOKEN, port=8765,
                    command_prefix=(sys.executable, str(fake_malvalid)), extra_env={"FAKE_EXTRA": "yes"})
    with authed_client(make_app(s)) as c:
        rid = submit(c, s, tmp_path)
        wait_status(s, rid)
    call = [x for x in read_record(fake_record) if x["cmd"] == "run"][-1]
    assert call["FAKE_EXTRA"] == "yes"


@pytest.mark.parametrize("mode,want,exit_code,error", [
    ("exit1", "finished", 1, None),
    ("exit2", "failed", 2, "my_adapter.py: class Foo does not declare feature_version"),
    ("no-report", "failed", 0, "without writing report.json"),
    pytest.param("crash", "failed", -9, "killed by SIGKILL", marks=pytest.mark.posix),  # SIGKILL
    ("blocked2", "failed", 2, "exit code 2: error: a module errored"),
])
def test_exit_codes_map_to_job_statuses(client, settings, tmp_path, mode, want, exit_code, error):
    rid = submit(client, settings, tmp_path, mode)
    job = wait_status(settings, rid)
    assert (job["status"], job["exit_code"]) == (want, exit_code)
    if error is None:
        assert job["error"] is None
    else:
        assert error in job["error"]


@pytest.mark.posix
def test_fifo_queue_with_one_worker_and_cancel(client, settings, tmp_path):
    first = submit(client, settings, tmp_path, "sleep")
    second = submit(client, settings, tmp_path, "ok")
    job = wait_running(settings, first)
    wait_for(lambda: (settings.runs_dir / first / "progress.json").exists(), what="the first run to start")
    assert status(settings, second) == "queued"  # max_concurrent = 1
    api = client.get("/api/runs").json()
    assert {r["run_id"]: r["status"] for r in api} == {first: "running", second: "queued"}

    # cancel the queued one: immediate, it never starts
    r = cancel(client, settings, second)
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert status(settings, second) == "cancelled"
    assert not (settings.runs_dir / second / "console.log").exists()

    # cancel the running one: SIGTERM to its process group
    pid = job["pid"]
    r = cancel(client, settings, first)
    assert r.status_code == 200
    done = wait_status(settings, first, *TERMINAL)
    assert done["status"] == "cancelled" and done["error"] == "cancelled by the user"
    assert done["exit_code"] == -15
    wait_for(lambda: group_gone(pid), what="the process group to be gone")
    # nothing left to cancel
    assert cancel(client, settings, first).status_code == 409
    assert cancel(client, settings, second).status_code == 409


@pytest.mark.posix
def test_cancel_escalates_to_sigkill(client, settings, tmp_path):
    rid = submit(client, settings, tmp_path, "ignore-term")
    job = wait_running(settings, rid)
    # the fake ignores SIGTERM from before it writes progress.json
    wait_for(lambda: (settings.runs_dir / rid / "progress.json").exists(), what="the run to start")
    t0 = time.monotonic()
    assert cancel(client, settings, rid).status_code == 200
    done = wait_status(settings, rid, *TERMINAL)
    assert done["status"] == "cancelled" and done["exit_code"] == -9
    assert time.monotonic() - t0 >= settings.cancel_grace_s * 0.9
    assert group_gone(job["pid"])


def test_cancel_a_run_page_form_redirects(client, settings, tmp_path):
    rid = submit(client, settings, tmp_path, "sleep")
    wait_status(settings, rid, "running")
    r = client.post(f"/api/runs/{rid}/cancel", data={"csrf": settings.csrf_token}, headers={"accept": "text/html"},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"/runs/{rid}"
    assert wait_status(settings, rid)["status"] == "cancelled"


def test_cancel_and_delete_of_unknown_or_cli_runs(client, settings):
    cli_run_dir(settings.runs_dir, "cli-run", "sample_report_ready.json")
    assert cancel(client, settings, "nope").status_code == 404
    assert delete(client, settings, "nope").status_code == 404
    assert cancel(client, settings, "..").status_code in (404, 405)
    r = cancel(client, settings, "cli-run")
    assert r.status_code == 409 and "not started from the web UI" in r.json()["message"]
    r = delete(client, settings, "cli-run")
    assert r.status_code == 200 and not (settings.runs_dir / "cli-run").exists()


def test_two_workers_run_two_jobs_at_once(tmp_path, fake_malvalid):
    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=tmp_path / "runs", token=TOKEN, port=8765, max_concurrent=2, cancel_grace_s=1.0,
                    command_prefix=(sys.executable, str(fake_malvalid)))
    with authed_client(make_app(s)) as c:
        a = submit(c, s, tmp_path, "sleep")
        b = submit(c, s, tmp_path, "sleep")
        wait_status(s, a, "running")
        wait_status(s, b, "running")
        for rid in (a, b):
            cancel(c, s, rid)
        for rid in (a, b):
            assert wait_status(s, rid)["status"] == "cancelled"


# --------------------------------------------------------------------------------------------------
# Delete
# --------------------------------------------------------------------------------------------------


def test_delete_is_refused_while_queued_or_running(client, settings, tmp_path):
    running = submit(client, settings, tmp_path, "sleep")
    queued = submit(client, settings, tmp_path, "ok")
    wait_status(settings, running, "running")
    for rid in (running, queued):
        r = delete(client, settings, rid)
        assert r.status_code == 409 and "cancel it first" in r.json()["message"]
        assert (settings.runs_dir / rid / "job.json").exists()
    cancel(client, settings, queued)
    cancel(client, settings, running)
    wait_status(settings, running)
    for rid in (running, queued):
        assert delete(client, settings, rid).status_code == 200
        assert not (settings.runs_dir / rid).exists()
    assert [r["run_id"] for r in client.get("/api/runs").json()] == []


def test_delete_removes_the_run_and_its_submission(client, settings, tmp_path):
    rid = run_id_from(post_run(client, settings, {"mode": "upload"},
                               [("adapter_file", ("my_adapter.py", b"# FAKE: ok\n")),
                                ("model_files", ("model.txt", b"tree\n"))]))
    job = wait_status(settings, rid)
    sub = settings.runs_dir / "submissions" / job["submission_id"]
    assert sub.is_dir()
    r = client.post(f"/api/runs/{rid}/delete", data={"csrf": settings.csrf_token}, headers={"accept": "text/html"},
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert not (settings.runs_dir / rid).exists() and not sub.exists()
    assert client.get(f"/runs/{rid}").status_code == 404


def test_delete_refuses_dirs_with_foreign_files(client, settings):
    d = cli_run_dir(settings.runs_dir, "cli-run", "sample_report_ready.json")
    (d / "my_notes.txt").write_text("important")
    r = delete(client, settings, "cli-run")
    assert r.status_code == 409 and "my_notes.txt" in r.json()["message"]
    assert (d / "my_notes.txt").exists()


def test_delete_refuses_a_cli_run_still_in_progress(client, settings):
    d = settings.runs_dir / "cli-live"
    d.mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "malvalid", "run"])
    try:
        write_json_atomic(d / "progress.json", {"stage": "modules", "pid": proc.pid, "modules": []})
        r = delete(client, settings, "cli-live")
        assert r.status_code == 409 and "command line" in r.json()["message"]
    finally:
        proc.kill()
        proc.wait()
    assert delete(client, settings, "cli-live").status_code == 200


# --------------------------------------------------------------------------------------------------
# Start-up reconciliation and shutdown
# --------------------------------------------------------------------------------------------------


def _job(rid: str, status: str, **kw: Any) -> dict[str, Any]:
    rec = {"schema": "malvalid-job/1", "run_id": rid, "created_at": "2026-09-30T10:00:00Z",
           "started_at": "2026-09-30T10:00:01Z" if status == "running" else None, "finished_at": None,
           "status": status, "argv": ["malvalid", "run"], "mode": "path", "adapter": "/x/a.py",
           "display_name": "x", "submission_id": None, "options": {}, "pid": None, "exit_code": None, "error": None}
    rec.update(kw)
    return rec


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_startup_reconciliation(tmp_path, fake_malvalid):
    runs = tmp_path / "runs"
    specs = {
        "20260930T100000Z-00000001": _job("20260930T100000Z-00000001", "running", pid=_dead_pid()),
        "20260930T100000Z-00000002": _job("20260930T100000Z-00000002", "queued"),
        "20260930T100000Z-00000003": _job("20260930T100000Z-00000003", "running", pid=None),
        "20260930T100000Z-00000004": _job("20260930T100000Z-00000004", "running", pid=_dead_pid()),
        "20260930T100000Z-00000005": _job("20260930T100000Z-00000005", "finished", exit_code=0,
                                          finished_at="2026-09-30T10:01:00Z"),
        "20260930T100000Z-00000006": _job("20260930T100000Z-00000006", "running", pid=1),  # pid reused by init
    }
    for rid, rec in specs.items():
        (runs / rid).mkdir(parents=True)
        write_json_atomic(runs / rid / "job.json", rec)
    # #4 completed its report while no server was watching
    rep = json.loads((REPO / "tests" / "fixtures" / "sample_report_ready.json").read_text())
    rep["run"]["started_at"] = "2026-09-30T10:00:02Z"
    (runs / "20260930T100000Z-00000004" / "report.json").write_text(json.dumps(rep))

    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=runs, token=TOKEN, port=8765, command_prefix=(sys.executable, str(fake_malvalid)))
    with authed_client(make_app(s)) as c:
        got = {r["run_id"]: r["status"] for r in c.get("/api/runs").json()}
    assert got == {
        "20260930T100000Z-00000001": "interrupted",
        "20260930T100000Z-00000002": "interrupted",
        "20260930T100000Z-00000003": "interrupted",
        "20260930T100000Z-00000004": "finished",
        "20260930T100000Z-00000005": "finished",
        "20260930T100000Z-00000006": "interrupted",
    }
    j1 = job_json(s, "20260930T100000Z-00000001")
    assert j1["finished_at"] and "process is gone" in j1["error"]
    assert "before this run started" in job_json(s, "20260930T100000Z-00000002")["error"]
    j4 = job_json(s, "20260930T100000Z-00000004")
    assert j4["exit_code"] == 0 and "completed while the web server was not running" in j4["note"]
    assert j4["finished_at"] == rep["run"]["finished_at"]  # when it really ended, not the restart time
    assert job_json(s, "20260930T100000Z-00000005")["finished_at"] == "2026-09-30T10:01:00Z"  # untouched


@pytest.mark.posix
def test_a_run_still_alive_at_startup_is_adopted(tmp_path, fake_malvalid):
    runs = tmp_path / "runs"
    rid = "20260930T100000Z-0000000a"
    d = runs / rid
    d.mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, "-c", "import sys, time; time.sleep(60)", "malvalid", "run", "--out", str(d)],
                            start_new_session=True)
    try:
        write_json_atomic(d / "job.json", _job(rid, "running", pid=proc.pid))
        s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=runs, token=TOKEN, port=8765, cancel_grace_s=1.0,
                        command_prefix=(sys.executable, str(fake_malvalid)))
        with authed_client(make_app(s)) as c:
            assert c.get(f"/api/runs/{rid}").json()["summary"]["status"] == "running"
            assert delete(c, s, rid).status_code == 409
            assert cancel(c, s, rid).status_code == 200  # SIGTERM to the adopted process group
            job = wait_status(s, rid, "cancelled", timeout=15)
            assert job["error"] == "cancelled by the user"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


@pytest.mark.posix
def test_an_adopted_run_occupies_a_worker_slot(tmp_path, fake_malvalid):
    """Regression (correctness-adopted-run-ignores-max-concurrent): with max_concurrent=1, a run adopted
    from an earlier server blocks the queue; a graceful shutdown marks the adopted run interrupted."""
    runs = tmp_path / "runs"
    rid = "20260930T100000Z-0000000b"
    d = runs / rid
    d.mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, "-c", "import sys, time; time.sleep(60)", "malvalid", "run", "--out", str(d)],
                            start_new_session=True)
    try:
        write_json_atomic(d / "job.json", _job(rid, "running", pid=proc.pid))
        s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=runs, token=TOKEN, port=8765, cancel_grace_s=1.0, max_concurrent=1,
                        command_prefix=(sys.executable, str(fake_malvalid)))
        with authed_client(make_app(s)) as c:
            new = submit(c, s, tmp_path, "ok")
            time.sleep(1.5)
            assert status(s, new) == "queued"  # waits for the adopted run's slot
            proc.kill()
            proc.wait()
            assert wait_status(s, new, "finished", timeout=30)["status"] == "finished"
            assert status(s, rid) == "failed"  # it died without a report

        # second case: the server shuts down while an adopted run is alive
        rid2 = "20260930T100000Z-0000000c"
        d2 = runs / rid2
        d2.mkdir()
        proc2 = subprocess.Popen([sys.executable, "-c", "import sys, time; time.sleep(60)", "malvalid", "run",
                                  "--out", str(d2)], start_new_session=True)
        try:
            write_json_atomic(d2 / "job.json", _job(rid2, "running", pid=proc2.pid))
            with authed_client(make_app(s)) as c:
                assert c.get(f"/api/runs/{rid2}").json()["summary"]["status"] == "running"
            assert job_json(s, rid2)["status"] == "interrupted"
        finally:
            if proc2.poll() is None:
                proc2.kill()
            proc2.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


@pytest.mark.posix
def test_shutdown_interrupts_active_runs(tmp_path, fake_malvalid):
    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=tmp_path / "runs", token=TOKEN, port=8765, cancel_grace_s=1.0,
                    command_prefix=(sys.executable, str(fake_malvalid)))
    with authed_client(make_app(s)) as c:
        running = submit(c, s, tmp_path, "sleep")
        queued = submit(c, s, tmp_path, "ok")
        pid = wait_running(s, running)["pid"]
        wait_for(lambda: pid_alive(pid), what="the run process")
    # leaving the client ends the app's lifespan: the job manager shuts down
    assert status(s, running) == "interrupted" and status(s, queued) == "interrupted"
    assert "stopped" in job_json(s, running)["error"]
    wait_for(lambda: group_gone(pid), what="the run's process group to be gone")


def test_job_manager_without_the_app(tmp_path, fake_malvalid):
    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=tmp_path / "runs", token=TOKEN, port=8765,
                    command_prefix=(sys.executable, str(fake_malvalid)))
    store = RunStore(s.runs_dir)
    jobs = JobManager(s, store)
    seen: list[str] = []
    jobs.on_change = seen.append
    ad = fake_adapter(tmp_path)
    spec = JobSpec(argv=s.malvalid_command() + ["run", "--adapter", str(ad), "--out", "{run_dir}"], mode="path",
                   adapter=str(ad), display_name="x", submission_id=None, options={})
    rid = jobs.submit(spec)  # starts the workers on first use
    try:
        wait_for(lambda: (jobs.job(rid) or {}).get("status") == "finished", what="the run to finish")
        assert jobs.job(rid)["argv"][-1] == str(store.root / rid)
        assert rid in seen and not jobs.is_active(rid) and jobs.active_ids() == []
        with pytest.raises(JobError) as e:
            jobs.cancel(rid)
        assert e.value.status == 409
    finally:
        jobs.shutdown()


# --------------------------------------------------------------------------------------------------
# A real `malvalid run --no-sandbox` on the toy adapter
# --------------------------------------------------------------------------------------------------


TOY_POLICY = {
    "corpus": "toy_v1_corpus",
    "modules": {"file_safety": {"enabled": True}, "performance": {"enabled": True},
                **{m: {"enabled": False} for m in ("drift", "membership_inf", "backdoor_screen", "extraction",
                                                   "explanation")}},
    "runtime": {"threads": 2},
    "verdict": {"required_for_ready": []},
}


def test_real_toy_run_through_the_web_ui(tmp_path, toy_python_path, app):
    """Upload the toy adapter + LightGBM model + policy; the subprocess is the real `malvalid run`
    (the toy plugins reach it through a test-only sitecustomize on PYTHONPATH)."""
    adapter = make_adapter_dir(tmp_path, "basedetector_adapter")
    runs = tmp_path / "runs"
    env = {"PYTHONPATH": str(toy_python_path), "OMP_NUM_THREADS": "2",
           "MALVALID_CORPUS_DIR": str(tmp_path / "no_corpora")}
    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=runs, token=TOKEN, port=8765, extra_env=env, cancel_grace_s=5.0)
    assert s.malvalid_command() == [sys.executable, "-m", "malvalid"]
    web = make_app(s)
    with authed_client(web) as c:
        files = [("adapter_file", (adapter.name, adapter.read_bytes())),
                 ("model_files", ("model.txt", (adapter.parent / "model.txt").read_bytes())),
                 ("manifest_file", ("train_hashes.txt", (adapter.parent / "train_hashes.txt").read_bytes())),
                 ("config_file", ("gate.yaml", yaml.safe_dump(TOY_POLICY).encode()))]
        r = post_run(c, s, {"mode": "upload", "no_sandbox": "on", "confirm_no_sandbox": "on", "seed": "3",
                            "title": "Toy detector"}, files)
        rid = run_id_from(r)
        # while it runs, the run page shows the job and (soon) the progress file
        assert c.get(f"/runs/{rid}").status_code == 200
        job = wait_status(s, rid, timeout=300)
        run_dir = runs / rid
        console = (run_dir / "console.log").read_text()
        assert job["status"] == "finished", (job, console[-3000:])
        assert job["exit_code"] in (0, 1)

        report = json.loads((run_dir / "report.json").read_text())
        progress = json.loads((run_dir / "progress.json").read_text())
        assert report["title"] == "Toy detector" and report["run"]["seed"] == 3
        assert progress["schema"] == "malvalid-progress/1" and progress["stage"] == "done"
        assert progress["run_id"] == report["run"]["id"] and progress["exit_code"] == job["exit_code"]
        # the report names the run by its web id (malvalid serve passes --run-id)
        assert report["run"]["id"] == rid and ("--run-id", rid) in list(zip(job["argv"], job["argv"][1:]))
        assert "--run-id" not in report["run"]["command"]
        assert [m["id"] for m in progress["modules"]] == ["file_safety", "performance"]
        assert [m["status"] for m in progress["modules"]] == [m["status"] for m in report["modules"]]
        assert all(isinstance(m["duration_s"], (int, float)) for m in progress["modules"])
        assert progress["verdict"] == {k: report["verdict"][k] for k in ("verdict", "label", "score", "coverage")}
        assert progress["module"] is None

        api = c.get(f"/api/runs/{rid}").json()
        summ = api["summary"]
        assert summ["verdict"] == report["verdict"]["verdict"] and summ["score"] == report["verdict"]["score"]
        assert summ["feature_version"] == "toy_v1" and summ["corpus"] == "toy_v1_corpus"
        assert summ["class_name"] == "MinimalDetector" and summ["title"] == "Toy detector"
        assert summ["n_modules"] == 2 and summ["source"] == "web"

        assert c.get(f"/runs/{rid}").status_code == 200
        ctx = captured(web, "run_detail.html.j2")
        assert ctx["files"]["report_html"] and ctx["files"]["run_log"] and ctx["files"]["console_log"]
        html = c.get(f"/runs/{rid}/report.html")
        assert html.status_code == 200 and "<html" in html.text.lower()
        assert (run_dir / "private").is_dir()
        assert c.get(f"/runs/{rid}/private").status_code == 404

        # compare it with a CLI run in the same runs dir
        cli_run_dir(runs, "cli-ready", "sample_report_ready.json")
        assert c.get(f"/compare?ids={rid},cli-ready").status_code == 200
        rows = captured(web, "compare.html.j2")["rows"]
        perf = next(r for r in rows if r["module_id"] == "performance")
        assert perf["cells"][0]["key_metric"] == "fpr" and perf["cells"][0]["key_value"] is not None

        # the uploads stayed in the submission dir; deleting removes both
        sub = runs / "submissions" / job["submission_id"]
        assert {"basedetector_adapter.py", "model.txt", "train_hashes.txt", "gate.yaml"} <= {p.name for p in sub.iterdir()}
        assert delete(c, s, rid).status_code == 200
        assert not run_dir.exists() and not sub.exists()


def test_scripted_submission_with_bearer_and_csrf_header(app, settings, tmp_path):
    """What docs/web.md shows for scripts: bearer auth + X-CSRF-Token, no cookie."""
    import hashlib
    import hmac

    from starlette.testclient import TestClient

    csrf = hmac.new(TOKEN.encode(), b"csrf", hashlib.sha256).hexdigest()
    ad = fake_adapter(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        auth = {"authorization": f"Bearer {TOKEN}"}
        r = c.post("/runs", headers={**auth, "x-csrf-token": csrf}, data={"mode": "path", "adapter_path": str(ad)},
                   files=[("model_files", ("", b""))], follow_redirects=False)
        rid = run_id_from(r)
        assert wait_status(settings, rid)["status"] == "finished"
        assert c.get(f"/api/runs/{rid}", headers=auth).json()["summary"]["verdict"] == "ready"
        assert c.post(f"/api/runs/{rid}/delete", headers={**auth, "x-csrf-token": csrf}).json()["deleted"] is True



def test_runs_and_validation_use_the_launch_directory(tmp_path, fake_malvalid, fake_record):
    """Regression (correctness-cwd-run-dir-relative-config-paths): a relative `corpus_dir: corp` in a policy
    resolves against the directory malvalid serve was started in, as it does for `malvalid run`, and an
    adapter writing to its cwd does not litter the run dir (which would block Delete)."""
    launch = tmp_path / "launch"
    launch.mkdir()
    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=tmp_path / "runs", token=TOKEN, port=8765, launch_dir=launch,
                    command_prefix=(sys.executable, str(fake_malvalid)), cancel_grace_s=1.0)
    app = make_app(s)
    with authed_client(app) as c:
        rid = submit(c, s, tmp_path, "ok")
        assert wait_status(s, rid)["status"] == "finished"
        ad = tmp_path / "v_adapter.py"
        ad.write_text("class A: pass\n")
        r = c.post("/api/validate", data={"csrf": s.csrf_token, "mode": "path", "adapter_path": str(ad)},
                   headers=JSON)
        assert r.status_code == 200, r.text
    calls = read_record(fake_record)
    assert [c["cwd"] for c in calls if c["cmd"] in ("run", "validate-adapter")] == [str(launch)] * 2
    assert RunStore(s.runs_dir).unexpected_entries(rid) == []
