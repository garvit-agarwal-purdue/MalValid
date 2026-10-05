"""``progress.json`` (schema ``malvalid-progress/1``): the runner's live stage file.

The runner rewrites it atomically at every stage (``starting → inspecting → scanning → loading →
modules → verdict → writing → done``, or ``failed``); the local web UI polls it to show which module
is running. Runs go through the monkeypatched in-process sandbox of ``test_runner_support``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from typer.testing import CliRunner

from malvalid import registry, runner
from malvalid.cli import app
from malvalid.core import AdapterError, GateCheck, ModuleResult, Requirement
from malvalid.runner import PROGRESS_SCHEMA, PROGRESS_STAGES, run_gate
from tests.unit.test_runner_support import (  # noqa: F401 - fake_plugins is a fixture
    FakeM0Abort,
    FakeSandbox,
    _Fake,
    fake_plugins,
    install_fake_sandbox,
    make_cfg,
    make_decl,
    run_opts,
    train_hashes,
)

pytestmark = pytest.mark.usefixtures("fake_plugins")


class ProbeMod(_Fake):
    """Reads progress.json from disk while it runs (what a UI polling the file would see)."""

    id = "t_probe"
    code = "TP"
    title = "Reads the progress file"
    requires = (Requirement.QUERY_ONLY,)
    snapshots: ClassVar[list[dict[str, Any]]] = []

    def run(self, ctx) -> ModuleResult:
        type(self).snapshots.append(json.loads((ctx.run_dir / "progress.json").read_text()))
        return self.result(ctx, finding="ok", checks=[GateCheck.evaluate("x", 1.0, ">=", 0.5, ideal=1.0, floor=0.0)])


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeSandbox:
    registry.register("modules", ProbeMod.id, ProbeMod)
    ProbeMod.snapshots = []
    yield install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, hashes=train_hashes())))
    registry.unregister("modules", ProbeMod.id)


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every state written to progress.json, in order."""
    states: list[dict[str, Any]] = []
    orig = runner._Progress._write

    def _write(self):  # type: ignore[no-untyped-def]
        orig(self)
        states.append(copy.deepcopy(self.state))

    monkeypatch.setattr(runner._Progress, "_write", _write)
    return states


def _progress(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "run" / "progress.json").read_text())


def _stage_sequence(states: list[dict[str, Any]]) -> list[str]:
    seq: list[str] = []
    for s in states:
        if not seq or seq[-1] != s["stage"]:
            seq.append(s["stage"])
    return seq


def test_full_run_writes_every_stage_in_order(tmp_path, sandbox, recorded):
    out = run_gate(run_opts(tmp_path, make_cfg(["t_pass", "t_probe", "t_warn"]),
                            progress_path=tmp_path / "run" / "progress.json"))
    seq = _stage_sequence(recorded)
    assert seq == ["starting", "inspecting", "scanning", "loading", "modules", "verdict", "writing", "done"]
    assert all(s in PROGRESS_STAGES for s in seq)

    final = _progress(tmp_path)
    assert final == recorded[-1]
    assert final["schema"] == PROGRESS_SCHEMA
    assert final["run_id"] == out.report["run"]["id"]
    assert final["stage"] == "done" and final["module"] is None
    assert final["exit_code"] == out.exit_code
    assert final["started_at"] == out.report["run"]["started_at"]
    assert final["updated_at"] >= final["started_at"]
    assert isinstance(final["pid"], int)
    v = out.report["verdict"]
    assert final["verdict"] == {k: v[k] for k in ("verdict", "label", "score", "coverage")}
    # Modules: M0 first, then scorecard order, with the report's final statuses and durations.
    assert [m["id"] for m in final["modules"]] == [m["module_id"] for m in out.report["modules"]]
    for pm, rm in zip(final["modules"], out.report["modules"]):
        assert (pm["code"], pm["title"], pm["status"]) == (rm["code"], rm["title"], rm["status"])
        assert pm["duration_s"] == rm["duration_s"]
    # No temp files left behind; strict JSON on disk.
    assert sorted(p.name for p in (tmp_path / "run").glob(".progress.json*")) == []
    json.loads((tmp_path / "run" / "progress.json").read_text(), parse_constant=lambda c: pytest.fail(c))


def test_modules_start_pending_and_the_running_module_is_visible(tmp_path, sandbox, recorded):
    run_gate(run_opts(tmp_path, make_cfg(["t_pass", "t_probe", "t_warn"]),
                      progress_path=tmp_path / "run" / "progress.json"))
    planned = next(s for s in recorded if s["modules"])
    assert [m["id"] for m in planned["modules"]] == ["file_safety", "t_pass", "t_warn", "t_probe"]
    assert {m["status"] for m in planned["modules"]} == {"pending"}
    assert all(m["duration_s"] is None for m in planned["modules"])

    snap = ProbeMod.snapshots[0]  # read from disk by the module itself while it ran
    assert snap["stage"] == "modules"
    assert snap["module"] == {"id": "t_probe", "code": "TP", "title": "Reads the progress file"}
    by_id = {m["id"]: m["status"] for m in snap["modules"]}
    assert by_id == {"file_safety": "pass", "t_pass": "pass", "t_warn": "warn", "t_probe": "running"}
    assert snap["verdict"] is None

    scanning = [s for s in recorded if s["stage"] == "scanning" and s["module"]]
    assert scanning and scanning[0]["module"]["id"] == "file_safety"


def test_m0_abort_marks_the_rest_skipped(tmp_path, sandbox, recorded):
    sandbox.m0 = FakeM0Abort
    out = run_gate(run_opts(tmp_path, make_cfg(["t_pass", "t_warn"]),
                            progress_path=tmp_path / "run" / "progress.json"))
    final = _progress(tmp_path)
    assert final["stage"] == "done" and final["verdict"]["verdict"] == "blocked"
    assert {m["id"]: m["status"] for m in final["modules"]} == {
        "file_safety": "fail", "t_pass": "skipped", "t_warn": "skipped"}
    assert "loading" not in _stage_sequence(recorded)  # the model is never loaded
    assert out.exit_code == 1


def test_load_failure_still_reaches_done(tmp_path, sandbox):
    sandbox.open_error = RuntimeError("cannot unpack model")
    out = run_gate(run_opts(tmp_path, make_cfg(["t_pass"]), progress_path=tmp_path / "run" / "progress.json"))
    final = _progress(tmp_path)
    assert final["stage"] == "done" and final["exit_code"] == out.exit_code == 2
    assert {m["id"]: m["status"] for m in final["modules"]}["t_pass"] == "skipped"


def test_adapter_error_is_recorded_as_failed(tmp_path, sandbox):
    sandbox.inspect_error = AdapterError("my_adapter.py: class Foo does not declare feature_version")
    with pytest.raises(AdapterError):
        run_gate(run_opts(tmp_path, make_cfg(["t_pass"]), progress_path=tmp_path / "run" / "progress.json"))
    final = _progress(tmp_path)
    assert final["stage"] == "failed"
    assert "does not declare feature_version" in final["message"]
    assert final["verdict"] is None and final["exit_code"] is None
    assert not (tmp_path / "run" / "report.json").exists()


def test_module_crash_mid_run_is_an_error_status_not_a_failed_run(tmp_path, sandbox):
    run_gate(run_opts(tmp_path, make_cfg(["t_pass", "t_crash"]), progress_path=tmp_path / "run" / "progress.json"))
    final = _progress(tmp_path)
    assert final["stage"] == "done"
    assert {m["id"]: m["status"] for m in final["modules"]}["t_crash"] == "error"


def test_unwritable_progress_path_never_breaks_a_run(tmp_path, sandbox):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    out = run_gate(run_opts(tmp_path, make_cfg(["t_pass"]), progress_path=blocker / "progress.json"))
    assert out.exit_code == 0 and out.report_json.exists()
    # A directory where the file should be is equally harmless.
    (tmp_path / "dir_progress").mkdir()
    out = run_gate(run_opts(tmp_path, make_cfg(["t_pass"]), progress_path=tmp_path / "dir_progress"))
    assert out.exit_code == 0


def test_no_progress_path_writes_nothing(tmp_path, sandbox):
    run_gate(run_opts(tmp_path, make_cfg(["t_pass"])))
    assert not (tmp_path / "run" / "progress.json").exists()


def test_relative_progress_path_resolves_against_cwd(tmp_path, sandbox, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_gate(run_opts(tmp_path, make_cfg(["t_pass"]), progress_path=Path("run") / "progress.json"))
    assert _progress(tmp_path)["stage"] == "done"


def test_cli_run_always_writes_progress(tmp_path, sandbox):
    doc = {"corpus": "toy_v1_corpus", "modules": {"file_safety": {"enabled": True}, "t_pass": {"enabled": True}},
           "verdict": {"required_for_ready": []}, "runtime": {"sandbox": False}}
    for mid in ("performance", "drift", "membership_inf", "backdoor_screen", "extraction", "explanation"):
        doc["modules"][mid] = {"enabled": False}
    cfg = tmp_path / "gate.yaml"
    cfg.write_text(yaml.safe_dump(doc))
    r = CliRunner().invoke(app, ["run", "--adapter", str(tmp_path / "adapter" / "my_adapter.py"), "--config", str(cfg),
                                 "--out", str(tmp_path / "run"), "--no-html"], env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    final = _progress(tmp_path)
    rep = json.loads((tmp_path / "run" / "report.json").read_text())
    assert final["stage"] == "done" and final["run_id"] == rep["run"]["id"]
    assert [m["id"] for m in final["modules"]] == ["file_safety", "t_pass"]


def test_run_id_option_is_recorded_everywhere(tmp_path, sandbox):
    """``RunOptions.run_id`` (what ``malvalid serve`` passes) names the run in report.json and progress.json."""
    out = run_gate(run_opts(tmp_path, make_cfg(["t_pass"]), progress_path=tmp_path / "run" / "progress.json",
                            run_id="20260930T120000Z-0123abcd"))
    assert out.report["run"]["id"] == "20260930T120000Z-0123abcd"
    assert _progress(tmp_path)["run_id"] == "20260930T120000Z-0123abcd"
    # default: a fresh timestamped id
    out2 = run_gate(dataclasses.replace(run_opts(tmp_path, make_cfg(["t_pass"])), out_dir=tmp_path / "run2"))
    assert out2.report["run"]["id"] != "20260930T120000Z-0123abcd"
    assert len(out2.report["run"]["id"].split("-")[-1]) == 8


def _cli_policy(tmp_path: Path) -> Path:
    doc = {"corpus": "toy_v1_corpus", "modules": {"file_safety": {"enabled": True}, "t_pass": {"enabled": True}},
           "verdict": {"required_for_ready": []}, "runtime": {"sandbox": False}}
    for mid in ("performance", "drift", "membership_inf", "backdoor_screen", "extraction", "explanation"):
        doc["modules"][mid] = {"enabled": False}
    cfg = tmp_path / "gate.yaml"
    cfg.write_text(yaml.safe_dump(doc))
    return cfg


def test_cli_run_id_flag(tmp_path, sandbox):
    cfg = _cli_policy(tmp_path)
    base = ["run", "--adapter", str(tmp_path / "adapter" / "my_adapter.py"), "--config", str(cfg),
            "--out", str(tmp_path / "run"), "--no-html"]
    r = CliRunner().invoke(app, base + ["--run-id", "web-run.1"], env={"COLUMNS": "200"})
    assert r.exit_code == 0, r.output
    rep = json.loads((tmp_path / "run" / "report.json").read_text())
    assert rep["run"]["id"] == "web-run.1" and _progress(tmp_path)["run_id"] == "web-run.1"
    assert "--run-id" not in rep["run"]["command"]  # a re-run from the terminal gets its own id
    assert "--run-id" not in CliRunner().invoke(app, ["run", "--help"], env={"COLUMNS": "200"}).output  # hidden
    for bad in ("../x", "-x", "a b", "", "x" * 129):
        r = CliRunner().invoke(app, base + ["--run-id", bad], env={"COLUMNS": "200"})
        assert r.exit_code == 2 and "--run-id" in r.output, (bad, r.output)


def test_real_toy_run_writes_every_stage(tmp_path, recorded):
    """No fake sandbox: the real loader, M0 scan and M1 on the toy LightGBM adapter, in-process
    (``runtime.sandbox: false``) — what ``malvalid run --no-sandbox`` writes."""
    from tests.fixtures.adapters.builders import make_adapter_dir

    adapter = make_adapter_dir(tmp_path, "basedetector_adapter")
    progress_path = tmp_path / "run" / "progress.json"
    out = run_gate(runner.RunOptions(adapter=adapter, config=make_cfg(["performance"], threads=2),
                                     out_dir=tmp_path / "run", write_html=False, progress_path=progress_path))
    assert _stage_sequence(recorded) == ["starting", "inspecting", "scanning", "loading", "modules", "verdict",
                                         "writing", "done"]
    final = json.loads(progress_path.read_text())
    assert [m["id"] for m in final["modules"]] == ["file_safety", "performance"]
    assert [m["status"] for m in final["modules"]] == [m["status"] for m in out.report["modules"]]
    assert final["verdict"]["verdict"] == out.report["verdict"]["verdict"] and final["exit_code"] == out.exit_code
    # a poller sees M0 while scanning and M1 while it runs, each marked "running"
    running = {s["module"]["id"] for s in recorded if s.get("module")}
    assert running == {"file_safety", "performance"} or running == {"performance"}
    during = next(s for s in recorded if (s.get("module") or {}).get("id") == "performance")
    assert during["stage"] == "modules"
    assert {m["id"]: m["status"] for m in during["modules"]}["performance"] == "running"
    assert {m["id"]: m["status"] for m in during["modules"]}["file_safety"] == out.report["modules"][0]["status"]
