"""CLI tests (typer.testing.CliRunner) for every ``malvalid`` command.

``malvalid run`` goes through the real runner with the sandbox seams monkeypatched to in-process
fakes (see ``test_runner_support``); the sandbox probe and adapter validator (other agents'
components) are replaced by fake modules in ``sys.modules`` so these tests pin the CLI contract.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from malvalid import __version__, registry
from malvalid.cli import app, render_summary
from malvalid.config import load_config
from malvalid.core import AdapterError
from malvalid.corpora.base import CorpusProvider, write_corpus
from tests.unit.test_runner_support import (  # noqa: F401 - fake_plugins is a fixture
    FakeM0Abort,
    FakeSandbox,
    fake_plugins,
    install_fake_sandbox,
    make_decl,
    toy_corpus,
    train_hashes,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"

cli = CliRunner()


def invoke(*args: str, **kw: Any):
    """Invoke the CLI on a wide virtual terminal (rich reads COLUMNS) so assertions see whole lines."""
    kw.setdefault("env", {"COLUMNS": "220"})
    return cli.invoke(app, [str(a) for a in args], catch_exceptions=False, **kw)


def one_line_error(result) -> str:
    """The error message (stderr) must be exactly one line starting with 'error:'."""
    lines = [ln for ln in result.stderr.strip().splitlines() if ln.strip()]
    assert lines, result.output
    assert lines[-1].startswith("error: "), result.stderr
    assert "Traceback" not in result.stderr
    return lines[-1]


# --------------------------------------------------------------------------------------------------
# Root / help / version
# --------------------------------------------------------------------------------------------------


class TestRoot:
    def test_version(self):
        r = invoke("--version")
        assert r.exit_code == 0 and r.stdout.strip() == f"malvalid {__version__}"

    def test_help_uses_researcher_framing(self):
        r = invoke("--help")
        assert r.exit_code == 0
        out = " ".join(r.stdout.split())
        assert "production-readiness verdict" in out and "0-100 score" in out
        for cmd in ("run", "list-modules", "init-config", "validate-adapter", "corpus", "report", "sandbox-check"):
            assert cmd in out

    def test_no_args_shows_help(self):
        r = cli.invoke(app, [])
        assert "Usage" in r.output

    def test_run_help(self):
        r = invoke("run", "--help")
        # Typer forces Rich terminal styling when GITHUB_ACTIONS is set, which splits "--adapter" with ANSI codes.
        out = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", r.stdout).split())
        assert r.exit_code == 0
        assert "production-readiness verdict" in out
        for opt in ("--adapter", "--config", "--out", "--allow-pickle", "--model", "--class", "--only", "--skip",
                    "--corpus-dir", "--seed", "--no-sandbox", "--no-html", "--fail-on", "-v"):
            assert opt in out, opt

    @pytest.mark.parametrize("sub", [["corpus", "--help"], ["report", "--help"], ["validate-adapter", "--help"],
                                     ["sandbox-check", "--help"], ["init-config", "--help"], ["list-modules", "--help"],
                                     ["corpus", "build", "--help"], ["report", "render", "--help"]])
    def test_subcommand_help(self, sub):
        assert invoke(*sub).exit_code == 0

    def test_python_dash_m(self):
        r = subprocess.run([sys.executable, "-m", "malvalid", "--version"], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and __version__ in r.stdout

    def test_unknown_command_is_usage_error(self):
        r = cli.invoke(app, ["frobnicate"])
        assert r.exit_code == 2


# --------------------------------------------------------------------------------------------------
# list-modules / init-config
# --------------------------------------------------------------------------------------------------


class TestListAndInit:
    def test_list_modules_json(self):
        r = invoke("list-modules", "--json")
        assert r.exit_code == 0
        data = json.loads(r.stdout)
        ids = [m["id"] for m in data["modules"]]
        assert ids[0] == "file_safety" and "performance" in ids
        perf = next(m for m in data["modules"] if m["id"] == "performance")
        assert perf["code"] == "M1" and perf["enabled_by_default"] is True and perf["weight"] == 3.0
        assert "unavailable" in data

    def test_list_modules_table(self):
        r = invoke("list-modules")
        assert r.exit_code == 0 and "M0" in r.stdout and "M1" in r.stdout

    def test_init_config(self, tmp_path):
        p = tmp_path / "gate.yaml"
        r = invoke("init-config", p)
        assert r.exit_code == 0 and p.exists()
        cfg = load_config(p)
        assert cfg.modules["performance"].gate.value == "hard"
        r = invoke("init-config", p)
        assert r.exit_code == 2 and "already exists" in one_line_error(r) and "--force" in r.stderr
        p.write_text("# edited\n")
        assert invoke("init-config", p, "--force").exit_code == 0
        assert "performance" in p.read_text()

    def test_init_config_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert invoke("init-config").exit_code == 0
        assert (tmp_path / "gate.yaml").exists()


# --------------------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------------------


def _gate_yaml(tmp_path: Path, enabled: dict[str, Any], **top: Any) -> Path:
    """A gate policy that disables the built-in modules and enables the given fakes."""
    mods: dict[str, Any] = {mid: {"enabled": False} for mid in
                            ("performance", "drift", "membership_inf", "backdoor_screen", "extraction", "explanation")}
    for mid, extra in enabled.items():
        mods[mid] = {"enabled": True, **(extra or {})}
    doc = {"corpus": "toy_v1_corpus", "modules": mods, "verdict": {"required_for_ready": []},
           "runtime": {"sandbox": False}, **top}
    p = tmp_path / "gate.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


@pytest.fixture
def sandbox(monkeypatch, tmp_path, fake_plugins) -> FakeSandbox:
    return install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, hashes=train_hashes())))


def _run_args(tmp_path: Path, cfg: Path, *extra: str) -> list[str]:
    return ["run", "--adapter", str(tmp_path / "adapter" / "my_adapter.py"), "--config", str(cfg),
            "--out", str(tmp_path / "run"), "--no-html", *extra]


class TestRun:
    def test_ready_exit_0_and_summary(self, tmp_path, sandbox):
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None})))
        assert r.exit_code == 0, r.output
        out = r.stdout
        assert "READY" in out and "Ready (meets the gate policy)" in out
        assert "Production-readiness score" in out and "/100" in out
        assert "Coverage" in out and "Per-axis scorecard" in out
        assert "T1" in out and "M0" in out
        assert "report.json" in out and "run.log" in out and "Exit code 0" in out
        assert "Not run (disabled in the config):" in out and "performance" in out
        rep = json.loads((tmp_path / "run" / "report.json").read_text())
        assert rep["verdict"]["verdict"] == "ready"
        assert "performance" in rep["gate"]["disabled"]
        assert rep["run"]["command"].startswith("malvalid run --adapter ")
        assert "--no-html" in rep["run"]["command"]

    def test_hard_fail_exit_1(self, tmp_path, sandbox):
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None, "t_hard": None})))
        assert r.exit_code == 1
        assert "BLOCKED" in r.stdout and "Blockers" in r.stdout and "hard gate failed" in r.stdout
        assert "/100" in r.stdout and "Exit code 1" in r.stdout

    def test_error_exit_2(self, tmp_path, sandbox):
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None, "t_crash": None})))
        assert r.exit_code == 2
        assert "ERROR" in r.stdout and "module crashed: ValueError: boom" in r.stdout
        assert "Exit code 2" in r.stdout

    def test_fail_on(self, tmp_path, sandbox):
        cfg = _gate_yaml(tmp_path, {"t_pass": None, "t_warn": None})
        r = invoke(*_run_args(tmp_path, cfg))
        assert r.exit_code == 0 and "CONDITIONAL" in r.stdout and "Why not READY" in r.stdout
        r = invoke(*_run_args(tmp_path, cfg, "--fail-on", "conditional"))
        assert r.exit_code == 1
        rep = json.loads((tmp_path / "run" / "report.json").read_text())
        assert rep["gate"]["fail_on"] == "conditional" and "--fail-on conditional" in rep["run"]["command"]

    def test_bad_fail_on_is_one_line_exit_2(self, tmp_path, sandbox):
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None}), "--fail-on", "ready"))
        assert r.exit_code == 2 and "--fail-on must be one of" in one_line_error(r)

    def test_m0_abort(self, tmp_path, sandbox):
        sandbox.m0 = FakeM0Abort
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None})))
        assert r.exit_code == 1 and "BLOCKED" in r.stdout
        assert "run aborted" in " ".join(r.stdout.split())

    def test_options_reach_the_run(self, tmp_path, sandbox):
        corpus_root = tmp_path / "corpora"
        corpus_root.mkdir()
        other = tmp_path / "m.json"
        other.write_text("{}")
        cfg = _gate_yaml(tmp_path, {"t_pass": None, "t_warn": None, "t_hard": None})
        r = invoke(*_run_args(tmp_path, cfg, "--seed", "11", "--only", "T1", "--only", "t_hard", "--skip", "T3",
                              "--corpus-dir", str(corpus_root), "--model", str(other), "--allow-pickle",
                              "--class", "ToyDetector", "--no-sandbox"))
        # t_warn (not in --only) and t_hard (--skip wins over --only) are enabled in the config: they are
        # reported as deselected skips, and the deselected hard gate t_hard blocks the verdict.
        assert r.exit_code == 1, r.output
        rep = json.loads((tmp_path / "run" / "report.json").read_text())
        assert rep["run"]["seed"] == 11 and rep["config"]["runtime"]["seed"] == 11
        assert [m["module_id"] for m in rep["modules"]] == ["file_safety", "t_pass", "t_warn", "t_hard"]
        assert [m["status"] for m in rep["modules"]] == ["pass", "pass", "skipped", "skipped"]
        assert rep["gate"]["deselected"] == ["t_warn", "t_hard"]
        assert "Partial run — not run (deselected with --only/--skip): t_warn, t_hard" in r.stdout
        assert rep["config"]["corpus_dir"] == str(corpus_root.resolve())
        assert rep["config"]["runtime"]["sandbox"] is False and rep["config"]["runtime"]["allow_pickle"] is True
        assert rep["model"]["artifacts"][0]["path"] == str(other.resolve())
        assert sandbox.policies[0]["allow_pickle"] is True

    def test_verify_corpus_flag(self, tmp_path, sandbox):
        """Regression (xcomp-run-corpus-verify-trusts-stat-cache): an opt-in full re-hash for CI."""
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None}), "--verify-corpus"))
        assert r.exit_code == 0, r.output
        rep = json.loads((tmp_path / "run" / "report.json").read_text())
        assert rep["config"]["runtime"]["corpus_verification"] == "full"
        assert "--verify-corpus" in rep["run"]["command"]
        assert "verification" in rep["corpus"]  # None for the in-memory toy corpus
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None})))
        rep = json.loads((tmp_path / "run" / "report.json").read_text())
        assert rep["config"]["runtime"]["corpus_verification"] == "cached"

    def test_corpus_option(self, tmp_path, sandbox):
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None}), "--corpus", "t_missing_corpus"))
        rep = json.loads((tmp_path / "run" / "report.json").read_text())
        assert rep["corpus"]["name"] == "t_missing_corpus"
        assert "SKIPPED" in r.stdout and "not found at" in " ".join(r.stdout.split())

    def test_html_written_by_default(self, tmp_path, sandbox, monkeypatch):
        from malvalid import runner

        monkeypatch.setattr(runner, "_write_html", lambda rep, p: (Path(p).write_text("<html></html>"), Path(p))[1])
        args = [a for a in _run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None})) if a != "--no-html"]
        r = invoke(*args)
        assert r.exit_code == 0 and (tmp_path / "run" / "report.html").exists()
        assert "report.html" in r.stdout and "(not written)" not in r.stdout

    def test_missing_config_file(self, tmp_path, sandbox):
        r = invoke(*_run_args(tmp_path, tmp_path / "nope.yaml"))
        assert r.exit_code == 2 and "config file not found" in one_line_error(r)

    def test_invalid_config(self, tmp_path, sandbox):
        p = tmp_path / "bad.yaml"
        p.write_text("runtime:\n  seeed: 1\n")
        r = invoke(*_run_args(tmp_path, p))
        assert r.exit_code == 2 and "invalid gate config" in one_line_error(r)

    def test_unknown_module_param(self, tmp_path, sandbox):
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": {"min_rat": 1}})))
        assert r.exit_code == 2 and "unknown parameter" in one_line_error(r)

    def test_adapter_error_one_line_and_verbose_traceback(self, tmp_path, sandbox):
        sandbox.inspect_error = AdapterError("MyDetector does not define feature_version;\nadd it")
        cfg = _gate_yaml(tmp_path, {"t_pass": None})
        r = invoke(*_run_args(tmp_path, cfg))
        assert r.exit_code == 2
        assert one_line_error(r) == "error: MyDetector does not define feature_version; add it"
        r = invoke(*_run_args(tmp_path, cfg, "-v"))
        assert r.exit_code == 2 and "Traceback" in r.stderr

    def test_missing_adapter(self, tmp_path, sandbox):
        cfg = _gate_yaml(tmp_path, {"t_pass": None})
        r = invoke("run", "--adapter", tmp_path / "nope.py", "--config", cfg, "--out", tmp_path / "r")
        assert r.exit_code == 2 and "adapter not found" in one_line_error(r)

    def test_unknown_only(self, tmp_path, sandbox):
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None}), "--only", "M42"))
        assert r.exit_code == 2 and "unknown module 'M42'" in one_line_error(r)

    def test_internal_error_hint(self, tmp_path, sandbox, monkeypatch):
        from malvalid import runner

        def boom(*a, **k):
            raise ZeroDivisionError("oops")

        monkeypatch.setattr(runner, "_open_model", boom)
        monkeypatch.setattr(runner, "compute_verdict", boom)
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None})))
        assert r.exit_code == 2
        assert "internal error: ZeroDivisionError: oops (re-run with -v" in one_line_error(r)

    def test_error_stays_one_line_on_narrow_terminal(self, tmp_path, sandbox):
        sandbox.inspect_error = AdapterError("x" * 40 + " a long explanation that would wrap at sixty columns " * 3)
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None})), env={"COLUMNS": "60"})
        assert r.exit_code == 2 and one_line_error(r).endswith("sixty columns")

    def test_summary_at_80_columns(self, tmp_path, sandbox):
        r = invoke(*_run_args(tmp_path, _gate_yaml(tmp_path, {"t_pass": None, "t_warn": None})), env={"COLUMNS": "80"})
        assert r.exit_code == 0
        assert max(len(ln) for ln in r.stdout.splitlines() if "report.json" not in ln and "run.log" not in ln) <= 80
        assert "Per-axis scorecard" in r.stdout and "CONDITIONAL" in r.stdout

    def test_missing_required_option(self):
        r = cli.invoke(app, ["run"])
        assert r.exit_code == 2 and "--adapter" in r.output


class TestRenderSummary:
    @pytest.mark.parametrize("name", ["sample_report.json", "sample_report_aborted.json",
                                      "sample_report_conditional.json", "sample_report_not_ready.json",
                                      "sample_report_ready.json"])
    def test_renders_fixture_reports(self, name, capsys):
        from rich.console import Console

        p = FIXTURES / name
        if not p.exists():
            pytest.skip(f"{name} fixture not present")
        rep = json.loads(p.read_text())
        con = Console(width=160, record=True)
        render_summary(rep, con)
        text = con.export_text()
        assert rep["verdict"]["verdict"].upper().replace("_", " ") in text
        assert "Per-axis scorecard" in text

    def test_capped_score_is_explained(self):
        from rich.console import Console

        con = Console(width=160, record=True)
        render_summary({"verdict": {"verdict": "blocked", "label": "Blocked — a hard gate failed", "score": 49.0,
                                    "raw_score": 83.2, "capped": True, "coverage": 0.9,
                                    "bands": {"blocked_score_cap": 49.0}, "blockers": ["M1 x: hard gate failed"],
                                    "reasons": ["M1 x: hard gate failed"]}}, con)
        text = con.export_text()
        assert "49.0/100" in text and "capped at 49 (uncapped 83.2)" in text
        assert text.count("M1 x: hard gate failed") == 1  # blockers are not repeated as reasons

    def test_robust_to_minimal_report(self):
        from rich.console import Console

        con = Console(width=100, record=True)
        render_summary({"verdict": {"verdict": "not_ready", "score": None}}, con)
        text = con.export_text()
        assert "NOT READY" in text and "n/a" in text


# --------------------------------------------------------------------------------------------------
# report render
# --------------------------------------------------------------------------------------------------


class TestReportRender:
    def _report(self, tmp_path: Path) -> Path:
        src = FIXTURES / "sample_report.json"
        if not src.exists():
            pytest.skip("sample_report.json fixture not present")
        dst = tmp_path / "report.json"
        dst.write_text(src.read_text())
        return dst

    def test_render_next_to_json(self, tmp_path):
        pytest.importorskip("malvalid.report.html")
        rj = self._report(tmp_path)
        r = invoke("report", "render", rj)
        assert r.exit_code == 0, r.output
        assert (tmp_path / "report.html").exists() and "Wrote" in r.stdout

    def test_render_custom_out(self, tmp_path, monkeypatch):
        calls = []
        fake = types.ModuleType("malvalid.report.html")
        fake.write_html = lambda rep, path: (calls.append((rep["schema_version"], Path(path))), Path(path))[1]
        monkeypatch.setitem(sys.modules, "malvalid.report.html", fake)
        rj = self._report(tmp_path)
        r = invoke("report", "render", rj, "-o", tmp_path / "x.html")
        assert r.exit_code == 0 and calls == [("malvalid-report/1", tmp_path / "x.html")]

    def test_renderer_missing(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "malvalid.report.html", None)
        r = invoke("report", "render", self._report(tmp_path))
        assert r.exit_code == 2 and "renderer is not available" in one_line_error(r)

    def test_renderer_crash(self, tmp_path, monkeypatch):
        fake = types.ModuleType("malvalid.report.html")

        def boom(rep, path):
            raise KeyError("verdict")

        fake.write_html = boom
        monkeypatch.setitem(sys.modules, "malvalid.report.html", fake)
        r = invoke("report", "render", self._report(tmp_path))
        assert r.exit_code == 2 and "could not render" in one_line_error(r)

    def test_not_a_report(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text('{"hello": 1}')
        r = invoke("report", "render", p)
        assert r.exit_code == 2 and "not a malvalid report" in one_line_error(r)
        p.write_text("{nope")
        r = invoke("report", "render", p)
        assert r.exit_code == 2 and "not valid JSON" in one_line_error(r)
        r = invoke("report", "render", tmp_path / "missing.json")
        assert r.exit_code == 2 and "report file not found" in one_line_error(r)


# --------------------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------------------


class DiskCorpusProvider(CorpusProvider):
    """Uses the base on-disk load; directory given with --corpus-dir."""

    name: ClassVar[str] = "t_disk_corpus"
    feature_version: ClassVar[str] = "toy_v1"
    version: ClassVar[str] = "v1"
    builds: ClassVar[list[tuple[Any, ...]]] = []

    def build(self, source, out, **kwargs):
        type(self).builds.append((source, out, kwargs))
        return {"name": self.name, "version": self.version, "n": 3, "dim": 32, "content_hash": "abc123"}


@pytest.fixture
def disk_provider():
    registry.register("corpora", DiskCorpusProvider.name, DiskCorpusProvider)
    DiskCorpusProvider.builds = []
    yield DiskCorpusProvider
    registry.unregister("corpora", DiskCorpusProvider.name)


def _write_disk_corpus(d: Path) -> Path:
    c = toy_corpus()
    idx = np.arange(50)
    write_corpus(d, c.X[idx], name="t_disk_corpus", version="v1", feature_version="toy_v1",
                 sha256=c.sha256[idx].tolist(), label=c.label[idx], timestamp=c.timestamp[idx],
                 split=c.split[idx].tolist(), roles={"eval": ["test"]})
    return d


class TestCorpus:
    def test_list(self, disk_provider):
        r = invoke("corpus", "list", "--json")
        assert r.exit_code == 0
        names = {c["name"] for c in json.loads(r.stdout)["corpora"]}
        assert {"toy_v1_corpus", "t_disk_corpus"} <= names
        r = invoke("corpus", "list")
        assert r.exit_code == 0 and "toy_v1_corpus" in r.stdout

    def test_info_in_memory(self):
        r = invoke("corpus", "info", "toy_v1_corpus", "--json")
        assert r.exit_code == 0
        info = json.loads(r.stdout)
        assert info["available"] is True and info["summary"]["content_hash"] == toy_corpus().content_hash
        r = invoke("corpus", "info", "toy_v1_corpus")
        assert r.exit_code == 0 and "Content hash" in r.stdout and "Splits" in r.stdout

    def test_info_missing(self, tmp_path, disk_provider):
        r = invoke("corpus", "info", "t_disk_corpus", "--corpus-dir", tmp_path / "none")
        assert r.exit_code == 0 and "Not present" in r.stdout and "corpus build" in r.stdout

    def test_info_unknown(self):
        r = invoke("corpus", "info", "nope")
        assert r.exit_code == 2 and "unknown corpus 'nope'" in one_line_error(r)

    def test_verify_ok_and_tampered(self, tmp_path, disk_provider):
        d = _write_disk_corpus(tmp_path / "c")
        r = invoke("corpus", "verify", "t_disk_corpus", "--corpus-dir", d)
        assert r.exit_code == 0 and "OK" in r.stdout and "2 file(s) re-hashed" in r.stdout
        # --corpus-dir may also be the root that contains <name>/
        root = tmp_path / "root"
        _write_disk_corpus(root / "t_disk_corpus")
        assert invoke("corpus", "verify", "t_disk_corpus", "--corpus-dir", root).exit_code == 0
        # Tamper with one byte of X.npy: verification must fail (exit 1), sidecar cache ignored.
        x = d / "X.npy"
        b = bytearray(x.read_bytes())
        b[-1] ^= 0xFF
        x.write_bytes(bytes(b))
        r = invoke("corpus", "verify", "t_disk_corpus", "--corpus-dir", d)
        assert r.exit_code == 1 and "verification FAILED" in r.stderr

    def test_verify_missing(self, tmp_path, disk_provider):
        r = invoke("corpus", "verify", "t_disk_corpus", "--corpus-dir", tmp_path / "none")
        assert r.exit_code == 2 and "corpus build" in one_line_error(r)

    def test_verify_in_memory(self):
        r = invoke("corpus", "verify", "toy_v1_corpus")
        assert r.exit_code == 0 and "generated in memory" in r.stdout

    def test_build_delegates_to_provider(self, tmp_path, disk_provider):
        src = tmp_path / "src"
        src.mkdir()
        r = invoke("corpus", "build", "t_disk_corpus", "--source", src, "--out", tmp_path / "out", "--workers", "3")
        assert r.exit_code == 0, r.output
        assert disk_provider.builds == [(src.resolve(), (tmp_path / "out").resolve(), {"workers": 3})]
        assert "Built" in r.stdout and "abc123" in r.stdout

    def test_build_errors(self, tmp_path, disk_provider):
        r = invoke("corpus", "build", "t_disk_corpus", "--source", tmp_path / "missing")
        assert r.exit_code == 2 and "directory not found" in one_line_error(r)
        r = invoke("corpus", "build", "toy_v1_corpus", "--source", tmp_path, "--out", tmp_path / "o")
        assert r.exit_code == 2 and "cannot be built locally" in one_line_error(r)


# --------------------------------------------------------------------------------------------------
# sandbox-check / validate-adapter (other agents' components, faked)
# --------------------------------------------------------------------------------------------------


def _fake_host(monkeypatch, backends: dict[str, dict]) -> None:
    import malvalid.sandbox as sb_pkg

    fake = types.ModuleType("malvalid.sandbox.host")
    fake.probe_backends = lambda: backends
    monkeypatch.setitem(sys.modules, "malvalid.sandbox.host", fake)
    monkeypatch.setattr(sb_pkg, "host", fake, raising=False)


class TestSandboxCheck:
    def test_isolated_backend(self, monkeypatch):
        _fake_host(monkeypatch, {"bwrap": {"available": True, "network_isolated": True, "detail": "ok"},
                                 "subprocess": {"available": True, "network_isolated": False, "detail": ""}})
        r = invoke("sandbox-check")
        assert r.exit_code == 0 and "bwrap" in r.stdout and "network isolated" in r.stdout
        r = invoke("sandbox-check", "--json")
        assert r.exit_code == 0 and json.loads(r.stdout)["bwrap"]["available"] is True

    def test_no_isolation(self, monkeypatch):
        _fake_host(monkeypatch, {"subprocess": {"available": True, "network_isolated": False, "detail": ""}})
        r = invoke("sandbox-check")
        assert r.exit_code == 1 and "NOT isolated" in " ".join(r.stdout.split())

    def test_sandbox_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "malvalid.sandbox.host", None)
        r = invoke("sandbox-check")
        assert r.exit_code == 2 and "sandbox is not available" in one_line_error(r)


class _Rep:
    def __init__(self, ok: bool):
        self.ok = ok

    def to_dict(self):
        return {"ok": self.ok, "checks": [{"name": "shape", "ok": self.ok, "detail": "(n,)"}]}

    def render_text(self):
        return f"adapter validation: {'OK' if self.ok else 'FAILED'}"


def _fake_validate(monkeypatch, ok: bool, calls: list) -> None:
    fake = types.ModuleType("malvalid.adapters.validate")

    def validate_adapter(adapter_path, cfg, *, allow_pickle=False, class_name=None, model_paths=()):
        calls.append((adapter_path, allow_pickle, class_name, list(model_paths), cfg.runtime.allow_pickle))
        return _Rep(ok)

    fake.validate_adapter = validate_adapter
    monkeypatch.setitem(sys.modules, "malvalid.adapters.validate", fake)


class TestValidateAdapter:
    def test_ok(self, tmp_path, monkeypatch):
        calls: list = []
        _fake_validate(monkeypatch, True, calls)
        a = tmp_path / "a.py"
        a.write_text("")
        m = tmp_path / "m.txt"
        m.write_text("")
        r = invoke("validate-adapter", "--adapter", a, "--allow-pickle", "--model", m, "--class", "Det")
        assert r.exit_code == 0 and "adapter validation: OK" in r.stdout
        assert calls == [(a.resolve(), True, "Det", [m.resolve()], True)]
        r = invoke("validate-adapter", "--adapter", a, "--json")
        assert r.exit_code == 0 and json.loads(r.stdout)["ok"] is True

    def test_failed_check_exit_1(self, tmp_path, monkeypatch):
        _fake_validate(monkeypatch, False, [])
        a = tmp_path / "a.py"
        a.write_text("")
        r = invoke("validate-adapter", "--adapter", a)
        assert r.exit_code == 1 and "FAILED" in r.stdout

    def test_missing_adapter(self, tmp_path, monkeypatch):
        _fake_validate(monkeypatch, True, [])
        r = invoke("validate-adapter", "--adapter", tmp_path / "nope.py")
        assert r.exit_code == 2 and "adapter not found" in one_line_error(r)

    def test_validator_missing(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "malvalid.adapters.validate", None)
        a = tmp_path / "a.py"
        a.write_text("")
        r = invoke("validate-adapter", "--adapter", a)
        assert r.exit_code == 2 and "not available" in one_line_error(r)
