"""Runner orchestration tests (malvalid.runner.run_gate) with in-process sandbox fakes.

Every run here uses the toy_v1 schema / ``toy_v1_corpus`` and a small LightGBM model through the
monkeypatched sandbox seams (see ``test_runner_support``); fake modules exercise each path of the
pipeline: pass / warn / hard fail / skips / crash / timeouts / M0 abort / corpus problems / tree
fidelity / exit codes / strict JSON / private-data scrubbing.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from pathlib import Path

import numpy as np
import pytest

from malvalid import REPORT_SCHEMA_VERSION, registry, runner
from malvalid.config import validate_against_registry
from malvalid.context import module_seed
from malvalid.core import AdapterError, ConfigError, Requirement, SandboxError
from malvalid.runner import RunOptions, resolve_module_ids, run_gate, validate_fail_on
from tests.unit.test_runner_support import (  # noqa: F401 - fake_plugins is a fixture
    CrashMod,
    FakeM0Abort,
    FakeM0Crash,
    FakeM0Pass,
    FakeSandbox,
    HardFailMod,
    NeedsAnyMod,
    NeedsHashesMod,
    NeedsTreesMod,
    PassMod,
    WarnMod,
    fake_plugins,
    install_fake_sandbox,
    make_cfg,
    make_decl,
    module,
    run_opts,
    toy_corpus,
    train_hashes,
)

pytestmark = pytest.mark.usefixtures("fake_plugins")


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeSandbox:
    fs = FakeSandbox(decl=make_decl(tmp_path, hashes=train_hashes()))
    return install_fake_sandbox(monkeypatch, fs)


def _run(tmp_path: Path, cfg, **kw):
    return run_gate(run_opts(tmp_path, cfg, **kw))


def _strict(report: dict) -> str:
    return json.dumps(report, allow_nan=False)


# --------------------------------------------------------------------------------------------------
# Happy path + report structure
# --------------------------------------------------------------------------------------------------


class TestPassingRun:
    def test_pass_run_ready_exit_0(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass", "t_trees"]))
        rep = out.report
        assert out.exit_code == 0
        assert rep["verdict"]["verdict"] == "ready"
        assert [m["module_id"] for m in rep["modules"]] == ["file_safety", "t_pass", "t_trees"]
        assert module(rep, "t_pass")["status"] == "pass"
        assert module(rep, "t_trees")["status"] == "pass"
        assert sandbox.calls == ["policy", "inspect", "open"]
        assert sandbox.models[0].closed

    def test_report_json_matches_outcome_and_is_strict(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass"]))
        on_disk = json.loads(out.report_json.read_text())
        assert on_disk == json.loads(_strict(out.report))
        assert out.report_json == tmp_path / "run" / "report.json"
        assert out.report_html is None and not (tmp_path / "run" / "report.html").exists()

    def test_report_top_level_contract(self, tmp_path, sandbox):
        rep = _run(tmp_path, make_cfg(["t_pass"]), command="malvalid run --adapter x").report
        assert rep["schema_version"] == REPORT_SCHEMA_VERSION
        for key in ("tool", "run", "verdict", "gate", "model", "corpus", "schema", "training_manifest",
                    "config", "environment", "sandbox", "modules", "artifacts", "warnings", "disclaimers"):
            assert key in rep, key
        assert rep["tool"]["name"] == "malvalid"
        run = rep["run"]
        for key in ("id", "started_at", "finished_at", "duration_s", "command", "seed", "out_dir"):
            assert key in run
        assert run["command"] == "malvalid run --adapter x"
        assert run["started_at"].endswith("Z") and run["duration_s"] >= 0
        gate = rep["gate"]
        for key in ("exit_code", "fail_on", "hard_gates", "failed_hard", "errored", "aborted"):
            assert key in gate
        assert gate["hard_gates"][0] == {"module_id": "file_safety", "code": "M0", "title": "Model file safety",
                                         "status": "pass", "gate_outcome": "passed"}
        model = rep["model"]
        assert model["feature_version"] == "toy_v1" and model["operating_threshold"] == 0.5
        art = model["artifacts"][0]
        assert set(art) >= {"path", "sha256", "size", "format", "is_pickle"}
        assert len(art["sha256"]) == 64 and art["is_pickle"] is False
        assert len(model["adapter_sha256"]) == 64
        assert set(model["tree_access"]) >= {"available", "n_trees", "fidelity_max_abs_diff", "note"}
        assert rep["schema"]["name"] == "toy_v1" and rep["schema"]["dim"] == 32
        assert "featurize_available" in rep["schema"]
        env = rep["environment"]
        assert {"python", "platform", "cpu_count", "libraries"} <= set(env)
        for lib in ("numpy", "scipy", "sklearn", "lightgbm", "xgboost", "shap", "art", "modelscan",
                    "onnxruntime", "tesseract", "lief", "pefile", "jinja2", "pydantic"):
            assert lib in env["libraries"]
        assert env["libraries"]["numpy"] == np.__version__
        assert rep["config"]["corpus"] == "toy_v1_corpus"
        assert rep["sandbox"]["backend"] == "in-process-fake"
        assert rep["disclaimers"]

    def test_corpus_block(self, tmp_path, sandbox):
        rep = _run(tmp_path, make_cfg(["t_pass"])).report
        c = rep["corpus"]
        assert c["name"] == "toy_v1_corpus" and c["error"] is None
        assert c["content_hash"] == toy_corpus().content_hash
        assert c["n"] == toy_corpus().n and c["feature_version"] == "toy_v1"
        assert c["excluded_training_members"] == 0
        assert any("synthetic" in d for d in rep["disclaimers"])

    def test_tree_access_verified(self, tmp_path, sandbox):
        rep = _run(tmp_path, make_cfg(["t_trees"])).report
        ta = rep["model"]["tree_access"]
        assert ta["available"] is True and ta["n_trees"] == 20
        assert ta["fidelity_max_abs_diff"] <= 1e-4 and ta["rows_source"] == "corpus"
        assert 0 < ta["n_rows_checked"] <= runner.TREE_FIDELITY_MAX_ROWS
        assert "tree_access" in rep["capabilities"]

    def test_context_handed_to_modules(self, tmp_path, sandbox):
        cfg = make_cfg(["t_pass"], seed=7)
        _run(tmp_path, cfg)
        ctx = PassMod.seen[-1]
        assert ctx.seed == module_seed(7, "t_pass")
        assert ctx.params == {"min_rate": 0.5}
        assert ctx.schema.name == "toy_v1" and ctx.corpus is not None
        assert ctx.training_hashes == frozenset(train_hashes())
        assert ctx.training_cutoff.isoformat() == "2017-12-31"
        assert ctx.run_dir == (tmp_path / "run").resolve()
        assert ctx.private_dir == ctx.run_dir / "private" and ctx.private_dir.is_dir()
        assert ctx.deadline is not None and ctx.extras["adapter_path"].endswith("my_adapter.py")
        # The model got a deadline for the module and it was cleared afterwards.
        m = sandbox.models[0]
        assert m.deadlines[-1] is None and any(d is not None for d in m.deadlines)

    def test_module_params_from_config(self, tmp_path, sandbox):
        cfg = make_cfg(["t_pass"])
        cfg.modules["t_pass"] = type(cfg.modules["t_pass"])(enabled=True, min_rate=0.8)
        rep = _run(tmp_path, cfg).report
        assert module(rep, "t_pass")["params"] == {"min_rate": 0.8}
        assert module(rep, "t_pass")["checks"][0]["threshold"] == 0.8

    def test_run_log_written(self, tmp_path, sandbox):
        _run(tmp_path, make_cfg(["t_pass"]))
        text = (tmp_path / "run" / "run.log").read_text()
        assert "verdict: ready" in text and "T1 Fake passing axis: pass" in text

    def test_determinism_same_seed(self, tmp_path, sandbox):
        def strip(rep):
            return [{k: v for k, v in m.items() if k != "duration_s"} for m in rep["modules"]]

        a = run_gate(RunOptions(adapter=tmp_path / "adapter" / "my_adapter.py", config=make_cfg(["t_pass"], seed=3),
                                out_dir=tmp_path / "a", write_html=False)).report
        b = run_gate(RunOptions(adapter=tmp_path / "adapter" / "my_adapter.py", config=make_cfg(["t_pass"], seed=3),
                                out_dir=tmp_path / "b", write_html=False)).report
        assert strip(a) == strip(b)
        assert a["verdict"] == b["verdict"]
        assert a["run"]["id"] != b["run"]["id"]


# --------------------------------------------------------------------------------------------------
# Gate outcomes and exit codes
# --------------------------------------------------------------------------------------------------


class TestGatesAndExitCodes:
    def test_warn_gate_failure(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass", "t_warn"]))
        m = module(out.report, "t_warn")
        assert m["status"] == "warn" and m["gate"] == "warn" and m["gate_outcome"] == "failed"
        assert out.report["verdict"]["verdict"] in ("conditional", "not_ready")
        assert out.exit_code == 0  # default --fail-on blocked

    @pytest.mark.parametrize(
        "fail_on, expected", [("blocked", 0), ("not_ready", 0), ("conditional", 1)]
    )
    def test_fail_on_levels_for_conditional(self, tmp_path, sandbox, fail_on, expected):
        out = _run(tmp_path, make_cfg(["t_pass", "t_warn"]), fail_on=fail_on)
        assert out.report["verdict"]["verdict"] == "conditional"
        assert out.exit_code == expected
        assert out.report["gate"]["fail_on"] == fail_on
        assert out.report["gate"]["exit_code"] == expected

    def test_hard_fail_blocks_exit_1(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass", "t_hard"]))
        m = module(out.report, "t_hard")
        assert m["status"] == "fail" and m["gate"] == "hard" and m["gate_outcome"] == "failed"
        v = out.report["verdict"]
        assert v["verdict"] == "blocked" and v["score"] <= 49
        assert any("T3" in b for b in v["blockers"])
        assert out.exit_code == 1
        assert out.report["gate"]["failed_hard"] == ["t_hard"]
        assert v["label"] == "Blocked — a hard gate failed"
        assert v["summary"].startswith("Blocked — a hard gate failed. Production-readiness score ")
        if v["capped"]:
            assert v["summary"].endswith("Score capped at 49 because the run is blocked.")

    def test_config_gate_overrides_module_default(self, tmp_path, sandbox):
        # t_warn defaults to a warn gate; the config makes it hard => BLOCKED.
        out = _run(tmp_path, make_cfg({"t_pass": None, "t_warn": "hard"}))
        assert module(out.report, "t_warn")["status"] == "fail"
        assert out.exit_code == 1
        # ...and a hard-by-default module downgraded to warn only warns.
        out = run_gate(RunOptions(adapter=tmp_path / "adapter" / "my_adapter.py",
                                  config=make_cfg({"t_pass": None, "t_hard": "warn"}),
                                  out_dir=tmp_path / "r2", write_html=False))
        assert module(out.report, "t_hard")["status"] == "warn" and out.exit_code == 0

    def test_report_config_records_effective_gate(self, tmp_path, sandbox):
        # t_hard defaults to a hard gate and the config does not set one: the report says "hard".
        rep = _run(tmp_path, make_cfg(["t_hard", "t_warn"])).report
        assert rep["config"]["modules"]["t_hard"]["gate"] == "hard"
        assert rep["config"]["modules"]["t_warn"]["gate"] == "warn"
        assert module(rep, "t_hard")["gate"] == "hard"

    def test_crash_is_error_exit_2(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass", "t_crash"]))
        m = module(out.report, "t_crash")
        assert m["status"] == "error" and m["gate_outcome"] == "not_evaluated"
        assert m["finding"] == "module crashed: ValueError: boom: something went wrong"
        assert "Traceback" in m["error"] and "second line" in m["error"]
        assert m["score"] is None and m["duration_s"] is not None
        assert out.report["verdict"]["verdict"] == "blocked"
        # verdict.py's label would claim a hard gate failed; the report names the real cause.
        assert out.report["verdict"]["label"] == "Blocked — a module errored"
        assert "hard gate failed" not in out.report["verdict"]["summary"]
        assert out.exit_code == 2
        assert out.report["gate"]["errored"] == ["t_crash"]
        # The crash did not stop later modules.
        assert module(out.report, "t_pass")["status"] == "pass"

    def test_invalid_fail_on_rejected(self, tmp_path, sandbox):
        with pytest.raises(ConfigError, match="--fail-on"):
            _run(tmp_path, make_cfg(["t_pass"]), fail_on="ready")
        assert validate_fail_on("not_ready") == "not_ready"

    def test_bad_module_return_is_error(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_badret"]))
        m = module(out.report, "t_badret")
        assert m["status"] == "error" and "expected ModuleResult" in m["finding"]
        assert out.exit_code == 2

    def test_result_gate_is_corrected_to_configured_gate(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg({"t_pass": None, "t_wronggate": "hard"}))
        m = module(out.report, "t_wronggate")
        assert m["gate"] == "hard" and m["status"] == "fail"
        assert m["score"] is None  # NaN score treated as unscored
        assert any("non-finite" in n for n in m["notes"])
        assert out.exit_code == 1


# --------------------------------------------------------------------------------------------------
# Skips
# --------------------------------------------------------------------------------------------------


class TestSkips:
    def test_skip_by_requirement_cites_specific_hint(self, tmp_path, monkeypatch):
        fs = install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, hashes=None)))
        out = _run(tmp_path, make_cfg(["t_pass", "t_hashes", "t_sample"]))
        h = module(out.report, "t_hashes")
        assert h["status"] == "skipped" and h["score"] is None
        assert h["skip_reason"] == "no training manifest (training_hashes_path not declared)"
        assert h["finding"].startswith("Not run — ")
        s = module(out.report, "t_sample")
        assert s["skip_reason"] == "no sample corpus (sample_dir not set or missing)"
        assert NeedsHashesMod.seen == []
        assert out.report["training_manifest"]["declared"] is False
        v = out.report["verdict"]
        assert v["coverage"] < 1 and any("was skipped" in r for r in v["reasons"])
        assert fs.calls[-1] == "open"

    def test_requires_any_one_satisfied_runs(self, tmp_path, monkeypatch):
        # Corpus missing, but tree access (verified on random vectors) satisfies requires_any.
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path)))
        out = _run(tmp_path, make_cfg(["t_any", "t_pass"], corpus="t_missing_corpus"))
        assert module(out.report, "t_any")["status"] == "pass"
        assert NeedsAnyMod.seen and NeedsAnyMod.seen[-1].corpus is None
        ta = out.report["model"]["tree_access"]
        assert ta["available"] and ta["rows_source"] == "random"

    def test_requires_any_none_satisfied_skips(self, tmp_path, monkeypatch):
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path), trees=False))
        out = _run(tmp_path, make_cfg(["t_any"], corpus="t_missing_corpus"))
        r = module(out.report, "t_any")["skip_reason"]
        assert r.startswith("needs tree access or feature space, and none is available: ")
        assert "model does not expose its tree structure" in r
        assert "not found at" in r  # cites the corpus error

    def test_corpus_unavailable_skips_feature_space_modules(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass", "t_warn"], corpus="t_missing_corpus"))
        rep = out.report
        p = module(rep, "t_pass")
        assert p["status"] == "skipped"
        assert p["skip_reason"].startswith("no canonical corpus in the model's feature_version is loaded (")
        assert "t_missing_corpus" in p["skip_reason"] and "not found at" in p["skip_reason"]
        # Query-only modules still run.
        assert module(rep, "t_warn")["status"] == "warn"
        assert rep["corpus"]["error"] and rep["corpus"]["available"] is False
        assert rep["corpus"]["excluded_training_members"] is None
        assert any("canonical corpus not usable" in w for w in rep["warnings"])
        assert "feature_space" not in rep["capabilities"]

    def test_corpus_load_failure_is_recorded(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass"], corpus="t_broken_corpus"))
        err = out.report["corpus"]["error"]
        assert "OSError" in err and "disk on fire" in err
        assert "disk on fire" in module(out.report, "t_pass")["skip_reason"]

    def test_feature_version_mismatch(self, tmp_path, monkeypatch):
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, feature_version="toy_v2")))
        out = _run(tmp_path, make_cfg(["t_pass", "t_trees"]))
        rep = out.report
        err = rep["corpus"]["error"]
        assert "feature space 'toy_v1'" in err and "feature_version 'toy_v2'" in err
        assert rep["corpus"]["available"] is True and rep["corpus"]["used_for_evaluation"] is False
        assert module(rep, "t_pass")["status"] == "skipped"
        assert "toy_v2" in module(rep, "t_pass")["skip_reason"]
        assert rep["schema"]["name"] == "toy_v2"
        # Tree fidelity fell back to random vectors (the corpus is not in the model's space).
        assert rep["model"]["tree_access"]["rows_source"] == "random"
        assert module(rep, "t_trees")["status"] == "pass"

    def test_unknown_feature_version_is_adapter_error(self, tmp_path, monkeypatch):
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, feature_version="ember_v9")))
        with pytest.raises(AdapterError, match="ember_v9"):
            _run(tmp_path, make_cfg(["t_pass"]))

    def test_unknown_corpus_is_config_error(self, tmp_path, sandbox):
        with pytest.raises(ConfigError, match="unknown corpus 'nope'"):
            _run(tmp_path, make_cfg(["t_pass"], corpus="nope"))

    def test_disabled_modules_do_not_run(self, tmp_path, sandbox):
        cfg = make_cfg(["t_pass", "t_warn"])
        cfg.modules["t_warn"] = type(cfg.modules["t_warn"])(enabled=False)
        rep = _run(tmp_path, cfg).report
        assert [m["module_id"] for m in rep["modules"]] == ["file_safety", "t_pass"]
        assert "t_warn" in rep["gate"]["disabled"]


# --------------------------------------------------------------------------------------------------
# Timeouts
# --------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM not available")
class TestTimeouts:
    def test_sigalrm_timeout_is_error(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_slow", "t_pass"], max_seconds_per_module=0.3))
        m = module(out.report, "t_slow")
        assert m["status"] == "error"
        assert "exceeded 0.3 s" in m["finding"]
        assert "ModuleTimeout" in m["error"]
        assert m["duration_s"] < 5
        assert out.exit_code == 2
        # The sandboxed model is restarted after a hard timeout; the next module still runs.
        assert sandbox.models[0].restarts == 1
        assert module(out.report, "t_pass")["status"] == "pass"
        # No stray timer is left armed.
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0

    def test_restart_after_sigalrm_timeout_clears_the_expired_deadline(self, tmp_path, monkeypatch):
        """Regression (xcomp-sigalrm-restart-uses-expired-deadline): the restart after a SIGALRM timeout
        must not run under the module deadline that just expired (the real SandboxedModel refuses to
        start then), or every later module is skipped as 'worker unavailable'."""
        from tests.unit.test_runner_support import RecordingModel, _Detector, toy_booster

        class StrictRestartModel(RecordingModel):
            def restart(self) -> None:  # like SandboxedModel.start(): no time left => ModuleTimeout
                if self.deadlines and self.deadlines[-1] is not None and time.monotonic() >= self.deadlines[-1]:
                    raise runner.ModuleTimeout("the module deadline passed while the sandboxed model was (re)starting")
                super().restart()

        fs = FakeSandbox(decl=make_decl(tmp_path, hashes=train_hashes()))

        def _factory() -> StrictRestartModel:
            m = StrictRestartModel(_Detector(toy_booster()), fs.decl)
            fs.models.append(m)
            return m

        fs.model_factory = _factory  # type: ignore[method-assign]
        install_fake_sandbox(monkeypatch, fs)
        out = _run(tmp_path, make_cfg(["t_slow", "t_deadline"], max_seconds_per_module=0.3))
        rep = out.report
        assert module(rep, "t_slow")["status"] == "error"
        assert fs.models[0].restarts == 1
        nxt = module(rep, "t_deadline")  # runs after T9: it must have run (and timed out cooperatively)
        assert nxt["status"] == "error" and "exceeded 0.3 s" in nxt["finding"], nxt
        assert not any("could not be restarted" in w for w in rep["warnings"]), rep["warnings"]

    def test_swallowed_timeout_still_error(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_swallow"], max_seconds_per_module=0.3))
        m = module(out.report, "t_swallow")
        assert m["status"] == "error" and "exceeded" in m["finding"] and m["score"] is None

    def test_cooperative_deadline_is_error(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_deadline"], max_seconds_per_module=0.2))
        m = module(out.report, "t_deadline")
        assert m["status"] == "error" and "exceeded 0.2 s" in m["finding"]
        assert sandbox.models[0].restarts == 0  # the timer never fired; the worker is healthy

    def test_outer_timer_restored(self, tmp_path, sandbox):
        fired = []
        prev = signal.signal(signal.SIGALRM, lambda *a: fired.append(True))
        try:
            signal.setitimer(signal.ITIMER_REAL, 30.0)
            _run(tmp_path, make_cfg(["t_pass"], max_seconds_per_module=5))
            remaining = signal.getitimer(signal.ITIMER_REAL)[0]
            assert 0 < remaining <= 30.0
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)
        assert not fired


# --------------------------------------------------------------------------------------------------
# M0 first / abort / model load failure
# --------------------------------------------------------------------------------------------------


class TestFileSafetyFirst:
    def test_m0_runs_first_on_unloaded_placeholder(self, tmp_path, sandbox):
        _run(tmp_path, make_cfg(["t_pass"]))
        ctx = FakeM0Pass.seen[-1]
        assert type(ctx.model).__name__ == "_UnloadedModel"
        with pytest.raises(RuntimeError, match="not loaded"):
            ctx.model.predict_proba(np.zeros((1, 32), dtype=np.float32))
        assert ctx.extras["artifact_paths"] == list(sandbox.decl.model_paths)
        assert ctx.extras["allow_pickle"] is False
        assert ctx.gate.value == "hard"

    def test_m0_abort_blocks_and_skips_everything(self, tmp_path, sandbox):
        sandbox.m0 = FakeM0Abort
        out = _run(tmp_path, make_cfg(["t_pass", "t_warn", "t_crash"]))
        rep = out.report
        assert "open" not in sandbox.calls  # the untrusted model was never loaded
        assert module(rep, "file_safety")["status"] == "fail"
        for mid in ("t_pass", "t_warn", "t_crash"):
            m = module(rep, mid)
            assert m["status"] == "skipped"
            assert m["skip_reason"].startswith("run aborted by M0: CRITICAL")
        v = rep["verdict"]
        assert v["verdict"] == "blocked" and v["score"] is None
        assert any("run aborted before the model was loaded" in b for b in v["blockers"])
        assert rep["gate"]["aborted"] is True and out.exit_code == 1
        assert v["label"] == "Blocked — unsafe model artifact"  # most specific cause wins
        assert rep["model"]["tree_access"]["available"] is False
        assert PassMod.seen == [] and CrashMod.seen == []

    def test_m0_crash_refuses_to_load(self, tmp_path, sandbox):
        sandbox.m0 = FakeM0Crash
        out = _run(tmp_path, make_cfg(["t_pass"]))
        assert "open" not in sandbox.calls
        assert module(out.report, "file_safety")["status"] == "error"
        assert "refusing to load an unscanned model" in out.report["gate"]["abort_reason"]
        assert out.report["verdict"]["label"] == "Blocked — the model-file safety scan did not complete"
        assert module(out.report, "t_pass")["status"] == "skipped"
        assert out.exit_code == 2

    def test_m0_disabled_in_config_still_runs(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass"], m0={"enabled": False}))
        assert module(out.report, "file_safety")["status"] == "pass"
        assert any("file_safety (M0) is disabled" in w for w in out.report["warnings"])

    def test_m0_warn_gate_forced_hard(self, tmp_path, sandbox):
        sandbox.m0 = FakeM0Abort
        out = _run(tmp_path, make_cfg(["t_pass"], m0={"gate": "warn"}))
        m0 = module(out.report, "file_safety")
        assert m0["gate"] == "hard" and m0["status"] == "fail"
        assert any("M0 runs as a hard gate" in w for w in out.report["warnings"])
        assert out.exit_code == 1

    def test_m0_cannot_be_skipped(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass"]), skip=["M0"])
        assert module(out.report, "file_safety")["status"] == "pass"
        assert any("cannot be skipped" in w for w in out.report["warnings"])

    def test_m0_unavailable_is_error_and_aborts(self, tmp_path, sandbox, monkeypatch):
        def _missing():
            raise KeyError("unknown module 'file_safety'")

        monkeypatch.setattr(runner, "_file_safety_cls", _missing)
        out = _run(tmp_path, make_cfg(["t_pass"]))
        m0 = module(out.report, "file_safety")
        assert m0["status"] == "error" and "could not be scanned" in m0["finding"]
        assert "open" not in sandbox.calls and out.exit_code == 2

    def test_model_load_failure(self, tmp_path, sandbox):
        sandbox.open_error = SandboxError("worker died: ImportError: no module named 'mylib'")
        out = _run(tmp_path, make_cfg(["t_pass", "t_warn"]))
        rep = out.report
        assert module(rep, "file_safety")["status"] == "pass"
        for mid in ("t_pass", "t_warn"):
            assert module(rep, mid)["skip_reason"].startswith("model failed to load: SandboxError: worker died")
        assert rep["verdict"]["verdict"] == "blocked"
        assert rep["model"]["load_error"].startswith("SandboxError")
        assert rep["verdict"]["label"] == "Blocked — the model failed to load in the sandbox"
        assert rep["verdict"]["summary"] == "Blocked — the model failed to load in the sandbox. No axis could be scored."
        assert out.exit_code == 2

    def test_adapter_inspection_error_propagates(self, tmp_path, sandbox):
        sandbox.inspect_error = AdapterError("MyDetector.operating_threshold must be in [0, 1]")
        with pytest.raises(AdapterError, match="operating_threshold"):
            _run(tmp_path, make_cfg(["t_pass"]))
        text = (tmp_path / "run" / "run.log").read_text()
        assert "run failed: MyDetector.operating_threshold must be in [0, 1]" in text
        assert "Traceback" not in text

    def test_internal_error_traceback_goes_to_run_log(self, tmp_path, sandbox, monkeypatch, caplog):
        def boom(*a, **k):
            raise ZeroDivisionError("oops")

        monkeypatch.setattr(runner, "compute_verdict", boom)
        with caplog.at_level(logging.INFO, logger="malvalid"), pytest.raises(ZeroDivisionError):
            _run(tmp_path, make_cfg(["t_pass"]))
        text = (tmp_path / "run" / "run.log").read_text()
        assert "internal error: ZeroDivisionError('oops')" in text and "Traceback" in text
        assert not any(r.exc_info for r in caplog.records)  # no traceback on the console handlers

    def test_missing_adapter_file(self, tmp_path, sandbox):
        opts = RunOptions(adapter=tmp_path / "nope.py", config=make_cfg(["t_pass"]), out_dir=tmp_path / "r")
        with pytest.raises(AdapterError, match="adapter not found"):
            run_gate(opts)

    def test_undeclared_artifacts_found_by_m0_are_reported(self, tmp_path, sandbox, monkeypatch):
        """No ``model_path`` declared: the files M0 discovered identify the model in ``model.artifacts``."""
        found = tmp_path / "adapter" / "model.txt"

        class FakeM0Discover(FakeM0Pass):
            def run(self, ctx):
                res = super().run(ctx)
                res.details = {"abort": False, "artifacts": [
                    {"path": str(found), "sha256": "ab" * 32, "size": 17, "format": "lightgbm_text",
                     "is_pickle": False, "declared": False, "findings": [{"severity": "LOW"}]},
                    {"path": str(tmp_path / "adapter" / "my_adapter.py"), "declared": True},  # not a model file
                    "garbage",
                ]}
                return res

        sandbox.decl = make_decl(tmp_path, hashes=train_hashes(), model_paths=())
        monkeypatch.setattr(runner, "_file_safety_cls", lambda: FakeM0Discover)
        out = _run(tmp_path, make_cfg(["t_pass"]))
        arts = out.report["model"]["artifacts"]
        assert arts == [{"path": str(found), "sha256": "ab" * 32, "size": 17, "format": "lightgbm_text",
                         "is_pickle": False, "declared": False}]
        _strict(out.report)

    def test_unavailable_module_warned_once(self, tmp_path, sandbox, monkeypatch):
        """An uninstalled module is reported as unavailable, not again as an unknown verdict weight."""
        from malvalid.config import ModuleConfig

        registry.unavailable("modules")  # make sure the group is loaded before patching
        monkeypatch.setitem(registry._unavailable["modules"], "t_ghost", "ModuleNotFoundError: t_ghost")
        cfg = make_cfg(["t_pass"], verdict={"weights": {"t_pass": 1.0, "t_ghost": 1.0}})
        cfg.modules["t_ghost"] = ModuleConfig(enabled=False)
        out = _run(tmp_path, cfg)
        ghost = [w for w in out.report["warnings"] if "t_ghost" in w]
        assert ghost == ["config: module 't_ghost' unavailable (ModuleNotFoundError: t_ghost)"], ghost

    def test_declared_artifacts_are_marked(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass"]))
        assert [a["declared"] for a in out.report["model"]["artifacts"]] == [True]

    def test_model_override_paths(self, tmp_path, sandbox):
        other = tmp_path / "elsewhere" / "model.json"
        other.parent.mkdir()
        other.write_text("{}")
        out = _run(tmp_path, make_cfg(["t_pass"]), model_paths=[other])
        assert out.report["model"]["artifacts"][0]["path"] == str(other.resolve())
        assert other.parent.resolve() in sandbox.policies[0]["extra_ro"]
        with pytest.raises(AdapterError, match="--model"):
            _run(tmp_path, make_cfg(["t_pass"]), model_paths=[tmp_path / "missing.txt"])

    def test_allow_pickle_propagates(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass"]), allow_pickle=True)
        assert sandbox.policies[0]["allow_pickle"] is True
        assert FakeM0Pass.seen[-1].extras["allow_pickle"] is True
        assert out.report["run"]["allow_pickle"] is True


# --------------------------------------------------------------------------------------------------
# Training manifest / tree fidelity
# --------------------------------------------------------------------------------------------------


class TestManifestAndTrees:
    def test_training_member_exclusion_count(self, tmp_path, monkeypatch):
        hs = train_hashes(extra_eval_members=37)
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, hashes=hs)))
        rep = _run(tmp_path, make_cfg(["t_pass", "t_hashes"])).report
        assert rep["corpus"]["excluded_training_members"] == 37
        tm = rep["training_manifest"]
        assert tm["n_hashes"] == len(set(hs)) and tm["n_in_corpus"] == len(set(hs))
        assert tm["cutoff"] == "2017-12" and tm["cutoff_parsed"] == "2017-12-31"
        assert module(rep, "t_hashes")["status"] == "pass"
        # The module received the manifest so it can exclude members itself.
        assert PassMod.seen[-1].training_hashes == frozenset(hs)

    def test_bad_manifest_is_adapter_error(self, tmp_path, monkeypatch):
        decl = make_decl(tmp_path, hashes=["not-a-hash"])
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=decl))
        with pytest.raises(AdapterError, match="not sha256"):
            _run(tmp_path, make_cfg(["t_pass"]))

    def test_bad_cutoff_is_adapter_error(self, tmp_path, monkeypatch):
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, cutoff="last spring")))
        with pytest.raises(AdapterError, match="training_cutoff"):
            _run(tmp_path, make_cfg(["t_pass"]))

    def test_empty_manifest_warns(self, tmp_path, monkeypatch):
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, hashes=[])))
        rep = _run(tmp_path, make_cfg(["t_hashes"])).report
        assert module(rep, "t_hashes")["status"] == "skipped"
        assert any("contains no sha256 hashes" in w for w in rep["warnings"])

    def test_empty_manifest_is_not_reported_as_undeclared(self, tmp_path, monkeypatch):
        """Regression (xcomp-empty-manifest-reported-as-undeclared): a declared but empty manifest is
        described as such in skip reasons and in what modules receive, not as 'not declared'."""
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, hashes=[])))
        rep = _run(tmp_path, make_cfg(["t_hashes", "t_pass"])).report
        assert rep["training_manifest"]["declared"] is True and rep["training_manifest"]["n_hashes"] == 0
        r = module(rep, "t_hashes")["skip_reason"]
        assert "contains no sha256 hashes" in r and "not declared" not in r
        ctx = PassMod.seen[-1]
        assert ctx.extras["training_manifest_declared"] is True
        assert "contains no sha256 hashes" in ctx.missing_reason(Requirement.TRAINING_HASHES)

    def test_undeclared_manifest_keeps_the_generic_reason(self, tmp_path, monkeypatch):
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path, hashes=None)))
        _run(tmp_path, make_cfg(["t_pass"]))
        ctx = PassMod.seen[-1]
        assert ctx.extras["training_manifest_declared"] is False
        assert ctx.missing_reason(Requirement.TRAINING_HASHES) == "no training manifest (training_hashes_path not declared)"

    def test_rejected_trees_reason_reaches_modules(self, tmp_path, monkeypatch):
        """Regression (xcomp-generic-skip-reasons-contradict-runner): modules that satisfy requires_any
        and decide for themselves see the fidelity-check note, not 'does not expose its trees'."""
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path), offset=0.01))
        _run(tmp_path, make_cfg(["t_any"]))
        ctx = NeedsAnyMod.seen[-1]
        assert "does not reproduce predict_proba" in ctx.missing_reason(Requirement.TREE_ACCESS)
        assert ctx.missing_reason(Requirement.SAMPLE_DIR) == "no sample corpus (sample_dir not set or missing)"

    def test_corpus_error_reaches_modules(self, tmp_path, sandbox):
        _run(tmp_path, make_cfg(["t_warn"], corpus="t_missing_corpus"))
        ctx = WarnMod.seen[-1]
        why = ctx.missing_reason(Requirement.FEATURE_SPACE)
        assert why.startswith("no canonical corpus in the model's feature_version is loaded (") and "not found at" in why

    def test_tree_fidelity_failure_disables_tree_access(self, tmp_path, monkeypatch):
        fs = install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path), offset=0.01))
        rep = _run(tmp_path, make_cfg(["t_trees", "t_any"])).report
        ta = rep["model"]["tree_access"]
        assert ta["available"] is False and ta["n_trees"] == 20
        assert ta["fidelity_max_abs_diff"] > runner.TREE_FIDELITY_TOL
        assert "does not reproduce predict_proba" in ta["note"]
        assert "tree_access" not in rep["capabilities"]
        t = module(rep, "t_trees")
        assert t["status"] == "skipped" and "does not reproduce predict_proba" in t["skip_reason"]
        assert NeedsTreesMod.seen == []
        # requires_any still satisfied through the feature space.
        assert module(rep, "t_any")["status"] == "pass"
        assert any("tree access disabled" in w for w in rep["warnings"])
        assert fs.models[0].closed

    def test_model_without_trees(self, tmp_path, monkeypatch):
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path), trees=False))
        rep = _run(tmp_path, make_cfg(["t_trees"])).report
        assert rep["model"]["tree_access"]["available"] is False
        assert module(rep, "t_trees")["skip_reason"] == "model does not expose its tree structure"

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM not available")
    def test_hung_tree_export_is_bounded_by_the_module_budget(self, tmp_path, monkeypatch):
        """Regression (xcomp-tree-export-has-no-deadline): a tree_ensemble() that never returns must
        not hang the run: the export gets the module budget, tree access is disabled, the run goes on
        and report.json is written."""
        from tests.unit.test_runner_support import RecordingModel

        def _hang(self):
            end = time.monotonic() + 20.0
            while time.monotonic() < end:
                time.sleep(0.01)
            return None

        monkeypatch.setattr(RecordingModel, "tree_ensemble", _hang)
        fs = install_fake_sandbox(monkeypatch, FakeSandbox(decl=make_decl(tmp_path)))
        t0 = time.monotonic()
        out = _run(tmp_path, make_cfg(["t_pass", "t_trees"], max_seconds_per_module=0.4))
        assert time.monotonic() - t0 < 10
        rep = out.report
        ta = rep["model"]["tree_access"]
        assert ta["available"] is False and ta["export_timed_out"] is True
        assert "did not finish within 0.4 s" in ta["note"]
        assert module(rep, "t_pass")["status"] == "pass"
        t = module(rep, "t_trees")
        assert t["status"] == "skipped" and "did not finish within 0.4 s" in t["skip_reason"]
        assert any("tree export" in w for w in rep["warnings"])
        assert out.report_json.exists()
        # The export ran under a deadline, which was cleared before the modules started.
        m = fs.models[0]
        assert m.deadlines[0] is not None and m.deadlines[1] is None
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


# --------------------------------------------------------------------------------------------------
# Strict JSON / private data / HTML
# --------------------------------------------------------------------------------------------------


class TestReportHygiene:
    def test_strict_json_with_nonfinite_values(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass", "t_dirty"]))
        text = _strict(out.report)
        json.loads(out.report_json.read_text())  # file parses
        assert "NaN" not in out.report_json.read_text() and "Infinity" not in out.report_json.read_text()
        m = module(json.loads(text), "t_dirty")
        assert m["metrics"]["nan"] is None and m["metrics"]["inf"] is None
        assert m["metrics"]["arr"] == [1.0, None] and m["metrics"]["np_scalar"] == 0.25
        assert m["metrics"]["date"] == "2020-01-02"
        assert m["checks"][0]["value"] is None  # inf check value
        assert m["status"] == "warn"  # the NaN check could not be evaluated -> never a silent pass

    def test_nothing_under_private_is_referenced(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_dirty"]))
        private = (tmp_path / "run" / "private").resolve()
        assert (private / "t_dirty" / "vectors.npz").exists()  # the data is on disk...
        text = out.report_json.read_text()
        assert str(private) not in text and "vectors.npz" not in text  # ...but never referenced
        assert "private run data withheld" in text
        assert str(private) not in _strict(out.report)
        m = module(out.report, "t_dirty")
        assert "private run data withheld" in m["details"]["private_file"]
        assert out.report["sandbox"]["scratch_dir"] == "<private run data withheld>"

    def test_html_written_via_seam(self, tmp_path, sandbox, monkeypatch):
        seen = {}

        def fake_html(report, path):
            seen["report"] = report
            Path(path).write_text("<html>ok</html>")
            return Path(path)

        monkeypatch.setattr(runner, "_write_html", fake_html)
        out = _run(tmp_path, make_cfg(["t_pass"]), write_html=True)
        assert out.report_html == tmp_path / "run" / "report.html" and out.report_html.exists()
        assert seen["report"]["verdict"] == out.report["verdict"]
        assert json.loads(out.report_json.read_text())["run"]["report_html"] == str(out.report_html)

    def test_html_failure_still_writes_json(self, tmp_path, sandbox, monkeypatch):
        def broken(report, path):
            raise RuntimeError("template missing")

        monkeypatch.setattr(runner, "_write_html", broken)
        out = _run(tmp_path, make_cfg(["t_pass"]), write_html=True)
        assert out.report_html is None and out.report_json.exists()
        on_disk = json.loads(out.report_json.read_text())
        assert any("report.html could not be written" in w and "template missing" in w for w in on_disk["warnings"])
        assert on_disk["run"]["report_html"] is None
        assert out.exit_code == 0

    def test_config_report_html_false(self, tmp_path, sandbox, monkeypatch):
        monkeypatch.setattr(runner, "_write_html", lambda r, p: pytest.fail("html must not be written"))
        cfg = make_cfg(["t_pass"])
        cfg.report.html = False
        assert _run(tmp_path, cfg, write_html=True).report_html is None

    def test_real_html_renderer(self, tmp_path, sandbox):
        pytest.importorskip("malvalid.report.html")
        out = _run(tmp_path, make_cfg(["t_pass", "t_warn", "t_hashes", "t_dirty"]), write_html=True)
        assert not [w for w in out.report["warnings"] if "report.html" in w], out.report["warnings"]
        html = out.report_html.read_text()
        assert "<html" in html.lower()
        assert str((tmp_path / "run" / "private").resolve()) not in html

    def test_sandbox_disabled_warning_and_disclaimer(self, tmp_path, sandbox):
        rep = _run(tmp_path, make_cfg(["t_pass"], sandbox=False)).report
        assert any("sandbox is disabled" in w for w in rep["warnings"])
        assert any("without the sandbox" in d for d in rep["disclaimers"])


# --------------------------------------------------------------------------------------------------
# Selection / config validation
# --------------------------------------------------------------------------------------------------


class TestSelection:
    def test_only_by_code_and_id(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass", "t_warn", "t_hard"]), only=["T1,t_warn"])
        rep = out.report
        # t_hard is enabled in the config: deselected, it is reported (skipped), not silently dropped.
        assert [m["module_id"] for m in rep["modules"]] == ["file_safety", "t_pass", "t_warn", "t_hard"]
        assert [module(rep, m)["status"] for m in ("t_pass", "t_warn")] == ["pass", "warn"]
        assert module(rep, "t_hard")["status"] == "skipped"
        assert "not listed in --only" in module(rep, "t_hard")["skip_reason"]
        assert HardFailMod.seen == []
        assert rep["run"]["only"] == ["T1,t_warn"]
        assert rep["gate"]["deselected"] == ["t_hard"] and rep["gate"]["disabled"] == []

    def test_only_runs_config_disabled_module(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass"]), only=["t_warn"])
        rep = out.report
        assert [m["module_id"] for m in rep["modules"]] == ["file_safety", "t_pass", "t_warn"]
        assert module(rep, "t_pass")["status"] == "skipped" and module(rep, "t_warn")["status"] == "warn"
        assert PassMod.seen == []

    def test_skip(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass", "t_warn"]), skip=["t2"])
        rep = out.report
        assert [m["module_id"] for m in rep["modules"]] == ["file_safety", "t_pass", "t_warn"]
        assert module(rep, "t_warn")["status"] == "skipped"
        assert "excluded with --skip" in module(rep, "t_warn")["skip_reason"]
        assert rep["gate"]["deselected"] == ["t_warn"] and "t_warn" not in rep["gate"]["disabled"]

    def test_skip_does_not_bypass_a_failing_hard_gate(self, tmp_path, sandbox):
        """Regression (xcomp-only-skip-bypasses-hard-gate): --skip on the failing hard gate must still
        block (an unevaluated hard gate), keep it in the coverage denominator and in gate.hard_gates."""
        cfg = make_cfg(["t_pass", "t_hard"])
        full = _run(tmp_path, cfg)
        assert full.report["verdict"]["verdict"] == "blocked" and full.exit_code == 1
        out = _run(tmp_path, cfg, skip=["T3"])
        rep, v = out.report, out.report["verdict"]
        assert HardFailMod.seen and len(HardFailMod.seen) == 1  # only the full run ran it
        assert out.exit_code == 1 and rep["gate"]["passed"] is False
        assert v["verdict"] == "blocked"
        assert v["label"] == "Blocked — a hard gate was deselected with --only/--skip"
        assert any("T3" in b and "hard gate not evaluated" in b and "--skip" in b for b in v["blockers"])
        assert v["coverage"] == 0.5  # t_hard's weight still counts as enabled
        hard = {h["module_id"]: h for h in rep["gate"]["hard_gates"]}
        assert hard["t_hard"]["status"] == "skipped" and hard["t_hard"]["gate_outcome"] == "not_evaluated"
        assert rep["gate"]["not_evaluated_hard"] == ["t_hard"]
        assert rep["gate"]["deselected"] == ["t_hard"] and rep["gate"]["disabled"] == []
        assert any(w.startswith("partial run:") and "t_hard" in w for w in rep["warnings"])

    def test_deselected_hard_gate_blocks_only_under_skipped_hard_gate_fails(self, tmp_path, sandbox):
        out = _run(tmp_path, make_cfg(["t_pass", "t_hard"], skipped_hard_gate_fails=False), skip=["t_hard"])
        v = out.report["verdict"]
        assert v["verdict"] != "blocked" and out.exit_code == 0
        assert v["coverage"] == 0.5 and v["verdict"] == "conditional"  # thin coverage: never READY

    def test_only_subset_reports_partial_coverage_not_ready(self, tmp_path, sandbox):
        """Regression (docs-001): a one-module --only run must not read as READY at 100% coverage."""
        cfg = make_cfg(["t_pass", "t_warn", "t_any"])
        out = _run(tmp_path, cfg, only=["T1"], fail_on="conditional")
        v = out.report["verdict"]
        assert v["coverage"] == pytest.approx(1 / 3, abs=1e-4)
        assert v["verdict"] == "conditional" and out.exit_code == 1
        assert any("of the weighted battery was evaluated" in r for r in v["reasons"])
        assert sum("was skipped (lowers coverage)" in r for r in v["reasons"]) == 2
        assert "33% of the weighted battery evaluated" in v["summary"]

    def test_unknown_only_is_config_error(self, tmp_path, sandbox):
        with pytest.raises(ConfigError, match="--only: unknown module 'M99'"):
            _run(tmp_path, make_cfg(["t_pass"]), only=["M99"])

    def test_resolve_module_ids(self):
        assert resolve_module_ids(None) is None
        assert resolve_module_ids(["T1", "t_pass", " t_warn ,"]) == ["t_pass", "t_warn"]
        with pytest.raises(ConfigError, match="--skip"):
            resolve_module_ids(["zzz"], flag="--skip")

    def test_bad_module_param_value_fails_before_anything_is_loaded(self, tmp_path, sandbox):
        """Regression (xcomp-config-values-validated-late (a)): a bad parameter *value* is a one-line
        config error up front, not an M2 crash after the model has been loaded."""
        cfg = make_cfg(["t_pass"])
        cfg.modules["drift"] = type(cfg.modules["t_pass"])(enabled=True, min_aut_f1="0.7")
        with pytest.raises(ConfigError, match="drift.min_aut_f1 must be a number"):
            _run(tmp_path, cfg)
        assert "inspect" not in sandbox.calls and "open" not in sandbox.calls
        assert validate_against_registry(make_cfg(["t_pass"])) == []

    def test_every_real_module_accepts_its_defaults(self):
        for mid, cls in registry.modules().items():
            cls.validate_params(dict(cls.default_params))  # must not raise

    def test_blocked_score_cap_inside_the_bands_warns(self, tmp_path, sandbox):
        """Regression (xcomp-config-values-validated-late (b))."""
        rep = _run(tmp_path, make_cfg(["t_pass"], verdict={"blocked_score_cap": 95})).report
        assert any("blocked_score_cap (95) is not below verdict.conditional_min (60)" in w for w in rep["warnings"])
        assert not any("blocked_score_cap" in w for w in _run(tmp_path, make_cfg(["t_pass"])).report["warnings"])

    def test_unknown_module_param_is_config_error(self, tmp_path, sandbox):
        cfg = make_cfg(["t_pass"])
        cfg.modules["t_pass"] = type(cfg.modules["t_pass"])(enabled=True, min_rat=0.5)
        with pytest.raises(ConfigError, match="unknown parameter"):
            _run(tmp_path, cfg)

    def test_config_warnings_recorded(self, tmp_path, sandbox):
        cfg = make_cfg(["t_pass"], verdict={"weights": {"not_a_module": 1.0}})
        assert validate_against_registry(cfg)
        rep = _run(tmp_path, cfg).report
        assert any("not_a_module" in w for w in rep["warnings"])

    def test_run_log_handler_removed(self, tmp_path, sandbox):
        before = list(logging.getLogger("malvalid").handlers)
        _run(tmp_path, make_cfg(["t_pass"]))
        assert logging.getLogger("malvalid").handlers == before


def test_registry_restored_after_fixture():
    # Sanity: the fakes are registered only while the fixture is active (this test uses it too).
    assert "t_pass" in registry.modules()
