"""End to end through the *real* sandbox: ``malvalid run`` on a tiny toy LightGBM adapter.

Nothing is faked here: the adapter is imported in a sandboxed worker (``inspect_adapter``), M0 scans
the model file before ``load()``, the model is loaded in the worker (``open_model``), tree access is
verified against ``predict_proba`` over IPC, and M1 (performance) runs on the ``toy_v1_corpus``.
Marked ``slow`` because it spawns sandbox workers (a few seconds).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from malvalid import registry
from malvalid.cli import app
from tests.unit.test_runner_support import toy_booster, train_hashes

pytestmark = pytest.mark.slow

ADAPTER = textwrap.dedent(
    '''
    """Toy submission for the malvalid runner's end-to-end test."""
    from malvalid.adapter import BaseDetector


    class ToyDetector(BaseDetector):
        feature_version = "toy_v1"
        model_kind = "lightgbm"
        operating_threshold = 0.5
        model_path = "model.txt"
        training_hashes_path = "train_hashes.txt"
        training_cutoff = "2017-12"
    '''
)


def _adapter_dir(tmp_path: Path) -> Path:
    d = tmp_path / "submission"
    d.mkdir()
    (d / "toy_adapter.py").write_text(ADAPTER)
    toy_booster().save_model(str(d / "model.txt"))
    (d / "train_hashes.txt").write_text("\n".join(train_hashes()) + "\n")
    return d / "toy_adapter.py"


def _config(tmp_path: Path) -> Path:
    known = [*registry.modules(), *registry.unavailable("modules")]
    mods = {m: {"enabled": False} for m in known if m not in ("file_safety", "dummy")}
    mods["performance"] = {"enabled": True}
    cfg = {"corpus": "toy_v1_corpus", "runtime": {"threads": 2, "max_seconds_per_module": 300},
           "modules": mods}
    p = tmp_path / "gate.yaml"
    p.write_text(json.dumps(cfg))  # JSON is valid YAML
    return p


def test_run_through_real_sandbox(tmp_path: Path) -> None:
    for need in ("file_safety", "performance"):
        if need not in registry.modules():
            pytest.skip(f"{need} not importable: {registry.unavailable('modules').get(need)}")
    try:
        from malvalid.sandbox.host import probe_backends
    except ImportError as e:  # pragma: no cover - sandbox not built yet
        pytest.skip(f"sandbox not importable: {e}")
    if not any(v.get("available") for v in probe_backends().values()):
        pytest.skip("no sandbox backend available on this host")

    adapter = _adapter_dir(tmp_path)
    out = tmp_path / "run"
    res = CliRunner().invoke(
        app, ["run", "--adapter", str(adapter), "--config", str(_config(tmp_path)), "--out", str(out), "--no-html"]
    )
    assert res.exit_code in (0, 1), res.output
    rep = json.loads((out / "report.json").read_text())
    json.dumps(rep, allow_nan=False)

    mods = {m["module_id"]: m for m in rep["modules"]}
    assert list(mods)[:2] == ["file_safety", "performance"]
    assert mods["file_safety"]["status"] == "pass", mods["file_safety"]["finding"]
    assert mods["performance"]["status"] in ("pass", "warn", "fail"), mods["performance"]["finding"]
    assert mods["performance"]["score"] is not None

    model = rep["model"]
    assert model["class_name"] == "ToyDetector"
    assert model["load_error"] is None
    assert model["artifacts"][0]["format"] and model["artifacts"][0]["sha256"]
    assert model["artifacts"][0]["is_pickle"] is False
    assert model["tree_access"]["available"] is True, model["tree_access"]
    assert model["tree_access"]["fidelity_max_abs_diff"] <= 1e-4
    assert model["query_count"] and model["query_count"] > 0

    tm = rep["training_manifest"]
    assert tm["n_hashes"] == len(train_hashes()) and tm["cutoff_parsed"] == "2017-12-31"
    assert rep["corpus"]["error"] is None
    assert rep["sandbox"].get("backend"), rep["sandbox"]
    # The worker's scratch dir lives under private/, which the report never references.
    text = (out / "report.json").read_text()
    assert str(out / "private") not in text and str((out / "private").resolve()) not in text
    assert rep["sandbox"]["scratch_dir"] == "<private run data withheld>"
    assert (out / "run.log").is_file()
    assert "Verdict" in res.output or "VERDICT" in res.output.upper()


# --------------------------------------------------------------------------------------------------
# Deadlines around the real SandboxedModel
# --------------------------------------------------------------------------------------------------

HANG_TREES_METHOD = '''
    def tree_ensemble(self):  # a buggy / hostile export that never returns
        import time
        time.sleep(100000)
'''


def _need_real_sandbox() -> None:
    for need in ("file_safety", "performance"):
        if need not in registry.modules():
            pytest.skip(f"{need} not importable: {registry.unavailable('modules').get(need)}")
    try:
        from malvalid.sandbox.host import probe_backends
    except ImportError as e:  # pragma: no cover - sandbox not built yet
        pytest.skip(f"sandbox not importable: {e}")
    if not any(v.get("available") for v in probe_backends().values()):
        pytest.skip("no sandbox backend available on this host")


def _config_with(tmp_path: Path, enabled: dict, **runtime) -> Path:
    known = [*registry.modules(), *registry.unavailable("modules")]
    mods = {m: {"enabled": False} for m in known if m not in ("file_safety", "dummy")}
    mods.update(enabled)
    cfg = {"corpus": "toy_v1_corpus", "runtime": {"threads": 2, "max_seconds_per_module": 300, **runtime},
           "modules": mods, "verdict": {"required_for_ready": []}}
    p = tmp_path / "gate.yaml"
    p.write_text(json.dumps(cfg))
    return p


def test_hung_tree_export_is_bounded_in_the_real_sandbox(tmp_path: Path) -> None:
    """Regression (xcomp-tree-export-has-no-deadline): the adapter's tree_ensemble() never returns.
    The export is bounded by max_seconds_per_module, the worker is restarted lazily, M1 still runs
    and report.json is written."""
    import time

    _need_real_sandbox()
    adapter = _adapter_dir(tmp_path)
    adapter.write_text(ADAPTER + HANG_TREES_METHOD)
    out = tmp_path / "run"
    t0 = time.monotonic()
    res = CliRunner().invoke(app, ["run", "--adapter", str(adapter), "--config",
                                   str(_config_with(tmp_path, {"performance": {"enabled": True}},
                                                    max_seconds_per_module=6)),
                                   "--out", str(out), "--no-html"])
    assert time.monotonic() - t0 < 90, "the run must not hang on the tree export"
    assert res.exit_code in (0, 1), res.output
    rep = json.loads((out / "report.json").read_text())
    ta = rep["model"]["tree_access"]
    assert ta["available"] is False and ta.get("export_timed_out") is True, ta
    assert "did not finish within 6 s" in ta["note"]
    m1 = {m["module_id"]: m for m in rep["modules"]}["performance"]
    assert m1["status"] in ("pass", "warn", "fail"), m1
    assert json.loads((out / "progress.json").read_text())["stage"] == "done"


def test_sigalrm_timeout_restart_keeps_later_modules_in_the_real_sandbox(tmp_path: Path) -> None:
    """Regression (xcomp-sigalrm-restart-uses-expired-deadline): a module that overruns in harness code
    (SIGALRM) triggers a worker restart; the restart must not run under the expired module deadline,
    so the next module still gets a working model."""
    from malvalid.core import GateCheck, Requirement
    from malvalid.core import Module as _Module
    from tests.unit.test_runner_support import SlowMod

    class ScoresAfterSlow(_Module):
        id = "t_after_slow"
        code = "TZ"
        title = "Scores the model after a timeout"
        requires = (Requirement.FEATURE_SPACE,)

        def run(self, ctx):
            p = ctx.score(ctx.corpus.take(ctx.corpus.eval_indices(1)[:32]))
            c = GateCheck.evaluate("n", float(p.size), ">=", 1.0, ideal=32.0, floor=0.0)
            return self.result(ctx, finding=f"scored {p.size} rows", checks=[c])

    _need_real_sandbox()
    registry.register("modules", SlowMod.id, SlowMod)
    registry.register("modules", ScoresAfterSlow.id, ScoresAfterSlow)
    try:
        adapter = _adapter_dir(tmp_path)
        out = tmp_path / "run"
        cfg = _config_with(tmp_path, {"t_slow": {"enabled": True}, "t_after_slow": {"enabled": True}},
                           max_seconds_per_module=3)
        res = CliRunner().invoke(app, ["run", "--adapter", str(adapter), "--config", str(cfg),
                                       "--out", str(out), "--no-html"])
        assert res.exit_code == 2, res.output  # t_slow errored
        rep = json.loads((out / "report.json").read_text())
        mods = {m["module_id"]: m for m in rep["modules"]}
        assert mods["t_slow"]["status"] == "error" and "exceeded 3 s" in mods["t_slow"]["finding"]
        after = mods["t_after_slow"]
        assert after["status"] == "pass", after
        assert not any("could not be restarted" in w for w in rep["warnings"]), rep["warnings"]
    finally:
        registry.unregister("modules", SlowMod.id)
        registry.unregister("modules", ScoresAfterSlow.id)
