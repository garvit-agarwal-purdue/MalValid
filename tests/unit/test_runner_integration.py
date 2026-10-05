"""Integration: the runner driving the *real* test modules (whichever are installed) on toy data.

Only the sandbox is faked (in-process model over a LightGBM booster trained on ``toy_v1_corpus``);
M0 file safety, M1 performance, M2 drift, M4 membership, M5, M6 extraction and M7 run for real.
Modules that are not importable yet are left disabled, so this test tracks the build as it lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from malvalid import registry, runner
from malvalid.config import load_config
from tests.unit.test_runner_support import (
    FakeSandbox,
    install_fake_sandbox,
    make_decl,
    toy_booster,
    train_hashes,
)

BUILTIN = ("performance", "drift", "membership_inf", "backdoor_screen", "extraction", "explanation")
FAST_PARAMS = {
    "drift": {"min_window_samples": 20},
    "membership_inf": {"min_members": 50, "max_per_side": 500},
    "extraction": {"query_budgets": [100, 500], "fidelity_budget": 500, "n_eval": 300},
}


@pytest.fixture
def real_run(tmp_path, monkeypatch):
    if "file_safety" not in registry.modules():
        pytest.skip(f"file_safety not importable: {registry.unavailable('modules').get('file_safety')}")
    decl = make_decl(tmp_path, hashes=train_hashes())
    toy_booster().save_model(decl.model_paths[0])  # a genuine LightGBM text model for M0 to scan
    install_fake_sandbox(monkeypatch, FakeSandbox(decl=decl))
    monkeypatch.setattr(runner, "_file_safety_cls", lambda: registry.get_module("file_safety"))
    available = registry.modules()
    mods = {}
    for mid in BUILTIN:
        on = mid in available
        mods[mid] = {"enabled": on, **(FAST_PARAMS.get(mid, {}) if on else {})}
    cfg = load_config(None, overrides={"corpus": "toy_v1_corpus", "runtime": {"sandbox": False, "threads": 2},
                                       "modules": mods})
    out = runner.run_gate(runner.RunOptions(adapter=Path(decl.adapter_path), config=cfg, out_dir=tmp_path / "run",
                                            write_html=True, command="pytest"))
    return out, [m for m in BUILTIN if m in available]


def test_real_modules_through_the_runner(real_run, tmp_path):
    out, ran = real_run
    rep = out.report
    json.dumps(rep, allow_nan=False)
    ids = [m["module_id"] for m in rep["modules"]]
    assert ids[0] == "file_safety" and ids[1:] == [m for m in registry.modules() if m in ran]
    assert rep["modules"][0]["status"] == "pass", rep["modules"][0]["finding"]
    errored = {m["module_id"]: m["error"] for m in rep["modules"] if m["status"] == "error"}
    assert not errored, errored
    for m in rep["modules"]:
        assert m["status"] in ("pass", "warn", "fail", "skipped")
        assert m["duration_s"] is not None or m["status"] == "skipped"
    assert rep["verdict"]["verdict"] in ("ready", "conditional", "not_ready", "blocked")
    assert out.exit_code in (0, 1)
    assert rep["model"]["tree_access"]["available"] is True
    # Every chart/table a module referenced is present in the artifact store.
    for m in rep["modules"]:
        for key in m["artifacts"]:
            assert key in rep["artifacts"], key
    assert str((tmp_path / "run" / "private").resolve()) not in out.report_json.read_text()
    if out.report_html is None:
        pytest.fail(f"report.html not written: {rep['warnings']}")
