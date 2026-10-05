"""Sandboxed model runner: isolation, output validation, deadlines, crashes, tree access.

One bwrap worker (the toolbox adapter) serves most tests; the first feature of the first row
selects a misbehaviour (see ``tests/fixtures/adapters/toolbox_adapter.py``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

import malvalid.testing  # noqa: F401 - registers the toy_v1 schema
from malvalid.core import AdapterContractError, AdapterError, ModuleTimeout, SandboxError
from malvalid.loaders.trees import TreeEnsemble, from_lightgbm_dump
from malvalid.sandbox.host import SandboxedModel, SandboxPolicy, inspect_adapter, open_model, probe_backends
from tests.fixtures.adapters import toolbox_adapter as tb
from tests.fixtures.adapters.builders import eval_rows, make_adapter_dir

SECRET = "MALVALID_TEST_SECRET"

pytestmark = [pytest.mark.sandbox, pytest.mark.skipif(
    not probe_backends().get("bwrap", {}).get("available"), reason="bubblewrap is not usable on this host"
)]


def _rows(mode: float | None = None, n: int = 50) -> np.ndarray:
    X = eval_rows(n).copy()
    if mode is not None:
        X[0, 0] = mode
    return X


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("sandbox")
    adapter = make_adapter_dir(tmp, "toolbox_adapter")
    policy = SandboxPolicy(backend="bwrap", scratch_dir=tmp / "scratch", threads=1, memory_mb=4096, chunk_rows=64)
    os.environ[SECRET] = "hunter2"
    try:
        decl = inspect_adapter(adapter, policy)
        model = open_model(adapter, policy, declarations=decl)
    finally:
        os.environ.pop(SECRET, None)
    booster = lgb.Booster(model_file=str(adapter.parent / "model.txt"))
    yield {"adapter": adapter, "policy": policy, "decl": decl, "model": model, "booster": booster, "tmp": tmp}
    model.close()


# ---- backends & declarations ----------------------------------------------------------------------


def test_probe_backends_reports_all_backends() -> None:
    res = probe_backends()
    assert set(res) >= {"bwrap", "unshare", "subprocess"}
    for name, r in res.items():
        assert {"available", "network_isolated", "detail"} <= set(r), name
    assert res["bwrap"]["network_isolated"] and res["bwrap"]["readonly_root"] and res["bwrap"]["pid_namespace"]
    assert res["subprocess"]["available"] and not res["subprocess"]["network_isolated"]
    json.dumps(res)


def test_inspect_resolves_declarations(env) -> None:
    d = env["decl"]
    here = env["adapter"].parent
    assert d.class_name == "ToolboxDetector"
    assert (d.feature_version, d.model_kind, d.operating_threshold) == ("toy_v1", "lightgbm", 0.5)
    assert d.model_paths == (str(here / "model.txt"),)
    assert d.training_hashes_path == str(here / "train_hashes.txt")
    assert d.training_cutoff == "2017-12"
    assert d.has_featurize is True
    assert d.adapter_path == str(env["adapter"])


# ---- scoring ----------------------------------------------------------------------------------------


def test_predict_proba_matches_native_and_is_chunked(env) -> None:
    m: SandboxedModel = env["model"]
    X = _rows(n=200)
    before = m.query_count
    p = m.predict_proba(X)
    assert p.dtype == np.float64 and p.shape == (200,)
    np.testing.assert_allclose(p, env["booster"].predict(X), rtol=0, atol=1e-12)
    assert m.query_count - before == 200  # 4 chunks of <= 64 rows, counted per row
    y = m.predict(X)
    assert y.dtype == np.int8
    np.testing.assert_array_equal(y, (p >= 0.5).astype(np.int8))


@pytest.mark.parametrize(
    ("mode", "fragment"),
    [
        (tb.MODE_N_BY_2, "predict_proba(X)[:, 1]"),
        (tb.MODE_NAN, "non-finite"),
        (tb.MODE_OUT_OF_RANGE, "outside [0, 1]"),
        (tb.MODE_WRONG_LENGTH, "expected (50,)"),
    ],
)
def test_predict_proba_contract_violations_are_caught(env, mode: float, fragment: str) -> None:
    m = env["model"]
    pid = m.worker_pid
    with pytest.raises(AdapterContractError) as ei:
        m.predict_proba(_rows(mode))
    assert fragment in str(ei.value)
    assert m.worker_pid == pid  # a contract violation does not cost a restart
    assert m.predict_proba(_rows()).shape == (50,)


def test_predict_contract_violation(env) -> None:
    with pytest.raises(AdapterContractError, match=r"outside \{0, 1\}"):
        env["model"].predict(_rows(tb.MODE_BAD_PREDICT))


def test_adapter_exception_is_reported_with_traceback(env) -> None:
    with pytest.raises(AdapterError) as ei:
        env["model"].predict_proba(_rows(tb.MODE_RAISE))
    msg = str(ei.value)
    assert "ValueError" in msg and "raised on purpose" in msg and "toolbox_adapter.py" in msg
    assert env["model"].predict_proba(_rows()).shape == (50,)


def test_bad_input_shape_is_rejected_before_sending(env) -> None:
    with pytest.raises(ValueError, match="2-D"):
        env["model"].predict_proba(np.zeros(32, dtype=np.float32))


# ---- isolation ---------------------------------------------------------------------------------------


def test_worker_isolation_probes(env) -> None:
    m = env["model"]
    out = m.predict_proba(_rows(tb.MODE_PROBE, n=len(tb.PROBES)))
    got = dict(zip(tb.PROBES, out.tolist()))
    assert got["network_connect"] == 0.0, "socket connect must fail inside the worker"
    assert got["write_adapter_dir"] == 0.0, "the adapter directory must be read-only"
    assert got["write_site_packages"] == 0.0, "the host file system must be read-only"
    assert got["write_scratch"] == 1.0, "the scratch directory must stay writable"
    assert got["secret_env_visible"] == 0.0, "the worker environment must be scrubbed"
    assert got["home_listable"] == 0.0, "$HOME must be hidden"
    assert got["pickle_loads_refused"] == 1.0, "pickle.loads must be refused without allow_pickle"
    assert got["rlimit_as_finite"] == 1.0, "RLIMIT_AS must be set"
    # nothing leaked onto the host
    assert not (env["adapter"].parent / "sandbox_write_probe.txt").exists()
    assert not (Path(np.__file__).resolve().parent.parent / "malvalid_sandbox_write_probe.txt").exists()
    assert (m.scratch / "work" / "probe.txt").exists()  # the scratch write is real


def test_self_check_and_sandbox_info(env) -> None:
    m = env["model"]
    sc = m.self_check()
    assert sc["network_isolated"] is True
    info = m.sandbox_info()
    json.dumps(info, allow_nan=False)
    assert info["enabled"] is True and info["backend"] == "bwrap" and info["network_isolated"] is True
    assert info["readonly_root"] is True and info["home_hidden"] is True
    assert info["allow_pickle"] is False and "pickle.loads" in info["pickle_guards"]
    assert info["rlimits"]["as"] == 4096 << 20 and info["rlimits"]["core"] == 0
    assert info["env_scrubbed"] is True and info["warnings"] == []
    assert Path(info["worker_log"]).is_file()


# ---- featurize & trees -------------------------------------------------------------------------------


def test_featurize_runs_adapter_featurizer_on_benign_exe(env) -> None:
    m = env["model"]
    exe = Path(__import__("setuptools").__file__).parent / "cli-64.exe"
    assert m.has_featurize() and m.featurize_source() == "adapter"
    raw = exe.read_bytes()
    v = m.featurize(raw)
    assert v.dtype == np.float32 and v.shape == (32,) and v[0] == len(raw)
    np.testing.assert_allclose(v[1:8].sum(), 1.0, rtol=1e-5)


def test_tree_ensemble_round_trip_and_fidelity(env) -> None:
    m = env["model"]
    te = m.tree_ensemble()
    assert isinstance(te, TreeEnsemble)
    assert m.tree_ensemble() is te  # cached
    assert "lightgbm" in (m.tree_access_note or "")
    ref = from_lightgbm_dump(env["booster"].dump_model())
    assert te.n_trees == ref.n_trees == 30
    X = _rows(n=300)
    p = m.predict_proba(X)
    assert te.fidelity(X, p) < 1e-9
    np.testing.assert_allclose(te.predict_proba(X), ref.predict_proba(X), atol=1e-12)
    again = TreeEnsemble.from_bytes(te.to_bytes())
    np.testing.assert_array_equal(again.predict_raw(X), te.predict_raw(X))


# ---- deadlines & crashes -----------------------------------------------------------------------------


def test_deadline_kills_slow_call_and_next_call_restarts(env) -> None:
    m = env["model"]
    info0 = m.sandbox_info()
    t0 = time.monotonic()
    m.set_deadline(t0 + 1.0)
    try:
        with pytest.raises(ModuleTimeout, match="deadline"):
            m.predict_proba(_rows(tb.MODE_SLOW))
    finally:
        m.set_deadline(None)
    assert time.monotonic() - t0 < 15
    p = m.predict_proba(_rows())  # transparent restart + reload
    np.testing.assert_allclose(p, env["booster"].predict(_rows()), atol=1e-12)
    info = m.sandbox_info()
    assert info["timeouts"] == info0["timeouts"] + 1
    assert info["restarts"] == info0["restarts"] + 1


def test_expired_deadline_fails_fast(env) -> None:
    m = env["model"]
    m.set_deadline(time.monotonic() - 1)
    try:
        with pytest.raises(ModuleTimeout):
            m.predict_proba(_rows())
    finally:
        m.set_deadline(None)
    assert m.predict_proba(_rows()).shape == (50,)


@pytest.mark.parametrize("mode", [tb.MODE_EXIT, tb.MODE_SEGFAULT])
def test_worker_crash_raises_sandbox_error_with_log_tail(env, mode: float) -> None:
    m = env["model"]
    crashes = m.sandbox_info()["crashes"]
    with pytest.raises(SandboxError) as ei:
        m.predict_proba(_rows(mode))
    msg = str(ei.value)
    assert "crashed" in msg and "worker.log" in msg
    if mode == tb.MODE_EXIT:
        assert "exit code 7" in msg and "exiting on purpose" in msg
    else:
        assert "SIGSEGV" in msg
    assert m.sandbox_info()["crashes"] == crashes + 1
    assert m.predict_proba(_rows()).shape == (50,)  # restarted


def test_explicit_restart_and_close(env, tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    pol = SandboxPolicy(backend="bwrap", threads=1, memory_mb=4096)  # own temporary scratch
    with open_model(adapter, pol, declarations=inspect_adapter(adapter, pol)) as m:
        pid = m.worker_pid
        m.restart()
        assert m.worker_pid != pid
        assert m.predict_proba(_rows()).shape == (50,)
        scratch = m.scratch
    assert not scratch.exists()
    with pytest.raises(SandboxError, match="closed"):
        m.predict_proba(_rows())
