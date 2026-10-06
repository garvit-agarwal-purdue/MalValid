"""Reduced-isolation fallback: explicit opt-in, the ``isolation`` field, and the cross-platform
(stdin/stdout, Windows-style) worker channel exercised on POSIX."""

from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

import malvalid.testing  # noqa: F401 - registers the toy_v1 schema
from malvalid.config import GateConfig
from malvalid.core import SandboxError
from malvalid.sandbox import host as H
from malvalid.sandbox.host import (
    REDUCED_ISOLATION_WARNING,
    SandboxedModel,
    SandboxPolicy,
    inspect_adapter,
    isolation_level,
    open_model,
    select_backend,
)
from malvalid.sandbox.protocol import ChannelClosed, ChannelTimeout, PipeChannel, pack_message
from malvalid.sandbox.worker import apply_rlimits, self_check
from tests.fixtures.adapters.builders import eval_rows, make_adapter_dir

NO_SANDBOX_PROBES = {
    "bwrap": {"available": False, "network_isolated": False, "detail": "bwrap not found on PATH"},
    "unshare": {"available": False, "network_isolated": False, "detail": "unshare not found on PATH"},
    "subprocess": {"available": True, "network_isolated": False, "detail": "plain subprocess"},
}


@pytest.fixture
def no_os_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(H, "probe_backends", lambda refresh=False: {k: dict(v) for k, v in NO_SANDBOX_PROBES.items()})


# ---- levels and selection ----------------------------------------------------------------------------


def test_isolation_levels() -> None:
    assert isolation_level("bwrap", True) == "os_sandbox"
    assert isolation_level("unshare", True) == "os_sandbox"
    assert isolation_level("bwrap", False) == "process_only"
    assert isolation_level("subprocess", False) == "process_only"
    assert isolation_level("in-process") == "none"
    assert isolation_level(None) == "none"


def test_auto_refuses_without_opt_in(no_os_sandbox: None) -> None:
    with pytest.raises(SandboxError) as ei:
        select_backend("auto")
    msg = str(ei.value)
    assert "--allow-reduced-isolation" in msg and "process_only" in msg and "bwrap not found" in msg


def test_auto_falls_back_with_opt_in(no_os_sandbox: None) -> None:
    backend, warnings = select_backend("auto", allow_reduced_isolation=True)
    assert backend == "subprocess"
    assert warnings[0] == REDUCED_ISOLATION_WARNING
    assert any("NETWORK NOT ISOLATED" in w for w in warnings)


def test_opt_in_never_downgrades_a_working_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = {k: dict(v) for k, v in NO_SANDBOX_PROBES.items()}
    probes["bwrap"].update(available=True, network_isolated=True)
    monkeypatch.setattr(H, "probe_backends", lambda refresh=False: probes)
    assert select_backend("auto", allow_reduced_isolation=True) == ("bwrap", [])
    assert H.os_sandbox_available() is True


def test_simulate_env_hides_os_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(H.SIMULATE_NO_OS_SANDBOX_ENV, "1")
    for b in ("bwrap", "unshare"):
        r = H._probe_one(b)
        assert r["available"] is False and H.SIMULATE_NO_OS_SANDBOX_ENV in r["detail"]


def test_non_linux_reports_backends_linux_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(H.SIMULATE_NO_OS_SANDBOX_ENV, raising=False)
    monkeypatch.setattr(H, "_LINUX", False)
    r = H._probe_one("bwrap")
    assert r["available"] is False and "Linux-only" in r["detail"]


def test_policy_reads_config_flag(tmp_path: Path) -> None:
    cfg = GateConfig.model_validate({"runtime": {"allow_reduced_isolation": True}})
    pol = SandboxPolicy.from_config(cfg, run_dir=tmp_path)
    assert pol.allow_reduced_isolation is True and pol.to_dict()["allow_reduced_isolation"] is True
    assert SandboxPolicy.from_config(GateConfig(), run_dir=tmp_path).allow_reduced_isolation is False


# ---- end to end through a real worker ----------------------------------------------------------------


def test_reduced_isolation_model_reports_process_only(tmp_path: Path, no_os_sandbox: None) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    pol = SandboxPolicy(threads=1, memory_mb=4096, allow_reduced_isolation=True)
    with open_model(adapter, pol, declarations=inspect_adapter(adapter, pol)) as m:
        assert isinstance(m, SandboxedModel) and m.backend == "subprocess"
        booster = lgb.Booster(model_file=str(adapter.parent / "model.txt"))
        np.testing.assert_allclose(m.predict_proba(eval_rows(20)), booster.predict(eval_rows(20)), atol=1e-12)
        info = m.sandbox_info()
    assert info["isolation"] == "process_only"
    assert info["warnings"][0] == REDUCED_ISOLATION_WARNING
    assert info["pickle_guards"], "pickle refusal must stay on in reduced isolation"
    if os.name == "nt":  # no setrlimit there: the host puts the worker in a Job Object with a memory limit
        assert info["rlimits"]["job"]["process_memory_mb"] == 4096
    else:
        assert info["rlimits"]["as"] == 4096 << 20


def test_reduced_isolation_refused_without_flag(tmp_path: Path, no_os_sandbox: None) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    with pytest.raises(SandboxError, match="allow-reduced-isolation"):
        inspect_adapter(adapter, SandboxPolicy(threads=1, memory_mb=4096))


def test_stdio_channel_worker_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Windows transport (worker stdin/stdout + PipeChannel), run on POSIX."""
    monkeypatch.setenv(H.IPC_ENV, "stdio")
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    pol = SandboxPolicy(backend="subprocess", threads=1, memory_mb=4096)
    with open_model(adapter, pol, declarations=inspect_adapter(adapter, pol)) as m:
        assert isinstance(m._worker.channel, PipeChannel)  # type: ignore[union-attr]
        booster = lgb.Booster(model_file=str(adapter.parent / "model.txt"))
        X = eval_rows(50)
        np.testing.assert_allclose(m.predict_proba(X), booster.predict(X), atol=1e-12)
        assert m.tree_ensemble() is not None
        info = m.sandbox_info()
        assert info["ipc"] == "stdio" and info["isolation"] == "process_only"
        m.restart()
        np.testing.assert_allclose(m.predict_proba(X[:5]), booster.predict(X[:5]), atol=1e-12)


# ---- PipeChannel unit behaviour ---------------------------------------------------------------------


class _SlowReader(io.RawIOBase):
    def __init__(self, data: bytes, delay: float = 0.0, hang: bool = False) -> None:
        self._data = bytearray(data)
        self._delay = delay
        self._hang = hang
        self.closed_evt = threading.Event()

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:  # type: ignore[override]
        if self._delay:
            time.sleep(self._delay)
        if not self._data:
            if self._hang:
                self.closed_evt.wait(5)
            return b""
        out = bytes(self._data[: min(n, 7)])  # tiny chunks: frames arrive in pieces
        del self._data[: len(out)]
        return out


def test_pipe_channel_reassembles_frames_and_detects_eof() -> None:
    frame = b"".join(pack_message({"ok": True, "id": 1}, {"y": b"x" * 100}))
    ch = PipeChannel(_SlowReader(frame + frame), io.BytesIO(), tick=0.05)
    for _ in range(2):
        msg = ch.recv(deadline=time.monotonic() + 5)
        assert msg.ok and msg.payloads["y"] == b"x" * 100
    with pytest.raises(ChannelClosed):
        ch.recv(deadline=time.monotonic() + 5)
    ch.close()


def test_pipe_channel_deadline_and_dead_worker() -> None:
    r = _SlowReader(b"", hang=True)
    ch = PipeChannel(r, io.BytesIO(), tick=0.05)
    with pytest.raises(ChannelTimeout):
        ch.recv(deadline=time.monotonic() + 0.3)
    ch2 = PipeChannel(_SlowReader(b"", hang=True), io.BytesIO(), alive=lambda: False, tick=0.05)
    with pytest.raises(ChannelClosed):
        ch2.recv(deadline=time.monotonic() + 5)
    r.closed_evt.set()


def test_pipe_channel_bounded_buffer_still_delivers_large_payload() -> None:
    big = b"z" * (3 << 20)
    frame = b"".join(pack_message({"ok": True, "id": 1}, {"t": big}))
    ch = PipeChannel(io.BufferedReader(io.BytesIO(frame)), io.BytesIO(), max_buffer=1 << 20, tick=0.05)
    msg = ch.recv(deadline=time.monotonic() + 10)
    assert msg.payloads["t"] == big


def test_pipe_channel_send_writes_frames() -> None:
    out = io.BytesIO()
    ch = PipeChannel(_SlowReader(b"", hang=True), out, tick=0.05)
    ch.send({"op": "ping", "id": 3}, deadline=time.monotonic() + 5)
    assert out.getvalue().startswith(b"MGW1")


# ---- worker helpers degrade on platforms without resource/statvfs/getuid -------------------------------


def test_worker_helpers_without_posix_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):  # type: ignore[no-untyped-def]
        if name == "resource":
            raise ImportError("no resource module")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "resource", raising=False)
    assert apply_rlimits(1024, 64) == {"as": None, "core": None, "nofile": None, "supported": False}
    monkeypatch.setattr(builtins, "__import__", real_import)
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.delattr(os, "statvfs", raising=False)
    info = self_check(None)
    assert info["uid"] is None and info["readonly_root"] is None


def test_kill_process_without_killpg(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import sys

    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    monkeypatch.delattr(os, "killpg", raising=False)
    H._kill_process(p)
    assert p.poll() is not None
