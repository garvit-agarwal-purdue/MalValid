"""What the double-click launchers rely on: ``serve --find-free-port``, ``--allow-reduced-isolation`` reaching
every run, the reduced-isolation banners, the verdict's ``isolation`` field, and SIGHUP (closing the terminal
window) stopping the server gracefully."""

from __future__ import annotations

import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")

from malvalid.cli import _config_overrides  # noqa: E402
from malvalid.runner import REDUCED_ISOLATION_NOTE, _add_isolation  # noqa: E402
from malvalid.sandbox import host as H  # noqa: E402
from malvalid.web import serve as web_serve  # noqa: E402
from malvalid.web.onboarding import demo_submission  # noqa: E402
from malvalid.web.settings import WebSettings  # noqa: E402
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    PORT,
    TOKEN,
    authed_client,
    fake_malvalid,
    fake_record,
    make_app,
    post_run,
    read_record,
    run_id_from,
)

pytestmark = pytest.mark.web


# ---- port picking ------------------------------------------------------------------------------------


def test_bind_first_free_skips_a_busy_port() -> None:
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    port = busy.getsockname()[1]
    try:
        with pytest.raises(OSError):
            web_serve.bind_socket("127.0.0.1", port)
        s = web_serve.bind_first_free("127.0.0.1", port, attempts=50)
        try:
            got = s.getsockname()[1]
            assert port < got < port + 50
        finally:
            s.close()
        with pytest.raises(OSError):
            web_serve.bind_first_free("127.0.0.1", port, attempts=1)
    finally:
        busy.close()


def test_bind_first_free_prefers_the_requested_port() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    s = web_serve.bind_first_free("127.0.0.1", port)
    try:
        assert s.getsockname()[1] == port
    finally:
        s.close()
    with pytest.raises(ValueError):
        web_serve.bind_first_free("127.0.0.1", 0)


# ---- the reduced-isolation option reaches every run ------------------------------------------------------


def _settings(tmp_path: Path, fake: Path, **kw: Any) -> WebSettings:
    return WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=PORT, cancel_grace_s=1.0,
                       command_prefix=(sys.executable, str(fake)), validate_timeout_s=30, **kw)


def _wait_for_run_argv(rec: Path) -> list[str]:
    for _ in range(200):
        runs = [r for r in read_record(rec) if r.get("cmd") == "run" or "run" in (r.get("argv") or [])[:1]]
        if runs:
            return list(runs[-1]["argv"])
        time.sleep(0.05)
    raise AssertionError(f"no run recorded: {read_record(rec)}")


@pytest.mark.parametrize("flag", [False, True])
def test_demo_run_gets_the_flag_only_when_the_server_has_it(tmp_path: Path, fake_malvalid: Path, fake_record: Path,
                                                             monkeypatch: pytest.MonkeyPatch, flag: bool) -> None:
    demo = demo_submission()
    if demo is None:
        pytest.skip("no examples/synthetic_demo in this checkout")
    monkeypatch.setattr(H, "os_sandbox_available", lambda: False)
    app = make_app(_settings(tmp_path, fake_malvalid, allow_reduced_isolation=flag))
    with authed_client(app) as c:
        html = c.get("/runs/new").text
        assert ("data-reduced-isolation" in html) is flag
        run_id_from(post_run(c, app.state.settings, {"mode": "path", **demo}))
        argv = _wait_for_run_argv(fake_record)
    assert ("--allow-reduced-isolation" in argv) is flag
    assert "--no-sandbox" not in argv
    assert app.state.settings.public_dict()["allow_reduced_isolation"] is flag


def test_no_banner_when_the_os_sandbox_works(tmp_path: Path, fake_malvalid: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(H, "os_sandbox_available", lambda: True)
    app = make_app(_settings(tmp_path, fake_malvalid, allow_reduced_isolation=True))
    with authed_client(app) as c:
        assert "data-reduced-isolation" not in c.get("/runs/new").text


def test_validate_config_carries_the_flag(tmp_path: Path, fake_malvalid: Path) -> None:
    import yaml

    from malvalid.web.forms import RunRequest

    s = _settings(tmp_path, fake_malvalid, allow_reduced_isolation=True)
    adapter = tmp_path / "a.py"
    adapter.write_text("x = 1\n")
    req = RunRequest(mode="upload", adapter=adapter, base_config=None, config_source="default")
    argv = req.validate_argv(s, tmp_path)
    cfg = Path(argv[argv.index("--config") + 1])
    assert yaml.safe_load(cfg.read_text())["runtime"]["allow_reduced_isolation"] is True
    assert "--allow-reduced-isolation" in req.run_argv(s, None)


def test_cli_override() -> None:
    assert _config_overrides(allow_reduced_isolation=True) == {"runtime": {"allow_reduced_isolation": True}}
    assert _config_overrides() == {}


# ---- the verdict's isolation field ----------------------------------------------------------------------


def test_verdict_isolation_field() -> None:
    v = {"summary": "Ready. Production-readiness score 90.0/100."}
    _add_isolation(v, "process_only")
    assert v["isolation"] == "process_only" and v["isolation_warning"] == REDUCED_ISOLATION_NOTE
    assert v["summary"].endswith(REDUCED_ISOLATION_NOTE) and "this platform has no OS sandbox" in v["summary"]
    full = {"summary": "Ready."}
    _add_isolation(full, "os_sandbox")
    assert full == {"summary": "Ready.", "isolation": "os_sandbox", "isolation_warning": None}
    unknown = {"summary": "Blocked."}
    _add_isolation(unknown, None)
    assert unknown["isolation"] is None and unknown["summary"] == "Blocked."


# ---- closing the terminal window ------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_sighup_stops_the_server_gracefully() -> None:
    calls: list[int] = []

    class FakeServer:
        def handle_exit(self, sig: int, frame: Any) -> None:
            calls.append(sig)

    old = signal.getsignal(signal.SIGHUP)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        web_serve._stop_on_hangup(FakeServer())
        os.kill(os.getpid(), signal.SIGHUP)
        for _ in range(50):
            if calls:
                break
            time.sleep(0.01)
        assert calls == [signal.SIGTERM]
        # an explicit choice (nohup's SIG_IGN) is respected
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        web_serve._stop_on_hangup(FakeServer())
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, old)


# ---- bubblewrap sees the interpreter by the path it is started with --------------------------------------


@pytest.mark.posix
def test_symlinked_interpreter_dirs_are_bound(tmp_path: Path) -> None:
    real = tmp_path / "python" / "cpython-3.11.16-x" / "bin"
    real.mkdir(parents=True)
    (real / "python3.11").write_text("")
    (tmp_path / "python" / "cpython-3.11-x").symlink_to(tmp_path / "python" / "cpython-3.11.16-x")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(tmp_path / "python" / "cpython-3.11-x" / "bin" / "python3.11")
    dirs = H._symlinked_dirs([str(venv_bin / "python")])
    assert str(tmp_path / "python" / "cpython-3.11-x") in dirs
    assert str(venv_bin) in dirs


def test_venv_home_directory_never_binds_its_parent(tmp_path: Path) -> None:
    home_bin = tmp_path / "home" / "bin"
    home_bin.mkdir(parents=True)
    dirs = H._symlinked_dirs([str(home_bin)])
    assert str(tmp_path / "home") not in dirs


def test_policy_cannot_opt_into_reduced_isolation(tmp_path: Path, fake_malvalid: Path) -> None:
    app = make_app(_settings(tmp_path, fake_malvalid))
    pol = b"runtime:\n  allow_reduced_isolation: true\n"
    with authed_client(app) as c:
        r = post_run(c, app.state.settings, {"mode": "upload"},
                     files=[("adapter_file", ("adapter.py", b"x = 1\n")), ("config_file", ("gate.yaml", pol))],
                     headers={"accept": "application/json"})
    assert r.status_code == 400 and "allow_reduced_isolation" in r.text
