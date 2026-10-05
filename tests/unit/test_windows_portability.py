"""Windows branches of the web job runner, pid probe, atomic writers and link helpers, exercised on
Linux by monkeypatching the platform flags (``malvalid._platform.WINDOWS``, ``malvalid.web.jobs._WINDOWS``)
and faking ``taskkill`` / kernel32 / ``os.symlink``. Nothing here starts a real Windows-only API."""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from malvalid import _platform
from malvalid.web import jobs as jobs_mod
from malvalid.web import store as store_mod
from tests.unit.test_file_safety import EVAL_PICKLE, _never_unpickle  # noqa: F401 - autouse fixture

# Simulates Windows on a POSIX host (e.g. deletes signal.SIGKILL), so it cannot run on real Windows.
pytestmark = pytest.mark.posix

CTRL_BREAK = 1  # signal.CTRL_BREAK_EVENT on Windows


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend to be Windows: platform flags on, CTRL_BREAK_EVENT present, SIGKILL absent, and any
    POSIX process-group call fails the test."""
    monkeypatch.setattr(_platform, "WINDOWS", True)
    monkeypatch.setattr(jobs_mod, "_WINDOWS", True)
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", CTRL_BREAK, raising=False)
    monkeypatch.delattr(signal, "SIGKILL")

    def no_posix(*a: Any, **k: Any) -> Any:
        raise AssertionError("POSIX signal API used on the Windows path")

    for name in ("kill", "killpg", "getpgid"):
        monkeypatch.setattr(os, name, no_posix, raising=False)


class FakeRun:
    """Stands in for ``subprocess.run`` and records taskkill invocations."""

    def __init__(self, returncode: int = 0, exc: BaseException | None = None):
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.returncode = returncode
        self.exc = exc

    def __call__(self, argv: list[str], **kw: Any) -> Any:
        self.calls.append((list(argv), kw))
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(returncode=self.returncode)


class FakeProc:
    """Minimal ``Popen`` look-alike."""

    def __init__(self, pid: int = 4242, *, running: bool = True):
        self.pid = pid
        self.returncode: int | None = None if running else 0
        self.signals: list[int] = []
        self.killed = 0

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def kill(self) -> None:
        self.killed += 1
        self.returncode = 1

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout or 0)
        return self.returncode


def _manager(tmp_path: Path, **settings: Any) -> jobs_mod.JobManager:
    s = SimpleNamespace(**{"cancel_grace_s": 0.0, "extra_env": {}, "max_concurrent": 1, "launch_dir": tmp_path,
                           **settings})
    return jobs_mod.JobManager(s, None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------
# process start / stop (web/jobs.py)
# --------------------------------------------------------------------------------------------------


def test_popen_kwargs_posix_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs_mod, "_WINDOWS", False)
    assert jobs_mod._popen_group_kwargs() == {"start_new_session": True}


def test_popen_kwargs_windows_new_process_group(windows: None) -> None:
    kw = jobs_mod._popen_group_kwargs()
    assert "start_new_session" not in kw
    assert kw["creationflags"] == getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)


def test_posix_stop_and_kill_use_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs_mod, "_WINDOWS", False)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(jobs_mod, "_signal_group", lambda pid, sig: sent.append((pid, sig)) or True)
    run = FakeRun()
    monkeypatch.setattr(jobs_mod.subprocess, "run", run)
    assert jobs_mod._stop_group(7) and jobs_mod._kill_tree(7)
    assert sent == [(7, signal.SIGTERM), (7, signal.SIGKILL)] and not run.calls


def test_windows_graceful_stop_only_for_owned_popen(windows: None) -> None:
    proc = FakeProc()
    assert jobs_mod._stop_group(proc.pid, proc) is True
    assert proc.signals == [CTRL_BREAK]
    # a bare pid (adopted from an earlier server) is never sent a console control event
    assert jobs_mod._stop_group(1234) is False
    # nor a process that already exited
    done = FakeProc(running=False)
    assert jobs_mod._stop_group(done.pid, done) is False and done.signals == []


def test_windows_graceful_stop_refused(windows: None) -> None:
    proc = FakeProc()

    def refuse(sig: int) -> None:
        raise OSError(errno.EINVAL, "no console")

    proc.send_signal = refuse  # type: ignore[method-assign]
    assert jobs_mod._stop_group(proc.pid, proc) is False


def test_windows_kill_tree_runs_taskkill(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun(returncode=0)
    monkeypatch.setattr(jobs_mod.subprocess, "run", run)
    proc = FakeProc(pid=555)
    assert jobs_mod._kill_tree(proc.pid, proc) is True
    (argv, kw), = run.calls
    assert Path(argv[0].replace("\\", "/")).name.lower() == "taskkill.exe"
    assert argv[1:] == ["/F", "/T", "/PID", "555"]
    assert kw["timeout"] == jobs_mod.TASKKILL_TIMEOUT_S and kw["stdin"] is subprocess.DEVNULL
    assert proc.killed == 1  # still "running" in the fake: Popen.kill() as a backstop


@pytest.mark.parametrize("exc", [OSError("taskkill missing"), subprocess.TimeoutExpired("taskkill", 15)])
def test_windows_kill_tree_falls_back_to_proc_kill(windows: None, monkeypatch: pytest.MonkeyPatch,
                                                   exc: BaseException) -> None:
    run = FakeRun(exc=exc)
    monkeypatch.setattr(jobs_mod.subprocess, "run", run)
    proc = FakeProc()
    assert jobs_mod._kill_tree(proc.pid, proc) is True
    assert len(run.calls) == 1 and proc.killed == 1


def test_windows_kill_tree_skips_exited_and_foreign_pids(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun()
    monkeypatch.setattr(jobs_mod.subprocess, "run", run)
    assert jobs_mod._kill_tree(9, FakeProc(running=False)) is False
    seen: list[Any] = []
    monkeypatch.setattr(jobs_mod, "pid_alive", lambda pid, expect=(): seen.append((pid, tuple(expect))) or False)
    assert jobs_mod._kill_tree(9) is False  # adopted pid no longer a malvalid process: never killed
    assert seen == [(9, ("malvalid",))] and not run.calls
    monkeypatch.setattr(jobs_mod, "pid_alive", lambda pid, expect=(): True)
    assert jobs_mod._kill_tree(9) is True and run.calls[0][0][-1] == "9"


def test_signal_group_without_killpg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "killpg")
    assert jobs_mod._signal_group(os.getpid(), 0) is False


def _wait_for(pred: Any, timeout: float = 5.0) -> None:
    t0 = time.monotonic()
    while not pred():
        assert time.monotonic() - t0 < timeout, "timed out"
        time.sleep(0.01)


def test_windows_cancel_breaks_then_kills_tree(windows: None, monkeypatch: pytest.MonkeyPatch,
                                               tmp_path: Path) -> None:
    run = FakeRun()
    monkeypatch.setattr(jobs_mod.subprocess, "run", run)
    jm = _manager(tmp_path, cancel_grace_s=0.05)
    proc = FakeProc(pid=777)
    jm._terminate("r1", proc.pid, proc)  # type: ignore[arg-type]
    assert proc.signals == [CTRL_BREAK]
    _wait_for(lambda: run.calls)
    assert run.calls[0][0][1:] == ["/F", "/T", "/PID", "777"]


def test_windows_cancel_adopted_run_kills_immediately(windows: None, monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:
    run = FakeRun()
    monkeypatch.setattr(jobs_mod.subprocess, "run", run)
    monkeypatch.setattr(jobs_mod, "pid_alive", lambda pid, expect=(): True)
    jm = _manager(tmp_path, cancel_grace_s=3600.0)  # the grace period is skipped: no graceful stop exists
    jm._terminate("r1", 888, None)
    _wait_for(lambda: run.calls, timeout=2.0)
    assert run.calls[0][0][-1] == "888"


def test_windows_shutdown_stops_owned_and_adopted(windows: None, monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    run = FakeRun()
    monkeypatch.setattr(jobs_mod.subprocess, "run", run)
    monkeypatch.setattr(jobs_mod, "pid_alive", lambda pid, expect=(): True)
    jm = _manager(tmp_path)
    proc = FakeProc(pid=101)
    jm._started = True
    jm._procs["a"] = proc  # type: ignore[assignment]
    jm._adopted["b"] = 202
    jm.shutdown(grace_s=0.01)
    assert proc.signals == [CTRL_BREAK]
    assert sorted(c[0][-1] for c in run.calls) == ["101", "202"]


def test_windows_validation_timeout_kills_tree(windows: None, monkeypatch: pytest.MonkeyPatch,
                                               tmp_path: Path) -> None:
    run = FakeRun()
    monkeypatch.setattr(jobs_mod.subprocess, "run", run)
    made: list[Any] = []

    class TimeoutPopen(FakeProc):
        def __init__(self, argv: list[str], **kw: Any):
            super().__init__(pid=303)
            self.kw = kw
            self.n = 0
            made.append(self)

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            self.n += 1
            if self.n == 1:
                raise subprocess.TimeoutExpired("malvalid", timeout or 0)
            assert timeout == jobs_mod.KILL_DRAIN_TIMEOUT_S  # never an unbounded wait on Windows
            self.returncode = 1
            return b'{"ok": false, "checks": []}', b""

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", TimeoutPopen)
    jm = _manager(tmp_path)
    res = jm.validate(["malvalid", "validate-adapter", "--json", "x.py"], cwd=tmp_path, timeout_s=0.01)
    assert res["timed_out"] and not res["ok"] and "was stopped" in res["error"]
    p, = made
    assert "start_new_session" not in p.kw and p.kw["creationflags"]
    assert run.calls and run.calls[0][0][-1] == "303"


def test_windows_validation_straggler_holding_pipes(windows: None, monkeypatch: pytest.MonkeyPatch,
                                                    tmp_path: Path) -> None:
    monkeypatch.setattr(jobs_mod.subprocess, "run", FakeRun())

    class StuckPopen(FakeProc):
        def __init__(self, argv: list[str], **kw: Any):
            super().__init__(pid=404)

        def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
            raise subprocess.TimeoutExpired("malvalid", timeout or 0)

        def wait(self, timeout: float | None = None) -> int:
            return 1

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", StuckPopen)
    raw = _manager(tmp_path)._json_subprocess(["malvalid"], cwd=tmp_path, timeout=0.01)
    assert raw["timed_out"] and raw["result"] is None and raw["stderr"] == ""  # returned, did not hang


# --------------------------------------------------------------------------------------------------
# pid_alive (web/store.py) with a fake kernel32
# --------------------------------------------------------------------------------------------------


class FakeKernel32:
    def __init__(self, *, handle: int = 0x1234, exit_code: int = _platform.STILL_ACTIVE,
                 last_error: int = 0, image: str | None = r"C:\Python311\python.exe"):
        self.handle, self.exit_code, self.last_error, self.image = handle, exit_code, last_error, image
        self.opened: list[tuple[int, Any, int]] = []
        self.closed: list[Any] = []

    def OpenProcess(self, access: int, inherit: Any, pid: int) -> int:  # noqa: N802
        self.opened.append((access, inherit, pid))
        return self.handle

    def GetExitCodeProcess(self, h: Any, ref: Any) -> int:  # noqa: N802
        assert h == self.handle
        ref._obj.value = self.exit_code
        return 1

    def QueryFullProcessImageNameW(self, h: Any, flags: int, buf: Any, size_ref: Any) -> int:  # noqa: N802
        if self.image is None:
            return 0
        buf.value = self.image
        size_ref._obj.value = len(self.image)
        return 1

    def CloseHandle(self, h: Any) -> int:  # noqa: N802
        self.closed.append(h)
        return 1


@pytest.fixture
def fake_k32(windows: None, monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(**kw: Any) -> FakeKernel32:
        k = FakeKernel32(**kw)
        monkeypatch.setattr(_platform, "_kernel32", lambda: k)
        monkeypatch.setattr(_platform, "_last_error", lambda: k.last_error)
        return k

    return install


def test_pid_alive_windows_running(fake_k32: Any) -> None:
    k = fake_k32()
    assert store_mod.pid_alive(4321) is True
    assert k.opened == [(_platform.PROCESS_QUERY_LIMITED_INFORMATION, False, 4321)]
    assert k.closed == [k.handle]


def test_pid_alive_windows_exited(fake_k32: Any) -> None:
    k = fake_k32(exit_code=0)
    assert store_mod.pid_alive(4321) is False and k.closed == [k.handle]


@pytest.mark.parametrize(("err", "alive"), [(_platform.ERROR_ACCESS_DENIED, True), (87, False)])
def test_pid_alive_windows_open_fails(fake_k32: Any, err: int, alive: bool) -> None:
    k = fake_k32(handle=0, last_error=err)
    assert store_mod.pid_alive(4321) is alive and k.closed == []


@pytest.mark.parametrize(("image", "alive"), [
    (r"C:\proj\.venv\Scripts\python.exe", True),
    (r"C:\Users\me\AppData\Local\Programs\Python\Python311\pythonw.exe", True),
    (r"C:\proj\.venv\Scripts\malvalid.exe", True),
    (r"C:\Windows\System32\notepad.exe", False),  # pid reused by something else
    (None, True),  # image unknown: cannot tell, keep the conservative answer
])
def test_pid_alive_windows_expect_checks_image(fake_k32: Any, image: str | None, alive: bool) -> None:
    k = fake_k32(image=image)
    assert store_mod.pid_alive(4321, expect=("malvalid", r"C:\runs\x")) is alive
    assert store_mod.pid_alive(4321) is True  # no expectation: liveness only
    assert len(k.closed) == len(k.opened) == 2


def test_pid_alive_windows_rejects_bad_pids_without_probing(fake_k32: Any) -> None:
    k = fake_k32()
    for bad in (None, 0, -1, 2**31, "x", True):
        assert store_mod.pid_alive(bad) is False
    assert k.opened == []


def test_pid_alive_windows_probe_error_is_dead(windows: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Any:
        raise OSError("kernel32 unavailable")

    monkeypatch.setattr(_platform, "_kernel32", broken)
    assert store_mod.pid_alive(4321) is False


def test_kernel32_prototypes_are_pointer_sized() -> None:
    from ctypes import wintypes

    assert ctypes.sizeof(wintypes.HANDLE) == ctypes.sizeof(ctypes.c_void_p)


# --------------------------------------------------------------------------------------------------
# os.replace retries (store.write_json_atomic, report.json, progress.json)
# --------------------------------------------------------------------------------------------------


class FlakyReplace:
    def __init__(self, fail_times: int):
        self.fail_times, self.calls = fail_times, 0
        self.real = os.replace

    def __call__(self, src: Any, dst: Any) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise PermissionError(errno.EACCES, "The process cannot access the file", str(dst))
        self.real(src, dst)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(_platform.time, "sleep", slept.append)
    return slept


def test_replace_file_retries_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_sleep: list[float]) -> None:
    monkeypatch.setattr(_platform, "WINDOWS", True)
    flaky = FlakyReplace(3)
    monkeypatch.setattr(os, "replace", flaky)
    (tmp_path / "a").write_text("new")
    (tmp_path / "b").write_text("old")
    _platform.replace_file(tmp_path / "a", tmp_path / "b")
    assert flaky.calls == 4 and (tmp_path / "b").read_text() == "new"
    assert no_sleep == [_platform.REPLACE_RETRY_S] * 3


def test_replace_file_gives_up_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_sleep: list[float]) -> None:
    monkeypatch.setattr(_platform, "WINDOWS", True)
    flaky = FlakyReplace(10**6)
    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(PermissionError):
        _platform.replace_file(tmp_path / "a", tmp_path / "b")
    assert flaky.calls == _platform.REPLACE_ATTEMPTS and len(no_sleep) == _platform.REPLACE_ATTEMPTS - 1


def test_replace_file_no_retry_on_posix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_sleep: list[float]) -> None:
    monkeypatch.setattr(_platform, "WINDOWS", False)
    flaky = FlakyReplace(1)
    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(PermissionError):
        _platform.replace_file(tmp_path / "a", tmp_path / "b")
    assert flaky.calls == 1 and no_sleep == []


def test_atomic_writers_retry_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_sleep: list[float]) -> None:
    import datetime as dt

    from malvalid import runner
    from malvalid.report.json_writer import write_report_json

    monkeypatch.setattr(_platform, "WINDOWS", True)
    flaky = FlakyReplace(2)
    monkeypatch.setattr(os, "replace", flaky)
    store_mod.write_json_atomic(tmp_path / "job.json", {"a": 1})
    assert flaky.calls == 3 and '"a": 1' in (tmp_path / "job.json").read_text()

    flaky.calls = 0
    write_report_json({"schema_version": "malvalid-report/1", "x": 1}, tmp_path / "report.json")
    assert flaky.calls == 3 and (tmp_path / "report.json").is_file()

    flaky.calls = 0
    progress = runner._Progress(tmp_path / "progress.json", "rid", dt.datetime.now(dt.timezone.utc))
    flaky.calls = 0
    (tmp_path / "progress.json").unlink(missing_ok=True)
    progress._write()
    assert flaky.calls == 3 and (tmp_path / "progress.json").is_file()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


# --------------------------------------------------------------------------------------------------
# symlink fallbacks (submission folder, M0 modelscan aliases)
# --------------------------------------------------------------------------------------------------


def _no_symlink(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    calls: list[Any] = []

    def refuse(src: Any, dst: Any, *a: Any, **k: Any) -> None:
        calls.append((src, dst))
        raise OSError(22, "A required privilege is not held by the client", str(dst), _platform.ERROR_PRIVILEGE_NOT_HELD)

    monkeypatch.setattr(os, "symlink", refuse)
    return calls


def test_link_or_copy_prefers_symlink(tmp_path: Path) -> None:
    (tmp_path / "src").write_bytes(b"abc")
    assert _platform.link_or_copy(tmp_path / "src", tmp_path / "dst") == "symlink"
    assert (tmp_path / "dst").is_symlink()


def test_link_or_copy_hardlink_then_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.write_bytes(b"model bytes")
    calls = _no_symlink(monkeypatch)
    assert _platform.link_or_copy(src, tmp_path / "hl") == "hardlink"
    assert (tmp_path / "hl").read_bytes() == b"model bytes" and os.stat(tmp_path / "hl").st_nlink == 2
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EXDEV, "cross-device")))
    assert _platform.link_or_copy(src, tmp_path / "cp") == "copy"
    assert (tmp_path / "cp").read_bytes() == b"model bytes" and not (tmp_path / "cp").is_symlink()
    assert len(calls) == 2


def test_link_or_copy_never_overwrites(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "src").write_bytes(b"new")
    (tmp_path / "dst").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        _platform.link_or_copy(tmp_path / "src", tmp_path / "dst")
    _no_symlink(monkeypatch)
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EPERM, "no")))
    with pytest.raises(FileExistsError):
        _platform.link_or_copy(tmp_path / "src", tmp_path / "dst")
    assert (tmp_path / "dst").read_bytes() == b"old"


def test_link_or_copy_all_fail_raises_symlink_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_symlink(monkeypatch)
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EPERM, "no")))
    with pytest.raises(OSError) as ei:
        _platform.link_or_copy(tmp_path / "missing", tmp_path / "dst")
    assert getattr(ei.value, "winerror", None) == _platform.ERROR_PRIVILEGE_NOT_HELD or ei.value.errno == 22
    assert not (tmp_path / "dst").exists()


@pytest.mark.parametrize("hardlink", [True, False])
def test_model_submission_without_symlinks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hardlink: bool) -> None:
    from malvalid.adapters.spec import load_spec
    from malvalid.loaders.base import load_model
    from malvalid.submission import prepare_model_submission
    from tests.unit import model_factories as mf

    model = tmp_path / "models" / "v2.txt"
    model.parent.mkdir()
    mf.save_lgbm(model, 2381)
    hashes = tmp_path / "models" / "hashes.txt"
    hashes.write_text("a" * 64 + "\n")
    _no_symlink(monkeypatch)
    if not hardlink:
        monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EXDEV, "cross-device")))
    r = prepare_model_submission(model, tmp_path / "sub", threshold=0.8, training_hashes=hashes)
    s = load_spec(r.spec_path)
    linked = tmp_path / "sub" / s.model_file
    assert not linked.is_symlink() and linked.read_bytes() == model.read_bytes()
    assert (os.stat(linked).st_ino == os.stat(model).st_ino) is hardlink
    assert (tmp_path / "sub" / str(s.training_hashes_file)).read_text() == hashes.read_text()
    # the worker loads the model by basename from the spec's folder: a copy works the same as a link
    assert load_model(linked, s.model_kind) is not None
    # re-preparing into the same folder replaces the earlier link/copy
    r2 = prepare_model_submission(model, tmp_path / "sub", threshold=0.7)
    assert load_spec(r2.spec_path).operating_threshold == 0.7


def test_m0_modelscan_alias_without_symlinks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from malvalid.modules.file_safety import scan_artifacts

    f = tmp_path / "weights.model"  # pickle content under an extension modelscan does not know
    f.write_bytes(EVAL_PICKLE)
    calls = _no_symlink(monkeypatch)
    monkeypatch.setattr(os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EXDEV, "cross-device")))
    rep = scan_artifacts([f], allow_pickle=True)
    a = rep.artifacts[0]
    assert a.is_pickle and a.max_severity() == "CRITICAL" and rep.abort
    assert calls and a.scanned_as == "pickle (by content)"  # the alias was copied, not symlinked
    assert any("modelscan" in x.sources for x in a.findings)  # ...and modelscan still scanned it
    assert f.read_bytes() == EVAL_PICKLE


def test_threads_left_clean() -> None:
    # the cancel tests start daemon Timer threads; make sure none is stuck forever
    _wait_for(lambda: not [t for t in threading.enumerate() if isinstance(t, threading.Timer) and t.is_alive()],
              timeout=10.0)
