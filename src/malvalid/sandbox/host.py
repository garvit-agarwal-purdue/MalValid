"""Host side of the malvalid sandbox: isolation backends, the worker process, and the model proxy.

The submitted model is untrusted input. It is imported, loaded and queried only inside a separate
worker process (:mod:`malvalid.sandbox.worker`) that runs

* under **bubblewrap** (preferred): read-only root, private ``/tmp``, ``/run`` and ``$HOME`` hidden,
  one writable scratch directory, no network (own network namespace), own PID/IPC/UTS namespaces,
  no capabilities, killed with its parent;
* else under ``unshare -rn`` (own user + network + PID namespaces; the file system is *not* restricted);
* else, **only when explicitly allowed** (``runtime.allow_reduced_isolation`` / ``--allow-reduced-isolation``,
  or ``sandbox_backend: subprocess``), as a plain subprocess: *reduced isolation* — a separate process with
  the no-unpickle IPC, pickle refusal and resource limits, but no network or file-system isolation. This is
  what Windows, macOS and Linux hosts without user namespaces get; every report says so
  (``isolation: process_only``). Without the opt-in, ``auto`` refuses to run instead of downgrading silently.

Isolation levels recorded in ``report.json`` (:func:`isolation_level`): ``os_sandbox`` (bwrap/unshare with
the network isolated), ``process_only`` (separate worker process only) and ``none`` (in-process debug mode).

In every backend the worker gets a scrubbed environment, ``RLIMIT_AS``/``RLIMIT_NOFILE``/
``RLIMIT_CORE=0`` (on Windows: a Job Object with a per-process memory limit, no child processes and
kill-on-close instead), bounded thread pools, and the pickle guards of
:func:`malvalid.sandbox.worker.install_pickle_guards`. Host and worker talk over two dedicated pipe
fds (on Windows: the worker's stdin/stdout) with :mod:`malvalid.sandbox.protocol`; nothing the worker sends is ever unpickled, and every
response is validated here before it reaches a test module.

``SandboxPolicy(enabled=False)`` runs the adapter in-process for debugging trusted models only.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import weakref
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from malvalid.context import ModelDeclarations
from malvalid.core import (
    AdapterContractError,
    AdapterError,
    FeaturizeUnavailable,
    ModuleTimeout,
    PickleRefused,
    SandboxError,
    UnsupportedTreeError,
)
from malvalid.sandbox.worker import PREIMPORTS
from malvalid.sandbox.protocol import (
    Channel,
    ChannelClosed,
    PipeChannel,
    ChannelTimeout,
    ProtocolError,
    decode_array,
    encode_array,
)

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.config import GateConfig
    from malvalid.loaders.trees import TreeEnsemble

log = logging.getLogger("malvalid.sandbox")

BACKENDS = ("bwrap", "unshare", "subprocess")
_ISOLATING = ("bwrap", "unshare")
_WINDOWS = os.name == "nt"
_LINUX = sys.platform.startswith("linux")
#: Isolation levels recorded in report.json (``sandbox.isolation`` and ``verdict.isolation``).
ISOLATION_LEVELS = ("os_sandbox", "process_only", "none")
#: Testing hook: treat bwrap/unshare as unavailable, as on a platform without an OS sandbox. It can only
#: *remove* isolation backends, and runs still refuse to start without the explicit reduced-isolation opt-in.
SIMULATE_NO_OS_SANDBOX_ENV = "MALVALID_SIMULATE_NO_OS_SANDBOX"
#: ``stdio`` forces the stdin/stdout worker channel (the Windows transport) on POSIX, for testing.
IPC_ENV = "MALVALID_SANDBOX_IPC"
REDUCED_ISOLATION_WARNING = (
    "REDUCED ISOLATION: this platform has no OS sandbox. The model ran in a separate worker process (no-unpickle "
    "IPC, pickle refusal, resource limits where the OS supports them) but with NO network or file-system "
    "isolation. Only evaluate models whose origin you trust; for untrusted models use Linux with bubblewrap "
    "(or WSL2 / a Linux VM)."
)
WORKER_LOG = "worker.log"
_MAX_TREE_BYTES = 4 << 30
_MAX_FEATURIZE_INPUT = 1 << 30
_MAX_HEADER_ONLY = 1 << 20


# --------------------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------------------


@dataclass
class SandboxPolicy:
    """How the submitted model is isolated. Build it with :meth:`from_config` in a run."""

    enabled: bool = True  # False => trusted in-process mode (debug only; loud warning)
    backend: str = "auto"  # auto | bwrap | unshare | subprocess
    allow_pickle: bool = False
    memory_mb: int = 32768
    threads: int = 8
    chunk_rows: int = 20000
    ro_paths: tuple[Path, ...] = ()  # extra read-only paths (adapter dir, model files)
    scratch_dir: Path | None = None  # writable worker dir (runner passes run_dir/private/sandbox)
    startup_timeout_s: float = 300.0
    nofile: int = 4096
    #: ``auto`` may fall back to a plain worker process (``isolation: process_only``) when no OS sandbox
    #: works here. Off by default: without it such a run fails with an actionable error.
    allow_reduced_isolation: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("auto",) + BACKENDS:
            raise ValueError(f"unknown sandbox backend {self.backend!r}; use auto, bwrap, unshare or subprocess")
        self.ro_paths = tuple(Path(p) for p in self.ro_paths)
        if self.scratch_dir is not None:
            self.scratch_dir = Path(self.scratch_dir)
        if self.chunk_rows < 1:
            raise ValueError("chunk_rows must be >= 1")

    @classmethod
    def from_config(
        cls,
        cfg: "GateConfig",
        *,
        run_dir: Path,
        allow_pickle: bool | None = None,
        extra_ro: Sequence[Path] = (),
    ) -> "SandboxPolicy":
        rt = cfg.runtime
        return cls(
            enabled=bool(rt.sandbox),
            backend=str(rt.sandbox_backend),
            allow_pickle=bool(rt.allow_pickle if allow_pickle is None else allow_pickle),
            memory_mb=int(rt.sandbox_memory_mb),
            threads=int(rt.threads),
            chunk_rows=int(rt.chunk_rows),
            ro_paths=tuple(Path(p).expanduser().resolve() for p in extra_ro),
            scratch_dir=Path(run_dir) / "private" / "sandbox",
            allow_reduced_isolation=bool(getattr(rt, "allow_reduced_isolation", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "allow_pickle": self.allow_pickle,
            "memory_mb": self.memory_mb,
            "threads": self.threads,
            "chunk_rows": self.chunk_rows,
            "ro_paths": [str(p) for p in self.ro_paths],
            "scratch_dir": None if self.scratch_dir is None else str(self.scratch_dir),
            "startup_timeout_s": self.startup_timeout_s,
            "allow_reduced_isolation": self.allow_reduced_isolation,
        }


# --------------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------------


def _home_dirs() -> list[Path]:
    out: list[Path] = []
    cands = [os.environ.get("HOME")]
    with contextlib.suppress(Exception):
        import pwd

        cands.append(pwd.getpwuid(os.getuid()).pw_dir)
    if _WINDOWS:
        with contextlib.suppress(Exception):
            cands.append(str(Path.home()))
    for c in cands:
        if c:
            p = Path(c).resolve()
            if p != Path("/") and p.is_dir() and p not in out:
                out.append(p)
    return out


def _masked_roots() -> list[Path]:
    """Directories replaced by empty tmpfs inside the bwrap sandbox."""
    roots = [Path("/tmp"), Path("/dev/shm")]
    if Path("/run").is_dir():
        roots.append(Path("/run"))
    return roots + _home_dirs()


def _is_under(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def _package_root() -> Path:
    import malvalid

    return Path(malvalid.__file__).resolve().parent.parent


def _operator_pythonpath() -> list[str]:
    """Existing ``$PYTHONPATH`` entries of the harness process (the worker runs ``python -I``)."""
    out: list[str] = []
    for c in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        with contextlib.suppress(OSError):
            if c and Path(c).exists():
                r = str(Path(c).resolve())
                if r not in out:
                    out.append(r)
    return out


def _python_paths() -> list[Path]:
    """Paths the worker's interpreter needs to read (re-exposed if they sit under a masked dir)."""
    cands = [sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.path.realpath(sys.executable)),
             os.path.dirname(sys.executable), str(_package_root())]
    cands += [p for p in sys.path if p]
    cands += _operator_pythonpath()
    out: list[Path] = []
    for c in cands:
        with contextlib.suppress(OSError):
            p = Path(c).resolve()
            if p.exists() and p not in out:
                out.append(p)
    # The interpreter is started by its unresolved path (a venv's bin/python is a symlink, e.g. into uv's
    # ``cpython-3.11-<platform>`` link to ``cpython-3.11.16-<platform>``, and pyvenv.cfg's ``home`` may
    # name the link): bind every symlinked directory on those paths as spelled, or they do not exist
    # inside the sandbox.
    for d in _symlinked_dirs([sys.executable, _venv_home()]):
        q = Path(d)
        if q.exists() and q not in out:
            out.append(q)
    return out


def _venv_home() -> str | None:
    """The ``home`` of the running venv's ``pyvenv.cfg`` (where its base interpreter lives), if any."""
    with contextlib.suppress(OSError):
        cfg = Path(sys.prefix) / "pyvenv.cfg"
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().lower() == "home" and value.strip():
                return value.strip()
    return None


def _symlinked_dirs(paths: Sequence[str | None]) -> list[str]:
    """Every directory along ``paths`` and their symlink chains that is itself a symlink (unresolved,
    absolute), plus the parent directory of each hop. Each is bound as spelled inside bwrap."""
    out: list[str] = []

    def add(x: str) -> None:
        if x not in out:
            out.append(x)

    todo = [os.path.abspath(p) for p in paths if p]
    seen: set[str] = set()
    while todo and len(seen) < 64:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        parts = Path(cur).parts
        for i in range(2, len(parts)):
            prefix = os.path.join(*parts[:i])
            if os.path.islink(prefix):
                add(prefix)
        if not os.path.isdir(cur):  # a file's directory; never the parent of a directory (could be $HOME)
            add(os.path.dirname(cur))
        if os.path.islink(cur):
            with contextlib.suppress(OSError):
                todo.append(os.path.normpath(os.path.join(os.path.dirname(cur), os.readlink(cur))))
    return out


def _bwrap_prefix(work: Path, ro_paths: Sequence[Path]) -> list[str]:
    masks = _masked_roots()
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/dev/shm"]
    if Path("/run").is_dir():
        cmd += ["--tmpfs", "/run"]
    for h in _home_dirs():
        cmd += ["--tmpfs", str(h)]
    rebind: list[Path] = []
    explicit = {Path(q) for q in ro_paths}
    for p in list(ro_paths) + _python_paths():
        p = Path(p)
        if not p.exists() or p in rebind:
            continue
        if p not in explicit and any(_is_under(m, p) for m in masks):
            continue  # an interpreter path never re-exposes a masked root (or a parent of one), e.g. $HOME
        if any(_is_under(p, m) for m in masks) and not _is_under(p, work):
            rebind.append(p)
    # Parents before children so nested binds stack correctly.
    for p in sorted(rebind, key=lambda q: len(q.parts)):
        cmd += ["--ro-bind", str(p), str(p)]
    cmd += [
        "--bind", str(work), str(work),
        "--unshare-net", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup-try",
        "--hostname", "malvalid-sandbox", "--cap-drop", "ALL",
        "--die-with-parent", "--new-session", "--chdir", str(work), "--",
    ]
    return cmd


def _unshare_prefix(kill_child: bool = True) -> list[str]:
    cmd = ["unshare", "--user", "--map-root-user", "--net", "--pid", "--fork"]
    if kill_child:
        cmd.append("--kill-child")
    return cmd + ["--"]


def _backend_prefix(backend: str, work: Path, ro_paths: Sequence[Path]) -> list[str]:
    if backend == "bwrap":
        return _bwrap_prefix(work, ro_paths)
    if backend == "unshare":
        return _unshare_prefix(kill_child=bool(_probe_cache.get("unshare", {}).get("kill_child", True)))
    return []


_PROBE_SCRIPT = r"""
import json, os, socket, sys
out = {"pid": os.getpid(), "uid": os.getuid()}
try:
    out["interfaces"] = sorted({n for _, n in socket.if_nameindex()})
except OSError as e:
    out["interfaces"] = None; out["interfaces_error"] = str(e)
try:
    out["readonly_root"] = bool(os.statvfs("/").f_flag & os.ST_RDONLY)
except OSError:
    out["readonly_root"] = None
home = sys.argv[1] if len(sys.argv) > 1 else ""
def only_mounts(d, depth):  # same rule as malvalid.sandbox.worker.home_hidden: bind mounts are not "home"
    if depth > 64:
        return False
    try:
        names = os.listdir(d)
    except OSError:
        return True
    for name in names:
        q = os.path.join(d, name)
        if os.path.islink(q) or not (os.path.ismount(q) or (os.path.isdir(q) and only_mounts(q, depth + 1))):
            return False
    return True
if home:
    out["home_hidden"] = (not os.path.isdir(home)) or only_mounts(home, 0)
try:
    os.write(int(sys.argv[2]), b"fd-ok")
    out["fd_passing"] = True
except Exception as e:
    out["fd_passing"] = False
print(json.dumps(out))
"""

_probe_cache: dict[str, dict[str, Any]] = {}
_probe_lock = threading.Lock()


def _probe_one(backend: str) -> dict[str, Any]:
    res: dict[str, Any] = {"available": False, "network_isolated": False, "detail": ""}
    if backend in ("bwrap", "unshare") and os.environ.get(SIMULATE_NO_OS_SANDBOX_ENV, "").strip() not in ("", "0"):
        res["detail"] = f"disabled by {SIMULATE_NO_OS_SANDBOX_ENV} (simulating a platform without an OS sandbox)"
        return res
    if backend in ("bwrap", "unshare") and not _LINUX:
        res["detail"] = f"{backend} is Linux-only (this platform is {sys.platform})"
        return res
    if backend in ("bwrap", "unshare"):
        exe = shutil.which(backend)
        if exe is None:
            res["detail"] = f"{backend} not found on PATH"
            return res
        res["path"] = exe
    if backend == "subprocess":
        res.update(available=True, network_isolated=False, pid_namespace=False, readonly_root=False,
                   detail="plain subprocess: NO network or file-system isolation (resource limits and pickle guards only)")
        return res
    tmp = Path(tempfile.mkdtemp(prefix="malvalid-probe-"))
    r, w = os.pipe()
    try:
        home = _home_dirs()
        if backend == "unshare":
            hp = subprocess.run(["unshare", "--help"], capture_output=True, text=True, timeout=10)
            res["kill_child"] = "--kill-child" in (hp.stdout + hp.stderr)
            cmd = _unshare_prefix(kill_child=res["kill_child"])
        else:
            cmd = _backend_prefix(backend, tmp, [])
        cmd += [sys.executable, "-I", "-S", "-c", _PROBE_SCRIPT, str(home[0]) if home else "", str(w)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30, pass_fds=(w,), cwd=str(tmp),
                           env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, stdin=subprocess.DEVNULL)
        if p.returncode != 0:
            res["detail"] = f"{backend} failed (exit {p.returncode}): {(p.stderr or p.stdout).strip()[-400:]}"
            return res
        info = json.loads(p.stdout.strip().splitlines()[-1])
        ifaces = info.get("interfaces")
        res["available"] = bool(info.get("fd_passing"))
        res["network_isolated"] = ifaces is not None and set(ifaces) <= {"lo"}
        res["pid_namespace"] = int(info.get("pid", 0)) <= 3
        res["readonly_root"] = bool(info.get("readonly_root"))
        res["home_hidden"] = info.get("home_hidden")
        if backend == "bwrap":
            with contextlib.suppress(Exception):
                res["version"] = subprocess.run(["bwrap", "--version"], capture_output=True, text=True,
                                                timeout=10).stdout.strip()
        parts = []
        parts.append("network isolated" if res["network_isolated"] else f"NETWORK VISIBLE (interfaces: {ifaces})")
        parts.append("own PID namespace" if res["pid_namespace"] else "shared PID namespace")
        parts.append("read-only root" if res["readonly_root"] else "file system NOT restricted")
        if res.get("home_hidden"):
            parts.append("$HOME hidden")
        if not res["available"]:
            parts.append("fd passing failed")
        res["detail"] = ", ".join(parts)
        return res
    except (subprocess.TimeoutExpired, OSError, ValueError, json.JSONDecodeError) as e:
        res["detail"] = f"{backend} probe failed: {type(e).__name__}: {e}"
        return res
    finally:
        with contextlib.suppress(OSError):
            os.close(r)
        with contextlib.suppress(OSError):
            os.close(w)
        shutil.rmtree(tmp, ignore_errors=True)


def probe_backends(*, refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Which isolation backends work here: ``{name: {"available", "network_isolated", "detail", ...}}``.

    Each backend is exercised for real (a tiny interpreter is started under it and reports its
    network interfaces, PID, root-mount flags and whether ``$HOME`` is visible). Cached per process.
    """
    with _probe_lock:
        for b in BACKENDS:
            if refresh or b not in _probe_cache:
                _probe_cache[b] = _probe_one(b)
        return {b: dict(_probe_cache[b]) for b in BACKENDS}


def isolation_level(backend: str | None, network_isolated: bool | None = None) -> str:
    """``os_sandbox`` | ``process_only`` | ``none`` for a resolved backend (``in-process`` = ``none``)."""
    if backend in _ISOLATING and network_isolated is not False:
        return "os_sandbox"
    if backend in _ISOLATING or backend == "subprocess":
        return "process_only"
    return "none"


def no_os_sandbox_message(probes: dict[str, dict[str, Any]] | None = None) -> str:
    """The actionable error raised when ``auto`` finds no OS sandbox and reduced isolation is not allowed."""
    probes = probes if probes is not None else probe_backends()
    return (
        "no OS sandbox is available on this machine "
        f"(bwrap: {probes['bwrap'].get('detail')}; unshare: {probes['unshare'].get('detail')}), so the model "
        "cannot be loaded with network and file-system isolation. malvalid does not downgrade silently. To run it "
        "anyway in a separate worker process with REDUCED isolation (no-unpickle IPC, pickle refusal and resource "
        "limits, but NO network or file-system isolation), pass --allow-reduced-isolation to `malvalid run` / "
        "`malvalid serve` (or set runtime.allow_reduced_isolation: true); the report then records "
        "isolation: process_only. For full isolation use Linux with bubblewrap (or enable unprivileged user "
        "namespaces), e.g. inside WSL2 or a Linux VM. `malvalid sandbox-check` shows what works here."
    )


def select_backend(requested: str, *, allow_reduced_isolation: bool = False) -> tuple[str, list[str]]:
    """Resolve ``auto`` to the strongest working backend. Returns ``(backend, warnings)``.

    ``auto`` with no working OS sandbox raises :class:`SandboxError` unless ``allow_reduced_isolation``
    is set, in which case it returns ``subprocess`` with a reduced-isolation warning.
    """
    warnings: list[str] = []
    if requested == "subprocess":
        warnings.append(
            "NETWORK NOT ISOLATED: sandbox_backend is 'subprocess', so the model runs WITHOUT network or "
            "file-system isolation (resource limits and pickle guards only)"
        )
        return "subprocess", warnings
    probes = probe_backends()
    if requested in _ISOLATING:
        pr = probes[requested]
        if not pr.get("available"):
            raise SandboxError(
                f"sandbox backend {requested!r} was requested but does not work here: {pr.get('detail')}. "
                "Run `malvalid sandbox-check`, or set runtime.sandbox_backend: auto."
            )
        return requested, warnings
    for b in _ISOLATING:
        pr = probes[b]
        if pr.get("available") and pr.get("network_isolated"):
            if b != "bwrap":
                warnings.append(
                    f"bubblewrap is unavailable ({probes['bwrap'].get('detail')}); using {b}: the network is "
                    "isolated but the file system is NOT restricted"
                )
            return b, warnings
    if not allow_reduced_isolation:
        raise SandboxError(no_os_sandbox_message(probes))
    warnings.append(REDUCED_ISOLATION_WARNING)
    warnings.append(
        "NETWORK NOT ISOLATED: no isolating sandbox backend works on this machine "
        f"(bwrap: {probes['bwrap'].get('detail')}; unshare: {probes['unshare'].get('detail')}). The model runs "
        "as a plain subprocess with resource limits and pickle guards only (reduced isolation was allowed "
        "explicitly)."
    )
    return "subprocess", warnings


def os_sandbox_available() -> bool:
    """Does ``auto`` find a network-isolating OS sandbox here (bwrap or unshare)?"""
    probes = probe_backends()
    return any(probes[b].get("available") and probes[b].get("network_isolated") for b in _ISOLATING)


# --------------------------------------------------------------------------------------------------
# Worker process management
# --------------------------------------------------------------------------------------------------

_BOOTSTRAP = (
    "import sys\n"
    "_p = {root!r}\n"
    "if _p not in sys.path: sys.path.insert(0, _p)\n"
    "from malvalid.sandbox.worker import main\n"
    "sys.exit(main(sys.argv[1:]))\n"
)

_ERRORS: dict[str, type[Exception]] = {
    "PickleRefused": PickleRefused,
    "AdapterContractError": AdapterContractError,
    "AdapterError": AdapterError,
    "FeaturizeUnavailable": FeaturizeUnavailable,
    "UnsupportedTreeError": UnsupportedTreeError,
}


def _kill_process(proc: subprocess.Popen[bytes] | None, job: Any = None) -> None:
    if proc is not None and proc.poll() is None:
        killpg = getattr(os, "killpg", None)
        sigkill = getattr(signal, "SIGKILL", None)
        if killpg is not None and sigkill is not None:
            with contextlib.suppress(OSError, ProcessLookupError):
                killpg(proc.pid, sigkill)
        with contextlib.suppress(OSError, ProcessLookupError):
            proc.kill()  # TerminateProcess on Windows
    if job is not None:  # Windows: closing the kill-on-close Job Object ends every process in it
        _close_job(job)
    if proc is not None:
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)


_CRASH_SIGNALS = tuple(getattr(signal, n) for n in ("SIGKILL", "SIGABRT") if hasattr(signal, n))


# ---- Windows Job Object (resource limits for the worker; no ``resource`` module there) -------------


def _windows_job(proc: subprocess.Popen[bytes], memory_mb: int) -> tuple[Any, dict[str, Any]]:  # pragma: no cover
    """Put ``proc`` in a new Job Object: per-process memory limit, no further child processes
    (beyond a venv launcher's real interpreter), kill-on-close (the worker dies with the host), and no
    error-reporting dialog on a crash. Returns ``(job_handle, limits)``. Windows only."""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _Io),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL

    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    # A venv's python.exe is a launcher that starts the base interpreter as a child process.
    base = getattr(sys, "_base_executable", sys.executable) or sys.executable
    launcher = os.path.normcase(os.path.abspath(base)) != os.path.normcase(os.path.abspath(sys.executable))
    active = 2 if launcher else 1
    info = _Extended()
    flags = 0x2000 | 0x0400 | 0x0008  # KILL_ON_JOB_CLOSE | DIE_ON_UNHANDLED_EXCEPTION | ACTIVE_PROCESS
    if memory_mb:
        flags |= 0x0100  # PROCESS_MEMORY
        info.ProcessMemoryLimit = int(memory_mb) << 20
    info.BasicLimitInformation.LimitFlags = flags
    info.BasicLimitInformation.ActiveProcessLimit = active
    try:
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):  # ExtendedLimitInformation
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        if not k32.AssignProcessToJobObject(job, int(proc._handle)):  # type: ignore[attr-defined]
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    except BaseException:
        k32.CloseHandle(job)
        raise
    return job, {"supported": True, "mechanism": "windows_job_object", "process_memory_mb": int(memory_mb) or None,
                 "active_process_limit": active, "kill_on_close": True}


_CREATE_SUSPENDED = 0x00000004


def _resume_process(proc: subprocess.Popen[bytes]) -> None:  # pragma: no cover - Windows only
    """Resume a process started with CREATE_SUSPENDED (Popen keeps only the process handle)."""
    import ctypes
    from ctypes import wintypes

    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    status = ntdll.NtResumeProcess(int(proc._handle))  # type: ignore[attr-defined]
    if status != 0:
        raise OSError(f"NtResumeProcess failed (NTSTATUS {status & 0xFFFFFFFF:#010x})")


def _close_job(job: Any) -> None:  # pragma: no cover - Windows only
    with contextlib.suppress(Exception):
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle(job)


def _ipc_mode() -> str:
    """``stdio`` (worker stdin/stdout; required on Windows) or ``fds`` (two dedicated pipe fds)."""
    if _WINDOWS:
        return "stdio"
    return "stdio" if os.environ.get(IPC_ENV, "").strip().lower() == "stdio" else "fds"


def _describe_exit(rc: int | None, backend: str) -> str:
    if rc is None:
        return "still running"
    if rc < 0:
        with contextlib.suppress(ValueError):
            return f"killed by {signal.Signals(-rc).name}"
        return f"killed by signal {-rc}"
    if backend == "bwrap" and rc > 128:
        with contextlib.suppress(ValueError):
            return f"killed by {signal.Signals(rc - 128).name} (exit {rc})"
    return f"exit code {rc}"


class _WorkerProcess:
    """One spawned worker: process, channel, log offsets. Not reusable after :meth:`kill`."""

    def __init__(self, *, adapter: Path, policy: SandboxPolicy, backend: str, scratch: Path, class_name: str | None,
                 preimport: Sequence[str], ro_paths: Sequence[Path]):
        self.backend = backend
        self.scratch = scratch
        self.log_path = scratch / WORKER_LOG
        self.work = scratch / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "home").mkdir(exist_ok=True)
        (self.work / "tmp").mkdir(exist_ok=True)
        self.policy = policy
        self.proc: subprocess.Popen[bytes] | None = None
        self.channel: Channel | PipeChannel | None = None
        self.ipc = _ipc_mode()
        self.job: Any = None  # Windows Job Object handle
        self.job_limits: dict[str, Any] | None = None
        self.hello: dict[str, Any] = {}
        self._req = 0
        self._log_offset = 0
        self._spawn(adapter, class_name, preimport, ro_paths)

    def _env(self) -> dict[str, str]:
        t = str(max(1, int(self.policy.threads)))
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HOME": str(self.work / "home"),
            "TMPDIR": "/tmp" if self.backend == "bwrap" else str(self.work / "tmp"),
            "MALVALID_IN_SANDBOX": "1",
            "MALVALID_SANDBOX_BACKEND": self.backend,
            "MALVALID_SCRATCH": str(self.work),
            "MALVALID_ALLOW_PICKLE": "1" if self.policy.allow_pickle else "0",
            "OMP_NUM_THREADS": t,
            "OPENBLAS_NUM_THREADS": t,
            "MKL_NUM_THREADS": t,
            "NUMEXPR_NUM_THREADS": t,
            "VECLIB_MAXIMUM_THREADS": t,
            "MALLOC_ARENA_MAX": "4",
        }
        if _WINDOWS:  # pragma: no cover - Windows only
            # A Windows interpreter needs SYSTEMROOT (crypto, sockets) and its own DLL directories on PATH.
            for k in ("SYSTEMROOT", "WINDIR", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
                      "PROCESSOR_IDENTIFIER", "OS", "PATHEXT"):
                if os.environ.get(k):
                    env[k] = os.environ[k]
            sysroot = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or r"C:\Windows"
            dirs = [os.path.dirname(sys.executable), sys.base_prefix, os.path.join(sys.base_prefix, "DLLs"),
                    os.path.join(sysroot, "System32"), sysroot]
            env["PATH"] = os.pathsep.join(dict.fromkeys(d for d in dirs if d))
            tmp = str(self.work / "tmp")
            home = str(self.work / "home")
            env.update(TMPDIR=tmp, TEMP=tmp, TMP=tmp, USERPROFILE=home, APPDATA=home, LOCALAPPDATA=home)
        return env

    def _spawn(self, adapter: Path, class_name: str | None, preimport: Sequence[str], ro_paths: Sequence[Path]) -> None:
        prefix = _backend_prefix(self.backend, self.work, ro_paths)
        stdio = self.ipc == "stdio"
        fds: tuple[int, ...] = ()
        if stdio:
            chan_args = ["--stdio"]
        else:
            req_r, req_w = os.pipe()
            resp_r, resp_w = os.pipe()
            fds = (req_r, req_w, resp_r, resp_w)
            chan_args = ["--in-fd", str(req_r), "--out-fd", str(resp_w)]
        cmd = prefix + [
            sys.executable, "-I", "-B", "-c", _BOOTSTRAP.format(root=str(_package_root())), *chan_args,
            "--memory-mb", str(int(self.policy.memory_mb)), "--nofile", str(int(self.policy.nofile)),
        ]
        popen_kw: dict[str, Any] = {}
        if _WINDOWS:  # pragma: no cover - Windows only
            # Started suspended: the Job Object must hold the process before it can start a child (a venv's
            # python.exe launches the real interpreter as a child, which then inherits the job).
            popen_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | _CREATE_SUSPENDED
        else:
            popen_kw["start_new_session"] = True
        if stdio:
            popen_kw.update(stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        else:
            popen_kw.update(stdin=subprocess.DEVNULL, pass_fds=(req_r, resp_w))
        self.scratch.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as lf:
            self._log_offset = lf.tell()
            lf.write(
                f"\n===== malvalid sandbox worker start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"backend={self.backend} ipc={self.ipc} adapter={adapter} =====\n".encode()
            )
            lf.flush()
            try:
                if stdio:
                    self.proc = subprocess.Popen(cmd, stderr=lf, env=self._env(), cwd=str(self.work),
                                                 close_fds=True, **popen_kw)
                else:
                    self.proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=self._env(), cwd=str(self.work),
                                                 close_fds=True, **popen_kw)
            except OSError as e:
                for fd in fds:
                    with contextlib.suppress(OSError):
                        os.close(fd)
                raise SandboxError(f"could not start the sandbox worker with backend {self.backend}: {e}") from e
        if _WINDOWS:  # pragma: no cover - Windows only
            # While the process is still suspended; and no submitted code runs before the init message anyway.
            try:
                self.job, self.job_limits = _windows_job(self.proc, int(self.policy.memory_mb))
            except Exception as e:  # noqa: BLE001 - reported in sandbox_info
                self.job_limits = {"supported": False, "error": f"{type(e).__name__}: {e}"}
                log.warning("could not apply Windows Job Object limits to the sandbox worker: %s", e)
            try:
                _resume_process(self.proc)
            except Exception as e:  # noqa: BLE001
                _kill_process(self.proc, self.job)
                self.job = None
                for fd in fds:
                    with contextlib.suppress(OSError):
                        os.close(fd)
                raise SandboxError(f"could not resume the sandbox worker process: {e}") from e
        if not stdio:
            os.close(fds[0])
            os.close(fds[3])
        proc = self.proc
        self._finalizer = weakref.finalize(self, _kill_process, proc, self.job)
        if stdio:
            assert proc.stdout is not None and proc.stdin is not None
            self.channel = PipeChannel(proc.stdout, proc.stdin, alive=lambda: proc.poll() is None)
        else:
            self.channel = Channel(fds[2], fds[1], alive=lambda: proc.poll() is None)
        log.debug("spawned sandbox worker pid=%s backend=%s", proc.pid, self.backend)
        self._init_msg = {
            "op": "init",
            "adapter_path": str(adapter),
            "class_name": class_name,
            "allow_pickle": bool(self.policy.allow_pickle),
            "preimport": list(preimport),
            "host_home": str(_home_dirs()[0]) if _home_dirs() else None,
            "sys_path": _operator_pythonpath(),
        }

    # ---- lifecycle -----------------------------------------------------------------------------

    def handshake(self, deadline: float | None) -> dict[str, Any]:
        assert self.channel is not None
        try:
            self.channel.send(self._init_msg, deadline=deadline)
            msg = self.channel.recv(deadline=deadline, max_payload=0)
        except ChannelTimeout as e:
            self.kill()
            raise SandboxError(
                f"the sandbox worker did not start within the time limit ({self.policy.startup_timeout_s:.0f} s)."
                + self.log_tail_text()
            ) from e
        except (ChannelClosed, ProtocolError) as e:
            self.kill()
            raise SandboxError(
                f"the sandbox worker failed to start ({_describe_exit(self.returncode(), self.backend)})."
                + self.log_tail_text()
            ) from e
        if not msg.ok:
            self.kill()
            raise self._error_from(msg.header, "start-up")
        self.hello = msg.header
        return msg.header

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def returncode(self) -> int | None:
        if self.proc is None:
            return None
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=2)
        return self.proc.poll()

    def kill(self) -> None:
        fin = getattr(self, "_finalizer", None)
        if fin is not None:
            fin()  # _kill_process(proc, job) exactly once (the Job Object handle must not be closed twice)
        else:
            _kill_process(self.proc, self.job)
        self.job = None
        if self.channel is not None:
            self.channel.close()

    def shutdown(self, timeout: float = 3.0) -> None:
        if self.alive() and self.channel is not None:
            with contextlib.suppress(Exception):
                self._req += 1
                self.channel.send({"op": "shutdown", "id": self._req}, deadline=time.monotonic() + timeout)
                self.channel.recv(deadline=time.monotonic() + timeout, max_payload=0)
            with contextlib.suppress(Exception):
                assert self.proc is not None
                self.proc.wait(timeout=timeout)
        self.kill()

    def log_tail(self, max_lines: int = 40, max_bytes: int = 16000) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                start = max(self._log_offset, end - max_bytes)
                f.seek(start)
                data = f.read(end - start)
        except OSError:
            return ""
        lines = data.decode("utf-8", "replace").splitlines()
        return "\n".join(lines[-max_lines:])

    def log_tail_text(self) -> str:
        tail = self.log_tail()
        return f" Last lines of {self.log_path}:\n{tail}" if tail.strip() else f" (worker log {self.log_path} is empty)"

    # ---- requests ------------------------------------------------------------------------------

    def _error_from(self, hdr: dict[str, Any], op: str) -> Exception:
        et = str(hdr.get("error_type") or "Error")
        msg = str(hdr.get("message") or "")
        tb = str(hdr.get("traceback") or "")
        cls = _ERRORS.get(et)
        if cls is PickleRefused:
            return PickleRefused(msg)
        if cls is not None:
            return cls(msg)
        if et == "MemoryError":
            return SandboxError(
                f"the model ran out of memory during {op} inside the sandbox (limit runtime.sandbox_memory_mb="
                f"{self.policy.memory_mb}). Raise the limit if the model legitimately needs more.\n{tb[-3000:]}"
            )
        return AdapterError(f"the adapter's {op} raised {et}: {msg}\n--- worker traceback ---\n{tb[-6000:]}")

    def call(
        self,
        op: str,
        header: dict[str, Any] | None = None,
        payloads: dict[str, bytes] | None = None,
        *,
        deadline: float | None,
        max_payload: int,
        timeout_hint: str = "",
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        """One request/response. Raises the mapped adapter error, ``ModuleTimeout`` or ``SandboxError``."""
        if self.channel is None or not self.alive():
            raise SandboxError(
                f"the sandbox worker is not running ({_describe_exit(self.returncode(), self.backend)})."
                + self.log_tail_text()
            )
        self._req += 1
        rid = self._req
        hdr = {"op": op, "id": rid, **(header or {})}
        try:
            self.channel.send(hdr, payloads, deadline=deadline)
            msg = self.channel.recv(deadline=deadline, max_payload=max_payload)
        except ChannelTimeout as e:
            self.kill()
            raise ModuleTimeout(
                f"the sandboxed model did not finish {op} before the deadline{timeout_hint}; the worker was "
                "stopped and will be restarted on the next call"
            ) from e
        except ChannelClosed as e:
            rc = self.returncode()
            self.kill()
            hint = ""
            if rc is not None and (rc in tuple(-s for s in _CRASH_SIGNALS) + tuple(128 + s for s in _CRASH_SIGNALS)):
                hint = (f" If the model ran out of memory, raise runtime.sandbox_memory_mb (currently "
                        f"{self.policy.memory_mb} MB).")
            raise SandboxError(
                f"the sandboxed worker crashed during {op} ({_describe_exit(rc, self.backend)}).{hint}"
                + self.log_tail_text()
            ) from e
        except ProtocolError as e:
            self.kill()
            raise SandboxError(f"the sandboxed worker sent an invalid response to {op}: {e}") from e
        except BaseException:
            # Interrupted mid-frame (SIGALRM module timeout, Ctrl-C): the channel is no longer in sync.
            self.kill()
            raise
        if msg.header.get("id") != rid:
            self.kill()
            raise SandboxError(f"the sandboxed worker answered out of order ({msg.header.get('id')} != {rid})")
        if not msg.ok:
            raise self._error_from(msg.header, op)
        return msg.header, msg.payloads


# --------------------------------------------------------------------------------------------------
# Output validation (the worker is untrusted)
# --------------------------------------------------------------------------------------------------


def validate_proba_output(arr: np.ndarray, n: int, *, offset: int = 0) -> np.ndarray:
    """Enforce the ``predict_proba`` contract: finite float (n,) in [0, 1]."""
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] == n:
        raise AdapterContractError(
            f"predict_proba returned shape {a.shape}; the contract is (n,) malicious-class scores. "
            "If you wrap a scikit-learn style model, return model.predict_proba(X)[:, 1]"
        )
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.reshape(-1)
    if a.shape != (n,):
        raise AdapterContractError(f"predict_proba returned shape {a.shape}; expected ({n},)")
    a = a.astype(np.float64, copy=False)
    bad = ~np.isfinite(a)
    if bad.any():
        first = int(np.flatnonzero(bad)[0]) + offset
        raise AdapterContractError(
            f"predict_proba returned {int(bad.sum())} non-finite value(s) (NaN/inf; first at row {first})"
        )
    if a.size and (a.min() < -1e-9 or a.max() > 1 + 1e-9):
        raise AdapterContractError(
            f"predict_proba returned values outside [0, 1] (min={a.min():.6g}, max={a.max():.6g}); it must "
            "return probabilities, not raw margins or logits"
        )
    return np.clip(a, 0.0, 1.0)


def validate_pred_output(arr: np.ndarray, n: int) -> np.ndarray:
    """Enforce the ``predict`` contract: (n,) values in {0, 1}."""
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.reshape(-1)
    if a.shape != (n,):
        raise AdapterContractError(f"predict returned shape {a.shape}; expected ({n},)")
    if a.size and not np.all(np.isin(a, (0, 1))):
        bad = a[~np.isin(a, (0, 1))]
        raise AdapterContractError(f"predict returned values outside {{0, 1}} (e.g. {bad[:3].tolist()})")
    return a.astype(np.int8)


def _as_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2-D (n, d) feature matrix, got shape {X.shape}")
    return np.ascontiguousarray(X, dtype=np.float32)


def _schema_dim(feature_version: str) -> int | None:
    try:
        from malvalid import registry

        return int(registry.get_schema(feature_version).dim)
    except Exception:  # noqa: BLE001 - unknown schema: no dimension check
        return None


def check_tree_payload(data: bytes, *, max_bytes: int = _MAX_TREE_BYTES) -> None:
    """Reject a tree payload from the (untrusted) worker before :meth:`TreeEnsemble.from_bytes` parses it.

    ``TreeEnsemble.to_bytes`` writes an uncompressed ``.npz`` (a zip of ``.npy`` members). A hostile
    worker could instead send deflate members that expand enormously (a zip bomb) in the harness, so
    only stored (uncompressed) ``.npy`` members whose declared sizes fit inside the payload are accepted.
    Raises :class:`ValueError` with the reason.
    """
    if len(data) > max_bytes:
        raise ValueError(f"tree payload of {len(data)} bytes exceeds the {max_bytes >> 20} MiB limit")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, OSError, ValueError) as e:
        raise ValueError(f"tree payload is not an .npz archive ({e})") from e
    if not infos or len(infos) > 64:
        raise ValueError(f"tree payload has {len(infos)} members; expected a handful of .npy arrays")
    total = 0
    for zi in infos:
        if zi.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"tree payload member {zi.filename!r} is compressed; only stored members are accepted")
        if not zi.filename.endswith(".npy"):
            raise ValueError(f"tree payload member {zi.filename!r} is not a .npy array")
        total += int(zi.file_size)
    if total > len(data):
        raise ValueError(f"tree payload members declare {total} bytes but the payload is only {len(data)} bytes")


# --------------------------------------------------------------------------------------------------
# Model handles
# --------------------------------------------------------------------------------------------------


class _HandleBase:
    """Chunking, validation, query counting and deadline bookkeeping shared by both handles."""

    declarations: ModelDeclarations
    policy: SandboxPolicy

    def __init__(self, declarations: ModelDeclarations, policy: SandboxPolicy):
        self.declarations = declarations
        self.policy = policy
        self._queries = 0
        self._deadline: float | None = None
        self._trees: Any = False  # False = not fetched yet
        self.tree_access_note: str | None = None
        self._featurize_source: str | None = None
        self._closed = False

    # abstract-ish
    def _proba_chunk(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def _pred_chunk(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def query_count(self) -> int:
        return self._queries

    def set_deadline(self, deadline: float | None) -> None:
        """``time.monotonic()`` deadline for every following request (``None`` = no limit)."""
        self._deadline = None if deadline is None else float(deadline)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = _as_matrix(X)
        n = X.shape[0]
        out = np.empty(n, dtype=np.float64)
        step = max(1, int(self.policy.chunk_rows))
        for s in range(0, n, step):
            chunk = X[s : s + step]
            out[s : s + chunk.shape[0]] = validate_proba_output(self._proba_chunk(chunk), chunk.shape[0], offset=s)
            self._queries += chunk.shape[0]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.declarations.extras.get("predict_from_proba"):
            # Model-file-only submissions: the decision rule is predict_proba >= operating_threshold by
            # definition, applied here so an auto-calibrated threshold takes effect without a reload.
            p = self.predict_proba(X)
            return (p >= float(self.declarations.operating_threshold)).astype(np.int8)
        X = _as_matrix(X)
        n = X.shape[0]
        out = np.empty(n, dtype=np.int8)
        step = max(1, int(self.policy.chunk_rows))
        for s in range(0, n, step):
            chunk = X[s : s + step]
            out[s : s + chunk.shape[0]] = validate_pred_output(self._pred_chunk(chunk), chunk.shape[0])
            self._queries += chunk.shape[0]
        return out

    def has_featurize(self) -> bool:
        return self._featurize_source is not None

    def featurize_source(self) -> str | None:
        """``"adapter"`` (adapter.featurize), ``"schema"`` (the schema's extractor) or ``None``."""
        return self._featurize_source

    def _check_vector(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        dim = _schema_dim(self.declarations.feature_version)
        if dim is not None and v.shape[0] != dim:
            raise AdapterContractError(
                f"featurize returned {v.shape[0]} features; schema {self.declarations.feature_version} has {dim}"
            )
        if not np.all(np.isfinite(v)):
            raise AdapterContractError("featurize returned non-finite values")
        return v

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:  # pragma: no cover - overridden
        self._closed = True


class SandboxedModel(_HandleBase):
    """A :class:`~malvalid.context.ModelHandle` whose model lives in an isolated worker process.

    Build it with :func:`open_model`. Every response is validated here. A request that runs past
    :meth:`set_deadline` kills the worker and raises :class:`~malvalid.core.ModuleTimeout`; a worker
    crash raises :class:`~malvalid.core.SandboxError` with the tail of ``worker.log``. Either way the
    next call restarts the worker and reloads the model transparently.
    """

    def __init__(self, adapter_path: Path, policy: SandboxPolicy, *, declarations: ModelDeclarations,
                 class_name: str | None = None):
        super().__init__(declarations, policy)
        self.adapter_path = Path(adapter_path).resolve()
        self.class_name = class_name or declarations.class_name or None
        self.backend, self._warnings = select_backend(policy.backend,
                                                      allow_reduced_isolation=policy.allow_reduced_isolation)
        self._own_scratch = policy.scratch_dir is None
        self.scratch = Path(policy.scratch_dir) if policy.scratch_dir else Path(tempfile.mkdtemp(prefix="malvalid-sandbox-"))
        self.scratch.mkdir(parents=True, exist_ok=True)
        self._worker: _WorkerProcess | None = None
        self._restarts = 0
        self._load_info: dict[str, Any] = {}
        self._hello: dict[str, Any] = {}
        self._crashes = 0
        self._timeouts = 0
        kind = str(declarations.model_kind or "").lower()
        self._preimport: tuple[str, ...] = PREIMPORTS.get(kind, ())

    # ---- lifecycle -----------------------------------------------------------------------------

    def _ro_paths(self) -> list[Path]:
        paths = [self.adapter_path.parent, *self.policy.ro_paths]
        for p in self.declarations.model_paths:
            paths.append(Path(p).parent)
        if self.declarations.training_hashes_path:
            paths.append(Path(self.declarations.training_hashes_path).parent)
        return paths

    def start(self) -> None:
        """Spawn the worker and call the adapter's ``load()`` (bounded by ``startup_timeout_s``)."""
        if self._closed:
            raise SandboxError("this sandboxed model handle is closed")
        t0 = time.monotonic()
        deadline = t0 + float(self.policy.startup_timeout_s)
        if self._deadline is not None:
            deadline = min(deadline, max(self._deadline, t0))
        w = _WorkerProcess(adapter=self.adapter_path, policy=self.policy, backend=self.backend, scratch=self.scratch,
                           class_name=self.class_name, preimport=self._preimport, ro_paths=self._ro_paths())
        self._worker = w
        try:
            self._hello = w.handshake(deadline)
            hdr, _ = w.call("load", deadline=deadline, max_payload=0)
        except BaseException as e:
            w.kill()
            self._worker = None
            if isinstance(e, (ModuleTimeout, SandboxError)) and not isinstance(e, PickleRefused) \
                    and time.monotonic() >= deadline - 0.05:
                if self._deadline is not None and time.monotonic() >= self._deadline - 0.05:
                    raise ModuleTimeout(
                        "the module deadline passed while the sandboxed model was (re)starting"
                    ) from e
                if isinstance(e, ModuleTimeout):
                    raise SandboxError(
                        f"the adapter's load() did not finish within the sandbox start-up limit "
                        f"({self.policy.startup_timeout_s:.0f} s)." + w.log_tail_text()
                    ) from e
            raise
        self._load_info = dict(hdr.get("load") or {})
        self._featurize_source = self._load_info.get("featurize_source")
        self._load_info["startup_s"] = round(time.monotonic() - t0, 3)
        log.info("model loaded in the sandbox (%s, %.2f s)", self.backend, time.monotonic() - t0)

    def _ensure(self) -> _WorkerProcess:
        if self._closed:
            raise SandboxError("this sandboxed model handle is closed")
        w = self._worker
        if w is not None and w.alive():
            return w
        if w is not None:
            w.kill()
        self._worker = None
        self._restarts += 1
        log.info("restarting the sandboxed model worker (restart #%d)", self._restarts)
        self.start()
        assert self._worker is not None
        return self._worker

    def restart(self) -> None:
        """Kill the worker (if any) and start a fresh one with the model reloaded."""
        if self._worker is not None:
            self._worker.kill()
            self._worker = None
        self._restarts += 1
        self.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._worker.shutdown()
            self._worker = None
        if self._own_scratch:
            shutil.rmtree(self.scratch, ignore_errors=True)

    # ---- requests ------------------------------------------------------------------------------

    def _call(self, op: str, header: dict[str, Any] | None = None, payloads: dict[str, bytes] | None = None, *,
              max_payload: int) -> tuple[dict[str, Any], dict[str, bytes]]:
        w = self._ensure()
        try:
            return w.call(op, header, payloads, deadline=self._deadline, max_payload=max_payload)
        except BaseException as e:
            if not w.alive():  # the worker was stopped (timeout/interrupt) or died: restart on next call
                if isinstance(e, ModuleTimeout):
                    self._timeouts += 1
                elif isinstance(e, SandboxError):
                    self._crashes += 1
                w.kill()
                self._worker = None
            raise

    def _proba_chunk(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        _, pl = self._call("predict_proba", payloads={"X": encode_array(X)}, max_payload=8 * (4 * m + 16) + 4096)
        return decode_array(pl.get("y", b""), max_elements=4 * m + 16)

    def _pred_chunk(self, X: np.ndarray) -> np.ndarray:
        m = X.shape[0]
        _, pl = self._call("predict", payloads={"X": encode_array(X)}, max_payload=8 * (4 * m + 16) + 4096)
        return decode_array(pl.get("y", b""), max_elements=4 * m + 16)

    def featurize(self, raw: bytes) -> np.ndarray:
        """Raw PE bytes -> (dim,) float32, computed inside the sandbox (never in the harness)."""
        if self._featurize_source is None:
            raise FeaturizeUnavailable(
                "no raw-bytes featurizer: the adapter has no featurize() and the schema's extractor is unavailable"
            )
        raw = bytes(raw)
        if len(raw) > _MAX_FEATURIZE_INPUT:
            raise ValueError(f"input of {len(raw)} bytes exceeds the {_MAX_FEATURIZE_INPUT >> 20} MiB featurize limit")
        _, pl = self._call("featurize", payloads={"raw": raw}, max_payload=64 << 20)
        return self._check_vector(decode_array(pl.get("v", b""), max_elements=8 << 20))

    def tree_ensemble(self) -> "TreeEnsemble | None":
        """Normalized trees (fetched once, then cached); ``None`` with :attr:`tree_access_note` if unavailable."""
        if self._trees is not False:
            return self._trees  # type: ignore[no-any-return]
        from malvalid.loaders.trees import TreeEnsemble

        try:
            hdr, pl = self._call("tree_ensemble", max_payload=_MAX_TREE_BYTES)
        except ModuleTimeout:
            raise
        except (AdapterError, SandboxError, UnsupportedTreeError) as e:
            self.tree_access_note = f"tree extraction failed: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
            self._trees = None
            return None
        if not hdr.get("available"):
            self.tree_access_note = str(hdr.get("reason") or "trees unavailable")
            self._trees = None
            return None
        try:
            data = pl.get("trees", b"")
            check_tree_payload(data)
            te = TreeEnsemble.from_bytes(data)
        except Exception as e:  # noqa: BLE001 - untrusted payload
            self.tree_access_note = f"the worker sent an invalid tree payload ({type(e).__name__}: {e})"
            self._trees = None
            return None
        self.tree_access_note = str(hdr.get("how") or "")
        self._trees = te
        return te

    def self_check(self) -> dict[str, Any]:
        """Fresh isolation self-report from inside the worker."""
        hdr, _ = self._call("self_check", max_payload=0)
        return dict(hdr.get("self_check") or {})

    # ---- info ----------------------------------------------------------------------------------

    @property
    def worker_log(self) -> Path:
        return self.scratch / WORKER_LOG

    @property
    def worker_pid(self) -> int | None:
        w = self._worker
        return w.proc.pid if w is not None and w.proc is not None else None

    def sandbox_info(self) -> dict[str, Any]:
        sc = dict(self._hello.get("self_check") or {})
        net = bool(sc.get("network_isolated")) if sc else self.backend in _ISOLATING
        warnings = list(self._warnings)
        if not net and not any("NETWORK NOT ISOLATED" in w for w in warnings):
            warnings.append("NETWORK NOT ISOLATED: the model worker could reach the host's network interfaces")
        fs = {
            "bwrap": "read-only root; private /tmp; /run and $HOME hidden; writable scratch only",
            "unshare": "NOT restricted (unshare backend isolates network and PIDs only)",
            "subprocess": "NOT restricted",
        }[self.backend]
        isolation = isolation_level(self.backend, net)
        if isolation != "os_sandbox" and REDUCED_ISOLATION_WARNING not in warnings:
            warnings.insert(0, REDUCED_ISOLATION_WARNING)
        w = self._worker
        rlimits = self._hello.get("rlimits")
        if w is not None and w.job_limits is not None:
            rlimits = {**dict(rlimits or {}), "job": w.job_limits}
        return {
            "enabled": True,
            "isolation": isolation,
            "platform": sys.platform,
            "backend": self.backend,
            "backend_requested": self.policy.backend,
            "network_isolated": net,
            "pid_namespace": self.backend in _ISOLATING,
            "filesystem": fs,
            "readonly_root": sc.get("readonly_root"),
            "home_hidden": sc.get("home_hidden"),
            "memory_mb": self.policy.memory_mb,
            "threads": self.policy.threads,
            "chunk_rows": self.policy.chunk_rows,
            "allow_pickle": self.policy.allow_pickle,
            "pickle_guards": list(self._hello.get("pickle_guards") or []),
            "rlimits": rlimits,
            "ipc": _ipc_mode(),
            "env_scrubbed": True,
            "worker_python": sc.get("python"),
            "worker_log": str(self.worker_log),
            "scratch_dir": str(self.scratch / "work"),
            "restarts": self._restarts,
            "timeouts": self._timeouts,
            "crashes": self._crashes,
            "startup_s": self._load_info.get("startup_s"),
            "load_s": self._load_info.get("load_s"),
            "featurize_source": self._featurize_source,
            "tree_access_note": self.tree_access_note,
            "warnings": warnings,
        }


class InProcessModel(_HandleBase):
    """``SandboxPolicy(enabled=False)``: the adapter runs inside the harness process — NO isolation.

    For debugging trusted models only. Deadlines are checked between chunks (a hung call cannot be
    interrupted), and the pickle guards are not installed (the ``MALVALID_ALLOW_PICKLE`` policy of
    :func:`malvalid.loaders.base.load_model` still applies during ``load()``).
    """

    def __init__(self, adapter_path: Path, policy: SandboxPolicy, *, declarations: ModelDeclarations,
                 class_name: str | None = None):
        super().__init__(declarations, policy)
        from malvalid.sandbox.worker import AdapterRuntime

        self.adapter_path = Path(adapter_path).resolve()
        self.runtime = AdapterRuntime(self.adapter_path, class_name=class_name or declarations.class_name or None,
                                      unique_module_name=True)
        self._restarts = 0
        self._load_info: dict[str, Any] = {}

    def start(self) -> None:
        old = os.environ.get("MALVALID_ALLOW_PICKLE")
        os.environ["MALVALID_ALLOW_PICKLE"] = "1" if self.policy.allow_pickle else "0"
        cwd = os.getcwd()
        try:
            with contextlib.suppress(OSError):
                os.chdir(self.adapter_path.parent)
            self._load_info = self.runtime.load()
        finally:
            with contextlib.suppress(OSError):
                os.chdir(cwd)
            if old is None:
                os.environ.pop("MALVALID_ALLOW_PICKLE", None)
            else:
                os.environ["MALVALID_ALLOW_PICKLE"] = old
        self._featurize_source = self._load_info.get("featurize_source")

    def _check(self) -> None:
        if self._closed:
            raise SandboxError("this model handle is closed")
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise ModuleTimeout("the in-process model call started after the module deadline")

    def _proba_chunk(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return self.runtime.predict_proba(X)

    def _pred_chunk(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return self.runtime.predict(X)

    def featurize(self, raw: bytes) -> np.ndarray:
        self._check()
        return self._check_vector(self.runtime.featurize(bytes(raw)))

    def tree_ensemble(self) -> "TreeEnsemble | None":
        if self._trees is False:
            try:
                te, how = self.runtime.tree_ensemble()
            except Exception as e:  # noqa: BLE001 - tree access is optional
                te, how = None, f"tree extraction failed: {type(e).__name__}: {e}"
            self._trees, self.tree_access_note = te, how
        return self._trees  # type: ignore[no-any-return]

    def restart(self) -> None:
        self._restarts += 1
        self._trees = False
        self.start()

    def close(self) -> None:
        self._closed = True
        self.runtime.inst = None

    def sandbox_info(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "isolation": "none",
            "platform": sys.platform,
            "backend": "in-process",
            "backend_requested": self.policy.backend,
            "network_isolated": False,
            "pid_namespace": False,
            "filesystem": "NOT restricted",
            "allow_pickle": self.policy.allow_pickle,
            "pickle_guards": [],
            "env_scrubbed": False,
            "chunk_rows": self.policy.chunk_rows,
            "restarts": self._restarts,
            "load_s": self._load_info.get("load_s"),
            "featurize_source": self._featurize_source,
            "tree_access_note": self.tree_access_note,
            "warnings": [
                "SANDBOX DISABLED: the submitted adapter and model ran inside the malvalid process with no "
                "network, file-system, memory or pickle isolation. Use this only to debug a trusted model; "
                "never for a production-readiness decision on a model from someone else."
            ],
        }


# --------------------------------------------------------------------------------------------------
# inspect_adapter / open_model
# --------------------------------------------------------------------------------------------------


def _resolve_rel(value: str, base: Path) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def _declarations_from_inspect(
    adapter: Path, info: dict[str, Any], model_paths_override: Sequence[Path]
) -> ModelDeclarations:
    cls_name = str(info.get("class_name") or "")
    raw: dict[str, dict[str, Any]] = info.get("declarations") or {}
    base = adapter.parent
    problems: list[str] = []

    def get(attr: str) -> dict[str, Any]:
        return raw.get(attr) or {"present": False}

    missing = [a for a in ("feature_version", "model_kind", "operating_threshold", "training_hashes_path",
                           "training_cutoff") if not get(a).get("present")]
    for a in missing:
        hint = " (set it to None if unknown)" if a in ("training_hashes_path", "training_cutoff") else ""
        problems.append(f"missing class attribute {a}{hint}")

    def bad_type(attr: str, d: dict[str, Any], want: str) -> None:
        problems.append(f"{attr} must be {want}, got {d.get('type')} {d.get('repr', repr(d.get('value')))}")

    fv = get("feature_version")
    feature_version = ""
    if fv.get("present"):
        if fv.get("type") != "str" or not str(fv.get("value") or "").strip():
            bad_type("feature_version", fv, "a non-empty string such as 'ember_v2' or 'ember_v3'")
        else:
            feature_version = str(fv["value"]).strip()
    mk = get("model_kind")
    model_kind = ""
    if mk.get("present"):
        if mk.get("type") != "str" or not str(mk.get("value") or "").strip():
            bad_type("model_kind", mk, "a non-empty string such as 'lightgbm', 'xgboost', 'sklearn_gbdt' or 'onnx'")
        else:
            model_kind = str(mk["value"]).strip()
    th = get("operating_threshold")
    threshold = float("nan")
    if th.get("present"):
        v = th.get("value")
        if th.get("type") == "bool" or not isinstance(v, (int, float)):
            bad_type("operating_threshold", th, "a number in [0, 1]")
        elif not (math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0):
            problems.append(f"operating_threshold must be in [0, 1] (the score cut-off used in production), got {v}")
        else:
            threshold = float(v)
    hp = get("training_hashes_path")
    hashes_path: str | None = None
    if hp.get("present") and hp.get("value") is not None:
        if hp.get("type") not in ("str", "path"):
            bad_type("training_hashes_path", hp, "a path string or None")
        elif str(hp["value"]).strip():
            rp = _resolve_rel(str(hp["value"]), base)
            if not rp.is_file():
                problems.append(
                    f"training_hashes_path {hp['value']!r} does not exist (resolved to {rp}; relative paths are "
                    "resolved against the adapter's directory)"
                )
            hashes_path = str(rp)
    elif hp.get("present") and hp.get("type") not in ("NoneType",):
        bad_type("training_hashes_path", hp, "a path string or None")
    tc = get("training_cutoff")
    cutoff: str | None = None
    if tc.get("present") and tc.get("value") is not None:
        if tc.get("type") not in ("str", "date"):
            bad_type("training_cutoff", tc, "a date string 'YYYY-MM-DD' / 'YYYY-MM' / 'YYYY' or None")
        else:
            cutoff = str(tc["value"]).strip() or None
            if cutoff:
                try:
                    from malvalid.manifest import parse_training_cutoff

                    parse_training_cutoff(cutoff)
                except AdapterError as e:
                    problems.append(str(e))
                except ImportError:  # pragma: no cover - manifest module missing
                    pass
    elif tc.get("present") and tc.get("type") not in ("NoneType",):
        bad_type("training_cutoff", tc, "a date string or None")

    declared: list[str] = []
    for attr in ("model_path", "model_paths"):
        d = get(attr)
        if not d.get("present") or d.get("value") is None:
            if d.get("present") and d.get("type") not in ("NoneType",):
                bad_type(attr, d, "a path string" if attr == "model_path" else "a list of path strings")
            continue
        if attr == "model_path" and d.get("type") in ("str", "path"):
            declared.append(str(d["value"]))
        elif attr == "model_paths" and d.get("type") in ("list", "str", "path"):
            v = d["value"]
            declared.extend([str(v)] if isinstance(v, str) else [str(x) for x in v])
        else:
            bad_type(attr, d, "a path string" if attr == "model_path" else "a list of path strings")

    model_paths: list[str] = []
    if model_paths_override:
        source = "override"
        for p in model_paths_override:
            rp = Path(p).expanduser().resolve()
            if not rp.exists():
                problems.append(f"--model {p}: file not found")
            model_paths.append(str(rp))
    else:
        source = "adapter" if declared else "none"
        for v in declared:
            rp = _resolve_rel(v, base)
            if not rp.exists():
                problems.append(
                    f"model_path {v!r} does not exist (resolved to {rp}; relative paths are resolved against the "
                    "adapter's directory)"
                )
            model_paths.append(str(rp))
    model_paths = list(dict.fromkeys(model_paths))
    if not info.get("load_is_classmethod", True):
        problems.append(f"{cls_name}.load must be a @classmethod returning a loaded instance")
    if problems:
        raise AdapterError(
            f"{adapter.name}: class {cls_name or '?'} does not meet the submission contract:\n  - "
            + "\n  - ".join(problems)
        )
    return ModelDeclarations(
        feature_version=feature_version,
        model_kind=model_kind,
        operating_threshold=threshold,
        training_hashes_path=hashes_path,
        training_cutoff=cutoff,
        model_paths=tuple(model_paths),
        adapter_path=str(adapter),
        class_name=cls_name,
        has_featurize=bool(info.get("has_featurize")),
        extras={
            "module_name": info.get("module_name"),
            "has_tree_ensemble": bool(info.get("has_tree_ensemble")),
            "declared_model_paths": declared,
            "model_paths_source": source,
        },
    )


def _check_adapter_file(adapter_path: Path) -> Path:
    adapter = Path(adapter_path).expanduser().resolve()
    if not adapter.is_file():
        raise AdapterError(f"adapter not found: {adapter}")
    if adapter.suffix != ".py" and adapter.suffix.lower() != ".json":
        raise AdapterError(
            f"the adapter must be a Python file (.py) or a malvalid model spec (.json), got {adapter.name}"
        )
    return adapter


def declarations_from_spec(spec_path: Path, model_paths_override: Sequence[Path] = ()) -> ModelDeclarations:
    """:class:`ModelDeclarations` of a model-file-only submission, read from its data-only spec.

    The spec is plain JSON validated by :func:`malvalid.adapters.spec.load_spec`, so the host reads it
    directly; no submitted code is involved. The sandboxed worker re-reads and re-validates the same
    file when it loads the model.
    """
    from malvalid.adapters.spec import load_spec

    spec_path = Path(spec_path).resolve()
    spec = load_spec(spec_path)
    base = spec_path.parent
    problems: list[str] = []
    if model_paths_override:
        model_paths = [str(Path(p).expanduser().resolve()) for p in model_paths_override]
        source = "override"
    else:
        model_paths = [str((base / spec.model_file).resolve())]
        source = "spec"
    for mp in model_paths:
        if not Path(mp).exists():
            problems.append(f"model file {Path(mp).name} not found ({mp})")
    hashes_path: str | None = None
    if spec.training_hashes_file:
        hp = (base / spec.training_hashes_file).resolve()
        if not hp.is_file():
            problems.append(f"training hash list {spec.training_hashes_file} not found ({hp})")
        hashes_path = str(hp)
    if problems:
        raise AdapterError(f"{spec_path.name}: " + "; ".join(problems))
    threshold = spec.operating_threshold
    extras: dict[str, Any] = {
        "submission": "model_file",
        "spec": spec.to_dict(),
        "predict_from_proba": True,
        "threshold_source": "calibration_pending" if spec.calibrate else "declared",
        "declared_model_paths": [spec.model_file],
        "model_paths_source": source,
        "has_tree_ensemble": False,
        "module_name": None,
    }
    return ModelDeclarations(
        feature_version=spec.feature_version,
        model_kind=spec.model_kind,
        operating_threshold=float(threshold) if threshold is not None else float("nan"),
        training_hashes_path=hashes_path,
        training_cutoff=spec.training_cutoff,
        model_paths=tuple(dict.fromkeys(model_paths)),
        adapter_path=str(spec_path),
        class_name="ModelFileDetector",
        has_featurize=False,
        extras=extras,
    )


def inspect_adapter(
    adapter_path: Path,
    policy: SandboxPolicy,
    *,
    class_name: str | None = None,
    model_paths_override: Sequence[Path] = (),
) -> ModelDeclarations:
    """Import the adapter in a sandboxed worker WITHOUT calling ``load()`` and validate its declarations.

    Raises :class:`~malvalid.core.AdapterError` with an actionable message on any contract problem.
    """
    adapter = _check_adapter_file(adapter_path)
    if adapter.suffix.lower() == ".json":
        return declarations_from_spec(adapter, model_paths_override)
    if not policy.enabled:
        log.warning("sandbox disabled: importing the adapter %s in-process (debug mode, no isolation)", adapter)
        from malvalid.sandbox.worker import AdapterRuntime

        info = AdapterRuntime(adapter, class_name=class_name, unique_module_name=True).inspect()
        return _declarations_from_inspect(adapter, info, model_paths_override)
    backend, warnings = select_backend(policy.backend, allow_reduced_isolation=policy.allow_reduced_isolation)
    for w in warnings:
        log.warning("%s", w)
    own = policy.scratch_dir is None
    scratch = Path(policy.scratch_dir) if policy.scratch_dir else Path(tempfile.mkdtemp(prefix="malvalid-sandbox-"))
    ro = [adapter.parent, *policy.ro_paths, *(Path(p).parent for p in model_paths_override)]
    deadline = time.monotonic() + float(policy.startup_timeout_s)
    w = _WorkerProcess(adapter=adapter, policy=policy, backend=backend, scratch=scratch, class_name=class_name,
                       preimport=(), ro_paths=ro)
    try:
        w.handshake(deadline)
        try:
            hdr, _ = w.call("inspect", deadline=deadline, max_payload=0,
                            timeout_hint=f" (start-up limit {policy.startup_timeout_s:.0f} s)")
        except ModuleTimeout as e:
            raise AdapterError(f"importing the adapter took longer than {policy.startup_timeout_s:.0f} s") from e
    finally:
        w.shutdown()
        if own:
            shutil.rmtree(scratch, ignore_errors=True)
    return _declarations_from_inspect(adapter, dict(hdr.get("inspect") or {}), model_paths_override)


def open_model(
    adapter_path: Path,
    policy: SandboxPolicy,
    *,
    declarations: ModelDeclarations,
    class_name: str | None = None,
) -> SandboxedModel | InProcessModel:
    """Spawn the worker and call the adapter's ``load()``. The caller must :meth:`~SandboxedModel.close` it."""
    adapter = _check_adapter_file(adapter_path)
    handle: SandboxedModel | InProcessModel
    if not policy.enabled:
        log.warning(
            "SANDBOX DISABLED: loading %s in-process with no isolation — debug trusted models only", adapter
        )
        handle = InProcessModel(adapter, policy, declarations=declarations, class_name=class_name)
    else:
        handle = SandboxedModel(adapter, policy, declarations=declarations, class_name=class_name)
    try:
        handle.start()
    except BaseException:
        handle.close()
        raise
    return handle


__all__ = [
    "BACKENDS",
    "InProcessModel",
    "SandboxPolicy",
    "SandboxedModel",
    "ISOLATION_LEVELS",
    "REDUCED_ISOLATION_WARNING",
    "check_tree_payload",
    "declarations_from_spec",
    "isolation_level",
    "no_os_sandbox_message",
    "os_sandbox_available",
    "inspect_adapter",
    "open_model",
    "probe_backends",
    "select_backend",
    "validate_pred_output",
    "validate_proba_output",
]
