"""Small cross-platform helpers (Windows branches only; POSIX behaviour is unchanged).

* :func:`replace_file` — ``os.replace`` that, on Windows only, retries briefly while the destination
  is held open by a reader (Windows refuses to replace an open file with ``PermissionError``).
* :func:`link_or_copy` — symlink, else hard link, else copy (Windows without Developer Mode cannot
  create symlinks: ``OSError`` / WinError 1314).
* :func:`win_pid_alive` — process liveness via ``OpenProcess`` + ``GetExitCodeProcess`` (``os.kill(pid, 0)``
  is *not* a probe on Windows: signal 0 is ``CTRL_C_EVENT``). ``ctypes`` is imported, and kernel32 loaded,
  only when it is called, which happens only on Windows.
"""

from __future__ import annotations

import contextlib
import errno
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

WINDOWS = os.name == "nt"

#: ``os.replace`` attempts on Windows while the destination is open elsewhere, and the pause between them.
REPLACE_ATTEMPTS = 10
REPLACE_RETRY_S = 0.05

#: Windows ``ERROR_PRIVILEGE_NOT_HELD`` (creating a symlink without Developer Mode / admin rights).
ERROR_PRIVILEGE_NOT_HELD = 1314


def replace_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """``os.replace(src, dst)``; on Windows, retry up to :data:`REPLACE_ATTEMPTS` times on
    ``PermissionError`` (``dst`` open by a reader), then re-raise."""
    if not WINDOWS:
        os.replace(src, dst)
        return
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt >= REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_RETRY_S)


def link_or_copy(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> str:
    """Make ``dst`` refer to the content of ``src``: a symlink when possible, else a hard link (same
    volume), else a copy (``shutil.copy2``). Returns ``"symlink"``, ``"hardlink"`` or ``"copy"``.

    ``dst`` must not exist (``FileExistsError`` is raised as from ``os.symlink``, never overwritten).
    If every method fails, the symlink error is raised. Callers must only rely on ``dst``'s *content*
    (a hard link or copy does not resolve to ``src``).
    """
    try:
        os.symlink(src, dst)
        return "symlink"
    except FileExistsError:
        raise
    except OSError as e:
        first = e
    try:
        os.link(src, dst)
        return "hardlink"
    except FileExistsError:
        raise
    except OSError:
        pass
    try:
        if os.path.lexists(dst):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dst))
        shutil.copy2(src, dst)
        return "copy"
    except FileExistsError:
        raise
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(dst)
        raise first from None


# --------------------------------------------------------------------------------------------------
# Windows process probing (ctypes, kernel32) — never imported or called on POSIX
# --------------------------------------------------------------------------------------------------

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259
ERROR_ACCESS_DENIED = 5
_kernel32_cache: Any = None


def _kernel32() -> Any:
    """kernel32 (own ``WinDLL`` instance, so these prototypes never clash with other ctypes users)."""
    global _kernel32_cache
    if _kernel32_cache is None:
        import ctypes
        from ctypes import wintypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE  # pointer-sized
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                 ctypes.POINTER(wintypes.DWORD)]
        k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        _kernel32_cache = k
    return _kernel32_cache


def _last_error() -> int:
    import ctypes

    return int(ctypes.get_last_error())  # type: ignore[attr-defined]


def _image_name(k: Any, handle: Any) -> str | None:
    import ctypes
    from ctypes import wintypes

    buf = ctypes.create_unicode_buffer(32768)
    size = wintypes.DWORD(len(buf))
    if not k.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
        return None
    return buf.value or None


def _is_malvalid_image(image: str) -> bool:
    """Could a process with this executable be a ``malvalid`` run (``python -m malvalid`` or a launcher)?"""
    stem = Path(image.replace("\\", "/")).name.lower()
    return stem.startswith("python") or "malvalid" in stem


def win_pid_alive(pid: int, *, expect: Iterable[str] = ()) -> bool:
    """Windows: is ``pid`` a running process? A process we may not open (access denied) counts as alive.

    Windows offers no cheap access to another process's command line, so ``expect`` is checked only
    loosely: the executable must be a Python interpreter or a ``malvalid`` launcher (a weaker guard
    against pid reuse than the ``/proc`` command-line check on Linux).
    """
    import ctypes
    from ctypes import wintypes

    k = _kernel32()
    handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return _last_error() == ERROR_ACCESS_DENIED
    try:
        code = wintypes.DWORD(0)
        if not k.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        if code.value != STILL_ACTIVE:
            return False
        if any(e for e in expect):
            image = _image_name(k, handle)
            if image is not None and not _is_malvalid_image(image):
                return False
        return True
    finally:
        k.CloseHandle(handle)


__all__ = ["WINDOWS", "link_or_copy", "replace_file", "win_pid_alive"]
