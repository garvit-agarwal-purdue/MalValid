"""Wire protocol between the malvalid host process and its sandboxed model worker.

Frame layout (integers big-endian)::

    b"MGW1" | u32 header_len | header (UTF-8 JSON object) | payload_0 | payload_1 | ...

The JSON header lists its binary payloads under ``"payloads": [{"name", "kind", "size"}, ...]`` in the
order in which they follow the header. ``kind`` is ``"npy"`` (a numpy array written with
``np.save(..., allow_pickle=False)``) or ``"raw"`` (opaque bytes, e.g. a PE file for ``featurize`` or a
serialized :class:`~malvalid.loaders.trees.TreeEnsemble`).

**Nothing on this channel is ever unpickled.** Headers are parsed with the stdlib JSON parser, arrays
are decoded with ``allow_pickle=False`` and restricted to plain numeric dtypes, and every size is
bounded before any allocation. The worker side is untrusted: the host treats every frame it receives as
hostile input.

Two I/O flavours are provided:

* :func:`write_message` / :func:`read_message` — blocking, used inside the worker;
* :class:`Channel` — non-blocking with deadlines and a liveness callback, used by the host so that a hung
  or crashed worker can never hang the harness (POSIX: ``poll`` on two dedicated pipe fds);
* :class:`PipeChannel` — the same interface over the worker's stdin/stdout pipes, with deadlines enforced by
  helper threads. Used where pipe fds cannot be passed to a child or polled (Windows), and selectable on
  POSIX for testing (``MALVALID_SANDBOX_IPC=stdio``).
"""

from __future__ import annotations

import io
import json
import os
import select
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, Mapping

import numpy as np

from malvalid.core import SandboxError

MAGIC = b"MGW1"
_PREFIX = struct.Struct(">4sI")
MAX_HEADER_BYTES = 16 << 20  # 16 MiB of JSON is far more than any legitimate header
DEFAULT_MAX_PAYLOAD_BYTES = 16 << 30  # per frame; hosts pass tighter, per-request bounds
_IO_CHUNK = 1 << 20
_NUMERIC_KINDS = frozenset("biuf")


class ProtocolError(SandboxError):
    """A malformed, oversized, or otherwise invalid frame was received."""


class ChannelClosed(SandboxError):
    """The peer closed its end of the channel (for the host: the worker died or exited)."""


class ChannelTimeout(SandboxError):
    """A deadline expired while waiting on the channel."""


# --------------------------------------------------------------------------------------------------
# Payload encoding
# --------------------------------------------------------------------------------------------------


def encode_array(arr: Any) -> bytes:
    """Serialize a numeric numpy array as ``.npy`` bytes (never pickles; object arrays are refused)."""
    a = np.asarray(arr)
    if a.dtype.kind not in _NUMERIC_KINDS:
        raise ProtocolError(f"refusing to send a non-numeric array (dtype={a.dtype})")
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


def decode_array(data: bytes | bytearray | memoryview, *, max_elements: int | None = None) -> np.ndarray:
    """Decode ``.npy`` bytes produced by :func:`encode_array`.

    Refuses pickled/object/structured dtypes and arrays larger than ``max_elements``.
    """
    bio = io.BytesIO(bytes(data) if isinstance(data, memoryview) else data)
    try:
        version = np.lib.format.read_magic(bio)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(bio)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(bio)
        else:
            raise ProtocolError(f"unsupported .npy format version {version}")
    except ProtocolError:
        raise
    except Exception as e:  # noqa: BLE001 - any parse failure is a protocol violation
        raise ProtocolError(f"invalid array payload: {e}") from e
    if dtype.kind not in _NUMERIC_KINDS or dtype.hasobject or dtype.fields is not None:
        raise ProtocolError(f"refusing array payload with dtype {dtype!r}")
    n_el = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if max_elements is not None and n_el > max_elements:
        raise ProtocolError(f"array payload has {n_el} elements; at most {max_elements} expected")
    bio.seek(0)
    try:
        arr = np.load(bio, allow_pickle=False)
    except Exception as e:  # noqa: BLE001
        raise ProtocolError(f"invalid array payload: {e}") from e
    if not isinstance(arr, np.ndarray):
        raise ProtocolError("array payload did not decode to an ndarray")
    return arr


@dataclass
class Message:
    """One decoded frame."""

    header: dict[str, Any]
    payloads: dict[str, bytes] = field(default_factory=dict)

    def array(self, name: str, *, max_elements: int | None = None) -> np.ndarray:
        if name not in self.payloads:
            raise ProtocolError(f"frame has no payload {name!r}")
        return decode_array(self.payloads[name], max_elements=max_elements)

    @property
    def ok(self) -> bool:
        return self.header.get("ok") is True


def _payload_kind(name: str, data: bytes) -> str:
    return "npy" if data[:6] == b"\x93NUMPY" else "raw"


def pack_message(header: Mapping[str, Any], payloads: Mapping[str, bytes] | None = None) -> list[bytes]:
    """Serialize a frame into a list of byte segments (prefix+header, then each payload)."""
    payloads = dict(payloads or {})
    hdr = dict(header)
    hdr["payloads"] = [
        {"name": str(k), "kind": _payload_kind(k, v), "size": len(v)} for k, v in payloads.items()
    ]
    raw = json.dumps(hdr, separators=(",", ":"), default=str).encode("utf-8")
    if len(raw) > MAX_HEADER_BYTES:
        raise ProtocolError(f"header of {len(raw)} bytes exceeds {MAX_HEADER_BYTES}")
    return [_PREFIX.pack(MAGIC, len(raw)) + raw, *payloads.values()]


def _parse_header(raw: bytes, max_payload: int) -> tuple[dict[str, Any], list[tuple[str, int]]]:
    try:
        hdr = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"invalid frame header: {e}") from e
    if not isinstance(hdr, dict):
        raise ProtocolError("frame header is not a JSON object")
    specs = hdr.pop("payloads", [])
    if not isinstance(specs, list):
        raise ProtocolError("frame header 'payloads' is not a list")
    out: list[tuple[str, int]] = []
    total = 0
    for s in specs:
        if not isinstance(s, dict) or not isinstance(s.get("name"), str) or not isinstance(s.get("size"), int):
            raise ProtocolError("malformed payload descriptor in frame header")
        size = int(s["size"])
        if size < 0:
            raise ProtocolError("negative payload size")
        total += size
        if total > max_payload:
            raise ProtocolError(
                f"frame declares {total} payload bytes; at most {max_payload} are acceptable for this request"
            )
        out.append((s["name"], size))
    return hdr, out


def _check_prefix(prefix: bytes) -> int:
    magic, hlen = _PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ProtocolError(f"bad frame magic {magic!r}")
    if hlen > MAX_HEADER_BYTES:
        raise ProtocolError(f"frame header of {hlen} bytes exceeds {MAX_HEADER_BYTES}")
    return int(hlen)


# --------------------------------------------------------------------------------------------------
# Blocking I/O (worker side)
# --------------------------------------------------------------------------------------------------


def _write_all_blocking(fd: int, data: bytes | memoryview) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view[:_IO_CHUNK])
        view = view[n:]


def _readinto(fd: int, view: memoryview) -> int:
    """``os.readv`` into ``view`` where available, else ``os.read`` + copy (Windows has no ``readv``)."""
    if _HAS_READV:
        return os.readv(fd, [view])
    data = os.read(fd, len(view))
    view[: len(data)] = data
    return len(data)


_HAS_READV = hasattr(os, "readv")


def _read_exact_blocking(fd: int, n: int, *, eof_ok: bool = False) -> bytes | None:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = _readinto(fd, view[got : got + _IO_CHUNK])
        if k == 0:
            if eof_ok and got == 0:
                return None
            raise ChannelClosed(f"peer closed the channel mid-frame ({got}/{n} bytes)")
        got += k
    return bytes(buf)


def write_message(fd: int, header: Mapping[str, Any], payloads: Mapping[str, bytes] | None = None) -> None:
    """Blocking write of one frame to ``fd``."""
    for seg in pack_message(header, payloads):
        _write_all_blocking(fd, seg)


def read_message(fd: int, *, max_payload: int = DEFAULT_MAX_PAYLOAD_BYTES) -> Message:
    """Blocking read of one frame. Raises ``EOFError`` on a clean EOF at a frame boundary."""
    prefix = _read_exact_blocking(fd, _PREFIX.size, eof_ok=True)
    if prefix is None:
        raise EOFError("channel closed")
    hlen = _check_prefix(prefix)
    hdr, specs = _parse_header(_read_exact_blocking(fd, hlen) or b"", max_payload)
    payloads = {name: (_read_exact_blocking(fd, size) or b"") for name, size in specs}
    return Message(hdr, payloads)


# --------------------------------------------------------------------------------------------------
# Non-blocking I/O with deadlines (host side)
# --------------------------------------------------------------------------------------------------


_POLLIN = getattr(select, "POLLIN", 1)
_POLLOUT = getattr(select, "POLLOUT", 4)


class _Poller:
    """``select.poll`` on one fd, or ``select.select`` where ``poll`` is missing (some macOS builds)."""

    def __init__(self, fd: int, *, writable: bool) -> None:
        self.fd = fd
        self.writable = writable
        self._poll: Any = None
        if hasattr(select, "poll"):
            self._poll = select.poll()
            self._poll.register(fd, (select.POLLOUT if writable else select.POLLIN) | select.POLLHUP | select.POLLERR)

    def poll(self, timeout_ms: int) -> bool:
        if self._poll is not None:
            return bool(self._poll.poll(timeout_ms))
        r, w, x = select.select([] if self.writable else [self.fd], [self.fd] if self.writable else [], [self.fd],
                                max(0, timeout_ms) / 1000.0)
        return bool(r or w or x)


class Channel:
    """Host end of the worker channel: non-blocking pipe fds, deadlines, and liveness checks.

    ``alive`` is polled every ``tick`` seconds while waiting; when it returns False and no data is
    pending the wait raises :class:`ChannelClosed`.
    """

    def __init__(
        self,
        read_fd: int,
        write_fd: int,
        *,
        alive: Callable[[], bool] | None = None,
        tick: float = 0.2,
    ) -> None:
        self.read_fd = read_fd
        self.write_fd = write_fd
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        self._alive = alive or (lambda: True)
        self._tick = tick
        self._closed = False

    # ---- low level -----------------------------------------------------------------------------

    def _wait(self, fd: int, events: int, deadline: float | None) -> None:
        poller = _Poller(fd, writable=events == _POLLOUT)
        while True:
            timeout = self._tick
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ChannelTimeout("deadline expired while waiting for the sandboxed worker")
                timeout = min(timeout, remaining)
            ready = poller.poll(max(1, int(timeout * 1000)))
            if ready:
                return  # readable/writable, or HUP/ERR (the subsequent read/write reports it)
            if not self._alive():
                # One last non-blocking look: the worker may have exited right after writing.
                if poller.poll(0):
                    return
                raise ChannelClosed("sandboxed worker exited")

    def _write_all(self, data: bytes | memoryview, deadline: float | None) -> None:
        view = memoryview(data)
        while view:
            try:
                n = os.write(self.write_fd, view[:_IO_CHUNK])
            except BlockingIOError:
                self._wait(self.write_fd, _POLLOUT, deadline)
                continue
            except (BrokenPipeError, ConnectionResetError) as e:
                raise ChannelClosed("sandboxed worker closed its request pipe") from e
            view = view[n:]

    def _read_exact(self, n: int, deadline: float | None, *, eof_ok: bool = False) -> bytes | None:
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            try:
                k = _readinto(self.read_fd, view[got : got + _IO_CHUNK])
            except BlockingIOError:
                self._wait(self.read_fd, _POLLIN, deadline)
                continue
            if k == 0:
                if eof_ok and got == 0:
                    return None
                raise ChannelClosed(f"sandboxed worker closed the channel mid-frame ({got}/{n} bytes)")
            got += k
        return bytes(buf)

    # ---- frames --------------------------------------------------------------------------------

    def send(
        self,
        header: Mapping[str, Any],
        payloads: Mapping[str, bytes] | None = None,
        *,
        deadline: float | None = None,
    ) -> None:
        for seg in pack_message(header, payloads):
            self._write_all(seg, deadline)

    def recv(self, *, deadline: float | None = None, max_payload: int = DEFAULT_MAX_PAYLOAD_BYTES) -> Message:
        prefix = self._read_exact(_PREFIX.size, deadline, eof_ok=True)
        if prefix is None:
            raise ChannelClosed("sandboxed worker closed the channel")
        hlen = _check_prefix(prefix)
        hdr, specs = _parse_header(self._read_exact(hlen, deadline) or b"", max_payload)
        payloads = {name: (self._read_exact(size, deadline) or b"") for name, size in specs}
        return Message(hdr, payloads)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for fd in (self.read_fd, self.write_fd):
            try:
                os.close(fd)
            except OSError:
                pass


class PipeChannel:
    """Host end of the worker channel over the worker's stdin/stdout pipes (file objects).

    Same interface as :class:`Channel`. Used where pipe fds can be neither passed to a child nor
    polled (Windows). A reader thread drains the worker's stdout into a bounded buffer (at most
    ``max_buffer`` bytes ahead of the consumer, so a flooding worker cannot exhaust host memory),
    and each :meth:`send` runs its blocking writes on a helper thread, so every wait honours the
    deadline and the liveness callback. On a timeout the caller kills the worker, which unblocks
    any pending read or write.
    """

    def __init__(
        self,
        read_file: BinaryIO,
        write_file: BinaryIO,
        *,
        alive: Callable[[], bool] | None = None,
        tick: float = 0.2,
        max_buffer: int = 64 << 20,
    ) -> None:
        self._rf = read_file
        self._wf = write_file
        self._alive = alive or (lambda: True)
        self._tick = tick
        self._max_buffer = max(_IO_CHUNK, int(max_buffer))
        self._buf = bytearray()
        self._eof = False
        self._closed = False
        self._cond = threading.Condition()
        self._reader = threading.Thread(target=self._read_loop, name="malvalid-sandbox-reader", daemon=True)
        self._reader.start()

    # ---- reader thread ---------------------------------------------------------------------------

    def _read_loop(self) -> None:
        try:
            while True:
                with self._cond:
                    while len(self._buf) >= self._max_buffer and not self._closed:
                        self._cond.wait(self._tick)
                    if self._closed:
                        return
                data = self._rf.read(_IO_CHUNK)
                if not data:
                    return
                with self._cond:
                    self._buf += data
                    self._cond.notify_all()
        except (OSError, ValueError):  # closed under us, or the worker died
            return
        finally:
            with self._cond:
                self._eof = True
                self._cond.notify_all()

    def _remaining(self, deadline: float | None) -> float:
        timeout = self._tick
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ChannelTimeout("deadline expired while waiting for the sandboxed worker")
            timeout = min(timeout, remaining)
        return timeout

    def _read_exact(self, n: int, deadline: float | None, *, eof_ok: bool = False) -> bytes | None:
        out = bytearray()
        with self._cond:
            while len(out) < n:
                if self._buf:
                    take = min(n - len(out), len(self._buf))
                    out += self._buf[:take]
                    del self._buf[:take]
                    self._cond.notify_all()
                    continue
                if self._eof:
                    if eof_ok and not out:
                        return None
                    raise ChannelClosed(f"sandboxed worker closed the channel mid-frame ({len(out)}/{n} bytes)")
                self._cond.wait(self._remaining(deadline))
                if not self._buf and not self._eof and not self._alive():
                    # The worker exited: give the reader a moment to deliver what it already wrote.
                    self._cond.wait(self._tick)
                    if not self._buf and not self._eof:
                        raise ChannelClosed("sandboxed worker exited")
        return bytes(out)

    # ---- writes ----------------------------------------------------------------------------------

    def _write_segments(self, segments: list[bytes], deadline: float | None) -> None:
        err: list[BaseException] = []
        done = threading.Event()

        def run() -> None:
            try:
                for seg in segments:
                    view = memoryview(seg)
                    while view:
                        k = self._wf.write(view[:_IO_CHUNK])
                        view = view[(k or 0):]  # None = nothing written (non-blocking): retry
                self._wf.flush()
            except BaseException as e:  # noqa: BLE001 - handed to the caller
                err.append(e)
            finally:
                done.set()

        t = threading.Thread(target=run, name="malvalid-sandbox-writer", daemon=True)
        t.start()
        while not done.wait(self._remaining(deadline)):
            if not self._alive():
                if done.wait(self._tick):
                    break
                raise ChannelClosed("sandboxed worker exited")
        if err:
            e = err[0]
            if isinstance(e, (BrokenPipeError, ConnectionResetError, OSError, ValueError)):
                raise ChannelClosed("sandboxed worker closed its request pipe") from e
            raise e

    # ---- frames ----------------------------------------------------------------------------------

    def send(
        self,
        header: Mapping[str, Any],
        payloads: Mapping[str, bytes] | None = None,
        *,
        deadline: float | None = None,
    ) -> None:
        if self._closed:
            raise ChannelClosed("channel closed")
        self._write_segments(pack_message(header, payloads), deadline)

    def recv(self, *, deadline: float | None = None, max_payload: int = DEFAULT_MAX_PAYLOAD_BYTES) -> Message:
        prefix = self._read_exact(_PREFIX.size, deadline, eof_ok=True)
        if prefix is None:
            raise ChannelClosed("sandboxed worker closed the channel")
        hlen = _check_prefix(prefix)
        hdr, specs = _parse_header(self._read_exact(hlen, deadline) or b"", max_payload)
        payloads = {name: (self._read_exact(size, deadline) or b"") for name, size in specs}
        return Message(hdr, payloads)

    def close(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._buf.clear()
            self._cond.notify_all()
        for f in (self._wf, self._rf):
            try:
                f.close()
            except (OSError, ValueError):
                pass
