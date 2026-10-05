"""M0 — model file safety.

Every model artifact is scanned **before anything is deserialized**:

1. `modelscan <https://github.com/protectai/modelscan>`_ (Python API) on each artifact whose format it
   supports. modelscan dispatches on file extension, so pickles hidden behind other extensions,
   compressed joblib containers, and zip archives are presented to it under a supported name (a
   symlink, or the decompressed stream in a private temp dir). Formats modelscan has no scanner for
   (LightGBM text, XGBoost JSON/UBJ, ONNX) are recorded as ``scanned=False`` with the reason.
2. malvalid's own static pickle-opcode scan (``pickletools.genops``; nothing is ever unpickled) as
   defence in depth: every global a pickle can import (``GLOBAL``/``INST``/``STACK_GLOBAL``) is
   classified against a small allowlist of model libraries and a denylist of code-execution,
   process, file and network modules. When the opcode stream is interleaved with raw data (joblib
   embeds numpy buffers), a byte-pattern sweep finds globals the sequential parser cannot reach.

The same global policy (:func:`classify_pickle_global`) is enforced at runtime inside the sandbox
worker when ``--allow-pickle`` is given, so a pickle that slipped past the static scan still cannot
import ``os.system`` and friends.
"""

from __future__ import annotations

import bz2
import contextlib
import gzip
import hashlib
import io
import logging
import lzma
import mmap
import os
import pickletools
import re
import tempfile
import time
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Iterator, Sequence

import numpy as np

from malvalid._platform import link_or_copy
from malvalid.core import ConfigError, GateCheck, GateMode, GateOutcome, Module, ModuleResult, Status, to_jsonable
from malvalid.loaders.base import PICKLE_EXTENSIONS, sniff_format

log = logging.getLogger("malvalid.file_safety")

SEVERITIES: tuple[str, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
SEVERITY_RANK: dict[str, int] = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

#: Largest decompressed stream M0 will materialize for scanning (decompression-bomb guard).
MAX_DECOMPRESSED_BYTES = 8 << 30
#: Largest zip member read into memory for the byte-pattern sweep.
MAX_MEMBER_SWEEP_BYTES = 1 << 30

REEXPORT_HINT = (
    "Re-export the model in a non-executable format — LightGBM: booster.save_model('model.txt'); "
    "XGBoost: model.save_model('model.json') or '.ubj'; scikit-learn: convert to ONNX with skl2onnx — "
    "or re-run with --allow-pickle (runtime.allow_pickle: true) if you trust the file and its origin."
)

# --------------------------------------------------------------------------------------------------
# Pickle global policy (shared with the sandbox worker)
# --------------------------------------------------------------------------------------------------

# Plain data types a model pickle legitimately references.
_SAFE_BUILTIN_NAMES = frozenset(
    {
        "object", "list", "dict", "set", "frozenset", "tuple", "bytearray", "bytes", "str", "int",
        "float", "complex", "bool", "slice", "range", "NoneType", "Ellipsis", "NotImplemented",
    }
)
# Builtins that are code-execution / attribute-walking gadgets.
_DANGEROUS_BUILTIN_NAMES = frozenset(
    {
        "eval", "exec", "execfile", "compile", "open", "__import__", "getattr", "setattr", "delattr",
        "apply", "breakpoint", "input", "globals", "locals", "vars", "reload", "exit", "quit", "help",
        "memoryview", "type", "super", "classmethod", "staticmethod", "property",
    }
)
_EXACT_ALLOWED = frozenset(
    {
        ("copyreg", "_reconstructor"),
        ("copy_reg", "_reconstructor"),
        ("copyreg", "__newobj__"),
        ("_codecs", "encode"),
        ("functools", "partial"),
        ("array", "_array_reconstructor"),
        ("array", "array"),
        ("pathlib", "Path"),
        ("pathlib", "PosixPath"),
        ("pathlib", "WindowsPath"),
        ("pathlib", "PurePath"),
        ("pathlib", "PurePosixPath"),
        ("pathlib", "PureWindowsPath"),
        ("uuid", "UUID"),
        ("re", "_compile"),
    }
)
# Model / numeric libraries whose classes appear in legitimate model pickles.
_ALLOWED_ROOTS = (
    "numpy", "scipy", "sklearn", "lightgbm", "xgboost", "pandas", "joblib.numpy_pickle",
    "joblib.numpy_pickle_compat", "collections", "datetime", "decimal", "fractions", "_collections",
)
# Importing *anything* from these is a code-execution, process, file-system or deserialization gadget.
_CRITICAL_ROOTS = (
    "os", "posix", "nt", "subprocess", "_posixsubprocess", "sys", "runpy", "pty", "shutil", "pickle",
    "_pickle", "cPickle", "marshal", "ctypes", "_ctypes", "importlib", "imp", "zipimport", "code",
    "codeop", "pdb", "bdb", "asyncio", "multiprocessing", "concurrent", "signal", "commands",
    "platform", "threading", "_thread", "socket", "_socket", "ssl", "select", "selectors",
    "builtins.__import__", "dill", "cloudpickle", "torch.hub", "torch.serialization", "numpy.testing",
    "numpy.distutils", "numpy.f2py", "setuptools", "distutils", "pip", "tempfile", "glob", "io",
    "_io", "fileinput", "linecache", "gc", "inspect", "types", "ast", "timeit", "trace", "sysconfig",
)
# Network access.
_HIGH_ROOTS = (
    "webbrowser", "http", "httplib", "urllib", "urllib2", "urllib3", "requests", "aiohttp", "ftplib",
    "telnetlib", "smtplib", "poplib", "imaplib", "xmlrpc", "socketserver", "paramiko", "boto3",
)
# Callables that execute code or load further (unscanned) serialized data, even inside allowed roots.
_DANGEROUS_NAMES = frozenset(
    {
        "eval", "exec", "execfile", "system", "__import__", "runstring", "exec_command", "load",
        "loads", "read_pickle", "load_library", "getattr", "attrgetter", "methodcaller", "call",
        "check_call", "check_output", "run", "Popen", "spawn", "fork", "compile",
    }
)
_DANGEROUS_NAME_PREFIXES = ("popen", "spawn", "exec", "system")


def _under(module: str, roots: Sequence[str]) -> str | None:
    for r in roots:
        if module == r or module.startswith(r + "."):
            return r
    return None


def classify_pickle_global(module: str, name: str) -> tuple[str | None, str]:
    """Classify a global a pickle imports. Returns ``(severity | None, reason)``.

    ``None`` means allowlisted (plain data / model classes). ``CRITICAL`` = code execution, process,
    file-system or deserialization gadget; ``HIGH`` = network access; ``MEDIUM`` = outside the
    allowlist (e.g. a custom class — its ``__reduce__``/``__setstate__`` could run code on load).
    """
    module = str(module or "")
    name = str(name or "")
    leaf = name.rsplit(".", 1)[-1]
    if not module or not name or module == "<unknown>" or name == "<unknown>":
        return "HIGH", "global could not be resolved statically (obfuscated STACK_GLOBAL)"
    if module in ("builtins", "__builtin__"):
        if name in _SAFE_BUILTIN_NAMES:
            return None, "plain builtin type"
        if name in _DANGEROUS_BUILTIN_NAMES:
            return "CRITICAL", f"builtins.{name} can execute code or walk attributes"
        return "MEDIUM", f"builtin {name!r} is outside malvalid's allowlist"
    if (module, name) in _EXACT_ALLOWED:
        return None, "allowlisted helper"
    if (module, name) in {("operator", "attrgetter"), ("operator", "methodcaller")}:
        return "CRITICAL", f"{module}.{name} is a known attribute-walking gadget"
    root = _under(module, _CRITICAL_ROOTS)
    if root is not None:
        return "CRITICAL", f"module {root!r} gives code execution, process, file or deserialization access"
    root = _under(module, _HIGH_ROOTS)
    if root is not None:
        return "HIGH", f"module {root!r} gives network access"
    if leaf in _DANGEROUS_NAMES or leaf.lower().startswith(_DANGEROUS_NAME_PREFIXES):
        return "CRITICAL", f"{module}.{name} executes code or loads further serialized data"
    if _under(module, _ALLOWED_ROOTS) is not None:
        return None, "allowlisted model/numeric library"
    return "MEDIUM", f"module {module!r} is outside malvalid's allowlist (custom classes can run code on load)"


# --------------------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------------------


@dataclass
class Finding:
    """One issue in one artifact. ``sources`` lists every scanner that reported it."""

    severity: str
    description: str
    sources: list[str]
    module: str | None = None
    operator: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "description": self.description,
            "sources": list(self.sources),
            "module": self.module,
            "operator": self.operator,
            "detail": self.detail,
        }


@dataclass
class ArtifactReport:
    path: str
    sha256: str | None
    size: int | None
    format: str
    is_pickle: bool
    container: str | None = None  # "pickle", "zlib", "gzip", "bz2", "xz", "zip", "npy-object"
    declared: bool = True  # False => undeclared file found in the adapter directory
    scanned: bool = False  # modelscan scanned it
    scan_reason: str | None = None  # why modelscan did not scan it
    scanned_as: str | None = None  # how modelscan was pointed at it (e.g. "decompressed pickle")
    findings: list[Finding] = field(default_factory=list)
    opcode_scan: dict[str, Any] = field(default_factory=dict)
    policy_violation: str | None = None  # e.g. pickle without --allow-pickle
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name

    def max_severity(self) -> str | None:
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: SEVERITY_RANK[f.severity]).severity

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "format": self.format,
            "is_pickle": self.is_pickle,
            "container": self.container,
            "declared": self.declared,
            "scanned": self.scanned,
            "scan_reason": self.scan_reason,
            "scanned_as": self.scanned_as,
            "findings": [f.to_dict() for f in self.findings],
            "opcode_scan": to_jsonable(self.opcode_scan),
            "policy_violation": self.policy_violation,
            "notes": list(self.notes),
            "error": self.error,
        }


@dataclass
class ScanReport:
    artifacts: list[ArtifactReport]
    allow_pickle: bool
    scanner: str = "modelscan"
    scanner_version: str | None = None
    scanner_error: str | None = None  # modelscan could not be imported
    duration_s: float = 0.0

    def findings(self) -> Iterator[tuple[ArtifactReport, Finding]]:
        for a in self.artifacts:
            for f in a.findings:
                yield a, f

    def counts(self) -> dict[str, int]:
        c = {s: 0 for s in SEVERITIES}
        for _, f in self.findings():
            c[f.severity] += 1
        return c

    def count_at_or_above(self, severity: str) -> int:
        rank = SEVERITY_RANK[severity]
        return sum(1 for _, f in self.findings() if SEVERITY_RANK[f.severity] >= rank)

    @property
    def n_pickle(self) -> int:
        return sum(1 for a in self.artifacts if a.is_pickle)

    @property
    def policy_violations(self) -> list[ArtifactReport]:
        return [a for a in self.artifacts if a.policy_violation]

    @property
    def abort(self) -> bool:
        """True if the run must stop before the model is deserialized."""
        return self.counts()["CRITICAL"] > 0 or bool(self.policy_violations)

    def abort_reasons(self) -> list[str]:
        out: list[str] = []
        for a, f in self.findings():
            if f.severity == "CRITICAL":
                what = f"{f.module}.{f.operator}" if f.module and f.operator else f.description
                out.append(f"{a.name}: CRITICAL — {what}")
        for a in self.policy_violations:
            out.append(f"{a.name}: {a.policy_violation}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanner": self.scanner,
            "scanner_version": self.scanner_version,
            "scanner_error": self.scanner_error,
            "allow_pickle": self.allow_pickle,
            "counts": self.counts(),
            "n_pickle": self.n_pickle,
            "abort": self.abort,
            "abort_reasons": self.abort_reasons(),
            "duration_s": self.duration_s,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


# --------------------------------------------------------------------------------------------------
# Format detection (never deserializes)
# --------------------------------------------------------------------------------------------------

_SAFE_FORMATS = frozenset({"lightgbm-text", "json", "ubj", "onnx-protobuf", "numpy"})
FORMAT_DESCRIPTIONS = {
    "pickle": "Python pickle",
    "joblib-compressed": "compressed pickle container (joblib)",
    "zip-pickle": "zip archive containing pickles (e.g. PyTorch)",
    "zip": "zip archive",
    "json": "JSON model (XGBoost save_model / LightGBM dump_model)",
    "ubj": "XGBoost UBJSON model",
    "lightgbm-text": "LightGBM text model",
    "onnx-protobuf": "ONNX protobuf model",
    "numpy": "numpy .npy array",
    "compressed": "compressed file (not a pickle)",
    "directory": "directory",
    "unknown": "unrecognized format",
}
_PROTO0_START = frozenset(b"(cdl]})NIFSVLiKJGMTUX")


def _decompressor_for(path: Path) -> tuple[str, Any] | None:
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:3] == b"\x1f\x8b\x08":
        return "gzip", gzip.open
    if head[:3] == b"BZh":
        return "bz2", bz2.open
    if head[:6] == b"\xfd7zXZ\x00":
        return "xz", lzma.open
    if head[:1] == b"\x78" and len(head) > 1 and head[1] in (0x01, 0x5E, 0x9C, 0xDA):
        return "zlib", _ZlibReader
    return None


class _ZlibReader(io.RawIOBase):
    """Streaming reader over a zlib-compressed file (joblib's default compressor)."""

    def __init__(self, path: Path | str, mode: str = "rb") -> None:
        super().__init__()
        self._f = open(path, "rb")
        self._d = zlib.decompressobj()
        self._buf = b""
        self._eof = False
        self._pos = 0

    def readable(self) -> bool:
        return True

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            chunks = [self._buf]
            self._buf = b""
            while not self._eof:
                chunks.append(self._more())
            out = b"".join(chunks)
            self._pos += len(out)
            return out
        while len(self._buf) < n and not self._eof:
            self._buf += self._more()
        out, self._buf = self._buf[:n], self._buf[n:]
        self._pos += len(out)
        return out

    def _more(self) -> bytes:
        raw = self._f.read(1 << 20)
        if not raw:
            self._eof = True
            return self._d.flush()
        out = self._d.decompress(raw)
        if self._d.eof:
            self._eof = True
        return out

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._f.close()
        super().close()


def _looks_like_pickle_head(head: bytes) -> bool:
    return (len(head) > 1 and head[:1] == b"\x80" and 2 <= head[1] <= 5) or (
        len(head) > 0 and head[0] in _PROTO0_START
    )


def _parses_to_stop(f: IO[bytes], *, size_hint: int | None = None) -> bool:
    """True if ``f`` parses as pickle opcodes up to a STOP that ends (or nearly ends) the stream."""
    try:
        last = None
        for op, _arg, pos in pickletools.genops(f):
            last = (op.name, pos)
        if last is None or last[0] != "STOP":
            return False
        if size_hint is not None and last[1] is not None:
            return size_hint - (last[1] + 1) <= 16
        return True
    except Exception:  # noqa: BLE001 - not a pickle
        return False


def _npy_object_offset(path: Path) -> int | None:
    """If ``path`` is an .npy file holding an object array (i.e. a pickle), the data offset."""
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                _, _, dtype = np.lib.format.read_array_header_1_0(f)
            else:
                _, _, dtype = np.lib.format.read_array_header_2_0(f)
            return f.tell() if dtype.hasobject else None
    except Exception:  # noqa: BLE001
        return None


def _zip_pickle_members(path: Path, limit: int = 2000) -> list[str]:
    out: list[str] = []
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist()[:limit]:
                if info.is_dir():
                    continue
                with z.open(info) as m:
                    head = m.read(8)
                if info.filename.endswith((".pkl", ".pickle")) or (
                    head[:1] == b"\x80" and len(head) > 1 and 2 <= head[1] <= 5
                ):
                    out.append(info.filename)
                elif head[:6] == b"\x93NUMPY":
                    with z.open(info) as m:
                        data = m.read(4096)
                    if b"'|O'" in data[:512] or b'"|O"' in data[:512]:
                        out.append(info.filename)
    except (zipfile.BadZipFile, OSError, RuntimeError):
        return out
    return out


def detect_format(path: str | Path) -> tuple[str, bool, str | None]:
    """Content-based ``(format, is_pickle, container)``; never deserializes.

    ``is_pickle`` is true when the content is (or wraps) a pickle, or when the file has a pickle
    extension and its content is not a recognized non-executable format.
    """
    p = Path(path)
    if p.is_dir():
        return "directory", False, None
    fmt = sniff_format(p)
    suffix_pickle = p.suffix.lower() in PICKLE_EXTENSIONS
    if fmt == "pickle":
        return fmt, True, "pickle"
    if fmt == "zip-pickle":
        return fmt, True, "zip"
    if fmt == "zip":
        members = _zip_pickle_members(p)
        return ("zip-pickle", True, "zip") if members else (fmt, suffix_pickle, "zip" if suffix_pickle else None)
    if fmt == "joblib-compressed":
        dec = _decompressor_for(p)
        head = b""
        if dec is not None:
            with contextlib.suppress(Exception), dec[1](p, "rb") as f:
                head = f.read(64)
        if _looks_like_pickle_head(head):
            return fmt, True, dec[0] if dec else None
        return "compressed", suffix_pickle, dec[0] if dec else None
    if fmt == "numpy":
        if _npy_object_offset(p) is not None:
            return fmt, True, "npy-object"
        return fmt, False, None
    if fmt in _SAFE_FORMATS:
        return fmt, False, None
    # Unknown content: protocol 0/1 pickles have no magic number.
    try:
        with open(p, "rb") as f:
            head = f.read(1)
            if head and head[0] in _PROTO0_START:
                f.seek(0)
                if _parses_to_stop(f, size_hint=p.stat().st_size):
                    return "pickle", True, "pickle"
    except OSError:
        pass
    return fmt, suffix_pickle, "pickle" if suffix_pickle else None


# --------------------------------------------------------------------------------------------------
# Static pickle-opcode scan
# --------------------------------------------------------------------------------------------------

_STRING_OPS = frozenset(
    {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "STRING", "BINSTRING", "SHORT_BINSTRING"}
)
_MEMO_PUT = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
_MEMO_GET = frozenset({"GET", "BINGET", "LONG_BINGET"})
_CALL_OPS = frozenset({"REDUCE", "OBJ", "INST", "NEWOBJ", "NEWOBJ_EX", "BUILD"})

_GLOBAL_RE = re.compile(rb"(?:^|[\x00-\xff])[ci]([A-Za-z_][\w.]{0,255})\n([A-Za-z_][\w.]{0,255})\n")
_STACK_GLOBAL_RE = re.compile(
    rb"\x8c([\x01-\xff])([\w.]{1,255})\x94?(?:\x8c([\x01-\xff])([\w.]{1,255})\x94?)\x93"
)


@dataclass
class _Global:
    module: str
    name: str
    opcode: str
    offset: int | None
    via: str  # "opcodes" | "byte-sweep"


def _genops_globals(f: IO[bytes]) -> tuple[list[_Global], dict[str, Any]]:
    """Walk every pickle in ``f`` sequentially; stop at the first unparseable byte."""
    globals_: list[_Global] = []
    stats: dict[str, Any] = {"n_ops": 0, "n_pickles": 0, "n_calls": 0, "complete": False, "error": None}
    memo: dict[Any, Any] = {}
    stack: list[Any] = []  # approximation of recently pushed values (strings matter)
    try:
        while True:
            saw_op = False
            for op, arg, pos in pickletools.genops(f):
                saw_op = True
                stats["n_ops"] += 1
                name = op.name
                if name in _STRING_OPS:
                    stack.append(arg if isinstance(arg, str) else (arg.decode("latin-1") if isinstance(arg, bytes) else arg))
                elif name == "MEMOIZE":
                    memo[len(memo)] = stack[-1] if stack else None
                elif name in _MEMO_PUT:
                    memo[arg] = stack[-1] if stack else None
                elif name in _MEMO_GET:
                    stack.append(memo.get(arg))
                elif name in ("GLOBAL", "INST"):
                    mod, _, nm = str(arg).partition(" ")
                    globals_.append(_Global(mod, nm, name, pos, "opcodes"))
                    stack.append(None)
                    if name == "INST":
                        stats["n_calls"] += 1
                elif name == "STACK_GLOBAL":
                    nm = stack[-1] if len(stack) >= 1 else None
                    mod = stack[-2] if len(stack) >= 2 else None
                    globals_.append(
                        _Global(
                            mod if isinstance(mod, str) else "<unknown>",
                            nm if isinstance(nm, str) else "<unknown>",
                            name,
                            pos,
                            "opcodes",
                        )
                    )
                    stack.append(None)
                else:
                    if name in _CALL_OPS:
                        stats["n_calls"] += 1
                    if op.stack_after:
                        stack.append(None)
                if len(stack) > 64:
                    del stack[:-32]
            if not saw_op:
                break
            stats["n_pickles"] += 1
            nxt = f.read(1)
            if not nxt:
                stats["complete"] = True
                break
            # Another pickle follows (joblib / multi-dump files); rewind one byte if possible.
            try:
                f.seek(-1, os.SEEK_CUR)
            except (OSError, io.UnsupportedOperation):
                stats["error"] = "trailing data after STOP on a non-seekable stream"
                break
    except Exception as e:  # noqa: BLE001 - raw data inside the stream (e.g. joblib arrays)
        stats["error"] = f"{type(e).__name__}: {e}"
        stats["stopped_at"] = _safe_tell(f)
    return globals_, stats


def _safe_tell(f: IO[bytes]) -> int | None:
    try:
        return int(f.tell())
    except Exception:  # noqa: BLE001
        return None


def _sweep_globals(buf: Any) -> list[_Global]:
    """Regex sweep for GLOBAL/INST and STACK_GLOBAL string pairs anywhere in ``buf``."""
    out: list[_Global] = []
    for m in _GLOBAL_RE.finditer(buf):
        out.append(_Global(m.group(1).decode("latin-1"), m.group(2).decode("latin-1"), "GLOBAL", m.start(1) - 1, "byte-sweep"))
    for m in _STACK_GLOBAL_RE.finditer(buf):
        l1, s1, l2, s2 = m.group(1)[0], m.group(2), m.group(3)[0], m.group(4)
        if l1 == len(s1) and l2 == len(s2):
            out.append(_Global(s1.decode("latin-1"), s2.decode("latin-1"), "STACK_GLOBAL", m.start(), "byte-sweep"))
    return out


def _scan_stream(open_stream: Any, sweep_source: Any | None) -> dict[str, Any]:
    """Opcode scan of one pickle stream (+ sweep when the parse is incomplete)."""
    with open_stream() as f:
        globals_, stats = _genops_globals(f)
    swept = 0
    if not stats["complete"] and sweep_source is not None:
        with sweep_source() as buf:
            if buf is not None:
                seen = {(g.module, g.name) for g in globals_}
                for g in _sweep_globals(buf):
                    if (g.module, g.name) not in seen:
                        seen.add((g.module, g.name))
                        globals_.append(g)
                        swept += 1
                stats["swept"] = True
    stats["n_swept_globals"] = swept
    return {"globals": globals_, **stats}


@contextlib.contextmanager
def _mmap_file(path: Path, offset: int = 0) -> Iterator[Any]:
    if path.stat().st_size == 0:
        yield b""
        return
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        yield memoryview(mm)[offset:] if offset else mm


def _opcode_scan_artifact(p: Path, container: str | None, tmp: Path) -> tuple[dict[str, Any], Path | None]:
    """Run the static scan for one artifact. Returns (scan summary, decompressed temp path or None)."""
    decompressed: Path | None = None
    results: list[tuple[str, dict[str, Any]]] = []
    if container in (None, "pickle"):
        results.append(("", _scan_stream(lambda: open(p, "rb"), lambda: _mmap_file(p))))
    elif container == "npy-object":
        off = _npy_object_offset(p) or 0

        def _open_npy() -> IO[bytes]:
            f = open(p, "rb")
            f.seek(off)
            return f

        results.append(("", _scan_stream(_open_npy, lambda: _mmap_file(p, off))))
    elif container in ("gzip", "bz2", "xz", "zlib"):
        dec = _decompressor_for(p)
        if dec is None:
            raise ValueError("unrecognized compression")
        decompressed = tmp / (p.name + ".decompressed.pkl")
        total = 0
        with dec[1](p, "rb") as src, open(decompressed, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DECOMPRESSED_BYTES:
                    raise OverflowError(
                        f"decompressed stream exceeds {MAX_DECOMPRESSED_BYTES >> 30} GiB (possible decompression bomb)"
                    )
                dst.write(chunk)
        dp = decompressed
        results.append(("", _scan_stream(lambda: open(dp, "rb"), lambda: _mmap_file(dp))))
    elif container == "zip":
        members = _zip_pickle_members(p)
        with zipfile.ZipFile(p) as z:
            for mname in members:
                info = z.getinfo(mname)

                def _open_member(z: zipfile.ZipFile = z, mname: str = mname) -> IO[bytes]:
                    m = z.open(mname)
                    head = m.read(6)
                    if head == b"\x93NUMPY":  # object .npy member: skip the header
                        m.close()
                        raw = z.read(mname)
                        bio = io.BytesIO(raw)
                        np.lib.format.read_magic(bio)
                        np.lib.format.read_array_header_1_0(bio)
                        return io.BytesIO(raw[bio.tell():])
                    m.close()
                    return z.open(mname)

                @contextlib.contextmanager
                def _member_buf(z: zipfile.ZipFile = z, info: zipfile.ZipInfo = info) -> Iterator[Any]:
                    yield z.read(info) if info.file_size <= MAX_MEMBER_SWEEP_BYTES else None

                results.append((mname, _scan_stream(_open_member, _member_buf)))
    summary: dict[str, Any] = {
        "ran": True,
        "complete": all(r["complete"] for _, r in results) if results else False,
        "n_ops": sum(r["n_ops"] for _, r in results),
        "n_pickles": sum(r["n_pickles"] for _, r in results),
        "n_calls": sum(r["n_calls"] for _, r in results),
        "errors": [f"{m or p.name}: {r['error']}" for m, r in results if r.get("error")],
        "members": [m for m, _ in results if m],
        "globals": [],
    }
    for member, r in results:
        for g in r["globals"]:
            summary["globals"].append((member, g))
    return summary, decompressed


# --------------------------------------------------------------------------------------------------
# modelscan
# --------------------------------------------------------------------------------------------------


def modelscan_version() -> tuple[str | None, str | None]:
    """``(version, error)`` for the installed modelscan."""
    try:
        from modelscan._version import __version__

        return str(__version__), None
    except Exception as e:  # noqa: BLE001
        return None, f"modelscan is not importable ({type(e).__name__}: {e})"


def _modelscan_extensions() -> tuple[set[str], set[str]]:
    from modelscan.settings import DEFAULT_SETTINGS

    exts: set[str] = set()
    for s in DEFAULT_SETTINGS["scanners"].values():
        exts.update(s.get("supported_extensions", []))
    return exts, set(DEFAULT_SETTINGS.get("supported_zip_extensions", []))


def _run_modelscan(target: Path) -> dict[str, Any]:
    """modelscan's Python API on one file; returns its JSON-style report dict."""
    import copy

    from modelscan.modelscan import ModelScan
    from modelscan.settings import DEFAULT_SETTINGS

    ms_log = logging.getLogger("modelscan")
    old = ms_log.level
    ms_log.setLevel(logging.CRITICAL)
    try:
        return ModelScan(settings=copy.deepcopy(DEFAULT_SETTINGS)).scan(str(target))
    finally:
        ms_log.setLevel(old)


def _modelscan_artifact(a: ArtifactReport, p: Path, decompressed: Path | None, tmp: Path) -> list[Finding]:
    exts, zip_exts = _modelscan_extensions()
    suffix = p.suffix.lower()
    target: Path | None = None
    if decompressed is not None:
        target, a.scanned_as = decompressed, "decompressed pickle stream"
    elif a.container == "zip" and suffix not in exts | zip_exts:
        target = tmp / (p.name + ".zip")
        link_or_copy(p, target)  # symlink; hard link or copy where symlinks are unavailable (Windows)
        a.scanned_as = "zip archive (by content)"
    elif a.container in ("pickle",) and suffix not in exts:
        target = tmp / (p.name + ".pkl")
        link_or_copy(p, target)  # symlink; hard link or copy where symlinks are unavailable (Windows)
        a.scanned_as = "pickle (by content)"
    elif a.container == "npy-object" and suffix != ".npy":
        target = tmp / (p.name + ".npy")
        link_or_copy(p, target)  # symlink; hard link or copy where symlinks are unavailable (Windows)
        a.scanned_as = "numpy object array (by content)"
    elif suffix in exts | zip_exts:
        target = p
        a.scanned_as = f"{suffix} file"
    if target is None:
        a.scanned = False
        desc = FORMAT_DESCRIPTIONS.get(a.format, a.format)
        a.scan_reason = f"modelscan has no scanner for {desc} files"
        return []
    rep = _run_modelscan(target)
    summary = rep.get("summary", {})
    n_scanned = int(summary.get("scanned", {}).get("total_scanned", 0) or 0)
    errors = rep.get("errors") or []
    skipped = summary.get("skipped", {}).get("skipped_files", []) or []
    findings = [
        Finding(
            severity=str(i.get("severity", "MEDIUM")).upper() if str(i.get("severity", "")).upper() in SEVERITY_RANK else "MEDIUM",
            description=str(i.get("description", "")),
            sources=["modelscan"],
            module=i.get("module"),
            operator=i.get("operator"),
            detail=i.get("scanner"),
        )
        for i in rep.get("issues") or []
    ]
    a.scanned = n_scanned > 0 or bool(findings)
    if not a.scanned:
        why = "; ".join(str(e.get("description", e)) for e in errors if isinstance(e, dict)) or "; ".join(
            f"{s.get('category')}: {s.get('description')}" for s in skipped if isinstance(s, dict)
        )
        a.scan_reason = f"modelscan did not scan it ({why or 'no scanner matched'})"
    elif errors:
        a.notes.append(
            "modelscan reported errors while scanning: "
            + "; ".join(str(e.get("description", e)) for e in errors if isinstance(e, dict))
        )
    return findings


# --------------------------------------------------------------------------------------------------
# scan_artifacts
# --------------------------------------------------------------------------------------------------


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _merge_findings(findings: list[Finding]) -> list[Finding]:
    merged: dict[tuple[Any, ...], Finding] = {}
    order: list[tuple[Any, ...]] = []
    for f in findings:
        key = (f.module, f.operator) if f.module and f.operator else (f.description,)
        if key in merged:
            m = merged[key]
            if SEVERITY_RANK[f.severity] > SEVERITY_RANK[m.severity]:
                m.severity = f.severity
            for s in f.sources:
                if s not in m.sources:
                    m.sources.append(s)
            if f.detail and f.detail not in (m.detail or ""):
                m.detail = f"{m.detail}; {f.detail}" if m.detail else f.detail
        else:
            merged[key] = Finding(f.severity, f.description, list(f.sources), f.module, f.operator, f.detail)
            order.append(key)
    return sorted((merged[k] for k in order), key=lambda f: -SEVERITY_RANK[f.severity])


def _expand(paths: Sequence[str | Path], limit: int = 2000) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file() and "__pycache__" not in sub.parts:
                    out.append(sub)
                    if len(out) >= limit:
                        break
        else:
            out.append(p)
    return out


def _scan_one(p: Path, *, allow_pickle: bool, declared: bool, tmp: Path, use_modelscan: bool) -> ArtifactReport:
    if not p.exists():
        return ArtifactReport(
            path=str(p), sha256=None, size=None, format="missing", is_pickle=False, declared=declared,
            scan_reason="file does not exist",
            findings=[Finding("HIGH", f"declared artifact {p} does not exist; it cannot be verified", ["malvalid-policy"])],
            error="not found",
        )
    fmt, is_pickle, container = detect_format(p)
    a = ArtifactReport(
        path=str(p.resolve()), sha256=_sha256(p), size=p.stat().st_size, format=fmt, is_pickle=is_pickle,
        container=container, declared=declared,
    )
    findings: list[Finding] = []
    decompressed: Path | None = None
    if is_pickle:
        try:
            summary, decompressed = _opcode_scan_artifact(p, container, tmp)
        except OverflowError as e:
            summary = {"ran": False, "error": str(e)}
            findings.append(Finding("HIGH", str(e), ["malvalid-opcodes"]))
        except Exception as e:  # noqa: BLE001 - a scan failure must never pass silently
            summary = {"ran": False, "error": f"{type(e).__name__}: {e}"}
            findings.append(
                Finding("MEDIUM", f"malvalid's opcode scan could not parse the pickle ({type(e).__name__}: {e})", ["malvalid-opcodes"])
            )
        globals_ = summary.pop("globals", []) if isinstance(summary.get("globals"), list) else []
        listed: list[dict[str, Any]] = []
        for member, g in globals_:
            sev, reason = classify_pickle_global(g.module, g.name)
            listed.append(
                {"module": g.module, "name": g.name, "opcode": g.opcode, "offset": g.offset, "via": g.via,
                 "member": member or None, "severity": sev}
            )
            if sev is not None:
                where = f" in member {member}" if member else ""
                via = " (found by byte-pattern sweep past non-pickle data)" if g.via == "byte-sweep" else ""
                findings.append(
                    Finding(
                        sev,
                        f"pickle imports {g.module}.{g.name}{where}: {reason}{via}",
                        ["malvalid-opcodes" if g.via == "opcodes" else "malvalid-byte-sweep"],
                        module=g.module,
                        operator=g.name,
                    )
                )
        summary["globals"] = listed[:500]
        summary["n_globals"] = len(listed)
        a.opcode_scan = summary
    else:
        a.opcode_scan = {"ran": False, "reason": "not a pickle-based artifact"}
    if use_modelscan:
        try:
            findings.extend(_modelscan_artifact(a, p, decompressed, tmp))
        except Exception as e:  # noqa: BLE001
            a.scanned = False
            a.scan_reason = f"modelscan failed: {type(e).__name__}: {e}"
    else:
        a.scanned = False
        a.scan_reason = "modelscan is not installed"
    if decompressed is not None:
        with contextlib.suppress(OSError):
            decompressed.unlink()
    if not is_pickle:
        if fmt in _SAFE_FORMATS:
            a.notes.append(f"{FORMAT_DESCRIPTIONS.get(fmt, fmt)}: a data format that does not execute code on load")
        elif fmt in ("unknown", "compressed", "zip"):
            findings.append(
                Finding(
                    "LOW",
                    f"{p.name}: {FORMAT_DESCRIPTIONS.get(fmt, fmt)}; malvalid could not confirm it is non-executable",
                    ["malvalid-format"],
                )
            )
    if is_pickle and not allow_pickle:
        kind = {"zlib": "zlib-compressed joblib", "gzip": "gzip-compressed", "bz2": "bz2-compressed",
                "xz": "xz-compressed", "zip": "zip-wrapped", "npy-object": "numpy object-array"}.get(container or "", "")
        what = f"{kind} pickle" if kind else "pickle"
        if declared:
            a.policy_violation = (
                f"{p.name} is a {what}-based artifact and --allow-pickle was not given. Pickles can execute "
                f"arbitrary code when loaded, so malvalid refuses them by default. {REEXPORT_HINT}"
            )
        else:
            findings.append(
                Finding(
                    "MEDIUM",
                    f"undeclared {what} file {p.name} in the adapter directory; the sandbox will refuse to "
                    "unpickle it without --allow-pickle (remove it, or declare it as model_path)",
                    ["malvalid-policy"],
                )
            )
    a.findings = _merge_findings(findings)
    return a


def scan_artifacts(
    paths: Sequence[str | Path],
    *,
    allow_pickle: bool,
    undeclared: Sequence[str | Path] = (),
) -> ScanReport:
    """Scan model artifacts without deserializing them.

    ``paths`` are the model artifacts the adapter declares (directories are expanded). ``undeclared``
    are other model-like files found next to the adapter: they are scanned the same way, but a pickle
    among them is reported as a finding rather than a policy violation (the sandbox refuses to load
    it anyway without ``--allow-pickle``).
    """
    t0 = time.monotonic()
    version, err = modelscan_version()
    reports: list[ArtifactReport] = []
    seen: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="malvalid-m0-") as td:
        tmp = Path(td)
        for declared, group in ((True, _expand(paths)), (False, _expand(undeclared))):
            for p in group:
                key = str(p.resolve()) if p.exists() else str(p)
                if key in seen:
                    continue
                seen.add(key)
                log.info("M0: scanning %s", p)
                reports.append(
                    _scan_one(p, allow_pickle=allow_pickle, declared=declared, tmp=tmp, use_modelscan=version is not None)
                )
    return ScanReport(
        artifacts=reports,
        allow_pickle=allow_pickle,
        scanner_version=version,
        scanner_error=err,
        duration_s=time.monotonic() - t0,
    )


# --------------------------------------------------------------------------------------------------
# Adapter-directory discovery
# --------------------------------------------------------------------------------------------------

_SKIP_DIRS = frozenset(
    {"__pycache__", ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules", ".mypy_cache",
     ".pytest_cache", ".ruff_cache", "site-packages", ".ipynb_checkpoints", ".tox", "build", "dist"}
)
_STRONG_MODEL_EXT = frozenset(PICKLE_EXTENSIONS) | frozenset(
    {".model", ".lgb", ".ubj", ".onnx", ".h5", ".hdf5", ".keras", ".pb", ".safetensors", ".bst", ".xgb",
     ".cbm", ".tflite", ".npy", ".npz"}
)
_TEXTISH_EXT = frozenset({".py", ".pyc", ".md", ".rst", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".csv",
                          ".tsv", ".sh", ".ipynb", ".log", ".html", ".css", ".js"})


def _json_is_model(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            head = f.read(1 << 16)
    except OSError:
        return False
    return any(m in head for m in (b'"learner"', b'"tree_info"', b'"max_feature_idx"', b'"num_tree_per_iteration"'))


def is_model_like(p: Path) -> bool:
    """Heuristic: is ``p`` a serialized model artifact (by extension or by content)?"""
    suf = p.suffix.lower()
    if suf in _STRONG_MODEL_EXT:
        return True
    if suf in _TEXTISH_EXT:
        return False
    if suf == ".json":
        return _json_is_model(p)
    fmt = sniff_format(p)
    if fmt == "lightgbm-text":
        return True
    if fmt in ("pickle", "zip-pickle"):
        return True
    if fmt in ("joblib-compressed", "zip", "unknown"):
        return detect_format(p)[1]
    return False


def discover_model_files(adapter_dir: str | Path, *, max_depth: int = 3, max_files: int = 5000) -> list[Path]:
    """Model-like files under ``adapter_dir`` (skipping VCS/venv/cache dirs)."""
    root = Path(adapter_dir)
    out: list[Path] = []
    visited = 0
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        depth = len(d.relative_to(root).parts)
        dirnames[:] = sorted(n for n in dirnames if n not in _SKIP_DIRS and not n.startswith(".") and depth < max_depth)
        for fn in sorted(filenames):
            visited += 1
            if visited > max_files:
                log.warning("M0: stopped discovering files in %s after %d entries", root, max_files)
                return out
            p = d / fn
            if fn.startswith(".") or not p.is_file():
                continue
            with contextlib.suppress(OSError):
                if is_model_like(p):
                    out.append(p)
    return out


_ADAPTER_PICKLE_CALL = re.compile(
    r"\b(?:pickle|cPickle|_pickle|dill|cloudpickle)\.loads?\s*\(|\bjoblib\.load\s*\(|\btorch\.load\s*\("
    r"|\b(?:np|numpy)\.load\s*\([^)]*allow_pickle\s*=\s*True|\b(?:pd|pandas)\.read_pickle\s*\("
)


def adapter_pickle_calls(adapter_path: str | Path) -> list[tuple[int, str]]:
    """Lines of the adapter source that call a pickle-based loader (informational)."""
    try:
        text = Path(adapter_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        m = _ADAPTER_PICKLE_CALL.search(line)
        if m and not line.lstrip().startswith("#"):
            out.append((i, m.group(0).rstrip("(").strip()))
    return out


# --------------------------------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------------------------------


class FileSafetyModule(Module):
    id = "file_safety"
    code = "M0"
    title = "Model file safety"
    description = (
        "Scans every model artifact with modelscan and malvalid's static pickle-opcode scan before "
        "anything is deserialized, and refuses pickle-based artifacts unless --allow-pickle is given. "
        "A CRITICAL finding (or a pickle without --allow-pickle) aborts the run before load."
    )
    requires = ()
    default_gate = GateMode.HARD
    default_params = {"fail_on": "HIGH", "scan_adapter_dir": True}

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        fail_on = str(params.get("fail_on", "HIGH")).upper()
        if fail_on not in SEVERITY_RANK:
            raise ConfigError(f"file_safety.fail_on must be one of {list(SEVERITIES)}, got {fail_on!r}")

    def run(self, ctx: Any) -> ModuleResult:
        fail_on = str(ctx.params.get("fail_on", "HIGH")).upper()
        if fail_on not in SEVERITY_RANK:
            raise ConfigError(f"file_safety.fail_on must be one of {list(SEVERITIES)}, got {fail_on!r}")
        decl = getattr(ctx.model, "declarations", None)
        extras = ctx.extras or {}
        declared_paths = [Path(p) for p in (extras.get("artifact_paths") or getattr(decl, "model_paths", ()) or ())]
        adapter_path = extras.get("adapter_path") or getattr(decl, "adapter_path", "") or ""
        allow_pickle = bool(extras.get("allow_pickle", getattr(getattr(ctx.config, "runtime", None), "allow_pickle", False)))
        adapter_dir = Path(adapter_path).parent if adapter_path else None
        notes: list[str] = []

        discovered: list[Path] = []
        if ctx.params.get("scan_adapter_dir", True) and adapter_dir is not None and adapter_dir.is_dir():
            discovered = discover_model_files(adapter_dir)
        declared_set = {p.resolve() for p in _expand(declared_paths) if p.exists()}
        if declared_paths:
            primary, stray = declared_paths, [p for p in discovered if p.resolve() not in declared_set]
        else:
            primary, stray = discovered, []
            if discovered:
                notes.append(
                    "The adapter does not declare model_path/model_paths; M0 scanned the model-like files it "
                    "found in the adapter directory instead. Declare model_path so the gate scans exactly "
                    "the file your adapter loads."
                )
        report = scan_artifacts(primary, allow_pickle=allow_pickle, undeclared=stray)
        counts = report.counts()
        n_art = len(report.artifacts)
        at_or_above = report.count_at_or_above(fail_on)
        violations = report.policy_violations

        checks = [
            GateCheck.evaluate(
                "critical_findings", counts["CRITICAL"], "==", 0, metric="n_critical",
                description="CRITICAL scanner findings (any one aborts the run before load)",
            ),
        ]
        if fail_on != "CRITICAL":
            checks.append(
                GateCheck.evaluate(
                    f"findings_at_or_above_{fail_on.lower()}", at_or_above, "==", 0,
                    metric=f"n_at_or_above_{fail_on.lower()}",
                    description=f"scanner findings with severity >= {fail_on} (file_safety.fail_on)",
                )
            )
        checks.append(
            GateCheck.evaluate(
                "pickle_without_allow_pickle", len(violations), "==", 0, metric="n_policy_violations",
                description="pickle-based model artifacts submitted without --allow-pickle",
            )
        )
        if n_art == 0:
            checks.append(
                GateCheck.evaluate(
                    "artifacts_scanned", None, ">=", 1, metric="n_artifacts",
                    description="at least one model artifact must be scanned",
                )
            )

        # ---- notes -------------------------------------------------------------------------------
        allowed_pickles = [a for a in report.artifacts if a.is_pickle and allow_pickle]
        if allowed_pickles:
            notes.insert(
                0,
                "PICKLE ACCEPTED (--allow-pickle): "
                + ", ".join(a.name for a in allowed_pickles)
                + " can execute arbitrary code when loaded. It was scanned and is loaded only inside the "
                "sandbox, where dangerous imports are still refused, but a production deployment would "
                "load it without those guards. Prefer a non-pickle format.",
            )
        if report.scanner_error:
            notes.append(f"{report.scanner_error}; only malvalid's own opcode scan ran.")
        for a in report.artifacts:
            if not a.scanned and a.scan_reason and not a.is_pickle and a.format in _SAFE_FORMATS:
                notes.append(f"{a.name}: {a.scan_reason}; it is a {FORMAT_DESCRIPTIONS.get(a.format, a.format)}, "
                             "which does not execute code on load.")
            elif not a.scanned and a.scan_reason and a.is_pickle:
                notes.append(f"{a.name}: {a.scan_reason}; malvalid's opcode scan still ran.")
            notes.extend(f"{a.name}: {n}" for n in a.notes if "does not execute code" not in n)
        if adapter_path:
            calls = adapter_pickle_calls(adapter_path)
            if calls and not allow_pickle:
                notes.append(
                    "The adapter source calls pickle-based loaders ("
                    + ", ".join(f"line {ln}: {c}" for ln, c in calls[:5])
                    + "); without --allow-pickle the sandbox refuses them at load time."
                )

        # ---- verdict text ------------------------------------------------------------------------
        abort = report.abort
        medium_or_low = [f for _, f in report.findings() if SEVERITY_RANK[f.severity] < SEVERITY_RANK[fail_on]]
        if abort:
            crit = [(a, f) for a, f in report.findings() if f.severity == "CRITICAL"]
            if crit:
                a, f = crit[0]
                what = f"{f.module}.{f.operator}" if f.module and f.operator else f.description
                finding = (
                    f"CRITICAL: {a.name} contains unsafe serialized code ({what}"
                    + (f" and {len(crit) - 1} more" if len(crit) > 1 else "")
                    + "). The run was aborted before the model was ever deserialized — do not load this file."
                )
            else:
                a = violations[0]
                finding = (
                    f"{a.name} is a pickle-based artifact and --allow-pickle was not given, so the run was "
                    f"aborted before loading it. {REEXPORT_HINT}"
                )
        elif at_or_above:
            worst = [(a, f) for a, f in report.findings() if SEVERITY_RANK[f.severity] >= SEVERITY_RANK[fail_on]]
            a, f = worst[0]
            finding = (
                f"{at_or_above} finding(s) at or above {fail_on} (e.g. {a.name}: {f.description}). "
                "The file-safety gate failed."
            )
        elif n_art == 0:
            finding = (
                "No model artifact was found to scan. Declare model_path (or model_paths) on your adapter so "
                "M0 can scan the file before it is loaded."
            )
        elif allowed_pickles:
            finding = (
                f"Scanned {n_art} artifact(s); no dangerous imports found. "
                f"{allowed_pickles[0].name} is a pickle accepted via --allow-pickle — it can still run code "
                "on load; prefer a non-pickle format for production."
            )
        else:
            kinds = ", ".join(f"{a.name}: {FORMAT_DESCRIPTIONS.get(a.format, a.format)}" for a in report.artifacts[:3])
            finding = f"Scanned {n_art} artifact(s) ({kinds}); no unsafe content found."
            if medium_or_low:
                finding += f" {len(medium_or_low)} lower-severity finding(s) are listed below."

        advisory = bool(allowed_pickles) or report.scanner_error is not None or any(
            SEVERITY_RANK[f.severity] >= SEVERITY_RANK["MEDIUM"] for f in medium_or_low
        )
        res = self.result(
            ctx,
            finding=finding,
            checks=checks,
            metrics={
                "n_artifacts": n_art,
                "n_pickle": report.n_pickle,
                "n_critical": counts["CRITICAL"],
                "n_high": counts["HIGH"],
                "n_medium": counts["MEDIUM"],
                "n_low": counts["LOW"],
                "n_policy_violations": len(violations),
                "n_modelscan_scanned": sum(1 for a in report.artifacts if a.scanned),
                "scanner": report.scanner,
                "scanner_version": report.scanner_version,
                "allow_pickle": allow_pickle,
                "fail_on": fail_on,
            },
            details={
                "abort": abort,
                "abort_reasons": report.abort_reasons(),
                "artifacts": [a.to_dict() for a in report.artifacts],
                "model_paths_declared": bool(declared_paths),
                "adapter_dir_scanned": bool(ctx.params.get("scan_adapter_dir", True) and adapter_dir is not None),
                "discovered_files": [str(p) for p in discovered],
                "scan_duration_s": round(report.duration_s, 3),
            },
            notes=notes,
            advisory_warn=advisory,
            score=0.0,  # replaced below
        )
        if res.gate_outcome is GateOutcome.FAILED or abort:
            res.score = 0.0
        elif res.gate_outcome is GateOutcome.NOT_EVALUATED and n_art == 0:
            res.score = None
        elif allowed_pickles:
            res.score = 0.75
        else:
            res.score = 1.0
        if abort and res.status is not Status.FAIL and ctx.gate is GateMode.HARD:
            res.status = Status.FAIL  # pragma: no cover - derive_status already yields FAIL
        self._add_tables(ctx, report)
        res.artifacts = ctx.artifacts.keys_for(self.id)
        return res

    def _add_tables(self, ctx: Any, report: ScanReport) -> None:
        rows = []
        for a in report.artifacts:
            rows.append(
                [
                    a.name,
                    FORMAT_DESCRIPTIONS.get(a.format, a.format),
                    "yes" if a.is_pickle else "no",
                    "yes" if a.declared else "no (found in adapter dir)",
                    (a.sha256 or "")[:16],
                    "scanned" if a.scanned else f"not scanned: {a.scan_reason or ''}",
                    a.max_severity() or "none",
                ]
            )
        ctx.artifacts.add_table(
            self.id, "artifacts", title="Scanned model artifacts",
            columns=["file", "format", "pickle", "declared", "sha256 (prefix)", "modelscan", "worst finding"],
            rows=rows,
        )
        frows = [
            [a.name, f.severity, f.description, ", ".join(f.sources)] for a, f in report.findings()
        ]
        if frows:
            ctx.artifacts.add_table(
                self.id, "findings", title="File-safety findings",
                columns=["file", "severity", "finding", "reported by"], rows=frows,
            )
