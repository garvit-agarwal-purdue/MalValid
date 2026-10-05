"""Streaming, size-capped parsing of the new-run form (``POST /runs``, ``POST /api/validate``).

Multipart bodies are parsed incrementally with ``python-multipart``: each file part is written
straight into the submission directory under a sanitized basename while its size is counted, so an
oversized upload is stopped at the cap (``--max-upload-mb``, per file) and its partial file deleted —
nothing is spooled to a temp dir first. Extensions are allowlisted per field, names are reduced to
``[A-Za-z0-9._-]``, and after parsing, pickle-based model files (by extension or content) are refused
unless the form's ``allow_pickle`` box is ticked. Nothing uploaded is ever imported, unpickled or
otherwise deserialized by the web server.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable
from urllib.parse import parse_qsl

from starlette.requests import Request

try:  # python-multipart >= 0.0.13 ships as ``python_multipart``
    from python_multipart.multipart import MultipartParser, parse_options_header
    from python_multipart.exceptions import MultipartParseError
except ImportError:  # pragma: no cover - older python-multipart
    from multipart.multipart import MultipartParser, parse_options_header  # type: ignore[no-redef]
    from multipart.exceptions import MultipartParseError  # type: ignore[no-redef]

MODEL_EXTENSIONS = (".txt", ".json", ".ubj", ".model", ".onnx", ".pkl", ".pickle", ".joblib", ".jbl")
MANIFEST_EXTENSIONS = (".txt", ".csv", ".tsv")
CONFIG_EXTENSIONS = (".yaml", ".yml")
FILE_FIELDS: dict[str, tuple[str, ...]] = {
    "model_file": MODEL_EXTENSIONS,
    "adapter_file": (".py",),
    "model_files": MODEL_EXTENSIONS,
    "manifest_file": MANIFEST_EXTENSIONS,
    "config_file": CONFIG_EXTENSIONS,
}
FIELD_LABELS = {
    "model_file": "model file",
    "adapter_file": "adapter",
    "model_files": "model file",
    "manifest_file": "training manifest",
    "config_file": "gate policy (config)",
}
SINGLE_FILE_FIELDS = frozenset({"model_file", "adapter_file", "manifest_file", "config_file"})
PICKLE_FORMATS = ("pickle", "joblib-compressed", "zip-pickle")

MAX_FIELDS = 200
MAX_FIELD_BYTES = 64 * 1024
MAX_FILES = 64
MAX_URLENCODED_BYTES = 1024 * 1024
#: After an error, keep reading (and discarding) at most this much so the browser gets the answer.
DRAIN_LIMIT_BYTES = 1024 * 1024 * 1024
MAX_NAME_LEN = 128

PICKLE_REFUSED = (
    "{name} is a pickle-based model file. Loading a pickle can run arbitrary code, so malvalid refuses "
    "pickles unless you tick \"Allow pickle\" — and even then the file is only loaded inside the sandbox, "
    "after the M0 file-safety scan. Prefer a format that cannot execute code: LightGBM .txt, "
    "XGBoost .json/.ubj, or ONNX."
)

_BAD_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(Exception):
    """The request body cannot be accepted (too large, malformed, bad CSRF token...)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def sanitize_filename(raw: str | None) -> str | None:
    """Basename reduced to ``[A-Za-z0-9._-]`` (runs of other characters become ``_``).

    Directories (``/`` or ``\\``) are stripped; empty and dot-leading names are rejected (None).
    """
    if raw is None:
        return None
    name = str(raw).replace("\\", "/").split("/")[-1].strip()
    if not name or name.startswith("."):
        return None
    name = _BAD_CHARS.sub("_", name)
    if not name or name.startswith(".") or set(name) <= {".", "_"}:
        return None
    if len(name) > MAX_NAME_LEN:
        stem, dot, ext = name.rpartition(".")
        if dot and 0 < len(ext) < 16:
            name = stem[: MAX_NAME_LEN - len(ext) - 1] + "." + ext
        else:
            name = name[:MAX_NAME_LEN]
    return name


def _suffix(name: str) -> str:
    i = name.rfind(".")
    return name[i:].lower() if i > 0 else ""


@dataclass
class SavedFile:
    field: str
    name: str  # sanitized basename
    original: str  # as sent by the client
    path: Path
    size: int = 0


@dataclass
class FormData:
    """Parsed form: text fields, saved files and per-file problems (each makes the form invalid)."""

    fields: dict[str, list[str]] = field(default_factory=dict)
    files: dict[str, list[SavedFile]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def get(self, name: str, default: str = "") -> str:
        vals = self.fields.get(name)
        return vals[-1] if vals else default

    def getlist(self, name: str) -> list[str]:
        return list(self.fields.get(name) or [])

    def file(self, name: str) -> SavedFile | None:
        fs = self.files.get(name) or []
        return fs[0] if fs else None

    def all_files(self) -> list[SavedFile]:
        return [f for fs in self.files.values() for f in fs]


@dataclass
class _Part:
    name: str = ""
    kind: str = "skip"  # field | file | skip
    disposition: bytes = b""
    buf: bytearray = field(default_factory=bytearray)
    fh: IO[bytes] | None = None
    saved: SavedFile | None = None


class _MultipartSink:
    """python-multipart callbacks: fields into memory (capped), files straight to ``dest``."""

    def __init__(self, dest: Path, *, max_file_bytes: int, csrf_check: Callable[[str], bool] | None,
                 max_fields: int = MAX_FIELDS, max_field_bytes: int = MAX_FIELD_BYTES, max_files: int = MAX_FILES):
        self.dest = dest
        self.max_file_bytes = max_file_bytes
        self.csrf_check = csrf_check
        self.max_fields = max_fields
        self.max_field_bytes = max_field_bytes
        self.max_files = max_files
        self.form = FormData()
        self.part = _Part()
        self._hname = b""
        self._hval = b""
        self._n_fields = 0
        self._n_files = 0
        self._names: set[str] = set()
        self.complete = False

    def callbacks(self) -> dict[str, Callable[..., None]]:
        return {
            "on_part_begin": self.on_part_begin,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_end": self.on_end,
        }

    # ---- headers -------------------------------------------------------------------------------

    def on_part_begin(self) -> None:
        self.part = _Part()
        self._hname = self._hval = b""

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._hname += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._hval += data[start:end]

    def on_header_end(self) -> None:
        if self._hname.strip().lower() == b"content-disposition":
            self.part.disposition = self._hval
        self._hname = self._hval = b""

    def on_headers_finished(self) -> None:
        _, options = parse_options_header(self.part.disposition)
        raw_name = options.get(b"name")
        if raw_name is None:
            raise UploadError(400, "malformed form upload (a part has no field name)")
        name = raw_name.decode("utf-8", "replace")
        self.part.name = name
        if b"filename" in options:
            self._begin_file(name, options[b"filename"].decode("utf-8", "replace"))
        else:
            self._n_fields += 1
            if self._n_fields > self.max_fields:
                raise UploadError(400, f"too many form fields (limit {self.max_fields})")
            self.part.kind = "field"

    def _begin_file(self, fld: str, filename: str) -> None:
        part = self.part
        if not filename or fld not in FILE_FIELDS or self.max_files == 0:
            # an empty <input type=file>, a field this form does not have, or a form that takes no
            # files at all (cancel/delete): the part is skipped, never written anywhere
            return
        label = FIELD_LABELS[fld]
        name = sanitize_filename(filename)
        if name is None:
            self.form.problems.append(f"{label}: {filename!r} is not a usable file name")
            return
        allowed = FILE_FIELDS[fld]
        suffix = _suffix(name)
        if suffix not in allowed:
            self.form.problems.append(f"{label} {name}: only {' '.join(allowed)} files are accepted here")
            return
        if fld == "adapter_file" and not name.endswith(".py"):
            name = name[: -len(suffix)] + ".py"
        if fld in SINGLE_FILE_FIELDS and self.form.files.get(fld):
            self.form.problems.append(f"only one {label} can be uploaded")
            return
        if name.lower() in self._names:
            self.form.problems.append(f"two uploaded files are both named {name}; rename one")
            return
        self._n_files += 1
        if self._n_files > self.max_files:
            raise UploadError(400, f"too many files (limit {self.max_files})")
        path = self.dest / name
        try:
            fh = open(path, "xb", buffering=1024 * 1024)
        except OSError as e:
            raise UploadError(500, f"could not store {name}: {e.strerror or e}") from e
        self._names.add(name.lower())
        part.kind = "file"
        part.fh = fh
        part.saved = SavedFile(field=fld, name=name, original=filename, path=path)

    # ---- data ----------------------------------------------------------------------------------

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        part = self.part
        n = end - start
        if part.kind == "field":
            if len(part.buf) + n > self.max_field_bytes:
                raise UploadError(400, f"form field {part.name!r} is too large")
            part.buf += data[start:end]
        elif part.kind == "file" and part.fh is not None and part.saved is not None:
            part.saved.size += n
            if part.saved.size > self.max_file_bytes:
                mb = self.max_file_bytes / (1024 * 1024)
                raise UploadError(
                    413,
                    f"{part.saved.name} is larger than the upload limit of {mb:g} MB per file "
                    "(start `malvalid serve` with a larger --max-upload-mb, or use path mode for files "
                    "already on this machine)",
                )
            part.fh.write(data[start:end])

    def on_part_end(self) -> None:
        part = self.part
        if part.kind == "field":
            value = bytes(part.buf).decode("utf-8", "replace")
            self.form.fields.setdefault(part.name, []).append(value)
            if part.name == "csrf" and self.csrf_check is not None and not self.csrf_check(value):
                raise UploadError(403, "missing or invalid CSRF token (reload the page and try again)")
        elif part.kind == "file" and part.fh is not None and part.saved is not None:
            part.fh.close()
            part.fh = None
            self.form.files.setdefault(part.saved.field, []).append(part.saved)
        self.part = _Part()

    def on_end(self) -> None:
        self.complete = True

    def close(self) -> None:
        fh = self.part.fh
        if fh is not None:
            with contextlib.suppress(OSError):
                fh.close()
            self.part.fh = None


async def _drain(stream: Any) -> None:
    drained = 0
    with contextlib.suppress(Exception):
        async for chunk in stream:
            drained += len(chunk)
            if drained > DRAIN_LIMIT_BYTES:
                break


async def parse_form(
    request: Request,
    dest: Path,
    *,
    max_file_bytes: int,
    csrf_check: Callable[[str], bool] | None = None,
    max_files: int = MAX_FILES,
) -> FormData:
    """Parse a ``multipart/form-data`` or urlencoded body; files are written into ``dest``.

    Raises :class:`UploadError` (status 400/403/413/415); the caller deletes ``dest`` then. The rest of
    the body is drained (up to :data:`DRAIN_LIMIT_BYTES`) so a browser still receives the error page.
    """
    ctype = request.headers.get("content-type") or ""
    kind, params = parse_options_header(ctype)
    kind_s = kind.decode("latin-1").lower() if isinstance(kind, bytes) else str(kind).lower()
    stream = request.stream()
    if kind_s == "application/x-www-form-urlencoded":
        body = bytearray()
        async for chunk in stream:
            body += chunk
            if len(body) > MAX_URLENCODED_BYTES:
                await _drain(stream)
                raise UploadError(413, "form too large")
        form = FormData()
        try:
            pairs = parse_qsl(bytes(body).decode("utf-8", "replace"), keep_blank_values=True,
                              max_num_fields=MAX_FIELDS)
        except ValueError as e:
            raise UploadError(400, f"malformed form: {e}") from e
        for k, v in pairs:
            form.fields.setdefault(k, []).append(v)
        if csrf_check is not None and "csrf" in form.fields and not csrf_check(form.get("csrf")):
            raise UploadError(403, "missing or invalid CSRF token (reload the page and try again)")
        return form
    if kind_s != "multipart/form-data":
        await _drain(stream)
        raise UploadError(415, "expected a form submission (multipart/form-data)")
    boundary = params.get(b"boundary")
    if not boundary:
        await _drain(stream)
        raise UploadError(400, "malformed form upload (no multipart boundary)")
    sink = _MultipartSink(dest, max_file_bytes=max_file_bytes, csrf_check=csrf_check, max_files=max_files)
    parser = MultipartParser(boundary, sink.callbacks())
    error: UploadError | None = None
    try:
        async for chunk in stream:
            try:
                parser.write(chunk)
            except UploadError as e:
                error = e
            except MultipartParseError as e:
                error = UploadError(400, f"malformed form upload ({e})")
            if error is not None:
                break
        if error is None:
            try:
                parser.finalize()
            except MultipartParseError as e:
                error = UploadError(400, f"malformed form upload ({e})")
            except UploadError as e:
                error = e
        if error is None and not sink.complete:
            error = UploadError(400, "incomplete form upload (the connection was cut short?)")
    finally:
        sink.close()
    if error is not None:
        await _drain(stream)
        raise error
    return sink.form


def content_problems(form: FormData, *, allow_pickle: bool) -> list[str]:
    """Checks on the uploaded bytes (read, never deserialized): pickles, empty files, binary adapter."""
    from malvalid.loaders.base import is_pickle_artifact, sniff_format

    problems: list[str] = []
    for f in form.all_files():
        if f.size == 0:
            problems.append(f"{FIELD_LABELS.get(f.field, f.field)} {f.name} is empty")
    for f in (form.files.get("model_files") or []) + (form.files.get("model_file") or []):
        if f.size and is_pickle_artifact(f.path) and not allow_pickle:
            problems.append(PICKLE_REFUSED.format(name=f.name))
    for fld in ("manifest_file", "config_file", "adapter_file"):
        for f in form.files.get(fld) or []:
            if f.size and sniff_format(f.path) in PICKLE_FORMATS:
                problems.append(f"{FIELD_LABELS[fld]} {f.name} is a pickle or compressed container, "
                                "not a text file")
    ad = form.file("adapter_file")
    if ad is not None and ad.size:
        try:
            with open(ad.path, "rb") as fh:
                head = fh.read(65536)
        except OSError:
            head = b""
        if b"\0" in head:
            problems.append(f"adapter {ad.name} is not a Python source file (it contains binary data)")
    return problems


__all__ = [
    "CONFIG_EXTENSIONS",
    "FILE_FIELDS",
    "FormData",
    "MANIFEST_EXTENSIONS",
    "MODEL_EXTENSIONS",
    "PICKLE_REFUSED",
    "SavedFile",
    "UploadError",
    "content_problems",
    "parse_form",
    "sanitize_filename",
]
