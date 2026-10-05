"""The runs directory as the web UI sees it: run ids, run records and ``RunSummary`` dicts.

Layout (``docs/WEB_CONTRACT.md`` §2)::

    <runs_dir>/<run_id>/             job.json (web runs only), progress.json, report.json,
                                     report.html, run.log, console.log, private/
    <runs_dir>/submissions/<id>/     uploaded adapter/model/manifest/config of a web run

Runs started with ``malvalid run --out <runs_dir>/<name>`` are listed too (``source: "cli"``).
Every reader here is tolerant: a missing, partial or malformed JSON file is treated as absent and
any field of a summary may be None. Run ids are validated and resolved strictly inside the runs
directory; only an allowlist of top-level files is ever served, never anything under ``private/``.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import math
import os
import re
import secrets
import shutil
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from malvalid import _platform

log = logging.getLogger("malvalid.web.store")

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SUBMISSIONS = "submissions"
#: Marker file of a submission dir holding a model inspected with ``POST /runs/inspect`` and not yet
#: submitted (dot-leading: never an uploaded name). Such dirs are deleted after INSPECT_MAX_AGE_S.
INSPECT_MARKER = ".inspect-pending"
INSPECT_MAX_AGE_S = 24 * 3600.0
JOB_SCHEMA = "malvalid-job/1"
PROGRESS_SCHEMA = "malvalid-progress/1"
REPORT_SCHEMA = "malvalid-report/1"
#: Run files written before the rename to MalValid declare these; such runs are still listed and shown.
LEGACY_SCHEMAS: dict[str, str] = {JOB_SCHEMA: "malguard-job/1", PROGRESS_SCHEMA: "malguard-progress/1",
                                  REPORT_SCHEMA: "malguard-report/1"}


def _schema_ok(value: Any, want: str) -> bool:
    """Is ``value`` (a file's declared schema) ``want``, its pre-rename spelling, or absent?"""
    return value in (want, LEGACY_SCHEMAS.get(want), None)

#: Raw files served per run (name -> media type). Nothing else in a run dir is ever served.
RAW_FILES: dict[str, str] = {
    "report.html": "text/html; charset=utf-8",
    "report.json": "application/json",
    "run.log": "text/plain; charset=utf-8",
    "console.log": "text/plain; charset=utf-8",
}
#: Entries malvalid itself creates in a run dir (a dir with anything else is never deleted).
RUN_DIR_ENTRIES = frozenset({"job.json", "progress.json", "report.json", "report.html", "run.log",
                             "console.log", "private", "submission"})

JOB_STATUSES = ("queued", "running", "finished", "failed", "cancelled", "interrupted")
ACTIVE_STATUSES = frozenset({"queued", "running"})
MODULE_STATUSES = ("pass", "warn", "fail", "skipped", "error")
VERDICTS = ("ready", "conditional", "not_ready", "blocked")

MAX_JSON_BYTES = 128 * 1024 * 1024
SUMMARY_FIELDS = (
    "run_id", "source", "status", "created_at", "finished_at", "duration_s", "display_name", "adapter",
    "class_name", "model_kind", "feature_version", "operating_threshold", "corpus", "verdict", "verdict_label",
    "score", "coverage", "exit_code", "n_modules", "n_pass", "n_warn", "n_fail", "n_skipped", "n_error", "title",
)


# --------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime | None = None) -> str:
    return (t or utcnow()).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(v: Any) -> dt.datetime | None:
    if not isinstance(v, str) or not v.strip():
        return None
    s = v.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        t = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def new_timestamped_id() -> str:
    """``YYYYMMDDTHHMMSSZ-<8 hex>`` (web run ids and submission ids)."""
    return f"{utcnow():%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def valid_run_id(run_id: Any) -> bool:
    return isinstance(run_id, str) and bool(RUN_ID_RE.match(run_id)) and run_id != SUBMISSIONS


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and math.isfinite(v) and v.is_integer():
        return int(v)
    return None


def _str(v: Any) -> str | None:
    if v is None or isinstance(v, (dict, list)):
        return None
    s = str(v)
    return s if s.strip() else None


def _d(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _l(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def read_json_file(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any] | None:
    """A JSON object from ``path``, or None if missing, not a regular file, too big or malformed."""
    try:
        st = path.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            return None
        with open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8", "replace"))
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically (temp file in the same dir + ``os.replace``)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, allow_nan=False, default=str)
            f.write("\n")
        _platform.replace_file(tmp, path)  # retries briefly on Windows while a reader holds ``path``
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def tail_text(path: Path, *, max_bytes: int = 65536, max_lines: int = 60) -> str | None:
    """The last lines of a text file (None if absent)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return None
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line is probably cut
    return "\n".join(lines[-max_lines:])


def pid_alive(pid: Any, *, expect: Iterable[str] = ()) -> bool:
    """Is ``pid`` a live (non-zombie) process? With ``expect``, its command line must contain each
    string (guards against pid reuse; skipped where ``/proc`` is unavailable). On Windows the probe is
    ``OpenProcess`` + ``GetExitCodeProcess`` (``os.kill(pid, 0)`` would send CTRL_C_EVENT there), and
    ``expect`` only checks that the executable is a Python interpreter / malvalid launcher."""
    p = _int(pid)
    if p is None or p <= 0:
        return False
    if p > 2**31 - 1:  # not a pid (os.kill would raise OverflowError)
        return False
    if _platform.WINDOWS:
        try:
            return _platform.win_pid_alive(p, expect=expect)
        except (OSError, AttributeError, ValueError):
            return False
    try:
        os.kill(p, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    except (OSError, OverflowError, ValueError):
        return False
    proc = Path(f"/proc/{p}")
    if proc.is_dir():
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[-1].split()
            if fields and fields[0] in ("Z", "X"):
                return False
        except OSError:
            return False
        expect = [e for e in expect if e]
        if expect:
            try:
                cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                return False
            if not all(e in cmd for e in expect):
                return False
    return True


# --------------------------------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------------------------------


@dataclass
class RunRecord:
    """Everything the UI knows about one run directory (any part may be None)."""

    run_id: str
    path: Path
    job: dict[str, Any] | None
    progress: dict[str, Any] | None
    report: dict[str, Any] | None
    files: dict[str, bool]
    summary: dict[str, Any]
    #: Why the run's files could not be (fully) used, e.g. an unreadable report.json; None if fine.
    problem: str | None = None
    #: report.json belongs to an earlier run into the same --out dir (ignored; see :func:`stale_report`).
    stale_report: bool = False


def _files(run_dir: Path) -> dict[str, bool]:
    def ok(name: str) -> bool:
        p = run_dir / name
        try:
            return p.is_file() and not p.is_symlink() and p.resolve().parent == run_dir.resolve()
        except OSError:
            return False

    return {
        "report_html": ok("report.html"),
        "report_json": ok("report.json"),
        "run_log": ok("run.log"),
        "console_log": ok("console.log"),
    }


def _valid_job(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if d is None or not _schema_ok(d.get("schema"), JOB_SCHEMA) or "status" not in d:
        return None
    return d


def _valid_progress(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if d is None or not _schema_ok(d.get("schema"), PROGRESS_SCHEMA) or "stage" not in d:
        return None
    return d


def _valid_report(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if d is None:
        return None
    if not _schema_ok(d.get("schema_version"), REPORT_SCHEMA):
        return None
    if not any(k in d for k in ("verdict", "modules", "run")):
        return None
    return d


def stale_report(progress: dict[str, Any] | None, report: dict[str, Any] | None) -> bool:
    """Is ``report`` left over from an earlier run into the same directory?

    ``malvalid run`` into an existing ``--out`` dir does not delete the previous report.json; the new
    run's progress.json carries its own run id. When both ids are known and differ, the report is not
    this run's result (the new run is still going, or failed before writing its own report).
    """
    if progress is None or report is None:
        return False
    pid = progress.get("run_id")
    rid = _d(report.get("run")).get("id")
    return isinstance(pid, str) and isinstance(rid, str) and bool(pid) and bool(rid) and pid != rid


def cli_status(progress: dict[str, Any] | None, report: dict[str, Any] | None, run_dir: Path) -> str:
    """Status of a run the web UI did not start (``malvalid run`` in a terminal)."""
    if report is not None:
        code = _int(_d(report.get("gate")).get("exit_code"))
        return "failed" if code is not None and code not in (0, 1) else "finished"
    if progress is None:
        return "failed"
    stage = progress.get("stage")
    if stage in ("done", "failed"):
        return "failed"  # done without a report, or the run raised
    if pid_alive(progress.get("pid"), expect=("malvalid",)):
        return "running"
    return "interrupted"


def module_counts(modules: Iterable[Any]) -> dict[str, int]:
    counts = {s: 0 for s in MODULE_STATUSES}
    n = 0
    for m in modules:
        if not isinstance(m, dict):
            continue
        n += 1
        st = m.get("status")
        if isinstance(st, str) and st in counts:
            counts[st] += 1
    return {"n_modules": n, **{f"n_{s}": c for s, c in counts.items()}}


def build_summary(
    run_id: str,
    run_dir: Path,
    job: dict[str, Any] | None,
    progress: dict[str, Any] | None,
    report: dict[str, Any] | None,
    *,
    status: str | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """The ``RunSummary`` dict of the web contract (every field present, any may be None)."""
    now = now or utcnow()
    rep = report or {}
    run = _d(rep.get("run"))
    ver = _d(rep.get("verdict"))
    gate = _d(rep.get("gate"))
    model = _d(rep.get("model"))
    corpus = _d(rep.get("corpus"))
    cfg = _d(rep.get("config"))
    jb = job or {}
    opts = _d(jb.get("options"))
    pg = progress or {}
    pv = _d(pg.get("verdict"))

    if status is None:
        status = str(jb.get("status")) if job is not None else cli_status(progress, report, run_dir)
    if status not in JOB_STATUSES:
        status = "failed"

    created = _str(jb.get("created_at")) or _str(run.get("started_at")) or _str(pg.get("started_at"))
    if created is None:
        with contextlib.suppress(OSError):
            created = iso(dt.datetime.fromtimestamp(run_dir.stat().st_mtime, dt.timezone.utc))
    finished = _str(jb.get("finished_at")) if job is not None else None
    finished = finished or _str(run.get("finished_at"))
    if finished is None and pg.get("stage") in ("done", "failed") and job is None:
        finished = _str(pg.get("updated_at"))

    duration = _num(run.get("duration_s")) if report is not None else None
    if duration is None:
        start = parse_iso(jb.get("started_at")) or parse_iso(pg.get("started_at")) or parse_iso(run.get("started_at"))
        end = parse_iso(finished)
        if start is not None and end is not None:
            duration = max(0.0, (end - start).total_seconds())
        elif start is not None and status == "running":
            duration = max(0.0, (now - start).total_seconds())

    modules = _l(rep.get("modules")) if report is not None else _l(pg.get("modules"))
    counts = module_counts(modules)

    title = _str(rep.get("title")) or _str(opts.get("title"))
    class_name = _str(model.get("class_name")) or _str(opts.get("class_name"))
    adapter = _str(jb.get("adapter")) or _str(model.get("adapter_path"))
    display = _str(jb.get("display_name")) or title
    if display is None:
        base = Path(adapter).name if adapter else None
        display = " · ".join(x for x in (class_name, base) if x) or run_id

    if report is not None:
        verdict = _str(ver.get("verdict"))
        label = _str(ver.get("label"))
        score = _num(ver.get("score"))
        coverage = _num(ver.get("coverage"))
    else:
        verdict = _str(pv.get("verdict"))
        label = _str(pv.get("label"))
        score = _num(pv.get("score"))
        coverage = _num(pv.get("coverage"))
    if verdict is not None and verdict not in VERDICTS:
        verdict = None

    exit_code = _int(jb.get("exit_code"))
    if exit_code is None:
        exit_code = _int(gate.get("exit_code"))
    if exit_code is None:
        exit_code = _int(pg.get("exit_code"))

    return {
        "run_id": run_id,
        "source": "web" if job is not None else "cli",
        "status": status,
        "created_at": created,
        "finished_at": finished,
        "duration_s": None if duration is None else round(duration, 3),
        "display_name": display,
        "adapter": adapter,
        "class_name": class_name,
        "model_kind": _str(model.get("model_kind")),
        "feature_version": _str(model.get("feature_version")),
        "operating_threshold": _num(model.get("operating_threshold")),
        "corpus": _str(corpus.get("name")) or _str(cfg.get("corpus")) or _str(opts.get("corpus")),
        "verdict": verdict,
        "verdict_label": label,
        "score": score,
        "coverage": coverage,
        "exit_code": exit_code,
        **counts,
        "title": title,
    }


def _sort_key(summary: dict[str, Any], seq: Any = None) -> tuple[float, int, str]:
    """Newest first (with ``reverse=True``): creation time, then the submission sequence number of web
    runs (``job.json`` ``seq``, ns resolution) so same-second submissions keep their queue order."""
    t = parse_iso(summary.get("created_at"))
    n = _int(seq)
    return (t.timestamp() if t else 0.0, n if n is not None else 0, str(summary.get("run_id")))


#: Prefix of the URL-safe alias of a run directory whose name is not a valid run id
#: (``malvalid run --out "runs/my model v2"``): ``enc-`` + the name's UTF-8 bytes in hex.
ALIAS_PREFIX = "enc-"


def alias_for(name: str) -> str | None:
    """The run id under which a non-conforming directory name is listed (None if it cannot have one)."""
    if not name or name.startswith(".") or name in (SUBMISSIONS,) or "/" in name or "\\" in name or "\0" in name:
        return None
    alias = ALIAS_PREFIX + name.encode("utf-8", "surrogateescape").hex()
    return alias if RUN_ID_RE.match(alias) else None


def name_for_alias(run_id: str) -> str | None:
    """The directory name an ``enc-`` alias stands for, or None."""
    if not run_id.startswith(ALIAS_PREFIX):
        return None
    try:
        name = bytes.fromhex(run_id[len(ALIAS_PREFIX):]).decode("utf-8", "surrogateescape")
    except ValueError:
        return None
    if valid_run_id(name) or alias_for(name) != run_id:  # only for names that need one
        return None
    return name


# --------------------------------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------------------------------


class RunNotFound(LookupError):
    """No run with this id (or the id is invalid / escapes the runs directory)."""


class RunStore:
    """Read access to ``<runs_dir>`` plus id allocation and directory removal."""

    def __init__(self, runs_dir: Path):
        self.root = Path(runs_dir).expanduser().resolve()
        self._cache: dict[Path, tuple[tuple[int, int], dict[str, Any] | None]] = {}
        self._lock = threading.Lock()

    # ---- paths ---------------------------------------------------------------------------------

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / SUBMISSIONS).mkdir(exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        """The directory of an existing run; raises :class:`RunNotFound` otherwise."""
        if not valid_run_id(run_id):
            raise RunNotFound(run_id)
        p = self.root / run_id
        alias = name_for_alias(run_id)
        if alias is not None and not p.exists():
            p = self.root / alias
        try:
            if p.is_symlink() or not p.is_dir():
                raise RunNotFound(run_id)
            rp = p.resolve()
        except OSError as e:
            raise RunNotFound(run_id) from e
        if rp.parent != self.root or not is_within(rp, self.root):
            raise RunNotFound(run_id)
        return rp

    def exists(self, run_id: str) -> bool:
        try:
            self.run_dir(run_id)
            return True
        except RunNotFound:
            return False

    def new_run_dir(self) -> tuple[str, Path]:
        """Allocate ``YYYYMMDDTHHMMSSZ-<8 hex>`` and create its directory."""
        self.ensure_root()
        for _ in range(100):
            rid = new_timestamped_id()
            p = self.root / rid
            try:
                p.mkdir()
            except FileExistsError:
                continue
            return rid, p
        raise RuntimeError("could not allocate a run id")  # pragma: no cover

    def submission_dir(self, submission_id: str) -> Path:
        if not valid_run_id(submission_id):
            raise ValueError(f"invalid submission id {submission_id!r}")
        return self.root / SUBMISSIONS / submission_id

    def new_submission_dir(self) -> tuple[str, Path]:
        self.ensure_root()
        for _ in range(100):
            sid = new_timestamped_id()
            p = self.root / SUBMISSIONS / sid
            try:
                p.mkdir()
            except FileExistsError:
                continue
            return sid, p
        raise RuntimeError("could not allocate a submission id")  # pragma: no cover

    # ---- inspected model uploads (``POST /runs/inspect``) -----------------------------------------

    def mark_inspection(self, submission_id: str) -> Path:
        """Mark a submission dir as an inspected, not yet submitted model upload; returns the marker."""
        marker = self.submission_dir(submission_id) / INSPECT_MARKER
        marker.write_text(iso() + "\n", encoding="utf-8")
        return marker

    def inspection_dir(self, submission_id: Any) -> Path | None:
        """The directory of a pending inspected upload, or None.

        Only a real directory (no symlink) directly under ``<runs_dir>/submissions`` that carries the
        inspection marker qualifies, so a form can never point at a run's own submission or outside.
        """
        if not isinstance(submission_id, str) or not valid_run_id(submission_id):
            return None
        root = self.root / SUBMISSIONS
        p = root / submission_id
        try:
            if p.is_symlink() or not p.is_dir() or p.resolve().parent != root.resolve():
                return None
            marker = p / INSPECT_MARKER
            if marker.is_symlink() or not marker.is_file():
                return None
        except OSError:
            return None
        return p

    def inspected_model(self, submission_id: Any, extensions: Iterable[str]) -> Path | None:
        """The single model file of a pending inspected upload (a regular file, not a symlink), or None."""
        d = self.inspection_dir(submission_id)
        if d is None:
            return None
        exts = tuple(e.lower() for e in extensions)
        try:
            files = [e for e in d.iterdir() if not e.name.startswith(".")]
        except OSError:
            return None
        if len(files) != 1:
            return None
        f = files[0]
        try:
            if f.is_symlink() or not f.is_file() or f.suffix.lower() not in exts:
                return None
        except OSError:
            return None
        return f

    def cleanup_inspections(self, *, max_age_s: float = INSPECT_MAX_AGE_S, keep: Iterable[str] = ()) -> list[str]:
        """Delete inspected uploads that were never submitted and are older than ``max_age_s``."""
        removed: list[str] = []
        root = self.root / SUBMISSIONS
        keep_set = set(keep)
        now = time.time()
        try:
            entries = list(root.iterdir())
        except OSError:
            return removed
        for e in entries:
            if e.name in keep_set or self.inspection_dir(e.name) is None:
                continue
            try:
                age = now - (e / INSPECT_MARKER).stat().st_mtime
            except OSError:
                continue
            if age > max_age_s:
                self.remove_submission(e.name)
                removed.append(e.name)
        return removed

    def raw_file(self, run_id: str, name: str) -> Path | None:
        """Path of an allowlisted raw file of a run, or None (unknown name / absent / escapes)."""
        if name not in RAW_FILES:
            return None
        d = self.run_dir(run_id)
        p = d / name
        try:
            # A regular file directly in the run dir: no symlinks (which could point into private/
            # or out of the runs directory), nothing below a subdirectory.
            if p.is_symlink():
                return None
            rp = p.resolve()
            if rp.parent != d or not is_within(rp, d) or not rp.is_file():
                return None
        except OSError:
            return None
        return rp

    # ---- reading -------------------------------------------------------------------------------

    def _read_cached(self, path: Path, digest: Any = None) -> dict[str, Any] | None:
        """``read_json_file`` memoised on (mtime_ns, size); ``digest`` keeps only what summaries need."""
        try:
            st = path.stat()
        except OSError:
            with self._lock:
                self._cache.pop(path, None)
            return None
        key = (st.st_mtime_ns, st.st_size)
        with self._lock:
            hit = self._cache.get(path)
        if hit is not None and hit[0] == key:
            return hit[1]
        data = read_json_file(path)
        if digest is not None and data is not None:
            data = digest(data)
        with self._lock:
            if len(self._cache) > 2048:
                self._cache.clear()
            self._cache[path] = (key, data)
        return data

    def read_job(self, run_id: str) -> dict[str, Any] | None:
        return _valid_job(read_json_file(self.run_dir(run_id) / "job.json"))

    def read_progress(self, run_id: str) -> dict[str, Any] | None:
        return _valid_progress(read_json_file(self.run_dir(run_id) / "progress.json"))

    def read_report(self, run_id: str) -> dict[str, Any] | None:
        return _valid_report(read_json_file(self.run_dir(run_id) / "report.json"))

    def list_run_ids(self) -> list[str]:
        try:
            entries = list(os.scandir(self.root))
        except OSError:
            return []
        out = []
        names = {e.name for e in entries}
        for e in entries:
            rid: str | None = e.name
            if not valid_run_id(e.name):
                # e.g. `malvalid run --out "runs/my model v2"`: listed under a URL-safe alias
                rid = alias_for(e.name)
                if rid is None or rid in names:
                    continue
            try:
                if e.is_symlink() or not e.is_dir():
                    continue
            except OSError:
                continue
            p = Path(e.path)
            if any((p / n).exists() for n in ("job.json", "progress.json", "report.json")):
                out.append(rid)
        return out

    def load(self, run_id: str, *, job: dict[str, Any] | None = None, full_report: bool = True,
             status: str | None = None) -> RunRecord:
        """Read one run. ``job`` overrides job.json (the job manager's live record).

        Never fails on malformed run files: a run whose files exist but cannot be used is returned with
        :attr:`RunRecord.problem` set (status ``failed``) instead of disappearing.
        """
        d = self.run_dir(run_id)
        cached = not full_report  # the listing path: memoise parsed files on (mtime, size)
        if job is None:
            job = _valid_job(self._read_cached(d / "job.json") if cached else read_json_file(d / "job.json"))
        progress = _valid_progress(self._read_cached(d / "progress.json") if cached
                                   else read_json_file(d / "progress.json"))
        if full_report:
            report = _valid_report(read_json_file(d / "report.json"))
        else:
            report = self._read_cached(d / "report.json", _valid_report_digest)
        files = _files(d)
        problem: str | None = None
        stale = job is None and stale_report(progress, report)
        if stale:
            report = None
            files = {**files, "report_html": False, "report_json": False}
        if job is None and progress is None and report is None:
            present = [n for n in ("job.json", "progress.json", "report.json") if (d / n).exists()]
            if not present or _foreign_only(d, present):
                raise RunNotFound(run_id)
            problem = (f"{', '.join(present)} could not be read (missing fields, not valid JSON, or larger than "
                       f"{MAX_JSON_BYTES // (1024 * 1024)} MB)")
            files = {**files, "report_html": False}
        try:
            summary = build_summary(run_id, d, job, progress, report, status=status)
        except Exception as e:  # malformed fields: degrade, never 500 or vanish
            log.warning("run %s: malformed run files (%s: %s)", run_id, type(e).__name__, e)
            problem = problem or f"the run's files are malformed ({type(e).__name__}: {e})"
            progress = report = None
            files = {**files, "report_html": False}
            try:  # keep a web job's record (status, Cancel) when only progress/report are broken
                summary = build_summary(run_id, d, job, None, None, status=status if job is not None else "failed")
            except Exception:
                job = None
                summary = build_summary(run_id, d, None, None, None, status="failed")
        if problem:
            if job is None and summary.get("status") not in ACTIVE_STATUSES:
                summary["status"] = "failed"
            if summary.get("display_name") in (None, run_id):
                summary["display_name"] = f"{run_id} (unreadable run files)"
        return RunRecord(run_id, d, job, progress, report, files, summary, problem=problem, stale_report=stale)

    def summary(self, run_id: str, *, job: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.load(run_id, job=job, full_report=False).summary

    def summaries(self, jobs: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Every run, newest first. ``jobs``: live job records by run id (override job.json)."""
        keyed = []
        for rid in self.list_run_ids():
            try:
                rec = self.load(rid, job=(jobs or {}).get(rid), full_report=False)
            except RunNotFound:
                continue
            except Exception as e:  # pragma: no cover - load() degrades; belt and braces
                log.warning("could not read run %s: %s", rid, e)
                continue
            keyed.append((_sort_key(rec.summary, (rec.job or {}).get("seq")), rec.summary))
        keyed.sort(key=lambda kv: kv[0], reverse=True)
        return [sm for _, sm in keyed]

    # ---- removal -------------------------------------------------------------------------------

    def unexpected_entries(self, run_id: str) -> list[str]:
        """Names in a run dir that malvalid does not create (such a dir is never deleted)."""
        d = self.run_dir(run_id)
        odd = []
        for e in os.scandir(d):
            if e.name in RUN_DIR_ENTRIES or (e.name.startswith(".") and e.name.endswith(".tmp")):
                continue
            odd.append(e.name)
        return sorted(odd)

    def remove_run_dir(self, run_id: str) -> None:
        d = self.run_dir(run_id)
        shutil.rmtree(d)
        with self._lock:
            for k in [k for k in self._cache if is_within(k, d)]:
                self._cache.pop(k, None)

    def remove_submission(self, submission_id: str | None) -> None:
        if not submission_id or not valid_run_id(str(submission_id)):
            return
        p = self.root / SUBMISSIONS / str(submission_id)
        try:
            if p.is_symlink() or not p.is_dir() or p.resolve().parent != (self.root / SUBMISSIONS).resolve():
                return
        except OSError:
            return
        shutil.rmtree(p, ignore_errors=True)


_SCHEMA_KEYS = {"job.json": ("schema", JOB_SCHEMA), "progress.json": ("schema", PROGRESS_SCHEMA),
                "report.json": ("schema_version", REPORT_SCHEMA)}


def _foreign_only(run_dir: Path, names: list[str]) -> bool:
    """Do all these files declare another tool's schema? (such a directory is not a malvalid run)"""
    for n in names:
        data = read_json_file(run_dir / n)
        key, want = _SCHEMA_KEYS[n]
        if data is None or _schema_ok(data.get(key), want):
            return False
    return True


def _valid_report_digest(report: dict[str, Any]) -> dict[str, Any] | None:
    """The digest of a report that passes the same validity test as a full read (else None), so the
    dashboard and the run page agree on whether a run has a report."""
    if _valid_report(report) is None:
        return None
    return _report_digest(report)


def _report_digest(report: dict[str, Any]) -> dict[str, Any]:
    """The parts of report.json a RunSummary needs (keeps the listing cache small)."""
    keep: dict[str, Any] = {}
    for k in ("schema_version", "title"):
        if k in report:
            keep[k] = report[k]
    run = _d(report.get("run"))
    keep["run"] = {k: run.get(k) for k in ("id", "started_at", "finished_at", "duration_s")}
    ver = _d(report.get("verdict"))
    keep["verdict"] = {k: ver.get(k) for k in ("verdict", "label", "score", "coverage")}
    keep["gate"] = {"exit_code": _d(report.get("gate")).get("exit_code")}
    model = _d(report.get("model"))
    keep["model"] = {k: model.get(k) for k in ("class_name", "adapter_path", "model_kind", "feature_version",
                                                "operating_threshold")}
    keep["corpus"] = {"name": _d(report.get("corpus")).get("name")}
    keep["config"] = {"corpus": _d(report.get("config")).get("corpus")}
    keep["modules"] = [{"status": m.get("status")} for m in _l(report.get("modules")) if isinstance(m, dict)]
    return keep


__all__ = [
    "ACTIVE_STATUSES",
    "ALIAS_PREFIX",
    "INSPECT_MARKER",
    "INSPECT_MAX_AGE_S",
    "JOB_SCHEMA",
    "RAW_FILES",
    "RUN_ID_RE",
    "RunNotFound",
    "RunRecord",
    "RunStore",
    "SUMMARY_FIELDS",
    "alias_for",
    "build_summary",
    "iso",
    "new_timestamped_id",
    "parse_iso",
    "pid_alive",
    "read_json_file",
    "stale_report",
    "tail_text",
    "utcnow",
    "valid_run_id",
    "write_json_atomic",
]
