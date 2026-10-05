"""Background execution of web-submitted runs.

Every web run is the same ``malvalid run`` the CLI performs, in its own subprocess
(``[sys.executable, "-m", "malvalid", "run", "--adapter", A, "--out", <run_dir>, …]``, new session,
stdout+stderr → ``<run_dir>/console.log``, environment inherited so ``MALVALID_CORPUS_DIR`` & co.
apply). The web server process itself never imports an adapter, loads a model or unpickles anything.

:class:`JobManager` keeps a FIFO queue drained by ``max_concurrent`` worker threads and mirrors each
job's state into ``<run_dir>/job.json`` (schema ``malvalid-job/1``, written atomically):

``queued → running → finished | failed | cancelled``; ``interrupted`` when the server stops (or
restarts and finds the process gone). ``finished`` means the CLI exited 0/1 and wrote report.json;
``failed`` means exit 2 or no report. Cancel = SIGTERM to the run's process group, SIGKILL after
``cancel_grace_s`` (on Windows: CTRL_BREAK_EVENT to the run's process group, then ``taskkill /F /T``
on its process tree). On start-up, ``job.json`` files left ``queued``/``running`` by an earlier server are
reconciled: a still-running process is adopted (watched until it exits), a dead one is
``interrupted`` (or ``finished`` if it completed its report while the server was down).
"""

from __future__ import annotations

import collections
import contextlib
import copy
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from malvalid.web.settings import WebSettings
from malvalid.web.store import (
    ACTIVE_STATUSES,
    JOB_SCHEMA,
    RunNotFound,
    RunStore,
    iso,
    parse_iso,
    pid_alive,
    read_json_file,
    tail_text,
    write_json_atomic,
)

log = logging.getLogger("malvalid.web.jobs")

ADOPT_POLL_S = 1.0
SHUTDOWN_GRACE_S = 5.0
VALIDATION_CONCURRENCY = 2
INSPECT_TIMEOUT_S = 120.0
#: Windows: how long ``taskkill /F /T`` may take, and how long to drain a killed validator's pipes.
TASKKILL_TIMEOUT_S = 15.0
KILL_DRAIN_TIMEOUT_S = 10.0
_WINDOWS = os.name == "nt"
_ERROR_LINE = re.compile(r"^\s*error:\s*(.+?)\s*$")


class JobError(Exception):
    """A job operation cannot be performed (HTTP status in ``status``)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class JobSpec:
    """What :meth:`JobManager.submit` needs (built from the new-run form)."""

    argv: list[str]
    mode: str  # upload | path
    adapter: str
    display_name: str
    submission_id: str | None
    options: dict[str, Any]


def _subprocess_env(settings: WebSettings) -> dict[str, str]:
    env = dict(os.environ)
    env.update(settings.extra_env)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONSAFEPATH"] = "1"  # never import from the working directory
    env.setdefault("COLUMNS", "120")
    return env


def _signal_group(pid: int, sig: int) -> bool:
    """Send ``sig`` to the process group led by ``pid`` (falls back to the process itself). POSIX only."""
    if not hasattr(os, "killpg"):
        return False
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _popen_group_kwargs() -> dict[str, Any]:
    """Popen arguments putting a run / validation child in its own process group: a new session on
    POSIX; ``CREATE_NEW_PROCESS_GROUP`` on Windows (Ctrl+C in the server console then does not reach it,
    and it can be sent CTRL_BREAK_EVENT)."""
    if _WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)}
    return {"start_new_session": True}


def _stop_group(pid: int, proc: subprocess.Popen[bytes] | None = None) -> bool:
    """Ask a run to stop: SIGTERM to its process group (POSIX). On Windows CTRL_BREAK_EVENT is sent
    only to the process group of a ``Popen`` this server owns (a stale pid could name an unrelated
    group); returns False when nothing could be sent, so the caller goes straight to :func:`_kill_tree`."""
    if not _WINDOWS:
        return _signal_group(pid, signal.SIGTERM)
    if proc is None or proc.poll() is not None:
        return False
    try:
        proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT"))
        return True
    except (OSError, ValueError, AttributeError):
        return False


def _taskkill_exe() -> str:
    exe = os.path.join(os.environ.get("SystemRoot") or r"C:\Windows", "System32", "taskkill.exe")
    return exe if os.path.isfile(exe) else "taskkill.exe"


def _kill_tree(pid: int, proc: subprocess.Popen[bytes] | None = None) -> bool:
    """Kill a run and everything it started: SIGKILL to its process group (POSIX, which also reaps
    stragglers left in the group). On Windows ``taskkill /F /T /PID`` (the sandbox worker is a child of
    the run), falling back to ``proc.kill()``; a pid not owned through ``proc`` is killed only while it
    still looks like a malvalid process."""
    sigkill = getattr(signal, "SIGKILL", None)
    if not _WINDOWS and sigkill is not None:
        return _signal_group(pid, sigkill)
    if proc is None and not pid_alive(pid, expect=("malvalid",)):
        return False
    if proc is not None and proc.poll() is not None:
        return False  # already exited (and its pid may no longer be ours to kill)
    ok = False
    try:
        r = subprocess.run(
            [_taskkill_exe(), "/F", "/T", "/PID", str(int(pid))], stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=TASKKILL_TIMEOUT_S,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        ok = r.returncode == 0
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        log.warning("taskkill of pid %s failed: %s", pid, e)
    if proc is not None and proc.poll() is None:
        with contextlib.suppress(OSError):
            proc.kill()
            ok = True
    return ok


def _report_finished_at(report: dict[str, Any] | None) -> str | None:
    """``report.run.finished_at`` when it is a valid timestamp (a run that ended while nobody watched)."""
    run = (report or {}).get("run")
    v = run.get("finished_at") if isinstance(run, dict) else None
    return v if parse_iso(v) is not None else None


def _last_error_line(text: str | None) -> str | None:
    if not text:
        return None
    for line in reversed(text.splitlines()):
        m = _ERROR_LINE.match(line)
        if m:
            return m.group(1)
    return None


class JobManager:
    """FIFO queue of web runs + worker threads + the lifecycle of their subprocesses."""

    def __init__(self, settings: WebSettings, store: RunStore):
        self.settings = settings
        self.store = store
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._queue: collections.deque[str] = collections.deque()
        self._jobs: dict[str, dict[str, Any]] = {}  # live records of jobs this server knows about
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._adopted: dict[str, int] = {}  # run_id -> pid of a run started by an earlier server
        self._cancel: set[str] = set()
        self._busy = 0  # jobs the workers are executing (adopted runs count against max_concurrent too)
        self._threads: list[threading.Thread] = []
        self._started = False
        self._stopping = False
        self._validate_sem = threading.BoundedSemaphore(VALIDATION_CONCURRENCY)
        self.on_change: Callable[[str], None] | None = None  # test hook

    # ---- lifecycle ---------------------------------------------------------------------------------

    def start(self) -> None:
        """Reconcile leftover job.json files, then start the worker threads (idempotent)."""
        with self._lock:
            if self._started:
                return
            self._started = True
            self._stopping = False
        self.store.ensure_root()
        self.reconcile()
        for i in range(self.settings.max_concurrent):
            t = threading.Thread(target=self._worker, name=f"malvalid-web-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def shutdown(self, *, grace_s: float = SHUTDOWN_GRACE_S) -> None:
        """Stop the workers; queued and running jobs end ``interrupted`` (their processes are terminated)."""
        with self._cv:
            if not self._started or self._stopping:
                return
            self._stopping = True
            queued = list(self._queue)
            self._queue.clear()
            procs = dict(self._procs)
            adopted = dict(self._adopted)
            self._cv.notify_all()
        for rid in queued:
            self._finish(rid, "interrupted", error="the web server was stopped before this run started")
        for rid, proc in procs.items():
            _stop_group(proc.pid, proc)
        for rid, pid in adopted.items():
            _stop_group(pid)
        deadline = time.monotonic() + grace_s
        for proc in procs.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        for proc in procs.values():
            _kill_tree(proc.pid, proc)
        for pid in adopted.values():
            if pid_alive(pid):
                _kill_tree(pid)
        for rid in adopted:
            self._finish(rid, "interrupted", error="the web server was stopped while this run was in progress")
        for t in self._threads:
            t.join(timeout=grace_s)
        self._threads.clear()
        with self._lock:
            self._started = False

    def work_dir(self, fallback: Path) -> Path:
        """Working directory of run / validation subprocesses: where ``malvalid serve`` was started, so
        relative paths in a policy (``corpus_dir: corp``) mean what they mean for ``malvalid run`` typed in
        that terminal, and an adapter's stray files never land in (and block deletion of) the run dir.
        ``PYTHONSAFEPATH`` keeps that directory off ``sys.path``."""
        d = self.settings.launch_dir
        return d if d.is_dir() else fallback

    # ---- queries -------------------------------------------------------------------------------

    def job(self, run_id: str) -> dict[str, Any] | None:
        """The live record of a job (a copy), else its job.json, else None (a CLI run)."""
        with self._lock:
            rec = self._jobs.get(run_id)
            if rec is not None:
                return copy.deepcopy(rec)
        try:
            return self.store.read_job(run_id)
        except RunNotFound:
            return None

    def queue_ahead(self, run_id: str) -> list[dict[str, Any]]:
        """For a queued job: the jobs that run before it (running ones first, then earlier queued ones),
        as ``{run_id, display_name, status}``. Empty for anything else."""
        with self._lock:
            if run_id not in self._queue:
                return []
            ahead = [rid for rid, rec in self._jobs.items() if rec.get("status") == "running"]
            ahead += list(self._queue)[: list(self._queue).index(run_id)]
            return [{"run_id": rid, "display_name": (self._jobs.get(rid) or {}).get("display_name") or rid,
                     "status": (self._jobs.get(rid) or {}).get("status")} for rid in ahead]

    def live_jobs(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: copy.deepcopy(v) for k, v in self._jobs.items()}

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(run_id)
            return rec is not None and rec.get("status") in ACTIVE_STATUSES

    def active_ids(self) -> list[str]:
        with self._lock:
            return [k for k, v in self._jobs.items() if v.get("status") in ACTIVE_STATUSES]

    # ---- persistence -----------------------------------------------------------------------------

    def _save(self, rec: dict[str, Any]) -> None:
        """Write a job record to its job.json (caller holds the lock)."""
        rid = rec["run_id"]
        try:
            d = self.store.run_dir(rid)
        except RunNotFound:
            log.warning("run %s: run directory vanished; job state not saved", rid)
            return
        try:
            write_json_atomic(d / "job.json", rec)
        except (OSError, TypeError, ValueError) as e:
            log.error("run %s: could not write job.json: %s", rid, e)
        cb = self.on_change
        if cb is not None:
            with contextlib.suppress(Exception):
                cb(rid)

    def _finish(self, run_id: str, status: str, *, error: str | None = None, exit_code: int | None = None,
                note: str | None = None, finished_at: str | None = None) -> None:
        with self._cv:
            rec = self._jobs.get(run_id)
            if rec is None:
                return
            rec["status"] = status
            rec["finished_at"] = finished_at or iso()
            if exit_code is not None:
                rec["exit_code"] = exit_code
            if error is not None:
                rec["error"] = error
            if note:
                rec["note"] = note
            self._save(rec)
            self._procs.pop(run_id, None)
            self._adopted.pop(run_id, None)
            self._cancel.discard(run_id)
            self._cv.notify_all()

    # ---- submit / cancel / delete ------------------------------------------------------------

    def submit(self, spec: JobSpec, *, run_id: str | None = None, run_dir: Path | None = None) -> str:
        """Create the run dir + job.json (``queued``) and enqueue the job; returns the run id.

        ``spec.argv`` may contain the placeholders ``{run_dir}`` and ``{run_id}``, replaced by the new
        run directory and its id.
        """
        if run_id is None or run_dir is None:
            run_id, run_dir = self.store.new_run_dir()
        argv = [a.replace("{run_dir}", str(run_dir)).replace("{run_id}", run_id) for a in spec.argv]
        rec: dict[str, Any] = {
            "schema": JOB_SCHEMA,
            "run_id": run_id,
            "created_at": iso(),
            "seq": time.time_ns(),  # orders same-second submissions (created_at has 1 s resolution)
            "started_at": None,
            "finished_at": None,
            "status": "queued",
            "argv": argv,
            "mode": spec.mode,
            "adapter": spec.adapter,
            "display_name": spec.display_name,
            "submission_id": spec.submission_id,
            "options": spec.options,
            "pid": None,
            "exit_code": None,
            "error": None,
        }
        with self._cv:
            if self._stopping:
                raise JobError(503, "the server is shutting down")
            self._jobs[run_id] = rec
            self._save(rec)
            self._queue.append(run_id)
            self._cv.notify()
        if not self._started:
            self.start()
        log.info("queued run %s (%s)", run_id, spec.display_name)
        return run_id

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Cancel a queued or running job. Returns the job record (a running one stays ``running``
        until its process has exited, then becomes ``cancelled``)."""
        self.store.run_dir(run_id)  # 404 for unknown ids
        with self._cv:
            rec = self._jobs.get(run_id)
            if rec is None or rec.get("status") not in ACTIVE_STATUSES:
                status = (rec or self.store.read_job(run_id) or {}).get("status")
                if status is None:
                    raise JobError(409, "this run was not started from the web UI, so it cannot be cancelled here")
                raise JobError(409, f"this run is {status}, not queued or running")
            if rec["status"] == "queued":
                with contextlib.suppress(ValueError):
                    self._queue.remove(run_id)
                rec["status"] = "cancelled"
                rec["finished_at"] = iso()
                rec["error"] = "cancelled before it started"
                self._save(rec)
                return copy.deepcopy(rec)
            self._cancel.add(run_id)
            rec["cancel_requested_at"] = iso()
            self._save(rec)
            proc = self._procs.get(run_id)
            pid = proc.pid if proc is not None else self._adopted.get(run_id)
        if pid is not None:
            self._terminate(run_id, pid, proc)
        return self.job(run_id) or {}

    def _terminate(self, run_id: str, pid: int, proc: subprocess.Popen[bytes] | None) -> None:
        if _WINDOWS:
            log.info("cancelling run %s (CTRL_BREAK to process group %d, then taskkill)", run_id, pid)
        else:
            log.info("cancelling run %s (SIGTERM to process group %d)", run_id, pid)
        sent = _stop_group(pid, proc)
        grace = max(0.0, float(self.settings.cancel_grace_s))
        if _WINDOWS and not sent:
            grace = 0.0  # no graceful stop possible (adopted run / CTRL_BREAK refused): kill the tree now

        def _escalate() -> None:
            alive = proc.poll() is None if proc is not None else pid_alive(pid)
            if alive and (sent or not _WINDOWS):
                log.warning("run %s did not stop within %.0f s of SIGTERM; sending SIGKILL", run_id, grace)
            _kill_tree(pid, proc)  # also reaps stragglers left in the group

        t = threading.Timer(grace, _escalate)
        t.daemon = True
        t.start()

    def delete(self, run_id: str) -> None:
        """Remove a run directory (and its submission); refused while the run is queued/running."""
        self.store.run_dir(run_id)
        with self._lock:
            if self.is_active(run_id):
                raise JobError(409, "this run is still queued or running; cancel it first")
            job = self.job(run_id)
            if job is None:
                prog = self.store.read_progress(run_id)
                if (prog and prog.get("stage") not in ("done", "failed")
                        and pid_alive(prog.get("pid"), expect=("malvalid",))):
                    raise JobError(409, "this run is still running (started from the command line)")
            elif job.get("status") in ACTIVE_STATUSES:
                raise JobError(409, "this run is still queued or running; cancel it first")
            odd = self.store.unexpected_entries(run_id)
            if odd:
                raise JobError(
                    409,
                    "the run directory contains files malvalid did not create ("
                    + ", ".join(odd[:5]) + ("…" if len(odd) > 5 else "") + "); delete it by hand if you are sure",
                )
            self.store.remove_run_dir(run_id)
            self._jobs.pop(run_id, None)
            if job is not None:
                self.store.remove_submission(job.get("submission_id"))
        log.info("deleted run %s", run_id)

    # ---- workers -------------------------------------------------------------------------------

    def _slots_free(self) -> bool:
        """Is a run slot free? (caller holds the lock). Runs adopted from an earlier server occupy slots."""
        return self._busy + len(self._adopted) < self.settings.max_concurrent

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._stopping and not (self._queue and self._slots_free()):
                    self._cv.wait()
                if self._stopping:
                    return
                run_id = self._queue.popleft()
                self._busy += 1
            try:
                self._execute(run_id)
            except Exception as e:  # never let a worker thread die
                log.exception("run %s: internal error in the job worker", run_id)
                self._finish(run_id, "failed", error=f"internal error in the web job worker: {type(e).__name__}: {e}")
            finally:
                with self._cv:
                    self._busy -= 1
                    self._cv.notify_all()

    def _execute(self, run_id: str) -> None:
        with self._cv:
            rec = self._jobs.get(run_id)
            if rec is None or rec.get("status") != "queued" or self._stopping:
                return
            rec["status"] = "running"
            rec["started_at"] = iso()
            self._save(rec)
            argv = list(rec["argv"])
        try:
            run_dir = self.store.run_dir(run_id)
        except RunNotFound:
            self._finish(run_id, "failed", error="the run directory disappeared before the run started")
            return
        console = run_dir / "console.log"
        try:
            with open(console, "ab") as out:
                out.write(f"$ {' '.join(argv)}\n".encode("utf-8", "replace"))
                out.flush()
                proc = subprocess.Popen(
                    argv, cwd=str(self.work_dir(run_dir)), stdin=subprocess.DEVNULL, stdout=out,
                    stderr=subprocess.STDOUT,
                    env=_subprocess_env(self.settings), close_fds=True, **_popen_group_kwargs(),
                )
        except OSError as e:
            self._finish(run_id, "failed", error=f"could not start malvalid: {e}")
            return
        with self._cv:
            rec["pid"] = proc.pid
            self._procs[run_id] = proc
            self._save(rec)
            cancel_now = run_id in self._cancel or self._stopping
        if cancel_now:
            self._terminate(run_id, proc.pid, proc)
        log.info("run %s started (pid %d)", run_id, proc.pid)
        rc = proc.wait()
        self._complete(run_id, rc, run_dir)

    def _complete(self, run_id: str, rc: int | None, run_dir: Path) -> None:
        report = read_json_file(run_dir / "report.json")
        report_ok = report is not None and isinstance(report.get("verdict"), dict)
        with self._lock:
            cancelled = run_id in self._cancel
            stopping = self._stopping
        if cancelled:
            self._finish(run_id, "cancelled", exit_code=rc, error="cancelled by the user")
        elif stopping and not report_ok:
            self._finish(run_id, "interrupted", exit_code=rc,
                         error="the web server was stopped while this run was in progress")
        elif rc in (0, 1) and report_ok:
            self._finish(run_id, "finished", exit_code=rc)
        else:
            self._finish(run_id, "failed", exit_code=rc, error=self._failure_reason(rc, run_dir, report))
        log.info("run %s ended with exit code %s", run_id, rc)

    def _failure_reason(self, rc: int | None, run_dir: Path, report: dict[str, Any] | None) -> str:
        if report is not None and rc not in (0, 1):
            gate = report.get("gate") if isinstance(report.get("gate"), dict) else {}
            meaning = gate.get("exit_meaning") if isinstance(gate, dict) else None
            if isinstance(rc, int) and rc < 0:  # e.g. a native library crashing at interpreter exit
                try:
                    sig = signal.Signals(-rc).name
                except ValueError:
                    sig = f"signal {-rc}"
                return f"MalValid crashed ({sig}) after writing report.json"
            return f"exit code {rc}: " + (str(meaning) if meaning else "the gate result is not trustworthy")
        msg = _last_error_line(tail_text(run_dir / "console.log", max_lines=200))
        if msg:
            return msg
        if rc is not None and rc < 0:
            try:
                name = signal.Signals(-rc).name
            except ValueError:
                name = f"signal {-rc}"
            return f"malvalid was killed by {name}"
        if report is None:
            return f"malvalid exited with code {rc} without writing report.json (see console.log)"
        return f"malvalid exited with code {rc}"

    # ---- start-up reconciliation -------------------------------------------------------------

    def reconcile(self) -> list[str]:
        """Fix up job.json files an earlier server left ``queued``/``running``. Returns the run ids touched."""
        touched: list[str] = []
        for rid in self.store.list_run_ids():
            try:
                d = self.store.run_dir(rid)
                job = self.store.read_job(rid)
            except RunNotFound:
                continue
            if job is None or job.get("status") not in ACTIVE_STATUSES:
                continue
            with self._lock:
                if rid in self._jobs:
                    continue
                touched.append(rid)
                self._jobs[rid] = job
                pid = job.get("pid")
                if job.get("status") == "running" and pid_alive(pid, expect=("malvalid", str(d))):
                    self._adopted[rid] = int(pid)
                    log.info("run %s is still running (pid %s); watching it", rid, pid)
                    threading.Thread(target=self._watch_adopted, args=(rid, int(pid), d),
                                     name=f"malvalid-web-adopt-{rid}", daemon=True).start()
                    continue
            report = read_json_file(d / "report.json")
            started = parse_iso(job.get("started_at"))
            rep_started = parse_iso((report or {}).get("run", {}).get("started_at")
                                    if isinstance((report or {}).get("run"), dict) else None)
            if (job.get("status") == "running" and report is not None and isinstance(report.get("verdict"), dict)
                    and (started is None or rep_started is None or rep_started >= started.replace(microsecond=0))):
                code = (report.get("gate") or {}).get("exit_code") if isinstance(report.get("gate"), dict) else None
                status = "finished" if code in (0, 1) else "failed"
                self._finish(rid, status, exit_code=code if isinstance(code, int) else None,
                             note="the run completed while the web server was not running",
                             error=None if status == "finished" else self._failure_reason(code, d, report),
                             finished_at=_report_finished_at(report))
            else:
                what = "started" if job.get("status") == "running" else "was queued"
                self._finish(rid, "interrupted",
                             error=f"the web server stopped while this run {what}; its process is gone"
                             if what == "started" else
                             "the web server stopped before this run started")
        return touched

    def _watch_adopted(self, run_id: str, pid: int, run_dir: Path) -> None:
        while pid_alive(pid, expect=("malvalid", str(run_dir))):
            with self._lock:
                if self._stopping and run_id not in self._adopted:
                    return
            time.sleep(ADOPT_POLL_S)
        report = read_json_file(run_dir / "report.json")
        with self._lock:
            cancelled = run_id in self._cancel
            stopping = self._stopping
        if cancelled:
            self._finish(run_id, "cancelled", error="cancelled by the user")
        elif report is not None and isinstance(report.get("verdict"), dict):
            code = (report.get("gate") or {}).get("exit_code") if isinstance(report.get("gate"), dict) else None
            status = "finished" if code in (0, 1) else "failed"
            self._finish(run_id, status, exit_code=code if isinstance(code, int) else None,
                         error=None if status == "finished" else self._failure_reason(code, run_dir, report),
                         finished_at=_report_finished_at(report))
        elif stopping:
            self._finish(run_id, "interrupted", error="the web server was stopped while this run was in progress")
        else:
            self._finish(run_id, "failed", error=self._failure_reason(None, run_dir, None))

    # ---- adapter validation ------------------------------------------------------------------

    def _json_subprocess(self, argv: list[str], *, cwd: Path, timeout: float) -> dict[str, Any]:
        """Run a ``malvalid … --json`` command (at most VALIDATION_CONCURRENCY at once) and parse its output.

        Returns ``{"result": dict | None, "exit_code", "timed_out", "elapsed_s", "stderr", "start_error"}``.
        """
        t0 = time.monotonic()
        with self._validate_sem:
            try:
                proc = subprocess.Popen(
                    argv, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=_subprocess_env(self.settings), close_fds=True, **_popen_group_kwargs(),
                )
            except OSError as e:
                return {"result": None, "exit_code": None, "timed_out": False, "elapsed_s": 0.0, "stderr": "",
                        "start_error": f"could not start malvalid: {e}"}
            timed_out = False
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_tree(proc.pid, proc)
                if _WINDOWS:  # a straggler that escaped the tree kill must not hang us on its pipe handles
                    try:
                        out, err = proc.communicate(timeout=KILL_DRAIN_TIMEOUT_S)
                    except subprocess.TimeoutExpired:
                        with contextlib.suppress(OSError):
                            proc.kill()
                        with contextlib.suppress(subprocess.TimeoutExpired):
                            proc.wait(timeout=KILL_DRAIN_TIMEOUT_S)
                        out, err = b"", b""
                else:
                    out, err = proc.communicate()
        stdout = out.decode("utf-8", "replace")
        stderr = err.decode("utf-8", "replace")
        result: dict[str, Any] | None = None
        with contextlib.suppress(ValueError):
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                result = parsed
        if result is None:
            i = stdout.find("{")
            if i >= 0:
                with contextlib.suppress(ValueError):
                    parsed = json.loads(stdout[i:])
                    if isinstance(parsed, dict):
                        result = parsed
        return {"result": result, "exit_code": proc.returncode, "timed_out": timed_out,
                "elapsed_s": round(time.monotonic() - t0, 3), "stderr": stderr, "start_error": None}

    @staticmethod
    def _stderr_tail(stderr: str) -> str:
        tail = "\n".join(stderr.strip().splitlines()[-40:])
        return tail[-8000:] if tail else ""

    def validate(self, argv: list[str], *, cwd: Path, timeout_s: float | None = None) -> dict[str, Any]:
        """Run ``malvalid validate-adapter --json …`` in a subprocess; returns its report + exit status.

        Never raises for validator problems: they come back as ``{"ok": False, "error": …}``.
        """
        timeout = float(timeout_s if timeout_s is not None else self.settings.validate_timeout_s)
        raw = self._json_subprocess(argv, cwd=cwd, timeout=timeout)
        if raw["start_error"]:
            return {"ok": False, "checks": [], "exit_code": None, "error": raw["start_error"]}
        result = raw["result"]
        res: dict[str, Any] = dict(result or {"ok": False, "checks": []})
        res.setdefault("checks", [])
        res["exit_code"] = raw["exit_code"]
        res["timed_out"] = raw["timed_out"]
        res["elapsed_s"] = raw["elapsed_s"]
        res["stderr_tail"] = self._stderr_tail(raw["stderr"])
        if raw["timed_out"]:
            res["ok"] = False
            res["error"] = f"validation did not finish within {timeout:g} s and was stopped"
        elif result is None:
            res["ok"] = False
            res["error"] = _last_error_line(raw["stderr"]) or (
                f"malvalid validate-adapter exited with code {raw['exit_code']} without a result")
        else:
            res.setdefault("error", None)
        return res

    # ---- model inspection --------------------------------------------------------------------

    def inspect(self, model: Path, *, cwd: Path, timeout_s: float | None = None) -> dict[str, Any]:
        """Run ``malvalid inspect-model --json MODEL`` in a subprocess (the web server never opens a model
        itself). Returns its result (``ok, format, model_kind, n_features, feature_version, …``) plus
        ``exit_code`` and ``error`` (set when the inspection itself failed: timeout, crash, no output)."""
        timeout = float(timeout_s if timeout_s is not None else min(self.settings.validate_timeout_s,
                                                                    INSPECT_TIMEOUT_S))
        argv = self.settings.malvalid_command() + ["inspect-model", "--json", str(model)]
        raw = self._json_subprocess(argv, cwd=cwd, timeout=timeout)
        if raw["start_error"]:
            return {"ok": False, "notes": [], "errors": [], "exit_code": None, "error": raw["start_error"]}
        result = raw["result"]
        res: dict[str, Any] = dict(result or {"ok": False})
        for k in ("notes", "errors"):
            v = res.get(k)
            res[k] = [str(x) for x in v if isinstance(x, (str, int, float))] if isinstance(v, list) else []
        res["exit_code"] = raw["exit_code"]
        res["timed_out"] = raw["timed_out"]
        res["elapsed_s"] = raw["elapsed_s"]
        if raw["timed_out"]:
            res["ok"] = False
            res["error"] = f"inspecting the model did not finish within {timeout:g} s and was stopped"
        elif result is None:
            res["ok"] = False
            res["error"] = _last_error_line(raw["stderr"]) or (
                f"malvalid inspect-model exited with code {raw['exit_code']} without a result")
        else:
            res["ok"] = bool(res.get("ok"))
            res["error"] = None
        return res


__all__ = ["JobError", "JobManager", "JobSpec"]
