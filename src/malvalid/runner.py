"""Run orchestration: from a researcher's adapter to a production-readiness verdict.

:func:`run_gate` is the whole pipeline behind ``malvalid run``:

1. validate the gate config against the plugin registry;
2. inspect the adapter inside the sandbox (declarations only — ``load()`` is not called);
3. parse the training manifest and cutoff, resolve the feature schema, load and verify the
   canonical corpus (an unavailable or mismatched corpus is recorded, not fatal: modules that need
   the feature space are skipped with that reason);
4. **M0 file safety always runs first**, against an unloaded placeholder model; if it aborts, every
   other module is skipped and the verdict is BLOCKED — the untrusted artifact is never loaded;
5. load the model in the sandbox, verify tree access (tree dump vs ``predict_proba``), compute the
   run's capabilities;
6. run each enabled module in scorecard order with its own seed, deadline and SIGALRM timer;
   unmet requirements => ``skipped`` with the specific reason, exceptions => ``error``;
7. compute the verdict + exit code and write ``report.json`` (+ ``report.html``) and ``run.log``.

With :attr:`RunOptions.progress_path` set (``malvalid run`` always sets ``<out>/progress.json``), the
run also keeps a small ``malvalid-progress/1`` JSON file up to date at every stage, so a UI (the
local web UI's run page) can show which module is running. Failing to write it never breaks a run.

The sandbox API is imported lazily behind small seams (:func:`_make_policy`,
:func:`_inspect_adapter`, :func:`_open_model`, :func:`_write_html`, :func:`_file_safety_cls`) so the
runner can be unit-tested with :class:`malvalid.testing.InProcessModel` fakes.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import math
import os
import signal
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import numpy as np

from malvalid import REPORT_SCHEMA_VERSION, __version__, registry
from malvalid.config import GateConfig, module_params, validate_against_registry
from malvalid.context import (
    REQUIREMENT_HINTS,
    ArtifactStore,
    ModelDeclarations,
    ModelHandle,
    RunContext,
    compute_capabilities,
    module_seed,
)
from malvalid.core import (
    AdapterContractError,
    AdapterError,
    ConfigError,
    CorpusUnavailable,
    GateMode,
    GateOutcome,
    MalValidError,
    Module,
    ModuleResult,
    ModuleTimeout,
    Requirement,
    SandboxError,
    Status,
    skipped_result,
    to_jsonable,
    unmet_requirements,
)
from malvalid.manifest import TrainingManifest, count_excluded_members, load_training_manifest
from malvalid.verdict import Verdict, compute_verdict, exit_code_for

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.corpora.base import Corpus
    from malvalid.loaders.trees import TreeEnsemble
    from malvalid.schemas.base import FeatureSchema

log = logging.getLogger("malvalid.runner")

FILE_SAFETY_ID = "file_safety"
TREE_FIDELITY_TOL = 1e-4
TREE_FIDELITY_MAX_ROWS = 2000
TREE_FIDELITY_RANDOM_ROWS = 512
FAIL_ON_CHOICES = ("blocked", "not_ready", "conditional")

PROGRESS_SCHEMA = "malvalid-progress/1"
#: Stages of ``progress.json`` in the order a run passes through them (``failed`` can follow any).
PROGRESS_STAGES = ("starting", "inspecting", "scanning", "loading", "modules", "verdict", "writing", "done", "failed")

EXIT_MEANINGS = {
    0: "passed: no hard gate failed, no module errored, verdict above --fail-on",
    1: "gate failed: the run is BLOCKED (a hard gate failed / could not be evaluated / M0 aborted) "
    "or the verdict is at or below --fail-on",
    2: "error: a module errored or the model could not be loaded — the gate result is not trustworthy",
}

#: Shown in the verdict, the warnings and the disclaimers when the model ran with ``isolation: process_only``.
REDUCED_ISOLATION_NOTE = (
    "Reduced isolation: this platform has no OS sandbox. The model ran in a separate worker process (pickle "
    "refusal, resource limits where supported) without network or file-system isolation."
)
_ISOLATION_WARNINGS = {
    "process_only": REDUCED_ISOLATION_NOTE,
    "none": "No isolation: the sandbox was disabled and the model ran inside the malvalid process.",
}


def _add_isolation(verdict_block: dict[str, Any], isolation: str | None) -> None:
    """``verdict.isolation`` (os_sandbox | process_only | none | None) and, when it is not the full OS
    sandbox, ``verdict.isolation_warning`` appended to the summary. The verdict and score are unchanged."""
    verdict_block["isolation"] = isolation
    warning = _ISOLATION_WARNINGS.get(str(isolation))
    verdict_block["isolation_warning"] = warning
    if warning and verdict_block.get("summary"):
        verdict_block["summary"] = f"{verdict_block['summary']} {warning}"


BASE_DISCLAIMERS = (
    "The verdict and score apply the gate policy recorded under `config` (thresholds, gate modes, "
    "weights) to the canonical corpus recorded under `corpus`. They are evidence for a promotion "
    "decision, not a guarantee of behaviour on your production traffic.",
    "Skipped modules never count as passes: they lower coverage, and each skip reason is listed.",
    "Screening modules (marked 'screening') can surface red flags but cannot certify their absence; "
    "for example, a clean backdoor screen is not proof that the model is backdoor-free.",
    "This version of malvalid does not evaluate adversarial (evasion) robustness; a READY verdict "
    "makes no claim about resistance to evasion attacks.",
)


# --------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------


@dataclass
class RunOptions:
    """Inputs of one gate run (the CLI's ``malvalid run`` builds this)."""

    adapter: Path
    config: GateConfig
    out_dir: Path
    allow_pickle: bool = False
    class_name: str | None = None
    model_paths: Sequence[Path] = ()
    only: Sequence[str] | None = None  # module ids or codes; M0 always runs
    skip: Sequence[str] = ()
    fail_on: str = "blocked"
    write_html: bool = True
    command: str = ""
    #: Where to keep the ``malvalid-progress/1`` stage file (None = don't write one).
    progress_path: Path | None = None
    #: The run id recorded in report.json / report.html / progress.json (None = a new
    #: ``YYYYMMDDTHHMMSSZ-<8 hex>`` id). ``malvalid serve`` passes its web run id so all of them agree.
    run_id: str | None = None
    #: Notes from preparing a model-file-only submission (recorded as report warnings).
    notes: Sequence[str] = ()


@dataclass
class RunOutcome:
    report: dict[str, Any]
    exit_code: int
    report_json: Path
    report_html: Path | None


def run_gate(opts: RunOptions) -> RunOutcome:
    """Run the full gate for one submitted detector and write the run directory.

    Raises :class:`ConfigError` / :class:`AdapterError` (and other :class:`MalValidError`) for
    problems that prevent any meaningful run — an invalid config, a missing or non-conforming
    adapter, an unknown feature version. Everything after the adapter has been inspected is
    reported in ``report.json`` instead (a load failure yields a BLOCKED report with exit code 2).
    """
    validate_fail_on(opts.fail_on)
    out_dir = Path(opts.out_dir).expanduser().resolve()
    private_dir = out_dir / "private"
    out_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    with _run_log(out_dir / "run.log") as run_log:
        gate_run: _GateRun | None = None
        try:
            gate_run = _GateRun(opts, out_dir, private_dir)
            return gate_run.execute()
        except BaseException as e:
            if gate_run is not None:
                gate_run.progress.fail(e)
            # Record the failure in run.log only: the caller (e.g. the CLI) reports it to the user,
            # and a traceback on the console would bury the one-line actionable message.
            if isinstance(e, MalValidError):
                msg, exc_info = "run failed: %s", None
            elif isinstance(e, KeyboardInterrupt):
                msg, exc_info = "run interrupted%s", None
            else:
                msg, exc_info = "internal error: %r", (type(e), e, e.__traceback__)
            rec = log.makeRecord(log.name, logging.ERROR, __file__, 0, msg, (e,), exc_info)
            run_log.handle(rec)
            raise


def validate_fail_on(fail_on: str) -> str:
    if fail_on not in FAIL_ON_CHOICES:
        raise ConfigError(f"--fail-on must be one of {', '.join(FAIL_ON_CHOICES)} (got {fail_on!r})")
    return fail_on


def resolve_module_ids(names: Sequence[str] | None, *, flag: str = "--only") -> list[str] | None:
    """Map module ids or scorecard codes (``performance`` / ``M1``, comma-separated ok) to ids."""
    if names is None:
        return None
    mods = registry.modules()
    by_code = {str(getattr(c, "code", "")).lower(): mid for mid, c in mods.items()}
    out: list[str] = []
    for item in names:
        for raw in str(item).split(","):
            name = raw.strip()
            if not name:
                continue
            if name in mods:
                mid = name
            elif name.lower() in by_code:
                mid = by_code[name.lower()]
            else:
                why = registry.unavailable("modules").get(name)
                extra = f" (it failed to import: {why})" if why else ""
                raise ConfigError(
                    f"{flag}: unknown module {name!r}{extra}; known: "
                    + ", ".join(f"{c.code}={m}" for m, c in mods.items())
                )
            if mid not in out:
                out.append(mid)
    return out


# --------------------------------------------------------------------------------------------------
# Seams (lazy imports of components owned by other agents; monkeypatched in tests)
# --------------------------------------------------------------------------------------------------


def _make_policy(cfg: GateConfig, *, run_dir: Path, allow_pickle: bool, extra_ro: Sequence[Path]) -> Any:
    try:
        from malvalid.sandbox.host import SandboxPolicy
    except ImportError as e:
        raise SandboxError(f"the malvalid sandbox is not available in this installation ({e})") from e
    return SandboxPolicy.from_config(cfg, run_dir=run_dir, allow_pickle=allow_pickle, extra_ro=extra_ro)


def _inspect_adapter(
    adapter: Path, policy: Any, *, class_name: str | None, model_paths_override: Sequence[Path]
) -> ModelDeclarations:
    try:
        from malvalid.sandbox.host import inspect_adapter
    except ImportError as e:
        raise SandboxError(f"the malvalid sandbox is not available in this installation ({e})") from e
    return inspect_adapter(adapter, policy, class_name=class_name, model_paths_override=model_paths_override)


def _open_model(adapter: Path, policy: Any, *, declarations: ModelDeclarations, class_name: str | None) -> ModelHandle:
    try:
        from malvalid.sandbox.host import open_model
    except ImportError as e:
        raise SandboxError(f"the malvalid sandbox is not available in this installation ({e})") from e
    return open_model(adapter, policy, declarations=declarations, class_name=class_name)


def _write_html(report: dict[str, Any], path: Path) -> Path:
    from malvalid.report.html import write_html

    return Path(write_html(report, path))


def _file_safety_cls() -> type[Module]:
    return registry.get_module(FILE_SAFETY_ID)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="seconds").replace("+00:00", "Z")


@contextlib.contextmanager
def _run_log(path: Path) -> Iterator[logging.Handler]:
    """Mirror the ``malvalid`` logger into ``run.log`` for the duration of the run."""
    lg = logging.getLogger("malvalid")
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    old_level = lg.level
    if lg.getEffectiveLevel() > logging.INFO:
        lg.setLevel(logging.INFO)
    lg.addHandler(handler)
    try:
        yield handler
    finally:
        lg.removeHandler(handler)
        handler.close()
        lg.setLevel(old_level)


@contextlib.contextmanager
def _alarm(seconds: float | None, on_fire: Callable[[], None], message: str) -> Iterator[bool]:
    """Raise :class:`ModuleTimeout` in the main thread after ``seconds`` (SIGALRM).

    Yields whether the timer is armed (it cannot be outside the main thread or on platforms
    without SIGALRM; the RunContext / sandbox deadlines still apply there).
    """
    armed = (
        seconds is not None
        and seconds > 0
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )
    if not armed:
        yield False
        return

    def _handler(signum: int, frame: Any) -> None:
        on_fire()
        raise ModuleTimeout(message)

    prev_handler = signal.signal(signal.SIGALRM, _handler)
    prev_delay, _ = signal.setitimer(signal.ITIMER_REAL, float(seconds))  # type: ignore[arg-type]
    t0 = time.monotonic()
    try:
        yield True
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        finally:
            signal.signal(signal.SIGALRM, prev_handler)
            if prev_delay > 0:  # restore an outer timer (e.g. a test-suite watchdog)
                signal.setitimer(signal.ITIMER_REAL, max(prev_delay - (time.monotonic() - t0), 1e-3))


def _sha256_file(p: Path) -> str:
    from malvalid.corpora.base import sha256_file

    return sha256_file(p)


def _artifact_info(path: str | Path) -> dict[str, Any]:
    """Identity of a model artifact (hash, size, sniffed format) — read as bytes, never loaded."""
    from malvalid.loaders.base import is_pickle_artifact, sniff_format

    p = Path(path)
    info: dict[str, Any] = {"path": str(p), "sha256": None, "size": None, "format": None, "is_pickle": None}
    try:
        if p.is_file():
            info.update(
                sha256=_sha256_file(p),
                size=p.stat().st_size,
                format=sniff_format(p),
                is_pickle=bool(is_pickle_artifact(p)),
            )
        elif p.is_dir():
            info.update(format="directory", is_pickle=None)
        else:
            info["error"] = "file not found"
    except OSError as e:
        info["error"] = f"cannot read artifact: {e}"
    return info


def _first_line(text: str, limit: int = 300) -> str:
    s = (text or "").strip().splitlines()
    line = s[0] if s else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


class _UnloadedModel:
    """Stand-in handed to M0: only ``declarations`` is valid — the model is not loaded yet."""

    def __init__(self, declarations: ModelDeclarations):
        self.declarations = declarations

    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the model is not loaded: file safety (M0) runs before load()")

    predict_proba = _refuse
    predict = _refuse
    featurize = _refuse

    def has_featurize(self) -> bool:
        return False

    def tree_ensemble(self) -> None:
        return None

    @property
    def query_count(self) -> int:
        return 0

    def close(self) -> None:
        pass


def _gate_for(cfg: GateConfig, cls: type[Module]) -> GateMode:
    """Configured gate mode, or the module's default when the config does not set one."""
    mc = cfg.modules.get(cls.id)
    if mc is not None and "gate" in mc.model_fields_set:
        return GateMode(mc.gate)
    return cls.default_gate


def _skip_reason(
    cls: type[Module], missing: Sequence[Requirement], why: dict[Requirement, str] | None = None
) -> str:
    """The researcher-facing reason a module was not run.

    ``why`` replaces the generic :data:`REQUIREMENT_HINTS` text with a run-specific explanation,
    e.g. the corpus error for FEATURE_SPACE or the failed tree-fidelity check for TREE_ACCESS.
    """
    why = why or {}

    def hint(r: Requirement) -> str:
        return why.get(r) or REQUIREMENT_HINTS.get(r, r.value)

    parts = [hint(r) for r in missing if r in cls.requires]
    any_missing = [r for r in missing if r in cls.requires_any and r not in cls.requires]
    if any_missing:
        if len(cls.requires_any) > 1:
            names = " or ".join(r.value.replace("_", " ") for r in cls.requires_any)
            parts.append(f"needs {names}, and none is available: " + "; ".join(hint(r) for r in any_missing))
        else:
            parts.extend(hint(r) for r in any_missing)
    return "; ".join(parts) if parts else "requirements not met"


class _Progress:
    """The run's ``progress.json`` (schema ``malvalid-progress/1``), rewritten atomically per stage.

    ``{schema, run_id, pid, stage, message, module, modules, started_at, updated_at, verdict,
    exit_code}``: ``stage`` is one of :data:`PROGRESS_STAGES`; ``module`` is the running module
    ``{id, code, title}`` or None; ``modules`` lists every planned module in scorecard order with
    ``status`` (``pending | running | pass | warn | fail | skipped | error``) and ``duration_s``;
    ``verdict`` is ``{verdict, label, score, coverage}`` once known. Readers must tolerate a
    missing or stale file. Every write is best-effort: a failure is logged at debug level and never
    interrupts the run.
    """

    def __init__(self, path: Path | str | None, run_id: str, started: dt.datetime):
        self.path: Path | None = None
        if path is not None:
            try:
                self.path = Path(path).expanduser().resolve()
            except (OSError, RuntimeError, ValueError) as e:  # pragma: no cover - defensive
                log.debug("progress file %s unusable: %s", path, e)
        self.state: dict[str, Any] = {
            "schema": PROGRESS_SCHEMA,
            "run_id": run_id,
            "pid": os.getpid(),
            "stage": "starting",
            "message": "starting the run",
            "module": None,
            "modules": [],
            "started_at": _iso(started),
            "updated_at": _iso(started),
            "verdict": None,
            "exit_code": None,
        }
        self._write()

    # ---- updates -------------------------------------------------------------------------------

    def set_modules(self, classes: Sequence[Any]) -> None:
        """The planned modules (scorecard order); all start as ``pending``."""
        mods: list[dict[str, Any]] = []
        seen: set[str] = set()
        for c in classes:
            mid = str(getattr(c, "id", "") or "")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            mods.append({"id": mid, "code": getattr(c, "code", None), "title": getattr(c, "title", None),
                         "status": "pending", "duration_s": None})
        self.state["modules"] = mods
        self._write()

    def stage(self, stage: str, message: str = "") -> None:
        self.state["stage"] = stage
        self.state["message"] = message
        if stage not in ("scanning", "modules"):
            self.state["module"] = None
        self._write()

    def _entry(self, module_id: str, code: Any = None, title: Any = None) -> dict[str, Any]:
        for m in self.state["modules"]:
            if m["id"] == module_id:
                return m
        m = {"id": module_id, "code": code, "title": title, "status": "pending", "duration_s": None}
        self.state["modules"].append(m)
        return m

    def module_started(self, cls: type[Module]) -> None:
        m = self._entry(cls.id, cls.code, cls.title)
        m["status"] = "running"
        self.state["module"] = {"id": cls.id, "code": cls.code, "title": cls.title}
        self.state["message"] = f"{cls.code} {cls.title}: running"
        self._write()

    def module_finished(self, res: ModuleResult) -> None:
        m = self._entry(res.module_id, res.code, res.title)
        status = res.status.value if isinstance(res.status, Status) else str(res.status)
        m["status"] = status
        m["duration_s"] = res.duration_s
        cur = self.state.get("module")
        if isinstance(cur, dict) and cur.get("id") == res.module_id:
            self.state["module"] = None
        self._write()

    def set_verdict(self, block: dict[str, Any]) -> None:
        self.state["verdict"] = {k: block.get(k) for k in ("verdict", "label", "score", "coverage")}
        self._write()

    def done(self, exit_code: int, message: str) -> None:
        self.state["exit_code"] = int(exit_code)
        self.stage("done", message)

    def fail(self, e: BaseException) -> None:
        if isinstance(e, KeyboardInterrupt):
            msg = "run interrupted"
        elif isinstance(e, MalValidError):
            msg = f"run failed: {_first_line(str(e), 2000)}"
        else:
            msg = f"internal error: {type(e).__name__}: {_first_line(str(e), 2000)}"
        cur = self.state.get("module")
        if isinstance(cur, dict):
            self._entry(str(cur.get("id")))["status"] = "error"
        self.stage("failed", msg)

    # ---- I/O -----------------------------------------------------------------------------------

    def _write(self) -> None:
        if self.path is None:
            return
        self.state["updated_at"] = _iso(_utcnow())
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            text = json.dumps(to_jsonable(self.state), allow_nan=False)
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            from malvalid._platform import replace_file  # Windows: brief retry while the web UI reads it

            replace_file(tmp, self.path)
        except Exception as e:  # progress is a convenience: never let it break a run
            log.debug("could not write progress file %s: %s", self.path, e)
            with contextlib.suppress(OSError):
                tmp.unlink()


# --------------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------------


@dataclass
class _CorpusState:
    name: str
    corpus: "Corpus | None" = None  # loaded corpus (may be unusable, see error)
    error: str | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> "Corpus | None":
        return self.corpus if self.error is None else None


class _GateRun:
    def __init__(self, opts: RunOptions, out_dir: Path, private_dir: Path):
        self.opts = opts
        self.cfg: GateConfig = opts.config
        self.out_dir = out_dir
        self.private_dir = private_dir
        self.started = _utcnow()
        self.t0 = time.monotonic()
        self.run_id = getattr(opts, "run_id", None) or f"{self.started:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.progress = _Progress(getattr(opts, "progress_path", None), self.run_id, self.started)
        self.warnings: list[str] = []
        self.artifacts = ArtifactStore(private_dir)
        self.results: list[ModuleResult] = []
        self.abort_reason: str | None = None
        self.load_error: str | None = None
        self.model_dead: str | None = None
        self.model: Any = None
        self.caps: frozenset[Requirement] = frozenset()
        self.tree_info: dict[str, Any] = {"available": False, "n_trees": None, "fidelity_max_abs_diff": None,
                                          "note": "model not loaded"}
        self.max_s = float(self.cfg.runtime.max_seconds_per_module)

    # ---- small utilities -----------------------------------------------------------------------

    def warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)

    @property
    def seed(self) -> int:
        return int(self.cfg.runtime.seed)

    # ---- pipeline ------------------------------------------------------------------------------

    def execute(self) -> RunOutcome:
        cfg, opts = self.cfg, self.opts
        log.info("malvalid %s — run %s → %s", __version__, self.run_id, self.out_dir)
        unavailable = registry.unavailable("modules")
        for w in validate_against_registry(cfg):
            # "verdict config references unknown module 'x'" repeats "module 'x' unavailable (...)".
            if w.startswith("verdict config references unknown module") and any(
                f"'{mid}'" in w for mid in unavailable
            ):
                log.debug("config: %s", w)
                continue
            self.warn(f"config: {w}")
        plan, disabled = self._plan_modules()
        self.progress.set_modules([self._m0_identity(), *plan])
        allow_pickle = bool(opts.allow_pickle or cfg.runtime.allow_pickle)

        adapter = Path(opts.adapter).expanduser().resolve()
        if not adapter.exists():
            raise AdapterError(f"adapter not found: {adapter}")
        overrides: list[Path] = []
        for p in opts.model_paths or ():
            rp = Path(p).expanduser().resolve()
            if not rp.exists():
                raise AdapterError(f"--model {p}: file not found")
            overrides.append(rp)
        extra_ro = [adapter.parent] + sorted({p.parent for p in overrides})
        if not cfg.runtime.sandbox:
            self.warn(
                "the sandbox is disabled (runtime.sandbox: false / --no-sandbox): the adapter and "
                "model run in-process with no isolation — use this only for trusted models while debugging"
            )
        policy = _make_policy(cfg, run_dir=self.out_dir, allow_pickle=allow_pickle, extra_ro=extra_ro)

        log.info("inspecting adapter %s (declarations only; load() is not called yet)", adapter)
        self.progress.stage("inspecting", f"inspecting the adapter {adapter.name} (declarations only; load() is not called yet)")
        decl = _inspect_adapter(adapter, policy, class_name=opts.class_name, model_paths_override=overrides)
        self.decl = decl
        self.manifest = load_training_manifest(decl)
        for w in self.manifest.warnings:
            self.warn(w)
        self.hashes = self.manifest.usable_hashes()
        self.cutoff = self.manifest.cutoff_parsed
        self.schema = self._resolve_schema(decl.feature_version)
        self.progress.stage("inspecting", f"loading the canonical corpus {cfg.corpus} (verifying content hashes)")
        self.cs = self._load_corpus(decl.feature_version)
        for note in getattr(opts, "notes", ()) or ():
            self.warn(str(note))
        self.calibration: dict[str, Any] | None = None
        spec = decl.extras.get("spec") if isinstance(decl.extras, dict) else None
        if isinstance(spec, dict) and spec.get("threshold_source") == "calibrate":
            self._plan_calibration(float(spec.get("calibrate_fpr") or 0.0))
        self.sample_dir = Path(self.cfg.sample_dir).expanduser() if self.cfg.sample_dir else None
        self.excluded = count_excluded_members(self.cs.usable, self.hashes)
        if self.excluded:
            log.info("%d training members found in the evaluation split; they are excluded from evaluation",
                     self.excluded)
        self.adapter = adapter
        self.allow_pickle = allow_pickle
        self.policy = policy
        self.model_artifacts = [{**_artifact_info(p), "declared": True} for p in decl.model_paths]

        # ---- M0 first, before anything is deserialized ----
        self.progress.stage("scanning", "scanning the model artifact(s) before anything is loaded (M0 file safety)")
        m0 = self._run_file_safety()
        self.results.append(m0)
        self.progress.module_finished(m0)
        self._merge_scanned_artifacts(m0)
        if m0.details.get("abort") or m0.status is Status.ERROR:
            if m0.status is Status.ERROR:
                self.abort_reason = (
                    f"M0 file-safety scan did not complete ({_first_line(m0.finding)}); "
                    "refusing to load an unscanned model"
                )
            else:
                self.abort_reason = f"M0 {m0.title}: {_first_line(m0.finding)}"
            log.error("run aborted by M0: %s", self.abort_reason)
            for cls in plan:
                self._append_skip(cls, self.deselected.get(cls.id) or f"run aborted by M0: {_first_line(m0.finding)}")
        else:
            self._load_and_run(plan)
        return self._finish(disabled)

    # ---- threshold calibration (model-file-only submissions) -------------------------------------

    def _plan_calibration(self, target_fpr: float) -> None:
        """Hold out the calibration slice before anything is evaluated (see :mod:`malvalid.submission`)."""
        from malvalid.submission import MIN_CALIBRATION_ROWS, hold_out_calibration_slice

        corpus = self.cs.usable
        if corpus is None:
            raise AdapterError(
                "automatic threshold calibration needs the canonical corpus, which is not usable "
                f"({self.cs.error}); install the corpus or give the operating threshold yourself (--threshold)"
            )
        period = str(getattr(self.cfg.runtime, "calibration_period", None) or "earliest")
        view, idx, meta = hold_out_calibration_slice(corpus, self.hashes, period)
        if meta.get("fallback_reason"):
            self.warn(f"threshold calibration: calibration_period {period}: {meta['fallback_reason']}")
        if idx.size < MIN_CALIBRATION_ROWS:
            raise AdapterError(
                f"the corpus {corpus.name} has only {idx.size} benign rows available for threshold calibration "
                f"(at least {MIN_CALIBRATION_ROWS} needed); give the operating threshold yourself (--threshold)"
            )
        self.cs.corpus = view
        self.calibration = {"target_fpr": target_fpr, "indices": idx, **meta}
        log.info("holding out %d benign rows of %s for threshold calibration (%s)", idx.size, corpus.name,
                 meta["rule"])

    def _calibrate(self) -> None:
        """Score the held-out slice in the sandbox and fix the operating threshold for the whole run."""
        import dataclasses

        from malvalid.submission import calibrate_threshold

        cal = self.calibration
        assert cal is not None and self.model is not None and self.cs.corpus is not None
        self.progress.stage("loading", f"calibrating the operating threshold to {cal['target_fpr']:.2%} FPR on "
                                       f"{cal['indices'].size:,} held-out benign rows")
        corpus = self.cs.corpus
        idx = cal["indices"]
        scores = np.empty(idx.size, dtype=np.float64)
        step = 50000
        for s0 in range(0, idx.size, step):
            part = idx[s0 : s0 + step]
            p = np.asarray(self.model.predict_proba(corpus.take(part)), dtype=np.float64).reshape(-1)
            if p.shape[0] != part.size:
                raise AdapterContractError(f"predict_proba returned {p.shape[0]} scores for {part.size} rows")
            bad = ~np.isfinite(p)
            if bad.any():  # the sandbox handle enforces this too; other handles may not
                raise AdapterContractError(
                    f"predict_proba returned {int(bad.sum())} non-finite value(s) (NaN/inf; first at row "
                    f"{int(np.flatnonzero(bad)[0]) + s0}) on the held-out calibration rows"
                )
            scores[s0 : s0 + part.size] = p
        t, achieved = calibrate_threshold(scores, cal["target_fpr"])
        info = {k: v for k, v in cal.items() if k != "indices"}
        info.update(threshold=t, achieved_fpr=achieved, n_benign=int(idx.size))
        extras = dict(self.decl.extras)
        extras["threshold_source"] = "calibrated"
        extras["threshold_calibration"] = info
        self.decl = dataclasses.replace(self.decl, operating_threshold=t, extras=extras)
        self.model.declarations = self.decl
        self.calibration_info = info
        # Raw scores stay in memory for M1's threshold bootstrap (RunContext.extras); never in report.json.
        self.calibration_scores = scores
        per = cal.get("period") or None
        when = f" dated {per[0]}..{per[1]}" if per else ""
        if cal.get("policy") == "earliest":
            how = (f"the earliest {idx.size:,} benign rows{when} of the {corpus.name} evaluation split (a held "
                   "threshold: every test scores later rows only")
            after = cal.get("scored_benign_after_period")
            if after is not None and after < 0.999:
                how += f"; {after:.0%} of the scored benign rows post-date the calibration period"
            how += ")"
        else:
            how = (f"{idx.size:,} benign rows{when} hash-sampled across the {corpus.name} evaluation split (uniform "
                   "policy: the threshold has seen the benign data of every period it is tested on, so FPR "
                   "is close to the target by construction)")
        self.warn(
            f"operating threshold auto-calibrated to {t:.6g} (target FPR {cal['target_fpr']:.2%}, achieved "
            f"{achieved:.3%} on {how}; those rows are excluded from every test in this run). It is not a "
            "threshold you declared: pass the cut-off you ship for a verdict on your real operating point"
        )

    # ---- planning ------------------------------------------------------------------------------

    @staticmethod
    def _m0_identity() -> Any:
        """``id/code/title`` of the file-safety module (M0), even when it failed to import."""
        cls = registry.modules().get(FILE_SAFETY_ID)
        if cls is not None:
            return cls
        return type("_M0", (), {"id": FILE_SAFETY_ID, "code": "M0", "title": "Model file safety"})

    def _plan_modules(self) -> tuple[list[type[Module]], list[str]]:
        """Modules reported after M0 (scorecard order) and the ids the config disables.

        A module that the config enables but ``--only`` / ``--skip`` deselects stays in the plan and
        is reported as ``skipped`` (reason in :attr:`deselected`): it still counts toward the enabled
        weight, so coverage reflects the partial run, and a deselected hard gate is an unevaluated
        hard gate (BLOCKED under ``runtime.skipped_hard_gate_fails``). ``disabled`` lists only
        modules with ``enabled: false`` in the config.
        """
        cfg, opts = self.cfg, self.opts
        only = resolve_module_ids(opts.only, flag="--only")
        skip = resolve_module_ids(list(opts.skip or ()), flag="--skip") or []
        if FILE_SAFETY_ID in skip:
            self.warn("file_safety (M0) cannot be skipped: the artifact is always scanned before it is loaded")
        mods = registry.modules()
        plan: list[type[Module]] = []
        disabled: list[str] = []
        self.deselected: dict[str, str] = {}
        for mid, cls in mods.items():
            if mid == FILE_SAFETY_ID:
                continue
            mc = cfg.modules.get(mid)
            enabled = bool(mc is not None and mc.enabled)
            selected = (mid in only) if only is not None else enabled
            if mid in skip:
                selected = False
            if selected:
                if only is not None and not enabled:
                    log.info("%s %s is disabled in the config but was selected with --only", cls.code, mid)
                plan.append(cls)
            elif enabled:
                how = "excluded with --skip" if mid in skip else "not listed in --only"
                self.deselected[mid] = f"not selected for this run: {how} (enabled in the gate config)"
                plan.append(cls)
            elif mc is not None or mid in (only or ()):
                disabled.append(mid)
        if self.deselected:
            by_code = {c.id: c.code for c in plan}
            names = ", ".join(f"{by_code.get(m, m)} {m}" for m in self.deselected)
            hard = [m for m in self.deselected if _gate_for(cfg, mods[m]) is GateMode.HARD]
            self.warn(
                f"partial run: {len(self.deselected)} module(s) enabled in the gate config were deselected "
                f"with --only/--skip ({names}); they are reported as skipped and lower coverage"
                + (f", and the deselected hard gate(s) {', '.join(hard)} block the verdict" if hard
                   and cfg.runtime.skipped_hard_gate_fails else "")
            )
        return plan, disabled

    def _resolve_schema(self, feature_version: str) -> "FeatureSchema":
        try:
            return registry.get_schema(feature_version)
        except KeyError as e:
            raise AdapterError(
                f"the adapter declares feature_version {feature_version!r}, which no installed feature "
                f"schema provides ({e.args[0] if e.args else e})"
            ) from e

    def _load_corpus(self, feature_version: str) -> _CorpusState:
        cfg = self.cfg
        name = cfg.corpus
        st = _CorpusState(name=name, info={"name": name})
        try:
            provider = registry.get_corpus_provider(name)
        except KeyError as e:
            why = registry.unavailable("corpora").get(name)
            if why is None:
                raise ConfigError(
                    f"unknown corpus {name!r} in the config; known corpora: {', '.join(sorted(registry.corpora()))}"
                ) from e
            st.error = f"corpus provider {name!r} failed to import: {why}"
            self.warn(f"canonical corpus unavailable: {st.error}")
            return st
        try:
            location = provider.locate(cfg)
        except Exception:  # pragma: no cover - defensive
            location = None
        st.info.update(provider=provider.info(), location=str(location) if location is not None else None)
        try:
            if not provider.is_available(cfg):
                hint = provider.unavailable_hint(location) if location is not None else f"corpus {name!r} not found"
                raise CorpusUnavailable(hint)
            log.info("loading canonical corpus %s (verifying content hashes%s)", name,
                     ", full re-hash" if cfg.runtime.corpus_verification == "full" else "")
            corpus = provider.load(cfg, verify=True)
        except CorpusUnavailable as e:
            st.error = str(e)
        except Exception as e:
            st.error = f"failed to load corpus {name!r}: {type(e).__name__}: {e}"
        else:
            st.corpus = corpus
            if corpus.feature_version != feature_version:
                alts = sorted(
                    n for n, c in registry.corpora().items() if getattr(c, "feature_version", None) == feature_version
                )
                st.error = (
                    f"corpus {name!r} is in feature space {corpus.feature_version!r} but the model declares "
                    f"feature_version {feature_version!r}"
                    + (f"; use a {feature_version} corpus such as {', '.join(alts)}" if alts else "")
                )
            elif corpus.dim != self.schema.dim:
                st.error = (
                    f"corpus {name!r} has {corpus.dim} features but schema {self.schema.name!r} "
                    f"defines {self.schema.dim}"
                )
        if st.error:
            self.warn(f"canonical corpus not usable — modules that need the feature space will be skipped: {st.error}")
        return st

    # ---- module execution ----------------------------------------------------------------------

    def _extras(self) -> dict[str, Any]:
        return {
            "adapter_path": str(self.adapter),
            "artifact_paths": list(self.decl.model_paths),
            "allow_pickle": self.allow_pickle,
            "corpus_error": self.cs.error,
            "tree_access": dict(self.tree_info),
            "sandbox_enabled": bool(self.cfg.runtime.sandbox),
            # Run-specific reasons for missing capabilities ({requirement value: text}); modules
            # read them through RunContext.missing_reason() so in-module skips cite the real cause.
            "unmet_reasons": {r.value: t for r, t in self._unmet_detail().items()},
            "training_manifest_declared": bool(getattr(getattr(self, "manifest", None), "path", None)),
            # Auto-calibrated threshold only: the held-out benign scores and the FPR target it was fit
            # to, so M1 can bootstrap the threshold's own uncertainty (in memory only, not reported).
            "threshold_calibration": self._calibration_extra(),
        }

    def _calibration_extra(self) -> dict[str, Any] | None:
        scores = getattr(self, "calibration_scores", None)
        info = getattr(self, "calibration_info", None) or {}
        if scores is None or info.get("target_fpr") is None:
            return None
        return {"scores": scores, "target_fpr": float(info["target_fpr"]), "threshold": info.get("threshold"),
                "policy": info.get("policy"), "period": info.get("period"),
                "scored_benign_after_period": info.get("scored_benign_after_period")}

    def _context(self, cls: type[Module], *, model: Any, gate: GateMode, params: dict[str, Any]) -> RunContext:
        s = module_seed(self.seed, cls.id)
        return RunContext(
            config=self.cfg,
            module_id=cls.id,
            params=params,
            gate=gate,
            model=model,
            schema=self.schema,
            corpus=self.cs.usable,
            training_hashes=self.hashes,
            training_cutoff=self.cutoff,
            sample_dir=self.sample_dir,
            run_dir=self.out_dir,
            private_dir=self.private_dir,
            artifacts=self.artifacts,
            seed=s,
            rng=np.random.default_rng(s),
            capabilities=self.caps,
            deadline=time.monotonic() + self.max_s,
            log=logging.getLogger(f"malvalid.modules.{cls.id}"),
            extras=self._extras(),
        )

    def _append_skip(self, cls: type[Module], reason: str) -> ModuleResult:
        gate = _gate_for(self.cfg, cls)
        res = skipped_result(cls, gate, reason, module_params(self.cfg, cls))
        res.seed = module_seed(self.seed, cls.id)
        self.results.append(res)
        self.progress.module_finished(res)
        log.info("%s %s: skipped — %s", cls.code, cls.title, reason)
        return res

    def _error_result(
        self, cls: type[Module], ctx: RunContext, finding: str, error: str
    ) -> ModuleResult:
        return ModuleResult(
            module_id=cls.id,
            code=cls.code,
            title=cls.title,
            status=Status.ERROR,
            gate=ctx.gate,
            gate_outcome=GateOutcome.NOT_EVALUATED,
            finding=finding,
            params=dict(ctx.params),
            artifacts=self.artifacts.keys_for(cls.id),
            screening=bool(getattr(cls, "screening", False)),
            error=error,
            seed=ctx.seed,
        )

    def _normalize(self, cls: type[Module], ctx: RunContext, res: Any) -> ModuleResult:
        """Guard the report against malformed module results."""
        if not isinstance(res, ModuleResult):
            raise TypeError(f"{cls.__name__}.run() returned {type(res).__name__}, expected ModuleResult")
        try:
            res.status = Status(res.status)
            res.gate = GateMode(res.gate)
            res.gate_outcome = GateOutcome(res.gate_outcome)
        except ValueError as e:
            raise TypeError(f"{cls.__name__}.run() returned an invalid status/gate: {e}") from e
        if (res.module_id, res.code, res.title) != (cls.id, cls.code, cls.title):
            res.notes.append(f"result identity corrected from {res.module_id!r}/{res.code!r}")
            res.module_id, res.code, res.title = cls.id, cls.code, cls.title
        if res.gate is not ctx.gate:
            res.gate = ctx.gate
            if res.gate_outcome is GateOutcome.FAILED and res.status in (Status.FAIL, Status.WARN):
                res.status = Status.FAIL if ctx.gate is GateMode.HARD else Status.WARN
        if res.score is not None:
            try:
                sc = float(res.score)
            except (TypeError, ValueError):
                sc = float("nan")
            if not math.isfinite(sc):
                res.notes.append("module returned a non-finite axis score; treated as unscored")
                res.score = None
            else:
                res.score = min(max(sc, 0.0), 1.0)
        if res.status is Status.SKIPPED and not res.skip_reason:
            res.skip_reason = _first_line(res.finding) or "skipped by the module"
        if res.seed is None:
            res.seed = ctx.seed
        return res

    def _execute(self, cls: type[Module], ctx: RunContext) -> ModuleResult:
        """Run one module with its deadline + SIGALRM timer; never raises (except interrupts)."""
        model = ctx.model
        fired: list[bool] = []
        grace = min(5.0, 0.05 * self.max_s)
        msg = f"module {cls.id} exceeded runtime.max_seconds_per_module ({self.max_s:g} s)"
        set_deadline = getattr(model, "set_deadline", None)
        t = time.monotonic()
        log.info("%s %s: running", cls.code, cls.title)
        self.progress.module_started(cls)
        try:
            if callable(set_deadline):
                set_deadline(ctx.deadline)
            with _alarm(self.max_s + grace, lambda: fired.append(True), msg):
                raw = cls().run(ctx)
            if fired:  # the module swallowed the timeout exception
                raise ModuleTimeout(msg)
            res = self._normalize(cls, ctx, raw)
        except ModuleTimeout:
            res = self._error_result(
                cls, ctx, f"Not completed — exceeded {self.max_s:g} s (runtime.max_seconds_per_module).",
                traceback.format_exc(),
            )
            if fired:
                self._restart_model("after a module timeout")
        except Exception as e:
            res = self._error_result(
                cls, ctx, f"module crashed: {type(e).__name__}: {_first_line(str(e))}", traceback.format_exc()
            )
            if isinstance(e, SandboxError):
                self._restart_model(f"after {cls.code} crashed the sandbox worker")
        finally:
            if callable(set_deadline):
                try:
                    set_deadline(None)
                except Exception as e:  # pragma: no cover - defensive
                    log.debug("set_deadline(None) failed: %s", e)
        res.duration_s = round(time.monotonic() - t, 3)
        log.info("%s %s: %s (%.1f s)", cls.code, cls.title, res.status.value, res.duration_s)
        self.progress.module_finished(res)
        return res

    def _unmet_detail(self) -> dict[Requirement, str]:
        """Run-specific explanations for unmet requirements (cited in skip reasons)."""
        why: dict[Requirement, str] = {}
        if self.cs.error:
            why[Requirement.FEATURE_SPACE] = f"{REQUIREMENT_HINTS[Requirement.FEATURE_SPACE]} ({self.cs.error})"
        if Requirement.TREE_ACCESS not in self.caps and (
            self.tree_info.get("n_trees") or self.tree_info.get("export_timed_out")
        ):
            # The model exposes trees, but they were rejected (e.g. they failed the fidelity check)
            # or the export did not finish in time.
            why[Requirement.TREE_ACCESS] = str(self.tree_info.get("note") or "tree access disabled")
        manifest = getattr(self, "manifest", None)
        if (Requirement.TRAINING_HASHES not in self.caps and manifest is not None and manifest.path
                and not manifest.hashes):
            why[Requirement.TRAINING_HASHES] = (
                f"the declared training manifest {manifest.path} contains no sha256 hashes "
                "(training_hashes_path is set, but the file lists none)"
            )
        return why

    def _merge_scanned_artifacts(self, m0: ModuleResult) -> None:
        """Add model files M0 found on its own (undeclared, in the adapter dir) to ``model.artifacts``.

        Without ``model_path`` the declared list is empty; M0's scan is then the only record of which
        binary was evaluated. Only identity fields are copied — findings stay in M0's own section.
        """
        scanned = m0.details.get("artifacts") if isinstance(m0.details, dict) else None
        if not isinstance(scanned, list):
            return
        known = {str(Path(a["path"]).resolve()) for a in self.model_artifacts if a.get("path")}
        for a in scanned:
            if not isinstance(a, dict) or not a.get("path") or a.get("declared", True):
                continue
            key = str(Path(str(a["path"])).resolve())
            if key in known:
                continue
            known.add(key)
            self.model_artifacts.append({
                "path": str(a["path"]), "sha256": a.get("sha256"), "size": a.get("size"),
                "format": a.get("format"), "is_pickle": a.get("is_pickle"), "declared": False,
            })

    def _restart_model(self, why: str) -> None:
        restart = getattr(self.model, "restart", None)
        if self.model is None or not callable(restart):
            return
        # The module deadline that just expired is still set on the handle: a restart under it would
        # leave the worker handshake no time and fail although the worker is healthy (the overrun was
        # in harness code), and every later module would be skipped. Restarts are bounded by the
        # sandbox's own start-up limit instead.
        set_deadline = getattr(self.model, "set_deadline", None)
        if callable(set_deadline):
            try:
                set_deadline(None)
            except Exception as e:  # pragma: no cover - defensive
                log.debug("set_deadline(None) before restart failed: %s", e)
        try:
            log.info("restarting the sandboxed model %s", why)
            restart()
        except Exception as e:
            self.model_dead = f"{type(e).__name__}: {_first_line(str(e))}"
            self.warn(f"the sandboxed model could not be restarted {why}: {self.model_dead}")

    def _run_file_safety(self) -> ModuleResult:
        cfg = self.cfg
        try:
            cls = _file_safety_cls()
        except Exception as e:
            why = registry.unavailable("modules").get(FILE_SAFETY_ID) or f"{type(e).__name__}: {e}"
            self.warn(f"file safety (M0) is unavailable: {why}")
            return ModuleResult(
                module_id=FILE_SAFETY_ID,
                code="M0",
                title="Model file safety",
                status=Status.ERROR,
                gate=GateMode.HARD,
                gate_outcome=GateOutcome.NOT_EVALUATED,
                finding=f"file-safety scanner unavailable ({_first_line(why)}); the artifact could not be scanned",
                error=str(why),
                seed=module_seed(self.seed, FILE_SAFETY_ID),
                duration_s=0.0,
            )
        mc = cfg.modules.get(FILE_SAFETY_ID)
        if mc is not None and not mc.enabled:
            self.warn(
                "file_safety (M0) is disabled in the config, but malvalid always scans the model artifact "
                "before loading it; running it anyway"
            )
        gate = GateMode.HARD
        if _gate_for(cfg, cls) is not GateMode.HARD:
            self.warn(
                "file_safety (M0) gate is set to 'warn' in the config, but an unsafe model artifact always "
                "blocks promotion; M0 runs as a hard gate"
            )
        placeholder = _UnloadedModel(self.decl)
        self.caps = compute_capabilities(
            model=None, schema=self.schema, corpus=self.cs.usable, training_hashes=self.hashes,
            training_cutoff=self.cutoff, sample_dir=self.sample_dir, tree_access=False,
        )
        ctx = self._context(cls, model=placeholder, gate=gate, params=module_params(cfg, cls))
        return self._execute(cls, ctx)

    def _load_and_run(self, plan: list[type[Module]]) -> None:
        log.info("loading the model in the sandbox (calls %s.load())", self.decl.class_name or "the adapter class")
        self.progress.stage(
            "loading",
            f"loading the model (calls {self.decl.class_name or 'the adapter class'}.load()"
            + (" in the sandbox)" if self.cfg.runtime.sandbox else " in-process: sandbox disabled)"),
        )
        try:
            self.model = _open_model(self.adapter, self.policy, declarations=self.decl, class_name=self.opts.class_name)
        except Exception as e:
            detail = f"{type(e).__name__}: {_first_line(str(e))}"
            self.load_error = detail
            self.abort_reason = f"model load() failed in the sandbox ({detail})"
            log.error("model load failed: %s", detail)
            log.debug("load traceback:\n%s", traceback.format_exc())
            for cls in plan:
                self._append_skip(cls, self.deselected.get(cls.id) or f"model failed to load: {detail}")
            return
        if getattr(self, "calibration", None):
            try:
                self._calibrate()
            except Exception as e:
                detail = f"{type(e).__name__}: {_first_line(str(e))}"
                self.load_error = f"threshold calibration failed: {detail}"
                self.abort_reason = f"threshold calibration failed ({detail})"
                log.error("threshold calibration failed: %s", detail)
                log.debug("calibration traceback:\n%s", traceback.format_exc())
                for cls in plan:
                    self._append_skip(cls, self.deselected.get(cls.id) or f"threshold calibration failed: {detail}")
                self._collect_model_info()
                return
        try:
            trees = self._tree_access()
            self.caps = compute_capabilities(
                model=self.model, schema=self.schema, corpus=self.cs.usable, training_hashes=self.hashes,
                training_cutoff=self.cutoff, sample_dir=self.sample_dir, tree_access=trees is not None,
            )
            log.info("capabilities: %s", ", ".join(sorted(c.value for c in self.caps)) or "none")
            self.progress.stage("modules", f"running {len(plan) - len(self.deselected)} test module(s)")
            for cls in plan:
                if cls.id in self.deselected:
                    self._append_skip(cls, self.deselected[cls.id])
                    continue
                if self.model_dead:
                    self._append_skip(cls, f"the sandboxed model worker is unavailable after an earlier failure ({self.model_dead})")
                    continue
                missing = unmet_requirements(cls, self.caps)
                if missing:
                    self._append_skip(cls, _skip_reason(cls, missing, self._unmet_detail()))
                    continue
                gate = _gate_for(self.cfg, cls)
                ctx = self._context(cls, model=self.model, gate=gate, params=module_params(self.cfg, cls))
                self.results.append(self._execute(cls, ctx))
        finally:
            self._collect_model_info()

    def _collect_model_info(self) -> None:
        m = self.model
        self.query_count = None
        self.sandbox_info: dict[str, Any] | None = None
        self.featurize_source = None
        if m is None:
            return
        try:
            self.query_count = int(m.query_count)
        except Exception:
            pass
        fn = getattr(m, "sandbox_info", None)
        if callable(fn):
            try:
                self.sandbox_info = dict(fn())
            except Exception as e:
                self.sandbox_info = {"error": f"sandbox_info() failed: {e}"}
        fs = getattr(m, "featurize_source", None)
        if callable(fs):
            try:
                self.featurize_source = fs()
            except Exception:
                pass
        try:
            m.close()
        except Exception as e:
            log.debug("model close failed: %s", e)

    # ---- tree access ---------------------------------------------------------------------------

    def _fidelity_rows(self, trees: "TreeEnsemble") -> tuple[np.ndarray, str]:
        rng = np.random.default_rng(module_seed(self.seed, "tree_fidelity"))
        corpus = self.cs.usable
        if corpus is not None and corpus.n:
            from malvalid.corpora.base import ROLE_EVAL

            idx = corpus.indices(role=ROLE_EVAL)
            if idx.size == 0:
                idx = corpus.indices()
            if idx.size:
                idx = corpus.subsample(idx, TREE_FIDELITY_MAX_ROWS, rng)
                return corpus.take(idx), "corpus"
        # No corpus: random vectors straddling the model's own split thresholds, so that every
        # branch has a chance to be exercised (uniform noise would hit only a sliver of the tree).
        n, d = TREE_FIDELITY_RANDOM_ROWS, self.schema.dim
        X = rng.random((n, d)).astype(np.float64)
        thr: dict[int, list[float]] = {}
        for t in trees.trees:
            internal = t.children_left >= 0
            for f, v in zip(t.feature[internal].tolist(), t.threshold[internal].tolist()):
                if 0 <= f < d and math.isfinite(v):
                    thr.setdefault(int(f), []).append(float(v))
        for f, vals in thr.items():
            arr = np.asarray(vals)
            pick = arr[rng.integers(0, arr.size, size=n)]
            step = np.maximum(np.abs(pick) * 1e-3, 1e-3)
            X[:, f] = pick + np.where(rng.random(n) < 0.5, -step, step)
        return X.astype(np.float32), "random"

    def _tree_access(self) -> "TreeEnsemble | None":
        info: dict[str, Any] = {"available": False, "n_trees": None, "fidelity_max_abs_diff": None,
                                "n_rows_checked": 0, "rows_source": None, "tolerance": TREE_FIDELITY_TOL,
                                "note": None}
        self.tree_info = info
        model = self.model
        # The export runs the untrusted adapter's tree_ensemble() (or a loader conversion) in the
        # worker: bound it by the module budget like every other call into the model.
        set_deadline = getattr(model, "set_deadline", None)
        fired: list[bool] = []
        timeout_note = (
            f"the tree export (tree_ensemble()) did not finish within {self.max_s:g} s "
            "(runtime.max_seconds_per_module); tree access disabled, so tree-based analyses are unavailable"
        )
        try:
            if callable(set_deadline):
                set_deadline(time.monotonic() + self.max_s)
            with _alarm(self.max_s + min(5.0, 0.05 * self.max_s), lambda: fired.append(True), timeout_note):
                trees = model.tree_ensemble()
            if fired:  # the handle swallowed the timer's exception (e.g. the in-process handle)
                raise ModuleTimeout(timeout_note)
        except ModuleTimeout:
            info.update(note=timeout_note, export_timed_out=True)
            self.warn(f"tree access disabled: {timeout_note}")
            return None
        except Exception as e:
            info["note"] = f"tree structure could not be read ({type(e).__name__}: {_first_line(str(e))}); tree-based analyses are unavailable"
            return None
        finally:
            if callable(set_deadline):
                try:
                    set_deadline(None)
                except Exception:  # pragma: no cover
                    pass
        if trees is None:
            info["note"] = "the model does not expose its tree structure; tree-based analyses are unavailable"
            return None
        info["n_trees"] = int(trees.n_trees)
        info["model_kind"] = getattr(trees, "model_kind", None)
        if trees.n_features > self.schema.dim:
            info["note"] = (
                f"tree dump uses {trees.n_features} features but schema {self.schema.name!r} has "
                f"{self.schema.dim}; tree access disabled"
            )
            return None
        set_deadline = getattr(model, "set_deadline", None)
        try:
            X, source = self._fidelity_rows(trees)
            if callable(set_deadline):
                set_deadline(time.monotonic() + self.max_s)
            p_model = np.asarray(model.predict_proba(X), dtype=np.float64)
            p_tree = np.asarray(trees.predict_proba(X), dtype=np.float64)
            diff = float(np.max(np.abs(p_tree - p_model))) if X.shape[0] else 0.0
        except Exception as e:
            info["note"] = f"tree fidelity check failed ({type(e).__name__}: {_first_line(str(e))}); tree access disabled"
            return None
        finally:
            if callable(set_deadline):
                try:
                    set_deadline(None)
                except Exception:  # pragma: no cover
                    pass
        info.update(n_rows_checked=int(X.shape[0]), rows_source=source,
                    fidelity_max_abs_diff=diff if math.isfinite(diff) else None)
        if not math.isfinite(diff) or diff > TREE_FIDELITY_TOL:
            info["note"] = (
                f"the tree dump does not reproduce predict_proba (max |Δ| = {diff:.3g} > {TREE_FIDELITY_TOL:g} "
                f"on {X.shape[0]} {source} rows); tree access disabled so tree-based analyses cannot "
                "misdescribe the deployed model"
            )
            self.warn(f"tree access disabled: {info['note']}")
            return None
        info["available"] = True
        info["note"] = f"verified: tree dump matches predict_proba (max |Δ| = {diff:.2g} on {X.shape[0]} {source} rows)"
        return trees

    # ---- report --------------------------------------------------------------------------------

    def _disclaimers(self) -> list[str]:
        out = list(BASE_DISCLAIMERS)
        c = self.cs.corpus
        if c is not None and getattr(c, "synthetic", False):
            out.append(
                "The canonical corpus used here is synthetic: results exercise the pipeline and are "
                "not evidence of real-world production readiness."
            )
        if not self.cfg.runtime.sandbox:
            out.append("The model ran without the sandbox (in-process); isolation guarantees do not apply to this run.")
        elif self._sandbox_block().get("isolation") == "process_only":
            out.append(REDUCED_ISOLATION_NOTE)
        return out

    def _corpus_block(self) -> dict[str, Any]:
        cs = self.cs
        block: dict[str, Any] = {"name": cs.name}
        if cs.corpus is not None:
            try:
                block.update(cs.corpus.summary())
            except Exception as e:  # pragma: no cover - defensive
                block["summary_error"] = str(e)
            if cs.corpus.path is not None:
                block["path"] = str(cs.corpus.path)
        block.update(
            available=cs.corpus is not None,
            used_for_evaluation=cs.usable is not None,
            # How the corpus files were checked on load: {mode: full | cached | partial, files,
            # verified_at}; "cached" trusts the size/mtime/ctime/inode stamp of the last full hash.
            verification=getattr(cs.corpus, "verification", None),
            error=cs.error,
            excluded_training_members=self.excluded,
            provider=cs.info.get("provider"),
            location=cs.info.get("location"),
        )
        cal = getattr(self, "calibration", None)
        if cal:
            block["calibration_holdout"] = {
                **{k: v for k, v in cal.items() if k != "indices"},
                **(getattr(self, "calibration_info", None) or {}),
            }
        return block

    def _schema_block(self) -> dict[str, Any]:
        try:
            extractor = bool(self.schema.featurize_available())
        except Exception:
            extractor = False
        return {
            "name": self.schema.name,
            "dim": int(self.schema.dim),
            "description": getattr(self.schema, "description", ""),
            "featurize_available": Requirement.FEATURIZE in self.caps,
            "featurize_source": getattr(self, "featurize_source", None),
            "schema_extractor_available": extractor,
        }

    def _blocked_cause(self) -> str | None:
        """Why the run is BLOCKED, in the order a researcher should fix things (None if not blocked)."""
        rs = self.results
        if any(r.gate is GateMode.HARD and r.gate_outcome is GateOutcome.FAILED for r in rs):
            return None  # verdict.py's own label ("a hard gate failed") is accurate
        if self.load_error:
            return "the model failed to load in the sandbox"
        m0 = next((r for r in rs if r.module_id == FILE_SAFETY_ID), None)
        if m0 is not None and m0.status is Status.ERROR:
            return "the model-file safety scan did not complete"
        if any(r.status is Status.ERROR for r in rs):
            return "a module errored"
        if self.abort_reason:
            return "the run was aborted before the model was evaluated"
        deselected = getattr(self, "deselected", {})
        if any(r.gate is GateMode.HARD and r.module_id in deselected for r in rs):
            return "a hard gate was deselected with --only/--skip"
        return "a hard gate could not be evaluated"

    def _verdict_block(self, vr: Any) -> dict[str, Any]:
        """``VerdictResult.to_dict()`` with the headline text made accurate.

        Works around two wording defects in the frozen ``verdict.py`` (reported as contract change
        requests): the BLOCKED label/summary always say "a hard gate failed" even when the cause is an
        errored module, an M0 abort or an unevaluated hard gate; and the summary rounds the score to an
        integer (79.6 -> "80/100") although the bands compare the unrounded value. The verdict, score,
        axes and reasons are untouched.
        """
        d = vr.to_dict()
        label = str(d.get("label") or "")
        if vr.verdict is Verdict.BLOCKED:
            cause = self._blocked_cause()
            if cause is not None:
                label = f"Blocked — {cause}"
        d["label"] = label
        score, cov = d.get("score"), d.get("coverage")
        if score is None:
            d["summary"] = f"{label}. No axis could be scored."
        else:
            d["summary"] = (f"{label}. Production-readiness score {float(score):.1f}/100 "
                            f"({float(cov or 0.0):.0%} of the weighted battery evaluated).")
            if d.get("capped"):
                cap = (d.get("bands") or {}).get("blocked_score_cap")
                d["summary"] += f" Score capped at {float(cap):g} because the run is blocked."
        return d

    def _gate_block(self, exit_code: int, disabled: list[str]) -> dict[str, Any]:
        hard = [r for r in self.results if r.gate is GateMode.HARD]
        return {
            "exit_code": exit_code,
            "exit_meaning": EXIT_MEANINGS.get(exit_code, ""),
            "passed": exit_code == 0,
            "fail_on": self.opts.fail_on,
            "hard_gates": [
                {"module_id": r.module_id, "code": r.code, "title": r.title, "status": r.status.value,
                 "gate_outcome": r.gate_outcome.value}
                for r in hard
            ],
            "failed_hard": [r.module_id for r in hard if r.gate_outcome is GateOutcome.FAILED],
            "not_evaluated_hard": [r.module_id for r in hard if r.gate_outcome is GateOutcome.NOT_EVALUATED
                                   and r.status is not Status.ERROR],
            "errored": [r.module_id for r in self.results if r.status is Status.ERROR],
            "skipped": [r.module_id for r in self.results if r.status is Status.SKIPPED],
            "aborted": self.abort_reason is not None,
            "abort_reason": self.abort_reason,
            "load_error": self.load_error,
            "disabled": disabled,
            # Enabled in the config but not selected with --only/--skip: reported as skipped modules.
            "deselected": list(getattr(self, "deselected", {})),
        }

    def _config_block(self) -> dict[str, Any]:
        """``GateConfig.to_dict()`` with each module's *effective* gate mode.

        ``ModuleConfig.gate`` defaults to ``warn`` when the YAML omits it, but the runner then applies
        the module's own ``default_gate`` (see :func:`_gate_for`); record the gate actually applied.
        """
        d = self.cfg.to_dict()
        mods = registry.modules()
        for mid, mc in self.cfg.modules.items():
            cls = mods.get(mid)
            if cls is not None and "gate" not in mc.model_fields_set and mid in d.get("modules", {}):
                d["modules"][mid]["gate"] = cls.default_gate.value
        return d

    def _sandbox_block(self) -> dict[str, Any]:
        info = dict(getattr(self, "sandbox_info", None) or {})
        info.setdefault("enabled", bool(self.cfg.runtime.sandbox))
        if not info.get("backend"):
            info.setdefault("note", "the model handle did not report sandbox details")
        # os_sandbox | process_only | none, from the model handle; None = not determined (never loaded).
        info.setdefault("isolation", None)
        info.setdefault("allow_reduced_isolation", bool(getattr(self.cfg.runtime, "allow_reduced_isolation", False)))
        return info

    def _finish(self, disabled: list[str]) -> RunOutcome:
        from malvalid.environment import capture_environment
        from malvalid.report.json_writer import finalize_report, write_report_json

        if not hasattr(self, "sandbox_info"):
            self._collect_model_info()
        self.progress.stage("verdict", "computing the production-readiness verdict")
        vr = compute_verdict(self.results, self.cfg, aborted=self.abort_reason)
        exit_code = exit_code_for(vr, self.results, self.opts.fail_on)
        if self.load_error:
            exit_code = max(exit_code, 2)
        sb = self._sandbox_block()
        verdict_block = self._verdict_block(vr)
        _add_isolation(verdict_block, sb.get("isolation"))
        self.progress.set_verdict(verdict_block)
        if sb.get("isolation") == "process_only":
            self.warn(REDUCED_ISOLATION_NOTE)
        if sb.get("network_isolated") is False:
            self.warn("the sandbox could not isolate the network for the model worker (see `sandbox`)")
        finished = _utcnow()
        report_json = self.out_dir / "report.json"
        want_html = bool(self.opts.write_html and self.cfg.report.html)
        report_html = self.out_dir / "report.html" if want_html else None
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "tool": {"name": "malvalid", "version": __version__},
            "title": self.cfg.report.title,
            "run": {
                "id": self.run_id,
                "started_at": _iso(self.started),
                "finished_at": _iso(finished),
                "duration_s": round(time.monotonic() - self.t0, 3),
                "command": self.opts.command,
                "seed": self.seed,
                "out_dir": str(self.out_dir),
                "report_json": str(report_json),
                "report_html": str(report_html) if report_html else None,
                "run_log": str(self.out_dir / "run.log"),
                "only": list(self.opts.only) if self.opts.only is not None else None,
                "skip": list(self.opts.skip or ()),
                "allow_pickle": self.allow_pickle,
            },
            "verdict": verdict_block,
            "gate": self._gate_block(exit_code, disabled),
            "model": {
                **self.decl.to_dict(),
                "artifacts": self.model_artifacts,
                "adapter_sha256": _sha256_file(self.adapter) if self.adapter.is_file() else None,
                "tree_access": self.tree_info,
                "load_error": self.load_error,
                "query_count": getattr(self, "query_count", None),
            },
            "corpus": self._corpus_block(),
            "schema": self._schema_block(),
            "training_manifest": self.manifest.to_report(self.cs.usable),
            "capabilities": sorted(c.value for c in self.caps),
            "config": self._config_block(),
            "environment": capture_environment(),
            "sandbox": sb,
            "modules": [r.to_dict() for r in self.results],
            "artifacts": self.artifacts.to_dict(),
            "warnings": self.warnings,
            "disclaimers": self._disclaimers(),
        }
        self.progress.stage("writing", "writing report.json" + (" and report.html" if report_html else ""))
        report = finalize_report(report, private_dir=self.private_dir)
        if report_html is not None:
            try:
                report_html = _write_html(report, report_html)
            except Exception as e:
                msg = f"report.html could not be written ({type(e).__name__}: {_first_line(str(e))}); report.json is complete"
                log.warning(msg)
                log.debug("html traceback:\n%s", traceback.format_exc())
                report["warnings"].append(msg)
                report["run"]["report_html"] = None
                report_html = None
        write_report_json(report, report_json, private_dir=self.private_dir)
        log.info("verdict: %s — score %s/100, coverage %.0f%% — exit %d",
                 vr.verdict.value, "n/a" if vr.score is None else f"{vr.score:.1f}", 100 * vr.coverage, exit_code)
        self.progress.done(exit_code, str(verdict_block.get("summary") or verdict_block.get("label") or "done"))
        return RunOutcome(report=report, exit_code=exit_code, report_json=report_json, report_html=report_html)


__all__ = [
    "PROGRESS_SCHEMA",
    "PROGRESS_STAGES",
    "RunOptions",
    "RunOutcome",
    "run_gate",
    "resolve_module_ids",
    "validate_fail_on",
    "EXIT_MEANINGS",
    "FAIL_ON_CHOICES",
    "Verdict",
    "TrainingManifest",
]
