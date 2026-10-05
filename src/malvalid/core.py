"""Core types shared by every malvalid component.

This module is the contract layer: test modules, the runner, and the report writer all speak in
these types. Keep it dependency-light (numpy only) and stable.
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import math
import operator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.context import RunContext


# --------------------------------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------------------------------


class Status(str, enum.Enum):
    """Outcome of one module run, as shown on the scorecard."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIPPED = "skipped"
    ERROR = "error"


class GateMode(str, enum.Enum):
    """How a module's gate failure is treated. ``hard`` failures make the run exit non-zero."""

    HARD = "hard"
    WARN = "warn"


class GateOutcome(str, enum.Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


class Requirement(str, enum.Enum):
    """Inputs a module may need. The runner skips a module (never fake-passes it) when unmet."""

    FEATURE_SPACE = "feature_space"  # canonical corpus in the model's feature_version is loaded
    SAMPLE_DIR = "sample_dir"  # operator-supplied raw sample directory exists
    FEATURIZE = "featurize"  # raw bytes -> vector (adapter.featurize or schema.featurize)
    TRAINING_HASHES = "training_hashes"  # training manifest (sha256 list) declared and parsed
    TRAINING_CUTOFF = "training_cutoff"  # ISO date of newest training sample declared
    TREE_ACCESS = "tree_access"  # normalized TreeEnsemble available from the model
    QUERY_ONLY = "query_only"  # black-box query access (satisfied once the model is loaded)


# --------------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------------


class MalValidError(Exception):
    """Base class for all malvalid errors."""


class ConfigError(MalValidError):
    """Invalid gate configuration."""


class AdapterError(MalValidError):
    """The submitted adapter does not conform to the submission contract."""


class AdapterContractError(AdapterError):
    """The adapter returned something that violates the contract (shape, range, dtype...)."""


class SandboxError(MalValidError):
    """The sandboxed worker failed, crashed, or violated a policy."""


class PickleRefused(SandboxError):
    """A pickle load was attempted without --allow-pickle (or a disallowed global was requested)."""


class ModuleTimeout(MalValidError):
    """A module exceeded runtime.max_seconds_per_module."""


class RunAborted(MalValidError):
    """The whole run must stop (e.g. M0 found a CRITICAL issue before load)."""


def unmet_requirements(module_cls: type["Module"], caps: "frozenset[Requirement]") -> list[Requirement]:
    """Requirements of ``module_cls`` not satisfied by ``caps`` (empty list => runnable).

    For ``requires_any`` the whole group is returned when none of its members is satisfied.
    """
    missing = [r for r in module_cls.requires if r not in caps]
    if module_cls.requires_any and not any(r in caps for r in module_cls.requires_any):
        missing.extend(r for r in module_cls.requires_any if r not in missing)
    return missing


class FeaturizeUnavailable(MalValidError):
    """Raw-bytes featurization is not available (no adapter featurize and schema lacks deps)."""


class CorpusUnavailable(MalValidError):
    """The canonical corpus is not present locally or failed verification."""


class UnsupportedTreeError(MalValidError):
    """A model's trees cannot be normalized (e.g. categorical splits, unsupported objective)."""


# --------------------------------------------------------------------------------------------------
# Gate checks
# --------------------------------------------------------------------------------------------------

_OPS = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
}


@dataclass
class GateCheck:
    """One thresholded comparison that contributes to a module's gate outcome.

    ``passed is None`` means the check could not be evaluated (e.g. the metric is NaN); that never
    counts as a pass.
    """

    name: str
    metric: str
    op: str
    threshold: float
    value: float | None
    passed: bool | None
    description: str = ""
    # Axis-score anchors (see malvalid.scoring). score is 0..1, None if not evaluable/unscored.
    ideal: float | None = None
    floor: float | None = None
    scale: str = "linear"
    score: float | None = None

    @classmethod
    def evaluate(
        cls,
        name: str,
        value: float | int | None,
        op: str,
        threshold: float | int,
        *,
        metric: str | None = None,
        description: str = "",
        ideal: float | None = None,
        floor: float | None = None,
        scale: str = "linear",
    ) -> "GateCheck":
        """Evaluate ``value <op> threshold``. Pass ``ideal``/``floor`` to grade it 0..1 for the
        production-readiness score (strongly recommended for every gated metric)."""
        if op not in _OPS:
            raise ValueError(f"unknown comparison operator {op!r}")
        v: float | None
        if value is None:
            v = None
        else:
            v = float(value)
            if math.isnan(v):
                v = None
        passed = None if v is None else bool(_OPS[op](v, float(threshold)))
        score: float | None = None
        if ideal is not None and floor is not None and v is not None and op != "==":
            from malvalid.scoring import metric_score

            score = metric_score(v, op, float(threshold), ideal=float(ideal), floor=float(floor), scale=scale)
        elif v is not None and op == "==" and ideal is None and floor is None:
            score = 1.0 if passed else 0.0
        return cls(
            name=name,
            metric=metric or name,
            op=op,
            threshold=float(threshold),
            value=v,
            passed=passed,
            description=description,
            ideal=None if ideal is None else float(ideal),
            floor=None if floor is None else float(floor),
            scale=scale,
            score=score,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------------


@dataclass
class ModuleResult:
    """Everything one module reports. Must be JSON-serializable via ``to_dict``."""

    module_id: str
    code: str
    title: str
    status: Status
    gate: GateMode
    gate_outcome: GateOutcome
    finding: str
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: list[GateCheck] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)  # thresholds/params actually used
    details: dict[str, Any] = field(default_factory=dict)  # structured extras (tables, lists)
    artifacts: list[str] = field(default_factory=list)  # keys into the report's artifact store
    notes: list[str] = field(default_factory=list)  # caveats shown with the finding
    screening: bool = False  # True => report labels this as screening, not certification
    # Axis production-readiness score in [0, 1] (None when skipped/errored/unscorable). Feeds the
    # headline verdict (malvalid.verdict). Defaults to the weakest per-check score.
    score: float | None = None
    skip_reason: str | None = None
    error: str | None = None  # traceback text when status == error
    duration_s: float | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "code": self.code,
            "title": self.title,
            "status": self.status.value,
            "gate": self.gate.value,
            "gate_outcome": self.gate_outcome.value,
            "finding": self.finding,
            "metrics": to_jsonable(self.metrics),
            "checks": [to_jsonable(c.to_dict()) for c in self.checks],
            "params": to_jsonable(self.params),
            "details": to_jsonable(self.details),
            "artifacts": list(self.artifacts),
            "notes": list(self.notes),
            "screening": self.screening,
            "score": to_jsonable(self.score),
            "skip_reason": self.skip_reason,
            "error": self.error,
            "duration_s": self.duration_s,
            "seed": self.seed,
        }


def derive_status(
    checks: list[GateCheck], gate: GateMode, *, advisory_warn: bool = False
) -> tuple[Status, GateOutcome]:
    """The single rule that maps checks + gate mode to (status, gate_outcome).

    - any check failed          -> gate FAILED; status FAIL if gate is hard else WARN
    - all evaluable checks pass -> gate PASSED; status PASS (WARN if advisory_warn)
    - some checks unevaluable   -> gate NOT_EVALUATED; status WARN (never a silent pass)
    - no checks at all          -> gate NOT_EVALUATED; status PASS (WARN if advisory_warn)
    """
    if any(c.passed is False for c in checks):
        return (Status.FAIL if gate is GateMode.HARD else Status.WARN), GateOutcome.FAILED
    if any(c.passed is None for c in checks):
        return Status.WARN, GateOutcome.NOT_EVALUATED
    if checks:
        return (Status.WARN if advisory_warn else Status.PASS), GateOutcome.PASSED
    return (Status.WARN if advisory_warn else Status.PASS), GateOutcome.NOT_EVALUATED


# --------------------------------------------------------------------------------------------------
# Module interface
# --------------------------------------------------------------------------------------------------


class Module(abc.ABC):
    """A test module plugin.

    Subclasses set the class attributes and implement :meth:`run`. Build results with
    :meth:`result` / :meth:`skip` so status derivation is uniform across modules. Modules must be
    deterministic given ``ctx.rng`` / ``ctx.seed`` and must never report a pass for work not done.
    """

    id: ClassVar[str]  # config key, e.g. "performance"
    code: ClassVar[str]  # scorecard code, e.g. "M1"
    title: ClassVar[str]  # e.g. "Performance & calibration"
    description: ClassVar[str] = ""  # one paragraph: what it measures, shown in list-modules/report
    requires: ClassVar[tuple[Requirement, ...]] = ()  # ALL must be satisfied
    requires_any: ClassVar[tuple[Requirement, ...]] = ()  # if non-empty, at least ONE must be
    enhanced_by: ClassVar[tuple[Requirement, ...]] = ()  # optional inputs that unlock more
    screening: ClassVar[bool] = False
    default_gate: ClassVar[GateMode] = GateMode.WARN
    # Every tunable parameter with its default. Config keys not listed here are a ConfigError.
    default_params: ClassVar[dict[str, Any]] = {}

    @abc.abstractmethod
    def run(self, ctx: "RunContext") -> ModuleResult:
        """Run the test. Raise freely; the runner converts exceptions into status=error."""

    @classmethod
    def validate_params(cls, params: dict[str, Any]) -> None:
        """Check parameter *values* up front; raise :class:`ConfigError` for a bad one.

        ``params`` are :attr:`default_params` overlaid with the config's values.
        :func:`malvalid.config.validate_against_registry` calls this before anything is loaded, so
        a bad value fails the run with a one-line config error instead of erroring the module after
        the model has been loaded. Checks that need the run (e.g. feature names of the schema) stay
        in :meth:`run`. The default accepts everything.
        """

    # ---- helpers -------------------------------------------------------------------------------

    def result(
        self,
        ctx: "RunContext",
        *,
        finding: str,
        checks: list[GateCheck] | None = None,
        metrics: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        notes: list[str] | None = None,
        advisory_warn: bool = False,
        score: float | None = None,
        score_rule: str = "min",
    ) -> ModuleResult:
        """Build a result. ``score`` (0..1) overrides the default axis score, which is the
        ``score_rule`` ("min" | "mean") combination of the checks' scores."""
        checks = list(checks or [])
        status, outcome = derive_status(checks, ctx.gate, advisory_warn=advisory_warn)
        if score is None:
            from malvalid.scoring import combine_axis

            score = combine_axis([c.score for c in checks], score_rule)
        else:
            score = float(score)
            score = min(max(score, 0.0), 1.0) if math.isfinite(score) else None
        return ModuleResult(
            module_id=self.id,
            code=self.code,
            title=self.title,
            status=status,
            gate=ctx.gate,
            gate_outcome=outcome,
            finding=finding,
            metrics=dict(metrics or {}),
            checks=checks,
            params=dict(ctx.params),
            details=dict(details or {}),
            artifacts=list(artifacts if artifacts is not None else ctx.artifacts.keys_for(self.id)),
            notes=list(notes or []),
            screening=self.screening,
            score=score,
            seed=ctx.seed,
        )

    def skip(self, ctx: "RunContext", reason: str, *, notes: list[str] | None = None) -> ModuleResult:
        return ModuleResult(
            module_id=self.id,
            code=self.code,
            title=self.title,
            status=Status.SKIPPED,
            gate=ctx.gate,
            gate_outcome=GateOutcome.NOT_EVALUATED,
            finding=f"Not run — {reason}",
            params=dict(ctx.params),
            notes=list(notes or []),
            screening=self.screening,
            skip_reason=reason,
            seed=ctx.seed,
        )

    @classmethod
    def info(cls) -> dict[str, Any]:
        return {
            "id": cls.id,
            "code": cls.code,
            "title": cls.title,
            "description": cls.description,
            "requires": [r.value for r in cls.requires],
            "requires_any": [r.value for r in cls.requires_any],
            "enhanced_by": [r.value for r in cls.enhanced_by],
            "screening": cls.screening,
            "default_gate": cls.default_gate.value,
            "default_params": to_jsonable(cls.default_params),
        }


def skipped_result(
    module_cls: type[Module], gate: GateMode, reason: str, params: dict[str, Any] | None = None
) -> ModuleResult:
    """Build a skipped result without instantiating a RunContext (used by the runner)."""
    return ModuleResult(
        module_id=module_cls.id,
        code=module_cls.code,
        title=module_cls.title,
        status=Status.SKIPPED,
        gate=gate,
        gate_outcome=GateOutcome.NOT_EVALUATED,
        finding=f"Not run — {reason}",
        params=dict(params or {}),
        screening=module_cls.screening,
        skip_reason=reason,
    )


# --------------------------------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays, enums, dataclasses, paths, dates to JSON types.

    Non-finite floats become ``None`` so report.json is strict JSON.
    """
    import datetime as _dt
    from pathlib import PurePath

    import numpy as np

    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (int,)) and not isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(x) for x in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, (_dt.date, _dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, PurePath):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    return str(obj)
