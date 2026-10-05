"""Headline production-readiness verdict and score.

A researcher submits one detector; malvalid answers "does this model meet the gate policy for
promotion to production?" with a verdict backed by a 0–100 score (evidence, not a certificate):

=============  ==========================================================================
READY          score >= ready_min, enough of the battery evaluated, no gate failed
CONDITIONAL    score >= conditional_min, but a warn gate failed / coverage is thin / a
               required axis was not evaluated — promotion needs a human decision
NOT_READY      score < conditional_min (or nothing could be scored)
BLOCKED        a hard gate failed, a module errored, a hard-gate module could not be
               evaluated, or M0 aborted the run — never promotable, score capped
=============  ==========================================================================

The score is the weighted mean of per-axis scores (each 0..1, see :mod:`malvalid.scoring`) over
the modules that actually ran. Skipped modules never contribute a score; they lower ``coverage``
instead. The score can never override a failed hard gate: BLOCKED caps it at
``blocked_score_cap``. Weights are organizational policy and configurable in ``gate.yaml``.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from malvalid.core import GateMode, GateOutcome, ModuleResult, Status, to_jsonable

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.config import GateConfig


class Verdict(str, enum.Enum):
    READY = "ready"
    CONDITIONAL = "conditional"
    NOT_READY = "not_ready"
    BLOCKED = "blocked"


VERDICT_ORDER = [Verdict.READY, Verdict.CONDITIONAL, Verdict.NOT_READY, Verdict.BLOCKED]

VERDICT_LABELS = {
    Verdict.READY: "Ready (meets the gate policy)",
    Verdict.CONDITIONAL: "Conditionally ready — review before promotion",
    Verdict.NOT_READY: "Not ready",
    Verdict.BLOCKED: "Blocked — not promotable",
}

# Most specific first: the BLOCKED headline names the highest-priority cause present.
_CAUSE_PRIORITY = (
    "unsafe model artifact",
    "a hard gate failed",
    "a module errored",
    "a hard gate could not be evaluated",
    "run aborted before evaluation",
)

# Modules that feed the gate but are not a scored axis (preconditions).
UNSCORED_MODULES = frozenset({"file_safety", "dummy"})


@dataclass
class AxisScore:
    module_id: str
    code: str
    title: str
    weight: float
    score: float | None  # 0..100, None if not evaluated
    status: str
    gate: str
    gate_outcome: str
    screening: bool
    counted: bool  # contributed to the weighted mean

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self.__dict__)


@dataclass
class VerdictResult:
    verdict: Verdict
    label: str
    score: float | None  # 0..100 headline score (after any cap), None if nothing scored
    raw_score: float | None  # weighted mean before cap
    capped: bool
    coverage: float  # evaluated weight / enabled weight, 0..1
    reasons: list[str]
    axes: list[AxisScore]
    bands: dict[str, float]
    blockers: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "label": self.label,
            "score": self.score,
            "raw_score": self.raw_score,
            "capped": self.capped,
            "coverage": self.coverage,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "axes": [a.to_dict() for a in self.axes],
            "bands": dict(self.bands),
            "summary": self.summary,
        }


def _r1(x: float | None) -> float | None:
    return None if x is None else round(float(x), 1)


def compute_verdict(
    results: Iterable[ModuleResult],
    cfg: "GateConfig",
    *,
    aborted: str | None = None,
) -> VerdictResult:
    """Derive the headline verdict from module results and the ``verdict:`` config block."""
    vc = cfg.verdict
    results = list(results)
    weights = dict(vc.weights)
    blockers: list[str] = []
    reasons: list[str] = []

    causes: list[str] = []  # short cause per blocker, used for the BLOCKED label/summary
    if aborted:
        blockers.append(f"run aborted before the model was loaded: {aborted}")
        causes.append("run aborted before evaluation")
    for r in results:
        if r.status is Status.ERROR:
            blockers.append(f"{r.code} {r.title} errored")
            causes.append("a module errored")
        elif r.gate is GateMode.HARD and r.gate_outcome is GateOutcome.FAILED:
            blockers.append(f"{r.code} {r.title}: hard gate failed")
            causes.append("unsafe model artifact" if r.module_id == "file_safety" else "a hard gate failed")
        elif r.module_id == "file_safety" and r.gate_outcome is GateOutcome.FAILED:
            # An unsafe artifact blocks promotion even if an operator demoted M0 to a warn gate.
            blockers.append(f"{r.code} {r.title}: unsafe model artifact")
            causes.append("unsafe model artifact")
        elif (
            r.gate is GateMode.HARD
            and r.gate_outcome is GateOutcome.NOT_EVALUATED
            and cfg.runtime.skipped_hard_gate_fails
            and r.module_id not in ("dummy",)
            and not (r.module_id == "file_safety" and r.status is Status.PASS)
        ):
            why = r.skip_reason or "gate could not be evaluated"
            blockers.append(f"{r.code} {r.title}: hard gate not evaluated ({why})")
            causes.append("a hard gate could not be evaluated")

    axes: list[AxisScore] = []
    total_w = 0.0
    eval_w = 0.0
    acc = 0.0
    for r in results:
        if r.module_id in UNSCORED_MODULES:
            continue
        w = float(weights.get(r.module_id, vc.default_weight))
        s: float | None = None
        if r.score is not None and math.isfinite(float(r.score)):
            s = min(max(float(r.score), 0.0), 1.0)
        counted = w > 0 and s is not None and r.status not in (Status.SKIPPED, Status.ERROR)
        if w > 0:
            total_w += w
        if counted:
            eval_w += w
            acc += w * s  # type: ignore[operator]
        axes.append(
            AxisScore(
                module_id=r.module_id,
                code=r.code,
                title=r.title,
                weight=w,
                score=None if s is None or r.status in (Status.SKIPPED, Status.ERROR) else _r1(100 * s),
                status=r.status.value,
                gate=r.gate.value,
                gate_outcome=r.gate_outcome.value,
                screening=r.screening,
                counted=counted,
            )
        )
    # Round before comparing against the bands so the displayed score always agrees with the verdict.
    raw = _r1(100.0 * acc / eval_w) if eval_w > 0 else None
    coverage = eval_w / total_w if total_w > 0 else 0.0
    bands = {
        "ready_min": float(vc.ready_min),
        "conditional_min": float(vc.conditional_min),
        "min_coverage_ready": float(vc.min_coverage_ready),
        "blocked_score_cap": float(vc.blocked_score_cap),
    }

    capped = False
    score = raw
    if blockers:
        verdict = Verdict.BLOCKED
        if raw is not None and raw > vc.blocked_score_cap:
            score, capped = float(vc.blocked_score_cap), True
        reasons.extend(blockers)
    elif raw is None:
        verdict = Verdict.NOT_READY
        reasons.append("no scored axis could be evaluated — there is no evidence the model is safe to promote")
    else:
        failed_warn = [
            f"{r.code} {r.title}: warn gate failed"
            for r in results
            if r.gate_outcome is GateOutcome.FAILED and r.module_id not in UNSCORED_MODULES
        ]
        by_id = {r.module_id: r for r in results}
        missing_required = []
        for mid in vc.required_for_ready:
            rr = by_id.get(mid)
            if rr is None or rr.status in (Status.SKIPPED, Status.ERROR) or rr.gate_outcome is not GateOutcome.PASSED:
                title = f"{rr.code} {rr.title}" if rr else mid
                state = "not run" if rr is None else (rr.skip_reason or rr.status.value)
                missing_required.append(f"{title} must pass for a READY verdict ({state})")
        thin = coverage < vc.min_coverage_ready
        if raw >= vc.ready_min and not failed_warn and not missing_required and not thin:
            verdict = Verdict.READY
        elif raw >= vc.conditional_min:
            verdict = Verdict.CONDITIONAL
            if raw < vc.ready_min:
                reasons.append(f"score {raw:.1f} is below the READY band ({vc.ready_min:g})")
        else:
            verdict = Verdict.NOT_READY
            reasons.append(f"score {raw:.1f} is below the CONDITIONAL band ({vc.conditional_min:g})")
        reasons.extend(failed_warn)
        reasons.extend(missing_required)
        if thin:
            reasons.append(
                f"only {coverage:.0%} of the weighted battery was evaluated "
                f"(READY needs {vc.min_coverage_ready:.0%}); skipped tests are not passes"
            )
    for a in axes:
        if a.status == Status.SKIPPED.value and a.weight > 0:
            reasons.append(f"{a.code} {a.title} was skipped (lowers coverage)")

    label = VERDICT_LABELS[verdict]
    cause = next((c for c in _CAUSE_PRIORITY if c in causes), causes[0] if causes else None)
    if verdict is Verdict.BLOCKED and cause:
        label = f"Blocked — {cause}"
    if score is None:
        summary = f"{label}. No axis could be scored."
    else:
        summary = f"{label}. Production-readiness score {score:.1f}/100 ({coverage:.0%} of the weighted battery evaluated)."
        if capped:
            summary += f" Score capped at {vc.blocked_score_cap:g} because the run is blocked ({cause or 'see blockers'})."
    return VerdictResult(
        verdict=verdict,
        label=label,
        score=_r1(score),
        raw_score=_r1(raw),
        capped=capped,
        coverage=round(coverage, 4),
        reasons=reasons,
        axes=axes,
        bands=bands,
        blockers=blockers,
        summary=summary,
    )


def exit_code_for(vr: VerdictResult, results: Iterable[ModuleResult], fail_on: str = "blocked") -> int:
    """Process exit code for CI.

    0  — no hard gate failed and no module errored (and verdict not at/below ``fail_on``)
    1  — a hard gate failed / run blocked, or verdict is at/below ``fail_on``
    2  — a module errored (takes precedence: the gate result is not trustworthy)
    """
    results = list(results)
    allowed = (Verdict.BLOCKED.value, Verdict.NOT_READY.value, Verdict.CONDITIONAL.value)
    if fail_on not in allowed:
        raise ValueError(f"--fail-on must be one of {list(allowed)}")
    level = Verdict(fail_on)
    if any(r.status is Status.ERROR for r in results):
        return 2
    if vr.verdict is Verdict.BLOCKED:
        return 1
    if VERDICT_ORDER.index(vr.verdict) >= VERDICT_ORDER.index(level):
        return 1
    return 0
