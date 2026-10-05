"""Per-check axis scoring: maps a metric value to a 0..1 "production-readiness" score.

Every gated metric is graded on three anchors the module declares:

* ``floor``     — clearly unacceptable (score 0)
* ``threshold`` — the org's gate policy (score :data:`PASS_LINE`)
* ``ideal``     — nothing left to gain (score 1)

Scores interpolate piecewise-linearly between anchors (optionally in log space, for rates such as
FPR that span orders of magnitude). By construction a value that fails its gate scores strictly
below :data:`PASS_LINE`, and a value that passes scores at or above it, so the axis score can never
contradict the gate outcome.
"""

from __future__ import annotations

import math

PASS_LINE = 0.75  # score of a metric sitting exactly on its gate threshold

_EPS = 1e-12


def _t(v: float, scale: str) -> float:
    if scale == "log":
        return math.log10(max(v, _EPS))
    return v


def metric_score(
    value: float | None,
    op: str,
    threshold: float,
    *,
    ideal: float,
    floor: float,
    scale: str = "linear",
) -> float | None:
    """Grade ``value`` against the gate ``value <op> threshold``.

    ``op`` is ``">="``/``">"`` (higher is better) or ``"<="``/``"<"`` (lower is better). ``ideal`` must
    lie on the good side of ``threshold`` and ``floor`` on the bad side. Returns None for missing
    or NaN values (an unevaluated check has no score — it never defaults to good).
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if op in (">=", ">"):
        higher_better = True
    elif op in ("<=", "<"):
        higher_better = False
    else:
        raise ValueError(f"cannot score operator {op!r}")
    if scale not in ("linear", "log"):
        raise ValueError(f"unknown scale {scale!r}")
    v, t, i, f = (_t(float(x), scale) for x in (value, threshold, ideal, floor))
    if not higher_better:  # mirror so that "higher is better" below
        v, t, i, f = -v, -t, -i, -f
    if not (f < t < i):
        raise ValueError(
            f"score anchors must satisfy floor < threshold < ideal on the good-side axis "
            f"(got floor={floor}, threshold={threshold}, ideal={ideal}, op={op})"
        )
    passes = v >= t if op in (">=", "<=") else v > t
    if v >= i:
        s = 1.0
    elif v >= t:
        s = PASS_LINE + (1.0 - PASS_LINE) * (v - t) / (i - t)
    elif v <= f:
        s = 0.0
    else:
        s = PASS_LINE * (v - f) / (t - f)
    # Strict operators at exactly the threshold: failing value must stay below the pass line.
    if not passes:
        s = min(s, PASS_LINE - 1e-6)
    elif s < PASS_LINE:
        s = PASS_LINE
    return float(min(max(s, 0.0), 1.0))


def combine_axis(scores: list[float | None], how: str = "min") -> float | None:
    """Combine per-check scores into one axis score (default: weakest link)."""
    vals = [s for s in scores if s is not None]
    if not vals:
        return None
    if how == "min":
        return float(min(vals))
    if how == "mean":
        return float(sum(vals) / len(vals))
    raise ValueError(f"unknown combine rule {how!r}")
