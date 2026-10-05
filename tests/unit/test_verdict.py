"""Thorough tests of the frozen scoring (malvalid.scoring) and verdict (malvalid.verdict) logic.

Tests marked ``xfail(strict=True)`` document defects in the frozen files (reported as contract
change requests by the runner agent). They assert the *intended* behaviour, so they will XPASS —
and fail the suite as a reminder to drop the marker — once the fix lands.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from malvalid.config import GateConfig, RuntimeConfig, VerdictConfig
from malvalid.core import GateCheck, GateMode, GateOutcome, Module, ModuleResult, Status, derive_status
from malvalid.scoring import PASS_LINE, combine_axis, metric_score
from malvalid.verdict import (
    UNSCORED_MODULES,
    VERDICT_LABELS,
    VERDICT_ORDER,
    Verdict,
    compute_verdict,
    exit_code_for,
)

# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def mr(
    mid: str,
    score: float | None,
    *,
    status: Status = Status.PASS,
    gate: GateMode = GateMode.WARN,
    outcome: GateOutcome | None = None,
    code: str | None = None,
    skip_reason: str | None = None,
) -> ModuleResult:
    if outcome is None:
        outcome = {
            Status.PASS: GateOutcome.PASSED,
            Status.WARN: GateOutcome.FAILED,
            Status.FAIL: GateOutcome.FAILED,
            Status.SKIPPED: GateOutcome.NOT_EVALUATED,
            Status.ERROR: GateOutcome.NOT_EVALUATED,
        }[status]
    return ModuleResult(
        module_id=mid, code=code or f"T-{mid}", title=mid.title(), status=status, gate=gate,
        gate_outcome=outcome, finding="finding", score=score, skip_reason=skip_reason,
    )


def cfg_(**verdict_kw) -> GateConfig:
    kw = {"required_for_ready": [], "weights": {}}
    kw.update(verdict_kw)
    return GateConfig(verdict=VerdictConfig(**kw))


def strict_json(obj) -> str:
    return json.dumps(obj, allow_nan=False)


# --------------------------------------------------------------------------------------------------
# scoring.metric_score
# --------------------------------------------------------------------------------------------------


class TestMetricScore:
    def test_anchor_points_higher_is_better(self):
        kw = dict(ideal=0.99, floor=0.8)
        assert metric_score(0.8, ">=", 0.95, **kw) == 0.0
        assert metric_score(0.5, ">=", 0.95, **kw) == 0.0
        assert metric_score(0.95, ">=", 0.95, **kw) == pytest.approx(PASS_LINE)
        assert metric_score(0.99, ">=", 0.95, **kw) == 1.0
        assert metric_score(1.0, ">=", 0.95, **kw) == 1.0

    def test_linear_interpolation(self):
        kw = dict(ideal=1.0, floor=0.0)
        assert metric_score(0.25, ">=", 0.5, **kw) == pytest.approx(PASS_LINE / 2)
        assert metric_score(0.75, ">=", 0.5, **kw) == pytest.approx(PASS_LINE + (1 - PASS_LINE) / 2)

    def test_lower_is_better_mirrors(self):
        kw = dict(ideal=0.0, floor=1.0)
        assert metric_score(0.5, "<=", 0.5, **kw) == pytest.approx(PASS_LINE)
        assert metric_score(0.0, "<=", 0.5, **kw) == 1.0
        assert metric_score(1.0, "<=", 0.5, **kw) == 0.0
        assert metric_score(0.75, "<=", 0.5, **kw) == pytest.approx(PASS_LINE / 2)
        assert metric_score(0.25, "<=", 0.5, **kw) == pytest.approx(PASS_LINE + (1 - PASS_LINE) / 2)

    def test_log_scale_fpr(self):
        t = 0.01
        kw = dict(ideal=t / 10, floor=min(0.5, 5 * t), scale="log")
        assert metric_score(t, "<=", t, **kw) == pytest.approx(PASS_LINE)
        assert metric_score(t / 10, "<=", t, **kw) == 1.0
        assert metric_score(0.0, "<=", t, **kw) == 1.0  # FPR of exactly 0 is clipped, not log(0)
        # geometric midpoint between threshold and ideal
        assert metric_score(math.sqrt(t * t / 10), "<=", t, **kw) == pytest.approx(PASS_LINE + (1 - PASS_LINE) / 2)
        assert metric_score(0.05, "<=", t, **kw) == 0.0
        assert 0.0 < metric_score(0.02, "<=", t, **kw) < PASS_LINE

    def test_strict_operators_at_threshold_fail(self):
        s = metric_score(0.5, ">", 0.5, ideal=1.0, floor=0.0)
        assert s < PASS_LINE and s == pytest.approx(PASS_LINE, abs=1e-5)
        s = metric_score(0.5, "<", 0.5, ideal=0.0, floor=1.0)
        assert s < PASS_LINE
        assert metric_score(0.5, ">=", 0.5, ideal=1.0, floor=0.0) == pytest.approx(PASS_LINE)

    @pytest.mark.parametrize("v", [None, float("nan")])
    def test_missing_values_have_no_score(self, v):
        assert metric_score(v, ">=", 0.5, ideal=1.0, floor=0.0) is None

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="operator"):
            metric_score(0.5, "==", 0.5, ideal=1.0, floor=0.0)
        with pytest.raises(ValueError, match="scale"):
            metric_score(0.5, ">=", 0.5, ideal=1.0, floor=0.0, scale="sqrt")
        with pytest.raises(ValueError, match="anchors"):
            metric_score(0.5, ">=", 0.5, ideal=0.0, floor=1.0)  # anchors on the wrong sides
        with pytest.raises(ValueError, match="anchors"):
            metric_score(0.5, "<=", 0.5, ideal=1.0, floor=0.0)
        with pytest.raises(ValueError, match="anchors"):
            metric_score(0.5, ">=", 0.5, ideal=0.5, floor=0.0)  # ideal == threshold

    def test_score_never_contradicts_gate_property(self):
        """For random anchors and values: score >= PASS_LINE  <=>  the comparison passes."""
        rng = np.random.default_rng(1234)
        ops = [">=", ">", "<=", "<"]
        for _ in range(4000):
            op = ops[rng.integers(4)]
            scale = "log" if rng.random() < 0.4 else "linear"
            if scale == "log":
                t = 10 ** rng.uniform(-4, 0)
                lo, hi = t / 10 ** rng.uniform(0.1, 2), t * 10 ** rng.uniform(0.1, 2)
                v = 10 ** rng.uniform(-6, 1) if rng.random() < 0.9 else t
            else:
                t = rng.uniform(-5, 5)
                lo, hi = t - rng.uniform(0.01, 3), t + rng.uniform(0.01, 3)
                v = rng.uniform(-10, 10) if rng.random() < 0.9 else t
            ideal, floor = (hi, lo) if op in (">=", ">") else (lo, hi)
            s = metric_score(v, op, t, ideal=ideal, floor=floor, scale=scale)
            passes = {">=": v >= t, ">": v > t, "<=": v <= t, "<": v < t}[op]
            assert 0.0 <= s <= 1.0
            assert (s >= PASS_LINE) == passes, (v, op, t, ideal, floor, scale, s)

    def test_monotone_in_value(self):
        vals = np.linspace(0.5, 1.0, 101)
        scores = [metric_score(float(v), ">=", 0.95, ideal=0.995, floor=0.8) for v in vals]
        assert all(b >= a for a, b in zip(scores, scores[1:]))


class TestCombineAxis:
    def test_rules(self):
        assert combine_axis([0.9, 0.5, None]) == 0.5
        assert combine_axis([0.9, 0.5, None], "mean") == pytest.approx(0.7)
        assert combine_axis([None, None]) is None
        assert combine_axis([]) is None
        with pytest.raises(ValueError):
            combine_axis([0.5], "max")


# --------------------------------------------------------------------------------------------------
# GateCheck / derive_status / Module.result
# --------------------------------------------------------------------------------------------------


class TestGateCheck:
    def test_evaluate_with_anchors(self):
        c = GateCheck.evaluate("fpr", 0.005, "<=", 0.01, ideal=0.001, floor=0.05, scale="log", description="d")
        assert c.passed is True and c.metric == "fpr" and c.value == 0.005
        assert PASS_LINE < c.score < 1.0
        assert c.to_dict()["scale"] == "log"
        strict_json(c.to_dict())

    def test_nan_value_is_unevaluated_not_passed(self):
        c = GateCheck.evaluate("x", float("nan"), ">=", 0.5, ideal=1.0, floor=0.0)
        assert c.value is None and c.passed is None and c.score is None

    def test_none_value(self):
        c = GateCheck.evaluate("x", None, ">=", 0.5, ideal=1.0, floor=0.0)
        assert c.passed is None and c.score is None

    def test_no_anchors_means_unscored(self):
        c = GateCheck.evaluate("x", 0.7, ">=", 0.5)
        assert c.passed is True and c.score is None

    def test_equality_check_scores_binary(self):
        assert GateCheck.evaluate("ok", 1.0, "==", 1.0).score == 1.0
        assert GateCheck.evaluate("ok", 0.0, "==", 1.0).score == 0.0

    def test_metric_name_override_and_numpy_inputs(self):
        c = GateCheck.evaluate("dr", np.float32(0.97), ">=", np.float64(0.95), metric="detection_rate",
                               ideal=0.995, floor=0.8)
        assert c.metric == "detection_rate" and isinstance(c.value, float) and isinstance(c.threshold, float)

    def test_unknown_operator(self):
        with pytest.raises(ValueError):
            GateCheck.evaluate("x", 1.0, "!=", 1.0)


def _chk(passed):
    return GateCheck(name="c", metric="c", op=">=", threshold=0.5, value=None if passed is None else 1.0, passed=passed)


class TestDeriveStatus:
    @pytest.mark.parametrize(
        "checks, gate, expected",
        [
            ([_chk(True), _chk(False)], GateMode.HARD, (Status.FAIL, GateOutcome.FAILED)),
            ([_chk(True), _chk(False)], GateMode.WARN, (Status.WARN, GateOutcome.FAILED)),
            ([_chk(True), _chk(None)], GateMode.HARD, (Status.WARN, GateOutcome.NOT_EVALUATED)),
            ([_chk(False), _chk(None)], GateMode.HARD, (Status.FAIL, GateOutcome.FAILED)),
            ([_chk(True)], GateMode.HARD, (Status.PASS, GateOutcome.PASSED)),
            ([], GateMode.HARD, (Status.PASS, GateOutcome.NOT_EVALUATED)),
        ],
    )
    def test_rules(self, checks, gate, expected):
        assert derive_status(checks, gate) == expected

    def test_advisory_warn(self):
        assert derive_status([_chk(True)], GateMode.WARN, advisory_warn=True) == (Status.WARN, GateOutcome.PASSED)
        assert derive_status([], GateMode.WARN, advisory_warn=True) == (Status.WARN, GateOutcome.NOT_EVALUATED)


class _M(Module):
    id = "unit_m"
    code = "U1"
    title = "Unit"

    def run(self, ctx):  # pragma: no cover - not used
        raise NotImplementedError


class _Ctx:
    """Just enough of a RunContext for Module.result()."""

    def __init__(self, gate=GateMode.WARN):
        from malvalid.context import ArtifactStore

        self.gate = gate
        self.params = {"p": 1}
        self.artifacts = ArtifactStore()
        self.seed = 7


class TestModuleResultHelper:
    def test_default_axis_score_is_weakest_check(self):
        checks = [
            GateCheck.evaluate("a", 0.99, ">=", 0.95, ideal=0.995, floor=0.8),
            GateCheck.evaluate("b", 0.004, "<=", 0.01, ideal=0.001, floor=0.05, scale="log"),
        ]
        r = _M().result(_Ctx(), finding="f", checks=checks)
        assert r.score == pytest.approx(min(c.score for c in checks))
        r2 = _M().result(_Ctx(), finding="f", checks=checks, score_rule="mean")
        assert r2.score == pytest.approx(sum(c.score for c in checks) / 2)
        assert r.seed == 7 and r.params == {"p": 1}

    def test_explicit_score_is_clamped(self):
        assert _M().result(_Ctx(), finding="f", score=1.7).score == 1.0
        assert _M().result(_Ctx(), finding="f", score=-0.2).score == 0.0

    def test_nan_explicit_score_becomes_unscored(self):
        assert _M().result(_Ctx(), finding="f", score=float("nan")).score is None

    def test_skip_helper(self):
        r = _M().skip(_Ctx(GateMode.HARD), "no corpus")
        assert r.status is Status.SKIPPED and r.gate_outcome is GateOutcome.NOT_EVALUATED
        assert r.skip_reason == "no corpus" and r.finding == "Not run — no corpus" and r.score is None

    def test_to_dict_is_strict_json_with_infinite_check_value(self):
        c = GateCheck.evaluate("ratio", float("inf"), "<=", 1.0, ideal=0.1, floor=5.0)
        r = _M().result(_Ctx(), finding="f", checks=[c])
        strict_json(r.to_dict())


# --------------------------------------------------------------------------------------------------
# compute_verdict
# --------------------------------------------------------------------------------------------------


class TestVerdictBands:
    def test_ready(self):
        cfg = cfg_(required_for_ready=["performance"])
        vr = compute_verdict([mr("file_safety", 1.0, gate=GateMode.HARD, code="M0"),
                              mr("performance", 0.95, gate=GateMode.HARD), mr("drift", 0.9)], cfg)
        assert vr.verdict is Verdict.READY and vr.label == VERDICT_LABELS[Verdict.READY]
        assert vr.score == pytest.approx(92.5) and vr.raw_score == vr.score and not vr.capped
        assert vr.coverage == 1.0 and vr.reasons == [] and vr.blockers == []
        assert "92.5/100" in vr.summary
        assert [a.module_id for a in vr.axes] == ["performance", "drift"]  # M0 is unscored
        strict_json(vr.to_dict())

    def test_conditional_by_score(self):
        vr = compute_verdict([mr("a", 0.7)], cfg_())
        assert vr.verdict is Verdict.CONDITIONAL
        assert any("below the READY band" in r for r in vr.reasons)

    def test_not_ready_by_score(self):
        vr = compute_verdict([mr("a", 0.5)], cfg_())
        assert vr.verdict is Verdict.NOT_READY
        assert any("below the CONDITIONAL band" in r for r in vr.reasons)

    def test_band_boundaries_inclusive(self):
        assert compute_verdict([mr("a", 0.80)], cfg_()).verdict is Verdict.READY
        assert compute_verdict([mr("a", 0.60)], cfg_()).verdict is Verdict.CONDITIONAL
        assert compute_verdict([mr("a", 0.5994)], cfg_()).verdict is Verdict.NOT_READY  # 59.9 shown

    def test_custom_bands(self):
        cfg = cfg_(ready_min=90, conditional_min=70)
        assert compute_verdict([mr("a", 0.85)], cfg).verdict is Verdict.CONDITIONAL
        assert compute_verdict([mr("a", 0.65)], cfg).verdict is Verdict.NOT_READY
        assert compute_verdict([mr("a", 0.95)], cfg).bands["ready_min"] == 90.0

    def test_failed_warn_gate_prevents_ready(self):
        vr = compute_verdict([mr("a", 0.95), mr("b", 0.7, status=Status.WARN)], cfg_())
        assert vr.raw_score >= 80 and vr.verdict is Verdict.CONDITIONAL
        assert any("warn gate failed" in r for r in vr.reasons)

    def test_missing_required_axis_prevents_ready(self):
        cfg = cfg_(required_for_ready=["performance"])
        vr = compute_verdict([mr("performance", None, status=Status.SKIPPED, gate=GateMode.HARD,
                                 skip_reason="no corpus"), mr("drift", 0.95)],
                             cfg.model_copy(update={"runtime": RuntimeConfig(skipped_hard_gate_fails=False)}))
        assert vr.verdict is Verdict.CONDITIONAL
        assert any("must pass for a READY verdict (no corpus)" in r for r in vr.reasons)
        vr2 = compute_verdict([mr("drift", 0.95)], cfg)
        assert any("performance must pass" in r and "(not run)" in r for r in vr2.reasons)

    def test_thin_coverage_prevents_ready(self):
        cfg = cfg_(weights={"a": 1, "b": 3})
        vr = compute_verdict([mr("a", 1.0), mr("b", None, status=Status.SKIPPED, skip_reason="x")], cfg)
        assert vr.coverage == pytest.approx(0.25) and vr.verdict is Verdict.CONDITIONAL
        assert any("only 25% of the weighted battery" in r for r in vr.reasons)
        assert any("was skipped (lowers coverage)" in r for r in vr.reasons)

    def test_nothing_scored(self):
        vr = compute_verdict([mr("a", None, status=Status.SKIPPED, skip_reason="x")], cfg_())
        assert vr.verdict is Verdict.NOT_READY and vr.score is None and vr.raw_score is None
        assert vr.coverage == 0.0 and "No axis could be scored" in vr.summary
        strict_json(vr.to_dict())

    def test_empty_results(self):
        vr = compute_verdict([], cfg_())
        assert vr.verdict is Verdict.NOT_READY and vr.coverage == 0.0 and vr.axes == []


class TestVerdictWeights:
    def test_weighted_mean(self):
        cfg = cfg_(weights={"a": 3, "b": 1})
        vr = compute_verdict([mr("a", 1.0), mr("b", 0.6)], cfg)
        assert vr.raw_score == pytest.approx(90.0)

    def test_default_weight(self):
        cfg = cfg_(weights={"a": 2}, default_weight=0.5)
        vr = compute_verdict([mr("a", 1.0), mr("b", 0.0)], cfg)
        assert vr.raw_score == pytest.approx(100 * 2 / 2.5)

    def test_zero_weight_is_informational(self):
        cfg = cfg_(weights={"a": 1, "b": 0})
        vr = compute_verdict([mr("a", 0.9), mr("b", 0.0), mr("c", None, status=Status.SKIPPED, skip_reason="x")],
                             cfg_(weights={"a": 1, "b": 0, "c": 0}))
        assert vr.raw_score == pytest.approx(90.0) and vr.coverage == 1.0
        b = next(a for a in vr.axes if a.module_id == "b")
        assert b.counted is False and b.score == 0.0
        assert not any("C was skipped" in r for r in vr.reasons)  # zero-weight skips do not lower coverage
        del cfg

    def test_unscored_modules_excluded(self):
        assert {"file_safety", "dummy"} <= UNSCORED_MODULES
        vr = compute_verdict([mr("dummy", 0.0), mr("file_safety", 0.0, code="M0"), mr("a", 0.9)], cfg_())
        assert [a.module_id for a in vr.axes] == ["a"] and vr.raw_score == pytest.approx(90.0)

    def test_axis_scores_on_0_100_scale(self):
        vr = compute_verdict([mr("a", 0.87654)], cfg_())
        assert vr.axes[0].score == 87.7 and vr.axes[0].counted is True


class TestBlocked:
    def test_hard_gate_failure_caps_score(self):
        vr = compute_verdict([mr("a", 0.7, status=Status.FAIL, gate=GateMode.HARD), mr("b", 1.0)], cfg_())
        assert vr.verdict is Verdict.BLOCKED and vr.capped is True
        assert vr.score == 49.0 and vr.raw_score == pytest.approx(85.0)
        assert vr.blockers == ["T-a A: hard gate failed"] and vr.blockers[0] in vr.reasons
        assert "capped at 49" in vr.summary

    def test_low_raw_score_is_not_capped_upward(self):
        vr = compute_verdict([mr("a", 0.2, status=Status.FAIL, gate=GateMode.HARD)], cfg_())
        assert vr.verdict is Verdict.BLOCKED and vr.capped is False and vr.score == pytest.approx(20.0)

    def test_custom_cap(self):
        vr = compute_verdict([mr("a", 0.7, status=Status.FAIL, gate=GateMode.HARD)], cfg_(blocked_score_cap=30))
        assert vr.score == 30.0

    def test_errored_module_blocks(self):
        vr = compute_verdict([mr("a", None, status=Status.ERROR), mr("b", 0.95)], cfg_())
        assert vr.verdict is Verdict.BLOCKED and "T-a A errored" in vr.blockers
        a = next(x for x in vr.axes if x.module_id == "a")
        assert a.counted is False and a.score is None

    def test_errored_module_score_is_ignored(self):
        vr = compute_verdict([mr("a", 0.1, status=Status.ERROR), mr("b", 0.95)], cfg_())
        assert vr.raw_score == pytest.approx(95.0)

    def test_skipped_hard_gate_blocks_by_default(self):
        vr = compute_verdict([mr("a", None, status=Status.SKIPPED, gate=GateMode.HARD, skip_reason="no corpus"),
                              mr("b", 0.95)], cfg_())
        assert vr.verdict is Verdict.BLOCKED
        assert vr.blockers == ["T-a A: hard gate not evaluated (no corpus)"]

    def test_skipped_hard_gate_policy_off(self):
        cfg = cfg_().model_copy(update={"runtime": RuntimeConfig(skipped_hard_gate_fails=False)})
        vr = compute_verdict([mr("a", None, status=Status.SKIPPED, gate=GateMode.HARD, skip_reason="x"),
                              mr("b", 0.95)], cfg)
        assert vr.verdict is not Verdict.BLOCKED

    def test_hard_gate_with_unevaluable_check_blocks(self):
        vr = compute_verdict([mr("a", 0.9, status=Status.WARN, gate=GateMode.HARD, outcome=GateOutcome.NOT_EVALUATED)],
                             cfg_())
        assert vr.verdict is Verdict.BLOCKED and "gate could not be evaluated" in vr.blockers[0]

    def test_m0_pass_without_checks_is_not_a_blocker(self):
        m0 = mr("file_safety", 1.0, gate=GateMode.HARD, outcome=GateOutcome.NOT_EVALUATED, code="M0")
        vr = compute_verdict([m0, mr("a", 0.95)], cfg_())
        assert vr.verdict is Verdict.READY

    def test_m0_hard_failure_blocks(self):
        m0 = mr("file_safety", 0.0, status=Status.FAIL, gate=GateMode.HARD, code="M0")
        vr = compute_verdict([m0, mr("a", 0.95)], cfg_())
        assert vr.verdict is Verdict.BLOCKED

    def test_dummy_hard_not_evaluated_is_not_a_blocker(self):
        d = mr("dummy", None, status=Status.SKIPPED, gate=GateMode.HARD)
        assert compute_verdict([d, mr("a", 0.95)], cfg_()).verdict is Verdict.READY

    def test_aborted_run(self):
        vr = compute_verdict([mr("file_safety", 0.0, status=Status.FAIL, gate=GateMode.HARD, code="M0"),
                              mr("a", None, status=Status.SKIPPED, skip_reason="run aborted by M0: x")],
                             cfg_(), aborted="pickle artifact")
        assert vr.verdict is Verdict.BLOCKED and vr.score is None
        assert vr.blockers[0] == "run aborted before the model was loaded: pickle artifact"

    def test_aborted_with_no_results(self):
        vr = compute_verdict([], cfg_(), aborted="x")
        assert vr.verdict is Verdict.BLOCKED and vr.coverage == 0.0
        strict_json(vr.to_dict())


class TestExitCodes:
    def test_precedence(self):
        ok = [mr("a", 0.95)]
        assert exit_code_for(compute_verdict(ok, cfg_()), ok) == 0
        fail = [mr("a", 0.9, status=Status.FAIL, gate=GateMode.HARD)]
        assert exit_code_for(compute_verdict(fail, cfg_()), fail) == 1
        err = fail + [mr("b", None, status=Status.ERROR)]
        assert exit_code_for(compute_verdict(err, cfg_()), err) == 2

    @pytest.mark.parametrize(
        "score, fail_on, code",
        [
            (0.95, "blocked", 0), (0.95, "conditional", 0), (0.95, "not_ready", 0),
            (0.70, "blocked", 0), (0.70, "not_ready", 0), (0.70, "conditional", 1),
            (0.40, "blocked", 0), (0.40, "not_ready", 1), (0.40, "conditional", 1),
        ],
    )
    def test_fail_on_levels(self, score, fail_on, code):
        res = [mr("a", score)]
        assert exit_code_for(compute_verdict(res, cfg_()), res, fail_on) == code

    def test_invalid_fail_on(self):
        res = [mr("a", 0.95)]
        with pytest.raises(ValueError, match="--fail-on"):
            exit_code_for(compute_verdict(res, cfg_()), res, "bogus")

    def test_invalid_fail_on_rejected_even_when_blocked(self):
        res = [mr("a", 0.9, status=Status.FAIL, gate=GateMode.HARD)]
        with pytest.raises(ValueError):
            exit_code_for(compute_verdict(res, cfg_()), res, "bogus")

    def test_verdict_order(self):
        assert VERDICT_ORDER == [Verdict.READY, Verdict.CONDITIONAL, Verdict.NOT_READY, Verdict.BLOCKED]


# --------------------------------------------------------------------------------------------------
# Documented defects (see contract_change_requests)
# --------------------------------------------------------------------------------------------------


class TestKnownDefects:
    def test_displayed_score_agrees_with_band(self):
        vr = compute_verdict([mr("a", 0.7996), mr("b", 0.7996)], cfg_())
        # A researcher sees 80.0/100; READY needs 80 — the band decision must agree with the display.
        assert (vr.score >= 80) == (vr.verdict is Verdict.READY)

    def test_summary_does_not_round_into_the_next_band(self):
        vr = compute_verdict([mr("a", 0.795)], cfg_())
        assert vr.verdict is Verdict.CONDITIONAL and "80/100" not in vr.summary

    def test_nan_axis_score_is_ignored(self):
        vr = compute_verdict([mr("a", float("nan")), mr("b", 0.9)], cfg_())
        assert vr.score == pytest.approx(90.0)
        strict_json(vr.to_dict())

    def test_out_of_range_axis_score_is_clamped(self):
        assert compute_verdict([mr("a", 1.5)], cfg_()).score <= 100

    def test_failed_file_safety_warn_gate_prevents_ready(self):
        m0 = mr("file_safety", 0.0, status=Status.WARN, gate=GateMode.WARN, code="M0")
        vr = compute_verdict([m0, mr("performance", 0.95)], cfg_())
        assert vr.verdict is not Verdict.READY

    def test_blocked_label_matches_its_cause(self):
        # Blocked because a module errored (no hard gate failed): the headline must not claim one did.
        vr = compute_verdict([mr("a", None, status=Status.ERROR), mr("b", 0.95)], cfg_())
        assert vr.verdict is Verdict.BLOCKED
        assert "hard gate failed" not in vr.label

    def test_capped_summary_matches_its_cause(self):
        vr = compute_verdict([mr("a", None, status=Status.ERROR), mr("b", 0.95)], cfg_())
        assert vr.capped and "because a hard gate failed" not in vr.summary



class TestRunnerVerdictWording:
    """The runner's report block works around the wording defects above without changing the verdict."""

    def test_report_summary_keeps_one_decimal(self):
        import types

        from malvalid.runner import _GateRun

        vr = compute_verdict([mr("a", 0.795)], cfg_())
        d = _GateRun._verdict_block(types.SimpleNamespace(), vr)
        assert d["verdict"] == "conditional" and d["score"] == vr.score
        assert "79.5/100" in d["summary"] and "80/100" not in d["summary"]
        assert d["label"] == vr.label and d["axes"] == vr.to_dict()["axes"]
