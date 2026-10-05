"""Unit tests for M1 performance & calibration (malvalid.modules.performance)."""

from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import pytest

from malvalid.config import GateConfig, load_config
from malvalid.context import REQUIREMENT_HINTS, ModelDeclarations
from malvalid.core import ConfigError, GateOutcome, ModuleTimeout, Requirement, Status, unmet_requirements
from malvalid.corpora.base import Corpus
from malvalid.modules.performance import (
    PerformanceModule,
    expected_calibration_error,
    graded_check,
    roc_arrays,
    threshold_for_fpr,
    tpr_at_fpr,
    wilson_interval,
)
from malvalid.testing import (
    InProcessModel,
    make_context,
    make_toy_corpus,
    train_toy_lgbm,
    training_hashes_for,
)
from malvalid.verdict import Verdict, compute_verdict

# Params under which the clean toy model (FPR ~2.9%, DR ~93.8% at 0.5) passes.
LENIENT = {"max_fpr": 0.05, "min_detection": 0.90}


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return make_toy_corpus()


@pytest.fixture(scope="module")
def booster(corpus):
    return train_toy_lgbm(corpus)


@pytest.fixture(scope="module")
def hashes(corpus) -> frozenset[str]:
    return training_hashes_for(corpus)


def _model(booster, threshold: float = 0.5) -> InProcessModel:
    return InProcessModel.from_lightgbm(booster, threshold=threshold, feature_version="toy_v1")


def _run(model, corpus, **kw):
    ctx = make_context(PerformanceModule, model=model, corpus=corpus, **kw)
    return PerformanceModule().run(ctx), ctx


class _FnDetector:
    """Detector whose score is feature 0 (for hand-built corpora)."""

    def __init__(self, threshold: float = 0.5, predict_threshold: float | None = None):
        self.t = threshold if predict_threshold is None else predict_threshold

    def predict_proba(self, X):
        return X[:, 0].astype(np.float64)

    def predict(self, X):
        return (X[:, 0] >= self.t).astype(np.int8)


def _fn_model(threshold: float = 0.5, predict_threshold: float | None = None) -> InProcessModel:
    decl = ModelDeclarations(
        feature_version="toy_v1", model_kind="custom", operating_threshold=threshold,
        training_hashes_path=None, training_cutoff=None,
    )
    return InProcessModel(_FnDetector(threshold, predict_threshold), decl)


def _handmade(scores: np.ndarray, labels: np.ndarray, *, roles=None) -> Corpus:
    n = scores.size
    X = np.zeros((n, 32), dtype=np.float32)
    X[:, 0] = scores
    manifest = {
        "format": "malvalid-corpus/1", "name": "handmade", "version": "1", "feature_version": "toy_v1",
        "dim": 32, "n": n, "roles": roles or {"eval": ["test"]}, "files": {"X": "none"},
    }
    return Corpus(
        name="handmade", version="1", feature_version="toy_v1", content_hash="handmade", manifest=manifest, X=X,
        sha256=np.array([hashlib.sha256(f"h{i}".encode()).hexdigest() for i in range(n)], dtype="<U64"),
        label=labels.astype(np.int8), timestamp=np.full(n, np.datetime64("2018-01-15", "D")),
        split=np.array(["test"] * n, dtype="<U16"), synthetic=True,
    )


# --------------------------------------------------------------------------------------------------
# pass / fail
# --------------------------------------------------------------------------------------------------


def test_pass_case(corpus, booster, hashes):
    r, ctx = _run(_model(booster), corpus, training_hashes=hashes, params=LENIENT)
    assert r.status is Status.PASS
    assert r.gate_outcome is GateOutcome.PASSED
    assert r.score is not None and r.score >= 0.75
    m = r.metrics
    for key in ("detection_rate", "fpr", "n_benign", "n_malicious", "threshold", "auroc", "auprc", "brier", "ece",
                "tpr_at_fpr_0.001", "tpr_at_fpr_0.01", "threshold_at_max_fpr", "detection_rate_ci95", "fpr_ci95",
                "challenge_detection_rate"):
        assert key in m, key
    assert m["n_benign"] == 997 and m["n_malicious"] == 855
    assert m["fpr"] <= 0.05 and m["detection_rate"] >= 0.90
    assert m["auroc"] > 0.95
    assert {c.name for c in r.checks} == {"fpr", "detection_rate"}
    assert all(c.passed for c in r.checks)
    keys = set(ctx.artifacts.keys_for("performance"))
    assert {"performance.roc", "performance.pr", "performance.calibration", "performance.score_hist"} <= keys
    assert set(r.artifacts) == keys
    roc = ctx.artifacts.get("performance.roc")
    assert roc["xscale"] == "log"
    assert any(rl["axis"] == "x" and rl["value"] == 0.05 for rl in roc["reference_lines"])
    assert any("operating point" in s["label"] for s in roc["series"])
    assert ctx.artifacts.get("performance.calibration")["note"] == "diagonal"
    hist = ctx.artifacts.get("performance.score_hist")
    assert len(hist["series"]) == 2 and hist["reference_lines"][0]["value"] == 0.5
    json.dumps({"r": r.to_dict(), "a": ctx.artifacts.to_dict()}, allow_nan=False)
    assert "Both hard-gate conditions hold" in r.finding


def test_low_threshold_high_fpr_fails_hard_gate(corpus, booster, hashes):
    r, _ = _run(_model(booster, threshold=0.01), corpus, training_hashes=hashes)
    assert r.status is Status.FAIL
    assert r.gate_outcome is GateOutcome.FAILED
    fpr_check = next(c for c in r.checks if c.name == "fpr")
    assert fpr_check.passed is False and fpr_check.value > 0.01
    assert r.score is not None and r.score < 0.75
    assert "false-positive rate" in r.finding.lower()
    assert "would meet max_fpr" in r.finding  # actionable retune hint
    # the headline verdict is blocked by the failed hard gate
    vr = compute_verdict([r], GateConfig())
    assert vr.verdict is Verdict.BLOCKED
    assert vr.score is None or vr.score <= 49


def test_overpredicting_model_fails_hard_gate(corpus, hashes):
    # Train with 40% of benign training labels flipped to malicious: the model over-predicts malware.
    rng = np.random.default_rng(1)
    labels = corpus.label.copy()
    ben_train = corpus.indices(splits=("train",), label=0)
    flip = rng.choice(ben_train, size=int(0.4 * ben_train.size), replace=False)
    labels[flip] = 1
    bad = train_toy_lgbm(corpus, labels=labels)
    r, _ = _run(_model(bad), corpus, training_hashes=hashes, params=LENIENT)  # clean model passes these
    assert r.status is Status.FAIL
    assert r.gate_outcome is GateOutcome.FAILED
    assert next(c for c in r.checks if c.name == "fpr").passed is False
    assert r.metrics["fpr"] > 0.05


# --------------------------------------------------------------------------------------------------
# metric sanity vs scikit-learn
# --------------------------------------------------------------------------------------------------


def test_metrics_match_sklearn(corpus, booster, hashes):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

    r, _ = _run(_model(booster), corpus, training_hashes=hashes, params={"calibration_bins": 10})
    ben, mal = corpus.eval_indices(0, exclude_hashes=hashes), corpus.eval_indices(1, exclude_hashes=hashes)
    sb, sm = booster.predict(corpus.take(ben)), booster.predict(corpus.take(mal))
    y = np.r_[np.zeros(sb.size), np.ones(sm.size)]
    s = np.r_[sb, sm]
    m = r.metrics
    assert m["fpr"] == pytest.approx(np.mean(sb >= 0.5))
    assert m["detection_rate"] == pytest.approx(np.mean(sm >= 0.5))
    assert m["auroc"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert m["auprc"] == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert m["brier"] == pytest.approx(brier_score_loss(y, s), abs=1e-12)
    # ECE: naive per-bin loop over sklearn's own reliability curve
    prob_true, prob_pred = calibration_curve(y, s, n_bins=10, strategy="uniform")
    edges = np.linspace(0, 1, 11)
    ids = np.searchsorted(edges[1:-1], s)
    counts = np.array([np.sum(ids == b) for b in range(10)])
    naive = np.sum(counts[counts > 0] / s.size * np.abs(prob_true - prob_pred))
    assert m["ece"] == pytest.approx(naive, abs=1e-12)
    # TPR at FPR targets: best TPR over thresholds with FPR <= target (sklearn ROC)
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    for tgt in (0.001, 0.01):
        assert m[f"tpr_at_fpr_{tgt}"] == pytest.approx(tpr[fpr <= tgt].max())
    # threshold_at_max_fpr: meets max_fpr, and the next lower distinct score would not
    th = m["threshold_at_max_fpr"]
    assert np.mean(sb >= th) <= 0.01
    lower = s[s < th]
    if lower.size:
        assert np.mean(sb >= lower.max()) > 0.01


def test_handmade_exact_values():
    # 750 benign: 100 score 0.9 (FP); 750 malicious: 625 score 0.9 (TP), 125 score 0.1.
    s = np.r_[np.full(100, 0.9), np.full(650, 0.1), np.full(625, 0.9), np.full(125, 0.1)]
    y = np.r_[np.zeros(750), np.ones(750)]
    c = _handmade(s, y)
    r, _ = _run(_fn_model(0.5), c, params={"max_fpr": 0.2, "min_detection": 0.8})
    assert r.metrics["fpr"] == pytest.approx(100 / 750)
    assert r.metrics["detection_rate"] == pytest.approx(625 / 750)
    assert r.metrics["fpr_ci95"] == pytest.approx(wilson_interval(100, 750))
    assert r.status is Status.PASS
    assert r.details["confusion"] == {"tp": 625, "fn": 125, "fp": 100, "tn": 650}
    assert "challenge_detection_rate" not in r.metrics  # no challenge role in this corpus
    assert r.details["challenge"] == {"available": False}


def _doubled(c: Corpus) -> Corpus:
    """The same corpus with every row (and its sha256) appearing twice."""
    two = lambda a: np.concatenate([a, a])  # noqa: E731
    return Corpus(
        name=c.name, version=c.version, feature_version=c.feature_version, content_hash="doubled",
        manifest={**c.manifest, "n": 2 * c.n}, X=two(np.asarray(c.X)), sha256=two(c.sha256), label=two(c.label),
        timestamp=two(c.timestamp), split=two(c.split), synthetic=True,
    )


def test_duplicate_rows_counted_once():
    s = np.r_[np.full(100, 0.9), np.full(650, 0.1), np.full(625, 0.9), np.full(125, 0.1)]
    y = np.r_[np.zeros(750), np.ones(750)]
    params = {"max_fpr": 0.2, "min_detection": 0.8}
    r1, _ = _run(_fn_model(0.5), _handmade(s, y), params=params)
    r2, _ = _run(_fn_model(0.5), _doubled(_handmade(s, y)), params=params)
    assert r2.metrics["n_benign"] == 750 and r2.metrics["n_malicious"] == 750
    assert r2.details["eval_set"]["duplicates_removed"] == {"benign": 750, "malicious": 750}
    assert r1.details["eval_set"]["duplicates_removed"] == {"benign": 0, "malicious": 0}
    for key in ("fpr", "detection_rate", "fpr_ci95", "detection_rate_ci95", "auroc", "ece"):
        assert r2.metrics[key] == pytest.approx(r1.metrics[key]), key
    assert any("duplicate" in n for n in r2.notes)
    assert not any("duplicate" in n for n in r1.notes)


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.27753, abs=1e-5)
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.23659, abs=1e-5) and hi == pytest.approx(0.76341, abs=1e-5)
    assert wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) is None


def test_roc_helpers_and_ece_edge_cases():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.6, 0.9])
    f, t, thr = roc_arrays(y, s)
    assert tpr_at_fpr(f, t, 0.0) == pytest.approx(0.5)
    th, fp, tp = threshold_for_fpr(f, t, thr, 0.0)
    assert th == pytest.approx(0.9) and fp == 0.0 and tp == 0.5
    # all benign share the top score: no finite threshold meets FPR 0
    th2, _, _ = threshold_for_fpr(*roc_arrays(np.array([0, 1]), np.array([1.0, 1.0])), 0.0)
    assert th2 is None
    ece, table = expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0]), 5)
    assert ece == 0.0 and sum(table["count"]) == 2


def test_graded_check_extreme_thresholds():
    c = graded_check("detection_rate", 1.0, ">=", 1.0, ideal=1.0, floor=0.85)
    assert c.passed and c.score is not None
    c = graded_check("fpr", 0.2, "<=", 0.9, ideal=0.09, floor=0.5, scale="log")
    assert c.passed and c.score is not None and c.score >= 0.75
    c = graded_check("detection_rate", 0.5, ">=", 0.0, ideal=0.9, floor=0.0)
    assert c.passed


# --------------------------------------------------------------------------------------------------
# eval-set hygiene, subsampling, challenge
# --------------------------------------------------------------------------------------------------


def test_training_members_excluded_and_reported(corpus, booster, hashes):
    ben = corpus.eval_indices(0)[:50]
    mal = corpus.eval_indices(1)[:30]
    leaked = hashes | frozenset(corpus.sha256[np.r_[ben, mal]].tolist())
    r, _ = _run(_model(booster), corpus, training_hashes=leaked)
    assert r.metrics["n_benign"] == corpus.eval_indices(0).size - 50
    assert r.metrics["n_malicious"] == corpus.eval_indices(1).size - 30
    assert r.details["excluded_training_members"] == {"benign": 50, "malicious": 30}
    assert any("80 canonical eval samples" in n for n in r.notes)


def test_no_manifest_note(corpus, booster):
    r, _ = _run(_model(booster), corpus)
    assert r.details["excluded_training_members"] is None
    assert any("No training manifest" in n for n in r.notes)


def test_empty_declared_manifest_note(corpus, booster):
    """Regression (xcomp-empty-manifest-reported-as-undeclared)."""
    ctx = make_context(PerformanceModule, model=_model(booster), corpus=corpus)
    ctx.extras["training_manifest_declared"] = True
    r = PerformanceModule().run(ctx)
    assert any(n.startswith("The declared training manifest contains no sha256 hashes") for n in r.notes)
    assert not any("No training manifest was declared" in n for n in r.notes)


def test_all_eval_rows_are_members_is_unevaluated(corpus, booster):
    everything = frozenset(corpus.sha256.tolist())
    r, _ = _run(_model(booster), corpus, training_hashes=everything)
    assert r.status is Status.WARN
    assert r.gate_outcome is GateOutcome.NOT_EVALUATED
    assert all(c.passed is None for c in r.checks)
    assert "not evaluated" in r.finding
    vr = compute_verdict([r], GateConfig())
    assert vr.verdict is Verdict.BLOCKED  # an unevaluated hard gate never promotes


def test_subsampling_recorded_and_deterministic(corpus, booster, hashes):
    r1, _ = _run(_model(booster), corpus, training_hashes=hashes, params={"max_samples_per_class": 300})
    r2, _ = _run(_model(booster), corpus, training_hashes=hashes, params={"max_samples_per_class": 300})
    assert r1.metrics["n_benign"] == 300 and r1.metrics["n_malicious"] == 300
    ev = r1.details["eval_set"]
    assert ev["subsampled"] is True and ev["benign_available"] == 997 and ev["benign_used"] == 300
    assert any("Subsampled" in n for n in r1.notes)
    assert r1.metrics == r2.metrics


def test_challenge_toggle(corpus, booster, hashes):
    r, _ = _run(_model(booster), corpus, training_hashes=hashes, params={"include_challenge": False})
    assert "challenge_detection_rate" not in r.metrics
    r, _ = _run(_model(booster), corpus, training_hashes=hashes)
    ch = corpus.indices(role="challenge", label=1)
    assert r.metrics["challenge_detection_rate"] == pytest.approx(np.mean(booster.predict(corpus.take(ch)) >= 0.5))
    assert r.details["challenge"]["n"] == ch.size


def test_scores_in_batches(corpus, booster, hashes):
    model = _model(booster)
    calls: list[int] = []
    orig = model.predict_proba

    def spy(X):
        calls.append(X.shape[0])
        return orig(X)

    model.predict_proba = spy  # type: ignore[method-assign]
    ctx = make_context(PerformanceModule, model=model, corpus=corpus, training_hashes=hashes)
    ctx.config = GateConfig(runtime={"chunk_rows": 256})
    PerformanceModule().run(ctx)
    assert max(calls) <= 256
    assert sum(calls) == 997 + 855 + 200  # benign + malicious + challenge, each row scored once


def test_predict_inconsistency_noted():
    s = np.r_[np.linspace(0, 0.4, 200), np.linspace(0.6, 1.0, 200)]
    y = np.r_[np.zeros(200), np.ones(200)]
    r, _ = _run(_fn_model(0.5, predict_threshold=0.8), _handmade(s, y))
    assert r.details["predict_consistency"]["disagreements"] > 0
    assert any("predict() disagrees" in n for n in r.notes)


# --------------------------------------------------------------------------------------------------
# skips, params, deadline
# --------------------------------------------------------------------------------------------------


def test_skip_without_corpus(booster):
    ctx = make_context(PerformanceModule, model=_model(booster), corpus=None)
    assert unmet_requirements(PerformanceModule, ctx.capabilities) == [Requirement.FEATURE_SPACE]
    r = PerformanceModule().run(ctx)
    assert r.status is Status.SKIPPED
    assert r.gate_outcome is GateOutcome.NOT_EVALUATED
    assert r.skip_reason == REQUIREMENT_HINTS[Requirement.FEATURE_SPACE]
    assert r.score is None


@pytest.mark.parametrize(
    "params",
    [{"max_fpr": 0.0}, {"max_fpr": 1.5}, {"max_fpr": "0.01"}, {"min_detection": 1.2}, {"calibration_bins": 1},
     {"max_samples_per_class": 0}, {"include_challenge": "yes"}],
)
def test_invalid_params(corpus, booster, params):
    ctx = make_context(PerformanceModule, model=_model(booster), corpus=corpus, params=params)
    with pytest.raises(ConfigError):
        PerformanceModule().run(ctx)


def test_default_gate_yaml_keys_are_params():
    cfg = load_config()
    assert set(cfg.module("performance").params()) <= set(PerformanceModule.default_params)
    assert cfg.module("performance").gate.value == "hard"
    assert PerformanceModule.default_gate.value == "hard"


def test_deadline_enforced(corpus, booster):
    ctx = make_context(PerformanceModule, model=_model(booster), corpus=corpus, deadline=time.monotonic() - 1)
    with pytest.raises(ModuleTimeout):
        PerformanceModule().run(ctx)


def test_json_safe_with_single_class():
    s = np.linspace(0, 0.3, 50)
    r, ctx = _run(_fn_model(0.5), _handmade(s, np.zeros(50)))
    assert r.metrics["auroc"] is None and r.metrics["detection_rate"] is None
    assert r.status is Status.WARN and r.gate_outcome is GateOutcome.NOT_EVALUATED
    json.dumps({"r": r.to_dict(), "a": ctx.artifacts.to_dict()}, allow_nan=False)
    assert not math.isnan(r.metrics["ece"])


def test_non_finite_scores_raise_clear_error():
    """Defense in depth: the sandbox and InProcessModel reject NaN scores, but other handles may not."""
    s = np.r_[np.linspace(0, 0.4, 100), np.linspace(0.6, 1.0, 100)]
    s[150] = np.nan
    y = np.r_[np.zeros(100), np.ones(100)]
    ctx = make_context(PerformanceModule, model=_fn_model(0.5), corpus=_handmade(s, y))
    object.__setattr__(ctx, "score", lambda X: X[:, 0].astype(np.float64))  # an unvalidated handle
    with pytest.raises(ValueError, match="non-finite score.*corpus row 150"):
        PerformanceModule().run(ctx)


# --------------------------------------------------------------------------------------------------
# threshold-uncertainty bootstrap (auto-calibrated thresholds only)
# --------------------------------------------------------------------------------------------------


def _calibrated_case(min_detection: float, *, seed: int = 0, resamples: int | None = None):
    """Uniform benign scores; malicious scores straddle the calibrated threshold (~0.95)."""
    from malvalid.submission import calibrate_threshold

    r = np.random.default_rng(1)
    cal = r.uniform(0.0, 1.0, 2000)
    t, _ = calibrate_threshold(cal, 0.05)
    ben = r.uniform(0.0, 1.0, 3000)
    mal = r.uniform(0.9, 1.0, 3000)
    corpus = _handmade(np.concatenate([ben, mal]), np.r_[np.zeros(3000), np.ones(3000)])
    params = {"max_fpr": 0.2, "min_detection": min_detection}
    if resamples is not None:
        params["threshold_bootstrap_resamples"] = resamples
    ctx = make_context(PerformanceModule, model=_fn_model(t), corpus=corpus, params=params, seed=seed)
    ctx.extras["threshold_calibration"] = {"scores": cal, "target_fpr": 0.05, "threshold": t}
    return PerformanceModule().run(ctx), ctx, t


def test_threshold_bootstrap_borderline_note_and_intervals():
    r, _, t = _calibrated_case(0.5)
    tb = r.details["threshold_bootstrap"]
    assert tb["applicable"] and tb["ran"] and tb["resamples"] == 1000 and tb["n_calibration"] == 2000
    m = r.metrics
    lo, hi = m["threshold_ci95"]
    assert lo < t < hi
    for key, point in (("fpr_ci95_with_threshold", m["fpr"]), ("detection_rate_ci95_with_threshold", m["detection_rate"])):
        lo, hi = m[key]
        assert lo <= point <= hi, key
    # threshold noise widens the detection-rate interval well beyond the eval-only Wilson interval
    w = m["detection_rate_ci95"]
    assert (m["detection_rate_ci95_with_threshold"][1] - m["detection_rate_ci95_with_threshold"][0]) > 2 * (w[1] - w[0])
    pf = tb["pass_fraction"]
    assert pf["fpr"] == 1.0 and 0.025 < pf["detection_rate"] < 0.975
    assert pf["both"] == pytest.approx(pf["detection_rate"])
    assert tb["decided_within_noise"] == ["detection_rate"]
    notes = [n for n in r.notes if "within threshold noise" in n]
    assert len(notes) == 1 and notes[0].startswith("Detection-rate gate decided within threshold noise")
    assert "bootstrap replicates" in notes[0]
    json.dumps(r.to_dict() if hasattr(r, "to_dict") else r.details, allow_nan=False)
    assert "scores" not in tb


def test_threshold_bootstrap_clear_case_has_no_note():
    r, _, _ = _calibrated_case(0.1)
    tb = r.details["threshold_bootstrap"]
    assert tb["pass_fraction"]["detection_rate"] == 1.0 and tb["pass_fraction"]["fpr"] == 1.0
    assert tb["decided_within_noise"] == []
    assert not any("threshold noise" in n for n in r.notes)


def test_threshold_bootstrap_reproducible_with_seed():
    a, _, _ = _calibrated_case(0.5, seed=7)
    b, _, _ = _calibrated_case(0.5, seed=7)
    c, _, _ = _calibrated_case(0.5, seed=8)
    strip = lambda d: {k: v for k, v in d.items() if k != "duration_s"}  # noqa: E731
    assert strip(a.details["threshold_bootstrap"]) == strip(b.details["threshold_bootstrap"])
    assert strip(a.details["threshold_bootstrap"]) != strip(c.details["threshold_bootstrap"])


def test_threshold_bootstrap_leaves_point_estimates_and_gate_unchanged():
    with_bs, _, t = _calibrated_case(0.5)
    without, _, _ = _calibrated_case(0.5, resamples=0)
    assert without.details["threshold_bootstrap"] == {"applicable": True, "ran": False,
                                                       "reason": "threshold_bootstrap_resamples is 0"}
    new = {"threshold_ci95", "fpr_ci95_with_threshold", "detection_rate_ci95_with_threshold"}
    assert set(with_bs.metrics) - set(without.metrics) == new
    assert {k: v for k, v in with_bs.metrics.items() if k not in new} == without.metrics
    assert [(c.name, c.value, c.passed, c.score) for c in with_bs.checks] == \
        [(c.name, c.value, c.passed, c.score) for c in without.checks]
    assert with_bs.status is without.status and with_bs.score == without.score
    assert with_bs.gate_outcome is without.gate_outcome


def test_threshold_bootstrap_skipped_for_declared_threshold(corpus, booster, hashes):
    r, _ = _run(_model(booster), corpus, training_hashes=hashes, params=LENIENT)
    tb = r.details["threshold_bootstrap"]
    assert tb["applicable"] is False and "declared" in tb["reason"]
    assert not {"threshold_ci95", "fpr_ci95_with_threshold", "detection_rate_ci95_with_threshold"} & set(r.metrics)
    assert not any("threshold noise" in n for n in r.notes)


def test_threshold_bootstrap_matches_materialised_resampling():
    """The binomial shortcut for the eval rows agrees with explicitly resampling them."""
    from malvalid.modules.performance import threshold_bootstrap
    from malvalid.submission import calibrate_threshold

    r = np.random.default_rng(3)
    cal, ben, mal = r.uniform(0, 1, 1500), r.uniform(0, 1, 800), r.beta(8, 1, 600)
    fast = threshold_bootstrap(cal, 0.02, ben, mal, max_fpr=0.03, min_dr=0.6, resamples=3000,
                               rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    th, fpr, dr = [], [], []
    for _ in range(3000):
        t = calibrate_threshold(rng.choice(cal, cal.size), 0.02)[0]
        th.append(t)
        fpr.append(np.mean(rng.choice(ben, ben.size) >= t))
        dr.append(np.mean(rng.choice(mal, mal.size) >= t))
    for key, ref in (("threshold_ci95", th), ("fpr_ci95", fpr), ("detection_rate_ci95", dr)):
        np.testing.assert_allclose(fast[key], np.quantile(ref, [0.025, 0.975]), atol=0.006, err_msg=key)
    assert fast["pass_fraction"]["detection_rate"] == pytest.approx(np.mean(np.asarray(dr) >= 0.6), abs=0.03)
    assert fast["pass_fraction"]["fpr"] == pytest.approx(np.mean(np.asarray(fpr) <= 0.03), abs=0.03)


def test_threshold_bootstrap_param_validation():
    PerformanceModule.validate_params({**PerformanceModule.default_params, "threshold_bootstrap_resamples": 0})
    PerformanceModule.validate_params({k: v for k, v in PerformanceModule.default_params.items()
                                       if k != "threshold_bootstrap_resamples"})
    for bad in (-1, 1.5, "100", True):
        with pytest.raises(ConfigError, match="threshold_bootstrap_resamples"):
            PerformanceModule.validate_params({**PerformanceModule.default_params, "threshold_bootstrap_resamples": bad})


def test_threshold_bootstrap_timeout_does_not_error_the_gate(monkeypatch):
    import malvalid.modules.performance as perf

    ok, _, _ = _calibrated_case(0.5)

    def expire(*a, **k):
        raise ModuleTimeout("time limit")

    monkeypatch.setattr(perf, "threshold_bootstrap", expire)
    r, _, _ = _calibrated_case(0.5)
    tb = r.details["threshold_bootstrap"]
    assert tb["applicable"] and tb["ran"] is False and "ModuleTimeout" in tb["reason"]
    assert r.status is ok.status and r.gate_outcome is ok.gate_outcome and r.score == ok.score
    assert r.metrics == {k: v for k, v in ok.metrics.items() if k in r.metrics}
    assert "threshold_ci95" not in r.metrics
