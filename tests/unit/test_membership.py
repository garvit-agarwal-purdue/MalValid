"""Unit tests for M4 membership inference (toy corpus + LightGBM from malvalid.testing)."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
import yaml

from malvalid import registry
from malvalid.config import GateConfig, default_config_text
from malvalid.core import GateOutcome, Requirement, Status, unmet_requirements
from malvalid.modules.membership import (
    DESIGN_NEAREST_TIME,
    DESIGN_TIME_MATCHED,
    MembershipInferenceModule,
    TooFewSamples,
    attack_metrics,
    log_confidence,
    select_membership_sample,
)
from malvalid.testing import InProcessModel, make_context, make_toy_corpus, train_toy_lgbm

REGULARIZED = {"num_leaves": 7, "min_data_in_leaf": 50, "lambda_l2": 10.0, "learning_rate": 0.05}
OVERFIT = {"num_leaves": 128, "min_data_in_leaf": 1, "lambda_l2": 0.0, "learning_rate": 0.3}
CFG = GateConfig.model_validate({"runtime": {"threads": 2}})


@pytest.fixture(scope="module")
def corpus():
    # A harder toy task (no string signal, strong drift) so that memorization is measurable.
    return make_toy_corpus(n=8000, seed=0, string_signal=0.0, drift=0.9)


@pytest.fixture(scope="module")
def members(corpus):
    """Members = a random half of the train split."""
    tr = corpus.indices(splits=("train",))
    pick = np.sort(np.random.default_rng(123).choice(tr, tr.size // 2, replace=False))
    mask = np.zeros(corpus.n, dtype=bool)
    mask[pick] = True
    return mask, frozenset(corpus.sha256[pick].tolist())


def _model(booster):
    return InProcessModel.from_lightgbm(booster, threshold=0.5, feature_version="toy_v1")


@pytest.fixture(scope="module")
def regularized_model(corpus, members):
    return _model(train_toy_lgbm(corpus, rounds=40, params=REGULARIZED, row_filter=members[0]))


@pytest.fixture(scope="module")
def overfit_model(corpus, members):
    return _model(train_toy_lgbm(corpus, rounds=300, params=OVERFIT, row_filter=members[0]))


def _run(model, corpus, hashes, params=None, seed=0):
    ctx = make_context(MembershipInferenceModule, model=model, corpus=corpus, training_hashes=hashes,
                       params=params, seed=seed)
    ctx.config = CFG
    return MembershipInferenceModule().run(ctx), ctx


def test_registered_and_params_cover_default_gate():
    assert registry.get_module("membership_inf") is MembershipInferenceModule
    yaml_keys = set(yaml.safe_load(default_config_text())["modules"]["membership_inf"]) - {"enabled", "gate"}
    assert yaml_keys <= set(MembershipInferenceModule.default_params)
    assert MembershipInferenceModule.requires == (Requirement.FEATURE_SPACE, Requirement.TRAINING_HASHES)


def test_regularized_model_passes(corpus, members, regularized_model):
    res, ctx = _run(regularized_model, corpus, members[1])
    assert res.status is Status.PASS, res.finding
    assert res.gate_outcome is GateOutcome.PASSED
    m = res.metrics
    assert m["advantage"] <= 0.10
    assert m["sampling_design"] == DESIGN_TIME_MATCHED and m["advantage_is_upper_bound"] is False
    assert set(res.details["attacks"]) == {"loss_threshold", "art_rf", "art_gb"}
    for k in ("loss_threshold", "art_rf", "art_gb"):
        for suffix in ("advantage", "best_advantage", "auroc", "tpr_at_fpr_0.01"):
            assert f"{k}_{suffix}" in m
    assert m["advantage"] == max(m[f"{k}_advantage"] for k in ("loss_threshold", "art_rf", "art_gb"))
    assert res.score is not None and res.score >= 0.75
    chk = res.checks[0]
    assert (chk.op, chk.threshold, chk.ideal, chk.floor) == ("<=", 0.10, pytest.approx(0.01), pytest.approx(0.30))
    assert set(res.artifacts) == {"membership_inf.attack_roc", "membership_inf.member_score_hist"}
    assert any("What this leakage would expose" in n for n in res.notes)
    json.dumps(res.to_dict(), allow_nan=False)
    json.dumps(ctx.artifacts.to_dict(), allow_nan=False)


def test_overfit_model_fails_gate(corpus, members, overfit_model):
    res, _ = _run(overfit_model, corpus, members[1])
    assert res.status is Status.WARN  # warn gate: reported, does not block
    assert res.gate_outcome is GateOutcome.FAILED
    assert res.metrics["advantage"] > 0.10
    assert res.score is not None and res.score < 0.75
    assert "above the 0.10 limit" in res.finding
    assert res.metrics["accuracy_gap"] > 0  # memorized members, worse on unseen files


def test_skip_without_manifest(corpus, regularized_model):
    ctx = make_context(MembershipInferenceModule, model=regularized_model, corpus=corpus,
                       training_hashes=None)
    assert Requirement.TRAINING_HASHES in unmet_requirements(MembershipInferenceModule, ctx.capabilities)
    res = MembershipInferenceModule().run(ctx)
    assert res.status is Status.SKIPPED and res.gate_outcome is GateOutcome.NOT_EVALUATED
    assert "training manifest" in res.skip_reason
    assert res.score is None


def test_skip_too_few_members(corpus, members, regularized_model):
    few = frozenset(sorted(members[1])[:50])
    res, _ = _run(regularized_model, corpus, few)
    assert res.status is Status.SKIPPED
    assert res.skip_reason.startswith("training manifest has 50 members in the canonical corpus")
    # A manifest whose hashes are not in the corpus at all is the same skip path.
    res2, _ = _run(regularized_model, corpus, frozenset({"0" * 64, "f" * 64}))
    assert res2.status is Status.SKIPPED and "has 0 members" in res2.skip_reason


def test_no_time_overlap_is_reported_as_upper_bound(corpus):
    # Members = the whole train split => every non-member is from later months.
    all_train = frozenset(corpus.sha256[corpus.indices(splits=("train",))].tolist())
    booster = train_toy_lgbm(corpus, rounds=40, params=REGULARIZED)
    res, _ = _run(_model(booster), corpus, all_train)
    assert res.status is not Status.SKIPPED
    assert res.metrics["sampling_design"] == DESIGN_NEAREST_TIME
    assert res.metrics["advantage_is_upper_bound"] is True
    assert any(n.startswith("UPPER BOUND") for n in res.notes)
    assert "upper bound" in res.finding
    assert res.details["sampling"]["median_pair_time_gap_days"] is not None
    # The max_per_side cap on the nearest-time path is recorded too.
    capped, _ = _run(_model(booster), corpus, all_train, params={"max_per_side": 300})
    assert capped.metrics["n_members_used"] == 300
    assert capped.details["sampling"]["pairs_before_cap"] > 300
    assert any("closest in time" in n for n in capped.notes)


def test_sampling_is_matched_by_month_and_label(corpus, members):
    s = select_membership_sample(corpus, members[1], rng=np.random.default_rng(0), min_members=200,
                                 max_per_side=None)
    assert s.design == DESIGN_TIME_MATCHED
    assert s.member_idx.size == s.nonmember_idx.size >= 200
    assert not set(s.member_idx.tolist()) & set(s.nonmember_idx.tolist())
    assert set(corpus.sha256[s.member_idx].tolist()) <= members[1]
    assert not set(corpus.sha256[s.nonmember_idx].tolist()) & members[1]

    def strata(idx):
        return Counter(zip(corpus.timestamp[idx].astype("datetime64[M]").tolist(), corpus.label[idx].tolist()))

    assert strata(s.member_idx) == strata(s.nonmember_idx)
    # Pairs are matched individually too.
    assert np.all(corpus.label[s.member_idx] == corpus.label[s.nonmember_idx])
    with pytest.raises(TooFewSamples):
        select_membership_sample(corpus, members[1], rng=np.random.default_rng(0), min_members=10**6,
                                 max_per_side=None)


def test_max_per_side_subsample_is_recorded(corpus, members, regularized_model):
    res, _ = _run(regularized_model, corpus, members[1], params={"max_per_side": 300})
    assert res.metrics["n_members_used"] == res.metrics["n_nonmembers_used"] == 300
    assert res.details["sampling"]["pairs_before_cap"] > 300
    assert any(n.startswith("Subsampled 300 of") for n in res.notes)
    assert res.metrics["n_attack_test_pairs"] == 150


def test_label_only_matching_when_time_matching_disabled(corpus, members, regularized_model):
    res, _ = _run(regularized_model, corpus, members[1], params={"time_matched": False})
    assert res.metrics["sampling_design"] == "label_matched"
    assert any("matched by label only" in n for n in res.notes)


def test_deterministic_given_seed(corpus, members, regularized_model):
    a, _ = _run(regularized_model, corpus, members[1], seed=7)
    b, _ = _run(regularized_model, corpus, members[1], seed=7)
    assert a.metrics == b.metrics


def test_invalid_params_raise(corpus, members, regularized_model):
    with pytest.raises(ValueError, match="test_fraction"):
        _run(regularized_model, corpus, members[1], params={"test_fraction": 1.0})


def test_attack_metric_helpers():
    rng = np.random.default_rng(0)
    m, n = rng.normal(1.0, 1.0, 2000), rng.normal(0.0, 1.0, 2000)
    r = attack_metrics(m, n, decision=lambda s: s >= 0.5)
    assert 0.3 < r["advantage"] < 0.45 and r["best_advantage"] >= r["advantage"] - 1e-9
    assert 0.7 < r["auroc"] < 0.8 and 0.0 <= r["tpr_at_fpr_0.01"] < 0.2
    same = attack_metrics(n, rng.normal(0.0, 1.0, 2000), decision=lambda s: s >= 0.0)
    assert abs(same["advantage"]) < 0.06 and abs(same["auroc"] - 0.5) < 0.04
    lc = log_confidence(np.array([1.0, 0.0, 0.9, 0.1]), np.array([0, 1, 1, 0]))
    assert np.all(np.isfinite(lc)) and lc[2] == pytest.approx(np.log(0.9))


def test_challenge_rows_never_used_as_nonmembers(corpus):
    all_train = frozenset(corpus.sha256[corpus.indices(splits=("train",))].tolist())
    s = select_membership_sample(corpus, all_train, rng=np.random.default_rng(0), min_members=200,
                                 max_per_side=None)
    assert s.design == DESIGN_NEAREST_TIME and s.upper_bound
    assert s.info["excluded_challenge_rows"] > 0
    assert not np.any(corpus.split[s.nonmember_idx] == "challenge")


def test_calibration_rows_join_nonmember_pool_only_when_requested(corpus, members):
    import dataclasses

    # Calibration rows must share the members' months to be drawn by the time-matched pairing, so take
    # benign non-member train rows (a test-period row would be a candidate but never be paired).
    tr_idx = corpus.indices(splits=("train",), label=0, exclude_hashes=members[1])
    cal_idx = tr_idx[: max(tr_idx.size // 3, 50)]
    split = corpus.split.copy()
    split[cal_idx] = "calibration"
    roles = {**dict(corpus.manifest.get("roles") or {}), "pool": sorted(set(corpus.split.tolist()))}
    view = dataclasses.replace(corpus, split=split, manifest={**corpus.manifest, "roles": roles})
    base = select_membership_sample(view, members[1], rng=np.random.default_rng(0), min_members=50,
                                    max_per_side=None)
    ext = select_membership_sample(view, members[1], rng=np.random.default_rng(0), min_members=50,
                                   max_per_side=None, extra_nonmember_splits=("calibration",))
    assert not np.any(view.split[base.nonmember_idx] == "calibration")
    assert ext.n_nonmember_candidates == base.n_nonmember_candidates + cal_idx.size
    assert np.any(view.split[ext.nonmember_idx] == "calibration")
    # still never a training member
    assert not set(view.sha256[ext.nonmember_idx].tolist()) & members[1]
