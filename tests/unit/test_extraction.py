"""Unit tests for M6 model-extraction susceptibility (toy corpus + LightGBM from malvalid.testing)."""

from __future__ import annotations

import copy
import json
import time

import numpy as np
import pytest
import yaml

from malvalid import registry
from malvalid.config import GateConfig, default_config_text
from malvalid.core import GateOutcome, Requirement, Status
from malvalid.modules import extraction as ext
from malvalid.modules.extraction import METHOD_ART, METHOD_DIRECT, ExtractionModule
from malvalid.testing import InProcessModel, make_context, make_toy_corpus, train_toy_lgbm

CFG = GateConfig.model_validate({"runtime": {"threads": 2}})
BUDGETS = {"query_budgets": [20, 100, 400, 1600], "fidelity_budget": 400}


@pytest.fixture(scope="module")
def corpus():
    return make_toy_corpus(n=4000, seed=0)


@pytest.fixture(scope="module")
def model(corpus):
    return InProcessModel.from_lightgbm(train_toy_lgbm(corpus), threshold=0.5, feature_version="toy_v1")


def _run(model, corpus, params=None, seed=0):
    ctx = make_context(ExtractionModule, model=model, corpus=corpus, params=params, seed=seed)
    ctx.config = CFG
    return ExtractionModule().run(ctx), ctx


def test_registered_and_params_cover_default_gate():
    assert registry.get_module("extraction") is ExtractionModule
    cfg = yaml.safe_load(default_config_text())["modules"]["extraction"]
    assert cfg["enabled"] is False  # off by default
    assert set(cfg) - {"enabled", "gate"} <= set(ExtractionModule.default_params)
    assert ExtractionModule.requires == (Requirement.QUERY_ONLY,)


def test_fidelity_increases_with_budget(corpus, model):
    t0 = time.monotonic()
    res, ctx = _run(model, corpus, BUDGETS)
    assert time.monotonic() - t0 < 30
    m = res.metrics
    assert m["budgets"] == [20, 100, 400, 1600]
    f = m["fidelities"]
    assert f[-1] > f[0] + 0.03
    assert all(b >= a - 0.02 for a, b in zip(f, f[1:])), f
    assert m["method"] == METHOD_ART
    assert m["fidelity"] == pytest.approx(f[2]) and m["fidelity_budget_effective"] == 400
    assert m["n_query_pool"] == corpus.indices(labeled_only=False).size - corpus.indices(
        role="eval", labeled_only=False).size
    chk = res.checks[0]
    assert (chk.op, chk.threshold, chk.ideal, chk.floor) == ("<=", 0.95, 0.80, 0.995)
    art = ctx.artifacts.get("extraction.fidelity_vs_queries")
    assert art["xscale"] == "log" and len(art["series"][0]["x"]) == 4
    # ART re-queries per budget; plus one pass over the eval rows.
    assert res.details["total_model_queries"] == sum(m["budgets"]) + m["n_eval"]
    assert any("queryable service" in n for n in res.notes)
    assert "FunctionallyEquivalentExtraction" in res.details["art_applicability"]
    json.dumps(res.to_dict(), allow_nan=False)
    json.dumps(ctx.artifacts.to_dict(), allow_nan=False)


def test_gate_outcomes(corpus, model):
    strict, _ = _run(model, corpus, {**BUDGETS, "max_fidelity": 0.5})
    assert strict.status is Status.WARN and strict.gate_outcome is GateOutcome.FAILED
    assert strict.score is not None and strict.score < 0.75
    assert "cloned cheaply" in strict.finding
    lax, _ = _run(model, corpus, {**BUDGETS, "max_fidelity": 0.999})
    assert lax.status is Status.PASS and lax.gate_outcome is GateOutcome.PASSED
    assert lax.score is not None and lax.score >= 0.75


def test_skip_without_corpus(model):
    res, _ = _run(model, None)
    assert res.status is Status.SKIPPED and res.gate_outcome is GateOutcome.NOT_EVALUATED
    assert res.skip_reason == "no query distribution: no canonical corpus in the model's feature_version is loaded"


def test_skip_without_corpus_cites_the_runner_corpus_error(model):
    """Regression (xcomp-generic-skip-reasons-contradict-runner): the runner's specific corpus error
    (e.g. an ember_v2 corpus for an ember_v3 model) reaches M6's skip reason."""
    ctx = make_context(ExtractionModule, model=model, corpus=None)
    ctx.config = CFG
    why = ("no canonical corpus in the model's feature_version is loaded (corpus 'ember_v2_2018' is in "
           "feature space 'ember_v2' but the model declares feature_version 'ember_v3')")
    ctx.extras["unmet_reasons"] = {"feature_space": why}
    res = ExtractionModule().run(ctx)
    assert res.status is Status.SKIPPED and res.skip_reason == f"no query distribution: {why}"


def test_budget_beyond_pool_is_not_a_pass(corpus, model):
    res, _ = _run(model, corpus, {"query_budgets": [100, 1000], "fidelity_budget": 10**6,
                                  "max_fidelity": 0.999})
    pool = res.metrics["n_query_pool"]
    assert res.metrics["fidelity_budget_effective"] == pool
    assert res.checks[0].value is None and res.gate_outcome is GateOutcome.NOT_EVALUATED
    assert res.status is Status.WARN
    assert any("lower bound" in n for n in res.notes)
    # ...but if fidelity already exceeds the limit at the smaller budget, that is a real failure.
    res2, _ = _run(model, corpus, {"query_budgets": [100], "fidelity_budget": 10**6, "max_fidelity": 0.5})
    assert res2.gate_outcome is GateOutcome.FAILED


def test_direct_fallback_without_art(corpus, model, monkeypatch):
    def boom():
        raise ImportError("ART not installed")

    monkeypatch.setattr(ext, "_import_art", boom)
    res, _ = _run(model, corpus, BUDGETS)
    assert res.metrics["method"] == METHOD_DIRECT
    assert any("direct query-and-fit" in n for n in res.notes)
    f = res.metrics["fidelities"]
    assert f[-1] > f[0]
    # Direct mode queries the largest budget once.
    assert res.details["total_model_queries"] == max(res.metrics["budgets"]) + res.metrics["n_eval"]


@pytest.mark.parametrize("kind", ["random_forest", "logistic_regression"])
def test_other_surrogates(corpus, model, kind):
    res, _ = _run(model, corpus, {"query_budgets": [100, 800], "fidelity_budget": 800, "surrogate": kind})
    assert res.metrics["surrogate"] == kind and res.status is not Status.ERROR
    assert 0.5 < res.metrics["fidelity"] <= 1.0


def test_unknown_surrogate_raises(corpus, model):
    with pytest.raises(ValueError, match="surrogate"):
        _run(model, corpus, {"surrogate": "mlp"})


def test_deterministic_given_seed(corpus, model):
    a, _ = _run(model, corpus, BUDGETS, seed=3)
    b, _ = _run(model, corpus, BUDGETS, seed=3)
    assert a.metrics == b.metrics


def test_constant_labels_give_constant_surrogate():
    s = ext.fit_surrogate("lightgbm", np.zeros((10, 3), np.float32), np.ones(10), seed=0, threads=1)
    assert np.all(s.predict_hard(np.zeros((4, 3), np.float32)) == 1)


def _with_roles(corpus, roles):
    c = copy.copy(corpus)
    c.manifest = {**corpus.manifest, "roles": roles}
    return c


def test_pool_equal_to_eval_reserves_held_out_rows(corpus, model):
    # Like ember_v3_2024: pool and eval are the same split, so no pool row lies outside eval.
    c = _with_roles(corpus, {"eval": ["test"], "pool": ["test"]})
    res, _ = _run(model, c, {"query_budgets": [50, 400], "fidelity_budget": 400, "n_eval": 300})
    assert res.status is not Status.SKIPPED, res.skip_reason
    m, d = res.metrics, res.details["query_pool"]
    assert m["query_pool_source"] == ext.QUERY_POOL_EVAL_REMAINDER == d["source"]
    n_test = c.indices(splits=["test"], labeled_only=False).size
    assert m["n_eval"] == 300 and m["n_query_pool"] == n_test - 300
    assert any("reserved first" in n for n in res.notes)


def test_query_rows_disjoint_from_held_out_rows_and_hashes(corpus):
    c = _with_roles(corpus, {"eval": ["test"], "pool": ["test"]})
    ev, _, q, info = ext.select_query_and_eval_rows(c, training_hashes=None, n_eval=None,
                                                     rng=np.random.default_rng(0))
    assert ev.size == c.indices(role="eval", labeled_only=False).size // 2  # capped at half
    assert np.intersect1d(ev, q).size == 0
    # A duplicated sha256 (same file under two rows) must not leak into the query pool.
    c2 = copy.copy(corpus)
    sha = corpus.sha256.copy()
    train_rows = corpus.indices(splits=["train"], labeled_only=False)
    test_rows = corpus.indices(splits=["test"], labeled_only=False)
    sha[train_rows[:5]] = sha[test_rows[:5]]
    c2.sha256 = sha
    ev2, _, q2, info2 = ext.select_query_and_eval_rows(c2, training_hashes=None, n_eval=None,
                                                        rng=np.random.default_rng(0))
    assert info2["source"] == ext.QUERY_POOL_NON_EVAL and info2["excluded_duplicate_sha256"] == 5
    assert not np.isin(c2.sha256[q2], c2.sha256[ev2]).any()


def test_skip_when_no_row_can_be_held_out(corpus, model):
    c = _with_roles(corpus, {"eval": ["test"], "pool": ["test"]})
    tiny = copy.copy(c)
    keep = c.indices(splits=["test"], labeled_only=False)[:1]
    mask = np.zeros(c.n, bool)
    mask[keep] = True
    tiny.split = np.where(mask, c.split, np.array("other", dtype=c.split.dtype))
    res, _ = _run(model, tiny, {"query_budgets": [10], "fidelity_budget": 10})
    assert res.status is Status.SKIPPED and "held-out eval rows" in res.skip_reason
