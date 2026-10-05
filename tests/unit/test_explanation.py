"""Unit tests for M7 explanation & spurious-feature reliance (malvalid.modules.explanation)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from malvalid.core import ConfigError, GateOutcome, Requirement, Status, unmet_requirements
from malvalid.loaders.trees import from_lightgbm_dump
from malvalid.modules import explanation as expl
from malvalid.modules.explanation import ExplanationModule, concentration, controllable_share, treeshap_cost
from malvalid.schemas.base import Controllability
from malvalid.scoring import PASS_LINE
from malvalid.testing import InProcessModel, ToySchema, make_context, make_toy_corpus, train_toy_lgbm


# --------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def string_corpus():
    """Toy corpus whose dominant class signal is one APPEND_ONLY string-count feature (strings[0])."""
    return make_toy_corpus(3000, seed=0, string_signal=3.0)


@pytest.fixture(scope="module")
def string_booster(string_corpus):
    return train_toy_lgbm(string_corpus)


@pytest.fixture(scope="module")
def fixed_corpus():
    """Toy corpus whose class signal lives only in FIXED code features (spread over code[0..7])."""
    c = make_toy_corpus(3000, seed=0, string_signal=0.0)
    y = c.label == 1
    X = c.X.astype(np.float64)
    X[y, 16] -= 1.0  # remove the CONTROLLABLE header signal
    X[:, 1:8] = np.random.default_rng(5).dirichlet(np.ones(7) * 2.0, size=X.shape[0])  # and the histogram one
    X[y, 26:32] += 1.2
    c.X[:] = X.astype(np.float32)
    return c


@pytest.fixture(scope="module")
def fixed_booster(fixed_corpus):
    return train_toy_lgbm(fixed_corpus)


def _model(booster, *, trees: bool = True) -> InProcessModel:
    m = InProcessModel.from_lightgbm(booster, threshold=0.5, feature_version="toy_v1")
    if trees:
        return m

    class _NoTrees:  # wraps predict_proba only — no native_model / tree_ensemble()
        def predict_proba(self, X):
            return booster.predict(X)

        def predict(self, X):
            return (booster.predict(X) >= 0.5).astype(np.int8)

    return InProcessModel(_NoTrees(), m.declarations)


def _run(model, corpus, schema=None, **params):
    ctx = make_context(ExplanationModule, model=model, corpus=corpus, schema=schema, params=params)
    return ExplanationModule().run(ctx), ctx


# --------------------------------------------------------------------------------------------------
# Known-bad fixture and pass case
# --------------------------------------------------------------------------------------------------


def test_string_keyed_model_trips_m7(string_booster, string_corpus):
    res, ctx = _run(_model(string_booster), string_corpus, shap_samples=400)
    assert res.status is Status.WARN and res.gate_outcome is GateOutcome.FAILED
    m = res.metrics
    assert m["attribution_source"] == "tree_shap"
    assert m["top_feature"] == "strings[0]"
    assert m["top_feature_controllability"] == Controllability.APPEND_ONLY.value
    assert m["top_feature_share"] > 0.30
    assert m["controllable_share"] > 0.50
    assert m["share_append_only"] > 0.8
    checks = {c.name: c for c in res.checks}
    assert checks["controllable_share"].passed is False
    assert checks["top_feature_share"].passed is False
    assert res.score is not None and res.score < PASS_LINE
    top = res.details["top_features"][0]
    assert top["feature"] == "strings[0]" and top["group"] == "strings" and top["attacker_controllable"] is True
    assert "strings[0]" in res.finding and "append_only" in res.finding
    keys = set(ctx.artifacts.keys_for("explanation"))
    assert {"explanation.top_features", "explanation.group_importance", "explanation.top_features_table"} <= keys
    assert ctx.artifacts.get("explanation.top_features")["kind"] == "bar"
    assert ctx.artifacts.get("explanation.top_features_table")["type"] == "table"
    json.dumps(res.to_dict(), allow_nan=False)
    json.dumps(ctx.artifacts.to_dict(), allow_nan=False)


def test_model_relying_on_fixed_code_features_passes(fixed_booster, fixed_corpus):
    res, _ = _run(_model(fixed_booster), fixed_corpus, shap_samples=400)
    assert res.status is Status.PASS and res.gate_outcome is GateOutcome.PASSED
    m = res.metrics
    assert m["top_feature_controllability"] == Controllability.FIXED.value
    assert m["top_feature"].startswith("code[")
    assert m["controllable_share"] < 0.15
    assert m["top_feature_share"] < 0.30
    assert m["share_fixed"] > 0.8
    assert m["effective_n_features"] > 4
    assert res.score is not None and res.score >= PASS_LINE


def test_shares_are_consistent(string_booster, string_corpus):
    res, _ = _run(_model(string_booster), string_corpus, shap_samples=200, top_k=5)
    d = res.details
    assert sum(d["by_group"].values()) == pytest.approx(1.0)
    assert sum(d["by_controllability"].values()) == pytest.approx(1.0)
    assert sum(d["by_group_gain"].values()) == pytest.approx(1.0)
    assert len(d["top_features"]) <= 5 and res.metrics["top_k"] == 5
    shares = [r["share"] for r in d["top_features"]]
    assert shares == sorted(shares, reverse=True)
    assert res.metrics["top_k_share"] == pytest.approx(sum(shares))
    assert res.metrics["shap_samples_used"] == 200
    s = d["attribution"]["sample"]
    assert s["n_benign"] == s["n_malicious"] == 100  # class-balanced eval sample
    assert res.metrics["gain_top_feature"] == "strings[0]"


def test_deterministic_given_seed(fixed_booster, fixed_corpus):
    a, _ = _run(_model(fixed_booster), fixed_corpus, shap_samples=100)
    b, _ = _run(_model(fixed_booster), fixed_corpus, shap_samples=100)
    assert a.metrics == b.metrics


# --------------------------------------------------------------------------------------------------
# Degrade paths
# --------------------------------------------------------------------------------------------------


def test_non_tree_model_uses_model_agnostic_shap(string_booster, string_corpus):
    model = _model(string_booster, trees=False)
    res, ctx = _run(model, string_corpus, shap_samples=24)
    assert Requirement.TREE_ACCESS not in ctx.capabilities
    m = res.metrics
    assert m["attribution_source"] == "permutation_shap"
    assert 0 < m["shap_samples_used"] <= 24
    assert m["top_feature"] == "strings[0]"
    assert res.gate_outcome is GateOutcome.FAILED
    assert "gain_top_feature" not in m  # no trees, no gain
    info = res.details["attribution"]
    assert info["output_space"].startswith("probability")
    assert info["model_rows_scored"] > 0
    assert any("model-agnostic permutation SHAP" in n for n in res.notes)
    assert model.query_count >= info["model_rows_scored"]


def test_rejected_trees_note_cites_the_runner_note(fixed_booster, fixed_corpus):
    """Regression (xcomp-generic-skip-reasons-contradict-runner): M7's agnostic-SHAP note cites why tree
    access is unavailable (here: the runner rejected the dump) instead of 'does not expose its trees'."""
    ctx = make_context(ExplanationModule, model=_model(fixed_booster, trees=False), corpus=fixed_corpus,
                       params={"shap_samples": 12})
    ctx.extras["unmet_reasons"] = {"tree_access": 'the tree dump does not reproduce predict_proba (max |Δ| = 0.25 > 0.0001 on 2000 corpus rows); tree access disabled so tree-based analyses cannot misdescribe the deployed model'}
    res = ExplanationModule().run(ctx)
    note = next(n for n in res.notes if "model-agnostic permutation SHAP" in n)
    assert "does not reproduce predict_proba" in note and "does not expose its trees" not in note


def test_non_tree_model_is_deterministic(fixed_booster, fixed_corpus):
    a, _ = _run(_model(fixed_booster, trees=False), fixed_corpus, shap_samples=12)
    b, _ = _run(_model(fixed_booster, trees=False), fixed_corpus, shap_samples=12)
    assert a.metrics == b.metrics
    assert a.status is Status.PASS


def test_trees_without_corpus_fall_back_to_gain(string_booster):
    ctx = make_context(ExplanationModule, model=_model(string_booster), corpus=None, schema=ToySchema())
    assert Requirement.FEATURE_SPACE not in ctx.capabilities
    res = ExplanationModule().run(ctx)
    assert res.metrics["attribution_source"] == "tree_gain"
    assert res.metrics["shap_samples_used"] == 0
    assert res.metrics["top_feature"] == "strings[0]"
    assert res.gate_outcome is GateOutcome.FAILED
    assert any("split gain only" in n for n in res.notes)


def test_no_trees_no_corpus_is_skipped(string_booster):
    model = _model(string_booster, trees=False)
    ctx = make_context(ExplanationModule, model=model, corpus=None, schema=ToySchema())
    assert unmet_requirements(ExplanationModule, ctx.capabilities)  # the runner skips this
    res = ExplanationModule().run(ctx)
    assert res.status is Status.SKIPPED


def test_treeshap_budget_cap_is_recorded(monkeypatch, fixed_booster, fixed_corpus):
    monkeypatch.setattr(expl, "SHAP_TIME_BUDGET_S", 1e-9)
    res, _ = _run(_model(fixed_booster), fixed_corpus, shap_samples=500)
    assert res.metrics["shap_samples_requested"] == 500
    assert res.metrics["shap_samples_used"] == expl.SHAP_MIN_SAMPLES
    assert res.details["attribution"]["budget_cap"] == 0
    assert any("instead of shap_samples = 500" in n for n in res.notes)


def test_uniform_cover_uses_interventional_treeshap(string_booster, string_corpus):
    te = from_lightgbm_dump(string_booster.dump_model())
    for t in te.trees:
        t.cover[:] = 1.0
    te.meta["cover"] = "uniform"

    class _Det:
        def predict_proba(self, X):
            return string_booster.predict(X)

        def predict(self, X):
            return (string_booster.predict(X) >= 0.5).astype(np.int8)

        def tree_ensemble(self):
            return te

    model = InProcessModel(_Det(), _model(string_booster).declarations)
    res, _ = _run(model, string_corpus, shap_samples=40)
    assert res.metrics["attribution_source"] == "tree_shap_interventional"
    assert res.details["attribution"]["background_rows"] > 0
    assert res.metrics["top_feature"] == "strings[0]"


# --------------------------------------------------------------------------------------------------
# Helpers and params
# --------------------------------------------------------------------------------------------------


def test_concentration_and_controllable_share():
    s = np.array([0.5, 0.25, 0.25, 0.0])
    c = concentration(s, top_k=2)
    assert c["top_feature_share"] == 0.5
    assert c["top_k_share"] == 0.75
    assert c["effective_n_features"] == pytest.approx(1 / (0.25 + 0.0625 + 0.0625))
    C = Controllability
    lv = [C.CONTROLLABLE, C.APPEND_ONLY, C.DERIVED, C.FIXED]
    assert controllable_share(s, lv) == pytest.approx(0.5 * 1.0 + 0.25 * 0.75)  # DERIVED/FIXED excluded
    assert np.isnan(concentration(np.zeros(3), 2)["top_feature_share"])


def test_treeshap_cost_grows_with_ensemble(string_corpus):
    small = from_lightgbm_dump(train_toy_lgbm(string_corpus, rounds=10).dump_model())
    big = from_lightgbm_dump(train_toy_lgbm(string_corpus, rounds=40, params={"num_leaves": 31}).dump_model())
    assert 0 < treeshap_cost(small) < treeshap_cost(big)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"top_k": 0}, "top_k"),
        ({"shap_samples": 0}, "shap_samples"),
        ({"max_controllable_share": 1.5}, "max_controllable_share"),
        ({"max_top_feature_share": "0.3"}, "max_top_feature_share"),
    ],
)
def test_bad_params(fixed_booster, fixed_corpus, params, match):
    with pytest.raises(ConfigError, match=match):
        _run(_model(fixed_booster), fixed_corpus, **params)


def test_module_contract():
    from malvalid import registry
    from malvalid.config import load_config, validate_against_registry

    assert registry.get_module("explanation") is ExplanationModule
    assert set(ExplanationModule.requires_any) == {Requirement.TREE_ACCESS, Requirement.FEATURE_SPACE}
    assert set(ExplanationModule.default_params) == {"top_k", "shap_samples", "max_controllable_share",
                                                     "max_top_feature_share"}
    cfg = load_config(None, overrides={"modules": {"explanation": {"top_k": 10, "shap_samples": 500}}})
    validate_against_registry(cfg)


class _ReasonedToySchema(ToySchema):
    """Toy schema that also exposes per-feature controllability reasons (like ember_v2)."""

    def controllability_reasons(self):
        return [f"why {n}" for n in self.feature_names()]


def test_top_features_table_carries_schema_controllability_reasons(string_booster, string_corpus):
    res, ctx = _run(_model(string_booster), string_corpus, schema=_ReasonedToySchema(), shap_samples=200)
    top = res.details["top_features"][0]
    assert top["controllability_reason"] == f"why {top['feature']}"
    tbl = ctx.artifacts.get("explanation.top_features_table")
    assert tbl["columns"][-1] == "why (schema)"
    assert tbl["rows"][0][-1] == f"why {top['feature']}"


def test_top_features_table_without_reasons(string_booster, string_corpus):
    res, ctx = _run(_model(string_booster), string_corpus, shap_samples=200)
    assert res.details["top_features"][0]["controllability_reason"] is None
    assert "why (schema)" not in ctx.artifacts.get("explanation.top_features_table")["columns"]
    assert expl.controllability_reasons(object(), 3) is None


def test_categorical_model_treeshap_matches_lightgbm_contrib():
    """TreeSHAP on the loader's categorical threshold-chain rewrite equals LightGBM's own
    pred_contrib (the evenly shared cover leaves the cover-weighted expectations unchanged)."""
    import lightgbm as lgb
    import shap

    from malvalid.loaders.lightgbm_loader import LightGBMLoader

    corpus = make_toy_corpus(4000, seed=0)
    tr = corpus.indices(splits=("train",))
    X = np.asarray(corpus.take(tr), dtype=np.float64)
    y = corpus.label[tr]
    X[:, 17] = np.where(y == 1, np.random.default_rng(0).integers(0, 5, tr.size), np.random.default_rng(1).integers(3, 9, tr.size))
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 8, "verbose": -1, "num_threads": 1, "seed": 0, "min_data_per_group": 5},
        lgb.Dataset(X, y, categorical_feature=[17]), num_boost_round=15,
    )
    loader = LightGBMLoader()
    te = loader.tree_ensemble(booster)
    assert te.meta.get("categorical_splits_expanded", 0) > 0
    sv = np.asarray(shap.TreeExplainer(te.to_shap_model()).shap_values(X[:40], check_additivity=False))
    np.testing.assert_allclose(sv, booster.predict(X[:40], pred_contrib=True)[:, :-1], atol=1e-9)

    class _Det:
        native_model = booster

        def predict_proba(self, X):
            return booster.predict(X)

        def predict(self, X):
            return (booster.predict(X) >= 0.5).astype(np.int8)

        def tree_ensemble(self):
            return te

    res, _ = _run(InProcessModel(_Det(), _model(booster).declarations), corpus, shap_samples=100)
    assert res.metrics["attribution_source"] == "tree_shap"
    assert res.details["attribution"]["categorical_splits_expanded"] > 0


def test_agnostic_chunking_matches_single_call(monkeypatch, fixed_booster, fixed_corpus):
    a, _ = _run(_model(fixed_booster, trees=False), fixed_corpus, shap_samples=12)
    monkeypatch.setattr(expl, "AGNOSTIC_CHUNK", 1000)  # one explainer call for all rows
    b, _ = _run(_model(fixed_booster, trees=False), fixed_corpus, shap_samples=12)
    for k in ("controllable_share", "top_feature_share", "effective_n_features"):
        assert a.metrics[k] == pytest.approx(b.metrics[k], rel=1e-9, abs=1e-12)


def test_agnostic_stops_early_near_deadline(monkeypatch, fixed_booster, fixed_corpus):
    import time as _time

    monkeypatch.setattr(expl, "SHAP_STOP_RESERVE_S", 1e6)  # "almost out of time" after the first chunk
    ctx = make_context(ExplanationModule, model=_model(fixed_booster, trees=False), corpus=fixed_corpus,
                       params={"shap_samples": 16}, deadline=_time.monotonic() + 3600)
    res = ExplanationModule().run(ctx)
    info = res.details["attribution"]
    assert info["stopped_early"] is True and info["used"] == expl.AGNOSTIC_CHUNK
    assert any("stopped after" in n for n in res.notes)
    assert res.metrics["shap_samples_used"] == expl.AGNOSTIC_CHUNK
    assert res.gate_outcome is not None


def test_class_interleaved_prefix_is_balanced(fixed_corpus):
    ben = fixed_corpus.eval_indices(0)[:5]
    mal = fixed_corpus.eval_indices(1)[:3]
    out = expl._class_interleaved(fixed_corpus, np.sort(np.concatenate([ben, mal])))
    assert sorted(out.tolist()) == sorted(np.concatenate([ben, mal]).tolist())
    assert fixed_corpus.label[out[:6]].tolist() == [0, 1, 0, 1, 0, 1]
