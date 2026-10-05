"""Unit tests for M5 backdoor / poisoning screening (malvalid.modules.backdoor)."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from malvalid.core import ConfigError, GateOutcome, Requirement, Status
from malvalid.loaders.trees import LEAF, Tree, TreeEnsemble, from_lightgbm_dump
from malvalid.modules.backdoor import (
    ABSENCE_NOTE,
    BackdoorScreenModule,
    parse_triggers,
    scan_check_score,
    static_tree_scan,
)
from malvalid.scoring import PASS_LINE
from malvalid.testing import (
    InProcessModel,
    ToySchema,
    make_context,
    make_toy_corpus,
    train_toy_lgbm,
)

TRIGGER_FEATURE = 21  # header[5]: CONTROLLABLE, carries no class signal in the toy corpus
TRIGGER_NAME = "header[5]"
TRIGGER_VALUE = 7.5  # ~7.5 sigma for a N(0, 1) feature: a rare value


# --------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus():
    return make_toy_corpus(4000, seed=0)


@pytest.fixture(scope="module")
def clean_booster(corpus):
    return train_toy_lgbm(corpus)


@pytest.fixture(scope="module")
def backdoor_booster():
    """Poison 6% of the malicious training rows: stamp the trigger value and relabel them benign."""
    poisoned = make_toy_corpus(4000, seed=0)
    rng = np.random.default_rng(1)
    tr = poisoned.indices(splits=("train",))
    mal = tr[poisoned.label[tr] == 1]
    pois = rng.choice(mal, size=int(0.06 * mal.size), replace=False)
    labels = poisoned.label.copy()
    poisoned.X[pois, TRIGGER_FEATURE] = TRIGGER_VALUE
    labels[pois] = 0
    return train_toy_lgbm(poisoned, labels=labels)


def _model(booster, *, trees: bool = True) -> InProcessModel:
    m = InProcessModel.from_lightgbm(booster, threshold=0.5, feature_version="toy_v1")
    if trees:
        return m

    class _NoTrees:  # predict_proba only: no native_model, no tree_ensemble()
        def predict_proba(self, X):
            return booster.predict(X)

        def predict(self, X):
            return (booster.predict(X) >= 0.5).astype(np.int8)

    return InProcessModel(_NoTrees(), m.declarations)


def _trigger(name: str = "planted", key=TRIGGER_NAME, value: float = TRIGGER_VALUE) -> dict:
    return {"name": name, "features": {key: value}}


def _run(model, corpus, **params):
    ctx = make_context(BackdoorScreenModule, model=model, corpus=corpus, params=params)
    return BackdoorScreenModule().run(ctx), ctx


# --------------------------------------------------------------------------------------------------
# Known-bad fixture: planted trigger
# --------------------------------------------------------------------------------------------------


def test_backdoored_model_is_flagged_by_scan_and_trigger(backdoor_booster, corpus):
    res, ctx = _run(_model(backdoor_booster), corpus, triggers=[_trigger()])
    assert res.status is Status.WARN
    assert res.gate_outcome is GateOutcome.FAILED
    assert res.screening is True
    m = res.metrics
    # static scan: rules gated on the trigger feature are flagged
    assert m["scan_ran"] is True
    assert m["n_flagged_rules"] > 0
    assert m["top_suspicious_feature"] == TRIGGER_NAME
    assert m["top_suspicion"] >= 0.5
    assert TRIGGER_NAME in res.details["scan"]["flagged_features"]
    top = res.details["scan"]["top_rules"][0]
    assert top["dominant_feature"] == TRIGGER_NAME and top["flagged"] is True
    assert any(TRIGGER_NAME in c for c in top["conditions"])
    assert top["leaf_value"] < 0  # routes to a benign leaf
    # operator trigger test: large targeted drop
    assert m["max_trigger_drop"] > 0.5
    assert m["worst_trigger"] == "planted"
    row = res.details["triggers"]["results"][0]
    assert row["flagged"] is True and row["dr_after"] < row["dr_before"]
    # checks and score
    by_name = {c.name: c for c in res.checks}
    assert by_name["n_flagged_rules"].passed is False
    assert by_name["max_trigger_drop"].passed is False
    assert by_name["n_flagged_rules"].score < PASS_LINE
    assert res.score is not None and res.score < PASS_LINE
    # mandatory screening note + ART note, artifacts
    assert ABSENCE_NOTE in res.notes
    assert "absence of findings is not proof" in res.finding
    assert res.details["art_poisoning_detectors"]["applicable"] is False
    keys = set(ctx.artifacts.keys_for("backdoor_screen"))
    assert {"backdoor_screen.suspicious_rules", "backdoor_screen.suspicious_features",
            "backdoor_screen.trigger_tests", "backdoor_screen.feature_suspicion"} <= keys
    json.dumps(res.to_dict(), allow_nan=False)
    json.dumps(ctx.artifacts.to_dict(), allow_nan=False)


def test_clean_model_is_not_flagged(clean_booster, backdoor_booster, corpus):
    clean, _ = _run(_model(clean_booster), corpus, triggers=[_trigger()])
    bad, _ = _run(_model(backdoor_booster), corpus)
    assert clean.metrics["n_flagged_rules"] == 0
    assert clean.metrics["top_suspicion"] < 0.5
    assert clean.metrics["top_suspicion"] < bad.metrics["top_suspicion"] - 0.3
    assert clean.metrics["max_trigger_drop"] < 0.05
    assert clean.status is Status.PASS and clean.gate_outcome is GateOutcome.PASSED
    scan_check = next(c for c in clean.checks if c.name == "n_flagged_rules")
    assert scan_check.passed is True and scan_check.score == 1.0
    assert ABSENCE_NOTE in clean.notes  # carried even when nothing is found
    assert clean.screening is True


def test_no_triggers_scan_only_notes_missing_trigger_test(backdoor_booster, corpus):
    res, _ = _run(_model(backdoor_booster), corpus)
    assert [c.name for c in res.checks] == ["n_flagged_rules"]
    assert res.metrics["n_triggers"] == 0
    assert any("No trigger hypotheses supplied" in n for n in res.notes)


# --------------------------------------------------------------------------------------------------
# Trigger parsing / validation
# --------------------------------------------------------------------------------------------------


def test_trigger_by_name_index_and_numeric_string_agree(backdoor_booster, corpus):
    res, _ = _run(
        _model(backdoor_booster), corpus,
        triggers=[_trigger("by_name"), _trigger("by_index", key=TRIGGER_FEATURE), _trigger("by_str", key=str(TRIGGER_FEATURE))],
    )
    drops = {r["name"]: r["drop"] for r in res.details["triggers"]["results"]}
    assert drops["by_name"] == drops["by_index"] == drops["by_str"]
    assert res.metrics["n_flagged_triggers"] == 3
    assert all(r["features"] == {TRIGGER_NAME: TRIGGER_VALUE} for r in res.details["triggers"]["results"])


@pytest.mark.parametrize(
    "items, match",
    [
        ([{"name": "t", "features": {"headr[5]": 1.0}}], r"unknown feature 'headr\[5\]'.*did you mean 'header\[5\]'"),
        ([{"name": "t", "features": {"no_such_thing": 1.0}}], r"unknown feature 'no_such_thing'.*feature_names"),
        ([{"name": "t", "features": {99: 1.0}}], r"index 99 is outside"),
        ([{"name": "t", "features": {"header[5]": "big"}}], r"must be a number"),
        ([{"name": "t", "features": {"header[5]": float("nan")}}], r"must be finite"),
        ([{"name": "t", "features": {}}], r"non-empty mapping"),
        ([{"name": "t", "feature": {"header[5]": 1.0}}], r"unknown key"),
        (["header[5]"], r"must be a mapping"),
        ([_trigger("a"), _trigger("a")], r"duplicate trigger name"),
        ({"header[5]": 1.0}, r"must be a list"),
    ],
)
def test_parse_triggers_rejects_bad_specs(items, match):
    with pytest.raises(ConfigError, match=match):
        parse_triggers(items, ToySchema())


def test_parse_triggers_accepts_mapping_with_triggers_key():
    out = parse_triggers({"triggers": [{"features": {"code[0]": 1, 5: 2.5}}]}, ToySchema())
    assert out[0].name == "trigger_1"
    assert out[0].features == {24: 1.0, 5: 2.5}
    assert out[0].labels[24] == "code[0]"


def test_unknown_trigger_name_errors_in_run(clean_booster, corpus):
    with pytest.raises(ConfigError, match="did you mean"):
        _run(_model(clean_booster), corpus, triggers=[{"name": "t", "features": {"header[05]": 1.0}}])


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_triggers_path_file(tmp_path, backdoor_booster, corpus, suffix):
    spec = {"triggers": [_trigger("from_file")]}
    p = tmp_path / f"triggers{suffix}"
    if suffix == ".json":
        p.write_text(json.dumps(spec))
    else:
        import yaml

        p.write_text(yaml.safe_dump(spec))
    res, _ = _run(_model(backdoor_booster), corpus, triggers_path=str(p), triggers=[_trigger("inline", key=16, value=0.0)])
    names = [r["name"] for r in res.details["triggers"]["results"]]
    assert names == ["inline", "from_file"]
    assert res.metrics["worst_trigger"] == "from_file"


def test_triggers_path_missing_file(clean_booster, corpus, tmp_path):
    with pytest.raises(ConfigError, match="file not found"):
        _run(_model(clean_booster), corpus, triggers_path=str(tmp_path / "nope.yaml"))


@pytest.mark.parametrize(
    "params, match",
    [
        ({"max_trigger_drop": 1.5}, "max_trigger_drop"),
        ({"n_samples": 0}, "n_samples"),
        ({"min_suspicion": 0.0}, "min_suspicion"),
        ({"low_importance_quantile": 1.0}, "low_importance_quantile"),
        ({"max_rules_reported": -1}, "max_rules_reported"),
        ({"triggers": "header[5]=1"}, "triggers must be a list"),
    ],
)
def test_bad_params(clean_booster, corpus, params, match):
    with pytest.raises(ConfigError, match=match):
        _run(_model(clean_booster), corpus, **params)


# --------------------------------------------------------------------------------------------------
# Degrade / skip paths
# --------------------------------------------------------------------------------------------------


def test_without_tree_access_runs_trigger_test_only(backdoor_booster, corpus):
    model = _model(backdoor_booster, trees=False)
    res, ctx = _run(model, corpus, triggers=[_trigger()])
    assert Requirement.TREE_ACCESS not in ctx.capabilities
    assert res.metrics["scan_ran"] is False
    assert res.details["scan"]["ran"] is False
    assert [c.name for c in res.checks] == ["max_trigger_drop"]
    assert res.metrics["max_trigger_drop"] > 0.5
    assert res.gate_outcome is GateOutcome.FAILED
    assert any("Static tree-structure scan not run" in n for n in res.notes)
    assert "neural" not in res.details["art_poisoning_detectors"]["reason"]
    assert ABSENCE_NOTE in res.notes


def test_skip_without_trees_and_without_triggers(clean_booster, corpus):
    res, _ = _run(_model(clean_booster, trees=False), corpus)
    assert res.status is Status.SKIPPED
    assert "tree structure" in res.skip_reason and "trigger" in res.skip_reason
    assert res.screening is True


def test_rejected_trees_skip_cites_the_runner_note(clean_booster, corpus):
    """Regression (xcomp-generic-skip-reasons-contradict-runner): trees that the runner rejected (failed
    fidelity check) are not described as 'model does not expose its tree structure'."""
    ctx = make_context(BackdoorScreenModule, model=_model(clean_booster, trees=False), corpus=corpus)
    ctx.extras["unmet_reasons"] = {"tree_access": 'the tree dump does not reproduce predict_proba (max |Δ| = 0.25 > 0.0001 on 2000 corpus rows); tree access disabled so tree-based analyses cannot misdescribe the deployed model'}
    res = BackdoorScreenModule().run(ctx)
    assert res.status is Status.SKIPPED
    assert "does not reproduce predict_proba" in res.skip_reason
    assert "does not expose its tree structure" not in res.skip_reason


def test_rejected_trees_scan_note_cites_the_runner_note(clean_booster, corpus):
    ctx = make_context(BackdoorScreenModule, model=_model(clean_booster, trees=False), corpus=corpus,
                       params={"triggers": [_trigger()]})
    ctx.extras["unmet_reasons"] = {"tree_access": 'the tree dump does not reproduce predict_proba (max |Δ| = 0.25 > 0.0001 on 2000 corpus rows); tree access disabled so tree-based analyses cannot misdescribe the deployed model'}
    res = BackdoorScreenModule().run(ctx)
    assert any(n.startswith("Static tree-structure scan not run: the tree dump does not reproduce") for n in res.notes)


def test_trees_without_corpus_scan_runs_and_triggers_unevaluated(backdoor_booster):
    model = _model(backdoor_booster)
    ctx = make_context(BackdoorScreenModule, model=model, corpus=None, schema=ToySchema(),
                       params={"triggers": [_trigger()]})
    assert Requirement.FEATURE_SPACE not in ctx.capabilities
    res = BackdoorScreenModule().run(ctx)
    assert res.metrics["scan_ran"] is True and res.metrics["n_flagged_rules"] > 0
    trig = next(c for c in res.checks if c.name == "max_trigger_drop")
    assert trig.value is None and trig.passed is None
    assert res.details["triggers"]["ran"] is False
    assert res.gate_outcome is GateOutcome.FAILED  # the scan check failed


def test_trees_without_corpus_clean_is_not_evaluated_when_triggers_given(clean_booster):
    ctx = make_context(BackdoorScreenModule, model=_model(clean_booster), corpus=None, schema=ToySchema(),
                       params={"triggers": [_trigger()]})
    res = BackdoorScreenModule().run(ctx)
    assert res.gate_outcome is GateOutcome.NOT_EVALUATED
    assert res.status is Status.WARN  # never a silent pass for requested-but-untested triggers


def test_trigger_sampling_is_recorded_and_seeded(backdoor_booster, corpus):
    a, _ = _run(_model(backdoor_booster), corpus, triggers=[_trigger()], n_samples=150)
    b, _ = _run(_model(backdoor_booster), corpus, triggers=[_trigger()], n_samples=150)
    assert a.metrics["n_trigger_samples"] == 150
    assert a.details["triggers"]["subsampled"] is True
    assert any("seeded random sample of 150" in n for n in a.notes)
    assert a.metrics["max_trigger_drop"] == b.metrics["max_trigger_drop"]


# --------------------------------------------------------------------------------------------------
# Static scan internals
# --------------------------------------------------------------------------------------------------


def test_static_scan_direct(backdoor_booster, clean_booster):
    names = ToySchema().feature_names()
    bd = static_tree_scan(from_lightgbm_dump(backdoor_booster.dump_model()), feature_names=names, max_rules=5)
    cl = static_tree_scan(from_lightgbm_dump(clean_booster.dump_model()), feature_names=names, max_rules=5)
    assert int(np.argmax(bd.feature_suspicion)) == TRIGGER_FEATURE
    assert bd.feature_suspicion[TRIGGER_FEATURE] >= 0.9
    assert cl.feature_suspicion.max() < 0.5
    assert len(bd.top_rules) == 5
    assert [r["rank"] for r in bd.top_rules] == [1, 2, 3, 4, 5]
    assert bd.top_rules[0]["suspicion"] >= bd.top_rules[-1]["suspicion"]
    assert 0.0 < bd.top_rules[0]["cover_fraction"] < 1.0


def test_scan_rule_count_and_determinism(backdoor_booster):
    te = from_lightgbm_dump(backdoor_booster.dump_model())
    a = static_tree_scan(te, max_rules=10)
    b = static_tree_scan(te, max_rules=10)
    n_leaves = sum(int((t.children_left == LEAF).sum()) for t in te.trees if t.n_nodes > 1)
    assert a.n_rules == n_leaves
    assert a.top_rules == b.top_rules


def _random_ensemble(n_trees: int, depth: int, n_features: int, seed: int = 0) -> TreeEnsemble:
    """Complete binary trees with random splits/covers (a large ensemble for the speed test)."""
    rng = np.random.default_rng(seed)
    m = 2 ** (depth + 1) - 1
    n_int = 2**depth - 1
    trees = []
    for _ in range(n_trees):
        cl = np.full(m, LEAF, dtype=np.int32)
        cr = np.full(m, LEAF, dtype=np.int32)
        cl[:n_int] = 2 * np.arange(n_int) + 1
        cr[:n_int] = 2 * np.arange(n_int) + 2
        feat = np.full(m, LEAF, dtype=np.int32)
        feat[:n_int] = rng.integers(0, n_features, n_int)
        cover = np.zeros(m)
        cover[n_int:] = rng.integers(1, 200, m - n_int)
        gain = np.zeros(m)
        gain[:n_int] = rng.exponential(10.0, n_int)
        value = np.zeros(m)
        value[n_int:] = rng.normal(0, 0.1, m - n_int)
        trees.append(Tree(cl, cr, cl.copy(), feat, rng.normal(0, 1, m), value, cover, gain, np.zeros(m, np.int8)))
    return TreeEnsemble(trees=trees, n_features=n_features, model_kind="synthetic")


def test_scan_is_fast_on_large_ensemble():
    te = _random_ensemble(1000, 6, 2381)  # 1000 trees x 64 leaves over an EMBER-sized feature space
    t0 = time.monotonic()
    scan = static_tree_scan(te, max_rules=50)
    assert time.monotonic() - t0 < 20.0
    assert scan.n_rules == 1000 * 64
    assert scan.n_trees_scanned == 1000


def test_scan_without_cover_statistics_is_not_evaluated(clean_booster, corpus):
    te = from_lightgbm_dump(clean_booster.dump_model())
    for t in te.trees:
        t.cover[:] = 0.0

    class _Det:
        def predict_proba(self, X):
            return clean_booster.predict(X)

        def predict(self, X):
            return (clean_booster.predict(X) >= 0.5).astype(np.int8)

        def tree_ensemble(self):
            return te

    model = InProcessModel(_Det(), _model(clean_booster).declarations)
    res, _ = _run(model, corpus)
    chk = next(c for c in res.checks if c.name == "n_flagged_rules")
    assert chk.value is None and res.gate_outcome is GateOutcome.NOT_EVALUATED
    assert any("no training-cover statistics" in n for n in res.notes)


def test_scan_check_score_formula():
    assert scan_check_score(0, 0.45, 0.5) == 1.0
    s1 = scan_check_score(3, 0.6, 0.5)
    s2 = scan_check_score(3, 0.95, 0.5)
    assert 0.0 <= s2 < s1 < PASS_LINE
    assert scan_check_score(1, 1.0, 0.5) == 0.0
    assert scan_check_score(1, 1.0, 1.0) == 0.0


def test_module_contract():
    from malvalid import registry
    from malvalid.config import load_config, validate_against_registry

    assert registry.get_module("backdoor_screen") is BackdoorScreenModule
    assert BackdoorScreenModule.screening is True
    assert set(BackdoorScreenModule.requires_any) == {Requirement.TREE_ACCESS, Requirement.FEATURE_SPACE}
    cfg = load_config(None, overrides={"modules": {"backdoor_screen": {"min_suspicion": 0.6, "triggers": []}}})
    validate_against_registry(cfg)


def test_categorical_model_scan_runs_with_cover_note(corpus):
    """LightGBM categorical splits are rewritten by the loader as threshold chains with evenly shared
    cover; the scan still runs and says its narrowness/cover numbers are approximate there."""
    import lightgbm as lgb

    from malvalid.loaders.lightgbm_loader import LightGBMLoader

    tr = corpus.indices(splits=("train",))
    X = np.asarray(corpus.take(tr), dtype=np.float64)
    y = corpus.label[tr]
    X[:, 17] = np.where(y == 1, np.random.default_rng(0).integers(0, 5, tr.size), np.random.default_rng(1).integers(3, 9, tr.size))
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 8, "verbose": -1, "num_threads": 1, "seed": 0, "min_data_per_group": 5},
        lgb.Dataset(X, y, categorical_feature=[17]), num_boost_round=15,
    )
    loader = LightGBMLoader()

    class _Det:
        native_model = booster

        def predict_proba(self, X):
            return booster.predict(X)

        def predict(self, X):
            return (booster.predict(X) >= 0.5).astype(np.int8)

        def tree_ensemble(self):
            return loader.tree_ensemble(booster)

    model = InProcessModel(_Det(), _model(booster).declarations)
    assert model.tree_ensemble().meta.get("categorical_splits_expanded", 0) > 0
    res, _ = _run(model, None)
    assert res.metrics["scan_ran"] is True
    assert res.details["scan"]["categorical_splits_expanded"] > 0
    assert any("categorical splits" in n and "approximate" in n for n in res.notes)
