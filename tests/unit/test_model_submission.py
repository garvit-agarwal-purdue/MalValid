"""Model-file-only submissions: prepare_model_submission, the held-out calibration slice,
calibrate_threshold and the runner's auto-calibration."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from malvalid import registry, runner
from malvalid.adapters.spec import SPEC_FILENAME, load_spec
from malvalid.context import ModelDeclarations, RunContext
from malvalid.core import AdapterError, GateCheck, Module, ModuleResult, Requirement
from malvalid.corpora.base import ROLE_CHALLENGE, ROLE_EVAL, ROLE_POOL, ROLE_TEMPORAL
from malvalid.submission import (
    CALIBRATION_PERIODS,
    CALIBRATION_SPLIT,
    DEFAULT_CALIBRATE_FPR,
    DEFAULT_CALIBRATION_PERIOD,
    CalibrationError,
    calibrate_threshold,
    hold_out_calibration_slice,
    prepare_model_submission,
)
from malvalid.testing import make_toy_corpus
from tests.unit import model_factories as mf
from tests.unit.test_runner_support import (  # noqa: F401 - fake_plugins is a fixture
    FakeSandbox,
    RecordingModel,
    fake_plugins,
    install_fake_sandbox,
    make_cfg,
    make_decl,
    run_opts,
    toy_corpus,
    train_hashes,
)

# --------------------------------------------------------------------------------------------------
# prepare_model_submission
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    mf.save_lgbm(d / "v2.txt", 2381)
    mf.save_lgbm(d / "v3.txt", 2568)
    mf.save_lgbm(d / "bad100.txt", 100)
    mf.save_pickle(d / "gbdt.pkl", 2381)
    (d / "garbage.dat").write_bytes(bytes(range(256)) * 3)
    (d / "hashes.txt").write_text("a" * 64 + "\n")
    return d


class TestPrepare:
    @pytest.mark.posix
    def test_threshold(self, models, tmp_path):
        r = prepare_model_submission(models / "v2.txt", tmp_path / "sub", threshold=0.8)
        assert r.spec_path == tmp_path / "sub" / SPEC_FILENAME
        s = load_spec(r.spec_path)
        assert (s.model_kind, s.feature_version, s.operating_threshold, s.threshold_source) == (
            "lightgbm", "ember_v2", 0.8, "declared")
        link = tmp_path / "sub" / s.model_file
        assert link.is_symlink() and link.resolve() == (models / "v2.txt").resolve()
        assert r.notes == []

    def test_v3_detected(self, models, tmp_path):
        r = prepare_model_submission(models / "v3.txt", tmp_path / "s", threshold=0.5)
        assert r.spec.feature_version == "ember_v3"

    def test_default_calibrates_with_note(self, models, tmp_path):
        r = prepare_model_submission(models / "v2.txt", tmp_path / "s")
        assert r.spec.calibrate and r.spec.calibrate_fpr == DEFAULT_CALIBRATE_FPR == 0.005
        assert r.spec.operating_threshold is None
        assert any("calibrated" in n for n in r.notes)

    def test_calibrate_fpr(self, models, tmp_path):
        r = prepare_model_submission(models / "v2.txt", tmp_path / "s", calibrate_fpr=0.01)
        assert r.spec.calibrate_fpr == 0.01 and not r.notes

    def test_both_rejected(self, models, tmp_path):
        with pytest.raises(AdapterError, match="either --threshold or --calibrate-fpr"):
            prepare_model_submission(models / "v2.txt", tmp_path / "s", threshold=0.5, calibrate_fpr=0.01)

    @pytest.mark.parametrize("t", [-0.1, 1.5, float("nan"), True])
    def test_bad_threshold(self, models, tmp_path, t):
        with pytest.raises(AdapterError, match="--threshold"):
            prepare_model_submission(models / "v2.txt", tmp_path / "s", threshold=t)

    @pytest.mark.parametrize("f", [0.0, 0.5, -1, float("nan")])
    def test_bad_fpr(self, models, tmp_path, f):
        with pytest.raises(AdapterError, match="--calibrate-fpr"):
            prepare_model_submission(models / "v2.txt", tmp_path / "s", calibrate_fpr=f)

    def test_feature_count_mismatch(self, models, tmp_path):
        with pytest.raises(AdapterError, match=r"model expects 100 features; malvalid supports EMBER v2 \(2381\) "
                                                r"and EMBER v3 \(2568\)"):
            prepare_model_submission(models / "bad100.txt", tmp_path / "s", threshold=0.5)
        assert not (tmp_path / "s" / SPEC_FILENAME).exists()

    def test_feature_version_conflict(self, models, tmp_path):
        with pytest.raises(AdapterError, match="expects 2381 features but --feature-version ember_v3 has 2568"):
            prepare_model_submission(models / "v2.txt", tmp_path / "s", threshold=0.5, feature_version="ember_v3")

    def test_unknown_feature_version_and_kind(self, models, tmp_path):
        with pytest.raises(AdapterError, match="--feature-version must be one of"):
            prepare_model_submission(models / "v2.txt", tmp_path / "s", threshold=0.5, feature_version="nope")
        with pytest.raises(AdapterError, match="--model-kind must be one of"):
            prepare_model_submission(models / "v2.txt", tmp_path / "s", threshold=0.5, model_kind="torch")

    def test_garbage_file(self, models, tmp_path):
        with pytest.raises(AdapterError, match="could not detect the model kind"):
            prepare_model_submission(models / "garbage.dat", tmp_path / "s", threshold=0.5)

    def test_missing_file(self, tmp_path):
        with pytest.raises(AdapterError, match="file not found"):
            prepare_model_submission(tmp_path / "nope.txt", tmp_path / "s", threshold=0.5)

    def test_pickle_needs_allow_pickle_and_feature_version(self, models, tmp_path):
        with pytest.raises(AdapterError, match="--allow-pickle"):
            prepare_model_submission(models / "gbdt.pkl", tmp_path / "s", threshold=0.5, feature_version="ember_v2")
        with pytest.raises(AdapterError, match="could not detect the feature version"):
            prepare_model_submission(models / "gbdt.pkl", tmp_path / "s", threshold=0.5, allow_pickle=True)
        r = prepare_model_submission(models / "gbdt.pkl", tmp_path / "s", threshold=0.5, allow_pickle=True,
                                     feature_version="ember_v2")
        assert r.spec.model_kind == "sklearn_gbdt" and r.info.is_pickle

    @pytest.mark.posix
    def test_training_files_and_cutoff(self, models, tmp_path):
        r = prepare_model_submission(models / "v2.txt", tmp_path / "s", threshold=0.5,
                                     training_hashes=models / "hashes.txt", training_cutoff="2018-10")
        assert r.spec.training_cutoff == "2018-10"
        assert (tmp_path / "s" / r.spec.training_hashes_file).is_symlink()
        with pytest.raises(AdapterError, match="--training-hashes"):
            prepare_model_submission(models / "v2.txt", tmp_path / "t", threshold=0.5,
                                     training_hashes=models / "missing.txt")
        with pytest.raises(AdapterError, match="training_cutoff"):
            prepare_model_submission(models / "v2.txt", tmp_path / "u", threshold=0.5, training_cutoff="2018-13")

    def test_hostile_file_names_are_sanitized(self, models, tmp_path):
        src = tmp_path / "we ird;name$.txt"
        src.write_bytes((models / "v2.txt").read_bytes())
        r = prepare_model_submission(src, tmp_path / "s", threshold=0.5)
        assert "/" not in r.spec.model_file and (tmp_path / "s" / r.spec.model_file).exists()
        src2 = tmp_path / SPEC_FILENAME.replace(".json", ".txt")  # not the reserved .json name, control
        src2.write_bytes(src.read_bytes())
        r2 = prepare_model_submission(src2, tmp_path / "s2", threshold=0.5)
        assert r2.spec.model_file != SPEC_FILENAME

    def test_model_kind_override_note(self, models, tmp_path):
        r = prepare_model_submission(models / "v2.txt", tmp_path / "s", threshold=0.5, model_kind="xgboost")
        assert r.spec.model_kind == "xgboost" and any("overrides" in n for n in r.notes)


# --------------------------------------------------------------------------------------------------
# calibrate_threshold
# --------------------------------------------------------------------------------------------------


class TestCalibrateThreshold:
    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("target", [0.001, 0.005, 0.01, 0.05, 0.1])
    def test_achieved_within_target(self, seed, target):
        s = np.random.default_rng(seed).random(2000)
        t, ach = calibrate_threshold(s, target)
        assert ach <= target + 1e-12 and 0 <= t <= 1
        assert ach == pytest.approx(float(np.mean(s >= t)))
        # smallest such threshold: lowering to the next lower score would exceed the target (or k allows all)
        k = int(np.floor(target * s.size + 1e-9))
        assert np.count_nonzero(s >= t) == k or k == 0

    def test_monotone_in_target(self):
        s = np.random.default_rng(0).random(5000)
        ts = [calibrate_threshold(s, f)[0] for f in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)]
        assert all(a >= b for a, b in zip(ts, ts[1:]))
        achs = [calibrate_threshold(s, f)[1] for f in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)]
        assert all(a <= b for a, b in zip(achs, achs[1:]))

    def test_predict_rule_is_ge(self):
        s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        t, ach = calibrate_threshold(s, 0.2)
        assert ach == 0.2 and int((s >= t).sum()) == 2

    def test_all_equal_scores(self):
        s = np.full(100, 0.4)
        t, ach = calibrate_threshold(s, 0.01)
        assert t > 0.4 and ach == 0.0  # everything is tied: only a threshold above them keeps FPR <= target

    def test_k_zero_small_n(self):
        s = np.array([0.2, 0.9, 0.5])
        t, ach = calibrate_threshold(s, 0.005)  # floor(0.015) = 0 false positives allowed
        assert ach == 0.0 and t > 0.9

    def test_n_one(self):
        t, ach = calibrate_threshold(np.array([0.3]), 0.1)
        assert ach == 0.0 and t > 0.3

    def test_k_ge_n_gives_zero(self):
        t, ach = calibrate_threshold(np.array([0.3, 0.6]), 0.1 * 20)  # target above 1: allowed all
        assert t == 0.0 and ach == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            calibrate_threshold(np.array([]), 0.01)

    def test_accepts_2d_and_ints(self):
        t, ach = calibrate_threshold(np.arange(100).reshape(10, 10) / 100.0, 0.05)
        assert ach <= 0.05

    def test_saturated_scores_raise(self):
        s = np.concatenate([np.full(5, 1.0), np.random.default_rng(0).random(995) * 0.5])
        with pytest.raises(CalibrationError, match="not achievable.*score 1.0.*--threshold.*--calibrate-fpr"):
            calibrate_threshold(s, 0.001)

    def test_saturated_scores_ok_when_target_allows(self):
        s = np.concatenate([np.full(5, 1.0), np.random.default_rng(0).random(995) * 0.5])
        t, ach = calibrate_threshold(s, 0.01)  # 9 false positives allowed, only 5 saturated
        assert ach <= 0.01 and t <= 1.0


# --------------------------------------------------------------------------------------------------
# hold_out_calibration_slice
# --------------------------------------------------------------------------------------------------


def _rule(sha: str) -> bool:
    return int(sha[:8], 16) % 10 == 0


@pytest.fixture(scope="module")
def corpus():
    return make_toy_corpus(6000, seed=3)


class TestHoldOut:
    def test_selected_rows(self, corpus):
        train = frozenset(corpus.sha256[corpus.indices(splits=("train",))].tolist())
        view, idx, meta = hold_out_calibration_slice(corpus, train, "uniform")
        assert idx.size > 50 and meta["n_rows"] == idx.size and meta["policy"] == "uniform"
        eval_splits = set(corpus.role_splits(ROLE_EVAL))
        assert set(corpus.label[idx].tolist()) == {0}
        assert set(corpus.split[idx].tolist()) <= eval_splits
        assert all(_rule(h) for h in corpus.sha256[idx].tolist())
        assert not (set(corpus.sha256[idx].tolist()) & train)
        # complete: every benign eval row satisfying the rule and not in training is selected
        want = [i for i in corpus.indices(role=ROLE_EVAL, label=0).tolist()
                if _rule(str(corpus.sha256[i])) and str(corpus.sha256[i]) not in train]
        assert idx.tolist() == want

    def test_training_hashes_excluded(self, corpus):
        _, idx_all, _ = hold_out_calibration_slice(corpus, None, "uniform")
        drop = frozenset(corpus.sha256[idx_all[:10]].tolist())
        _, idx, _ = hold_out_calibration_slice(corpus, drop, "uniform")
        assert idx.size == idx_all.size - 10 and not (set(corpus.sha256[idx].tolist()) & drop)

    def test_training_hashes_excluded_earliest(self, corpus):
        _, idx_all, _ = hold_out_calibration_slice(corpus, None, "earliest")
        drop = frozenset(corpus.sha256[idx_all[:10]].tolist())
        _, idx, _ = hold_out_calibration_slice(corpus, drop, "earliest")
        assert idx.size > 0 and not (set(corpus.sha256[idx].tolist()) & drop)

    @pytest.mark.parametrize("period", CALIBRATION_PERIODS)
    def test_view_excludes_rows_from_every_role(self, corpus, period):
        view, idx, meta = hold_out_calibration_slice(corpus, None, period)
        assert meta["policy"] == period
        held = set(idx.tolist())
        assert (view.split[idx] == CALIBRATION_SPLIT).all() and meta["split_name"] == CALIBRATION_SPLIT
        for role in (ROLE_EVAL, ROLE_TEMPORAL, ROLE_POOL, ROLE_CHALLENGE):
            for labeled_only in (True, False):
                assert not (set(view.indices(role=role, labeled_only=labeled_only).tolist()) & held), role
        for lab in (0, 1):
            orig = corpus.eval_indices(lab)
            got = view.eval_indices(lab)
            if lab == 0:
                assert got.tolist() == [i for i in orig.tolist() if i not in held]
            else:
                assert got.tolist() == orig.tolist()
        # the original corpus is untouched
        assert (corpus.split != CALIBRATION_SPLIT).all()
        assert set(corpus.indices(role=ROLE_EVAL).tolist()) >= held

    def test_eval_indices_with_training_exclusion(self, corpus):
        train = frozenset(corpus.sha256[corpus.indices(splits=("train",))].tolist())
        view, idx, _ = hold_out_calibration_slice(corpus, train)
        want = [i for i in corpus.eval_indices(0, train).tolist() if i not in set(idx.tolist())]
        assert view.eval_indices(0, train).tolist() == want

    def test_x_shared_not_copied(self, corpus):
        view, _, _ = hold_out_calibration_slice(corpus, None)
        assert view.X is corpus.X and np.shares_memory(view.X, corpus.X)
        assert view.sha256 is corpus.sha256 and view.label is corpus.label

    def test_explicit_pool_role_kept_explicit(self, corpus):
        view, idx, _ = hold_out_calibration_slice(corpus, None)
        assert CALIBRATION_SPLIT not in view.role_splits(ROLE_POOL)
        assert view.manifest["roles"][ROLE_POOL]
        assert "pool" not in (corpus.manifest.get("roles") or {})  # original manifest not mutated

    def test_duplicate_hash_rows_all_held_out(self, corpus):
        # a row in the train split carrying the same hash as a held-out candidate must not leak into evaluation
        _, idx0, _ = hold_out_calibration_slice(corpus, None)
        victim = int(idx0[0])
        donor = int(corpus.indices(splits=("train",))[0])
        sha = corpus.sha256.copy()
        sha[donor] = sha[victim]
        c2 = dataclasses.replace(corpus, sha256=sha, _hash_index=None)
        view, idx, _ = hold_out_calibration_slice(c2, None)
        assert view.split[donor] == CALIBRATION_SPLIT and view.split[victim] == CALIBRATION_SPLIT

    @pytest.mark.parametrize("period", CALIBRATION_PERIODS)
    def test_deterministic(self, corpus, period):
        a = hold_out_calibration_slice(corpus, None, period)[1]
        b = hold_out_calibration_slice(corpus, None, period)[1]
        assert a.tolist() == b.tolist()


def _eval_benign(c, train=None):
    return c.indices(role=ROLE_EVAL, label=0, exclude_hashes=train or None)


def _with_timestamps(c, ts):
    return dataclasses.replace(c, timestamp=np.asarray(ts, dtype="datetime64[D]"), _hash_index=None)


class TestEarliestPeriod:
    """calibration_period "earliest": a held threshold fit on the earliest benign eval rows."""

    def test_default_policy_is_earliest(self, corpus):
        from malvalid.config import load_config

        assert DEFAULT_CALIBRATION_PERIOD == "earliest" and set(CALIBRATION_PERIODS) == {"earliest", "uniform"}
        assert load_config().runtime.calibration_period == "earliest"
        _, _, meta = hold_out_calibration_slice(corpus, None)
        assert meta["policy"] == meta["requested_policy"] == "earliest" and "fallback_reason" not in meta

    def test_selects_the_earliest_whole_days(self, corpus):
        ben = _eval_benign(corpus)
        _, idx, meta = hold_out_calibration_slice(corpus, None, "earliest")
        n_target = -(-ben.size // 10)
        assert meta["n_target"] == n_target and n_target <= idx.size <= 2 * n_target
        assert set(corpus.label[idx].tolist()) == {0} and set(idx.tolist()) <= set(ben.tolist())
        rest = np.setdiff1d(ben, idx)
        # day-level corpus: the boundary day is taken whole, so every scored benign row is strictly later
        assert meta["boundary_split"] is False
        assert corpus.timestamp[idx].max() < corpus.timestamp[rest].min()
        assert meta["scored_benign_after_period"] == 1.0
        assert meta["period"] == [str(corpus.timestamp[idx].min()), str(corpus.timestamp[idx].max())]
        assert meta["period"][0] == str(corpus.timestamp[ben].min())
        assert "earliest" in meta["rule"] and "by timestamp" in meta["rule"]

    def test_earlier_than_uniform(self, corpus):
        _, e, _ = hold_out_calibration_slice(corpus, None, "earliest")
        _, u, mu = hold_out_calibration_slice(corpus, None, "uniform")
        assert corpus.timestamp[e].max() < corpus.timestamp[u].max()
        assert mu["scored_benign_after_period"] < 0.05  # uniform spans (almost) the whole period

    def test_coarse_timestamps_split_the_boundary_by_hash(self, corpus):
        # month-level corpus (EMBER2018-like): every eval row carries the same date
        ts = corpus.timestamp.copy()
        ev = np.isin(corpus.split, corpus.role_splits(ROLE_EVAL))
        ts[ev] = np.datetime64("2018-11-01")
        c = _with_timestamps(corpus, ts)
        ben = _eval_benign(c)
        _, idx, meta = hold_out_calibration_slice(c, None, "earliest")
        n_target = -(-ben.size // 10)
        assert idx.size == n_target and meta["boundary_split"] is True
        want = sorted(ben.tolist(), key=lambda i: str(c.sha256[i]))[:n_target]
        assert idx.tolist() == sorted(want)
        assert meta["scored_benign_after_period"] == 0.0 and meta["period"] == ["2018-11-01", "2018-11-01"]

    def test_two_coarse_periods_keep_the_later_one_whole(self, corpus):
        # two month-level timestamps: the slice comes only from the first, the second is scored in full
        ts = corpus.timestamp.copy()
        ben = _eval_benign(corpus)
        half = np.zeros(corpus.n, bool)
        half[ben[: ben.size // 2]] = True
        ev = np.isin(corpus.split, corpus.role_splits(ROLE_EVAL))
        ts[ev] = np.datetime64("2018-12-01")
        ts[half] = np.datetime64("2018-11-01")
        c = _with_timestamps(corpus, ts)
        _, idx, meta = hold_out_calibration_slice(c, None, "earliest")
        assert (c.timestamp[idx] == np.datetime64("2018-11-01")).all()
        assert meta["boundary_split"] is True
        assert 0.45 < meta["scored_benign_after_period"] < 0.6

    def test_order_independent(self, corpus):
        """The slice depends only on (timestamp, sha256), not on row order (or worker count)."""
        perm = np.random.default_rng(1).permutation(corpus.n)
        c2 = dataclasses.replace(
            corpus, X=corpus.X[perm], sha256=corpus.sha256[perm], label=corpus.label[perm],
            timestamp=corpus.timestamp[perm], split=corpus.split[perm], _hash_index=None)
        a = set(corpus.sha256[hold_out_calibration_slice(corpus, None, "earliest")[1]].tolist())
        b = set(c2.sha256[hold_out_calibration_slice(c2, None, "earliest")[1]].tolist())
        assert a == b

    def test_undated_rows_never_selected(self, corpus):
        ts = corpus.timestamp.copy()
        ben = _eval_benign(corpus)
        ts[ben[:: 7]] = np.datetime64("NaT")
        c = _with_timestamps(corpus, ts)
        _, idx, meta = hold_out_calibration_slice(c, None, "earliest")
        assert meta["policy"] == "earliest" and not np.isnat(c.timestamp[idx]).any()

    def test_falls_back_to_uniform_when_too_few_rows_are_dated(self, corpus):
        ts = corpus.timestamp.copy()
        ben = _eval_benign(corpus)
        ts[ben[: int(ben.size * 0.95)]] = np.datetime64("NaT")
        c = _with_timestamps(corpus, ts)
        _, idx, meta = hold_out_calibration_slice(c, None, "earliest")
        _, uni, _ = hold_out_calibration_slice(c, None, "uniform")
        assert meta["policy"] == "uniform" and meta["requested_policy"] == "earliest"
        assert "fewer than" in meta["fallback_reason"] and idx.tolist() == uni.tolist()

    def test_unknown_period_rejected(self, corpus):
        with pytest.raises(ValueError, match="calibration period"):
            hold_out_calibration_slice(corpus, None, "latest")

    def test_config_rejects_unknown_period(self, tmp_path):
        from malvalid.config import load_config
        from malvalid.core import ConfigError

        p = tmp_path / "g.yaml"
        p.write_text("runtime:\n  calibration_period: latest\n")
        with pytest.raises(ConfigError):
            load_config(p)
        p.write_text("runtime:\n  calibration_period: uniform\n")
        assert load_config(p).runtime.calibration_period == "uniform"


# --------------------------------------------------------------------------------------------------
# runner: auto-calibration on the held-out rows only
# --------------------------------------------------------------------------------------------------


class _ProbeMod(Module):
    id = "t_calprobe"
    code = "TX"
    title = "Records what the corpus exposes to modules"
    requires = (Requirement.FEATURE_SPACE,)
    seen: list[dict] = []

    def run(self, ctx: RunContext) -> ModuleResult:
        c = ctx.corpus
        assert c is not None
        type(self).seen.append({
            "split_names": set(c.split.tolist()),
            "eval0": c.eval_indices(0, ctx.training_hashes).tolist(),
            "roles": {r: c.indices(role=r).tolist() for r in (ROLE_EVAL, ROLE_TEMPORAL, ROLE_POOL, ROLE_CHALLENGE)},
            "threshold": ctx.model.declarations.operating_threshold,
        })
        p = ctx.score(c.take(c.eval_indices(1, ctx.training_hashes)[:32]))
        return self.result(ctx, finding="ok", checks=[GateCheck.evaluate("x", 1.0, ">=", 0.5, ideal=1.0, floor=0.0)],
                           metrics={"n": int(p.size)})


def _spec_decl(tmp_path: Path, **spec_over) -> ModelDeclarations:
    base = make_decl(tmp_path, hashes=train_hashes())
    spec = {"schema": "malvalid-model-spec/1", "model_kind": "lightgbm", "feature_version": "toy_v1",
            "operating_threshold": None, "threshold_source": "calibrate", "calibrate_fpr": 0.02,
            "model_file": "model.txt", "training_hashes_file": "train_hashes.txt", "training_cutoff": "2017-12"}
    spec.update(spec_over)
    extras = {"submission": "model_file", "spec": spec, "predict_from_proba": True,
              "threshold_source": "calibration_pending" if spec["threshold_source"] == "calibrate" else "declared"}
    thr = spec["operating_threshold"]
    return dataclasses.replace(base, extras=extras, operating_threshold=float("nan") if thr is None else thr)


@pytest.fixture
def probe_registered(fake_plugins):
    registry.register("modules", _ProbeMod.id, _ProbeMod)
    _ProbeMod.seen = []
    yield
    registry.unregister("modules", _ProbeMod.id)


@pytest.fixture
def calibrated_run(monkeypatch, tmp_path, probe_registered):
    import malvalid.submission as sub

    monkeypatch.setattr(sub, "MIN_CALIBRATION_ROWS", 20)
    fs = install_fake_sandbox(monkeypatch, FakeSandbox(decl=_spec_decl(tmp_path)))
    queries: list[np.ndarray] = []
    orig = RecordingModel.predict_proba

    def spy(self, X):
        queries.append(np.array(X, copy=True))
        return orig(self, X)

    monkeypatch.setattr(RecordingModel, "predict_proba", spy)
    out = runner.run_gate(run_opts(tmp_path, make_cfg(["t_calprobe"])))
    return out, fs, queries


class TestRunnerCalibration:
    def test_scores_come_only_from_held_out_rows(self, calibrated_run):
        out, fs, queries = calibrated_run
        corp = toy_corpus()
        hashes = frozenset(train_hashes())
        _, idx, _ = hold_out_calibration_slice(corp, hashes)
        assert idx.size >= 20
        assert len(queries[0]) == idx.size  # the very first model query is the calibration pass
        np.testing.assert_array_equal(queries[0], corp.take(idx))
        # and the held-out rows are benign eval-role rows, never scored by any later query
        later = np.concatenate(queries[1:]) if len(queries) > 1 else np.empty((0, corp.dim), np.float32)
        held_rows = {r.tobytes() for r in queries[0]}
        assert not any(r.tobytes() in held_rows for r in later)

    def test_modules_see_no_calibration_rows(self, calibrated_run):
        out, fs, queries = calibrated_run
        corp = toy_corpus()
        _, idx, _ = hold_out_calibration_slice(corp, frozenset(train_hashes()))
        held = set(idx.tolist())
        seen = _ProbeMod.seen[0]
        assert CALIBRATION_SPLIT in seen["split_names"]
        assert not (set(seen["eval0"]) & held)
        for role, rows in seen["roles"].items():
            assert not (set(rows) & held), role

    def test_threshold_is_calibrated_before_modules_run(self, calibrated_run):
        out, fs, queries = calibrated_run
        thr = _ProbeMod.seen[0]["threshold"]
        assert not np.isnan(thr) and 0.0 <= thr <= 1.0
        corp = toy_corpus()
        _, idx, _ = hold_out_calibration_slice(corp, frozenset(train_hashes()))
        scores = fs.models[0].detector.predict_proba(corp.take(idx))
        exp_t, exp_ach = calibrate_threshold(scores, 0.02)
        assert thr == pytest.approx(exp_t) and exp_ach <= 0.02

    def test_report(self, calibrated_run):
        out, fs, queries = calibrated_run
        rep = out.report
        json.dumps(rep, allow_nan=False)
        extras = rep["model"]["extras"]
        assert extras["threshold_source"] == "calibrated"
        cal = extras["threshold_calibration"]
        assert cal["target_fpr"] == 0.02 and cal["achieved_fpr"] <= 0.02 and cal["n_benign"] >= 20
        assert rep["model"]["operating_threshold"] == pytest.approx(cal["threshold"])
        assert any("auto-calibrated" in w for w in rep["warnings"])
        hold = rep["corpus"]["calibration_holdout"]
        assert hold["split_name"] == CALIBRATION_SPLIT and hold["n_rows"] == cal["n_benign"]
        assert "indices" not in hold and hold["target_fpr"] == 0.02
        # default policy: a held threshold on the earliest benign rows, recorded everywhere
        assert hold["policy"] == cal["policy"] == "earliest" and hold["period"] == cal["period"]
        assert any("auto-calibrated" in w and "held" in w for w in rep["warnings"])

    @pytest.mark.parametrize("period", CALIBRATION_PERIODS)
    def test_config_policy_selects_rows(self, monkeypatch, tmp_path, probe_registered, period):
        import malvalid.submission as sub

        monkeypatch.setattr(sub, "MIN_CALIBRATION_ROWS", 20)
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=_spec_decl(tmp_path)))
        out = runner.run_gate(run_opts(tmp_path, make_cfg(["t_calprobe"], calibration_period=period)))
        corp = toy_corpus()
        _, idx, meta = hold_out_calibration_slice(corp, frozenset(train_hashes()), period)
        hold = out.report["corpus"]["calibration_holdout"]
        assert hold["policy"] == period and hold["n_rows"] == idx.size and hold["period"] == meta["period"]
        seen = _ProbeMod.seen[0]
        held = set(idx.tolist())
        for role, rows in seen["roles"].items():  # excluded from every role, whichever the policy
            assert not (set(rows) & held), role
        assert not (set(seen["eval0"]) & held)
        assert out.report["config"]["runtime"]["calibration_period"] == period

    def test_fallback_is_warned(self, monkeypatch, tmp_path, probe_registered):
        import malvalid.submission as sub

        monkeypatch.setattr(sub, "MIN_CALIBRATION_ROWS", 20)
        monkeypatch.setattr(sub, "_earliest_selection", lambda c, b: (None, {"fallback_reason": "too few dated"}))
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=_spec_decl(tmp_path)))
        out = runner.run_gate(run_opts(tmp_path, make_cfg(["t_calprobe"])))
        hold = out.report["corpus"]["calibration_holdout"]
        assert hold["policy"] == "uniform" and hold["requested_policy"] == "earliest"
        assert any("calibration_period earliest: too few dated" in w for w in out.report["warnings"])

    def test_declared_threshold_has_no_calibration(self, monkeypatch, tmp_path, probe_registered):
        fs = install_fake_sandbox(monkeypatch, FakeSandbox(decl=_spec_decl(
            tmp_path, threshold_source="declared", operating_threshold=0.7, calibrate_fpr=None)))
        out = runner.run_gate(run_opts(tmp_path, make_cfg(["t_calprobe"])))
        rep = out.report
        assert rep["model"]["extras"].get("threshold_source") != "calibrated"
        assert "calibration_holdout" not in rep["corpus"]
        assert not any("auto-calibrated" in w for w in rep["warnings"])
        assert CALIBRATION_SPLIT not in _ProbeMod.seen[0]["split_names"]
        assert _ProbeMod.seen[0]["threshold"] == 0.7

    def test_too_few_rows_refused(self, monkeypatch, tmp_path, probe_registered):
        # default MIN_CALIBRATION_ROWS (200) exceeds the toy corpus's held-out slice
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=_spec_decl(tmp_path)))
        with pytest.raises(AdapterError, match="benign rows available for threshold calibration"):
            runner.run_gate(run_opts(tmp_path, make_cfg(["t_calprobe"])))


class TestCalibrationFiniteGuard:
    def test_calibrate_threshold_rejects_nan(self):
        s = np.linspace(0.0, 1.0, 100)
        s[3] = np.nan
        with pytest.raises(ValueError, match="not finite"):
            calibrate_threshold(s, 0.05)

    def test_calibrate_threshold_rejects_inf(self):
        with pytest.raises(ValueError, match="not finite"):
            calibrate_threshold(np.array([0.1, np.inf, 0.2]), 0.5)

    def test_runner_rejects_non_finite_calibration_scores(self, monkeypatch, tmp_path, probe_registered):
        """A handle that does not validate its output (e.g. in-process) cannot hand NaN to calibration."""
        from malvalid.core import AdapterContractError

        import malvalid.submission as sub

        monkeypatch.setattr(sub, "MIN_CALIBRATION_ROWS", 20)
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=_spec_decl(tmp_path)))
        orig = RecordingModel.predict_proba

        def nan_scores(self, X):
            p = np.array(orig(self, X), dtype=np.float64)
            p[1] = np.nan
            return p

        monkeypatch.setattr(RecordingModel, "predict_proba", nan_scores)
        called: list[np.ndarray] = []
        real = sub.calibrate_threshold
        monkeypatch.setattr(sub, "calibrate_threshold", lambda s, f: called.append(s) or real(s, f))
        out = runner.run_gate(run_opts(tmp_path, make_cfg(["t_calprobe"])))
        gate = out.report["gate"]
        assert AdapterContractError.__name__ in gate["load_error"] and "non-finite" in gate["load_error"]
        assert "threshold calibration failed" in gate["abort_reason"]
        assert not called  # rejected before the threshold rule ever sees the scores
        assert not _ProbeMod.seen  # no module ran on an undefined threshold


class TestRunnerCalibrationBootstrap:
    def test_m1_gets_calibration_scores_but_report_does_not(self, monkeypatch, tmp_path, fake_plugins):
        import malvalid.submission as sub

        monkeypatch.setattr(sub, "MIN_CALIBRATION_ROWS", 20)
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=_spec_decl(tmp_path)))
        out = runner.run_gate(run_opts(tmp_path, make_cfg(["performance"])))
        rep = out.report
        m1 = next(m for m in rep["modules"] if m["module_id"] == "performance")
        tb = m1["details"]["threshold_bootstrap"]
        n_cal = rep["model"]["extras"]["threshold_calibration"]["n_benign"]
        assert tb["applicable"] and tb["ran"] and tb["n_calibration"] == n_cal
        assert tb["resamples"] == 1000 and tb["target_fpr"] == 0.02
        lo, hi = m1["metrics"]["threshold_ci95"]
        assert lo <= rep["model"]["operating_threshold"] <= hi
        assert m1["metrics"]["fpr_ci95_with_threshold"] == tb["fpr_ci95"]
        text = json.dumps(rep, allow_nan=False)
        assert '"scores"' not in text  # raw calibration scores are never persisted
        pol = m1["details"]["threshold_calibration_policy"]
        assert pol["policy"] == "earliest" and pol["period"] == rep["corpus"]["calibration_holdout"]["period"]
        assert any(n.startswith("Held threshold") for n in m1["notes"])

    @pytest.mark.parametrize("period", CALIBRATION_PERIODS)
    def test_bootstrap_uses_the_rows_actually_calibrated_on(self, monkeypatch, tmp_path, fake_plugins, period):
        import malvalid.modules.performance as perf
        import malvalid.submission as sub

        monkeypatch.setattr(sub, "MIN_CALIBRATION_ROWS", 20)
        fs = install_fake_sandbox(monkeypatch, FakeSandbox(decl=_spec_decl(tmp_path)))
        seen: list[np.ndarray] = []
        real = perf.threshold_bootstrap
        monkeypatch.setattr(perf, "threshold_bootstrap", lambda cal, *a, **k: seen.append(np.array(cal)) or real(cal, *a, **k))
        out = runner.run_gate(run_opts(tmp_path, make_cfg(["performance"], calibration_period=period)))
        corp = toy_corpus()
        _, idx, _ = hold_out_calibration_slice(corp, frozenset(train_hashes()), period)
        want = np.asarray(fs.models[0].detector.predict_proba(corp.take(idx)), dtype=np.float64)
        assert len(seen) == 1
        np.testing.assert_allclose(seen[0], want)
        m1 = next(m for m in out.report["modules"] if m["module_id"] == "performance")
        assert m1["details"]["threshold_calibration_policy"]["policy"] == period
        assert m1["details"]["threshold_bootstrap"]["n_calibration"] == idx.size

    def test_m2_reports_the_policy_and_never_scores_calibration_rows(self, monkeypatch, tmp_path, fake_plugins):
        import malvalid.submission as sub

        monkeypatch.setattr(sub, "MIN_CALIBRATION_ROWS", 20)
        install_fake_sandbox(monkeypatch, FakeSandbox(decl=_spec_decl(tmp_path)))
        scored: list[np.ndarray] = []
        orig = RecordingModel.predict_proba
        monkeypatch.setattr(RecordingModel, "predict_proba", lambda self, X: scored.append(np.array(X)) or orig(self, X))
        out = runner.run_gate(run_opts(tmp_path, make_cfg(["drift"])))
        m2 = next(m for m in out.report["modules"] if m["module_id"] == "drift")
        assert m2["status"] in ("pass", "warn", "fail"), m2["finding"]
        pol = m2["details"]["threshold_calibration_policy"]
        assert pol["policy"] == "earliest"
        assert any(n.startswith("Held threshold") and "per-window FPR" in n for n in m2["notes"])
        corp = toy_corpus()
        _, idx, _ = hold_out_calibration_slice(corp, frozenset(train_hashes()), "earliest")
        held = {r.tobytes() for r in corp.take(idx)}
        assert len(scored) >= 2
        assert not any(r.tobytes() in held for q in scored[1:] for r in q)  # only the calibration pass saw them
