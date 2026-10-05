"""Unit tests for M2 temporal drift (malvalid.modules.drift)."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import pytest

from malvalid.config import load_config
from malvalid.context import REQUIREMENT_HINTS, ModelDeclarations
from malvalid.core import ConfigError, GateOutcome, Requirement, Status, unmet_requirements
from malvalid.corpora.base import Corpus
from malvalid.modules import drift as drift_mod
from malvalid.modules.drift import (
    DriftModule,
    aut_sample_weighted,
    aut_unweighted,
    binary_counts,
    auroc,
    fpr_budget_summary,
    operating_point_stats,
    load_temporal,
    partition_windows,
    train_test_temporally_consistent,
)
from malvalid.testing import InProcessModel, make_context, make_toy_corpus, train_toy_lgbm, training_hashes_for

CUTOFF = dt.date(2017, 12, 31)  # toy 'train' split is 2017-01..2017-12


# --------------------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drifting():
    c = make_toy_corpus(n=8000, drift=1.0, string_signal=0.0)
    return c, train_toy_lgbm(c)


@pytest.fixture(scope="module")
def stable():
    c = make_toy_corpus(n=8000, drift=0.0, string_signal=0.0)
    return c, train_toy_lgbm(c)


@pytest.fixture(scope="module")
def small():
    c = make_toy_corpus()  # 4000 rows, ~150 per month
    return c, train_toy_lgbm(c)


def _lgbm(booster, threshold=0.5):
    return InProcessModel.from_lightgbm(booster, threshold=threshold, feature_version="toy_v1")


def _run(model, corpus, **kw):
    ctx = make_context(DriftModule, model=model, corpus=corpus, **kw)
    return DriftModule().run(ctx), ctx


class _FnDetector:
    def predict_proba(self, X):
        return X[:, 0].astype(np.float64)

    def predict(self, X):
        return (X[:, 0] >= 0.5).astype(np.int8)


def _fn_model(threshold=0.5):
    decl = ModelDeclarations(
        feature_version="toy_v1", model_kind="custom", operating_threshold=threshold,
        training_hashes_path=None, training_cutoff=None,
    )
    return InProcessModel(_FnDetector(), decl)


def _block(month: str, n_mal_hit: int, n_mal_miss: int, n_ben_ok: int, n_ben_fp: int, *, mal_day=None, ben_day=None):
    """Rows for one month: (date, label, score). Score 0.9 => flagged, 0.1 => not flagged."""
    rows = []
    base = np.datetime64(month + "-01", "D")

    def day(i, fixed):
        return base + np.timedelta64(fixed if fixed is not None else i % 28, "D")

    k = 0
    for lab, score, cnt, fixed in ((1, 0.9, n_mal_hit, mal_day), (1, 0.1, n_mal_miss, mal_day),
                                   (0, 0.1, n_ben_ok, ben_day), (0, 0.9, n_ben_fp, ben_day)):
        for _ in range(cnt):
            rows.append((day(k, fixed), lab, score))
            k += 1
    return rows


def _handmade(rows) -> Corpus:
    n = len(rows)
    X = np.zeros((n, 32), dtype=np.float32)
    X[:, 0] = [r[2] for r in rows]
    manifest = {
        "format": "malvalid-corpus/1", "name": "handmade", "version": "1", "feature_version": "toy_v1",
        "dim": 32, "n": n, "roles": {"eval": ["test"], "temporal": ["test"]}, "files": {"X": "none"},
    }
    return Corpus(
        name="handmade", version="1", feature_version="toy_v1", content_hash="handmade", manifest=manifest, X=X,
        sha256=np.array([hashlib.sha256(f"d{i}".encode()).hexdigest() for i in range(n)], dtype="<U64"),
        label=np.array([r[1] for r in rows], dtype=np.int8),
        timestamp=np.array([r[0] for r in rows], dtype="datetime64[D]"),
        split=np.array(["test"] * n, dtype="<U16"), synthetic=True,
    )


# --------------------------------------------------------------------------------------------------
# AUT
# --------------------------------------------------------------------------------------------------


def test_aut_hand_computed():
    f, n = [0.9, 0.8, 0.5], [1000, 100, 1000]
    assert aut_unweighted(f) == pytest.approx(((0.9 + 0.8) / 2 + (0.8 + 0.5) / 2) / 2)  # 0.75
    # sum_k (n_k f_k + n_k+1 f_k+1) / sum_k (n_k + n_k+1) = (980 + 580) / 2200
    assert aut_sample_weighted(f, n) == pytest.approx(1560 / 2200)
    # a sparse dip barely moves the weighted AUT
    assert aut_unweighted([0.9, 0.2, 0.9]) == pytest.approx(0.55)
    assert aut_sample_weighted([0.9, 0.2, 0.9], [1000, 10, 1000]) == pytest.approx(1804 / 2020)
    # equal windows => weighted == unweighted; < 2 windows => None
    assert aut_sample_weighted([0.7, 0.9, 0.4, 0.6], [5, 5, 5, 5]) == pytest.approx(aut_unweighted([0.7, 0.9, 0.4, 0.6]))
    assert aut_unweighted([0.9]) is None and aut_sample_weighted([0.9], [10]) is None
    with pytest.raises(ValueError):
        aut_sample_weighted([0.1, 0.2], [1])


def test_aut_matches_tesseract_metric_when_available():
    tm = pytest.importorskip("tesseract.metrics", reason="tesseract.metrics needs matplotlib")
    f = [0.9, 0.85, 0.7, 0.72]
    assert aut_unweighted(f) == pytest.approx(tm.aut(f))


def test_weighted_vs_unweighted_on_unequal_windows():
    # Jan: 400 rows, perfect (F1 1).  Feb: 100 rows, TP 25 FN 25 FP 0 (F1 2/3).
    # Mar: 1000 rows, TP 400 FN 100 FP 100 (F1 0.8).
    rows = (_block("2018-01", 200, 0, 200, 0) + _block("2018-02", 25, 25, 50, 0)
            + _block("2018-03", 400, 100, 400, 100))
    r, ctx = _run(_fn_model(), _handmade(rows), training_cutoff=CUTOFF, params={"min_window_samples": 50})
    f1 = [w["f1"] for w in r.details["windows"]]
    assert f1 == pytest.approx([1.0, 2 / 3, 0.8])
    assert [w["n"] for w in r.details["windows"]] == [400, 100, 1000]
    assert r.metrics["aut_f1"] == pytest.approx(((1 + 2 / 3) / 2 + (2 / 3 + 0.8) / 2) / 2)  # 0.78333
    expected_w = (400 * 1.0 + 100 * 2 / 3 + 100 * 2 / 3 + 1000 * 0.8) / ((400 + 100) + (100 + 1000))
    assert r.metrics["aut_f1_weighted"] == pytest.approx(expected_w)  # 0.83333
    assert r.metrics["aut_f1_weighted"] == pytest.approx(0.8333333333)
    assert r.checks[0].name == "aut_f1_weighted" and r.checks[0].value == pytest.approx(expected_w)
    assert r.checks[0].passed is True
    # the March window has 100/500 benign false positives (20% FPR): the operating-point check flags it
    assert r.checks[1].name == "window_fpr_within_budget" and r.checks[1].passed is False
    assert r.metrics["first_window"] == "2018-01" and r.metrics["last_window"] == "2018-03"
    assert r.metrics["f1_drop"] == pytest.approx(0.2)
    json.dumps({"r": r.to_dict(), "a": ctx.artifacts.to_dict()}, allow_nan=False)


def test_duplicate_rows_counted_once():
    rows = (_block("2018-01", 200, 0, 200, 0) + _block("2018-02", 25, 25, 50, 0)
            + _block("2018-03", 400, 100, 400, 100))
    c = _handmade(rows)
    two = lambda a: np.concatenate([a, a])  # noqa: E731
    d = Corpus(
        name=c.name, version=c.version, feature_version=c.feature_version, content_hash="doubled",
        manifest={**c.manifest, "n": 2 * c.n}, X=two(np.asarray(c.X)), sha256=two(c.sha256), label=two(c.label),
        timestamp=two(c.timestamp), split=two(c.split), synthetic=True,
    )
    params = {"min_window_samples": 50}
    r1, _ = _run(_fn_model(), c, training_cutoff=CUTOFF, params=params)
    r2, _ = _run(_fn_model(), d, training_cutoff=CUTOFF, params=params)
    assert r1.details["duplicates_removed"] == 0 and r2.details["duplicates_removed"] == 1500
    assert [w["n"] for w in r2.details["windows"]] == [400, 100, 1000]
    assert r2.metrics["aut_f1_weighted"] == pytest.approx(r1.metrics["aut_f1_weighted"])
    assert any("duplicate" in n for n in r2.notes)


# --------------------------------------------------------------------------------------------------
# known decay on the drifting toy corpus
# --------------------------------------------------------------------------------------------------


def test_known_decay(drifting, stable):
    (cd, bd), (cs, bs) = drifting, stable
    rd, ctx = _run(_lgbm(bd), cd, training_cutoff=CUTOFF, training_hashes=training_hashes_for(cd))
    rs, _ = _run(_lgbm(bs), cs, training_cutoff=CUTOFF, training_hashes=training_hashes_for(cs))
    for r in (rd, rs):
        assert r.metrics["n_windows"] == 12
        assert r.metrics["first_window"] == "2018-01" and r.metrics["last_window"] == "2018-12"
        assert r.metrics["effective_cutoff"] == "2017-12-31"
    assert rd.metrics["f1_drop"] > 0.15
    assert abs(rs.metrics["f1_drop"]) < 0.08
    assert rd.metrics["aut_f1_weighted"] < rs.metrics["aut_f1_weighted"] - 0.1
    assert rd.metrics["f1_last"] < rd.metrics["aut_f1_weighted"] < rd.metrics["f1_first"]
    # roughly equal windows => weighted ~ unweighted
    assert rd.metrics["aut_f1_weighted"] == pytest.approx(rd.metrics["aut_f1"], abs=0.01)
    # windows are chronological and all post-date the cutoff (C1)
    starts = [w["start"] for w in rd.details["windows"]]
    assert starts == sorted(starts) and starts[0] > "2017-12-31"
    assert rd.details["temporal_constraints"]["c1"]["holds"] is True
    assert {"drift.decay", "drift.window_sizes"} <= set(ctx.artifacts.keys_for("drift"))
    decay = ctx.artifacts.get("drift.decay")
    assert decay["series"][0]["label"] == "F1" and decay["series"][0]["x"][0] == 1.0
    json.dumps({"r": rd.to_dict(), "a": ctx.artifacts.to_dict()}, allow_nan=False)
    # a stricter drift policy flags the decaying model (warn gate => WARN, never FAIL)
    rf, _ = _run(_lgbm(bd), cd, training_cutoff=CUTOFF, params={"min_aut_f1": 0.85})
    assert rf.status is Status.WARN and rf.gate_outcome is GateOutcome.FAILED
    assert rf.score is not None and rf.score < 0.75
    assert "degrades" in rf.finding


def test_window_metrics_match_sklearn(drifting):
    from sklearn.metrics import f1_score, precision_score, recall_score

    c, b = drifting
    r, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF)
    idx = c.indices(role="temporal", after=CUTOFF)
    months = c.timestamp[idx].astype("datetime64[M]")
    for w in r.details["windows"]:
        sel = idx[months == np.datetime64(w["window"], "M")]
        y, p = c.label[sel], (b.predict(c.take(sel)) >= 0.5).astype(int)
        assert w["n"] == sel.size
        assert w["f1"] == pytest.approx(f1_score(y, p))
        assert w["precision"] == pytest.approx(precision_score(y, p, zero_division=0))
        assert w["recall"] == pytest.approx(recall_score(y, p))


# --------------------------------------------------------------------------------------------------
# cutoff handling
# --------------------------------------------------------------------------------------------------


def test_cutoff_after_corpus_end_is_unevaluable(small):
    c, b = small
    r, _ = _run(_lgbm(b), c, training_cutoff=dt.date(2019, 6, 30))
    assert r.status is Status.WARN  # never a pass
    assert r.gate_outcome is GateOutcome.NOT_EVALUATED
    assert r.checks[0].value is None and r.checks[0].passed is None
    assert r.score is None
    assert r.metrics["n_windows"] == 0 and r.metrics["aut_f1_weighted"] is None
    assert "newest dated sample" in r.finding and "no future samples" in r.finding
    assert "not passed" in r.finding


def test_members_after_declared_cutoff_move_effective_cutoff(small):
    c, b = small
    hashes = training_hashes_for(c)  # train split spans 2017-01..2017-12
    latest = c.timestamp[c.indices(include_hashes=hashes, labeled_only=False)].max()
    r, _ = _run(_lgbm(b), c, training_cutoff=dt.date(2017, 6, 30), training_hashes=hashes,
                params={"min_window_samples": 50})
    c1 = r.details["temporal_constraints"]["c1"]
    assert c1["adjusted"] is True and c1["members_after_declared_cutoff"] > 0
    assert r.metrics["effective_cutoff"] == str(latest)
    assert any("Inconsistent training_cutoff" in n for n in r.notes)
    # no evaluated window starts before the effective cutoff
    assert all(w["start"] > str(latest) for w in r.details["windows"])
    assert c1["holds"] is True


def test_no_manifest_note(small):
    c, b = small
    r, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF, params={"min_window_samples": 50})
    assert r.details["temporal_constraints"]["c1"]["members_in_corpus"] is None
    assert any("No training manifest" in n for n in r.notes)


def test_empty_declared_manifest_note(small):
    """Regression (xcomp-empty-manifest-reported-as-undeclared)."""
    c, b = small
    ctx = make_context(DriftModule, model=_lgbm(b), corpus=c, training_cutoff=CUTOFF,
                       params={"min_window_samples": 50})
    ctx.extras["training_manifest_declared"] = True
    r = DriftModule().run(ctx)
    assert any(n.startswith("The declared training manifest contains no sha256 hashes") for n in r.notes)


def test_skip_without_training_cutoff(small):
    c, b = small
    ctx = make_context(DriftModule, model=_lgbm(b), corpus=c)
    assert unmet_requirements(DriftModule, ctx.capabilities) == [Requirement.TRAINING_CUTOFF]
    r = DriftModule().run(ctx)
    assert r.status is Status.SKIPPED and r.gate_outcome is GateOutcome.NOT_EVALUATED
    assert r.skip_reason.startswith(REQUIREMENT_HINTS[Requirement.TRAINING_CUTOFF])
    assert "training_cutoff" in r.skip_reason


def test_skip_without_corpus(small):
    _, b = small
    ctx = make_context(DriftModule, model=_lgbm(b), corpus=None, training_cutoff=CUTOFF)
    assert Requirement.FEATURE_SPACE in unmet_requirements(DriftModule, ctx.capabilities)
    r = DriftModule().run(ctx)
    assert r.status is Status.SKIPPED
    assert r.skip_reason == REQUIREMENT_HINTS[Requirement.FEATURE_SPACE]


# --------------------------------------------------------------------------------------------------
# windows: too few, dropped, caps, granularity
# --------------------------------------------------------------------------------------------------


def test_single_window_is_unevaluable(small):
    c, b = small
    r, _ = _run(_lgbm(b), c, training_cutoff=dt.date(2018, 11, 30), params={"min_window_samples": 50})
    assert r.metrics["n_windows"] == 1 and r.metrics["first_window"] == "2018-12"
    assert r.metrics["f1_first"] is not None and r.metrics["aut_f1_weighted"] is None
    assert r.status is Status.WARN and r.gate_outcome is GateOutcome.NOT_EVALUATED
    assert r.checks[0].passed is None
    assert "AUT needs at least 2" in r.finding


def test_small_windows_dropped_and_listed(small):
    c, b = small
    # default min_window_samples=200 but the small toy corpus has ~150 per month
    r, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF)
    assert r.metrics["n_windows"] == 0
    assert len(r.details["dropped_windows"]) == 12
    assert all("min_window_samples" in d["reason"] for d in r.details["dropped_windows"])
    assert r.status is Status.WARN and r.checks[0].passed is None
    sizes = [w["n"] for w in r.details["windows"]]
    thr = sorted(sizes)[4]  # drop the 4 smallest windows
    r2, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF, params={"min_window_samples": thr})
    assert r2.metrics["n_windows"] == sum(s >= thr for s in sizes)
    assert len(r2.details["dropped_windows"]) == sum(s < thr for s in sizes)
    assert any("dropped" in n for n in r2.notes)


def test_require_both_classes():
    rows = (_block("2018-01", 90, 10, 100, 0) + _block("2018-02", 80, 20, 0, 0)  # Feb: malware only
            + _block("2018-03", 70, 30, 95, 5))
    c = _handmade(rows)
    r, _ = _run(_fn_model(), c, training_cutoff=CUTOFF, params={"min_window_samples": 50})
    assert r.metrics["n_windows"] == 2
    assert r.details["dropped_windows"] == [{"window": "2018-02", "n": 100, "reason": "no benign samples (require_both_classes)"}]
    r2, _ = _run(_fn_model(), c, training_cutoff=CUTOFF, params={"min_window_samples": 50, "require_both_classes": False})
    assert r2.metrics["n_windows"] == 3
    feb = next(w for w in r2.details["windows"] if w["window"] == "2018-02")
    assert feb["fpr"] is None and feb["recall"] == pytest.approx(0.8)
    # benign-only windows are always dropped (F1 undefined)
    rows3 = rows + _block("2018-04", 0, 0, 100, 0)
    r3, _ = _run(_fn_model(), _handmade(rows3), training_cutoff=CUTOFF,
                 params={"min_window_samples": 50, "require_both_classes": False})
    assert any(d["window"] == "2018-04" and "no malicious" in d["reason"] for d in r3.details["dropped_windows"])


def test_max_windows_and_subsampling(drifting):
    c, b = drifting
    r, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF, params={"max_windows": 3})
    assert r.metrics["n_windows"] == 3 and r.metrics["last_window"] == "2018-03"
    assert r.details["beyond_max_windows"]["windows"] == 9
    assert any("max_windows" in n for n in r.notes)
    p = {"max_samples_per_window": 220, "min_window_samples": 200}
    r1, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF, params=p)
    r2, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF, params=p)
    assert all(w["n"] <= 220 for w in r1.details["windows"])
    assert r1.details["subsampling"]["windows_subsampled"] > 0
    assert r1.metrics == r2.metrics  # deterministic given the seed


def test_quarter_granularity(drifting):
    c, b = drifting
    r, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF, params={"granularity": "quarter"})
    assert [w["window"] for w in r.details["windows"]] == ["2018Q1", "2018Q2", "2018Q3", "2018Q4"]
    assert r.metrics["n_windows"] == 4
    assert r.details["temporal_constraints"]["c2"]["holds"] is True


def test_c2_and_c3_flags():
    # Q1: malware all in January, goodware all in March => C2 misaligned.
    # Q2/Q3: aligned; Q3 has a very different malware ratio => C3 flag.
    rows = (_block("2018-01", 100, 0, 0, 0, mal_day=5) + _block("2018-03", 0, 0, 100, 0, ben_day=20)
            + _block("2018-05", 90, 10, 100, 0) + _block("2018-08", 180, 20, 40, 0))
    r, _ = _run(_fn_model(), _handmade(rows), training_cutoff=CUTOFF,
                params={"granularity": "quarter", "min_window_samples": 50})
    c2 = r.details["temporal_constraints"]["c2"]
    c3 = r.details["temporal_constraints"]["c3"]
    assert c2["windows_misaligned"] == ["2018Q1"] and c2["holds"] is False
    assert "2018Q3" in c3["windows_deviating"] and c3["holds"] is False
    assert any(n.startswith("C2:") for n in r.notes) and any(n.startswith("C3:") for n in r.notes)


@pytest.mark.parametrize(
    "params",
    [{"granularity": "fortnight"}, {"granularity": 3}, {"min_aut_f1": 1.5}, {"min_window_samples": 0},
     {"max_windows": 1}, {"max_samples_per_window": 50, "min_window_samples": 200}, {"require_both_classes": 1}],
)
def test_invalid_params(small, params):
    c, b = small
    ctx = make_context(DriftModule, model=_lgbm(b), corpus=c, training_cutoff=CUTOFF, params=params)
    with pytest.raises(ConfigError):
        DriftModule().run(ctx)


def test_plural_granularity_accepted(drifting):
    c, b = drifting
    r, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF, params={"granularity": "Months"})
    assert r.metrics["n_windows"] == 12


def test_default_gate_yaml_keys_are_params():
    cfg = load_config()
    assert set(cfg.module("drift").params()) <= set(DriftModule.default_params)
    assert DriftModule.default_gate.value == cfg.module("drift").gate.value == "warn"


# --------------------------------------------------------------------------------------------------
# TESSERACT equivalence (installed package vs vendored copy vs vectorized checks)
# --------------------------------------------------------------------------------------------------


def _random_dates(rng, n, start="2017-01-01", span=900):
    base = np.datetime64(start, "D")
    return base + rng.integers(0, span, size=n).astype("timedelta64[D]")


def test_c1_vectorized_matches_tesseract():
    tt, _ = load_temporal()
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = _random_dates(rng, int(rng.integers(1, 30)), span=200)
        b = _random_dates(rng, int(rng.integers(1, 30)), start="2017-06-01", span=200)
        expected = tt.assert_train_test_temporal_consistency(list(a.astype(object)), list(b.astype(object))) \
            if hasattr(tt, "assert_train_test_temporal_consistency") else bool(a.max() <= b.min())
        assert train_test_temporally_consistent(a, b) == expected


def test_vendored_matches_installed_partition():
    pytest.importorskip("tesseract.temporal")
    installed, desc = load_temporal()
    vendored, vdesc = load_temporal(prefer_installed=False)
    assert "installed" in desc and "vendored" in vdesc
    rng = np.random.default_rng(1)
    t = _random_dates(rng, 500).astype("datetime64[us]").astype(object)
    start = dt.datetime(2017, 1, 1)
    for gran in ("month", "quarter", "week", "year", "day"):
        a = installed.time_aware_indexes(t, 0, 1, gran, start_date=start)
        b = vendored.time_aware_indexes(t, 0, 1, gran, start_date=start)
        assert a == b
        assert installed.get_relative_delta(2, gran) == vendored.get_relative_delta(2, gran)


def test_module_runs_on_vendored_backend(drifting, monkeypatch):
    c, b = drifting
    r_inst, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF)
    monkeypatch.setattr(drift_mod, "load_temporal", lambda prefer_installed=True: (
        __import__("malvalid._vendor.tesseract_temporal", fromlist=["x"]), drift_mod._VENDORED_LABEL))
    r_vend, _ = _run(_lgbm(b), c, training_cutoff=CUTOFF)
    assert "vendored" in r_vend.details["tesseract"]
    assert r_vend.metrics == r_inst.metrics


def test_partition_mid_month_cutoff_aligns_to_cutoff(drifting):
    c, _ = drifting
    tt, _ = load_temporal()
    cut = dt.date(2018, 1, 14)
    idx = c.indices(role="temporal", after=cut)
    ws = partition_windows(tt, c, idx, cut, "month")
    assert ws[0].start == dt.date(2018, 1, 15) and ws[0].end == dt.date(2018, 2, 14)
    assert sum(w.idx_available.size for w in ws) == idx.size
    for w in ws:
        ts = c.timestamp[w.idx_available]
        assert ts.min() >= np.datetime64(w.start) and ts.max() <= np.datetime64(w.end)


def test_binary_counts_zero_division():
    m = binary_counts(np.array([1, 1, 0]), np.array([0.1, 0.2, 0.3]), 0.5)
    assert m["f1"] == 0.0 and m["precision"] == 0.0 and m["recall"] == 0.0 and m["fpr"] == 0.0
    m = binary_counts(np.array([1, 1]), np.array([0.9, 0.1]), 0.5)
    assert m["fpr"] is None and m["recall"] == 0.5


# --------------------------------------------------------------------------------------------------
# operating-point drift: per-window FPR / FNR with Wilson CIs, AUROC, budget check, granularity
# --------------------------------------------------------------------------------------------------


def test_auroc_matches_sklearn_and_handles_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500) * 0.5 + 0.3 * y, 1)  # many ties
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert auroc(np.ones(5), np.arange(5)) is None and auroc(np.zeros(5), np.arange(5)) is None


def test_operating_point_stats_ci_and_empty_classes():
    from malvalid.modules.performance import wilson_interval

    y = np.array([1] * 10 + [0] * 90)
    s = np.array([0.9] * 7 + [0.1] * 3 + [0.9] * 2 + [0.1] * 88)
    st = operating_point_stats(binary_counts(y, s, 0.5), y, s)
    assert st["fnr"] == pytest.approx(0.3) and st["fnr_ci95"] == wilson_interval(3, 10)
    assert st["fpr_ci95"] == wilson_interval(2, 90)
    assert 0.0 <= st["auroc"] <= 1.0
    y0 = np.zeros(20, dtype=int)
    st0 = operating_point_stats(binary_counts(y0, np.zeros(20), 0.5), y0, np.zeros(20))
    assert st0["fnr"] is None and st0["fnr_ci95"] is None and st0["auroc"] is None


def test_fpr_budget_summary_unevaluable_and_significance():
    w = [
        {"label": "a", "fp": 1, "n_ben": 10},        # too few benign: unevaluable
        {"label": "b", "fp": 105, "n_ben": 10000},   # 1.05%: over budget but CI includes 1%
        {"label": "c", "fp": 200, "n_ben": 10000},   # 2%: significantly over
    ]
    s = fpr_budget_summary(w, 0.01, 100)
    assert s["unevaluable_windows"] == ["a"] and s["n_evaluable"] == 2
    assert s["windows_over_budget"] == ["b", "c"] and s["windows_significantly_over_budget"] == ["c"]
    assert s["max_window_fpr"] == pytest.approx(0.02) and s["max_window_fpr_label"] == "c"
    assert s["value"] > 0.01
    ok = fpr_budget_summary(w[:2], 0.01, 100)
    assert ok["value"] < 0.01 and ok["windows_significantly_over_budget"] == []
    none = fpr_budget_summary(w[:1], 0.01, 100)
    assert none["value"] is None and none["max_window_fpr"] is None


def _fpr_rows(fp_by_month, n_ben=2000):
    rows = []
    for i, fp in enumerate(fp_by_month, start=1):
        rows += _block(f"2018-{i:02d}", 150, 50, n_ben - fp, fp)
    return rows


def test_window_fields_and_headline():
    c = _handmade(_fpr_rows([10, 20, 40]))  # FPR 0.5% -> 1% -> 2%; miss rate 25% throughout
    r, ctx = _run(_fn_model(), c, training_cutoff=CUTOFF, params={"min_window_samples": 50})
    w = r.details["windows"]
    assert [x["fpr"] for x in w] == pytest.approx([0.005, 0.01, 0.02])
    assert all(x["fnr"] == pytest.approx(0.25) for x in w)
    assert all(len(x["fpr_ci95"]) == 2 and x["fpr_ci95"][0] <= x["fpr"] <= x["fpr_ci95"][1] for x in w)
    assert all(x["auroc"] is not None for x in w)
    m = r.metrics
    assert m["fnr_first"] == pytest.approx(0.25) and m["fpr_last"] == pytest.approx(0.02)
    assert m["max_window_fpr"] == pytest.approx(0.02) and m["windows_over_fpr_budget"] == 1
    assert "Operating point: AUROC" in r.finding and "miss rate" in r.finding and "significantly above the budget" in r.finding
    assert r.details["operating_point"]["windows_significantly_over_budget"] == [w[2]["window"]]
    assert {"drift.window_fpr", "drift.window_miss_rate"} <= set(ctx.artifacts._items)
    labels = [s_["label"] for s_ in ctx.artifacts._items["drift.window_fpr"]["series"]]
    assert labels[0] == "FPR" and "FPR 95% CI upper" in labels


def test_fpr_check_is_ungraded_and_gates_only_when_significant():
    ok = _handmade(_fpr_rows([10, 12, 15]))  # 0.5-0.75%
    r, _ = _run(_fn_model(), ok, training_cutoff=CUTOFF, params={"min_window_samples": 50})
    chk = {c.name: c for c in r.checks}
    assert chk["window_fpr_within_budget"].passed is True and chk["window_fpr_within_budget"].score is None
    assert r.status is Status.PASS
    bad = _handmade(_fpr_rows([10, 12, 60]))  # 3% in the last window
    rb, _ = _run(_fn_model(), bad, training_cutoff=CUTOFF, params={"min_window_samples": 50})
    chk_b = {c.name: c for c in rb.checks}
    assert chk_b["window_fpr_within_budget"].passed is False
    # unscored: the axis score (and therefore the headline score) is the AUT-only value
    assert rb.score == pytest.approx(chk_b["aut_f1_weighted"].score)
    assert rb.status is Status.WARN


def test_fpr_check_unevaluable_with_few_benign():
    rows = _block("2018-01", 90, 10, 30, 0) + _block("2018-02", 90, 10, 30, 0)
    r, _ = _run(_fn_model(), _handmade(rows), training_cutoff=CUTOFF, params={"min_window_samples": 50})
    chk = {c.name: c for c in r.checks}["window_fpr_within_budget"]
    assert chk.value is None and chk.passed is None
    assert r.metrics["aut_f1_weighted"] is not None
    assert r.details["operating_point"]["unevaluable_windows"] == ["2018-01", "2018-02"]
    assert r.status is Status.WARN and "unassessed" in r.finding


def test_budget_read_from_performance_config():
    from types import SimpleNamespace

    from malvalid.config import GateConfig

    cfg = GateConfig.model_validate({"modules": {"performance": {"max_fpr": 0.001}}})
    assert DriftModule._budget(SimpleNamespace(config=cfg)) == pytest.approx(0.001)
    assert DriftModule._budget(SimpleNamespace(config=GateConfig())) == pytest.approx(0.01)


def _daily_rows(n_days, per_day=60, start="2018-01-01"):
    rows = []
    base = np.datetime64(start, "D")
    for d in range(n_days):
        for j in range(per_day):
            lab = j % 2
            rows.append((base + np.timedelta64(d, "D"), lab, 0.9 if lab else 0.1))
    return rows


def test_granularity_auto_selects_week_for_day_level_data():
    c = _handmade(_daily_rows(70))  # 70 days x 60 rows, weekly windows have 420 rows
    r, _ = _run(_fn_model(), c, training_cutoff=dt.date(2017, 12, 31), params={"granularity": "auto"})
    assert r.details["granularity"] == "week" and r.details["granularity_requested"] == "auto"
    assert r.metrics["n_windows"] == 10
    assert any("granularity=auto selected 'week'" in n for n in r.notes)
    rm, _ = _run(_fn_model(), c, training_cutoff=dt.date(2017, 12, 31), params={"granularity": "month"})
    assert rm.details["granularity"] == "month" and rm.metrics["n_windows"] < 10


def test_granularity_auto_falls_back_to_month():
    # month-level timestamps (everything on the 1st)
    rows = []
    for m in (1, 2, 3, 4):
        base = np.datetime64(f"2018-{m:02d}-01", "D")
        rows += [(base, j % 2, 0.9 if j % 2 else 0.1) for j in range(400)]
    r, _ = _run(_fn_model(), _handmade(rows), training_cutoff=dt.date(2017, 12, 31), params={"granularity": "auto"})
    assert r.details["granularity"] == "month" and r.details["granularity_auto"]["day_level_timestamps"] is False
    # day-level but weekly windows too small for min_window_samples
    r2, _ = _run(_fn_model(), _handmade(_daily_rows(70, per_day=10)), training_cutoff=dt.date(2017, 12, 31),
                 params={"granularity": "auto"})
    assert r2.details["granularity"] == "month" and "month" in r2.details["granularity_auto"]["reason"]
    # too many weekly windows for max_windows
    r3, _ = _run(_fn_model(), _handmade(_daily_rows(70)), training_cutoff=dt.date(2017, 12, 31),
                 params={"granularity": "auto", "max_windows": 5})
    assert r3.details["granularity"] == "month"


def test_granularity_auto_is_valid_param_and_default_stays_month():
    DriftModule.validate_params({**DriftModule.default_params, "granularity": "auto"})
    assert DriftModule.default_params["granularity"] == "month"
    with pytest.raises(ConfigError):
        DriftModule.validate_params({**DriftModule.default_params, "granularity": "hourly"})
    with pytest.raises(ConfigError):
        DriftModule.validate_params({**DriftModule.default_params, "min_window_benign": 0})
