"""Unit tests for the synthetic demo/CI corpora (malvalid.corpora.synthetic: synthetic_v2 / synthetic_v3)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from malvalid.config import GateConfig
from malvalid.core import CorpusUnavailable
from malvalid.corpora import synthetic as S
from malvalid.corpora.base import load_corpus_dir
from malvalid.schemas.ember_v2 import EmberV2Schema
from malvalid.schemas.ember_v3 import EmberV3Schema

SCHEMAS = {"ember_v2": EmberV2Schema(), "ember_v3": EmberV3Schema()}
SMALL = S.SyntheticParams(n=2000)


@pytest.fixture(scope="module")
def default_data() -> dict[str, tuple[S.SyntheticData, float]]:
    """The two default-size corpora (20,000 rows each) and their generation times."""
    out = {}
    for fv in ("ember_v2", "ember_v3"):
        t0 = time.monotonic()
        d = S.generate(fv)
        out[fv] = (d, time.monotonic() - t0)
    return out


def _group(fv: str, name: str) -> slice:
    g = {x.name: x for x in SCHEMAS[fv].groups()}[name]
    return slice(g.start, g.stop)


def _col(fv: str, name: str) -> int:
    return SCHEMAS[fv].feature_names().index(name)


# --------------------------------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fv", ["ember_v2", "ember_v3"])
def test_default_generation_is_fast_and_in_the_real_layout(default_data: Any, fv: str) -> None:
    d, secs = default_data[fv]
    assert secs < 30.0, f"generation took {secs:.1f}s"
    schema = SCHEMAS[fv]
    schema.check_layout()
    assert d.X.shape == (20000, schema.dim) and d.X.dtype == np.float32
    assert np.isfinite(d.X).all()
    assert d.X.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("fv", ["ember_v2", "ember_v3"])
def test_generation_is_deterministic(fv: str) -> None:
    a, b = S.generate(fv, SMALL), S.generate(fv, SMALL)
    for f in ("X", "sha256", "label", "timestamp", "split"):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))
    c = S.generate(fv, S.SyntheticParams(n=2000, seed=1))
    assert not np.array_equal(a.X, c.X)
    assert set(a.sha256.tolist()).isdisjoint(c.sha256.tolist())


@pytest.mark.parametrize("fv", ["ember_v2", "ember_v3"])
def test_histograms_are_normalised_and_counts_integral(default_data: Any, fv: str) -> None:
    X = default_data[fv][0].X
    for grp in ("histogram", "byteentropy"):
        block = X[:, _group(fv, grp)]
        assert block.min() >= 0
        np.testing.assert_allclose(block.sum(1), 1.0, atol=1e-5)
    counts = ["strings.numstrings", "strings.printables", "general.size"]
    counts += (["section.num_sections", "general.imports", "general.exports"] if fv == "ember_v2" else
               ["section.n_sections", "imports.n_functions", "imports.n_libraries", "pefilewarnings.count",
                "richheader.n_pairs"])
    for nm in counts:
        v = X[:, _col(fv, nm)]
        assert np.all(v >= 0) and np.array_equal(v, np.round(v)), nm
    assert X[:, _col(fv, "section.num_sections" if fv == "ember_v2" else "section.n_sections")].min() >= 1
    # printables = numstrings x avlength (the real relationship), up to rounding to whole characters
    ns, av, pr = (X[:, _col(fv, f"strings.{k}")].astype(np.float64) for k in ("numstrings", "avlength", "printables"))
    np.testing.assert_allclose(pr, ns * av, rtol=1e-3, atol=1.0)


def test_hashed_tokens_use_the_real_vectorizer_buckets() -> None:
    from malvalid.schemas import _thrember_port as tp

    x = tp.vectorize({"imports": {"KERNEL32.dll": ["CreateFileA"]}})
    lib = x[tp.OFFSETS["imports"] + 2 : tp.OFFSETS["imports"] + 258]
    fn = x[tp.OFFSETS["imports"] + 258 : tp.OFFSETS["imports"] + 1282]
    assert int(np.flatnonzero(lib)[0]) == S.hash_bucket("kernel32.dll", 256)[0]
    assert int(np.flatnonzero(fn)[0]) == S.hash_bucket("kernel32.dll:CreateFileA", 1024)[0]


@pytest.mark.parametrize("fv,start", [("ember_v2", "2017-01"), ("ember_v3", "2022-01")])
def test_splits_months_labels_and_hashes(default_data: Any, fv: str, start: str) -> None:
    d = default_data[fv][0]
    months = d.timestamp.astype("datetime64[M]")
    first = np.datetime64(start, "M")
    assert months.min() == first and months.max() == first + 35
    assert np.unique(months).size == 36
    assert np.all(np.diff(d.timestamp.astype(np.int64)) >= 0)  # chronological
    by = {s: d.split == s for s in S.SPLITS}
    assert set(np.unique(d.split).tolist()) == set(S.SPLITS)
    for s in ("train", "holdout"):
        assert months[by[s]].min() == first and months[by[s]].max() == first + 11
    for s in ("test", "challenge"):
        assert months[by[s]].min() == first + 12 and months[by[s]].max() == first + 35
    assert 0.15 < by["holdout"].sum() / (by["train"].sum() + by["holdout"].sum()) < 0.25
    assert set(np.unique(d.label).tolist()) == {0, 1}
    assert np.all(d.label[by["challenge"]] == 1)
    assert 0.45 < d.label[by["test"]].mean() < 0.55
    assert np.unique(d.sha256).size == d.X.shape[0]
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in d.sha256[:50])


# --------------------------------------------------------------------------------------------------
# Model quality: a LightGBM trained on train must be good but not perfect, and decay over time
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def models(default_data: Any) -> dict[str, Any]:
    """A LightGBM per feature space, trained on the ``train`` split only."""
    lgb = pytest.importorskip("lightgbm")
    out = {}
    for fv, (d, _) in default_data.items():
        tr = d.split == "train"
        out[fv] = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1, n_jobs=4, verbose=-1,
                                     random_state=0, importance_type="gain").fit(d.X[tr], d.label[tr])
    return out


@pytest.mark.parametrize("fv", ["ember_v2", "ember_v3"])
def test_lightgbm_reaches_realistic_auroc_and_f1_decays(default_data: Any, models: dict[str, Any], fv: str) -> None:
    from sklearn.metrics import f1_score, roc_auc_score

    d = default_data[fv][0]
    model = models[fv]
    te = d.split == "test"
    p = model.predict_proba(d.X[te])[:, 1]
    auroc = roc_auc_score(d.label[te], p)
    assert 0.95 <= auroc <= 0.995, auroc
    months = d.timestamp[te].astype("datetime64[M]")
    f1 = np.array([f1_score(d.label[te][months == m], p[months == m] >= 0.5) for m in np.unique(months)])
    assert f1.size == 24
    assert f1[:6].mean() - f1[-6:].mean() >= 0.03, f1.round(3)
    assert np.polyfit(np.arange(f1.size), f1, 1)[0] < 0  # downward trend
    ch = d.split == "challenge"
    det_test = (p[d.label[te] == 1] >= 0.5).mean()
    det_challenge = (model.predict_proba(d.X[ch])[:, 1] >= 0.5).mean()
    assert det_challenge < det_test - 0.2  # the challenge split is evasive


@pytest.mark.parametrize("fv", ["ember_v2", "ember_v3"])
def test_label_signal_is_spread_over_several_groups(models: dict[str, Any], fv: str) -> None:
    gain = models[fv].booster_.feature_importance("gain")
    share = gain / gain.sum()
    groups = {g.name: float(share[g.start : g.stop].sum()) for g in SCHEMAS[fv].groups()}
    assert sum(v >= 0.03 for v in groups.values()) >= 5, groups
    assert max(groups.values()) < 0.6, groups


# --------------------------------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def small_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(S.ENV_ROWS, "2000")
    monkeypatch.delenv(S.ENV_SEED, raising=False)
    monkeypatch.setenv("MALVALID_CORPUS_DIR", str(tmp_path / "corpora"))
    return tmp_path


def test_registry_and_identity() -> None:
    from malvalid import registry

    v2, v3 = registry.get_corpus_provider("synthetic_v2"), registry.get_corpus_provider("synthetic_v3")
    assert isinstance(v2, S.SyntheticEmberV2Provider) and isinstance(v3, S.SyntheticEmberV3Provider)
    assert (v2.feature_version, v3.feature_version) == ("ember_v2", "ember_v3")
    for p in (v2, v3):
        assert p.synthetic is True and p.is_available() and p.expected_content_hash is None
        assert "SYNTHETIC" in p.description and p.info()["synthetic"] is True


@pytest.mark.parametrize("cls", [S.SyntheticEmberV2Provider, S.SyntheticEmberV3Provider])
def test_load_generates_caches_and_labels_synthetic(small_rows: Path, cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
    p = cls()
    d = p.locate()
    assert d == small_rows / "corpora" / f"{p.name}-n2000-seed0"
    c = p.load()
    assert c.synthetic is True and c.manifest["synthetic"] is True and c.summary()["synthetic"] is True
    assert (c.n, c.dim) == (2000, SCHEMAS[p.feature_version].dim) and c.feature_version == p.feature_version
    assert c.roles() == S.ROLES and c.eval_indices(1).size > 0 and c.eval_indices(0).size > 0
    assert "SYNTHETIC" in c.manifest["source"]["note"]
    assert (d / "manifest.json").exists()
    # in-memory generation gives the same bytes, hence the same content hash
    assert p.in_memory().content_hash == c.content_hash
    # second load reuses the cache without regenerating
    monkeypatch.setattr(S, "generate", lambda *a, **k: pytest.fail("regenerated a cached corpus"))
    c2 = p.load()
    assert c2.content_hash == c.content_hash and c2.synthetic is True


def test_build_to_explicit_directory(tmp_path: Path) -> None:
    m = S.SyntheticEmberV3Provider().build(None, tmp_path / "out", n=1000, seed=3)
    c = load_corpus_dir(tmp_path / "out")  # verifies file hashes
    assert m["synthetic"] is True and c.synthetic is True and c.n == 1000
    assert m["extra"]["params"]["seed"] == 3 and m["extra"]["generator_version"] == S.GENERATOR_VERSION
    again = S.SyntheticEmberV3Provider().build(None, tmp_path / "again", n=1000, seed=3)
    assert again["content_hash"] == m["content_hash"]
    assert not any(p.name.startswith(".out.tmp") for p in tmp_path.iterdir())


def test_config_corpus_dir_is_used(tmp_path: Path, small_rows: Path) -> None:
    c = S.SyntheticEmberV2Provider().load(GateConfig(corpus_dir=str(tmp_path / "mine")))
    assert c.path == tmp_path / "mine" and c.n == 2000


def test_falls_back_to_memory_when_cache_is_unwritable(small_rows: Path, monkeypatch: pytest.MonkeyPatch,
                                                       caplog: pytest.LogCaptureFixture) -> None:
    p = S.SyntheticEmberV3Provider()

    def fail(*a: Any, **k: Any) -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(p, "build", fail)
    with caplog.at_level("WARNING", logger="malvalid.corpora.synthetic"):
        c = p.load()
    assert c.path is None and c.synthetic is True and c.n == 2000
    assert "in memory" in caplog.text


def test_refuses_a_non_synthetic_directory(tmp_path: Path) -> None:
    d = tmp_path / "real"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"name": "ember_v2_2018", "synthetic": False}))
    with pytest.raises(CorpusUnavailable, match="not the synthetic corpus"):
        S.SyntheticEmberV2Provider().load(GateConfig(corpus_dir=str(d)))


def test_stale_generator_version_is_regenerated(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = S.SyntheticEmberV3Provider()
    d = tmp_path / "c"
    m = p.build(None, d, n=500, seed=0)
    stale = dict(m, extra=dict(m["extra"], generator_version="0"))
    (d / "manifest.json").write_text(json.dumps(stale))
    with caplog.at_level("WARNING", logger="malvalid.corpora.synthetic"):
        c = p.load(GateConfig(corpus_dir=str(d)))
    assert "regenerating" in caplog.text
    assert c.content_hash == m["content_hash"] and c.n == 500


def test_parameter_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="at least 200 rows"):
        S.SyntheticParams(n=10).validate()
    with pytest.raises(ValueError, match="train_months"):
        S.SyntheticParams(months=12, train_months=12).validate()
    with pytest.raises(ValueError, match="unknown synthetic feature version"):
        S.generate("ember_v9", SMALL)
    monkeypatch.setenv(S.ENV_ROWS, "many")
    with pytest.raises(CorpusUnavailable, match="not an integer"):
        S.SyntheticEmberV3Provider.params()
