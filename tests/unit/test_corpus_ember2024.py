"""Unit tests for the canonical EMBER2024 corpus provider (malvalid.corpora.ember2024).

Fast tests build tiny corpora from generated EMBER v3 raw rows laid out like the published
release: weekly ``<start>_<end>_<Win32|Win64>_test.jsonl`` files inside ``Win32_test.zip`` /
``Win64_test.zip`` (each sample listed twice, the second copy carrying extra capa data, exactly as
in the real files) and ``<start>_<end>_challenge_malicious.jsonl`` files (mixed file types) inside
``challenge.zip``.

Tests marked ``ember`` use the real canonical corpus and are skipped unless it is available:

* ``MALVALID_EMBER2024_DIR``: the built corpus directory, else ``$MALVALID_CORPUS_DIR/ember_v3_2024``;
* ``MALVALID_EMBER2024_SOURCE``: optional, the directory holding the three published zips
  (enables the re-vectorisation spot check);
* ``MALVALID_EMBER2024_MODEL``: optional, the published ``EMBER2024_PE.model`` (enables the slow
  AUROC correctness gate).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from malvalid.config import GateConfig
from malvalid.core import CorpusUnavailable
from malvalid.corpora import ember2024 as M
from malvalid.corpora.base import load_corpus_dir
from malvalid.schemas import _thrember_port as tp

W1 = "2024-09-22_2024-09-28"
W2 = "2024-09-29_2024-10-05"
C1 = "2023-09-24_2023-09-30"
C2 = "2024-10-06_2024-10-12"

# --------------------------------------------------------------------------------------------------
# Generated source files
# --------------------------------------------------------------------------------------------------

_SECTIONS = [".text", ".rdata", ".data", ".rsrc", ".reloc", "UPX0", "UPX1", ""]
_PROPS = ["CNT_CODE", "CNT_INITIALIZED_DATA", "MEM_EXECUTE", "MEM_READ", "MEM_WRITE"]
_LIBS = {"KERNEL32.dll": ["CreateFileA", "ReadFile", "ExitProcess"], "USER32.dll": ["MessageBoxW"],
         "ADVAPI32.dll": ["RegOpenKeyExW", "RegSetValueExW"], "WS2_32.dll": ["connect", "send"]}


def _epoch(day: str, hour: int = 13) -> int:
    d = dt.datetime.fromisoformat(day).replace(hour=hour, tzinfo=dt.timezone.utc)
    return int(d.timestamp())


def raw_row(i: int, *, label: int, day: str, file_type: str = "Win32") -> dict[str, Any]:
    """A structurally valid EMBER2024 JSONL row whose feature values depend only on ``i``."""
    rng = np.random.default_rng(7000 + i)
    secs = []
    for nm in rng.choice(_SECTIONS, int(rng.integers(1, 5)), replace=False):
        size = int(rng.integers(0, 8) * 512)
        secs.append({"name": str(nm), "size": size, "entropy": float(np.round(rng.uniform(0, 8), 6)),
                     "vsize": int(rng.integers(1, 9) * 4096), "size_ratio": float(np.round(rng.uniform(0, 1), 6)),
                     "vsize_ratio": float(np.round(rng.uniform(0, 2), 6)),
                     "props": [str(p) for p in rng.choice(_PROPS, int(rng.integers(1, 4)), replace=False)]})
    libs = [str(x) for x in rng.choice(list(_LIBS), int(rng.integers(0, 4)), replace=False)]
    dist = rng.integers(0, 40, 96).tolist()
    return {
        "md5": f"{i:032x}",
        "sha1": f"{i:040x}",
        "sha256": f"{i:064x}",
        "first_submission_date": _epoch(day, hour=int(rng.integers(0, 24))),
        "label": label,
        "file_type": file_type,
        "general": {"size": int(rng.integers(1000, 10**6)), "entropy": float(rng.uniform(0, 8)), "is_pe": 1,
                    "start_bytes": [77, 90, 144, 0]},
        "histogram": rng.integers(0, 500, 256).tolist(),
        "byteentropy": rng.integers(0, 500, 256).tolist(),
        "strings": {"numstrings": int(rng.integers(0, 900)), "avlength": float(rng.uniform(5, 30)),
                    "printabledist": dist, "printables": int(sum(dist)), "entropy": float(rng.uniform(0, 6.6)),
                    "string_counts": {"file": int(rng.integers(0, 9)), "registry": int(rng.integers(0, 3))}},
        "header": {"coff": {"timestamp": int(rng.integers(0, 2**31)), "machine": "IMAGE_FILE_MACHINE_I386",
                            "characteristics": ["EXECUTABLE_IMAGE"]},
                   "optional": {"subsystem": "IMAGE_SUBSYSTEM_WINDOWS_GUI", "sizeof_code": int(rng.integers(0, 10**5)),
                                "dll_characteristics": ["NX_COMPAT"]},
                   "dos": {"e_magic": 23117, "e_lfanew": 256}},
        "section": {"entry": secs[0]["name"], "sections": secs,
                    "overlay": {"size": 0, "size_ratio": 0, "entropy": 0}},
        "imports": {lib: _LIBS[lib] for lib in libs},
        "exports": ["DllMain"] if i % 5 == 0 else [],
        "richheader": [int(rng.integers(0, 2**20)), int(rng.integers(1, 50))] if i % 3 else [],
        "pefilewarnings": [],
    }


def _write_jsonl(rows: list[dict[str, Any]], *, doubled: bool) -> bytes:
    """Serialise like the published weekly test files: every row, then every row again + capa data."""
    lines = [json.dumps(r) for r in rows]
    if doubled:
        lines += [json.dumps({**r, "capa": {"attack": ["DISCOVERY::File and Directory Discovery"]}}) for r in rows]
    return ("\n".join(lines) + "\n").encode()


def make_source(root: Path, *, challenge: bool = True, dup_across_split: bool = False) -> dict[str, Any]:
    """Write a miniature EMBER2024 release under ``root``; returns the expected corpus contents."""
    root.mkdir(parents=True, exist_ok=True)
    w1_32 = [raw_row(i, label=i % 2, day="2024-09-23") for i in range(0, 12)]
    w2_32 = [raw_row(i, label=i % 2, day="2024-10-01") for i in range(100, 110)]
    w2_32.append(dict(w1_32[4], first_submission_date=_epoch("2024-10-02")))  # benign file again in week 2
    w1_64 = [raw_row(i, label=i % 2, day="2024-09-25", file_type="Win64") for i in range(200, 206)]
    with zipfile.ZipFile(root / "Win32_test.zip", "w") as zf:
        zf.writestr(f"{W1}_Win32_test.jsonl", _write_jsonl(w1_32, doubled=True))
        zf.writestr(f"{W2}_Win32_test.jsonl", _write_jsonl(w2_32, doubled=True))
        zf.writestr("2024-09-15_2024-09-21_Win32_train.jsonl", _write_jsonl([raw_row(999, label=1, day="2024-09-16")], doubled=False))
    with zipfile.ZipFile(root / "Win64_test.zip", "w") as zf:
        zf.writestr(f"{W1}_Win64_test.jsonl", _write_jsonl(w1_64, doubled=True))
        zf.writestr(f"{W1}_PDF_test.jsonl", _write_jsonl([raw_row(998, label=1, day="2024-09-23", file_type="PDF")], doubled=False))
    ch1: list[dict[str, Any]] = []
    ch2: list[dict[str, Any]] = []
    if challenge:
        ch1 = [raw_row(300, label=1, day="2023-09-25"), raw_row(301, label=1, day="2023-09-26", file_type="Dot_Net"),
               raw_row(302, label=1, day="2023-09-27", file_type="APK")]  # APK is not a PE: dropped
        ch2 = [raw_row(310, label=1, day="2024-10-07", file_type="Win64")]
        if dup_across_split:
            ch2.append(w1_32[1])  # a malicious test file also listed in the challenge set
        with zipfile.ZipFile(root / "challenge.zip", "w") as zf:
            zf.writestr(f"{C1}_challenge_malicious.jsonl", _write_jsonl(ch1, doubled=False))
            zf.writestr(f"{C2}_challenge_malicious.jsonl", _write_jsonl(ch2, doubled=False))
    test_rows = w1_32 + w1_64 + [r for r in w2_32 if r["sha256"] != w1_32[4]["sha256"]]
    challenge_rows = [r for r in ch1 + ch2 if r["file_type"] in M.DEFAULT_CHALLENGE_FILE_TYPES]
    return {"test": test_rows, "challenge": challenge_rows}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = tmp_path_factory.mktemp("ember2024")
    expected = make_source(root / "src")
    manifest = M.Ember2024Provider().build(root / "src", root / "out", workers=2, chunk_rows=4)
    return expected, manifest, root


# --------------------------------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------------------------------


def test_discover_members_selects_pe_test_and_challenge(tmp_path: Path) -> None:
    make_source(tmp_path)
    members = M.discover_members(tmp_path)
    names = [m.name for m in members]
    assert names == [f"{W1}_Win32_test.jsonl", f"{W1}_Win64_test.jsonl", f"{W2}_Win32_test.jsonl",
                     f"{C1}_challenge_malicious.jsonl", f"{C2}_challenge_malicious.jsonl"]
    assert [m.split for m in members] == ["test"] * 3 + ["challenge"] * 2
    assert [m.file_type for m in members] == ["Win32", "Win64", "Win32", None, None]
    assert all(m.in_zip for m in members)
    only32 = M.discover_members(tmp_path, test_file_types=("Win32",))
    assert [m.name for m in only32 if m.split == "test"] == [f"{W1}_Win32_test.jsonl", f"{W2}_Win32_test.jsonl"]


def test_discover_members_counts_each_weekly_file_once(tmp_path: Path) -> None:
    """A zip plus its extracted copy (even extracted twice) must not be read twice."""
    make_source(tmp_path)
    for sub in ("extracted", "extracted_again"):
        with zipfile.ZipFile(tmp_path / "Win32_test.zip") as zf:
            zf.extract(f"{W1}_Win32_test.jsonl", tmp_path / sub)
    members = M.discover_members(tmp_path)
    w1 = [m for m in members if m.name.endswith(f"{W1}_Win32_test.jsonl")]
    assert len(w1) == 1 and not w1[0].in_zip  # the loose file wins over the zip member
    assert len(members) == 5


def test_discover_members_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        M.discover_members(tmp_path / "missing")
    (tmp_path / "Win32_test.zip").write_bytes(b"PK\x03\x04 truncated download")
    with pytest.raises(ValueError, match="incomplete download"):
        M.discover_members(tmp_path)


def test_plan_dedup_keeps_first_occurrence_per_split() -> None:
    mk = lambda name, split: M.SourceMember(name, split, None, "x", False)  # noqa: E731
    members = [mk("a", "test"), mk("b", "test"), mk("c", "challenge")]
    hashes = [["h1", "h2", "h1"], ["h2", "h3"], ["h1"]]
    keeps, digs, dropped, per_split = M.plan_dedup(members, hashes)
    assert [k.tolist() for k in keeps] == [[True, True, False], [False, True], [True]]
    assert [d.tolist() for d in digs] == [[True, True, True], [True, False], [False]]
    assert dropped == [(0, 2, 0, 0), (1, 0, 0, 1)]
    assert per_split == {"test": 2, "challenge": 0}


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------


def test_build_rows_labels_timestamps_and_features(built: tuple[dict[str, Any], dict[str, Any], Path]) -> None:
    expected, manifest, root = built
    c = load_corpus_dir(root / "out")
    rows = expected["test"] + expected["challenge"]
    assert c.n == len(rows) == manifest["n"] and c.dim == tp.DIM == 2568
    assert c.sha256.tolist() == [r["sha256"] for r in rows]
    assert np.unique(c.sha256).size == c.n
    assert c.split.tolist() == ["test"] * len(expected["test"]) + ["challenge"] * len(expected["challenge"])
    assert c.label.tolist() == [r["label"] for r in rows]
    days = [np.datetime64(dt.datetime.fromtimestamp(r["first_submission_date"], dt.timezone.utc).date()) for r in rows]
    assert c.timestamp.tolist() == [d.astype(object) for d in days]
    np.testing.assert_array_equal(c.take(np.arange(c.n)), tp.vectorize_batch(rows))
    assert c.synthetic is False and c.feature_version == "ember_v3"
    assert c.roles() == {"eval": ["test"], "challenge": ["challenge"], "temporal": ["test", "challenge"], "pool": ["test"]}
    assert set(c.indices(role="challenge").tolist()) == set(range(len(expected["test"]), c.n))
    assert np.all(c.label[c.indices(role="challenge", labeled_only=False)] == 1)


def test_build_manifest_records_provenance(built: tuple[dict[str, Any], dict[str, Any], Path]) -> None:
    expected, manifest, _ = built
    ex = manifest["extra"]
    assert manifest["name"] == "ember_v3_2024" and manifest["synthetic"] is False
    assert manifest["version"] == "2+custom" and ex["canonical"] is False  # a fixture is never canonical
    # every test row is listed twice; the benign file repeated in week 2 drops both of its copies there
    assert ex["duplicates_dropped"] == {"test": len(expected["test"]) + 2, "challenge": 0}
    assert ex["duplicate_feature_conflicts"] == 0
    assert ex["rows_by_split_and_file_type"] == {"test": {"Win32": 22, "Win64": 6},
                                                 "challenge": {"Dot_Net": 1, "Win32": 1, "Win64": 1}}
    assert ex["sha256_in_both_test_and_challenge"] == 0
    src = manifest["source"]
    assert src["dataset"] == "EMBER2024" and src["zips"] == ["Win32_test.zip", "Win64_test.zip", "challenge.zip"]
    assert "first_submission_date" in src["timestamp_field"]


def test_build_is_deterministic_across_worker_counts(built: tuple[dict[str, Any], dict[str, Any], Path]) -> None:
    _, manifest, root = built
    again = M.Ember2024Provider().build(root / "src", root / "out1", workers=1, chunk_rows=1000)
    assert again["files"] == manifest["files"]
    assert again["content_hash"] == manifest["content_hash"]


def test_build_refuses_duplicate_files_across_splits(tmp_path: Path) -> None:
    make_source(tmp_path / "src", dup_across_split=True)
    with pytest.raises(RuntimeError, match="distinct sha256"):
        M.Ember2024Provider().build(tmp_path / "src", tmp_path / "out", workers=1)


def test_build_without_challenge_zip(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    expected = make_source(tmp_path / "src", challenge=False)
    with caplog.at_level("WARNING", logger="malvalid.corpora.ember2024"):
        m = M.Ember2024Provider().build(tmp_path / "src", tmp_path / "out", workers=1)
    assert "without challenge split" in caplog.text
    c = load_corpus_dir(tmp_path / "out")
    assert c.n == len(expected["test"]) and set(c.split.tolist()) == {"test"}
    assert m["roles"]["challenge"] == [] and m["roles"]["temporal"] == ["test"]
    assert c.indices(role="challenge").size == 0


def test_build_limit_and_file_type_options(tmp_path: Path) -> None:
    make_source(tmp_path / "src")
    m = M.Ember2024Provider().build(tmp_path / "src", tmp_path / "out", workers=1, limit_per_member=3,
                                    test_file_types=("Win64",))
    c = load_corpus_dir(tmp_path / "out")
    assert m["extra"]["rows_by_split_and_file_type"]["test"] == {"Win64": 3}
    assert c.indices(role="eval").size == 3 and m["extra"]["limit_per_member"] == 3


def test_build_errors_are_actionable(tmp_path: Path) -> None:
    p = M.Ember2024Provider()
    with pytest.raises(ValueError, match="hf download"):
        p.build(None, tmp_path / "out")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no EMBER2024 test feature files"):
        p.build(tmp_path / "empty", tmp_path / "out")


# --------------------------------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------------------------------


def test_provider_identity_and_registry() -> None:
    from malvalid import registry

    p = registry.get_corpus_provider("ember_v3_2024")
    assert isinstance(p, M.Ember2024Provider)
    assert p.name == "ember_v3_2024" and p.feature_version == "ember_v3" and p.synthetic is False
    assert p.expected_content_hash is not None and len(p.expected_content_hash) == 64
    info = p.info()
    assert info["expected_content_hash"] == p.expected_content_hash


def test_unavailable_hint_explains_download_and_build(tmp_path: Path) -> None:
    p = M.Ember2024Provider()
    cfg = GateConfig(corpus_dir=str(tmp_path / "nowhere"))
    assert not p.is_available(cfg)
    with pytest.raises(CorpusUnavailable) as ei:
        p.load(cfg)
    msg = str(ei.value)
    for needle in ("huggingface.co/datasets/joyce8/EMBER2024", "Win32_test.zip", "challenge.zip",
                   "malvalid corpus build ember_v3_2024", "MALVALID_CORPUS_DIR"):
        assert needle in msg


def test_provider_load_refuses_non_canonical_build(built: tuple[dict[str, Any], dict[str, Any], Path]) -> None:
    _, _, root = built
    with pytest.raises(CorpusUnavailable, match="pinned"):
        M.Ember2024Provider().load(GateConfig(corpus_dir=str(root / "out")))


# --------------------------------------------------------------------------------------------------
# Real canonical corpus (skipped unless available)
# --------------------------------------------------------------------------------------------------

REAL_N = 484_855
REAL_TEST = {"n": 479_987, "malicious": 240_000, "benign": 239_987, "unlabeled": 0}
REAL_CHALLENGE = {"n": 4_868, "malicious": 4_868, "benign": 0, "unlabeled": 0}


def _real_corpus_dir() -> Path | None:
    if os.environ.get("MALVALID_EMBER2024_DIR"):
        return Path(os.environ["MALVALID_EMBER2024_DIR"])
    if os.environ.get("MALVALID_CORPUS_DIR"):
        return Path(os.environ["MALVALID_CORPUS_DIR"]) / "ember_v3_2024"
    return None


@pytest.fixture(scope="module")
def real_corpus() -> Any:
    d = _real_corpus_dir()
    if d is None or not (d / "manifest.json").exists():
        pytest.skip("canonical ember_v3_2024 corpus not available (set MALVALID_EMBER2024_DIR)")
    return M.Ember2024Provider().load(GateConfig(corpus_dir=str(d)))  # verifies the pinned hash


@pytest.mark.ember
def test_real_corpus_shape_splits_labels_hash(real_corpus: Any) -> None:
    c = real_corpus
    assert c.content_hash == M.Ember2024Provider.expected_content_hash
    assert (c.n, c.dim) == (REAL_N, 2568) and c.X.dtype == np.float32
    s = c.summary()["splits"]
    assert s["test"] == REAL_TEST and s["challenge"] == REAL_CHALLENGE
    assert np.all(c.split[: REAL_TEST["n"]] == "test") and np.all(c.split[REAL_TEST["n"]:] == "challenge")
    assert np.unique(c.sha256).size == c.n
    assert not np.isnat(c.timestamp).any()
    test_ts = c.timestamp[c.split == "test"]
    assert test_ts.min() == np.datetime64("2024-09-22") and test_ts.max() == np.datetime64("2024-12-14")
    ch_ts = c.timestamp[c.split == "challenge"]
    assert ch_ts.min() >= np.datetime64("2023-09-24") and ch_ts.max() <= np.datetime64("2024-12-14")
    ex = c.manifest["extra"]
    assert ex["canonical"] is True and ex["duplicate_feature_conflicts"] == 0
    assert ex["rows_by_split_and_file_type"] == {"test": {"Win32": 359_994, "Win64": 119_993},
                                                 "challenge": {"Dot_Net": 829, "Win32": 3225, "Win64": 814}}
    rows = np.random.default_rng(0).choice(c.n, 2000, replace=False)
    assert np.isfinite(c.take(rows)).all()


@pytest.mark.ember
@pytest.mark.slow
def test_real_corpus_matches_source_jsonl(real_corpus: Any) -> None:
    src = os.environ.get("MALVALID_EMBER2024_SOURCE")
    if not src or not (Path(src) / "Win32_test.zip").exists():
        pytest.skip("set MALVALID_EMBER2024_SOURCE to the directory holding the EMBER2024 zips")
    c = real_corpus
    with zipfile.ZipFile(Path(src) / "Win32_test.zip") as zf, zf.open(f"{W1}_Win32_test.jsonl") as f:
        recs = [json.loads(next(f)) for _ in range(200)]
    np.testing.assert_array_equal(c.take(np.arange(200)), tp.vectorize_batch(recs))
    assert c.sha256[:200].tolist() == [r["sha256"] for r in recs]
    assert c.label[:200].tolist() == [r["label"] for r in recs]


@pytest.mark.ember
@pytest.mark.slow
def test_real_corpus_reproduces_published_model_quality(real_corpus: Any) -> None:
    """Correctness gate: the published EMBER2024 PE LightGBM model on the vectorised test set."""
    model = os.environ.get("MALVALID_EMBER2024_MODEL")
    if not model or not Path(model).is_file():
        pytest.skip("set MALVALID_EMBER2024_MODEL to EMBER2024_PE.model")
    lgb = pytest.importorskip("lightgbm")
    from sklearn.metrics import roc_auc_score

    c = real_corpus
    booster = lgb.Booster(model_file=model)
    idx = c.indices(role="eval")
    y = c.label[idx]
    p = np.concatenate([booster.predict(c.take(idx[a : a + 100_000]), num_threads=4)
                        for a in range(0, idx.size, 100_000)])
    # EMBER2024 paper, Table 5: Win32 0.9984, Win64 0.9989 (all PE incl. .NET 0.9982).
    auroc = roc_auc_score(y, p)
    assert auroc > 0.997, auroc  # measured: 0.99832
    ch = c.indices(role="challenge")
    pc = booster.predict(c.take(ch), num_threads=4)
    yc = np.r_[np.zeros(int((y == 0).sum())), np.ones(ch.size)]
    # Table 6 (challenge malicious vs test benign): all PE 0.9643.
    auroc_c = roc_auc_score(yc, np.r_[p[y == 0], pc])
    assert auroc_c > 0.95, auroc_c  # measured: 0.96722
