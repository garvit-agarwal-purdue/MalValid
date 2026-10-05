"""Unit tests for the canonical EMBER2018 corpus provider (malvalid.corpora.ember2018).

Fast tests build tiny corpora from synthetic EMBER v2 raw records laid out like the real release
(``train_features_0..5.jsonl`` + ``test_features.jsonl``). Tests marked ``ember`` use the real
canonical corpus and are skipped unless it is available:

* ``MALVALID_EMBER2018_DIR`` (or ``MALVALID_EMBER2018``): the built corpus directory, else
  ``$MALVALID_CORPUS_DIR/ember_v2_2018``;
* ``MALVALID_EMBER2018_SOURCE``: optional, the extracted EMBER2018 JSONL directory (enables the
  re-vectorisation spot check);
* ``MALVALID_EMBER2018_MODEL``: optional, the published ``ember_model_2018.txt`` (enables the slow
  AUROC check).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from malvalid.config import GateConfig
from malvalid.core import CorpusUnavailable
from malvalid.corpora import ember2018 as M
from malvalid.corpora.base import compute_content_hash, load_corpus_dir, sha256_file
from malvalid.schemas.ember_v2 import vectorize_raw_batch

# --------------------------------------------------------------------------------------------------
# Synthetic source files
# --------------------------------------------------------------------------------------------------

_SECTION_NAMES = [".text", ".rdata", ".data", ".rsrc", ".reloc", "UPX0", "UPX1", ""]
_LIBS = ["KERNEL32.dll", "USER32.dll", "advapi32.dll", "WS2_32.dll", "msvcrt.dll"]


def synthetic_record(i: int, label: int, appeared: str) -> dict[str, Any]:
    """A structurally valid EMBER v2 raw record whose values depend only on ``i``."""
    rng = np.random.default_rng(1000 + i)
    n_sec = int(rng.integers(1, 5))
    names = [str(x) for x in rng.choice(_SECTION_NAMES, n_sec, replace=False)]
    sections = [
        {
            "name": nm,
            "size": int(rng.integers(0, 3) * 512),
            "entropy": float(np.round(rng.uniform(0, 8), 6)),
            "vsize": int(rng.integers(1, 9) * 4096),
            "props": [str(p) for p in rng.choice(["CNT_CODE", "MEM_EXECUTE", "MEM_READ", "MEM_WRITE"],
                                                 int(rng.integers(1, 4)), replace=False)],
        }
        for nm in names
    ]
    libs = [str(x) for x in rng.choice(_LIBS, int(rng.integers(0, 4)), replace=False)]
    printabledist = rng.integers(0, 40, 96).tolist()
    return {
        "sha256": f"{i:064x}",
        "md5": f"{i:032x}",
        "appeared": appeared,
        "label": label,
        "avclass": "",
        "histogram": rng.integers(0, 500, 256).tolist(),
        "byteentropy": rng.integers(0, 500, 256).tolist(),
        "strings": {
            "numstrings": int(rng.integers(0, 300)), "avlength": float(rng.uniform(5, 20)),
            "printabledist": printabledist, "printables": int(sum(printabledist)),
            "entropy": float(rng.uniform(4, 6.5)), "paths": int(rng.integers(0, 3)),
            "urls": int(rng.integers(0, 3)), "registry": int(rng.integers(0, 3)), "MZ": int(rng.integers(0, 3)),
        },
        "general": {
            "size": int(rng.integers(1000, 10**6)), "vsize": int(rng.integers(4096, 10**6)),
            "has_debug": int(rng.integers(0, 2)), "exports": 0, "imports": 3, "has_relocations": 1,
            "has_resources": int(rng.integers(0, 2)), "has_signature": 0, "has_tls": 0, "symbols": 0,
        },
        "header": {
            "coff": {"timestamp": int(rng.integers(0, 2**31)), "machine": "I386",
                     "characteristics": ["EXECUTABLE_IMAGE", "CHARA_32BIT_MACHINE"]},
            "optional": {
                "subsystem": "WINDOWS_GUI", "dll_characteristics": ["NX_COMPAT"], "magic": "PE32",
                "major_image_version": 0, "minor_image_version": 0, "major_linker_version": 14,
                "minor_linker_version": int(rng.integers(0, 30)), "major_operating_system_version": 6,
                "minor_operating_system_version": 0, "major_subsystem_version": 6,
                "minor_subsystem_version": 0, "sizeof_code": int(rng.integers(0, 10**5)),
                "sizeof_headers": 1024, "sizeof_heap_commit": 4096,
            },
        },
        "section": {"entry": names[0], "sections": sections},
        "imports": {lib: [f"Func{int(k)}" for k in rng.integers(0, 50, 3)] for lib in libs},
        "exports": [f"Export{int(k)}" for k in rng.integers(0, 9, int(rng.integers(0, 3)))],
        "datadirectories": [
            {"name": n, "size": int(rng.integers(0, 1000)), "virtual_address": int(rng.integers(0, 10**5))}
            for n in ("EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE",
                      "CERTIFICATE_TABLE", "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE",
                      "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE", "BOUND_IMPORT", "IAT",
                      "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER")
        ],
    }


# rows per source file; train_features_5 is large enough (> 64 KiB) to be split into several chunks
_ROWS = {"train_features_0.jsonl": 3, "train_features_1.jsonl": 4, "train_features_2.jsonl": 2,
         "train_features_3.jsonl": 3, "train_features_4.jsonl": 1, "train_features_5.jsonl": 40,
         "test_features.jsonl": 9}


def write_source(d: Path) -> list[dict[str, Any]]:
    """Write the seven JSONL files; returns all records in canonical row order."""
    d.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    i = 0
    for fn, split in M.SOURCE_FILES:
        rows = []
        for _ in range(_ROWS[fn]):
            if split == "test":
                label, appeared = i % 2, f"2018-{11 + i % 2}"
            else:
                label, appeared = (i % 3) - 1, f"2018-{1 + i % 10:02d}"
            rows.append(synthetic_record(i, label, appeared))
            i += 1
        # the last file has no trailing newline (the reader must still count its final record)
        text = "\n".join(json.dumps(r) for r in rows) + ("" if fn == "test_features.jsonl" else "\n")
        (d / fn).write_text(text)
        records.extend(rows)
    return records


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict[str, Any]]]:
    d = tmp_path_factory.mktemp("ember2018_src") / "ember2018"
    return d, write_source(d)


@pytest.fixture(scope="module")
def built(source: tuple[Path, list[dict[str, Any]]], tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("built") / "ember_v2_2018"
    M.Ember2018Provider().build(source[0], out, workers=1, chunk_bytes=1 << 16, batch_rows=5)
    return out


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------


def test_build_contents(built: Path, source: tuple[Path, list[dict[str, Any]]]) -> None:
    records = source[1]
    c = load_corpus_dir(built)
    n = len(records)
    assert (c.n, c.dim) == (n, 2381)
    assert c.name == "ember_v2_2018" and c.feature_version == "ember_v2" and c.version == M.Ember2018Provider.version
    np.testing.assert_array_equal(c.take(np.arange(n)), vectorize_raw_batch(records))
    assert c.sha256.tolist() == [r["sha256"] for r in records]
    assert c.label.tolist() == [r["label"] for r in records]
    assert c.timestamp.tolist() == [np.datetime64(r["appeared"] + "-01", "D").item() for r in records]
    n_test = _ROWS["test_features.jsonl"]
    assert c.split.tolist() == ["train"] * (n - n_test) + ["test"] * n_test
    assert c.roles() == {"eval": ["test"], "temporal": ["train", "test"], "pool": ["train", "test"], "challenge": []}
    assert np.array_equal(c.eval_indices(1), np.array([i for i in range(n - n_test, n) if records[i]["label"] == 1]))
    assert len(c.indices(splits=["train"], label=-1)) == sum(
        1 for r in records[: n - n_test] if r["label"] == -1
    )


def test_build_manifest_provenance(built: Path, source: tuple[Path, list[dict[str, Any]]]) -> None:
    m = json.loads((built / "manifest.json").read_text())
    assert m["content_hash"] == compute_content_hash(m)
    src = m["source"]
    assert src["dataset"] == M.DATASET_NAME and src["archive_sha256"] == M.ARCHIVE_SHA256
    assert src["download_url"] == M.ARCHIVE_URL and src["license"].startswith("MIT")
    assert [f["file"] for f in src["files"]] == [fn for fn, _ in M.SOURCE_FILES]
    for f in src["files"]:
        assert f["records"] == _ROWS[f["file"]]
        assert f["sha256"] == sha256_file(source[0] / f["file"])
    ex = m["extra"]
    assert ex["build_recipe"] == M.BUILD_RECIPE and ex["rows_with_nan"] == 0 and ex["duplicate_sha256"] == 0
    assert ex["counts"]["test"]["n"] == _ROWS["test_features.jsonl"]
    assert ex["counts"]["test"]["first_month"] == "2018-11" and ex["counts"]["test"]["last_month"] == "2018-12"


def test_build_is_deterministic_across_workers_and_chunking(
    built: Path, source: tuple[Path, list[dict[str, Any]]], tmp_path: Path
) -> None:
    out = tmp_path / "again"
    m2 = M.Ember2018Provider().build(source[0], out, workers=2, chunk_bytes=1 << 20, batch_rows=512)
    m1 = json.loads((built / "manifest.json").read_text())
    assert m2["content_hash"] == m1["content_hash"]
    assert m2["files"] == m1["files"]
    for fn in ("X.npy", "meta.npz"):
        assert (out / fn).read_bytes() == (built / fn).read_bytes()


def test_build_warns_on_unexpected_row_counts(
    source: tuple[Path, list[dict[str, Any]]], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="malvalid.corpora.ember2018"):
        m = M.Ember2018Provider().build(source[0], tmp_path / "c", workers=1)
    assert "EMBER2018 release has 800000" in caplog.text
    assert "differs from the canonical" in caplog.text  # pinned hash cannot match a toy source
    assert m["n"] == sum(_ROWS.values())


def test_rebuild_over_existing_corpus(built: Path, source: tuple[Path, list[dict[str, Any]]], tmp_path: Path) -> None:
    out = tmp_path / "c"
    M.Ember2018Provider().build(source[0], out, workers=1)
    m = M.Ember2018Provider().build(source[0], out, workers=1)  # replaces its own previous build
    assert m["content_hash"] == json.loads((built / "manifest.json").read_text())["content_hash"]


def test_source_directory_can_be_the_parent(source: tuple[Path, list[dict[str, Any]]], tmp_path: Path) -> None:
    m = M.Ember2018Provider().build(source[0].parent, tmp_path / "c", workers=1)  # finds ember2018/
    assert m["n"] == sum(_ROWS.values())


def test_build_errors_are_actionable(tmp_path: Path) -> None:
    prov = M.Ember2018Provider()
    with pytest.raises(M.Ember2018BuildError, match="no --source"):
        prov.build(None, tmp_path / "o")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(M.Ember2018BuildError) as ei:
        prov.build(empty, tmp_path / "o")
    assert "train_features_0.jsonl" in str(ei.value) and M.ARCHIVE_URL in str(ei.value)
    tarball = tmp_path / M.ARCHIVE_NAME
    tarball.write_bytes(b"BZh9")
    with pytest.raises(M.Ember2018BuildError, match="extract it first"):
        prov.build(tarball, tmp_path / "o")


def test_foreign_files_in_output_dir_are_refused(source: tuple[Path, list[dict[str, Any]]], tmp_path: Path) -> None:
    out = tmp_path / "busy"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(M.Ember2018BuildError, match="already contains other files"):
        M.Ember2018Provider().build(source[0], out, workers=1)
    assert (out / "notes.txt").read_text() == "keep me"


@pytest.mark.parametrize(
    "line, message",
    [
        ("{not json", "not valid JSON"),
        ('{"sha256": "abc", "label": 0}', "malformed 'sha256'"),
        (json.dumps({**synthetic_record(999, 0, "2018-01"), "label": 2}), "'label' must be 1, 0 or -1"),
        (json.dumps({**synthetic_record(999, 0, "2018-01"), "label": True}), "'label' must be 1, 0 or -1"),
        (json.dumps({k: v for k, v in synthetic_record(999, 0, "2018-01").items() if k != "imports"}),
         "missing top-level field 'imports'"),
    ],
)
def test_malformed_source_record(tmp_path: Path, line: str, message: str) -> None:
    src = tmp_path / "src"
    write_source(src)
    with open(src / "train_features_2.jsonl", "a") as f:
        f.write(line + "\n")
    out = tmp_path / "out"
    with pytest.raises(M.Ember2018BuildError) as ei:
        M.Ember2018Provider().build(src, out, workers=1)
    assert message in str(ei.value) and "train_features_2.jsonl" in str(ei.value)
    assert not (out / "manifest.json").exists() and not (out / "X.npy").exists()


def test_missing_appeared_month_gives_nat(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "src"
    write_source(src)
    with open(src / "train_features_0.jsonl", "a") as f:
        f.write(json.dumps(synthetic_record(998, 1, "unknown")) + "\n")
    with caplog.at_level(logging.WARNING, logger="malvalid.corpora.ember2018"):
        m = M.Ember2018Provider().build(src, tmp_path / "out", workers=1)
    assert m["extra"]["rows_without_timestamp"] == 1 and "no parseable 'appeared' month" in caplog.text
    c = load_corpus_dir(tmp_path / "out")
    assert int(np.isnat(c.timestamp).sum()) == 1


def test_month_start() -> None:
    assert M._month_start("2018-03") == np.datetime64("2018-03-01")
    assert M._month_start("2006-12-15") == np.datetime64("2006-12-01")
    for bad in ("2018-13", "18-03", "", None, 201803):
        assert np.isnat(M._month_start(bad))


# --------------------------------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------------------------------


def test_provider_identity_and_registry() -> None:
    from malvalid import registry

    prov = registry.get_corpus_provider("ember_v2_2018")
    assert isinstance(prov, M.Ember2018Provider)
    assert prov.feature_version == "ember_v2" and not prov.synthetic
    h = prov.expected_content_hash
    assert isinstance(h, str) and len(h) == 64 and int(h, 16) >= 0
    info = prov.info()
    assert info["expected_content_hash"] == h and info["name"] == "ember_v2_2018"


def test_unavailable_hint_explains_how_to_build(tmp_path: Path) -> None:
    prov = M.Ember2018Provider()
    cfg = GateConfig(corpus_dir=str(tmp_path / "nowhere"))
    assert not prov.is_available(cfg)
    with pytest.raises(CorpusUnavailable) as ei:
        prov.load(cfg)
    text = str(ei.value)
    assert M.ARCHIVE_URL in text and M.ARCHIVE_SHA256 in text
    assert "malvalid corpus build ember_v2_2018 --source" in text and "tar -xjf" in text


def test_provider_refuses_non_canonical_build(built: Path) -> None:
    prov = M.Ember2018Provider()
    cfg = GateConfig(corpus_dir=str(built))
    assert prov.is_available(cfg)
    with pytest.raises(CorpusUnavailable, match="does not match the pinned"):
        prov.load(cfg)


def test_provider_loads_when_hash_matches(built: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = json.loads((built / "manifest.json").read_text())
    monkeypatch.setattr(M.Ember2018Provider, "expected_content_hash", m["content_hash"])
    c = M.Ember2018Provider().load(GateConfig(corpus_dir=str(built)))
    assert c.content_hash == m["content_hash"] and c.summary()["splits"]["test"]["n"] == _ROWS["test_features.jsonl"]


# --------------------------------------------------------------------------------------------------
# Real canonical corpus (optional)
# --------------------------------------------------------------------------------------------------


def _real_corpus_dir() -> Path | None:
    for var in ("MALVALID_EMBER2018_DIR", "MALVALID_EMBER2018"):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("MALVALID_CORPUS_DIR"):
        return Path(os.environ["MALVALID_CORPUS_DIR"]) / "ember_v2_2018"
    return None


@pytest.fixture(scope="module")
def real_corpus() -> Any:
    d = _real_corpus_dir()
    if d is None or not (d / "manifest.json").exists():
        pytest.skip("canonical ember_v2_2018 corpus not available (set MALVALID_EMBER2018_DIR)")
    return M.Ember2018Provider().load(GateConfig(corpus_dir=str(d)))  # verifies the pinned hash


@pytest.mark.ember
def test_real_corpus_shape_splits_labels_hash(real_corpus: Any) -> None:
    c = real_corpus
    assert c.content_hash == M.Ember2018Provider.expected_content_hash
    assert (c.n, c.dim) == (1_000_000, 2381) and c.X.dtype == np.float32
    s = c.summary()["splits"]
    assert s["train"] == {"n": 800_000, "malicious": 300_000, "benign": 300_000, "unlabeled": 200_000}
    assert s["test"] == {"n": 200_000, "malicious": 100_000, "benign": 100_000, "unlabeled": 0}
    assert np.all(c.split[:800_000] == "train") and np.all(c.split[800_000:] == "test")
    assert not np.isnat(c.timestamp).any()
    test_ts = c.timestamp[800_000:]
    assert test_ts.min() == np.datetime64("2018-11-01") and test_ts.max() == np.datetime64("2018-12-01")
    assert np.unique(c.sha256).size == c.n
    assert c.eval_indices(1).size == 100_000 and c.eval_indices(0).size == 100_000
    assert c.roles()["eval"] == ["test"] and c.roles()["challenge"] == []
    rows = np.random.default_rng(0).choice(c.n, 2000, replace=False)
    assert np.isfinite(c.take(rows)).all()


@pytest.mark.ember
def test_real_corpus_matches_source_jsonl(real_corpus: Any) -> None:
    src = os.environ.get("MALVALID_EMBER2018_SOURCE")
    if not src or not (Path(src) / "test_features.jsonl").exists():
        pytest.skip("set MALVALID_EMBER2018_SOURCE to the extracted EMBER2018 JSONL directory")
    c = real_corpus
    row0 = 0
    for fn, _split in M.SOURCE_FILES:
        with open(Path(src) / fn) as f:
            recs = [json.loads(next(f)) for _ in range(100)]
        idx = np.arange(row0, row0 + 100)
        np.testing.assert_array_equal(c.take(idx), vectorize_raw_batch(recs))
        assert c.sha256[idx].tolist() == [r["sha256"] for r in recs]
        row0 += next(x["records"] for x in c.manifest["source"]["files"] if x["file"] == fn)


@pytest.mark.ember
@pytest.mark.slow
def test_real_corpus_reproduces_published_model_quality(real_corpus: Any) -> None:
    """The correctness gate: the published EMBER2018 LightGBM model on the vectorised test set."""
    model = os.environ.get("MALVALID_EMBER2018_MODEL")
    if not model or not Path(model).is_file():
        pytest.skip("set MALVALID_EMBER2018_MODEL to ember_model_2018.txt")
    lgb = pytest.importorskip("lightgbm")
    from sklearn.metrics import roc_auc_score

    c = real_corpus
    idx = c.indices(role="eval")
    y = c.label[idx]
    p = lgb.Booster(model_file=model).predict(c.take(idx), num_threads=4)
    # published in the EMBER authors' ember2018 notebook: ROC AUC 0.9964289467999999,
    # threshold 0.8336 at 1% FPR with 96.498% detection
    assert abs(roc_auc_score(y, p) - 0.9964289468) < 1e-6
    benign = np.sort(p[y == 0])
    thr = benign[int(np.ceil(0.99 * benign.size)) - 1]  # 1% FPR
    assert abs(thr - 0.8336) < 1e-3
    assert np.mean(p[y == 1] > thr) > 0.964
