"""Files written before the rename from malguard to MalValid are still read.

The canonical EMBER corpora are ~10 GB and declare ``malguard-corpus/1``; they must load unchanged and
keep their pinned content hashes. Reports, run-state files and model specs written under the old name
are read too. Nothing is ever written with the old identifiers.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from malvalid import (
    CORPUS_FORMAT,
    LEGACY_CORPUS_FORMATS,
    REPORT_SCHEMA_VERSION,
    SUPPORTED_CORPUS_FORMATS,
)
from malvalid.adapters.spec import SPEC_SCHEMA, validate_spec
from malvalid.config import GateConfig
from malvalid.core import CorpusUnavailable
from malvalid.corpora.base import CorpusProvider, compute_content_hash, load_corpus_dir, write_corpus
from malvalid.report.html import render_html
from malvalid.report.json_writer import load_report_json
from malvalid.web import store as ST

LEGACY_CORPUS = "malguard-corpus/1"
NEW_SIDECAR = ".malvalid-verified.json"
LEGACY_SIDECAR = ".malguard-verified.json"


def _pre_rename_content_hash(manifest: dict[str, Any]) -> str:
    """The content hash exactly as malguard 0.1.0a1 computed it (format string included verbatim)."""
    ident = {
        "format": LEGACY_CORPUS,
        "name": manifest["name"],
        "version": manifest["version"],
        "feature_version": manifest["feature_version"],
        "dim": manifest["dim"],
        "n": manifest["n"],
        "files": dict(sorted(manifest["files"].items())),
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


def _corpus(d: Path, n: int = 48, dim: int = 6) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    return write_corpus(
        d, rng.random((n, dim)).astype(np.float32), name="t_legacy_corpus", version="v1",
        feature_version="toy_v1", sha256=np.array([f"{i:064x}" for i in range(n)]),
        label=(np.arange(n) % 2).astype(np.int8), timestamp=np.full(n, np.datetime64("2020-01-01", "D")),
        split=np.array(["test"] * n), roles={"eval": ["test"]},
    )


def _as_pre_rename(d: Path) -> dict[str, Any]:
    """Turn a freshly written corpus dir into what malguard 0.1.0a1 wrote: legacy format, legacy sidecar."""
    m = json.loads((d / "manifest.json").read_text())
    m["format"] = LEGACY_CORPUS
    m["created_by"] = "malguard 0.1.0a1"
    m["content_hash"] = _pre_rename_content_hash(m)
    (d / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True))
    (d / NEW_SIDECAR).rename(d / LEGACY_SIDECAR)
    return m


def test_new_writes_use_the_new_format_and_old_one_is_still_supported() -> None:
    assert CORPUS_FORMAT == "malvalid-corpus/1"
    assert LEGACY_CORPUS in LEGACY_CORPUS_FORMATS
    assert set(SUPPORTED_CORPUS_FORMATS) == {CORPUS_FORMAT, LEGACY_CORPUS}


def test_content_hash_is_the_pre_rename_hash_for_both_spellings(tmp_path: Path) -> None:
    m = _corpus(tmp_path / "c")
    assert m["format"] == CORPUS_FORMAT
    legacy = dict(m, format=LEGACY_CORPUS)
    # the stored hash of a new build equals what the pre-rename code computed for the same data, so a
    # canonical rebuild still reproduces the pinned hashes (e.g. b76ea441... for ember_v2_2018)
    assert m["content_hash"] == _pre_rename_content_hash(m)
    assert compute_content_hash(legacy) == compute_content_hash(m) == _pre_rename_content_hash(m)
    # every other identity field still enters the hash
    assert compute_content_hash(dict(m, n=m["n"] + 1)) != m["content_hash"]
    assert compute_content_hash(dict(m, format="malvalid-corpus/2")) != m["content_hash"]


def test_pre_rename_corpus_loads_with_its_cached_verification(tmp_path: Path) -> None:
    d = tmp_path / "c"
    _corpus(d)
    m = _as_pre_rename(d)
    c = load_corpus_dir(d, expected_content_hash=m["content_hash"])
    assert c.content_hash == m["content_hash"]
    assert c.manifest["format"] == LEGACY_CORPUS
    # the legacy verification cache is trusted: nothing is re-hashed and nothing new is written
    assert c.verification["mode"] == "cached"
    assert sorted(p.name for p in d.iterdir()) == sorted([LEGACY_SIDECAR, "X.npy", "manifest.json", "meta.npz"])


def test_pre_rename_corpus_is_still_verified(tmp_path: Path) -> None:
    d = tmp_path / "c"
    _corpus(d)
    _as_pre_rename(d)
    assert load_corpus_dir(d, force_verify=True).verification["mode"] == "full"
    m = json.loads((d / "manifest.json").read_text())
    m["files"]["X.npy"] = "0" * 64
    m["content_hash"] = _pre_rename_content_hash(m)
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(CorpusUnavailable, match="sha256"):
        load_corpus_dir(d)


def test_unknown_corpus_format_is_refused(tmp_path: Path) -> None:
    d = tmp_path / "c"
    _corpus(d)
    m = json.loads((d / "manifest.json").read_text())
    m["format"] = "othertool-corpus/1"
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(CorpusUnavailable, match="unsupported corpus format"):
        load_corpus_dir(d)


def test_provider_with_a_pinned_hash_loads_a_pre_rename_corpus(tmp_path: Path) -> None:
    d = tmp_path / "t_legacy_corpus"
    _corpus(d)
    m = _as_pre_rename(d)

    class Pinned(CorpusProvider):
        name: ClassVar[str] = "t_legacy_corpus"
        feature_version: ClassVar[str] = "toy_v1"
        version: ClassVar[str] = "v1"
        expected_content_hash: ClassVar[str | None] = m["content_hash"]

    c = Pinned().load(GateConfig(corpus_dir=str(tmp_path)))
    assert c.content_hash == m["content_hash"] and c.n == 48


# ---- reports, run-state files, model specs ---------------------------------------------------------


def _report(schema: str) -> dict[str, Any]:
    return {"schema_version": schema, "tool": {"name": "malguard", "version": "0.1.0a1"},
            "run": {"id": "r1"}, "verdict": {"verdict": "ready", "score": 90.0}, "modules": []}


def test_pre_rename_report_is_read_without_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = tmp_path / "report.json"
    p.write_text(json.dumps(_report("malguard-report/1")))
    with caplog.at_level(logging.WARNING):
        assert load_report_json(p)["schema_version"] == "malguard-report/1"
    assert "schema_version" not in caplog.text
    p.write_text(json.dumps(_report("othertool-report/1")))
    with pytest.raises(Exception, match="not a malvalid report"):
        load_report_json(p)


def test_pre_rename_report_renders_without_the_schema_warning() -> None:
    html = render_html(_report("malguard-report/1"))
    assert "this renderer targets" not in html
    assert "MalValid" in html
    assert "this renderer targets" in render_html(_report("malvalid-report/99"))
    assert REPORT_SCHEMA_VERSION == "malvalid-report/1"


def test_web_store_lists_pre_rename_runs() -> None:
    assert ST._valid_report(_report("malguard-report/1")) is not None
    assert ST._valid_job({"schema": "malguard-job/1", "status": "finished"}) is not None
    assert ST._valid_progress({"schema": "malguard-progress/1", "stage": "done"}) is not None
    assert ST._valid_report(_report("othertool-report/1")) is None
    assert ST._valid_job({"schema": "othertool-job/1", "status": "finished"}) is None


def test_pre_rename_model_spec_is_accepted() -> None:
    doc = {"schema": "malguard-model-spec/1", "model_kind": "lightgbm", "feature_version": "ember_v2",
           "operating_threshold": 0.8, "threshold_source": "declared", "calibrate_fpr": None,
           "model_file": "model.txt", "training_hashes_file": None, "training_cutoff": None}
    s = validate_spec(doc)
    assert s.to_dict()["schema"] == SPEC_SCHEMA == "malvalid-model-spec/1"  # rewritten under the new name


# --- synthetic row ids --------------------------------------------------------------------------------


def test_synthetic_row_ids_keep_pre_rename_namespace():
    """The bundled demo's training manifest was written before the rename; its hashes must stay members."""
    from malvalid.corpora import synthetic as S

    assert S.ROW_SHA256_NAMESPACE == "malguard-synthetic"
    prm = S.SyntheticParams()
    ids = set(S.row_sha256("synthetic_v2", prm.seed, prm.n).tolist())
    assert hashlib.sha256(f"malguard-synthetic/synthetic_v2/seed{prm.seed}/n{prm.n}/0".encode()).hexdigest() in ids
    manifest = Path(__file__).resolve().parents[2] / "examples" / "synthetic_demo" / "train_sha256.txt"
    if not manifest.exists():
        pytest.skip("examples/ not present (e.g. running from an sdist)")
    demo = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(demo) > 1000 and set(demo) <= ids
