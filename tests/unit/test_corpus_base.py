"""Corpus location and on-disk verification (malvalid.corpora.base) — regressions from the release review."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from malvalid.config import GateConfig, load_config
from malvalid.core import CorpusUnavailable
from malvalid.corpora import synthetic as S
from malvalid.corpora.base import (
    full_verification_requested,
    load_corpus_dir,
    manifest_name,
    resolve_corpus_dir,
    verify_corpus_dir,
    write_corpus,
)
from malvalid.corpora.ember2018 import Ember2018Provider


def _small_corpus(d: Path, n: int = 64, dim: int = 8) -> dict:
    rng = np.random.default_rng(0)
    X = rng.random((n, dim)).astype(np.float32)
    return write_corpus(
        d, X, name="t_base_corpus", version="v0", feature_version="toy_v1",
        sha256=np.array([f"{i:064x}" for i in range(n)]), label=(np.arange(n) % 2).astype(np.int8),
        timestamp=np.full(n, np.datetime64("2020-01-01", "D")), split=np.array(["test"] * n),
        roles={"eval": ["test"]},
    )


def _fake_manifest(d: Path, name: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"name": name}))


# --------------------------------------------------------------------------------------------------
# xcomp-run-corpus-verify-trusts-stat-cache
# --------------------------------------------------------------------------------------------------


def test_in_place_edit_with_restored_mtime_is_rehashed(tmp_path: Path) -> None:
    d = tmp_path / "c"
    _small_corpus(d)
    assert load_corpus_dir(d).verification["mode"] == "cached"  # the build's sidecar is trusted
    time.sleep(0.05)
    p = d / "X.npy"
    st = p.stat()
    X = np.load(p, mmap_mode="r+")
    X[:8] = 0
    X.flush()
    del X
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))  # size and mtime are unchanged
    assert p.stat().st_size == st.st_size and p.stat().st_mtime_ns == st.st_mtime_ns
    with pytest.raises(CorpusUnavailable, match=r"X\.npy: sha256 .* != manifest"):
        load_corpus_dir(d)


def test_verification_mode_is_recorded(tmp_path: Path) -> None:
    d = tmp_path / "c"
    m = _small_corpus(d)
    cached = load_corpus_dir(d).verification
    assert cached["mode"] == "cached" and set(cached["files"].values()) == {"cached"}
    assert cached["verified_at"] and cached["verified_at"].endswith("Z")
    full = load_corpus_dir(d, force_verify=True).verification
    assert full["mode"] == "full" and set(full["files"]) == set(m["files"])
    assert load_corpus_dir(d, verify=False).verification is None
    # A copy (new inodes) is re-hashed once, then trusted again.
    shutil.copytree(d, tmp_path / "copy")
    assert load_corpus_dir(tmp_path / "copy").verification["mode"] == "full"
    assert load_corpus_dir(tmp_path / "copy").verification["mode"] == "cached"
    assert verify_corpus_dir(d, m, force=True)["mode"] == "full"


def test_old_sidecar_stamps_are_rehashed_once(tmp_path: Path) -> None:
    d = tmp_path / "c"
    _small_corpus(d)
    side = d / ".malvalid-verified.json"
    old = {fn: {"stamp": e["stamp"][:2], "sha256": e["sha256"]} for fn, e in json.loads(side.read_text()).items()}
    side.write_text(json.dumps(old))  # the size+mtime format of earlier releases
    assert load_corpus_dir(d).verification["mode"] == "full"
    assert load_corpus_dir(d).verification["mode"] == "cached"


def test_full_verification_option() -> None:
    assert full_verification_requested(None) is False
    assert full_verification_requested(load_config()) is False
    assert full_verification_requested(load_config(overrides={"runtime": {"corpus_verification": "full"}})) is True


# --------------------------------------------------------------------------------------------------
# xcomp-yaml-corpus-dir-root-pollutes
# --------------------------------------------------------------------------------------------------


def test_resolve_corpus_dir_accepts_the_corpus_or_a_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_manifest(root / "ember_v2_2018", "ember_v2_2018")
    assert resolve_corpus_dir(root, "ember_v2_2018") == root / "ember_v2_2018"
    assert resolve_corpus_dir(root / "ember_v2_2018", "ember_v2_2018") == root / "ember_v2_2018"
    assert resolve_corpus_dir(root, "synthetic_v2") == root  # nothing to descend into
    # A root that was polluted with another corpus's files still resolves to <root>/<name>.
    _fake_manifest(root, "synthetic_v2")
    assert manifest_name(root) == "synthetic_v2"
    assert resolve_corpus_dir(root, "ember_v2_2018") == root / "ember_v2_2018"
    assert resolve_corpus_dir(root, "synthetic_v2") == root


def test_yaml_corpus_dir_root_works_like_the_cli(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _fake_manifest(root / "ember_v2_2018", "ember_v2_2018")
    assert Ember2018Provider().locate(GateConfig(corpus_dir=str(root))) == root / "ember_v2_2018"


def test_provider_refuses_a_directory_holding_another_corpus(tmp_path: Path) -> None:
    d = tmp_path / "syn"
    S.SyntheticEmberV2Provider().build(None, d, n=500, seed=0)
    with pytest.raises(CorpusUnavailable, match="holds corpus 'synthetic_v2', not 'ember_v2_2018'"):
        Ember2018Provider().load(GateConfig(corpus_dir=str(d)))


@pytest.fixture
def small_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(S.ENV_ROWS, "500")
    monkeypatch.delenv(S.ENV_SEED, raising=False)
    monkeypatch.setenv("MALVALID_CORPUS_DIR", str(tmp_path / "corpora"))
    return tmp_path


def test_synthetic_corpus_dir_root_loads_the_existing_corpus(tmp_path: Path, small_rows: Path) -> None:
    root = tmp_path / "shared"
    S.SyntheticEmberV2Provider().build(None, root / "synthetic_v2", n=500, seed=0)
    (root / "README").write_text("shared corpora root\n")
    c = S.SyntheticEmberV2Provider().load(GateConfig(corpus_dir=str(root)))
    assert c.path == root / "synthetic_v2"
    assert sorted(p.name for p in root.iterdir()) == ["README", "synthetic_v2"]  # nothing written at the root


def test_synthetic_generation_never_pollutes_a_root(tmp_path: Path, small_rows: Path) -> None:
    root = tmp_path / "shared"
    _fake_manifest(root / "ember_v2_2018", "ember_v2_2018")
    p = S.SyntheticEmberV2Provider()
    c = p.load(GateConfig(corpus_dir=str(root)))
    sub = root / "synthetic_v2-n500-seed0"
    assert c.path == sub and (sub / "manifest.json").exists() and c.verification["mode"] == "generated"
    assert not (root / "manifest.json").exists() and not (root / "X.npy").exists()
    # ...and the ember corpus next to it still resolves.
    assert Ember2018Provider().locate(GateConfig(corpus_dir=str(root))) == root / "ember_v2_2018"


def test_synthetic_refuses_to_generate_into_a_foreign_directory(small_rows: Path) -> None:
    p = S.SyntheticEmberV2Provider()
    d = p.locate()
    d.mkdir(parents=True)
    (d / "notes.txt").write_text("not a corpus\n")
    with pytest.raises(CorpusUnavailable, match="refusing to generate the synthetic corpus"):
        p.load()
    assert sorted(x.name for x in d.iterdir()) == ["notes.txt"]


def test_synthetic_corpus_dir_that_is_the_corpus_or_empty_still_works(tmp_path: Path, small_rows: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    c = S.SyntheticEmberV2Provider().load(GateConfig(corpus_dir=str(empty)))
    assert c.path == empty and (empty / "manifest.json").exists()
    again = S.SyntheticEmberV2Provider().load(GateConfig(corpus_dir=str(empty)))
    assert again.path == empty and again.verification["mode"] == "cached"
