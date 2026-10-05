"""Shared autouse fixtures for the unit tests."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Tests marked ``posix`` rely on POSIX process groups, signals (SIGKILL), file modes or symlinks,
    or simulate Windows on a POSIX host; they are skipped on Windows."""
    if os.name != "nt":
        return
    skip = pytest.mark.skip(reason="POSIX only (process groups, signals, file modes or symlinks)")
    for item in items:
        if "posix" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _stub_ember_corpora_for_web_tests(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory,
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """Web tests that submit model files assume the canonical EMBER corpora are built: give them a stub
    corpus root (a manifest each) so the missing-corpus check does not refuse the form.
    ``test_web_corpus_missing.py`` simulates a fresh machine instead."""
    mod = request.module.__name__.rsplit(".", 1)[-1]
    if not mod.startswith("test_web") or mod == "test_web_corpus_missing":
        return
    root = tmp_path_factory.mktemp("stub_corpora")
    for n in ("ember_v2_2018", "ember_v3_2024"):
        (root / n).mkdir()
        (root / n / "manifest.json").write_text("{}")
    monkeypatch.setenv("MALVALID_CORPUS_DIR", str(root))
