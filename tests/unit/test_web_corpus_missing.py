"""A fresh machine has no EMBER corpora: the web UI explains it (before queueing a run) and the bundled
synthetic demo and explicit synthetic corpora keep working."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("starlette")

from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    app,
    client,
    fake_malvalid,
    job_json,
    post_run,
    run_id_from,
    settings,
)

pytestmark = pytest.mark.web


@pytest.fixture(autouse=True)
def empty_corpus_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh machine: empty HOME / cache, no MALVALID_CORPUS_DIR."""
    home = tmp_path / "fresh_home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    monkeypatch.delenv("MALVALID_CORPUS_DIR", raising=False)
    return home / ".cache" / "malvalid" / "corpora"


def _model(tmp_path: Path) -> Path:
    p = tmp_path / "m.txt"
    p.write_text("tree\nversion=v3\n")
    return p


def test_explicit_missing_ember_corpus_is_refused_with_a_friendly_message(client, settings, adapter_py):
    r = post_run(client, settings, {"mode": "path", "adapter_path": str(adapter_py), "corpus": "ember_v2_2018"})
    assert r.status_code == 400
    assert "is not on this machine" in r.text and "synthetic demo" in r.text
    assert "malvalid corpus build ember_v2_2018 --source ember2018/" in r.text
    assert "Traceback" not in r.text
    assert not list(settings.runs_dir.glob("*/job.json"))  # nothing was queued


def test_ember_v3_message_has_its_own_build_command(client, settings, adapter_py):
    r = post_run(client, settings, {"mode": "path", "adapter_path": str(adapter_py), "corpus": "ember_v3_2024"})
    assert r.status_code == 400
    assert "malvalid corpus build ember_v3_2024 --source SRC" in r.text


@pytest.mark.parametrize("fv,corpus", [("ember_v2", "ember_v2_2018"), ("ember_v3", "ember_v3_2024")])
def test_model_submission_auto_selected_corpus_is_checked(client, settings, tmp_path, fv, corpus):
    r = post_run(client, settings, {"mode": "path", "adapter_path": str(_model(tmp_path)), "feature_version": fv})
    assert r.status_code == 400
    assert f"corpus &#39;{corpus}&#39; is not on this machine" in r.text or f"'{corpus}' is not on this machine" in r.text
    assert f"malvalid corpus build {corpus}" in r.text


def test_corpora_page_shows_build_instructions(client):
    page = client.get("/corpora").text
    assert "Not installed" in page
    assert "malvalid corpus build ember_v2_2018 --source ember2018/" in page
    assert "malvalid corpus build ember_v3_2024 --source SRC" in page
    assert "Traceback" not in page


def test_demo_still_works_with_no_corpora_built(client, settings, empty_corpus_root):
    from malvalid.web.onboarding import demo_submission

    demo = demo_submission()
    if demo is None:
        pytest.skip("examples/synthetic_demo is not in this checkout")
    assert not empty_corpus_root.exists()
    r = post_run(client, settings, {"mode": "path", **demo})
    job = job_json(settings, run_id_from(r))
    assert job["mode"] == "path"


def test_explicit_synthetic_corpus_is_accepted(client, settings, adapter_py):
    for name in ("synthetic_v2", "synthetic_v3"):
        r = post_run(client, settings, {"mode": "path", "adapter_path": str(adapter_py), "corpus": name})
        job_json(settings, run_id_from(r))


@pytest.fixture
def adapter_py(tmp_path: Path) -> Path:
    from tests.unit.test_web_backend_support import fake_adapter

    return fake_adapter(tmp_path)
