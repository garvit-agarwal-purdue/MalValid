"""Risky web features are off unless ``malvalid serve`` enables them, and ``--token-file``.

* path mode ("Use files on this machine": the server reads files by path) needs ``--allow-path-mode``;
* running without the sandbox (``no_sandbox``) needs ``--allow-no-sandbox``;
* without the flag the controls are hidden and ``POST /runs``, ``POST /runs/inspect`` and
  ``POST /api/validate`` answer 403 before any path is read or any run is queued;
* ``--token-file`` keeps the access token across restarts (created 0600 when missing, validated when present).
"""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")

from typer.testing import CliRunner  # noqa: E402

from malvalid.cli import app as cli_app  # noqa: E402
from malvalid.web.forms import (  # noqa: E402
    CORPUS_DIR_DISABLED,
    NO_SANDBOX_DISABLED,
    PATH_MODE_DISABLED,
    corpus_dir_allowed,
)
from malvalid.web.onboarding import demo_submission  # noqa: E402
from malvalid.web.settings import (  # noqa: E402
    TokenFileError,
    WebSettings,
    load_or_create_token_file,
)
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    PORT,
    TOKEN,
    authed_client,
    captured,
    fake_malvalid,
    fake_record,
    make_app,
    post_run,
    read_record,
    run_id_from,
)

pytestmark = pytest.mark.web

JSON = {"accept": "application/json"}
HTML = {"accept": "text/html,application/xhtml+xml"}
LGBM_STUB = b"tree\nversion=v4\nnum_class=1\nmax_feature_idx=2380\n"


def _settings(tmp_path: Path, fake: Path, **kw: Any) -> WebSettings:
    return WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=PORT, cancel_grace_s=1.0,
                       command_prefix=(sys.executable, str(fake)), validate_timeout_s=30, **kw)


@pytest.fixture
def locked(tmp_path: Path, fake_malvalid: Path) -> Any:
    """An app with the defaults: no path mode, no unsandboxed runs."""
    a = make_app(_settings(tmp_path, fake_malvalid))
    with authed_client(a) as c:
        yield a, c


@pytest.fixture
def open_app(tmp_path: Path, fake_malvalid: Path) -> Any:
    a = make_app(_settings(tmp_path, fake_malvalid, allow_path_mode=True, allow_no_sandbox=True))
    with authed_client(a) as c:
        yield a, c


@pytest.fixture
def secret_file(tmp_path: Path) -> Path:
    """A file the server can read: path mode must never be able to name it without the flag."""
    p = tmp_path / "elsewhere" / "secret_model.txt"
    p.parent.mkdir()
    p.write_bytes(LGBM_STUB)
    return p


def _run_dirs(s: WebSettings) -> list[Path]:
    return [p for p in s.runs_dir.iterdir() if p.name != "submissions"] if s.runs_dir.exists() else []


def _submissions(s: WebSettings) -> list[Path]:
    d = s.runs_dir / "submissions"
    return list(d.iterdir()) if d.exists() else []


# ---- defaults ------------------------------------------------------------------------------------


def test_risky_features_are_off_by_default(tmp_path: Path) -> None:
    s = WebSettings(runs_dir=tmp_path / "r", token=TOKEN)
    assert s.allow_path_mode is False and s.allow_no_sandbox is False
    pub = s.public_dict()
    assert pub["allow_path_mode"] is False and pub["allow_no_sandbox"] is False
    on = WebSettings(runs_dir=tmp_path / "r", token=TOKEN, allow_path_mode=True, allow_no_sandbox=True)
    assert on.public_dict()["allow_path_mode"] is True and on.public_dict()["allow_no_sandbox"] is True


# ---- templates -----------------------------------------------------------------------------------


def test_new_run_page_hides_disabled_controls(locked: Any) -> None:
    app, c = locked
    html = c.get("/runs/new").text
    assert 'name="mode" value="path" data-mode-radio' not in html
    assert 'id="adapter_path"' not in html and 'id="training_hashes_path"' not in html
    assert 'id="config_path"' not in html
    assert 'id="no_sandbox"' not in html and 'id="confirm_no_sandbox"' not in html
    assert "data-no-sandbox-disabled" in html
    assert 'name="mode" value="model" data-mode-radio checked' in html  # model mode stays the default


def test_new_run_page_shows_enabled_controls(open_app: Any) -> None:
    app, c = open_app
    html = c.get("/runs/new").text
    assert 'name="mode" value="path" data-mode-radio' in html and 'id="adapter_path"' in html
    assert 'id="no_sandbox"' in html and 'id="confirm_no_sandbox"' in html
    assert "data-no-sandbox-disabled" not in html


def test_a_sticky_path_mode_form_falls_back_to_model_mode(locked: Any) -> None:
    app, c = locked
    r = post_run(c, app.state.settings, {"mode": "path", "adapter_path": "/etc/hostname"}, headers=HTML)
    assert r.status_code == 403
    assert 'name="mode" value="model" data-mode-radio checked' in r.text
    assert "/etc/hostname" not in r.text  # the path field is not rendered at all


# ---- server-side rejection -----------------------------------------------------------------------


def test_path_mode_runs_are_refused_without_the_flag(locked: Any, secret_file: Path, fake_record: Path) -> None:
    app, c = locked
    s = app.state.settings
    for headers in (HTML, JSON):
        r = post_run(c, s, {"mode": "path", "adapter_path": str(secret_file)}, headers=headers)
        assert r.status_code == 403, r.text[:500]
        if headers is JSON:
            body = r.json()
            assert body["ok"] is False and body["errors"] == [PATH_MODE_DISABLED]
        else:
            assert PATH_MODE_DISABLED in captured(app, "new_run.html.j2")["errors"]
            assert "--allow-path-mode" in r.text
    assert _run_dirs(s) == [] and _submissions(s) == [] and read_record(fake_record) == []


def test_path_mode_inspect_and_validate_are_refused_without_the_flag(locked: Any, secret_file: Path,
                                                                      fake_record: Path) -> None:
    app, c = locked
    s = app.state.settings
    calls: list[Path] = []
    app.state.jobs.inspect = lambda model, **kw: calls.append(model) or {"ok": True}
    r = post_run(c, s, {"mode": "path", "adapter_path": str(secret_file)}, url="/runs/inspect", headers=JSON)
    assert r.status_code == 403 and r.json()["errors"] == [PATH_MODE_DISABLED]
    r = post_run(c, s, {"mode": "path", "adapter_path": str(secret_file)}, url="/runs/inspect", headers=HTML)
    assert r.status_code == 403 and "--allow-path-mode" in r.text
    r = post_run(c, s, {"mode": "path", "adapter_path": str(secret_file)}, url="/api/validate")
    assert r.status_code == 403 and r.json() == {"ok": False, "checks": [], "errors": [PATH_MODE_DISABLED]}
    assert calls == [] and read_record(fake_record) == [] and _submissions(s) == []


def test_no_sandbox_is_refused_without_the_flag(locked: Any, fake_record: Path) -> None:
    app, c = locked
    s = app.state.settings
    files = [("model_file", ("model.txt", LGBM_STUB))]
    data = {"mode": "model", "no_sandbox": "on", "confirm_no_sandbox": "on"}
    r = post_run(c, s, data, files, headers=JSON)
    assert r.status_code == 403 and r.json()["errors"] == [NO_SANDBOX_DISABLED]
    r = post_run(c, s, data, files, headers=HTML)
    assert r.status_code == 403 and "--allow-no-sandbox" in r.text
    r = post_run(c, s, {**data, "mode": "upload"}, [("adapter_file", ("a.py", b"class D: pass\n"))],
                 url="/api/validate")
    assert r.status_code == 403 and r.json()["errors"] == [NO_SANDBOX_DISABLED]
    r = post_run(c, s, data, files, url="/runs/inspect", headers=JSON)
    assert r.status_code == 403 and r.json()["errors"] == [NO_SANDBOX_DISABLED]
    both = post_run(c, s, {"mode": "path", "adapter_path": "/x.txt", "no_sandbox": "1"}, headers=JSON)
    assert both.status_code == 403 and both.json()["errors"] == [PATH_MODE_DISABLED, NO_SANDBOX_DISABLED]
    assert _run_dirs(s) == [] and _submissions(s) == [] and read_record(fake_record) == []


def test_a_sandboxed_model_run_is_still_accepted(locked: Any) -> None:
    app, c = locked
    r = post_run(c, app.state.settings, {"mode": "model"}, [("model_file", ("model.txt", LGBM_STUB))])
    run_id_from(r)


def test_the_flags_enable_path_mode_and_no_sandbox(open_app: Any, secret_file: Path) -> None:
    app, c = open_app
    s = app.state.settings
    r = post_run(c, s, {"mode": "path", "adapter_path": str(secret_file), "no_sandbox": "on",
                        "confirm_no_sandbox": "on"})
    rid = run_id_from(r)
    job = (s.runs_dir / rid / "job.json").read_text()
    assert str(secret_file) in job and '"no_sandbox": true' in job


def test_the_bundled_demo_still_runs_without_path_mode(locked: Any) -> None:
    demo = demo_submission()
    if demo is None:
        pytest.skip("no examples/synthetic_demo in this checkout")
    app, c = locked
    html = c.get("/runs/new").text
    assert "data-demo-form" in html
    r = post_run(c, app.state.settings, {"mode": "path", **demo})
    run_id_from(r)
    # ... but only with the server's own paths: another adapter or a hash list is refused
    other = post_run(c, app.state.settings, {"mode": "path", **demo, "adapter_path": demo["config_path"]},
                     headers=JSON)
    assert other.status_code == 403
    hashes = post_run(c, app.state.settings, {"mode": "path", **demo, "training_hashes_path": "/etc/passwd"},
                      headers=JSON)
    assert hashes.status_code == 403
    unsandboxed = post_run(c, app.state.settings, {"mode": "path", **demo, "no_sandbox": "on",
                                                   "confirm_no_sandbox": "on"}, headers=JSON)
    assert unsandboxed.status_code == 403 and unsandboxed.json()["errors"] == [NO_SANDBOX_DISABLED]


def test_rerun_of_a_path_mode_run_drops_disabled_options(locked: Any, tmp_path: Path) -> None:
    import json

    app, c = locked
    s = app.state.settings
    d = s.runs_dir / "20260101T000000Z-aaaaaaaa"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({
        "run_id": d.name, "mode": "path", "adapter": "/home/x/model.txt", "status": "finished",
        "options": {"no_sandbox": True, "training_hashes": "/home/x/h.txt", "title": "old"}}))
    r = c.get(f"/runs/new?from={d.name}")
    assert r.status_code == 200
    ctx = captured(app, "new_run.html.j2")
    assert ctx["form"]["mode"] == "model" and "adapter_path" not in ctx["form"]
    assert "no_sandbox" not in ctx["form"] and "training_hashes_path" not in ctx["form"]
    assert "disabled on this server" in ctx["notice"]


# ---- corpus_dir ----------------------------------------------------------------------------------


@pytest.fixture
def corpus_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "corpora"
    root.mkdir()
    monkeypatch.setenv("MALVALID_CORPUS_DIR", str(root))
    return root


def test_corpus_dir_field_is_hidden_without_path_mode(locked: Any, corpus_root: Path) -> None:
    assert 'id="corpus_dir"' not in locked[1].get("/runs/new").text


def test_corpus_dir_field_is_shown_with_path_mode(open_app: Any, corpus_root: Path) -> None:
    assert 'id="corpus_dir"' in open_app[1].get("/runs/new").text


def test_corpus_dir_other_than_the_servers_is_refused(locked: Any, corpus_root: Path, tmp_path: Path,
                                                      fake_record: Path) -> None:
    app, c = locked
    s = app.state.settings
    other = tmp_path / "elsewhere"
    other.mkdir()
    files = [("model_file", ("model.txt", LGBM_STUB))]
    data = {"mode": "model", "corpus_dir": str(other)}
    r = post_run(c, s, data, files, headers=JSON)
    assert r.status_code == 403 and r.json()["errors"] == [CORPUS_DIR_DISABLED]
    r = post_run(c, s, data, files, headers=HTML)
    assert r.status_code == 403 and CORPUS_DIR_DISABLED in captured(app, "new_run.html.j2")["errors"]
    r = post_run(c, s, data, files, url="/runs/inspect", headers=JSON)
    assert r.status_code == 403 and r.json()["errors"] == [CORPUS_DIR_DISABLED]
    r = post_run(c, s, {**data, "mode": "upload"}, [("adapter_file", ("a.py", b"class D: pass\n"))],
                 url="/api/validate")
    assert r.status_code == 403 and r.json()["errors"] == [CORPUS_DIR_DISABLED]
    assert _run_dirs(s) == [] and _submissions(s) == [] and read_record(fake_record) == []


def test_corpus_dir_equal_to_the_servers_or_empty_is_accepted(locked: Any, corpus_root: Path) -> None:
    app, c = locked
    s = app.state.settings
    for n in ("ember_v2_2018", "ember_v3_2024"):  # built corpora: only the corpus_dir policy is under test
        (corpus_root / n).mkdir(parents=True, exist_ok=True)
        (corpus_root / n / "manifest.json").write_text("{}")
    files = [("model_file", ("model.txt", LGBM_STUB))]
    run_id_from(post_run(c, s, {"mode": "model", "corpus_dir": ""}, files))
    run_id_from(post_run(c, s, {"mode": "model", "corpus_dir": str(corpus_root) + "/"}, files))


def test_corpus_dir_check_unit(tmp_path: Path, corpus_root: Path) -> None:
    off = WebSettings(runs_dir=tmp_path / "r", token=TOKEN)
    on = WebSettings(runs_dir=tmp_path / "r", token=TOKEN, allow_path_mode=True)
    assert corpus_dir_allowed(None, off) and corpus_dir_allowed("  ", off)
    assert corpus_dir_allowed(str(corpus_root), off) and corpus_dir_allowed(corpus_root / ".." / "corpora", off)
    assert not corpus_dir_allowed(str(tmp_path), off) and not corpus_dir_allowed(str(corpus_root / "sub"), off)
    assert corpus_dir_allowed(str(tmp_path), on)


def test_corpus_dir_is_allowed_with_path_mode(open_app: Any, corpus_root: Path, tmp_path: Path) -> None:
    app, c = open_app
    other = tmp_path / "mycorpus"
    other.mkdir()
    (other / "manifest.json").write_text("{}")  # a built corpus: only the corpus_dir policy is under test
    r = post_run(c, app.state.settings, {"mode": "model", "corpus_dir": str(other)},
                 [("model_file", ("model.txt", LGBM_STUB))])
    run_id_from(r)


def test_rerun_drops_a_disabled_corpus_dir(locked: Any, corpus_root: Path) -> None:
    import json

    app, c = locked
    d = app.state.settings.runs_dir / "20260101T000000Z-bbbbbbbb"
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps({
        "run_id": d.name, "mode": "model", "adapter": "m.txt", "status": "finished",
        "options": {"corpus_dir": "/somewhere/else", "title": "old"}}))
    assert c.get(f"/runs/new?from={d.name}").status_code == 200
    ctx = captured(app, "new_run.html.j2")
    assert "corpus_dir" not in ctx["form"] and "disabled on this server" in ctx["notice"]


# ---- serve flags ---------------------------------------------------------------------------------


@pytest.fixture
def captured_serve(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from malvalid.web import serve as web_serve

    seen: dict[str, Any] = {}

    def fake_run_server(settings: WebSettings, sock: Any, *, open_browser: bool = False,
                        log_level: str = "warning") -> None:
        seen["settings"] = settings

    monkeypatch.setattr(web_serve, "run_server", fake_run_server)
    monkeypatch.delenv("MALVALID_SERVE_TOKEN", raising=False)
    return seen


def _serve(*args: str) -> Any:
    return CliRunner().invoke(cli_app, ["serve", "--port", "0", "--no-browser", *args], env={"COLUMNS": "200"})


def test_serve_flags_reach_the_settings(tmp_path: Path, captured_serve: dict[str, Any]) -> None:
    r = _serve("--runs-dir", str(tmp_path / "runs"), "--token", TOKEN)
    assert r.exit_code == 0, r.output
    s = captured_serve["settings"]
    assert s.allow_path_mode is False and s.allow_no_sandbox is False
    r = _serve("--runs-dir", str(tmp_path / "runs"), "--token", TOKEN, "--allow-path-mode", "--allow-no-sandbox")
    assert r.exit_code == 0, r.output
    s = captured_serve["settings"]
    assert s.allow_path_mode is True and s.allow_no_sandbox is True
    assert "--allow-path-mode" in r.output and "--allow-no-sandbox" in r.output  # startup warnings


# ---- --token-file --------------------------------------------------------------------------------


@pytest.mark.posix
def test_token_file_is_created_0600_and_reused(tmp_path: Path, captured_serve: dict[str, Any]) -> None:
    f = tmp_path / "tok"
    r = _serve("--runs-dir", str(tmp_path / "runs"), "--token-file", str(f))
    assert r.exit_code == 0, r.output
    token = captured_serve["settings"].token
    assert f.read_text() == token + "\n"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert len(token) >= 40 and re.fullmatch(r"[A-Za-z0-9_-]+", token)
    assert r.output.count(token) == 2  # only in the printed login links (127.0.0.1 and *.localhost)
    assert captured_serve["settings"].public_dict()["token_source"].startswith("--token-file")
    r = _serve("--runs-dir", str(tmp_path / "runs"), "--token-file", str(f))
    assert r.exit_code == 0 and captured_serve["settings"].token == token  # the same link after a restart
    assert "Created the token file" not in r.output


def test_token_file_validation_never_echoes_the_content(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    for content in ("short-token\n", "has spaces in the token value\n", "x" * 600 + "\n", "\n"):
        bad.write_text(content)
        with pytest.raises(TokenFileError) as e:
            load_or_create_token_file(bad)
        assert content.strip() == "" or content.strip() not in str(e.value)
    d = tmp_path / "a-dir"
    d.mkdir()
    with pytest.raises(TokenFileError, match="not a regular file"):
        load_or_create_token_file(d)
    with pytest.raises(TokenFileError, match="cannot create"):
        load_or_create_token_file(tmp_path / "missing-dir" / "tok")


@pytest.mark.posix
def test_token_file_seeded_by_hand_and_loose_permissions(tmp_path: Path, captured_serve: dict[str, Any]) -> None:
    f = tmp_path / "seeded"
    seed = "Zx3-fakeFAKEfake_seedSEEDseed0123456789ABCD"  # the format `secrets.token_urlsafe(32)` makes
    f.write_text(seed + "\n# comment lines after the first are ignored\n")
    os.chmod(f, 0o644)
    token, created, loose = load_or_create_token_file(f)
    assert (token, created, loose) == (seed, False, True)
    r = _serve("--runs-dir", str(tmp_path / "runs"), "--token-file", str(f))
    assert r.exit_code == 0, r.output
    assert captured_serve["settings"].token == seed
    assert "readable by other users" in r.output
    os.chmod(f, 0o600)
    assert load_or_create_token_file(f) == (seed, False, False)


@pytest.mark.posix
def test_token_file_never_follows_a_dangling_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(TokenFileError):
        load_or_create_token_file(link)
    assert not target.exists()
