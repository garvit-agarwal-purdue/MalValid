"""Security layer of the local web UI (``docs/WEB_CONTRACT.md`` §1).

Token login (``?token=`` → HttpOnly/SameSite=Strict cookie + 303 without the token), cookie or
bearer authentication on every route except ``/healthz`` and ``/static/*``, the Host-header
allowlist, CSRF on every POST (header or form field, plus Origin/Referer checks), the security
headers, run-id validation and the raw-file allowlist (nothing under ``private/`` is ever served).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from malvalid.web.security import APP_CSP, REPORT_CSP  # noqa: E402
from malvalid.web.settings import COOKIE_NAME, WebSettings  # noqa: E402
from malvalid.web.store import RunNotFound, RunStore, valid_run_id  # noqa: E402
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    BASE,
    PORT,
    TOKEN,
    anon,
    app,
    captured,
    cli_run_dir,
    client,
    fake_malvalid,
    make_app,
    settings,
)

pytestmark = pytest.mark.web

HTML = {"accept": "text/html,application/xhtml+xml"}
JSON = {"accept": "application/json"}


def _cli_run(settings: WebSettings, run_id: str = "cli-run-1") -> Path:
    return cli_run_dir(settings.runs_dir, run_id, "sample_report.json",
                       report__html="<!doctype html><title>r</title>", run__log="log line\n",
                       console__log="console line\n")


# --------------------------------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------------------------------


def test_healthz_and_static_need_no_token(anon):
    r = anon.get("/healthz")
    assert r.status_code == 200 and r.text == "ok"
    for name in ("app.css", "app.js"):
        r = anon.get(f"/static/{name}")
        assert r.status_code == 200, name
    assert anon.get("/static/nope.css").status_code == 404


@pytest.mark.parametrize("path", ["/", "/runs/new", "/compare", "/corpora", "/modules", "/system",
                                  "/runs/cli-run-1", "/runs/cli-run-1/report.json", "/no-such-page"])
def test_every_page_needs_the_token(anon, settings, path):
    _cli_run(settings)
    r = anon.get(path, headers=HTML)
    assert r.status_code == 401
    assert "text/html" in r.headers["content-type"]
    assert "malvalid serve" in r.text  # explains how to get in: open the printed URL
    assert TOKEN not in r.text and settings.csrf_token not in r.text


def test_api_401_is_json(anon):
    r = anon.get("/api/runs")
    assert r.status_code == 401
    body = r.json()
    assert body["status"] == 401 and "malvalid serve" in body["message"]


def test_login_link_sets_the_cookie_and_strips_the_token(anon, settings):
    r = anon.get(f"/?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE_NAME}_{PORT}=")  # port-scoped name: instances never clobber
    assert TOKEN not in cookie  # a random session id, never the launch token
    low = cookie.lower()
    assert "httponly" in low and "samesite=strict" in low and "path=/" in low
    # The browser now holds the cookie: the redirect target works without the token.
    r = anon.get("/", headers=HTML)
    assert r.status_code == 200


def test_the_raw_token_in_a_cookie_is_not_a_session(app, settings):
    """Regression (security-cookie-cross-port-leak): the cookie holds a session id, not the token."""
    with TestClient(app, base_url=BASE) as c:
        c.cookies.set(settings.cookie_name, TOKEN)
        assert c.get("/api/runs").status_code == 401
        c.cookies.set("malvalid_token", TOKEN)  # the old cookie name
        assert c.get("/api/runs").status_code == 401


def test_each_login_gets_its_own_session(app, settings):
    ids = set()
    for _ in range(2):
        with TestClient(app, base_url=BASE) as c:
            r = c.get(f"/?token={TOKEN}", follow_redirects=False)
            ids.add(r.cookies.get(settings.cookie_name))
            assert c.get("/api/runs").status_code == 200
    assert len(ids) == 2 and None not in ids


def test_a_session_does_not_survive_a_restart(tmp_path, settings):
    """Sessions live in the server process: a new launch (same --token) signs everyone out."""
    sid = settings.sessions.new_session()
    s2 = WebSettings(runs_dir=settings.runs_dir, token=TOKEN, port=PORT)
    app2 = make_app(s2)
    with TestClient(app2, base_url=BASE) as c:
        c.cookies.set(s2.cookie_name, sid)
        assert c.get("/api/runs").status_code == 401


def test_the_printed_link_uses_a_secret_localhost_name(settings):
    name = settings.loopback_name
    assert name.startswith("mg-") and name.endswith(".localhost") and len(name) > 20
    assert settings.login_url() == f"http://{name}:{PORT}/?token={TOKEN}"
    assert settings.fallback_login_url() == f"http://127.0.0.1:{PORT}/?token={TOKEN}"
    assert f"{name}:{PORT}" in settings.allowed_hosts()
    other = WebSettings(runs_dir=settings.runs_dir, token=TOKEN, port=PORT)
    assert other.loopback_name != name  # per launch
    plain = WebSettings(runs_dir=settings.runs_dir, token=TOKEN, port=PORT, loopback_name="")
    assert plain.login_url() == f"http://127.0.0.1:{PORT}/?token={TOKEN}" and plain.fallback_login_url() is None


def test_login_on_the_secret_name_scopes_the_cookie_to_it(app, settings):
    base = f"http://{settings.loopback_name}:{PORT}"
    with TestClient(app, base_url=base) as c:
        r = c.get(f"/?token={TOKEN}", follow_redirects=False)
        assert r.status_code == 303
        assert "domain=" not in r.headers["set-cookie"].lower()  # host-only: this name, not 127.0.0.1
        assert c.get("/api/runs").status_code == 200


def test_unauthenticated_responses_never_name_the_secret_host(anon, settings):
    name = settings.loopback_name
    for r in (anon.get("/", headers=HTML), anon.get("/api/runs"), anon.get("/?token=nope", headers=HTML),
              anon.get("/", headers={"host": "evil.example"}), anon.get("/?login=nope", headers=HTML)):
        assert name not in r.text and TOKEN not in r.text


def test_the_browser_link_is_one_time_and_carries_no_token(anon, settings):
    url = settings.browser_login_url()
    assert TOKEN not in url and "?login=" in url
    path = url[url.index("/?"):]
    r = anon.get(path, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert anon.get("/api/runs").status_code == 200
    anon.cookies.clear()
    r = anon.get(path, headers=HTML, follow_redirects=False)  # used once already
    assert r.status_code == 401 and "set-cookie" not in r.headers
    assert "expired or was already used" in r.text


def test_a_login_nonce_expires(settings):
    n = settings.sessions.new_login_nonce(ttl_s=-1)
    assert settings.sessions.consume_login_nonce(n) is False
    n = settings.sessions.new_login_nonce()
    assert settings.sessions.consume_login_nonce("x" + n) is False
    assert settings.sessions.consume_login_nonce(n) is True
    assert settings.sessions.consume_login_nonce(n) is False


def test_two_servers_use_different_cookie_names(tmp_path):
    a = WebSettings(runs_dir=tmp_path / "a", port=8765)
    b = WebSettings(runs_dir=tmp_path / "b", port=8766)
    assert a.cookie_name != b.cookie_name


def test_login_redirect_keeps_other_query_parameters(anon):
    r = anon.get(f"/compare?ids=a,b&token={TOKEN}&x=1", follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("/compare?") and "token" not in loc and TOKEN not in loc
    assert "ids=a%2Cb" in loc and "x=1" in loc


def test_login_redirect_never_leaves_the_site(anon):
    r = anon.get(f"{BASE}//evil.example/x?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("/") and not loc.startswith("//") and "evil.example/x" in loc


@pytest.mark.parametrize("bad", ["wrong", TOKEN + "x", TOKEN[:-1], ""])
def test_a_wrong_token_link_is_refused(anon, bad):
    r = anon.get(f"/?token={bad}", headers=HTML, follow_redirects=False)
    assert r.status_code == 401
    assert "set-cookie" not in r.headers


def test_a_token_link_only_logs_in_on_get(anon, settings):
    r = anon.post(f"/api/runs/x/delete?token={TOKEN}", headers={"x-csrf-token": settings.csrf_token})
    assert r.status_code == 401
    assert "set-cookie" not in r.headers


def test_bearer_token(anon):
    assert anon.get("/api/runs", headers={"authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert anon.get("/api/runs", headers={"authorization": f"bearer {TOKEN}"}).status_code == 200
    for value in ("Bearer nope", f"Basic {TOKEN}", TOKEN, "Bearer "):
        assert anon.get("/api/runs", headers={"authorization": value}).status_code == 401, value


def test_a_wrong_cookie_is_refused(app, settings):
    with TestClient(app, base_url=BASE) as c:
        c.cookies.set(settings.cookie_name, "not-a-session")
        assert c.get("/api/runs").status_code == 401
        c.cookies.set(settings.cookie_name, settings.sessions.new_session())
        assert c.get("/api/runs").status_code == 200


def test_401_page_does_not_leak_the_csrf_token(app):
    with TestClient(app, base_url=BASE) as c:
        c.get("/", headers=HTML)
    ctx = captured(app, "error.html.j2")
    assert ctx["status"] == 401 and ctx["csrf_token"] == ""


# --------------------------------------------------------------------------------------------------
# Host allowlist (DNS rebinding)
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "127.0.0.1:9999", "127.0.0.1",
                                  "localhost", "10.0.0.1:8765", "127.0.0.1:8765.evil.example", ""])
def test_unknown_host_headers_get_400(client, host):
    for path in ("/", "/healthz", "/static/app.css", f"/?token={TOKEN}"):
        r = client.get(path, headers={"host": host}, follow_redirects=False)
        assert r.status_code == 400, (host, path)
        assert "set-cookie" not in r.headers


@pytest.mark.parametrize("host", [f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}", f"LOCALHOST:{PORT}"])
def test_loopback_host_headers_are_accepted(client, host):
    assert client.get("/api/runs", headers={"host": host}).status_code == 200


def test_allow_remote_adds_only_the_bound_host(tmp_path):
    s = WebSettings(runs_dir=tmp_path / "r", host="10.1.2.3", port=9000, allow_remote=True, token=TOKEN)
    assert s.allowed_hosts() == {"127.0.0.1:9000", "localhost:9000", "[::1]:9000", "10.1.2.3:9000"}
    local = WebSettings(runs_dir=tmp_path / "r", port=9000, token=TOKEN)
    assert "10.1.2.3:9000" not in local.allowed_hosts()
    with pytest.raises(ValueError, match="allow_remote"):
        WebSettings(runs_dir=tmp_path / "r", host="10.1.2.3", token=TOKEN)
    with pytest.raises(ValueError, match="allow_remote"):
        WebSettings(runs_dir=tmp_path / "r", host="0.0.0.0", token=TOKEN)


# --------------------------------------------------------------------------------------------------
# CSRF
# --------------------------------------------------------------------------------------------------


def _post_targets(settings: WebSettings) -> list[str]:
    _cli_run(settings)
    return ["/runs", "/api/validate", "/api/runs/cli-run-1/cancel", "/api/runs/cli-run-1/delete"]


def _no_side_effects(settings: WebSettings) -> None:
    assert (settings.runs_dir / "cli-run-1" / "report.json").is_file()  # not deleted
    subs = settings.runs_dir / "submissions"
    assert not subs.exists() or list(subs.iterdir()) == []  # every refused upload cleaned up
    assert [p.name for p in settings.runs_dir.iterdir() if p.name not in ("cli-run-1", "submissions")] == []


def test_every_post_without_a_csrf_token_is_refused(client, settings):
    for url in _post_targets(settings):
        # no token at all: JSON body, empty body, form without the field
        assert client.post(url, json={"mode": "path"}).status_code == 403, url
        assert client.post(url).status_code == 403, url
        assert client.post(url, data={"mode": "path", "adapter_path": "/x.py"}).status_code == 403, url
        r = client.post(url, data={"mode": "path"}, files=[("model_files", ("m.txt", b"tree"))])
        assert r.status_code == 403, url
    _no_side_effects(settings)


def test_every_post_with_a_wrong_csrf_token_is_refused(client, settings):
    wrong = "0" * 64
    for url in _post_targets(settings):
        assert client.post(url, headers={"x-csrf-token": wrong}).status_code == 403, url
        assert client.post(url, headers={"x-csrf-token": TOKEN}).status_code == 403, url  # the token is not the csrf
        assert client.post(url, data={"csrf": wrong}).status_code == 403, url
        r = client.post(url, data={"csrf": wrong}, files=[("adapter_file", ("a.py", b"x = 1\n"))])
        assert r.status_code == 403, url
    _no_side_effects(settings)


def test_csrf_as_form_field_or_header_is_accepted(client, settings):
    _cli_run(settings, "cli-a")
    _cli_run(settings, "cli-b")
    r = client.post("/api/runs/cli-a/delete", data={"csrf": settings.csrf_token}, headers=HTML,
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    r = client.post("/api/runs/cli-b/delete", headers={"x-csrf-token": settings.csrf_token, **JSON})
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert not (settings.runs_dir / "cli-a").exists() and not (settings.runs_dir / "cli-b").exists()


def test_the_csrf_token_is_the_hmac_of_the_launch_token(settings):
    import hashlib
    import hmac

    assert settings.csrf_token == hmac.new(TOKEN.encode(), b"csrf", hashlib.sha256).hexdigest()
    assert settings.csrf_token != TOKEN


@pytest.mark.parametrize("headers", [
    {"origin": "http://evil.example"},
    {"origin": f"http://evil.example:{PORT}"},
    {"origin": f"http://127.0.0.1:{PORT + 1}"},
    {"referer": "http://evil.example/page"},
    {"sec-fetch-site": "cross-site"},
    {"sec-fetch-site": "same-site"},
    {"origin": "file://"},
])
def test_cross_origin_posts_are_refused_even_with_the_token(client, settings, headers):
    _cli_run(settings)
    r = client.post("/api/runs/cli-run-1/delete", headers={"x-csrf-token": settings.csrf_token, **headers})
    assert r.status_code == 403
    assert (settings.runs_dir / "cli-run-1").exists()


@pytest.mark.parametrize("headers", [
    {"origin": f"http://127.0.0.1:{PORT}"},
    {"origin": f"http://localhost:{PORT}", "sec-fetch-site": "same-origin"},
    {"origin": "null"},  # same-origin form post from a no-referrer page
    {"referer": f"http://127.0.0.1:{PORT}/runs/cli-run-1"},
])
def test_same_origin_posts_are_accepted(client, settings, headers):
    _cli_run(settings)
    r = client.post("/api/runs/cli-run-1/delete", headers={"x-csrf-token": settings.csrf_token, **headers, **JSON})
    assert r.status_code == 200, r.text


def test_pages_expose_the_csrf_token_to_the_forms(client, app, settings):
    r = client.get("/runs/new")
    assert r.status_code == 200
    assert captured(app, "new_run.html.j2")["csrf_token"] == settings.csrf_token
    assert settings.csrf_token in r.text
    assert TOKEN not in r.text  # the launch token itself never appears in a page


# --------------------------------------------------------------------------------------------------
# Headers
# --------------------------------------------------------------------------------------------------


def _assert_app_headers(r) -> None:
    h = r.headers
    assert h["content-security-policy"] == APP_CSP
    assert h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "no-referrer"
    assert h["cache-control"] == "no-store"


def test_security_headers_on_every_kind_of_response(client, anon, settings):
    _cli_run(settings)
    for path in ("/", "/runs/new", "/runs/cli-run-1", "/api/runs", "/api/runs/cli-run-1", "/runs/cli-run-1/report.json",
                 "/runs/cli-run-1/run.log", "/static/app.css", "/healthz", "/nope", "/runs/nope"):
        _assert_app_headers(client.get(path))
    _assert_app_headers(anon.get("/", headers=HTML))  # 401
    _assert_app_headers(client.get("/", headers={"host": "evil.example"}))  # 400
    _assert_app_headers(client.post("/api/runs/cli-run-1/cancel"))  # 403
    _assert_app_headers(anon.get(f"/?token={TOKEN}", follow_redirects=False))  # 303 login


def test_app_csp_forbids_inline_script_and_style():
    directives = dict(d.strip().split(" ", 1) for d in APP_CSP.split(";"))
    assert directives["script-src"] == "'self'" and directives["style-src"] == "'self'"
    assert directives["default-src"] == "'self'" and directives["base-uri"] == "'none'"
    assert directives["frame-ancestors"] == "'self'" and directives["form-action"] == "'self'"


def test_report_html_keeps_its_own_csp_and_is_sandboxed(client, settings):
    d = _cli_run(settings)
    body = '<!doctype html><meta http-equiv="Content-Security-Policy" content="default-src \'none\'"><p>r</p>'
    (d / "report.html").write_text(body)
    r = client.get("/runs/cli-run-1/report.html")
    assert r.status_code == 200 and r.text == body  # served as-is
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-security-policy"] == REPORT_CSP
    assert "sandbox allow-scripts" in REPORT_CSP and "allow-same-origin" not in REPORT_CSP
    assert "x-malvalid-raw-report" not in r.headers


# --------------------------------------------------------------------------------------------------
# Run ids and raw files
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rid", ["..", ".", ".hidden", "-x", "_x", "a" * 129, "a b", "a%00b", "submissions",
                                 "a/b", "a\\b", "ü", ""])
def test_invalid_run_ids(rid):
    assert not valid_run_id(rid)


def test_valid_run_ids():
    for rid in ("20260930T101500Z-deadbeef", "my-run_1.2", "a", "A" * 128):
        assert valid_run_id(rid)


@pytest.mark.parametrize("path", ["/runs/..%2F..%2Fetc%2Fpasswd", "/runs/%2e%2e/report.json", "/runs/.hidden",
                                  "/runs/..%5Creport.json", "/api/runs/%2e%2e", "/runs/submissions",
                                  "/runs/submissions/report.json", "/api/runs/submissions"])
def test_traversal_through_the_run_id_is_404(client, settings, path):
    _cli_run(settings)
    (settings.runs_dir / "report.json").write_text(json.dumps({"verdict": {}}))  # bait one level up
    r = client.get(path)
    assert r.status_code == 404, (path, r.status_code)


@pytest.mark.posix
def test_symlinked_run_dirs_are_neither_listed_nor_served(client, settings, tmp_path):
    outside = tmp_path / "outside"
    cli_run_dir(outside.parent, "outside", "sample_report.json", run__log="secret\n")
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(outside, settings.runs_dir / "linked")
    assert client.get("/runs/linked").status_code == 404
    assert client.get("/runs/linked/run.log").status_code == 404
    assert client.get("/api/runs/linked").status_code == 404
    assert [s["run_id"] for s in client.get("/api/runs").json()] == []
    store = RunStore(settings.runs_dir)
    with pytest.raises(RunNotFound):
        store.run_dir("linked")


def test_only_allowlisted_raw_files_are_served(client, settings):
    d = _cli_run(settings)
    (d / "job.json").write_text(json.dumps({"schema": "malvalid-job/1", "status": "finished", "run_id": "cli-run-1"}))
    (d / "progress.json").write_text(json.dumps({"stage": "done"}))
    (d / "notes.txt").write_text("x")
    for name, ctype in (("report.json", "application/json"), ("report.html", "text/html"),
                        ("run.log", "text/plain"), ("console.log", "text/plain")):
        r = client.get(f"/runs/cli-run-1/{name}")
        assert r.status_code == 200, name
        assert r.headers["content-type"].startswith(ctype)
    for name in ("job.json", "progress.json", "notes.txt", "private", "%2e%2e", "report.JSON", "report.json.tmp"):
        assert client.get(f"/runs/cli-run-1/{name}").status_code == 404, name


@pytest.mark.posix
def test_nothing_under_private_is_ever_served(client, settings):
    d = _cli_run(settings)
    (d / "private").mkdir()
    (d / "private" / "secret.bin").write_bytes(b"SECRET-PAYLOAD")
    (d / "private" / "report.json").write_text("{}")
    for path in ("/runs/cli-run-1/private/secret.bin", "/runs/cli-run-1/private/report.json",
                 "/runs/cli-run-1/private", "/runs/cli-run-1/private%2Fsecret.bin",
                 "/runs/cli-run-1/..%2Fcli-run-1%2Fprivate%2Fsecret.bin", "/static/../private/secret.bin"):
        r = client.get(path)
        assert r.status_code == 404, path
        assert b"SECRET-PAYLOAD" not in r.content
    # A raw-file name that is a symlink into private/ (or anywhere) is refused too.
    (d / "run.log").unlink()
    os.symlink(d / "private" / "secret.bin", d / "run.log")
    r = client.get("/runs/cli-run-1/run.log")
    assert r.status_code == 404 and b"SECRET-PAYLOAD" not in r.content
    assert client.get("/api/runs/cli-run-1").json()["files"]["run_log"] is False


def test_websocket_connections_are_closed(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/"):
            pass
