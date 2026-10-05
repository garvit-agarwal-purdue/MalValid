"""``malvalid serve`` behind a reverse proxy (``--root-path`` / ``--trusted-host``, ``docs/web.md``).

Open OnDemand reaches a compute-node port as ``https://<gateway>/node/<host>/<port>/...`` (prefix
kept) or ``/rnode/<host>/<port>/...`` (prefix stripped). Every link, asset, form action, iframe,
redirect and the cookie ``Path`` must carry the prefix; requests must route with or without it; the
proxy's public host is accepted in the Host and Origin/Referer checks without weakening the token
login, CSRF or the security headers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from malvalid.web.security import APP_CSP  # noqa: E402
from malvalid.web.settings import WebSettings, normalize_root_path, normalize_trusted_host  # noqa: E402
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    PORT,
    TOKEN,
    cli_run_dir,
    fake_malvalid,
    make_app,
)

pytestmark = pytest.mark.web

ROOT = "/node/b002.example.edu/8765"
GATEWAY = "gateway.example.edu"
NODE = "b002.example.edu"
PUBLIC = f"https://{GATEWAY}"
HTML = {"accept": "text/html,application/xhtml+xml"}
JSON = {"accept": "application/json"}
URL_ATTR = re.compile(r'\b(?:href|src|action|data-next)="([^"#][^"]*)"')


@pytest.fixture
def settings(tmp_path: Path, fake_malvalid: Path) -> WebSettings:
    return WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=PORT, cancel_grace_s=1.0,
                       command_prefix=(sys.executable, str(fake_malvalid)), validate_timeout_s=30,
                       root_path=ROOT + "/", trusted_hosts=(GATEWAY, NODE), allow_path_mode=True)


@pytest.fixture
def app(settings: WebSettings) -> Any:
    return make_app(settings)


def _client(app: Any, base: str = PUBLIC, login: bool = True) -> TestClient:
    c = TestClient(app, base_url=base)
    if login:
        s = app.state.settings
        c.cookies.set(s.cookie_name, s.sessions.new_session())
    return c


@pytest.fixture
def client(app: Any):
    with _client(app) as c:
        yield c


@pytest.fixture
def anon(app: Any):
    with _client(app, login=False) as c:
        yield c


def _cli_run(settings: WebSettings, run_id: str = "cli-run-1") -> Path:
    return cli_run_dir(settings.runs_dir, run_id, "sample_report.json",
                       report__html="<!doctype html><title>r</title>", run__log="log\n", console__log="c\n")


def _urls(html: str) -> list[str]:
    return [u for u in URL_ATTR.findall(html) if not u.startswith(("http:", "https:", "mailto:"))]


# --------------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw, want", [("", ""), ("/", ""), (None, ""), ("/node/h/8765", "/node/h/8765"),
                                       ("/node/h/8765/", "/node/h/8765"), ("node/h/1", "/node/h/1"),
                                       ("/rnode/b002.cluster.example.edu/8765",
                                        "/rnode/b002.cluster.example.edu/8765")])
def test_root_path_is_normalized(raw, want):
    assert normalize_root_path(raw) == want


@pytest.mark.parametrize("raw", ["/a/../b", "/a//b", "/a b", "/a?x=1", "/a#f", "/./a", "/a/<x>", '/a"b', "/%2e%2e"])
def test_bad_root_paths_are_refused(raw):
    with pytest.raises(ValueError):
        normalize_root_path(raw)


@pytest.mark.parametrize("raw", ["", "*", "*.example.edu", "https://gw.example.edu", "gw.example.edu/x",
                                 "gw example", "gw.example.edu:abc"])
def test_bad_trusted_hosts_are_refused(raw):
    with pytest.raises(ValueError):
        normalize_trusted_host(raw)


def test_trusted_hosts_extend_the_allowlist(settings, tmp_path):
    allowed = settings.allowed_hosts()
    assert {GATEWAY, f"{GATEWAY}:{PORT}", NODE, f"{NODE}:{PORT}"} <= allowed
    exact = WebSettings(runs_dir=tmp_path / "r", token=TOKEN, port=PORT, trusted_hosts=("GW.Example.edu:443",))
    assert "gw.example.edu:443" in exact.allowed_hosts()
    assert "gw.example.edu" not in exact.allowed_hosts() and f"gw.example.edu:{PORT}" not in exact.allowed_hosts()
    plain = WebSettings(runs_dir=tmp_path / "r", token=TOKEN, port=PORT)
    assert not any(GATEWAY in h for h in plain.allowed_hosts())
    assert plain.root_path == "" and plain.cookie_path == "/" and plain.proxy_login_url() is None


def test_proxy_settings_do_not_need_allow_remote(settings):
    assert settings.host == "127.0.0.1" and not settings.allow_remote
    assert settings.root_path == ROOT and settings.cookie_path == ROOT + "/"
    assert settings.url("/runs/x") == f"{ROOT}/runs/x" and settings.url("/") == f"{ROOT}/"
    assert settings.proxy_login_url() == f"{PUBLIC}{ROOT}/?token={TOKEN}"
    assert settings.login_url().endswith(f":{PORT}{ROOT}/?token={TOKEN}")
    shown = settings.public_dict()
    assert shown["root_path"] == ROOT and shown["trusted_hosts"] == [GATEWAY, NODE]


# --------------------------------------------------------------------------------------------------
# Host / Origin checks
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("host", [GATEWAY, f"{NODE}:{PORT}", f"{GATEWAY}:{PORT}", "127.0.0.1:8765"])
def test_trusted_hosts_are_accepted(app, host):
    with _client(app) as c:
        assert c.get(ROOT + "/", headers={"host": host, **HTML}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", f"evil.example:{PORT}", f"{GATEWAY}.evil.example",
                                  f"x{GATEWAY}", f"{GATEWAY}:9999", f"{NODE}:9999", "", "b003.example.edu:8765"])
def test_untrusted_hosts_are_rejected(app, host):
    with _client(app) as c:
        r = c.get(ROOT + "/", headers={"host": host, **HTML})
    assert r.status_code == 400 and "Host header" in r.text


def test_x_forwarded_headers_do_not_bypass_the_host_check(app):
    with _client(app) as c:
        r = c.get(ROOT + "/", headers={"host": "evil.example", "x-forwarded-host": GATEWAY,
                                       "x-forwarded-proto": "https", "x-forwarded-for": "127.0.0.1"})
    assert r.status_code == 400


def test_a_trusted_host_is_not_a_login(anon):
    """The proxy's own authentication is extra: the per-launch token is still required."""
    for path in (ROOT + "/", "/", ROOT + "/api/runs", ROOT + "/runs/new"):
        r = anon.get(path, headers={"x-forwarded-user": "someone", **HTML})
        assert r.status_code == 401, path
        assert TOKEN not in r.text
    assert anon.get(ROOT + "/api/runs").json()["status"] == 401


def _delete(c: TestClient, settings: WebSettings, origin: str | None = None, **headers: str):
    h = {"x-csrf-token": settings.csrf_token, **JSON, **headers}
    if origin is not None:
        h["origin"] = origin
    return c.post(f"{ROOT}/api/runs/cli-run-1/delete", headers=h)


def test_posts_from_the_trusted_origin_are_accepted(client, settings):
    _cli_run(settings)
    r = _delete(client, settings, PUBLIC, **{"sec-fetch-site": "same-origin"})
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert not (settings.runs_dir / "cli-run-1").exists()


@pytest.mark.parametrize("headers", [
    {"origin": "https://evil.example"},
    {"origin": f"https://{GATEWAY}.evil.example"},
    {"origin": f"https://{GATEWAY}:9999"},
    {"referer": "https://evil.example/node/b002.example.edu/8765/"},
    {"sec-fetch-site": "cross-site"},
    {"sec-fetch-site": "same-site"},
])
def test_cross_origin_posts_are_refused_behind_the_proxy(client, settings, headers):
    _cli_run(settings)
    r = client.post(f"{ROOT}/api/runs/cli-run-1/delete",
                    headers={"x-csrf-token": settings.csrf_token, **JSON, **headers})
    assert r.status_code == 403
    assert (settings.runs_dir / "cli-run-1" / "report.json").is_file()


def test_csrf_is_still_required_behind_the_proxy(client, settings):
    _cli_run(settings)
    r = client.post(f"{ROOT}/api/runs/cli-run-1/delete", headers={"origin": PUBLIC, **JSON})
    assert r.status_code == 403
    r = client.post(f"{ROOT}/api/runs/cli-run-1/delete", headers={"origin": PUBLIC, "x-csrf-token": "nope", **JSON})
    assert r.status_code == 403
    r = client.post(f"{ROOT}/api/runs/cli-run-1/delete", data={"csrf": "nope"}, headers={"origin": PUBLIC, **HTML})
    assert r.status_code == 403
    assert (settings.runs_dir / "cli-run-1" / "report.json").is_file()


# --------------------------------------------------------------------------------------------------
# Login, cookie and redirects under the prefix
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [ROOT + "/", "/"])  # prefix kept (/node/) or stripped (/rnode/)
def test_login_under_the_prefix(app, settings, path):
    with _client(app, login=False) as c:
        r = c.get(f"{path}?token={TOKEN}&x=1", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == f"{ROOT}/?x=1"  # path-only: same site, with the prefix
        cookie = r.headers["set-cookie"]
        assert cookie.startswith(f"{settings.cookie_name}=") and TOKEN not in cookie
        low = cookie.lower()
        assert f"path={ROOT.lower()}/" in low and "httponly" in low and "samesite=strict" in low
        # The browser now holds the cookie for the prefix only.
        assert c.get(ROOT + "/", headers=HTML).status_code == 200
        assert c.get(ROOT + "/api/runs").status_code == 200
        # A sibling app on the same proxy origin (another port) never receives it.
        assert c.get("/node/b002.example.edu/87650/", headers=HTML).status_code == 401
        assert c.get("/node/b002.example.edu/8766/", headers=HTML).status_code == 401


def test_login_redirect_never_leaves_the_site_under_the_prefix(anon):
    for p in (f"{ROOT}//evil.example/x", f"{ROOT}/\\evil.example", "//evil.example", "/\\\\evil.example"):
        r = anon.get(f"{p}?token={TOKEN}", follow_redirects=False)
        assert r.status_code == 303, p
        loc = r.headers["location"]
        assert loc.startswith(ROOT + "/") and not loc.startswith(ROOT + "//") and "\\" not in loc[len(ROOT):], loc


def test_one_time_login_link_under_the_prefix(app, settings):
    url = settings.browser_login_url()
    assert f"{ROOT}/?login=" in url
    nonce = url.split("login=")[1]
    with _client(app, login=False) as c:
        r = c.get(f"{ROOT}/?login={nonce}", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == f"{ROOT}/"
        assert c.get(f"{ROOT}/?login={nonce}", follow_redirects=False).status_code == 401  # used up


def test_the_bare_prefix_redirects_to_the_slash(anon):
    r = anon.get(f"{ROOT}?token={TOKEN}", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == f"{ROOT}/?token={TOKEN}"


def test_bad_token_under_the_prefix_is_refused(anon):
    r = anon.get(f"{ROOT}/?token=wrong", headers=HTML, follow_redirects=False)
    assert r.status_code == 401 and "set-cookie" not in r.headers


def test_unauthenticated_page_points_to_the_prefixed_link(anon):
    r = anon.get(ROOT + "/runs/new", headers=HTML)
    assert r.status_code == 401
    assert f"{ROOT}/?token=" in r.text
    assert all(u.startswith(ROOT + "/") for u in _urls(r.text)), _urls(r.text)


# --------------------------------------------------------------------------------------------------
# Links, assets, forms, iframes and JS under the prefix
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/runs/new", "/compare", "/corpora", "/modules", "/system",
                                  "/runs/cli-run-1", "/compare?ids=cli-run-1,cli-run-2", "/no-such-page"])
def test_every_generated_url_carries_the_prefix(client, settings, path):
    _cli_run(settings)
    _cli_run(settings, "cli-run-2")
    r = client.get(ROOT + path, headers=HTML)
    assert r.status_code in (200, 404), (path, r.status_code)
    urls = _urls(r.text)
    assert urls, path
    bad = [u for u in urls if not u.startswith(ROOT + "/")]
    assert not bad, (path, bad)
    assert f'<meta name="root-path" content="{ROOT}">' in r.text
    assert r.headers["content-security-policy"] == APP_CSP


def test_run_page_iframe_and_raw_files_under_the_prefix(client, settings):
    _cli_run(settings)
    r = client.get(f"{ROOT}/runs/cli-run-1", headers=HTML)
    assert r.status_code == 200
    assert f'<iframe src="{ROOT}/runs/cli-run-1/report.html"' in r.text
    assert f'href="{ROOT}/runs/cli-run-1/report.json"' in r.text
    for name in ("report.html", "report.json", "run.log", "console.log"):
        assert client.get(f"{ROOT}/runs/cli-run-1/{name}").status_code == 200, name


def test_the_new_run_form_and_demo_form_post_under_the_prefix(client):
    r = client.get(ROOT + "/runs/new", headers=HTML)
    assert f'action="{ROOT}/runs"' in r.text
    r = client.get(ROOT + "/", headers=HTML)
    assert f'action="{ROOT}/runs"' in r.text or f'action="{ROOT}/compare"' in r.text


@pytest.mark.parametrize("prefix", [ROOT, ""])
def test_static_assets_with_and_without_the_prefix(anon, prefix):
    for name in ("app.css", "app.js", "favicon.svg"):
        r = anon.get(f"{prefix}/static/{name}")
        assert r.status_code == 200, name
    assert anon.get(f"{prefix}/static/nope.css").status_code == 404
    assert anon.get(f"{prefix}/healthz").text == "ok"


def test_routes_work_with_and_without_the_prefix(client):
    for p in ("/", "/runs/new", "/api/runs"):
        assert client.get(ROOT + p, headers=HTML).status_code == 200, p
        assert client.get(p, headers=HTML).status_code == 200, p


def test_the_prefix_is_matched_on_segment_boundaries(client):
    assert client.get(ROOT + "0/", headers=HTML).status_code == 404  # /node/.../87650/ is not ours


def test_post_redirects_carry_the_prefix(client, settings):
    _cli_run(settings)
    r = client.post(f"{ROOT}/api/runs/cli-run-1/delete", data={"csrf": settings.csrf_token},
                    headers={"origin": PUBLIC, **HTML}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == f"{ROOT}/"


def test_submitted_run_redirects_under_the_prefix(client, settings, tmp_path):
    adapter = tmp_path / "adapter.py"
    adapter.write_text("class A: pass\n")
    from tests.unit.test_web_backend_support import post_run

    r = post_run(client, settings, {"mode": "path", "adapter_path": str(adapter), "corpus": "synthetic_v2"},
                 url=f"{ROOT}/runs", headers={"origin": "null", **HTML})
    assert r.status_code == 303, r.text[:2000]
    assert re.fullmatch(re.escape(ROOT) + r"/runs/[A-Za-z0-9][A-Za-z0-9._-]*", r.headers["location"])


def test_js_builds_fetch_urls_from_the_root_path_meta():
    js = (Path(__file__).resolve().parents[2] / "src/malvalid/web/static/app.js").read_text(encoding="utf-8")
    assert 'meta[name="root-path"]' in js
    for call in ('appUrl("/api/runs/"', 'appUrl("/api/runs?active=1")', 'appUrl("/compare?ids="',
                 'appUrl("/api/validate")'):
        assert call in js, call
    assert not re.search(r'(fetch|api|assign)\(\s*"/', js)


def test_without_a_root_path_nothing_changes(tmp_path, fake_malvalid):
    s = WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=PORT,
                    command_prefix=(sys.executable, str(fake_malvalid)))
    app = make_app(s)
    with TestClient(app, base_url=f"http://127.0.0.1:{PORT}") as c:
        r = c.get(f"/?token={TOKEN}", follow_redirects=False)
        assert r.headers["location"] == "/" and "path=/;" in r.headers["set-cookie"].lower().replace(" ", "") + ";"
        page = c.get("/", headers=HTML).text
        assert '<meta name="root-path" content="">' in page
        assert all(u.startswith("/") and not u.startswith("//") for u in _urls(page))
        assert c.get("/", headers={"host": GATEWAY}).status_code == 400


# --------------------------------------------------------------------------------------------------
# CLI: flags, printed link, and a real server behind simulated Open OnDemand headers
# --------------------------------------------------------------------------------------------------


def _invoke(*args: str):
    from typer.testing import CliRunner

    from malvalid.cli import app as cli_app

    return CliRunner().invoke(cli_app, ["serve", *args], env={"COLUMNS": "300"})


@pytest.fixture
def captured_serve(monkeypatch):
    from malvalid.web import serve as web_serve

    seen: dict = {}

    def fake_run_server(settings, sock, *, open_browser=False, log_level="warning"):
        seen["settings"] = settings

    monkeypatch.setattr(web_serve, "run_server", fake_run_server)
    monkeypatch.delenv("MALVALID_SERVE_TOKEN", raising=False)
    return seen


def test_serve_flags_and_printed_proxy_link(tmp_path, captured_serve):
    r = _invoke("--token", TOKEN, "--port", "0", "--no-browser", "--runs-dir", str(tmp_path / "runs"),
                "--root-path", "/node/b002.example.edu/8765/", "--trusted-host", "Gateway.Example.edu",
                "--trusted-host", "b002.example.edu", "--allow-client", "198.51.100.18")
    assert r.exit_code == 0, r.output
    s = captured_serve["settings"]
    assert s.root_path == ROOT and s.trusted_hosts == (GATEWAY, NODE) and not s.allow_remote
    assert s.allowed_clients == ("198.51.100.18/32",)
    assert f"https://{GATEWAY}{ROOT}/?token={TOKEN}" in r.output
    assert f"http://127.0.0.1:{s.port}{ROOT}/?token={TOKEN}" in r.output


@pytest.mark.parametrize("args", [["--root-path", "/a/../b"], ["--trusted-host", "*.example.edu"],
                                  ["--trusted-host", "https://gw.example.edu"], ["--allow-client", "gw.example"]])
def test_serve_refuses_bad_proxy_flags(tmp_path, captured_serve, args):
    r = _invoke("--port", "0", "--no-browser", "--runs-dir", str(tmp_path / "runs"), *args)
    assert r.exit_code == 2, r.output
    assert "settings" not in captured_serve


def test_real_server_behind_simulated_ondemand_headers(tmp_path):
    """A real ``malvalid serve`` on loopback, requested exactly as an Open OnDemand ``/node/`` proxy
    would (Host = the backend ``<node>:<port>``, path with the prefix, browser Origin = the gateway)."""
    import os
    import signal
    import socket
    import subprocess
    import time

    httpx = pytest.importorskip("httpx")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    root = f"/node/{NODE}/{port}"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "COLUMNS": "300"}
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "malvalid", "serve", "--port", str(port), "--token", TOKEN, "--no-browser",
         "--runs-dir", str(tmp_path / "runs"), "--root-path", root, "--trusted-host", GATEWAY,
         "--trusted-host", NODE],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(tmp_path))
    base = f"http://127.0.0.1:{port}"
    proxy = {"host": f"{NODE}:{port}", "x-forwarded-host": GATEWAY, "x-forwarded-proto": "https",
             "x-forwarded-for": "10.0.0.9"}
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                httpx.get(f"{base}/healthz", timeout=5)
                break
            except httpx.TransportError:
                assert proc.poll() is None and time.monotonic() < deadline
                time.sleep(0.1)
        with httpx.Client(base_url=base, timeout=30, headers=proxy) as c:
            assert c.get(f"{root}/", headers=HTML).status_code == 401
            r = c.get(f"{root}/?token={TOKEN}")
            assert r.status_code == 303 and r.headers["location"] == f"{root}/"
            assert f"path={root}/" in r.headers["set-cookie"].lower()
            sid = r.headers["set-cookie"].split(";", 1)[0].split("=", 1)[1]
            c.cookies.set(f"malvalid_session_{port}", sid)
            page = c.get(f"{root}/", headers=HTML)
            assert page.status_code == 200
            assert f'href="{root}/static/app.css' in page.text
            csrf = re.search(r'name="csrf-token" content="([0-9a-f]+)"', page.text).group(1)
            assert c.get(f"{root}/static/app.css").status_code == 200
            assert c.get(f"{root}/api/runs").json() == []
            cli_run_dir(tmp_path / "runs", "cli-run-1", "sample_report.json")
            ok = c.post(f"{root}/api/runs/cli-run-1/delete",
                        headers={"origin": f"https://{GATEWAY}", "x-csrf-token": csrf, **JSON})
            assert ok.status_code == 200 and ok.json()["deleted"] is True
            bad = c.post(f"{root}/api/runs/x/delete",
                         headers={"origin": "https://evil.example", "x-csrf-token": csrf, **JSON})
            assert bad.status_code == 403
            assert c.get(f"{root}/", headers={"host": f"evil.example:{port}"}).status_code == 400
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    err = proc.stderr.read() if proc.stderr else ""
    assert proc.returncode == 0, err
    assert TOKEN not in err


# --------------------------------------------------------------------------------------------------
# --allow-client: only the proxy (and loopback) may connect
# --------------------------------------------------------------------------------------------------


def test_allow_client_limits_tcp_peers(tmp_path, fake_malvalid):
    from malvalid.web.settings import normalize_client_network

    s = WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=PORT, root_path=ROOT, trusted_hosts=(GATEWAY,),
                    allowed_clients=("198.51.100.18",), command_prefix=(sys.executable, str(fake_malvalid)))
    assert s.allowed_clients == ("198.51.100.18/32",)
    assert s.client_allowed("198.51.100.18") and s.client_allowed("127.0.0.1") and s.client_allowed("::1")
    assert s.client_allowed("::ffff:198.51.100.18")
    assert not s.client_allowed("198.51.100.19") and not s.client_allowed("testclient")
    assert not s.client_allowed(None)
    assert WebSettings(runs_dir=tmp_path / "r", token=TOKEN).client_allowed("10.9.9.9")  # unset: any peer
    with pytest.raises(ValueError):
        normalize_client_network("gateway.example.edu")
    app = make_app(s)
    for peer, want in (("198.51.100.18", 200), ("127.0.0.1", 200), ("203.0.113.7", 403)):
        with TestClient(app, base_url=PUBLIC, client=(peer, 40000)) as c:
            c.cookies.set(s.cookie_name, s.sessions.new_session())
            r = c.get(ROOT + "/", headers={"x-forwarded-for": "198.51.100.18", **HTML})
            assert r.status_code == want, (peer, r.status_code)
            assert c.get(ROOT + "/healthz").status_code == want
