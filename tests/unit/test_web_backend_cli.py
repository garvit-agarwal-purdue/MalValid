"""``malvalid serve`` (``docs/WEB_CONTRACT.md`` §1, §8): flag validation (non-loopback hosts need
``--allow-remote``, token format, missing web deps ⇒ exit 2) and a real server subprocess on a free
port that prints the tokenized login URL and enforces the token and Host checks."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time

import pytest
from typer.testing import CliRunner

pytest.importorskip("starlette")
httpx = pytest.importorskip("httpx")

from malvalid.cli import app  # noqa: E402

pytestmark = pytest.mark.web

TOKEN = "cli-test-token-0123456789"
URL_RE = re.compile(r"http://127\.0\.0\.1:(\d+)/\?token=([A-Za-z0-9._~-]+)")


def invoke(*args: str):
    return CliRunner().invoke(app, ["serve", *args], env={"COLUMNS": "200"})


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "10.0.0.1", "example.org", ""])
def test_serve_refuses_non_loopback_hosts_without_allow_remote(tmp_path, host):
    r = invoke("--host", host, "--no-browser", "--runs-dir", str(tmp_path / "runs"))
    assert r.exit_code == 2, r.output
    assert "--allow-remote" in r.output


def test_serve_validates_its_flags(tmp_path):
    r = invoke("--token", "has spaces", "--no-browser")
    assert r.exit_code == 2 and "--token" in r.output
    r = invoke("--config", str(tmp_path / "missing.yaml"), "--no-browser")
    assert r.exit_code == 2 and "not found" in r.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("modules:\n  performance:\n    no_such_param: 1\n")
    assert invoke("--config", str(bad), "--no-browser").exit_code == 2
    f = tmp_path / "a-file"
    f.write_text("x")
    r = invoke("--runs-dir", str(f), "--no-browser")
    assert r.exit_code == 2 and "not a directory" in r.output
    assert invoke("--port", "70000").exit_code == 2
    assert invoke("--max-concurrent", "0").exit_code == 2


def test_serve_without_the_web_extra(monkeypatch):
    import malvalid.web

    monkeypatch.setattr(malvalid.web, "missing_web_dependencies", lambda: ["starlette", "uvicorn"])
    r = invoke("--no-browser")
    assert r.exit_code == 2
    assert "pip install -e '.[web]'" in r.output


def test_serve_port_in_use(tmp_path):
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        r = invoke("--port", str(s.getsockname()[1]), "--no-browser", "--runs-dir", str(tmp_path / "runs"))
        assert r.exit_code == 2 and "cannot listen" in r.output
    finally:
        s.close()


def _read_url(proc: subprocess.Popen[str], timeout: float = 60.0) -> tuple[int, str, list[str]]:
    lines: list[str] = []
    found: dict[str, tuple[int, str]] = {}

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            m = URL_RE.search(line)
            if m and "url" not in found:
                found["url"] = (int(m.group(1)), m.group(2))

    threading.Thread(target=reader, daemon=True).start()
    end = time.monotonic() + timeout
    while time.monotonic() < end and "url" not in found and proc.poll() is None:
        time.sleep(0.05)
    assert "url" in found, f"no login URL printed; output so far: {''.join(lines)!r}"
    return found["url"][0], found["url"][1], lines


def test_serve_end_to_end(tmp_path):
    runs = tmp_path / "runs"
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "COLUMNS": "200"}
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "malvalid", "serve", "--port", "0", "--token", TOKEN, "--runs-dir", str(runs)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(tmp_path),
    )
    try:
        port, token, lines = _read_url(proc)
        assert token == TOKEN and port > 0
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        while True:
            try:
                r = httpx.get(f"{base}/healthz", timeout=5)
                break
            except httpx.TransportError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        assert r.status_code == 200 and r.text == "ok"
        assert httpx.get(f"{base}/", timeout=5).status_code == 401
        r = httpx.get(f"{base}/?token={TOKEN}", timeout=5)
        assert r.status_code == 303 and r.headers["location"] == "/"
        cookie = r.headers["set-cookie"]
        assert TOKEN not in cookie and cookie.startswith(f"malvalid_session_{port}=")
        sid = cookie.split(";", 1)[0].split("=", 1)[1]
        with httpx.Client(base_url=base, cookies={f"malvalid_session_{port}": sid}, timeout=30) as c:
            page = c.get("/")
            assert page.status_code == 200 and "text/html" in page.headers["content-type"]
            assert page.headers["content-security-policy"].startswith("default-src 'self'")
            assert c.get("/api/runs").json() == []
            assert c.get("/", headers={"host": f"evil.example:{port}"}).status_code == 400
            assert c.get("/static/app.css").status_code == 200
        assert "httponly" in cookie.lower()
        assert runs.is_dir()
    finally:
        if os.name == "nt":
            proc.terminate()  # Windows cannot deliver SIGINT to a child process
        else:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    err = proc.stderr.read() if proc.stderr else ""
    assert proc.returncode == 0 or os.name == "nt", err  # graceful SIGINT shutdown (POSIX)
    assert TOKEN not in err  # no access log carrying the login URL


# ---- token handling (security-token-in-browser-argv, security-cookie-cross-port-leak) ------------


@pytest.fixture
def captured_serve(monkeypatch):
    """Run `malvalid serve` up to the point where the server would start; capture its settings."""
    from malvalid.web import serve as web_serve

    seen: dict = {}

    def fake_run_server(settings, sock, *, open_browser=False, log_level="warning"):
        seen["settings"] = settings
        seen["open_browser"] = open_browser

    monkeypatch.setattr(web_serve, "run_server", fake_run_server)
    monkeypatch.delenv("MALVALID_SERVE_TOKEN", raising=False)
    return seen


def test_serve_reads_the_token_from_a_file(tmp_path, captured_serve):
    f = tmp_path / "token.txt"
    f.write_text(TOKEN + "\n")
    r = invoke("--token-file", str(f), "--port", "0", "--no-browser", "--runs-dir", str(tmp_path / "runs"))
    assert r.exit_code == 0, r.output
    assert captured_serve["settings"].token == TOKEN
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    r = invoke("--token-file", str(empty), "--port", "0", "--no-browser")
    assert r.exit_code == 2 and "empty" in r.output
    r = invoke("--token-file", str(tmp_path / "no-dir" / "nope.txt"), "--port", "0", "--no-browser")
    assert r.exit_code == 2 and "cannot create" in r.output  # a missing file is created, not its folder


def test_serve_reads_the_token_from_the_environment(tmp_path, captured_serve):
    r = CliRunner().invoke(app, ["serve", "--port", "0", "--no-browser", "--runs-dir", str(tmp_path / "runs")],
                           env={"COLUMNS": "200", "MALVALID_SERVE_TOKEN": TOKEN})
    assert r.exit_code == 0, r.output
    assert captured_serve["settings"].token == TOKEN


def test_serve_prints_the_private_localhost_link_and_a_fallback(tmp_path, captured_serve):
    r = invoke("--token", TOKEN, "--port", "0", "--no-browser", "--runs-dir", str(tmp_path / "runs"))
    assert r.exit_code == 0, r.output
    s = captured_serve["settings"]
    assert f"http://{s.loopback_name}:{s.port}/?token={TOKEN}" in r.output
    assert f"http://127.0.0.1:{s.port}/?token={TOKEN}" in r.output


def test_the_launched_browser_gets_a_one_time_link(tmp_path, monkeypatch):
    """The URL handed to the browser (and so to its command line) carries a nonce, not the token."""
    import uvicorn

    from malvalid.web import serve as web_serve
    from malvalid.web.settings import WebSettings

    opened: list[str] = []

    class FakeTimer:
        def __init__(self, delay, fn, args=()):
            self.fn, self.args, self.daemon = fn, args, False

        def start(self):
            self.fn(*self.args)

    monkeypatch.setattr(web_serve.threading, "Timer", FakeTimer)
    monkeypatch.setattr(web_serve.webbrowser, "open", lambda url, new=0: opened.append(url) or True)
    monkeypatch.setattr(uvicorn.Server, "run", lambda self, sockets=None: None)
    s = WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=8765)
    web_serve.run_server(s, sock=None, open_browser=True)  # type: ignore[arg-type]
    assert len(opened) == 1
    url = opened[0]
    assert TOKEN not in url and url.startswith(f"{s.base_url}/?login=")
    assert s.sessions.consume_login_nonce(url.split("login=", 1)[1]) is True


def test_run_title_flag_sets_the_report_title_without_touching_the_policy(tmp_path, monkeypatch):
    """`malvalid serve` passes a run title as the hidden `malvalid run --title` (not a rewritten policy)."""
    import malvalid.runner as runner

    seen = {}

    class Stop(Exception):
        pass

    def fake_run_gate(opts):
        seen["opts"] = opts
        raise Stop

    monkeypatch.setattr(runner, "run_gate", fake_run_gate)
    pol = tmp_path / "gate.yaml"
    pol.write_text("report:\n  title: from policy\n")
    ad = tmp_path / "ad.py"
    ad.write_text("x = 1\n")
    r = CliRunner().invoke(app, ["run", "--adapter", str(ad), "--config", str(pol), "--out", str(tmp_path / "o"),
                                 "--title", "  web   title "])
    assert "opts" in seen, r.output
    cfg = seen["opts"].config
    assert cfg.report.title == "web title"
    assert cfg.source_path == str(pol.resolve())  # relative paths still resolve next to the user's policy
    assert "--title" not in seen["opts"].command
