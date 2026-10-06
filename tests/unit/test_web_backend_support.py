"""Shared fixtures for the web-backend tests (``malvalid serve``).

* ``settings`` / ``app`` / ``client``: an app over a temp runs dir, reached through Starlette's
  TestClient at ``http://127.0.0.1:8765`` with the session cookie set (``anon`` has no cookie).
  ``app.state.renderer.captured`` records every rendered ``(template, context)``.
* ``fake_malvalid``: a tiny stand-in for ``python -m malvalid`` (stdlib only, starts in ~50 ms) used
  as the job command. Its behaviour is chosen by a ``# FAKE: <mode>`` line in the submitted adapter
  (``ok``, ``exit1``, ``exit2``, ``sleep``, ``ignore-term``, ``crash``, ``no-report``); for
  ``validate-adapter`` it prints a JSON report (``invalid``, ``garbage``, ``sleep`` modes).
  Every invocation appends ``{argv, cwd, env}`` to ``$FAKE_RECORD`` when set.
* ``toy_python_path``: a directory with a ``sitecustomize.py`` importing :mod:`malvalid.testing`, so a
  *real* ``malvalid`` subprocess knows the toy ``toy_v1`` schema and ``toy_v1_corpus`` corpus.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip("starlette")
pytest.importorskip("httpx")

from starlette.testclient import TestClient  # noqa: E402

from malvalid.web.app import create_app  # noqa: E402
from malvalid.web.settings import COOKIE_NAME, WebSettings  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
TOKEN = "test-token-0123456789abcdef"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"

FAKE_MALVALID = r'''
import json, os, signal, sys, time

args = sys.argv[1:]
cmd = args[0] if args else ""


def opt(name, default=None):
    if name in args:
        i = args.index(name)
        return args[i + 1] if i + 1 < len(args) else default
    return default


adapter = opt("--adapter")
mode = "ok"
try:
    with open(adapter, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# FAKE:"):
                mode = line.split(":", 1)[1].strip()
except (OSError, TypeError):
    pass
rec = os.environ.get("FAKE_RECORD")
if rec:
    with open(rec, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"cmd": cmd, "argv": args, "cwd": os.getcwd(), "mode": mode,
                             "PYTHONSAFEPATH": os.environ.get("PYTHONSAFEPATH"),
                             "FAKE_EXTRA": os.environ.get("FAKE_EXTRA")}) + "\n")

if cmd == "validate-adapter":
    if mode == "sleep":
        time.sleep(60)
    if mode == "garbage":
        print("this is not json")
        print("error: the adapter exploded while importing", file=sys.stderr)
        sys.exit(2)
    ok = mode != "invalid"
    print(json.dumps({"ok": ok, "adapter_path": adapter, "n_failed": 0 if ok else 1, "argv": args,
                      "checks": [{"name": "declarations", "ok": True, "status": "pass", "detail": "fake"},
                                 {"name": "load", "ok": ok, "status": "pass" if ok else "fail",
                                  "detail": "fake load"}]}))
    sys.exit(0 if ok else 1)

if cmd != "run":
    print("error: fake malvalid only knows run and validate-adapter", file=sys.stderr)
    sys.exit(2)

out = opt("--out")
print("fake malvalid run: starting", flush=True)
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
modules = [
    {"module_id": "file_safety", "code": "M0", "title": "Model file safety", "status": "pass",
     "gate": "hard", "gate_outcome": "passed", "checks": [], "metrics": {}, "params": {}, "score": 1.0},
    {"module_id": "performance", "code": "M1", "title": "Performance & calibration", "status": "pass",
     "gate": "hard", "gate_outcome": "passed", "score": 0.9, "params": {"max_fpr": 0.01},
     "metrics": {"fpr": 0.004, "detection_rate": 0.97},
     "checks": [{"name": "fpr", "metric": "fpr", "op": "<=", "threshold": 0.01, "value": 0.004, "passed": True},
                {"name": "detection_rate", "metric": "detection_rate", "op": ">=", "threshold": 0.95,
                 "value": 0.97, "passed": True}]},
]
progress = {"schema": "malvalid-progress/1", "run_id": "fake", "pid": os.getpid(), "stage": "modules",
            "message": "running", "module": {"id": "performance", "code": "M1", "title": "Performance"},
            "modules": [{"id": "file_safety", "code": "M0", "title": "Model file safety", "status": "pass",
                         "duration_s": 0.1},
                        {"id": "performance", "code": "M1", "title": "Performance", "status": "running",
                         "duration_s": None}],
            "started_at": now, "updated_at": now, "verdict": None, "exit_code": None}


def write(name, data):
    tmp = os.path.join(out, "." + name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, os.path.join(out, name))


os.makedirs(out, exist_ok=True)
if mode == "ignore-term":  # before progress.json exists, so a test can wait for it
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
write("progress.json", progress)
if mode == "sleep":
    time.sleep(120)
if mode == "ignore-term":
    for _ in range(1200):
        time.sleep(0.1)
if mode == "exit2":
    print("error: my_adapter.py: class Foo does not declare feature_version", file=sys.stderr)
    sys.exit(2)
if mode == "crash":
    os.kill(os.getpid(), signal.SIGKILL)
if mode == "no-report":
    sys.exit(0)
code = 1 if mode == "exit1" else (2 if mode == "blocked2" else 0)
verdict = {"verdict": "ready" if code == 0 else "blocked", "label": "fake label", "score": 91.5 if code == 0 else 40.0,
           "coverage": 1.0, "axes": [{"module_id": "performance", "score": 90.0}]}
report = {"schema_version": "malvalid-report/1", "title": opt("--title"),
          "run": {"id": "fake-run", "started_at": now, "finished_at": now, "duration_s": 1.5},
          "verdict": verdict,
          "gate": {"exit_code": code, "fail_on": opt("--fail-on", "blocked"),
                   "exit_meaning": "error: a module errored or the model could not be loaded" if code == 2 else ""},
          "model": {"class_name": "FakeDetector", "adapter_path": adapter, "model_kind": "lightgbm",
                    "feature_version": "toy_v1", "operating_threshold": 0.5},
          "corpus": {"name": opt("--corpus", "toy_v1_corpus"), "content_hash": "abc"},
          "config": {"corpus": opt("--corpus", "toy_v1_corpus"), "runtime": {"seed": int(opt("--seed", "0"))}},
          "modules": modules}
progress.update(stage="done", module=None, verdict={k: verdict[k] for k in ("verdict", "label", "score", "coverage")},
                exit_code=code)
progress["modules"][1]["status"] = "pass"
write("report.json", report)
with open(os.path.join(out, "report.html"), "w", encoding="utf-8") as fh:
    fh.write("<!doctype html><html><head><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'\">"
             "<title>fake report</title></head><body>fake report</body></html>")
write("progress.json", progress)
print("fake malvalid run: done", flush=True)
sys.exit(code)
'''

TOY_SITECUSTOMIZE = '''\
# Test-only: register malvalid's toy feature schema + in-memory toy corpus in every interpreter.
try:
    import malvalid.testing  # noqa: F401
except Exception as e:  # pragma: no cover
    import sys
    print(f"sitecustomize: could not import malvalid.testing: {e}", file=sys.stderr)
'''


def wait_for(fn: Callable[[], Any], timeout: float = 30.0, interval: float = 0.05, what: str = "condition") -> Any:
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        last = fn()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout:g}s waiting for {what} (last value: {last!r})")


def csrf_for(settings: WebSettings) -> str:
    return settings.csrf_token


def captured(app: Any, template: str) -> dict[str, Any]:
    """The context of the most recent render of ``template``."""
    for name, ctx in reversed(app.state.renderer.captured):
        if name == template:
            return ctx
    raise AssertionError(f"{template} was not rendered; rendered: {[n for n, _ in app.state.renderer.captured]}")


def make_app(settings: WebSettings) -> Any:
    app = create_app(settings, warm_plugins=False)
    app.state.renderer.captured = []
    return app


def authed_client(app: Any) -> TestClient:
    c = TestClient(app, base_url=BASE)
    s = app.state.settings
    c.cookies.set(s.cookie_name, s.sessions.new_session())  # what a ?token= login stores
    return c


@pytest.fixture
def fake_malvalid(tmp_path: Path) -> Path:
    p = tmp_path / "fake_malvalid.py"
    p.write_text(FAKE_MALVALID, encoding="utf-8")
    return p


@pytest.fixture
def fake_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rec = tmp_path / "fake_record.jsonl"
    monkeypatch.setenv("FAKE_RECORD", str(rec))
    return rec


def read_record(rec: Path) -> list[dict[str, Any]]:
    if not rec.exists():
        return []
    return [json.loads(line) for line in rec.read_text().splitlines() if line.strip()]


@pytest.fixture
def settings(tmp_path: Path, fake_malvalid: Path) -> WebSettings:
    # Started with --allow-path-mode --allow-no-sandbox: the suites built on this fixture exercise both
    # (the off-by-default behaviour is tested in test_web_feature_flags.py).
    return WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=PORT, cancel_grace_s=1.0,
                       command_prefix=(sys.executable, str(fake_malvalid)), validate_timeout_s=30,
                       allow_path_mode=True, allow_no_sandbox=True)


@pytest.fixture
def app(settings: WebSettings) -> Any:
    return make_app(settings)


@pytest.fixture
def client(app: Any):
    with authed_client(app) as c:
        yield c


@pytest.fixture
def anon(app: Any):
    with TestClient(app, base_url=BASE) as c:
        yield c


@pytest.fixture
def toy_python_path(tmp_path: Path) -> Path:
    d = tmp_path / "toy_site"
    d.mkdir()
    (d / "sitecustomize.py").write_text(TOY_SITECUSTOMIZE, encoding="utf-8")
    return d


def fake_adapter(tmp_path: Path, mode: str = "ok", name: str = "my_adapter.py") -> Path:
    d = tmp_path / f"adapter_{mode}_{time.monotonic_ns()}"
    d.mkdir()
    p = d / name
    p.write_text(f"# FAKE: {mode}\nclass Detector:\n    pass\n", encoding="utf-8")
    return p


def cli_run_dir(runs: Path, run_id: str, report_fixture: str | None = None, **files: Any) -> Path:
    """A run dir as ``malvalid run --out`` leaves it (report from a fixture and/or given files)."""
    d = runs / run_id
    d.mkdir(parents=True, exist_ok=True)
    if report_fixture:
        shutil.copy(FIXTURES / report_fixture, d / "report.json")
    for name, content in files.items():
        p = d / name.replace("__", ".")
        if isinstance(content, (dict, list)):
            p.write_text(json.dumps(content))
        elif isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_bytes(str(content).encode("utf-8"))  # byte for byte: no CRLF translation on Windows
    return d


def post_run(client: TestClient, settings: WebSettings, data: dict[str, Any],
             files: list[tuple[str, tuple[str, bytes]]] | None = None, *, csrf: bool = True,
             url: str = "/runs", **kw: Any) -> Any:
    """Multipart POST of the new-run form (csrf as a form field, like the HTML page does)."""
    fields = {k: v for k, v in data.items() if v is not None}
    if csrf:
        fields = {"csrf": settings.csrf_token, **fields}
    multipart = list(files or [])
    if not multipart:  # force multipart even without files
        multipart = [("model_files", ("", b""))]
    return client.post(url, data=fields, files=multipart, follow_redirects=False, **kw)


def job_json(settings: WebSettings, run_id: str) -> dict[str, Any]:
    return json.loads((settings.runs_dir / run_id / "job.json").read_text())


def run_id_from(resp: Any) -> str:
    assert resp.status_code == 303, (resp.status_code, resp.text[:2000])
    loc = resp.headers["location"]
    assert loc.startswith("/runs/"), loc
    return loc.rsplit("/", 1)[-1]


def env_without_pythonpath() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def test_fake_malvalid_is_valid_python(fake_malvalid: Path) -> None:
    """Self-check of the support module (keeps this file a valid pytest target on its own)."""
    compile(fake_malvalid.read_text(), str(fake_malvalid), "exec")
    compile(TOY_SITECUSTOMIZE, "sitecustomize.py", "exec")
