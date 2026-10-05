"""Templates, static assets and display filters of the local web UI (docs/WEB_CONTRACT.md §5-§7).

Every template is rendered with a Jinja environment configured as the contract says (autoescape on,
default Undefined, FILTERS registered) against the example contexts in tests/fixtures/web_contexts.py.
"""

from __future__ import annotations

import html.parser
import re
import time
from pathlib import Path

import jinja2
import pytest

from malvalid.web import filters as F
from malvalid.web.filters import FILTERS
from tests.fixtures import web_contexts as W

WEB = Path(F.__file__).resolve().parent
TEMPLATES = WEB / "templates"
STATIC = WEB / "static"
REPORT_CSS = WEB.parent / "report" / "static" / "report.css"
PAGE_TEMPLATES = sorted(p.name for p in TEMPLATES.glob("*.html.j2") if not p.name.startswith("_") and p.name != "base.html.j2")
NAV = {"dashboard": "/", "new": "/runs/new", "compare": "/compare", "corpora": "/corpora", "modules": "/modules",
       "system": "/system"}


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def make_env(**kw) -> jinja2.Environment:
    opts = {"loader": jinja2.FileSystemLoader(str(TEMPLATES)), "autoescape": True}
    opts.update(kw)
    env = jinja2.Environment(**opts)
    env.filters.update(FILTERS)
    return env


@pytest.fixture(scope="module")
def env() -> jinja2.Environment:
    return make_env()


def render(env: jinja2.Environment, template: str, ctx: dict) -> str:
    return env.get_template(template).render(**ctx)


class Page(html.parser.HTMLParser):
    """A flat DOM summary: start tags with attributes, text, and inline script/style content."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[str] = []
        self.script_text: list[str] = []
        self.style_tags = 0
        self._in_script = False
        self._in_title = False
        self.title = ""
        self.feed(text)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self._in_script = True
        if tag == "style":
            self.style_tags += 1
        if tag == "title":
            self._in_title = True

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_script:
            self.script_text.append(data)
        elif self._in_title:
            self.title += data
        else:
            self.text.append(data)

    def find(self, tag: str | None = None, **attrs) -> list[dict[str, str | None]]:
        out = []
        for t, a in self.tags:
            if tag and t != tag:
                continue
            if all((a.get(k.replace("_", "-")) == v) if v is not True else (k.replace("_", "-") in a)
                   for k, v in attrs.items()):
                out.append(a)
        return out

    @property
    def all_text(self) -> str:
        return " ".join(self.text)


def check_page(page_html: str, ctx: dict) -> Page:
    """Invariants every rendered page must satisfy."""
    p = Page(page_html)
    # CSP: no inline script, no inline style, no event-handler attributes, no javascript: URLs
    for tag, attrs in p.tags:
        for name, value in attrs.items():
            assert not name.startswith("on"), f"inline handler {name}= on <{tag}>"
            assert name != "style", f"inline style on <{tag}>"
            if name in ("href", "src", "action", "formaction") and value:
                assert not value.strip().lower().startswith(("javascript:", "data:text", "vbscript:")), value
                assert not value.startswith("//"), f"protocol-relative URL {value}"
                assert not re.match(r"^[a-z]+:", value, re.I), f"external URL {value} (the UI works offline)"
        if tag == "script":
            assert attrs.get("src", "").startswith("/static/"), "scripts must be static files"
        if tag == "link" and attrs.get("rel") == "stylesheet":
            assert attrs.get("href", "").startswith("/static/")
    assert "".join(p.script_text).strip() == "", "inline <script> content"
    assert p.style_tags == 0, "<style> element"
    # structure
    assert p.find("html", lang="en")
    assert p.title.strip() and "MalValid" in p.title
    assert len(p.find("main", id="main")) == 1
    assert len(p.find("h1")) == 1, "exactly one <h1>"
    csrf = p.find("meta", name="csrf-token")
    assert len(csrf) == 1 and csrf[0].get("content") == (ctx.get("csrf_token") or "")
    assert p.find("script", src=True) and p.find("link", rel="stylesheet")
    # ids unique; labels, aria-describedby and aria-labelledby point at existing ids
    ids = [a["id"] for _, a in p.tags if a.get("id")]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate ids {dupes}"
    for _, a in p.tags:
        for key in ("for", "aria-describedby", "aria-labelledby"):
            for ref in (a.get(key) or "").split():
                assert ref in ids, f"{key}={ref} points nowhere"
    # every POST form carries the CSRF field
    for _, a in p.tags:
        pass
    for form in re.findall(r"<form\b[^>]*>.*?</form>", page_html, re.S):
        if re.search(r'method="post"', form, re.I):
            assert f'name="csrf" value="{ctx.get("csrf_token") or ""}"' in form, "POST form without the csrf field"
    # nothing leaks as Python / Jinja artefacts
    text = p.all_text
    assert not re.search(r"\bNone\b", text), "None rendered as text"
    assert "UndefinedError" not in text and "jinja2.exceptions" not in text and "<jinja2" not in text.lower()
    assert not re.search(r"\bUndefined\b", text.replace("Undefined behaviour", ""))
    assert "{{" not in page_html and "{%" not in page_html
    return p


def _hostile_in(obj) -> bool:
    if isinstance(obj, str):
        return any(h in obj for h in W.HOSTILE_STRINGS)
    if isinstance(obj, dict):
        return any(_hostile_in(k) or _hostile_in(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return any(_hostile_in(x) for x in obj)
    return False


# --------------------------------------------------------------------------------------------------
# every fixture renders cleanly
# --------------------------------------------------------------------------------------------------


def test_every_template_has_fixtures():
    covered = {tpl for tpl, _ in W.CONTEXTS.values()}
    assert set(PAGE_TEMPLATES) == covered == set(W.TEMPLATE_KEYS)
    assert "base.html.j2" not in covered and (TEMPLATES / "base.html.j2").is_file()


@pytest.mark.parametrize("case", sorted(W.CONTEXTS))
def test_fixture_renders(env, case):
    template, ctx = W.get(case)
    out = render(env, template, ctx)
    check_page(out, ctx)
    if _hostile_in(ctx):
        for h in W.HOSTILE_STRINGS:
            assert h not in out, f"unescaped {h!r}"
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out or "&lt;img src=x onerror=alert(1)&gt;" in out


@pytest.mark.parametrize("template", PAGE_TEMPLATES)
@pytest.mark.parametrize("variant", ["all_none", "empty"])
def test_templates_tolerate_missing_and_none(env, template, variant):
    ctx = W.all_none(template) if variant == "all_none" else {}
    check_page(render(env, template, ctx), ctx)


@pytest.mark.parametrize("template", PAGE_TEMPLATES)
def test_templates_tolerate_wrong_types(env, template):
    ctx = {k: "x" for k in W.COMMON_KEYS + W.TEMPLATE_KEYS[template]}
    ctx.update({"csrf_token": "t", "status": 500})
    for k in W.TEMPLATE_KEYS[template]:
        if k == "errors":
            ctx[k] = [None, 3, "x"]
        elif k in ("runs", "active", "rows", "config_diff", "modules", "corpora"):
            ctx[k] = [None, 3, "x", {"run_id": None}]
        elif k in ("counts", "report", "progress", "job", "files", "form", "defaults", "run", "backends",
                   "environment", "settings"):
            ctx[k] = {"x": object()} if k != "run" else {"run_id": "r1", "score": "high", "verdict": 7}
    check_page(render(env, template, ctx), ctx)


# --------------------------------------------------------------------------------------------------
# autoescape (fail closed) and escaping
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [{"autoescape": False}, {"autoescape": jinja2.select_autoescape()}])
def test_render_refuses_without_autoescape(kw):
    env = make_env(**kw)
    template, ctx = W.get("dashboard")
    with pytest.raises(RuntimeError, match="autoescap"):
        render(env, template, ctx)


def test_select_autoescape_with_j2_extension_works():
    env = make_env(autoescape=jinja2.select_autoescape(["html", "j2"]))
    template, ctx = W.get("run_hostile")
    out = render(env, template, ctx)
    assert W.HOSTILE not in out


def test_hostile_values_escaped_in_attributes(env):
    template, ctx = W.get("new_run_errors")
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    # the sticky value stays inside its attribute: no attribute injection
    ap = p.find("input", id="adapter_path")[0]
    assert ap["value"] == W.HOSTILE_ATTR
    assert not any(k == "onmouseover" for _, a in p.tags for k in a)
    title = p.find("input", id="title")[0]
    assert title["value"] == W.HOSTILE


def test_hostile_run_ids_cannot_break_urls(env):
    template, ctx = W.get("run_sparse")
    ctx["run"] = W.summary('x"><script>alert(1)</script>', status="finished")
    ctx["files"] = W.FILES_ALL
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    for a in p.find("a", href=True) + p.find("form", action=True):
        url = a.get("href") or a.get("action")
        assert "<" not in url and '"' not in url and " " not in url


# --------------------------------------------------------------------------------------------------
# layout: nav, CSRF, assets
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("nav", list(NAV) + [None, "nonsense"])
def test_nav_active_highlighting(env, nav):
    template, ctx = W.get("corpora")
    ctx["nav_active"] = nav
    p = check_page(render(env, template, ctx), ctx)
    current = p.find("a", aria_current="page")
    if nav in NAV:
        assert [a["href"] for a in current] == [NAV[nav]]
    else:
        assert current == []
    nav_links = [a["href"] for t, a in p.tags if t == "a" and a.get("href") in NAV.values()]
    for href in NAV.values():
        assert href in nav_links


def test_csrf_token_exposed_to_js_and_forms(env):
    template, ctx = W.get("run_running")
    ctx["csrf_token"] = "tok-123"
    p = check_page(render(env, template, ctx), ctx)
    assert p.find("meta", name="csrf-token")[0]["content"] == "tok-123"
    assert p.find("input", name="csrf", value="tok-123")


def test_assets_are_versioned_static_files(env):
    template, ctx = W.get("dashboard")
    p = check_page(render(env, template, ctx), ctx)
    assert p.find("link", rel="stylesheet")[0]["href"] == "/static/app.css?v=0.1.0"
    assert p.find("script")[0]["src"] == "/static/app.js?v=0.1.0"
    assert "defer" not in p.find("script")[0] and "async" not in p.find("script")[0]  # theme before first paint


# --------------------------------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------------------------------


def test_dashboard(env):
    template, ctx = W.get("dashboard")
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    assert p.find("section", data_dashboard_active=True), "auto-refresh hook while runs are active"
    form = p.find("form", data_compare_form=True)[0]
    assert form["action"] == "/compare" and form["method"] == "get"
    boxes = {a["value"]: a for a in p.find("input", name="ids")}
    assert set(boxes) == {r["run_id"] for r in ctx["runs"] if r["run_id"]}
    for r in ctx["runs"]:
        has_verdict = r["verdict"] in ("ready", "conditional", "not_ready", "blocked")
        assert ("disabled" in boxes[r["run_id"]]) is (not has_verdict), r["run_id"]
    assert "92.3" in out and "81.4" in out and "49.0" in out
    assert p.find("a", href=f"/runs/{W.RID['ready']}")
    text = p.all_text
    for word in ("Ready", "Conditional", "Not ready", "Blocked", "Running", "Queued", "Failed", "Cancelled",
                 "Interrupted"):
        assert word in text
    assert "CLI" in text  # the command-line run is marked


def test_dashboard_idle_has_no_auto_refresh(env):
    template, ctx = W.get("dashboard_idle")
    p = check_page(render(env, template, ctx), ctx)
    assert not p.find("section", data_dashboard_active=True)


def test_dashboard_empty_state_tells_what_to_do(env):
    template, ctx = W.get("dashboard_empty")
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    assert "No runs yet" in p.all_text
    assert p.find("a", href="/runs/new")
    assert "malvalid validate-adapter --adapter my_adapter.py" in p.all_text
    assert f"malvalid run --adapter my_adapter.py --out {W.RUNS_DIR}/my-first-run" in p.all_text
    assert not p.find("table")


def test_new_run_form_matches_the_contract(env):
    template, ctx = W.get("new_run")
    p = check_page(render(env, template, ctx), ctx)
    form = p.find("form", data_run_form=True)[0]
    assert form["method"] == "post" and form["action"] == "/runs" and form["enctype"] == "multipart/form-data"
    names = {a.get("name") for t, a in p.tags if t in ("input", "select", "textarea") and a.get("name")}
    assert names >= {"csrf", "mode", "adapter_file", "model_files", "manifest_file", "config_file", "adapter_path",
                     "config_path", "class_name", "only", "skip", "corpus", "corpus_dir", "seed", "fail_on",
                     "allow_pickle", "no_sandbox", "confirm_no_sandbox", "title"}
    assert "multiple" in p.find("input", name="model_files")[0]
    accept = {a["name"]: set(a["accept"].split(",")) for a in p.find("input", type="file")}
    assert accept["adapter_file"] == {".py"}
    assert accept["model_files"] == {".txt", ".json", ".ubj", ".model", ".onnx", ".pkl", ".pickle", ".joblib", ".jbl"}
    assert accept["manifest_file"] == {".txt", ".csv", ".tsv"}
    assert accept["config_file"] == {".yaml", ".yml"}
    assert [a["value"] for a in p.find("option") if a.get("value") in ("blocked", "not_ready", "conditional")] == [
        "blocked", "not_ready", "conditional"]
    assert p.find("option", value="blocked", selected=True)
    # "Upload your model" (just the model file) is the default; the own-adapter upload stays available
    assert p.find("input", name="mode", value="model", checked=True)
    assert p.find("input", name="mode", value="upload") and p.find("input", name="mode", value="path")
    # empty by default: a pre-filled seed would override the runtime.seed of an uploaded gate policy
    seed = p.find("input", id="seed")[0]
    assert seed["value"] == "" and seed["placeholder"] == "Policy default (0)" and seed["data-default"] == "0"
    assert p.find("form", data_max_mb="4096")
    assert p.find("button", data_validate=True)
    corpus_select = re.search(r'<select id="corpus".*?</select>', render(env, template, ctx), re.S).group(0)
    corpora = re.findall(r'<option value="((?:ember|synthetic)[^"]*)"', corpus_select)
    assert corpora == [c["name"] for c in W.CORPORA]
    text = p.all_text
    # researcher-facing explanations of the risky / non-obvious options
    assert "can run arbitrary code" in text                          # allow_pickle risk
    assert "no network or file-system isolation" in text             # no_sandbox danger
    assert "never changes the verdict or the score" in text          # fail_on meaning
    assert "M4 membership inference" in text and "M2 temporal drift" in text  # what manifest / cutoff unlock


def test_new_run_errors_are_sticky(env):
    template, ctx = W.get("new_run_errors")
    p = check_page(render(env, template, ctx), ctx)
    alert = p.find("div", role="alert")
    assert alert and alert[0].get("id") == "form-errors"
    assert "does not exist" in p.all_text and "pickle-based" in p.all_text
    assert p.find("input", name="mode", value="path", checked=True)
    assert not p.find("input", name="mode", value="upload", checked=True)
    assert p.find("option", value="ember_v3_2024", selected=True)
    assert p.find("option", value="not_ready", selected=True)
    assert p.find("input", id="seed")[0]["value"] == "7"
    assert p.find("input", id="only")[0]["value"] == "M1,M2"
    assert "checked" in p.find("input", id="allow_pickle")[0]
    assert "checked" in p.find("input", id="no_sandbox")[0]
    assert "checked" not in p.find("input", id="confirm_no_sandbox")[0]
    assert "choose the files again" not in p.all_text  # path mode: nothing was uploaded


def test_run_page_finished(env):
    template, ctx = W.get("run_ready")
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    rid = W.RID["ready"]
    page = p.find("div", data_run_page=True)[0]
    assert page["data-run-id"] == rid and page["data-status"] == "finished"
    assert "Ready" in p.all_text and "92.3" in p.all_text and "/100" in p.all_text
    assert p.title.startswith("Ready 92.3")
    frame = p.find("iframe")[0]
    assert frame["src"] == f"/runs/{rid}/report.html"
    assert frame["sandbox"] == "allow-scripts"  # never allow-same-origin
    for f in ("report.html", "report.json", "run.log", "console.log"):
        assert p.find("a", href=f"/runs/{rid}/{f}")
    delete = p.find("form", action=f"/api/runs/{rid}/delete")
    assert delete and delete[0]["data-confirm"] and delete[0]["method"] == "post"
    assert not p.find("form", action=f"/api/runs/{rid}/cancel")
    assert not p.find("meta", http_equiv="refresh")
    # the scorecard lists every module of the report
    for m in ctx["report"]["modules"]:
        assert m["code"] in p.all_text
    assert "Printed from MalValid" in p.all_text


def test_run_page_running(env):
    template, ctx = W.get("run_running")
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    rid = W.RID["running"]
    assert p.find("div", data_run_page=True)[0]["data-status"] == "running"
    cancel = p.find("form", action=f"/api/runs/{rid}/cancel")[0]
    assert cancel["method"] == "post" and cancel["data-confirm"] and cancel["data-next"] == f"/runs/{rid}"
    assert not p.find("form", action=f"/api/runs/{rid}/delete")
    assert not p.find("iframe")
    assert "<noscript><meta http-equiv=\"refresh\" content=\"5\"></noscript>" in out
    assert p.find("a", href=f"/runs/{rid}")  # manual refresh link without JS
    assert p.find("p", aria_live="polite")
    rows = p.find("li", data_module=True)
    assert [r["data-module"] for r in rows] == [m["id"] for m in ctx["progress"]["modules"]]
    assert p.find("li", **{"class": "status-running"})
    assert "M2 Temporal drift" in p.all_text and "3 of 6 tests finished" in p.all_text
    steps = p.find("li", data_step=True)
    assert [s["class"] for s in steps] == ["done", "done", "done", "current", "todo", "todo"]
    assert p.find("li", data_step="modules", aria_current="step")


def test_run_page_queued_without_progress(env):
    template, ctx = W.get("run_queued")
    p = check_page(render(env, template, ctx), ctx)
    assert p.find("div", data_run_page=True)[0]["data-status"] == "queued"
    assert "Waiting for a free worker" in p.all_text
    assert "The test list appears once the adapter has been inspected." in p.all_text


def test_run_page_failed_without_report(env):
    template, ctx = W.get("run_failed")
    p = check_page(render(env, template, ctx), ctx)
    rid = W.RID["failed"]
    assert "Run failed" in p.all_text and "exit code 2" in p.all_text
    assert "does not declare operating_threshold" in p.all_text
    assert p.find("a", href=f"/runs/{rid}/console.log") and p.find("a", href=f"/runs/{rid}/run.log")
    assert p.find("a", href="/runs/new")
    assert not p.find("iframe")


def test_run_page_failed_with_report_flags_exit_2(env):
    template, ctx = W.get("run_blocked_exit2")
    p = check_page(render(env, template, ctx), ctx)
    assert "The run exited with code 2" in p.all_text
    assert "Blocked" in p.all_text and "capped at 49" in p.all_text
    assert "errored" in p.all_text.lower()


@pytest.mark.parametrize("case,word", [("run_cancelled", "Cancelled"), ("run_interrupted", "Interrupted"),
                                       ("run_aborted", "Blocked"), ("run_not_ready", "Not ready"),
                                       ("run_conditional_cli", "Conditional")])
def test_run_page_states(env, case, word):
    template, ctx = W.get(case)
    p = check_page(render(env, template, ctx), ctx)
    assert p.find("h2", id="status-h" if case in ("run_cancelled", "run_interrupted") else "verdict-h")
    assert word in p.all_text


def test_run_page_cli_run(env):
    template, ctx = W.get("run_conditional_cli")
    p = check_page(render(env, template, ctx), ctx)
    assert "from the command line" in p.all_text
    assert ctx["report"]["run"]["command"] in p.all_text


def test_compare(env):
    template, ctx = W.get("compare")
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    heads = p.find("th", scope="col")
    assert len(heads) == 1 + len(ctx["runs"])
    assert "Highest score" in p.all_text
    assert len([a for a in p.find("th") if "col-best" in (a.get("class") or "")]) == 1
    for d in ctx["config_diff"]:
        assert d["key"].split(".")[-1] in p.all_text
    assert "not set" in p.all_text  # a None config value
    assert "93.0" in p.all_text  # a 0..1 cell score is shown on the 0..100 scale
    assert "not in this run" in p.all_text


@pytest.mark.parametrize("case", ["compare_empty", "compare_one", "compare_none"])
def test_compare_needs_two_runs(env, case):
    template, ctx = W.get(case)
    p = check_page(render(env, template, ctx), ctx)
    assert not p.find("table")
    assert p.find("a", href="/") and "tick 2 to 6 runs" in p.all_text.lower()


def test_corpora(env):
    template, ctx = W.get("corpora")
    p = check_page(render(env, template, ctx), ctx)
    text = p.all_text
    assert "Installed" in text and "Not installed" in text and "Synthetic" in text
    assert "malvalid corpus build ember_v3_2024" in text  # the unavailable corpus says how to get it
    assert W.CORPORA[0]["content_hash"] in text


def test_corpora_empty(env):
    template, ctx = W.get("corpora_empty")
    p = check_page(render(env, template, ctx), ctx)
    assert "malvalid corpus list" in p.all_text


def test_modules(env):
    template, ctx = W.get("modules")
    p = check_page(render(env, template, ctx), ctx)
    text = p.all_text
    for m in W.MODULES:
        assert m["title"] in text
    assert "Training manifest" in text and "Training cutoff" in text
    assert "off by default" in text and "Screening" in text
    assert p.find("pre", id="policy-yaml") and p.find("button", data_copy="policy-yaml")
    assert "malvalid init-config gate.yaml" in text


def test_system_never_shows_secrets(env):
    template, ctx = W.get("system")
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    for secret in W.SECRET_VALUES:
        assert secret not in out
    assert "bwrap" in p.all_text and "used by default" in p.all_text
    assert "not installed" in p.all_text  # a missing optional library


def test_system_without_isolation_warns(env):
    template, ctx = W.get("system_no_isolation")
    p = check_page(render(env, template, ctx), ctx)
    assert "No isolating backend works here" in p.all_text
    assert "Remote access is on" in p.all_text


@pytest.mark.parametrize("case,code,phrase", [("error_404", "404", "Run not found"),
                                               ("error_401", "401", "?token="),
                                               ("error_403", "403", "Reload the page"),
                                               ("error_413", "413", "Use files on this machine"),
                                               ("error_500", "500", "malvalid serve")])
def test_error_pages(env, case, code, phrase):
    template, ctx = W.get(case)
    p = check_page(render(env, template, ctx), ctx)
    assert code in p.all_text and phrase in p.all_text


# --------------------------------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------------------------------


def test_contract_filters_present():
    for name in ("fmt_score", "fmt_pct", "fmt_time", "fmt_duration", "verdict_class", "status_class",
                 "verdict_label"):
        assert callable(FILTERS[name])


@pytest.mark.parametrize("value,expected", [(72.4, "72.4"), (72, "72.0"), (0, "0.0"), (99.96, "100.0"),
                                            (None, "—"), ("x", "—"), (float("nan"), "—"), (True, "—"),
                                            ("55.5", "55.5"), (-0.01, "0.0")])
def test_fmt_score(value, expected):
    assert F.fmt_score(value) == expected


@pytest.mark.parametrize("value,expected", [(0.0123, "1.23%"), (1, "100%"), (0.5, "50%"), (0.83333, "83.33%"),
                                            (0, "0%"), (None, "—"), ("bad", "—"), (float("inf"), "—")])
def test_fmt_pct(value, expected):
    assert F.fmt_pct(value) == expected


def test_fmt_pct_digits():
    assert F.fmt_pct(0.8333, 0) == "83%" and F.fmt_pct(0.004, 1) == "0.4%"


@pytest.mark.parametrize("value,expected", [(192, "3m 12s"), (0.83, "0.8s"), (45, "45s"), (59.6, "1m 0s"),
                                            (3600, "1h 0m"), (3725, "1h 2m"), (90061, "1d 1h"), (None, "—"),
                                            (-1, "—"), ("12", "12s")])
def test_fmt_duration(value, expected):
    assert F.fmt_duration(value) == expected


@pytest.fixture()
def tz(monkeypatch):
    def set_tz(name):
        monkeypatch.setenv("TZ", name)
        time.tzset()
    yield set_tz
    monkeypatch.undo()
    time.tzset()


def test_fmt_time_local(tz):
    tz("UTC")
    assert F.fmt_time("2026-09-29T14:03:11Z") == "2026-09-29 14:03"
    assert F.fmt_time("2026-09-29T14:03:11+00:00") == "2026-09-29 14:03"
    assert F.fmt_time("2026-09-29T14:03:11") == "2026-09-29 14:03"  # naive values are UTC
    tz("Asia/Kolkata")
    assert F.fmt_time("2026-09-29T14:03:11Z") == "2026-09-29 19:33"
    assert F.fmt_time(None) == "—" and F.fmt_time("") == "—"
    assert F.fmt_time("yesterday-ish") == "yesterday-ish"
    assert F.fmt_iso("2026-09-29T14:03:11+02:00") == "2026-09-29T12:03:11Z"
    assert F.fmt_iso("nope") == "" and F.fmt_iso(None) == ""


@pytest.mark.parametrize("verdict,cls,label", [("ready", "tone-good", "Ready"),
                                               ("conditional", "tone-warning", "Conditional"),
                                               ("not_ready", "tone-serious", "Not ready"),
                                               ("NOT READY", "tone-serious", "Not ready"),
                                               ("blocked", "tone-critical", "Blocked"),
                                               (None, "tone-neutral", "No verdict"),
                                               ("<b>", "tone-neutral", "No verdict")])
def test_verdict_vocabulary(verdict, cls, label):
    assert F.verdict_class(verdict) == cls and F.verdict_label(verdict) == label
    assert F.verdict_icon(verdict).startswith("i-")


def test_verdict_enum_values():
    from malvalid.verdict import Verdict

    assert F.verdict_label(Verdict.NOT_READY) == "Not ready"
    assert F.verdict_class(Verdict.READY) == "tone-good"


@pytest.mark.parametrize("status,cls", [("pass", "tone-good"), ("warn", "tone-warning"), ("fail", "tone-critical"),
                                        ("error", "tone-critical"), ("skipped", "tone-neutral"),
                                        ("pending", "tone-neutral"), ("running", "tone-info"),
                                        ("queued", "tone-neutral"), ("finished", "tone-neutral"),
                                        ("failed", "tone-critical"), ("cancelled", "tone-neutral"),
                                        ("interrupted", "tone-warning"), (None, "tone-neutral"),
                                        ("weird", "tone-neutral")])
def test_status_class(status, cls):
    assert F.status_class(status) == cls


def test_status_label_and_key():
    assert F.status_label("skip") == "Skipped" and F.status_key("skip") == "skipped"
    assert F.status_label("some_new_state") == "Some new state" and F.status_key("some_new_state") == "unknown"
    assert F.status_label(None) == "Unknown"


def test_every_icon_exists_in_the_sprite():
    sprite = set(re.findall(r'<symbol id="([^"]+)"', (TEMPLATES / "_ui.html.j2").read_text()))
    names = {F.status_icon(s) for s in list(F._STATUSES) + ["?"]}
    names |= {F.verdict_icon(v) for v in ["ready", "conditional", "not_ready", "blocked", None]}
    js = (STATIC / "app.js").read_text()
    names |= set(re.findall(r'"(i-[a-z-]+)"', js))
    templates = "".join(p.read_text() for p in TEMPLATES.glob("*.j2"))
    names |= set(re.findall(r'"(i-[a-z-]+)"', templates))
    assert names <= sprite, names - sprite


def test_view_filters_never_raise_on_garbage():
    garbage = [None, jinja2.Undefined(), 3, "x", [], {}, {"verdict": "x", "modules": "y"},
               {"modules": [None, {"checks": "nope", "status": 5}]}, W.REPORT_MALFORMED]
    for g in garbage:
        rv = F.report_view(g)
        assert set(rv) >= {"present", "verdict", "modules", "gate", "model", "corpus", "run", "warnings"}
        pv = F.progress_view(g)
        assert isinstance(pv["modules"], list) and isinstance(pv["steps"], list)
        jv = F.job_view(g)
        assert isinstance(jv["options"], list)
        for name, fn in FILTERS.items():
            if name == "autoescape_guard":
                continue
            fn(g)  # must not raise


def test_report_view_primary_check_and_scores():
    rv = F.report_view(W.REPORT_BLOCKED)
    by = {m["code"]: m for m in rv["modules"]}
    assert rv["verdict"]["key"] == "blocked" and rv["verdict"]["capped"] and rv["verdict"]["raw"] == 53.1
    assert by["M1"]["primary"]["metric"] == "fpr" and by["M1"]["primary"]["passed"] is False
    assert by["M1"]["score"] == 36.0 and by["M1"]["weight"] == 3.0
    assert by["M0"]["unscored"] and by["M0"]["score"] is None
    assert by["M6"]["status"] == "error" and by["M6"]["error_head"]
    assert rv["gate"]["exit_code"] == 2 and rv["gate"]["hard"][1]["tone"] == "critical"
    assert rv["verdict"]["blockers"] and all(r not in rv["verdict"]["blockers"] for r in rv["verdict"]["reasons"])


def test_progress_view():
    pv = F.progress_view(W.PROGRESS_RUNNING)
    assert pv["n_total"] == 6 and pv["n_done"] == 3 and pv["done_step"] == 50
    assert pv["module"]["code"] == "M2" and pv["stage_label"].startswith("Running")
    assert [s["state"] for s in pv["steps"]] == ["done", "done", "done", "current", "todo", "todo"]
    assert [s["state"] for s in F.stage_steps("done")] == ["done"] * 6
    assert [s["state"] for s in F.stage_steps("failed")] == ["todo"] * 6


def test_job_view_command_and_options():
    j = W.job(W.RID["ready"], "finished", options={"skip": "M6", "allow_pickle": "on", "csrf": "zzz",
                                                   "api_token": "zzz"})
    jv = F.job_view(j)
    assert jv["command"].startswith("malvalid run --adapter ")
    labels = {o["label"]: o["value"] for o in jv["options"]}
    assert labels["Skipped modules"] == "M6" and labels["Allow pickle"] == "yes"
    assert "zzz" not in str(jv["options"])


def test_small_helpers():
    assert F.cli_command([W.PYTHON, "-m", "malvalid", "run", "--adapter", "a b.py"]) == \
        "malvalid run --adapter 'a b.py'"
    assert F.shell_join(["a", "b c"]) == "a 'b c'" and F.shell_quote("x y") == "'x y'"
    assert F.run_name({"adapter": "/x/y/adapter.py"}) == "adapter.py"
    assert F.run_name({"run_id": "r1"}) == "r1" and F.run_name(None) == "Untitled run"
    assert F.model_line({"model_kind": "lightgbm", "feature_version": "ember_v2", "operating_threshold": 0.8}) == \
        "lightgbm · ember_v2 · threshold 0.8"
    assert [c["status"] for c in F.module_counts({"n_pass": 3, "n_fail": 1, "n_skipped": 0})] == ["pass", "fail"]
    assert F.pct_step(72.6) == 73 and F.pct_step(140) == 100 and F.pct_step(None) == 0
    assert F.frac_step(0.834) == 83 and F.axis_pct(0.93) == 93.0 and F.axis_pct(61.7) == 61.7
    assert F.is_verdict("blocked") and not F.is_verdict("finished")
    assert F.truthy("on") and F.truthy(True) and not F.truthy("") and not F.truthy(None)
    assert F.sandbox_choice(W.BACKENDS) == "bwrap" and F.sandbox_choice(W.BACKENDS_NONE) == "subprocess"
    assert [k for k, _ in F.public_items(W.SETTINGS)] == sorted(k for k in W.SETTINGS if k not in ("token", "csrf_secret"))
    assert str(F.break_dots("a.b<c>")) == "a.<wbr>b&lt;c&gt;"
    assert F.requirement_label("training_hashes") == "Training manifest"


# --------------------------------------------------------------------------------------------------
# static assets
# --------------------------------------------------------------------------------------------------


def _tokens(css: str, selector_start: str) -> dict[str, str]:
    block = css[css.index(selector_start):]
    block = block[: block.index("}")]
    return dict(re.findall(r"(--[a-z0-9-]+):\s*([^;]+);", block))


def test_css_shares_the_report_color_tokens():
    app, rep = (STATIC / "app.css").read_text(), REPORT_CSS.read_text()
    a, r = _tokens(app, ":root {"), _tokens(rep, ":root {")
    shared = [k for k in r if k.startswith(("--good", "--warning", "--serious", "--critical", "--neutral", "--s"))
              or k in ("--page", "--surface", "--ink", "--link", "--focus")]
    assert shared and all(a.get(k) == r[k] for k in shared), {k: (a.get(k), r[k]) for k in shared if a.get(k) != r[k]}
    ad, rd = _tokens(app, ':root[data-theme="dark"]'), _tokens(rep, ':root[data-theme="dark"]')
    assert all(ad.get(k) == v for k, v in rd.items() if k in shared or k.endswith("-ink"))
    assert "prefers-color-scheme: dark" in app and "@media print" in app and "prefers-reduced-motion" in app


def test_css_is_self_contained_and_has_the_utilities():
    css = (STATIC / "app.css").read_text()
    assert "@import" not in css and not re.search(r"url\((?!#)", css), "no external assets"
    for i in range(101):
        assert f".w-{i}{{width:{i}%}}" in css and f".l-{i}{{left:{i}%}}" in css


def test_rendered_classes_have_css(env):
    css = (STATIC / "app.css").read_text()
    for case in ("dashboard", "run_ready", "run_running", "compare", "new_run_errors"):
        template, ctx = W.get(case)
        out = render(env, template, ctx)
        for cls in set(re.findall(r'class="([^"]+)"', out)):
            for c in cls.split():
                if c.startswith(("tone-", "w-", "l-", "band-", "chip", "badge", "btn", "verdict")):
                    assert re.search(r"\." + re.escape(c) + r"\b", css), f".{c} is not styled"


def test_js_is_safe_and_matches_the_templates(env):
    js = (STATIC / "app.js").read_text()
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function",
                   "setAttribute(\"style\"", "https://"):
        assert banned not in js, banned
    assert "http://" not in js.replace("http://www.w3.org/2000/svg", ""), "external URL in app.js"
    assert "X-CSRF-Token" in js and 'meta[name="csrf-token"]' in js
    # every data-* hook app.js looks for exists in some rendered page
    pages = "".join(render(env, *W.get(c)) for c in
                    ("dashboard", "new_run", "new_run_errors", "run_running", "run_ready", "compare", "modules"))
    hooks = set(re.findall(r'\[(data-[a-z-]+)', js))
    missing = {h for h in hooks if h not in pages}
    assert not missing, missing
    # JS keeps the same status vocabulary as the filters
    for key, (tone, label, icon) in F._STATUSES.items():
        assert f'{key}: ["{tone}", "{label}", "{icon}"]' in js
    for key, short, _long in F.STAGES:
        assert f'["{key}", "{short}"' in js


def test_favicon_is_a_local_svg():
    svg = (STATIC / "favicon.svg").read_text()
    assert svg.lstrip().startswith("<svg") and "script" not in svg.lower() and "href=\"http" not in svg
