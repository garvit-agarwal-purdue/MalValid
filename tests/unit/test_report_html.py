"""Tests for the self-contained HTML report (``malvalid.report.html``)."""

from __future__ import annotations

import base64
import copy
import hashlib
import html as htmllib
import importlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from malvalid.report.html import render_html, write_html

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
VARIANT_FILES = {
    "blocked": "sample_report.json",
    "ready": "sample_report_ready.json",
    "conditional": "sample_report_conditional.json",
    "not_ready": "sample_report_not_ready.json",
    "aborted": "sample_report_aborted.json",
}
VERDICT_WORD = {"blocked": "Blocked", "ready": "Ready", "conditional": "Conditional", "not_ready": "Not ready",
                "aborted": "Blocked"}
MAX_BYTES = 1_500_000


def _load(variant: str) -> dict[str, Any]:
    return json.loads((FIXTURES / VARIANT_FILES[variant]).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rendered() -> dict[str, str]:
    return {v: render_html(_load(v)) for v in VARIANT_FILES}


def _generator():
    spec = importlib.util.spec_from_file_location("make_sample_report", FIXTURES / "make_sample_report.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _visible_text(html: str) -> str:
    """Rough text content (tags stripped, entities decoded) for 'is it shown' assertions."""
    body = re.sub(r"<script\b.*?</script>", " ", html, flags=re.S)
    body = re.sub(r"<style\b.*?</style>", " ", body, flags=re.S)
    return htmllib.unescape(re.sub(r"<[^>]+>", " ", body))


def _chart_json(html: str) -> dict[str, Any]:
    m = re.search(r'<script type="application/json" id="mg-chart-data">(.*?)</script>', html, re.S)
    assert m, "chart data block missing"
    return json.loads(m.group(1))


# --------------------------------------------------------------------------------------------------
# Fixtures themselves
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANT_FILES))
def test_fixture_is_valid_report_json(variant: str) -> None:
    rep = _load(variant)
    assert rep["schema_version"] == "malvalid-report/1"
    for key in ("tool", "run", "verdict", "gate", "model", "corpus", "schema", "training_manifest", "config",
                "environment", "sandbox", "modules", "artifacts", "warnings", "disclaimers"):
        assert key in rep, key
    json.dumps(rep, allow_nan=False)  # the runner's contract
    codes = [m["code"] for m in rep["modules"]]
    assert codes[0] == "M0" and "M3" not in codes  # M0 first; M3 is out of scope
    expected = {"aborted": "blocked"}.get(variant, variant)
    assert rep["verdict"]["verdict"] == expected


def _drift(built: Any, committed: Any, path: str = "") -> list[str]:
    """Differences between two JSON values; floats compare to ~1e-9 relative (see below)."""
    if isinstance(built, bool) or isinstance(committed, bool) or not (
        isinstance(built, (int, float)) and isinstance(committed, (int, float))
    ):
        if type(built) is not type(committed):
            return [f"{path or '/'}: {built!r:.80} != {committed!r:.80}"]
    if isinstance(built, dict):
        out = [f"{path}/{k}: only in one of them" for k in sorted(set(built) ^ set(committed))]
        for k in sorted(set(built) & set(committed)):
            out += _drift(built[k], committed[k], f"{path}/{k}")
        return out
    if isinstance(built, list):
        if len(built) != len(committed):
            return [f"{path}: length {len(built)} != {len(committed)}"]
        return [d for i, (a, b) in enumerate(zip(built, committed)) for d in _drift(a, b, f"{path}[{i}]")]
    if isinstance(built, float) or isinstance(committed, float):
        ok = math.isclose(built, committed, rel_tol=1e-9, abs_tol=1e-12)
    else:
        ok = built == committed
    return [] if ok else [f"{path or '/'}: {built!r:.80} != {committed!r:.80}"]


def test_fixtures_match_generator() -> None:
    """The committed fixtures are what make_sample_report.py produces (no drift).

    Floats are compared to ~1e-9 relative, not bit for bit: numpy picks SIMD kernels by CPU (AVX-512 or
    not), so the generator's unrounded chart series differ in the last ulp between machines, e.g. on
    GitHub runners of different CPU types. Real drift (a changed value, key or string) still fails.
    """
    gen = _generator()
    for variant, name in VARIANT_FILES.items():
        built = json.loads(json.dumps(gen.build_report(variant), allow_nan=False))
        drift = _drift(built, _load(variant))
        assert not drift, f"{name} is stale: re-run tests/fixtures/make_sample_report.py\n" + "\n".join(drift[:20])


def test_fixture_drift_check_ignores_only_float_rounding() -> None:
    assert _drift({"a": [0.8829675244250249, 1, "x"]}, {"a": [0.882967524425025, 1, "x"]}) == []
    assert _drift({"a": [0.8829675244250249]}, {"a": [0.8829]})
    assert _drift({"a": 1}, {"a": True}) and _drift({"a": "x"}, {"a": "y"}) and _drift({"a": 1}, {"b": 1})
    assert _drift([1, 2], [1, 2, 3]) and _drift({"a": None}, {"a": 0.0})


def test_kitchen_sink_covers_every_chart_and_state() -> None:
    rep = _load("blocked")
    arts = rep["artifacts"]
    for key in ("performance.roc", "performance.pr", "performance.calibration", "performance.score_hist",
                "drift.decay", "drift.window_sizes", "membership_inf.attack_roc", "membership_inf.member_score_hist",
                "extraction.fidelity_vs_queries", "explanation.top_features", "explanation.group_importance"):
        assert arts[key]["type"] == "chart", key
    assert any(a["type"] == "table" for a in arts.values())
    assert arts["performance.roc"]["xscale"] == "log" and arts["extraction.fidelity_vs_queries"]["xscale"] == "log"
    assert arts["performance.calibration"]["note"].startswith("diagonal")
    assert arts["drift.window_sizes"]["kind"] == "bar" and arts["explanation.top_features"]["kind"] == "bar"
    statuses = {m["status"] for m in rep["modules"]}
    assert {"pass", "fail", "warn", "skipped", "error"} <= statuses
    assert any(m.get("screening") for m in rep["modules"])
    assert any(m["status"] == "error" and "Traceback" in (m.get("error") or "") for m in rep["modules"])


# --------------------------------------------------------------------------------------------------
# Rendering every variant
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANT_FILES))
def test_renders_every_variant(rendered: dict[str, str], variant: str) -> None:
    html = rendered[variant]
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    rep = _load(variant)
    text = _visible_text(html)
    assert VERDICT_WORD[variant] in text
    score = rep["verdict"].get("score")
    if score is not None:
        assert f"{score:.1f}/100" in text or f"{score:.1f}" in text
    # verdict banner first, scorecard directly beneath the gate strip, modules after
    i_verdict = html.index('class="verdict ')
    i_gates = html.index('class="gates"')
    i_scorecard = html.index('id="scorecard"')
    i_appendix = html.index('id="appendix"')
    assert i_verdict < i_gates < i_scorecard < i_appendix
    for m in rep["modules"]:
        assert m["title"] in text or htmllib.escape(m["title"]) in html
    assert len(html.encode("utf-8")) < MAX_BYTES


def test_module_sections_in_code_order(rendered: dict[str, str]) -> None:
    html = rendered["blocked"]
    rep = _load("blocked")
    positions = [html.index(f'id="m-{m["module_id"]}"') for m in rep["modules"]]
    assert positions == sorted(positions)


@pytest.mark.parametrize("variant", sorted(VARIANT_FILES))
def test_no_external_references(rendered: dict[str, str], variant: str) -> None:
    html = rendered[variant]
    ext = r"""(?:https?:)?//"""
    assert not re.search(r"""\b(?:src|href|action|poster|formaction|data|xlink:href)\s*=\s*["']?\s*""" + ext, html, re.I)
    assert not re.search(r"@import", html, re.I)
    assert not re.search(r"""url\(\s*["']?\s*""" + ext, html, re.I)
    assert not re.search(r"<link\b", html, re.I)
    assert not re.search(r"<script\b[^>]*\bsrc\s*=", html, re.I)
    assert not re.search(r"<(?:iframe|object|embed|img)\b", html, re.I)


def test_csp_pins_the_inline_script(rendered: dict[str, str]) -> None:
    html = rendered["blocked"]
    csp = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)"', html)
    assert csp and "default-src 'none'" in csp.group(1)
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    digest = base64.b64encode(hashlib.sha256(scripts[0].encode("utf-8")).digest()).decode()
    assert f"'sha256-{digest}'" in csp.group(1)


def test_light_dark_print_and_mobile_styles(rendered: dict[str, str]) -> None:
    html = rendered["ready"]
    assert "prefers-color-scheme: dark" in html
    assert '[data-theme="dark"]' in html
    assert "@media print" in html
    assert "@media (max-width" in html
    assert '<meta name="viewport"' in html


# --------------------------------------------------------------------------------------------------
# Escaping
# --------------------------------------------------------------------------------------------------


def test_report_strings_are_escaped(rendered: dict[str, str]) -> None:
    html = rendered["blocked"]
    gen = _generator()
    for payload in (gen.XSS_SCRIPT, gen.XSS_IMG, gen.XSS_BREAKOUT):
        assert payload not in html
    assert "<script>alert" not in html
    assert "onerror=alert" not in html.replace("onerror=alert(1)&gt;", "")  # only the escaped form may appear
    assert "&lt;script&gt;alert(" in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    # exactly two script elements: the chart data (inert JSON) and our pinned script
    assert len(re.findall(r"<script\b", html, re.I)) == 2


def test_chart_data_block_cannot_break_out() -> None:
    rep = _load("ready")
    evil = "</script><script>alert('x')</script> &  "
    rep["artifacts"]["performance.roc"]["series"][0]["label"] = evil
    rep["artifacts"]["performance.roc"]["title"] = "<b>roc</b>"
    html = render_html(rep)
    assert "</script><script>alert" not in html
    data = _chart_json(html)
    labels = [s["label"] for spec in data.values() for s in spec.get("series", [])]
    assert evil in labels  # round-trips intact as data
    assert "&lt;b&gt;roc&lt;/b&gt;" in html and "<b>roc</b>" not in html


# --------------------------------------------------------------------------------------------------
# Content: skipped, errored, charts, tables, appendix
# --------------------------------------------------------------------------------------------------


def test_skipped_and_errored_modules_are_visible(rendered: dict[str, str]) -> None:
    html = rendered["blocked"]
    rep = _load("blocked")
    text = _visible_text(html)
    skipped = [m for m in rep["modules"] if m["status"] == "skipped"]
    errored = [m for m in rep["modules"] if m["status"] == "error"]
    assert skipped and errored
    for m in skipped:
        assert m["skip_reason"] in text
        section = html[html.index(f'id="m-{m["module_id"]}"'):]
        assert "Not run:" in _visible_text(section[: section.index("</section>")])
    for m in errored:
        section = html[html.index(f'id="m-{m["module_id"]}"'):]
        section = section[: section.index("</section>")]
        assert '<details class="traceback">' in section  # collapsed by default
        tb = re.search(r'<pre class="code">(.*?)</pre>', section, re.S)
        assert tb and htmllib.unescape(tb.group(1)).strip() == m["error"].strip()
        assert "This test crashed" in _visible_text(section)


def test_every_artifact_key_is_rendered(rendered: dict[str, str]) -> None:
    for variant, html in rendered.items():
        rep = _load(variant)
        charts = {k for k, a in rep["artifacts"].items() if a.get("type") == "chart"}
        tables = {k for k, a in rep["artifacts"].items() if a.get("type") == "table"}
        figure_keys = set(re.findall(r'<figure class="chart" id="c\d+" data-chart="c\d+" data-key="([^"]+)"', html))
        table_keys = set(re.findall(r'<div class="artifact-table" data-key="([^"]+)"', html))
        assert charts == figure_keys, variant
        assert tables <= table_keys, variant
        specs = _chart_json(html)
        assert {s["key"] for s in specs.values()} == charts


def test_chart_features_render(rendered: dict[str, str]) -> None:
    html = rendered["blocked"]
    specs = {s["key"]: s for s in _chart_json(html).values()}
    assert specs["performance.roc"]["x"]["log"] is True
    assert specs["extraction.fidelity_vs_queries"]["x"]["log"] is True
    assert specs["performance.score_hist"]["kind"] == "step"
    assert specs["explanation.top_features"]["kind"] == "bar"

    def figure(key: str) -> str:
        start = html.index(f'data-key="{key}"')
        return html[start: html.index("</figure>", start)]

    assert 'class="diag"' in figure("performance.calibration")
    assert 'class="diag"' not in figure("performance.pr")
    assert 'class="ref"' in figure("performance.roc") and "max_fpr" in figure("performance.roc")
    assert 'class="legend"' in figure("drift.decay")  # >= 2 series -> legend
    assert 'class="legend"' not in figure("performance.pr")  # one series -> named in the caption
    assert "10⁻⁵" in figure("performance.roc") or "0.001" in figure("performance.roc")  # log ticks
    assert 'class="hover"' in figure("performance.roc")  # hover layer present
    assert "horiz" in figure("explanation.top_features")  # long category labels -> horizontal bars


def test_scorecard_and_screening(rendered: dict[str, str]) -> None:
    html = rendered["blocked"]
    card = html[html.index('id="scorecard"'): html.index("</section>", html.index('id="scorecard"'))]
    rep = _load("blocked")
    for m in rep["modules"]:
        assert f"#m-{m['module_id']}" in card
    assert card.count('class="badge screening"') == sum(1 for m in rep["modules"] if m.get("screening"))
    # every status chip carries text, not just color
    for chip in re.findall(r'<span class="chip tone-[a-z]+">(.*?)</span></span>', card, re.S):
        assert _visible_text(chip).strip()


def test_verdict_banner_details(rendered: dict[str, str]) -> None:
    blocked = rendered["blocked"]
    assert "blocked cap" in blocked and "capped at" in _visible_text(blocked)
    assert "Blocking issues" in blocked
    ready = rendered["ready"]
    assert "blocked cap" not in ready and "Blocking issues" not in ready
    aborted = _visible_text(rendered["aborted"])
    assert "Run aborted by M0" in aborted
    assert "pickle" in aborted
    cond = _visible_text(rendered["conditional"])
    assert "Why this verdict" in cond


def test_appendix_reproducibility(rendered: dict[str, str]) -> None:
    html = rendered["blocked"]
    rep = _load("blocked")
    appendix = html[html.index('id="appendix"'):]
    text = _visible_text(appendix)
    assert rep["corpus"]["content_hash"] in text
    assert rep["model"]["artifacts"][0]["sha256"] in text
    assert rep["environment"]["libraries"]["lightgbm"] in text
    assert "not installed" in text  # missing optional libraries are called out
    for m in rep["modules"]:
        if m.get("seed") is not None:
            assert f"{m['seed']:,}" in text or str(m["seed"]) in text
    assert "Gate configuration" in text and "<details" in appendix
    for d in rep["disclaimers"]:
        assert d in text


# --------------------------------------------------------------------------------------------------
# Robustness: missing / malformed keys never crash the renderer
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"schema_version": "malvalid-report/1"},
        {"verdict": "ready", "modules": {"not": "a list"}, "artifacts": ["nope"], "gate": None},
        {"modules": [None, 3, {"module_id": "x"}, {"status": "weird", "checks": [None, {"op": "<="}]}]},
        {"verdict": {"verdict": "ready", "score": "NaN", "coverage": None, "bands": {"ready_min": "x"}}},
        {"schema_version": "malvalid-report/99", "tool": {"name": "malvalid"}},
    ],
)
def test_tolerates_missing_and_malformed_keys(report: dict[str, Any]) -> None:
    html = render_html(report)
    assert "</html>" in html
    assert len(html) > 1000


def test_drops_each_top_level_key() -> None:
    base = _load("blocked")
    for key in list(base):
        rep = copy.deepcopy(base)
        del rep[key]
        assert "</html>" in render_html(rep), key


def test_malformed_charts_degrade_gracefully() -> None:
    rep = _load("ready")
    arts = rep["artifacts"]
    arts["x.weird"] = {"type": "chart", "module": "x", "name": "weird", "title": "Weird", "kind": "sparkle",
                       "series": [{"label": "a", "x": [1, "b", None, 4], "y": [1, 2]}], "xscale": "log",
                       "yscale": "log", "xlim": [0, -1], "ylim": ["a", "b"], "reference_lines": [{"value": "x"}]}
    arts["x.logzeros"] = {"type": "chart", "module": "x", "name": "logzeros", "title": "Zeros", "kind": "line",
                          "xscale": "log", "series": [{"label": "z", "x": [0, 0, -1], "y": [1, 2, 3]}]}
    arts["x.emptybar"] = {"type": "chart", "module": "x", "name": "emptybar", "title": "Empty", "kind": "bar",
                          "series": [{"label": "b", "x": ["a", "b"], "y": [None, None]}]}
    arts["x.noseries"] = {"type": "chart", "module": "x", "name": "noseries", "title": "None"}
    arts["x.scatter"] = {"type": "chart", "module": "x", "name": "scatter", "title": "Scatter", "kind": "scatter",
                         "series": [{"label": "p", "x": [1, 2, 3], "y": [3, 1, 2]}, {"label": "q", "x": [2], "y": [2]}]}
    arts["x.negbar"] = {"type": "chart", "module": "x", "name": "negbar", "title": "Neg", "kind": "bar",
                        "series": [{"label": "d", "x": ["up", "down"], "y": [0.3, -0.2]}],
                        "reference_lines": [{"axis": "y", "value": 0.1, "label": "limit"}]}
    arts["x.table"] = {"type": "table", "module": "x", "name": "table", "title": "Ragged",
                       "columns": ["a"], "rows": [[1, 2, 3], "scalar", [None]], "truncated": True}
    html = render_html(rep)
    for key in ("x.weird", "x.logzeros", "x.emptybar", "x.noseries", "x.scatter", "x.negbar"):
        assert f'data-key="{key}"' in html, key
    assert 'data-key="x.table"' in html
    assert "Other artifacts" in html  # artifacts no module claims are still shown
    assert "unknown chart kind" in html
    json.loads(json.dumps(_chart_json(html), allow_nan=False))


def test_index_note_maps_x_to_periods() -> None:
    """M2 decay charts put window numbers on x and a ``"1=2024-01; 2=2024-02"`` note; the renderer
    labels ticks and hover readouts with the periods instead of printing the raw mapping."""
    from malvalid.report.html import _parse_xnames

    assert _parse_xnames("1=2024-01; 2=2024-02; 10=2024-10") == {"1": "2024-01", "2": "2024-02", "10": "2024-10"}
    for prose in ("", "diagonal", "dropped: 2025-01", "An FPR of 0 is drawn at 1e-05 on the log axis.",
                  "1=2024-01", "a=b; c=d", "Classification rule: score >= threshold; see docs"):
        assert _parse_xnames(prose) is None, prose

    html = render_html(_load("blocked"))
    spec = next(v for v in _chart_json(html).values() if v["key"] == "drift.decay")
    assert spec["xnames"]["1"] == "2024-01" and spec["xnames"]["12"] == "2024-12"
    fig = re.search(r'data-key="drift\.decay".*?</figure>', html, re.S)
    assert fig, "decay figure missing"
    text = _visible_text(fig.group(0))
    assert "2024-01" in text and "1 = 2024-01" in text
    assert "1=2024-01; 2=2024-02" not in text  # the raw mapping is replaced by a readable caption
    # A prose note is left alone.
    assert "An FPR of 0 is drawn at 1e-05 on the log axis." in _visible_text(html)
    assert "xnames" not in next(v for v in _chart_json(html).values() if v["key"] == "performance.roc")


def test_bar_orientation() -> None:
    from malvalid.report.html import _ChartBuilder

    months = [f"2023-{m:02d}" for m in range(1, 13)] + [f"2024-{m:02d}" for m in range(1, 13)]
    b = _ChartBuilder()
    short = b.render("drift.window_sizes", {"type": "chart", "title": "Windows", "kind": "bar",
                                            "series": [{"label": "malicious", "x": months, "y": list(range(24))},
                                                       {"label": "benign", "x": months, "y": list(range(24))}],
                                            "reference_lines": [{"axis": "y", "value": 5, "label": "min_window_samples"}]})
    assert "horiz" not in str(short["svg"]) and b.specs[short["id"]]["orient"] == "v"
    long_ = b.render("explanation.top_features", {"type": "chart", "title": "Top", "kind": "bar",
                                                  "series": [{"label": "share", "x": ["strings.printables", "general.size"],
                                                              "y": [0.3, 0.1]}],
                                                  "reference_lines": [{"axis": "y", "value": 0.2, "label": "limit"}]})
    assert b.specs[long_["id"]]["orient"] == "h" and ">limit</text>" in str(long_["svg"])


def test_section_failure_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    import malvalid.report.html as mod

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(mod, "_appendix_view", boom)
    html = render_html(_load("ready"))
    assert "This section could not be rendered (RuntimeError: synthetic failure)." in html
    assert "Per-axis scorecard" in html


def test_runner_style_gate_block() -> None:
    rep = _load("ready")
    rep["gate"].update(aborted=True, abort_reason="model.pkl is a pickle and --allow-pickle was not given")
    text = _visible_text(render_html(rep))
    assert "Run aborted by M0" in text and "model.pkl is a pickle" in text
    rep = _load("ready")
    rep["gate"]["exit_meaning"] = "custom meaning from the runner"
    rep["gate"]["load_error"] = "ImportError: no module named foo"
    text = _visible_text(render_html(rep))
    assert "custom meaning from the runner" in text
    assert "The model could not be loaded" in text


# --------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------


def test_render_is_deterministic() -> None:
    rep = _load("blocked")
    assert render_html(rep) == render_html(copy.deepcopy(rep))


def test_render_does_not_mutate_input() -> None:
    rep = _load("blocked")
    before = json.dumps(rep, sort_keys=True)
    render_html(rep)
    assert json.dumps(rep, sort_keys=True) == before


def test_write_html(tmp_path: Path) -> None:
    out = write_html(_load("conditional"), tmp_path / "nested" / "dir" / "report.html")
    assert isinstance(out, Path) and out.is_file()
    content = out.read_text(encoding="utf-8")
    assert content == render_html(_load("conditional"))
    assert not list(out.parent.glob("*.tmp"))
    assert out.stat().st_size < MAX_BYTES


# --------------------------------------------------------------------------------------------------
# Real module output (toy data): whatever the finished modules emit must render cleanly
# --------------------------------------------------------------------------------------------------

_REAL_MODULES = [
    ("performance", "PerformanceModule", {}),
    ("drift", "DriftModule", {"min_window_samples": 20}),
    ("membership", "MembershipInferenceModule", {}),
    ("extraction", "ExtractionModule", {"query_budgets": [100, 1000], "fidelity_budget": 1000, "n_eval": 500}),
]


def test_renders_real_module_output() -> None:
    """Run the modules that exist in this tree on the toy corpus and render their real artifacts.

    A module that is missing or crashes here is some other component's problem, so it is skipped;
    this test only fails when the renderer mishandles what a module really emits."""
    import datetime as dt
    import logging

    from malvalid import testing as T

    corpus = T.make_toy_corpus(seed=0)
    model = T.InProcessModel.from_lightgbm(T.train_toy_lgbm(corpus), threshold=0.5, feature_version="toy_v1")
    hashes = T.training_hashes_for(corpus)
    results: list[dict[str, Any]] = []
    arts: dict[str, Any] = {}
    for modname, clsname, params in _REAL_MODULES:
        try:
            mod = importlib.import_module(f"malvalid.modules.{modname}")
            cls = getattr(mod, clsname)
            ctx = T.make_context(cls, model=model, corpus=corpus, training_hashes=hashes,
                                 training_cutoff=dt.date(2017, 12, 31), params=params)
            res = cls().run(ctx)
        except Exception as exc:  # pragma: no cover - depends on other components
            logging.getLogger("malvalid.tests").warning("skipping %s: %s", modname, exc)
            continue
        results.append(res.to_dict())
        arts.update(ctx.artifacts.to_dict())
    if not arts:
        pytest.skip("no finished modules produced artifacts")
    rep = json.loads(json.dumps({"schema_version": "malvalid-report/1", "modules": results, "artifacts": arts},
                                allow_nan=False))
    html = render_html(rep)
    assert "could not be drawn" not in html
    specs = {v["key"]: v for v in _chart_json(html).values()}
    for key, art in arts.items():
        assert f'data-key="{key}"' in html, key
        if art.get("type") == "chart":
            assert key in specs, key
    if "drift.decay" in specs:
        assert specs["drift.decay"].get("xnames"), "M2's window-number note should map x to periods"
