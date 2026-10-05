"""Self-contained HTML production-readiness report.

``render_html(report)`` turns a ``malvalid-report/1`` dict (see the build contract §4.4) into one
HTML file a researcher can open offline, attach to a paper, or print:

* the verdict banner and 0–100 score come first, the per-axis scorecard directly beneath;
* one section per module (finding, checks, metrics, charts, tables, notes, skip reason or
  traceback), then an appendix with everything needed to reproduce the run.

Design constraints: no network access of any kind (inline CSS/JS, charts drawn here as SVG — no
third-party chart library), every report string is HTML-escaped (jinja2 autoescape, plus
``markupsafe.escape`` for the SVG built in Python), a Content-Security-Policy pins the one inline
script by hash, and every section tolerates missing or malformed keys (a section that cannot be
rendered is replaced by a visible notice instead of failing the whole report).

Charts are rendered server-side as static SVG so they display without JavaScript and print as
vectors; the inline script only adds hover readouts, keyboard focus, a data-table view and CSV
export. Chart specs follow :meth:`malvalid.context.ArtifactStore.add_chart`.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jinja2
from markupsafe import Markup, escape

log = logging.getLogger("malvalid.report")

__all__ = ["render_html", "write_html"]

_PKG_DIR = Path(__file__).resolve().parent
_TEMPLATE_DIR = _PKG_DIR / "templates"
_STATIC_DIR = _PKG_DIR / "static"
_TEMPLATE = "report.html.j2"

REPORT_SCHEMA = "malvalid-report/1"
#: Schema versions rendered without a warning (includes the pre-rename ``malguard-report/1``).
try:  # pragma: no cover - import guard only
    from malvalid import SUPPORTED_REPORT_SCHEMA_VERSIONS as _SUPPORTED_SCHEMAS
except Exception:  # pragma: no cover
    _SUPPORTED_SCHEMAS = (REPORT_SCHEMA, "malguard-report/1")
#: Display name for the tool identifiers a report may carry.
_TOOL_DISPLAY = {"malvalid": "MalValid", "malguard": "MalValid"}

# Modules that are preconditions rather than scored axes (mirrors malvalid.verdict).
try:  # pragma: no cover - import guard only
    from malvalid.verdict import UNSCORED_MODULES as _UNSCORED
except Exception:  # pragma: no cover
    _UNSCORED = frozenset({"file_safety", "dummy"})

try:  # pragma: no cover - import guard only
    from malvalid.scoring import PASS_LINE as _PASS_LINE
except Exception:  # pragma: no cover
    _PASS_LINE = 0.75


# ==================================================================================================
# Safe accessors
# ==================================================================================================


def _d(x: Any) -> dict[str, Any]:
    """``x`` if it is a mapping, else an empty dict."""
    return dict(x) if isinstance(x, Mapping) else {}


def _l(x: Any) -> list[Any]:
    """``x`` if it is a list/tuple, else an empty list."""
    return list(x) if isinstance(x, (list, tuple)) else []


def _s(x: Any, default: str = "") -> str:
    """A display string (None -> ``default``)."""
    if x is None:
        return default
    if isinstance(x, str):
        return x
    return str(x)


def _num(x: Any) -> float | None:
    """A finite float, or None (bools and non-numbers are None)."""
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, str):
        try:
            v = float(x)
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-") or "x"


# ==================================================================================================
# Formatting (also exposed to the template as filters)
# ==================================================================================================


def fmt_num(v: Any, sig: int = 4) -> str:
    """Compact, precise display of a metric value."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "—"
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e15 or a < 1e-3:
            return f"{v:.2e}"
        if v.is_integer() and a >= 1:
            return f"{v:,.0f}"
        if a >= 1000:
            return f"{v:,.1f}"
        return f"{v:.{sig}g}"
    return _s(v)


def fmt_value(v: Any, *, max_items: int = 12, max_chars: int = 240) -> str:
    """Display any JSON value on one line (lists/dicts compacted and truncated)."""
    if v is None or isinstance(v, (bool, int, float)):
        return fmt_num(v)
    if isinstance(v, str):
        return v if len(v) <= max_chars * 4 else v[: max_chars * 4] + "…"
    if isinstance(v, (list, tuple)):
        items = [fmt_value(x, max_items=4, max_chars=60) for x in list(v)[:max_items]]
        more = f", … (+{len(v) - max_items})" if len(v) > max_items else ""
        s = "[" + ", ".join(items) + more + "]"
        return s if len(s) <= max_chars else s[: max_chars - 1] + "…"
    if isinstance(v, Mapping):
        try:
            s = json.dumps(v, ensure_ascii=False, default=str, sort_keys=False)
        except (TypeError, ValueError):
            s = str(v)
        return s if len(s) <= max_chars else s[: max_chars - 1] + "…"
    return _s(v)


def fmt_pct(v: Any, digits: int = 0) -> str:
    x = _num(v)
    return "—" if x is None else f"{x:.{digits}%}"


def fmt_bytes(v: Any) -> str:
    x = _num(v)
    if x is None:
        return "—"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024 or unit == "TiB":
            return f"{x:,.0f} {unit}" if unit == "B" else f"{x:,.1f} {unit}"
        x /= 1024
    return "—"  # pragma: no cover


def fmt_duration(v: Any) -> str:
    x = _num(v)
    if x is None:
        return "—"
    if x < 1:
        return f"{x * 1000:.0f} ms"
    if x < 60:
        return f"{x:.1f} s"
    if x < 3600:
        return f"{int(x // 60)} min {int(round(x % 60)):02d} s"
    return f"{int(x // 3600)} h {int(round((x % 3600) / 60)):02d} min"


def fmt_time(v: Any) -> str:
    s = _s(v)
    if not s:
        return "—"
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    if t.tzinfo is not None:
        t = t.astimezone(dt.timezone.utc)
        return t.strftime("%Y-%m-%d %H:%M:%S UTC")
    return t.strftime("%Y-%m-%d %H:%M:%S")


_OPS = {"<=": "≤", ">=": "≥", "<": "<", ">": ">", "==": "="}


def fmt_op(op: Any) -> str:
    return _OPS.get(_s(op), _s(op))


# ==================================================================================================
# Status vocabulary: every status is color + icon + text (never color alone)
# ==================================================================================================


@dataclass(frozen=True)
class Tone:
    tone: str  # good | warning | serious | critical | neutral
    label: str
    icon: str


_STATUS: dict[str, Tone] = {
    "pass": Tone("good", "Pass", "i-pass"),
    "warn": Tone("warning", "Warn", "i-warn"),
    "fail": Tone("critical", "Fail", "i-fail"),
    "skipped": Tone("neutral", "Skipped", "i-skip"),
    "error": Tone("critical", "Error", "i-error"),
}
_UNKNOWN = Tone("neutral", "Unknown", "i-unknown")

_OUTCOME: dict[str, Tone] = {
    "passed": Tone("good", "Gate passed", "i-pass"),
    "failed": Tone("critical", "Gate failed", "i-fail"),
    "not_evaluated": Tone("neutral", "Not evaluated", "i-unknown"),
}

_VERDICT: dict[str, dict[str, str]] = {
    "ready": {
        "tone": "good", "icon": "i-pass", "word": "Ready",
        "meaning": "No gate failed and enough of the battery was evaluated. The evidence supports "
        "promoting this model, subject to your usual review.",
    },
    "conditional": {
        "tone": "warning", "icon": "i-warn", "word": "Conditional",
        "meaning": "Promotion needs a human decision. Review the reasons below before putting this "
        "model into production.",
    },
    "not_ready": {
        "tone": "serious", "icon": "i-fail", "word": "Not ready",
        "meaning": "The evidence does not support putting this model into production yet.",
    },
    "blocked": {
        "tone": "critical", "icon": "i-block", "word": "Blocked",
        "meaning": "Not promotable as submitted: a hard gate failed, a test errored, or the run was "
        "aborted. Fix the blocking issues and re-run.",
    },
}
_VERDICT_UNKNOWN = {
    "tone": "neutral", "icon": "i-unknown", "word": "No verdict",
    "meaning": "This report does not contain a production-readiness verdict.",
}


def _status(s: Any) -> Tone:
    return _STATUS.get(_s(s).lower(), _UNKNOWN)


def _outcome(s: Any) -> Tone:
    return _OUTCOME.get(_s(s).lower(), Tone("neutral", _s(s, "—") or "—", "i-unknown"))


# Short researcher-facing descriptions of the built-in modules.
_BLURBS: dict[str, str] = {
    "file_safety": "Scans your model artifacts (modelscan plus a static pickle-opcode scan) before anything "
    "is deserialized. A precondition for the run; it does not contribute to the score.",
    "performance": "Detection rate and false-positive rate at your declared operating threshold on the "
    "canonical corpus (training members excluded), with ROC / PR curves and calibration.",
    "drift": "Time-aware evaluation on monthly windows after your training cutoff (TESSERACT-style "
    "constraints); gates on the sample-weighted area under time of F1.",
    "membership_inf": "Black-box membership inference: how well scores alone reveal whether a sample was "
    "in your training set (an OPSEC / data-privacy concern).",
    "backdoor_screen": "Screens for backdoor-like behaviour: operator-supplied trigger hypotheses and a "
    "static scan of the tree ensemble's root-to-leaf paths. Screening only.",
    "extraction": "How cheaply a surrogate can clone the model from query access alone. Only relevant "
    "if the model will be exposed as a queryable service.",
    "explanation": "Global attribution (tree gain and SHAP): does the model lean on features a file's "
    "author can change without changing what the program does?",
    "dummy": "Pipeline smoke test: scores a few random vectors to prove the adapter works end to end.",
}

_SCREENING_TEXT = (
    "Screening test: it can find evidence of a problem, but absence of findings is not proof of "
    "absence. Certifying a model clean would need training-time access."
)

# Actionable next steps keyed by fragments of the runner's skip reasons (REQUIREMENT_HINTS).
_SKIP_HELP: tuple[tuple[str, str], ...] = (
    ("run aborted by M0", "Resolve the M0 file-safety finding (for example, re-export the model in a "
     "non-pickle format) and re-run."),
    ("training_hashes_path", "Declare training_hashes_path on your adapter: a file with one training-sample "
     "sha256 per line (or a CSV with a sha256 column)."),
    ("training manifest", "The training manifest has too few members in the canonical corpus for a "
     "meaningful attack; this is expected for models trained on private data."),
    ("training_cutoff", "Declare training_cutoff on your adapter (YYYY-MM-DD, YYYY-MM or YYYY): the date "
     "of the newest training sample."),
    ("canonical corpus", "Install or build the canonical corpus for your model's feature_version "
     "(malvalid corpus list / malvalid corpus build) or pass --corpus-dir."),
    ("tree structure", "Expose native_model (LightGBM / XGBoost / sklearn) or a tree_ensemble() method on "
     "your adapter to enable tree-based analysis."),
    ("sample_dir", "Set sample_dir in gate.yaml to a directory you populate in your own isolated "
     "environment."),
    ("featurize", "Add featurize(raw: bytes) to your adapter, or install the featurize extra."),
    ("model not loaded", "The model could not be loaded; see the run warnings and M0."),
)


def _skip_help(reason: str) -> str:
    low = reason.lower()
    for frag, text in _SKIP_HELP:
        if frag.lower() in low:
            return text
    return ""


# ==================================================================================================
# SVG charts
# ==================================================================================================

_W = 560.0  # SVG viewBox width (the figure scales to its column)
_PLOT_H = 236.0  # plot-area height of x/y charts
_CHAR_W = 6.3  # approximate glyph advance at the 11px tick font
_MAX_SERIES_COLORS = 8
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _e(s: Any) -> str:
    """HTML/SVG-escape any value into a plain str."""
    return str(escape(_s(s)))


def _f(v: float) -> str:
    """Coordinate formatting (one decimal, no '-0.0')."""
    s = f"{v:.1f}"
    return "0.0" if s == "-0.0" else s


def _r5(v: Any) -> float | None:
    x = _num(v)
    return None if x is None else float(f"{x:.5g}")


_XNAME_SEG = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*=\s*(\S.*?)\s*$")


def _xkey(v: float) -> str:
    """Key under which an x value is looked up in an index->label map (matches JS ``String(x)``)."""
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def _parse_xnames(note: str) -> dict[str, str] | None:
    """Parse a chart note of the form ``"1=2024-01; 2=2024-02; ..."`` (M2 decay charts use it to map
    window numbers to calendar periods). Returns ``None`` unless *every* segment has that shape and
    there are at least two of them, so ordinary prose notes are never misread."""
    segs = [x for x in note.split(";") if x.strip()]
    if len(segs) < 2:
        return None
    out: dict[str, str] = {}
    for seg in segs:
        m = _XNAME_SEG.match(seg)
        if not m or len(m.group(2)) > 40:
            return None
        out[_xkey(float(m.group(1)))] = m.group(2)
    return out


class _Axis:
    """Maps data values to SVG pixels (linear or log10)."""

    def __init__(self, lo: float, hi: float, log_: bool, p0: float, p1: float):
        self.lo, self.hi, self.log, self.p0, self.p1 = lo, hi, log_, p0, p1
        self._a = math.log10(lo) if log_ else lo
        self._b = math.log10(hi) if log_ else hi

    def ok(self, v: Any) -> bool:
        x = _num(v)
        return x is not None and (not self.log or x > 0)

    def __call__(self, v: float) -> float:
        t = math.log10(v) if self.log else v
        return self.p0 + (t - self._a) / (self._b - self._a) * (self.p1 - self.p0)

    def inside(self, v: float, tol: float = 1e-9) -> bool:
        if not self.ok(v):
            return False
        span = abs(self._b - self._a)
        t = math.log10(v) if self.log else v
        return self._a - tol * span <= t <= self._b + tol * span

    def spec(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "log": self.log, "p0": round(self.p0, 2), "p1": round(self.p1, 2)}


def _nice_step(raw: float) -> float:
    if not (raw > 0 and math.isfinite(raw)):
        return 1.0
    e = math.floor(math.log10(raw))
    f = raw / 10**e
    nf = 1.0 if f < 1.5 else 2.0 if f < 3.0 else 5.0 if f < 7.0 else 10.0
    return nf * 10**e


def _lin_ticks(lo: float, hi: float, n: int = 5) -> tuple[list[float], float]:
    step = _nice_step((hi - lo) / max(n, 1))
    start = math.ceil(lo / step - 1e-9)
    ticks = []
    k = start
    while k * step <= hi + step * 1e-9 and len(ticks) < 50:
        ticks.append(round(k * step, 12))
        k += 1
    return ticks, step


def _log_ticks(lo: float, hi: float) -> list[float]:
    a = math.floor(math.log10(lo) + 1e-9)
    b = math.ceil(math.log10(hi) - 1e-9)
    decades = [10.0**k for k in range(a, b + 1) if lo * (1 - 1e-9) <= 10.0**k <= hi * (1 + 1e-9)]
    if len(decades) >= 3:
        while len(decades) > 8:
            decades = decades[::2]
        return decades
    ticks = [m * 10.0**k for k in range(a - 1, b + 1) for m in (1, 2, 5)]
    ticks = [t for t in ticks if lo * (1 - 1e-9) <= t <= hi * (1 + 1e-9)]
    return ticks if len(ticks) >= 2 else [lo, hi]


def _compact(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:g}B"
    if a >= 1e6:
        return f"{v / 1e6:g}M"
    if a >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _fmt_log_tick(v: float) -> str:
    k = math.log10(v)
    kr = round(k)
    if abs(k - kr) < 1e-9 and (kr < -3 or kr > 5):
        return "10" + str(kr).translate(_SUPERSCRIPT)
    if v < 1e-3:
        m = v / 10.0**math.floor(k)
        return f"{m:g}×10" + str(math.floor(k)).translate(_SUPERSCRIPT)
    return _compact(v)


def _fmt_lin_tick(v: float, step: float) -> str:
    if abs(v) < step * 1e-9:
        return "0"
    if max(abs(v), step) >= 1e5:
        return _compact(v)
    dec = max(0, -math.floor(math.log10(step) + 1e-12))
    s = f"{v:,.{dec}f}"
    return "0" if s in ("-0", "-0.0") else s


def _ticks(ax: _Axis, n: int = 5) -> list[tuple[float, str]]:
    if ax.log:
        return [(t, _fmt_log_tick(t)) for t in _log_ticks(ax.lo, ax.hi)]
    ticks, step = _lin_ticks(ax.lo, ax.hi, n)
    return [(t, _fmt_lin_tick(t, step)) for t in ticks]


def _valid_lim(lim: Any, log_: bool) -> tuple[float, float] | None:
    vals = _l(lim)
    if len(vals) != 2:
        return None
    lo, hi = _num(vals[0]), _num(vals[1])
    if lo is None or hi is None or not lo < hi or (log_ and lo <= 0):
        return None
    return lo, hi


def _domain(values: list[float], lim: Any, log_: bool, *, include_zero: bool = False) -> tuple[float, float]:
    fixed = _valid_lim(lim, log_)
    if fixed:
        return fixed
    vals = [v for v in values if v is not None and math.isfinite(v) and (not log_ or v > 0)]
    if include_zero and not log_:
        vals.append(0.0)
    if not vals:
        return (1.0, 10.0) if log_ else (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    if log_:
        if lo == hi:
            lo, hi = lo / math.sqrt(10), hi * math.sqrt(10)
        a, b = math.log10(lo), math.log10(hi)
        pad = (b - a) * 0.04
        return 10 ** (a - pad), 10 ** (b + pad)
    if lo == hi:
        d = abs(lo) * 0.1 or 1.0
        lo, hi = lo - d, hi + d
    step = _nice_step((hi - lo) / 5)
    return math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step


def _bar_path(x: float, y: float, w: float, h: float, *, horizontal: bool, negative: bool) -> str:
    """A bar with a 4px rounded data-end and a square baseline end."""
    r = max(0.0, min(4.0, (h if horizontal else w) / 2, (w if horizontal else h)))
    if horizontal:  # x..x+w is the value extent, y..y+h the thickness
        if negative:  # data end on the left
            return (f"M{_f(x + w)},{_f(y)}H{_f(x + r)}Q{_f(x)},{_f(y)} {_f(x)},{_f(y + r)}"
                    f"V{_f(y + h - r)}Q{_f(x)},{_f(y + h)} {_f(x + r)},{_f(y + h)}H{_f(x + w)}Z")
        return (f"M{_f(x)},{_f(y)}H{_f(x + w - r)}Q{_f(x + w)},{_f(y)} {_f(x + w)},{_f(y + r)}"
                f"V{_f(y + h - r)}Q{_f(x + w)},{_f(y + h)} {_f(x + w - r)},{_f(y + h)}H{_f(x)}Z")
    if negative:  # data end at the bottom
        return (f"M{_f(x)},{_f(y)}V{_f(y + h - r)}Q{_f(x)},{_f(y + h)} {_f(x + r)},{_f(y + h)}"
                f"H{_f(x + w - r)}Q{_f(x + w)},{_f(y + h)} {_f(x + w)},{_f(y + h - r)}V{_f(y)}Z")
    return (f"M{_f(x)},{_f(y + h)}V{_f(y + r)}Q{_f(x)},{_f(y)} {_f(x + r)},{_f(y)}"
            f"H{_f(x + w - r)}Q{_f(x + w)},{_f(y)} {_f(x + w)},{_f(y + r)}V{_f(y + h)}Z")


def _marker(shape: int, x: float, y: float, r: float = 4.0) -> str:
    """Marker path for series ``shape`` (circle, square, triangle, diamond)."""
    k = shape % 4
    if k == 0:
        return f"M{_f(x - r)},{_f(y)}a{r},{r} 0 1,0 {_f(2 * r)},0a{r},{r} 0 1,0 {_f(-2 * r)},0Z"
    if k == 1:
        q = r * 0.9
        return f"M{_f(x - q)},{_f(y - q)}h{_f(2 * q)}v{_f(2 * q)}h{_f(-2 * q)}Z"
    if k == 2:
        q = r * 1.15
        return f"M{_f(x)},{_f(y - q)}L{_f(x + q)},{_f(y + q * 0.8)}L{_f(x - q)},{_f(y + q * 0.8)}Z"
    q = r * 1.2
    return f"M{_f(x)},{_f(y - q)}L{_f(x + q)},{_f(y)}L{_f(x)},{_f(y + q)}L{_f(x - q)},{_f(y)}Z"


def _legend_key(kind: str, j: int) -> str:
    if kind == "bar":
        inner = f'<rect x="2" y="1" width="12" height="10" rx="2" class="b b{j}"/>'
    elif kind == "scatter":
        inner = f'<path d="{_marker(j, 8, 6, 3.6)}" class="m m{j}"/>'
    else:
        inner = f'<line x1="1" y1="6" x2="17" y2="6" class="ln s{j}"/>'
    return f'<svg class="key" viewBox="0 0 18 12" aria-hidden="true" focusable="false">{inner}</svg>'


class _LabelPlacer:
    """Greedy placement of reference-line labels: first candidate whose box overlaps no earlier label."""

    def __init__(self) -> None:
        self.boxes: list[tuple[float, float, float, float]] = []

    @staticmethod
    def _box(lbl: str, x: float, y: float, anchor: str) -> tuple[float, float, float, float]:
        w = len(lbl) * 6.0 + 4
        x0 = x - w if anchor == "end" else x
        return (x0, y - 10, x0 + w, y + 3)

    def text(self, lbl: str, cands: Sequence[tuple[float, float, str]]) -> str:
        pick = cands[0]
        for c in cands:
            b = self._box(lbl, *c)
            if not any(b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3] for o in self.boxes):
                pick = c
                break
        self.boxes.append(self._box(lbl, *pick))
        x, y, anchor = pick
        return f'<text class="ref-lbl" x="{_f(x)}" y="{_f(y)}" text-anchor="{anchor}">{_e(lbl)}</text>'


class _ChartBuilder:
    """Renders ArtifactStore chart dicts to static SVG + a small JSON spec for the hover layer."""

    def __init__(self) -> None:
        self.n = 0
        self.specs: dict[str, dict[str, Any]] = {}

    # ---- public ---------------------------------------------------------------------------------

    def render(self, key: str, art: Mapping[str, Any]) -> dict[str, Any]:
        """Return a view dict for one chart figure (never raises)."""
        cid = f"c{self.n}"
        self.n += 1
        title = _s(art.get("title")) or _s(art.get("name")) or key
        kind = _s(art.get("kind"), "line").lower()
        view: dict[str, Any] = {
            "id": cid, "key": key, "title": title, "kind": kind, "legend": [], "subtitle": "",
            "svg": Markup(""), "note": "", "error": "",
        }
        note = _s(art.get("note"))
        diagonal = note.strip().lower().startswith("diagonal")
        caption = note.strip()[len("diagonal"):].lstrip(" :;—-") if diagonal else note
        view["note"] = caption or ("Dotted diagonal: y = x." if diagonal else "")
        xnames = None if diagonal or kind == "bar" else _parse_xnames(note)
        if xnames:
            items = sorted(xnames.items(), key=lambda kv: float(kv[0]))
            span = f"{items[0][0]} = {items[0][1]} … {items[-1][0]} = {items[-1][1]}"
            view["note"] = f"x-axis numbers map to periods: {span} (hover a point for its period)."
        series = [s for s in _l(art.get("series")) if isinstance(s, Mapping)]
        try:
            if kind == "bar":
                svg, spec, legend = self._bar(cid, title, art, series)
            else:
                if kind not in ("line", "step", "scatter"):
                    view["note"] = (view["note"] + " " if view["note"] else "") + f"(unknown chart kind {kind!r}; drawn as a line)"
                    kind = "line"
                svg, spec, legend = self._xy(cid, title, kind, art, series, diagonal, xnames)
        except Exception as exc:  # a malformed chart must never break the report
            log.warning("chart %s could not be rendered: %s", key, exc, exc_info=True)
            view["error"] = f"This chart could not be drawn ({type(exc).__name__}: {exc})."
            return view
        view["svg"] = Markup(svg)
        if len(legend) >= 2:
            view["legend"] = legend
        elif len(legend) == 1 and legend[0]["label"]:
            view["subtitle"] = legend[0]["label"]
        spec.update({"key": key, "title": title})
        self.specs[cid] = spec
        return view

    def json_blob(self) -> Markup:
        """All hover specs as JSON that is safe inside a ``<script type="application/json">``."""
        s = json.dumps(self.specs, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        s = s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        s = s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        return Markup(s)

    # ---- helpers --------------------------------------------------------------------------------

    @staticmethod
    def _labels(series: list[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
        return [
            {"label": _s(s.get("label")), "key": Markup(_legend_key(kind, j % _MAX_SERIES_COLORS))}
            for j, s in enumerate(series)
        ]

    @staticmethod
    def _svg_open(cid: str, title: str, w: float, h: float, extra_cls: str, desc: str) -> str:
        return (
            f'<svg class="plot{extra_cls}" viewBox="0 0 {_f(w)} {_f(h)}" role="img" tabindex="0" '
            f'aria-labelledby="{cid}-t {cid}-d" preserveAspectRatio="xMidYMid meet">'
            f'<title id="{cid}-t">{_e(title)}</title><desc id="{cid}-d">{_e(desc)}</desc>'
        )

    # ---- x/y charts (line, step, scatter) -------------------------------------------------------

    def _xy(
        self, cid: str, title: str, kind: str, art: Mapping[str, Any], series: list[Mapping[str, Any]],
        diagonal: bool, xnames: Mapping[str, str] | None = None,
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        xlog = _s(art.get("xscale")).lower() == "log"
        ylog = _s(art.get("yscale")).lower() == "log"
        pts: list[tuple[list[float | None], list[float | None]]] = []
        for s in series:
            xs = [_num(v) for v in _l(s.get("x"))]
            ys = [_num(v) for v in _l(s.get("y"))]
            n = min(len(xs), len(ys))
            pts.append((xs[:n], ys[:n]))
        refs = [r for r in _l(art.get("reference_lines")) if isinstance(r, Mapping) and _num(r.get("value")) is not None]
        xvals = [x for xs, _ in pts for x in xs if x is not None]
        yvals = [y for _, ys in pts for y in ys if y is not None]
        xvals += [_num(r.get("value")) for r in refs if _s(r.get("axis"), "y").lower() == "x"]  # type: ignore[misc]
        yvals += [_num(r.get("value")) for r in refs if _s(r.get("axis"), "y").lower() != "x"]  # type: ignore[misc]
        xd = _domain(xvals, art.get("xlim"), xlog)
        yd = _domain(yvals, art.get("ylim"), ylog)

        # Layout: left margin fits the widest y tick label.
        probe = _Axis(yd[0], yd[1], ylog, 0, 1)
        yt_probe = _ticks(probe)
        ml = 20 + max((len(lbl) for _, lbl in yt_probe), default=3) * _CHAR_W + 10
        mt, mr, mb = 14.0, 18.0, 46.0
        h = mt + _PLOT_H + mb
        x0, x1, ytop, ybot = ml, _W - mr, mt, mt + _PLOT_H
        X = _Axis(xd[0], xd[1], xlog, x0, x1)
        Y = _Axis(yd[0], yd[1], ylog, ybot, ytop)
        n_series = len(series)
        many = " many" if n_series > 3 else ""
        n_pts = sum(1 for xs, ys in pts for a, b in zip(xs, ys) if X.ok(a) and Y.ok(b))
        desc = (f"{kind} chart, {n_series} series, {n_pts} points; x: {_s(art.get('xlabel')) or 'x'}"
                f"{' (log scale)' if xlog else ''}; y: {_s(art.get('ylabel')) or 'y'}{' (log scale)' if ylog else ''}.")
        out = [self._svg_open(cid, title, _W, h, f" k-{kind}{many}", desc)]
        out.append(f'<defs><clipPath id="{cid}-clip"><rect x="{_f(x0)}" y="{_f(ytop - 1)}" '
                   f'width="{_f(x1 - x0)}" height="{_f(ybot - ytop + 2)}"/></clipPath></defs>')

        # Grid + ticks
        g = ['<g class="grid">']
        ax = ['<g class="axis">']
        for t, lbl in _ticks(Y):
            if not Y.inside(t):
                continue
            py = Y(t)
            g.append(f'<line x1="{_f(x0)}" x2="{_f(x1)}" y1="{_f(py)}" y2="{_f(py)}"/>')
            ax.append(f'<text x="{_f(x0 - 6)}" y="{_f(py + 3.5)}" text-anchor="end">{_e(lbl)}</text>')
        if xnames:  # categorical-looking x: one tick per mapped value, labelled with its period
            xt = sorted((float(k), nm) for k, nm in xnames.items() if X.inside(float(k)))
        else:
            xt = [(t, lbl) for t, lbl in _ticks(X, 6) if X.inside(t)]
        # Thin x tick labels that would collide.
        if xt:
            widest = max(len(lbl) for _, lbl in xt) * _CHAR_W + 10
            every = max(1, math.ceil(widest * len(xt) / max(x1 - x0, 1)))
        else:
            every = 1
        if xnames:
            xt = xt[::every]
            every = 1
        for i, (t, lbl) in enumerate(xt):
            px = X(t)
            g.append(f'<line x1="{_f(px)}" x2="{_f(px)}" y1="{_f(ytop)}" y2="{_f(ybot)}"/>')
            if i % every == 0:
                ax.append(f'<text x="{_f(px)}" y="{_f(ybot + 16)}" text-anchor="middle">{_e(lbl)}</text>')
        g.append("</g>")
        ax.append(f'<line class="base" x1="{_f(x0)}" x2="{_f(x1)}" y1="{_f(ybot)}" y2="{_f(ybot)}"/>')
        ax.append(f'<line class="base" x1="{_f(x0)}" x2="{_f(x0)}" y1="{_f(ytop)}" y2="{_f(ybot)}"/>')
        xl, yl = _s(art.get("xlabel")), _s(art.get("ylabel"))
        if xl:
            ax.append(f'<text class="lbl" x="{_f((x0 + x1) / 2)}" y="{_f(h - 8)}" text-anchor="middle">{_e(xl)}</text>')
        if yl:
            cy = (ytop + ybot) / 2
            ax.append(f'<text class="lbl" x="12" y="{_f(cy)}" text-anchor="middle" '
                      f'transform="rotate(-90 12 {_f(cy)})">{_e(yl)}</text>')
        ax.append("</g>")
        out += g + ax

        # Diagonal y = x (sampled so it is correct on mixed linear/log axes)
        data = [f'<g clip-path="url(#{cid}-clip)">']
        if diagonal:
            lo, hi = max(xd[0], yd[0]), min(xd[1], yd[1])
            if lo < hi and (not (xlog or ylog) or lo > 0):
                if xlog or ylog:
                    a, b = math.log10(lo), math.log10(hi)
                    ts = [10 ** (a + (b - a) * i / 48) for i in range(49)]
                else:
                    ts = [lo, hi]
                d = "M" + "L".join(f"{_f(X(t))},{_f(Y(t))}" for t in ts)
                data.append(f'<path class="diag" d="{d}"/>')

        # Series
        spec_series = []
        for j, ((xs, ys), s) in enumerate(zip(pts, series)):
            slot = j % _MAX_SERIES_COLORS
            spec_series.append({"label": _s(s.get("label")), "x": [_r5(v) for v in xs], "y": [_r5(v) for v in ys]})
            valid = [(X(a), Y(b)) if (X.ok(a) and Y.ok(b)) else None for a, b in zip(xs, ys)]  # type: ignore[arg-type]
            if kind == "scatter":
                marks = "".join(f'<path d="{_marker(j, px, py)}"/>' for p in valid if p for px, py in [p])
                data.append(f'<g class="m m{slot}">{marks}</g>')
                continue
            runs: list[list[int]] = [[]]  # runs of consecutive plottable point indices
            for i, p in enumerate(valid):
                if p is None:
                    if runs[-1]:
                        runs.append([])
                else:
                    runs[-1].append(i)
            runs = [r for r in runs if r]
            if kind == "step":
                d = self._step_path(runs, valid)
            else:
                d = "".join(
                    "M" + "L".join(f"{_f(valid[i][0])},{_f(valid[i][1])}" for i in r)  # type: ignore[index]
                    for r in runs if len(r) > 1
                )
            if d:
                data.append(f'<path class="ln s{slot}" d="{d}"/>')
            singles = [valid[r[0]] for r in runs if len(r) == 1]
            if singles and kind != "step":
                marks = "".join(f'<path d="{_marker(j, a, b, 4.5)}"/>' for a, b in singles)  # type: ignore[misc]
                data.append(f'<g class="m m{slot} pt">{marks}</g>')
        data.append("</g>")
        out += data

        # Reference lines (drawn over data, labelled with a surface halo; labels avoid each other)
        placer = _LabelPlacer()
        for r in refs:
            v = _num(r.get("value"))
            lbl = _s(r.get("label"))
            if _s(r.get("axis"), "y").lower() == "x":
                if v is None or not X.inside(v):
                    continue
                px = X(v)
                out.append(f'<line class="ref" x1="{_f(px)}" x2="{_f(px)}" y1="{_f(ytop)}" y2="{_f(ybot)}"/>')
                if lbl:
                    right = px > (x0 + x1) / 2
                    sides = ("end", "start") if right else ("start", "end")
                    cands = [(px - 4 if a == "end" else px + 4, y, a) for y in (ytop + 10, ybot - 6) for a in sides]
                    out.append(placer.text(lbl, cands))
            else:
                if v is None or not Y.inside(v):
                    continue
                py = Y(v)
                out.append(f'<line class="ref" x1="{_f(x0)}" x2="{_f(x1)}" y1="{_f(py)}" y2="{_f(py)}"/>')
                if lbl:
                    cands = [(x, y, a) for x, a in ((x1 - 4, "end"), (x0 + 4, "start")) for y in (py - 4, py + 13)]
                    out.append(placer.text(lbl, cands))

        # Hover layer (positioned by the inline script)
        hover = ['<g class="hover" visibility="hidden">',
                 f'<line class="xhair" x1="0" x2="0" y1="{_f(ytop)}" y2="{_f(ybot)}"/>']
        hover += [f'<circle class="hdot m{j % _MAX_SERIES_COLORS}" r="4.5" cx="-10" cy="-10"/>' for j in range(n_series)]
        hover.append("</g>")
        out += hover
        if n_pts == 0:
            out.append(f'<text class="empty" x="{_f((x0 + x1) / 2)}" y="{_f((ytop + ybot) / 2)}" '
                       f'text-anchor="middle">No plottable points</text>')
        out.append("</svg>")
        spec = {
            "kind": kind, "w": _W, "h": h, "plot": [round(x0, 1), round(ytop, 1), round(x1, 1), round(ybot, 1)],
            "x": X.spec(), "y": Y.spec(), "xlabel": xl, "ylabel": yl, "series": spec_series,
        }
        if xnames:
            spec["xnames"] = dict(xnames)
        return "".join(out), spec, self._labels(series, kind)

    @staticmethod
    def _step_path(runs: list[list[int]], valid: list[tuple[float, float] | None]) -> str:
        """steps-post: y[i] holds on [x[i], x[i+1]). A run ends at the next plottable x (so a gap
        closes its last step there); the final step extends one median bin width."""
        xs = [p[0] for p in valid if p]
        widths = sorted(b - a for a, b in zip(xs, xs[1:]) if b > a)
        tail = widths[len(widths) // 2] if widths else 0.0
        parts = []
        for r in runs:
            pts = [valid[i] for i in r]
            d = f"M{_f(pts[0][0])},{_f(pts[0][1])}"  # type: ignore[index]
            for p in pts[1:]:
                d += f"H{_f(p[0])}V{_f(p[1])}"  # type: ignore[index]
            last_x = pts[-1][0]  # type: ignore[index]
            nxt = next((valid[k][0] for k in range(r[-1] + 1, len(valid)) if valid[k]), None)  # type: ignore[index]
            close_x = nxt if nxt is not None else last_x + tail
            if close_x > last_x:
                d += f"H{_f(close_x)}"
            parts.append(d)
        return "".join(parts)

    # ---- bar charts -----------------------------------------------------------------------------

    def _bar(
        self, cid: str, title: str, art: Mapping[str, Any], series: list[Mapping[str, Any]]
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        cats: list[str] = []
        seen: dict[str, int] = {}
        for s in series:
            for c in _l(s.get("x")):
                c = _s(c)
                if c not in seen:
                    seen[c] = len(cats)
                    cats.append(c)
        values: list[list[float | None]] = []
        for s in series:
            row: list[float | None] = [None] * len(cats)
            for c, v in zip(_l(s.get("x")), _l(s.get("y"))):
                row[seen[_s(c)]] = _num(v)
            values.append(row)
        ylog = _s(art.get("yscale")).lower() == "log"
        allv = [v for row in values for v in row if v is not None]
        refs = [r for r in _l(art.get("reference_lines")) if isinstance(r, Mapping) and _num(r.get("value")) is not None
                and _s(r.get("axis"), "y").lower() != "x"]
        vd = _domain(allv + [_num(r.get("value")) for r in refs], art.get("ylim"), ylog, include_zero=True)  # type: ignore[list-item]
        n, k = len(cats), max(1, len(series))
        maxlen = max((len(c) for c in cats), default=0)
        # Long category names read best as rows; short ordered labels (e.g. M2's 'YYYY-MM' windows)
        # stay as columns so time runs left to right, until there are too many to fit.
        horizontal = maxlen > 10 or n > 40
        xl, yl = _s(art.get("xlabel")), _s(art.get("ylabel"))
        single = len(series) == 1
        if horizontal:
            label_chars = min(maxlen, 30)
            ml = 12 + label_chars * _CHAR_W
            mr = 56.0 if single else 20.0
            thick = 14.0 if k == 1 else max(6.0, min(12.0, 22.0 / k))
            band = max(20.0, k * thick + (k - 1) * 2 + 8)
            mt, mb = (22.0 if any(_s(r.get("label")) for r in refs) else 8.0), 44.0
            h = mt + max(n, 1) * band + mb
            x0, x1, ytop, ybot = ml, _W - mr, mt, mt + max(n, 1) * band
            V = _Axis(vd[0], vd[1], ylog, x0, x1)
        else:
            probe = _Axis(vd[0], vd[1], ylog, 0, 1)
            ml = 20 + max((len(lbl) for _, lbl in _ticks(probe)), default=3) * _CHAR_W + 10
            mr, mt, mb = 18.0, 14.0, 46.0
            h = mt + _PLOT_H + mb
            x0, x1, ytop, ybot = ml, _W - mr, mt, mt + _PLOT_H
            band = (x1 - x0) / max(n, 1)
            thick = max(2.0, min(24.0, (band * 0.72 - 2 * (k - 1)) / k))
            V = _Axis(vd[0], vd[1], ylog, ybot, ytop)
        base_v = vd[0] if ylog else (0.0 if vd[0] <= 0 <= vd[1] else (vd[0] if vd[0] > 0 else vd[1]))
        base = V(base_v)
        desc = f"bar chart, {len(series)} series over {n} categories; values: {yl or 'value'}{' (log scale)' if ylog else ''}."
        out = [self._svg_open(cid, title, _W, h, " k-bar" + (" horiz" if horizontal else ""), desc)]
        g, ax = ['<g class="grid">'], ['<g class="axis">']
        for t, lbl in _ticks(V, 5):
            if not V.inside(t):
                continue
            p = V(t)
            if horizontal:
                g.append(f'<line x1="{_f(p)}" x2="{_f(p)}" y1="{_f(ytop)}" y2="{_f(ybot)}"/>')
                ax.append(f'<text x="{_f(p)}" y="{_f(ybot + 16)}" text-anchor="middle">{_e(lbl)}</text>')
            else:
                g.append(f'<line x1="{_f(x0)}" x2="{_f(x1)}" y1="{_f(p)}" y2="{_f(p)}"/>')
                ax.append(f'<text x="{_f(x0 - 6)}" y="{_f(p + 3.5)}" text-anchor="end">{_e(lbl)}</text>')
        g.append("</g>")
        if horizontal:
            ax.append(f'<line class="base" x1="{_f(base)}" x2="{_f(base)}" y1="{_f(ytop)}" y2="{_f(ybot)}"/>')
            max_chars = max(4, int((ml - 12) / _CHAR_W))
            for i, c in enumerate(cats):
                cy = ytop + (i + 0.5) * band
                shown = c if len(c) <= max_chars else c[: max_chars - 1] + "…"
                ax.append(f'<text class="cat" x="{_f(x0 - 6)}" y="{_f(cy + 3.5)}" text-anchor="end">'
                          f'<title>{_e(c)}</title>{_e(shown)}</text>')
            if yl:
                ax.append(f'<text class="lbl" x="{_f((x0 + x1) / 2)}" y="{_f(h - 8)}" text-anchor="middle">{_e(yl)}</text>')
        else:
            ax.append(f'<line class="base" x1="{_f(x0)}" x2="{_f(x1)}" y1="{_f(base)}" y2="{_f(base)}"/>')
            widest = maxlen * _CHAR_W + 8
            every = max(1, math.ceil(widest / max(band, 1)))
            max_chars = 14
            for i, c in enumerate(cats):
                if i % every:
                    continue
                cx = x0 + (i + 0.5) * band
                shown = c if len(c) <= max_chars else c[: max_chars - 1] + "…"
                ax.append(f'<text class="cat" x="{_f(cx)}" y="{_f(ybot + 16)}" text-anchor="middle">'
                          f'<title>{_e(c)}</title>{_e(shown)}</text>')
            if xl:
                ax.append(f'<text class="lbl" x="{_f((x0 + x1) / 2)}" y="{_f(h - 8)}" text-anchor="middle">{_e(xl)}</text>')
            if yl:
                cy = (ytop + ybot) / 2
                ax.append(f'<text class="lbl" x="12" y="{_f(cy)}" text-anchor="middle" '
                          f'transform="rotate(-90 12 {_f(cy)})">{_e(yl)}</text>')
        ax.append("</g>")
        out += g + ax
        bars = []
        group = k * thick + (k - 1) * 2
        for j, row in enumerate(values):
            slot = j % _MAX_SERIES_COLORS
            paths = []
            for i, v in enumerate(row):
                if v is None or not V.ok(v):
                    continue
                vv = min(max(v, vd[0]), vd[1])
                p = V(vv)
                off = (band - group) / 2 + j * (thick + 2)
                if horizontal:
                    y = ytop + i * band + off
                    neg = p < base
                    x, w = (p, base - p) if neg else (base, p - base)
                    if w > 0.2:
                        paths.append(f'<path d="{_bar_path(x, y, w, thick, horizontal=True, negative=neg)}"/>')
                    if single:
                        tx = p - 4 if neg else p + 4
                        bars.append(f'<text class="tip-lbl" x="{_f(tx)}" y="{_f(y + thick / 2 + 3.5)}" '
                                    f'text-anchor="{"end" if neg else "start"}">{_e(fmt_num(v, 3))}</text>')
                else:
                    x = x0 + i * band + off
                    neg = p > base
                    y, hh = (base, p - base) if neg else (p, base - p)
                    if hh > 0.2:
                        paths.append(f'<path d="{_bar_path(x, y, thick, hh, horizontal=False, negative=neg)}"/>')
            bars.insert(0, f'<g class="b b{slot}">{"".join(paths)}</g>')
        out += bars
        for r in refs:
            v = _num(r.get("value"))
            if v is None or not V.inside(v):
                continue
            p = V(v)
            lbl = _s(r.get("label"))
            if horizontal:
                out.append(f'<line class="ref" x1="{_f(p)}" x2="{_f(p)}" y1="{_f(ytop)}" y2="{_f(ybot)}"/>')
                if lbl:
                    right = p > (x0 + x1) / 2
                    out.append(f'<text class="ref-lbl" x="{_f(p - 4 if right else p + 4)}" y="{_f(ytop - 6)}" '
                               f'text-anchor="{"end" if right else "start"}">{_e(lbl)}</text>')
            else:
                out.append(f'<line class="ref" x1="{_f(x0)}" x2="{_f(x1)}" y1="{_f(p)}" y2="{_f(p)}"/>')
                if lbl:
                    out.append(f'<text class="ref-lbl" x="{_f(x1 - 4)}" y="{_f(p - 4)}" text-anchor="end">{_e(lbl)}</text>')
        out.append('<g class="hover" visibility="hidden"><rect class="hband" x="0" y="0" width="0" height="0" rx="3"/></g>')
        if not allv:
            out.append(f'<text class="empty" x="{_f((x0 + x1) / 2)}" y="{_f((ytop + ybot) / 2)}" '
                       f'text-anchor="middle">No values</text>')
        out.append("</svg>")
        spec = {
            "kind": "bar", "orient": "h" if horizontal else "v", "w": _W, "h": h,
            "plot": [round(x0, 1), round(ytop, 1), round(x1, 1), round(ybot, 1)],
            "b0": round(ytop if horizontal else x0, 2), "band": round(band, 3), "cats": cats,
            "xlabel": xl, "ylabel": yl,
            "series": [{"label": _s(s.get("label")), "y": [_r5(v) for v in row]} for s, row in zip(series, values)],
        }
        return "".join(out), spec, self._labels(series, "bar")


# ==================================================================================================
# Generic value rendering (module details, sandbox info, ...)
# ==================================================================================================


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _cell(v: Any) -> str:
    cls = ' class="num"' if isinstance(v, (int, float)) and not isinstance(v, bool) else ""
    return f"<td{cls}>{_e(fmt_value(v, max_chars=160))}</td>"


def _value_html(v: Any, depth: int = 0, *, max_rows: int = 60) -> str:
    """Escaped HTML for an arbitrary JSON value: scalars inline, lists of records as tables,
    mappings as key/value lists (depth-limited, truncated)."""
    if _is_scalar(v):
        return f'<span class="val">{_e(fmt_value(v))}</span>'
    if isinstance(v, (list, tuple)):
        items = list(v)
        if not items:
            return '<span class="muted">(none)</span>'
        if all(isinstance(x, Mapping) for x in items):
            cols: list[str] = []
            for x in items:
                for c in x:
                    if c not in cols:
                        cols.append(str(c))
            if len(cols) <= 12 and depth < 3:
                head = "".join(f"<th>{_e(c)}</th>" for c in cols)
                body = "".join(
                    "<tr>" + "".join(_cell(x.get(c)) for c in cols) + "</tr>" for x in items[:max_rows]
                )
                more = (f'<p class="muted small">Showing {max_rows} of {len(items)} rows.</p>'
                        if len(items) > max_rows else "")
                return f'<div class="table-wrap"><table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{more}'
        if all(_is_scalar(x) for x in items):
            return f'<span class="val">{_e(fmt_value(items, max_items=40, max_chars=600))}</span>'
    if isinstance(v, Mapping) and depth < 3:
        if not v:
            return '<span class="muted">(none)</span>'
        rows = "".join(
            f"<dt>{_e(k)}</dt><dd>{_value_html(x, depth + 1, max_rows=max_rows)}</dd>" for k, x in list(v.items())[:80]
        )
        return f'<dl class="kv">{rows}</dl>'
    try:
        text = json.dumps(v, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(v)
    if len(text) > 20000:
        text = text[:20000] + "\n… (truncated)"
    return f'<pre class="code">{_e(text)}</pre>'


def _kv(pairs: Sequence[tuple[str, Any]]) -> list[dict[str, Any]]:
    """Key/value rows for the template (values pre-formatted, None kept as a dash)."""
    return [{"k": k, "v": fmt_value(v) if not isinstance(v, Markup) else v, "mono": False} for k, v in pairs]


# ==================================================================================================
# View model
# ==================================================================================================


def _safe(section: str, fn: Callable[..., Any], *args: Any, default: Any = None) -> Any:
    """Run a section builder; on failure log it and return a visible placeholder."""
    try:
        return fn(*args)
    except Exception as exc:
        log.warning("report section %r could not be rendered: %s", section, exc, exc_info=True)
        out = dict(default or {})
        out["section_error"] = f"This section could not be rendered ({type(exc).__name__}: {exc})."
        return out


def _verdict_view(rep: Mapping[str, Any]) -> dict[str, Any]:
    v = _d(rep.get("verdict"))
    key = _s(v.get("verdict")).lower()
    meta = _VERDICT.get(key, _VERDICT_UNKNOWN)
    bands = _d(v.get("bands"))
    ready_min = _num(bands.get("ready_min"))
    cond_min = _num(bands.get("conditional_min"))
    cap = _num(bands.get("blocked_score_cap"))
    ready_min = 80.0 if ready_min is None else min(max(ready_min, 0.0), 100.0)
    cond_min = 60.0 if cond_min is None else min(max(cond_min, 0.0), ready_min)
    cap = 49.0 if cap is None else min(max(cap, 0.0), 100.0)
    score = _num(v.get("score"))
    raw = _num(v.get("raw_score"))
    capped = bool(v.get("capped"))
    blockers = [_s(b) for b in _l(v.get("blockers")) if _s(b)]
    reasons = [_s(r) for r in _l(v.get("reasons")) if _s(r) and _s(r) not in blockers]
    clamp = lambda x: None if x is None else min(max(x, 0.0), 100.0)  # noqa: E731
    return {
        "present": bool(v),
        "key": key or "unknown",
        "tone": meta["tone"], "icon": meta["icon"], "word": meta["word"], "meaning": meta["meaning"],
        "label": _s(v.get("label")),
        "summary": _s(v.get("summary")),
        "score": score,
        "score_text": "—" if score is None else f"{score:.1f}",
        "score_pos": clamp(score),
        "raw": raw,
        "raw_pos": clamp(raw) if capped and raw is not None else None,
        "capped": capped,
        "coverage": _num(v.get("coverage")),
        "min_coverage": _num(bands.get("min_coverage_ready")),
        "ready_min": ready_min, "cond_min": cond_min, "cap": cap,
        "show_cap": key == "blocked" or capped,
        "blockers": blockers,
        "reasons": reasons,
        "isolation": _s(v.get("isolation")) or None,
        "isolation_warning": _s(v.get("isolation_warning")) or None,
    }


def _axes_by_id(rep: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for a in _l(_d(rep.get("verdict")).get("axes")):
        a = _d(a)
        if _s(a.get("module_id")):
            out[_s(a.get("module_id"))] = a
    return out


def _check_view(c: Mapping[str, Any]) -> dict[str, Any]:
    passed = c.get("passed")
    score = _num(c.get("score"))
    metric = _s(c.get("metric")) or _s(c.get("name")) or "check"
    return {
        "name": _s(c.get("name")) or metric,
        "metric": metric,
        "op": fmt_op(c.get("op")),
        "threshold": fmt_num(_num(c.get("threshold")) if _num(c.get("threshold")) is not None else c.get("threshold")),
        "value": fmt_num(_num(c.get("value"))) if c.get("value") is not None else "—",
        "passed": passed if isinstance(passed, bool) else None,
        "score": None if score is None else round(100 * min(max(score, 0.0), 1.0), 1),
        "ideal": fmt_num(_num(c.get("ideal"))) if _num(c.get("ideal")) is not None else "",
        "floor": fmt_num(_num(c.get("floor"))) if _num(c.get("floor")) is not None else "",
        "scale": _s(c.get("scale")),
        "description": _s(c.get("description")),
    }


def _primary_check(checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The check that explains the axis: a failing one, else the weakest-scored, else unevaluated."""
    if not checks:
        return None
    failing = [c for c in checks if c["passed"] is False]
    pool = failing or checks
    scored = [c for c in pool if c["score"] is not None]
    if scored:
        return min(scored, key=lambda c: c["score"])
    unevaluated = [c for c in checks if c["passed"] is None]
    return (unevaluated or pool)[0]


def _modules_view(rep: Mapping[str, Any], charts: _ChartBuilder) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per-module sections plus the leftover artifacts that belong to no module."""
    arts = _d(rep.get("artifacts"))
    axes = _axes_by_id(rep)
    used: set[str] = set()
    anchors: set[str] = set()
    views = []
    for idx, m in enumerate(_l(rep.get("modules"))):
        m = _d(m)
        view = _safe(f"module {idx}", _module_view, m, idx, arts, axes, charts, used,
                     default={"code": _s(m.get("code"), "?"), "title": _s(m.get("title"), "module"),
                              "id": _s(m.get("module_id")), "tone": "neutral", "status_label": "Unknown",
                              "status_icon": "i-unknown", "checks": [], "charts": [], "tables": []})
        anchor = "m-" + _slug(view.get("id") or view.get("code") or str(idx))
        while anchor in anchors:
            anchor += f"-{idx}"
        anchors.add(anchor)
        view["anchor"] = anchor
        views.append(view)
    extras = []
    for key, art in arts.items():
        if key in used:
            continue
        extras.append(_artifact_view(key, art, charts))
        used.add(key)
    return views, extras


def _artifact_view(key: str, art: Any, charts: _ChartBuilder) -> dict[str, Any]:
    art = _d(art)
    typ = _s(art.get("type")).lower()
    if typ == "table" or ("columns" in art and "rows" in art and "series" not in art):
        return {"type": "table", **_table_view(key, art)}
    if typ == "chart" or "series" in art:
        return {"type": "chart", **charts.render(key, art)}
    return {"type": "other", "key": key, "title": _s(art.get("title")) or key, "html": Markup(_value_html(art))}


def _table_view(key: str, art: Mapping[str, Any]) -> dict[str, Any]:
    cols = [_s(c) for c in _l(art.get("columns"))]
    rows = []
    width = len(cols)
    for r in _l(art.get("rows")):
        cells = _l(r) if isinstance(r, (list, tuple)) else [r]
        width = max(width, len(cells))
        rows.append([{"v": fmt_value(c, max_chars=300),
                      "num": isinstance(c, (int, float)) and not isinstance(c, bool)} for c in cells])
    cols += [""] * (width - len(cols))
    num_cols = [
        bool(rows) and all(r[i]["num"] or r[i]["v"] == "—" for r in rows if i < len(r))
        and any(i < len(r) and r[i]["num"] for r in rows)
        for i in range(width)
    ]
    return {"key": key, "title": _s(art.get("title")) or _s(art.get("name")) or key, "columns": cols, "rows": rows,
            "num_cols": num_cols, "truncated": bool(art.get("truncated")), "note": _s(art.get("note"))}


def _module_view(
    m: Mapping[str, Any], idx: int, arts: Mapping[str, Any], axes: Mapping[str, Any], charts: _ChartBuilder,
    used: set[str],
) -> dict[str, Any]:
    mid = _s(m.get("module_id"))
    status_key = _s(m.get("status")).lower()
    st = _status(status_key)
    oc = _outcome(m.get("gate_outcome"))
    checks = [_check_view(_d(c)) for c in _l(m.get("checks")) if isinstance(c, Mapping)]
    primary = _primary_check(checks)
    axis = _d(axes.get(mid))
    unscored = mid in _UNSCORED or (not axis and mid == "file_safety")
    if axis:
        score = _num(axis.get("score"))
        weight = _num(axis.get("weight"))
        counted = bool(axis.get("counted"))
    else:
        s = _num(m.get("score"))
        score = None if s is None or status_key in ("skipped", "error") else round(100 * s, 1)
        weight, counted = None, False
    # Artifacts: those the module lists, then any others the store attributes to it.
    keys: list[str] = []
    for k in _l(m.get("artifacts")):
        k = _s(k)
        if k and k not in keys:
            keys.append(k)
    for k, a in arts.items():
        if _s(_d(a).get("module")) == mid and mid and k not in keys:
            keys.append(k)
    chart_views, table_views, other_views, missing = [], [], [], []
    for k in keys:
        if k in used:
            continue
        if k not in arts:
            missing.append(k)
            continue
        used.add(k)
        av = _artifact_view(k, arts[k], charts)
        {"chart": chart_views, "table": table_views}.get(av["type"], other_views).append(av)
    metrics = [{"k": _s(k), "v": fmt_value(v), "gated": any(c["metric"] == k for c in checks)}
               for k, v in _d(m.get("metrics")).items()]
    details = _d(m.get("details"))
    params = _d(m.get("params"))
    error = _s(m.get("error"))
    err_lines = [ln for ln in error.strip().splitlines() if ln.strip()]
    skip_reason = _s(m.get("skip_reason"))
    finding = _s(m.get("finding"))
    return {
        "id": mid,
        "code": _s(m.get("code"), "?"),
        "title": _s(m.get("title")) or mid or f"module {idx + 1}",
        "status": status_key or "unknown",
        "tone": st.tone, "status_label": st.label, "status_icon": st.icon,
        "gate": _s(m.get("gate")).lower() or "—",
        "outcome": _s(m.get("gate_outcome")).lower(),
        "outcome_tone": oc.tone, "outcome_label": oc.label, "outcome_icon": oc.icon,
        "screening": bool(m.get("screening")),
        "score": score,
        "weight": weight,
        "counted": counted,
        "unscored": unscored,
        "blurb": _s(m.get("description")) or _BLURBS.get(mid, ""),
        "finding": finding,
        "notes": [_s(n) for n in _l(m.get("notes")) if _s(n)],
        "checks": checks,
        "primary": primary,
        "metrics": metrics,
        "charts": chart_views,
        "tables": table_views,
        "others": other_views,
        "missing_artifacts": missing,
        "details_html": Markup(_value_html(details)) if details else None,
        "params": [{"k": _s(k), "v": fmt_value(v)} for k, v in params.items()],
        "skip_reason": skip_reason,
        "skip_help": _skip_help(skip_reason) if skip_reason or status_key == "skipped" else "",
        "error": error,
        "error_head": err_lines[-1] if err_lines else "",
        "partial": status_key == "error" and bool(chart_views or table_views),
        "duration": fmt_duration(m.get("duration_s")) if m.get("duration_s") is not None else "",
        "seed": m.get("seed"),
        # The generic screening caveat, unless the module already states it in its own notes.
        "screening_text": "" if any("not proof" in n.lower() for n in _l(m.get("notes")) if isinstance(n, str))
        else _SCREENING_TEXT,
    }


_EXIT_MEANINGS = {
    0: "CI passes: no hard gate failed, no module errored, verdict above --fail-on",
    1: "CI fails: the run is blocked or the verdict is at or below --fail-on",
    2: "CI fails: a module errored or the model could not be loaded, so the gate result is not trustworthy",
}


def _gate_view(rep: Mapping[str, Any], modules: list[dict[str, Any]]) -> dict[str, Any]:
    gate = _d(rep.get("gate"))
    hard = []
    for h in _l(gate.get("hard_gates")):
        h = _d(h)
        st = _status(h.get("status"))
        oc = _outcome(h.get("gate_outcome"))
        mid = _s(h.get("module_id"))
        anchor = next((m["anchor"] for m in modules if m.get("id") == mid), "")
        hard.append({"code": _s(h.get("code"), "?"), "title": _s(h.get("title")) or mid, "anchor": anchor,
                     "tone": oc.tone if st.tone != "critical" else "critical", "outcome_label": oc.label,
                     "icon": oc.icon if st.tone != "critical" else st.icon, "status_label": st.label})
    if not hard:  # derive from the module list
        for m in modules:
            if m.get("gate") == "hard":
                hard.append({"code": m["code"], "title": m["title"], "anchor": m["anchor"],
                             "tone": m["outcome_tone"] if m["tone"] != "critical" else "critical",
                             "outcome_label": m["outcome_label"],
                             "icon": m["outcome_icon"] if m["tone"] != "critical" else m["status_icon"],
                             "status_label": m["status_label"]})
    code = gate.get("exit_code")
    code_i = code if isinstance(code, int) and not isinstance(code, bool) else None
    meaning = _s(gate.get("exit_meaning")) or _EXIT_MEANINGS.get(-1 if code_i is None else code_i, "")
    aborted = gate.get("aborted")
    abort_reason = _s(gate.get("abort_reason"))
    if not abort_reason and aborted and not isinstance(aborted, bool):
        abort_reason = _s(aborted)  # older reports stored the reason in "aborted"
    if aborted is True and not abort_reason:
        abort_reason = "no reason recorded"
    load_error = _s(gate.get("load_error")) or _s(_d(rep.get("model")).get("load_error"))
    return {
        "hard": hard,
        "exit_code": "—" if code_i is None else str(code_i),
        "exit_tone": "neutral" if code_i is None else ("good" if code_i == 0 else "critical"),
        "exit_meaning": meaning,
        "fail_on": _s(gate.get("fail_on")),
        "aborted": abort_reason,
        "load_error": "" if abort_reason else load_error,
        "disabled": [_s(x) for x in _l(gate.get("disabled")) if _s(x)],
        "deselected": [_s(x) for x in _l(gate.get("deselected")) if _s(x)],
    }


def _threshold_note(decl: Mapping[str, Any]) -> str:
    """"auto-calibrated (…)" for a threshold malvalid calibrated on a held-out slice, else ""."""
    extras = _d(decl.get("extras"))
    if _s(extras.get("threshold_source")) != "calibrated":
        return ""
    cal = _d(extras.get("threshold_calibration"))
    fpr = _num(cal.get("target_fpr"))
    n = cal.get("n_benign")
    bits = []
    if fpr is not None:
        bits.append(f"target FPR {fpr:.2%}")
    period = cal.get("period")
    when = (f" dated {_s(period[0])} to {_s(period[1])}"
            if isinstance(period, list) and len(period) == 2 else "")
    policy = _s(cal.get("policy"))
    if isinstance(n, int):
        if policy == "earliest":
            bits.append(f"on the earliest {n:,} benign rows{when}, held out from every test (held threshold)")
        elif policy == "uniform":
            bits.append(f"on {n:,} benign rows{when} hash-sampled across the test period and held out from every "
                        "test (uniform policy: FPR is close to the target by construction)")
        else:
            bits.append(f"on {n:,} held-out benign rows not used by any test")
    return "auto-calibrated" + (f" ({', '.join(bits)})" if bits else "")


def _header_view(rep: Mapping[str, Any]) -> dict[str, Any]:
    tool = _d(rep.get("tool"))
    run = _d(rep.get("run"))
    model = _d(rep.get("model"))
    decl = {**_d(model.get("declarations")), **model}
    cfg_title = _s(rep.get("title")) or _s(_d(_d(rep.get("config")).get("report")).get("title"))
    thr = _num(decl.get("operating_threshold"))
    model_bits = [b for b in (_s(decl.get("class_name")), _s(decl.get("model_kind")), _s(decl.get("feature_version"))) if b]
    return {
        "tool": _TOOL_DISPLAY.get(_s(tool.get("name"), "malvalid") or "malvalid", _s(tool.get("name"))),
        "version": _s(tool.get("version")),
        "schema": _s(rep.get("schema_version")),
        "schema_ok": _s(rep.get("schema_version")) in ("", *_SUPPORTED_SCHEMAS),
        "title": cfg_title or "Production-readiness report",
        "custom_title": bool(cfg_title),
        "model_line": " · ".join(model_bits),
        "threshold": "" if thr is None else fmt_num(thr),
        "threshold_note": _threshold_note(decl),
        "adapter": _s(decl.get("adapter_path")),
        "run_id": _s(run.get("id")),
        "started": fmt_time(run.get("started_at")) if run.get("started_at") else "",
        "duration": fmt_duration(run.get("duration_s")) if run.get("duration_s") is not None else "",
    }


def _callouts(rep: Mapping[str, Any]) -> dict[str, Any]:
    corpus = _d(rep.get("corpus"))
    return {
        "warnings": [_s(w) for w in _l(rep.get("warnings")) if _s(w)],
        "synthetic": bool(corpus.get("synthetic")),
        "corpus_error": _s(corpus.get("error")),
        "corpus_name": _s(corpus.get("name")),
    }


def _appendix_view(rep: Mapping[str, Any], modules: list[dict[str, Any]]) -> dict[str, Any]:
    model = _d(rep.get("model"))
    decl = {**_d(model.get("declarations")), **model}
    decl_keys = ("feature_version", "model_kind", "operating_threshold", "class_name", "adapter_path",
                 "training_hashes_path", "training_cutoff", "has_featurize", "model_paths")
    model_rows = _kv([(k, decl.get(k)) for k in decl_keys if k in decl])
    extras = _d(decl.get("extras"))
    if extras:
        model_rows += _kv([("extras", extras)])
    if decl.get("adapter_sha256"):
        model_rows.append({"k": "adapter_sha256", "v": _s(decl.get("adapter_sha256")), "mono": True})
    artifacts = []
    for a in _l(model.get("artifacts")):
        a = _d(a)
        artifacts.append({"path": _s(a.get("path")), "sha256": _s(a.get("sha256")), "size": fmt_bytes(a.get("size")),
                          "format": _s(a.get("format")), "pickle": a.get("is_pickle")})
    ta = _d(model.get("tree_access"))
    tree_rows = _kv([(k, ta.get(k)) for k in ("available", "n_trees", "fidelity_max_abs_diff", "note") if k in ta])
    for k in ("load_error", "query_count"):
        if model.get(k) not in (None, ""):
            model_rows += _kv([(k, model.get(k))])
    caps = [_s(c) for c in _l(rep.get("capabilities")) if _s(c)]
    if "capabilities" in rep:
        model_rows.append({"k": "capabilities", "v": ", ".join(caps) if caps else "(none)", "mono": False})

    tm = _d(rep.get("training_manifest"))
    tm_keys = ("path", "declared", "n_hashes", "n_in_corpus", "cutoff", "cutoff_parsed")
    tm_rows = _kv([(k, tm.get(k)) for k in tm_keys if k in tm])
    tm_rows += _kv([(k, v) for k, v in tm.items() if k not in tm_keys and k != "warnings"])
    tm_warnings = [_s(w) for w in _l(tm.get("warnings")) if _s(w)]

    corpus = _d(rep.get("corpus"))
    c_keys = ("name", "version", "feature_version", "n", "dim", "synthetic", "time_range", "source",
              "excluded_training_members", "error")
    corpus_rows = _kv([(k, corpus.get(k)) for k in c_keys if k in corpus])
    splits = []
    for name, sp in _d(corpus.get("splits")).items():
        sp = _d(sp)
        splits.append({"name": _s(name), **{k: fmt_num(sp.get(k)) for k in ("n", "malicious", "benign", "unlabeled")}})
    roles = [{"role": _s(r), "splits": ", ".join(_s(x) for x in _l(v)) if isinstance(v, (list, tuple)) else _s(v)}
             for r, v in _d(corpus.get("roles")).items()]
    known = set(c_keys) | {"content_hash", "splits", "roles"}
    corpus_rows += _kv([(k, v) for k, v in corpus.items() if k not in known])

    schema = _d(rep.get("schema"))
    schema_rows = _kv(list(schema.items()))

    sandbox = rep.get("sandbox")
    sb = _d(sandbox)
    sandbox_flag = ""
    if sb.get("enabled") is False:
        sandbox_flag = "The sandbox was disabled: the model ran in-process (debug mode)."
    elif sb.get("isolation") == "process_only":
        sandbox_flag = ("Reduced isolation (isolation: process_only): this platform has no OS sandbox. The model ran "
                        "in a separate worker process without network or file-system isolation.")
    elif sb.get("network_isolated") is False:
        sandbox_flag = "The sandbox worker was NOT network-isolated."

    env = _d(rep.get("environment"))
    env_rows = _kv([(k, v) for k, v in env.items() if k != "libraries"])
    libs = [{"name": _s(k), "version": _s(v) if v not in (None, "") else "not installed", "missing": v in (None, "")}
            for k, v in _d(env.get("libraries")).items()]

    run = _d(rep.get("run"))
    run_rows = []
    for k in ("id", "started_at", "finished_at", "duration_s", "seed", "out_dir", "command"):
        if k not in run:
            continue
        v = run.get(k)
        if k in ("started_at", "finished_at"):
            v = fmt_time(v)
        elif k == "duration_s":
            v = fmt_duration(v)
        run_rows.append({"k": k, "v": fmt_value(v), "mono": k in ("out_dir", "command")})
    run_rows += _kv([(k, v) for k, v in run.items() if k not in ("id", "started_at", "finished_at", "duration_s", "seed", "out_dir", "command")])
    seeds = [{"code": m["code"], "title": m["title"], "seed": fmt_value(m.get("seed")), "duration": m.get("duration") or "—"}
             for m in modules]

    config = rep.get("config")
    config_text = ""
    if config is not None:
        try:
            import yaml

            config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=100)
        except Exception:  # pragma: no cover - yaml is a core dependency
            config_text = json.dumps(config, indent=2, ensure_ascii=False, default=str)

    return {
        "model_rows": model_rows,
        "artifacts": artifacts,
        "tree_rows": tree_rows,
        "tm_rows": tm_rows,
        "tm_warnings": tm_warnings,
        "corpus_rows": corpus_rows,
        "content_hash": _s(corpus.get("content_hash")),
        "splits": splits,
        "roles": roles,
        "schema_rows": schema_rows,
        "sandbox_html": Markup(_value_html(sandbox)) if sandbox not in (None, {}, []) else None,
        "sandbox_flag": sandbox_flag,
        "env_rows": env_rows,
        "libs": libs,
        "run_rows": run_rows,
        "seeds": seeds,
        "config_text": config_text,
        "disclaimers": [_s(d) for d in _l(rep.get("disclaimers")) if _s(d)],
    }


def _build_view(report: Any) -> dict[str, Any]:
    rep = _d(report)
    charts = _ChartBuilder()
    modules, extra_artifacts = _safe("modules", _modules_view, rep, charts, default={}) if rep else ([], [])
    if isinstance(modules, dict):  # the whole module list failed; keep the error visible
        err = modules.get("section_error", "")
        modules, extra_artifacts = [], []
        modules_error = err
    else:
        modules_error = ""
    scored = [m for m in modules if m.get("counted") and m.get("weight")]
    total_w = sum(float(m["weight"]) for m in scored)
    return {
        "header": _safe("header", _header_view, rep, default={"tool": "MalValid", "title": "Production-readiness report"}),
        "verdict": _safe("verdict", _verdict_view, rep, default={"key": "unknown", "tone": "neutral", "icon": "i-unknown",
                                                                  "word": "No verdict", "score_text": "—"}),
        "gate": _safe("gate", _gate_view, rep, modules, default={"hard": []}),
        "callouts": _safe("callouts", _callouts, rep, default={"warnings": []}),
        "modules": modules,
        "modules_error": modules_error,
        "total_weight": total_w,
        "extra_artifacts": extra_artifacts,
        "appendix": _safe("appendix", _appendix_view, rep, modules, default={}),
        "chart_json": charts.json_blob(),
        "pass_line": round(100 * _PASS_LINE),
        "empty": not rep,
    }


# ==================================================================================================
# Rendering
# ==================================================================================================


@lru_cache(maxsize=1)
def _env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
        undefined=jinja2.ChainableUndefined,  # missing keys render as empty, never raise
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(num=fmt_num, value=fmt_value, pct=fmt_pct, bytes=fmt_bytes, duration=fmt_duration,
                       time=fmt_time, op=fmt_op)
    return env


@lru_cache(maxsize=1)
def _assets() -> tuple[str, str, str]:
    """(css, js, CSP hash of js). Assets are ours and trusted; guard against tag breakouts anyway."""
    css = (_STATIC_DIR / "report.css").read_text(encoding="utf-8")
    js = (_STATIC_DIR / "report.js").read_text(encoding="utf-8")
    if "</style" in css.lower() or "</script" in js.lower():  # pragma: no cover - asset bug
        raise RuntimeError("report assets must not contain closing style/script tags")
    digest = base64.b64encode(hashlib.sha256(js.encode("utf-8")).digest()).decode("ascii")
    return css, js, digest


def render_html(report: dict) -> str:
    """Render a ``malvalid-report/1`` dict as one self-contained HTML document.

    Never fetches anything; tolerates missing keys (sections degrade to visible placeholders).
    """
    view = _build_view(report)
    css, js, js_hash = _assets()
    tpl = _env().get_template(_TEMPLATE)
    return tpl.render(v=view, css=Markup(css), js=Markup(js), js_hash=js_hash)


def write_html(report: dict, path: str | Path) -> Path:
    """Render ``report`` and write it to ``path`` (UTF-8). Parent directories are created."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(report)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(p)
    log.info("wrote HTML report %s (%.0f KiB)", p, len(html.encode("utf-8")) / 1024)
    return p
