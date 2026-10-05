"""Display filters for the malvalid local web UI (``malvalid serve``).

The backend registers every entry of :data:`FILTERS` on its Jinja environment (autoescape on,
``StrictUndefined`` off). Two kinds of filters live here:

* **formatting** — ``fmt_score``, ``fmt_pct``, ``fmt_time``, ``fmt_duration``, ``verdict_class``,
  ``status_class``, ``verdict_label`` (the names the web contract pins) plus a few more
  (``fmt_num``, ``status_label``, ``verdict_icon``, ...);
* **views** — ``report_view``, ``progress_view``, ``job_view``, ... turn the loosely typed dicts a page
  receives (``report.json``, ``progress.json``, ``job.json``; any field may be missing, None or
  malformed) into flat dicts whose keys always exist, so templates never chain attribute lookups on
  data that may be absent.

Every filter accepts anything (None, :class:`jinja2.Undefined`, wrong types) and never raises —
except :func:`autoescape_guard`, which fails closed when the environment does not autoescape.
Filters return plain ``str``, so autoescaping applies to their output; the one exception,
``break_dots``, escapes its input itself and only adds ``<wbr>`` tags.
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import math
import re
import shlex
import signal as _signal
from collections.abc import Callable, Mapping
from typing import Any

import jinja2
from markupsafe import Markup, escape

__all__ = ["FILTERS"]

# The pass line of an axis score (0..100): a metric exactly on its threshold scores 75.
try:  # pragma: no cover - import guard only
    from malvalid.scoring import PASS_LINE as _PASS_FRAC
except Exception:  # pragma: no cover
    _PASS_FRAC = 0.75
PASS_LINE = round(100 * float(_PASS_FRAC), 1)

try:  # pragma: no cover - import guard only
    from malvalid.verdict import UNSCORED_MODULES as _UNSCORED
except Exception:  # pragma: no cover
    _UNSCORED = frozenset({"file_safety", "dummy"})

DASH = "—"


# ==================================================================================================
# Coercion helpers
# ==================================================================================================


def _v(x: Any) -> Any:
    """``x`` with jinja2's Undefined mapped to None."""
    return None if isinstance(x, jinja2.Undefined) else x


def _d(x: Any) -> dict[str, Any]:
    x = _v(x)
    return dict(x) if isinstance(x, Mapping) else {}


def _l(x: Any) -> list[Any]:
    x = _v(x)
    return list(x) if isinstance(x, (list, tuple)) else []


def _s(x: Any, default: str = "") -> str:
    x = _v(x)
    if x is None:
        return default
    if isinstance(x, str):
        return x
    value = getattr(x, "value", None)  # str enums (Verdict, Status, ...)
    if isinstance(value, str):
        return value
    return str(x)


def _key(x: Any) -> str:
    """Normalized vocabulary key: ``"Not ready"`` / ``"NOT-READY"`` -> ``"not_ready"``."""
    return re.sub(r"[\s-]+", "_", _s(x).strip().lower())


def _num(x: Any) -> float | None:
    """A finite float, or None (bools and non-numbers are None)."""
    x = _v(x)
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, str):
        try:
            f = float(x.strip())
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _int(x: Any) -> int | None:
    f = _num(x)
    return None if f is None or not f.is_integer() else int(f)


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def truthy(x: Any) -> bool:
    """Form-style truthiness: True, 1, "on", "true", "yes", "1" (case-insensitive)."""
    x = _v(x)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in ("1", "on", "true", "yes", "y", "checked")
    return False


# ==================================================================================================
# Formatting
# ==================================================================================================


def fmt_score(v: Any, digits: int = 1) -> str:
    """0-100 score for display: ``72.4 -> "72.4"``, ``None -> "—"``."""
    x = _num(v)
    if x is None:
        return DASH
    s = f"{x:.{max(int(digits), 0)}f}"
    return "0" + s[2:] if s.startswith("-0") and float(s) == 0 else s


def fmt_pct(v: Any, digits: int = 2) -> str:
    """Fraction as a percentage, trailing zeros trimmed: ``0.0123 -> "1.23%"``, ``1 -> "100%"``."""
    x = _num(v)
    if x is None:
        return DASH
    s = f"{x * 100:.{max(int(digits), 0)}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s + "%"


def fmt_num(v: Any, sig: int = 4) -> str:
    """Compact, precise display of a metric value (mirrors the HTML report)."""
    v = _v(v)
    if v is None:
        return DASH
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if not math.isfinite(v):
            return DASH
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1e15 or a < 1e-3:
            return f"{v:.2e}"
        if v.is_integer() and a >= 1:
            return f"{v:,.0f}"
        if a >= 1000:
            return f"{v:,.1f}"
        return f"{v:.{max(int(sig), 1)}g}"
    return _s(v)


def fmt_value(v: Any, max_chars: int = 160) -> str:
    """Any JSON value on one line (lists/dicts compacted and truncated); None -> "—"."""
    v = _v(v)
    if v is None or isinstance(v, (bool, int, float)):
        return fmt_num(v)
    if isinstance(v, str):
        return v if len(v) <= max_chars else v[: max_chars - 1] + "…"
    try:
        s = json.dumps(v, ensure_ascii=False, default=str, separators=(", ", ": "))
    except (TypeError, ValueError):
        s = str(v)
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


def fmt_bytes(v: Any) -> str:
    x = _num(v)
    if x is None:
        return DASH
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024 or unit == "TiB":
            return f"{x:,.0f} {unit}" if unit == "B" else f"{x:,.1f} {unit}"
        x /= 1024
    return DASH  # pragma: no cover


def fmt_duration(v: Any) -> str:
    """Seconds for display: ``192 -> "3m 12s"``, ``0.8 -> "0.8s"``, ``None -> "—"``."""
    x = _num(v)
    if x is None or x < 0:
        return DASH
    if x < 10:
        return f"{x:.1f}s"
    t = int(round(x))
    if t < 60:
        return f"{t}s"
    if t < 3600:
        return f"{t // 60}m {t % 60}s"
    if t < 86400:
        return f"{t // 3600}h {(t % 3600) // 60}m"
    return f"{t // 86400}d {(t % 86400) // 3600}h"


def _parse_time(v: Any) -> dt.datetime | None:
    """ISO string / datetime / date / epoch seconds -> aware datetime (naive values are UTC:
    malvalid writes UTC everywhere). None if unparseable."""
    v = _v(v)
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, dt.datetime):
        t = v
    elif isinstance(v, dt.date):
        t = dt.datetime(v.year, v.month, v.day)
    elif isinstance(v, (int, float)):
        if not math.isfinite(float(v)):
            return None
        try:
            return dt.datetime.fromtimestamp(float(v), tz=dt.timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            t = dt.datetime.fromisoformat(s[:-1] + "+00:00" if s[-1:] in ("Z", "z") else s)
        except ValueError:
            return None
    else:
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=dt.timezone.utc)


def fmt_time(v: Any, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Timestamp in the server's local time zone: ISO -> ``"2026-09-29 14:03"``.

    An unparseable value is returned as given (it is escaped like any other string); None -> "—".
    """
    t = _parse_time(v)
    if t is None:
        s = _s(v).strip()
        return s if s else DASH
    try:
        return t.astimezone().strftime(fmt)
    except (OverflowError, OSError, ValueError):  # pragma: no cover - exotic platforms
        return t.strftime(fmt)


def fmt_iso(v: Any) -> str:
    """Machine-readable UTC timestamp for ``<time datetime=…>`` ("" if unparseable)."""
    t = _parse_time(v)
    if t is None:
        return ""
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_OPS = {"<=": "≤", ">=": "≥", "<": "<", ">": ">", "==": "=", "!=": "≠"}


def fmt_op(op: Any) -> str:
    return _OPS.get(_s(op).strip(), _s(op))


# ==================================================================================================
# Vocabulary: every verdict / status is shown as color (tone class) + icon + text
# ==================================================================================================

_VERDICTS: dict[str, dict[str, str]] = {
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
_NO_VERDICT = {
    "tone": "neutral", "icon": "i-unknown", "word": "No verdict",
    "meaning": "This run has no production-readiness verdict.",
}


def _verdict_meta(v: Any) -> dict[str, str]:
    return _VERDICTS.get(_key(v), _NO_VERDICT)


def verdict_class(v: Any) -> str:
    """CSS tone class of a verdict: ``ready -> "tone-good"`` … unknown/None -> ``"tone-neutral"``."""
    return "tone-" + _verdict_meta(v)["tone"]


def verdict_label(v: Any) -> str:
    """Human label of a verdict: ``not_ready -> "Not ready"``, None -> ``"No verdict"``."""
    return _verdict_meta(v)["word"]


def verdict_icon(v: Any) -> str:
    return _verdict_meta(v)["icon"]


def verdict_meaning(v: Any) -> str:
    return _verdict_meta(v)["meaning"]


def is_verdict(v: Any) -> bool:
    """True for one of the four verdicts (``ready | conditional | not_ready | blocked``)."""
    return _key(v) in _VERDICTS


# status -> (tone, label, icon). Module statuses, progress statuses and job statuses share it.
_STATUSES: dict[str, tuple[str, str, str]] = {
    # module results
    "pass": ("good", "Pass", "i-pass"),
    "warn": ("warning", "Warn", "i-warn"),
    "fail": ("critical", "Fail", "i-fail"),
    "error": ("critical", "Error", "i-error"),
    "skipped": ("neutral", "Skipped", "i-skip"),
    # progress
    "pending": ("neutral", "Pending", "i-pending"),
    "running": ("info", "Running", "i-running"),
    # jobs
    "queued": ("neutral", "Queued", "i-clock"),
    "finished": ("neutral", "Finished", "i-check"),
    "failed": ("critical", "Failed", "i-error"),
    "cancelled": ("neutral", "Cancelled", "i-stop"),
    "interrupted": ("warning", "Interrupted", "i-warn"),
    # stages
    "done": ("good", "Done", "i-check"),
}
_ALIASES = {"passed": "pass", "warning": "warn", "failure": "fail", "errored": "error", "skip": "skipped",
            "canceled": "cancelled", "complete": "finished", "completed": "finished"}


def _status_meta(s: Any) -> tuple[str, str, str]:
    k = _key(s)
    k = _ALIASES.get(k, k)
    if k in _STATUSES:
        return _STATUSES[k]
    label = _s(s).strip().replace("_", " ")
    return ("neutral", label[:1].upper() + label[1:] if label else "Unknown", "i-unknown")


def status_class(s: Any) -> str:
    """CSS tone class of a module / job status: ``pass -> "tone-good"``, ``running -> "tone-info"``."""
    return "tone-" + _status_meta(s)[0]


def status_label(s: Any) -> str:
    return _status_meta(s)[1]


def status_icon(s: Any) -> str:
    return _status_meta(s)[2]


def status_key(s: Any) -> str:
    """Normalized status key for CSS hooks (``status-<key>``); unknown values -> ``"unknown"``."""
    k = _key(s)
    k = _ALIASES.get(k, k)
    return k if k in _STATUSES else "unknown"


_OUTCOMES = {
    "passed": ("good", "Gate passed", "i-pass"),
    "failed": ("critical", "Gate failed", "i-fail"),
    "not_evaluated": ("neutral", "Not evaluated", "i-unknown"),
}


def outcome_label(s: Any) -> str:
    return _OUTCOMES.get(_key(s), ("neutral", _s(s) or DASH, "i-unknown"))[1]


def outcome_class(s: Any) -> str:
    return "tone-" + _OUTCOMES.get(_key(s), ("neutral", "", ""))[0]


ACTIVE_STATUSES = ("queued", "running")
TERMINAL_STATUSES = ("finished", "failed", "cancelled", "interrupted")


def is_active(s: Any) -> bool:
    """True while a job is queued or running."""
    return status_key(s) in ACTIVE_STATUSES


# Runner stages (progress.json ``stage``) in order, with the label shown to researchers.
STAGES: tuple[tuple[str, str, str], ...] = (
    ("inspecting", "Inspect adapter", "Inspecting the adapter and its declarations"),
    ("scanning", "Scan files", "Scanning the model files before anything is loaded (M0)"),
    ("loading", "Load model", "Loading the model in the sandbox"),
    ("modules", "Run tests", "Running the test modules"),
    ("verdict", "Verdict", "Computing the verdict and score"),
    ("writing", "Write report", "Writing report.json and report.html"),
)
_STAGE_TEXT = {k: long for k, _short, long in STAGES}
_STAGE_TEXT.update(starting="Starting the run", done="Finished", failed="The run stopped with an error",
                   queued="Waiting for a free worker")


def stage_label(stage: Any) -> str:
    k = _key(stage)
    if k in _STAGE_TEXT:
        return _STAGE_TEXT[k]
    return _s(stage).strip() or "Waiting to start"


def stage_steps(stage: Any) -> list[dict[str, str]]:
    """The stepper: ``[{key, label, state}]`` with state ``done | current | todo | failed``."""
    k = _key(stage)
    order = [s[0] for s in STAGES]
    if k == "done":
        cur = len(order)
    elif k in order:
        cur = order.index(k)
    else:  # starting, queued, failed, unknown
        cur = -1
    steps = []
    for i, (key, short, _long) in enumerate(STAGES):
        if k == "failed":
            state = "todo"
        elif i < cur:
            state = "done"
        elif i == cur:
            state = "current"
        else:
            state = "todo"
        steps.append({"key": key, "label": short, "state": state})
    return steps


# ==================================================================================================
# Small helpers for templates
# ==================================================================================================


def pct_step(v: Any) -> int:
    """A 0..100 value rounded and clamped to an int, for the ``w-N`` / ``l-N`` width classes."""
    x = _num(v)
    return 0 if x is None else int(round(_clamp(x, 0.0, 100.0)))


def frac_step(v: Any) -> int:
    """A 0..1 fraction as an int 0..100 for the ``w-N`` / ``l-N`` classes."""
    x = _num(v)
    return 0 if x is None else pct_step(x * 100.0)


def axis_pct(v: Any) -> float | None:
    """An axis score on the 0..100 scale. Values in [0, 1] are read as fractions (ModuleResult.score)."""
    x = _num(v)
    if x is None:
        return None
    if 0.0 <= x <= 1.0:
        x *= 100.0
    return round(_clamp(x, 0.0, 100.0), 1)


def shell_join(argv: Any) -> str:
    """A command line from an argv list (shell-quoted); a string is returned unchanged."""
    argv = _v(argv)
    if isinstance(argv, str):
        return argv
    items = [_s(a) for a in _l(argv) if _v(a) is not None]
    return shlex.join(items) if items else ""


def shell_quote(v: Any) -> str:
    """One shell-quoted word (``"my runs/x" -> "'my runs/x'"``)."""
    return shlex.quote(_s(v))


def cli_command(argv: Any) -> str:
    """``[python, -m, malvalid, run, …]`` -> ``"malvalid run …"`` (what a researcher would type)."""
    argv = _v(argv)
    if isinstance(argv, str):
        return argv
    raw = [_s(a) for a in _l(argv) if _v(a) is not None]
    items: list[str] = []
    skip_next = False
    for a in raw:  # --run-id is internal to `malvalid serve`: a terminal re-run gets its own id
        if skip_next:
            skip_next = False
        elif a == "--run-id":
            skip_next = True
        elif not a.startswith("--run-id="):
            items.append(a)
    for i in range(len(items) - 1):
        if items[i] == "-m" and items[i + 1] == "malvalid":
            return shlex.join(["malvalid", *items[i + 2:]])
    if items and re.search(r"(^|/)malvalid$", items[0]):
        return shlex.join(["malvalid", *items[1:]])
    return shlex.join(items) if items else ""


def break_dots(v: Any) -> Markup:
    """Escaped text with optional line breaks after dots and slashes (long config keys, paths)."""
    return Markup(re.sub(r"([./])", r"\1<wbr>", str(escape(_s(v)))))


def basename(v: Any) -> str:
    s = _s(v).rstrip("/\\")
    return re.split(r"[/\\]", s)[-1] if s else ""


def run_name(run: Any) -> str:
    """The name a run is shown under: its title, else display name, class, adapter file or id."""
    r = _d(run)
    for k in ("title", "display_name", "class_name"):
        s = _s(r.get(k)).strip()
        if s:
            return s
    b = basename(r.get("adapter"))
    if b:
        return b
    return _s(r.get("run_id")).strip() or "Untitled run"


def model_line(run: Any) -> str:
    """``"lightgbm · ember_v2 · threshold 0.8"`` from a RunSummary (missing parts omitted)."""
    r = _d(run)
    parts = [_s(r.get("model_kind")).strip(), _s(r.get("feature_version")).strip()]
    thr = _num(r.get("operating_threshold"))
    if thr is not None:
        parts.append(f"threshold {fmt_num(thr, 6)}")
    return " · ".join(p for p in parts if p)


_COUNT_ORDER = (("pass", "n_pass", "passed"), ("warn", "n_warn", "warned"), ("fail", "n_fail", "failed"),
                ("error", "n_error", "errored"), ("skipped", "n_skipped", "skipped"))


def module_counts(run: Any) -> list[dict[str, Any]]:
    """Non-zero per-status module counts of a RunSummary, in a fixed order, for compact display."""
    r = _d(run)
    out = []
    for status, field, verb in _COUNT_ORDER:
        n = _int(r.get(field))
        if n:
            tone, label, icon = _status_meta(status)
            out.append({"status": status, "n": n, "tone": tone, "icon": icon, "label": label,
                        "text": f"{n} {verb}"})
    return out


_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "feature_space": ("Canonical corpus",
                      "A canonical corpus in your model's feature_version must be installed (see Corpora)."),
    "training_hashes": ("Training manifest",
                        "Declare training_hashes_path on your adapter: one training-sample sha256 per line "
                        "(or a CSV/TSV with a sha256 column)."),
    "training_cutoff": ("Training cutoff",
                        "Declare training_cutoff on your adapter (YYYY-MM-DD, YYYY-MM or YYYY): the date of "
                        "your newest training sample."),
    "tree_access": ("Tree access",
                    "Expose native_model (LightGBM / XGBoost / scikit-learn) or a tree_ensemble() method."),
    "query_only": ("Model loads", "Only needs the model to load; nothing else to declare."),
    "featurize": ("Raw-bytes featurizer", "Add featurize(raw: bytes) to your adapter."),
    "sample_dir": ("Sample directory", "Set sample_dir in gate.yaml (unused in this build)."),
}


def requirement_label(req: Any) -> str:
    k = _key(req)
    return _REQUIREMENTS.get(k, (_s(req) or DASH, ""))[0]


def requirement_hint(req: Any) -> str:
    return _REQUIREMENTS.get(_key(req), ("", ""))[1]


def sandbox_choice(backends: Any) -> str:
    """The backend ``runtime.sandbox_backend: auto`` resolves to (mirrors malvalid.sandbox.host)."""
    b = _d(backends)
    for name in ("bwrap", "unshare"):
        info = _d(b.get(name))
        if truthy(info.get("available")) and truthy(info.get("network_isolated")):
            return name
    return "subprocess" if b else ""


_SECRET = re.compile(r"token|secret|password|passwd|cookie|key$", re.IGNORECASE)


def public_items(d: Any) -> list[tuple[str, Any]]:
    """``d.items()`` sorted by key, dropping anything that looks like a secret (tokens are never shown)."""
    return [(str(k), val) for k, val in sorted(_d(d).items(), key=lambda kv: str(kv[0]))
            if not _SECRET.search(str(k))]


# ==================================================================================================
# Views: normalized dicts for the run page
# ==================================================================================================


def _check_view(c: Mapping[str, Any]) -> dict[str, Any]:
    passed = _v(c.get("passed"))
    score = _num(c.get("score"))
    metric = _s(c.get("metric")) or _s(c.get("name")) or "check"
    thr = c.get("threshold")
    return {
        "metric": metric,
        "op": fmt_op(c.get("op")),
        "threshold": fmt_num(_num(thr)) if _num(thr) is not None else (_s(thr) or DASH),
        "value": fmt_num(_num(c.get("value"))) if _num(c.get("value")) is not None else (_s(c.get("value")) or DASH),
        "passed": passed if isinstance(passed, bool) else None,
        "score": None if score is None else round(100 * _clamp(score, 0.0, 1.0), 1),
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


def _first_line(s: str) -> str:
    lines = [ln.strip() for ln in s.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _verdict_view(rep: Mapping[str, Any]) -> dict[str, Any]:
    v = _d(rep.get("verdict"))
    key = _key(v.get("verdict"))
    meta = _verdict_meta(key)
    bands = _d(v.get("bands"))
    ready_min = _num(bands.get("ready_min"))
    cond_min = _num(bands.get("conditional_min"))
    cap = _num(bands.get("blocked_score_cap"))
    ready_min = 80.0 if ready_min is None else _clamp(ready_min, 0.0, 100.0)
    cond_min = 60.0 if cond_min is None else _clamp(cond_min, 0.0, ready_min)
    cap = 49.0 if cap is None else _clamp(cap, 0.0, 100.0)
    score = _num(v.get("score"))
    raw = _num(v.get("raw_score"))
    capped = truthy(v.get("capped"))
    blockers = [_s(b) for b in _l(v.get("blockers")) if _s(b)]
    reasons = [_s(r) for r in _l(v.get("reasons")) if _s(r) and _s(r) not in blockers]
    return {
        "present": bool(v),
        "key": key if key in _VERDICTS else "",
        "word": meta["word"], "tone": meta["tone"], "icon": meta["icon"], "meaning": meta["meaning"],
        "label": _s(v.get("label")),
        "summary": _s(v.get("summary")),
        "score": score,
        "raw": raw if capped else None,
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
    out: dict[str, dict[str, Any]] = {}
    for a in _l(_d(rep.get("verdict")).get("axes")):
        a = _d(a)
        if _s(a.get("module_id")):
            out[_s(a.get("module_id"))] = a
    return out


def _module_row(m: Mapping[str, Any], idx: int, axes: Mapping[str, Any]) -> dict[str, Any]:
    mid = _s(m.get("module_id"))
    status = status_key(m.get("status"))
    tone, slabel, sicon = _status_meta(m.get("status"))
    checks = [_check_view(_d(c)) for c in _l(m.get("checks")) if isinstance(_v(c), Mapping)]
    axis = _d(axes.get(mid))
    unscored = mid in _UNSCORED
    if axis:
        score = _num(axis.get("score"))
        weight = _num(axis.get("weight"))
        counted = truthy(axis.get("counted"))
    else:
        score = None if status in ("skipped", "error") else axis_pct(m.get("score"))
        weight, counted = None, False
    error = _s(m.get("error"))
    return {
        "id": mid,
        "anchor": "m-" + (re.sub(r"[^A-Za-z0-9_-]+", "-", mid).strip("-") or str(idx)),
        "code": _s(m.get("code")) or "?",
        "title": _s(m.get("title")) or mid or f"Module {idx + 1}",
        "status": status,
        "tone": tone, "status_label": slabel, "icon": sicon,
        "gate": _key(m.get("gate")) or "",
        "outcome": _key(m.get("gate_outcome")),
        "outcome_label": outcome_label(m.get("gate_outcome")) if _s(m.get("gate_outcome")) else "",
        "screening": truthy(m.get("screening")),
        "unscored": unscored,
        "score": None if unscored else score,
        "weight": weight,
        "counted": counted,
        "finding": _s(m.get("finding")),
        "skip_reason": _s(m.get("skip_reason")),
        "error_head": _first_line(error),
        "duration_s": _num(m.get("duration_s")),
        "primary": _primary_check(checks),
        "n_checks": len(checks),
        "checks": checks,
        "corpus_skip": False,  # set by report_view: skipped because the canonical corpus is unusable
    }


def _norm_text(t: str) -> str:
    return " ".join(t.lower().split()).rstrip(".")


def _mentions(text: str, other: str) -> bool:
    """Does ``text`` restate ``other`` (one contains the other, ignoring case and spacing)?"""
    a, b = _norm_text(text or ""), _norm_text(other or "")
    return bool(a and b) and (b in a or a in b)


def _blocked_meaning(gate: Mapping[str, Any], modules: list[dict[str, Any]], aborted: str, load_error: str) -> str:
    """The BLOCKED explanation for this run's actual blocker (not the generic three-way sentence)."""
    tail = " Fix the blocking issues and re-run."
    if aborted:
        return "Not promotable as submitted: M0 model file safety aborted the run before the model was loaded." + tail
    if load_error:
        return "Not promotable as submitted: the model could not be loaded, so nothing was evaluated." + tail
    failed, unevaluated, errored = [], [], []
    for h in _l(gate.get("hard_gates")):
        h = _d(h)
        name = (_s(h.get("code")) + " " + (_s(h.get("title")) or _s(h.get("module_id")))).strip()
        if _key(h.get("status")) == "error":
            errored.append(name)
        elif _key(h.get("gate_outcome")) == "failed":
            failed.append(name)
        elif _key(h.get("gate_outcome")) == "not_evaluated":
            unevaluated.append(name)
    errored += [f"{m['code']} {m['title']}" for m in modules
                if m["status"] == "error" and f"{m['code']} {m['title']}" not in errored]
    parts = []
    if failed:
        parts.append(("a hard gate failed (" if len(failed) == 1 else "hard gates failed (") + ", ".join(failed) + ")")
    if unevaluated:
        parts.append(("a hard gate could not be evaluated (" if len(unevaluated) == 1
                      else "hard gates could not be evaluated (") + ", ".join(unevaluated) + ")")
    if errored:
        parts.append(("a test errored (" if len(errored) == 1 else "tests errored (") + ", ".join(errored) + ")")
    if not parts:
        return _VERDICTS["blocked"]["meaning"]
    return "Not promotable as submitted: " + "; ".join(parts) + "." + tail


def report_view(report: Any) -> dict[str, Any]:
    """Normalize a ``malvalid-report/1`` dict for the run page. Every key always exists."""
    rep = _d(report)
    axes = _axes_by_id(rep)
    modules = [_module_row(_d(m), i, axes) for i, m in enumerate(_l(rep.get("modules")))
               if isinstance(_v(m), Mapping)]
    gate = _d(rep.get("gate"))
    model = _d(rep.get("model"))
    corpus = _d(rep.get("corpus"))
    run = _d(rep.get("run"))
    manifest = _d(rep.get("training_manifest"))
    sandbox = _d(rep.get("sandbox"))
    tool = _d(rep.get("tool"))
    aborted = _v(gate.get("aborted"))
    abort_reason = _s(gate.get("abort_reason"))
    if not abort_reason and aborted and not isinstance(aborted, bool):
        abort_reason = _s(aborted)
    if aborted is True and not abort_reason:
        abort_reason = "no reason recorded"
    hard = []
    for h in _l(gate.get("hard_gates")):
        h = _d(h)
        st = _status_meta(h.get("status"))
        oc = _OUTCOMES.get(_key(h.get("gate_outcome")), ("neutral", "Not evaluated", "i-unknown"))
        critical = st[0] == "critical"
        hard.append({"code": _s(h.get("code")) or "?", "title": _s(h.get("title")) or _s(h.get("module_id")),
                     "tone": "critical" if critical else oc[0], "icon": st[2] if critical else oc[2],
                     "label": oc[1] + (f" · {st[1]}" if st[1] in ("Error", "Skipped") else "")})
    exit_code = _int(gate.get("exit_code"))
    verdict = _verdict_view(rep)
    load_error = "" if abort_reason else (_s(gate.get("load_error")) or _s(model.get("load_error")))
    if verdict["key"] == "blocked":
        verdict["meaning"] = _blocked_meaning(gate, modules, abort_reason, load_error)
    corpus_error = _s(corpus.get("error"))
    warnings = [_s(w) for w in _l(rep.get("warnings")) if _s(w)]
    if corpus_error:
        # The corpus problem is shown once (its own callout); skipped rows point at it instead of
        # repeating it, and run warnings that only restate it are dropped.
        for m in modules:
            if m["status"] == "skipped" and _mentions(m["skip_reason"] or m["finding"], corpus_error):
                m["corpus_skip"] = True
        warnings = [w for w in warnings if not _mentions(w, corpus_error)]
    return {
        "present": bool(rep),
        "title": _s(rep.get("title")),
        "tool_version": _s(tool.get("version")),
        "verdict": verdict,
        "modules": modules,
        "gate": {
            "exit_code": exit_code,
            "exit_meaning": _s(gate.get("exit_meaning")),
            "fail_on": _s(gate.get("fail_on")),
            "aborted": abort_reason,
            "load_error": load_error,
            "hard": hard,
            "disabled": [_s(x) for x in _l(gate.get("disabled")) if _s(x)],
        },
        "model": {
            "class_name": _s(model.get("class_name")),
            "model_kind": _s(model.get("model_kind")),
            "feature_version": _s(model.get("feature_version")),
            "operating_threshold": _num(model.get("operating_threshold")),
            "training_cutoff": _s(model.get("training_cutoff")),
            "training_hashes_path": _s(model.get("training_hashes_path")),
            "adapter_path": _s(model.get("adapter_path")),
            "adapter_sha256": _s(model.get("adapter_sha256")),
            "model_paths": [_s(p) for p in _l(model.get("model_paths")) if _s(p)],
            **_threshold_source(model),
        },
        "corpus": {
            "name": _s(corpus.get("name")),
            "version": _s(corpus.get("version")),
            "feature_version": _s(corpus.get("feature_version")),
            "n": _int(corpus.get("n")),
            "synthetic": truthy(corpus.get("synthetic")),
            "error": _s(corpus.get("error")),
            "content_hash": _s(corpus.get("content_hash")),
            "excluded_training_members": _int(corpus.get("excluded_training_members")),
        },
        "manifest": {
            "declared": truthy(manifest.get("declared")),
            "n_hashes": _int(manifest.get("n_hashes")),
            "n_in_corpus": _int(manifest.get("n_in_corpus")),
            "cutoff": _s(manifest.get("cutoff")),
        },
        "run": {
            "id": _s(run.get("id")),
            "started_at": _s(run.get("started_at")),
            "finished_at": _s(run.get("finished_at")),
            "duration_s": _num(run.get("duration_s")),
            "command": _s(run.get("command")),
            "seed": _int(run.get("seed")),
        },
        "sandbox": {
            "present": bool(sandbox),
            "enabled": truthy(sandbox.get("enabled")),
            "backend": _s(sandbox.get("backend")),
            "network_isolated": truthy(sandbox.get("network_isolated")),
            "detail": _s(sandbox.get("detail")),
            "isolation": _s(sandbox.get("isolation")) or None,
        },
        "warnings": warnings,
    }


_DONE = ("pass", "warn", "fail", "skipped", "error")


def progress_view(progress: Any) -> dict[str, Any]:
    """Normalize a ``malvalid-progress/1`` dict for the live run page. Every key always exists."""
    p = _d(progress)
    mods = []
    for i, m in enumerate(_l(p.get("modules"))):
        m = _d(m)
        if not m:
            continue
        tone, label, icon = _status_meta(m.get("status") or "pending")
        mods.append({
            "id": _s(m.get("id")),
            "code": _s(m.get("code")) or "?",
            "title": _s(m.get("title")) or _s(m.get("id")) or f"Module {i + 1}",
            "status": status_key(m.get("status") or "pending"),
            "tone": tone, "status_label": label, "icon": icon,
            "duration_s": _num(m.get("duration_s")),
        })
    cur = _d(p.get("module"))
    n_done = sum(1 for m in mods if m["status"] in _DONE)
    verdict = _d(p.get("verdict"))
    stage = _key(p.get("stage"))
    return {
        "present": bool(p),
        "stage": stage,
        "stage_label": stage_label(stage) if stage else "",
        "steps": stage_steps(stage),
        "message": _s(p.get("message")),
        "module": {"id": _s(cur.get("id")), "code": _s(cur.get("code")), "title": _s(cur.get("title"))}
        if cur else None,
        "modules": mods,
        "n_total": len(mods),
        "n_done": n_done,
        "done_step": pct_step(100.0 * n_done / len(mods)) if mods else 0,
        "started_at": _s(p.get("started_at")),
        "updated_at": _s(p.get("updated_at")),
        "verdict": {
            "verdict": _key(verdict.get("verdict")),
            "label": _s(verdict.get("label")),
            "score": _num(verdict.get("score")),
            "coverage": _num(verdict.get("coverage")),
        } if verdict else None,
    }


# Job options shown on the run page, in this order, with researcher-facing labels.
_OPTION_LABELS: tuple[tuple[str, str], ...] = (
    ("title", "Run title"),
    ("class_name", "Adapter class"),
    ("model_kind", "Model kind"),
    ("feature_version", "Feature version"),
    ("threshold_mode", "Threshold choice"),
    ("training_cutoff", "Training cutoff"),
    ("training_hashes", "Training hash list"),
    ("corpus", "Corpus"),
    ("corpus_dir", "Corpus directory"),
    ("seed", "Seed"),
    ("only", "Only modules"),
    ("skip", "Skipped modules"),
    ("fail_on", "Fail on"),
    ("allow_pickle", "Allow pickle"),
    ("no_sandbox", "Sandbox disabled"),
    ("config", "Gate policy"),
    ("config_source", "Policy from"),
    ("files", "Uploaded files"),
    ("config_path", "Gate policy"),
    ("adapter_path", "Adapter path"),
)
_OPTION_HIDDEN = {"csrf", "confirm_no_sandbox", "mode", "submission_kind", "threshold", "calibrate_fpr",
                  "model_submission"}
# job.json ``options.config_source`` (written by web/forms.py) -> what the run page says.
_CONFIG_SOURCES = {
    "default": "the packaged default policy",
    "server": "the server's default (malvalid serve --config)",
    "upload": "an uploaded file",
    "path": "a file on this machine",
}


def _uploaded_names(files: Any) -> str:
    """``options.files`` ({adapter, models: [...], manifest, config}) as a comma-separated name list."""
    f = _d(files)
    names: list[str] = []
    for key in ("model", "adapter", "models", "manifest", "config"):
        v = _v(f.get(key))
        for n in (v if isinstance(v, (list, tuple)) else [v]):
            if _s(n):
                names.append(_s(n))
    return ", ".join(names)


def _threshold_choice(opts: dict[str, Any]) -> str:
    """``options.threshold_mode`` (+ ``threshold`` / ``calibrate_fpr``) of a model-file run, in words."""
    mode = _s(opts.get("threshold_mode"))
    if mode == "declared":
        t = _num(opts.get("threshold"))
        return f"{fmt_num(t, 6)} (declared)" if t is not None else "declared"
    if mode == "calibrate":
        f = _num(opts.get("calibrate_fpr"))
        return ("auto-calibrated" + (f" to {fmt_pct(f, 1)} FPR" if f is not None else "")
                + " on a held-out slice of the corpus")
    return ""


def _threshold_source(model: dict[str, Any]) -> dict[str, Any]:
    """Where the operating threshold came from (``report.model.extras``): ``threshold_source`` is
    "calibrated" when malvalid computed it on a held-out slice, and ``calibration`` then holds
    ``target_fpr, achieved_fpr, n_benign, rule, corpus``."""
    extras = _d(model.get("extras")) or _d(_d(model.get("declarations")).get("extras"))
    source = _s(extras.get("threshold_source"))
    cal = _d(extras.get("threshold_calibration"))
    calibration = None
    if source == "calibrated" and cal:
        calibration = {
            "target_fpr": _num(cal.get("target_fpr")),
            "achieved_fpr": _num(cal.get("achieved_fpr")),
            "n_benign": _int(cal.get("n_benign")),
            "rule": _s(cal.get("rule")),
            "corpus": _s(cal.get("corpus")),
            "policy": _s(cal.get("policy")) or None,
            "period": ([_s(x) for x in cal["period"]] if isinstance(cal.get("period"), list)
                       and len(cal["period"]) == 2 else None),
        }
    return {"threshold_source": source, "calibration": calibration}


def job_view(job: Any) -> dict[str, Any]:
    """Normalize a ``malvalid-job/1`` dict for the run page. Every key always exists."""
    j = _d(job)
    opts = _d(j.get("options"))
    rows = []
    seen = set()
    for key, label in _OPTION_LABELS:
        if key in opts:
            seen.add(key)
            val = _v(opts[key])
            if val in (None, "", [], ()):
                continue
            if key in ("allow_pickle", "no_sandbox"):
                if not truthy(val):
                    continue
                val = "yes"
            elif key == "config_source":
                val = _CONFIG_SOURCES.get(_s(val), _s(val))
            elif key == "files":
                val = _uploaded_names(val)
                if not val:
                    continue
            elif key == "threshold_mode":
                val = _threshold_choice(opts)
                if not val:
                    continue
            rows.append({"key": key, "label": label, "value": fmt_value(val, 400)})
    for key, val in public_items(opts):
        if key in seen or key in _OPTION_HIDDEN or _v(val) in (None, "", [], ()):
            continue
        rows.append({"key": key, "label": key.replace("_", " ").capitalize(), "value": fmt_value(val, 400)})
    return {
        "present": bool(j),
        "status": status_key(j.get("status")) if _s(j.get("status")) else "",
        "mode": _key(j.get("mode")),
        "adapter": _s(j.get("adapter")),
        "display_name": _s(j.get("display_name")),
        "submission_id": _s(j.get("submission_id")),
        "created_at": _s(j.get("created_at")),
        "started_at": _s(j.get("started_at")),
        "finished_at": _s(j.get("finished_at")),
        "exit_code": _int(j.get("exit_code")),
        "error": _s(j.get("error")),
        "command": cli_command(j.get("argv")),
        "options": rows,
        "no_sandbox": truthy(opts.get("no_sandbox")),
        "allow_pickle": truthy(opts.get("allow_pickle")),
        "model_file": _s(opts.get("submission_kind")) == "model" or _key(j.get("mode")) == "model",
    }


def exit_signal(code: Any) -> str:
    """A negative subprocess return code as a signal name (-11 -> "SIGSEGV"); "" otherwise."""
    c = _int(code)
    if c is None or c >= 0:
        return ""
    try:
        return _signal.Signals(-c).name
    except ValueError:
        return f"signal {-c}"


_TRACEBACK = "Traceback (most recent call last)"


def error_parts(text: Any) -> dict[str, Any]:
    """Split a flattened error message for display: ``head`` (the first sentence), ``items`` (the
    ``" - "`` bullet list of an adapter-contract error) and ``traceback`` (from "Traceback ..." on)."""
    t = _s(text).strip()
    tb = ""
    i = t.find(_TRACEBACK)
    if i >= 0:
        t, tb = t[:i].rstrip(), t[i:]
    items: list[str] = []
    m = re.search(r":\s+-\s+", t)
    if m:
        head = t[: m.start() + 1]
        items = [x.strip() for x in re.split(r"\s+-\s+", t[m.end():]) if x.strip()]
    else:
        head = t
    return {"head": head, "items": items, "traceback": tb}


@functools.lru_cache(maxsize=64)
def _corpus_synthetic(name: str) -> bool:
    try:
        from malvalid import registry

        cls = registry.corpora().get(name)
        return bool(cls is not None and dict(cls().info()).get("synthetic"))
    except Exception:  # pragma: no cover - a broken provider is simply "not known synthetic"
        return False


def synthetic_corpus(name: Any) -> bool:
    """Is ``name`` a registered synthetic corpus? (its scores are not evidence about real data)"""
    n = _s(name).strip()
    return bool(n) and _corpus_synthetic(n)


@functools.lru_cache(maxsize=1)
def _module_labels() -> dict[str, str]:
    try:
        from malvalid import registry

        return {mid: f"{getattr(cls, 'code', '?')} {getattr(cls, 'title', mid)}"
                for mid, cls in registry.modules().items()}
    except Exception:  # pragma: no cover
        return {}


def module_label(module_id: Any) -> str:
    """``"M6 Model-extraction susceptibility"`` for a module id (the id itself if unknown)."""
    mid = _s(module_id)
    return _module_labels().get(mid, mid)


# ==================================================================================================
# Safety
# ==================================================================================================


@jinja2.pass_eval_context
def autoescape_guard(eval_ctx: Any, value: Any = "") -> str:
    """Fail closed if the template is rendered without autoescaping (``base.html.j2`` calls it).

    ``jinja2.select_autoescape()`` with its defaults does *not* match ``*.html.j2`` names, which would
    render every run title, adapter path and module finding unescaped.
    """
    if not getattr(eval_ctx, "autoescape", False):
        raise RuntimeError(
            "malvalid web templates must be rendered with autoescaping on "
            "(jinja2.Environment(autoescape=True); select_autoescape() does not match .html.j2 names)"
        )
    return ""


FILTERS: dict[str, Callable[..., Any]] = {
    # formatting (names pinned by the web contract)
    "fmt_score": fmt_score,
    "fmt_pct": fmt_pct,
    "fmt_time": fmt_time,
    "fmt_duration": fmt_duration,
    "verdict_class": verdict_class,
    "status_class": status_class,
    "verdict_label": verdict_label,
    # more formatting
    "fmt_num": fmt_num,
    "fmt_value": fmt_value,
    "fmt_bytes": fmt_bytes,
    "fmt_iso": fmt_iso,
    "fmt_op": fmt_op,
    "verdict_icon": verdict_icon,
    "verdict_meaning": verdict_meaning,
    "is_verdict": is_verdict,
    "status_label": status_label,
    "status_icon": status_icon,
    "status_key": status_key,
    "outcome_label": outcome_label,
    "outcome_class": outcome_class,
    "stage_label": stage_label,
    "stage_steps": stage_steps,
    "is_active": is_active,
    "truthy": truthy,
    "pct_step": pct_step,
    "frac_step": frac_step,
    "axis_pct": axis_pct,
    "shell_join": shell_join,
    "shell_quote": shell_quote,
    "cli_command": cli_command,
    "basename": basename,
    "break_dots": break_dots,
    "run_name": run_name,
    "model_line": model_line,
    "module_counts": module_counts,
    "requirement_label": requirement_label,
    "requirement_hint": requirement_hint,
    "sandbox_choice": sandbox_choice,
    "public_items": public_items,
    "exit_signal": exit_signal,
    "error_parts": error_parts,
    "synthetic_corpus": synthetic_corpus,
    "module_label": module_label,
    # views
    "report_view": report_view,
    "progress_view": progress_view,
    "job_view": job_view,
    # safety
    "autoescape_guard": autoescape_guard,
}
