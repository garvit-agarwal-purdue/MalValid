"""Side-by-side comparison of 2–6 runs (``GET /compare?ids=a,b,…``).

One row per module (scorecard order, union over the runs) with one cell per run: status, axis score
(0–100, as on the report's scorecard), gate outcome and the module's key metric against its
threshold. Key metrics per module (result keys as the modules write them):

* M0 file safety — the first failed check, else ``n_critical``;
* M1 performance — ``fpr`` vs ``max_fpr`` (then ``detection_rate`` vs ``min_detection``);
* M2 drift — ``aut_f1_weighted`` vs ``min_aut_f1``;
* M4 membership inference — worst-case ``advantage`` vs ``max_advantage``;
* M5 backdoor screen — ``n_flagged_rules`` (static tree scan), else ``max_trigger_drop``;
* M6 extraction — surrogate ``fidelity`` at ``fidelity_budget`` queries vs ``max_fidelity``;
* M7 explanation — ``controllable_share`` vs ``max_controllable_share`` (then ``top_feature_share``).

Other (plugin) modules show their first failed check, else their first check. Plus the differences
between the runs' gate policies (``config_diff``) and the best run (verdict first, then score).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

VERDICT_RANK = {"ready": 0, "conditional": 1, "not_ready": 2, "blocked": 3}
UNSCORED = frozenset({"file_safety", "dummy"})
MAX_COMPARE = 6
MIN_COMPARE = 2

#: module id -> ((metric, threshold param), ...) — the first with a value is the cell's key metric.
KEY_METRICS: dict[str, tuple[tuple[str, str | None], ...]] = {
    "file_safety": (("n_critical", None),),
    "performance": (("fpr", "max_fpr"), ("detection_rate", "min_detection")),
    "drift": (("aut_f1_weighted", "min_aut_f1"),),
    "membership_inf": (("advantage", "max_advantage"),),
    "backdoor_screen": (("n_flagged_rules", None), ("max_trigger_drop", "max_trigger_drop")),
    "extraction": (("fidelity", "max_fidelity"),),
    "explanation": (("controllable_share", "max_controllable_share"), ("top_feature_share", "max_top_feature_share")),
}
#: Comparison used when a metric is read from ``metrics`` (no matching check in the result).
DEFAULT_OPS = {
    "fpr": "<=", "detection_rate": ">=", "aut_f1_weighted": ">=", "advantage": "<=", "n_flagged_rules": "<=",
    "max_trigger_drop": "<=", "fidelity": "<=", "controllable_share": "<=", "top_feature_share": "<=",
    "n_critical": "<=",
}
#: Modules whose first failed check replaces the key metric (the failure is what matters).
FAILED_CHECK_FIRST = frozenset({"file_safety"})
#: Config keys that are bookkeeping, not policy.
IGNORED_CONFIG_KEYS = frozenset({"source_path"})


def _d(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _l(v: Any) -> list[Any]:
    return v if isinstance(v, list) else []


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _code_key(code: Any, module_id: str) -> tuple[Any, ...]:
    """Natural sort: M0 < M1 < … < M10 < P1."""
    parts = re.findall(r"\d+|\D+", str(code or "~"))
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts) + ((2, 0, module_id),)


def _check_entry(c: dict[str, Any]) -> dict[str, Any]:
    passed = c.get("passed")
    return {
        "metric": str(c.get("metric") or c.get("name") or ""),
        "value": _num(c.get("value")),
        "threshold": _num(c.get("threshold")),
        "op": c.get("op") if isinstance(c.get("op"), str) else None,
        "passed": passed if isinstance(passed, bool) else None,
    }


def _metric_entry(mod: dict[str, Any], metric: str, param: str | None) -> dict[str, Any] | None:
    checks = [c for c in _l(mod.get("checks")) if isinstance(c, dict)]
    c = next((c for c in checks if c.get("metric") == metric), None) or next(
        (c for c in checks if c.get("name") == metric), None)
    if c is not None:
        e = _check_entry(c)
        e["metric"] = metric
        return e
    metrics = _d(mod.get("metrics"))
    if metric in metrics:
        return {
            "metric": metric,
            "value": _num(metrics.get(metric)),
            "threshold": _num(_d(mod.get("params")).get(param)) if param else None,
            "op": DEFAULT_OPS.get(metric),
            "passed": None,
        }
    return None


def key_metrics(mod: dict[str, Any]) -> list[dict[str, Any]]:
    """Key metric entries ``{metric, value, threshold, op, passed}`` of one module result (primary first)."""
    mid = str(mod.get("module_id") or "")
    checks = [c for c in _l(mod.get("checks")) if isinstance(c, dict)]
    failed = [c for c in checks if c.get("passed") is False]
    entries: list[dict[str, Any]] = []
    spec = KEY_METRICS.get(mid)
    if spec is None or (mid in FAILED_CHECK_FIRST and failed):
        pick = failed or checks
        if pick:
            entries.append(_check_entry(pick[0]))
    for metric, param in spec or ():
        e = _metric_entry(mod, metric, param)
        if e is not None and all(e["metric"] != x["metric"] for x in entries):
            entries.append(e)
    if spec is not None and len(entries) > 1 and entries[0]["value"] is None:
        with_value = [e for e in entries if e["value"] is not None]
        if with_value:
            entries.remove(with_value[0])
            entries.insert(0, with_value[0])
    if spec is not None and len(entries) > 1 and entries[0]["passed"] is not False:
        # A failed key check leads (M1 failing on detection_rate must not show a passing fpr first).
        failing = next((e for e in entries if e["passed"] is False), None)
        if failing is not None:
            entries.remove(failing)
            entries.insert(0, failing)
    return entries


def _axis_scores(report: dict[str, Any]) -> dict[str, float | None] | None:
    axes = _l(_d(report.get("verdict")).get("axes"))
    if not axes:
        return None
    return {str(a.get("module_id")): _num(a.get("score")) for a in axes if isinstance(a, dict)}


def build_cell(mod: dict[str, Any] | None, axes: dict[str, float | None] | None) -> dict[str, Any]:
    if mod is None:
        return {"status": None, "score": None, "gate_outcome": None, "key_metric": None, "key_value": None,
                "threshold": None, "op": None, "passed": None, "metrics": [], "skip_reason": None,
                "finding": None, "screening": False}
    mid = str(mod.get("module_id") or "")
    if axes is not None:
        score = axes.get(mid)
    elif mid in UNSCORED:
        score = None
    else:
        s = _num(mod.get("score"))
        score = None if s is None else round(100.0 * s, 1)
    entries = key_metrics(mod) if mod.get("status") not in ("skipped",) else []
    primary = entries[0] if entries else {}
    finding = mod.get("finding")
    return {
        "status": mod.get("status") if isinstance(mod.get("status"), str) else None,
        "score": score,
        "gate_outcome": mod.get("gate_outcome") if isinstance(mod.get("gate_outcome"), str) else None,
        "key_metric": primary.get("metric"),
        "key_value": primary.get("value"),
        "threshold": primary.get("threshold"),
        "op": primary.get("op"),
        "passed": primary.get("passed"),
        "metrics": entries,
        "skip_reason": mod.get("skip_reason") if isinstance(mod.get("skip_reason"), str) else None,
        "finding": finding if isinstance(finding, str) else None,
        "screening": bool(mod.get("screening")),
    }


def build_rows(reports: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    """One row per module id present in any report, in scorecard order."""
    meta: dict[str, tuple[Any, Any]] = {}
    by_run: list[dict[str, dict[str, Any]]] = []
    for rep in reports:
        mods: dict[str, dict[str, Any]] = {}
        for m in _l(_d(rep).get("modules")):
            if not isinstance(m, dict) or not m.get("module_id"):
                continue
            mid = str(m["module_id"])
            mods.setdefault(mid, m)
            meta.setdefault(mid, (m.get("code"), m.get("title")))
        by_run.append(mods)
    axes = [_axis_scores(_d(rep)) if rep is not None else None for rep in reports]
    order = sorted(meta, key=lambda mid: _code_key(meta[mid][0], mid))
    rows = []
    for mid in order:
        code, title = meta[mid]
        rows.append({
            "module_id": mid,
            "code": code,
            "title": title,
            "cells": [build_cell(mods.get(mid), ax) if rep is not None else build_cell(None, None)
                      for mods, ax, rep in zip(by_run, axes, reports)],
        })
    return rows


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if not prefix and k in IGNORED_CONFIG_KEYS:
                continue
            if isinstance(v, dict) and v:
                out.update(flatten(v, key))
            else:
                out[key] = v
    return out


def _canon(v: Any) -> str:
    try:
        return json.dumps(v, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(v)


def config_diff(reports: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    """``[{key, values}]`` for every policy key whose value differs between the runs with a report.

    Keys are dotted paths into the report's ``config`` (e.g. ``modules.performance.max_fpr``) plus
    ``corpus.content_hash`` (the exact corpus build) and ``fail_on``.
    """
    flats: list[dict[str, Any] | None] = []
    for rep in reports:
        if rep is None:
            flats.append(None)
            continue
        f = flatten(_d(rep.get("config")))
        f["corpus.content_hash"] = _d(rep.get("corpus")).get("content_hash")
        f["fail_on"] = _d(rep.get("gate")).get("fail_on")
        flats.append(f)
    present = [f for f in flats if f is not None]
    if len(present) < 2:
        return []
    keys: list[str] = []
    for f in present:
        for k in f:
            if k not in keys:
                keys.append(k)
    out = []
    for k in keys:
        vals = [f.get(k) if f is not None else None for f in flats]
        canon = {_canon(f.get(k)) for f in present}
        if len(canon) > 1:
            out.append({"key": k, "values": vals})
    return out


def best_run_id(summaries: Iterable[dict[str, Any]]) -> str | None:
    """The best of the compared runs with a verdict: verdict first, then score, then coverage."""
    best: tuple[Any, ...] | None = None
    best_id: str | None = None
    for s in summaries:
        v = s.get("verdict")
        if v not in VERDICT_RANK:
            continue
        score = _num(s.get("score"))
        cov = _num(s.get("coverage"))
        key = (VERDICT_RANK[v], -(score if score is not None else -1.0), -(cov if cov is not None else -1.0))
        if best is None or key < best:
            best, best_id = key, s.get("run_id")
    return best_id


#: Config-diff keys that change how a run is scored (thresholds, weights, bands, gates, enabled tests).
#: A difference in any of them makes the runs' verdicts and scores incomparable.
POLICY_KEY_PREFIXES = ("modules.", "verdict.", "fail_on")


def policy_differences(diff: Iterable[dict[str, Any]]) -> list[str]:
    """The keys of ``config_diff`` rows that are scoring policy (not corpus, seed or report cosmetics)."""
    return [str(d.get("key")) for d in diff
            if isinstance(d, dict) and str(d.get("key") or "").startswith(POLICY_KEY_PREFIXES)]


def _hard_gate_problem(report: dict[str, Any] | None) -> str | None:
    """Why a BLOCKED run is blocked, in one phrase (from the report's hard gates / abort / load error)."""
    rep = _d(report)
    gate = _d(rep.get("gate"))
    if gate.get("aborted"):
        return "M0 model file safety aborted the run"
    if gate.get("load_error") or _d(rep.get("model")).get("load_error"):
        return "the model could not be loaded"
    failed, unevaluated, errored = [], [], []
    for h in _l(gate.get("hard_gates")):
        h = _d(h)
        name = " ".join(x for x in (str(h.get("code") or ""), str(h.get("title") or h.get("module_id") or "")) if x)
        outcome = str(h.get("gate_outcome") or "")
        if str(h.get("status") or "") == "error":
            errored.append(name)
        elif outcome == "failed":
            failed.append(name)
        elif outcome in ("not_evaluated", "could_not_evaluate", "unevaluated", "skipped"):
            unevaluated.append(name)
    if failed:
        return ", ".join(failed) + " (hard gate) failed"
    if errored:
        return ", ".join(errored) + " errored"
    if unevaluated:
        return ", ".join(unevaluated) + " (hard gate) could not be evaluated"
    return None


def verdict_lines(summaries: list[dict[str, Any]], reports: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    """One plain sentence per run for the "which is safer to ship" summary above the table."""
    out = []
    for sm, rep in zip(summaries, reports):
        v = sm.get("verdict")
        label = {"ready": "Ready", "conditional": "Conditional", "not_ready": "Not ready",
                 "blocked": "Blocked"}.get(str(v), "No verdict")
        score = _num(sm.get("score"))
        text = label + (f" {score:.1f}" if score is not None else "")
        why = None
        if v == "blocked":
            why = _hard_gate_problem(rep)
        elif v in ("conditional", "not_ready"):
            reasons = [str(r) for r in _l(_d(_d(rep).get("verdict")).get("reasons")) if r]
            why = reasons[0] if reasons else None
        if sm.get("status") == "failed":
            why = "the run failed, so its result is not trustworthy" + (f" ({why})" if why else "")
        out.append({"run_id": sm.get("run_id"), "verdict": v, "text": text, "why": why})
    return out


def parse_ids(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        for part in str(v).split(","):
            p = part.strip()
            if p and p not in out:
                out.append(p)
    return out


__all__ = ["KEY_METRICS", "MAX_COMPARE", "MIN_COMPARE", "POLICY_KEY_PREFIXES", "best_run_id", "build_cell",
           "build_rows", "config_diff", "flatten", "key_metrics", "parse_ids", "policy_differences",
           "verdict_lines"]
