"""End-to-end tests: ``python -m malvalid ...`` as a subprocess through the real sandbox.

Each scenario builds its own tiny synthetic_v2 corpus + LightGBM (see conftest.py). No fakes.
"""

from __future__ import annotations

import html as htmllib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from .conftest import REPORT_KEYS, CliResult, load_report, write_adapter, write_config

VERDICTS = ("ready", "conditional", "not_ready", "blocked")
MODULE_ORDER = ["M0", "M1", "M2", "M4", "M5", "M7"]
VOLATILE = re.compile(r"(duration|elapsed|seconds|time|started|finished|_at$|path|dir|^id$|_id$|uuid|log|scratch|location|budget_cap)", re.I)


def _mods(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["module_id"]: m for m in report["modules"]}


def _walk_strings(obj: Any):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


def _timeless(metrics: dict[str, Any]) -> dict[str, Any]:
    """Metrics minus wall-clock readings (scan_seconds, duration_s, ...): everything else must match."""
    return {k: v for k, v in metrics.items() if not re.search(r"(seconds|duration|elapsed|_s$)", k)}


def _scrub(obj: Any, out: Path) -> Any:
    """Drop timings/paths/ids so two runs can be compared for equality."""
    if isinstance(obj, dict):
        return {k: _scrub(v, out) for k, v in obj.items() if not VOLATILE.search(str(k))}
    if isinstance(obj, list):
        return [_scrub(v, out) for v in obj]
    if isinstance(obj, str):
        return obj.replace(str(out), "<OUT>")
    return obj


def _external_refs(page: str) -> list[str]:
    pats = [r"""(?:src|href|action|poster|data)\s*=\s*["']?\s*(?:https?:)?//[^\s"'>]+""",
            r"""url\(\s*["']?\s*(?:https?:)?//[^)]+""",
            r"""@import\s+(?:url\()?\s*["']?\s*(?:https?:)?//[^\s"')]+"""]
    return [m.group(0) for p in pats for m in re.finditer(p, page, re.I)]


# ---------------------------------------------------------------------------------------------
# (1) happy path
# ---------------------------------------------------------------------------------------------
class TestHappyPath:
    def test_exit_code_and_files(self, happy_run):
        res: CliResult = happy_run["res"]
        out: Path = happy_run["out"]
        rep = happy_run["report"]
        assert res.returncode == rep["gate"]["exit_code"], res.output
        assert res.returncode == 0, res.output  # default --fail-on blocked, nothing blocks this run
        for f in ("report.json", "report.html", "run.log"):
            assert (out / f).is_file(), f
        assert "conditional" in res.output.lower() or "ready" in res.output.lower()

    def test_report_has_every_contract_key(self, happy_run):
        rep = happy_run["report"]  # loaded with strict json (no NaN/Infinity)
        missing = [k for k in REPORT_KEYS if k not in rep]
        assert not missing, missing
        assert rep["schema_version"] == "malvalid-report/1"
        assert rep["tool"]["name"] and rep["tool"]["version"]
        assert rep["run"]["seed"] == 7
        # round-trips as strict JSON
        json.dumps(rep, allow_nan=False)
        assert rep["corpus"]["name"] == "synthetic_v2" and rep["corpus"]["error"] is None
        assert rep["corpus"]["synthetic"] is True
        assert rep["schema"]["name"] == "ember_v2" and rep["schema"]["dim"] == 2381
        assert rep["model"]["load_error"] is None
        assert rep["model"]["tree_access"]["available"] is True
        assert rep["model"]["artifacts"][0]["is_pickle"] is False
        assert rep["sandbox"].get("backend")
        assert rep["disclaimers"], "report must carry disclaimers"

    def test_module_order_and_skip_reasons(self, happy_run):
        rep = happy_run["report"]
        assert [m["code"] for m in rep["modules"]] == MODULE_ORDER
        assert rep["modules"][0]["module_id"] == "file_safety"
        for m in rep["modules"]:
            assert m["status"] in ("pass", "warn", "fail", "skipped", "error"), m["module_id"]
            if m["status"] == "skipped":
                assert m.get("skip_reason"), f"{m['module_id']} skipped without skip_reason"
            else:
                assert m["finding"], m["module_id"]
        # every module that ran on this fully-equipped run produced real evidence
        for mid in ("performance", "drift", "membership_inf", "explanation"):
            m = _mods(rep)[mid]
            assert m["status"] != "skipped", (mid, m.get("skip_reason"))
            assert m["metrics"], mid

    def test_verdict_consistent_with_modules(self, happy_run):
        rep = happy_run["report"]
        v, gate, mods = rep["verdict"], rep["gate"], _mods(rep)
        bands = v["bands"]
        assert v["verdict"] in VERDICTS and v["label"] and v["summary"]
        assert 0.0 <= v["score"] <= 100.0 and 0.0 <= v["coverage"] <= 1.0
        # nothing failed, so nothing blocks
        assert v["blockers"] == [] and v["verdict"] != "blocked" and v["capped"] is False
        assert gate["failed_hard"] == [] and gate["errored"] == [] and gate["aborted"] in (False, None)
        assert gate["not_evaluated_hard"] == []
        assert all(h["gate_outcome"] == "passed" for h in gate["hard_gates"])
        assert not [m for m in mods.values() if m["status"] in ("fail", "error") and m["gate"] == "hard"]
        # score is the weighted mean of the counted axes (which are the module scores x 100)
        counted = [a for a in v["axes"] if a["counted"]]
        assert counted
        for a in counted:
            assert a["score"] == pytest.approx(mods[a["module_id"]]["score"] * 100, abs=0.15)
        wmean = sum(a["weight"] * a["score"] for a in counted) / sum(a["weight"] for a in counted)
        assert v["raw_score"] == pytest.approx(wmean, abs=0.2)
        assert v["score"] == pytest.approx(v["raw_score"], abs=0.2)  # not capped
        # every module that ran is one axis; all of them ran => full coverage
        ran = [m for m in rep["modules"] if m["status"] != "skipped" and m["module_id"] != "file_safety"]
        assert len(counted) == len(ran)
        if not [m for m in rep["modules"] if m["status"] == "skipped"]:
            assert v["coverage"] == pytest.approx(1.0)
        # verdict lies in the band the score says (scores are compared at one decimal)
        s = round(v["score"], 1)
        if v["verdict"] == "ready":
            assert s >= bands["ready_min"] and v["coverage"] >= bands["min_coverage_ready"]
        elif v["verdict"] == "conditional":
            assert bands["conditional_min"] <= s
            assert s < bands["ready_min"] or v["coverage"] < bands["min_coverage_ready"] or v["reasons"]
        else:
            assert s < bands["conditional_min"]
        assert v["reasons"] or v["verdict"] == "ready"

    def test_m1_metrics_match_gate(self, happy_run):
        m1 = _mods(happy_run["report"])["performance"]
        met = m1["metrics"]
        assert met["n_benign"] > 0 and met["n_malicious"] > 0
        assert 0 <= met["fpr"] <= 1 and 0 <= met["detection_rate"] <= 1
        checks = {c["name"]: c for c in m1["checks"]}
        assert checks["fpr"]["value"] == pytest.approx(met["fpr"])
        assert checks["fpr"]["passed"] is True or checks["fpr"].get("outcome") in ("pass", "passed", True) or m1["status"] == "pass"
        assert m1["gate"] == "hard" and m1["status"] == "pass"

    def test_report_never_references_private_dir(self, happy_run):
        out: Path = happy_run["out"]
        rep = happy_run["report"]
        priv = {str(out / "private"), str((out / "private").resolve())}
        for s in _walk_strings(rep):
            assert not any(p in s for p in priv), s[:200]
        text = (out / "report.json").read_text()
        assert str(out / "private") not in text
        html_text = (out / "report.html").read_text()
        assert not any(p in html_text for p in priv)

    def test_html_report_self_contained(self, happy_run):
        out: Path = happy_run["out"]
        rep = happy_run["report"]
        page = (out / "report.html").read_text()
        assert page.lstrip().lower().startswith("<!doctype html")
        assert not _external_refs(page), _external_refs(page)[:5]
        text = htmllib.unescape(page)
        assert rep["verdict"]["label"] in text
        assert f"{rep['verdict']['score']:.1f}" in text or str(round(rep["verdict"]["score"])) in text
        for m in rep["modules"]:
            assert m["title"] in text, m["title"]


# ---------------------------------------------------------------------------------------------
# skipped modules carry a reason (adapter without a training manifest)
# ---------------------------------------------------------------------------------------------
def test_modules_without_inputs_are_skipped_with_reason(run_cli, trained, gate_config, e2e_root):
    adapter = write_adapter(e2e_root / "sub_nohash", trained, with_hashes=False)
    out = e2e_root / "run_nohash"
    res = run_cli("run", "--adapter", adapter, "--config", gate_config, "--out", out)
    assert res.returncode in (0, 1), res.output
    rep = load_report(out)
    m4 = _mods(rep)["membership_inf"]
    assert m4["status"] == "skipped"
    assert "training" in m4["skip_reason"].lower() or "manifest" in m4["skip_reason"].lower()
    assert m4["score"] is None
    v = rep["verdict"]
    assert v["coverage"] < 1.0, "a skipped module must lower coverage"
    assert "membership_inf" not in [a["module_id"] for a in v["axes"] if a["counted"]]
    # the rest still ran and produced a verdict
    assert _mods(rep)["performance"]["status"] in ("pass", "warn", "fail")


# ---------------------------------------------------------------------------------------------
# (2) determinism
# ---------------------------------------------------------------------------------------------
def test_same_seed_gives_identical_module_metrics(run_cli, good_adapter, gate_config, happy_run, e2e_root):
    out2 = e2e_root / "run_repeat"
    res = run_cli("run", "--adapter", good_adapter, "--config", gate_config, "--out", out2, "--seed", "7")
    assert res.returncode == happy_run["res"].returncode, res.output
    a, b = happy_run["report"], load_report(out2)
    assert b["run"]["seed"] == 7
    assert [m["module_id"] for m in a["modules"]] == [m["module_id"] for m in b["modules"]]
    for ma, mb in zip(a["modules"], b["modules"]):
        mid = ma["module_id"]
        assert ma["status"] == mb["status"], mid
        assert ma["score"] == mb["score"], mid
        assert _timeless(ma["metrics"]) == _timeless(mb["metrics"]), mid
        assert _scrub(ma["checks"], happy_run["out"]) == _scrub(mb["checks"], out2), mid
        assert _scrub(ma["details"], happy_run["out"]) == _scrub(mb["details"], out2), mid
        assert _scrub(ma["finding"], happy_run["out"]) == _scrub(mb["finding"], out2), mid
    va, vb = a["verdict"], b["verdict"]
    for k in ("verdict", "score", "raw_score", "coverage", "reasons", "blockers", "axes"):
        assert va[k] == vb[k], k
    assert a["corpus"]["content_hash"] == b["corpus"]["content_hash"]
    assert a["model"]["adapter_sha256"] == b["model"]["adapter_sha256"]


# ---------------------------------------------------------------------------------------------
# (3) high false-positive rate => BLOCKED
# ---------------------------------------------------------------------------------------------
def test_high_fpr_model_is_blocked(run_cli, trained, gate_config, e2e_root):
    adapter = write_adapter(e2e_root / "sub_highfpr", trained, threshold=0.001)
    out = e2e_root / "run_highfpr"
    res = run_cli("run", "--adapter", adapter, "--config", gate_config, "--out", out)
    assert res.returncode == 1, res.output
    rep = load_report(out)
    assert rep["gate"]["exit_code"] == 1
    m1 = _mods(rep)["performance"]
    assert m1["status"] == "fail" and m1["gate"] == "hard"
    assert m1["metrics"]["fpr"] > 0.15, m1["metrics"]
    v = rep["verdict"]
    assert v["verdict"] == "blocked"
    cap = v["bands"]["blocked_score_cap"]
    assert v["score"] <= cap
    assert v["capped"] == (v["raw_score"] > cap)  # the cap only bites when the raw score is above it
    assert v["blockers"], "a failed hard gate must be listed as a blocker"
    assert rep["gate"]["failed_hard"], rep["gate"]
    assert any(h["module_id"] == "performance" and h["gate_outcome"] == "failed" for h in rep["gate"]["hard_gates"])
    assert "blocked" in res.output.lower()
    assert "BLOCKED" in htmllib.unescape((out / "report.html").read_text()).upper()


def test_skipping_a_failing_hard_gate_still_blocks(run_cli, trained, gate_config, e2e_root):
    """``--skip M1`` must not bypass the hard gate the same model fails: a deselected hard gate is an
    unevaluated hard gate (BLOCKED, exit 1), it stays in the coverage denominator, and it is reported
    as deselected rather than "disabled in the config"."""
    adapter = write_adapter(e2e_root / "sub_highfpr_skip", trained, threshold=0.001)
    out = e2e_root / "run_highfpr_skip"
    res = run_cli("run", "--adapter", adapter, "--config", gate_config, "--out", out, "--skip", "M1", "--no-html")
    assert res.returncode == 1, res.output
    rep = load_report(out)
    m1 = _mods(rep)["performance"]
    assert m1["status"] == "skipped" and "--skip" in m1["skip_reason"], m1
    v = rep["verdict"]
    assert v["verdict"] == "blocked", v
    assert v["coverage"] < 1.0
    assert any("hard gate not evaluated" in b for b in v["blockers"]), v["blockers"]
    gate = rep["gate"]
    assert gate["exit_code"] == 1 and gate["passed"] is False
    assert any(h["module_id"] == "performance" and h["gate_outcome"] == "not_evaluated" for h in gate["hard_gates"])
    assert gate["deselected"] == ["performance"] and "performance" not in gate["disabled"]
    assert "deselected" in res.output.lower()


# ---------------------------------------------------------------------------------------------
# (4) pickle artifacts
# ---------------------------------------------------------------------------------------------
JOBLIB_ADAPTER = '''
"""E2E submission: a scikit-learn model stored as a joblib pickle."""
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent


class PickleDetector:
    feature_version = "ember_v2"
    model_kind = "sklearn_gbdt"
    operating_threshold = 0.5
    training_hashes_path = None
    training_cutoff = "2017-12"
    model_path = "model.joblib"

    def __init__(self, est):
        self.native_model = est

    @classmethod
    def load(cls):
        return cls(joblib.load(HERE / cls.model_path))

    def predict_proba(self, X):
        return self.native_model.predict_proba(X)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= self.operating_threshold).astype(np.int8)
'''


@pytest.fixture(scope="module")
def pickle_adapter(e2e_root: Path, corpus) -> Path:
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    idx = np.flatnonzero((corpus.split == "train") & (corpus.label >= 0))
    est = HistGradientBoostingClassifier(max_iter=20, max_leaf_nodes=8, random_state=0)
    est.fit(np.asarray(corpus.X[idx], dtype=np.float32), corpus.label[idx].astype(int))
    d = e2e_root / "sub_pickle"
    d.mkdir()
    joblib.dump(est, d / "model.joblib")
    (d / "pickle_adapter.py").write_text(JOBLIB_ADAPTER)
    return d / "pickle_adapter.py"


def test_pickle_without_allow_pickle_aborts_at_m0(run_cli, pickle_adapter, gate_config, e2e_root):
    out = e2e_root / "run_pickle_refused"
    res = run_cli("run", "--adapter", pickle_adapter, "--config", gate_config, "--out", out)
    assert res.returncode == 1, res.output
    rep = load_report(out)
    mods = rep["modules"]
    assert mods[0]["module_id"] == "file_safety" and mods[0]["status"] == "fail"
    assert mods[0]["details"].get("abort") is True
    assert "pickle" in mods[0]["finding"].lower()
    assert rep["model"]["artifacts"][0]["is_pickle"] is True
    assert rep["gate"]["aborted"] and rep["gate"]["abort_reason"]
    v = rep["verdict"]
    assert v["verdict"] == "blocked" and v["blockers"]
    assert v["score"] is None or v["score"] <= v["bands"]["blocked_score_cap"]  # nothing ran => no score
    others = mods[1:]
    assert others, "the other modules must still be listed"
    for m in others:
        assert m["status"] == "skipped", m["module_id"]
        assert "aborted by M0" in m["skip_reason"], m["skip_reason"]
    # M0 aborted before the model was ever loaded: no queries reached the sandbox
    assert not rep["model"].get("query_count")
    assert rep["model"]["load_error"] is None


def test_pickle_with_allow_pickle_runs_with_warning(run_cli, pickle_adapter, gate_config, e2e_root):
    out = e2e_root / "run_pickle_allowed"
    res = run_cli("run", "--adapter", pickle_adapter, "--config", gate_config, "--out", out, "--allow-pickle")
    assert res.returncode in (0, 1), res.output
    rep = load_report(out)
    m0 = rep["modules"][0]
    assert m0["status"] == "warn", (m0["status"], m0["finding"])
    assert not m0["details"].get("abort")
    assert any("pickle" in n.lower() for n in m0["notes"] + [m0["finding"]])
    assert rep["run"]["allow_pickle"] is True
    m1 = _mods(rep)["performance"]
    assert m1["status"] in ("pass", "warn", "fail"), m1
    assert m1["metrics"]["n_benign"] > 0
    assert rep["model"]["load_error"] is None and rep["model"]["query_count"] > 0
    assert not any("aborted by M0" in (m.get("skip_reason") or "") for m in rep["modules"])


# ---------------------------------------------------------------------------------------------
# (5) --fail-on
# ---------------------------------------------------------------------------------------------
def test_fail_on_conditional_turns_conditional_into_exit_1(run_cli, good_adapter, gate_config, happy_run, e2e_root):
    assert happy_run["report"]["verdict"]["verdict"] == "conditional", "fixture drifted: expected a CONDITIONAL run"
    assert happy_run["res"].returncode == 0
    out = e2e_root / "run_failon_conditional"
    res = run_cli("run", "--adapter", good_adapter, "--config", gate_config, "--out", out,
                  "--seed", "7", "--fail-on", "conditional")
    assert res.returncode == 1, res.output
    rep = load_report(out)
    assert rep["verdict"]["verdict"] == "conditional"
    assert rep["gate"]["fail_on"] == "conditional" and rep["gate"]["exit_code"] == 1
    # a threshold below the verdict does not trip
    res2 = run_cli("run", "--adapter", good_adapter, "--config", gate_config, "--out", e2e_root / "run_failon_np",
                   "--seed", "7", "--fail-on", "not_ready", "--no-html")
    assert res2.returncode == 0, res2.output


def test_fail_on_not_ready_on_not_ready_run(run_cli, trained, e2e_root):
    # Same detector, stricter policy bands: the score no longer reaches CONDITIONAL.
    cfg = write_config(e2e_root / "gate_strict.yaml", verdict={"ready_min": 99, "conditional_min": 95})
    adapter = write_adapter(e2e_root / "sub_strict", trained)
    out = e2e_root / "run_strict"
    res = run_cli("run", "--adapter", adapter, "--config", cfg, "--out", out, "--seed", "7", "--fail-on", "not_ready")
    rep = load_report(out)
    assert rep["verdict"]["verdict"] == "not_ready", rep["verdict"]["summary"]
    assert res.returncode == 1, res.output
    res2 = run_cli("run", "--adapter", adapter, "--config", cfg, "--out", e2e_root / "run_strict2", "--seed", "7",
                   "--no-html")
    assert res2.returncode == 0, res2.output  # default --fail-on blocked


# ---------------------------------------------------------------------------------------------
# (6) validate-adapter
# ---------------------------------------------------------------------------------------------
BROKEN_ADAPTER = '''
"""E2E submission that violates the predict_proba contract by returning (n, 2)."""
from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent


class BrokenDetector:
    feature_version = "ember_v2"
    model_kind = "lightgbm"
    operating_threshold = 0.5
    training_hashes_path = None
    training_cutoff = None
    model_path = "model.txt"

    def __init__(self, booster):
        self.booster = booster

    @classmethod
    def load(cls):
        return cls(lgb.Booster(model_file=str(HERE / cls.model_path)))

    def predict_proba(self, X):
        p = self.booster.predict(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.booster.predict(X) >= self.operating_threshold).astype(np.int8)
'''


def test_validate_adapter_ok_for_good_adapter(run_cli, good_adapter, gate_config):
    res = run_cli("validate-adapter", "--adapter", good_adapter, "--config", gate_config, "--json")
    assert res.returncode == 0, res.output
    rep = json.loads(res.stdout)
    assert rep["ok"] is True and rep["n_failed"] == 0
    names = {c["name"] for c in rep["checks"]}
    assert {"declarations", "file_safety", "load", "predict_proba", "predict"} <= names
    assert all(c["ok"] for c in rep["checks"])


def test_validate_adapter_flags_broken_adapter(run_cli, trained, gate_config, e2e_root):
    d = e2e_root / "sub_broken"
    write_adapter(d, trained)  # copies model.txt
    (d / "broken_adapter.py").write_text(BROKEN_ADAPTER)
    (d / "e2e_adapter.py").unlink()  # single adapter class in the directory
    res = run_cli("validate-adapter", "--adapter", d / "broken_adapter.py", "--config", gate_config, "--json")
    assert res.returncode == 1, res.output
    rep = json.loads(res.stdout)
    assert rep["ok"] is False and rep["n_failed"] >= 1
    failed = [c for c in rep["checks"] if not c["ok"]]
    assert failed
    assert any(c["name"] == "predict_proba" for c in failed), failed
    detail = " ".join(c["detail"] for c in failed).lower()
    assert "shape" in detail or "(n,)" in detail or "(256, 2)" in detail or "2)" in detail, detail


# ---------------------------------------------------------------------------------------------
# (7) report render
# ---------------------------------------------------------------------------------------------
def test_report_render_reproduces_html(run_cli, happy_run, e2e_root):
    out: Path = happy_run["out"]
    dst = e2e_root / "rerendered.html"
    res = run_cli("report", "render", out / "report.json", "-o", dst)
    assert res.returncode == 0, res.output
    assert dst.is_file() and dst.stat().st_size > 5000
    page, orig = dst.read_text(), (out / "report.html").read_text()
    assert page == orig, "re-rendering the same report.json must reproduce report.html"
    assert not _external_refs(page)
    assert happy_run["report"]["verdict"]["label"] in htmllib.unescape(page)
    # default output: report.html next to the JSON (use a copy so the original stays untouched)
    copy = e2e_root / "render_copy"
    copy.mkdir()
    (copy / "report.json").write_text((out / "report.json").read_text())
    res2 = run_cli("report", "render", copy / "report.json")
    assert res2.returncode == 0, res2.output
    assert (copy / "report.html").read_text() == orig


# ---------------------------------------------------------------------------------------------
# (8) corpus verify
# ---------------------------------------------------------------------------------------------
def test_run_records_corpus_verification_and_verify_corpus_rehashes(run_cli, good_adapter, gate_config,
                                                                     happy_run, e2e_root):
    """Regression (xcomp-run-corpus-verify-trusts-stat-cache): report.json says how the corpus files were
    checked, and --verify-corpus forces a full re-hash."""
    ver = happy_run["report"]["corpus"]["verification"]
    assert ver["mode"] in ("cached", "full", "partial", "generated"), ver
    out = e2e_root / "run_verify_corpus"
    res = run_cli("run", "--adapter", good_adapter, "--config", gate_config, "--out", out, "--only", "M1",
                  "--verify-corpus", "--no-html")
    assert res.returncode in (0, 1), res.output
    rep = load_report(out)
    assert rep["corpus"]["verification"]["mode"] == "full"
    assert set(rep["corpus"]["verification"]["files"].values()) == {"hashed"}
    assert rep["config"]["runtime"]["corpus_verification"] == "full"


def test_corpus_verify_synthetic_v2(run_cli, corpus):
    res = run_cli("corpus", "verify", "synthetic_v2")
    assert res.returncode == 0, res.output
    assert res.stdout.startswith("OK synthetic_v2"), res.stdout
    assert corpus.summary()["content_hash"] in res.stdout
