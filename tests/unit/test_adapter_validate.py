"""``validate_adapter``: the pre-flight report researchers run before ``malvalid run``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import malvalid.testing  # noqa: F401 - registers toy_v1 + toy_v1_corpus
from malvalid.adapters.validate import ValidationReport, benign_test_executable, validate_adapter
from malvalid.config import GateConfig
from malvalid.core import to_jsonable
from malvalid.sandbox.host import probe_backends
from tests.fixtures.adapters.builders import make_adapter_dir

pytestmark = pytest.mark.skipif(not probe_backends()["bwrap"]["available"], reason="bubblewrap is not usable here")


def _cfg(corpus: str = "toy_v1_corpus", **runtime) -> GateConfig:
    rt = {"sandbox_backend": "bwrap", "threads": 1, "sandbox_memory_mb": 4096, **runtime}
    return GateConfig.model_validate({"corpus": corpus, "runtime": rt})


def _statuses(rep: ValidationReport) -> dict[str, str]:
    return {c.name: c.status for c in rep.checks}


def _serializable(rep: ValidationReport) -> dict:
    d = rep.to_dict()
    json.dumps(to_jsonable(d), allow_nan=False)
    assert rep.render_text().startswith("malvalid adapter validation:")
    return d


def test_good_adapter_passes_every_check(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    rep = validate_adapter(adapter, _cfg(), work_dir=tmp_path / "work")
    st = _statuses(rep)
    assert rep.ok, rep.render_text()
    for name in ("declarations", "feature_schema", "training_manifest", "file_safety", "load", "sandbox",
                 "predict_proba", "batch_independence", "predict", "determinism", "tree_access"):
        assert st[name] == "pass", (name, rep.render_text())
    assert st["featurize"] == ("pass" if benign_test_executable() else "skip")
    assert rep.declarations is not None and rep.declarations.class_name == "ToolboxDetector"
    assert "toy_v1_corpus" in (rep.probe_source or "")
    assert rep.sandbox and rep.sandbox["backend"] == "bwrap"
    d = _serializable(rep)
    assert d["ok"] is True and d["n_failed"] == 0
    assert (tmp_path / "work" / "private" / "sandbox" / "worker.log").is_file()
    assert "Result: OK" in rep.render_text()


def test_random_vectors_when_corpus_is_unavailable(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    rep = validate_adapter(adapter, _cfg(corpus="ember_v2_2018"))
    assert rep.ok, rep.render_text()
    assert "random" in (rep.probe_source or "") and "ember_v2" in (rep.probe_source or "")


def test_broken_adapter_fails_predict_proba_with_actionable_message(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "broken_adapter")
    rep = validate_adapter(adapter, _cfg())
    st = _statuses(rep)
    assert not rep.ok
    assert st["load"] == "pass" and st["predict_proba"] == "fail"
    assert "[:, 1]" in rep.check("predict_proba").detail
    assert st["training_manifest"] == "warn"  # hashes/cutoff declared None
    d = _serializable(rep)
    assert d["ok"] is False and d["n_failed"] >= 1
    assert "FAIL" in rep.render_text() and "FAILED" in rep.render_text()


def test_invalid_declarations_stop_validation(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "nodecl_adapter", model=None)
    rep = validate_adapter(adapter, _cfg())
    assert [c.name for c in rep.checks] == ["config", "declarations"]
    assert not rep.ok and "operating_threshold" in rep.check("declarations").detail
    _serializable(rep)


def test_pickle_artifact_stops_before_load_unless_allowed(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "joblib_adapter", model="joblib")
    rep = validate_adapter(adapter, _cfg())
    st = _statuses(rep)
    assert st["file_safety"] == "fail" and "load" not in st
    assert "--allow-pickle" in rep.check("file_safety").detail
    allowed = validate_adapter(adapter, _cfg(), allow_pickle=True)
    st2 = _statuses(allowed)
    assert st2["file_safety"] == "warn" and st2["load"] == "pass" and st2["predict_proba"] == "pass", \
        allowed.render_text()
    assert allowed.ok, allowed.render_text()


# Optional local data for the slow "ember" tests: set MALVALID_TEST_DATA to a directory holding
# models/ember_model_2018.txt and corpora/. Without it those tests skip.
_DATA = Path(os.environ.get("MALVALID_TEST_DATA", "/nonexistent-malvalid-test-data"))
REF_MODEL = _DATA / "models" / "ember_model_2018.txt"
CORPORA = _DATA / "corpora"


@pytest.mark.slow
@pytest.mark.ember
@pytest.mark.skipif(not (REF_MODEL.is_file() and (CORPORA / "ember_v2_2018" / "manifest.json").is_file()),
                    reason="reference EMBER2018 model / canonical corpus not present")
def test_reference_ember2018_model_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MALVALID_CORPUS_DIR", str(CORPORA))
    adapter = tmp_path / "ref_adapter.py"
    adapter.write_text(
        "from malvalid.adapter import BaseDetector\n\n\n"
        "class Ember2018Reference(BaseDetector):\n"
        "    feature_version = 'ember_v2'\n"
        "    model_kind = 'lightgbm'\n"
        "    operating_threshold = 0.8336\n"
        "    training_cutoff = '2018-10'\n"
        f"    model_path = {str(REF_MODEL)!r}\n"
    )
    rep = validate_adapter(adapter, _cfg(corpus="ember_v2_2018", threads=4, sandbox_memory_mb=16384))
    st = _statuses(rep)
    assert rep.ok, rep.render_text()
    assert st["tree_access"] == "pass" and st["featurize"] == "pass" and st["predict_proba"] == "pass"
    assert "ember_v2_2018" in (rep.probe_source or "")


def test_config_the_run_would_refuse_fails_preflight(tmp_path: Path) -> None:
    """Regression (xcomp-validate-adapter-accepts-invalid-config): a gate config that `malvalid run`
    rejects (a typo'd parameter name, a bad parameter value) must fail pre-flight too."""
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    rt = {"sandbox_backend": "bwrap", "threads": 1, "sandbox_memory_mb": 4096}
    typo = GateConfig.model_validate({"corpus": "toy_v1_corpus", "runtime": rt,
                                      "modules": {"performance": {"max_fp": 0.02}}})
    rep = validate_adapter(adapter, typo)
    assert not rep.ok and "Result: OK" not in rep.render_text()
    c = rep.check("config")
    assert c is not None and c.status == "fail" and "max_fp" in c.detail
    assert rep.check("load").status == "pass"  # the adapter itself is still checked
    bad_value = GateConfig.model_validate({"corpus": "toy_v1_corpus", "runtime": rt,
                                           "modules": {"drift": {"min_aut_f1": "0.7"}}})
    c = validate_adapter(adapter, bad_value).check("config")
    assert c.status == "fail" and "drift.min_aut_f1 must be a number" in c.detail


def test_valid_config_check_passes(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "nodecl_adapter", model=None)
    rep = validate_adapter(adapter, _cfg())
    assert rep.check("config").status == "pass"
