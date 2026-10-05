"""M0 model file safety: modelscan + static pickle-opcode scan, pickle policy, gate/abort semantics.

Malicious-looking pickles are hand-assembled from opcodes and are NEVER loaded: every test runs with
the unpickling entry points patched to fail loudly.
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import joblib
import joblib.numpy_pickle_compat  # noqa: F401 - subclasses Unpickler at import
import numpy as np
import pytest

import malvalid.testing  # noqa: F401 - registers the toy_v1 schema
from malvalid.context import ModelDeclarations
from malvalid.core import ConfigError, GateOutcome, Status, to_jsonable
from malvalid.modules.file_safety import (
    FileSafetyModule,
    classify_pickle_global,
    detect_format,
    discover_model_files,
    scan_artifacts,
)
from malvalid.testing import make_context
from tests.fixtures.adapters.builders import GETPID_PICKLE, make_adapter_dir

#: ``PROTO 2 | GLOBAL builtins eval | BINUNICODE "'harmless'" | TUPLE1 | REDUCE | STOP`` —
#: would call ``eval("'harmless'")`` (a string literal) if it were ever loaded.
EVAL_PICKLE = b"\x80\x02cbuiltins\neval\nX\x0a\x00\x00\x00'harmless'\x85R."
#: Protocol-4 STACK_GLOBAL form of ``os.system("true")`` (never loaded).
STACK_GLOBAL_PICKLE = (
    b"\x80\x04\x8c\x02os\x94\x8c\x06system\x94\x93\x94\x8c\x04true\x94\x85\x94R\x94."
)

_ORIG_UNPICKLER = pickle.Unpickler


@pytest.fixture(autouse=True)
def _never_unpickle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to deserialize during a scan fails the test."""

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("M0 must never deserialize an artifact")

    class NoLoadUnpickler(_ORIG_UNPICKLER):  # stays subclassable for libraries imported later
        def load(self) -> Any:
            boom()

    for name in ("load", "loads", "_load", "_loads"):
        monkeypatch.setattr(pickle, name, boom)
    monkeypatch.setattr(pickle, "Unpickler", NoLoadUnpickler)
    monkeypatch.setattr(pickle._Unpickler, "load", boom)
    monkeypatch.setattr(joblib, "load", boom)
    monkeypatch.setattr(np, "load", boom)


class _Unloaded:
    """The runner's M0 placeholder: only ``declarations`` is meaningful."""

    def __init__(self, decl: ModelDeclarations):
        self.declarations = decl

    def has_featurize(self) -> bool:
        return False

    def tree_ensemble(self) -> None:
        return None


def _run_m0(adapter: Path, artifacts: list[Path] | None, *, allow_pickle: bool = False, **params: Any):
    decl = ModelDeclarations(
        feature_version="toy_v1", model_kind="lightgbm", operating_threshold=0.5, training_hashes_path=None,
        training_cutoff=None, model_paths=tuple(str(p) for p in (artifacts or ())), adapter_path=str(adapter),
        class_name="X",
    )
    ctx = make_context(FileSafetyModule, model=_Unloaded(decl), corpus=None, params=params)
    ctx.extras.update({"artifact_paths": [str(p) for p in (artifacts or [])], "adapter_path": str(adapter),
                       "allow_pickle": allow_pickle})
    res = FileSafetyModule().run(ctx)
    json.dumps(to_jsonable(res.to_dict()), allow_nan=False)
    return res


# ---- static classification ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "name", "severity"),
    [
        ("builtins", "eval", "CRITICAL"),
        ("builtins", "getattr", "CRITICAL"),
        ("os", "system", "CRITICAL"),
        ("posix", "system", "CRITICAL"),
        ("subprocess", "Popen", "CRITICAL"),
        ("socket", "create_connection", "CRITICAL"),
        ("urllib.request", "urlopen", "HIGH"),
        ("numpy.core.multiarray", "_reconstruct", None),
        ("sklearn.ensemble._gb", "GradientBoostingClassifier", None),
        ("collections", "OrderedDict", None),
        ("builtins", "dict", None),
        ("numpy", "load", "CRITICAL"),
        ("mylab.models", "Detector", "MEDIUM"),
    ],
)
def test_classify_pickle_global(module: str, name: str, severity: str | None) -> None:
    assert classify_pickle_global(module, name)[0] == severity


# ---- scan_artifacts ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("payload", "what"), [(EVAL_PICKLE, "builtins.eval"), (STACK_GLOBAL_PICKLE, "os.system")])
def test_hand_assembled_malicious_pickle_is_critical_without_loading(tmp_path: Path, payload: bytes, what: str) -> None:
    f = tmp_path / "model.pkl"
    f.write_bytes(payload)
    rep = scan_artifacts([f], allow_pickle=True)
    assert rep.abort
    a = rep.artifacts[0]
    assert a.is_pickle and a.format == "pickle"
    crit = [x for x in a.findings if x.severity == "CRITICAL"]
    assert crit and any(f"{x.module}.{x.operator}" == what for x in crit)
    assert "malvalid-opcodes" in crit[0].sources
    assert any(what in r for r in rep.abort_reasons())
    json.dumps(to_jsonable(rep.to_dict()), allow_nan=False)


def test_malicious_pickle_hidden_in_zip_and_compressed_containers(tmp_path: Path) -> None:
    z = tmp_path / "model.pt"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("archive/data.pkl", EVAL_PICKLE)
        zf.writestr("archive/version", "3\n")
    gz = tmp_path / "model.joblib"
    import zlib

    gz.write_bytes(zlib.compress(STACK_GLOBAL_PICKLE))
    rep = scan_artifacts([z, gz], allow_pickle=True)
    by = {a.name: a for a in rep.artifacts}
    assert by["model.pt"].is_pickle and by["model.pt"].max_severity() == "CRITICAL"
    assert by["model.joblib"].is_pickle and by["model.joblib"].max_severity() == "CRITICAL"


def test_benign_pickle_is_policy_violation_but_not_critical(tmp_path: Path) -> None:
    f = tmp_path / "state.pkl"
    import collections

    f.write_bytes(pickle.dumps(collections.OrderedDict(a=np.arange(3)), protocol=4))
    rep = scan_artifacts([f], allow_pickle=False)
    assert rep.counts()["CRITICAL"] == 0 and rep.abort
    assert rep.artifacts[0].policy_violation and "--allow-pickle" in rep.artifacts[0].policy_violation
    ok = scan_artifacts([f], allow_pickle=True)
    assert not ok.abort and ok.counts() == {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}


def test_lightgbm_text_is_safe_and_unscanned_with_reason(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    f = adapter.parent / "model.txt"
    assert detect_format(f)[0] == "lightgbm-text"
    rep = scan_artifacts([f], allow_pickle=False)
    a = rep.artifacts[0]
    assert not rep.abort and not a.is_pickle and a.findings == []
    assert a.scanned is False and "LightGBM" in (a.scan_reason or "")
    assert len(a.sha256) == 64 and a.size == f.stat().st_size
    assert rep.scanner == "modelscan" and rep.scanner_version


def test_missing_artifact_is_reported(tmp_path: Path) -> None:
    rep = scan_artifacts([tmp_path / "nope.txt"], allow_pickle=False)
    assert rep.artifacts[0].error


# ---- FileSafetyModule -------------------------------------------------------------------------------


def test_m0_passes_clean_lightgbm_model(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    res = _run_m0(adapter, [adapter.parent / "model.txt"])
    assert res.status is Status.PASS and res.gate_outcome is GateOutcome.PASSED
    assert res.score == 1.0 and res.details["abort"] is False
    for k in ("n_artifacts", "n_pickle", "n_critical", "n_high", "n_medium", "n_low", "scanner", "scanner_version"):
        assert k in res.metrics
    assert res.metrics["n_artifacts"] == 1 and res.metrics["n_pickle"] == 0
    assert "no unsafe content" in res.finding


def test_m0_blocks_sklearn_joblib_without_allow_pickle(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "joblib_adapter", model="joblib")
    res = _run_m0(adapter, [adapter.parent / "model.joblib"])
    assert res.gate_outcome is GateOutcome.FAILED and res.status is Status.FAIL
    assert res.details["abort"] is True and res.score == 0.0
    assert res.metrics["n_pickle"] == 1 and res.metrics["n_critical"] == 0
    assert "--allow-pickle" in res.finding and "skl2onnx" in res.finding  # explains how to re-export
    assert any("joblib" in n for n in res.notes)  # the adapter source calls joblib.load


def test_m0_warns_on_allowed_pickle(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "joblib_adapter", model="joblib")
    res = _run_m0(adapter, [adapter.parent / "model.joblib"], allow_pickle=True)
    assert res.details["abort"] is False and res.gate_outcome is GateOutcome.PASSED
    assert res.status is Status.WARN and res.score == 0.75
    assert res.notes and res.notes[0].startswith("PICKLE ACCEPTED")


def test_m0_aborts_on_critical_pickle_even_with_allow_pickle(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "pickle_adapter", model="payload")
    res = _run_m0(adapter, [adapter.parent / "payload.pkl"], allow_pickle=True)
    assert res.details["abort"] is True and res.status is Status.FAIL and res.score == 0.0
    assert res.metrics["n_critical"] >= 1 and "os.getpid" in res.finding
    assert (adapter.parent / "payload.pkl").read_bytes() == GETPID_PICKLE


def test_m0_scans_undeclared_files_in_adapter_dir(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    (adapter.parent / "backup.pkl").write_bytes(EVAL_PICKLE)
    res = _run_m0(adapter, [adapter.parent / "model.txt"])
    assert res.details["abort"] is True and res.metrics["n_critical"] >= 1
    stray = [a for a in res.details["artifacts"] if a["path"].endswith("backup.pkl")]
    assert stray and stray[0]["declared"] is False
    off = _run_m0(adapter, [adapter.parent / "model.txt"], scan_adapter_dir=False)
    assert off.details["abort"] is False and off.metrics["n_artifacts"] == 1


def test_m0_discovers_model_files_when_none_declared(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    assert [p.name for p in discover_model_files(adapter.parent)] == ["model.txt"]
    res = _run_m0(adapter, None)
    assert res.metrics["n_artifacts"] == 1 and res.gate_outcome is GateOutcome.PASSED
    assert any("model_path" in n for n in res.notes)


def test_m0_high_findings_fail_gate_per_fail_on(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter", model=None)
    f = adapter.parent / "net.pkl"
    f.write_bytes(b"\x80\x02curllib.request\nurlopen\n)R.")  # HIGH (network), never loaded
    res = _run_m0(adapter, [f], allow_pickle=True)
    assert res.metrics["n_high"] >= 1 and res.metrics["n_critical"] == 0
    assert res.gate_outcome is GateOutcome.FAILED and res.details["abort"] is False
    lenient = _run_m0(adapter, [f], allow_pickle=True, fail_on="CRITICAL")
    assert lenient.gate_outcome is GateOutcome.PASSED
    with pytest.raises(ConfigError):
        _run_m0(adapter, [f], fail_on="SEVERE")


def test_m0_tables_are_recorded(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    res = _run_m0(adapter, [adapter.parent / "model.txt"])
    assert any(k.endswith(".artifacts") for k in res.artifacts)


def test_detect_format_never_unpickles_numpy_object_arrays(tmp_path: Path) -> None:
    f = tmp_path / "weights.npy"
    buf = io.BytesIO()
    np.save(buf, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    f.write_bytes(buf.getvalue())
    rep = scan_artifacts([f], allow_pickle=False)
    assert rep.artifacts[0].is_pickle and rep.abort
