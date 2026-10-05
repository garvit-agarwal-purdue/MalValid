"""The data-only model spec (malvalid_model.json): strict validation, and loading it in the worker."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from malvalid.adapters.spec import (
    SPEC_FILENAME,
    SPEC_SCHEMA,
    ModelSpec,
    build_detector_module,
    load_spec,
    validate_spec,
    write_spec,
)
from malvalid.core import AdapterError, PickleRefused
from malvalid.sandbox.host import SandboxPolicy, inspect_adapter, open_model, probe_backends
from malvalid.sandbox.worker import AdapterRuntime
from tests.unit import model_factories as mf

GOOD = {
    "schema": SPEC_SCHEMA,
    "model_kind": "lightgbm",
    "feature_version": "ember_v2",
    "operating_threshold": 0.8,
    "threshold_source": "declared",
    "calibrate_fpr": None,
    "model_file": "model.txt",
    "training_hashes_file": None,
    "training_cutoff": None,
}


def doc(**kw):
    d = copy.deepcopy(GOOD)
    d.update(kw)
    return d


def bad(**kw):
    with pytest.raises(AdapterError) as e:
        validate_spec(doc(**kw))
    return str(e.value)


# --------------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------------


class TestValidate:
    def test_good_roundtrip(self, tmp_path):
        s = validate_spec(GOOD)
        assert isinstance(s, ModelSpec) and not s.calibrate and s.operating_threshold == 0.8
        p = write_spec(s, tmp_path / "sub")
        assert p.name == SPEC_FILENAME
        assert load_spec(p) == s
        assert json.loads(p.read_text())["schema"] == SPEC_SCHEMA

    @pytest.mark.parametrize("t", [0, 1, 0.0, 1.0, 0.5, 1e-9])
    def test_threshold_ok(self, t):
        assert validate_spec(doc(operating_threshold=t)).operating_threshold == float(t)

    @pytest.mark.parametrize("t", [-0.1, 1.5, float("nan"), float("inf"), "0.5", True, False, [0.5], {}])
    def test_threshold_rejected(self, t):
        assert "operating_threshold" in bad(operating_threshold=t)

    def test_threshold_missing_when_declared(self):
        d = doc()
        del d["operating_threshold"]
        with pytest.raises(AdapterError, match="operating_threshold is required"):
            validate_spec(d)

    def test_threshold_with_calibrate_rejected(self):
        assert "must be null" in bad(threshold_source="calibrate", operating_threshold=0.5, calibrate_fpr=0.01)

    def test_calibrate_ok(self):
        s = validate_spec(doc(threshold_source="calibrate", operating_threshold=None, calibrate_fpr=0.01))
        assert s.calibrate and s.operating_threshold is None and s.calibrate_fpr == 0.01

    @pytest.mark.parametrize("fpr", [None, 0, 0.0, 0.5, -0.01, 0.1000001, float("nan"), "0.01", True])
    def test_calibrate_fpr_rejected(self, fpr):
        assert "calibrate_fpr" in bad(threshold_source="calibrate", operating_threshold=None, calibrate_fpr=fpr)

    def test_calibrate_fpr_upper_bound_inclusive(self):
        assert validate_spec(doc(threshold_source="calibrate", operating_threshold=None, calibrate_fpr=0.1)).calibrate

    def test_fpr_without_calibrate_rejected(self):
        assert "only allowed" in bad(calibrate_fpr=0.01)

    def test_enums(self):
        assert "model_kind" in bad(model_kind="pytorch")
        assert "model_kind" in bad(model_kind=None)
        assert "threshold_source" in bad(threshold_source="auto")

    def test_unknown_key_and_schema(self):
        assert "unknown key" in bad(extra="x")
        assert "schema must be" in bad(schema="malvalid-model-spec/2")

    def test_not_an_object(self):
        for v in ([], "x", 3, None):
            with pytest.raises(AdapterError, match="JSON object"):
                validate_spec(v)

    @pytest.mark.parametrize("name", ["../x.txt", "a/b.txt", ".hidden", "x\\y", SPEC_FILENAME,
                                      SPEC_FILENAME.upper(), "", "..", ".", "x\x00y", "/etc/passwd", 5, None])
    def test_model_file_rejected(self, name):
        assert "model_file" in bad(model_file=name)

    @pytest.mark.parametrize("name", ["../h.txt", "a/b", ".h", "malvalid_model.json", ""])
    def test_hashes_file_rejected(self, name):
        assert "training_hashes_file" in bad(training_hashes_file=name)

    def test_hashes_file_same_as_model(self):
        assert "must not be the model file" in bad(training_hashes_file="model.txt")

    @pytest.mark.parametrize("c", ["2018-13", "18-10", "2018/10", "2018-10-32", "2018-00", "abc", "", 2018,
                                   "2018-02-30x"])
    def test_cutoff_rejected(self, c):
        assert "training_cutoff" in bad(training_cutoff=c)

    @pytest.mark.parametrize("c", ["2018", "2018-10", "2018-10-05", None])
    def test_cutoff_ok(self, c):
        assert validate_spec(doc(training_cutoff=c)).training_cutoff == c

    @pytest.mark.parametrize("fv", ["ember_v9", "EMBER", "../x", "", None, 3])
    def test_feature_version_rejected(self, fv):
        assert "feature_version" in bad(feature_version=fv)

    def test_feature_version_registry_toggle(self):
        assert validate_spec(doc(feature_version="ember_v9"), check_registry=False).feature_version == "ember_v9"

    def test_all_problems_listed(self):
        msg = bad(model_kind="x", operating_threshold=2, model_file="../a")
        assert msg.count("\n  - ") >= 3

    def test_load_spec_errors(self, tmp_path):
        with pytest.raises(AdapterError, match="cannot read"):
            load_spec(tmp_path / "nope.json")
        p = tmp_path / "s.json"
        p.write_text("{not json")
        with pytest.raises(AdapterError, match="not valid JSON"):
            load_spec(p)
        p.write_bytes(b" " * (64 * 1024 + 1))
        with pytest.raises(AdapterError, match="larger than"):
            load_spec(p)

    def test_write_spec_validates(self, tmp_path):
        with pytest.raises(AdapterError):
            write_spec(doc(operating_threshold=7), tmp_path)
        assert not (tmp_path / SPEC_FILENAME).exists()


# --------------------------------------------------------------------------------------------------
# loading the spec detector
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("spec_sub")
    booster = mf.save_lgbm(d / "model.txt", 2381)
    write_spec(doc(operating_threshold=0.5), d)
    return d, booster


def _rows(n=40, seed=3):
    return np.random.default_rng(seed).normal(size=(n, 2381)).astype(np.float32)


class TestInProcess:
    def test_inspect_load_predict(self, spec_dir):
        d, booster = spec_dir
        rt = AdapterRuntime(d / SPEC_FILENAME, unique_module_name=True)
        info = rt.inspect()
        assert info["class_name"] == "ModelFileDetector"
        decl = info["declarations"]
        assert decl["feature_version"]["value"] == "ember_v2"
        assert decl["model_kind"]["value"] == "lightgbm"
        assert decl["operating_threshold"]["value"] == 0.5
        rt.load()
        X = _rows()
        p = rt.inst.predict_proba(X)
        np.testing.assert_allclose(p, booster.predict(X), rtol=1e-6, atol=1e-9)
        np.testing.assert_array_equal(rt.inst.predict(X), (p >= 0.5).astype(int))

    def test_calibrate_spec_gets_placeholder_threshold(self, tmp_path):
        mf.save_lgbm(tmp_path / "model.txt")
        write_spec(doc(threshold_source="calibrate", operating_threshold=None, calibrate_fpr=0.01), tmp_path)
        info = AdapterRuntime(tmp_path / SPEC_FILENAME, unique_module_name=True).inspect()
        assert info["declarations"]["operating_threshold"]["value"] == 0.5

    def test_spec_dir_not_on_sys_path_and_module_not_imported(self, tmp_path):
        mf.save_lgbm(tmp_path / "model.txt")
        marker = tmp_path / "lightgbm_evil.py"
        marker.write_text("import os\nopen(os.path.join(os.path.dirname(__file__), 'PWNED'), 'w').write('x')\n")
        (tmp_path / "sitecustomize.py").write_text("raise SystemExit('executed')\n")
        write_spec(doc(), tmp_path)
        before = list(sys.path)
        rt = AdapterRuntime(tmp_path / SPEC_FILENAME, unique_module_name=True)
        rt.inspect()
        rt.load()
        assert str(tmp_path) not in sys.path and sys.path == before
        assert not (tmp_path / "PWNED").exists()
        assert "lightgbm_evil" not in sys.modules and "sitecustomize" not in {
            m for m, mod in sys.modules.items() if getattr(mod, "__file__", "") == str(tmp_path / "sitecustomize.py")}

    def test_python_adapter_still_adds_dir(self, tmp_path):
        """Control: the .py path keeps its old behaviour, so the spec test above is meaningful."""
        mf.save_lgbm(tmp_path / "model.txt")
        (tmp_path / "ctl_adapter.py").write_text(
            "from malvalid.adapter import BaseDetector\n"
            "class D(BaseDetector):\n    feature_version='ember_v2'\n    model_kind='lightgbm'\n"
            "    operating_threshold=0.5\n    model_path='model.txt'\n")
        try:
            AdapterRuntime(tmp_path / "ctl_adapter.py", unique_module_name=True).inspect()
            assert str(tmp_path) in sys.path
        finally:
            if str(tmp_path) in sys.path:
                sys.path.remove(str(tmp_path))

    def test_missing_model_file(self, tmp_path):
        write_spec(doc(), tmp_path)
        rt = AdapterRuntime(tmp_path / SPEC_FILENAME, unique_module_name=True)
        rt.inspect()
        with pytest.raises(AdapterError, match="not found"):
            rt.load()

    def test_invalid_spec_is_adapter_error(self, tmp_path):
        (tmp_path / SPEC_FILENAME).write_text(json.dumps(doc(operating_threshold=9)))
        with pytest.raises(AdapterError, match="invalid model spec"):
            AdapterRuntime(tmp_path / SPEC_FILENAME, unique_module_name=True).inspect()

    def test_build_detector_module_values(self, spec_dir):
        d, _ = spec_dir
        mod = build_detector_module(d / SPEC_FILENAME, "m_x")
        cls = mod.ModelFileDetector
        assert (cls.feature_version, cls.model_kind, cls.model_path, cls.operating_threshold) == (
            "ember_v2", "lightgbm", "model.txt", 0.5)

    def test_pickle_refused_in_process(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MALVALID_ALLOW_PICKLE", raising=False)
        mf.save_pickle(tmp_path / "model.pkl")
        write_spec(doc(model_kind="sklearn_gbdt", model_file="model.pkl"), tmp_path)
        rt = AdapterRuntime(tmp_path / SPEC_FILENAME, unique_module_name=True)
        rt.inspect()
        with pytest.raises((PickleRefused, AdapterError)) as e:
            rt.load()
        assert "pickle" in str(e.value).lower()


_isolating = [n for n, v in probe_backends().items() if v.get("available") and v.get("network_isolated")]


@pytest.mark.slow
@pytest.mark.skipif(not _isolating, reason="no isolating sandbox backend on this host")
class TestRealSandbox:
    @pytest.fixture
    def policy(self, tmp_path_factory):
        return SandboxPolicy(scratch_dir=tmp_path_factory.mktemp("scratch"), threads=1, memory_mb=4096,
                             chunk_rows=64)

    def test_predict_proba_and_host_side_threshold(self, spec_dir, policy):
        d, booster = spec_dir
        spec = d / SPEC_FILENAME
        decl = inspect_adapter(spec, policy)
        assert decl.extras["predict_from_proba"] is True and decl.extras["submission"] == "model_file"
        assert decl.operating_threshold == 0.5 and decl.feature_version == "ember_v2"
        assert decl.model_paths == (str((d / "model.txt").resolve()),)
        handle = open_model(spec, policy, declarations=decl)
        try:
            X = _rows(60)
            p = handle.predict_proba(X)
            np.testing.assert_allclose(p, booster.predict(X), rtol=1e-5, atol=1e-7)
            assert 0.05 < float(np.mean(p >= 0.5)) < 0.95  # both classes present: the threshold test is meaningful
            np.testing.assert_array_equal(handle.predict(X), (p >= 0.5).astype(np.int8))
            handle.declarations = dataclasses.replace(handle.declarations, operating_threshold=0.0)
            assert handle.predict(X).all()
            handle.declarations = dataclasses.replace(handle.declarations, operating_threshold=1.0)
            assert not handle.predict(X).any() or (p >= 1.0).any()
            mid = float(np.median(p))
            handle.declarations = dataclasses.replace(handle.declarations, operating_threshold=mid)
            np.testing.assert_array_equal(handle.predict(X), (p >= mid).astype(np.int8))
        finally:
            handle.close()

    @pytest.mark.parametrize("kind,name", [("lightgbm", "model.txt"), ("xgboost", "model.json"),
                                           ("xgboost", "model.ubj")])
    def test_non_ascii_model_path_in_worker(self, tmp_path, policy, kind, name):
        # The sandboxed worker loads the model from a submission folder whose path has non-ASCII characters
        # (spec model_file names are ASCII by rule, the folder can be anything).
        # Saved under an ASCII name first, so the test does not depend on the library writing such a path.
        d = tmp_path / "José_模型"
        d.mkdir()
        ascii_path = tmp_path / ("m" + Path(name).suffix)
        booster = mf.save_lgbm(ascii_path) if kind == "lightgbm" else mf.save_xgb(ascii_path)
        (d / name).write_bytes(ascii_path.read_bytes())
        write_spec(doc(model_kind=kind, model_file=name, operating_threshold=0.5), d)
        spec = d / SPEC_FILENAME
        decl = inspect_adapter(spec, policy)
        assert decl.model_paths == (str((d / name).resolve()),)
        h = open_model(spec, policy, declarations=decl)
        try:
            X = _rows(30)
            if kind == "lightgbm":
                ref = booster.predict(X)
            else:
                import xgboost as xgb

                ref = booster.predict(xgb.DMatrix(X))
            np.testing.assert_allclose(h.predict_proba(X), ref, rtol=1e-5, atol=1e-7)
        finally:
            h.close()

    def test_calibrate_spec_declarations_pending(self, tmp_path, policy):
        mf.save_lgbm(tmp_path / "model.txt")
        write_spec(doc(threshold_source="calibrate", operating_threshold=None, calibrate_fpr=0.01), tmp_path)
        decl = inspect_adapter(tmp_path / SPEC_FILENAME, policy)
        assert decl.extras["threshold_source"] == "calibration_pending"
        assert math.isnan(decl.operating_threshold)
        assert decl.extras["spec"]["calibrate_fpr"] == 0.01

    def test_spec_dir_code_not_executed(self, tmp_path, policy):
        mf.save_lgbm(tmp_path / "model.txt")
        (tmp_path / "lightgbm.py").write_text("raise RuntimeError('shadowed lightgbm was imported')\n")
        (tmp_path / "sitecustomize.py").write_text("raise RuntimeError('sitecustomize executed')\n")
        write_spec(doc(), tmp_path)
        spec = tmp_path / SPEC_FILENAME
        decl = inspect_adapter(spec, policy)
        h = open_model(spec, policy, declarations=decl)
        try:
            assert h.predict_proba(_rows(5)).shape == (5,)
        finally:
            h.close()

    def test_pickle_refused_without_allow_pickle(self, tmp_path, policy):
        mf.save_pickle(tmp_path / "model.pkl")
        write_spec(doc(model_kind="sklearn_gbdt", model_file="model.pkl"), tmp_path)
        spec = tmp_path / SPEC_FILENAME
        decl = inspect_adapter(spec, policy)
        with pytest.raises((PickleRefused, AdapterError)) as e:
            h = open_model(spec, policy, declarations=decl)
            h.close()
        assert "pickle" in str(e.value).lower()
