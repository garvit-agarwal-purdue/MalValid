"""Sandbox policy: pickle guards, adapter discovery/declarations, fallback backends, in-process mode,
and the ``malvalid.adapter`` helpers."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

import malvalid.testing  # noqa: F401 - registers the toy_v1 schema
from malvalid.adapter import BaseDetector, resolve_path
from malvalid.config import GateConfig
from malvalid.core import AdapterError, PickleRefused, SandboxError
from malvalid.sandbox.host import (
    InProcessModel,
    SandboxedModel,
    SandboxPolicy,
    inspect_adapter,
    open_model,
    probe_backends,
    select_backend,
)
from malvalid.sandbox.worker import install_pickle_guards
from tests.fixtures.adapters import toolbox_adapter as tb
from tests.fixtures.adapters.builders import eval_rows, make_adapter_dir

BACKENDS = probe_backends()
_skip_without_bwrap = pytest.mark.skipif(not BACKENDS["bwrap"]["available"], reason="bubblewrap is not usable here")


def needs_bwrap(test):
    """Mark a test as needing the bubblewrap sandbox (marker ``sandbox``) and skip it where bwrap is unusable."""
    return pytest.mark.sandbox(_skip_without_bwrap(test))


def _pol(**kw) -> SandboxPolicy:
    kw.setdefault("backend", "bwrap" if BACKENDS["bwrap"]["available"] else "auto")
    return SandboxPolicy(threads=1, memory_mb=4096, **kw)


def test_fixture_lightgbm_model_keeps_lf_line_endings(tmp_path: Path) -> None:
    # A CRLF LightGBM text model (a text-mode write on Windows) makes lgb.Booster(model_file=...) abort the
    # interpreter, which took down the whole Windows test run.
    data = (make_adapter_dir(tmp_path, "toolbox_adapter").parent / "model.txt").read_bytes()
    assert data.startswith(b"tree") and b"\r" not in data


# ---- policy -----------------------------------------------------------------------------------------


def test_policy_from_config(tmp_path: Path) -> None:
    cfg = GateConfig.model_validate({"runtime": {"sandbox_backend": "unshare", "sandbox_memory_mb": 2048,
                                                 "threads": 3, "chunk_rows": 500, "allow_pickle": True}})
    pol = SandboxPolicy.from_config(cfg, run_dir=tmp_path, extra_ro=[tmp_path / "models"])
    assert (pol.enabled, pol.backend, pol.memory_mb, pol.threads, pol.chunk_rows) == (True, "unshare", 2048, 3, 500)
    assert pol.allow_pickle is True
    assert pol.scratch_dir == tmp_path / "private" / "sandbox"
    assert pol.ro_paths == ((tmp_path / "models").resolve(),)
    assert SandboxPolicy.from_config(cfg, run_dir=tmp_path, allow_pickle=False).allow_pickle is False
    json.dumps(pol.to_dict())
    with pytest.raises(ValueError, match="unknown sandbox backend"):
        SandboxPolicy(backend="docker")


def test_select_backend_auto_prefers_isolation() -> None:
    if BACKENDS["bwrap"]["available"] or BACKENDS["unshare"]["available"]:
        backend, warnings = select_backend("auto")
        if BACKENDS["bwrap"]["available"]:
            assert backend == "bwrap" and warnings == []
    else:  # no OS sandbox (e.g. Windows CI): auto refuses unless reduced isolation is allowed
        with pytest.raises(SandboxError):
            select_backend("auto")
        assert select_backend("auto", allow_reduced_isolation=True)[0] == "subprocess"
    assert select_backend("subprocess")[0] == "subprocess"
    assert any("NETWORK NOT ISOLATED" in w for w in select_backend("subprocess")[1])


# ---- declarations & discovery -----------------------------------------------------------------------


@needs_bwrap
def test_inspect_rejects_bad_declarations_with_all_problems(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "nodecl_adapter", model=None)
    with pytest.raises(AdapterError) as ei:
        inspect_adapter(adapter, _pol())
    msg = str(ei.value)
    assert "operating_threshold" in msg
    assert "does_not_exist.txt" in msg
    assert "last spring" in msg


@needs_bwrap
def test_class_discovery_ambiguity_and_explicit_class(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "pickle_adapter", model="payload")
    with pytest.raises(AdapterError) as ei:
        inspect_adapter(adapter, _pol())
    assert "BenignPickleDetector" in str(ei.value) and "GlobalPickleDetector" in str(ei.value)
    assert "--class" in str(ei.value)
    d = inspect_adapter(adapter, _pol(), class_name="BenignPickleDetector")
    assert d.class_name == "BenignPickleDetector" and d.model_paths == ()
    with pytest.raises(AdapterError, match="NoSuchClass"):
        inspect_adapter(adapter, _pol(), class_name="NoSuchClass")


def test_inspect_rejects_non_python_file(tmp_path: Path) -> None:
    f = tmp_path / "adapter.txt"
    f.write_text("x")
    with pytest.raises(AdapterError, match=r"\.py"):
        inspect_adapter(f, _pol())
    with pytest.raises(AdapterError, match="not found"):
        inspect_adapter(tmp_path / "missing.py", _pol())


@needs_bwrap
def test_model_path_override(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    other = tmp_path / "elsewhere" / "other.txt"
    other.parent.mkdir()
    other.write_bytes((adapter.parent / "model.txt").read_bytes())  # byte copy: keep LF line endings
    d = inspect_adapter(adapter, _pol(), model_paths_override=[other])
    assert d.model_paths == (str(other.resolve()),)


# ---- pickle guards ----------------------------------------------------------------------------------


@needs_bwrap
def test_adapter_pickle_loads_refused_unless_allowed(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "pickle_adapter", model="payload")
    pol = _pol()
    decl = inspect_adapter(adapter, pol, class_name="BenignPickleDetector")
    with pytest.raises(PickleRefused) as ei:
        open_model(adapter, pol, declarations=decl, class_name="BenignPickleDetector")
    assert "pickle.loads" in str(ei.value) and "--allow-pickle" in str(ei.value)
    allowed = _pol(allow_pickle=True)
    with open_model(adapter, allowed, declarations=decl, class_name="BenignPickleDetector") as m:
        np.testing.assert_allclose(m.predict_proba(np.zeros((3, 32), np.float32)), 0.25)
        assert m.sandbox_info()["allow_pickle"] is True


@needs_bwrap
def test_dangerous_pickle_global_refused_even_with_allow_pickle(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "pickle_adapter", model="payload")
    pol = _pol(allow_pickle=True)
    decl = inspect_adapter(adapter, pol, class_name="GlobalPickleDetector")
    with pytest.raises(PickleRefused, match="os.getpid"):
        open_model(adapter, pol, declarations=decl, class_name="GlobalPickleDetector")


@needs_bwrap
def test_joblib_model_refused_unless_allowed(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "joblib_adapter", model="joblib")
    decl = inspect_adapter(adapter, _pol())
    with pytest.raises(PickleRefused, match="joblib"):
        open_model(adapter, _pol(), declarations=decl)
    with open_model(adapter, _pol(allow_pickle=True), declarations=decl) as m:
        p = m.predict_proba(eval_rows(20))
        assert p.shape == (20,) and np.all((p >= 0) & (p <= 1))
        te = m.tree_ensemble()  # sklearn GBDT through the sklearn loader
        if te is not None:
            assert te.fidelity(eval_rows(20), p) < 1e-6


def test_install_pickle_guards_in_a_child_process() -> None:
    """The guards patch the interpreter globally, so exercise them in a throw-away interpreter."""
    import subprocess
    import sys

    code = (
        "import pickle, io, numpy as np\n"
        "from malvalid.sandbox.worker import install_pickle_guards\n"
        "from malvalid.core import PickleRefused\n"
        "names = install_pickle_guards(False)\n"
        "data = pickle.dumps({'a': 1})\n"
        "hits = 0\n"
        "for fn in (lambda: pickle.loads(data), lambda: pickle.load(io.BytesIO(data)),\n"
        "           lambda: pickle.Unpickler(io.BytesIO(data)).load(),\n"
        "           lambda: pickle._Unpickler(io.BytesIO(data)).load(),\n"
        "           lambda: np.load(io.BytesIO(b''), allow_pickle=True)):\n"
        "    try:\n"
        "        fn()\n"
        "    except PickleRefused:\n"
        "        hits += 1\n"
        "buf = io.BytesIO(); np.save(buf, np.arange(3)); buf.seek(0)\n"
        "assert np.load(buf).tolist() == [0, 1, 2]\n"
        "print(hits, 'pickle.loads' in names)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["5", "True"]
    assert callable(install_pickle_guards)


# ---- fallback backends & in-process mode ------------------------------------------------------------


@pytest.mark.sandbox
@pytest.mark.skipif(not BACKENDS["unshare"]["available"], reason="unshare -rn is not usable here")
def test_unshare_backend_blocks_network(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    pol = SandboxPolicy(backend="unshare", threads=1, memory_mb=4096)
    with open_model(adapter, pol, declarations=inspect_adapter(adapter, pol)) as m:
        X = eval_rows(4).copy()
        X[0, 0] = tb.MODE_NET_PROBE
        assert m.predict_proba(X).tolist() == [0.0] * 4
        info = m.sandbox_info()
        assert info["backend"] == "unshare" and info["network_isolated"] is True
        assert "NOT restricted" in info["filesystem"]


def test_subprocess_backend_warns_loudly(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    pol = SandboxPolicy(backend="subprocess", threads=1, memory_mb=4096)
    with open_model(adapter, pol, declarations=inspect_adapter(adapter, pol)) as m:
        assert isinstance(m, SandboxedModel)
        booster = lgb.Booster(model_file=str(adapter.parent / "model.txt"))
        np.testing.assert_allclose(m.predict_proba(eval_rows(30)), booster.predict(eval_rows(30)), atol=1e-12)
        info = m.sandbox_info()
        assert info["backend"] == "subprocess" and info["network_isolated"] is False
        assert any("NETWORK NOT ISOLATED" in w for w in info["warnings"])


def test_in_process_mode_is_loudly_flagged(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    pol = SandboxPolicy(enabled=False)
    decl = inspect_adapter(adapter, pol)
    assert decl.class_name == "ToolboxDetector"
    with open_model(adapter, pol, declarations=decl) as m:
        assert isinstance(m, InProcessModel)
        booster = lgb.Booster(model_file=str(adapter.parent / "model.txt"))
        np.testing.assert_allclose(m.predict_proba(eval_rows(30)), booster.predict(eval_rows(30)), atol=1e-12)
        te = m.tree_ensemble()
        assert te is not None and te.n_trees == 30
        info = m.sandbox_info()
        assert info["enabled"] is False and info["network_isolated"] is False
        assert any("SANDBOX DISABLED" in w for w in info["warnings"])
        json.dumps(info)


# ---- malvalid.adapter helpers -----------------------------------------------------------------------


@needs_bwrap
def test_basedetector_adapter_in_sandbox(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "basedetector_adapter")
    pol = _pol()
    decl = inspect_adapter(adapter, pol)
    assert decl.class_name == "MinimalDetector"
    assert decl.training_hashes_path is None and decl.training_cutoff is None
    with open_model(adapter, pol, declarations=decl) as m:
        booster = lgb.Booster(model_file=str(adapter.parent / "model.txt"))
        X = eval_rows(64)
        p = m.predict_proba(X)
        np.testing.assert_allclose(p, booster.predict(X), atol=1e-12)
        np.testing.assert_array_equal(m.predict(X), (p >= 0.5).astype(np.int8))
        te = m.tree_ensemble()
        assert te is not None and te.fidelity(X, p) < 1e-9


def test_basedetector_in_process_helpers(tmp_path: Path) -> None:
    adapter = make_adapter_dir(tmp_path, "toolbox_adapter")
    booster = lgb.Booster(model_file=str(adapter.parent / "model.txt"))

    class D(BaseDetector):
        feature_version = "toy_v1"
        model_kind = "lightgbm"
        operating_threshold = 0.3

    d = D(booster)
    X = eval_rows(10)
    np.testing.assert_allclose(d.predict_proba(X), booster.predict(X))
    np.testing.assert_array_equal(d.predict(X), (booster.predict(X) >= 0.3).astype(np.int8))
    with pytest.raises(AdapterError, match="model_path"):
        D.load()
    with pytest.raises(AdapterError, match="native_model"):
        D().predict_proba(X)
    assert resolve_path("x.txt", tmp_path) == (tmp_path / "x.txt").resolve()
    assert resolve_path("/abs/x.txt") == Path("/abs/x.txt")


@needs_bwrap
def test_schema_featurizer_runs_inside_the_worker(tmp_path: Path) -> None:
    from malvalid import registry
    from tests.fixtures.adapters.builders import make_ember_v2_model

    schema = registry.get_schema("ember_v2")
    if not schema.featurize_available():
        pytest.skip("the ember_v2 raw-PE extractor is not installed")
    adapter = make_adapter_dir(tmp_path, "ember_schema_adapter", model=None)
    make_ember_v2_model(adapter)
    pol = _pol()
    decl = inspect_adapter(adapter, pol)
    assert decl.has_featurize is False
    exe = Path(__import__("setuptools").__file__).parent / "cli-64.exe"
    with open_model(adapter, pol, declarations=decl) as m:
        assert m.has_featurize() and m.featurize_source() == "schema"
        v = m.featurize(exe.read_bytes())
        assert v.shape == (schema.dim,) and v.dtype == np.float32
        np.testing.assert_allclose(v, schema.featurize(exe.read_bytes()), rtol=1e-6)
        assert m.predict_proba(v[None, :]).shape == (1,)
        assert m.sandbox_info()["featurize_source"] == "schema"


@needs_bwrap
def test_adapter_helper_imports_next_to_adapter_and_via_pythonpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = make_adapter_dir(tmp_path, "helper_import_adapter", model=None)
    (adapter.parent / "mg_side_helper.py").write_text("VALUE = 0.5\n")
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "mg_path_helper.py").write_text("VALUE = 0.5\n")
    monkeypatch.setenv("PYTHONPATH", str(lib))
    pol = _pol()
    decl = inspect_adapter(adapter, pol)
    with open_model(adapter, pol, declarations=decl) as m:
        np.testing.assert_allclose(m.predict_proba(np.zeros((2, 32), np.float32)), 0.25)


def test_home_hidden_helper_without_mounts(tmp_path: Path) -> None:
    from malvalid.sandbox.worker import home_hidden

    assert home_hidden(str(tmp_path / "missing")) is True
    home = tmp_path / "home"
    (home / "a" / "b").mkdir(parents=True)
    assert home_hidden(str(home)) is True  # only empty directories: nothing to see
    (home / "a" / "b" / "notes.txt").write_text("private")
    assert home_hidden(str(home)) is False  # an ordinary file anywhere under home is visible


@needs_bwrap
def test_home_hidden_when_adapter_lives_under_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A checkout under $HOME (e.g. /home/runner/work/... on CI, ~/Downloads/MalValid for users): bwrap hides
    # $HOME and re-exposes only the adapter folder read-only, which must still count as "$HOME hidden".
    home = tmp_path / "home"
    adapter = make_adapter_dir(home / "work" / "project", "toolbox_adapter")
    (home / "secret.txt").write_text("private")
    (home / "Documents").mkdir()
    (home / "Documents" / "notes.txt").write_text("private")
    monkeypatch.setenv("HOME", str(home))
    pol = _pol(scratch_dir=tmp_path / "scratch")
    decl = inspect_adapter(adapter, pol)
    with open_model(adapter, pol, declarations=decl) as m:
        info = m.sandbox_info()
        assert info["backend"] == "bwrap" and info["home_hidden"] is True
        assert m.predict_proba(eval_rows(5)).shape == (5,)
