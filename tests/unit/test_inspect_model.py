"""inspect_model: kind / feature count / feature version inferred from file content, never by executing it."""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import pytest

from malvalid.inspect_model import ModelInfo, inspect_model
from tests.unit import model_factories as mf

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def no_unpickle(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def boom(*a, **k):
        calls.append("called")
        raise AssertionError("a pickle was loaded during inspection")

    monkeypatch.setattr(pickle, "load", boom)
    monkeypatch.setattr(pickle, "loads", boom)
    monkeypatch.setattr(pickle, "Unpickler", boom)
    try:
        import joblib

        monkeypatch.setattr(joblib, "load", boom)
    except ImportError:  # pragma: no cover
        pass
    return calls


class TestLightGBM:
    @pytest.mark.parametrize("n, fv, corpus", [(2381, "ember_v2", "ember_v2_2018"), (2568, "ember_v3", "ember_v3_2024")])
    def test_text_model(self, tmp_path, n, fv, corpus):
        p = tmp_path / "m.txt"
        mf.save_lgbm(p, n)
        info = inspect_model(p)
        assert info.ok and not info.errors
        assert (info.format, info.model_kind, info.n_features) == ("lightgbm-text", "lightgbm", n)
        assert (info.feature_version, info.default_corpus, info.is_pickle) == (fv, corpus, False)

    def test_content_beats_extension(self, tmp_path):
        mf.save_lgbm(tmp_path / "m.txt")
        shutil.copy(tmp_path / "m.txt", tmp_path / "EMBER_like.model")
        a, b = inspect_model(tmp_path / "m.txt"), inspect_model(tmp_path / "EMBER_like.model")
        assert b.model_kind == "lightgbm" and b.n_features == 2381 and b.feature_version == "ember_v2"
        assert (a.model_kind, a.n_features, a.format) == (b.model_kind, b.n_features, b.format)

    def test_wrong_feature_count_error(self, tmp_path):
        p = tmp_path / "m.txt"
        mf.save_lgbm(p, 100)
        info = inspect_model(p)
        assert not info.ok
        assert info.n_features == 100 and info.feature_version is None and info.model_kind == "lightgbm"
        assert any("model expects 100 features; malvalid supports EMBER v2 (2381) and EMBER v3 (2568)" in e
                   for e in info.errors)

    def test_to_dict_keys(self, tmp_path):
        mf.save_lgbm(tmp_path / "m.txt")
        d = inspect_model(tmp_path / "m.txt").to_dict()
        assert {"ok", "model_kind", "n_features", "feature_version", "default_corpus", "is_pickle", "errors",
                "notes", "details", "supported", "format"} <= set(d)
        assert d["supported"]["2381"] == "ember_v2" and d["supported"]["2568"] == "ember_v3"


class TestXGBoost:
    @pytest.mark.parametrize("ext, fmt", [("json", "xgboost-json"), ("ubj", "xgboost-ubj")])
    @pytest.mark.parametrize("n, fv", [(2381, "ember_v2"), (2568, "ember_v3")])
    def test_json_and_ubj(self, tmp_path, ext, fmt, n, fv):
        pytest.importorskip("xgboost")
        p = tmp_path / f"m.{ext}"
        mf.save_xgb(p, n)
        info = inspect_model(p)
        assert info.ok, info.errors
        assert (info.format, info.model_kind, info.n_features, info.feature_version) == (fmt, "xgboost", n, fv)

    def test_ubj_with_wrong_extension(self, tmp_path):
        pytest.importorskip("xgboost")
        mf.save_xgb(tmp_path / "m.ubj")
        shutil.copy(tmp_path / "m.ubj", tmp_path / "weights.bin")
        info = inspect_model(tmp_path / "weights.bin")
        assert info.format == "xgboost-ubj" and info.n_features == 2381

    def test_json_that_is_not_xgboost(self, tmp_path):
        p = tmp_path / "x.json"
        p.write_text('{"hello": 1}')
        info = inspect_model(p)
        assert not info.ok and info.model_kind is None and info.errors


class TestONNX:
    def test_onnx(self, tmp_path):
        pytest.importorskip("skl2onnx")
        p = tmp_path / "m.onnx"
        mf.save_onnx(p, 2568)
        info = inspect_model(p)
        assert info.ok, info.errors
        assert (info.format, info.model_kind, info.n_features, info.feature_version, info.default_corpus) == (
            "onnx", "onnx", 2568, "ember_v3", "ember_v3_2024")

    def test_onnx_wrong_dim(self, tmp_path):
        pytest.importorskip("skl2onnx")
        p = tmp_path / "m.onnx"
        mf.save_onnx(p, 50)
        info = inspect_model(p)
        assert not info.ok and any("model expects 50 features" in e for e in info.errors)


class TestPickle:
    @pytest.mark.parametrize("saver, name", [(mf.save_pickle, "m.pkl"), (mf.save_joblib, "m.joblib"),
                                             (mf.save_pickle, "m.bin")])
    def test_static_scan_only(self, tmp_path, no_unpickle, saver, name):
        p = tmp_path / name
        saver(p)
        info = inspect_model(p)
        assert info.is_pickle is True and info.format == "pickle"
        assert info.model_kind == "sklearn_gbdt"
        assert info.feature_version is None and info.n_features is None
        assert not info.ok
        assert no_unpickle == []
        assert any("pickle" in n for n in info.notes)


class TestUnrecognised:
    def test_garbage(self, tmp_path):
        p = tmp_path / "g.dat"
        p.write_bytes(bytes(range(256)) * 4)
        info = inspect_model(p)
        assert not info.ok and info.model_kind is None
        assert any("unrecognized model format" in e for e in info.errors)

    def test_empty_and_missing(self, tmp_path):
        (tmp_path / "e.txt").write_bytes(b"")
        assert any("empty" in e for e in inspect_model(tmp_path / "e.txt").errors)
        assert any("not found" in e for e in inspect_model(tmp_path / "nope.txt").errors)

    def test_text_not_lightgbm(self, tmp_path):
        p = tmp_path / "t.txt"
        p.write_text("just some words\n")
        assert not inspect_model(p).ok

    def test_returns_modelinfo(self, tmp_path):
        assert isinstance(inspect_model(tmp_path), ModelInfo)


class TestRealExamples:
    @pytest.mark.parametrize("rel, n, fv, corpus", [
        ("examples/lightgbm_ember2018/ember_model_2018.txt", 2381, "ember_v2", "ember_v2_2018"),
        ("examples/lightgbm_ember2024/EMBER2024_PE.model", 2568, "ember_v3", "ember_v3_2024"),
    ])
    def test_example(self, rel, n, fv, corpus):
        p = ROOT / rel
        if not p.is_file():
            pytest.skip(f"{rel} not present")
        info = inspect_model(p)
        assert info.ok, info.errors
        assert (info.model_kind, info.n_features, info.feature_version, info.default_corpus) == (
            "lightgbm", n, fv, corpus)
