"""``malvalid run --model`` (no --adapter) and ``malvalid inspect-model``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from malvalid import registry, runner
from malvalid.adapters.spec import SPEC_FILENAME, load_spec
from malvalid.cli import app
from tests.unit import model_factories as mf

cli = CliRunner()


@pytest.fixture(scope="module")
def mdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli_models")
    mf.save_lgbm(d / "m.txt", 2381)
    mf.save_lgbm(d / "m3.txt", 2568)
    mf.save_lgbm(d / "bad.txt", 100)
    mf.save_pickle(d / "g.pkl", 2381)
    (d / "garbage.dat").write_bytes(bytes(range(256)) * 3)
    (d / "train.txt").write_text("b" * 64 + "\n")
    return d


@pytest.fixture
def captured(monkeypatch):
    """Replace run_gate: capture the RunOptions, return a minimal outcome."""
    box: dict = {}

    def fake_run_gate(opts):
        box["opts"] = opts
        return SimpleNamespace(report={}, exit_code=0)

    monkeypatch.setattr(runner, "run_gate", fake_run_gate)
    monkeypatch.setattr("malvalid.cli.render_summary", lambda *a, **k: None)
    return box


def run(*args, **kw):
    return cli.invoke(app, ["run", *map(str, args)], **kw)


def spec_of(out: Path):
    return load_spec(out / "submission" / SPEC_FILENAME)


class TestRunModel:
    @pytest.mark.posix
    def test_threshold(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m.txt", "--threshold", "0.8", "--out", tmp_path / "o", "--no-html")
        assert res.exit_code == 0, res.output
        o = captured["opts"]
        sp = tmp_path / "o" / "submission" / SPEC_FILENAME
        assert Path(o.adapter) == sp and sp.is_file()
        s = spec_of(tmp_path / "o")
        assert (s.model_kind, s.feature_version, s.operating_threshold, s.threshold_source) == (
            "lightgbm", "ember_v2", 0.8, "declared")
        assert list(o.model_paths) == []
        assert (tmp_path / "o" / "submission" / s.model_file).is_symlink()
        assert "--model" in o.command and "--threshold" in o.command

    def test_calibrate_fpr(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m.txt", "--calibrate-fpr", "0.01", "--out", tmp_path / "o")
        assert res.exit_code == 0, res.output
        s = spec_of(tmp_path / "o")
        assert s.calibrate and s.calibrate_fpr == 0.01 and s.operating_threshold is None

    def test_calibration_period_default_and_flag(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m.txt", "--out", tmp_path / "a")
        assert res.exit_code == 0, res.output
        assert captured["opts"].config.runtime.calibration_period == "earliest"
        res = run("--model", mdir / "m.txt", "--calibrate-fpr", "0.01", "--calibration-period", "Uniform",
                  "--out", tmp_path / "b")
        assert res.exit_code == 0, res.output
        o = captured["opts"]
        assert o.config.runtime.calibration_period == "uniform"
        assert "--calibration-period uniform" in o.command

    @pytest.mark.parametrize("extra, msg", [
        (("--calibration-period", "latest"), "must be one of earliest | uniform"),
        (("--threshold", "0.8", "--calibration-period", "uniform"), "only applies to a calibrated threshold"),
    ])
    def test_calibration_period_usage_errors(self, mdir, tmp_path, captured, extra, msg):
        res = run("--model", mdir / "m.txt", *extra, "--out", tmp_path / "o")
        assert res.exit_code == 2 and msg in " ".join(res.output.split()), res.output
        assert "opts" not in captured

    def test_calibration_period_with_adapter_rejected(self, mdir, tmp_path, captured):
        res = run("--adapter", mdir / "m.txt", "--calibration-period", "uniform", "--out", tmp_path / "o")
        assert res.exit_code == 2 and "--calibration-period" in res.output and "opts" not in captured

    def test_default_calibrates_0005_with_note(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m.txt", "--out", tmp_path / "o")
        assert res.exit_code == 0, res.output
        s = spec_of(tmp_path / "o")
        assert s.calibrate and s.calibrate_fpr == 0.005
        assert any("calibrated" in n for n in captured["opts"].notes)
        assert "calibrated" in res.output

    def test_both_is_usage_error(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m.txt", "--threshold", "0.5", "--calibrate-fpr", "0.01", "--out", tmp_path / "o")
        assert res.exit_code == 2
        assert "either --threshold or --calibrate-fpr" in res.output
        assert "opts" not in captured

    def test_pickle_without_allow_pickle(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "g.pkl", "--threshold", "0.5", "--feature-version", "ember_v2",
                  "--out", tmp_path / "o")
        assert res.exit_code == 2 and "--allow-pickle" in res.output and "opts" not in captured

    def test_pickle_with_allow_pickle(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "g.pkl", "--threshold", "0.5", "--feature-version", "ember_v2",
                  "--allow-pickle", "--out", tmp_path / "o")
        assert res.exit_code == 0, res.output
        assert spec_of(tmp_path / "o").model_kind == "sklearn_gbdt" and captured["opts"].allow_pickle is True

    def test_spec_flags_with_adapter_rejected(self, mdir, tmp_path, captured):
        ad = tmp_path / "a.py"
        ad.write_text("# adapter\n")
        for flag, val in (("--threshold", "0.5"), ("--calibrate-fpr", "0.01"), ("--feature-version", "ember_v2"),
                          ("--model-kind", "lightgbm"), ("--training-cutoff", "2018-10"),
                          ("--training-hashes", str(mdir / "train.txt"))):
            res = run("--adapter", ad, "--model", mdir / "m.txt", flag, val, "--out", tmp_path / "o")
            assert res.exit_code == 2, (flag, res.output)
            assert flag in res.output and "without --adapter" in res.output
        assert "opts" not in captured

    def test_missing_model_and_multiple_models(self, mdir, tmp_path, captured):
        assert run("--out", tmp_path / "o").exit_code == 2
        res = run("--model", mdir / "m.txt", "--model", mdir / "m3.txt", "--threshold", "0.5")
        assert res.exit_code == 2 and "exactly one" in res.output

    def test_feature_count_mismatch(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "bad.txt", "--threshold", "0.5", "--out", tmp_path / "o")
        assert res.exit_code == 2
        assert "model expects 100 features" in res.output and "EMBER v2 (2381)" in res.output

    def test_feature_version_conflict(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m.txt", "--threshold", "0.5", "--feature-version", "ember_v3",
                  "--out", tmp_path / "o")
        assert res.exit_code == 2 and "2381" in res.output and "ember_v3" in res.output

    def test_garbage_model(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "garbage.dat", "--threshold", "0.5", "--out", tmp_path / "o")
        assert res.exit_code == 2 and "could not detect" in res.output

    def test_training_files(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m.txt", "--threshold", "0.5", "--training-hashes", mdir / "train.txt",
                  "--training-cutoff", "2018-10", "--out", tmp_path / "o")
        assert res.exit_code == 0, res.output
        s = spec_of(tmp_path / "o")
        assert s.training_cutoff == "2018-10" and s.training_hashes_file
        assert "--training-hashes" in captured["opts"].command

    def test_exit_code_propagates(self, mdir, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "run_gate", lambda o: SimpleNamespace(report={}, exit_code=1))
        monkeypatch.setattr("malvalid.cli.render_summary", lambda *a, **k: None)
        assert run("--model", mdir / "m.txt", "--threshold", "0.5", "--out", tmp_path / "o").exit_code == 1


class TestCorpusSwitch:
    def _policy(self, tmp_path, corpus):
        p = tmp_path / "gate.yaml"
        p.write_text(json.dumps({"corpus": corpus}))
        return p

    def test_v3_model_switches_corpus(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m3.txt", "--threshold", "0.5", "--out", tmp_path / "o")
        assert res.exit_code == 0, res.output
        opts = captured["opts"]
        assert opts.config.corpus == "ember_v3_2024"
        assert any("ember_v3_2024" in n and "ember_v3" in n for n in opts.notes)

    def test_v3_model_with_explicit_policy_corpus_v2_switches(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m3.txt", "--threshold", "0.5", "--config",
                  self._policy(tmp_path, "ember_v2_2018"), "--out", tmp_path / "o")
        assert res.exit_code == 0, res.output
        assert captured["opts"].config.corpus == "ember_v3_2024"

    def test_explicit_corpus_wins(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m3.txt", "--threshold", "0.5", "--corpus", "ember_v2_2018",
                  "--out", tmp_path / "o")
        assert res.exit_code == 0, res.output
        assert captured["opts"].config.corpus == "ember_v2_2018"
        assert not any("evaluated on the" in n for n in captured["opts"].notes)

    def test_v2_model_keeps_policy_corpus(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m.txt", "--threshold", "0.5", "--out", tmp_path / "o")
        assert res.exit_code == 0, res.output
        assert captured["opts"].config.corpus == "ember_v2_2018"

    def test_v2_model_keeps_synthetic_v2_policy(self, mdir, tmp_path, captured):
        res = run("--model", mdir / "m.txt", "--threshold", "0.5", "--config",
                  self._policy(tmp_path, "synthetic_v2"), "--out", tmp_path / "o")
        assert res.exit_code == 0, res.output
        assert captured["opts"].config.corpus == "synthetic_v2"


class TestInspectModelCmd:
    def test_json_ok(self, mdir):
        res = cli.invoke(app, ["inspect-model", str(mdir / "m.txt"), "--json"])
        assert res.exit_code == 0, res.output
        d = json.loads(res.output)
        assert d["ok"] is True and d["model_kind"] == "lightgbm" and d["n_features"] == 2381
        assert d["feature_version"] == "ember_v2" and d["default_corpus"] == "ember_v2_2018"
        assert {"format", "is_pickle", "errors", "notes", "details", "supported", "file_name"} <= set(d)

    def test_json_v3(self, mdir):
        d = json.loads(cli.invoke(app, ["inspect-model", str(mdir / "m3.txt"), "--json"]).output)
        assert d["feature_version"] == "ember_v3" and d["default_corpus"] == "ember_v3_2024"

    def test_json_needs_manual_choice(self, mdir):
        res = cli.invoke(app, ["inspect-model", str(mdir / "bad.txt"), "--json"])
        assert res.exit_code == 1
        d = json.loads(res.output)
        assert d["ok"] is False and d["feature_version"] is None
        assert any("model expects 100 features" in e for e in d["errors"])

    def test_json_pickle(self, mdir):
        res = cli.invoke(app, ["inspect-model", str(mdir / "g.pkl"), "--json"])
        assert res.exit_code == 1
        d = json.loads(res.output)
        assert d["is_pickle"] is True and d["model_kind"] == "sklearn_gbdt"

    def test_table_output(self, mdir):
        res = cli.invoke(app, ["inspect-model", str(mdir / "m.txt")])
        assert res.exit_code == 0
        assert "lightgbm" in res.output and "ember_v2" in res.output and "malvalid run --model" in res.output
        res = cli.invoke(app, ["inspect-model", str(mdir / "bad.txt")])
        assert res.exit_code == 1 and "100" in res.output

    def test_missing_file(self, tmp_path):
        res = cli.invoke(app, ["inspect-model", str(tmp_path / "nope.txt")])
        assert res.exit_code == 2 and "not found" in res.output


# --------------------------------------------------------------------------------------------------
# One real run: --model equals the equivalent adapter run, through the real sandbox
# --------------------------------------------------------------------------------------------------

_ADAPTER = '''
from malvalid.adapter import BaseDetector


class E2EDetector(BaseDetector):
    feature_version = "ember_v2"
    model_kind = "lightgbm"
    operating_threshold = {thr!r}
    model_path = "model.txt"
    training_hashes_path = "train_sha256.txt"
    training_cutoff = "2017-12"
'''


def _isolating() -> bool:
    try:
        from malvalid.sandbox.host import probe_backends

        return any(v.get("available") and v.get("network_isolated") for v in probe_backends().values())
    except Exception:  # noqa: BLE001
        return False


def _setup(tmp_path, monkeypatch, rows: str):
    """Synthetic corpus (private dir), a tiny LightGBM trained on its train split, adapter + policy files."""
    import lightgbm as lgb

    monkeypatch.setenv("MALVALID_CORPUS_DIR", str(tmp_path / "corpora"))
    monkeypatch.setenv("MALVALID_SYNTHETIC_ROWS", rows)
    monkeypatch.delenv("MALVALID_SYNTHETIC_SEED", raising=False)
    prov = registry.get_corpus_provider("synthetic_v2")
    prov = prov() if isinstance(prov, type) else prov
    corpus = prov.load()
    idx = np.flatnonzero((corpus.split == "train") & (corpus.label >= 0))
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1, "min_data_in_leaf": 10, "num_threads": 2,
         "seed": 0, "deterministic": True, "force_row_wise": True, "verbose": -1},
        lgb.Dataset(np.asarray(corpus.X[idx], dtype=np.float32), label=corpus.label[idx].astype(int)),
        num_boost_round=30)
    src = tmp_path / "src"
    src.mkdir()
    booster.save_model(str(src / "model.txt"))
    (src / "train_sha256.txt").write_text("\n".join(str(corpus.sha256[i]) for i in idx) + "\n")
    (src / "adapter.py").write_text(_ADAPTER.format(thr=0.5))
    mods = {m: {"enabled": False} for m in [*registry.modules(), *registry.unavailable("modules")]
            if m not in ("file_safety", "dummy")}
    mods["performance"] = {"enabled": True, "gate": "hard", "max_fpr": 0.15, "min_detection": 0.6}
    mods["drift"] = {"enabled": True, "min_window_samples": 40, "min_aut_f1": 0.5}
    cfg = tmp_path / "gate.yaml"
    cfg.write_text(json.dumps({"corpus": "synthetic_v2", "runtime": {"seed": 0, "threads": 2,
                                                                    "max_seconds_per_module": 300},
                               "modules": mods}))
    return src, cfg


def _go(tmp_path, cfg, name, *args, codes=(0, 1)):
    out = tmp_path / name
    res = cli.invoke(app, ["run", "--config", str(cfg), "--out", str(out), "--no-html", *args])
    assert res.exit_code in codes, res.output
    return res, (json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else None)




@pytest.mark.slow
@pytest.mark.skipif(not _isolating(), reason="no isolating sandbox backend on this host")
def test_model_run_equals_adapter_run(tmp_path, monkeypatch):
    src, cfg = _setup(tmp_path, monkeypatch, "4000")
    train = ("--training-hashes", str(src / "train_sha256.txt"), "--training-cutoff", "2017-12")
    _, a = _go(tmp_path, cfg, "adapter_run", "--adapter", str(src / "adapter.py"))
    _, m = _go(tmp_path, cfg, "model_run", "--model", str(src / "model.txt"), "--threshold", "0.5", *train)

    assert m["model"]["load_error"] is None, m["model"]
    assert m["verdict"]["verdict"] == a["verdict"]["verdict"]
    am = {x["module_id"]: x for x in a["modules"]}
    mm = {x["module_id"]: x for x in m["modules"]}
    assert list(am) == list(mm)
    assert mm["file_safety"]["status"] == "pass"
    for k in ("performance", "drift"):
        assert mm[k]["status"] == am[k]["status"], k
        assert mm[k]["score"] is not None
        assert mm[k]["score"] == pytest.approx(am[k]["score"], abs=1e-9), k
    assert m["model"]["extras"].get("threshold_source") != "calibrated"
    assert "calibration_holdout" not in m["corpus"]

    # the 4000-row corpus has too few benign rows to calibrate on: a clear refusal, not a crash
    res, _ = _go(tmp_path, cfg, "model_cal_small", "--model", str(src / "model.txt"), "--calibrate-fpr", "0.01",
                 *train, codes=(2,))
    assert "benign rows available for threshold calibration" in res.output and "--threshold" in res.output


@pytest.mark.slow
@pytest.mark.skipif(not _isolating(), reason="no isolating sandbox backend on this host")
def test_model_run_auto_calibrated(tmp_path, monkeypatch):
    src, cfg = _setup(tmp_path, monkeypatch, "16000")
    _, c = _go(tmp_path, cfg, "model_cal", "--model", str(src / "model.txt"), "--calibrate-fpr", "0.01",
               "--training-hashes", str(src / "train_sha256.txt"), "--training-cutoff", "2017-12")
    assert c["model"]["load_error"] is None, c["model"]
    assert c["model"]["extras"]["threshold_source"] == "calibrated"
    cal = c["model"]["extras"]["threshold_calibration"]
    assert cal["achieved_fpr"] <= 0.01 and c["model"]["operating_threshold"] == pytest.approx(cal["threshold"])
    assert c["corpus"]["calibration_holdout"]["n_rows"] >= 200
    assert any("auto-calibrated" in w for w in c["warnings"])
    mm = {x["module_id"]: x for x in c["modules"]}
    assert mm["performance"]["status"] in ("pass", "warn", "fail") and mm["performance"]["score"] is not None
