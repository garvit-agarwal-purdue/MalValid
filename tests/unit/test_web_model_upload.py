"""Model-file submissions in the web UI ("Upload your model", ``malvalid run --model``).

* ``POST /runs/inspect`` streams the model into a pending submission dir and runs
  ``malvalid inspect-model --json`` on it in a subprocess (HTML re-render with the detected values, or JSON);
* ``POST /runs`` with ``model_submission`` reuses that stored model (moved into the run's own submission),
  a newly chosen file wins, and only marked inspection dirs directly under ``submissions/`` qualify;
* the form -> argv mapping (``--model``, ``--threshold`` / ``--calibrate-fpr``, ``--training-hashes`` ...),
  strict validation, CSRF, cleanup of abandoned inspections, and the reverse-proxy prefix.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("starlette")

from malvalid.web.settings import WebSettings  # noqa: E402
from malvalid.web.store import INSPECT_MARKER  # noqa: E402
from malvalid.web.uploads import PICKLE_REFUSED  # noqa: E402
from tests.fixtures.adapters.builders import GETPID_PICKLE  # noqa: E402
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    FIXTURES,
    PORT,
    TOKEN,
    app,
    authed_client,
    captured,
    client,
    cli_run_dir,
    fake_malvalid,
    job_json,
    make_app,
    post_run,
    run_id_from,
    settings,
)

pytestmark = pytest.mark.web

JSON = {"accept": "application/json"}
HTML = {"accept": "text/html,application/xhtml+xml"}
LGBM_STUB = b"tree\nversion=v4\nnum_class=1\nmax_feature_idx=2380\n"


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _train_lgbm(path: Path, n_features: int) -> Path:
    np = pytest.importorskip("numpy")
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X = rng.random((60, n_features))
    y = (rng.random(60) > 0.5).astype(int)
    booster = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 2, "num_leaves": 4},
                        lgb.Dataset(X, y), num_boost_round=2)
    booster.save_model(str(path))
    return path


@pytest.fixture(scope="module")
def lgbm_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("lgbm")
    return {"v2": _train_lgbm(d / "model.txt", 2381), "small": _train_lgbm(d / "small.txt", 100)}


DETECTED = {"ok": True, "format": "lightgbm-text", "model_kind": "lightgbm", "n_features": 2381,
            "feature_version": "ember_v2", "default_corpus": "ember_v2_2018", "is_pickle": False,
            "details": {"lightgbm_version": "v4"}, "notes": [], "errors": [],
            "supported": {"2381": "ember_v2", "2568": "ember_v3"}, "exit_code": 0, "error": None}


@pytest.fixture
def inspect_calls(app: Any) -> list[Path]:
    """Replace the inspect-model subprocess with a canned result (records the inspected paths)."""
    calls: list[Path] = []

    def fake(model: Path, *, cwd: Path, timeout_s: float | None = None) -> dict[str, Any]:
        calls.append(Path(model))
        res = dict(DETECTED)
        if Path(model).suffix in (".pkl", ".joblib"):
            res.update(ok=False, format="pickle", is_pickle=True, model_kind="sklearn_gbdt", n_features=None,
                       feature_version=None, default_corpus=None)
        return res

    app.state.jobs.inspect = fake
    return calls


def subs(settings: WebSettings) -> list[Path]:
    d = settings.runs_dir / "submissions"
    return sorted(p for p in d.iterdir()) if d.exists() else []


def run_dirs(settings: WebSettings) -> list[Path]:
    return sorted(p for p in settings.runs_dir.iterdir() if p.name != "submissions") if settings.runs_dir.exists() else []


def inspect_post(client, settings, data: dict[str, Any] | None = None, files=None, *, url: str = "/runs/inspect",
                 headers: dict[str, str] | None = None, csrf: bool = True):
    return post_run(client, settings, {"mode": "model", **(data or {})}, files, url=url, csrf=csrf,
                    headers=headers or HTML)


def model_run(client, settings, data: dict[str, Any] | None = None, files=None, **kw):
    return post_run(client, settings, {"mode": "model", **(data or {})}, files, **kw)


def argv_value(argv: list[str], flag: str) -> str:
    assert flag in argv, (flag, argv)
    return argv[argv.index(flag) + 1]


def form_errors(resp, app, *, status: int = 400) -> list[str]:
    assert resp.status_code == status, resp.text[:800]
    return captured(app, "new_run.html.j2")["errors"]


# --------------------------------------------------------------------------------------------------
# The real subprocess end to end
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def real_app(tmp_path: Path) -> Any:
    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=tmp_path / "runs", token=TOKEN, port=PORT, command_prefix=(), validate_timeout_s=120)
    return make_app(s)


def test_inspect_runs_the_real_inspect_model_subprocess(real_app, lgbm_models):
    s = real_app.state.settings
    with authed_client(real_app) as c:
        r = inspect_post(c, s, {"title": "My LightGBM", "seed": "7", "threshold_mode": "declared", "threshold": "0.8"},
                         [("model_file", ("model.txt", lgbm_models["v2"].read_bytes())),
                          ("manifest_file", ("hashes.txt", b"a" * 64 + b"\n"))])
    assert r.status_code == 200, r.text[:2000]
    ctx = captured(real_app, "new_run.html.j2")
    insp = ctx["inspect"]
    assert insp["model_kind"] == "lightgbm" and insp["feature_version"] == "ember_v2"
    assert insp["n_features"] == 2381 and insp["default_corpus"] == "ember_v2_2018" and insp["ok"] is True
    sid = insp["model_submission"]
    f = ctx["form"]
    # detected values pre-selected, the user's own fields kept
    assert f["model_kind"] == "lightgbm" and f["feature_version"] == "ember_v2"
    assert f["title"] == "My LightGBM" and f["seed"] == "7" and f["threshold"] == "0.8"
    assert f["threshold_mode"] == "declared" and f["model_submission"] == sid and f["mode"] == "model"
    html = r.text
    assert "detected from model" in html and "2381 features" in html and "ember_v2_2018" in html
    assert f'name="model_submission" value="{sid}"' in html
    assert re.search(r"Using uploaded <strong>model\.txt</strong> \([^)]+\) — choose a file to replace it", html)
    assert re.search(r'<option value="ember_v2" selected>', html) and re.search(r'<option value="lightgbm" selected>', html)
    assert 'value="My LightGBM"' in html
    # stored: the model + marker + saved result; the manifest is dropped (and the user told so)
    d = s.runs_dir / "submissions" / sid
    assert sorted(p.name for p in d.iterdir()) == [".inspect-pending", ".inspect.json", "model.txt"]
    assert "hashes.txt" in (ctx["notice"] or "")
    assert run_dirs(s) == []


def test_real_inspection_of_a_feature_mismatch_shows_the_error(real_app, lgbm_models):
    s = real_app.state.settings
    with authed_client(real_app) as c:
        r = inspect_post(c, s, files=[("model_file", ("small.txt", lgbm_models["small"].read_bytes()))])
        assert r.status_code == 200
        insp = captured(real_app, "new_run.html.j2")["inspect"]
        assert insp["n_features"] == 100 and insp["feature_version"] is None and insp["ok"] is False
        assert any("100 features" in e for e in insp["errors"])
        assert "model expects 100 features; malvalid supports EMBER v2 (2381) and EMBER v3 (2568)" in r.text
        assert "cannot be evaluated as it is" in r.text
        # the same, as JSON
        r = inspect_post(c, s, files=[("model_file", ("small.txt", lgbm_models["small"].read_bytes()))], headers=JSON)
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
        d = r.json()
        assert d["n_features"] == 100 and d["feature_version"] is None and d["model_submission"]
        assert "path" not in d  # server paths stay on the server


def test_inspection_failure_is_reported_not_raised(app, client, settings):
    # the test settings' malvalid is a stand-in that knows no inspect-model: the error comes back
    r = inspect_post(client, settings, files=[("model_file", ("model.txt", LGBM_STUB))])
    assert r.status_code == 200
    insp = captured(app, "new_run.html.j2")["inspect"]
    assert insp["ok"] is False and "fake malvalid only knows" in insp["error"]
    assert "Could not inspect model.txt" in r.text


# --------------------------------------------------------------------------------------------------
# Inspect (mocked subprocess)
# --------------------------------------------------------------------------------------------------


def test_inspect_json_answer(client, settings, inspect_calls):
    r = inspect_post(client, settings, files=[("model_file", ("m.json", b'{"learner": {}}')),
                                              ("config_file", ("gate.yaml", b"corpus: x\n"))], headers=JSON)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model_kind"] == "lightgbm" and d["feature_version"] == "ember_v2" and d["n_features"] == 2381
    assert d["file_name"] == "m.json" and d["size"] == len(b'{"learner": {}}') and d["discarded"] == ["gate.yaml"]
    sdir = settings.runs_dir / "submissions" / d["model_submission"]
    assert inspect_calls == [sdir / "m.json"]
    assert (sdir / INSPECT_MARKER).is_file() and not (sdir / "gate.yaml").exists()


def test_inspect_needs_a_model(client, app, settings, inspect_calls):
    r = inspect_post(client, settings)
    assert "choose your model file" in " ".join(form_errors(r, app))
    r = inspect_post(client, settings, files=[("model_file", ("model.exe", b"MZ"))])
    assert any("only" in e and ".onnx" in e for e in form_errors(r, app))
    r = inspect_post(client, settings, files=[("model_file", ("model.txt", b""))])
    assert any("empty" in e for e in form_errors(r, app))
    assert inspect_calls == [] and subs(settings) == []


def test_reinspect_without_a_new_file_uses_the_stored_model(client, app, settings, inspect_calls):
    sid = inspect_post(client, settings, files=[("model_file", ("model.txt", LGBM_STUB))], headers=JSON).json()[
        "model_submission"]
    r = inspect_post(client, settings, {"model_submission": sid})
    assert r.status_code == 200
    assert captured(app, "new_run.html.j2")["inspect"]["model_submission"] == sid
    assert len(inspect_calls) == 2 and inspect_calls[0] == inspect_calls[1]
    assert [p.name for p in subs(settings)] == [sid]
    # a new file replaces the earlier inspection
    sid2 = inspect_post(client, settings, {"model_submission": sid},
                        [("model_file", ("other.txt", LGBM_STUB))], headers=JSON).json()["model_submission"]
    assert sid2 != sid and [p.name for p in subs(settings)] == [sid2]


def test_inspect_requires_csrf(client, settings, inspect_calls):
    for kw in ({"csrf": False}, {}):
        data = {} if kw else {"csrf": "wrong-token"}
        r = inspect_post(client, settings, data, [("model_file", ("model.txt", LGBM_STUB))], **kw)
        assert r.status_code == 403
    assert inspect_calls == [] and subs(settings) == []


def test_pickle_inspection_warns_and_does_not_choose_the_feature_version(client, app, settings, inspect_calls):
    r = inspect_post(client, settings, files=[("model_file", ("model.pkl", GETPID_PICKLE))])
    assert r.status_code == 200
    ctx = captured(app, "new_run.html.j2")
    assert ctx["inspect"]["is_pickle"] is True
    assert ctx["form"].get("feature_version") in (None, "")  # never guessed for a pickle
    assert "never unpickles" in r.text


def test_abandoned_inspections_are_cleaned_up(client, settings, inspect_calls):
    old = inspect_post(client, settings, files=[("model_file", ("a.txt", LGBM_STUB))], headers=JSON).json()["model_submission"]
    young = inspect_post(client, settings, files=[("model_file", ("b.txt", LGBM_STUB))], headers=JSON).json()["model_submission"]
    old_marker = settings.runs_dir / "submissions" / old / INSPECT_MARKER
    t = time.time() - 25 * 3600
    os.utime(old_marker, (t, t))
    unmarked = settings.runs_dir / "submissions" / "20200101T000000Z-deadbeef"  # a run's own submission
    unmarked.mkdir()
    os.utime(unmarked, (t, t))
    inspect_post(client, settings, files=[("model_file", ("c.txt", LGBM_STUB))], headers=JSON)
    names = {p.name for p in subs(settings)}
    assert old not in names and young in names and unmarked.name in names and len(names) == 3


def test_pending_inspections_are_not_runs(client, settings, inspect_calls):
    inspect_post(client, settings, files=[("model_file", ("a.txt", LGBM_STUB))])
    assert client.get("/api/runs").json() == []
    assert client.get("/", headers=HTML).status_code == 200


def test_validate_adapter_is_not_for_model_files(client, settings):
    r = post_run(client, settings, {"mode": "model"}, [("model_file", ("model.txt", LGBM_STUB))],
                 url="/api/validate", headers=JSON)
    assert r.status_code == 400 and "Inspect model" in r.json()["errors"][0]


# --------------------------------------------------------------------------------------------------
# Start run: argv, reuse of the inspected model
# --------------------------------------------------------------------------------------------------


def test_start_run_reuses_the_inspected_model(client, settings, inspect_calls):
    sid = inspect_post(client, settings, files=[("model_file", ("model.txt", LGBM_STUB))], headers=JSON).json()[
        "model_submission"]
    r = model_run(client, settings, {"model_submission": sid, "model_kind": "lightgbm", "feature_version": "",
                                     "threshold_mode": "declared", "threshold": "0.42", "calibrate_fpr": "0.005",
                                     "training_cutoff": "2018-10", "title": "rc-2"},
                  [("manifest_file", ("train_sha256.txt", b"a" * 64 + b"\n"))])
    rid = run_id_from(r)
    job = job_json(settings, rid)
    new_sdir = settings.runs_dir / "submissions" / job["submission_id"]
    argv = job["argv"]
    assert argv[2:4] == ["run", "--model"] and "--adapter" not in argv and "--class" not in argv
    assert argv_value(argv, "--model") == str(new_sdir / "model.txt")
    assert argv_value(argv, "--model-kind") == "lightgbm" and "--feature-version" not in argv
    assert argv_value(argv, "--threshold") == "0.42" and "--calibrate-fpr" not in argv
    assert argv_value(argv, "--training-cutoff") == "2018-10"
    assert argv_value(argv, "--training-hashes") == str(new_sdir / "train_sha256.txt")
    assert argv.index("--out") > argv.index("--training-hashes") and argv_value(argv, "--run-id") == rid
    assert (new_sdir / "model.txt").read_bytes() == LGBM_STUB
    assert not (settings.runs_dir / "submissions" / sid).exists()  # moved, the inspection dir is gone
    assert job["mode"] == "model" and job["adapter"] == str(new_sdir / "model.txt")
    o = job["options"]
    assert o["submission_kind"] == "model" and o["model_kind"] == "lightgbm" and o["feature_version"] is None
    assert o["threshold_mode"] == "declared" and o["threshold"] == 0.42 and o["calibrate_fpr"] is None
    assert o["training_cutoff"] == "2018-10"
    assert o["files"] == {"model": "model.txt", "manifest": "train_sha256.txt"}
    assert job["display_name"] == "rc-2"
    # the same id cannot be used twice
    r = model_run(client, settings, {"model_submission": sid})
    assert r.status_code == 400 and "no longer on the server" in r.text


def test_start_run_without_inspecting_calibrates_by_default(client, settings):
    rid = run_id_from(model_run(client, settings, files=[("model_file", ("model.onnx", b"\x08\x07onnx"))]))
    job = job_json(settings, rid)
    argv = job["argv"]
    assert argv_value(argv, "--calibrate-fpr") == "0.005" and "--threshold" not in argv
    assert "--model-kind" not in argv and "--feature-version" not in argv and "--training-hashes" not in argv
    assert job["options"]["threshold_mode"] == "calibrate" and job["options"]["calibrate_fpr"] == 0.005
    assert job["display_name"] == "model.onnx"


@pytest.mark.parametrize("fpr, flag", [("0.001", "0.001"), ("0.01", "0.01"), ("0.005", "0.005")])
def test_calibration_targets(client, settings, fpr, flag):
    rid = run_id_from(model_run(client, settings, {"threshold_mode": "calibrate", "calibrate_fpr": fpr,
                                                   "threshold": "0.9"},
                                [("model_file", ("model.txt", LGBM_STUB))]))
    argv = job_json(settings, rid)["argv"]
    assert argv_value(argv, "--calibrate-fpr") == flag and "--threshold" not in argv


def test_a_new_model_file_wins_over_the_inspected_one(client, settings, inspect_calls):
    sid = inspect_post(client, settings, files=[("model_file", ("old.txt", LGBM_STUB))], headers=JSON).json()[
        "model_submission"]
    rid = run_id_from(model_run(client, settings, {"model_submission": sid},
                                [("model_file", ("new.txt", LGBM_STUB + b"\n"))]))
    job = job_json(settings, rid)
    assert Path(argv_value(job["argv"], "--model")).name == "new.txt"
    assert not (settings.runs_dir / "submissions" / sid).exists()


def test_mode_is_inferred_from_the_files(client, settings):
    rid = run_id_from(post_run(client, settings, {}, [("model_file", ("model.txt", LGBM_STUB))]))
    job = job_json(settings, rid)
    assert job["mode"] == "model" and "--model" in job["argv"]


def test_unused_adapter_files_are_not_kept_in_model_mode(client, settings):
    rid = run_id_from(model_run(client, settings, files=[("model_file", ("model.txt", LGBM_STUB)),
                                                         ("adapter_file", ("stray.py", b"x = 1\n"))]))
    sdir = settings.runs_dir / "submissions" / job_json(settings, rid)["submission_id"]
    assert sorted(p.name for p in sdir.iterdir()) == ["model.txt"]


def test_path_mode_with_a_model_file(client, settings, tmp_path):
    m = tmp_path / "models" / "lgbm.txt"
    m.parent.mkdir()
    m.write_bytes(LGBM_STUB)
    h = tmp_path / "models" / "train.txt"
    h.write_text("a" * 64 + "\n")
    rid = run_id_from(post_run(client, settings, {"mode": "path", "adapter_path": str(m), "feature_version": "ember_v2",
                                                  "threshold_mode": "calibrate", "calibrate_fpr": "0.01",
                                                  "training_hashes_path": str(h), "training_cutoff": "2018-10-31"}))
    job = job_json(settings, rid)
    argv = job["argv"]
    assert argv_value(argv, "--model") == str(m.resolve()) and argv_value(argv, "--training-hashes") == str(h.resolve())
    assert argv_value(argv, "--feature-version") == "ember_v2" and argv_value(argv, "--calibrate-fpr") == "0.01"
    assert argv_value(argv, "--training-cutoff") == "2018-10-31"
    assert job["mode"] == "path" and job["adapter"] == str(m.resolve()) and job["submission_id"] is None
    assert job["options"]["training_hashes"] == str(h.resolve())


def test_path_mode_adapter_rejects_model_settings(client, app, settings, tmp_path):
    ad = tmp_path / "adapter.py"
    ad.write_text("class A: pass\n")
    r = post_run(client, settings, {"mode": "path", "adapter_path": str(ad), "threshold_mode": "declared",
                                    "threshold": "0.5"})
    assert any("only apply to a model file" in e for e in form_errors(r, app))
    # the form's defaults (auto, calibrate 0.5%) are fine with an adapter
    r = post_run(client, settings, {"mode": "path", "adapter_path": str(ad), "threshold_mode": "calibrate",
                                    "calibrate_fpr": "0.005", "model_kind": "", "feature_version": ""})
    assert "--adapter" in job_json(settings, run_id_from(r))["argv"]


# --------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("data, needle", [
    ({"threshold_mode": "declared", "threshold": "1.5"}, "number from 0 to 1"),
    ({"threshold_mode": "declared", "threshold": "-0.1"}, "number from 0 to 1"),
    ({"threshold_mode": "declared", "threshold": "nan"}, "number from 0 to 1"),
    ({"threshold_mode": "declared", "threshold": "inf"}, "number from 0 to 1"),
    ({"threshold_mode": "declared", "threshold": "abc"}, "number from 0 to 1"),
    ({"threshold_mode": "declared", "threshold": ""}, "enter the operating threshold"),
    ({"threshold_mode": "sometimes"}, "threshold choice"),
    ({"calibrate_fpr": "0.2"}, "calibration target"),
    ({"calibrate_fpr": "0.05"}, "calibration target"),
    ({"model_kind": "pytorch"}, "model kind must be"),
    ({"feature_version": "ember_v9"}, "feature version must be"),
    ({"training_cutoff": "2018-13"}, "training cutoff"),
    ({"training_cutoff": "18-10"}, "training cutoff"),
    ({"training_cutoff": "2018-02-30"}, "training cutoff"),
    ({"training_cutoff": "2018-10; rm -rf /"}, "training cutoff"),
    ({"class_name": "MyDetector"}, "adapter class only applies"),
])
def test_model_form_validation(client, app, settings, data, needle):
    r = model_run(client, settings, data, [("model_file", ("model.txt", LGBM_STUB))])
    errs = form_errors(r, app)
    assert any(needle in e for e in errs), errs
    assert run_dirs(settings) == [] and subs(settings) == []


def test_model_mode_needs_a_model(client, app, settings):
    assert any("choose your model file" in e for e in form_errors(model_run(client, settings), app))


def test_pickles_need_allow_pickle_and_explicit_choices(client, app, settings):
    files = [("model_file", ("model.pkl", GETPID_PICKLE))]
    errs = form_errors(model_run(client, settings, files=files), app)
    assert PICKLE_REFUSED.format(name="model.pkl") in errs
    errs = form_errors(model_run(client, settings, {"allow_pickle": "on"}, files), app)
    assert any("choose its model kind and feature version" in e for e in errs)
    errs = form_errors(model_run(client, settings, {"allow_pickle": "on", "model_kind": "sklearn_gbdt"}, files), app)
    assert any("choose its model kind and feature version" in e for e in errs)
    rid = run_id_from(model_run(client, settings, {"allow_pickle": "on", "model_kind": "sklearn_gbdt",
                                                   "feature_version": "ember_v2"}, files))
    argv = job_json(settings, rid)["argv"]
    assert "--allow-pickle" in argv and argv_value(argv, "--model-kind") == "sklearn_gbdt"
    # sniffed by content, whatever the extension
    errs = form_errors(model_run(client, settings, files=[("model_file", ("model.txt", GETPID_PICKLE))]), app)
    assert PICKLE_REFUSED.format(name="model.txt") in errs


@pytest.mark.parametrize("sid", ["../x", "..", ".", "submissions", "/etc", "a/b", "..%2Fx", "x" * 300, "",
                                 "20200101T000000Z-unmarked", "20200101T000000Z-symlink", "20200101T000000Z-twofiles"])
@pytest.mark.posix
def test_model_submission_ids_are_checked(client, app, settings, tmp_path, inspect_calls, sid):
    root = settings.runs_dir / "submissions"
    root.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "secret.txt").write_text("secret")
    (victim / INSPECT_MARKER).write_text("x")
    (root / "20200101T000000Z-unmarked").mkdir()
    (root / "20200101T000000Z-unmarked" / "model.txt").write_bytes(LGBM_STUB)  # a run's submission: no marker
    os.symlink(victim, root / "20200101T000000Z-symlink")
    two = root / "20200101T000000Z-twofiles"
    two.mkdir()
    for n in ("a.txt", "b.txt", INSPECT_MARKER):
        (two / n).write_bytes(LGBM_STUB)
    r = model_run(client, settings, {"model_submission": sid})
    errs = form_errors(r, app)
    assert any("no longer on the server" in e or "choose your model file" in e for e in errs), errs
    r = inspect_post(client, settings, {"model_submission": sid})
    assert r.status_code == 400
    assert inspect_calls == []
    assert (victim / "secret.txt").read_text() == "secret"
    assert (root / "20200101T000000Z-unmarked" / "model.txt").exists() and len(list(two.iterdir())) == 3
    assert run_dirs(settings) == []


def test_errors_after_inspection_keep_the_stored_model(client, app, settings, inspect_calls):
    sid = inspect_post(client, settings, files=[("model_file", ("model.txt", LGBM_STUB))], headers=JSON).json()[
        "model_submission"]
    r = model_run(client, settings, {"model_submission": sid, "threshold_mode": "declared", "threshold": "7"})
    assert r.status_code == 400
    ctx = captured(app, "new_run.html.j2")
    assert ctx["form"]["model_submission"] == sid and ctx["inspect"]["file_name"] == "model.txt"
    assert f'name="model_submission" value="{sid}"' in r.text and "Using uploaded" in r.text
    assert (settings.runs_dir / "submissions" / sid / "model.txt").exists()
    rid = run_id_from(model_run(client, settings, {"model_submission": sid, "threshold_mode": "declared",
                                                   "threshold": "0.7"}))
    assert argv_value(job_json(settings, rid)["argv"], "--threshold") == "0.7"


# --------------------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------------------


def test_new_run_page_offers_the_model_flow(client):
    html = client.get("/runs/new", headers=HTML).text
    assert re.search(r'name="mode" value="model"[^>]*checked', html)
    assert 'formaction="/runs/inspect"' in html and 'name="model_file"' in html
    for name in ("model_kind", "feature_version", "threshold_mode", "threshold", "calibrate_fpr", "training_cutoff",
                 "manifest_file", "config_file", "adapter_file", "model_files", "adapter_path"):
        assert f'name="{name}"' in html, name
    assert 'type="month"' in html and "held-out slice of the corpus" in html and "no test uses" in html
    assert re.search(r'<option value="0.005" selected>0.5% false positives', html)
    assert "Auto (matches the model&#39;s feature version)" in html or "Auto (matches the model's feature version)" in html
    assert "Advanced: own adapter" in html
    assert "malvalid run --model my_model.txt" in html


def test_rerun_copies_the_model_settings(client, app, settings):
    rid = run_id_from(model_run(client, settings, {"model_kind": "xgboost", "feature_version": "ember_v3",
                                                   "threshold_mode": "declared", "threshold": "0.25",
                                                   "training_cutoff": "2023-09"},
                                [("model_file", ("model.json", b"{}"))]))
    client.get(f"/runs/new?from={rid}", headers=HTML)
    ctx = captured(app, "new_run.html.j2")
    f = ctx["form"]
    assert f["mode"] == "model" and f["model_kind"] == "xgboost" and f["feature_version"] == "ember_v3"
    assert f["threshold_mode"] == "declared" and f["threshold"] == "0.25" and f["training_cutoff"] == "2023-09"
    assert "model.json" in ctx["notice"]


def test_run_page_names_the_model_and_the_threshold_choice(client, settings):
    rid = run_id_from(model_run(client, settings, {"calibrate_fpr": "0.001"}, [("model_file", ("model.txt", LGBM_STUB))]))
    html = client.get(f"/runs/{rid}", headers=HTML).text
    assert "Uploaded model file" in html and "auto-calibrated to 0.1% FPR on a held-out slice of the corpus" in html


def test_run_page_shows_a_calibrated_threshold(client, settings):
    report = json.loads((FIXTURES / "sample_report.json").read_text())
    report.setdefault("model", {})["operating_threshold"] = 0.935856
    report["model"]["extras"] = {"threshold_source": "calibrated", "threshold_calibration": {
        "target_fpr": 0.005, "achieved_fpr": 0.0048, "n_benign": 625, "corpus": "ember_v2_2018",
        "rule": "benign rows of the corpus eval role (test) with int(sha256[:8], 16) % 10 == 0"}}
    cli_run_dir(settings.runs_dir, "cal-run", report__json=report)
    html = client.get("/runs/cal-run", headers=HTML).text
    assert "auto-calibrated (target FPR 0.5%, on 625 held-out benign rows" in html
    assert "excluded from every test" in html


def test_dashboard_onboarding_starts_with_the_model_file(client):
    text = client.get("/", headers=HTML).text
    assert "malvalid inspect-model my_model.txt" in text and "malvalid run --model my_model.txt" in text
    assert "Advanced: own adapter" in text


# --------------------------------------------------------------------------------------------------
# Behind a reverse proxy (--root-path)
# --------------------------------------------------------------------------------------------------


ROOT = "/node/b002.example.edu/8765"
PUBLIC = "https://gateway.example.edu"


@pytest.fixture
def proxied(tmp_path: Path, fake_malvalid: Path):
    from starlette.testclient import TestClient

    s = WebSettings(allow_path_mode=True, allow_no_sandbox=True, runs_dir=tmp_path / "runs", token=TOKEN, port=PORT, cancel_grace_s=1.0,
                    command_prefix=(sys.executable, str(fake_malvalid)), validate_timeout_s=30,
                    root_path=ROOT + "/", trusted_hosts=("gateway.example.edu",))
    a = make_app(s)
    a.state.jobs.inspect = lambda model, *, cwd, timeout_s=None: dict(DETECTED)
    with TestClient(a, base_url=PUBLIC) as c:
        c.cookies.set(s.cookie_name, s.sessions.new_session())
        yield a, s, c


def test_inspect_and_submit_under_the_prefix(proxied):
    a, s, c = proxied
    html = c.get(ROOT + "/runs/new", headers=HTML).text
    assert f'action="{ROOT}/runs"' in html and f'formaction="{ROOT}/runs/inspect"' in html
    r = inspect_post(c, s, files=[("model_file", ("model.txt", LGBM_STUB))], url=f"{ROOT}/runs/inspect",
                     headers={"origin": PUBLIC, **HTML})
    assert r.status_code == 200, r.text[:1000]
    assert f'action="{ROOT}/runs"' in r.text and f'formaction="{ROOT}/runs/inspect"' in r.text
    sid = captured(a, "new_run.html.j2")["inspect"]["model_submission"]
    r = model_run(c, s, {"model_submission": sid}, url=f"{ROOT}/runs", headers={"origin": PUBLIC, **HTML})
    assert r.status_code == 303, r.text[:1000]
    assert re.fullmatch(re.escape(ROOT) + r"/runs/[A-Za-z0-9][A-Za-z0-9._-]*", r.headers["location"])


def test_js_inspects_under_the_root_path():
    js = (Path(__file__).resolve().parents[2] / "src/malvalid/web/static/app.js").read_text(encoding="utf-8")
    assert 'appUrl("/runs/inspect")' in js


# --------------------------------------------------------------------------------------------------
# Template fixtures (tests/fixtures/web_contexts.py)
# --------------------------------------------------------------------------------------------------


def test_inspected_form_template():
    from tests.fixtures import web_contexts as W
    from tests.unit.test_web_templates import check_page, make_env, render

    env = make_env()
    template, ctx = W.get("new_run_inspected")
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    assert p.find("input", name="mode", value="model", checked=True)
    assert p.find("input", name="model_submission")[0]["value"] == "20260930T101500Z-1a2b3c4d"
    assert p.find("option", value="lightgbm", selected=True) and p.find("option", value="ember_v2", selected=True)
    assert p.find("input", name="threshold_mode", value="declared", checked=True)
    assert p.find("input", id="threshold")[0]["value"] == "0.8336"
    assert p.find("input", id="training_cutoff")[0]["value"] == "2018-10"
    assert "detected from model: lightgbm" in p.all_text and "detected from model: ember_v2 (2381 features)" in p.all_text
    assert re.search(r"Using uploaded\s+lgbm_rc2\.txt\s+\(766\.1 KiB\) — choose a file to replace it\.", p.all_text)
    assert "malvalid run --model lgbm_rc2.txt" in p.all_text
    template, ctx = W.get("new_run_inspected_hostile")
    out = render(env, template, ctx)
    p = check_page(out, ctx)
    assert p.find("input", name="model_submission")[0]["value"] == W.HOSTILE_ATTR
    assert not any(k == "onmouseover" for _, a in p.tags for k in a)
