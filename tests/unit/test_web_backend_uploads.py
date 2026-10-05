"""New-run form of the local web UI: streamed, size-capped uploads with sanitized names and
extension allowlists, pickle refusal (by extension or content) unless ``allow_pickle`` is ticked,
path mode, option validation, the ``malvalid run`` argv, sticky form values after an error, and
``POST /api/validate`` (``malvalid validate-adapter --json`` in a subprocess)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("starlette")

import malvalid.testing  # noqa: E402,F401 - registers the toy_v1 schema and the toy_v1_corpus corpus
from malvalid.web.settings import WebSettings  # noqa: E402
from malvalid.web.uploads import PICKLE_REFUSED, sanitize_filename  # noqa: E402
from tests.fixtures.adapters.builders import GETPID_PICKLE  # noqa: E402
from tests.unit.test_web_backend_support import (  # noqa: E402,F401 - fixtures
    TOKEN,
    app,
    authed_client,
    captured,
    client,
    fake_adapter,
    fake_malvalid,
    fake_record,
    job_json,
    make_app,
    post_run,
    read_record,
    run_id_from,
    settings,
    wait_for,
)

pytestmark = pytest.mark.web

ADAPTER = b"# FAKE: ok\nclass Detector:\n    pass\n"
LGBM_TEXT = b"tree\nversion=v4\nnum_class=1\n"


def submissions(settings: WebSettings) -> list[Path]:
    d = settings.runs_dir / "submissions"
    return sorted(d.iterdir()) if d.exists() else []


def run_dirs(settings: WebSettings) -> list[Path]:
    return sorted(p for p in settings.runs_dir.iterdir() if p.name != "submissions") if settings.runs_dir.exists() else []


def upload(client, settings, *, adapter=("my_adapter.py", ADAPTER), models=(("model.txt", LGBM_TEXT),),
           data: dict[str, Any] | None = None, extra: list | None = None, **kw):
    files = []
    if adapter is not None:
        files.append(("adapter_file", adapter))
    for m in models:
        files.append(("model_files", m))
    files += extra or []
    return post_run(client, settings, {"mode": "upload", **(data or {})}, files, **kw)


def assert_form_error(resp, app, *, status: int = 400, contains: str = "") -> dict[str, Any]:
    assert resp.status_code == status, resp.text[:500]
    ctx = captured(app, "new_run.html.j2")
    assert ctx["errors"], "no error shown"
    if contains:
        assert any(contains in e for e in ctx["errors"]), ctx["errors"]
    return ctx


# --------------------------------------------------------------------------------------------------
# File names
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("model.txt", "model.txt"),
    ("../../etc/passwd", "passwd"),
    ("/abs/path/adapter.py", "adapter.py"),
    ("C:\\Users\\me\\my adapter.py", "my_adapter.py"),
    ("we$ird na;me (1).json", "we_ird_na_me_1_.json"),
    ("mödel.onnx", "m_del.onnx"),
    ("a\x00b.txt", "a_b.txt"),
    (".bashrc", None),
    ("../.ssh", None),
    ("", None),
    ("   ", None),
    ("dir/", None),
    ("...", None),
    ("$$$", None),
    (None, None),
])
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_caps_the_length_and_keeps_the_extension():
    name = sanitize_filename("x" * 500 + ".json")
    assert name is not None and len(name) == 128 and name.endswith(".json")


def test_uploaded_names_are_sanitized_and_stay_in_the_submission_dir(client, app, settings, tmp_path):
    r = upload(client, settings, adapter=("../../../evil adapter.py", ADAPTER),
               models=(("..\\..\\m o d e l.txt", LGBM_TEXT),),
               extra=[("manifest_file", ("/etc/train hashes.csv", b"sha256\n" + b"a" * 64 + b"\n"))])
    rid = run_id_from(r)
    job = job_json(settings, rid)
    sub = settings.runs_dir / "submissions" / job["submission_id"]
    assert sorted(p.name for p in sub.iterdir()) == ["evil_adapter.py", "m_o_d_e_l.txt", "train_hashes.csv"]
    assert job["adapter"] == str(sub / "evil_adapter.py")
    assert job["argv"][job["argv"].index("--adapter") + 1] == str(sub / "evil_adapter.py")
    assert job["mode"] == "upload"
    assert job["options"]["files"] == {"adapter": "evil_adapter.py", "models": ["m_o_d_e_l.txt"],
                                       "manifest": "train_hashes.csv"}
    assert not (tmp_path / "evil adapter.py").exists() and not (settings.runs_dir / "evil_adapter.py").exists()


def test_unusable_or_duplicate_names_are_refused(client, app, settings):
    r = upload(client, settings, models=((".hidden.txt", LGBM_TEXT),))
    assert_form_error(r, app, contains="not a usable file name")
    r = upload(client, settings, models=(("m.txt", LGBM_TEXT), ("m.txt", LGBM_TEXT)))
    assert_form_error(r, app, contains="both named m.txt")
    r = upload(client, settings, extra=[("config_file", ("a.yaml", b"corpus: x\n")), ("config_file", ("b.yaml", b"x: 1\n"))])
    assert_form_error(r, app, contains="only one")
    assert submissions(settings) == [] and run_dirs(settings) == []


# --------------------------------------------------------------------------------------------------
# Extension allowlists, size cap, content checks
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("field,name", [
    ("adapter_file", "adapter.txt"),
    ("adapter_file", "adapter.pyc"),
    ("adapter_file", "adapter.py.exe"),
    ("model_files", "model.exe"),
    ("model_files", "model.dll"),
    ("model_files", "model"),
    ("model_files", "model.pt"),
    ("manifest_file", "hashes.json"),
    ("config_file", "gate.json"),
    ("config_file", "gate.py"),
])
def test_extension_allowlists(client, app, settings, field, name):
    if field == "adapter_file":
        r = upload(client, settings, adapter=(name, ADAPTER))
    else:
        r = upload(client, settings, extra=[(field, (name, b"data\n"))])
    ctx = assert_form_error(r, app, contains="files are accepted here")
    assert ctx["form"]["mode"] == "upload"
    assert submissions(settings) == [] and run_dirs(settings) == []


@pytest.mark.parametrize("name", ["model.txt", "model.json", "model.ubj", "model.model", "model.onnx", "MODEL.TXT"])
def test_accepted_model_extensions(client, settings, name):
    run_id_from(upload(client, settings, models=((name, LGBM_TEXT),)))


def test_size_cap_is_enforced_while_streaming(tmp_path, fake_malvalid):
    s = WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=8765, max_upload_mb=1,
                    command_prefix=(sys.executable, str(fake_malvalid)))
    app = make_app(s)
    with authed_client(app) as c:
        big = b"0" * (1024 * 1024 + 1)
        r = upload(c, s, models=(("model.txt", big),))
        assert r.status_code == 413
        ctx = captured(app, "new_run.html.j2")
        assert any("larger than the upload limit of 1 MB" in e for e in ctx["errors"])
        assert submissions(s) == [] and run_dirs(s) == []  # the partial file is gone
        # exactly at the cap is fine
        run_id_from(upload(c, s, models=(("model.txt", b"0" * (1024 * 1024)),)))


def test_pickle_models_are_refused_without_allow_pickle(client, app, settings):
    for name, content in (("model.pkl", GETPID_PICKLE), ("model.joblib", b"anything"), ("model.pickle", b"x"),
                          ("model.jbl", b"x")):
        r = upload(client, settings, models=((name, content),))
        ctx = assert_form_error(r, app, contains="pickle-based model file")
        assert PICKLE_REFUSED.format(name=name) in ctx["errors"]
    assert submissions(settings) == [] and run_dirs(settings) == []


def test_pickle_content_is_sniffed_whatever_the_extension(client, app, settings):
    for name in ("model.txt", "model.json", "model.onnx"):
        r = upload(client, settings, models=((name, GETPID_PICKLE),))
        assert_form_error(r, app, contains="pickle-based model file")
    # the zlib-compressed joblib container too
    r = upload(client, settings, models=(("model.model", b"\x78\x9c" + b"\x00" * 32),))
    assert_form_error(r, app, contains="pickle-based model file")
    assert submissions(settings) == []


def test_allow_pickle_accepts_the_pickle_and_passes_the_flag(client, settings, fake_record):
    rid = run_id_from(upload(client, settings, models=(("model.pkl", GETPID_PICKLE),), data={"allow_pickle": "on"}))
    job = job_json(settings, rid)
    assert "--allow-pickle" in job["argv"] and job["options"]["allow_pickle"] is True
    sub = settings.runs_dir / "submissions" / job["submission_id"]
    assert (sub / "model.pkl").read_bytes() == GETPID_PICKLE  # stored, never unpickled here


def test_other_content_checks(client, app, settings):
    r = upload(client, settings, adapter=("a.py", b""))
    assert_form_error(r, app, contains="is empty")
    r = upload(client, settings, adapter=("a.py", b"\x7fELF\x00\x00binary"))
    assert_form_error(r, app, contains="binary data")
    r = upload(client, settings, extra=[("manifest_file", ("h.txt", GETPID_PICKLE))])
    assert_form_error(r, app, contains="not a text file")
    r = upload(client, settings, adapter=None)
    assert_form_error(r, app, contains="choose your adapter")


def test_uploaded_policy_is_validated(client, app, settings):
    r = upload(client, settings, extra=[("config_file", ("gate.yaml", b"modules:\n  performance:\n    max_fpx: 0.1\n"))])
    assert_form_error(r, app, contains="gate policy")
    r = upload(client, settings, extra=[("config_file", ("gate.yaml", b"{{{ not yaml"))])
    assert_form_error(r, app, contains="gate policy")
    assert submissions(settings) == []
    good = b"corpus: toy_v1_corpus\nmodules:\n  performance:\n    max_fpr: 0.02\n"
    rid = run_id_from(upload(client, settings, extra=[("config_file", ("gate.yaml", good))]))
    job = job_json(settings, rid)
    sub = settings.runs_dir / "submissions" / job["submission_id"]
    assert job["argv"][job["argv"].index("--config") + 1] == str(sub / "gate.yaml")
    assert job["options"]["config_source"] == "upload" and job["options"]["corpus"] == "toy_v1_corpus"


def test_malformed_multipart_bodies(client, settings):
    base = {"x-csrf-token": settings.csrf_token}
    r = client.post("/runs", content=b"garbage", headers={**base, "content-type": "multipart/form-data"})
    assert r.status_code == 400
    r = client.post("/runs", content=b"--xyz\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\nupl",
                    headers={**base, "content-type": "multipart/form-data; boundary=xyz"})
    assert r.status_code == 400
    r = client.post("/runs", content=b"{}", headers={**base, "content-type": "application/json"})
    assert r.status_code == 415
    assert submissions(settings) == [] and run_dirs(settings) == []


# --------------------------------------------------------------------------------------------------
# Options, argv, sticky values
# --------------------------------------------------------------------------------------------------


def test_options_become_run_flags(client, settings, tmp_path):
    corpus_dir = tmp_path / "corpora"
    (corpus_dir / "toy_v1_corpus").mkdir(parents=True)
    (corpus_dir / "toy_v1_corpus" / "manifest.json").write_text("{}")
    rid = run_id_from(upload(client, settings, data={
        "class_name": "Detector", "only": "M1,drift", "skip": "M7", "corpus": "toy_v1_corpus",
        "corpus_dir": str(corpus_dir), "seed": "7", "fail_on": "not_ready", "no_sandbox": "on",
        "confirm_no_sandbox": "on",
    }))
    job = job_json(settings, rid)
    argv = job["argv"]
    run_dir = settings.runs_dir / rid
    assert argv[:3] == list(settings.malvalid_command()) + ["run"]
    assert argv[argv.index("--out") + 1] == str(run_dir)
    pairs = list(zip(argv, argv[1:]))
    for pair in (("--class", "Detector"), ("--only", "performance"), ("--only", "drift"), ("--skip", "explanation"),
                 ("--corpus", "toy_v1_corpus"), ("--corpus-dir", str(corpus_dir / "toy_v1_corpus")), ("--seed", "7"),
                 ("--fail-on", "not_ready")):
        assert pair in pairs, pair
    assert "--no-sandbox" in argv and "--allow-pickle" not in argv
    assert job["options"]["only"] == ["performance", "drift"] and job["options"]["seed"] == 7
    assert job["options"]["no_sandbox"] is True and job["options"]["fail_on"] == "not_ready"
    assert job["display_name"] == "Detector · my_adapter.py"


def test_defaults_add_no_flags(client, settings):
    job = job_json(settings, run_id_from(upload(client, settings)))
    argv = job["argv"]
    for flag in ("--class", "--only", "--skip", "--corpus", "--corpus-dir", "--seed", "--fail-on", "--no-sandbox",
                 "--allow-pickle", "--config"):
        assert flag not in argv, flag
    assert job["options"]["config_source"] == "default"


def test_title_goes_to_malvalid_run_as_a_flag(client, settings):
    job = job_json(settings, run_id_from(upload(client, settings, data={"title": "  Nightly   <b>rc</b> "})))
    argv = job["argv"]
    assert argv[argv.index("--title") + 1] == "Nightly <b>rc</b>"
    assert "--config" not in argv  # no rewritten policy just to carry a title
    assert job["display_name"] == "Nightly <b>rc</b>" and job["options"]["title"] == "Nightly <b>rc</b>"


def test_a_title_keeps_the_users_policy_file(client, settings, tmp_path):
    """Regression (correctness-relpath-title-rewrites-policy): with a title, --config is still the user's
    own policy, so relative paths in it (triggers_path) resolve exactly as from a terminal."""
    pol_dir = tmp_path / "policy"
    pol_dir.mkdir()
    (pol_dir / "triggers.yaml").write_text("- name: t\n  features: {x: 1}\n")
    pol = pol_dir / "gate.yaml"
    pol.write_text("modules:\n  backdoor_screen:\n    triggers_path: triggers.yaml\n")
    ad = tmp_path / "ad.py"
    ad.write_text("class A: pass\n")
    r = client.post("/runs", data={"csrf": settings.csrf_token, "mode": "path", "adapter_path": str(ad),
                                   "config_path": str(pol), "title": "with title"}, follow_redirects=False)
    assert r.status_code == 303, r.text
    job = job_json(settings, r.headers["location"].rsplit("/", 1)[1])
    argv = job["argv"]
    assert argv[argv.index("--config") + 1] == str(pol.resolve())
    assert argv[argv.index("--title") + 1] == "with title"
    assert not list((settings.runs_dir / "submissions").glob("*/.malvalid-gate.yaml"))


def test_the_validation_policy_copy_absolutizes_relative_module_paths(tmp_path):
    from malvalid.web.forms import absolutize_policy_paths

    pol_dir = tmp_path / "policy"
    pol_dir.mkdir()
    (pol_dir / "triggers.yaml").write_text("[]\n")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "here.yaml").write_text("[]\n")
    doc = {"corpus_dir": "corp", "modules": {"backdoor_screen": {"triggers_path": "triggers.yaml"},
                                             "x": {"other_path": "here.yaml", "missing_path": "nope.yaml",
                                                   "abs_path": "/etc/hosts", "n": 3}}}
    out = absolutize_policy_paths(doc, pol_dir, cwd)
    assert out["modules"]["backdoor_screen"]["triggers_path"] == str((pol_dir / "triggers.yaml").resolve())
    assert out["modules"]["x"] == doc["modules"]["x"]  # cwd-relative, missing, absolute: unchanged
    assert out["corpus_dir"] == "corp" and doc["modules"]["backdoor_screen"]["triggers_path"] == "triggers.yaml"


def test_server_default_policy_is_used(tmp_path, fake_malvalid):
    pol = tmp_path / "policy.yaml"
    pol.write_text("corpus: toy_v1_corpus\nmodules:\n  drift:\n    enabled: false\n")
    s = WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=8765, config_path=pol,
                    command_prefix=(sys.executable, str(fake_malvalid)))
    app = make_app(s)
    with authed_client(app) as c:
        assert c.get("/runs/new").status_code == 200
        ctx = captured(app, "new_run.html.j2")
        assert ctx["default_config_text"] == pol.read_text()
        assert ctx["defaults"]["corpus"] == "toy_v1_corpus"
        job = job_json(s, run_id_from(upload(c, s, data={"title": "t"})))
        assert job["argv"][job["argv"].index("--config") + 1] == str(pol.resolve())
        assert job["argv"][job["argv"].index("--title") + 1] == "t"
        assert job["options"]["config_source"] == "server"


@pytest.mark.parametrize("data,contains", [
    ({"mode": "ftp"}, "mode must be"),
    ({"only": "M99"}, "M99"),
    ({"skip": "nonsense_module"}, "nonsense_module"),
    ({"corpus": "no_such_corpus"}, "unknown corpus"),
    ({"corpus_dir": "/definitely/not/here"}, "does not exist"),
    ({"seed": "-1"}, "seed"),
    ({"seed": "1.5"}, "seed"),
    ({"fail_on": "ready"}, "fail_on"),
    ({"class_name": "not a class"}, "class name"),
    ({"no_sandbox": "on"}, "tick the confirmation box"),
    ({"title": "x" * 201}, "title"),
])
def test_invalid_options_rerender_the_form_with_sticky_values(client, app, settings, data, contains):
    base = {"title": "keep me", "class_name": "Detector", "seed": "3"}
    r = upload(client, settings, data={**base, **data})
    ctx = assert_form_error(r, app, contains=contains)
    form = ctx["form"]
    for k, v in {**base, **data}.items():
        if k in ("no_sandbox",):
            assert form[k] == "on"
        else:
            assert form[k] == v, k
    assert form["uploaded"]["adapter_file"] == "my_adapter.py"
    # the page still has everything it needs to render
    for k in ("modules", "corpora", "default_config_text", "defaults", "max_upload_mb"):
        assert k in ctx
    assert submissions(settings) == [] and run_dirs(settings) == []


def test_new_run_page_context(client, app, settings):
    r = client.get("/runs/new")
    assert r.status_code == 200
    ctx = captured(app, "new_run.html.j2")
    assert ctx["nav_active"] == "new" and ctx["errors"] == [] and ctx["form"] == {}
    assert ctx["max_upload_mb"] == settings.max_upload_mb
    assert ctx["defaults"]["fail_on"] == "blocked" and "seed" in ctx["defaults"] and "corpus" in ctx["defaults"]
    ids = {m["id"] for m in ctx["modules"]}
    assert {"file_safety", "performance", "drift"} <= ids
    for m in ctx["modules"]:
        assert {"id", "code", "title"} <= set(m)
    names = {c["name"] for c in ctx["corpora"]}
    assert "toy_v1_corpus" in names
    for c in ctx["corpora"]:
        assert {"name", "feature_version", "version", "description", "available", "location", "n", "content_hash",
                "synthetic", "hint"} <= set(c)
    assert "modules:" in ctx["default_config_text"]


def test_json_clients_get_json_form_errors(client, settings):
    r = upload(client, settings, adapter=("a.txt", ADAPTER), headers={"accept": "application/json"})
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False and any("files are accepted here" in e for e in body["errors"])


# --------------------------------------------------------------------------------------------------
# Path mode
# --------------------------------------------------------------------------------------------------


def test_path_mode(client, settings, tmp_path):
    ad = fake_adapter(tmp_path)
    pol = tmp_path / "gate.yml"
    pol.write_text("corpus: toy_v1_corpus\n")
    # a file chosen before switching to path mode is ignored (and not kept)
    r = post_run(client, settings, {"mode": "path", "adapter_path": str(ad), "config_path": str(pol)},
                 [("adapter_file", ("stray.txt", b"x"))])
    job = job_json(settings, run_id_from(r))
    assert job["mode"] == "path" and job["adapter"] == str(ad.resolve())
    assert job["argv"][job["argv"].index("--adapter") + 1] == str(ad.resolve())
    assert job["argv"][job["argv"].index("--config") + 1] == str(pol.resolve())
    assert job["submission_id"] is None and submissions(settings) == []
    assert job["options"]["config_source"] == "path"


@pytest.mark.parametrize("which", ["missing", "dir", "not_py", "empty", "config_missing", "config_ext"])
def test_path_mode_needs_an_existing_py_file(client, app, settings, tmp_path, which):
    ad = fake_adapter(tmp_path)
    data = {"mode": "path", "adapter_path": str(ad)}
    if which == "missing":
        data["adapter_path"] = str(tmp_path / "nope.py")
    elif which == "dir":
        (tmp_path / "pkg.py").mkdir()
        data["adapter_path"] = str(tmp_path / "pkg.py")
    elif which == "not_py":  # (.txt is a model file since model-file submissions: neither is .cfg)
        data["adapter_path"] = str(ad.with_suffix(".cfg"))
        ad.with_suffix(".cfg").write_text("x")
    elif which == "empty":
        data["adapter_path"] = "  "
    elif which == "config_missing":
        data["config_path"] = str(tmp_path / "nope.yaml")
    elif which == "config_ext":
        (tmp_path / "gate.json").write_text("{}")
        data["config_path"] = str(tmp_path / "gate.json")
    r = post_run(client, settings, data)
    ctx = assert_form_error(r, app)
    assert ctx["form"]["mode"] == "path"
    assert run_dirs(settings) == [] and submissions(settings) == []


def test_path_mode_with_a_title_stores_nothing(client, settings, tmp_path):
    ad = fake_adapter(tmp_path)
    job = job_json(settings, run_id_from(post_run(client, settings, {"mode": "path", "adapter_path": str(ad),
                                                                     "title": "T"})))
    assert job["submission_id"] is None and submissions(settings) == []
    assert job["argv"][job["argv"].index("--title") + 1] == "T"


# --------------------------------------------------------------------------------------------------
# POST /api/validate
# --------------------------------------------------------------------------------------------------


def _validate(client, settings, data, files=None, **kw):
    return post_run(client, settings, data, files, url="/api/validate", **kw)


def test_validate_path_mode(client, settings, tmp_path, fake_record):
    ad = fake_adapter(tmp_path)
    r = _validate(client, settings, {"mode": "path", "adapter_path": str(ad), "class_name": "Detector",
                                     "no_sandbox": "on", "confirm_no_sandbox": "on", "corpus": "toy_v1_corpus",
                                     "seed": "5"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["exit_code"] == 0 and body["timed_out"] is False
    assert [c["name"] for c in body["checks"]] == ["declarations", "load"]
    call = read_record(fake_record)[-1]
    assert call["cmd"] == "validate-adapter" and call["PYTHONSAFEPATH"] == "1"
    argv = call["argv"]
    assert "--json" in argv and argv[argv.index("--adapter") + 1] == str(ad.resolve())
    assert ("--class", "Detector") in list(zip(argv, argv[1:]))
    cfg = argv[argv.index("--config") + 1]
    # the form's options were folded into a policy file, removed after the validation
    assert not Path(cfg).exists()
    assert submissions(settings) == [] and run_dirs(settings) == []


def test_validate_upload_mode_and_its_policy(client, settings, fake_record, monkeypatch):
    seen: dict[str, Any] = {}
    real_validate = type(client.app.state.jobs).validate

    def spy(self, argv, *, cwd, timeout_s=None):
        cfg = Path(argv[argv.index("--config") + 1])
        seen["policy"] = yaml.safe_load(cfg.read_text())
        seen["files"] = sorted(p.name for p in cfg.parent.iterdir())
        return real_validate(self, argv, cwd=cwd, timeout_s=timeout_s)

    monkeypatch.setattr(type(client.app.state.jobs), "validate", spy)
    r = _validate(client, settings, {"mode": "upload", "no_sandbox": "on", "confirm_no_sandbox": "on",
                                     "allow_pickle": "on", "seed": "9"},
                  [("adapter_file", ("my_adapter.py", ADAPTER)), ("model_files", ("model.txt", LGBM_TEXT))])
    assert r.status_code == 200 and r.json()["ok"] is True
    assert seen["policy"]["runtime"] == {"sandbox": False, "allow_pickle": True, "seed": 9}
    assert seen["files"] == [".malvalid-gate.yaml", "model.txt", "my_adapter.py"]
    assert "--allow-pickle" in read_record(fake_record)[-1]["argv"]
    assert submissions(settings) == []  # uploads of a validation are not kept


def test_validate_reports_failures(client, settings, tmp_path):
    r = _validate(client, settings, {"mode": "path", "adapter_path": str(fake_adapter(tmp_path, "invalid"))})
    body = r.json()
    assert r.status_code == 200 and body["ok"] is False and body["exit_code"] == 1
    r = _validate(client, settings, {"mode": "path", "adapter_path": str(fake_adapter(tmp_path, "garbage"))})
    body = r.json()
    assert body["ok"] is False and body["exit_code"] == 2
    assert body["error"] == "the adapter exploded while importing"
    assert "exploded" in body["stderr_tail"]


def test_validate_form_errors_are_json(client, settings, tmp_path):
    r = _validate(client, settings, {"mode": "path", "adapter_path": str(tmp_path / "nope.py")})
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False and body["errors"]
    r = _validate(client, settings, {"mode": "upload"}, [("model_files", ("m.pkl", GETPID_PICKLE))])
    assert r.status_code == 400
    assert any("pickle" in e for e in r.json()["errors"])
    assert submissions(settings) == []


def test_validate_timeout(tmp_path, fake_malvalid):
    s = WebSettings(runs_dir=tmp_path / "runs", token=TOKEN, port=8765, validate_timeout_s=1.0, allow_path_mode=True,
                    command_prefix=(sys.executable, str(fake_malvalid)))
    app = make_app(s)
    with authed_client(app) as c:
        r = _validate(c, s, {"mode": "path", "adapter_path": str(fake_adapter(tmp_path, "sleep"))})
        body = r.json()
        assert body["ok"] is False and body["timed_out"] is True and "did not finish within 1 s" in body["error"]
        assert body["elapsed_s"] < 30


def test_validate_needs_csrf(client, settings, tmp_path):
    r = _validate(client, settings, {"mode": "path", "adapter_path": str(fake_adapter(tmp_path))}, csrf=False)
    assert r.status_code == 403
