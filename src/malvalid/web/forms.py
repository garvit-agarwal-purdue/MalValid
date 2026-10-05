"""The new-run form: validation and argv construction shared by ``POST /runs`` and ``POST /api/validate``.

Fields (``docs/WEB_CONTRACT.md`` §5): ``mode=model|upload|path`` (missing: ``upload`` when an
``adapter_file`` is sent, ``model`` when a ``model_file`` / ``model_submission`` is, else ``upload``).
Model mode: ``model_file`` (or ``model_submission``, the id of an inspected upload), ``manifest_file``,
``config_file``; upload mode (custom adapter): ``adapter_file``, ``model_files`` (multiple),
``manifest_file``, ``config_file``; path mode: ``adapter_path`` (an adapter ``.py`` or a model file),
``config_path``, ``training_hashes_path``. Model-file options: ``model_kind, feature_version,
threshold_mode (calibrate|declared), threshold, calibrate_fpr, training_cutoff``; options: ``class_name,
only, skip, corpus, corpus_dir, seed, fail_on, allow_pickle, no_sandbox, confirm_no_sandbox, title``.

Everything here reads files as text or YAML (``yaml.safe_load``) only; the adapter is never imported.
"""

from __future__ import annotations

import copy
import datetime as dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from malvalid.web.settings import WebSettings
from malvalid.web.uploads import MODEL_EXTENSIONS, FormData, content_problems

TEXT_FIELDS = ("mode", "adapter_path", "config_path", "class_name", "only", "skip", "corpus", "corpus_dir",
               "seed", "fail_on", "allow_pickle", "no_sandbox", "confirm_no_sandbox", "title",
               "model_kind", "feature_version", "threshold_mode", "threshold", "calibrate_fpr", "training_cutoff",
               "training_hashes_path", "model_submission")
CHECKBOXES = ("allow_pickle", "no_sandbox", "confirm_no_sandbox")
FAIL_ON_CHOICES = ("blocked", "not_ready", "conditional")
MODES = ("model", "upload", "path")
#: Model-file submissions (``malvalid run --model``): the choices the form offers ("" = auto-detect).
MODEL_KINDS = ("lightgbm", "xgboost", "sklearn_gbdt", "onnx")
FEATURE_VERSIONS = ("ember_v2", "ember_v3")
THRESHOLD_MODES = ("calibrate", "declared")
CALIBRATE_FPRS = (0.001, 0.005, 0.01)
DEFAULT_CALIBRATE_FPR = 0.005
_CUTOFF_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])(-\d{2})?$")
MAX_TITLE_LEN = 200
EFFECTIVE_CONFIG_NAME = ".malvalid-gate.yaml"  # dot-leading: can never collide with an upload


def truthy(v: Any) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "on", "yes")


def _split(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        for part in str(v).split(","):
            p = part.strip()
            if p and p not in out:
                out.append(p)
    return out


def sticky_form(form: FormData | None) -> dict[str, Any]:
    """The text fields as submitted (re-filled after a failed POST; files must be chosen again)."""
    if form is None:
        return {}
    out: dict[str, Any] = {}
    for k in TEXT_FIELDS:
        vals = form.getlist(k)
        if not vals:
            continue
        if k in ("only", "skip"):
            ids = _split(vals)
            out[k] = ",".join(ids)
            out[f"{k}_ids"] = ids
        elif k in CHECKBOXES:
            out[k] = "on" if any(truthy(v) for v in vals) else ""
        else:
            out[k] = vals[-1]
    uploaded = {f.field: f.name for f in form.all_files()}
    if uploaded:
        out["uploaded"] = uploaded
    return out


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


#: Suffixes of policy keys under ``modules.<id>`` that hold file or directory paths.
_PATH_KEY_SUFFIXES = ("_path", "_dir", "_file")


def absolutize_policy_paths(doc: dict[str, Any], policy_dir: Path, cwd: Path) -> dict[str, Any]:
    """A copy of a policy document whose relative module file paths are absolute.

    malvalid resolves such a path against the working directory first, then against the policy's own
    directory; a copy written elsewhere would lose the second option. A path that exists relative to
    ``cwd`` is kept as is (it means the same from the copy); one that exists next to the policy is made
    absolute; anything else is left alone (malvalid reports it). Top-level ``corpus_dir`` /
    ``sample_dir`` are working-directory relative in malvalid and are left unchanged.
    """
    out = copy.deepcopy(doc)
    mods = out.get("modules")
    if not isinstance(mods, dict):
        return out
    for params in mods.values():
        if not isinstance(params, dict):
            continue
        for k, v in list(params.items()):
            if not (isinstance(k, str) and k.endswith(_PATH_KEY_SUFFIXES) and isinstance(v, str) and v.strip()):
                continue
            pv = Path(v).expanduser()
            if pv.is_absolute() or (cwd / pv).exists():
                continue
            cand = policy_dir / pv
            if cand.exists():
                params[k] = str(cand.resolve())
    return out


_CLASS_RE = re.compile(r"^class\s+([A-Za-z_]\w*)\s*(\(([^)]*)\))?\s*:", re.MULTILINE)


def guess_class_name(adapter: Path) -> str | None:
    """The detector class an adapter file most likely submits, read as *text* (never imported): the only
    class that names ``Detector`` among its bases, else the only top-level class. None if unclear."""
    try:
        with open(adapter, "rb") as f:
            text = f.read(1024 * 1024).decode("utf-8", "replace")
    except OSError:
        return None
    classes = [(m.group(1), m.group(3) or "") for m in _CLASS_RE.finditer(text)]
    detectors = [name for name, bases in classes if "Detector" in bases]
    if len(detectors) == 1:
        return detectors[0]
    if len(classes) == 1:
        return classes[0][0]
    return None


def resolve_corpus_dir(d: Path, corpus_name: str | None) -> Path:
    """Like ``malvalid run --corpus-dir``: the corpus itself, or a root containing ``<name>/``."""
    d = d.expanduser().resolve()
    if corpus_name and not (d / "manifest.json").exists() and (d / corpus_name / "manifest.json").exists():
        return d / corpus_name
    return d


@dataclass
class RunRequest:
    """A validated new-run form."""

    mode: str
    adapter: Path
    base_config: Path | None  # the user's policy (upload / path) or the server default; None = packaged
    config_source: str  # upload | path | server | default
    class_name: str | None = None
    only: list[str] = field(default_factory=list)
    skip: list[str] = field(default_factory=list)
    corpus: str | None = None
    corpus_dir: Path | None = None
    seed: int | None = None
    fail_on: str = "blocked"
    allow_pickle: bool = False
    no_sandbox: bool = False
    title: str | None = None
    submission_id: str | None = None
    submission_dir: Path | None = None
    uploaded: dict[str, Any] = field(default_factory=dict)
    effective_corpus: str | None = None  # the corpus the run will use (form, else policy)
    # ---- model-file submissions (``malvalid run --model``; ``adapter`` is then the model path) ----
    model: Path | None = None
    model_kind: str | None = None  # None = detected by the CLI
    feature_version: str | None = None  # None = detected by the CLI
    threshold_mode: str = "calibrate"  # calibrate | declared
    threshold: float | None = None
    calibrate_fpr: float | None = None
    training_cutoff: str | None = None
    training_hashes: Path | None = None
    #: An inspected upload (``POST /runs/inspect``) whose model file this run takes over.
    reused_submission: str | None = None

    @property
    def is_model(self) -> bool:
        """A model file submitted without an adapter (``malvalid run --model``)."""
        return self.model is not None

    @property
    def display_name(self) -> str:
        if self.title:
            return self.title
        if self.model is not None:
            return " · ".join(x for x in (self.model_kind, self.model.name) if x)
        cls = self.class_name or guess_class_name(self.adapter)
        return " · ".join(x for x in (cls, self.adapter.name) if x)

    def use_model(self, path: Path) -> None:
        """Point the request at ``path`` (the inspected model, after it was moved into this submission)."""
        self.model = path
        self.adapter = path

    def model_options(self) -> dict[str, Any]:
        """The model-file choices recorded in ``job.json`` (empty for adapter submissions)."""
        if self.model is None:
            return {}
        return {
            "submission_kind": "model",
            "model_kind": self.model_kind,
            "feature_version": self.feature_version,
            "threshold_mode": self.threshold_mode,
            "threshold": self.threshold,
            "calibrate_fpr": self.calibrate_fpr,
            "training_cutoff": self.training_cutoff,
            "training_hashes": (str(self.training_hashes) if self.training_hashes is not None and self.mode == "path"
                                else None),
        }

    def options(self) -> dict[str, Any]:
        """The form options as recorded in ``job.json``."""
        return {
            **self.model_options(),
            "title": self.title,
            "class_name": self.class_name,
            "only": list(self.only),
            "skip": list(self.skip),
            "corpus": self.corpus or self.effective_corpus,
            "corpus_dir": str(self.corpus_dir) if self.corpus_dir else None,
            "seed": self.seed,
            "fail_on": self.fail_on,
            "allow_pickle": self.allow_pickle,
            "no_sandbox": self.no_sandbox,
            "config": str(self.base_config) if self.base_config else None,
            "config_source": self.config_source,
            "files": dict(self.uploaded),
        }

    # ---- effective policy ------------------------------------------------------------------------

    def write_effective_config(self, overrides: dict[str, Any], dest_dir: Path, *,
                               cwd: Path | None = None) -> Path:
        """``base_config`` + ``overrides`` as YAML in ``dest_dir`` (policy for options without a CLI flag).

        The copy lives somewhere else than the user's policy, so relative module file paths (e.g.
        ``modules.backdoor_screen.triggers_path``, which malvalid resolves against the working directory
        and then the policy's own directory) are made absolute first; ``cwd`` is the directory the
        subprocess will run in.
        """
        doc: dict[str, Any] = {}
        if self.base_config is not None:
            loaded = yaml.safe_load(self.base_config.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                doc = loaded
        if self.base_config is not None:
            doc = absolutize_policy_paths(doc, self.base_config.parent, cwd or Path.cwd())
        merged = deep_merge(doc, overrides)
        src = str(self.base_config) if self.base_config else "the packaged default policy"
        text = (f"# Gate policy for one malvalid web run: {src}\n"
                "# plus the options chosen in the web form. Written by `malvalid serve`.\n"
                + yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / EFFECTIVE_CONFIG_NAME
        path.write_text(text, encoding="utf-8")
        return path

    def run_config(self) -> Path | None:
        """The ``--config`` for ``malvalid run``: the user's policy, unchanged (the title travels as
        ``--title``, so relative paths in the policy keep meaning what they mean from a terminal)."""
        return self.base_config

    def model_argv(self) -> list[str]:
        """``--model M [--model-kind K] [--feature-version V] (--threshold T | --calibrate-fpr F) ...``."""
        assert self.model is not None
        argv = ["--model", str(self.model)]
        if self.model_kind:
            argv += ["--model-kind", self.model_kind]
        if self.feature_version:
            argv += ["--feature-version", self.feature_version]
        if self.threshold_mode == "declared" and self.threshold is not None:
            argv += ["--threshold", repr(float(self.threshold))]
        else:
            argv += ["--calibrate-fpr", repr(float(self.calibrate_fpr or DEFAULT_CALIBRATE_FPR))]
        if self.training_cutoff:
            argv += ["--training-cutoff", self.training_cutoff]
        if self.training_hashes is not None:
            argv += ["--training-hashes", str(self.training_hashes)]
        return argv

    def run_argv(self, settings: WebSettings, config: Path | None, out_dir: str = "{run_dir}",
                 run_id: str | None = "{run_id}") -> list[str]:
        if self.model is not None:
            argv = settings.malvalid_command() + ["run"] + self.model_argv() + ["--out", out_dir]
        else:
            argv = settings.malvalid_command() + ["run", "--adapter", str(self.adapter), "--out", out_dir]
        if run_id:  # the report records the web run id, so the UI and report.html agree
            argv += ["--run-id", run_id]
        if config is not None:
            argv += ["--config", str(config)]
        if self.class_name and self.model is None:
            argv += ["--class", self.class_name]
        for m in self.only:
            argv += ["--only", m]
        for m in self.skip:
            argv += ["--skip", m]
        if self.corpus:
            argv += ["--corpus", self.corpus]
        if self.corpus_dir is not None:
            argv += ["--corpus-dir", str(self.corpus_dir)]
        if self.seed is not None:
            argv += ["--seed", str(self.seed)]
        if self.allow_pickle:
            argv.append("--allow-pickle")
        if self.no_sandbox:
            argv.append("--no-sandbox")
        elif settings.allow_reduced_isolation:
            argv.append("--allow-reduced-isolation")
        if self.fail_on != "blocked":
            argv += ["--fail-on", self.fail_on]
        if self.title:
            argv += ["--title", self.title]
        return argv

    def validate_argv(self, settings: WebSettings, dest_dir: Path) -> list[str]:
        """``malvalid validate-adapter --json`` with the form's options folded into a policy file
        (validate-adapter has no --corpus / --corpus-dir / --no-sandbox flags). Adapters only."""
        if self.model is not None:
            raise ValueError("validate-adapter checks an adapter; a model file is checked with inspect-model")
        over: dict[str, Any] = {}
        if self.corpus:
            over["corpus"] = self.corpus
        if self.corpus_dir is not None:
            over["corpus_dir"] = str(self.corpus_dir)
        rt: dict[str, Any] = {}
        if self.no_sandbox:
            rt["sandbox"] = False
        elif settings.allow_reduced_isolation:
            rt["allow_reduced_isolation"] = True
        if self.allow_pickle:
            rt["allow_pickle"] = True
        if self.seed is not None:
            rt["seed"] = self.seed
        if rt:
            over["runtime"] = rt
        cfg = self.write_effective_config(over, dest_dir, cwd=settings.launch_dir)
        argv = settings.malvalid_command() + ["validate-adapter", "--json", "--adapter", str(self.adapter),
                                              "--config", str(cfg)]
        if self.allow_pickle:
            argv.append("--allow-pickle")
        if self.class_name:
            argv += ["--class", self.class_name]
        return argv


def _check_config(path: Path, errors: list[str], label: str) -> Any:
    from malvalid.config import load_config, validate_against_registry
    from malvalid.core import MalValidError

    try:
        cfg = load_config(path)
        validate_against_registry(cfg)
        return cfg
    except MalValidError as e:
        errors.append(f"{label}: {e}")
    except (OSError, ValueError) as e:
        errors.append(f"{label}: {e}")
    return None


def _hide_internal_modules(msg: str) -> str:
    """Drop malvalid's own smoke-test module (``X0=dummy``) from "known: ..." lists shown to researchers."""
    return re.sub(r",\s*X0=dummy\b|\bX0=dummy,\s*", "", msg)


def default_policy(settings: WebSettings) -> Any:
    """The policy new runs get when the form brings none (``serve --config`` or the packaged one)."""
    from malvalid.config import load_config

    return load_config(settings.config_path) if settings.config_path else load_config()


def build_request(
    form: FormData,
    settings: WebSettings,
    *,
    submission_id: str | None,
    submission_dir: Path | None,
) -> tuple[RunRequest | None, list[str]]:
    """Validate the form. Returns ``(request, [])`` or ``(None, errors)`` (errors are user-facing)."""
    from malvalid import registry
    from malvalid.core import MalValidError
    from malvalid.runner import resolve_module_ids

    errors: list[str] = []
    disabled = disabled_feature_errors(form, settings)
    if disabled:  # never touch a server path or build an unsandboxed run the server does not allow
        return None, disabled
    mode = infer_mode(form)
    if mode not in MODES:
        errors.append(f"mode must be 'model', 'upload' or 'path' (got {mode!r})")
        mode = "upload"
    if mode in ("model", "upload"):  # problems with files are moot in path mode (they are not used)
        errors.extend(form.problems)
    allow_pickle = truthy(form.get("allow_pickle"))
    no_sandbox = truthy(form.get("no_sandbox"))

    adapter: Path | None = None
    model: Path | None = None
    training_hashes: Path | None = None
    base_config: Path | None = None
    config_source = "default"
    uploaded: dict[str, Any] = {}
    if mode == "model":
        mf = form.file("model_file")
        if mf is None:
            errors.append("choose your model file (LightGBM .txt/.model, XGBoost .json/.ubj, ONNX .onnx; "
                          "pickles need “Allow pickle”)")
        else:
            model = mf.path
            uploaded["model"] = mf.name
        hf = form.file("manifest_file")
        if hf is not None:
            uploaded["manifest"] = hf.name
            training_hashes = hf.path
        cf = form.file("config_file")
        if cf is not None:
            uploaded["config"] = cf.name
            base_config = cf.path
            config_source = "upload"
        errors.extend(content_problems(form, allow_pickle=allow_pickle))
    elif mode == "upload":
        ad = form.file("adapter_file")
        if ad is None:
            errors.append("choose your adapter .py file (or switch to path mode to use one on this machine)")
        else:
            adapter = ad.path
            uploaded["adapter"] = ad.name
        uploaded["models"] = [f.name for f in form.files.get("model_files") or []]
        mf = form.file("manifest_file")
        if mf is not None:
            uploaded["manifest"] = mf.name
        cf = form.file("config_file")
        if cf is not None:
            uploaded["config"] = cf.name
            base_config = cf.path
            config_source = "upload"
        errors.extend(content_problems(form, allow_pickle=allow_pickle))
    else:
        raw = form.get("adapter_path").strip()
        if not raw:
            errors.append("enter the path of your model file or adapter .py file on this machine")
        else:
            p = Path(raw).expanduser()
            try:
                p = p.resolve()
            except (OSError, RuntimeError):
                pass
            if not p.is_file():
                errors.append(f"adapter path {raw!r} does not exist or is not a file")
            elif p.suffix == ".py":
                adapter = p
            elif p.suffix.lower() in MODEL_EXTENSIONS:
                model = p
                if p.stat().st_size == 0:
                    errors.append(f"model file {raw!r} is empty")
                else:
                    from malvalid.loaders.base import is_pickle_artifact

                    if is_pickle_artifact(p) and not allow_pickle:
                        from malvalid.web.uploads import PICKLE_REFUSED

                        errors.append(PICKLE_REFUSED.format(name=p.name))
            else:
                errors.append(f"adapter path {raw!r} is not a .py file (an adapter) or a model file "
                              f"({' '.join(MODEL_EXTENSIONS)})")
        rawh = form.get("training_hashes_path").strip()
        if rawh:
            h = Path(rawh).expanduser()
            try:
                h = h.resolve()
            except (OSError, RuntimeError):
                pass
            if not h.is_file():
                errors.append(f"training hash list {rawh!r} does not exist or is not a file")
            else:
                training_hashes = h
        rawc = form.get("config_path").strip()
        if rawc:
            c = Path(rawc).expanduser()
            try:
                c = c.resolve()
            except (OSError, RuntimeError):
                pass
            if not c.is_file():
                errors.append(f"config path {rawc!r} does not exist or is not a file")
            elif c.suffix.lower() not in (".yaml", ".yml"):
                errors.append(f"config path {rawc!r} is not a .yaml/.yml file")
            else:
                base_config = c
                config_source = "path"

    cfg = None
    if base_config is not None:
        cfg = _check_config(base_config, errors, "gate policy")
    elif settings.config_path is not None:
        base_config = settings.config_path
        config_source = "server"
        cfg = _check_config(base_config, errors, "server default policy (--config)")
    else:
        try:
            cfg = default_policy(settings)
        except MalValidError as e:  # pragma: no cover - the packaged policy is valid
            errors.append(str(e))

    class_name = form.get("class_name").strip() or None
    if class_name is not None and not class_name.isidentifier():
        errors.append(f"class name {class_name!r} is not a valid Python class name")

    mopts = model_choices(form, errors)
    if model is not None:
        if class_name is not None:
            errors.append("the adapter class only applies to an adapter .py file, not to a model file")
            class_name = None
        _check_pickle_choices(model, mopts, allow_pickle, errors)
    elif adapter is not None and (mopts["explicit"] or training_hashes is not None):
        errors.append("model kind, feature version, a declared threshold, the training cutoff and the training "
                      "hash list only apply to a model file; an adapter declares them itself")

    only: list[str] = []
    skip: list[str] = []
    try:
        only = resolve_module_ids(_split(form.getlist("only")), flag="only") or []
    except MalValidError as e:
        errors.append(_hide_internal_modules(str(e)))
    try:
        skip = resolve_module_ids(_split(form.getlist("skip")), flag="skip") or []
    except MalValidError as e:
        errors.append(_hide_internal_modules(str(e)))

    corpus = form.get("corpus").strip() or None
    if corpus is not None:
        known = registry.corpora()
        if corpus not in known:
            why = registry.unavailable("corpora").get(corpus)
            errors.append(f"unknown corpus {corpus!r}" + (f" (it failed to import: {why})" if why else "")
                          + f"; known: {', '.join(sorted(known))}")
    effective_corpus = corpus or (getattr(cfg, "corpus", None) if cfg is not None else None)
    cfg_rt = getattr(cfg, "runtime", None) if cfg is not None else None
    if cfg_rt is not None and getattr(cfg_rt, "allow_reduced_isolation", False) and not settings.allow_reduced_isolation:
        # a policy file must not opt into reduced isolation behind the server's back
        errors.append("The policy sets runtime.allow_reduced_isolation, but this server was not started with "
                      "--allow-reduced-isolation. Remove it from the policy.")
    cfg_cdir = getattr(cfg, "corpus_dir", None) if cfg is not None else None
    if cfg_cdir and not corpus_dir_allowed(cfg_cdir, settings):  # a policy file must not bypass the restriction
        errors.append(CORPUS_DIR_DISABLED.replace("“Corpus directory” empty", "the policy's `corpus_dir` unset"))

    corpus_dir: Path | None = None
    rawd = form.get("corpus_dir").strip()
    if rawd:
        d = Path(rawd).expanduser()
        if not d.is_dir():
            errors.append(f"corpus directory {rawd!r} does not exist")
        else:
            corpus_dir = resolve_corpus_dir(d, effective_corpus)

    seed: int | None = None
    raws = form.get("seed").strip()
    if raws:
        try:
            seed = int(raws)
            if not 0 <= seed < 2**63:
                raise ValueError
        except ValueError:
            errors.append(f"seed must be a non-negative whole number (got {raws!r})")
            seed = None

    fail_on = (form.get("fail_on") or "blocked").strip() or "blocked"
    if fail_on not in FAIL_ON_CHOICES:
        errors.append(f"fail_on must be one of {', '.join(FAIL_ON_CHOICES)} (got {fail_on!r})")

    if no_sandbox and not truthy(form.get("confirm_no_sandbox")):
        errors.append(
            "running without the sandbox loads your adapter and model inside malvalid with no isolation "
            "(no network block, full file-system access); tick the confirmation box if you trust them"
        )

    title = " ".join(form.get("title").split()) or None
    if title is not None and len(title) > MAX_TITLE_LEN:
        errors.append(f"title is longer than {MAX_TITLE_LEN} characters")

    if not errors:
        missing = missing_corpus_error(corpus, model, mopts, cfg, corpus_dir)
        if missing:
            errors.append(missing)

    target = adapter if adapter is not None else model
    if errors or target is None:
        return None, errors or ["the form is incomplete"]
    if model is not None and corpus is None:
        effective_corpus = None  # malvalid run picks the corpus of the model's feature version
    req = RunRequest(
        mode=mode, adapter=target, base_config=base_config, config_source=config_source,
        class_name=class_name, only=only, skip=skip, corpus=corpus, corpus_dir=corpus_dir, seed=seed,
        fail_on=fail_on, allow_pickle=allow_pickle, no_sandbox=no_sandbox, title=title,
        submission_id=submission_id, submission_dir=submission_dir, uploaded=uploaded,
        effective_corpus=effective_corpus,
    )
    if model is not None:
        req.model = model
        req.model_kind = mopts["model_kind"]
        req.feature_version = mopts["feature_version"]
        req.threshold_mode = mopts["threshold_mode"]
        req.threshold = mopts["threshold"]
        req.calibrate_fpr = mopts["calibrate_fpr"]
        req.training_cutoff = mopts["training_cutoff"]
        req.training_hashes = training_hashes
    return req, []


PATH_MODE_DISABLED = ("“Use files on this machine” (path mode) is disabled on this server: it would let the "
                      "server read files by path. Upload your model file instead, or start the server with "
                      "`malvalid serve --allow-path-mode`.")
NO_SANDBOX_DISABLED = ("“Run without the sandbox” is disabled on this server: submitted models always load in the "
                       "sandbox. Start the server with `malvalid serve --allow-no-sandbox` to allow it, or use "
                       "`malvalid run --no-sandbox` on the command line.")


MISSING_CORPUS_INTRO = ("The corpus {name!r} is not on this machine, so this run cannot be started. The real EMBER "
                        "corpora are not shipped with malvalid; the synthetic demo (the \u201cRun the synthetic demo\u201d "
                        "button, or the synthetic_v2 / synthetic_v3 corpora) works without any download. "
                        "To use {name}, build it once:\n")


def missing_corpus_message(name: str, corpus_dir: Path | None = None, cfg: Any = None) -> str | None:
    """A friendly message when canonical corpus ``name`` is a real (non-synthetic) corpus that is not built.

    The build steps are the provider's own ``unavailable_hint`` (one copy of the instructions)."""
    from types import SimpleNamespace

    from malvalid import registry

    try:
        prov = registry.get_corpus_provider(name)
        if getattr(prov, "synthetic", False):
            return None
        cdir = corpus_dir or (getattr(cfg, "corpus_dir", None) if cfg is not None and getattr(cfg, "corpus", None) == name else None)
        use = SimpleNamespace(corpus_dir=cdir) if cdir else None
        if prov.is_available(use):
            return None
        return MISSING_CORPUS_INTRO.format(name=name) + prov.unavailable_hint(prov.locate(use))
    except Exception:  # noqa: BLE001 - unknown/broken provider: the runner reports it
        return None


def missing_corpus_error(corpus: str | None, model: Path | None, mopts: dict[str, Any], cfg: Any,
                         corpus_dir: Path | None) -> str | None:
    """The friendly missing-corpus error for the corpus this submission will use, or None.

    Only an explicit corpus choice, or a model file (whose corpus is picked from its feature version,
    like ``malvalid run``), is checked; an adapter run on the policy's corpus is left to the runner."""
    name = corpus
    if name is None and model is not None:
        fv = mopts.get("feature_version")
        if fv is None:
            try:
                from malvalid.inspect_model import inspect_model

                fv = inspect_model(model).feature_version
            except Exception:  # noqa: BLE001 - undetectable here: the run reports it
                fv = None
        from malvalid.inspect_model import default_corpus_for

        name = default_corpus_for(fv)
        pol = getattr(cfg, "corpus", None)
        if name and pol and pol != name:
            try:
                from malvalid import registry

                if registry.get_corpus_provider(pol).info().get("feature_version") == fv:
                    name = None  # the policy's corpus already matches the model's feature space
            except Exception:  # noqa: BLE001
                pass
    return missing_corpus_message(name, corpus_dir, cfg) if name else None


def is_bundled_demo(form: FormData) -> bool:
    """Is this path-mode form exactly the bundled synthetic demo (the "Run the demo" button)? Its
    paths are fixed by the server (:func:`malvalid.web.onboarding.demo_submission`), so it is accepted
    without ``--allow-path-mode``; any other path, or a training hash list, is not."""
    from malvalid.web.onboarding import demo_submission

    demo = demo_submission()
    if demo is None or form.get("training_hashes_path").strip():
        return False

    def same(raw: str, expected: str | None) -> bool:
        if not raw:
            return expected is None
        if expected is None:
            return False
        try:
            return Path(raw).resolve() == Path(expected).resolve()
        except (OSError, RuntimeError, ValueError):
            return False

    return (same(form.get("adapter_path").strip(), demo["adapter_path"])
            and (not form.get("config_path").strip() or same(form.get("config_path").strip(), demo["config_path"])))


CORPUS_DIR_DISABLED = ("A custom corpus directory is disabled on this server: it would let the server read "
                       "any directory it can access. Leave “Corpus directory” empty to use the server's corpus "
                       "directory, or start the server with `malvalid serve --allow-path-mode`.")


def configured_corpus_root() -> Path:
    """The server's corpus directory: ``$MALVALID_CORPUS_DIR`` or the default cache location."""
    from malvalid.corpora.base import default_corpus_root

    return default_corpus_root().expanduser().resolve()


def corpus_dir_allowed(raw: str | Path | None, settings: WebSettings) -> bool:
    """True when ``raw`` is empty, path mode is on, or it names the server's configured corpus directory."""
    if raw is None or not str(raw).strip() or settings.allow_path_mode:
        return True
    try:
        return Path(str(raw).strip()).expanduser().resolve() == configured_corpus_root()
    except (OSError, RuntimeError, ValueError):
        return False


def disabled_feature_errors(form: FormData, settings: WebSettings) -> list[str]:
    """Requests for features this server was not started with (``--allow-path-mode``,
    ``--allow-no-sandbox``); the caller answers 403. Checked before anything is read by path.
    The bundled demo (fixed server-side paths) stays available in path mode."""
    errors: list[str] = []
    if infer_mode(form) == "path" and not settings.allow_path_mode and not is_bundled_demo(form):
        errors.append(PATH_MODE_DISABLED)
    if not corpus_dir_allowed(form.get("corpus_dir"), settings):
        errors.append(CORPUS_DIR_DISABLED)
    if truthy(form.get("no_sandbox")) and not settings.allow_no_sandbox:
        errors.append(NO_SANDBOX_DISABLED)
    return errors


def infer_mode(form: FormData) -> str:
    """The form's ``mode``; when absent: ``upload`` for an adapter upload (the pre-model-file API),
    ``model`` for a model file (or an inspected one), else ``upload``."""
    mode = form.get("mode").strip().lower()
    if mode:
        return mode
    if form.file("adapter_file") is not None:
        return "upload"
    if form.file("model_file") is not None or form.get("model_submission").strip():
        return "model"
    return "upload"


def _auto(v: str) -> str | None:
    v = v.strip()
    return None if v.lower() in ("", "auto") else v


def model_choices(form: FormData, errors: list[str]) -> dict[str, Any]:
    """Model kind, feature version, threshold and training cutoff of a model-file submission, validated.

    ``explicit`` is true when anything beyond the defaults (auto-detect, calibrate at 0.5% FPR) was chosen.
    """
    out: dict[str, Any] = {"model_kind": None, "feature_version": None, "threshold_mode": "calibrate",
                           "threshold": None, "calibrate_fpr": DEFAULT_CALIBRATE_FPR, "training_cutoff": None,
                           "explicit": False}
    kind = _auto(form.get("model_kind"))
    if kind is not None:
        if kind not in MODEL_KINDS:
            errors.append(f"model kind must be Auto-detect or one of {', '.join(MODEL_KINDS)} (got {kind!r})")
        else:
            out["model_kind"] = kind
            out["explicit"] = True
    fv = _auto(form.get("feature_version"))
    if fv is not None:
        if fv not in FEATURE_VERSIONS:
            errors.append(f"feature version must be Auto-detect or one of {', '.join(FEATURE_VERSIONS)} (got {fv!r})")
        else:
            out["feature_version"] = fv
            out["explicit"] = True
    raw_thr = form.get("threshold").strip()
    tm = form.get("threshold_mode").strip().lower() or ("declared" if raw_thr else "calibrate")
    if tm not in THRESHOLD_MODES:
        errors.append(f"threshold choice must be 'calibrate' or 'declared' (got {tm!r})")
        tm = "calibrate"
    out["threshold_mode"] = tm
    if tm == "declared":
        out["explicit"] = True
        out["calibrate_fpr"] = None
        if not raw_thr:
            errors.append("enter the operating threshold you ship (a number from 0 to 1), or let malvalid "
                          "calibrate one")
        else:
            try:
                t = float(raw_thr)
                if not (math.isfinite(t) and 0.0 <= t <= 1.0):
                    raise ValueError
                out["threshold"] = t
            except ValueError:
                errors.append(f"the operating threshold must be a number from 0 to 1 (got {raw_thr!r})")
    else:
        raw_fpr = form.get("calibrate_fpr").strip()
        if raw_fpr:
            try:
                f = float(raw_fpr)
                match = [c for c in CALIBRATE_FPRS if math.isclose(f, c, rel_tol=1e-9)]
                if not match:
                    raise ValueError
                out["calibrate_fpr"] = match[0]
            except ValueError:
                errors.append("the calibration target must be 0.1%, 0.5% or 1% false positives "
                              f"(0.001, 0.005 or 0.01; got {raw_fpr!r})")
    cutoff = form.get("training_cutoff").strip()
    if cutoff:
        ok = bool(_CUTOFF_RE.match(cutoff))
        if ok and len(cutoff) == 10:
            try:
                dt.date.fromisoformat(cutoff)
            except ValueError:
                ok = False
        if not ok:
            errors.append(f"training cutoff must be a month (YYYY-MM) or a date (YYYY-MM-DD) (got {cutoff!r})")
        else:
            out["training_cutoff"] = cutoff
            out["explicit"] = True
    return out


def _check_pickle_choices(model: Path, mopts: dict[str, Any], allow_pickle: bool, errors: list[str]) -> None:
    """A pickle is never opened to look inside, so its kind and feature version must be chosen."""
    from malvalid.loaders.base import is_pickle_artifact

    try:
        pickled = model.is_file() and model.stat().st_size > 0 and is_pickle_artifact(model)
    except OSError:
        return
    if pickled and allow_pickle and (mopts["model_kind"] is None or mopts["feature_version"] is None):
        errors.append(f"{model.name} is a pickle-based file, which malvalid never unpickles to look inside: "
                      "choose its model kind and feature version instead of Auto-detect")


__all__ = ["CALIBRATE_FPRS", "EFFECTIVE_CONFIG_NAME", "FEATURE_VERSIONS", "MODEL_KINDS", "RunRequest",
           "absolutize_policy_paths", "TEXT_FIELDS", "build_request", "default_policy", "deep_merge",
           "CORPUS_DIR_DISABLED", "corpus_dir_allowed", "disabled_feature_errors", "infer_mode", "is_bundled_demo", "model_choices", "resolve_corpus_dir",
           "sticky_form", "truthy"]
