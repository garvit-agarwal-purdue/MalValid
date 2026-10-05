"""HTML pages of the local web UI (templates and contexts per ``docs/WEB_CONTRACT.md`` §5).

Page handlers are plain functions (Starlette runs them in its thread pool, so reading run
directories never blocks the event loop). ``POST /runs`` is async: it streams the upload.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

from malvalid.web.compare import (
    MAX_COMPARE,
    MIN_COMPARE,
    best_run_id,
    build_rows,
    config_diff,
    parse_ids,
    policy_differences,
    verdict_lines,
)
from malvalid.web.forms import (
    FEATURE_VERSIONS,
    MODEL_KINDS,
    build_request,
    corpus_dir_allowed,
    default_policy,
    disabled_feature_errors,
    infer_mode,
    sticky_form,
)
from malvalid.web.jobs import JobError, JobManager, JobSpec
from malvalid.web.onboarding import ADAPTER_SKELETON, demo_submission
from malvalid.web.render import Renderer
from malvalid.web.security import CsrfError, RAW_REPORT_MARKER, csrf_valid, require_csrf, wants_json
from malvalid.web.settings import WebSettings
from malvalid.web.store import (
    ACTIVE_STATUSES,
    RAW_FILES,
    RunNotFound,
    RunRecord,
    RunStore,
    read_json_file,
)
from malvalid.web.uploads import MODEL_EXTENSIONS, FormData, SavedFile, UploadError, parse_form

log = logging.getLogger("malvalid.web.routes")

MAX_RAW_REPORT_BYTES = 256 * 1024 * 1024
MAX_RAW_LOG_BYTES = 16 * 1024 * 1024


# --------------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------------


def settings_of(request: Request) -> WebSettings:
    return request.app.state.settings


def store_of(request: Request) -> RunStore:
    return request.app.state.store


def jobs_of(request: Request) -> JobManager:
    return request.app.state.jobs


def renderer_of(request: Request) -> Renderer:
    return request.app.state.renderer


def load_record(request: Request, run_id: str) -> RunRecord:
    """A run's record with the job manager's live job state (raises RunNotFound)."""
    jobs = jobs_of(request)
    store = store_of(request)
    job = jobs.job(run_id) if store.exists(run_id) else None
    return store.load(run_id, job=job)


def not_found(request: Request, what: str = "run") -> Response:
    return renderer_of(request).error(request, 404, "Not found", f"There is no such {what} in this runs directory.")


def dashboard_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"ready": 0, "conditional": 0, "not_ready": 0, "blocked": 0, "running": 0, "failed": 0}
    for r in runs:
        if r.get("status") in ACTIVE_STATUSES:
            counts["running"] += 1
        elif r.get("status") in ("failed", "interrupted"):
            # before the verdict: an exit-2 run may carry a BLOCKED verdict, but its result is not
            # trustworthy, so it counts as a failed run, not a blocked model
            counts["failed"] += 1
        elif r.get("verdict") in counts:
            counts[str(r["verdict"])] += 1
    return counts


def _default_config_text(settings: WebSettings) -> str:
    from malvalid.config import default_config_text

    if settings.config_path is not None:
        try:
            return settings.config_path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("cannot read the server default policy %s: %s", settings.config_path, e)
    return default_config_text()


def _policy(settings: WebSettings) -> Any:
    try:
        return default_policy(settings)
    except Exception as e:
        log.warning("default policy unusable (%s); using the packaged one", e)
        from malvalid.config import load_config

        return load_config()


#: Registered modules that exist for malvalid's own pipeline tests (X0 "Pipeline smoke test"); the web
#: UI does not list them (``malvalid list-modules`` still does).
INTERNAL_MODULES = frozenset({"dummy"})
#: Rows the dashboard shows before "Show all" (the newest first; counts and active runs cover all).
DASHBOARD_LIMIT = 200


def module_infos(settings: WebSettings) -> list[dict[str, Any]]:
    """``Module.info()`` of every registered module (+ ``enabled_by_default`` and verdict ``weight``)."""
    from malvalid import registry
    from malvalid.verdict import UNSCORED_MODULES

    cfg = _policy(settings)
    rows = []
    for mid, cls in registry.modules().items():
        if mid in INTERNAL_MODULES:  # developer smoke tests are not part of the battery researchers see
            continue
        try:
            info = dict(cls.info())
        except Exception as e:  # a broken plugin must not hide the others
            info = {"id": mid, "code": getattr(cls, "code", None), "title": getattr(cls, "title", mid),
                    "description": f"info() failed: {e}"}
        mc = cfg.modules.get(mid)
        info["enabled_by_default"] = bool(mc is not None and mc.enabled)
        # Preconditions (M0) gate the verdict but are not a scored axis: no weight to show.
        info["scored"] = mid not in UNSCORED_MODULES
        info["weight"] = (float(cfg.verdict.weights.get(mid, cfg.verdict.default_weight))
                          if info["scored"] else None)
        info["available"] = True
        rows.append(info)
    for mid, why in sorted(registry.unavailable("modules").items()):
        rows.append({"id": mid, "code": None, "title": mid, "description": f"failed to import: {why}",
                     "requires": [], "requires_any": [], "enhanced_by": [], "screening": False,
                     "default_gate": None, "default_params": {}, "enabled_by_default": False, "weight": 0.0,
                     "available": False})
    return rows


def corpus_infos(settings: WebSettings) -> list[dict[str, Any]]:
    """``CorpusInfo`` for every registered canonical corpus (never loads feature data)."""
    from malvalid import registry

    cfg = _policy(settings)
    out = []
    for name, cls in registry.corpora().items():
        info: dict[str, Any] = {"name": name, "feature_version": None, "version": None, "description": None,
                                "available": False, "location": None, "n": None, "content_hash": None,
                                "synthetic": False, "hint": None}
        try:
            prov = cls()
            pinfo = dict(prov.info())
            use_cfg = cfg if (cfg.corpus == name and cfg.corpus_dir) else None
            loc = prov.locate(use_cfg)
            avail = bool(prov.is_available(use_cfg))
            manifest = read_json_file(loc / "manifest.json", max_bytes=16 * 1024 * 1024)
            info.update(
                feature_version=pinfo.get("feature_version"), version=pinfo.get("version"),
                description=pinfo.get("description"), synthetic=bool(pinfo.get("synthetic")),
                available=avail, location=str(loc),
                expected_content_hash=pinfo.get("expected_content_hash"),
            )
            if manifest is not None:
                info["n"] = manifest.get("n") if isinstance(manifest.get("n"), int) else None
                info["content_hash"] = manifest.get("content_hash") if isinstance(manifest.get("content_hash"), str) else None
            if not avail:
                info["hint"] = prov.unavailable_hint(loc)
            elif manifest is None and info["synthetic"]:
                info["hint"] = "generated on first use (no download needed)"
        except Exception as e:
            info["hint"] = f"corpus provider error: {type(e).__name__}: {e}"
        out.append(info)
    for name, why in sorted(registry.unavailable("corpora").items()):
        out.append({"name": name, "feature_version": None, "version": None, "description": None,
                    "available": False, "location": None, "n": None, "content_hash": None, "synthetic": False,
                    "hint": f"failed to import: {why}"})
    return out


def new_run_context(settings: WebSettings, *, errors: list[str], form: dict[str, Any],
                    notice: str | None = None, inspect: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _policy(settings)
    return {
        "inspect": inspect,
        "modules": module_infos(settings),
        "corpora": corpus_infos(settings),
        "default_config_text": _default_config_text(settings),
        "defaults": {"corpus": cfg.corpus, "seed": cfg.runtime.seed, "fail_on": "blocked"},
        "max_upload_mb": settings.max_upload_mb,
        "errors": list(errors),
        "form": dict(form),
        "notice": notice,
        "adapter_skeleton": ADAPTER_SKELETON,
        "demo": demo_submission(),
        "allow_path_mode": settings.allow_path_mode,
        "allow_no_sandbox": settings.allow_no_sandbox,
    }


def rerun_form(job: dict[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    """New-run form values copied from a web run's ``job.json`` ("Re-run with these settings")."""
    if not isinstance(job, dict):
        return {}, None
    opts = job.get("options") if isinstance(job.get("options"), dict) else {}
    mode = job.get("mode") if job.get("mode") in ("path", "model") else "upload"
    form: dict[str, Any] = {"mode": mode}
    for k in ("class_name", "title", "corpus", "corpus_dir", "fail_on", "training_cutoff"):
        v = opts.get(k)
        if isinstance(v, str) and v:
            form[k] = v
    if opts.get("model_kind") in MODEL_KINDS:
        form["model_kind"] = opts["model_kind"]
    if opts.get("feature_version") in FEATURE_VERSIONS:
        form["feature_version"] = opts["feature_version"]
    if opts.get("threshold_mode") in ("calibrate", "declared"):
        form["threshold_mode"] = opts["threshold_mode"]
    for k in ("threshold", "calibrate_fpr"):
        v = opts.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            form[k] = repr(float(v))
    if mode == "path" and isinstance(opts.get("training_hashes"), str) and opts["training_hashes"]:
        form["training_hashes_path"] = opts["training_hashes"]
    if isinstance(opts.get("seed"), int) and not isinstance(opts.get("seed"), bool):
        form["seed"] = str(opts["seed"])
    for k in ("only", "skip"):
        ids = [str(x) for x in opts.get(k) or [] if isinstance(x, str)]
        if ids:
            form[k] = ",".join(ids)
            form[f"{k}_ids"] = ids
    for k in ("allow_pickle", "no_sandbox"):  # the no-sandbox confirmation must be given again
        if opts.get(k) is True:
            form[k] = "on"
    name = job.get("display_name") or job.get("run_id") or "the earlier run"
    if mode == "path":
        if isinstance(job.get("adapter"), str):
            form["adapter_path"] = job["adapter"]
        if opts.get("config_source") == "path" and isinstance(opts.get("config"), str):
            form["config_path"] = opts["config"]
        notice = f"Settings copied from “{name}”. Check them, fix what needs fixing, and start the run."
    else:
        files = opts.get("files") if isinstance(opts.get("files"), dict) else {}
        names = ([files.get("model"), files.get("adapter")] + list(files.get("models") or [])
                 + [files.get("manifest"), files.get("config")])
        names = [str(n) for n in names if isinstance(n, str) and n]
        notice = (f"Settings copied from “{name}”. Browsers cannot re-use uploaded files, so choose them again"
                  + (f" ({', '.join(names)})" if names else "") + ".")
    return form, notice


# --------------------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------------------


def healthz(request: Request) -> Response:
    return PlainTextResponse("ok")


def dashboard(request: Request) -> Response:
    every = store_of(request).summaries(jobs_of(request).live_jobs())
    show_all = request.query_params.get("all") == "1"
    runs = every if show_all else every[:DASHBOARD_LIMIT]
    ctx = {"runs": runs, "counts": dashboard_counts(every),
           "active": [r for r in every if r.get("status") in ACTIVE_STATUSES],
           "total_runs": len(every), "limited": len(runs) < len(every),
           "adapter_skeleton": ADAPTER_SKELETON, "demo": demo_submission()}
    return renderer_of(request).page(request, "dashboard.html.j2", ctx, nav_active="dashboard")


def _drop_disabled_features(settings: WebSettings, form: dict[str, Any],
                            notice: str | None) -> tuple[dict[str, Any], str | None]:
    """A re-run form without the options this server does not allow (path mode, no sandbox)."""
    dropped: list[str] = []
    if form.get("mode") == "path" and not settings.allow_path_mode:
        for k in ("adapter_path", "training_hashes_path", "config_path"):
            form.pop(k, None)
        form["mode"] = "model"
        dropped.append("“Use files on this machine” is disabled on this server: upload the files instead")
    if form.get("corpus_dir") and not corpus_dir_allowed(form.get("corpus_dir"), settings):
        form.pop("corpus_dir", None)
        dropped.append("A custom corpus directory is disabled on this server: the server's corpus directory is used")
    if form.get("no_sandbox") and not settings.allow_no_sandbox:
        form.pop("no_sandbox", None)
        dropped.append("“Run without the sandbox” is disabled on this server")
    if dropped:
        notice = " ".join(filter(None, [notice, "; ".join(dropped) + "."]))
    return form, notice


def new_run(request: Request) -> Response:
    form: dict[str, Any] = {}
    notice = None
    src = request.query_params.get("from")
    if src:
        job = jobs_of(request).job(src) if store_of(request).exists(src) else None
        form, notice = rerun_form(job)
        if not form:
            notice = "That run has no web submission settings to copy (it was started from the command line)."
        form, notice = _drop_disabled_features(settings_of(request), form, notice)
    ctx = new_run_context(settings_of(request), errors=[], form=form, notice=notice)
    return renderer_of(request).page(request, "new_run.html.j2", ctx, nav_active="new")


def _form_error_response(request: Request, errors: list[str], form: FormData | None, status: int) -> Response:
    if wants_json(request):
        from malvalid.web.api import SafeJSONResponse

        return SafeJSONResponse({"ok": False, "checks": [], "errors": errors}, status_code=status)
    sticky = sticky_form(form)
    inspect = None
    if form is not None and infer_mode(form) == "model":
        inspect = load_inspection(store_of(request), sticky.get("model_submission"))
        if inspect is None:
            sticky.pop("model_submission", None)  # gone (submitted or cleaned up): choose the file again
    ctx = new_run_context(settings_of(request), errors=errors, form=sticky, inspect=inspect)
    return renderer_of(request).page(request, "new_run.html.j2", ctx, status=status, nav_active="new")


# --------------------------------------------------------------------------------------------------
# Model inspection (``POST /runs/inspect``): upload a model, see what malvalid detects, then start
# --------------------------------------------------------------------------------------------------

INSPECTION_FILE = ".inspect.json"  # dot-leading: never an uploaded name


def _int_or_none(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _str_or_none(v: Any) -> str | None:
    return v if isinstance(v, str) and v else None


def inspection_view(result: dict[str, Any], *, model_submission: str | None, file_name: str,
                    size: int | None) -> dict[str, Any]:
    """The ``malvalid inspect-model --json`` result as the new-run page and its JSON answer use it."""
    kind = result.get("model_kind")
    fv = result.get("feature_version")
    supported = result.get("supported") if isinstance(result.get("supported"), dict) else {}
    details = result.get("details") if isinstance(result.get("details"), dict) else {}
    return {
        "ok": result.get("ok") is True,
        "model_submission": model_submission,
        "file_name": file_name,
        "size": size,
        "format": _str_or_none(result.get("format")),
        "model_kind": kind if kind in MODEL_KINDS else None,
        "n_features": _int_or_none(result.get("n_features")),
        "feature_version": fv if fv in FEATURE_VERSIONS else None,
        "default_corpus": _str_or_none(result.get("default_corpus")),
        "is_pickle": result.get("is_pickle") is True,
        "notes": [str(x) for x in result.get("notes") or [] if isinstance(x, str)][:20],
        "errors": [str(x) for x in result.get("errors") or [] if isinstance(x, str)][:20],
        "error": _str_or_none(result.get("error")),
        "supported": {str(k): str(v) for k, v in supported.items() if isinstance(v, str)},
        "details": {str(k): v for k, v in details.items() if isinstance(v, (str, int, float, bool)) or v is None},
    }


def load_inspection(store: RunStore, submission_id: Any) -> dict[str, Any] | None:
    """The saved inspection of a pending inspected upload (None if it is gone or not one)."""
    path = store.inspected_model(submission_id, MODEL_EXTENSIONS)
    if path is None:
        return None
    saved = read_json_file(path.parent / INSPECTION_FILE, max_bytes=1024 * 1024) or {}
    try:
        size = path.stat().st_size
    except OSError:
        return None
    return inspection_view(saved, model_submission=str(submission_id), file_name=path.name, size=size)


def _apply_detection(form: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    """Pre-select what was detected where the form still says Auto-detect."""
    for key in ("model_kind", "feature_version"):
        if not str(form.get(key) or "").strip() or str(form.get(key)).strip().lower() == "auto":
            if view.get(key) and not (key == "feature_version" and view.get("is_pickle")):
                form[key] = view[key]
    return form


def _discard_other_files(form: FormData, keep: SavedFile | None) -> list[str]:
    """Delete every uploaded file but ``keep`` (an inspection keeps only the model); returns their names."""
    gone = []
    for f in form.all_files():
        if keep is not None and f.path == keep.path:
            continue
        f.path.unlink(missing_ok=True)
        gone.append(f.name)
    return gone


async def inspect_model(request: Request) -> Response:
    """``POST /runs/inspect``: store the uploaded model (or take an earlier inspected one), run
    ``malvalid inspect-model --json`` on it in a subprocess, and show the form again with what was
    detected (JSON when the request asks for it)."""
    settings = settings_of(request)
    store = store_of(request)
    jobs = jobs_of(request)
    form, sid, err = await receive_submission(request)
    if err is not None or form is None:
        return err  # type: ignore[return-value]
    disabled = disabled_feature_errors(form, settings)
    if disabled:
        await run_in_threadpool(store.remove_submission, sid)
        return _form_error_response(request, disabled, form, 403)
    sticky = sticky_form(form)
    sticky.pop("uploaded", None)
    previous = form.get("model_submission").strip() or None
    upload = form.file("model_file")
    discarded: list[str] = []
    problems = [p for p in form.problems if p.startswith(("model file", "only one model file"))]
    path_mode = infer_mode(form) == "path"
    if path_mode:  # a model file already on this machine: inspected where it is
        await run_in_threadpool(store.remove_submission, sid)
        raw = form.get("adapter_path").strip()
        target = None
        if raw:
            p = Path(raw).expanduser()
            with contextlib.suppress(OSError, RuntimeError):
                p = p.resolve()
            if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS:
                target = p
        if target is None:
            return _form_error_response(request, ["“Inspect model” needs the path of a model file "
                                                  f"({' '.join(MODEL_EXTENSIONS)}) on this machine"], form, 400)
        insp_sid = None
    elif upload is not None:
        if problems or upload.size == 0:
            await run_in_threadpool(store.remove_submission, sid)
            return _form_error_response(request, problems or [f"model file {upload.name} is empty"], form, 400)
        discarded = await run_in_threadpool(_discard_other_files, form, upload)
        await run_in_threadpool(store.mark_inspection, sid)
        if previous and previous != sid and store.inspection_dir(previous) is not None:
            await run_in_threadpool(store.remove_submission, previous)  # replaced by this upload
        target = upload.path
        insp_sid = sid
    else:
        await run_in_threadpool(store.remove_submission, sid)
        stored = store.inspected_model(previous, MODEL_EXTENSIONS) if previous else None
        if stored is None:
            sticky.pop("model_submission", None)
            msgs = problems or [("choose your model file, then click “Inspect model”" if not previous else
                                 "the model uploaded earlier is no longer on the server; choose the file again")]
            if wants_json(request):
                from malvalid.web.api import SafeJSONResponse

                return SafeJSONResponse({"ok": False, "errors": msgs}, status_code=400)
            ctx = new_run_context(settings, errors=msgs, form={**sticky, "mode": "model"})
            return renderer_of(request).page(request, "new_run.html.j2", ctx, status=400, nav_active="new")
        target = stored
        insp_sid = previous
    if insp_sid is not None:
        keep = [insp_sid]
        await run_in_threadpool(lambda: store.cleanup_inspections(keep=keep))
    result = await run_in_threadpool(jobs.inspect, target, cwd=jobs.work_dir(store.root))
    try:
        size = target.stat().st_size
    except OSError:
        size = None
    view = inspection_view(result, model_submission=insp_sid, file_name=target.name, size=size)
    if insp_sid is not None:
        def _save() -> None:
            from malvalid.web.store import write_json_atomic

            with contextlib.suppress(OSError, TypeError, ValueError):
                write_json_atomic(store.submission_dir(insp_sid) / INSPECTION_FILE, view)
        await run_in_threadpool(_save)
    if wants_json(request):
        from malvalid.web.api import SafeJSONResponse

        return SafeJSONResponse({**view, "discarded": discarded})
    form_ctx = _apply_detection(dict(sticky), view)
    if insp_sid is not None:
        form_ctx["mode"] = "model"
        form_ctx["model_submission"] = insp_sid
    notice = None
    if discarded:
        notice = ("Only the model file is kept between “Inspect model” and “Start run”: choose "
                  + ", ".join(discarded) + " again.")
    ctx = new_run_context(settings, errors=[], form=form_ctx, notice=notice, inspect=view)
    return renderer_of(request).page(request, "new_run.html.j2", ctx, nav_active="new")


def adopt_inspected_model(store: RunStore, form: FormData) -> tuple[str | None, str | None]:
    """Model mode without a new model file: put the inspected upload named by ``model_submission`` into
    the form (its file stays where it is until the run is queued). Returns ``(submission id, error)``."""
    if infer_mode(form) != "model" or form.file("model_file") is not None:
        return None, None
    sid = form.get("model_submission").strip()
    if not sid:
        return None, None
    path = store.inspected_model(sid, MODEL_EXTENSIONS)
    if path is None:
        return None, ("the model uploaded with “Inspect model” is no longer on the server (it was used by "
                      "another run or cleaned up); choose the model file again")
    try:
        size = path.stat().st_size
    except OSError:
        return None, "the inspected model file cannot be read; choose the model file again"
    form.files["model_file"] = [SavedFile(field="model_file", name=path.name, original=path.name, path=path,
                                          size=size)]
    return sid, None


async def receive_submission(request: Request) -> tuple[FormData | None, str, Response | None]:
    """Stream the new-run form into a fresh submission dir; verify CSRF.

    Returns ``(form, submission_id, None)`` or ``(None, submission_id, error_response)`` (the
    submission dir is removed on error).
    """
    settings = settings_of(request)
    store = store_of(request)
    sid, sdir = await run_in_threadpool(store.new_submission_dir)
    try:
        form = await parse_form(request, sdir, max_file_bytes=settings.max_upload_bytes,
                                csrf_check=lambda v: csrf_valid(settings, v))
    except UploadError as e:
        await run_in_threadpool(store.remove_submission, sid)
        if e.status == 403:
            return None, sid, renderer_of(request).error(request, 403, "Request refused", e.message)
        return None, sid, _form_error_response(request, [e.message], None, e.status)
    try:
        require_csrf(request, form.get("csrf") or None)
    except CsrfError as e:
        await run_in_threadpool(store.remove_submission, sid)
        return None, sid, renderer_of(request).error(request, 403, "Request refused", str(e))
    return form, sid, None


async def submit_run(request: Request) -> Response:
    """``POST /runs``: validate the form, queue the run, 303 to its page (400 + the form on errors)."""
    settings = settings_of(request)
    store = store_of(request)
    form, sid, err = await receive_submission(request)
    if err is not None or form is None:
        return err  # type: ignore[return-value]
    disabled = disabled_feature_errors(form, settings)
    if disabled:
        await run_in_threadpool(store.remove_submission, sid)
        return _form_error_response(request, disabled, form, 403)
    sdir = store.submission_dir(sid)
    reused, reuse_error = await run_in_threadpool(adopt_inspected_model, store, form)
    if reuse_error is not None:
        await run_in_threadpool(store.remove_submission, sid)
        return _form_error_response(request, [reuse_error], form, 400)
    req, errors = await run_in_threadpool(build_request, form, settings, submission_id=sid, submission_dir=sdir)
    if errors or req is None:
        await run_in_threadpool(store.remove_submission, sid)
        return _form_error_response(request, errors, form, 400)
    previous = form.get("model_submission").strip() or None

    def _prepare() -> JobSpec:
        if req.mode == "path":
            for f in form.all_files():  # files chosen before switching to path mode are not used
                f.path.unlink(missing_ok=True)
            store.remove_submission(sid)  # nothing of a path-mode run is stored here
            req.submission_id = None
            req.submission_dir = None
        elif req.is_model:
            for fld in ("adapter_file", "model_files"):  # custom-adapter files are not used by a model run
                for f in form.files.get(fld) or []:
                    f.path.unlink(missing_ok=True)
            if reused is not None and req.model is not None:
                # take over the inspected model: move it into this run's submission, drop the old dir
                dest = sdir / req.model.name
                os.replace(req.model, dest)
                req.use_model(dest)
                store.remove_submission(reused)
            elif previous and store.inspection_dir(previous) is not None:
                store.remove_submission(previous)  # a newly chosen model file replaced the inspected one
        config = req.run_config()
        return JobSpec(argv=req.run_argv(settings, config), mode=req.mode, adapter=str(req.adapter),
                       display_name=req.display_name, submission_id=req.submission_id, options=req.options())

    try:
        spec = await run_in_threadpool(_prepare)
        run_id = await run_in_threadpool(jobs_of(request).submit, spec)
    except JobError as e:
        await run_in_threadpool(store.remove_submission, sid)
        return renderer_of(request).error(request, e.status, "Could not queue the run", e.message)
    except OSError as e:
        await run_in_threadpool(store.remove_submission, sid)
        return renderer_of(request).error(request, 500, "Could not queue the run", f"{e.strerror or e}")
    return RedirectResponse(settings.url(f"/runs/{run_id}"), status_code=303)


def run_detail(request: Request) -> Response:
    run_id = request.path_params["run_id"]
    try:
        rec = load_record(request, run_id)
    except RunNotFound:
        return not_found(request)
    ctx = {"run": rec.summary, "job": rec.job, "progress": rec.progress, "report": rec.report, "files": rec.files,
           "problem": rec.problem, "stale_report": rec.stale_report,
           "queue_ahead": jobs_of(request).queue_ahead(run_id)}
    return renderer_of(request).page(request, "run_detail.html.j2", ctx, nav_active="dashboard")


def raw_file(request: Request) -> Response:
    """``/runs/{id}/report.html | report.json | run.log | console.log`` (allowlist; 404 otherwise)."""
    run_id = request.path_params["run_id"]
    name = request.path_params["name"]
    try:
        path = store_of(request).raw_file(run_id, name)
    except RunNotFound:
        path = None
    if path is None:
        return not_found(request, "file")
    is_log = name.endswith(".log")
    limit = MAX_RAW_LOG_BYTES if is_log else MAX_RAW_REPORT_BYTES
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > limit and not is_log:
                return renderer_of(request).error(request, 413, "File too large",
                                                  f"{name} is {size} bytes; open it from {path.parent}.")
            f.seek(max(0, size - limit))
            data = f.read(limit)
    except OSError:
        return not_found(request, "file")
    if is_log and size > limit:
        data = f"[… first {size - limit} bytes omitted …]\n".encode() + data
    headers = {RAW_REPORT_MARKER: "1"} if name == "report.html" else {}
    return Response(data, media_type=RAW_FILES[name], headers=headers)


def compare(request: Request) -> Response:
    ids = parse_ids(request.query_params.getlist("ids"))
    errors: list[str] = []
    if len(ids) > MAX_COMPARE:
        errors.append(f"compare at most {MAX_COMPARE} runs at a time; showing the first {MAX_COMPARE}")
        ids = ids[:MAX_COMPARE]
    records: list[RunRecord] = []
    for rid in ids:
        try:
            records.append(load_record(request, rid))
        except RunNotFound:
            errors.append(f"there is no run {rid!r}")
    status = 200
    if ids and len(records) < MIN_COMPARE:
        errors.append(f"pick at least {MIN_COMPARE} runs to compare")
        status = 400
    summaries = [r.summary for r in records]
    reports = [r.report for r in records]
    enough = len(records) >= MIN_COMPARE
    diff = config_diff(reports) if enough else []
    policy_keys = policy_differences(diff)
    corpora_used = {sm.get("corpus") for sm in summaries if sm.get("corpus")}
    ctx = {
        "runs": summaries,
        "rows": build_rows(reports) if enough else [],
        "config_diff": diff,
        "best_run_id": best_run_id(summaries) if enough else None,
        # "Highest score" only means "better model" when the runs were scored the same way
        "best_comparable": enough and not policy_keys and len(corpora_used) <= 1,
        "policy_keys": policy_keys,
        "verdict_lines": verdict_lines(summaries, reports) if enough else [],
        "errors": errors,
        "selected_ids": [r.run_id for r in records],
    }
    return renderer_of(request).page(request, "compare.html.j2", ctx, status=status, nav_active="compare")


def corpora(request: Request) -> Response:
    ctx = {"corpora": corpus_infos(settings_of(request))}
    return renderer_of(request).page(request, "corpora.html.j2", ctx, nav_active="corpora")


def modules(request: Request) -> Response:
    s = settings_of(request)
    ctx = {"modules": module_infos(s), "default_config_text": _default_config_text(s)}
    return renderer_of(request).page(request, "modules.html.j2", ctx, nav_active="modules")


def probe_sandbox() -> dict[str, Any]:
    try:
        from malvalid.sandbox.host import probe_backends
    except Exception as e:
        return {"sandbox": {"available": False, "network_isolated": False,
                            "detail": f"the malvalid sandbox is not available in this installation ({e})"}}
    try:
        return probe_backends()
    except Exception as e:
        return {"probe": {"available": False, "network_isolated": False, "detail": f"probe failed: {e}"}}


def environment_info(settings: WebSettings) -> dict[str, Any]:
    from importlib import metadata as md

    from malvalid.environment import capture_environment

    try:
        env = dict(capture_environment())
    except Exception as e:  # pragma: no cover - defensive
        env = {"error": str(e)}
    web: dict[str, str | None] = {}
    for dist in ("starlette", "uvicorn", "python-multipart", "httpx"):
        try:
            web[dist] = md.version(dist)
        except md.PackageNotFoundError:
            web[dist] = None
    env["web"] = web
    env["python_executable"] = sys.executable
    env["working_directory"] = os.getcwd()
    env["corpus_dir"] = os.environ.get("MALVALID_CORPUS_DIR")
    return env


def system(request: Request) -> Response:
    s = settings_of(request)
    ctx = {"backends": probe_sandbox(), "environment": environment_info(s), "settings": s.public_dict()}
    return renderer_of(request).page(request, "system.html.j2", ctx, nav_active="system")


__all__ = [
    "compare", "corpora", "corpus_infos", "dashboard", "dashboard_counts", "healthz", "inspect_model",
    "inspection_view", "load_inspection", "module_infos", "modules", "new_run", "new_run_context", "raw_file",
    "receive_submission", "run_detail", "submit_run", "system",
]
