"""JSON API of the local web UI (``docs/WEB_CONTRACT.md`` §5).

``GET /api/runs`` → ``[RunSummary]``; ``GET /api/runs/{id}`` → ``{summary, job, progress, files,
console_tail}``; ``POST /api/runs/{id}/cancel`` and ``POST /api/runs/{id}/delete`` (JSON, or a 303
redirect for a plain HTML form); ``POST /api/validate`` → the ``malvalid validate-adapter --json``
report plus ``exit_code``/``error``. Every POST needs the CSRF token (header or ``csrf`` field).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from malvalid.core import to_jsonable
from malvalid.web.forms import EFFECTIVE_CONFIG_NAME, build_request, disabled_feature_errors, infer_mode
from malvalid.web.jobs import JobError
from malvalid.web.routes import jobs_of, load_record, receive_submission, renderer_of, settings_of, store_of
from malvalid.web.security import CsrfError, csrf_valid, require_csrf, wants_json
from malvalid.web.store import ACTIVE_STATUSES, RunNotFound, tail_text
from malvalid.web.uploads import UploadError, parse_form

log = logging.getLogger("malvalid.web.api")

CONSOLE_TAIL_LINES = 40
MODEL_NOT_VALIDATED = ("“Validate adapter” checks a custom adapter .py file. For a model file, use “Inspect model” "
                       "(malvalid reads the file and shows the kind and feature version it detects).")


class SafeJSONResponse(JSONResponse):
    """JSON with non-finite floats as null (run files are read leniently and may hold NaN)."""

    def render(self, content: Any) -> bytes:
        return json.dumps(to_jsonable(content), ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")


def _error(request: Request, status: int, title: str, message: str) -> Response:
    return renderer_of(request).error(request, status, title, message)


def api_runs(request: Request) -> Response:
    """Every run, newest first; ``?active=1``: only queued/running ones (the dashboard's auto-refresh)."""
    runs = store_of(request).summaries(jobs_of(request).live_jobs())
    if request.query_params.get("active") == "1":
        runs = [r for r in runs if r.get("status") in ACTIVE_STATUSES]
    return SafeJSONResponse(runs)


def api_run(request: Request) -> Response:
    run_id = request.path_params["run_id"]
    try:
        rec = load_record(request, run_id)
    except RunNotFound:
        return _error(request, 404, "Not found", f"there is no run {run_id!r}")
    tail = tail_text(rec.path / "console.log", max_lines=CONSOLE_TAIL_LINES)
    return SafeJSONResponse({"summary": rec.summary, "job": rec.job, "progress": rec.progress,
                             "files": rec.files, "console_tail": tail})


async def _small_form_csrf(request: Request) -> Response | None:
    """Read a small action form (cancel/delete) and enforce CSRF; returns an error response or None."""
    settings = settings_of(request)
    form_token: str | None = None
    if not (request.scope.get("state") or {}).get("csrf_verified"):
        try:
            # Small action forms carry no files; any file part is ignored (never written anywhere).
            form = await parse_form(request, store_of(request).root / ".nonexistent", max_file_bytes=0,
                                    csrf_check=lambda v: csrf_valid(settings, v), max_files=0)
            form_token = form.get("csrf") or None
        except UploadError as e:
            return _error(request, e.status if e.status in (400, 403, 413, 415) else 400, "Request refused", e.message)
    try:
        require_csrf(request, form_token)
    except CsrfError as e:
        return _error(request, 403, "Request refused", str(e))
    return None


async def api_cancel(request: Request) -> Response:
    run_id = request.path_params["run_id"]
    err = await _small_form_csrf(request)
    if err is not None:
        return err
    try:
        job = await run_in_threadpool(jobs_of(request).cancel, run_id)
    except RunNotFound:
        return _error(request, 404, "Not found", f"there is no run {run_id!r}")
    except JobError as e:
        return _error(request, e.status, "Cannot cancel", e.message)
    if not wants_json(request):
        return RedirectResponse(settings_of(request).url(f"/runs/{run_id}"), status_code=303)
    return SafeJSONResponse({"ok": True, "run_id": run_id, "status": job.get("status"), "job": job})


async def api_delete(request: Request) -> Response:
    run_id = request.path_params["run_id"]
    err = await _small_form_csrf(request)
    if err is not None:
        return err
    try:
        await run_in_threadpool(jobs_of(request).delete, run_id)
    except RunNotFound:
        return _error(request, 404, "Not found", f"there is no run {run_id!r}")
    except JobError as e:
        return _error(request, e.status, "Cannot delete", e.message)
    except OSError as e:
        return _error(request, 500, "Cannot delete", f"removing the run directory failed: {e.strerror or e}")
    if not wants_json(request):
        return RedirectResponse(settings_of(request).url("/"), status_code=303)
    return SafeJSONResponse({"ok": True, "run_id": run_id, "deleted": True})


_EFFECTIVE_CONFIG_RE = re.compile(r"[^\s'\"`]*" + re.escape(EFFECTIVE_CONFIG_NAME))


def _relabel_effective_config(result: dict[str, Any], base_config: Any) -> None:
    """validate-adapter names the ``--config`` it got: the server's merged ``.malvalid-gate.yaml``.
    Show the policy the user chose instead, so they are not pointed at a file they never wrote."""
    name = getattr(base_config, "name", None)
    label = f"{name} (with this form's options)" if name else "the default gate policy (with this form's options)"
    for check in result.get("checks") or []:
        if isinstance(check, dict):
            for key in ("detail", "message", "error"):
                text = check.get(key)
                if isinstance(text, str) and EFFECTIVE_CONFIG_NAME in text:
                    text = _EFFECTIVE_CONFIG_RE.sub(label, text)
                    check[key] = ("T" + text[1:]) if text.startswith("the default gate policy") else text


async def api_validate(request: Request) -> Response:
    """Run ``malvalid validate-adapter --json`` on the submitted form (same fields as ``POST /runs``)."""
    settings = settings_of(request)
    store = store_of(request)
    form, sid, err = await receive_submission(request)
    if err is not None or form is None:
        return err  # type: ignore[return-value]
    sdir = store.submission_dir(sid)
    try:
        disabled = disabled_feature_errors(form, settings)
        if disabled:
            return SafeJSONResponse({"ok": False, "checks": [], "errors": disabled}, status_code=403)
        if infer_mode(form) == "model":
            return SafeJSONResponse({"ok": False, "checks": [], "errors": [MODEL_NOT_VALIDATED]}, status_code=400)
        req, errors = await run_in_threadpool(build_request, form, settings, submission_id=sid, submission_dir=sdir)
        if errors or req is None:
            return SafeJSONResponse({"ok": False, "checks": [], "errors": errors}, status_code=400)
        if req.is_model:
            return SafeJSONResponse({"ok": False, "checks": [], "errors": [MODEL_NOT_VALIDATED]}, status_code=400)
        argv = await run_in_threadpool(req.validate_argv, settings, sdir)
        result = await run_in_threadpool(jobs_of(request).validate, argv,
                                         cwd=jobs_of(request).work_dir(store.root))
        _relabel_effective_config(result, req.base_config)
        result.setdefault("errors", [])
        return SafeJSONResponse(result)
    finally:
        await run_in_threadpool(store.remove_submission, sid)


__all__ = ["SafeJSONResponse", "api_cancel", "api_delete", "api_run", "api_runs", "api_validate"]
