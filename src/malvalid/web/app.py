"""Assembly of the local web UI: :func:`create_app` wires settings, store, jobs, pages and security.

Security model (``docs/WEB_CONTRACT.md`` §1, ``docs/web.md``): bound to loopback by default, a
per-launch token in an HttpOnly/SameSite=Strict cookie (or ``Authorization: Bearer``), a Host-header
allowlist, CSRF tokens on every POST, a strict CSP. The web server never imports an adapter, loads a
model or unpickles anything: every run, adapter validation and model inspection is a ``malvalid``
subprocess.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from pathlib import Path
from typing import Any, AsyncIterator

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from malvalid.web import api, routes
from malvalid.web.jobs import JobManager
from malvalid.web.render import STATIC_DIR, Renderer
from malvalid.web.security import SecurityMiddleware
from malvalid.web.settings import WebSettings
from malvalid.web.store import RunStore

log = logging.getLogger("malvalid.web")

_HTTP_TITLES = {400: "Bad request", 403: "Forbidden", 404: "Not found", 405: "Method not allowed",
                413: "Too large", 415: "Unsupported request"}


def _warm_plugins() -> None:
    """Import the plugin registry in the background so the first form page is quick."""
    try:
        from malvalid import registry

        registry.modules()
        registry.corpora()
    except Exception as e:  # pragma: no cover - diagnostics only
        log.debug("plugin warm-up failed: %s", e)


def create_app(settings: WebSettings, *, template_dir: Path | None = None, warm_plugins: bool = True) -> Starlette:
    """The Starlette app of ``malvalid serve``. Jobs start with the app's lifespan (or on first submit)."""
    store = RunStore(settings.runs_dir)
    jobs = JobManager(settings, store)
    renderer = Renderer(settings, template_dir)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        await run_in_threadpool(jobs.start)
        if warm_plugins:
            threading.Thread(target=_warm_plugins, name="malvalid-web-warmup", daemon=True).start()
        try:
            yield
        finally:
            await run_in_threadpool(jobs.shutdown)

    async def http_exception(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPException)
        title = _HTTP_TITLES.get(exc.status_code, "Error")
        detail = str(exc.detail or "")
        if exc.status_code == 404:
            detail = "There is no such page."
        return renderer.error(request, exc.status_code, title, detail or title)

    async def server_error(request: Request, exc: Exception) -> Response:
        return renderer.server_error(request, exc)

    route_list: list[Any] = [
        Route("/", routes.dashboard, name="dashboard"),
        Route("/healthz", routes.healthz, name="healthz"),
        Route("/runs/new", routes.new_run, name="new_run"),
        Route("/runs", routes.submit_run, methods=["POST"], name="submit_run"),
        Route("/runs/inspect", routes.inspect_model, methods=["POST"], name="inspect_model"),
        Route("/runs/{run_id}", routes.run_detail, name="run_detail"),
        Route("/runs/{run_id}/{name}", routes.raw_file, name="raw_file"),
        Route("/compare", routes.compare, name="compare"),
        Route("/corpora", routes.corpora, name="corpora"),
        Route("/modules", routes.modules, name="modules"),
        Route("/system", routes.system, name="system"),
        Route("/api/validate", api.api_validate, methods=["POST"], name="api_validate"),
        Route("/api/runs", api.api_runs, name="api_runs"),
        Route("/api/runs/{run_id}", api.api_run, name="api_run"),
        Route("/api/runs/{run_id}/cancel", api.api_cancel, methods=["POST"], name="api_cancel"),
        Route("/api/runs/{run_id}/delete", api.api_delete, methods=["POST"], name="api_delete"),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static"),
    ]
    app = Starlette(
        routes=route_list,
        lifespan=lifespan,
        exception_handlers={HTTPException: http_exception, Exception: server_error},
    )
    app.state.settings = settings
    app.state.store = store
    app.state.jobs = jobs
    app.state.renderer = renderer
    app.state.error_response = renderer.error
    app.add_middleware(SecurityMiddleware, settings=settings, error_response=renderer.error)
    return app


__all__ = ["create_app"]
