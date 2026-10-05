"""Page rendering: the Jinja environment, the common template context and error pages.

Templates live in ``malvalid/web/templates/`` (``*.html.j2``, all extending ``base.html.j2``); display
filters come from :data:`malvalid.web.filters.FILTERS` (identity fallbacks if that module cannot be
imported). The environment autoescapes everything and uses ``ChainableUndefined`` (a missing key
renders empty and never raises).

Every page gets the common context ``app_version, csrf_token, nav_active, bind, runs_dir,
request_path, root`` on top of its own (``root``: the ``--root-path`` URL prefix, also a Jinja
global; every app link is written ``{{ root }}/...``). :attr:`Renderer.captured` is a test hook: set
it to a list and every render appends ``(template_name, context)``.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Callable

import jinja2
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from malvalid import __version__
from malvalid.web.security import security_headers, wants_json
from malvalid.web.settings import WebSettings

log = logging.getLogger("malvalid.web.render")

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
CONTRACT_FILTERS = ("fmt_score", "fmt_pct", "fmt_time", "fmt_duration", "verdict_class", "status_class",
                    "verdict_label")


def _identity(value: Any = None, *args: Any, **kwargs: Any) -> Any:
    return "—" if value is None else value


def load_filters() -> dict[str, Callable[..., Any]]:
    """The frontend's display filters, or identity stand-ins if they cannot be imported."""
    try:
        from malvalid.web.filters import FILTERS

        filters = dict(FILTERS)
    except Exception as e:  # a broken filters module must not take the server down
        log.warning("display filters unavailable (%s: %s); using identity filters", type(e).__name__, e)
        filters = {}
    for name in CONTRACT_FILTERS:
        filters.setdefault(name, _identity)
    return filters


def make_environment(template_dir: Path | None = None) -> jinja2.Environment:
    """The Jinja environment of the web UI (autoescape on, lenient undefined, contract filters)."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir or TEMPLATE_DIR)),
        autoescape=True,
        undefined=jinja2.ChainableUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(load_filters())
    return env


def _fallback_page(template: str, context: dict[str, Any], status: int, root: str = "") -> str:
    """Minimal page used when a template is missing (e.g. a partial installation)."""
    title = context.get("title") or template.replace(".html.j2", "").replace("_", " ")
    body = context.get("message") or ""
    errors = context.get("errors") or []
    shown = {k: v for k, v in context.items() if k not in ("csrf_token",)}
    try:
        dump = json.dumps(shown, indent=2, default=str)[:200000]
    except (TypeError, ValueError):
        dump = repr(shown)[:200000]
    items = "".join(f"<li>{html.escape(str(e))}</li>" for e in errors)
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(str(title))} · MalValid</title></head><body>"
        f"<p><a href=\"{html.escape(root)}/\">MalValid</a></p><h1>{html.escape(str(title))}</h1>"
        f"<p>{html.escape(str(body))}</p>" + (f"<ul>{items}</ul>" if items else "")
        + f"<p>(template {html.escape(template)} is not installed; showing the raw page data, status {status})</p>"
        f"<pre>{html.escape(dump)}</pre></body></html>"
    )


class Renderer:
    """Renders pages and error responses for one app."""

    def __init__(self, settings: WebSettings, template_dir: Path | None = None):
        self.settings = settings
        self.env = make_environment(template_dir)
        #: URL prefix (``--root-path``) for every app link; a global so imported macros see it too.
        self.env.globals["root"] = settings.root_path
        self.captured: list[tuple[str, dict[str, Any]]] | None = None
        self._no_os_sandbox: bool | None = None

    def common(self, request: Request, nav_active: str | None) -> dict[str, Any]:
        authed = bool((request.scope.get("state") or {}).get("authenticated"))
        return {
            "app_version": __version__,
            "csrf_token": self.settings.csrf_token if authed else "",
            "nav_active": nav_active,
            "bind": self.settings.bind,
            "runs_dir": str(self.settings.runs_dir),
            "request_path": request.url.path,
            "root": self.settings.root_path,
            "allow_path_mode": self.settings.allow_path_mode,
            "allow_no_sandbox": self.settings.allow_no_sandbox,
            "reduced_isolation": self._reduced_isolation(),
        }

    def _reduced_isolation(self) -> bool:
        """Started with ``--allow-reduced-isolation`` on a machine without an OS sandbox (probed once)."""
        if not self.settings.allow_reduced_isolation:
            return False
        if self._no_os_sandbox is None:
            try:
                from malvalid.sandbox.host import os_sandbox_available

                self._no_os_sandbox = not os_sandbox_available()
            except Exception:  # noqa: BLE001 - the banner is advisory
                self._no_os_sandbox = True
        return self._no_os_sandbox

    def html(self, request: Request, template: str, context: dict[str, Any], *, status: int = 200,
             nav_active: str | None = None) -> str:
        ctx = {**self.common(request, nav_active), **context}
        if self.captured is not None:
            self.captured.append((template, ctx))
        try:
            tpl = self.env.get_template(template)
        except jinja2.TemplateNotFound:
            log.warning("template %s not found; rendering a fallback page", template)
            return _fallback_page(template, ctx, status, self.settings.root_path)
        return tpl.render(ctx)

    def page(self, request: Request, template: str, context: dict[str, Any], *, status: int = 200,
             nav_active: str | None = None) -> Response:
        try:
            body = self.html(request, template, context, status=status, nav_active=nav_active)
        except Exception:
            log.exception("rendering %s failed", template)
            return self.error(request, 500, "Page error",
                              "This page could not be rendered (see the server log for details).")
        return HTMLResponse(body, status_code=status)

    def error(self, request: Request, status: int, title: str, message: str) -> Response:
        """An error page (HTML) or ``{"error", "message"}`` (API / JSON clients)."""
        if wants_json(request):
            return JSONResponse({"error": title, "message": message, "status": status}, status_code=status)
        ctx = {"status": status, "title": title, "message": message}
        try:
            body = self.html(request, "error.html.j2", ctx, status=status)
        except Exception:
            log.exception("rendering the error page failed")
            body = _fallback_page("error.html.j2", {**ctx, "errors": []}, status, self.settings.root_path)
        return HTMLResponse(body, status_code=status)

    def server_error(self, request: Request, exc: Exception) -> Response:
        """500 handler (runs outside the security middleware, so it sets the headers itself)."""
        log.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        resp = self.error(request, 500, "Internal error",
                          "Something went wrong inside the MalValid UI; details are in the server log.")
        for k, v in security_headers().items():
            resp.headers[k] = v
        return resp


__all__ = ["CONTRACT_FILTERS", "Renderer", "STATIC_DIR", "TEMPLATE_DIR", "load_filters", "make_environment"]
