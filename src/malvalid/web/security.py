"""Security layer of the local web UI: one ASGI middleware in front of every route.

In order, for each HTTP request:

1. **Host allowlist** (DNS-rebinding defence): the ``Host`` header must be one of
   :meth:`WebSettings.allowed_hosts` (``127.0.0.1:<port>``, ``localhost:<port>``, ``[::1]:<port>``,
   plus the bound host with ``--allow-remote``); anything else gets 400.
2. **Login**: ``GET ...?token=<token>`` with the right token — or ``?login=<nonce>`` with a one-time
   nonce from :meth:`WebSettings.browser_login_url` (the link handed to the browser on launch, so the
   token never sits on a browser command line) — creates a session: a random session id goes into the
   ``malvalid_session_<port>`` cookie (HttpOnly, SameSite=Strict, Path=/ or ``<root_path>/``), and the response
   303-redirects to the same URL without the secret. The launch token itself is never put in a cookie:
   browsers send loopback cookies to every port of the same host name, so another local user's
   listener could otherwise capture it. The printed link also uses a secret per-launch
   ``mg-<hex>.localhost`` host name, so the session cookie is not sent to ``127.0.0.1:*`` at all.
3. **Authentication**: every route except ``GET /healthz`` and ``/static/*`` needs a live session
   cookie or ``Authorization: Bearer <token>`` (constant-time comparison). Unauthenticated HTML
   requests get a 401 page explaining to open the URL printed by ``malvalid serve``; API requests
   get JSON. Nothing shown to an unauthenticated client names the secret host name.
4. **CSRF** on state-changing methods: ``Origin`` / ``Referer`` (when present) must be on the
   allowlist and ``Sec-Fetch-Site`` must not be cross-site; then either the ``X-CSRF-Token`` header
   equals ``hmac(token, "csrf")`` (checked here) or — for HTML form posts — the ``csrf`` form field
   does, which the route checks with :func:`require_csrf` after parsing the form (a multipart
   upload is streamed, so it cannot be read here). Requests that are neither get 403.
5. **Response headers**: CSP (no inline script/style), ``nosniff``, ``no-referrer``, ``no-store``,
   frame/opener isolation. ``report.html`` keeps its own CSP (it pins its one inline script by
   hash) and is additionally sandboxed by a header CSP.

**Behind a reverse proxy** (``--root-path`` / ``--trusted-host``, e.g. Open OnDemand's
``/node/<host>/<port>/``): a request path that starts with ``<root_path>/`` has the prefix removed
before routing (a prefix-keeping proxy); any other path is routed as is (a prefix-stripping proxy or
direct access); ``<root_path>`` alone redirects to ``<root_path>/``. Every link, redirect and the
cookie ``Path`` carry the prefix. ``--trusted-host`` only extends the Host and Origin/Referer
allowlist; ``--allow-client`` (optional) refuses TCP peers other than loopback and the proxy; the token login, CSRF and the headers above are unchanged, and ``X-Forwarded-*`` headers
are ignored.

``Origin: null`` is treated as "no origin information": browsers send it for same-origin form posts
from pages served with ``Referrer-Policy: no-referrer`` (as all app pages are), so rejecting it
would break the app's own forms; the CSRF token remains required.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any, Callable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from malvalid.web.settings import WebSettings

log = logging.getLogger("malvalid.web.security")

#: CSP of every app page: no inline script/style, frames only from ourselves.
APP_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-src 'self'; "
    "frame-ancestors 'self'; form-action 'self'; base-uri 'none'"
)
#: Extra header CSP for a served ``report.html``: it carries its own (meta) CSP with a script hash;
#: this only sandboxes it (opaque origin, even when opened directly) and limits who may frame it.
REPORT_CSP = "sandbox allow-scripts allow-popups allow-popups-to-escape-sandbox; frame-ancestors 'self'"
#: Response header a route sets to mark a raw ``report.html``; the middleware strips it.
RAW_REPORT_MARKER = "x-malvalid-raw-report"

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
EXEMPT_PREFIXES = ("/static/",)
EXEMPT_PATHS = frozenset({"/healthz"})
FORM_CONTENT_TYPES = ("multipart/form-data", "application/x-www-form-urlencoded")

ErrorFactory = Callable[[Request, int, str, str], Response]


def _eq(a: str | None, b: str) -> bool:
    if a is None:
        return False
    return hmac.compare_digest(a.encode("utf-8", "replace"), b.encode("utf-8", "replace"))


def token_valid(settings: WebSettings, value: str | None) -> bool:
    return _eq(value, settings.token)


def csrf_valid(settings: WebSettings, value: str | None) -> bool:
    return _eq(value, settings.csrf_token)


def bearer_token(headers: Headers | Mapping[str, str]) -> str | None:
    auth = headers.get("authorization") or ""
    scheme, _, value = auth.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


def is_authenticated(settings: WebSettings, request: Request) -> bool:
    return settings.sessions.valid(request.cookies.get(settings.cookie_name)) or token_valid(
        settings, bearer_token(request.headers)
    )


def security_headers(*, raw_report: bool = False) -> dict[str, str]:
    """Headers added to every response (``raw_report``: a served ``report.html``)."""
    return {
        "Content-Security-Policy": REPORT_CSP if raw_report else APP_CSP,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-store",
        "X-Frame-Options": "SAMEORIGIN",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }


def apply_security_headers(headers: MutableHeaders) -> None:
    raw = headers.get(RAW_REPORT_MARKER) is not None
    if raw:
        del headers[RAW_REPORT_MARKER]
    for k, v in security_headers(raw_report=raw).items():
        headers[k] = v


def wants_json(request: Request) -> bool:
    """Should errors / results for this request be JSON (API, fetch) rather than an HTML page?"""
    accept = (request.headers.get("accept") or "").lower()
    if request.url.path.startswith("/api/"):
        # A no-JS HTML form posting to /api/... prefers a redirect / HTML page.
        return not ("text/html" in accept and "application/json" not in accept)
    return "application/json" in accept and "text/html" not in accept


class CsrfError(Exception):
    """The request carries no valid CSRF token."""


def require_csrf(request: Request, form_value: str | None) -> None:
    """Raise :class:`CsrfError` unless the middleware verified ``X-CSRF-Token`` or ``form_value`` is valid.

    Every route handling a state-changing request must call this before acting.
    """
    settings: WebSettings = request.app.state.settings
    state = request.scope.get("state") or {}
    if state.get("csrf_verified"):
        return
    if csrf_valid(settings, form_value):
        request.scope.setdefault("state", {})["csrf_verified"] = True
        return
    raise CsrfError("missing or invalid CSRF token (reload the page and try again)")


def _netloc(url: str) -> str | None:
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return parts.netloc.lower()


class SecurityMiddleware:
    """Host allowlist, token login/authentication, CSRF origin checks and security headers."""

    def __init__(self, app: ASGIApp, *, settings: WebSettings, error_response: ErrorFactory):
        self.app = app
        self.settings = settings
        self.allowed = settings.allowed_hosts()
        self.error_response = error_response

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":  # the UI has no websockets
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                apply_security_headers(MutableHeaders(scope=message))
            await send(message)

        request = Request(scope)
        response = self._check(scope, request)
        if response is not None:
            await response(scope, receive, send_with_headers)
            return
        await self.app(scope, receive, send_with_headers)

    # ---- checks ----------------------------------------------------------------------------------

    def _check(self, scope: Scope, request: Request) -> Response | None:
        s = self.settings
        peer = (scope.get("client") or (None, None))[0]
        if not s.client_allowed(peer):
            log.warning("refused a connection from %s (not in --allow-client)", str(peer)[:64])
            return PlainTextResponse("403 Forbidden: this malvalid server only accepts connections from its "
                                     "configured reverse proxy (--allow-client).", status_code=403)
        host = (request.headers.get("host") or "").strip().lower()
        if host not in self.allowed:
            log.warning("rejected request with Host header %r", host[:200])
            return PlainTextResponse(
                "400 Bad Request: unexpected Host header. The malvalid UI only answers requests addressed to "
                "the link printed by malvalid serve (DNS-rebinding protection).",
                status_code=400,
            )
        state: dict[str, Any] = scope.setdefault("state", {})
        path = scope.get("path") or "/"
        method = scope.get("method", "GET").upper()
        root = s.root_path
        if root:
            if path == root:  # "/node/h/p" → "/node/h/p/": the cookie Path needs the trailing slash
                qs = (scope.get("query_string") or b"").decode("latin-1")
                return RedirectResponse(root + "/" + ("?" + qs if qs else ""), status_code=307)
            if path.startswith(root + "/"):  # a prefix-keeping proxy: route on the app path
                path = path[len(root):]
                scope["path"] = path
                raw = scope.get("raw_path")
                if isinstance(raw, bytes) and raw.startswith(root.encode("latin-1") + b"/"):
                    scope["raw_path"] = raw[len(root):]
                else:
                    scope["raw_path"] = path.encode("utf-8")

        # Login: ?token=... → cookie + redirect to the same URL without the token.
        query = parse_qsl((scope.get("query_string") or b"").decode("latin-1"), keep_blank_values=True)
        supplied = [v for k, v in query if k == "token"]
        nonces = [v for k, v in query if k == "login"]
        if supplied or nonces:
            ok = method in ("GET", "HEAD") and (
                token_valid(s, supplied[-1]) if supplied else s.sessions.consume_login_nonce(nonces[-1]))
            if ok:
                rest = [(k, v) for k, v in query if k not in ("token", "login")]
                # Same-origin path only: "//host/..." would be a protocol-relative (off-site) URL.
                location = s.url(path.lstrip("/\\")) + ("?" + urlencode(rest) if rest else "")
                resp = RedirectResponse(location, status_code=303)
                resp.set_cookie(s.cookie_name, s.sessions.new_session(), httponly=True, samesite="strict",
                                path=s.cookie_path)
                return resp
            state["authenticated"] = False
            if supplied:
                msg = ("This link's token is not valid for the running malvalid server (each malvalid serve "
                       "launch prints a new one). Open the link printed in the terminal where malvalid serve "
                       "is running.")
            else:
                msg = ("This one-time sign-in link has expired or was already used. Open the link printed in "
                       "the terminal where malvalid serve is running.")
            return self.error_response(request, 401, "Invalid access link", msg)

        authed = is_authenticated(s, request)
        state["authenticated"] = authed
        exempt = path in EXEMPT_PATHS or path.startswith(EXEMPT_PREFIXES)
        if not authed and not exempt:
            where = (f"it ends with {s.root_path}/?token=" if s.root_path else
                     f"it starts with {s.public_base_url}/?token= or a private *.localhost address")
            return self.error_response(
                request, 401, "Open the link from your terminal",
                "This malvalid UI is private to whoever started it. Open the link that malvalid serve printed "
                f"in your terminal ({where}); it signs this browser in for as long as the server runs.",
            )

        if method not in SAFE_METHODS:
            problem = self._cross_origin_problem(request)
            if problem:
                log.warning("refused %s %s: %s", method, path, problem)
                return self.error_response(request, 403, "Request refused", problem)
            header = request.headers.get("x-csrf-token")
            if header is not None:
                if not csrf_valid(s, header):
                    return self.error_response(request, 403, "Request refused",
                                               "Invalid CSRF token; reload the page and try again.")
                state["csrf_verified"] = True
            else:
                ctype = (request.headers.get("content-type") or "").lower()
                if not ctype.startswith(FORM_CONTENT_TYPES):
                    return self.error_response(request, 403, "Request refused",
                                               "Missing CSRF token (send the X-CSRF-Token header).")
                state["csrf_verified"] = False  # the route checks the `csrf` form field
        return None

    def _cross_origin_problem(self, request: Request) -> str | None:
        h = request.headers
        site = (h.get("sec-fetch-site") or "").lower()
        if site in ("cross-site", "same-site"):
            return f"cross-origin request refused (Sec-Fetch-Site: {site})"
        origin = h.get("origin")
        if origin is not None and origin.strip().lower() != "null":
            if _netloc(origin.strip()) not in self.allowed:
                return "cross-origin request refused (Origin not allowed)"
            return None
        referer = h.get("referer")
        if referer:
            if _netloc(referer.strip()) not in self.allowed:
                return "cross-origin request refused (Referer not allowed)"
        return None


def csrf_error_response(request: Request, message: str) -> Response:
    factory: ErrorFactory = request.app.state.error_response
    return factory(request, 403, "Request refused", message)


__all__ = [
    "APP_CSP",
    "RAW_REPORT_MARKER",
    "REPORT_CSP",
    "CsrfError",
    "SecurityMiddleware",
    "apply_security_headers",
    "csrf_error_response",
    "csrf_valid",
    "is_authenticated",
    "require_csrf",
    "security_headers",
    "token_valid",
    "wants_json",
]
