"""``malvalid serve``: bind the socket, print the private URL, open a browser, run uvicorn.

The CLI command (``malvalid.cli.serve_cmd``) validates the flags and calls :func:`bind_socket` then
:func:`run_server`. The socket is bound before the server starts so ``--port 0`` resolves to a real
port for the printed URL and the Host-header allowlist.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import webbrowser

from malvalid.web.settings import WebSettings, bind_address

log = logging.getLogger("malvalid.web.serve")


def has_display() -> bool:
    """Can a browser plausibly be opened on this machine?"""
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def bind_socket(host: str, port: int) -> socket.socket:
    """A listening TCP socket on ``host:port`` (``port`` 0 = any free port). Raises OSError."""
    addr = bind_address(host)
    infos = socket.getaddrinfo(addr, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    last: OSError | None = None
    for family, stype, proto, _canon, sockaddr in infos:
        sock = socket.socket(family, stype, proto)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: SO_REUSEADDR would let us share a busy port
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6 and addr == "::":
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            sock.bind(sockaddr)
            sock.listen(128)
            sock.set_inheritable(True)
            return sock
        except OSError as e:
            last = e
            sock.close()
    raise last or OSError(f"cannot bind {host}:{port}")


def bind_first_free(host: str, port: int, attempts: int = 100) -> socket.socket:
    """Bind ``port``, else the next free port above it (``attempts`` ports in all). Raises the first
    OSError when none is free (``--find-free-port``; the launchers use it to prefer 8765)."""
    if not 0 < port <= 65535:
        raise ValueError("bind_first_free needs a concrete port")
    first: OSError | None = None
    for p in range(port, min(65535, port + max(1, attempts) - 1) + 1):
        try:
            return bind_socket(host, p)
        except OSError as e:
            first = first or e
    assert first is not None
    raise first


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url, new=2)
    except Exception as e:  # pragma: no cover - platform dependent
        log.debug("could not open a browser: %s", e)


def _stop_on_hangup(server: object) -> None:
    """Closing the terminal window (SIGHUP) stops the server gracefully, like Ctrl+C, so queued and
    running runs are stopped too instead of being orphaned (uvicorn handles only SIGINT/SIGTERM).
    Mapped to SIGTERM: a terminal often sends SIGHUP twice, and a second SIGINT would make uvicorn
    force-exit without the lifespan shutdown that stops the runs."""
    import signal

    sighup = getattr(signal, "SIGHUP", None)
    if sighup is None or threading.current_thread() is not threading.main_thread():
        return
    try:
        if signal.getsignal(sighup) not in (signal.SIG_DFL, None):
            return  # someone (nohup, a supervisor) already decided what SIGHUP means
        signal.signal(sighup, lambda signum, frame: server.handle_exit(signal.SIGTERM, frame))  # type: ignore[attr-defined]
    except (ValueError, OSError):  # pragma: no cover - not the main interpreter thread
        pass


def run_server(settings: WebSettings, sock: socket.socket, *, open_browser: bool = False,
               log_level: str = "warning") -> None:
    """Serve the UI on ``sock`` until interrupted (Ctrl+C stops it; running jobs are terminated)."""
    import uvicorn

    from malvalid.web.app import create_app

    app = create_app(settings)
    config = uvicorn.Config(
        app,
        log_level=log_level,
        access_log=False,  # the first request carries the token in its query string
        proxy_headers=False,
        server_header=False,
        lifespan="on",
        timeout_graceful_shutdown=15,
    )
    server = uvicorn.Server(config)
    _stop_on_hangup(server)
    if open_browser:
        # A one-time, short-lived link: the browser's command line (visible to other local users in
        # `ps`) never carries the access token.
        t = threading.Timer(0.7, _open_browser, args=(settings.browser_login_url(),))
        t.daemon = True
        t.start()
    server.run(sockets=[sock])


__all__ = ["bind_first_free", "bind_socket", "has_display", "run_server"]
