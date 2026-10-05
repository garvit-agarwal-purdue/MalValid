"""Settings of one ``malvalid serve`` process (the local, single-user web UI).

The UI binds ``127.0.0.1`` by default and authenticates every request with a per-launch secret
(:attr:`WebSettings.token`). Everything security-relevant that depends on the launch parameters —
the Host-header allowlist, the CSRF token, the printed login URL — is derived here.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import socket
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_RUNS_DIR = "malvalid-runs"  # same default as `malvalid run`
DEFAULT_MAX_UPLOAD_MB = 4096
DEFAULT_VALIDATE_TIMEOUT_S = 600.0
DEFAULT_CANCEL_GRACE_S = 10.0

#: Prefix of the HttpOnly, SameSite=Strict session cookie. The full name carries the port
#: (``malvalid_session_<port>``) so two servers on one host never overwrite each other's cookie.
#: Its value is a random per-browser session id, never the launch token.
COOKIE_NAME = "malvalid_session"
#: Seconds a one-time ``/?login=<nonce>`` link (the one handed to the browser on launch) stays valid.
LOGIN_NONCE_TTL_S = 120.0
#: Sessions kept per server (the oldest is dropped beyond this).
MAX_SESSIONS = 256
#: Tokens go into a URL and a cookie: URL-safe characters only.
TOKEN_RE = re.compile(r"^[A-Za-z0-9._~-]{1,512}$")
#: Below this length a ``--token`` is accepted (scripts/tests) but flagged as weak.
MIN_STRONG_TOKEN_LEN = 16
#: ``--root-path``: the public path prefix under a reverse proxy, e.g. ``/node/<host>/<port>``
#: (one or more ``/segment`` parts of URL-safe characters; no ``.``/``..`` segments).
ROOT_PATH_RE = re.compile(r"^(/[A-Za-z0-9_~:@+-][A-Za-z0-9._~:@+-]*)+$")
#: ``--trusted-host``: a DNS name or IP literal, optionally with ``:port`` (no wildcards).
TRUSTED_HOST_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,62})(?:\.[a-z0-9](?:[a-z0-9-]{0,62}))*|\[[0-9a-f:.]+\])(?::\d{1,5})?$")


def normalize_root_path(value: str | None) -> str:
    """``--root-path`` as ``/a/b`` (no trailing slash); ``""`` = served at ``/``. Raises ValueError."""
    v = (value or "").strip()
    if v in ("", "/"):
        return ""
    v = v.rstrip("/")
    if not v.startswith("/"):
        v = "/" + v
    if not ROOT_PATH_RE.match(v) or any(seg in (".", "..") for seg in v.split("/")):
        raise ValueError(f"invalid root path {value!r}: use a URL path such as /node/<host>/<port>")
    return v


def normalize_client_network(value: str) -> str:
    """``--allow-client``: an IP address or CIDR network, normalized (``10.0.0.0/8``). Raises ValueError."""
    try:
        return str(ipaddress.ip_network(str(value).strip(), strict=False))
    except ValueError:
        raise ValueError(f"invalid client address {value!r}: give an IP address or a CIDR network") from None


def normalize_trusted_host(value: str) -> str:
    """``--trusted-host`` lower-cased; ``name`` or ``name:port``. Raises ValueError."""
    v = str(value).strip().lower()
    if v.startswith(("http://", "https://")) or "/" in v or "*" in v or not TRUSTED_HOST_RE.match(v):
        raise ValueError(f"invalid trusted host {value!r}: give a host name such as gateway.example.edu "
                         "(optionally with :port), without scheme, path or wildcards")
    return v


def _strip_brackets(host: str) -> str:
    h = host.strip()
    return h[1:-1] if h.startswith("[") and h.endswith("]") else h


def is_loopback_host(host: str) -> bool:
    """``127.0.0.1``, any other ``127/8`` address, ``::1`` and ``localhost``."""
    h = _strip_brackets(host).lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def is_wildcard_host(host: str) -> bool:
    return _strip_brackets(host) in ("", "0.0.0.0", "::")


def url_host(host: str) -> str:
    """``host`` as it appears in a URL / Host header (IPv6 literals in brackets)."""
    h = _strip_brackets(host)
    return f"[{h}]" if ":" in h else h


def bind_address(host: str) -> str:
    """The address to bind for ``host`` (``localhost`` binds the IPv4 loopback)."""
    h = _strip_brackets(host)
    return "127.0.0.1" if h.lower() == "localhost" else h


class Sessions:
    """Browser sessions and one-time login nonces of one server process (in memory only).

    A valid ``?token=`` (or one-time ``?login=``) link creates a session: a random id stored in the
    session cookie. The launch token itself never goes into a cookie, so a cookie that leaks (browsers
    send loopback cookies to every port of the same host name) does not reveal the token, and a
    restart of the server invalidates every session.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, float] = OrderedDict()
        self._nonces: dict[str, float] = {}

    def new_session(self) -> str:
        sid = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[sid] = time.time()
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
        return sid

    def valid(self, sid: str | None) -> bool:
        if not sid or not isinstance(sid, str) or len(sid) > 128:
            return False
        with self._lock:
            known = list(self._sessions)
        ok = False
        for k in known:  # constant-time comparison against every live session
            ok |= hmac.compare_digest(k.encode("utf-8", "replace"), sid.encode("utf-8", "replace"))
        return ok

    def new_login_nonce(self, ttl_s: float = LOGIN_NONCE_TTL_S) -> str:
        """A single-use login secret valid for ``ttl_s`` seconds (safe to put on a command line)."""
        nonce = secrets.token_urlsafe(24)
        now = time.monotonic()
        with self._lock:
            self._nonces = {k: v for k, v in self._nonces.items() if v > now}
            self._nonces[nonce] = now + float(ttl_s)
        return nonce

    def consume_login_nonce(self, nonce: str | None) -> bool:
        if not nonce or not isinstance(nonce, str):
            return False
        now = time.monotonic()
        with self._lock:
            match = None
            for k in list(self._nonces):
                if hmac.compare_digest(k.encode("utf-8", "replace"), nonce.encode("utf-8", "replace")):
                    match = k
            if match is None:
                return False
            expires = self._nonces.pop(match)
        return expires > now


class TokenFileError(ValueError):
    """``--token-file`` cannot be used (the message never contains the token)."""


def load_or_create_token_file(path: Path) -> tuple[str, bool, bool]:
    """``--token-file``: the access token kept in ``path`` so the login link survives restarts.

    If ``path`` exists, its first line is the token: it must be 16-512 URL-safe characters
    (``A-Z a-z 0-9 . _ ~ -``). Otherwise a fresh ``secrets.token_urlsafe(32)`` is written to a new file
    created with mode 0600 (``O_EXCL``: an existing file or symlink is never overwritten).
    Returns ``(token, created, loose_permissions)``; ``loose_permissions`` is True when an existing
    file is accessible by group or others. Raises :class:`TokenFileError`, whose message never
    includes the token.
    """
    import os
    import stat

    p = Path(path).expanduser()
    if p.exists() or p.is_symlink():
        try:
            st = p.stat()
            if not stat.S_ISREG(st.st_mode):
                raise TokenFileError(f"{path} is not a regular file")
            lines = p.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as e:
            raise TokenFileError(f"{path}: cannot read it ({type(e).__name__})") from None
        token = lines[0].strip() if lines else ""
        if not token:
            raise TokenFileError(f"{path}: the file is empty")
        if not TOKEN_RE.match(token) or len(token) < MIN_STRONG_TOKEN_LEN:
            raise TokenFileError(f"{path}: the first line must be a token of {MIN_STRONG_TOKEN_LEN}-512 URL-safe "
                                 "characters (A-Z a-z 0-9 . _ ~ -); delete the file to generate a new one")
        return token, False, bool(st.st_mode & 0o077)
    token = secrets.token_urlsafe(32)
    try:
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as e:
        raise TokenFileError(f"{path}: cannot create it ({e.strerror or type(e).__name__})") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token + "\n")
    except OSError as e:
        p.unlink(missing_ok=True)
        raise TokenFileError(f"{path}: cannot write it ({e.strerror or type(e).__name__})") from None
    return token, True, False


def random_loopback_name() -> str:
    """A per-launch ``mg-<16 hex>.localhost`` name (browsers resolve ``*.localhost`` to loopback)."""
    return f"mg-{secrets.token_hex(8)}.localhost"


@dataclass
class WebSettings:
    """Launch parameters of the web UI (``malvalid serve`` flags)."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    runs_dir: Path = Path(DEFAULT_RUNS_DIR)
    #: Default gate policy for new runs (``--config``); None = the packaged default policy.
    config_path: Path | None = None
    max_concurrent: int = 1
    max_upload_mb: int = DEFAULT_MAX_UPLOAD_MB
    allow_remote: bool = False
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    #: ``malvalid validate-adapter`` subprocess timeout (``POST /api/validate``).
    validate_timeout_s: float = DEFAULT_VALIDATE_TIMEOUT_S
    #: Seconds between SIGTERM and SIGKILL when a run is cancelled.
    cancel_grace_s: float = DEFAULT_CANCEL_GRACE_S
    #: Command that starts malvalid in a subprocess; empty = ``[sys.executable, "-m", "malvalid"]``.
    command_prefix: tuple[str, ...] = ()
    #: Extra environment variables for run/validation subprocesses (added to the inherited env).
    extra_env: dict[str, str] = field(default_factory=dict)
    #: Additional accepted Host header values (``name`` or ``name:port``), e.g. a DNS name.
    extra_hosts: tuple[str, ...] = ()
    #: ``--root-path``: public path prefix when the UI is reached through a reverse proxy (Open
    #: OnDemand ``/node/<host>/<port>`` or ``/rnode/<host>/<port>``). Every generated URL, redirect and
    #: the cookie path carry it; requests are accepted with or without it (prefix-keeping and
    #: prefix-stripping proxies). ``""`` = served at ``/``.
    root_path: str = ""
    #: ``--trusted-host``: the reverse proxy's public host name(s), added to the Host-header and
    #: Origin/Referer allowlist (``name`` matches ``name`` and ``name:<port>``; ``name:port`` is exact).
    #: The token login stays required; ``X-Forwarded-*`` headers are never used.
    trusted_hosts: tuple[str, ...] = ()
    #: ``--allow-client``: when non-empty, only TCP peers in these networks (plus loopback) are served,
    #: e.g. the reverse proxy's address when ``--allow-remote`` binds a cluster-internal interface.
    #: Checked on the socket peer address (``X-Forwarded-For`` is ignored).
    allowed_clients: tuple[str, ...] = ()
    #: Per-launch secret host name the printed link uses on loopback binds (``mg-<hex>.localhost``),
    #: so the session cookie is scoped to it and never sent to other services on ``127.0.0.1:*``.
    #: None = pick a random one (loopback binds only); "" = none (links use 127.0.0.1).
    loopback_name: str | None = None
    #: Working directory of run / validation subprocesses: where ``malvalid serve`` was started, so
    #: relative paths in a gate policy mean what they mean for ``malvalid run`` in that terminal.
    launch_dir: Path = field(default_factory=Path.cwd)
    #: ``--allow-path-mode``: accept "Use files on this machine" (path mode), which makes the server
    #: read model, adapter, training-hash and gate-policy files from its own file system by path.
    #: Off by default: anyone holding the link could otherwise point a run at any file the server can read.
    allow_path_mode: bool = False
    #: ``--allow-no-sandbox``: accept "Run without the sandbox" (``no_sandbox``), which loads the
    #: submitted model or adapter inside an unsandboxed process. Off by default. ``malvalid run
    #: --no-sandbox`` on the command line is not affected.
    allow_no_sandbox: bool = False
    #: ``--allow-reduced-isolation``: on a machine with no OS sandbox (Windows, macOS, Linux without user
    #: namespaces) runs and validations use a plain worker process (``isolation: process_only``, flagged in
    #: every report and in the UI) instead of failing. Passed to every ``malvalid run`` /
    #: ``validate-adapter`` subprocess; it never downgrades a machine where bwrap/unshare works.
    allow_reduced_isolation: bool = False
    #: Where the token came from, for the System page (``"generated"``, ``"token-file"``, ``"flag"``).
    token_source: str = "generated"
    #: Browser sessions and one-time login links (in memory; a restart signs everyone out).
    sessions: Sessions = field(default_factory=Sessions, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir).expanduser().resolve()
        if self.config_path is not None:
            self.config_path = Path(self.config_path).expanduser().resolve()
        self.port = int(self.port)
        self.allow_path_mode = bool(self.allow_path_mode)
        self.allow_no_sandbox = bool(self.allow_no_sandbox)
        self.allow_reduced_isolation = bool(self.allow_reduced_isolation)
        if not 0 <= self.port <= 65535:
            raise ValueError(f"port must be between 0 and 65535 (got {self.port})")
        if int(self.max_concurrent) < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = int(self.max_concurrent)
        if int(self.max_upload_mb) < 1:
            raise ValueError("max_upload_mb must be >= 1")
        self.max_upload_mb = int(self.max_upload_mb)
        if not isinstance(self.token, str) or not TOKEN_RE.match(self.token):
            raise ValueError("the access token must be 1-512 URL-safe characters (A-Z a-z 0-9 . _ ~ -)")
        if not is_loopback_host(self.host) and not self.allow_remote:
            raise ValueError(
                f"refusing to serve on non-loopback host {self.host!r} without allow_remote "
                "(--allow-remote): the UI has no TLS and runs submitted models"
            )
        self.command_prefix = tuple(str(c) for c in self.command_prefix)
        self.extra_hosts = tuple(str(h) for h in self.extra_hosts)
        self.root_path = normalize_root_path(self.root_path)
        self.trusted_hosts = tuple(dict.fromkeys(normalize_trusted_host(h) for h in self.trusted_hosts))
        self.allowed_clients = tuple(dict.fromkeys(normalize_client_network(c) for c in self.allowed_clients))
        self.extra_env = {str(k): str(v) for k, v in (self.extra_env or {}).items()}
        self.launch_dir = Path(self.launch_dir).expanduser().resolve()
        if self.loopback_name is None:
            loop = is_loopback_host(self.host) and not is_wildcard_host(self.host)
            self.loopback_name = random_loopback_name() if loop else ""
        self.loopback_name = str(self.loopback_name).strip().lower()
        if self.loopback_name and not re.fullmatch(r"[a-z0-9-]{1,63}\.localhost", self.loopback_name):
            raise ValueError("loopback_name must look like <label>.localhost")

    # ---- derived values ------------------------------------------------------------------------

    @property
    def csrf_token(self) -> str:
        """``hmac(token, "csrf")`` hex: required on every state-changing request."""
        return hmac.new(self.token.encode("utf-8"), b"csrf", hashlib.sha256).hexdigest()

    @property
    def weak_token(self) -> bool:
        return len(self.token) < MIN_STRONG_TOKEN_LEN

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def bind(self) -> str:
        """``host:port`` as shown in the UI footer."""
        return f"{url_host(self.host)}:{self.port}"

    @property
    def cookie_name(self) -> str:
        """``malvalid_session_<port>``: one cookie per server, so instances never clobber each other."""
        return f"{COOKIE_NAME}_{self.port}"

    def client_allowed(self, peer: str | None) -> bool:
        """May a connection from TCP peer address ``peer`` be served (``--allow-client``)?"""
        if not self.allowed_clients:
            return True
        try:
            addr = ipaddress.ip_address(_strip_brackets(str(peer or "")))
        except ValueError:
            return False
        if addr.is_loopback:
            return True
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        return any(addr in ipaddress.ip_network(n) for n in self.allowed_clients)

    @property
    def cookie_path(self) -> str:
        """``Path`` of the session cookie: ``/`` or ``<root_path>/`` (with the trailing slash, so a
        sibling ``/node/<host>/<port>0`` or any other app on the proxy's origin never receives it)."""
        return f"{self.root_path}/"

    def url(self, path: str) -> str:
        """An app-absolute path (``/runs/x``) as the browser must request it (with :attr:`root_path`)."""
        return f"{self.root_path}/{path.lstrip('/')}"

    def proxy_login_url(self) -> str | None:
        """The login link through the reverse proxy (``https://<first trusted host><root_path>/?token=``),
        when both ``--trusted-host`` and ``--root-path`` are set."""
        if not (self.trusted_hosts and self.root_path):
            return None
        return f"https://{self.trusted_hosts[0]}{self.root_path}/?token={self.token}"

    @property
    def public_base_url(self) -> str:
        """The plain address (``http://127.0.0.1:<port>``); safe to show to unauthenticated clients."""
        host = "127.0.0.1" if is_wildcard_host(self.host) else url_host(self.host)
        if host.lower() == "localhost":
            host = "127.0.0.1"
        return f"http://{host}:{self.port}"

    @property
    def base_url(self) -> str:
        """The address the printed link uses: the secret per-launch ``*.localhost`` name on loopback
        binds (never shown to unauthenticated clients), else :attr:`public_base_url`."""
        if self.loopback_name:
            return f"http://{self.loopback_name}:{self.port}"
        return self.public_base_url

    def login_url(self) -> str:
        """The URL printed at launch: opening it signs the browser in (a session cookie)."""
        return f"{self.base_url}{self.root_path}/?token={self.token}"

    def fallback_login_url(self) -> str | None:
        """The same link on ``127.0.0.1`` for browsers that cannot resolve ``*.localhost`` names."""
        if not self.loopback_name:
            return None
        return f"{self.public_base_url}{self.root_path}/?token={self.token}"

    def browser_login_url(self) -> str:
        """A one-time link for the browser launched by ``malvalid serve``: it carries a short-lived,
        single-use nonce instead of the token, so the browser's command line does not leak the token."""
        return f"{self.base_url}{self.root_path}/?login={self.sessions.new_login_nonce()}"

    def remote_urls(self) -> list[str]:
        """Login URLs for other machines (wildcard binds with ``--allow-remote``)."""
        if not (self.allow_remote and is_wildcard_host(self.host)):
            return []
        out = []
        for name in _machine_names():
            out.append(f"http://{url_host(name)}:{self.port}{self.root_path}/?token={self.token}")
        return out

    def malvalid_command(self) -> list[str]:
        return list(self.command_prefix) if self.command_prefix else [sys.executable, "-m", "malvalid"]

    def allowed_hosts(self) -> frozenset[str]:
        """Accepted ``Host`` header values (DNS-rebinding defence)."""
        p = self.port
        names = ["127.0.0.1", "localhost", "[::1]"]
        if self.loopback_name:
            names.append(self.loopback_name)
        if self.allow_remote:
            if is_wildcard_host(self.host):
                names += [url_host(n) for n in _machine_names()]
            else:
                names.append(url_host(self.host))
        out: set[str] = set()
        for n in names:
            out.add(f"{n.lower()}:{p}")
            if p == 80:
                out.add(n.lower())
        for h in self.extra_hosts:
            h = h.strip().lower()
            if not h:
                continue
            out.add(h if re.search(r":\d+$", h) else f"{h}:{p}")
        for h in self.trusted_hosts:
            out.add(h)
            if not re.search(r":\d+$", h):
                out.add(f"{h}:{p}")  # a proxy that forwards Host as <name>:<backend port>
        return frozenset(out)

    def public_dict(self) -> dict[str, Any]:
        """Settings shown on the System page (never the token)."""
        return {
            "host": self.host,
            "port": self.port,
            "bind": self.bind,
            "runs_dir": str(self.runs_dir),
            "launch_dir": str(self.launch_dir),
            "config_path": str(self.config_path) if self.config_path else None,
            "max_concurrent": self.max_concurrent,
            "max_upload_mb": self.max_upload_mb,
            "allow_remote": self.allow_remote,
            "validate_timeout_s": self.validate_timeout_s,
            "cancel_grace_s": self.cancel_grace_s,
            "malvalid_command": " ".join(self.malvalid_command()),
            "allowed_hosts": sorted(self.allowed_hosts()),
            "root_path": self.root_path or "/",
            "trusted_hosts": list(self.trusted_hosts),
            "allowed_clients": list(self.allowed_clients) or "any (token required)",
            "token_source": ("weak (--token shorter than 16 characters)" if self.weak_token else
                             {"token-file": "--token-file (kept across restarts)",
                              "flag": "--token / MALVALID_SERVE_TOKEN"}.get(self.token_source, "per-launch secret")),
            "allow_path_mode": self.allow_path_mode,
            "allow_no_sandbox": self.allow_no_sandbox,
            "allow_reduced_isolation": self.allow_reduced_isolation,
        }


def _machine_names() -> list[str]:
    """This machine's host name, FQDN and their addresses (for wildcard binds)."""
    names: list[str] = []
    for fn in (socket.gethostname, socket.getfqdn):
        try:
            n = fn()
        except OSError:
            continue
        if n and n not in names:
            names.append(n)
    for n in list(names):
        try:
            for info in socket.getaddrinfo(n, None):
                addr = str(info[4][0])
                if addr and addr not in names:
                    names.append(addr)
        except OSError:
            pass
    return names


__all__ = [
    "COOKIE_NAME",
    "DEFAULT_HOST",
    "DEFAULT_MAX_UPLOAD_MB",
    "DEFAULT_PORT",
    "DEFAULT_RUNS_DIR",
    "Sessions",
    "TokenFileError",
    "WebSettings",
    "bind_address",
    "is_loopback_host",
    "is_wildcard_host",
    "load_or_create_token_file",
    "normalize_client_network",
    "normalize_root_path",
    "normalize_trusted_host",
    "url_host",
]
