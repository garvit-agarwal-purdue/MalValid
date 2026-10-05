"""malvalid's local web UI (``malvalid serve``; needs the ``malvalid[web]`` extra).

A local, single-user front end over the same engine as the CLI: submit an adapter + model, watch the
run progress module by module, read the production-readiness verdict and report, browse and compare
past runs. Every run is a ``malvalid run`` subprocess with the same sandbox; the web server itself
never imports an adapter, loads a model or unpickles anything. See ``docs/web.md``.

``create_app(settings)`` builds the Starlette app; :func:`malvalid.web.serve.serve` runs it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["WebSettings", "create_app", "missing_web_dependencies"]

_WEB_DEPENDENCIES = (("starlette", "starlette"), ("uvicorn", "uvicorn"), ("python_multipart", "python-multipart"))


def missing_web_dependencies() -> list[str]:
    """Distribution names of the ``malvalid[web]`` extra that cannot be imported."""
    import importlib.util

    missing = []
    for module, dist in _WEB_DEPENDENCIES:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found and module == "python_multipart":
            try:
                found = importlib.util.find_spec("multipart") is not None
            except (ImportError, ValueError):
                found = False
        if not found:
            missing.append(dist)
    return missing


def __getattr__(name: str) -> Any:  # lazy: importing malvalid.web must not require the extra
    if name == "WebSettings":
        from malvalid.web.settings import WebSettings

        return WebSettings
    if name == "create_app":
        from malvalid.web.app import create_app

        return create_app
    raise AttributeError(name)
