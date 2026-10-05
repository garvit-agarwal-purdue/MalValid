"""Capture the execution environment recorded in every report.

A production-readiness verdict is only auditable later if the report says exactly which library
versions produced it (spec §11). Versions come from installed distribution metadata, so nothing
heavy is imported just to read a version string.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import logging
import os
import platform
import sys
from typing import Any

log = logging.getLogger("malvalid.environment")

# report key -> candidate distribution names (first one found wins)
LIBRARIES: dict[str, tuple[str, ...]] = {
    "numpy": ("numpy",),
    "scipy": ("scipy",),
    "sklearn": ("scikit-learn",),
    "lightgbm": ("lightgbm",),
    "xgboost": ("xgboost",),
    "shap": ("shap",),
    "art": ("adversarial-robustness-toolbox",),
    "modelscan": ("modelscan",),
    "onnx": ("onnx",),
    "onnxruntime": ("onnxruntime", "onnxruntime-gpu"),
    "tesseract": ("tesseract", "tesseract-ml"),
    "lief": ("lief",),
    "pefile": ("pefile",),
    "jinja2": ("jinja2", "Jinja2"),
    "pydantic": ("pydantic",),
    "pyyaml": ("pyyaml", "PyYAML"),
    "typer": ("typer",),
    "rich": ("rich",),
}

# Environment variables that affect numerical results / threading (recorded verbatim).
_RECORDED_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MALVALID_THREADS")


def _dist_version(names: tuple[str, ...]) -> str | None:
    for n in names:
        try:
            return md.version(n)
        except md.PackageNotFoundError:
            continue
        except Exception as e:  # pragma: no cover - broken metadata
            log.debug("metadata lookup for %s failed: %s", n, e)
    return None


def _tesseract_version() -> str | None:
    v = _dist_version(LIBRARIES["tesseract"])
    if v is not None:
        return v
    try:
        if importlib.util.find_spec("tesseract") is not None:
            return "installed (version unknown)"
    except (ImportError, ValueError):
        pass
    try:
        if importlib.util.find_spec("malvalid._vendor.tesseract_temporal") is not None:
            return "vendored (malvalid._vendor.tesseract_temporal)"
    except (ImportError, ValueError):
        pass
    return None


def library_versions() -> dict[str, str | None]:
    """{report key: version or None} for every library that can influence a result."""
    out: dict[str, str | None] = {}
    for key, names in LIBRARIES.items():
        out[key] = _tesseract_version() if key == "tesseract" else _dist_version(names)
    return out


def capture_environment() -> dict[str, Any]:
    """The ``environment`` block of report.json."""
    from malvalid import __version__

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "byteorder": sys.byteorder,
        "malvalid": __version__,
        "libraries": library_versions(),
        "env": {k: os.environ[k] for k in _RECORDED_ENV if k in os.environ},
    }
