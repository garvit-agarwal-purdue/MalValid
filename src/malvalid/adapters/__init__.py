"""Adapter tooling: :func:`malvalid.adapters.validate.validate_adapter` (``malvalid validate-adapter``).

The helpers adapters import live in :mod:`malvalid.adapter`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ValidationReport", "validate_adapter"]


def __getattr__(name: str) -> Any:  # lazy: importing malvalid.adapters must stay cheap
    if name in __all__:
        from malvalid.adapters import validate

        return getattr(validate, name)
    raise AttributeError(name)
