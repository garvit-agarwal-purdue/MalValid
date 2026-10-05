"""Helpers for writing a malvalid adapter (the small Python file that submits your detector).

A minimal adapter for a LightGBM model saved with ``booster.save_model("model.txt")``::

    from malvalid.adapter import BaseDetector

    class MyDetector(BaseDetector):
        feature_version = "ember_v2"          # must match a malvalid feature schema
        model_kind = "lightgbm"               # lightgbm | xgboost | sklearn_gbdt | onnx
        operating_threshold = 0.8336          # the score cut-off you would ship
        training_hashes_path = "train_sha256.txt"   # or None
        training_cutoff = "2018-10-31"        # newest training sample, or None
        model_path = "model.txt"              # scanned by M0 before anything is loaded

:class:`BaseDetector` provides ``load()`` (via the matching :mod:`malvalid.loaders` plugin),
``predict_proba()`` and ``predict()``. Override ``load``/``predict_proba`` for custom pipelines,
and add ``featurize(raw: bytes)`` if your model has its own raw-PE feature extractor.

Relative paths (``model_path``, ``training_hashes_path``) are resolved against the adapter file's
directory. ``load()`` runs inside malvalid's sandbox: no network, read-only file system, and pickles
are refused unless the operator passes ``--allow-pickle``.
"""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from malvalid.core import AdapterError

__all__ = ["BaseDetector", "adapter_dir", "load_model", "resolve_path"]


def _caller_dir(depth: int = 2) -> Path | None:
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is None:
                return None
            frame = frame.f_back
        fn = frame.f_globals.get("__file__") if frame is not None else None
        return Path(fn).resolve().parent if fn else None
    finally:
        del frame


def adapter_dir(obj: Any = None) -> Path:
    """Directory of the adapter file defining ``obj`` (a class or instance), else of the caller."""
    if obj is not None:
        cls = obj if isinstance(obj, type) else type(obj)
        mod = sys.modules.get(cls.__module__)
        fn = getattr(mod, "__file__", None)
        if fn:
            return Path(fn).resolve().parent
    d = _caller_dir()
    return d if d is not None else Path.cwd()


def resolve_path(path: str | os.PathLike[str], base: str | os.PathLike[str] | None = None) -> Path:
    """Resolve ``path`` against ``base`` (default: the calling adapter's directory) unless absolute."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    b = Path(base) if base is not None else (_caller_dir() or Path.cwd())
    return (b / p).resolve()


def load_model(path: str | os.PathLike[str], kind: str | None = None, *, allow_pickle: bool | None = None) -> Any:
    """Load a model artifact with the matching :mod:`malvalid.loaders` plugin.

    ``path`` may be relative to the calling adapter's directory. ``kind`` is a loader name
    (``lightgbm``, ``xgboost``, ``sklearn_gbdt``, ``onnx``); by default it is chosen by file extension.
    ``allow_pickle=None`` follows the sandbox policy (``--allow-pickle``); pickle-based artifacts are
    refused otherwise.
    """
    from malvalid.loaders.base import load_model as _load

    p = Path(path).expanduser()
    if not p.is_absolute():
        base = _caller_dir()
        cand = (base / p).resolve() if base is not None else p.resolve()
        p = cand if cand.exists() or not p.exists() else p.resolve()  # adapter dir first, then cwd
    if not p.exists():
        raise AdapterError(f"model artifact not found: {p}")
    return _load(p, kind, allow_pickle=allow_pickle)


class BaseDetector:
    """Mixin implementing the parts of the submission contract that rarely need customizing.

    Subclasses must set ``feature_version``, ``model_kind`` and ``operating_threshold``; they should
    set ``model_path`` (or ``model_paths``) so M0 can scan the artifact before it is loaded.
    ``training_hashes_path`` and ``training_cutoff`` default to ``None`` (unknown), which makes the
    membership-inference and drift tests skip.
    """

    feature_version: ClassVar[str]
    model_kind: ClassVar[str]
    operating_threshold: ClassVar[float]
    training_hashes_path: ClassVar[str | None] = None
    training_cutoff: ClassVar[str | None] = None

    #: The native model object (``lgb.Booster``, xgboost model, sklearn estimator, ONNX session).
    #: Setting it lets malvalid read tree structure for the tree-based tests.
    native_model: Any = None

    def __init__(self, native_model: Any = None):
        self.native_model = native_model
        self._loader: Any = None

    @classmethod
    def load(cls) -> "BaseDetector":
        """Default loader: ``model_path`` through the ``model_kind`` loader plugin."""
        mp = getattr(cls, "model_path", None)
        if mp is None:
            paths = getattr(cls, "model_paths", None) or ()
            mp = paths[0] if len(paths) == 1 else None
        if mp is None:
            raise AdapterError(
                f"{cls.__name__} does not declare model_path, so BaseDetector.load() cannot know what to load; "
                "declare model_path or override load()"
            )
        kind = getattr(cls, "model_kind", None)
        path = resolve_path(mp, adapter_dir(cls))
        from malvalid.loaders.base import load_model as _load

        try:
            native = _load(path, kind)
        except KeyError:
            native = _load(path, None)
        return cls(native)

    def _native_loader(self) -> Any:
        if getattr(self, "_loader", None) is None:
            from malvalid.loaders.base import find_loader_for_native

            if self.native_model is None:
                raise AdapterError(f"{type(self).__name__}: native_model is not set; override predict_proba()")
            self._loader = find_loader_for_native(self.native_model)
            if getattr(self, "_loader", None) is None:
                raise AdapterError(
                    f"{type(self).__name__}: no malvalid loader understands native_model of type "
                    f"{type(self.native_model).__name__}; override predict_proba()"
                )
        return self._loader

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n,) malicious-class score in [0, 1] via the loader plugin for ``native_model``."""
        return np.asarray(self._native_loader().predict_proba(self.native_model, X), dtype=np.float64).reshape(-1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(n,) int8 in {0, 1}: ``predict_proba(X) >= operating_threshold``."""
        p = np.asarray(self.predict_proba(X), dtype=np.float64).reshape(-1)
        return (p >= float(self.operating_threshold)).astype(np.int8)
