"""The sandboxed model worker: imports the submitted adapter, loads the model and serves requests.

This module runs in two places:

* inside the isolated worker process (``python -I -B -c <bootstrap> --in-fd N --out-fd M ...``),
  started by :mod:`malvalid.sandbox.host`. Requests and responses travel over two dedicated pipe fds
  using :mod:`malvalid.sandbox.protocol` (JSON headers + ``.npy`` payloads — never pickles). The
  adapter's own stdout/stderr go to ``<scratch>/worker.log``. With ``--stdio`` (Windows, where pipe
  fds cannot be passed to a child) the channel is the process's original stdin/stdout instead, and
  fds 0/1 are re-pointed at ``os.devnull``/the log before any submitted code runs;
* in-process, when the operator disables the sandbox for debugging (``runtime.sandbox: false``), via
  :class:`AdapterRuntime` directly.

Start-up order inside the worker (security relevant): resource limits -> read the init message ->
pre-import model libraries -> install the pickle guards -> import the adapter.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import functools
import importlib
import importlib.util
import inspect
import io
import keyword
import logging
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from malvalid.core import (
    AdapterContractError,
    AdapterError,
    FeaturizeUnavailable,
    PickleRefused,
    UnsupportedTreeError,
)

log = logging.getLogger("malvalid.sandbox.worker")

#: Libraries pre-imported (before the pickle guards) for a declared ``model_kind``.
PREIMPORTS: dict[str, tuple[str, ...]] = {
    "lightgbm": ("lightgbm",),
    "xgboost": ("xgboost",),
    "sklearn": ("sklearn", "joblib"),
    "sklearn_gbdt": ("sklearn", "joblib"),
    "onnx": ("onnxruntime",),
}

REQUIRED_DECLARATIONS = (
    "feature_version",
    "model_kind",
    "operating_threshold",
    "training_hashes_path",
    "training_cutoff",
)
OPTIONAL_DECLARATIONS = ("model_path", "model_paths")

PICKLE_REFUSED_HINT = (
    "Pickles can execute arbitrary code when loaded, so malvalid refuses them by default. Re-export the "
    "model in a non-pickle format (LightGBM booster.save_model('model.txt'), XGBoost "
    "save_model('model.json'/'model.ubj'), scikit-learn -> ONNX via skl2onnx), or re-run with "
    "--allow-pickle if you trust the file and its origin."
)


# --------------------------------------------------------------------------------------------------
# Pickle guards
# --------------------------------------------------------------------------------------------------


def install_pickle_guards(allow_pickle: bool) -> list[str]:
    """Patch the unpickling entry points of this process. Returns the list of patched names.

    ``allow_pickle=False``: ``pickle.load/loads``, ``pickle.Unpickler.load`` (C and pure-Python),
    ``joblib.load``, ``numpy.load(allow_pickle=True)`` and ``dill``/``cloudpickle`` loads raise
    :class:`~malvalid.core.PickleRefused`.

    ``allow_pickle=True``: unpickling works, but every global a pickle imports is checked against
    malvalid's pickle policy (:func:`malvalid.modules.file_safety.classify_pickle_global`); CRITICAL
    and HIGH globals (``os.system``, ``builtins.eval``, ``subprocess``, sockets, ...) are refused
    before they are imported.

    Install this *after* importing model libraries and *before* importing the adapter.
    """
    import pickle

    import _pickle

    from malvalid.modules.file_safety import classify_pickle_global

    patched: list[str] = []

    def refuse(what: str) -> None:
        raise PickleRefused(f"{what} was called inside the sandbox but --allow-pickle was not given. {PICKLE_REFUSED_HINT}")

    def check_global(module: str, name: str) -> None:
        sev, reason = classify_pickle_global(module, name)
        if sev in ("CRITICAL", "HIGH"):
            raise PickleRefused(
                f"refused to unpickle global {module}.{name}: {reason}. malvalid's runtime pickle policy "
                "blocks dangerous imports even with --allow-pickle."
            )

    c_unpickler = _pickle.Unpickler
    py_unpickler = pickle._Unpickler  # type: ignore[attr-defined]

    class GuardedUnpickler(c_unpickler):  # type: ignore[misc, valid-type]
        """C unpickler with malvalid's pickle policy (subclassable like pickle.Unpickler)."""

        def find_class(self, module: str, name: str) -> Any:
            check_global(module, name)
            return super().find_class(module, name)

        def load(self) -> Any:
            if not allow_pickle:
                refuse(f"{type(self).__name__}.load")
            return super().load()

    GuardedUnpickler.__module__ = "pickle"
    GuardedUnpickler.__qualname__ = GuardedUnpickler.__name__ = "Unpickler"

    def g_load(file: Any, *, fix_imports: bool = True, encoding: str = "ASCII", errors: str = "strict",
               buffers: Any = None) -> Any:
        if not allow_pickle:
            refuse("pickle.load")
        return GuardedUnpickler(file, fix_imports=fix_imports, encoding=encoding, errors=errors,
                                buffers=buffers).load()

    def g_loads(data: Any, /, *, fix_imports: bool = True, encoding: str = "ASCII", errors: str = "strict",
                buffers: Any = None) -> Any:
        if not allow_pickle:
            refuse("pickle.loads")
        if isinstance(data, str):
            raise TypeError("Can't load pickle from unicode string")
        return GuardedUnpickler(io.BytesIO(data), fix_imports=fix_imports, encoding=encoding, errors=errors,
                                buffers=buffers).load()

    for mod in (pickle, _pickle):
        mod.load = g_load  # type: ignore[assignment]
        mod.loads = g_loads  # type: ignore[assignment]
        mod.Unpickler = GuardedUnpickler  # type: ignore[assignment, misc]
    pickle._load = g_load  # type: ignore[attr-defined]
    pickle._loads = g_loads  # type: ignore[attr-defined]
    patched += ["pickle.load", "pickle.loads", "pickle.Unpickler", "_pickle.load", "_pickle.loads", "_pickle.Unpickler"]

    orig_py_load = py_unpickler.load
    orig_py_find = py_unpickler.find_class

    def py_load(self: Any) -> Any:
        if not allow_pickle:
            refuse(f"{type(self).__module__}.{type(self).__name__}.load")
        return orig_py_load(self)

    def py_find(self: Any, module: str, name: str) -> Any:
        check_global(module, name)
        return orig_py_find(self, module, name)

    py_unpickler.load = py_load
    py_unpickler.find_class = py_find
    patched += ["pickle._Unpickler.load", "pickle._Unpickler.find_class"]

    # numpy.load(allow_pickle=True): numpy's object-array path calls pickle.load (guarded above);
    # refuse early with a clear message when pickles are not allowed.
    np_load = np.load

    @functools.wraps(np_load)
    def g_np_load(file: Any, *args: Any, **kwargs: Any) -> Any:
        ap = kwargs.get("allow_pickle", args[1] if len(args) > 1 else False)
        if ap and not allow_pickle:
            refuse("numpy.load(allow_pickle=True)")
        return np_load(file, *args, **kwargs)

    np.load = g_np_load  # type: ignore[assignment]
    patched.append("numpy.load")

    # Libraries already imported that bound the originals at import time.
    if "joblib" in sys.modules and not allow_pickle:
        import joblib

        def g_joblib_load(*a: Any, **k: Any) -> Any:
            refuse("joblib.load")

        joblib.load = g_joblib_load
        with contextlib.suppress(Exception):
            import joblib.numpy_pickle as jnp

            jnp.load = g_joblib_load
        patched.append("joblib.load")
    for name in ("dill", "cloudpickle"):
        mod = sys.modules.get(name)
        if mod is None:
            continue
        if allow_pickle:
            if name == "cloudpickle":
                mod.load, mod.loads = g_load, g_loads  # type: ignore[attr-defined]
        else:
            def g_refuse(*a: Any, _n: str = name, **k: Any) -> Any:
                refuse(f"{_n}.load")

            mod.load = g_refuse  # type: ignore[attr-defined]
            mod.loads = g_refuse  # type: ignore[attr-defined]
        patched.append(f"{name}.load/loads")
    return patched


# --------------------------------------------------------------------------------------------------
# Adapter runtime (shared by the worker process and the in-process debug mode)
# --------------------------------------------------------------------------------------------------


def _encode_declared(v: Any) -> dict[str, Any]:
    """JSON-safe description of one declared class attribute (the host validates it)."""
    t = type(v).__name__
    if v is None or isinstance(v, (str, bool)):
        return {"present": True, "type": t, "value": v}
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        fv = float(v)
        return {"present": True, "type": t, "value": fv if math.isfinite(fv) else str(fv)}
    if isinstance(v, np.generic) and v.dtype.kind in "iuf":
        fv = float(v)
        return {"present": True, "type": t, "value": fv if math.isfinite(fv) else str(fv)}
    if isinstance(v, os.PathLike):
        return {"present": True, "type": "path", "value": os.fspath(v)}
    if isinstance(v, (_dt.date, _dt.datetime)):
        return {"present": True, "type": "date", "value": v.isoformat()}
    if isinstance(v, (list, tuple)) and all(isinstance(x, (str, os.PathLike)) for x in v):
        return {"present": True, "type": "list", "value": [os.fspath(x) for x in v]}
    return {"present": True, "type": t, "value": None, "repr": repr(v)[:200]}


def _module_name_for(path: Path, unique: bool) -> str:
    stem = path.stem
    ok = stem.isidentifier() and not keyword.iskeyword(stem) and stem not in sys.modules
    if ok and not unique:
        with contextlib.suppress(Exception):
            ok = importlib.util.find_spec(stem) is None  # never shadow an installed module
    if ok and not unique:
        return stem
    import hashlib

    return "malvalid_adapter_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]


class AdapterRuntime:
    """Imports an adapter file, finds the detector class, loads it and calls it.

    Output conversion happens here (to plain numeric numpy arrays); contract *validation* happens on
    the host side, which never trusts the worker.
    """

    def __init__(self, adapter_path: str | Path, *, class_name: str | None = None, unique_module_name: bool = False):
        self.adapter_path = Path(adapter_path).resolve()
        self.class_name = class_name or None
        self.unique_module_name = unique_module_name
        self.module: ModuleType | None = None
        self.cls: type | None = None
        self.inst: Any = None
        self._schema: Any = None
        self._featurize_source: str | None | bool = False  # False = not computed

    # ---- import & discovery --------------------------------------------------------------------

    def import_module(self) -> ModuleType:
        if self.module is not None:
            return self.module
        p = self.adapter_path
        if not p.is_file():
            raise AdapterError(f"adapter file not found: {p}")
        if p.suffix.lower() == ".json":
            # A model-file-only submission: a data-only spec configures malvalid's packaged
            # SpecDetector. Nothing from the submission folder is imported or executed, and the
            # folder is not put on sys.path.
            import hashlib

            from malvalid.adapters.spec import build_detector_module

            name = "malvalid_spec_" + hashlib.sha256(str(p).encode()).hexdigest()[:12]
            try:
                mod = build_detector_module(p, name)
            except AdapterError:
                raise
            except Exception as e:  # noqa: BLE001
                raise AdapterError(f"the model spec {p.name} could not be used: {type(e).__name__}: {e}") from e
            sys.modules[name] = mod
            self.module = mod
            return mod
        adir = str(p.parent)
        if adir not in sys.path:
            sys.path.insert(0, adir)
        name = _module_name_for(p, self.unique_module_name)
        spec = importlib.util.spec_from_file_location(name, str(p))
        if spec is None or spec.loader is None:
            raise AdapterError(f"cannot import {p} as a Python module")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except PickleRefused:
            sys.modules.pop(name, None)
            raise
        except BaseException as e:  # noqa: BLE001 - report any import-time failure to the operator
            sys.modules.pop(name, None)
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)[-6:])
            raise AdapterError(f"importing the adapter {p.name} failed: {type(e).__name__}: {e}\n{tb}") from e
        self.module = mod
        return mod

    def candidates(self) -> tuple[list[type], list[str]]:
        """Classes defined in the adapter module that look like detectors (+ near misses)."""
        mod = self.import_module()
        good: list[type] = []
        near: list[str] = []
        for _, obj in sorted(vars(mod).items()):
            if not isinstance(obj, type) or getattr(obj, "__module__", None) != mod.__name__:
                continue
            need = ("predict_proba", "load", "feature_version")
            have = [a for a in need if hasattr(obj, a)]
            if len(have) == len(need):
                good.append(obj)
            elif "predict_proba" in have or "load" in have:
                near.append(f"{obj.__name__} (missing {', '.join(a for a in need if a not in have)})")
        return good, near

    def find_class(self) -> type:
        if self.cls is not None:
            return self.cls
        mod = self.import_module()
        if self.class_name:
            obj = getattr(mod, self.class_name, None)
            if not isinstance(obj, type):
                good, _ = self.candidates()
                raise AdapterError(
                    f"class {self.class_name!r} not found in {self.adapter_path.name}"
                    + (f"; detector classes there: {', '.join(c.__name__ for c in good)}" if good else "")
                )
            self.cls = obj
            return obj
        good, near = self.candidates()
        if len(good) == 1:
            self.cls = good[0]
            return good[0]
        if not good:
            hint = f" Near misses: {'; '.join(near)}." if near else ""
            raise AdapterError(
                f"no detector class found in {self.adapter_path.name}: expected one class defining "
                f"predict_proba, a load() classmethod and feature_version (plus model_kind, "
                f"operating_threshold, training_hashes_path, training_cutoff).{hint}"
            )
        raise AdapterError(
            f"{self.adapter_path.name} defines several detector classes ({', '.join(c.__name__ for c in good)}); "
            "choose one with --class NAME"
        )

    def inspect(self) -> dict[str, Any]:
        """Declarations of the detector class, read without calling ``load()``."""
        cls = self.find_class()
        decl = {}
        for attr in REQUIRED_DECLARATIONS + OPTIONAL_DECLARATIONS:
            if hasattr(cls, attr):
                try:
                    decl[attr] = _encode_declared(inspect.getattr_static(cls, attr))
                    if isinstance(inspect.getattr_static(cls, attr), (property, staticmethod, classmethod)):
                        decl[attr] = {"present": True, "type": "property", "value": None,
                                      "repr": f"{attr} is a property/method; declare it as a plain class attribute"}
                except Exception as e:  # noqa: BLE001
                    decl[attr] = {"present": True, "type": "error", "value": None, "repr": f"{type(e).__name__}: {e}"}
            else:
                decl[attr] = {"present": False}
        load_attr = inspect.getattr_static(cls, "load", None)
        return {
            "class_name": cls.__name__,
            "module_name": getattr(self.module, "__name__", ""),
            "declarations": decl,
            "load_is_classmethod": isinstance(load_attr, (classmethod, staticmethod)),
            "has_featurize": callable(getattr(cls, "featurize", None)),
            "has_predict": callable(getattr(cls, "predict", None)),
            "has_tree_ensemble": callable(getattr(cls, "tree_ensemble", None)),
        }

    # ---- load ----------------------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        cls = self.find_class()
        load_attr = inspect.getattr_static(cls, "load", None)
        if not isinstance(load_attr, (classmethod, staticmethod)):
            raise AdapterError(f"{cls.__name__}.load must be a @classmethod that returns an instance of {cls.__name__}")
        t0 = time.monotonic()
        inst = cls.load()
        if inst is None:
            raise AdapterError(f"{cls.__name__}.load() returned None; it must return the loaded detector instance")
        for meth in ("predict_proba", "predict"):
            if not callable(getattr(inst, meth, None)):
                raise AdapterError(f"the object returned by {cls.__name__}.load() has no callable {meth}()")
        self.inst = inst
        self._featurize_source = False
        native = getattr(inst, "native_model", None)
        return {
            "class_name": cls.__name__,
            "load_s": round(time.monotonic() - t0, 3),
            "instance_type": type(inst).__name__,
            "native_model_type": None if native is None else f"{type(native).__module__}.{type(native).__name__}",
            "featurize_source": self.featurize_source(),
            "has_tree_ensemble": callable(getattr(inst, "tree_ensemble", None)),
        }

    def _require_loaded(self) -> Any:
        if self.inst is None:
            raise AdapterError("the model is not loaded (load() has not been called)")
        return self.inst

    # ---- scoring -------------------------------------------------------------------------------

    @staticmethod
    def _numeric(out: Any, method: str, n: int) -> np.ndarray:
        try:
            arr = np.asarray(out)
        except Exception as e:  # noqa: BLE001
            raise AdapterContractError(f"{method} returned {type(out).__name__}, which is not array-like ({e})") from e
        if arr.dtype.kind not in "biuf":
            try:
                arr = arr.astype(np.float64)
            except Exception as e:  # noqa: BLE001
                raise AdapterContractError(
                    f"{method} returned a non-numeric array (dtype={arr.dtype}); expected ({n},) floats"
                ) from e
        if arr.size > 4 * n + 16:
            raise AdapterContractError(f"{method} returned shape {arr.shape}; expected ({n},)")
        return np.ascontiguousarray(arr)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        inst = self._require_loaded()
        return self._numeric(inst.predict_proba(X), "predict_proba", X.shape[0]).astype(np.float64, copy=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        inst = self._require_loaded()
        arr = self._numeric(inst.predict(X), "predict", X.shape[0])
        return arr if arr.dtype.kind in "biu" else arr.astype(np.float64, copy=False)

    # ---- featurize -----------------------------------------------------------------------------

    def _schema_obj(self) -> Any:
        if self._schema is None:
            from malvalid import registry

            fv = getattr(self.find_class(), "feature_version", None)
            self._schema = registry.get_schema(str(fv))
        return self._schema

    def featurize_source(self) -> str | None:
        if self._featurize_source is not False:
            return self._featurize_source  # type: ignore[return-value]
        src: str | None = None
        inst = self.inst if self.inst is not None else self.find_class()
        if callable(getattr(inst, "featurize", None)):
            src = "adapter"
        else:
            try:
                if self._schema_obj().featurize_available():
                    src = "schema"
            except Exception as e:  # noqa: BLE001 - schema unknown or extractor deps missing
                log.info("schema featurizer unavailable: %s", e)
        self._featurize_source = src
        return src

    def featurize(self, raw: bytes) -> np.ndarray:
        src = self.featurize_source()
        if src == "adapter":
            out = self._require_loaded().featurize(raw)
        elif src == "schema":
            out = self._schema_obj().featurize(raw)
        else:
            raise FeaturizeUnavailable(
                "no raw-bytes featurizer: the adapter has no featurize() and the schema's extractor is unavailable"
            )
        arr = self._numeric(out, "featurize", 1 << 20)
        return arr.reshape(-1).astype(np.float32, copy=False)

    # ---- trees ---------------------------------------------------------------------------------

    def tree_ensemble(self) -> tuple[Any, str]:
        """``(TreeEnsemble | None, how/why)``."""
        from malvalid.loaders.trees import TreeEnsemble

        inst = self._require_loaded()
        fn = getattr(inst, "tree_ensemble", None)
        try:
            if callable(fn):
                te = fn()
                if te is None:
                    return None, "the adapter's tree_ensemble() returned None"
                if not isinstance(te, TreeEnsemble):
                    raise AdapterContractError(
                        f"tree_ensemble() must return malvalid.loaders.trees.TreeEnsemble, got {type(te).__name__}"
                    )
                return te, "adapter.tree_ensemble()"
            native = getattr(inst, "native_model", None)
            if native is None:
                return None, "the adapter exposes neither tree_ensemble() nor native_model"
            loader = None
            try:
                from malvalid.loaders.base import find_loader_for_native

                loader = find_loader_for_native(native)
            except Exception as e:  # noqa: BLE001 - loader plugins are optional
                log.info("find_loader_for_native failed: %s", e)
            if loader is not None:
                return loader.tree_ensemble(native), f"native_model via the {loader.kind} loader"
            booster = getattr(native, "booster_", native)
            if type(booster).__module__.startswith("lightgbm") and callable(getattr(booster, "dump_model", None)):
                from malvalid.loaders.trees import from_lightgbm_dump

                return from_lightgbm_dump(booster.dump_model()), "native_model (LightGBM dump_model)"
            return None, f"no model loader understands native_model of type {type(native).__module__}.{type(native).__name__}"
        except UnsupportedTreeError as e:
            return None, f"trees cannot be normalized: {e}"


# --------------------------------------------------------------------------------------------------
# Worker process
# --------------------------------------------------------------------------------------------------


def apply_rlimits(memory_mb: int | None, nofile: int | None) -> dict[str, Any]:
    """Lower (soft and hard) resource limits of this process. Returns what is in force.

    Where the ``resource`` module does not exist (Windows) nothing is set here and ``{"supported":
    False}`` is returned; the host applies a Job Object memory limit to the worker instead.
    """
    try:
        import resource
    except ImportError:
        return {"as": None, "core": None, "nofile": None, "supported": False}

    out: dict[str, Any] = {}

    def _set(res: int, value: int) -> None:
        soft, hard = resource.getrlimit(res)
        v = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(res, (v, v))

    if memory_mb:
        with contextlib.suppress(ValueError, OSError):
            _set(resource.RLIMIT_AS, int(memory_mb) << 20)
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if nofile:
        with contextlib.suppress(ValueError, OSError):
            _set(resource.RLIMIT_NOFILE, int(nofile))
    for name, res in (("as", resource.RLIMIT_AS), ("core", resource.RLIMIT_CORE), ("nofile", resource.RLIMIT_NOFILE)):
        soft, _ = resource.getrlimit(res)
        out[name] = None if soft == resource.RLIM_INFINITY else int(soft)
    return out


def _take_stdio_channel() -> tuple[int, int]:
    """``--stdio``: keep private duplicates of fds 0/1 as the channel, then point fd 0 at
    ``os.devnull`` and fd 1 at the log (fd 2), so nothing the adapter prints can corrupt the protocol."""
    in_fd, out_fd = os.dup(0), os.dup(1)
    if os.name == "nt":  # pragma: no cover - Windows only
        import msvcrt

        for fd in (in_fd, out_fd):
            msvcrt.setmode(fd, os.O_BINARY)
    null = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(null, 0)
    finally:
        os.close(null)
    os.dup2(2, 1)
    return in_fd, out_fd


def _set_parent_death_signal() -> None:
    """Die with the parent (host / bwrap / unshare) even if the channel is not being read."""
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL(None, use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0)
    except Exception:  # noqa: BLE001 - best effort, Linux only
        pass


def home_hidden(home: str) -> bool:
    """Whether nothing of ``home`` is visible except bind mounts and the empty directories leading to them.

    Under bubblewrap ``$HOME`` is an empty tmpfs; the host re-exposes read-only only what the worker needs
    (the Python environment, the adapter and model) when those live under ``$HOME``. Each of those is a
    mount point (a different device from the tmpfs), so it does not count as the home directory being
    visible. Without a file-system sandbox the real home shows ordinary files and directories: False.
    """

    def only_mounts(d: str, depth: int) -> bool:
        if depth > 64:
            return False
        try:
            names = os.listdir(d)
        except OSError:  # not readable: nothing visible
            return True
        for name in names:
            q = os.path.join(d, name)
            if os.path.islink(q) or not (os.path.ismount(q) or (os.path.isdir(q) and only_mounts(q, depth + 1))):
                return False
        return True

    if not os.path.isdir(home):
        return True
    return only_mounts(home, 0)


def self_check(host_home: str | None = None) -> dict[str, Any]:
    """What the worker can observe about its own isolation (reported in ``sandbox_info``)."""
    getuid = getattr(os, "getuid", None)
    info: dict[str, Any] = {"pid": os.getpid(), "uid": getuid() if getuid else None, "python": sys.version.split()[0],
                            "platform": sys.platform}
    try:
        info["interfaces"] = sorted({n for _, n in socket.if_nameindex()})
    except OSError as e:
        info["interfaces"] = None
        info["interfaces_error"] = str(e)
    info["network_isolated"] = info["interfaces"] is not None and set(info["interfaces"]) <= {"lo"}
    try:
        info["readonly_root"] = bool(os.statvfs("/").f_flag & os.ST_RDONLY)
    except (OSError, AttributeError):  # no statvfs on Windows
        info["readonly_root"] = None
    if host_home:
        info["home_hidden"] = home_hidden(host_home)
    info["env_keys"] = sorted(os.environ)
    info["threads_env"] = os.environ.get("OMP_NUM_THREADS")
    return info


def _error_header(rid: Any, e: BaseException) -> dict[str, Any]:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    return {
        "ok": False,
        "id": rid,
        "error_type": type(e).__name__,
        "error_module": type(e).__module__,
        "message": str(e)[:8000],
        "traceback": tb[-12000:],
    }


def _serve(in_fd: int, out_fd: int, init: dict[str, Any], limits: dict[str, Any]) -> int:
    from malvalid.sandbox.protocol import encode_array, read_message, write_message

    t0 = time.monotonic()
    # The operator's PYTHONPATH (dropped by ``python -I``); appended, so it never shadows the stdlib.
    for entry in init.get("sys_path") or ():
        if isinstance(entry, str) and entry and entry not in sys.path:
            sys.path.append(entry)
    preimported: list[str] = []
    for name in init.get("preimport") or ():
        try:
            importlib.import_module(name)
            preimported.append(name)
        except Exception as e:  # noqa: BLE001 - the adapter will report a real failure itself
            log.info("pre-import of %s failed: %s", name, e)
    allow_pickle = bool(init.get("allow_pickle"))
    guards = install_pickle_guards(allow_pickle)
    adapter = Path(init["adapter_path"])
    with contextlib.suppress(OSError):
        os.chdir(adapter.parent)
    runtime = AdapterRuntime(adapter, class_name=init.get("class_name"))
    hello = {
        "ok": True,
        "op": "hello",
        "startup_s": round(time.monotonic() - t0, 3),
        "self_check": self_check(init.get("host_home")),
        "rlimits": limits,
        "pickle_guards": guards,
        "allow_pickle": allow_pickle,
        "preimported": preimported,
    }
    write_message(out_fd, hello)
    log.info("worker ready (allow_pickle=%s, preimported=%s)", allow_pickle, preimported)

    while True:
        try:
            msg = read_message(in_fd)
        except EOFError:
            log.info("host closed the channel; exiting")
            return 0
        hdr = msg.header
        op = hdr.get("op")
        rid = hdr.get("id")
        try:
            if op == "inspect":
                write_message(out_fd, {"ok": True, "id": rid, "inspect": runtime.inspect()})
            elif op == "load":
                write_message(out_fd, {"ok": True, "id": rid, "load": runtime.load()})
            elif op in ("predict_proba", "predict"):
                X = msg.array("X")
                fn: Callable[[np.ndarray], np.ndarray] = getattr(runtime, op)
                y = fn(X)
                write_message(out_fd, {"ok": True, "id": rid}, {"y": encode_array(y)})
            elif op == "featurize":
                v = runtime.featurize(msg.payloads.get("raw", b""))
                write_message(out_fd, {"ok": True, "id": rid}, {"v": encode_array(v)})
            elif op == "tree_ensemble":
                te, how = runtime.tree_ensemble()
                if te is None:
                    write_message(out_fd, {"ok": True, "id": rid, "available": False, "reason": how})
                else:
                    write_message(
                        out_fd,
                        {"ok": True, "id": rid, "available": True, "how": how, "n_trees": te.n_trees},
                        {"trees": te.to_bytes()},
                    )
            elif op == "self_check":
                write_message(out_fd, {"ok": True, "id": rid, "self_check": self_check(init.get("host_home"))})
            elif op == "ping":
                write_message(out_fd, {"ok": True, "id": rid})
            elif op == "shutdown":
                write_message(out_fd, {"ok": True, "id": rid})
                return 0
            else:
                write_message(out_fd, {"ok": False, "id": rid, "error_type": "ProtocolError",
                                       "message": f"unknown op {op!r}"})
        except (BrokenPipeError, ConnectionResetError):
            return 0
        except BaseException as e:  # noqa: BLE001 - adapter errors (incl. SystemExit) go to the host
            if isinstance(e, KeyboardInterrupt):
                raise
            log.warning("%s failed: %s: %s", op, type(e).__name__, e)
            try:
                write_message(out_fd, _error_header(rid, e))
            except (BrokenPipeError, ConnectionResetError):
                return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the worker process (see :mod:`malvalid.sandbox.host`)."""
    ap = argparse.ArgumentParser(prog="malvalid-sandbox-worker")
    ap.add_argument("--in-fd", type=int, default=None)
    ap.add_argument("--out-fd", type=int, default=None)
    ap.add_argument("--stdio", action="store_true",
                    help="use the original stdin/stdout as the channel (Windows); fds 0/1 are then re-pointed")
    ap.add_argument("--memory-mb", type=int, default=0)
    ap.add_argument("--nofile", type=int, default=4096)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    if args.stdio:
        args.in_fd, args.out_fd = _take_stdio_channel()
    elif args.in_fd is None or args.out_fd is None:
        ap.error("--in-fd and --out-fd are required unless --stdio is given")

    _set_parent_death_signal()
    limits = apply_rlimits(args.memory_mb, args.nofile)
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s worker[%(process)d] %(levelname)s %(name)s: %(message)s",
    )
    with contextlib.suppress(Exception):
        import faulthandler

        faulthandler.enable(file=sys.stderr)
    from malvalid.sandbox.protocol import read_message, write_message

    try:
        init_msg = read_message(args.in_fd)
    except EOFError:
        return 0
    init = init_msg.header
    try:
        return _serve(args.in_fd, args.out_fd, init, limits)
    except BaseException as e:  # noqa: BLE001 - start-up failure: tell the host, then exit
        log.error("worker start-up failed: %s", e, exc_info=True)
        with contextlib.suppress(Exception):
            write_message(args.out_fd, _error_header(None, e))
        return 3


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
