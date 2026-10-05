"""Infer what malvalid needs to know about a submitted model file, without running it.

:func:`inspect_model` reads a model artifact *as data* in the calling process and reports

* ``model_kind`` — from the file's content (magic bytes / structure), not only its extension: a
  LightGBM text model (``tree`` header, ``version=v…``; e.g. ``EMBER2024_PE.model``), an XGBoost
  JSON or UBJSON model (``learner`` object), or an ONNX protobuf;
* ``n_features`` — LightGBM ``max_feature_idx + 1`` / ``feature_names``, XGBoost
  ``learner_model_param.num_feature``, ONNX graph input shape;
* ``feature_version`` — the installed feature schema with that many features (EMBER v2: 2381,
  EMBER v3: 2568), plus the default canonical corpus for it.

Nothing is executed or deserialized with a model library: LightGBM headers are parsed as text, XGBoost
JSON with the stdlib parser and UBJSON with a small bounded parser here, ONNX with the ``onnx``
protobuf bindings when installed. **Pickle-based files are never unpickled**: a static opcode scan
(:func:`malvalid.loaders.lightgbm_loader.pickle_top_module`) may *suggest* a kind, but the feature
version must be chosen by the user, and loading still requires the explicit allow-pickle opt-in.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_CORPORA",
    "ModelInfo",
    "default_corpus_for",
    "feature_version_for",
    "inspect_model",
    "supported_feature_dims",
]

#: Canonical corpus used by default for each feature version (real data preferred over synthetic).
DEFAULT_CORPORA: dict[str, str] = {"ember_v2": "ember_v2_2018", "ember_v3": "ember_v3_2024"}
MAX_JSON_BYTES = 2 << 30  # XGBoost JSON models above 2 GiB are not parsed here
MAX_HEADER_BYTES = 8 << 20  # LightGBM header (feature_names can be long)
_FALLBACK_DIMS = {2381: "ember_v2", 2568: "ember_v3"}


@dataclass
class ModelInfo:
    """What :func:`inspect_model` found. ``None`` fields could not be determined."""

    path: str
    file_name: str
    size: int
    format: str  # lightgbm-text | xgboost-json | xgboost-ubj | onnx | pickle | unknown
    model_kind: str | None = None
    n_features: int | None = None
    feature_version: str | None = None
    default_corpus: str | None = None
    is_pickle: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Everything needed was detected (kind + feature version) and nothing is wrong."""
        return not self.errors and self.model_kind is not None and self.feature_version is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "path": self.path,
            "file_name": self.file_name,
            "size": self.size,
            "format": self.format,
            "model_kind": self.model_kind,
            "n_features": self.n_features,
            "feature_version": self.feature_version,
            "default_corpus": self.default_corpus,
            "is_pickle": self.is_pickle,
            "details": dict(self.details),
            "notes": list(self.notes),
            "errors": list(self.errors),
            "supported": {str(k): v for k, v in supported_feature_dims().items()},
        }


# --------------------------------------------------------------------------------------------------
# Feature versions and corpora
# --------------------------------------------------------------------------------------------------


def supported_feature_dims() -> dict[int, str]:
    """``{n_features: feature_version}`` of the installed feature schemas."""
    out: dict[int, str] = {}
    try:
        from malvalid import registry

        for name in sorted(registry.feature_schemas()):
            try:
                dim = int(registry.get_schema(name).dim)
            except Exception:  # noqa: BLE001 - a broken schema plugin must not hide the others
                continue
            out.setdefault(dim, name)
    except Exception:  # noqa: BLE001  # pragma: no cover - registry unavailable
        pass
    return out or dict(_FALLBACK_DIMS)


def _supported_text() -> str:
    dims = supported_feature_dims()
    names = {"ember_v2": "EMBER v2", "ember_v3": "EMBER v3"}
    return " and ".join(f"{names.get(v, v)} ({k})" for k, v in sorted(dims.items(), key=lambda kv: kv[1]))


def feature_version_for(n_features: int | None) -> str | None:
    if n_features is None:
        return None
    return supported_feature_dims().get(int(n_features))


def default_corpus_for(feature_version: str | None) -> str | None:
    """The canonical corpus a model in ``feature_version`` is evaluated on by default."""
    if not feature_version:
        return None
    if feature_version in DEFAULT_CORPORA:
        return DEFAULT_CORPORA[feature_version]
    try:  # another installed feature version: its first non-synthetic corpus
        from malvalid import registry

        for name, cls in sorted(registry.corpora().items()):
            try:
                info = cls().info()
            except Exception:  # noqa: BLE001
                continue
            if info.get("feature_version") == feature_version and not info.get("synthetic"):
                return name
    except Exception:  # noqa: BLE001  # pragma: no cover
        pass
    return None


# --------------------------------------------------------------------------------------------------
# Format parsers (data only)
# --------------------------------------------------------------------------------------------------


def _lightgbm_header(path: Path) -> dict[str, str]:
    """``key=value`` lines of a LightGBM text model before the first tree (bounded read)."""
    out: dict[str, str] = {}
    read = 0
    with open(path, "rb") as f:
        first = f.readline(1024)
        read += len(first)
        if first.strip() != b"tree":
            return out
        while read < MAX_HEADER_BYTES:
            line = f.readline(MAX_HEADER_BYTES - read)
            if not line:
                break
            read += len(line)
            s = line.strip()
            if not s:
                continue
            if s.startswith(b"Tree=") or s == b"end of trees":
                break
            k, eq, v = s.partition(b"=")
            if eq and re.fullmatch(rb"[A-Za-z_][A-Za-z0-9_]*", k):
                out[k.decode("ascii")] = v.decode("utf-8", "replace")
    return out


def _inspect_lightgbm(path: Path, info: ModelInfo) -> None:
    hdr = _lightgbm_header(path)
    info.format = "lightgbm-text"
    info.model_kind = "lightgbm"
    info.details["lightgbm_version"] = hdr.get("version")
    info.details["objective"] = hdr.get("objective")
    n: int | None = None
    if "max_feature_idx" in hdr:
        try:
            n = int(hdr["max_feature_idx"]) + 1
        except ValueError:
            info.notes.append(f"unreadable max_feature_idx {hdr['max_feature_idx'][:40]!r}")
    names = hdr.get("feature_names")
    if names:
        n_names = len(names.split())
        info.details["n_feature_names"] = n_names
        if n is None:
            n = n_names
        elif n_names != n:
            info.notes.append(f"feature_names lists {n_names} names but max_feature_idx implies {n}")
    if hdr.get("num_class") not in (None, "1"):
        info.notes.append(f"num_class={hdr.get('num_class')}: malvalid expects a binary (malicious-vs-benign) model")
    obj = (hdr.get("objective") or "").split(" ")[0]
    if obj and obj not in ("binary", "cross_entropy", "xentropy", "regression", "regression_l2", "regression_l1"):
        info.notes.append(f"objective {obj!r} may not produce malicious-class probabilities")
    info.n_features = n
    if n is None:
        info.errors.append("the LightGBM header has no max_feature_idx/feature_names; cannot tell how many features the model expects")


class _UBJError(ValueError):
    pass


class _UBJReader:
    """Minimal bounded UBJSON reader (the subset XGBoost writes). Containers are capped in size."""

    _FIXED = {b"i": (">b", 1), b"U": (">B", 1), b"I": (">h", 2), b"l": (">i", 4), b"L": (">q", 8),
              b"d": (">f", 4), b"D": (">d", 8)}
    MAX_ITEMS = 50_000_000

    def __init__(self, data: bytes):
        self.b = data
        self.i = 0

    def _take(self, n: int) -> bytes:
        if n < 0 or self.i + n > len(self.b):
            raise _UBJError("truncated UBJSON")
        out = self.b[self.i : self.i + n]
        self.i += n
        return out

    def _marker(self) -> bytes:
        while True:
            m = self._take(1)
            if m != b"N":  # no-op
                return m

    def _int(self, m: bytes) -> int:
        if m not in (b"i", b"U", b"I", b"l", b"L"):
            raise _UBJError(f"expected an integer marker, got {m!r}")
        fmt, n = self._FIXED[m]
        return int(struct.unpack(fmt, self._take(n))[0])

    def _str(self) -> str:
        n = self._int(self._marker())
        return self._take(n).decode("utf-8", "replace")

    def value(self, m: bytes | None = None, depth: int = 0) -> Any:
        if depth > 64:
            raise _UBJError("UBJSON nesting too deep")
        m = m or self._marker()
        if m in self._FIXED:
            fmt, n = self._FIXED[m]
            return struct.unpack(fmt, self._take(n))[0]
        if m == b"Z":
            return None
        if m == b"T":
            return True
        if m == b"F":
            return False
        if m == b"C":
            return self._take(1).decode("latin-1")
        if m in (b"S", b"H"):
            return self._str()
        if m == b"[":
            return self._container(depth, is_obj=False)
        if m == b"{":
            return self._container(depth, is_obj=True)
        raise _UBJError(f"unknown UBJSON marker {m!r}")

    def _container(self, depth: int, *, is_obj: bool) -> Any:
        typ: bytes | None = None
        count: int | None = None
        if self.b[self.i : self.i + 1] == b"$":
            self.i += 1
            typ = self._take(1)
        if self.b[self.i : self.i + 1] == b"#":
            self.i += 1
            count = self._int(self._marker())
            if count > self.MAX_ITEMS:
                raise _UBJError("UBJSON container too large")
        if typ is not None and count is None:
            raise _UBJError("typed UBJSON container without a count")
        if not is_obj and typ in self._FIXED and count is not None:  # typed numeric array: skip the bytes
            self._take(self._FIXED[typ][1] * count)
            return []  # contents are irrelevant for inspection
        out_obj: dict[str, Any] = {}
        out_arr: list[Any] = []
        k = 0
        while True:
            if count is None:
                if self.b[self.i : self.i + 1] == b"}" and is_obj or self.b[self.i : self.i + 1] == b"]" and not is_obj:
                    self.i += 1
                    break
            elif k >= count:
                break
            if is_obj:
                key = self._str()
                out_obj[key] = self.value(typ, depth + 1)
            else:
                out_arr.append(self.value(typ, depth + 1))
            k += 1
        return out_obj if is_obj else out_arr


def _xgb_learner(doc: Any) -> dict[str, Any] | None:
    if isinstance(doc, dict) and isinstance(doc.get("learner"), dict):
        return doc["learner"]
    return None


def _inspect_xgboost_doc(doc: Any, info: ModelInfo) -> None:
    learner = _xgb_learner(doc)
    if learner is None:
        info.errors.append("this JSON file is not an XGBoost model (no 'learner' object)")
        return
    info.model_kind = "xgboost"
    ver = doc.get("version") if isinstance(doc, dict) else None
    if isinstance(ver, list):
        info.details["xgboost_version"] = ".".join(str(v) for v in ver)
    params = learner.get("learner_model_param") if isinstance(learner.get("learner_model_param"), dict) else {}
    obj = learner.get("objective") if isinstance(learner.get("objective"), dict) else {}
    info.details["objective"] = obj.get("name")
    if obj.get("name") and not str(obj.get("name")).startswith(("binary:", "reg:logistic")):
        info.notes.append(f"objective {obj.get('name')!r} may not produce malicious-class probabilities")
    raw = params.get("num_feature")
    try:
        info.n_features = int(str(raw)) if raw is not None else None
    except ValueError:
        info.n_features = None
    if info.n_features is None:
        names = learner.get("feature_names")
        if isinstance(names, list) and names:
            info.n_features = len(names)
    if info.n_features is None:
        info.errors.append("the XGBoost model does not record num_feature; cannot tell how many features it expects")


def _inspect_json(path: Path, info: ModelInfo) -> None:
    if info.size > MAX_JSON_BYTES:
        info.errors.append(f"the JSON model is larger than {MAX_JSON_BYTES >> 30} GiB; choose the model kind and feature version manually")
        info.format = "json"
        return
    try:
        with open(path, "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, MemoryError) as e:
        info.format = "json"
        info.errors.append(f"not a valid JSON model file ({type(e).__name__})")
        return
    info.format = "xgboost-json"
    _inspect_xgboost_doc(doc, info)
    if info.model_kind is None:
        info.format = "json"


def _inspect_ubj(path: Path, info: ModelInfo) -> bool:
    """True if the file parsed as an UBJSON object."""
    if info.size > MAX_JSON_BYTES:
        return False
    try:
        with open(path, "rb") as f:
            data = f.read()
        r = _UBJReader(data)
        if r._marker() != b"{":
            return False
        doc = r._container(0, is_obj=True)
    except (_UBJError, struct.error, RecursionError, OSError, MemoryError):
        return False
    if _xgb_learner(doc) is None:
        return False
    info.format = "xgboost-ubj"
    _inspect_xgboost_doc(doc, info)
    return True


def _inspect_onnx(path: Path, info: ModelInfo) -> None:
    info.format = "onnx"
    info.model_kind = "onnx"
    try:
        import onnx  # protobuf bindings: parses the graph description, runs nothing
    except ImportError:
        info.notes.append("the onnx package is not installed, so the input shape could not be read")
        return
    try:
        proto = onnx.load(str(path), load_external_data=False)
    except Exception as e:  # noqa: BLE001
        info.model_kind = None
        info.format = "unknown"
        info.errors.append(f"not a readable ONNX model ({type(e).__name__}: {str(e)[:200]})")
        return
    graph = proto.graph
    init_names = {t.name for t in graph.initializer}
    inputs = [i for i in graph.input if i.name not in init_names]
    info.details["inputs"] = [i.name for i in inputs]
    info.details["producer"] = proto.producer_name or None
    if len(inputs) != 1:
        info.notes.append(f"the ONNX graph has {len(inputs)} inputs; malvalid feeds one (n, features) float matrix")
    if inputs:
        dims = inputs[0].type.tensor_type.shape.dim
        if dims and dims[-1].HasField("dim_value") and dims[-1].dim_value > 0:
            info.n_features = int(dims[-1].dim_value)
        else:
            info.notes.append("the ONNX input's feature dimension is symbolic; choose the feature version manually")


def _inspect_pickle(path: Path, info: ModelInfo) -> None:
    info.format = "pickle"
    info.is_pickle = True
    hint: str | None = None
    try:
        from malvalid.loaders.lightgbm_loader import pickle_top_module  # static opcode scan, never unpickles

        top = pickle_top_module(path)
    except Exception:  # noqa: BLE001
        top = None
    if top:
        pkg = top.split(".")[0]
        hint = {"sklearn": "sklearn_gbdt", "lightgbm": "lightgbm", "xgboost": "xgboost"}.get(pkg)
        info.details["pickle_top_module"] = top
    info.model_kind = hint
    info.notes.append(
        "pickle-based file: malvalid never unpickles it outside the sandbox, so the number of features cannot "
        "be read; choose the feature version yourself, and tick allow-pickle (pickles can run code when loaded)"
        + (f". It appears to contain a {top} object." if top else "")
    )


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------


def inspect_model(path: str | os.PathLike[str]) -> ModelInfo:
    """Inspect one model file (see the module docstring). Never raises for a bad file: problems go
    to :attr:`ModelInfo.errors`."""
    from malvalid.loaders.base import PICKLE_EXTENSIONS, sniff_format

    p = Path(path)
    try:
        size = p.stat().st_size if p.is_file() else -1
    except OSError:
        size = -1
    info = ModelInfo(path=str(p), file_name=p.name, size=max(size, 0), format="unknown")
    if size < 0:
        info.errors.append(f"model file not found: {p}")
        return info
    if size == 0:
        info.errors.append(f"{p.name} is empty")
        return info
    fmt = sniff_format(p)
    try:
        if fmt in ("pickle", "joblib-compressed", "zip-pickle") or (
            p.suffix.lower() in PICKLE_EXTENSIONS and fmt not in ("lightgbm-text", "json", "onnx-protobuf")
        ):
            _inspect_pickle(p, info)
        elif fmt == "lightgbm-text":
            _inspect_lightgbm(p, info)
        elif fmt == "json":
            # XGBoost UBJSON also starts with "{": try it when the bytes are not JSON text.
            _inspect_json(p, info)
            if info.model_kind is None and info.errors and info.errors[-1].startswith("not a valid JSON"):
                probe = ModelInfo(path=info.path, file_name=info.file_name, size=info.size, format="unknown")
                if _inspect_ubj(p, probe):
                    info = probe
        elif p.suffix.lower() == ".onnx" or fmt == "onnx-protobuf":
            _inspect_onnx(p, info)
        elif _inspect_ubj(p, info):
            pass
        else:
            info.errors.append(
                f"{p.name}: unrecognized model format. malvalid reads LightGBM text models, XGBoost .json/.ubj "
                "models and ONNX files; for anything else submit a custom adapter (Advanced)"
            )
    except OSError as e:
        info.errors.append(f"cannot read {p.name}: {e.strerror or e}")
        return info
    if info.n_features is not None:
        info.feature_version = feature_version_for(info.n_features)
        if info.feature_version is None:
            info.errors.append(
                f"model expects {info.n_features} features; malvalid supports {_supported_text()}"
            )
    if info.feature_version:
        info.default_corpus = default_corpus_for(info.feature_version)
    return info
