"""ModelLoader plugin interface and artifact-format helpers.

Loaders run *inside the sandboxed worker* (the artifact is untrusted). They turn an artifact path
into a native model object, score it, and expose its trees as a normalized
:class:`~malvalid.loaders.trees.TreeEnsemble`.
"""

from __future__ import annotations

import abc
import zipfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from malvalid.core import PickleRefused, UnsupportedTreeError
from malvalid.loaders.trees import TreeEnsemble

# Extensions that are (or usually wrap) Python pickles.
PICKLE_EXTENSIONS = (".pkl", ".pickle", ".joblib", ".jbl", ".sav", ".dill", ".pt", ".pth", ".bin", ".ckpt")


def sniff_format(path: Path) -> str:
    """Best-effort content sniffing: 'pickle', 'joblib-compressed', 'zip-pickle', 'zip', 'json',
    'ubj', 'lightgbm-text', 'onnx-protobuf', 'numpy', or 'unknown'. Never deserializes."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(64)
    except OSError:
        return "unknown"
    if not head:
        return "unknown"
    if head[:1] == b"\x80" and len(head) > 1 and 2 <= head[1] <= 5:
        return "pickle"
    if head[:6] == b"\x93NUMPY":
        return "numpy"
    if head[:2] == b"PK":
        try:
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
            if any(n.endswith((".pkl", "data.pkl")) for n in names):
                return "zip-pickle"
        except zipfile.BadZipFile:
            return "unknown"
        return "zip"
    if head[:1] in (b"\x78",) and len(head) > 1 and head[1] in (0x01, 0x5E, 0x9C, 0xDA):
        return "joblib-compressed"  # zlib stream; joblib's default compressed pickle container
    if head[:3] in (b"\x1f\x8b\x08",) or head[:3] == b"BZh" or head[:6] == b"\xfd7zXZ\x00":
        return "joblib-compressed"  # gzip / bz2 / xz — joblib supports all as pickle containers
    stripped = head.lstrip()
    if stripped[:1] in (b"{", b"["):
        return "json"
    if head.startswith(b"tree\n") or head.startswith(b"tree\r\n"):
        return "lightgbm-text"
    if head[:1] == b"{" or head[:1] in (b"L", b"i", b"U", b"$", b"#"):  # UBJSON markers
        return "ubj"
    if head[:1] == b"\x08":  # protobuf field 1 (ir_version) varint — ONNX ModelProto
        return "onnx-protobuf"
    return "unknown"


def is_pickle_artifact(path: Path) -> bool:
    """True if the artifact is pickle-based by extension or by content."""
    p = Path(path)
    return p.suffix.lower() in PICKLE_EXTENSIONS or sniff_format(p) in (
        "pickle",
        "joblib-compressed",
        "zip-pickle",
    )


class ModelLoader(abc.ABC):
    """Loader plugin (registered under ``malvalid.model_loaders``)."""

    kind: ClassVar[str]  # matches SubmittedDetector.model_kind, e.g. "lightgbm"
    extensions: ClassVar[tuple[str, ...]]  # all artifact extensions this loader handles
    safe_extensions: ClassVar[tuple[str, ...]] = ()  # non-executable formats (preferred)
    description: ClassVar[str] = ""

    def can_load(self, path: Path) -> bool:
        return Path(path).suffix.lower() in self.extensions

    def is_pickle(self, path: Path) -> bool:
        return is_pickle_artifact(path)

    def check_pickle_policy(self, path: Path, allow_pickle: bool) -> None:
        if self.is_pickle(path) and not allow_pickle:
            raise PickleRefused(
                f"{path} is a pickle-based artifact; pickles can execute code on load. "
                "Re-export the model in a non-pickle format (LightGBM .txt, XGBoost .json/.ubj, "
                "ONNX) or pass --allow-pickle to accept the risk."
            )

    @abc.abstractmethod
    def load(self, path: Path, *, allow_pickle: bool = False) -> Any:
        """Deserialize the artifact into a native model object (worker-side only)."""

    @abc.abstractmethod
    def supports_native(self, obj: Any) -> bool:
        """True if ``obj`` is a native model object this loader understands."""

    @abc.abstractmethod
    def predict_proba(self, native: Any, X: np.ndarray) -> np.ndarray:
        """(n,) malicious-class probability."""

    def tree_ensemble(self, native: Any) -> TreeEnsemble:
        """Normalized trees. Raise UnsupportedTreeError if the model can't expose them."""
        raise UnsupportedTreeError(f"{self.kind} loader cannot expose tree structure")

    def info(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "extensions": list(self.extensions),
            "safe_extensions": list(self.safe_extensions),
            "description": self.description,
        }


def load_model(path: str | Path, kind: str | None = None, *, allow_pickle: bool | None = None) -> Any:
    """Convenience for adapters: load an artifact with the matching registered loader.

    ``allow_pickle=None`` defers to the sandbox policy (the ``MALVALID_ALLOW_PICKLE`` env var the
    worker sets from ``--allow-pickle``); outside a sandbox it defaults to False.
    """
    import os

    from malvalid import registry

    p = Path(path)
    if allow_pickle is None:
        allow_pickle = os.environ.get("MALVALID_ALLOW_PICKLE") == "1"
    loaders = registry.model_loaders()
    if kind is not None:
        if kind not in loaders:
            raise KeyError(f"no model loader for kind {kind!r}; known: {sorted(loaders)}")
        cands = [loaders[kind]()]
    else:
        cands = [cls() for cls in loaders.values() if cls().can_load(p)]
        if not cands:
            raise KeyError(f"no model loader handles {p.suffix!r} artifacts")
    return cands[0].load(p, allow_pickle=allow_pickle)


def find_loader_for_native(obj: Any) -> ModelLoader | None:
    from malvalid import registry

    for cls in registry.model_loaders().values():
        ld = cls()
        try:
            if ld.supports_native(obj):
                return ld
        except Exception:  # pragma: no cover - defensive
            continue
    return None
