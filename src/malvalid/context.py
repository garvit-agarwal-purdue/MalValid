"""Objects handed to test modules: the model handle protocol, run context, artifact store."""

from __future__ import annotations

import datetime as dt
import logging
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

import numpy as np

from malvalid.core import GateMode, ModuleTimeout, Requirement, to_jsonable

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.config import GateConfig
    from malvalid.corpora.base import Corpus
    from malvalid.loaders.trees import TreeEnsemble
    from malvalid.schemas.base import FeatureSchema


# --------------------------------------------------------------------------------------------------
# Model declarations & handle
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelDeclarations:
    """What the adapter declares about itself (read in the sandbox before ``load()``)."""

    feature_version: str
    model_kind: str
    operating_threshold: float
    training_hashes_path: str | None  # resolved absolute path, or None
    training_cutoff: str | None  # ISO date string as declared, or None
    model_paths: tuple[str, ...] = ()  # resolved absolute artifact paths (scanned by M0)
    adapter_path: str = ""
    class_name: str = ""
    has_featurize: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self.__dict__)


@runtime_checkable
class ModelHandle(Protocol):
    """Query interface to the submitted model. Implemented by the sandbox proxy
    (:class:`malvalid.sandbox.host.SandboxedModel`) and :class:`malvalid.testing.InProcessModel`.

    Implementations validate outputs: ``predict_proba`` returns a finite float64 (n,) array in
    [0, 1]; ``predict`` returns an int8 (n,) array in {0, 1}.
    """

    declarations: ModelDeclarations

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def has_featurize(self) -> bool:
        """True if raw bytes can be featurized (adapter featurize, else the schema's extractor
        inside the sandbox)."""
        ...

    def featurize(self, raw: bytes) -> np.ndarray:
        """Raw PE bytes -> (dim,) float32 vector, computed inside the sandbox."""
        ...

    def tree_ensemble(self) -> "TreeEnsemble | None": ...

    @property
    def query_count(self) -> int: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------------------------------
# Artifact store (curves/tables for the report)
# --------------------------------------------------------------------------------------------------


def _downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    idx = np.unique(np.linspace(0, x.size - 1, max_points).round().astype(np.int64))
    return x[idx], y[idx]


class ArtifactStore:
    """Charts and tables that go into report.json / report.html, plus private run-dir files.

    Chart artifacts are small (downsampled). Anything sensitive — adversarial vectors, perturbed
    samples — goes through :meth:`save_private`, which writes under ``private_dir`` and is never
    referenced from the report (see spec §11).
    """

    def __init__(self, private_dir: Path | None = None):
        self._items: dict[str, dict[str, Any]] = {}
        self.private_dir = private_dir

    def add_chart(
        self,
        module_id: str,
        name: str,
        *,
        title: str,
        series: Sequence[dict[str, Any]],
        kind: str = "line",
        xlabel: str = "",
        ylabel: str = "",
        xscale: str = "linear",
        yscale: str = "linear",
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        reference_lines: Sequence[dict[str, Any]] = (),
        max_points: int = 400,
        note: str = "",
    ) -> str:
        """Add a chart. ``series`` items: ``{"label": str, "x": array, "y": array}``.

        ``kind``: "line" | "step" | "scatter" | "bar" (bar: x are category labels).
        ``reference_lines``: ``{"axis": "x"|"y", "value": float, "label": str}`` (e.g. threshold,
        diagonal is drawn by the renderer when kind == "line" and ``note == "diagonal"``).
        """
        key = f"{module_id}.{name}"
        out_series = []
        for s in series:
            if kind == "bar":
                xs, ys = list(s["x"]), np.asarray(s["y"], dtype=np.float64)
                out_series.append({"label": s.get("label", ""), "x": xs, "y": ys})
                continue
            x = np.asarray(s["x"], dtype=np.float64)
            y = np.asarray(s["y"], dtype=np.float64)
            if x.shape != y.shape:
                raise ValueError(f"chart {key}: x/y shape mismatch {x.shape} vs {y.shape}")
            x, y = _downsample(x, y, max_points)
            out_series.append({"label": s.get("label", ""), "x": x, "y": y})
        self._items[key] = to_jsonable(
            {
                "type": "chart",
                "module": module_id,
                "name": name,
                "title": title,
                "kind": kind,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "xscale": xscale,
                "yscale": yscale,
                "xlim": list(xlim) if xlim else None,
                "ylim": list(ylim) if ylim else None,
                "reference_lines": list(reference_lines),
                "series": out_series,
                "note": note,
            }
        )
        return key

    def add_curve(
        self, module_id: str, name: str, x: Any, y: Any, *, title: str, label: str = "", **kw: Any
    ) -> str:
        return self.add_chart(module_id, name, title=title, series=[{"label": label, "x": x, "y": y}], **kw)

    def add_table(
        self,
        module_id: str,
        name: str,
        *,
        title: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        note: str = "",
        max_rows: int = 200,
    ) -> str:
        key = f"{module_id}.{name}"
        self._items[key] = to_jsonable(
            {
                "type": "table",
                "module": module_id,
                "name": name,
                "title": title,
                "columns": list(columns),
                "rows": [list(r) for r in list(rows)[:max_rows]],
                "truncated": len(rows) > max_rows,
                "note": note,
            }
        )
        return key

    def save_private(self, module_id: str, name: str, **arrays: np.ndarray) -> Path | None:
        """Write arrays to ``private_dir/<module>/<name>.npz`` (never part of the report)."""
        if self.private_dir is None:
            return None
        d = self.private_dir / module_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.npz"
        np.savez_compressed(p, **arrays)
        return p

    def keys_for(self, module_id: str) -> list[str]:
        return [k for k, v in self._items.items() if v.get("module") == module_id]

    def get(self, key: str) -> dict[str, Any]:
        return self._items[key]

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return dict(self._items)


# --------------------------------------------------------------------------------------------------
# Run context
# --------------------------------------------------------------------------------------------------


def module_seed(base_seed: int, module_id: str) -> int:
    """Stable per-module seed derived from the config seed (recorded in the report)."""
    return (int(base_seed) * 1_000_003 + zlib.crc32(module_id.encode())) % (2**32)


@dataclass
class RunContext:
    """Everything a module may use. Built fresh per module by the runner."""

    config: "GateConfig"
    module_id: str
    params: dict[str, Any]
    gate: GateMode
    model: ModelHandle
    schema: "FeatureSchema"
    corpus: "Corpus | None"
    training_hashes: frozenset[str] | None
    training_cutoff: dt.date | None
    sample_dir: Path | None
    run_dir: Path
    private_dir: Path
    artifacts: ArtifactStore
    seed: int
    rng: np.random.Generator
    capabilities: frozenset[Requirement]
    deadline: float | None = None  # time.monotonic() deadline
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("malvalid"))
    extras: dict[str, Any] = field(default_factory=dict)  # runner-provided extras (e.g. M0 info)

    def time_left(self) -> float:
        return float("inf") if self.deadline is None else self.deadline - time.monotonic()

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ModuleTimeout(
                f"module {self.module_id} exceeded runtime.max_seconds_per_module"
            )

    def has(self, req: Requirement) -> bool:
        return req in self.capabilities

    def missing_reason(self, req: Requirement) -> str:
        """Why ``req`` is unavailable in this run (for skip reasons and notes).

        The runner records run-specific explanations in ``extras["unmet_reasons"]`` — the corpus
        error for FEATURE_SPACE, the rejected tree dump for TREE_ACCESS, an empty training manifest
        for TRAINING_HASHES — so a module's own skip cites the real cause; otherwise the generic
        :data:`REQUIREMENT_HINTS` text is returned.
        """
        why = (self.extras or {}).get("unmet_reasons")
        text = why.get(req.value) if isinstance(why, dict) else None
        return str(text) if text else REQUIREMENT_HINTS.get(req, req.value)

    @property
    def threshold(self) -> float:
        return float(self.model.declarations.operating_threshold)

    def score(self, X: np.ndarray) -> np.ndarray:
        """``model.predict_proba`` with a deadline check (use this in loops)."""
        self.check_deadline()
        return self.model.predict_proba(X)


def compute_capabilities(
    *,
    model: ModelHandle | None,
    schema: "FeatureSchema | None",
    corpus: "Corpus | None",
    training_hashes: frozenset[str] | None,
    training_cutoff: dt.date | None,
    sample_dir: Path | None,
    tree_access: bool,
) -> frozenset[Requirement]:
    """The single definition of which Requirements are satisfied for a run."""
    caps: set[Requirement] = set()
    if model is not None:
        caps.add(Requirement.QUERY_ONLY)
    if (
        model is not None
        and schema is not None
        and corpus is not None
        and corpus.feature_version == schema.name == model.declarations.feature_version
    ):
        caps.add(Requirement.FEATURE_SPACE)
    if training_hashes:
        caps.add(Requirement.TRAINING_HASHES)
    if training_cutoff is not None:
        caps.add(Requirement.TRAINING_CUTOFF)
    if tree_access:
        caps.add(Requirement.TREE_ACCESS)
    if sample_dir is not None and Path(sample_dir).is_dir():
        caps.add(Requirement.SAMPLE_DIR)
    # Featurization of raw (possibly live-malware) bytes always happens behind the model handle —
    # i.e. inside the sandbox worker — never in the harness process.
    if model is not None and model.has_featurize():
        caps.add(Requirement.FEATURIZE)
    return frozenset(caps)


REQUIREMENT_HINTS: dict[Requirement, str] = {
    Requirement.FEATURE_SPACE: "no canonical corpus in the model's feature_version is loaded",
    Requirement.SAMPLE_DIR: "no sample corpus (sample_dir not set or missing)",
    Requirement.FEATURIZE: "no raw-bytes featurizer (adapter has no featurize() and the schema's extractor is unavailable)",
    Requirement.TRAINING_HASHES: "no training manifest (training_hashes_path not declared)",
    Requirement.TRAINING_CUTOFF: "training_cutoff unknown (not declared by the adapter)",
    Requirement.TREE_ACCESS: "model does not expose its tree structure",
    Requirement.QUERY_ONLY: "model not loaded",
}
