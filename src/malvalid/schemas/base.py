"""FeatureSchema plugin interface.

A FeatureSchema describes a fixed tabular feature space (e.g. EMBER v2 = 2381 floats): its layout
as contiguous named groups, how attacker-controllable each feature is, how to build vectors from
raw inputs.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


class Controllability(str, enum.Enum):
    """How freely an attacker can move a feature without breaking the program.

    CONTROLLABLE : settable to arbitrary values at ~zero cost (e.g. COFF timestamp, checksum,
                   DOS stub, section names, debug/rich-header fields).
    APPEND_ONLY  : can only be increased / added to (e.g. add imports, add sections, append
                   strings or overlay bytes, file size).
    DERIVED      : moves only as a side effect of other file changes (e.g. byte histogram or
                   entropy histogram shift when bytes are appended).
    FIXED        : cannot change without altering program semantics (e.g. machine type,
                   entry-point section characteristics, the original import set).
    """

    CONTROLLABLE = "controllable"
    APPEND_ONLY = "append_only"
    DERIVED = "derived"
    FIXED = "fixed"


# Default weight an attacker-controllability analysis (M7) gives each level.
CONTROLLABILITY_WEIGHT: dict[Controllability, float] = {
    Controllability.CONTROLLABLE: 1.0,
    Controllability.APPEND_ONLY: 0.75,
    Controllability.DERIVED: 0.5,
    Controllability.FIXED: 0.0,
}


@dataclass(frozen=True)
class FeatureGroup:
    """A contiguous block ``[start, stop)`` of the feature vector."""

    name: str
    start: int
    stop: int
    controllability: Controllability
    description: str = ""

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


class FeatureSchema(abc.ABC):
    """Base class for feature-schema plugins (registered under ``malvalid.feature_schemas``)."""

    name: ClassVar[str]  # e.g. "ember_v2" — matched against SubmittedDetector.feature_version
    dim: ClassVar[int]
    description: ClassVar[str] = ""

    # ---- layout ----------------------------------------------------------------------------

    @abc.abstractmethod
    def groups(self) -> list[FeatureGroup]:
        """Contiguous groups covering exactly ``[0, dim)`` in order."""

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for g in self.groups():
            names.extend(f"{g.name}[{i}]" for i in range(g.size))
        return names

    def feature_controllability(self) -> list[Controllability]:
        """Per-feature controllability. Override for finer granularity than groups."""
        out: list[Controllability] = []
        for g in self.groups():
            out.extend([g.controllability] * g.size)
        return out

    # ---- derived helpers (no need to override) -----------------------------------------------

    def group_of(self, index: int) -> FeatureGroup:
        for g in self.groups():
            if g.start <= index < g.stop:
                return g
        raise IndexError(f"feature index {index} outside schema {self.name} (dim={self.dim})")

    def group_index(self) -> np.ndarray:
        """(dim,) int array mapping each feature to its group's position in ``groups()``."""
        gi = np.empty(self.dim, dtype=np.int32)
        for k, g in enumerate(self.groups()):
            gi[g.start : g.stop] = k
        return gi

    def controllability_array(self) -> np.ndarray:
        """(dim,) array of Controllability values (object dtype)."""
        return np.array(self.feature_controllability(), dtype=object)

    def mask(self, *levels: Controllability) -> np.ndarray:
        """(dim,) bool mask of features whose controllability is in ``levels``."""
        levels_set = set(levels)
        return np.array([c in levels_set for c in self.feature_controllability()], dtype=bool)

    def check_layout(self) -> None:
        """Raise ValueError if groups/names/controllability are inconsistent with ``dim``."""
        pos = 0
        for g in self.groups():
            if g.start != pos or g.stop <= g.start:
                raise ValueError(f"{self.name}: group {g.name} is not contiguous at {pos}")
            pos = g.stop
        if pos != self.dim:
            raise ValueError(f"{self.name}: groups cover {pos} features, dim is {self.dim}")
        if len(self.feature_names()) != self.dim:
            raise ValueError(f"{self.name}: feature_names() length != dim")
        if len(self.feature_controllability()) != self.dim:
            raise ValueError(f"{self.name}: feature_controllability() length != dim")

    def validate_matrix(self, X: np.ndarray) -> None:
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(
                f"feature matrix has shape {X.shape}; schema {self.name} expects (n, {self.dim})"
            )

    # ---- construction from raw inputs --------------------------------------------------------

    def vectorize_raw(self, raw: dict[str, Any]) -> np.ndarray:
        """EMBER-style raw JSON feature dict -> (dim,) float32 vector."""
        raise NotImplementedError(f"{self.name} does not implement vectorize_raw")

    def featurize_available(self) -> bool:
        """True if :meth:`featurize` can run in this environment (optional deps present)."""
        return False

    def featurize(self, raw: bytes) -> np.ndarray:
        """Raw PE bytes -> (dim,) float32 vector. Raises FeaturizeUnavailable if not possible."""
        from malvalid.core import FeaturizeUnavailable

        raise FeaturizeUnavailable(f"{self.name} has no raw-bytes featurizer in this environment")

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "description": self.description,
            "groups": [
                {
                    "name": g.name,
                    "start": g.start,
                    "stop": g.stop,
                    "controllability": g.controllability.value,
                    "description": g.description,
                }
                for g in self.groups()
            ],
            "featurize_available": self.featurize_available(),
        }
