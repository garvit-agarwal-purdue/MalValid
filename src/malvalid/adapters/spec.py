"""Model-file-only submissions: a data-only *spec* instead of a Python adapter.

A researcher who submits just a model file (``malvalid run --model model.txt`` or the web UI's
"Upload your model" form) gets a small JSON spec written next to (symlinks of) the model file::

    {
      "schema": "malvalid-model-spec/1",
      "model_kind": "lightgbm",              # lightgbm | xgboost | sklearn_gbdt | onnx
      "feature_version": "ember_v2",         # a registered feature schema
      "operating_threshold": 0.8336,         # float in [0, 1]; null when threshold_source == "calibrate"
      "threshold_source": "declared",        # declared | calibrate
      "calibrate_fpr": null,                 # target FPR in (0, 0.1] when threshold_source == "calibrate"
      "model_file": "ember_model_2018.txt",  # plain basename inside the spec's directory
      "training_hashes_file": "train_sha256.txt",   # plain basename or null
      "training_cutoff": "2018-10"           # YYYY, YYYY-MM or YYYY-MM-DD, or null
    }

The spec is **data, never code**: no Python is generated from user input. Every field is validated
strictly by :func:`validate_spec` (enums, a finite threshold in [0, 1], a date regex, file names that
are plain basenames), and the sandboxed worker turns a valid spec into a subclass of the packaged
:class:`SpecDetector` (:func:`build_detector_module`), which loads the model through the regular
:mod:`malvalid.loaders` plugins — so the model still runs inside the sandbox with every protection
(bubblewrap, no network, pickle guards, no-unpickle IPC) that an adapter submission gets.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from malvalid.adapter import BaseDetector
from malvalid.core import AdapterError

__all__ = [
    "CALIBRATE_FPR_CHOICES",
    "MODEL_KINDS",
    "SPEC_FILENAME",
    "SPEC_SCHEMA",
    "ModelSpec",
    "SpecDetector",
    "build_detector_module",
    "is_spec_path",
    "load_spec",
    "validate_spec",
    "write_spec",
]

SPEC_SCHEMA = "malvalid-model-spec/1"
#: The same schema as written before the rename to MalValid (read, never written).
LEGACY_SPEC_SCHEMAS = ("malguard-model-spec/1",)
SPEC_FILENAME = "malvalid_model.json"
MAX_SPEC_BYTES = 64 * 1024
#: The model kinds a spec may name (the built-in loader plugins).
MODEL_KINDS = ("lightgbm", "xgboost", "sklearn_gbdt", "onnx")
THRESHOLD_SOURCES = ("declared", "calibrate")
#: Target false-positive rates offered for automatic threshold calibration.
CALIBRATE_FPR_CHOICES = (0.001, 0.005, 0.01)
MAX_CALIBRATE_FPR = 0.1
_KEYS = frozenset({
    "schema", "model_kind", "feature_version", "operating_threshold", "threshold_source", "calibrate_fpr",
    "model_file", "training_hashes_file", "training_cutoff",
})
_BASENAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,254}$")
_CUTOFF_RE = re.compile(r"^\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?$")
_FEATURE_VERSION_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class ModelSpec:
    """A validated spec (see the module docstring)."""

    model_kind: str
    feature_version: str
    operating_threshold: float | None
    threshold_source: str
    calibrate_fpr: float | None
    model_file: str
    training_hashes_file: str | None
    training_cutoff: str | None

    @property
    def calibrate(self) -> bool:
        return self.threshold_source == "calibrate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SPEC_SCHEMA,
            "model_kind": self.model_kind,
            "feature_version": self.feature_version,
            "operating_threshold": self.operating_threshold,
            "threshold_source": self.threshold_source,
            "calibrate_fpr": self.calibrate_fpr,
            "model_file": self.model_file,
            "training_hashes_file": self.training_hashes_file,
            "training_cutoff": self.training_cutoff,
        }


def is_spec_path(path: str | os.PathLike[str]) -> bool:
    """Does ``path`` name a spec file (``*.json``) rather than a Python adapter?"""
    return Path(path).suffix.lower() == ".json"


def _basename(value: Any, field: str, problems: list[str], *, optional: bool) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        problems.append(f"{field} must be a file name" + (" or null" if optional else ""))
        return None
    if (
        value in (".", "..")
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or not _BASENAME_RE.fullmatch(value)
        or value.lower() == SPEC_FILENAME
    ):
        problems.append(
            f"{field} {value!r} must be a plain file name inside the submission folder "
            "(letters, digits, '.', '_', '-'; no directories, no leading dot)"
        )
        return None
    return value


def _known_feature_versions() -> set[str] | None:
    try:
        from malvalid import registry

        return set(registry.feature_schemas())
    except Exception:  # pragma: no cover - registry broken: fall back to the syntactic check
        return None


def validate_spec(data: Any, *, check_registry: bool = True) -> ModelSpec:
    """Strictly validate a spec document; raises :class:`AdapterError` listing every problem."""
    if not isinstance(data, dict):
        raise AdapterError("the model spec must be a JSON object")
    problems: list[str] = []
    unknown = sorted(str(k) for k in data if k not in _KEYS)
    if unknown:
        problems.append(f"unknown key(s) {', '.join(unknown)}")
    schema = data.get("schema", SPEC_SCHEMA)
    if schema != SPEC_SCHEMA and schema not in LEGACY_SPEC_SCHEMAS:
        problems.append(f"schema must be {SPEC_SCHEMA!r}, got {schema!r}")

    kind = data.get("model_kind")
    if not isinstance(kind, str) or kind not in MODEL_KINDS:
        problems.append(f"model_kind must be one of {', '.join(MODEL_KINDS)}, got {kind!r}")
        kind = None

    fv = data.get("feature_version")
    if not isinstance(fv, str) or not _FEATURE_VERSION_RE.fullmatch(fv):
        problems.append(f"feature_version must be a feature schema name such as 'ember_v2', got {fv!r}")
        fv = None
    elif check_registry:
        known = _known_feature_versions()
        if known is not None and fv not in known:
            problems.append(f"feature_version {fv!r} is not installed; known: {', '.join(sorted(known))}")

    source = data.get("threshold_source", "declared")
    if source not in THRESHOLD_SOURCES:
        problems.append(f"threshold_source must be one of {', '.join(THRESHOLD_SOURCES)}, got {source!r}")
        source = "declared"

    thr_raw = data.get("operating_threshold")
    thr: float | None = None
    if thr_raw is None:
        if source == "declared":
            problems.append("operating_threshold is required (a number in [0, 1]) unless threshold_source is 'calibrate'")
    elif isinstance(thr_raw, bool) or not isinstance(thr_raw, (int, float)):
        problems.append(f"operating_threshold must be a number in [0, 1], got {thr_raw!r}")
    elif not (math.isfinite(float(thr_raw)) and 0.0 <= float(thr_raw) <= 1.0):
        problems.append(f"operating_threshold must be in [0, 1], got {thr_raw!r}")
    else:
        thr = float(thr_raw)
    if source == "calibrate" and thr_raw is not None:
        problems.append("operating_threshold must be null when threshold_source is 'calibrate'")

    fpr_raw = data.get("calibrate_fpr")
    fpr: float | None = None
    if source == "calibrate":
        if isinstance(fpr_raw, bool) or not isinstance(fpr_raw, (int, float)) or not (
            math.isfinite(float(fpr_raw)) and 0.0 < float(fpr_raw) <= MAX_CALIBRATE_FPR
        ):
            problems.append(f"calibrate_fpr must be a number in (0, {MAX_CALIBRATE_FPR}], got {fpr_raw!r}")
        else:
            fpr = float(fpr_raw)
    elif fpr_raw is not None:
        problems.append("calibrate_fpr is only allowed when threshold_source is 'calibrate'")

    model_file = _basename(data.get("model_file"), "model_file", problems, optional=False)
    hashes_file = _basename(data.get("training_hashes_file"), "training_hashes_file", problems, optional=True)
    if model_file and hashes_file and model_file == hashes_file:
        problems.append("training_hashes_file must not be the model file")

    cutoff = data.get("training_cutoff")
    if cutoff is not None:
        if not isinstance(cutoff, str) or not _CUTOFF_RE.fullmatch(cutoff):
            problems.append(f"training_cutoff must be YYYY, YYYY-MM or YYYY-MM-DD (or null), got {cutoff!r}")
            cutoff = None
        else:
            from malvalid.manifest import parse_training_cutoff

            try:
                parse_training_cutoff(cutoff)
            except AdapterError as e:
                problems.append(str(e))
                cutoff = None

    if problems:
        raise AdapterError("invalid model spec:\n  - " + "\n  - ".join(problems))
    assert kind is not None and fv is not None and model_file is not None
    return ModelSpec(
        model_kind=kind, feature_version=fv, operating_threshold=thr, threshold_source=str(source),
        calibrate_fpr=fpr, model_file=model_file, training_hashes_file=hashes_file, training_cutoff=cutoff,
    )


def load_spec(path: str | os.PathLike[str], *, check_registry: bool = True) -> ModelSpec:
    """Read and validate a spec file (JSON only; at most 64 KiB)."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            raw = f.read(MAX_SPEC_BYTES + 1)
    except OSError as e:
        raise AdapterError(f"cannot read the model spec {p}: {e.strerror or e}") from e
    if len(raw) > MAX_SPEC_BYTES:
        raise AdapterError(f"the model spec {p.name} is larger than {MAX_SPEC_BYTES} bytes")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise AdapterError(f"the model spec {p.name} is not valid JSON: {e}") from e
    return validate_spec(data, check_registry=check_registry)


def write_spec(spec: ModelSpec | dict[str, Any], dest_dir: str | os.PathLike[str]) -> Path:
    """Validate ``spec`` and write it as ``<dest_dir>/malvalid_model.json``."""
    s = spec if isinstance(spec, ModelSpec) else validate_spec(spec)
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / SPEC_FILENAME
    path.write_text(json.dumps(s.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


class SpecDetector(BaseDetector):
    """The built-in detector behind a spec: ``BaseDetector`` with every declaration taken from the spec.

    ``predict`` is ``predict_proba >= operating_threshold``. For a calibrated threshold the host
    computes ``predict`` itself from ``predict_proba`` (the worker's copy of the threshold is a
    placeholder until calibration has run).
    """

    #: Placeholder threshold while calibration is pending (never used for a verdict).
    PENDING_THRESHOLD = 0.5
    spec_path: str = ""

    @classmethod
    def load(cls) -> "SpecDetector":
        from malvalid.loaders.base import load_model as _load

        path = Path(cls.spec_path).parent / str(cls.model_path)
        if not path.exists():
            raise AdapterError(f"model file {cls.model_path!r} named by the model spec was not found next to it")
        return cls(_load(path, str(cls.model_kind)))


def build_detector_module(spec_path: str | os.PathLike[str], module_name: str) -> ModuleType:
    """A module holding one :class:`SpecDetector` subclass configured from the spec at ``spec_path``.

    Used by the sandboxed worker in place of importing a Python adapter. Only packaged code runs; the
    spec contributes validated values.
    """
    p = Path(spec_path).resolve()
    spec = load_spec(p)
    attrs: dict[str, Any] = {
        "__module__": module_name,
        "__doc__": f"malvalid built-in detector for {spec.model_file} ({spec.model_kind}, {spec.feature_version})",
        "feature_version": spec.feature_version,
        "model_kind": spec.model_kind,
        "operating_threshold": (
            spec.operating_threshold if spec.operating_threshold is not None else SpecDetector.PENDING_THRESHOLD
        ),
        "model_path": spec.model_file,
        "training_hashes_path": spec.training_hashes_file,
        "training_cutoff": spec.training_cutoff,
        "spec_path": str(p),
    }
    cls = type("ModelFileDetector", (SpecDetector,), attrs)
    mod = ModuleType(module_name)
    mod.__file__ = str(p)
    mod.ModelFileDetector = cls  # type: ignore[attr-defined]
    return mod
