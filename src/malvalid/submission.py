"""Model-file-only submissions: from a model file (+ a few choices) to a validated model spec.

``malvalid run --model model.txt`` (no ``--adapter``) and the web UI's "Upload your model" form go
through :func:`prepare_model_submission`: the model file is inspected as data
(:func:`malvalid.inspect_model.inspect_model`), the researcher's explicit choices override what was
detected, and a data-only spec (:mod:`malvalid.adapters.spec`) is written into a fresh submission
folder next to symlinks of the model file and the optional training-hash list. The run then uses the
spec exactly like an adapter: the model is loaded only in the sandboxed worker.

Threshold calibration (``calibrate_fpr``) is implemented by the runner (:func:`calibrate_threshold`)
on a held-out slice of the canonical corpus that is removed from every evaluation role for that run
(:func:`hold_out_calibration_slice`). Which benign rows form the slice is the calibration period
policy (``runtime.calibration_period`` / ``--calibration-period``):

* ``earliest`` (default) — a *held* threshold: the earliest 10% of the eval role's dated benign rows,
  so the threshold is fit on data from before (almost) every row it is scored on, as in deployment;
* ``uniform`` — a 10% sha256 slice across the whole eval period: the threshold has seen the benign
  distribution of every period it is scored on, so FPR at that threshold is close to the target by
  construction (a *matched* threshold; optimistic under drift).
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from malvalid.adapters.spec import (
    MAX_CALIBRATE_FPR,
    MODEL_KINDS,
    SPEC_FILENAME,
    ModelSpec,
    validate_spec,
    write_spec,
)
from malvalid._platform import link_or_copy
from malvalid.core import AdapterError, MalValidError

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.corpora.base import Corpus
    from malvalid.inspect_model import ModelInfo

__all__ = [
    "CALIBRATION_MODULUS",
    "CALIBRATION_PERIODS",
    "DEFAULT_CALIBRATION_PERIOD",
    "CALIBRATION_SPLIT",
    "DEFAULT_CALIBRATE_FPR",
    "PreparedSubmission",
    "calibrate_threshold",
    "CalibrationError",
    "hold_out_calibration_slice",
    "prepare_model_submission",
]

#: Target FPR used when the researcher gives neither a threshold nor a calibration target. Half of the
#: default M1 ``max_fpr`` gate (1%), leaving room for sampling noise between calibration and test rows.
DEFAULT_CALIBRATE_FPR = 0.005
#: Split name given to the held-out calibration rows (not part of any corpus role).
CALIBRATION_SPLIT = "calibration"
#: A benign eval row is held out for calibration iff int(sha256[:8], 16) % CALIBRATION_MODULUS == 0
#: (about 10%; deterministic, seed-independent, and identical files always fall on the same side).
CALIBRATION_MODULUS = 10
MIN_CALIBRATION_ROWS = 200
#: Calibration period policies: ``earliest`` (held threshold, fit before the scored period) or
#: ``uniform`` (hash slice across the whole eval period). See the module docstring.
CALIBRATION_PERIODS = ("earliest", "uniform")
DEFAULT_CALIBRATION_PERIOD = "earliest"
#: ``earliest`` takes every benign row of the boundary timestamp (a whole day, or a whole month on a
#: month-level corpus) unless that would make the slice larger than this multiple of its target size;
#: then the boundary timestamp is split by sha256 order.
EARLIEST_BOUNDARY_SLACK = 2.0
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class PreparedSubmission:
    spec_path: Path
    spec: ModelSpec
    info: "ModelInfo"
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"spec_path": str(self.spec_path), "spec": self.spec.to_dict(), "detected": self.info.to_dict(),
                "notes": list(self.notes)}


def _link_name(src: Path, taken: set[str], fallback: str) -> str:
    name = _SAFE_NAME.sub("_", src.name).lstrip("._") or fallback
    if name.lower() == SPEC_FILENAME:
        name = "user_" + name
    base, dot, ext = name.rpartition(".")
    cand, i = name, 1
    while cand.lower() in taken:
        cand = f"{base or name}_{i}{dot}{ext}" if dot else f"{name}_{i}"
        i += 1
    taken.add(cand.lower())
    return cand[:200]


def _link(src: Path, dest: Path) -> None:
    """Symlink ``src`` into the submission folder; where symlinks are unavailable (Windows without
    Developer Mode) a hard link, else a copy. Only the content matters downstream: the spec names the
    file by basename and the worker loads it from the spec's folder; the report records its sha256."""
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        link_or_copy(src, dest)
    except OSError as e:  # pragma: no cover - no symlink, hard link or copy possible
        raise AdapterError(f"cannot link {src} into the submission folder {dest.parent}: {e.strerror or e}") from e


def prepare_model_submission(
    model: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    *,
    model_kind: str | None = None,
    feature_version: str | None = None,
    threshold: float | None = None,
    calibrate_fpr: float | None = None,
    training_cutoff: str | None = None,
    training_hashes: str | os.PathLike[str] | None = None,
    allow_pickle: bool = False,
) -> PreparedSubmission:
    """Inspect ``model``, merge the explicit choices, and write the spec into ``dest_dir``.

    Raises :class:`AdapterError` with a one-line, actionable message when something required cannot
    be determined (an unknown feature count, a pickle without ``allow_pickle``, ...).
    """
    from malvalid.inspect_model import inspect_model, supported_feature_dims

    src = Path(model).expanduser()
    if not src.is_file():
        raise AdapterError(f"--model {model}: file not found")
    src = src.resolve()
    info = inspect_model(src)
    notes: list[str] = []

    if model_kind is not None and model_kind not in MODEL_KINDS:
        raise AdapterError(f"--model-kind must be one of {', '.join(MODEL_KINDS)} (got {model_kind!r})")
    if info.is_pickle and not allow_pickle:
        raise AdapterError(
            f"{src.name} is a pickle-based model file; pickles can run arbitrary code when loaded, so malvalid "
            "refuses them unless you pass --allow-pickle (it is then loaded only inside the sandbox, after the M0 "
            "scan). Safer: re-export it as LightGBM .txt, XGBoost .json/.ubj or ONNX"
        )
    kind = model_kind or info.model_kind
    if kind is None:
        why = "; ".join(info.errors) or "the format was not recognized"
        raise AdapterError(f"could not detect the model kind of {src.name} ({why}); pass --model-kind "
                           f"({' | '.join(MODEL_KINDS)}) or submit a custom adapter with --adapter")
    if model_kind and info.model_kind and model_kind != info.model_kind and not info.is_pickle:
        notes.append(f"--model-kind {model_kind} overrides the detected kind {info.model_kind}")

    dims = supported_feature_dims()
    fv = feature_version or info.feature_version
    if feature_version is not None:
        if feature_version not in dims.values():
            raise AdapterError(f"--feature-version must be one of {', '.join(sorted(dims.values()))} "
                               f"(got {feature_version!r})")
        want = next(d for d, v in dims.items() if v == feature_version)
        if info.n_features is not None and info.n_features != want:
            raise AdapterError(f"{src.name} expects {info.n_features} features but --feature-version "
                               f"{feature_version} has {want}")
    if fv is None:
        if info.n_features is not None:
            raise AdapterError(next((e for e in info.errors if "features" in e),
                                    f"model expects {info.n_features} features, which matches no installed feature schema"))
        raise AdapterError(f"could not detect the feature version of {src.name}"
                           + (f" ({'; '.join(info.errors)})" if info.errors else "")
                           + f"; pass --feature-version ({' | '.join(sorted(dims.values()))})")
    if info.errors and not (model_kind and feature_version):
        raise AdapterError(f"{src.name}: " + "; ".join(info.errors))

    if threshold is not None and calibrate_fpr is not None:
        raise AdapterError("give either --threshold or --calibrate-fpr, not both")
    if threshold is None and calibrate_fpr is None:
        calibrate_fpr = DEFAULT_CALIBRATE_FPR
        notes.append(
            f"no operating threshold given: it will be calibrated to {DEFAULT_CALIBRATE_FPR:.1%} FPR on a held-out "
            "calibration slice of the corpus (pass --threshold T to use the cut-off you ship)"
        )
    if threshold is not None and (isinstance(threshold, bool) or not math.isfinite(float(threshold))
                                  or not 0.0 <= float(threshold) <= 1.0):
        raise AdapterError(f"--threshold must be a number in [0, 1] (got {threshold!r})")
    if calibrate_fpr is not None and not (math.isfinite(float(calibrate_fpr)) and 0.0 < float(calibrate_fpr) <= MAX_CALIBRATE_FPR):
        raise AdapterError(f"--calibrate-fpr must be in (0, {MAX_CALIBRATE_FPR}] (got {calibrate_fpr!r}; 0.01 = 1%)")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    model_name = _link_name(src, taken, "model")
    hashes_name: str | None = None
    hashes_src: Path | None = None
    if training_hashes is not None:
        hashes_src = Path(training_hashes).expanduser()
        if not hashes_src.is_file():
            raise AdapterError(f"--training-hashes {training_hashes}: file not found")
        hashes_src = hashes_src.resolve()
        hashes_name = _link_name(hashes_src, taken, "train_sha256.txt")
    doc = {
        "model_kind": kind,
        "feature_version": fv,
        "operating_threshold": float(threshold) if threshold is not None else None,
        "threshold_source": "declared" if threshold is not None else "calibrate",
        "calibrate_fpr": float(calibrate_fpr) if threshold is None and calibrate_fpr is not None else None,
        "model_file": model_name,
        "training_hashes_file": hashes_name,
        "training_cutoff": (training_cutoff or None),
    }
    spec = validate_spec(doc)  # same strict checks the worker applies
    _link(src, dest / model_name)
    if hashes_src is not None and hashes_name is not None:
        _link(hashes_src, dest / hashes_name)
    path = write_spec(spec, dest)
    return PreparedSubmission(spec_path=path, spec=spec, info=info, notes=notes)


# --------------------------------------------------------------------------------------------------
# Threshold calibration on a held-out slice
# --------------------------------------------------------------------------------------------------


def _held_out_mask(sha256: np.ndarray) -> np.ndarray:
    return np.fromiter(
        ((int(h[:8], 16) % CALIBRATION_MODULUS == 0) if len(h) >= 8 else False for h in sha256.tolist()),
        dtype=bool, count=sha256.shape[0],
    )


def _earliest_selection(corpus: "Corpus", ben: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """The ``earliest`` policy: the first ``ceil(len(ben) / CALIBRATION_MODULUS)`` dated benign rows
    in (timestamp, sha256) order, completed to the end of the boundary timestamp when that keeps the
    slice within ``EARLIEST_BOUNDARY_SLACK`` x its target size. ``None`` (with the reason) when too
    few benign rows are dated to fill the slice. Deterministic: depends only on timestamps and hashes."""
    n_target = int(math.ceil(ben.size / CALIBRATION_MODULUS)) if ben.size else 0
    ts = corpus.timestamp[ben].astype("datetime64[D]")
    dated = ben[~np.isnat(ts)]
    if n_target == 0 or dated.size < max(n_target, 1):
        return None, {"fallback_reason": (
            f"only {dated.size:,} of {ben.size:,} benign eval rows are dated, fewer than the {n_target:,} "
            "the earliest-period slice needs; used the uniform sha256 slice instead")}
    tsd = corpus.timestamp[dated].astype("datetime64[D]")
    order = np.lexsort((corpus.sha256[dated], tsd.astype(np.int64)))
    srt, tss = dated[order], tsd[order]
    boundary = tss[n_target - 1]
    through = int(np.searchsorted(tss.astype(np.int64), boundary.astype(np.int64), side="right"))
    take = through if through <= EARLIEST_BOUNDARY_SLACK * n_target else n_target
    return np.sort(srt[:take]), {"boundary_split": bool(take < through), "n_target": n_target}


def hold_out_calibration_slice(
    corpus: "Corpus",
    training_hashes: frozenset[str] | set[str] | None = None,
    period: str = DEFAULT_CALIBRATION_PERIOD,
) -> tuple["Corpus", np.ndarray, dict[str, Any]]:
    """A view of ``corpus`` in which the calibration rows belong to no role, plus their indices.

    Calibration rows are benign rows of the corpus's ``eval`` role (minus declared training members)
    chosen by ``period``:

    * ``"earliest"``: the earliest ``ceil(n / 10)`` of the ``n`` candidate rows by timestamp (ties
      broken by sha256), extended to the end of the boundary timestamp when that at most doubles the
      slice. On a day-level corpus this is the first ~10% of days; on a corpus whose timestamps are
      coarser than the slice (e.g. every row of a month dated on its first day) it is a sha256-ordered
      part of the earliest timestamp. Falls back to ``"uniform"`` when too few rows are dated
      (``meta["fallback_reason"]``).
    * ``"uniform"``: rows whose sha256 satisfies ``int(sha256[:8], 16) % 10 == 0``.

    In the returned view their ``split`` is ``"calibration"``, which no role lists, so every module
    (M1 eval, M2 temporal windows, M4/M6 pools, challenge sets) excludes them for the whole run. The
    feature matrix is shared (no copy).
    """
    from malvalid.corpora.base import ROLE_EVAL, ROLE_POOL

    if period not in CALIBRATION_PERIODS:
        raise ValueError(f"calibration period must be one of {', '.join(CALIBRATION_PERIODS)} (got {period!r})")
    eval_splits = corpus.role_splits(ROLE_EVAL)
    ben = corpus.indices(role=ROLE_EVAL, label=0, exclude_hashes=training_hashes or None)
    used, extra = period, {}
    sel = None
    if period == "earliest" and ben.size:
        sel, extra = _earliest_selection(corpus, ben)
        if sel is None:
            used = "uniform"
    if sel is None:
        sel = ben[_held_out_mask(corpus.sha256[ben])] if ben.size else ben
    # identical files must not straddle calibration and evaluation: hold out every row with a selected hash
    if sel.size:
        chosen = set(corpus.sha256[sel].tolist())
        same = np.flatnonzero(np.fromiter((h in chosen for h in corpus.sha256.tolist()), bool, corpus.n))
    else:
        same = sel
    split = corpus.split.copy()
    split[same] = CALIBRATION_SPLIT
    manifest = dict(corpus.manifest)
    roles = {k: list(v) for k, v in dict(manifest.get("roles") or {}).items()}
    if ROLE_POOL not in roles:  # the implicit pool would be "every split", calibration included
        roles[ROLE_POOL] = sorted(set(corpus.split.tolist()))
    manifest["roles"] = roles
    view = dataclasses.replace(corpus, split=split, manifest=manifest)
    excl = ", excluding declared training members" if training_hashes else ""
    if used == "earliest":
        rule = (f"earliest 1/{CALIBRATION_MODULUS} of the dated benign rows of the corpus eval role "
                f"({', '.join(eval_splits)}) by timestamp (ties by sha256){excl}")
    else:
        rule = (f"benign rows of the corpus eval role ({', '.join(eval_splits)}) with "
                f"int(sha256[:8], 16) % {CALIBRATION_MODULUS} == 0{excl}")
    meta = {
        "policy": used,
        "requested_policy": period,
        "rule": rule,
        **_period_meta(corpus, ben, sel),
        **extra,
        "eval_splits": eval_splits,
        "n_rows": int(sel.size),
        "n_rows_removed_from_corpus": int(same.size),
        "split_name": CALIBRATION_SPLIT,
        "corpus": corpus.name,
    }
    return view, sel, meta


def _period_meta(corpus: "Corpus", ben: np.ndarray, sel: np.ndarray) -> dict[str, Any]:
    """Dates of the calibration rows and how many of the remaining (scored) benign rows post-date them."""
    ts_sel = corpus.timestamp[sel].astype("datetime64[D]")
    ts_sel = ts_sel[~np.isnat(ts_sel)]
    if ts_sel.size == 0:
        return {"period": None, "scored_benign_after_period": None}
    rest = np.setdiff1d(ben, sel, assume_unique=True)
    ts_rest = corpus.timestamp[rest].astype("datetime64[D]")
    ts_rest = ts_rest[~np.isnat(ts_rest)]
    after = float(np.count_nonzero(ts_rest > ts_sel.max()) / ts_rest.size) if ts_rest.size else None
    return {"period": [str(ts_sel.min()), str(ts_sel.max())], "scored_benign_after_period": after}


class CalibrationError(MalValidError, ValueError):
    """The requested false-positive target cannot be met by any threshold in [0, 1]."""


def calibrate_threshold(benign_scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """The smallest threshold whose FPR on ``benign_scores`` is at most ``target_fpr``.

    Returns ``(threshold, achieved_fpr)`` with predict = score >= threshold.
    """
    s = np.sort(np.asarray(benign_scores, dtype=np.float64).reshape(-1))[::-1]
    n = s.size
    if n == 0:
        raise ValueError("no benign scores to calibrate on")
    if not np.isfinite(s).all():  # NaN sorts first in descending order and would become the threshold
        raise ValueError(f"{int(np.count_nonzero(~np.isfinite(s)))} of {n:,} benign calibration scores are "
                         "not finite (NaN/inf); scores must be finite probabilities in [0, 1]")
    k = int(math.floor(float(target_fpr) * n + 1e-9))  # false positives allowed
    if k >= n:
        t = 0.0
    else:
        t = float(np.nextafter(s[k], np.inf))  # strictly above the (k+1)-th largest score
    if t > 1.0:
        n_sat = int(np.count_nonzero(s >= 1.0))
        raise CalibrationError(
            f"target FPR {float(target_fpr):.3%} is not achievable: {n_sat:,} of {n:,} benign calibration samples "
            "score 1.0, so no threshold <= 1.0 keeps the false-positive rate that low; give an explicit "
            "--threshold or a higher --calibrate-fpr"
        )
    t = max(t, 0.0)
    achieved = float(np.count_nonzero(s >= t)) / n
    assert achieved <= float(target_fpr) + 1e-9, (achieved, target_fpr)  # guard: never above target
    return t, achieved
