"""report.json: strict-JSON serialization, private-data scrubbing, and loading.

The report is the auditable record of a production-readiness verdict. Two invariants hold for
every report written here:

* it is strict JSON — ``json.dumps(report, allow_nan=False)`` succeeds (non-finite floats become
  ``null``);
* nothing under the run's ``private/`` directory is referenced from it (spec §11): any string
  that mentions that directory is redacted before writing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from malvalid import REPORT_SCHEMA_VERSION, SUPPORTED_REPORT_SCHEMA_VERSIONS, _platform
from malvalid.core import MalValidError, to_jsonable

log = logging.getLogger("malvalid.report")

PRIVATE_PLACEHOLDER = "<private run data withheld>"
_PATH_TAIL = r"[^\s\"'<>|,;)\]]*"


def _private_variants(private_dir: Path) -> list[str]:
    cands = {str(private_dir), os.path.abspath(private_dir)}
    try:
        cands.add(str(Path(private_dir).resolve()))
    except OSError:  # pragma: no cover
        pass
    # Longest first so a resolved path is replaced before a prefix of it.
    return sorted((c for c in cands if c and c != os.sep), key=len, reverse=True)


def scrub_private_refs(obj: Any, private_dirs: Iterable[Path | str]) -> Any:
    """Return ``obj`` with every mention of a private directory (and the path under it) redacted."""
    variants: list[str] = []
    for d in private_dirs:
        variants.extend(_private_variants(Path(d)))
    if not variants:
        return obj
    rx = re.compile("|".join(re.escape(v) + _PATH_TAIL for v in variants))

    def walk(x: Any) -> Any:
        if isinstance(x, str):
            return rx.sub(PRIVATE_PLACEHOLDER, x) if any(v in x for v in variants) else x
        if isinstance(x, dict):
            return {walk(k) if isinstance(k, str) else k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(obj)


def finalize_report(report: dict[str, Any], *, private_dir: Path | None = None) -> dict[str, Any]:
    """Make ``report`` strict-JSON-safe and scrub private references (returns a new dict)."""
    out = to_jsonable(report)
    if private_dir is not None:
        out = scrub_private_refs(out, [private_dir])
    return out


def dumps_report(report: dict[str, Any]) -> str:
    """Serialize a finalized report (raises ValueError if it still contains NaN/inf)."""
    return json.dumps(report, allow_nan=False, indent=2, ensure_ascii=False) + "\n"


def write_report_json(
    report: dict[str, Any], path: str | Path, *, private_dir: Path | None = None
) -> Path:
    """Finalize and atomically write ``report`` to ``path``; returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_report(finalize_report(report, private_dir=private_dir))
    fd, tmp = tempfile.mkstemp(prefix=".report-", suffix=".json.tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        _platform.replace_file(tmp, p)  # retries briefly on Windows while a reader holds report.json
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    log.debug("wrote %s (%d bytes)", p, len(text))
    return p


def load_report_json(path: str | Path) -> dict[str, Any]:
    """Read a report.json written by malvalid (raises MalValidError with a clear message)."""
    p = Path(path)
    if not p.exists():
        raise MalValidError(f"report file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        raise MalValidError(f"cannot read {p}: {e}") from e
    except json.JSONDecodeError as e:
        raise MalValidError(f"{p} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise MalValidError(f"{p} is not a malvalid report (top level is not an object)")
    sv = data.get("schema_version")
    # "malguard-report/" is the same schema family, written before the rename to MalValid
    if not isinstance(sv, str) or not sv.startswith(("malvalid-report/", "malguard-report/")):
        raise MalValidError(
            f"{p} is not a malvalid report (schema_version {sv!r}; expected {REPORT_SCHEMA_VERSION!r})"
        )
    if sv not in SUPPORTED_REPORT_SCHEMA_VERSIONS:
        log.warning("%s has schema_version %s; this malvalid writes %s", p, sv, REPORT_SCHEMA_VERSION)
    return data
