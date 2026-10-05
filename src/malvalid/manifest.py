"""Training-manifest parsing: the adapter's ``training_hashes_path`` and ``training_cutoff``.

These two declarations tell malvalid which canonical-corpus samples the researcher's model was
trained on (so they are excluded from evaluation and can serve as members for the
membership-inference test) and when its training data ends (so drift windows only ever look at
the future).

* ``training_hashes_path`` — a text file with one sha256 hex digest per line (case-insensitive;
  blank lines and ``#`` comments ignored), or a CSV/TSV file with a ``sha256`` column. A ``.gz``
  suffix is decompressed transparently.
* ``training_cutoff`` — ``YYYY-MM-DD``, ``YYYY-MM`` (last day of that month) or ``YYYY``
  (December 31). An ISO datetime is accepted and truncated to its date.

Everything here is plain data parsing done in the harness process; nothing is executed.
"""

from __future__ import annotations

import calendar
import csv
import datetime as dt
import gzip
import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from malvalid.core import AdapterError

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.context import ModelDeclarations
    from malvalid.corpora.base import Corpus

log = logging.getLogger("malvalid.manifest")

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_HASH_COLUMNS = ("sha256", "sha-256", "sha_256", "hash", "sha256_hash")
_MAX_BAD_LINES_SHOWN = 3


# --------------------------------------------------------------------------------------------------
# training_cutoff
# --------------------------------------------------------------------------------------------------


def parse_training_cutoff(value: str | dt.date | None) -> dt.date | None:
    """Parse a declared ``training_cutoff`` into a date (``None`` stays ``None``).

    Accepts ``YYYY-MM-DD``, ``YYYY-MM`` (=> last day of the month), ``YYYY`` (=> Dec 31) and ISO
    datetimes (``2018-10-31T12:00:00`` => 2018-10-31). Raises :class:`AdapterError` otherwise.
    """
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if not isinstance(value, str):
        raise AdapterError(
            f"training_cutoff must be a string like '2018-10-31', '2018-10' or '2018' (or None); "
            f"got {type(value).__name__} {value!r}"
        )
    s = value.strip()
    if not s:
        return None
    try:
        if re.fullmatch(r"\d{4}", s):
            return dt.date(int(s), 12, 31)
        m = re.fullmatch(r"(\d{4})-(\d{1,2})", s)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            return dt.date(y, mo, calendar.monthrange(y, mo)[1])
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return dt.date.fromisoformat(s)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ].*", s):
            return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError as e:
        raise AdapterError(f"training_cutoff {value!r} is not a valid date: {e}") from e
    raise AdapterError(
        f"training_cutoff {value!r} is not recognised; use 'YYYY-MM-DD', 'YYYY-MM' (last day of "
        "that month) or 'YYYY' (Dec 31) — the date of the newest sample in your training set"
    )


# --------------------------------------------------------------------------------------------------
# training_hashes_path
# --------------------------------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rt", encoding="utf-8-sig", errors="strict") as f:
                return f.read()
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise AdapterError(
            f"training manifest {path} is not UTF-8 text ({e}); it must list one sha256 per line "
            "or be a CSV/TSV with a 'sha256' column"
        ) from e
    except (OSError, EOFError) as e:
        raise AdapterError(f"cannot read training manifest {path}: {e}") from e


def _looks_tabular(first_line: str) -> str | None:
    """Return the delimiter if ``first_line`` is a CSV/TSV header naming a hash column."""
    for delim in ("\t", ",", ";"):
        if delim in first_line:
            cols = [c.strip().strip('"').strip("'").lower() for c in first_line.split(delim)]
            if any(c in _HASH_COLUMNS for c in cols):
                return delim
    return None


def _parse_tabular(text: str, delim: str, path: Path) -> set[str]:
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    header = [c.strip().lower() for c in next(reader)]
    col = next(i for i, c in enumerate(header) if c in _HASH_COLUMNS)
    out: set[str] = set()
    bad: list[str] = []
    for lineno, row in enumerate(reader, start=2):
        if not row or all(not c.strip() for c in row):
            continue
        if row[0].lstrip().startswith("#"):
            continue
        cell = row[col].strip() if col < len(row) else ""
        if not _SHA256_RE.match(cell):
            bad.append(f"line {lineno}: {cell[:80]!r}")
            continue
        out.add(cell.lower())
    if bad:
        raise AdapterError(
            f"training manifest {path}: {len(bad)} row(s) in column {header[col]!r} are not sha256 "
            f"hex digests (first: {'; '.join(bad[:_MAX_BAD_LINES_SHOWN])})"
        )
    return out


def _parse_lines(text: str, path: Path) -> set[str]:
    out: set[str] = set()
    bad: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if not _SHA256_RE.match(line):
            # Headerless CSV/TSV whose first column is the hash: accept the first field.
            first = re.split(r"[\t,; ]", line, maxsplit=1)[0].strip().strip('"')
            if first != line and _SHA256_RE.match(first):
                out.add(first.lower())
                continue
            bad.append(f"line {lineno}: {line[:80]!r}")
            continue
        out.add(line.lower())
    if bad:
        raise AdapterError(
            f"training manifest {path}: {len(bad)} line(s) are not sha256 hex digests (first: "
            f"{'; '.join(bad[:_MAX_BAD_LINES_SHOWN])}). Expected one 64-character sha256 per line, "
            "or a CSV/TSV file with a 'sha256' header column"
        )
    return out


def parse_training_hashes(path: str | Path) -> frozenset[str]:
    """Read a training manifest into a frozenset of lowercase sha256 hex digests.

    Raises :class:`AdapterError` (naming the offending line) for unreadable files or malformed
    entries. An empty manifest returns an empty set; callers decide how to report that.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise AdapterError(
            f"training manifest not found: {p} (training_hashes_path is resolved relative to the "
            "adapter file's directory)"
        )
    if p.is_dir():
        raise AdapterError(f"training_hashes_path {p} is a directory; point it at a file of sha256s")
    text = _read_text(p)
    first = next((ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")), "")
    delim = _looks_tabular(first)
    hashes = _parse_tabular(text, delim, p) if delim else _parse_lines(text, p)
    log.debug("parsed %d training hashes from %s", len(hashes), p)
    return frozenset(hashes)


# --------------------------------------------------------------------------------------------------
# The combined manifest recorded in the report
# --------------------------------------------------------------------------------------------------


@dataclass
class TrainingManifest:
    """The training manifest + cutoff as the runner uses and reports them."""

    path: str | None
    hashes: frozenset[str] | None
    cutoff: str | None  # as declared
    cutoff_parsed: dt.date | None
    warnings: list[str] = field(default_factory=list)

    @property
    def n_hashes(self) -> int:
        return len(self.hashes) if self.hashes else 0

    def usable_hashes(self) -> frozenset[str] | None:
        """The hash set if non-empty, else None (an empty manifest satisfies nothing)."""
        return self.hashes if self.hashes else None

    def n_in_corpus(self, corpus: "Corpus | None") -> int | None:
        """How many manifest hashes occur anywhere in ``corpus`` (None if either is missing)."""
        if corpus is None or not self.hashes:
            return None
        idx = corpus.hash_index()
        return int(sum(1 for h in self.hashes if h in idx))

    def to_report(self, corpus: "Corpus | None") -> dict[str, Any]:
        return {
            "path": self.path,
            "declared": self.path is not None,
            "n_hashes": self.n_hashes if self.hashes is not None else None,
            "n_in_corpus": self.n_in_corpus(corpus),
            "cutoff": self.cutoff,
            "cutoff_parsed": self.cutoff_parsed.isoformat() if self.cutoff_parsed else None,
            "warnings": list(self.warnings),
        }


def load_training_manifest(decl: "ModelDeclarations") -> TrainingManifest:
    """Parse the adapter's ``training_hashes_path`` and ``training_cutoff`` declarations."""
    warnings: list[str] = []
    hashes: frozenset[str] | None = None
    path = decl.training_hashes_path
    if path:
        hashes = parse_training_hashes(path)
        if not hashes:
            warnings.append(
                f"training manifest {path} contains no sha256 hashes; training-member exclusion "
                "and membership inference are unavailable"
            )
    cutoff = parse_training_cutoff(decl.training_cutoff)
    return TrainingManifest(
        path=str(path) if path else None,
        hashes=hashes,
        cutoff=decl.training_cutoff,
        cutoff_parsed=cutoff,
        warnings=warnings,
    )


def count_excluded_members(corpus: "Corpus | None", hashes: frozenset[str] | None) -> int | None:
    """Number of evaluation-role corpus rows that are training members (excluded from eval)."""
    if corpus is None or not hashes:
        return None if corpus is None else 0
    from malvalid.corpora.base import ROLE_EVAL

    return int(corpus.indices(role=ROLE_EVAL, include_hashes=hashes).size)
