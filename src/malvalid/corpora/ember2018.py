"""Canonical EMBER2018 corpus (feature version 2) — provider ``ember_v2_2018``.

The corpus is built locally from the public EMBER2018 feature release (feature vectors, hashes,
labels and first-seen months only — no PE files): the seven JSONL files of
``ember_dataset_2018_2.tar.bz2`` are vectorised with :mod:`malvalid.schemas.ember_v2` and written
in the ``malvalid-corpus/1`` format.

Layout of the built corpus

* rows: ``train_features_0.jsonl`` ... ``train_features_5.jsonl`` then ``test_features.jsonl``,
  in file order (800,000 train rows incl. 200,000 unlabeled, then 200,000 test rows);
* ``label``: the dataset's ``label`` (1 malicious, 0 benign, -1 unlabeled);
* ``timestamp``: the first day of the ``appeared`` month (``"2018-03"`` -> 2018-03-01);
* ``split``: ``train`` / ``test``;
* roles: ``eval=[test]``, ``temporal=[train, test]``, ``pool=[train, test]``, ``challenge=[]``.

The build is deterministic: the same source files give byte-identical ``X.npy`` / ``meta.npz``
and therefore the same ``content_hash`` on any machine, whatever the number of worker processes.
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import hashlib
import json
import logging
import multiprocessing
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np

from malvalid.core import CorpusUnavailable, MalValidError
from malvalid.corpora.base import (
    LABEL_BENIGN,
    LABEL_MALICIOUS,
    LABEL_UNLABELED,
    ROLE_CHALLENGE,
    ROLE_EVAL,
    ROLE_POOL,
    ROLE_TEMPORAL,
    CorpusProvider,
    CorpusWriter,
    compute_content_hash,
    sha256_file,
)

log = logging.getLogger("malvalid.corpora.ember2018")

# ---- the public dataset -------------------------------------------------------------------------

DATASET_NAME = "EMBER2018 (feature version 2)"
ARCHIVE_NAME = "ember_dataset_2018_2.tar.bz2"
ARCHIVE_URL = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"
ARCHIVE_SHA256 = "b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812"
ARCHIVE_BYTES = 1_696_539_273
HOMEPAGE = "https://github.com/elastic/ember"
DATA_LICENSE = "MIT (EMBER data files; the EMBER source code is AGPL-3.0 and is not used)"
CITATION = (
    "H. S. Anderson and P. Roth. EMBER: An Open Dataset for Training Static PE Malware Machine "
    "Learning Models. arXiv:1804.04637, 2018."
)

SOURCE_FILES: tuple[tuple[str, str], ...] = (
    ("train_features_0.jsonl", "train"),
    ("train_features_1.jsonl", "train"),
    ("train_features_2.jsonl", "train"),
    ("train_features_3.jsonl", "train"),
    ("train_features_4.jsonl", "train"),
    ("train_features_5.jsonl", "train"),
    ("test_features.jsonl", "test"),
)
EXPECTED_ROWS = {"train": 800_000, "test": 200_000}
ROLES: dict[str, list[str]] = {
    ROLE_EVAL: ["test"],
    ROLE_TEMPORAL: ["train", "test"],
    ROLE_POOL: ["train", "test"],
    ROLE_CHALLENGE: [],
}
BUILD_RECIPE = "ember2018-jsonl-to-ember_v2/r1"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})")
_READ_BLOCK = 8 << 20  # bytes per read while scanning source files


class Ember2018BuildError(MalValidError):
    """The EMBER2018 source files are missing or malformed, or the build failed."""


# ---- phase 1: scan source files (sha256 + newline-aligned chunks) --------------------------------


@dataclasses.dataclass(frozen=True)
class _FileScan:
    path: str
    size: int
    sha256: str
    chunks: tuple[tuple[int, int, int], ...]  # (start byte, end byte, records)

    @property
    def rows(self) -> int:
        return sum(c[2] for c in self.chunks)


def _scan_file(path: str, chunk_bytes: int) -> _FileScan:
    """Hash a JSONL file and cut it into newline-aligned chunks of about ``chunk_bytes``."""
    h = hashlib.sha256()
    chunks: list[tuple[int, int, int]] = []
    pos = start = lines = 0
    last = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(_READ_BLOCK)
            if not block:
                break
            h.update(block)
            end = pos + len(block)
            if end - start >= chunk_bytes and (cut := block.rfind(b"\n")) >= 0:
                lines += block.count(b"\n", 0, cut + 1)
                chunks.append((start, pos + cut + 1, lines))
                start = pos + cut + 1
                lines = block.count(b"\n", cut + 1)
            else:
                lines += block.count(b"\n")
            pos = end
            last = block[-1:]
    if pos > start:
        if last != b"\n":
            lines += 1  # final record without a trailing newline
        chunks.append((start, pos, lines))
    return _FileScan(path=path, size=pos, sha256=h.hexdigest(), chunks=tuple(chunks))


# ---- phase 2: vectorise chunks straight into X.npy ----------------------------------------------


@dataclasses.dataclass(frozen=True)
class _ChunkTask:
    path: str
    start: int
    end: int
    n_rows: int
    row0: int
    x_path: str
    batch_rows: int


@dataclasses.dataclass
class _ChunkResult:
    sha256: np.ndarray  # <U64
    label: np.ndarray  # int8
    timestamp: np.ndarray  # datetime64[D]
    nan_rows: int
    bad_timestamps: int


def _month_start(appeared: Any) -> np.datetime64:
    if isinstance(appeared, str):
        m = _MONTH_RE.match(appeared)
        if m and 1 <= int(m.group(2)) <= 12:
            return np.datetime64(f"{m.group(1)}-{m.group(2)}-01", "D")
    return np.datetime64("NaT", "D")


def _vectorize_chunk(task: _ChunkTask) -> _ChunkResult:
    """Parse, validate and vectorise one chunk; write its rows into the shared X.npy memmap."""
    from malvalid.schemas.ember_v2 import vectorize_raw_batch

    name = os.path.basename(task.path)
    with open(task.path, "rb") as f:
        f.seek(task.start)
        data = f.read(task.end - task.start)
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    if len(lines) != task.n_rows:
        raise Ember2018BuildError(
            f"{name}: expected {task.n_rows} records in bytes {task.start}-{task.end}, found "
            f"{len(lines)} (was the file modified during the build?)"
        )
    n = len(lines)
    sha = np.empty(n, dtype="<U64")
    label = np.empty(n, dtype=np.int8)
    ts = np.empty(n, dtype="datetime64[D]")
    nan_rows = bad_ts = 0
    X = np.load(task.x_path, mmap_mode="r+")
    try:
        for b0 in range(0, n, task.batch_rows):
            raws: list[dict[str, Any]] = []
            for k, line in enumerate(lines[b0 : b0 + task.batch_rows]):
                i = b0 + k
                where = f"{name} record {i + 1} of bytes {task.start}-{task.end}"
                try:
                    r = json.loads(line)
                except ValueError as e:
                    raise Ember2018BuildError(f"{where}: not valid JSON ({e})") from None
                if not isinstance(r, dict):
                    raise Ember2018BuildError(f"{where}: expected a JSON object")
                s = r.get("sha256")
                if not isinstance(s, str) or not _SHA256_RE.match(s):
                    raise Ember2018BuildError(f"{where}: missing or malformed 'sha256'")
                lab = r.get("label")
                if lab not in (LABEL_MALICIOUS, LABEL_BENIGN, LABEL_UNLABELED) or isinstance(lab, bool):
                    raise Ember2018BuildError(f"{where}: 'label' must be 1, 0 or -1 (got {lab!r})")
                sha[i] = s.lower()
                label[i] = lab
                ts[i] = _month_start(r.get("appeared"))
                if np.isnat(ts[i]):
                    bad_ts += 1
                raws.append(r)
            try:
                V = vectorize_raw_batch(raws)
            except ValueError as e:
                raise Ember2018BuildError(f"{name} (bytes {task.start}-{task.end}): {e}") from None
            nan_rows += int(np.isnan(V).any(axis=1).sum())
            X[task.row0 + b0 : task.row0 + b0 + len(raws)] = V
        X.flush()
    finally:
        del X
    return _ChunkResult(sha256=sha, label=label, timestamp=ts, nan_rows=nan_rows, bad_timestamps=bad_ts)


# ---- deterministic meta.npz ---------------------------------------------------------------------


def _write_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed .npz whose bytes depend only on the arrays (fixed zip metadata).

    ``numpy.savez`` stamps the current time into every zip member, which would make the corpus
    content hash differ between otherwise identical builds.
    """
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for key, arr in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)
    os.replace(tmp, path)


def _make_meta_deterministic(out: Path, manifest: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    _write_npz_deterministic(out / "meta.npz", arrays)
    manifest = dict(manifest)
    manifest["files"] = dict(manifest["files"], **{"meta.npz": sha256_file(out / "meta.npz")})
    manifest["content_hash"] = compute_content_hash(manifest)
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    try:  # refresh the verification cache written by CorpusWriter.finalize
        from malvalid.corpora.base import _write_verify_sidecar

        _write_verify_sidecar(out, manifest)
    except ImportError:  # pragma: no cover - base.py without the helper
        (out / ".malvalid-verified.json").unlink(missing_ok=True)
    return manifest


# ---- helpers -----------------------------------------------------------------------------------


def _resolve_source(source: Path | str | None) -> Path:
    if source is None:
        raise Ember2018BuildError(
            "no --source given: pass the directory that contains the extracted EMBER2018 files "
            "(train_features_0.jsonl ... test_features.jsonl)"
        )
    src = Path(source).expanduser()
    if src.is_file() and src.name.endswith((".tar.bz2", ".tar", ".tbz2")):
        raise Ember2018BuildError(
            f"{src} is the compressed archive; extract it first (tar -xjf {src.name}) and pass "
            "the resulting ember2018/ directory as --source"
        )
    candidates = [src, src / "ember2018"]
    for c in candidates:
        if all((c / fn).is_file() for fn, _ in SOURCE_FILES):
            return c
    missing = [fn for fn, _ in SOURCE_FILES if not (src / fn).is_file()]
    raise Ember2018BuildError(
        f"{src} does not contain the EMBER2018 feature files; missing: {', '.join(missing)}. "
        f"Download {ARCHIVE_URL} and extract it (tar -xjf {ARCHIVE_NAME})."
    )


def _archive_checksum_note(src: Path) -> dict[str, Any]:
    """Cross-check a ``ember_dataset_2018_2.sha256`` file next to the source, if there is one."""
    for d in (src, src.parent, src.parent.parent):
        f = d / "ember_dataset_2018_2.sha256"
        if f.is_file():
            try:
                recorded = f.read_text().split()[0].strip().lower()
            except (OSError, IndexError):
                continue
            if recorded != ARCHIVE_SHA256:
                log.warning(
                    "%s records sha256 %s for the archive, but the published EMBER2018 archive "
                    "hash is %s — the source may not be the official release", f, recorded, ARCHIVE_SHA256,
                )
            return {"archive_sha256_recorded": recorded, "archive_sha256_matches": recorded == ARCHIVE_SHA256}
    return {}


def _prepare_out(out: Path) -> None:
    ours = {"manifest.json", "X.npy", "meta.npz", ".malvalid-verified.json", "meta.npz.tmp",
            ".malguard-verified.json"}  # the last: verification cache of a build from before the rename
    if out.exists():
        if not out.is_dir():
            raise Ember2018BuildError(f"--out {out} exists and is not a directory")
        others = sorted(p.name for p in out.iterdir() if p.name not in ours)
        if others:
            raise Ember2018BuildError(
                f"--out {out} already contains other files ({', '.join(others[:5])}); choose an "
                "empty directory (the corpus needs ~10 GB)"
            )
        # an older build (or a partial one) here is replaced; drop its manifest first so a
        # half-written directory is never mistaken for a complete corpus
        (out / "manifest.json").unlink(missing_ok=True)
        for side in (".malvalid-verified.json", ".malguard-verified.json"):
            (out / side).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)


def _default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


# ---- provider ----------------------------------------------------------------------------------


class Ember2018Provider(CorpusProvider):
    """EMBER2018 feature-version-2 corpus: 1,000,000 rows x 2381 features (build it locally)."""

    name: ClassVar[str] = "ember_v2_2018"
    feature_version: ClassVar[str] = "ember_v2"
    version: ClassVar[str] = "2018.2-r1"
    description: ClassVar[str] = (
        "EMBER2018 (feature version 2): 800k train rows (300k malicious, 300k benign, 200k "
        "unlabeled; first seen Jan-Oct 2018, except 50k benign rows first seen 2006-2017) and "
        "200k test rows (100k/100k, first seen Nov-Dec 2018), vectorised to ember_v2; "
        "eval = test, temporal/pool = train + test"
    )
    # Content hash of the canonical build (recipe r1 over the official 2018_2 release); a build
    # from unmodified source files reproduces it exactly. See docs/modules/corpus_ember2018.md.
    expected_content_hash: ClassVar[str | None] = (
        "b76ea441be1a62197c7d19801174309e988eea54740ff22f03fa75e66beb4b98"
    )

    def unavailable_hint(self, d: Path) -> str:
        return (
            f"canonical corpus {self.name!r} is not built yet (looked in {d}). Build it once from "
            f"the public EMBER2018 feature release (feature vectors only, no executables):\n"
            f"  1. download {ARCHIVE_URL} (1.7 GB; sha256 {ARCHIVE_SHA256})\n"
            f"  2. extract it: tar -xjf {ARCHIVE_NAME}   (creates ember2018/, ~11 GB of JSONL)\n"
            f"  3. malvalid corpus build {self.name} --source ember2018/ --out {d} --workers 8\n"
            f"     (~10 GB output; about a minute with 8-10 workers, longer with fewer)\n"
            f"If you build it elsewhere, point malvalid at it with MALVALID_CORPUS_DIR=<parent "
            f"directory> or `corpus_dir: <corpus directory>` in gate.yaml."
        )

    def build(
        self,
        source: Path | None,
        out: Path,
        *,
        workers: int | None = None,
        chunk_bytes: int = 64 << 20,
        batch_rows: int = 512,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the corpus from the extracted EMBER2018 JSONL files in ``source`` into ``out``.

        ``workers`` processes (default: min(8, cpus - 1); ``1`` = in-process) vectorise
        newline-aligned chunks of about ``chunk_bytes`` in parallel; the result does not depend
        on ``workers``, ``chunk_bytes`` or ``batch_rows``. Returns the written manifest.
        """
        if kwargs:
            log.warning("ignoring unsupported build options: %s", ", ".join(sorted(kwargs)))
        src = _resolve_source(source)
        out = Path(out).expanduser()
        n_workers = _default_workers() if workers is None else max(1, int(workers))
        chunk_bytes = max(1 << 16, int(chunk_bytes))
        batch_rows = max(1, int(batch_rows))
        _prepare_out(out)
        t0 = time.monotonic()
        paths = [str(src / fn) for fn, _ in SOURCE_FILES]
        log.info("building %s from %s into %s with %d worker(s)", self.name, src, out, n_workers)

        pool: cf.ProcessPoolExecutor | None = None
        if n_workers > 1:
            pool = cf.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")
            )
        writer: CorpusWriter | None = None
        try:
            # phase 1: hash + chunk every file
            if pool is not None:
                scans = list(pool.map(_scan_file, paths, [chunk_bytes] * len(paths)))
            else:
                scans = [_scan_file(p, chunk_bytes) for p in paths]
            n_total = sum(s.rows for s in scans)
            if n_total == 0:
                raise Ember2018BuildError(f"{src}: the EMBER2018 files contain no records")
            split_rows: dict[str, int] = {}
            for (fn, split), s in zip(SOURCE_FILES, scans):
                split_rows[split] = split_rows.get(split, 0) + s.rows
                log.info("  %s: %d records, %.2f GB, sha256 %s", fn, s.rows, s.size / 1e9, s.sha256)
            for split, expected in EXPECTED_ROWS.items():
                if split_rows.get(split, 0) != expected:
                    log.warning(
                        "%s split has %d records; the EMBER2018 release has %d — this will not "
                        "match the canonical corpus", split, split_rows.get(split, 0), expected,
                    )

            # phase 2: vectorise into X.npy
            from malvalid.schemas.ember_v2 import DIM

            writer = CorpusWriter(out, n_total, DIM)
            x_path = str(out / "X.npy")
            tasks: list[_ChunkTask] = []
            splits: list[str] = []
            row = 0
            for (fn, split), s in zip(SOURCE_FILES, scans):
                for start, end, k in s.chunks:
                    if k:
                        tasks.append(_ChunkTask(s.path, start, end, k, row, x_path, batch_rows))
                        row += k
                splits.extend([split] * s.rows)
            results = pool.map(_vectorize_chunk, tasks) if pool is not None else map(_vectorize_chunk, tasks)
            parts: list[_ChunkResult] = []
            done = 0
            for task, res in zip(tasks, results):
                parts.append(res)
                done += task.n_rows
                el = time.monotonic() - t0
                log.info("  vectorised %d / %d rows (%.0f%%, %.0f s)", done, n_total, 100 * done / n_total, el)
        except BaseException:
            if writer is not None:
                for fn in ("X.npy", "meta.npz"):
                    (out / fn).unlink(missing_ok=True)
            raise
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)

        sha = np.concatenate([p.sha256 for p in parts])
        label = np.concatenate([p.label for p in parts])
        ts = np.concatenate([p.timestamp for p in parts])
        nan_rows = sum(p.nan_rows for p in parts)
        bad_ts = sum(p.bad_timestamps for p in parts)
        n_unique = int(np.unique(sha).size)
        if n_unique != sha.size:
            log.warning("%d duplicate sha256 values in the source", sha.size - n_unique)
        if bad_ts:
            log.warning("%d records have no parseable 'appeared' month (timestamp NaT)", bad_ts)

        counts: dict[str, dict[str, Any]] = {}
        spl = np.asarray(splits, dtype="<U16")
        for split in ("train", "test"):
            sel = spl == split
            t_sel = ts[sel & ~np.isnat(ts)]
            counts[split] = {
                "n": int(sel.sum()),
                "malicious": int(np.sum(label[sel] == LABEL_MALICIOUS)),
                "benign": int(np.sum(label[sel] == LABEL_BENIGN)),
                "unlabeled": int(np.sum(label[sel] == LABEL_UNLABELED)),
                "first_month": str(t_sel.min())[:7] if t_sel.size else None,
                "last_month": str(t_sel.max())[:7] if t_sel.size else None,
            }
        source_meta: dict[str, Any] = {
            "dataset": DATASET_NAME,
            "homepage": HOMEPAGE,
            "download_url": ARCHIVE_URL,
            "archive": ARCHIVE_NAME,
            "archive_sha256": ARCHIVE_SHA256,
            "archive_bytes": ARCHIVE_BYTES,
            "license": DATA_LICENSE,
            "citation": CITATION,
            "files": [
                {"file": fn, "split": split, "bytes": s.size, "records": s.rows, "sha256": s.sha256}
                for (fn, split), s in zip(SOURCE_FILES, scans)
            ],
            **_archive_checksum_note(src),
        }
        extra = {
            "build_recipe": BUILD_RECIPE,
            "vectorizer": "malvalid.schemas.ember_v2.vectorize_raw_batch",
            "timestamp_rule": "first day of the record's 'appeared' month",
            "row_order": "train_features_0..5.jsonl then test_features.jsonl, file order",
            "counts": counts,
            "rows_with_nan": nan_rows,
            "rows_without_timestamp": bad_ts,
            "duplicate_sha256": int(sha.size - n_unique),
        }
        manifest = writer.finalize(
            name=self.name,
            version=self.version,
            feature_version=self.feature_version,
            sha256=sha,
            label=label,
            timestamp=ts,
            split=spl,
            roles=ROLES,
            description=self.description,
            source=source_meta,
            synthetic=False,
            extra=extra,
        )
        manifest = _make_meta_deterministic(
            out,
            manifest,
            {
                "sha256": np.array([s.lower() for s in sha], dtype="<U64"),
                "label": label.astype(np.int8),
                "timestamp": ts.astype("datetime64[D]"),
                "split": spl,
            },
        )
        el = time.monotonic() - t0
        pinned = self.expected_content_hash
        if pinned and manifest["content_hash"] != pinned:
            log.warning(
                "built %s in %.0f s, but its content hash %s differs from the canonical %s; "
                "`load` will refuse it (were the source files modified?)",
                self.name, el, manifest["content_hash"], pinned,
            )
        else:
            log.info("built %s: %d rows in %.0f s, content hash %s", self.name, n_total, el, manifest["content_hash"])
        return manifest


def build_corpus(source: Path, out: Path, *, workers: int | None = None, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper: ``Ember2018Provider().build(source, out, workers=workers)``."""
    return Ember2018Provider().build(source, out, workers=workers, **kwargs)


def load_or_hint(config: Any = None) -> Any:
    """Load the corpus, or raise CorpusUnavailable with build instructions."""
    prov = Ember2018Provider()
    if not prov.is_available(config):
        raise CorpusUnavailable(prov.unavailable_hint(prov.locate(config)))
    return prov.load(config)


__all__: Sequence[str] = (
    "ARCHIVE_SHA256",
    "ARCHIVE_URL",
    "Ember2018BuildError",
    "Ember2018Provider",
    "ROLES",
    "SOURCE_FILES",
    "build_corpus",
)
