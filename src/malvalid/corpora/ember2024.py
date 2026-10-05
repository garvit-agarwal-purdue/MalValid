"""EMBER2024 canonical corpus (``ember_v3_2024``) in EMBER feature version 3.

Built from the published EMBER2024 *feature* files (no binaries): the PE test set — Win32 and
Win64, 12 weekly files each, first seen 2024-09-22 .. 2024-12-14 — plus the PE rows (Win32, Win64,
.NET) of the challenge set (malicious files that initially evaded ~70 AV engines on VirusTotal).
Rows are vectorized with the thrember-exact ember_v3 vectorizer; timestamps are the dataset's
``first_submission_date`` (first seen on VirusTotal, UTC day); labels are 1 malicious / 0 benign.

The published weekly test files list every sample twice (the second copy only adds capa results)
and 13 benign files appear in two consecutive weeks; the build keeps the first occurrence of each
sha256 per split after checking that the dropped copies carry identical feature values.

Splits and roles: ``test`` (eval, temporal, pool) and ``challenge`` (challenge, temporal). The
EMBER2024 *train* split is deliberately not part of the canonical corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterator, Sequence

import numpy as np

from malvalid.corpora.base import (
    LABEL_UNLABELED,
    ROLE_CHALLENGE,
    ROLE_EVAL,
    ROLE_POOL,
    ROLE_TEMPORAL,
    CorpusProvider,
    CorpusWriter,
)

log = logging.getLogger("malvalid.corpora.ember2024")

HF_DATASET_URL = "https://huggingface.co/datasets/joyce8/EMBER2024"
SOURCE_ZIPS = ("Win32_test.zip", "Win64_test.zip", "challenge.zip")
DEFAULT_TEST_FILE_TYPES = ("Win32", "Win64")
DEFAULT_CHALLENGE_FILE_TYPES = ("Win32", "Win64", "Dot_Net")
SPLIT_ORDER = ("test", "challenge")

_MEMBER_RE = re.compile(
    r"^(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})_"
    r"(?:(?P<ftype>Win32|Win64|Dot_Net|APK|ELF|PDF)_(?P<subset>train|test)|(?P<challenge>challenge)_malicious)"
    r"\.jsonl$"
)


# --------------------------------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceMember:
    """One weekly EMBER2024 JSONL file, either loose on disk or inside a zip."""

    name: str  # member file name, e.g. 2024-09-22_2024-09-28_Win32_test.jsonl
    split: str  # "test" | "challenge"
    file_type: str | None  # from the file name (None for challenge files: mixed types)
    path: str  # the .jsonl file, or the .zip containing it
    in_zip: bool


def discover_members(source: Path, *, test_file_types: Sequence[str] = DEFAULT_TEST_FILE_TYPES) -> list[SourceMember]:
    """Find the test/challenge JSONL files under ``source`` (zips and/or extracted files).

    Train files and non-selected file types are ignored. A loose ``.jsonl`` wins over the same
    member inside a zip. Result is sorted: test members by name, then challenge members by name.
    """
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(f"EMBER2024 source directory not found: {source}")
    found: dict[str, SourceMember] = {}

    def consider(name: str, path: Path, in_zip: bool) -> None:
        base = name.rsplit("/", 1)[-1]
        m = _MEMBER_RE.match(base)
        if not m:
            return
        if m.group("challenge"):
            split, ftype = "challenge", None
        else:
            if m.group("subset") != "test" or m.group("ftype") not in test_file_types:
                return
            split, ftype = "test", m.group("ftype")
        if base in found and not found[base].in_zip:
            return  # loose file already registered
        found[base] = SourceMember(name if in_zip else base, split, ftype, str(path), in_zip)

    for z in sorted(source.glob("*.zip")):
        try:
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.endswith(".jsonl"):
                        consider(n, z, True)
        except zipfile.BadZipFile as e:
            raise ValueError(f"{z} is not a valid zip (incomplete download?): {e}") from e
    for p in sorted(source.rglob("*.jsonl")):
        consider(p.name, p, False)
    members = sorted(found.values(), key=lambda m: (SPLIT_ORDER.index(m.split), m.name.rsplit("/", 1)[-1]))
    return members


# --------------------------------------------------------------------------------------------------
# Workers
# --------------------------------------------------------------------------------------------------


def _iter_rows(member: SourceMember) -> Iterator[bytes]:
    """Non-blank JSONL lines of one member, streamed (zip members are never extracted)."""
    if member.in_zip:
        with zipfile.ZipFile(member.path) as zf, zf.open(member.name) as f:
            for line in f:
                if line.strip():
                    yield line
    else:
        with open(member.path, "rb") as f:
            for line in f:
                if line.strip():
                    yield line


def _selected(row: dict[str, Any], member: SourceMember, challenge_types: Sequence[str]) -> bool:
    """Row filter shared by both passes (challenge files mix file types; test files do not)."""
    if member.split == "challenge":
        return row.get("file_type") in challenge_types
    return True


_SHA_RE = re.compile(rb'"sha256"\s*:\s*"([0-9a-fA-F]{64})"')


def _scan_task(args: tuple[SourceMember, tuple[str, ...], int | None]) -> dict[str, Any]:
    """Pass 1: the sha256 of every selected row, in file order (cheap: no full JSON parse for test rows)."""
    member, challenge_types, limit = args
    hashes: list[str] = []
    for line in _iter_rows(member):
        if member.split == "challenge":
            row = json.loads(line)
            if not _selected(row, member, challenge_types):
                continue
            h = str(row.get("sha256", "")).lower()
        else:
            m = _SHA_RE.search(line)  # the top-level key comes first (md5, sha1, sha256, ...)
            h = m.group(1).decode().lower() if m else str(json.loads(line).get("sha256", "")).lower()
        if len(h) != 64:
            raise ValueError(f"{member.name}: row {len(hashes)} has no valid sha256")
        hashes.append(h)
        if limit is not None and len(hashes) >= limit:
            break
    return {"name": member.name, "sha256": hashes}


def _epoch_to_day(v: Any) -> np.datetime64:
    if v is None:
        return np.datetime64("NaT", "D")
    try:
        return np.datetime64(int(v), "s").astype("datetime64[D]")
    except (TypeError, ValueError, OverflowError):
        return np.datetime64("NaT", "D")


def _feature_digest(feats: dict[str, Any]) -> str:
    """Digest of a row's feature groups (used to prove duplicate rows carry identical features)."""
    blob = json.dumps(feats, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest()


@dataclass(frozen=True)
class _VectorizeTask:
    member: SourceMember
    offset: int  # first output row of this member's kept rows
    hashes: list[str]  # pass-1 sha256 per selected row
    keep: np.ndarray  # bool per selected row: False = later duplicate of an earlier row in the split
    digest: np.ndarray  # bool per selected row: compute a feature digest (sha256 occurs > 1x in split)
    x_path: str
    challenge_types: tuple[str, ...]
    chunk_rows: int
    limit: int | None


def _vectorize_task(task: _VectorizeTask) -> dict[str, Any]:
    """Pass 2: parse + vectorize the kept rows of one member straight into the X.npy memmap."""
    from malvalid.schemas import _thrember_port as tp

    member = task.member
    X = np.load(task.x_path, mmap_mode="r+")
    lab: list[int] = []
    ts: list[np.datetime64] = []
    ftypes: list[str] = []
    digests: dict[int, str] = {}
    buf: list[dict[str, Any]] = []
    pos = task.offset
    t0 = time.time()

    def flush() -> None:
        nonlocal pos
        if buf:
            X[pos : pos + len(buf)] = tp.vectorize_batch(buf)
            pos += len(buf)
            buf.clear()

    j = 0  # index among selected rows
    n_expected = len(task.hashes)
    for line in _iter_rows(member):
        if j >= n_expected:
            break
        row = json.loads(line)
        if not _selected(row, member, task.challenge_types):
            continue
        h = str(row.get("sha256", "")).lower()
        if h != task.hashes[j]:
            raise RuntimeError(f"{member.name}: row {j} sha256 changed between passes ({task.hashes[j]} -> {h})")
        # keep only the feature groups (drop tags/capa blobs early to bound memory)
        feats = {k: row[k] for k in tp.OFFSETS if k in row}
        if task.digest[j]:
            digests[j] = _feature_digest(feats)
        if task.keep[j]:
            ft = row.get("file_type")
            if member.file_type is not None and ft is not None and ft != member.file_type:
                raise ValueError(f"{member.name}: row {j} has file_type {ft!r}, expected {member.file_type!r}")
            label = row.get("label")
            lab.append(LABEL_UNLABELED if label is None else int(label))
            ts.append(_epoch_to_day(row.get("first_submission_date")))
            ftypes.append(str(ft))
            buf.append(feats)
            if len(buf) >= task.chunk_rows:
                flush()
        j += 1
    flush()
    X.flush()
    del X
    n_keep = int(task.keep.sum())
    if j != n_expected or len(lab) != n_keep:
        raise RuntimeError(f"{member.name}: pass 1 selected {n_expected} rows, pass 2 saw {j} ({len(lab)} kept, expected {n_keep})")
    log.info("vectorized %s: %d rows (%d duplicates skipped) in %.1fs", member.name, n_keep, j - n_keep, time.time() - t0)
    return {
        "name": member.name,
        "label": np.asarray(lab, dtype=np.int8),
        "timestamp": np.asarray(ts, dtype="datetime64[D]"),
        "file_type": ftypes,
        "digests": digests,
    }


def plan_dedup(
    members: Sequence[SourceMember], hashes: Sequence[Sequence[str]]
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int, int, int]], dict[str, int]]:
    """Keep the first occurrence of every sha256 within a split (members in canonical order).

    Returns per-member ``keep`` and ``digest`` masks, the list of dropped duplicates as
    ``(member, row, first_member, first_row)``, and dropped counts per split. The published
    EMBER2024 weekly test files list every sample twice (the second copy only adds capa results),
    so without this the test split would double-count all 480k PE files.
    """
    first: dict[str, dict[str, tuple[int, int]]] = {}
    count: dict[str, dict[str, int]] = {}
    for m, hs in zip(members, hashes):
        c = count.setdefault(m.split, {})
        for h in hs:
            c[h] = c.get(h, 0) + 1
    keeps: list[np.ndarray] = []
    digs: list[np.ndarray] = []
    dropped: list[tuple[int, int, int, int]] = []
    per_split: dict[str, int] = {}
    for mi, (m, hs) in enumerate(zip(members, hashes)):
        seen = first.setdefault(m.split, {})
        c = count[m.split]
        keep = np.ones(len(hs), dtype=bool)
        dig = np.zeros(len(hs), dtype=bool)
        for j, h in enumerate(hs):
            if c[h] > 1:
                dig[j] = True
            prev = seen.get(h)
            if prev is None:
                seen[h] = (mi, j)
            else:
                keep[j] = False
                dropped.append((mi, j, prev[0], prev[1]))
        per_split[m.split] = per_split.get(m.split, 0) + int((~keep).sum())
        keeps.append(keep)
        digs.append(dig)
    return keeps, digs, dropped, per_split


# --------------------------------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------------------------------

# Expected shape of the canonical build (README: 30k Win32 + 10k Win64 PE files per week, 12 test
# weeks; the challenge set holds 6,315 files of which 4,868 are PE).
CANONICAL_ROWS = {"test": {"Win32": 360_000, "Win64": 120_000}, "challenge": {"Dot_Net": 829, "Win32": 3225, "Win64": 814}}


class Ember2024Provider(CorpusProvider):
    """EMBER2024 PE test + challenge sets in ember_v3 (registered as ``ember_v3_2024``)."""

    name: ClassVar[str] = "ember_v3_2024"
    feature_version: ClassVar[str] = "ember_v3"
    version: ClassVar[str] = "2"
    description: ClassVar[str] = (
        "EMBER2024 PE test set (Win32+Win64, 480k files first seen 2024-09-22..2024-12-14) and the "
        "4,868 PE rows of the EMBER2024 challenge set, in EMBER feature version 3 (thrember)."
    )
    # Content hash of the canonical build (`malvalid corpus build ember_v3_2024` from the three
    # published zips, default options). See docs/modules/corpus_ember2024.md.
    expected_content_hash: ClassVar[str | None] = "4f275e15e00ee11cdc67f176bbda932b3feaa92ec6d8584420cb04714e5110cc"
    synthetic: ClassVar[bool] = False

    ROLES: ClassVar[dict[str, list[str]]] = {
        ROLE_EVAL: ["test"],
        ROLE_CHALLENGE: ["challenge"],
        ROLE_TEMPORAL: ["test", "challenge"],
        ROLE_POOL: ["test"],
    }

    @classmethod
    def _roles(cls, has_challenge: bool) -> dict[str, list[str]]:
        roles = {k: list(v) for k, v in cls.ROLES.items()}
        if not has_challenge:
            roles[ROLE_CHALLENGE] = []
            roles[ROLE_TEMPORAL] = ["test"]
        return roles

    def unavailable_hint(self, d: Path) -> str:
        return (
            f"canonical corpus {self.name!r} (EMBER2024, ember_v3 features) not found at {d}. "
            f"It is built locally from the public EMBER2024 feature files (no binaries): download "
            f"{', '.join(SOURCE_ZIPS)} from {HF_DATASET_URL} (e.g. `hf download joyce8/EMBER2024 "
            f"{' '.join(SOURCE_ZIPS)} --repo-type dataset --local-dir SRC`, ~3.8 GB), then run "
            f"`malvalid corpus build {self.name} --source SRC --out {d} --workers 8` "
            f"(~5 GB on disk, about 5 minutes with 8 workers). To keep it elsewhere, set "
            f"MALVALID_CORPUS_DIR to the parent directory of '{self.name}/' (or `corpus_dir` in the "
            f"gate config / `--corpus-dir` to the corpus directory itself)."
        )

    def build(
        self,
        source: Path | None,
        out: Path,
        *,
        workers: int = 8,
        test_file_types: Sequence[str] = DEFAULT_TEST_FILE_TYPES,
        challenge_file_types: Sequence[str] = DEFAULT_CHALLENGE_FILE_TYPES,
        chunk_rows: int = 1000,
        limit_per_member: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Vectorize the EMBER2024 test + challenge feature files into ``out``.

        ``source`` holds the downloaded zips (or their extracted ``.jsonl`` files). ``workers``
        processes parse and vectorize weekly files in parallel (zip members are streamed, never
        extracted). Rows are ordered test (by week, Win32 then Win64 per file name) then challenge;
        a sha256 seen earlier in the same split is dropped (the published test files list every
        sample twice). Non-default options or ``limit_per_member`` produce a non-canonical corpus,
        labelled as such in the manifest (its content hash will not match the pinned one).
        """
        if source is None:
            raise ValueError(self.unavailable_hint(Path(out)))
        source, out = Path(source), Path(out)
        members = discover_members(source, test_file_types=tuple(test_file_types))
        splits_found = {m.split for m in members}
        if "test" not in splits_found:
            raise FileNotFoundError(
                f"no EMBER2024 test feature files under {source} (expected {SOURCE_ZIPS[0]} / "
                f"{SOURCE_ZIPS[1]} or their *_test.jsonl members). {self.unavailable_hint(out)}"
            )
        if "challenge" not in splits_found:
            log.warning("no challenge.zip / *_challenge_malicious.jsonl under %s: building without challenge split", source)
        canonical = (
            tuple(test_file_types) == DEFAULT_TEST_FILE_TYPES
            and tuple(challenge_file_types) == DEFAULT_CHALLENGE_FILE_TYPES
            and limit_per_member is None
            and splits_found == set(SPLIT_ORDER)
            and sum(m.split == "test" for m in members) == 24
            and sum(m.split == "challenge" for m in members) == 64
        )
        workers = max(1, min(int(workers), len(members)))
        ctypes_ = tuple(challenge_file_types)
        ctx = mp.get_context("spawn")
        t0 = time.time()
        log.info("EMBER2024 build: %d source files, %d workers -> %s", len(members), workers, out)
        with ctx.Pool(workers, initializer=_worker_init) as pool:
            scans = pool.map(_scan_task, [(m, ctypes_, limit_per_member) for m in members], chunksize=1)
            hashes = [s["sha256"] for s in scans]
            keeps, digs, dropped, dropped_by_split = plan_dedup(members, hashes)
            counts = [int(k.sum()) for k in keeps]
            n = int(sum(counts))
            if n == 0:
                raise ValueError(f"no usable rows found under {source}")
            log.info("pass 1 done in %.0fs: %d rows selected, %d duplicate sha256 rows dropped %s",
                     time.time() - t0, sum(len(h) for h in hashes), len(dropped), dropped_by_split)
            from malvalid.schemas import _thrember_port as tp

            out.mkdir(parents=True, exist_ok=True)
            writer = CorpusWriter(out, n, tp.DIM)
            offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int).tolist()
            # Largest files first for load balance; results are re-ordered by member below.
            order = sorted(range(len(members)), key=lambda i: -len(hashes[i]))
            tasks = [
                _VectorizeTask(members[i], offsets[i], hashes[i], keeps[i], digs[i], str(out / "X.npy"),
                               ctypes_, int(chunk_rows), limit_per_member)
                for i in order
            ]
            results: dict[str, dict[str, Any]] = {}
            for k, r in enumerate(pool.imap_unordered(_vectorize_task, tasks, chunksize=1), 1):
                results[r["name"]] = r
                log.info("  [%d/%d] %s: %d rows (%.0fs elapsed)", k, len(tasks), r["name"], len(r["label"]), time.time() - t0)
        # Every dropped duplicate must carry the same feature groups as the row that was kept.
        conflicts = [
            (members[mi].name, hashes[mi][j])
            for mi, j, fi, fj in dropped
            if results[members[mi].name]["digests"].get(j) != results[members[fi].name]["digests"].get(fj)
        ]
        if conflicts:
            log.warning("%d duplicate sha256 rows had different feature values; kept the first occurrence (e.g. %s)",
                        len(conflicts), conflicts[:3])
        sha: list[str] = []
        lab, ts, spl, ftypes = [], [], [], []
        for m, hs, keep in zip(members, hashes, keeps):
            r = results[m.name]
            kept = [h for h, k in zip(hs, keep) if k]
            assert len(kept) == len(r["label"])
            sha.extend(kept)
            lab.append(r["label"])
            ts.append(r["timestamp"])
            spl.extend([m.split] * len(kept))
            ftypes.extend(r["file_type"])
        label = np.concatenate(lab)
        timestamp = np.concatenate(ts)
        split = np.asarray(spl)
        by_type: dict[str, dict[str, int]] = {}
        for s_name in SPLIT_ORDER:
            sel = split == s_name
            if sel.any():
                u, k = np.unique(np.asarray(ftypes)[sel], return_counts=True)
                by_type[s_name] = {str(a): int(b) for a, b in zip(u, k)}
        # Fail loudly if any sha256 survived twice (e.g. a source read twice, or a file listed in
        # both test and challenge): a corpus row must identify exactly one file.
        n_unique = len(set(sha))
        if len(sha) != n or n_unique != n:
            raise RuntimeError(
                f"EMBER2024 build produced {len(sha)} rows with {n_unique} distinct sha256 (expected {n} "
                f"unique rows); refusing to write a corpus with duplicate files. Check that {source} holds "
                f"each weekly file only once (a zip and its extracted .jsonl count once)."
            )
        cross_split = 0
        ts_ok = timestamp[~np.isnat(timestamp)]
        if canonical and by_type != CANONICAL_ROWS:
            log.warning("row counts %s differ from the published EMBER2024 counts %s", by_type, CANONICAL_ROWS)
        manifest = writer.finalize(
            name=self.name,
            version=self.version if canonical else f"{self.version}+custom",
            feature_version=self.feature_version,
            sha256=sha,
            label=label,
            timestamp=timestamp,
            split=split.tolist(),
            roles=self._roles("challenge" in splits_found),
            description=self.description,
            source={
                "dataset": "EMBER2024",
                "url": HF_DATASET_URL,
                "paper": "Joyce et al., EMBER2024 - A Benchmark Dataset for Holistic Evaluation of Malware Classifiers, KDD 2025",
                "license_note": "EMBER2024 feature files; only feature vectors, hashes, labels and timestamps are stored",
                "files": {sp: sum(m.split == sp for m in members) for sp in SPLIT_ORDER},
                "zips": sorted({Path(m.path).name for m in members if m.in_zip}),
                "timestamp_field": "first_submission_date (first seen on VirusTotal, UTC day)",
                "label_field": "label (1 malicious, 0 benign)",
                "vectorizer": "malvalid.schemas._thrember_port (thrember-exact, feature version 3)",
            },
            synthetic=False,
            extra={
                "canonical": canonical,
                "build_recipe": "ember2024-jsonl-to-ember_v3/r2 (first occurrence of each sha256 per split)",
                "test_file_types": list(test_file_types),
                "challenge_file_types": list(challenge_file_types),
                "limit_per_member": limit_per_member,
                "rows_by_split_and_file_type": by_type,
                "duplicates_dropped": dropped_by_split,
                "duplicate_feature_conflicts": len(conflicts),
                "sha256_in_both_test_and_challenge": cross_split,
                "time_range": [str(ts_ok.min()), str(ts_ok.max())] if ts_ok.size else None,
            },
        )
        log.info(
            "EMBER2024 corpus built: %d rows (%s) in %.0fs, content_hash=%s%s",
            n, by_type, time.time() - t0, manifest["content_hash"],
            "" if canonical else " (non-canonical build)",
        )
        if canonical and self.expected_content_hash and manifest["content_hash"] != self.expected_content_hash:
            log.warning(
                "content hash %s differs from the pinned %s: the source files or library versions "
                "differ from the reference build; `load()` will refuse this corpus",
                manifest["content_hash"], self.expected_content_hash,
            )
        return manifest


def _worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
