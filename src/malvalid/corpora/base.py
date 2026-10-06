"""Canonical evaluation corpus: on-disk format, loading/verification, and the provider interface.

On-disk format ``malvalid-corpus/1`` (a directory; corpora written before the rename to MalValid
declare the identical legacy format ``malguard-corpus/1`` and load unchanged)::

    manifest.json   identity, provenance, per-file sha256, content_hash, split/role definitions
    X.npy           float32 (n, dim) feature matrix — opened with mmap_mode="r"
    meta.npz        sha256 <U64, label int8 (1 malicious, 0 benign, -1 unlabeled),
                    timestamp datetime64[D] (NaT if unknown), split <U16

The ``content_hash`` pins everything a result depends on, so a report is reproducible: it is the
sha256 over the identity fields (format, name, version, feature_version, dim, n) and the sorted
per-file hashes. The format enters the hash in its canonical spelling (:data:`HASH_FORMAT_ID`), so
``malvalid-corpus/1`` and the legacy ``malguard-corpus/1`` give the same hash for the same data: the
pinned hashes of the canonical builds stay valid both for existing corpora and for rebuilds. No raw samples ever live here —
only feature vectors, hashes, timestamps and labels.
"""

from __future__ import annotations

import abc
import datetime as dt
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Sequence

import numpy as np

from malvalid import CORPUS_FORMAT, LEGACY_CORPUS_FORMATS, SUPPORTED_CORPUS_FORMATS
from malvalid.core import CorpusUnavailable

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.config import GateConfig

LABEL_MALICIOUS = 1
LABEL_BENIGN = 0
LABEL_UNLABELED = -1

# Standard roles. A manifest maps each role to the list of split names that make it up.
ROLE_EVAL = "eval"  # held-out labeled data for M1/M3/M5/M7 (never the provider's train split)
ROLE_TEMPORAL = "temporal"  # timestamped labeled data for M2 windows
ROLE_CHALLENGE = "challenge"  # hard malicious samples that historically slipped past detectors
ROLE_POOL = "pool"  # everything usable as attacker/background data (M4 non-members, M6 queries)

_VERIFY_SIDECAR = ".malvalid-verified.json"
#: Verification cache written before the rename to MalValid. It is still trusted when present (so the
#: 10 GB EMBER corpora are not re-hashed) and is never written.
_LEGACY_VERIFY_SIDECARS: tuple[str, ...] = (".malguard-verified.json",)

#: The format identifier that enters :func:`compute_content_hash`. Version 1 of the format was
#: introduced as ``malguard-corpus/1``; every pinned content hash was computed with that string, so
#: all spellings of version 1 hash as it.
HASH_FORMAT_ID: dict[str, str] = {CORPUS_FORMAT: "malguard-corpus/1", **{f: f for f in LEGACY_CORPUS_FORMATS}}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compute_content_hash(manifest: dict[str, Any]) -> str:
    fmt = manifest["format"]
    ident = {
        "format": HASH_FORMAT_ID.get(fmt, fmt),
        "name": manifest["name"],
        "version": manifest["version"],
        "feature_version": manifest["feature_version"],
        "dim": manifest["dim"],
        "n": manifest["n"],
        "files": dict(sorted(manifest["files"].items())),
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()


@dataclass
class Corpus:
    """A loaded canonical corpus. ``X`` is typically a read-only memmap; use :meth:`take`."""

    name: str
    version: str
    feature_version: str
    content_hash: str
    manifest: dict[str, Any]
    X: np.ndarray
    sha256: np.ndarray  # (n,) <U64 lowercase hex
    label: np.ndarray  # (n,) int8
    timestamp: np.ndarray  # (n,) datetime64[D]
    split: np.ndarray  # (n,) <U16
    path: Path | None = None
    synthetic: bool = False
    #: How the on-disk files were checked on load (see :func:`verify_corpus_dir`); None if not verified.
    verification: dict[str, Any] | None = None
    _hash_index: dict[str, int] | None = field(default=None, repr=False)

    # ---- basics ------------------------------------------------------------------------------

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])

    def has_timestamps(self) -> bool:
        return bool(np.any(~np.isnat(self.timestamp)))

    def roles(self) -> dict[str, list[str]]:
        return dict(self.manifest.get("roles", {}))

    def role_splits(self, role: str) -> list[str]:
        roles = self.roles()
        if role == ROLE_POOL and role not in roles:
            return sorted(set(self.split.tolist()))
        return list(roles.get(role, []))

    def take(self, idx: np.ndarray | Sequence[int]) -> np.ndarray:
        """Materialize rows as a contiguous float32 array (sorted access is fastest on memmaps)."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        order = np.argsort(idx, kind="stable")
        out = np.empty((idx.size, self.dim), dtype=np.float32)
        out[order] = np.asarray(self.X[idx[order]], dtype=np.float32)
        return out

    def hash_index(self) -> dict[str, int]:
        if self._hash_index is None:
            self._hash_index = {h: i for i, h in enumerate(self.sha256.tolist())}
        return self._hash_index

    # ---- selection ---------------------------------------------------------------------------

    def indices(
        self,
        *,
        role: str | None = None,
        splits: Iterable[str] | None = None,
        label: int | Iterable[int] | None = None,
        labeled_only: bool = True,
        after: dt.date | None = None,
        before: dt.date | None = None,
        require_timestamp: bool = False,
        include_hashes: frozenset[str] | set[str] | None = None,
        exclude_hashes: frozenset[str] | set[str] | None = None,
    ) -> np.ndarray:
        """Row indices matching all given filters (sorted ascending).

        ``after``/``before`` are exclusive bounds on the timestamp (rows without a timestamp are
        dropped when either bound is given or ``require_timestamp``).
        """
        m = np.ones(self.n, dtype=bool)
        if role is not None:
            m &= np.isin(self.split, np.array(self.role_splits(role), dtype=self.split.dtype))
        if splits is not None:
            m &= np.isin(self.split, np.array(list(splits), dtype=self.split.dtype))
        if label is not None:
            labels = [label] if isinstance(label, (int, np.integer)) else list(label)
            m &= np.isin(self.label, np.array(labels, dtype=self.label.dtype))
        elif labeled_only:
            m &= self.label != LABEL_UNLABELED
        if after is not None or before is not None or require_timestamp:
            m &= ~np.isnat(self.timestamp)
        if after is not None:
            m &= self.timestamp > np.datetime64(after, "D")
        if before is not None:
            m &= self.timestamp < np.datetime64(before, "D")
        idx = np.flatnonzero(m)
        if include_hashes is not None:
            keep = np.fromiter((h in include_hashes for h in self.sha256[idx]), bool, idx.size)
            idx = idx[keep]
        if exclude_hashes:
            keep = np.fromiter((h not in exclude_hashes for h in self.sha256[idx]), bool, idx.size)
            idx = idx[keep]
        return idx

    def eval_indices(self, label: int, exclude_hashes: frozenset[str] | None = None) -> np.ndarray:
        return self.indices(role=ROLE_EVAL, label=label, exclude_hashes=exclude_hashes)

    @staticmethod
    def subsample(
        idx: np.ndarray, max_n: int | None, rng: np.random.Generator
    ) -> np.ndarray:
        """Uniform subsample without replacement (sorted). ``max_n=None`` keeps all."""
        idx = np.asarray(idx, dtype=np.int64)
        if max_n is None or idx.size <= max_n:
            return idx
        return np.sort(rng.choice(idx, size=int(max_n), replace=False))

    def summary(self) -> dict[str, Any]:
        splits, counts = np.unique(self.split, return_counts=True)
        by_split = {}
        for s, c in zip(splits.tolist(), counts.tolist()):
            sel = self.split == s
            by_split[s] = {
                "n": int(c),
                "malicious": int(np.sum(self.label[sel] == LABEL_MALICIOUS)),
                "benign": int(np.sum(self.label[sel] == LABEL_BENIGN)),
                "unlabeled": int(np.sum(self.label[sel] == LABEL_UNLABELED)),
            }
        ts = self.timestamp[~np.isnat(self.timestamp)]
        return {
            "name": self.name,
            "version": self.version,
            "feature_version": self.feature_version,
            "content_hash": self.content_hash,
            "n": self.n,
            "dim": self.dim,
            "synthetic": self.synthetic,
            "splits": by_split,
            "roles": self.roles(),
            "time_range": [str(ts.min()), str(ts.max())] if ts.size else None,
            "source": self.manifest.get("source"),
        }


# --------------------------------------------------------------------------------------------------
# Reading and writing the on-disk format
# --------------------------------------------------------------------------------------------------


class CorpusWriter:
    """Incrementally write a corpus directory (streaming rows into a memmapped X.npy)."""

    def __init__(self, out_dir: Path, n: int, dim: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n, self.dim = int(n), int(dim)
        self._X = np.lib.format.open_memmap(
            self.out_dir / "X.npy", mode="w+", dtype=np.float32, shape=(self.n, self.dim)
        )

    def write_rows(self, start: int, rows: np.ndarray) -> None:
        rows = np.asarray(rows, dtype=np.float32)
        self._X[start : start + rows.shape[0]] = rows

    def close(self) -> None:
        """Release the X.npy memmap (idempotent). Windows cannot delete a file that is still mapped,
        so a failed build calls this before removing its partial output."""
        X = self.__dict__.pop("_X", None)
        if X is not None:
            del X

    def finalize(
        self,
        *,
        name: str,
        version: str,
        feature_version: str,
        sha256: Sequence[str],
        label: np.ndarray,
        timestamp: np.ndarray,
        split: Sequence[str],
        roles: dict[str, list[str]],
        description: str = "",
        source: dict[str, Any] | None = None,
        synthetic: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._X.flush()
        self.close()
        sha = np.array([s.lower() for s in sha256], dtype="<U64")
        lab = np.asarray(label, dtype=np.int8)
        ts = np.asarray(timestamp, dtype="datetime64[D]")
        spl = np.array(list(split), dtype="<U16")
        for arr, nm in ((sha, "sha256"), (lab, "label"), (ts, "timestamp"), (spl, "split")):
            if arr.shape[0] != self.n:
                raise ValueError(f"meta field {nm} has {arr.shape[0]} rows, expected {self.n}")
        write_npz_deterministic(
            self.out_dir / "meta.npz", {"sha256": sha, "label": lab, "timestamp": ts, "split": spl}
        )
        from malvalid import __version__

        manifest: dict[str, Any] = {
            "format": CORPUS_FORMAT,
            "name": name,
            "version": version,
            "feature_version": feature_version,
            "dim": self.dim,
            "n": self.n,
            "description": description,
            "source": source or {},
            "synthetic": synthetic,
            "roles": roles,
            "files": {
                "X.npy": sha256_file(self.out_dir / "X.npy"),
                "meta.npz": sha256_file(self.out_dir / "meta.npz"),
            },
            "created_by": f"malvalid {__version__}",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        if extra:
            manifest["extra"] = extra
        manifest["content_hash"] = compute_content_hash(manifest)
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        write_verify_sidecar(self.out_dir, manifest)
        return manifest


def write_corpus(
    out_dir: Path, X: np.ndarray, **finalize_kwargs: Any
) -> dict[str, Any]:
    """Write a whole in-memory corpus at once (convenience for small/synthetic corpora)."""
    w = CorpusWriter(out_dir, X.shape[0], X.shape[1])
    w.write_rows(0, X)
    return w.finalize(**finalize_kwargs)


def _windows_change_time(p: Path) -> int | None:  # pragma: no cover - Windows only
    """NTFS ChangeTime (the POSIX ctime equivalent) of ``p``, or None if it cannot be read.

    On Windows ``st_ctime`` is the creation time, which an in-place edit does not change."""
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class _FileBasicInfo(ctypes.Structure):
            _fields_ = [("CreationTime", ctypes.c_int64), ("LastAccessTime", ctypes.c_int64),
                        ("LastWriteTime", ctypes.c_int64), ("ChangeTime", ctypes.c_int64),
                        ("FileAttributes", wintypes.DWORD)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = k32.GetFileInformationByHandleEx
        fn.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        fn.restype = wintypes.BOOL
        info = _FileBasicInfo()
        with open(p, "rb") as f:
            if not fn(msvcrt.get_osfhandle(f.fileno()), 0, ctypes.byref(info), ctypes.sizeof(info)):  # FileBasicInfo
                return None
        return int(info.ChangeTime)
    except Exception:  # noqa: BLE001 - fall back to st_ctime
        return None


def _file_stamp(p: Path) -> list[int]:
    """What the verification cache compares: size, mtime, ctime and inode.

    ``os.utime`` can restore an mtime but not the ctime, and a copied or replaced file gets a new
    inode, so an in-place edit that keeps size and mtime is re-hashed instead of trusted. On Windows
    the NTFS ChangeTime stands in for the ctime (``st_ctime`` is the creation time there).
    """
    st = p.stat()
    ctime = int(st.st_ctime_ns)
    if os.name == "nt":  # pragma: no cover - Windows only
        change = _windows_change_time(p)
        ctime = change if change is not None else ctime
    return [int(st.st_size), int(st.st_mtime_ns), ctime, int(st.st_ino)]


def _utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed .npz whose bytes depend only on the arrays (fixed zip metadata), so
    corpus content hashes are reproducible across rebuilds. Never pickles."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for key, arr in arrays.items():
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asarray(arr), allow_pickle=False)


def write_verify_sidecar(d: Path, manifest: dict[str, Any], verified_at: dict[str, str] | None = None) -> None:
    try:
        now = _utc_iso()
        data = {
            fn: {"stamp": _file_stamp(d / fn), "sha256": h, "verified_at": (verified_at or {}).get(fn, now)}
            for fn, h in manifest["files"].items()
        }
        with open(d / _VERIFY_SIDECAR, "w") as f:
            json.dump(data, f)
    except OSError:
        pass


def verify_corpus_dir(d: Path, manifest: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Check every file against the manifest hash and return how it was checked.

    A file whose size, mtime, ctime and inode match the ``.malvalid-verified.json`` cache is
    trusted without re-hashing (the 10 GB EMBER corpora would otherwise be re-read on every run);
    ``force`` re-hashes everything. Returns ``{"mode": "full" | "cached" | "partial", "files":
    {name: "hashed" | "cached"}, "verified_at"}`` where ``verified_at`` is when the oldest trusted
    hash was last computed (now, for a full check).
    """
    cache: dict[str, Any] = {}
    if not force:
        for name in (_VERIFY_SIDECAR, *_LEGACY_VERIFY_SIDECARS):  # current name first
            side = d / name
            if not side.exists():
                continue
            try:
                cache = json.loads(side.read_text())
            except (OSError, ValueError):
                cache = {}
            if isinstance(cache, dict) and cache:
                break
    if compute_content_hash(manifest) != manifest.get("content_hash"):
        raise CorpusUnavailable(f"{d}: manifest content_hash does not match its own contents")
    now = _utc_iso()
    how: dict[str, str] = {}
    when: dict[str, str | None] = {}
    for fn, expected in manifest["files"].items():
        p = d / fn
        if not p.exists():
            raise CorpusUnavailable(f"{d}: missing corpus file {fn}")
        c = cache.get(fn) if isinstance(cache, dict) else None
        if isinstance(c, dict) and c.get("stamp") == _file_stamp(p) and c.get("sha256") == expected:
            how[fn] = "cached"
            when[fn] = str(c["verified_at"]) if c.get("verified_at") else None
            continue
        got = sha256_file(p)
        if got != expected:
            raise CorpusUnavailable(f"{d}/{fn}: sha256 {got} != manifest {expected}")
        how[fn], when[fn] = "hashed", now
    if any(v == "hashed" for v in how.values()):
        write_verify_sidecar(d, manifest, verified_at={k: v for k, v in when.items() if v is not None})
    modes = set(how.values())
    mode = "full" if modes <= {"hashed"} else ("cached" if modes == {"cached"} else "partial")
    stamps = [v for v in when.values() if v]
    return {"mode": mode, "files": how, "verified_at": min(stamps) if len(stamps) == len(when) else None}


_write_verify_sidecar = write_verify_sidecar  # backwards-compatible alias


def load_corpus_dir(
    d: Path,
    *,
    verify: bool = True,
    expected_content_hash: str | None = None,
    mmap: bool = True,
    force_verify: bool = False,
) -> Corpus:
    d = Path(d)
    mpath = d / "manifest.json"
    if not mpath.exists():
        raise CorpusUnavailable(f"no corpus at {d} (manifest.json missing)")
    manifest = json.loads(mpath.read_text())
    if manifest.get("format") not in SUPPORTED_CORPUS_FORMATS:
        raise CorpusUnavailable(f"{d}: unsupported corpus format {manifest.get('format')!r}")
    if expected_content_hash and manifest.get("content_hash") != expected_content_hash:
        raise CorpusUnavailable(
            f"{d}: content_hash {manifest.get('content_hash')} does not match the pinned "
            f"{expected_content_hash} for this corpus version"
        )
    verification = verify_corpus_dir(d, manifest, force=force_verify) if verify else None
    X = np.load(d / "X.npy", mmap_mode="r" if mmap else None, allow_pickle=False)
    with np.load(d / "meta.npz", allow_pickle=False) as meta:
        sha, lab = meta["sha256"], meta["label"]
        ts, spl = meta["timestamp"].astype("datetime64[D]"), meta["split"]
    if X.shape != (manifest["n"], manifest["dim"]):
        raise CorpusUnavailable(f"{d}: X.npy shape {X.shape} != manifest ({manifest['n']}, {manifest['dim']})")
    return Corpus(
        name=manifest["name"],
        version=manifest["version"],
        feature_version=manifest["feature_version"],
        content_hash=manifest["content_hash"],
        manifest=manifest,
        X=X,
        sha256=sha,
        label=lab,
        timestamp=ts,
        split=spl,
        path=d,
        synthetic=bool(manifest.get("synthetic", False)),
        verification=verification,
    )


# --------------------------------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------------------------------


def default_corpus_root() -> Path:
    env = os.environ.get("MALVALID_CORPUS_DIR")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "malvalid" / "corpora"


#: The files of a corpus directory (anything else in it means it is not just a corpus).
CORPUS_DIR_FILES = frozenset({"X.npy", "meta.npz", "manifest.json", _VERIFY_SIDECAR, *_LEGACY_VERIFY_SIDECARS})


def manifest_name(d: Path) -> str | None:
    """The ``name`` recorded in ``d/manifest.json`` (None if there is no readable manifest)."""
    try:
        name = json.loads((Path(d) / "manifest.json").read_text()).get("name")
    except (OSError, ValueError, AttributeError):
        return None
    return str(name) if name is not None else None


def resolve_corpus_dir(d: str | Path, name: str) -> Path:
    """Resolve a ``corpus_dir`` setting (gate.yaml or ``--corpus-dir``) for corpus ``name``.

    It may name the corpus directory itself or a root that contains ``<name>/`` (the same layout
    as ``$MALVALID_CORPUS_DIR``): ``d/<name>`` is used when it holds a corpus and ``d`` does not
    itself hold corpus ``name``.
    """
    d = Path(d).expanduser()
    sub = d / name
    if (sub / "manifest.json").exists() and manifest_name(d) != name:
        return sub
    return d


def full_verification_requested(config: "GateConfig | None") -> bool:
    """``runtime.corpus_verification: full`` (``malvalid run --verify-corpus``): re-hash every file."""
    rt = getattr(config, "runtime", None)
    return str(getattr(rt, "corpus_verification", "cached") or "cached") == "full"


class CorpusProvider(abc.ABC):
    """A versioned canonical corpus (registered under ``malvalid.corpora``).

    Resolution order for the data directory: ``config.corpus_dir`` (the corpus itself or a root
    containing ``<name>/``, see :func:`resolve_corpus_dir`) > ``$MALVALID_CORPUS_DIR/<name>`` >
    ``~/.cache/malvalid/corpora/<name>``.
    """

    name: ClassVar[str]  # registry key and config ``corpus:`` value, e.g. "ember_v2_2018"
    feature_version: ClassVar[str]
    version: ClassVar[str]
    description: ClassVar[str] = ""
    # sha256 content hash of the canonical build; None if not pinned (e.g. synthetic corpora).
    expected_content_hash: ClassVar[str | None] = None
    synthetic: ClassVar[bool] = False

    def locate(self, config: "GateConfig | None" = None) -> Path:
        if config is not None and getattr(config, "corpus_dir", None):
            return resolve_corpus_dir(config.corpus_dir, self.name)
        return default_corpus_root() / self.name

    def is_available(self, config: "GateConfig | None" = None) -> bool:
        return (self.locate(config) / "manifest.json").exists()

    def load(self, config: "GateConfig | None" = None, *, verify: bool = True) -> Corpus:
        d = self.locate(config)
        if not (d / "manifest.json").exists():
            raise CorpusUnavailable(self.unavailable_hint(d))
        other = manifest_name(d)
        if other is not None and other != self.name:
            raise CorpusUnavailable(
                f"{d} holds corpus {other!r}, not {self.name!r}; point corpus_dir / --corpus-dir at the "
                f"{self.name} directory or at a root that contains {self.name}/"
            )
        corpus = load_corpus_dir(d, verify=verify, expected_content_hash=self.expected_content_hash,
                                 force_verify=full_verification_requested(config))
        if corpus.feature_version != self.feature_version:
            raise CorpusUnavailable(
                f"{d}: corpus feature_version {corpus.feature_version} != provider {self.feature_version}"
            )
        return corpus

    def unavailable_hint(self, d: Path) -> str:
        return f"canonical corpus {self.name!r} not found at {d}"

    def build(self, source: Path | None, out: Path, **kwargs: Any) -> dict[str, Any]:
        """Build the canonical corpus directory from a raw source (optional per provider)."""
        raise NotImplementedError(f"corpus {self.name} cannot be built locally")

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_version": self.feature_version,
            "version": self.version,
            "description": self.description,
            "expected_content_hash": self.expected_content_hash,
            "synthetic": self.synthetic,
        }
