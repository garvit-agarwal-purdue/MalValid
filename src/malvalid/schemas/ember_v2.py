"""EMBER feature version 2 (the EMBER2018 feature space): 2381 float32 features.

This schema lets malvalid evaluate detectors trained on EMBER2018-style vectors. It provides

* the pinned group layout (``V2_GROUPS``; contract §4.1 — do not change the boundaries),
* one human-readable name per feature (``general.size``, ``header.coff.timestamp``,
  ``imports.functions_hashed[17]`` ...),
* a per-feature :class:`~malvalid.schemas.base.Controllability` level with a one-line reason
  (:meth:`EmberV2Schema.controllability_reasons`) — M7 uses it to flag detectors that lean on
  features a file's author can change without changing what the program does,
* :meth:`EmberV2Schema.vectorize_raw` / :meth:`EmberV2Schema.vectorize_raw_batch`: EMBER raw-JSON
  records (one line of ``train_features_*.jsonl`` / ``test_features.jsonl``) -> vectors, and
* an optional raw-PE :meth:`EmberV2Schema.featurize` built on LIEF (the ``featurize`` extra).

Implementation notes
--------------------
The vectoriser is an independent implementation written from the dataset's documented
semantics; no code from the EMBER repository is used. Its correctness gate is the published
EMBER2018 LightGBM model: scoring the vectorised EMBER2018 test set must reproduce the model's
published quality (see ``docs/modules/corpus_ember2018.md`` for the measured numbers).

Semantics that matter for fidelity with vectors the published models were trained on:

* Hashed features use the "hashing trick" as implemented by scikit-learn's ``FeatureHasher``:
  signed 32-bit MurmurHash3 (seed 0) of the UTF-8 token, bucket ``|h| mod width``, sign ``+1``
  when ``h >= 0`` else ``-1``; weights of colliding tokens are summed in float64.
* ``histogram`` / ``byteentropy`` are normalised in float32 (an all-zero histogram gives NaN,
  which tree models treat as missing). The other groups are assembled in float64 and rounded
  to float32 once.
* ``section.entry_name_hashed`` hashes the entry-section name *character by character*, and
  ``section.entry_characteristics_hashed`` collects the flags of every section whose name equals
  the entry-section name (both are properties of the published vectors).
* ``imports.libraries_hashed`` hashes the set of lower-cased library names;
  ``imports.functions_hashed`` hashes ``"<library lower-case>:<function>"`` for every entry.
* ``datadirectories`` is positional: raw entry ``i`` fills ``2i`` (size) and ``2i+1``
  (virtual address) for the first 15 entries; names are taken from the PE specification order.

``featurize`` is approximate: EMBER2018 was extracted with LIEF 0.9, while malvalid uses a
modern LIEF (>= 0.14). Byte-level groups (histogram, byteentropy, strings) are computed from the
bytes and match; PE-structure fields are mapped to the LIEF 0.9 vocabulary through the PE/COFF
specification tables below, but parser differences (e.g. malformed headers, symbol counting,
section entropy on truncated sections) can move individual values.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, ClassVar

import numpy as np

from malvalid.schemas.base import Controllability as C
from malvalid.schemas.base import FeatureGroup, FeatureSchema

log = logging.getLogger("malvalid.schemas.ember_v2")

# --------------------------------------------------------------------------------------------------
# Pinned layout (contract §4.1) — do not change the boundaries.
# --------------------------------------------------------------------------------------------------

V2_GROUPS = (
    ("histogram", 0, 256, C.DERIVED, "Byte histogram (256 bins, normalized)"),
    ("byteentropy", 256, 512, C.DERIVED, "Byte-entropy histogram (16x16, window 2048 step 1024)"),
    ("strings", 512, 616, C.APPEND_ONLY, "Printable-string statistics"),
    ("general", 616, 626, C.APPEND_ONLY, "General file info (size, vsize, flags, counts)"),
    ("header", 626, 688, C.CONTROLLABLE, "COFF + optional header fields"),
    ("section", 688, 943, C.APPEND_ONLY, "Section info (counts + hashed name/size/entropy/vsize/props)"),
    ("imports", 943, 2223, C.APPEND_ONLY, "Imported libraries (256 hashed) + functions (1024 hashed)"),
    ("exports", 2223, 2351, C.APPEND_ONLY, "Exported functions (128 hashed)"),
    ("datadirectories", 2351, 2381, C.DERIVED, "15 data directories x (size, virtual_address)"),
)

DIM = 2381
GROUP_SLICES: dict[str, slice] = {g[0]: slice(g[1], g[2]) for g in V2_GROUPS}

# Field order inside groups (this order *is* the vector layout).
GENERAL_FIELDS: tuple[str, ...] = (
    "size", "vsize", "has_debug", "exports", "imports",
    "has_relocations", "has_resources", "has_signature", "has_tls", "symbols",
)
OPTIONAL_INT_FIELDS: tuple[str, ...] = (
    "major_image_version", "minor_image_version",
    "major_linker_version", "minor_linker_version",
    "major_operating_system_version", "minor_operating_system_version",
    "major_subsystem_version", "minor_subsystem_version",
    "sizeof_code", "sizeof_headers", "sizeof_heap_commit",
)
STRING_COUNT_FIELDS: tuple[str, ...] = ("paths", "urls", "registry", "MZ")
SECTION_COUNT_FIELDS: tuple[str, ...] = (
    "num_sections", "num_zero_size", "num_empty_name", "num_rx", "num_w",
)
# PE/COFF data-directory order (the first 15 entries of the optional header).
DATA_DIRECTORY_NAMES: tuple[str, ...] = (
    "EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE", "CERTIFICATE_TABLE",
    "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "TLS_TABLE",
    "LOAD_CONFIG_TABLE", "BOUND_IMPORT", "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER",
)

# Hashed-block widths.
_W_HEADER_ENUM = 10
_W_SECTION = 50
_W_LIBRARIES = 256
_W_FUNCTIONS = 1024
_W_EXPORTS = 128

# --------------------------------------------------------------------------------------------------
# PE/COFF vocabularies (names as they appear in EMBER2018 raw JSON, i.e. the LIEF 0.9 spelling).
# Used both by featurize() and by the per-bucket controllability analysis of the hashed header flags.
# --------------------------------------------------------------------------------------------------

# COFF file-header characteristics: name -> (bit, role). Roles: "info" = ignored by the Windows
# loader (settable/clearable without changing behaviour); "opt_in" = safe to clear, may break some
# programs when set; "structural" = changes what the image is or how it is loaded.
COFF_CHARACTERISTICS: dict[str, tuple[int, str]] = {
    "RELOCS_STRIPPED": (0x0001, "structural"),
    "EXECUTABLE_IMAGE": (0x0002, "structural"),
    "LINE_NUMS_STRIPPED": (0x0004, "info"),
    "LOCAL_SYMS_STRIPPED": (0x0008, "info"),
    "AGGRESSIVE_WS_TRIM": (0x0010, "info"),
    "LARGE_ADDRESS_AWARE": (0x0020, "opt_in"),
    "BYTES_REVERSED_LO": (0x0080, "info"),
    "CHARA_32BIT_MACHINE": (0x0100, "info"),
    "DEBUG_STRIPPED": (0x0200, "info"),
    "REMOVABLE_RUN_FROM_SWAP": (0x0400, "info"),
    "NET_RUN_FROM_SWAP": (0x0800, "info"),
    "SYSTEM": (0x1000, "structural"),
    "DLL": (0x2000, "structural"),
    "UP_SYSTEM_ONLY": (0x4000, "structural"),
    "BYTES_REVERSED_HI": (0x8000, "info"),
}

# Optional-header DllCharacteristics: name -> (bit, role) with the same role vocabulary.
DLL_CHARACTERISTICS: dict[str, tuple[int, str]] = {
    "HIGH_ENTROPY_VA": (0x0020, "opt_in"),
    "DYNAMIC_BASE": (0x0040, "opt_in"),
    "FORCE_INTEGRITY": (0x0080, "structural"),
    "NX_COMPAT": (0x0100, "opt_in"),
    "NO_ISOLATION": (0x0200, "structural"),
    "NO_SEH": (0x0400, "structural"),
    "NO_BIND": (0x0800, "info"),
    "APPCONTAINER": (0x1000, "structural"),
    "WDM_DRIVER": (0x2000, "structural"),
    "GUARD_CF": (0x4000, "opt_in"),
    "TERMINAL_SERVER_AWARE": (0x8000, "info"),
}

# Section characteristics: name -> mask. A name is listed when ``characteristics & mask != 0``;
# for the 4-bit ALIGN_* field that yields every ALIGN_* whose code shares a bit with the section's
# code, which is how the EMBER2018 raw data spells alignment.
SECTION_CHARACTERISTICS: dict[str, int] = {
    "TYPE_NO_PAD": 0x00000008,
    "CNT_CODE": 0x00000020,
    "CNT_INITIALIZED_DATA": 0x00000040,
    "CNT_UNINITIALIZED_DATA": 0x00000080,
    "LNK_OTHER": 0x00000100,
    "LNK_INFO": 0x00000200,
    "LNK_REMOVE": 0x00000800,
    "LNK_COMDAT": 0x00001000,
    "GPREL": 0x00008000,
    "MEM_16BIT": 0x00020000,
    "MEM_LOCKED": 0x00040000,
    "MEM_PRELOAD": 0x00080000,
    **{f"ALIGN_{1 << (code - 1)}BYTES": code << 20 for code in range(1, 15)},
    "LNK_NRELOC_OVFL": 0x01000000,
    "MEM_DISCARDABLE": 0x02000000,
    "MEM_NOT_CACHED": 0x04000000,
    "MEM_NOT_PAGED": 0x08000000,
    "MEM_SHARED": 0x10000000,
    "MEM_EXECUTE": 0x20000000,
    "MEM_READ": 0x40000000,
    "MEM_WRITE": 0x80000000,
}

MACHINE_TYPES: dict[int, str] = {
    0x0: "UNKNOWN", 0x1D3: "AM33", 0x8664: "AMD64", 0x1C0: "ARM", 0xAA64: "ARM64",
    0x1C4: "ARMNT", 0xEBC: "EBC", 0x14C: "I386", 0x200: "IA64", 0x9041: "M32R",
    0x266: "MIPS16", 0x366: "MIPSFPU", 0x466: "MIPSFPU16", 0x1F0: "POWERPC", 0x1F1: "POWERPCFP",
    0x166: "R4000", 0x1A2: "SH3", 0x1A3: "SH3DSP", 0x1A6: "SH4", 0x1A8: "SH5", 0x1C2: "THUMB",
    0x169: "WCEMIPSV2",
}
SUBSYSTEMS: dict[int, str] = {
    0: "UNKNOWN", 1: "NATIVE", 2: "WINDOWS_GUI", 3: "WINDOWS_CUI", 5: "OS2_CUI", 7: "POSIX_CUI",
    8: "NATIVE_WINDOWS", 9: "WINDOWS_CE_GUI", 10: "EFI_APPLICATION", 11: "EFI_BOOT_SERVICE_DRIVER",
    12: "EFI_RUNTIME_DRIVER", 13: "EFI_ROM", 14: "XBOX", 16: "WINDOWS_BOOT_APPLICATION",
}
PE_MAGIC: dict[int, str] = {0x10B: "PE32", 0x20B: "PE32_PLUS"}
_UNKNOWN_ENUM = "???"  # how LIEF 0.9 spelled values missing from its enums

# --------------------------------------------------------------------------------------------------
# Hashing trick
# --------------------------------------------------------------------------------------------------

try:  # scikit-learn is a core dependency; its MurmurHash3 binding is the reference hash.
    from sklearn.utils import murmurhash3_32 as _murmurhash3_32
except ImportError:  # pragma: no cover - sklearn is a hard dependency
    _murmurhash3_32 = None


def _murmur3_signed(data: bytes) -> int:
    """Pure-Python MurmurHash3_x86_32 (seed 0) as a signed 32-bit int (fallback only)."""
    c1, c2, h = 0xCC9E2D51, 0x1B873593, 0
    n = len(data)
    nblocks = n // 4
    for i in range(nblocks):
        k = int.from_bytes(data[4 * i : 4 * i + 4], "little")
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    tail = data[4 * nblocks :]
    k = 0
    for j in range(len(tail) - 1, -1, -1):
        k = (k << 8) | tail[j]
    if tail:
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
    h ^= n
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h - (1 << 32) if h & 0x80000000 else h


def _token_bytes(token: Any) -> bytes:
    if isinstance(token, bytes):
        return token
    s = token if isinstance(token, str) else str(token)
    try:
        return s.encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates from JSON escapes; keep them hashable
        return s.encode("utf-8", "surrogatepass")


class TokenHashCache:
    """Memoised signed MurmurHash3 of tokens (shared by every hashed block; widths differ only
    in the final ``|h| mod width``)."""

    def __init__(self, max_entries: int = 1_500_000):
        self.max_entries = int(max_entries)
        self._memo: dict[Any, int] = {}

    def _compute(self, token: Any) -> int:
        b = _token_bytes(token)
        h = _murmurhash3_32(b, seed=0) if _murmurhash3_32 is not None else _murmur3_signed(b)
        if len(self._memo) >= self.max_entries:
            self._memo.clear()
        self._memo[token] = h
        return int(h)

    def hashes(self, tokens: list[Any]) -> np.ndarray:
        memo, compute = self._memo, self._compute
        return np.fromiter(
            (memo[t] if t in memo else compute(t) for t in tokens), dtype=np.int64, count=len(tokens)
        )


_HASHES = TokenHashCache()


def hash_bucket(token: str, width: int) -> tuple[int, int]:
    """(bucket, sign) that ``token`` contributes to in a hashed block of ``width`` features."""
    h = int(_HASHES.hashes([token])[0])
    return abs(h) % width, (1 if h >= 0 else -1)


def _hashed_block(
    n_rows: int,
    width: int,
    counts: Sequence[int],
    tokens: list[Any],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Hash ``tokens`` (row ``r`` owns ``counts[r]`` consecutive tokens) into a float64
    ``(n_rows, width)`` block. ``weights=None`` means every token has weight 1."""
    out_len = n_rows * width
    if not tokens:
        return np.zeros((n_rows, width), dtype=np.float64)
    h = _HASHES.hashes(tokens)
    rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.asarray(counts, dtype=np.int64))
    flat = rows * width + np.abs(h) % width
    signs = np.where(h >= 0, 1.0, -1.0)
    w = signs if weights is None else signs * np.asarray(weights, dtype=np.float64)
    return np.bincount(flat, weights=w, minlength=out_len).reshape(n_rows, width)


# --------------------------------------------------------------------------------------------------
# Raw JSON -> vectors
# --------------------------------------------------------------------------------------------------

_REQUIRED_TOP = (
    "histogram", "byteentropy", "strings", "general", "header", "section", "imports",
    "exports", "datadirectories",
)


def _describe_problem(raw: Any) -> str | None:
    """Return a human-readable description of what is wrong with one raw record (or None)."""
    if not isinstance(raw, dict):
        return f"expected a JSON object (dict), got {type(raw).__name__}"
    for key in _REQUIRED_TOP:
        if key not in raw:
            return f"missing top-level field {key!r}"
    for key in ("histogram", "byteentropy"):
        v = raw[key]
        if not isinstance(v, (list, tuple, np.ndarray)) or len(v) != 256:
            return f"{key!r} must be a list of 256 counts"
    s = raw["strings"]
    if not isinstance(s, dict):
        return "'strings' must be an object"
    for key in ("numstrings", "avlength", "printables", "printabledist", "entropy", *STRING_COUNT_FIELDS):
        if key not in s:
            return f"missing field 'strings.{key}'"
    if len(s["printabledist"]) != 96:
        return "'strings.printabledist' must have 96 entries"
    for key in GENERAL_FIELDS:
        if key not in raw["general"]:
            return f"missing field 'general.{key}'"
    hdr = raw["header"]
    for sub, keys in (
        ("coff", ("timestamp", "machine", "characteristics")),
        ("optional", ("subsystem", "dll_characteristics", "magic", *OPTIONAL_INT_FIELDS)),
    ):
        if not isinstance(hdr, dict) or sub not in hdr:
            return f"missing field 'header.{sub}'"
        for key in keys:
            if key not in hdr[sub]:
                return f"missing field 'header.{sub}.{key}'"
    for sub, key in (("coff", "characteristics"), ("optional", "dll_characteristics")):
        if not isinstance(hdr[sub][key], (list, tuple)):
            return f"'header.{sub}.{key}' must be a list of flag names"
    sec = raw["section"]
    if not isinstance(sec, dict) or "entry" not in sec or "sections" not in sec:
        return "'section' must be an object with 'entry' and 'sections'"
    for j, one in enumerate(sec["sections"]):
        for key in ("name", "size", "entropy", "vsize", "props"):
            if key not in one:
                return f"missing field 'section.sections[{j}].{key}'"
        if not isinstance(one["props"], (list, tuple)):
            return f"'section.sections[{j}].props' must be a list of flag names"
    if not isinstance(raw["imports"], dict):
        return "'imports' must be an object mapping library name -> list of function names"
    for lib, entries in raw["imports"].items():
        if (
            not isinstance(lib, str)
            or not isinstance(entries, (list, tuple))
            or not all(isinstance(e, str) for e in entries)
        ):
            return f"'imports[{lib!r}]' must be a list of strings"
    if not isinstance(raw["exports"], list) or not all(isinstance(e, str) for e in raw["exports"]):
        return "'exports' must be a list of strings"
    for j, dd in enumerate(raw["datadirectories"]):
        if "size" not in dd or "virtual_address" not in dd:
            return f"'datadirectories[{j}]' needs 'size' and 'virtual_address'"
    return None


def _str_list(value: Any) -> list[str]:
    """A raw-record list of names; a bare string is an error (it would hash per character)."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected a list of strings, got {type(value).__name__}")
    return value if isinstance(value, list) else list(value)


def _normalized_counts(rows: list[Any]) -> np.ndarray:
    """(n, 256) float32 ``counts / counts.sum()`` per row, in float32 arithmetic."""
    counts = np.asarray(rows, dtype=np.float32)
    if counts.ndim != 2 or counts.shape[1] != 256:
        raise ValueError(f"expected 256 counts per record, got array of shape {counts.shape}")
    totals = np.empty(counts.shape[0], dtype=np.float32)
    for i in range(counts.shape[0]):  # 1-D float32 reduction per row (pairwise, as numpy does)
        totals[i] = counts[i].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return counts / totals[:, None]


def _fill_strings(raws: Sequence[dict[str, Any]], out: np.ndarray) -> None:
    n = len(raws)
    blk = np.empty((n, 104), dtype=np.float64)
    blk[:, 0] = [r["strings"]["numstrings"] for r in raws]
    blk[:, 1] = [r["strings"]["avlength"] for r in raws]
    printables = np.asarray([r["strings"]["printables"] for r in raws], dtype=np.float64)
    blk[:, 2] = printables
    dist = np.asarray([r["strings"]["printabledist"] for r in raws], dtype=np.float64)
    blk[:, 3:99] = dist / np.where(printables > 0, printables, 1.0)[:, None]
    blk[:, 99] = [r["strings"]["entropy"] for r in raws]
    for j, key in enumerate(STRING_COUNT_FIELDS):
        blk[:, 100 + j] = [r["strings"][key] for r in raws]
    out[:, GROUP_SLICES["strings"]] = blk


def _fill_general(raws: Sequence[dict[str, Any]], out: np.ndarray) -> None:
    vals = np.asarray([[r["general"][k] for k in GENERAL_FIELDS] for r in raws], dtype=np.float64)
    out[:, GROUP_SLICES["general"]] = vals


def _fill_header(raws: Sequence[dict[str, Any]], out: np.ndarray) -> None:
    n = len(raws)
    blk = np.empty((n, 62), dtype=np.float64)
    blk[:, 0] = [r["header"]["coff"]["timestamp"] for r in raws]
    ones = [1] * n
    col = 1
    for sub, key, many in (
        ("coff", "machine", False),
        ("coff", "characteristics", True),
        ("optional", "subsystem", False),
        ("optional", "dll_characteristics", True),
        ("optional", "magic", False),
    ):
        if many:
            lists = [_str_list(r["header"][sub][key]) for r in raws]
            tokens = [t for lst in lists for t in lst]
            counts: Sequence[int] = [len(lst) for lst in lists]
        else:
            tokens = [r["header"][sub][key] for r in raws]
            counts = ones
        blk[:, col : col + _W_HEADER_ENUM] = _hashed_block(n, _W_HEADER_ENUM, counts, tokens)
        col += _W_HEADER_ENUM
    blk[:, 51:62] = np.asarray(
        [[r["header"]["optional"][k] for k in OPTIONAL_INT_FIELDS] for r in raws], dtype=np.float64
    )
    out[:, GROUP_SLICES["header"]] = blk


def _fill_section(raws: Sequence[dict[str, Any]], out: np.ndarray) -> None:
    n = len(raws)
    blk = np.empty((n, 255), dtype=np.float64)
    names: list[str] = []
    sizes: list[float] = []
    entropies: list[float] = []
    vsizes: list[float] = []
    per_row: list[int] = []
    entry_chars: list[str] = []
    entry_counts: list[int] = []
    entry_props: list[str] = []
    entry_prop_counts: list[int] = []
    for i, r in enumerate(raws):
        sec = r["section"]
        sections = sec["sections"]
        entry = sec["entry"]
        n_zero = n_empty = n_rx = n_w = 0
        props_before = len(entry_props)
        for s in sections:
            props = _str_list(s["props"])
            name = s["name"]
            if s["size"] == 0:
                n_zero += 1
            if name == "":
                n_empty += 1
            if "MEM_READ" in props and "MEM_EXECUTE" in props:
                n_rx += 1
            if "MEM_WRITE" in props:
                n_w += 1
            if name == entry:
                entry_props.extend(props)
            names.append(name)
            sizes.append(s["size"])
            entropies.append(s["entropy"])
            vsizes.append(s["vsize"])
        blk[i, 0:5] = (len(sections), n_zero, n_empty, n_rx, n_w)
        per_row.append(len(sections))
        chars = list(entry)
        entry_chars.extend(chars)
        entry_counts.append(len(chars))
        entry_prop_counts.append(len(entry_props) - props_before)
    blk[:, 5:55] = _hashed_block(n, _W_SECTION, per_row, names, sizes)
    blk[:, 55:105] = _hashed_block(n, _W_SECTION, per_row, names, entropies)
    blk[:, 105:155] = _hashed_block(n, _W_SECTION, per_row, names, vsizes)
    blk[:, 155:205] = _hashed_block(n, _W_SECTION, entry_counts, entry_chars)
    blk[:, 205:255] = _hashed_block(n, _W_SECTION, entry_prop_counts, entry_props)
    out[:, GROUP_SLICES["section"]] = blk


def _fill_imports(raws: Sequence[dict[str, Any]], out: np.ndarray) -> None:
    n = len(raws)
    libs: list[str] = []
    lib_counts: list[int] = []
    funcs: list[str] = []
    func_counts: list[int] = []
    for r in raws:
        imports = r["imports"]
        unique = {lib.lower() for lib in imports}
        libs.extend(unique)
        lib_counts.append(len(unique))
        before = len(funcs)
        for lib, entries in imports.items():
            prefix = lib.lower() + ":"
            funcs.extend([prefix + e for e in _str_list(entries)])
        func_counts.append(len(funcs) - before)
    sl = GROUP_SLICES["imports"]
    out[:, sl.start : sl.start + _W_LIBRARIES] = _hashed_block(n, _W_LIBRARIES, lib_counts, libs)
    out[:, sl.start + _W_LIBRARIES : sl.stop] = _hashed_block(n, _W_FUNCTIONS, func_counts, funcs)


def _fill_exports(raws: Sequence[dict[str, Any]], out: np.ndarray) -> None:
    n = len(raws)
    lists = [_str_list(r["exports"]) for r in raws]
    tokens = [t for lst in lists for t in lst]
    out[:, GROUP_SLICES["exports"]] = _hashed_block(n, _W_EXPORTS, [len(x) for x in lists], tokens)


def _fill_datadirectories(raws: Sequence[dict[str, Any]], out: np.ndarray) -> None:
    n = len(raws)
    blk = np.zeros((n, 2 * len(DATA_DIRECTORY_NAMES)), dtype=np.float64)
    k = len(DATA_DIRECTORY_NAMES)
    for i, r in enumerate(raws):
        for j, dd in enumerate(r["datadirectories"][:k]):
            blk[i, 2 * j] = dd["size"]
            blk[i, 2 * j + 1] = dd["virtual_address"]
    out[:, GROUP_SLICES["datadirectories"]] = blk


def vectorize_raw_batch(raws: Sequence[dict[str, Any]]) -> np.ndarray:
    """EMBER v2 raw records -> ``(n, 2381)`` float32 matrix.

    Raises ``ValueError`` naming the first malformed record and field.
    """
    raws = list(raws)
    n = len(raws)
    out = np.empty((n, DIM), dtype=np.float32)
    if n == 0:
        return out
    try:
        out[:, GROUP_SLICES["histogram"]] = _normalized_counts([r["histogram"] for r in raws])
        out[:, GROUP_SLICES["byteentropy"]] = _normalized_counts([r["byteentropy"] for r in raws])
        _fill_strings(raws, out)
        _fill_general(raws, out)
        _fill_header(raws, out)
        _fill_section(raws, out)
        _fill_imports(raws, out)
        _fill_exports(raws, out)
        _fill_datadirectories(raws, out)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as e:
        for i, r in enumerate(raws):
            problem = _describe_problem(r)
            if problem is not None:
                ident = r.get("sha256", "?") if isinstance(r, dict) else "?"
                raise ValueError(
                    f"ember_v2: raw record {i} (sha256={ident}) is not a valid EMBER v2 feature "
                    f"record: {problem}"
                ) from e
        raise ValueError(f"ember_v2: could not vectorize raw records: {type(e).__name__}: {e}") from e
    return out


def vectorize_raw(raw: dict[str, Any]) -> np.ndarray:
    """One EMBER v2 raw record -> ``(2381,)`` float32 vector."""
    return vectorize_raw_batch([raw])[0]


def iter_vectorized(
    raws: Iterable[dict[str, Any]], batch_size: int = 1024
) -> Iterator[np.ndarray]:
    """Vectorise a stream of raw records in batches of ``batch_size`` rows."""
    buf: list[dict[str, Any]] = []
    for r in raws:
        buf.append(r)
        if len(buf) >= batch_size:
            yield vectorize_raw_batch(buf)
            buf = []
    if buf:
        yield vectorize_raw_batch(buf)


# --------------------------------------------------------------------------------------------------
# Feature names and controllability
# --------------------------------------------------------------------------------------------------

_ROLE_LEVEL = {"info": C.CONTROLLABLE, "opt_in": C.DERIVED, "structural": C.FIXED}


def _flag_bucket_levels(flags: dict[str, tuple[int, str]], label: str) -> list[tuple[C, str]]:
    """Controllability of each of the 10 buckets a hashed flag list lands in.

    A bucket moves only through the flags that hash into it: all freely settable ("info") ⇒
    CONTROLLABLE; only structural flags (or no flag at all) ⇒ FIXED; otherwise the bucket can be
    nudged but not set freely ⇒ DERIVED.
    """
    members: list[list[str]] = [[] for _ in range(_W_HEADER_ENUM)]
    for name in flags:
        bucket, _sign = hash_bucket(name, _W_HEADER_ENUM)
        members[bucket].append(name)
    out: list[tuple[C, str]] = []
    for names in members:
        roles = {flags[n][1] for n in names}
        if not names:
            out.append((C.FIXED, f"no known {label} flag hashes here; constant in practice"))
            continue
        listed = ", ".join(sorted(names))
        if roles == {"info"}:
            level = C.CONTROLLABLE
            why = "only informational flags the loader ignores"
        elif roles == {"structural"}:
            level = C.FIXED
            why = "only flags that change what the image is or how it loads"
        else:
            level = C.DERIVED
            why = "mixes freely settable and loader-relevant flags; can be nudged, not set freely"
        out.append((level, f"{why} ({listed})"))
    return out


def _build_feature_table() -> list[tuple[str, C, str]]:
    t: list[tuple[str, C, str]] = []

    # histogram / byteentropy -------------------------------------------------------------------
    why = "normalised whole-file distribution; moves only as a side effect of changing file bytes"
    t += [(f"histogram[0x{b:02X}]", C.DERIVED, why) for b in range(256)]
    t += [
        (f"byteentropy.bin{h:02d}[0x{nib:X}_]", C.DERIVED, why)
        for h in range(16)
        for nib in range(16)
    ]

    # strings ------------------------------------------------------------------------------------
    appendable = "count over file bytes; appended data (overlay, new section) can only add to it"
    ratio = "ratio/distribution over all strings; moves only as a side effect of added strings"
    t.append(("strings.numstrings", C.APPEND_ONLY, appendable))
    t.append(("strings.avlength", C.DERIVED, ratio))
    t.append(("strings.printables", C.APPEND_ONLY, appendable))
    t += [(f"strings.printabledist[0x{0x20 + k:02X}]", C.DERIVED, ratio) for k in range(96)]
    t.append(("strings.entropy", C.DERIVED, ratio))
    t += [(f"strings.{k}", C.APPEND_ONLY, appendable) for k in STRING_COUNT_FIELDS]

    # general ------------------------------------------------------------------------------------
    general = {
        "size": (C.APPEND_ONLY, "file size grows with appended data; cannot shrink below the code"),
        "vsize": (C.APPEND_ONLY, "image size grows when sections are added or enlarged"),
        "has_debug": (C.CONTROLLABLE, "debug directory is not needed to run; strip or add freely"),
        "exports": (C.APPEND_ONLY, "exports can be added; removing used ones breaks callers"),
        "imports": (C.APPEND_ONLY, "imports can be added (unused); used ones cannot be removed"),
        "has_relocations": (C.FIXED, "relocations determine whether the image can be rebased"),
        "has_resources": (C.APPEND_ONLY, "a resource directory can be added; existing ones may be used"),
        "has_signature": (C.CONTROLLABLE, "Authenticode blob lives outside the image; add/strip freely"),
        "has_tls": (C.FIXED, "TLS callbacks run code before the entry point"),
        "symbols": (C.CONTROLLABLE, "COFF symbol table is ignored by the loader"),
    }
    t += [(f"general.{k}", *general[k]) for k in GENERAL_FIELDS]

    # header -------------------------------------------------------------------------------------
    t.append(("header.coff.timestamp", C.CONTROLLABLE, "link timestamp is informational; any value works"))
    fixed_enum = {
        "coff.machine": "target CPU; changing it changes whether/where the program runs",
        "optional.subsystem": "GUI/console/driver/EFI subsystem changes how the program starts",
        "optional.magic": "PE32 vs PE32+ is the image format itself",
    }
    for field, flags, label in (
        ("coff.machine", None, ""),
        ("coff.characteristics", COFF_CHARACTERISTICS, "COFF characteristics"),
        ("optional.subsystem", None, ""),
        ("optional.dll_characteristics", DLL_CHARACTERISTICS, "DllCharacteristics"),
        ("optional.magic", None, ""),
    ):
        if flags is None:
            t += [(f"header.{field}_hashed[{k}]", C.FIXED, fixed_enum[field]) for k in range(_W_HEADER_ENUM)]
        else:
            for k, (level, reason) in enumerate(_flag_bucket_levels(flags, label)):
                t.append((f"header.{field}_hashed[{k}]", level, reason))
    version = "version stamp the Windows loader does not act on"
    optional = {
        "major_image_version": (C.CONTROLLABLE, version),
        "minor_image_version": (C.CONTROLLABLE, version),
        "major_linker_version": (C.CONTROLLABLE, version),
        "minor_linker_version": (C.CONTROLLABLE, version),
        "major_operating_system_version": (C.CONTROLLABLE, version),
        "minor_operating_system_version": (C.CONTROLLABLE, version),
        "major_subsystem_version": (
            C.FIXED, "checked by the loader against the running OS and selects compatibility behaviour"
        ),
        "minor_subsystem_version": (
            C.FIXED, "checked by the loader against the running OS and selects compatibility behaviour"
        ),
        "sizeof_code": (C.CONTROLLABLE, "informational sum of code-section sizes; not used by the loader"),
        "sizeof_headers": (C.APPEND_ONLY, "must cover the real headers; grows when section headers are added"),
        "sizeof_heap_commit": (C.CONTROLLABLE, "a resource hint; any sane value preserves behaviour"),
    }
    t += [(f"header.optional.{k}", *optional[k]) for k in OPTIONAL_INT_FIELDS]

    # section ------------------------------------------------------------------------------------
    section_counts = {
        "num_sections": (C.APPEND_ONLY, "sections can be added, not removed"),
        "num_zero_size": (C.APPEND_ONLY, "empty sections can be added"),
        "num_empty_name": (C.CONTROLLABLE, "section names are free text; any count up to num_sections"),
        "num_rx": (C.APPEND_ONLY, "executable sections can be added; the code section must stay RX"),
        "num_w": (C.APPEND_ONLY, "writable sections can be added"),
    }
    t += [(f"section.{k}", *section_counts[k]) for k in SECTION_COUNT_FIELDS]
    by_name = "bucketed by section name: adding (or renaming) sections adds to buckets; values come from real content"
    for field in ("sizes_hashed", "entropy_hashed", "vsize_hashed"):
        t += [(f"section.{field}[{k}]", C.APPEND_ONLY, by_name) for k in range(_W_SECTION)]
    t += [
        (f"section.entry_name_hashed[{k}]", C.CONTROLLABLE, "entry section can be renamed freely")
        for k in range(_W_SECTION)
    ]
    t += [
        (
            f"section.entry_characteristics_hashed[{k}]",
            C.APPEND_ONLY,
            "flags can be added to the entry section (and to same-named sections); it must stay executable",
        )
        for k in range(_W_SECTION)
    ]

    # imports / exports --------------------------------------------------------------------------
    t += [
        (f"imports.libraries_hashed[{k}]", C.APPEND_ONLY, "libraries can be added to the import table, not removed")
        for k in range(_W_LIBRARIES)
    ]
    t += [
        (f"imports.functions_hashed[{k}]", C.APPEND_ONLY, "unused imports can be added; used ones must stay")
        for k in range(_W_FUNCTIONS)
    ]
    t += [
        (f"exports.functions_hashed[{k}]", C.APPEND_ONLY, "export entries can be added")
        for k in range(_W_EXPORTS)
    ]

    # data directories ---------------------------------------------------------------------------
    grows = {"EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "IAT", "DELAY_IMPORT_DESCRIPTOR"}
    free = {"CERTIFICATE_TABLE", "DEBUG", "ARCHITECTURE", "GLOBAL_PTR", "BOUND_IMPORT"}
    for d in DATA_DIRECTORY_NAMES:
        if d in grows:
            t.append((f"datadirectories.{d}.size", C.APPEND_ONLY, "table grows when entries are added"))
            t.append((
                f"datadirectories.{d}.virtual_address", C.DERIVED,
                "moves as a side effect of rebuilding the table; not chosen directly",
            ))
        elif d in free:
            reason = "not used to run the image (signature blob, debug data, reserved/IA-64-only, stale bind cache)"
            t.append((f"datadirectories.{d}.size", C.CONTROLLABLE, reason))
            t.append((f"datadirectories.{d}.virtual_address", C.CONTROLLABLE, reason))
        else:
            reason = "unwind data, relocations, TLS, load config and the .NET header affect execution"
            t.append((f"datadirectories.{d}.size", C.FIXED, reason))
            t.append((f"datadirectories.{d}.virtual_address", C.FIXED, reason))

    if len(t) != DIM:  # pragma: no cover - guarded by tests
        raise AssertionError(f"ember_v2 feature table has {len(t)} entries, expected {DIM}")
    return t


_TABLE: list[tuple[str, C, str]] | None = None
_NAME_INDEX: dict[str, int] | None = None


def feature_table() -> list[tuple[str, C, str]]:
    """``[(name, controllability, reason)]`` for all 2381 features, in vector order."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _build_feature_table()
    return _TABLE


# --------------------------------------------------------------------------------------------------
# Raw PE bytes -> EMBER v2 raw record (optional; needs LIEF)
# --------------------------------------------------------------------------------------------------

_WINDOW = 2048
_STEP = 1024
_RE_STRINGS = re.compile(rb"[\x20-\x7f]{5,}")
_RE_PATHS = re.compile(rb"c:\\", re.IGNORECASE)
_RE_URLS = re.compile(rb"https?://", re.IGNORECASE)
_RE_REGISTRY = re.compile(rb"HKEY_")
_RE_MZ = re.compile(rb"MZ")


def lief_available() -> bool:
    try:
        import lief  # noqa: F401
    except Exception:
        return False
    return True


def _block_entropy_bin_exact(counts: np.ndarray) -> int:
    """Entropy bin of one window from its 16 nibble counts, in float32 like the dataset."""
    p = counts[counts > 0].astype(np.float32) / np.float32(_WINDOW)
    h = np.sum(-p * np.log2(p)) * np.float32(2)
    return min(int(h * np.float32(2)), 15)


def byte_entropy_histogram(data: bytes) -> np.ndarray:
    """(256,) int64 raw byte/entropy histogram (entropy bin x high nibble), window 2048, step 1024."""
    a = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros((16, 16), dtype=np.int64)
    if a.size < _WINDOW:
        c = np.bincount(a >> 4, minlength=16)
        out[_block_entropy_bin_exact(c)] += c
        return out.reshape(-1)
    n_windows = (a.size - _WINDOW) // _STEP + 1
    n_chunks = n_windows + 1  # window k = chunk k + chunk k+1 (chunks of _STEP bytes)
    chunk_counts = np.empty((n_chunks, 16), dtype=np.int64)
    slab = 4096  # chunks per slab keeps the temporary small for large files
    for s in range(0, n_chunks, slab):
        e = min(n_chunks, s + slab)
        nib = (a[s * _STEP : e * _STEP] >> 4).reshape(e - s, _STEP).astype(np.int64)
        flat = nib + (np.arange(e - s, dtype=np.int64) * 16)[:, None]
        chunk_counts[s:e] = np.bincount(flat.reshape(-1), minlength=(e - s) * 16).reshape(e - s, 16)
    win = chunk_counts[:-1] + chunk_counts[1:]
    p = win.astype(np.float32) / np.float32(_WINDOW)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(win > 0, -p * np.log2(np.where(win > 0, p, 1)), 0).astype(np.float64)
    h2 = terms.sum(axis=1) * 4.0  # (entropy * 2) * 2
    bins = np.minimum(np.floor(h2).astype(np.int64), 15)
    # Windows whose value sits next to a bin edge are recomputed exactly in float32.
    near = np.flatnonzero(np.abs(h2 - np.round(h2)) < 1e-3)
    for k in near:
        bins[k] = _block_entropy_bin_exact(win[k])
    np.add.at(out, bins, win)
    return out.reshape(-1)


def string_features(data: bytes) -> dict[str, Any]:
    """The ``strings`` raw object for a byte string."""
    found = _RE_STRINGS.findall(data)
    if found:
        joined = b"".join(found)
        c = np.bincount(np.frombuffer(joined, dtype=np.uint8).astype(np.int64) - 0x20, minlength=96)
        total = int(c.sum())
        p = c[c > 0].astype(np.float32) / np.float32(total)
        entropy = float(np.sum(-p * np.log2(p)))
        avlength = len(joined) / len(found)
    else:
        c = np.zeros(96, dtype=np.int64)
        total, entropy, avlength = 0, 0.0, 0
    return {
        "numstrings": len(found),
        "avlength": avlength,
        "printabledist": c.tolist(),
        "printables": total,
        "entropy": entropy,
        "paths": len(_RE_PATHS.findall(data)),
        "urls": len(_RE_URLS.findall(data)),
        "registry": len(_RE_REGISTRY.findall(data)),
        "MZ": len(_RE_MZ.findall(data)),
    }


def _flag_names(value: int, table: dict[str, tuple[int, str]]) -> list[str]:
    return [name for name, (bit, _role) in table.items() if value & bit]


def _section_props(value: int) -> list[str]:
    return [name for name, mask in SECTION_CHARACTERISTICS.items() if value & mask]


def _empty_structure(size: int) -> dict[str, Any]:
    return {
        "general": {
            "size": size, "vsize": 0, "has_debug": 0, "exports": 0, "imports": 0,
            "has_relocations": 0, "has_resources": 0, "has_signature": 0, "has_tls": 0, "symbols": 0,
        },
        "header": {
            "coff": {"timestamp": 0, "machine": "", "characteristics": []},
            "optional": {
                "subsystem": "", "dll_characteristics": [], "magic": "",
                **{k: 0 for k in OPTIONAL_INT_FIELDS},
            },
        },
        "section": {"entry": "", "sections": []},
        "imports": {},
        "exports": [],
        "datadirectories": [],
    }


def _parse_pe(data: bytes) -> Any:
    import lief

    try:
        lief.logging.disable()
    except Exception:  # pragma: no cover - older LIEF
        pass
    try:
        return lief.PE.parse(io.BytesIO(data))
    except Exception as e:  # malformed input: byte-level features still apply
        log.debug("LIEF could not parse input: %s", e)
        return None


def _entry_section_name(binary: Any, sections: list[Any]) -> str:
    # EMBER2018 was extracted with LIEF 0.9, which looked the entry point's *virtual address* up
    # as a *file offset* and fell back to the first executable section; the data reflects that
    # (e.g. UPX-packed files report UPX0). Reproduce it so vectors match the training data.
    opt = binary.optional_header
    va = int(opt.imagebase) + int(opt.addressof_entrypoint)
    for s in sections:
        start = int(s.pointerto_raw_data)
        if start <= va < start + int(s.sizeof_raw_data):
            return str(s.name)
    for s in sections:
        if int(s.characteristics) & SECTION_CHARACTERISTICS["MEM_EXECUTE"]:
            return str(s.name)
    return ""


def _pe_structure(data: bytes) -> dict[str, Any]:
    raw = _empty_structure(len(data))
    binary = _parse_pe(data)
    if binary is None:
        return raw
    hdr, opt = binary.header, binary.optional_header
    sections = list(binary.sections)
    exports = [str(getattr(f, "name", f))[:10000] for f in binary.exported_functions]
    has_sig = getattr(binary, "has_signatures", None)
    if has_sig is None:  # pragma: no cover - LIEF < 0.11
        has_sig = getattr(binary, "has_signature", False)
    raw["general"] = {
        "size": len(data),
        "vsize": int(binary.virtual_size),
        "has_debug": int(bool(binary.has_debug)),
        "exports": len(exports),
        "imports": len(list(binary.imported_functions)),
        "has_relocations": int(bool(binary.has_relocations)),
        "has_resources": int(bool(binary.has_resources)),
        "has_signature": int(bool(has_sig)),
        "has_tls": int(bool(binary.has_tls)),
        "symbols": len(list(binary.symbols)),
    }
    raw["header"] = {
        "coff": {
            "timestamp": int(hdr.time_date_stamps),
            "machine": MACHINE_TYPES.get(int(hdr.machine), _UNKNOWN_ENUM),
            "characteristics": _flag_names(int(hdr.characteristics), COFF_CHARACTERISTICS),
        },
        "optional": {
            "subsystem": SUBSYSTEMS.get(int(opt.subsystem), _UNKNOWN_ENUM),
            "dll_characteristics": _flag_names(int(opt.dll_characteristics), DLL_CHARACTERISTICS),
            "magic": PE_MAGIC.get(int(opt.magic), _UNKNOWN_ENUM),
            **{k: int(getattr(opt, k)) for k in OPTIONAL_INT_FIELDS},
        },
    }
    raw["section"] = {
        "entry": _entry_section_name(binary, sections),
        "sections": [
            {
                "name": str(s.name),
                "size": int(s.size),
                "entropy": float(s.entropy),
                "vsize": int(s.virtual_size),
                "props": _section_props(int(s.characteristics)),
            }
            for s in sections
        ],
    }
    imports: dict[str, list[str]] = {}
    for lib in binary.imports:
        entries = imports.setdefault(str(lib.name), [])
        for e in lib.entries:
            if e.is_ordinal:
                entries.append(f"ordinal{int(e.ordinal)}")
            else:
                entries.append(str(e.name)[:10000])
    raw["imports"] = imports
    raw["exports"] = exports
    raw["datadirectories"] = [
        {"name": DATA_DIRECTORY_NAMES[i], "size": int(dd.size), "virtual_address": int(dd.rva)}
        for i, dd in enumerate(list(binary.data_directories)[: len(DATA_DIRECTORY_NAMES)])
    ]
    return raw


def raw_features(data: bytes) -> dict[str, Any]:
    """Raw PE bytes -> an EMBER v2 raw record (the JSON structure of the dataset). Needs LIEF."""
    import hashlib

    if not lief_available():
        from malvalid.core import FeaturizeUnavailable

        raise FeaturizeUnavailable(
            "ember_v2.featurize needs LIEF: install the [featurize] extra (from the MalValid source folder: "
            "pip install -e '.[featurize]'; or: pip install lief)"
        )
    data = bytes(data)
    raw: dict[str, Any] = {"sha256": hashlib.sha256(data).hexdigest()}
    raw["histogram"] = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).tolist()
    raw["byteentropy"] = byte_entropy_histogram(data).tolist()
    raw["strings"] = string_features(data)
    raw.update(_pe_structure(data))
    return raw


# --------------------------------------------------------------------------------------------------
# The schema plugin
# --------------------------------------------------------------------------------------------------


class EmberV2Schema(FeatureSchema):
    """EMBER feature version 2 (EMBER2018), 2381 float32 features."""

    name: ClassVar[str] = "ember_v2"
    dim: ClassVar[int] = DIM
    description: ClassVar[str] = (
        "EMBER feature version 2 (EMBER2018, Anderson & Roth 2018): byte/entropy histograms, "
        "string statistics, PE header, section, import/export and data-directory features"
    )

    def groups(self) -> list[FeatureGroup]:
        return [FeatureGroup(n, a, b, c, d) for n, a, b, c, d in V2_GROUPS]

    # ---- names / controllability ------------------------------------------------------------

    def feature_names(self) -> list[str]:
        return [n for n, _c, _r in feature_table()]

    def feature_controllability(self) -> list[C]:
        return [c for _n, c, _r in feature_table()]

    def controllability_reasons(self) -> list[str]:
        """One short justification per feature for its controllability level."""
        return [r for _n, _c, r in feature_table()]

    def feature_index(self, name: str) -> int:
        """Index of a feature by name (e.g. ``"header.coff.timestamp"`` -> 626)."""
        global _NAME_INDEX
        if _NAME_INDEX is None:
            _NAME_INDEX = {n: i for i, (n, _c, _r) in enumerate(feature_table())}
        try:
            return _NAME_INDEX[name]
        except KeyError:
            raise KeyError(f"ember_v2 has no feature named {name!r}") from None

    def feature_info(self, index: int) -> dict[str, Any]:
        """Name, group, controllability and its justification for one feature index."""
        if not 0 <= index < DIM:
            raise IndexError(f"feature index {index} outside ember_v2 (dim={DIM})")
        name, level, reason = feature_table()[index]
        return {
            "index": index,
            "name": name,
            "group": self.group_of(index).name,
            "controllability": level.value,
            "reason": reason,
        }

    # ---- raw JSON -> vectors ---------------------------------------------------------------

    def vectorize_raw(self, raw: dict[str, Any]) -> np.ndarray:
        return vectorize_raw(raw)

    def vectorize_raw_batch(self, raws: Sequence[dict[str, Any]]) -> np.ndarray:
        return vectorize_raw_batch(raws)

    # ---- raw bytes -> vectors (optional) ----------------------------------------------------

    def featurize_available(self) -> bool:
        return lief_available()

    def raw_features(self, data: bytes) -> dict[str, Any]:
        """Raw PE bytes -> EMBER v2 raw record (approximate; see the module docstring)."""
        return raw_features(data)

    def featurize(self, raw: bytes) -> np.ndarray:
        """Raw PE bytes -> ``(2381,)`` float32 vector (approximate; LIEF >= 0.14 vs 0.9)."""
        return vectorize_raw(raw_features(raw))

    def info(self) -> dict[str, Any]:
        d = super().info()
        levels = self.feature_controllability()
        d["controllability_counts"] = {lv.value: int(sum(1 for x in levels if x is lv)) for lv in C}
        d["featurize_note"] = (
            "featurize() uses modern LIEF; EMBER2018 was extracted with LIEF 0.9, so PE-structure "
            "features can differ slightly from the dataset's"
        )
        return d


def _selfcheck_layout() -> None:  # pragma: no cover - import-time guard
    pos = 0
    for _n, a, b, _c, _d in V2_GROUPS:
        if a != pos:
            raise AssertionError("ember_v2 V2_GROUPS not contiguous")
        pos = b
    if pos != DIM:
        raise AssertionError("ember_v2 V2_GROUPS do not cover DIM")


_selfcheck_layout()
