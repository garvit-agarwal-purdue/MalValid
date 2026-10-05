# This file contains code ported from thrember, the reference implementation of EMBER feature
# version 3 (EMBER2024): https://github.com/FutureComputing4AI/EMBER2024
# (src/thrember/features.py and src/thrember/pefile_warnings.txt at commit
# 0ef753e81d98bf209f71b03cd331dfc190b5b54d).
#
# Copyright (c) the EMBER2024 / thrember authors (https://github.com/FutureComputing4AI/EMBER2024)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications for malvalid (also Apache-2.0):
#   * Restructured from per-group classes into module-level functions, and added a *batched*
#     vectorizer (`vectorize_batch`) that runs each FeatureHasher once per batch. The produced
#     vectors are identical to thrember's `PEFeatureExtractor.process_raw_features`, including its
#     float32/float64 casting order and its quirks (see QUIRKS below).
#   * Raw-bytes extraction (`raw_features`): pefile is imported lazily and is optional; signify is
#     optional (without it, Authenticode features are exact for unsigned files and approximate for
#     signed ones); empty inputs no longer raise; string-regex counting scans all strings at once
#     (same counts, much faster); pefile-warning normalisation is deterministic (longest matching
#     suffix/prefix wins; thrember iterates a Python set, whose order varies between runs); unknown
#     pefile warnings are logged instead of printed; non-UTF-8 DLL/function names are decoded
#     with errors="ignore" instead of raising.
#   * Rows with a zero-sum byte histogram produce zeros instead of NaN.
#   * The byte/entropy histogram (`_raw_byteentropy`), the printable-character statistics of the
#     strings group (`_printable_profile`) and the section counts (`_section_general`) are malvalid's
#     own code rather than ported (the byte/entropy histogram is shared with the ember_v2 schema);
#     their outputs are identical to thrember's.
#
# Provenance note: thrember's features.py descends from the feature extractor of the original EMBER
# repository (https://github.com/elastic/ember, source code under AGPL-3.0), whose module docstring
# and several helper implementations it keeps. malvalid uses thrember under the Apache-2.0 license
# its authors publish it with, and replaced the spans that were identical to EMBER's code with its
# own implementations (see NOTICE).
"""EMBER feature version 3 ("thrember") vectorizer and optional raw-PE extractor.

QUIRKS reproduced on purpose (models trained on EMBER2024 vectors depend on them):

* ``exports[0]`` is ``128`` whenever the file has any export, else ``0`` (thrember stores the
  length of the hashed vector, not the number of exports).
* the 16th data directory (``RESERVED``) is never written, so its two features are always 0.
* the ``email_addr`` string regex is identical to ``mac_addr``.
* section / export / rich-header hashers use signed hashing (``alternate_sign=True``), import
  hashers do not.
"""

from __future__ import annotations

import functools
import hashlib
import io
import logging
import math
import re
from typing import Any, Sequence

import numpy as np

log = logging.getLogger("malvalid.schemas.ember_v3")

# --------------------------------------------------------------------------------------------------
# Layout constants (thrember feature order)
# --------------------------------------------------------------------------------------------------

# Regexes run over every printable string (thrember StringExtractor._regexes).
STRING_REGEXES: dict[str, tuple[str, int]] = {
    # IOC strings
    "url": ("\\b(?:http|https|ftp):\\/\\/[a-zA-Z0-9-._~:?#[\\]@!$&'()*+,;=]+", 0),
    "ipv4_addr": (
        "\\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}"
        "(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\b",
        0,
    ),
    "ipv6_addr": (
        "\\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\\b|\\b(?:[A-Fa-f0-9]{1,4}:){1,7}:\\b"
        "|\\b:[A-Fa-f0-9]{1,4}(?::[A-Fa-f0-9]{1,4}){1,6}\\b",
        0,
    ),
    "mac_addr": ("\\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\\b", 0),
    "email_addr": ("\\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\\b", 0),  # sic (upstream)
    "btc_wallet": ("[13][a-km-zA-HJ-NP-Z1-9]{25,34}", 0),
    # Windows strings
    "file_path": ("\\bC:/", 0),
    "dos_msg": ("!This program ", 0),
    "registry_key": ("\\b(?:KHEY_|KHLM|HKCU)", 0),
    # Linux strings
    "/dev/": ("/dev/", 0),
    "/proc/": ("/proc/", 0),
    "/bin/": ("/bin/", 0),
    "/usr/": ("/usr/", 0),
    "/tmp/": ("/tmp/", 0),
    # PDF strings
    "/URI": ("/URI", 0),
    "/FlateDecode": ("/FlateDecode", 0),
    "/EmbeddedFile": ("/EmbeddedFile", 0),
    # HTML and JS strings
    "html": ("html", re.IGNORECASE),
    "javascript": ("javascript", re.IGNORECASE),
    "<script": ("<script", re.IGNORECASE),
    ".click(": (".click", re.IGNORECASE),
    "onlick": ("onclick", re.IGNORECASE),
    # Powershell strings
    "powershell": ("powershell", re.IGNORECASE),
    "Invoke-Expression": ("Invoke-Expression", 0),
    "Invoke-Command": ("Invoke-Command", 0),
    "Start-process": ("Start-process", 0),
    # Network strings
    "get": ("GET /", re.IGNORECASE),
    "post": ("POST /", re.IGNORECASE),
    "http": ("HTTP/", re.IGNORECASE),
    "http://": ("http://", re.IGNORECASE),
    "https://": ("https://", re.IGNORECASE),
    "ftp": ("ftp:", re.IGNORECASE),
    "useragent": ("User-Agent", re.IGNORECASE),
    "cookie": ("cookie", re.IGNORECASE),
    "internet": ("internet", re.IGNORECASE),
    "download": ("download", re.IGNORECASE),
    "connect": ("connect", re.IGNORECASE),
    # Cryptography and encoding strings
    "base64": ("base64", re.IGNORECASE),
    "base64string": ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", 0),
    "crypt": ("crypt", 0),
    "encode": ("encode", re.IGNORECASE),
    "decode": ("decode", re.IGNORECASE),
    # Miscellaneous strings
    "cache": ("cache", re.IGNORECASE),
    "certificate": ("certificate", re.IGNORECASE),
    "clipboard": ("clipboard", re.IGNORECASE),
    "command": ("command", re.IGNORECASE),
    "create": ("create", re.IGNORECASE),
    "debug": ("debug", re.IGNORECASE),
    "delete": ("delete", re.IGNORECASE),
    "desktop": ("desktop", re.IGNORECASE),
    "directory": ("directory", re.IGNORECASE),
    "disk": ("disk", re.IGNORECASE),
    "environment": ("environment", re.IGNORECASE),
    "enum": ("enum", re.IGNORECASE),
    "exit": ("exit", re.IGNORECASE),
    "file": ("file", re.IGNORECASE),
    "hostname": ("hostname", re.IGNORECASE),
    "install": ("install", re.IGNORECASE),
    "hidden": ("hidden", re.IGNORECASE),
    "keyboard": ("keyboard", re.IGNORECASE),
    "memory": ("memory", re.IGNORECASE),
    "module": ("module", re.IGNORECASE),
    "mutex": ("mutex", re.IGNORECASE),
    "password": ("password", re.IGNORECASE),
    "privilege": ("privilege", re.IGNORECASE),
    "process": ("process", re.IGNORECASE),
    "remote": ("remote", re.IGNORECASE),
    "resource": ("resource", re.IGNORECASE),
    "security": ("security", re.IGNORECASE),
    "service": ("service", re.IGNORECASE),
    "shell": ("shell", re.IGNORECASE),
    "snapshot": ("snapshot", re.IGNORECASE),
    "system": ("system", re.IGNORECASE),
    "thread": ("thread", re.IGNORECASE),
    "token": ("token", re.IGNORECASE),
    "wallet": ("wallet", re.IGNORECASE),
    "window": ("window", re.IGNORECASE),
}
# Vector position of each regex count = rank of its name in sorted order (as upstream).
REGEX_NAMES: tuple[str, ...] = tuple(sorted(STRING_REGEXES))
REGEX_INDEX: dict[str, int] = {k: i for i, k in enumerate(REGEX_NAMES)}

MACHINE_TYPES: tuple[str, ...] = (
    "IMAGE_FILE_MACHINE_UNKNOWN", "IMAGE_FILE_MACHINE_I386", "IMAGE_FILE_MACHINE_R3000",
    "IMAGE_FILE_MACHINE_R4000", "IMAGE_FILE_MACHINE_R10000", "IMAGE_FILE_MACHINE_WCEMIPSV2",
    "IMAGE_FILE_MACHINE_ALPHA", "IMAGE_FILE_MACHINE_SH3", "IMAGE_FILE_MACHINE_SH3DSP",
    "IMAGE_FILE_MACHINE_SH3E", "IMAGE_FILE_MACHINE_SH4", "IMAGE_FILE_MACHINE_SH5",
    "IMAGE_FILE_MACHINE_ARM", "IMAGE_FILE_MACHINE_THUMB", "IMAGE_FILE_MACHINE_ARMNT",
    "IMAGE_FILE_MACHINE_AM33", "IMAGE_FILE_MACHINE_POWERPC", "IMAGE_FILE_MACHINE_POWERPCFP",
    "IMAGE_FILE_MACHINE_IA64", "IMAGE_FILE_MACHINE_MIPS16", "IMAGE_FILE_MACHINE_ALPHA64",
    "IMAGE_FILE_MACHINE_AXP64", "IMAGE_FILE_MACHINE_MIPSFPU", "IMAGE_FILE_MACHINE_MIPSFPU16",
    "IMAGE_FILE_MACHINE_TRICORE", "IMAGE_FILE_MACHINE_CEF", "IMAGE_FILE_MACHINE_EBC",
    "IMAGE_FILE_MACHINE_RISCV32", "IMAGE_FILE_MACHINE_RISCV64", "IMAGE_FILE_MACHINE_RISCV128",
    "IMAGE_FILE_MACHINE_LOONGARCH32", "IMAGE_FILE_MACHINE_LOONGARCH64", "IMAGE_FILE_MACHINE_AMD64",
    "IMAGE_FILE_MACHINE_M32R", "IMAGE_FILE_MACHINE_ARM64", "IMAGE_FILE_MACHINE_CEE",
)
MACHINE_INDEX: dict[str, int] = {m: i for i, m in enumerate(MACHINE_TYPES)}

SUBSYSTEM_TYPES: tuple[str, ...] = (
    "IMAGE_SUBSYSTEM_UNKNOWN", "IMAGE_SUBSYSTEM_NATIVE", "IMAGE_SUBSYSTEM_WINDOWS_GUI",
    "IMAGE_SUBSYSTEM_WINDOWS_CUI", "IMAGE_SUBSYSTEM_OS2_CUI", "IMAGE_SUBSYSTEM_POSIX_CUI",
    "IMAGE_SUBSYSTEM_NATIVE_WINDOWS", "IMAGE_SUBSYSTEM_WINDOWS_CE_GUI",
    "IMAGE_SUBSYSTEM_EFI_APPLICATION", "IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER",
    "IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER", "IMAGE_SUBSYSTEM_EFI_ROM", "IMAGE_SUBSYSTEM_XBOX",
    "IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION",
)
SUBSYSTEM_INDEX: dict[str, int] = {s: i for i, s in enumerate(SUBSYSTEM_TYPES)}

IMAGE_CHARACTERISTICS: tuple[str, ...] = (
    "RELOCS_STRIPPED", "EXECUTABLE_IMAGE", "LINE_NUMS_STRIPPED", "LOCAL_SYMS_STRIPPED",
    "AGGRESIVE_WS_TRIM", "LARGE_ADDRESS_AWARE", "16BIT_MACHINE", "BYTES_REVERSED_LO",
    "32BIT_MACHINE", "DEBUG_STRIPPED", "REMOVABLE_RUN_FROM_SWAP", "NET_RUN_FROM_SWAP", "SYSTEM",
    "DLL", "UP_SYSTEM_ONLY", "BYTES_REVERSED_HI",
)
DLL_CHARACTERISTICS: tuple[str, ...] = (
    "HIGH_ENTROPY_VA", "DYNAMIC_BASE", "FORCE_INTEGRITY", "NX_COMPAT", "NO_ISOLATION", "NO_SEH",
    "NO_BIND", "APPCONTAINER", "WDM_DRIVER", "GUARD_CF", "TERMINAL_SERVER_AWARE",
)
DOS_MEMBERS: tuple[str, ...] = (
    "e_magic", "e_cblp", "e_cp", "e_crlc", "e_cparhdr", "e_minalloc", "e_maxalloc", "e_ss", "e_sp",
    "e_csum", "e_ip", "e_cs", "e_lfarlc", "e_ovno", "e_oemid", "e_oeminfo", "e_lfanew",
)
# The 30 scalar header fields, in vector order: (raw section, raw key).
HEADER_SCALARS: tuple[tuple[str, str], ...] = (
    ("coff", "timestamp"), ("coff", "number_of_sections"), ("coff", "number_of_symbols"),
    ("coff", "sizeof_optional_header"), ("coff", "pointer_to_symbol_table"),
    ("coff", "machine"), ("optional", "subsystem"),  # categorical indices
    ("optional", "major_image_version"), ("optional", "minor_image_version"),
    ("optional", "major_linker_version"), ("optional", "minor_linker_version"),
    ("optional", "major_operating_system_version"), ("optional", "minor_operating_system_version"),
    ("optional", "major_subsystem_version"), ("optional", "minor_subsystem_version"),
    ("optional", "sizeof_code"), ("optional", "sizeof_headers"), ("optional", "sizeof_image"),
    ("optional", "sizeof_initialized_data"), ("optional", "sizeof_uninitialized_data"),
    ("optional", "sizeof_stack_reserve"), ("optional", "sizeof_stack_commit"),
    ("optional", "sizeof_heap_reserve"), ("optional", "sizeof_heap_commit"),
    ("optional", "address_of_entrypoint"), ("optional", "base_of_code"),
    ("optional", "image_base"), ("optional", "section_alignment"), ("optional", "checksum"),
    ("optional", "number_of_rvas_and_sizes"),
)
DATA_DIRECTORY_NAMES: tuple[str, ...] = (
    "EXPORT", "IMPORT", "RESOURCE", "EXCEPTION", "SECURITY", "BASERELOC", "DEBUG", "COPYRIGHT",
    "GLOBALPTR", "TLS", "LOAD_CONFIG", "BOUND_IMPORT", "IAT", "DELAY_IMPORT", "COM_DESCRIPTOR",
    "RESERVED",
)
DATA_DIRECTORY_INDEX: dict[str, int] = {k: i for i, k in enumerate(DATA_DIRECTORY_NAMES)}

AUTHENTICODE_FIELDS: tuple[str, ...] = (
    "num_certs", "self_signed", "empty_program_name", "no_countersigner", "parse_error",
    "chain_max_depth", "latest_signing_time", "signing_time_diff",
)

# thrember/pefile_warnings.txt, in file order (a warning's one-hot index is its line number).
PEFILE_WARNINGS: tuple[str, ...] = (
    "AddressOfEntryPoint lies outside the sections' boundaries...",
    "Bad RVA in relocation data...",
    "Byte 0x...",
    "Corrupt header...",
    "Damaged Import Table information...",
    "Don't know how to parse LOAD_CONFIG information for non-PE32...",
    "Error, too many imported symbols...",
    "Error parsing a resource directory data entry...",
    "Error parsing export directory at RVA...",
    "Error parsing resource of type RT_STRING at...",
    "Error parsing StringFileInfo/VarFileInfo struct...",
    "Error parsing the Delay import directory...",
    "Error parsing the Delay import directory at RVA...",
    "Error parsing the import directory at RVA...",
    "Error parsing the import directory. Invalid Import data at RVA...",
    "Error parsing the import table. Entries go beyond bounds...",
    "Error parsing the import table. AddressOfData overlaps with THUNK_DATA for THUNK at RVA...",
    "Error parsing the import table. Invalid data at RVA...",
    "Error parsing the resources directory. Excessively nested table depth...",
    "Error parsing the resources directory. The directory contains...",
    "Error parsing the resources directory. The file contains at least...",
    "Error parsing the resources directory. Entry...",
    "Error parsing the resources directory, attempting to read entry name. Entry names overlap...",
    "Error parsing the resources directory, attempting to read entry name. Can't read unicode "
    "string at offset...",
    "Error parsing the version information, attempting to read OffsetToData with RVA...",
    "Error parsing the version information, attempting to read VS_VERSION_INFO string...",
    "Error parsing the version information, attempting to read VarFileInfo Var string...",
    "Error parsing the version information, attempting to read StringFileInfo string...",
    "Error parsing the version information, attempting to read StringTable string...",
    "Error parsing the version information, attempting to read StringTable Key string...",
    "Error parsing the version information, to read StringTable Value string...",
    "Excessive number of imports...",
    "Export directory contains more than 10 repeated entries...",
    "Failed parsing FunctionEntry of UNWIND_INFO at...",
    "Failed rendering pascal string, attempting to read from RVA 0x...",
    "Failed rendering unicode string, attempting to read from RVA 0x...",
    "Failed to process directory...",
    "FunctionEntry of UNWIND_INFO at...",
    "If SectionAlignment...",
    "If FileAlignment > 0x200 it should be a power of 2. Value...",
    "Imported symbols contain entries typical of packed executables...",
    "Invalid bdd dynamic relocation...",
    "Invalid bdd info...",
    "Invalid debug information...",
    "Invalid function override header...",
    "Invalid function override info...",
    "Invalid IMAGE_DYNAMIC_RELOCATION_TABLE information...",
    "Invalid LOAD_CONFIG information...",
    "Invalid relocation information. Can't read...",
    "Invalid relocation information. SizeOfBlock too large...",
    "Invalid relocation information. VirtualAddress outside...",
    "Invalid resources directory. Can't read...",
    "Invalid resources directory. Can't parse directory data at RVA...",
    "Invalid TLS information. Can't read...",
    "Invalid type 0x...",
    "Invalid VS_VERSION_INFO block...",
    "No parsing available for IMAGE_DYNAMIC_RELOCATION_TABLE...",
    "Overlapping offsets in relocation data...",
    "Possibly corrupt file. AddressOfEntryPoint lies outside the file...",
    "Relocating image but PE does not have (or pefile cannot parse) a DIRECTORY_ENTRY_BASERELOC...",
    "Resource size...",
    "Rich Header is malformed...",
    "Rich Header is not in Microsoft format, possibly malformed...",
    "RVA AddressOfFunctions in the export directory points to an invalid...",
    "RVA AddressOfNames in the export directory points to an invalid...",
    "RVA of IMAGE_BOUND_IMPORT_DESCRIPTOR points...",
    "SizeOfHeaders is smaller than AddressOfEntryPoint...",
    "Suspicious flags set for section...",
    "Suspicious NumberOfRvaAndSizes in the Optional Header...",
    "Suspicious value found parsing section...",
    "The Bound Imports directory exists but can't be parsed...",
    "Too many warnings parsing section. Aborting...",
    "Too many errors parsing the Delay import directory...",
    "Too many errors parsing the import directory...",
    "Too many sections...",
    "Unknown UNWIND_CODE at...",
    "Unsupported version of UNWIND_INFO...",
    "...Contents are null-bytes.",
    "...No data in the file (is this corkami's virtsectblXP?).",
    "...PointerToRawData points beyond the end of the file.",
    "...PointerToRawData should normally be a multiple of FileAlignment, this might imply the file "
    "is trying to confuse tools which parse this incorrectly.",
    "...SizeOfRawData is larger than file.",
    "...VirtualSize is extremely large > 256MiB",
    "...VirtualAddress is beyond 0x10000000",
    "...symbol entries. Assuming corrupt.",
    "...ordinal entries. Assuming corrupt.",
    "...Assuming corrupt.",
)
WARNING_INDEX: dict[str, int] = {w: i for i, w in enumerate(PEFILE_WARNINGS)}

# Group sizes and offsets (must match malvalid.schemas.ember_v3.V3_GROUPS).
GENERAL_DIM = 7
HISTOGRAM_DIM = 256
BYTEENTROPY_DIM = 256
STRINGS_DIM = 3 + 96 + 1 + len(REGEX_NAMES)  # 177
HEADER_DIM = len(HEADER_SCALARS) + len(IMAGE_CHARACTERISTICS) + len(DLL_CHARACTERISTICS) + len(DOS_MEMBERS)
SECTION_DIM = 11 + 50 + 50 + 50 + 50 + 10 + 3  # 224
IMPORTS_DIM = 2 + 256 + 1024
EXPORTS_DIM = 1 + 128
DATADIRECTORIES_DIM = 2 * len(DATA_DIRECTORY_NAMES) + 2
RICHHEADER_DIM = 1 + 32
AUTHENTICODE_DIM = len(AUTHENTICODE_FIELDS)
PEFILEWARNINGS_DIM = len(PEFILE_WARNINGS) + 1

GROUP_DIMS: tuple[tuple[str, int], ...] = (
    ("general", GENERAL_DIM),
    ("histogram", HISTOGRAM_DIM),
    ("byteentropy", BYTEENTROPY_DIM),
    ("strings", STRINGS_DIM),
    ("header", HEADER_DIM),
    ("section", SECTION_DIM),
    ("imports", IMPORTS_DIM),
    ("exports", EXPORTS_DIM),
    ("datadirectories", DATADIRECTORIES_DIM),
    ("richheader", RICHHEADER_DIM),
    ("authenticode", AUTHENTICODE_DIM),
    ("pefilewarnings", PEFILEWARNINGS_DIM),
)
OFFSETS: dict[str, int] = {}
_pos = 0
for _name, _d in GROUP_DIMS:
    OFFSETS[_name] = _pos
    _pos += _d
DIM = _pos
assert DIM == 2568, DIM
assert STRINGS_DIM == 177 and HEADER_DIM == 74 and PEFILEWARNINGS_DIM == 88

# Offsets of the hashed blocks inside the section group.
SEC_GENERAL, SEC_SIZES, SEC_VSIZES, SEC_ENTROPY, SEC_CHARS, SEC_ENTRY, SEC_OVERLAY = (
    0, 11, 61, 111, 161, 211, 221,
)


# --------------------------------------------------------------------------------------------------
# Hashing (sklearn FeatureHasher, exactly as upstream)
# --------------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _hasher(n_features: int, input_type: str, alternate_sign: bool = True) -> Any:
    from sklearn.feature_extraction import FeatureHasher

    return FeatureHasher(n_features, input_type=input_type, alternate_sign=alternate_sign)


def _hash_rows(samples: list[list[Any]], n_features: int, input_type: str, alternate_sign: bool = True) -> np.ndarray:
    """(len(samples), n_features) float64 dense hashed matrix."""
    if not samples:
        return np.zeros((0, n_features), dtype=np.float64)
    return _hasher(n_features, input_type, alternate_sign).transform(samples).toarray()


# --------------------------------------------------------------------------------------------------
# Vectorization: raw JSON feature dict(s) -> float32 vector(s)
# --------------------------------------------------------------------------------------------------


def _normalized_hist(rows: list[Any], width: int) -> np.ndarray:
    """Batch of count lists -> float32 counts / float32 row sum (upstream arithmetic)."""
    out = np.zeros((len(rows), width), dtype=np.float32)
    good = [i for i, r in enumerate(rows) if isinstance(r, (list, tuple)) and len(r) == width]
    if not good:
        return out
    counts = np.array([rows[i] for i in good], dtype=np.float32)
    s = counts.sum(axis=1, keepdims=True, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = counts / s
    norm[~np.isfinite(norm)] = 0.0  # empty file: upstream would produce NaN
    out[good] = norm
    return out


def _strings_vec(raw: dict[str, Any]) -> np.ndarray:
    if not raw:
        return np.zeros(STRINGS_DIM, dtype=np.float32)
    printables = raw.get("printables", 0) or 0
    hist_divisor = float(printables) if printables > 0 else 1.0
    counts = np.zeros(len(REGEX_NAMES), dtype=np.float32)
    for regex, count in (raw.get("string_counts") or {}).items():
        idx = REGEX_INDEX.get(regex)
        if idx is not None:
            counts[idx] = count
    dist = raw.get("printabledist") or [0] * 96
    return np.hstack(
        [
            raw.get("numstrings", 0),
            raw.get("avlength", 0),
            printables,
            np.asarray(dist) / hist_divisor,
            raw.get("entropy", 0),
            counts,
        ]
    ).astype(np.float32)


def _header_vec(raw: dict[str, Any]) -> np.ndarray:
    if not raw:
        return np.zeros(HEADER_DIM, dtype=np.float32)
    coff = raw.get("coff") or {}
    opt = raw.get("optional") or {}
    dos = raw.get("dos") or {}
    vals: list[Any] = []
    for sec, key in HEADER_SCALARS:
        if key == "machine":
            vals.append(MACHINE_INDEX.get(coff.get("machine", ""), 0))
        elif key == "subsystem":
            vals.append(SUBSYSTEM_INDEX.get(opt.get("subsystem", ""), 0))
        else:
            vals.append((coff if sec == "coff" else opt).get(key, 0))
    chars = coff.get("characteristics") or []
    dllc = opt.get("dll_characteristics") or []
    vals.extend(ch in chars for ch in IMAGE_CHARACTERISTICS)
    vals.extend(ch in dllc for ch in DLL_CHARACTERISTICS)
    vals.extend(dos.get(m, 0) for m in DOS_MEMBERS)
    # np.array on python ints/bools infers int64 (like upstream np.hstack) before the float32 cast.
    return np.array(vals).astype(np.float32)


def _section_general(raw: dict[str, Any]) -> tuple[list[float], list[float]]:
    sections = raw.get("sections") or []
    overlay = raw.get("overlay") or {"size": 0, "size_ratio": 0, "entropy": 0}
    entropies = [s["entropy"] for s in sections] + [overlay.get("entropy", 0)] + [0]
    size_ratios = [s["size_ratio"] for s in sections] + [overlay.get("size_ratio", 0)] + [0]
    vsize_ratios = [s["vsize_ratio"] for s in sections] + [0]
    # Section counts: empty (raw size 0), unnamed, readable+executable, writable.
    n_empty = n_unnamed = n_rx = n_w = 0
    for sec in sections:
        flags = sec["props"]
        n_empty += sec["size"] == 0
        n_unnamed += sec["name"] == ""
        n_rx += "MEM_READ" in flags and "MEM_EXECUTE" in flags
        n_w += "MEM_WRITE" in flags
    general = [
        len(sections),
        n_empty,
        n_unnamed,
        n_rx,
        n_w,
        max(entropies),
        min(entropies),
        max(size_ratios),
        min(size_ratios),
        max(vsize_ratios),
        min(vsize_ratios),
    ]
    tail = [overlay.get("size", 0), overlay.get("size_ratio", 0), overlay.get("entropy", 0)]
    return general, tail


def _datadir_vec(raw: list[Any]) -> np.ndarray:
    features = np.zeros(DATADIRECTORIES_DIM, dtype=np.float32)
    if not raw:
        return features
    # Upstream iterates range(1, len(raw) - 1): the last directory (RESERVED) is never written.
    for i in range(1, len(raw) - 1):
        idx = DATA_DIRECTORY_INDEX.get(raw[i].get("name", ""))
        if idx is None:
            continue
        features[2 * idx] = raw[i]["size"]
        features[2 * idx + 1] = raw[i]["virtual_address"]
    features[-2] = raw[0].get("has_relocs", 0)
    features[-1] = raw[0].get("has_dynamic_relocs", 0)
    return features


def _authenticode_vec(raw: dict[str, Any]) -> np.ndarray:
    if not raw:
        return np.zeros(AUTHENTICODE_DIM, dtype=np.float32)
    return np.hstack([raw.get(k, 0) for k in AUTHENTICODE_FIELDS]).astype(np.float32)


def _warnings_vec(raw: list[str]) -> np.ndarray:
    if not raw:
        return np.zeros(PEFILEWARNINGS_DIM, dtype=np.float32)
    ids = np.zeros(PEFILEWARNINGS_DIM, dtype=np.float32)
    for w in raw:
        i = WARNING_INDEX.get(w)
        if i is None:
            log.debug("unknown normalised pefile warning in raw features: %r", w)
            continue
        ids[i] = 1.0
    ids[-1] = len(raw)
    return ids


def vectorize_batch(raws: Sequence[dict[str, Any]]) -> np.ndarray:
    """EMBER v3 raw feature dicts (thrember JSONL rows) -> (n, 2568) float32.

    Missing groups are treated like upstream's "not a PE" encoding (all zeros for that group).
    """
    n = len(raws)
    X = np.zeros((n, DIM), dtype=np.float32)
    if n == 0:
        return X
    o = OFFSETS

    # general (7): size, entropy, is_pe, start_bytes[4]
    for i, r in enumerate(raws):
        g = r.get("general") or {}
        if g:
            X[i, o["general"] : o["general"] + GENERAL_DIM] = np.hstack(
                [g.get("size", 0), g.get("entropy", 0), g.get("is_pe", 0), g.get("start_bytes") or [0, 0, 0, 0]],
                dtype=np.float32,
            )

    # byte histograms
    X[:, o["histogram"] : o["histogram"] + HISTOGRAM_DIM] = _normalized_hist(
        [r.get("histogram") for r in raws], HISTOGRAM_DIM
    )
    X[:, o["byteentropy"] : o["byteentropy"] + BYTEENTROPY_DIM] = _normalized_hist(
        [r.get("byteentropy") for r in raws], BYTEENTROPY_DIM
    )

    # per-row dense groups
    for i, r in enumerate(raws):
        X[i, o["strings"] : o["strings"] + STRINGS_DIM] = _strings_vec(r.get("strings") or {})
        X[i, o["header"] : o["header"] + HEADER_DIM] = _header_vec(r.get("header") or {})
        X[i, o["datadirectories"] : o["datadirectories"] + DATADIRECTORIES_DIM] = _datadir_vec(
            r.get("datadirectories") or []
        )
        X[i, o["authenticode"] : o["authenticode"] + AUTHENTICODE_DIM] = _authenticode_vec(
            r.get("authenticode") or {}
        )
        X[i, o["pefilewarnings"] : o["pefilewarnings"] + PEFILEWARNINGS_DIM] = _warnings_vec(
            r.get("pefilewarnings") or []
        )

    # section (224): general(11) + 4x50 hashed + entry(10) + overlay(3)
    sec_rows = [i for i, r in enumerate(raws) if r.get("section")]
    if sec_rows:
        sizes, vsizes, ents, chars, entry = [], [], [], [], []
        base = o["section"]
        for i in sec_rows:
            s = raws[i]["section"]
            secs = s.get("sections") or []
            general, tail = _section_general(s)
            X[i, base + SEC_GENERAL : base + SEC_GENERAL + 11] = np.asarray(general, dtype=np.float64)
            X[i, base + SEC_OVERLAY : base + SEC_OVERLAY + 3] = np.asarray(tail, dtype=np.float64)
            sizes.append([(x["name"], x["size"]) for x in secs])
            vsizes.append([(x["name"], x["vsize"]) for x in secs])
            ents.append([(x["name"], x["entropy"]) for x in secs])
            chars.append([f"{x['name']}:{p}" for x in secs for p in x["props"]])
            entry.append([s.get("entry", "")])
        for block, samples, width, kind in (
            (SEC_SIZES, sizes, 50, "pair"),
            (SEC_VSIZES, vsizes, 50, "pair"),
            (SEC_ENTROPY, ents, 50, "pair"),
            (SEC_CHARS, chars, 50, "string"),
            (SEC_ENTRY, entry, 10, "string"),
        ):
            X[sec_rows, base + block : base + block + width] = _hash_rows(samples, width, kind, True)

    # imports (1282): [n_functions, n_libraries] + libraries(256) + functions(1024), unsigned hashing
    imp_rows = [i for i, r in enumerate(raws) if r.get("imports")]
    if imp_rows:
        libs_s, funcs_s = [], []
        base = o["imports"]
        for i in imp_rows:
            imp = raws[i]["imports"]
            libraries = list({lib.lower() for lib in imp})
            funcs = [lib.lower() + ":" + e for lib, elist in imp.items() for e in elist]
            X[i, base] = len(funcs)
            X[i, base + 1] = len(libraries)
            libs_s.append(libraries)
            funcs_s.append(funcs)
        X[imp_rows, base + 2 : base + 258] = _hash_rows(libs_s, 256, "string", False)
        X[imp_rows, base + 258 : base + 1282] = _hash_rows(funcs_s, 1024, "string", False)

    # exports (129): [128 if any export else 0] + hashed(128), signed hashing
    exp_rows = [i for i, r in enumerate(raws) if r.get("exports")]
    if exp_rows:
        base = o["exports"]
        X[exp_rows, base] = 128.0  # upstream stores len(exports_hashed) (quirk)
        X[exp_rows, base + 1 : base + 129] = _hash_rows([list(raws[i]["exports"]) for i in exp_rows], 128, "string", True)

    # richheader (33): n_pairs + hashed (str(compid), count) pairs, signed hashing
    rich_rows = [i for i, r in enumerate(raws) if r.get("richheader")]
    if rich_rows:
        base = o["richheader"]
        pairs_s = []
        for i in rich_rows:
            v = raws[i]["richheader"]
            X[i, base] = int(len(v) / 2)
            pairs_s.append([(str(v[k]), v[k + 1]) for k in range(0, len(v) - 1, 2)])
        X[rich_rows, base + 1 : base + 33] = _hash_rows(pairs_s, 32, "pair", True)

    return X


def vectorize(raw: dict[str, Any]) -> np.ndarray:
    """One raw feature dict -> (2568,) float32."""
    return vectorize_batch([raw])[0]


# --------------------------------------------------------------------------------------------------
# Raw extraction: PE bytes -> raw feature dict (needs pefile; signify optional)
# --------------------------------------------------------------------------------------------------


def pefile_available() -> bool:
    try:
        import pefile  # noqa: F401
    except Exception:
        return False
    return True


def signify_available() -> bool:
    try:
        from signify.authenticode import SignedPEFile  # noqa: F401
    except Exception:
        return False
    return True


def _byte_entropy(data: bytes) -> float:
    """Shannon entropy (bits) of a byte string, as pefile's entropy_H / upstream."""
    size = len(data)
    if size == 0:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    # Sum in order of first occurrence, like iterating upstream's collections.Counter.
    present = np.flatnonzero(counts)
    order = sorted(present.tolist(), key=lambda v: data.find(bytes((v,))))
    entropy = 0.0
    for v in order:
        p_x = float(counts[v]) / size
        entropy -= p_x * math.log(p_x, 2)
    return entropy


def _raw_general(bytez: bytes, pe: Any) -> dict[str, Any]:
    size = len(bytez)
    sb = [int(bytez[k]) if size > k else 0 for k in range(4)]
    return {"size": size, "entropy": _byte_entropy(bytez), "is_pe": 0 if pe is None else 1, "start_bytes": sb}


def _raw_histogram(bytez: bytes) -> list[int]:
    return np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256).tolist()


def _raw_byteentropy(bytez: bytes) -> list[int]:
    """The 16x16 byte/entropy histogram (entropy bin x high nibble; window 2048, step 1024), row-major.

    Feature versions 2 and 3 define this group identically, so it is computed by malvalid's own
    implementation, :func:`malvalid.schemas.ember_v2.byte_entropy_histogram` (float32 entropy per
    window, like the published vectors).
    """
    from malvalid.schemas.ember_v2 import byte_entropy_histogram

    return [int(v) for v in byte_entropy_histogram(bytez)]


_ALLSTRINGS = re.compile(b"[\x20-\x7f]{5,}")


@functools.lru_cache(maxsize=1)
def _compiled_regexes() -> dict[str, re.Pattern[str]]:
    return {k: re.compile(p, f) for k, (p, f) in STRING_REGEXES.items()}


def _printable_profile(found: list[bytes]) -> tuple[list[Any], int, float, float]:
    """Histogram of the 96 printable characters (0x20..0x7f) over the strings, its total, the
    Shannon entropy (bits) of that distribution and the mean string length.

    The probabilities are float32-rounded counts divided in float64 by the total, which is what the
    reference produces under numpy >= 2; without strings the histogram is 96 float zeros.
    """
    if not found:
        return [0.0] * 96, 0, 0.0, 0
    text = b"".join(found)
    hist = np.bincount(np.frombuffer(text, dtype=np.uint8), minlength=0x80)[0x20:0x80]
    total = int(hist.sum())
    q = hist[hist > 0].astype(np.float32).astype(np.float64) / float(total)
    entropy = float(np.sum(-q * np.log2(q)))
    return hist.tolist(), total, entropy, len(text) / len(found)


def _raw_strings(bytez: bytes) -> dict[str, Any]:
    allstrings = _ALLSTRINGS.findall(bytez)
    dist, printables, entropy, avlength = _printable_profile(allstrings)

    # Count strings with >= 1 match per regex. Strings are joined with "\n", which no regex can
    # match and which is a non-word character, so matches (and \b boundaries) never cross strings;
    # this equals upstream's per-string re.search loop.
    string_counts: dict[str, int] = {}
    if allstrings:
        text = "\n".join(s.decode("ascii", errors="ignore") for s in allstrings)
        starts = np.cumsum([0] + [len(s) + 1 for s in allstrings[:-1]])
        for k, rx in _compiled_regexes().items():
            pos = [m.start() for m in rx.finditer(text)]
            if pos:
                owners = np.searchsorted(starts, np.asarray(pos), side="right") - 1
                string_counts[k] = int(np.unique(owners).size)
    string_counts = dict(sorted(string_counts.items()))
    return {
        "numstrings": len(allstrings),
        "avlength": avlength,
        "printabledist": dist,
        "printables": printables,
        "entropy": entropy,
        "string_counts": string_counts,
    }


def _section_name(section: Any) -> str:
    return section.Name.strip(b"\x00").decode(errors="ignore").lower()


def _raw_section(bytez: bytes, pe: Any) -> dict[str, Any]:
    import pefile

    if pe is None:
        return {}
    entry_section = ""
    aoep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    for section in pe.sections:
        if section.contains_rva(aoep):
            entry_section = _section_name(section)
    isection = 0
    while entry_section == "" and isection < len(pe.sections):
        if pe.sections[isection].Characteristics & 0x20000000 > 0:
            entry_section = _section_name(pe.sections[isection])
        isection += 1
    raw_obj: dict[str, Any] = {"entry": entry_section}
    raw_obj["sections"] = [
        {
            "name": _section_name(s),
            "size": s.SizeOfRawData,
            "entropy": s.get_entropy(),
            "vsize": s.Misc_VirtualSize,
            "size_ratio": s.SizeOfRawData / len(bytez),
            "vsize_ratio": s.SizeOfRawData / max(s.Misc_VirtualSize, 1),
            "props": [sc[10:] for sc, _ in pefile.section_characteristics if s.__dict__.get(sc)],
        }
        for s in pe.sections
    ]
    raw_obj["overlay"] = {"size": 0, "size_ratio": 0, "entropy": 0}
    overlay = pe.get_overlay()
    if overlay is not None:
        raw_obj["overlay"] = {
            "size": len(overlay),
            "size_ratio": len(overlay) / len(bytez),
            "entropy": _byte_entropy(overlay),
        }
    return raw_obj


def _raw_imports(pe: Any) -> dict[str, list[str]]:
    imports: dict[str, list[str]] = {}
    if pe is None or "DIRECTORY_ENTRY_IMPORT" not in pe.__dict__:
        return imports
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        dll_name = entry.dll.decode(errors="ignore")
        imports[dll_name] = []
        for lib in entry.imports:
            if lib.name is not None and len(lib.name):
                imports[dll_name].append(lib.name.decode(errors="ignore")[:10000])
            elif lib.ordinal is not None:
                imports[dll_name].append(f"{dll_name}:ordinal{lib.ordinal}")
    return imports


def _raw_exports(pe: Any) -> list[str]:
    if pe is None:
        return []
    out: list[str] = []
    if "DIRECTORY_ENTRY_EXPORT" in pe.__dict__:
        for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            if exp.name is not None and len(exp.name):
                out.append(exp.name.decode(errors="ignore")[:10000])
            elif exp.ordinal is not None:
                out.append(f"ordinal{exp.ordinal}")
    return out


def _raw_header(pe: Any) -> dict[str, Any]:
    import pefile

    if pe is None:
        return {}
    fh, oh = pe.FILE_HEADER, pe.OPTIONAL_HEADER
    coff = {
        "timestamp": fh.TimeDateStamp,
        "machine": pefile.MACHINE_TYPE.get(fh.Machine, "IMAGE_FILE_MACHINE_UNKNOWN"),
        "number_of_sections": fh.NumberOfSections,
        "number_of_symbols": fh.NumberOfSymbols,
        "sizeof_optional_header": fh.SizeOfOptionalHeader,
        "pointer_to_symbol_table": fh.PointerToSymbolTable,
        "characteristics": [k[11:] for k, v in fh.__dict__.items() if k.startswith("IMAGE_FILE_") and v],
    }
    optional = {
        "magic": oh.Magic,
        "subsystem": pefile.SUBSYSTEM_TYPE.get(oh.Subsystem, "IMAGE_SUBSYSTEM_UNKNOWN"),
        "major_image_version": oh.MajorImageVersion,
        "minor_image_version": oh.MinorImageVersion,
        "major_linker_version": oh.MajorLinkerVersion,
        "minor_linker_version": oh.MinorLinkerVersion,
        "major_operating_system_version": oh.MajorOperatingSystemVersion,
        "minor_operating_system_version": oh.MinorOperatingSystemVersion,
        "major_subsystem_version": oh.MajorSubsystemVersion,
        "minor_subsystem_version": oh.MinorSubsystemVersion,
        "sizeof_code": oh.SizeOfCode,
        "sizeof_headers": oh.SizeOfHeaders,
        "sizeof_image": oh.SizeOfImage,
        "sizeof_initialized_data": oh.SizeOfInitializedData,
        "sizeof_uninitialized_data": oh.SizeOfUninitializedData,
        "sizeof_stack_reserve": oh.SizeOfStackReserve,
        "sizeof_stack_commit": oh.SizeOfStackCommit,
        "sizeof_heap_reserve": oh.SizeOfHeapReserve,
        "sizeof_heap_commit": oh.SizeOfHeapCommit,
        "address_of_entrypoint": oh.AddressOfEntryPoint,
        "base_of_code": oh.BaseOfCode,
        "base_of_data": 0,  # present in upstream raw JSON, never vectorized
        "image_base": oh.ImageBase,
        "section_alignment": oh.SectionAlignment,
        "checksum": oh.CheckSum,
        "number_of_rvas_and_sizes": oh.NumberOfRvaAndSizes,
        "dll_characteristics": [
            k[25:] for k, v in oh.__dict__.items() if k.startswith("IMAGE_DLLCHARACTERISTICS_") and v
        ],
    }
    dos = {m: 0 for m in DOS_MEMBERS}
    dos_dict = pe.DOS_HEADER.dump_dict()
    for m in DOS_MEMBERS:
        entry = dos_dict.get(m)
        if isinstance(entry, dict) and entry.get("Value") is not None:
            dos[m] = entry["Value"]
    return {"coff": coff, "optional": optional, "dos": dos}


def _raw_datadirectories(pe: Any) -> list[dict[str, Any]]:
    if pe is None:
        return []
    out: list[dict[str, Any]] = [
        {"has_relocs": int(pe.has_relocs()), "has_dynamic_relocs": int(pe.has_dynamic_relocs())}
    ]
    for dd in pe.OPTIONAL_HEADER.DATA_DIRECTORY:
        out.append(
            {
                "name": str(dd.name).replace("IMAGE_DIRECTORY_ENTRY_", ""),
                "size": dd.Size,
                "virtual_address": dd.VirtualAddress,
            }
        )
    return out


def _raw_richheader(pe: Any) -> list[int]:
    if pe is not None and getattr(pe, "RICH_HEADER", None) is not None:
        return list(pe.RICH_HEADER.values)
    return []


_warned_no_signify = False


def _security_dir(pe: Any) -> tuple[int, int]:
    try:
        dd = pe.OPTIONAL_HEADER.DATA_DIRECTORY[4]  # IMAGE_DIRECTORY_ENTRY_SECURITY
        return int(dd.VirtualAddress), int(dd.Size)
    except Exception:
        return 0, 0


def _raw_authenticode(bytez: bytes, pe: Any) -> dict[str, Any]:
    global _warned_no_signify
    if pe is None:
        return {}
    raw_obj: dict[str, Any] = {k: 0 for k in AUTHENTICODE_FIELDS}
    offset, size = _security_dir(pe)
    if not signify_available():
        if offset and size:
            # Approximation without signify: count PKCS#7 WIN_CERTIFICATE entries only.
            if not _warned_no_signify:
                log.warning(
                    "signify is not installed: Authenticode features of signed files are approximate "
                    "(install 'signify' for exact EMBER v3 features)"
                )
                _warned_no_signify = True
            pos, end = offset, min(offset + size, len(bytez))
            while pos + 8 <= end:
                length = int.from_bytes(bytez[pos : pos + 4], "little")
                ctype = int.from_bytes(bytez[pos + 6 : pos + 8], "little")
                if length < 8:
                    break
                if ctype == 2:
                    raw_obj["num_certs"] += 1
                pos += (length + 7) & ~7
        return raw_obj

    import signify
    from signify.authenticode import SignedPEFile

    try:
        signed_pe = SignedPEFile(io.BytesIO(bytez))
        it = getattr(signed_pe, "iter_signed_datas", None)
        datas = it() if callable(it) else getattr(signed_pe, "signed_datas", [])
        for signed_data in datas:
            raw_obj["num_certs"] += 1
            if signed_data.signer_info.program_name is None:
                raw_obj["empty_program_name"] = 1
            countersigner = signed_data.signer_info.countersigner
            if countersigner is not None:
                signing_time = countersigner.signing_time.timestamp()
                if signing_time >= raw_obj["latest_signing_time"]:
                    raw_obj["latest_signing_time"] = signing_time
                raw_obj["signing_time_diff"] = signing_time - pe.FILE_HEADER.TimeDateStamp
            else:
                raw_obj["no_countersigner"] = 1
            certs = signed_data.certificates
            if len(certs) > raw_obj["chain_max_depth"]:
                raw_obj["chain_max_depth"] = len(certs)
            for cert in certs[:-1]:
                if cert.issuer == cert.subject:
                    raw_obj["self_signed"] = 1
    except (signify.exceptions.ParseError, ValueError, KeyError):  # SignerInfoParseError is a ParseError
        raw_obj["parse_error"] = 1
    except Exception as e:  # defensive: never let a malformed signature crash featurization
        log.debug("signify failed: %s", e)
        raw_obj["parse_error"] = 1
    return raw_obj


# Longest pattern first, so the most specific normalised warning wins (deterministic).
_WARN_SUFFIXES = tuple(sorted((w[3:] for w in PEFILE_WARNINGS if w.startswith("...")), key=len, reverse=True))
_WARN_PREFIXES = tuple(sorted((w[:-3] for w in PEFILE_WARNINGS if not w.startswith("...")), key=len, reverse=True))


def _normalize_warnings(warnings: Sequence[str]) -> list[str]:
    suffixes, prefixes = _WARN_SUFFIXES, _WARN_PREFIXES
    norm: set[str] = set()
    for warning in set(warnings):
        suf = next((s for s in suffixes if warning.endswith(s)), None)
        if suf is not None:
            norm.add("..." + suf)
            continue
        pre = next((p for p in prefixes if warning.startswith(p)), None)
        if pre is not None:
            norm.add(pre + "...")
            continue
        log.debug("unknown pefile warning (not an EMBER v3 feature): %s", warning)
    return sorted(norm)


def raw_features(bytez: bytes) -> dict[str, Any]:
    """Raw PE bytes -> EMBER v3 raw feature dict (the thrember JSONL row format, features only)."""
    import pefile

    bytez = bytes(bytez)
    pe = None
    try:
        pe = pefile.PE(data=bytez)
    except (pefile.PEFormatError, AttributeError):
        pe = None
    except Exception as e:  # malformed input must not crash the caller
        log.debug("pefile failed to parse input (%s); treating as non-PE", e)
        pe = None
    try:
        return {
            "sha256": hashlib.sha256(bytez).hexdigest(),
            "general": _raw_general(bytez, pe),
            "histogram": _raw_histogram(bytez),
            "byteentropy": _raw_byteentropy(bytez),
            "strings": _raw_strings(bytez),
            "header": _raw_header(pe),
            "section": _raw_section(bytez, pe),
            "imports": _raw_imports(pe),
            "exports": _raw_exports(pe),
            "datadirectories": _raw_datadirectories(pe),
            "richheader": _raw_richheader(pe),
            "authenticode": _raw_authenticode(bytez, pe),
            "pefilewarnings": _normalize_warnings(pe.get_warnings()) if pe is not None else [],
        }
    finally:
        if pe is not None:
            pe.close()


def feature_vector(bytez: bytes) -> np.ndarray:
    return vectorize(raw_features(bytez))
