"""EMBER feature version 3 (EMBER2024 / thrember): 2568 float32 features.

Group boundaries below are PINNED by the build contract (docs/BUILD_CONTRACT.md §4.1); do not change
them. Per-feature names and controllability are defined here; the vectorizer and the optional
raw-PE extractor are an Apache-2.0 port of thrember (``malvalid.schemas._thrember_port``).

Controllability describes how freely the *author of a file* can change a feature without changing
what the program does (M7 uses it to flag detectors that lean on trivially settable features):

* ``controllable`` — free to set (timestamps, checksum, DOS-stub fields, rich header, section
  names and therefore every name-keyed hashed section bucket, linker/OS/image versions, most
  mitigation flags);
* ``append_only`` — can be increased/added to (file size, strings, imports, exports, sections,
  overlay size, attached signatures);
* ``derived`` — moves only as a side effect of other edits (byte/entropy histograms, section
  entropy/size ratios, data-directory sizes, parser warnings);
* ``fixed`` — tied to program semantics (machine, subsystem, entry point, image base, section
  alignment, DLL/executable flags, the MZ magic).
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import numpy as np

from malvalid.schemas import _thrember_port as tp
from malvalid.schemas.base import Controllability as C
from malvalid.schemas.base import FeatureGroup, FeatureSchema

V3_GROUPS = (
    ("general", 0, 7, C.APPEND_ONLY, "General file info"),
    ("histogram", 7, 263, C.DERIVED, "Byte histogram (256 bins, normalized)"),
    ("byteentropy", 263, 519, C.DERIVED, "Byte-entropy histogram"),
    ("strings", 519, 696, C.APPEND_ONLY, "String statistics and regex-category counts"),
    ("header", 696, 770, C.CONTROLLABLE, "DOS/COFF/optional header fields"),
    ("section", 770, 994, C.APPEND_ONLY, "Section info (counts, hashed names/sizes/entropy, overlay)"),
    ("imports", 994, 2276, C.APPEND_ONLY, "Imports (counts + hashed libraries + hashed functions)"),
    ("exports", 2276, 2405, C.APPEND_ONLY, "Exports (count + hashed names)"),
    ("datadirectories", 2405, 2439, C.DERIVED, "Data directories (16 x size/va + 2)"),
    ("richheader", 2439, 2472, C.CONTROLLABLE, "Rich header (count + hashed entries)"),
    ("authenticode", 2472, 2480, C.APPEND_ONLY, "Authenticode signature summary"),
    ("pefilewarnings", 2480, 2568, C.DERIVED, "pefile parser warnings (87 one-hot + count)"),
)

# Indices of the features thrember/EMBER2024 treat as categorical when training LightGBM
# (general.is_pe, general.start_bytes[0..3], header.coff.machine, header.optional.subsystem).
CATEGORICAL_FEATURES: tuple[int, ...] = (2, 3, 4, 5, 6, 701, 702)

# ---- per-feature controllability tables ---------------------------------------------------------

_HEADER_SCALAR_CTRL: dict[str, C] = {
    "timestamp": C.CONTROLLABLE,
    "number_of_sections": C.APPEND_ONLY,
    "number_of_symbols": C.CONTROLLABLE,  # COFF symbol table is ignored for images
    "sizeof_optional_header": C.FIXED,
    "pointer_to_symbol_table": C.CONTROLLABLE,
    "machine": C.FIXED,
    "subsystem": C.FIXED,
    "major_image_version": C.CONTROLLABLE,
    "minor_image_version": C.CONTROLLABLE,
    "major_linker_version": C.CONTROLLABLE,
    "minor_linker_version": C.CONTROLLABLE,
    "major_operating_system_version": C.CONTROLLABLE,
    "minor_operating_system_version": C.CONTROLLABLE,
    "major_subsystem_version": C.CONTROLLABLE,
    "minor_subsystem_version": C.CONTROLLABLE,
    "sizeof_code": C.CONTROLLABLE,  # informational, not used by the loader
    "sizeof_headers": C.DERIVED,
    "sizeof_image": C.DERIVED,
    "sizeof_initialized_data": C.CONTROLLABLE,
    "sizeof_uninitialized_data": C.CONTROLLABLE,
    "sizeof_stack_reserve": C.CONTROLLABLE,
    "sizeof_stack_commit": C.CONTROLLABLE,
    "sizeof_heap_reserve": C.CONTROLLABLE,
    "sizeof_heap_commit": C.CONTROLLABLE,
    "address_of_entrypoint": C.FIXED,
    "base_of_code": C.CONTROLLABLE,  # not used by the loader
    "image_base": C.FIXED,
    "section_alignment": C.FIXED,
    "checksum": C.CONTROLLABLE,  # not verified for user-mode executables
    "number_of_rvas_and_sizes": C.FIXED,
}
_IMAGE_CHAR_FIXED = {"RELOCS_STRIPPED", "EXECUTABLE_IMAGE", "LARGE_ADDRESS_AWARE", "32BIT_MACHINE", "SYSTEM", "DLL"}
_DLL_CHAR_FIXED = {"FORCE_INTEGRITY", "APPCONTAINER", "WDM_DRIVER"}
_DOS_FIXED = {"e_magic", "e_lfanew"}
_DATADIR_CTRL: dict[str, C] = {
    "EXPORT": C.DERIVED,
    "IMPORT": C.DERIVED,
    "RESOURCE": C.DERIVED,
    "EXCEPTION": C.FIXED,
    "SECURITY": C.CONTROLLABLE,  # signatures can be stripped or attached freely
    "BASERELOC": C.DERIVED,
    "DEBUG": C.CONTROLLABLE,
    "COPYRIGHT": C.CONTROLLABLE,  # "architecture", unused
    "GLOBALPTR": C.CONTROLLABLE,
    "TLS": C.FIXED,
    "LOAD_CONFIG": C.FIXED,
    "BOUND_IMPORT": C.CONTROLLABLE,
    "IAT": C.DERIVED,
    "DELAY_IMPORT": C.DERIVED,
    "COM_DESCRIPTOR": C.FIXED,
    "RESERVED": C.FIXED,  # never written by thrember: always 0
}
_AUTH_CTRL: dict[str, C] = {
    "num_certs": C.APPEND_ONLY,
    "self_signed": C.CONTROLLABLE,
    "empty_program_name": C.CONTROLLABLE,
    "no_countersigner": C.CONTROLLABLE,
    "parse_error": C.CONTROLLABLE,
    "chain_max_depth": C.APPEND_ONLY,
    "latest_signing_time": C.CONTROLLABLE,
    "signing_time_diff": C.CONTROLLABLE,
}


def _printable(i: int) -> str:
    return f"0x{i + 0x20:02x}"


@functools.lru_cache(maxsize=1)
def _layout() -> tuple[tuple[str, ...], tuple[C, ...]]:
    names: list[str] = []
    ctrl: list[C] = []

    def add(name: str, c: C) -> None:
        names.append(name)
        ctrl.append(c)

    # general
    add("general.size", C.APPEND_ONLY)
    add("general.entropy", C.DERIVED)
    add("general.is_pe", C.FIXED)
    add("general.start_bytes[0]", C.FIXED)  # 'M'
    add("general.start_bytes[1]", C.FIXED)  # 'Z'
    add("general.start_bytes[2]", C.CONTROLLABLE)  # e_cblp, ignored by the loader
    add("general.start_bytes[3]", C.CONTROLLABLE)
    # byte histograms
    for b in range(256):
        add(f"histogram[0x{b:02x}]", C.DERIVED)
    for h in range(16):
        for b in range(16):
            add(f"byteentropy[H{h},B{b}]", C.DERIVED)
    # strings
    add("strings.numstrings", C.APPEND_ONLY)
    add("strings.avlength", C.DERIVED)
    add("strings.printables", C.APPEND_ONLY)
    for i in range(96):
        add(f"strings.printabledist[{_printable(i)}]", C.DERIVED)
    add("strings.entropy", C.DERIVED)
    for rx in tp.REGEX_NAMES:
        add(f"strings.regex[{rx}]", C.APPEND_ONLY)
    # header
    for sec, key in tp.HEADER_SCALARS:
        add(f"header.{sec}.{key}", _HEADER_SCALAR_CTRL[key])
    for ch in tp.IMAGE_CHARACTERISTICS:
        add(f"header.coff.characteristics.{ch}", C.FIXED if ch in _IMAGE_CHAR_FIXED else C.CONTROLLABLE)
    for ch in tp.DLL_CHARACTERISTICS:
        add(f"header.optional.dll_characteristics.{ch}", C.FIXED if ch in _DLL_CHAR_FIXED else C.CONTROLLABLE)
    for m in tp.DOS_MEMBERS:
        add(f"header.dos.{m}", C.FIXED if m in _DOS_FIXED else C.CONTROLLABLE)
    # section
    add("section.n_sections", C.APPEND_ONLY)
    add("section.n_zero_size", C.APPEND_ONLY)
    add("section.n_empty_name", C.CONTROLLABLE)
    add("section.n_rx", C.APPEND_ONLY)
    add("section.n_w", C.APPEND_ONLY)
    for stat in ("max_entropy", "min_entropy", "max_size_ratio", "min_size_ratio", "max_vsize_ratio", "min_vsize_ratio"):
        add(f"section.{stat}", C.DERIVED)
    # Hashed by section *name*: renaming a section (free) moves its value to another bucket.
    for block in ("sizes_hashed", "vsizes_hashed", "entropy_hashed", "characteristics_hashed"):
        for i in range(50):
            add(f"section.{block}[{i}]", C.CONTROLLABLE)
    for i in range(10):
        add(f"section.entry_name_hashed[{i}]", C.CONTROLLABLE)
    add("section.overlay.size", C.APPEND_ONLY)
    add("section.overlay.size_ratio", C.APPEND_ONLY)
    add("section.overlay.entropy", C.DERIVED)
    # imports
    add("imports.n_functions", C.APPEND_ONLY)
    add("imports.n_libraries", C.APPEND_ONLY)
    for i in range(256):
        add(f"imports.libraries_hashed[{i}]", C.APPEND_ONLY)
    for i in range(1024):
        add(f"imports.functions_hashed[{i}]", C.APPEND_ONLY)
    # exports
    add("exports.nonempty_x128", C.APPEND_ONLY)  # 128 if the file exports anything (upstream quirk)
    for i in range(128):
        add(f"exports.names_hashed[{i}]", C.APPEND_ONLY)
    # data directories
    for d in tp.DATA_DIRECTORY_NAMES:
        add(f"datadirectories.{d}.size", _DATADIR_CTRL[d])
        add(f"datadirectories.{d}.virtual_address", _DATADIR_CTRL[d])
    add("datadirectories.has_relocs", C.DERIVED)
    add("datadirectories.has_dynamic_relocs", C.DERIVED)
    # rich header (ignored by the loader; can be removed or forged)
    add("richheader.n_pairs", C.CONTROLLABLE)
    for i in range(32):
        add(f"richheader.entries_hashed[{i}]", C.CONTROLLABLE)
    # authenticode
    for f in tp.AUTHENTICODE_FIELDS:
        add(f"authenticode.{f}", _AUTH_CTRL[f])
    # pefile warnings
    for w in tp.PEFILE_WARNINGS:
        add(f"pefilewarnings[{w}]", C.DERIVED)
    add("pefilewarnings.count", C.DERIVED)
    return tuple(names), tuple(ctrl)


class EmberV3Schema(FeatureSchema):
    """EMBER2024 feature version 3 (thrember), 2568 features in 12 groups."""

    name = "ember_v3"
    dim = 2568
    description = "EMBER feature version 3 (EMBER2024, thrember)."

    def groups(self) -> list[FeatureGroup]:
        return [FeatureGroup(n, a, b, c, d) for n, a, b, c, d in V3_GROUPS]

    def feature_names(self) -> list[str]:
        return list(_layout()[0])

    def feature_controllability(self) -> list[C]:
        return list(_layout()[1])

    @property
    def categorical_features(self) -> tuple[int, ...]:
        """Feature indices EMBER2024's reference models declare categorical to LightGBM."""
        return CATEGORICAL_FEATURES

    # ---- raw JSON -> vectors -----------------------------------------------------------------

    def vectorize_raw(self, raw: dict[str, Any]) -> np.ndarray:
        """One EMBER2024 raw-feature dict (a thrember JSONL row) -> (2568,) float32."""
        return tp.vectorize(raw)

    def vectorize_raw_batch(self, raws: Sequence[dict[str, Any]]) -> np.ndarray:
        """Many raw-feature dicts -> (n, 2568) float32 (hashers run once per batch)."""
        return tp.vectorize_batch(list(raws))

    # ---- raw bytes -> vector -----------------------------------------------------------------

    def featurize_available(self) -> bool:
        return tp.pefile_available()

    def raw_features(self, raw: bytes) -> dict[str, Any]:
        """Raw PE bytes -> EMBER v3 raw-feature dict (requires pefile)."""
        self._require_pefile()
        return tp.raw_features(raw)

    def featurize(self, raw: bytes) -> np.ndarray:
        """Raw PE bytes -> (2568,) float32, identical to thrember's ``PEFeatureExtractor``.

        Needs ``pefile``; ``signify`` is optional (without it, Authenticode features of *signed*
        files are approximate: only the certificate count is filled in).
        """
        self._require_pefile()
        return tp.vectorize(tp.raw_features(raw))

    def _require_pefile(self) -> None:
        if not tp.pefile_available():
            from malvalid.core import FeaturizeUnavailable

            raise FeaturizeUnavailable(
                "ember_v3 raw-bytes featurization needs the 'pefile' package "
                "(pip install pefile; optionally signify for exact Authenticode features)"
            )

    def info(self) -> dict[str, Any]:
        d = super().info()
        d["categorical_features"] = list(CATEGORICAL_FEATURES)
        d["authenticode_exact"] = tp.signify_available()
        return d
