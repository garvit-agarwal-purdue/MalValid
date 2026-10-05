"""Synthetic EMBER-shaped canonical corpora (``synthetic_v2`` / ``synthetic_v3``) for demos and CI.

These corpora let the whole gate run end to end without the real EMBER data. They are **not**
evidence about any real detector: every report built on them is labelled synthetic
(``manifest["synthetic"] = True`` → ``Corpus.synthetic``), and the numbers only exercise the
pipeline.

How the data is made
--------------------
A seeded *world* defines benign software clusters (MSVC apps, signed system components,
installers, .NET apps, Delphi apps, MinGW tools, Go/Rust binaries, protected games) and malicious
families (packed trojans, injectors, downloaders, ransomware, stealers, .NET RATs, bundlers, Go
malware, virtualised RATs). Each family has a birth month and a lifetime, so new families appear
after the training period, and within a family the "marker" imports, section names and string
habits rotate gradually. Each synthetic file is a latent description — toolchain, packer,
capabilities, sections, imports, exports, strings, header fields, signature, rich header,
parser warnings — drawn from its cluster (blended a little with a cluster of the opposite class
so that the classes overlap) and then *rendered* into the real ``ember_v2`` (2381) or
``ember_v3`` (2568) feature layout:

* byte and byte-entropy histograms are normalised and follow the section composition;
* counts are integral (strings, imports, sections, warnings) and hashed blocks use the same
  MurmurHash3 buckets and signs as the real vectorizers (``lib:function`` tokens, section names,
  ``name:property`` tokens, rich-header comp-ids), so, e.g., ``kernel32.dll`` lands in its real
  ``imports.libraries_hashed`` bucket;
* scalars keep their real relationships (``printables = numstrings × avlength``,
  ``strings.entropy`` = entropy of ``printabledist``, ``sizeof_image`` from the section layout,
  ``has_signature`` ⇔ a security directory ⇔ an overlay, …).

The label signal is spread over many groups (packing and entropy, imports, strings, section
names, header/toolchain, signatures, exports, …). A LightGBM trained on the ``train`` split
reaches a test AUROC of roughly 0.96–0.99, and its F1 decays over the 24 post-training months;
the ``challenge`` split holds evasive, benign-looking malware from the test period.

Splits (monthly, 36 months: ``start`` = 2017-01 for v2, 2022-01 for v3): ``train`` (first 12
months, 80 %), ``holdout`` (the other 20 % of those months, handy as validation data and as
time-matched non-members for M4), ``test`` (months 13–36) and ``challenge`` (malicious, test
period). Roles: eval = [test], temporal = [train, holdout, test], challenge = [challenge],
pool = [train, holdout, test].

Everything is a deterministic function of ``(seed, n, feature space)`` and the numpy version; the
corpus is generated on first load (≈ 3–10 s for the default 20,000 rows) and cached under the
corpus root; if that directory is not writable it is kept in memory with the same content hash.
"""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from malvalid import CORPUS_FORMAT
from malvalid.core import CorpusUnavailable
from malvalid.corpora.base import (
    CORPUS_DIR_FILES,
    ROLE_CHALLENGE,
    ROLE_EVAL,
    ROLE_POOL,
    ROLE_TEMPORAL,
    Corpus,
    CorpusProvider,
    CorpusWriter,
    compute_content_hash,
    full_verification_requested,
    load_corpus_dir,
    manifest_name,
    write_npz_deterministic,
    write_verify_sidecar,
)

if TYPE_CHECKING:  # pragma: no cover
    from malvalid.config import GateConfig

log = logging.getLogger("malvalid.corpora.synthetic")

GENERATOR = "malvalid.corpora.synthetic"
GENERATOR_VERSION = "1"
ENV_ROWS = "MALVALID_SYNTHETIC_ROWS"
ENV_SEED = "MALVALID_SYNTHETIC_SEED"
SPLITS = ("train", "holdout", "test", "challenge")
ROLES: dict[str, list[str]] = {
    ROLE_EVAL: ["test"],
    ROLE_TEMPORAL: ["train", "holdout", "test"],
    ROLE_CHALLENGE: ["challenge"],
    ROLE_POOL: ["train", "holdout", "test"],
}


@dataclass(frozen=True)
class SyntheticParams:
    """Size and shape of a synthetic corpus (all defaults give the canonical demo corpus)."""

    n: int = 20000
    seed: int = 0
    months: int = 36
    train_months: int = 12
    holdout_fraction: float = 0.2
    malicious_fraction: float = 0.5
    challenge_fraction: float = 0.03
    # Class overlap: every row is blended with a cluster of the other class by w ~ Beta(a, b).
    overlap_a: float = 0.5
    overlap_b: float = 6.0

    def validate(self) -> None:
        if self.n < 200:
            raise ValueError(f"synthetic corpus needs at least 200 rows (got n={self.n})")
        if not 2 <= self.train_months < self.months:
            raise ValueError("train_months must be >= 2 and < months")
        for nm in ("holdout_fraction", "malicious_fraction", "challenge_fraction"):
            v = getattr(self, nm)
            if not 0.0 <= v < 1.0:
                raise ValueError(f"{nm} must be in [0, 1) (got {v})")


# ==================================================================================================
# Hashing (MurmurHash3, exactly like sklearn's FeatureHasher used by both real vectorizers)
# ==================================================================================================


@functools.lru_cache(maxsize=65536)
def _mm3(token: str) -> int:
    from sklearn.utils import murmurhash3_32

    return int(murmurhash3_32(token, seed=0))


def hash_bucket(token: str, width: int) -> tuple[int, int]:
    """(bucket, sign) that ``token`` contributes to in a FeatureHasher block of ``width``."""
    h = _mm3(token)
    return abs(h) % width, (1 if h >= 0 else -1)


def _hash_matrix(token_lists: list[list[str]], width: int, signed: bool) -> np.ndarray:
    """(len(token_lists), width) matrix: row k = hashed bag of ``token_lists[k]``."""
    M = np.zeros((len(token_lists), width), dtype=np.float64)
    for k, toks in enumerate(token_lists):
        for t in toks:
            b, s = hash_bucket(t, width)
            M[k, b] += s if signed else 1.0
    return M


# ==================================================================================================
# World vocabulary
# ==================================================================================================

TOOLCHAINS = ("msvc_modern", "msvc_legacy", "delphi", "mingw", "golang", "dotnet", "asm")
T_MSVC, T_LEGACY, T_DELPHI, T_MINGW, T_GO, T_DOTNET, T_ASM = range(len(TOOLCHAINS))
PACKERS = ("none", "upx", "vmprotect", "custom", "installer")
P_NONE, P_UPX, P_VMP, P_CUSTOM, P_INSTALLER = range(len(PACKERS))
CAPS = (
    "gui", "file_io", "registry", "network_http", "network_socket", "crypto", "process_injection",
    "keylog", "service", "anti_debug", "shell", "com", "sysinfo", "multimedia", "printing", "security",
)
CAP = {c: i for i, c in enumerate(CAPS)}
TSMODES = ("real", "zero", "random", "stale", "future")

# (library, vocabulary size). Function names are synthetic except for a few structural ones.
LIBRARIES: tuple[tuple[str, int], ...] = (
    ("kernel32.dll", 240), ("user32.dll", 160), ("advapi32.dll", 90), ("gdi32.dll", 80),
    ("shell32.dll", 40), ("ole32.dll", 40), ("oleaut32.dll", 40), ("msvcrt.dll", 120),
    ("ws2_32.dll", 40), ("wininet.dll", 40), ("ntdll.dll", 60), ("comctl32.dll", 20),
    ("shlwapi.dll", 40), ("crypt32.dll", 30), ("version.dll", 6), ("winmm.dll", 15),
    ("mscoree.dll", 2), ("vcruntime140.dll", 25), ("api-ms-win-crt-runtime-l1-1-0.dll", 40),
    ("api-ms-win-crt-stdio-l1-1-0.dll", 25), ("api-ms-win-crt-heap-l1-1-0.dll", 6),
    ("api-ms-win-crt-string-l1-1-0.dll", 15), ("msvcp140.dll", 40), ("comdlg32.dll", 8),
    ("winhttp.dll", 20), ("urlmon.dll", 6), ("iphlpapi.dll", 10), ("psapi.dll", 10),
    ("dbghelp.dll", 10), ("bcrypt.dll", 12), ("setupapi.dll", 15), ("rpcrt4.dll", 15),
    ("netapi32.dll", 10), ("wtsapi32.dll", 6), ("userenv.dll", 5), ("secur32.dll", 6),
)
_KNOWN_FUNCS: dict[str, tuple[str, ...]] = {
    "kernel32.dll": ("LoadLibraryA", "GetProcAddress", "VirtualProtect", "VirtualAlloc", "VirtualFree",
                     "ExitProcess", "GetModuleHandleA"),
    "mscoree.dll": ("_CorExeMain", "_CorDllMain"),
}
PACKER_STUB = ("LoadLibraryA", "GetProcAddress", "VirtualProtect", "VirtualAlloc", "VirtualFree", "ExitProcess")

CAP_LIBS: dict[str, tuple[tuple[str, float], ...]] = {
    "gui": (("user32.dll", .55), ("gdi32.dll", .5), ("comctl32.dll", .6), ("comdlg32.dll", .6), ("shell32.dll", .1)),
    "file_io": (("kernel32.dll", .12), ("shlwapi.dll", .3), ("shell32.dll", .2)),
    "registry": (("advapi32.dll", .25), ("shlwapi.dll", .1)),
    "network_http": (("wininet.dll", .6), ("winhttp.dll", .6), ("urlmon.dll", .6)),
    "network_socket": (("ws2_32.dll", .7), ("iphlpapi.dll", .5)),
    "crypto": (("crypt32.dll", .5), ("bcrypt.dll", .6), ("advapi32.dll", .1)),
    "process_injection": (("kernel32.dll", .05), ("ntdll.dll", .3), ("psapi.dll", .5)),
    "keylog": (("user32.dll", .08),),
    "service": (("advapi32.dll", .15), ("wtsapi32.dll", .5), ("userenv.dll", .5)),
    "anti_debug": (("kernel32.dll", .03), ("ntdll.dll", .15), ("dbghelp.dll", .4)),
    "shell": (("shell32.dll", .3), ("kernel32.dll", .03)),
    "com": (("ole32.dll", .6), ("oleaut32.dll", .6), ("rpcrt4.dll", .3)),
    "sysinfo": (("kernel32.dll", .06), ("advapi32.dll", .05), ("version.dll", .8), ("netapi32.dll", .4),
                ("setupapi.dll", .3)),
    "multimedia": (("winmm.dll", .7), ("gdi32.dll", .2)),
    "printing": (("gdi32.dll", .15), ("comdlg32.dll", .3)),
    "security": (("secur32.dll", .6), ("advapi32.dll", .1), ("crypt32.dll", .2)),
}
# Toolchain runtime imports: (library, fraction of its vocabulary, mean inclusion probability).
TOOL_LIBS: dict[int, tuple[tuple[str, float, float], ...]] = {
    T_MSVC: (("kernel32.dll", .17, .9), ("vcruntime140.dll", .5, .55), ("api-ms-win-crt-runtime-l1-1-0.dll", .4, .55),
             ("api-ms-win-crt-stdio-l1-1-0.dll", .3, .5), ("api-ms-win-crt-heap-l1-1-0.dll", .6, .55),
             ("api-ms-win-crt-string-l1-1-0.dll", .3, .4), ("msvcp140.dll", .3, .25)),
    T_LEGACY: (("kernel32.dll", .2, .85), ("msvcrt.dll", .25, .45), ("user32.dll", .05, .5)),
    T_DELPHI: (("kernel32.dll", .35, .9), ("user32.dll", .3, .85), ("oleaut32.dll", .3, .8), ("advapi32.dll", .08, .8),
               ("version.dll", .6, .8), ("gdi32.dll", .25, .7), ("ole32.dll", .1, .6), ("comctl32.dll", .2, .6)),
    T_MINGW: (("kernel32.dll", .12, .9), ("msvcrt.dll", .4, .8)),
    T_GO: (("kernel32.dll", .25, .97),),
    T_DOTNET: (),
    T_ASM: (("kernel32.dll", .04, .8), ("user32.dll", .03, .6)),
}
CAP_IMPORT_SCALE = np.array([1.0, 1.0, 1.0, 1.0, 0.04, 0.0, 0.6])  # per toolchain

REGEX_TIERS: dict[str, float] = {  # base count per 1000 strings
    **{k: 2.5 for k in ("file", "get", "create", "process", "thread", "module", "memory", "system", "window",
                        "resource", "exit", "directory", "enum", "delete", "security", "debug", "environment")},
    **{k: 0.8 for k in ("command", "install", "cache", "disk", "service", "token", "privilege", "hidden",
                        "desktop", "keyboard", "clipboard", "remote", "shell", "connect", "internet", "download",
                        "post", "http", "url", "html", "encode", "decode", "crypt", "certificate", "password",
                        "cookie", "mutex", "snapshot", "hostname", "file_path", "registry_key", "https://",
                        "http://", "base64")},
    **{k: 0.15 for k in ("useragent", "email_addr", "ipv4_addr", "base64string", "powershell", "ftp",
                         "Invoke-Expression", "Invoke-Command", "Start-process", "<script", "javascript",
                         "wallet", "mac_addr", "ipv6_addr", "btc_wallet")},
    **{k: 0.02 for k in (".click(", "onlick", "/FlateDecode", "/EmbeddedFile", "/URI", "/bin/", "/dev/",
                         "/proc/", "/tmp/", "/usr/")},
}
CAP_REGEX: dict[str, tuple[str, ...]] = {
    "gui": ("window", "desktop", "resource", "clipboard"),
    "file_io": ("file", "file_path", "directory", "delete", "create", "disk", "enum", "cache"),
    "registry": ("registry_key", "install", "security"),
    "network_http": ("http", "http://", "https://", "url", "internet", "download", "post", "get", "useragent",
                     "cookie", "hostname", "html"),
    "network_socket": ("connect", "ipv4_addr", "hostname", "remote", "ftp", "mac_addr"),
    "crypto": ("crypt", "encode", "decode", "base64", "base64string", "certificate", "wallet", "btc_wallet"),
    "process_injection": ("process", "thread", "memory", "remote", "snapshot", "module", "privilege", "token"),
    "keylog": ("keyboard", "clipboard", "window", "hidden"),
    "service": ("service", "install", "system"),
    "anti_debug": ("debug", "mutex", "exit", "hidden", "snapshot"),
    "shell": ("shell", "command", "powershell", "Invoke-Expression", "Invoke-Command", "Start-process", "environment"),
    "com": ("create", "module", "system"),
    "sysinfo": ("system", "environment", "hostname", "disk"),
    "multimedia": ("resource",),
    "printing": ("resource", "window"),
    "security": ("security", "password", "token", "certificate", "privilege"),
}

# Section concepts: (name, kind, props, entropy mean, entropy sd, relative raw-size weight).
_CODE = ("CNT_CODE", "MEM_EXECUTE", "MEM_READ")
_RDATA = ("CNT_INITIALIZED_DATA", "MEM_READ")
_DATA = ("CNT_INITIALIZED_DATA", "MEM_READ", "MEM_WRITE")
_RELOC = ("CNT_INITIALIZED_DATA", "MEM_DISCARDABLE", "MEM_READ")
_UNINIT = ("CNT_UNINITIALIZED_DATA", "MEM_READ", "MEM_WRITE")
_UPX0 = ("CNT_UNINITIALIZED_DATA", "MEM_EXECUTE", "MEM_READ", "MEM_WRITE")
_PACKED = ("CNT_INITIALIZED_DATA", "MEM_EXECUTE", "MEM_READ", "MEM_WRITE")
_PCODE = ("CNT_CODE", "CNT_INITIALIZED_DATA", "MEM_EXECUTE", "MEM_READ", "MEM_WRITE")
KINDS = ("code", "rdata", "data", "rsrc", "reloc", "pdata", "uninit", "packed", "tls", "idata")
BASE_SECTIONS: tuple[tuple[str, str, tuple[str, ...], float, float, float], ...] = (
    (".text", "code", _CODE, 6.3, 0.35, 3.0),
    (".rdata", "rdata", _RDATA, 4.9, 0.6, 1.2),
    (".data", "data", _DATA, 2.4, 1.2, 0.5),
    (".rsrc", "rsrc", _RDATA, 4.8, 1.4, 1.0),
    (".reloc", "reloc", _RELOC, 5.4, 0.5, 0.15),
    (".pdata", "pdata", _RDATA, 5.1, 0.4, 0.2),
    (".idata", "idata", _DATA, 4.1, 0.6, 0.08),
    (".tls", "tls", _DATA, 0.2, 0.2, 0.01),
    (".bss", "uninit", _UNINIT, 0.0, 0.0, 0.0),
    ("CODE", "code", _CODE, 6.5, 0.3, 3.0),
    ("DATA", "data", _DATA, 4.3, 0.8, 0.4),
    ("BSS", "uninit", _UNINIT, 0.0, 0.0, 0.0),
    (".edata", "rdata", _RDATA, 4.5, 0.6, 0.05),
    (".CRT", "data", _DATA, 0.3, 0.2, 0.01),
    (".gfids", "rdata", _RDATA, 1.5, 0.6, 0.01),
    (".00cfg", "rdata", _RDATA, 0.5, 0.3, 0.005),
    (".symtab", "rdata", _RDATA, 5.0, 0.5, 0.3),
    (".xdata", "rdata", _RDATA, 4.5, 0.5, 0.05),
    ("UPX0", "uninit", _UPX0, 0.0, 0.0, 0.0),
    ("UPX1", "packed", _PACKED, 7.9, 0.06, 3.0),
    (".vmp0", "packed", _PCODE, 7.7, 0.2, 2.0),
    (".vmp1", "packed", _PCODE, 7.9, 0.08, 2.5),
    (".ndata", "uninit", _UNINIT, 0.0, 0.0, 0.0),
    ("", "packed", _PCODE, 7.8, 0.15, 2.5),
    (".sdata", "data", _DATA, 3.0, 0.8, 0.02),
    (".didat", "data", _DATA, 1.5, 0.6, 0.01),
    ("_RDATA", "rdata", _RDATA, 2.5, 0.8, 0.01),
)
SEC_INDEX = {s[0]: i for i, s in enumerate(BASE_SECTIONS)}
N_FAMILY_SECTIONS = 3  # custom section names per malicious family
# Relative composition of each section kind: (zero padding, x86 code, text/tables, compressed, resource)
KIND_COMPOSITION = {
    "code": (0.10, 0.80, 0.07, 0.00, 0.03), "rdata": (0.28, 0.12, 0.45, 0.02, 0.13),
    "data": (0.60, 0.05, 0.25, 0.00, 0.10), "rsrc": (0.18, 0.00, 0.15, 0.27, 0.40),
    "reloc": (0.30, 0.40, 0.10, 0.00, 0.20), "pdata": (0.22, 0.48, 0.00, 0.00, 0.30),
    "uninit": (1.00, 0.00, 0.00, 0.00, 0.00), "packed": (0.00, 0.02, 0.00, 0.98, 0.00),
    "tls": (0.95, 0.00, 0.00, 0.00, 0.05), "idata": (0.40, 0.00, 0.50, 0.00, 0.10),
}
# Probability that each toolchain lays out a base section (row-level flags handled separately).
_TOOL_SECTIONS: dict[int, dict[str, float]] = {
    T_MSVC: {".text": 1, ".rdata": 1, ".data": .97, ".gfids": .2, ".00cfg": .3, "_RDATA": .1, ".didat": .05},
    T_LEGACY: {".text": 1, ".rdata": .95, ".data": 1, ".idata": .1},
    T_DELPHI: {"CODE": .7, ".text": .3, "DATA": .7, ".data": .3, "BSS": .7, ".bss": .3, ".idata": 1, ".rdata": 1,
               ".edata": .1, ".didat": .3},
    T_MINGW: {".text": 1, ".data": 1, ".rdata": 1, ".bss": 1, ".idata": 1, ".CRT": 1, ".xdata": .5},
    T_GO: {".text": 1, ".rdata": 1, ".data": 1, ".idata": 1, ".symtab": 1},
    T_DOTNET: {".text": 1, ".sdata": .03},
    T_ASM: {".text": 1, ".data": .8, ".rdata": .4, ".idata": .3},
}
_PACKER_SECTIONS: dict[int, dict[str, float]] = {
    P_NONE: {}, P_UPX: {"UPX0": 1, "UPX1": 1}, P_VMP: {".vmp0": 1, ".vmp1": 1},
    P_CUSTOM: {"": .35}, P_INSTALLER: {".ndata": 1},
}
PACKER_REPLACES_LAYOUT = np.array([False, True, False, True, False])
ENTRY_PRIORITY = {".text": 5, "CODE": 5, "UPX1": 9, ".vmp0": 9, "": 7}

DATA_DIRS = ("EXPORT", "IMPORT", "RESOURCE", "EXCEPTION", "SECURITY", "BASERELOC", "DEBUG", "COPYRIGHT",
             "GLOBALPTR", "TLS", "LOAD_CONFIG", "BOUND_IMPORT", "IAT", "DELAY_IMPORT", "COM_DESCRIPTOR", "RESERVED")
DD = {d: i for i, d in enumerate(DATA_DIRS)}
COFF_FLAGS = ("RELOCS_STRIPPED", "EXECUTABLE_IMAGE", "LINE_NUMS_STRIPPED", "LOCAL_SYMS_STRIPPED", "AGGRESIVE_WS_TRIM",
              "LARGE_ADDRESS_AWARE", "16BIT_MACHINE", "BYTES_REVERSED_LO", "32BIT_MACHINE", "DEBUG_STRIPPED",
              "REMOVABLE_RUN_FROM_SWAP", "NET_RUN_FROM_SWAP", "SYSTEM", "DLL", "UP_SYSTEM_ONLY", "BYTES_REVERSED_HI")
DLL_FLAGS = ("HIGH_ENTROPY_VA", "DYNAMIC_BASE", "FORCE_INTEGRITY", "NX_COMPAT", "NO_ISOLATION", "NO_SEH",
             "NO_BIND", "APPCONTAINER", "WDM_DRIVER", "GUARD_CF", "TERMINAL_SERVER_AWARE")
CF = {f: i for i, f in enumerate(COFF_FLAGS)}
DF = {f: i for i, f in enumerate(DLL_FLAGS)}
# ember_v2 (LIEF 0.9) spells two COFF flags differently and has no 16BIT_MACHINE.
_V2_COFF_TOKEN = {"32BIT_MACHINE": "CHARA_32BIT_MACHINE", "AGGRESIVE_WS_TRIM": "AGGRESSIVE_WS_TRIM",
                  "16BIT_MACHINE": None}


# ==================================================================================================
# Clusters (benign software kinds and malicious families)
# ==================================================================================================

_BASE_PROFILE: dict[str, Any] = {
    "tool": {T_MSVC: 1.0}, "packer": {P_NONE: 1.0}, "caps": {}, "tsmode": {"real": 1.0},
    "log_size": 12.0, "log_size_sd": 1.1, "p64": 0.3, "p_dll": 0.2, "p_gui": 0.65, "p_signed": 0.2,
    "p_overlay": 0.08, "overlay_ratio": 0.15, "p_debug": 0.6, "p_resources": 0.85, "p_relocs": 0.6,
    "p_tls": 0.1, "p_rich_strip": 0.02, "p_checksum": 0.3, "p_image_version": 0.15, "str_density": 1.6,
    "avlen": 2.5, "obf": 0.05, "warn_rate": 0.5, "exp_mu": 2.5, "lag": 5.5, "p_self_signed": 0.03,
    "p_no_countersig": 0.02, "p_modern": 0.8, "marker_p": 0.0, "bump": 0.03,
}
BENIGN_CLUSTERS: tuple[tuple[str, float, float, dict[str, Any]], ...] = (
    # name, weight at start, weight change per year, profile overrides
    ("msvc_app", 0.30, 0.0, dict(tool={T_MSVC: .75, T_LEGACY: .25}, p64=.3, p_dll=.25, p_signed=.45,
     caps=dict(gui=.8, file_io=.9, registry=.6, network_http=.3, crypto=.2, com=.5, printing=.2, sysinfo=.6,
               shell=.3, multimedia=.2, security=.2), log_size=12.6)),
    ("system_component", 0.18, 0.0, dict(tool={T_MSVC: 1.0}, p64=.65, p_dll=.7, p_signed=.95, p_checksum=.95,
     p_image_version=.9, tsmode=dict(real=.3, random=.7), caps=dict(file_io=.8, registry=.8, com=.7, security=.5,
     sysinfo=.7, service=.4, crypto=.3, network_socket=.2), exp_mu=3.5, log_size=11.8, p_modern=.95)),
    ("installer", 0.12, 0.0, dict(tool={T_LEGACY: .55, T_DELPHI: .45}, packer={P_INSTALLER: .9, P_NONE: .1},
     p64=.02, p_dll=0.0, p_signed=.65, p_overlay=.97, overlay_ratio=.85, caps=dict(gui=.9, file_io=.95,
     registry=.8, shell=.8, com=.6, sysinfo=.5, network_http=.15), log_size=14.5, log_size_sd=1.3,
     p_modern=.5)),
    ("dotnet_app", 0.12, 0.0, dict(tool={T_DOTNET: 1.0}, p64=.1, p_dll=.4, p_signed=.3,
     caps=dict(gui=.6, file_io=.8, network_http=.4, crypto=.3, registry=.4, sysinfo=.4), log_size=11.5,
     tsmode=dict(real=.5, random=.5))),
    ("delphi_app", 0.06, -0.01, dict(tool={T_DELPHI: 1.0}, p64=.1, p_dll=.1, p_signed=.2, p_relocs=.1,
     caps=dict(gui=.95, file_io=.9, registry=.7, com=.5, printing=.3, network_socket=.2), log_size=14.0,
     tsmode=dict(real=.4, stale=.6), p_modern=.3)),
    ("mingw_tool", 0.07, 0.0, dict(tool={T_MINGW: 1.0}, p64=.4, p_dll=.3, p_gui=.3, p_signed=.05, p_debug=.2,
     p_resources=.3, caps=dict(file_io=.95, network_socket=.3, crypto=.2, sysinfo=.4), log_size=12.2,
     p_modern=.4)),
    ("go_rust", 0.03, 0.045, dict(tool={T_GO: 1.0}, p64=.8, p_dll=.02, p_gui=.3, p_signed=.25, p_resources=.2,
     caps=dict(file_io=.9, network_http=.6, crypto=.6, network_socket=.4), log_size=15.2, log_size_sd=0.8,
     tsmode=dict(zero=1.0), str_density=1.4, avlen=3.1, p_debug=0.0)),
    ("protected_game", 0.05, 0.0, dict(tool={T_MSVC: 1.0}, packer={P_VMP: .6, P_NONE: .4}, p64=.7, p_signed=.8,
     caps=dict(gui=.9, multimedia=.95, file_io=.9, network_socket=.6, anti_debug=.6, crypto=.4, registry=.5),
     log_size=15.5, p_overlay=.2)),
)
ARCHETYPES: dict[str, dict[str, Any]] = {
    "packed_trojan": dict(tool={T_LEGACY: .3, T_MINGW: .2, T_DELPHI: .2, T_MSVC: .3},
                          packer={P_NONE: .2, P_UPX: .6, P_CUSTOM: .2},
                          caps=dict(file_io=.8, registry=.7, network_http=.6, shell=.5, sysinfo=.6),
                          p_signed=.03, log_size=12.3, tsmode=dict(real=.4, zero=.2, stale=.3, random=.1)),
    "injector": dict(tool={T_MSVC: .6, T_LEGACY: .3, T_ASM: .1}, packer={P_NONE: .5, P_CUSTOM: .3, P_VMP: .2},
                     caps=dict(process_injection=.9, anti_debug=.7, sysinfo=.6, file_io=.6, registry=.5),
                     p_signed=.05, log_size=12.0, tsmode=dict(real=.5, zero=.2, stale=.2, future=.1)),
    "downloader": dict(tool={T_MSVC: .4, T_LEGACY: .3, T_ASM: .3}, packer={P_NONE: .7, P_UPX: .2, P_CUSTOM: .1},
                       caps=dict(network_http=.95, shell=.8, file_io=.7), p_signed=.04, log_size=11.0,
                       p_resources=.4, tsmode=dict(real=.5, zero=.2, stale=.2, random=.1)),
    "ransomware": dict(tool={T_MSVC: .7, T_MINGW: .15, T_GO: .15}, packer={P_NONE: .8, P_CUSTOM: .2},
                       caps=dict(crypto=.95, file_io=.95, shell=.6, sysinfo=.7, anti_debug=.3),
                       p_signed=.05, log_size=12.8),
    "stealer": dict(tool={T_MSVC: .5, T_LEGACY: .2, T_DELPHI: .15, T_GO: .15},
                    packer={P_NONE: .6, P_UPX: .2, P_CUSTOM: .2},
                    caps=dict(crypto=.7, network_http=.8, registry=.8, keylog=.5, file_io=.8, security=.6),
                    p_signed=.05, log_size=13.4),
    "dotnet_rat": dict(tool={T_DOTNET: 1.0}, packer={P_NONE: .9, P_CUSTOM: .1},
                       caps=dict(network_http=.8, keylog=.6, shell=.6, registry=.6, crypto=.5, network_socket=.5),
                       p_signed=.04, log_size=12.5, obf=.5, tsmode=dict(real=.4, random=.4, future=.2)),
    "adware_bundler": dict(tool={T_LEGACY: .4, T_DELPHI: .3, T_MSVC: .3}, packer={P_INSTALLER: .8, P_NONE: .2},
                           caps=dict(network_http=.8, registry=.7, com=.5, gui=.8, file_io=.8, shell=.5),
                           p_signed=.5, p_self_signed=.3, p_no_countersig=.3, p_overlay=.85, overlay_ratio=.8,
                           log_size=14.2),
    "go_malware": dict(tool={T_GO: 1.0}, packer={P_NONE: .8, P_UPX: .2},
                       caps=dict(network_http=.9, crypto=.6, shell=.7, file_io=.8, network_socket=.5),
                       p_signed=.03, log_size=15.0, log_size_sd=.8, tsmode=dict(zero=1.0), p_debug=0.0,
                       p_resources=.1),
    "vmprotected_rat": dict(tool={T_MSVC: 1.0}, packer={P_VMP: .9, P_NONE: .1},
                            caps=dict(process_injection=.7, anti_debug=.9, keylog=.5, network_http=.7,
                                      network_socket=.5), p_signed=.1, log_size=14.0),
}
_MAL_COMMON: dict[str, Any] = dict(p_gui=.55, p_dll=.12, p_debug=.2, p_resources=.7, p_checksum=.1,
                                   p_image_version=.05, p_rich_strip=.15, warn_rate=1.3, lag=3.6, marker_p=.7,
                                   exp_mu=1.2, p_modern=.45, str_density=1.9, avlen=2.2, bump=.06,
                                   p_self_signed=.35, p_no_countersig=.3, p_overlay=.15, overlay_ratio=.3)
# (archetype, birth month, lifetime in months, marker rotation period in months)
FAMILIES: tuple[tuple[str, int, int, int], ...] = (
    ("packed_trojan", -24, 40, 8), ("injector", -18, 36, 9), ("downloader", -12, 60, 7),
    ("stealer", -6, 28, 6), ("adware_bundler", -10, 50, 10), ("dotnet_rat", 2, 26, 7),
    ("ransomware", 6, 32, 8), ("packed_trojan", 13, 23, 6), ("vmprotected_rat", 15, 22, 8),
    ("go_malware", 18, 18, 6), ("stealer", 21, 16, 5), ("dotnet_rat", 25, 12, 5),
    ("downloader", 28, 9, 4), ("ransomware", 31, 6, 4),
)
_SCALARS = ("log_size", "log_size_sd", "p64", "p_dll", "p_gui", "p_signed", "p_overlay", "overlay_ratio", "p_debug",
            "p_resources", "p_relocs", "p_tls", "p_rich_strip", "p_checksum", "p_image_version", "str_density",
            "avlen", "obf", "warn_rate", "exp_mu", "lag", "p_self_signed", "p_no_countersig", "p_modern", "bump")
_SC = {k: i for i, k in enumerate(_SCALARS)}


def _vec(d: dict[Any, float], keys: tuple[Any, ...] | range, default: float = 0.0) -> np.ndarray:
    return np.array([float(d.get(k, default)) for k in keys], dtype=np.float64)


class _World:
    """Everything that is fixed for a seed: vocabularies, hashes, cluster profiles over time."""

    def __init__(self, seed: int, months: int):
        rng = np.random.default_rng([seed, 0x5EED, 1])
        self.months = months
        # ---- import vocabulary ----------------------------------------------------------------
        tokens: list[str] = []
        lib_of: list[int] = []
        self.lib_names = [name for name, _ in LIBRARIES]
        self.lib_range: dict[str, tuple[int, int]] = {}
        for li, (lib, size) in enumerate(LIBRARIES):
            start = len(tokens)
            known = _KNOWN_FUNCS.get(lib, ())
            names = list(known) + [f"{lib.split('.')[0].capitalize()}Fn{k:03d}" for k in range(size - len(known))]
            for nm in names[:size]:
                tokens.append(f"{lib}:{nm}")
                lib_of.append(li)
            self.lib_range[lib] = (start, len(tokens))
        self.func_tokens = tokens
        self.F = len(tokens)
        self.L = len(LIBRARIES)
        self.lib_of = np.array(lib_of, dtype=np.int64)
        self.func_index = {t: i for i, t in enumerate(tokens)}
        self.stub = np.array([self.func_index[f"kernel32.dll:{f}"] for f in PACKER_STUB])
        self.cor_exe = self.func_index["mscoree.dll:_CorExeMain"]
        self.cor_dll = self.func_index["mscoree.dll:_CorDllMain"]

        def pick(lib: str, frac: float) -> np.ndarray:
            a, b = self.lib_range[lib]
            k = max(1, int(round(frac * (b - a))))
            return np.sort(rng.choice(np.arange(a, b), size=k, replace=False))

        self.log1m_cap = np.zeros((len(CAPS), self.F))
        for c, libs in CAP_LIBS.items():
            for lib, frac in libs:
                idx = pick(lib, frac)
                q = rng.beta(2.0, 3.0, size=idx.size) * 0.9
                self.log1m_cap[CAP[c], idx] += np.log1p(-q)
        self.log1m_tool = np.zeros((len(TOOLCHAINS), self.F))
        for t, libs in TOOL_LIBS.items():
            for lib, frac, qm in libs:
                idx = pick(lib, frac)
                q = np.clip(rng.beta(qm * 8, (1 - qm) * 8 + 1e-3, size=idx.size), 0.01, 0.995)
                self.log1m_tool[t, idx] += np.log1p(-q)
        lib_onehot = np.zeros((self.F, self.L), dtype=np.float32)
        lib_onehot[np.arange(self.F), self.lib_of] = 1.0
        self.lib_onehot = lib_onehot
        # ---- exports -------------------------------------------------------------------------
        common = ["DllRegisterServer", "DllUnregisterServer", "DllGetClassObject", "DllCanUnloadNow",
                  "DllInstall", "ServiceMain", "DllMain", "Initialize", "Uninitialize", "GetVersion",
                  "Start", "Stop", "Run", "Init", "Register", "Unregister"]
        self.exp_tokens = common + [f"Export{k:04d}" for k in range(384)]
        self.E = len(self.exp_tokens)
        pop = 1.0 / np.arange(1, self.E + 1) ** 0.8
        self.exp_pop = pop / pop.sum()
        # ---- sections ------------------------------------------------------------------------
        sections = list(BASE_SECTIONS)
        letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
        n_fam = len(FAMILIES)
        self.family_sections = np.zeros((n_fam, N_FAMILY_SECTIONS), dtype=np.int64)
        for f in range(n_fam):
            for j in range(N_FAMILY_SECTIONS):
                nm = "." + "".join(rng.choice(letters, size=int(rng.integers(3, 7))))
                kind = ("packed", "code", "data")[j % 3]
                props = {"packed": _PCODE, "code": _CODE, "data": _DATA}[kind]
                ent = {"packed": 7.85, "code": 6.4, "data": 3.5}[kind]
                self.family_sections[f, j] = len(sections)
                sections.append((nm, kind, props, ent, 0.15 if kind == "packed" else 0.5, 2.0 if kind != "data" else 0.3))
        self.sections = sections
        self.S = len(sections)
        self.sec_names = [s[0] for s in sections]
        self.sec_kind = np.array([KINDS.index(s[1]) for s in sections])
        self.sec_props = [s[2] for s in sections]
        self.sec_ent = np.array([s[3] for s in sections])
        self.sec_ent_sd = np.array([s[4] for s in sections])
        self.sec_w = np.array([s[5] for s in sections])
        self.sec_code = np.array(["CNT_CODE" in p for p in self.sec_props])
        self.sec_rx = np.array([("MEM_READ" in p and "MEM_EXECUTE" in p) for p in self.sec_props])
        self.sec_w_flag = np.array(["MEM_WRITE" in p for p in self.sec_props])
        self.sec_uninit = self.sec_kind == KINDS.index("uninit")
        self.sec_initdata = np.array(["CNT_INITIALIZED_DATA" in p for p in self.sec_props]) & ~self.sec_code
        self.sec_comp = np.array([KIND_COMPOSITION[KINDS[k]] for k in self.sec_kind])  # (S, 5)
        self.entry_priority = np.array([ENTRY_PRIORITY.get(s[0], 0) for s in sections], dtype=np.float64)
        self.entry_priority[self.family_sections[:, :2].ravel()] = 8.0
        self.tool_sec = np.zeros((len(TOOLCHAINS), self.S))
        for t, d in _TOOL_SECTIONS.items():
            for nm, p in d.items():
                self.tool_sec[t, SEC_INDEX[nm]] = p
        self.packer_sec = np.zeros((len(PACKERS), self.S))
        for p_, d in _PACKER_SECTIONS.items():
            for nm, p in d.items():
                self.packer_sec[p_, SEC_INDEX[nm]] = p
        # ---- strings -------------------------------------------------------------------------
        self.regex_names = sorted(REGEX_TIERS)  # == thrember's sorted REGEX_NAMES minus dos_msg
        self.regex_log_base = np.log(np.array([REGEX_TIERS[r] for r in self.regex_names]))
        ri = {r: i for i, r in enumerate(self.regex_names)}
        self.cap_regex = np.zeros((len(CAPS), len(self.regex_names)))
        for c, rs in CAP_REGEX.items():
            for r in rs:
                self.cap_regex[CAP[c], ri[r]] = 1.3
        self.printable_english = _english_printables()
        self.printable_b64 = _b64_printables()
        # ---- byte histograms -----------------------------------------------------------------
        self.hist_profiles = _hist_profiles(rng)  # (5, 256)
        self.entropy_bins = np.array([  # (5, 16) distribution over entropy bins per component
            _bump16(0.3, 0.8), _bump16(10.8, 1.0), _bump16(8.6, 1.1), _bump16(15.0, 0.4), _bump16(10.5, 2.5),
        ])
        # ---- warnings ------------------------------------------------------------------------
        w = rng.gamma(0.25, 1.0, size=87)
        self.warn_base = w / w.sum()
        # ---- rich header comp-ids (per msvc toolchain and build year) --------------------------
        self.rich_years = np.arange(2008, 2027)
        self.rich_pool = 16
        self.rich_H = np.zeros((2, self.rich_years.size, self.rich_pool, 32))
        prodids = rng.choice(np.arange(0x00F0, 0x0110), size=(2, self.rich_pool))
        for t in range(2):
            for yi, year in enumerate(self.rich_years):
                build = 20000 + (int(year) - 2008) * 900 + (0 if t == 0 else -9000)
                for k in range(self.rich_pool):
                    comp = (int(prodids[t, k]) << 16) | max(1, build + int(rng.integers(0, 40)))
                    b, s = hash_bucket(str(comp), 32)
                    self.rich_H[t, yi, k, b] = s
        # ---- clusters ------------------------------------------------------------------------
        self._build_clusters(rng)
        self._build_hashes()

    # ------------------------------------------------------------------------------------------

    def _profile(self, overrides: dict[str, Any], base: dict[str, Any] | None = None) -> dict[str, Any]:
        p = dict(_BASE_PROFILE)
        if base:
            p.update(base)
        p.update(overrides)
        return p

    def _build_clusters(self, rng: np.random.Generator) -> None:
        M = self.months
        months = np.arange(M)
        profiles: list[dict[str, Any]] = []
        labels: list[int] = []
        weights: list[np.ndarray] = []
        names: list[str] = []
        births: list[float] = []
        periods: list[int] = []
        for name, w0, dw, ov in BENIGN_CLUSTERS:
            profiles.append(self._profile(ov))
            labels.append(0)
            weights.append(np.maximum(w0 + dw * months / 12.0, 0.005))
            names.append(name)
            births.append(-120.0)
            periods.append(0)
        for fi, (arch, birth, life, period) in enumerate(FAMILIES):
            prof = self._profile(ARCHETYPES[arch], _MAL_COMMON)
            prof = dict(prof)
            prof["caps"] = {k: float(np.clip(v * rng.uniform(0.8, 1.15), 0.02, 0.98)) for k, v in prof["caps"].items()}
            prof["log_size"] = prof["log_size"] + rng.normal(0, 0.3)
            prof["p_signed"] = float(np.clip(prof["p_signed"] * rng.uniform(0.5, 1.5), 0, 0.9))
            profiles.append(prof)
            labels.append(1)
            end = birth + life
            ramp_up = np.clip((months - birth + 1) / 3.0, 0, 1)
            ramp_dn = np.clip((end - months) / 3.0, 0, 1)
            weights.append(ramp_up * ramp_dn * rng.uniform(0.7, 1.3))
            names.append(f"{arch}_{fi:02d}")
            births.append(float(birth))
            periods.append(period)
        self.cluster_names = names
        self.cluster_label = np.array(labels, dtype=np.int8)
        self.K = len(profiles)
        W = np.array(weights)  # (K, M)
        for lab in (0, 1):
            sel = self.cluster_label == lab
            tot = W[sel].sum(0)
            if np.any(tot <= 0):
                raise RuntimeError("synthetic world: a month has no active cluster")
            W[sel] = W[sel] / tot
        self.cluster_weight = W
        K, F, S, E = self.K, self.F, self.S, self.E
        R = len(self.regex_names)
        T = {
            "tool": np.zeros((K, M, len(TOOLCHAINS))),
            "packer": np.zeros((K, M, len(PACKERS))),
            "caps": np.zeros((K, M, len(CAPS))),
            "tsmode": np.zeros((K, M, len(TSMODES))),
            "scal": np.zeros((K, M, len(_SCALARS))),
            "marker_imp": np.zeros((K, M, F), dtype=np.float32),
            "marker_sec": np.zeros((K, M, S), dtype=np.float32),
            "regex_bump": np.zeros((K, M, R)),
            "hist_bump": np.zeros((K, M, 256)),
            "print_bump": np.zeros((K, M, 96)),
            "warn_dist": np.zeros((K, M, 87)),
            "exp_dist": np.zeros((K, M, E), dtype=np.float32),
        }
        years = months / 12.0
        for k, prof in enumerate(profiles):
            mal = labels[k] == 1
            age = np.maximum(months - births[k], 0) / 12.0 if mal else years
            tool = _vec(prof["tool"], range(len(TOOLCHAINS)))
            packer = _vec(prof["packer"], range(len(PACKERS)))
            caps = _vec(prof["caps"], CAPS, 0.02)
            tsm = _vec(prof["tsmode"], TSMODES)
            scal = np.array([float(prof[s]) for s in _SCALARS])
            d_caps = rng.normal(0, 0.08, size=len(CAPS)) if mal else np.zeros(len(CAPS))
            d_pack = rng.normal(0, 0.5, size=len(PACKERS)) if mal else np.zeros(len(PACKERS))
            hb0 = rng.dirichlet(np.full(256, 0.08))
            hb1 = rng.dirichlet(np.full(256, 0.08))
            pb0 = rng.dirichlet(np.full(96, 0.2))
            rb0 = rng.normal(0, 0.6, size=R) * (rng.random(R) < 0.3)
            rb1 = rng.normal(0, 0.6, size=R) * (rng.random(R) < 0.3)
            wd = rng.dirichlet(np.full(87, 0.3))
            ed = rng.dirichlet(np.full(E, 0.05))
            cap_libs = [lib for c, v in prof["caps"].items() if v > 0.4 for lib, _ in CAP_LIBS[c]] or ["kernel32.dll"]
            pool = np.unique(np.concatenate([np.arange(*self.lib_range[lib]) for lib in cap_libs]))
            n_versions = 12
            marker_seq = rng.choice(pool, size=min(pool.size, 10 + 3 * n_versions), replace=False)
            life_months = max(1.0, (FAMILIES[k - len(BENIGN_CLUSTERS)][2] if mal else 120.0))
            for m in months:
                a = float(age[m])
                if mal:
                    t_ = tool
                    c_ = np.clip(caps + d_caps * a, 0.02, 0.98)
                    p_ = packer * np.exp(d_pack * a)
                    s_ = scal.copy()
                    s_[_SC["p_signed"]] = min(0.9, s_[_SC["p_signed"]] + 0.03 * a)
                    s_[_SC["log_size"]] += 0.12 * a
                    frac = min(1.0, a * 12.0 / life_months)
                    hb = (1 - frac) * hb0 + frac * hb1
                    rb = (1 - frac) * rb0 + frac * rb1
                    v = int(min(n_versions - 1, max(0.0, a * 12.0) // max(periods[k], 1)))
                    mk = marker_seq[3 * v: 3 * v + 10]
                    T["marker_imp"][k, m, mk] = prof["marker_p"]
                    fs = self.family_sections[k - len(BENIGN_CLUSTERS)]
                    T["marker_sec"][k, m, fs[:2] if v < 3 else fs[1:]] = 0.6
                else:
                    t_ = tool.copy()
                    if t_[T_LEGACY] > 0 and t_[T_MSVC] > 0:
                        moved = t_[T_LEGACY] * (1 - np.exp(-0.35 * a))
                        t_[T_LEGACY] -= moved
                        t_[T_MSVC] += moved
                    c_ = caps
                    p_ = packer
                    s_ = scal.copy()
                    s_[_SC["p64"]] = min(0.95, s_[_SC["p64"]] + 0.08 * a)
                    s_[_SC["p_signed"]] = min(0.98, s_[_SC["p_signed"]] + 0.02 * a)
                    s_[_SC["log_size"]] += 0.06 * a
                    s_[_SC["p_modern"]] = min(0.98, s_[_SC["p_modern"]] + 0.04 * a)
                    hb, rb = hb0, rb0 * 0.5
                T["tool"][k, m] = t_ / t_.sum()
                T["packer"][k, m] = p_ / p_.sum()
                T["caps"][k, m] = c_
                T["tsmode"][k, m] = tsm / tsm.sum()
                T["scal"][k, m] = s_
                T["regex_bump"][k, m] = rb
                T["hist_bump"][k, m] = hb
                T["print_bump"][k, m] = pb0
                T["warn_dist"][k, m] = 0.6 * self.warn_base + 0.4 * wd
                T["exp_dist"][k, m] = 0.5 * self.exp_pop + 0.5 * ed
        self.tables = T

    def _build_hashes(self) -> None:
        """Bucket matrices for every hashed block of both feature versions."""
        lib_tok = [lib.lower() for lib in self.lib_names]
        self.h_lib_u256 = _hash_matrix([[t] for t in lib_tok], 256, signed=False)
        self.h_lib_s256 = _hash_matrix([[t] for t in lib_tok], 256, signed=True)
        ftok = [t.split(":", 1)[0].lower() + ":" + t.split(":", 1)[1] for t in self.func_tokens]
        self.h_fn_u1024 = _hash_matrix([[t] for t in ftok], 1024, signed=False)
        self.h_fn_s1024 = _hash_matrix([[t] for t in ftok], 1024, signed=True)
        self.h_exp_s128 = _hash_matrix([[t] for t in self.exp_tokens], 128, signed=True)
        v3n = [nm.lower() for nm in self.sec_names]  # pefile names are lower-cased by thrember
        self.h_sec3 = _hash_matrix([[t] for t in v3n], 50, signed=True)
        self.h_secchar3 = _hash_matrix([[f"{nm}:{p}" for p in props] for nm, props in zip(v3n, self.sec_props)], 50, True)
        self.h_entry3 = _hash_matrix([[t] for t in v3n], 10, signed=True)
        self.h_sec2 = _hash_matrix([[t] for t in self.sec_names], 50, signed=True)
        self.h_entry_chars2 = _hash_matrix([list(nm) for nm in self.sec_names], 50, signed=True)
        self.h_entry_props2 = _hash_matrix([list(p) for p in self.sec_props], 50, signed=True)
        self.h_machine2 = _hash_matrix([["I386"], ["AMD64"]], 10, True)
        self.h_subsys2 = _hash_matrix([["UNKNOWN"], ["NATIVE"], ["WINDOWS_GUI"], ["WINDOWS_CUI"]], 10, True)
        self.h_magic2 = _hash_matrix([["PE32"], ["PE32_PLUS"]], 10, True)
        coff_tok = [[_V2_COFF_TOKEN.get(f, f)] if _V2_COFF_TOKEN.get(f, f) else [] for f in COFF_FLAGS]
        self.h_coff2 = _hash_matrix(coff_tok, 10, True)
        self.h_dll2 = _hash_matrix([[f] for f in DLL_FLAGS], 10, True)


def _bump16(center: float, width: float) -> np.ndarray:
    x = np.arange(16)
    v = np.exp(-0.5 * ((x - center) / width) ** 2) + 1e-4
    return v / v.sum()


def _english_printables() -> np.ndarray:
    """Relative frequency of the 96 printable characters 0x20..0x7f in PE strings."""
    p = np.full(96, 0.2)
    p[0] = 12.0  # space
    eng = "etaoinshrdlcumwfgypbvkjxqz"
    freq = np.linspace(8.0, 0.3, len(eng))
    for ch, f in zip(eng, freq):
        p[ord(ch) - 0x20] = f
        p[ord(ch.upper()) - 0x20] = f * 0.35
    for d in "0123456789":
        p[ord(d) - 0x20] = 1.2
    for ch, f in zip("._-:\\/%()=,", (2.5, 1.5, 0.8, 0.8, 0.9, 0.6, 0.7, 0.5, 0.5, 0.4, 0.5)):
        p[ord(ch) - 0x20] = f
    return p / p.sum()


def _b64_printables() -> np.ndarray:
    p = np.full(96, 0.05)
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=":
        p[ord(ch) - 0x20] = 1.0
    return p / p.sum()


def _hist_profiles(rng: np.random.Generator) -> np.ndarray:
    """(5, 256) byte distributions: zero padding, x86 code, text/tables, compressed, resources."""
    zero = np.full(256, 0.02 / 255)
    zero[0x00] = 0.97
    zero[0xFF] += 0.01
    code = rng.gamma(1.5, 1.0, size=256) * 0.0015 + 0.0008
    for b, w in {0x00: .12, 0xFF: .05, 0x8B: .045, 0x89: .03, 0x48: .035, 0xE8: .02, 0x45: .018, 0x24: .02,
                 0x83: .025, 0x0F: .02, 0x85: .014, 0x74: .014, 0x75: .014, 0xC3: .008, 0xCC: .02, 0x4C: .014,
                 0x44: .014, 0x10: .01, 0x08: .01, 0x01: .012, 0x04: .01, 0x50: .008, 0x33: .008, 0xC0: .01}.items():
        code[b] += w
    text = np.full(256, 0.0004)
    text[0x00] = 0.22
    text[0x20] = 0.05
    for c in range(0x61, 0x7B):
        text[c] = 0.012 + 0.01 * rng.random()
    for c in range(0x41, 0x5B):
        text[c] = 0.004 + 0.004 * rng.random()
    for c in range(0x30, 0x3A):
        text[c] = 0.004
    comp = np.full(256, 1.0 / 256)
    res = rng.gamma(2.0, 1.0, size=256) * 0.003
    res[0x00] += 0.15
    res[0xFF] += 0.06
    out = np.stack([zero, code, text, comp, res])
    return out / out.sum(1, keepdims=True)


@functools.lru_cache(maxsize=4)
def _world(seed: int, months: int) -> _World:
    return _World(seed, months)


# ==================================================================================================
# Row sampling
# ==================================================================================================


def _cat(rng: np.random.Generator, P: np.ndarray) -> np.ndarray:
    """Row-wise categorical draw from (n, K) non-negative weights."""
    c = np.cumsum(P, axis=1)
    u = rng.random(P.shape[0]) * c[:, -1]
    return np.minimum((u[:, None] > c).sum(1), P.shape[1] - 1)


def _entropy_bits(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(p), 0.0)
    return -t.sum(axis=-1)


def _month_start(start: dt.date, k: int) -> dt.date:
    m = start.month - 1 + k
    return dt.date(start.year + m // 12, m % 12 + 1, 1)


@dataclass
class _Rows:
    """Latent description of every synthetic file (feature-version independent)."""

    y: np.ndarray
    month: np.ndarray
    day: np.ndarray  # datetime64[D]
    split: np.ndarray
    cluster: np.ndarray
    attrs: dict[str, np.ndarray]


def _sample_rows(world: _World, prm: SyntheticParams, start: dt.date) -> _Rows:
    rng = np.random.default_rng([prm.seed, prm.n, 0xC0FFEE])
    n, M = prm.n, prm.months
    Tb = world.tables
    # ---- time, label, cluster, split -----------------------------------------------------------
    month = rng.integers(0, M, size=n)
    n_ch = int(round(prm.challenge_fraction * n))
    y = (rng.random(n) < prm.malicious_fraction).astype(np.int8)
    split = np.where(month < prm.train_months, "train", "test").astype("<U16")
    tr = np.flatnonzero(split == "train")
    split[tr[rng.random(tr.size) < prm.holdout_fraction]] = "holdout"
    te = np.flatnonzero(split == "test")
    challenge = np.zeros(n, dtype=bool)
    if n_ch and te.size:
        ch = rng.choice(te, size=min(n_ch, te.size), replace=False)
        challenge[ch] = True
        y[ch] = 1
        split[ch] = "challenge"
    W = world.cluster_weight  # (K, M)
    lab = world.cluster_label
    cluster = np.empty(n, dtype=np.int64)
    other = np.empty(n, dtype=np.int64)
    for cls in (0, 1):
        for m in range(M):
            sel = np.flatnonzero((y == cls) & (month == m))
            if sel.size == 0:
                continue
            own = np.flatnonzero(lab == cls)
            opp = np.flatnonzero(lab != cls)
            pw = W[own, m]
            if cls == 1:  # evasive challenge rows come from the newest families
                births = np.array([FAMILIES[k - len(BENIGN_CLUSTERS)][1] for k in own])
                young = pw * np.exp(-0.08 * np.clip(m - births, 0, 60))
                pch = young / young.sum()
                cluster[sel] = np.where(challenge[sel], own[rng.choice(own.size, size=sel.size, p=pch)],
                                        own[rng.choice(own.size, size=sel.size, p=pw / pw.sum())])
            else:
                cluster[sel] = own[rng.choice(own.size, size=sel.size, p=pw / pw.sum())]
            po = W[opp, m]
            other[sel] = opp[rng.choice(opp.size, size=sel.size, p=po / po.sum())]
    w = rng.beta(prm.overlap_a, prm.overlap_b, size=n)
    w[challenge] = rng.uniform(0.45, 0.8, size=int(challenge.sum()))
    day = np.array([np.datetime64(_month_start(start, int(m)), "D") for m in range(M)])[month]
    day = day + rng.integers(0, 28, size=n).astype("timedelta64[D]")
    first_seen = (day - np.datetime64("1970-01-01", "D")).astype(np.int64) * 86400 + rng.integers(0, 86400, n)
    year_frac = 1970 + first_seen / (365.25 * 86400)

    def blend(key: str) -> np.ndarray:
        a = Tb[key][cluster, month]
        b = Tb[key][other, month]
        ww = w.reshape((-1,) + (1,) * (a.ndim - 1)).astype(a.dtype)
        return (1 - ww) * a + ww * b

    sc = blend("scal")

    def S(name: str) -> np.ndarray:
        return sc[:, _SC[name]]

    def bern(p: np.ndarray | float) -> np.ndarray:
        return rng.random(n) < p

    A: dict[str, np.ndarray] = {}
    tool = _cat(rng, blend("tool"))
    packer = _cat(rng, blend("packer"))
    packer[tool == T_DOTNET] = np.where(packer[tool == T_DOTNET] == P_CUSTOM, P_CUSTOM, P_NONE)
    caps = rng.random((n, len(CAPS))) < blend("caps")
    is_dotnet = tool == T_DOTNET
    is64 = bern(S("p64"))
    is64[is_dotnet] = bern(0.12)[is_dotnet]
    is_dll = bern(S("p_dll"))
    is_dll[packer == P_INSTALLER] = False
    gui = bern(S("p_gui"))
    signed = bern(S("p_signed"))
    modern = bern(S("p_modern"))
    packed = np.isin(packer, (P_UPX, P_VMP, P_CUSTOM))
    has_res = bern(S("p_resources")) | (packer == P_INSTALLER) | (gui & bern(0.7))
    tool_reloc = np.array([0.9, 0.12, 0.05, 0.3, 0.3 + 0.02 * (start.year - 2017), 1.0, 0.05])
    relocs = is_dll & bern(0.97) | (~is_dll & bern(np.clip(tool_reloc[tool] * S("p_relocs") / 0.6, 0, 1)))
    relocs[is_dotnet] = True
    relocs[packer == P_UPX] &= is_dll[packer == P_UPX]
    tool_tls = np.array([0.08, 0.05, 1.0, 1.0, 0.0, 0.0, 0.02])
    tls = bern(np.clip(tool_tls[tool] + S("p_tls") - 0.1, 0, 1))
    tool_dbg = np.array([0.9, 0.5, 0.05, 0.15, 0.0, 0.95, 0.05])
    debug = bern(np.clip(tool_dbg[tool] * S("p_debug") / 0.6, 0, 1)) & ~(packer == P_UPX)
    # ---- time stamps ---------------------------------------------------------------------------
    tool_ts = np.array([[.9, 0, .1, 0, 0], [.95, 0, 0, .05, 0], [.3, 0, 0, .7, 0], [.95, .05, 0, 0, 0],
                        [0, 1, 0, 0, 0], [.6, 0, .4, 0, 0], [.8, .1, .1, 0, 0]])
    tsmode = _cat(rng, 0.5 * tool_ts[tool] + 0.5 * blend("tsmode"))
    lag_days = np.exp(rng.normal(S("lag"), 1.0))
    compile_ts = first_seen - (lag_days * 86400).astype(np.int64)
    ts = compile_ts.copy()
    ts[tsmode == 1] = 0
    ts[tsmode == 2] = rng.integers(0, 2**32 - 1, size=int((tsmode == 2).sum()))
    stale = tsmode == 3
    ts[stale] = np.where(tool[stale] == T_DELPHI, 708992537, rng.integers(631152000, 1104537600, int(stale.sum())))
    fut = tsmode == 4
    ts[fut] = first_seen[fut] + rng.integers(86400 * 30, 86400 * 3650, int(fut.sum()))
    build_year = np.clip(1970 + compile_ts / (365.25 * 86400), 2008, 2026)
    # ---- size, overlay, signature ----------------------------------------------------------------
    log_size = rng.normal(S("log_size") + 0.06 * (year_frac - start.year), S("log_size_sd"))
    size0 = np.exp(np.clip(log_size, np.log(4096), np.log(60e6)))
    overlay_flag = bern(S("p_overlay")) | (packer == P_INSTALLER)
    ratio = np.clip(rng.normal(S("overlay_ratio"), 0.12), 0.002, 0.995)
    ovl_extra = np.where(overlay_flag, np.round(size0 * ratio), 0)
    cert = np.where(signed, np.round(np.exp(rng.normal(9.0, 0.35, n)) / 8) * 8, 0)
    overlay_size = (ovl_extra + cert).astype(np.int64)
    ovl_ent = np.where(packer == P_INSTALLER, rng.normal(7.985, 0.01, n),
                       np.where(ovl_extra > 0, rng.uniform(1.0, 8.0, n), rng.normal(7.2, 0.25, n)))
    ovl_ent = np.where(overlay_size > 0, np.clip(ovl_ent, 0, 8), 0.0)
    # ---- sections ------------------------------------------------------------------------------
    Psec = world.tool_sec[tool].copy()
    repl = PACKER_REPLACES_LAYOUT[packer]
    Psec[repl] *= 0.0
    Psec = np.maximum(Psec, world.packer_sec[packer])
    Psec = np.maximum(Psec, blend("marker_sec"))
    Psec[:, SEC_INDEX[".rsrc"]] = has_res
    Psec[:, SEC_INDEX[".reloc"]] = relocs
    Psec[:, SEC_INDEX[".pdata"]] = is64 & np.isin(tool, (T_MSVC, T_LEGACY, T_MINGW)) & ~repl
    Psec[:, SEC_INDEX[".tls"]] = np.maximum(Psec[:, SEC_INDEX[".tls"]], (tls & ~repl & (tool != T_GO)) * 1.0)
    present = rng.random((n, world.S)) < Psec
    # guarantee an entry-capable section
    no_code = ~(present & (world.entry_priority > 0)[None, :]).any(1)
    present[no_code, SEC_INDEX[".text"]] = True
    n_sec = present.sum(1)
    headers = np.where(tool == T_DOTNET, 512, np.where(packed & bern(0.3), 4096, np.where(bern(0.7), 1024, 512)))
    raw_total = np.maximum(size0 - overlay_size - headers, 512.0 * n_sec)
    wgt = world.sec_w[None, :] * np.exp(rng.normal(0, 0.5, (n, world.S))) * present * ~world.sec_uninit[None, :]
    wgt[:, SEC_INDEX[".rsrc"]] *= np.exp(rng.normal(0, 1.0, n))
    wsum = wgt.sum(1, keepdims=True)
    wsum[wsum == 0] = 1.0
    raw = np.floor(wgt / wsum * raw_total[:, None] / 512.0) * 512.0
    raw[present & ~world.sec_uninit[None, :] & (raw == 0)] = 512.0
    raw[~present] = 0
    vs = raw * rng.uniform(0.85, 1.0, (n, world.S))
    data_k = world.sec_kind == KINDS.index("data")
    vs[:, data_k] = raw[:, data_k] * np.exp(rng.normal(0.4, 0.8, (n, int(data_k.sum()))))
    un = world.sec_uninit
    vs[:, un] = np.exp(rng.normal(np.log(np.maximum(raw_total, 4096))[:, None] + 0.4, 0.5, (n, int(un.sum()))))
    vs[:, SEC_INDEX[".bss"]] = np.exp(rng.normal(9.5, 1.0, n))
    vs[:, SEC_INDEX["BSS"]] = np.exp(rng.normal(9.0, 0.8, n))
    vs = np.round(vs) * present
    ent = np.clip(rng.normal(world.sec_ent[None, :], world.sec_ent_sd[None, :], (n, world.S)), 0, 8)
    ent[:, SEC_INDEX[".rsrc"]] = np.clip(ent[:, SEC_INDEX[".rsrc"]] + 1.5 * (packer == P_INSTALLER), 0, 8)
    ent = np.where(raw > 0, ent, 0.0) * present
    size = (headers + raw.sum(1) + overlay_size).astype(np.int64)
    va_step = np.ceil(np.maximum(vs, raw) / 4096.0) * 4096.0 * present
    va = 4096.0 + np.cumsum(va_step, axis=1) - va_step
    sizeof_image = (4096 + va_step.sum(1)).astype(np.int64)
    entry = np.argmax(present * (world.entry_priority[None, :] + 1e-3) + present * 1e-6, axis=1)
    # ---- byte histograms -------------------------------------------------------------------------
    comp = raw @ world.sec_comp  # (n, 5)
    ovl_comp = np.where((packer == P_INSTALLER)[:, None], [0, 0, 0.02, 0.98, 0], [0.1, 0, 0.2, 0.5, 0.2])
    comp = comp + overlay_size[:, None] * ovl_comp + headers[:, None] * np.array([0.8, 0, 0.2, 0, 0])
    comp = comp / comp.sum(1, keepdims=True)
    comp = rng.dirichlet(np.full(5, 1.0), size=n) * 0.08 + comp * 0.92
    hmean = comp @ world.hist_profiles
    hb = blend("hist_bump")
    bw = S("bump")[:, None]
    hmean = (1 - bw) * hmean + bw * hb
    hist = rng.gamma(np.maximum(hmean * 600.0, 1e-4))
    hist[hist < 1e-7 * hist.sum(1, keepdims=True)] = 0.0
    hist /= hist.sum(1, keepdims=True)
    ebin = comp @ world.entropy_bins
    ebin = rng.gamma(ebin * 60.0 + 1e-3)
    ebin /= ebin.sum(1, keepdims=True)
    nib = hist.reshape(n, 16, 16).sum(2)  # by high nibble
    lam = (np.arange(16) / 15.0)[None, :, None]
    low = np.zeros(16)
    low[0], low[15] = 0.85, 0.15
    Q = (1 - lam) * low[None, None, :] + lam * (0.5 * nib[:, None, :] + 0.5 / 16)
    be = ebin[:, :, None] * Q
    be = rng.gamma(np.maximum(be * 400.0, 1e-5))
    be[be < 3e-4 * be.sum((1, 2), keepdims=True)] = 0.0
    be = be.reshape(n, 256)
    be /= be.sum(1, keepdims=True)
    # ---- strings ---------------------------------------------------------------------------------
    dens = np.exp(rng.normal(S("str_density"), 0.6))
    numstrings = np.floor(size / 1024.0 * dens).astype(np.int64)
    avlen = np.maximum(5.0, np.exp(rng.normal(S("avlen"), 0.35)))
    printables = np.round(numstrings * avlen).astype(np.int64)
    obf = np.clip(S("obf") + rng.normal(0, 0.12, n), 0, 1)[:, None]
    pmean = (1 - obf) * world.printable_english[None, :] + obf * world.printable_b64[None, :]
    pmean = 0.9 * pmean + 0.1 * blend("print_bump")
    rep_w = rng.beta(0.3, 6.0, n)[:, None]  # repetitive padding-like strings lower the entropy
    rep = np.zeros((n, 96))
    rep[np.arange(n), rng.choice(np.array([0x41, 0x20, 0x2E, 0x30, 0x3D]) - 0x20, n)] = 1.0
    pmean = (1 - rep_w) * pmean + rep_w * rep
    pdist = rng.gamma(np.maximum(pmean * 80.0, 1e-4))
    pdist /= pdist.sum(1, keepdims=True)
    pcounts = rng.multinomial(printables, pdist)
    rate = world.regex_log_base[None, :] + caps @ world.cap_regex + blend("regex_bump") + np.log(0.25)
    regex = rng.poisson(np.exp(rate) * (numstrings[:, None] / 1000.0))
    dos_msg = np.where(tool == T_GO, 0, 1) + rng.poisson(0.15, n)
    mz = 1 + rng.poisson(np.where(packer == P_INSTALLER, 6.0, 1.0) + 25.0 * S("bump"))
    # ---- imports ---------------------------------------------------------------------------------
    marker = blend("marker_imp")
    presence = np.zeros((n, world.F), dtype=bool)
    step = 4096
    for a in range(0, n, step):
        b = min(n, a + step)
        L = world.log1m_tool[tool[a:b]] + CAP_IMPORT_SCALE[tool[a:b]][:, None] * (caps[a:b] @ world.log1m_cap)
        L = L + np.log1p(-np.clip(marker[a:b].astype(np.float64), 0, 0.999)) * (CAP_IMPORT_SCALE[tool[a:b]][:, None] > 0)
        presence[a:b] = rng.random((b - a, world.F)) < -np.expm1(L)
    presence[is_dotnet] = False
    presence[is_dotnet & ~is_dll, world.cor_exe] = True
    presence[is_dotnet & is_dll, world.cor_dll] = True
    pk = np.flatnonzero(packed & ~is_dotnet & bern(0.9))
    if pk.size:
        orig = presence[pk]
        keep = np.zeros_like(orig)
        for lib, (a, b) in world.lib_range.items():
            seg = orig[:, a:b]
            has = seg.any(1)
            first = np.argmax(seg, axis=1)
            keep[np.flatnonzero(has), a + first[has]] = True
        keep[:, world.stub] = rng.random((pk.size, world.stub.size)) < 0.8
        presence[pk] = keep
    no_imp = ((packer == P_CUSTOM) & bern(0.35)) | ((tool == T_ASM) & bern(0.1))
    presence[no_imp] = False
    n_functions = presence.sum(1)
    lib_present = (presence.astype(np.float32) @ world.lib_onehot) > 0
    n_libraries = lib_present.sum(1)
    # ---- exports ---------------------------------------------------------------------------------
    n_exp_target = np.where(is_dll & ~is_dotnet, np.exp(rng.normal(S("exp_mu"), 1.1)),
                            np.where(bern(0.03), rng.integers(1, 4, n), 0))
    ed = blend("exp_dist")
    exp_present = rng.random((n, world.E)) < -np.expm1(-n_exp_target[:, None] * ed)
    n_exports = exp_present.sum(1)
    # ---- header fields -------------------------------------------------------------------------
    yr = build_year
    msvc_minor = np.clip(np.round(10 + (yr - 2017) * 5.5 + rng.normal(0, 1.5, n)), 0, 44)
    lk_major = np.select([tool == T_MSVC, tool == T_LEGACY, tool == T_DELPHI, tool == T_MINGW, tool == T_GO,
                          tool == T_DOTNET], [14, rng.choice([6, 7, 8, 9, 10, 11, 12], n), 2, 2, 3,
                                              rng.choice([8, 11, 48, 80], n, p=[.3, .3, .3, .1])],
                         rng.choice([5, 1, 10], n))
    lk_minor = np.select([tool == T_MSVC, tool == T_LEGACY, tool == T_DELPHI, tool == T_MINGW],
                         [msvc_minor, rng.choice([0, 10], n), 25, rng.integers(20, 42, n)], 0)
    os_major = np.select([tool == T_MSVC, tool == T_LEGACY, tool == T_GO],
                         [np.where(modern, 6, 5), rng.choice([4, 5], n), np.where(start.year >= 2020, 6, 4)],
                         np.where(tool == T_DOTNET, 4, np.where(tool == T_DELPHI, rng.choice([4, 5], n), 4)))
    os_minor = np.where((tool == T_MSVC) & ~modern, 1, np.where((tool == T_GO) & (start.year >= 2020), 1, 0))
    ss_major, ss_minor = os_major.copy(), os_minor.copy()
    img = bern(S("p_image_version"))
    img_major = np.where(img, rng.choice([1, 6, 10, 2, 5], n, p=[.2, .2, .4, .1, .1]), 0)
    img_minor = np.where(img & bern(0.3), rng.integers(1, 4, n), 0)
    code_raw = (raw * world.sec_code[None, :]).sum(1)
    init_raw = (raw * world.sec_initdata[None, :]).sum(1)
    uninit_vs = (vs * world.sec_uninit[None, :]).sum(1)
    sizeof_headers = headers
    stack_res = np.select([tool == T_GO, tool == T_MINGW], [0x200000, 0x200000],
                          np.where(bern(0.1), 0x40000, 0x100000))
    stack_commit = np.select([tool == T_DELPHI, tool == T_GO], [0x4000, 0x1000], np.where(bern(0.05), 0x2000, 0x1000))
    heap_res = np.full(n, 0x100000)
    heap_commit = np.full(n, 0x1000)
    ep_sec = entry
    entry_va = va[np.arange(n), ep_sec]
    entry_raw = raw[np.arange(n), ep_sec]
    aoep = np.round(entry_va + rng.random(n) * np.maximum(entry_raw, 16) * 0.9).astype(np.int64)
    aoep[is_dll & ~is_dotnet & bern(0.05)] = 0
    base_of_code = np.where(packer == P_UPX, entry_va, 4096).astype(np.int64)
    image_base = np.select([is_dll & is64, is_dll & ~is64, is64 & (tool != T_GO)],
                           [0x180000000, 0x10000000, 0x140000000], 0x400000).astype(np.int64)
    checksum = np.where(bern(S("p_checksum")), size + rng.integers(-40000, 40000, n), 0).clip(0)
    n_symbols = np.where((tool == T_MINGW) & bern(0.4), rng.integers(100, 5000, n), 0)
    ptr_symtab = np.where(n_symbols > 0, (size * rng.uniform(0.5, 0.95, n)).astype(np.int64), 0)
    coff = np.zeros((n, len(COFF_FLAGS)), dtype=bool)
    coff[:, CF["EXECUTABLE_IMAGE"]] = bern(0.995)
    coff[:, CF["RELOCS_STRIPPED"]] = ~relocs & ~is_dll
    old = np.isin(tool, (T_LEGACY, T_DELPHI, T_MINGW, T_ASM))
    coff[:, CF["LINE_NUMS_STRIPPED"]] = old & bern(0.75)
    coff[:, CF["LOCAL_SYMS_STRIPPED"]] = coff[:, CF["LINE_NUMS_STRIPPED"]] & bern(0.95)
    coff[:, CF["LARGE_ADDRESS_AWARE"]] = is64 | bern(0.12)
    coff[:, CF["BYTES_REVERSED_LO"]] = (tool == T_DELPHI) & bern(0.8)
    coff[:, CF["BYTES_REVERSED_HI"]] = coff[:, CF["BYTES_REVERSED_LO"]]
    coff[:, CF["32BIT_MACHINE"]] = ~is64
    coff[:, CF["DEBUG_STRIPPED"]] = (np.isin(tool, (T_MINGW, T_ASM)) & bern(0.3)) | bern(0.01)
    coff[:, CF["REMOVABLE_RUN_FROM_SWAP"]] = (packer == P_INSTALLER) & bern(0.05)
    coff[:, CF["NET_RUN_FROM_SWAP"]] = coff[:, CF["REMOVABLE_RUN_FROM_SWAP"]]
    coff[:, CF["DLL"]] = is_dll
    era = (start.year - 2017) / 5.0
    dllc = np.zeros((n, len(DLL_FLAGS)), dtype=bool)
    p_dyn = np.array([0.97, 0.25, 0.15 + 0.3 * era, 0.35 + 0.3 * era, 0.1 + 0.8 * era, 0.97, 0.05])
    dyn = relocs & bern(np.clip(p_dyn[tool] * (0.5 + 0.5 * modern), 0, 1))
    dllc[:, DF["DYNAMIC_BASE"]] = dyn
    dllc[:, DF["NX_COMPAT"]] = dyn & bern(0.97) | (~dyn & bern(0.05))
    dllc[:, DF["HIGH_ENTROPY_VA"]] = is64 & dyn & np.isin(tool, (T_MSVC, T_GO, T_DOTNET)) & bern(0.9)
    dllc[:, DF["GUARD_CF"]] = (tool == T_MSVC) & modern & bern(0.15 + 0.25 * era)
    dllc[:, DF["TERMINAL_SERVER_AWARE"]] = np.isin(tool, (T_MSVC, T_DOTNET)) & ~is_dll & bern(0.85)
    dllc[:, DF["NO_SEH"]] = (is_dotnet & bern(0.95)) | ((tool == T_MSVC) & ~is64 & bern(0.05))
    dllc[:, DF["FORCE_INTEGRITY"]] = signed & bern(0.01)
    dllc[:, DF["NO_ISOLATION"]] = bern(0.006)
    dllc[:, DF["APPCONTAINER"]] = bern(0.003)
    dllc[:, DF["NO_BIND"]] = bern(0.001)
    # DOS header (17 members in thrember order)
    dos = np.zeros((n, 17), dtype=np.int64)
    dos[:, 0] = 23117
    delphi = tool == T_DELPHI
    dos[:, 1] = np.where(delphi, 80, 144)
    dos[:, 2] = np.where(delphi, 2, 3)
    dos[:, 4] = 4
    dos[:, 5] = np.where(delphi, 15, 0)
    dos[:, 6] = np.where(delphi, 65535, 65535)
    dos[:, 8] = 184
    dos[:, 12] = 64
    dos[:, 13] = np.where(delphi, 26, 0)
    weird = (tool == T_ASM) & bern(0.3)
    dos[weird, 1] = 0
    dos[weird, 2] = 0
    dos[weird, 8] = 0
    # rich header
    has_rich = np.isin(tool, (T_MSVC, T_LEGACY)) & ~bern(S("p_rich_strip"))
    n_pairs_target = np.where(tool == T_MSVC, 3 + rng.poisson(6, n), 2 + rng.poisson(3, n))
    rich_sel = rng.random((n, world.rich_pool)) < (n_pairs_target / world.rich_pool)[:, None]
    rich_sel &= has_rich[:, None]
    n_pairs = rich_sel.sum(1)
    has_rich &= n_pairs > 0
    rich_counts = np.round(np.exp(rng.normal(2.5, 1.6, (n, world.rich_pool)))) + 1
    rich = np.zeros((n, 32))
    yi = np.clip(np.round(build_year).astype(int) - int(world.rich_years[0]), 0, world.rich_years.size - 1)
    tt = np.where(tool == T_MSVC, 0, 1)
    for t in (0, 1):
        for yv in np.unique(yi):
            sel = np.flatnonzero(has_rich & (tt == t) & (yi == yv))
            if sel.size:
                rich[sel] = np.einsum("nk,kb->nb", rich_sel[sel] * rich_counts[sel], world.rich_H[t, yv])
    lfanew = np.where(has_rich, 128 + 24 + 8 * n_pairs, np.where(delphi, 256, 128))
    lfanew[weird] = 64
    dos[:, 16] = lfanew
    # ---- data directories ----------------------------------------------------------------------
    ptr = np.where(is64, 8, 4)
    rdata_va = va[:, SEC_INDEX[".rdata"]]
    rdata_va = np.where(rdata_va > 0, rdata_va, entry_va + entry_raw)
    dsz = np.zeros((n, 16))
    dva = np.zeros((n, 16))
    has_imp = n_functions > 0

    def setdd(name: str, mask: np.ndarray, size_: np.ndarray, va_: np.ndarray) -> None:
        j = DD[name]
        dsz[:, j] = np.where(mask, size_, 0)
        dva[:, j] = np.where(mask, va_, 0)

    setdd("EXPORT", n_exports > 0, 40 + n_exports * 18, rdata_va + rng.integers(0x100, 0x8000, n))
    setdd("IMPORT", has_imp, 20 * (n_libraries + 1), rdata_va + rng.integers(0x200, 0x9000, n))
    setdd("RESOURCE", present[:, SEC_INDEX[".rsrc"]], np.maximum(vs[:, SEC_INDEX[".rsrc"]] * 0.97, 64).round(),
          va[:, SEC_INDEX[".rsrc"]])
    exc = is64 & ~packed & ~is_dotnet & (tool != T_GO) | (is64 & (tool == T_GO))
    setdd("EXCEPTION", exc, np.round(code_raw / 90 / 12) * 12 + 12, np.where(va[:, SEC_INDEX[".pdata"]] > 0,
                                                                             va[:, SEC_INDEX[".pdata"]], rdata_va))
    setdd("SECURITY", signed, cert, size - cert)
    setdd("BASERELOC", present[:, SEC_INDEX[".reloc"]], np.round(code_raw / 60 / 4) * 4 + 8, va[:, SEC_INDEX[".reloc"]])
    setdd("DEBUG", debug, 28 * rng.integers(1, 5, n), rdata_va + rng.integers(0x40, 0x2000, n))
    setdd("TLS", tls & ~is_dotnet, np.where(is64, 40, 24), rdata_va + rng.integers(0x40, 0x3000, n))
    lc_sizes = np.array([64, 72, 92, 112, 148, 160, 192, 256, 280, 312, 320])
    lc_idx = np.clip(np.round((build_year - 2008) / 18 * 10 + rng.normal(0, 1, n)), 0, 10).astype(int)
    setdd("LOAD_CONFIG", (tool == T_MSVC) & ~(packer == P_UPX), lc_sizes[lc_idx], rdata_va + rng.integers(0x40, 0x4000, n))
    setdd("BOUND_IMPORT", (tool == T_LEGACY) & bern(0.03), 24 + 8 * rng.integers(0, 4, n), 0x200 + 0 * rdata_va)
    setdd("IAT", has_imp & ~(packer == P_UPX), (n_functions + n_libraries) * ptr, rdata_va)
    setdd("DELAY_IMPORT", (tool == T_MSVC) & bern(0.08), 32 * rng.integers(1, 5, n), rdata_va + rng.integers(0x40, 0x3000, n))
    setdd("COM_DESCRIPTOR", is_dotnet, np.full(n, 72), np.full(n, 0x2008))
    has_dyn_relocs = (tool == T_MSVC) & is64 & bern(0.01)
    # ---- authenticode --------------------------------------------------------------------------
    auth = np.zeros((n, 8))
    n_certs = 1 + bern(0.55) + bern(0.12)
    sign_time = compile_ts + rng.integers(0, 86400 * 5, n)
    auth[:, 0] = np.where(signed, n_certs, 0)
    auth[:, 1] = signed & bern(S("p_self_signed"))
    auth[:, 2] = signed & bern(0.4)
    auth[:, 3] = signed & bern(S("p_no_countersig"))
    auth[:, 4] = signed & bern(0.003)
    auth[:, 5] = np.where(signed, np.where(auth[:, 1] > 0, 1, 2 + rng.poisson(0.7, n)), 0)
    auth[:, 6] = np.where(signed & (auth[:, 3] == 0), sign_time, 0)
    auth[:, 7] = np.where(auth[:, 6] > 0, np.where(bern(0.05), sign_time, np.round(np.exp(rng.normal(4, 2.5, n)))), 0)
    # ---- pefile warnings -----------------------------------------------------------------------
    wrate = S("warn_rate") * (1.0 + 1.5 * packed)
    wcount = rng.poisson(wrate)
    wd = blend("warn_dist")
    warn = np.zeros((n, 87), dtype=bool)
    for k in range(int(wcount.max(initial=0))):
        sel = np.flatnonzero(wcount > k)
        warn[sel, _cat(rng, wd[sel])] = True
    idx_packed_warn = 40  # "Imported symbols contain entries typical of packed executables..."
    idx_susp_flags = 67   # "Suspicious flags set for section..."
    rwx = (present & world.sec_rx[None, :] & world.sec_w_flag[None, :]).any(1)
    add1 = packed & bern(0.7)
    add2 = rwx & bern(0.6)
    warn[add1, idx_packed_warn] = True
    warn[add2, idx_susp_flags] = True
    warn_count = wcount + add1 + add2

    A.update(
        tool=tool, packer=packer, is64=is64, is_dll=is_dll, gui=gui, signed=signed, dotnet=is_dotnet,
        has_res=has_res, relocs=relocs, tls=tls, debug=debug, ts=ts, size=size, sizeof_image=sizeof_image,
        overlay_size=overlay_size, overlay_ent=ovl_ent, sec_present=present, sec_raw=raw, sec_vs=vs,
        sec_ent=ent, entry=entry, hist=hist, byteentropy=be, numstrings=numstrings, avlen=avlen,
        printables=printables, pcounts=pcounts, regex=regex, dos_msg=dos_msg, mz=mz,
        presence=presence, lib_present=lib_present, n_functions=n_functions, n_libraries=n_libraries,
        exp_present=exp_present, n_exports=n_exports, lk_major=lk_major, lk_minor=lk_minor, os_major=os_major,
        os_minor=os_minor, ss_major=ss_major, ss_minor=ss_minor, img_major=img_major, img_minor=img_minor,
        sizeof_code=code_raw, sizeof_init=init_raw, sizeof_uninit=uninit_vs, sizeof_headers=sizeof_headers,
        stack_res=stack_res, stack_commit=stack_commit, heap_res=heap_res, heap_commit=heap_commit, aoep=aoep,
        base_of_code=base_of_code, image_base=image_base, checksum=checksum, n_symbols=n_symbols,
        ptr_symtab=ptr_symtab, coff=coff, dllc=dllc, dos=dos, n_pairs=np.where(has_rich, n_pairs, 0), rich=rich,
        dd_size=dsz, dd_va=dva, has_dyn_relocs=has_dyn_relocs, auth=auth, warn=warn, warn_count=warn_count,
        subsystem=np.where(gui, 2, 3),
    )
    return _Rows(y=y, month=month, day=day, split=split, cluster=cluster, attrs=A)


# ==================================================================================================
# Rendering into the real feature layouts
# ==================================================================================================


def _render_v3(rows: _Rows, world: _World) -> np.ndarray:
    from malvalid.schemas import _thrember_port as tp

    A = rows.attrs
    n = rows.y.size
    X = np.zeros((n, tp.DIM), dtype=np.float32)
    o = tp.OFFSETS
    g = o["general"]
    X[:, g + 0] = A["size"]
    X[:, g + 1] = _entropy_bits(A["hist"])
    X[:, g + 2] = 1
    X[:, g + 3] = 77
    X[:, g + 4] = 90
    X[:, g + 5] = A["dos"][:, 1] & 0xFF
    X[:, g + 6] = A["dos"][:, 1] >> 8
    X[:, o["histogram"]: o["histogram"] + 256] = A["hist"]
    X[:, o["byteentropy"]: o["byteentropy"] + 256] = A["byteentropy"]
    s = o["strings"]
    pr = A["printables"]
    X[:, s + 0] = A["numstrings"]
    X[:, s + 1] = np.where(A["numstrings"] > 0, A["avlen"], 0)
    X[:, s + 2] = pr
    dist = A["pcounts"] / np.maximum(pr, 1)[:, None]
    X[:, s + 3: s + 99] = dist
    X[:, s + 99] = _entropy_bits(dist)
    reg = np.zeros((n, len(tp.REGEX_NAMES)))
    for j, nm in enumerate(world.regex_names):
        reg[:, tp.REGEX_INDEX[nm]] = A["regex"][:, j]
    reg[:, tp.REGEX_INDEX["dos_msg"]] = A["dos_msg"]
    X[:, s + 100: s + 100 + len(tp.REGEX_NAMES)] = reg
    h = o["header"]
    machine = np.where(A["is64"], tp.MACHINE_INDEX["IMAGE_FILE_MACHINE_AMD64"], tp.MACHINE_INDEX["IMAGE_FILE_MACHINE_I386"])
    scal = {
        "timestamp": A["ts"], "number_of_sections": A["sec_present"].sum(1), "number_of_symbols": A["n_symbols"],
        "sizeof_optional_header": np.where(A["is64"], 240, 224), "pointer_to_symbol_table": A["ptr_symtab"],
        "machine": machine, "subsystem": A["subsystem"], "major_image_version": A["img_major"],
        "minor_image_version": A["img_minor"], "major_linker_version": A["lk_major"],
        "minor_linker_version": A["lk_minor"], "major_operating_system_version": A["os_major"],
        "minor_operating_system_version": A["os_minor"], "major_subsystem_version": A["ss_major"],
        "minor_subsystem_version": A["ss_minor"], "sizeof_code": A["sizeof_code"],
        "sizeof_headers": A["sizeof_headers"], "sizeof_image": A["sizeof_image"],
        "sizeof_initialized_data": A["sizeof_init"], "sizeof_uninitialized_data": A["sizeof_uninit"],
        "sizeof_stack_reserve": A["stack_res"], "sizeof_stack_commit": A["stack_commit"],
        "sizeof_heap_reserve": A["heap_res"], "sizeof_heap_commit": A["heap_commit"],
        "address_of_entrypoint": A["aoep"], "base_of_code": A["base_of_code"], "image_base": A["image_base"],
        "section_alignment": np.full(n, 4096), "checksum": A["checksum"], "number_of_rvas_and_sizes": np.full(n, 16),
    }
    for j, (_, key) in enumerate(tp.HEADER_SCALARS):
        X[:, h + j] = scal[key]
    j0 = h + len(tp.HEADER_SCALARS)
    X[:, j0: j0 + 16] = A["coff"][:, [COFF_FLAGS.index(f) for f in tp.IMAGE_CHARACTERISTICS]]
    X[:, j0 + 16: j0 + 27] = A["dllc"][:, [DLL_FLAGS.index(f) for f in tp.DLL_CHARACTERISTICS]]
    X[:, j0 + 27: j0 + 44] = A["dos"]
    sc = o["section"]
    P, raw, vs, ent = A["sec_present"], A["sec_raw"], A["sec_vs"], A["sec_ent"]
    size = A["size"].astype(np.float64)
    ovl = A["overlay_size"].astype(np.float64)
    names_empty = np.array([nm == "" for nm in world.sec_names])
    X[:, sc + 0] = P.sum(1)
    X[:, sc + 1] = (P & (raw == 0)).sum(1)
    X[:, sc + 2] = (P & names_empty[None, :]).sum(1)
    X[:, sc + 3] = (P & world.sec_rx[None, :]).sum(1)
    X[:, sc + 4] = (P & world.sec_w_flag[None, :]).sum(1)
    X[:, sc + 5] = np.maximum(np.where(P, ent, 0).max(1), A["overlay_ent"])
    X[:, sc + 7] = np.maximum((raw / size[:, None]).max(1), ovl / size)
    X[:, sc + 9] = np.where(P, raw / np.maximum(vs, 1), 0).max(1)
    X[:, sc + 11: sc + 61] = raw @ world.h_sec3
    X[:, sc + 61: sc + 111] = vs @ world.h_sec3
    X[:, sc + 111: sc + 161] = ent @ world.h_sec3
    X[:, sc + 161: sc + 211] = P.astype(np.float64) @ world.h_secchar3
    X[:, sc + 211: sc + 221] = world.h_entry3[A["entry"]]
    X[:, sc + 221] = ovl
    X[:, sc + 222] = ovl / size
    X[:, sc + 223] = A["overlay_ent"]
    im = o["imports"]
    X[:, im] = A["n_functions"]
    X[:, im + 1] = A["n_libraries"]
    X[:, im + 2: im + 258] = A["lib_present"].astype(np.float64) @ world.h_lib_u256
    X[:, im + 258: im + 1282] = _sparse_dot(A["presence"], world.h_fn_u1024)
    ex = o["exports"]
    X[:, ex] = np.where(A["n_exports"] > 0, 128, 0)
    X[:, ex + 1: ex + 129] = A["exp_present"].astype(np.float64) @ world.h_exp_s128
    d = o["datadirectories"]
    dd = np.zeros((n, 34))
    dd[:, 0:32:2] = A["dd_size"]
    dd[:, 1:32:2] = A["dd_va"]
    dd[:, 30:32] = 0  # RESERVED is never written by thrember
    dd[:, 32] = A["relocs"]
    dd[:, 33] = A["has_dyn_relocs"]
    X[:, d: d + 34] = dd
    r = o["richheader"]
    X[:, r] = A["n_pairs"]
    X[:, r + 1: r + 33] = A["rich"]
    X[:, o["authenticode"]: o["authenticode"] + 8] = A["auth"]
    w = o["pefilewarnings"]
    X[:, w: w + 87] = A["warn"]
    X[:, w + 87] = A["warn_count"]
    return X


def _render_v2(rows: _Rows, world: _World) -> np.ndarray:
    from malvalid.schemas.ember_v2 import DIM, GROUP_SLICES

    A = rows.attrs
    n = rows.y.size
    X = np.zeros((n, DIM), dtype=np.float32)
    X[:, GROUP_SLICES["histogram"]] = A["hist"]
    X[:, GROUP_SLICES["byteentropy"]] = A["byteentropy"]
    s = GROUP_SLICES["strings"].start
    pr = A["printables"]
    X[:, s + 0] = A["numstrings"]
    X[:, s + 1] = np.where(A["numstrings"] > 0, A["avlen"], 0)
    X[:, s + 2] = pr
    dist = A["pcounts"] / np.maximum(pr, 1)[:, None]
    X[:, s + 3: s + 99] = dist
    X[:, s + 99] = _entropy_bits(dist)
    ri = {nm: j for j, nm in enumerate(world.regex_names)}
    X[:, s + 100] = np.round(A["regex"][:, ri["file_path"]] * 0.6)  # "c:\" paths
    X[:, s + 101] = A["regex"][:, ri["http://"]] + A["regex"][:, ri["https://"]]
    X[:, s + 102] = A["regex"][:, ri["registry_key"]]
    X[:, s + 103] = A["mz"]
    g = GROUP_SLICES["general"].start
    X[:, g + 0] = A["size"]
    X[:, g + 1] = A["sizeof_image"]
    X[:, g + 2] = A["debug"]
    X[:, g + 3] = A["n_exports"]
    X[:, g + 4] = A["n_functions"]
    X[:, g + 5] = A["relocs"]
    X[:, g + 6] = A["has_res"]
    X[:, g + 7] = A["signed"]
    X[:, g + 8] = A["tls"]
    X[:, g + 9] = A["n_symbols"]
    h = GROUP_SLICES["header"].start
    X[:, h] = A["ts"]
    X[:, h + 1: h + 11] = world.h_machine2[A["is64"].astype(int)]
    X[:, h + 11: h + 21] = A["coff"].astype(np.float64) @ world.h_coff2
    X[:, h + 21: h + 31] = world.h_subsys2[A["subsystem"]]
    X[:, h + 31: h + 41] = A["dllc"].astype(np.float64) @ world.h_dll2
    X[:, h + 41: h + 51] = world.h_magic2[A["is64"].astype(int)]
    for j, key in enumerate(("img_major", "img_minor", "lk_major", "lk_minor", "os_major", "os_minor",
                             "ss_major", "ss_minor", "sizeof_code", "sizeof_headers", "heap_commit")):
        X[:, h + 51 + j] = A[key]
    sc = GROUP_SLICES["section"].start
    P, raw, vs, ent = A["sec_present"], A["sec_raw"], A["sec_vs"], A["sec_ent"]
    names_empty = np.array([nm == "" for nm in world.sec_names])
    X[:, sc + 0] = P.sum(1)
    X[:, sc + 1] = (P & (raw == 0)).sum(1)
    X[:, sc + 2] = (P & names_empty[None, :]).sum(1)
    X[:, sc + 3] = (P & world.sec_rx[None, :]).sum(1)
    X[:, sc + 4] = (P & world.sec_w_flag[None, :]).sum(1)
    X[:, sc + 5: sc + 55] = raw @ world.h_sec2
    X[:, sc + 55: sc + 105] = ent @ world.h_sec2
    X[:, sc + 105: sc + 155] = vs @ world.h_sec2
    X[:, sc + 155: sc + 205] = world.h_entry_chars2[A["entry"]]
    X[:, sc + 205: sc + 255] = world.h_entry_props2[A["entry"]]
    im = GROUP_SLICES["imports"].start
    X[:, im: im + 256] = A["lib_present"].astype(np.float64) @ world.h_lib_s256
    X[:, im + 256: im + 1280] = _sparse_dot(A["presence"], world.h_fn_s1024)
    X[:, GROUP_SLICES["exports"]] = A["exp_present"].astype(np.float64) @ world.h_exp_s128
    d = GROUP_SLICES["datadirectories"].start
    X[:, d: d + 30: 2] = A["dd_size"][:, :15]
    X[:, d + 1: d + 30: 2] = A["dd_va"][:, :15]
    return X


def _sparse_dot(B: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Dense bool (n, F) @ dense (F, W) via scipy.sparse (B is ~5 % dense)."""
    from scipy import sparse

    return np.asarray((sparse.csr_matrix(B, dtype=np.float64) @ sparse.csr_matrix(H)).toarray())


# ==================================================================================================
# Corpus assembly
# ==================================================================================================


@dataclass
class SyntheticData:
    """A generated corpus before it is written: arrays plus the manifest identity fields."""

    X: np.ndarray
    sha256: np.ndarray
    label: np.ndarray
    timestamp: np.ndarray
    split: np.ndarray
    cluster: np.ndarray  # generator cluster id per row (not stored in the corpus; for tests/docs)
    cluster_names: list[str]


def generate(feature_version: str, params: SyntheticParams | None = None, *, name: str | None = None) -> SyntheticData:
    """Generate a synthetic corpus in ``ember_v2`` or ``ember_v3`` space (deterministic)."""
    prm = params or SyntheticParams()
    prm.validate()
    if feature_version not in _START:
        raise ValueError(f"unknown synthetic feature version {feature_version!r} (expected ember_v2 or ember_v3)")
    t0 = time.monotonic()
    world = _world(prm.seed, prm.months)
    rows = _sample_rows(world, prm, _START[feature_version])
    X = (_render_v2 if feature_version == "ember_v2" else _render_v3)(rows, world)
    if not np.all(np.isfinite(X)):  # pragma: no cover - generator invariant
        raise RuntimeError("synthetic generator produced non-finite features")
    order = np.lexsort((np.arange(prm.n), rows.day))  # chronological, stable
    nm = name or f"synthetic_{feature_version.split('_')[-1]}"
    sha = row_sha256(nm, prm.seed, prm.n)
    log.info("generated %s: %d rows x %d in %.1fs", nm, prm.n, X.shape[1], time.monotonic() - t0)
    return SyntheticData(
        X=np.ascontiguousarray(X[order]), sha256=sha, label=rows.y[order].astype(np.int8),
        timestamp=rows.day[order].astype("datetime64[D]"), split=rows.split[order], cluster=rows.cluster[order],
        cluster_names=list(world.cluster_names),
    )


_START = {"ember_v2": dt.date(2017, 1, 1), "ember_v3": dt.date(2022, 1, 1)}

# Namespace of the synthetic row ids. It is part of every row's sha256, so it is frozen at the spelling used
# before the rename to MalValid: changing it would change every synthetic hash, and the bundled demo's
# training manifest (examples/synthetic_demo/train_sha256.txt) would no longer match the corpus (M4 skips).
ROW_SHA256_NAMESPACE = "malguard-synthetic"


def row_sha256(name: str, seed: int, n: int) -> np.ndarray:
    """The synthetic row ids: ``sha256("<namespace>/<name>/seed<seed>/n<n>/<i>")`` for ``i < n``."""
    return np.array([hashlib.sha256(f"{ROW_SHA256_NAMESPACE}/{name}/seed{seed}/n{n}/{i}".encode()).hexdigest()
                     for i in range(n)], dtype="<U64")


class _HashSink(io.RawIOBase):
    """Write-only stream that only hashes what it is given."""

    def __init__(self) -> None:
        self.h = hashlib.sha256()

    def writable(self) -> bool:
        return True

    def write(self, b: Any) -> int:
        mv = memoryview(b)
        self.h.update(mv)
        return mv.nbytes


def _npy_sha256(X: np.ndarray) -> str:
    sink = _HashSink()
    np.lib.format.write_array(sink, np.ascontiguousarray(X, dtype=np.float32), allow_pickle=False)
    return sink.h.hexdigest()


def _meta_arrays(data: SyntheticData) -> dict[str, np.ndarray]:
    return {"sha256": data.sha256.astype("<U64"), "label": data.label.astype(np.int8),
            "timestamp": data.timestamp.astype("datetime64[D]"), "split": data.split.astype("<U16")}


def _meta_sha256(data: SyntheticData) -> str:
    """sha256 of the ``meta.npz`` that :meth:`CorpusWriter.finalize` would write for ``data``."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "meta.npz"
        write_npz_deterministic(p, _meta_arrays(data))
        return hashlib.sha256(p.read_bytes()).hexdigest()


# ==================================================================================================
# Providers
# ==================================================================================================


def _foreign_entries(d: Path) -> list[str]:
    """Names in ``d`` that are not corpus files (empty if ``d`` is missing, empty or a corpus dir)."""
    try:
        return sorted(p.name for p in Path(d).iterdir() if p.name not in CORPUS_DIR_FILES)
    except OSError:  # missing, not a directory, unreadable: nothing of the user's to protect here
        return []


class _SyntheticProvider(CorpusProvider):
    """Base for the synthetic providers: generated on first load, cached under the corpus root."""

    feature_version: ClassVar[str]
    version: ClassVar[str] = GENERATOR_VERSION
    synthetic: ClassVar[bool] = True
    expected_content_hash: ClassVar[str | None] = None  # depends on numpy's RNG streams; not pinned

    # ---- parameters ------------------------------------------------------------------------------

    @classmethod
    def params(cls) -> SyntheticParams:
        """Default parameters, overridable with $MALVALID_SYNTHETIC_ROWS / $MALVALID_SYNTHETIC_SEED."""
        kw: dict[str, Any] = {}
        for env, key in ((ENV_ROWS, "n"), (ENV_SEED, "seed")):
            v = os.environ.get(env)
            if v:
                try:
                    kw[key] = int(v)
                except ValueError as e:
                    raise CorpusUnavailable(f"${env}={v!r} is not an integer") from e
        return SyntheticParams(**kw)

    def locate(self, config: "GateConfig | None" = None) -> Path:
        """``corpus_dir`` may be this corpus's directory or a root holding it (like
        ``$MALVALID_CORPUS_DIR``). A non-empty directory that is not a corpus is treated as a root,
        so the corpus is generated into ``<dir>/<name>`` and never into the directory itself."""
        from malvalid.corpora.base import default_corpus_root

        prm = self.params()
        sub = f"{self.name}" + ("" if prm == SyntheticParams() else f"-n{prm.n}-seed{prm.seed}")
        if config is not None and getattr(config, "corpus_dir", None):
            d = Path(config.corpus_dir).expanduser()
            if manifest_name(d) != self.name:
                for cand in dict.fromkeys((sub, self.name)):
                    if (d / cand / "manifest.json").exists():
                        return d / cand
            if not (d / "manifest.json").exists() and _foreign_entries(d):
                return d / sub
            return d
        return default_corpus_root() / sub

    def is_available(self, config: "GateConfig | None" = None) -> bool:
        return True  # generated on demand

    def unavailable_hint(self, d: Path) -> str:
        return (f"synthetic corpus {self.name!r} could not be generated at {d}; it needs no download — "
                f"run `malvalid corpus build {self.name}` or point --corpus-dir at a writable directory")

    # ---- generation ------------------------------------------------------------------------------

    def generate(self, params: SyntheticParams | None = None) -> SyntheticData:
        return generate(self.feature_version, params or self.params(), name=self.name)

    def _manifest_fields(self, prm: SyntheticParams, data: SyntheticData) -> dict[str, Any]:
        splits, counts = np.unique(data.split, return_counts=True)
        ts = data.timestamp
        return dict(
            name=self.name,
            version=self.version,
            feature_version=self.feature_version,
            sha256=data.sha256,
            label=data.label,
            timestamp=data.timestamp,
            split=data.split,
            roles={k: list(v) for k, v in ROLES.items()},
            description=self.description,
            source={
                "generator": GENERATOR,
                "generator_version": GENERATOR_VERSION,
                "params": asdict(prm),
                "numpy": np.__version__,
                "note": "SYNTHETIC data for demos and CI; not evidence about any real detector.",
            },
            synthetic=True,
            extra={
                "generator_version": GENERATOR_VERSION,
                "params": asdict(prm),
                "rows_by_split": {str(s): int(c) for s, c in zip(splits, counts)},
                "time_range": [str(ts.min()), str(ts.max())],
                "clusters": data.cluster_names,
            },
        )

    def build(self, source: Path | None, out: Path, **kwargs: Any) -> dict[str, Any]:
        """Generate the corpus into ``out`` (``source`` is ignored; ``n``/``seed`` may be given)."""
        kw = {k: kwargs[k] for k in ("n", "seed") if kwargs.get(k) is not None}
        prm = SyntheticParams(**{**asdict(self.params()), **kw})
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self.generate(prm)
        tmp = out.parent / f".{out.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            writer = CorpusWriter(tmp, data.X.shape[0], data.X.shape[1])
            writer.write_rows(0, data.X)
            m = writer.finalize(**self._manifest_fields(prm, data))  # meta.npz is byte-deterministic
            out.mkdir(parents=True, exist_ok=True)
            for fn in ("X.npy", "meta.npz", ".malvalid-verified.json", "manifest.json"):  # manifest last
                if (tmp / fn).exists():
                    os.replace(tmp / fn, out / fn)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        write_verify_sidecar(out, m)  # re-stamp the verification cache for the final paths
        log.info("synthetic corpus %s written to %s (content_hash=%s)", self.name, out, m["content_hash"])
        return m

    def in_memory(self, params: SyntheticParams | None = None) -> Corpus:
        """The corpus as an in-memory :class:`Corpus` (same content hash as the on-disk build)."""
        prm = params or self.params()
        data = self.generate(prm)
        f = self._manifest_fields(prm, data)
        manifest: dict[str, Any] = {
            "format": CORPUS_FORMAT, "name": f["name"], "version": f["version"],
            "feature_version": f["feature_version"], "dim": int(data.X.shape[1]), "n": int(data.X.shape[0]),
            "description": f["description"], "source": f["source"], "synthetic": True, "roles": f["roles"],
            "files": {"X.npy": _npy_sha256(data.X), "meta.npz": _meta_sha256(data)}, "extra": f["extra"],
            "created_by": "malvalid (in memory)",
        }
        manifest["content_hash"] = compute_content_hash(manifest)
        return Corpus(
            name=self.name, version=self.version, feature_version=self.feature_version,
            content_hash=manifest["content_hash"], manifest=manifest, X=data.X, sha256=data.sha256,
            label=data.label, timestamp=data.timestamp, split=data.split, path=None, synthetic=True,
        )

    def load(self, config: "GateConfig | None" = None, *, verify: bool = True) -> Corpus:
        d = self.locate(config)
        if (d / "manifest.json").exists():
            try:
                manifest = json.loads((d / "manifest.json").read_text())
            except (OSError, ValueError) as e:
                raise CorpusUnavailable(f"{d}: unreadable manifest.json ({e})") from e
            if not manifest.get("synthetic") or manifest.get("name") != self.name:
                raise CorpusUnavailable(
                    f"{d} holds corpus {manifest.get('name')!r} (synthetic={manifest.get('synthetic')}), "
                    f"not the synthetic corpus {self.name!r}; point --corpus-dir elsewhere")
            if (manifest.get("extra") or {}).get("generator_version") == GENERATOR_VERSION:
                corpus = load_corpus_dir(d, verify=verify, force_verify=full_verification_requested(config))
                if corpus.feature_version != self.feature_version:
                    raise CorpusUnavailable(f"{d}: feature_version {corpus.feature_version} != {self.feature_version}")
                corpus.synthetic = True
                return corpus
            log.warning("%s: synthetic corpus from generator version %s; regenerating with version %s",
                        d, (manifest.get("extra") or {}).get("generator_version"), GENERATOR_VERSION)
            prm_d = (manifest.get("extra") or {}).get("params") or {}
            prm = SyntheticParams(**{k: v for k, v in prm_d.items() if k in SyntheticParams.__dataclass_fields__})
        else:
            prm = self.params()
            foreign = _foreign_entries(d)
            if foreign:
                # Never generate into a directory that holds something else (e.g. a corpora root):
                # the build would drop X.npy / meta.npz / manifest.json among the user's files.
                raise CorpusUnavailable(
                    f"{d} is not empty and holds no corpus (found {', '.join(foreign[:3])}"
                    f"{', ...' if len(foreign) > 3 else ''}); refusing to generate the synthetic corpus "
                    f"{self.name!r} into it. Point corpus_dir / --corpus-dir at an empty or new directory, "
                    f"or at a root that contains {self.name}/"
                )
        try:
            self.build(None, d, n=prm.n, seed=prm.seed)
        except OSError as e:
            log.warning("cannot cache synthetic corpus %s at %s (%s); generating it in memory", self.name, d, e)
            return self.in_memory(prm)
        corpus = load_corpus_dir(d, verify=False)
        corpus.synthetic = True
        corpus.verification = {"mode": "generated", "files": {fn: "hashed" for fn in corpus.manifest.get("files", {})},
                               "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")}
        return corpus

    def info(self) -> dict[str, Any]:
        d = super().info()
        d["params"] = asdict(self.params())
        d["generator_version"] = GENERATOR_VERSION
        return d


class SyntheticEmberV2Provider(_SyntheticProvider):
    """``synthetic_v2``: 20,000 synthetic rows in ember_v2 space, 2017-01 .. 2019-12."""

    name: ClassVar[str] = "synthetic_v2"
    feature_version: ClassVar[str] = "ember_v2"
    description: ClassVar[str] = (
        "SYNTHETIC EMBER v2-shaped corpus (2381 features, 36 monthly windows 2017-01..2019-12, "
        "train/holdout/test/challenge) for demos and CI - not evidence about any real detector."
    )


class SyntheticEmberV3Provider(_SyntheticProvider):
    """``synthetic_v3``: 20,000 synthetic rows in ember_v3 space, 2022-01 .. 2024-12."""

    name: ClassVar[str] = "synthetic_v3"
    feature_version: ClassVar[str] = "ember_v3"
    description: ClassVar[str] = (
        "SYNTHETIC EMBER v3-shaped corpus (2568 features, 36 monthly windows 2022-01..2024-12, "
        "train/holdout/test/challenge) for demos and CI - not evidence about any real detector."
    )
