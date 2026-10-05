"""Unit tests for the ember_v2 feature schema (malvalid.schemas.ember_v2).

Fast tests use three real EMBER2018 raw records embedded below (feature data only: hashes,
histograms, header fields, import names; no executable content). Those records come from the
EMBER2018 dataset by H. S. Anderson and P. Roth (Endgame, Inc., now Elastic), distributed under the
MIT licence; see the EMBER entry in NOTICE for the attribution and licence text. Tests marked ``ember`` also
compare against the canonical corpus when it is available (see test_corpus_ember2018.py for the
environment variables).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from malvalid.core import FeaturizeUnavailable
from malvalid.schemas import ember_v2 as E
from malvalid.schemas.base import Controllability as C

# --------------------------------------------------------------------------------------------------
# Embedded real records (zlib + base64 of three JSONL lines from the public EMBER2018 release):
#   0: test_features.jsonl line 34709   -> corpus row 834708, benign, ordinal imports
#   1: test_features.jsonl line 55441   -> corpus row 855440, malicious, exports, entry name "    "
#   2: train_features_1.jsonl line 44403 -> corpus row  94402, unlabeled, UPX-packed (entry UPX0)
# --------------------------------------------------------------------------------------------------
_RECORDS_B64 = (
    "eNrtWltzG8exfj+/QoVnWTX3S14SCIRl1OGtQDCS7XKhFsBCQkICLAByzKj83zNf98xiQYIKScuJk5hFALs705fp6emZ/rY/"
    "dTYfKmVd5w8vOs5qXbuZqKdVNNIZW5tKClFHVwmr/VRbMRdGuYmbTuKk1m5SKVmpqTd+GmKYdF6+6FzPLFj5EIypZe1tNZ2I"
    "WmgVzKSaOaOCjqGy6Frd3NTVup6hvxIyfCUlHl9Vk/oqPRPo8uP0qtps0ANNHxab7er9urpOD773yruXL1R8+SK8fCFl+qRf"
    "//KFTk+0Srf25Yv0L8t/6mzSj07/6bGJdGmYOtANP0qNOjCb1CroWaKzTAuRzFcVxqY0OGrA88htioktfyK3EU/moUm6KQ8l"
    "U/qGCj++MOXROX6qmlEH7sNNksfJXT2RicJdl1vFKgvqIdOlwiCYgOwIu5hCr/FEsEmN5R7FPhI2BcesvCPOrLgoVhFFe1aD"
    "uaoyN6IxhGG+smUN/o7levcw3OsErfz+aJmCratbjAUPURV7iDIvsnXbsGg3uZYaqvWfbr0rZKY1zTrPpWz5odmfFlt4ikO3"
    "el+lO31k0aQZ6s4krgiTrWVg/Q9pGU1ut3W93K5XN7dYSE4asy/13/4vfYTTp5G5vCiUY683eTG5ZpU1ns6rSesQ2ZPQFSve"
    "YxIwPQ7WkGVFMo+yarHw3W/MCjCECtDVKV6l+Oj8wQzD6yScjIIdx44Q+JIW7b9AxQh7y5hjMcKEQ8xKv5FCB/Q3OVg4DgAx"
    "TQRNBgKD/e1Z/Wn/WFKb7XqxfI+N6lNn+fF6d4sZS9vYVb18v/2Q7qV6JYx3Mjbfqf0m9d5Wk6t6ljY4LMnQCkZi/1rciXvt"
    "wBDp/0Ff2r81ZfXIQtUOJ02A1iV4NpHalAgkPyur2U1pp3TlseE9J0e1/EDyR5VRWA7RuhDp/UDXMnxjOjK2TR07u9BmX4ng"
    "hXPSa2u1cWTravthk48XH9dX5XJdv0+2X9/m25PvMFc/p6v39bJeV1c0s5vF3+t0ERWG0/kx30qlsOY6H6rNeFZPPr7HM+jx"
    "081qvS0CFtflLua+6/pqNa22i9Wy9OGnm9XH9ZTGI/OzzeL9stp+XNetfttG9c3t9WTFd1D4Q13N6jXpO13N53SxXVzXm211"
    "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"
)
RECORD_CORPUS_ROWS = (834708, 855440, 94402)
# sha256 of vectorize_raw_batch(RECORDS) as float32 bytes; equals the canonical corpus rows above
# (checked by test_embedded_records_match_canonical_corpus) and scores 8.3e-05 / 0.99969 / 2.7e-05
# with the published EMBER2018 LightGBM model.
PINNED_DIGEST = "5c32836f1d07220b745f4fffceba72a4c02955a922fb05d36bd9a59f6d75c0a6"

PINNED_GROUPS = [
    ("histogram", 0, 256), ("byteentropy", 256, 512), ("strings", 512, 616),
    ("general", 616, 626), ("header", 626, 688), ("section", 688, 943),
    ("imports", 943, 2223), ("exports", 2223, 2351), ("datadirectories", 2351, 2381),
]


def load_records() -> list[dict[str, Any]]:
    raw = zlib.decompress(base64.b64decode("".join(_RECORDS_B64)))
    return [json.loads(line) for line in raw.split(b"\n")]


RECORDS = load_records()


@pytest.fixture(scope="module")
def schema() -> E.EmberV2Schema:
    return E.EmberV2Schema()


def _setuptools_exes() -> list[Path]:
    try:
        import setuptools
    except ImportError:  # pragma: no cover
        return []
    d = Path(setuptools.__file__).parent
    return [d / n for n in ("cli-32.exe", "cli-64.exe", "gui-32.exe", "gui-64.exe") if (d / n).is_file()]


# --------------------------------------------------------------------------------------------------
# Independent reference vectoriser (test oracle): written from the documented EMBER v2 layout with
# scikit-learn's FeatureHasher, one record at a time, deliberately unlike the batched implementation.
# --------------------------------------------------------------------------------------------------


def _hashed(width: int, items: list[Any], input_type: str = "string") -> np.ndarray:
    from sklearn.feature_extraction import FeatureHasher

    if not items:
        return np.zeros(width)
    return FeatureHasher(width, input_type=input_type).transform([items]).toarray()[0]


def reference_vector(r: dict[str, Any]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for key in ("histogram", "byteentropy"):
        c = np.asarray(r[key], dtype=np.float32)
        parts.append((c / c.sum()).astype(np.float64))
    s = r["strings"]
    div = float(s["printables"]) if s["printables"] > 0 else 1.0
    parts.append(np.concatenate([
        [s["numstrings"], s["avlength"], s["printables"]],
        np.asarray(s["printabledist"], dtype=np.float64) / div,
        [s["entropy"], s["paths"], s["urls"], s["registry"], s["MZ"]],
    ]))
    parts.append(np.array([r["general"][k] for k in E.GENERAL_FIELDS], dtype=np.float64))
    coff, opt = r["header"]["coff"], r["header"]["optional"]
    parts.append(np.concatenate([
        [coff["timestamp"]],
        _hashed(10, [coff["machine"]]),
        _hashed(10, list(coff["characteristics"])),
        _hashed(10, [opt["subsystem"]]),
        _hashed(10, list(opt["dll_characteristics"])),
        _hashed(10, [opt["magic"]]),
        [opt[k] for k in E.OPTIONAL_INT_FIELDS],
    ]))
    secs, entry = r["section"]["sections"], r["section"]["entry"]
    parts.append(np.concatenate([
        [
            len(secs),
            sum(1 for x in secs if x["size"] == 0),
            sum(1 for x in secs if x["name"] == ""),
            sum(1 for x in secs if "MEM_READ" in x["props"] and "MEM_EXECUTE" in x["props"]),
            sum(1 for x in secs if "MEM_WRITE" in x["props"]),
        ],
        _hashed(50, [(x["name"], x["size"]) for x in secs], "pair"),
        _hashed(50, [(x["name"], x["entropy"]) for x in secs], "pair"),
        _hashed(50, [(x["name"], x["vsize"]) for x in secs], "pair"),
        _hashed(50, list(entry)),
        _hashed(50, [p for x in secs if x["name"] == entry for p in x["props"]]),
    ]))
    libs = sorted({lib.lower() for lib in r["imports"]})
    funcs = [f"{lib.lower()}:{fn}" for lib, fns in r["imports"].items() for fn in fns]
    parts.append(np.concatenate([_hashed(256, libs), _hashed(1024, funcs)]))
    parts.append(_hashed(128, list(r["exports"])))
    dd = np.zeros(30)
    for i, d in enumerate(r["datadirectories"][:15]):
        dd[2 * i], dd[2 * i + 1] = d["size"], d["virtual_address"]
    parts.append(dd)
    return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------------------------------
# Layout, names, controllability
# --------------------------------------------------------------------------------------------------


def test_pinned_group_boundaries(schema: E.EmberV2Schema) -> None:
    assert schema.name == "ember_v2" and schema.dim == 2381
    assert [(g.name, g.start, g.stop) for g in schema.groups()] == PINNED_GROUPS
    schema.check_layout()
    assert {k: (v.start, v.stop) for k, v in E.GROUP_SLICES.items()} == {n: (a, b) for n, a, b in PINNED_GROUPS}


def test_registered_in_registry() -> None:
    from malvalid import registry

    assert isinstance(registry.get_schema("ember_v2"), E.EmberV2Schema)


def test_feature_names_unique_and_grouped(schema: E.EmberV2Schema) -> None:
    names = schema.feature_names()
    assert len(names) == 2381 and len(set(names)) == 2381
    for g in schema.groups():
        assert all(n.startswith(g.name + ".") or n.startswith(g.name + "[") for n in names[g.slice]), g.name
    expected = {
        0: "histogram[0x00]",
        255: "histogram[0xFF]",
        256: "byteentropy.bin00[0x0_]",
        512: "strings.numstrings",
        615: "strings.MZ",
        616: "general.size",
        626: "header.coff.timestamp",
        627: "header.coff.machine_hashed[0]",
        677: "header.optional.major_image_version",
        687: "header.optional.sizeof_heap_commit",
        688: "section.num_sections",
        943: "imports.libraries_hashed[0]",
        1199: "imports.functions_hashed[0]",
        1216: "imports.functions_hashed[17]",
        2223: "exports.functions_hashed[0]",
        2351: "datadirectories.EXPORT_TABLE.size",
        2380: "datadirectories.CLR_RUNTIME_HEADER.virtual_address",
    }
    for i, n in expected.items():
        assert names[i] == n
        assert schema.feature_index(n) == i
    with pytest.raises(KeyError, match="no feature named"):
        schema.feature_index("general.nope")


def test_controllability_levels(schema: E.EmberV2Schema) -> None:
    ctrl = schema.feature_controllability()
    assert len(ctrl) == 2381 and set(ctrl) == set(C)
    by_name = dict(zip(schema.feature_names(), ctrl))
    assert by_name["header.coff.timestamp"] is C.CONTROLLABLE
    for k in ("major_image_version", "minor_linker_version", "major_operating_system_version"):
        assert by_name[f"header.optional.{k}"] is C.CONTROLLABLE
    assert by_name["general.has_signature"] is C.CONTROLLABLE
    assert by_name["general.size"] is C.APPEND_ONLY
    assert by_name["strings.numstrings"] is C.APPEND_ONLY
    assert by_name["section.num_sections"] is C.APPEND_ONLY
    assert by_name["general.has_tls"] is C.FIXED
    assert by_name["header.optional.major_subsystem_version"] is C.FIXED
    assert by_name["datadirectories.CERTIFICATE_TABLE.size"] is C.CONTROLLABLE
    assert by_name["datadirectories.TLS_TABLE.size"] is C.FIXED
    for g, level in (("histogram", C.DERIVED), ("byteentropy", C.DERIVED), ("imports", C.APPEND_ONLY),
                     ("exports", C.APPEND_ONLY)):
        assert {ctrl[i] for i in range(*E.GROUP_SLICES[g].indices(2381))} == {level}, g
    for k in range(10):
        assert by_name[f"header.coff.machine_hashed[{k}]"] is C.FIXED
        assert by_name[f"header.optional.magic_hashed[{k}]"] is C.FIXED
    reasons = schema.controllability_reasons()
    assert len(reasons) == 2381 and all(isinstance(r, str) and r for r in reasons)
    m = schema.mask(C.CONTROLLABLE)
    assert m.dtype == bool and m[626] and not m[0]
    info = schema.feature_info(626)
    assert info["group"] == "header" and info["controllability"] == "controllable" and info["reason"]
    with pytest.raises(IndexError):
        schema.feature_info(2381)


def test_hashed_flag_buckets_follow_their_flags() -> None:
    """A header-flag bucket is CONTROLLABLE only if every flag hashing there is loader-ignored."""
    table = E.feature_table()
    start = 626 + 1 + 10  # coff.characteristics_hashed
    for name, (_bit, role) in E.COFF_CHARACTERISTICS.items():
        bucket, _ = E.hash_bucket(name, 10)
        level = table[start + bucket][1]
        if role == "structural":
            assert level is not C.CONTROLLABLE, name
        assert name in table[start + bucket][2]


def test_info_reports_counts_and_featurize_note(schema: E.EmberV2Schema) -> None:
    info = schema.info()
    assert info["name"] == "ember_v2" and info["dim"] == 2381 and len(info["groups"]) == 9
    assert sum(info["controllability_counts"].values()) == 2381
    assert "LIEF" in info["featurize_note"]
    json.dumps(info)


# --------------------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------------------


def test_hash_bucket_matches_sklearn_feature_hasher() -> None:
    tokens = ["kernel32.dll", "kernel32.dll:CreateFileW", "UPX0", "ä€", ".text", "", "ordinal17"]
    for width in (10, 50, 128, 256, 1024):
        for t in tokens:
            b, sign = E.hash_bucket(t, width)
            ref = _hashed(width, [t])
            expect = np.zeros(width)
            expect[b] = sign
            np.testing.assert_array_equal(ref, expect)


def test_pure_python_murmur_matches_sklearn() -> None:
    from sklearn.utils import murmurhash3_32

    rng = np.random.default_rng(0)
    for n in list(range(0, 12)) + [31, 64, 200]:
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        assert E._murmur3_signed(data) == murmurhash3_32(data, seed=0)


# --------------------------------------------------------------------------------------------------
# vectorize_raw / vectorize_raw_batch
# --------------------------------------------------------------------------------------------------


def test_vectorize_matches_pinned_digest_and_reference(schema: E.EmberV2Schema) -> None:
    V = schema.vectorize_raw_batch(RECORDS)
    assert V.shape == (3, 2381) and V.dtype == np.float32 and np.isfinite(V).all()
    assert hashlib.sha256(np.ascontiguousarray(V).tobytes()).hexdigest() == PINNED_DIGEST
    for r, v in zip(RECORDS, V):
        np.testing.assert_allclose(v, reference_vector(r), rtol=1e-6, atol=0)
        np.testing.assert_array_equal(schema.vectorize_raw(r), v)


def test_vectorize_semantics_spot_checks(schema: E.EmberV2Schema) -> None:
    r = RECORDS[1]
    v = schema.vectorize_raw(r)
    names = schema.feature_names()
    at = {n: v[i] for i, n in enumerate(names)}
    assert at["general.size"] == np.float32(r["general"]["size"])
    assert at["header.coff.timestamp"] == np.float32(r["header"]["coff"]["timestamp"])
    assert at["section.num_sections"] == len(r["section"]["sections"])
    assert abs(float(v[E.GROUP_SLICES["histogram"]].sum()) - 1.0) < 1e-5
    assert abs(float(v[E.GROUP_SLICES["byteentropy"]].sum()) - 1.0) < 1e-5
    s = r["strings"]
    assert at["strings.printabledist[0x41]"] == np.float32(s["printabledist"][0x21] / s["printables"])
    dd = r["datadirectories"]
    assert at["datadirectories.IMPORT_TABLE.size"] == dd[1]["size"]
    assert at["datadirectories.IMPORT_TABLE.virtual_address"] == dd[1]["virtual_address"]
    # imports: libraries are case-folded before hashing (KERNEL32.DLL == kernel32.dll)
    b, sign = E.hash_bucket("kernel32.dll", 256)
    assert v[943 + b] != 0
    # entry section name "    " is hashed character by character: 4 x the same token
    b, sign = E.hash_bucket(" ", 50)
    assert v[688 + 155 + b] == 4 * sign


def test_batch_equals_single_and_iter(schema: E.EmberV2Schema) -> None:
    many = RECORDS * 5
    V = schema.vectorize_raw_batch(many)
    for i, r in enumerate(many):
        np.testing.assert_array_equal(V[i], schema.vectorize_raw(r))
    chunks = list(E.iter_vectorized(iter(many), batch_size=4))
    assert [c.shape[0] for c in chunks] == [4, 4, 4, 3]
    np.testing.assert_array_equal(np.vstack(chunks), V)
    assert schema.vectorize_raw_batch([]).shape == (0, 2381)


def test_all_zero_histogram_gives_nan(schema: E.EmberV2Schema) -> None:
    r = json.loads(json.dumps(RECORDS[0]))
    r["histogram"] = [0] * 256
    v = schema.vectorize_raw(r)
    assert np.isnan(v[:256]).all() and np.isfinite(v[256:]).all()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda r: r.pop("imports"), "missing top-level field 'imports'"),
        (lambda r: r["header"]["coff"].pop("timestamp"), "header.coff.timestamp"),
        (lambda r: r.__setitem__("histogram", [1, 2, 3]), "'histogram' must be a list of 256 counts"),
        (lambda r: r["strings"].pop("MZ"), "strings.MZ"),
        (lambda r: r["section"]["sections"][0].pop("props"), "section.sections[0].props"),
        (lambda r: r.__setitem__("exports", "nope"), "'exports' must be a list of strings"),
        (lambda r: r["imports"].__setitem__("kernel32.dll", "LoadLibraryA"), "imports['kernel32.dll']"),
        (lambda r: r["header"]["optional"].__setitem__("dll_characteristics", "NX_COMPAT"),
         "'header.optional.dll_characteristics' must be a list"),
        (lambda r: r["section"]["sections"][1].__setitem__("props", "MEM_READ"),
         "'section.sections[1].props' must be a list"),
    ],
)
def test_malformed_records_name_the_problem(schema: E.EmberV2Schema, mutate: Any, message: str) -> None:
    good = RECORDS[0]
    bad = json.loads(json.dumps(RECORDS[2]))
    mutate(bad)
    with pytest.raises(ValueError) as ei:
        schema.vectorize_raw_batch([good, bad])
    text = str(ei.value)
    assert message in text and "record 1" in text and bad["sha256"] in text


def test_non_dict_record_rejected(schema: E.EmberV2Schema) -> None:
    with pytest.raises(ValueError, match="expected a JSON object"):
        schema.vectorize_raw_batch([RECORDS[0], ["not", "a", "record"]])  # type: ignore[list-item]


# --------------------------------------------------------------------------------------------------
# Byte-level helpers used by featurize (pure numpy, no LIEF needed)
# --------------------------------------------------------------------------------------------------


def _naive_byteentropy(data: bytes) -> np.ndarray:
    a = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros((16, 16), dtype=np.int64)
    blocks = [a] if a.size < 2048 else [a[i : i + 2048] for i in range(0, a.size - 2048 + 1, 1024)]
    for blk in blocks:
        c = np.bincount(blk >> 4, minlength=16)
        p = c[c > 0].astype(np.float32) / np.float32(2048)
        h = np.float32(np.sum(-p * np.log2(p))) * np.float32(2)
        out[min(int(h * np.float32(2)), 15)] += c
    return out.reshape(-1)


@pytest.mark.parametrize("n", [0, 100, 2047, 2048, 3071, 3072, 10_000, 70_000])
def test_byte_entropy_histogram_matches_windowed_definition(n: int) -> None:
    rng = np.random.default_rng(n)
    # mix of low- and high-entropy regions so many bins are exercised
    data = bytearray(rng.integers(0, 256, n, dtype=np.uint8).tobytes())
    for k in range(0, n, 5000):
        data[k : k + 1500] = bytes(rng.integers(0, 4, min(1500, n - k), dtype=np.uint8))
    data = bytes(data)
    got = E.byte_entropy_histogram(data)
    np.testing.assert_array_equal(got, _naive_byteentropy(data))


def test_string_features() -> None:
    data = b"\x00\x01hello world\x00abc\x00C:\\Windows\\x.dll\x00http://a.b https://c.d\x00HKEY_LOCAL\x00MZ\x90"
    s = E.string_features(data)
    # runs of >= 5 printable bytes: "hello world", "C:\\Windows\\x.dll", "http://a.b https://c.d",
    # "HKEY_LOCAL" ("abc" and "MZ" are too short)
    assert s["numstrings"] == 4
    assert s["avlength"] == (11 + 16 + 22 + 10) / 4
    assert s["paths"] == 1 and s["urls"] == 2 and s["registry"] == 1 and s["MZ"] == 1
    assert s["printables"] == sum(s["printabledist"]) and len(s["printabledist"]) == 96
    assert 0 < s["entropy"] < 7
    empty = E.string_features(b"\x00\x01\x02")
    assert empty["numstrings"] == 0 and empty["avlength"] == 0 and empty["entropy"] == 0.0


# --------------------------------------------------------------------------------------------------
# featurize (raw PE bytes; needs LIEF). Only benign setuptools launcher executables are used.
# --------------------------------------------------------------------------------------------------

needs_lief = pytest.mark.skipif(not E.lief_available(), reason="lief not installed (the featurize extra)")


def test_featurize_available_reflects_lief_import(schema: E.EmberV2Schema, monkeypatch: pytest.MonkeyPatch) -> None:
    assert schema.featurize_available() == E.lief_available()
    monkeypatch.setitem(sys.modules, "lief", None)  # makes "import lief" raise ImportError
    assert schema.featurize_available() is False
    with pytest.raises(FeaturizeUnavailable, match="pip install"):
        schema.featurize(b"MZ" + b"\x00" * 64)


@needs_lief
@pytest.mark.parametrize("exe", _setuptools_exes(), ids=lambda p: p.name)
def test_featurize_setuptools_exe(schema: E.EmberV2Schema, exe: Path) -> None:
    data = exe.read_bytes()
    v = schema.featurize(data)
    assert v.shape == (2381,) and v.dtype == np.float32 and np.isfinite(v).all()
    names = schema.feature_names()
    at = {n: v[i] for i, n in enumerate(names)}
    # byte-level groups are exact
    hist = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float32)
    np.testing.assert_array_equal(v[:256], hist / hist.sum())
    assert at["general.size"] == len(data)
    # PE fields read independently from the headers
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    machine, n_sections, timestamp = struct.unpack_from("<HHI", data, pe + 4)
    assert at["header.coff.timestamp"] == np.float32(timestamp)
    assert at["section.num_sections"] == n_sections
    raw = schema.raw_features(data)
    assert raw["header"]["coff"]["machine"] == {0x14C: "I386", 0x8664: "AMD64"}[machine]
    assert raw["header"]["optional"]["magic"] == ("PE32" if machine == 0x14C else "PE32_PLUS")
    assert "kernel32.dll" in {k.lower() for k in raw["imports"]}
    assert raw["section"]["entry"] == ".text"
    b, _ = E.hash_bucket("kernel32.dll", 256)
    assert v[943 + b] != 0
    assert raw["sha256"] == hashlib.sha256(data).hexdigest()
    assert json.loads(json.dumps(raw)) == raw  # a valid EMBER-style raw record
    np.testing.assert_array_equal(schema.vectorize_raw(raw), v)


@needs_lief
def test_featurize_distinguishes_architectures(schema: E.EmberV2Schema) -> None:
    exes = {p.name: p for p in _setuptools_exes()}
    if not {"cli-32.exe", "cli-64.exe"} <= set(exes):
        pytest.skip("setuptools launcher executables not found")
    v32 = schema.featurize(exes["cli-32.exe"].read_bytes())
    v64 = schema.featurize(exes["cli-64.exe"].read_bytes())
    machine = slice(627, 637)
    assert not np.array_equal(v32[machine], v64[machine])


@needs_lief
def test_featurize_non_pe_input_keeps_byte_features(schema: E.EmberV2Schema) -> None:
    data = b"just some text, not an executable\n" * 200
    v = schema.featurize(data)
    assert np.isfinite(v).all()
    assert v[616] == len(data)  # general.size
    assert v[E.GROUP_SLICES["imports"]].sum() == 0 and v[688] == 0  # no imports, no sections


# --------------------------------------------------------------------------------------------------
# Real data (optional)
# --------------------------------------------------------------------------------------------------


def _real_corpus_dir() -> Path | None:
    for var in ("MALVALID_EMBER2018_DIR", "MALVALID_EMBER2018"):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("MALVALID_CORPUS_DIR"):
        return Path(os.environ["MALVALID_CORPUS_DIR"]) / "ember_v2_2018"
    return None


@pytest.mark.ember
def test_embedded_records_match_canonical_corpus(schema: E.EmberV2Schema) -> None:
    d = _real_corpus_dir()
    if d is None or not (d / "manifest.json").exists():
        pytest.skip("canonical ember_v2_2018 corpus not available (set MALVALID_EMBER2018_DIR)")
    from malvalid.corpora.base import load_corpus_dir

    c = load_corpus_dir(d, verify=False)
    rows = np.array(RECORD_CORPUS_ROWS)
    assert [c.sha256[i] for i in rows] == [r["sha256"] for r in RECORDS]
    np.testing.assert_array_equal(c.take(rows), schema.vectorize_raw_batch(RECORDS))
