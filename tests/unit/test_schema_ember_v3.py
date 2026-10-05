"""ember_v3 FeatureSchema: pinned layout, names/controllability, vectorizer, raw-PE featurizer."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from malvalid.core import FeaturizeUnavailable
from malvalid.schemas import _thrember_port as tp
from malvalid.schemas.base import Controllability as C
from malvalid.schemas.ember_v3 import CATEGORICAL_FEATURES, EmberV3Schema

REPO = Path(__file__).resolve().parents[2]
SETUPTOOLS = REPO / ".venv/lib/python3.11/site-packages/setuptools"
BENIGN_EXES = sorted(SETUPTOOLS.glob("*.exe"))
# Optional local reference data (thrember checkout under ref_ember2024/, EMBER2024 files, corpora/):
# set MALVALID_TEST_DATA to enable the oracle/real-data tests; they skip otherwise.
DATA = Path(os.environ.get("MALVALID_TEST_DATA", "/nonexistent-malvalid-test-data"))
THREMBER_FEATURES = DATA / "ref_ember2024" / "src" / "thrember" / "features.py"

PINNED = [
    ("general", 0, 7), ("histogram", 7, 263), ("byteentropy", 263, 519), ("strings", 519, 696),
    ("header", 696, 770), ("section", 770, 994), ("imports", 994, 2276), ("exports", 2276, 2405),
    ("datadirectories", 2405, 2439), ("richheader", 2439, 2472), ("authenticode", 2472, 2480),
    ("pefilewarnings", 2480, 2568),
]


@pytest.fixture(scope="module")
def schema() -> EmberV3Schema:
    return EmberV3Schema()


@pytest.fixture(scope="module")
def names(schema: EmberV3Schema) -> list[str]:
    return schema.feature_names()


# --------------------------------------------------------------------------------------------------
# layout, names, controllability
# --------------------------------------------------------------------------------------------------


def test_pinned_group_boundaries(schema: EmberV3Schema) -> None:
    schema.check_layout()
    assert schema.dim == tp.DIM == 2568
    assert [(g.name, g.start, g.stop) for g in schema.groups()] == PINNED
    assert {g: tp.OFFSETS[g] for g, _, _ in PINNED} == {g: a for g, a, _ in PINNED}


def test_registry_resolves_ember_v3() -> None:
    from malvalid import registry

    assert isinstance(registry.get_schema("ember_v3"), EmberV3Schema)


def test_feature_names_unique_and_positioned(names: list[str]) -> None:
    assert len(names) == 2568 and len(set(names)) == 2568
    expect = {
        0: "general.size", 1: "general.entropy", 7: "histogram[0x00]", 262: "histogram[0xff]",
        263: "byteentropy[H0,B0]", 518: "byteentropy[H15,B15]", 519: "strings.numstrings",
        522: "strings.printabledist[0x20]", 617: "strings.printabledist[0x7f]", 618: "strings.entropy",
        696: "header.coff.timestamp", 701: "header.coff.machine", 702: "header.optional.subsystem",
        770: "section.n_sections", 994: "imports.n_functions", 995: "imports.n_libraries",
        996: "imports.libraries_hashed[0]", 1252: "imports.functions_hashed[0]",
        2275: "imports.functions_hashed[1023]", 2276: "exports.nonempty_x128", 2405: "datadirectories.EXPORT.size",
        2437: "datadirectories.has_relocs", 2439: "richheader.n_pairs", 2472: "authenticode.num_certs",
        2567: "pefilewarnings.count",
    }
    for i, nm in expect.items():
        assert names[i] == nm, (i, names[i], nm)
    for g in EmberV3Schema().groups():  # every name starts with its group's name
        assert all(n.startswith(g.name) for n in names[g.start:g.stop]), g.name
    rx = [n for n in names if n.startswith("strings.regex[")]  # thrember's sorted order
    assert rx == [f"strings.regex[{r}]" for r in sorted(tp.STRING_REGEXES)]


def test_controllability(schema: EmberV3Schema, names: list[str]) -> None:
    ctrl = dict(zip(names, schema.feature_controllability()))
    assert ctrl["header.coff.timestamp"] is C.CONTROLLABLE
    assert ctrl["header.optional.checksum"] is C.CONTROLLABLE
    assert ctrl["header.coff.machine"] is C.FIXED
    assert ctrl["header.optional.address_of_entrypoint"] is C.FIXED
    assert ctrl["general.size"] is C.APPEND_ONLY
    assert ctrl["imports.functions_hashed[17]"] is C.APPEND_ONLY
    assert ctrl["histogram[0x41]"] is C.DERIVED
    assert ctrl["section.sizes_hashed[3]"] is C.CONTROLLABLE  # keyed by the (free) section name
    assert ctrl["richheader.entries_hashed[0]"] is C.CONTROLLABLE
    assert ctrl["header.dos.e_magic"] is C.FIXED and ctrl["header.dos.e_cblp"] is C.CONTROLLABLE
    counts = {lvl: int(schema.mask(lvl).sum()) for lvl in C}
    assert all(v > 0 for v in counts.values()), counts
    assert sum(counts.values()) == schema.dim
    ca = schema.controllability_array()
    assert ca.shape == (2568,) and set(ca.tolist()) == set(C)


def test_info_and_categorical(schema: EmberV3Schema, names: list[str]) -> None:
    info = schema.info()
    assert info["name"] == "ember_v3" and info["dim"] == 2568
    assert info["categorical_features"] == list(CATEGORICAL_FEATURES)
    assert [names[i] for i in CATEGORICAL_FEATURES] == [
        "general.is_pe", "general.start_bytes[0]", "general.start_bytes[1]", "general.start_bytes[2]",
        "general.start_bytes[3]", "header.coff.machine", "header.optional.subsystem",
    ]
    json.dumps(info)


# --------------------------------------------------------------------------------------------------
# vectorize_raw on a hand-built raw dict (expectations computed independently of the port)
# --------------------------------------------------------------------------------------------------


def _hashed(tokens: list, width: int, kind: str, signed: bool) -> np.ndarray:
    from sklearn.feature_extraction import FeatureHasher

    return FeatureHasher(width, input_type=kind, alternate_sign=signed).transform([tokens]).toarray()[0]


def _raw() -> dict:
    dist = [0] * 96
    dist[ord("a") - 0x20] = 12
    dist[ord("B") - 0x20] = 6
    return {
        "sha256": "0" * 64,
        "general": {"size": 4096, "entropy": 5.25, "is_pe": 1, "start_bytes": [77, 90, 144, 0]},
        "histogram": [16] * 256,
        "byteentropy": [0] * 255 + [8],
        "strings": {"numstrings": 3, "avlength": 6.0, "printables": 18, "printabledist": dist, "entropy": 0.92,
                    "string_counts": {"http://": 2, "dos_msg": 1, "not-a-regex": 9}},
        "header": {
            "coff": {"timestamp": 1700000000, "machine": "IMAGE_FILE_MACHINE_AMD64", "number_of_sections": 1,
                     "characteristics": ["EXECUTABLE_IMAGE", "DLL"], "sizeof_optional_header": 240},
            "optional": {"subsystem": "IMAGE_SUBSYSTEM_WINDOWS_GUI", "dll_characteristics": ["NX_COMPAT"],
                         "major_linker_version": 14, "sizeof_image": 8192, "image_base": 0x180000000},
            "dos": {"e_magic": 23117, "e_lfanew": 232},
        },
        "section": {
            "entry": ".text",
            "sections": [{"name": ".text", "size": 1024, "entropy": 6.0, "vsize": 800, "size_ratio": 0.25,
                          "vsize_ratio": 1.28, "props": ["CNT_CODE", "MEM_EXECUTE", "MEM_READ"]}],
            "overlay": {"size": 40, "size_ratio": 0.01, "entropy": 3.0},
        },
        "imports": {"KERNEL32.dll": ["CreateFileW", "ReadFile"]},
        "exports": ["Foo"],
        "datadirectories": [{"has_relocs": 1, "has_dynamic_relocs": 0},
                            {"name": "EXPORT", "size": 10, "virtual_address": 20},
                            {"name": "IMPORT", "size": 30, "virtual_address": 40},
                            {"name": "RESERVED", "size": 5, "virtual_address": 6}],
        "richheader": [0x00FF0000 | 30148, 5, 0x01030000 | 33030, 7],
        "authenticode": {"num_certs": 1, "chain_max_depth": 3, "latest_signing_time": 1700000100},
        "pefilewarnings": ["Byte 0x...", "Suspicious flags set for section..."],
    }


def test_vectorize_raw_hand_built(schema: EmberV3Schema, names: list[str]) -> None:
    x = schema.vectorize_raw(_raw())
    assert x.shape == (2568,) and x.dtype == np.float32
    v = dict(zip(names, x.tolist()))
    assert v["general.size"] == 4096 and v["general.entropy"] == 5.25 and v["general.is_pe"] == 1
    assert [v[f"general.start_bytes[{i}]"] for i in range(4)] == [77, 90, 144, 0]
    np.testing.assert_allclose(x[7:263], 1 / 256)
    assert v["byteentropy[H15,B15]"] == 1.0 and x[263:519].sum() == 1.0
    assert v["strings.printables"] == 18 and v["strings.entropy"] == pytest.approx(0.92)
    assert v["strings.printabledist[0x61]"] == pytest.approx(12 / 18)
    assert v["strings.printabledist[0x42]"] == pytest.approx(6 / 18)
    assert v["strings.regex[http://]"] == 2 and v["strings.regex[dos_msg]"] == 1
    assert v["header.coff.machine"] == tp.MACHINE_INDEX["IMAGE_FILE_MACHINE_AMD64"] == 32
    assert v["header.optional.subsystem"] == 2
    assert v["header.coff.characteristics.DLL"] == 1 and v["header.coff.characteristics.EXECUTABLE_IMAGE"] == 1
    assert v["header.coff.characteristics.32BIT_MACHINE"] == 0
    assert v["header.optional.dll_characteristics.NX_COMPAT"] == 1
    assert v["header.dos.e_lfanew"] == 232 and v["header.optional.image_base"] == float(np.float32(0x180000000))
    # section general block: max over sections + overlay + 0, min always 0 (upstream quirk)
    assert (v["section.n_sections"], v["section.n_rx"], v["section.n_w"]) == (1, 1, 0)
    assert v["section.max_entropy"] == 6.0 and v["section.min_entropy"] == 0.0
    assert v["section.max_size_ratio"] == 0.25 and v["section.max_vsize_ratio"] == pytest.approx(1.28)
    assert v["section.overlay.size"] == 40 and v["section.overlay.entropy"] == 3.0
    np.testing.assert_array_equal(x[781:831], _hashed([(".text", 1024)], 50, "pair", True))
    np.testing.assert_array_equal(x[981:991], _hashed([".text"], 10, "string", True))
    # imports: lower-cased library, unsigned hashing
    assert v["imports.n_functions"] == 2 and v["imports.n_libraries"] == 1
    np.testing.assert_array_equal(x[996:1252], _hashed(["kernel32.dll"], 256, "string", False))
    np.testing.assert_array_equal(
        x[1252:2276], _hashed(["kernel32.dll:CreateFileW", "kernel32.dll:ReadFile"], 1024, "string", False))
    # exports: 128 marker (upstream quirk) + signed hashing
    assert v["exports.nonempty_x128"] == 128
    np.testing.assert_array_equal(x[2277:2405], _hashed(["Foo"], 128, "string", True))
    # data directories: last entry (RESERVED) is never written upstream
    assert v["datadirectories.EXPORT.size"] == 10 and v["datadirectories.IMPORT.virtual_address"] == 40
    assert v["datadirectories.RESERVED.size"] == 0 and v["datadirectories.has_relocs"] == 1
    assert v["richheader.n_pairs"] == 2
    np.testing.assert_array_equal(
        x[2440:2472], _hashed([(str(0x00FF0000 | 30148), 5), (str(0x01030000 | 33030), 7)], 32, "pair", True))
    assert v["authenticode.num_certs"] == 1 and v["authenticode.chain_max_depth"] == 3
    assert v["pefilewarnings[Byte 0x...]"] == 1 and v["pefilewarnings.count"] == 2


def test_vectorize_missing_groups_are_zero(schema: EmberV3Schema) -> None:
    x = schema.vectorize_raw(
        {"sha256": "1" * 64, "general": {"size": 10, "entropy": 0.0, "is_pe": 0, "start_bytes": [1, 2, 3, 4]}})
    assert x[0] == 10 and np.count_nonzero(x) == 5  # size + 4 start bytes


def test_vectorize_batch_matches_single(schema: EmberV3Schema) -> None:
    a = _raw()
    b = _raw()
    b["imports"] = {}
    b["exports"] = []
    b["general"]["size"] = 1
    X = schema.vectorize_raw_batch([a, b, a])
    assert X.shape == (3, 2568) and X.dtype == np.float32
    np.testing.assert_array_equal(X[0], schema.vectorize_raw(a))
    np.testing.assert_array_equal(X[1], schema.vectorize_raw(b))
    np.testing.assert_array_equal(X[2], X[0])
    assert schema.vectorize_raw_batch([]).shape == (0, 2568)


# --------------------------------------------------------------------------------------------------
# raw bytes -> vector (pefile)
# --------------------------------------------------------------------------------------------------

needs_pefile = pytest.mark.skipif(not tp.pefile_available(), reason="pefile not installed")


def test_featurize_available_iff_pefile(schema: EmberV3Schema) -> None:
    assert schema.featurize_available() == (importlib.util.find_spec("pefile") is not None)


def test_featurize_unavailable_message(schema: EmberV3Schema, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tp, "pefile_available", lambda: False)
    assert not schema.featurize_available()
    with pytest.raises(FeaturizeUnavailable, match="pefile"):
        schema.featurize(b"MZ")


@needs_pefile
@pytest.mark.skipif(not BENIGN_EXES, reason="setuptools launcher executables not found")
def test_featurize_benign_setuptools_exes(schema: EmberV3Schema, names: list[str]) -> None:
    idx = {n: i for i, n in enumerate(names)}
    machines = {"32": "IMAGE_FILE_MACHINE_I386", "64": "IMAGE_FILE_MACHINE_AMD64", "arm64": "IMAGE_FILE_MACHINE_ARM64"}
    for exe in BENIGN_EXES:
        data = exe.read_bytes()
        x = schema.featurize(data)
        assert x.shape == (2568,) and x.dtype == np.float32 and np.all(np.isfinite(x)), exe.name
        assert x[idx["general.size"]] == len(data) and x[idx["general.is_pe"]] == 1
        assert list(x[3:7]) == list(data[:4])
        assert abs(float(x[7:263].sum()) - 1.0) < 1e-5 and abs(float(x[263:519].sum()) - 1.0) < 1e-5
        assert x[idx["imports.n_functions"]] > 0 and x[idx["section.n_sections"]] >= 3
        arch = exe.stem.split("-")[-1] if "-" in exe.stem else None
        if arch in machines:
            assert x[idx["header.coff.machine"]] == tp.MACHINE_INDEX[machines[arch]], exe.name
        raw = schema.raw_features(data)
        np.testing.assert_array_equal(schema.vectorize_raw(raw), x)
        json.dumps(raw)  # raw features are plain JSON (like the EMBER2024 JSONL rows)


@needs_pefile
def test_featurize_non_pe_bytes(schema: EmberV3Schema, names: list[str]) -> None:
    data = b"hello world, this is not a PE file" * 10
    x = schema.featurize(data)
    assert x[0] == len(data) and x[names.index("general.is_pe")] == 0
    assert x[names.index("imports.n_functions")] == 0


def _load_thrember(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import upstream thrember's features.py as an oracle, with a stub ``signify`` if absent.

    The stub reports "no signatures", which is what signify does for unsigned files such as the
    setuptools launchers.
    """
    if importlib.util.find_spec("signify") is None:
        signify = types.ModuleType("signify")
        auth = types.ModuleType("signify.authenticode")
        exc = types.ModuleType("signify.exceptions")

        class _Err(Exception):
            pass

        exc.SignerInfoParseError = type("SignerInfoParseError", (_Err,), {})
        exc.ParseError = type("ParseError", (_Err,), {})

        class SignedPEFile:
            def __init__(self, f):  # stub
                self.f = f

            def iter_signed_datas(self):
                return iter(())

        auth.SignedPEFile = SignedPEFile
        signify.authenticode = auth
        signify.exceptions = exc
        for k, m in (("signify", signify), ("signify.authenticode", auth), ("signify.exceptions", exc)):
            monkeypatch.setitem(sys.modules, k, m)
    spec = importlib.util.spec_from_file_location("_thrember_oracle_features", THREMBER_FEATURES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@needs_pefile
@pytest.mark.skipif(not THREMBER_FEATURES.exists() or not BENIGN_EXES, reason="thrember reference / exes not present")
def test_featurize_matches_upstream_thrember(schema: EmberV3Schema, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle = _load_thrember(monkeypatch).PEFeatureExtractor()
    assert oracle.dim == schema.dim
    for exe in BENIGN_EXES:
        data = exe.read_bytes()
        ref = oracle.feature_vector(data)
        np.testing.assert_array_equal(schema.featurize(data), ref, err_msg=exe.name)
        # the vectorizer alone matches too, on upstream's own raw dict
        np.testing.assert_array_equal(schema.vectorize_raw(json.loads(json.dumps(oracle.raw_features(data)))), ref)


# --------------------------------------------------------------------------------------------------
# real EMBER2024 rows (optional; needs the local data)
# --------------------------------------------------------------------------------------------------

CHALLENGE_ZIP = DATA / "ember2024" / "challenge.zip"
CORPUS_V3 = DATA / "corpora" / "ember_v3_2024"


@pytest.mark.skipif(not (CHALLENGE_ZIP.exists() and (CORPUS_V3 / "manifest.json").exists()),
                    reason="EMBER2024 challenge.zip / canonical corpus not present")
def test_vectorize_real_rows_match_canonical_corpus(schema: EmberV3Schema) -> None:
    """Rows vectorized now are bit-identical to the rows in the pinned canonical build."""
    import zipfile

    from malvalid.corpora.base import load_corpus_dir

    corpus = load_corpus_dir(CORPUS_V3, verify=False)
    rows = []
    with zipfile.ZipFile(CHALLENGE_ZIP) as zf:
        member = sorted(n for n in zf.namelist() if n.endswith(".jsonl"))[0]
        with zf.open(member) as f:
            for line in f:
                r = json.loads(line)
                if r.get("file_type") in ("Win32", "Win64", "Dot_Net"):
                    rows.append(r)
                if len(rows) >= 25:
                    break
    hi = corpus.hash_index()
    idx = np.array([hi[r["sha256"].lower()] for r in rows])
    np.testing.assert_array_equal(schema.vectorize_raw_batch(rows), corpus.take(idx))
