"""Training-manifest parsing (training_hashes_path + training_cutoff)."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib

import pytest

from malvalid.context import ModelDeclarations
from malvalid.core import AdapterError
from malvalid.manifest import (
    TrainingManifest,
    count_excluded_members,
    load_training_manifest,
    parse_training_cutoff,
    parse_training_hashes,
)
from malvalid.testing import make_toy_corpus, training_hashes_for


def _h(i: int) -> str:
    return hashlib.sha256(f"sample-{i}".encode()).hexdigest()


# ---- cutoff ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2018-10-31", dt.date(2018, 10, 31)),
        ("2018-10", dt.date(2018, 10, 31)),
        ("2018-11", dt.date(2018, 11, 30)),
        ("2020-02", dt.date(2020, 2, 29)),  # leap year
        ("2019-02", dt.date(2019, 2, 28)),
        ("2018-1", dt.date(2018, 1, 31)),
        ("2018", dt.date(2018, 12, 31)),
        (" 2018-10-31 ", dt.date(2018, 10, 31)),
        ("2018-10-31T23:59:59", dt.date(2018, 10, 31)),
        ("2018-10-31T23:59:59Z", dt.date(2018, 10, 31)),
        ("2018-10-31 08:00:00+02:00", dt.date(2018, 10, 31)),
        (dt.date(2017, 5, 1), dt.date(2017, 5, 1)),
        (dt.datetime(2017, 5, 1, 12), dt.date(2017, 5, 1)),
        (None, None),
        ("", None),
    ],
)
def test_parse_cutoff_accepts(value, expected):
    assert parse_training_cutoff(value) == expected


@pytest.mark.parametrize("value", ["2018-13", "2018-02-30", "31/10/2018", "Oct 2018", "18-10-31", "2018-10-31x"])
def test_parse_cutoff_rejects_with_actionable_message(value):
    with pytest.raises(AdapterError) as ei:
        parse_training_cutoff(value)
    assert "training_cutoff" in str(ei.value)


def test_parse_cutoff_rejects_non_string():
    with pytest.raises(AdapterError, match="must be a string"):
        parse_training_cutoff(20181031)  # type: ignore[arg-type]


# ---- hashes ----------------------------------------------------------------------------------------


def test_plain_hash_file_case_comments_blank_lines(tmp_path):
    hs = [_h(i) for i in range(5)]
    p = tmp_path / "train.txt"
    p.write_text(
        "# training set of my detector\n\n"
        + hs[0].upper() + "\n"
        + hs[1] + "   # inline comment\n"
        + "  " + hs[2] + "  \n"
        + hs[3] + "\n" + hs[3] + "\n"  # duplicate
        + hs[4] + "\r\n"
    )
    got = parse_training_hashes(p)
    assert got == frozenset(hs)
    assert all(h == h.lower() for h in got)


def test_bom_is_ignored(tmp_path):
    p = tmp_path / "bom.txt"
    p.write_bytes(b"\xef\xbb\xbf" + (_h(1) + "\n").encode())
    assert parse_training_hashes(p) == {_h(1)}


def test_gzip_manifest(tmp_path):
    p = tmp_path / "train.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write("\n".join(_h(i) for i in range(10)) + "\n")
    assert len(parse_training_hashes(p)) == 10


@pytest.mark.parametrize("delim", [",", "\t", ";"])
def test_csv_tsv_with_sha256_column(tmp_path, delim):
    p = tmp_path / "train.csv"
    rows = [delim.join(["label", "SHA256", "first_seen"])]
    rows += [delim.join(["1", _h(i).upper(), "2018-01-01"]) for i in range(4)]
    p.write_text("\n".join(rows) + "\n\n")
    assert parse_training_hashes(p) == {_h(i) for i in range(4)}


def test_csv_alternative_hash_column_names(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("hash,label\n" + "\n".join(f"{_h(i)},0" for i in range(3)))
    assert parse_training_hashes(p) == {_h(i) for i in range(3)}


def test_headerless_csv_first_column(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("\n".join(f"{_h(i)},1,2018-01-01" for i in range(3)))
    assert parse_training_hashes(p) == {_h(i) for i in range(3)}


def test_malformed_line_names_the_line(tmp_path):
    p = tmp_path / "train.txt"
    p.write_text(_h(0) + "\n" + "not-a-hash\n" + _h(1)[:-1] + "\n")
    with pytest.raises(AdapterError) as ei:
        parse_training_hashes(p)
    msg = str(ei.value)
    assert "line 2" in msg and "not-a-hash" in msg and "2 line(s)" in msg


def test_malformed_csv_cell_names_the_row(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text("sha256,label\n" + _h(0) + ",1\nzzz,0\n")
    with pytest.raises(AdapterError, match="line 3"):
        parse_training_hashes(p)


def test_missing_file_and_directory(tmp_path):
    with pytest.raises(AdapterError, match="not found"):
        parse_training_hashes(tmp_path / "nope.txt")
    with pytest.raises(AdapterError, match="directory"):
        parse_training_hashes(tmp_path)


def test_binary_file_rejected(tmp_path):
    p = tmp_path / "train.bin"
    p.write_bytes(b"\xff\xfe\x00\x81" * 10)
    with pytest.raises(AdapterError, match="UTF-8"):
        parse_training_hashes(p)


def test_empty_manifest_warns_but_does_not_raise(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("# nothing here\n\n")
    decl = ModelDeclarations("toy_v1", "lightgbm", 0.5, str(p), None)
    tm = load_training_manifest(decl)
    assert tm.hashes == frozenset() and tm.usable_hashes() is None
    assert tm.n_hashes == 0 and tm.warnings and "no sha256" in tm.warnings[0]


# ---- combined manifest -----------------------------------------------------------------------------


def test_load_training_manifest_and_report_counts(tmp_path):
    corpus = make_toy_corpus(n=600, seed=3)
    train = training_hashes_for(corpus, ("train",))
    eval_members = corpus.sha256[corpus.indices(splits=("test",))][:7].tolist()
    outsiders = [_h(i) for i in range(5)]
    p = tmp_path / "hashes.txt"
    p.write_text("\n".join(sorted(train) + eval_members + outsiders))
    decl = ModelDeclarations("toy_v1", "lightgbm", 0.5, str(p), "2017-12")
    tm = load_training_manifest(decl)
    assert isinstance(tm, TrainingManifest)
    assert tm.n_hashes == len(train) + 7 + 5
    assert tm.cutoff_parsed == dt.date(2017, 12, 31)
    rep = tm.to_report(corpus)
    assert rep["n_in_corpus"] == len(train) + 7
    assert rep["cutoff"] == "2017-12" and rep["cutoff_parsed"] == "2017-12-31"
    assert rep["declared"] is True
    assert count_excluded_members(corpus, tm.usable_hashes()) == 7


def test_manifest_without_declarations():
    decl = ModelDeclarations("toy_v1", "lightgbm", 0.5, None, None)
    tm = load_training_manifest(decl)
    assert tm.hashes is None and tm.cutoff_parsed is None and not tm.warnings
    rep = tm.to_report(None)
    assert rep == {
        "path": None, "declared": False, "n_hashes": None, "n_in_corpus": None,
        "cutoff": None, "cutoff_parsed": None, "warnings": [],
    }
    assert count_excluded_members(None, None) is None
    assert count_excluded_members(make_toy_corpus(n=50), None) == 0
