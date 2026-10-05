"""Release hygiene: what the sdist ships, licensing statements, and doc facts that drifted before.

Regression tests for release-review findings docs-003, docs-004, docs-005, docs-007, docs-009,
docs-010, docs-011, docs-012 and xcomp-readme-full-run-timing.
"""

from __future__ import annotations

import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INTERNAL_DOCS = ("DESIGN.md", "docs/BUILD_CONTRACT.md", "docs/WEB_CONTRACT.md", "docs/SPEC.html")


def _read(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present (e.g. running from an sdist)")
    return path.read_text(encoding="utf-8")


def _sdist_excludes() -> list[str]:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return cfg["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]


# --- docs-011: sdist contents --------------------------------------------------------------------


def test_sdist_exclude_list_covers_internal_build_docs():
    excludes = {e.lstrip("/") for e in _sdist_excludes()}
    for rel in INTERNAL_DOCS:
        assert rel in excludes, f"{rel} must be excluded from the sdist"
    assert "examples/" in excludes


def test_no_duplicate_spec_or_stray_html_at_repo_root():
    assert not list(ROOT.glob("*.html")), "the product spec lives only in docs/SPEC.html"


def test_built_sdist_omits_internal_docs_and_ember_fixture(tmp_path):
    sdist = pytest.importorskip("hatchling.builders.sdist")
    out = next(iter(sdist.SdistBuilder(str(ROOT)).build(directory=str(tmp_path), versions=["standard"])))
    with tarfile.open(out) as tar:
        names = {n.split("/", 1)[1] for n in tar.getnames() if "/" in n}
    for rel in INTERNAL_DOCS:
        assert rel not in names
    assert not any(n.endswith(".html") and "/" not in n for n in names)
    assert not any(n.startswith("examples/") for n in names)
    assert not any(n.startswith("tests/fixtures/") and n.endswith(".jsonl") for n in names)
    assert {"LICENSE", "NOTICE", "README.md"} <= names


def test_gitignore_covers_example_symlinks_and_logs():
    text = _read(".gitignore")
    assert "examples/lightgbm_ember2018/ember_model_2018.txt" in text
    assert "examples/*/train.log" in text
    assert "examples/*/*.model" in text


def test_shipped_docs_have_no_maintainer_scratch_paths():
    files = [ROOT / "README.md", ROOT / "NOTICE"]
    files += sorted((ROOT / "docs" / "modules").glob("*.md"))
    files += sorted((ROOT / "docs" / "results").glob("*"))
    files += sorted((ROOT / "examples").glob("**/README.md"))
    bad = [str(f.relative_to(ROOT)) for f in files if f.is_file() and "/scratch/" in f.read_text(encoding="utf-8")]
    assert not bad, f"maintainer-specific /scratch paths in shipped docs: {bad}"


def test_readme_does_not_link_internal_build_notes():
    text = _read("README.md")
    assert "](DESIGN.md)" not in text
    assert "](docs/BUILD_CONTRACT.md)" not in text


# --- docs-009: bundled EMBER data ----------------------------------------------------------------


def test_no_bundled_ember_jsonl_fixtures():
    assert not list((ROOT / "tests" / "fixtures").glob("*.jsonl"))


def test_notice_covers_embedded_ember2018_records_with_mit_text():
    text = _read("NOTICE")
    assert "three EMBER2018 raw feature records" in text
    assert "tests/unit/test_schema_ember_v2.py" in text
    assert "Permission is hereby granted, free of charge" in text
    assert "NOT bundled with malvalid" not in text


# --- docs-003 / docs-010: licenses ---------------------------------------------------------------


def test_ember2024_is_described_as_apache_not_mit():
    notice = _read("NOTICE")
    assert "EMBER2024 dataset and benchmark models: Apache-2.0" in notice
    readme = _read("README.md")
    assert "EMBER2018 and EMBER2024 data files are\nMIT-licensed" not in readme
    assert "EMBER2024 dataset and benchmark models are Apache-2.0" in readme
    assert "Apache-2.0" in _read("docs/modules/corpus_ember2024.md")


def test_notice_does_not_claim_all_dependencies_are_mit_apache_bsd3():
    text = _read("NOTICE")
    assert "carry permissive licenses\n(MIT, Apache-2.0, or BSD-3-Clause)" not in text
    assert "nvidia-nccl-cu12" in text


# --- docs-004 / docs-005 / docs-007 / docs-012: measured facts ------------------------------------


@pytest.mark.parametrize(
    "rel", ["README.md", "examples/lightgbm_ember2024/README.md", "docs/modules/corpus_ember2024.md"]
)
def test_ember2024_m2_window_count_is_three(rel):
    text = " ".join(_read(rel).split())
    assert "four monthly" not in text and "four on EMBER2024" not in text
    assert "three" in text


def test_examples_index_timings_match_measured_runs():
    text = _read("examples/README.md")
    assert "few minutes" not in text
    assert "full run ~10 min" in text
    assert "training ~5 min on 8 threads" in text


def test_report_size_and_challenge_detection_are_current():
    assert "80–200 KB" not in _read("docs/modules/report.md")
    assert "challenge set: 70.7%" not in _read("docs/results/REAL_DATA_RESULTS.md")


def test_readme_adapter_snippet_is_not_presented_as_the_synthetic_demo_adapter():
    text = _read("README.md")
    assert "This is the whole adapter for the synthetic demo" not in text
    assert "operating_threshold = 0.8336" not in text


def test_readme_headline_results_table_has_no_partial_run_row():
    text = _read("README.md")
    assert "| 100.0 (M0 + M5 only) | n/a |" not in text


def test_readme_quickstart_step2_returns_to_repo_root():
    text = _read("README.md")
    step2 = text[text.index("### 2. The real-data path") :]
    intro = " ".join(step2[: step2.index("```bash")].split())
    assert "from the repository root" in intro and "cd ../.." in intro


# --- fresh-clone demo: the web UI's "Run the synthetic demo" button needs these files ------------------

DEMO_FILES = ("adapter.py", "gate.yaml", "model.txt", "threshold.txt", "train_sha256.txt")


def _git(*args: str) -> str:
    import shutil
    import subprocess

    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False)
    if r.returncode not in (0, 1):
        pytest.skip(f"git {args[0]} failed: {r.stderr.strip()}")
    return r.stdout


def test_synthetic_demo_model_is_published_in_the_repository():
    tracked = set(_git("ls-files", "examples/synthetic_demo").split())
    missing = [n for n in DEMO_FILES if f"examples/synthetic_demo/{n}" not in tracked]
    assert not missing, f"not tracked by git, so a fresh clone has no demo button: {missing}"
    ignored = _git("check-ignore", "--no-index", *[f"examples/synthetic_demo/{n}" for n in DEMO_FILES]).split()
    assert not ignored, f".gitignore hides demo files from a fresh clone: {ignored}"


def test_checkout_offers_the_synthetic_demo():
    import malvalid
    from malvalid.web.onboarding import demo_dir, demo_submission

    if not Path(malvalid.__file__).resolve().is_relative_to(ROOT.resolve()):
        pytest.skip("malvalid is not imported from this checkout (non-editable install)")
    if not all((ROOT / "examples" / "synthetic_demo" / n).is_file() for n in DEMO_FILES):
        pytest.skip("examples/ not present (e.g. running from an sdist)")
    d = demo_dir()
    assert d is not None and d.resolve() == (ROOT / "examples" / "synthetic_demo").resolve()
    assert demo_submission() is not None
