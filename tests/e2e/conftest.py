"""Shared fixtures for the malvalid end-to-end suite.

Every test here builds what it needs from scratch (a small synthetic_v2 corpus, a tiny LightGBM,
an adapter file) and drives the real CLI (``python -m malvalid ...``) as a subprocess through the
real sandbox. Nothing is faked. The suite skips cleanly when the host has no isolating sandbox
backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

SYNTHETIC_ROWS = "4000"
CORPUS = "synthetic_v2"
ADAPTER_SRC = textwrap.dedent(
    '''
    """E2E submission: a tiny LightGBM detector on the synthetic_v2 (EMBER v2) feature space."""
    from malvalid.adapter import BaseDetector


    class E2EDetector(BaseDetector):
        feature_version = "ember_v2"
        model_kind = "lightgbm"
        operating_threshold = {threshold}
        model_path = "model.txt"
        training_hashes_path = {hashes}
        training_cutoff = "2017-12"
    '''
)

# Every top-level key contract §4.4 promises in report.json.
REPORT_KEYS = (
    "schema_version", "tool", "run", "verdict", "gate", "model", "corpus", "schema",
    "training_manifest", "config", "environment", "sandbox", "modules", "artifacts",
    "warnings", "disclaimers",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: end-to-end tests that drive the real CLI and sandbox")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    here = Path(__file__).parent.resolve()
    for item in items:
        if here in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.e2e)


@dataclass
class CliResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + "\n" + self.stderr


@pytest.fixture(scope="session")
def sandbox_backend() -> str:
    """Name of an isolating sandbox backend, or skip the whole suite."""
    try:
        from malvalid.sandbox.host import probe_backends
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"malvalid.sandbox.host not importable: {e}")
    probes = probe_backends()
    for name, info in probes.items():
        if info.get("available") and info.get("network_isolated"):
            return name
    pytest.skip(f"no isolating sandbox backend on this host: {json.dumps(probes, default=str)[:400]}")


@pytest.fixture(scope="session")
def e2e_root(tmp_path_factory: pytest.TempPathFactory, sandbox_backend: str) -> Path:
    return tmp_path_factory.mktemp("malvalid_e2e")


@pytest.fixture(scope="session")
def cli_env(e2e_root: Path) -> dict[str, str]:
    """Environment for every CLI subprocess: private tiny corpus root, capped threads."""
    env = dict(os.environ)
    env.update(
        MALVALID_CORPUS_DIR=str(e2e_root / "corpora"),
        MALVALID_SYNTHETIC_ROWS=SYNTHETIC_ROWS,
        OMP_NUM_THREADS="4",
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for k in ("MALVALID_SYNTHETIC_SEED", "MALVALID_ALLOW_PICKLE"):
        env.pop(k, None)
    return env


@pytest.fixture(scope="session")
def run_cli(cli_env: dict[str, str], e2e_root: Path):
    def _run(*args: str | os.PathLike[str], timeout: float = 600) -> CliResult:
        p = subprocess.run(
            [sys.executable, "-m", "malvalid", *map(str, args)],
            env=cli_env, cwd=str(e2e_root), capture_output=True, text=True, timeout=timeout,
        )
        return CliResult(p.returncode, p.stdout, p.stderr)

    return _run


@pytest.fixture(scope="session")
def corpus(cli_env: dict[str, str], e2e_root: Path):
    """The tiny synthetic_v2 corpus, generated (and cached) under the private corpus root."""
    from malvalid import registry

    old = {k: os.environ.get(k) for k in ("MALVALID_CORPUS_DIR", "MALVALID_SYNTHETIC_ROWS")}
    os.environ["MALVALID_CORPUS_DIR"] = cli_env["MALVALID_CORPUS_DIR"]
    os.environ["MALVALID_SYNTHETIC_ROWS"] = SYNTHETIC_ROWS
    try:
        provider = registry.get_corpus_provider(CORPUS)
        provider = provider() if isinstance(provider, type) else provider
        return provider.load()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="session")
def trained(corpus, e2e_root: Path) -> dict[str, Any]:
    """A tiny LightGBM trained on the corpus's ``train`` split only, plus its training manifest."""
    import lightgbm as lgb
    import numpy as np

    idx = np.flatnonzero((corpus.split == "train") & (corpus.label >= 0))
    X = np.asarray(corpus.X[idx], dtype=np.float32)
    y = corpus.label[idx].astype(int)
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 15, "learning_rate": 0.1, "min_data_in_leaf": 10,
         "num_threads": 4, "seed": 0, "deterministic": True, "force_row_wise": True, "verbose": -1},
        lgb.Dataset(X, label=y), num_boost_round=40,
    )
    d = e2e_root / "model_src"
    d.mkdir()
    booster.save_model(str(d / "model.txt"))
    (d / "train_sha256.txt").write_text("\n".join(str(corpus.sha256[i]) for i in idx) + "\n")
    return {"dir": d, "booster": booster, "n_train": int(len(idx))}


def write_adapter(dst: Path, trained: dict[str, Any], *, threshold: float = 0.5,
                  with_hashes: bool = True, model: str = "model.txt") -> Path:
    """Copy the trained model next to a fresh adapter file in ``dst`` and return the adapter path."""
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(trained["dir"] / "model.txt", dst / model)
    if with_hashes:
        shutil.copy(trained["dir"] / "train_sha256.txt", dst / "train_sha256.txt")
    src = ADAPTER_SRC.format(threshold=repr(threshold),
                             hashes=repr("train_sha256.txt") if with_hashes else "None")
    src = src.replace('model_path = "model.txt"', f"model_path = {model!r}")
    p = dst / "e2e_adapter.py"
    p.write_text(src)
    return p


def write_config(path: Path, **overrides: Any) -> Path:
    """A gate policy (JSON is valid YAML) sized for the 4000-row synthetic corpus."""
    cfg: dict[str, Any] = {
        "corpus": CORPUS,
        "runtime": {"seed": 0, "threads": 4, "max_seconds_per_module": 600},
        "modules": {
            "performance": {"enabled": True, "gate": "hard", "max_fpr": 0.15, "min_detection": 0.60},
            "drift": {"enabled": True, "min_window_samples": 40, "min_aut_f1": 0.5},
            "membership_inf": {"enabled": True, "min_members": 200, "max_per_side": 1000},
            "backdoor_screen": {"enabled": True, "n_samples": 500},
            "extraction": {"enabled": False},
            "explanation": {"enabled": True, "shap_samples": 300},
        },
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    path.write_text(json.dumps(cfg, indent=2))
    return path


@pytest.fixture(scope="session")
def good_adapter(e2e_root: Path, trained) -> Path:
    return write_adapter(e2e_root / "sub_good", trained)


@pytest.fixture(scope="session")
def gate_config(e2e_root: Path) -> Path:
    return write_config(e2e_root / "gate.yaml")


@pytest.fixture(scope="session")
def happy_run(run_cli, good_adapter, gate_config, e2e_root: Path) -> dict[str, Any]:
    out = e2e_root / "run_happy"
    res = run_cli("run", "--adapter", good_adapter, "--config", gate_config, "--out", out, "--seed", "7")
    return {"res": res, "out": out, "report": load_report(out)}


def load_report(out: Path) -> dict[str, Any]:
    def _no_const(c: str) -> Any:
        raise ValueError(f"non-strict JSON constant {c!r}")

    return json.loads((out / "report.json").read_text(), parse_constant=_no_const)
