"""Build throw-away adapter directories for tests: copy an adapter file next to freshly trained
model artifacts in ``tmp_path`` (no model binaries are committed to the repository)."""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent

#: Hand-assembled protocol-2 pickle whose only global is ``os.getpid`` (harmless if it ever ran,
#: but CRITICAL under malvalid's pickle policy). ``PROTO 2 | GLOBAL os getpid | EMPTY_TUPLE | REDUCE | STOP``.
GETPID_PICKLE = b"\x80\x02cos\ngetpid\n)R."


@lru_cache(maxsize=1)
def toy_corpus() -> Any:
    from malvalid.testing import make_toy_corpus

    return make_toy_corpus(n=3000, seed=0)


@lru_cache(maxsize=1)
def toy_booster_text() -> str:
    from malvalid.testing import train_toy_lgbm

    return train_toy_lgbm(toy_corpus(), rounds=30, params={"num_threads": 1}).model_to_string()


def make_adapter_dir(tmp_path: Path, adapter: str, *, model: str | None = "lgbm", subdir: str = "adapter") -> Path:
    """Copy ``tests/fixtures/adapters/<adapter>.py`` into ``tmp_path/subdir`` with its model files.

    ``model``: ``"lgbm"`` (model.txt), ``"joblib"`` (model.joblib, sklearn GBDT), ``"payload"``
    (payload.pkl = :data:`GETPID_PICKLE`) or ``None``. Returns the adapter file path.
    """
    d = Path(tmp_path) / subdir
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{adapter}.py"
    shutil.copy(HERE / f"{adapter}.py", dst)
    c = toy_corpus()
    if model == "lgbm":
        (d / "model.txt").write_text(toy_booster_text())
    elif model == "joblib":
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier

        idx = c.indices(splits=("train",))
        est = GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=0)
        est.fit(c.take(idx), c.label[idx])
        joblib.dump(est, d / "model.joblib")
    elif model == "payload":
        (d / "payload.pkl").write_bytes(GETPID_PICKLE)
    train = c.sha256[c.indices(splits=("train",))][:50]
    (d / "train_hashes.txt").write_text("\n".join(train.tolist()) + "\n")
    return dst


def eval_rows(n: int = 200) -> np.ndarray:
    c = toy_corpus()
    return np.ascontiguousarray(c.take(c.indices(splits=("test",))[:n]), dtype=np.float32)


def make_ember_v2_model(adapter_file: Path, *, dim: int = 2381, rounds: int = 5) -> Path:
    """Train a tiny LightGBM model on random ``ember_v2``-shaped vectors next to ``adapter_file``."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.random((400, dim)).astype(np.float32)
    y = (X[:, 0] + X[:, 700] > 1.0).astype(np.int32)
    params = {"objective": "binary", "num_leaves": 4, "verbose": -1, "num_threads": 1, "seed": 0}
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=rounds)
    out = Path(adapter_file).parent / "model.txt"
    booster.save_model(str(out))
    return out
