"""Tiny benign models trained in-test for the model-file-only submission tests (no real data)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def make_xy(n_features: int, n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def train_lgbm(n_features: int = 2381, rounds: int = 2, seed: int = 0) -> Any:
    import lightgbm as lgb

    X, y = make_xy(n_features, seed=seed)
    return lgb.train(
        {"objective": "binary", "num_leaves": 4, "min_data_in_leaf": 5, "verbose": -1, "num_threads": 1,
         "seed": seed, "deterministic": True, "force_row_wise": True},
        lgb.Dataset(X, label=y), num_boost_round=rounds,
    )


def save_lgbm(path: Path, n_features: int = 2381) -> Any:
    b = train_lgbm(n_features)
    b.save_model(str(path))
    return b


def save_xgb(path: Path, n_features: int = 2381) -> Any:
    import xgboost as xgb

    X, y = make_xy(n_features)
    b = xgb.train({"objective": "binary:logistic", "max_depth": 2, "nthread": 1, "verbosity": 0},
                  xgb.DMatrix(X, label=y), num_boost_round=2)
    b.save_model(str(path))
    return b


def sklearn_gbdt(n_features: int = 2381) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = make_xy(n_features, n=80)
    return GradientBoostingClassifier(n_estimators=2, max_depth=2, random_state=0).fit(X, y)


def save_pickle(path: Path, n_features: int = 2381) -> Any:
    est = sklearn_gbdt(n_features)
    with open(path, "wb") as f:
        pickle.dump(est, f)
    return est


def save_joblib(path: Path, n_features: int = 2381) -> Any:
    import joblib

    est = sklearn_gbdt(n_features)
    joblib.dump(est, path)
    return est


def save_onnx(path: Path, n_features: int = 2568) -> None:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    est = sklearn_gbdt(n_features)
    onx = to_onnx(est, initial_types=[("input", FloatTensorType([None, n_features]))],
                  target_opset={"": 17, "ai.onnx.ml": 3})
    path.write_bytes(onx.SerializeToString())
