"""Test adapter: violates the predict_proba contract by returning (n, 2) class probabilities."""

from pathlib import Path

import lightgbm as lgb
import numpy as np

HERE = Path(__file__).resolve().parent


class BrokenDetector:
    feature_version = "toy_v1"
    model_kind = "lightgbm"
    operating_threshold = 0.5
    training_hashes_path = None
    training_cutoff = None
    model_path = "model.txt"

    def __init__(self, booster):
        self.booster = booster

    @classmethod
    def load(cls):
        return cls(lgb.Booster(model_file=str(HERE / cls.model_path)))

    def predict_proba(self, X):
        p = self.booster.predict(X)
        return np.column_stack([1.0 - p, p])  # sklearn-style: wrong for malvalid

    def predict(self, X):
        return (self.booster.predict(X) >= self.operating_threshold).astype(np.int8)
