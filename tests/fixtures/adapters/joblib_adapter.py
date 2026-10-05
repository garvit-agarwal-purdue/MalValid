"""Test adapter: a scikit-learn model stored with joblib (a pickle container)."""

from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent


class JoblibDetector:
    feature_version = "toy_v1"
    model_kind = "sklearn_gbdt"
    operating_threshold = 0.5
    training_hashes_path = None
    training_cutoff = None
    model_path = "model.joblib"

    def __init__(self, est):
        self.native_model = est

    @classmethod
    def load(cls):
        return cls(joblib.load(HERE / cls.model_path))

    def predict_proba(self, X):
        return self.native_model.predict_proba(X)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= self.operating_threshold).astype(np.int8)
