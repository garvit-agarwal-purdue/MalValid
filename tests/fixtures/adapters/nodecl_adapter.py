"""Test adapter: incomplete / invalid declarations (inspect_adapter must reject it)."""

import numpy as np


class HalfDeclaredDetector:
    feature_version = "toy_v1"
    model_kind = "lightgbm"
    # operating_threshold is missing
    training_hashes_path = "does_not_exist.txt"
    training_cutoff = "last spring"

    @classmethod
    def load(cls):
        return cls()

    def predict_proba(self, X):
        return np.zeros(len(X))

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int8)
