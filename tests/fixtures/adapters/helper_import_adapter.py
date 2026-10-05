"""Test adapter: imports one helper module that sits next to it and one found via $PYTHONPATH
(both written by the test), as researchers' adapters commonly do."""

import mg_path_helper
import mg_side_helper
import numpy as np


class HelperDetector:
    feature_version = "toy_v1"
    model_kind = "custom"
    operating_threshold = 0.5
    training_hashes_path = None
    training_cutoff = None

    @classmethod
    def load(cls):
        return cls()

    def predict_proba(self, X):
        return np.full(len(X), mg_side_helper.VALUE * mg_path_helper.VALUE)

    def predict(self, X):
        return (self.predict_proba(X) >= self.operating_threshold).astype(np.int8)
