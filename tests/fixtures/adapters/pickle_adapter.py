"""Test adapter: two detector classes that unpickle during load() (for the pickle-guard tests).

Defining two classes also exercises class discovery: without ``--class`` the choice is ambiguous.
"""

import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class _Const:
    feature_version = "toy_v1"
    model_kind = "custom"
    operating_threshold = 0.5
    training_hashes_path = None
    training_cutoff = None

    def __init__(self, value):
        self.value = float(value)

    def predict_proba(self, X):
        return np.full(len(X), self.value)

    def predict(self, X):
        return (self.predict_proba(X) >= self.operating_threshold).astype(np.int8)


class BenignPickleDetector(_Const):
    """Unpickles a harmless in-memory dict (refused unless --allow-pickle)."""

    @classmethod
    def load(cls):
        state = pickle.loads(pickle.dumps({"value": 0.25}))
        return cls(state["value"])


class GlobalPickleDetector(_Const):
    """Unpickles ``payload.pkl``, whose only global is ``os.getpid`` (harmless, but CRITICAL by policy)."""

    @classmethod
    def load(cls):
        with open(HERE / "payload.pkl", "rb") as f:
            pickle.load(f)
        return cls(0.5)
