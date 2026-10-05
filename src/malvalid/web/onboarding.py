"""First-run help for the web UI: an adapter skeleton to copy and the bundled synthetic demo.

Nothing here imports or executes an adapter; the demo is submitted like any path-mode run (the
``malvalid run`` subprocess loads it in the sandbox).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: A minimal adapter that satisfies the submission contract (shown on the New run page and the empty
#: dashboard). It mirrors ``examples/*/adapter.py``.
ADAPTER_SKELETON = '''\
"""malvalid adapter for my detector (edit the values below)."""

from malvalid.adapter import BaseDetector


class MyDetector(BaseDetector):
    # Feature space the model was trained on: "ember_v2" (EMBER 2018) or "ember_v3" (EMBER2024).
    feature_version = "ember_v2"
    # Loader for model_path: "lightgbm" (.txt), "xgboost" (.json/.ubj), "onnx" or "sklearn".
    model_kind = "lightgbm"
    # The score threshold you would ship: predict_proba(x) >= threshold means malicious.
    operating_threshold = 0.5
    # Paths are relative to this file. Upload the model under this exact name.
    model_path = "model.txt"
    # Optional: one sha256 per line of your training samples (unlocks M4 membership inference).
    training_hashes_path = None  # e.g. "train_sha256.txt"
    # Optional: newest training sample date (unlocks M2 temporal drift).
    training_cutoff = None  # e.g. "2017-12-31"

    # BaseDetector.load() loads model_path with the model_kind loader, and predict_proba(X)
    # returns an (n,) malicious-class score in [0, 1]. Override either if your model needs it:
    #
    # @classmethod
    # def load(cls):
    #     ...
    #     return cls(native_model)
    #
    # def predict_proba(self, X):
    #     ...
'''


def demo_dir() -> Path | None:
    """``examples/synthetic_demo`` of a source checkout, when present (not part of an installed wheel)."""
    import malvalid

    root = Path(malvalid.__file__).resolve().parents[2]
    d = root / "examples" / "synthetic_demo"
    if all((d / n).is_file() for n in ("adapter.py", "gate.yaml", "model.txt")):
        return d
    return None


def demo_submission() -> dict[str, Any] | None:
    """Path-mode form fields that run the bundled synthetic demo, or None when it is not available."""
    d = demo_dir()
    if d is None:
        return None
    return {"adapter_path": str(d / "adapter.py"), "config_path": str(d / "gate.yaml"),
            "title": "Synthetic demo (pipeline check, not real-world evidence)"}


__all__ = ["ADAPTER_SKELETON", "demo_dir", "demo_submission"]
