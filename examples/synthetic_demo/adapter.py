"""malvalid adapter for the demo LightGBM detector trained by train.py on SYNTHETIC data.

    python train.py
    malvalid run --adapter adapter.py --config gate.yaml --out runs/synthetic-demo

Synthetic corpora are for demos and CI only: the verdict says nothing about real-world readiness.
"""

from pathlib import Path

from malvalid.adapter import BaseDetector

_HERE = Path(__file__).resolve().parent


def _threshold() -> float:
    p = _HERE / "threshold.txt"
    return float(p.read_text().strip()) if p.exists() else 0.5


class SyntheticDemoLightGBM(BaseDetector):
    feature_version = "ember_v2"
    model_kind = "lightgbm"
    operating_threshold = _threshold()  # chosen by train.py for 1% FPR on the `holdout` split
    model_path = "model.txt"  # LightGBM text format: safe, non-pickle
    training_hashes_path = "train_sha256.txt"
    training_cutoff = "2017-12-28"  # newest synthetic train row (train.py prints this value)
