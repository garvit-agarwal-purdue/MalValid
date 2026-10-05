"""malvalid adapter for an XGBoost detector trained on EMBER2018 by train.py.

    python train.py
    malvalid run --adapter adapter.py --out runs/xgb-ember2018
"""

from pathlib import Path

from malvalid.adapter import BaseDetector

_HERE = Path(__file__).resolve().parent


def _threshold() -> float:
    p = _HERE / "threshold.txt"
    return float(p.read_text().strip()) if p.exists() else 0.5


class Ember2018XGBoost(BaseDetector):
    feature_version = "ember_v2"
    model_kind = "xgboost"
    operating_threshold = _threshold()  # chosen by train.py for 1% FPR on held-out training data
    model_path = "model.json"  # XGBoost JSON: a safe, non-pickle format
    training_hashes_path = "train_sha256.txt"
    training_cutoff = "2018-10"
