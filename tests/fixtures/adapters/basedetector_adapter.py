"""Test adapter: the smallest possible submission, built on malvalid.adapter.BaseDetector."""

from malvalid.adapter import BaseDetector


class MinimalDetector(BaseDetector):
    feature_version = "toy_v1"
    model_kind = "lightgbm"
    operating_threshold = 0.5
    model_path = "model.txt"
