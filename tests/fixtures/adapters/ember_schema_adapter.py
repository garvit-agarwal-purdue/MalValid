"""Test adapter: an ember_v2-space LightGBM model with no featurize() of its own, so malvalid falls
back to the ember_v2 schema's extractor inside the sandbox. ``model.txt`` is trained at test time on
random vectors (see ``builders.make_ember_v2_model``)."""

from malvalid.adapter import BaseDetector


class EmberSchemaDetector(BaseDetector):
    feature_version = "ember_v2"
    model_kind = "lightgbm"
    operating_threshold = 0.5
    model_path = "model.txt"
