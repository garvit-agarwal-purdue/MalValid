"""malvalid adapter for the published EMBER2018 LightGBM model (ember_model_2018.txt).

This is the smallest realistic submission: a LightGBM booster saved in LightGBM's text format
(a safe, non-pickle format), scored on EMBER feature version 2 vectors.

Before running the gate, from this directory:

    ln -s /path/to/ember2018/ember_model_2018.txt .     # ships inside the EMBER2018 tarball
    python make_manifest.py                              # writes train_sha256.txt (training members)
    malvalid run --adapter adapter.py --out runs/ember2018
"""

from malvalid.adapter import BaseDetector


class Ember2018LightGBM(BaseDetector):
    feature_version = "ember_v2"
    model_kind = "lightgbm"
    # The EMBER authors' operating point: 1% FPR on the EMBER2018 test set (96.5% detection).
    operating_threshold = 0.8336
    model_path = "ember_model_2018.txt"
    # The model was trained on the labeled EMBER2018 train split, whose newest samples appeared in
    # October 2018. make_manifest.py writes their sha256s here.
    training_hashes_path = "train_sha256.txt"
    training_cutoff = "2018-10"
