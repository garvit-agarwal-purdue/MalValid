"""malvalid adapter for the published EMBER2024 PE benchmark model (EMBER2024_PE.model).

The model file is NOT bundled with malvalid. Download it (see README.md) into this directory:

    hf download joyce8/EMBER2024-benchmark-models EMBER2024_PE.model --local-dir .
    malvalid run --adapter adapter.py --out runs/ember2024

It is a LightGBM booster in LightGBM's text format (a safe, non-pickle format) that scores EMBER
feature version 3 vectors (2568 features), the feature space of the ``ember_v3_2024`` corpus.
"""

from malvalid.adapter import BaseDetector


class Ember2024LightGBM(BaseDetector):
    feature_version = "ember_v3"
    model_kind = "lightgbm"
    # 0.5 is the benchmark's default cut-off, NOT a threshold tuned for 1% FPR. On the canonical
    # ember_v3_2024 test split it gives FPR ~1.3% and detection ~97.5%, so M1's 1% FPR hard gate is
    # expected to fail. For your own deployment, choose a threshold on your own held-out data,
    # never on the gate's eval set (see README.md).
    operating_threshold = 0.5
    model_path = "EMBER2024_PE.model"
    # The EMBER2024 train split (2023-09-24 .. 2024-09-21) is not in the canonical corpus, so no
    # training manifest exists and M4 (membership inference) skips. A skip is not a pass.
    training_hashes_path = None
    # Newest first-seen date of the EMBER2024 training period; M2 scores only windows after it.
    training_cutoff = "2024-09-21"
