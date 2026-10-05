# xgboost_ember2018: train your own detector, choose a threshold, submit it

A complete bring-your-own-model flow. `train.py` trains an XGBoost detector on the canonical EMBER2018
corpus and writes everything a MalValid submission needs. The model file alone is enough to submit
(`malvalid run --model model.json ...`); the 15-line adapter is the equivalent, because the XGBoost JSON
format is understood by MalValid's `xgboost` loader.

## Steps

You need about 25 GB of free disk and 8+ cores. Run the commands from the repository root.

```bash
# 1. Get the EMBER2018 feature release (feature JSON only, no PE files)
curl -LO https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
sha256sum ember_dataset_2018_2.tar.bz2      # b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812
tar -xjf ember_dataset_2018_2.tar.bz2       # creates ember2018/

# 2. Build the canonical corpus (about 1 minute with 8-10 workers)
export MALVALID_CORPUS_DIR=$HOME/malvalid-corpora
malvalid corpus build ember_v2_2018 --source ember2018/ --workers 8

# 3. Train (about 5 minutes on 8 threads). Writes model.json, threshold.txt, train_sha256.txt
cd examples/xgboost_ember2018
python train.py --threads 8

# 4. Run the gate on the model file alone (threshold.txt holds the value train.py chose, 0.865389)
malvalid inspect-model model.json                       # XGBoost, 2381 features -> ember_v2 -> ember_v2_2018
malvalid run --model model.json --threshold "$(cat threshold.txt)" \
    --training-cutoff 2018-10 --training-hashes train_sha256.txt --out runs/xgb-ember2018
```

The same submission through the example's adapter (equivalent):

```bash
malvalid validate-adapter --adapter adapter.py
malvalid run --adapter adapter.py --out runs/xgb-ember2018
```

Then open `runs/xgb-ember2018/report.html`. The default policy already uses `ember_v2_2018`, and with
`--model` the corpus follows the detected feature version. For a quick smoke test:
`malvalid run --model model.json --threshold "$(cat threshold.txt)" --only M1 --out runs/m1` (about 15 s).
Skipping `--training-cutoff` skips M2 and skipping `--training-hashes` skips M4, which lowers coverage and the score.

## What train.py does

* Loads the labeled `train` rows of the corpus (600,000 malicious + benign; the 200,000 unlabeled
  rows are ignored).
* **Holds out the last training month, Oct 2018 (86,414 rows), for validation and fits on all
  earlier labeled train rows (513,586 rows: Jan-Sep 2018 plus the benign rows first seen before
  2018).** Validation AUC 0.9923.
* **Picks the operating threshold for 1% FPR on that validation month** (0.865389; 89.7% detection
  there). It never looks at the corpus's `test` split, which is what M1 scores. Tuning a threshold on
  the gate's own evaluation data would inflate the M1 result, so keep this structure for your own
  models: threshold from validation data, gate on data the model has never seen.
* Writes `model.json` (XGBoost JSON, a safe non-pickle format), `threshold.txt`, and
  `train_sha256.txt` (all 600,000 rows used for fitting or validation: the model's training
  members).
* Settings: `--rounds 600`, depth 10, eta 0.1, subsample 0.8, colsample 0.5, seed 0. The log is
  printed to the terminal; redirect it to `train.log` if you want to keep it (git ignores it).

## What the submission declares

The `--model` command passes the same values that `adapter.py` declares:

```python
class Ember2018XGBoost(BaseDetector):
    feature_version = "ember_v2"
    model_kind = "xgboost"
    operating_threshold = _threshold()      # reads threshold.txt (0.865389)
    model_path = "model.json"
    training_hashes_path = "train_sha256.txt"
    training_cutoff = "2018-10"
```

* `feature_version = "ember_v2"`: the model scores EMBER feature version 2 vectors (2381 floats).
* `model_kind = "xgboost"`: load with MalValid's XGBoost loader (`.json`/`.ubj`; pickles are refused
  unless you pass `--allow-pickle`). `BaseDetector` provides `load()`, `predict_proba()` and `predict()`.
* `operating_threshold`: the shipped cut-off. The adapter reads `threshold.txt` written by `train.py`;
  hard-coding the number works equally well. M1 measures FPR and detection at exactly this value.
* `model_path = "model.json"`: scanned by M0 before it is loaded.
* `training_hashes_path`: the training members for M4 membership inference. It lists validation-month
  rows too, because the model was tuned on them.
* `training_cutoff = "2018-10"`: newest training data is October 2018. M2 scores the windows after it
  (the corpus's test split is November and December 2018, so two monthly windows).

## What to expect

Measured for an M1-only run: FPR 0.89% (under the 1% limit) but detection 92.5% at the threshold
(minimum 95.0%), so the M1 hard gate **fails** and the verdict is BLOCKED (score capped at 49).
That is a real finding and a good illustration of the gate: this model is weaker than the published
LightGBM one, and a threshold tuned to 1% FPR on October 2018 does not buy the 95% detection the
default policy demands. Options are a stronger model (more rounds or trees, more data), or a
deliberately different policy in your own `gate.yaml`. Compare it with
[`lightgbm_ember2018`](../lightgbm_ember2018/README.md), which passes M1.

## Files

`train.py`, `adapter.py`; generated by `train.py`: `model.json` (13 MB), `threshold.txt`,
`train_sha256.txt`. None of these generated files is committed.
