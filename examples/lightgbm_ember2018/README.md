# lightgbm_ember2018: the published EMBER2018 LightGBM model

The reference model from the EMBER2018 release (`ember_model_2018.txt`, LightGBM, 1000 trees) on the
canonical `ember_v2_2018` corpus. It is the smallest realistic submission: a booster in LightGBM's
text format (safe, no pickle), a training manifest, and a declared operating point.

## Steps

You need about 25 GB of free disk (1.7 GB download, ~11 GB extracted, ~10 GB corpus) and 8+ cores.
The extracted feature files can be deleted after step 3. Run the commands from the repository root.

```bash
# 1. Get the EMBER2018 feature release (feature JSON only, no PE files)
curl -LO https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
sha256sum ember_dataset_2018_2.tar.bz2      # b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812
tar -xjf ember_dataset_2018_2.tar.bz2       # creates ember2018/ with train_features_*.jsonl, test_features.jsonl
                                            # and ember_model_2018.txt

# 2. Build the canonical corpus (about 1 minute with 8-10 workers)
export MALVALID_CORPUS_DIR=$HOME/malvalid-corpora     # anywhere with ~10 GB free
malvalid corpus build ember_v2_2018 --source ember2018/ --workers 8
malvalid corpus verify ember_v2_2018                   # optional: re-hash against the manifest

# 3. Link the model and write the training manifest
cd examples/lightgbm_ember2018
ln -sf "$(realpath ../../ember2018/ember_model_2018.txt)" ember_model_2018.txt   # adjust to where you extracted
python make_manifest.py                                # writes train_sha256.txt (600,000 hashes)

# 4. Run the gate on the model file alone (the corpus follows the detected feature version)
malvalid inspect-model ember_model_2018.txt            # LightGBM, 2381 features -> ember_v2 -> ember_v2_2018
malvalid run --model ember_model_2018.txt --threshold 0.8336 \
    --training-cutoff 2018-10 --training-hashes train_sha256.txt --out runs/ember2018
```

The same submission through the example's adapter (equivalent, and the route for custom pipelines):

```bash
malvalid validate-adapter --adapter adapter.py
malvalid run --adapter adapter.py --out runs/ember2018
```

Then open `runs/ember2018/report.html` in a browser (the terminal prints the path). The default
gate policy uses `corpus: ember_v2_2018`, so no `--config` is needed. The model file is not stored in the repository (it is 127 MB):
step 3 links the copy you extracted in step 1.

For a quick smoke test of the model and corpus, run one module: `malvalid run --model ember_model_2018.txt
--threshold 0.8336 --only M1 --out runs/m1` (M0 always runs too; about 1 minute). Leaving out
`--training-cutoff` skips M2, and leaving out `--training-hashes` skips M4 (coverage and the score drop).

## What the submission declares

The `--model` command above passes the same values on the command line that `adapter.py` declares:

```python
class Ember2018LightGBM(BaseDetector):
    feature_version = "ember_v2"
    model_kind = "lightgbm"
    operating_threshold = 0.8336
    model_path = "ember_model_2018.txt"
    training_hashes_path = "train_sha256.txt"
    training_cutoff = "2018-10"
```

* `feature_version = "ember_v2"`: the model scores EMBER feature version 2 vectors (2381 floats).
  MalValid pairs it with the `ember_v2` schema and only accepts `ember_v2` corpora.
* `model_kind = "lightgbm"`: load through MalValid's LightGBM loader. The loader also gives M5 and M7
  direct access to the trees (`BaseDetector` sets `native_model`, so you write no `load()` or
  `predict_proba()` yourself).
* `operating_threshold = 0.8336`: the EMBER authors' operating point, 1% FPR on the EMBER2018 test
  set. M1 measures FPR and detection rate at exactly this value. It is a published number, so it was
  tuned on the same test split the gate evaluates on; that is fine for a reproduction, but a threshold
  for your own model must come from your own validation data (see `../xgboost_ember2018`).
* `model_path`: the artifact M0 scans before anything is deserialized. It is relative to the adapter's
  directory.
* `training_hashes_path`: the sha256 of every labeled sample the model was trained on. `make_manifest.py`
  reads them from the corpus's `train` split. M4 uses them as the members; the runner also keeps
  training samples out of evaluation sets.
* `training_cutoff = "2018-10"`: the newest training samples appeared in October 2018
  (`YYYY-MM` means the last day of the month). M2 evaluates only windows after it. The EMBER2018 test
  split covers November and December 2018, so M2 has just two monthly windows here.

## What to expect

Measured on a 28-core cluster allocation for an M1-only run (score and verdict cover only the modules that ran):

* M0 pass. M1 pass: FPR 1.00% and detection 96.5% at threshold 0.8336, matching the published
  EMBER2018 benchmark. The 1% FPR gate is met exactly because the threshold is the benign-score
  quantile on this very split. Verdict CONDITIONAL, score 75.0 (M1 scores 0.75 at the gate boundary
  and rises to 1.0 at the ideal; a READY verdict needs 80+ and more coverage).
* M4 has no time overlap to work with: the train split ends in 2018-10 and the test split starts
  in 2018-11, so M4 compares October members with November non-members and reports an upper bound.
  With the defaults it measured an advantage of 0.178, above the 0.10 warn limit. That is a
  screening result about this model, not a bug in the run.

## Files

* `adapter.py`: the submission.
* `make_manifest.py`: writes `train_sha256.txt` from the corpus (needs step 2 first).
* `ember_model_2018.txt`: the model, from the EMBER2018 tarball (you create the symlink in step 3; it is not in the repository).
* `train_sha256.txt`: generated by `make_manifest.py`.

## License note

The EMBER2018 data and model are from the EMBER project (data files MIT; the EMBER source repository
is AGPL-3.0 and MalValid does not use its code). Cite Anderson and Roth, "EMBER: An Open Dataset for
Training Static PE Malware Machine Learning Models", 2018.
