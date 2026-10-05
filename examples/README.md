# MalValid examples

Four ready-to-run detector submissions. Each one is a model file plus a few values (threshold, training
cutoff, training hashes) that you pass to `malvalid run --model`; each also ships an equivalent
`adapter.py`, the advanced route for custom pipelines. From nothing to a rendered `report.html` takes one page of
steps, given in each example's README. Run the commands in each example's README from the
repository root: they `cd` into the example directory themselves.

**Start with `synthetic_demo`.** It needs no downloads and finishes in under a minute. The three
EMBER examples need real feature data (multi-GB downloads) and give results that mean something.

| Example | Model | Feature schema (`feature_version`) | Corpus needed | What it demonstrates | Time to run |
|---|---|---|---|---|---|
| [`synthetic_demo`](synthetic_demo/README.md) | LightGBM (200 trees), trained by `train.py` | `ember_v2` (2381) | `synthetic_v2`, generated on first use (no download) | The whole workflow, no downloads: train, write the manifest, validate, run all default modules, read the report. **Synthetic data: demos and CI only.** | train 4 s, validate 5 s, full run 10 s (measured) |
| [`lightgbm_ember2018`](lightgbm_ember2018/README.md) | The published EMBER2018 LightGBM model (`ember_model_2018.txt`, 1000 trees) | `ember_v2` (2381) | `ember_v2_2018` (1.7 GB download, ~9.5 GB corpus on disk) | A published model with its full training manifest, so all modules can run, including M4 membership inference and M2 drift on the months after training. | corpus build ~1 min with 10 workers; M1-only run ~1 min; full run ~10 min (M7 exact TreeSHAP on the 1000-tree model takes ~8 min) |
| [`xgboost_ember2018`](xgboost_ember2018/README.md) | XGBoost (600 rounds), trained by `train.py` | `ember_v2` (2381) | `ember_v2_2018` | Bring-your-own-model flow with a second framework (XGBoost), and how to choose an operating threshold on validation data instead of on the gate's eval set. | training ~5 min on 8 threads; M1-only run ~15 s |
| [`lightgbm_ember2024`](lightgbm_ember2024/README.md) | The published EMBER2024 PE benchmark LightGBM model (`EMBER2024_PE.model`, 500 trees; you download it, it is not bundled) | `ember_v3` (2568) | `ember_v3_2024` (3.8 GB download, ~5 GB corpus) | A model on the newer feature schema; an operating threshold that is not tuned for 1% FPR (M1 is expected to fail the hard gate); a submission without a training manifest (M4 skips, and skipped is not a pass). | corpus build ~1.5 min with 10 workers; M1-only run ~35 s |

Timings are wall-clock on a cluster node (AMD EPYC 9654, a 28-core allocation shared with other jobs;
`nproc` reports more cores than the allocation), each run using at most 8 threads. Corpus builds are
one-off.

## The steps every example follows

```bash
malvalid corpus build <corpus>  --source DIR   # once: canonical evaluation corpus (skip for synthetic_*)
malvalid inspect-model MODEL_FILE              # what malvalid detects: kind, features, feature version, corpus
malvalid run --model MODEL_FILE --threshold T [--training-cutoff D] [--training-hashes FILE] --out runs/x
```

For example, `lightgbm_ember2018`: `malvalid run --model ember_model_2018.txt --threshold 0.8336
--training-cutoff 2018-10 --training-hashes train_sha256.txt`. Without `--corpus`, the corpus follows the
detected feature version. Leaving out `--training-cutoff` skips M2 and leaving out `--training-hashes`
skips M4; either lowers coverage and the score. Without `--threshold` (or `--calibrate-fpr F`) MalValid
calibrates a threshold at 0.5% FPR on a held-out slice that is excluded from scoring; that is not your
production operating point.

The adapter route is equivalent and stays available for custom pipelines:

```bash
malvalid validate-adapter --adapter adapter.py # does the adapter meet the submission contract?
malvalid run --adapter adapter.py --out runs/x # verdict + score + report.html
```

`malvalid run` exits 0 when the verdict clears `--fail-on`, 1 when it is BLOCKED (or at or below
`--fail-on`), and 2 on a module error or a usage/adapter error. An example that ends BLOCKED is
working as intended: the gate is telling you something about the detector.

## What an adapter declares

Each declaration has a `--model` counterpart: `feature_version` is `--feature-version` (detected from the file), `model_kind` is `--model-kind` (detected), `operating_threshold` is `--threshold`, `model_path` is `--model`, `training_hashes_path` is `--training-hashes` and `training_cutoff` is `--training-cutoff`.

| Declaration | Meaning |
|---|---|
| `feature_version` | Feature space the model scores: `ember_v2` (2381 features) or `ember_v3` (2568). Selects the schema and which corpora are compatible. |
| `model_kind` | Loader plugin: `lightgbm`, `xgboost`, `sklearn_gbdt` or `onnx`. |
| `operating_threshold` | The score cut-off you would ship. M1 measures FPR and detection rate at exactly this threshold. |
| `model_path` | The artifact. M0 scans it before anything is deserialized. Relative paths resolve against the adapter's directory. |
| `training_hashes_path` | One sha256 per line for every training sample. Enables M4 membership inference and removes training samples from evaluation. `None` makes M4 skip. |
| `training_cutoff` | Newest training-sample date (`YYYY-MM-DD`, `YYYY-MM` or `YYYY`). M2 evaluates only the windows after it. `None` makes M2 skip. |

Modules that skip never count as passes: they lower the report's `coverage`, and a READY verdict
needs enough coverage.

## Two rules the examples follow

1. **Choose your threshold on your own validation data.** Never tune it on the gate's evaluation
   set: that inflates the M1 result. `xgboost_ember2018/train.py` and `synthetic_demo/train.py` show
   how (a held-out slice of the training period).
2. **Synthetic corpora prove nothing.** `synthetic_v2` and `synthetic_v3` exist for demos and CI. A
   verdict on them says nothing about real-world readiness.

No example contains malware or executables. The corpora hold feature vectors, hashes, labels and
dates only.
