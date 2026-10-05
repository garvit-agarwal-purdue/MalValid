# lightgbm_ember2024: the published EMBER2024 PE benchmark model

The Win32/Win64/.NET PE model from the EMBER2024 benchmark (`EMBER2024_PE.model`, LightGBM, 500 trees,
2568 features), scored on the canonical `ember_v3_2024` corpus. It shows a model on the newer
`ember_v3` feature schema, an operating threshold that was not tuned for 1% FPR, and a submission
without a training manifest.

**The model file is not bundled with malvalid.** You download it from Hugging Face. Check the license
terms on the model and dataset cards before use.

## Steps

You need about 15 GB of free disk (3.8 GB of downloads, ~5 GB corpus) and 8+ cores. Run the commands
from the repository root.

```bash
pip install -U "huggingface_hub[cli]"        # provides the `hf` command

# 1. Download the model into this example's directory
cd examples/lightgbm_ember2024
hf download joyce8/EMBER2024-benchmark-models EMBER2024_PE.model --local-dir .

# 2. Download the EMBER2024 feature files the corpus is built from (feature vectors only, no binaries)
hf download joyce8/EMBER2024 Win32_test.zip Win64_test.zip challenge.zip --repo-type dataset --local-dir ~/ember2024

# 3. Build the canonical corpus (about 1.5 minutes with 10 workers; zip members are streamed, not extracted)
export MALVALID_CORPUS_DIR=$HOME/malvalid-corpora
malvalid corpus build ember_v3_2024 --source ~/ember2024 --workers 8

# 4. Run the gate on the model file alone. 2568 features -> ember_v3 -> ember_v3_2024 is picked for you
malvalid inspect-model EMBER2024_PE.model        # LightGBM despite the .model extension
malvalid run --model EMBER2024_PE.model --threshold 0.5 --training-cutoff 2024-09-21 --out runs/ember2024
```

The same submission through the example's adapter. An adapter run uses the default policy's `ember_v2`
corpus unless you pass `--corpus`:

```bash
malvalid validate-adapter --adapter adapter.py
malvalid run --adapter adapter.py --corpus ember_v3_2024 --out runs/ember2024
```

Then open `runs/ember2024/report.html`. With an adapter, omitting `--corpus ember_v3_2024` (or a `gate.yaml`
with `corpus: ember_v3_2024`) blocks the run with "corpus in feature space ember_v2 but the model
declares ember_v3": the gate refuses to score a model on vectors from the wrong feature space. With
`--model` the corpus follows the detected feature version, so the mismatch cannot happen by default.

`validate-adapter` works before the corpus is built (it probes with schema-shaped random vectors and a
benign PE file shipped in setuptools). It reports one warning, that no training manifest is declared;
see below. Smoke test with one module: `malvalid run --model EMBER2024_PE.model --threshold 0.5
--only M1 --out runs/m1` (about 35 s).

## What the submission declares

The `--model` command passes the same values that `adapter.py` declares (no `--training-hashes`: see below):

```python
class Ember2024LightGBM(BaseDetector):
    feature_version = "ember_v3"
    model_kind = "lightgbm"
    operating_threshold = 0.5
    model_path = "EMBER2024_PE.model"
    training_hashes_path = None
    training_cutoff = "2024-09-21"
```

* `feature_version = "ember_v3"`: EMBER feature version 3, 2568 floats. MalValid's `ember_v3` schema
  implements the same vectorizer as the EMBER2024 authors' `thrember` package (Apache-2.0), so column
  order matches the model.
* `model_kind = "lightgbm"`: the `.model` file is LightGBM's text format (safe, no pickle), so the
  extension differs from `.txt` but the LightGBM loader reads it. M0 scans it before loading.
* `operating_threshold = 0.5` (`--threshold 0.5`): the benchmark's default cut-off. See the next section.
* `model_path`: relative to the adapter's directory. With `--local-dir .` above the file lands beside
  `adapter.py`. To keep the model elsewhere, pass `--model /path/EMBER2024_PE.model` to `malvalid run`.
* `training_hashes_path = None` (no `--training-hashes`): no training manifest, see below.
* `training_cutoff = "2024-09-21"`: the end of the EMBER2024 training period
  (2023-09-24 to 2024-09-21). M2 evaluates the windows after it: the canonical test rows run from
  2024-09-22 to 2024-12-14, so there are three monthly windows (2024-09-22 to 10-21, 10-22 to 11-21,
  11-22 to 12-21), the last one partial.

## M1 is expected to fail the 1% FPR hard gate at 0.5

Measured with an M1-only run on the canonical corpus (239,987 benign, 240,000 malicious test rows):

| At threshold | FPR | Detection |
|---|---|---|
| 0.5 (declared) | 1.28% | 97.53% |
| 0.58 | 1.00% | 97.07% |

The default policy requires FPR at most 1%, so M1 fails, the run is BLOCKED and the score is capped at
49 (M1 alone scores 63.5). That is the correct outcome for a model declared at 0.5, not a defect in
the run. The threshold is a deployment decision, and this benchmark cut-off was never meant to meet
a 1% FPR budget.

The M1 finding states the threshold that would meet the limit **on the gate's evaluation data**
("a threshold of 0.58 would meet max_fpr"). Treat that as a diagnostic. **Do not copy it into your
adapter or `--threshold` and re-run.** A threshold read off the gate's own eval set makes the M1 result circular: it
passes by construction and tells you nothing about how the detector will behave on data you have not
seen. For a real deployment, choose the threshold on your own held-out data (a time-ordered validation
slice of your own traffic or training period, the way `../xgboost_ember2018/train.py` does), set
`operating_threshold` to that value, and let the gate check it on data the choice never touched.

## No training manifest: M4 skips, and skipped is not a pass

The EMBER2024 *train* split is not part of the canonical `ember_v3_2024` corpus (which holds only the
test and challenge splits), so there is no list of training hashes to declare. Consequences:

* **M4 (membership inference) is skipped.** The report says so. A skipped module never counts as a
  pass: it lowers `coverage`, and the verdict can never be READY unless enough of the weighted battery
  ran. Members of the training set are simply not measurable here; the absence of an M4 result is not
  evidence of no leakage.

If you train your own model on EMBER2024 and want M4 to run, it needs a canonical corpus that contains
your training rows; the EMBER2024 test and challenge rows here are not training members of such a model.

## Files

`adapter.py` is the only file in the repository. `EMBER2024_PE.model` is downloaded by you (step 1) and is
ignored by git. Do not commit or redistribute it.

## References

Joyce et al., "EMBER2024: A Benchmark Dataset for Holistic Evaluation of Malware Classifiers"
(KDD 2025). Model repository `joyce8/EMBER2024-benchmark-models`, dataset `joyce8/EMBER2024`, feature
code `thrember` (Apache-2.0).
