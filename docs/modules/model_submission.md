# Model-file submissions (`malvalid run --model`)

A researcher can submit **just a model file**. MalValid inspects it as data, writes a small data-only
spec, and runs the usual battery. The Python adapter remains the advanced path (custom `featurize`,
preprocessing, several files, other feature spaces). Code: `src/malvalid/inspect_model.py`,
`src/malvalid/submission.py`, `src/malvalid/adapters/spec.py`.

## Inference rules (`malvalid inspect-model FILE [--json]`)

The file is read as data in the calling process. Nothing is executed or deserialized with a model library.

| Detected | Rule |
|---|---|
| Model kind | From content, not the extension. LightGBM text model: `tree` header and `version=v...` (so `EMBER2024_PE.model` is LightGBM). XGBoost: a JSON or UBJSON `learner` object. ONNX: protobuf, parsed with the `onnx` package. |
| Input features | LightGBM `max_feature_idx + 1` (or `feature_names`); XGBoost `num_feature`; ONNX graph input shape. |
| Feature version | 2381 features is `ember_v2`, 2568 is `ember_v3`. Any other count is a clear error: use an adapter with your own schema. |
| Default corpus | `ember_v2` is `ember_v2_2018`, `ember_v3` is `ember_v3_2024`. Used when you pass neither `--corpus` nor a `--config` naming one. |

Pickle and joblib files are **never unpickled** outside the sandbox. A static opcode scan may suggest a
kind, but you must pass `--model-kind` and `--feature-version` yourself and opt in with `--allow-pickle`.
Explicit `--model-kind` / `--feature-version` override detection (the override is noted in the report).

## The spec

`<out>/submission/malvalid_model.json`, schema `malvalid-model-spec/1`, next to symlinks to the model
and the optional hash list:

```json
{
  "schema": "malvalid-model-spec/1",
  "model_kind": "lightgbm",
  "feature_version": "ember_v2",
  "operating_threshold": 0.8336,
  "threshold_source": "declared",
  "calibrate_fpr": null,
  "model_file": "ember_model_2018.txt",
  "training_hashes_file": "train_sha256.txt",
  "training_cutoff": "2018-10"
}
```

`threshold_source` is `declared` (a `--threshold` in [0, 1]) or `calibrate` (`operating_threshold` is
null and `calibrate_fpr` is in (0, 0.1]). The spec is **data, never code**: no Python is generated from
user input. Every field is validated strictly (enums, a finite threshold, a date regex, plain-basename
file names, a size cap). The sandboxed worker turns a valid spec into a subclass of the packaged
`malvalid.adapters.spec.SpecDetector`, which loads the model through the normal loader plugins. The model
therefore loads only inside the sandbox, with the same protections as an adapter (bubblewrap, no network,
pickle guards, no-unpickle IPC).

## Threshold and calibration

* `--threshold T`: the cut-off you ship. M1 measures FPR and detection at exactly `T`.
* `--calibrate-fpr F` (for example 0.005): the threshold is the smallest value whose FPR on the
  calibration slice is at most `F`.
* Neither flag: MalValid calibrates at 0.5% FPR (half the default M1 1% gate, leaving room for sampling
  noise) and says so in the report warnings.

**Calibration split.** About 10% of the benign rows of the corpus's eval split (minus declared training
members) are held out. Which 10% is the **calibration period** policy, `runtime.calibration_period` in the
gate config or `--calibration-period` on the CLI:

| Policy | Rows | What the threshold has seen |
|---|---|---|
| `earliest` (default) | the earliest `ceil(n / 10)` of the `n` dated benign eval rows by timestamp, ties broken by sha256; the boundary timestamp is taken whole unless that would more than double the slice | only data from before (almost) every row it is scored on: a *held* threshold, as in deployment |
| `uniform` | rows with `int(sha256[:8], 16) % 10 == 0` | benign data from every period it is scored on: a *matched* threshold, so FPR at it is close to the target by construction |

On EMBER2024 (day-level timestamps) `earliest` takes the benign rows first seen 2024-09-22 to 2024-09-30;
every scored benign row is later. On EMBER2018 every test row is dated on the first of its month, so
`earliest` takes a sha256-ordered tenth of the November benign rows; the December rows are scored in full and
November keeps 40k scored benign rows, so M2 keeps both monthly windows. `corpus.calibration_holdout` records
`policy`, `period` (first and last calibration date), `boundary_split` and `scored_benign_after_period` (the
share of scored benign rows dated after the calibration period: 1.0 on EMBER2024, 0.56 on EMBER2018). If too
few benign rows are dated to fill the slice, `earliest` falls back to `uniform` with a warning
(`fallback_reason`). Malicious rows of the calibration period stay in the evaluation; the threshold never
sees malware. Because the calibration rows are excluded from M4's non-member pool too, `earliest` removes
the benign rows nearest in time to the training data from it. On EMBER2018 this raised M4's advantage
(already an upper bound there) from 0.155 to 0.181.

Whatever the policy, the selected rows are relabelled split `calibration`, which no role lists, so every
module (M1, M2 windows, M4 and M6 pools, challenge sets) excludes them for that whole run. The run needs
at least 200 such rows.

*Why `earliest` is the default.* A deployed threshold is fit once on past data and then meets future data.
With `uniform`, the threshold has already seen the benign distribution of every M2 window, so M1's FPR lands
on the target almost exactly and the per-window FPRs centre on it. On EMBER2024 (LightGBM, target 0.5%),
`uniform` reports an M1 FPR of 0.50%; a threshold held from the first nine days gives 0.81% on the later rows
(0.75% → 0.90% across M2's monthly windows), outside the uniform run's 95% interval with threshold noise
(0.39%–0.58%). The held number is what a deployment would see. On EMBER2018 the two policies agree within
noise (0.53% vs 0.49%), because its test rows only carry two month-level dates. See
[`docs/results/REAL_DATA_RESULTS.md`](../results/REAL_DATA_RESULTS.md) for both runs.

**Why this is sound.**

1. *Disjoint from scoring.* A calibration row is in no evaluation role, so the threshold is never fitted
   on data that is used for scoring.
2. *Deterministic.* Membership depends only on the sha256 (and, for `earliest`, the timestamp), not on a
   seed, worker count or row order, so the same model and corpus always give the same slice and threshold.
3. *Identical files never straddle.* Every row whose sha256 matches a selected row is held out too, so a
   duplicate of a calibration file cannot appear in the evaluation set.
4. *Training members are excluded* when `--training-hashes` is given, so the threshold is not fitted on
   rows the model has seen.

The report records `model.extras.threshold_source` (`declared` or `calibrated`),
`model.extras.threshold_calibration` (target, achieved FPR, row counts) and `corpus.calibration_holdout`.
A calibrated threshold is **not your production operating point**: it targets an FPR on the canonical
corpus's class balance. Pass `--threshold` for a verdict on the cut-off you ship.

**Target not achievable.** If benign calibration samples score exactly 1.0 and the allowed false-positive
count (`floor(F * n)`) is smaller than their number, no threshold <= 1.0 meets the target. MalValid then
fails the run with `target FPR X% is not achievable: N benign calibration samples score 1.0; give an explicit
--threshold or a higher --calibrate-fpr` (a one-line error on the CLI, a failed run with that cause in the
web UI) instead of reporting a threshold whose FPR is above the target. In the normal path the achieved FPR
is guaranteed to be at most the target.

## Web UI

The web form (default mode `model`) maps onto these flags: `model_file`, `model_kind`, `feature_version`,
`threshold_mode` (`calibrate` with `calibrate_fpr`, default 0.5%, or `declared` with `threshold`),
`training_cutoff`, and the training-hash list (`manifest_file` upload, or `training_hashes_path` in path
mode). **Inspect model** (`POST /runs/inspect`) runs `malvalid inspect-model --json` in a subprocess and
pre-fills kind and feature version. Pickles need the **Allow pickle-based model files** opt-in. **Advanced:
own adapter** is the adapter upload. See [`docs/web.md`](../web.md).

## Optional inputs and what skipping them costs

| Option | Unlocks | If omitted |
|---|---|---|
| `--training-cutoff YYYY-MM[-DD]` | M2 drift (windows after the cutoff) | M2 is skipped. |
| `--training-hashes FILE` | M4 membership inference; training members are kept out of evaluation | M4 is skipped; overlapping training rows stay in the evaluation set. |

A skipped test never counts as a pass: coverage and the score drop, and READY needs enough coverage.

## Limitations

* Only `ember_v2` (2381) and `ember_v3` (2568) feature counts are recognised.
* Pickle-based files need `--model-kind`, `--feature-version` and `--allow-pickle` by hand.
* ONNX feature-dimension detection needs the `onnx` package (the `onnx` extra); without it, pass
  `--feature-version`.
* A calibrated threshold removes about 10% of the benign eval rows from scoring (plus duplicates of
  them), which slightly changes M1 and M2 estimates compared with a declared threshold.
* Custom featurization, preprocessing, multiple model files and other feature spaces need `--adapter`.
