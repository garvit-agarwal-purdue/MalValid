# M1 — Performance & calibration (`performance`)

**Question:** at the operating threshold you declared, how many benign files would your detector
flag, and how much malware does it catch, on canonical held-out data it was not trained on?

This is the only **hard** gate by default. A detector whose false-positive rate is too high cannot
go to production however well it does elsewhere: it floods analysts with alerts and quarantines
legitimate software. A failed M1 check means the verdict is `blocked`.

| | |
|---|---|
| Requires | `feature_space` (a canonical corpus in the model's feature version) |
| Enhanced by | `training_hashes`: your training samples are excluded from the eval set |
| Default gate | `hard` |
| Access used | black-box `predict_proba` via `ctx.score` in batches of `runtime.chunk_rows`; `predict` is spot-checked on up to 1000 rows |
| Source | `src/malvalid/modules/performance.py` |

## Parameters

| key | default | meaning |
|---|---|---|
| `max_fpr` | `0.01` | gate: FPR at the operating threshold must be ≤ this. Must be in (0, 1). |
| `min_detection` | `0.95` | gate: detection rate (TPR) at the operating threshold must be ≥ this |
| `max_samples_per_class` | `null` | optional uniform subsample cap per class. `null` scores every eval row. Any subsampling is recorded in `details.eval_set` and in a note. |
| `calibration_bins` | `15` | number of equal-width bins on [0, 1] for ECE and the reliability diagram |
| `include_challenge` | `true` | also report the detection rate on the corpus's `challenge` role (informational, never gated) |
| `threshold_bootstrap_resamples` | `1000` | bootstrap replicates for the threshold-uncertainty intervals when MalValid auto-calibrated the threshold (`--calibrate-fpr`); `0` turns the bootstrap off. Ignored for a declared threshold. |

## Method

1. **Eval set.** Uses the corpus's `eval` role (for EMBER2018 this is the 200k-row test split, for
   EMBER2024 the PE test split), benign (label 0) and malicious (label 1) separately. Rows whose
   sha256 is in your training manifest are removed, and the count is reported in
   `details.excluded_training_members` and in a note. Scoring training members would inflate
   every number. Without a training manifest, a note says the numbers may be optimistic.
   If the corpus contains the same file (sha256) more than once, only its first row is kept, so
   each file counts once. The number dropped is in `details.eval_set.duplicates_removed`, with a
   note. Counting a file twice would bias the metrics toward repeated files and make the Wilson
   intervals too narrow.
2. **Scoring.** Each class is scored once through `ctx.score` in bounded batches. Only the batch
   being scored is read from the memory-mapped corpus. `ctx.check_deadline()` runs between batches.
   A batch containing a NaN or infinite score stops the module with an error naming the corpus
   row. The sandbox rejects such scores before they get here, so this is a second line of defence.
3. **Operating point.** A sample counts as malicious iff `predict_proba(x) >= operating_threshold`.
   This gives `fpr = FP / n_benign` and `detection_rate = TP / n_malicious`, each with a 95%
   **Wilson score interval** (`fpr_ci95`, `detection_rate_ci95`). If a point estimate passes but
   its interval crosses the policy limit, a note says the eval set is too small to be sure.
   A spot check on up to 1000 rows compares `predict()` with `predict_proba() >= threshold` and
   adds a note if they disagree.
4. **Threshold-free metrics.** These need both classes. The full ROC comes from scikit-learn
   (`roc_curve(drop_intermediate=False)`). `auroc` is the trapezoid area and `auprc` is average
   precision. `tpr_at_fpr_0.001` / `tpr_at_fpr_0.01` are the highest TPR over thresholds with
   FPR ≤ the target. `threshold_at_max_fpr` is the lowest threshold whose benign FPR is
   ≤ `max_fpr`, i.e. the threshold that would meet the policy on this data. It is `null` when no
   finite threshold does, for example when more than `max_fpr` of benign samples share the top
   score. When the FPR check fails, the finding says what detection rate that threshold would give.
5. **Calibration.** `brier` = mean((score − label)²). `ece` = Σ_b (n_b / N) · |mean(label in b) −
   mean(score in b)| over `calibration_bins` equal-width bins, with the same binning as sklearn's
   `calibration_curve`. The per-bin table is in `details.calibration`. Both are measured at the eval
   set's class balance, and a note says so.
6. **Challenge set.** When `include_challenge` is set and the corpus's `challenge` role is not
   empty (EMBER2024 has one; EMBER2018 does not), malicious challenge rows minus training members
   are scored at the operating threshold. The result is reported as `challenge_detection_rate`
   with a Wilson interval in `details.challenge`. It is never gated.

7. **Threshold uncertainty (auto-calibrated thresholds only).** With `--calibrate-fpr` the
   threshold is fitted on the held-out benign calibration slice (about 10k rows on EMBER2018 and
   24k-27k on EMBER2024, so about 50 and 120-130 exceedances at 0.5% FPR) and is itself a random quantity. The Wilson intervals ignore
   that. M1 therefore runs a seeded percentile bootstrap (`threshold_bootstrap_resamples`
   replicates, drawn from the module's `ctx.rng` after every other draw): each replicate re-draws
   the calibration scores with replacement, re-fits the threshold with the same rule
   (`malvalid.submission.calibrate_threshold`), then re-draws the benign and the malicious eval
   rows with replacement and measures FPR and detection rate at that threshold. The eval resample
   is done exactly without materialising it: at a fixed threshold the number of rows of an
   `n`-row resample scoring at or above it is `Binomial(n, q)`, where `q` is the fraction of the
   original rows that do. The runner hands the calibration scores to M1 in memory
   (`ctx.extras["threshold_calibration"]`); they are never written to `report.json`. This adds
   well under a second on EMBER2018. Results are `threshold_ci95`, `fpr_ci95_with_threshold`,
   `detection_rate_ci95_with_threshold` and `details.threshold_bootstrap` (replicates, calibration
   rows, the intervals, and `pass_fraction` — the share of replicates in which `fpr <= max_fpr`,
   `detection_rate >= min_detection`, and both hold). When a gate outcome flips inside the
   interval (pass fraction strictly between 2.5% and 97.5%), it is listed in
   `details.threshold_bootstrap.decided_within_noise` and a note says so, e.g. "Detection-rate
   gate decided within threshold noise: it fails on the point estimate … but passes in 8.3% of
   1000 bootstrap replicates …". This is informational: the gate checks, scores and verdict
   still use the point estimates. For a declared threshold (`--threshold` or an adapter's
   `operating_threshold`) there is no calibration noise, and `details.threshold_bootstrap` is
   `{"applicable": false, ...}`.
   The bootstrap re-draws exactly the rows the threshold was fitted on, so under either calibration
   period policy it covers sampling noise only. Drift of the benign data between the calibration
   rows and the scored rows is not noise: under the default `earliest` policy (a held threshold fit
   on the earliest benign rows) it shows up directly in the measured FPR; under `uniform` it is
   hidden, because the calibration rows span the scored period. `details.threshold_calibration_policy`
   records the policy, the calibration period and the share of scored benign rows dated after it,
   and a note states which kind of threshold was used. See
   [`model_submission.md`](model_submission.md#threshold-and-calibration).

## Checks and score anchors

`t` is the configured threshold. Anchors feed `scoring.metric_score` (floor → 0, threshold → 0.75,
ideal → 1). If a formula degenerates at an extreme threshold (e.g. `min_detection: 1.0`), the
anchors are nudged just enough to stay ordered.

| check | condition | ideal | floor | scale |
|---|---|---|---|---|
| `fpr` | `fpr <= max_fpr` | t / 10 | min(0.5, 5t) | log |
| `detection_rate` | `detection_rate >= min_detection` | t + 0.9 (1 − t) | max(0, t − 0.15) | linear |

If a class has no usable eval rows, for example because every eval row is in the training
manifest, its check value is `null`. The hard gate then counts as **unevaluated**, which blocks
the verdict; it never passes by default.

## Metrics

`detection_rate, fpr, n_benign, n_malicious, threshold, auroc, auprc, brier, ece,
tpr_at_fpr_0.001, tpr_at_fpr_0.01, threshold_at_max_fpr, detection_rate_ci95, fpr_ci95`, plus
`challenge_detection_rate` when the challenge role is not empty, and `threshold_ci95`,
`fpr_ci95_with_threshold`, `detection_rate_ci95_with_threshold` when the threshold was
auto-calibrated.

`details` has `eval_set` (splits, available/used counts, subsampling, duplicates removed), `excluded_training_members`,
`confusion`, `operating_points` (declared threshold, threshold at `max_fpr`, at FPR 0.1% and
1%), `at_threshold_at_max_fpr`, `calibration` (per-bin table), `predict_consistency` and
`challenge` and `threshold_bootstrap`.

## Charts and tables

* `roc`: ROC on a log FPR axis with the operating point and `max_fpr` / `min_detection`
  reference lines. An FPR of 0 is drawn at the left edge of the axis.
* `pr`: precision against recall, with the operating point.
* `calibration`: reliability diagram (mean predicted score against observed fraction malicious
  per non-empty bin), with the diagonal.
* `score_hist`: per-class score distributions (fraction of class per bin) with the threshold line.
* Table `operating_points`.

## Scale

On the EMBER2018 canonical corpus (200k eval rows × 2381 features), M1 scores every row once, plus
up to 1000 rows through `predict()`. Memory is bounded by `runtime.chunk_rows` × dim floats per
batch. Use `max_samples_per_class` only if scoring is too slow for your model. It is recorded,
and it widens the Wilson intervals.

## Known limitations

* The numbers describe the canonical corpus's time period and class balance. Precision and
  calibration at a real deployment's much lower malware prevalence will be worse. A note says this.
* The eval split must not overlap your training data. Declare `training_hashes_path` so MalValid
  can check that.
