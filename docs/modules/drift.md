# M2 — Temporal drift (`drift`)

**Question:** your detector was trained on the past and will be deployed on the future. How fast
does its detection quality decay on samples that appeared *after* its training data?

Malware changes continuously: new families, new packers, new toolchains. A detector evaluated on a
random split of the same period as its training data will look far better than it performs after
deployment (Pendlebury et al., "TESSERACT: Eliminating Experimental Bias in Malware Classification
across Space and Time", USENIX Security 2019). M2 runs a TESSERACT-style time-aware evaluation on
the canonical corpus and summarizes it as **AUT(F1)**, the area under the F1-over-time curve.

| | |
|---|---|
| Requires | `feature_space` (canonical corpus in the model's feature version) **and** `training_cutoff` (declared on the adapter) |
| Enhanced by | `training_hashes`: training members are excluded, and the declared cutoff is cross-checked against them |
| Default gate | `warn` |
| Access used | black-box `predict_proba` via `ctx.score`, in batches |
| TESSERACT | installed `tesseract.temporal` (BSD-3) when importable, else the vendored subset in `src/malvalid/_vendor/tesseract_temporal.py`. Which one ran is recorded in `details.tesseract`. |
| Source | `src/malvalid/modules/drift.py` |

## Parameters

| key | default | meaning |
|---|---|---|
| `min_aut_f1` | `0.70` | gate: sample-weighted AUT(F1) must be ≥ this |
| `granularity` | `"month"` | window length: `year`, `quarter`, `month`, `week` or `day` (plurals accepted), or `auto` (see below) |
| `min_window_benign` | `100` | windows with fewer benign rows have an *unevaluable* FPR: listed, excluded from the operating-point check, never counted as within budget |
| `min_window_samples` | `200` | windows with fewer samples are dropped (and listed) |
| `max_windows` | `36` | evaluate at most this many non-empty windows after the cutoff. Later windows are listed in `details.beyond_max_windows`. |
| `max_samples_per_window` | `null` | optional uniform subsample cap per window (must be ≥ `min_window_samples`). Recorded in `details.subsampling` and a note. |
| `require_both_classes` | `true` | drop windows that have no benign samples. Windows with no malicious samples are always dropped, because F1 is undefined there. |

## Method

1. **Effective cutoff (TESSERACT C1: training strictly precedes testing).** Start from the declared
   `training_cutoff` (`YYYY-MM-DD`, `YYYY-MM` = end of month, `YYYY` = Dec 31). If a training
   manifest is declared and some of its samples in the corpus are dated *after* that cutoff, the
   declaration is inconsistent. M2 adds a note (`Inconsistent training_cutoff: …`) and uses the
   newest training-member timestamp as the **effective cutoff**, so no window overlaps training
   data. `details.temporal_constraints.c1` records the declared and effective cutoffs, how many
   members are in the corpus and how many post-date the declaration, and whether C1 holds on the
   rows actually evaluated.
2. **Candidate rows.** Labeled rows in the corpus's `temporal` role, dated strictly after the
   effective cutoff, with training-manifest members removed and repeated files (same sha256)
   kept once. The number of duplicates dropped is in `details.duplicates_removed`, with a note.
3. **Windows.** Rows are partitioned with TESSERACT's `time_aware_indexes(t, 0, 1, granularity,
   start_date=cutoff + 1 day)` into consecutive windows of one granularity unit each. With a
   month-end cutoff and `month`, the windows are calendar months. Each window keeps its period
   number `k` (1 = first period after the cutoff), and empty periods are skipped without
   renumbering. At most `max_windows` non-empty windows are considered.
4. **Selection.** A window is dropped, with a reason, if it has fewer than `min_window_samples`
   rows, no malicious rows, or (with `require_both_classes`) no benign rows. Dropped windows are
   listed in `details.dropped_windows`, shown in the `window_sizes` chart and summarized in a note.
   AUT uses the remaining windows in time order.
5. **Per window**, at the declared operating threshold (`score >= threshold` = malicious): n,
   n_mal, n_ben, precision, recall, F1, FPR, **miss rate (FNR = 1 − recall)** with Wilson 95%
   intervals for FPR and FNR (the same `wilson_interval` as M1), and **AUROC** (rank-based, ties
   averaged; `null` when a window has one class). Undefined ratios are 0 (sklearn's `zero_division=0`),
   except FPR, which is `null` when a window has no benign rows. All used windows are scored in one
   pass over sorted indices through `ctx.score`.
6. **Other temporal constraints.**
   * **C2 (malware/goodware time alignment).** In each window, the median timestamps of the
     malicious and benign samples should be at most 1 month apart (TESSERACT's `month_variance`,
     via `month_difference`). Misaligned windows are listed with a note, because there the model
     may be separating samples by age rather than by maliciousness.
   * **C3 (realistic malware ratio).** Reports the per-window malware ratio (min, median, max).
     Windows more than 0.10 from the median are flagged, because F1 depends on prevalence. A note
     also says that F1 is measured at the corpus's class balance, not at deployment prevalence.
7. **AUT(F1).** With per-window F1 values `f_1..f_N` and evaluated window sizes `n_1..n_N`:
   * unweighted (TESSERACT): `AUT = Σ_{k=1}^{N−1} (f_k + f_{k+1}) / 2 / (N − 1)`;
   * **sample-weighted** (gated):
     `AUT_w = Σ_k ((n_k f_k + n_{k+1} f_{k+1}) / (n_k + n_{k+1})) · (n_k + n_{k+1}) / Σ_j (n_j + n_{j+1})`.
     Each segment's height is the size-weighted mean of its two windows, and its weight is the
     number of samples it covers. A sparse window therefore cannot count as much as a dense one,
     which is a known AUT pitfall. With equal window sizes the two values agree.

   Example: F1 = (0.9, 0.5, 0.8) with n = (1000, 100, 1000) gives unweighted
   AUT = ((0.9+0.5)/2 + (0.5+0.8)/2)/2 = 0.675. The weighted segment heights are
   (900+50)/1100 = 0.8636 and (50+800)/1100 = 0.7727 with equal weights 1100/2200, so
   AUT_w = 0.8182. The small, poor middle window barely moves the weighted value.

## Operating-point drift (why AUT(F1) is not enough)

AUROC and AUT(F1) are threshold-averaged or F1-based summaries and can look healthy while the
*frozen* threshold you actually ship drifts. In the drift paper ("Drift at the Alert Threshold")
AUROC fell only 0.999 → 0.993 while the miss rate at a frozen 0.1%-FPR threshold rose 5.4% → 15.4% and
the realised FPR went from 0.10% to 0.50%; a 0.029 AUT gap corresponded to 18.5 pp of missed
malware. M2 therefore reports, per window, realised FPR and FNR with 95% CIs next to AUROC, and
the finding opens its second half with a headline such as:

> Operating point: AUROC 0.999→0.997 while the miss rate at the shipped threshold goes 2.2%→3.1% and the realised FPR 0.83%→2.04% (first→last window; FPR budget 1.00%).

On the real EMBER2024 run (lightgbm_ember2024, threshold 0.5) AUT(F1) is 0.98 and AUROC barely
moves (0.9985 → 0.9977) while the realised FPR climbs 1.08% → 1.51% and the miss rate 2.3% → 3.0%.

**Check `window_fpr_within_budget`.** `value` = the largest per-window Wilson 95% *lower* bound of
the realised FPR; condition `value <= max_fpr`, where `max_fpr` is read from the `performance`
(M1) section of the gate config (default 0.01). It fails only when some window's FPR is
*statistically* above the budget, so a point estimate that brushes the limit by sampling noise
(e.g. EMBER2018 December: 1.04%, CI 0.95%–1.13%) passes; such windows are named in the finding.
Windows with fewer than `min_window_benign` benign rows are unevaluable; if none is evaluable the check has
value `null` (status `warn`, never a pass).

**It is gated but ungraded.** The check has no ideal/floor anchors, so it has no score: it can turn
M2's status to `warn` (gate failed: the module is flagged and READY is blocked) but it **never changes the
0-100 axis score or headline score**, which stay AUT(F1)-only. The M1 hard gate remains the place where the
FPR budget is enforced on the eval set; this check says the budget no longer holds on *newer* data.

**Auto-calibrated thresholds.** With `--calibrate-fpr`, which benign rows the threshold was fit on
decides what the per-window FPRs mean. Under the default `runtime.calibration_period: earliest` it is a
*held* threshold, fit on the earliest ~10% of the eval benign rows and never refit, so the per-window
FPRs show how the false-positive rate moves after deployment; the windows inside the calibration period
have fewer (or, on weekly windows, no) benign rows, and a single-class window is dropped as usual. Under
`uniform` the calibration rows are hash-sampled from every window, so the per-window FPRs centre on the
target by construction (a *matched* threshold, a diagnostic only). `details.threshold_calibration_policy`
and a note record which one was used. On EMBER2024 (LightGBM, 0.5% target) the monthly FPRs are 0.42% → 0.52% →
0.56% under `uniform` and 0.75% → 0.79% → 0.90% under `earliest`.

Metrics added: `auroc_first/last`, `fnr_first/last`, `fpr_first/last`, `max_window_fpr`,
`max_window_fpr_window`, `max_window_fpr_ci95`, `max_fpr_budget`, `windows_over_fpr_budget`;
`details.operating_point` has the full summary (unevaluable windows, windows over budget and
significantly over budget).

## Granularity

The default stays `month`: AUT(F1) depends on the window length, so changing the default would
silently re-anchor `min_aut_f1`. Two finer options exist:

* `granularity: week` (or `day`) — explicit.
* `granularity: auto` — use `week` when the candidate timestamps are day-level (not all on the same
  day of the month), there are 4 to `max_windows` weekly windows, and at least 95% of the rows fall
  in weekly windows with `min_window_samples` rows, `min_window_benign` benign rows and a malicious
  row; otherwise `month`. The decision and its reason are in `details.granularity_auto` and a note.
  EMBER2018 timestamps are month-level, so `auto` gives `month` there. On `ember_v3_2024` it gives 12
  weekly windows of about 40k rows instead of 3 monthly ones, with the same AUT(F1) to three decimals
  (0.9803 vs 0.9802) but a much clearer picture of the FPR (0.83% → 2.04% across weeks).

## Check and score anchors

| check | condition | ideal | floor |
|---|---|---|---|
| `aut_f1_weighted` | `aut_f1_weighted >= min_aut_f1` | min(1, t + 0.2) | max(0, t − 0.3) |
| `window_fpr_within_budget` | max per-window FPR 95% lower bound `<= M1 max_fpr` | none (ungraded) | none (ungraded) |

**Unevaluable, never passed.** If fewer than 2 usable windows remain, the check value is `null`.
With the `warn` gate the status is `warn`, and the finding explains why. This happens when the
effective cutoff is on or after the newest temporal sample in the corpus, when every later sample
is a training member, or when only one window survives selection.

## Skip reasons

* no canonical corpus in the model's feature version (`feature_space`);
* no `training_cutoff` declared on the adapter. The skip message says how to declare one;
* the corpus defines no `temporal` role, or it has no timestamped labeled rows.

## Metrics

`n_windows, first_window, last_window, f1_first, f1_last, f1_drop (= f1_first − f1_last), aut_f1,
aut_f1_weighted, effective_cutoff`, plus the operating-point metrics listed above. Each row of
`details.windows` also has `fnr`, `fpr_ci95`, `fnr_ci95`, `auroc`, `fp`, `fn`.

`details` has `tesseract` (backend), `granularity` (the one used), `granularity_requested`, `granularity_auto`, `corpus_temporal_range`, `duplicates_removed`, `windows` (the
per-window table, including dropped windows with their status and C2 gap), `dropped_windows`,
`beyond_max_windows`, `subsampling`, `n_samples_evaluated`, `temporal_constraints` (`c1`, `c2`,
`c3`) and `unevaluable_reason` when applicable.

## Charts and tables

* `decay`: F1, precision and recall per used window against the period number after the cutoff,
  with a `min_aut_f1` reference line. The chart note maps period numbers to window labels.
* `window_sizes`: malicious and benign samples per considered window (bar), with the
  `min_window_samples` line. Dropped windows are named in the note.
* `window_fpr`: realised FPR per window with its 95% CI band on its own y-scale and the M1 `max_fpr` line.
* `window_miss_rate`: FNR per window with its 95% CI band, own y-scale.
* Table `windows`: per-window results (incl. AUROC, FPR/FNR and their CIs) and status.

## Corpus notes

* **EMBER2018 (`ember_v2_2018`)**: the temporal role is train + test. Rows are dated by the first
  day of their "appeared" month. A model trained on the EMBER2018 train split (cutoff `2018-10`)
  gets two windows, 2018-11 and 2018-12, from the test split.
* **EMBER2024 (`ember_v3_2024`)**: the temporal role is test + challenge (first seen 2023-09 to
  2024-12). A model declaring a cutoff before 2024-09 gets windows through 2024-12.
* A cutoff after the corpus's last date makes M2 unevaluable, which is correct. MalValid has no
  samples newer than your model.

## Scale

Every post-cutoff temporal row is scored once. For EMBER2018 with a 2018-10 cutoff that is 200k
rows × 2381. Use `max_samples_per_window` to bound the cost for slow models. Subsampling is
recorded, and the AUT weights use the evaluated counts.
