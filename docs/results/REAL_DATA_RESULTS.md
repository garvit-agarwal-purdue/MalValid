# Real-data results: MalValid run on the example detectors

Date: 2026-09-30. Machine: HPC cluster node (AMD EPYC 9654; a 28-core allocation shared with other jobs, `nproc` reports 96; each run used at most 8 threads and at most 2 runs were in flight at once, so wall times include contention). MalValid version 0.1.0a1 (the pre-release then named malguard; the results are unchanged by the rename). Sandbox: bwrap with the network isolated, for every run (0 restarts, 0 timeouts, 0 crashes in all runs).

These are real runs of the real gate on the real corpora.

> **Note on M2 statuses.** These runs predate M2's per-window false-positive check
> (`window_fpr_within_budget`: no monthly window whose FPR is significantly above M1's `max_fpr`). That
> check is ungraded, so it never changes a score, but with current code it can turn an M2 `pass` below into
> `warn` (which only matters for a verdict that would otherwise be `READY`). Re-run with current code for
> current statuses. The synthetic demo was re-run with current code: its M2 is now `warn` (worst window
> FPR 2.3%, 95% lower bound 1.07%), with the same score and verdict. The machine-readable copy is [`results.json`](results.json). Raw reports (`report.json`, `report.html`, private dirs) stay in the build scratch directory (`<build-dir>/release_realdata/runs/`) and are not copied here.

## Corpora (content hashes verified by the loader on every run)

| Corpus | Rows x features | Content hash (sha256) |
|---|---|---|
| `ember_v2_2018` | 1,000,000 x 2381 | `b76ea441be1a62197c7d19801174309e988eea54740ff22f03fa75e66beb4b98` |
| `ember_v3_2024` | 484,855 x 2568 | `4f275e15e00ee11cdc67f176bbda932b3feaa92ec6d8584420cb04714e5110cc` |
| `synthetic_v2` (generated on first load, synthetic) | 2381 features | `7a57a1f2ce89898e7b101bbd7478c6756d6666743ac629a35683326b506fe761` |

## Model artifacts

| Example | Model sha256 |
|---|---|
| lightgbm_ember2018 (published EMBER2018 LightGBM, 1000 trees) and the no-tree stress run | `509de4a83e2d76b69fb3025ced0429e5f40f6a95ca40122b9eedd03317fff5e2` |
| xgboost_ember2018 (600 rounds, trained by the example's train.py) | `64901aec8111cf940e488efea9098eaafba01ff7e41554d527458f00821583b6` |
| lightgbm_ember2024 (published EMBER2024 PE model, 500 trees) | `4252027863492ac138785c8c18576f43dad77d00faddc14e8c0072e8db419f99` |
| synthetic_demo (200 trees, train.py) | `33153fc398ac646e97e0ba425b532e794711b064bdabdd43965024e061b74974` |

## Cross-model summary

| Run | Verdict | Score | Coverage | M1 FPR / DR | M2 AUT(F1) | M4 advantage | M5 flags | M7 controllable share | Wall time | Exit |
|---|---|---|---|---|---|---|---|---|---|---|
| lightgbm_ember2018 | CONDITIONAL | 80.5 | 100% | 1.00% / 96.5% | 0.977 | 0.178 (upper bound) | 0 | 39.5% (tree_shap) | 546 s | 0 |
| xgboost_ember2018 | BLOCKED | 49.0 | 100% | 0.89% / 92.5% | 0.957 | 0.136 (upper bound) | 0 | 34.1% (tree_shap) | 83 s | 1 |
| lightgbm_ember2024 (ember_v3) | BLOCKED | 49.0 | 87% | 1.28% / 97.5% | 0.980 | skipped (no manifest) | 0 | 44.7% (tree_shap) | 173 s | 1 |
| synthetic_demo (synthetic data) | BLOCKED | 45.2 | 100% | 1.21% / 69.4% | 0.808 | 0.228 | 0 | 44.7% (tree_shap) | 10 s | 1 |
| lightgbm_ember2018 + M6 | CONDITIONAL | 80.3 | 100% | 1.00% / 96.5% | 0.977 | 0.178 (upper bound) | 0 | 39.5% (tree_shap) | 594 s | 0 |
| lightgbm_ember2018, no tree access | CONDITIONAL | 77.7 | 87% | 1.00% / 96.5% | 0.977 | 0.178 (upper bound) | skipped (needs trees) | 37.0% (permutation_shap) | 255 s | 0 |
| no-tree adapter, M5 triggers only: partial run (`--only M5`), not a readiness score | CONDITIONAL | 100.0 recorded (M0 + M5 only) | 100% of enabled modules (1 of 7 run) | not run | not run | not run | trigger test ok (0 flagged) | not run | 9 s | 0 |

Reading the table:

- The headline numbers are the published EMBER2018 LightGBM model (CONDITIONAL, 80.5) and the two BLOCKED detectors. The XGBoost model is blocked by the M1 hard gate because its detection rate at 0.89% FPR is 92.5% (limit 95%). The EMBER2024 model is blocked because its declared threshold 0.5 gives 1.28% FPR (limit 1%). Both are genuine results, not bugs; the example READMEs say so.
- A BLOCKED score is capped at 49. The uncapped (raw) scores were 78.1 (XGBoost) and 79.9 (EMBER2024). The synthetic demo's raw score equals its capped score (45.2) because it was already below the cap.
- M4 advantage on the EMBER2018 corpus is an **upper bound**: training members (all before 2018-11) and non-members (the corpus's post-cutoff rows) do not overlap in time, so the attack partly detects time shift. The synthetic demo's M4 is time-matched, not a bound.
- M7 'controllable share' is the controllability-weighted share of attribution (limit 50%). The unweighted share of attribution on features classed fully `controllable` is much lower, see the per-run tables.
- `synthetic_demo` numbers exercise the pipeline and say nothing about real detectors.

## Run 1a: examples/lightgbm_ember2018, default gate

Command (`$R` = `<build-dir>/release_realdata`, the maintainers' build scratch directory; run from the example directory; `MALVALID_CORPUS_DIR=$HOME/malvalid-corpora`, `OMP_NUM_THREADS=8`, wrapped in `/usr/bin/time -v`):

```
malvalid run --adapter examples/lightgbm_ember2018/adapter.py --out $R/runs/lightgbm_ember2018
```

Exit code 0. Verdict **CONDITIONAL**, score **80.5**/100, coverage 100%. Total wall time 546 s (run duration 544 s; user CPU 474 s), peak RSS 3.24 GB. Sandbox startup 3.349 s, model load 1.557 s. Tree access: yes (verified: tree dump matches predict_proba (max |Δ| = 5.6e-17 on 2000 corpus rows)).

Why not READY: M4 Membership inference (training-data leakage): warn gate failed.

Disabled in config: extraction.

| Module | Status | Gate | Axis score | Key metric vs threshold | Duration (s) | Share of 1800 s budget |
|---|---|---|---|---|---|---|
| M0 file_safety | pass | hard / passed | 100.0 | critical_findings 0 (need == 0.0) ok; findings_at_or_above_high 0 (need == 0.0) ok; pickle_without_allow_pickle 0 (need == 0.0) ok | 0.1 | 0.0% |
| M1 performance | pass | hard / passed | 75.0 | fpr 0.01 (need <= 0.01) ok; detection_rate 0.965 (need >= 0.95) ok | 18.5 | 1.0% |
| M2 drift | pass | warn / passed | 100.0 | aut_f1_weighted 0.9772 (need >= 0.7) ok | 20.1 | 1.1% |
| M4 membership_inf | warn | warn / failed | 45.9 | membership_advantage 0.1776 (need <= 0.1) FAIL | 3.4 | 0.2% |
| M5 backdoor_screen | pass | warn / passed | 100.0 | n_flagged_rules 0 (need <= 0.0) ok | 1.4 | 0.1% |
| M7 explanation | pass | warn / passed | 82.5 | controllable_share 0.3945 (need <= 0.5) ok; top_feature_share 0.04942 (need <= 0.3) ok | 464.1 | 25.8% |

- M1: AUROC 0.9964, TPR at 0.1% FPR 86.9%, ECE 0.0153, eval rows 100,000 benign + 100,000 malicious.
- M4: worst attack art_rf, advantage 0.178 (AUROC up to 0.615), members used 5000, sampling design nearest_time.
- M5: 1000 trees, 1,479,647 rules scanned in 1.4 s, top suspicion 0.11 (flag threshold 0.50).
- M7: source tree_shap, rows explained 531 of 2000 requested, top feature section.num_rx (append_only) at 4.9%, controllable-weighted share 39.5%; of total attribution: 12.6% controllable, 35.8% append-only, 42.5% derived, 9.1% fixed.

## Run 1b: examples/xgboost_ember2018, default gate

Command (`$R` = `<build-dir>/release_realdata`, the maintainers' build scratch directory; run from the example directory; `MALVALID_CORPUS_DIR=$HOME/malvalid-corpora`, `OMP_NUM_THREADS=8`, wrapped in `/usr/bin/time -v`):

```
malvalid run --adapter examples/xgboost_ember2018/adapter.py --out $R/runs/xgboost_ember2018
```

Exit code 1. Verdict **BLOCKED**, score **49.0**/100, raw 78.1, capped at 49, coverage 100%. Total wall time 83 s (run duration 81 s; user CPU 70 s), peak RSS 2.78 GB. Sandbox startup 2.399 s, model load 0.444 s. Tree access: yes (verified: tree dump matches predict_proba (max |Δ| = 2.5e-07 on 2000 corpus rows)).

Why not READY: M1 Performance & calibration: hard gate failed.

Disabled in config: extraction.

| Module | Status | Gate | Axis score | Key metric vs threshold | Duration (s) | Share of 1800 s budget |
|---|---|---|---|---|---|---|
| M0 file_safety | pass | hard / passed | 100.0 | critical_findings 0 (need == 0.0) ok; findings_at_or_above_high 0 (need == 0.0) ok; pickle_without_allow_pickle 0 (need == 0.0) ok | 0.0 | 0.0% |
| M1 performance | fail | hard / failed | 62.5 | fpr 0.00891 (need <= 0.01) ok; detection_rate 0.9251 (need >= 0.95) FAIL | 6.3 | 0.4% |
| M2 drift | pass | warn / passed | 100.0 | aut_f1_weighted 0.9566 (need >= 0.7) ok | 5.8 | 0.3% |
| M4 membership_inf | warn | warn / failed | 61.5 | membership_advantage 0.136 (need <= 0.1) FAIL | 2.6 | 0.1% |
| M5 backdoor_screen | pass | warn / passed | 100.0 | n_flagged_rules 0 (need <= 0.0) ok | 0.4 | 0.0% |
| M7 explanation | pass | warn / passed | 86.4 | controllable_share 0.3409 (need <= 0.5) ok; top_feature_share 0.02339 (need <= 0.3) ok | 59.5 | 3.3% |

- M1: AUROC 0.9931, TPR at 0.1% FPR 64.8%, ECE 0.0089, eval rows 100,000 benign + 100,000 malicious.
- M4: worst attack art_gb, advantage 0.136 (AUROC up to 0.591), members used 5000, sampling design nearest_time.
- M5: 600 trees, 103,554 rules scanned in 0.3 s, top suspicion 0.11 (flag threshold 0.50).
- M7: source tree_shap, rows explained 2000 of 2000 requested, top feature section.num_rx (append_only) at 2.3%, controllable-weighted share 34.1%; of total attribution: 10.4% controllable, 31.6% append-only, 51.7% derived, 6.3% fixed.

## Run 1c: examples/lightgbm_ember2024, default gate (corpus ember_v3_2024)

Command (`$R` = `<build-dir>/release_realdata`, the maintainers' build scratch directory; run from the example directory; `MALVALID_CORPUS_DIR=$HOME/malvalid-corpora`, `OMP_NUM_THREADS=8`, wrapped in `/usr/bin/time -v`):

```
malvalid run --adapter examples/lightgbm_ember2024/adapter.py --out $R/runs/lightgbm_ember2024 --corpus ember_v3_2024
```

Exit code 1. Verdict **BLOCKED**, score **49.0**/100, raw 79.9, capped at 49, coverage 87%. Total wall time 173 s (run duration 172 s; user CPU 127 s), peak RSS 5.41 GB. Sandbox startup 1.495 s, model load 0.064 s. Tree access: yes (verified: tree dump matches predict_proba (max |Δ| = 1.1e-16 on 2000 corpus rows)).

Why not READY: M1 Performance & calibration: hard gate failed; M4 Membership inference (training-data leakage) was skipped (lowers coverage).

Disabled in config: extraction.

| Module | Status | Gate | Axis score | Key metric vs threshold | Duration (s) | Share of 1800 s budget |
|---|---|---|---|---|---|---|
| M0 file_safety | pass | hard / passed | 100.0 | critical_findings 0 (need == 0.0) ok; findings_at_or_above_high 0 (need == 0.0) ok; pickle_without_allow_pickle 0 (need == 0.0) ok | 0.0 | 0.0% |
| M1 performance | fail | hard / failed | 63.5 | fpr 0.01281 (need <= 0.01) FAIL; detection_rate 0.9753 (need >= 0.95) ok | 30.4 | 1.7% |
| M2 drift | pass | warn / passed | 100.0 | aut_f1_weighted 0.9802 (need >= 0.7) ok | 12.4 | 0.7% |
| M4 membership_inf | skipped | warn / not_evaluated | - | skipped: no training manifest (training_hashes_path not declared) | - | - |
| M5 backdoor_screen | pass | warn / passed | 100.0 | n_flagged_rules 0 (need <= 0.0) ok | 0.4 | 0.0% |
| M7 explanation | pass | warn / passed | 78.8 | controllable_share 0.4469 (need <= 0.5) ok; top_feature_share 0.07106 (need <= 0.3) ok | 118.3 | 6.6% |

- M1: AUROC 0.9983, TPR at 0.1% FPR 90.1%, ECE 0.0059, eval rows 239,987 benign + 240,000 malicious, detection on the challenge set 70.6%.
- M5: 500 trees, 72,423 rules scanned in 0.4 s, top suspicion 0.04 (flag threshold 0.50).
- M7: source tree_shap, rows explained 2000 of 2000 requested, top feature header.coff.characteristics.DLL (fixed) at 7.1%, controllable-weighted share 44.7%; of total attribution: 23.0% controllable, 28.9% append-only, 36.9% derived, 11.2% fixed.

## Run 1d: examples/synthetic_demo, its own gate.yaml (corpus synthetic_v2)

Command (`$R` = `<build-dir>/release_realdata`, the maintainers' build scratch directory; run from the example directory; `MALVALID_CORPUS_DIR=$HOME/malvalid-corpora`, `OMP_NUM_THREADS=8`, wrapped in `/usr/bin/time -v`):

```
malvalid run --adapter examples/synthetic_demo/adapter.py --config examples/synthetic_demo/gate.yaml --out $R/runs/synthetic_demo
```

Exit code 1. Verdict **BLOCKED**, score **45.2**/100, coverage 100%. Total wall time 10 s (run duration 9 s; user CPU 7 s), peak RSS 0.53 GB. Sandbox startup 1.712 s, model load 0.026 s. Tree access: yes (verified: tree dump matches predict_proba (max |Δ| = 2.2e-16 on 2000 corpus rows)).

Why not READY: M1 Performance & calibration: hard gate failed.

Disabled in config: extraction.

| Module | Status | Gate | Axis score | Key metric vs threshold | Duration (s) | Share of 1800 s budget |
|---|---|---|---|---|---|---|
| M0 file_safety | pass | hard / passed | 100.0 | critical_findings 0 (need == 0.0) ok; findings_at_or_above_high 0 (need == 0.0) ok; pickle_without_allow_pickle 0 (need == 0.0) ok | 0.0 | 0.0% |
| M1 performance | fail | hard / failed | 0.0 | fpr 0.01209 (need <= 0.01) FAIL; detection_rate 0.6936 (need >= 0.95) FAIL | 0.4 | 0.0% |
| M2 drift | warn (current code; pass when measured) | warn / failed | 88.5 | aut_f1_weighted 0.8078 (need >= 0.7) ok; window_fpr_lower_bound 0.01066 (need <= 0.01) FAIL (ungraded) | 0.3 | 0.0% |
| M4 membership_inf | warn | warn / failed | 27.1 | membership_advantage 0.2277 (need <= 0.1) FAIL | 1.2 | 0.1% |
| M5 backdoor_screen | pass | warn / passed | 100.0 | n_flagged_rules 0 (need <= 0.0) ok | 0.1 | 0.0% |
| M7 explanation | pass | warn / passed | 78.8 | controllable_share 0.4466 (need <= 0.5) ok; top_feature_share 0.0489 (need <= 0.3) ok | 2.5 | 0.1% |

- M1: AUROC 0.9784, TPR at 0.1% FPR 34.7%, ECE 0.0569, eval rows 6,449 benign + 6,381 malicious, detection on the challenge set 11.0%.
- M4: worst attack loss_threshold, advantage 0.228 (AUROC up to 0.642), members used 1255, sampling design time_matched.
- M5: 200 trees, 6,200 rules scanned in 0.1 s, top suspicion 0.33 (flag threshold 0.50).
- M7: source tree_shap, rows explained 2000 of 2000 requested, top feature section.entry_name_hashed[39] (controllable) at 4.9%, controllable-weighted share 44.7%; of total attribution: 20.8% controllable, 31.9% append-only, 46.5% derived, 0.9% fixed.

## Run 2: examples/lightgbm_ember2018 with extraction (M6) enabled

Command (`$R` = `<build-dir>/release_realdata`, the maintainers' build scratch directory; run from the example directory; `MALVALID_CORPUS_DIR=$HOME/malvalid-corpora`, `OMP_NUM_THREADS=8`, wrapped in `/usr/bin/time -v`):

```
malvalid run --adapter examples/lightgbm_ember2018/adapter.py --config $R/configs/extraction.yaml --out $R/runs/lightgbm_ember2018_m6
```

Exit code 0. Verdict **CONDITIONAL**, score **80.3**/100, coverage 100%. Total wall time 594 s (run duration 592 s; user CPU 652 s), peak RSS 5.15 GB. Sandbox startup 3.274 s, model load 1.594 s. Tree access: yes (verified: tree dump matches predict_proba (max |Δ| = 5.6e-17 on 2000 corpus rows)).

Why not READY: M4 Membership inference (training-data leakage): warn gate failed.

| Module | Status | Gate | Axis score | Key metric vs threshold | Duration (s) | Share of 1800 s budget |
|---|---|---|---|---|---|---|
| M0 file_safety | pass | hard / passed | 100.0 | critical_findings 0 (need == 0.0) ok; findings_at_or_above_high 0 (need == 0.0) ok; pickle_without_allow_pickle 0 (need == 0.0) ok | 0.1 | 0.0% |
| M1 performance | pass | hard / passed | 75.0 | fpr 0.01 (need <= 0.01) ok; detection_rate 0.965 (need >= 0.95) ok | 17.6 | 1.0% |
| M2 drift | pass | warn / passed | 100.0 | aut_f1_weighted 0.9772 (need >= 0.7) ok | 17.1 | 0.9% |
| M4 membership_inf | warn | warn / failed | 45.9 | membership_advantage 0.1776 (need <= 0.1) FAIL | 3.0 | 0.2% |
| M5 backdoor_screen | pass | warn / passed | 100.0 | n_flagged_rules 0 (need <= 0.0) ok | 1.3 | 0.1% |
| M6 extraction | pass | warn / passed | 77.6 | surrogate_fidelity 0.9344 (need <= 0.95) ok | 51.8 | 2.9% |
| M7 explanation | pass | warn / passed | 82.5 | controllable_share 0.3945 (need <= 0.5) ok; top_feature_share 0.04942 (need <= 0.3) ok | 465.1 | 25.8% |

- M1: AUROC 0.9964, TPR at 0.1% FPR 86.9%, ECE 0.0153, eval rows 100,000 benign + 100,000 malicious.
- M4: worst attack art_rf, advantage 0.178 (AUROC up to 0.615), members used 5000, sampling design nearest_time.
- M5: 1000 trees, 1,479,647 rules scanned in 1.3 s, top suspicion 0.11 (flag threshold 0.50).
- M6: surrogate fidelity by query budget {100: 0.75, 1000: 0.869, 10000: 0.934, 50000: 0.957} (hard-label queries, KnockoffNets, lightgbm surrogate); model accuracy 0.979; majority baseline 0.516.
- M7: source tree_shap, rows explained 531 of 2000 requested, top feature section.num_rx (append_only) at 4.9%, controllable-weighted share 39.5%; of total attribution: 12.6% controllable, 35.8% append-only, 42.5% derived, 9.1% fixed.

## Run 3: no-tree-access stress run (same EMBER2018 model behind a black-box adapter)

Command (`$R` = `<build-dir>/release_realdata`, the maintainers' build scratch directory; run from the example directory; `MALVALID_CORPUS_DIR=$HOME/malvalid-corpora`, `OMP_NUM_THREADS=8`, wrapped in `/usr/bin/time -v`):

```
malvalid run --adapter $R/notrees/adapter.py --out $R/runs/notrees_lightgbm_ember2018
```

Exit code 0. Verdict **CONDITIONAL**, score **77.7**/100, coverage 87%. Total wall time 255 s (run duration 253 s; user CPU 42 s), peak RSS 5.61 GB. Sandbox startup 3.208 s, model load 1.406 s. Tree access: no (the model does not expose its tree structure; tree-based analyses are unavailable).

Why not READY: score 77.7 is below the READY band (80); M4 Membership inference (training-data leakage): warn gate failed; M5 Backdoor / poisoning screening was skipped (lowers coverage).

Disabled in config: extraction.

| Module | Status | Gate | Axis score | Key metric vs threshold | Duration (s) | Share of 1800 s budget |
|---|---|---|---|---|---|---|
| M0 file_safety | pass | hard / passed | 100.0 | critical_findings 0 (need == 0.0) ok; findings_at_or_above_high 0 (need == 0.0) ok; pickle_without_allow_pickle 0 (need == 0.0) ok | 0.1 | 0.0% |
| M1 performance | pass | hard / passed | 75.0 | fpr 0.01 (need <= 0.01) ok; detection_rate 0.965 (need >= 0.95) ok | 17.7 | 1.0% |
| M2 drift | pass | warn / passed | 100.0 | aut_f1_weighted 0.9772 (need >= 0.7) ok | 17.7 | 1.0% |
| M4 membership_inf | warn | warn / failed | 45.9 | membership_advantage 0.1776 (need <= 0.1) FAIL | 6.3 | 0.3% |
| M5 backdoor_screen | skipped | warn / not_evaluated | - | skipped: model does not expose its tree structure (the static scan needs the trees) and no trigger hypotheses were supplied (backdoor_screen.triggers / triggers_path) | 0.0 | 0.0% |
| M7 explanation | pass | warn / passed | 84.3 | controllable_share 0.3699 (need <= 0.5) ok; top_feature_share 0.05561 (need <= 0.3) ok | 205.0 | 11.4% |

- M1: AUROC 0.9964, TPR at 0.1% FPR 86.9%, ECE 0.0153, eval rows 100,000 benign + 100,000 malicious.
- M4: worst attack art_rf, advantage 0.178 (AUROC up to 0.615), members used 5000, sampling design nearest_time.
- M7: source permutation_shap, rows explained 52 of 2000 requested, top feature section.num_rx (append_only) at 5.6%, controllable-weighted share 37.0%; of total attribution: 12.0% controllable, 33.3% append-only, 49.4% derived, 5.3% fixed.

## Run 3b (supplementary): no-tree stress adapter, M5 trigger-hypothesis path only (--only M5)

Command (`$R` = `<build-dir>/release_realdata`, the maintainers' build scratch directory; run from the example directory; `MALVALID_CORPUS_DIR=$HOME/malvalid-corpora`, `OMP_NUM_THREADS=8`, wrapped in `/usr/bin/time -v`):

```
malvalid run --adapter $R/notrees/adapter.py --config $R/configs/triggers.yaml --out $R/runs/notrees_m5_triggers --only M5
```

Exit code 0. Verdict **CONDITIONAL**, score **100.0**/100, coverage 100%. Total wall time 9 s (run duration 7 s; user CPU 5 s), peak RSS 0.83 GB. Sandbox startup 3.137 s, model load 1.354 s. Tree access: no (the model does not expose its tree structure; tree-based analyses are unavailable).

Why not READY: performance must pass for a READY verdict (not run).

Disabled in config: performance, drift, membership_inf, extraction, explanation.

| Module | Status | Gate | Axis score | Key metric vs threshold | Duration (s) | Share of 1800 s budget |
|---|---|---|---|---|---|---|
| M0 file_safety | pass | hard / passed | 100.0 | critical_findings 0 (need == 0.0) ok; findings_at_or_above_high 0 (need == 0.0) ok; pickle_without_allow_pickle 0 (need == 0.0) ok | 0.2 | 0.0% |
| M5 backdoor_screen | pass | warn / passed | 100.0 | max_trigger_drop 0.0075 (need <= 0.2) ok | 0.7 | 0.0% |

- M5: trigger test only, worst trigger zero_timestamp lowered detection by 0.7 pp.

## Stress run detail: no tree access (Run 3)

The adapter in `$R/notrees/adapter.py` loads the same `ember_model_2018.txt` but stores the booster in a name-mangled private attribute, so it exposes neither `native_model` nor `tree_ensemble()`. MalValid reports: "the adapter exposes neither tree_ensemble() nor native_model". Only `predict_proba` is available through the sandbox.

| Measure | With trees (Run 1a) | No tree access (Run 3) |
|---|---|---|
| Verdict / score / coverage | CONDITIONAL / 80.5 / 100% | CONDITIONAL / 77.7 / 87% |
| M1 and M2 | identical (FPR 1.00%, DR 96.5%, AUT 0.977) | identical (black-box paths) |
| M4 advantage | 0.1776 | 0.1776 (identical; M4 is black-box) |
| M5 | pass, 1000 trees, 1,479,647 rules, 0 flagged, 1.4 s | **skipped**: no tree structure and no trigger hypotheses supplied |
| M7 attribution | exact TreeSHAP, 531 rows, 464 s | model-agnostic permutation SHAP, **52 rows** (2000 requested), 205 s |
| M7 controllable share / top feature | 39.5% / section.num_rx 4.9% | 37.0% / section.num_rx 5.6% |
| Total wall time | 546 s | 255 s |

- **M7 finished well inside `max_seconds_per_module` (1800 s)**: 205 s, 11.4% of the budget. It capped the explained rows at 52 (background 16 rows) to bound the number of model queries for a 2381-feature schema, and said so in its notes. The attribution is therefore estimated from far fewer rows than the exact path, but it agreed closely with exact TreeSHAP: same top feature (`section.num_rx`), controllable share 37.0% versus 39.5%, top-feature share 5.6% versus 4.9%. Both verdicts on the M7 gates are pass.
- **M5 takes its degraded path as designed**: without trees and without trigger hypotheses it is skipped (and so lowers coverage to 87%, and is not counted as a pass). Run 3b supplies two trigger hypotheses in a config (`header.coff.timestamp = 0`, `strings.numstrings = 5`) and runs `--only M5` through the sandbox: the trigger test runs with 2000 malicious rows, detection before 96.1%, the worst trigger (`zero_timestamp`) lowers detection by 0.7 pp (limit 20 pp), so M5 passes. MalValid correctly notes that `few_strings` stamps values outside the range seen in the detected vectors (an out-of-distribution caution).
- The score is lower (77.7 vs 80.5) only because M5 (score 100) drops out of the weighted mean and M7's score is 84.3 instead of 82.5. The verdict is CONDITIONAL in both cases.

## Anomalies and observations

- **No module errored, timed out, crashed a worker, or was skipped unexpectedly** in any of the seven runs. Skips that did occur were intended: M6 is off by default; M4 skips for the EMBER2024 example (no `training_hashes_path`); M5 skips for the no-tree adapter.
- **Nothing came close to its time budget.** The slowest module was M7 exact TreeSHAP on the 1000-tree EMBER2018 model at 464 s to 465 s (26% of 1800 s). M7 caps TreeSHAP work at about 600 s and so explained 531 rows instead of the requested 2000 (about 1.13 s per row); it says so in its notes. The other EMBER models explained all 2000 rows (XGBoost 60 s, EMBER2024 118 s).
- **Wall time of the default EMBER2018 run is about 9 minutes** (546 s), dominated by M7. `examples/README.md` gives about 10 minutes for this run.
- The EMBER2018 LightGBM model sits exactly on the M1 FPR limit (FPR 1.00% at its declared threshold 0.8336, which is the 1%-FPR operating point, so there is no margin). M1 passes the hard gate but scores only 75.0 (threshold maps to 0.75), and notes that the 95% interval of the FPR reaches 1.06%. A slightly higher operating threshold would give margin.
- M4 advantage is above the 0.10 warn limit for both real models where M4 ran (0.178 EMBER2018 LightGBM, 0.136 XGBoost) and for the synthetic demo (0.228); for EMBER2018 it is an upper bound (no time overlap). The published LightGBM model has member accuracy 1.000 versus non-member accuracy 0.979 on the sampled pairs.
- EMBER2024 M1 also reports detection on the challenge set: 70.6%, far below the 97.5% on the ordinary eval set. This is informational (not gated); the challenge split consists of samples the benchmark deliberately makes harder, so a large gap is plausible, but it was not investigated here.
- EMBER2024 M5 notes that 371 categorical splits were rewritten as threshold chains, so cover-based suspicion values under those splits are approximate.
- Run 3b reports verdict CONDITIONAL, score 100.0 and coverage 100% with only M0 and M5 enabled: coverage is measured over the modules that are enabled, so disabling a module in the config does not lower it (the reasons line does say 'performance must pass for a READY verdict (not run)').
- Runs 1a and Run 3 started at the same time and Runs 1b, 1c, 2 overlapped with others (2 concurrent runs maximum); durations include that contention and the shared node's load.

## Calibration period of an auto-calibrated threshold (model-file submissions, 2026-10-01)

The runs above all use a **declared** threshold (an adapter's `operating_threshold`), so the calibration
period policy does not apply to them. It applies only to model-file submissions with `--calibrate-fpr`
(or no threshold). These runs compare the two policies (`runtime.calibration_period`, default `earliest`
since 2026-10-01; before that MalValid always used `uniform`):

```bash
M=<models-dir>; export MALVALID_CORPUS_DIR=<corpora> OMP_NUM_THREADS=8
malvalid run --model $M/EMBER2024_PE.model --calibrate-fpr 0.005 --calibration-period earliest|uniform \
    --training-cutoff 2024-09-21 --out ...                                  # full run
malvalid run --model $M/ember_model_2018.txt --calibrate-fpr 0.005 --calibration-period earliest|uniform \
    --training-cutoff 2018-10 --training-hashes examples/lightgbm_ember2018/train_sha256.txt --out ...
# the 1% rows: same with --calibrate-fpr 0.01 --only M1,M2 (and no --training-hashes)
```

Benign eval rows available: EMBER2024 test 239,987 (about 17k-21k per week, 2024-09-22..2024-12-14,
day-level dates); EMBER2018 test 100,000 (50,000 dated 2018-11-01 and 50,000 dated 2018-12-01).
`earliest` held out 26,736 EMBER2024 rows (every benign row of 2024-09-22..30; all scored benign rows are
later) and 10,000 EMBER2018 rows (a sha256-ordered tenth of November; 56% of the scored benign rows are
December). `uniform` held out 24,254 and 10,011 rows spread over the whole test period.

| Corpus, target | Policy | Threshold | M1 FPR (Wilson 95%) | FPR 95% with threshold noise | M1 DR (with threshold noise) | M2 monthly FPR | M2 monthly FNR | M1 / verdict |
|---|---|---|---|---|---|---|---|---|
| EMBER2024, 0.5% | uniform | 0.7478 | 0.496% (0.468-0.527) | 0.387-0.582% | 95.55% (94.89-95.92) | 0.42 → 0.52 → 0.56% | 4.1 → 4.8 → 5.1% | pass 78.0 / READY 86.6 |
| EMBER2024, 0.5% | **earliest** | 0.6468 | **0.810% (0.773-0.849)** | 0.692-0.965% | 96.57% (96.17-96.93) | 0.75 → 0.79 → 0.90% | 3.2 → 3.7 → 4.0% | pass 77.3 / READY 86.3 |
| EMBER2024, 1% | uniform | 0.5746 | 1.018% (0.977-1.062) | 0.871-1.160% | 97.10% | 0.87 → 1.02 → 1.21% | 2.7 → 3.1 → 3.5% | fail (FPR) |
| EMBER2024, 1% | **earliest** | 0.4460 | **1.614% (1.561-1.668)** | 1.471-1.806% | 97.83% | 1.44 → 1.59 → 1.80% | 2.1 → 2.4 → 2.6% | fail (FPR) |
| EMBER2018, 0.5% | uniform | 0.9743 | 0.489% (0.445-0.537) | 0.380-0.630% | 94.45% (93.84-95.58) | 0.50 → 0.48% | 4.5 → 6.6% | fail (DR) / BLOCKED (raw 80.5) |
| EMBER2018, 0.5% | earliest | 0.9695 | 0.526% (0.480-0.575) | 0.406-0.681% | 94.60% (94.11-95.73) | 0.53 → 0.53% | 4.4 → 6.4% | fail (DR) / BLOCKED (raw 79.6) |
| EMBER2018, 1% | uniform | 0.8228 | 1.032% (0.968-1.101) | 0.857-1.201% | 96.54% | 0.99 → 1.08% | 2.6 → 4.3% | fail (FPR) |
| EMBER2018, 1% | earliest | 0.7973 | 1.102% (1.036-1.173) | 0.881-1.358% | 96.62% | 1.06 → 1.14% | 2.6 → 4.2% | fail (FPR) |

Reading the table:

- On **EMBER2024** the held threshold reveals a materially different operating point. Fit on the first nine
  days, it gives 1.6x the target FPR on the later data at both targets (0.81% for 0.5%, 1.61% for 1%), and the
  FPR keeps rising month by month. The uniform runs land on the target (0.496%, 1.018%) because their
  calibration rows come from the same weeks they are scored on. The held FPR lies outside the uniform run's
  interval with threshold noise. That interval covers sampling noise only, not drift of the benign data. The
  benign scores of later weeks are higher, so the held threshold is lower and detection is about 1 pp higher.
- On **EMBER2018** the policies differ by less than the noise (0.53% vs 0.49% FPR), and M2 keeps both monthly
  windows (November keeps 40,000 scored benign rows). Its test rows carry only two month-level dates, so
  there is little time structure to hold out.
- Verdicts at the default 0.5% target do not change: EMBER2024 READY in both runs (86.3 vs 86.6), EMBER2018
  BLOCKED by the M1 detection-rate gate in both. The M4 axis of the two EMBER2018 runs differs (advantage 0.181 vs
  0.155, score 44.7 vs 54.5). M4 draws non-members nearest in time to the training members, and `earliest`
  removes November benign rows, the ones closest to training, from that pool. So part of the gap is the
  policy, which adds time shift to an advantage that is already an upper bound. The rest is sampling
  (2,500 test pairs, SE about 0.02).
- Raw outputs: `<build-dir>/insights_item2/` (`benign_counts.json`, `offline_compare.json`,
  `evidence_table.json`, and `runs/`).

## Reproducing

From the repository root (each line runs in its own subshell, so the `cd` does not carry over):

```bash
export MALVALID_CORPUS_DIR=$HOME/malvalid-corpora OMP_NUM_THREADS=8
(cd examples/lightgbm_ember2018 && malvalid run --adapter adapter.py --out runs/lgbm2018)
(cd examples/xgboost_ember2018 && malvalid run --adapter adapter.py --out runs/xgb2018)
(cd examples/lightgbm_ember2024 && malvalid run --adapter adapter.py --corpus ember_v3_2024 --out runs/lgbm2024)
(cd examples/synthetic_demo && malvalid run --adapter adapter.py --config gate.yaml --out runs/demo)
```

Run 2 uses the default gate with `modules.extraction.enabled: true` (all other keys as default). Run 3 uses the default gate with the stress adapter. Both configs and the stress adapter are kept in the build scratch directory `release_realdata/{configs,notrees}/`.
