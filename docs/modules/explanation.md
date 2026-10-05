# M7 — Explanation & spurious-feature reliance (`explanation`)

**Question:** does your detector base its decisions on features that an attacker can set or append
for free?

Several real-world bypasses of ML anti-virus engines needed no change to the malicious code. The
attacker appended benign-looking strings or overlay bytes, and the model flipped its verdict
because it had learned to trust those features. M7 measures where the model's decision mass sits
and checks it against the feature schema's per-feature **controllability**
(`FeatureSchema.feature_controllability()`):

| level | meaning | weight in `controllable_share` |
|---|---|---|
| `controllable` | settable to any value at ~zero cost (COFF timestamp, checksum, debug/certificate directory fields, …) | 1.0 |
| `append_only` | can only be added to (strings, imports, sections, overlay, file size) | 0.75 |
| `derived` | moves only as a side effect of other changes (byte / entropy histograms) | not counted |
| `fixed` | tied to program semantics (machine type, entry-point characteristics, …) | not counted |

The weights are `malvalid.schemas.base.CONTROLLABILITY_WEIGHT`.

> **Attribution only.** M7 shows which features the decisions rest on and whether an attacker
> could set them. It does not perturb inputs or test whether changing a feature actually evades
> the model. Adversarial robustness testing is out of scope for this build.

| | |
|---|---|
| Requires | `tree_access` **or** `feature_space` (at least one) |
| Default gate | `warn` |
| Access used | the normalized `TreeEnsemble` (TreeSHAP and gain, computed in the trusted parent process), or black-box `predict_proba` via `ctx.score` for the model-agnostic path |
| Source | `src/malvalid/modules/explanation.py` |

## Parameters

| key | default | meaning |
|---|---|---|
| `top_k` | `20` | number of top features in the table, chart and `top_k_share` |
| `shap_samples` | `2000` | eval rows to explain (class-balanced, seeded). This is an upper bound: large ensembles and the model-agnostic path are capped (see *Budget*) and the number actually used is recorded |
| `max_controllable_share` | `0.50` | gate: `controllable_share` must be ≤ this |
| `max_top_feature_share` | `0.30` | gate: `top_feature_share` must be ≤ this |

## Method

1. **Global importance from the trees** (tree access): total split gain per feature
   (`TreeEnsemble.feature_importance("gain")`). Split counts are used if the trees carry no gains.
2. **Mean |SHAP| over eval rows.** The sample draws half benign and half malicious rows from the
   corpus's `eval` role, excludes training-manifest members, and keeps one row per sha256.
   * *Trees available:* exact TreeSHAP, `shap.TreeExplainer(TreeEnsemble.to_shap_model())` in
     path-dependent mode, in raw-margin (log-odds) space. If the ensemble has no training cover
     (for example an ONNX export, `meta.cover == "uniform"`), interventional TreeSHAP is used
     instead, with 50 background eval rows.
   * *No tree access:* model-agnostic `shap.PermutationExplainer` through `ctx.score`, in
     probability space. It uses one antithetic permutation per row
     (`max_evals = 2·dim + 1`, which gives exact efficiency) and an `Independent` masker over a
     disjoint background sample: 32 rows for dim ≤ 512, otherwise 16. The explainer's global
     numpy seeding is isolated: the state is saved and restored around the call.
   * *Trees but no corpus:* SHAP needs eval rows, so the shares come from split gain, with a note.
3. **Shares.** `s_f = a_f / Σ a`, where `a` is the mean |SHAP| (or the gain on the gain-only path).
   The shares are aggregated per feature group and per controllability level, both for SHAP and
   for gain.

### Metrics

* `controllable_share = Σ_f w(c_f) · s_f` over `controllable` and `append_only` features.
* `top_feature_share = max_f s_f`, with `top_feature` and `top_feature_controllability`.
* `effective_n_features = 1 / Σ_f s_f²` (inverse Herfindahl index).
* `top_k_share`, `top_k_attacker_controllable_share`, `n_top_k_attacker_controllable`.
* `share_controllable / share_append_only / share_derived / share_fixed` (unweighted).
* `gain_controllable_share`, `gain_top_feature_share`, `gain_top_feature` (tree access only).
* `attribution_source` (`tree_shap` | `tree_shap_interventional` | `permutation_shap` |
  `tree_gain`), `shap_samples_requested`, `shap_samples_used`, `n_features_used`.

`details`: `top_features` (rank, feature, group, controllability, the schema's reason for that
level when it provides `controllability_reasons()` (ember_v2 does), share, gain share), `by_group`,
`by_controllability` (and their `_gain` versions), and `attribution` (sample composition, budget
cap, estimated cost, seconds, output space).

## Checks and score

| check | condition | anchors |
|---|---|---|
| `controllable_share` | `≤ max_controllable_share` | ideal 0.15, floor 0.9 |
| `top_feature_share` | `≤ max_top_feature_share` | ideal 0.05, floor 0.6 |

The axis score is the weaker of the two check scores.

## Charts and tables

* `top_features` (bar): shares of the `top_k` features, with a second series for split gain when
  both are available.
* `group_importance` (bar): share per feature group (SHAP and gain).
* `top_features_table` (table): the top features cross-referenced with group and controllability.
  When the schema offers `controllability_reasons()`, a `why (schema)` column gives the one-line
  justification of each feature's level (for example, for `header.coff.timestamp`: "link timestamp
  is informational; any value works").

## Budget (time and queries)

Exact TreeSHAP costs roughly `Σ_leaves (depth² + depth)` per row, and M7 computes this
deterministically from the tree structure (`treeshap_cost`, measured at 2.1–3.1 ns per unit on the
build host; budgeted at 4 ns). The number of rows explained is
`min(shap_samples, max(20, 600 s / estimated cost per row))`. The 600 s term is also limited to 40 %
of the module's remaining time. The chunk loop stops early if less than 90 s of the module budget is
left, and a note records this. Both the requested and the used sample sizes are reported.

* EMBER2018 reference LightGBM (1000 trees, ~1500 leaves each, 2381 features): the estimate is
  1.13 s per row, so **531 rows** are explained (265 benign, 266 malicious). This took **469 s**
  (0.88 s per row; shap's TreeSHAP runs on one core), well inside the 1800 s module limit.
  That sample is large enough. Mean |SHAP| over a 2000-row class-balanced reference sample
  (LightGBM's own multi-threaded `pred_contrib`, the same path-dependent TreeSHAP) gives
  `controllable_share` 39.06 % and `top_feature_share` 4.77 %. Resampling 531 of those rows
  20 times gives 39.11 % ± 0.23 pp and 4.79 % ± 0.08 pp (± one standard deviation). At 100 rows
  the spread is still only ± 0.6 pp and ± 0.3 pp.
* EMBER2024 reference LightGBM (500 trees, 2568 features, 371 categorical splits rewritten by the
  loader): the estimate is 0.095 s per row, so all **2000** requested rows are explained, in
  **119 s**. TreeSHAP on the rewritten ensemble matches LightGBM's `pred_contrib` on the original
  model to 1.9e-12, so the rewrite does not change the attributions.
* Toy models: a fraction of a second.

The model-agnostic path is bounded by a model-query budget of at most 4 M model rows,
`n = min(shap_samples, 256, max(8, 4 000 000 / ((2·dim + 1) · background)))`. For `toy_v1` that is
256 rows; for ember_v2 / ember_v3 it is 52 / 48 rows. Only features that differ from the
background are permuted, so the actual count is usually lower, and every row sent to the model is
counted in `details.attribution.model_rows_scored`. The explainer runs in chunks of 4 rows, which
gives the same values as a single call. If a slow model brings the module within 90 s (plus one
chunk) of its deadline, it stops early, and a note records how many rows were explained. Rows are
processed benign, malicious, benign, … so a shortened sample stays class-balanced.

Measured with the EMBER2018 reference LightGBM in-process, with its trees hidden (4 threads):
52 rows took **334 s** and scored 2.2 M model rows. The result agrees with exact TreeSHAP on the
same model: `controllable_share` 37.4 % vs 39.5 %, and the same top feature, `section.num_rx`
(5.4 % vs 4.9 %). Through the sandbox, expect extra time for transferring the vectors
(2.2 M × 2381 float32 ≈ 21 GB).

## Results on the reference models

These runs use the full canonical corpora and the reference LightGBM models in-process
(`malvalid.testing.InProcessModel`, trees via `LightGBMLoader.tree_ensemble`), at default
parameters.

| | EMBER2018 (`ember_v2_2018`) | EMBER2024 (`ember_v3_2024`) |
|---|---|---|
| attribution | exact TreeSHAP, 531 of 2000 requested rows (time cap) | exact TreeSHAP, 2000 rows |
| module wall time | 470 s | 119 s |
| `controllable_share` (limit 0.50) | **39.5 %** | **44.7 %** |
| `top_feature_share` (limit 0.30) | 4.9 % (`section.num_rx`, append_only) | 7.1 % (`header.coff.characteristics.DLL`, fixed) |
| `effective_n_features` | 101.9 | 86.8 |
| top-20 share / attacker-controllable among the top 20 | 33.2 % / 14 | 33.2 % / 12 |
| unweighted share: controllable / append_only / derived / fixed | 12.6 / 35.8 / 42.5 / 9.1 % | 23.0 / 28.9 / 36.9 / 11.2 % |
| split-gain view: `controllable_share` / top feature | 33.9 % / `header.coff.characteristics_hashed[0]` 10.0 % | 39.6 % / `header.coff.characteristics.DLL` 30.4 % |
| verdict | pass (axis score 0.83) | pass (axis score 0.79) |

Both models pass, but neither has much margin. Close to 40 % of their decision mass sits on
features an attacker can set or append to. The EMBER2018 model's top five by mean |SHAP| are
`section.num_rx` (append_only, 4.9 %), `datadirectories.CERTIFICATE_TABLE.size` (controllable,
4.8 %), `header.coff.characteristics_hashed[0]` (fixed, 3.0 %),
`header.optional.subsystem_hashed[8]` (fixed, 2.2 %) and
`section.entry_characteristics_hashed[37]` (append_only, 1.9 %). `header.coff.timestamp`
(controllable, 1.9 %) is 7th. The EMBER2024 model's top five are
`header.coff.characteristics.DLL` (fixed, 7.1 %), a pefile warning about suspicious section
flags (derived, 4.4 %), `authenticode.latest_signing_time` (controllable, 3.5 %),
`section.n_rx` (append_only, 1.9 %) and `authenticode.chain_max_depth` (append_only, 1.5 %).
Split gain overstates the single DLL flag on the EMBER2024 model (30.4 % of gain, but 7.1 % of
|SHAP|). This is why M7 gates on SHAP when it is available.

## Skip / degrade paths

| situation | behaviour |
|---|---|
| trees + corpus | TreeSHAP + gain |
| corpus only (model hides its trees, or the runner disabled tree access after a fidelity mismatch) | model-agnostic permutation SHAP on a small sample |
| trees only (no corpus) | gain-only shares, with a note |
| LightGBM model with categorical splits | the loader rewrites them as threshold chains; TreeSHAP stays exact (verified against `pred_contrib`), and the count is recorded in `details.attribution.categorical_splits_expanded` |
| slow model on the model-agnostic path | stops early near the deadline with a note, instead of running into the module timeout |
| `shap` not importable | gain-only shares if trees are available, with a note |
| neither `tree_access` nor `feature_space` | the runner skips the module (`requires_any`) |

## Interpreting results

A high `controllable_share` means that much of the decision rests on features a malware author can
change without touching the payload. Retraining with those features removed, capped, or made
monotone usually costs little accuracy and removes a whole class of trivial bypasses. A high
`top_feature_share`, or a low `effective_n_features`, means the model depends on one or a few
features. Check whether that feature is a data artifact, such as a timestamp or a size that
separates your benign and malicious sources rather than behaviours.
