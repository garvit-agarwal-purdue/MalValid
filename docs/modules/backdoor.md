# M5 — Backdoor / poisoning screening (`backdoor_screen`)

**Question:** are there signs that your detector was trained (for example through poisoned training
data) to let specifically marked malware through?

A backdoored malware detector behaves normally on ordinary files but calls a malicious file benign
when it carries a *trigger*, such as a particular string count, import, or header value. Label-flip
poisoning of a boosted-tree detector plants exactly this: a few trees learn a narrow rule that sends
"trigger present" to a strongly benign leaf.

> **Screening only.** M5 can find evidence of a backdoor. It cannot certify that a model has none.
> The report always carries the note *"absence of findings is NOT proof that the model is
> backdoor-free"*. Certifying backdoor-freedom needs training-time access (the training data, or a
> retrain), and this gate does not have it. The result has `screening: true`.

| | |
|---|---|
| Requires | `tree_access` **or** `feature_space` (at least one) |
| Default gate | `warn` |
| Access used | (a) black-box `predict_proba` via `ctx.score`; (b) the normalized `TreeEnsemble` (plain arrays, analysed in the trusted parent process) |
| Source | `src/malvalid/modules/backdoor.py` |

## Parameters

| key | default | meaning |
|---|---|---|
| `triggers` | `[]` | trigger hypotheses: a list of `{name, features: {feature_name_or_index: value}}` |
| `triggers_path` | `null` | YAML/JSON file with the same list (or `{triggers: [...]}`); a relative path is resolved against the working directory, then against the gate config's directory |
| `max_trigger_drop` | `0.20` | gate: the largest detection-rate drop caused by any trigger must be ≤ this |
| `n_samples` | `2000` | malicious eval rows used for the trigger test (seeded subsample; `null` = all) |
| `max_rules_reported` | `50` | number of most-suspicious root→leaf rules listed in the report |
| `low_importance_quantile` | `0.5` | features whose non-gate importance is at or below this percentile count as "otherwise low-importance" |
| `min_suspicion` | `0.5` | a rule whose suspicion is ≥ this is flagged |

Trigger features can be given by **name**, using the names from the schema's `feature_names()`
(e.g. `header.coff.timestamp`, `imports.functions_hashed[438]`, `strings.numstrings` for
`ember_v2`), or by **integer index**. A string made of digits is read as an index. If a name is
unknown, the run stops with a `ConfigError` that suggests close matches (for example `unknown
feature 'headr[5]'; did you mean 'header[5]'?`). Values must be finite numbers.

```yaml
modules:
  backdoor_screen:
    triggers:
      - name: timestamp-zero
        features: {header.coff.timestamp: 0}
      - name: import-438
        features: {"imports.functions_hashed[438]": 94, "strings.numstrings": 5000}
```

## Method

### (a) Operator trigger hypotheses (needs `feature_space`)

1. Take the malicious rows of the corpus's `eval` role, minus the training-manifest members. Keep
   one row per sha256 and draw a seeded subsample of `n_samples` rows.
2. Score them at the declared `operating_threshold`. `DR_before` is the fraction detected.
3. For each trigger, write its values into every **detected** vector and score those vectors again.
   `DR_after` is the fraction of the sample still detected. Undetected rows are left unchanged, so
   they stay undetected.
4. `drop = DR_before − DR_after`. A trigger is flagged when `drop > max_trigger_drop`, the same
   condition that fails the gate check. The table also reports `evasion_rate` (the share of
   detected rows that the trigger flips) and the mean score shift. If a trigger writes a value
   outside the range seen in the detected vectors, a note says so: a large drop there may come
   from out-of-distribution inputs rather than a planted backdoor.

If triggers are supplied but no corpus is loaded, the trigger check is recorded as *not evaluated*
(status `warn`, never a silent pass).

### (b) Static tree-structure anomaly scan (needs `tree_access`)

The scan never stamps or scores a sample. It reads the ensemble's structure: splits, thresholds,
leaf values, training cover, and split gain. Subtree means are recomputed from the leaves
(cover-weighted), because some LightGBM dumps store internal values without shrinkage.

* A **gate** branch is a split child that takes at most `ρ ≤ 0.5` of its parent's training cover
  and whose subtree mean output is lower (more benign) than its sibling's.
* `gate_share[f]` = Σ over gate branches on feature `f` of `gain × narrow(ρ) × low_cover(φ)`,
  divided by the ensemble's total split gain. `φ` is the branch's share of its tree's root cover.
  `narrow` and `low_cover` are log-ramps from 0 to 1: `narrow` is 0 at `ρ ≥ 0.5` and 1 at
  `ρ ≤ 0.02`; `low_cover` is 0 at `φ ≥ 0.25` and 1 at `φ ≤ 0.005`.
* `low_importance[f] = clip((1 − pct) / (1 − low_importance_quantile), 0, 1)`, where `pct` is the
  percentile of `f`'s **non-gate** gain among the features the ensemble uses.
* `feature_suspicion[f] = low_importance[f] × clip((gate_share[f] − 0.002) / 0.008, 0, 1)`.
* For each root→leaf **rule**:
  * `benign` measures how far the leaf sits below its tree's cover-weighted mean, relative to the
    tree's largest leaf deviation (0..1).
  * `narrowness` is the log-ramp of the product of the gate `ρ` values on the path.
  * `low_cover` is the log-ramp of the leaf's share of the tree's cover.
  * The **dominant** feature is the gate feature on the path with the highest feature suspicion.
  * `suspicion = feature_suspicion[dominant] × sqrt(benign × max(narrowness, low_cover))`.
    Rules with no gate, or with a leaf that is not benign, score 0.

The report lists the top `max_rules_reported` rules. Each entry shows the conditions (merged per
feature into readable intervals), leaf value (raw margin), cover, cover share, suspicion, and
dominant feature. It also aggregates the scores per feature.

**Calibration.** The constants were tuned on toy models from `malvalid.testing` and on LightGBM
models trained on real EMBER2018 feature vectors, with and without a planted label-flip trigger.
The EMBER2018 experiment trains two LightGBM models (300 trees, 64 leaves, 200k labeled rows from
the corpus's `train` split, seed 0). The *poisoned* model gets a trigger on a feature in the bottom
third of the clean model's split gain (`imports.functions_hashed[838]`): a value never seen in
training (63) is written into 0.5 % or 2 % of the malicious training rows, and those rows are
relabeled benign. Both models are then screened with the planted trigger supplied as a hypothesis.

| model | top rule suspicion | flagged rules | trigger drop (DR before → after) |
|---|---|---|---|
| toy, 6 % poisoned (`header[5]`) | 0.97 | 77 | large (see tests) |
| toy, clean | 0.25 | 0 | — |
| EMBER2018 LightGBM, 0.5 % poisoned (499 rows) | 0.84 | 46 | 96.4 pp (96.4 % → 0.0 %) |
| EMBER2018 LightGBM, 2 % poisoned (1998 rows) | 1.00 | 87 | 96.4 pp (96.4 % → 0.0 %) |
| EMBER2018 LightGBM, clean (same recipe) | 0.04 | 0 | 0.0 pp |
| public EMBER2018 reference model (1000 trees, ~1500 leaves each) | 0.11 | 0 | — |
| public EMBER2024 reference model (500 trees) | 0.04 | 0 | — |

Both poisoned models are flagged on the planted feature, and the clean models stay far below
`min_suspicion = 0.5`. The scan is a heuristic. A trigger spread over several features that the
model also uses heavily for ordinary decisions would score lower. This is one reason M5 is
screening only.

LightGBM categorical splits (the EMBER2024 reference model has 371) are rewritten by the loader
as chains of threshold splits. Each copy of a categorical subtree gets an even share of the
original training cover, so narrowness and cover under those splits are approximate. A note says
so, and `details.scan.categorical_splits_expanded` records the count.

The ensemble's structure is a weak signal when it has no training cover. ONNX exports, for
example, give every leaf cover 1. The scan still runs in that case, and a note says its scores are
less reliable. When the trees carry no split gains, node cover is used as the importance weight.
If no tree has usable cover at all, the scan check is recorded as not evaluated.

### ART

ART's poisoning and backdoor detectors (activation clustering, spectral signatures, provenance and
RONI defences) work on neural-network activations and/or need the training data. They do not apply
to tree ensembles, and a black-box gate does not have what they need for other model types either.
`details.art_poisoning_detectors` records them as not applicable, with the reason.

## Checks and score

| check | condition | score anchors |
|---|---|---|
| `max_trigger_drop` (only when triggers are supplied) | `max_trigger_drop ≤ max_trigger_drop` | ideal 0.02, floor 0.8 |
| `n_flagged_rules` (only when the scan ran) | `n_flagged_rules ≤ 0` | 1.0 if none flagged, else `0.75 × (1 − s) / (1 − min_suspicion)`, kept just below the 0.75 pass line, where `s` is the top rule suspicion |

The axis score is the weakest check score.

## Metrics

`n_triggers`, `scan_ran`. When the scan ran: `n_trees`, `n_trees_scanned`, `n_rules_scanned`,
`n_candidate_rules` (suspicion > 0), `n_flagged_rules`, `n_flagged_features`, `top_suspicion`,
`top_feature_suspicion`, `top_suspicious_feature`, `scan_seconds`. When triggers ran:
`max_trigger_drop`, `worst_trigger`, `n_flagged_triggers`, `dr_before`, `n_trigger_samples`,
`n_trigger_detected`.

`details`: `scan` (`top_rules`, `features`, `flagged_features`, calibration constants, score
formula), `triggers` (per-trigger results, sampling, drop definition), and
`art_poisoning_detectors`.

## Charts and tables

* `feature_suspicion` (bar): the 15 most suspicious features, with a `min_suspicion` line.
* `suspicious_features` (table) and `suspicious_rules` (table).
* `trigger_tests` (table). `trigger_drop` (bar) is added when more than one trigger is supplied or
  any trigger is flagged.

## Skip / degrade paths

| situation | behaviour |
|---|---|
| no tree access **and** no triggers | skipped: "model does not expose its tree structure … and no trigger hypotheses were supplied" |
| no tree access, triggers given | trigger test only, with a note that the scan did not run |
| tree access, no triggers | scan only, with a note explaining how to supply triggers |
| tree access, no corpus, triggers given | scan runs; the trigger check is *not evaluated* (warn) |
| neither `tree_access` nor `feature_space` | the runner skips the module (`requires_any`) |

## Results on the reference models

These runs use the full canonical corpora and the reference LightGBM models in-process
(`malvalid.testing.InProcessModel`, trees via `LightGBMLoader.tree_ensemble`, 4 OpenMP threads).
Two illustrative trigger hypotheses were supplied: `zero_timestamp`
(`header.coff.timestamp = 0`) and `few_strings` (`strings.numstrings = 5`).

| | EMBER2018 (`ember_v2_2018`, threshold 0.8336) | EMBER2024 (`ember_v3_2024`, threshold 0.5) |
|---|---|---|
| trees / root→leaf rules | 1000 / 1,479,647 | 500 / 72,423 (371 categorical splits expanded) |
| static scan time | 1.4 s | 0.4 s |
| top rule suspicion | 0.11 (`imports.functions_hashed[942]`) | 0.04 (`datadirectories.SECURITY.virtual_address`) |
| flagged rules | 0 | 0 |
| trigger sample (unique malicious eval rows) | 2000 of 100,000; DR 96.1 % | 2000 of 240,000; DR 97.1 % |
| `zero_timestamp` drop | 0.75 pp | 0.25 pp |
| `few_strings` drop | 0.15 pp (value outside the observed range, noted) | 0.0 pp |
| module wall time (after the trees are loaded) | 19.7 s | 3.0 s |
| verdict | pass (axis score 1.0) | pass (axis score 1.0) |

Neither reference model shows a backdoor signature. By design, that is not proof that they have
none. Normalizing the EMBER2018 model's trees takes about 30 s, and the runner does this once for
all modules. The scan is vectorized per tree level: a 1000-tree × 64-leaf ensemble scans in well
under a second (`test_scan_is_fast_on_large_ensemble`).
