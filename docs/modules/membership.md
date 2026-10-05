# M4 — Membership inference (`membership_inf`)

**Question:** if your detector is deployed where others can query it, can they tell which files were
in its training set?

Membership leakage matters for a malware detector for two reasons. It is an OPSEC and
competitive-intelligence problem: it tells a malware author whether their sample was collected and
trained on, and it tells a competitor what you had. And if your training data included customer
telemetry or customer-submitted files, it is a privacy and contractual problem, because it can
confirm that a specific customer's file was in your data.

| | |
|---|---|
| Requires | `feature_space` (canonical corpus in the model's feature version) **and** `training_hashes` (the adapter's `training_hashes_path`) |
| Default gate | `warn` |
| Access used | black-box `predict_proba` only (`ctx.score`) |
| Source | `src/malvalid/modules/membership.py` |

## Parameters

| key | default | meaning |
|---|---|---|
| `max_advantage` | `0.10` | gate: worst-case membership advantage must be ≤ this |
| `max_per_side` | `5000` | cap on member/non-member pairs (subsampling is recorded in notes and `details.sampling`) |
| `min_members` | `200` | skip if fewer training-manifest members are in the corpus (or fewer non-members can be paired) |
| `test_fraction` | `0.5` | fraction of pairs held out to evaluate the attacks (the rest trains them) |
| `time_matched` | `true` | match non-members to the members' months and labels |

## Method

1. **Members** are labeled corpus rows whose sha256 is in the training manifest. Unlabeled rows are
   never used.
2. **Non-members** are labeled rows in the corpus's `pool` role that are *not* in the manifest.
   Rows in the `challenge` role are left out, because they were picked for being atypical and would
   make "unseen" look different from "trained on" for reasons that have nothing to do with
   memorization.
3. **Pairing (`time_matched: true`).** Members and non-members are stratified by (calendar month,
   label). In each stratum, the same number of rows is drawn from each side, and each member is
   paired with a non-member from the same stratum. Members with no partner are dropped, and the
   notes record how many. If fewer than `min_members` pairs can be matched this way (typically
   because you trained on *every* sample from the months the corpus covers), the module falls back
   to **nearest-time** non-members. For each label it takes the members closest in time to the
   non-members and the non-members closest in time to those members. The result is flagged
   `advantage_is_upper_bound: true`, and a note starting with `UPPER BOUND` explains that concept
   drift between time periods inflates the advantage. With `time_matched: false`, or when the
   corpus has no timestamps, pairs are matched by label only, and a caveat says so.
4. Pairs above `max_per_side` are subsampled (uniformly for time-matched and label-matched pairs;
   on the nearest-time path the cap keeps the pairs closest in time). The note and
   `details.sampling.pairs_before_cap` record how many pairs were available. The pairs are then split into
   attack-train and attack-test sets according to `test_fraction`, so both halves keep the matched
   structure.
5. Every paired row is scored once, in large batches, through `ctx.score`.
6. **Attacks.** Each attack is fit on the attack-train half and evaluated on the held-out half.
   * `loss_threshold`: the loss/confidence-threshold attack (Yeom et al., 2018). The attack score
     is log P(true label), so a lower cross-entropy loss looks more like a member. The threshold
     that maximizes TPR − FPR is chosen on the attack-train half.
   * `art_rf`, `art_gb`: ART 1.20's `MembershipInferenceBlackBox` with `attack_model_type="rf"` and
     `"gb"`, using `input_type="prediction"` (two-class probabilities plus the one-hot true label).
     The target is an ART `BlackBoxClassifier(predict_fn=ctx.score → (n, 2) [1−p, p],
     input_shape=(dim,), nb_classes=2)`. Predictions are passed to `fit`/`infer` precomputed
     (`pred=` / `test_pred=`). ART therefore does not re-query the model in its default 128-row
     batches, and the sandbox only ever sees large batches. The attack models are seeded
     (`random_state`) from the module seed. The random forest predicts single-threaded to keep
     results bit-for-bit reproducible.
7. **Per attack** (on the held-out pairs):
   * `advantage`: TPR − FPR at the attack's own decision threshold. For the loss attack that is the
     threshold chosen on the train half; for ART it is a member probability > 0.5, which is ART's
     rounding rule.
   * `best_advantage`: max TPR − FPR over all thresholds on the held-out pairs. This is optimistic
     for the attacker.
   * `auroc` and `tpr_at_fpr_0.01`.
8. **Gate:** the worst case, meaning the largest `advantage` over the attacks. If one ART attack
   crashes, it is excluded, recorded in `details.attack_errors` and noted. The other attacks still
   count.

## Checks and score

`advantage <= max_advantage`, graded with `ideal = t/10` and `floor = min(1, 3t)`. At the default
`t = 0.10` that gives ideal 0.01 and floor 0.30. The score is 1.0 at or below the ideal, 0.75 at the
threshold and 0.0 at or beyond the floor, interpolated linearly in between (`malvalid.scoring`).

## Metrics

`advantage`, `worst_attack`, `advantage_is_upper_bound`, `sampling_design`
(`time_matched | nearest_time | label_matched`), `time_matched`, `n_manifest_hashes`,
`n_members_in_corpus`, `n_members_used`, `n_nonmembers_used`, `n_attack_train_pairs`,
`n_attack_test_pairs`, `member_accuracy`, `nonmember_accuracy`, `accuracy_gap` (both at the
operating threshold), `max_auroc`, `max_best_advantage`, `max_tpr_at_fpr_0.01`, and for each attack
`<attack>_{advantage,best_advantage,auroc,tpr_at_fpr_0.01}`.

`details.attacks` holds the per-attack numbers, including TPR and FPR at the decision threshold.
`details.sampling` holds the design, strata count, pairs before the cap, member and non-member time
ranges and label counts, the excluded challenge rows and (for nearest-time) the median time gap
between paired rows.

## Charts

* `attack_roc`: ROC of each attack on the held-out pairs, with the diagonal and a 1 % FPR marker.
* `member_score_hist`: the distribution of the model's cross-entropy loss on the true label (log10)
  for members vs non-members, with the loss attack's threshold marked. Visible separation between
  the two distributions is memorization.

## Skip paths (never reported as a pass)

* No training manifest: "no training manifest (training_hashes_path not declared)". The runner skips
  on the missing `training_hashes` requirement, and the module checks again defensively.
* Too few members: "training manifest has N members in the canonical corpus (need at least
  min_members=…)".
* Too few non-members: "only N labeled non-member rows could be paired with the M training members…".

## Interpreting the result

Every result carries a plain-language paragraph about what the measured leakage would expose. An
advantage of 0 is chance and 1 is perfect identification. As a rough guide, ≤ 0.05 is small,
0.05–0.15 is moderate and > 0.15 is large. Leakage goes down with regularization (fewer or
shallower trees, a larger `min_data_in_leaf`, bagging or feature subsampling) and with serving
coarse verdicts instead of raw scores.

This is an empirical audit against specific attacks, not a privacy guarantee. For a formal bound,
train with differential privacy.

## On the canonical corpora

* `ember_v2_2018` with the EMBER2018 reference model (manifest = the 800k train rows, 600k labeled):
  the train split ends in 2018-10 and the test split starts in 2018-11, so there is **no time
  overlap** and M4 always takes the nearest-time, upper-bound path (October 2018 members vs November
  2018 non-members, median gap 31 days). Measured with defaults: worst-case advantage 0.178 (ART
  random forest; loss attack 0.168, ART gradient boosting 0.169), attack AUROC up to 0.615, member
  accuracy 1.000 vs non-member 0.979. The gate fails, reported as an upper bound.
* `ember_v3_2024` holds only the EMBER2024 test and challenge sets, so a model trained on the
  EMBER2024 train split has no members in it and M4 skips ("training manifest has 0 members in the
  canonical corpus").

## Runtime

With the defaults (5000 pairs), scoring 10k rows dominates. The attacks take about 1–2 s. On the
real `ember_v2_2018` corpus (1M × 2381) with the reference model, the whole module took 4–8 s on
4 threads (shared machine).
