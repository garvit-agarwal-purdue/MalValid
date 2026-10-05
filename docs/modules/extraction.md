# M6 — Model-extraction susceptibility (`extraction`)

**Question:** if your detector is exposed as a query service that returns verdicts, how many queries
does it take to clone it?

A clone lets an attacker search for evasions offline, without ever touching your service, and it
free-rides on your training data and labeling effort. This matters **only if the model will be
queryable**, for example through a cloud file-reputation or scanning API. For that reason the module
is **off by default**.

| | |
|---|---|
| Requires | `query_only` (plus a loaded canonical corpus as the attacker's query distribution) |
| Default gate | `warn` (disabled in `default_gate.yaml`; enable it with `extraction: {enabled: true}`) |
| Access used | black-box verdicts: `ctx.score(X) >= operating_threshold` |
| Source | `src/malvalid/modules/extraction.py` |

## Parameters

| key | default | meaning |
|---|---|---|
| `query_budgets` | `[100, 1000, 10000, 50000]` | query budgets to evaluate (`fidelity_budget` is always added) |
| `fidelity_budget` | `10000` | the budget at which the gate is evaluated |
| `max_fidelity` | `0.95` | gate: surrogate fidelity at `fidelity_budget` must be ≤ this |
| `surrogate` | `"lightgbm"` | the attacker's model: `lightgbm`, `random_forest` or `logistic_regression` |
| `n_eval` | `5000` | held-out eval rows used to measure fidelity and accuracy (`null` = all) |

## Method

1. **Query distribution.** The attacker queries corpus rows from the `pool` role, minus every
   `eval`-role row, so the rows used to measure fidelity are never queried for training
   (`ember_v2_2018`: the 800k train rows, labeled or not). If a corpus has **no** pool rows outside
   its eval role (`ember_v3_2024`, where pool and eval are both the EMBER2024 test set), the
   held-out eval sample is reserved first (at most half the eval rows) and the queries come from the
   remaining pool rows; `metrics.query_pool_source` is then `"eval_remainder"` instead of
   `"pool_non_eval"`, and a note says so. In both cases query rows whose sha256 also appears among
   the held-out rows are dropped (a guard for corpora with duplicate hashes), so the same file is never
   both queried and measured on. One random permutation of the pool is drawn, and budget *b* uses
   its first *b* rows. Larger budgets are therefore supersets of smaller ones, the way an attacker
   accumulates queries. Budgets larger than the pool are capped at the pool size, and the notes say
   so.
2. **Labels.** The attacker only sees **hard verdicts** at your operating threshold, the way a
   service would return them. Raw scores would make extraction cheaper.
3. **Surrogate.** For each budget, the surrogate is fit on (queried rows, verdicts). The LightGBM
   surrogate uses 200 rounds, 31 leaves, learning rate 0.1, `min_data_in_leaf = clamp(b/50, 1, 20)`
   and L2 = 1. It is deterministic and seeded from the module seed. A budget whose verdicts all fall
   in one class yields a constant surrogate.
4. **Measurement.** On up to `n_eval` held-out eval rows, from which training-manifest members are
   excluded when the manifest is known, the module reports:
   * `fidelity`: the fraction of rows where the surrogate's verdict equals your model's verdict;
   * `accuracy`: the surrogate's agreement with the true labels, on labeled rows only.
   Your model's own accuracy and the majority-class baseline fidelity are reported alongside for
   context.

### ART usage

ART 1.20 has three extraction attacks. This module uses ART where one applies and records why the
others don't (`details.art_applicability`).

* **`KnockoffNets(sampling_strategy="random")` — used.** It is exactly the query-and-fit procedure
  above. The victim is an ART `BlackBoxClassifier` whose `predict_fn` returns one-hot verdicts from
  `ctx.score`, with `batch_size_query = b` so each budget is a single batched query. None of ART's
  own estimator wrappers can be the thieved classifier for a non-neural surrogate:
  `LightGBMClassifier.fit` raises `NotImplementedError`, and its constructor rejects binary
  boosters. `SklearnClassifier.fit` forwards the neural-network-only kwargs (`batch_size`,
  `nb_epochs`, `verbose`) to the sklearn estimator, which raises `TypeError`, and it does not accept
  `LGBMClassifier` at all. The module therefore passes a minimal ART `ClassifierMixin` shim that
  fits the surrogate on the one-hot stolen labels. KnockoffNets draws its query order from the
  global NumPy RNG, so the module seeds it for the call and then restores the previous state.
* `KnockoffNets(sampling_strategy="adaptive")` — **not applicable.** It refits the thieved
  classifier after every single query, with rewards computed from its probability outputs. That is
  designed for neural networks and infeasible for a tree surrogate.
* `CopycatCNN` — **nothing beyond KnockoffNets(random).** It runs the same random query-and-fit
  procedure, specialised to CNN image classifiers.
* `FunctionallyEquivalentExtraction` — **not applicable.** The victim must be a two-layer ReLU
  network with logit outputs (ART `NeuralNetworkMixin`).

If ART cannot be imported, or KnockoffNets fails, the module falls back to the equivalent **direct
query-and-fit** procedure. In that case `metrics.method` is `"direct query-and-fit"` and the notes
say why. With ART, each budget re-queries its rows. The direct path queries the largest budget once.
`details.total_model_queries` records the exact count.

## Check and score

`fidelity <= max_fidelity` at `fidelity_budget`, graded with `ideal = 0.80` and `floor = 0.995`
(kept strictly on either side of the threshold if you configure an extreme `max_fidelity`).

If the query pool is **smaller than `fidelity_budget`**, the module measures fidelity at the pool
size. Fidelity grows with the budget, so that measurement is a lower bound. If it already exceeds
`max_fidelity`, the check fails. Otherwise the check is **not evaluated**: the status is `warn`,
never a pass.

## Metrics

`fidelity`, `fidelity_budget`, `fidelity_budget_effective`, `accuracy_at_fidelity_budget`,
`model_accuracy`, `majority_baseline_fidelity`, `max_fidelity_observed`,
`queries_to_exceed_max_fidelity` (the smallest evaluated budget whose fidelity is above the limit,
or `null`), `budgets`, `fidelities`, `accuracies` (parallel lists), `surrogate`, `method`, `n_eval`,
`n_query_pool`, `query_pool_source`.

`details.budgets` holds the per-budget rows (budget, fidelity, accuracy, seconds);
`details.query_pool` records where the queries came from (pool/eval splits, pool size, duplicate
hashes excluded).

## Chart

* `fidelity_vs_queries`: fidelity and surrogate accuracy against the number of queries, on a log
  x-axis, with reference lines at `max_fidelity`, at `fidelity_budget` and at your model's accuracy.

## Skip paths

* No corpus loaded: "no query distribution: canonical corpus unavailable".
* Corpus in a different feature version from the model: "no query distribution: …".
* No eval rows to hold out: "no held-out eval rows in the canonical corpus to measure fidelity on".
* Nothing left to query after reserving the held-out rows: "no query distribution: …".

## Reading the result

High fidelity at a small budget means the decision function is easy to copy from verdicts alone.
Tree ensembles on EMBER-style features are typically very clonable: a good detector is also a
low-complexity function of a few strong features. The usual mitigations are operational (rate
limits, query auditing, coarse or delayed verdicts, and never returning raw scores) rather than
changes to the model.

The threat model is deliberately simple: random queries from the corpus distribution. An attacker
with smarter query selection, or with access to scores, would do at least as well, so treat the
measured fidelity as a lower bound on what an attacker could achieve.

## Runtime

Measured on the real canonical corpora with the reference LightGBM models, default parameters and
4 threads (shared machine, so timings are noisy):

| corpus | query pool | fidelity at 100 / 1k / 10k / 50k queries | wall time |
|---|---|---|---|
| `ember_v2_2018` + `ember_model_2018.txt` (threshold 0.8336) | 800k train rows | 0.750 / 0.869 / 0.934 / 0.957 (pass at 10k) | 71 s |
| `ember_v3_2024` + `EMBER2024_PE.model` (threshold 0.5) | 475k test rows not held out | 0.860 / 0.930 / 0.966 / 0.983 (fail at 10k) | 44 s |

The 10k and 50k surrogate fits dominate (roughly 12 s and 27–31 s). Measured 2026-09-29 on the corpora as then built (`ember_v2_2018`: 1,000,000 rows; `ember_v3_2024`: 484,855 rows). Memory peaks at roughly
3 × (50k × dim × 4 B): the query matrix, KnockoffNets' copy of it and the LightGBM dataset.
