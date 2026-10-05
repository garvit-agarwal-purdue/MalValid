# Runner, CLI and `report.json`

`malvalid run` takes one trained malware detector (through a small adapter), runs the test battery
against it, and returns a **production-readiness verdict**: `READY`, `CONDITIONAL`, `NOT_READY` or
`BLOCKED`, with a 0–100 score and the per-axis evidence behind it. This page covers how a run is
put together, what ends up in the run directory, and what each exit code means.

| | |
|---|---|
| Source | `src/malvalid/runner.py` (`run_gate`), `cli.py`, `manifest.py`, `environment.py`, `report/json_writer.py`, `__main__.py` |
| Entry points | `malvalid …` (console script) and `python -m malvalid …` |
| Tests | `tests/unit/test_runner.py`, `test_cli.py`, `test_manifest.py`, `test_verdict.py` (fakes: `test_runner_support.py`); integration: `test_runner_integration.py` (real modules, faked sandbox), `test_runner_sandbox.py` (`slow`: real sandbox, nothing faked) |

## Quick start

```bash
malvalid init-config gate.yaml                      # the default gate policy, ready to edit
malvalid validate-adapter --adapter my_adapter.py   # check the submission contract first
malvalid run --adapter my_adapter.py --config gate.yaml --out runs/candidate-7
```

The terminal summary shows the verdict and score, the coverage, a per-axis scorecard, the
blockers and the reasons the model is not `READY`, any modules the config disabled, then the
paths of `report.json`, `report.html` and `run.log`.

## Pipeline (`runner.run_gate`)

1. **Validate the config** against the plugin registry. Unknown modules or parameters are errors,
   because a typo in a gate should never quietly change policy.
2. **Inspect the adapter** in the sandbox (`sandbox.host.inspect_adapter`). This reads the
   declarations only. `load()` is **not** called.
3. **Parse the training manifest and cutoff** (`manifest.py`, see below). Resolve the feature schema
   from `feature_version`. An unknown feature version stops the run with an `AdapterError`.
4. **Load and verify the canonical corpus** (`cfg.corpus`, content hashes checked). If the corpus is
   missing, fails verification, or uses a different feature space from the model, the run carries
   on: the problem is recorded in `corpus.error`, and every module that needs the feature space is
   skipped with that error as its reason.
5. **M0 file safety always runs first**, against an unloaded placeholder model. The model artifact
   is scanned before it is ever deserialized. M0 runs even when it is disabled in the config or
   listed in `--skip` (with a warning), and it is always a hard gate. If M0 aborts (a CRITICAL
   finding, or a pickle without `--allow-pickle`), or M0 itself errors, no other module runs:
   each one is recorded as `skipped` with "run aborted by M0: …", and the verdict is `BLOCKED`.
6. **Load the model** in the sandbox (`open_model`, which calls the adapter's `load()`). If loading
   fails, every module is skipped with "model failed to load: …", the verdict is `BLOCKED`, and the
   exit code is 2.
7. **Verify tree access.** When the model exposes a tree ensemble, the runner checks that it
   reproduces `predict_proba`. The check uses up to 2,000 eval rows from the corpus, or, with no
   usable corpus, 512 random vectors placed on both sides of the model's own split thresholds. If
   max |Δ| is above 1e-4, tree access is turned off and a note is added. Tree-based analyses must
   describe the model you will deploy, not an approximation of it. The result is recorded under
   `model.tree_access`.
8. **Run each selected module** in scorecard order (M1, M2, …):
   * If a requirement is not met, the module is `skipped` with the specific reason: the missing
     training manifest, the corpus error, the failed fidelity check, and so on. For a module with
     `requires_any`, the reason names every alternative.
   * Otherwise the module gets a fresh `RunContext`:
     * seed `module_seed(runtime.seed, id)`;
     * deadline now + `runtime.max_seconds_per_module`, also passed to the sandboxed model;
     * a `SIGALRM` timer that raises `ModuleTimeout` in the main thread.
   * If the module raises, it is recorded as `error`, with the traceback in `error` and the finding
     "module crashed: …".
   * A timeout is also an `error` ("exceeded N s"), including when the module swallowed the
     exception. After a hard timeout or a sandbox crash, the model worker is restarted.
   * `duration_s` is recorded for every module.
9. **Compute the verdict and exit code** with the frozen `verdict.compute_verdict` /
   `exit_code_for`. Then write `report.json`, `report.html` (if that fails, the JSON is still
   written and a warning is added) and `run.log`.

Adapter, config and usage problems that make a meaningful run impossible are raised as
`MalValidError`s: `ConfigError`, `AdapterError`, `SandboxError`. The CLI turns them into a
one-line message and exit code 2. Anything that happens after the adapter has been inspected goes
into `report.json` instead.

## Selecting modules

* A module runs when it is `enabled: true` in the gate policy.
* `--only M1,drift` runs exactly those modules, including ones disabled in the config. Modules can
  be named by id or by scorecard code, case-insensitively, and the flag is repeatable.
* `--skip` removes modules from the selection.
* M0 always runs.
* A module that is disabled or not selected does not appear in `modules`. The ones named in the
  config are listed under `gate.disabled`.

## Training manifest (`manifest.py`)

* **`training_hashes_path`** can be either:
  * a text file with one sha256 per line (case-insensitive; blank lines and `#` comments are
    ignored; for a headerless CSV, the first field is used); or
  * a CSV/TSV/semicolon file with a header column named `sha256` (also accepted: `sha-256`,
    `sha_256`, `hash`, `sha256_hash`).

  A `.gz` suffix is decompressed. A malformed line is an `AdapterError` that names the line. An
  empty manifest gives a warning and satisfies nothing.
* **`training_cutoff`** can be:
  * `YYYY-MM-DD`;
  * `YYYY-MM`, meaning the last day of that month;
  * `YYYY`, meaning 31 December;
  * an ISO datetime, which is truncated to its date.
* **Training-member exclusion.** Rows in the corpus's `eval` role whose hash is in the manifest
  are excluded from evaluation by the modules. `corpus.excluded_training_members` records how many
  there were. `training_manifest.n_in_corpus` counts manifest hashes found anywhere in the corpus.

## Run directory

```
<out>/report.json    the auditable record (schema "malvalid-report/1")
<out>/report.html    self-contained human report (unless --no-html / report.html: false)
<out>/run.log        full log of the run, including the reason a run failed
<out>/private/       sensitive module outputs and the sandbox scratch dir, never referenced by the report
```

## `report.json`

The report is strict JSON: `json.dumps(report, allow_nan=False)` always succeeds, because
non-finite numbers become `null`. Before writing, every string that mentions the run's
`private/` directory is replaced with `<private run data withheld>`, as spec §11 requires.

| key | content |
|---|---|
| `schema_version` | `"malvalid-report/1"` |
| `tool` | `{name, version}` |
| `run` | `id, started_at, finished_at` (UTC, `Z`), `duration_s, command, seed, out_dir, report_json, report_html, run_log, only, skip, allow_pickle` |
| `verdict` | `VerdictResult.to_dict()`: `verdict, label, score, raw_score, capped, coverage, reasons, blockers, axes, bands, summary`. The runner rewrites only `label`/`summary` so a BLOCKED headline names its real cause (a failed hard gate, an errored module, an M0 scan that did not complete, a model that failed to load, an unevaluated hard gate) and the summary shows the score to one decimal, as the bands compare it |
| `gate` | `exit_code, exit_meaning, passed, fail_on, hard_gates[{module_id, code, title, status, gate_outcome}], failed_hard, not_evaluated_hard, errored, skipped, aborted, abort_reason, load_error, disabled` |
| `model` | adapter declarations + `artifacts[{path, sha256, size, format, is_pickle, declared}]` (`declared: false` = a model file M0 found in the adapter directory because the adapter declares no `model_path`), `adapter_sha256`, `tree_access{available, n_trees, fidelity_max_abs_diff, n_rows_checked, rows_source, tolerance, note}`, `load_error`, `query_count` |
| `corpus` | `Corpus.summary()` (name, version, feature_version, content_hash, n, dim, splits, roles, time_range, source) + `available, used_for_evaluation, error, excluded_training_members, provider, location` |
| `schema` | `name, dim, description, featurize_available, featurize_source, schema_extractor_available` |
| `training_manifest` | `path, declared, n_hashes, n_in_corpus, cutoff, cutoff_parsed, warnings` |
| `capabilities` | the requirements satisfied in this run (e.g. `feature_space`, `tree_access`) |
| `config` | `GateConfig.to_dict()`: the exact policy applied |
| `environment` | `python, platform, cpu_count, …, libraries{numpy scipy sklearn lightgbm xgboost shap art modelscan onnx onnxruntime tesseract lief pefile jinja2 pydantic pyyaml typer rich}`, threading env vars |
| `sandbox` | `SandboxedModel.sandbox_info()` (backend, network isolation, limits) |
| `modules` | `ModuleResult.to_dict()` list: M0 first, then scorecard order |
| `artifacts` | charts and tables (`ArtifactStore.to_dict()`) |
| `warnings`, `disclaimers` | run-level caveats (e.g. sandbox disabled, synthetic corpus, no evasion-robustness claim) |

## Exit codes (`malvalid run`)

| code | meaning |
|---|---|
| 0 | no hard gate failed, no module errored, and the verdict is above `--fail-on` |
| 1 | the run is `BLOCKED` (a hard gate failed or could not be evaluated, or M0 aborted), or the verdict is at or below `--fail-on` (`blocked` (default), `not_ready`, `conditional`) |
| 2 | a module errored or the model could not be loaded (the gate result cannot be trusted), **or** a usage, config or adapter error (one-line message; add `-v` for the traceback) |

## Commands

| command | purpose |
|---|---|
| `malvalid run --adapter A [--config C] [--out DIR] [--allow-pickle] [--model PATH]… [--class NAME] [--only ids] [--skip ids] [--corpus NAME] [--corpus-dir D] [--seed N] [--no-sandbox] [--no-html] [--fail-on L] [-v]` | evaluate a detector |
| `malvalid list-modules [--json]` | the test battery, its requirements, default gates and verdict weights |
| `malvalid init-config [PATH=gate.yaml] [--force]` | write the default gate policy |
| `malvalid validate-adapter --adapter A [--config C] [--allow-pickle] [--model P]… [--class NAME] [--json]` | check the submission contract (exit 0 ok, 1 a check failed, 2 could not run) |
| `malvalid corpus list \| info NAME \| verify NAME \| build NAME --source DIR [--out DIR] [--workers N]` | canonical corpora. `verify` re-hashes every file (exit 1 on mismatch). `build` delegates to the provider's `build()` |
| `malvalid report render REPORT_JSON [-o OUT]` | re-render `report.html` from a `report.json` |
| `malvalid sandbox-check [--json]` | which isolation backends work here (exit 1 if none isolates the network) |
| `malvalid --version` | version |

`--corpus-dir` accepts either the corpus directory itself or a root that contains `<corpus>/`.

## Testing seams

The runner reaches other components only through small module-level functions:
`_make_policy`, `_inspect_adapter`, `_open_model` (sandbox), `_file_safety_cls` (M0) and
`_write_html` (HTML report). Tests monkeypatch these with `malvalid.testing.InProcessModel`-based
fakes on the `toy_v1` schema and the `toy_v1_corpus` provider. See `tests/unit/test_runner_support.py`.
