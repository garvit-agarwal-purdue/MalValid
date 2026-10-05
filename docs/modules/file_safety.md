# M0 — Model file safety (`file_safety`)

**Question:** is it safe to even *open* the model file you submitted?

Many model formats are Python pickles (`.pkl`, `.joblib`, `.pt`, `.sav`, …). Loading a pickle can run
arbitrary code, and a detector artifact that ran code on load would be a supply-chain problem in
production long before its accuracy matters. M0 therefore runs **first, on every run, before anything
is deserialized**. A dangerous artifact aborts the run: the model is never loaded, every other module
is marked `skipped`, and the verdict is `blocked`.

| | |
|---|---|
| Requires | nothing (runs on the adapter's declarations only; the model is not loaded yet) |
| Default gate | `hard` — and the runner always runs it, even if disabled in the config |
| Scored | unscored in the headline verdict (M0 blocks instead); `score` is still set: 1.0 clean, 0.75 pickle accepted with `--allow-pickle`, 0.0 blocking |
| Source | `src/malvalid/modules/file_safety.py` |

## What is scanned

* The artifacts the adapter declares with `model_path` / `model_paths` (relative to the adapter file;
  `--model PATH` overrides them). Directories are expanded.
* With `scan_adapter_dir: true` (default), every other *model-like* file in the adapter's directory
  (up to 3 levels; `.git`, `__pycache__`, virtualenvs etc. are skipped). A stray `backup.pkl` next to
  the adapter is scanned too: the adapter could load it even though it is not declared.
* If the adapter declares nothing, M0 scans the model-like files it finds and adds a note asking you to
  declare `model_path` so M0 scans exactly the file `load()` reads.

## How

Nothing is ever unpickled, imported or executed. Per artifact M0 records path, sha256, size, the
content-sniffed format (not just the extension), whether it is pickle-based, and:

1. **modelscan** (ProtectAI, Apache-2.0; Python API `modelscan.modelscan.ModelScan(...).scan(path)`,
   the same scanners the `modelscan` CLI's JSON report uses). Pickle, joblib, NumPy, PyTorch zip,
   Keras/H5, SavedModel and zip containers are scanned. Artifacts whose extension does not match
   their content (e.g. a pickle saved as `.bin`, a torch zip saved as `.model`) are handed to
   modelscan under the right extension via a symlink. Formats modelscan has no scanner for (LightGBM
   text, XGBoost JSON/UBJ, ONNX) are recorded as `scanned: false` with the reason — these are data
   formats that do not execute code on load.
2. **MalValid's static opcode scan** (defence in depth), using `pickletools.genops` over every pickle
   stream — plain pickles, each pickle inside a zip (e.g. `archive/data.pkl`), zlib/gzip/bz2/xz
   compressed joblib files (streamed decompression with a size cap), and NumPy object arrays. It
   resolves `GLOBAL`, `INST` and `STACK_GLOBAL` (tracking the string stack and memo), and a byte sweep
   picks up globals after opcodes that `genops` cannot parse (joblib embeds raw array buffers).
   Every imported global is classified:

   | severity | what | examples |
   |---|---|---|
   | CRITICAL | code execution, process, file-system or deserialization gadgets | `builtins.eval/exec/getattr/__import__`, `os.*`, `posix.*`, `subprocess.*`, `sys.*`, `pickle.*`, `marshal`, `ctypes`, `importlib`, `socket`, `shutil`, `operator.attrgetter`, `numpy.load`, `*.system`, `*.popen*` |
   | HIGH | network access; unresolvable (obfuscated) globals | `urllib.*`, `http.*`, `requests.*` |
   | MEDIUM | anything outside the allowlist (a custom class's `__reduce__` could run code) | `mylab.models.Detector` |
   | allowlisted | plain data and model/numeric libraries | `numpy.*`, `scipy.*`, `sklearn.*`, `lightgbm.*`, `xgboost.*`, `pandas.*`, `collections.*`, `builtins.dict/list/...`, `copyreg._reconstructor` |

Findings from both scanners are merged per (module, operator); each finding lists which scanner(s)
reported it.

## Gate

| check | rule |
|---|---|
| `critical_findings` | number of CRITICAL findings `== 0` — any one **aborts** the run |
| `findings_at_or_above_<fail_on>` | findings with severity ≥ `fail_on` `== 0` (omitted when `fail_on: CRITICAL`) |
| `pickle_without_allow_pickle` | pickle-based declared artifacts without `--allow-pickle` `== 0` — **aborts** the run |
| `artifacts_scanned` | at least one artifact was found (not evaluated when there is none) |

`details.abort = true` on any CRITICAL finding or a pickle without `--allow-pickle`; the finding says
why and how to re-export (LightGBM `booster.save_model("model.txt")`, XGBoost
`save_model("model.json")`/`.ubj`, scikit-learn → ONNX via `skl2onnx`).

With `--allow-pickle` (`runtime.allow_pickle: true`) a scanned-clean pickle is accepted, but M0 adds a
loud `PICKLE ACCEPTED` note and its status is at least `warn`. Even then, the sandbox worker refuses
CRITICAL/HIGH globals at unpickling time (see `docs/modules/sandbox.md`).

## Parameters

| key | default | meaning |
|---|---|---|
| `fail_on` | `"HIGH"` | lowest severity that fails the gate (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) |
| `scan_adapter_dir` | `true` | also scan undeclared model-like files in the adapter's directory |

## Inputs (from the runner)

`ctx.extras["artifact_paths"]` (declared/overridden artifacts), `ctx.extras["adapter_path"]`,
`ctx.extras["allow_pickle"]`; `ctx.model` is an unloaded placeholder (only `.declarations` is used).

## Output

* **Metrics:** `n_artifacts, n_pickle, n_critical, n_high, n_medium, n_low, n_policy_violations,
  n_modelscan_scanned, scanner, scanner_version, allow_pickle, fail_on`.
* **Details:** `abort`, `abort_reasons`, per-artifact records (`path, sha256, size, format,
  is_pickle, container, declared, scanned, scan_reason, scanned_as, opcode_scan{globals…},
  findings[{severity, description, sources, module, operator}], policy_violation, notes`),
  `discovered_files`, `scan_duration_s`.
* **Tables:** `file_safety.artifacts` (one row per artifact) and `file_safety.findings`.
* **Notes:** pickle accepted, modelscan unavailable/unsupported formats, and pickle-loading calls
  (`pickle.load`, `joblib.load`, `torch.load`, …) found in the adapter source when `--allow-pickle`
  is not given.

## Limits

Static scanning cannot prove a pickle is safe: an allowlisted class can still be abused in ways no
allowlist anticipates, and M0 does not look inside non-pickle formats for anything but their format.
That is why pickles are refused by default and why the model is only ever loaded inside the sandbox.
