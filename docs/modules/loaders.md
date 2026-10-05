# Model loaders (`lightgbm`, `xgboost`, `sklearn_gbdt`, `onnx`)

**What they do:** a loader turns your detector's saved artifact into a live model *inside the
sandboxed worker*, scores it (`predict_proba(native, X) -> (n,)` malicious-class probability) and,
for tree ensembles, exports the trees as a framework-independent
[`TreeEnsemble`](../../src/malvalid/loaders/trees.py). The trusted parent process only ever
receives plain numbers from `TreeEnsemble.to_bytes()`, never your deserialized model object.
M5 (backdoor screen) and M7 (explanation) use those trees. When a model cannot expose them, the
loader raises `UnsupportedTreeError` and those modules fall back to model-agnostic methods.

| `model_kind` | safe formats (preferred) | pickle formats (`--allow-pickle` only) | tree access | code |
|---|---|---|---|---|
| `lightgbm` | `.txt`, `.model`, `.lgb` (LightGBM text model) | pickled `Booster` / `LGBMClassifier` | yes: numeric **and categorical** splits, NaN, zero-as-missing | `src/malvalid/loaders/lightgbm_loader.py` |
| `xgboost` | `.json`, `.ubj` (also `.model` / `.bst` / `.xgb` holding JSON/UBJSON) | pickled `Booster` / `XGBClassifier` | yes: `gbtree` and `dart`, NaN via `default_left` | `src/malvalid/loaders/xgboost_loader.py` |
| `sklearn_gbdt` | none (scikit-learn models only exist as pickles) | `.pkl`, `.pickle`, `.joblib`, `.jbl`, `.sav` | yes: (Hist)GradientBoosting, RandomForest, ExtraTrees, DecisionTree (binary) | `src/malvalid/loaders/sklearn_loader.py` |
| `onnx` | `.onnx` (scored by onnxruntime, CPU provider, no custom ops) | never unpickles | yes, for one `ai.onnx.ml` `TreeEnsembleClassifier`/`Regressor` node | `src/malvalid/loaders/onnx_loader.py` |

All four are registered under the `malvalid.model_loaders` entry point (see `pyproject.toml`);
third parties can add more by subclassing `malvalid.loaders.base.ModelLoader`.

## Getting a safe artifact

Pickles can execute arbitrary code when they are loaded. MalValid refuses them unless you pass
`--allow-pickle`. Even then it logs a loud warning, and M0 marks the run at least `warn`. Export
in a safe format instead:

```python
booster.save_model("model.txt")            # LightGBM (clf.booster_.save_model for LGBMClassifier)
booster.save_model("model.json")           # XGBoost (or "model.ubj"; clf.get_booster() for XGBClassifier)
skl2onnx.to_onnx(est, X[:1], target_opset={"": 17, "ai.onnx.ml": 3},
                 options={id(est): {"zipmap": False}})       # scikit-learn -> ONNX
```

LightGBM cannot read back its own `dump_model()` JSON, so a `.json` file is always treated as an
XGBoost model. Save LightGBM models with `save_model(...)`.

## Pickle policy

* `load(path, allow_pickle=False)` calls `ModelLoader.check_pickle_policy` **before anything is
  deserialized**. It refuses any artifact that is pickle-based by extension (`PICKLE_EXTENSIONS`)
  or by content (`sniff_format`: plain pickle, joblib zlib/gzip/bz2/xz containers, zip-wrapped
  pickles), so a pickle renamed to `model.txt` or `model.json` is caught too. The error message
  explains how to re-export.
* `sklearn_gbdt` treats every artifact as a pickle. `onnx` never unpickles, not even with
  `--allow-pickle`.
* Routing (`can_load`) never unpickles either. For pickles it reads the opcode stream statically
  (`pickletools.genops`, with joblib containers decompressed up to 1 MiB) to find which framework
  the pickled object comes from. A pickled `LGBMClassifier` therefore routes to `lightgbm` and a
  pickled `GradientBoostingClassifier` to `sklearn_gbdt`. A `.model` file routes by content: a
  `tree` header means LightGBM, a `learner` key means XGBoost.
* Inside the sandbox, `malvalid.loaders.base.load_model(path)` defers to `MALVALID_ALLOW_PICKLE`,
  which the worker sets from `--allow-pickle`.

## Using the loaders from an adapter

```python
from malvalid.adapter import BaseDetector

class MyDetector(BaseDetector):
    feature_version = "ember_v3"
    model_kind = "lightgbm"
    operating_threshold = 0.5
    model_path = "model.txt"          # relative to the adapter file; M0 scans it before load()
```

`BaseDetector.load()` loads `model_path` with the `model_kind` loader and sets `native_model`.
Its `predict_proba` then scores through the same loader. If you write your own `load()`, use
`malvalid.adapter.load_model(path)` (which resolves paths relative to the adapter) and keep the
result in `self.native_model`. Setting `native_model` lets the runner find the matching loader
(`find_loader_for_native`) and extract the trees. The runner then checks that the trees reproduce
your adapter's `predict_proba` (max |Δ| ≤ 1e-4 on up to 2000 corpus rows) before M5/M7 use them.

For ONNX, `load_model` returns a `malvalid.loaders.onnx_loader.ONNXModel`. `native_model` may
also be an `onnxruntime.InferenceSession`, an `onnx.ModelProto` or the `.onnx` path. For serialized
bytes, wrap them first with `ONNXModel(data)`. Output selection is automatic: an output named `*prob*`, else a ZipMap
`seq(map)` output, else the first float tensor. The malicious class is the label `1` / `"1"` /
`True` / `"malicious"` / `"malware"`. If none of those is present, MalValid uses the last sorted
class (scikit-learn's `classes_[1]`) and logs a warning. Override either choice with
`ONNXModel(path, output="probabilities", positive_class="evil")`. An output outside [0, 1]
(logits or raw scores) is an error; it is never silently taken as a probability.

Threads: LightGBM, XGBoost and onnxruntime use the sandbox's thread budget (`MALVALID_THREADS`,
else `OMP_NUM_THREADS`). onnxruntime ignores `OMP_NUM_THREADS`, so the session's intra-op thread
count is set explicitly.

## Tree access: how each framework is normalized

`TreeEnsemble` semantics: a row goes left iff `x[feature] <= threshold`, NaN goes to
`children_default`, and leaf values include shrinkage. `proba = sigmoid(base_score + Σ leaves)`
for logistic models, and a clipped mean for forests. Every conversion below is tested to match
the framework's own probabilities to ≤ 1e-6 on rows sitting exactly on, and one float32 ulp either
side of, every split threshold, with NaNs. SHAP `TreeExplainer` on `to_shap_model()` is additive
to ≤ 1e-6 in margin space.

**LightGBM.** `Booster.dump_model()` goes through the core converter `from_lightgbm_dump`, then
two fixes make it exact under LightGBM's input handling:
* *Zero band.* LightGBM treats |x| ≤ 1e-35f as 0 before comparing, so thresholds inside that
  band are moved to its edge.
* *`missing_type=None`.* LightGBM compares NaN as 0 on such nodes, so they are rewritten as
  NaN-default nodes. This is also what SHAP needs.

**Categorical splits** (`decision_type "=="`, used by the EMBER2024 reference model) send a row
left iff `int(x)` (truncated toward zero) is a non-negative member of the category set; NaN goes
right. The loader rewrites each one as a balanced chain of `<=` splits over the intervals where
that test is constant (category 0 is `(-1, 1)`, category c is `[c, c+1)`), with a copy of the
subtree under each interval. The result is checked against `predict(raw_score=True)`. Copies share
the original cover and gain evenly, so total gain/cover importance and the SHAP expected value are
unchanged. Split *counts* (`feature_importance("split")`) are inflated. `meta` records
`categorical_splits_expanded` and `nodes_after_expansion`. Linear-leaf (`linear_tree`),
multi-class and non-binary/non-regression objectives raise `UnsupportedTreeError`. A fitted
`LGBMClassifier` with early stopping uses its best iteration.

**XGBoost.** Trees are parsed from `Booster.save_raw("json")`, which holds the same content as
`trees_to_dataframe()` but stores floats exactly. The tests cross-check the two.
* XGBoost splits are `x < t` in float32. They become `x <= nextafter(float32(t), -inf)`
  (`strict_lt_to_le`), which is exact for float32 feature vectors (all MalValid corpora are
  float32).
* `base_score` is stored in probability space for logistic objectives. XGBoost 3.x writes it as a
  vector string such as `"[3.44E-1]"`, possibly estimated from the data. It becomes
  `logit(base_score)`, and every conversion is verified against `predict(output_margin=True)` on
  probe rows. A dump that does not reproduce the margins raises `UnsupportedTreeError` rather than
  returning wrong trees.
* Supported objectives: `binary:logistic` and `reg:logistic` (logistic), `binary:logitraw` (the
  probability is the sigmoid of the margin), and squared-error-style regression (clipped
  identity). DART trees are scaled by `weight_drop`. An `XGBClassifier` with early stopping uses
  trees up to `best_iteration`. `gblinear`, multi-class and categorical splits are unsupported.

**scikit-learn.** `sklearn_gbdt` handles:
* `GradientBoostingClassifier`: raw = init + learning_rate · Σ trees. `init` must be `"zero"` or
  the default prior (the link of the float32-eps-clipped class prior, as sklearn computes it).
  `loss="exponential"` predicts `expit(2·raw)`, and the factor 2 is folded in.
* `HistGradientBoostingClassifier`: `_baseline_prediction` plus the `_predictors` leaf values
  (already shrunk). Splits use the real-valued bin threshold `num_threshold`; NaN follows
  `missing_go_to_left`. Categorical features are unsupported.
* RandomForest, ExtraTrees and DecisionTree: the averaged class-1 leaf fraction.

A `Pipeline` is unwrapped only if every step before the estimator is `"passthrough"`. Any real
preprocessing makes tree access unavailable, because the thresholds would be in the transformed
space. The malicious class is `classes_[1]`, the second of the sorted labels.

**ONNX.** A single `TreeEnsembleClassifier`/`TreeEnsembleRegressor` node (ai.onnx.ml opset ≤ 3)
that reads the graph input directly (only `Cast`/`Identity` in between) is converted:
* `BRANCH_LEQ`/`LT`/`GTE`/`GT` become `<=` splits and NaN follows
  `nodes_missing_value_tracks_true`. `EQ`/`NEQ`/`MEMBER` splits and the opset-5 `TreeEnsemble`
  operator are unsupported.
* The output transform is identified empirically. Each candidate interpretation of `base_values`,
  `post_transform` and onnxruntime's binary-classifier conventions is scored against the session's
  own probabilities on probe rows (ties, float32 neighbours, NaN). A verified least-squares fit is
  the fallback for layouts no standard interpretation explains. One example: with leaf weights on
  *both* classes, onnxruntime 1.30's column 1 is `sigmoid(s0 + b1)`. A graph that post-processes
  the scores in any other way raises `UnsupportedTreeError`.
* ONNX graphs carry no training cover, so leaves get cover 1 (`meta["cover"] == "uniform"`). That
  keeps SHAP additive, but cover-based heuristics are uninformative for ONNX models.
* onnxruntime 1.30 cannot load double-output tree operators at ai.onnx.ml opset 3, and
  skl2onnx 1.20 cannot export `HistGradientBoostingClassifier` at that opset.

## Reference models

Measured on 2000 random rows of each canonical corpus:

| model | trees / nodes | max \|Δ proba\| vs LightGBM | SHAP additivity | load + convert |
|---|---|---|---|---|
| `ember_model_2018.txt` (ember_v2) | 1000 / 2.96 M | 2e-19 | 1e-13 | 32 s (24 s is LightGBM's own `dump_model()`); `to_bytes()` is 145 MB |
| `EMBER2024_PE.model` (ember_v3) | 500 / 144 k (214 categorical splits expanded) | 1e-16 | 2e-11 | 1.3 s; 7 MB |

## Tests

```bash
.venv/bin/python -m pytest tests/unit/test_loaders.py tests/unit/test_loaders_lightgbm.py \
    tests/unit/test_loaders_xgboost.py tests/unit/test_loaders_sklearn.py tests/unit/test_loaders_onnx.py -q
```

`test_loaders.py` covers registry, routing, the pickle policy (a fixture fails the test if
anything unpickles before the policy check) and native-object detection. The per-framework files
train small binary models on tie-heavy random data with NaNs and assert native vs `TreeEnsemble`
probabilities ≤ 1e-6, SHAP additivity ≤ 1e-6 and a `to_bytes`/`from_bytes` round trip.
