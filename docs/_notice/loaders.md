# Third-party notice entries — agent loaders (ModelLoader plugins)

No third-party source code was copied, ported or vendored into:

* `src/malvalid/loaders/lightgbm_loader.py`
* `src/malvalid/loaders/xgboost_loader.py`
* `src/malvalid/loaders/sklearn_loader.py`
* `src/malvalid/loaders/onnx_loader.py`
* `tests/unit/test_loaders.py`, `tests/unit/test_loaders_lightgbm.py`, `tests/unit/test_loaders_xgboost.py`,
  `tests/unit/test_loaders_sklearn.py`, `tests/unit/test_loaders_onnx.py`

All of it is original code written for MalValid under Apache-2.0. Framework split semantics
(LightGBM's zero band and categorical test, XGBoost's strict `<` splits and `base_score`
convention, scikit-learn's init/baseline predictions, onnxruntime's binary-classifier output
conventions) were determined from each library's public behaviour and documentation and are
verified numerically in the tests. No library source was reproduced.

The following libraries are **imported** at runtime (not copied). All are already declared in
`pyproject.toml` (core dependencies, the `onnx` extra, or `dev`):

| name | license | URL | used in |
|---|---|---|---|
| LightGBM | MIT | https://github.com/microsoft/LightGBM | `lightgbm_loader.py` (`Booster`, `dump_model`, `predict`) |
| XGBoost | Apache-2.0 | https://github.com/dmlc/xgboost | `xgboost_loader.py` (`Booster.load_model`, `save_raw`, `predict`, `DMatrix`) |
| scikit-learn | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn | `sklearn_loader.py` (fitted estimator attributes `estimators_`, `tree_`, `_predictors`, `_baseline_prediction`, `init_`) |
| joblib | BSD-3-Clause | https://github.com/joblib/joblib | `lightgbm_loader.load_pickled` (only with `--allow-pickle`) |
| onnxruntime (`onnx` extra) | MIT | https://github.com/microsoft/onnxruntime | `onnx_loader.py` (`InferenceSession`, CPU provider only) |
| onnx (`onnx` extra) | Apache-2.0 | https://github.com/onnx/onnx | `onnx_loader.py` (reading `ai.onnx.ml` tree attributes, `numpy_helper`) |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy | all loaders |
| skl2onnx (`dev`, tests only) | Apache-2.0 | https://github.com/onnx/sklearn-onnx | `tests/unit/test_loaders_onnx.py` (exporting scikit-learn models to ONNX) |
| SHAP (tests only here) | MIT | https://github.com/shap/shap | tests check `TreeExplainer` additivity on `TreeEnsemble.to_shap_model()` |
