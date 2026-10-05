# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/)
and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First public release.

### Changed

- The project is now called MalValid (package, CLI and import name `malvalid`); pre-release builds
  were called malguard. Files those builds wrote are still read: corpora in the `malguard-corpus/1`
  format (same content hashes, no rebuild needed), `malguard-report/1` reports, run folders and
  `malguard-model-spec/1` model specs. Environment variables are now `MALVALID_*`.

### Added

- `malvalid` CLI: `run`, `validate-adapter`, `inspect-model`, `list-modules`,
  `corpus list/info/verify/build`, `init-config`, `report render`, `sandbox-check`, `serve`.
- Submit a model file directly (`--model`, LightGBM, XGBoost, scikit-learn GBDT, ONNX) or through a
  Python adapter.
- Test modules M0 file safety, M1 performance, M2 temporal drift, M4 membership inference, M5 backdoor
  screening, M6 extraction, M7 explanation, combined into a 0-100 score and a READY / CONDITIONAL /
  NOT READY / BLOCKED verdict with a YAML gate policy.
- Sandboxed model runner (bubblewrap or unshare) with pickle refusal and a validated wire protocol.
- EMBER v2 (2018) and v3 (2024) feature schemas and corpus builders, plus synthetic corpora for demos.
- Local web UI (`malvalid serve`, `[web]` extra) with token login, uploads, run history and comparison.
- HTML and JSON reports.
- Examples: `synthetic_demo`, `lightgbm_ember2018`, `xgboost_ember2018`, `lightgbm_ember2024`.
- Double-click launchers for Windows (x64, and Windows 11 on Arm through an x64 Python), macOS and Linux
  that install MalValid into a private per-user folder and start the web UI. They warn when LightGBM /
  XGBoost cannot load because a system library is missing (macOS: Homebrew `libomp`; Windows: the
  Microsoft Visual C++ Redistributable).
  When they must download `uv`, they check it against a SHA-256 pinned in the launcher; on Windows they
  only use a 64-bit Python 3.11.
- LightGBM and XGBoost model files load from paths with non-ASCII characters on Windows.
