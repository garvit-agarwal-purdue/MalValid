# Third-party notice entries: agent schema_v3_synth (ember_v3 schema, EMBER2024 corpus, synthetic corpora)

## Ported code

### thrember (EMBER2024 reference feature extractor): Apache License 2.0

* Upstream: https://github.com/FutureComputing4AI/EMBER2024 (package `thrember`), commit
  `0ef753e81d98bf209f71b03cd331dfc190b5b54d`.
* Copyright: the EMBER2024 / thrember authors (Joyce et al.). The upstream `LICENSE` is the standard
  Apache License 2.0 text with no separate NOTICE file.
* Ported from: `src/thrember/features.py` (the `process_raw_features` vectorization of every feature
  group, and the raw PE extraction built on pefile) and `src/thrember/pefile_warnings.txt` (the
  list of normalised pefile warnings, embedded as `PEFILE_WARNINGS`).
* Into: `src/malvalid/schemas/_thrember_port.py`. That file keeps the upstream copyright and Apache-2.0
  license header and lists the modifications made for malvalid: restructured into functions,
  a batched vectorizer, lazy/optional pefile and signify, deterministic warning normalisation,
  logging instead of printing, and zeros instead of NaN for empty histograms. The output vectors are
  unchanged.
* Used by: `src/malvalid/schemas/ember_v3.py` (schema), `src/malvalid/corpora/ember2024.py` (corpus
  build), and indirectly `src/malvalid/corpora/synthetic.py`, which uses only the pinned layout.

No code was taken from the original EMBER repository (elastic/ember, AGPL-3.0). The ember_v2 schema
belongs to another agent. `src/malvalid/corpora/synthetic.py` and `src/malvalid/corpora/ember2024.py`
are original MalValid code under Apache-2.0.

## Data (not redistributed)

* **EMBER2024 feature files** (https://huggingface.co/datasets/joyce8/EMBER2024; Joyce et al.,
  *EMBER2024: A Benchmark Dataset for Holistic Evaluation of Malware Classifiers*, KDD 2025;
  license Apache-2.0 per the dataset card). The
  `ember_v3_2024` corpus is built locally by the user from the downloaded files. MalValid ships
  neither the data nor any derived vectors. A built corpus stores only feature vectors, sha256,
  labels and first-seen dates, never binaries.
* **EMBER2024_PE.model** (thrember benchmark LightGBM; https://huggingface.co/joyce8/EMBER2024-benchmark-models,
  Apache-2.0 per the model card) is used only for the correctness check in
  `docs/modules/corpus_ember2024.md`. It is not shipped.

## Runtime libraries (imported, not copied)

* `pefile` (MIT, https://github.com/erocarrera/pefile) is optional and enables
  `EmberV3Schema.featurize`. It is not yet declared in `pyproject.toml`; see this agent's
  dependency request.
* `signify` (MIT, https://github.com/ralphje/signify) is optional and gives exact Authenticode
  features for signed files.
* `scikit-learn` (BSD-3-Clause; `FeatureHasher`, `murmurhash3_32`) and `numpy` (BSD-3-Clause) are
  already core dependencies.
