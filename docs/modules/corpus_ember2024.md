# `ember_v3_2024`: EMBER2024 canonical corpus and the `ember_v3` feature schema

Owner: build agent `schema_v3_synth`. Code: `src/malvalid/corpora/ember2024.py` (provider),
`src/malvalid/schemas/ember_v3.py` (schema), `src/malvalid/schemas/_thrember_port.py` (vectorizer and
raw-PE extractor, Apache-2.0 port of thrember). Tests: `tests/unit/test_corpus_ember2024.py`,
`tests/unit/test_schema_ember_v3.py`.

## What the corpus is

These are the canonical evaluation rows for detectors that declare `feature_version = "ember_v3"`
(EMBER feature version 3, 2568 float32 features). The corpus is built from the public EMBER2024
**feature files**. It contains feature vectors, sha256, labels and first-seen dates only, and never
any binaries.
The EMBER2024 dataset and its benchmark models are licensed Apache-2.0 (per their Hugging Face
dataset and model cards); MalValid does not redistribute them.

| split       | rows    | malicious | benign  | file types                            | first seen              |
|-------------|--------:|----------:|--------:|---------------------------------------|-------------------------|
| `test`      | 479,987 | 240,000   | 239,987 | Win32 359,994 · Win64 119,993          | 2024-09-22 … 2024-12-14 |
| `challenge` | 4,868   | 4,868     | 0       | Win32 3,225 · Win64 814 · .NET 829     | 2023-09-24 … 2024-12-14 |

Total 484,855 × 2568, about 5.0 GB of `X.npy` plus a 160 MB `meta.npz`.

Roles: `eval = [test]`, `challenge = [challenge]`, `temporal = [test, challenge]` (every challenge
row has a timestamp), `pool = [test]`. The EMBER2024 *train* split (2023-09-24 … 2024-09-21) is
deliberately left out: it is what reference models were trained on.

* **Labels:** the dataset's `label` field (1 malicious, 0 benign). There are no unlabeled rows.
* **Timestamps:** `first_submission_date` (first submission to VirusTotal, epoch seconds), truncated
  to the UTC day.
* **Challenge set:** the EMBER2024 files that initially evaded about 70 AV engines on VirusTotal and
  were later found to be malicious. Only the PE rows are kept (Win32, Win64, .NET). The 1,447 APK/ELF/PDF
  rows are dropped.
* **Duplicates:** each published weekly test file lists every sample **twice**. The second copy
  only adds capa results, and its feature groups are byte-identical (checked by digest during the
  build: 0 conflicts). Also, 13 benign files appear in two consecutive weeks. The build keeps the first
  occurrence of each sha256 per split, which drops 480,013 rows. It then **asserts that every sha256 in
  the corpus is unique** and refuses to write the corpus otherwise. That assertion also catches a
  source directory whose weekly files would be read twice. A zip member and an extracted copy of the
  same file count once, and the loose file wins.
* **Test file types:** only `Win32_test.zip` and `Win64_test.zip` are used, so the eval split holds no
  .NET files. `.NET` rows appear only in the challenge split.

## Building it

```bash
# 1. download the three feature zips (~3.8 GB) from https://huggingface.co/datasets/joyce8/EMBER2024
hf download joyce8/EMBER2024 Win32_test.zip Win64_test.zip challenge.zip --repo-type dataset --local-dir SRC
# 2. vectorize (zip members are streamed, never extracted)
malvalid corpus build ember_v3_2024 --source SRC --out $MALVALID_CORPUS_DIR/ember_v3_2024 --workers 8
```

The build makes two passes over the source files. Pass 1 collects every row's sha256 and plans
deduplication. Pass 2 runs `workers` spawn processes, each of which vectorizes one weekly file
straight into the memory-mapped `X.npy`. Rows are ordered test first (by weekly file name) and
then challenge. On this node the build takes **85 s with 10 workers**. Non-default options
(`test_file_types`, `challenge_file_types`, `limit_per_member`) or an incomplete source give a
corpus labelled `version: "2+custom"`, `extra.canonical: false`. Its content hash will not match the
pinned one, so `Ember2024Provider.load()` refuses it. `load_corpus_dir()` still reads it.

The error message for a missing corpus (`unavailable_hint`) includes these download and build
commands, plus how to point `MALVALID_CORPUS_DIR`, `corpus_dir` or `--corpus-dir` at the corpus.

### Pinned content hash and determinism

```
expected_content_hash = 4f275e15e00ee11cdc67f176bbda932b3feaa92ec6d8584420cb04714e5110cc
X.npy    sha256 8c2322212ea1e2d8017c9732f6e266f1a418b02e5ac35622f343cb1c3ea88dcc
meta.npz sha256 6cebb75157f17e494f4c3913a57917e0e72e82d58a8689b72d2555c1a4076706
```

Verification on 2026-09-29, using the current `CorpusWriter.finalize`, which writes `meta.npz`
deterministically with `write_npz_deterministic`: two independent builds from the published zips,
each with 10 workers, gave byte-identical `X.npy` and `meta.npz` and the pinned content hash above.
The maintainers' canonical copy (`$MALVALID_CORPUS_DIR/ember_v3_2024`) has the same
file hashes. The row order and the vectors do not depend on the number of workers; a unit test
checks 1 worker against 2. The `manifest.extra` block records per-split and per-file-type counts,
the dropped duplicates, the feature-conflict count and the time range.

## Correctness gate: the published EMBER2024 model on this corpus

`EMBER2024_PE.model` (the thrember benchmark LightGBM for PE files, 500 trees, 2568 features) was
scored on the corpus rows with 4 threads. The published numbers are from Joyce et al., *EMBER2024: A
Benchmark Dataset for Holistic Evaluation of Malware Classifiers* (KDD 2025): Table 5 for the test set
and Table 6 for the challenge set, where challenge malware is combined with the test-set benign files.

| metric                                        | MalValid corpus              | published (EMBER2024 paper)                                 |
|-----------------------------------------------|------------------------------|-------------------------------------------------------------|
| test ROC AUC (eval split, Win32+Win64)        | **0.99832**                  | all PE 0.9982 · Win32 0.9984 · Win64 0.9989 · .NET 0.9980    |
| test PR AUC                                   | 0.99845                      | all PE 0.9983 · Win32 0.9986 · Win64 0.9990                 |
| TPR at 1 % FPR (threshold 0.580)              | 97.07 %                      | n/a                                                         |
| TPR at 0.1 % FPR (threshold 0.939)            | 90.13 %                      | n/a                                                         |
| detection rate / FPR at threshold 0.5         | 97.53 % / 1.28 %             | n/a                                                         |
| challenge ROC AUC (challenge vs test benign)  | **0.96722**                  | all PE 0.9643 · Win32 0.9689 · Win64 0.9503 · .NET 0.9865    |
| challenge PR AUC                              | 0.66909                      | all PE 0.6250                                               |
| challenge detection rate at 0.5 / at 1 % FPR  | 70.65 % / 63.99 %            | n/a                                                         |

The test AUROC lies between the published Win32 and all-PE values, as expected: the eval split has
no .NET rows, and three quarters of it is Win32. The challenge AUROC lies between the published
Win32 and all-PE values for the same reason. The benign reference set here contains no .NET files,
which also explains the higher PR AUC. Both results agree with the paper to within about 0.003, so the
vectorizer order is correct: a wrong column order would collapse the AUROC. The test
`test_real_corpus_reproduces_published_model_quality` re-checks these numbers (AUROC > 0.997 and
challenge AUROC > 0.95). It is marked `ember` + `slow` and needs `MALVALID_EMBER2024_MODEL`.

Monthly F1 at threshold 0.5 on the temporal rows after the model's training period (> 2024-09-21):

| month   | rows    | of which challenge | F1     |
|---------|--------:|-------------------:|-------:|
| 2024-09 | 50,820  | 38                 | 0.9834 |
| 2024-10 | 175,172 | 235                | 0.9818 |
| 2024-11 | 174,520 | 249                | 0.9797 |
| 2024-12 | 80,080  | 83                 | 0.9762 |

Only 605 of the 4,868 challenge rows fall after the EMBER2024 training period. Including them in
`temporal` lowers F1 by at most 0.001 per month. A detector trained on EMBER2024 therefore has
three monthly M2 windows after its 2024-09-21 cutoff (2024-09-22 to 10-21, 10-22 to 11-21, 11-22 to
12-21; the corpus ends 2024-12-14, so the last one is partial), and M2's AUT needs at least two.

## The `ember_v3` feature schema

The group boundaries are pinned by the contract (§4.1):

| group | range | size | default controllability |
|---|---|---:|---|
| general | [0, 7) | 7 | append_only (size), derived (entropy), fixed (is_pe, "MZ"), controllable (start_bytes[2:4]) |
| histogram | [7, 263) | 256 | derived |
| byteentropy | [263, 519) | 256 | derived |
| strings | [519, 696) | 177 | append_only (counts, regex counts), derived (avlength, printabledist, entropy) |
| header | [696, 770) | 74 | per field: controllable (timestamp, versions, checksum, stack/heap sizes, DOS-stub fields, most flags), derived (sizeof_headers, sizeof_image), fixed (machine, subsystem, entry point, image base, section alignment, EXECUTABLE_IMAGE/DLL/32BIT flags, FORCE_INTEGRITY, e_magic, e_lfanew) |
| section | [770, 994) | 224 | append_only (counts, overlay size), derived (entropy/size ratios), controllable (every name-keyed hashed bucket, since section names are free) |
| imports | [994, 2276) | 1282 | append_only |
| exports | [2276, 2405) | 129 | append_only |
| datadirectories | [2405, 2439) | 34 | per directory: derived (IMPORT, EXPORT, RESOURCE, BASERELOC, IAT, …), controllable (SECURITY, DEBUG, BOUND_IMPORT, …), fixed (TLS, LOAD_CONFIG, EXCEPTION, COM_DESCRIPTOR); has_relocs flags derived |
| richheader | [2439, 2472) | 33 | controllable (the loader ignores it) |
| authenticode | [2472, 2480) | 8 | append_only (num_certs, chain_max_depth), controllable (self_signed, signing times, parse_error, …) |
| pefilewarnings | [2480, 2568) | 88 | derived |

* `feature_names()` gives stable, human-readable per-feature names (`general.size`,
  `header.coff.timestamp`, `imports.functions_hashed[17]`, `section.entry_name_hashed[3]`,
  `pefilewarnings[<warning text>]`, …). `feature_controllability()` uses the same meaning as ember_v2:
  how freely the author of a file can change the feature without changing program behaviour. M7
  uses it.
* `vectorize_raw(dict)` and `vectorize_raw_batch(list[dict])` take thrember JSONL rows and produce
  vectors identical to thrember's `PEFeatureExtractor.process_raw_features`, including its quirks:
  `exports[0]` is 128 whenever the file has exports; the RESERVED data directory is never written;
  the `email_addr` regex equals `mac_addr`; the float32/float64 cast order matches upstream. The batch
  variant runs each FeatureHasher once per batch. It vectorizes the whole corpus in about 85 s.
* `featurize(raw_bytes)` is available iff `pefile` is importable (`featurize_available()`).
  `signify` is optional: without it, Authenticode features are exact for unsigned files and
  approximate for signed ones (only the certificate count). The tests featurize the benign
  setuptools launcher executables and compare against upstream thrember when its source is
  present.
* `categorical_features` lists the 7 indices EMBER2024's reference models declare categorical.

## Environment variables for the real-data tests

`MALVALID_EMBER2024_DIR` is the corpus directory (default `$MALVALID_CORPUS_DIR/ember_v3_2024`).
`MALVALID_EMBER2024_SOURCE` is the directory with the zips and enables the re-vectorization spot
check. `MALVALID_EMBER2024_MODEL` is `EMBER2024_PE.model` and enables the AUROC gate.

```bash
MALVALID_CORPUS_DIR=$HOME/malvalid-corpora \
MALVALID_EMBER2024_SOURCE=$HOME/ember2024 \
MALVALID_EMBER2024_MODEL=$HOME/ember2024/EMBER2024_PE.model \
.venv/bin/python -m pytest tests/unit/test_corpus_ember2024.py -q -m ember   # 3 passed, ~13 s
```
