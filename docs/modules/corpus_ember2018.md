# EMBER2018 canonical corpus (`ember_v2_2018`) and the `ember_v2` feature schema

**What it is:** the canonical evaluation data for detectors trained on EMBER2018-style vectors
(EMBER feature version 2, 2381 float32 features). MalValid's modules score your detector on it:
M1 on the held-out test split, M2 on monthly windows after your training cutoff, M4 and M6 on the
pool. It holds only feature vectors, sha256 hashes, labels and first-seen months. It contains no
executables.

| | |
|---|---|
| Provider | `ember_v2_2018` → `malvalid.corpora.ember2018:Ember2018Provider` |
| Schema | `ember_v2` → `malvalid.schemas.ember_v2:EmberV2Schema` (dim 2381) |
| Rows | 1,000,000: train 800,000 (300k malicious, 300k benign, 200k unlabeled) + test 200,000 (100k / 100k) |
| Version | `2018.2-r1` (source release `2018_2`, build recipe `r1`) |
| Content hash (pinned) | `b76ea441be1a62197c7d19801174309e988eea54740ff22f03fa75e66beb4b98` |
| Size on disk | 9.5 GB `X.npy` + 329 MB `meta.npz` |
| Source | `ember_dataset_2018_2.tar.bz2`, sha256 `b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812` |
| Data license | MIT (EMBER data files). The EMBER source code is AGPL-3.0; MalValid does not use it. |
| Code | `src/malvalid/corpora/ember2018.py`, `src/malvalid/schemas/ember_v2.py` |

## Getting the corpus

MalValid does not redistribute the corpus. Build it once from the public EMBER2018 feature
release:

```bash
# 1. download the feature release (1.7 GB, feature JSON only, no PE files)
curl -LO https://ember.elastic.co/ember_dataset_2018_2.tar.bz2
sha256sum ember_dataset_2018_2.tar.bz2   # b6052eb8d350a49a8d5a5396fbe7d16cf42848b86ff969b77464434cf2997812
# 2. extract it (creates ember2018/, about 11 GB of JSONL)
tar -xjf ember_dataset_2018_2.tar.bz2
# 3. vectorise it into the malvalid-corpus/1 format
malvalid corpus build ember_v2_2018 --source ember2018/ --workers 8
#    default output: $MALVALID_CORPUS_DIR/ember_v2_2018 (else ~/.cache/malvalid/corpora/ember_v2_2018)
malvalid corpus verify ember_v2_2018
```

`--source` may be the `ember2018/` directory or its parent. With 10 worker processes the build
took **49 s** on a shared 28-core node; one worker takes several minutes. If the corpus is
missing, MalValid reports these same steps (`Ember2018Provider.unavailable_hint`).

The build is **deterministic**. The same source files give byte-identical `X.npy` and `meta.npz`,
and so the same content hash, whatever the worker count or chunk size. `meta.npz` is written with
fixed zip metadata because `numpy.savez` stamps the current time into it. A clean rebuild of the
canonical corpus from the official release reproduced the pinned hash exactly. `load()` refuses a
directory with a different hash, so every report built on this corpus refers to the same data.

## Layout

* **Row order:** `train_features_0.jsonl` … `train_features_5.jsonl`, then `test_features.jsonl`,
  in file order. Train is rows 0–799,999 and test is rows 800,000–999,999.
* **`label`:** the record's `label`: 1 malicious, 0 benign, −1 unlabeled (train only).
* **`timestamp`:** the first day of the record's `appeared` month (`"2018-03"` → 2018-03-01). Every
  row has one. Test rows first appeared in Nov–Dec 2018. Train rows first appeared in Jan–Oct 2018,
  except `train_features_0.jsonl`: 50,000 benign rows first seen between 2006-12 and 2017-12.
* **`split`:** `train` / `test`.
* **Roles:** `eval = [test]` (M1, M5, M7), `temporal = [train, test]` (M2 windows after the
  declared cutoff), `pool = [train, test]` (M4 non-members, M6 queries), `challenge = []`
  (EMBER2018 has no challenge set).
* **Manifest provenance (`source`):** dataset name, homepage, download URL, archive name, bytes and
  sha256, license, citation, and per-file byte counts, record counts and sha256. It also records
  whether an `ember_dataset_2018_2.sha256` file found next to the source matches the official hash.
  `extra` holds the build recipe, the timestamp rule, per-split counts and month range, and the
  number of rows with NaN (0), without a timestamp (0), and with a duplicate sha256 (0).

## Correctness gate

The vectoriser was checked against the published EMBER2018 LightGBM model
(`ember_model_2018.txt`, trained by the EMBER authors on these features):

| data | rows | ROC AUC | threshold at 1% FPR | detection at 1% FPR | threshold at 0.1% FPR | detection at 0.1% FPR |
|---|---|---|---|---|---|---|
| test split (`eval`) | 200,000 | **0.996429** | 0.8336 | 96.498% | 0.99957 | 86.935% |
| train sample (labeled, seed 0) | 100,000 | 1.000000 | 9.2e-05 | 100% | 2.6e-04 | 100% |

Thresholds are benign-score quantiles, so the FPR is exactly 1.000% / 0.100% on the test split.

* **The numbers match the published benchmark exactly.** The EMBER authors' EMBER2018 notebook
  (`resources/ember2018-notebook.ipynb` in the EMBER repository, checked 2026-09-29) prints ROC
  AUC `0.9964289467999999` for this model on the test set. That is the same value computed here,
  to every printed digit. It also gives a 1%-FPR threshold of 0.8336 with 96.498% detection,
  which matches too. At 0.1% FPR the notebook rounds the threshold to 0.9996, which gives 0.098%
  FPR and 86.808% detection. The exact benign quantile used here (0.99957) gives 0.100% and
  86.935%.
* The often-quoted ~0.999 AUROC is the **EMBER2017** (feature version 1) benchmark. EMBER2018
  was sampled to be harder, and ~0.9964 is this model's published quality on it.
* The model separates its own training rows perfectly (AUROC 1.0 on a 100k-row sample). This only
  happens if the train vectors are the ones it was trained on, down to hashing signs and bucket
  indices.
* 17,235 rows sampled from all seven files (the first 1,000 of each, then every 97th) were
  re-vectorised with the current code. They matched the corpus bit for bit, including sha256,
  label, split and timestamp.
* Unit tests compare the batched vectoriser with an independent per-record oracle written with
  scikit-learn's `FeatureHasher`. They also pin a digest for three embedded real records and check
  those records against corpus rows 834708, 855440 and 94402.

## The `ember_v2` schema

### Groups (pinned, contract §4.1)

| group | range | size | contents |
|---|---|---|---|
| `histogram` | [0, 256) | 256 | byte histogram, normalised to sum 1 |
| `byteentropy` | [256, 512) | 256 | 16 entropy bins × 16 high-nibble bins over 2048-byte windows, step 1024, normalised |
| `strings` | [512, 616) | 104 | printable-string count, average length, printable count, distribution of the 96 printable characters, entropy, and counts of `c:\`, `http(s)://`, `HKEY_`, `MZ` |
| `general` | [616, 626) | 10 | size, virtual size, has_debug, export/import counts, has_relocations/resources/signature/tls, COFF symbols |
| `header` | [626, 688) | 62 | COFF timestamp; machine, characteristics, subsystem, DllCharacteristics and magic hashed into 10 buckets each; 11 optional-header integers |
| `section` | [688, 943) | 255 | 5 counts; per-section size, entropy and virtual size hashed by name (50 buckets each); entry-section name; entry-section flags |
| `imports` | [943, 2223) | 1280 | 256 hashed libraries and 1024 hashed `library:function` entries |
| `exports` | [2223, 2351) | 128 | hashed exported names |
| `datadirectories` | [2351, 2381) | 30 | size and RVA of the 15 PE data directories |

Feature names follow `<group>.<field>` with an index for hashed or binned blocks. Examples:
`general.size`, `header.coff.timestamp`, `header.optional.major_linker_version`,
`imports.functions_hashed[17]`, `byteentropy.bin07[0xA_]` (entropy bin 7, high nibble A),
`strings.printabledist[0x41]` (the fraction of printable characters that are `A`),
`datadirectories.IMPORT_TABLE.size`. Use `schema.feature_index(name)` and `schema.feature_info(i)`
to look them up.

### Vectorising raw EMBER JSON

`vectorize_raw(record) -> (2381,) float32` and `vectorize_raw_batch(records) -> (n, 2381)` take
records in the dataset's JSONL format. The batch path hashes every token of a batch in one pass,
so it is much faster than looping. A malformed record raises `ValueError` that gives the record's
position, its sha256 and the offending field. The semantics that matter for fidelity:

* Hashed blocks use scikit-learn's hashing trick: signed 32-bit MurmurHash3 (seed 0), bucket
  `|h| mod width`, sign −1 for negative hashes, weights of colliding tokens summed.
* Library names are lower-cased and de-duplicated. Functions are hashed as
  `"<library lower-case>:<name>"`, and ordinal imports appear as `ordinalN`.
* The entry-section name is hashed **character by character**. The entry-flags block collects the
  flags of every section whose name equals the entry-section name. Both behaviours are properties
  of the published vectors.
* The histograms are normalised in float32. An all-zero histogram gives NaN, which tree models
  treat as missing. All other groups are computed in float64 and rounded to float32 once.

### `featurize(raw_bytes)` (optional, needs LIEF)

The `featurize` extra (from a source checkout: `pip install -e '.[featurize]'`) installs `lief>=0.14`. `featurize_available()` returns
whether `import lief` works. `featurize` computes the byte-level groups (histogram, byteentropy,
strings) directly from the bytes, so they match the dataset exactly. It then parses the PE with
LIEF and maps the header, section, import, export and data-directory fields to the LIEF 0.9
vocabulary the dataset uses. Examples of that vocabulary: machine `I386`, section flags such as
`MEM_EXECUTE`, and one `ALIGN_*` name for every alignment code that shares a bit with the
section's code. The entry section is resolved the way LIEF 0.9 did it: the entry point's virtual
address is looked up as a file offset, and on a miss the first executable section is used. This
is why UPX-packed files report `UPX0`.

**Treat `featurize` output as approximate.** EMBER2018 was extracted with LIEF 0.9.0, while
MalValid runs modern LIEF (tested with 0.17.6). The parsers differ on malformed or unusual files.
Examples include symbol counting, imported-function counting for odd import tables, signature
detection, and section entropy on truncated sections. Individual PE-structure values can
therefore differ from what EMBER's own extractor would have produced. It was tested on the benign
setuptools launcher executables (`cli-32/64.exe`, `gui-32/64.exe`); MalValid never uses malware for
testing. If input is not a PE, the structure groups are left empty and the byte-level groups are
still filled in. `schema.raw_features(bytes)` returns the intermediate EMBER-style JSON record.

### Controllability

For each feature, the controllability level says how freely a file's author can change it
**without changing what the program does**. M7 uses it to flag detectors that lean on features an
author can set trivially. Counts: controllable 79, append-only 1629, derived 621, fixed 52.
`schema.controllability_reasons()` gives a one-line reason for every feature.

| features | level | why |
|---|---|---|
| `header.coff.timestamp` | controllable | link timestamp is informational; the loader ignores it |
| `header.optional.{major,minor}_{image,linker,operating_system}_version` | controllable | version stamps the loader does not act on |
| `header.optional.sizeof_code`, `sizeof_heap_commit` | controllable | informational sum / resource hint; any sane value works |
| `general.has_debug`, `general.has_signature`, `general.symbols` | controllable | debug directory, Authenticode blob and COFF symbol table are not needed to run |
| `datadirectories.{CERTIFICATE_TABLE, DEBUG, ARCHITECTURE, GLOBAL_PTR, BOUND_IMPORT}.*` | controllable | not used to run the image (signature, debug data, reserved or IA-64-only, stale bind cache) |
| `section.entry_name_hashed[*]`, `section.num_empty_name` | controllable | section names are free text |
| COFF-characteristics / DllCharacteristics buckets | per bucket | a bucket moves only through the flags that hash into it. Only loader-ignored flags (e.g. `LINE_NUMS_STRIPPED`, `DEBUG_STRIPPED`, `NO_BIND`) → controllable. Only structural flags (`DLL`, `SYSTEM`, `NO_SEH`, `APPCONTAINER`, `FORCE_INTEGRITY`) or none → fixed. A mix, or opt-in hardening flags such as `NX_COMPAT`, `DYNAMIC_BASE`, `GUARD_CF` → derived |
| `general.size`, `general.vsize`, `header.optional.sizeof_headers` | append-only | appending data or sections only grows them |
| `strings.numstrings`, `printables`, `paths`, `urls`, `registry`, `MZ` | append-only | counts over file bytes; appended data can only add |
| `imports.*`, `exports.*`, `general.imports`, `general.exports`, `general.has_resources` | append-only | entries can be added. Removing ones the program uses breaks it |
| `section.num_sections`, `num_zero_size`, `num_rx`, `num_w`, `section.{sizes,entropy,vsize}_hashed`, `section.entry_characteristics_hashed` | append-only | sections and flags can be added; the code section must stay executable |
| `datadirectories.{EXPORT_TABLE, IMPORT_TABLE, RESOURCE_TABLE, IAT, DELAY_IMPORT_DESCRIPTOR}.size` | append-only | tables grow when entries are added (their RVAs are derived) |
| `histogram[*]`, `byteentropy[*]` | derived | whole-file distributions; they move only as a side effect of other byte changes |
| `strings.avlength`, `strings.printabledist[*]`, `strings.entropy` | derived | ratios over all strings; they move only as a side effect of added strings |
| `header.coff.machine_hashed[*]`, `header.optional.subsystem_hashed[*]`, `header.optional.magic_hashed[*]` | fixed | target CPU, subsystem and PE32/PE32+ format define whether and how the program runs |
| `header.optional.{major,minor}_subsystem_version` | fixed | checked by the loader against the running OS |
| `general.has_relocations`, `general.has_tls` | fixed | rebasing and TLS callbacks affect execution |
| `datadirectories.{EXCEPTION_TABLE, BASE_RELOCATION_TABLE, TLS_TABLE, LOAD_CONFIG_TABLE, CLR_RUNTIME_HEADER}.*` | fixed | unwind data, relocations, TLS, load config and the .NET header affect execution |

EMBER v2 has no PE checksum feature. The optional-header checksum is controllable in principle,
but it is not part of this vector.

## Tests

```bash
.venv/bin/python -m pytest tests/unit/test_schema_ember_v2.py tests/unit/test_corpus_ember2018.py -q
# with the real data (the slow AUROC gate takes about 30 s with 4 threads):
MALVALID_EMBER2018_DIR=/path/to/corpora/ember_v2_2018 \
MALVALID_EMBER2018_SOURCE=/path/to/ember2018 \
MALVALID_EMBER2018_MODEL=/path/to/ember_model_2018.txt \
  .venv/bin/python -m pytest tests/unit/test_schema_ember_v2.py tests/unit/test_corpus_ember2018.py -q -m ember
```

The fast tests build small corpora from synthetic records laid out like the real release. They
cover determinism across worker counts and chunk sizes, provenance, roles, actionable build
errors, the pinned-hash refusal and the install hint. `ember` tests are skipped unless the
environment variables above point at real data. `MALVALID_CORPUS_DIR/ember_v2_2018` is also
accepted for the corpus.

## Known limitations

* The train split's labels come from the EMBER release (derived from 2018 AV scan results).
  Label noise at that level is inherited.
* 50,000 benign train rows first appeared before 2018. If your training cutoff is before 2018,
  M2's windows between the cutoff and 2018 contain only benign rows, so M2 drops them under
  `require_both_classes`.
* `featurize` is approximate (see above). For exact EMBER vectors, use the JSON records from the
  release or your own LIEF 0.9 pipeline, and give your adapter a `featurize`.
