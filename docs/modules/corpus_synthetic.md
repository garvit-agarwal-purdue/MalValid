# `synthetic_v2` / `synthetic_v3`: synthetic EMBER-shaped corpora for demos and CI

Owner: build agent `schema_v3_synth`. Code: `src/malvalid/corpora/synthetic.py`
(`SyntheticEmberV2Provider`, `SyntheticEmberV3Provider`, `generate()`). Tests:
`tests/unit/test_corpus_synthetic.py`.

## Purpose, and the disclaimer that goes with it

These corpora let the whole gate run end to end without the multi-GB EMBER downloads: in CI, in
the README demo, and when trying MalValid on a toy detector. **They are not evidence about any real
detector.** Every synthetic corpus carries `manifest["synthetic"] = true`, which sets
`Corpus.synthetic` and `Corpus.summary()["synthetic"]`. `manifest.source.note` reads "SYNTHETIC data
for demos and CI; not evidence about any real detector.", and the provider descriptions start with
"SYNTHETIC". The runner and the report use `Corpus.synthetic` to add their disclaimers.

| provider       | feature space           | months (36)       | train/holdout months | test/challenge months |
|----------------|-------------------------|-------------------|----------------------|-----------------------|
| `synthetic_v2` | `ember_v2` (2381)       | 2017-01 … 2019-12 | 2017-01 … 2017-12    | 2018-01 … 2019-12     |
| `synthetic_v3` | `ember_v3` (2568)       | 2022-01 … 2024-12 | 2022-01 … 2022-12    | 2023-01 … 2024-12     |

The default size is 20,000 rows: train 5,315, holdout 1,255, test 12,830, challenge 600. Train,
holdout and test are about 50 % malicious, and challenge is 100 % malicious. Roles: `eval = [test]`,
`temporal = [train, holdout, test]`, `challenge = [challenge]`, `pool = [train, holdout, test]`. The
`holdout` split covers the training months but is not part of `train`. It serves as validation data
and as time-matched non-members for M4. A demo detector should be trained on `train` only, and its
training manifest should list the `train` hashes.

Each row has its own sha256: `sha256("malvalid-synthetic/<name>/seed<seed>/n<n>/<i>")`. These never
collide with real file hashes in practice. Timestamps are daily and sorted chronologically.

## How the data is generated

A seeded *world* defines benign software clusters and malicious families:

* Benign clusters: MSVC apps, signed system components, installers, .NET apps, Delphi apps, MinGW
  tools, Go/Rust binaries, protected games.
* Malicious families: packed trojans, injectors, downloaders, ransomware, stealers, .NET RATs,
  bundlers, Go malware, virtualised RATs.

Each family has a birth month and a lifetime, so new families appear after the training period.
Within a family, the marker imports, section names and string habits rotate gradually. This is the
drift. Each row starts as a latent file description:

* toolchain, packer, capabilities;
* sections, imports, exports, strings;
* header fields, signature, rich header, parser warnings.

This description is drawn from its cluster, then blended by `w ~ Beta(0.5, 6)` with a cluster of the
other class so that the classes overlap. It is then rendered into the real feature layout:

* byte and byte-entropy histograms are normalised (each row sums to 1) and follow the section
  composition;
* counts are integral: string counts, printables, import/function/library counts, section counts,
  warnings, rich-header pairs, file size;
* hashed blocks use the same MurmurHash3 buckets and signs as the real vectorizers
  (`lib:function` tokens, section names, `name:property` tokens, rich-header comp-ids). For example,
  `kernel32.dll` lands in its real `imports.libraries_hashed` bucket, so a real EMBER-trained model's
  splits are at least structurally meaningful on the synthetic rows;
* scalar relationships hold: `printables = numstrings × avlength`, `strings.entropy` is the entropy
  of `printabledist`, `sizeof_image` follows the section layout, and `has_signature` ⇔ a security
  directory ⇔ an overlay.

The `challenge` split holds evasive, benign-looking malware from the test period.

## Measured behaviour (default parameters, numpy 2.4.6)

The model is a LightGBM (200 trees, 31 leaves, lr 0.1) trained on `train` only:

| metric                                   | `synthetic_v2`         | `synthetic_v3`         |
|------------------------------------------|------------------------|------------------------|
| generation time, 20,000 rows             | 2.9 s                  | 2.9 s                  |
| test AUROC                               | 0.980                  | 0.981                  |
| monthly F1 at 0.5, first → last test month | 0.954 → 0.890 (min 0.860) | 0.959 → 0.867          |
| challenge detection rate at 0.5          | 0.33                   | 0.32                   |
| gain share by group                      | section 0.47, imports 0.17, histogram 0.16, strings 0.07, datadirectories 0.06, header 0.04 | section 0.41, imports 0.21, histogram 0.11, strings 0.07, pefilewarnings 0.07, header 0.05, authenticode 0.03, datadirectories 0.03 |

On `synthetic_v3`, the top single feature holds 9 % of mean |SHAP| and the effective number of
features (1/Herfindahl) is about 50, so M7 sees a model that is not keyed on one feature. The
default content hashes were `7a57a1f2…` (v2) and `dbd3f563…` (v3) under numpy
2.4.6. They are not pinned (`expected_content_hash = None`) because they depend on numpy's RNG
streams.

The unit tests assert these properties with margins. They check that AUROC is in [0.95, 0.995],
that mean F1 over the first 6 test months exceeds the last 6 by ≥ 0.03 with a negative trend, and
that challenge detection is at least 0.2 below test detection. They also require at least 5 feature
groups with ≥ 3 % of the gain and no group above 60 %. They check normalised histograms, integral
counts, the real hash buckets, determinism, and generation under 30 s.

## Loading, caching and parameters

* `provider.load(cfg)` looks for `corpus_dir` from the config, else `$MALVALID_CORPUS_DIR/<name>`,
  else `~/.cache/malvalid/corpora/<name>`. If `manifest.json` is missing, it generates the corpus
  there first. It writes to a temporary sibling directory and renames into place, with the manifest
  last. Later loads reuse and verify the cache.
* If the directory is not writable, it logs a warning and returns an **in-memory** corpus
  (`path=None`) with the same content hash as the on-disk build would have. The same is available
  explicitly through `provider.in_memory()`.
* A directory that holds some other corpus (another name, or `synthetic: false`) is refused with a
  clear message. A cache written by an older `GENERATOR_VERSION` is regenerated with the parameters
  recorded in its manifest.
* `$MALVALID_SYNTHETIC_ROWS` and `$MALVALID_SYNTHETIC_SEED` override the default size and seed.
  Non-default parameters get their own cache directory, `<name>-n<rows>-seed<seed>`. The CLI
  equivalent is `malvalid corpus build synthetic_v3 --out DIR`; `build()` also accepts `n=` and `seed=`.
* `generate(feature_version, SyntheticParams(...))` returns the arrays directly
  (`X, sha256, label, timestamp, split`, plus the generator `cluster` per row for analysis). A
  given seed, size and feature space always produce the same corpus.

`SyntheticParams` has these fields and defaults:

* `n=20000`, `seed=0`;
* `months=36`, `train_months=12`;
* `holdout_fraction=0.2`, `malicious_fraction=0.5`, `challenge_fraction=0.03`;
* `overlap_a=0.5`, `overlap_b=6.0`.

`n` must be at least 200.
