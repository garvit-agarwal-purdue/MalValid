# Third-party notice entries — agent schema_v2 (ember_v2 feature schema, EMBER2018 corpus)

No third-party source code was copied, ported or vendored into:

* `src/malvalid/schemas/ember_v2.py`
* `src/malvalid/corpora/ember2018.py`
* `tests/unit/test_schema_ember_v2.py`
* `tests/unit/test_corpus_ember2018.py`

All of it is original code written for MalValid under Apache-2.0. The AGPL-3.0 EMBER repository
(https://github.com/elastic/ember) was consulted only for the **semantics of the published data**
(field meanings, the vector layout and the hashing-trick widths), which the vectoriser has to
reproduce so that vectors match the released dataset and models. No EMBER code was copied or
translated line by line.

| name | license | URL | how it is used |
|---|---|---|---|
| EMBER2018 feature data (`ember_dataset_2018_2.tar.bz2`) | MIT (data files; the EMBER code is AGPL-3.0 and unused) | https://github.com/elastic/ember | not bundled; `malvalid corpus build ember_v2_2018` derives the canonical corpus locally from a user-downloaded copy. Citation: H. S. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models", arXiv:1804.04637, 2018. `tests/unit/test_schema_ember_v2.py` embeds three records (feature JSON only) from the release as test data; they are the only EMBER data in the repository and `NOTICE` reproduces the MIT permission notice for them. |
| LIEF | Apache-2.0 | https://github.com/lief-project/LIEF | optional runtime import (`malvalid[featurize]` extra, already declared) in `ember_v2.py` `featurize` / `raw_features` |
| scikit-learn (`sklearn.utils.murmurhash3_32`, `FeatureHasher` in tests) | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn | imported (core dependency): the reference MurmurHash3 for the hashing trick; `FeatureHasher` is the independent test oracle |
| MurmurHash3 algorithm (Austin Appleby) | public domain | https://github.com/aappleby/smhasher | `ember_v2._murmur3_signed` is an original pure-Python implementation of the public-domain algorithm, used only if scikit-learn's binding is missing |
