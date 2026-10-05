# Contributing to MalValid

Thanks for helping. Bug reports, docs fixes and new test modules are all welcome.

## Development setup

Python 3.11 or newer. [uv](https://docs.astral.sh/uv/) is the quickest route:

```bash
git clone https://github.com/garvit-agarwal-purdue/MalValid.git   # or your fork
cd MalValid
uv venv --python 3.11
source .venv/bin/activate
uv pip install -c constraints/lock-py311.txt -e ".[dev,onnx,featurize,web]"
```

Plain pip works too: `python3.11 -m venv .venv && . .venv/bin/activate && pip install -c constraints/lock-py311.txt -e ".[dev,onnx,featurize,web]"`.

On Linux, install `bubblewrap` (`sudo apt-get install bubblewrap`) so the sandbox tests run; without a
usable sandbox backend they skip. Check what your host supports with `malvalid sandbox-check`.

## Running the tests

```bash
python -m pytest tests/unit tests/e2e -q            # full suite
python -m pytest -m "not slow" -q                   # skip long tests
python -m pytest tests/unit/test_web_proxy.py -q    # one file
```

Tests marked `ember` need the real EMBER corpora and skip otherwise. To run them, set:

* `MALVALID_CORPUS_DIR` to a directory holding the built `ember_v2_2018/` and `ember_v3_2024/` corpora, or
  `MALVALID_EMBER2018_DIR` / `MALVALID_EMBER2024_DIR` to the corpus directories themselves;
* optionally `MALVALID_EMBER2018_SOURCE` / `MALVALID_EMBER2024_SOURCE` (the extracted raw dataset, for the
  build tests) and `MALVALID_EMBER2018_MODEL` / `MALVALID_EMBER2024_MODEL` (the published models);
* `MALVALID_TEST_DATA` to a directory holding `models/`, `corpora/` and `ref_ember2024/`, for the
  adapter-validation EMBER tests.

Tests marked `sandbox` need bubblewrap or user namespaces. Tests marked `posix` need POSIX process
groups, signals, file modes or symlinks and are skipped on Windows. Web tests need the `web` extra.

## Pull requests

* Keep changes focused and add or update tests.
* Do not commit models, corpora, run directories, tokens or any malware. `.gitignore` covers the usual
  suspects; check `git status` before committing.
* Do not add dependencies under AGPL or other incompatible licences. Code ported from elsewhere needs its
  licence text in the file header and an entry in `NOTICE`.
* Security problems: report them privately through
  [GitHub private vulnerability reporting](https://github.com/garvit-agarwal-purdue/MalValid/security/advisories/new)
  as described in [`SECURITY.md`](SECURITY.md), not in a public issue.

By contributing you agree that your contribution is licensed under the Apache License 2.0.
