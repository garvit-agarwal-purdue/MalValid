# Third-party notice entries — agent sandbox (sandboxed model runner, adapter helpers, M0 file safety)

No third-party source code was copied, ported or vendored into:

* `src/malvalid/sandbox/__init__.py`, `host.py`, `worker.py`, `protocol.py`
* `src/malvalid/adapter.py`
* `src/malvalid/adapters/__init__.py`, `validate.py`
* `src/malvalid/modules/file_safety.py`
* `tests/unit/test_sandbox.py`, `test_sandbox_policy.py`, `test_sandbox_protocol.py`,
  `test_file_safety.py`, `test_adapter_validate.py`, `tests/fixtures/adapters/*`

All of it is original code written for MalValid under Apache-2.0. The static pickle-opcode scan in
`file_safety.py` uses the Python standard library's `pickletools.genops` (PSF License) through its
public API. The pickle allowlist and denylist were written for MalValid and not taken from any
other scanner.

Libraries **imported** at runtime (not copied). All are already declared in `pyproject.toml` or
come in with a declared dependency:

| name | license | URL | used in |
|---|---|---|---|
| modelscan (Protect AI) | Apache-2.0 | https://github.com/protectai/modelscan | `modules/file_safety.py` (`modelscan.modelscan.ModelScan` Python API, `modelscan.settings.DEFAULT_SETTINGS`) |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy | wire format (`np.save`/`np.load` with `allow_pickle=False`), everywhere |
| joblib (via scikit-learn) | BSD-3-Clause | https://github.com/joblib/joblib | `sandbox/worker.py` (its `load` is guarded/refused); test fixtures (`joblib_adapter.py`) |
| LightGBM | MIT | https://github.com/microsoft/LightGBM | `sandbox/worker.py` (fallback `Booster.dump_model` tree access), `adapter.py`, tests |
| scikit-learn | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn | test fixtures (a benign `GradientBoostingClassifier` saved with joblib) |

External programs **executed** (not bundled, linked or modified; MalValid only builds their
command lines and runs them as separate processes):

| name | license | URL | used in |
|---|---|---|---|
| bubblewrap (`bwrap`) | LGPL-2.0-or-later | https://github.com/containers/bubblewrap | `sandbox/host.py` (preferred isolation backend) |
| util-linux `unshare` | GPL-2.0-or-later | https://github.com/util-linux/util-linux | `sandbox/host.py` (fallback isolation backend) |

Test inputs: the **benign** Windows launcher executables shipped with setuptools
(`setuptools/cli-64.exe`, MIT, https://github.com/pypa/setuptools) are read at test time as PE
featurizer inputs. They are not copied into the repository.
