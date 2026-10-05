# Third-party notice entries — agent runner (runner, CLI, manifest, environment, report.json)

No third-party source code was copied, ported or vendored into:

* `src/malvalid/runner.py`
* `src/malvalid/cli.py`
* `src/malvalid/__main__.py`
* `src/malvalid/manifest.py`
* `src/malvalid/environment.py`
* `src/malvalid/report/json_writer.py`
* `tests/unit/test_runner.py`, `tests/unit/test_runner_support.py`, `tests/unit/test_runner_integration.py`,
  `tests/unit/test_runner_sandbox.py`, `tests/unit/test_cli.py`, `tests/unit/test_manifest.py`,
  `tests/unit/test_verdict.py`

All of it is original code written for MalValid under Apache-2.0.

The following already-declared core dependencies are **imported** at runtime (not copied):

| name | license | URL | used in |
|---|---|---|---|
| Typer | MIT | https://github.com/fastapi/typer | `cli.py` (command-line interface), tests (`typer.testing.CliRunner`) |
| Click | BSD-3-Clause | https://github.com/pallets/click | `cli.py` (exception types re-raised by the error guard) |
| Rich | MIT | https://github.com/Textualize/rich | `cli.py` (terminal verdict summary, tables, log handler) |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy | `runner.py` (tree-fidelity check, seeds) |
| PyYAML | MIT | https://github.com/yaml/pyyaml | tests (`test_cli.py` writes gate policies) |
