# Sandboxed model runner (`malvalid.sandbox`)

MalValid treats the submitted adapter and model files as **untrusted input**. They are imported,
loaded and queried only inside a separate worker process. The harness talks to that process over a
wire format that never unpickles anything, and it checks every answer before a test module sees it.
This page covers what the sandbox protects against, what each backend actually isolates, and how
to debug an adapter when something goes wrong.

| | |
|---|---|
| Source | `src/malvalid/sandbox/host.py` (policy, backends, `SandboxedModel`), `sandbox/worker.py` (worker, pickle guards), `sandbox/protocol.py` (wire format), `src/malvalid/adapter.py` (adapter helpers), `src/malvalid/adapters/validate.py` (`malvalid validate-adapter`) |
| Config | `runtime.sandbox`, `runtime.sandbox_backend`, `runtime.allow_reduced_isolation` (or `--allow-reduced-isolation`), `runtime.sandbox_memory_mb`, `runtime.threads`, `runtime.chunk_rows`, `runtime.allow_pickle` (or `--allow-pickle`) |
| Tests | `tests/unit/test_sandbox.py`, `test_sandbox_policy.py`, `test_sandbox_protocol.py`, `test_sandbox_isolation.py`, `test_adapter_validate.py` |
| Check the host | `malvalid sandbox-check` (runs `probe_backends()`) |

## Threat model

The main case is a researcher gating **their own** detector. The sandbox is there to stop accidents
and to keep a gate run from changing the machine: an adapter that phones home, a pickled model
that runs code when it loads, a model that uses all the RAM on a shared node, or a crash that takes
the harness down with it. Running a model **someone else** sent you is a harder problem. The
sandbox raises the cost of an attack in that case, but it is not a security boundary you should
rely on alone (see [Multi-tenant use](#multi-tenant-use)).

In scope:

* **Code execution on load.** Pickle-based formats (`.pkl`, `.joblib`, `.pt`, …) run arbitrary code
  when they are deserialized. M0 (`file_safety`) scans every artifact statically first. The worker
  then refuses to unpickle anything unless `--allow-pickle` is given. With the flag, every global a
  pickle imports is still checked against MalValid's pickle policy, and dangerous ones such as
  `os.system`, `builtins.eval`, `subprocess.*` or sockets are refused before they are imported.
* **Exfiltration and network side effects.** Under `bwrap` and `unshare` the worker gets its own
  network namespace with only a loopback interface, so `connect()` to anything outside fails.
* **Writes to the host.** Under `bwrap` the whole file system is mounted read-only. `/tmp`,
  `/dev/shm`, `/run` and `$HOME` are replaced by empty tmpfs mounts, and the run's scratch directory
  (`<run>/private/sandbox/work`) is the only writable path the worker can reach.
* **Secrets in the environment.** The worker starts with a new, minimal environment: `PATH`,
  locale, `HOME` pointing at the scratch dir, thread limits and MalValid's own variables. API
  tokens, cloud credentials and `SSH_AUTH_SOCK` from your shell never reach the adapter.
* **Resource exhaustion.** `RLIMIT_AS` (`runtime.sandbox_memory_mb`), `RLIMIT_NOFILE` and
  `RLIMIT_CORE=0` are set. BLAS/OpenMP pools are capped at `runtime.threads`. The worker dies with
  its parent, and every request has a deadline.
* **A malicious or broken worker attacking the harness.** Replies are length-prefixed JSON headers
  plus raw `.npy` payloads written with `allow_pickle=False`. Only numeric dtypes are accepted,
  every size is checked before memory is allocated, and a tree payload must be an uncompressed
  `.npz`, so a zip bomb is rejected. `predict_proba` and `predict` outputs are checked against the
  contract: shape `(n,)`, finite values, `[0, 1]` for probabilities and `{0, 1}` for labels.

Out of scope: kernel exploits and container escapes, side channels, and attacks on the Python
interpreter or native libraries (LightGBM, XGBoost, onnxruntime) that the worker shares with the
host installation. An isolating backend makes these harder, but it does not rule them out.

## Backends

`runtime.sandbox_backend: auto` (the default) probes the backends in the order below and uses the
first one that works **and** isolates the network. `probe_backends()` actually starts a small
interpreter under each backend and reports what that interpreter can observe: network interfaces,
its PID, whether `/` is mounted read-only, whether `$HOME` is visible, and whether pipe fds reach it.

| backend | network | PIDs | file system | notes |
|---|---|---|---|---|
| `bwrap` (bubblewrap) | own netns, loopback only | own PID namespace (`--unshare-pid`), plus IPC/UTS/cgroup | read-only `/`; tmpfs over `/tmp`, `/dev/shm`, `/run`, `$HOME`; one writable scratch bind | `--cap-drop ALL`, `--die-with-parent`, `--new-session` (no TTY injection). Preferred. |
| `unshare` (`unshare -rn --pid --fork`) | own netns, loopback only | own PID namespace | **not restricted**: the adapter can write wherever your user can | used when bubblewrap is missing; `sandbox_info` says so |
| `subprocess` | **not isolated** | shared | **not restricted** | *reduced isolation*: separate process, rlimits (Windows: a Job Object with a memory limit, no child processes, kill-on-close), scrubbed env and pickle guards only; `sandbox_info.warnings` starts with `REDUCED ISOLATION`. `auto` uses it **only** with `runtime.allow_reduced_isolation: true` / `--allow-reduced-isolation`; otherwise a host without bwrap/unshare is an error |
| in-process (`runtime.sandbox: false` / `--no-sandbox`) | none | none | none | debugging **trusted** models only; no pickle guards, and deadlines are only checked between chunks; `sandbox_info.warnings` starts with `SANDBOX DISABLED` |

In every mode the backend, what it isolated, and any warnings are recorded under `sandbox` in
`report.json` and shown in the HTML appendix. A reader can always tell how the model was run.
Requesting `bwrap` or `unshare` explicitly on a host where it does not work is an error. It never
falls back silently.

**Isolation level.** `sandbox.isolation` and `verdict.isolation` in `report.json` record how the
model ran: `os_sandbox` (bwrap or unshare, network isolated), `process_only` (the `subprocess`
backend: reduced isolation) or `none` (in-process debug mode). `null` means the model was never
loaded. Anything other than `os_sandbox` adds `verdict.isolation_warning`, appended to the verdict
summary, and a warning on the run page and in `report.html`. The verdict and score are unchanged.
Windows, macOS and Linux hosts without unprivileged user namespaces have no OS sandbox, so they can
only reach `process_only`. That needs the explicit opt-in, which the
[double-click launchers](../launchers.md) pass on those platforms only.

**Windows.** Pipe fds cannot be passed to a child there, so the worker talks over its own stdin and
stdout (`--stdio`). It moves fds 0 and 1 to `os.devnull` and the log before any submitted code runs.
The host reads with helper threads and a bounded buffer, so deadlines still apply. The same
transport can be forced on POSIX with `MALVALID_SANDBOX_IPC=stdio`, which the tests do.
`MALVALID_SIMULATE_NO_OS_SANDBOX=1` hides bwrap and unshare from the probe to test the
reduced-isolation path on Linux; it can only remove isolation backends, and runs still need the
opt-in.

## Lifecycle

1. `inspect_adapter(adapter, policy)` imports the adapter in a short-lived worker **without calling
   `load()`**. It finds the adapter class (an explicit `--class`, or the single class that defines
   `predict_proba`, `load` and `feature_version`), checks the declarations (types, threshold in
   [0, 1], a parseable cutoff, an existing training-hash file), resolves `model_path(s)`, and raises
   `AdapterError` listing every problem it found.
2. M0 scans the resolved artifacts. The model is never loaded before that scan passes.
3. `open_model(...)` starts the long-lived worker. The model libraries named by `model_kind` are
   imported first, then the pickle guards are installed, then the adapter is imported and
   `cls.load()` is called (limited by `startup_timeout_s`).
4. Modules call `predict_proba` / `predict`. Each call is split into `chunk_rows` requests and every
   reply is checked. `featurize(raw_bytes)` runs the adapter's `featurize`, or else the schema's
   extractor inside the worker. The harness process never parses a PE file.
   `tree_ensemble()` runs the adapter's `tree_ensemble()`, or else `native_model` through the
   model loaders, and sends the normalized trees back once. They are cached after that.
5. **Deadlines.** The runner calls `set_deadline()` for each module. A request still running at the
   deadline kills the worker and raises `ModuleTimeout`. The next call restarts the worker and
   reloads the model, so the following module is not affected.
6. **Crashes.** If the worker dies (segfault, `os._exit`, OOM kill), the call raises `SandboxError`
   with the exit reason and the last lines of `<run>/private/sandbox/worker.log`. The adapter's
   stdout and stderr, Python tracebacks and `faulthandler` dumps all go to that log.

## Debugging an adapter

* Start with `malvalid validate-adapter --adapter my_adapter.py`. It runs the same path as a real
  run (inspect → M0 scan → sandboxed load) and then probes the model on a few corpus rows, or on
  random vectors with the right shape if no corpus is available. It reports one line per check:
  `declarations`, `feature_schema`, `training_manifest`, `file_safety`, `load`, `predict_proba`,
  `batch_independence`, `predict` (must equal `predict_proba >= operating_threshold`),
  `determinism`, `featurize` (on a benign setuptools `.exe`), `tree_access` (with fidelity) and
  `sandbox`.
* Read `worker.log` in the run's `private/sandbox/` directory. Whatever your adapter prints ends up
  there.
* `PickleRefused` means the adapter or a library called `pickle.load(s)`, `joblib.load`,
  `numpy.load(allow_pickle=True)`, or a `dill` or `cloudpickle` loader. Re-export the model to a
  non-executable format (LightGBM `.txt`, XGBoost `.json`/`.ubj`, ONNX). If you trust the file,
  pass `--allow-pickle`: the run then records that pickle was allowed, and M0 reports at least
  `warn`.
* `--no-sandbox` runs the adapter in-process so you can use a debugger. Only do this with a model
  you trust. The report marks the run as unsandboxed.

## Multi-tenant use

bubblewrap on a shared node is a reasonable default when researchers run the gate on their own
models. **If you gate models submitted by other people** (a shared service, a competition, a model
marketplace), also run the whole of MalValid inside a disposable container or VM with a stronger
boundary: gVisor (`runsc`), Kata/Firecracker microVMs, or at least rootless Podman/Docker with a
seccomp profile, no network, a read-only image and a per-run volume. Inside such a container,
keep `sandbox_backend: auto` so the worker is still separated from the harness. Never set
`--allow-pickle` for a model whose origin you cannot verify.
