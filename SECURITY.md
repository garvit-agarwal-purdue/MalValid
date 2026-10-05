# Security policy

MalValid loads and runs **untrusted models and adapters** by design, so its security behaviour matters.
This page describes how to report a problem and what the tool does and does not protect against.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Report it privately through GitHub's
private vulnerability reporting: open
<https://github.com/garvit-agarwal-purdue/MalValid/security/advisories/new>, or use the
**Report a vulnerability** button on the repository's **Security** tab. Only the maintainers can see
the report. If that page is not available, open a public issue that only asks for a private contact
(no details of the problem), and a maintainer will reach out.

Include the MalValid version (`malvalid --version`), your OS, a minimal reproduction (a model file or
adapter that triggers it, built from your own synthetic data, never real malware), and the impact you
see. You should get an acknowledgement within a week. Fixes are released as a patch version with a note
in `CHANGELOG.md`.

Examples of what we want to hear about: a way for a model or adapter to escape the sandbox, write
outside the run directory, reach the network, or read the host environment; a pickle that gets past the
refusal or the pickle policy; a path traversal or request forgery in the web UI; a bypass of the web
UI's token, `Host`/`Origin` checks or upload limits.

## Threat model

The main use case is a researcher gating their **own** detector. Running a model **someone else** sent
you is harder; the defences below raise the cost of an attack but are not a hardened security boundary.
Run such models in a throwaway VM or container as well.

* **Sandbox.** The adapter and model are imported, loaded and queried only in a separate worker process
  with a minimal environment (your tokens and `SSH_AUTH_SOCK` do not reach it) and memory/file limits.
  With `bubblewrap` the worker also has no network, a read-only root file system with `$HOME` hidden and
  one writable scratch directory. With the `unshare` fallback (user namespaces, no bubblewrap) the network
  is isolated but the file system is **not** restricted. Check your host with `malvalid sandbox-check`.
  The wire format between the harness and the worker never unpickles anything and every reply is
  validated.
* **Reduced isolation.** On hosts without an OS sandbox (Windows, macOS, Linux without bubblewrap or
  unprivileged user namespaces), `malvalid run` refuses by default. With `--allow-reduced-isolation` (or
  `runtime.allow_reduced_isolation: true`), which the launchers pass on such hosts, the model runs in a
  plain worker process: pickle refusal, the no-unpickle channel, a scrubbed environment and resource
  limits, but **no network or file-system isolation**. Reports record `isolation: process_only` and the
  UI and `report.html` show a warning. Use it only for models whose origin you trust.
* **Pickle refusal.** Pickle-based artifacts (`.pkl`, `.joblib`, `.pt`, ...) are refused unless you pass
  `--allow-pickle`, and even then every global a pickle imports is checked against a deny policy. M0
  (file safety, built on modelscan plus MalValid's own opcode scan) inspects the file before anything
  is deserialized. Never pass `--allow-pickle` for a model whose origin you cannot verify.
* **No-sandbox is off by default.** `--no-sandbox` (and `runtime.sandbox: false`) runs the model in the
  harness process with your privileges. Use it only for models you trust. In the web UI the matching
  checkbox is hidden and rejected unless the server was started with `--allow-no-sandbox`.
* **Path mode is off by default.** The web UI's "use files on this machine" mode lets the server read
  files by path. It is hidden and rejected unless the server was started with `--allow-path-mode`.
* **Web UI.** `malvalid serve` binds to loopback, requires a random access token, enforces `Host` and
  `Origin`/`Referer` checks and CSRF tokens, and sends a strict Content-Security-Policy. `--allow-remote`
  weakens the network exposure; keep it behind a trusted proxy and use `--allow-client`. See `docs/web.md`.
* **Out of scope.** Kernel and container escapes, side channels, and vulnerabilities in third-party
  native libraries (LightGBM, XGBoost, onnxruntime) shared with the host installation. Report those
  upstream.

## MalValid never ships or generates malware

MalValid works on feature vectors, hashes, labels and timestamps. It contains no malware samples and no
tools to create them, and the examples and tests do not need any. Please do not attach malware to issues
or pull requests. If a report needs a sample, describe how to obtain it from a public source instead.
