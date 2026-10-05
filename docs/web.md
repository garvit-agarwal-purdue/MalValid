# The MalValid web UI (`malvalid serve`)

`malvalid serve` starts a small **local, single-user** web UI on your own machine. Use it to submit a trained
malware detector (just your model file; a Python adapter is the advanced path), watch the run module by module, read the
production-readiness verdict and its 0–100 score with the per-axis evidence, and browse and compare earlier
runs.

The UI is a front end over the same engine as the command line. Every run it starts is the same
`malvalid run` you would type in a terminal, executed in a subprocess with the same sandbox and gate policy.
**The web server process never imports your adapter, loads your model or unpickles anything.** (Inspecting a
model file runs `malvalid inspect-model` in a subprocess, which reads the file as data and never unpickles it.) It reads
uploads as bytes and YAML (`yaml.safe_load`) only.

## Contents

1. [Starting the UI](#1-starting-the-ui)
2. [Security model](#2-security-model)
3. [Submitting a run](#3-submitting-a-run)
4. [Watching a run](#4-watching-a-run)
5. [Reading results](#5-reading-results)
6. [Comparing runs](#6-comparing-runs)
7. [The runs directory](#7-the-runs-directory)
8. [Scripting the UI](#8-scripting-the-ui)
9. [Limitations](#9-limitations)

## 1. Starting the UI

The UI needs the `web` extra (Starlette, uvicorn, python-multipart). MalValid is installed from a source
checkout (it is not on PyPI); from the repository root:

```bash
pip install -c constraints/lock-py311.txt -e '.[web]'
malvalid serve
```

The double-click launchers ([`launchers.md`](launchers.md)) do this for you.

It prints a private link and opens it in your browser (unless you pass `--no-browser` or there is no
display):

```
malvalid web UI <version> — open this private link in your browser:
  http://mg-3f9c1a0e5b7d2c44.localhost:8765/?token=Q2m…
If your browser cannot open *.localhost addresses, use this one instead (its sign-in cookie is also sent to other services on 127.0.0.1):
  http://127.0.0.1:8765/?token=Q2m…
Runs directory: /home/you/project/malvalid-runs
Keep the link private: it signs in whoever opens it. Press Ctrl+C to stop (queued and running runs are stopped too).
```

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Interface to listen on. Only loopback addresses are accepted unless you pass `--allow-remote`. |
| `--port` | `8765` | TCP port. `0` picks a free port. The printed link shows the port actually used. |
| `--find-free-port` | off | If `--port` is in use, listen on the next free port above it (up to 100 tried) instead of failing. The [double-click launchers](launchers.md) use it to prefer 8765. |
| `--runs-dir` | `./malvalid-runs` | Where web runs are stored. This is the same default as `malvalid run --out`, and runs that `malvalid run` writes here are listed too. |
| `--config` | packaged policy | Default gate policy (YAML) for new runs. A run's form can bring its own policy. |
| `--max-concurrent` | `1` | Runs executed at the same time. Runs are heavy, so the default is one at a time; later submissions wait in a first-in, first-out queue. |
| `--max-upload-mb` | `4096` | Per-file upload limit, enforced while the file streams in. |
| `--allow-remote` | off | Permit a non-loopback `--host` (see [§2](#2-security-model)). |
| `--no-browser` | off | Do not open a browser. |
| `--token` | random | Use this access token instead of a fresh per-launch secret. For scripts and tests only: 1–512 URL-safe characters, with a warning below 16. Also read from the `MALVALID_SERVE_TOKEN` environment variable. Other users of the machine can see command lines (`ps`), so prefer the variable or `--token-file`. |
| `--token-file` | none | Keep the access token in this file so the login link survives restarts. If the file exists, its first line is the token (16–512 URL-safe characters: `A-Z a-z 0-9 . _ ~ -`), otherwise startup fails with exit 2 (the message never shows the token; delete the file to regenerate). If it does not exist, a fresh random token is generated and the file is created with mode `0600` (never overwriting an existing file or following a symlink). A warning is printed if an existing file is readable by group or others. Keep it in a directory that is not world-readable, for example `~/.config/malvalid/serve-token`. |
| `--allow-path-mode` | off | Enable **Use files on this machine** (`mode=path`): the server reads model, adapter, training-hash and policy files from its own file system by path. Off by default: the option is hidden and the server rejects `mode=path` with HTTP 403 (on `POST /runs`, `POST /runs/inspect` and `POST /api/validate`). The bundled synthetic demo button still works. Prints a warning at startup. It also unlocks the **Corpus directory** field: without the flag the field is hidden and any `corpus_dir` other than the server's own corpus directory (`$MALVALID_CORPUS_DIR`, else `~/.cache/malvalid/corpora`) is rejected with HTTP 403 (a `corpus_dir` inside an uploaded policy with 400), so a client cannot make the server read arbitrary directories. See [§2](#2-security-model). |
| `--allow-no-sandbox` | off | Enable the **Run without the sandbox** checkbox. Off by default: the checkbox is hidden and any request with `no_sandbox` set is rejected with 403. `malvalid run --no-sandbox` on the command line is unaffected. Prints a warning at startup. |
| `--allow-reduced-isolation` | off | On a machine with **no OS sandbox** (Windows, macOS, Linux without bubblewrap or user namespaces), run models in a plain worker process (no-unpickle channel, pickle refusal and resource limits, but **no network or file-system isolation**) instead of failing. Passed to every `malvalid run` / `validate-adapter` subprocess; it never downgrades a machine where the sandbox works. A startup panel, a banner on New run and System, the run page and `report.html` show it, and `report.json` records `isolation: process_only` in `verdict` and `sandbox`. The [launchers](launchers.md#6-isolation-on-windows-and-macos) pass it only where it is needed. |
| `--root-path` | none | Public URL path prefix when the UI is reached through a reverse proxy, such as `/node/<host>/<port>` on Open OnDemand. See [Behind a proxy](#behind-a-proxy-open-ondemand--browser-vs-code). |
| `--trusted-host` | none | Public host name of that proxy (repeatable). It is added to the `Host` and `Origin`/`Referer` allowlist; the access token is still required. The first one is used in the printed proxy link. |
| `--allow-client` | any | Serve only TCP connections from this IP address or CIDR network (repeatable; loopback is always allowed), for example the proxy's address. |
| `-v` | off | Debug logging. |

`malvalid serve` exits with 0 after a normal shutdown (Ctrl+C). It exits with 2 for a usage error, a port
already in use, or missing web dependencies (the message says to run `pip install -e '.[web]'` from the source folder).

**On a remote machine** (a lab server or a cluster node), keep the default loopback bind and tunnel the port
over SSH instead of using `--allow-remote`:

```bash
ssh -L 8765:127.0.0.1:8765 you@gpu-box      # then, on gpu-box: malvalid serve --no-browser
# open the printed http://mg-….localhost:8765/?token=… link in your local browser
```

Chrome, Edge and Firefox resolve every `*.localhost` name to your own machine, so the printed link works
through the tunnel. If your browser does not, use the printed `127.0.0.1` link instead.

### Behind a proxy (Open OnDemand / browser VS Code)

If you cannot forward ports — for example you work in VS Code in the browser, started from an
[Open OnDemand](https://openondemand.org/) (OOD) portal on a cluster — reach the UI through the portal's
node proxy instead. OOD forwards `https://<portal>/node/<node>/<port>/…` to `http://<node>:<port>/node/<node>/<port>/…`
(the prefix is kept) and `https://<portal>/rnode/<node>/<port>/…` to `http://<node>:<port>/…` (the prefix is
stripped), after its own single sign-on. Your VS Code session uses the same mechanism: its address is
`https://<portal>/node/<node>/<vscode-port>/`.

The portal web server connects to the node over the cluster network, not over loopback, so the UI must listen
on the node's cluster address. On the compute node:

```bash
NODE=$(hostname -f)                                  # e.g. node07.cluster.example.edu
ADDR=$(getent ahostsv4 "$NODE" | awk 'NR==1{print $1}')   # the address the portal connects to
PORTAL=ondemand.example.edu                          # your OOD portal host name
PORT=8765
malvalid serve --no-browser --port $PORT --host $ADDR --allow-remote \
    --root-path /node/$NODE/$PORT --trusted-host $PORTAL --trusted-host $NODE \
    --allow-client "$(getent ahostsv4 $PORTAL | awk 'NR==1{print $1}')" \
    --token-file ~/.config/malvalid/serve-token
```

**One command per OOD job.** OOD jobs are short and land on different nodes, so `scripts/serve_ood.sh [PORT]`
does all of the above for the node you are on: it detects the short and fully qualified host name and the
node's address, uses the root path `/node/<fqdn>/<port>` (port: `$1`, else `$PORT`, else 8765), trusts the
portal and the node host name, allows only the portal's IP and the node, creates the token file on first
use (mode `0600`) and reads it afterwards, stops this user's previous `malvalid serve` on that port, starts the
new server detached and prints `https://<portal>/node/<fqdn>/<port>/?token=…` to the terminal only (the log
has the token redacted). The script has no site-specific defaults: you must set `MG_PORTAL_HOST`. Other
environment variables (all optional): `PORT`, `MALVALID_BIN`, `MG_PORTAL_IP` (default: resolved from the
portal host name), `MG_NODE_IP` (default: resolved from the node's host name), `MG_TOKEN_FILE`
(default `~/.config/malvalid/serve-token`), `MG_RUNS_DIR` (default `~/malvalid-runs`), `MG_LOG` (default
`~/.local/state/malvalid/serve.log`), `MALVALID_CORPUS_DIR` (passed through when set) and `MG_EXTRA_ARGS`
for more `malvalid serve` flags.

```bash
export MG_PORTAL_HOST=ondemand.example.edu
scripts/serve_ood.sh            # port 8765
scripts/serve_ood.sh 8800       # or: PORT=8800 scripts/serve_ood.sh
```

`--token-file` keeps the login link stable across restarts (the file is created with mode `0600` on first start). Path mode and no-sandbox stay off unless you pass `--allow-path-mode` or `--allow-no-sandbox`; leave them off on a shared portal origin.

It prints the link to open, `https://<portal>/node/<node>/<port>/?token=…`. The same server also answers
`https://<portal>/rnode/<node>/<port>/…` if you pass `--root-path /rnode/$NODE/$PORT` instead; requests are
routed with or without the prefix, so either proxy style works. What each flag does:

* `--root-path` puts the prefix on every link, form action, asset, iframe, JavaScript request and redirect,
  and scopes the session cookie to `Path=<prefix>/`, so other apps on the portal (another port, another
  user's session) never receive it. `<prefix>` alone redirects to `<prefix>/`.
* `--trusted-host` adds the portal's host name to the `Host` header allowlist (DNS-rebinding defence) and to
  the `Origin`/`Referer` check of state-changing requests. A bare name matches `name` and `name:<port>`;
  OOD's Apache normally sends `Host: <node>:<port>` (hence `--trusted-host $NODE`), and the browser sends
  `Origin: https://<portal>`. `X-Forwarded-*` headers are ignored.
* `--allow-remote` with `--host $ADDR` is needed because the portal connects from another machine; bind the
  cluster address rather than `0.0.0.0`. `--allow-client` then refuses every TCP peer except the portal and
  loopback, so other users of the cluster network cannot reach the server even with a guessed link.

Nothing else changes: the token login is still required (the portal's sign-in is in addition to
it, not instead of it), as are the CSRF token, the Content-Security-Policy and the other headers. The token is
also still the only protection on the hop between the portal and the node, which is plain HTTP on the
cluster network (as it is for the VS Code session itself).

Caveat: every OOD app is served from the same portal origin (`https://<portal>`). Pages of another app on that
portal — including another user's app, if you open their link while signed in — are same-origin with this UI.
Do not open untrusted `/node/…` or `/rnode/…` links in the browser where you are signed in to `malvalid serve`,
and stop the server (Ctrl+C or `kill`) when you are done.

If the page answers "unexpected Host header", the portal sends a `Host` value that is not allowed yet: the
server log (`malvalid serve` output) shows the refused value; add it with `--trusted-host`. If it answers
"only accepts connections from its configured reverse proxy", the log shows the refused peer address.

Environment variables of the server are inherited by every run. For example, `MALVALID_CORPUS_DIR` (where
the canonical corpora live) applies to web runs exactly as it does to `malvalid run`. Runs (and **Validate
adapter**) execute in the directory where you started `malvalid serve`, so a relative path in a policy, such as
`corpus_dir: corpora`, means the same as for `malvalid run` typed in that terminal. Relative paths you type into
the form are resolved against that directory too.

## 2. Security model

The UI can run arbitrary submitted models on your machine, so it is locked down by default:

* **Loopback only.** It binds `127.0.0.1` unless you choose otherwise. A non-loopback `--host`, including
  `0.0.0.0`, is refused unless you also pass `--allow-remote`, and even then it prints a loud warning.
  Remote mode has no TLS: the link, your uploads and your results travel in clear text, and the token is the
  only protection. Use it only on a trusted network. An SSH tunnel is almost always the better choice.
* **Access token.** Each launch generates a fresh secret (`secrets.token_urlsafe(32)`) unless you use
  `--token-file` (the token then persists in that file, mode `0600`). Opening
  the printed `/?token=…` link signs the browser in: the server creates a random session id, stores it in an
  `HttpOnly`, `SameSite=Strict` cookie named `malvalid_session_<port>` (so two servers on one machine do not
  overwrite each other's sign-in), then redirects to the same page *without* the token, so the token does not
  linger in the address bar or browser history. The token itself is never stored in a cookie. Every page and
  API call except `GET /healthz` and `/static/*` requires a live session cookie or an
  `Authorization: Bearer <token>` header. Secrets are compared in constant time. A browser without the
  cookie gets a page telling it to open the printed link. Restarting the server signs every browser out (sessions
  are in memory) and, unless `--token-file` is used, invalidates old links; with `--token-file` the browser
  signs in again through the same `?token=` link.
* **A private host name per launch.** On a loopback bind the printed link uses a random name such as
  `mg-3f9c1a0e5b7d2c44.localhost`. Browsers send cookies to every port of the same host name, so a sign-in
  cookie for `127.0.0.1` would also reach any other program listening on `127.0.0.1`. The cookie for the
  random name reaches only this server. The name is never shown to a browser that has not signed in. The
  fallback `127.0.0.1` link works too, with the weaker cookie scoping.
* **No token on the browser's command line.** When `malvalid serve` opens your browser, it passes a
  one-time sign-in link (`/?login=<nonce>`, valid once and for 2 minutes) instead of the token, because
  other users of the machine can read command lines with `ps`. Pass `--token` through `MALVALID_SERVE_TOKEN`
  or `--token-file` for the same reason.
* **Path mode and no-sandbox are off by default.** Behind a shared-origin reverse proxy such as Open OnDemand
  the UI is more exposed than on localhost, so `mode=path` (the server reads files from its own file system)
  and the no-sandbox checkbox need `--allow-path-mode` / `--allow-no-sandbox`. Without them the options are
  hidden and the server answers 403 on `POST /runs`, `POST /runs/inspect` and `POST /api/validate`. Both flags
  print a startup warning and show on the System page. The bundled synthetic demo (fixed server-side paths)
  works without `--allow-path-mode`.
* **Host-header allowlist** (DNS-rebinding defence). Requests are answered only when addressed to the
  per-launch `*.localhost` name, `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>` (plus the bound
  host with `--allow-remote`). Anything else gets `400`.
* **CSRF protection.** Every state-changing request (submit, validate, cancel, delete) must carry a CSRF
  token, either in the `X-CSRF-Token` header or the `csrf` form field. The token is an HMAC of the access
  token (`hmac_sha256(token, "csrf")`, hex). If an `Origin` or `Referer` header is present it must be on the
  host allowlist, and cross-site `Sec-Fetch-Site` requests are refused.
* **Strict page headers.** App pages carry `Content-Security-Policy: default-src 'self'; script-src 'self';
  style-src 'self'; …; frame-ancestors 'self'; base-uri 'none'`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer` and `Cache-Control: no-store`. They use no inline scripts or styles and no
  external assets, so the UI works offline. The run's `report.html` is served as-is with its own CSP (it
  pins its single inline script by hash). It is embedded in a sandboxed iframe
  (`sandbox="allow-scripts"`, no same-origin access) and is also sandboxed by a response header when opened
  directly.
* **Uploads.** Files land in `<runs_dir>/submissions/<submission_id>/` under sanitized names: directories are
  stripped, names are reduced to `[A-Za-z0-9._-]`, and empty or dot-leading names are refused. Each field has
  an extension allowlist:

  | Field | Accepted |
  |---|---|
  | Adapter | `.py` |
  | Model files | `.txt .json .ubj .model .onnx .pkl .pickle .joblib .jbl` |
  | Training manifest | `.txt .csv .tsv` |
  | Gate policy | `.yaml .yml` |

  The size cap (`--max-upload-mb`, per file) is enforced while the upload streams to disk. An oversized
  upload is stopped and its partial file deleted. A refused submission leaves nothing behind.
* **Pickles are refused by default.** A pickle-based model file (by extension *or* by content, so a pickle
  renamed to `model.txt` is still caught) is rejected unless you tick **Allow pickle-based model files**.
  Loading a pickle can run arbitrary code. Even when allowed, the file is loaded only inside the sandbox,
  after the M0 file-safety scan. Prefer formats that cannot execute code: LightGBM `.txt`, XGBoost
  `.json`/`.ubj`, ONNX.
* **Isolation of runs.** Each run is a separate `malvalid run` process in its own session (process group),
  started in the directory where `malvalid serve` was launched with `PYTHONSAFEPATH=1` (so that directory is
  not on the import path), and sandboxed as `malvalid run` always is.
  **Run without the sandbox** (`--no-sandbox`) needs `--allow-no-sandbox` on the server and an explicit second confirmation, because it loads the
  model in the run process with no isolation.
* **Only known files are served.** Run ids are validated (`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`) and resolved
  strictly inside the runs directory. Symlinks are never followed out of it. For each run, only
  `report.html`, `report.json`, `run.log` and `console.log` can be downloaded. Nothing under a run's
  `private/` directory (where modules keep sensitive by-products) is ever served.

What the token does *not* protect: anyone who can read your runs directory can read your results and
uploads, and anyone who has the link can run models as you. Keep the link private and the runs directory
readable only by you.

**Shared machines** (login nodes, shared GPU boxes, remote desktops). Other users of the machine can connect
to every loopback port, including this server's. They cannot get past the token, but they can run their own
listener on another `127.0.0.1` port. If your browser runs on that same machine and you signed in with the
`127.0.0.1` fallback link, your session cookie is also sent to their listener whenever your browser loads one
of its pages. Use the printed `*.localhost` link, or run the browser on your own computer through an SSH
tunnel (see above). Anyone holding a session can do everything you can in the UI, including path-mode runs
that execute an adapter as your user.

## 3. Submitting a run

Open **New run**. The form opens in `model` mode (**Upload your model**), the primary flow. The other two
modes are **Use files on this machine** (`path`) and **Advanced: own adapter** (`upload`). A POST that
carries no `mode` field is read as `upload` when it has an `adapter_file`, as `model` when it has a
`model_file` (or the id of an inspected one), and as `upload` otherwise (the pre-model-file API).

### Upload your model (primary)

1. **Drop the model file** (LightGBM `.txt`/`.model`, XGBoost `.json`/`.ubj`, ONNX `.onnx`).
2. Click **Inspect model** (it submits the form to `POST /runs/inspect`, which stores the file as an
   inspected upload and re-renders the page; send `Accept: application/json` for a JSON answer). The server runs `malvalid inspect-model` on the upload in a subprocess and
   fills in the **model kind** and **feature version** dropdowns, marked "detected from model". Detection
   reads the file as data: the model kind comes from its content (so `EMBER2024_PE.model` is recognised as
   LightGBM), and the input-feature count maps to a feature version (2381 is `ember_v2`, 2568 is
   `ember_v3`; any other count is reported as an error). You can change either dropdown. The corpus follows
   the feature version unless you pick one. For pickle or joblib files nothing is detected (they are never
   unpickled outside the sandbox): choose the kind and feature version yourself and tick **Allow
   pickle-based model files**.
3. **Choose the threshold** (`threshold_mode`): **Calibrate it for me** (`calibrate`, the default, with
   `calibrate_fpr` of 0.1%, 0.5% (preselected) or 1%) or **Use the threshold I ship** (`declared`, with
   `threshold` in [0, 1]). With no threshold given, MalValid calibrates at 0.5% FPR. A calibrated threshold
   is fitted on a held-out slice: about 10% of the benign rows of the corpus evaluation split (minus
   training hashes), which is disjoint from, and excluded from, every test in that run. By default these
   are the earliest benign rows by date, so the threshold is tested only on later data, as when deployed
   (`runtime.calibration_period: earliest` in the gate policy; `uniform` picks them by file hash across
   the whole period instead). The run page shows the rule and the dates of the calibration rows.
   It is not your production operating point; enter your own value for a verdict on the cut-off you ship.
   If the target FPR cannot be met (benign calibration scores saturate at 1.0), the run fails with a clear
   error asking for an explicit threshold or a higher calibration FPR; it never silently exceeds the target. See [`docs/modules/model_submission.md`](modules/model_submission.md).
4. Optionally add a **training-cutoff month** (`training_cutoff`, unlocks M2 drift), a **training-hash list**
   (uploaded as `manifest_file` in model mode; `training_hashes_path` in path mode; one sha256 per
   line, or CSV/TSV with a `sha256` column; unlocks M4 membership inference and keeps training members out
   of evaluation), a **gate-policy YAML**, and **fail-on**. A field you leave out skips the test that needs
   it, which lowers coverage and the score.
5. **Start run.** The page shows the equivalent `malvalid run --model ...` command.

### Advanced: own adapter

Use this when the built-in loaders are not enough (your own featurization, preprocessing, several model
files). Choose your adapter `.py`, your model file(s), optionally the training manifest and a gate policy
YAML. All files land in one submission directory, so relative paths in your adapter
(`model_path = "model.txt"`, `training_hashes_path = "train_hashes.txt"`) resolve exactly as they do next to
your adapter at home. **Validate adapter** (below) applies to this path.

### Path mode

Only available when the server was started with `--allow-path-mode`. Point at a model file or an adapter that is already on this machine (the path must exist), and optionally
at a policy YAML. Nothing is copied. The run uses your files in place, which suits large models.

The rest of the form maps onto `malvalid run` flags. The page shows the equivalent terminal command.

| Form field | `malvalid run` flag | Notes |
|---|---|---|
| Model file (`model_file`) | `--model` | Model mode. Pickles need **Allow pickle-based model files** (opt-in). |
| Model kind / Feature version (`model_kind`, `feature_version`) | `--model-kind` / `--feature-version` | Pre-filled by **Inspect model**; required for pickles. |
| Threshold (`threshold_mode`, `threshold`, `calibrate_fpr`) | `--threshold` or `--calibrate-fpr` | Known value, or calibrate at 0.1% / 0.5% / 1% FPR (default 0.5%). |
| Training cutoff / Training hashes (`training_cutoff`, `manifest_file` or `training_hashes_path`) | `--training-cutoff` / `--training-hashes` | Optional; unlock M2 and M4. |
| Adapter class | `--class` | Adapter submissions only; needed when the file defines several detector classes. |
| Canonical corpus | `--corpus` | Defaults to the policy's `corpus`. The **Corpora** page shows which corpora are available on this machine. |
| Corpus directory | `--corpus-dir` | The corpus itself or a root containing `<corpus>/`. |
| Seed | `--seed` | Leave it empty to use the policy's `runtime.seed` (a value you enter overrides it). Recorded in the report. |
| Run only / Skip | `--only` / `--skip` | Module ids or codes (`M1`, `drift`, …). M0 always runs. |
| Fail on | `--fail-on` | Which verdicts make the CI exit code non-zero (`blocked`, `not_ready`, `conditional`). |
| Allow pickle-based model files | `--allow-pickle` | See [§2](#2-security-model). |
| Run without the sandbox | `--no-sandbox` | Needs `--allow-no-sandbox` and the confirmation box. |
| Run title | policy `report.title` | Shown on the dashboard and in the report. Passed to `malvalid run` with the internal `--title` flag; your policy file is used unchanged, so relative paths in it (such as `triggers_path`) resolve exactly as from a terminal. Without a title, the dashboard shows the detector class read from the adapter's text and the file name. |

The EMBER corpora (`ember_v2_2018`, `ember_v3_2024`) are not shipped and are absent on a fresh machine. Choosing one that is not built (explicitly, or implicitly by submitting a model file of that feature version) is refused with HTTP 400 before a run is queued; the message says the corpus is not on this machine, that the synthetic demo and `synthetic_v2` / `synthetic_v3` still work with no download, and shows the provider's `malvalid corpus build ...` steps. The **Corpora** page shows the same steps for each corpus that is not installed.

A form with problems comes back with the errors listed and your values kept. File inputs must be chosen again,
because browsers do not allow a page to pre-fill them. In adapter upload mode, the form warns (and asks before
starting) when no model file is chosen: your adapter's `model_path` must be one of the uploaded files.

**Re-run with these settings** on a run's page opens New run with that run's options filled in (path mode: the
same adapter and policy paths, or model mode with a notice when `--allow-path-mode` is off; upload mode: choose the files again). The no-sandbox confirmation is never
copied. New to MalValid? The empty dashboard and New run show a copyable **adapter skeleton**, and in a
source checkout installed in editable mode (`pip install -e`, as the launchers do) a **Run the synthetic
demo** button submits `examples/synthetic_demo`, whose small trained model ships with the repository (a
pipeline check on synthetic data, not evidence about a real model). A non-editable install cannot find the
`examples/` folder, so the button is not shown there.

**Validate adapter** runs `malvalid validate-adapter --json` on the same inputs in a subprocess (timeout
600 s). It checks the declarations, scans the artifact, loads the model in the sandbox and probes
`predict_proba` / `predict` / `featurize` / tree access, then shows the check list without starting a run.
Uploads made only for validation are deleted afterwards.

Submitting creates a run id of the form `YYYYMMDDTHHMMSSZ-<8 hex>` and takes you to the run's page. The same id
names the run inside `report.json`, `report.html` and `progress.json`: the UI passes it to `malvalid run` with
the internal `--run-id` flag, which the terminal command shown on the run page leaves out (a re-run from a
shell gets its own id). If another run is executing, the new one waits in the queue.

## 4. Watching a run

The run page polls the server every 2 seconds while the run is queued or running. It shows the current stage
(`starting → inspecting → scanning → loading → modules → verdict → writing → done`), the module that is
running, and each module's status as it finishes (`pending`, `running`, `pass`, `warn`, `fail`, `skipped`,
`error`). The page reloads by itself when the run ends. Everything the run printed, including your
adapter's own output and any traceback, is in `console.log`, linked from the run page. `GET /api/runs/<id>`
also returns its last 40 lines as `console_tail`, and the run page shows them live under "Latest output" (a
collapsed panel; useful during long steps such as TreeSHAP on EMBER, which can take several minutes).

`malvalid run` writes this stage information to `<run_dir>/progress.json` (schema `malvalid-progress/1`). It
does this for terminal runs too, so the UI can also follow a `malvalid run --out <runs_dir>/<name>` started
from a shell.

A queued run shows how long it has waited and which runs are ahead of it; its elapsed clock restarts when it
actually starts. Runs adopted from an earlier server (see [§7](#7-the-runs-directory)) occupy a worker slot,
so `--max-concurrent` is never exceeded.

**Cancel** stops a queued run immediately. For a running one, it sends SIGTERM to the run's whole process
group, then SIGKILL if the run is still alive 10 seconds later. On Windows each run is started in its own
process group; Cancel sends it CTRL_BREAK_EVENT, then ends the whole process tree with `taskkill /F /T`. The run ends as `cancelled`. Runs started with
`malvalid run` in a terminal cannot be cancelled from the UI; stop them in their terminal.

## 5. Reading results

A finished run shows the **verdict** (`ready`, `conditional`, `not_ready`, `blocked`) with its 0–100 score
and coverage, the blocking issues and the reasons behind the verdict, and the per-axis scorecard. The full
`report.html` is embedded below. `report.json`, `run.log` and `console.log` are one click away. The report is
the same file `malvalid run` writes; see its disclaimers for what the verdict does and does not claim.

A web run's status (also shown on the dashboard) is one of:

| Status | Meaning |
|---|---|
| `queued` | Waiting for a free worker. |
| `running` | The `malvalid run` process is executing. |
| `finished` | `malvalid run` exited 0 (gate passed) or 1 (gate failed) and wrote `report.json`. Read the verdict. |
| `failed` | Exit code 2 (the gate result is not trustworthy, e.g. a module errored or the model could not be loaded), no report was written, or MalValid crashed (a signal such as SIGSEGV) even after writing a complete report. The page shows the cause: the error line, the adapter-contract problems as a list, and any traceback collapsed. |
| `cancelled` | You cancelled it. |
| `interrupted` | The server stopped (or restarted and found the process gone) before the run completed; for a terminal run, its process ended before it finished. |

The dashboard lists the newest 200 runs in the runs directory (with a **Show all** link), newest first, with
a verdict summary over all of them (ready / conditional / not ready / blocked / running / failed). A run that
ended with exit code 2 counts as failed even when its report carries a Blocked verdict. Runs written by
`malvalid run` from a terminal are labelled as CLI runs; a folder whose name has characters outside
`[A-Za-z0-9._-]` (for example `my model v2`) is listed under an `enc-<hex>` id. Runs on a synthetic corpus are
badged **Synthetic data**. While runs are active, the dashboard polls `GET /api/runs?active=1` and reloads
when that set changes.

When a second `malvalid run` writes into an `--out` folder that already holds a report, the old report is
ignored (its run id differs from the new run's `progress.json`), so the dashboard never shows the earlier
verdict as the current result. A run folder whose files cannot be read is listed as failed with a note,
instead of disappearing.

## 6. Comparing runs

Tick two to six runs on the dashboard and choose **Compare selected**, or open `/compare?ids=a,b,c`. The compare view
has one row per module and one column per run. Each cell shows the module's status, its axis score (0–100),
the gate outcome and the module's key metric against its threshold. When one of a module's key checks
failed, that check leads the cell (an M1 that fails on detection shows `detection_rate`, not the passing
`fpr`):

| Module | Key metric |
|---|---|
| M0 file safety | The first failed check, else `n_critical`. |
| M1 performance | `fpr` vs `max_fpr`, or `detection_rate` vs `min_detection` when that failed. |
| M2 drift | `aut_f1_weighted` vs `min_aut_f1`. |
| M4 membership inference | Worst-case `advantage` vs `max_advantage`. |
| M5 backdoor screen | `n_flagged_rules` (static tree scan), else `max_trigger_drop`. |
| M6 extraction | Surrogate `fidelity` at `fidelity_budget` queries vs `max_fidelity`. |
| M7 explanation | `controllable_share` vs `max_controllable_share`, or `top_feature_share` when that failed. |

At the bottom of the table, **Gate-policy and option differences** lists every gate-policy setting that differs between the runs: a
threshold, a disabled module, the corpus build (`corpus.content_hash`) or `fail_on`. Above the table, one line
per run states its verdict and why (for example "Blocked, because M1 Performance (hard gate) failed"). When
the runs used the same corpus and the same scoring policy, the best run (verdict first, then score, then
coverage) is marked **Highest score**. When any threshold, weight, band, gate or enabled test differs, or the
corpora differ, a warning at the top says the scores are not comparable and no run is marked.

## 7. The runs directory

```
<runs_dir>/
  <run_id>/                    one per run (web runs and `malvalid run --out <runs_dir>/<name>` runs)
    job.json                   web job record (absent for terminal runs)
    progress.json              stage file written by the runner
    report.json report.html    the report
    run.log console.log        the runner's log and the subprocess's stdout/stderr
    private/                   sensitive by-products; never served
  submissions/<submission_id>/ uploaded adapter / model / manifest / policy of a web run
```

JSON files are written atomically. **Delete** (on a run's page) removes the run directory and its submission
directory. It is refused while the run is queued or running, and for directories that contain files MalValid
did not create (delete those by hand).

When the server starts, it reconciles `job.json` files that a previous server left `queued` or `running`. A
run whose process is still alive is adopted and watched until it exits. A dead one becomes `interrupted`, or
`finished` if it completed its report while no server was running. Runs are never restarted automatically;
submit again to rerun.

## 8. Scripting the UI

For scripts, pass the token as a bearer header. POSTs also need the CSRF header, which you can compute from
the token:

```bash
TOKEN=…   # the token from the printed link, or pass your own with --token
CSRF=$(python -c 'import hashlib, hmac, sys; print(hmac.new(sys.argv[1].encode(), b"csrf", hashlib.sha256).hexdigest())' "$TOKEN")
URL=http://127.0.0.1:8765

curl -s -H "Authorization: Bearer $TOKEN" $URL/api/runs                        # [RunSummary]
curl -s -D- -o /dev/null -H "Authorization: Bearer $TOKEN" -H "X-CSRF-Token: $CSRF" \
     -F mode=model -F model_file=@my_model.txt $URL/runs                       # 303, Location: /runs/<id>
# mode=path (-F adapter_path=/abs/path/...) needs `malvalid serve --allow-path-mode`, else 403
curl -s -H "Authorization: Bearer $TOKEN" $URL/api/runs/<id>                   # {summary, job, progress, …}
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "X-CSRF-Token: $CSRF" $URL/api/runs/<id>/cancel
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "X-CSRF-Token: $CSRF" $URL/api/runs/<id>/delete
```

`POST /runs/inspect` takes `model_file` (or `model_submission`, the id of an earlier inspection, or in path mode
`adapter_path`) and returns the detected model kind, feature version and feature count (JSON with
`Accept: application/json`). Model-mode runs use `-F mode=model -F model_file=@model.txt -F threshold_mode=calibrate
-F calibrate_fpr=0.005` on `POST /runs`.

`POST /api/validate` takes the same form fields as `POST /runs` and returns the `validate-adapter` JSON report
plus `exit_code`, `timed_out` and `error`. `GET /healthz` returns `ok` without authentication.

For CI pipelines, `malvalid run` itself is simpler: it needs no server, and its exit code is the gate result.

## 9. Limitations

* **Single user, local machine.** There are no accounts. Whoever holds the link has full access, and there is
  no TLS. `--allow-remote` is for trusted networks only.
* **Windows.** The UI runs on Windows (the [launcher](launchers.md) starts it there). Runs are started in
  their own process group and cancelled with CTRL_BREAK_EVENT, then `taskkill /F /T`. Models run with
  reduced isolation, because Windows has no OS sandbox MalValid can use (see
  [launchers.md §6](launchers.md#6-isolation-on-windows-and-macos)).
* **One run at a time by default.** Raise `--max-concurrent` only if the machine has the memory and cores for
  several full runs.
* **Runs do not survive a server restart.** Stopping the server stops queued and running runs (`interrupted`).
  A run started by an earlier server that is still alive is adopted, but a lost run is never resumed.
* **Uploads are kept** in `submissions/` until you delete the run. They are capped per file
  (`--max-upload-mb`). For very large models, use path mode (needs `--allow-path-mode`).
* **The policy is a file.** The UI shows the default policy and accepts a policy YAML. It does not edit
  thresholds in the browser; use `malvalid init-config` and edit the YAML.
* **No adversarial-robustness testing.** This version of MalValid does not evaluate evasion robustness
  (module M3 is out of scope), so a `ready` verdict makes no claim about resistance to evasion attacks.
* **Validation timeout.** `Validate adapter` stops after 600 seconds; a full run has no web-side time limit
  beyond the policy's own deadlines.
