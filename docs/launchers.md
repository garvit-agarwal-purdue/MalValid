# Double-click launchers

The `launchers/` folder starts MalValid without any typing. On first use it installs MalValid into a
private folder. Every time, it starts the web UI on your own computer and opens your browser on it,
already signed in. You then upload your model file and get a verdict and a 0–100 score.

| Your system | Double-click | What it runs |
|---|---|---|
| Windows 10 / 11 (x64), Windows 11 on Arm | `launchers\malvalid.bat` | `launch.ps1` (Windows PowerShell 5.1, built in) |
| Linux | `launchers/malvalid.desktop`, or run `launchers/malvalid.sh` | `malvalid.sh` (bash) |
| macOS | `launchers/malvalid.command` | `malvalid.sh` |

You need an internet connection the first time, and about 1.5 GB of disk space. You do not need Python
installed beforehand.

LightGBM and XGBoost, which load most model files, need one system library that the launcher does not
install:

* **macOS:** the OpenMP runtime from Homebrew. Install [Homebrew](https://brew.sh), then run
  `brew install libomp` in Terminal.
* **Windows:** the [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).
  Most PCs already have it.
* **Linux:** the OpenMP runtime `libgomp`, which almost every distribution already has.

If it is missing, setup still finishes, and the launcher warns at every start that LightGBM and XGBoost
models cannot be loaded. Install the library and start MalValid again; no reinstall is needed.

**Windows on Arm:** LightGBM and XGBoost publish no Arm builds for Windows, so the launcher sets up an
x64 Python, which Windows 11 runs through its built-in x64 emulation (it does not use a Python you
installed yourself). Windows 10 on Arm cannot run x64 programs and is not supported; use WSL2 there.

## Contents

1. [Get MalValid](#1-get-malvalid)
2. [Start it](#2-start-it)
3. [What happens on the first run](#3-what-happens-on-the-first-run)
4. [Later runs, stopping, updating](#4-later-runs-stopping-updating)
5. [Where your data lives](#5-where-your-data-lives)
6. [Isolation on Windows and macOS](#6-isolation-on-windows-and-macos)
7. [The EMBER corpora](#7-the-ember-corpora)
8. [Settings](#8-settings)
9. [Uninstall](#9-uninstall)
10. [Troubleshooting](#10-troubleshooting)

## 1. Get MalValid

Either clone the repository:

```bash
git clone https://github.com/garvit-agarwal-purdue/MalValid.git
```

or download the ZIP from the repository page (**Code → Download ZIP**) and **extract all of it**. On
Windows, right-click the ZIP and choose **Extract All**. A launcher run from inside the ZIP does not work.

The launchers install MalValid *from the folder they sit in*, so keep the `launchers/` folder inside
the MalValid folder. You can put that folder anywhere, including a path with spaces or accents. If you
move it later, the next start notices and sets up again.

## 2. Start it

* **Windows:** open the extracted folder, then `launchers`, and double-click **`malvalid.bat`**.
  * Windows may first show *"Windows protected your PC"* or *"The publisher could not be verified"*. The
    file came from the internet and is not signed. Choose **More info → Run anyway**, or **Run**.
  * To stop the prompts, right-click `malvalid.bat` and `launch.ps1`, choose **Properties**, tick
    **Unblock**, and click OK.
* **Linux:** double-click **`malvalid.desktop`**. If your file manager opens it as text, or will not run
  it, use one of these:
  * In a terminal, run `./launchers/malvalid.sh`.
  * Run `./launchers/install-desktop-entry.sh` once. That puts **MalValid** in your applications menu.
    Add `--desktop` to also get a desktop icon, then right-click the icon and choose **Allow Launching**.
* **macOS:** double-click **`malvalid.command`**.
  * The first time, macOS may refuse to open it because it is from an unidentified developer (the
    launchers are not signed or notarized).
    * **macOS 14 and earlier:** right-click (or Ctrl-click) the file, choose **Open**, then **Open** again.
    * **macOS 15 and later:** the right-click route no longer works. Double-click the file once and close
      the warning, then open **System Settings → Privacy & Security**, scroll down and click
      **Open Anyway** next to `malvalid.command`. Alternatively, run
      `xattr -dr com.apple.quarantine /path/to/MalValid` once in Terminal.
  * If it says the file is not executable (some ZIP tools drop the permission), run
    `chmod +x launchers/*.command launchers/*.sh` once in Terminal.

A console window opens and shows what is happening. **Keep it open while you use MalValid.**

## 3. What happens on the first run

1. **The launcher finds an installer.** It uses `uv`, Astral's standalone Python installer, if `uv` is
   on your PATH. If not, it uses Python 3.11 if you have it (on Windows, only a 64-bit Python 3.11). If you
   have neither, it downloads `uv` 0.12.19 once from the official GitHub release, checks the file against a
   SHA-256 checksum pinned in the launcher script itself (copied from that release's published checksums,
   not fetched alongside the download), and stores it in the app folder. A file that does not match is
   deleted and not used. The launchers pin checksums for Windows (x86_64, ARM64), Linux (x86_64, aarch64;
   glibc and musl) and macOS (Intel, Apple silicon); on any other platform they stop and ask you to install
   uv or Python 3.11 yourself. It prints exactly what it downloads and from where, for example:

   ```
   malvalid:   downloading: uv 0.12.19 for x86_64-pc-windows-msvc
   malvalid:   from:        https://github.com/astral-sh/uv/releases/download/0.12.19/uv-x86_64-pc-windows-msvc.zip
   malvalid:   sha256:      6dbb02d79e419522f1c500f0adb1cddcff0cda7d59b0d66ea7f5e3b4a1b2f5f0 (pinned in this launcher)
   ```

2. **`[1/3]`** It creates a private Python 3.11 environment in the app folder. uv downloads Python 3.11
   itself if needed, so your system Python is never touched.
3. **`[2/3]`** It installs MalValid with its web UI, ONNX model support and raw-PE featurization (the
   `web`, `onnx` and `featurize` extras), using the tested versions pinned in
   `constraints/lock-py311.txt`. If the pinned versions cannot be installed on your platform, it says so
   and installs the newest compatible versions instead. If the ONNX or featurization packages cannot be
   installed at all on your platform, it warns and installs the web UI without them: LightGBM and XGBoost
   model files still work, ONNX model files do not. This step downloads about 300 MB and takes a few
   minutes.
4. **`[3/3]`** It checks that the installation works, then starts the server. It also checks that
   LightGBM and XGBoost load, and warns if a system library is missing (see the top of this page).

The server listens on **127.0.0.1 only**, so other computers cannot reach it. It uses port 8765, or the
next free port if 8765 is taken. Your default browser then opens on the UI, **already signed in**: the
launcher hands the browser a one-time login link that expires after two minutes, so you never copy a
token. If the browser does not open, use the private link printed in the window. It signs in whoever
opens it, so do not share it.

On a fresh machine, try **Run the synthetic demo** on the dashboard (or under **New run**). It runs a
complete evaluation of the small demo model shipped in `examples/synthetic_demo/` on generated synthetic
data, with nothing to download, and shows a full verdict and report (it ends `BLOCKED`, on purpose). The demo checks the pipeline; its verdict says nothing about a real model.

## 4. Later runs, stopping, updating

* **Later runs** skip the setup and start in a few seconds. The launcher reinstalls only when it needs
  to: when MalValid's version, dependencies or pinned constraints change, when the folder moved, or when
  the environment is broken. MalValid is installed in editable mode, so code updates in the folder apply
  without a reinstall.
* **To stop MalValid,** close the console window or press **Ctrl+C** in it. Queued and running
  evaluations stop too. On Windows, cmd may ask *"Terminate batch job (Y/N)?"* after Ctrl+C; either
  answer is fine.
* **To update,** run `git pull` in the MalValid folder, or download and extract a new ZIP. Then
  double-click the launcher again.

## 5. Where your data lives

Your runs, settings and the Python environment are stored outside the MalValid folder (Python may add
`__pycache__` folders inside it, because MalValid is installed from it in editable mode). Everything goes
into one per-user **app folder**:

| | Windows | Linux | macOS |
|---|---|---|---|
| App folder | `%LOCALAPPDATA%\malvalid` | `${XDG_DATA_HOME:-~/.local/share}/malvalid` | `~/Library/Application Support/malvalid` |

| Inside the app folder | What it is |
|---|---|
| `runs/` | Your evaluations: one folder per run with `report.json`, `report.html`, logs and the uploaded files. |
| `venv/` | The private Python environment with malvalid. |
| `python/` | Python 3.11, if uv had to download it. |
| `cache/` | uv's download cache (unless you already set `UV_CACHE_DIR`). |
| `uv/` | The downloaded `uv`, only if the launcher had to fetch it. |
| `install.stamp` | Fingerprint used to skip reinstalling. |

The EMBER corpora, if you build them, live in `~/.cache/malvalid/corpora` (Windows:
`C:\Users\<you>\.cache\malvalid\corpora`) or in `$MALVALID_CORPUS_DIR` (see [§7](#7-the-ember-corpora)).

## 6. Isolation on Windows and macOS

MalValid treats every submitted model as untrusted. It scans the file before anything opens it. It
refuses pickle files unless you opt in. It loads and queries the model only in a separate worker
process, which talks to MalValid over a channel that never unpickles anything.

**Linux:** the worker runs in an **OS sandbox**, and reports record `"isolation": "os_sandbox"`:

* **with bubblewrap (`bwrap`):** no network, a read-only file system, your home directory hidden and one
  writable scratch directory;
* **with user namespaces only (no bubblewrap):** no network, but the file system is **not** restricted.
  The launcher prints a warning in this case. Install bubblewrap for full isolation.

**Windows and macOS** have no such sandbox that MalValid can use. The same goes for Linux systems where
unprivileged user namespaces are disabled. There, MalValid does not quietly run the model with less
protection. A plain `malvalid run` refuses with an explanation, and the launchers start the server with
the explicit option `--allow-reduced-isolation`. Models then run with **reduced isolation**:

* a separate worker process, with the same no-unpickle channel, pickle refusal and scrubbed environment;
* resource limits where the OS has them:
  * Windows: a Job Object with a memory limit and no child processes, killed with malvalid.
  * Linux: rlimits.
  * macOS: best effort.
* **no network and no file-system isolation.** A malicious model file that escaped the pickle guards
  could read your files or use the network.

The reduction is always visible:

* the launcher window prints a warning at startup;
* the web UI shows a "Reduced isolation: this platform has no OS sandbox" banner on **New run** and
  **System**;
* every run page and `report.html` shows it next to the verdict;
* `report.json` records it in the verdict and the sandbox section:

```json
"verdict": { "verdict": "conditional", "score": 81.4, "isolation": "process_only",
             "isolation_warning": "Reduced isolation: this platform has no OS sandbox. ...", ... },
"sandbox": { "isolation": "process_only", "backend": "subprocess", ... }
```

`isolation` is `os_sandbox` (bubblewrap or user namespaces), `process_only` (reduced isolation) or
`none` (the debug-only in-process mode, `--no-sandbox`). The flag never changes the verdict or the score.
It only describes how well the run was protected.

**Recommendation:** use the launcher on Windows or macOS for models you trained yourself, or whose
origin you trust. To evaluate models from other people, run MalValid on Linux with bubblewrap. On
Windows that can be **WSL2**: install Ubuntu from the Microsoft Store, install bubblewrap with
`sudo apt install bubblewrap`, then run `launchers/malvalid.sh` inside WSL. Its `127.0.0.1` port is
reachable from your Windows browser. `malvalid sandbox-check` shows what your machine supports.

## 7. The EMBER corpora

The real evaluation corpora (EMBER2018 for `ember_v2` models, EMBER2024 for `ember_v3` models) hold
millions of feature vectors, about 5–10 GB each. They are not shipped with MalValid, and the launcher does
not download them. Without them:

* the synthetic demo and the synthetic corpora (`synthetic_v2`, `synthetic_v3`) work out of the box;
* if you submit a real model, or pick an EMBER corpus that is not built, the New run page refuses the
  run before it starts, with a message that names the corpus and gives the exact build steps.

To build a corpus, use the `malvalid` command inside the app folder's environment:

```bash
# Linux
~/.local/share/malvalid/venv/bin/malvalid corpus build ember_v2_2018 --source ember2018/ --workers 8
# macOS
~/Library/Application\ Support/malvalid/venv/bin/malvalid corpus build ember_v2_2018 --source ember2018/ --workers 8
```

```bat
rem Windows (cmd)
"%LOCALAPPDATA%\malvalid\venv\Scripts\malvalid.exe" corpus build ember_v2_2018 --source ember2018\ --workers 8
```

The download and extraction steps for each corpus are on the **Corpora** page of the UI, in
`malvalid corpus info ember_v2_2018` (or `ember_v3_2024`), and in [`docs/modules/corpus_ember2018.md`](modules/corpus_ember2018.md) and
[`corpus_ember2024.md`](modules/corpus_ember2024.md).

## 8. Settings

The defaults suit most people. To change them, set these environment variables before starting the
launcher:

| Variable | Default | Meaning |
|---|---|---|
| `MALVALID_HOME` | see [§5](#5-where-your-data-lives) | The app folder. |
| `MALVALID_RUNS_DIR` | `<app folder>/runs` | Where runs are stored. |
| `MALVALID_PORT` | `8765` | First port to try; the next free one is used if it is busy. |
| `MALVALID_REINSTALL` | unset | `1` forces a fresh setup of the environment. |
| `MALVALID_NO_PAUSE` | unset | Linux and macOS: `1` closes the window after an error without waiting for Enter. |
| `MALVALID_CORPUS_DIR` | `~/.cache/malvalid/corpora` | Where built EMBER corpora are found (read by MalValid itself, so it also applies outside the launchers). |
| `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` | inside the app folder | uv's cache and Python downloads. |

Arguments after the launcher name go to `malvalid serve`, for example `./launchers/malvalid.sh --no-browser`
or `malvalid.bat --no-browser`. The `malvalid serve` flags the launchers use are `--find-free-port` and
`--allow-reduced-isolation` (Windows and macOS, and Linux without an OS sandbox). See
[docs/web.md](web.md) for both.

## 9. Uninstall

1. Close the MalValid window.
2. Delete the app folder from [§5](#5-where-your-data-lives). This removes your runs too, so copy
   `runs/` first if you want to keep any reports.
3. Delete the MalValid folder you downloaded, and `~/.cache/malvalid` if you built corpora.
4. Linux only: if you used `install-desktop-entry.sh`, run it with `--uninstall` first.

Nothing else is installed. The launchers do not change PATH, the registry, or your shell profile.

## 10. Troubleshooting

| Problem | What to do |
|---|---|
| The window closes immediately (Windows) | Open `cmd`, `cd` into the `launchers` folder and run `malvalid.bat` to read the message. |
| "running scripts is disabled" / the script is blocked (Windows) | The launcher already passes `-ExecutionPolicy Bypass`. If your organisation enforces a policy through Group Policy or AppLocker, that cannot be bypassed: ask your IT team, or use WSL2 ([§6](#6-isolation-on-windows-and-macos)). |
| `could not download …` | Check your internet connection. Behind a proxy, set `HTTPS_PROXY` (e.g. `http://proxy.example:3128`) before starting; uv, pip and the launchers' own download all use it. On Windows the uv download also follows the system proxy settings. |
| "checksum mismatch for uv-…" or "no pinned checksum for uv" | The downloaded `uv` did not match the checksum pinned in the launcher (a broken or tampered download, or a proxy that rewrites files), or there is no prebuilt `uv` with a pinned checksum for your platform. Nothing was installed. Install [uv](https://docs.astral.sh/uv/) or Python 3.11 (64-bit on Windows) yourself and start the launcher again. |
| "installing the pinned versions (constraints/lock-py311.txt) failed" | Only a warning: the launcher retries with the newest compatible versions. The messages just above it say why the pinned install failed. |
| macOS: "LightGBM / XGBoost cannot be loaded", or a run fails with `Library not loaded: @rpath/libomp.dylib` | Install the OpenMP runtime: install [Homebrew](https://brew.sh), run `brew install libomp`, and start MalValid again. |
| Windows: "LightGBM / XGBoost cannot be loaded", or a run fails with `Could not find module ... lib_lightgbm.dll (or one of its dependencies)` | Install the [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) and start MalValid again. On Windows on Arm, install the x64 redistributable too. |
| Windows on Arm: setup fails while installing packages | Use Windows 11 (it runs the x64 Python the launcher sets up). On Windows 10 on Arm, use WSL2. If you set up MalValid with an earlier launcher, set `MALVALID_REINSTALL=1` once. |
| macOS 15+: "Apple could not verify" and only **Done** / **Move to Trash** | See [§2](#2-start-it): **System Settings → Privacy & Security → Open Anyway**. |
| ONNX models fail with "ONNX models need onnxruntime" | The ONNX packages could not be installed on your platform during setup (the window warned). First set `MALVALID_REINSTALL=1` once and start the launcher again ([§8](#8-settings)). To install them by hand, open a terminal in the MalValid folder. The environment usually comes from uv, and has no pip, so use uv (on your PATH, or in the app folder's `uv` subfolder). Linux and macOS: `uv pip install --python "<app folder>/venv/bin/python" -c constraints/lock-py311.txt -e ".[web,onnx,featurize]"`. Windows: `"<app folder>\uv\uv.exe" pip install --python "<app folder>\venv\Scripts\python.exe" -c constraints\lock-py311.txt -e ".[web,onnx,featurize]"`. If the launcher used your own Python 3.11 instead of uv, run `"<app folder>/venv/bin/python" -m pip install` with the same arguments (Windows: `venv\Scripts\python.exe`). |
| "another MalValid window is setting up MalValid right now" | Wait for the other window to finish. If none is open, start again; a lock left by a closed window is cleared automatically. |
| The browser shows "This site can't be reached" for `mg-….localhost` | Some browsers (older Safari versions) do not resolve `*.localhost`. Use the second link the window prints, `http://127.0.0.1:<port>/?token=…`. |
| The browser did not open | Copy the private link from the window into your browser. On a Linux machine without a display (SSH), forward the port shown in the printed link (8765 unless it was busy): `ssh -L 8765:127.0.0.1:8765 you@host`, then open the printed `127.0.0.1` link locally. |
| "Signed out" / 401 after restarting | Each start makes a new private link. Use the browser window the launcher just opened, or the new link in the window. |
| Port 8765 is in use | Nothing to do: the next free port is used and printed. Set `MALVALID_PORT` to prefer another. |
| A run fails with "no OS sandbox is available" | You started MalValid some other way, without `--allow-reduced-isolation`. Start it with the launcher, or read [§6](#6-isolation-on-windows-and-macos). |
| Linux: reduced isolation although you are on Linux | bubblewrap is missing, or unprivileged user namespaces are disabled (Ubuntu 24.04 and later restrict them with AppArmor). Run `~/.local/share/malvalid/venv/bin/malvalid sandbox-check` for details. Install `bubblewrap`, or ask your administrator to allow user namespaces. |
| Something is broken after an update | Set `MALVALID_REINSTALL=1` once, or delete the `venv` folder in the app folder. Your runs are kept. |
| Antivirus quarantines files during setup (Windows) | Python packages contain native libraries that some scanners flag. Allow the app folder, or install from a WSL2 shell instead. |
| Very long Windows paths | Keep the MalValid folder near the drive root (e.g. `C:\malvalid`) if setup fails with path-length errors, or enable long paths in Windows. |
