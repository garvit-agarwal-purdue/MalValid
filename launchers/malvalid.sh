#!/usr/bin/env bash
# malvalid launcher for Linux (and macOS, via malvalid.command).
#
#   Double-click malvalid.desktop (Linux) or malvalid.command (macOS), or run: ./launchers/malvalid.sh
#
# First run: sets up malvalid in a private environment under your user data directory (needs internet,
# a few minutes). Later runs start in seconds. It then starts the malvalid web UI on this computer only
# (127.0.0.1) and opens your browser on it, already signed in. Close the window (or press Ctrl+C) to stop.
#
# Settings (environment variables, all optional):
#   MALVALID_HOME      app directory (default: ${XDG_DATA_HOME:-~/.local/share}/malvalid;
#                      macOS: ~/Library/Application Support/malvalid). Delete it to uninstall.
#   MALVALID_RUNS_DIR  where runs are stored (default: $MALVALID_HOME/runs)
#   MALVALID_PORT      first port to try (default 8765; the next free one is used if it is busy)
#   MALVALID_REINSTALL=1  force a fresh install of the environment
#   MALVALID_NO_PAUSE=1   never wait for Enter before closing after an error
# Extra arguments are passed to `malvalid serve` (e.g. --no-browser).
set -euo pipefail

UV_VERSION="0.12.19"   # pinned uv release, downloaded only when neither uv nor Python 3.11 is installed
PY_VERSION="3.11"
LAUNCHER_REV="3"       # bump to force a reinstall after a launcher change

say()  { printf 'malvalid: %s\n' "$*"; }
warn() { printf 'malvalid: WARNING: %s\n' "$*" >&2; }
pause_if_interactive() {
  if [ -t 0 ] && [ -t 1 ] && [ -z "${MALVALID_NO_PAUSE:-}" ]; then
    printf 'Press Enter to close this window. '
    read -r _ || true
  fi
}
die() { printf 'malvalid: ERROR: %s\n' "$*" >&2; pause_if_interactive; exit 1; }

# ---- where things are ------------------------------------------------------------------------------

# Directory of this script, following symlinks (no `readlink -f`: macOS lacks it).
script_dir() {
  local p="$1" d
  while [ -h "$p" ]; do
    d="$(cd -P "$(dirname "$p")" && pwd)"
    p="$(readlink "$p")"
    case "$p" in /*) ;; *) p="$d/$p" ;; esac
  done
  cd -P "$(dirname "$p")" && pwd
}
HERE="$(script_dir "${BASH_SOURCE[0]}")"
REPO="$(cd "$HERE/.." && pwd -P)"
[ -f "$REPO/pyproject.toml" ] && [ -d "$REPO/src/malvalid" ] \
  || die "this launcher must stay in the launchers/ folder of the MalValid download ($REPO does not look like one)"
CONSTRAINTS="$REPO/constraints/lock-py311.txt"

OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
  DEFAULT_APP="$HOME/Library/Application Support/malvalid"
else
  DEFAULT_APP="${XDG_DATA_HOME:-$HOME/.local/share}/malvalid"
fi
APP="${MALVALID_HOME:-$DEFAULT_APP}"
VENV="$APP/venv"
PY="$VENV/bin/python"
RUNS="${MALVALID_RUNS_DIR:-$APP/runs}"
STAMP="$APP/install.stamp"
# Keep uv's Python downloads and cache inside the app directory, so deleting it uninstalls everything.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$APP/python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$APP/cache}"

# ---- helpers ---------------------------------------------------------------------------------------

fingerprint() {
  { printf 'rev=%s repo=%s py=%s\n' "$LAUNCHER_REV" "$REPO" "$PY_VERSION"
    cat "$REPO/pyproject.toml"
    [ -f "$CONSTRAINTS" ] && cat "$CONSTRAINTS"
  } | cksum | awk '{print $1 "-" $2}'
}

needs_install() {
  [ -n "${MALVALID_REINSTALL:-}" ] && return 0
  [ -x "$PY" ] || return 0
  [ -f "$STAMP" ] || return 0
  [ "$(cat "$STAMP" 2>/dev/null)" = "$(fingerprint)" ] || return 0
  "$PY" -c 'import malvalid.web, uvicorn, starlette' >/dev/null 2>&1 || return 0
  return 1
}

find_uv() {
  if [ -n "${MALVALID_UV:-}" ] && [ -x "$MALVALID_UV" ]; then printf '%s\n' "$MALVALID_UV"; return 0; fi
  if command -v uv >/dev/null 2>&1; then command -v uv; return 0; fi
  if [ -x "$APP/uv/uv" ]; then printf '%s\n' "$APP/uv/uv"; return 0; fi
  return 1
}

# A Python 3.11 that can create virtual environments (Debian/Ubuntu need python3.11-venv for that).
find_python311() {
  local c v
  for c in python3.11 python3 python; do
    command -v "$c" >/dev/null 2>&1 || continue
    # macOS: /usr/bin/python3 is a stub that pops up an "install the developer tools" dialog when the
    # Command Line Tools are missing; skip it unless they are installed.
    if [ "$OS" = "Darwin" ] && [ "$(command -v "$c")" = "/usr/bin/$c" ] && ! xcode-select -p >/dev/null 2>&1; then
      continue
    fi
    v="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ "$v" = "$PY_VERSION" ] && "$c" -c 'import venv, ensurepip' >/dev/null 2>&1; then
      command -v "$c"; return 0
    fi
  done
  return 1
}

fetch() {  # fetch URL FILE
  if command -v curl >/dev/null 2>&1; then
    curl -fL --proto '=https' --tlsv1.2 --retry 3 --progress-bar -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$2" "$1"
  else
    die "need curl or wget to download uv (or install Python $PY_VERSION or uv yourself, then run this again)"
  fi
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# SHA-256 of each uv $UV_VERSION archive this launcher may download, copied from the official release's
# per-asset .sha256 files (cross-checked against its dist-manifest.json and sha256.sum). They are pinned
# here, not fetched next to the archive, so a tampered release asset cannot pass the check.
# Update them together with UV_VERSION.
uv_sha256() {
  case "$1" in
    x86_64-unknown-linux-gnu)   echo "23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8" ;;
    aarch64-unknown-linux-gnu)  echo "0804e9b164c64b6914182d5920c08551958a095986f10a3731056df701126436" ;;
    x86_64-unknown-linux-musl)  echo "db7278c9f57981338fddff1fb250e11964bc0a4fafcb9eed8303fdb117dc067b" ;;
    aarch64-unknown-linux-musl) echo "ad8d8448a2ff642ba62c2f684d7dd22a03f8eb3fc9918c2c3e8ec975f4ed6710" ;;
    x86_64-apple-darwin)        echo "cb5fa57bafe68fc0fb94b17f06bee0b0b9a7feb94ccbd110445afa0696e39273" ;;
    aarch64-apple-darwin)       echo "a9a8df1eedeb192f2e47e40e2faabfb387db4b850209118786d42f89dde3e0ba" ;;
    *) echo "" ;;
  esac
}

download_uv() {
  local arch a target asset url tmp want got
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) a="x86_64" ;;
    aarch64|arm64) a="aarch64" ;;
    *) die "there is no prebuilt uv for this CPU ($arch); install Python $PY_VERSION (or uv) yourself and run this again" ;;
  esac
  if [ "$OS" = "Darwin" ]; then
    target="$a-apple-darwin"
  elif (ldd --version 2>&1 || true) | grep -qi musl; then   # musl's ldd exits 1 (pipefail)
    target="$a-unknown-linux-musl"
  else
    target="$a-unknown-linux-gnu"
  fi
  asset="uv-$target.tar.gz"
  url="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$asset"
  want="$(uv_sha256 "$target")"
  if [ -z "$want" ]; then
    die "this launcher has no pinned checksum for uv $UV_VERSION on $target, so it will not download it; install uv (https://docs.astral.sh/uv/) or Python $PY_VERSION yourself and run this again"
  fi
  say "Neither uv nor Python $PY_VERSION was found, so the launcher downloads uv once"
  say "(uv is the standalone Python installer from astral.sh; it then installs Python $PY_VERSION for MalValid)."
  say "  downloading: uv $UV_VERSION for $target"
  say "  from:        $url"
  say "  sha256:      $want (pinned in this launcher)"
  say "  to:          $APP/uv/uv"
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/malvalid-uv.XXXXXX")"
  fetch "$url" "$tmp/$asset" || { rm -rf "$tmp"; die "could not download $url (are you online? behind a proxy, set HTTPS_PROXY)"; }
  got="$(sha256_of "$tmp/$asset")"
  if [ "$want" != "$got" ]; then
    rm -rf "$tmp"; die "checksum mismatch for $asset (expected $want, got $got); not using it"
  fi
  tar -xzf "$tmp/$asset" -C "$tmp"
  mkdir -p "$APP/uv"
  if [ -f "$tmp/uv-$target/uv" ]; then cp "$tmp/uv-$target/uv" "$APP/uv/uv"; else cp "$(find "$tmp" -type f -name uv | head -n 1)" "$APP/uv/uv"; fi
  chmod +x "$APP/uv/uv"
  rm -rf "$tmp"
  say "uv $UV_VERSION installed (checksum verified)."
}

# pip_install EXTRAS: install the repository (editable) with these extras, pinned versions first.
# Runs from the repository with relative paths: uv splits an absolute constraints path at spaces
# ("/Users/Jane Doe/..."), which would make the pinned install fail for no real reason.
pip_install() {
  local pkg=".[$1]" c="constraints/lock-py311.txt"
  ( cd "$REPO" || exit 1
    if [ -n "$UV" ]; then
      "$UV" pip install --python "$PY" -c "$c" -e "$pkg" && exit 0
      warn "installing the pinned versions (constraints/lock-py311.txt) failed (see above);"
      warn "retrying with the newest compatible versions"
      "$UV" pip install --python "$PY" -e "$pkg"
    else
      "$PY" -m pip install -c "$c" -e "$pkg" && exit 0
      warn "installing the pinned versions (constraints/lock-py311.txt) failed (see above);"
      warn "retrying with the newest compatible versions"
      "$PY" -m pip install -e "$pkg"
    fi
  )
}

# The model libraries load native code that needs a system OpenMP runtime on some platforms (macOS:
# Homebrew libomp; the wheels do not bundle it). Prints what is missing and returns 1 if they cannot load.
check_model_libs() {
  local err
  err="$("$PY" -c 'import lightgbm, xgboost' 2>&1 >/dev/null)" && return 0
  warn "LightGBM / XGBoost cannot be loaded, so LightGBM and XGBoost models (and the synthetic demo) will fail:"
  [ -z "$err" ] || printf '%s\n' "$err" | tail -n 3 | sed 's/^/    /' >&2
  if [ "$OS" = "Darwin" ]; then
    warn "on macOS they need the OpenMP runtime from Homebrew: install Homebrew (https://brew.sh), then run"
    warn "    brew install libomp"
    warn "in Terminal and start MalValid again (no reinstall needed)."
  else
    warn "the usual cause is a missing OpenMP runtime (libgomp; e.g. 'sudo apt install libgomp1' on Debian/Ubuntu)."
  fi
  return 1
}

install_malvalid() {
  local syspy=""
  UV=""
  if UV="$(find_uv)"; then
    say "Using uv: $UV"
  elif syspy="$(find_python311)"; then
    UV=""
    say "Using Python $PY_VERSION: $syspy"
  else
    download_uv
    UV="$APP/uv/uv"
  fi
  say "Setting up MalValid in $APP"
  say "(first run only: downloads about 300 MB of Python packages and takes a few minutes)"
  rm -rf "$VENV" "$STAMP"
  if [ -n "$UV" ]; then
    say "[1/3] Creating a private Python $PY_VERSION environment (uv downloads Python itself if needed)"
    "$UV" venv --quiet --python "$PY_VERSION" "$VENV" || die "could not create the Python environment with uv"
  else
    say "[1/3] Creating a private Python $PY_VERSION environment"
    "$syspy" -m venv "$VENV" || die "could not create a virtual environment with $syspy"
    "$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
  fi
  say "[2/3] Installing MalValid and its dependencies (pinned versions from constraints/lock-py311.txt)"
  # The web UI plus ONNX models and raw-PE featurization. If the optional extras have no build for this
  # platform, fall back to the web UI alone (LightGBM and XGBoost models still work).
  if ! pip_install "web,onnx,featurize"; then
    warn "the ONNX / raw-PE extras could not be installed here; installing the web UI without them"
    warn "(LightGBM and XGBoost model files work; ONNX model files will not)"
    pip_install "web" || die "installing malvalid failed (see the messages above)"
  fi
  say "[3/3] Checking the installation"
  "$PY" -c 'import malvalid.web, uvicorn, starlette' || die "malvalid was installed but does not import (see above)"
  fingerprint > "$STAMP"
  say "Setup complete."
  MODEL_LIBS_CHECKED=1
  if ! check_model_libs && [ -t 0 ] && [ -t 1 ] && [ -z "${MALVALID_NO_PAUSE:-}" ]; then
    printf 'Press Enter to start MalValid anyway (ONNX models still work). '
    read -r _ || true
  fi
}

# ---- main ------------------------------------------------------------------------------------------

mkdir -p "$APP" || die "cannot create $APP"
MODEL_LIBS_CHECKED=""
if needs_install; then
  # One installer at a time (a double double-click).
  LOCK="$APP/.install-lock"
  if ! mkdir "$LOCK" 2>/dev/null; then
    oldpid="$(cat "$LOCK/pid" 2>/dev/null || true)"
    if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
      die "another MalValid window is setting up MalValid right now; wait for it to finish, then start it again"
    fi
    rm -rf "$LOCK"   # left behind by a setup window that was closed
    mkdir "$LOCK" || die "cannot create $LOCK"
  fi
  echo "$$" > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT
  trap 'rm -rf "$LOCK"; exit 130' INT HUP TERM
  install_malvalid
  rm -rf "$LOCK"
  trap - EXIT INT HUP TERM
fi

# Isolation: bubblewrap/user namespaces give an OS sandbox on Linux. Where there is none (macOS, or a
# Linux without them) the server is started with --allow-reduced-isolation, and every run says so.
ISO_FLAG=""
if [ "$OS" = "Darwin" ]; then
  ISO_FLAG="--allow-reduced-isolation"
elif ! "$PY" -m malvalid sandbox-check --json >/dev/null 2>&1; then
  ISO_FLAG="--allow-reduced-isolation"
fi
if [ -n "$ISO_FLAG" ]; then
  warn "this computer has no OS sandbox (bubblewrap / user namespaces), so models run with REDUCED ISOLATION:"
  warn "a separate worker process with pickle refusal and resource limits, but no network or file-system"
  warn "isolation. Reports are marked 'isolation: process_only'. Only evaluate models whose origin you trust."
elif ! "$PY" -m malvalid sandbox-check --json 2>/dev/null \
     | "$PY" -c 'import json, sys; sys.exit(0 if json.load(sys.stdin).get("bwrap", {}).get("available") else 1)' 2>/dev/null; then
  warn "bubblewrap is not available, so the sandbox uses user namespaces only: the model has no network,"
  warn "but the file system is NOT restricted. Install bubblewrap for full isolation."
fi

[ -n "$MODEL_LIBS_CHECKED" ] || check_model_libs || true

mkdir -p "$RUNS" || die "cannot create the runs directory $RUNS"
cd "$APP"
echo
echo "=============================================================================="
echo "  MalValid is running - close this window to stop it (or press Ctrl+C)."
if [ "$OS" = "Darwin" ] || [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
  echo "  Your browser opens on it automatically, already signed in."
fi
echo "  Runs are saved in: $RUNS"
echo "=============================================================================="
if [ "$OS" != "Darwin" ] && [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  echo "  No display was found: open the link printed below in a browser on this computer"
  echo "  (or forward the port shown in that link over SSH: ssh -L PORT:127.0.0.1:PORT this-host)."
fi
echo

set +e
if [ -n "$ISO_FLAG" ]; then
  "$PY" -m malvalid serve --host 127.0.0.1 --port "${MALVALID_PORT:-8765}" --find-free-port --runs-dir "$RUNS" "$ISO_FLAG" ${1+"$@"}
else
  "$PY" -m malvalid serve --host 127.0.0.1 --port "${MALVALID_PORT:-8765}" --find-free-port --runs-dir "$RUNS" ${1+"$@"}
fi
rc=$?
set -e
case "$rc" in 0|129|130|143) rc=0 ;; esac   # Ctrl+C, window closed, stopped: normal
if [ "$rc" -eq 0 ]; then
  say "MalValid stopped."
else
  printf 'malvalid: the server stopped with exit code %s (see the messages above).\n' "$rc" >&2
  pause_if_interactive
fi
exit "$rc"
