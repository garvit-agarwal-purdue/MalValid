#!/usr/bin/env bash
# Start (or restart) the malvalid web UI on THIS node behind an Open OnDemand (OOD) style node proxy.
#
#   MG_PORTAL_HOST=ondemand.example.edu scripts/serve_ood.sh [PORT]     # PORT defaults to $PORT, then 8765
#
# Run it in any new OOD job: it detects the node, stops this user's previous `malvalid serve` on the
# same port, starts a new detached one and prints the browser link (with the access token) to the
# terminal only; the log never contains the token. See "Behind a proxy" in docs/web.md.
#
# Environment (defaults in brackets):
#   MG_PORTAL_HOST        REQUIRED. Host name of your OOD portal, e.g. ondemand.example.edu
#   MG_PORTAL_IP          IP the portal connects from, the only remote client allowed
#                         [resolved from MG_PORTAL_HOST]
#   MG_NODE_IP            address to listen on [the node's address as resolved from its host name]
#   PORT                  port [8765]
#   MALVALID_BIN          malvalid executable [`malvalid` on PATH, else <repo>/.venv/bin/malvalid]
#   MG_TOKEN_FILE         access-token file, created (mode 0600) on first start and kept across
#                         restarts [${XDG_CONFIG_HOME:-~/.config}/malvalid/serve-token]
#   MG_RUNS_DIR           runs directory [~/malvalid-runs]
#   MG_LOG                server log [${XDG_STATE_HOME:-~/.local/state}/malvalid/serve.log]
#   MALVALID_CORPUS_DIR   corpus directory (passed through when set; otherwise malvalid's default)
#   MG_EXTRA_ARGS         extra `malvalid serve` arguments (word-split)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-${PORT:-8765}}"
if [ -n "${MALVALID_BIN:-}" ]; then :
elif command -v malvalid >/dev/null 2>&1; then MALVALID_BIN="$(command -v malvalid)"
else MALVALID_BIN="$REPO/.venv/bin/malvalid"; fi
PORTAL_HOST="${MG_PORTAL_HOST:-}"
PORTAL_IP="${MG_PORTAL_IP:-}"
TOKEN_FILE="${MG_TOKEN_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/malvalid/serve-token}"
RUNS_DIR="${MG_RUNS_DIR:-$HOME/malvalid-runs}"
LOG="${MG_LOG:-${XDG_STATE_HOME:-$HOME/.local/state}/malvalid/serve.log}"

die() { echo "serve_ood.sh: $*" >&2; exit 1; }

[ -n "$PORTAL_HOST" ] || die "set MG_PORTAL_HOST to your OnDemand portal host name (e.g. ondemand.example.edu)"
case "$PORT" in ''|*[!0-9]*) die "port must be a number (got '$PORT')";; esac
[ "$PORT" -ge 1024 ] && [ "$PORT" -le 65535 ] || die "port must be between 1024 and 65535"
[ -x "$MALVALID_BIN" ] || die "malvalid not found at $MALVALID_BIN (set MALVALID_BIN)"
[ -n "$PORTAL_IP" ] || PORTAL_IP="$(getent ahostsv4 "$PORTAL_HOST" 2>/dev/null | awk 'NR==1{print $1}')"
[ -n "$PORTAL_IP" ] || die "cannot resolve $PORTAL_HOST; set MG_PORTAL_IP to the portal's address"

# ---- access token: create once (mode 0600), reuse afterwards ----------------------------------------
if [ ! -s "$TOKEN_FILE" ]; then
  mkdir -p "$(dirname "$TOKEN_FILE")"
  ( umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$TOKEN_FILE" )
  chmod 600 "$TOKEN_FILE"
fi

# ---- this node: short name, FQDN, listen address ------------------------------------------------------
SHORT="$(hostname -s)"
FQDN="$(hostname -f 2>/dev/null || true)"
case "$FQDN" in *.*) ;; *) FQDN="$(getent hosts "$SHORT" 2>/dev/null | awk '{for(i=2;i<=NF;i++) if($i ~ /\./){print $i; exit}}')";; esac
[ -n "$FQDN" ] || die "cannot determine the fully qualified host name of $SHORT"
NODE_IP="${MG_NODE_IP:-$(getent ahostsv4 "$FQDN" 2>/dev/null | awk 'NR==1{print $1}')}"
[ -n "$NODE_IP" ] || die "cannot determine the address of $SHORT; set MG_NODE_IP"
ROOT_PATH="/node/$FQDN/$PORT"

# the server process itself (python ... /malvalid serve ... --port N), not the launcher shell around it
PAT="^[^ ]*python[0-9.]* [^ ]*/malvalid serve .*--port $PORT( |\$)"

# ---- stop this user's previous malvalid serve on this node and port --------------------------------
OLD="$(pgrep -u "$(id -u)" -f "$PAT" || true)"
if [ -n "$OLD" ]; then
  echo "stopping previous malvalid serve on $SHORT:$PORT (pid $(echo $OLD)) ..."
  # shellcheck disable=SC2086
  kill $OLD 2>/dev/null || true
  for _ in $(seq 1 100); do
    # shellcheck disable=SC2086
    kill -0 $OLD 2>/dev/null || break
    sleep 0.2
  done
  # shellcheck disable=SC2086
  kill -9 $OLD 2>/dev/null || true
fi

# ---- start detached; the log is passed through a filter so the token never reaches it --------------
mkdir -p "$(dirname "$LOG")" "$RUNS_DIR"
[ -s "$LOG" ] && mv -f "$LOG" "$LOG.prev"
ARGS=(serve --port "$PORT" --runs-dir "$RUNS_DIR" --no-browser --host "$NODE_IP" --allow-remote
      --root-path "$ROOT_PATH"
      --trusted-host "$PORTAL_HOST" --trusted-host "$FQDN"
      --allow-client "$PORTAL_IP" --allow-client "$NODE_IP"
      --token-file "$TOKEN_FILE")
# shellcheck disable=SC2206
[ -n "${MG_EXTRA_ARGS:-}" ] && ARGS+=($MG_EXTRA_ARGS)

cd "$REPO"
export COLUMNS=200
setsid bash -c '
  log=$1; shift
  exec > >(sed -u "s/token=[^[:space:]]*/token=REDACTED/g" >>"$log") 2>&1
  exec "$@"' _ "$LOG" "$MALVALID_BIN" "${ARGS[@]}" </dev/null >/dev/null 2>&1 &
disown

# ---- wait until it answers, then print the link ------------------------------------------------------
URL="http://$NODE_IP:$PORT$ROOT_PATH/healthz"
up=""
for _ in $(seq 1 100); do
  if curl -fsS -m 2 -o /dev/null -H "Host: $FQDN:$PORT" "$URL" 2>/dev/null; then up=1; break; fi
  sleep 0.3
done
PID="$(pgrep -u "$(id -u)" -f "$PAT" | head -n1 || true)"
[ -n "$up" ] || { echo "server did not come up; see $LOG" >&2; tail -n 20 "$LOG" >&2 || true; exit 1; }

TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
echo "malvalid serve running on $SHORT ($NODE_IP) port $PORT, pid ${PID:-?}; log: $LOG"
echo "corpus dir: ${MALVALID_CORPUS_DIR:-(malvalid default)}   runs dir: $RUNS_DIR"
echo "Open (private, contains the access token):"
echo "  https://$PORTAL_HOST$ROOT_PATH/?token=$TOKEN"
