#!/usr/bin/env bash
# Install a "malvalid" entry in your applications menu that starts this folder's malvalid.sh.
#
#   ./launchers/install-desktop-entry.sh             # applications menu only
#   ./launchers/install-desktop-entry.sh --desktop   # also put an icon on the desktop
#   ./launchers/install-desktop-entry.sh --uninstall # remove both again
#
# Re-run it if you move the malvalid folder. It only writes ~/.local/share/applications/malvalid.desktop
# (and the desktop copy); nothing else.
set -euo pipefail

here="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher="$here/malvalid.sh"
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
entry="$apps/malvalid.desktop"
desktop_dir=""
if command -v xdg-user-dir >/dev/null 2>&1; then desktop_dir="$(xdg-user-dir DESKTOP 2>/dev/null || true)"; fi
[ -n "$desktop_dir" ] && [ "$desktop_dir" != "$HOME" ] || desktop_dir="$HOME/Desktop"
desktop_copy="$desktop_dir/malvalid.desktop"

want_desktop=0
case "${1:-}" in
  --desktop) want_desktop=1 ;;
  --uninstall)
    rm -f "$entry" "$desktop_copy"
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$apps" >/dev/null 2>&1 || true
    echo "Removed the MalValid menu entry (the app data in ${XDG_DATA_HOME:-$HOME/.local/share}/malvalid is untouched)."
    exit 0 ;;
  "") ;;
  *) echo "usage: $0 [--desktop | --uninstall]" >&2; exit 2 ;;
esac

[ -f "$launcher" ] || { echo "error: $launcher not found" >&2; exit 1; }
chmod +x "$launcher"

# Desktop Entry Exec quoting: inside "...", escape \ " ` $ with a backslash; in a .desktop value every
# backslash must itself be written as \\ ; and a literal % is written %%.
exec_quote() {
  local s="$1"
  s="${s//\\/\\\\}"; s="${s//\"/\\\"}"; s="${s//\`/\\\`}"; s="${s//\$/\\\$}"
  s="${s//\\/\\\\}"
  s="${s//%/%%}"
  printf '"%s"' "$s"
}

mkdir -p "$apps"
tmp="$(mktemp "$apps/.malvalid.desktop.XXXXXX")"
{
  echo "[Desktop Entry]"
  echo "Type=Application"
  echo "Version=1.0"
  echo "Name=MalValid"
  echo "GenericName=Malware-detector test bench"
  echo "Comment=Test a malware-detection model and get a verdict and a 0-100 score (local web UI)"
  echo "Exec=$(exec_quote "$launcher")"
  echo "Path=$HOME"
  echo "Terminal=true"
  echo "Icon=utilities-terminal"
  echo "Categories=Development;"
  echo "StartupNotify=false"
} > "$tmp"
chmod 755 "$tmp"
mv -f "$tmp" "$entry"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$apps" >/dev/null 2>&1 || true
echo "Installed $entry"
echo "  -> starts $launcher"

if [ "$want_desktop" = 1 ]; then
  mkdir -p "$desktop_dir"
  cp -f "$entry" "$desktop_copy"
  chmod 755 "$desktop_copy"
  # GNOME asks before running a desktop file unless it is marked trusted.
  command -v gio >/dev/null 2>&1 && gio set "$desktop_copy" metadata::trusted true >/dev/null 2>&1 || true
  echo "Installed $desktop_copy (if your desktop shows it as untrusted: right-click it, Allow Launching)"
fi
