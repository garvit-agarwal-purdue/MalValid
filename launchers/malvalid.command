#!/bin/bash
# macOS: double-click this file in Finder to start malvalid (it runs malvalid.sh from the same folder).
# The first time, macOS may refuse to open it because it is from an unidentified developer:
#   macOS 14 and earlier: right-click it, choose Open, then Open.
#   macOS 15 and later:   double-click it once, then open System Settings > Privacy & Security and click
#                         "Open Anyway" (or, in Terminal: xattr -dr com.apple.quarantine <MalValid folder>).
exec /bin/bash "$(cd "$(dirname "$0")" && pwd)/malvalid.sh" ${1+"$@"}
