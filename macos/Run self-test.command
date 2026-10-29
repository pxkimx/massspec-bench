#!/bin/bash
# Double-click: installs/updates the environment if needed, then runs a short simulated analysis of every
# kind to confirm everything works on this Mac. Nothing is started.
cd "$(dirname "$0")" || exit 1
APP="$(pwd)/MassSpec Bench.app/Contents/Resources/app"
[ -d "$APP" ] || APP="$(pwd)"
MODE=terminal exec /bin/bash "$APP/launch.sh" --selftest
