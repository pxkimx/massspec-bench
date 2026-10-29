#!/bin/bash
# Double-click this file: it opens Terminal so you can watch every step (useful if the app itself
# will not open). Same environment and data as the app.
cd "$(dirname "$0")" || exit 1
APP="$(pwd)/MassSpec Bench.app/Contents/Resources/app"
[ -d "$APP" ] || APP="$(pwd)"
MODE=terminal exec /bin/bash "$APP/launch.sh"
