#!/bin/bash
# Stops the MassSpec Bench server if it is running (same as "Quit MassSpec Bench" in the sidebar).
PORT="${PORT:-8766}"
if curl -s -m 2 "http://localhost:$PORT/api/settings" >/dev/null 2>&1; then
  curl -s -m 2 -X POST "http://localhost:$PORT/api/quit" >/dev/null 2>&1; sleep 1
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | xargs kill 2>/dev/null
  echo "MassSpec Bench stopped."
else
  echo "MassSpec Bench was not running."
fi
sleep 1
