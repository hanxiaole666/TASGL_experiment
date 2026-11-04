#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
PY="${TASGL_PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[ERROR] Python not found: $PY"
  CODE=127
elif [ "$#" -eq 0 ]; then
  "$PY" log_runner.py --program test_p0 --python "$PY" -- test_p0.py -v
  CODE=$?
  echo "[INFO] No experiment was launched. Pass P0 subcommands to run one."
else
  "$PY" log_runner.py --program p0_experiments --python "$PY" -- p0_experiments.py "$@"
  CODE=$?
fi
echo
if [ "$CODE" -eq 0 ]; then echo "[DONE] Exit code 0"; else echo "[STOP] Exit code $CODE"; fi
echo "[INFO] Logs are in Test-log. P0 results are isolated in results/p0."
if [ "${TASGL_NO_PAUSE:-0}" != "1" ]; then
  read -r -p "Press Enter to exit..." _
fi
exit "$CODE"
