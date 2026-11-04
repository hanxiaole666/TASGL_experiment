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
  "$PY" log_runner.py --program test_p2 --python "$PY" -- test_p2.py -v
  CODE=$?
  echo "[INFO] No experiment was launched. Pass P2 subcommands to run one."
else
  "$PY" log_runner.py --program p2_experiments --python "$PY" -- p2_experiments.py "$@"
  CODE=$?
fi
echo
if [ "$CODE" -eq 0 ]; then echo "[DONE] Exit code 0"; else echo "[STOP] Exit code $CODE"; fi
echo "[INFO] Exit code 2 = expected protocol stop (no independent window / unfair budget)."
echo "[INFO] Logs are in Test-log. P2 results are isolated in results/p2."
if [ "${TASGL_NO_PAUSE:-0}" != "1" ]; then
  read -r -p "Press Enter to exit..." _
fi
exit "$CODE"
