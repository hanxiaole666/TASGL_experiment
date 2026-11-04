#!/usr/bin/env bash
# H1-H5 协议 6：开发硬门槛 -> 锁定 -> 5 种子确认；结束或失败后等待按键。
set -u
# Portable interpreter: TASGL_PYTHON (or PY) overrides the default python3.
PY="${TASGL_PYTHON:-${PY:-python3}}"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
cd "$(cd "$(dirname "$0")" && pwd)"
pause_on_exit() {
  local status=$?
  trap - EXIT
  printf '\n按任意键退出...'
  if [ -t 0 ]; then IFS= read -rsn 1 key || true
  elif [ -r /dev/tty ]; then IFS= read -rsn 1 key < /dev/tty || true
  else IFS= read -rsn 1 key || true; fi
  echo
  exit "$status"
}
trap pause_on_exit EXIT
mkdir -p Test-log results
run_step() {
  local prog="$1"; shift
  "$PY" log_runner.py --program "$prog" --python "$PY" -- "${prog}.py" "$@"
  local status=$?
  if [ "$status" -ne 0 ]; then exit "$status"; fi
}
run_step download_data --dataset elliptic
run_step exp_hypotheses --phase select --selection_seeds 0 1
run_step exp_hypotheses --phase confirm --seeds 0 1 2 3 4
echo '流程结束：查看 results/v4/；all_pass=false 时必须如实报告。'
