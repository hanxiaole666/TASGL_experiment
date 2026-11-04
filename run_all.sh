#!/usr/bin/env bash
# run_all.sh — 一键复现论文全部实验 (单卡 RTX 2060 可运行)
#
# 功能:
#   1. 全部数据集免账号自动下载 (HTTP Range 断点续传)
#   2. 每条命令输出 [开始]/[结束] 时间, 并写入 Test-log/程序名-log-日期.txt
#   3. 所有实验支持断点续跑: 中断后重跑本脚本即可自动跳过已完成部分
#   4. 执行结束(或失败)后按任意键退出, 窗口不会自动关闭
#
# 用法: bash run_all.sh
# 自定义解释器: TASGL_PYTHON=/path/to/python bash run_all.sh  (默认 python3)

cd "$(cd "$(dirname "$0")" && pwd)"
PY="${TASGL_PYTHON:-${PY:-python3}}"
# 统一 UTF-8, 避免 Windows GBK 控制台导致中文日志乱码
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# 无论正常完成还是任一步骤失败, 都等待按键并保留原始退出码。
pause_on_exit() {
  local status=$?
  trap - EXIT
  printf '\n按任意键退出...'
  if [ -t 0 ]; then
    IFS= read -rsn 1 key || true
  elif [ -r /dev/tty ]; then
    IFS= read -rsn 1 key < /dev/tty || IFS= read -rsn 1 key || true
  else
    IFS= read -rsn 1 key || true
  fi
  echo
  exit "$status"
}
trap pause_on_exit EXIT

LOG_DIR="Test-log"
DATE_TAG=$(date +%Y%m%d)
mkdir -p "$LOG_DIR" results figs

# run_step <程序名(不带.py)> [参数...]
run_step() {
  local prog="$1"; shift
  "$PY" log_runner.py --program "$prog" --python "$PY" -- "${prog}.py" "$@"
  local status=$?
  if [ "$status" -ne 0 ]; then
    exit "$status"
  fi
}

echo "== 步骤 0: 全部数据集免账号下载 (断点续传) =="
run_step download_data --dataset all

echo "== E1+E2: 主实验与效率对比 (5 seeds) =="
run_step train --seeds 0 1 2 3 4

echo "== Pareto 前沿分析 =="
run_step pareto_analysis

echo "== E4: 消融实验 =="
run_step exp_ablation --seeds 0 1 2

echo "== E3: 时序鲁棒性 =="
run_step exp_temporal --seeds 0 1 2

echo "== E5: 对抗鲁棒性 =="
run_step exp_robustness --seeds 0 1

echo "== E7: 不确定性质量 =="
run_step exp_uncertainty

echo "== E6: 跨链泛化 (Elliptic -> Ethereum, 数据自动下载) =="
run_step exp_crosschain

echo "全部实验完成: 结果见 results/, 图表见 figs/, 日志见 $LOG_DIR/"
