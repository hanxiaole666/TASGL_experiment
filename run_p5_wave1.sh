#!/usr/bin/env bash
# P5 终审轮第一阶段: 按顺序跑三组长时间实验
#   1) BitHeNet 扩到 20 种子 (摆脱 p=0.0020 的检验下限)
#   2) H3 / H4 十种子区间估计
#   3) 第三数据集 DGraph-Fin 探针 (2 种子, 判断 6GB 设备能否承载 3.7M 节点)
set -u
cd "$(cd "$(dirname "$0")" && pwd)"
PY="${TASGL_PYTHON:-python3}"
LOG=results/p5/wave1.log
mkdir -p results/p5
{
  echo "=== [1] BitHeNet 20 seeds ==="
  date /t
} > "$LOG"

echo "=== [1] BitHeNet 20 seeds ===" >> "$LOG"
"$PY" -u p4_experiments.py second-dataset run \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --out results/p4/second_dataset_20seeds.json >> "$LOG" 2>&1
echo "stage1 rc=$?" >> "$LOG"

echo "=== [2] H3/H4 ten-seed interval estimation ===" >> "$LOG"
"$PY" -u p4_interval.py \
  --models tasgl mlp graphsage \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --epochs 200 >> "$LOG" 2>&1
echo "stage2 rc=$?" >> "$LOG"

echo "=== [3] DGraph-Fin probe (2 seeds) ===" >> "$LOG"
"$PY" -u p4_third.py --seeds 0 1 --epochs 30 \
  --out results/p5/dgraphfin_probe.json >> "$LOG" 2>&1
echo "stage3 rc=$?" >> "$LOG"

echo "=== WAVE1 DONE ===" >> "$LOG"
date /t >> "$LOG"
