#!/usr/bin/env bash
# P5 终审轮第二阶段: BitHeNet 扩到 20 种子 + 跨时间窗稳定性 + 多重比较
#
# 与 wave1 的区别 (wave1 第 1 阶段失败的两个原因都在这里修掉):
#   1) 显存: losses.py 的 [P,N] 相似度矩阵改为按行分块, 峰值从 256 MiB
#      降到 64 MiB 量级 (数学结果不变, 见 test_losses_chunk.py)。
#   2) 配置: wave1 误用了 --search 的默认路径 results/p4/hpo_search.json,
#      把 Elliptic 的 HPO 配置 (hidden=64) 套到了 BitHeNet 上, 既撑爆显存,
#      也破坏了与已确认十种子结果的可比性。这里显式指向一个不存在的文件,
#      使 config={} 与 results/p4/second_dataset_confirm.json 完全一致。
#      "该数据集上不做任何超参搜索" 是它保持独立性的前提, 产物里会记录
#      config_source 字段。
set -u
cd "$(cd "$(dirname "$0")" && pwd)"
PY="${TASGL_PYTHON:-python3}"
LOG=results/p5/wave2.log
mkdir -p results/p5

echo "=== [1] BitHeNet 20 seeds + 3 sub-windows ===" > "$LOG"
"$PY" -u p4_experiments.py second-dataset run \
  --search results/p5/_no_search_by_design.json \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --sub_windows 35-39,40-44,45-49 \
  --out results/p4/second_dataset_20seeds.json >> "$LOG" 2>&1
echo "stage1 rc=$?" >> "$LOG"

echo "=== [2] 20-seed analysis: Holm + power design + cross-window ===" >> "$LOG"
"$PY" -u p5_bith20.py build \
  --in results/p4/second_dataset_20seeds.json >> "$LOG" 2>&1
echo "stage2 rc=$?" >> "$LOG"

echo "=== WAVE2 DONE ===" >> "$LOG"
