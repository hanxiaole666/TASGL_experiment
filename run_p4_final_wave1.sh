#!/usr/bin/env bash
# P4 终版 第一波重跑 (修复 build_model 超参透传缺陷之后)
#   1) 等预算 HPO 搜索 (7 模型 x 6 trial x 2 seed x 100 epoch)
#   2) Elliptic 十种子主比较 (使用修正后的 HPO 胜出配置)
#   3) BitHeNet 独立数据集确认 (同一套配置)
set -u
cd "$(cd "$(dirname "$0")" && pwd)"
PY="${TASGL_PYTHON:-python3}"
mkdir -p results/p4

echo "=== [1/3] HPO search ===" | tee -a results/p4/wave1.log
date >> results/p4/wave1.log
$PY p4_experiments.py hpo search \
  --models tasgl mlp gcn gat graphsage sgc dsgat \
  --seeds 0 1 --trials 6 --epochs 100 \
  --out results/p4/hpo_search.json >> results/p4/wave1.log 2>&1
echo "hpo rc=$?" >> results/p4/wave1.log
date >> results/p4/wave1.log

echo "=== [2/3] final 10-seed ===" | tee -a results/p4/wave1.log
$PY p4_experiments.py final \
  --models tasgl graphsage mlp \
  --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 200 \
  --search results/p4/hpo_search.json \
  --out results/p4/final_comparison.json >> results/p4/wave1.log 2>&1
echo "final rc=$?" >> results/p4/wave1.log
date >> results/p4/wave1.log

echo "=== [3/3] BitHeNet confirmation ===" | tee -a results/p4/wave1.log
$PY p4_experiments.py second-dataset run \
  --models tasgl graphsage mlp \
  --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 120 \
  --search results/p4/hpo_search.json \
  --out results/p4/second_dataset_confirm.json >> results/p4/wave1.log 2>&1
echo "second rc=$?" >> results/p4/wave1.log
date >> results/p4/wave1.log
echo "=== WAVE1 DONE ===" >> results/p4/wave1.log
