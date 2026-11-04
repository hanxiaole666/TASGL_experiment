#!/usr/bin/env bash
# BitHeNet 独立确认重跑 (wave1 第 3 步因 CUDA OOM 失败)
#   先尝试等预算 HPO 胜出配置 + expandable_segments;
#   若仍 OOM, 回退到默认配置并在日志中记录。
set -u
cd "$(cd "$(dirname "$0")" && pwd)"
PY="${TASGL_PYTHON:-python3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [try 1] BitHeNet with tuned config ===" >> results/p4/wave2.log
date >> results/p4/wave2.log
$PY p4_experiments.py second-dataset run \
  --models tasgl graphsage mlp \
  --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 120 \
  --search results/p4/hpo_search.json \
  --out results/p4/second_dataset_confirm.json >> results/p4/wave2.log 2>&1
rc=$?
echo "try1 rc=$rc" >> results/p4/wave2.log

if [ $rc -ne 0 ]; then
  echo "=== [try 2] fallback: default config ===" >> results/p4/wave2.log
  date >> results/p4/wave2.log
  $PY p4_experiments.py second-dataset run \
    --models tasgl graphsage mlp \
    --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 120 \
    --search results/p4/__none__.json \
    --out results/p4/second_dataset_confirm.json >> results/p4/wave2.log 2>&1
  echo "try2 rc=$?" >> results/p4/wave2.log
  date >> results/p4/wave2.log
fi
echo "=== WAVE2 DONE ===" >> results/p4/wave2.log
