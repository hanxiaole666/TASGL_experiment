#!/usr/bin/env bash
# run_p4.sh — P4 终版证据层一键执行 (Linux/macOS/Git Bash)
# 用法: bash run_p4.sh          全流程
#       bash run_p4.sh quick    只跑不需要长训练的部分
set -u
PY=${PY:-python}
MODE=${1:-all}

run () { echo "=============== $* ==============="; "$PY" "$@" || exit $?; }

run p4_experiments.py env --out results/p4/environment.json
run p4_experiments.py thresholds justify --out results/p4/threshold_justification.json
run p4_experiments.py rolling plan --out results/p4/rolling_plan.json
run p4_experiments.py second-dataset plan --out results/p4/second_dataset_plan.json

if [ "$MODE" = "quick" ]; then echo "[P4] quick 模式完成"; exit 0; fi

run p4_experiments.py rootcause --seeds 0 --epochs 200 --reproduce 0.3049 \
    --out results/p4/rootcause_seed0.json
run p4_experiments.py gate-audit --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 200 \
    --out results/p4/gate_audit_10seeds.json
run p4_experiments.py hpo search --trials 6 --seeds 0 1 --epochs 100 \
    --out results/p4/hpo_search.json
run p4_experiments.py hpo audit --budget p4_hpo_budget.json \
    --search results/p4/hpo_search.json --out results/p4/hpo_audit.json
run p4_experiments.py final --seeds 0 1 2 3 4 5 6 7 8 9 \
    --models tasgl graphsage mlp --epochs 200 \
    --out results/p4/final_comparison.json
run p4_experiments.py second-dataset run --seeds 0 1 2 3 4 5 6 7 8 9 \
    --models tasgl graphsage mlp --epochs 100 \
    --out results/p4/second_dataset_independent.json
run p4_experiments.py cost measure --epochs 30 --out results/p4/cost_measurement.json
run p4_experiments.py report --out results/p4/report.json

echo "[P4] 全流程完成"
