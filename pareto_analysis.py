"""pareto_analysis.py — 成本-精度 Pareto 前沿计算与论文级可视化

输入: results/main_results.json (train.py 输出)
输出:
  figs/pareto_params.pdf / figs/pareto_latency.pdf  (含误差棒)
  results/pareto_table.csv
  results/pareto_frontier_check.json   TASGL 是否位于前沿 (投稿检查项)
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COST_KEYS = {'params': '# Trainable parameters',
             'latency_ms': 'Inference latency (ms)'}

# 固定模型集的论文图标注位置; 避免相同参数量的模型文字互相覆盖。
# 未知模型仍使用通用交错偏移, 不影响数值或前沿计算。
LABEL_OFFSETS = {
    'params': {
        'tasgl': (-12, -18, 'right'), 'mlp': (-10, 11, 'right'),
        'graphsage': (8, 8, 'left'), 'gcn': (8, 7, 'left'),
        'gat': (-8, 11, 'right'), 'dsgat': (8, -13, 'left'),
        'sgc': (8, -12, 'left'),
    },
    'latency_ms': {
        'tasgl': (8, -18, 'left'), 'mlp': (8, -12, 'left'),
        'graphsage': (8, 8, 'left'), 'gcn': (-8, 0, 'right'),
        'gat': (8, 9, 'left'), 'dsgat': (-8, -14, 'right'),
        'sgc': (8, -12, 'left'),
    },
}


def pareto_frontier(points, cost, acc):
    """返回非支配点索引；相同成本时只保留最高精度，与输入顺序无关。"""
    points = np.asarray(points)
    cost = np.asarray(cost, dtype=float)
    acc = np.asarray(acc, dtype=float)
    if points.ndim != 1 or not (len(points) == len(cost) == len(acc)):
        raise ValueError('points、cost、acc 必须是一维且长度相同')
    if not (np.isfinite(cost).all() and np.isfinite(acc).all()):
        raise ValueError('Pareto 成本与精度必须为有限值')
    front = []
    for i in range(len(points)):
        # j 支配 i：成本不高、精度不低，且至少一项严格更优。
        dominated = any(
            cost[j] <= cost[i] and acc[j] >= acc[i] and
            (cost[j] < cost[i] or acc[j] > acc[i])
            for j in range(len(points)) if j != i)
        if not dominated:
            front.append(i)
    # 绘图需要按成本升序连线；同成本高精度点优先，最后按原位置稳定排序。
    return sorted(front, key=lambda i: (cost[i], -acc[i], i))


def plot_pareto(df, cost_key, save_path, highlight='tasgl'):
    plt.figure(figsize=(3.5, 2.8))
    cost = df[cost_key].values
    acc = df['f1_mean'].values
    front = pareto_frontier(np.arange(len(df)), cost, acc)
    # 前沿连线
    plt.plot(cost[front], acc[front], ':', color='gray', linewidth=1.0,
             zorder=1)
    for i, row in df.iterrows():
        is_hl = row['model'] == highlight
        plt.errorbar(row[cost_key], row['f1_mean'], yerr=row['f1_std'],
                     fmt='*' if is_hl else 'o',
                     color='#d62728' if is_hl else None,
                     markersize=14 if is_hl else 7, capsize=3,
                     linewidth=1.5 if is_hl else 1.0,
                     zorder=3 if is_hl else 2)
        dx, dy, ha = LABEL_OFFSETS.get(cost_key, {}).get(
            row['model'], (6, 8 if i % 2 else -12, 'left'))
        plt.annotate(row['model'].upper(),
                     (row[cost_key], row['f1_mean']),
                     textcoords='offset points', xytext=(dx, dy), ha=ha,
                     fontsize=7, color='#d62728' if is_hl else 'gray')
    plt.xscale('log')
    plt.xlabel(COST_KEYS[cost_key])
    plt.ylabel('Test F1 (illicit)')
    plt.margins(x=0.16, y=0.16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f'saved {save_path}')


def check_on_frontier(df, highlight='tasgl'):
    out = {}
    for ck in COST_KEYS:
        front = pareto_frontier(np.arange(len(df)), df[ck].values,
                                df['f1_mean'].values)
        idx = df.index[df['model'] == highlight]
        out[ck] = bool(len(idx) and idx[0] in front)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results/main_results.json')
    ap.add_argument('--highlight', default='tasgl')
    args = ap.parse_args()

    if not os.path.exists(args.results):
        raise SystemExit(
            f'[错误] 未找到结果文件: {args.results}\n'
            f'请先运行: python train.py --seeds 0 1 2 3 4\n'
            f'(train.py 支持断点续跑, 中断后重跑同一命令即可; '
            f'若只想先画图, 可用 --results 指定已有结果文件。)')

    with open(args.results) as f:
        results = json.load(f)
    os.makedirs('results', exist_ok=True)

    rows = []
    for name, runs in results.items():
        df_r = pd.DataFrame(runs)
        rows.append({
            'model': name,
            'f1_mean': df_r['f1'].mean(), 'f1_std': df_r['f1'].std(),
            'pr_auc_mean': df_r['pr_auc'].mean(),
            'params': df_r['params'].iloc[0],
            'latency_ms': df_r['latency_ms'].mean(),
            'peak_mem_mb': df_r['peak_mem_mb'].mean(),
            'throughput': df_r['throughput'].mean(),
            'model_size_mb': df_r['model_size_mb'].iloc[0],
        })
    df = pd.DataFrame(rows).reset_index(drop=True)
    df.to_csv('results/pareto_table.csv', index=False)
    print(df.to_string(index=False))

    plot_pareto(df, 'params', 'figs/pareto_params.pdf', args.highlight)
    plot_pareto(df, 'latency_ms', 'figs/pareto_latency.pdf', args.highlight)

    check = check_on_frontier(df, args.highlight)
    with open('results/pareto_frontier_check.json', 'w') as f:
        json.dump(check, f, indent=2)
    print('\nTASGL on frontier:', check)
    if not all(check.values()):
        print('[结论] 当前结果不支持 TASGL 在所有成本维度上位于 Pareto 前沿; '
              '请如实报告, 不要根据测试集结果反复调参。若修改模型设计, '
              '需预先锁定新方案并按完整协议重跑。')


if __name__ == '__main__':
    main()
