"""train.py — E1 主实验 + E2 效率对比 (支持断点续跑)

用法:
  python train.py --seeds 0 1 2 3 4
  python train.py --seeds 0 1 2 3 4 --overwrite   # 忽略历史结果全部重跑

断点续跑机制 (两级):
  1. 运行级: 每个 (模型, 种子) 完成后立即写入
     results/runs/{model}_seed{seed}.json; 闪退后重跑自动跳过已完成组合。
  2. epoch 级: 训练中每个 epoch 保存
     results/runs/{model}_seed{seed}.ckpt (模型+优化器+最优状态),
     闪退后从最近 epoch 继续训练, 完成后自动删除。

输出:
  results/main_results.json     全部模型 × 种子的精度 + 效率指标
  results/main_summary.csv      均值±std 汇总表 (论文 Table 2 数据源)
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch

from data_loader import load_elliptic, make_graph_data, standard_split
from models import build_model
from common import set_seed, train_model, save_json, load_if_done, \
    PROTOCOL_VERSION
from metrics import eval_with_threshold, measure_efficiency, wilcoxon_test


def run_one(model_name, data, train_mask, val_mask, test_mask, device,
            seed, epochs, tasgl_kw, ckpt_path=None):
    set_seed(seed)
    model = build_model(model_name, data['in_dim'], **tasgl_kw)
    # TASGL 走 TASGLLoss (loss_fn=None 触发), 基线走类别加权 CE (loss_fn=False)
    lf = None if model_name == 'tasgl' else False
    model, _ = train_model(model, data, train_mask, val_mask, device,
                           epochs=epochs, loss_fn=lf, verbose=False,
                           ckpt_path=ckpt_path)
    acc = eval_with_threshold(model, data, val_mask, test_mask,
                              with_uncertainty=(model_name == 'tasgl'))
    eff = measure_efficiency(model, data, device)
    return {**acc, **eff, 'protocol': PROTOCOL_VERSION}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/elliptic')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--models', nargs='+',
                    default=['tasgl', 'mlp', 'gcn', 'gat', 'graphsage',
                             'sgc', 'dsgat'])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--hidden', type=int, default=48)
    ap.add_argument('--rff_dim', type=int, default=16)
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--num_hops', type=int, default=4)
    ap.add_argument('--out', default='results/main_results.json')
    ap.add_argument('--overwrite', action='store_true',
                    help='忽略已有断点与结果, 全部重新训练')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    run_dir = os.path.join(os.path.dirname(args.out) or '.', 'runs')
    os.makedirs(run_dir, exist_ok=True)

    X, labels, time_steps, edge_index, _ = load_elliptic(args.data_dir)
    data = make_graph_data(X, labels, time_steps, edge_index, device)
    train_np, test_np = standard_split(time_steps, labels, train_end=34)
    # 验证集 = 训练时间步的最后 5 步
    val_np = (time_steps > 29) & (time_steps <= 34) & (labels >= 0)
    train_np = train_np & ~val_np
    train_mask = torch.tensor(train_np).to(device)
    val_mask = torch.tensor(val_np).to(device)
    test_mask = torch.tensor(test_np).to(device)
    print(f'train/val/test = {train_np.sum()}/{val_np.sum()}/{test_np.sum()}')

    tasgl_kw = dict(hidden=args.hidden, rff_dim=args.rff_dim,
                    rank=args.rank, num_hops=args.num_hops)
    results = {}
    total = len(args.models) * len(args.seeds)
    done = 0
    for name in args.models:
        results[name] = []
        for seed in args.seeds:
            done += 1
            run_path = os.path.join(run_dir, f'{name}_seed{seed}.json')
            ckpt_path = os.path.join(run_dir, f'{name}_seed{seed}.ckpt')
            cached = load_if_done(run_path, args.overwrite)
            if cached is not None:
                r = cached
                print(f'[{done}/{total}] {name} seed={seed}: '
                      f'已完成, 跳过 (F1={r["f1"]:.4f})')
            else:
                if args.overwrite and os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                print(f'[{done}/{total}] 训练 {name} seed={seed} ...')
                r = run_one(name, data, train_mask, val_mask, test_mask,
                            device, seed, args.epochs, tasgl_kw,
                            ckpt_path=ckpt_path)
                r['seed'] = seed
                save_json(r, run_path)   # 单个组合完成立即落盘
                print(f'[{name}][seed={seed}] F1={r["f1"]:.4f} '
                      f'PR-AUC={r["pr_auc"]:.4f} params={r["params"]:,} '
                      f'latency={r["latency_ms"]:.2f}ms')
            results[name].append(r)
        save_json(results, args.out)     # 每个模型结束同步主结果文件
    save_json(results, args.out)

    # --- 汇总表 ---
    rows = []
    for name, runs in results.items():
        df = pd.DataFrame(runs)
        row = {'model': name}
        for k in ('f1', 'pr_auc', 'recall_at_fpr_1pct', 'latency_ms',
                  'peak_mem_mb', 'throughput'):
            row[k] = f'{df[k].mean():.4f}±{df[k].std():.4f}'
        row['params'] = int(df['params'].iloc[0])
        row['model_size_mb'] = df['model_size_mb'].iloc[0]
        if 'ece' in df:
            row['ece'] = f'{df["ece"].mean():.4f}'
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out.replace('.json', '_summary.csv'), index=False)
    print('\n' + summary.to_string(index=False))

    # --- Wilcoxon 显著性 (TASGL vs 各基线, 跨种子配对) ---
    if 'tasgl' in results:
        sig = {}
        for name, runs in results.items():
            if name == 'tasgl':
                continue
            a = [r['f1'] for r in results['tasgl']]
            b = [r['f1'] for r in runs]
            sig[f'tasgl_vs_{name}'] = wilcoxon_test(a, b)
        save_json(sig, args.out.replace('.json', '_significance.json'))
        print('\nWilcoxon:', sig)


if __name__ == '__main__':
    main()
