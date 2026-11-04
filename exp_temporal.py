"""exp_temporal.py — E3 时序鲁棒性: 滚动评估 + 时间外推

两部分:
  (A) 滚动评估: train_end ∈ {20, 25, 30, 34}, 全部固定在 35–49
      时间步测试, 避免不同测试区间造成伪“外推损失” (H4)
  (B) 逐时间步 F1: 标准划分 (train_end=34) 下每个测试时间步的 F1

输出:
  results/temporal_results.json
  figs/temporal_f1_curve.pdf  (论文 Figure: F1 vs time_step)
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_elliptic, make_graph_data, standard_split
from models import build_model
from common import set_seed, train_model, save_json, load_if_done, \
    resume_paths, PROTOCOL_VERSION
from metrics import eval_with_threshold, tune_threshold

# 仅滚动实验改变协议: 旧结果使用 >train_end 的不同测试区间, 不可比较。
# 使用独立协议号可使旧 rolling 缓存自动重跑, 其他已完成实验不受影响。
ROLLING_PROTOCOL_VERSION = 4


def per_timestep_f1(model, data, time_steps, labels, device, threshold=0.5):
    """逐测试时间步计算 F1 (阈值由验证集调优后传入)。"""
    model.eval()
    with torch.no_grad():
        logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                          data['num_nodes'], hard_mask=True)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = (probs >= threshold).astype(int)
    from sklearn.metrics import f1_score
    out = {}
    for t in sorted(np.unique(time_steps)):
        m = (time_steps == t) & (labels >= 0)
        if m.sum() > 0 and len(np.unique(labels[m])) > 1:
            out[int(t)] = f1_score(labels[m], preds[m], zero_division=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/elliptic')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--models', nargs='+', default=['tasgl', 'gcn', 'gat'])
    ap.add_argument('--train_ends', type=int, nargs='+',
                    default=[20, 25, 30, 34])
    ap.add_argument('--test_start', type=int, default=35,
                    help='所有 rolling 配置共用的固定测试起点 (默认 35)')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--out', default='results/temporal_results.json')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if not args.train_ends or args.test_start <= max(args.train_ends):
        raise ValueError('固定测试起点必须晚于所有 train_ends, 避免训练/测试重叠')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, labels, time_steps, edge_index, _ = load_elliptic(args.data_dir)
    data = make_graph_data(X, labels, time_steps, edge_index, device)

    results = {'rolling': {}, 'per_timestep': {},
               'rolling_protocol': ROLLING_PROTOCOL_VERSION,
               'fixed_test_start': args.test_start,
               'per_timestep_seed': 0}
    for name in args.models:
        # (A) 滚动评估
        for te_end in args.train_ends:
            key = f'{name}_trainle{te_end}'
            results['rolling'][key] = []
            for seed in args.seeds:
                rp, cp = resume_paths('temporal', f'{key}_seed{seed}')
                cached = load_if_done(rp, args.overwrite,
                                      protocol=ROLLING_PROTOCOL_VERSION)
                if cached is not None and (
                        cached.get('train_end') != te_end or
                        cached.get('test_start') != args.test_start or
                        cached.get('seed', seed) != seed):
                    print(f'[{key}][{seed}] 缓存划分/种子与本次配置不符, 重跑')
                    cached = None
                if cached is not None:
                    # 协议 4 首批结果未显式保存 seed; 文件路径已唯一确定种子,
                    # 因而只回填元数据, 不改指标, 也无需为此重新训练。
                    if 'seed' not in cached:
                        cached['seed'] = seed
                        save_json(cached, rp)
                        print(f'[{key}][{seed}] 已补记 seed 元数据 (数值不变)')
                    results['rolling'][key].append(cached)
                    print(f'[{key}][{seed}] 已完成, 跳过 '
                          f'(F1={cached["f1"]:.4f})')
                    continue
                set_seed(seed)
                tr = (time_steps <= te_end) & (labels >= 0)
                te = (time_steps >= args.test_start) & (labels >= 0)
                va = (time_steps > te_end - 5) & (time_steps <= te_end) \
                    & (labels >= 0)
                tr = tr & ~va
                masks = [torch.tensor(m).to(device) for m in (tr, va, te)]
                model = build_model(name, data['in_dim'])
                lf = None if name == 'tasgl' else False
                model, _ = train_model(model, data, *masks[:2], device,
                                       epochs=args.epochs, loss_fn=lf,
                                       ckpt_path=cp,
                                       protocol=ROLLING_PROTOCOL_VERSION)
                r = {**eval_with_threshold(model, data, masks[1], masks[2]),
                     'seed': seed, 'train_end': te_end,
                     'test_start': args.test_start,
                     'protocol': ROLLING_PROTOCOL_VERSION}
                save_json(r, rp)
                results['rolling'][key].append(r)
                print(f'[{key}][{seed}] 固定测试 {args.test_start}–49 '
                      f'F1={r["f1"]:.4f}')

        # (B) 标准划分逐时间步
        rp, cp = resume_paths('temporal', f'{name}_pertimestep')
        cached = load_if_done(rp, args.overwrite)
        if cached is not None and cached.get('seed', 0) != 0:
            print(f'[{name}] 逐时间步缓存种子不符, 重跑')
            cached = None
        if cached is not None:
            # 该缓存路径始终由下方固定的 seed=0 生成, 可安全补记。
            if 'seed' not in cached:
                cached['seed'] = 0
                save_json(cached, rp)
            results['per_timestep'][name] = {
                int(k): v for k, v in cached['values'].items()}
            print(f'[{name}] 逐时间步结果已存在, 跳过 (seed=0)')
            continue
        tr, te = standard_split(time_steps, labels, 34)
        va = (time_steps > 29) & (time_steps <= 34) & (labels >= 0)
        tr = tr & ~va
        masks = [torch.tensor(m).to(device) for m in (tr, va, te)]
        set_seed(0)
        model = build_model(name, data['in_dim'])
        lf = None if name == 'tasgl' else False
        model, _ = train_model(model, data, *masks[:2], device,
                               epochs=args.epochs, loss_fn=lf, ckpt_path=cp)
        thr = tune_threshold(model, data, masks[1])
        results['per_timestep'][name] = per_timestep_f1(
            model, data, time_steps, labels, device, threshold=thr)
        save_json({'protocol': PROTOCOL_VERSION, 'seed': 0,
                   'threshold': thr,
                   'values': results['per_timestep'][name]}, rp)

    save_json(results, args.out)

    # --- 论文图: F1 vs time_step ---
    plt.figure(figsize=(3.5, 2.6))
    for name in args.models:
        d = results['per_timestep'][name]
        ts = sorted(k for k in d if k >= 35)
        style = '-*' if name == 'tasgl' else '--o'
        plt.plot(ts, [d[t] for t in ts], style, markersize=4,
                 linewidth=1.5 if name == 'tasgl' else 1.0,
                 label=name.upper())
    plt.xlabel('Time step')
    plt.ylabel('F1 (illicit), seed 0')
    plt.legend(fontsize=7)
    plt.tight_layout()
    import os
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/temporal_f1_curve.pdf')
    plt.close()
    print('saved figs/temporal_f1_curve.pdf')

    # --- 时间外推损失摘要 (H4) ---
    for name in args.models:
        k34 = f'{name}_trainle34'
        for te_end in (20, 25):
            k = f'{name}_trainle{te_end}'
            if k in results['rolling'] and k34 in results['rolling']:
                f_short = np.mean([r['f1'] for r in results['rolling'][k]])
                f_full = np.mean([r['f1'] for r in results['rolling'][k34]])
                print(f'{name}: train≤{te_end} vs ≤34, 固定测试 '
                      f'{args.test_start}–49 的外推损失 = '
                      f'{(f_full - f_short) / max(f_full, 1e-9) * 100:.1f}%')


if __name__ == '__main__':
    main()
