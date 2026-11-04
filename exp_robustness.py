"""exp_robustness.py — E5 对抗鲁棒性系统评测

四类攻击 (设计方案第七节), TASGL vs 稠密基线 (GCN/GAT):
  1. 特征扰动   欺诈节点特征加高斯噪声 σ ∈ {0.1, 0.3, 0.5}
  2. 结构扰动   随机删边 {10%, 30%, 50%}
  3. 对抗边注入 正常节点间注入虚假边 {10%, 30%}
  4. 标签翻转   训练标签翻转 {5%, 10%} (训练前施加)

核心叙事: 稀疏化本身是正则化 —— TASGL 的 F1 下降应显著小于稠密基线。

输出: results/robustness_results.json + _summary.csv (含各攻击下 ΔF1)
"""
import argparse
import numpy as np
import pandas as pd
import torch

from data_loader import load_elliptic, make_graph_data, standard_split, \
    compute_time_gaps
from models import build_model
from common import set_seed, train_model, save_json, load_if_done, \
    resume_paths, PROTOCOL_VERSION
from metrics import evaluate, tune_threshold


# ------------------------------------------------------------------
# 攻击实现 (仅污染测试图; 标签翻转在训练前施加)
# ------------------------------------------------------------------
def attack_feature_noise(data, test_mask, sigma, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    d = {k: v for k, v in data.items()}
    x = data['x'].clone()
    sel = test_mask.nonzero(as_tuple=True)[0]
    noise = torch.randn((sel.numel(), x.size(1)), generator=g).to(x.device) \
        * sigma
    x[sel] = x[sel] + noise
    d['x'] = x
    return d


def attack_edge_drop(data, rate, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    d = {k: v for k, v in data.items()}
    E = data['edge_index'].size(1)
    keep = torch.rand(E, generator=g) > rate
    keep = keep.to(data['edge_index'].device)
    d['edge_index'] = data['edge_index'][:, keep]
    d['time_gaps'] = data['time_gaps'][keep]
    return d


def attack_edge_inject(data, labels_np, rate, seed=0):
    rng = np.random.default_rng(seed)
    d = {k: v for k, v in data.items()}
    licit = np.where(labels_np == 0)[0]
    n_new = int(data['edge_index'].size(1) * rate)
    src = rng.choice(licit, n_new)
    dst = rng.choice(licit, n_new)
    new_edges = torch.tensor(np.stack([src, dst]), dtype=torch.long,
                             device=data['edge_index'].device)
    d['edge_index'] = torch.cat([data['edge_index'], new_edges], dim=1)
    d['time_gaps'] = torch.cat([data['time_gaps'],
                                torch.zeros(n_new, device=data['x'].device)])
    return d


def flip_labels(labels_np, train_mask_np, rate, seed=0):
    rng = np.random.default_rng(seed)
    y = labels_np.copy()
    idx = np.where(train_mask_np)[0]
    k = int(len(idx) * rate)
    flip = rng.choice(idx, k, replace=False)
    y[flip] = 1 - y[flip]
    return y


ATTACKS = {
    'clean':           [('0', None)],
    'feature_noise':   [('0.1', 0.1), ('0.3', 0.3), ('0.5', 0.5)],
    'edge_drop':       [('0.1', 0.1), ('0.3', 0.3), ('0.5', 0.5)],
    'edge_inject':     [('0.1', 0.1), ('0.3', 0.3)],
    'label_flip':      [('0.05', 0.05), ('0.10', 0.10)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/elliptic')
    ap.add_argument('--models', nargs='+', default=['tasgl', 'gcn', 'gat'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--out', default='results/robustness_results.json')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, labels, time_steps, edge_index, _ = load_elliptic(args.data_dir)
    tr_np, te_np = standard_split(time_steps, labels, 34)
    va_np = (time_steps > 29) & (time_steps <= 34) & (labels >= 0)
    tr_np = tr_np & ~va_np

    results = {}
    for name in args.models:
        results[name] = {}
        for seed in args.seeds:
            set_seed(seed)
            for atk_name, levels in ATTACKS.items():
                for lv, param in levels:
                    key = f'{atk_name}:{lv}'
                    rp, cp = resume_paths('robustness',
                                          f'{name}_seed{seed}_{key}')
                    cached = load_if_done(rp, args.overwrite)
                    if cached is not None:
                        results[name].setdefault(key, []).append(cached['f1'])
                        print(f'[{name}][{key}][{seed}] 已完成, 跳过 '
                              f'(F1={cached["f1"]:.4f})')
                        continue
                    # 标签翻转需重新训练, 其余攻击只污染测试图
                    if atk_name == 'label_flip':
                        y_train = flip_labels(labels, tr_np, param, seed)
                    else:
                        y_train = labels
                    # 训练时使用(可能被翻转的)标签
                    data = make_graph_data(X, y_train, time_steps,
                                           edge_index, device)
                    masks = [torch.tensor(m).to(device)
                             for m in (tr_np, va_np, te_np)]
                    set_seed(seed)
                    model = build_model(name, data['in_dim'])
                    lf = None if name == 'tasgl' else False
                    model, _ = train_model(model, data, *masks[:2], device,
                                           epochs=args.epochs, loss_fn=lf,
                                           ckpt_path=cp)
                    # 训练完成后才恢复真标签, 供验证集调阈值与测试集评估
                    data['y'] = torch.tensor(labels, dtype=torch.long,
                                             device=device)
                    # 测试期攻击
                    test_data = data
                    if atk_name == 'feature_noise' and param:
                        test_data = attack_feature_noise(data, masks[2],
                                                         param, seed)
                    elif atk_name == 'edge_drop' and param is not None:
                        test_data = attack_edge_drop(data, param, seed)
                    elif atk_name == 'edge_inject' and param:
                        test_data = attack_edge_inject(data, labels, param,
                                                       seed)
                    # 阈值在未受攻击的验证集上调优, 测试集保持攻击后评估
                    thr = tune_threshold(model, data, masks[1])
                    r = {**evaluate(model, test_data, masks[2], threshold=thr),
                         'threshold': thr, 'protocol': PROTOCOL_VERSION}
                    key = f'{atk_name}:{lv}'
                    save_json(r, rp)
                    results[name].setdefault(key, []).append(r['f1'])
                    print(f'[{name}][{atk_name}:{lv}][{seed}] '
                          f'F1={r["f1"]:.4f}')
    save_json(results, args.out)

    # --- 汇总: 相对 clean 的 F1 下降幅度 ---
    rows = []
    for name, d in results.items():
        clean = np.mean(d['clean:0'])
        for key, f1s in d.items():
            if key == 'clean:0':
                continue
            drop = (clean - np.mean(f1s)) / max(clean, 1e-9)
            rows.append({'model': name, 'attack': key,
                         'f1': f'{np.mean(f1s):.4f}±{np.std(f1s):.4f}',
                         'rel_drop': f'{drop * 100:.1f}%'})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out.replace('.json', '_summary.csv'), index=False)
    print('\n' + summary.to_string(index=False))


if __name__ == '__main__':
    main()
