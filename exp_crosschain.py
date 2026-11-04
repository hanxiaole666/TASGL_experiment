"""exp_crosschain.py — E6 跨链泛化: Elliptic → Ethereum 零目标标签迁移

方法:
  - Elliptic 与 Ethereum Fraud 各自标准化并独立 PCA 至 out_dim 维
  - 模型只在 Elliptic 标签上训练, Ethereum 标签仅用于最终评估
  - Ethereum 侧无图结构 -> 构建 kNN 交易图 (特征相似度 top-k)

注意 (必须写入论文局限): 独立 PCA 只统一维数, 不对齐主成分语义;
目标域标准化/PCA/kNN 使用了无标签目标特征, 因而属于 zero-target-label
的弱迁移基线, 而非严格 inductive zero-shot 或共享语义空间。

输出: results/crosschain_results.json
"""
import argparse
import numpy as np
import torch
from sklearn.decomposition import PCA

from data_loader import load_elliptic, make_graph_data, standard_split, \
    load_ethereum_fraud
from models import build_model
from common import set_seed, train_model, save_json, load_if_done, \
    resume_paths
from metrics import evaluate, tune_threshold


# 独立 PCA 与旧共享 PCA 的结果不可混用; 只作废跨链缓存, 不影响其他实验。
CROSSCHAIN_PROTOCOL_VERSION = 4


def knn_graph(X, k=8):
    """特征空间 kNN 图; 显式屏蔽对角线, 不假设自相似度一定最高。"""
    n = X.shape[0]
    if not 1 <= k < n:
        raise ValueError(f'kNN 的 k={k} 必须满足 1 <= k < {n}')
    xt = torch.as_tensor(X, dtype=torch.float32)
    sim = xt @ xt.T / (xt.norm(dim=1, keepdim=True) @
                       xt.norm(dim=1, keepdim=True).T + 1e-9)
    sim.fill_diagonal_(-float('inf'))
    idx = sim.topk(k, dim=1).indices
    src = torch.arange(n).unsqueeze(1).expand(-1, k).flatten()
    dst = idx.flatten()
    return torch.stack([src, dst])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--elliptic_dir', default='./data/elliptic')
    ap.add_argument('--eth_csv',
                    default='./data/ethereum/transaction_dataset.csv')
    ap.add_argument('--out_dim', type=int, default=32)
    ap.add_argument('--models', nargs='+', default=['tasgl', 'gcn'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--out', default='results/crosschain_results.json')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 源域: Elliptic (源域独立 PCA 降维) ---
    X, labels, time_steps, edge_index, _ = load_elliptic(args.elliptic_dir)
    pca = PCA(n_components=args.out_dim).fit(X)
    Xs = pca.transform(X).astype(np.float32)
    src_data = make_graph_data(Xs, labels, time_steps, edge_index, device)
    tr, te = standard_split(time_steps, labels, 34)
    va = (time_steps > 29) & (time_steps <= 34) & (labels >= 0)
    tr = tr & ~va
    masks = [torch.tensor(m).to(device) for m in (tr, va, te)]

    # --- 目标域: Ethereum (独立 PCA + kNN 图) ---
    # loader 已返回 out_dim 维; 不得再送入期待 165 维的源域 PCA。
    Xt, yt = load_ethereum_fraud(args.eth_csv, args.out_dim)
    if Xt.shape[1] != Xs.shape[1]:
        raise ValueError(f'跨链维数不一致: source={Xs.shape}, target={Xt.shape}')
    et_edge = knn_graph(Xt, k=8)
    tgt_data = {
        'x': torch.tensor(Xt, dtype=torch.float32).to(device),
        'y': torch.tensor(yt, dtype=torch.long).to(device),
        'edge_index': et_edge.to(device),
        'time_gaps': torch.zeros(et_edge.size(1)).to(device),
        'num_nodes': Xt.shape[0], 'in_dim': Xt.shape[1],
    }
    tgt_mask = torch.ones(Xt.shape[0], dtype=torch.bool).to(device)

    results = {'_meta': {
        'protocol': CROSSCHAIN_PROTOCOL_VERSION,
        'alignment': 'independent_pca',
        'semantic_alignment': False,
        'interpretation': 'diagnostic_only_not_transfer_evidence',
        'out_dim': args.out_dim,
        'target_labels_used_for_training': False,
        'target_features_used_transductively': True,
        'source_nodes': int(Xs.shape[0]), 'target_nodes': int(Xt.shape[0]),
        'knn_k': 8,
    }}
    print('[跨链边界] 独立 PCA 不对齐主成分语义; 结果仅作弱迁移诊断, '
          '不能单独作为跨链泛化能力证据')
    for name in args.models:
        results[name] = []
        for seed in args.seeds:
            rp, cp = resume_paths('crosschain', f'{name}_seed{seed}')
            cached = load_if_done(rp, args.overwrite,
                                  protocol=CROSSCHAIN_PROTOCOL_VERSION)
            if cached is not None:
                results[name].append(cached)
                print(f'[{name}][{seed}] 已完成, 跳过 '
                      f'(target F1={cached["target_f1"]:.4f})')
                continue
            set_seed(seed)
            model = build_model(name, args.out_dim)
            lf = None if name == 'tasgl' else False
            model, _ = train_model(model, src_data, *masks[:2], device,
                                   epochs=args.epochs, loss_fn=lf,
                                   ckpt_path=cp,
                                   protocol=CROSSCHAIN_PROTOCOL_VERSION)
            thr = tune_threshold(model, src_data, masks[1])
            src_f1 = evaluate(model, src_data, masks[2], threshold=thr)['f1']
            # 阈值只来自源域验证集, 不使用目标域任何标签。
            tgt = evaluate(model, tgt_data, tgt_mask, threshold=thr)
            r = {'seed': seed, 'source_f1': src_f1,
                 'target_f1': tgt['f1'],
                 'target_pr_auc': tgt['pr_auc'], 'threshold': thr,
                 'protocol': CROSSCHAIN_PROTOCOL_VERSION}
            save_json(r, rp)
            results[name].append(r)
            print(f'[{name}][{seed}] src F1={src_f1:.4f} -> '
                  f'target F1={tgt["f1"]:.4f} '
                  f'PR-AUC={tgt["pr_auc"]:.4f}')
    save_json(results, args.out)


if __name__ == '__main__':
    main()
