"""data_loader.py — Elliptic / Ethereum Fraud 数据集加载与时序图构建

Elliptic 数据来源: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
需要三个文件放入 data_dir:
  elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv

Ethereum Fraud 数据来源: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def load_elliptic(data_dir='./data/elliptic'):
    """加载 Elliptic 数据集。

    Returns:
        X          : [N, 165] float32 标准化特征
        labels     : [N] int64, 1=illicit, 0=licit, -1=unknown
        time_steps : [N] int, 1..49
        edge_index : [2, E] long
        id2idx     : txId -> 行索引
    """
    # CSV 缺失时自动从 PyG 公开镜像下载 (免账号, 支持断点续传)
    from download_data import ensure_elliptic, ELLIPTIC_FILES
    if not all(os.path.exists(os.path.join(data_dir, f))
               for f in ELLIPTIC_FILES):
        ensure_elliptic(data_dir)

    features = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
    classes = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
    edges = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')

    tx_ids = features.iloc[:, 0].values
    time_steps = features.iloc[:, 1].values.astype(np.int64)
    X = features.iloc[:, 2:].values.astype(np.float32)

    X = StandardScaler().fit_transform(X).astype(np.float32)

    id2idx = {tid: i for i, tid in enumerate(tx_ids)}

    label_map = {'1': 1, '2': 0, 'unknown': -1}
    labels = (classes['class'].astype(str).map(label_map)
              .fillna(-1).values.astype(np.int64))
    # classes 与 features 行序一致(Kaggle 版本约定), 若不匹配则按 txId 对齐:
    if 'txId' in classes.columns and not (classes['txId'].values == tx_ids).all():
        aligned = np.full(len(tx_ids), -1, dtype=np.int64)
        for row in classes.itertuples():
            idx = id2idx.get(getattr(row, 'txId', None))
            if idx is not None:
                aligned[idx] = label_map.get(str(getattr(row, 'class')), -1)
        labels = aligned

    src = edges['txId1'].map(id2idx).values.astype(np.float64)
    dst = edges['txId2'].map(id2idx).values.astype(np.float64)
    valid = ~(np.isnan(src) | np.isnan(dst))
    edge_index = torch.tensor(
        np.stack([src[valid].astype(np.int64), dst[valid].astype(np.int64)]),
        dtype=torch.long)
    # 转为无向(金融风险沿双向传播), 同时保留有向语义由时序门控处理
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return X, labels, time_steps, edge_index, id2idx


def compute_time_gaps(edge_index, time_steps):
    """每条边两端节点的时间步差 Δt (float32 [E])。"""
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    gaps = np.abs(time_steps[row] - time_steps[col]).astype(np.float32)
    return torch.tensor(gaps, dtype=torch.float32)


def build_temporal_slices(X, labels, time_steps, edge_index):
    """按时间步切分子图, 供时序滚动评估 (E3) 使用。

    Returns: list of dict, 每个含 x/y/edge_index/local_time_steps
    """
    slices = []
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    for t in sorted(np.unique(time_steps)):
        mask = time_steps == t
        node_idx = np.where(mask)[0]
        node_map = {old: new for new, old in enumerate(node_idx)}
        sub_mask = mask[src] & mask[dst]
        if sub_mask.sum() > 0:
            sub_src = np.array([node_map[s] for s in src[sub_mask]])
            sub_dst = np.array([node_map[d] for d in dst[sub_mask]])
            sub_ei = torch.tensor(np.stack([sub_src, sub_dst]), dtype=torch.long)
        else:
            sub_ei = torch.zeros((2, 0), dtype=torch.long)
        slices.append({
            'time_step': int(t),
            'x': torch.tensor(X[node_idx], dtype=torch.float32),
            'y': torch.tensor(labels[node_idx], dtype=torch.long),
            'edge_index': sub_ei,
            'time_gaps': torch.zeros(sub_ei.size(1)),  # 同时间步内 Δt=0
        })
    return slices


def standard_split(time_steps, labels, train_end=34):
    """Elliptic 标准时序划分: 1..train_end 训练, 其余测试 (仅已标注节点)。"""
    train_mask = (time_steps <= train_end) & (labels >= 0)
    test_mask = (time_steps > train_end) & (labels >= 0)
    return train_mask, test_mask


def make_graph_data(X, labels, time_steps, edge_index, device):
    """打包为统一图数据字典 (全实验脚本共用)。"""
    time_gaps = compute_time_gaps(edge_index, time_steps)
    return {
        'x': torch.tensor(X, dtype=torch.float32).to(device),
        'y': torch.tensor(labels, dtype=torch.long).to(device),
        'edge_index': edge_index.to(device),
        'time_gaps': time_gaps.to(device),
        'time_steps': torch.tensor(time_steps, dtype=torch.long).to(device),
        'num_nodes': X.shape[0],
        'in_dim': X.shape[1],
    }


def load_ethereum_fraud(csv_path='./data/ethereum/transaction_dataset.csv',
                        out_dim=32, scaler_fit_X=None):
    """加载 Ethereum Fraud; 在目标域独立标准化并 PCA 至 out_dim 维。

    这只是无目标标签的维度对齐, 不是与 Elliptic 共享 PCA 基, 也不保证
    主成分语义一一对应。论文中应称 zero-target-label / 独立 PCA 弱基线。

    Returns: X_pca [N, out_dim], labels [N] (1=fraud, 0=normal)
    """
    from sklearn.decomposition import PCA
    if scaler_fit_X is not None:
        raise ValueError('不同原始特征空间不能直接用同一 PCA 基; '
                         '请使用目标域独立 PCA, 或显式实现语义映射')
    # CSV 缺失时自动从 GitHub 公开镜像下载 (免账号, 支持断点续传)
    if not os.path.exists(csv_path):
        from download_data import ensure_ethereum
        ensure_ethereum(os.path.dirname(csv_path) or './data/ethereum')
    df = pd.read_csv(csv_path)
    if 'FLAG' not in df.columns:
        raise ValueError('Ethereum Fraud 缺少 FLAG 标签列')
    y = df['FLAG'].values.astype(np.int64)
    feats = df.drop(columns=[c for c in df.columns
                             if c.lower() in ('flag', 'address', 'index',
                                              'unnamed: 0')])
    feats = feats.select_dtypes(include=[np.number]).fillna(0.0)
    X = StandardScaler().fit_transform(feats.values.astype(np.float32))
    if min(X.shape) < out_dim:
        raise ValueError(f'PCA 维数 {out_dim} 超过目标数据限制 {X.shape}')
    pca = PCA(n_components=out_dim).fit(X)
    return pca.transform(X).astype(np.float32), y


def load_dgraphfin(data_dir='./data/dgraphfin', device=None):
    """加载 DGraph-Fin (E1 增强 / 与 ST-CGNN 对比用)。

    npz 缺失时自动从官方 OSS 下载 (免账号, 断点续传)。
    官方划分: train_mask / valid_mask / test_mask 为节点索引数组。
    Returns: (data dict, masks dict) — 与 make_graph_data 兼容的字典结构。
    """
    npz_path = os.path.join(data_dir, 'dgraphfin.npz')
    if not os.path.exists(npz_path):
        from download_data import ensure_dgraphfin
        ensure_dgraphfin(data_dir)
    with np.load(npz_path) as z:
        x = torch.from_numpy(z['x']).float()
        y = torch.from_numpy(z['y']).long()
        edge_index = torch.from_numpy(z['edge_index']).long().t().contiguous()
        edge_time = torch.from_numpy(z['edge_timestamp']).long()
        n = x.size(0)
        masks = {}
        for split, key in (('train', 'train_mask'), ('val', 'valid_mask'),
                           ('test', 'test_mask')):
            idx = torch.from_numpy(z[key]).long()
            m = torch.zeros(n, dtype=torch.bool)
            m[idx] = True
            masks[split] = m
    data = {
        'x': x, 'y': y, 'edge_index': edge_index,
        'time_gaps': torch.zeros(edge_index.size(1)),
        'time_steps': edge_time, 'num_nodes': n, 'in_dim': x.size(1),
    }
    if device is not None:
        data = {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in data.items()}
        masks = {k: v.to(device) for k, v in masks.items()}
    return data, masks
