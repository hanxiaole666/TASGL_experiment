"""P0 数据入口：原始 Elliptic、DGraph-Fin 与无标签预测文件。"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ELLIPTIC_FILES = ('elliptic_txs_features.csv', 'elliptic_txs_classes.csv',
                  'elliptic_txs_edgelist.csv')


def load_elliptic_raw(data_dir='./data/elliptic', label_ceiling=None):
    """读取原始特征，不做全图标准化。

    label_ceiling 仅允许训练/验证程序看到截止时间内标签；之外统一置 -1。
    最终测试标签由 P0-1 独立评估器读取，不在开发或预测冻结阶段使用。
    """
    paths = [os.path.join(data_dir, n) for n in ELLIPTIC_FILES]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Elliptic 文件不存在: {path}')
    feat = pd.read_csv(paths[0], header=None)
    cls = pd.read_csv(paths[1])
    edges = pd.read_csv(paths[2])
    tx_ids = feat.iloc[:, 0].to_numpy()
    node_time = feat.iloc[:, 1].to_numpy().astype(np.int64)
    x = feat.iloc[:, 2:].to_numpy().astype(np.float32)
    index = {int(tx): i for i, tx in enumerate(tx_ids)}
    y = np.full(len(tx_ids), -1, dtype=np.int64)
    for tx, label in zip(cls['txId'], cls['class']):
        label = str(label)
        if label in ('1', '2'):
            # 与 data_loader.load_elliptic 一致：1=illicit 正类，2=licit。
            y[index[int(tx)]] = 1 if label == '1' else 0
    edge = edges.to_numpy(dtype=np.int64)
    valid = np.array([(int(a) in index and int(b) in index)
                      for a, b in edge], dtype=bool)
    edge = edge[valid]
    edge_index = np.array([[index[int(a)] for a in edge[:, 0]],
                           [index[int(b)] for b in edge[:, 1]]], dtype=np.int64)
    # 与既有协议一致：无向化，去重。
    undirected = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    edge_index = np.unique(undirected, axis=1)
    if label_ceiling is not None:
        y = y.copy()
        y[node_time > int(label_ceiling)] = -1
    return {'x': x, 'y': y, 'edge_index': edge_index,
            'node_time': node_time, 'tx_ids': tx_ids,
            'node_ids': tx_ids.copy(), 'n_features': x.shape[1],
            'time_semantics': 'node_transaction_time_step',
            'source_files': paths}


def load_dgraphfin(data_dir='./data/dgraphfin', label_ceiling=None):
    """读取官方 DGraphFin.npz；测试标签在开发阶段立即屏蔽。

    DGraph-Fin 提供 edge_timestamp 而非节点到达时间，因此本加载器支持官方
    掩码外部验证，但不能据此声称严格节点 rolling；P0-7 默认使用 Elliptic。
    """
    path = os.path.join(data_dir, 'dgraphfin.npz')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'DGraph-Fin 文件不存在: {path}')
    with np.load(path, allow_pickle=False) as z:
        required = {'x', 'y', 'edge_index', 'train_mask', 'valid_mask', 'test_mask'}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f'DGraph-Fin 缺字段: {sorted(missing)}')
        x = np.asarray(z['x'], dtype=np.float32)
        raw_y = np.asarray(z['y'], dtype=np.int64).reshape(-1)
        train_mask = np.asarray(z['train_mask']).reshape(-1).astype(bool)
        valid_mask = np.asarray(z['valid_mask']).reshape(-1).astype(bool)
        test_mask = np.asarray(z['test_mask']).reshape(-1).astype(bool)
        if len(train_mask) != len(x):
            # 官方文件部分版本只保存有标签节点的掩码；其长度应与 y 一致。
            if len(train_mask) == len(raw_y) and len(raw_y) != len(x):
                raise ValueError('掩码仅覆盖压缩标签数组；需显式节点 ID 映射，禁止猜测')
            raise ValueError('DGraph-Fin 掩码与节点数不一致')
        if len(raw_y) != len(x):
            raise ValueError('当前程序要求逐节点 y；不猜测官方压缩标签映射')
        y = raw_y.copy()
        visible = train_mask | valid_mask
        y[~visible] = -1
        result = {'x': x, 'y': y, 'edge_index': np.asarray(z['edge_index'],
                  dtype=np.int64), 'train_mask': train_mask,
                  'valid_mask': valid_mask, 'test_mask': test_mask,
                  'node_ids': np.arange(len(x)),
                  'edge_timestamp': (np.asarray(z['edge_timestamp'])
                                     if 'edge_timestamp' in z.files else None),
                  'n_features': x.shape[1], 'time_semantics': 'edge_timestamp_only',
                  'source_files': [path]}
    return result


def validate_prediction_npz(path):
    """冻结预测不得携带 y_true/labels，防止把测试答案混入锁产物。"""
    with np.load(path, allow_pickle=False) as z:
        forbidden = {'y', 'y_true', 'labels', 'target'} & set(z.files)
        if forbidden:
            raise ValueError(f'预测文件禁止包含标签字段: {sorted(forbidden)}')
        required = {'node_id', 'p_prob', 'seed'}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f'预测文件缺字段: {sorted(missing)}')
        node_id = np.asarray(z['node_id'])
        p = np.asarray(z['p_prob'], dtype=np.float64).reshape(-1)
        seed = int(np.asarray(z['seed']).reshape(-1)[0])
        if len(node_id) != len(p) or len(np.unique(node_id)) != len(node_id):
            raise ValueError('node_id 与概率长度不一致或节点重复')
        if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError('预测概率必须在 [0,1]')
        return {'node_id': node_id, 'p_prob': p, 'seed': seed,
                'path': os.path.abspath(path)}
