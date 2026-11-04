"""p4_bitheteronet.py — BitHeNet 第二数据集加载与语义对齐审计

为什么选它作为独立确认数据集
----------------------------
审稿意见 2/要求 1: "单一数据集 (只有 Elliptic 端到端)" 与 "必须有一个
未参与开发的独立数据集, 且语义对齐而不是独立 PCA"。

BitHeNet 与 Elliptic 的对齐是**任务/标签/时间协议层面**的对齐:
  * 同为比特币交易级"违法 vs 合法"二分类
  * 标签极性一致 (1=illicit, 0=licit), 违法占比 4545/46045 = 9.87%,
    与 Elliptic 的 9.8% 几乎相同
  * 时间粒度一致: 同为 49 个时间步, 可用完全相同的
    "训练 1–34 / 测试 35–49" 时序协议
  * 特征**不做跨数据集 PCA 投影**: 每个数据集使用自己的原生特征,
    标准化器只在该数据集训练窗内拟合

本数据集在本项目中从未被任何开发过程使用 (prior_exposure=false),
因此在其官方时间划分上的一次性锁定评估构成真正的独立确认。

图构建口径 (必须写明)
--------------------
交易-交易边: 两笔交易共享同一个地址 (同为输入或同为输出地址)。
为避免交易所类枢纽地址产生巨大团, 单个地址关联的交易数超过
max_group 时跳过该地址的组内连边。该口径会低估枢纽处的连边密度,
必须在论文中说明。
"""
import hashlib
import json
import os

import numpy as np

LABEL_MAP = {0: 0, 1: 1, 2: -1}     # 0=licit, 1=illicit, 2=unknown


def load_bitheteronet(data_dir='./data/bitheteronet', max_group=20,
                      build_edges=True, verbose=False):
    """加载 BitHeNet 交易图。

    Returns:
        X        : [N, 182] float32 (未标准化, 由调用方按训练窗拟合)
        labels   : [N] int64, 1=illicit, 0=licit, -1=unknown
        ts       : [N] int64, 1..49
        edge_index: [2, E] int64 (无向)
        meta     : dict
    """
    import pickle
    tx_attrs = np.load(os.path.join(data_dir, 'tx_attrs.npy'))
    tx_classes = np.load(os.path.join(data_dir, 'tx_classes.npy'))
    tx_ts = np.load(os.path.join(data_dir, 'tx_time_step.npy'))
    if tx_attrs.dtype != np.float32:
        tx_attrs = tx_attrs.astype(np.float32)
    labels = np.vectorize(lambda c: LABEL_MAP.get(int(c), -1))(tx_classes)
    labels = labels.astype(np.int64)
    ts = tx_ts.astype(np.int64)
    n = int(labels.shape[0])

    edge_index = np.zeros((2, 0), dtype=np.int64)
    if build_edges:
        with open(os.path.join(data_dir, 'data.pkl'), 'rb') as f:
            d = pickle.load(f)
        groups = {}
        for key in ('Tx_id2Output_Addr_ids', 'Tx_id2Input_Addr_ids'):
            if key not in d:
                continue
            for tx, addrs in d[key].items():
                t = int(tx)
                if t >= n:
                    continue
                for a in addrs:
                    groups.setdefault(int(a), []).append(t)
        src, dst = [], []
        n_skipped = 0
        for a, txs in groups.items():
            if len(txs) < 2 or len(txs) > max_group:
                if len(txs) > max_group:
                    n_skipped += 1
                continue
            uniq = sorted(set(txs))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    src.append(uniq[i])
                    dst.append(uniq[j])
        if src:
            e = np.stack([np.asarray(src), np.asarray(dst)])
            edge_index = np.concatenate([e, e[::-1]], axis=1).astype(np.int64)
        if verbose:
            print(f'[BitHeNet] 地址组 {len(groups)}, 超上限跳过 {n_skipped}, '
                  f'无向边 {edge_index.shape[1]}')

    meta = {
        'schema': 'p4_bitheteronet_v1',
        'n_transactions': n,
        'n_features': int(tx_attrs.shape[1]),
        'n_edges_undirected': int(edge_index.shape[1]),
        'max_group': int(max_group),
        'n_labeled': int((labels >= 0).sum()),
        'n_illicit': int((labels == 1).sum()),
        'n_licit': int((labels == 0).sum()),
        'illicit_rate_labeled': float((labels == 1).sum() /
                                      max((labels >= 0).sum(), 1)),
        'time_steps': [int(ts.min()), int(ts.max())],
        'label_map': {'0': 'licit', '1': 'illicit', '2': 'unknown'},
        'edge_rule': '两笔交易共享同一地址(输入或输出)则连边; '
                     f'单地址关联交易数 > {max_group} 时跳过组内连边',
        'edge_rule_limitation': '会低估交易所/混币器等枢纽地址处的连边密度',
        'prior_exposure': False,
        'prior_exposure_note': '本数据集在本项目中从未用于任何开发/选择/调参',
    }
    meta['fingerprint'] = hashlib.sha256(
        json.dumps({k: meta[k] for k in
                    ('n_transactions', 'n_features',
                     'n_edges_undirected', 'max_group', 'n_illicit',
                     'time_steps')}, sort_keys=True).encode()).hexdigest()[:24]
    return tx_attrs, labels, ts, edge_index, meta


def semantic_alignment_report(a, b):
    """两个数据集的语义对齐审计。a/b 为 load_* 返回的 meta 风格字典。

    只审计"可证实的对齐": 任务类型、标签极性、正类占比、时间粒度。
    特征空间不做对齐声明 (跨数据集 PCA 投影被明确拒绝)。
    """
    def _g(d, k, *alts):
        for kk in (k,) + alts:
            if kk in d and d[kk] is not None:
                return d[kk]
        return None

    checks = {
        'same_task': _g(a, 'task') == _g(b, 'task'),
        'same_label_polarity': _g(a, 'label_polarity') == _g(b, 'label_polarity'),
        'positive_rate_within_2x': None,
        'same_time_granularity': _g(a, 'time_steps') == _g(b, 'time_steps'),
    }
    pa, pb = _g(a, 'illicit_rate_labeled'), _g(b, 'illicit_rate_labeled')
    if pa and pb:
        checks['positive_rate_within_2x'] = bool(
            0.5 <= (pa / pb) <= 2.0)
    return {
        'schema': 'p4_semantic_alignment_v1',
        'datasets': [_g(a, 'name'), _g(b, 'name')],
        'checks': checks,
        'positive_rate': {str(_g(a, 'name')): pa, str(_g(b, 'name')): pb},
        'feature_space_aligned': False,
        'feature_space_note': ('不使用跨数据集 PCA 投影; 每个数据集使用自己的'
                               '原生特征, 标准化器只在该数据集训练窗内拟合'),
        'aligned': all(v for v in checks.values()
                       if v is not None) and
                   all(v is not None for v in checks.values()),
    }


def elliptic_meta(labels, ts, name='elliptic'):
    return {'name': name, 'task': 'bitcoin_tx_illicit_detection',
            'label_polarity': '1=illicit,0=licit',
            'illicit_rate_labeled': float((labels == 1).sum() /
                                          max((labels >= 0).sum(), 1)),
            'time_steps': [int(ts.min()), int(ts.max())]}


def bitheteronet_meta(meta):
    return {'name': 'bitheteronet',
            'task': 'bitcoin_tx_illicit_detection',
            'label_polarity': '1=illicit,0=licit',
            'illicit_rate_labeled': meta['illicit_rate_labeled'],
            'time_steps': meta['time_steps']}
