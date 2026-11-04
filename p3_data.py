"""p3_data.py — 真实数据集文件的兼容加载器。

存在理由：`p0_data.load_dgraphfin` 假定 train/valid/test 是长度等于节点数的
布尔向量。本机真实 `DGraphFin.npz` 的掩码是**索引数组**
（train=857,899 / valid=183,862 / test=183,840，节点 3,700,550），且标签
`y ∈ {0,1,2,3}` 中只有 0/1 是有标注的。旧加载器会直接抛
`ValueError('DGraph-Fin 掩码与节点数不一致')`，使 P0-6 无法在真实文件上运行。

本模块不修改历史 p0_data 行为，只提供兼容加载与显式核验。
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


DGRAPH_LABEL_MAP = {'negative': 0, 'positive': 1, 'unlabeled': (2, 3)}


def _to_bool_mask(raw_mask, num_nodes, name):
    """把官方索引数组或布尔向量统一成布尔掩码；形状歧义一律拒绝猜测。"""
    m = np.asarray(raw_mask)
    if m.ndim != 1:
        raise ValueError(f'{name} 必须是一维，实际 {m.shape}')
    if m.dtype == bool:
        if len(m) != num_nodes:
            raise ValueError(f'{name} 布尔掩码长度 {len(m)} != 节点数 {num_nodes}')
        return m
    if not np.issubdtype(m.dtype, np.integer):
        raise ValueError(f'{name} 掩码 dtype 必须是布尔或整型，实际 {m.dtype}')
    unique = np.unique(m)
    if len(m) == num_nodes and set(unique.tolist()) <= {0, 1}:
        return m.astype(bool)
    if m.min() < 0 or m.max() >= num_nodes:
        raise ValueError(f'{name} 索引越界: [{m.min()}, {m.max()}], 节点数 {num_nodes}')
    if len(np.unique(m)) != len(m):
        raise ValueError(f'{name} 索引数组含重复节点')
    out = np.zeros(num_nodes, dtype=bool)
    out[m] = True
    return out


def load_dgraphfin_real(data_dir='./data/dgraphfin', reveal_test_labels=False):
    """读取官方 DGraph-Fin，兼容索引掩码，并把 {2,3} 之外的标签置为未知。

    默认与 p0_data 一致：测试标签在返回前立即屏蔽，训练阶段不可见。
    reveal_test_labels=True 只允许被锁后最终评估调用，用于读取官方测试标签。
    """
    path = Path(data_dir) / 'dgraphfin.npz'
    if not path.is_file():
        raise FileNotFoundError(f'DGraph-Fin 文件不存在: {path}')
    with np.load(path, allow_pickle=False) as z:
        required = {'x', 'y', 'edge_index', 'train_mask', 'valid_mask', 'test_mask'}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f'DGraph-Fin 缺字段: {sorted(missing)}')
        x = np.asarray(z['x'], dtype=np.float64)
        raw_y = np.asarray(z['y']).reshape(-1)
        if raw_y.dtype.kind not in 'iu':
            raise ValueError(f'y 必须是整型，实际 {raw_y.dtype}')
        y_raw = raw_y.astype(np.int64)
        n = len(x)
        if len(y_raw) != n:
            raise ValueError(f'逐节点 y 长度 {len(y_raw)} != 节点数 {n}')
        masks = {}
        kinds = {}
        for name in ('train_mask', 'valid_mask', 'test_mask'):
            masks[name] = _to_bool_mask(z[name], n, name)
            kinds[name] = ('index_array' if len(np.asarray(z[name])) != n
                           else 'boolean_vector')
        # 代码库统一采用 PyG 的 (2, E) 约定；官方文件为 (E, 2)，必须显式转置
        # 并记录布局，避免把 (E,2) 误当成 (2,E) 后静默产生越界或空图。
        edge_index = np.asarray(z['edge_index'], dtype=np.int64)
        if edge_index.ndim != 2:
            raise ValueError(f'edge_index 必须是二维，实际 {edge_index.shape}')
        if edge_index.shape[0] == 2:
            layout = 'already_2_x_E'
        elif edge_index.shape[1] == 2:
            edge_index = np.ascontiguousarray(edge_index.T)
            layout = 'transposed_from_E_x_2'
        else:
            raise ValueError(f'edge_index 形状既非 (2,E) 也非 (E,2): {edge_index.shape}')
        if edge_index.min() < 0 or edge_index.max() >= n:
            raise ValueError(f'edge_index 越界: [{edge_index.min()}, '
                             f'{edge_index.max()}], 节点数 {n}')
        # 标签映射：1=正类（高风险），0=负类，其余（官方 2/3）为未标注
        y = np.full(n, -1, dtype=np.int64)
        y[y_raw == 1] = 1
        y[y_raw == 0] = 0
        visible = masks['train_mask'] | masks['valid_mask']
        y_masked = y.copy()
        y_masked[~visible] = -1
        result = {'x': x.astype(np.float32),
                  'y': y if reveal_test_labels else y_masked,
                  'reveal_test_labels': bool(reveal_test_labels),
                  'edge_index': edge_index,
                  'train_mask': masks['train_mask'],
                  'valid_mask': masks['valid_mask'],
                  'test_mask': masks['test_mask'],
                  'node_ids': np.arange(n, dtype=np.int64),
                  'edge_timestamp': (np.asarray(z['edge_timestamp'])
                                     if 'edge_timestamp' in z.files else None),
                  'edge_type': (np.asarray(z['edge_type'])
                                if 'edge_type' in z.files else None),
                  'n_features': int(x.shape[1]),
                  'edge_index_layout': layout,
                  'n_edges': int(edge_index.shape[1]),
                  'time_semantics': 'edge_timestamp_only',
                  'source_files': [str(path)],
                  'mask_kind': kinds,
                  'label_mapping': {'1': 'positive_high_risk', '0': 'negative',
                                    'other': 'unlabeled_ignored'},
                  'raw_label_counts': {int(v): int(c) for v, c in
                                       zip(*np.unique(y_raw, return_counts=True))}}
    audit_view = dict(result)
    audit_view['y'] = y_masked
    result['split_audit'] = audit_official_split(audit_view)
    return result


def audit_official_split(raw):
    """核验官方划分：三段必须互斥，标注比例须逐段报告，不得静默。"""
    train, valid, test = (raw['train_mask'], raw['valid_mask'],
                          raw['test_mask'])
    y = raw['y']
    rows = {}
    counts = {}
    for name, mask in (('train', train), ('valid', valid), ('test', test)):
        idx = np.flatnonzero(mask)
        labeled = idx[y[idx] >= 0]
        pos = int(np.sum(y[labeled] == 1))
        rows[name] = {'n_nodes': int(mask.sum()),
                      'n_labeled': int(len(labeled)),
                      'n_positive': pos,
                      'n_negative': int(np.sum(y[labeled] == 0)),
                      'positive_rate': (float(pos / len(labeled))
                                        if len(labeled) else None)}
        counts[name] = int(mask.sum())
    overlaps = {
        'train_valid': int(np.sum(train & valid)),
        'train_test': int(np.sum(train & test)),
        'valid_test': int(np.sum(valid & test))}
    return {'per_split': rows, 'overlaps': overlaps,
            'disjoint': all(v == 0 for v in overlaps.values()),
            'train_and_valid_labels_only': bool(
                np.all(y[test] == -1)),
            'note': ('官方边时间戳不等于节点到达时间，本划分不能称为严格节点 rolling')}


def split_fingerprint(raw):
    """划分指纹：训练/验证/测试掩码与已标注标签的字节级哈希。"""
    import hashlib
    h = hashlib.sha256()
    for name in ('train_mask', 'valid_mask', 'test_mask'):
        h.update(np.ascontiguousarray(raw[name]).tobytes())
    labeled = np.flatnonzero(raw['y'] >= 0)
    h.update(np.ascontiguousarray(labeled).tobytes())
    h.update(np.ascontiguousarray(raw['y'][labeled]).tobytes())
    return h.hexdigest()


def dataset_file_hashes(data_dir, names):
    """原始数据文件 SHA-256；用于独立性声明绑定。"""
    import hashlib
    out = {}
    for name in names:
        path = Path(data_dir) / name
        if not path.is_file():
            out[name] = None
            continue
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                digest.update(block)
        out[name] = digest.hexdigest()
    return out
