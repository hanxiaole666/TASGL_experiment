"""P0 新增模型：强时序对照与可扩展 mini-batch GraphSAGE。"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TemporalGraphSAGE(nn.Module):
    """GraphSAGE 聚合 + 显式节点时间 Fourier 编码的强时序对照。

    该模块只用于 P0-4 独立 HPO，不与历史默认超参数基线混写。
    """
    def __init__(self, in_dim, hidden, dropout=0.3, out_dim=1,
                 time_bands=8, max_time=49):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, hidden)
        self.neigh_lin = nn.Linear(in_dim, hidden, bias=False)
        self.time_lin = nn.Linear(2 * time_bands, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, out_dim)
        self.time_bands = time_bands
        self.max_time = max_time

    def forward(self, x, edge_index, time_steps=None):
        if time_steps is None:
            raise ValueError('TemporalGraphSAGE 需要节点 time_steps')
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        deg.index_add_(0, row, torch.ones_like(row, dtype=x.dtype))
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        agg = agg / deg.clamp_min(1).unsqueeze(-1)
        t = time_steps.to(x.dtype).reshape(-1, 1) / float(self.max_time)
        bands = torch.arange(1, self.time_bands + 1, device=x.device,
                             dtype=x.dtype).reshape(1, -1)
        angle = torch.pi * t * bands
        temporal = torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)
        h = self.self_lin(x) + self.neigh_lin(agg) + self.time_lin(temporal)
        h = self.norm(torch.nn.functional.silu(h))
        return self.out(self.dropout(h))


class TemporalGraphSAGEAdapter(nn.Module):
    """适配 common.train_model 的四参数图接口。"""
    def __init__(self, in_dim, hidden, dropout=0.3, out_dim=2):
        super().__init__()
        self.core = TemporalGraphSAGE(in_dim, hidden, dropout, out_dim)
        self.node_time = None

    def forward(self, x, edge_index, time_gaps=None, num_nodes=None,
                **kwargs):
        if self.node_time is None:
            raise ValueError('训练前必须设置 adapter.node_time')
        return self.core(x, edge_index, self.node_time), {}


class NeighborGraphSAGE(nn.Module):
    """one-hop 自特征 + 均值邻居，适合 DGraph-Fin mini-batch。"""
    def __init__(self, in_dim, hidden=48, dropout=0.3):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, hidden)
        self.neigh_lin = nn.Linear(in_dim, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x_center, x_neighbors, neighbor_mask):
        mask = neighbor_mask.to(x_center.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        agg = (x_neighbors * mask).sum(dim=1) / denom
        agg = torch.where((denom > 0), agg, torch.zeros_like(agg))
        h = self.self_lin(x_center) + self.neigh_lin(agg)
        h = self.norm(torch.nn.functional.silu(h))
        return self.out(self.dropout(h)).reshape(-1)


class SIGNBaseline(nn.Module):
    """SIGN：固定的 0..K 跳传播，各跳独立线性变换后拼接分类。"""
    def __init__(self, in_dim, hidden=64, num_hops=2, dropout=0.3):
        super().__init__()
        self.num_hops = int(num_hops)
        self.transforms = nn.ModuleList([
            nn.Linear(in_dim, hidden) for _ in range(self.num_hops + 1)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear((self.num_hops + 1) * hidden, 2)

    def forward(self, x, edge_index, dt=None, num_nodes=None, **kwargs):
        from models import masked_aggregate
        row, col = edge_index
        n = num_nodes or x.size(0)
        cur, features = x, []
        for k in range(self.num_hops + 1):
            if k > 0:
                cur = masked_aggregate(cur[row], col, n)
            features.append(torch.nn.functional.silu(self.transforms[k](cur)))
        h = torch.cat(features, dim=1)
        return self.classifier(self.dropout(h)), {'h': h, 'gates': None}


class FeatureMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).reshape(-1)


class FixedNeighborSampler:
    """确定性 CSR 邻居采样；-1 padding 不参与均值聚合。"""
    def __init__(self, edge_index, num_nodes, fanout=16, seed=0):
        edge = np.asarray(edge_index, dtype=np.int64)
        if edge.shape[0] != 2 or np.any(edge < 0) or np.any(edge >= num_nodes):
            raise ValueError('edge_index 越界或形状错误')
        self.num_nodes = int(num_nodes)
        self.fanout = int(fanout)
        self.rng = np.random.RandomState(seed)
        order = np.argsort(edge[0], kind='stable')
        self.neighbors = edge[1, order]
        self.ptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.add.at(self.ptr, edge[0] + 1, 1)
        self.ptr = np.cumsum(self.ptr)

    def sample(self, nodes):
        nodes = np.asarray(nodes, dtype=np.int64)
        b, k = len(nodes), self.fanout
        out = np.zeros((b, k), dtype=np.int64)
        mask = np.zeros((b, k), dtype=bool)
        for i, node in enumerate(nodes):
            start, end = self.ptr[node], self.ptr[node + 1]
            degree = end - start
            if degree <= 0:
                continue
            take = min(k, degree)
            if degree > take:
                chosen = self.rng.choice(degree, size=take, replace=False)
            else:
                chosen = np.arange(degree)
            out[i, :take] = self.neighbors[start + chosen]
            mask[i, :take] = True
        return out, mask

    def sample_seed(self, seed):
        self.rng = np.random.RandomState(seed)
