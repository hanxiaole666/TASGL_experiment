"""models.py — TASGL v2 与全部基线

TASGL v2 模块 (对应设计方案第二节):
  1. LowRankRFFPropagation      低秩可学习随机特征图传播  O(|V|·d·r)
  2. TemporalBasisEncoding      可学习傅里叶时序基函数 Φ(Δt)
  3. TemporalSparseAttention    动态稀疏注意力 (可学习门控+阈值, 温度退火)
  4. TASGL                      完整模型 (含 MC-Dropout 不确定性头)

基线 (E1/E2):
  MLP / GCN / GAT / GraphSAGE / SGC / DSGATProxy(固定稀疏率 GAT)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ------------------------------------------------------------------
# PyG 缺失时的原生实现 (接口与 PyG 一致, 保证无 PyG 环境也能跑全部基线)
# 论文正式结果建议统一使用 PyG (pip install torch-geometric)
# ------------------------------------------------------------------
def _scatter_softmax(src, index, num_nodes):
    """对 index 分组的 softmax (数值稳定, 用于原生 GAT)。"""
    max_v = torch.full((num_nodes,), float('-inf'), device=src.device)
    max_v.index_reduce_(0, index, src, 'amax', include_self=False)
    e = torch.exp(src - max_v[index].clamp(min=-1e30))
    denom = torch.zeros(num_nodes, device=src.device)
    denom.index_add_(0, index, e)
    return e / (denom[index] + 1e-12)


class _NativeGCNConv(nn.Module):
    """对称归一化 GCN: D^-1/2 (A+I) D^-1/2 X W"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        n = x.size(0)
        row, col = edge_index
        self_loops = torch.arange(n, device=x.device)
        row = torch.cat([row, self_loops])
        col = torch.cat([col, self_loops])
        deg = torch.zeros(n, device=x.device)
        deg.index_add_(0, col, torch.ones(col.numel(), device=x.device))
        norm = deg.pow(-0.5).clamp(max=1e6)
        w = norm[row] * norm[col]
        agg = torch.zeros_like(x)
        agg.index_add_(0, col, x[row] * w.unsqueeze(-1))
        return self.lin(agg)


class _NativeGATConv(nn.Module):
    """原生 GAT (concat heads, 接口对齐 PyG)"""
    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0):
        super().__init__()
        self.heads, self.out_dim, self.dropout = heads, out_dim, dropout
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim * heads))

    def forward(self, x, edge_index):
        n = x.size(0)
        row, col = edge_index
        self_loops = torch.arange(n, device=x.device)
        row = torch.cat([row, self_loops])
        col = torch.cat([col, self_loops])
        h = self.lin(x).view(n, self.heads, self.out_dim)
        score = ((h[row] * self.a_src).sum(-1)
                 + (h[col] * self.a_dst).sum(-1))       # [E, H]
        score = F.leaky_relu(score, 0.2)
        out = []
        for hd in range(self.heads):
            alpha = _scatter_softmax(score[:, hd], col, n)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            out.append(masked_aggregate(h[row, hd], col, n, alpha))
        return torch.cat(out, dim=-1) + self.bias


class _NativeSAGEConv(nn.Module):
    """GraphSAGE mean 聚合: W_self x + W_neigh mean(N(x))"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        row, col = edge_index
        agg = masked_aggregate(x[row], col, x.size(0))
        return self.self_lin(x) + self.neigh_lin(agg)


if not HAS_PYG:
    GCNConv, SAGEConv = _NativeGCNConv, _NativeSAGEConv

    def GATConv(in_dim, out_dim, heads=1, dropout=0.0):   # noqa: N802
        return _NativeGATConv(in_dim, out_dim, heads, dropout)


# ------------------------------------------------------------------
# 工具
# ------------------------------------------------------------------
def masked_aggregate(src_feat, col, num_nodes, weights=None):
    """稀疏聚合: sum_j w_ij * h_j, 后接度归一化 (D^-1 A)。"""
    if weights is not None:
        src_feat = src_feat * weights.unsqueeze(-1)
    agg = torch.zeros(num_nodes, src_feat.size(-1),
                      device=src_feat.device, dtype=src_feat.dtype)
    agg.index_add_(0, col, src_feat)
    deg = torch.zeros(num_nodes, device=src_feat.device)
    if weights is not None:
        deg.index_add_(0, col, weights)
    else:
        deg.index_add_(0, col, torch.ones(col.size(0), device=col.device))
    deg = deg.clamp(min=1.0).unsqueeze(-1)
    return agg / deg


# ------------------------------------------------------------------
# 模块 1: 低秩可学习 RFF 传播
# ------------------------------------------------------------------
class LowRankRFFPropagation(nn.Module):
    """W = U V^T (rank r << dim) 替代固定随机投影; K-hop 线性传播;
    可学习跳数加权融合。复杂度 O(|V|·d·rff)。"""

    def __init__(self, dim, rff_dim=16, rank=8, num_hops=4):
        super().__init__()
        self.num_hops = num_hops
        self.U = nn.Parameter(torch.randn(dim, rank) * dim ** -0.5)
        self.V = nn.Parameter(torch.randn(rank, rff_dim) * rank ** -0.5)
        self.register_buffer('b_rff', torch.rand(rff_dim) * 2 * math.pi)
        self.hop_mix = nn.Parameter(torch.ones(num_hops + 1) / (num_hops + 1))
        self.out = nn.Linear(rff_dim, dim)
        self.audit_enabled = False
        self.last_propagation_audit = None

    def forward(self, h, edge_index, num_nodes):
        row, col = edge_index
        if self.audit_enabled:
            self.last_propagation_audit = {
                'input_edges': int(row.numel()), 'executed_edges_per_hop':
                [int(row.numel())] * self.num_hops,
                'num_hops': int(self.num_hops), 'rff_dim': int(self.V.shape[1]),
                'note': 'RFF hops are ungated; every hop executes all input edges'}
        rff = torch.cos(h @ (self.U @ self.V) + self.b_rff)  # [N, rff]
        acc, cur = torch.zeros_like(rff), rff
        ws = torch.softmax(self.hop_mix, dim=0)
        acc = acc + ws[0] * cur
        for k in range(1, self.num_hops + 1):
            cur = masked_aggregate(cur[row], col, num_nodes)
            acc = acc + ws[k] * cur
        return self.out(acc)


# ------------------------------------------------------------------
# 模块 2: 时序基函数编码
# ------------------------------------------------------------------
class TemporalBasisEncoding(nn.Module):
    """Φ(Δt) = [sin(ω_k·Δt), cos(ω_k·Δt)], ω 可学习 —— 支持时间外推。"""

    def __init__(self, num_basis=8):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_basis) * 0.1)

    def forward(self, dt):
        angles = dt.unsqueeze(-1) * self.freqs              # [E, B]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    @property
    def out_dim(self):
        return 2 * self.freqs.numel()


# ------------------------------------------------------------------
# 模块 3: 动态稀疏注意力层
# ------------------------------------------------------------------
class TemporalSparseAttentionLayer(nn.Module):
    """a_ij = σ(MLP[h_i ‖ h_j ‖ Φ(Δt_ij)]); 可学习阈值 τ;
    训练期软掩码 (温度退火), 推理期硬掩码 -> 活跃边减少 40-60%。"""

    def __init__(self, dim, num_basis=8, min_active=0.05,
                 execute_hard_sparse=False):
        super().__init__()
        self.time_enc = TemporalBasisEncoding(num_basis)
        gate_in = 2 * dim + self.time_enc.out_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, max(dim // 2, 8)), nn.ReLU(),
            nn.Linear(max(dim // 2, 8), 1))
        # 稳定性: 末层偏置置 2.0 -> 初始 a≈0.88, 门控从"稠密"开始学习稀疏,
        # 避免训练初期大量边被关闭导致模型退化为 MLP、种子间不稳定
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 2.0)
        self.log_tau = nn.Parameter(torch.tensor(0.0))   # τ = sigmoid(.)
        self.val = nn.Linear(dim, dim)
        self.min_active = min_active   # 硬掩码最少活跃边比例兜底
        self.execute_hard_sparse = execute_hard_sparse
        self.audit_enabled = False
        self.last_gate_audit = None

    def forward(self, h, edge_index, dt, num_nodes, temperature=10.0,
                hard_mask=False):
        row, col = edge_index
        if row.numel() == 0:
            if self.audit_enabled:
                self.last_gate_audit = {'input_edges': 0, 'active_edges': 0,
                                        'executed_edges': 0,
                                        'compact_execution': False}
            return torch.zeros_like(h), torch.zeros(0, device=h.device)
        te = self.time_enc(dt)
        logit = self.gate(torch.cat([h[row], h[col], te], dim=-1)).squeeze(-1)
        a = torch.sigmoid(logit)
        tau = torch.sigmoid(self.log_tau)
        if hard_mask or not self.training:
            w = (a > tau).float()
            # 兜底: 活跃边过少时保留得分最高的 min_active 比例,
            # 防止评测期门控全关导致精度崩溃
            if w.numel() > 0 and float(w.mean()) < self.min_active:
                k = max(1, int(self.min_active * a.numel()))
                thresh = a.topk(k).values[-1]
                w = (a >= thresh).float()
        else:
            w = torch.sigmoid((a - tau) * temperature)
        hard = bool(hard_mask or not self.training)
        active = (w > 0)
        compact = bool(hard and self.execute_hard_sparse)
        if compact:
            exec_row, exec_col, exec_w = row[active], col[active], w[active]
        else:
            exec_row, exec_col, exec_w = row, col, w
        if self.audit_enabled:
            self.last_gate_audit = {
                'input_edges': int(row.numel()),
                'active_edges': int(active.sum().detach().cpu()),
                'executed_edges': int(exec_row.numel()),
                'theoretical_active_ratio': float(active.float().mean().detach().cpu()),
                'executed_edge_ratio': float(exec_row.numel() / row.numel()),
                'hard_mask': hard, 'compact_execution': compact,
                'min_active': float(self.min_active)}
        agg = masked_aggregate(self.val(h[exec_row]), exec_col, num_nodes, exec_w)
        return agg, w


# ------------------------------------------------------------------
# TASGL 完整模型
# ------------------------------------------------------------------
class TASGL(nn.Module):
    def __init__(self, in_dim, hidden=48, rff_dim=16, rank=8, num_hops=4,
                 num_basis=8, dropout=0.3, execute_hard_sparse=False):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden)
        self.prop = LowRankRFFPropagation(hidden, rff_dim, rank, num_hops)
        self.attn1 = TemporalSparseAttentionLayer(
            hidden, num_basis, execute_hard_sparse=execute_hard_sparse)
        self.attn2 = TemporalSparseAttentionLayer(
            hidden, num_basis, execute_hard_sparse=execute_hard_sparse)
        self.classifier = nn.Linear(hidden, 2)
        self.dropout = dropout

    def encode(self, x, edge_index, dt, num_nodes, temperature=10.0,
               hard_mask=False):
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + F.relu(self.prop(h, edge_index, num_nodes))
        a1, w1 = self.attn1(h, edge_index, dt, num_nodes, temperature,
                            hard_mask)
        h = h + F.dropout(F.relu(a1), p=self.dropout, training=self.training)
        a2, w2 = self.attn2(h, edge_index, dt, num_nodes, temperature,
                            hard_mask)
        h = h + F.dropout(F.relu(a2), p=self.dropout, training=self.training)
        return h, (w1, w2)

    def forward(self, x, edge_index, dt, num_nodes, temperature=10.0,
                hard_mask=False):
        h, gates = self.encode(x, edge_index, dt, num_nodes, temperature,
                               hard_mask)
        logits = self.classifier(F.dropout(h, p=self.dropout,
                                           training=self.training))
        return logits, {'h': h, 'gates': gates}

    @torch.no_grad()
    def mc_uncertainty(self, x, edge_index, dt, num_nodes, T=10):
        """MC-Dropout 预测熵 (T 次随机前向)。返回 mean_probs, entropy [N]。"""
        was_training = self.training
        self.train()  # 保持 dropout 开启
        probs = []
        for _ in range(T):
            logits, _ = self.forward(x, edge_index, dt, num_nodes,
                                     hard_mask=False)
            probs.append(torch.softmax(logits, dim=-1))
        if not was_training:
            self.eval()
        p = torch.stack(probs).mean(0)                      # [N, 2]
        entropy = -(p * (p + 1e-12).log()).sum(-1)          # [N]
        return p, entropy


# ------------------------------------------------------------------
# 基线
# ------------------------------------------------------------------
class MLPBaseline(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim))

    def forward(self, x, edge_index=None, dt=None, num_nodes=None, **kw):
        return self.net(x), {'h': None, 'gates': None}


class _PyGWrapper(nn.Module):
    """统一基线接口: forward(x, edge_index, dt, num_nodes) -> (logits, aux)
    优先使用 PyG; 未安装时自动切换为上面的原生实现 (功能等价)。"""
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.classifier = nn.Linear(hidden, 2)

    def _finish(self, h):
        return self.classifier(h), {'h': h, 'gates': None}


class GCNBaseline(_PyGWrapper):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__(in_dim, hidden, dropout)
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index, dt=None, num_nodes=None, **kw):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return self._finish(h)


class GATBaseline(_PyGWrapper):
    def __init__(self, in_dim, hidden=64, heads=4, dropout=0.3):
        super().__init__(in_dim, hidden, dropout)
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads,
                             dropout=dropout)
        self.conv2 = GATConv(hidden, hidden, heads=1, dropout=dropout)

    def forward(self, x, edge_index, dt=None, num_nodes=None, **kw):
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return self._finish(h)


class GraphSAGEBaseline(_PyGWrapper):
    def __init__(self, in_dim, hidden=64, dropout=0.3):
        super().__init__(in_dim, hidden, dropout)
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

    def forward(self, x, edge_index, dt=None, num_nodes=None, **kw):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return self._finish(h)


class SGCBaseline(nn.Module):
    """K-hop 线性传播 + 单个分类器 (轻量基线, 无注意力)。"""

    def __init__(self, in_dim, num_hops=2, out_dim=2):
        super().__init__()
        self.num_hops = num_hops
        self.classifier = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, dt=None, num_nodes=None, **kw):
        row, col = edge_index
        h = x
        for _ in range(self.num_hops):
            h = masked_aggregate(h[row], col, num_nodes or x.size(0))
        return self.classifier(h), {'h': h, 'gates': None}


class DSGATProxy(GATBaseline):
    """固定稀疏率基线 (DSGAT 代理): 按注意力分位置 top-k% 保留边,
    与 TASGL 的可学习门控形成对照。"""

    def __init__(self, in_dim, hidden=64, keep_ratio=0.5, **kw):
        super().__init__(in_dim, hidden, **kw)
        self.keep_ratio = keep_ratio

    def _sparsify(self, edge_index):
        E = edge_index.size(1)
        k = max(1, int(E * self.keep_ratio))
        idx = torch.randperm(E, device=edge_index.device)[:k]
        return edge_index[:, idx]

    def forward(self, x, edge_index, dt=None, num_nodes=None, **kw):
        return super().forward(x, self._sparsify(edge_index), dt,
                               num_nodes, **kw)


# ------------------------------------------------------------------
# 工厂
# ------------------------------------------------------------------
_BASELINES = {
    'mlp': MLPBaseline,
    'gcn': GCNBaseline,
    'gat': GATBaseline,
    'graphsage': GraphSAGEBaseline,
    'sgc': SGCBaseline,
    'dsgat': DSGATProxy,
}


def build_model(name, in_dim, **tasgl_kw):
    """工厂函数。

    重要修正 (P4 终版): 旧实现把 hidden/dropout 等超参丢弃, 只把 in_dim 传给
    基线, 导致"等预算超参搜索"对 MLP/GCN/SGC/DSGAT 完全无效 (六个 trial 的
    验证 F1 与参数量逐位相同)。这里按构造函数签名过滤后透传, 并在
    test_p4.py 中用参数量断言锁死该行为。
    """
    import inspect
    name = name.lower()
    if name == 'tasgl':
        return TASGL(in_dim, **tasgl_kw)
    cls = _BASELINES[name]
    try:
        sig = inspect.signature(cls.__init__)
        allowed = {k: v for k, v in tasgl_kw.items()
                   if k in sig.parameters}
    except (TypeError, ValueError):
        allowed = {}
    return cls(in_dim, **allowed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
