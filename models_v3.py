"""TASGL-Lite v3：固定稀疏时间传播、可选图教师蒸馏与后验校准。

设计目标是在不读取测试标签的前提下，分别处理精度、真实推理开销、结构扰动、
时间外推和概率校准。该文件只定义模型与校准器；实验协议见 exp_hypotheses.py。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SparseTemporalMM(torch.autograd.Function):
    """稀疏时间传播的自定义反向，避免 autograd 生成 N×N 稠密梯度。

    前向：out = A(raw_decay) h。
    反向：grad_h = Aᵀ grad_out；raw_decay 的梯度按边分块累加，
    显存开销为 O(N·h + E·chunk_h)，而不是 O(N²)。
    """

    @staticmethod
    def forward(ctx, h, edge_index, time_gaps, raw_decay, num_nodes):
        row, col = edge_index
        if row.numel() == 0:
            idx = torch.empty((2, 0), dtype=torch.long, device=h.device)
            weights = torch.empty(0, dtype=h.dtype, device=h.device)
            norm = weights
            gaps = weights
        else:
            deg = torch.zeros(num_nodes, device=h.device, dtype=h.dtype)
            deg.index_add_(0, col, torch.ones_like(col, dtype=h.dtype))
            norm = 1.0 / deg[col].clamp(min=1.0)
            gaps = time_gaps.to(dtype=h.dtype).clamp(min=0, max=50)
            decay = F.softplus(raw_decay)
            weights = norm / (1.0 + decay * gaps)
            idx = torch.stack([col, row])
        adj = torch.sparse_coo_tensor(idx, weights, (num_nodes, num_nodes),
                                      device=h.device, dtype=h.dtype).coalesce()
        ctx.save_for_backward(h, edge_index, gaps, raw_decay, norm)
        ctx.adj = adj
        return torch.sparse.mm(adj, h)

    @staticmethod
    def backward(ctx, grad_out):
        h, edge_index, gaps, raw_decay, norm = ctx.saved_tensors
        row, col = edge_index
        grad_h = None
        if ctx.needs_input_grad[0]:
            grad_h = torch.sparse.mm(ctx.adj.transpose(0, 1).coalesce(),
                                     grad_out.contiguous())
        grad_raw = None
        if ctx.needs_input_grad[3]:
            grad_raw = torch.zeros_like(raw_decay)
            if row.numel() > 0:
                decay = F.softplus(raw_decay)
                sigmoid = torch.sigmoid(raw_decay)
                denom = 1.0 + decay * gaps
                # 分块避免同时生成多份 E×hidden 中间张量。
                chunk = 65536
                for start in range(0, row.numel(), chunk):
                    end = min(start + chunk, row.numel())
                    grad_w = (grad_out[col[start:end]]
                              * h[row[start:end]]).sum(dim=-1)
                    grad_raw = grad_raw + (
                        -grad_w * norm[start:end] * gaps[start:end]
                        * sigmoid / denom[start:end].pow(2)).sum()
        return grad_h, None, None, grad_raw, None


class SparseTemporalPropagation(nn.Module):
    """行归一化稀疏传播：out_i = sum_j w_ij h_j / deg_i。

    与逐边注意力不同，本层不为每条边运行 MLP。时间衰减只使用相对时间差，
    不依赖绝对时间步。评测静态图时缓存归一化邻接矩阵；稀疏矩阵乘法仍计入
    每次前向延迟，因此不能把图计算宣称为零成本。
    """

    def __init__(self):
        super().__init__()
        self.raw_decay = nn.Parameter(torch.tensor(-2.0))
        self._cached_adj = None
        self._cached_key = None
        self._cached_inputs = None

    def _adjacency(self, edge_index, time_gaps, num_nodes):
        row, col = edge_index
        if row.numel() == 0:
            idx = torch.empty((2, 0), dtype=torch.long, device=row.device)
            return torch.sparse_coo_tensor(idx, torch.empty(0, device=row.device),
                                           (num_nodes, num_nodes)).coalesce()
        deg = torch.zeros(num_nodes, device=row.device, dtype=torch.float32)
        deg.index_add_(0, col, torch.ones_like(col, dtype=torch.float32))
        decay = F.softplus(self.raw_decay)
        gaps = time_gaps.to(dtype=torch.float32).clamp(min=0, max=50)
        weights = (1.0 / deg[col].clamp(min=1.0)) / (1.0 + decay * gaps)
        indices = torch.stack([col, row])
        return torch.sparse_coo_tensor(indices, weights,
                                       (num_nodes, num_nodes)).coalesce()

    def forward(self, h, edge_index, time_gaps, num_nodes):
        # 需要梯度时走自定义稀疏反向。原生 sparse.mm 对可学习稀疏值
        # 在某些 CUDA 版本会申请 N×N 稠密梯度（Elliptic 约 154.68 GiB）。
        if torch.is_grad_enabled() and \
                (h.requires_grad or self.raw_decay.requires_grad):
            return _SparseTemporalMM.apply(h, edge_index, time_gaps,
                                           self.raw_decay, num_nodes)
        # 静态评测缓存一次归一化图；稀疏矩阵乘法仍计入每次延迟。
        key = None
        if not self.training:
            key = (edge_index.data_ptr(), time_gaps.data_ptr(),
                   int(edge_index._version), int(time_gaps._version),
                   int(self.raw_decay._version), num_nodes, h.device, h.dtype)
        if key is not None and key == self._cached_key:
            adj = self._cached_adj
        else:
            adj = self._adjacency(edge_index, time_gaps, num_nodes)
            if key is not None:
                self._cached_key, self._cached_adj = key, adj
                # 持有输入引用，避免旧张量释放后新图复用同一 data_ptr。
                self._cached_inputs = (edge_index, time_gaps)
        return torch.sparse.mm(adj, h)


class TASGLite(nn.Module):
    """轻量学生：165→h 投影 + 0/1/2 次固定传播 + 小型融合分类器。

    hops=0 时是真正的 graph-less 学生，可配合图教师软标签训练；此时删边不会
    改变其预测。hops>0 时保留真实稀疏图计算，延迟测量包含稀疏传播。
    """

    def __init__(self, in_dim, hidden=48, hops=2, dropout=0.2):
        super().__init__()
        if hops not in (0, 1, 2):
            raise ValueError('hops 仅支持 0/1/2')
        self.in_dim, self.hidden_dim, self.hops = in_dim, hidden, hops
        self.dropout = float(dropout)
        self.input_proj = nn.Linear(in_dim, hidden)
        self.prop = SparseTemporalPropagation()
        self.fusion = nn.Linear((hops + 1) * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, 2)
        self.raw_skip = nn.Linear(in_dim, 2)

    def encode(self, x, edge_index, dt, num_nodes, **kwargs):
        h0 = F.silu(self.input_proj(x))
        views = [h0]
        cur = h0
        for _ in range(self.hops):
            cur = self.prop(cur, edge_index, dt, num_nodes)
            views.append(cur)
        h = self.fusion(torch.cat(views, dim=-1)) + h0
        h = self.norm(h)
        h = F.silu(h)
        return F.dropout(h, p=self.dropout, training=self.training)

    def forward(self, x, edge_index, dt, num_nodes, **kwargs):
        h = self.encode(x, edge_index, dt, num_nodes)
        logits = self.classifier(h) + self.raw_skip(x)
        return logits, {'h': h, 'gates': None}

    @torch.no_grad()
    def mc_uncertainty(self, x, edge_index, dt, num_nodes, T=10):
        """仅用于选择性预测对照；主延迟不使用 MC 前向。"""
        was_training = self.training
        self.train()
        probs = []
        for _ in range(T):
            logits, _ = self.forward(x, edge_index, dt, num_nodes)
            probs.append(torch.softmax(logits, dim=-1))
        if not was_training:
            self.eval()
        p = torch.stack(probs).mean(0)
        entropy = -(p * (p + 1e-12).log()).sum(-1)
        return p, entropy


class CalibratedInference(nn.Module):
    """把后验校准计入部署前向，供真实效率测量使用。"""

    def __init__(self, model, calibrator):
        super().__init__()
        self.model = model
        self.calibrator = calibrator

    def forward(self, *args, **kwargs):
        logits, aux = self.model(*args, **kwargs)
        return self.calibrator.transform(logits), aux


class EDLHead(nn.Module):
    """同骨干 EDL 不确定性代理基线，不等同于完整 ST-CGNN。"""

    def __init__(self, hidden):
        super().__init__()
        self.evidence = nn.Linear(hidden, 2)

    def forward(self, h):
        evidence = F.softplus(self.evidence(h))
        alpha = evidence + 1.0
        probs = alpha / alpha.sum(dim=-1, keepdim=True)
        vacuity = 2.0 / alpha.sum(dim=-1)
        return evidence, alpha, probs, vacuity


class LogitCalibrator:
    """验证集后验校准：单温度或温度＋类别偏置。

    temp_bias 仅比标量温度多一个可识别参数（illicit logit bias），用于纠正
    类别加权训练造成的先验偏移。拟合目标是未加权验证集 NLL，而非测试 ECE。
    """

    def __init__(self, method='temp_bias', temperature=1.0, bias=0.0):
        if method not in ('temp', 'temp_bias'):
            raise ValueError('method 必须为 temp 或 temp_bias')
        self.method = method
        self.temperature = float(temperature)
        self.bias = float(bias if method == 'temp_bias' else 0.0)

    def fit(self, logits, labels, max_iter=60):
        x = torch.as_tensor(logits, dtype=torch.float32)
        y = torch.as_tensor(labels, dtype=torch.long)
        if x.ndim != 2 or x.size(1) != 2 or x.size(0) != y.numel():
            raise ValueError('校准输入必须为 [N,2] logits 与 [N] 标签')
        if y.numel() == 0 or torch.unique(y).numel() < 2:
            raise ValueError('校准集必须同时包含两个类别')
        log_t = torch.nn.Parameter(torch.zeros(()))
        bias = torch.nn.Parameter(torch.zeros(()))
        params = [log_t] + ([bias] if self.method == 'temp_bias' else [])
        opt = torch.optim.LBFGS(params, lr=0.2, max_iter=max_iter,
                                tolerance_grad=1e-7, tolerance_change=1e-9)

        def closure():
            opt.zero_grad()
            z = x / log_t.clamp(-3, 3).exp()
            if self.method == 'temp_bias':
                z = z + torch.stack([torch.zeros_like(bias), bias])
            loss = F.cross_entropy(z, y)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            self.temperature = float(log_t.clamp(-3, 3).exp().item())
            self.bias = float(bias.item()) if self.method == 'temp_bias' else 0.0
        return self

    def transform(self, logits):
        z = torch.as_tensor(logits, dtype=torch.float32) / self.temperature
        if self.method == 'temp_bias':
            shift = torch.tensor([0.0, self.bias], dtype=z.dtype,
                                 device=z.device)
            z = z + shift
        return z

    def probabilities(self, logits):
        return torch.softmax(self.transform(logits), dim=-1)

    def to_dict(self):
        return {'method': self.method, 'temperature': self.temperature,
                'bias': self.bias}

    @classmethod
    def from_dict(cls, d):
        return cls(d['method'], d['temperature'], d.get('bias', 0.0))


def symmetric_kl(student_logits, teacher_logits, temperature=2.0):
    """双向 KL 一致性；temperature 同时软化两个预测分布。"""
    t = float(temperature)
    a = F.log_softmax(student_logits / t, dim=-1)
    b = F.log_softmax(teacher_logits / t, dim=-1)
    pa, pb = a.exp(), b.exp()
    return 0.5 * ((pa * (a - b)).sum(-1).mean()
                  + (pb * (b - a)).sum(-1).mean()) * t * t


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    t = float(temperature)
    return F.kl_div(F.log_softmax(student_logits / t, dim=-1),
                    torch.softmax(teacher_logits.detach() / t, dim=-1),
                    reduction='batchmean') * t * t


def temporal_alignment_loss(h, time_steps, train_end, max_nodes=1024):
    """仅用训练截止点内节点的均值/方差对齐，约束时间不变表示。"""
    visible = time_steps <= train_end
    ts = time_steps[visible]
    z = h[visible]
    if z.size(0) > max_nodes:
        idx = torch.randperm(z.size(0), device=z.device)[:max_nodes]
        z, ts = z[idx], ts[idx]
    if z.size(0) < 8:
        return h.sum() * 0.0
    median = ts.float().median()
    early, late = z[ts <= median], z[ts > median]
    if early.size(0) < 2 or late.size(0) < 2:
        return h.sum() * 0.0
    mean_loss = (early.mean(0) - late.mean(0)).pow(2).mean()
    var_loss = (early.var(0, unbiased=False)
                - late.var(0, unbiased=False)).pow(2).mean()
    return mean_loss + 0.2 * var_loss


def dirichlet_kl_to_uniform(alpha):
    """KL(Dir(alpha) || Dir(1))，用于 EDL 的错误证据正则。"""
    k = alpha.size(-1)
    digamma_sum = torch.digamma(alpha.sum(-1, keepdim=True))
    return (torch.lgamma(alpha.sum(-1)) - torch.lgamma(torch.tensor(
                float(k), device=alpha.device))
            - torch.lgamma(alpha).sum(-1)
            + ((alpha - 1.0) * (torch.digamma(alpha) - digamma_sum)).sum(-1))
