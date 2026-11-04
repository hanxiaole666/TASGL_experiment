"""losses.py — TASGL v2 损失函数

1. UncertaintyDROLoss   : MC-Dropout 不确定性 + 分布鲁棒优化 (Wasserstein 球
                          近似: 高不确定节点分配更大扰动预算/损失权重)
2. HardNegContrastiveLoss: 不确定性加权监督对比损失 + 硬负样本挖掘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyDROLoss(nn.Module):
    """不确定性驱动的 DRO 分类损失。

    机制:
      (1) 类别加权 CE (应对 9.8% 类别不平衡);
      (2) DRO 权重 w_i ∝ exp(u_i / ρ): 预测熵 u_i 越大的节点获得越大权重,
          等价于在不确定性定义的 Wasserstein 球内做最坏情况优化;
      (3) 特征扰动正则: 训练中向高不确定节点注入按其熵缩放的高斯噪声
          (对抗扰动预算分配的一阶近似)。
    """

    def __init__(self, class_weights=None, rho=1.0, perturb_scale=0.1,
                 use_dro=True, use_perturb=True):
        super().__init__()
        self.register_buffer('class_weights',
                             class_weights if class_weights is not None
                             else torch.ones(2))
        self.rho = rho
        self.perturb_scale = perturb_scale
        self.use_dro = use_dro
        self.use_perturb = use_perturb

    def feature_perturb(self, x, entropy, mask):
        """对高不确定训练节点注入熵缩放噪声 (仅训练期调用)。"""
        if not self.use_perturb or not self.training:
            return x
        scale = entropy.detach()[mask].unsqueeze(-1) * self.perturb_scale
        noise = torch.randn_like(x[mask]) * scale
        out = x.clone()
        out[mask] = x[mask] + noise
        return out

    def forward(self, logits, targets, entropy=None, mask=None):
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        ce = F.cross_entropy(logits[mask], targets[mask],
                             weight=self.class_weights, reduction='none')
        if self.use_dro and entropy is not None:
            u = entropy.detach()[mask]
            w = torch.softmax(u / max(self.rho, 1e-6), dim=0) * u.numel()
            ce = ce * w
        return ce.mean()


class HardNegContrastiveLoss(nn.Module):
    """不确定性加权监督对比损失 + 硬负样本挖掘。

    闭环: 高不确定正常节点 -> 挖掘为硬负样本 -> 决策边界更精细 ->
          不确定性下降。
    """

    def __init__(self, temperature=0.1, topk_hard=32, unc_weight=2.0,
                 enabled=True):
        super().__init__()
        self.temperature = temperature
        self.topk_hard = topk_hard
        self.unc_weight = unc_weight
        self.enabled = enabled

    #: 分块计算时的单块字节上限。控制 (P,N) 相似度矩阵的峰值显存,
    #: 不改变任何数值结果。
    CHUNK_BYTES = 64 * 1024 * 1024

    def forward(self, h, labels, entropy=None, mask=None):
        if not self.enabled:
            return torch.tensor(0.0, device=h.device)
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        idx = torch.where(mask)[0]
        h, y = h[idx], labels[idx]
        pos = (y == 1).nonzero(as_tuple=True)[0]
        neg = (y == 0).nonzero(as_tuple=True)[0]
        if pos.numel() == 0 or neg.numel() == 0:
            return torch.tensor(0.0, device=h.device)

        z = F.normalize(h, dim=-1)
        # --- 硬负样本: 与欺诈节点相似度最高 (或不确定性最高) 的正常节点 ---
        hard_score = self._max_sim_per_neg(z[pos], z[neg])
        if entropy is not None:
            hard_score = hard_score + entropy.detach()[idx][neg]
        k = min(self.topk_hard, neg.numel())
        hard_neg = neg[hard_score.topk(k).indices]

        anchors = z[pos]                                    # 以欺诈为锚
        pos_sim = (anchors @ z[pos].T) / self.temperature   # [P, P]
        P = pos_sim.size(1)
        if P < 2:
            return torch.tensor(0.0, device=h.device)
        # 排除自正样本 (对角线), 否则损失被 trivial self-similarity 主导
        diag = torch.eye(P, dtype=torch.bool, device=h.device)
        pos_sim = pos_sim.masked_fill(diag, -1e9)
        neg_sim = (anchors @ z[hard_neg].T) / self.temperature
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        targets = torch.zeros_like(logits)
        targets[:, :P] = 1.0 / (P - 1)
        targets[:, :P][diag] = 0.0
        loss = -(targets * F.log_softmax(logits, dim=1)).sum(1)

        if entropy is not None:
            w = 1.0 + self.unc_weight * entropy.detach()[idx][pos]
            loss = loss * w
        return loss.mean()

    def _max_sim_per_neg(self, zp, zn):
        """每个负样本与全部正样本的最大余弦相似度 (分块, 不物化 [P,N])。

        数学上等于 ``(zp @ zn.T).max(dim=0).values``。分块只改变计算顺序,
        不改变结果: 浮点 max 与结合顺序无关, 每个点积的计算本身不变。
        在 6 GB 设备上 ``[P,N]`` 曾达 256 MiB, 是 BitHeNet 训练 OOM 的直接
        触发点, 因此这里按行分块把峰值限制到 CHUNK_BYTES。
        """
        P, N = zp.size(0), zn.size(0)
        rows = max(1, int(self.CHUNK_BYTES // max(1, N * zp.element_size())))
        out = None
        for i in range(0, P, rows):
            blk = zp[i:i + rows] @ zn.T                     # [c, N]
            m = blk.max(dim=0).values
            out = m if out is None else torch.maximum(out, m)
            del blk, m
        return out


class TASGLLoss(nn.Module):
    """组合损失: L = L_DRO + λ_con · L_contrastive"""

    def __init__(self, class_weights=None, lambda_con=0.1, **kw):
        super().__init__()
        # lambda_con 默认 0.1: 对比损失数值量级 (约 5~10) 远大于 CE,
        # 权重过大时会淹没分类梯度, 导致精度低且种子间不稳定
        self.dro = UncertaintyDROLoss(class_weights, **{
            k: v for k, v in kw.items()
            if k in ('rho', 'perturb_scale', 'use_dro', 'use_perturb')})
        self.con = HardNegContrastiveLoss(**{
            k: v for k, v in kw.items()
            if k in ('temperature', 'topk_hard', 'unc_weight', 'enabled')})
        self.lambda_con = lambda_con

    def forward(self, logits, h, targets, entropy=None, mask=None):
        l_cls = self.dro(logits, targets, entropy, mask)
        l_con = self.con(h, targets, entropy, mask)
        return l_cls + self.lambda_con * l_con, {
            'cls': l_cls.item(), 'con': l_con.item()}
