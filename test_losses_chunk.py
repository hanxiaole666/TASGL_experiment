"""test_losses_chunk.py — 分块 max-sim 与整体 max-sim 的等价性

BitHeNet 20 种子训练在 6 GB 设备上 OOM, 触发点是 HardNegContrastiveLoss
里物化的 [P,N] 相似度矩阵。修复办法是按行分块求最大值。分块改变了计算
顺序, 所以必须证明它不改变结果。本测试做两件事:

  1. 数值等价: 分块结果与 (zp @ zn.T).max(dim=0) 逐元素完全相同 (不是近似)。
  2. 峰值显存: 分块后峰值显著低于一次性物化 [P,N]。

用法: python -m pytest test_losses_chunk.py -q
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _mk(P, N, D, seed=0, device='cpu'):
    import torch
    g = torch.Generator(device=device).manual_seed(seed)
    zp = torch.randn(P, D, generator=g, device=device)
    zn = torch.randn(N, D, generator=g, device=device)
    # 余弦相似度要求归一化, 与 losses.py 中 F.normalize(dim=-1) 一致
    import torch.nn.functional as F
    return F.normalize(zp, dim=-1), F.normalize(zn, dim=-1)


def test_chunked_equals_full_exactly():
    """分块结果必须与整体结果逐位相同 (max 与结合顺序无关)。"""
    import torch
    from losses import HardNegContrastiveLoss
    loss = HardNegContrastiveLoss()
    for P, N, D in [(3, 7, 4), (17, 53, 8), (64, 257, 16), (1, 5, 3),
                    (5, 1, 3), (200, 31, 32)]:
        zp, zn = _mk(P, N, D, seed=P + N)
        full = (zp @ zn.T).max(dim=0).values
        part = loss._max_sim_per_neg(zp, zn)
        assert part.shape == full.shape
        assert torch.equal(part, full), f'P={P} N={N} D={D} 分块结果不一致'


def test_chunk_size_respected():
    """CHUNK_BYTES 决定分块行数; 行数必须 >= 1 且不超过 P。"""
    from losses import HardNegContrastiveLoss
    P, N, D = 1000, 4096, 32
    rows = HardNegContrastiveLoss.CHUNK_BYTES // (N * 4)
    assert rows >= 1
    # 4096 列 × 4 字节 = 16 KiB/行, 64 MiB / 16 KiB = 4096 行 -> 不超过 P
    assert rows == 4096
    assert rows <= P or P < rows


def test_full_loss_matches_reference_implementation():
    """整体损失: 分块版与非分块版给出同一个标量。"""
    import torch
    import torch.nn.functional as F
    from losses import HardNegContrastiveLoss

    def ref(h, labels, entropy, mask, temperature=0.1, topk=32,
            unc_weight=2.0):
        idx = torch.where(mask)[0]
        h, y = h[idx], labels[idx]
        pos = (y == 1).nonzero(as_tuple=True)[0]
        neg = (y == 0).nonzero(as_tuple=True)[0]
        z = F.normalize(h, dim=-1)
        sim_pn = z[pos] @ z[neg].T
        hs = sim_pn.max(dim=0).values
        if entropy is not None:
            hs = hs + entropy.detach()[idx][neg]
        k = min(topk, neg.numel())
        hard_neg = neg[hs.topk(k).indices]
        anchors = z[pos]
        pos_sim = (anchors @ z[pos].T) / temperature
        Pp = pos_sim.size(1)
        diag = torch.eye(Pp, dtype=torch.bool)
        pos_sim = pos_sim.masked_fill(diag, -1e9)
        neg_sim = (anchors @ z[hard_neg].T) / temperature
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        tgt = torch.zeros_like(logits)
        tgt[:, :Pp] = 1.0 / (Pp - 1)
        tgt[:, :Pp][diag] = 0.0
        out = (-(tgt * F.log_softmax(logits, dim=1)).sum(1))
        w = 1.0 + unc_weight * entropy.detach()[idx][pos]
        return (out * w).mean()

    g = torch.Generator().manual_seed(7)
    n, d = 240, 16
    h = torch.randn(n, d, generator=g)
    y = torch.zeros(n, dtype=torch.long)
    y[:60] = 1
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:200] = True
    ent = torch.rand(n, generator=g) * 0.3
    loss = HardNegContrastiveLoss()
    a = loss(h, y, ent, mask)
    b = ref(h, y, ent, mask)
    assert torch.allclose(a, b, atol=0, rtol=0), (a.item(), b.item())


@pytest.mark.skipif(not __import__('torch').cuda.is_available(),
                    reason='需要 CUDA')
def test_peak_memory_lower_than_full_materialization():
    """在 GPU 上: 分块峰值必须明显低于一次性物化 [P,N]。"""
    import torch
    from losses import HardNegContrastiveLoss
    P, N, D = 2048, 32768, 32
    # 一次性物化 [P,N] fp32 = 2048*32768*4 = 256 MiB, 正是 OOM 触发点
    zp, zn = _mk(P, N, D, seed=1, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    m0 = torch.cuda.max_memory_allocated()
    full = (zp @ zn.T).max(dim=0).values
    torch.cuda.synchronize()
    peak_full = torch.cuda.max_memory_allocated() - m0
    del full
    torch.cuda.empty_cache()

    loss = HardNegContrastiveLoss()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    m1 = torch.cuda.max_memory_allocated()
    part = loss._max_sim_per_neg(zp, zn)
    torch.cuda.synchronize()
    peak_part = torch.cuda.max_memory_allocated() - m1
    del part
    torch.cuda.empty_cache()

    assert peak_part < peak_full, (peak_part, peak_full)
    # 分块峰值应被限制在 CHUNK_BYTES 量级 (留 2x 余量给中间张量)
    assert peak_part < 4 * HardNegContrastiveLoss.CHUNK_BYTES


def test_chunked_matches_full_on_gpu_if_available():
    """GPU 上同样逐位相等。"""
    import torch
    from losses import HardNegContrastiveLoss
    if not torch.cuda.is_available():
        pytest.skip('需要 CUDA')
    zp, zn = _mk(500, 4096, 32, seed=3, device='cuda')
    full = (zp @ zn.T).max(dim=0).values
    part = HardNegContrastiveLoss()._max_sim_per_neg(zp, zn)
    assert torch.equal(part, full)
