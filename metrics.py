"""metrics.py — 精度指标与效率测量 (E1/E2/E7 共用)

效率测量规范 (设计方案第五节):
  延迟 batch=1, warmup 10, 平均 100 次, cuda.synchronize 对齐;
  显存仅测前向传播峰值; 同时报告参数量与模型大小 MB。
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (f1_score, precision_recall_curve, auc,
                             precision_score, recall_score)

# numpy 版本兼容: trapezoid (numpy>=2.0) / trapz (旧版)
_trapz = getattr(np, 'trapezoid', None) or np.trapz


# ------------------------------------------------------------------
# 精度指标
# ------------------------------------------------------------------
def accuracy_metrics(probs_pos, labels, threshold=0.5):
    """probs_pos: 欺诈类概率 [M]; labels: 0/1 [M]; threshold: 决策阈值。"""
    preds = (probs_pos >= threshold).astype(int)
    out = {
        'f1': f1_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
    }
    p, r, _ = precision_recall_curve(labels, probs_pos)
    out['pr_auc'] = auc(r, p)
    out['recall_at_fpr_1pct'] = recall_at_fpr(labels, probs_pos, 0.01)
    return out


def recall_at_fpr(labels, scores, fpr_target=0.01):
    """Recall@FPR=1% —— 金融风控的关键操作点。"""
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    P = labels.sum()
    if P == 0:
        return 0.0
    N = len(labels) - P
    best = 0.0
    tp, fp = 0, 0
    for lab in labels_sorted:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / max(N, 1)
        if fpr <= fpr_target:
            best = tp / P
        else:
            break
    return best


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE (E7): probs [M, 2]。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    ece, bins = 0.0, np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def selective_prediction_risk(probs_pos, labels, uncertainty, steps=20,
                              threshold=0.5):
    """选择性预测曲线 (E7): 按不确定性升序纳入样本,
    返回 (coverages, risks)。曲线下面积越小越好。
    threshold 必须与主 F1 一样来自验证集, 不能在测试集调优。"""
    order = np.argsort(uncertainty)
    coverages, risks = [], []
    for q in np.linspace(1.0 / steps, 1.0, steps):
        n = max(2, int(len(order) * q))
        sel = order[:n]
        pred = (probs_pos[sel] >= threshold).astype(int)
        risk = 1.0 - f1_score(labels[sel], pred, zero_division=0)
        coverages.append(q)
        risks.append(risk)
    return np.array(coverages), np.array(risks)


# ------------------------------------------------------------------
# 效率测量
# ------------------------------------------------------------------
@torch.no_grad()
def measure_efficiency(model, data, device, n_warmup=10, n_repeat=100):
    """返回 dict: params / model_size_mb / latency_ms / peak_mem_mb /
    throughput (nodes/s) / active_edge_ratio (TASGL 硬掩码活跃边比例)。"""
    model.eval()
    fwd = lambda: model(data['x'], data['edge_index'], data['time_gaps'],
                        data['num_nodes'], hard_mask=True)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size()
                  for p in model.parameters()) / 1024 ** 2

    for _ in range(n_warmup):
        fwd()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_repeat):
        out = fwd()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / n_repeat * 1000
    peak_mem = (torch.cuda.max_memory_allocated() / 1024 ** 2
                if device.type == 'cuda' else 0.0)

    active_ratio = None
    aux = out[1] if isinstance(out, tuple) else None
    if aux and aux.get('gates') and aux['gates'][0] is not None and \
            aux['gates'][0].numel() > 0:
        ws = torch.cat([g for g in aux['gates'] if g.numel() > 0])
        active_ratio = float((ws > 0).float().mean().item())

    return {
        'params': int(params),
        'model_size_mb': round(size_mb, 3),
        'latency_ms': round(latency_ms, 3),
        'peak_mem_mb': round(peak_mem, 1),
        'throughput': round(data['num_nodes'] / (latency_ms / 1000), 1),
        'active_edge_ratio': active_ratio,
    }


@torch.no_grad()
def tune_threshold(model, data, val_mask, grid=None):
    """在验证集上网格搜索使 F1 最大的决策阈值 (不使用测试集, 无泄漏)。

    类别加权 CE 会使概率整体偏移, 固定 0.5 阈值会严重拉低 F1;
    实测 Elliptic 上该协议可将 MLP F1 从 0.30 提升至 0.64。
    """
    if grid is None:
        grid = np.linspace(0.02, 0.98, 49)
    model.eval()
    logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                      data['num_nodes'], hard_mask=True)
    pv = F.softmax(logits[val_mask], dim=-1)[:, 1].cpu().numpy()
    yv = data['y'][val_mask].cpu().numpy()
    f1s = [f1_score(yv, pv >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


@torch.no_grad()
def eval_with_threshold(model, data, val_mask, test_mask,
                        with_uncertainty=False, mc_T=10):
    """统一评估协议: 验证集调阈值 -> 测试集评估。返回指标(含 threshold)。"""
    thr = tune_threshold(model, data, val_mask)
    acc = evaluate(model, data, test_mask, threshold=thr,
                   with_uncertainty=with_uncertainty, mc_T=mc_T)
    acc['threshold'] = thr
    return acc


@torch.no_grad()
def evaluate(model, data, mask, with_uncertainty=False, mc_T=10,
             threshold=0.5):
    """统一评估: 返回 accuracy 指标 (+ 不确定性指标)。"""
    model.eval()
    logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                      data['num_nodes'], hard_mask=True)
    probs = F.softmax(logits[mask], dim=-1)[:, 1].cpu().numpy()
    labels = data['y'][mask].cpu().numpy()
    res = accuracy_metrics(probs, labels, threshold=threshold)
    if with_uncertainty and hasattr(model, 'mc_uncertainty'):
        p_all, ent = model.mc_uncertainty(
            data['x'], data['edge_index'], data['time_gaps'],
            data['num_nodes'], T=mc_T)
        p_sel = p_all[mask].cpu().numpy()
        ent_sel = ent[mask].cpu().numpy()
        res['ece'] = expected_calibration_error(p_sel, labels)
        res['mean_uncertainty'] = float(ent_sel.mean())
        cov, risk = selective_prediction_risk(probs, labels, ent_sel,
                                              threshold=threshold)
        res['selective_auc'] = float(_trapz(risk, cov))
    return res


# ------------------------------------------------------------------
# 统计检验
# ------------------------------------------------------------------
def wilcoxon_test(a, b):
    """Wilcoxon 符号秩检验 (TASGL vs 基线, 配对各时间步/种子 F1)。"""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) < 5 or (a == b).all():
        return None
    stat, p = wilcoxon(a, b)
    return {'statistic': float(stat), 'p_value': float(p),
            'significant': bool(p < 0.05)}
