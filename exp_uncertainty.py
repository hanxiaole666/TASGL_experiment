"""exp_uncertainty.py — E7 不确定性质量: ECE + 选择性预测曲线

对照: TASGL (MC-Dropout + DRO) vs EDL 头基线 (evidential deep learning,
ST-CGNN 类方法的轻量代理) vs 普通 softmax 置信度。

输出:
  results/uncertainty_results.json
  figs/selective_prediction.pdf   覆盖率-风险曲线 (论文 Figure)
  figs/uncertainty_timeline.pdf   测试早段 vs 晚段的不确定性时间曲线
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_elliptic, make_graph_data, standard_split
from models import TASGL, build_model
from common import set_seed, train_model, save_json, load_if_done, \
    resume_paths, PROTOCOL_VERSION
from metrics import (expected_calibration_error,
                     selective_prediction_risk, accuracy_metrics)

# v2: 仅把无实证依据的 Known/Novel 改为 Early/Late, 并绘制真实时间步。
TIMELINE_SEMANTICS_VERSION = 2


class EDLHead(nn.Module):
    """Evidential 不确定性基线: Dirichlet 证据头 (ST-CGNN 轻量代理)。"""

    def __init__(self, backbone_out=48):
        super().__init__()
        self.evidence = nn.Linear(backbone_out, 2)

    def forward(self, h):
        e = F.softplus(self.evidence(h))
        alpha = e + 1.0
        S = alpha.sum(-1, keepdim=True)
        probs = alpha / S
        uncertainty = (2.0 / S).squeeze(-1)          # vacuity
        return probs, uncertainty


def evaluate_with_unc(backbone, head, data, mask, mode, val_mask=None):
    backbone.eval()

    def probs_unc(m):
        with torch.no_grad():
            if mode == 'tasgl':
                probs, ent = backbone.mc_uncertainty(
                    data['x'], data['edge_index'], data['time_gaps'],
                    data['num_nodes'], T=10)
                return probs[m].cpu().numpy(), ent[m].cpu().numpy()
            h, _ = backbone.encode(data['x'], data['edge_index'],
                                   data['time_gaps'], data['num_nodes'])
            probs, u = head(h)
            return probs[m].cpu().numpy(), u[m].cpu().numpy()

    p, u = probs_unc(mask)
    y = data['y'][mask].cpu().numpy()
    # 与其他实验一致: 阈值在该模式自己的验证集概率上调优
    thr = 0.5
    if val_mask is not None:
        pv, _ = probs_unc(val_mask)
        yv = data['y'][val_mask].cpu().numpy()
        from sklearn.metrics import f1_score
        grid = np.linspace(0.02, 0.98, 49)
        thr = float(grid[int(np.argmax(
            [f1_score(yv, pv[:, 1] >= t, zero_division=0) for t in grid]))])
    return {
        **accuracy_metrics(p[:, 1], y, threshold=thr),
        'threshold': thr,
        'ece': expected_calibration_error(p, y),
        'selective': selective_prediction_risk(p[:, 1], y, u,
                                               threshold=thr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/elliptic')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/uncertainty_results.json')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    cached = load_if_done(args.out, args.overwrite)
    cache_complete = (
        cached is not None and
        cached.get('timeline_semantics') == TIMELINE_SEMANTICS_VERSION and
        cached.get('seed') == args.seed and
        'timeline' in cached and 'curves' in cached)
    if cache_complete:
        print(f'[uncertainty] 当前协议及 seed={args.seed} 的结果已存在: '
              f'{args.out}, 跳过 (加 --overwrite 可重跑)')
        return
    if cached is not None:
        print('[uncertainty] 旧缓存缺少当前种子/完整时间图字段, 或图语义为旧版; '
              '自动重跑以避免复用其他种子的结果')

    rp, cp = resume_paths('uncertainty', f'backbone_seed{args.seed}')
    if args.overwrite and os.path.exists(cp):
        os.remove(cp)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, labels, time_steps, edge_index, _ = load_elliptic(args.data_dir)
    data = make_graph_data(X, labels, time_steps, edge_index, device)
    tr, te = standard_split(time_steps, labels, 34)
    va = (time_steps > 29) & (time_steps <= 34) & (labels >= 0)
    tr = tr & ~va
    masks = [torch.tensor(m).to(device) for m in (tr, va, te)]

    set_seed(args.seed)
    backbone = TASGL(data['in_dim']).to(device)
    backbone, _ = train_model(backbone, data, *masks[:2], device,
                              epochs=args.epochs, loss_fn=None,
                              ckpt_path=cp)

    # --- EDL 头: 冻结 backbone, 仅训练证据头 ---
    edl = EDLHead().to(device)
    opt = torch.optim.AdamW(edl.parameters(), lr=1e-3)
    for epoch in range(100):
        backbone.eval()
        with torch.no_grad():
            h, _ = backbone.encode(data['x'], data['edge_index'],
                                   data['time_gaps'], data['num_nodes'])
        probs, u = edl(h)
        loss = F.nll_loss((probs[masks[0]] + 1e-9).log(),
                          data['y'][masks[0]])
        opt.zero_grad(); loss.backward(); opt.step()

    results = {'seed': args.seed}
    curves = {}
    for mode, head in (('tasgl', None), ('edl', edl)):
        r = evaluate_with_unc(backbone, head, data, masks[2], mode,
                              val_mask=masks[1])
        cov, risk = r.pop('selective')
        results[mode] = r
        curves[mode] = (cov.tolist(), risk.tolist())
        print(f'[{mode}] F1={r["f1"]:.4f} ECE={r["ece"]:.4f}')
    results['curves'] = curves
    results['protocol'] = PROTOCOL_VERSION
    results['timeline_semantics'] = TIMELINE_SEMANTICS_VERSION

    # --- 选择性预测曲线图 ---
    os.makedirs('figs', exist_ok=True)
    plt.figure(figsize=(3.5, 2.6))
    for mode, (cov, risk) in curves.items():
        plt.plot(cov, risk, '-*' if mode == 'tasgl' else '--o',
                 markersize=4, label=mode.upper())
    plt.xlabel('Coverage')
    plt.ylabel('Selective risk (1 - F1)')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('figs/selective_prediction.pdf')
    plt.close()

    # --- 不确定性时间曲线: 只按时间位置划分, 不虚构已知/新型语义 ---
    _, ent = backbone.mc_uncertainty(data['x'], data['edge_index'],
                                     data['time_gaps'], data['num_nodes'],
                                     T=10)
    ent = ent.cpu().numpy()
    early = {'ts': [], 'entropy': []}
    late = {'ts': [], 'entropy': []}
    for t in range(35, 50):
        m = (time_steps == t) & (labels == 1)
        if m.sum() == 0:
            continue
        target = early if t <= 41 else late
        target['ts'].append(t)
        target['entropy'].append(float(ent[m].mean()))
    results['timeline'] = {'early': early, 'late': late}
    plt.figure(figsize=(3.5, 2.6))
    plt.plot(early['ts'], early['entropy'], '-o', markersize=4,
             label='Early test window (35–41)')
    plt.plot(late['ts'], late['entropy'], '-s', markersize=4,
             label='Late test window (42–49)')
    plt.xlabel('Elliptic time step')
    plt.ylabel('Predictive entropy')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('figs/uncertainty_timeline.pdf')
    plt.close()
    # 曲线、时间图和图文件都完成后再写完成缓存, 避免崩溃后跳过缺失输出。
    save_json(results, args.out)
    print('saved figs/selective_prediction.pdf, figs/uncertainty_timeline.pdf')


if __name__ == '__main__':
    main()
