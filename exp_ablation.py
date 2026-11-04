"""exp_ablation.py — E4 消融实验

6 个变体 (对照设计方案 E4):
  full            完整 TASGL
  w/o_rff         去掉低秩 RFF 传播 (替换为恒等映射)
  w/o_sparse      去掉动态稀疏注意力 (稠密消息传递)
  w/o_dro         去掉 DRO (仅类别加权 CE)
  w/o_contrastive 去掉对比损失
  w/o_time_basis  时序基函数退化为 Δt 标量
  w/o_lowrank     低秩可学习投影退化为固定随机投影

输出 results/ablation_results.json (+ _summary.csv): ΔF1, Δ延迟
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_loader import load_elliptic, make_graph_data, standard_split
from models import TASGL, masked_aggregate
from common import set_seed, train_model, save_json, load_if_done, \
    resume_paths, PROTOCOL_VERSION
from losses import TASGLLoss, UncertaintyDROLoss, HardNegContrastiveLoss
from metrics import eval_with_threshold, measure_efficiency


class TASGLAblation(TASGL):
    """支持开关各模块的消融版本。"""

    def __init__(self, *a, use_rff=True, use_sparse=True, use_time_basis=True,
                 use_lowrank=True, **kw):
        super().__init__(*a, **kw)
        self.use_rff, self.use_sparse = use_rff, use_sparse
        if not use_time_basis:
            for layer in (self.attn1, self.attn2):
                dim = layer.val.in_features
                layer.time_enc = _ScalarTime()
                layer.gate = nn.Sequential(
                    nn.Linear(2 * dim + 1, max(dim // 2, 8)), nn.ReLU(),
                    nn.Linear(max(dim // 2, 8), 1))
        if not use_lowrank:
            d = self.prop.U.size(0)
            r = self.prop.V.size(1)
            fixed = torch.randn(d, r) * d ** -0.5
            self.prop.register_buffer('W_fixed', fixed)

    def encode(self, x, edge_index, dt, num_nodes, temperature=10.0,
               hard_mask=False):
        h = torch.relu(self.input_proj(x))
        h = torch.dropout(h, self.dropout, self.training)
        if self.use_rff:
            if hasattr(self.prop, 'W_fixed'):   # 固定随机投影
                rff = torch.cos(h @ self.prop.W_fixed + self.prop.b_rff)
                row, col = edge_index
                acc = torch.zeros_like(rff)
                ws = torch.softmax(self.prop.hop_mix, dim=0)
                acc, cur = acc + ws[0] * rff, rff
                for k in range(1, self.prop.num_hops + 1):
                    cur = masked_aggregate(cur[row], col, num_nodes)
                    acc = acc + ws[k] * cur
                h = h + torch.relu(self.prop.out(acc))
            else:
                h = h + torch.relu(self.prop(h, edge_index, num_nodes))
        if self.use_sparse:
            a1, w1 = self.attn1(h, edge_index, dt, num_nodes, temperature,
                                hard_mask)
            h = h + torch.dropout(torch.relu(a1), self.dropout,
                                  self.training)
            a2, w2 = self.attn2(h, edge_index, dt, num_nodes, temperature,
                                hard_mask)
            h = h + torch.dropout(torch.relu(a2), self.dropout,
                                  self.training)
            gates = (w1, w2)
        else:  # 稠密消息传递
            row, col = edge_index
            a1 = masked_aggregate(self.attn1.val(h[row]), col, num_nodes)
            h = h + torch.dropout(torch.relu(a1), self.dropout,
                                  self.training)
            a2 = masked_aggregate(self.attn2.val(h[row]), col, num_nodes)
            h = h + torch.dropout(torch.relu(a2), self.dropout,
                                  self.training)
            gates = None
        return h, gates


class _ScalarTime(nn.Module):
    out_dim = 1

    def forward(self, dt):
        return dt.unsqueeze(-1)


VARIANTS = {
    'full':            dict(),
    'w/o_rff':         dict(use_rff=False),
    'w/o_sparse':      dict(use_sparse=False),
    'w/o_time_basis':  dict(use_time_basis=False),
    'w/o_lowrank':     dict(use_lowrank=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/elliptic')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--out', default='results/ablation_results.json')
    ap.add_argument('--overwrite', action='store_true',
                    help='忽略已有断点与结果, 全部重新训练')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, labels, time_steps, edge_index, _ = load_elliptic(args.data_dir)
    data = make_graph_data(X, labels, time_steps, edge_index, device)
    tr, te = standard_split(time_steps, labels, 34)
    va = (time_steps > 29) & (time_steps <= 34) & (labels >= 0)
    tr = tr & ~va
    masks = [torch.tensor(m).to(device) for m in (tr, va, te)]
    train_mask, val_mask, test_mask = masks

    variants = list(VARIANTS.items()) + [('w/o_dro', None),
                                         ('w/o_contrastive', None)]
    results = {}
    for vname, vkw in variants:
        results[vname] = []
        for seed in args.seeds:
            rp, cp = resume_paths('ablation', f'{vname}_seed{seed}')
            cached = load_if_done(rp, args.overwrite)
            if cached is not None:
                results[vname].append(cached)
                print(f'[{vname}][{seed}] 已完成, 跳过 '
                      f'(F1={cached["f1"]:.4f})')
                continue
            set_seed(seed)
            if isinstance(vkw, dict):
                model = TASGLAblation(data['in_dim'], **vkw)
            else:  # 损失消融: 结构完整, 换损失
                model = TASGL(data['in_dim'])
            model = model.to(device)
            if vname == 'w/o_dro':
                lf = TASGLLoss(use_dro=False, use_perturb=False)
            elif vname == 'w/o_contrastive':
                lf = TASGLLoss(enabled=False)
            else:
                lf = None
            model, _ = train_model(model, data, train_mask, val_mask,
                                   device, epochs=args.epochs, loss_fn=lf,
                                   ckpt_path=cp)
            acc = eval_with_threshold(model, data, val_mask, test_mask)
            eff = measure_efficiency(model, data, device, n_repeat=50)
            r = {**acc, **eff, 'seed': seed,
                 'protocol': PROTOCOL_VERSION}
            save_json(r, rp)          # 单组合完成立即落盘
            results[vname].append(r)
            print(f'[{vname}][{seed}] F1={acc["f1"]:.4f} '
                  f'lat={eff["latency_ms"]:.2f}ms')
    save_json(results, args.out)

    base = pd.DataFrame(results['full'])
    rows = []
    for vname, runs in results.items():
        df = pd.DataFrame(runs)
        rows.append({
            'variant': vname,
            'f1': f'{df.f1.mean():.4f}±{df.f1.std():.4f}',
            'dF1': f'{df.f1.mean() - base.f1.mean():+.4f}',
            'latency_ms': f'{df.latency_ms.mean():.2f}',
            'd_latency_ms': f'{df.latency_ms.mean() - base.latency_ms.mean():+.2f}',
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out.replace('.json', '_summary.csv'), index=False)
    print('\n' + summary.to_string(index=False))


if __name__ == '__main__':
    main()
