"""p4_interval.py — H3 与 H4 的十种子区间估计

为什么需要这个程序
------------------
注册后的 H3 是"报告删边后的相对下降比率的点估计与种子 bootstrap 95%CI,
不设 0.6/1.0 硬门限"; H4 是"短训练计划相对参考计划的 F1 损失"。
但既有的 E5 压力矩阵只有两种子、没有逐种子取值, 时序诊断只有三种子,
两者都算不出区间。没有区间就只有点估计, 而点估计在两种子上没有意义。
本程序按十种子把两个量都测出来。

H3 (删边鲁棒性)
---------------
对每个 (模型, 种子): 用参考计划训练一次, 在干净的验证窗上定阈值,
然后在 50% 随机删边后的测试图上重新评估, 逐种子记录
    clean_f1, dropped_f1, drop_ratio = (clean - dropped) / clean
再对种子做 bootstrap, 报告点估计 + 95%CI; 同时对 TASGL 与每个基线做
逐种子配对比较 (差的 bootstrap CI)。

H4 (时间外推)
-------------
对每个 (模型, 种子): 额外用"短训练计划"(只训练到 train_end=20) 再训一次,
与同种子参考计划的 F1 配对, 记录
    rel_loss = (f1_ref - f1_short) / f1_ref
同样报告点估计 + 95%CI 与配对差。

严禁
----
- 不插值、不用其他种子补缺失种子; 某 (模型, 种子) 失败就记为缺失并在
  n_effective 里写明, 不做填充。
- 不做 oracle 之外的攻击解读: 随机删边是结构性扰动, 不是对抗攻击。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

SCHEMA_H3 = 'p4_h3_robustness_interval_v1'
SCHEMA_H4 = 'p4_h4_temporal_interval_v1'


def boot_ci(xs, n_boot=10000, alpha=0.05, seed=12345):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    if len(xs) < 2:
        return None, None, len(xs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xs), size=(n_boot, len(xs)))
    means = xs[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi, len(xs)


def mean_sd(xs):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    if len(xs) == 0:
        return None, None, 0
    sd = float(xs.std(ddof=1)) if len(xs) > 1 else None
    return float(xs.mean()), sd, len(xs)


def paired_bootstrap(a, b, n_boot=10000, alpha=0.05, seed=12345):
    """逐种子配对差的 bootstrap CI 与 Cohen dz。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n < 2:
        return {}
    d = a[:n] - b[:n]
    lo, hi, _ = boot_ci(list(d), n_boot, alpha, seed)
    sd = float(d.std(ddof=1)) if n > 1 else None
    dz = float(d.mean() / sd) if sd else None
    return {'n': int(n), 'mean_diff': float(d.mean()), 'sd_diff': sd,
            'ci95': [lo, hi], 'cohen_dz': dz,
            'ci_excludes_zero': bool(lo is not None and (lo > 0 or hi < 0)),
            'positive_count': int((d > 0).sum())}


def _edge_drop(data, rate, seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    d = {k: v for k, v in data.items()}
    E = data['edge_index'].size(1)
    keep = (torch.rand(E, generator=g) > rate).to(data['edge_index'].device)
    d['edge_index'] = data['edge_index'][:, keep]
    d['time_gaps'] = data['time_gaps'][keep]
    return d


def run(args):
    import p4_core
    import p4_measure
    from common import set_seed
    from data_loader import load_elliptic, standard_split, make_graph_data
    from metrics import eval_with_threshold, evaluate
    from models import build_model
    from p4_experiments import _device, _masks, train_with_audit

    dev = torch.device(_device())
    X, labels, ts, ei, _ = load_elliptic(args.data_dir)
    tr_np, te_np = standard_split(ts, labels, 34)
    va_np = (ts > 29) & (ts <= 34) & (labels >= 0)
    tr_ref = tr_np & ~va_np
    tr_short = (ts <= args.short_end) & (labels >= 0) & ~va_np
    sc = p4_core.fit_scaler_train_only(X, np.where(tr_ref)[0])
    Xs = sc.transform(X).astype(np.float32)
    data = make_graph_data(Xs, labels, ts,
                           torch.as_tensor(ei, dtype=torch.long), dev)

    def _mk(tr_mask):
        return [torch.tensor(m).to(dev) for m in (tr_mask, va_np, te_np)]

    masks_ref = _mk(tr_ref)
    masks_short = _mk(tr_short)

    search = {}
    if args.search and os.path.exists(args.search):
        search = json.load(open(args.search, encoding='utf-8'))

    per_seed = {}
    for name in args.models:
        kw = {}
        if search.get('models', {}).get(name):
            kw = search['models'][name]['best']['config']
        per_seed[name] = {}
        for s in args.seeds:
            rec = {'seed': int(s)}
            try:
                # --- 参考计划 ---
                set_seed(s)
                m = build_model(name, data['in_dim'], **kw)
                m, tr = train_with_audit(m, data, masks_ref,
                                         epochs=args.epochs,
                                         loss_fn=(None if name == 'tasgl'
                                                  else False),
                                         min_epochs=args.min_epochs)
                acc = eval_with_threshold(m, data, masks_ref[1], masks_ref[2])
                thr = acc['threshold']
                rec['ref_f1'] = float(acc['f1'])
                rec['threshold'] = float(thr)
                rec['ref_epochs'] = tr.get('epochs')
                # --- H3: 50% 随机删边后重新评估 (同一模型, 同一阈值) ---
                d2 = _edge_drop(data, args.drop_rate, s)
                acc2 = evaluate(m, d2, masks_ref[2], threshold=thr)
                rec['dropped_f1'] = float(acc2['f1'])
                rec['drop_ratio'] = ((rec['ref_f1'] - rec['dropped_f1'])
                                     / rec['ref_f1']) if rec['ref_f1'] else None
                # --- H4: 短训练计划 ---
                set_seed(s)
                m2 = build_model(name, data['in_dim'], **kw)
                m2, tr2 = train_with_audit(m2, data, masks_short,
                                           epochs=args.epochs,
                                           loss_fn=(None if name == 'tasgl'
                                                    else False),
                                           min_epochs=args.min_epochs)
                acc3 = eval_with_threshold(m2, data, masks_short[1],
                                           masks_ref[2])
                rec['short_f1'] = float(acc3['f1'])
                rec['short_epochs'] = tr2.get('epochs')
                rec['rel_loss'] = ((rec['ref_f1'] - rec['short_f1'])
                                   / rec['ref_f1']) if rec['ref_f1'] else None
                rec['status'] = 'ok'
            except Exception as e:                      # noqa: BLE001
                rec['status'] = 'failed: %s' % type(e).__name__
                print('[%s][seed %d] FAILED %s' % (name, s, e), flush=True)
            per_seed[name][int(s)] = rec
            print('[%s][seed %s] ref=%.4f dropped=%.4f drop_ratio=%.4f '
                  'short=%.4f rel_loss=%s' % (
                      name, s, rec.get('ref_f1') or 0,
                      rec.get('dropped_f1') or 0, rec.get('drop_ratio') or 0,
                      rec.get('short_f1') or 0, rec.get('rel_loss')),
                  flush=True)
            m = None
            m2 = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    # ---------- 汇总 ----------
    def _series(name, key):
        return [per_seed[name][s].get(key) for s in args.seeds
                if per_seed[name].get(s, {}).get('status') == 'ok']

    h3 = {'schema': SCHEMA_H3, 'drop_rate': args.drop_rate,
          'note': ('随机删边是结构性扰动, 不是对抗攻击; 因此本表只支持'
                   'H3 的区间估计, 不支持任何对抗鲁棒性结论'),
          'per_model': {}, 'paired': {}}
    h4 = {'schema': SCHEMA_H4, 'short_train_end': args.short_end,
          'per_model': {}, 'paired': {}}
    for name in args.models:
        for tag, key, out in (('h3', 'drop_ratio', h3),
                              ('h4', 'rel_loss', h4)):
            xs = _series(name, key)
            mu, sd, n = mean_sd(xs)
            lo, hi, _ = boot_ci(xs, args.n_boot)
            out['per_model'][name] = {
                'n': n, 'mean': mu, 'sd': sd, 'ci95': [lo, hi],
                'seeds_used': [s for s in args.seeds
                               if per_seed[name].get(s, {}).get('status')
                               == 'ok'],
                'per_seed': {str(s): per_seed[name][s].get(key)
                             for s in args.seeds},
            }
    base = 'tasgl' if 'tasgl' in per_seed else args.models[0]
    for name in args.models:
        if name == base:
            continue
        a3 = _series(base, 'drop_ratio')
        b3 = _series(name, 'drop_ratio')
        h3['paired'][name] = paired_bootstrap(a3, b3, args.n_boot)
        a4 = _series(base, 'rel_loss')
        b4 = _series(name, 'rel_loss')
        h4['paired'][name] = paired_bootstrap(a4, b4, args.n_boot)

    h3['environment'] = p4_measure.environment_fingerprint()
    h4['environment'] = h3['environment']
    h3['models'] = args.models
    h4['models'] = args.models
    h3['seeds'] = list(args.seeds)
    h4['seeds'] = list(args.seeds)
    os.makedirs(os.path.dirname(args.out_h3) or '.', exist_ok=True)
    p4_core.save_json(h3, args.out_h3)
    p4_core.save_json(h4, args.out_h4)
    print('saved', args.out_h3, args.out_h4)
    return h3, h4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/elliptic')
    ap.add_argument('--models', nargs='+',
                    default=['tasgl', 'mlp', 'graphsage'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--min_epochs', type=int, default=0)
    ap.add_argument('--drop_rate', type=float, default=0.5)
    ap.add_argument('--short_end', type=int, default=20)
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--search', default='results/p4/hpo_search.json')
    ap.add_argument('--out_h3', default='results/p4/h3_interval.json')
    ap.add_argument('--out_h4', default='results/p4/h4_interval.json')
    a = ap.parse_args()
    run(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
