"""p4_third.py — 第三数据集 DGraph-Fin 的训练与评估

为什么加第三数据集
------------------
BitHeNet 是唯一一个"从未参与开发"的数据集, 但只有一个独立数据集时,
任何显著结果都可能是该数据集的特性而非方法特性。DGraph-Fin 提供第三个
数据源, 用来判断结论是否能再往外走一步。

必须声明的限制
--------------
- DGraph-Fin 只有边级时间戳 (edge_timestamp), 没有逐交易时间步,
  因此无法套用 Elliptic / BitHeNet 的严格 rolling 时间划分; 改用官方划分。
- 没有逐交易时间步意味着 TASGL 的时间基分量没有输入, 该分量在本数据集上
  退化为常数; 所以本数据集只能检验"非时间部分", 不能检验时间外推。
- 3,700,550 个节点远大于 Elliptic 的 203,769 个, 6 GB 设备可能放不下。
  放不下就如实记为设备预算不足, 不做任何"缩小后外推"的替代。

严禁
----
- 不把官方划分说成时间划分。
- 不在时间分量为常数的情况下声称检验了时间建模。
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch

SCHEMA = 'p4_third_dataset_v1'


def run(args):
    import p4_core
    import p4_measure
    from common import set_seed
    from data_loader import make_graph_data
    from metrics import eval_with_threshold, measure_efficiency
    from models import build_model
    from p4_experiments import _device, train_with_audit
    from p3_data import load_dgraphfin_real

    dev = torch.device(_device())
    raw = load_dgraphfin_real(args.data_dir, reveal_test_labels=True)
    x = raw['x']
    y = raw['y']
    ei = raw['edge_index']
    n = len(x)
    print('[DGraph-Fin] nodes=%d features=%d edges=%d' % (n, raw['n_features'],
                                                          raw['n_edges']))
    print('[DGraph-Fin] mask_kind=%s split_audit=%s' % (
        raw['mask_kind'], json.dumps(raw['split_audit'], ensure_ascii=False)))

    # 标准化: 只在训练节点上拟合
    tr_np = raw['train_mask']
    sc = p4_core.fit_scaler_train_only(x.astype(np.float64),
                                       np.where(tr_np)[0])
    Xs = sc.transform(x.astype(np.float64)).astype(np.float32)

    # 无逐交易时间步: 时间差全零, 并在结果里写明
    time_steps = np.zeros(n, dtype=np.int64)
    data = make_graph_data(Xs, y, time_steps,
                           torch.as_tensor(ei, dtype=torch.long), dev)
    masks = [torch.tensor(m).to(dev) for m in (raw['train_mask'],
                                               raw['valid_mask'],
                                               raw['test_mask'])]

    search = {}
    if args.search and os.path.exists(args.search):
        search = json.load(open(args.search, encoding='utf-8'))
    res = {'schema': SCHEMA, 'dataset': 'dgraphfin',
           'n_nodes': int(n), 'n_features': int(raw['n_features']),
           'n_edges': int(raw['n_edges']),
           'split': 'official DGraph-Fin split (not a temporal split)',
           'time_semantics': raw['time_semantics'],
           'temporal_component': ('degenerate: no per-transaction time step, '
                                  'so all time gaps are zero'),
           'what_this_can_test': ('non-temporal components only; it cannot '
                                  'test temporal extrapolation'),
           # 划分审计必须随结果落盘, 否则论文里的正类率无从追溯
           'mask_kind': raw['mask_kind'],
           'split_audit': raw['split_audit'],
           'models': {}, 'seeds': list(args.seeds)}
    for name in args.models:
        rows = []
        kw = {}
        if search.get('models', {}).get(name):
            kw = search['models'][name]['best']['config']
        for s in args.seeds:
            rec = {'seed': int(s)}
            try:
                set_seed(s)
                m = build_model(name, data['in_dim'], **kw)
                m, tr = train_with_audit(m, data, masks, epochs=args.epochs,
                                         loss_fn=(None if name == 'tasgl'
                                                  else False),
                                         min_epochs=args.min_epochs)
                acc = eval_with_threshold(m, data, masks[1], masks[2])
                rec.update({'f1': float(acc['f1']),
                            'precision': float(acc['precision']),
                            'recall': float(acc['recall']),
                            'threshold': float(acc['threshold']),
                            'epochs': tr.get('epochs'),
                            'params': int(sum(p.numel()
                                              for p in m.parameters())),
                            'status': 'ok'})
            except torch.cuda.OutOfMemoryError as e:
                rec['status'] = 'oom: %s' % str(e)[:120]
                print('[%s][seed %d] OOM' % (name, s), flush=True)
            except Exception as e:                      # noqa: BLE001
                rec['status'] = 'failed: %s: %s' % (type(e).__name__,
                                                    str(e)[:120])
            rows.append(rec)
            print('[%s][seed %s] %s' % (name, s, rec.get('status')),
                  flush=True)
            m = None
            gc.collect()
            torch.cuda.empty_cache()
        res['models'][name] = {'config': kw, 'rows': rows}
    ok = {k: [r['f1'] for r in v['rows'] if r.get('status') == 'ok']
          for k, v in res['models'].items()}
    for k, v in ok.items():
        if len(v) > 1:
            res['models'][k]['mean_f1'] = float(np.mean(v))
            res['models'][k]['sd_f1'] = float(np.std(v, ddof=1))
    res['environment'] = p4_measure.environment_fingerprint()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    p4_core.save_json(res, args.out)
    print('saved', args.out)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data/dgraphfin')
    ap.add_argument('--models', nargs='+',
                    default=['tasgl', 'mlp', 'graphsage'])
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--min_epochs', type=int, default=0)
    ap.add_argument('--search', default='results/p4/hpo_search.json')
    ap.add_argument('--out', default='results/p5/dgraphfin.json')
    a = ap.parse_args()
    run(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
