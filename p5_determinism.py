"""p5_determinism.py — 同种子重复运行的确定性探针

为什么需要它
------------
归档日志里 BitHeNet 的 "tasgl seed 0" 出现过三个不同的 F1: 0.5484
(second_dataset.log), 0.5373 (second_dataset_tasgl.log), 0.5383
(second_dataset_confirm.json)。这有三种可能:
  (a) 三次运行的命令不同 (例如 epochs 不同), 那么它们本来就不是同一次实验;
  (b) 命令相同但结果不可复现, 那是一个必须写进论文的可复现性缺陷;
  (c) 数据加载顺序受外部状态影响。
不查清楚就在论文里写"十种子全部支持 TASGL"是不行的。

本模块在同一个进程里用同一份代码路径连续训练同一个种子两次, 比较:
  * 测试集概率的 SHA-256 (逐位是否一致)
  * F1 / 阈值 / 参数量
并同时跑一个不同种子作为阴性对照 (不同种子当然应该不同)。

产出: results/p5/determinism.json

用法:
  python p5_determinism.py --dataset elliptic --model tasgl --epochs 60
"""
import argparse
import hashlib
import json
import os

SCHEMA = 'p5_determinism_probe_v1'


def _sha(t):
    import numpy as np
    a = t.detach().cpu().numpy().astype('float64')
    return hashlib.sha256(a.tobytes()).hexdigest()[:24]


def _probs(model, data, mask):
    import torch
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        logits, _ = model(data['x'], data['edge_index'],
                          data['time_gaps'], data['num_nodes'],
                          hard_mask=True)
        return F.softmax(logits[mask], dim=-1)[:, 1]


#: 保留最近一次的概率向量, 用于量化"差多少"而不只是"是否逐位相同"
_LAST = {}


def run_once(dataset, model, seed, epochs, data_dir):
    """完整训练一次并返回可比较的指纹。"""
    import numpy as np
    import torch
    from common import set_seed
    from metrics import eval_with_threshold, measure_efficiency
    from models import build_model

    if dataset == 'elliptic':
        import p4_experiments as p4e
        data, ts, labels, _ = p4e._load_elliptic(data_dir)
        masks = p4e._masks(ts, labels)
    else:
        import p4_experiments as p4e
        data, ts, labels, _ = p4e._load_second(data_dir, 29)
        masks = p4e._masks(ts, labels, train_end=29)

    set_seed(seed)
    import gc
    m = build_model(model, data['in_dim'])
    lf = None if model == 'tasgl' else False
    m, tr = p4_experiments_train(m, data, masks, epochs, lf)
    acc = eval_with_threshold(m, data, masks[1], masks[2])
    eff = measure_efficiency(m, data, _dev())
    p = _probs(m, data, masks[2])
    out = {'seed': seed, 'f1': float(acc['f1']),
           'threshold': float(acc['threshold']),
           'params': int(eff['params']), 'epochs': tr['epochs'],
           'probs_sha256': _sha(p), 'n_scored': int(p.numel())}
    _LAST['probs'] = p.detach().cpu().numpy().astype('float64').copy()
    del m, p
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _dev():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def p4_experiments_train(model, data, masks, epochs, lf):
    import p4_experiments as p4e
    return p4e.train_with_audit(model, data, masks, epochs=epochs,
                                loss_fn=lf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='elliptic',
                    choices=['elliptic', 'bitheteronet'])
    ap.add_argument('--model', default='tasgl')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--other_seed', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--data_dir', default='./data/elliptic')
    ap.add_argument('--out', default='results/p5/determinism.json')
    a = ap.parse_args()
    if a.dataset == 'bitheteronet' and a.data_dir == './data/elliptic':
        a.data_dir = './data/bitheteronet'

    r1 = run_once(a.dataset, a.model, a.seed, a.epochs, a.data_dir)
    p1 = _LAST['probs'].copy()
    r2 = run_once(a.dataset, a.model, a.seed, a.epochs, a.data_dir)
    p2 = _LAST['probs'].copy()
    ctrl = run_once(a.dataset, a.model, a.other_seed, a.epochs, a.data_dir)

    same = r1['probs_sha256'] == r2['probs_sha256']
    # 逐位不同不代表结论不同: 量化最大偏差, 好让读者判断影响
    import numpy as _np
    max_prob_diff = float(_np.abs(p1 - p2).max()) if p1.shape == p2.shape \
        else None
    r1 = dict(r1, n_scored=int(p1.size))
    r2 = dict(r2, n_scored=int(p2.size))
    ctrl_diff = r1['probs_sha256'] != ctrl['probs_sha256']
    out = {'schema': SCHEMA, 'dataset': a.dataset, 'model': a.model,
           'epochs': a.epochs,
           'run1': r1, 'run2': r2, 'control_other_seed': ctrl,
           'same_seed_bitwise_identical': bool(same),
           'max_prob_abs_diff_same_seed': max_prob_diff,
           'same_seed_equivalent_at_1e-6': bool(
               max_prob_diff is not None and max_prob_diff < 1e-6),
           'f1_abs_diff_same_seed': abs(r1['f1'] - r2['f1']),
           'control_differs': bool(ctrl_diff),
           'verdict': ('reproducible' if same and ctrl_diff
                       else 'bitwise_different_metric_stable'
                       if (not same and ctrl_diff
                           and abs(r1['f1'] - r2['f1']) < 1e-12
                           and (max_prob_diff or 1) < 1e-6)
                       else 'not_bitwise_reproducible' if not same
                       else 'inconclusive_control_identical'),
           'note': ('同一进程内连续两次训练同一种子; 若逐位不一致, 说明该栈上'
                    '存在未受控的非确定性来源, 逐种子数值不能在跨运行意义上复用')}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ('run1', 'run2', 'control_other_seed')},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
