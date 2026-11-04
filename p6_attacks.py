"""p6_attacks.py — H6: 预登记的非 oracle 攻击矩阵。

顺序很重要
----------
本项目的门限纪律是"先登记, 再测量"。H6 之前被判为 not_testable, 原因是
唯一的攻击证据 (E5 边扰动) 由已知标签构造, 属于 oracle。这里补的是非
oracle 路径, 但补的方式不是"跑完看结果再挑一个说法":

  1. ``register`` 先把攻击矩阵、攻击者知识、配对单位、聚合方式写进门限表
     并哈希锁定 (p4_threshold_table_v3)。H6 仍然没有 pass/fail 门限, 因为
     告警容量锚点缺失, 任何通过线都会是任意值。
  2. ``build`` 先校验锁里的 attack_matrix_sha256 与当前攻击矩阵文件一致,
     再执行。锁与文件不一致就直接拒绝, 避免"改了攻击再改口径"。
  3. ``tex`` 只输出产物里已有的数, 不手写字面量。

攻击者能拿到什么
----------------
构造攻击的输入只有三样: 图结构、节点特征、干净模型的预测。函数签名里
没有标签入口 (p1_attacks.apply_nonoracle_attack), 标签只在冻结攻击图之后
进入打分环节。攻击范围限定在评测窗口 incident 的边上。

      命令
--------------
python p6_attacks.py register
python p6_attacks.py build --models tasgl graphsage mlp
python p6_attacks.py tex
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

import p4_thresholds

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, 'results')
THR_DIR = os.path.join(RESULTS, 'p4', 'thresholds')
P6_DIR = os.path.join(RESULTS, 'p6')
ATTACK_MATRIX = os.path.join(ROOT, 'p1_attack_matrix.json')
DEFAULT_EVAL_WINDOW = (35, 49)
DEFAULT_MODELS = ('tasgl', 'graphsage', 'mlp')


# ------------------------------------------------------------------
# 登记
# ------------------------------------------------------------------
def _file_sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def register(n_model_seeds=10, out_dir=THR_DIR, attack_matrix=ATTACK_MATRIX,
             anchors_path=os.path.join(ROOT, 'p4_anchors.json')):
    """把 H6 的测量协议写进门限表并锁定。必须在 build 之前跑。

    H1/H2/H5 的锚点直接沿用 p4_anchors.json, 一个数字都不改; 这次升版只
    动 H6 的状态与测量协议, 因此不改变任何已有判决。
    """
    import p4_core
    anchors = json.load(open(anchors_path, encoding='utf-8'))
    proto = p4_thresholds.h6_measurement_protocol(
        attack_matrix_path=attack_matrix, n_model_seeds=n_model_seeds,
        eval_window=DEFAULT_EVAL_WINDOW, dataset='elliptic')
    res = p4_thresholds.register_justified_thresholds(
        anchors, out_dir=out_dir, declared_seeds=list(range(n_model_seeds)),
        h6_protocol=proto)
    os.makedirs(out_dir, exist_ok=True)
    p4_core.save_json(res, os.path.join(out_dir, 'register.json'))
    print('[P6][H6] 登记版本 =', res['table']['registration_version'])
    print('[P6][H6] lock_id  =', res['lock_id'])
    print('[P6][H6] 攻击矩阵 sha256 =', proto['attack_matrix_sha256'][:16],
          '(%d 个攻击 x %d 个攻击种子)'
          % (len(proto['declared_attacks']), len(proto['attack_seeds'])))
    return res


def load_lock(out_dir=THR_DIR):
    path = os.path.join(out_dir, 'register.json')
    if not os.path.exists(path):
        raise RuntimeError('H6 测量协议未登记: 先跑 p6_attacks.py register')
    reg = json.load(open(path, encoding='utf-8'))
    hyp = (reg.get('table') or {}).get('hypotheses') or {}
    proto = (hyp.get('H6') or {}).get('measurement_protocol')
    if not proto:
        raise RuntimeError('登记表里没有 H6 测量协议; H6 仍是 not_testable')
    return reg, proto


def verify_lock(proto, attack_matrix=ATTACK_MATRIX):
    """执行前的一致性闸门: 锁定协议必须与当前攻击矩阵文件逐位一致。"""
    sha = _file_sha256(attack_matrix)
    if sha != proto.get('attack_matrix_sha256'):
        raise RuntimeError(
            '攻击矩阵文件已改动: 锁定 %s, 当前 %s。改攻击必须先重新登记'
            % (str(proto.get('attack_matrix_sha256'))[:16], sha[:16]))
    return sha


# ------------------------------------------------------------------
# 攻击构建
# ------------------------------------------------------------------
def classify_attacks(specs, x, edge, time_np, eval_mask, p_probe, seed=0):
    """判定每个攻击是否依赖干净预测, 决定扰动能否跨模型/种子复用。

    ``selection != confidence_targeted`` 的攻击只看图与特征, 干净预测不进入
    构造, 因此同一个 (attack, attack_seed) 的扰动图对所有模型种子都相同,
    可以只构造一次。这里不靠推断, 而是用两个不同的干净预测各构造一次比对
    sha256; 若实测与推断不一致就报错, 绝不静默复用。
    """
    from p1_attacks import apply_nonoracle_attack
    p_other = 1.0 - np.asarray(p_probe, dtype=np.float64)
    out = []
    for spec in specs:
        _, _, m1 = apply_nonoracle_attack(x, edge, time_np, eval_mask,
                                          p_probe, spec, seed)
        _, _, m2 = apply_nonoracle_attack(x, edge, time_np, eval_mask,
                                          p_other, spec, seed)
        empirical = (m1['x_sha256'] == m2['x_sha256']
                     and m1['edge_sha256'] == m2['edge_sha256'])
        derived = spec['selection'] != 'confidence_targeted'
        if empirical != derived:
            raise RuntimeError(
                '攻击 %s 的预测依赖性与代码路径推断不一致 (实测 %s, 推断 %s); '
                '拒绝猜测, 请先检查 p1_attacks 的实现'
                % (spec['name'], empirical, derived))
        out.append(dict(spec, p_independent=bool(empirical)))
    return out


class PerturbationCache:
    """按 (attack, attack_seed[, 干净预测哈希]) 复用扰动图。"""

    def __init__(self):
        self.store = {}
        self.hits = 0
        self.misses = 0

    def key(self, spec, attack_seed, p=None):
        if spec.get('p_independent'):
            return (spec['name'], int(attack_seed), '-')
        return (spec['name'], int(attack_seed),
                hashlib.sha256(np.ascontiguousarray(
                    np.asarray(p, dtype=np.float64)).tobytes()).hexdigest()[:16])


    def get(self, key):
        hit = self.store.get(key)
        if hit is None:
            self.misses += 1
        else:
            self.hits += 1
        return hit

    def put(self, key, value):
        self.store[key] = value
        return value


def cache_applies(spec):
    """只缓存边扰动。

    特征噪声的扰动是一整份 203769x165 的 float32 副本 (~134 MB), 缓存
    六份要吃掉近 1 GB 内存; 而它重新构造只要 0.5 s, 不值得换内存。
    """
    return str(spec.get('capability', '')).startswith('edge_')


# ------------------------------------------------------------------
# 执行
# ------------------------------------------------------------------
def _probs_all_nodes(model, data):
    import torch.nn.functional as F
    model.eval()
    with torch.no_grad():
        logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                          data['num_nodes'], hard_mask=True)
    return F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()


def build(models=DEFAULT_MODELS, seeds=None, attack_seeds=None,
          data_dir='./data/elliptic', epochs=200, min_epochs=0,
          search_path=None, out_dir=P6_DIR, device=None,
          eval_window=DEFAULT_EVAL_WINDOW, cache=True):
    import p1_attacks
    from common import set_seed
    from data_loader import compute_time_gaps
    from metrics import eval_with_threshold, evaluate
    from models import build_model
    from p4_experiments import _load_elliptic, _masks, train_with_audit

    _, proto = load_lock()
    matrix_sha = verify_lock(proto)
    cfg = json.load(open(ATTACK_MATRIX, encoding='utf-8'))
    p1_attacks.validate_attack_matrix(cfg)
    seeds = list(seeds if seeds is not None else range(proto['n_model_seeds']))
    attack_seeds = list(attack_seeds if attack_seeds is not None
                        else proto['attack_seeds'])
    if set(attack_seeds) - set(cfg['attack_seeds']):
        raise ValueError('不得在未重新登记的情况下增加 attack_seeds')
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, 'ckpt'); os.makedirs(ckpt_dir, exist_ok=True)
    runs_dir = os.path.join(out_dir, 'runs'); os.makedirs(runs_dir, exist_ok=True)

    t0 = time.time()
    data, ts, labels, _ = _load_elliptic(data_dir, device=dev)
    masks = _masks(ts, labels)
    search_path = search_path or os.path.join(RESULTS, 'p4', 'hpo_search.json')
    search = json.load(open(search_path, encoding='utf-8')) \
        if os.path.exists(search_path) else {}
    x_cpu = data['x'].detach().cpu().numpy()
    edge_cpu = data['edge_index'].detach().cpu().numpy()
    time_np = np.asarray(ts, dtype=np.int64)
    eval_mask = (time_np >= eval_window[0]) & (time_np <= eval_window[1])
    print('[P6][H6] 评测窗口 %s, 窗口内节点 %d (有标签 %d)'
          % (list(eval_window), int(eval_mask.sum()),
             int((eval_mask & (np.asarray(labels) >= 0)).sum())))

    rows, clean_rows = [], []
    cache_store = PerturbationCache() if cache else None
    attack_specs = None
    for model_name in models:
        kw = search.get('models', {}).get(model_name, {}).get('best', {}).get('config') or {}
        for seed in seeds:
            ckpt = os.path.join(ckpt_dir, f'{model_name}_seed{seed}.pt')
            set_seed(seed)
            model = build_model(model_name, data['in_dim'], **kw)
            if os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location='cpu',
                                                weights_only=True))
                model.to(dev).eval()
                reuse = True
            else:
                model, _ = train_with_audit(
                    model, data, masks, epochs=epochs,
                    loss_fn=(None if model_name == 'tasgl' else False),
                    min_epochs=min_epochs)
                torch.save(model.state_dict(), ckpt)
                reuse = False
            acc = eval_with_threshold(model, data, masks[1], masks[2])
            thr = float(acc['threshold'])
            clean_rows.append({'model': model_name, 'seed': int(seed),
                               'f1': float(acc['f1']),
                               'precision': float(acc['precision']),
                               'recall': float(acc['recall']),
                               'threshold': thr, 'reused_checkpoint': reuse,
                               'config': kw})
            clean_p = _probs_all_nodes(model, data)
            if attack_specs is None:
                attack_specs = classify_attacks(
                    cfg['attacks'], x_cpu, edge_cpu, time_np, eval_mask, clean_p)
                print('[P6][H6] 预测无关攻击 (可复用扰动): '
                      + ', '.join(s['name'] for s in attack_specs
                                  if s['p_independent']) or '(无)')
            for spec in attack_specs:
                for attack_seed in attack_seeds:
                    stem = f'{model_name}_seed{seed}_{spec["name"]}_as{attack_seed}'
                    row_path = os.path.join(runs_dir, stem + '.json')
                    if os.path.exists(row_path):
                        rows.append(json.load(open(row_path, encoding='utf-8')))
                        continue
                    key = None
                    entry = None
                    if cache_store is not None:
                        key = cache_store.key(spec, attack_seed, clean_p)
                        entry = cache_store.get(key)
                    if entry is None:
                        ax, ae, manifest = p1_attacks.apply_nonoracle_attack(
                            x_cpu, edge_cpu, time_np, eval_mask, clean_p,
                            spec, attack_seed)
                        entry = (ax, ae, manifest)
                        if cache_store is not None and cache_applies(spec):
                            cache_store.put(key, entry)
                    ax, ae, manifest = entry
                    edge_t = torch.as_tensor(ae, dtype=torch.long)
                    gaps = compute_time_gaps(edge_t, time_np)
                    attacked = dict(data)
                    attacked['x'] = torch.as_tensor(ax, dtype=torch.float32).to(dev)
                    attacked['edge_index'] = edge_t.to(dev)
                    attacked['time_gaps'] = gaps.to(dev)
                    m = evaluate(model, attacked, masks[2], threshold=thr)
                    clean = float(acc['f1'])
                    attacked_f1 = float(m['f1'])
                    drop = None if clean <= 0 else (clean - attacked_f1) / clean
                    row = {'model': model_name, 'seed': int(seed),
                           'attack_name': spec['name'],
                           'attack_seed': int(attack_seed),
                           'clean_f1': clean, 'attacked_f1': attacked_f1,
                           'delta_f1': float(clean - attacked_f1),
                           'relative_drop': drop,
                           'attacked_precision': float(m['precision']),
                           'attacked_recall': float(m['recall']),
                           'threshold': thr,
                           'capability': spec['capability'],
                           'selection': spec['selection'],
                           'knowledge': spec['knowledge'],
                           'goal': spec['goal'],
                           'budget_type': spec['budget_type'],
                           'budget': spec['budget'],
                           'oracle_labels': False,
                           'p_independent': bool(spec['p_independent']),
                           'cost': manifest['cost'],
                           'perturbed_directed_edges':
                               manifest['perturbed_directed_edges'],
                           'x_sha256': manifest['x_sha256'],
                           'edge_sha256': manifest['edge_sha256']}
                    with open(row_path, 'w', encoding='utf-8') as f:
                        json.dump(row, f, ensure_ascii=False)
                    rows.append(row)
            print('[P6][H6] %s seed=%d clean F1=%.4f thr=%.2f (%.0fs)'
                  % (model_name, seed, acc['f1'], thr, time.time() - t0))
    out = summarise(rows, clean_rows, models, seeds, attack_seeds,
                    attack_specs or cfg['attacks'], proto, eval_window,
                    matrix_sha, cache_store)
    path = os.path.join(out_dir, 'nonoracle_attack_matrix.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('[P6][H6] 完成 %d 条记录 -> %s (%.0fs)'
          % (len(rows), path, time.time() - t0))
    return out


def _boot_ci(xs, n_boot=10000, seed=12345):
    from p4_interval import boot_ci
    return boot_ci(xs, n_boot=n_boot, seed=seed)


def _mean_sd(xs):
    from p4_interval import mean_sd
    return mean_sd(xs)


def _reproduce_check(clean_rows,
                     path=os.path.join(RESULTS, 'p4', 'final_comparison.json')):
    """干净 F1 必须复现已发表的十种子结果, 否则整条攻击链路的基线就不可信。

    训练在同一台机器上有数值抖动 (tab:repro 已量化), 所以这里报的是逐种子
    最大偏差, 而不是要求逐位相同。
    """
    if not os.path.exists(path) or not clean_rows:
        return None
    ref = json.load(open(path, encoding='utf-8')).get('models') or {}
    diffs = []
    for row in clean_rows:
        rows = (ref.get(row['model']) or {}).get('rows') or []
        for r in rows:
            if int(r.get('seed', -1)) == int(row['seed']):
                diffs.append({'model': row['model'], 'seed': int(row['seed']),
                              'final_f1': float(r['f1']),
                              'this_run_f1': float(row['f1']),
                              'abs_diff': abs(float(r['f1']) - float(row['f1']))})
    if not diffs:
        return None
    by_model = {}
    for d in diffs:
        m = by_model.setdefault(d['model'], [])
        m.append(d['abs_diff'])
    return {'per_seed': diffs,
            'max_abs_diff': max(d['abs_diff'] for d in diffs),
            'max_abs_diff_by_model': {k: max(v) for k, v in by_model.items()},
            'tolerance_note': ('same machine, same seeds; non-determinism is '
                               'quantified in tab:repro, so this is a '
                               'deviation report and not a bit-equality claim')}


def summarise(rows, clean_rows, models, seeds, attack_seeds, attack_specs,
              proto, eval_window, matrix_sha, cache_store=None):
    """聚合: 先在模型种子内对攻击种子取均值, 再跨种子 bootstrap。"""
    lookup = {(r['model'], r['seed'], r['attack_name'], r['attack_seed']): r
              for r in rows}
    clean = {(r['model'], r['seed']): r for r in clean_rows}
    per_attack = []
    for spec in attack_specs:
        entry = {'attack_name': spec['name'], 'capability': spec['capability'],
                 'selection': spec['selection'], 'knowledge': spec['knowledge'],
                 'goal': spec['goal'], 'budget_type': spec['budget_type'],
                 'budget': spec['budget'], 'oracle_labels': False,
                 'models': {}}
        costs = {}
        for model_name in models:
            drops, atk_f1s, clean_f1s = [], [], []
            for seed in seeds:
                base = clean[(model_name, seed)]['f1']
                vals = [lookup[(model_name, seed, spec['name'], a)]
                        for a in attack_seeds]
                drops.append(np.mean([v['relative_drop'] for v in vals
                                      if v['relative_drop'] is not None])
                             if any(v['relative_drop'] is not None
                                    for v in vals) else None)
                atk_f1s.append(float(np.mean([v['attacked_f1'] for v in vals])))
                clean_f1s.append(float(base))
            lo, hi, n = _boot_ci(drops)
            dm, ds, _ = _mean_sd(drops)
            am, asd, _ = _mean_sd(atk_f1s)
            cm, csd, _ = _mean_sd(clean_f1s)
            entry['models'][model_name] = {
                'clean_f1_mean': cm, 'clean_f1_sd': csd,
                'attacked_f1_mean': am, 'attacked_f1_sd': asd,
                'delta_f1_mean': (cm - am) if (cm is not None and am is not None)
                else None,
                'relative_drop_mean': dm, 'relative_drop_sd': ds,
                'relative_drop_ci95': [lo, hi], 'n_model_seeds': n,
                'ci_excludes_zero': bool(lo is not None and (lo > 0 or hi < 0)),
                'per_seed_drop': [None if d is None else float(d) for d in drops]}
            costs[model_name] = lookup[
                (model_name, seeds[0], spec['name'], attack_seeds[0])]['cost']
        entry['realised_cost_first_seed'] = costs
        per_attack.append(entry)
    # TASGL 与每个基线的下降差 (描述性, 不做检验, 不设门限)
    diffs = []
    if 'tasgl' in models:
        for e in per_attack:
            t = e['models'].get('tasgl') or {}
            for b in models:
                if b == 'tasgl':
                    continue
                bb = e['models'].get(b) or {}
                if not t.get('per_seed_drop') or not bb.get('per_seed_drop'):
                    continue
                d = [x - y for x, y in zip(bb['per_seed_drop'],
                                            t['per_seed_drop'])
                     if x is not None and y is not None]
                lo, hi, n = _boot_ci(d)
                dm, ds, _ = _mean_sd(d)
                diffs.append({'attack_name': e['attack_name'], 'baseline': b,
                              'baseline_minus_tasgl_drop': dm, 'sd': ds,
                              'ci95': [lo, hi], 'n': n,
                              'ci_excludes_zero': bool(
                                  lo is not None and (lo > 0 or hi < 0))})
    knowledge = sorted({s['knowledge'] for s in attack_specs})
    return {
        'schema': 'p6_nonoracle_attack_matrix_v1',
        'clean_f1_vs_final_comparison': _reproduce_check(clean_rows),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'registered_before_execution': True,
        'h6_registration_version': 'p4_threshold_table_v3',
        'attack_matrix_protocol': proto.get('attack_matrix_protocol'),
        'attack_matrix_sha256': matrix_sha,
        'dataset': 'elliptic', 'evaluation_window': list(eval_window),
        'models': list(models), 'seeds': list(seeds),
        'attack_seeds': list(attack_seeds),
        'n_runs': len(rows),
        'oracle_labels_used': False,
        'attack_construction_inputs': ['graph structure', 'node features',
                                       'clean model predictions'],
        'threshold_protocol': ('tuned once on the clean validation split and '
                               'frozen for every attacked evaluation'),
        'paired_unit': 'model seed',
        'attack_seed_role': 'nested repeat inside one model seed',
        'aggregation': ('mean over attack seeds within a model seed, then '
                        'bootstrap over model seeds'),
        'attacker_knowledge_declared': knowledge,
        'attacker_knowledge_not_declared': sorted(
            {'black_box', 'gray_box', 'white_box'} - set(knowledge)),
        'clean': clean_rows,
        'per_attack': per_attack,
        'tasgl_vs_baseline_drop_difference': diffs,
        'h6_status': 'measured_no_gate',
        'gate': None,
        'gate_reason': proto.get('gate_reason'),
        'cache': ({'hits': cache_store.hits, 'misses': cache_store.misses}
                  if cache_store is not None else None),
        'runs': rows,
    }


# ------------------------------------------------------------------
# 出表
# ------------------------------------------------------------------
def esc(s):
    out = str(s).replace('\\', '\\\\').replace('_', '\\_').replace('%', '\\%')
    return out.replace('&', '\\&').replace('#', '\\#')


def brk(s):
    """转义后再允许在下划线处断行。

    攻击名、能力、目标都是 snake_case, 转义后变成一整个不可断的词, 放进
    定宽列就会撑出 overfull。这里在每个 \\_ 后插 \\allowbreak。
    """
    # \allowbreak 后面必须跟 {} 断词, 否则 TeX 会把后面的字母一起吃进
    # 控制序列名 (\_\allowbreakdelete 会变成未知命令 \allowbreakdelete)。
    return esc(s).replace('\\_', '\\_\\allowbreak{}')


def _fmt(x, nd=4):
    return '--' if x is None else f'{float(x):.{nd}f}'


def _pct(x, nd=1):
    return '--' if x is None else f'{100.0 * float(x):.{nd}f}'


PAPER_DIR = os.path.join(ROOT, 'Doc', 'Kimi_Agent_CCF-A实验脚本-25',
                         'TASGL_TDSC投稿')


def tex(results_path=os.path.join(P6_DIR, 'nonoracle_attack_matrix.json'),
        out_dir=None):
    data = json.load(open(results_path, encoding='utf-8'))
    out_dir = out_dir or os.path.join(PAPER_DIR, 'generated_tables')
    os.makedirs(out_dir, exist_ok=True)
    models = data['models']
    lines = []
    lines.append('% AUTO-GENERATED by p6_attacks.py tex -- do not edit by hand')
    # 表 1: 协议 + 攻击者知识 + 成本
    lines.append('\\begin{table*}[!t]')
    lines.append('\\centering')
    lines.append('\\caption{Non-oracle attack matrix for H6: attacker '
                 'capability, knowledge, budget and realised cost. Every '
                 'attack is constructed from the graph, the node features and '
                 'the clean model predictions only; no label enters the '
                 'construction. Budget is the pre-registered fraction of '
                 'edges or the noise standard deviation; cost is what the '
                 'perturbation actually changed.}')
    lines.append('\\label{tab:h6matrix}')
    lines.append('\\footnotesize')
    lines.append('\\setlength{\\tabcolsep}{2pt}')
    lines.append('\\begin{tabularx}{\\textwidth}{'
                 '>{\\raggedright\\arraybackslash}p{0.95in}'
                 '>{\\raggedright\\arraybackslash}p{0.62in}'
                 '>{\\raggedright\\arraybackslash}p{0.78in}'
                 'll>{\\raggedright\\arraybackslash}X'
                 '>{\\raggedright\\arraybackslash}p{0.82in}}')
    lines.append('\\hline')
    lines.append('Attack & Capability & Selection & Knowledge & Budget & '
                 'Goal & Cost\\\\')
    lines.append('\\hline')
    for e in data['per_attack']:
        cost = e['realised_cost_first_seed'].get(models[0], {})
        if 'removed_undirected_pairs' in cost:
            c = f"{cost['removed_undirected_pairs']:,} edge pairs removed"
        elif 'injected_undirected_pairs' in cost:
            c = f"{cost['injected_undirected_pairs']:,} edge pairs injected"
        else:
            c = (f"{cost.get('noisy_evaluation_nodes', 0):,} nodes, "
                 f"$\\ell_2$={cost.get('feature_l2_cost', 0):.1f}")
        budget = ('%s=%s' % (brk(e['budget_type']), e['budget']))
        lines.append('%s & %s & %s & %s & %s & %s & %s\\\\'
                     % (brk(e['attack_name']), brk(e['capability']),
                        brk(e['selection']), brk(e['knowledge']), budget,
                        brk(e['goal']), c))
    lines.append('\\hline')
    lines.append('\\end{tabularx}')
    lines.append('\\par\\vspace{2pt}\\footnotesize '
                 + 'Goal is the attacker objective declared before execution. '
                 + 'Declared knowledge: '
                 + '/'.join(esc(k) for k in
                            data['attacker_knowledge_declared'])
                 + '; ' + ', '.join(
                     esc(k) for k in data['attacker_knowledge_not_declared'])
                 + ' access is not declared and is therefore not covered by '
                 + 'this matrix.\\par')
    lines.append('\\end{table*}')
    lines.append('')
    # 表 2: clean / attacked F1 与下降
    lines.append('\\begin{table*}[!t]')
    lines.append('\\centering')
    lines.append('\\caption{Clean and attacked illicit F1 under the '
                 'non-oracle matrix. Drop is the relative F1 loss in '
                 'percent, averaged over the %d attack seeds inside each '
                 'model seed and then bootstrapped over %d model seeds '
                 '(95%s CI). No gate is applied to H6.}'
                 % (len(data['attack_seeds']), len(data['seeds']),
                    '\\%'))
    lines.append('\\label{tab:h6effect}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{2.5pt}')
    lines.append('\\begin{tabularx}{\\textwidth}{'
                 '>{\\raggedright\\arraybackslash}p{1.0in}'
                 + 'c' * (2 * len(models)) + '}')
    lines.append('\\hline')
    head = ['Attack']
    for m in models:
        head += ['%s attacked' % esc(m), '%s drop %s' % (esc(m), '\\%')]
    lines.append(' & '.join(head) + '\\\\')
    lines.append('\\hline')
    for e in data['per_attack']:
        cells = [brk(e['attack_name'])]
        for m in models:
            mm = e['models'].get(m) or {}
            ci = mm.get('relative_drop_ci95') or [None, None]
            cells.append(_fmt(mm.get('attacked_f1_mean'), 3))
            if ci[0] is None:
                cells.append('--')
            else:
                cells.append('%s [%.1f, %.1f]'
                             % (_pct(mm.get('relative_drop_mean')),
                                100.0 * float(ci[0]), 100.0 * float(ci[1])))
        lines.append(' & '.join(cells) + '\\\\')
    lines.append('\\hline')
    clean_line = ['Clean F1']
    for m in models:
        rows = [r for r in data['clean'] if r['model'] == m]
        mu, sd, _ = _mean_sd([r['f1'] for r in rows])
        clean_line.append('%s$\\pm$%s' % (_fmt(mu, 3), _fmt(sd, 3)))
        clean_line.append('--')
    lines.append(' & '.join(clean_line) + '\\\\')
    lines.append('\\hline')
    lines.append('\\end{tabularx}')
    lines.append('\\end{table*}')
    path = os.path.join(out_dir, 'h6_attack.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('[P6][H6] wrote', path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description='P6 H6 非 oracle 攻击矩阵')
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('register', help='登记 H6 测量协议 (必须在 build 之前)')
    p.add_argument('--n_model_seeds', type=int, default=10)
    p.add_argument('--out_dir', default=THR_DIR)
    p.set_defaults(func=lambda a: (register(a.n_model_seeds, a.out_dir), 0)[1])
    p = sub.add_parser('build', help='执行攻击矩阵')
    p.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--attack_seeds', type=int, nargs='+')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--min_epochs', type=int, default=0)
    p.add_argument('--no_cache', action='store_true')
    p.add_argument('--device', default=None)
    p.add_argument('--out_dir', default=P6_DIR)
    p.set_defaults(func=lambda a: (build(
        a.models, a.seeds, a.attack_seeds, a.data_dir, a.epochs,
        a.min_epochs, out_dir=a.out_dir, device=a.device,
        cache=not a.no_cache), 0)[1])
    p = sub.add_parser('tex', help='生成 generated_tables/h6_attack.tex')
    p.add_argument('--results', default=os.path.join(
        P6_DIR, 'nonoracle_attack_matrix.json'))
    p.add_argument('--out_dir', default=None)
    p.set_defaults(func=lambda a: (tex(a.results, a.out_dir), 0)[1])
    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
