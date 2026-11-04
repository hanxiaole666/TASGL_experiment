"""paper_tdsc.py — TASGL TDSC 投稿稿的"论文数据生成与溯源"程序

作用
----
把 P0/P1/P2/P3/P4 各阶段落盘的结果文件 (results/**) 单向、确定地转换为
投稿目录里的论文表格 (.tex)、图输入 (.json) 与溯源清单 (manifest.json)。

设计原则 (与论文主张一致, 程序自身也必须遵守)
---------------------------------------------
1. 论文里出现的每一个数字都必须能由本程序追溯到某个结果文件的某个字段。
   缺失来源一律输出 "n/a" 并登记到 manifest.missing, 绝不用占位值填补。
2. 统计量全部由 p4_core 现算, 不在本文件里硬编码结论。
3. 生成过程确定性: bootstrap 固定随机种子, 输出按文件 SHA-256 校验。
4. 只报事实, 不做裁决: 裁决由 p4_core.adjudicate_claim / gate 审计给出。

用法
----
  python paper_tdsc.py build  [--results_dir ./results]
                              [--paper_dir <投稿目录>]
                              [--out_tables generated_tables]
  python paper_tdsc.py check  [--paper_dir <投稿目录>]
  python paper_tdsc.py ledger [--results_dir ./results]
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

import p4_core

SCHEMA = 'paper_tdsc_v1'

# ------------------------------------------------------------------
# 来源登记: 逻辑名 -> (相对路径, 是否必需)
# ------------------------------------------------------------------
SOURCES = {
    'final': ('results/p4/final_comparison.json', True),
    'hpo': ('results/p4/hpo_search.json', True),
    'hpo_audit': ('results/p4/hpo_audit.json', False),
    'gate_audit': ('results/p4/gate_audit_10seeds.json', False),
    'rootcause': ('results/p4/rootcause_seed0.json', False),
    'calib': ('results/p4/calibration_predictions_elliptic_seed0.json', False),
    'calib_mc': ('results/p4/calibration_predictions_elliptic_seed0_mc.json',
                 False),
    'cost': ('results/p4/cost_measurement.json', False),
    'screen': ('results/p4/per_node_screening.json', False),
    'env': ('results/p4/environment.json', False),
    'thr_just': ('results/p4/threshold_justification.json', False),
    'thr_audit': ('results/p4/threshold_audit.json', False),
    'thr_table': ('results/p4/thresholds/thresholds.json', False),
    'claims': ('results/p4/claims.json', False),
    'rolling': ('results/p4/rolling_plan.json', False),
    # 十种子归档产物已被二十种子执行取代 (见 p5_repro_audit: 归档内部对
    # 同一 seed 存在三个不同值, 且跨执行不可复现), 因此正文一律改用
    # 二十种子执行; 归档产物只作为可复现性审计的对照, 不作为结论来源。
    'second': ('results/p4/second_dataset_20seeds.json', False),
    'second_bl': ('results/p4/second_dataset_baselines.json', False),
    # P0-P3 历史结果 (用于对照与"证据状态"声明)
    'main5': ('results/main_results.json', False),
    'signif5': ('results/main_results_significance.json', False),
    'ablation': ('results/ablation_results.json', False),
    'robust': ('results/robustness_results.json', False),
    'temporal': ('results/temporal_results.json', False),
    'uncert': ('results/uncertainty_results.json', False),
    'crosschain': ('results/crosschain_results.json', False),
    'pareto': ('results/pareto_frontier_check.json', False),
    'p5': ('results/v3/confirm_91f1beb52cbad975/hypothesis_summary.json',
           False),
    'p5_lock': ('results/v3/lock.json', False),
    'p6': ('results/v4/select/summary.json', False),
}


# ------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def f4(x):
    return 'n/a' if x is None else f'{float(x):.4f}'


def f3(x):
    return 'n/a' if x is None else f'{float(x):.3f}'


def f2(x):
    return 'n/a' if x is None else f'{float(x):.2f}'


def f1(x):
    return 'n/a' if x is None else f'{float(x):.1f}'


def ci(lo_hi):
    if not lo_hi:
        return 'n/a'
    lo, hi = lo_hi
    if lo is None or hi is None:
        return 'n/a'
    return f'[{float(lo):.4f}, {float(hi):.4f}]'


def esc(s):
    """LaTeX 转义 (保守处理百分号与下划线)。"""
    return str(s).replace('%', r'\%').replace('_', r'\_')


NAME_MAP = {'tasgl': 'TASGL', 'graphsage': 'GraphSAGE', 'mlp': 'MLP',
            'gcn': 'GCN', 'gat': 'GAT', 'sgc': 'SGC', 'dsgat': 'DSGAT'}


def disp(name):
    return NAME_MAP.get(str(name).lower(), str(name))


def load_sources(results_dir):
    src, meta, missing = {}, {}, []
    for name, (rel, required) in SOURCES.items():
        p = os.path.join(results_dir, rel)
        if not os.path.exists(p):
            src[name] = None
            missing.append({'name': name, 'path': rel, 'required': required})
            continue
        with open(p, encoding='utf-8') as f:
            src[name] = json.load(f)
        meta[name] = {'path': rel, 'sha256': sha256_file(p),
                      'bytes': os.path.getsize(p)}
    return src, meta, missing


def assert_energy_not_measured(src):
    """Release guard: refuse to regenerate tables from a superseded source.

    The final manuscript states that energy was NOT measured because the
    development GPU exposes no power limit, and no joule or efficiency
    figure is reported anywhere.  Two historical result files carry
    power.draw-derived joule values (an A-variant cost measurement and an
    old H2 screening run).  Both are archived byte-for-byte under
    archive/superseded/.  Building from either would silently reintroduce
    claims the camera-ready paper withdrew.
    """
    c = src.get('cost') or {}
    ch = c.get('chain') or {}
    bad_cost = [s.get('stage') for s in (c.get('stages') or [])
                if s.get('energy_j') is not None
                or s.get('energy_measured') is True]
    if bad_cost or ch.get('energy_all_measured') \
            or ch.get('total_energy_j') is not None:
        raise ValueError(
            'cost source reports measured energy %r / total_energy_j=%r; '
            'the final manuscript reports no energy figure. Use the '
            'final-manuscript cost variant (all energy_j=null).'
            % (bad_cost, ch.get('total_energy_j')))
    s = src.get('screen') or {}
    bad_screen = [r.get('model') for r in (s.get('h2') or [])
                  if r.get('joule_per_tx') is not None]
    if bad_screen or s.get('power_measured') is True:
        raise ValueError(
            'H2 screening source reports measured energy %r; the final '
            'manuscript reports no joule figure. Withdrawn variants are '
            'archived under archive/superseded/.' % bad_screen)


# ------------------------------------------------------------------
# 区块 1: 十种子等预算主比较
# ------------------------------------------------------------------
def block_main(src):
    fin = src.get('final')
    if not fin:
        return {'available': False, 'models': [], 'rows': []}
    rows = []
    for name, m in fin['models'].items():
        rs = m['rows']
        rows.append({
            'model': name,
            'config': m.get('config', {}),
            'n': len(rs),
            'f1': float(np.mean([r['f1'] for r in rs])),
            'f1_sd': float(np.std([r['f1'] for r in rs], ddof=1))
            if len(rs) > 1 else None,
            'precision': float(np.mean([r['precision'] for r in rs])),
            'recall': float(np.mean([r['recall'] for r in rs])),
            'params': int(rs[0]['params']),
            'latency_ms': float(np.mean([r['latency_ms'] for r in rs])),
            'latency_sd': float(np.std([r['latency_ms'] for r in rs],
                                       ddof=1)) if len(rs) > 1 else None,
        })
    rows.sort(key=lambda r: -r['f1'])
    return {'available': True, 'models': [r['model'] for r in rows],
            'rows': rows, 'seeds': fin.get('seeds', [])}


def block_paired(src):
    """主比较配对检验 + 门控收敛子集的敏感性配对检验。"""
    fin = src.get('final')
    out = {'available': bool(fin), 'primary': [], 'sensitivity': []}
    if not fin:
        return out
    base = fin['models'].get('tasgl')
    if not base:
        return out
    a_all = [r['f1'] for r in base['rows']]
    for name, m in fin['models'].items():
        if name == 'tasgl':
            continue
        b_all = [r['f1'] for r in m['rows']]
        n = min(len(a_all), len(b_all))
        rep = p4_core.paired_report(a_all[:n], b_all[:n])
        rep['comparison'] = f'TASGL - {disp(name)}'
        rep['note'] = ('主分析: 全部种子, 不做挑选'
                       if n == len(a_all) else '种子数不一致, 已截断')
        out['primary'].append(rep)
    # 敏感性轨道: 只保留门控收敛的种子
    adm = [i for i, r in enumerate(base['rows'])
           if r.get('admission', {}).get('admit')]
    out['admitted_seeds'] = [base['rows'][i]['seed'] for i in adm]
    if len(adm) >= 3:
        a_s = [base['rows'][i]['f1'] for i in adm]
        for name, m in fin['models'].items():
            if name == 'tasgl':
                continue
            b_s = [m['rows'][i]['f1'] for i in adm
                   if i < len(m['rows'])]
            n = min(len(a_s), len(b_s))
            rep = p4_core.paired_report(a_s[:n], b_s[:n])
            rep['comparison'] = f'TASGL - {disp(name)}'
            rep['note'] = ('敏感性分析: 仅门控收敛种子 (事后子集, '
                           '不得作为确认结论)')
            out['sensitivity'].append(rep)
    return out


def block_dual_track(src):
    fin = src.get('final')
    if not fin or 'tasgl' not in fin['models']:
        return {'available': False}
    base = fin['models']['tasgl']
    recs = []
    for r in base['rows']:
        recs.append({'f1': r['f1'], 'admission': r.get('admission') or {}})
    dt = p4_core.dual_track_analysis(recs, key='f1')
    dt['available'] = True
    ma = (dt.get('primary_all_seeds') or {}).get('mean')
    mo = (dt.get('sensitivity_admitted') or {}).get('mean')
    dt['drift'] = None if (ma is None or mo is None) else abs(ma - mo)
    dt['n_all'] = len(recs)
    dt['n_admitted'] = sum(1 for r in recs
                           if (r.get('admission') or {}).get('admit'))
    dt['rejected_seeds'] = [base['rows'][i]['seed'] for i, r in enumerate(recs)
                            if not (r.get('admission') or {}).get('admit')]
    return dt


def block_gate(src):
    """逐种子门控审计 (优先取主比较内嵌的审计, 与论文主表同一批运行)。"""
    fin = src.get('final')
    rows = []
    if fin and 'tasgl' in fin['models']:
        for r in fin['models']['tasgl']['rows']:
            g = r.get('gate') or {}
            rows.append({
                'seed': r['seed'], 'epochs': r.get('epochs'),
                'f1': r['f1'],
                'min_active': g.get('min_active_edge_ratio'),
                'w2': g.get('max_gate_w2_norm'),
                'class': (r.get('admission') or {}).get('classification'),
                'threshold': r.get('threshold'),
            })
    if not rows and src.get('gate_audit'):
        for r in src['gate_audit']['rows']:
            g = r.get('gate') or {}
            rows.append({
                'seed': r['seed'], 'epochs': r.get('epochs'), 'f1': None,
                'min_active': g.get('min_active_edge_ratio'),
                'w2': g.get('max_gate_w2_norm'),
                'class': (r.get('admission') or {}).get('classification'),
                'threshold': None,
            })
    return {'available': bool(rows), 'rows': rows,
            'criteria': p4_core.GATE_CONVERGENCE_CRITERIA}


# ------------------------------------------------------------------
# 区块 2: 等预算 HPO
# ------------------------------------------------------------------
def block_hpo(src):
    h = src.get('hpo')
    if not h:
        return {'available': False, 'rows': []}
    rows = []
    for name, m in h['models'].items():
        tr = m.get('trajectory', [])
        vals = [t['mean_val_f1'] for t in tr]
        uniq = len({round(v, 8) for v in vals})
        rows.append({
            'model': name,
            'n_trials': m.get('n_trials', len(tr)),
            'n_seeds': len(m.get('seeds', [])),
            'best_config': m['best']['config'],
            'best_val_f1': float(m['best']['mean_val_f1']),
            'params': int(m['best']['params']),
            'span': float(max(vals) - min(vals)) if vals else None,
            'distinct_results': uniq,
            'config_sensitive': uniq > 1,
        })
    rows.sort(key=lambda r: -r['best_val_f1'])
    return {'available': True, 'rows': rows,
            'audit': src.get('hpo_audit'),
            'epochs': h.get('epochs'), 'seeds': h.get('seeds')}


# ------------------------------------------------------------------
# 区块 3: 门限依据
# ------------------------------------------------------------------
def block_thresholds(src):
    tj = src.get('thr_just') or {}
    ta = src.get('thr_audit') or {}
    tt = (src.get('thr_table') or {}).get('hypotheses', {})
    rows = []
    for key in sorted(tj.get('hypotheses', {})):
        v = tj['hypotheses'][key]
        rows.append({
            'key': key,
            'old': v.get('original_value'),
            'old_basis': v.get('original_basis'),
            'verdict': v.get('verdict'),
            'new': v.get('replacement'),
            'new_basis': v.get('replacement_basis'),
            'why': v.get('why_arbitrary') or v.get('why') or '',
        })
    new_rules = []
    for h in sorted(tt):
        v = tt[h]
        new_rules.append({
            'hypothesis': h, 'statement': v.get('statement'),
            'basis': v.get('basis'),
            'value': (v.get('derivation') or {}).get('value'),
            'params_budget': (v.get('params_budget') or {}).get('value'),
            'n_gates': len(v.get('gates') or []),
            'status': v.get('status'),
            'open_item': v.get('open_item'),
            # 原始假设字典: 规则行必须由数据字段拼出, 不许留占位英文
            '_hyp': v,
        })
    return {'available': bool(rows), 'withdrawn': rows, 'registered': new_rules,
            'audit': ta.get('per_hypothesis'),
            'all_justified': ta.get('all_justified'),
            'registration_version': (src.get('thr_table') or {}).get(
                'registration_version')}


# ------------------------------------------------------------------
# 区块 4: 校准
# ------------------------------------------------------------------
def block_calibration(src):
    out = {'available': False, 'rows': []}
    for key, label in (('calib', 'deterministic / hard gate'),
                       ('calib_mc', 'MC-Dropout T=10 / soft gate')):
        d = src.get(key)
        if not d:
            continue
        out['available'] = True
        det = d.get('detail', {})
        out['rows'].append({
            'variant': label,
            'overall_ece': d.get('overall_ece'),
            'pos_ece': d.get('positive_class_ece'),
            'neg_ece': d.get('negative_class_ece'),
            'brier': d.get('brier'),
            'nll': det.get('nll'),
            'bins': d.get('bins'),
            'n': d.get('n'),
            'pos_rate': d.get('positive_rate'),
            'f1': det.get('f1'),
        })
    out['warning'] = (src.get('calib') or {}).get('warning')
    return out


# ------------------------------------------------------------------
# 区块 5: 全链路成本
# ------------------------------------------------------------------
def block_cost(src):
    c = src.get('cost')
    if not c:
        return {'available': False}
    stages = [{'stage': s['stage'], 'label': s.get('label', ''),
               'seconds': s.get('total_seconds'),
               'repeats': s.get('repeats'),
               'peak_mem_mb': s.get('peak_mem_mb'),
               'energy_j': s.get('energy_j'),
               'measured': s.get('measured', True)}
              for s in c.get('stages', [])]
    return {'available': True, 'stages': stages, 'chain': c.get('chain'),
            'audit': c.get('audit'), 'deploy': c.get('deploy_efficiency'),
            'protocol': c.get('protocol'), 'env': c.get('environment'),
            'teacher': c.get('teacher_model'), 'kd_weight': c.get('kd_weight')}


# ------------------------------------------------------------------
# 区块 5b: H2 逐交易筛查延迟与能耗
# ------------------------------------------------------------------
def _amort_factor(r):
    """批大小摊薄倍数: 批大小 1 的每笔耗时 / 最大批的每笔耗时。

    优先用测量程序算好的值; 若产物里没有 (例如来自旧一次测量), 就用曲线
    现算, 两者公式相同。不在这里引入任何新的测量。
    """
    af = r.get('batch_amortisation_factor')
    if af:
        return af
    curve = r.get('batch_curve_ms_per_tx') or {}
    vals = [v for v in curve.values() if v]
    if r.get('ms_per_tx') and vals:
        return r['ms_per_tx'] / min(vals)
    return None


def block_h2(src):
    """H2 的语义匹配测量。

    门限是 per-node screening; 只有真正按"一笔/一批交易 + 局部子图"测出的
    延迟才能判 H2。整图一次前向的摊薄值另行标注, 不用于判门限。
    """
    d = src.get('screen')
    if not d:
        return {'available': False, 'rows': []}
    rows = []
    for r in d.get('h2') or []:
        m = (d.get('models') or {}).get(r.get('model')) or {}
        rows.append({
            'model': r.get('model'),
            'hops': m.get('hops'),
            'ms_per_tx': r.get('ms_per_tx'),
            'p95': r.get('ms_per_tx_p95'),
            'bs': r.get('primary_batch_size'),
            'budget': r.get('budget_ms'),
            'margin': r.get('margin_factor'),
            'j_per_tx': r.get('joule_per_tx'),
            'verdict': r.get('verdict'),
            'curve': r.get('batch_curve_ms_per_tx') or {},
            'amort_factor': _amort_factor(r),
            'full_ms': m.get('full_graph_forward_ms'),
            'amort_us': m.get('full_graph_amortized_us_per_node'),
        })
    rows.sort(key=lambda r: (r['model'] != 'tasgl', r['model']))
    return {'available': bool(rows), 'rows': rows,
            'budget': (d.get('h2') or [{}])[0].get('budget_ms'),
            'primary_bs': d.get('h2_primary_batch_size'),
            'rationale': d.get('h2_primary_batch_rationale'),
            'power': d.get('power_measured'),
            'n_warmup': d.get('n_warmup'), 'n_repeat': d.get('n_repeat')}


def tex_h2(b):
    if not b['available']:
        return '% h2: 来源缺失 (results/p4/per_node_screening.json)\n'
    body = []
    for r in b['rows']:
        curve = ', '.join('%s: %.3f' % (k, v)
                          for k, v in sorted(r['curve'].items(),
                                             key=lambda kv: int(kv[0])))
        # Final-manuscript rule: no J/tx column. Energy was not measured
        # (the device exposes no power limit), so no joule figure is
        # generated anywhere in the public release.
        body.append(' & '.join([
            disp(r['model']), str(r['hops']), f3(r['ms_per_tx']),
            f3(r['p95']), f3(r['budget']), f1(r['margin']),
            esc(r['verdict'])]) + ' \\\\')
    body.append('\\hline')
    for r in b['rows']:
        curve = ', '.join('%s: %.3f' % (k, v)
                          for k, v in sorted(r['curve'].items(),
                                             key=lambda kv: int(kv[0])))
        body.append('\\multicolumn{7}{p{0.96\\textwidth}}{\\footnotesize '
                    '%s batch curve (ms per transaction, including '
                    'subgraph construction): %s} \\\\'
                    % (esc(disp(r['model'])), esc(curve)))
    # 摊薄倍数写进表格脚注, 这样正文引用的倍数也能追溯到测量值
    amort = [r for r in b['rows'] if r.get('amort_factor')]
    if amort:
        body.append('\\multicolumn{7}{p{0.96\\textwidth}}{\\footnotesize '
                    'Batch amortisation, batch size 1 divided by the largest '
                    'measured batch: '
                    + ', '.join('%s %.0f' % (esc(disp(r['model'])),
                                             r['amort_factor'])
                                for r in amort) + '} \\\\')
    note = ('Semantics: per-transaction screening. For each transaction a '
            'local %d-hop ego subgraph is built and one forward pass is run '
            'on it. The reported number is the mean over %d timed repeats '
            'after %d warmup repeats, with a cuda synchronize on every '
            'iteration, at fp32. The verdict uses batch size %s because the '
            'budget comes from an arrival rate of 7 transactions per second; '
            'the batch curve is reported so the amortised numbers can be '
            'seen, but they are not used to judge the gate. Energy was not '
            'measured: the device exposes no power limit, so no joule '
            'figure is reported and the screening verdict is decided by '
            'latency only.') % (
        2, b.get('n_repeat'), b.get('n_warmup'), b.get('primary_bs'))
    return _tbl('Per-transaction screening latency (H2 gate, '
                'Elliptic test window).',
                'tab:h2', 'lcccccl',
                ['Model', 'Hops', 'ms/tx', 'p95 ms/tx', 'Budget ms',
                 'Margin', 'Verdict'], body, note=note,
                wide=True, xspec='lcccccl')


# ------------------------------------------------------------------
# 区块 6: seed-0 0.3049 根因
# ------------------------------------------------------------------
def block_rootcause(src):
    r = src.get('rootcause')
    if not r:
        return {'available': False}
    ps = (r.get('per_seed') or [{}])[0]
    rep = {'available': True,
           'reproduce': ps.get('reproduce'),
           'attribution_tuned': ps.get('attribution_tuned'),
           'attribution_fixed': ps.get('attribution_fixed_0.5'),
           'gate': ps.get('gate'),
           'admission': ps.get('admission'),
           'epochs': ps.get('train', {}).get('epochs'),
           'best_val_f1': ps.get('train', {}).get('best_val_f1')}
    return rep


# ------------------------------------------------------------------
# 区块 7: 第二数据集 (BitHeNet) 独立确认
# ------------------------------------------------------------------
def block_second(src):
    d = src.get('second') or src.get('second_bl')
    if not d:
        return {'available': False}
    which = 'confirm' if src.get('second') else 'baselines'
    rows = []
    for name, m in d.get('models', {}).items():
        rs = m['rows']
        rows.append({'model': name, 'n': len(rs), 'config': m.get('config'),
                     'f1': float(np.mean([r['f1'] for r in rs])),
                     'f1_sd': float(np.std([r['f1'] for r in rs], ddof=1))
                     if len(rs) > 1 else None,
                     'params': int(rs[0]['params']),
                     'latency_ms': float(np.mean(
                         [r['latency_ms'] for r in rs]))})
    rows.sort(key=lambda r: -r['f1'])
    # TASGL 在该数据集上的逐种子门控分类: 门控失效是否也能在独立数据上复现
    gate_rows = []
    tg = d.get('models', {}).get('tasgl')
    if tg:
        for r in tg['rows']:
            adm = r.get('admission') or {}
            g = r.get('gate') or {}
            gate_rows.append({
                'seed': r.get('seed'),
                'min_active': g.get('min_active_edge_ratio'),
                'w2': g.get('max_gate_w2_norm'),
                'class': adm.get('classification')})
    out = {'available': True, 'source': which, 'rows': rows,
           'dataset': d.get('dataset'), 'split': d.get('split'),
           'evidence_class': d.get('evidence_class'),
           'independence_basis': d.get('independence_basis'),
           'gate_rows': gate_rows,
           'n_gate_collapsed': sum(1 for r in gate_rows
                                   if r['class'] == 'gate_collapsed_open')}
    paired = d.get('paired') or {}
    if paired:
        out['paired'] = [{'comparison': 'TASGL - %s' % disp(k), **v}
                         for k, v in paired.items()]
    else:
        # 结果文件里没有现成的配对检验时, 现算前两名之间的差异
        if len(rows) >= 2:
            a = [r['f1'] for r in d['models'][rows[0]['model']]['rows']]
            b = [r['f1'] for r in d['models'][rows[1]['model']]['rows']]
            n = min(len(a), len(b))
            rep = p4_core.paired_report(a[:n], b[:n])
            rep['comparison'] = f"{rows[0]['model']} - {rows[1]['model']}"
            out['paired'] = [rep]
    return out


# ------------------------------------------------------------------
# 区块 8: 强断言裁决
# ------------------------------------------------------------------
def block_claims(src):
    c = src.get('claims')
    if not c:
        return {'available': False, 'rows': []}
    rows = [{'id': x.get('id'), 'kind': x.get('kind'),
             'claim': f"{x.get('a')} vs {x.get('b')} ({x.get('metric')})",
             'status': x.get('status'),
             'retraction_required': x.get('retraction_required'),
             'missing': x.get('missing_text'),
             'source': x.get('source')} for x in c.get('claims', [])]
    return {'available': True, 'rows': rows,
            'n_retraction': c.get('n_requiring_retraction'),
            'all_supported': c.get('all_supported')}


# ------------------------------------------------------------------
# 区块 9: 历史 (P0-P3) 对照
# ------------------------------------------------------------------
def _mean_sd(vals):
    if not vals:
        return None, None
    return (float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else None)


def block_legacy(src):
    out = {'available': False}
    m5 = src.get('main5')
    if m5:
        rows = []
        for name, rs in m5.items():
            f1s = [r['f1'] for r in rs]
            mu, sd = _mean_sd(f1s)
            lat = [r.get('latency_ms') for r in rs if r.get('latency_ms')]
            lm, ls = _mean_sd(lat)
            rows.append({'model': name, 'n': len(rs), 'f1': mu, 'f1_sd': sd,
                         'params': rs[0].get('params'),
                         'latency_ms': lm, 'latency_sd': ls,
                         'ece': _mean_sd([r.get('ece') for r in rs
                                          if r.get('ece') is not None])[0],
                         'active_edge_ratio': _mean_sd(
                             [r.get('active_edge_ratio') for r in rs
                              if r.get('active_edge_ratio') is not None])[0]})
        rows.sort(key=lambda r: -(r['f1'] or 0))
        out['main5'] = rows
        out['available'] = True
    out['signif5'] = src.get('signif5')
    ab = src.get('ablation')
    if ab:
        base = _mean_sd([r['f1'] for r in ab.get('full', [])])[0]
        rows = []
        for k, rs in ab.items():
            mu, sd = _mean_sd([r['f1'] for r in rs])
            rows.append({'variant': k, 'n': len(rs), 'f1': mu, 'f1_sd': sd,
                         'delta_vs_full': None if (mu is None or base is None)
                         else mu - base})
        out['ablation'] = rows
        out['available'] = True
    rb = src.get('robust')
    if rb:
        rows = []
        for model, pert in rb.items():
            clean = pert.get('clean:0')
            if not clean:
                continue
            for k, v in pert.items():
                if k.startswith('clean'):
                    continue
                f1 = v[0] if isinstance(v, (list, tuple)) else v
                drop = None if not clean[0] else (clean[0] - f1) / clean[0]
                rows.append({'model': model, 'perturbation': k, 'f1': f1,
                             'relative_drop': drop})
        out['robust'] = rows
        out['available'] = True
    out['pareto'] = src.get('pareto')
    u = src.get('uncert')
    if u:
        out['uncert'] = {k: v for k, v in u.items() if k != 'curves'}
        out['available'] = True
    cc = src.get('crosschain')
    if cc:
        out['crosschain'] = {'meta': cc.get('_meta'),
                             'rows': {k: v for k, v in cc.items()
                                      if k != '_meta'}}
        out['available'] = True
    tp = src.get('temporal')
    if tp:
        out['temporal'] = {k: [r['f1'] for r in v]
                           for k, v in tp.get('rolling', {}).items()}
        out['available'] = True
    return out


def block_protocols(src):
    p5, p5l, p6 = src.get('p5'), src.get('p5_lock'), src.get('p6')
    out = {'available': bool(p5 or p6)}
    if p5:
        out['p5'] = {'lock_id': p5.get('lock_id'),
                     'evidence_status': p5.get('evidence_status'),
                     'config': p5.get('config'),
                     'seeds': p5.get('seeds'),
                     'all_pass': p5.get('all_pass'),
                     'hypotheses': p5.get('hypotheses')}
    if p6:
        rows = p6.get('rows', [])
        out['p6'] = {'n_runs': len(rows),
                     'n_eligible': sum(1 for r in rows
                                       if r.get('eligible')),
                     'winner': p6.get('winner'),
                     'confirmation_authorized':
                         p6.get('confirmation_authorized'),
                     'targets': p6.get('development_targets')}
    return out


def block_env(src):
    e = src.get('env')
    if not e:
        return {'available': False}
    return {'available': True, 'complete': e.get('complete'),
            'missing': e.get('missing', []),
            'fingerprint': e.get('fingerprint'),
            'gpu': e.get('gpu_name'), 'gpu_mem_mb': e.get('gpu_total_mem_mb'),
            'driver': e.get('driver_version'), 'cuda': e.get('cuda_runtime'),
            'cudnn': e.get('cudnn'), 'torch': e.get('torch_version'),
            'python': e.get('python_version'), 'os': e.get('os'),
            'cpu': e.get('cpu'), 'numpy': e.get('numpy_version')}


# ------------------------------------------------------------------
# LaTeX 渲染
# ------------------------------------------------------------------
def _esc_pct(text):
    """转义未转义的百分号, 否则整行会被 LaTeX 当成注释吃掉。"""
    import re as _re
    return _re.sub(r'(?<!\\\\)%', r'\\%', str(text))


def _tbl(caption, label, colspec, header, body, note=None, foot=None,
         wide=False, xspec=None):
    """排版一张表。

    xspec 给出时改用 tabularx: 把长文本列写成 X, 由 LaTeX 自动折行,
    避免表格超出栏宽导致 overfull hbox。colspec 仍保留, 供 xspec 未给出
    时的普通 tabular 使用。
    """
    env = 'table*' if wide else 'table'
    width = '\\textwidth' if wide else '\\columnwidth'
    s = []
    s.append('\\begin{%s}[!t]' % env)
    s.append('\\centering\\footnotesize')
    s.append('\\setlength{\\tabcolsep}{4pt}')
    s.append(f'\\caption{{{caption}}}')
    s.append(f'\\label{{{label}}}')
    if xspec and len(xspec) != len(header):
        # 列符个数与表头个数不一致会把 X 放错列, 使窄列承载长文本并撑出 overfull
        raise ValueError('column spec / header mismatch in %s: %r vs %r'
                         % (label, xspec, header))
    if xspec:
        s.append('\\begin{tabularx}{%s}{%s}' % (width, xspec))
    else:
        s.append('\\begin{tabular}{%s}' % colspec)
    s.append('\\hline')
    s.append(' & '.join(header) + ' \\\\')
    s.append('\\hline')
    s.extend(body)
    s.append('\\hline')
    s.append('\\end{tabularx}' if xspec else '\\end{tabular}')
    if note:
        s.append('\\begin{flushleft}\\footnotesize ' +
                 _esc_pct(note) + '\\end{flushleft}')
    if foot:
        s.append(_esc_pct(foot))
    s.append('\\end{%s}' % env)
    return '\n'.join(s) + '\n'


def tex_main(b):
    if not b['available']:
        return '% main_results: 来源缺失 (results/p4/final_comparison.json)\n'
    body = []
    for r in b['rows']:
        body.append(' & '.join([
            disp(r['model']), str(r['n']), f"{r['params']:,}",
            f"{f1(r['latency_ms'])}$\\pm${f1(r['latency_sd'] or 0)}",
            f"{f4(r['f1'])}$\\pm${f4(r['f1_sd'] or 0)}",
            f4(r['precision']), f4(r['recall'])]) + ' \\\\')
    note = ('Ten seeds, equal hyperparameter budget, same run. '
            'Latency is one full-graph forward pass, not one transaction. '
            'F1 uses a threshold tuned on the validation window only.')
    return _tbl('Ten-seed equal-budget comparison on Elliptic '
                '(test window 35--49).',
                'tab:main10', 'lcccccc',
                ['Model', 'Seeds', 'Params', 'Latency (ms)', 'F1', 'Prec.',
                 'Rec.'], body, note=note, wide=True, xspec='Xcccccc')


def tex_paired(b):
    if not b['available']:
        return '% paired: 来源缺失\n'
    body = []
    for rep in b['primary'] + b['sensitivity']:
        body.append(' & '.join([
            esc(rep['comparison']), str(rep['n']),
            f4(rep['mean_diff']), ci(rep['ci95']), ci(rep['bootstrap_ci95']),
            f2(rep['cohen_dz']), f4(rep['t_p']), f4(rep['wilcoxon_p']),
            f4(rep['mde_at_power80']),
            'yes' if rep['ci_excludes_zero'] else 'no']) + ' \\\\')
    note = ('The paired unit is the seed. CI is the 95% interval of the '
            'mean difference; boot CI uses 2000 resamples. Wilcoxon is the '
            'exact two-sided test. MDE is the smallest effect detectable at '
            'power 0.8. Rows with n=7 use only the seeds whose gate '
            'converged. That subset was chosen after seeing the data, so it '
            'is exploratory and cannot confirm anything.')
    return _tbl('Paired seed-level tests for F1 differences.', 'tab:paired',
                'lccccccccc',
                ['Comparison', 'n', '$\\Delta$F1', '95\\% CI', 'Boot CI',
                 '$d_z$', '$p_t$', '$p_W$', 'MDE', 'CI excl. 0'],
                body, note=note,
                wide=True)


def tex_gate(b, dt):
    if not b['available']:
        return '% gate: 来源缺失\n'
    body = []
    for r in b['rows']:
        body.append(' & '.join([
            str(r['seed']), str(r['epochs'] if r['epochs'] is not None
                                else 'n/a'),
            f3(r['min_active']), f3(r['w2']), f4(r['f1']),
            f2(r['threshold']), _brk(r['class'] or 'n/a')]) + ' \\\\')
    note = ('Pre-registered convergence rule: minimum active-edge ratio '
            '$\\leq$ %.2f AND gate second-layer weight norm $\\geq$ %.2f. '
            'If both fail, the seed is gate\\_collapsed\\_open and the '
            'sparsity claim does not hold on that seed.' % (
                b['criteria'].get('primary_min_active_ratio_upper', 0.98),
                b['criteria'].get('secondary_min_gate_w2_norm', 0.05)))
    foot = ''
    if dt.get('available'):
        pa = dt.get('primary_all_seeds') or {}
        sa = dt.get('sensitivity_admitted') or {}
        foot = ('\\begin{flushleft}\\footnotesize Dual track: all seeds '
                f"{f4(pa.get('mean'))}$\\pm${f4(pa.get('sd'))} "
                f"(n={dt['n_all']}); admitted seeds "
                f"{f4(sa.get('mean'))}$\\pm${f4(sa.get('sd'))} "
                f"(n={dt['n_admitted']}); drift "
                f"{f4(dt.get('drift'))} against tolerance "
                f"{f4(dt.get('stability_tolerance'))}; verdict "
                f"{esc(dt.get('verdict'))}.\\end{{flushleft}}")
    return _tbl('Per-seed gate convergence audit for TASGL.', 'tab:gate',
                'ccccccc',
                ['Seed', 'Epochs', 'Min active', '$\\|W_2\\|$', 'F1',
                 'Thr.', 'Class'], body, note=note, foot=foot,
                wide=True, xspec='ccccccX')


def tex_hpo(b):
    if not b['available']:
        return '% hpo: 来源缺失\n'
    body = []
    for r in b['rows']:
        cfg = ', '.join(f'{k}={v}' for k, v in r['best_config'].items())
        body.append(' & '.join([
            disp(r['model']), str(r['n_trials']), str(r['n_seeds']),
            esc(cfg), f4(r['best_val_f1']), f"{r['params']:,}",
            f4(r['span']),
            'yes' if r['config_sensitive'] else 'no']) + ' \\\\')
    note = ('Same search budget for every model (trials $\\times$ seeds). '
            'Selection uses the validation window only. '
            'Config sensitive = no means the grid produced no measurable '
            'change for that model (SGC is a fixed linear model and takes '
            'no hidden/dropout). Such numbers must be read as untuned.')
    return _tbl('Equal-budget hyperparameter search (validation F1).',
                'tab:hpo', 'lccclccc',
                ['Model', 'Trials', 'Seeds', 'Best config', 'Val F1',
                 'Params', 'Span', 'Config sensitive'],
                body, note=note,
                wide=True, xspec='lccXcccc')


_REPLACEMENT_EN = {
    '报告 relative_drop_ratio 的点估计 + 95%CI (bootstrap over seeds), '
    '不设通过/证伪硬门限':
        'report the relative drop ratio point estimate with a seed '
        'bootstrap 95% CI; no hard pass/falsify gate',
}


def en_replacement(s):
    """结果文件里的替代规则字段可能含中文, 论文表格只输出英文。"""
    s = str(s)
    return _REPLACEMENT_EN.get(s, s)


def tex_thresholds(b):
    if not b['available']:
        return '% thresholds: 来源缺失\n'
    body = []
    # 按 H1..H6 的编号排序, 而不是按字符串: 否则 "H6_attack" 会排到
    # "H5_ece" 前面, 读者会以为编号乱了。
    def _hkey(r):
        import re as _re
        lab = _HIST_KEY_EN.get(r['key'], r['key'])
        m = _re.match(r'H(\d+)', str(lab))
        return (int(m.group(1)) if m else 99, str(lab))
    for r in sorted(b['withdrawn'], key=_hkey):
        body.append(' & '.join([
            esc(_HIST_KEY_EN.get(r['key'], r['key'])), str(r['old']),
            esc(r['old_basis']),
            esc(r['verdict']), _brk(en_replacement(r['new']))]) + ' \\\\')
    note = ('Every threshold with basis = arbitrary is withdrawn. If no '
            'derivable replacement exists, the hypothesis moves to interval '
            'estimation and gets no hard pass/falsify gate. The key '
            'H4\\_attack is the historical name of what is now H6: the '
            'attack-tolerance value was filed under H4 in the early '
            'registration, and H4 is now temporal extrapolation.')
    t1 = _tbl('Audit of the originally pre-registered thresholds.',
              'tab:thr_audit', 'lllll',
              ['Hypothesis', 'Old value', 'Old basis', 'Verdict',
               'Replacement'], body, note=note,
                wide=True, xspec='lllXX')
    body2 = []
    for r in b['registered']:
        val = r['value'] if r['value'] is not None else r['params_budget']
        body2.append(' & '.join([
            esc(r['hypothesis']),
            esc(en_statement(r['hypothesis'], r['statement'],
                             value=r['value'], params=r['params_budget'],
                             hyp=r.get('_hyp'))),
            esc(r['basis']),
            ('n/a' if val is None else
             (f4(val) if isinstance(val, float) else str(val))),
            str(r['n_gates'])]) + ' \\\\')
    note2 = ('Sub-cond. is the number of conjunctive sub-conditions a '
             'candidate must satisfy, not the number of hypotheses and not '
             'a score. H1 and H5 each have two because both halves must '
             'hold. A zero means the hypothesis carries no pass/fail gate: '
             'H3 and H4 are interval estimates and H6 is reported, so none '
             'of the three can return a pass. Of the six requirements, '
             'three are decidable gates.')
    return t1 + '\n' + _tbl(
        # 版本号带下划线, caption 不转义会触发 "Missing $ inserted"
        'Registered requirements after justification (registration version '
        + esc(str(b.get('registration_version') or 'v2')) + ').',
        'tab:thr_new', 'lllll',
        ['Hypothesis', 'Registered rule', 'Basis', 'Value', 'Sub-cond.'],
        body2, note=note2, wide=True, xspec='lXllc')


def tex_calibration(b):
    if not b['available']:
        return '% calibration: 来源缺失\n'
    body = []
    for r in b['rows']:
        body.append(' & '.join([
            esc(r['variant']), f4(r['overall_ece']), f4(r['pos_ece']),
            f4(r['neg_ece']), f4(r['brier']), f4(r['nll']),
            str(r['bins'])]) + ' \\\\')
    note = ('Positive-class ECE is class-mass weighted (bin weight = '
            'positives in the bin / total positives). Overall ECE shrinks '
            'towards zero when the positive rate is small, so it must not be '
            'used alone as a calibration result.')
    return _tbl('Calibration on Elliptic seed 0 (test window, 15 bins).',
                'tab:calib', 'lcccccc',
                ['Variant', 'Overall ECE', 'Pos.-class ECE', 'Neg.-class ECE',
                 'Brier', 'NLL', 'Bins'], body, note=note,
                wide=True, xspec='Xcccccc')


_STAGE_LABEL_EN = {
    'preprocessing': 'Feature preprocessing and scaling',
    'graph_construction': 'Graph construction (edge list to adjacency, '
                          'time gaps)',
    'teacher_training': 'Teacher training (one-off, must be amortized)',
    'distillation': 'Distillation / student training',
    'inference': 'Deployment inference',
    'threshold_selection': 'Validation threshold grid search',
    'calibration': 'Probability calibration fit',
    'update': 'Incremental update / retraining',
}

#: H1--H6 的英文规则行。这里的措辞必须与数据层 register.json 的编号一致:
#: H4 = 时间外推(区间估计), 攻击单列为 H6。二者在早期版本里被混在一起。
_STATEMENT_EN = {
    'H1': 'Mean F1 >= {value:.4f} (non-inferiority anchor from the strongest '
          'equal-budget baseline) AND deployed parameters <= {params:,}',
    # 口径必须写 "per transaction": 预算 7 QPS 描述的是到达率, 一次一笔
    # 才是该口径; 写 per node 会让读者把全图摊薄值拿来比门限。
    'H2': 'Per-transaction screening latency <= 142.857 ms (target 7 '
          'queries per second, one transaction scored at a time with its '
          'ego subgraph built and included)',
    'H3': 'Interval estimation: report the relative illicit-F1 drop after '
          '50% edge deletion as the point estimate with a ten-seed '
          'bootstrap 95% CI; no hard gate',
    'H4': 'Interval estimation: report the relative illicit-F1 loss of a '
          'short training schedule against the reference schedule on the '
          'same evaluation interval, as the point estimate with a '
          'ten-seed bootstrap 95% CI; no hard gate. This is a schedule '
          'change, not an attack',
    'H5': 'Positive-class ECE <= 0.05 and below the same-backbone baseline '
          'on the same metric',
    'H6': '{h6_rule}',
}

#: 历史登记里攻击容限被写在 H4 名下; 现在 H4 是时间外推、攻击单列为 H6。
#: 显示时改名, 并保留原名到新名的映射说明, 否则读者会看到两个不同的 H4。
_HIST_KEY_EN = {'H4_attack': 'H6_attack (filed as H4_attack)'}

#: H6 的规则行随登记版本变化: 未登记攻击容限锚点时是区间报告, 登记后是门限。
#: 这里不允许出现占位式英文 (早期版本曾输出 "source text not in English"),
#: 因为那等于把规则留白交给读者猜。
FORBIDDEN_STATEMENT = 'source text not in English'


def en_label(stage, label):
    return _STAGE_LABEL_EN.get(stage, label if label and not
                               any('\u4e00' <= ch <= '\u9fff'
                                   for ch in str(label)) else stage)


def _has_cjk(s):
    return any('\u4e00' <= ch <= '\u9fff' for ch in str(s))


def _h6_rule(hyp):
    """H6 \u7684\u89c4\u5219\u884c\u5fc5\u987b\u6765\u81ea\u6570\u636e, \u4e0d\u80fd\u7559\u5360\u4f4d\u82f1\u6587\u3002"""
    g = hyp.get('gates') or []
    if g:
        parts = ['%s %s %s' % (x.get('metric'), x.get('op'), x.get('value'))
                 for x in g]
        return 'Pass if ' + ' AND '.join(parts) + '.'
    if hyp.get('status') == 'not_testable':
        return ('Not testable: no non-oracle adaptive attack matrix exists. '
                'Outside the H1--H5 gate set and not part of all_pass.')
    if hyp.get('status') == 'measured_no_gate':
        proto = hyp.get('measurement_protocol') or {}
        n_atk = len(proto.get('declared_attacks') or [])
        n_seed = len(proto.get('attack_seeds') or [])
        n_model = proto.get('n_model_seeds')
        known = proto.get('attacker_knowledge_declared') or []
        return ('Measured, not gated: clean and attacked illicit F1 under the '
                'pre-registered non-oracle attack matrix (%d attacks x %d '
                'attack seeds, attacker knowledge: %s), reported as the '
                'relative drop with a %s-seed bootstrap 95%% CI; attacker '
                'knowledge and cost are declared per attack. No hard gate, '
                'because the alert-capacity anchor is missing. Outside the '
                'H1--H5 gate set.'
                % (n_atk, n_seed, '/'.join(str(k) for k in known), n_model))
    if hyp.get('open_item'):
        return ('Reported, not gated: clean and attacked illicit F1 under '
                'the declared non-oracle attacker, with a seed interval. '
                'No hard gate, because no operational tolerance anchor '
                'exists. Outside the H1--H5 gate set.')
    return ('Reported, not gated: see the locked threshold table for the '
            'quantity and the interval. Outside the H1--H5 gate set.')


def en_statement(h, text, value=None, params=None, hyp=None):
    """\u628a\u6765\u6e90\u91cc\u7684\u4e2d\u6587\u9648\u8ff0\u6362\u6210\u82f1\u6587\u3002

    H1 \u7684\u9648\u8ff0\u76f4\u63a5\u7531 register.json \u7684\u63a8\u5bfc\u503c\u683c\u5f0f\u5316, \u907f\u514d\u82f1\u6587\u6587\u6848\u91cc\u7684\u6570\u5b57\u4e0e
    \u673a\u5668\u63a8\u5bfc\u51fa\u7684\u95e8\u9650\u4e0d\u4e00\u81f4 (\u7ec8\u7248\u66fe\u51fa\u73b0\u6587\u6848 0.5163 / \u63a8\u5bfc\u503c 0.4320 \u5e76\u5b58)\u3002

    \u672a\u767b\u8bb0\u7684\u5047\u8bbe\u4e5f\u4e0d\u8bb8\u8f93\u51fa\u5360\u4f4d\u7b26\u3002\u65e9\u671f\u7248\u672c\u4f1a\u5199
    "(source text not in English; see result file)", \u90a3\u7b49\u4e8e\u628a\u89c4\u5219\u7559\u767d\u4ea4\u7ed9\u8bfb\u8005\u731c;
    \u8fd9\u91cc\u9000\u5316\u4e3a\u7531\u6570\u636e\u5b57\u6bb5 (report_only / quantity) \u62fc\u51fa\u7684\u53ef\u8bfb\u82f1\u6587\u3002
    """
    hyp = hyp or {}
    if h in _STATEMENT_EN:
        tpl = _STATEMENT_EN[h]
        if '{h6_rule' in tpl:
            return tpl.format(h6_rule=_h6_rule(hyp))
        if '{value' in tpl or '{params' in tpl:
            return tpl.format(value=float(value or 0.0),
                              params=int(params or 0))
        return tpl
    ro = [x for x in (hyp.get('report_only') or []) if not _has_cjk(x)]
    if ro:
        out = ('Interval estimation: report ' + ', '.join(str(x) for x in ro)
               + ', with a seed bootstrap interval; no hard gate.')
        q = hyp.get('quantity')
        if q and not _has_cjk(q):
            out += ' Quantity: ' + str(q) + '.'
        return out
    if hyp.get('status'):
        return ('Registered status: %s. See the locked threshold table.'
                % hyp['status'])
    return 'Registered in the locked threshold table; see the artifact.'

def tex_cost(b):
    if not b['available']:
        return '% cost: 来源缺失\n'
    ch = b['chain'] or {}
    # Final-manuscript accounting (Table S-XXII): one-off = graph
    # construction + teacher training + distillation + threshold selection
    # + calibration = 11.51 s; recurring = preprocessing + inference +
    # update = 14.41 s.  Keep this set in sync with the submitted table.
    one_off = {'graph_construction', 'teacher_training', 'distillation',
               'threshold_selection', 'calibration'}
    body = []
    for s in b['stages']:
        body.append(' & '.join([
            esc(s['stage']), esc(en_label(s['stage'], s['label'])),
            f2(s['seconds']),
            str(s['repeats']),
            f1(s['peak_mem_mb']) if s['peak_mem_mb'] else 'n/a',
            'one-off' if s['stage'] in one_off else 'recurring',
            ('%.1f J' % s['energy_j']) if s['energy_j'] is not None
            else 'not measured']) + ' \\\\')
    body.append('\\hline')
    tot_e = ch.get('total_energy_j')
    body.append(' & '.join(['total', 'full chain', f2(ch.get('total_seconds')),
                            '-', '-',
                            f"one-off {f2(ch.get('one_off_seconds'))} s / "
                            f"recurring {f2(ch.get('recurring_seconds'))} s",
                            ('%.1f J' % tot_e) if tot_e is not None
                            else 'not measured']) + ' \\\\')
    if tot_e is not None:
        note = ('Energy is measured by integrating nvidia-smi power.draw '
                'samples over each stage (GPU board power, not wall-plug '
                'power; the device reports no power limit, so no '
                'utilisation-normalised figure is given). Protocol: %s, '
                'warmup=%s, repeat=%s, sync=%s. This protocol is not '
                'comparable with a per-transaction latency budget; the '
                'per-transaction numbers are in Table~\\ref{tab:h2}.') % (
            esc((b['protocol'] or {}).get('semantics')),
            (b['protocol'] or {}).get('n_warmup'),
            (b['protocol'] or {}).get('n_repeat'),
            esc((b['protocol'] or {}).get('sync')))
    else:
        note = ('Energy was not measured; no synthetic values are filled in. '
                'Protocol: %s, warmup=%s, repeat=%s, sync=%s. '
                'This protocol is not comparable with a per-transaction '
                'latency budget.') % (
            esc((b['protocol'] or {}).get('semantics')),
            (b['protocol'] or {}).get('n_warmup'),
            (b['protocol'] or {}).get('n_repeat'),
            esc((b['protocol'] or {}).get('sync')))
    return _tbl('Measured teacher-student full-chain cost '
                '(Elliptic, single run).',
                'tab:cost', 'llccccl',
                ['Stage', 'Meaning', 'Seconds', 'Repeats', 'Peak mem (MB)',
                 'Type', 'Energy'], body, note=note,
                wide=True, xspec='lXcccXX')


def _brk(s):
    """把机器生成的长标记串变成可折行文本。

    例如 `g=soft|s=mc_dropout|T=10|t=tuned_same` 中间没有空格, LaTeX 无法
    折行, 在窄栏里会撑出 overfull hbox。这里把 `|` 换成 `, ` 提供断点。
    """
    t = esc(s).replace('|', ', ')
    return t.replace('\\_', '\\allowbreak\\_')


def tex_rootcause(b):
    if not b['available']:
        return '% rootcause: 来源缺失\n'
    body = []
    rep = b.get('reproduce') or {}
    body.append(' & '.join(['reproduce 0.3049',
                            'yes' if rep.get('reproduced') else 'no',
                            f4(rep.get('observed')), f4(rep.get('recomputed')),
                            f4(rep.get('abs_diff')), f4(rep.get('tol')),
                            _brk(rep.get('key', ''))]) + ' \\\\')
    for st in (b.get('attribution_tuned') or {}).get('steps', []):
        body.append(' & '.join([
            esc('threshold / MC step'),
            esc(st['from_key'].split('|')[-1] + ' -> ' +
                st['to_key'].split('|')[-1]),
            f4(st['from_value']), f4(st['to_value']), f4(st['delta']),
            f2(st['share']), esc(st['status'])]) + ' \\\\')
    note = ('The value 0.3049 does not come from an independent model '
            'state. Its ECE is bit-identical to the seed-0 main run, so it '
            'is the same model re-scored under a different forward '
            'protocol: soft gate, MC-Dropout, and a different threshold.')
    return _tbl('Root cause of the seed-0 F1 = 0.3049 report.',
                'tab:rootcause', 'llccccc',
                ['Item', 'From -> To', 'Before', 'After', '$\\Delta$',
                 'Share', 'Status'], body, note=note,
                wide=True, xspec='lXXcccX')


def _steps(v):
    """[1, 49] -> '1--49'; 其他形态原样返回。"""
    try:
        a, b = list(v)
        return f'{int(a)}--{int(b)}'
    except (TypeError, ValueError):
        return str(v)


def tex_second(b):
    if not b['available']:
        return '% second dataset: 来源缺失\n'
    body = []
    for r in b['rows']:
        body.append(' & '.join([
            disp(r['model']), str(r['n']), f4(r['f1']), f4(r['f1_sd'] or 0),
            f"{r['params']:,}", f1(r['latency_ms'])]) + ' \\\\')
    ds = b.get('dataset') or {}
    note = ('BitHeNet: %s transactions, %s features, %s time steps, '
            'illicit share among labelled nodes %.2f%%. This dataset was '
            'never used for any development, selection or tuning in this '
            'project. Alignment covers only the task and the label '
            'protocol; the feature space is not projected.') % (
        f"{ds.get('n_transactions'):,}", ds.get('n_features'),
        _steps(ds.get('time_steps')),
        100.0 * (ds.get('illicit_rate_labeled') or 0))
    if b.get('n_gate_collapsed'):
        note += (' The gate audit repeats on this dataset: %d of %d TASGL '
                 'seeds are gate-collapsed-open.' % (
                     b['n_gate_collapsed'], len(b.get('gate_rows') or [])))
    t = _tbl('Independent confirmation on BitHeNet (strict rolling '
             '1--29 / 30--34 / 35--49).',
             'tab:second', 'lccccc',
             ['Model', 'Seeds', 'F1', 'SD', 'Params', 'Latency (ms)'],
             body, note=note, xspec='Xccccc')
    p = b.get('paired') or []
    if not p:
        return t
    pbody = []
    for r in p:
        ci = r.get('ci95') or [None, None]
        pbody.append(' & '.join([
            esc(r.get('comparison', '')), str(r.get('n')),
            f4(r.get('mean_diff')),
            '[%s, %s]' % (f4(ci[0]), f4(ci[1])),
            f3(r.get('cohen_dz')), f4(r.get('t_p')), f4(r.get('wilcoxon_p')),
            f4(r.get('mde_at_power80')),
            'yes' if r.get('ci_excludes_zero') else 'no']) + ' \\\\')
    # 种子数必须来自数据, 不能写死: 数据源从十种子换到二十种子时, 写死的
    # 标题会与表里 n 列的 20 直接矛盾。
    n_seeds = None
    for r in p:
        if r.get('n'):
            n_seeds = int(r['n'])
            break
    nw = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
          7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven',
          12: 'twelve', 15: 'fifteen', 20: 'twenty', 25: 'twenty-five',
          30: 'thirty'}
    seed_txt = nw.get(n_seeds, str(n_seeds) if n_seeds else '?')
    return t + '\n' + _tbl(
        'Paired seed-level tests on BitHeNet (%s seeds).' % seed_txt,
        'tab:second_paired', 'lccccccccc',
        ['Comparison', 'n', '$\\Delta$F1', '95\\% CI', '$d_z$', '$p_t$',
         '$p_W$', 'MDE', 'CI excl. 0'], pbody,
        note='Same paired protocol as the main comparison: paired unit is '
             'the seed, sample SD with ddof=1, Student-t 95% CI, exact '
             'two-sided Wilcoxon, and the MDE at power 0.8. The gate values '
             'are the ones registered for Elliptic and are carried over, not '
             're-derived on this dataset.',
        wide=True)


def tex_legacy(b):
    if not b.get('main5'):
        return '% legacy: 来源缺失\n'
    body = []
    for r in b['main5']:
        body.append(' & '.join([
            disp(r['model']), str(r['n']),
            f"{f4(r['f1'])}$\\pm${f4(r['f1_sd'] or 0)}",
            f"{r['params']:,}",
            f"{f1(r['latency_ms'])}$\\pm${f1(r['latency_sd'] or 0)}",
            f4(r['ece'])]) + ' \\\\')
    note = ('Historical five-seed results (protocol 3). Shown only for '
            'contrast with the ten-seed equal-budget results. Their '
            'statistical resolution is too low: with n=5 the smallest '
            'attainable two-sided exact Wilcoxon p is 0.0625, so 0.05 '
            'cannot be reached.')
    t = _tbl('Historical five-seed results (protocol 3, for contrast only).',
             'tab:legacy5', 'lccccc',
             ['Model', 'Seeds', 'F1', 'Params', 'Latency (ms)', 'ECE'],
             body, note=note, xspec='Xccccc')
    if b.get('ablation'):
        body2 = []
        for r in b['ablation']:
            body2.append(' & '.join([
                esc(r['variant']), str(r['n']),
                f"{f4(r['f1'])}$\\pm${f4(r['f1_sd'] or 0)}",
                f4(r['delta_vs_full']) if r['variant'] != 'full' else '--'
            ]) + ' \\\\')
        t += '\n' + _tbl('Component ablation (five seeds, protocol 3).',
                         'tab:ablation', 'lccc',
                         ['Variant', 'Seeds', 'F1', '$\\Delta$ vs full'],
                         body2,
                         note='Five-seed means, with no paired test and no '
                              'confidence interval. Descriptive only.')
    return t


def tex_claims(b):
    if not b['available']:
        return '% claims: 来源缺失\n'
    body = []
    for r in b['rows']:
        body.append(' & '.join([
            esc(r['id']), esc(r['kind']), esc(r['claim']), esc(r['status']),
            'yes' if r['retraction_required'] else 'no']) + ' \\\\')
    note = ('Each strong claim is judged against the pre-registered '
            'evidence requirements. If any requirement is missing the claim '
            'is unverifiable and must be rewritten as "no significant '
            'difference was observed" or withdrawn.')
    return _tbl('Adjudication of strong claims in the earlier draft.',
                'tab:claims', 'llllc',
                ['ID', 'Kind', 'Claim', 'Status', 'Retraction'],
                body, note=note,
                wide=True, xspec='llXlX')


def tex_env(b):
    if not b['available']:
        return '% env: 来源缺失\n'
    body = [
        'GPU & %s (%s MiB) \\\\' % (esc(b['gpu']), f1(b['gpu_mem_mb'])),
        'Driver / CUDA / cuDNN & %s / %s / %s \\\\' % (
            esc(b['driver']), esc(b['cuda']), esc(b['cudnn'])),
        'PyTorch / Python / NumPy & %s / %s / %s \\\\' % (
            esc(b['torch']), esc(b['python']), esc(b['numpy'])),
        'OS / CPU & %s / %s \\\\' % (esc(b['os']), esc(b['cpu'])),
        'Fingerprint & \\texttt{%s} \\\\' % esc(b['fingerprint'][:16]),
        'Complete & %s \\\\' % ('yes' if b['complete'] else 'no'),
    ]
    return _tbl('Environment fingerprint for all timing and cost numbers.',
                'tab:env', 'll', ['Item', 'Value'], body,
                wide=True, xspec='lX')


# ------------------------------------------------------------------
# build
# ------------------------------------------------------------------
def build(results_dir, paper_dir, tables_sub='generated_tables'):
    src, meta, missing = load_sources(results_dir)
    assert_energy_not_measured(src)
    blocks = {
        'main': block_main(src),
        'paired': block_paired(src),
        'dual_track': block_dual_track(src),
        'gate': block_gate(src),
        'hpo': block_hpo(src),
        'thresholds': block_thresholds(src),
        'calibration': block_calibration(src),
        'cost': block_cost(src),
        'h2': block_h2(src),
        'rootcause': block_rootcause(src),
        'second': block_second(src),
        'claims': block_claims(src),
        'legacy': block_legacy(src),
        'protocols': block_protocols(src),
        'env': block_env(src),
        'rolling': src.get('rolling'),
    }
    # ---- 表格 ----
    tables = {
        'main_results.tex': tex_main(blocks['main']),
        'paired_stats.tex': tex_paired(blocks['paired']),
        'gate_audit.tex': tex_gate(blocks['gate'], blocks['dual_track']),
        'hpo_fair.tex': tex_hpo(blocks['hpo']),
        'thresholds.tex': tex_thresholds(blocks['thresholds']),
        'calibration.tex': tex_calibration(blocks['calibration']),
        'cost_chain.tex': tex_cost(blocks['cost']),
        'h2_screening.tex': tex_h2(blocks['h2']),
        'rootcause.tex': tex_rootcause(blocks['rootcause']),
        'second_dataset.tex': tex_second(blocks['second']),
        'claims.tex': tex_claims(blocks['claims']),
        'legacy_contrast.tex': tex_legacy(blocks['legacy']),
        'environment.tex': tex_env(blocks['env']),
    }
    tdir = os.path.join(paper_dir, tables_sub)
    os.makedirs(tdir, exist_ok=True)
    tmeta = {}
    for name, content in tables.items():
        p = os.path.join(tdir, name)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)
        tmeta[name] = {'sha256': sha256_file(p), 'bytes': len(content)}

    # ---- 图输入 ----
    fig = {
        'schema': 'paper_figure_inputs_v1',
        'main10': [{'model': r['model'], 'f1': r['f1'], 'f1_sd': r['f1_sd'],
                    'params': r['params'], 'latency_ms': r['latency_ms']}
                   for r in blocks['main'].get('rows', [])],
        'per_seed_f1': {m: [r['f1'] for r in v['rows']]
                        for m, v in ((src.get('final') or {}).get('models')
                                     or {}).items()},
        'gate': blocks['gate'].get('rows', []),
        'calibration': blocks['calibration'].get('rows', []),
        'cost': blocks['cost'].get('stages', []),
        'second': blocks['second'].get('rows', []),
        'legacy5': blocks['legacy'].get('main5', []),
        'ablation': blocks['legacy'].get('ablation', []),
    }
    ddir = os.path.join(paper_dir, 'generated_data')
    os.makedirs(ddir, exist_ok=True)
    fpath = os.path.join(ddir, 'figure_inputs.json')
    # 合并写回: 保留 paper_evidence.py 生成的历史键 (make_figures.py 依赖),
    # 只覆盖本程序新增的 P4 键。
    old = {}
    if os.path.exists(fpath):
        try:
            with open(fpath, encoding='utf-8') as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = {}
    merged = dict(old)
    merged.update(fig)
    merged['schema'] = fig['schema']
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    fmeta = {'figure_inputs.json': {'sha256': sha256_file(fpath)}}

    # ---- 溯源清单 ----
    prov = os.path.join(paper_dir, 'provenance')
    os.makedirs(prov, exist_ok=True)
    man = {
        'schema': SCHEMA,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'results_dir': os.path.abspath(results_dir),
        'paper_dir': os.path.abspath(paper_dir),
        'sources': meta,
        'missing_sources': missing,
        'tables': tmeta,
        'figures': fmeta,
        'key_numbers': {
            'main10': {r['model']: {'f1': r['f1'], 'f1_sd': r['f1_sd'],
                                    'params': r['params'],
                                    'latency_ms': r['latency_ms'],
                                    'latency_sd': r['latency_sd'],
                                    'precision': r['precision'],
                                    'recall': r['recall']}
                       for r in blocks['main'].get('rows', [])},
            'paired_primary': {r['comparison']: {
                'n': r['n'], 'mean_diff': r['mean_diff'], 'ci95': r['ci95'],
                'cohen_dz': r['cohen_dz'], 'wilcoxon_p': r['wilcoxon_p'],
                't_p': r['t_p'], 'mde_at_power80': r['mde_at_power80'],
                'ci_excludes_zero': r['ci_excludes_zero']}
                for r in blocks['paired'].get('primary', [])},
            'paired_sensitivity': {r['comparison']: {
                'n': r['n'], 'mean_diff': r['mean_diff'], 'ci95': r['ci95'],
                'cohen_dz': r['cohen_dz'], 'wilcoxon_p': r['wilcoxon_p'],
                't_p': r['t_p'], 'mde_at_power80': r['mde_at_power80'],
                'ci_excludes_zero': r['ci_excludes_zero']}
                for r in blocks['paired'].get('sensitivity', [])},
            'admitted_seeds': blocks['paired'].get('admitted_seeds'),
            'dual_track': {k: v for k, v in blocks['dual_track'].items()
                           if k in ('n_all', 'n_admitted', 'drift',
                                    'stability_tolerance', 'verdict',
                                    'rejected_seeds',
                                    'primary_all_seeds',
                                    'sensitivity_admitted')},
            'calibration': blocks['calibration'].get('rows', []),
            'cost_total_seconds': (blocks['cost'].get('chain') or {}).
            get('total_seconds'),
            'cost_one_off_seconds': (blocks['cost'].get('chain') or {}).
            get('one_off_seconds'),
            'cost_one_off_share': (lambda ch: (None if not ch or not
                                               ch.get('total_seconds')
                                               else ch['one_off_seconds'] /
                                               ch['total_seconds']))(
                blocks['cost'].get('chain')),
            'env_fingerprint': blocks['env'].get('fingerprint'),
            'thresholds_registered': blocks['thresholds'].get('registered'),
        },
    }
    mpath = os.path.join(prov, 'manifest.json')
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)

    # ---- 溯源连线 ----
    lines = ['# Provenance (generated by paper_tdsc.py)', '',
             f"- generated_utc: {man['generated_utc']}",
             f"- results_dir: {man['results_dir']}",
             f"- paper_dir: {man['paper_dir']}", '', '## Sources', '']
    for k, v in sorted(meta.items()):
        lines.append(f"- `{k}` <- `{v['path']}`  sha256=`{v['sha256'][:16]}`")
    if missing:
        lines += ['', '## Missing sources (outputs show n/a)', '']
        for m in missing:
            lines.append(f"- `{m['name']}` <- `{m['path']}` "
                         f"required={m['required']}")
    lines += ['', '## Tables', '']
    for k, v in sorted(tmeta.items()):
        lines.append(f"- `{tables_sub}/{k}`  sha256=`{v['sha256'][:16]}`")
    lines += ['', '## Regeneration', '',
              '```', 'python paper_tdsc.py build \\',
              f'  --results_dir {os.path.abspath(results_dir)} \\',
              f'  --paper_dir {os.path.abspath(paper_dir)}',
              '```', '']
    with open(os.path.join(prov, 'lineage.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'[paper_tdsc] wrote {len(tables)} tables -> {tdir}')
    print(f'[paper_tdsc] wrote figure inputs -> {fpath}')
    print(f'[paper_tdsc] wrote manifest -> {mpath}')
    if missing:
        print('[paper_tdsc] MISSING SOURCES: ' +
              ', '.join(m['name'] for m in missing))
    return man


# ------------------------------------------------------------------
# check: 论文正文里的数字是否都能溯源
# ------------------------------------------------------------------
NUM_RE = re.compile(r'(?<![\w.])(\d+\.\d+|\d+)(?![\w.])')
SKIP_PREFIX = ('\\label', '\\ref', '\\cite', '\\bibitem', '\\usepackage',
               '%', '\\begin{tabular}', '\\hline', '\\includegraphics',
               '\\thanks', '\\hyphenation', '\\title', '\\author',
               '\\markboth', '\\hypersetup', '\\begin{IEEEkeywords}')


def _scientific_mantissas(v):
    """Formatting variants of the SAME value written in scientific notation.

    A generated table prints a p-value as 0.000523 while the prose writes
    5.23x10^-4. Both are the same number, so the mantissa must be traceable
    rather than reported as unknown. Only powers of ten are applied; no
    new value is introduced.
    """
    out = set()
    x = abs(v)
    if x == 0:
        return out
    for k in range(1, 7):
        for m in (x * (10 ** k), x / (10 ** k)):
            if 1.0 <= m < 10.0:
                for nd in (1, 2, 3, 4, 5):
                    out.add(f'{m:.{nd}f}')
    return out


def _allowed_numbers(paper_dir):
    allowed = set()
    # 声明式常量 (数据定义 / 结构定义 / 协议配置), 不是实验结果。
    # 每个常量都带来源说明, 见 provenance/paper_constants.json。
    cf = os.path.join(paper_dir, 'provenance', 'paper_constants.json')
    if os.path.exists(cf):
        with open(cf, encoding='utf-8') as f:
            consts = json.load(f)
        for c in consts.get('constants', []):
            allowed.add(str(c['value']))
            try:
                fv = float(str(c['value']).replace(',', ''))
            except ValueError:
                continue
            for nd in (0, 1, 2, 3, 4, 5):
                allowed.add(f'{fv:.{nd}f}')
            allowed.update(_scientific_mantissas(fv))
    for root, _, files in os.walk(paper_dir):
        for fn in files:
            if not (fn.endswith('.tex') and 'generated_tables' in root):
                if not (fn.endswith('.json')):
                    continue
            p = os.path.join(root, fn)
            try:
                with open(p, encoding='utf-8') as f:
                    txt = f.read()
            except OSError:
                continue
            # 与 _scan 同一口径 (见 _normalize_numbers), 否则生成产物里的
            # "3,700,550" 会被切成 3 / 700 / 550, 正文里同一个数反而变成
            # 无法溯源。这是个误报, 不是真缺来源。
            txt = _normalize_numbers(txt)
            for m in NUM_RE.finditer(txt):
                allowed.add(m.group(1))
                fv = float(m.group(1))
                for nd in (1, 2, 3, 4, 5):
                    allowed.add(f'{fv:.{nd}f}')
                allowed.update(_scientific_mantissas(fv))
    return allowed


def check(paper_dir):
    allowed = _allowed_numbers(paper_dir)
    report = []
    for fn in sorted(os.listdir(paper_dir)):
        if not fn.endswith('.tex'):
            continue
        _scan(os.path.join(paper_dir, fn), allowed, report, fn)
    sdir = os.path.join(paper_dir, 'sections')
    if os.path.isdir(sdir):
        for fn in sorted(os.listdir(sdir)):
            if fn.endswith('.tex'):
                _scan(os.path.join(sdir, fn), allowed, report,
                      'sections/' + fn)
    unknown = [r for r in report if r['status'] == 'unknown']
    print(f'[paper_tdsc] scanned {len(report)} numbers, '
          f'{len(unknown)} not traceable to generated outputs')
    for r in unknown[:80]:
        print(f"  {r['file']}:{r['line']}  {r['value']}   | {r['context']}")
    out = os.path.join(paper_dir, 'provenance', 'traceability_check.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'schema': 'paper_traceability_check_v1',
                   'n_scanned': len(report), 'n_unknown': len(unknown),
                   'unknown': unknown}, f, ensure_ascii=False, indent=2)
    print(f'[paper_tdsc] wrote {out}')
    return len(unknown)


def _normalize_numbers(txt):
    """数字比对前的归一化, 正文与生成产物必须用同一口径。

    * ``20{,}480`` 与 ``20,480`` 是同一个数的两种 LaTeX 写法, 前者是
      千分位逗号加薄空格的惯用写法; 不归一化的话会被切成 "20" 和 "480"。
    * 千分位逗号本身也要合并, 否则 "3,700,550" 会被切成三个数。
    """
    txt = re.sub(r'\{\s*,\s*\}', '', txt)
    txt = re.sub(r'(?<=\d),(?=\d)', '', txt)
    return txt


def _scan(path, allowed, report, disp):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    for i, ln in enumerate(lines, 1):
        st = ln.strip()
        if not st or any(st.startswith(p) for p in SKIP_PREFIX):
            continue
        ln = _normalize_numbers(ln)
        for m in NUM_RE.finditer(ln):
            v = m.group(1)
            if v in allowed:
                report.append({'file': disp, 'line': i, 'value': v,
                               'status': 'ok', 'context': st[:120]})
                continue
            fv = float(v)
            hit = any(f'{fv:.{nd}f}' in allowed for nd in (1, 2, 3, 4, 5))
            report.append({'file': disp, 'line': i, 'value': v,
                           'status': 'ok' if hit else 'unknown',
                           'context': st[:120]})


# ------------------------------------------------------------------
# ledger: 证据台账核对 (输出机器可读的核对结果)
# ------------------------------------------------------------------
def ledger(results_dir, ledger_path=None):
    src, meta, missing = load_sources(results_dir)
    rows = []
    for name, (rel, required) in sorted(SOURCES.items()):
        rows.append({'source': name, 'path': rel, 'required': required,
                     'present': os.path.exists(os.path.join(results_dir, rel)),
                     'sha256': (meta.get(name) or {}).get('sha256')})
    print(json.dumps({'schema': 'paper_tdsc_ledger_v1', 'rows': rows},
                     ensure_ascii=False, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')

    p = sub.add_parser('build')
    p.add_argument('--results_dir', default='.',
                   help='包含 results/ 目录的项目根 (SOURCES 路径相对它)')
    p.add_argument('--paper_dir', required=True)
    p.add_argument('--tables_sub', default='generated_tables')
    p.set_defaults(func=lambda a: build(a.results_dir, a.paper_dir,
                                        a.tables_sub))

    p = sub.add_parser('check')
    p.add_argument('--paper_dir', required=True)
    p.set_defaults(func=lambda a: check(a.paper_dir))

    p = sub.add_parser('ledger')
    p.add_argument('--results_dir', default='.')
    p.set_defaults(func=lambda a: ledger(a.results_dir))

    args = ap.parse_args()
    if not getattr(args, 'func', None):
        ap.print_help()
        return 2
    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
