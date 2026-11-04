"""p2_core.py — P2-1 至 P2-5 可审计原语。

本模块补齐投稿评审的五项缺口，且不含任何预设结论；所有判定只来自传入的
真实观测值，缺失即标记为 incomplete_*，不允许把部分结果写成完整证据。

  P2-1  10–20 种子的预注册显著性检验：功效/MDE、Bootstrap、符号检验、Holm、
        嵌套单元不得冒充独立配对单元
  P2-2  预先固定并哈希锁定的 H1–H5 门限表（注册后不得事后改阈值）
  P2-3  未被污染的独立时间窗声明与严格 rolling 污染审计
  P2-4  基线公平超参搜索预算审计（等预算、等选择规则、等窗口）
  P2-5  正类 ECE 与分层校准（含最小桶计数、分层不足与单调性检查）
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

from p0_core import (apply_calibrator, atomic_json, canonical_json, file_sha256,
                     fit_calibrator, holm_adjust, probability_metrics, read_json,
                     sha256_bytes, strict_rolling_view)


# ------------------------------------------------------------------
# P2-1：预注册种子池与统计功效
# ------------------------------------------------------------------
DEFAULT_SEED_POOL = list(range(20))          # 默认 20 个种子，落在 10–20 区间
MIN_SEEDS = 10                                # 预设下限：低于此值不得声称完整显著性
MAX_SEEDS = 20


def validate_seed_pool(seeds, declared=None, min_seeds=MIN_SEEDS,
                       max_seeds=MAX_SEEDS):
    """校验预注册种子池；重复、越界或与已声明池不一致均拒绝。"""
    out = [int(s) for s in seeds]
    if not out:
        raise ValueError('种子池不得为空')
    if len(set(out)) != len(out):
        raise ValueError(f'种子池含重复种子: {out}')
    if any(s < 0 for s in out):
        raise ValueError('种子必须为非负整数')
    if declared is not None:
        declared = [int(s) for s in declared]
        missing = [s for s in declared if s not in out]
        extra = [s for s in out if s not in declared]
        if missing or extra:
            raise ValueError(f'种子池与预注册不一致: missing={missing} extra={extra}')
        out = declared
    if len(out) < min_seeds:
        raise ValueError(f'显著性检验至少需要 {min_seeds} 个种子，当前 {len(out)}')
    if len(out) > max_seeds:
        raise ValueError(f'种子池超过预注册上限 {max_seeds}')
    return out


def seed_pool_hash(seeds, meta=None):
    """种子池指纹：事后增删种子会改变哈希，用于检测 p-hacking。"""
    payload = {'seeds': [int(s) for s in seeds], 'meta': meta or {}}
    return sha256_bytes(canonical_json(payload).encode())


def wilcoxon_min_p(n):
    """n 个配对单元的精确双侧 Wilcoxon 最小可达 p 值（全部同号时）。"""
    n = int(n)
    if n < 1:
        raise ValueError('n 必须为正')
    return float(min(1.0, 2.0 / (2 ** n)))


def sign_test_p(diff):
    """精确二项符号检验；作为 Wilcoxon 的分布无关稳健性补充。"""
    d = np.asarray(diff, dtype=np.float64)
    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    n = pos + neg
    if n == 0:
        return {'p_value': 1.0, 'positive': 0, 'negative': 0, 'n_nonzero': 0}
    res = stats.binomtest(min(pos, neg), n, 0.5, alternative='two-sided')
    return {'p_value': float(res.pvalue), 'positive': pos, 'negative': neg,
            'n_nonzero': n}


def bootstrap_mean_ci(values, n_boot=10000, seed=0, alpha=0.05):
    """配对差值均值的百分位 Bootstrap 置信区间；种子固定以便复现。"""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return None
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(v), size=(int(n_boot), len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def required_cohen_dz(n, alpha=0.05, power=0.8):
    """配对 t 设计在给定种子数下达到目标功效所需的标准化效应量。"""
    n = int(n)
    if n < 2:
        raise ValueError('n 至少为 2')
    dof = n - 1
    t_crit = float(stats.t.ppf(1 - alpha / 2, dof))
    t_power = float(stats.t.ppf(power, dof))
    return float((t_crit + t_power) / math.sqrt(n))


def minimum_detectable_effect(n, sd=None, alpha=0.05, power=0.8):
    """给定种子数与配对差值标准差时，可检出的最小差值（原始单位）。

    sd 为 None 时只返回标准化效应量，不得事后用观测 sd 反推“功效已足够”。
    """
    dz = required_cohen_dz(n, alpha, power)
    return {'n': int(n), 'alpha': float(alpha), 'power': float(power),
            'required_cohen_dz': dz,
            'mde_units': (float(dz * sd) if sd is not None and sd > 0 else None),
            'wilcoxon_min_attainable_p': wilcoxon_min_p(n)}


def required_seeds_for_mde(mde_units, sd, alpha=0.05, power=0.8, max_n=200):
    """反解达到指定可检出差值所需的种子数；超出上限返回 None。"""
    if sd is None or sd <= 0 or mde_units is None or mde_units <= 0:
        return None
    for n in range(MIN_SEEDS, int(max_n) + 1):
        if minimum_detectable_effect(n, sd, alpha, power)['mde_units'] <= mde_units:
            return n
    return None


def observed_power(dz, n, alpha=0.05):
    """观测效应量下的近似功效（正态近似）。

    仅作描述性报告：事后功效不能反过来证明“无差异”，预注册的 MDE 才是
    有意义的灵敏度指标。
    """
    z = float(stats.norm.ppf(1 - alpha / 2))
    a = abs(float(dz)) * math.sqrt(int(n))
    return float(stats.norm.cdf(a - z) + stats.norm.cdf(-a - z))


def aggregate_records(records, value_key='f1', seed_key='seed', rule='mean'):
    """把嵌套重复（时间窗/攻击种子）折叠到种子级配对单元。

    窗口或攻击种子是嵌套重复，不是独立配对单元；直接展开会虚增 n 并伪造
    显著性，因此必须先按种子聚合。
    """
    if isinstance(records, dict):
        records = [{'seed': k, value_key: v} for k, v in records.items()]
    buckets = {}
    for row in records:
        seed = int(row[seed_key])
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError(f'seed={seed} 的 {value_key} 非有限数')
        buckets.setdefault(seed, []).append(value)
    out = []
    for seed in sorted(buckets):
        vals = np.asarray(buckets[seed], dtype=np.float64)
        agg = float(np.mean(vals)) if rule == 'mean' else float(np.median(vals))
        out.append({'seed': seed, value_key: agg, 'n_nested': int(len(vals))})
    return out


def _paired_base(records_a, records_b, declared_seeds, value_key, alpha):
    from p0_core import paired_statistics
    return paired_statistics(records_a, records_b, declared_seeds, value_key,
                             alpha)


def paired_statistics_multi(records_a, records_b, declared_seeds,
                            value_key='f1', alpha=0.05, n_boot=10000,
                            boot_seed=0, nested_rule='mean', sd_prior=None):
    """多种子配对检验：在 P0-5 基础上补充功效、Bootstrap、符号检验与嵌套审计。"""
    declared = validate_seed_pool(declared_seeds)
    agg_a = aggregate_records(records_a, value_key, rule=nested_rule)
    agg_b = aggregate_records(records_b, value_key, rule=nested_rule)
    nested = max([r['n_nested'] for r in agg_a + agg_b] or [1])
    base = _paired_base(agg_a, agg_b, declared, value_key, alpha)
    diff = None
    if base.get('per_seed'):
        diff = np.array([r['difference'] for r in base['per_seed']],
                        dtype=np.float64)
    extra = {'protocol': 'p2_multiseed_paired_v1',
             'nested_units_per_seed': int(nested),
             'nested_aggregation_rule': nested_rule,
             'nested_units_collapsed_to_seed': bool(nested > 1),
             'seed_pool_sha256': seed_pool_hash(declared),
             'bootstrap': {'n_boot': int(n_boot), 'seed': int(boot_seed),
                           'ci95_mean_difference': None},
             'sign_test': None,
             'sensitivity': minimum_detectable_effect(len(declared), sd_prior),
             'seed_pool_size_ok': bool(MIN_SEEDS <= len(declared) <= MAX_SEEDS)}
    if diff is not None and len(diff) >= 2:
        extra['bootstrap']['ci95_mean_difference'] = bootstrap_mean_ci(
            diff, n_boot, boot_seed, alpha)
        extra['sign_test'] = sign_test_p(diff)
        if base.get('cohen_dz') is not None:
            extra['observed_power_descriptive'] = observed_power(
                base['cohen_dz'], len(diff), alpha)
    extra['note'] = ('窗口/攻击种子为嵌套重复，已折叠到种子级；'
                     '事后增删种子会改变 seed_pool_sha256，不得视为同一轮预注册')
    return {**base, **extra}


def multi_comparison(p_values, alpha=0.05):
    """Holm 校正；p 值为 None（不完整）时不得进入显著性结论。"""
    valid = [i for i, p in enumerate(p_values) if p is not None]
    out = [None] * len(p_values)
    flags = [False] * len(p_values)
    if valid:
        adj = holm_adjust([p_values[i] for i in valid])
        for i, p in zip(valid, adj):
            out[i] = p
            flags[i] = bool(p < alpha)
    return out, flags


# ------------------------------------------------------------------
# P2-2：预先固定的 H1–H5 门限表
# ------------------------------------------------------------------
THRESHOLD_SCHEMA = 'p2_hypothesis_threshold_table_v1'
_OPS = {
    '>=': lambda a, b: a >= b,
    '<=': lambda a, b: a <= b,
    '>': lambda a, b: a > b,
    '<': lambda a, b: a < b,
    '==': lambda a, b: a == b,
}
_STATUS_ORDER = ['incomplete_missing_metric', 'incomplete_missing_observation',
                 'incomplete_missing_seeds', 'incomplete_seed_ledger',
                 'incomplete_missing_baseline', 'falsified',
                 'inconclusive', 'pass']


def _combine(statuses):
    """取最严重状态：incomplete > falsified > inconclusive > pass。"""
    if not statuses:
        return 'incomplete_missing_metric'
    for prefix in ('incomplete',):
        for s in statuses:
            if s.startswith(prefix):
                return s
    for s in statuses:
        if s == 'falsified':
            return 'falsified'
    for s in statuses:
        if s.startswith('inconclusive'):
            return s
    for s in statuses:
        if s != 'pass':
            return s
    return 'pass'


def validate_threshold_table(table):
    """门限表结构校验；缺字段或非法算子一律拒绝，避免静默放宽门槛。"""
    if table.get('schema') != THRESHOLD_SCHEMA:
        raise ValueError(f'门限表 schema 必须为 {THRESHOLD_SCHEMA}')
    if not table.get('hypotheses'):
        raise ValueError('门限表必须包含 hypotheses')
    for name, spec in sorted(table['hypotheses'].items()):
        if not spec.get('statement'):
            raise ValueError(f'{name} 缺少 statement')
        if not spec.get('gates'):
            raise ValueError(f'{name} 缺少 gates')
        for gate in spec['gates']:
            if not gate.get('metric'):
                raise ValueError(f'{name} 的 gate 缺少 metric')
            if 'pass_value' not in gate and 'compare_to_metric' not in gate:
                raise ValueError(f'{name}/{gate["metric"]} 缺少 pass_value 或 compare_to_metric')
            for key in ('pass_op',):
                if gate.get(key) not in _OPS:
                    raise ValueError(f'{name}/{gate["metric"]} 的 {key} 非法')
            if 'falsify_value' in gate and gate.get('falsify_op') not in _OPS:
                raise ValueError(f'{name}/{gate["metric"]} 的 falsify_op 非法')
    return True


def threshold_table_hash(table):
    """门限表指纹只由表内容决定；注册时间与署名等元信息不参与哈希。"""
    return sha256_bytes(canonical_json(
        {'schema': THRESHOLD_SCHEMA, 'table': table}).encode())


def register_thresholds(table, output_dir, meta=None):
    """在看到任何结果前注册门限表并写入锁；之后改表即被视为事后修改。"""
    validate_threshold_table(table)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    digest = threshold_table_hash(table)
    lock = {'schema': 'p2_threshold_lock_v1',
            'meta': meta or {},
            'table_sha256': digest,
            'lock_id': sha256_bytes(canonical_json(
                {'schema': 'p2_threshold_lock_v1',
                 'table_sha256': digest}).encode())[:24],
            'declared_seeds': list(table.get('declared_seeds', [])),
            'hypotheses': sorted(table['hypotheses']),
            'registered_utc': meta.get('registered_utc') if meta else None}
    atomic_json(out / 'thresholds.lock.json', lock)
    atomic_json(out / 'thresholds.json', table)
    return lock


def verify_thresholds(table, lock):
    """校验当前门限表与注册锁是否一致；返回差异列表。"""
    errors = []
    expected = threshold_table_hash(table)
    if lock.get('table_sha256') != expected:
        errors.append('门限表已被修改（table_sha256 不一致）')
    if sorted(table.get('hypotheses', {})) != list(lock.get('hypotheses', [])):
        errors.append('假设集合已改变')
    if list(table.get('declared_seeds', [])) != list(lock.get('declared_seeds', [])):
        errors.append('预声明种子已改变')
    return errors


def _gate_value(gate, metrics):
    if 'compare_to_metric' in gate:
        name = gate['compare_to_metric']
        if name not in metrics:
            return None
        return float(metrics[name])
    if 'pass_value' in gate:
        return float(gate['pass_value'])
    return None


def evaluate_gate(gate, metrics):
    """单条门槛判定：pass / falsified / inconclusive / incomplete_*。"""
    pre = gate.get('precondition')
    if pre:
        value = metrics.get(pre['metric'])
        if value is None:
            return 'incomplete_missing_metric'
        if not _OPS[pre['op']](float(value), float(pre['value'])):
            return pre.get('else_status', 'inconclusive_precondition_failed')
    value = metrics.get(gate['metric'])
    if value is None:
        return 'incomplete_missing_metric'
    value = float(value)
    bound = _gate_value(gate, metrics)
    if bound is None:
        return 'incomplete_missing_metric'
    if _OPS[gate['pass_op']](value, bound):
        return 'pass'
    if 'falsify_value' in gate:
        if _OPS[gate['falsify_op']](value, float(gate['falsify_value'])):
            return gate.get('on_fail', 'falsified')
    return gate.get('else', 'inconclusive')


def apply_threshold_table(table, observations, lock=None, declared_seeds=None):
    """把真实观测套入已注册的门限表；只输出判定，不做任何补偿。"""
    validate_threshold_table(table)
    declared = list(declared_seeds if declared_seeds is not None
                    else table.get('declared_seeds', []))
    result = {'protocol': 'p2_preregistered_threshold_apply_v1',
              'table_sha256': threshold_table_hash(table),
              'lock_id': (lock or {}).get('lock_id'),
              'declared_seeds': declared,
              'hypotheses': {},
              'observation_sha256': sha256_bytes(
                  canonical_json(observations).encode())}
    for name, spec in sorted(table['hypotheses'].items()):
        obs = observations.get(name)
        entry = {'statement': spec.get('statement'),
                 'source': spec.get('source'), 'status': None,
                 'per_baseline': None, 'gate_details': None,
                 'reported': None}
        if obs is None:
            entry['status'] = 'incomplete_missing_observation'
            result['hypotheses'][name] = entry
            continue
        seeds_present = obs.get('seeds_present')
        incomplete = None
        if spec.get('requires_seed_completeness', True):
            if seeds_present is None:
                incomplete = 'incomplete_seed_ledger'
            elif sorted(int(s) for s in seeds_present) != sorted(
                    int(s) for s in declared):
                incomplete = 'incomplete_missing_seeds'
        if spec.get('scope') == 'per_baseline':
            per = {}
            declared_b = spec.get('required_baselines') or sorted(
                obs.get('baselines', {}))
            for baseline in declared_b:
                b_obs = obs.get('baselines', {}).get(baseline)
                if b_obs is None:
                    per[baseline] = {'status': 'incomplete_missing_baseline'}
                    continue
                statuses = [evaluate_gate(g, b_obs.get('metrics', {}))
                            for g in spec['gates']]
                per[baseline] = {'status': _combine(statuses),
                                 'gate_details': statuses,
                                 'metrics': b_obs.get('metrics', {})}
            entry['per_baseline'] = per
            entry['status'] = incomplete or _combine(
                [v['status'] for v in per.values()])
        else:
            metrics = obs.get('metrics', {})
            statuses = [evaluate_gate(g, metrics) for g in spec['gates']]
            entry['gate_details'] = statuses
            entry['metrics'] = metrics
            entry['status'] = incomplete or _combine(statuses)
        reported = spec.get('report_only') or []
        entry['reported'] = {k: obs.get('metrics', {}).get(k)
                             for k in reported if k in obs.get('metrics', {})}
        result['hypotheses'][name] = entry
    result['all_pass'] = bool(
        result['hypotheses'] and
        all(v['status'] == 'pass' for v in result['hypotheses'].values()))
    result['statuses'] = {k: v['status'] for k, v in result['hypotheses'].items()}
    result['note'] = ('all_pass 仅在全部假设均为 pass 时为真；'
                      'inconclusive、falsified 与 incomplete_* 均不得写成达标')
    return result


# ------------------------------------------------------------------
# P2-3：独立时间窗声明与严格 rolling 污染审计
# ------------------------------------------------------------------
WINDOW_SCHEMA = 'p2_window_ledger_v1'


def classify_window(entry):
    """窗口污染分级：曾用于开发即不再独立。"""
    if entry.get('prior_exposure'):
        return 'contaminated_prior_exposure'
    if entry.get('used_for'):
        return 'contaminated_development_use'
    return 'held_out_independent'


def window_evidence_class(entry):
    cls = classify_window(entry)
    if cls == 'held_out_independent':
        return 'confirmatory_independent_window'
    return 'exploratory_reconfirmation_previously_observed'


def audit_window_disjointness(windows):
    """训练/校准/选择/评估窗口必须互不重叠；重叠即为污染。"""
    ranges = {}
    for name, entry in sorted(windows.items()):
        rng = entry.get('range')
        if not rng:
            continue
        ranges[name] = (int(rng[0]), int(rng[1]))
    violations = []
    names = sorted(ranges)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo = max(ranges[a][0], ranges[b][0])
            hi = min(ranges[a][1], ranges[b][1])
            if lo <= hi and windows[a].get('role') != windows[b].get('role'):
                violations.append({'a': a, 'b': b, 'overlap': [lo, hi]})
    return {'ranges': {k: list(v) for k, v in ranges.items()},
            'overlaps': violations, 'disjoint': not violations}


def declare_independent_window(ledger, start, end):
    """申请一段“未被污染”的独立时间窗；与已暴露区间重叠即拒绝。"""
    lo, hi = int(start), int(end)
    if lo > hi:
        raise ValueError('起始时间不得大于结束时间')
    conflicts = []
    for name, entry in sorted(ledger.get('windows', {}).items()):
        rng = entry.get('range')
        if not rng:
            continue
        if min(hi, int(rng[1])) >= max(lo, int(rng[0])):
            if classify_window(entry) != 'held_out_independent':
                conflicts.append({'window': name, 'range': list(rng),
                                  'classification': classify_window(entry)})
    ok = not conflicts
    return {'requested_range': [lo, hi],
            'independent_window_available': ok,
            'conflicts': conflicts,
            'evidence_class': ('confirmatory_independent_window' if ok else
                               'exploratory_reconfirmation_previously_observed'),
            'note': ('Elliptic 1–49 已被历史实验查看；若请求区间与已暴露区间重叠，'
                     '不得声称独立确认，只能称为探索性再确认')}


def audit_strict_rolling(raw, cutoff, train_end, label_ceiling=None,
                         recorded=None, tol=1e-6):
    """重算严格 rolling 视图并核对记录值：未来节点/边与统计量不得进入。

    raw 为 p0_data.load_elliptic_raw 的返回；recorded 为 P0-7 run.json 中的
    逐窗 window_audit 条目（可选）。
    """
    node_time = np.asarray(raw['node_time'])
    x = np.asarray(raw['x'])
    edge_index = np.asarray(raw['edge_index'])
    cutoff, train_end = int(cutoff), int(train_end)
    if train_end > cutoff:
        raise ValueError('训练截止点不得晚于当前滚动窗口 cutoff')
    fit_mask = node_time <= train_end
    view = strict_rolling_view(x, edge_index, node_time, cutoff, fit_mask)
    future_edge = (node_time[edge_index[0]] > cutoff) | \
                  (node_time[edge_index[1]] > cutoff)
    labels = np.asarray(raw['y'])
    visible_labels_beyond_ceiling = None
    if label_ceiling is not None:
        beyond = node_time > int(label_ceiling)
        visible_labels_beyond_ceiling = int(np.sum(labels[beyond] >= 0))
    audit = {'cutoff': cutoff, 'train_end': train_end,
             'visible_nodes': int(len(view['old_indices'])),
             'visible_edges': int(view['edge_index'].shape[1]),
             'future_nodes_excluded': int(np.sum(node_time > cutoff)),
             'edges_touching_future_excluded': int(np.sum(future_edge)),
             'scaler_fit_nodes': int(np.sum(fit_mask)),
             'scaler_fit_includes_future': bool(
                 np.any(node_time[fit_mask] > cutoff)),
             'max_visible_time': int(node_time[view['old_indices']].max()),
             'labels_visible_beyond_label_ceiling': visible_labels_beyond_ceiling,
             'scaler_mean_sha256': sha256_bytes(
                 np.ascontiguousarray(view['mean']).tobytes()),
             'scaler_scale_sha256': sha256_bytes(
                 np.ascontiguousarray(view['scale']).tobytes())}
    contamination = []
    if audit['max_visible_time'] > cutoff:
        contamination.append('视图包含 cutoff 之后的节点')
    if audit['scaler_fit_includes_future']:
        contamination.append('标准化拟合集包含 cutoff 之后的节点')
    if visible_labels_beyond_ceiling:
        contamination.append('标签截止点之后的标签仍然可见')
    mismatch = None
    if recorded:
        mismatch = {}
        for key, actual in (('visible_nodes', audit['visible_nodes']),
                            ('visible_edges', audit['visible_edges'])):
            if key in recorded and int(recorded[key]) != int(actual):
                mismatch[key] = {'recorded': recorded[key], 'recomputed': actual}
        if mismatch:
            contamination.append('记录窗口审计与重算不一致')
    audit['recorded_mismatch'] = mismatch
    audit['contamination_free'] = not contamination
    audit['contamination'] = contamination
    return audit


def scan_window_usage(scan_dirs, reserved_range, keys=None):
    """扫描实验目录中的 JSON 配置，检测是否触碰预留独立窗。"""
    keys = keys or ('train_start', 'train_end', 'val_start', 'val_end',
                    'eval_start', 'eval_end', 'time_start', 'time_end',
                    'cutoff', 'train_window', 'validation_window',
                    'evaluation', 'validation', 'range')
    lo, hi = int(reserved_range[0]), int(reserved_range[1])
    hits = []

    def walk(node, path, source):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in keys:
                    _check_value(v, f'{path}.{k}', source)
                walk(v, f'{path}.{k}', source)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f'{path}[{i}]', source)

    def _check_value(v, path, source):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            ints = [int(v)]
        elif isinstance(v, list) and v and all(
                isinstance(e, (int, float)) and not isinstance(e, bool)
                for e in v):
            ints = [int(e) for e in v]
        else:
            return
        if len(ints) == 1:
            if lo <= ints[0] <= hi:
                hits.append({'source': source, 'path': path, 'value': ints[0]})
        else:
            if min(hi, max(ints)) >= max(lo, min(ints)):
                hits.append({'source': source, 'path': path, 'value': ints})

    for root in scan_dirs:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob('*.json')):
            try:
                obj = read_json(path)
            except Exception:
                continue
            walk(obj, '', str(path))
    return {'reserved_range': [lo, hi], 'hits': hits, 'clean': not hits}


def check_prediction_contamination(pred_paths, raw, reserved_range):
    """冻结预测若覆盖预留窗节点，说明该窗已参与开发，不再独立。"""
    time_by_id = {int(t): int(tm) for t, tm in
                  zip(np.asarray(raw['tx_ids']).reshape(-1),
                      np.asarray(raw['node_time']).reshape(-1))}
    lo, hi = int(reserved_range[0]), int(reserved_range[1])
    rows = []
    for path in pred_paths:
        path = Path(path)
        if not path.exists():
            rows.append({'path': str(path), 'status': 'missing'})
            continue
        with np.load(path, allow_pickle=False) as z:
            ids = np.asarray(z['node_id']).reshape(-1)
        times = np.array([time_by_id.get(int(i), -1) for i in ids])
        touched = int(np.sum((times >= lo) & (times <= hi)))
        rows.append({'path': str(path), 'n_predicted': int(len(ids)),
                     'n_in_reserved_window': touched,
                     'clean': touched == 0})
    return {'reserved_range': [lo, hi], 'predictions': rows,
            'clean': all(r.get('clean') for r in rows)}


# ------------------------------------------------------------------
# P2-4：基线公平超参搜索预算审计
# ------------------------------------------------------------------
def hpo_budget_spec_hash(spec):
    return sha256_bytes(canonical_json(spec).encode())


def audit_hpo_fairness(winners, spec, runs_dir=None):
    """等预算 HPO 审计：每个模型的试验数、种子、窗口与选择规则必须一致。

    任何模型获得额外预算、额外种子或不同选择规则都会使公平对比失效。
    """
    violations = []
    per_model = {}
    declared_models = list(spec.get('models', []))
    for name in declared_models:
        entry = (winners.get('models') or {}).get(name)
        if entry is None:
            violations.append(f'缺模型: {name}')
            per_model[name] = {'present': False}
            continue
        seeds = [int(s) for s in entry.get('seeds', [])]
        trials = int(entry.get('trials_per_seed', -1))
        declared_seeds = [int(s) for s in spec.get('seeds', [])]
        if trials != int(spec.get('trials_per_seed', -1)):
            violations.append(f'{name}: 试验预算 {trials} != 预注册 '
                              f'{spec.get("trials_per_seed")}')
        missing = [s for s in declared_seeds if s not in seeds]
        if missing:
            violations.append(f'{name}: 缺预注册种子 {missing}')
        if entry.get('selection') != spec.get('selection'):
            violations.append(f'{name}: 选择规则与预注册不一致')
        by_seed = entry.get('by_seed', {})
        for seed, row in sorted(by_seed.items()):
            if row.get('train_window') != spec.get('train_window'):
                violations.append(f'{name}/seed{seed}: 训练窗口不一致')
            if row.get('validation_window') != spec.get('validation_window'):
                violations.append(f'{name}/seed{seed}: 验证窗口不一致')
            if row.get('test_labels_used'):
                violations.append(f'{name}/seed{seed}: HPO 使用了测试标签')
        per_model[name] = {'present': True, 'trials_per_seed': trials,
                           'n_seeds': len(seeds),
                           'search_cost_runs': trials * len(seeds),
                           'selection': entry.get('selection'),
                           'missing_seeds': missing}
    costs = {v['search_cost_runs'] for v in per_model.values()
             if v.get('present')}
    if len(costs) > 1:
        violations.append(f'搜索预算不相等: {sorted(costs)}')
    extra_runs = []
    if runs_dir and Path(runs_dir).exists():
        for name in declared_models:
            budget = int(spec.get('trials_per_seed', 0))
            declared_seeds = {int(s) for s in spec.get('seeds', [])}
            for path in sorted(Path(runs_dir).glob(f'{name}_trial*_seed*.json')):
                stem = path.stem
                try:
                    trial = int(stem.split('_trial')[1].split('_')[0])
                    seed = int(stem.split('_seed')[1])
                except (IndexError, ValueError):
                    extra_runs.append({'file': path.name, 'reason': 'unparsable'})
                    continue
                if trial >= budget or seed not in declared_seeds:
                    extra_runs.append({'file': path.name, 'trial': trial,
                                       'seed': seed, 'reason': 'outside_declared_budget'})
        if extra_runs:
            violations.append(f'存在预注册预算之外的运行: {len(extra_runs)} 个')
    return {'protocol': 'p2_hpo_fairness_audit_v1',
            'spec_sha256': hpo_budget_spec_hash(spec),
            'fair': not violations,
            'violations': violations,
            'per_model': per_model,
            'extra_runs': extra_runs,
            'note': '不相等预算下的精度对比不能称为公平基线比较'}


# ------------------------------------------------------------------
# P2-5：正类 ECE 与分层校准
# ------------------------------------------------------------------
def _bin_index(p, bins=15):
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    return np.clip(np.digitize(p, edges[1:-1], right=False), 0, int(bins) - 1)


def _ece_parts(y, p, bins=15, min_count=1):
    """返回 ECE、MCE、最小桶计数过滤后的 ECE 与逐桶明细。"""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    idx = _bin_index(p, bins)
    n = len(y)
    ece = 0.0
    mce = 0.0
    filtered = 0.0
    covered = 0
    rows = []
    for b in range(int(bins)):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            rows.append({'bin': b, 'count': 0, 'mean_confidence': None,
                         'accuracy': None, 'gap': None,
                         'meets_min_count': False})
            continue
        conf = float(np.mean(p[mask]))
        acc = float(np.mean(y[mask]))
        gap = abs(acc - conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        ok = count >= int(min_count)
        if ok:
            filtered += (count / n) * gap
            covered += count
        rows.append({'bin': b, 'count': count, 'mean_confidence': conf,
                     'accuracy': acc, 'gap': gap, 'meets_min_count': bool(ok)})
    coverage = covered / n if n else 0.0
    return {'ece': float(ece), 'mce': float(mce),
            'ece_min_count_filtered': (float(filtered / coverage)
                                       if coverage > 0 else None),
            'filtered_coverage': float(coverage),
            'nonempty_bins': int(sum(1 for r in rows if r['count'] > 0)),
            'rows': rows, 'n': int(n)}


def _class_weighted_ece(y, p, target_class, bins=15, min_count=1):
    """按目标类别的质量加权 ECE；桶仍按预测概率等宽划分。

    正类加权 ECE = Σ_b (n_b+ / N+) · |pos_rate_b − mean(p_b)|。
    与“只在正类样本上分桶”的字面定义不同：后者中正类准确率恒为 1，
    会退化成 1 − mean_{pos}(p)，无法反映校准形状，因此只作为对照报告。
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    idx = _bin_index(p, bins)
    positive_target = int(target_class) == 1
    cls_mask = y == int(target_class)
    n_cls = int(cls_mask.sum())
    if n_cls == 0:
        raise ValueError(f'类别 {target_class} 没有样本，无法计算该类加权 ECE')
    weighted = 0.0
    filtered = 0.0
    covered = 0
    mce = 0.0
    rows = []
    for b in range(int(bins)):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            rows.append({'bin': b, 'count': 0, 'n_target': 0,
                         'mean_confidence': None, 'target_rate': None,
                         'gap': None, 'class_weight': None,
                         'meets_min_count': False})
            continue
        n_target = int(np.sum(y[mask] == int(target_class)))
        mean_p = float(np.mean(p[mask]))
        target_rate = float(n_target / n_b)
        confidence = mean_p if positive_target else 1.0 - mean_p
        gap = abs(target_rate - confidence)
        weight = n_target / n_cls
        weighted += weight * gap
        mce = max(mce, gap)
        ok = n_b >= int(min_count)
        if ok:
            filtered += weight * gap
            covered += n_target
        rows.append({'bin': b, 'count': n_b, 'n_target': n_target,
                     'mean_confidence': confidence,
                     'target_rate': target_rate, 'gap': gap,
                     'class_weight': float(weight),
                     'meets_min_count': bool(ok)})
    coverage = covered / n_cls if n_cls else 0.0
    subset_mean_p = float(np.mean(p[cls_mask]))
    return {'weighted_ece': float(weighted),
            'weighted_ece_min_count_filtered': (float(filtered / coverage)
                                                if coverage > 0 else None),
            'filtered_class_coverage': float(coverage),
            'weighted_mce': float(mce),
            'n_class': n_cls,
            'nonempty_bins': int(sum(1 for r in rows if r['count'] > 0)),
            'class_subset_ece': (1.0 - subset_mean_p if positive_target
                                 else subset_mean_p),
            'class_subset_ece_is_degenerate': True,
            'mean_p_on_class': subset_mean_p,
            'rows': rows}


def positive_class_ece(y, p, bins=15, min_count=1):
    """正类校准误差：主指标为正类加权 ECE，字面子集 ECE 仅作对照。"""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) != len(p):
        raise ValueError('标签与概率长度不一致')
    if int(np.sum(y == 1)) == 0:
        raise ValueError('正类 ECE 需要至少一个正类样本')
    res = _class_weighted_ece(y, p, 1, bins, min_count)
    return {'positive_class_ece': res['weighted_ece'],
            'positive_class_ece_min_count_filtered':
                res['weighted_ece_min_count_filtered'],
            'filtered_positive_coverage': res['filtered_class_coverage'],
            'positive_class_mce': res['weighted_mce'],
            'positive_class_subset_ece': res['class_subset_ece'],
            'positive_class_subset_ece_is_degenerate': True,
            'mean_p_on_positives': res['mean_p_on_class'],
            'n_positive': res['n_class'],
            'nonempty_bins': res['nonempty_bins'],
            'degenerate_single_bin': bool(res['nonempty_bins'] <= 1),
            'definition': 'sum_b (n_b+/N+) * |positive_rate_b - mean(p_b)|',
            'definition_subset': '字面正类子集定义 = 1 - mean_{pos}(p)，已退化，仅作对照',
            'reliability_positive': res['rows']}


def negative_class_ece(y, p, bins=15, min_count=1):
    """负类校准误差：按负类质量加权，置信度取 1 − p。"""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if int(np.sum(y == 0)) == 0:
        raise ValueError('负类 ECE 需要至少一个负类样本')
    res = _class_weighted_ece(y, p, 0, bins, min_count)
    return {'negative_class_ece': res['weighted_ece'],
            'negative_class_ece_min_count_filtered':
                res['weighted_ece_min_count_filtered'],
            'filtered_negative_coverage': res['filtered_class_coverage'],
            'negative_class_mce': res['weighted_mce'],
            'negative_class_subset_ece': res['class_subset_ece'],
            'mean_p_on_negatives': res['mean_p_on_class'],
            'n_negative': res['n_class'],
            'nonempty_bins': res['nonempty_bins'],
            'definition': 'sum_b (n_b-/N-) * |negative_rate_b - mean(1-p_b)|',
            'reliability_negative': res['rows']}


def stratified_ece(y, p, strata, bins=15, min_stratum_n=100, min_count=1):
    """逐层 ECE：每层独立分桶，样本不足的层必须标记而非静默忽略。"""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    strata = np.asarray(strata)
    if len(strata) != len(y):
        raise ValueError('分层变量与标签长度不一致')
    rows = []
    for value in sorted(set(strata.tolist()), key=lambda v: (str(type(v)), v)):
        mask = strata == value
        n = int(mask.sum())
        n_pos = int(np.sum(y[mask] == 1))
        insufficient = n < int(min_stratum_n) or n_pos == 0 or n_pos == n
        if insufficient:
            rows.append({'stratum': value, 'n': n, 'n_positive': n_pos,
                         'ece': None, 'brier': None, 'nll': None,
                         'insufficient': True,
                         'reason': ('样本不足或缺少某一类，不计算层内 ECE'
                                    if n < int(min_stratum_n) else
                                    '层内只有一个类别，ECE 不可解释')})
            continue
        parts = _ece_parts(y[mask], p[mask], bins, min_count)
        prob = np.clip(p[mask], 1e-7, 1 - 1e-7)
        rows.append({'stratum': value, 'n': n, 'n_positive': n_pos,
                     'positive_rate': float(n_pos / n),
                     'ece': parts['ece'], 'mce': parts['mce'],
                     'brier': float(np.mean((p[mask] - y[mask]) ** 2)),
                     'nll': float(-np.mean(y[mask] * np.log(prob) +
                                           (1 - y[mask]) * np.log(1 - prob))),
                     'nonempty_bins': parts['nonempty_bins'],
                     'insufficient': False})
    valid = [r for r in rows if not r['insufficient']]
    eces = [r['ece'] for r in valid]
    return {'rows': rows,
            'n_strata': len(rows),
            'n_strata_insufficient': int(sum(1 for r in rows if r['insufficient'])),
            'stratified_ece_max': float(max(eces)) if eces else None,
            'stratified_ece_mean': float(np.mean(eces)) if eces else None,
            'min_stratum_n': int(min_stratum_n),
            'note': ('层内分桶与整体分桶不同；层样本不足时该层不参与 max/mean，'
                     '但必须原样列出，不得删除后声称全部达标')}


def calibration_quality_report(y, p, threshold=0.5, bins=15, min_count=1,
                               strata=None, min_stratum_n=100):
    """一次汇总：整体、正类、负类、分层与决策阈值处的校准质量。"""
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    base = probability_metrics(y, p, threshold=threshold, bins=bins)
    pos = positive_class_ece(y, p, bins, min_count)
    neg = negative_class_ece(y, p, bins, min_count)
    out = {'protocol': 'p2_calibration_quality_v1',
           'n': int(len(y)), 'bins': int(bins), 'min_count': int(min_count),
           'threshold': float(threshold),
           'ece_overall': base['ece'], 'brier': base['brier'],
           'nll': base['nll'], 'f1': base['f1'],
           'positive_class_ece': pos['positive_class_ece'],
            'positive_class_mce': pos['positive_class_mce'],
            'positive_class_ece_min_count_filtered':
                pos['positive_class_ece_min_count_filtered'],
            'positive_class_subset_ece': pos['positive_class_subset_ece'],
            'mean_p_on_positives': pos['mean_p_on_positives'],
            'positive_class_degenerate_single_bin': pos['degenerate_single_bin'],
            'n_positive': pos['n_positive'],
            'negative_class_ece': neg['negative_class_ece'],
            'mean_p_on_negatives': neg['mean_p_on_negatives'],
            'n_negative': neg['n_negative'],
            'sce_positive_negative_weighted': 0.5 * (
                pos['positive_class_ece'] + neg['negative_class_ece']),
            'binary_note': ('二分类下 Kumar 等 2019 的 classwise-ECE 与整体 ECE 数值相同，'
                            '因此本模块改为报告按类别质量加权的正/负类 ECE'),
           'overall_ece_parts': _ece_parts(y, p, bins, min_count),
           'reliability_positive': pos['reliability_positive'],
           'reliability_negative': neg['reliability_negative'],
           'strata': None}
    if strata:
        out['strata'] = {name: stratified_ece(y, p, values, bins,
                                              min_stratum_n, min_count)
                         for name, values in sorted(strata.items())}
    return out


def fit_stratified_temperature(logits, y, strata, min_pos=20, min_neg=20):
    """分层温度缩放：每层独立拟合单温度，样本不足则回退全局并显式标记。

    二分类下 Kumar 等 2019 的 classwise-ECE 与整体 ECE 数值相同，因此本
    模块额外报告按类别子集计算的正/负类 ECE，而不是重复整体值。
    """
    y = np.asarray(y, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    strata = np.asarray(strata)
    if len(y) != len(logits) or len(strata) != len(y):
        raise ValueError('logits、标签与分层长度不一致')
    if len(np.unique(y)) < 2:
        raise ValueError('分层温度拟合需要两个类别')
    global_cal = fit_calibrator('temperature', logits, y)
    per = {}
    for value in sorted(set(strata.tolist()), key=lambda v: (str(type(v)), v)):
        mask = strata == value
        n = int(mask.sum())
        n_pos = int(np.sum(y[mask] == 1))
        n_neg = n - n_pos
        fallback = (n_pos < int(min_pos) or n_neg < int(min_neg))
        if fallback:
            per[str(value)] = {'n': n, 'n_positive': n_pos, 'n_negative': n_neg,
                               'fallback_to_global': True,
                               'log_temperature': global_cal['log_temperature'],
                               'reason': '层内样本不足以独立拟合温度'}
            continue
        cal = fit_calibrator('temperature', logits[mask], y[mask])
        per[str(value)] = {'n': n, 'n_positive': n_pos, 'n_negative': n_neg,
                           'fallback_to_global': False,
                           'log_temperature': cal['log_temperature'],
                           'fit_nll': cal.get('fit_nll')}
    return {'protocol': 'p2_stratified_temperature_v1',
            'method': 'per_stratum_temperature_with_global_fallback',
            'global': global_cal, 'per_stratum': per,
            'min_pos': int(min_pos), 'min_neg': int(min_neg),
            'n_strata_fallback': int(sum(1 for v in per.values()
                                         if v['fallback_to_global']))}


def apply_stratified_temperature(calibrator, logits, strata):
    """按层应用温度；未知层回退全局温度。"""
    z = np.asarray(logits, dtype=np.float64)
    strata = np.asarray(strata)
    out = 1.0 / (1.0 + np.exp(-(z / np.exp(calibrator['global']['log_temperature']))))
    for key, entry in sorted(calibrator['per_stratum'].items()):
        mask = np.asarray([str(v) == key for v in strata.tolist()])
        if not np.any(mask):
            continue
        t = np.exp(entry['log_temperature'])
        out[mask] = 1.0 / (1.0 + np.exp(-(z[mask] / t)))
    return out


def monotonicity_report(raw_p, new_p):
    """校准变换应保持单调，否则会改变阈值排序并影响 F1 的可比性。"""
    a = np.asarray(raw_p, dtype=np.float64)
    b = np.asarray(new_p, dtype=np.float64)
    order_a = np.argsort(a, kind='stable')
    violations = int(np.sum(np.diff(b[order_a]) < -1e-12))
    rho = float(stats.spearmanr(a, b).statistic)
    return {'monotonicity_violations': violations,
            'spearman_correlation': rho,
            'rank_preserved': violations == 0,
            'note': '非单调校准会改变决策排序，F1 不再与校准前可比'}
