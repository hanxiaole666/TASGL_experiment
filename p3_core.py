"""p3_core.py — P3-1 至 P3-6 真实证据原语。

本模块把“程序已有但从未产出真实结果”的六项缺口串成可审计链条：
独立划分声明与锁定评估审计、十种子聚合、等预算公平结论、跨链语义核验与
迁移、逐层逐种子逐时间窗稀疏与延迟审计、H1–H5 观测收集与判定。

所有函数在数据缺失时返回 incomplete_* / pending_*，不虚构任何数值。
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from p0_core import atomic_json, canonical_json, file_sha256, read_json, \
    sha256_bytes
from p2_core import (apply_threshold_table, audit_hpo_fairness,
                     multi_comparison, paired_statistics_multi)
from p3_data import audit_official_split, split_fingerprint


# ------------------------------------------------------------------
# 通用工具
# ------------------------------------------------------------------
def _find_key(node, key):
    """在嵌套 JSON 中递归查找键；找不到返回 None（不猜测）。"""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def _mean_sd_ci(values, alpha=0.05):
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    n = len(v)
    if n == 0:
        return {'mean': None, 'sample_sd': None, 'ci95': None, 'n': 0}
    mean = float(np.mean(v))
    if n < 2:
        return {'mean': mean, 'sample_sd': None, 'ci95': None, 'n': n}
    sd = float(np.std(v, ddof=1))
    half = float(stats.t.ppf(1 - alpha / 2, n - 1) * sd / math.sqrt(n))
    return {'mean': mean, 'sample_sd': sd, 'ci95': [mean - half, mean + half],
            'n': n}


def _evaluation_mean(evaluation, metric='f1'):
    """优先取锁后评估的聚合值，否则从逐种子重算；两者不一致即报错。"""
    agg = evaluation.get('aggregate') or {}
    if metric in agg and agg[metric].get('mean') is not None:
        return float(agg[metric]['mean'])
    rows = evaluation.get('per_seed') or []
    values = [float(r[metric]) for r in rows if metric in r]
    if not values:
        return None
    return float(np.mean(values))


def _evaluation_seeds(evaluation):
    rows = evaluation.get('per_seed') or []
    return sorted(int(r['seed']) for r in rows)


def _read(path):
    """路径为空、不存在或不是文件时返回 None；不得回退到当前目录。"""
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return read_json(path)


# ------------------------------------------------------------------
# P3-1：未参与开发的独立数据／划分上的真实锁定评估
# ------------------------------------------------------------------
def declare_independent_split(dataset, root, raw, prior_exposure,
                              declared_by, evidence=None, file_hashes=None):
    """在读取最终测试指标之前声明独立划分，并绑定数据与划分指纹。

    prior_exposure=True 时证据等级只能是探索性再确认。
    """
    if not isinstance(prior_exposure, bool):
        raise ValueError('prior_exposure 必须显式声明 true/false')
    audit = raw.get('split_audit') or audit_official_split(raw)
    payload = {'schema': 'p3_independent_split_declaration_v1',
               'dataset': dataset, 'root': str(Path(root).resolve()),
               'prior_exposure': prior_exposure,
               'declared_by': declared_by,
               'split_fingerprint': split_fingerprint(raw),
               'file_sha256': file_hashes or {},
               'mask_kind': raw.get('mask_kind'),
               'split_audit': audit}
    declaration = {**payload,
                   'declaration_sha256': sha256_bytes(
                       canonical_json(payload).encode()),
                   'evidence_class': ('exploratory_reconfirmation_previously_observed'
                                      if prior_exposure else
                                      'confirmatory_independent_split'),
                   'evidence': evidence or []}
    return declaration


def audit_independent_evaluation(lock, evaluation, declared_seeds,
                                 require_confirmatory=False):
    """审计一次锁后评估是否真的配得上“独立确认”这一说法。"""
    errors = []
    if lock.get('lock_id') != evaluation.get('lock_id'):
        errors.append('评估的 lock_id 与锁不一致')
    if evaluation.get('test_labels_used_by_this_pipeline_before_lock') is not False:
        errors.append('评估未声明锁前未使用测试标签')
    present = _evaluation_seeds(evaluation)
    declared = sorted(int(s) for s in declared_seeds)
    missing = [s for s in declared if s not in present]
    extra = [s for s in present if s not in declared]
    if missing:
        errors.append(f'缺预声明种子: {missing}')
    if extra:
        errors.append(f'存在预声明之外的种子: {extra}')
    per_seed = evaluation.get('per_seed') or []
    if len(per_seed) != len(set(r.get('seed') for r in per_seed)):
        errors.append('逐种子记录存在重复种子')
    prior = evaluation.get('prior_test_exposure')
    if prior is None:
        errors.append('评估未声明 prior_test_exposure')
    evidence_class = evaluation.get('evidence_class')
    status = ('incomplete_missing_seeds' if missing or extra else
              'complete')
    if errors and status == 'complete':
        status = 'incomplete_audit_failed'
    confirmatory = (status == 'complete' and prior is False
                    and evidence_class ==
                    'locked_independent_evaluation_by_preregistration')
    result = {'protocol': 'p3_independent_evaluation_audit_v1',
              'lock_id': lock.get('lock_id'),
              'declared_seeds': declared, 'present_seeds': present,
              'missing_seeds': missing, 'extra_seeds': extra,
              'prior_test_exposure': prior,
              'evidence_class': evidence_class,
              'errors': errors, 'status': status,
              'confirmatory_independent': bool(confirmatory),
              'classification': ('confirmatory_independent' if confirmatory else
                                 'exploratory_reconfirmation_previously_observed'
                                 if prior else
                                 'incomplete_or_unclassified'),
              'blocked_by_requirement': bool(require_confirmatory
                                             and not confirmatory)}
    return result


# ------------------------------------------------------------------
# P3-2：DGraph-Fin 十种子的实际训练与指标
# ------------------------------------------------------------------
def aggregate_locked_evaluation(evaluation, declared_seeds,
                                metrics=('f1', 'pr_auc', 'average_precision',
                                         'roc_auc', 'ece', 'brier', 'nll'),
                                alpha=0.05):
    """把锁后评估聚合成多种子指标表；缺种子即 incomplete_missing_seeds。"""
    rows = {int(r['seed']): r for r in (evaluation.get('per_seed') or [])}
    declared = sorted(int(s) for s in declared_seeds)
    missing = [s for s in declared if s not in rows]
    extra = sorted(set(rows) - set(declared))
    out = {'protocol': 'p3_locked_evaluation_aggregate_v1',
           'declared_seeds': declared, 'missing_seeds': missing,
           'extra_seeds': extra, 'n_declared': len(declared),
           'n_available': len(declared) - len(missing),
           'alpha': alpha, 'sample_sd_ddof': 1,
           'status': ('pass_complete' if not missing and not extra else
                      'incomplete_missing_seeds'),
           'metrics': {}, 'per_seed': [rows[s] for s in declared if s in rows]}
    for metric in metrics:
        values = [float(rows[s][metric]) for s in declared
                  if s in rows and metric in rows[s]]
        out['metrics'][metric] = _mean_sd_ci(values, alpha)
    out['note'] = ('缺种子时不得用可用子集冒充十种子结论；'
                   '样本 SD 用 ddof=1，置信区间为双侧 t 95%')
    return out


# ------------------------------------------------------------------
# P3-3：等预算 HPO 后的公平精度、效率与统计结论
# ------------------------------------------------------------------
def pareto_frontier(models, cost_key, f1_key='f1_mean'):
    """成本—精度支配判定；相同成本时只保留更高精度，结果不依赖输入顺序。"""
    rows = [m for m in models if m.get(cost_key) is not None
            and m.get(f1_key) is not None]
    frontier = []
    for a in rows:
        dominated = False
        for b in rows:
            if b is a:
                continue
            cost_le = b[cost_key] <= a[cost_key]
            f1_ge = b[f1_key] >= a[f1_key]
            strict = b[cost_key] < a[cost_key] or b[f1_key] > a[f1_key]
            if cost_le and f1_ge and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(a['name'])
    return {'cost_key': cost_key, 'frontier': sorted(frontier),
            'n_models': len(rows),
            'reference_on_frontier': None}


def fair_comparison(hpo_winners, spec, evaluations, efficiency=None,
                    reference='tasgl', declared_seeds=None, alpha=0.05,
                    hpo_dir=None):
    """把等预算公平性、精度、效率与配对统计串成一条可发表结论。

    任一步骤不完整（不公平预算、缺种子、缺评估）都会阻断结论，而不是给出
    一个看似完整的表格。
    """
    fairness = audit_hpo_fairness(hpo_winners, spec,
                                  str(Path(hpo_dir) / 'runs') if hpo_dir else None)
    declared = sorted(int(s) for s in
                      (declared_seeds or spec.get('seeds') or []))
    blockers = []
    if not fairness['fair']:
        blockers.append('hpo_budget_unfair')
    per_model = {}
    for name, path in sorted(evaluations.items()):
        evaluation = _read(path)
        if evaluation is None:
            per_model[name] = {'present': False,
                               'status': 'incomplete_missing_evaluation'}
            blockers.append(f'missing_evaluation:{name}')
            continue
        agg = aggregate_locked_evaluation(evaluation, declared)
        per_model[name] = {'present': True, 'status': agg['status'],
                           'f1_mean': agg['metrics']['f1']['mean'],
                           'f1_sample_sd': agg['metrics']['f1']['sample_sd'],
                           'f1_ci95': agg['metrics']['f1']['ci95'],
                           'pr_auc_mean': agg['metrics'].get('pr_auc', {}).get('mean'),
                           'ece_mean': agg['metrics'].get('ece', {}).get('mean'),
                           'n_seeds': agg['n_available'],
                           'params': None, 'latency_ms': None}
        if agg['status'] != 'pass_complete':
            blockers.append(f'incomplete_seeds:{name}')
    # 参数量：优先取 HPO winner 记录
    for name, row in per_model.items():
        winner = (hpo_winners.get('models') or {}).get(name)
        if winner and row['present']:
            by_seed = winner.get('by_seed') or {}
            params = [int(v.get('params')) for v in by_seed.values()
                      if v.get('params') is not None]
            row['params'] = int(np.mean(params)) if params else None
    if efficiency:
        for name, row in per_model.items():
            entry = efficiency.get(name) or {}
            row['latency_ms'] = entry.get('mean_latency_ms')
            if row['params'] is None:
                row['params'] = entry.get('params')
    rows = [{'name': n, 'f1_mean': r.get('f1_mean'),
             'params': r.get('params'), 'latency_ms': r.get('latency_ms')}
            for n, r in per_model.items() if r.get('present')]
    pareto = {}
    for cost_key in ('params', 'latency_ms'):
        usable = [r for r in rows if r.get(cost_key) is not None
                  and r.get('f1_mean') is not None]
        if len(usable) < 2:
            pareto[cost_key] = {'frontier': None, 'n_models': len(usable),
                                'status': 'incomplete_insufficient_models'}
            blockers.append(f'insufficient_efficiency_data:{cost_key}')
            continue
        block = pareto_frontier(usable, cost_key)
        block['reference_on_frontier'] = reference in block['frontier']
        block['status'] = 'complete'
        pareto[cost_key] = block
    # 配对统计：参考模型 vs 其余模型
    comparisons = []
    ref_eval = _read(evaluations.get(reference, '')) if reference in evaluations else None
    for name in sorted(evaluations):
        if name == reference:
            continue
        other = _read(evaluations[name])
        if ref_eval is None or other is None:
            comparisons.append({'name': f'{reference}_vs_{name}',
                                'status': 'incomplete_missing_evaluation',
                                'p_value': None, 'p_holm': None})
            continue
        stat = paired_statistics_multi(ref_eval.get('per_seed'),
                                       other.get('per_seed'),
                                       declared, 'f1', alpha)
        stat['name'] = f'{reference}_vs_{name}'
        comparisons.append(stat)
    valid = [i for i, c in enumerate(comparisons) if c.get('p_value') is not None]
    if valid:
        adjusted, flags = multi_comparison(
            [comparisons[i]['p_value'] for i in valid], alpha)
        for i, adj, flag in zip(valid, adjusted, flags):
            comparisons[i]['p_holm'] = adj
            comparisons[i]['significant_after_holm'] = flag
    for i in set(range(len(comparisons))) - set(valid):
        comparisons[i]['p_holm'] = None
        comparisons[i]['significant_after_holm'] = False
    conclusions_allowed = not blockers
    return {'protocol': 'p3_fair_comparison_v1',
            'reference_model': reference,
            'declared_seeds': declared, 'alpha': alpha,
            'fairness': fairness,
            'per_model': per_model,
            'pareto': pareto,
            'pairwise_statistics': comparisons,
            'blockers': blockers,
            'conclusions_allowed': conclusions_allowed,
            'note': ('任一 blocker 存在时不得把表格写成“公平对比结论”；'
                     '不显著与不完整都必须原样报告')}


# ------------------------------------------------------------------
# P3-4：真实共享语义存在时的跨链迁移
# ------------------------------------------------------------------
def check_semantic_mapping(mapping, source_columns=None, target_columns=None):
    """核验跨链语义映射是否真实可核验；没有已核验语义则科学上被阻塞。"""
    if not isinstance(mapping, dict) or not mapping:
        return {'status': 'scientifically_blocked_missing_mapping',
                'usable': False, 'errors': ['映射为空或不是对象']}
    errors = []
    semantics = mapping.get('semantics') or mapping.get('canonical_features')
    if not semantics:
        errors.append('缺少 canonical 共享语义定义')
    label_mapping = mapping.get('label_mapping')
    if not label_mapping:
        errors.append('缺少源／目标标签映射')
    graph_rules = mapping.get('graph_rules')
    if not graph_rules:
        errors.append('缺少节点／边语义与方向、自环、去重规则')
    mode = mapping.get('mode')
    if mode not in ('zero_shot', 'few_shot'):
        errors.append('mode 必须是 zero_shot 或 few_shot')
    if source_columns and semantics:
        missing = [name for name in semantics
                   if semantics[name].get('source') not in source_columns]
        if missing:
            errors.append(f'源列不存在的语义: {missing}')
    if target_columns and semantics:
        missing = [name for name in semantics
                   if semantics[name].get('target') not in target_columns]
        if missing:
            errors.append(f'目标列不存在的语义: {missing}')
    verified = bool(mapping.get('semantics_verified'))
    if not verified:
        errors.append('共享语义未标注为已独立核验')
    else:
        for field in ('verified_by', 'evidence_uri'):
            if not mapping.get(field):
                errors.append(f'semantics_verified=true 但缺少 {field}')
    status = ('ready' if not errors else
              'scientifically_blocked_unverified_semantics')
    return {'status': status, 'usable': not errors, 'errors': errors,
            'n_canonical_features': len(semantics) if semantics else 0,
            'mode': mode,
            'note': ('独立 PCA 或相同维数不等于语义对齐；'
                     '缺少已核验的逐列语义时，跨链结论只能声明为被阻塞')}


def fit_source_classifier(x_source, y_source, hidden=64, epochs=30, lr=1e-3,
                          seed=0, device=None, positive_weight=None):
    """在源域共享语义特征上训练小型 MLP；不使用任何目标域标签。"""
    device = device or torch.device('cpu')
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    xs = np.asarray(x_source, dtype=np.float32)
    ys = np.asarray(y_source, dtype=np.int64)
    labeled = ys >= 0
    if labeled.sum() < 2 or len(np.unique(ys[labeled])) < 2:
        raise ValueError('源域标注样本不足或只有一个类别')
    model = torch.nn.Sequential(
        torch.nn.Linear(xs.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Dropout(0.3), torch.nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    idx = np.flatnonzero(labeled)
    pos = max(int((ys[idx] == 1).sum()), 1)
    neg = max(int((ys[idx] == 0).sum()), 1)
    pw = torch.tensor([neg / pos], dtype=torch.float32, device=device)
    rng = np.random.RandomState(int(seed))
    x_tensor = torch.as_tensor(xs[idx], device=device)
    y_tensor = torch.as_tensor(ys[idx], dtype=torch.float32, device=device)
    model.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(idx))
        for start in range(0, len(order), 1024):
            batch = torch.as_tensor(order[start:start + 1024], device=device)
            logits = model(x_tensor[batch]).reshape(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, y_tensor[batch], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_target(model, x_target, device=None, batch_size=4096):
    """在目标域共享语义特征上输出正类概率；不接触目标标签。"""
    device = device or torch.device('cpu')
    xt = torch.as_tensor(np.asarray(x_target, dtype=np.float32), device=device)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(xt), int(batch_size)):
            out.append(torch.sigmoid(model(xt[start:start + int(batch_size)])
                                     .reshape(-1)).detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)


# ------------------------------------------------------------------
# P3-5：逐层、逐种子、逐时间窗的稀疏与延迟审计
# ------------------------------------------------------------------
def _sync(device):
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.synchronize()


def _timed(fn, device, repeats, warmup):
    for _ in range(int(warmup)):
        out = fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(int(repeats)):
        out = fn()
    _sync(device)
    elapsed = (time.perf_counter() - start) / max(int(repeats), 1) * 1000.0
    return float(elapsed), out


def layerwise_sparsity_audit(model, data, device, repeats=20, warmup=5,
                             hard_mask=True):
    """逐层审计：理论活跃边、实际执行边、门控统计与逐层延迟。

    P0-9 只给全模型延迟与两层门控汇总；这里补逐层延迟与门控分布，
    并把 RFF 多跳“未门控、始终执行全部边”的事实按层列出。
    """
    device = torch.device(device)
    x = data['x']
    edge_index = data['edge_index']
    dt = data['time_gaps']
    num_nodes = data['num_nodes']
    was_training = model.training
    model.eval()
    for module in (model.prop, model.attn1, model.attn2):
        module.audit_enabled = True
    n_edges = int(edge_index.shape[1])
    rows = []
    with torch.no_grad():
        latency, h0 = _timed(lambda: F.relu(model.input_proj(x)), device,
                             repeats, warmup)
        rows.append({'layer': 'input_proj', 'kind': 'linear', 'gated': False,
                     'input_edges': n_edges, 'active_edges': None,
                     'executed_edges': None, 'active_ratio': None,
                     'executed_ratio': None, 'latency_ms': latency,
                     'params': int(sum(p.numel() for p in
                                       model.input_proj.parameters()))})
        latency, hp = _timed(lambda: model.prop(h0, edge_index, num_nodes),
                             device, repeats, warmup)
        prop_audit = dict(model.prop.last_propagation_audit or {})
        rows.append({'layer': 'rff_propagation',
                     'kind': 'low_rank_rff_ungated', 'gated': False,
                     'input_edges': n_edges,
                     'active_edges': n_edges,
                     'executed_edges': int(sum(prop_audit.get(
                         'executed_edges_per_hop', [n_edges]))),
                     'active_ratio': 1.0 if n_edges else 0.0,
                     'executed_ratio': 1.0 if n_edges else 0.0,
                     'num_hops': prop_audit.get('num_hops'),
                     'latency_ms': latency,
                     'params': int(sum(p.numel() for p in
                                       model.prop.parameters())),
                     'note': 'RFF 多跳未被门控，始终执行全部输入边'})
        h_in = h0 + F.relu(hp)
        for name, attn in (('attn1', model.attn1), ('attn2', model.attn2)):
            h_cur = h_in
            latency, (agg, w) = _timed(
                lambda: attn(h_cur, edge_index, dt, num_nodes, 10.0, True),
                device, repeats, warmup)
            audit = dict(attn.last_gate_audit or {})
            rows.append({'layer': name,
                         'kind': 'temporal_sparse_attention', 'gated': True,
                         'input_edges': audit.get('input_edges', n_edges),
                         'active_edges': audit.get('active_edges'),
                         'executed_edges': audit.get('executed_edges'),
                         'active_ratio': audit.get('theoretical_active_ratio'),
                         'executed_ratio': audit.get('executed_edge_ratio'),
                         'hard_mask': audit.get('hard_mask'),
                         'compact_execution': audit.get('compact_execution'),
                         'gate_threshold_tau': float(
                             torch.sigmoid(attn.log_tau).detach().cpu()),
                         'latency_ms': latency,
                         'params': int(sum(p.numel() for p in
                                           attn.parameters()))})
            h_in = h_in + F.relu(agg)
        latency, _ = _timed(lambda: model.classifier(h_in), device, repeats,
                            warmup)
        rows.append({'layer': 'classifier', 'kind': 'linear', 'gated': False,
                     'input_edges': n_edges, 'active_edges': None,
                     'executed_edges': None, 'active_ratio': None,
                     'executed_ratio': None, 'latency_ms': latency,
                     'params': int(sum(p.numel() for p in
                                       model.classifier.parameters()))})
    for module in (model.prop, model.attn1, model.attn2):
        module.audit_enabled = False
    if was_training:
        model.train()
    total = float(sum(r['latency_ms'] for r in rows))
    for row in rows:
        row['latency_share'] = float(row['latency_ms'] / total) if total else None
    return {'rows': rows, 'total_layer_latency_ms': total,
            'note': ('逐层延迟之和不等于端到端前向：未计入 dropout、残差与'
                     '张量搬运开销，不得把逐层之和写成端到端延迟')}


# ------------------------------------------------------------------
# P3-6：H1–H5 观测收集与实际判定
# ------------------------------------------------------------------
def _drop_from_pair(clean_eval, attacked_eval, metric='f1'):
    clean = _evaluation_mean(clean_eval, metric)
    attacked = _evaluation_mean(attacked_eval, metric)
    if clean is None or attacked is None or clean <= 0:
        return None
    return float(1.0 - attacked / clean)


def collect_hypothesis_observations(sources):
    """从真实产物中提取 H1–H5 观测；缺文件即 incomplete_missing_*。"""
    declared = sorted(int(s) for s in sources.get('seeds', []) or [])
    observations = {'seeds_declared': declared, 'missing_sources': []}

    def _skip(name, reason):
        observations['missing_sources'].append({'hypothesis': name,
                                                'reason': reason})

    # H1：锁后评估的均值 F1 + 部署参数
    spec = sources.get('H1') or {}
    evaluation = _read(spec.get('evaluation', '')) if spec.get('evaluation') else None
    if evaluation is None:
        _skip('H1', 'incomplete_missing_evaluation')
    else:
        metrics = {'mean_f1': _evaluation_mean(evaluation, 'f1')}
        params = spec.get('deploy_params')
        if params is None:
            winner = spec.get('hpo_winners')
            if winner:
                winners = _read(winner) or {}
                by_seed = (winners.get('models', {}).get(
                    spec.get('hpo_model', 'tasgl'), {}) or {}).get('by_seed', {})
                values = [int(v['params']) for v in by_seed.values()
                          if v.get('params') is not None]
                params = int(np.mean(values)) if values else None
        metrics['deploy_params'] = params
        for key in ('teacher_params_offline', 'calibrator_params'):
            if key in spec:
                metrics[key] = spec[key]
        observations['H1'] = {'metrics': metrics,
                              'seeds_present': _evaluation_seeds(evaluation)}

    # H2：端到端效率中的均值延迟
    spec = sources.get('H2') or {}
    latency = spec.get('mean_latency_ms')
    if latency is None and spec.get('efficiency'):
        found = _read(spec['efficiency'])
        key = spec.get('key', 'mean_latency_ms')
        latency = _find_key(found, key) if found is not None else None
    if latency is None:
        _skip('H2', 'incomplete_missing_latency')
    else:
        metrics = {'mean_latency_ms': float(latency)}
        for key in ('latency_includes_calibration', 'batch_size', 'warmup',
                    'repeats', 'device'):
            if key in spec:
                metrics[key] = spec[key]
        observations['H2'] = {'metrics': metrics,
                              'seeds_present': list(declared)}

    # H3：干净 vs 攻击后评估的相对下降，逐基线
    spec = sources.get('H3') or {}
    tasgl_clean = _read(spec.get('tasgl_clean', ''))
    tasgl_attacked = _read(spec.get('tasgl_attacked', ''))
    baselines = {}
    for name, paths in sorted((spec.get('baselines') or {}).items()):
        clean = _read(paths.get('clean', ''))
        attacked = _read(paths.get('attacked', ''))
        if clean is None or attacked is None:
            baselines[name] = {'metrics': {}, 'seeds_present': [],
                               'status_note': 'incomplete_missing_attack_runs'}
            continue
        base_drop = _drop_from_pair(clean, attacked)
        tasgl_drop = (_drop_from_pair(tasgl_clean, tasgl_attacked)
                      if tasgl_clean and tasgl_attacked else None)
        metrics = {'baseline_relative_drop': base_drop,
                   'tasgl_relative_drop': tasgl_drop}
        metrics['relative_drop_ratio'] = (
            float(tasgl_drop / base_drop)
            if tasgl_drop is not None and base_drop not in (None, 0) and
            base_drop > 0 else None)
        for key in ('attack_family', 'budget', 'attack_seed_nested'):
            if key in paths:
                metrics[key] = paths[key]
        baselines[name] = {'metrics': metrics,
                           'seeds_present': _evaluation_seeds(clean)}
    if tasgl_clean is None or tasgl_attacked is None or not baselines:
        _skip('H3', 'incomplete_missing_attack_runs_or_baselines')
    else:
        observations['H3'] = {'baselines': baselines,
                              'seeds_present': _evaluation_seeds(tasgl_clean)}

    # H4：参考训练 vs 短训练的相对 F1 损失
    spec = sources.get('H4') or {}
    reference = _read(spec.get('reference', ''))
    short = _read(spec.get('short', ''))
    if reference is None or short is None:
        _skip('H4', 'incomplete_missing_rolling_runs')
    else:
        f_ref = _evaluation_mean(reference, 'f1')
        f_short = _evaluation_mean(short, 'f1')
        metrics = {'mean_f1_reference': f_ref, 'mean_f1_short': f_short,
                   'relative_f1_loss': (float(1.0 - f_short / f_ref)
                                        if f_ref else None)}
        if spec.get('rolling_strictness'):
            metrics['rolling_strictness'] = spec['rolling_strictness']
        observations['H4'] = {'metrics': metrics,
                              'seeds_present': _evaluation_seeds(reference)}

    # H5：校准质量与同骨干基线
    spec = sources.get('H5') or {}
    calibration = _read(spec.get('calibration', ''))
    baseline = _read(spec.get('baseline', ''))
    if calibration is None or baseline is None:
        _skip('H5', 'incomplete_missing_calibration')
    else:
        key = spec.get('ece_key', 'ece_overall')
        ece = calibration.get(key, calibration.get('ece'))
        baseline_ece = baseline.get(key, baseline.get('ece'))
        metrics = {'ece': ece, 'baseline_ece': baseline_ece}
        for extra in ('positive_class_ece', 'negative_class_ece',
                      'stratified_ece_max', 'brier', 'nll', 'bins',
                      'min_count', 'strata_field', 'calibration_window'):
            if extra in calibration:
                metrics[extra] = calibration[extra]
        observations['H5'] = {'metrics': metrics,
                              'seeds_present': list(declared)}
    return observations


def build_verdicts(observations, threshold_table, lock=None):
    """把观测套入 P2-2 已注册的门限表，输出 pass/falsified/inconclusive/incomplete。"""
    declared = observations.get('seeds_declared') or \
        threshold_table.get('declared_seeds') or []
    payload = {k: v for k, v in observations.items() if k.startswith('H')}
    missing = observations.get('missing_sources') or []
    result = apply_threshold_table(threshold_table, payload, lock, declared)
    result['protocol'] = 'p3_hypothesis_verdicts_v1'
    result['observation_sources'] = {
        k: sorted((v or {}).get('metrics', {}).keys())
        if isinstance(v, dict) else None for k, v in payload.items()}
    result['missing_sources'] = missing
    return result
