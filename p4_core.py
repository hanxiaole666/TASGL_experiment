"""p4_core.py — P4 终版证据层核心算法

针对审稿意见中"结论不可追溯 / 门限无依据 / 无独立确认 / 统计效力不足 /
强断言过度"五类问题, 提供可复现的程序化判定。

模块
----
A. 种子失败根因审计
   - 门控收敛审计 (gate_convergence_report / classify_seed_failure)
   - 前向语义分解 (decompose_forward_semantics / attribute_path)
   - 运行准入与双重轨道分析 (run_admission / dual_track_analysis)
B. 严格 rolling-inductive 独立窗口
   - 诱导子图 + 仅训练窗拟合预处理 (strict_rolling_plan / induced_subgraph
     / fit_scaler_train_only)
   - 未来泄漏审计 (audit_no_future_leakage)
C. 强断言降级裁决 (adjudicate_claim / weaken_claims)
D. 效应量统计报告 (paired_report)
E. 正类 / 分类别校准 (复用 p2_core, 此处提供真实预测封装)

设计原则
--------
1. 只输出能被运行记录证实的结论; 缺数据一律返回 *_missing / unverifiable,
   绝不用合成值填补。
2. 所有门限必须带 basis 字段; basis='arbitrary' 一律判为 unjustified。
3. 拒绝"先看到结果再定判据": 准入规则必须预先注册 (pre-registered)。
"""
import hashlib
import json
import math
import os

import numpy as np

# ------------------------------------------------------------------
# 常量
# ------------------------------------------------------------------
ROOTCAUSE_SCHEMA = 'p4_rootcause_v1'
ROLLING_SCHEMA = 'p4_strict_rolling_v1'
CLAIM_SCHEMA = 'p4_claim_adjudication_v1'

# 门控收敛判据 (预先注册, 依据见 P4执行与判定说明.md 第 2 节)
GATE_CONVERGENCE_CRITERIA = {
    'schema': 'p4_gate_convergence_v1',
    # 主判据: 至少一层的硬掩码活跃边比例 <= 0.98, 即门控确实关掉了一部分边。
    'primary_min_active_ratio_upper': 0.98,
    # 辅判据: 门控输出层权重范数达到 0.05。观测到的双峰间隔为
    # 0.033 (失败种子) vs 0.245~0.259 (收敛种子), 取收敛侧最小值的 1/5
    # 作为保守下界。
    'secondary_min_gate_w2_norm': 0.05,
    'note': ('门控初始化 bias=2.0 / W2=0 / tau=0.5 使 a=sigmoid(2.0)=0.881>tau, '
             '训练初期全部边活跃。若训练未在门控收敛前结束, 模型退化为'
             '稠密图注意力, TASGL 的稀疏性主张在该种子上不成立。'),
}

# 前向语义三轴
GATE_AXIS = ('hard', 'soft')
STOCHASTIC_AXIS = ('deterministic', 'mc_dropout')
THRESHOLD_AXIS = ('tuned_same', 'fixed')


def _sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False,
                   default=str).encode('utf-8')).hexdigest()


def _num(x):
    try:
        return float(x)
    except Exception:
        return None


# ==================================================================
# A. 种子失败根因审计
# ==================================================================
def semantics_spec(gate='hard', stochastic='deterministic', mc_T=1,
                   threshold_source='tuned_same', fixed_threshold=0.5):
    """前向语义规格。

    gate          : 'hard' 硬掩码 (w=(a>tau)) / 'soft' 软掩码
    stochastic    : 'deterministic' eval 模式单次前向 /
                    'mc_dropout' T 次 dropout 前向取平均
    threshold_source: 'tuned_same' 在本语义自己的验证概率上网格调阈值 /
                    'fixed' 使用 fixed_threshold
    """
    if gate not in GATE_AXIS:
        raise ValueError(f'gate 必须是 {GATE_AXIS}')
    if stochastic not in STOCHASTIC_AXIS:
        raise ValueError(f'stochastic 必须是 {STOCHASTIC_AXIS}')
    if threshold_source not in THRESHOLD_AXIS:
        raise ValueError(f'threshold_source 必须是 {THRESHOLD_AXIS}')
    return {
        'gate': gate, 'stochastic': stochastic, 'mc_T': int(mc_T),
        'threshold_source': threshold_source,
        'fixed_threshold': float(fixed_threshold),
    }


def semantics_key(spec):
    return 'g=%s|s=%s|T=%d|t=%s' % (spec['gate'], spec['stochastic'],
                                    spec['mc_T'], spec['threshold_source'])


def tune_threshold(prob_pos_val, y_val, grid=None):
    """在给定语义自己的验证概率上网格搜索使 F1 最大的阈值。

    不使用测试集, 无泄漏。返回 (阈值, {阈值: F1})。"""
    from sklearn.metrics import f1_score
    if grid is None:
        grid = np.linspace(0.02, 0.98, 49)
    grid = np.asarray(grid, dtype=float)
    pv = np.asarray(prob_pos_val, dtype=float)
    yv = np.asarray(y_val, dtype=int)
    f1s = [f1_score(yv, pv >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))]), {round(float(t), 4): float(f)
                                              for t, f in zip(grid, f1s)}


def score_probabilities(prob, y, threshold):
    """prob: [M,2]; y: [M]。返回该阈值下的全部判别指标。"""
    from sklearn.metrics import f1_score, precision_score, recall_score, \
        average_precision_score, roc_auc_score
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)
    p1 = prob[:, 1]
    pred = (p1 >= threshold).astype(int)
    out = {
        'threshold': float(threshold),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'predicted_positive_rate': float(pred.mean()),
        'mean_p_positive': float(p1.mean()),
        'mean_entropy': float(
            (-(prob * np.log(prob + 1e-12)).sum(-1)).mean()),
    }
    if 0 < y.sum() < len(y):
        out['average_precision'] = float(average_precision_score(y, p1))
        out['roc_auc'] = float(roc_auc_score(y, p1))
    return out


def decompose_forward_semantics(model, data, masks, grid=None, mc_T=10,
                                fixed_threshold=0.5):
    """在 2x2 前向语义网格上打分, 用于定位指标的语义来源。

    masks = (train_mask, val_mask, test_mask), 均为 torch bool 张量。
    返回 {semantics_key: {threshold, tuned_threshold, fixed:{...},
          tuned:{...}}}。
    """
    import torch
    train_mask, val_mask, test_mask = masks
    y_val = data['y'][val_mask].detach().cpu().numpy()
    y_test = data['y'][test_mask].detach().cpu().numpy()
    cells = {}
    for gate in GATE_AXIS:
        for stochastic in STOCHASTIC_AXIS:
            spec = semantics_spec(gate=gate, stochastic=stochastic,
                                  mc_T=mc_T if stochastic == 'mc_dropout' else 1,
                                  threshold_source='tuned_same')
            prob = forward_probabilities(model, data, spec)
            pv = prob[val_mask.detach().cpu().numpy()][:, 1]
            thr, curve = tune_threshold(pv, y_val, grid)
            pt = prob[test_mask.detach().cpu().numpy()]
            cells[semantics_key(spec)] = {
                'spec': spec,
                'tuned_threshold': thr,
                'threshold_curve': curve,
                'tuned_same': score_probabilities(pt, y_test, thr),
                'fixed': score_probabilities(pt, y_test, fixed_threshold),
            }
    return cells


def forward_probabilities(model, data, spec):
    """按语义取全图概率 [N,2] (numpy)。不修改模型的持久模式。"""
    import torch
    semantics = semantics_spec(**{k: v for k, v in spec.items()})
    hard = semantics['gate'] == 'hard'
    was_training = model.training
    if semantics['stochastic'] == 'mc_dropout':
        model.train()
        with torch.no_grad():
            ps = []
            for _ in range(semantics['mc_T']):
                logits, _ = model(data['x'], data['edge_index'],
                                  data['time_gaps'], data['num_nodes'],
                                  hard_mask=hard)
                ps.append(torch.softmax(logits, dim=-1))
            prob = torch.stack(ps).mean(0)
    else:
        model.eval()
        with torch.no_grad():
            logits, _ = model(data['x'], data['edge_index'],
                              data['time_gaps'], data['num_nodes'],
                              hard_mask=hard)
            prob = torch.softmax(logits, dim=-1)
    if was_training:
        model.train()
    else:
        model.eval()
    return prob.detach().cpu().numpy()


def attribute_path(cells, path, metric='f1'):
    """沿一条语义路径做逐步归因。

    path: [semantics_key, ...], 相邻两项计算 delta。
    返回 [{step, from_key, to_key, delta, }] 与 share (|delta| 占比)。
    """
    steps = []
    for a, b in zip(path[:-1], path[1:]):
        va = cells.get(a, {}).get('fixed', {}).get(metric)
        vb = cells.get(b, {}).get('fixed', {}).get(metric)
        if va is None or vb is None:
            steps.append({'from_key': a, 'to_key': b, 'delta': None,
                          'status': 'missing_cell'})
        else:
            steps.append({'from_key': a, 'to_key': b, 'delta': float(vb - va),
                          'from_value': float(va), 'to_value': float(vb),
                          'status': 'ok'})
    tot = sum(abs(s['delta']) for s in steps if s.get('delta') is not None)
    for s in steps:
        s['share'] = (abs(s['delta']) / tot) if (s.get('delta') is not None
                                                 and tot > 0) else None
    return {'path': path, 'metric': metric, 'steps': steps,
            'total_delta': float(sum(s['delta'] for s in steps
                                     if s.get('delta') is not None)),
            'abs_total': float(tot)}


def reproduce_check(cells, observed, key=None, tol=0.01, metric='f1'):
    """复现性校验: 某语义格子的指标是否落在观测值 tol 邻域内。"""
    if key is None:
        # 默认取 E7 实际使用的语义: 软掩码 + MC-Dropout + 固定 0.5
        key = [k for k in cells
               if k.startswith('g=soft|s=mc_dropout')]
        key = key[0] if key else None
    if key is None or key not in cells:
        return {'reproduced': False, 'status': 'missing_cell', 'key': key}
    val = cells[key]['fixed'][metric]
    diff = abs(val - float(observed))
    return {'reproduced': bool(diff <= tol), 'key': key,
            'metric': metric, 'observed': float(observed),
            'recomputed': float(val), 'abs_diff': float(diff), 'tol': tol}


# --- 门控收敛审计 -------------------------------------------------
def gate_convergence_report(model, data, criteria=None):
    """审计训练后模型的时序稀疏门控是否真的收敛。

    返回每层: 活跃边比例 / 门控权重范数 / 偏置 / tau; 以及整体判据。
    """
    import torch
    criteria = criteria or GATE_CONVERGENCE_CRITERIA
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _, aux = model(data['x'], data['edge_index'], data['time_gaps'],
                       data['num_nodes'], hard_mask=True)
        gates = aux.get('gates') or (None, None)
    layers = []
    for li, layer in enumerate(getattr(model, 'gate_layers',
                                       (getattr(model, 'attn1', None),
                                        getattr(model, 'attn2', None)))):
        if layer is None:
            continue
        w = gates[li] if li < len(gates) else None
        head = layer.gate[-1]
        w2 = head.weight.detach().cpu()
        w1 = layer.gate[0].weight.detach().cpu()
        log_tau = float(layer.log_tau.detach().cpu())
        active = (float((w > 0).float().mean()) if w is not None
                  and w.numel() > 0 else None)
        layers.append({
            'layer': li + 1,
            'active_edge_ratio': active,
            'gate_w2_norm': float(w2.norm()),
            'gate_w2_absmax': float(w2.abs().max()),
            'gate_w1_norm': float(w1.norm()),
            'gate_bias': float(head.bias.detach().cpu().view(-1)[0]),
            'log_tau': log_tau,
            'tau': float(1.0 / (1.0 + math.exp(-log_tau))),
        })
    actives = [l['active_edge_ratio'] for l in layers
               if l['active_edge_ratio'] is not None]
    norms = [l['gate_w2_norm'] for l in layers]
    primary = bool(actives) and min(actives) <= \
        criteria['primary_min_active_ratio_upper']
    secondary = bool(norms) and max(norms) >= \
        criteria['secondary_min_gate_w2_norm']
    if was_training:
        model.train()
    return {
        'schema': 'p4_gate_convergence_report_v1',
        'criteria': criteria,
        'layers': layers,
        'min_active_edge_ratio': min(actives) if actives else None,
        'max_gate_w2_norm': max(norms) if norms else None,
        'primary_pass': primary,
        'secondary_pass': secondary,
        'converged': bool(primary and secondary),
    }


def classify_seed_failure(gate_report):
    """把种子判为 gate_collapsed_open / gate_converged / unknown。"""
    if not gate_report or gate_report.get('min_active_edge_ratio') is None:
        return {'status': 'unknown_missing_gate_audit',
                'action': '不得纳入任何关于稀疏性主张的统计'}
    if gate_report.get('converged'):
        return {'status': 'gate_converged', 'action': '可纳入主分析'}
    return {'status': 'gate_collapsed_open',
            'action': ('该种子上门控未收敛, 稀疏性主张不成立; '
                       '主分析仍须保留该种子, 另做敏感性分析')}


def run_admission(gate_report, criteria=None):
    """运行准入: 门控未收敛的种子标记为 rejected, 但不得静默删除。"""
    cls = classify_seed_failure(gate_report)
    return {'admit': cls['status'] == 'gate_converged',
            'classification': cls['status'],
            'action': cls['action'],
            'criteria_hash': _sha(criteria or GATE_CONVERGENCE_CRITERIA)}


def dual_track_analysis(records, key='f1', alpha=0.05):
    """双重轨道: 主分析用全部种子, 敏感性分析仅用通过准入的种子。

    若两轨结论方向不一致, 判为 conclusion_unstable -> 不得下强断言。
    """
    def _mean_sd(vals):
        v = np.asarray([x for x in vals if x is not None], dtype=float)
        if v.size == 0:
            return None, None
        if v.size < 2:
            return float(v[0]), None
        return float(v.mean()), float(v.std(ddof=1))

    allv = [r.get(key) for r in records]
    ok = [r.get(key) for r in records
          if (r.get('admission') or {}).get('admit')]
    ma, sa = _mean_sd(allv)
    mo, so = _mean_sd(ok)
    n_all = len([x for x in allv if x is not None])
    n_ok = len([x for x in ok if x is not None])
    # 预先注册的稳定性判据: 剔除未通过准入的种子后, 均值漂移不得超过
    # 全体标准差的 25%。漂移更大说明结论依赖"删掉坏种子", 判为不稳定。
    tol = (0.25 * sa) if sa else (abs(ma) * 0.05 + 1e-9 if ma else 0.0)
    stable = None
    if ma is not None and mo is not None:
        stable = bool(abs(ma - mo) <= tol)
    return {
        'schema': 'p4_dual_track_v1', 'key': key,
        'primary_all_seeds': {'n': n_all, 'mean': ma, 'sd': sa},
        'sensitivity_admitted': {'n': n_ok, 'mean': mo, 'sd': so},
        'rejected': n_all - n_ok, 'stability_tolerance': tol,
        'conclusion_stable': stable,
        'verdict': ('conclusion_stable' if stable else
                    'conclusion_unstable_no_strong_claim'
                    if stable is False else 'insufficient_data'),
    }


# ==================================================================
# B. 严格 rolling-inductive 独立窗口
# ==================================================================
def induced_subgraph(edge_index, time_steps, cutoff):
    """时间 <= cutoff 的诱导子图: 节点与边同时裁剪。

    未来节点/未来边均不可见 (边要求两端都在窗口内)。
    返回 (node_idx, local_edge_index)。
    """
    edge_index = np.asarray(edge_index)
    time_steps = np.asarray(time_steps)
    node_idx = np.where(time_steps <= cutoff)[0]
    keep = np.zeros(time_steps.shape[0], dtype=bool)
    keep[node_idx] = True
    src, dst = edge_index[0], edge_index[1]
    em = keep[src] & keep[dst]
    remap = -np.ones(time_steps.shape[0], dtype=np.int64)
    remap[node_idx] = np.arange(node_idx.shape[0])
    return node_idx, np.stack([remap[src[em]], remap[dst[em]]], axis=0)


def fit_scaler_train_only(X, visible_idx):
    """仅用训练窗可见节点拟合标准化器 (严格 inductive 要求)。

    现有 data_loader.load_elliptic 在全量节点上 fit StandardScaler,
    对严格 rolling 而言属预处理泄漏。
    """
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(np.asarray(X)[visible_idx])
    return sc


def strict_rolling_plan(time_steps, labels, train_end, val_end, test_end,
                        min_labeled=200):
    """严格 rolling 划分: 训练 <= train_end, 验证 (train_end, val_end],
    测试 (val_end, test_end]。全部落在同一时间轴上且不重叠。

    同时给出用于泄漏审计的窗口边界。
    """
    ts = np.asarray(time_steps)
    lab = np.asarray(labels)
    if not (train_end < val_end < test_end):
        raise ValueError('要求 train_end < val_end < test_end')
    tr = (ts <= train_end) & (lab >= 0)
    va = (ts > train_end) & (ts <= val_end) & (lab >= 0)
    te = (ts > val_end) & (ts <= test_end) & (lab >= 0)
    rec = {
        'schema': ROLLING_SCHEMA,
        'train_window': [int(ts[tr].min()) if tr.any() else None,
                         int(train_end)],
        'val_window': [int(train_end) + 1, int(val_end)],
        'test_window': [int(val_end) + 1, int(test_end)],
        'n_train': int(tr.sum()), 'n_val': int(va.sum()),
        'n_test': int(te.sum()),
        'pos_rate_train': float((lab[tr] == 1).mean()) if tr.any() else None,
        'pos_rate_val': float((lab[va] == 1).mean()) if va.any() else None,
        'pos_rate_test': float((lab[te] == 1).mean()) if te.any() else None,
        'sufficient_labeled': bool(min(tr.sum(), va.sum(), te.sum())
                                   >= min_labeled),
        'graph_cutoff': int(val_end),
        'induction_rule': '节点与边同时裁剪到 time<=val_end; '
                          '标准化器仅在 train 窗可见节点上拟合',
    }
    rec['fingerprint'] = _sha({k: rec[k] for k in
                               ('train_window', 'val_window', 'test_window',
                                'graph_cutoff')})
    return rec, (tr, va, te)


def audit_no_future_leakage(node_idx, edge_index_local, time_steps_local,
                            scaler_fit_idx, cutoff, train_end,
                            time_steps_global=None):
    """审计严格 inductive 是否成立: 逐项返回布尔。

    检查:
      1. 诱导子图节点全部 <= cutoff
      2. 诱导子图中不存在任何一端超出子图范围的边
      3. 标准化器拟合集不含 train_end 之后的节点
         (需传 time_steps_global; 否则该项降级为 weak_check)
    """
    ts = np.asarray(time_steps_local)
    ei = np.asarray(edge_index_local)
    if time_steps_global is not None and scaler_fit_idx is not None and \
            np.asarray(scaler_fit_idx).size > 0:
        gts = np.asarray(time_steps_global)
        sfi = np.asarray(scaler_fit_idx)
        scaler_ok = bool(gts[sfi].max() <= train_end)
        scaler_mode = 'checked_via_time_steps'
    else:
        scaler_ok = False
        scaler_mode = 'weak_check_index_vs_cutoff'
    checks = {
        'all_nodes_within_cutoff': bool(ts.size == 0 or ts.max() <= cutoff),
        'no_future_edge_endpoint': bool(ei.size == 0 or ei.max()
                                        < ts.shape[0]),
        'scaler_fit_indices_within_train': scaler_ok,
    }
    return {
        'schema': 'p4_no_future_leakage_v1',
        'cutoff': int(cutoff), 'train_end': int(train_end),
        'checks': checks, 'scaler_check_mode': scaler_mode,
        'passed': all(checks.values()),
        'note': ('全图前向 (transductive) 会把未来节点的特征带入聚合, '
                 '必须在报告中与严格 inductive 分开陈述'),
    }


def window_exposure_audit(declared_window, ledger):
    """窗口是否已被开发过程污染。ledger 为已声明窗口列表。

    返回 exposure 等级: never_used / used_in_selection / used_in_reporting。
    """
    a, b = declared_window
    hits = []
    for rec in ledger:
        w = rec.get('window') or [rec.get('start'), rec.get('end')]
        if w is None or w[0] is None:
            continue
        if not (b < w[0] or a > w[1]):
            hits.append({'name': rec.get('name'), 'window': w,
                         'usage': rec.get('usage'),
                         'prior_exposure': rec.get('prior_exposure')})
    if not hits:
        return {'window': [a, b], 'exposure': 'never_used',
                'usable_for_confirmation': True, 'overlaps': []}
    return {'window': [a, b], 'exposure': 'previously_observed',
            'usable_for_confirmation': False, 'overlaps': hits,
            'consequence': ('该窗口不得作为独立确认证据, 只能作为'
                            'exploratory_reconfirmation')}


# ==================================================================
# C. 强断言降级裁决
# ==================================================================
CLAIM_KINDS = ('dominated', 'below', 'superior_to', 'robust_to',
               'calibrated', 'faster_than')

CLAIM_REQUIREMENTS = {
    'dominated': ['n_seeds>=10', 'equal_budget_hpo', 'paired_test_holm',
                  'effect_size_reported', 'ci_excludes_zero',
                  'independent_or_declared_exploratory'],
    'below': ['n_seeds>=10', 'paired_test_holm', 'effect_size_reported',
              'ci_excludes_zero'],
    'superior_to': ['n_seeds>=10', 'equal_budget_hpo', 'paired_test_holm',
                    'effect_size_reported', 'ci_excludes_zero',
                    'independent_or_declared_exploratory'],
    'robust_to': ['non_oracle_attack', 'attacker_model_declared',
                  'n_attack_repetitions', 'clean_to_attack_transition'],
    'calibrated': ['positive_class_ece', 'classwise_ece',
                   'not_overall_ece_only'],
    'faster_than': ['environment_fingerprint', 'measurement_protocol',
                    'full_chain_cost'],
}

CLAIM_REQUIREMENT_TEXT = {
    'n_seeds>=10': '预注册 >=10 个种子 (或功效分析给出更少种子数)',
    'equal_budget_hpo': '每个基线完成等预算、仅用验证集的超参搜索',
    'paired_test_holm': '配对检验并做 Holm 多重比较校正',
    'effect_size_reported': '报告效应量 (Cohen dz) 与置信区间',
    'ci_excludes_zero': '差值的 95% 置信区间不包含 0',
    'independent_or_declared_exploratory': ('在未参与开发的独立划分上确认, '
                                            '或明确声明为探索性复核'),
    'non_oracle_attack': '真实非 oracle 自适应攻击 (而非标签注入式 oracle)',
    'attacker_model_declared': '声明攻击者知识/目标/成本',
    'n_attack_repetitions': '声明攻击种子与重复次数',
    'clean_to_attack_transition': '报告 clean->attack 的指标转移, 而非单一攻击态',
    'positive_class_ece': '报告正类 ECE (类别质量加权)',
    'classwise_ece': '报告分类别 ECE 分解',
    'not_overall_ece_only': '不能只用整体 ECE (不平衡下会退化)',
    'environment_fingerprint': '完整环境指纹 (GPU/CPU/驱动/CUDA/PyTorch/OS)',
    'measurement_protocol': '声明 warmup/重复次数/同步方式/批量口径',
    'full_chain_cost': '计入师生全链路成本 (含教师训练与蒸馏)',
}

WEAKENED_TEMPLATE = {
    'dominated': ('在本实验设定的等预算比较下, {a} 与 {b} 的 {metric} 差异'
                  '未达到预注册的统计判据 ({missing}); 只能表述为'
                  '"在当前设置下未观察到 {a} 对 {b} 的统计显著优势", '
                  '不得表述为"支配"。'),
    'below': ('{a} 的 {metric} 低于 {b}, 但 {missing}; 只能表述为'
              '"点估计较低, 差异未达统计显著"。'),
    'superior_to': ('{a} 的 {metric} 高于 {b}, 但 {missing}; 只能表述为'
                    '"点估计较高, 差异未达统计显著"。'),
    'robust_to': ('当前攻击实验为 oracle 注入式, 缺少 {missing}; '
                  '不得表述为"对自适应攻击鲁棒"。'),
    'calibrated': ('当前仅报告整体 ECE, 缺少 {missing}; '
                   '不得表述为"校准良好"。'),
    'faster_than': ('当前延迟/成本测量缺少 {missing}; '
                    '不得表述为"更快/更省", 只能给单点测量值并附口径。'),
}


def adjudicate_claim(kind, evidence, a='TASGL', b='基线', metric='F1'):
    """裁决一条强断言能否保留。

    evidence: {'n_seeds':int, 'equal_budget_hpo':bool, 'paired_p_holm':float,
               'effect_size':float, 'ci':[lo,hi], 'independent':bool, ...}
    返回 {status, satisfied, missing, weakened_text}。
    """
    if kind not in CLAIM_KINDS:
        raise ValueError(f'kind 必须是 {CLAIM_KINDS}')
    ev = evidence or {}
    reqs = CLAIM_REQUIREMENTS[kind]

    def _ok(r):
        if r == 'n_seeds>=10':
            n = ev.get('n_seeds')
            n_req = ev.get('n_seeds_required_by_power')
            if n is None:
                return False
            if n_req is not None and ev.get('power_analysis_provided'):
                return int(n) >= int(n_req)
            return int(n) >= 10
        if r == 'equal_budget_hpo':
            return bool(ev.get('equal_budget_hpo'))
        if r == 'paired_test_holm':
            return ev.get('paired_p_holm') is not None
        if r == 'effect_size_reported':
            return ev.get('effect_size') is not None
        if r == 'ci_excludes_zero':
            ci = ev.get('ci')
            return bool(ci) and (ci[0] > 0 or ci[1] < 0)
        if r == 'independent_or_declared_exploratory':
            return bool(ev.get('independent')) or \
                ev.get('exploratory_declared') is True
        if r == 'non_oracle_attack':
            return bool(ev.get('non_oracle_attack'))
        if r == 'attacker_model_declared':
            return bool(ev.get('attacker_model_declared'))
        if r == 'n_attack_repetitions':
            return ev.get('n_attack_repetitions') is not None
        if r == 'clean_to_attack_transition':
            return bool(ev.get('clean_to_attack_transition'))
        if r == 'positive_class_ece':
            return ev.get('positive_class_ece') is not None
        if r == 'classwise_ece':
            return ev.get('classwise_ece') is not None
        if r == 'not_overall_ece_only':
            return bool(ev.get('beyond_overall_ece'))
        if r == 'environment_fingerprint':
            return bool(ev.get('environment_fingerprint'))
        if r == 'measurement_protocol':
            return bool(ev.get('measurement_protocol'))
        if r == 'full_chain_cost':
            return bool(ev.get('full_chain_cost'))
        return False

    satisfied = [r for r in reqs if _ok(r)]
    missing = [r for r in reqs if not _ok(r)]
    if not missing:
        status = 'supported'
    elif len(satisfied) == 0:
        status = 'unverifiable'
    elif 'ci_excludes_zero' in missing and ev.get('ci') is not None:
        status = 'weakened'       # 方向存在但区间含 0
    elif ev.get('n_seeds') is not None and ev['n_seeds'] < 10:
        status = 'weakened'
    else:
        status = 'weakened'
    miss_txt = '; '.join(CLAIM_REQUIREMENT_TEXT[m] for m in missing) or '无'
    text = WEAKENED_TEMPLATE[kind].format(a=a, b=b, metric=metric,
                                          missing=miss_txt)
    return {'schema': CLAIM_SCHEMA, 'kind': kind, 'status': status,
            'satisfied': satisfied, 'missing': missing,
            'missing_text': miss_txt, 'weakened_text': text,
            'retraction_required': status in ('weakened', 'unverifiable')}


def weaken_claims(claims):
    """claims: [{id, kind, a, b, metric, evidence}] -> 裁决列表 + 汇总。"""
    out = [dict(c, **adjudicate_claim(c['kind'], c.get('evidence'),
                                      c.get('a', 'TASGL'),
                                      c.get('b', '基线'),
                                      c.get('metric', 'F1')))
           for c in claims]
    n_ret = sum(1 for o in out if o['retraction_required'])
    return {'schema': 'p4_weaken_claims_v1', 'claims': out,
            'n_claims': len(out), 'n_requiring_retraction': n_ret,
            'all_supported': n_ret == 0}


# ==================================================================
# D. 效应量统计报告
# ==================================================================
def paired_report(a, b, alpha=0.05, n_boot=2000, seed=0):
    """配对比较的完整报告: 均值差 / SD / 95%CI / Cohen dz / t 检验 /
    Wilcoxon / 符号检验 / MDE。缺少任一项都不得称"显著"。

    a, b 为同长度、同种子顺序的指标序列。
    """
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {'status': 'insufficient_data', 'n': 0}
    a, b = a[:n], b[:n]
    d = a - b
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1)) if n >= 2 else None
    se = sd / math.sqrt(n) if (sd and n >= 2) else None
    if se and se > 0:
        tcrit = float(stats.t.ppf(1 - alpha / 2, n - 1))
        ci = [mean_d - tcrit * se, mean_d + tcrit * se]
        tstat, tp = stats.ttest_rel(a, b)
        dz = mean_d / sd
    else:
        ci, tstat, tp, dz = None, None, None, None
    try:
        wstat, wp = stats.wilcoxon(a, b) if n >= 5 else (None, None)
    except Exception:
        wstat, wp = None, None
    k = int((d > 0).sum())
    sp = float(stats.binomtest(k, n, 0.5).pvalue) if n >= 1 else None
    rng = np.random.default_rng(seed)
    if n >= 2:
        boot = np.array([rng.choice(d, n, replace=True).mean()
                         for _ in range(n_boot)])
        bci = [float(np.percentile(boot, 100 * alpha / 2)),
               float(np.percentile(boot, 100 * (1 - alpha / 2)))]
    else:
        bci = None
    # 最小可检测效应 (双尾 t, 功效 0.8)
    if sd and n >= 2:
        tcrit = float(stats.t.ppf(1 - alpha / 2, n - 1))
        try:
            ncp = float(stats.nct.ppf(0.8, n - 1, 0.0)) if False else None
        except Exception:
            ncp = None
        mde = (tcrit + 0.8416) * se      # 正态近似
    else:
        mde = None
    return {
        'schema': 'p4_paired_report_v1', 'n': int(n), 'alpha': alpha,
        'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
        'mean_diff': mean_d, 'sd_diff': sd, 'se_diff': se,
        'ci95': [float(x) for x in ci] if ci else None,
        'bootstrap_ci95': bci,
        'cohen_dz': float(dz) if dz is not None else None,
        't_stat': float(tstat) if tstat is not None else None,
        't_p': float(tp) if tp is not None else None,
        'wilcoxon_stat': float(wstat) if wstat is not None else None,
        'wilcoxon_p': float(wp) if wp is not None else None,
        'sign_test_p': sp, 'positive_diff_count': k,
        'mde_at_power80': float(mde) if mde is not None else None,
        'ci_excludes_zero': bool(ci and (ci[0] > 0 or ci[1] < 0)),
        'min_attainable_wilcoxon_p': float(2.0 / 2 ** n),
        'wilcoxon_can_reach_alpha': bool(2.0 / 2 ** n <= alpha),
        'status': 'ok' if n >= 2 else 'insufficient_data',
        'caveat': ('n<10 时配对检验的最小可达 p 值大于 0.05, 统计效力不足, '
                   '任何"无显著差异"都只是"未检测到"'),
    }


# ==================================================================
# E. 正类 / 分类别校准封装
# ==================================================================
def classwise_calibration(prob, y, n_bins=15, threshold=0.5):
    """正类(类别质量加权)ECE + 负类 ECE + 整体 ECE + 分箱明细。

    prob 可以是 [M,2] (取第 1 列) 或 [M] 正类概率。
    整体 ECE 在极端不平衡下会退化到接近 0, 必须并列报告正类 ECE。
    """
    try:
        from p2_core import positive_class_ece, negative_class_ece, \
            calibration_quality_report
    except Exception:
        return {'status': 'p2_core_unavailable'}
    prob = np.asarray(prob, dtype=float)
    if prob.ndim == 2:
        prob = prob[:, 1]
    prob = np.clip(prob.ravel(), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=int)
    rep = calibration_quality_report(y, prob, threshold=threshold,
                                     bins=n_bins)
    pos = positive_class_ece(y, prob, n_bins)
    neg = negative_class_ece(y, prob, n_bins)
    return {
        'positive_class_ece': float(pos['positive_class_ece']),
        'negative_class_ece': float(neg['negative_class_ece']),
        'overall_ece': float(rep.get('ece_overall')),
        'brier': float(rep.get('brier')) if rep.get('brier') else None,
        'positive_rate': float((y == 1).mean()) if y.size else None,
        'n': int(len(y)),
        'bins': int(n_bins),
        'warning': ('整体 ECE 在正类占比很低时会退化到接近 0; '
                    '论文中必须以正类 ECE 为主指标'),
        'detail': rep,
    }


def save_json(obj, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path
