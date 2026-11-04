"""p0_core.py — P0 实验的可审计公共原语。

本模块只提供协议、统计、校准、严格滚动视图和语义对齐工具；不包含任何
预设实验结论。所有新结果写入 results/p0/，与 v2/v3/v4 历史结果隔离。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, log_loss


# ------------------------------------------------------------------
# 可复现文件与锁
# ------------------------------------------------------------------
def canonical_json(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk_size), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    """原子写 JSON；失败时保留旧文件，禁止静默覆盖不同锁。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(obj) + '\n'
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def make_lock(config, artifact_paths, code_paths, protocol='p0_lock_v1'):
    """在看到测试标签前绑定配置、预测/校准产物和实现代码。"""
    required = {'dataset', 'seeds', 'threshold', 'calibration',
                'statistics', 'exclusions', 'evaluation'}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f'锁配置缺少字段: {missing}')
    if not config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('seeds 必须非空且不得重复')
    if config['statistics'].get('test') != 'wilcoxon_exact_two_sided':
        raise ValueError('P0-5 必须预声明精确双侧 Wilcoxon')
    artifacts = {}
    for role, path in sorted(artifact_paths.items()):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'锁定产物不存在 ({role}): {path}')
        artifacts[role] = {'path': str(path.resolve()),
                           'bytes': path.stat().st_size,
                           'sha256': file_sha256(path)}
    code = {}
    for path in code_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'锁定代码不存在: {path}')
        code[str(path.resolve())] = file_sha256(path)
    payload = {'protocol': protocol, 'config': config,
               'artifacts': artifacts, 'code_sha256': code}
    return {**payload, 'lock_id': sha256_bytes(canonical_json(payload).encode())[:24]}


def verify_lock(lock, config=None, verify_artifacts=True, verify_code=True):
    """返回差异列表；空列表表示锁仍完全有效。"""
    errors = []
    payload = {k: lock[k] for k in ('protocol', 'config', 'artifacts',
                                   'code_sha256')}
    expected_id = sha256_bytes(canonical_json(payload).encode())[:24]
    if lock.get('lock_id') != expected_id:
        errors.append('lock_id 与锁内容不一致')
    if config is not None and canonical_json(config) != canonical_json(lock['config']):
        errors.append('配置与锁不一致')
    if verify_artifacts:
        for role, meta in lock['artifacts'].items():
            path = Path(meta['path'])
            if not path.is_file():
                errors.append(f'{role}: 文件不存在 {path}')
            elif file_sha256(path) != meta['sha256']:
                errors.append(f'{role}: SHA-256 与锁不一致')
    if verify_code:
        for name, digest in lock['code_sha256'].items():
            path = Path(name)
            if not path.is_file():
                errors.append(f'代码不存在: {name}')
            elif file_sha256(path) != digest:
                errors.append(f'代码已改变: {name}')
    return errors


# ------------------------------------------------------------------
# P0-5：预声明配对统计
# ------------------------------------------------------------------
def _records_by_seed(records, value_key):
    if isinstance(records, dict):
        records = [{'seed': k, value_key: v} for k, v in records.items()]
    out = {}
    for row in records:
        seed = int(row['seed'])
        if seed in out:
            raise ValueError(f'重复 seed={seed}；不得静默覆盖')
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError(f'seed={seed} 的 {value_key} 非有限数')
        out[seed] = value
    return out


def paired_statistics(records_a, records_b, declared_seeds, value_key='f1',
                      alpha=0.05):
    """完整种子审计 + 样本 SD + t CI + 配对效应量 + 精确双侧 Wilcoxon。

    缺少预声明种子时状态为 incomplete_missing_seeds；不会把可用子集冒充
    完整确认。零差值按预声明 wilcox 规则处理并报告有效对数。
    """
    a = _records_by_seed(records_a, value_key)
    b = _records_by_seed(records_b, value_key)
    declared = [int(s) for s in declared_seeds]
    missing_a = [s for s in declared if s not in a]
    missing_b = [s for s in declared if s not in b]
    common = [s for s in declared if s in a and s in b]
    extra_a = sorted(set(a) - set(declared))
    extra_b = sorted(set(b) - set(declared))
    result = {'declared_seeds': declared, 'missing_a': missing_a,
              'missing_b': missing_b, 'extra_a': extra_a, 'extra_b': extra_b,
              'paired_seeds': common, 'n_declared': len(declared),
              'n_available': len(common), 'alpha': alpha,
              'ddof': 1, 'ci': 'two_sided_student_t',
              'effect_size': 'cohen_dz', 'wilcoxon': 'exact_two_sided',
              'zero_method': 'wilcox'}
    status = 'pass_complete' if common == declared and not extra_a and not extra_b \
        else 'incomplete_missing_seeds'
    result['status'] = status
    if len(common) < 2:
        result.update(mean_a=None, mean_b=None, sample_sd_a=None,
                      sample_sd_b=None, mean_paired_difference=None,
                      ci95_difference=None, cohen_dz=None,
                      wilcoxon_statistic=None, p_value=None)
        return result
    av = np.array([a[s] for s in common], dtype=np.float64)
    bv = np.array([b[s] for s in common], dtype=np.float64)
    diff = av - bv
    n = len(diff)
    sd = float(np.std(diff, ddof=1))
    mean = float(np.mean(diff))
    constant_difference = bool(sd <= 1e-12)
    se = 0.0 if constant_difference else sd / math.sqrt(n)
    if se > 0:
        half = float(stats.t.ppf(1 - alpha / 2, n - 1) * se)
        ci = [mean - half, mean + half]
    else:
        ci = [mean, mean]
    nonzero = int(np.count_nonzero(diff))
    if nonzero == 0:
        w_stat, p_value = 0.0, 1.0
    else:
        w = stats.wilcoxon(diff, alternative='two-sided', zero_method='wilcox',
                           correction=False, method='exact')
        w_stat, p_value = float(w.statistic), float(w.pvalue)
    result.update(
        mean_a=float(np.mean(av)), mean_b=float(np.mean(bv)),
        sample_sd_a=float(np.std(av, ddof=1)),
        sample_sd_b=float(np.std(bv, ddof=1)),
        mean_paired_difference=mean, ci95_difference=ci,
        cohen_dz=(mean / sd if not constant_difference else None),
        median_paired_difference=float(np.median(diff)),
        wilcoxon_statistic=w_stat, p_value=p_value,
        effective_nonzero_pairs=nonzero,
        per_seed=[{'seed': s, 'a': a[s], 'b': b[s], 'difference': a[s] - b[s]}
                  for s in common])
    return result


def holm_adjust(p_values):
    """Holm 家族错误率校正；保持输入顺序。"""
    p = np.asarray(p_values, dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('p 值必须为 [0,1] 有限数')
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


# ------------------------------------------------------------------
# P0-3：同骨干、同 logits 的后验校准对照
# ------------------------------------------------------------------
def binary_logits(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim == 2 and z.shape[1] == 1:
        z = z[:, 0]
    if z.ndim == 1:
        return z
    if z.ndim == 2 and z.shape[1] == 2:
        return z[:, 1] - z[:, 0]
    raise ValueError('仅接受 [N]、[N,1] 或 [N,2] 二分类 logits')


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _two_log_probs(logits):
    z = binary_logits(logits)
    two = np.column_stack([np.zeros_like(z), z])
    m = np.max(two, axis=1, keepdims=True)
    e = np.exp(two - m)
    return two - m - np.log(np.sum(e, axis=1, keepdims=True))


def _nll_from_two_logits(two_logits, y):
    m = np.max(two_logits, axis=1)
    logz = m + np.log(np.exp(two_logits - m[:, None]).sum(axis=1))
    logp = two_logits - logz[:, None]
    return float(-np.mean(logp[np.arange(len(y)), y]))


def fit_calibrator(method, logits, y, regularization=1e-3):
    """在独立拟合集上拟合；不接触评估标签。

    temperature: Guo et al. 2017；vector: 二分类仿射缩放；
    dirichlet: Kull et al. 2019 的 log-probability 线性层 + ODIR 正则；
    isotonic: 保序回归。返回纯 JSON 可序列化对象。
    """
    y = np.asarray(y, dtype=np.int64)
    z = binary_logits(logits)
    if len(y) != len(z) or len(y) < 2 or len(np.unique(y)) < 2:
        raise ValueError('校准拟合集需含两个类别')
    if method == 'none':
        return {'method': 'none'}
    if method == 'temperature':
        def objective(v):
            return log_loss(y, _sigmoid(z / np.exp(v[0])), labels=[0, 1])
        opt = optimize.minimize(objective, [0.0], method='L-BFGS-B',
                                bounds=[(-5.0, 5.0)])
        if not opt.success:
            raise RuntimeError(f'temperature 拟合失败: {opt.message}')
        return {'method': method, 'log_temperature': float(opt.x[0]),
                'fit_nll': float(opt.fun)}
    if method == 'vector':
        def objective(v):
            return log_loss(y, _sigmoid(v[0] * z + v[1]), labels=[0, 1]) \
                + regularization * (v[0] - 1.0) ** 2
        opt = optimize.minimize(objective, [1.0, 0.0], method='L-BFGS-B',
                                bounds=[(-10, 10), (-20, 20)])
        if not opt.success:
            raise RuntimeError(f'vector 拟合失败: {opt.message}')
        return {'method': method, 'scale': float(opt.x[0]),
                'bias': float(opt.x[1]), 'fit_objective': float(opt.fun)}
    if method == 'dirichlet':
        lp = _two_log_probs(z)

        def objective(v):
            w = v[:4].reshape(2, 2)
            b = v[4:]
            out = lp @ w.T + b
            off = (w[0, 1] ** 2 + w[1, 0] ** 2)
            diag = ((w[0, 0] - 1) ** 2 + (w[1, 1] - 1) ** 2)
            return _nll_from_two_logits(out, y) + regularization * (
                off + 0.1 * diag + 0.01 * np.sum(b ** 2))
        opt = optimize.minimize(objective, np.array([1., 0., 0., 1., 0., 0.]),
                                method='L-BFGS-B',
                                options={'maxiter': 500})
        if not opt.success:
            raise RuntimeError(f'dirichlet 拟合失败: {opt.message}')
        return {'method': method, 'weights': opt.x[:4].reshape(2, 2).tolist(),
                'bias': opt.x[4:].tolist(), 'regularization': regularization,
                'fit_objective': float(opt.fun)}
    if method == 'isotonic':
        ir = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        ir.fit(_sigmoid(z), y)
        return {'method': method,
                'x_thresholds': np.asarray(ir.X_thresholds_).tolist(),
                'y_thresholds': np.asarray(ir.y_thresholds_).tolist()}
    raise ValueError(f'未知校准方法: {method}')


def apply_calibrator(calibrator, logits):
    method = calibrator['method']
    z = binary_logits(logits)
    if method == 'none':
        return _sigmoid(z)
    if method == 'temperature':
        return _sigmoid(z / np.exp(calibrator['log_temperature']))
    if method == 'vector':
        return _sigmoid(calibrator['scale'] * z + calibrator['bias'])
    if method == 'dirichlet':
        w = np.asarray(calibrator['weights'])
        b = np.asarray(calibrator['bias'])
        out = _two_log_probs(z) @ w.T + b
        m = np.max(out, axis=1, keepdims=True)
        p = np.exp(out - m)
        p /= np.sum(p, axis=1, keepdims=True)
        return p[:, 1]
    if method == 'isotonic':
        return np.interp(_sigmoid(z), calibrator['x_thresholds'],
                         calibrator['y_thresholds'])
    raise ValueError(method)


def expected_calibration_error(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(y[mask]) - np.mean(p[mask]))
    return float(ece)


def reliability_rows(y, p, bins=15):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        rows.append({'bin': b, 'lower': float(edges[b]),
                     'upper': float(edges[b + 1]), 'count': int(mask.sum()),
                     'mean_confidence': (float(np.mean(p[mask])) if np.any(mask) else None),
                     'positive_rate': (float(np.mean(y[mask])) if np.any(mask) else None)})
    return rows


def probability_metrics(y, p, threshold=0.5, bins=15):
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(np.int64)
    return {'f1': float(f1_score(y, pred, zero_division=0)),
            'ece': expected_calibration_error(y, p, bins),
            'brier': float(np.mean((p - y) ** 2)),
            'nll': float(log_loss(y, p, labels=[0, 1])),
            'threshold': float(threshold), 'n': int(len(y)),
            'positive_rate': float(np.mean(y))}


def risk_coverage_rows(y, p, threshold=0.5, coverages=None):
    """按 max(p,1-p) 置信度保留样本；风险定义为 1-F1。"""
    y, p = np.asarray(y), np.asarray(p)
    confidence = np.maximum(p, 1 - p)
    order = np.argsort(-confidence, kind='stable')
    coverages = coverages or [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for coverage in coverages:
        n = max(1, int(math.ceil(len(y) * coverage)))
        idx = order[:n]
        pred = (p[idx] >= threshold).astype(int)
        f1 = float(f1_score(y[idx], pred, zero_division=0))
        rows.append({'coverage': float(coverage), 'n': int(n),
                     'risk_one_minus_f1': 1.0 - f1,
                     'min_retained_confidence': float(confidence[idx].min())})
    return rows


# ------------------------------------------------------------------
# P0-7：严格 inductive rolling 与训练窗预处理
# ------------------------------------------------------------------
def strict_rolling_view(x, edge_index, node_time, cutoff, fit_mask,
                        min_std=1e-6):
    """仅返回 time<=cutoff 的诱导子图；标准化只拟合 fit_mask 训练节点。

    fit_mask 为原图布尔掩码。返回 old_indices/new_edges/mean/scale，调用者可
    验证未来节点、边和统计量均未进入该视图。
    """
    x = np.asarray(x, dtype=np.float64)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    node_time = np.asarray(node_time)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if x.shape[0] != len(node_time) or len(fit_mask) != len(node_time):
        raise ValueError('节点、时间和 fit_mask 长度不一致')
    visible = node_time <= cutoff
    fit = fit_mask & visible
    if not np.array_equal(fit, fit_mask):
        raise ValueError('训练拟合掩码包含 cutoff 之后节点')
    if fit.sum() < 2:
        raise ValueError('训练窗标准化至少需要两个节点')
    old_indices = np.flatnonzero(visible)
    remap = np.full(len(node_time), -1, dtype=np.int64)
    remap[old_indices] = np.arange(len(old_indices))
    edge_visible = visible[edge_index[0]] & visible[edge_index[1]]
    new_edges = remap[edge_index[:, edge_visible]]
    mean = x[fit].mean(axis=0)
    scale = x[fit].std(axis=0, ddof=0)
    scale = np.where(scale < min_std, 1.0, scale)
    x_scaled = (x[old_indices] - mean) / scale
    return {'x': x_scaled, 'edge_index': new_edges,
            'old_indices': old_indices, 'remap': remap,
            'mean': mean, 'scale': scale, 'cutoff': cutoff,
            'visible_edges': int(edge_visible.sum())}


def apply_frozen_scaler(x, mean, scale):
    return (np.asarray(x, dtype=np.float64) - np.asarray(mean)) / np.asarray(scale)


# ------------------------------------------------------------------
# P0-8：显式共享语义与零样本/少样本隔离
# ------------------------------------------------------------------
def semantic_align(source, source_columns, target, target_columns, semantics):
    """按 canonical -> source/target 显式列名对齐，禁止 PCA 代替语义。"""
    if not semantics:
        raise ValueError('共享语义映射不能为空')
    src = np.asarray(source)
    tgt = np.asarray(target)
    s_index = {c: i for i, c in enumerate(source_columns)}
    t_index = {c: i for i, c in enumerate(target_columns)}
    if len(s_index) != len(source_columns) or len(t_index) != len(target_columns):
        raise ValueError('特征列名不得重复')
    names = list(semantics)
    if len(set(names)) != len(names):
        raise ValueError('canonical 语义不得重复')
    s_cols, t_cols = [], []
    for name in names:
        rule = semantics[name]
        if not isinstance(rule, dict) or 'source' not in rule or 'target' not in rule:
            raise ValueError(f'{name}: 必须同时声明 source 与 target 列')
        if rule['source'] not in s_index:
            raise ValueError(f'{name}: 源列不存在 {rule["source"]}')
        if rule['target'] not in t_index:
            raise ValueError(f'{name}: 目标列不存在 {rule["target"]}')
        s_cols.append(s_index[rule['source']])
        t_cols.append(t_index[rule['target']])
    try:
        xs = src[:, s_cols].astype(np.float64)
        xt = tgt[:, t_cols].astype(np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(f'共享特征必须为数值列: {e}') from e
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(xt)):
        raise ValueError('共享特征含 NaN/Inf；需预先按训练窗规则处理')
    manifest = {'canonical_features': names,
                'source_columns': [source_columns[i] for i in s_cols],
                'target_columns': [target_columns[i] for i in t_cols],
                'alignment': 'explicit_named_semantics',
                'pca_used': False}
    return xs, xt, manifest


def map_labels(labels, mapping, canonical_order=('normal', 'positive')):
    """将原始字符串标签显式映射为 canonical id；未知标签拒绝静默忽略。"""
    inverse = {canonical: i for i, canonical in enumerate(canonical_order)}
    out = np.full(len(labels), -1, dtype=np.int64)
    for raw, canonical in mapping.items():
        if canonical not in inverse:
            raise ValueError(f'未知 canonical 标签: {canonical}')
        out[np.asarray(labels) == raw] = inverse[canonical]
    return out


def deterministic_few_shot(labels, eligible_mask, n_per_class, selection_seed,
                           node_ids=None):
    """仅从预声明目标训练池选少样本；以哈希排序，与测试表现无关。"""
    y = np.asarray(labels)
    eligible = np.asarray(eligible_mask, dtype=bool)
    ids = np.arange(len(y)) if node_ids is None else np.asarray(node_ids)
    selected = []
    for cls in sorted(np.unique(y[eligible])):
        if cls < 0:
            continue
        candidates = np.flatnonzero(eligible & (y == cls))
        ranked = sorted(candidates, key=lambda i: sha256_bytes(
            f'{selection_seed}|{ids[i]}'.encode()))
        if len(ranked) < n_per_class:
            raise ValueError(f'类别 {cls} 可用目标训练样本不足 {n_per_class}')
        selected.extend(ranked[:n_per_class])
    return np.array(sorted(selected), dtype=np.int64)
