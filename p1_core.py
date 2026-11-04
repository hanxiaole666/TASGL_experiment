"""P1 公共原语：完整指标、案例审计、成本与环境证据。

本模块不预设性能结论。最终测试指标和案例只能由 P0 锁后评估调用；
开发预测可生成诊断，但不得被重新命名为独立确认。
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, auc, confusion_matrix,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from metrics import recall_at_fpr
from p0_core import (atomic_json, canonical_json, file_sha256,
                     probability_metrics, read_json, sha256_bytes)


def _binary_inputs(y, p):
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or not len(y):
        raise ValueError('y 与 p_prob 必须为等长非空向量')
    if np.any((y != 0) & (y != 1)):
        raise ValueError('P1 二分类指标只接受 0/1 标签')
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('p_prob 必须为 [0,1] 有限数')
    return y, p


def extended_binary_metrics(y, p, threshold=0.5,
                            fpr_targets=(0.01, 0.05, 0.10), bins=15):
    """保留旧曲线面积定义，并增加 AP、ROC-AUC、混淆矩阵和多 FPR 点。"""
    y, p = _binary_inputs(y, p)
    if not 0 <= threshold <= 1:
        raise ValueError('threshold 必须位于 [0,1]，且须事先由验证集确定')
    if len(np.unique(y)) < 2:
        raise ValueError('排序指标要求评估标签同时包含两类')
    out = probability_metrics(y, p, threshold=threshold, bins=bins)
    pred = (p >= threshold).astype(np.int64)
    precision, recall, _ = precision_recall_curve(y, p)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr_targets = tuple(float(v) for v in fpr_targets)
    if not fpr_targets or any(v <= 0 or v > 1 for v in fpr_targets):
        raise ValueError('fpr_targets 必须位于 (0,1]')
    recalls = {str(v): float(recall_at_fpr(y, p, v)) for v in fpr_targets}
    out.update({
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'pr_auc': float(auc(recall, precision)),
        'pr_auc_curve': float(auc(recall, precision)),
        'average_precision': float(average_precision_score(y, p)),
        'roc_auc': float(roc_auc_score(y, p)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'recall_at_fpr': recalls,
        'recall_at_fpr_1pct': recalls[str(fpr_targets[0])]
            if math.isclose(fpr_targets[0], 0.01) else
            float(recall_at_fpr(y, p, 0.01)),
        'fpr_targets': list(fpr_targets),
    })
    return out


def metric_definitions():
    return {
        'pr_auc_curve': 'auc(recall, precision) after sklearn precision_recall_curve; not average precision',
        'average_precision': 'sklearn average_precision_score',
        'recall_at_fpr': 'legacy rank-traversal operating-point implementation in metrics.py; ties are not a new threshold-tuning rule',
        'threshold_policy': 'frozen before evaluation; never optimized on evaluation labels',
        'confusion_matrix': 'labels=[0,1], positive class=1',
    }


def build_case_audit(y, p, node_ids=None, node_times=None, threshold=0.5,
                     model='unknown', seed=0, condition='clean',
                     clean_p=None, include_correct=False):
    """生成 FP/FN 与可选对照案例；没有外部标注时保持待审查。"""
    y, p = _binary_inputs(y, p)
    n = len(y)
    ids = np.arange(n) if node_ids is None else np.asarray(node_ids).reshape(-1)
    times = np.full(n, -1) if node_times is None else np.asarray(node_times).reshape(-1)
    if len(ids) != n or len(times) != n:
        raise ValueError('案例节点 ID／时间与预测长度不一致')
    pred = (p >= threshold).astype(np.int64)
    clean = None if clean_p is None else _binary_inputs(y, clean_p)[1]
    clean_pred = None if clean is None else (clean >= threshold).astype(np.int64)
    rows = []
    for i in range(n):
        error = { (0, 0): 'TN', (0, 1): 'FP',
                  (1, 0): 'FN', (1, 1): 'TP' }[(int(y[i]), int(pred[i]))]
        if not include_correct and error in ('TN', 'TP'):
            continue
        transition = 'not_applicable'
        if clean_pred is not None:
            was_error = clean_pred[i] != y[i]
            is_error = pred[i] != y[i]
            transition = {(False, True): 'correct_to_error',
                          (True, False): 'error_to_correct',
                          (True, True): 'persistent_error',
                          (False, False): 'persistent_correct'}[(was_error, is_error)]
        case_id = sha256_bytes(
            f'{model}|{seed}|{condition}|{ids[i]}|{times[i]}'.encode())[:20]
        rows.append({
            'case_id': case_id, 'model': model, 'seed': int(seed),
            'condition': condition, 'node_id': str(ids[i]),
            'node_time': int(times[i]), 'true_label': int(y[i]),
            'predicted_label': int(pred[i]), 'p_illicit': float(p[i]),
            'threshold': float(threshold),
            'confidence': float(max(p[i], 1 - p[i])), 'error_type': error,
            'clean_p_illicit': None if clean is None else float(clean[i]),
            'clean_predicted_label': None if clean_pred is None else int(clean_pred[i]),
            'failure_transition': transition,
            'review_status': 'pending_external_review',
            'reviewer_alias': '', 'pattern_code': '', 'assessment': '',
            'evidence_uri': '', 'notes': '',
        })
    counts = Counter((r['condition'], r['node_time'], r['error_type'],
                      r['failure_transition']) for r in rows)
    summary = [{'condition': k[0], 'node_time': k[1], 'error_type': k[2],
                'failure_transition': k[3], 'count': v}
               for k, v in sorted(counts.items())]
    return rows, summary


def _atomic_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ['case_id']
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp',
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_case_audit(output_dir, rows, summary, metadata=None):
    out = Path(output_dir)
    fields = list(rows[0]) if rows else [
        'case_id', 'model', 'seed', 'condition', 'node_id', 'node_time',
        'true_label', 'predicted_label', 'p_illicit', 'threshold',
        'confidence', 'error_type', 'clean_p_illicit',
        'clean_predicted_label', 'failure_transition', 'review_status',
        'reviewer_alias', 'pattern_code', 'assessment', 'evidence_uri', 'notes']
    _atomic_csv(out / 'case_audit_queue.csv', rows, fields)
    result = {'protocol': 'p1_case_audit_v1', 'n_cases': len(rows),
              'external_validation_status': 'pending_external_annotations',
              'attention_heatmap_is_not_external_validation': True,
              'summary': summary, 'metadata': metadata or {}}
    atomic_json(out / 'case_audit_summary.json', result)
    return result


def merge_case_annotations(queue_path, annotations_path, output_dir):
    """合并真实外部标注；程序只校验结构，不冒充专家裁决。"""
    with open(queue_path, encoding='utf-8', newline='') as f:
        queue = list(csv.DictReader(f))
    with open(annotations_path, encoding='utf-8', newline='') as f:
        annotations = list(csv.DictReader(f))
    required = {'case_id', 'reviewer_alias', 'pattern_code', 'assessment'}
    allowed = {'confirmed_model_error', 'explained_by_data', 'uncertain',
               'not_reviewed'}
    by_case = {}
    for row in annotations:
        missing = required - set(row)
        if missing:
            raise ValueError(f'外部标注缺列: {sorted(missing)}')
        if not row['reviewer_alias'].strip() or '@' in row['reviewer_alias']:
            raise ValueError('reviewer_alias 必须为匿名别名，不得为空或为邮箱')
        if row['assessment'] not in allowed:
            raise ValueError(f'未知 assessment: {row["assessment"]}')
        if row['assessment'] != 'not_reviewed' and not row['pattern_code'].strip():
            raise ValueError('已审查案例必须提供 pattern_code')
        key = (row['case_id'], row['reviewer_alias'])
        if key in by_case:
            raise ValueError(f'重复外部标注: {key}')
        by_case[key] = row
    known = {r['case_id'] for r in queue}
    unknown = {r['case_id'] for r in annotations} - known
    if unknown:
        raise ValueError('外部标注包含队列外 case_id')
    merged = []
    for row in queue:
        matches = [a for (cid, _), a in by_case.items() if cid == row['case_id']]
        item = dict(row)
        if matches:
            item['review_status'] = 'external_annotations_present'
            item['reviewer_alias'] = ';'.join(sorted(a['reviewer_alias'] for a in matches))
            item['pattern_code'] = ';'.join(a['pattern_code'] for a in matches)
            item['assessment'] = ';'.join(a['assessment'] for a in matches)
            item['evidence_uri'] = ';'.join(a.get('evidence_uri', '') for a in matches)
            item['notes'] = ';'.join(a.get('notes', '') for a in matches)
        merged.append(item)
    out = Path(output_dir)
    _atomic_csv(out / 'case_audit_annotated.csv', merged,
                list(queue[0]) if queue else None)
    result = {
        'protocol': 'p1_external_case_annotation_v1', 'n_cases': len(queue),
        'n_annotations': len(annotations),
        'n_cases_with_annotations': sum(r['review_status'] ==
                                        'external_annotations_present' for r in merged),
        'n_reviewer_aliases': len({r['reviewer_alias'] for r in annotations}),
        'status': ('external_annotations_present_requires_scientific_adjudication'
                   if annotations else 'pending_external_annotations'),
        'automatic_expert_validation_claimed': False,
        'annotation_sha256': file_sha256(annotations_path),
    }
    atomic_json(out / 'annotation_audit.json', result)
    return result


def relative_f1_drop(clean_f1, attacked_f1):
    clean_f1, attacked_f1 = float(clean_f1), float(attacked_f1)
    if clean_f1 <= 0:
        return None
    return (clean_f1 - attacked_f1) / clean_f1


def h3_drop_decision(student_drop, baseline_drop):
    """基线非正下降必须不确定，不能据负下降制造学生优势。"""
    if student_drop is None or baseline_drop is None:
        return 'inconclusive_undefined_drop'
    student_drop, baseline_drop = float(student_drop), float(baseline_drop)
    if baseline_drop <= 0:
        return 'inconclusive_nonpositive_baseline_drop'
    if student_drop <= 0.6 * baseline_drop:
        return 'pass'
    if student_drop >= baseline_drop:
        return 'falsified'
    return 'inconclusive_between_bounds'


def summarize_cost_events(events, student_params=None, teacher_params=None,
                          calibrator_params=None):
    allowed = {'data_loading', 'preprocessing', 'graph_construction',
               'teacher_training', 'student_distillation', 'calibration',
               'threshold_selection', 'student_inference', 'model_update',
               'update_calibration', 'update_threshold', 'evaluation'}
    phases = {}
    for event in events:
        phase = event['phase']
        seconds = float(event['seconds'])
        if phase not in allowed or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'非法成本事件: {event}')
        phases.setdefault(phase, []).append(seconds)
    phase_seconds = {k: {'total_seconds': float(sum(v)), 'n_invocations': len(v),
                         'values_seconds': [float(x) for x in v]}
                     for k, v in sorted(phases.items())}
    initial = {'data_loading', 'preprocessing', 'graph_construction',
               'teacher_training', 'student_distillation', 'calibration',
               'threshold_selection'}
    update = {'model_update', 'update_calibration', 'update_threshold'}
    required = {'teacher_training', 'student_distillation', 'calibration',
                'student_inference'}
    return {
        'protocol': 'p1_teacher_student_full_cost_v1',
        'phase_seconds': phase_seconds,
        'initial_offline_system_seconds': float(sum(
            s for p, values in phases.items() if p in initial for s in values)),
        'update_system_seconds': float(sum(
            s for p, values in phases.items() if p in update for s in values)),
        'deployment_parameters': {'student': student_params,
                                  'calibrator': calibrator_params},
        'offline_teacher_parameters': teacher_params,
        'teacher_excluded_from_student_parameter_count': True,
        'teacher_included_in_offline_system_cost': 'teacher_training' in phases,
        'missing_required_phases': sorted(required - set(phases)),
        'complete': not bool(required - set(phases)),
    }


def environment_snapshot():
    """完整科学环境指纹；不采集主机名、用户名或凭据。"""
    packages = {}
    for name in ('torch', 'numpy', 'pandas', 'scikit-learn', 'scipy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu_model = platform.processor() or None
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        match = re.search(r'^model name\s*:\s*(.+)$', cpuinfo.read_text(
            encoding='utf-8', errors='ignore'), re.MULTILINE)
        if match:
            cpu_model = match.group(1).strip()
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            gpus.append({'index': i, 'name': torch.cuda.get_device_name(i),
                         'total_memory_bytes': int(prop.total_memory),
                         'compute_capability': list(torch.cuda.get_device_capability(i))})
    driver = None
    try:
        proc = subprocess.run([
            'nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5, check=False)
        if proc.returncode == 0:
            driver = proc.stdout.strip().splitlines()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        'protocol': 'p1_environment_fingerprint_v1',
        'captured_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'os': {'system': platform.system(), 'release': platform.release(),
               'version': platform.version(), 'machine': platform.machine()},
        'python': sys.version.split()[0], 'packages': packages,
        'cpu': {'model': cpu_model, 'logical_count': os.cpu_count()},
        'gpu': gpus, 'nvidia_driver_versions': driver,
        'cuda_runtime': torch.version.cuda,
        'cudnn_version': (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
        'determinism': {
            'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'known_nondeterminism_sources': [
                'sparse scatter/atomic operations where applicable',
                'cuDNN/CUDA kernel selection and hardware dependence',
                'CPU thread scheduling; bitwise reproducibility is not claimed'],
        },
        'privacy': 'hostname, username, home paths and credentials intentionally omitted',
    }


def compare_environment(reference, current):
    keys = [('python',), ('os', 'system'), ('os', 'release'),
            ('packages', 'torch'), ('packages', 'numpy'), ('cuda_runtime',),
            ('cudnn_version',)]
    mismatches = []
    for path in keys:
        a, b = reference, current
        for key in path:
            a, b = a.get(key), b.get(key)
        if a != b:
            mismatches.append({'field': '.'.join(path), 'reference': a, 'current': b})
    ref_gpus = [g.get('name') for g in reference.get('gpu', [])]
    cur_gpus = [g.get('name') for g in current.get('gpu', [])]
    if ref_gpus != cur_gpus:
        mismatches.append({'field': 'gpu.names', 'reference': ref_gpus,
                           'current': cur_gpus})
    return {'compatible': not mismatches, 'mismatches': mismatches,
            'note': '环境差异是可复现性风险，不得通过删除差异字段宣称一致'}
