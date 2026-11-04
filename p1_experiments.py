"""P1-1～P1-8 统一实验入口。

开发诊断默认写入 results/p1/；最终测试只能经 p0_experiments.py
independent lock/evaluate 产生。程序成功不等于审稿证据已经完整。
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import set_seed, train_model
from data_loader import compute_time_gaps
from models import GraphSAGEBaseline
from models_v3 import TASGLite
from p0_core import (apply_calibrator, atomic_json, canonical_json, file_sha256,
                     fit_calibrator, holm_adjust, paired_statistics, read_json,
                     sha256_bytes)
from p0_data import validate_prediction_npz
from p0_experiments import (_build_hpo_model, _guard_invocation,
                            _load_hpo_graph, _tune_probability_threshold)
from p1_attacks import apply_nonoracle_attack, validate_attack_matrix
from p1_core import (build_case_audit, compare_environment,
                     environment_snapshot, extended_binary_metrics, h3_drop_decision,
                     merge_case_annotations, metric_definitions,
                     relative_f1_drop, summarize_cost_events, write_case_audit)
from p1_release import stage_release, verify_release
from p1_system import (benchmark_inference, cold_start_summary,
                       load_development_graph_timed, sync)
from v3_engine import train_lite

ROOT = Path(__file__).resolve().parent
P1_ROOT = ROOT / 'results' / 'p1'
MODELS = ['tasgl', 'mlp', 'gcn', 'gat', 'graphsage', 'sgc', 'sign',
          'temporal_graphsage']


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe(name):
    return re.sub(r'[^A-Za-z0-9_.=-]+', '_', str(name))


def _require_unique(name, values):
    if len(set(values)) != len(values):
        raise ValueError(f'{name} 不得重复；重复项不能冒充独立重复')


def _data_hashes(data_dir):
    return {name: file_sha256(Path(data_dir) / name) for name in
            ('elliptic_txs_features.csv', 'elliptic_txs_classes.csv',
             'elliptic_txs_edgelist.csv')}


def _binary_logits(model, data):
    model.eval()
    with torch.no_grad():
        logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                          data['num_nodes'], hard_mask=True)
    return (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(-z)))


class BinaryVectorCalibratedInference(nn.Module):
    """two-logit 实现 z0'=a*z0, z1'=a*z1+b；包含在部署前向中。"""
    def __init__(self, model, scale, bias):
        super().__init__()
        self.model = model
        self.scale, self.bias = float(scale), float(bias)

    def forward(self, *args, **kwargs):
        logits, aux = self.model(*args, **kwargs)
        calibrated = torch.stack([
            logits[:, 0] * self.scale,
            logits[:, 1] * self.scale + self.bias], dim=1)
        return calibrated, aux


def _fit_validation_calibrator(binary, labels, val_mask):
    val = np.asarray(val_mask, dtype=bool)
    calibrator = fit_calibrator('vector', binary[val], labels[val])
    p_val = apply_calibrator(calibrator, binary[val])
    threshold, val_f1 = _tune_probability_threshold(labels[val], p_val)
    calibrator.update(threshold=float(threshold), validation_f1=float(val_f1),
                      fitting_data='development_validation_only')
    return calibrator


def _apply_binary_calibrator(calibrator, binary):
    return _sigmoid(calibrator['scale'] * binary + calibrator['bias'])


def _load_labeled_predictions(path):
    with np.load(path, allow_pickle=False) as z:
        if 'y' not in z.files:
            raise ValueError('开发指标输入必须含 y；最终无标签预测须走 P0 锁后评估')
        y = np.asarray(z['y'], dtype=np.int64).reshape(-1)
        if 'p_prob' in z.files:
            p = np.asarray(z['p_prob'], dtype=np.float64).reshape(-1)
        elif 'logits' in z.files:
            logits = np.asarray(z['logits'], dtype=np.float64)
            binary = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits
            p = _sigmoid(binary)
        else:
            raise ValueError('输入必须含 p_prob 或 logits')
        ids = (np.asarray(z['node_id']) if 'node_id' in z.files
               else np.arange(len(y)))
        times = (np.asarray(z['node_time']) if 'node_time' in z.files
                 else np.full(len(y), -1))
        seed = int(np.asarray(z['seed']).reshape(-1)[0]) if 'seed' in z.files else 0
    if not (len(y) == len(p) == len(ids) == len(times)):
        raise ValueError('预测、标签、节点 ID 和时间长度不一致')
    return {'y': y, 'p': p, 'node_ids': ids, 'node_times': times, 'seed': seed}


def cmd_metrics(args):
    if args.evaluation_role != 'development':
        raise ValueError('P1 指标 CLI 只接收开发预测；最终指标必须经 P0 锁后评估')
    pred = _load_labeled_predictions(args.input)
    out = Path(args.output_dir)
    _guard_invocation(out, {'input_sha256': file_sha256(args.input),
                            'threshold': args.threshold,
                            'threshold_source': args.threshold_source,
                            'fpr_targets': args.fpr_targets,
                            'evaluation_role': args.evaluation_role})
    path = out / 'extended_metrics.json'
    if path.exists():
        print('[P1-8] 同配置完整指标已完成，跳过')
        return 0
    metrics = extended_binary_metrics(pred['y'], pred['p'], args.threshold,
                                      tuple(args.fpr_targets))
    result = {'protocol': 'p1_extended_metrics_v1', 'metrics': metrics,
              'definitions': metric_definitions(), 'seed': pred['seed'],
              'threshold_source': args.threshold_source,
              'evaluation_role': args.evaluation_role,
              'test_threshold_optimization': False,
              'input_sha256': file_sha256(args.input)}
    atomic_json(path, result)
    print(f'[P1-8] AP={metrics["average_precision"]:.4f}; '
          f'ROC-AUC={metrics["roc_auc"]:.4f}; 输出={out}')
    return 0


def cmd_audit_cases(args):
    pred = _load_labeled_predictions(args.input)
    clean_p = None
    if args.clean_input:
        clean = _load_labeled_predictions(args.clean_input)
        if not np.array_equal(pred['node_ids'], clean['node_ids']):
            raise ValueError('clean_input 与攻击输入必须按相同节点顺序对齐')
        clean_p = clean['p']
    rows, summary = build_case_audit(
        pred['y'], pred['p'], pred['node_ids'], pred['node_times'],
        threshold=args.threshold, model=args.model, seed=args.seed,
        condition=args.condition, clean_p=clean_p,
        include_correct=args.include_correct)
    out = Path(args.output_dir)
    _guard_invocation(out, {'input_sha256': file_sha256(args.input),
                            'clean_sha256': (file_sha256(args.clean_input)
                                             if args.clean_input else None),
                            'threshold': args.threshold, 'model': args.model,
                            'seed': args.seed, 'condition': args.condition,
                            'include_correct': args.include_correct})
    if (out / 'case_audit_summary.json').exists():
        print('[P1-1] 同配置案例队列已完成，跳过')
        return 0
    result = write_case_audit(out, rows, summary, {
        'evaluation_role': 'development', 'model': args.model,
        'condition': args.condition, 'seed': args.seed,
        'external_pattern_labels_or_expert_review_required': True})
    print(f'[P1-1] 案例={result["n_cases"]}；状态=pending_external_annotations')
    return 0


def cmd_review_cases(args):
    out = Path(args.output_dir)
    _guard_invocation(out, {'queue_sha256': file_sha256(args.queue),
                            'annotations_sha256': file_sha256(args.annotations)})
    if (out / 'annotation_audit.json').exists():
        print('[P1-1] 同标注审计已完成，跳过')
        return 0
    result = merge_case_annotations(args.queue, args.annotations, out)
    print(f'[P1-1] 外部标注={result["n_annotations"]}；'
          '程序未自动宣称专家验证')
    return 0


def _prediction_payload(pred, p, seed):
    return {'node_id': pred['node_ids'], 'node_time': pred['node_times'],
            'p_prob': np.asarray(p, dtype=np.float64), 'y': pred['y'],
            'seed': np.array([seed], dtype=np.int64)}


def _flat_metric_row(row):
    skip = {'attack_manifest', 'recall_at_fpr', 'fpr_targets'}
    return {k: v for k, v in row.items() if k not in skip and
            not isinstance(v, (dict, list))}


def cmd_attacks(args):
    from p1_core import extended_binary_metrics
    config = read_json(args.attack_config)
    validate_attack_matrix(config)
    attack_seeds = args.attack_seeds or config['attack_seeds']
    _require_unique('models', args.models); _require_unique('seeds', args.seeds)
    _require_unique('attack_seeds', attack_seeds)
    if not set(attack_seeds).issubset(set(config['attack_seeds'])):
        raise ValueError('不得在未重新预注册的情况下增加 attack_seeds')
    out = Path(args.output_dir)
    winners_path = Path(args.hpo_dir) / 'hpo_winners.json'
    winners = read_json(winners_path)
    data_hashes = _data_hashes(args.data_dir)
    if winners.get('data_sha256') != data_hashes:
        raise RuntimeError('HPO winner 与当前原始数据哈希不一致；禁止跨数据续用')
    declared_model_seeds = {name: winners['models'].get(name, {}).get('seeds', [])
                            for name in args.models}
    model_seed_complete = all(set(args.seeds) == set(declared)
                              for declared in declared_model_seeds.values())
    attack_seed_complete = set(attack_seeds) == set(config['attack_seeds'])
    guard = {'data_sha256': data_hashes,
             'winners_sha256': file_sha256(winners_path),
             'attack_config_sha256': file_sha256(args.attack_config),
             'models': args.models, 'seeds': args.seeds,
             'attack_seeds': attack_seeds, 'eval': [args.eval_start, args.eval_end],
             'device': args.device}
    _guard_invocation(out, guard)
    summary_path = out / 'attack_summary.json'
    if summary_path.exists():
        print('[P1-2/P1-3] 同配置攻击矩阵已完成，保留结果并跳过')
        return 0
    for name in args.models:
        if name not in winners['models']:
            raise KeyError(f'HPO winner 不存在: {name}')
    device = torch.device(args.device)
    data, _, val_mask = _load_hpo_graph(args.data_dir, device)
    x_cpu = data['x'].detach().cpu().numpy()
    edge_cpu = data['edge_index'].detach().cpu().numpy()
    time_np = data['time_steps'].detach().cpu().numpy()
    labels = data['y'].detach().cpu().numpy()
    ids = pd.read_csv(Path(args.data_dir) / 'elliptic_txs_features.csv',
                      header=None, usecols=[0]).iloc[:, 0].to_numpy()
    eval_graph = (time_np >= args.eval_start) & (time_np <= args.eval_end)
    eval_labeled = eval_graph & (labels >= 0)
    if len(np.unique(labels[eval_labeled])) < 2:
        raise ValueError('开发攻击评估窗口需含两类标签')
    pred = {'y': labels[eval_labeled], 'node_ids': ids[eval_labeled],
            'node_times': time_np[eval_labeled]}
    val_np = val_mask.detach().cpu().numpy()
    run_dir = out / 'runs'; pred_dir = out / 'predictions'; case_dir = out / 'cases'
    for directory in (run_dir, pred_dir, case_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in args.models:
        for seed in args.seeds:
            winner = winners['models'][model_name]['by_seed'].get(str(seed))
            if winner is None:
                raise ValueError(f'{model_name} 缺预声明模型种子 {seed}')
            set_seed(seed)
            model = _build_hpo_model(model_name, data['in_dim'], winner['config'], data)
            state = torch.load(winner['state_file'], map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            model.to(device).eval()
            clean_binary = _binary_logits(model, data)
            calibrator = _fit_validation_calibrator(clean_binary, labels, val_np)
            clean_p = _apply_binary_calibrator(calibrator, clean_binary)
            clean_metrics = extended_binary_metrics(
                labels[eval_labeled], clean_p[eval_labeled], calibrator['threshold'])
            clean_stem = f'{_safe(model_name)}_seed{seed}_clean'
            clean_row = {'protocol': 'p1_nonoracle_attack_eval_v1',
                         'model': model_name, 'seed': int(seed),
                         'attack_name': 'clean', 'attack_seed': -1,
                         'source_run_protocol': winner['run_protocol'],
                         'threshold': calibrator['threshold'],
                         'test_labels_35_49_used': False, **clean_metrics}
            clean_path = run_dir / f'{clean_stem}.json'
            clean_pred_path = pred_dir / f'{clean_stem}.npz'
            clean_case_path = case_dir / clean_stem / 'case_audit_summary.json'
            if clean_path.exists() and clean_pred_path.exists() and clean_case_path.exists():
                clean_row = read_json(clean_path)
            else:
                np.savez_compressed(clean_pred_path,
                                    **_prediction_payload(pred, clean_p[eval_labeled], seed))
                case_rows, case_summary = build_case_audit(
                    pred['y'], clean_p[eval_labeled], pred['node_ids'],
                    pred['node_times'], calibrator['threshold'], model_name,
                    seed, 'clean')
                write_case_audit(case_dir / clean_stem, case_rows, case_summary,
                                 {'evaluation_role': 'development'})
                atomic_json(clean_path, clean_row)
            rows.append(clean_row)
            for spec in config['attacks']:
                for attack_seed in attack_seeds:
                    stem = f'{_safe(model_name)}_seed{seed}_{_safe(spec["name"])}_attackseed{attack_seed}'
                    row_path = run_dir / f'{stem}.json'
                    prediction_path = pred_dir / f'{stem}.npz'
                    case_summary_path = case_dir / stem / 'case_audit_summary.json'
                    if row_path.exists():
                        row = read_json(row_path)
                        if row.get('source_run_protocol') != winner['run_protocol']:
                            raise RuntimeError('攻击断点属于不同 HPO winner，拒绝续写')
                        if prediction_path.exists() and case_summary_path.exists():
                            rows.append(row)
                            continue
                        print(f'[P1-3] 补齐同配置中断产物: {stem}')
                    # 关键顺序：先只用图、特征和干净预测冻结攻击，再读取开发标签计算指标。
                    ax, ae, manifest = apply_nonoracle_attack(
                        x_cpu, edge_cpu, time_np, eval_graph, clean_p, spec, attack_seed)
                    edge_t = torch.as_tensor(ae, dtype=torch.long)
                    gaps = compute_time_gaps(edge_t, time_np)
                    attacked = dict(data)
                    attacked['x'] = torch.as_tensor(ax, dtype=torch.float32, device=device)
                    attacked['edge_index'] = edge_t.to(device)
                    attacked['time_gaps'] = gaps.to(device)
                    binary = _binary_logits(model, attacked)
                    p = _apply_binary_calibrator(calibrator, binary)
                    metrics = extended_binary_metrics(
                        labels[eval_labeled], p[eval_labeled], calibrator['threshold'])
                    row = {'protocol': 'p1_nonoracle_attack_eval_v1',
                           'model': model_name, 'seed': int(seed),
                           'attack_name': spec['name'], 'attack_seed': int(attack_seed),
                           'source_run_protocol': winner['run_protocol'],
                           'threshold': calibrator['threshold'],
                           'test_labels_35_49_used': False,
                           'attack_manifest': manifest, **metrics}
                    np.savez_compressed(prediction_path,
                                        **_prediction_payload(pred, p[eval_labeled], seed))
                    case_rows, case_summary = build_case_audit(
                        pred['y'], p[eval_labeled], pred['node_ids'],
                        pred['node_times'], calibrator['threshold'], model_name,
                        seed, spec['name'], clean_p=clean_p[eval_labeled])
                    write_case_audit(case_dir / stem, case_rows, case_summary,
                                     {'evaluation_role': 'development',
                                      'attack_seed': attack_seed,
                                      'attack_graph_sha256': manifest['edge_sha256']})
                    atomic_json(row_path, row)
                    rows.append(row)
                    print(f'[P1-3] {model_name} seed={seed} {spec["name"]} '
                          f'attack_seed={attack_seed} F1={row["f1"]:.4f}')
    lookup = {(r['model'], r['seed'], r['attack_name'], r['attack_seed']): r
              for r in rows}
    comparisons = []
    if 'tasgl' in args.models:
        for baseline in args.models:
            if baseline == 'tasgl':
                continue
            for spec in config['attacks']:
                for attack_seed in attack_seeds:
                    per_seed, a_records, b_records = [], [], []
                    for seed in args.seeds:
                        sa = lookup[('tasgl', seed, spec['name'], attack_seed)]
                        ba = lookup[(baseline, seed, spec['name'], attack_seed)]
                        sc = lookup[('tasgl', seed, 'clean', -1)]
                        bc = lookup[(baseline, seed, 'clean', -1)]
                        sd = relative_f1_drop(sc['f1'], sa['f1'])
                        bd = relative_f1_drop(bc['f1'], ba['f1'])
                        decision = h3_drop_decision(sd, bd)
                        per_seed.append({'seed': seed, 'student_drop': sd,
                                         'baseline_drop': bd, 'decision': decision})
                        a_records.append({'seed': seed, 'f1': sa['f1']})
                        b_records.append({'seed': seed, 'f1': ba['f1']})
                    stat = paired_statistics(a_records, b_records, args.seeds)
                    decisions = [r['decision'] for r in per_seed]
                    comparisons.append({
                        'student': 'tasgl', 'baseline': baseline,
                        'attack_name': spec['name'], 'attack_seed': attack_seed,
                        'per_seed': per_seed, 'paired_attacked_f1': stat,
                        'joint_h3_status': (
                            'incomplete_missing_model_seeds' if not model_seed_complete else
                            'pass' if all(d == 'pass' for d in decisions) else 'not_pass'),
                        'inconclusive_nonpositive_baseline_seeds': [
                            r['seed'] for r in per_seed if r['decision'] ==
                            'inconclusive_nonpositive_baseline_drop']})
    complete_p = [c['paired_attacked_f1']['p_value'] for c in comparisons
                  if c['paired_attacked_f1']['status'] == 'pass_complete']
    adjusted = holm_adjust(complete_p) if complete_p else []
    j = 0
    for comp in comparisons:
        if comp['paired_attacked_f1']['status'] == 'pass_complete':
            comp['holm_p_within_attack_family'] = adjusted[j]; j += 1
        else:
            comp['holm_p_within_attack_family'] = None
    result = {'protocol': 'p1_nonoracle_attack_matrix_v1',
              'status': ('complete_declared_matrix'
                         if model_seed_complete and attack_seed_complete else
                         'incomplete_declared_seed_subset'),
              'declared_model_seeds_by_model': declared_model_seeds,
              'executed_model_seeds': args.seeds,
              'declared_attack_seeds': config['attack_seeds'],
              'executed_attack_seeds': attack_seeds,
              'created_utc': _now(), 'development_only': True,
              'evaluation_window': [args.eval_start, args.eval_end],
              'test_labels_35_49_used': False,
              'future_labels_35_49_masked': True,
              'graph_visibility': 'transductive_full_graph_not_strict_rolling',
              'future_graph_structure_may_be_visible': True,
              'attack_construction_label_access': False,
              'model_seeds_are_paired_units': True,
              'attack_seeds_are_nested_repeats_not_independent_model_seeds': True,
              'oracle_historical_results_not_merged': True,
              'attack_config_sha256': file_sha256(args.attack_config),
              'n_runs': len(rows), 'runs': rows, 'h3_comparisons': comparisons}
    atomic_json(summary_path, result)
    pd.DataFrame([_flat_metric_row(r) for r in rows]).to_csv(
        out / 'attack_metrics.csv', index=False)
    print(f'[P1-2/P1-3] 完成 {len(rows)} 条开发攻击记录；输出={out}')
    return 0


def _winner_for(args, seed):
    winners = read_json(Path(args.hpo_dir) / 'hpo_winners.json')
    if args.model not in winners['models']:
        raise KeyError(f'HPO winner 不存在: {args.model}')
    row = winners['models'][args.model]['by_seed'].get(str(seed))
    if row is None:
        raise ValueError(f'缺 HPO winner seed={seed}')
    return row


def cmd_efficiency(args):
    _require_unique('seeds', args.seeds); _require_unique('devices', args.devices)
    _require_unique('batch_sizes', args.batch_sizes)
    out = Path(args.output_dir)
    data_hashes = _data_hashes(args.data_dir)
    winners = read_json(Path(args.hpo_dir) / 'hpo_winners.json')
    if winners.get('data_sha256') != data_hashes:
        raise RuntimeError('HPO winner 与当前原始数据哈希不一致；禁止跨数据续用')
    _guard_invocation(out, {'data_sha256': data_hashes,
                            'hpo_dir': str(Path(args.hpo_dir).resolve()),
                            'model': args.model, 'seeds': args.seeds,
                            'devices': args.devices, 'batch_sizes': args.batch_sizes,
                            'warmup': args.warmup, 'repeats': args.repeats,
                            'include_calibration': not args.skip_calibration})
    final_path = out / 'efficiency_summary.json'
    if final_path.exists():
        print('[P1-5] 同配置端到端效率审计已完成，跳过')
        return 0
    rows, events = [], []
    for seed in args.seeds:
        winner = _winner_for(args, seed)
        for device_name in args.devices:
            device = torch.device(device_name)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('请求了 CUDA 效率审计，但 CUDA 不可用')
            set_seed(seed)
            data, raw, scaler, load_timings = load_development_graph_timed(
                args.data_dir, device, label_ceiling=34)
            t0 = time.perf_counter()
            model = _build_hpo_model(args.model, data['in_dim'], winner['config'], data)
            state = torch.load(winner['state_file'], map_location='cpu', weights_only=True)
            model.load_state_dict(state)
            model.to(device).eval()
            artifact_seconds = time.perf_counter() - t0
            calibrator = None
            if not args.skip_calibration:
                labels = data['y'].detach().cpu().numpy()
                val = ((raw['node_time'] >= 30) & (raw['node_time'] <= 34)
                       & (labels >= 0))
                binary = _binary_logits(model, data)
                calibrator = _fit_validation_calibrator(binary, labels, val)
                model = BinaryVectorCalibratedInference(
                    model, calibrator['scale'], calibrator['bias']).to(device).eval()
            sync(device)
            with torch.no_grad():
                t0 = time.perf_counter()
                logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                                  data['num_nodes'], hard_mask=True)
                if logits.numel() == 0:
                    raise RuntimeError('空推理输出')
                sync(device)
                first_ms = (time.perf_counter() - t0) * 1000
            benchmark = benchmark_inference(
                model, data, device, args.batch_sizes, args.warmup, args.repeats)
            stem = f'{_safe(args.model)}_seed{seed}_{device.type}'
            row = {'protocol': 'p1_system_efficiency_v1', 'model': args.model,
                   'seed': int(seed), 'device': device.type,
                   'source_run_protocol': winner['run_protocol'],
                   'checkpoint_sha256': file_sha256(winner['state_file']),
                   'load_timings': load_timings,
                   'cold_start': cold_start_summary(load_timings, artifact_seconds, first_ms),
                   'benchmark': benchmark, 'calibrator': calibrator,
                   'calibration_included_in_inference': calibrator is not None,
                   'test_labels_35_49_used': False,
                   'transductive_development_preprocessing': True}
            atomic_json(out / f'{stem}.json', row)
            rows.append(row)
            print(f'[P1-5] {stem}; cold={row["cold_start"]["cold_start_total_seconds"]:.3f}s')
    result = {'protocol': 'p1_system_efficiency_v1', 'created_utc': _now(),
              'environment': environment_snapshot(), 'rows': rows,
              'development_only': True,
              'note': '数据加载、预处理、图构建、冷启动和稳态推理分列；图模型不伪造节点批量曲线'}
    atomic_json(final_path, result)
    flat = []
    for row in rows:
        for item in row['benchmark']['rows']:
            flat.append({'model': row['model'], 'seed': row['seed'],
                         'device': row['device'], **item})
    pd.DataFrame(flat).to_csv(out / 'efficiency_curve.csv', index=False)
    return 0


def _append_event(journal, events, phase, seconds, complete=True, **details):
    event = {'phase': phase, 'seconds': float(seconds), 'complete': bool(complete),
             'utc': _now(), **details}
    events.append(event)
    journal.parent.mkdir(parents=True, exist_ok=True)
    with open(journal, 'a', encoding='utf-8') as f:
        f.write(canonical_json(event) + '\n'); f.flush(); os.fsync(f.fileno())


@contextmanager
def _phase(journal, events, phase, **details):
    t0 = time.perf_counter(); complete = False
    try:
        yield
        complete = True
    finally:
        _append_event(journal, events, phase, time.perf_counter() - t0,
                      complete=complete, **details)


def _read_events(journal):
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def cmd_teacher_student_cost(args):
    _require_unique('seeds', args.seeds)
    out = Path(args.output_dir)
    cfg = {'data_sha256': _data_hashes(args.data_dir), 'seeds': args.seeds,
           'epochs': args.epochs, 'patience': args.patience,
           'update_epochs': args.update_epochs, 'hidden': args.hidden,
           'teacher_hidden': args.teacher_hidden, 'hops': args.hops,
           'lr': args.lr, 'wd': args.wd, 'kd_weight': args.kd_weight,
           'device': args.device, 'warmup': args.warmup, 'repeats': args.repeats}
    _guard_invocation(out, cfg)
    if (out / 'cost_summary.json').exists():
        print('[P1-4] 同配置全成本审计已完成，跳过')
        return 0
    device = torch.device(args.device)
    summaries = []
    for seed in args.seeds:
        set_seed(seed)
        stem = out / f'seed{seed}'; stem.mkdir(parents=True, exist_ok=True)
        seed_summary_path = stem / 'cost_summary.json'
        if seed_summary_path.exists():
            summaries.append(read_json(seed_summary_path)); continue
        journal = stem / 'cost_events.jsonl'
        events = _read_events(journal)
        data, raw, scaler, timings = load_development_graph_timed(
            args.data_dir, device, label_ceiling=34)
        for phase, seconds in timings.items():
            _append_event(journal, events, phase, seconds, seed=seed)
        labels = data['y'].detach().cpu().numpy()
        time_np = raw['node_time']
        train_mask = torch.as_tensor((time_np <= 29) & (labels >= 0), device=device)
        val_mask = torch.as_tensor((time_np >= 30) & (time_np <= 34) & (labels >= 0),
                                   device=device)
        masks = (train_mask, val_mask)
        run_protocol = 'p1_cost_' + sha256_bytes(canonical_json(
            {'config': cfg, 'seed': seed}).encode())[:20]
        teacher_path = stem / 'teacher.pt'; teacher_logits_path = stem / 'teacher_logits.pt'
        if teacher_path.exists() and teacher_logits_path.exists():
            teacher_logits = torch.load(teacher_logits_path, map_location=device,
                                        weights_only=True)
            teacher_params = int(sum(p.numel() for p in GraphSAGEBaseline(
                data['in_dim'], hidden=args.teacher_hidden).parameters()))
        else:
            teacher = GraphSAGEBaseline(data['in_dim'], hidden=args.teacher_hidden)
            teacher_params = int(sum(p.numel() for p in teacher.parameters()))
            with _phase(journal, events, 'teacher_training', seed=seed,
                        run_protocol=run_protocol):
                teacher, best = train_model(
                    teacher, data, train_mask, val_mask, device, epochs=args.epochs,
                    lr=args.lr, wd=args.wd, patience=args.patience, loss_fn=False,
                    ckpt_path=str(stem / 'teacher.ckpt'),
                    protocol=run_protocol + ':teacher')
                teacher.eval()
                with torch.no_grad():
                    logits, _ = teacher(data['x'], data['edge_index'],
                                        data['time_gaps'], data['num_nodes'])
                teacher_logits = logits.detach()
            torch.save({k: v.detach().cpu() for k, v in teacher.state_dict().items()},
                       teacher_path)
            torch.save(teacher_logits.cpu(), teacher_logits_path)
            teacher_logits = teacher_logits.to(device)
        student_path = stem / 'student_initial.pt'
        student = TASGLite(data['in_dim'], hidden=args.hidden, hops=args.hops)
        student_params = int(sum(p.numel() for p in student.parameters()))
        metadata = {'train_end': 29, 'val_start': 30, 'val_end': 34,
                    'schedule': {'train_end': 29, 'val_start': 30, 'val_end': 34},
                    'p1_cost_protocol': run_protocol}
        student_cfg = {'lr': args.lr, 'wd': args.wd, 'kd_weight': args.kd_weight,
                       'class_weight_power': 0.5}
        if student_path.exists():
            state = torch.load(student_path, map_location='cpu', weights_only=True)
            student.load_state_dict(state)
        else:
            with _phase(journal, events, 'student_distillation', seed=seed,
                        run_protocol=run_protocol):
                student, best = train_lite(
                    student, data, masks, student_cfg, device, args.epochs,
                    args.patience, str(stem / 'student.ckpt'), metadata,
                    teacher_logits=teacher_logits, verbose=args.verbose)
            torch.save({k: v.detach().cpu() for k, v in student.state_dict().items()},
                       student_path)
        student.to(device).eval()
        cal_path = stem / 'calibration.json'
        if cal_path.exists():
            calibrator = read_json(cal_path)
        else:
            binary = _binary_logits(student, data)
            val_np = val_mask.detach().cpu().numpy()
            with _phase(journal, events, 'calibration', seed=seed):
                calibrator = fit_calibrator('vector', binary[val_np], labels[val_np])
                atomic_json(stem / 'calibrator_only.json', calibrator)
            with _phase(journal, events, 'threshold_selection', seed=seed):
                p_val = apply_calibrator(calibrator, binary[val_np])
                threshold, val_f1 = _tune_probability_threshold(labels[val_np], p_val)
            calibrator.update(threshold=float(threshold), validation_f1=float(val_f1),
                              fitting_data='development_validation_only')
            atomic_json(cal_path, calibrator)
        deployed = BinaryVectorCalibratedInference(
            student, calibrator['scale'], calibrator['bias']).to(device).eval()
        benchmark_path = stem / 'deployment_benchmark.json'
        if benchmark_path.exists():
            benchmark = read_json(benchmark_path)
        else:
            benchmark = benchmark_inference(deployed, data, device,
                                            (1, 32, 128, 1024), args.warmup,
                                            args.repeats)
            first = benchmark['rows'][0]
            _append_event(journal, events, 'student_inference',
                          first['mean_pass_latency_ms'] / 1000.0, seed=seed,
                          measurement='steady_state_per_pass',
                          batch_size=first['batch_size'])
            atomic_json(benchmark_path, benchmark)
        # 更新成本与初始离线成本分开：固定训练窗的小幅更新 + 重新校准/阈值。
        updated_path = stem / 'student_updated.pt'
        if not updated_path.exists():
            updated = copy.deepcopy(student)
            with _phase(journal, events, 'model_update', seed=seed,
                        update_epochs=args.update_epochs):
                updated, _ = train_lite(
                    updated, data, masks, student_cfg, device, args.update_epochs,
                    args.update_epochs + 1, str(stem / 'update.ckpt'),
                    {**metadata, 'update_from': 'student_initial'},
                    teacher_logits=teacher_logits, verbose=False)
            torch.save({k: v.detach().cpu() for k, v in updated.state_dict().items()},
                       updated_path)
        else:
            updated = TASGLite(data['in_dim'], hidden=args.hidden, hops=args.hops)
            updated.load_state_dict(torch.load(updated_path, map_location='cpu',
                                               weights_only=True))
        updated.to(device).eval()
        update_cal_path = stem / 'update_calibration.json'
        if not update_cal_path.exists():
            binary = _binary_logits(updated, data)
            val_np = val_mask.detach().cpu().numpy()
            with _phase(journal, events, 'update_calibration', seed=seed):
                update_cal = fit_calibrator('vector', binary[val_np], labels[val_np])
            with _phase(journal, events, 'update_threshold', seed=seed):
                p_val = apply_calibrator(update_cal, binary[val_np])
                threshold, val_f1 = _tune_probability_threshold(labels[val_np], p_val)
            update_cal.update(threshold=float(threshold), validation_f1=float(val_f1))
            atomic_json(update_cal_path, update_cal)
        summary = summarize_cost_events(events, student_params, teacher_params, 2)
        summary.update(seed=int(seed), run_protocol=run_protocol,
                       development_only=True, test_labels_35_49_used=False,
                       update_cost_scope='fixed_window_rehearsal_proxy_not_temporal_generalization',
                       deployment_benchmark=benchmark,
                       calibrator=calibrator,
                       event_journal=str(journal.relative_to(out)))
        atomic_json(seed_summary_path, summary)
        summaries.append(summary)
        print(f'[P1-4] seed={seed}; initial_offline='
              f'{summary["initial_offline_system_seconds"]:.3f}s; '
              f'update={summary["update_system_seconds"]:.3f}s')
    aggregate = {'protocol': 'p1_teacher_student_full_cost_v1',
                 'created_utc': _now(), 'per_seed': summaries,
                 'development_only': True,
                 'note': '教师训练计入离线系统成本但不计入学生部署参数；初始与更新成本分列'}
    atomic_json(out / 'cost_summary.json', aggregate)
    return 0


def cmd_environment(args):
    out = Path(args.output_dir)
    reference_hash = file_sha256(args.reference) if args.reference else None
    _guard_invocation(out, {'reference_sha256': reference_hash})
    snapshot_path = out / 'environment.json'
    if snapshot_path.exists():
        print('[P1-6] 同目录环境快照已存在，保留并跳过')
        return 0
    snapshot = environment_snapshot()
    freeze_path = out / 'requirements-freeze.txt'
    proc = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f'pip freeze 失败: {proc.stderr}')
    freeze_path.write_text(proc.stdout, encoding='utf-8')
    snapshot['requirements_freeze_sha256'] = file_sha256(freeze_path)
    if args.reference:
        snapshot['comparison_with_reference'] = compare_environment(
            read_json(args.reference), snapshot)
    atomic_json(snapshot_path, snapshot)
    print(f'[P1-6] 环境快照与冻结依赖已保存: {out}')
    return 0


def cmd_release(args):
    includes = args.include
    if not includes:
        allowed = {'.py', '.json', '.md', '.txt', '.bat', '.sh'}
        includes = [p.name for p in ROOT.iterdir()
                    if p.is_file() and p.suffix in allowed]
    manifest = stage_release(ROOT, args.output_dir, includes,
                             read_json(args.lineage_config), args.raw_results,
                             anonymize_launchers=not args.no_anonymize_launchers)
    verification = verify_release(args.output_dir)
    if not verification['valid']:
        raise RuntimeError(f'发布复核失败: {verification["errors"]}')
    if not verification['anonymous_ready']:
        print('[P1-7] staging 完整性通过，但含未审查二进制；'
              '不得称为匿名就绪，需外部隐私复核。')
        return 2
    print(f'[P1-7] 匿名 staging 完成：{len(manifest["files"])} 个文件；'
          f'输出={args.output_dir}')
    return 0


def cmd_release_verify(args):
    result = verify_release(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['valid'] and result.get('anonymous_ready') else 2


def build_parser():
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ap = argparse.ArgumentParser(description='TASGL P1-1 至 P1-8 审稿可信度实验')
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('audit-cases', help='P1-1 FP/FN/时间/攻击案例队列')
    p.add_argument('--input', required=True); p.add_argument('--clean_input')
    p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--model', default='unknown'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--condition', default='clean'); p.add_argument('--include_correct', action='store_true')
    p.add_argument('--output_dir', required=True); p.set_defaults(func=cmd_audit_cases)

    p = sub.add_parser('review-cases', help='P1-1 合并真实外部标注')
    p.add_argument('--queue', required=True); p.add_argument('--annotations', required=True)
    p.add_argument('--output_dir', required=True); p.set_defaults(func=cmd_review_cases)

    p = sub.add_parser('attacks', help='P1-2/P1-3 非 oracle 攻击矩阵')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--hpo_dir', default=str(ROOT / 'results/p0/hpo'))
    p.add_argument('--models', nargs='+', default=['tasgl', 'gcn', 'gat', 'graphsage'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--attack_config', default=str(ROOT / 'p1_attack_matrix.json'))
    p.add_argument('--attack_seeds', type=int, nargs='+')
    p.add_argument('--eval_start', type=int, default=33); p.add_argument('--eval_end', type=int, default=34)
    p.add_argument('--device', default=default_device)
    p.add_argument('--output_dir', default=str(P1_ROOT / 'attacks'))
    p.set_defaults(func=cmd_attacks)

    p = sub.add_parser('teacher-student-cost', help='P1-4 教师/学生/校准/更新全成本')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--epochs', type=int, default=150); p.add_argument('--patience', type=int, default=25)
    p.add_argument('--update_epochs', type=int, default=1)
    p.add_argument('--hidden', type=int, default=48); p.add_argument('--teacher_hidden', type=int, default=64)
    p.add_argument('--hops', type=int, default=0, choices=[0, 1, 2])
    p.add_argument('--lr', type=float, default=0.005); p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--kd_weight', type=float, default=0.5)
    p.add_argument('--device', default=default_device); p.add_argument('--verbose', action='store_true')
    p.add_argument('--warmup', type=int, default=3); p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--output_dir', default=str(P1_ROOT / 'teacher_student_cost'))
    p.set_defaults(func=cmd_teacher_student_cost)

    p = sub.add_parser('efficiency', help='P1-5 冷启动/CPU/GPU/吞吐/内存/批量曲线')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--hpo_dir', default=str(ROOT / 'results/p0/hpo'))
    p.add_argument('--model', required=True, choices=MODELS)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--devices', nargs='+', default=['cpu'])
    p.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 32, 128, 1024])
    p.add_argument('--warmup', type=int, default=3); p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--skip_calibration', action='store_true')
    p.add_argument('--output_dir', default=str(P1_ROOT / 'efficiency'))
    p.set_defaults(func=cmd_efficiency)

    p = sub.add_parser('environment', help='P1-6 完整环境快照和冻结依赖')
    p.add_argument('--reference', help='可选：先前 environment.json，用于差异比较')
    p.add_argument('--output_dir', default=str(P1_ROOT / 'environment'))
    p.set_defaults(func=cmd_environment)

    p = sub.add_parser('release', help='P1-7 匿名 staging 与数据血缘')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--include', nargs='*', help='相对项目根目录的文件；默认仅顶层代码/配置/文档')
    p.add_argument('--raw_results', nargs='*', default=[], help='显式列出的原始结果文件，内容不改写')
    p.add_argument('--lineage_config', default=str(ROOT / 'p1_lineage.json'))
    p.add_argument('--no_anonymize_launchers', action='store_true')
    p.set_defaults(func=cmd_release)

    p = sub.add_parser('release-verify', help='P1-7 复核匿名 staging')
    p.add_argument('--output_dir', required=True); p.set_defaults(func=cmd_release_verify)

    p = sub.add_parser('metrics', help='P1-8 开发预测完整指标')
    p.add_argument('--input', required=True); p.add_argument('--threshold', type=float, required=True)
    p.add_argument('--threshold_source', required=True)
    p.add_argument('--fpr_targets', type=float, nargs='+', default=[0.01, 0.05, 0.10])
    p.add_argument('--evaluation_role', choices=['development'], default='development')
    p.add_argument('--output_dir', required=True); p.set_defaults(func=cmd_metrics)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        print(f'[P1][ERROR] {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
