"""p0_experiments.py — P0-1 至 P0-9 统一实验入口。

证据边界：本程序实现协议与工具，不预设任何假设通过。开发/训练阶段只保存
无测试标签的冻结预测；最终指标由 independent evaluate 在锁后计算。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import set_seed
from p0_core import (apply_calibrator, atomic_json, canonical_json,
                     deterministic_few_shot, file_sha256, fit_calibrator,
                     holm_adjust, make_lock, map_labels, paired_statistics,
                     probability_metrics, read_json, reliability_rows,
                     risk_coverage_rows, semantic_align, sha256_bytes,
                     strict_rolling_view, verify_lock)
from p0_data import (load_dgraphfin, load_elliptic_raw,
                     validate_prediction_npz)
from p0_models import (FeatureMLP, FixedNeighborSampler, NeighborGraphSAGE,
                       SIGNBaseline, TemporalGraphSAGEAdapter)
from p1_core import (build_case_audit, extended_binary_metrics,
                     metric_definitions, write_case_audit)

ROOT = Path(__file__).resolve().parent
P0_ROOT = ROOT / 'results' / 'p0'
CODE_FILES = [ROOT / n for n in (
    'p0_experiments.py', 'p0_core.py', 'p0_data.py', 'p0_models.py', 'p1_core.py',
    'models.py', 'models_v3.py', 'common.py', 'data_loader.py', 'metrics.py',
    'train.py', 'v3_engine.py', 'exp_hypotheses.py')]


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _guard_invocation(out, config):
    """同目录只允许同科学配置续跑；不同配置必须另建证据目录。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'invocation.json'
    digest = sha256_bytes(canonical_json(config).encode())
    if path.exists():
        old = read_json(path)
        if old.get('config_sha256') != digest:
            raise RuntimeError(f'{out} 已有不同调用配置；禁止覆盖历史证据，请换目录')
        return False
    atomic_json(path, {'config_sha256': digest, 'config': config,
                       'created_utc': _now()})
    return True


def _format_pattern(pattern, seed, dataset=None):
    return Path(str(pattern).format(seed=seed, dataset=dataset or ''))


def _locked_artifacts(config):
    dataset = config['dataset']['name']
    artifacts = {}
    root = Path(config['dataset']['root'])
    if dataset == 'elliptic':
        for name in ('elliptic_txs_features.csv', 'elliptic_txs_classes.csv',
                     'elliptic_txs_edgelist.csv'):
            artifacts[f'dataset_{name}'] = root / name
    elif dataset == 'dgraphfin':
        artifacts['dataset_dgraphfin.npz'] = root / 'dgraphfin.npz'
    else:
        raise ValueError(f'不支持的锁数据集: {dataset}')
    for seed in config['seeds']:
        artifacts[f'prediction_seed{seed}'] = _format_pattern(
            config['predictions']['path_pattern'], seed, dataset)
        cal = config['calibration']
        if cal.get('source') == 'artifact':
            artifacts[f'calibration_seed{seed}'] = _format_pattern(
                cal['path_pattern'], seed, dataset)
    return artifacts


def _check_existing_lock(path, new_lock):
    if path.exists():
        old = read_json(path)
        if old.get('lock_id') != new_lock['lock_id']:
            raise RuntimeError(f'已有不同锁 {path}；禁止覆盖，请使用新的独立队列目录')
        return True
    return False


# ------------------------------------------------------------------
# P0-1：独立配置锁 + 单次只读测试授权
# ------------------------------------------------------------------
def cmd_independent(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / 'lock.json'
    config = read_json(args.config)
    if not isinstance(config['evaluation'].get('prior_test_exposure'), bool):
        raise ValueError('evaluation.prior_test_exposure 必须显式声明 true/false')
    if args.phase == 'lock':
        artifacts = _locked_artifacts(config)
        lock = make_lock(config, artifacts, CODE_FILES)
        lock['created_utc'] = _now()
        lock['evidence_status'] = 'locked; final labels not read by this phase'
        if not _check_existing_lock(lock_path, lock):
            atomic_json(lock_path, lock)
        print(f'[P0-1] lock_id={lock["lock_id"]}; artifacts={len(lock["artifacts"])}')
        return 0

    if not lock_path.is_file():
        raise FileNotFoundError('必须先执行 independent lock')
    lock = read_json(lock_path)
    errors = verify_lock(lock, config=config, verify_artifacts=True,
                         verify_code=True)
    if errors:
        raise RuntimeError('锁校验失败: ' + '; '.join(errors))
    result_path = out / 'evaluation.json'
    if result_path.exists():
        print('[P0-1] 该锁已完成最终评估；禁止覆盖或借重跑挑选结果。')
        return 2
    receipt_path = out / 'evaluation_receipt.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['lock_id'] != lock['lock_id']:
            raise RuntimeError('评估收据属于另一锁，拒绝继续')
        if receipt.get('completed_utc'):
            raise RuntimeError('该锁已有完成收据但结果缺失；禁止删除结果后再次评估')
    else:
        receipt = {'lock_id': lock['lock_id'], 'started_utc': _now(),
                   'authorization': 'single_locked_configuration',
                   'technical_resume_same_lock': True}
        atomic_json(receipt_path, receipt)

    ids, labels, node_times = _final_evaluation_records(lock['config'])
    minimum = int(lock['config']['exclusions'].get(
        'minimum_labeled_test_nodes_per_seed', 1))
    if len(ids) < minimum:
        raise ValueError(f'最终评估标注节点 {len(ids)} 少于锁要求 {minimum}')
    run_dir = out / 'evaluation_runs'
    run_dir.mkdir(exist_ok=True)
    rows = []
    for seed in lock['config']['seeds']:
        seed_path = run_dir / f'seed{seed}.json'
        case_dir = run_dir / f'seed{seed}_cases'
        case_summary_path = case_dir / 'case_audit_summary.json'
        if seed_path.exists():
            row = read_json(seed_path)
            if row.get('lock_id') != lock['lock_id'] or row.get('seed') != seed:
                raise RuntimeError('逐种子评估记录与当前锁不匹配')
            if case_summary_path.exists():
                rows.append(row)
                print(f'[P0-1][seed={seed}] 同锁评估与案例记录已存在，跳过')
                continue
            # 技术中断发生在指标落盘与案例落盘之间：同锁重算并补齐，
            # 不读取另一配置、不更换种子，也不触发新的测试授权。
            print(f'[P0-1][seed={seed}] 补齐同锁缺失案例记录')
        pred = validate_prediction_npz(lock['artifacts'][f'prediction_seed{seed}']['path'])
        if pred['seed'] != int(seed):
            raise ValueError(f'预测 seed={pred["seed"]} 与锁 seed={seed} 不符')
        calibrator = {'method': 'none'}
        threshold = float(lock['config']['threshold'].get('value', 0.5))
        if lock['config']['calibration'].get('source') == 'artifact':
            calibrator = read_json(lock['artifacts'][f'calibration_seed{seed}']['path'])
            if lock['config']['threshold'].get('source') == 'calibration_artifact':
                threshold = float(calibrator['threshold'])
        logits = np.log(np.clip(pred['p_prob'], 1e-7, 1 - 1e-7) /
                        np.clip(1 - pred['p_prob'], 1e-7, 1))
        calibrated = apply_calibrator(calibrator, logits)
        p_by_id = {str(n): float(p) for n, p in zip(pred['node_id'], calibrated)}
        missing = [str(n) for n in ids if str(n) not in p_by_id]
        if missing:
            raise ValueError(f'seed={seed} 缺 {len(missing)} 个最终评估节点预测')
        p = np.array([p_by_id[str(n)] for n in ids])
        # P1-8 与 P1-1 均在同一锁、同一授权下计算；阈值不接触测试标签调优。
        row = extended_binary_metrics(labels, p, threshold)
        row.update(seed=int(seed), lock_id=lock['lock_id'],
                   n_predictions=int(len(p)))
        case_rows, case_summary = build_case_audit(
            labels, p, node_ids=ids, node_times=node_times, threshold=threshold,
            model=lock['config'].get('model_name', 'locked_model'), seed=seed,
            condition='locked_final', include_correct=False)
        case_info = write_case_audit(case_dir, case_rows, case_summary, {
            'lock_id': lock['lock_id'], 'evaluation_role': 'locked_final',
            'external_pattern_labels_or_expert_review_required': True})
        row.update(case_audit_path=str(case_summary_path.relative_to(out)),
                   case_audit_sha256=file_sha256(case_summary_path),
                   case_queue_path=str((case_dir / 'case_audit_queue.csv').relative_to(out)),
                   case_queue_sha256=file_sha256(case_dir / 'case_audit_queue.csv'),
                   n_error_cases=case_info['n_cases'],
                   external_case_validation_status='pending_external_annotations')
        atomic_json(seed_path, row)
        rows.append(row)
        print(f'[P0-1][seed={seed}] F1={row["f1"]:.4f} ECE={row["ece"]:.4f}')
    metrics = ('f1', 'pr_auc', 'average_precision', 'roc_auc',
               'recall_at_fpr_1pct', 'ece', 'brier', 'nll')
    aggregate = {k: {'mean': float(np.mean([r[k] for r in rows])),
                     'sample_sd': float(np.std([r[k] for r in rows], ddof=1))
                     if len(rows) > 1 else None} for k in metrics}
    fpr_targets = rows[0]['fpr_targets']
    if any(r['fpr_targets'] != fpr_targets for r in rows):
        raise RuntimeError('同锁各种子的 FPR 操作点定义不一致')
    fpr_aggregate = {}
    for target in fpr_targets:
        key = str(float(target))
        values = [r['recall_at_fpr'][key] for r in rows]
        fpr_aggregate[key] = {
            'mean': float(np.mean(values)),
            'sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}
    result = {'protocol': 'p0_independent_evaluation_v1',
              'lock_id': lock['lock_id'], 'completed_utc': _now(),
              'dataset': lock['config']['dataset']['name'],
              'seeds': lock['config']['seeds'], 'per_seed': rows,
              'aggregate': aggregate, 'recall_at_fpr_aggregate': fpr_aggregate,
              'metric_definitions': metric_definitions(),
              'confusion_matrix_sum': {k: int(sum(r[k] for r in rows))
                                       for k in ('tn', 'fp', 'fn', 'tp')},
              'case_audit_index': [{'seed': r['seed'],
                                    'path': r['case_audit_path'],
                                    'sha256': r['case_audit_sha256'],
                                    'queue_path': r['case_queue_path'],
                                    'queue_sha256': r['case_queue_sha256'],
                                    'n_error_cases': r['n_error_cases'],
                                    'external_validation_status':
                                        'pending_external_annotations'}
                                   for r in rows],
              'test_labels_used_by_this_pipeline_before_lock': False,
              'prior_test_exposure': lock['config']['evaluation']['prior_test_exposure'],
              'evidence_class': ('exploratory_reconfirmation_previously_observed'
                                 if lock['config']['evaluation']['prior_test_exposure']
                                 else 'locked_independent_evaluation_by_preregistration'),
              'note': '一次锁定配置的只读评估；不据此更换种子、阈值或配置后重称确认'}
    atomic_json(result_path, result)
    atomic_json(receipt_path, {'lock_id': lock['lock_id'],
                               'started_utc': receipt.get('started_utc', _now())
                               if receipt_path.exists() else _now(),
                               'completed_utc': _now(),
                               'evaluation_sha256': file_sha256(result_path),
                               'case_audit_sha256': {
                                   str(r['seed']): r['case_audit_sha256'] for r in rows}})
    print(f'[P0-1] 完成锁定评估: {result_path}')
    return 0


def _final_evaluation_records(config):
    """只允许被锁后的 independent evaluate 调用；同时保留案例所需时间。"""
    dataset = config['dataset']
    name = dataset['name']
    root = Path(dataset['root'])
    evaluation = config['evaluation']
    if name == 'elliptic':
        raw = load_elliptic_raw(str(root), label_ceiling=None)
        start, end = evaluation['time_start'], evaluation['time_end']
        mask = ((raw['node_time'] >= start) & (raw['node_time'] <= end)
                & (raw['y'] >= 0))
        return raw['node_ids'][mask], raw['y'][mask], raw['node_time'][mask]
    if name == 'dgraphfin':
        path = root / 'dgraphfin.npz'
        with np.load(path, allow_pickle=False) as z:
            y = np.asarray(z['y'], dtype=np.int64).reshape(-1)
            mask = np.asarray(z[evaluation.get('mask_key', 'test_mask')]).reshape(-1).astype(bool)
            if len(y) != len(mask):
                raise ValueError('最终标签与掩码长度不一致')
            mask &= y >= 0
            ids = np.flatnonzero(mask)
            return ids, y[mask], np.full(len(ids), -1, dtype=np.int64)
    raise ValueError(f'不支持的独立评估数据集: {name}')


def _final_evaluation_labels(config):
    ids, labels, _ = _final_evaluation_records(config)
    return ids, labels


# ------------------------------------------------------------------
# P0-2：仅开发窗口的候选筛选（复用协议 6 全硬门槛）
# ------------------------------------------------------------------
def cmd_develop(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _guard_invocation(out, {'data_dir': str(Path(args.data_dir).resolve()),
                            'selection_seeds': args.selection_seeds,
                            'epochs': args.epochs, 'patience': args.patience,
                            'protocol': 'p0_development_only_v1'})
    protocol = {'protocol': 'p0_development_only_v1',
                'test_labels_35_49_used': False,
                'reference_schedule': [29, 30, 32, 33, 34],
                'short_schedule': [24, 25, 29, 33, 34],
                'H1_f1_min': 0.73, 'H1_params_max': 20000,
                'H4_pass_max': 0.05, 'H4_falsify_above': 0.08,
                'selection_rule': 'all_noncompensatory_gates',
                'created_utc': _now()}
    protocol_path = out / 'development_protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        # created_utc 不参与科学配置比较；比较去除时间后的副本。
        old = read_json(protocol_path)
        a, b = dict(old), dict(protocol)
        a.pop('created_utc', None); b.pop('created_utc', None)
        if a != b:
            raise RuntimeError('开发协议已改变；请使用新目录，禁止覆盖历史')
    else:
        atomic_json(protocol_path, protocol)
    cmd = [sys.executable, str(ROOT / 'exp_hypotheses.py'), '--phase', 'select',
           '--data_dir', args.data_dir, '--out_root', str(out),
           '--lock', str(out / 'lock.json'),
           '--selection_seeds'] + [str(s) for s in args.selection_seeds] + [
           '--epochs', str(args.epochs), '--patience', str(args.patience)]
    print('[P0-2] ' + ' '.join(cmd))
    code = subprocess.call(cmd, cwd=str(ROOT))
    summary = out / 'select' / 'summary.json'
    if code == 2 and summary.is_file():
        print('[P0-2] 预期协议停止：无全门槛合格候选；不生成最终确认授权。')
    return code


# ------------------------------------------------------------------
# P0-3：忠实后验校准对照（不冒充完整 ST-CGNN）
# ------------------------------------------------------------------
def _load_labeled_logits(path):
    with np.load(path, allow_pickle=False) as z:
        if 'logits' not in z or 'y' not in z:
            raise ValueError(f'{path} 必须含 logits 和 y')
        return np.asarray(z['logits']), np.asarray(z['y'], dtype=np.int64)


def _tune_probability_threshold(y, p):
    from sklearn.metrics import f1_score
    best = (0.5, -1.0)
    for threshold in np.linspace(0.02, 0.98, 49):
        score = float(f1_score(y, (p >= threshold).astype(int), zero_division=0))
        if score > best[1] + 1e-12:
            best = (float(threshold), score)
    return best


def cmd_calibration(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fit_logits, fit_y = _load_labeled_logits(args.fit)
    select_logits, select_y = _load_labeled_logits(args.select)
    eval_logits, eval_y = _load_labeled_logits(args.eval)
    input_hashes = {role: file_sha256(path) for role, path in
                    (('fit', args.fit), ('select', args.select), ('eval', args.eval))}
    _guard_invocation(out, {'input_sha256': input_hashes,
                            'methods': args.methods,
                            'regularization': args.regularization,
                            'eval_role': args.eval_role})
    if (out / 'calibration_results.json').exists():
        print('[P0-3] 同配置校准对照已完成，保留原结果并跳过')
        return 0
    methods = args.methods
    candidates = []
    for method in methods:
        cal = fit_calibrator(method, fit_logits, fit_y,
                             regularization=args.regularization)
        ps = apply_calibrator(cal, select_logits)
        threshold, sf1 = _tune_probability_threshold(select_y, ps)
        select_metrics = probability_metrics(select_y, ps, threshold)
        pe = apply_calibrator(cal, eval_logits)
        eval_metrics = probability_metrics(eval_y, pe, threshold)
        candidates.append({'method': method, 'calibrator': cal,
                           'select': select_metrics,
                           'select_threshold_f1': float(sf1),
                           'eval': eval_metrics,
                           'reliability': reliability_rows(eval_y, pe),
                           'risk_coverage': risk_coverage_rows(eval_y, pe, threshold)})
    eligible = [c for c in candidates if np.isfinite(c['select']['nll'])]
    winner = min(eligible, key=lambda c: (c['select']['nll'],
                                          methods.index(c['method'])))
    result = {'protocol': 'p0_same_backbone_calibration_v1',
              'evaluation_role': args.eval_role,
              'fit_select_eval_separated': True,
              'method_selection': 'minimum_select_nll_then_predeclared_order',
              'input_sha256': input_hashes, 'winner': winner['method'],
              'candidates': candidates,
              'boundary': 'post-hoc calibration comparison; not a full ST-CGNN reproduction'}
    atomic_json(out / 'calibration_results.json', result)
    pd.DataFrame([{'method': c['method'],
                   **{'select_' + k: v for k, v in c['select'].items()},
                   **{'eval_' + k: v for k, v in c['eval'].items()}}
                  for c in candidates]).to_csv(out / 'calibration_summary.csv', index=False)
    for c in candidates:
        pd.DataFrame(c['reliability']).to_csv(
            out / f'reliability_{c["method"]}.csv', index=False)
        pd.DataFrame(c['risk_coverage']).to_csv(
            out / f'risk_coverage_{c["method"]}.csv', index=False)
    print(f'[P0-3] winner={winner["method"]}; 输出={out}')
    return 0


def cmd_calibration_export(args):
    """固定角色窗口训练同一骨干，导出 P0-3 的 fit/select/eval logits。"""
    from common import train_model
    out = Path(args.output_dir) / args.model
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data, _, _ = _load_hpo_graph(args.data_dir, device)
    time = data['time_steps'].detach().cpu().numpy()
    labels = data['y'].detach().cpu().numpy()
    train_mask = torch.as_tensor((time <= 27) & (labels >= 0), device=device)
    early_mask = torch.as_tensor((time >= 28) & (time <= 29) & (labels >= 0),
                                 device=device)
    data_hash = {name: file_sha256(Path(args.data_dir) / name) for name in
                 ('elliptic_txs_features.csv', 'elliptic_txs_classes.csv',
                  'elliptic_txs_edgelist.csv')}
    for seed in args.seeds:
        set_seed(seed)
        stem = out / f'seed{seed}'
        stem.mkdir(exist_ok=True)
        cfg = {'model': args.model, 'seed': seed, 'hidden': args.hidden,
               'dropout': args.dropout, 'hops': args.hops, 'lr': args.lr,
               'wd': args.wd, 'epochs': args.epochs, 'patience': args.patience,
               'train': [1, 27], 'early_stopping': [28, 29],
               'calibration_fit': [30, 31], 'method_select': [32, 32],
               'development_eval': [33, 34], 'data_sha256': data_hash}
        protocol = 'p0_calibration_export_' + sha256_bytes(
            canonical_json(cfg).encode())[:20]
        meta_path = stem / 'metadata.json'
        expected = [stem / f'{role}.npz' for role in ('fit', 'select', 'eval')]
        if meta_path.exists() and all(p.exists() for p in expected):
            if read_json(meta_path).get('run_protocol') != protocol:
                raise RuntimeError('校准导出配置改变；请使用新输出目录')
            print(f'[P0-3] seed={seed} 导出已完成，跳过')
            continue
        model_cfg = {'hidden': args.hidden, 'dropout': args.dropout,
                     'hops': args.hops}
        model = _build_hpo_model(args.model, data['in_dim'], model_cfg, data)
        model, best_val = train_model(
            model, data, train_mask, early_mask, device, epochs=args.epochs,
            lr=args.lr, wd=args.wd, patience=args.patience,
            loss_fn=(None if args.model == 'tasgl' else False),
            verbose=args.verbose, ckpt_path=str(stem / 'training.ckpt'),
            protocol=protocol)
        model.eval()
        with torch.no_grad():
            logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                              data['num_nodes'])
        binary = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
        roles = {'fit': (30, 31), 'select': (32, 32), 'eval': (33, 34)}
        counts = {}
        for role, (start, end) in roles.items():
            mask = (time >= start) & (time <= end) & (labels >= 0)
            if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
                raise ValueError(f'{role} 窗口缺少两类开发标签')
            np.savez(stem / f'{role}.npz', logits=binary[mask], y=labels[mask])
            counts[role] = int(mask.sum())
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   stem / 'model.pt')
        atomic_json(meta_path, {'protocol': 'p0_calibration_export_v1',
                                'run_protocol': protocol, 'config': cfg,
                                'counts': counts,
                                'early_stopping_validation_f1': float(best_val),
                                'test_labels_35_49_used': False})
        print(f'[P0-3] seed={seed} 同骨干 fit/select/eval 导出完成: {counts}')
    return 0


# ------------------------------------------------------------------
# P0-4：相同验证信息、试验数与早停规则的独立 HPO
# ------------------------------------------------------------------
_HPO_CONFIGS = [
    {'lr': 0.005, 'wd': 0.0, 'hidden': 32, 'hops': 2},
    {'lr': 0.010, 'wd': 1e-4, 'hidden': 64, 'hops': 3},
    {'lr': 0.020, 'wd': 5e-4, 'hidden': 48, 'hops': 4},
    {'lr': 0.005, 'wd': 1e-3, 'hidden': 64, 'hops': 2},
    {'lr': 0.002, 'wd': 1e-4, 'hidden': 48, 'hops': 3},
    {'lr': 0.010, 'wd': 0.0, 'hidden': 32, 'hops': 4},
]


def _hpo_trial_config(model, trial, trials):
    if trials > len(_HPO_CONFIGS):
        raise ValueError(f'预声明网格最多 {len(_HPO_CONFIGS)} 次；增加预算须先新注册')
    if trials < 1:
        raise ValueError('trials 必须为正')
    # 所有模型使用相同 trial 索引；预算完全相等。
    cfg = dict(_HPO_CONFIGS[trial])
    cfg.update(model=model, trial=trial, dropout=0.3,
               early_stopping='validation_f1', patience='shared_cli')
    return cfg


def _build_hpo_model(name, in_dim, cfg, data=None):
    if name == 'tasgl':
        from models import TASGL
        return TASGL(in_dim, hidden=cfg['hidden'], rff_dim=16, rank=8,
                     num_hops=cfg['hops'], dropout=cfg['dropout'])
    if name == 'temporal_graphsage':
        model = TemporalGraphSAGEAdapter(in_dim, cfg['hidden'], cfg['dropout'])
        model.node_time = data['time_steps']
        return model
    from models import (GATBaseline, GCNBaseline, GraphSAGEBaseline,
                        MLPBaseline, SGCBaseline)
    if name == 'sgc':
        return SGCBaseline(in_dim, num_hops=cfg['hops'])
    if name == 'sign':
        return SIGNBaseline(in_dim, hidden=cfg['hidden'],
                            num_hops=cfg['hops'], dropout=cfg['dropout'])
    classes = {'mlp': MLPBaseline, 'gcn': GCNBaseline,
               'gat': GATBaseline, 'graphsage': GraphSAGEBaseline}
    if name not in classes:
        raise ValueError(f'未知 HPO 模型: {name}')
    return classes[name](in_dim, hidden=cfg['hidden'], dropout=cfg['dropout'])


def _load_hpo_graph(data_dir, device):
    from data_loader import load_elliptic, make_graph_data
    x, labels, time_steps, edge_index, _ = load_elliptic(data_dir)
    labels = labels.copy()
    labels[time_steps >= 35] = -1
    data = make_graph_data(x, labels, time_steps, edge_index, device)
    train_np = (time_steps <= 29) & (labels >= 0)
    val_np = (time_steps >= 30) & (time_steps <= 34) & (labels >= 0)
    train_mask = torch.as_tensor(train_np, device=device)
    val_mask = torch.as_tensor(val_np, device=device)
    return data, train_mask, val_mask


def cmd_hpo(args):
    from common import train_model
    if args.overwrite:
        raise ValueError('P0 禁止覆盖已完成证据；请为新配置使用新 output_dir')
    out = Path(args.output_dir)
    run_dir = out / 'runs'
    run_dir.mkdir(parents=True, exist_ok=True)
    data_hash = {name: file_sha256(Path(args.data_dir) / name) for name in
                 ('elliptic_txs_features.csv', 'elliptic_txs_classes.csv',
                  'elliptic_txs_edgelist.csv')}
    device = torch.device(args.device)
    data, train_mask, val_mask = _load_hpo_graph(args.data_dir, device)
    winners_path = out / 'hpo_winners.json'
    winners = read_json(winners_path) if winners_path.exists() else {
        'protocol': 'p0_equal_budget_hpo_v1', 'data_sha256': data_hash,
        'models': {}}
    if winners.get('data_sha256') != data_hash:
        raise RuntimeError('HPO 数据哈希改变；请使用新输出目录')
    for name in args.models:
        previous = winners['models'].get(name)
        if previous and (previous.get('trials_per_seed') != args.trials
                         or previous.get('seeds') != args.seeds):
            raise RuntimeError(f'{name} 的 HPO 预算或种子已改变；请使用新输出目录')
        model_runs = {}
        for trial in range(args.trials):
            cfg = _hpo_trial_config(name, trial, args.trials)
            for seed in args.seeds:
                set_seed(seed)
                stem = f'{name}_trial{trial}_seed{seed}'
                rp, pp = run_dir / (stem + '.json'), run_dir / (stem + '.pt')
                run_protocol = 'p0_hpo_' + sha256_bytes(canonical_json({
                    'config': cfg, 'seed': seed, 'epochs': args.epochs,
                    'patience': args.patience, 'trials_budget': args.trials,
                    'validation_window': [30, 34],
                    'data_sha256': data_hash}).encode())[:16]
                cached = None
                if rp.exists() and pp.exists() and not args.overwrite:
                    try:
                        candidate = read_json(rp)
                        if candidate.get('run_protocol') == run_protocol:
                            cached = candidate
                    except Exception:
                        cached = None
                if cached is not None:
                    print(f'[P0-4] 跳过已完成 {stem}')
                    model_runs[str(seed) + ':' + str(trial)] = cached
                    continue
                print(f'[P0-4] {stem}; cfg={cfg}')
                model = _build_hpo_model(name, data['in_dim'], cfg, data)
                ckpt = run_dir / (stem + '.ckpt')
                model, best_val = train_model(
                    model, data, train_mask, val_mask, device,
                    epochs=args.epochs, lr=cfg['lr'], wd=cfg['wd'],
                    patience=args.patience,
                    loss_fn=(None if name == 'tasgl' else False), verbose=args.verbose,
                    ckpt_path=str(ckpt), protocol=run_protocol)
                state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                torch.save(state, pp)
                row = {'protocol': 'p0_equal_budget_hpo_v1',
                       'run_protocol': run_protocol, 'model': name,
                       'trial': trial, 'seed': seed, 'config': cfg,
                       'validation_f1_default_threshold': float(best_val),
                       'params': int(sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)),
                       'train_window': [1, 29], 'validation_window': [30, 34],
                       'test_labels_used': False, 'state_file': str(pp.resolve()),
                       'complete': True}
                atomic_json(rp, row)
                model_runs[str(seed) + ':' + str(trial)] = row
        by_seed = {}
        for seed in args.seeds:
            candidates = [model_runs[f'{seed}:{t}'] for t in range(args.trials)]
            winner = min(candidates, key=lambda r: (
                -r['validation_f1_default_threshold'], r['trial']))
            by_seed[str(seed)] = winner
        winners['models'][name] = {'trials_per_seed': args.trials,
                                   'seeds': args.seeds, 'by_seed': by_seed,
                                   'selection': 'max_validation_f1_then_lowest_trial'}
        atomic_json(winners_path, winners)
        print(f'[P0-4] {name} winners 已保存（未计算测试指标）')
    return 0


def cmd_freeze_predictions(args):
    if args.overwrite:
        raise ValueError('P0 禁止覆盖冻结预测；请使用新 output_dir')
    out = Path(args.hpo_dir)
    winners = read_json(out / 'hpo_winners.json')
    if args.model not in winners['models']:
        raise KeyError(f'HPO winner 不存在: {args.model}')
    frozen = out / 'frozen'
    frozen.mkdir(exist_ok=True)
    device = torch.device(args.device)
    data, _, val_mask = _load_hpo_graph(args.data_dir, device)
    tx_ids = pd.read_csv(Path(args.data_dir) / 'elliptic_txs_features.csv',
                         header=None, usecols=[0]).iloc[:, 0].to_numpy()
    time_steps = data['time_steps'].detach().cpu().numpy()
    labels = data['y'].detach().cpu().numpy()
    eval_nodes = np.flatnonzero((time_steps >= args.eval_start)
                                & (time_steps <= args.eval_end))
    for seed in args.seeds:
        row = winners['models'][args.model]['by_seed'].get(str(seed))
        if row is None:
            raise ValueError(f'缺 HPO winner seed={seed}；不得跳过预声明种子')
        pred_path = frozen / f'{args.model}_seed{seed}.npz'
        cal_path = frozen / f'{args.model}_seed{seed}.calibration.json'
        if pred_path.exists() and cal_path.exists() and not args.overwrite:
            validate_prediction_npz(pred_path)
            old_cal = read_json(cal_path)
            if old_cal.get('source_run_protocol') != row['run_protocol']:
                raise RuntimeError('冻结产物属于不同 HPO winner；请使用新目录')
            print(f'[P0-4] 冻结产物已存在并校验，跳过 seed={seed}')
            continue
        set_seed(seed)
        model = _build_hpo_model(args.model, data['in_dim'], row['config'], data)
        state = torch.load(row['state_file'], map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        with torch.no_grad():
            logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                              data['num_nodes'])
        binary = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
        val_np = val_mask.detach().cpu().numpy()
        calibrator = fit_calibrator('vector', binary[val_np], labels[val_np])
        val_p = apply_calibrator(calibrator, binary[val_np])
        threshold, val_f1 = _tune_probability_threshold(labels[val_np], val_p)
        calibrator.update(protocol='p0_frozen_validation_calibration_v1',
                          source_run_protocol=row['run_protocol'],
                          seed=int(seed), threshold=float(threshold),
                          validation_f1=float(val_f1),
                          validation_window=[30, 34], test_labels_used=False)
        p = 1.0 / (1.0 + np.exp(-binary[eval_nodes]))
        np.savez_compressed(pred_path, node_id=tx_ids[eval_nodes], p_prob=p,
                            seed=np.array([seed], dtype=np.int64))
        validate_prediction_npz(pred_path)
        atomic_json(cal_path, calibrator)
        print(f'[P0-4] seed={seed} 已冻结无标签预测与验证集校准器')
    return 0


# ------------------------------------------------------------------
# P0-5：预注册种子、配对单位与多重比较校正
# ------------------------------------------------------------------
def _evaluation_records(path, value_key):
    obj = read_json(path)
    if isinstance(obj, list):
        return obj
    if 'per_seed' in obj:
        return obj['per_seed']
    raise ValueError(f'{path} 不含 per_seed 记录')


def cmd_statistics(args):
    config = read_json(args.config)
    config_hash = sha256_bytes(canonical_json(config).encode())
    if Path(args.output).exists():
        old = read_json(args.output)
        if old.get('config_sha256') != config_hash:
            raise RuntimeError('统计输出属于不同预注册配置；禁止覆盖，请换路径')
        print('[P0-5] 同配置统计已完成，保留原结果并跳过')
        return 0
    seeds = config['seeds']
    alpha = float(config.get('alpha', 0.05))
    if config.get('test') != 'wilcoxon_exact_two_sided':
        raise ValueError('统计配置必须预声明 wilcoxon_exact_two_sided')
    rows = []
    for comparison in config['comparisons']:
        key = comparison.get('value_key', 'f1')
        a = _evaluation_records(comparison['a'], key)
        b = _evaluation_records(comparison['b'], key)
        row = paired_statistics(a, b, seeds, key, alpha)
        row['name'] = comparison['name']
        rows.append(row)
    valid = [i for i, r in enumerate(rows) if r['p_value'] is not None]
    adjusted = holm_adjust([rows[i]['p_value'] for i in valid]) if valid else []
    for i, p in zip(valid, adjusted):
        rows[i]['p_holm'] = p
        rows[i]['significant_after_holm'] = bool(p < alpha)
    for i in set(range(len(rows))) - set(valid):
        rows[i]['p_holm'] = None
        rows[i]['significant_after_holm'] = False
    complete = all(r['status'] == 'pass_complete' for r in rows)
    result = {'protocol': 'p0_preregistered_statistics_v1',
              'config_sha256': config_hash,
              'declared_seeds': seeds, 'alpha': alpha,
              'paired_unit': config.get('paired_unit', 'seed'),
              'sample_sd_ddof': 1, 'multiple_comparison': 'Holm',
              'complete': complete, 'comparisons': rows,
              'note': 'incomplete 不得进入最终显著性结论；负结果不得改写'}
    atomic_json(args.output, result)
    print(f'[P0-5] complete={complete}; 输出={args.output}')
    return 0


# ------------------------------------------------------------------
# P0-6：DGraph-Fin 官方划分完整训练；最终指标仍须 P0-1 锁后评估
# ------------------------------------------------------------------
def _mini_logits(model, model_name, x, sampler, nodes, batch_size, device):
    outputs = []
    for start in range(0, len(nodes), batch_size):
        batch = nodes[start:start + batch_size]
        center = torch.as_tensor(x[batch], device=device)
        if model_name == 'mlp':
            logits = model(center)
        else:
            neigh, mask = sampler.sample(batch)
            logits = model(center, torch.as_tensor(x[neigh], device=device),
                           torch.as_tensor(mask, device=device))
        outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([])


def _mini_train_batch(model, model_name, x, sampler, nodes, device):
    center = torch.as_tensor(x[nodes], device=device)
    if model_name == 'mlp':
        return model(center)
    neigh, mask = sampler.sample(nodes)
    return model(center, torch.as_tensor(x[neigh], device=device),
                 torch.as_tensor(mask, device=device))


def cmd_second_dataset(args):
    if args.overwrite:
        raise ValueError('P0 禁止覆盖第二数据集证据；请使用新 output_dir')
    if args.dataset != 'dgraphfin':
        raise ValueError('当前完整第二数据集训练入口使用 DGraph-Fin；'
                         'BitHeteroNet 需先获得安全的非 pickle 图关系')
    out = Path(args.output_dir) / args.dataset / args.model
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    raw = load_dgraphfin(args.data_dir)
    if raw['time_semantics'] != 'edge_timestamp_only':
        raise ValueError('DGraph-Fin 时间语义核验失败')
    data_path = Path(args.data_dir) / 'dgraphfin.npz'
    data_hash = file_sha256(data_path)
    x = np.nan_to_num(raw['x'], nan=0.0, posinf=0.0, neginf=0.0)
    train_idx = np.flatnonzero(raw['train_mask'] & (raw['y'] >= 0))
    val_idx = np.flatnonzero(raw['valid_mask'] & (raw['y'] >= 0))
    test_idx = np.flatnonzero(raw['test_mask'])
    if not len(train_idx) or not len(val_idx) or not len(test_idx):
        raise ValueError('官方 train/valid/test 掩码为空')
    mean = x[train_idx].mean(axis=0)
    scale = x[train_idx].std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = ((x - mean) / scale).astype(np.float32)
    y = raw['y']
    pos = max(int((y[train_idx] == 1).sum()), 1)
    neg = max(int((y[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)
    for seed in args.seeds:
        set_seed(seed)
        stem = out / f'seed{seed}'
        stem.mkdir(exist_ok=True)
        result_path = stem / 'run.json'
        pred_path = stem / 'test_predictions.npz'
        cal_path = stem / 'calibration.json'
        run_config = {'dataset': args.dataset, 'model': args.model,
                      'seed': seed, 'hidden': args.hidden,
                      'dropout': args.dropout, 'fanout': args.fanout,
                      'batch_size': args.batch_size, 'epochs': args.epochs,
                      'patience': args.patience, 'lr': args.lr,
                      'wd': args.wd, 'data_sha256': data_hash,
                      'split': 'official_train_valid_test_masks',
                      'standardization': 'official_train_nodes_only'}
        run_protocol = 'p0_second_' + sha256_bytes(
            canonical_json(run_config).encode())[:20]
        if result_path.exists() and pred_path.exists() and cal_path.exists() \
                and not args.overwrite:
            old = read_json(result_path)
            if old.get('run_protocol') != run_protocol:
                raise RuntimeError(f'seed={seed} 配置改变；请用新输出目录')
            validate_prediction_npz(pred_path)
            print(f'[P0-6] seed={seed} 已完成，跳过')
            continue
        model = (FeatureMLP(x.shape[1], args.hidden, args.dropout)
                 if args.model == 'mlp' else
                 NeighborGraphSAGE(x.shape[1], args.hidden, args.dropout)).to(device)
        sampler = FixedNeighborSampler(raw['edge_index'], len(x),
                                       args.fanout, seed)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        ckpt_path = stem / 'training.ckpt'
        start_epoch, best_val, best_state, bad = 0, -1.0, None, 0
        if ckpt_path.exists() and not args.overwrite:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ck.get('run_protocol') == run_protocol:
                model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
                best_val, best_state, bad = ck['best_val'], ck['best_state'], ck['bad']
                start_epoch = ck['epoch'] + 1
                print(f'[P0-6] seed={seed} 从 epoch {start_epoch} 恢复')
            else:
                raise RuntimeError('断点属于不同配置；拒绝混用，请换输出目录')
        for epoch in range(start_epoch, args.epochs):
            sampler.sample_seed(seed * 100003 + epoch)
            rng = np.random.RandomState(seed * 100003 + epoch)
            order = rng.permutation(train_idx)
            model.train()
            losses = []
            for start in range(0, len(order), args.batch_size):
                batch = order[start:start + args.batch_size]
                target = torch.as_tensor(y[batch], dtype=torch.float32,
                                         device=device)
                logits = _mini_train_batch(model, args.model, x, sampler,
                                           batch, device)
                loss = F.binary_cross_entropy_with_logits(
                    logits, target, pos_weight=pos_weight)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                val_logits = _mini_logits(model, args.model, x, sampler,
                                          val_idx, args.batch_size, device)
            val_p = 1 / (1 + np.exp(-val_logits))
            val_f1 = probability_metrics(y[val_idx], val_p, 0.5)['f1']
            if val_f1 > best_val:
                best_val = val_f1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            payload = {'model': model.state_dict(), 'opt': opt.state_dict(),
                       'best_val': best_val, 'best_state': best_state,
                       'bad': bad, 'epoch': epoch, 'run_protocol': run_protocol}
            torch.save(payload, ckpt_path.with_suffix('.tmp'))
            os.replace(ckpt_path.with_suffix('.tmp'), ckpt_path)
            print(f'[P0-6][seed={seed}] epoch={epoch} '
                  f'loss={np.mean(losses):.5f} val_f1={val_f1:.4f}')
            if bad >= args.patience:
                break
        if best_state is None:
            raise RuntimeError('训练未产生有效验证状态')
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            val_logits = _mini_logits(model, args.model, x, sampler,
                                      val_idx, args.batch_size, device)
            test_logits = _mini_logits(model, args.model, x, sampler,
                                       test_idx, args.batch_size, device)
        calibrator = fit_calibrator('vector', val_logits, y[val_idx])
        val_p = apply_calibrator(calibrator, val_logits)
        threshold, val_f1 = _tune_probability_threshold(y[val_idx], val_p)
        calibrator.update(protocol='p0_second_validation_calibration_v1',
                          threshold=float(threshold), seed=int(seed),
                          validation_f1=float(val_f1), test_labels_used=False)
        test_p = 1 / (1 + np.exp(-test_logits))
        np.savez_compressed(pred_path, node_id=test_idx.astype(np.int64),
                            p_prob=test_p, seed=np.array([seed]))
        validate_prediction_npz(pred_path)
        atomic_json(cal_path, calibrator)
        result = {'protocol': 'p0_second_dataset_predictions_v1',
                  'run_protocol': run_protocol, 'config': run_config,
                  'status': 'trained_predictions_ready_not_final_evaluation',
                  'best_validation_f1_default_threshold': float(best_val),
                  'calibrated_validation_f1': float(val_f1),
                  'train_valid_test_nodes': [len(train_idx), len(val_idx), len(test_idx)],
                  'positive_label': '1=high-risk borrower',
                  'time_boundary': 'official masks; edge timestamps only; not strict rolling',
                  'test_metrics_computed': False}
        atomic_json(result_path, result)
        print(f'[P0-6] seed={seed} 无标签测试预测已冻结；未计算最终指标')
    return 0


# ------------------------------------------------------------------
# P0-7：严格时间截断 inductive rolling（训练窗标准化 + 逐时步预测）
# ------------------------------------------------------------------
def _rolling_forward(model, model_name, view, raw, device):
    from data_loader import make_graph_data
    old = view['old_indices']
    data = make_graph_data(view['x'].astype(np.float32), raw['y'][old],
                           raw['node_time'][old],
                           torch.as_tensor(view['edge_index'], dtype=torch.long),
                           device)
    if model_name == 'temporal_graphsage':
        model.node_time = data['time_steps']
    with torch.no_grad():
        logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                          data['num_nodes'])
    local = np.flatnonzero(raw['node_time'][old] == view['cutoff'])
    return (raw['node_ids'][old[local]], logits[local].detach().cpu().numpy(),
            raw['y'][old[local]], {'time': int(view['cutoff']),
                                   'visible_nodes': int(len(old)),
                                   'visible_edges': int(view['visible_edges'])})


def cmd_rolling(args):
    if args.overwrite:
        raise ValueError('P0 禁止覆盖 rolling 证据；请使用新 output_dir')
    from common import class_weights_from
    from data_loader import make_graph_data
    from models import build_model
    out = Path(args.output_dir) / args.model
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    raw = load_elliptic_raw(args.data_dir, label_ceiling=args.val_end)
    fit_mask = raw['node_time'] <= args.train_end
    train_view = strict_rolling_view(raw['x'], raw['edge_index'],
                                     raw['node_time'], args.train_end, fit_mask)
    old = train_view['old_indices']
    train_data = make_graph_data(
        train_view['x'].astype(np.float32), raw['y'][old],
        raw['node_time'][old], torch.as_tensor(train_view['edge_index']),
        device)
    train_mask = torch.as_tensor(raw['y'][old] >= 0, device=device)
    data_hash = {Path(p).name: file_sha256(p) for p in raw['source_files']}
    for seed in args.seeds:
        set_seed(seed)
        stem = out / f'seed{seed}'
        stem.mkdir(exist_ok=True)
        cfg = {'model': args.model, 'seed': seed, 'hidden': args.hidden,
               'dropout': args.dropout, 'lr': args.lr, 'wd': args.wd,
               'epochs': args.epochs, 'train_end': args.train_end,
               'validation': [args.val_start, args.val_end],
               'evaluation': [args.eval_start, args.eval_end],
               'data_sha256': data_hash,
               'protocol': 'fixed_epochs_then_sequential_validation'}
        run_protocol = 'p0_rolling_' + sha256_bytes(
            canonical_json(cfg).encode())[:20]
        result_path, pred_path = stem / 'run.json', stem / 'test_predictions.npz'
        cal_path = stem / 'calibration.json'
        if result_path.exists() and pred_path.exists() and cal_path.exists() \
                and not args.overwrite:
            if read_json(result_path).get('run_protocol') != run_protocol:
                raise RuntimeError('rolling 配置改变；请使用新输出目录')
            validate_prediction_npz(pred_path)
            print(f'[P0-7] seed={seed} 已完成，跳过')
            continue
        model_cfg = {'hidden': args.hidden, 'dropout': args.dropout, 'hops': 2}
        model = _build_hpo_model(args.model, raw['n_features'], model_cfg,
                                 train_data).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        ckpt_path = stem / 'training.ckpt'
        start_epoch = 0
        if ckpt_path.exists() and not args.overwrite:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ck.get('run_protocol') != run_protocol:
                raise RuntimeError('rolling 断点属于不同配置，拒绝混用')
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
            start_epoch = ck['epoch'] + 1
            print(f'[P0-7] seed={seed} 从 epoch {start_epoch} 恢复')
        weights = class_weights_from(train_data['y'].cpu(), train_mask.cpu()).to(device)
        for epoch in range(start_epoch, args.epochs):
            model.train()
            logits, _ = model(train_data['x'], train_data['edge_index'],
                              train_data['time_gaps'], train_data['num_nodes'])
            loss = F.cross_entropy(logits[train_mask], train_data['y'][train_mask],
                                   weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'epoch': epoch, 'run_protocol': run_protocol},
                       ckpt_path.with_suffix('.tmp'))
            os.replace(ckpt_path.with_suffix('.tmp'), ckpt_path)
            if args.verbose or epoch == args.epochs - 1:
                print(f'[P0-7][seed={seed}] epoch={epoch} loss={loss.item():.5f}')
        model.eval()
        val_ids, val_logits, val_y, audit = [], [], [], []
        for t in range(args.val_start, args.val_end + 1):
            view = strict_rolling_view(raw['x'], raw['edge_index'],
                                       raw['node_time'], t, fit_mask)
            ids, logits, y, meta = _rolling_forward(model, args.model, view,
                                                    raw, device)
            val_ids.append(ids); val_logits.append(logits); val_y.append(y)
            audit.append(meta)
        val_logits = np.concatenate(val_logits)
        val_y = np.concatenate(val_y)
        binary_val = val_logits[:, 1] - val_logits[:, 0]
        labeled_val = val_y >= 0
        if labeled_val.sum() < 2 or len(np.unique(val_y[labeled_val])) < 2:
            raise ValueError('rolling 验证窗缺少两类标签')
        calibrator = fit_calibrator('vector', binary_val[labeled_val],
                                    val_y[labeled_val])
        val_p = apply_calibrator(calibrator, binary_val[labeled_val])
        threshold, val_f1 = _tune_probability_threshold(val_y[labeled_val], val_p)
        calibrator.update(protocol='p0_strict_rolling_validation_calibration_v1',
                          threshold=float(threshold), seed=int(seed),
                          validation_f1=float(val_f1), test_labels_used=False)
        test_ids, test_logits = [], []
        for t in range(args.eval_start, args.eval_end + 1):
            view = strict_rolling_view(raw['x'], raw['edge_index'],
                                       raw['node_time'], t, fit_mask)
            ids, logits, _, meta = _rolling_forward(model, args.model, view,
                                                    raw, device)
            test_ids.append(ids); test_logits.append(logits); audit.append(meta)
        test_ids = np.concatenate(test_ids)
        test_logits = np.concatenate(test_logits)
        p = 1 / (1 + np.exp(-(test_logits[:, 1] - test_logits[:, 0])))
        np.savez_compressed(pred_path, node_id=test_ids, p_prob=p,
                            seed=np.array([seed], dtype=np.int64))
        validate_prediction_npz(pred_path)
        atomic_json(cal_path, calibrator)
        result = {'protocol': 'p0_strict_inductive_rolling_v1',
                  'run_protocol': run_protocol, 'config': cfg,
                  'status': 'frozen_predictions_ready_not_final_evaluation',
                  'validation_f1': float(val_f1), 'window_audit': audit,
                  'future_nodes_edges_in_training': False,
                  'scaler_fit': 'train_window_visible_nodes_only',
                  'test_metrics_computed': False}
        atomic_json(result_path, result)
        print(f'[P0-7] seed={seed} 严格滚动预测已冻结；未计算测试指标')
    return 0


# ------------------------------------------------------------------
# P0-8：正式跨链语义、标签与图规则对齐；零/少目标标签分离
# ------------------------------------------------------------------
def cmd_crosschain(args):
    mapping = read_json(args.mapping)
    _guard_invocation(args.output_dir, {
        'source_sha256': file_sha256(args.source),
        'target_sha256': file_sha256(args.target),
        'mapping_sha256': file_sha256(args.mapping),
        'protocol': 'p0_explicit_crosschain_alignment_v1'})
    if (Path(args.output_dir) / 'alignment_manifest.json').exists():
        print('[P0-8] 同配置语义对齐已完成，保留原结果并跳过')
        return 0
    source = pd.read_csv(args.source)
    target = pd.read_csv(args.target)
    semantics = mapping['feature_semantics']
    xs, xt, manifest = semantic_align(
        source.to_numpy(), list(source.columns), target.to_numpy(),
        list(target.columns), semantics)
    graph_rules = mapping['graph_rules']
    for side in ('source', 'target'):
        if side not in graph_rules:
            raise ValueError(f'graph_rules 缺 {side}')
    required = ('node_unit', 'edge_semantics', 'directed', 'self_loops',
                'deduplication')
    for key in required:
        if graph_rules['source'].get(key) != graph_rules['target'].get(key):
            raise ValueError(f'图构建规则未对齐: {key}')
    mode = mapping['target_adaptation']['mode']
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sy = map_labels(source[mapping['source_label_column']].astype(str).to_numpy(),
                    mapping['source_label_map'])
    if np.any(sy < 0):
        raise ValueError('源标签含未映射值')
    adaptation = {'mode': mode, 'target_labels_used_for_training': False,
                  'selected_target_indices': []}
    if mode == 'few_shot':
        ty = map_labels(target[mapping['target_label_column']].astype(str).to_numpy(),
                        mapping['target_label_map'])
        pool_column = mapping['target_adaptation']['train_pool_column']
        pool_values = set(mapping['target_adaptation']['train_pool_values'])
        eligible = target[pool_column].isin(pool_values).to_numpy() & (ty >= 0)
        selected = deterministic_few_shot(
            ty, eligible, int(mapping['target_adaptation']['n_per_class']),
            int(mapping['target_adaptation']['selection_seed']),
            target[mapping['target_id_column']].to_numpy())
        adaptation.update(target_labels_used_for_training=True,
                          selected_target_indices=selected.tolist(),
                          selection='sha256_predeclared_within_target_train_pool',
                          test_pool_excluded=True)
        np.save(out / 'target_few_shot_labels.npy', ty[selected])
    elif mode != 'zero_shot':
        raise ValueError('target_adaptation.mode 仅允许 zero_shot/few_shot')
    # 零样本分支从不读取目标标签列；少样本只保存训练池所选标签。
    np.savez_compressed(out / 'aligned_features.npz', source_x=xs, target_x=xt,
                        source_y=sy)
    manifest.update(graph_rules=graph_rules, adaptation=adaptation,
                    source_file_sha256=file_sha256(args.source),
                    target_file_sha256=file_sha256(args.target),
                    mapping_sha256=file_sha256(args.mapping),
                    status='aligned_features_ready_not_transfer_result')
    atomic_json(out / 'alignment_manifest.json', manifest)
    print(f'[P0-8] mode={mode}; 显式语义对齐完成；未把对齐结果冒充迁移性能')
    return 0


# ------------------------------------------------------------------
# P0-9：逐层/种子/时间窗门控、实际执行边与稠密等效成本审计
# ------------------------------------------------------------------
def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _tasgl_latency(model, data, device, repeats):
    import time
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(data['x'], data['edge_index'], data['time_gaps'],
                  data['num_nodes'], hard_mask=True)
        _sync(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(data['x'], data['edge_index'], data['time_gaps'],
                  data['num_nodes'], hard_mask=True)
        _sync(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def _dense_rff_microbench(model, view, nodes, repeats, device):
    """小组件等价性验证：稠密 A@X 与稀疏 index_add 的 RFF 多跳传播。"""
    import time
    from models import masked_aggregate
    n = min(int(nodes), len(view['old_indices']))
    if n < 2:
        return {'status': 'skipped_too_few_nodes'}
    remap = np.full(len(view['old_indices']), -1, dtype=np.int64)
    remap[:n] = np.arange(n)
    edge = view['edge_index']
    keep = (edge[0] < n) & (edge[1] < n)
    edge = remap[edge[:, keep]]
    x = torch.as_tensor(view['x'][:n], dtype=torch.float32, device=device)
    edge_t = torch.as_tensor(edge, dtype=torch.long, device=device)
    prop = model.prop
    with torch.no_grad():
        h = F.relu(model.input_proj(x))
        adj = torch.zeros(n, n, device=device)
        if edge_t.numel():
            # masked_aggregate(src[row], col) 在 col 上归约；稠密行必须为 col。
            adj[edge_t[1], edge_t[0]] = 1.0
        adj = adj / adj.sum(dim=1, keepdim=True).clamp_min(1.0)

        def dense_forward():
            rff = torch.cos(h @ (prop.U @ prop.V) + prop.b_rff)
            acc, cur = torch.zeros_like(rff), rff
            ws = torch.softmax(prop.hop_mix, dim=0)
            acc = acc + ws[0] * cur
            for k in range(1, prop.num_hops + 1):
                cur = adj @ cur
                acc = acc + ws[k] * cur
            return prop.out(acc)

        sparse_out = prop(h, edge_t, n)
        dense_out = dense_forward()
        error = float((sparse_out - dense_out).abs().max().cpu())
        for fn in (lambda: prop(h, edge_t, n), dense_forward):
            for _ in range(2):
                fn()
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            prop(h, edge_t, n)
        _sync(device)
        sparse_ms = (time.perf_counter() - t0) * 1000 / repeats
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            dense_forward()
        _sync(device)
        dense_ms = (time.perf_counter() - t0) * 1000 / repeats
    return {'status': 'measured_component_microbenchmark', 'nodes': n,
            'edges': int(edge_t.shape[1]), 'sparse_ms': sparse_ms,
            'dense_ms': dense_ms, 'max_absolute_output_error': error,
            'equivalent_within_1e-4': bool(error < 1e-4),
            'boundary': 'RFF component only; not a full-graph dense TASGL latency claim'}


def cmd_sparsity(args):
    from data_loader import make_graph_data
    from models import TASGL
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_pattern and args.hpo_dir:
        raise ValueError('checkpoint_pattern 与 hpo_dir 只能二选一')
    winners, winners_hash = None, None
    if args.hpo_dir:
        winners_path = Path(args.hpo_dir) / 'hpo_winners.json'
        winners = read_json(winners_path)
        winners_hash = file_sha256(winners_path)
        if args.model != 'tasgl' or args.model not in winners['models']:
            raise ValueError('P0-9 TASGL 审计需要 hpo_winners 中的 tasgl winner')
    _guard_invocation(out, {'data_dir': str(Path(args.data_dir).resolve()),
                            'checkpoint_pattern': args.checkpoint_pattern,
                            'hpo_dir': args.hpo_dir, 'model': args.model,
                            'winners_sha256': winners_hash,
                            'seeds': args.seeds, 'windows': args.windows,
                            'train_end': args.train_end, 'hidden': args.hidden,
                            'rff_dim': args.rff_dim, 'rank': args.rank,
                            'num_hops': args.num_hops, 'repeats': args.repeats,
                            'dense_micro_nodes': args.dense_micro_nodes,
                            'device': args.device})
    if (out / 'sparsity_audit.json').exists():
        print('[P0-9] 同配置稀疏审计已完成，保留原结果并跳过')
        return 0
    device = torch.device(args.device)
    raw = load_elliptic_raw(args.data_dir, label_ceiling=args.train_end)
    fit_mask = raw['node_time'] <= args.train_end
    run_dir = out / 'runs'
    run_dir.mkdir(exist_ok=True)
    rows = []
    for seed in args.seeds:
        set_seed(seed)
        model = TASGL(raw['n_features'], hidden=args.hidden,
                      rff_dim=args.rff_dim, rank=args.rank,
                      num_hops=args.num_hops, execute_hard_sparse=False).to(device)
        trained = False
        checkpoint = None
        if args.checkpoint_pattern:
            checkpoint = Path(args.checkpoint_pattern.format(seed=seed))
        elif winners is not None:
            row = winners['models']['tasgl']['by_seed'].get(str(seed))
            if row is None:
                raise ValueError(f'hpo_winners 缺 tasgl seed={seed}')
            checkpoint = Path(row['state_file'])
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if 'best_state' in state and state['best_state'] is not None:
                state = state['best_state']
            elif 'model' in state:
                state = state['model']
            model.load_state_dict(state)
            trained = True
        model.eval()
        for window in args.windows:
            row_path = run_dir / f'seed{seed}_time{window}.json'
            if row_path.exists():
                row = read_json(row_path)
                if row.get('protocol') != 'p0_real_gate_execution_audit_v1':
                    raise RuntimeError('稀疏审计断点协议不匹配')
                rows.append(row)
                print(f'[P0-9][seed={seed}][t={window}] 已完成，跳过')
                continue
            view = strict_rolling_view(raw['x'], raw['edge_index'],
                                       raw['node_time'], window, fit_mask)
            old = view['old_indices']
            data = make_graph_data(view['x'].astype(np.float32), raw['y'][old],
                                   raw['node_time'][old],
                                   torch.as_tensor(view['edge_index']), device)
            for module in (model.prop, model.attn1, model.attn2):
                module.audit_enabled = True
            with torch.no_grad():
                model(data['x'], data['edge_index'], data['time_gaps'],
                      data['num_nodes'], hard_mask=True)
            gates = []
            for layer, attn in (('attn1', model.attn1), ('attn2', model.attn2)):
                gates.append({'layer': layer, **attn.last_gate_audit})
            prop_audit = dict(model.prop.last_propagation_audit)
            for module in (model.prop, model.attn1, model.attn2):
                module.audit_enabled = False
            latencies = {}
            for compact in (False, True):
                model.attn1.execute_hard_sparse = compact
                model.attn2.execute_hard_sparse = compact
                latencies[str(compact).lower()] = _tasgl_latency(
                    model, data, device, args.repeats)
            n, e = len(old), int(view['visible_edges'])
            executed = [g['executed_edges'] for g in gates]
            active = [g['active_edges'] for g in gates]
            compact_executed = [g['active_edges'] if g['hard_mask']
                                else g['executed_edges'] for g in gates]
            row = {'protocol': 'p0_real_gate_execution_audit_v1',
                   'seed': seed, 'time_window': window,
                   'trained_checkpoint': trained, 'visible_nodes': n,
                   'visible_edges': e, 'gates': gates,
                   'rff_propagation': prop_audit,
                   'mean_theoretical_active_edge_ratio':
                       float(np.mean(active) / max(e, 1)),
                   'mean_executed_attention_edge_ratio':
                       float(np.mean(executed) / max(e, 1)),
                   'mean_compact_executed_attention_edge_ratio':
                       float(np.mean(compact_executed) / max(e, 1)),
                   'rff_executed_edge_ratio': 1.0 if e else 0.0,
                   'latency_ms_legacy_uncompacted': latencies['false'],
                   'latency_ms_compact_execution': latencies['true'],
                   'analytic_dense_equivalent': {
                       'rff_dense_edge_evaluations': n * n * args.num_hops,
                       'attention_dense_edge_evaluations_per_two_layers': 2 * n * n,
                       'rff_sparse_edge_evaluations': e * args.num_hops,
                       'attention_legacy_edge_evaluations': 2 * e,
                       'attention_compact_edge_evaluations': int(sum(compact_executed)),
                       'note': 'edge-operation counts, not measured full dense FLOPs'},
                   'boundary': 'node/edge gates do not sparsify ungated RFF hops'}
            if args.dense_micro_nodes:
                row['dense_rff_microbenchmark'] = _dense_rff_microbench(
                    model, view, args.dense_micro_nodes, args.repeats, device)
            atomic_json(row_path, row)
            rows.append(row)
            print(f'[P0-9][seed={seed}][t={window}] E={e}; '
                  f'active={row["mean_theoretical_active_edge_ratio"]:.4f}; '
                  f'executed={row["mean_executed_attention_edge_ratio"]:.4f}; '
                  f'legacy={latencies["false"]:.3f}ms; '
                  f'compact={latencies["true"]:.3f}ms')
    result = {'protocol': 'p0_real_gate_execution_audit_v1',
              'trained_checkpoints': bool(args.checkpoint_pattern or args.hpo_dir),
              'seeds': args.seeds, 'windows': args.windows, 'rows': rows,
              'note': '理论非零、实际执行与稠密等效成本分列；不得仅报告软门控比例'}
    atomic_json(out / 'sparsity_audit.json', result)
    flat = []
    for r in rows:
        flat.append({k: v for k, v in r.items()
                     if k not in ('gates', 'rff_propagation',
                                  'analytic_dense_equivalent',
                                  'dense_rff_microbenchmark')})
    pd.DataFrame(flat).to_csv(out / 'sparsity_audit_summary.csv', index=False)
    return 0


# ------------------------------------------------------------------
# 命令行
# ------------------------------------------------------------------
def build_parser():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ap = argparse.ArgumentParser(description='TASGL P0-1 至 P0-9 可审计实验')
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('independent', help='P0-1 锁定/独立最终评估')
    p.add_argument('phase', choices=['lock', 'evaluate'])
    p.add_argument('--config', default=str(ROOT / 'p0_independent_config.json'))
    p.add_argument('--output_dir', default=str(P0_ROOT / 'independent'))
    p.set_defaults(func=cmd_independent)

    p = sub.add_parser('develop', help='P0-2 开发窗口候选筛选')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--output_dir', default=str(P0_ROOT / 'development'))
    p.add_argument('--selection_seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=25)
    p.set_defaults(func=cmd_develop)

    p = sub.add_parser('calibration-export', help='P0-3 导出隔离窗口同骨干 logits')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--model', default='graphsage',
                   choices=['tasgl', 'mlp', 'gcn', 'gat', 'graphsage', 'sgc',
                            'sign', 'temporal_graphsage'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--hops', type=int, default=2)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--device', default=device); p.add_argument('--verbose', action='store_true')
    p.add_argument('--output_dir', default=str(P0_ROOT / 'calibration_exports'))
    p.set_defaults(func=cmd_calibration_export)

    p = sub.add_parser('calibration', help='P0-3 同骨干校准对照')
    p.add_argument('--fit', required=True); p.add_argument('--select', required=True)
    p.add_argument('--eval', required=True)
    p.add_argument('--eval_role', default='development', choices=['development'],
                   help='最终评估必须经 independent 锁，不在此直接进行')
    p.add_argument('--methods', nargs='+', default=['none', 'temperature', 'vector',
                                                   'dirichlet', 'isotonic'])
    p.add_argument('--regularization', type=float, default=1e-3)
    p.add_argument('--output_dir', default=str(P0_ROOT / 'calibration'))
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser('hpo', help='P0-4 基线等预算独立 HPO')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--models', nargs='+', default=['tasgl', 'mlp', 'gcn', 'gat', 'graphsage',
                                                  'sgc', 'sign',
                                                  'temporal_graphsage'])
    p.add_argument('--trials', type=int, default=4)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--device', default=device); p.add_argument('--verbose', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--output_dir', default=str(P0_ROOT / 'hpo'))
    p.set_defaults(func=cmd_hpo)

    p = sub.add_parser('freeze-predictions', help='P0-4 冻结无标签预测')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--hpo_dir', default=str(P0_ROOT / 'hpo'))
    p.add_argument('--model', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--eval_start', type=int, default=35)
    p.add_argument('--eval_end', type=int, default=49)
    p.add_argument('--device', default=device); p.add_argument('--overwrite', action='store_true')
    p.set_defaults(func=cmd_freeze_predictions)

    p = sub.add_parser('statistics', help='P0-5 预注册配对统计')
    p.add_argument('--config', required=True)
    p.add_argument('--output', default=str(P0_ROOT / 'statistics.json'))
    p.set_defaults(func=cmd_statistics)

    p = sub.add_parser('second-dataset', help='P0-6 DGraph-Fin 完整训练')
    p.add_argument('--dataset', default='dgraphfin', choices=['dgraphfin'])
    p.add_argument('--data_dir', default='./data/dgraphfin')
    p.add_argument('--model', choices=['mlp', 'graphsage'], default='graphsage')
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--fanout', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--device', default=device); p.add_argument('--overwrite', action='store_true')
    p.add_argument('--output_dir', default=str(P0_ROOT / 'second_dataset'))
    p.set_defaults(func=cmd_second_dataset)

    p = sub.add_parser('rolling', help='P0-7 严格 inductive rolling')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--model', default='graphsage',
                   choices=['mlp', 'gcn', 'gat', 'graphsage', 'temporal_graphsage'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--train_end', type=int, default=29)
    p.add_argument('--val_start', type=int, default=30)
    p.add_argument('--val_end', type=int, default=34)
    p.add_argument('--eval_start', type=int, default=35)
    p.add_argument('--eval_end', type=int, default=49)
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--device', default=device); p.add_argument('--overwrite', action='store_true')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--output_dir', default=str(P0_ROOT / 'rolling'))
    p.set_defaults(func=cmd_rolling)

    p = sub.add_parser('crosschain', help='P0-8 显式跨链语义对齐')
    p.add_argument('--source', required=True); p.add_argument('--target', required=True)
    p.add_argument('--mapping', required=True)
    p.add_argument('--output_dir', default=str(P0_ROOT / 'crosschain'))
    p.set_defaults(func=cmd_crosschain)

    p = sub.add_parser('sparsity', help='P0-9 真实门控与成本审计')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--checkpoint_pattern', default=None,
                   help='可选，含 {seed}；与 hpo_dir 二选一')
    p.add_argument('--hpo_dir', default=None,
                   help='可选，自动读取 hpo_winners.json 的逐种子 TASGL 状态')
    p.add_argument('--model', default='tasgl', choices=['tasgl'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--windows', type=int, nargs='+', default=[29, 34, 39, 44, 49])
    p.add_argument('--train_end', type=int, default=29)
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--rff_dim', type=int, default=16)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--num_hops', type=int, default=4)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--dense_micro_nodes', type=int, default=0)
    p.add_argument('--device', default=device)
    p.add_argument('--output_dir', default=str(P0_ROOT / 'sparsity'))
    p.set_defaults(func=cmd_sparsity)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
