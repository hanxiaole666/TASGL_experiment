"""p3_experiments.py — P3-1 至 P3-6 真实证据入口。

证据边界：本程序把真实产物串成可审计链条，不虚构任何数值。缺数据、缺种子或
预算不公平时一律输出 incomplete_* / pending_* 并阻断结论。
新结果隔离在 results/p3/。
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from p0_core import (apply_calibrator, atomic_json, canonical_json,
                     file_sha256, fit_calibrator, read_json, sha256_bytes,
                     verify_lock)
from p0_data import validate_prediction_npz
from p0_models import FeatureMLP, NeighborGraphSAGE, FixedNeighborSampler
from p1_core import (build_case_audit, extended_binary_metrics,
                     metric_definitions, write_case_audit)
from p2_core import validate_seed_pool
from p3_core import (aggregate_locked_evaluation, audit_independent_evaluation,
                     build_verdicts, check_semantic_mapping,
                     collect_hypothesis_observations, declare_independent_split,
                     fair_comparison, fit_source_classifier,
                     layerwise_sparsity_audit, predict_target)
from p3_data import dataset_file_hashes, load_dgraphfin_real

ROOT = Path(__file__).resolve().parent
P3_ROOT = ROOT / 'results' / 'p3'
DEFAULT_SEEDS = list(range(10))


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _tune_probability_threshold(y, p):
    from sklearn.metrics import f1_score
    best = (0.5, -1.0)
    for threshold in np.linspace(0.02, 0.98, 49):
        score = float(f1_score(y, (p >= threshold).astype(int), zero_division=0))
        if score > best[1] + 1e-12:
            best = (float(threshold), score)
    return best


# ------------------------------------------------------------------
# P3-1：独立数据／划分的声明与锁定评估审计
# ------------------------------------------------------------------
def _final_records(cfg):
    """兼容真实文件的最终评估读取；索引掩码不再被误判为长度不一致。"""
    dataset = cfg['dataset']
    root = Path(dataset['root'])
    if dataset['name'] == 'dgraphfin':
        raw = load_dgraphfin_real(str(root), reveal_test_labels=True)
        mask = raw[cfg['evaluation'].get('mask_key', 'test_mask')]
        ids = np.flatnonzero(mask)
        return ids, raw['y'][ids], np.full(len(ids), -1, dtype=np.int64)
    if dataset['name'] == 'elliptic':
        from p0_data import load_elliptic_raw
        raw = load_elliptic_raw(str(root), label_ceiling=None)
        ev = cfg['evaluation']
        mask = ((raw['node_time'] >= ev['time_start'])
                & (raw['node_time'] <= ev['time_end']) & (raw['y'] >= 0))
        return raw['node_ids'][mask], raw['y'][mask], raw['node_time'][mask]
    raise ValueError(f'不支持的最终评估数据集: {dataset["name"]}')


def _independent_evaluate(args):
    """一次锁定配置的只读评估；语义与 P0-1 一致，仅替换掩码/标签读取。"""
    out = Path(args.output_dir)
    config = read_json(args.config)
    lock_path = out / 'lock.json'
    if not lock_path.is_file():
        raise FileNotFoundError('必须先执行 independent lock')
    lock = read_json(lock_path)
    errors = verify_lock(lock, config=config, verify_artifacts=True,
                         verify_code=not args.skip_code_verify)
    if errors:
        raise RuntimeError('锁校验失败: ' + '; '.join(errors))
    result_path = out / 'evaluation.json'
    if result_path.exists():
        print('[P3-1] 该锁已完成最终评估；禁止覆盖或借重跑挑选结果。')
        return 2
    receipt_path = out / 'evaluation_receipt.json'
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt['lock_id'] != lock['lock_id']:
            raise RuntimeError('评估收据属于另一锁，拒绝继续')
        if receipt.get('completed_utc'):
            raise RuntimeError('该锁已有完成收据但结果缺失；禁止删除结果后再次评估')
    else:
        atomic_json(receipt_path, {
            'lock_id': lock['lock_id'], 'started_utc': _now(),
            'authorization': 'single_locked_configuration',
            'technical_resume_same_lock': True})
    ids, labels, node_times = _final_records(lock['config'])
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
        if seed_path.exists() and case_summary_path.exists():
            row = read_json(seed_path)
            if row.get('lock_id') != lock['lock_id'] or row.get('seed') != seed:
                raise RuntimeError('逐种子评估记录与当前锁不匹配')
            rows.append(row)
            print(f'[P3-1][seed={seed}] 同锁评估已存在，跳过')
            continue
        pred = validate_prediction_npz(
            lock['artifacts'][f'prediction_seed{seed}']['path'])
        if pred['seed'] != int(seed):
            raise ValueError(f'预测 seed={pred["seed"]} 与锁 seed={seed} 不符')
        calibrator = {'method': 'none'}
        threshold = float(lock['config']['threshold'].get('value', 0.5))
        if lock['config']['calibration'].get('source') == 'artifact':
            calibrator = read_json(
                lock['artifacts'][f'calibration_seed{seed}']['path'])
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
                   n_error_cases=case_info['n_cases'],
                   external_case_validation_status='pending_external_annotations')
        atomic_json(seed_path, row)
        rows.append(row)
        print(f'[P3-1][seed={seed}] F1={row["f1"]:.4f} ECE={row["ece"]:.4f}')
    metrics = ('f1', 'pr_auc', 'average_precision', 'roc_auc',
               'recall_at_fpr_1pct', 'ece', 'brier', 'nll')
    aggregate = {k: {'mean': float(np.mean([r[k] for r in rows])),
                     'sample_sd': float(np.std([r[k] for r in rows], ddof=1))
                     if len(rows) > 1 else None} for k in metrics}
    result = {'protocol': 'p3_independent_evaluation_v1',
              'lock_id': lock['lock_id'], 'completed_utc': _now(),
              'dataset': lock['config']['dataset']['name'],
              'seeds': lock['config']['seeds'], 'per_seed': rows,
              'aggregate': aggregate,
              'metric_definitions': metric_definitions(),
              'confusion_matrix_sum': {k: int(sum(r[k] for r in rows))
                                       for k in ('tn', 'fp', 'fn', 'tp')},
              'test_labels_used_by_this_pipeline_before_lock': False,
              'prior_test_exposure':
                  lock['config']['evaluation']['prior_test_exposure'],
              'evidence_class': ('exploratory_reconfirmation_previously_observed'
                                 if lock['config']['evaluation']['prior_test_exposure']
                                 else 'locked_independent_evaluation_by_preregistration'),
              'note': '一次锁定配置的只读评估；不据此更换种子、阈值或配置后重称确认'}
    atomic_json(result_path, result)
    atomic_json(receipt_path, {'lock_id': lock['lock_id'],
                               'started_utc': read_json(receipt_path)['started_utc'],
                               'completed_utc': _now(),
                               'evaluation_sha256': file_sha256(result_path)})
    print(f'[P3-1] 完成锁定评估: {result_path}')
    return 0


def cmd_independent(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.action == 'evaluate':
        return _independent_evaluate(args)
    if args.action == 'declare':
        raw = load_dgraphfin_real(args.data_dir)
        hashes = dataset_file_hashes(args.data_dir, ['dgraphfin.npz'])
        declaration = declare_independent_split(
            args.dataset, args.data_dir, raw,
            prior_exposure=bool(args.prior_exposure),
            declared_by=args.declared_by,
            evidence=[e for e in (args.evidence or [])],
            file_hashes=hashes)
        declaration['declared_utc'] = _now()
        atomic_json(out / 'independent_split_declaration.json', declaration)
        print(f"[P3-1] 划分指纹={declaration['split_fingerprint'][:16]}; "
              f"证据等级={declaration['evidence_class']}")
        print(f"[P3-1] 划分互斥={declaration['split_audit']['disjoint']}; "
              f"测试标签仅锁后可见="
              f"{declaration['split_audit']['train_and_valid_labels_only']}")
        return 0
    lock = read_json(args.lock)
    evaluation = read_json(args.evaluation)
    seeds = validate_seed_pool(args.seeds if args.seeds else
                               lock['config']['seeds'], min_seeds=1)
    audit = audit_independent_evaluation(lock, evaluation, seeds,
                                         require_confirmatory=args.require_confirmatory)
    audit['audited_utc'] = _now()
    atomic_json(out / 'independent_evaluation_audit.json', audit)
    for error in audit['errors']:
        print(f'[P3-1][FAIL] {error}')
    print(f"[P3-1] status={audit['status']}; "
          f"classification={audit['classification']}")
    if audit['blocked_by_requirement']:
        print('[P3-1][STOP] 该评估不是独立确认，不能写成确认性证据')
        return 2
    return 0


# ------------------------------------------------------------------
# P3-2：DGraph-Fin 十种子真实训练与指标
# ------------------------------------------------------------------
def cmd_second_dataset(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.action == 'report':
        evaluation = read_json(args.evaluation)
        seeds = validate_seed_pool(args.seeds or DEFAULT_SEEDS, min_seeds=1)
        report = aggregate_locked_evaluation(evaluation, seeds)
        report['evaluation_sha256'] = file_sha256(args.evaluation)
        report['generated_utc'] = _now()
        atomic_json(out / 'second_dataset_metrics.json', report)
        print(f"[P3-2] status={report['status']}; "
              f"n_available={report['n_available']}/{report['n_declared']}")
        for metric, block in report['metrics'].items():
            if block['mean'] is None:
                continue
            print(f"[P3-2] {metric}: mean={block['mean']:.4f} "
                  f"sd={block['sample_sd']}")
        return 0 if report['status'] == 'pass_complete' else 2
    # train
    from p0_experiments import _mini_logits, _mini_train_batch
    raw = load_dgraphfin_real(args.data_dir)
    if raw['time_semantics'] != 'edge_timestamp_only':
        raise ValueError('DGraph-Fin 时间语义核验失败')
    device = torch.device(args.device)
    data_hash = file_sha256(Path(args.data_dir) / 'dgraphfin.npz')
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
    print(f'[P3-2] 真实划分: train={len(train_idx)} valid={len(val_idx)} '
          f'test={len(test_idx)}; 训练正类率={pos / (pos + neg):.4%}')
    for seed in args.seeds:
        from common import set_seed
        set_seed(seed)
        stem = out / f'seed{seed}'
        stem.mkdir(parents=True, exist_ok=True)
        result_path, pred_path = stem / 'run.json', stem / 'test_predictions.npz'
        cal_path = stem / 'calibration.json'
        run_config = {'dataset': 'dgraphfin', 'model': args.model, 'seed': seed,
                      'hidden': args.hidden, 'dropout': args.dropout,
                      'fanout': args.fanout, 'batch_size': args.batch_size,
                      'epochs': args.epochs, 'patience': args.patience,
                      'lr': args.lr, 'wd': args.wd, 'data_sha256': data_hash,
                      'split': 'official_train_valid_test_masks',
                      'standardization': 'official_train_nodes_only',
                      'loader': 'p3_data.load_dgraphfin_real'}
        run_protocol = 'p3_second_' + sha256_bytes(
            canonical_json(run_config).encode())[:20]
        if result_path.exists() and pred_path.exists() and cal_path.exists():
            old = read_json(result_path)
            if old.get('run_protocol') != run_protocol:
                raise RuntimeError(f'seed={seed} 配置改变；请用新输出目录')
            print(f'[P3-2] seed={seed} 已完成，跳过')
            continue
        model = (FeatureMLP(x.shape[1], args.hidden, args.dropout)
                 if args.model == 'mlp' else
                 NeighborGraphSAGE(x.shape[1], args.hidden, args.dropout)).to(device)
        sampler = FixedNeighborSampler(raw['edge_index'], len(x), args.fanout, seed)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.wd)
        ckpt_path = stem / 'training.ckpt'
        start_epoch, best_val, best_state, bad = 0, -1.0, None, 0
        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ck.get('run_protocol') == run_protocol:
                model.load_state_dict(ck['model'])
                opt.load_state_dict(ck['opt'])
                best_val, best_state, bad = ck['best_val'], ck['best_state'], ck['bad']
                start_epoch = ck['epoch'] + 1
                print(f'[P3-2] seed={seed} 从 epoch {start_epoch} 恢复')
            else:
                raise RuntimeError('断点属于不同配置；拒绝混用')
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
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                val_logits = _mini_logits(model, args.model, x, sampler,
                                          val_idx, args.batch_size, device)
            val_f1 = extended_binary_metrics(y[val_idx],
                                             1 / (1 + np.exp(-val_logits)),
                                             0.5)['f1']
            if val_f1 > best_val:
                best_val, bad = val_f1, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                bad += 1
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'best_val': best_val, 'best_state': best_state,
                        'bad': bad, 'epoch': epoch,
                        'run_protocol': run_protocol},
                       ckpt_path.with_suffix('.tmp'))
            ckpt_path.with_suffix('.tmp').replace(ckpt_path)
            print(f'[P3-2][seed={seed}] epoch={epoch} '
                  f'loss={np.mean(losses):.5f} val_f1={val_f1:.4f}')
            if bad >= args.patience:
                break
        if best_state is None:
            raise RuntimeError('训练未产生有效验证状态')
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_logits = _mini_logits(model, args.model, x, sampler, val_idx,
                                      args.batch_size, device)
            test_logits = _mini_logits(model, args.model, x, sampler, test_idx,
                                       args.batch_size, device)
        calibrator = fit_calibrator('vector', val_logits, y[val_idx])
        val_p = apply_calibrator(calibrator, val_logits)
        threshold, val_f1 = _tune_probability_threshold(y[val_idx], val_p)
        calibrator.update(protocol='p3_second_validation_calibration_v1',
                          threshold=float(threshold), seed=int(seed),
                          validation_f1=float(val_f1), test_labels_used=False)
        test_p = 1 / (1 + np.exp(-test_logits))
        np.savez_compressed(pred_path, node_id=test_idx.astype(np.int64),
                            p_prob=test_p, seed=np.array([seed]))
        atomic_json(cal_path, calibrator)
        atomic_json(result_path, {
            'protocol': 'p3_second_dataset_predictions_v1',
            'run_protocol': run_protocol, 'config': run_config,
            'status': 'trained_predictions_ready_not_final_evaluation',
            'best_validation_f1_default_threshold': float(best_val),
            'calibrated_validation_f1': float(val_f1),
            'train_valid_test_nodes': [len(train_idx), len(val_idx),
                                       len(test_idx)],
            'split_audit': raw['split_audit'],
            'test_metrics_computed': False})
        if ckpt_path.exists():
            ckpt_path.unlink()
        print(f'[P3-2] seed={seed} 无标签测试预测已冻结；未计算最终指标')
    return 0


# ------------------------------------------------------------------
# P3-3：等预算 HPO 后的公平精度、效率与统计结论
# ------------------------------------------------------------------
def cmd_fair_comparison(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = read_json(args.spec)
    evaluations = {}
    for item in args.evaluation or []:
        if '=' not in item:
            raise ValueError(f'--evaluation 必须形如 NAME=PATH，实际: {item}')
        name, path = item.split('=', 1)
        evaluations[name] = path
    winners = read_json(Path(args.hpo_dir) / 'hpo_winners.json')
    efficiency = read_json(args.efficiency) if args.efficiency else None
    seeds = args.seeds or spec.get('seeds')
    result = fair_comparison(winners, spec, evaluations, efficiency,
                             reference=args.reference, declared_seeds=seeds,
                             alpha=args.alpha, hpo_dir=args.hpo_dir)
    result['generated_utc'] = _now()
    result['evaluation_sha256'] = {n: (file_sha256(p) if Path(p).exists()
                                       else None)
                                   for n, p in evaluations.items()}
    atomic_json(out / 'fair_comparison.json', result)
    for name, row in sorted(result['per_model'].items()):
        print(f"[P3-3] {name}: status={row['status']} "
              f"f1={row.get('f1_mean')} params={row.get('params')}")
    for cost, block in sorted(result['pareto'].items()):
        print(f"[P3-3] Pareto[{cost}]: {block.get('frontier')} "
              f"reference_on_frontier={block.get('reference_on_frontier')}")
    if not result['conclusions_allowed']:
        print('[P3-3][STOP] 结论被阻断：')
        for blocker in result['blockers']:
            print(f'[P3-3] - {blocker}')
        return 2
    print('[P3-3] 公平对比结论可用（仍须原样报告不显著项）')
    return 0


# ------------------------------------------------------------------
# P3-4：真实共享语义存在时的跨链迁移
# ------------------------------------------------------------------
def _load_aligned(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    source = None
    for key in ('x_source', 'source', 'xs'):
        if key in data:
            source = data[key]
            break
    target = None
    for key in ('x_target', 'target', 'xt'):
        if key in data:
            target = data[key]
            break
    labels = None
    for key in ('y_source', 'y', 'labels'):
        if key in data:
            labels = np.asarray(data[key], dtype=np.int64)
            break
    if source is None or target is None:
        raise ValueError(f'{path} 缺少源/目标对齐特征（x_source/x_target）')
    return source, target, labels


def cmd_crosschain(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping = read_json(args.mapping)
    source_columns = (args.source_columns.split(',')
                      if args.source_columns else None)
    target_columns = (args.target_columns.split(',')
                      if args.target_columns else None)
    check = check_semantic_mapping(mapping, source_columns, target_columns)
    check['mapping_sha256'] = file_sha256(args.mapping)
    atomic_json(out / 'semantic_mapping_check.json', check)
    if not check['usable']:
        print('[P3-4][BLOCKED] 共享语义未通过核验，跨链结论被阻塞')
        for error in check['errors']:
            print(f'[P3-4] - {error}')
        return 2
    if args.action == 'check':
        print(f"[P3-4] 语义可用: {check['n_canonical_features']} 个 canonical 特征; "
              f"mode={check['mode']}")
        return 0
    x_source, x_target, y_source = _load_aligned(args.aligned)
    if y_source is None:
        raise ValueError('对齐文件缺少源域标签 y_source')
    device = torch.device(args.device)
    model = fit_source_classifier(x_source, y_source, hidden=args.hidden,
                                  epochs=args.epochs, lr=args.lr,
                                  seed=args.seed, device=device)
    p_target = predict_target(model, x_target, device=device)
    np.savez_compressed(out / 'target_predictions.npz',
                        node_id=np.arange(len(p_target)), p_prob=p_target,
                        seed=np.array([args.seed]))
    result = {'protocol': 'p3_crosschain_transfer_v1',
              'mapping_sha256': check['mapping_sha256'],
              'mode': check['mode'],
              'n_canonical_features': check['n_canonical_features'],
              'aligned_sha256': file_sha256(args.aligned),
              'n_source': int(len(x_source)), 'n_target': int(len(x_target)),
              'seed': int(args.seed),
              'target_labels_used_for_training': False,
              'status': 'frozen_target_predictions_no_target_metrics',
              'note': ('目标指标只能由 P0-1 独立锁后评估产生；'
                       '独立 PCA 或相同维数不构成语义对齐')}
    atomic_json(out / 'crosschain_transfer.json', result)
    print(f"[P3-4] 目标预测已冻结: n_target={len(p_target)}; "
          f"未使用目标标签训练")
    return 0


# ------------------------------------------------------------------
# P3-5：逐层、逐种子、逐时间窗稀疏与延迟审计
# ------------------------------------------------------------------
def cmd_sparsity_layerwise(args):
    from data_loader import make_graph_data
    from models import TASGL
    from p0_core import strict_rolling_view
    from p0_data import load_elliptic_raw
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if bool(args.checkpoint_pattern) == bool(args.hpo_dir):
        raise ValueError('checkpoint_pattern 与 hpo_dir 必须二选一')
    device = torch.device(args.device)
    raw = load_elliptic_raw(args.data_dir, label_ceiling=args.train_end)
    fit_mask = raw['node_time'] <= args.train_end
    rows = []
    trained = False
    for seed in args.seeds:
        from common import set_seed
        set_seed(seed)
        model = TASGL(raw['n_features'], hidden=args.hidden,
                      rff_dim=args.rff_dim, rank=args.rank,
                      num_hops=args.num_hops).to(device)
        checkpoint = None
        if args.checkpoint_pattern:
            checkpoint = Path(args.checkpoint_pattern.format(seed=seed))
        else:
            winners = read_json(Path(args.hpo_dir) / 'hpo_winners.json')
            row = (winners.get('models', {}).get('tasgl', {})
                   .get('by_seed', {}).get(str(seed)))
            if row is None:
                raise ValueError(f'hpo_winners 缺 tasgl seed={seed}')
            checkpoint = Path(row['state_file'])
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if 'best_state' in state and state['best_state'] is not None:
            state = state['best_state']
        elif 'model' in state:
            state = state['model']
        model.load_state_dict(state)
        trained = True
        model.eval()
        for window in args.windows:
            view = strict_rolling_view(raw['x'], raw['edge_index'],
                                       raw['node_time'], window, fit_mask)
            old = view['old_indices']
            data = make_graph_data(view['x'].astype(np.float32),
                                   raw['y'][old], raw['node_time'][old],
                                   torch.as_tensor(view['edge_index']), device)
            block = layerwise_sparsity_audit(model, data, device,
                                             repeats=args.repeats,
                                             warmup=args.warmup)
            row = {'protocol': 'p3_layerwise_sparsity_v1', 'seed': int(seed),
                   'time_window': int(window), 'trained_checkpoint': trained,
                   'visible_nodes': int(len(old)),
                   'visible_edges': int(view['visible_edges']),
                   'layers': block['rows'],
                   'total_layer_latency_ms': block['total_layer_latency_ms']}
            rows.append(row)
            print(f'[P3-5][seed={seed}][t={window}] '
                  f"逐层延迟合计={block['total_layer_latency_ms']:.3f}ms; "
                  f"attn1 活跃={row['layers'][2]['active_ratio']}")
    result = {'protocol': 'p3_layerwise_sparsity_v1',
              'trained_checkpoints': trained,
              'seeds': list(args.seeds), 'windows': list(args.windows),
              'repeats': args.repeats, 'warmup': args.warmup,
              'device': str(device), 'rows': rows,
              'note': ('逐层延迟之和不等于端到端延迟；RFF 多跳未门控，'
                       '不得只报告注意力层的活跃边比例')}
    atomic_json(out / 'layerwise_sparsity.json', result)
    print(f'[P3-5] trained_checkpoints={trained}; 组合数={len(rows)}')
    return 0


# ------------------------------------------------------------------
# P3-6：H1–H5 按实际结果判定
# ------------------------------------------------------------------
def cmd_verdicts(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = read_json(args.sources)
    table = read_json(args.table)
    lock = read_json(args.lock) if args.lock and Path(args.lock).exists() else None
    observations = collect_hypothesis_observations(sources)
    atomic_json(out / 'hypothesis_observations.json', observations)
    verdicts = build_verdicts(observations, table, lock)
    verdicts['generated_utc'] = _now()
    verdicts['sources_sha256'] = file_sha256(args.sources)
    atomic_json(out / 'hypothesis_verdicts.json', verdicts)
    for name, entry in sorted(verdicts['hypotheses'].items()):
        print(f'[P3-6] {name}: {entry["status"]}')
    print(f'[P3-6] all_pass={verdicts["all_pass"]}')
    return 0


# ------------------------------------------------------------------
P3_TRACEABILITY = [
    {'requirement': 'P3-1', 'gap': '独立数据/划分上的真实锁定评估',
     'core': ['declare_independent_split', 'audit_independent_evaluation'],
     'cli': 'independent declare|audit',
     'outputs': ['independent_split_declaration.json',
                 'independent_evaluation_audit.json']},
    {'requirement': 'P3-2', 'gap': 'DGraph-Fin 十种子实际训练与指标',
     'core': ['p3_data.load_dgraphfin_real', 'aggregate_locked_evaluation'],
     'cli': 'second-dataset train|report',
     'outputs': ['seed*/test_predictions.npz', 'second_dataset_metrics.json']},
    {'requirement': 'P3-3', 'gap': '等预算 HPO 后的公平精度/效率/统计结论',
     'core': ['fair_comparison', 'pareto_frontier'],
     'cli': 'fair-comparison',
     'outputs': ['fair_comparison.json']},
    {'requirement': 'P3-4', 'gap': '真实共享语义存在时的跨链迁移',
     'core': ['check_semantic_mapping', 'fit_source_classifier',
              'predict_target'],
     'cli': 'crosschain check|transfer',
     'outputs': ['semantic_mapping_check.json', 'target_predictions.npz',
                 'crosschain_transfer.json']},
    {'requirement': 'P3-5', 'gap': '逐层/逐种子/逐时间窗稀疏与延迟审计',
     'core': ['layerwise_sparsity_audit'],
     'cli': 'sparsity-layerwise',
     'outputs': ['layerwise_sparsity.json']},
    {'requirement': 'P3-6', 'gap': 'H1–H5 按实际结果判定',
     'core': ['collect_hypothesis_observations', 'build_verdicts'],
     'cli': 'verdicts',
     'outputs': ['hypothesis_observations.json', 'hypothesis_verdicts.json']},
]


def cmd_report(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ('p3_core.py', 'p3_data.py', 'p3_experiments.py', 'test_p3.py',
                 'p3_evidence_manifest.json', 'p3_verdict_sources.json'):
        path = ROOT / name
        if path.exists():
            files[name] = {'bytes': path.stat().st_size,
                           'sha256': file_sha256(path)}
    atomic_json(out / 'traceability_matrix_p3.json',
                {'protocol': 'p3_traceability_v1', 'generated_utc': _now(),
                 'code_sha256': files, 'traceability': P3_TRACEABILITY,
                 'note': '程序存在不等于真实证据已产生；见 p3_evidence_manifest.json'})
    print(f'[P3] 追溯矩阵已写入 {out / "traceability_matrix_p3.json"}')
    return 0


# ------------------------------------------------------------------
def build_parser():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ap = argparse.ArgumentParser(description='TASGL P3-1 至 P3-6 真实证据入口')
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('independent', help='P3-1 独立划分声明与锁定评估审计')
    p.add_argument('--action', choices=['declare', 'lock', 'evaluate', 'audit'],
                   default='audit')
    p.add_argument('--dataset', default='dgraphfin')
    p.add_argument('--data_dir', default='./data/dgraphfin')
    p.add_argument('--prior_exposure', type=int, default=0)
    p.add_argument('--declared_by', default='operator_attestation')
    p.add_argument('--evidence', nargs='*', default=None)
    p.add_argument('--config', default=None)
    p.add_argument('--skip_code_verify', action='store_true')
    p.add_argument('--lock', default=None)
    p.add_argument('--evaluation', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--require_confirmatory', action='store_true')
    p.add_argument('--output_dir', default=str(P3_ROOT / 'independent'))
    p.set_defaults(func=cmd_independent)

    p = sub.add_parser('second-dataset', help='P3-2 DGraph-Fin 十种子训练与指标')
    p.add_argument('--action', choices=['train', 'report'], default='report')
    p.add_argument('--data_dir', default='./data/dgraphfin')
    p.add_argument('--model', choices=['mlp', 'graphsage'], default='mlp')
    p.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--fanout', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--evaluation', default=None)
    p.add_argument('--device', default=device)
    p.add_argument('--output_dir', default=str(P3_ROOT / 'second_dataset'))
    p.set_defaults(func=cmd_second_dataset)

    p = sub.add_parser('fair-comparison', help='P3-3 公平精度/效率/统计结论')
    p.add_argument('--spec', default=str(ROOT / 'p2_hpo_budget.json'))
    p.add_argument('--hpo_dir', default=str(ROOT / 'results' / 'p0' / 'hpo'))
    p.add_argument('--evaluation', nargs='+', default=None,
                   help='形如 NAME=PATH，可重复')
    p.add_argument('--efficiency', default=None)
    p.add_argument('--reference', default='tasgl')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--output_dir', default=str(P3_ROOT / 'fair_comparison'))
    p.set_defaults(func=cmd_fair_comparison)

    p = sub.add_parser('crosschain', help='P3-4 跨链语义核验与迁移')
    p.add_argument('--action', choices=['check', 'transfer'], default='check')
    p.add_argument('--mapping', required=True)
    p.add_argument('--aligned', default=None)
    p.add_argument('--source_columns', default=None)
    p.add_argument('--target_columns', default=None)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=device)
    p.add_argument('--output_dir', default=str(P3_ROOT / 'crosschain'))
    p.set_defaults(func=cmd_crosschain)

    p = sub.add_parser('sparsity-layerwise',
                       help='P3-5 逐层/逐种子/逐时间窗稀疏与延迟审计')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--checkpoint_pattern', default=None)
    p.add_argument('--hpo_dir', default=None)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--windows', type=int, nargs='+', default=[29, 34, 39, 44, 49])
    p.add_argument('--train_end', type=int, default=29)
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--rff_dim', type=int, default=16)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--num_hops', type=int, default=4)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--device', default=device)
    p.add_argument('--output_dir', default=str(P3_ROOT / 'sparsity_layerwise'))
    p.set_defaults(func=cmd_sparsity_layerwise)

    p = sub.add_parser('verdicts', help='P3-6 H1–H5 实际判定')
    p.add_argument('--sources', default=str(ROOT / 'p3_verdict_sources.json'))
    p.add_argument('--table', default=str(ROOT / 'p2_hypothesis_thresholds.json'))
    p.add_argument('--lock', default=None)
    p.add_argument('--output_dir', default=str(P3_ROOT / 'verdicts'))
    p.set_defaults(func=cmd_verdicts)

    p = sub.add_parser('report', help='P3 追溯矩阵')
    p.add_argument('--output_dir', default=str(P3_ROOT / 'traceability'))
    p.set_defaults(func=cmd_report)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
