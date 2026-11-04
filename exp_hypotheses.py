"""H1–H5 两阶段研发协议：开发集选择 -> 配置锁定 -> 最终确认。

选择阶段屏蔽 35–49 标签，只使用 1–34 的标签与 33–34 开发评估窗口。
确认阶段必须使用 select 生成的 lock 文件；默认写入 results/v4，不覆盖 v2/v3 历史结果。

用法：
  python exp_hypotheses.py --phase select --selection_seeds 0 1
  python exp_hypotheses.py --phase confirm --seeds 0 1 2 3 4
  python exp_hypotheses.py --phase report

注意：35–49 已被历史实验反复查看，因此这里的 confirm 是“锁定后探索性确认”，
不是独立确认性研究。严格确认仍需未参与开发的数据集或预先保留的新划分。
"""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import numpy as np
import pandas as pd
import torch

from common import (set_seed, save_json, load_json)
from data_loader import load_elliptic, make_graph_data
from metrics import measure_efficiency
from models import build_model, count_params
from models_v3 import TASGLite, LogitCalibrator, CalibratedInference
from exp_robustness import attack_edge_drop
from v3_engine import (V3_PROTOCOL, make_masks, train_lite, predict_logits,
                       fit_calibrator_and_threshold, evaluate_logits,
                       evaluate_model, get_teacher_logits, fit_edl_head,
                       _torch_save_atomic)


def candidate_configs():
    """预注册的少量候选；不得在读取 35–49 指标后追加候选。"""
    base = dict(hidden=48, hops=2, dropout=0.2, lr=1e-3, wd=1e-4,
                class_weight_power=0.5, focal_gamma=0.0,
                edge_drop=0.15, robust_ce_weight=0.5,
                consistency_weight=0.10, kd_weight=0.0,
                temporal_weight=0.02, teacher='graphsage',
                calibration='temp_bias')
    configs = []
    for name, updates in [
        ('lite_h48', {}),
        ('lite_h32', {'hidden': 32, 'edge_drop': 0.25,
                      'consistency_weight': 0.15}),
        ('lite_kd', {'hops': 1, 'kd_weight': 0.30}),
        ('graphless_kd', {'hops': 0, 'edge_drop': 0.0,
                          'robust_ce_weight': 0.0,
                          'consistency_weight': 0.0, 'kd_weight': 0.50}),
        ('lite_robust', {'hops': 1, 'edge_drop': 0.40,
                         'consistency_weight': 0.20}),
        ('lite_unweighted', {'class_weight_power': 0.0}),
        ('lite_temp', {'calibration': 'temp'}),
    ]:
        cfg = copy.deepcopy(base)
        cfg.update(updates)
        cfg['name'] = name
        configs.append(cfg)
    return configs


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), default=str)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode('utf-8')).hexdigest()


# 这些门槛来自预先声明的 H1–H5，而非查看 35–49 结果后反向设定。
# 开发通过只允许进入探索性确认，并不保证最终测试通过。
DEVELOPMENT_TARGETS = {
    'f1_min': 0.73,
    'params_max': 20000,
    'latency_ms_max': 8.0,
    'edge_drop50_rel_drop_max': 0.05,
    'extrapolation_loss_pass': 0.05,
    'extrapolation_loss_falsify': 0.08,
    'ece_max': 0.05,
}
SELECTION_POLICY = 'all_development_gates_must_pass_no_compensation'


class NoEligibleCandidateError(RuntimeError):
    """没有候选同时满足开发硬门槛；这是协议停止，不是工程崩溃。"""


def development_gate_status(row):
    """逐项判定开发门槛；任何失败均不得被其他目标的优势补偿。"""
    t = DEVELOPMENT_TARGETS
    gates = {
        'H1_f1': row['f1'] >= t['f1_min'],
        'H1_params': row['params'] <= t['params_max'],
        'H2_latency': row['latency_ms'] <= t['latency_ms_max'],
        'H3_absolute_drop_proxy': (
            row['edge_drop50_rel_drop'] <= t['edge_drop50_rel_drop_max']),
        # H4 的 5%–8% 灰色区也不能作为“全部通过”的候选进入确认。
        'H4_extrapolation_proxy': (
            row['extrapolation_loss'] <= t['extrapolation_loss_pass']),
        'H5_ece_proxy': row['ece'] <= t['ece_max'],
    }
    h4_status = ('pass' if row['extrapolation_loss'] <=
                 t['extrapolation_loss_pass'] else
                 'falsified' if row['extrapolation_loss'] >
                 t['extrapolation_loss_falsify'] else 'inconclusive')
    return {'development_gates': gates,
            'failed_gates': [name for name, ok in gates.items() if not ok],
            'h4_development_status': h4_status,
            'eligible': all(gates.values())}


def schedule_metadata(phase, seed, train_end, val_start, val_end,
                      test_start, test_end):
    return {'phase': phase, 'seed': int(seed), 'train_end': int(train_end),
            'val_start': int(val_start), 'val_end': int(val_end),
            'test_start': int(test_start), 'test_end': int(test_end)}


def result_valid(path, metadata, overwrite=False):
    if overwrite or not os.path.exists(path):
        return None
    try:
        obj = load_json(path)
    except Exception:
        return None
    if not isinstance(obj, dict) or obj.get('protocol') != V3_PROTOCOL:
        return None
    if obj.get('metadata') != metadata or not obj.get('complete'):
        return None
    return obj


def load_model_state(path, metadata, model, device, overwrite=False):
    if overwrite or not os.path.exists(path):
        return False
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
        if obj.get('protocol') != V3_PROTOCOL or obj.get('metadata') != metadata:
            return False
        model.load_state_dict(obj['model'])
        return True
    except Exception:
        return False


def run_student(config, seed, schedule, data, device, args, run_dir,
                phase, lock_id=None, include_edl=False, measure_eff=False):
    """训练/恢复一个轻量模型，返回 clean、删边、效率及可选 EDL 结果。"""
    train_end, val_start, val_end, test_start, test_end = schedule
    sm = schedule_metadata(phase, seed, train_end, val_start, val_end,
                           test_start, test_end)
    metadata = {'schedule': sm, 'config': config, 'lock_id': lock_id}
    os.makedirs(run_dir, exist_ok=True)
    stem = f'{config["name"]}_seed{seed}_train{train_end}'
    rp = os.path.join(run_dir, stem + '.json')
    mp = os.path.join(run_dir, stem + '.model')
    cp = os.path.join(run_dir, stem + '.ckpt')
    cached = result_valid(rp, metadata, args.overwrite)
    if cached is not None:
        print(f'[v3][{stem}] 已完成，跳过')
        return cached

    masks = make_masks(data['time_steps'].cpu().numpy(),
                       data['y'].cpu().numpy(), train_end, val_start,
                       val_end, test_start, test_end, device=device)
    teacher_logits = None
    if config['kd_weight'] > 0:
        teacher_dir = os.path.join(os.path.dirname(run_dir), 'teachers')
        os.makedirs(teacher_dir, exist_ok=True)
        teacher_meta = {'schedule': sm, 'teacher': config['teacher'],
                        'data_labels_note': ('35-49 labels masked' if phase == 'select'
                                             else 'full labels; masks restrict losses')}
        tp = os.path.join(teacher_dir,
                          f'{phase}_{config["teacher"]}_seed{seed}_'
                          f'train{train_end}_val{val_start}_{val_end}.pt')
        teacher_logits = get_teacher_logits(
            config['teacher'], data, masks, device, seed, args.epochs,
            tp, tp + '.ckpt', teacher_meta, overwrite=args.overwrite)

    set_seed(seed)
    model = TASGLite(data['in_dim'], hidden=config['hidden'],
                     hops=config['hops'], dropout=config['dropout'])
    if not load_model_state(mp, metadata, model, device, args.overwrite):
        if args.overwrite and os.path.exists(cp):
            os.remove(cp)
        model, best_val = train_lite(
            model, data, masks, config, device, args.epochs, args.patience,
            cp, metadata, teacher_logits=teacher_logits,
            verbose=args.verbose)
        _torch_save_atomic({'protocol': V3_PROTOCOL, 'metadata': metadata,
                            'model': model.state_dict(),
                            'best_val': best_val}, mp)
    model = model.to(device)
    model.eval()

    logits = predict_logits(model, data)
    cal, threshold, val_f1 = fit_calibrator_and_threshold(
        logits, data['y'], masks[1], config['calibration'])
    identity = LogitCalibrator('temp', 1.0, 0.0)
    clean = evaluate_logits(logits, data['y'], masks[2], cal, threshold)
    raw = evaluate_logits(logits, data['y'], masks[2], identity, threshold)
    attacked_data = attack_edge_drop(data, 0.5, seed + 50000)
    attacked = evaluate_model(model, attacked_data, masks[2], cal, threshold)

    result = {'protocol': V3_PROTOCOL, 'metadata': metadata,
              'config': config, 'seed': seed, 'complete': False,
              'calibrator': cal.to_dict(), 'validation_f1': val_f1,
              'clean': clean, 'clean_uncalibrated': raw,
              'edge_drop50': attacked, 'params_model': count_params(model)}
    # 后验校准也计入部署参数；教师只离线训练，不计入学生部署参数。
    result['params'] = result['params_model'] + (1 if config['calibration'] == 'temp' else 2)
    result['teacher_params_excluded'] = config['kd_weight'] > 0
    if measure_eff:
        repeats = args.selection_eff_repeat if phase == 'select' else args.eff_repeat
        # 测量包含神经网络、稀疏传播（若有）以及校准变换；不只测活跃边。
        result['efficiency'] = measure_efficiency(
            CalibratedInference(model, cal), data, device, n_repeat=repeats)
        result['efficiency']['calibration_included'] = True
        result['efficiency']['graph_preprocessing'] = (
            'row normalization cached after warmup; sparse propagation measured every forward')
    if include_edl:
        edl_probs, _, edl_val_nll = fit_edl_head(
            model, data, masks, config, device, epochs=args.edl_epochs)
        edl_logits = edl_probs.clamp(min=1e-9).log()
        edl_cal, edl_thr, _ = fit_calibrator_and_threshold(
            edl_logits, data['y'], masks[1], config['calibration'])
        result['edl'] = {
            'note': 'same-backbone EDL proxy; not a full ST-CGNN reproduction',
            'validation_nll': edl_val_nll,
            'calibrator': edl_cal.to_dict(),
            'calibrated': evaluate_logits(edl_logits, data['y'], masks[2],
                                          edl_cal, edl_thr),
            'uncalibrated': evaluate_logits(edl_logits, data['y'], masks[2],
                                            identity, edl_thr)}
    result['complete'] = True
    save_json(result, rp)
    print(f'[v3][{stem}] F1={clean["f1"]:.4f} '
          f'ECE={clean["ece"]:.4f} drop50={attacked["f1"]:.4f} '
          f'params={result["params"]:,}')
    return result


def run_baseline(name, seed, config, data, device, args, run_dir, lock_id):
    """H3 稠密基线：相同早停/校准协议，但不强加 TASGL 专用增强。"""
    schedule = (34, 30, 34, 35, 49)
    sm = schedule_metadata('confirm', seed, *schedule)
    bcfg = dict(config)
    bcfg.update(edge_drop=0.0, robust_ce_weight=0.0,
                consistency_weight=0.0, kd_weight=0.0,
                temporal_weight=0.0, focal_gamma=0.0)
    metadata = {'schedule': sm, 'baseline': name,
                'training_config': bcfg, 'lock_id': lock_id}
    stem = f'{name}_seed{seed}_train34'
    rp = os.path.join(run_dir, stem + '.json')
    cached = result_valid(rp, metadata, args.overwrite)
    if cached is not None:
        print(f'[v3][{stem}] 已完成，跳过')
        return cached
    masks = make_masks(data['time_steps'].cpu().numpy(),
                       data['y'].cpu().numpy(), *schedule, device=device)
    set_seed(seed)
    model = build_model(name, data['in_dim'])
    mp, cp = os.path.join(run_dir, stem + '.model'), os.path.join(run_dir, stem + '.ckpt')
    if not load_model_state(mp, metadata, model, device, args.overwrite):
        model, _ = train_lite(model, data, masks, bcfg, device,
                              args.epochs, args.patience, cp, metadata,
                              verbose=args.verbose)
        _torch_save_atomic({'protocol': V3_PROTOCOL, 'metadata': metadata,
                            'model': model.state_dict()}, mp)
    model = model.to(device)
    logits = predict_logits(model, data)
    cal, threshold, _ = fit_calibrator_and_threshold(
        logits, data['y'], masks[1], config['calibration'])
    clean = evaluate_logits(logits, data['y'], masks[2], cal, threshold)
    attacked_data = attack_edge_drop(data, 0.5, seed + 50000)
    attacked = evaluate_model(model, attacked_data, masks[2], cal, threshold)
    result = {'protocol': V3_PROTOCOL, 'metadata': metadata, 'complete': True,
              'seed': seed, 'clean': clean, 'edge_drop50': attacked,
              'calibrator': cal.to_dict(), 'params': count_params(model)}
    save_json(result, rp)
    print(f'[v3][{stem}] F1={clean["f1"]:.4f} drop50={attacked["f1"]:.4f}')
    return result


def relative_drop(clean, attacked):
    return (clean - attacked) / max(clean, 1e-12)


def selection_score(row):
    """合格候选的次序分及失败诊断分；不能抵消任何开发硬门槛。"""
    t = DEVELOPMENT_TARGETS
    score = (max(0.0, (t['f1_min'] - row['f1']) / t['f1_min'])
             + max(0.0, (row['latency_ms'] - t['latency_ms_max']) /
                   t['latency_ms_max'])
             + max(0.0, row['edge_drop50_rel_drop'] -
                   t['edge_drop50_rel_drop_max'])
             + max(0.0, row['extrapolation_loss'] -
                   t['extrapolation_loss_pass'])
             + max(0.0, row['ece'] - t['ece_max']))
    return float(score + 1e-3 * (1 - row['f1']) + 1e-6 * row['latency_ms'])


def run_select(data, device, args):
    # 硬屏蔽最终测试标签；即使代码错误也无法用它们计算损失或指标。
    data = dict(data)
    data['y'] = data['y'].clone()
    data['y'][data['time_steps'] >= 35] = -1
    run_root = os.path.join(args.out_root, 'select')
    all_rows, raw_results = [], {}
    for cfg in candidate_configs():
        refs, shorts = [], []
        for seed in args.selection_seeds:
            ref = run_student(cfg, seed, (29, 30, 32, 33, 34), data,
                              device, args, os.path.join(run_root, 'runs'),
                              phase='select', measure_eff=True)
            short = run_student(cfg, seed, (24, 25, 29, 33, 34), data,
                                device, args, os.path.join(run_root, 'runs'),
                                phase='select')
            refs.append(ref); shorts.append(short)
        f1 = float(np.mean([r['clean']['f1'] for r in refs]))
        attacked = float(np.mean([r['edge_drop50']['f1'] for r in refs]))
        short_f1 = float(np.mean([r['clean']['f1'] for r in shorts]))
        row = {'name': cfg['name'], 'f1': f1,
               'pr_auc': float(np.mean([r['clean']['pr_auc'] for r in refs])),
               'ece': float(np.mean([r['clean']['ece'] for r in refs])),
               'brier': float(np.mean([r['clean']['brier'] for r in refs])),
               'latency_ms': float(np.mean([r['efficiency']['latency_ms'] for r in refs])),
               'params': int(refs[0]['params']),
               'edge_drop50_f1': attacked,
               'edge_drop50_rel_drop': relative_drop(f1, attacked),
               'short_f1': short_f1,
               'extrapolation_loss': relative_drop(f1, short_f1)}
        row['selection_score'] = selection_score(row)
        row.update(development_gate_status(row))
        all_rows.append(row)
        raw_results[cfg['name']] = {'reference': refs, 'short': shorts}
        print('[select]', json.dumps(row, ensure_ascii=False))

    eligible = [r for r in all_rows if r['eligible']]
    winner = (min(eligible, key=lambda r: (r['selection_score'], r['name']))
              if eligible else None)
    summary = {'protocol': V3_PROTOCOL, 'test_labels_35_49_used': False,
               'selection_policy': SELECTION_POLICY,
               'development_targets': DEVELOPMENT_TARGETS,
               'selection_seeds': args.selection_seeds,
               'reference_schedule': [29, 30, 32, 33, 34],
               'short_schedule': [24, 25, 29, 33, 34],
               'rows': all_rows,
               'winner': winner['name'] if winner is not None else None,
               'confirmation_authorized': winner is not None}
    summary_hash = digest(summary)
    os.makedirs(run_root, exist_ok=True)
    # 即使无合格候选，也先完整保存开发证据，供下一轮开发前审计。
    save_json(summary, os.path.join(run_root, 'summary.json'))
    save_json(raw_results, os.path.join(run_root, 'raw_results.json'))
    pd.DataFrame(all_rows).sort_values(
        ['eligible', 'selection_score'], ascending=[False, True]).to_csv(
            os.path.join(run_root, 'summary.csv'), index=False)
    if winner is None:
        raise NoEligibleCandidateError(
            '没有候选同时通过全部开发硬门槛；开发结果已保存，'
            '未生成新锁，禁止进入确认')
    cfg = next(c for c in candidate_configs() if c['name'] == winner['name'])
    lock_id = digest({'config': cfg, 'selection_hash': summary_hash})[:16]
    lock = {'protocol': V3_PROTOCOL, 'lock_id': lock_id,
            'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'evidence_status': 'locked exploratory confirmation; test 35-49 previously observed',
            'selection_policy': SELECTION_POLICY,
            'selection_summary_sha256': summary_hash,
            'config_sha256': digest(cfg), 'config': cfg,
            'selection': {k: summary[k] for k in
                          ('selection_seeds', 'reference_schedule',
                           'short_schedule', 'test_labels_35_49_used',
                           'selection_policy', 'confirmation_authorized')},
            'final': {'seeds': [0, 1, 2, 3, 4],
                      'reference_schedule': [34, 30, 34, 35, 49],
                      'short_schedule': [25, 21, 25, 35, 49],
                      'edge_drop': 0.5, 'targets':
                          {'H1_f1': 0.73, 'H1_params': 20000,
                           'H2_latency_ms': 8.0, 'H3_baseline_fraction': 0.6,
                           'H4_loss_pass': 0.05, 'H4_loss_falsify': 0.08,
                           'H5_ece': 0.05}}}
    save_json(lock, args.lock)
    print(f'[select] 锁定 {cfg["name"]}; lock={args.lock}; id={lock_id}')
    return lock


def check_hypotheses(refs, shorts, baselines, seeds_complete):
    f1 = float(np.mean([r['clean']['f1'] for r in refs]))
    params = int(refs[0]['params'])
    latency = float(np.mean([r['efficiency']['latency_ms'] for r in refs]))
    ece = float(np.mean([r['clean']['ece'] for r in refs]))
    edl_ece = float(np.mean([r['edl']['calibrated']['ece'] for r in refs]))
    short_f1 = float(np.mean([r['clean']['f1'] for r in shorts]))
    extra = relative_drop(f1, short_f1)
    tas_attacked = float(np.mean([r['edge_drop50']['f1'] for r in refs]))
    tas_drop = relative_drop(f1, tas_attacked)

    checks = {
        'H1': {'f1_mean': f1, 'params': params,
               'status': 'pass' if f1 >= 0.73 and params <= 20000 else 'falsified'},
        'H2': {'latency_ms_mean': latency,
               'status': 'pass' if latency <= 8 else 'falsified'},
        'H4': {'reference_f1': f1, 'short_f1': short_f1,
               'relative_loss_ratio_of_means': extra,
               'status': ('pass' if extra <= 0.05 else
                          'falsified' if extra > 0.08 else 'inconclusive')},
        'H5': {'ece_mean': ece, 'edl_same_backbone_ece_mean': edl_ece,
               'brier_mean': float(np.mean([r['clean']['brier'] for r in refs])),
               'nll_mean': float(np.mean([r['clean']['nll'] for r in refs])),
               'status': ('pass' if ece <= 0.05 and ece < edl_ece else
                          'falsified' if ece >= edl_ece else 'inconclusive')},
    }
    h3 = {'tasgl_clean_f1': f1, 'tasgl_edge_drop50_f1': tas_attacked,
          'tasgl_relative_drop': tas_drop, 'per_baseline': {}}
    for name, runs in baselines.items():
        bc = float(np.mean([r['clean']['f1'] for r in runs]))
        ba = float(np.mean([r['edge_drop50']['f1'] for r in runs]))
        bd = relative_drop(bc, ba)
        if bd <= 0:
            status = 'inconclusive_nonpositive_baseline_drop'
        elif tas_drop <= 0.6 * bd:
            status = 'pass'
        elif tas_drop >= bd:
            status = 'falsified'
        else:
            status = 'inconclusive'
        h3['per_baseline'][name] = {
            'clean_f1': bc, 'edge_drop50_f1': ba, 'relative_drop': bd,
            'target_tasgl_drop': 0.6 * bd if bd > 0 else None,
            'status': status}
    statuses = [v['status'] for v in h3['per_baseline'].values()]
    if statuses and all(s == 'pass' for s in statuses):
        h3['status'] = 'pass'
    elif any(s == 'falsified' for s in statuses):
        h3['status'] = 'falsified'
    else:
        h3['status'] = 'inconclusive'
    checks['H3'] = h3
    for h in checks.values():
        if not seeds_complete:
            h['evidence_complete'] = False
            h['status'] = 'incomplete_' + h['status']
        else:
            h['evidence_complete'] = True
    return checks


def run_confirm(data, device, args, lock):
    if digest(lock['config']) != lock['config_sha256']:
        raise ValueError('lock 配置哈希不匹配，拒绝确认')
    if digest({'config': lock['config'],
               'selection_hash': lock['selection_summary_sha256']})[:16] != lock['lock_id']:
        raise ValueError('lock_id 与配置/选择摘要不匹配，拒绝确认')
    selection_path = os.path.join(args.out_root, 'select', 'summary.json')
    if not os.path.exists(selection_path):
        raise ValueError('开发集摘要缺失，拒绝把本次运行称为锁定确认')
    selection_summary = load_json(selection_path)
    if digest(selection_summary) != lock['selection_summary_sha256']:
        raise ValueError('开发集摘要已被修改，拒绝把本次运行称为锁定确认')
    if (lock.get('selection_policy') != SELECTION_POLICY or
            selection_summary.get('selection_policy') != SELECTION_POLICY or
            selection_summary.get('confirmation_authorized') is not True or
            selection_summary.get('winner') != lock['config']['name']):
        raise ValueError('开发硬门槛未全部通过或锁策略不匹配，拒绝确认')
    cfg = lock['config']
    lock_id = lock['lock_id']
    out_dir = os.path.join(args.out_root, 'confirm_' + lock_id)
    run_dir = os.path.join(out_dir, 'runs')
    os.makedirs(run_dir, exist_ok=True)
    refs, shorts = [], []
    for seed in args.seeds:
        refs.append(run_student(cfg, seed, (34, 30, 34, 35, 49), data,
                                device, args, run_dir, phase='confirm',
                                lock_id=lock_id, include_edl=True,
                                measure_eff=True))
        shorts.append(run_student(cfg, seed, (25, 21, 25, 35, 49), data,
                                  device, args, run_dir, phase='confirm',
                                  lock_id=lock_id))
    baselines = {}
    for name in args.baselines:
        baselines[name] = [run_baseline(name, seed, cfg, data, device,
                                       args, run_dir, lock_id)
                           for seed in args.seeds]
    complete = sorted(args.seeds) == [0, 1, 2, 3, 4]
    checks = check_hypotheses(refs, shorts, baselines, complete)
    summary = {'protocol': V3_PROTOCOL, 'lock_id': lock_id,
               'evidence_status': lock['evidence_status'], 'config': cfg,
               'seeds': args.seeds, 'hypotheses': checks,
               'all_pass': all(v['status'] == 'pass' for v in checks.values()),
               'note': 'all_pass=false 时必须如实报告；不得根据本文件继续调参并重称确认'}
    save_json(summary, os.path.join(out_dir, 'hypothesis_summary.json'))
    save_json({'reference': refs, 'short': shorts, 'baselines': baselines},
              os.path.join(out_dir, 'raw_results.json'))
    rows = []
    for h, value in checks.items():
        rows.append({'hypothesis': h, 'status': value['status'],
                     'details': canonical(value)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'hypothesis_summary.csv'),
                              index=False)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f'[confirm] all_pass={summary["all_pass"]}; 输出={out_dir}')
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['select', 'confirm', 'report', 'all'],
                    default='all')
    ap.add_argument('--data_dir', default='./data/elliptic')
    ap.add_argument('--out_root', default='results/v4')
    ap.add_argument('--lock', default='results/v4/lock.json')
    ap.add_argument('--selection_seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--baselines', nargs='+', default=['gcn', 'gat'])
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--edl_epochs', type=int, default=100)
    ap.add_argument('--selection_eff_repeat', type=int, default=20)
    ap.add_argument('--eff_repeat', type=int, default=100)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    if args.seeds is None:
        args.seeds = [0, 1, 2, 3, 4]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}; phase={args.phase}; protocol={V3_PROTOCOL}')
    if args.phase == 'report':
        lock = load_json(args.lock)
        path = os.path.join(args.out_root, 'confirm_' + lock['lock_id'],
                            'hypothesis_summary.json')
        print(json.dumps(load_json(path), ensure_ascii=False, indent=2))
        return
    X, labels, time_steps, edge_index, _ = load_elliptic(args.data_dir)
    data = make_graph_data(X, labels, time_steps, edge_index, device)
    lock = None
    if args.phase in ('select', 'all'):
        try:
            lock = run_select(data, device, args)
        except NoEligibleCandidateError as exc:
            print(f'[select] 协议停止：{exc}')
            return 2
    if args.phase in ('confirm', 'all'):
        if lock is None:
            if not os.path.exists(args.lock):
                raise FileNotFoundError('必须先运行 --phase select 生成 lock')
            lock = load_json(args.lock)
        args.seeds = list(args.seeds)
        run_confirm(data, device, args, lock)


if __name__ == '__main__':
    raise SystemExit(main())
