"""p2_experiments.py — P2-1 至 P2-5 统一实验入口。

证据边界：本程序实现协议、审计与判定工具，不预设任何假设通过。所有判定只
来自传入的真实观测；缺失数据一律标记 incomplete_*，不得写为达标。
新结果隔离在 results/p2/，不覆盖 results/ 下 v2/v3/v4/p0/p1 历史证据。
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

from p0_core import atomic_json, canonical_json, file_sha256, read_json, \
    sha256_bytes
from p0_data import load_elliptic_raw
from p2_core import (DEFAULT_SEED_POOL, MIN_SEEDS, MAX_SEEDS,
                     apply_stratified_temperature, apply_threshold_table,
                     audit_hpo_fairness, audit_strict_rolling,
                     audit_window_disjointness, calibration_quality_report,
                     check_prediction_contamination, classify_window,
                     declare_independent_window, fit_stratified_temperature,
                     hpo_budget_spec_hash, minimum_detectable_effect,
                     monotonicity_report, multi_comparison,
                     paired_statistics_multi, register_thresholds,
                     required_seeds_for_mde, scan_window_usage,
                     seed_pool_hash, validate_seed_pool,
                     verify_thresholds, wilcoxon_min_p, window_evidence_class)

ROOT = Path(__file__).resolve().parent
P2_ROOT = ROOT / 'results' / 'p2'
CODE_FILES = [ROOT / n for n in ('p2_experiments.py', 'p2_core.py',
                                 'p0_core.py', 'p0_data.py', 'p0_models.py',
                                 'test_p2.py')]


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


# ------------------------------------------------------------------
# P2-1：种子池注册、校验与统计灵敏度
# ------------------------------------------------------------------
def cmd_seeds(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = validate_seed_pool(args.seeds if args.seeds else DEFAULT_SEED_POOL)
    registry_path = Path(args.registry) if args.registry else \
        out / 'seed_registry.json'
    payload = {'schema': 'p2_seed_registry_v1',
               'seeds': seeds,
               'n_seeds': len(seeds),
               'paired_unit': 'seed',
               'test': 'wilcoxon_exact_two_sided',
               'alpha': args.alpha}
    digest = seed_pool_hash(seeds, payload)
    if args.action == 'register':
        atomic_json(registry_path, {**payload, 'seed_pool_sha256': digest,
                                    'registered_utc': _now()})
        print(f'[P2-1] 已注册 {len(seeds)} 个种子; sha256={digest[:16]}')
        print(f'[P2-1] 精确双侧 Wilcoxon 最小可达 p = {wilcoxon_min_p(len(seeds)):.6f}')
        return 0
    if args.action == 'check':
        if not registry_path.exists():
            raise FileNotFoundError(f'种子注册表不存在: {registry_path}')
        old = read_json(registry_path)
        if old.get('seed_pool_sha256') != digest:
            raise RuntimeError('种子池与已注册不一致；事后增删种子不得视为同一轮预注册')
        print(f'[P2-1] 种子池一致: {len(seeds)} 个; sha256={digest[:16]}')
        return 0
    # power：预注册灵敏度表
    rows = []
    for n in range(MIN_SEEDS, MAX_SEEDS + 1):
        row = minimum_detectable_effect(n, args.sd, args.alpha, args.power)
        row['required_seeds_for_target'] = None
        rows.append(row)
    result = {'protocol': 'p2_seed_power_v1', 'alpha': args.alpha,
              'power': args.power, 'assumed_sd_of_differences': args.sd,
              'declared_seeds': seeds, 'seed_pool_sha256': digest,
              'sensitivity_table': rows,
              'note': ('MDE 为预注册灵敏度；事后用观测 sd 反推功效不能证明无差异，'
                       '也不得据此增删种子')}
    if args.sd and args.target_mde:
        result['required_seeds_for_target'] = required_seeds_for_mde(
            args.target_mde, args.sd, args.alpha, args.power)
    atomic_json(out / 'seed_power.json', result)
    for row in rows:
        print(f"[P2-1] n={row['n']:2d} 所需 dz={row['required_cohen_dz']:.4f} "
              f"MDE={row['mde_units'] if row['mde_units'] is None else round(row['mde_units'], 5)} "
              f"Wilcoxon 最小 p={row['wilcoxon_min_attainable_p']:.6f}")
    print(f'[P2-1] 灵敏度表已写入 {out / "seed_power.json"}')
    return 0


def cmd_statistics(args):
    """多种子配对显著性：只接受已注册种子池，缺种子即 incomplete。"""
    config = read_json(args.config)
    seeds = validate_seed_pool(config['seeds'])
    alpha = float(config.get('alpha', 0.05))

    def _records(value):
        return read_json(value) if isinstance(value, str) else value

    rows = []
    for comp in config['comparisons']:
        key = comp.get('value_key', 'f1')
        row = paired_statistics_multi(
            _records(comp['a_records']), _records(comp['b_records']),
            seeds, key, alpha,
            n_boot=int(config.get('n_boot', 10000)),
            boot_seed=int(config.get('boot_seed', 0)),
            sd_prior=comp.get('sd_prior'))
        row['name'] = comp['name']
        rows.append(row)
    p_values = [r.get('p_value') for r in rows]
    adjusted, flags = multi_comparison(p_values, alpha)
    for row, adj, flag in zip(rows, adjusted, flags):
        row['p_holm'] = adj
        row['significant_after_holm'] = flag
    complete = all(r['status'] == 'pass_complete' for r in rows)
    result = {'protocol': 'p2_multiseed_statistics_v1',
              'config_sha256': sha256_bytes(canonical_json(config).encode()),
              'declared_seeds': seeds, 'n_seeds': len(seeds),
              'alpha': alpha, 'multiple_comparison': 'Holm',
              'complete': complete, 'comparisons': rows,
              'note': ('不完整种子、不显著或负结果均须原样进入论文；'
                       '不得事后换检验、换种子或选择性报告')}
    atomic_json(args.output, result)
    print(f'[P2-1] complete={complete}; n_seeds={len(seeds)}; 输出={args.output}')
    return 0


# ------------------------------------------------------------------
# P2-2：预固定 H1–H5 门限表
# ------------------------------------------------------------------
def cmd_thresholds(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.action == 'register':
        table = read_json(args.table)
        lock = register_thresholds(table, out, {'registered_utc': _now(),
                                                'registered_by': 'p2_experiments'})
        print(f'[P2-2] 门限表已注册; lock_id={lock["lock_id"]}; '
              f'table_sha256={lock["table_sha256"][:16]}')
        print('[P2-2] 注册后不得再修改阈值；修改即视为新的探索版本')
        return 0
    if args.action == 'verify':
        table = read_json(args.table)
        lock = read_json(args.lock)
        errors = verify_thresholds(table, lock)
        if errors:
            for e in errors:
                print(f'[P2-2][FAIL] {e}')
            raise RuntimeError('门限表与注册锁不一致；禁止事后改阈值')
        print(f'[P2-2] 门限表与锁一致; lock_id={lock.get("lock_id")}')
        return 0
    # apply
    table = read_json(args.table)
    lock = read_json(args.lock) if args.lock and Path(args.lock).exists() else None
    if lock is not None:
        errors = verify_thresholds(table, lock)
        if errors:
            raise RuntimeError(f'门限表已被修改: {errors}')
    observations = read_json(args.observations)
    declared = args.seeds if args.seeds else table.get('declared_seeds')
    result = apply_threshold_table(table, observations, lock, declared)
    result['applied_utc'] = _now()
    atomic_json(out / 'hypothesis_verdicts.json', result)
    for name, entry in sorted(result['hypotheses'].items()):
        print(f'[P2-2] {name}: {entry["status"]}')
    print(f'[P2-2] all_pass={result["all_pass"]}')
    return 0


# ------------------------------------------------------------------
# P2-3：独立时间窗与严格 rolling 审计
# ------------------------------------------------------------------
def cmd_windows(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = read_json(args.ledger)
    if args.action == 'declare':
        if args.reserve_start is None or args.reserve_end is None:
            raise ValueError('declare 需要 --reserve_start 与 --reserve_end')
        result = declare_independent_window(ledger, args.reserve_start,
                                            args.reserve_end)
        result['declared_utc'] = _now()
        atomic_json(out / 'independent_window_declaration.json', result)
        if not result['independent_window_available']:
            print('[P2-3][STOP] 请求区间与已暴露窗口重叠，不能声明为独立确认窗')
            for c in result['conflicts']:
                print(f"[P2-3] 冲突窗口 {c['window']} {c['range']} "
                      f"({c['classification']})")
            return 2
        print(f'[P2-3] 独立窗口可用: {result["requested_range"]}')
        return 0
    if args.action == 'scan':
        if args.reserve_start is None or args.reserve_end is None:
            raise ValueError('scan 需要 --reserve_start 与 --reserve_end')
        reserved = (args.reserve_start, args.reserve_end)
        usage = scan_window_usage(args.scan_dirs or [], reserved)
        preds = None
        if args.predictions:
            raw = load_elliptic_raw(args.data_dir)
            preds = check_prediction_contamination(args.predictions, raw,
                                                   reserved)
        result = {'protocol': 'p2_window_contamination_scan_v1',
                  'config_usage': usage, 'predictions': preds,
                  'clean': usage['clean'] and (preds is None or preds['clean']),
                  'scan_dirs': [str(p) for p in (args.scan_dirs or [])]}
        atomic_json(out / 'contamination_scan.json', result)
        print(f'[P2-3] 配置命中 {len(usage["hits"])} 处; '
              f'clean={result["clean"]}')
        return 0 if result['clean'] else 2
    # audit
    report = {'protocol': 'p2_window_and_rolling_audit_v1',
              'ledger': ledger.get('schema'),
              'windows': {},
              'disjointness': audit_window_disjointness(
                  ledger.get('windows', {}))}
    for name, entry in sorted(ledger.get('windows', {}).items()):
        report['windows'][name] = {
            'range': entry.get('range'), 'role': entry.get('role'),
            'dataset': entry.get('dataset', ledger.get('dataset')),
            'classification': classify_window(entry),
            'evidence_class': window_evidence_class(entry),
            'prior_exposure': bool(entry.get('prior_exposure')),
            'used_for': entry.get('used_for', [])}
    rolling = []
    rolling_skipped = None
    if args.data_dir and not Path(args.data_dir).exists():
        rolling_skipped = f'数据目录不存在: {args.data_dir}；跳过逐窗重算'
        print(f'[P2-3][SKIP] {rolling_skipped}')
    elif args.data_dir:
        raw = load_elliptic_raw(args.data_dir,
                                label_ceiling=args.label_ceiling)
        recorded = {}
        if args.recorded and Path(args.recorded).exists():
            obj = read_json(args.recorded)
            for item in obj.get('window_audit', []) if isinstance(obj, dict) \
                    else obj:
                if isinstance(item, dict) and 'time' in item:
                    recorded[int(item['time'])] = item
        for cutoff in args.cutoffs:
            audit = audit_strict_rolling(raw, cutoff, args.train_end,
                                         args.label_ceiling,
                                         recorded.get(int(cutoff)))
            rolling.append(audit)
            print(f"[P2-3] cutoff={cutoff} 可见节点={audit['visible_nodes']} "
                  f"可见边={audit['visible_edges']} "
                  f"污染={not audit['contamination_free']}")
    report['strict_rolling'] = rolling
    report['rolling_audit_skipped'] = rolling_skipped
    report['rolling_contamination_free'] = all(r['contamination_free']
                                               for r in rolling) if rolling else None
    by_dataset = {}
    for name, info in report['windows'].items():
        ds = info.get('dataset') or 'unknown'
        by_dataset.setdefault(ds, []).append(
            {'window': name, 'classification': info['classification']})
    report['independent_window_by_dataset'] = {
        ds: any(w['classification'] == 'held_out_independent' for w in ws)
        for ds, ws in sorted(by_dataset.items())}
    report['independent_window_available'] = bool(
        report['independent_window_by_dataset'].get(ledger.get('dataset')))
    atomic_json(out / 'window_audit.json', report)
    print(f'[P2-3] 主数据集独立窗口可用='
          f'{report["independent_window_available"]}; '
          f'逐数据集={report["independent_window_by_dataset"]}; '
          f'rolling 无污染={report["rolling_contamination_free"]}')
    return 0


# ------------------------------------------------------------------
# P2-4：基线公平超参搜索
# ------------------------------------------------------------------
def cmd_hpo(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = read_json(args.spec)
    if args.action == 'spec':
        result = {'protocol': 'p2_hpo_budget_spec_v1', 'spec': spec,
                  'spec_sha256': hpo_budget_spec_hash(spec),
                  'registered_utc': _now()}
        atomic_json(out / 'hpo_budget_spec.lock.json', result)
        print(f"[P2-4] 预算规格已注册; sha256={result['spec_sha256'][:16]}")
        return 0
    winners_path = Path(args.hpo_dir) / 'hpo_winners.json'
    if not winners_path.exists():
        raise FileNotFoundError(f'HPO winners 不存在: {winners_path}')
    result = audit_hpo_fairness(read_json(winners_path), spec,
                                str(Path(args.hpo_dir) / 'runs'))
    result['audited_utc'] = _now()
    result['hpo_dir'] = str(Path(args.hpo_dir).resolve())
    atomic_json(out / 'hpo_fairness_audit.json', result)
    for name, row in sorted(result['per_model'].items()):
        print(f"[P2-4] {name}: present={row.get('present')} "
              f"cost={row.get('search_cost_runs')}")
    if not result['fair']:
        print('[P2-4][STOP] 预算不公平，不能称为公平基线对比')
        for v in result['violations']:
            print(f'[P2-4] {v}')
        return 2
    print('[P2-4] 等预算审计通过')
    return 0


# ------------------------------------------------------------------
# P2-5：正类 ECE 与分层校准
# ------------------------------------------------------------------
def _load_labeled(path):
    """读取带标签的预测文件（.npz 或 .json）。

    仅用于开发/校准窗口或已授权锁后的最终预测；程序不自动获取测试标签。
    """
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as z:
            data = {k: np.asarray(z[k]) for k in z.files}
    else:
        obj = read_json(path)
        data = {k: np.asarray(v) for k, v in obj.items()}
    y = None
    for key in ('y', 'y_true', 'labels'):
        if key in data:
            y = np.asarray(data[key], dtype=np.int64).reshape(-1)
            break
    if y is None:
        raise ValueError(f'{path} 缺少标签字段 y/y_true/labels')
    if 'p_prob' in data:
        p = np.asarray(data['p_prob'], dtype=np.float64).reshape(-1)
    elif 'p' in data:
        p = np.asarray(data['p'], dtype=np.float64).reshape(-1)
    elif 'logits' in data:
        p = _sigmoid(np.asarray(data['logits'], dtype=np.float64).reshape(-1))
    else:
        raise ValueError(f'{path} 缺少 p_prob/p/logits')
    if len(y) != len(p):
        raise ValueError('标签与概率长度不一致')
    extras = {k: v for k, v in data.items()
              if k not in ('y', 'y_true', 'labels', 'p_prob', 'p', 'logits')}
    return y, p, extras


def cmd_calibration(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    y, p, extras = _load_labeled(args.input)
    strata = None
    if args.strata_fields:
        strata = {}
        for field in args.strata_fields:
            if field not in extras:
                raise ValueError(f'{args.input} 缺少分层字段 {field}')
            strata[field] = extras[field]
    report = calibration_quality_report(y, p, threshold=args.threshold,
                                        bins=args.bins,
                                        min_count=args.min_count,
                                        strata=strata,
                                        min_stratum_n=args.min_stratum_n)
    report['input'] = str(Path(args.input).resolve())
    report['input_sha256'] = file_sha256(args.input)
    atomic_json(out / 'calibration_quality.json', report)
    print(f"[P2-5] 整体 ECE={report['ece_overall']:.5f} "
          f"正类 ECE={report['positive_class_ece']:.5f} "
          f"负类 ECE={report['negative_class_ece']:.5f}")
    if report['positive_class_degenerate_single_bin']:
        print('[P2-5][WARN] 正类样本只落在 <=1 个桶，正类 ECE 不可解释为校准良好')
    if report['strata']:
        for name, block in sorted(report['strata'].items()):
            print(f"[P2-5] 分层 {name}: max ECE={block['stratified_ece_max']} "
                  f"不足层={block['n_strata_insufficient']}")
    return 0


def cmd_stratified_calibration(args):
    """分层校准：在互不重叠的拟合窗与评估窗上比较全局温度与分层温度。"""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fit_path, eval_path = Path(args.fit), Path(args.eval)
    y_fit, _, fit_extra = _load_labeled(fit_path)
    y_eval, _, eval_extra = _load_labeled(eval_path)
    if args.strata_field not in fit_extra or args.strata_field not in eval_extra:
        raise ValueError(f'拟合/评估文件均需包含分层字段 {args.strata_field}')
    with np.load(fit_path, allow_pickle=False) as z:
        fit_logits = np.asarray(z['logits'], dtype=np.float64).reshape(-1)
    with np.load(eval_path, allow_pickle=False) as z:
        eval_logits = np.asarray(z['logits'], dtype=np.float64).reshape(-1)
    fit_strata = np.asarray(fit_extra[args.strata_field])
    eval_strata = np.asarray(eval_extra[args.strata_field])
    if args.fit_end is not None and args.eval_start is not None:
        if int(args.fit_end) >= int(args.eval_start):
            raise ValueError('校准拟合窗与评估窗必须互不重叠')
    calibrator = fit_stratified_temperature(fit_logits, y_fit, fit_strata,
                                            min_pos=args.min_pos,
                                            min_neg=args.min_neg)
    raw_p = _sigmoid(eval_logits)
    global_p = _sigmoid(eval_logits /
                        np.exp(calibrator['global']['log_temperature']))
    strat_p = apply_stratified_temperature(calibrator, eval_logits, eval_strata)
    blocks = {}
    for name, p in (('raw', raw_p), ('global_temperature', global_p),
                    ('stratified_temperature', strat_p)):
        quality = calibration_quality_report(y_eval, p, threshold=args.threshold,
                                             bins=args.bins,
                                             min_count=args.min_count,
                                             strata={args.strata_field: eval_strata},
                                             min_stratum_n=args.min_stratum_n)
        quality['monotonicity_vs_raw'] = None if name == 'raw' else \
            monotonicity_report(raw_p, p)
        blocks[name] = quality
    result = {'protocol': 'p2_stratified_calibration_v1',
              'strata_field': args.strata_field,
              'fit_file': str(fit_path.resolve()),
              'eval_file': str(eval_path.resolve()),
              'fit_sha256': file_sha256(fit_path),
              'eval_sha256': file_sha256(eval_path),
              'calibrator': calibrator,
              'comparisons': blocks,
              'note': ('层内样本不足时回退全局温度并显式标记；'
                       '分层校准只改变概率刻度，不改变单调排序时可以保持 F1 可比')}
    atomic_json(out / 'stratified_calibration.json', result)
    for name in ('raw', 'global_temperature', 'stratified_temperature'):
        b = blocks[name]
        print(f"[P2-5] {name}: ECE={b['ece_overall']:.5f} "
              f"正类 ECE={b['positive_class_ece']:.5f} "
              f"分层 max ECE={b['strata'][args.strata_field]['stratified_ece_max']}")
    return 0


# ------------------------------------------------------------------
# 追溯矩阵
# ------------------------------------------------------------------
TRACEABILITY = [
    {'requirement': 'P2-1', 'gap': '10–20 seeds 显著性检验',
     'core': ['validate_seed_pool', 'seed_pool_hash', 'paired_statistics_multi',
              'minimum_detectable_effect', 'bootstrap_mean_ci', 'sign_test_p',
              'multi_comparison'],
     'cli': 'seeds / statistics',
     'tests': ['test_seed_pool_registration_and_hash',
               'test_wilcoxon_min_p_and_mde_monotone',
               'test_paired_statistics_multi_missing_seeds',
               'test_nested_units_collapsed_to_seed'],
     'outputs': ['seed_registry.json', 'seed_power.json', 'statistics.json']},
    {'requirement': 'P2-2', 'gap': '预固定 H1–H5 门限表',
     'core': ['register_thresholds', 'verify_thresholds',
              'apply_threshold_table', 'evaluate_gate'],
     'cli': 'thresholds register|verify|apply',
     'tests': ['test_threshold_register_and_tamper_detection',
               'test_apply_h1_pass_falsified_incomplete',
               'test_h4_grey_zone_is_inconclusive',
               'test_h3_nonpositive_baseline_inconclusive',
               'test_h5_above_ceiling_but_better_is_inconclusive'],
     'outputs': ['thresholds.lock.json', 'hypothesis_verdicts.json']},
    {'requirement': 'P2-3', 'gap': '未被污染的独立时间窗与严格 rolling',
     'core': ['declare_independent_window', 'audit_strict_rolling',
              'scan_window_usage', 'check_prediction_contamination',
              'audit_window_disjointness'],
     'cli': 'windows declare|audit|scan',
     'tests': ['test_declare_independent_window_rejects_exposed_range',
               'test_strict_rolling_audit_detects_future_leakage',
               'test_scan_window_usage_detects_reserved_window',
               'test_prediction_contamination_check'],
     'outputs': ['independent_window_declaration.json', 'window_audit.json',
                 'contamination_scan.json']},
    {'requirement': 'P2-4', 'gap': '基线公平超参搜索',
     'core': ['audit_hpo_fairness', 'hpo_budget_spec_hash'],
     'cli': 'hpo spec|audit',
     'tests': ['test_hpo_fairness_equal_budget',
               'test_hpo_fairness_detects_extra_budget'],
     'outputs': ['hpo_budget_spec.lock.json', 'hpo_fairness_audit.json']},
    {'requirement': 'P2-5', 'gap': '正类 ECE 与分层校准',
     'core': ['positive_class_ece', 'negative_class_ece', 'stratified_ece',
              'calibration_quality_report', 'fit_stratified_temperature'],
     'cli': 'calibration / stratified-calibration',
     'tests': ['test_positive_class_ece_values',
               'test_stratified_ece_insufficient_stratum',
               'test_stratified_temperature_fallback_and_monotonicity',
               'test_calibration_cli_end_to_end'],
     'outputs': ['calibration_quality.json', 'stratified_calibration.json']},
]


def cmd_report(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in CODE_FILES + [ROOT / n for n in (
            'p2_seed_registry.json', 'p2_hypothesis_thresholds.json',
            'p2_window_ledger.json', 'p2_hpo_budget.json')]:
        if path.exists():
            files[str(path.name)] = {'bytes': path.stat().st_size,
                                     'sha256': file_sha256(path)}
    result = {'protocol': 'p2_traceability_v1', 'generated_utc': _now(),
              'code_sha256': files, 'traceability': TRACEABILITY,
              'note': ('本矩阵只说明“缺口—程序—测试—产物”的对应关系；'
                       '程序存在不等于假设已通过')}
    atomic_json(out / 'traceability_matrix.json', result)
    print(f'[P2] 追溯矩阵已写入 {out / "traceability_matrix.json"} '
          f'({len(TRACEABILITY)} 项缺口)')
    return 0


# ------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description='TASGL P2-1 至 P2-5 可审计实验')
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('seeds', help='P2-1 种子池注册/校验/灵敏度')
    p.add_argument('--action', choices=['register', 'check', 'power'],
                   default='power')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--registry', default=None)
    p.add_argument('--sd', type=float, default=None,
                   help='配对差值的先验标准差；用于换算 MDE 原始单位')
    p.add_argument('--target_mde', type=float, default=None)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--power', type=float, default=0.8)
    p.add_argument('--output_dir', default=str(P2_ROOT / 'seeds'))
    p.set_defaults(func=cmd_seeds)

    p = sub.add_parser('statistics', help='P2-1 多种子配对显著性检验')
    p.add_argument('--config', required=True)
    p.add_argument('--output', default=str(P2_ROOT / 'statistics.json'))
    p.set_defaults(func=cmd_statistics)

    p = sub.add_parser('thresholds', help='P2-2 预固定 H1–H5 门限表')
    p.add_argument('--action', choices=['register', 'verify', 'apply'],
                   default='register')
    p.add_argument('--table', default=str(ROOT / 'p2_hypothesis_thresholds.json'))
    p.add_argument('--lock', default=None)
    p.add_argument('--observations', default=None,
                   help='apply 时传入真实观测 JSON')
    p.add_argument('--seeds', type=int, nargs='+', default=None)
    p.add_argument('--output_dir', default=str(P2_ROOT / 'thresholds'))
    p.set_defaults(func=cmd_thresholds)

    p = sub.add_parser('windows', help='P2-3 独立时间窗与严格 rolling 审计')
    p.add_argument('--action', choices=['declare', 'audit', 'scan'],
                   default='audit')
    p.add_argument('--ledger', default=str(ROOT / 'p2_window_ledger.json'))
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--train_end', type=int, default=29)
    p.add_argument('--label_ceiling', type=int, default=34)
    p.add_argument('--cutoffs', type=int, nargs='+', default=[34, 39, 44, 49])
    p.add_argument('--recorded', default=None)
    p.add_argument('--reserve_start', type=int, default=None)
    p.add_argument('--reserve_end', type=int, default=None)
    p.add_argument('--scan_dirs', nargs='+', default=None)
    p.add_argument('--predictions', nargs='+', default=None)
    p.add_argument('--output_dir', default=str(P2_ROOT / 'windows'))
    p.set_defaults(func=cmd_windows)

    p = sub.add_parser('hpo', help='P2-4 基线公平超参搜索预算审计')
    p.add_argument('--action', choices=['spec', 'audit'], default='audit')
    p.add_argument('--spec', default=str(ROOT / 'p2_hpo_budget.json'))
    p.add_argument('--hpo_dir', default=str(ROOT / 'results' / 'p0' / 'hpo'))
    p.add_argument('--output_dir', default=str(P2_ROOT / 'hpo'))
    p.set_defaults(func=cmd_hpo)

    p = sub.add_parser('calibration', help='P2-5 正类 ECE 与分层校准诊断')
    p.add_argument('--input', required=True)
    p.add_argument('--bins', type=int, default=15)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--min_count', type=int, default=1)
    p.add_argument('--min_stratum_n', type=int, default=100)
    p.add_argument('--strata_fields', nargs='+', default=None)
    p.add_argument('--output_dir', default=str(P2_ROOT / 'calibration'))
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser('stratified-calibration',
                       help='P2-5 分层温度校准对照')
    p.add_argument('--fit', required=True)
    p.add_argument('--eval', required=True)
    p.add_argument('--strata_field', default='time')
    p.add_argument('--fit_end', type=int, default=None)
    p.add_argument('--eval_start', type=int, default=None)
    p.add_argument('--min_pos', type=int, default=20)
    p.add_argument('--min_neg', type=int, default=20)
    p.add_argument('--bins', type=int, default=15)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--min_count', type=int, default=1)
    p.add_argument('--min_stratum_n', type=int, default=100)
    p.add_argument('--output_dir', default=str(P2_ROOT / 'stratified'))
    p.set_defaults(func=cmd_stratified_calibration)

    p = sub.add_parser('report', help='P2 缺口—程序—测试—产物追溯矩阵')
    p.add_argument('--output_dir', default=str(P2_ROOT / 'traceability'))
    p.set_defaults(func=cmd_report)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == '__main__':
    raise SystemExit(main())
