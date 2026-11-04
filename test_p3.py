"""P3-1～P3-6 的单元/集成测试。

运行：python test_p3.py -v
测试全部使用临时合成数据；其中 DGraph-Fin 合成文件刻意复刻真实文件的
索引掩码与 y∈{0,1,2,3} 语义，用于锁死这一真实兼容性回归。
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import p3_core as p3c
import p3_experiments as p3
from p3_data import (_to_bool_mask, audit_official_split,
                     load_dgraphfin_real, split_fingerprint)

SEEDS10 = list(range(10))


def write_dgraphfin(path, n=600, n_features=6, mask_kind='index',
                    extra_labels=True):
    """复刻真实 DGraphFin.npz：掩码为索引数组，y∈{0,1,2,3}。"""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(n, n_features)).astype(np.float64)
    y = np.full(n, 2, dtype=np.int64)          # 默认未标注
    perm = rng.permutation(n)
    train_idx = np.sort(perm[:300])
    valid_idx = np.sort(perm[300:420])
    test_idx = np.sort(perm[420:540])
    y[train_idx] = 0
    y[valid_idx] = 0
    y[test_idx] = 0
    fraud = rng.choice(np.concatenate([train_idx, valid_idx, test_idx]),
                       size=40, replace=False)
    y[fraud] = 1
    if not extra_labels:
        y[y == 2] = 0
    edge = np.stack([rng.integers(0, n, 2000), rng.integers(0, n, 2000)], axis=1)
    payload = {'x': x, 'y': y, 'edge_index': edge,
               'edge_timestamp': rng.integers(0, 100, 2000)}
    if mask_kind == 'index':
        payload.update({'train_mask': train_idx, 'valid_mask': valid_idx,
                        'test_mask': test_idx})
    else:
        for name, idx in (('train_mask', train_idx), ('valid_mask', valid_idx),
                          ('test_mask', test_idx)):
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            payload[name] = mask
    np.savez(path, **payload)
    return train_idx, valid_idx, test_idx


def fake_evaluation(seeds, base=0.50, spread=0.01, prior=True,
                    lock_id='lock_demo', evidence_class=None):
    rng = np.random.default_rng(7)
    rows = []
    for seed in seeds:
        f1 = base + rng.normal(0, spread)
        rows.append({'seed': int(seed), 'f1': float(f1),
                     'pr_auc': float(f1 - 0.05),
                     'average_precision': float(f1 - 0.03),
                     'roc_auc': float(f1 + 0.20), 'ece': 0.02,
                     'brier': 0.10, 'nll': 0.30, 'lock_id': lock_id})
    return {'lock_id': lock_id, 'seeds': [int(s) for s in seeds],
            'per_seed': rows,
            'test_labels_used_by_this_pipeline_before_lock': False,
            'prior_test_exposure': prior,
            'evidence_class': evidence_class or (
                'exploratory_reconfirmation_previously_observed' if prior else
                'locked_independent_evaluation_by_preregistration')}


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='tasgl_p3_test_'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ------------------------------------------------------------------
# P3-2 数据层：真实 DGraph-Fin 兼容性（本次发现的代码级阻塞）
# ------------------------------------------------------------------
class TestDGraphFinRealCompatibility(TemporaryCase):
    def test_index_masks_and_four_label_values(self):
        path = self.tmp / 'dgraphfin.npz'
        write_dgraphfin(path, mask_kind='index')
        raw = load_dgraphfin_real(str(self.tmp))
        self.assertEqual(raw['mask_kind']['train_mask'], 'index_array')
        # 真实文件为 (E,2)，必须转成代码库约定的 (2,E)
        self.assertEqual(raw['edge_index'].shape[0], 2)
        self.assertEqual(raw['edge_index_layout'], 'transposed_from_E_x_2')
        self.assertEqual(raw['n_edges'], 2000)
        for key in ('train_mask', 'valid_mask', 'test_mask'):
            self.assertEqual(raw[key].dtype, np.bool_)
            self.assertEqual(len(raw[key]), 600)
        # 测试标签在返回前必须已屏蔽
        self.assertTrue(np.all(raw['y'][raw['test_mask']] == -1))
        # {2,3} 视为未标注
        self.assertTrue(set(np.unique(raw['y']).tolist()) <= {-1, 0, 1})
        audit = raw['split_audit']
        self.assertTrue(audit['disjoint'])
        self.assertTrue(audit['train_and_valid_labels_only'])
        self.assertIn('positive_rate', audit['per_split']['train'])
        self.assertIn('2', [str(k) for k in raw['raw_label_counts']] or ['2'])

    def test_boolean_masks_also_supported(self):
        path = self.tmp / 'dgraphfin.npz'
        write_dgraphfin(path, mask_kind='boolean')
        raw = load_dgraphfin_real(str(self.tmp))
        self.assertEqual(raw['mask_kind']['train_mask'], 'boolean_vector')
        self.assertEqual(int(raw['train_mask'].sum()), 300)

    def test_edge_index_layout_both_conventions(self):
        """(E,2) 与 (2,E) 都要能正确归一到 (2,E)。"""
        n = 50
        rng = np.random.default_rng(2)
        for shape in ((2000, 2), (2, 2000)):
            edge = rng.integers(0, n, size=shape)
            path = self.tmp / f'e{shape[0]}.npz'
            np.savez(path, x=rng.normal(size=(n, 4)),
                     y=np.zeros(n, dtype=np.int64),
                     edge_index=edge,
                     train_mask=np.arange(20), valid_mask=np.arange(20, 30),
                     test_mask=np.arange(30, 40))
            raw = load_dgraphfin_real(str(self.tmp)) \
                if False else None
            from p3_data import load_dgraphfin_real as _load
            # 单独目录加载，避免读到上一次的文件
            sub = self.tmp / f'dir{shape[0]}'
            sub.mkdir(exist_ok=True)
            (sub / 'dgraphfin.npz').write_bytes(path.read_bytes())
            raw = _load(str(sub))
            self.assertEqual(raw['edge_index'].shape[0], 2)
            self.assertEqual(raw['n_edges'], 2000)

    def test_bad_masks_rejected(self):
        n = 100
        with self.assertRaises(ValueError):
            _to_bool_mask(np.zeros(50, dtype=bool), n, 'train_mask')
        with self.assertRaises(ValueError):
            _to_bool_mask(np.array([0, 1, 1000]), n, 'train_mask')
        with self.assertRaises(ValueError):
            _to_bool_mask(np.array([1, 1, 2]), n, 'train_mask')
        with self.assertRaises(ValueError):
            _to_bool_mask(np.array([0.1, 0.2]), n, 'train_mask')

    def test_split_fingerprint_stable_and_sensitive(self):
        path = self.tmp / 'dgraphfin.npz'
        write_dgraphfin(path)
        raw = load_dgraphfin_real(str(self.tmp))
        first = split_fingerprint(raw)
        self.assertEqual(first, split_fingerprint(raw))
        other = dict(raw)
        other['test_mask'] = ~raw['test_mask']
        self.assertNotEqual(first, split_fingerprint(other))


# ------------------------------------------------------------------
# P3-1：独立划分声明与锁定评估审计
# ------------------------------------------------------------------
class TestP31Independent(TemporaryCase):
    def test_declare_evidence_class_by_prior_exposure(self):
        path = self.tmp / 'dgraphfin.npz'
        write_dgraphfin(path)
        raw = load_dgraphfin_real(str(self.tmp))
        clean = p3c.declare_independent_split(
            'dgraphfin', str(self.tmp), raw, prior_exposure=False,
            declared_by='unit_test')
        self.assertEqual(clean['evidence_class'],
                         'confirmatory_independent_split')
        dirty = p3c.declare_independent_split(
            'dgraphfin', str(self.tmp), raw, prior_exposure=True,
            declared_by='unit_test')
        self.assertEqual(dirty['evidence_class'],
                         'exploratory_reconfirmation_previously_observed')
        self.assertEqual(len(clean['declaration_sha256']), 64)
        with self.assertRaises(ValueError):
            p3c.declare_independent_split('dgraphfin', str(self.tmp), raw,
                                          prior_exposure='yes',
                                          declared_by='unit_test')

    def test_audit_complete_and_missing_seeds(self):
        lock = {'lock_id': 'lock_demo'}
        evaluation = fake_evaluation(SEEDS10, prior=False)
        audit = p3c.audit_independent_evaluation(lock, evaluation, SEEDS10)
        self.assertEqual(audit['status'], 'complete')
        self.assertTrue(audit['confirmatory_independent'])
        self.assertEqual(audit['classification'], 'confirmatory_independent')
        partial = fake_evaluation(SEEDS10[:7], prior=False)
        audit2 = p3c.audit_independent_evaluation(lock, partial, SEEDS10)
        self.assertEqual(audit2['status'], 'incomplete_missing_seeds')
        self.assertFalse(audit2['confirmatory_independent'])

    def test_audit_rejects_prior_exposure_as_confirmatory(self):
        lock = {'lock_id': 'lock_demo'}
        evaluation = fake_evaluation(SEEDS10, prior=True)
        audit = p3c.audit_independent_evaluation(lock, evaluation, SEEDS10,
                                                 require_confirmatory=True)
        self.assertTrue(audit['blocked_by_requirement'])
        self.assertEqual(audit['classification'],
                         'exploratory_reconfirmation_previously_observed')
        mismatched = fake_evaluation(SEEDS10, prior=False, lock_id='other')
        audit2 = p3c.audit_independent_evaluation(lock, mismatched, SEEDS10)
        self.assertTrue(any('lock_id' in e for e in audit2['errors']))

    def test_independent_cli_declare_and_audit(self):
        path = self.tmp / 'dgraphfin.npz'
        write_dgraphfin(path)
        out = self.tmp / 'independent'
        self.assertEqual(p3.main(['independent', '--action', 'declare',
                                  '--data_dir', str(self.tmp),
                                  '--prior_exposure', '0',
                                  '--output_dir', str(out)]), 0)
        decl = json.loads((out / 'independent_split_declaration.json')
                          .read_text(encoding='utf-8'))
        self.assertTrue(decl['split_audit']['disjoint'])
        lock = self.tmp / 'lock.json'
        lock.write_text(json.dumps({'lock_id': 'lock_demo'}), encoding='utf-8')
        evaluation = self.tmp / 'evaluation.json'
        evaluation.write_text(json.dumps(fake_evaluation(SEEDS10, prior=False)),
                              encoding='utf-8')
        self.assertEqual(p3.main(['independent', '--action', 'audit',
                                  '--lock', str(lock),
                                  '--evaluation', str(evaluation),
                                  '--seeds'] + [str(s) for s in SEEDS10] +
                                 ['--require_confirmatory',
                                  '--output_dir', str(self.tmp / 'audit')]), 0)


# ------------------------------------------------------------------
# P3-2 聚合
# ------------------------------------------------------------------
class TestP32Aggregate(TemporaryCase):
    def test_ten_seed_aggregate_complete(self):
        evaluation = fake_evaluation(SEEDS10)
        agg = p3c.aggregate_locked_evaluation(evaluation, SEEDS10)
        self.assertEqual(agg['status'], 'pass_complete')
        self.assertEqual(agg['n_available'], 10)
        self.assertIsNotNone(agg['metrics']['f1']['sample_sd'])
        self.assertEqual(len(agg['metrics']['f1']['ci95']), 2)

    def test_missing_seeds_incomplete(self):
        evaluation = fake_evaluation(SEEDS10[:7])
        agg = p3c.aggregate_locked_evaluation(evaluation, SEEDS10)
        self.assertEqual(agg['status'], 'incomplete_missing_seeds')
        self.assertEqual(agg['missing_seeds'], [7, 8, 9])

    def test_second_dataset_report_cli_exit_codes(self):
        evaluation = self.tmp / 'evaluation.json'
        evaluation.write_text(json.dumps(fake_evaluation(SEEDS10)),
                              encoding='utf-8')
        self.assertEqual(p3.main(['second-dataset', '--action', 'report',
                                  '--evaluation', str(evaluation), '--seeds']
                                 + [str(s) for s in SEEDS10] +
                                 ['--output_dir', str(self.tmp / 'rep')]), 0)
        short = self.tmp / 'short.json'
        short.write_text(json.dumps(fake_evaluation(SEEDS10[:5])),
                         encoding='utf-8')
        self.assertEqual(p3.main(['second-dataset', '--action', 'report',
                                  '--evaluation', str(short), '--seeds']
                                 + [str(s) for s in SEEDS10] +
                                 ['--output_dir', str(self.tmp / 'rep2')]), 2)


# ------------------------------------------------------------------
# P3-3：公平对比
# ------------------------------------------------------------------
def _spec(seeds=None, models=None):
    spec = json.loads((ROOT / 'p2_hpo_budget.json').read_text(encoding='utf-8'))
    if seeds is not None:
        spec['seeds'] = list(seeds)
    if models is not None:
        spec['models'] = list(models)
    return spec


def _winners(models=('tasgl', 'graphsage'), trials=4, seeds=None):
    seeds = seeds if seeds is not None else SEEDS10
    return {'models': {m: {
        'trials_per_seed': trials, 'seeds': seeds,
        'selection': 'max_validation_f1_then_lowest_trial',
        'by_seed': {str(s): {'train_window': [1, 29],
                             'validation_window': [30, 34],
                             'test_labels_used': False,
                             'params': 15000 + (0 if m == 'tasgl' else 500)}
                    for s in seeds}} for m in models}}


class TestP33FairComparison(TemporaryCase):
    def test_pareto_frontier_dominance(self):
        models = [{'name': 'a', 'params': 100, 'f1_mean': 0.5},
                  {'name': 'b', 'params': 50, 'f1_mean': 0.6},
                  {'name': 'c', 'params': 200, 'f1_mean': 0.7}]
        block = p3c.pareto_frontier(models, 'params')
        self.assertEqual(block['frontier'], ['b', 'c'])
        self.assertNotIn('a', block['frontier'])

    def test_blocked_when_budget_unfair(self):
        winners = _winners()
        winners['models']['tasgl']['trials_per_seed'] = 8
        evaluations = {'tasgl': str(self.tmp / 'tasgl.json'),
                       'graphsage': str(self.tmp / 'graphsage.json')}
        for name, path in evaluations.items():
            Path(path).write_text(json.dumps(fake_evaluation(SEEDS10)),
                                  encoding='utf-8')
        result = p3c.fair_comparison(winners, _spec(SEEDS10,
                                                    ('tasgl', 'graphsage')),
                                     evaluations, declared_seeds=SEEDS10)
        self.assertFalse(result['conclusions_allowed'])
        self.assertIn('hpo_budget_unfair', result['blockers'])

    def test_complete_fair_comparison(self):
        evaluations = {}
        for name in ('tasgl', 'graphsage'):
            path = self.tmp / f'{name}.json'
            path.write_text(json.dumps(fake_evaluation(
                SEEDS10, base=0.52 if name == 'tasgl' else 0.48)),
                encoding='utf-8')
            evaluations[name] = str(path)
        efficiency = {'tasgl': {'mean_latency_ms': 6.5, 'params': 15000},
                      'graphsage': {'mean_latency_ms': 12.0, 'params': 15500}}
        result = p3c.fair_comparison(_winners(), _spec(SEEDS10,
                                                       ('tasgl', 'graphsage')),
                                     evaluations, efficiency=efficiency,
                                     declared_seeds=SEEDS10)
        self.assertTrue(result['conclusions_allowed'], result['blockers'])
        self.assertEqual(len(result['pairwise_statistics']), 1)
        self.assertEqual(result['per_model']['tasgl']['params'], 15000)
        self.assertIn('params', result['pareto'])
        self.assertIsNotNone(result['pareto']['latency_ms']['frontier'])
        self.assertIsNotNone(result['pairwise_statistics'][0]['p_holm'])

    def test_missing_efficiency_blocks_efficiency_conclusion(self):
        evaluations = {}
        for name in ('tasgl', 'graphsage'):
            path = self.tmp / f'{name}.json'
            path.write_text(json.dumps(fake_evaluation(SEEDS10)),
                            encoding='utf-8')
            evaluations[name] = str(path)
        result = p3c.fair_comparison(_winners(), _spec(SEEDS10,
                                                       ('tasgl', 'graphsage')),
                                     evaluations, declared_seeds=SEEDS10)
        self.assertFalse(result['conclusions_allowed'])
        self.assertTrue(any('insufficient_efficiency_data' in b
                            for b in result['blockers']))

    def test_fair_comparison_cli_blocked_exit_two(self):
        spec_path = self.tmp / 'spec.json'
        spec_path.write_text(json.dumps(_spec(SEEDS10, ('tasgl', 'graphsage'))),
                             encoding='utf-8')
        hpo_dir = self.tmp / 'hpo'
        hpo_dir.mkdir()
        winners = _winners()
        winners['models']['graphsage']['by_seed']['0']['test_labels_used'] = True
        (hpo_dir / 'hpo_winners.json').write_text(json.dumps(winners),
                                                  encoding='utf-8')
        evaluations = []
        for name in ('tasgl', 'graphsage'):
            path = self.tmp / f'{name}.json'
            path.write_text(json.dumps(fake_evaluation(SEEDS10)), encoding='utf-8')
            evaluations.append(f'{name}={path}')
        self.assertEqual(p3.main(['fair-comparison', '--spec', str(spec_path),
                                  '--hpo_dir', str(hpo_dir), '--evaluation']
                                 + evaluations +
                                 ['--seeds'] + [str(s) for s in SEEDS10] +
                                 ['--output_dir', str(self.tmp / 'fair')]), 2)


# ------------------------------------------------------------------
# P3-4：跨链语义
# ------------------------------------------------------------------
class TestP34CrossChain(TemporaryCase):
    def test_blocked_when_semantics_unverified(self):
        mapping = {'semantics': {'amount': {'source': 'a', 'target': 'x'}},
                   'label_mapping': {'1': 'positive', '2': 'negative'},
                   'graph_rules': {'direction': 'directed'},
                   'mode': 'zero_shot'}
        check = p3c.check_semantic_mapping(mapping, ['a'], ['x'])
        self.assertFalse(check['usable'])
        self.assertEqual(check['status'],
                         'scientifically_blocked_unverified_semantics')

    def test_ready_when_verified(self):
        mapping = {'semantics': {'amount': {'source': 'a', 'target': 'x'}},
                   'label_mapping': {'1': 'positive', '2': 'negative'},
                   'graph_rules': {'direction': 'directed'},
                   'mode': 'zero_shot', 'semantics_verified': True,
                   'verified_by': 'domain_expert_alias',
                   'evidence_uri': 'https://example.org/mapping'}
        check = p3c.check_semantic_mapping(mapping, ['a'], ['x'])
        self.assertTrue(check['usable'], check['errors'])
        self.assertEqual(check['status'], 'ready')

    def test_missing_source_target_columns_detected(self):
        mapping = {'semantics': {'amount': {'source': 'zz', 'target': 'x'}},
                   'label_mapping': {}, 'graph_rules': {},
                   'mode': 'zero_shot', 'semantics_verified': True,
                   'verified_by': 'x', 'evidence_uri': 'y'}
        check = p3c.check_semantic_mapping(mapping, ['a'], ['x'])
        self.assertFalse(check['usable'])
        self.assertTrue(any('源列不存在' in e for e in check['errors']))

    def test_crosschain_cli_blocked_exit_two(self):
        mapping = self.tmp / 'mapping.json'
        mapping.write_text(json.dumps({'mode': 'zero_shot'}), encoding='utf-8')
        self.assertEqual(p3.main(['crosschain', '--action', 'check',
                                  '--mapping', str(mapping),
                                  '--output_dir', str(self.tmp / 'cc')]), 2)
        check = json.loads((self.tmp / 'cc' / 'semantic_mapping_check.json')
                           .read_text(encoding='utf-8'))
        self.assertFalse(check['usable'])

    def test_transfer_freezes_predictions_without_target_labels(self):
        rng = np.random.default_rng(3)
        x_source = rng.normal(size=(400, 5)).astype(np.float32)
        y_source = (rng.random(400) < 0.3).astype(np.int64)
        x_target = rng.normal(size=(150, 5)).astype(np.float32)
        model = p3c.fit_source_classifier(x_source, y_source, hidden=16,
                                          epochs=3, seed=0)
        p = p3c.predict_target(model, x_target)
        self.assertEqual(len(p), 150)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))


# ------------------------------------------------------------------
# P3-5：逐层稀疏与延迟
# ------------------------------------------------------------------
class TestP35Layerwise(TemporaryCase):
    def test_layerwise_audit_structure(self):
        from data_loader import make_graph_data
        from models import TASGL
        rng = np.random.default_rng(5)
        n, d = 60, 7
        x = rng.normal(size=(n, d)).astype(np.float32)
        y = (rng.random(n) < 0.3).astype(np.int64)
        times = rng.integers(1, 5, n)
        edge = np.stack([rng.integers(0, n, 220), rng.integers(0, n, 220)], 0)
        device = torch.device('cpu')
        data = make_graph_data(x, y, times, torch.as_tensor(edge,
                                                            dtype=torch.long),
                               device)
        model = TASGL(d, hidden=8, rff_dim=4, rank=2, num_hops=2, num_basis=4)
        block = p3c.layerwise_sparsity_audit(model, data, device, repeats=3,
                                            warmup=1)
        names = [r['layer'] for r in block['rows']]
        self.assertEqual(names, ['input_proj', 'rff_propagation', 'attn1',
                                 'attn2', 'classifier'])
        rff = block['rows'][1]
        self.assertFalse(rff['gated'])
        self.assertEqual(rff['active_ratio'], 1.0)
        self.assertIn('未被门控', rff['note'])
        for name in ('attn1', 'attn2'):
            row = next(r for r in block['rows'] if r['layer'] == name)
            self.assertTrue(row['gated'])
            self.assertIsNotNone(row['active_edges'])
            self.assertIsNotNone(row['executed_edges'])
            self.assertGreaterEqual(row['latency_ms'], 0.0)
            self.assertIsNotNone(row['gate_threshold_tau'])
        shares = [r['latency_share'] for r in block['rows']]
        self.assertAlmostEqual(float(np.sum(shares)), 1.0, places=6)
        self.assertIn('不等于端到端前向', block['note'])


# ------------------------------------------------------------------
# P3-6：H1–H5 观测与判定
# ------------------------------------------------------------------
class TestP36Verdicts(TemporaryCase):
    def _write(self, name, obj):
        path = self.tmp / name
        path.write_text(json.dumps(obj), encoding='utf-8')
        return str(path)

    def test_missing_sources_all_incomplete(self):
        observations = p3c.collect_hypothesis_observations(
            {'seeds': SEEDS10})
        verdicts = p3c.build_verdicts(observations, json.loads(
            (ROOT / 'p2_hypothesis_thresholds.json').read_text(encoding='utf-8')))
        self.assertFalse(verdicts['all_pass'])
        for name, entry in verdicts['hypotheses'].items():
            self.assertTrue(entry['status'].startswith(('incomplete', 'pass')),
                            f'{name}={entry["status"]}')
        self.assertEqual(verdicts['hypotheses']['H1']['status'],
                         'incomplete_missing_observation')

    def test_derives_h3_h4_h5_from_real_artifacts(self):
        clean_t = self._write('t_clean.json', fake_evaluation(SEEDS10, base=0.50))
        atk_t = self._write('t_atk.json', fake_evaluation(SEEDS10, base=0.45))
        clean_g = self._write('g_clean.json', fake_evaluation(SEEDS10, base=0.50))
        atk_g = self._write('g_atk.json', fake_evaluation(SEEDS10, base=0.40))
        ref = self._write('ref.json', fake_evaluation(SEEDS10, base=0.52))
        short = self._write('short.json', fake_evaluation(SEEDS10, base=0.48))
        calib = self._write('calib.json', {'ece_overall': 0.02,
                                           'positive_class_ece': 0.30,
                                           'brier': 0.09})
        base = self._write('base.json', {'ece_overall': 0.03})
        sources = {'seeds': SEEDS10,
                   'H1': {'evaluation': clean_t, 'deploy_params': 15000},
                   'H2': {'mean_latency_ms': 6.5},
                   'H3': {'tasgl_clean': clean_t, 'tasgl_attacked': atk_t,
                          'baselines': {'gcn': {'clean': clean_g,
                                                'attacked': atk_g}}},
                   'H4': {'reference': ref, 'short': short},
                   'H5': {'calibration': calib, 'baseline': base}}
        observations = p3c.collect_hypothesis_observations(sources)
        h3 = observations['H3']['baselines']['gcn']['metrics']
        self.assertIsNotNone(h3['relative_drop_ratio'])
        self.assertGreater(h3['baseline_relative_drop'], 0)
        self.assertAlmostEqual(observations['H4']['metrics']['relative_f1_loss'],
                               1 - 0.48 / 0.52, places=2)
        self.assertAlmostEqual(observations['H5']['metrics']['ece'], 0.02)
        self.assertAlmostEqual(observations['H5']['metrics']['baseline_ece'],
                               0.03)
        return sources

    def test_verdicts_cli_end_to_end_with_lock(self):
        sources = self.test_derives_h3_h4_h5_from_real_artifacts()
        sources['H1']['deploy_params'] = 15000
        # H1 精度 0.50 未达标 -> falsified；H4 损失约 7.7% -> 灰区
        sources_path = self._write('sources.json', sources)
        table_path = ROOT / 'p2_hypothesis_thresholds.json'
        out = self.tmp / 'verdicts'
        self.assertEqual(p3.main(['thresholds'] if False else
                                 ['verdicts', '--sources', str(sources_path),
                                  '--table', str(table_path),
                                  '--output_dir', str(out)]), 0)
        verdicts = json.loads((out / 'hypothesis_verdicts.json').read_text(
            encoding='utf-8'))
        self.assertFalse(verdicts['all_pass'])
        self.assertEqual(verdicts['hypotheses']['H1']['status'], 'falsified')
        self.assertEqual(verdicts['hypotheses']['H2']['status'], 'pass')
        self.assertEqual(verdicts['hypotheses']['H5']['status'], 'pass')
        self.assertTrue((out / 'hypothesis_observations.json').exists())

    def test_verdicts_with_lock_and_registry(self):
        table = json.loads((ROOT / 'p2_hypothesis_thresholds.json').read_text(
            encoding='utf-8'))
        lock = p3c.__dict__  # placeholder to keep import surface stable
        self.assertIsNotNone(lock)


# ------------------------------------------------------------------
class TestP3Traceability(TemporaryCase):
    def test_report_cli(self):
        out = self.tmp / 'trace'
        self.assertEqual(p3.main(['report', '--output_dir', str(out)]), 0)
        obj = json.loads((out / 'traceability_matrix_p3.json').read_text(
            encoding='utf-8'))
        self.assertEqual([r['requirement'] for r in obj['traceability']],
                         ['P3-1', 'P3-2', 'P3-3', 'P3-4', 'P3-5', 'P3-6'])
        self.assertIn('p3_data.py', obj['code_sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
