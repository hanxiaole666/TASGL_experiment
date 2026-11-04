"""P2-1～P2-5 的单元/集成测试。

运行：python test_p2.py -v
测试全部使用临时合成数据，不读取或覆盖正式 results/data。
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import p2_core as p2c
import p2_experiments as p2

SEEDS = list(range(20))


def write_elliptic(root, times=tuple(range(1, 8)), per_time=6, n_features=4):
    """写一份合成 Elliptic；时间步覆盖训练/验证/评估三段。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows, classes = [], []
    for t in times:
        for j in range(per_time):
            i = len(rows)
            tx = 1000 + i
            illicit = int((i % 3) == 0)
            feats = [0.1 * i + 0.02 * k + illicit * (0.3 + 0.05 * k)
                     for k in range(n_features)]
            rows.append([tx, t] + feats)
            classes.append([tx, '1' if illicit else '2'])
    pd.DataFrame(rows).to_csv(root / 'elliptic_txs_features.csv',
                              header=False, index=False)
    pd.DataFrame(classes, columns=['txId', 'class']).to_csv(
        root / 'elliptic_txs_classes.csv', index=False)
    n = len(rows)
    edges = [[1000 + i, 1000 + i + 1] for i in range(n - 1)]
    edges += [[1000 + i, 1000 + min(i + 3, n - 1)] for i in range(0, n - 3, 2)]
    pd.DataFrame(edges, columns=['txId1', 'txId2']).to_csv(
        root / 'elliptic_txs_edgelist.csv', index=False)
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='tasgl_p2_test_'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ------------------------------------------------------------------
# P2-1
# ------------------------------------------------------------------
class TestP21Seeds(TemporaryCase):
    def test_seed_pool_registration_and_hash(self):
        seeds = p2c.validate_seed_pool(SEEDS)
        self.assertEqual(len(seeds), 20)
        h1 = p2c.seed_pool_hash(seeds)
        h2 = p2c.seed_pool_hash(list(reversed(seeds)))
        self.assertNotEqual(h1, h2, '种子顺序改变也应改变指纹')
        with self.assertRaises(ValueError):
            p2c.validate_seed_pool([0, 0, 1])
        with self.assertRaises(ValueError):
            p2c.validate_seed_pool(list(range(5)), declared=SEEDS)
        with self.assertRaisesRegex(ValueError, '至少需要'):
            p2c.validate_seed_pool(list(range(5)))

    def test_wilcoxon_min_p_and_mde_monotone(self):
        self.assertAlmostEqual(p2c.wilcoxon_min_p(5), 0.0625)
        self.assertAlmostEqual(p2c.wilcoxon_min_p(10), 2 / 1024)
        self.assertLess(p2c.wilcoxon_min_p(20), 0.05)
        prev = None
        for n in range(10, 21):
            dz = p2c.required_cohen_dz(n)
            if prev is not None:
                self.assertLess(dz, prev, 'MDE 应随种子数单调下降')
            prev = dz
        mde = p2c.minimum_detectable_effect(20, sd=0.03)
        self.assertAlmostEqual(mde['mde_units'],
                               mde['required_cohen_dz'] * 0.03)
        self.assertEqual(p2c.required_seeds_for_mde(0.02, 0.03),
                         None if p2c.minimum_detectable_effect(
                             200, 0.03)['mde_units'] > 0.02 else
                         p2c.required_seeds_for_mde(0.02, 0.03))

    def test_paired_statistics_multi_missing_seeds(self):
        rng = np.random.default_rng(0)
        a = [{'seed': s, 'f1': 0.50 + rng.normal(0, 0.01)} for s in SEEDS]
        b = [{'seed': s, 'f1': 0.48 + rng.normal(0, 0.01)} for s in SEEDS]
        full = p2c.paired_statistics_multi(a, b, SEEDS, 'f1')
        self.assertEqual(full['status'], 'pass_complete')
        self.assertEqual(full['n_available'], 20)
        self.assertIsNotNone(full['bootstrap']['ci95_mean_difference'])
        self.assertIsNotNone(full['sign_test'])
        self.assertFalse(full['nested_units_collapsed_to_seed'])
        partial = p2c.paired_statistics_multi(a[:12], b[:12], SEEDS, 'f1')
        self.assertEqual(partial['status'], 'incomplete_missing_seeds')

    def test_nested_units_collapsed_to_seed(self):
        """窗口是嵌套重复，展开会虚增 n；必须折叠到种子级。"""
        rows_a, rows_b = [], []
        for s in SEEDS:
            for w in (35, 40, 45):
                rows_a.append({'seed': s, 'window': w, 'f1': 0.50 + 0.001 * w})
                rows_b.append({'seed': s, 'window': w, 'f1': 0.49 + 0.001 * w})
        res = p2c.paired_statistics_multi(rows_a, rows_b, SEEDS, 'f1')
        self.assertEqual(res['n_available'], 20)
        self.assertEqual(res['nested_units_per_seed'], 3)
        self.assertTrue(res['nested_units_collapsed_to_seed'])

    def test_bootstrap_reproducible_and_holm(self):
        diff = np.array([0.01] * 20)
        ci1 = p2c.bootstrap_mean_ci(diff, 1000, 7)
        ci2 = p2c.bootstrap_mean_ci(diff, 1000, 7)
        self.assertEqual(ci1, ci2)
        adj, flags = p2c.multi_comparison([0.01, 0.02, None], 0.05)
        self.assertIsNone(adj[2])
        self.assertFalse(flags[2])
        self.assertGreaterEqual(adj[1], 0.02)

    def test_seeds_cli_register_and_check(self):
        out = self.tmp / 'seeds'
        registry = self.tmp / 'seed_registry.json'
        self.assertEqual(p2.main(['seeds', '--action', 'register',
                                  '--registry', str(registry),
                                  '--output_dir', str(out)]), 0)
        self.assertTrue(registry.exists())
        self.assertEqual(p2.main(['seeds', '--action', 'check',
                                  '--registry', str(registry),
                                  '--output_dir', str(out)]), 0)
        with self.assertRaises(RuntimeError):
            p2.main(['seeds', '--action', 'check', '--seeds', '0', '1', '2',
                     '3', '4', '5', '6', '7', '8', '9', '10',
                     '--registry', str(registry), '--output_dir', str(out)])

    def test_seeds_power_table_cli(self):
        out = self.tmp / 'power'
        self.assertEqual(p2.main(['seeds', '--action', 'power', '--sd', '0.03',
                                  '--target_mde', '0.02',
                                  '--output_dir', str(out)]), 0)
        obj = json.loads((out / 'seed_power.json').read_text(encoding='utf-8'))
        self.assertEqual(len(obj['sensitivity_table']), 11)
        self.assertIn('required_seeds_for_target', obj)


# ------------------------------------------------------------------
# P2-2
# ------------------------------------------------------------------
def threshold_table():
    return json.loads((ROOT / 'p2_hypothesis_thresholds.json').read_text(
        encoding='utf-8'))


class TestP22Thresholds(TemporaryCase):
    def test_threshold_register_and_tamper_detection(self):
        table = threshold_table()
        lock = p2c.register_thresholds(table, self.tmp)
        self.assertEqual(len(lock['lock_id']), 24)
        self.assertEqual(p2c.verify_thresholds(table, lock), [])
        tampered = json.loads(json.dumps(table))
        tampered['hypotheses']['H1']['gates'][0]['pass_value'] = 0.40
        errors = p2c.verify_thresholds(tampered, lock)
        self.assertTrue(any('已被修改' in e for e in errors))
        with self.assertRaises(ValueError):
            p2c.validate_threshold_table({'schema': 'wrong', 'hypotheses': {}})

    def test_apply_h1_pass_falsified_incomplete(self):
        table = threshold_table()
        lock = p2c.register_thresholds(table, self.tmp)
        obs = {'H1': {'metrics': {'mean_f1': 0.80, 'deploy_params': 15000},
                      'seeds_present': SEEDS}}
        res = p2c.apply_threshold_table(table, obs, lock)
        self.assertEqual(res['hypotheses']['H1']['status'], 'pass')
        obs_low = {'H1': {'metrics': {'mean_f1': 0.51, 'deploy_params': 10849},
                          'seeds_present': SEEDS}}
        res_low = p2c.apply_threshold_table(table, obs_low, lock)
        self.assertEqual(res_low['hypotheses']['H1']['status'], 'falsified')
        obs_missing = {'H1': {'metrics': {'mean_f1': 0.80},
                             'seeds_present': SEEDS[:5]}}
        res_miss = p2c.apply_threshold_table(table, obs_missing, lock)
        self.assertEqual(res_miss['hypotheses']['H1']['status'],
                         'incomplete_missing_seeds')
        self.assertFalse(res_miss['all_pass'])

    def test_h4_grey_zone_is_inconclusive(self):
        table = threshold_table()
        lock = p2c.register_thresholds(table, self.tmp)
        for value, expected in ((0.04, 'pass'), (0.06, 'inconclusive_grey_zone_5_8'),
                                (0.12, 'falsified')):
            obs = {'H4': {'metrics': {'relative_f1_loss': value},
                          'seeds_present': SEEDS}}
            res = p2c.apply_threshold_table(table, obs, lock)
            self.assertEqual(res['hypotheses']['H4']['status'], expected,
                             f'relative_f1_loss={value}')

    def test_h3_nonpositive_baseline_inconclusive(self):
        table = threshold_table()
        lock = p2c.register_thresholds(table, self.tmp)
        obs = {'H3': {'seeds_present': SEEDS, 'baselines': {
            'gcn': {'metrics': {'baseline_relative_drop': 0.0,
                                'relative_drop_ratio': 0.1}},
            'gat': {'metrics': {'baseline_relative_drop': 0.10,
                                'relative_drop_ratio': 0.5}}}}}
        res = p2c.apply_threshold_table(table, obs, lock)
        self.assertEqual(res['hypotheses']['H3']['per_baseline']['gcn']['status'],
                         'inconclusive_nonpositive_baseline_drop')
        self.assertEqual(res['hypotheses']['H3']['per_baseline']['gat']['status'],
                         'pass')
        self.assertEqual(res['hypotheses']['H3']['status'],
                         'inconclusive_nonpositive_baseline_drop')
        obs_all_ok = {'H3': {'seeds_present': SEEDS, 'baselines': {
            b: {'metrics': {'baseline_relative_drop': 0.10,
                            'relative_drop_ratio': 0.5}}
            for b in ('gcn', 'gat')}}}
        self.assertEqual(p2c.apply_threshold_table(
            table, obs_all_ok, lock)['hypotheses']['H3']['status'], 'pass')

    def test_h5_above_ceiling_but_better_is_inconclusive(self):
        table = threshold_table()
        lock = p2c.register_thresholds(table, self.tmp)
        better_but_high = {'H5': {'metrics': {'ece': 0.07, 'baseline_ece': 0.09,
                                              'positive_class_ece': 0.21},
                                  'seeds_present': SEEDS}}
        res = p2c.apply_threshold_table(table, better_but_high, lock)
        self.assertEqual(res['hypotheses']['H5']['status'],
                         'inconclusive_above_absolute_ceiling')
        self.assertAlmostEqual(res['hypotheses']['H5']['reported']
                               ['positive_class_ece'], 0.21)
        good = {'H5': {'metrics': {'ece': 0.02, 'baseline_ece': 0.09},
                       'seeds_present': SEEDS}}
        self.assertEqual(p2c.apply_threshold_table(
            table, good, lock)['hypotheses']['H5']['status'], 'pass')
        worse = {'H5': {'metrics': {'ece': 0.12, 'baseline_ece': 0.09},
                        'seeds_present': SEEDS}}
        self.assertEqual(p2c.apply_threshold_table(
            table, worse, lock)['hypotheses']['H5']['status'], 'falsified')

    def test_thresholds_cli_end_to_end_and_all_pass_false(self):
        table = threshold_table()
        table_path = self.tmp / 'table.json'
        table_path.write_text(json.dumps(table, ensure_ascii=False),
                              encoding='utf-8')
        out = self.tmp / 'thresholds'
        self.assertEqual(p2.main(['thresholds', '--action', 'register',
                                  '--table', str(table_path),
                                  '--output_dir', str(out)]), 0)
        lock_path = out / 'thresholds.lock.json'
        self.assertTrue(lock_path.exists())
        self.assertEqual(p2.main(['thresholds', '--action', 'verify',
                                  '--table', str(table_path),
                                  '--lock', str(lock_path),
                                  '--output_dir', str(out)]), 0)
        obs = {h: {'metrics': {'mean_f1': 0.51, 'deploy_params': 10849,
                               'mean_latency_ms': 6.5, 'relative_f1_loss': 0.12,
                               'ece': 0.02, 'baseline_ece': 0.03},
                   'seeds_present': SEEDS} for h in ('H1', 'H2', 'H4', 'H5')}
        obs['H3'] = {'seeds_present': SEEDS, 'baselines': {
            b: {'metrics': {'baseline_relative_drop': 0.1,
                            'relative_drop_ratio': 0.5}}
            for b in ('gcn', 'gat')}}
        obs_path = self.tmp / 'obs.json'
        obs_path.write_text(json.dumps(obs), encoding='utf-8')
        self.assertEqual(p2.main(['thresholds', '--action', 'apply',
                                  '--table', str(table_path),
                                  '--lock', str(lock_path),
                                  '--observations', str(obs_path),
                                  '--output_dir', str(out)]), 0)
        verdicts = json.loads((out / 'hypothesis_verdicts.json').read_text(
            encoding='utf-8'))
        self.assertFalse(verdicts['all_pass'])
        self.assertEqual(verdicts['statuses']['H1'], 'falsified')
        self.assertEqual(verdicts['statuses']['H4'], 'falsified')
        self.assertEqual(verdicts['statuses']['H5'], 'pass')
        # 改阈值后必须被拒绝
        bad = json.loads(json.dumps(table))
        bad['hypotheses']['H1']['gates'][0]['pass_value'] = 0.30
        bad_path = self.tmp / 'bad_table.json'
        bad_path.write_text(json.dumps(bad), encoding='utf-8')
        with self.assertRaises(RuntimeError):
            p2.main(['thresholds', '--action', 'apply', '--table', str(bad_path),
                     '--lock', str(lock_path), '--observations', str(obs_path),
                     '--output_dir', str(self.tmp / 'out2')])


# ------------------------------------------------------------------
# P2-3
# ------------------------------------------------------------------
def window_ledger():
    return json.loads((ROOT / 'p2_window_ledger.json').read_text(
        encoding='utf-8'))


class TestP23Windows(TemporaryCase):
    def test_declare_independent_window_rejects_exposed_range(self):
        ledger = window_ledger()
        res = p2c.declare_independent_window(ledger, 43, 49)
        self.assertFalse(res['independent_window_available'])
        self.assertTrue(res['conflicts'])
        self.assertEqual(res['evidence_class'],
                         'exploratory_reconfirmation_previously_observed')
        clean = p2c.declare_independent_window(
            {'windows': {'only': {'range': [1, 10], 'role': 'train'}}}, 20, 30)
        self.assertTrue(clean['independent_window_available'])
        self.assertEqual(clean['evidence_class'],
                         'confirmatory_independent_window')

    def test_strict_rolling_audit_detects_future_leakage(self):
        from p0_data import load_elliptic_raw
        data_dir = self.tmp / 'elliptic'
        write_elliptic(data_dir)
        raw = load_elliptic_raw(str(data_dir), label_ceiling=5)
        audit = p2c.audit_strict_rolling(raw, cutoff=6, train_end=3,
                                         label_ceiling=5)
        self.assertTrue(audit['contamination_free'])
        self.assertEqual(audit['max_visible_time'], 6)
        self.assertGreater(audit['future_nodes_excluded'], 0)
        self.assertGreater(audit['edges_touching_future_excluded'], 0)
        # 训练截止点晚于 cutoff -> 直接拒绝
        with self.assertRaises(ValueError):
            p2c.audit_strict_rolling(raw, cutoff=3, train_end=6)
        # 记录值与重算不一致 -> 判为污染
        bad = {'time': 6, 'visible_nodes': audit['visible_nodes'] + 5,
               'visible_edges': audit['visible_edges']}
        audit2 = p2c.audit_strict_rolling(raw, cutoff=6, train_end=3,
                                          label_ceiling=5, recorded=bad)
        self.assertFalse(audit2['contamination_free'])
        self.assertIn('visible_nodes', audit2['recorded_mismatch'])

    def test_window_disjointness_and_ledger_classification(self):
        ledger = window_ledger()
        cls = {k: p2c.classify_window(v)
               for k, v in ledger['windows'].items()}
        self.assertEqual(cls['final_test'], 'contaminated_prior_exposure')
        self.assertEqual(cls['second_dataset_dgraphfin_test'],
                         'held_out_independent')
        disjoint = p2c.audit_window_disjointness(ledger['windows'])
        self.assertTrue(disjoint['disjoint'])
        overlap = {'train': {'range': [1, 30], 'role': 'train'},
                   'eval': {'range': [30, 40], 'role': 'evaluation'}}
        self.assertFalse(p2c.audit_window_disjointness(overlap)['disjoint'])

    def test_scan_window_usage_detects_reserved_window(self):
        cfg_dir = self.tmp / 'runs'
        cfg_dir.mkdir()
        (cfg_dir / 'ok.json').write_text(json.dumps(
            {'eval_start': 35, 'eval_end': 40}), encoding='utf-8')
        (cfg_dir / 'bad.json').write_text(json.dumps(
            {'eval_start': 43, 'eval_end': 49}), encoding='utf-8')
        res = p2c.scan_window_usage([str(cfg_dir)], (43, 49))
        self.assertFalse(res['clean'])
        self.assertEqual(len(res['hits']), 2)
        self.assertTrue(all('bad.json' in h['source'] for h in res['hits']))
        clean = p2c.scan_window_usage([str(cfg_dir)], (50, 60))
        self.assertTrue(clean['clean'])

    def test_prediction_contamination_check(self):
        from p0_data import load_elliptic_raw
        data_dir = self.tmp / 'elliptic'
        ids, times = write_elliptic(data_dir, times=tuple(range(1, 8)))
        raw = load_elliptic_raw(str(data_dir))
        reserved_ids = ids[times >= 6]
        clean_path = self.tmp / 'clean.npz'
        np.savez(clean_path, node_id=ids[times <= 5],
                 p_prob=np.full(int(np.sum(times <= 5)), 0.4),
                 seed=np.array([0]))
        dirty_path = self.tmp / 'dirty.npz'
        np.savez(dirty_path, node_id=reserved_ids,
                 p_prob=np.full(len(reserved_ids), 0.4), seed=np.array([0]))
        res = p2c.check_prediction_contamination(
            [str(clean_path), str(dirty_path)], raw, (6, 7))
        self.assertFalse(res['clean'])
        self.assertTrue(res['predictions'][0]['clean'])
        self.assertFalse(res['predictions'][1]['clean'])

    def test_windows_cli_declare_returns_two_and_audit_runs(self):
        out = self.tmp / 'windows'
        ledger_path = ROOT / 'p2_window_ledger.json'
        self.assertEqual(p2.main(['windows', '--action', 'declare',
                                  '--ledger', str(ledger_path),
                                  '--reserve_start', '43', '--reserve_end', '49',
                                  '--output_dir', str(out)]), 2)
        decl = json.loads((out / 'independent_window_declaration.json').read_text(
            encoding='utf-8'))
        self.assertFalse(decl['independent_window_available'])
        data_dir = self.tmp / 'elliptic'
        write_elliptic(data_dir)
        out2 = self.tmp / 'audit'
        self.assertEqual(p2.main(['windows', '--action', 'audit',
                                  '--ledger', str(ledger_path),
                                  '--data_dir', str(data_dir),
                                  '--train_end', '3', '--label_ceiling', '5',
                                  '--cutoffs', '5', '6', '7',
                                  '--output_dir', str(out2)]), 0)
        audit = json.loads((out2 / 'window_audit.json').read_text(encoding='utf-8'))
        self.assertTrue(audit['rolling_contamination_free'])
        self.assertFalse(audit['independent_window_available'])
        self.assertTrue(audit['independent_window_by_dataset']['dgraphfin'])
        self.assertFalse(audit['independent_window_by_dataset']['elliptic'])


# ------------------------------------------------------------------
# P2-4
# ------------------------------------------------------------------
class TestP24HpoFairness(TemporaryCase):
    def _spec(self):
        return json.loads((ROOT / 'p2_hpo_budget.json').read_text(
            encoding='utf-8'))

    def _winners(self, trials=4, seeds=None, selection=None):
        seeds = seeds if seeds is not None else SEEDS
        return {'models': {m: {
            'trials_per_seed': trials, 'seeds': seeds,
            'selection': selection or 'max_validation_f1_then_lowest_trial',
            'by_seed': {str(s): {'train_window': [1, 29],
                                 'validation_window': [30, 34],
                                 'test_labels_used': False}
                        for s in seeds}}
            for m in ('tasgl', 'mlp', 'gcn', 'gat', 'graphsage', 'sgc',
                      'sign', 'temporal_graphsage')}}

    def test_hpo_fairness_equal_budget(self):
        res = p2c.audit_hpo_fairness(self._winners(), self._spec())
        self.assertTrue(res['fair'], res['violations'])
        self.assertEqual(res['per_model']['tasgl']['search_cost_runs'], 80)

    def test_hpo_fairness_detects_extra_budget(self):
        winners = self._winners()
        winners['models']['tasgl']['trials_per_seed'] = 8
        res = p2c.audit_hpo_fairness(winners, self._spec())
        self.assertFalse(res['fair'])
        self.assertTrue(any('试验预算' in v for v in res['violations']))
        winners2 = self._winners(seeds=SEEDS[:10])
        res2 = p2c.audit_hpo_fairness(winners2, self._spec())
        self.assertTrue(any('缺预注册种子' in v for v in res2['violations']))
        winners3 = self._winners()
        del winners3['models']['gcn']
        self.assertTrue(any('缺模型' in v for v in
                            p2c.audit_hpo_fairness(winners3, self._spec())
                            ['violations']))

    def test_hpo_cli_spec_and_audit(self):
        out = self.tmp / 'hpo'
        self.assertEqual(p2.main(['hpo', '--action', 'spec',
                                  '--spec', str(ROOT / 'p2_hpo_budget.json'),
                                  '--output_dir', str(out)]), 0)
        self.assertTrue((out / 'hpo_budget_spec.lock.json').exists())
        hpo_dir = self.tmp / 'p0hpo'
        (hpo_dir / 'runs').mkdir(parents=True)
        hpo_dir.joinpath('hpo_winners.json').write_text(
            json.dumps(self._winners()), encoding='utf-8')
        self.assertEqual(p2.main(['hpo', '--action', 'audit',
                                  '--spec', str(ROOT / 'p2_hpo_budget.json'),
                                  '--hpo_dir', str(hpo_dir),
                                  '--output_dir', str(out)]), 0)
        bad = self._winners()
        bad['models']['gat']['by_seed']['0']['test_labels_used'] = True
        hpo_dir.joinpath('hpo_winners.json').write_text(json.dumps(bad),
                                                        encoding='utf-8')
        self.assertEqual(p2.main(['hpo', '--action', 'audit',
                                  '--spec', str(ROOT / 'p2_hpo_budget.json'),
                                  '--hpo_dir', str(hpo_dir),
                                  '--output_dir', str(self.tmp / 'hpo2')]), 2)


# ------------------------------------------------------------------
# P2-5
# ------------------------------------------------------------------
class TestP25Calibration(TemporaryCase):
    def test_positive_class_ece_values(self):
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        p = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
        # 两个桶，每桶 4 个样本、2 个正类；桶内正类率 0.5 与 mean(p) 相差 0.4。
        res = p2c.positive_class_ece(y, p, bins=15)
        self.assertAlmostEqual(res['positive_class_ece'], 0.4, places=6)
        self.assertEqual(res['n_positive'], 4)
        self.assertFalse(res['degenerate_single_bin'])
        # 字面子集定义退化：1 - mean_p(正类) = 1 - 0.5
        self.assertAlmostEqual(res['positive_class_subset_ece'], 0.5, places=6)
        self.assertTrue(res['positive_class_subset_ece_is_degenerate'])
        self.assertAlmostEqual(res['mean_p_on_positives'], 0.5, places=6)
        neg = p2c.negative_class_ece(y, p, bins=15)
        self.assertAlmostEqual(neg['negative_class_ece'], 0.4, places=6)
        self.assertAlmostEqual(neg['negative_class_subset_ece'], 0.5, places=6)
        # 校准良好的正类：高置信桶内正类率与预测一致
        good_y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        good_p = np.array([0.75] * 4 + [0.25] * 4)
        good = p2c.positive_class_ece(good_y, good_p, bins=15)
        self.assertAlmostEqual(good['positive_class_ece'], 0.25, places=6)
        with self.assertRaises(ValueError):
            p2c.positive_class_ece(np.array([0, 0]), np.array([0.2, 0.3]))

    def test_overall_ece_and_min_count_filter(self):
        y = np.array([0, 0, 0, 0, 1])
        p = np.array([0.1, 0.1, 0.1, 0.1, 0.9])
        parts = p2c._ece_parts(y, p, bins=15, min_count=2)
        self.assertIsNotNone(parts['ece_min_count_filtered'])
        self.assertLess(parts['filtered_coverage'], 1.0)

    def test_stratified_ece_insufficient_stratum(self):
        rng = np.random.default_rng(1)
        n = 400
        y = (rng.random(n) < 0.2).astype(np.int64)
        p = np.clip(rng.random(n) * 0.5 + y * 0.3, 1e-6, 1 - 1e-6)
        time = np.array([35] * 200 + [45] * 190 + [49] * 10)
        res = p2c.stratified_ece(y, p, time, bins=15, min_stratum_n=100)
        self.assertEqual(res['n_strata'], 3)
        self.assertEqual(res['n_strata_insufficient'], 1)
        self.assertIsNotNone(res['stratified_ece_max'])
        only_one_class = p2c.stratified_ece(np.array([1, 1, 1]),
                                            np.array([0.9, 0.8, 0.7]),
                                            np.array([1, 1, 1]),
                                            min_stratum_n=1)
        self.assertTrue(only_one_class['rows'][0]['insufficient'])
        self.assertIsNone(only_one_class['stratified_ece_max'])

    def test_stratified_temperature_fallback_and_monotonicity(self):
        rng = np.random.default_rng(2)
        n_fit, n_eval = 600, 600
        y_fit = (rng.random(n_fit) < 0.3).astype(np.int64)
        logits_fit = rng.normal(0, 2, n_fit) + y_fit * 2.0
        strata_fit = np.array([30] * 400 + [31] * 180 + [32] * 20)
        cal = p2c.fit_stratified_temperature(logits_fit, y_fit, strata_fit,
                                             min_pos=30, min_neg=30)
        self.assertEqual(cal['n_strata_fallback'], 1)
        self.assertTrue(cal['per_stratum']['32']['fallback_to_global'])
        y_eval = (rng.random(n_eval) < 0.3).astype(np.int64)
        logits_eval = rng.normal(0, 2, n_eval) + y_eval * 2.0
        strata_eval = np.array([35] * 300 + [36] * 300)
        raw_p = 1 / (1 + np.exp(-logits_eval))
        out = p2c.apply_stratified_temperature(cal, logits_eval, strata_eval)
        mono = p2c.monotonicity_report(raw_p, out)
        self.assertTrue(mono['rank_preserved'])
        self.assertAlmostEqual(mono['spearman_correlation'], 1.0, places=6)
        # 明显非单调的变换必须被抓到
        broken = out.copy()
        broken[0], broken[1] = broken[1] + 0.1, broken[0] - 0.1
        self.assertFalse(p2c.monotonicity_report(raw_p, broken)['rank_preserved'])

    def test_calibration_quality_report_and_cli(self):
        rng = np.random.default_rng(3)
        n = 500
        y = (rng.random(n) < 0.25).astype(np.int64)
        p = np.clip(rng.random(n) * 0.6 + y * 0.25, 1e-6, 1 - 1e-6)
        report = p2c.calibration_quality_report(
            y, p, threshold=0.5, bins=15, min_count=1,
            strata={'time': np.array([35] * 250 + [45] * 250)},
            min_stratum_n=100)
        self.assertIn('positive_class_ece', report)
        self.assertIn('sce_positive_negative_weighted', report)
        self.assertAlmostEqual(report['sce_positive_negative_weighted'],
                               0.5 * (report['positive_class_ece'] +
                                      report['negative_class_ece']))
        self.assertIsNotNone(report['strata']['time']['stratified_ece_max'])
        path = self.tmp / 'pred.npz'
        np.savez(path, y=y, p_prob=p,
                 time=np.array([35] * 250 + [45] * 250))
        out = self.tmp / 'calib'
        self.assertEqual(p2.main(['calibration', '--input', str(path),
                                  '--strata_fields', 'time',
                                  '--output_dir', str(out)]), 0)
        obj = json.loads((out / 'calibration_quality.json').read_text(
            encoding='utf-8'))
        self.assertAlmostEqual(obj['ece_overall'], report['ece_overall'])

    def test_stratified_calibration_cli_end_to_end(self):
        rng = np.random.default_rng(4)
        fit_path, eval_path = self.tmp / 'fit.npz', self.tmp / 'eval.npz'
        n = 800
        y_fit = (rng.random(n) < 0.3).astype(np.int64)
        logits_fit = rng.normal(0, 2, n) + y_fit * 2.0
        np.savez(fit_path, y=y_fit, logits=logits_fit,
                 time=np.array([30] * 500 + [31] * 300))
        y_eval = (rng.random(n) < 0.3).astype(np.int64)
        logits_eval = rng.normal(0, 2, n) + y_eval * 2.0
        np.savez(eval_path, y=y_eval, logits=logits_eval,
                 time=np.array([35] * 400 + [36] * 400))
        out = self.tmp / 'strat'
        self.assertEqual(p2.main(['stratified-calibration',
                                  '--fit', str(fit_path),
                                  '--eval', str(eval_path),
                                  '--strata_field', 'time',
                                  '--fit_end', '31', '--eval_start', '35',
                                  '--min_pos', '20', '--min_neg', '20',
                                  '--min_stratum_n', '100',
                                  '--output_dir', str(out)]), 0)
        obj = json.loads((out / 'stratified_calibration.json').read_text(
            encoding='utf-8'))
        self.assertIn('global_temperature', obj['comparisons'])
        self.assertIn('stratified_temperature', obj['comparisons'])
        self.assertTrue(obj['comparisons']['stratified_temperature']
                        ['monotonicity_vs_raw']['rank_preserved'])
        # 重叠窗口必须被拒绝
        with self.assertRaises(ValueError):
            p2.main(['stratified-calibration', '--fit', str(fit_path),
                     '--eval', str(eval_path), '--strata_field', 'time',
                     '--fit_end', '36', '--eval_start', '35',
                     '--output_dir', str(self.tmp / 'strat2')])


# ------------------------------------------------------------------
# 追溯矩阵
# ------------------------------------------------------------------
class TestTraceability(TemporaryCase):
    def test_report_cli_writes_matrix(self):
        out = self.tmp / 'trace'
        self.assertEqual(p2.main(['report', '--output_dir', str(out)]), 0)
        obj = json.loads((out / 'traceability_matrix.json').read_text(
            encoding='utf-8'))
        self.assertEqual(len(obj['traceability']), 5)
        self.assertEqual([r['requirement'] for r in obj['traceability']],
                         ['P2-1', 'P2-2', 'P2-3', 'P2-4', 'P2-5'])
        self.assertIn('p2_core.py', obj['code_sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
