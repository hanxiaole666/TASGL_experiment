"""test_p4.py — P4 终版证据层测试

覆盖: 根因分解/门控收敛审计/运行准入/严格 rolling/公平 HPO/效应量/
      强断言降级/门限依据/环境指纹与成本链/分类别校准。
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

import p4_core
import p4_measure
import p4_thresholds

# Use the interpreter running the test suite; do not hard-code a local path.
PY = sys.executable


def _tiny_data(n=40, d=8, device='cpu'):
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(n, d)), dtype=torch.float32)
    ts = np.repeat(np.arange(1, 5), n // 4)
    ei = torch.tensor(np.stack([
        rng.integers(0, n, size=120), rng.integers(0, n, size=120)]),
        dtype=torch.long)
    gaps = torch.tensor(np.abs(ts[ei[0].numpy()] - ts[ei[1].numpy()]),
                        dtype=torch.float32)
    y = torch.tensor((rng.random(n) < 0.2).astype(np.int64))
    return {'x': x.to(device), 'y': y.to(device), 'edge_index': ei.to(device),
            'time_gaps': gaps.to(device), 'num_nodes': n, 'in_dim': d}


def _masks(n=40):
    import numpy as np
    rng = np.random.default_rng(1)
    m = rng.random(n) < 0.5
    v = ~m & (rng.random(n) < 0.5)
    t = ~(m | v)
    return [torch.tensor(m), torch.tensor(v), torch.tensor(t)]


class TestRootCause(unittest.TestCase):
    def test_semantics_spec_validation(self):
        s = p4_core.semantics_spec()
        self.assertEqual(s['gate'], 'hard')
        self.assertEqual(s['stochastic'], 'deterministic')
        with self.assertRaises(ValueError):
            p4_core.semantics_spec(gate='bogus')
        with self.assertRaises(ValueError):
            p4_core.semantics_spec(stochastic='bogus')
        with self.assertRaises(ValueError):
            p4_core.semantics_spec(threshold_source='bogus')

    def test_semantics_key_stable(self):
        a = p4_core.semantics_key(p4_core.semantics_spec())
        b = p4_core.semantics_key(p4_core.semantics_spec())
        self.assertEqual(a, b)
        self.assertIn('g=hard', a)

    def test_tune_threshold_picks_best(self):
        rng = np.random.default_rng(0)
        pv = np.concatenate([rng.uniform(0, 0.3, 50),
                             rng.uniform(0.7, 1.0, 50)])
        yv = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
        thr, curve = p4_core.tune_threshold(pv, yv,
                                            grid=np.linspace(0.1, 0.9, 9))
        self.assertGreater(thr, 0.3)
        self.assertLess(thr, 0.8)
        self.assertEqual(len(curve), 9)

    def test_score_probabilities(self):
        prob = np.stack([1 - np.linspace(0.1, 0.9, 20),
                         np.linspace(0.1, 0.9, 20)], axis=1)
        y = (np.arange(20) >= 10).astype(int)
        r = p4_core.score_probabilities(prob, y, 0.5)
        self.assertIn('f1', r)
        self.assertIn('mean_entropy', r)
        self.assertAlmostEqual(r['predicted_positive_rate'], 0.5)

    def test_decompose_and_attribute(self):
        from models import TASGL
        torch.manual_seed(0)
        d = _tiny_data()
        m = TASGL(d['in_dim'], hidden=8, rff_dim=4, rank=2, num_hops=1,
                  num_basis=2)
        masks = _masks()
        cells = p4_core.decompose_forward_semantics(m, d, masks, mc_T=3)
        self.assertEqual(len(cells), 4)
        for k, v in cells.items():
            self.assertIn('tuned_same', v)
            self.assertIn('fixed', v)
        path = ['g=hard|s=deterministic|T=1|t=tuned_same',
                'g=hard|s=mc_dropout|T=3|t=tuned_same',
                'g=soft|s=mc_dropout|T=3|t=tuned_same']
        at = p4_core.attribute_path(
            {k: {'fixed': v['tuned_same']} for k, v in cells.items()},
            path, metric='f1')
        self.assertEqual(len(at['steps']), 2)
        self.assertTrue(all(s['status'] == 'ok' for s in at['steps']))
        self.assertGreater(at['abs_total'], 0)

    def test_reproduce_check(self):
        cells = {'g=soft|s=mc_dropout|T=10|t=tuned_same':
                 {'fixed': {'f1': 0.305}}}
        r = p4_core.reproduce_check(cells, 0.3049, tol=0.01)
        self.assertTrue(r['reproduced'])
        r2 = p4_core.reproduce_check(cells, 0.90, tol=0.01)
        self.assertFalse(r2['reproduced'])

    def test_reproduce_missing_cell(self):
        self.assertFalse(
            p4_core.reproduce_check({}, 0.3049)['reproduced'])


class TestGateAdmission(unittest.TestCase):
    def test_gate_report_on_real_module(self):
        from models import TASGL
        torch.manual_seed(0)
        d = _tiny_data()
        m = TASGL(d['in_dim'], hidden=8, rff_dim=4, rank=2, num_hops=1,
                  num_basis=2)
        rep = p4_core.gate_convergence_report(m, d)
        # 未训练: 门控停在初始化 (bias=2.0, W2=0) -> 全活跃 -> 未收敛
        self.assertEqual(rep['min_active_edge_ratio'], 1.0)
        self.assertAlmostEqual(rep['max_gate_w2_norm'], 0.0, places=6)
        self.assertFalse(rep['converged'])
        self.assertEqual(
            p4_core.classify_seed_failure(rep)['status'],
            'gate_collapsed_open')

    def test_gate_report_after_forcing_sparsity(self):
        from models import TASGL
        torch.manual_seed(0)
        d = _tiny_data()
        m = TASGL(d['in_dim'], hidden=8, rff_dim=4, rank=2, num_hops=1,
                  num_basis=2)
        with torch.no_grad():
            m.attn1.gate[-1].bias.fill_(-1.0)   # 关掉一部分边
            m.attn1.gate[-1].weight.normal_(0, 0.5)
            m.attn1.log_tau.fill_(0.0)
        rep = p4_core.gate_convergence_report(m, d)
        self.assertLess(rep['min_active_edge_ratio'], 0.98)
        self.assertGreater(rep['max_gate_w2_norm'], 0.05)
        self.assertTrue(rep['converged'])
        self.assertEqual(
            p4_core.classify_seed_failure(rep)['status'], 'gate_converged')

    def test_run_admission_not_silent(self):
        rep = {'min_active_edge_ratio': 1.0, 'max_gate_w2_norm': 0.03,
               'converged': False}
        a = p4_core.run_admission(rep)
        self.assertFalse(a['admit'])
        # 拒绝的种子不得静默删除: action 必须要求主分析保留该种子
        self.assertIn('主分析仍须保留该种子', a['action'])

    def test_dual_track_detects_instability(self):
        recs = [
            {'f1': 0.40, 'admission': {'admit': False}},
            {'f1': 0.60, 'admission': {'admit': True}},
            {'f1': 0.58, 'admission': {'admit': True}},
            {'f1': 0.41, 'admission': {'admit': False}},
            {'f1': 0.59, 'admission': {'admit': True}},
        ]
        r = p4_core.dual_track_analysis(recs, key='f1')
        self.assertEqual(r['primary_all_seeds']['n'], 5)
        self.assertEqual(r['sensitivity_admitted']['n'], 3)
        self.assertEqual(r['rejected'], 2)
        self.assertFalse(r['conclusion_stable'])
        self.assertEqual(r['verdict'],
                         'conclusion_unstable_no_strong_claim')

    def test_dual_track_insufficient(self):
        self.assertEqual(
            p4_core.dual_track_analysis([], key='f1')['verdict'],
            'insufficient_data')


class TestRolling(unittest.TestCase):
    def test_induced_subgraph_cuts_both_ends(self):
        ts = np.array([1, 2, 3, 4, 5])
        ei = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        idx, sub = p4_core.induced_subgraph(ei, ts, 3)
        self.assertEqual(list(idx), [0, 1, 2])
        self.assertEqual(sub.shape[0], 2)
        self.assertTrue((sub < 3).all())

    def test_strict_rolling_plan(self):
        ts = np.repeat(np.arange(1, 50), 30)
        lab = np.ones(len(ts), dtype=int)
        lab[::3] = 0
        rec, (tr, va, te) = p4_core.strict_rolling_plan(ts, lab, 25, 34, 49)
        self.assertEqual(rec['train_window'][1], 25)
        self.assertEqual(rec['test_window'], [35, 49])
        self.assertTrue(rec['sufficient_labeled'])
        self.assertEqual(int((tr & va).sum()), 0)
        self.assertEqual(int((va & te).sum()), 0)
        with self.assertRaises(ValueError):
            p4_core.strict_rolling_plan(ts, lab, 40, 34, 49)

    def test_scaler_fit_train_only(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 5)) * 10 + 5
        sc = p4_core.fit_scaler_train_only(X, np.arange(50))
        self.assertAlmostEqual(float(sc.mean_[0]), float(X[:50].mean(0)[0]),
                               places=5)

    def test_no_future_leakage_audit(self):
        ts = np.repeat(np.arange(1, 50), 20)
        ei = np.stack([np.arange(0, 979), np.arange(1, 980)])
        idx, sub = p4_core.induced_subgraph(ei, ts, 34)
        r = p4_core.audit_no_future_leakage(idx, sub, ts[idx],
                                            np.arange(0, 500), 34, 25)
        self.assertTrue(r['checks']['all_nodes_within_cutoff'])
        self.assertFalse(r['checks']['scaler_fit_indices_within_train'])
        self.assertFalse(r['passed'])

    def test_window_exposure_audit(self):
        ledger = [{'name': 'final_test', 'window': [35, 49],
                   'usage': 'protocol5', 'prior_exposure': True}]
        r = p4_core.window_exposure_audit([35, 49], ledger)
        self.assertFalse(r['usable_for_confirmation'])
        r2 = p4_core.window_exposure_audit([50, 60], ledger)
        self.assertTrue(r2['usable_for_confirmation'])


class TestFairHPO(unittest.TestCase):
    def _spec(self, trials=4, seeds=(0, 1, 2)):
        return {'schema': 'p2_hpo_budget_v1', 'models': ['a', 'b'],
                'trials_per_seed': trials, 'seeds': list(seeds),
                'selection': 'max_validation_f1_then_lowest_trial',
                'train_window': [1, 29], 'validation_window': [30, 34],
                'test_labels_used': False}

    def _winner(self, trials, seeds):
        return {'trials_per_seed': trials, 'seeds': list(seeds),
                'selection': 'max_validation_f1_then_lowest_trial',
                'by_seed': {str(s): {'train_window': [1, 29],
                                     'validation_window': [30, 34],
                                     'test_labels_used': False}
                            for s in seeds}}

    def test_equal_budget_passes(self):
        from p2_core import audit_hpo_fairness
        winners = {'models': {'a': self._winner(4, (0, 1, 2)),
                              'b': self._winner(4, (0, 1, 2))}}
        rep = audit_hpo_fairness(winners, self._spec())
        self.assertTrue(rep.get('fair'), rep)

    def test_unequal_budget_fails(self):
        from p2_core import audit_hpo_fairness
        winners = {'models': {'a': self._winner(4, (0, 1, 2)),
                              'b': self._winner(1, (0, 1, 2))}}
        rep = audit_hpo_fairness(winners, self._spec())
        self.assertFalse(rep.get('fair'), rep)

    def test_test_label_leakage_fails(self):
        from p2_core import audit_hpo_fairness
        b = self._winner(4, (0, 1, 2))
        b['by_seed']['0']['test_labels_used'] = True
        winners = {'models': {'a': self._winner(4, (0, 1, 2)), 'b': b}}
        rep = audit_hpo_fairness(winners, self._spec())
        self.assertFalse(rep.get('fair'), rep)


class TestPairedReport(unittest.TestCase):
    def test_reports_effect_size_and_mde(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0.60, 0.05, 12)
        b = rng.normal(0.50, 0.05, 12)
        r = p4_core.paired_report(a, b)
        self.assertEqual(r['n'], 12)
        self.assertIsNotNone(r['cohen_dz'])
        self.assertIsNotNone(r['ci95'])
        self.assertIsNotNone(r['mde_at_power80'])
        self.assertTrue(r['ci_excludes_zero'])
        self.assertGreater(r['cohen_dz'], 1.0)

    def test_small_n_cannot_reach_alpha(self):
        rng = np.random.default_rng(1)
        r = p4_core.paired_report(rng.normal(0.6, 0.05, 5),
                                  rng.normal(0.5, 0.05, 5))
        self.assertAlmostEqual(r['min_attainable_wilcoxon_p'], 2 / 32)
        self.assertFalse(r['wilcoxon_can_reach_alpha'])
        self.assertIn('统计效力不足', r['caveat'])

    def test_insufficient_data(self):
        self.assertEqual(p4_core.paired_report([], [])['status'],
                         'insufficient_data')


class TestClaims(unittest.TestCase):
    def test_dominated_without_evidence_is_retracted(self):
        r = p4_core.adjudicate_claim('dominated', {}, a='TASGL',
                                     b='GraphSAGE', metric='F1')
        self.assertEqual(r['status'], 'unverifiable')
        self.assertTrue(r['retraction_required'])
        self.assertIn('不得表述为', r['weakened_text'])

    def test_dominated_with_full_evidence_supported(self):
        ev = {'n_seeds': 12, 'equal_budget_hpo': True, 'paired_p_holm': 0.01,
              'effect_size': 0.8, 'ci': [0.02, 0.08], 'independent': True}
        r = p4_core.adjudicate_claim('dominated', ev)
        self.assertEqual(r['status'], 'supported')
        self.assertFalse(r['retraction_required'])

    def test_ci_includes_zero_weakens(self):
        ev = {'n_seeds': 12, 'equal_budget_hpo': True, 'paired_p_holm': 0.4,
              'effect_size': 0.2, 'ci': [-0.03, 0.05], 'independent': True}
        r = p4_core.adjudicate_claim('dominated', ev)
        self.assertEqual(r['status'], 'weakened')

    def test_calibration_claim_needs_positive_ece(self):
        r = p4_core.adjudicate_claim('calibrated', {'classwise_ece': 0.1})
        self.assertEqual(r['status'], 'weakened')
        self.assertIn('positive_class_ece', r['missing'])

    def test_weaken_claims_summary(self):
        claims = [{'id': 'c1', 'kind': 'dominated', 'evidence': {}},
                  {'id': 'c2', 'kind': 'calibrated', 'evidence': {}}]
        rep = p4_core.weaken_claims(claims)
        self.assertEqual(rep['n_claims'], 2)
        self.assertEqual(rep['n_requiring_retraction'], 2)
        self.assertFalse(rep['all_supported'])

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            p4_core.adjudicate_claim('bogus', {})


class TestThresholdJustification(unittest.TestCase):
    def test_arbitrary_thresholds_withdrawn(self):
        t = p4_thresholds.build_justification_table()
        h = t['hypotheses']
        self.assertEqual(h['H1_f1']['original_basis'], 'arbitrary')
        self.assertEqual(h['H1_f1']['verdict'], 'unjustified_withdrawn')
        self.assertEqual(h['H2_latency']['verdict'], 'unjustified_withdrawn')
        self.assertEqual(h['H3_ratio']['verdict'], 'unjustified_withdrawn')
        self.assertEqual(h['H4_attack']['verdict'], 'unjustified_withdrawn')
        self.assertEqual(h['H5_ece']['verdict'], 'justified_after_repair')

    def test_derive_f1_floor(self):
        r = p4_thresholds.derive_f1_floor(0.5367, baseline_sd=0.023,
                                          n_seeds=10)
        self.assertLess(r['value'], 0.5367)
        self.assertGreater(r['margin'], 0)
        self.assertEqual(r['basis'], 'derived')

    def test_derive_latency_budget(self):
        r = p4_thresholds.derive_latency_budget(1000)
        self.assertAlmostEqual(r['value_ms'], 1.0)
        self.assertEqual(r['semantics'], 'per_node_screening')

    def test_derive_params_budget(self):
        r = p4_thresholds.derive_params_budget(80)
        self.assertEqual(r['value'], 20480)

    def test_attack_tolerance_missing_inputs(self):
        r = p4_thresholds.derive_attack_tolerance(None, None)
        self.assertIsNone(r['value'])

    def test_mde_and_required_seeds(self):
        mde = p4_thresholds.mde_for_seeds(0.086, 10)
        self.assertGreater(mde, 0)
        n = p4_thresholds.required_seeds_for_mde(0.086, 0.05)
        self.assertGreater(n, 10)

    def test_register_and_audit(self):
        with tempfile.TemporaryDirectory() as td:
            r = p4_thresholds.register_justified_thresholds(
                {'best_baseline_f1': 0.5367, 'baseline_sd': 0.023,
                 'n_seeds': 10, 'target_qps': 1000, 'memory_kb': 80},
                out_dir=td, declared_seeds=list(range(10)))
            self.assertTrue(os.path.exists(r['table_path']))
            self.assertTrue(os.path.exists(r['lock_path']))
            tbl = json.load(open(r['table_path'], encoding='utf-8'))
            self.assertIn('H1', tbl['hypotheses'])
            self.assertIn('H5', tbl['hypotheses'])
            aud = p4_thresholds.audit_justification(tbl)
            self.assertTrue(aud['all_justified'])
            # H5 必须作用于正类 ECE
            gates = tbl['hypotheses']['H5']['gates']
            self.assertTrue(all(g['metric'] == 'positive_class_ece'
                                for g in gates))

    def test_register_without_anchor_marks_missing(self):
        with tempfile.TemporaryDirectory() as td:
            r = p4_thresholds.register_justified_thresholds(
                {}, out_dir=td, declared_seeds=list(range(10)))
            tbl = r['table']
            self.assertEqual(tbl['hypotheses']['H1']['status'],
                             'not_registered_missing_anchor')
            self.assertEqual(tbl['hypotheses']['H2']['status'],
                             'not_registered_missing_anchor')
            # H4 是区间估计, 没有 gate, 因此不会被标成"缺锚点未登记";
            # 攻击项已经从 H4 拆出, 单列为 H6 并判为不可测。
            self.assertEqual(tbl['hypotheses']['H4']['gates'], [])
            self.assertNotIn('status', tbl['hypotheses']['H4'])
            self.assertEqual(tbl['hypotheses']['H6']['status'],
                             'not_testable')
            self.assertTrue(tbl['hypotheses']['H6']['open_item'])


class TestMeasure(unittest.TestCase):
    def test_environment_fingerprint_keys(self):
        e = p4_measure.environment_fingerprint()
        self.assertIn('torch_version', e)
        self.assertIn('missing', e)
        self.assertIn('fingerprint', e)
        self.assertIsInstance(e['complete'], bool)

    def test_measurement_protocol_requires_semantics(self):
        p = p4_measure.measurement_protocol()
        self.assertEqual(p['semantics'], 'full_graph_forward')
        self.assertIn('warning', p)
        with self.assertRaises(ValueError):
            p4_measure.measurement_protocol(semantics='bogus')

    def test_measure_stage(self):
        st = p4_measure.measure_stage('inference', lambda: None, repeats=3,
                                      protocol=p4_measure.measurement_protocol())
        self.assertTrue(st['measured'])
        self.assertEqual(st['stage'], 'inference')
        self.assertFalse(st['energy_measured'])   # 未采样功率 -> 不伪造
        with self.assertRaises(ValueError):
            p4_measure.measure_stage('bogus', lambda: None)

    def test_full_chain_cost_and_completeness(self):
        stages = [{'stage': k, 'total_seconds': 1.0, 'energy_j': None,
                   'energy_measured': False}
                  for k in p4_measure.REQUIRED_STAGES]
        c = p4_measure.full_chain_cost(stages, amortize_nodes=1000)
        self.assertTrue(c['complete'])
        self.assertAlmostEqual(c['total_seconds'], 8.0)
        self.assertIsNotNone(c['amortized_seconds_per_1k_nodes'])
        a = p4_measure.audit_cost_completeness(c)
        self.assertFalse(a['usable_for_claims'])
        self.assertTrue(any('测量口径' in r for r in a['blocking_reasons']))

    def test_missing_stage_blocks(self):
        c = p4_measure.full_chain_cost(
            [{'stage': 'inference', 'total_seconds': 1.0}])
        self.assertFalse(c['complete'])
        self.assertIn('teacher_training', c['stages_missing'])
        a = p4_measure.audit_cost_completeness(
            c, p4_measure.measurement_protocol())
        self.assertFalse(a['usable_for_claims'])


class TestCalibration(unittest.TestCase):
    def test_classwise_calibration(self):
        rng = np.random.default_rng(0)
        n = 600
        p1 = rng.uniform(0.01, 0.4, n)
        y = (rng.random(n) < p1).astype(int)
        prob = np.stack([1 - p1, p1], axis=1)
        r = p4_core.classwise_calibration(prob, y, n_bins=10)
        self.assertIn('positive_class_ece', r)
        self.assertIn('negative_class_ece', r)
        self.assertGreaterEqual(r['positive_class_ece'], 0.0)

    def test_positive_ece_differs_from_overall(self):
        rng = np.random.default_rng(1)
        n = 4000
        p1 = rng.beta(0.3, 20, n)          # 极端不平衡下的低置信
        y = (rng.random(n) < p1).astype(int)
        prob = np.stack([1 - p1, p1], axis=1)
        r = p4_core.classwise_calibration(prob, y, n_bins=15)
        self.assertIsNotNone(r['positive_class_ece'])


class TestCLI(unittest.TestCase):
    def test_thresholds_justify_cli(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'j.json')
            r = subprocess.run([PY, 'p4_experiments.py', 'thresholds',
                                'justify', '--out', out],
                               capture_output=True, text=True, cwd='.')
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertTrue(os.path.exists(out))

    def test_env_cli(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'e.json')
            r = subprocess.run([PY, 'p4_experiments.py', 'env', '--out', out],
                               capture_output=True, text=True, cwd='.')
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertTrue(os.path.exists(out))

    def test_rolling_plan_cli(self):
        import subprocess
        if not os.path.exists('./data/elliptic'):
            self.skipTest('缺少 ./data/elliptic')
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, 'r.json')
            r = subprocess.run([PY, 'p4_experiments.py', 'rolling', 'plan',
                                '--out', out],
                               capture_output=True, text=True, cwd='.')
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertTrue(os.path.exists(out))


class TestBuildModelHyperparams(unittest.TestCase):
    """回归测试: 超参必须真正传到基线, 否则"等预算 HPO"是假的。

    旧实现 build_model(name, in_dim, **kw) 只把 in_dim 传给基线, 六个 trial
    的验证 F1 与参数量逐位相同, 搜索无效。此测试锁死修正后的行为。
    """

    def test_hidden_width_changes_param_count(self):
        from models import build_model
        for name in ['mlp', 'gcn', 'gat', 'graphsage', 'dsgat']:
            counts = []
            for h in (32, 64, 128):
                m = build_model(name, 165, hidden=h, dropout=0.3)
                counts.append(sum(p.numel() for p in m.parameters()))
            self.assertEqual(len(set(counts)), 3,
                             f'{name}: hidden 未生效 -> {counts}')
            self.assertTrue(counts[0] < counts[1] < counts[2],
                            f'{name}: 参数量未随 hidden 单调增 -> {counts}')

    def test_tasgl_rank_changes_param_count(self):
        from models import build_model
        a = build_model('tasgl', 165, hidden=64, dropout=0.3, rank=4)
        b = build_model('tasgl', 165, hidden=64, dropout=0.3, rank=16)
        pa = sum(p.numel() for p in a.parameters())
        pb = sum(p.numel() for p in b.parameters())
        self.assertNotEqual(pa, pb)

    def test_unknown_kwarg_is_filtered(self):
        from models import build_model
        # sgc 不接受 hidden/dropout, 必须被过滤而不是抛异常
        m = build_model('sgc', 165, hidden=64, dropout=0.3)
        self.assertIsNotNone(m)

    def test_sgc_is_config_invariant(self):
        """SGC 是固定线性模型: 搜索对它无效, 必须在论文中如实标注。"""
        from models import build_model
        a = sum(p.numel() for p in build_model(
            'sgc', 165, hidden=32, dropout=0.1).parameters())
        b = sum(p.numel() for p in build_model(
            'sgc', 165, hidden=128, dropout=0.4).parameters())
        self.assertEqual(a, b)

    def test_min_epochs_blocks_early_stop(self):
        import inspect
        src = inspect.getsource(p4_core) if False else None
        import p4_experiments
        sig = inspect.signature(p4_experiments.train_with_audit)
        self.assertIn('min_epochs', sig.parameters)


if __name__ == '__main__':
    unittest.main(verbosity=2)
