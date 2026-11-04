"""P1-1～P1-8 与锁后扩展指标/案例的合成回归。

运行：python test_p1.py -v
不读取正式 results/data，不生成论文性能结论，不伪造外部专家标注。
"""
from __future__ import annotations

import copy
import csv
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, auc)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import p0_experiments as p0
import p1_experiments as p1
from models import MLPBaseline
from models_v3 import TASGLite
from p0_core import file_sha256, read_json
from p1_attacks import apply_nonoracle_attack, validate_attack_matrix
from p1_core import (build_case_audit, compare_environment, environment_snapshot,
                     extended_binary_metrics, h3_drop_decision,
                     merge_case_annotations, summarize_cost_events,
                     write_case_audit)
from p1_release import FIXED_WINDOWS_INTERPRETER, stage_release, verify_release
from p1_system import benchmark_inference
from test_p0 import base_lock_config, write_elliptic


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='tasgl_p1_test_'))

    def tearDown(self):
        for path in self.tmp.rglob('*'):
            if path.is_file():
                try:
                    path.chmod(path.stat().st_mode | stat.S_IWUSR)
                except OSError:
                    pass
        shutil.rmtree(self.tmp, ignore_errors=True)


def attack_spec(name='noise', capability='feature_noise', selection='all_evaluation'):
    if capability == 'feature_noise':
        budget_type, budget = 'feature_std', 0.25
    else:
        budget_type, budget = 'edge_fraction', 0.50
    return {'name': name, 'capability': capability, 'selection': selection,
            'knowledge': 'gray_box' if selection == 'confidence_targeted' else 'black_box',
            'goal': 'untargeted_test', 'budget_type': budget_type,
            'budget': budget, 'target_scope': 'evaluation_incident',
            'oracle_labels': False, 'max_time_gap': 3}


def matrix_config(attacks):
    return {'protocol': 'p1_nonoracle_attack_matrix_v1', 'attack_seeds': [0, 1],
            'feature_noise_scope': 'evaluation_nodes_only',
            'nonpositive_baseline_rule': 'inconclusive', 'attacks': attacks}


def make_hpo_fixture(root, times=(1, 29, 30, 31, 32, 33, 34, 35, 36)):
    data_dir = root / 'elliptic'
    write_elliptic(data_dir, times=times, per_time=4, n_features=5)
    hpo_dir = root / 'hpo'; hpo_dir.mkdir()
    cfg = {'hidden': 4, 'dropout': 0.0, 'hops': 1}
    data, _, _ = p0._load_hpo_graph(str(data_dir), torch.device('cpu'))
    model = p0._build_hpo_model('mlp', data['in_dim'], cfg, data)
    state_path = hpo_dir / 'mlp_seed0.pt'
    torch.save(model.state_dict(), state_path)
    hashes = {name: file_sha256(data_dir / name) for name in
              ('elliptic_txs_features.csv', 'elliptic_txs_classes.csv',
               'elliptic_txs_edgelist.csv')}
    winner = {'protocol': 'p0_equal_budget_hpo_v1', 'run_protocol': 'synthetic_winner',
              'model': 'mlp', 'seed': 0, 'trial': 0, 'config': cfg,
              'state_file': str(state_path), 'complete': True,
              'test_labels_used': False}
    from p0_core import atomic_json
    atomic_json(hpo_dir / 'hpo_winners.json', {
        'protocol': 'p0_equal_budget_hpo_v1', 'data_sha256': hashes,
        'models': {'mlp': {'trials_per_seed': 1, 'seeds': [0],
                           'by_seed': {'0': winner},
                           'selection': 'synthetic_test_fixture'}}})
    return data_dir, hpo_dir


class TestP11CasesAndMetrics(TemporaryCase):
    def test_extended_metrics_match_reference_definitions(self):
        y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.8, 0.3, 0.7])
        out = extended_binary_metrics(y, p, threshold=0.5,
                                      fpr_targets=(0.01, 0.5, 1.0))
        self.assertEqual((out['tn'], out['fp'], out['fn'], out['tp']), (1, 1, 1, 1))
        self.assertAlmostEqual(out['f1'], 0.5)
        self.assertAlmostEqual(out['average_precision'], average_precision_score(y, p))
        self.assertAlmostEqual(out['roc_auc'], roc_auc_score(y, p))
        precision, recall, _ = precision_recall_curve(y, p)
        self.assertAlmostEqual(out['pr_auc'], auc(recall, precision))
        self.assertNotAlmostEqual(out['pr_auc'], out['average_precision'])
        self.assertEqual(set(out['recall_at_fpr']), {'0.01', '0.5', '1.0'})
        with self.assertRaisesRegex(ValueError, '同时包含两类'):
            extended_binary_metrics(np.zeros(3), np.array([0.1, 0.2, 0.3]))

    def test_case_queue_and_real_annotation_boundary(self):
        y = np.array([0, 0, 1, 1]); p = np.array([0.9, 0.1, 0.2, 0.8])
        clean = np.array([0.1, 0.1, 0.8, 0.8])
        rows, summary = build_case_audit(y, p, [10, 11, 12, 13], [33, 33, 34, 34],
                                         clean_p=clean, model='synthetic', seed=0)
        self.assertEqual([r['error_type'] for r in rows], ['FP', 'FN'])
        self.assertEqual([r['failure_transition'] for r in rows],
                         ['correct_to_error', 'correct_to_error'])
        self.assertTrue(all(r['review_status'] == 'pending_external_review' for r in rows))
        out = self.tmp / 'cases'
        write_case_audit(out, rows, summary, {'evaluation_role': 'development'})
        annotations = self.tmp / 'annotations.csv'
        pd.DataFrame([{'case_id': rows[0]['case_id'], 'reviewer_alias': 'reviewer_A',
                       'pattern_code': 'synthetic_structural_pattern',
                       'assessment': 'uncertain', 'evidence_uri': '', 'notes': 'fixture'}]
                     ).to_csv(annotations, index=False)
        merged = merge_case_annotations(out / 'case_audit_queue.csv', annotations,
                                        self.tmp / 'reviewed')
        self.assertEqual(merged['n_annotations'], 1)
        self.assertEqual(merged['status'],
                         'external_annotations_present_requires_scientific_adjudication')
        self.assertFalse(merged['automatic_expert_validation_claimed'])
        bad = self.tmp / 'bad_annotations.csv'
        synthetic_email = 'reviewer' + '@example.com'
        pd.DataFrame([{'case_id': rows[0]['case_id'], 'reviewer_alias': synthetic_email,
                       'pattern_code': 'x', 'assessment': 'uncertain'}]).to_csv(bad, index=False)
        with self.assertRaisesRegex(ValueError, '匿名别名'):
            merge_case_annotations(out / 'case_audit_queue.csv', bad, self.tmp / 'bad')

    def test_metrics_and_case_cli_resume_and_config_guard(self):
        path = self.tmp / 'development_predictions.npz'
        y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.8, 0.3, 0.7])
        np.savez(path, y=y, p_prob=p, node_id=np.arange(4),
                 node_time=np.array([33, 33, 34, 34]), seed=np.array([0]))
        out = self.tmp / 'metrics'
        argv = ['metrics', '--input', str(path), '--threshold', '0.5',
                '--threshold_source', 'synthetic_validation',
                '--output_dir', str(out)]
        self.assertEqual(p1.main(argv), 0)
        self.assertEqual(p1.main(argv), 0)
        result = read_json(out / 'extended_metrics.json')
        self.assertIn('roc_auc', result['metrics'])
        self.assertEqual(p1.main(['metrics', '--input', str(path), '--threshold', '0.7',
                                  '--threshold_source', 'synthetic_validation',
                                  '--output_dir', str(out)]), 1)
        case_out = self.tmp / 'case_cli'
        self.assertEqual(p1.main(['audit-cases', '--input', str(path), '--threshold', '0.5',
                                  '--model', 'synthetic', '--condition', 'attacked',
                                  '--output_dir', str(case_out)]), 0)
        self.assertTrue((case_out / 'case_audit_queue.csv').is_file())


class TestP12P13Attacks(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.n = 10
        chain = np.arange(self.n - 1)
        self.edge = np.concatenate([
            np.stack([chain, chain + 1]), np.stack([chain + 1, chain])], axis=1)
        self.x = np.arange(self.n * 3, dtype=np.float32).reshape(self.n, 3) / 10
        self.time = np.arange(self.n)
        self.eval_mask = np.arange(self.n) >= 6
        self.p = np.linspace(0.05, 0.95, self.n)

    def test_signature_has_no_label_entry_and_attacks_are_reproducible(self):
        signature = inspect.signature(apply_nonoracle_attack)
        self.assertNotIn('y', signature.parameters)
        self.assertFalse(any('label' in name for name in signature.parameters))
        spec = attack_spec('delete', 'edge_delete', 'random')
        x1, e1, m1 = apply_nonoracle_attack(self.x, self.edge, self.time,
                                            self.eval_mask, self.p, spec, 0)
        x2, e2, m2 = apply_nonoracle_attack(self.x, self.edge, self.time,
                                            self.eval_mask, self.p, spec, 0)
        np.testing.assert_array_equal(e1, e2)
        self.assertEqual(m1['edge_sha256'], m2['edge_sha256'])
        self.assertFalse(m1['oracle_labels_used'])
        removed = set(map(tuple, np.stack([np.minimum(self.edge[0], self.edge[1]),
                                          np.maximum(self.edge[0], self.edge[1])]).T)) - \
                  set(map(tuple, np.stack([np.minimum(e1[0], e1[1]),
                                          np.maximum(e1[0], e1[1])]).T))
        self.assertEqual(len(removed), m1['cost']['removed_undirected_pairs'])
        self.assertEqual(e1.shape[1], self.edge.shape[1] - 2 * len(removed))

    def test_injection_is_symmetric_deduplicated_and_time_constrained(self):
        spec = attack_spec('inject', 'edge_inject', 'degree_targeted')
        # 高度目标集中在早期节点；时间窗须与目标集合存在可行交集。
        spec['max_time_gap'] = 9
        _, attacked, manifest = apply_nonoracle_attack(
            self.x, self.edge, self.time, self.eval_mask, self.p, spec, 1)
        pairs = set(map(tuple, np.stack([np.minimum(attacked[0], attacked[1]),
                                        np.maximum(attacked[0], attacked[1])]).T))
        self.assertEqual(attacked.shape[1], 2 * len(pairs))
        self.assertTrue(all(u != v for u, v in pairs))
        self.assertTrue(all(abs(int(self.time[u]) - int(self.time[v])) <= 9
                            for u, v in pairs))
        self.assertGreater(manifest['cost']['injected_undirected_pairs'], 0)

    def test_injection_fast_path_reproduces_naive_reference(self):
        """注入循环原先用 list 做成员判断, 是 O(n^2)。

        改成 set 必须只是性能修复: RNG 抽取序列与接受/拒绝判定逐字不变,
        所以选出的边对序列也要逐位不变。这里用同一份 rng 重跑朴素版本。
        """
        import p1_attacks as pa
        spec = attack_spec('inject', 'edge_inject', 'degree_targeted')
        spec['max_time_gap'] = 9
        _, attacked, manifest = pa.apply_nonoracle_attack(
            self.x, self.edge, self.time, self.eval_mask, self.p, spec, 1)
        canonical = np.stack([np.minimum(attacked[0], attacked[1]),
                              np.maximum(attacked[0], attacked[1])])
        _, existing = pa._edge_pairs(np.asarray(self.edge, dtype=np.int64))
        seen, injected_order = set(), []
        for pair in map(tuple, canonical.T.tolist()):
            if pair in existing or pair in seen:
                continue
            seen.add(pair); injected_order.append(pair)
        # 朴素参考实现: 与优化前的循环逐行等价。
        edge = np.asarray(self.edge, dtype=np.int64)
        n = len(self.p)
        targets = pa._evaluation_targets(self.p, self.eval_mask,
                                         spec['selection'],
                                         spec.get('target_quantile', 0.5))
        degree = np.bincount(edge.reshape(-1), minlength=n)
        ranked = np.argsort(-degree, kind='stable')
        hubs = ranked[:max(2, min(n, int(0.05 * n) + 1))]
        requested = max(2, 2 * int(round(edge.shape[1] * float(spec['budget']) / 2)))
        rng = pa._stable_rng(spec['name'], 1)
        new_pairs, attempts = [], 0
        max_attempts = max(1000, requested * 100)
        while len(new_pairs) < requested // 2 and attempts < max_attempts:
            attempts += 1
            u = int(targets[rng.randint(len(targets))])
            v = int(hubs[rng.randint(len(hubs))])
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            if pair in existing or pair in new_pairs:
                continue
            if abs(int(self.time[u]) - int(self.time[v])) \
                    > int(spec['max_time_gap']):
                continue
            new_pairs.append(pair)
        self.assertEqual(injected_order, new_pairs)
        self.assertEqual(len(new_pairs),
                         manifest['cost']['injected_undirected_pairs'])

    def test_edge_pair_cache_does_not_bleed_across_graphs(self):
        import p1_attacks as pa
        pa._EDGE_PAIR_CACHE.clear()
        a_pairs, _ = pa._edge_pairs(self.edge)
        shifted = np.stack([self.edge[0] + 1000, self.edge[1] + 1000])
        b_pairs, _ = pa._edge_pairs(shifted)
        a_set = set(map(tuple, a_pairs.T.tolist()))
        b_set = set(map(tuple, b_pairs.T.tolist()))
        self.assertFalse(a_set & b_set)
        self.assertEqual(len(a_set), len(b_set))

    def test_feature_noise_never_changes_non_evaluation_nodes(self):
        spec = attack_spec('feature_confidence', 'feature_noise', 'confidence_targeted')
        changed, edge, manifest = apply_nonoracle_attack(
            self.x, self.edge, self.time, self.eval_mask, self.p, spec, 0)
        np.testing.assert_array_equal(changed[~self.eval_mask], self.x[~self.eval_mask])
        self.assertTrue(np.any(changed[self.eval_mask] != self.x[self.eval_mask]))
        np.testing.assert_array_equal(edge, self.edge)
        self.assertEqual(manifest['cost']['feature_changed_elements'],
                         manifest['cost']['noisy_evaluation_nodes'] * self.x.shape[1])

    def test_matrix_rejects_oracle_or_missing_nonpositive_rule(self):
        config = matrix_config([attack_spec()])
        self.assertTrue(validate_attack_matrix(config))
        bad = copy.deepcopy(config); bad['attacks'][0]['oracle_labels'] = True
        with self.assertRaisesRegex(ValueError, '禁止攻击构造使用真实标签'):
            validate_attack_matrix(bad)
        bad = copy.deepcopy(config); bad['nonpositive_baseline_rule'] = 'pass'
        with self.assertRaisesRegex(ValueError, 'inconclusive'):
            validate_attack_matrix(bad)
        self.assertEqual(h3_drop_decision(0.05, 0.10), 'pass')
        self.assertEqual(h3_drop_decision(0.08, 0.10), 'inconclusive_between_bounds')
        self.assertEqual(h3_drop_decision(0.12, 0.10), 'falsified')
        self.assertEqual(h3_drop_decision(-1.0, 0.0),
                         'inconclusive_nonpositive_baseline_drop')

    def test_attack_cli_stays_in_development_window_and_resumes(self):
        data_dir, hpo_dir = make_hpo_fixture(self.tmp)
        config = matrix_config([attack_spec('synthetic_noise')])
        config_path = self.tmp / 'attacks.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        out = self.tmp / 'attack_output'
        argv = ['attacks', '--data_dir', str(data_dir), '--hpo_dir', str(hpo_dir),
                '--models', 'mlp', '--seeds', '0', '--attack_seeds', '0',
                '--attack_config', str(config_path), '--device', 'cpu',
                '--output_dir', str(out)]
        self.assertEqual(p1.main(argv), 0)
        summary = read_json(out / 'attack_summary.json')
        self.assertEqual(summary['n_runs'], 2)
        self.assertTrue(summary['development_only'])
        self.assertFalse(summary['test_labels_35_49_used'])
        self.assertFalse(summary['attack_construction_label_access'])
        self.assertEqual(summary['status'], 'incomplete_declared_seed_subset')
        self.assertEqual(summary['graph_visibility'],
                         'transductive_full_graph_not_strict_rolling')
        self.assertEqual(summary['evaluation_window'], [33, 34])
        # 模拟指标行已存在、预测尚未落盘且总汇总尚未完成的技术中断。
        attacked_prediction = next(p for p in (out / 'predictions').glob('*.npz')
                                   if 'clean' not in p.name)
        attacked_prediction.unlink(); (out / 'attack_summary.json').unlink()
        self.assertEqual(p1.main(argv), 0)
        self.assertTrue(attacked_prediction.is_file())
        self.assertEqual(p1.main(argv), 0)
        self.assertEqual(len(list((out / 'runs').glob('*.json'))), 2)


class TestP14P15System(TemporaryCase):
    def test_cost_accounting_separates_initial_update_and_teacher(self):
        events = [
            {'phase': 'data_loading', 'seconds': 1.0},
            {'phase': 'preprocessing', 'seconds': 2.0},
            {'phase': 'graph_construction', 'seconds': 3.0},
            {'phase': 'teacher_training', 'seconds': 10.0},
            {'phase': 'student_distillation', 'seconds': 4.0},
            {'phase': 'calibration', 'seconds': 0.5},
            {'phase': 'threshold_selection', 'seconds': 0.25},
            {'phase': 'student_inference', 'seconds': 0.01},
            {'phase': 'model_update', 'seconds': 2.0},
            {'phase': 'update_calibration', 'seconds': 0.2},
            {'phase': 'update_threshold', 'seconds': 0.1},
        ]
        out = summarize_cost_events(events, student_params=100,
                                    teacher_params=5000, calibrator_params=2)
        self.assertTrue(out['complete'])
        self.assertAlmostEqual(out['initial_offline_system_seconds'], 20.75)
        self.assertAlmostEqual(out['update_system_seconds'], 2.3)
        self.assertEqual(out['deployment_parameters'], {'student': 100, 'calibrator': 2})
        self.assertEqual(out['offline_teacher_parameters'], 5000)
        self.assertTrue(out['teacher_excluded_from_student_parameter_count'])
        incomplete = summarize_cost_events(events[:2])
        self.assertFalse(incomplete['complete'])
        self.assertIn('teacher_training', incomplete['missing_required_phases'])

    def test_teacher_student_cost_cli_executes_all_phases(self):
        data_dir = self.tmp / 'elliptic'
        write_elliptic(data_dir, times=(1, 29, 30, 31, 32, 33, 34, 35),
                       per_time=4, n_features=5)
        out = self.tmp / 'cost'
        argv = ['teacher-student-cost', '--data_dir', str(data_dir), '--seeds', '0',
                '--epochs', '1', '--patience', '1', '--update_epochs', '1',
                '--hidden', '4', '--teacher_hidden', '4', '--hops', '0',
                '--device', 'cpu', '--warmup', '0', '--repeats', '1',
                '--output_dir', str(out)]
        self.assertEqual(p1.main(argv), 0)
        summary = read_json(out / 'cost_summary.json')['per_seed'][0]
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['missing_required_phases'], [])
        for phase in ('teacher_training', 'student_distillation', 'calibration',
                      'threshold_selection', 'student_inference', 'model_update',
                      'update_calibration', 'update_threshold'):
            self.assertIn(phase, summary['phase_seconds'])
        self.assertTrue(summary['deployment_benchmark']['batch_curve_supported'])
        self.assertEqual(p1.main(argv), 0)

    def test_batch_curve_only_for_graphless_models(self):
        n = 5
        data = {'x': torch.randn(n, 3), 'edge_index': torch.tensor([[0, 1], [1, 2]]),
                'time_gaps': torch.ones(2), 'num_nodes': n}
        mlp = MLPBaseline(3, hidden=4, dropout=0.0)
        result = benchmark_inference(mlp, data, torch.device('cpu'),
                                     batch_sizes=(2, 4), warmup=0, repeats=1)
        self.assertTrue(result['batch_curve_supported'])
        self.assertEqual([r['n_batches'] for r in result['rows']], [3, 2])
        self.assertTrue(all(r['batch_semantics'] == 'independent_node_batches'
                            for r in result['rows']))
        graph_model = TASGLite(3, hidden=4, hops=1, dropout=0.0)
        result = benchmark_inference(graph_model, data, torch.device('cpu'),
                                     batch_sizes=(1, 100), warmup=0, repeats=1)
        self.assertFalse(result['batch_curve_supported'])
        self.assertEqual(len(result['rows']), 1)
        self.assertEqual(result['rows'][0]['batch_size'], 'full_graph')
        self.assertEqual(result['rows'][0]['batch_semantics'], 'full_graph_forward')

    def test_efficiency_cli_reports_cold_start_and_real_curve(self):
        data_dir, hpo_dir = make_hpo_fixture(self.tmp)
        out = self.tmp / 'efficiency'
        argv = ['efficiency', '--data_dir', str(data_dir), '--hpo_dir', str(hpo_dir),
                '--model', 'mlp', '--seeds', '0', '--devices', 'cpu',
                '--batch_sizes', '2', '4', '--warmup', '0', '--repeats', '1',
                '--output_dir', str(out)]
        self.assertEqual(p1.main(argv), 0)
        result = read_json(out / 'efficiency_summary.json')
        row = result['rows'][0]
        self.assertTrue(row['benchmark']['batch_curve_supported'])
        self.assertEqual(len(row['benchmark']['rows']), 2)
        self.assertTrue(row['calibration_included_in_inference'])
        self.assertIn('cold_start_total_seconds', row['cold_start'])
        self.assertIn('data_loading', row['load_timings'])
        self.assertFalse(row['test_labels_35_49_used'])


class TestP16P17EnvironmentRelease(TemporaryCase):
    def test_environment_snapshot_and_difference_are_explicit(self):
        current = environment_snapshot()
        self.assertIn('torch', current['packages'])
        self.assertIn('known_nondeterminism_sources', current['determinism'])
        self.assertNotIn('hostname', current)
        reference = copy.deepcopy(current); reference['packages']['torch'] = 'different'
        comparison = compare_environment(reference, current)
        self.assertFalse(comparison['compatible'])
        self.assertTrue(any(x['field'] == 'packages.torch' for x in comparison['mismatches']))

    def test_release_anonymizes_copy_and_preserves_raw_results(self):
        source = self.tmp / 'source'; source.mkdir()
        launcher = source / 'run.bat'
        original = f'"{FIXED_WINDOWS_INTERPRETER}" -c "print(1)"\r\npause\r\n'
        # 已含 CRLF 的夹具必须按字节写入；Windows 文本模式会再次转换 \n。
        launcher.write_bytes(original.encode('utf-8'))
        raw = source / 'result.json'; raw.write_text('{"f1": 0.123}\n', encoding='utf-8')
        out = self.tmp / 'release'
        manifest = stage_release(source, out, ['run.bat'], {'dataset': 'synthetic'},
                                 ['result.json'])
        self.assertEqual(launcher.read_bytes(), original.encode('utf-8'))
        staged_bytes = (out / 'run.bat').read_bytes()
        expected = original.replace(FIXED_WINDOWS_INTERPRETER, 'python.exe').encode('utf-8')
        self.assertEqual(staged_bytes, expected)
        staged = staged_bytes.decode('utf-8')
        self.assertIn('python.exe', staged); self.assertNotIn(FIXED_WINDOWS_INTERPRETER, staged)
        self.assertEqual((out / 'result.json').read_bytes(), raw.read_bytes())
        self.assertFalse((out / 'result.json').stat().st_mode & stat.S_IWUSR)
        raw_item = next(item for item in manifest['files'] if item['raw_result'])
        self.assertEqual(raw_item['original_sha256'], raw_item['staged_sha256'])
        self.assertTrue(verify_release(out)['valid'])
        (out / 'extra.txt').write_text('unlisted', encoding='utf-8')
        self.assertFalse(verify_release(out)['valid'])

    def test_release_preserves_lf_crlf_and_mixed_newlines_exactly(self):
        cases = {
            'lf': f'"{FIXED_WINDOWS_INTERPRETER}" -c "print(1)"\npause\n',
            'crlf': f'"{FIXED_WINDOWS_INTERPRETER}" -c "print(1)"\r\npause\r\n',
            'mixed': f'@echo off\r\n"{FIXED_WINDOWS_INTERPRETER}" -c "print(1)"\npause\r\n',
        }
        for name, original in cases.items():
            with self.subTest(newlines=name):
                source = self.tmp / f'source_{name}'; source.mkdir()
                launcher = source / 'run.bat'
                launcher.write_bytes(original.encode('utf-8'))
                out = self.tmp / f'release_{name}'
                stage_release(source, out, ['run.bat'], {'dataset': 'synthetic'})
                expected = original.replace(
                    FIXED_WINDOWS_INTERPRETER, 'python.exe').encode('utf-8')
                self.assertEqual(launcher.read_bytes(), original.encode('utf-8'))
                self.assertEqual((out / 'run.bat').read_bytes(), expected)
                self.assertTrue(verify_release(out)['valid'])

    def test_release_blocks_sensitive_paths_and_network_pickle(self):
        source = self.tmp / 'source_sensitive'; source.mkdir()
        sensitive_path = 'C:' + r'\Users\alice\private\secret.txt'
        (source / 'secret.txt').write_text(sensitive_path, encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '匿名审计阻止发布'):
            stage_release(source, self.tmp / 'blocked', ['secret.txt'], {})
        (source / 'network.pkl').write_bytes(b'\x80\x04not-loaded')
        with self.assertRaisesRegex(ValueError, '网络 pickle 不进入发布包'):
            stage_release(source, self.tmp / 'pickle_blocked', [], {},
                          ['network.pkl'])
        (source / 'raw.npz').write_bytes(b'NPZ-synthetic-binary')
        binary_out = self.tmp / 'binary_pending'
        manifest = stage_release(source, binary_out, [], {}, ['raw.npz'])
        self.assertFalse(manifest['anonymous_ready'])
        verification = verify_release(binary_out)
        self.assertTrue(verification['valid'])
        self.assertFalse(verification['anonymous_ready'])


class TestP18LockedFinalIntegration(TemporaryCase):
    def test_extended_metrics_and_cases_require_the_same_p0_lock(self):
        data = self.tmp / 'elliptic_final'
        ids, times = write_elliptic(data, times=(35, 36), per_time=6)
        for seed in (0, 1):
            p = np.linspace(0.03, 0.97, len(ids))
            np.savez(self.tmp / f'pred{seed}.npz', node_id=ids, p_prob=p,
                     seed=np.array([seed]))
        config = base_lock_config(data, self.tmp / 'pred{seed}.npz')
        config['model_name'] = 'synthetic_locked_model'
        config_path = self.tmp / 'config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        out = self.tmp / 'independent'
        self.assertEqual(p0.main(['independent', 'lock', '--config', str(config_path),
                                  '--output_dir', str(out)]), 0)
        self.assertEqual(p0.main(['independent', 'evaluate', '--config', str(config_path),
                                  '--output_dir', str(out)]), 0)
        result = read_json(out / 'evaluation.json')
        for key in ('average_precision', 'roc_auc'):
            self.assertIn(key, result['aggregate'])
        self.assertIn('0.05', result['recall_at_fpr_aggregate'])
        self.assertEqual(set(result['confusion_matrix_sum']), {'tn', 'fp', 'fn', 'tp'})
        self.assertEqual(len(result['case_audit_index']), 2)
        for item in result['case_audit_index']:
            case_path = out / item['path']
            self.assertTrue(case_path.is_file())
            self.assertEqual(item['external_validation_status'], 'pending_external_annotations')
            queue = case_path.parent / 'case_audit_queue.csv'
            self.assertTrue(queue.is_file())
        receipt = read_json(out / 'evaluation_receipt.json')
        self.assertIn('case_audit_sha256', receipt)
        self.assertEqual(p0.main(['independent', 'evaluate', '--config', str(config_path),
                                  '--output_dir', str(out)]), 2)


class TestP1LaunchersAndIsolation(unittest.TestCase):
    def test_launchers_and_result_isolation(self):
        sh = (ROOT / 'run_p1.sh').read_text(encoding='utf-8')
        # Legacy Windows .bat wrappers are local-only (machine-specific
        # interpreter path); they are not part of the public snapshot.
        bat_path = ROOT / 'run_p1.bat'
        if bat_path.exists():
            bat = bat_path.read_text(encoding='utf-8')
            self.assertIn('log_runner.py', bat)
            self.assertIn('pause', bat.lower())
            self.assertIn('TASGL_NO_PAUSE', bat)
            self.assertIn('test_p1.py', bat)
        self.assertIn('log_runner.py', sh)
        self.assertIn('TASGL_NO_PAUSE', sh)
        self.assertIn('read -r -p', sh)
        self.assertIn('test_p1.py', sh)
        self.assertEqual(p1.P1_ROOT, ROOT / 'results' / 'p1')
        source = (ROOT / 'p0_experiments.py').read_text(encoding='utf-8')
        self.assertIn('extended_binary_metrics', source)
        self.assertIn('case_audit_index', source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
