"""P0-1～P0-9 与 BitHeteroNet 校验的单元/集成测试。

运行：python test_p0.py -v
测试全部使用临时合成数据，不读取或覆盖正式 results/data。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import download_data
import p0_experiments as p0
from exp_hypotheses import development_gate_status
from models import TASGL
from p0_core import (apply_calibrator, holm_adjust, make_lock,
                     paired_statistics, read_json, semantic_align,
                     strict_rolling_view, verify_lock)
from p0_data import load_elliptic_raw, validate_prediction_npz
from p0_models import FixedNeighborSampler, NeighborGraphSAGE


def write_elliptic(root, times=(1, 2, 3, 4, 5, 6), per_time=4, n_features=5):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows, classes = [], []
    n = len(times) * per_time
    for t in times:
        for j in range(per_time):
            i = len(rows)
            tx = 1000 + i
            illicit = int((i % 3) == 0)
            features = [0.1 * i + 0.01 * k + illicit * (0.2 + 0.03 * k)
                        for k in range(n_features)]
            rows.append([tx, t] + features)
            classes.append([tx, '1' if illicit else '2'])
    pd.DataFrame(rows).to_csv(root / 'elliptic_txs_features.csv',
                              header=False, index=False)
    pd.DataFrame(classes, columns=['txId', 'class']).to_csv(
        root / 'elliptic_txs_classes.csv', index=False)
    edges = [[1000 + i, 1000 + i + 1] for i in range(n - 1)]
    edges += [[1000 + i, 1000 + min(i + 2, n - 1)] for i in range(0, n - 2, 2)]
    pd.DataFrame(edges, columns=['txId1', 'txId2']).to_csv(
        root / 'elliptic_txs_edgelist.csv', index=False)
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def base_lock_config(data_dir, prediction_pattern, seeds=(0, 1)):
    return {
        'dataset': {'name': 'elliptic', 'root': str(data_dir)},
        'predictions': {'path_pattern': str(prediction_pattern),
                        'contains_labels': False},
        'seeds': list(seeds),
        'threshold': {'source': 'fixed', 'value': 0.5},
        'calibration': {'source': 'none'},
        'statistics': {'test': 'wilcoxon_exact_two_sided', 'alpha': 0.05},
        'exclusions': {'seed_replacement_after_test': False},
        'evaluation': {'time_start': 35, 'time_end': 36,
                       'prior_test_exposure': True}
    }


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='tasgl_p0_test_'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestP01IndependentLock(TemporaryCase):
    def test_lock_evaluate_and_one_shot_guard(self):
        data = self.tmp / 'elliptic'
        ids, _ = write_elliptic(data, times=(35, 36), per_time=4)
        for seed in (0, 1):
            p = np.linspace(0.05, 0.95, len(ids))
            np.savez(self.tmp / f'pred{seed}.npz', node_id=ids, p_prob=p,
                     seed=np.array([seed]))
        config = base_lock_config(data, self.tmp / 'pred{seed}.npz')
        config_path = self.tmp / 'config.json'
        config_path.write_text(json.dumps(config), encoding='utf-8')
        out = self.tmp / 'independent'
        self.assertEqual(p0.main(['independent', 'lock', '--config', str(config_path),
                                  '--output_dir', str(out)]), 0)
        self.assertEqual(p0.main(['independent', 'evaluate', '--config', str(config_path),
                                  '--output_dir', str(out)]), 0)
        result = read_json(out / 'evaluation.json')
        self.assertEqual(len(result['per_seed']), 2)
        self.assertFalse(result['test_labels_used_by_this_pipeline_before_lock'])
        self.assertEqual(result['evidence_class'],
                         'exploratory_reconfirmation_previously_observed')
        self.assertEqual(p0.main(['independent', 'evaluate', '--config', str(config_path),
                                  '--output_dir', str(out)]), 2)
        (out / 'evaluation.json').unlink()
        with self.assertRaisesRegex(RuntimeError, '禁止删除结果后再次评估'):
            p0.main(['independent', 'evaluate', '--config', str(config_path),
                     '--output_dir', str(out)])

    def test_lock_id_and_artifact_tamper_rejected(self):
        pred = self.tmp / 'p.npz'
        np.savez(pred, node_id=np.array([1]), p_prob=np.array([0.4]),
                 seed=np.array([0]))
        config = base_lock_config(self.tmp, self.tmp / 'p.npz', seeds=(0,))
        lock = make_lock(config, {'prediction_seed0': pred}, [])
        self.assertEqual(verify_lock(lock), [])
        bad = dict(lock)
        bad['config'] = dict(config)
        bad['config']['seeds'] = [99]
        self.assertTrue(any('lock_id' in e for e in verify_lock(bad)))
        with open(pred, 'ab') as f:
            f.write(b'x')
        self.assertTrue(verify_lock(lock))

    def test_prediction_file_cannot_contain_labels(self):
        path = self.tmp / 'bad.npz'
        np.savez(path, node_id=np.array([1]), p_prob=np.array([0.5]),
                 seed=np.array([0]), y_true=np.array([1]))
        with self.assertRaisesRegex(ValueError, '禁止包含标签'):
            validate_prediction_npz(path)


class TestP02DevelopmentGate(unittest.TestCase):
    def test_failed_accuracy_cannot_be_compensated(self):
        row = {'f1': 0.20, 'params': 100, 'latency_ms': 0.1,
               'edge_drop50_rel_drop': 0.0, 'extrapolation_loss': 0.0,
               'ece': 0.0}
        status = development_gate_status(row)
        self.assertFalse(status['eligible'])
        self.assertFalse(status['development_gates']['H1_f1'])
        self.assertIn('H1_f1', status['failed_gates'])


class TestP03Calibration(TemporaryCase):
    def test_fit_select_eval_and_reliability_outputs(self):
        rng = np.random.RandomState(7)
        paths = {}
        for role in ('fit', 'select', 'eval'):
            y = np.arange(90) % 2
            logits = (2 * y - 1) * 1.5 + rng.normal(0, 0.4, len(y))
            path = self.tmp / f'{role}.npz'
            np.savez(path, logits=logits, y=y)
            paths[role] = path
        out = self.tmp / 'calibration'
        code = p0.main(['calibration', '--fit', str(paths['fit']),
                        '--select', str(paths['select']), '--eval', str(paths['eval']),
                        '--output_dir', str(out)])
        self.assertEqual(code, 0)
        result = read_json(out / 'calibration_results.json')
        self.assertEqual(len(result['candidates']), 5)
        self.assertTrue(result['fit_select_eval_separated'])
        self.assertIn('not a full ST-CGNN', result['boundary'])
        for c in result['candidates']:
            self.assertEqual(len(c['reliability']), 15)
            self.assertEqual(c['eval']['n'], 90)
            self.assertTrue((out / f'reliability_{c["method"]}.csv').is_file())

    def test_backbone_calibration_window_export(self):
        data = self.tmp / 'elliptic'
        write_elliptic(data, times=tuple(range(1, 37)), per_time=2)
        out = self.tmp / 'exports'
        self.assertEqual(p0.main([
            'calibration-export', '--data_dir', str(data), '--output_dir', str(out),
            '--model', 'mlp', '--seeds', '0', '--epochs', '1', '--patience', '1',
            '--hidden', '8', '--device', 'cpu']), 0)
        stem = out / 'mlp' / 'seed0'
        meta = read_json(stem / 'metadata.json')
        self.assertEqual(meta['config']['train'], [1, 27])
        self.assertEqual(meta['config']['early_stopping'], [28, 29])
        self.assertEqual(meta['counts'], {'fit': 4, 'select': 2, 'eval': 4})
        self.assertFalse(meta['test_labels_35_49_used'])
        for role in ('fit', 'select', 'eval'):
            with np.load(stem / f'{role}.npz') as z:
                self.assertEqual(len(z['logits']), meta['counts'][role])


class TestP04EqualBudgetAndModels(TemporaryCase):
    def test_all_hpo_models_forward_and_shared_grid(self):
        x = torch.randn(12, 5)
        edge = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])
        time = torch.arange(12) % 4 + 1
        data = {'time_steps': time}
        for name in ('tasgl', 'mlp', 'gcn', 'gat', 'graphsage', 'sgc', 'sign',
                     'temporal_graphsage'):
            cfg = p0._hpo_trial_config(name, 0, 4)
            other = p0._hpo_trial_config('graphsage', 0, 4)
            for key in ('lr', 'wd', 'hidden', 'hops'):
                self.assertEqual(cfg[key], other[key])
            cfg['hidden'] = 8
            model = p0._build_hpo_model(name, 5, cfg, data)
            logits, _ = model(x, edge, torch.zeros(edge.shape[1]), len(x))
            self.assertEqual(tuple(logits.shape), (12, 2), name)

    def test_hpo_train_winner_and_frozen_prediction_integration(self):
        data = self.tmp / 'elliptic'
        write_elliptic(data, times=tuple(range(1, 37)), per_time=1)
        out = self.tmp / 'hpo'
        code = p0.main(['hpo', '--data_dir', str(data), '--output_dir', str(out),
                        '--models', 'mlp', 'sign', 'temporal_graphsage',
                        '--trials', '1', '--seeds', '0', '--epochs', '1',
                        '--patience', '1', '--device', 'cpu'])
        self.assertEqual(code, 0)
        winners = read_json(out / 'hpo_winners.json')
        self.assertEqual(set(winners['models']),
                         {'mlp', 'sign', 'temporal_graphsage'})
        for name in winners['models']:
            row = winners['models'][name]['by_seed']['0']
            self.assertTrue(Path(row['state_file']).is_file())
            self.assertFalse(row['test_labels_used'])
        self.assertEqual(p0.main([
            'freeze-predictions', '--data_dir', str(data), '--hpo_dir', str(out),
            '--model', 'mlp', '--seeds', '0', '--eval_start', '35',
            '--eval_end', '36', '--device', 'cpu']), 0)
        pred = validate_prediction_npz(out / 'frozen' / 'mlp_seed0.npz')
        self.assertEqual(len(pred['node_id']), 2)
        cal = read_json(out / 'frozen' / 'mlp_seed0.calibration.json')
        self.assertFalse(cal['test_labels_used'])


class TestP05Statistics(unittest.TestCase):
    def test_exact_wilcoxon_power_sd_and_incomplete(self):
        b = [0.40, 0.41, 0.42, 0.43, 0.44]
        a = [v + 0.1 for v in b]
        rows_a = [{'seed': i, 'f1': v} for i, v in enumerate(a)]
        rows_b = [{'seed': i, 'f1': v} for i, v in enumerate(b)]
        result = paired_statistics(rows_a, rows_b, list(range(5)))
        self.assertEqual(result['status'], 'pass_complete')
        self.assertAlmostEqual(result['p_value'], 0.0625)
        self.assertAlmostEqual(result['sample_sd_a'], np.std(a, ddof=1))
        self.assertIsNone(result['cohen_dz'])  # 恒定差值的样本 SD=0
        ten = paired_statistics(
            [{'seed': i, 'f1': 0.6 + i * 0.001} for i in range(10)],
            [{'seed': i, 'f1': 0.5 + i * 0.001} for i in range(10)],
            list(range(10)))
        self.assertLess(ten['p_value'], 0.05)
        missing = paired_statistics(rows_a[:-1], rows_b, list(range(5)))
        self.assertEqual(missing['status'], 'incomplete_missing_seeds')
        self.assertEqual(missing['missing_a'], [4])
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class TestP06SecondDataset(TemporaryCase):
    def test_minibatch_graphsage_freezes_unlabeled_predictions(self):
        rng = np.random.RandomState(3)
        n = 120
        x = rng.randn(n, 6).astype(np.float32)
        y = np.arange(n) % 2
        train = np.arange(n) < 60
        valid = (np.arange(n) >= 60) & (np.arange(n) < 90)
        test = np.arange(n) >= 90
        edge = np.array([np.arange(n - 1), np.arange(1, n)])
        edge = np.concatenate([edge, edge[::-1]], axis=1)
        np.savez(self.tmp / 'dgraphfin.npz', x=x, y=y, edge_index=edge,
                 train_mask=train, valid_mask=valid, test_mask=test,
                 edge_timestamp=np.arange(edge.shape[1]) % 7)
        out = self.tmp / 'second'
        code = p0.main(['second-dataset', '--data_dir', str(self.tmp),
                        '--output_dir', str(out), '--model', 'graphsage',
                        '--seeds', '0', '--epochs', '2', '--patience', '2',
                        '--hidden', '8', '--batch_size', '16', '--fanout', '4',
                        '--device', 'cpu'])
        self.assertEqual(code, 0)
        stem = out / 'dgraphfin' / 'graphsage' / 'seed0'
        result = read_json(stem / 'run.json')
        self.assertFalse(result['test_metrics_computed'])
        pred = validate_prediction_npz(stem / 'test_predictions.npz')
        self.assertEqual(len(pred['node_id']), 30)
        self.assertNotIn('y_true', np.load(stem / 'test_predictions.npz').files)


class TestP07StrictRolling(TemporaryCase):
    def test_future_isolation_and_full_rolling_smoke(self):
        x = np.arange(30, dtype=float).reshape(6, 5)
        time = np.array([1, 1, 2, 2, 3, 3])
        edge = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        fit = time <= 2
        view = strict_rolling_view(x, edge, time, 2, fit)
        self.assertEqual(len(view['old_indices']), 4)
        self.assertTrue(np.all(view['edge_index'] < 4))
        changed = x.copy(); changed[time > 2] += 1e6
        view2 = strict_rolling_view(changed, edge, time, 2, fit)
        np.testing.assert_allclose(view['x'], view2['x'])
        np.testing.assert_allclose(view['mean'], x[fit].mean(axis=0))

        data = self.tmp / 'elliptic'
        write_elliptic(data)
        raw = load_elliptic_raw(data, label_ceiling=4)
        self.assertTrue(np.all(raw['y'][raw['node_time'] > 4] == -1))
        self.assertEqual(raw['y'][0], 1)  # class=1 必须为 illicit 正类
        out = self.tmp / 'rolling'
        code = p0.main(['rolling', '--data_dir', str(data), '--output_dir', str(out),
                        '--model', 'graphsage', '--seeds', '0', '--epochs', '2',
                        '--hidden', '8', '--train_end', '2', '--val_start', '3',
                        '--val_end', '4', '--eval_start', '5', '--eval_end', '6',
                        '--device', 'cpu'])
        self.assertEqual(code, 0)
        stem = out / 'graphsage' / 'seed0'
        result = read_json(stem / 'run.json')
        self.assertFalse(result['test_metrics_computed'])
        self.assertEqual([w['time'] for w in result['window_audit']], [3, 4, 5, 6])
        nodes = [w['visible_nodes'] for w in result['window_audit']]
        self.assertEqual(nodes, sorted(nodes))
        pred = validate_prediction_npz(stem / 'test_predictions.npz')
        self.assertEqual(len(pred['node_id']), 8)


class TestP08SemanticAlignment(TemporaryCase):
    def _mapping(self, mode):
        rules = {'node_unit': 'transaction', 'edge_semantics': 'payment',
                 'directed': True, 'self_loops': False,
                 'deduplication': 'sorted_unique_source_target'}
        return {'feature_semantics': {
                    'amount': {'source': 'amount', 'target': 'amt'},
                    'hour': {'source': 'hour', 'target': 'hour'}},
                'source_label_column': 'label', 'source_label_map':
                    {'0': 'normal', '1': 'positive'},
                'target_label_column': 'flag', 'target_label_map':
                    {'0': 'normal', '1': 'positive'},
                'target_id_column': 'node_id',
                'graph_rules': {'source': rules, 'target': rules},
                'target_adaptation': {'mode': mode, 'n_per_class': 1,
                                      'selection_seed': 17,
                                      'train_pool_column': 'split',
                                      'train_pool_values': ['train']}}

    def test_zero_and_few_shot_separation(self):
        source = self.tmp / 'source.csv'; target = self.tmp / 'target.csv'
        pd.DataFrame({'amount': [1, 2, 3, 4], 'hour': [1, 2, 3, 4],
                      'label': [0, 1, 0, 1], 'node_id': range(4),
                      'split': ['train'] * 4}).to_csv(source, index=False)
        pd.DataFrame({'amt': [2, 4, 6, 8], 'hour': [2, 3, 4, 5],
                      'flag': [0, 1, 0, 1], 'node_id': range(10, 14),
                      'split': ['train', 'train', 'test', 'test']}).to_csv(target, index=False)
        for mode in ('zero_shot', 'few_shot'):
            mapping = self.tmp / f'{mode}.json'
            mapping.write_text(json.dumps(self._mapping(mode)), encoding='utf-8')
            out = self.tmp / mode
            self.assertEqual(p0.main(['crosschain', '--source', str(source),
                                      '--target', str(target), '--mapping', str(mapping),
                                      '--output_dir', str(out)]), 0)
            manifest = read_json(out / 'alignment_manifest.json')
            self.assertFalse(manifest['pca_used'])
            if mode == 'zero_shot':
                self.assertFalse(manifest['adaptation']['target_labels_used_for_training'])
            else:
                selected = manifest['adaptation']['selected_target_indices']
                self.assertEqual(len(selected), 2)
                self.assertTrue(all(i < 2 for i in selected))
        xs, xt, _ = semantic_align(source and pd.read_csv(source).to_numpy(),
                                  list(pd.read_csv(source).columns),
                                  pd.read_csv(target).to_numpy(),
                                  list(pd.read_csv(target).columns),
                                  self._mapping('zero_shot')['feature_semantics'])
        self.assertEqual(xs.shape[1], xt.shape[1])


class TestP09RealSparsity(TemporaryCase):
    def test_legacy_and_compact_execution_and_dense_equivalence(self):
        torch.manual_seed(0)
        model = TASGL(5, hidden=8, rff_dim=4, rank=2, num_hops=2,
                      execute_hard_sparse=False)
        model.eval()
        x = torch.randn(6, 5)
        time = np.array([1, 1, 2, 2, 3, 3])
        edge_np = np.array([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 5, 0]])
        edge = torch.tensor(edge_np)
        dt = torch.abs(torch.tensor(time)[edge[0]] - torch.tensor(time)[edge[1]]).float()
        for module in (model.prop, model.attn1, model.attn2):
            module.audit_enabled = True
        with torch.no_grad():
            legacy, _ = model(x, edge, dt, len(x), hard_mask=True)
            legacy_audit = dict(model.attn1.last_gate_audit)
            model.attn1.execute_hard_sparse = True
            model.attn2.execute_hard_sparse = True
            compact, _ = model(x, edge, dt, len(x), hard_mask=True)
            compact_audit = dict(model.attn1.last_gate_audit)
        self.assertEqual(legacy_audit['executed_edges'], edge.shape[1])
        self.assertLessEqual(compact_audit['executed_edges'], edge.shape[1])
        self.assertTrue(compact_audit['compact_execution'])
        self.assertLess(float((legacy - compact).abs().max()), 1e-5)
        self.assertEqual(model.prop.last_propagation_audit['executed_edges_per_hop'],
                         [edge.shape[1], edge.shape[1]])
        view = strict_rolling_view(x.numpy(), edge_np, time, 3, time <= 2)
        micro = p0._dense_rff_microbench(model, view, 6, 1, torch.device('cpu'))
        self.assertTrue(micro['equivalent_within_1e-4'], micro)


class TestLaunchersAndIsolation(unittest.TestCase):
    def test_windows_shell_logging_pause_and_result_isolation(self):
        sh = (ROOT / 'run_p0.sh').read_text(encoding='utf-8')
        # The legacy Windows .bat wrappers embed a machine-specific
        # interpreter path and are intentionally excluded from the public
        # open-source snapshot. Validate them only when they are present,
        # so the shipped test suite stays green on a clean checkout.
        bat_path = ROOT / 'run_p0.bat'
        bhn_path = ROOT / 'validate_bitheteronet.bat'
        if bat_path.exists() and bhn_path.exists():
            bat = bat_path.read_text(encoding='utf-8')
            bhn = bhn_path.read_text(encoding='utf-8')
            for text in (bat, bhn):
                self.assertIn('log_runner.py', text)
                self.assertIn('set "CODE=%ERRORLEVEL%"', text)
                self.assertIn('pause', text)
                self.assertIn('TASGL_NO_PAUSE', text)
                self.assertIn('exit /b %CODE%', text)
            self.assertIn('test_p0.py -v', bat)
            self.assertIn('p0_experiments.py %*', bat)
            self.assertIn('results\\p0', bat)
        self.assertIn('bash', sh.splitlines()[0])
        self.assertIn('log_runner.py', sh)
        self.assertIn('TASGL_NO_PAUSE', sh)
        self.assertIn('exit "$CODE"', sh)
        self.assertIn('results/p0', sh)
        self.assertEqual(p0.P0_ROOT, ROOT / 'results' / 'p0')


class TestBitHeteroNetValidation(TemporaryCase):
    def test_shape_truncation_html_and_no_pickle_execution(self):
        old = download_data.BHN_FILES
        self.assertEqual(old['address_attrs.npy']['shape'], (822942, 55))
        self.assertEqual(old['address_attrs.npy']['documented_shape'],
                         (822942, 54))
        self.assertEqual(old['tx_attrs.npy']['shape'], (202804, 182))
        self.assertEqual(old['tx_attrs.npy']['documented_shape'],
                         (202804, 183))
        try:
            download_data.BHN_FILES = {
                'address_attrs.npy': {
                    'id': 'x', 'shape': (3, 2),
                    'documented_shape': (3, 1)},
                'tx_attrs.npy': {
                    'id': 'x', 'shape': (4, 2),
                    'documented_shape': (4, 2)},
                'data.pkl': {'id': 'x', 'shape': None}}
            np.save(self.tmp / 'address_attrs.npy', np.zeros((3, 2)))
            np.save(self.tmp / 'tx_attrs.npy', np.zeros((5, 2)))
            with open(self.tmp / 'data.pkl', 'wb') as f:
                pickle.dump({'safe': 'static-check-only'}, f, protocol=4)
            with mock.patch('pickle.load',
                            side_effect=AssertionError('不得反序列化')):
                report = download_data.diagnose_bitheteronet(
                    self.tmp, verbose=False)
            self.assertEqual(report['invalid'], ['tx_attrs.npy'])
            self.assertEqual(report['documentation_mismatch'],
                             ['address_attrs.npy'])
            address = report['files']['address_attrs.npy']
            self.assertTrue(address['valid'])
            self.assertEqual(address['status'],
                             'valid_layout_documented_shape_mismatch')
            self.assertNotIn('expected_bytes', address)
            self.assertEqual(address['header_implied_file_bytes'],
                             address['actual_bytes'])
            self.assertLess(address['documented_shape_implied_file_bytes'],
                            address['header_implied_file_bytes'])
            tx = report['files']['tx_attrs.npy']
            self.assertEqual(tx['actual_shape'], [5, 2])
            self.assertTrue(tx['physical_complete'])
            self.assertFalse(tx['layout_shape_consistent'])
            self.assertEqual(tx['status'], 'shape_mismatch_physical_complete')
            self.assertEqual(tx['header_implied_file_bytes'],
                             tx['actual_bytes'])
            self.assertFalse(report['pickle_deserialized'])
            self.assertFalse(report['network_access'])
            self.assertFalse(report['column_semantics_fully_verified'])

            # 完整但 shape 未知时禁止把同一官方 ID 反复重下，也禁止自动放行。
            with mock.patch.object(
                    download_data, 'download_with_resume',
                    side_effect=AssertionError('不应发起网络下载')), \
                    mock.patch.object(
                    download_data, '_gdown_download_file',
                    side_effect=AssertionError('不应发起网络下载')):
                self.assertFalse(download_data.ensure_bitheteronet(
                    self.tmp, verbose=False))

            # 真正截断：头部可解析，但实际字节数小于头部隐含完整大小。
            np.save(self.tmp / 'tx_attrs.npy', np.zeros((4, 2)))
            with open(self.tmp / 'tx_attrs.npy', 'r+b') as f:
                f.truncate(os.path.getsize(self.tmp / 'tx_attrs.npy') - 3)
            report = download_data.diagnose_bitheteronet(
                self.tmp, verbose=False)
            tx = report['files']['tx_attrs.npy']
            self.assertFalse(tx['physical_complete'])
            self.assertEqual(tx['status'], 'physical_size_mismatch')
            self.assertGreater(tx['header_implied_file_bytes'],
                               tx['actual_bytes'])
            self.assertIn('物理大小不符', tx['error'])

            # 真实截断不是“未知完整布局”：允许进入续传通道；此处用本地
            # 合成写入代替网络，只验证路由，不伪造远程下载或官方哈希。
            def resume_candidate(url, dest, **kwargs):
                with open(dest, 'wb') as f:
                    np.save(f, np.zeros((4, 2)))

            with mock.patch.object(
                    download_data, 'download_with_resume',
                    side_effect=resume_candidate) as resume:
                self.assertEqual(download_data.ensure_bitheteronet(
                    self.tmp, verbose=False), self.tmp)
            self.assertEqual(resume.call_count, 1)
            report = download_data.diagnose_bitheteronet(
                self.tmp, verbose=False)
            self.assertEqual(report['invalid'], [])
            self.assertTrue(report['files']['tx_attrs.npy']['physical_complete'])

            (self.tmp / 'tx_attrs.npy').write_bytes(b'<html>error</html>')
            report = download_data.diagnose_bitheteronet(
                self.tmp, verbose=False)
            self.assertIn('HTML', report['files']['tx_attrs.npy']['error'])
        finally:
            download_data.BHN_FILES = old


if __name__ == '__main__':
    unittest.main(verbosity=2)
