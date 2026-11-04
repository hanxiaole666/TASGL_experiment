"""test_p6_attacks.py — H6 非 oracle 攻击矩阵的追溯性测试。

覆盖三件事, 顺序与论文主张一致:

  1. 协议先登记后执行: 登记表里必须有 v3 的测量协议, 且锁定哈希与攻击
     矩阵文件一致; 事后改攻击必须被拒绝。
  2. 攻击者边界: 标签不得进入攻击构造 (函数签名没有标签入口, 产物里
     oracle_labels_used=False), 攻击者知识逐条声明, 未声明的一律写明。
  3. 聚合口径: 先在模型种子内对攻击种子取均值, 再跨种子 bootstrap;
     出表不得引入 LaTeX 注释符或列数不匹配。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest

import numpy as np

import p4_thresholds
import p6_attacks

ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT = os.path.join(ROOT, 'results', 'p6',
                        'nonoracle_attack_matrix.json')
MATRIX = os.path.join(ROOT, 'p1_attack_matrix.json')
ANCHORS = os.path.join(ROOT, 'p4_anchors.json')


def _anchors():
    with open(ANCHORS, encoding='utf-8') as f:
        return json.load(f)


def _gate_values(table, names=('H1', 'H2', 'H5')):
    out = {}
    for h in names:
        gates = (table['hypotheses'].get(h) or {}).get('gates') or []
        out[h] = [(g.get('metric'), g.get('pass_op'), g.get('pass_value'),
                   g.get('falsify_value')) for g in gates]
    return out


class TestProtocolRegisteredBeforeExecution(unittest.TestCase):
    def test_protocol_records_matrix_hash_and_attacker_knowledge(self):
        proto = p4_thresholds.h6_measurement_protocol(
            attack_matrix_path=MATRIX, n_model_seeds=10)
        with open(MATRIX, 'rb') as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(proto['attack_matrix_sha256'], sha)
        self.assertEqual(proto['attack_matrix_protocol'],
                         'p1_nonoracle_attack_matrix_v1')
        self.assertFalse(proto['oracle_labels'])
        self.assertEqual(proto['attacker_knowledge_declared'],
                         ['black_box', 'gray_box'])
        self.assertEqual(proto['attacker_knowledge_not_declared'],
                         ['white_box'])
        self.assertIsNone(proto['gate'])
        self.assertIn('alert_capacity', proto['gate_reason'])
        self.assertEqual(proto['paired_unit'], 'model seed')
        self.assertIn('nested repeat', proto['attack_seed_role'])
        self.assertTrue(all(a['oracle_labels'] is False
                            for a in proto['declared_attacks']))

    def test_register_v3_does_not_move_h1_h2_h5(self):
        """升到 v3 只动 H6 的状态与测量协议, 门限数值一个字都不能改。"""
        anchors = _anchors()
        proto = p4_thresholds.h6_measurement_protocol(
            attack_matrix_path=MATRIX, n_model_seeds=10)
        with tempfile.TemporaryDirectory() as td:
            v2 = p4_thresholds.register_justified_thresholds(
                anchors, out_dir=td, declared_seeds=list(range(10)))
            with tempfile.TemporaryDirectory() as td3:
                v3 = p4_thresholds.register_justified_thresholds(
                    anchors, out_dir=td3, declared_seeds=list(range(10)),
                    h6_protocol=proto)
                self.assertEqual(_gate_values(v2['table']),
                                 _gate_values(v3['table']))
                self.assertEqual(v2['table']['registration_version'],
                                 'p4_threshold_table_v2')
                self.assertEqual(v3['table']['registration_version'],
                                 'p4_threshold_table_v3')
                self.assertEqual(v2['table']['hypotheses']['H6']['status'],
                                 'not_testable')
                self.assertEqual(v3['table']['hypotheses']['H6']['status'],
                                 'measured_no_gate')
                self.assertFalse(v3['table']['hypotheses']['H6']['open_item'])
                self.assertEqual(v3['table']['hypotheses']['H6']['gates'], [])

    def test_v3_changelog_declares_values_unchanged_and_pre_registration(self):
        anchors = _anchors()
        proto = p4_thresholds.h6_measurement_protocol(
            attack_matrix_path=MATRIX, n_model_seeds=10)
        with tempfile.TemporaryDirectory() as td:
            res = p4_thresholds.register_justified_thresholds(
                anchors, out_dir=td, declared_seeds=list(range(10)),
                h6_protocol=proto)
            entry = res['table']['registration_changelog'][-1]
            self.assertEqual(entry['version'], 'v3')
            self.assertFalse(entry['threshold_values_changed'])
            self.assertTrue(entry['registered_before_execution'])

    def test_verify_lock_rejects_an_edited_attack_matrix(self):
        """改了攻击却没重新登记, 必须被挡住。"""
        import p1_attacks
        with tempfile.TemporaryDirectory() as td:
            cfg = json.load(open(MATRIX, encoding='utf-8'))
            p1_attacks.validate_attack_matrix(cfg)
            path = os.path.join(td, 'matrix.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f)
            proto = p4_thresholds.h6_measurement_protocol(
                attack_matrix_path=path, n_model_seeds=10)
            p6_attacks.verify_lock(proto, attack_matrix=path)
            edited = copy.deepcopy(cfg)
            edited['attacks'][0]['budget'] = 0.9
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(edited, f)
            with self.assertRaisesRegex(RuntimeError, '攻击矩阵文件已改动'):
                p6_attacks.verify_lock(proto, attack_matrix=path)

    def test_load_lock_refuses_when_protocol_absent(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, '未登记'):
                p6_attacks.load_lock(out_dir=td)


class TestPredictionIndependence(unittest.TestCase):
    """扰动能否跨模型种子复用, 必须由实测决定, 不能靠推断。"""

    def setUp(self):
        self.n = 40
        rng = np.random.RandomState(0)
        self.x = rng.rand(self.n, 4).astype(np.float32)
        pairs = [(i, (i + 1) % self.n) for i in range(self.n)]
        pairs += [(i, (i + 5) % self.n) for i in range(0, self.n, 2)]
        self.edge = np.array([[a, b] for a, b in pairs]
                             + [[b, a] for a, b in pairs], dtype=np.int64).T
        self.time = np.repeat(np.arange(self.n // 4), 4)
        self.eval_mask = self.time >= 5
        self.p = rng.rand(self.n)

    def _specs(self):
        cfg = json.load(open(MATRIX, encoding='utf-8'))
        return cfg['attacks']

    def test_classification_matches_the_code_path(self):
        out = p6_attacks.classify_attacks(
            self._specs(), self.x, self.edge, self.time, self.eval_mask, self.p)
        by_name = {s['name']: s['p_independent'] for s in out}
        for name, flag in by_name.items():
            self.assertEqual(flag, 'confidence' not in name, name)
        self.assertTrue(by_name['edge_delete_random_10'])
        self.assertFalse(by_name['edge_delete_confidence_50'])
        self.assertFalse(by_name['feature_noise_confidence_05'])

    def test_classification_refuses_to_guess_when_probe_disagrees(self):
        """实测与推断不一致时必须报错, 不能静默选一个。"""
        import p1_attacks
        original = p1_attacks.apply_nonoracle_attack
        try:
            def always_same(*args, **kwargs):
                x, e, m = original(*args, **kwargs)
                m = dict(m, edge_sha256='fixed', x_sha256='fixed')
                return x, e, m
            p1_attacks.apply_nonoracle_attack = always_same
            with self.assertRaisesRegex(RuntimeError, '预测依赖性'):
                p6_attacks.classify_attacks(
                    self._specs(), self.x, self.edge, self.time,
                    self.eval_mask, self.p)
        finally:
            p1_attacks.apply_nonoracle_attack = original

    def test_cache_is_used_for_edges_only(self):
        self.assertTrue(p6_attacks.cache_applies({'capability': 'edge_delete'}))
        self.assertTrue(p6_attacks.cache_applies({'capability': 'edge_inject'}))
        self.assertFalse(
            p6_attacks.cache_applies({'capability': 'feature_noise'}))

    def test_cache_key_ignores_predictions_only_when_independent(self):
        cache = p6_attacks.PerturbationCache()
        ind = {'name': 'a', 'p_independent': True}
        dep = {'name': 'a', 'p_independent': False}
        p1 = np.array([0.1, 0.2, 0.3])
        p2 = np.array([0.9, 0.8, 0.7])
        self.assertEqual(cache.key(ind, 0, p1), cache.key(ind, 0, p2))
        self.assertNotEqual(cache.key(dep, 0, p1), cache.key(dep, 0, p2))
        self.assertEqual(cache.key(dep, 0, p1), cache.key(dep, 0, p1.copy()))


class TestAggregation(unittest.TestCase):
    def _rows(self, models=('tasgl',), seeds=(0, 1, 2), attack_seeds=(0, 1)):
        rows, clean = [], []
        for mi, m in enumerate(models):
            for s in seeds:
                base = 0.5 + 0.01 * s + 0.02 * mi
                clean.append({'model': m, 'seed': s, 'f1': base,
                              'threshold': 0.9})
                for a in attack_seeds:
                    # 攻击种子内的下降各不相同, 必须先取均值再跨种子聚合
                    drop = 0.10 * (a + 1) + 0.01 * s
                    rows.append({'model': m, 'seed': s, 'attack_name': 'atk',
                                 'attack_seed': a, 'clean_f1': base,
                                 'attacked_f1': base * (1 - drop),
                                 'relative_drop': drop, 'cost': {}})
        return rows, clean

    def test_attack_seeds_are_averaged_inside_each_model_seed(self):
        rows, clean = self._rows()
        out = p6_attacks.summarise(
            rows, clean, ['tasgl'], [0, 1, 2], [0, 1],
            [{'name': 'atk', 'capability': 'edge_delete', 'selection': 'random',
              'knowledge': 'black_box', 'goal': 'g', 'budget_type': 'edge_fraction',
              'budget': 0.1}], {'gate_reason': 'none'}, (35, 49), 'sha', None)
        entry = out['per_attack'][0]['models']['tasgl']
        # 每个种子内 = mean(0.10, 0.20) = 0.15, 再加 0.01*s -> 0.15,0.16,0.17
        self.assertAlmostEqual(entry['per_seed_drop'][0], 0.15)
        self.assertAlmostEqual(entry['per_seed_drop'][2], 0.17)
        self.assertAlmostEqual(entry['relative_drop_mean'], 0.16, places=6)
        self.assertEqual(entry['n_model_seeds'], 3)
        lo, hi = entry['relative_drop_ci95']
        self.assertLessEqual(lo, 0.16)
        self.assertGreaterEqual(hi, 0.16)

    def test_attack_seeds_are_not_counted_as_extra_model_seeds(self):
        rows, clean = self._rows()
        out = p6_attacks.summarise(
            rows, clean, ['tasgl'], [0, 1, 2], [0, 1],
            [{'name': 'atk', 'capability': 'edge_delete', 'selection': 'random',
              'knowledge': 'black_box', 'goal': 'g', 'budget_type': 'edge_fraction',
              'budget': 0.1}], {'gate_reason': 'none'}, (35, 49), 'sha', None)
        self.assertEqual(out['per_attack'][0]['models']['tasgl']
                         ['n_model_seeds'], 3)
        self.assertEqual(out['attack_seed_role'],
                         'nested repeat inside one model seed')

    def test_h6_status_is_measured_and_never_gated(self):
        rows, clean = self._rows()
        out = p6_attacks.summarise(
            rows, clean, ['tasgl'], [0, 1, 2], [0, 1],
            [{'name': 'atk', 'capability': 'edge_delete', 'selection': 'random',
              'knowledge': 'black_box', 'goal': 'g', 'budget_type': 'edge_fraction',
              'budget': 0.1}], {'gate_reason': 'none'}, (35, 49), 'sha', None)
        self.assertEqual(out['h6_status'], 'measured_no_gate')
        self.assertIsNone(out['gate'])
        self.assertFalse(out['oracle_labels_used'])


class TestTex(unittest.TestCase):
    def _data(self, models=('tasgl', 'mlp')):
        rows, clean = [], []
        for m in models:
            for s in (0, 1):
                clean.append({'model': m, 'seed': s, 'f1': 0.5,
                              'threshold': 0.9})
                for a in (0, 1):
                    rows.append({'model': m, 'seed': s, 'attack_name': 'atk',
                                 'attack_seed': a, 'clean_f1': 0.5,
                                 'attacked_f1': 0.4, 'relative_drop': 0.2,
                                 'cost': {'removed_undirected_pairs': 12}})
        return p6_attacks.summarise(
            rows, clean, list(models), [0, 1], [0, 1],
            [{'name': 'atk', 'capability': 'edge_delete', 'selection': 'random',
              'knowledge': 'black_box', 'goal': 'untargeted',
              'budget_type': 'edge_fraction', 'budget': 0.1}],
            {'gate_reason': 'none'}, (35, 49), 'sha', None)

    def test_tex_is_comment_free_and_column_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, 'in.json')
            with open(src, 'w', encoding='utf-8') as f:
                json.dump(self._data(), f)
            path = p6_attacks.tex(src, out_dir=td)
            text = open(path, encoding='utf-8').read()
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith('%'):
                continue
            # 未转义的 % 会把该行剩下的内容整段注释掉
            for i, ch in enumerate(line):
                if ch == '%':
                    self.assertTrue(i == 0 or line[i - 1] == '\\',
                                    '未转义的 %% 出现在: %s' % line)
        # 两个表都必须是表头列数 == 数据列数
        import re

        def ncol_of(spec):
            # 去掉 >{...} 构造块, 剩下的 l/c/r/X/p{..} 就是列说明符
            s = re.sub(r'>\{[^{}]*\}', '', spec)
            return len(re.findall(r'p\{[^{}]*\}|[lcrX]', s))

        pat = r'\\begin\{tabularx?\}\{([^}]*)\}(.*?)\\end\{tabularx?\}'
        found = re.findall(pat, text, re.S)
        self.assertTrue(found, '没有找到表体')
        for spec, inner in found:
            ncol = ncol_of(spec)
            for row in [r for r in inner.split('\\\\') if r.strip()]:
                if '\\hline' in row:
                    continue
                self.assertEqual(row.count('&') + 1, ncol,
                                 '列数不匹配: %s' % row.strip()[:60])
        self.assertIn('\\label{tab:h6matrix}', text)
        self.assertIn('\\label{tab:h6effect}', text)


@unittest.skipUnless(os.path.exists(ARTIFACT), '需要 results/p6 的实际产物')
class TestRealArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(ARTIFACT, encoding='utf-8') as f:
            cls.data = json.load(f)

    def test_matrix_is_complete_and_oracle_free(self):
        d = self.data
        expected = (len(d['models']) * len(d['seeds'])
                    * len(d['per_attack']) * len(d['attack_seeds']))
        self.assertEqual(d['n_runs'], expected)
        self.assertFalse(d['oracle_labels_used'])
        self.assertTrue(all(r['oracle_labels'] is False for r in d['runs']))
        self.assertNotIn('white_box', d['attacker_knowledge_declared'])
        self.assertEqual(d['attacker_knowledge_not_declared'], ['white_box'])

    def test_mlp_is_invariant_to_every_edge_attack(self):
        """MLP 不用邻接结构, 边攻击必须对它完全无效。

        这条既是正确性检查也是可解释性检查: 若边攻击居然改变了 MLP 的 F1,
        说明攻击图被错误地喂进了不该受影响的地方。
        """
        d = self.data
        if 'mlp' not in d['models']:
            self.skipTest('本次执行不含 mlp')
        for e in d['per_attack']:
            if not e['capability'].startswith('edge_'):
                continue
            self.assertAlmostEqual(e['models']['mlp']['relative_drop_mean'],
                                   0.0, places=12, msg=e['attack_name'])

    def test_clean_f1_reproduces_the_published_ten_seed_run(self):
        check = self.data.get('clean_f1_vs_final_comparison')
        self.assertIsNotNone(check)
        # tab:repro 量化的同机抖动量级远小于 0.02
        self.assertLess(check['max_abs_diff'], 0.02,
                        str(check.get('max_abs_diff_by_model')))

    def test_every_attack_declares_knowledge_and_cost(self):
        for e in self.data['per_attack']:
            self.assertIn(e['knowledge'], {'black_box', 'gray_box', 'white_box'})
            self.assertTrue(e['budget_type'])
            self.assertGreater(float(e['budget']), 0)
            cost = e['realised_cost_first_seed']
            self.assertTrue(cost)
            for model_cost in cost.values():
                self.assertTrue(model_cost)

    def test_h6_carries_no_gate(self):
        self.assertEqual(self.data['h6_status'], 'measured_no_gate')
        self.assertIsNone(self.data['gate'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
