"""test_ledger_reconcile.py — 台账对账程序的回归测试

覆盖点:
  1. 条目齐全 (EVD-01..18 / P0-1..9 / P1-1..8) 且编号唯一
  2. 状态取值合法, 字段非空
  3. build 在空结果目录下不崩且不编造: 数值一律 n.a., 状态为 open/partial
  4. 非 ASCII 文案必须被拦截 (论文表体只能是英文)
  5. 生成的 LaTeX 表列数与表头一致, 且不含未转义的 %
  6. 在有真实结果的目录下, EVD-02/EVD-03 必须取到实测数值
"""

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ledger_reconcile as L  # noqa: E402

REAL = os.environ.get('TASGL_RESULTS_DIR', HERE)


class TestItemIntegrity(unittest.TestCase):
    def test_all_item_ids_present(self):
        items = L.build_items(REAL)
        ids = [r['id'] for r in items]
        want = (['EVD-%02d' % i for i in range(1, 19)]
                + ['P0-%d' % i for i in range(1, 10)]
                + ['P1-%d' % i for i in range(1, 9)])
        for w in want:
            self.assertIn(w, ids, 'missing ledger item %s' % w)

    def test_ids_unique(self):
        ids = [r['id'] for r in L.build_items(REAL)]
        self.assertEqual(len(ids), len(set(ids)))

    def test_status_and_fields(self):
        for r in L.build_items(REAL):
            self.assertIn(r['status'], L.STATUS)
            for k in ('topic', 'ledger', 'observed', 'source', 'note'):
                self.assertTrue(r.get(k), '%s empty %s' % (r['id'], k))

    def test_no_non_ascii_in_paper_fields(self):
        bad = L._ascii_guard(L.build_items(REAL))
        self.assertEqual(bad, [], 'non-ASCII would leak into LaTeX: %s' % bad)


class TestNoFabrication(unittest.TestCase):
    """空结果目录: 不许出现任何编造数值。"""

    def test_empty_results_dir_has_no_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            items = L.build_items(d)
            for r in items:
                if r['id'] in ('EVD-02', 'EVD-03', 'EVD-04', 'EVD-05',
                               'EVD-09', 'EVD-18'):
                    self.assertTrue(
                        'n.a.' in r['observed']
                        or r['observed'].startswith('source missing'),
                        '%s produced a value with no source: %r'
                        % (r['id'], r['observed']))
            self.assertNotIn('unchanged', [r['status'] for r in items])

    def test_empty_results_dir_build_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = L.build(d, os.path.join(d, 'paper'))
            self.assertTrue(os.path.exists(
                os.path.join(d, 'paper', 'provenance',
                             'ledger_reconciliation.json')))
            self.assertEqual(out['counts'].get('unchanged', 0), 0)


class TestRenderers(unittest.TestCase):
    def test_ascii_guard_flags_chinese(self):
        items = [{'id': 'X', 'topic': '主题', 'ledger': 'a',
                  'observed': 'b', 'note': 'c', 'status': 'open',
                  'source': 's'}]
        self.assertEqual(L._ascii_guard(items), ['X.topic'])

    def test_tex_escapes_percent(self):
        items = [{'id': 'X', 'topic': 't', 'ledger': '50%',
                  'observed': 'n.a.', 'note': 'n', 'status': 'open',
                  'source': 's'}]
        tex = L.tex_table(items)
        self.assertIn('50\\%', tex)
        self.assertNotIn('50%', tex.replace('\\%', ''))

    def test_tex_column_count_matches_header(self):
        items = L.build_items(REAL)
        tex = L.tex_table(items)
        self.assertIn('\\begin{tabularx}{\\textwidth}{lXXXXl}', tex)
        for line in tex.splitlines():
            if line.endswith(' \\\\') and '&' in line:
                self.assertEqual(line.count('&'), 4,
                                 'row has the wrong column count: %s' % line)

    def test_json_roundtrip(self):
        items = L.build_items(REAL)
        blob = json.dumps({'items': items}, ensure_ascii=False)
        self.assertEqual(json.loads(blob)['items'], items)


class TestRealValues(unittest.TestCase):
    """有真实结果时, 关键条目必须读到实测数值。"""

    @unittest.skipUnless(
        os.path.exists(os.path.join(REAL, 'results', 'p4',
                                    'final_comparison.json')),
        'no P4 final comparison on disk')
    def test_evd02_reads_ten_seed_means(self):
        items = {r['id']: r for r in L.build_items(REAL)}
        obs = items['EVD-02']['observed']
        self.assertNotIn('n.a.', obs)
        self.assertIn('10 seeds', obs)
        self.assertEqual(items['EVD-02']['status'], 'updated')

    @unittest.skipUnless(
        os.path.exists(os.path.join(REAL, 'results', 'p4',
                                    'environment.json')),
        'no environment fingerprint on disk')
    def test_evd18_fingerprint_complete(self):
        items = {r['id']: r for r in L.build_items(REAL)}
        self.assertIn('complete = yes', items['EVD-18']['observed'])
        self.assertEqual(items['EVD-18']['status'], 'updated')


if __name__ == '__main__':
    unittest.main(verbosity=2)
