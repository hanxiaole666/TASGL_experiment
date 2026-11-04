"""test_paper_tdsc.py — 论文数据生成程序的测试

覆盖:
  1. build 能跑通并产出全部表格/图输入/溯源清单
  2. 缺失来源时输出 n/a, 绝不编造数值
  3. 表格里的数字能回溯到 manifest.key_numbers
  4. check 模式能扫出不可追溯的数字
  5. 格式化与命名工具正确
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import paper_tdsc as pd

# Use the interpreter running the test suite; do not hard-code a local path.
PY = sys.executable
NUM = re.compile(r'(?<![\w.])(\d+\.\d+)(?![\w.])')
#: 真正的投稿目录; 不存在时端到端用例跳过, 而不是编一个目录糊过去。
PAPER = os.path.join('Doc', 'Kimi_Agent_CCF-A实验脚本-25', 'TASGL_TDSC投稿')


class TestFormatters(unittest.TestCase):
    def test_f4(self):
        self.assertEqual(pd.f4(0.12345), '0.1235')
        self.assertEqual(pd.f4(None), 'n/a')

    def test_ci(self):
        self.assertEqual(pd.ci([0.1, 0.2]), '[0.1000, 0.2000]')
        self.assertEqual(pd.ci(None), 'n/a')
        self.assertEqual(pd.ci([None, 0.2]), 'n/a')

    def test_disp(self):
        self.assertEqual(pd.disp('graphsage'), 'GraphSAGE')
        self.assertEqual(pd.disp('tasgl'), 'TASGL')
        self.assertEqual(pd.disp('unknown_model'), 'unknown_model')

    def test_esc(self):
        self.assertIn(r'\%', pd.esc('50%'))
        self.assertIn(r'\_', pd.esc('gate_collapsed_open'))


class TestBlocksWithMissingSources(unittest.TestCase):
    """来源全缺时, 所有区块必须声明 unavailable, 且不得产出数字。"""

    def setUp(self):
        self.src = {k: None for k in pd.SOURCES}

    def test_all_blocks_unavailable(self):
        for fn in (pd.block_main, pd.block_paired, pd.block_dual_track,
                   pd.block_gate, pd.block_hpo, pd.block_thresholds,
                   pd.block_calibration, pd.block_cost, pd.block_rootcause,
                   pd.block_second, pd.block_claims, pd.block_protocols,
                   pd.block_env):
            b = fn(self.src)
            self.assertFalse(b.get('available'), fn.__name__)

    def test_tex_renders_placeholder(self):
        for fn, args in ((pd.tex_main, (pd.block_main(self.src),)),
                         (pd.tex_paired, (pd.block_paired(self.src),)),
                         (pd.tex_hpo, (pd.block_hpo(self.src),)),
                         (pd.tex_calibration,
                          (pd.block_calibration(self.src),)),
                         (pd.tex_cost, (pd.block_cost(self.src),)),
                         (pd.tex_second, (pd.block_second(self.src),)),
                         (pd.tex_claims, (pd.block_claims(self.src),))):
            out = fn(*args)
            self.assertTrue(out.lstrip().startswith('%'),
                            f'{fn.__name__} 应输出注释占位: {out[:60]}')
            self.assertNotIn('0.5', out.split('\n')[0])


class TestBuildAndTrace(unittest.TestCase):
    """端到端: 用真实 results 目录构建, 再校验可追溯性。"""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists('./results/p4/final_comparison.json'):
            raise unittest.SkipTest('缺少 results/p4/final_comparison.json')
        cls.td = tempfile.mkdtemp()
        cls.paper = os.path.join(cls.td, 'paper')
        os.makedirs(cls.paper)
        r = subprocess.run([PY, 'paper_tdsc.py', 'build', '--results_dir',
                            '.', '--paper_dir', cls.paper],
                           capture_output=True, text=True)
        cls.rc, cls.out = r.returncode, r.stdout
        cls.manifest = json.load(
            open(os.path.join(cls.paper, 'provenance', 'manifest.json'),
                 encoding='utf-8'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def test_build_exit_zero(self):
        self.assertEqual(self.rc, 0, self.out)

    def test_all_tables_written(self):
        for name in ('main_results.tex', 'paired_stats.tex', 'gate_audit.tex',
                     'hpo_fair.tex', 'thresholds.tex', 'calibration.tex',
                     'cost_chain.tex', 'rootcause.tex', 'second_dataset.tex',
                     'claims.tex', 'legacy_contrast.tex', 'environment.tex'):
            p = os.path.join(self.paper, 'generated_tables', name)
            self.assertTrue(os.path.exists(p), name)

    def test_manifest_records_sources(self):
        self.assertTrue(self.manifest['sources'])
        for k, v in self.manifest['sources'].items():
            self.assertIn('sha256', v)
            self.assertEqual(len(v['sha256']), 64)

    def test_figure_inputs_schema(self):
        d = json.load(open(os.path.join(self.paper, 'generated_data',
                                        'figure_inputs.json'),
                           encoding='utf-8'))
        self.assertEqual(d['schema'], 'paper_figure_inputs_v1')
        self.assertIn('main10', d)
        self.assertIn('gate', d)

    def test_main_table_numbers_traceable(self):
        kn = self.manifest['key_numbers']['main10']
        p = os.path.join(self.paper, 'generated_tables', 'main_results.tex')
        txt = open(p, encoding='utf-8').read()
        allowed = set()
        for v in kn.values():
            for x in (v['f1'], v['f1_sd'], v['latency_ms'],
                      v.get('latency_sd'), v.get('precision'),
                      v.get('recall')):
                if x is None:
                    continue
                for nd in range(1, 6):
                    allowed.add(f'{x:.{nd}f}')
            allowed.add(f"{v['params']:,}".replace(',', ''))
        unknown = []
        for line in txt.splitlines():
            if line.startswith('\\') or '&' not in line:
                continue
            for m in NUM.finditer(line):
                if m.group(1) not in allowed:
                    unknown.append((line.strip()[:60], m.group(1)))
        self.assertEqual(unknown, [], f'不可追溯: {unknown}')

    def test_check_mode_runs(self):
        r = subprocess.run([PY, 'paper_tdsc.py', 'check', '--paper_dir',
                            self.paper], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        rep = json.load(open(os.path.join(self.paper, 'provenance',
                                          'traceability_check.json'),
                             encoding='utf-8'))
        self.assertEqual(rep['schema'], 'paper_traceability_check_v1')

    def test_lineage_written(self):
        p = os.path.join(self.paper, 'provenance', 'lineage.md')
        self.assertTrue(os.path.exists(p))
        self.assertIn('## Sources', open(p, encoding='utf-8').read())


class TestNumberNormalisation(unittest.TestCase):
    """正文与生成产物必须用同一套数字归一化, 否则会互相"对不上"。"""

    def test_brace_comma_is_a_thousands_separator(self):
        # 20{,}480 与 20,480 是同一个数的两种 LaTeX 写法
        self.assertEqual(pd._normalize_numbers('20{,}480'), '20480')
        self.assertEqual(pd._normalize_numbers('20,480'), '20480')
        self.assertEqual(pd._normalize_numbers('3,700,550'), '3700550')

    def test_scan_and_allowed_use_the_same_rule(self):
        """生成器写 20,480, 正文写 20{,}480 —— 必须互相可追溯。"""
        with tempfile.TemporaryDirectory() as d:
            paper = os.path.join(d, 'paper')
            os.makedirs(os.path.join(paper, 'generated_tables'))
            with open(os.path.join(paper, 'generated_tables', 't.tex'),
                      'w', encoding='utf-8') as f:
                f.write('limit & 20,480 \\\\\n')
            with open(os.path.join(paper, 'body.tex'), 'w',
                      encoding='utf-8') as f:
                f.write('Pass if parameters $\\leq 20{,}480$.\n')
            n = pd.check(paper)
            self.assertEqual(n, 0, '同一个数在两种写法下必须能互相匹配')

    def test_real_difference_is_still_caught(self):
        """归一化不得把真不同的数变成相同。"""
        with tempfile.TemporaryDirectory() as d:
            paper = os.path.join(d, 'paper')
            os.makedirs(os.path.join(paper, 'generated_tables'))
            with open(os.path.join(paper, 'generated_tables', 't.tex'),
                      'w', encoding='utf-8') as f:
                f.write('limit & 20,480 \\\\\n')
            with open(os.path.join(paper, 'body.tex'), 'w',
                      encoding='utf-8') as f:
                f.write('Pass if parameters $\\leq 30,000$.\n')
            self.assertGreater(pd.check(paper), 0)


class TestEndToEndTraceability(unittest.TestCase):
    """正文里的每个数字都必须能在生成产物里找到出处。"""

    def test_body_numbers_all_traceable(self):
        if not os.path.isdir(PAPER):
            self.skipTest('submission directory not present')
        n = pd.check(PAPER)
        self.assertEqual(n, 0, '存在无法溯源的数字, 见 '
                               'provenance/traceability_check.json')


class TestNoFabrication(unittest.TestCase):
    """不允许出现合成值: 缺失必须是 n/a, 不能是 0 或 0.0。"""

    def test_missing_calibration_renders_na(self):
        src = {k: None for k in pd.SOURCES}
        src['calib'] = {'overall_ece': None, 'positive_class_ece': None,
                        'negative_class_ece': None, 'brier': None,
                        'bins': 15, 'n': 16670, 'detail': {}}
        txt = pd.tex_calibration(pd.block_calibration(src))
        self.assertIn('n/a', txt)
        self.assertNotIn('0.0000', txt)


if __name__ == '__main__':
    unittest.main(verbosity=2)
