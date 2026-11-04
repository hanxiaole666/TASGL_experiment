"""test_p5_tex_conformance.py — 模板合规检查本身也必须可验证。

C17 声称"摘要整段加粗是 IEEEtran 期刊模式的设计"。若这条检查只是无条件
返回 PASS, 它就不提供任何证据。这里用一个改掉 \\bfseries 的 class 文件
反向验证: 定义一旦变了, C17 必须 FAIL。
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import p5_tex_conformance

ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(ROOT, 'Doc', 'Kimi_Agent_CCF-A实验脚本-25',
                     'TASGL_TDSC投稿')


def _by_id(res):
    return {r['id']: r for r in res}


@unittest.skipUnless(os.path.exists(PAPER), '需要投稿目录')
class TestConformanceOnTheRealManuscript(unittest.TestCase):
    def test_every_check_passes(self):
        res = p5_tex_conformance.check(PAPER)
        failed = [r for r in res if r['status'] == 'FAIL']
        self.assertFalse(failed, str([(r['id'], r['detail'])
                                      for r in failed]))

    def test_c17_is_evidence_based_not_vacuous(self):
        """把 class 里的 \\bfseries 去掉, C17 必须转为 FAIL。"""
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(PAPER, 'IEEEtran.cls')
            cls = open(src, encoding='utf-8', errors='replace').read()
            self.assertIn('\\bfseries', cls)
            stripped = cls.replace('\\bfseries', '\\mdseries')
            self.assertNotEqual(stripped, cls,
                                '测试前提失效: class 里没有 \\bfseries, '
                                '说明 IEEEtran.cls 版本已变')
            mini = os.path.join(td, 'mini')
            os.makedirs(mini)
            shutil.copy(os.path.join(PAPER, 'main.tex'),
                        os.path.join(mini, 'main.tex'))
            with open(os.path.join(mini, 'IEEEtran.cls'), 'w',
                      encoding='utf-8') as f:
                f.write(stripped)
            res = _by_id(p5_tex_conformance.check(mini))
            self.assertEqual(res['C17']['status'], 'FAIL')

    def test_missing_class_file_is_reported_not_silently_passed(self):
        with tempfile.TemporaryDirectory() as td:
            mini = os.path.join(td, 'mini')
            os.makedirs(mini)
            shutil.copy(os.path.join(PAPER, 'main.tex'),
                        os.path.join(mini, 'main.tex'))
            res = _by_id(p5_tex_conformance.check(mini))
            self.assertEqual(res['C16']['status'], 'FAIL')
            self.assertEqual(res['C17']['status'], 'FAIL')


if __name__ == '__main__':
    unittest.main(verbosity=2)
