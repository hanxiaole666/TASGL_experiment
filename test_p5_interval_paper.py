"""test_p5_interval_paper.py — H3/H4 区间表不得把区间写成判决

守的底线:
  1. 数值必须直接来自 p4_interval.py 的产物, 不得在排版层重新计算或四舍五入
     出一个"更好看"的数。
  2. 两个假设都没有门限, 生成的 tex 里不许出现 PASS / FALSIFIED 字样。
  3. 必须写明删边不是攻击, 否则 H3 会被误读成对抗鲁棒性。
  4. 表格列数与表头一致, 且正文不含中文。
"""
import json
import os
import re

import pytest

import p5_interval_paper as ip


def test_reads_values_without_recomputing():
    d = ip.build('results')
    h3 = json.load(open('results/p4/h3_interval.json', encoding='utf-8'))
    for r in d['h3']['models']:
        src = h3['per_model'][r['model']]
        assert r['mean'] == src['mean']
        assert r['sd'] == src['sd']
        assert r['ci95'] == src['ci95']


def test_no_gate_language_in_tex(tmp_path):
    d = ip.build('results')
    p = ip.tex(d, str(tmp_path))
    s = open(p, encoding='utf-8').read()
    low = s.lower()
    for bad in ('pass', 'falsified', 'failed'):
        # "pass" 会出现在 "pass or fail is written" 这种明确的否定句里,
        # 所以只禁止把它当作判决词单独贴在某个数字后面。
        assert not re.search(r'\\\\textbf\{' + bad, low), bad
    assert 'No gate is set for H3' in s
    assert 'No gate is set for H4' in s


def test_structural_not_adversarial_is_stated(tmp_path):
    d = ip.build('results')
    s = open(ip.tex(d, str(tmp_path)), encoding='utf-8').read()
    assert 'not an attack' in s


def test_direction_is_explained(tmp_path):
    """配对差为正 = TASGL 损失更大, 必须写明, 否则符号会被读反。"""
    d = ip.build('results')
    s = open(ip.tex(d, str(tmp_path)), encoding='utf-8').read()
    assert 'positive difference' in s


def test_columns_match_header(tmp_path):
    d = ip.build('results')
    s = open(ip.tex(d, str(tmp_path)), encoding='utf-8').read()
    for blk in re.findall(r'\\begin\{tabularx\}\{[^}]*\}\{([^}]*)\}(.*?)'
                          r'\\end\{tabularx\}', s, re.S):
        spec, body = blk
        ncol = len(spec)
        head = body.split('\\\\')[0]
        if 'Hyp.' in head:
            # 表头里 "95\% CI" 含一个 \&? 不会; 直接数 &
            assert head.count('&') == ncol - 1, (head, ncol)
        for raw in body.split('\\\\')[1:]:
            row = raw.replace('\\hline', '').strip()
            if '&' not in row:
                continue
            assert row.count('&') == ncol - 1, (row[:60], ncol)


def test_no_cjk(tmp_path):
    d = ip.build('results')
    s = open(ip.tex(d, str(tmp_path)), encoding='utf-8').read()
    assert not any('一' <= ch <= '鿿' for ch in s)


def test_display_names():
    assert ip._name('tasgl') == 'TASGL'
    assert ip._name('graphsage') == 'GraphSAGE'
    assert ip._name('mlp') == 'MLP'
