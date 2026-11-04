"""test_p5_third_paper.py — 第三数据集表不得把"跑不动"写成"跑得还行"

守的底线:
  1. TASGL 的 OOM 必须显示成 oom, 不得用任何数字替代。
  2. 表必须写明时间语义退化和"不承载任何门限判决"。
  3. 正类率来自落盘产物, 没有就写不出来, 不许猜。
  4. 带下划线的值必须转义, 否则正文会掉进数学模式。
"""
import json
import os
import re

import pytest

import p5_third_paper as tp


def test_oom_is_reported_as_oom():
    d = tp.build('results')
    tasgl = [r for r in d['rows'] if r['model'] == 'TASGL']
    assert tasgl, '探针里没有 TASGL 行'
    for r in tasgl:
        assert r['status'] == 'oom'
        assert r['f1'] is None and r['precision'] is None


def test_no_gate_verdict_claim():
    d = tp.build('results')
    assert d['carries_no_gate_result'] is True
    assert 'time step' in (d['temporal_component'] or '').lower()
    assert 'temporal' in (d['what_this_can_test'] or '').lower()


def test_positive_rate_comes_from_artifact():
    d = tp.build('results')
    src = json.load(open('results/p5/dgraphfin_probe.json', encoding='utf-8'))
    if d['positive_rate'] is None:
        assert 'split_audit' not in src or not (
            src.get('split_audit') or {}).get('per_split')
    else:
        assert d['positive_rate'] == pytest.approx(
            src['split_audit']['per_split']['train']['positive_rate'])


def test_underscores_are_escaped(tmp_path):
    d = tp.build('results')
    s = open(tp.tex(d, str(tmp_path)), encoding='utf-8').read()
    flush = s.split('\\begin{flushleft}')[1]
    body = flush.split('\\end{flushleft}')[0]
    assert '\\_' in body or '_' not in body, body[:120]
    # 不允许出现未转义的下划线
    assert not re.search(r'(?<!\\)_', body)


def test_no_cjk_and_balanced(tmp_path):
    d = tp.build('results')
    s = open(tp.tex(d, str(tmp_path)), encoding='utf-8').read()
    assert not any('一' <= ch <= '鿿' for ch in s)
    assert s.count('\\begin{table}') == s.count('\\end{table}')
