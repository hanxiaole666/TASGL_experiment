"""test_p5_bith20.py — 20 种子分析的统计正确性与防自欺

守的底线:
  1. 最小可达 Wilcoxon p 必须按 2/2^n 报告, 且"顶到地板"必须能被识别。
  2. Holm 校正必须真正改变结论的可能性 (不是把原 p 抄一遍)。
  3. 跨窗稳定性只能用"同一模型同一阈值", 因此逐窗必须共用同一 seed 集合。
  4. 缺失种子必须让结论降级, 不得用截断凑数。
"""
import json
import os
import re
import tempfile

import pytest

import p5_bith20 as bh


def _rows(mu, sd, n, seed0=0, windows=None, drop=()):
    import numpy as np
    rng = np.random.default_rng(seed0)
    out = []
    for s in range(n):
        if s in drop:
            out.append({'seed': s, 'status': 'oom', 'f1': None})
            continue
        f1 = float(np.clip(rng.normal(mu, sd), 0.0, 1.0))
        r = {'seed': s, 'status': 'ok', 'f1': f1}
        if windows:
            r['windows'] = {w: {'f1': float(np.clip(
                f1 + rng.normal(0, 0.02), 0, 1)), 'n_labeled': 100}
                for w in windows}
        out.append(r)
    return out


def _mk(mu_base=0.56, mu_mlp=0.41, n=20, windows=None, drop=()):
    return {'schema': 'p4_second_dataset_v1', 'seeds': list(range(n)),
            'config_source': {'hpo_configs_applied': False},
            'models': {
                'tasgl': {'rows': _rows(mu_base, 0.03, n, 1, windows, drop),
                          'n_complete': n - len(drop), 'complete': not drop,
                          'mean_f1': mu_base, 'sd_f1': 0.03},
                'mlp': {'rows': _rows(mu_mlp, 0.05, n, 2, windows, drop),
                        'n_complete': n - len(drop), 'complete': not drop,
                        'mean_f1': mu_mlp, 'sd_f1': 0.05},
            }}


def test_min_attainable_p_scales_with_n(tmp_path):
    """n=10 地板 0.00195, n=20 地板 1.91e-6。"""
    d = _mk(n=20)
    p = tmp_path / 'x.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    res = bh.build(str(p))
    rep = res['comparisons']['mlp'] if 'mlp' in res['comparisons'] else None
    assert rep is not None
    assert rep['min_attainable_wilcoxon_p'] == pytest.approx(2 / 2 ** 20)
    f10 = res['power_design']['mlp']
    assert f10['min_attainable_p_n10'] == pytest.approx(0.001953125)
    assert f10['min_attainable_p_n20'] == pytest.approx(2 / 2 ** 20, rel=1e-6)


def test_holm_is_applied_and_can_change_conclusion(tmp_path):
    """Holm 后的 p 必须 >= 原始 p, 且至少不小于最小者。"""
    d = _mk(n=20)
    d['models']['graphsage'] = {
        'rows': _rows(0.50, 0.06, 20, 3), 'n_complete': 20,
        'complete': True, 'mean_f1': 0.50, 'sd_f1': 0.06}
    p = tmp_path / 'y.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    res = bh.build(str(p))
    assert set(res['family']) == {'mlp', 'graphsage'}
    for nm, rep in res['comparisons'].items():
        assert rep['wilcoxon_p_holm'] >= rep['wilcoxon_p'] - 1e-12
    # 两个比较中最小的那个 Holm 后等于 m*p
    ps = sorted(rep['wilcoxon_p'] for rep in res['comparisons'].values())
    hs = sorted(rep['wilcoxon_p_holm'] for rep in res['comparisons'].values())
    assert hs[0] == pytest.approx(min(1.0, 2 * ps[0]))


def test_cross_window_uses_same_seed_set(tmp_path):
    """逐窗配对必须在同一 seed 集合上做, 否则不是配对。"""
    wins = ['35-39', '40-44', '45-49']
    d = _mk(n=20, windows=wins)
    p = tmp_path / 'z.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    res = bh.build(str(p))
    cw = res['cross_window']['mlp']
    assert len(cw) == 3
    for w in cw:
        assert w['n_paired'] == 20
        assert w['paired_seeds'] == list(range(20))
        assert 0 <= w['n_pos'] <= 20


def test_missing_seeds_reduce_n_and_do_not_get_padded(tmp_path):
    d = _mk(n=20, drop=(3, 7))
    p = tmp_path / 'w.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    res = bh.build(str(p))
    rep = res['comparisons']['mlp']
    assert rep['n_paired'] == 18
    assert 3 not in rep['paired_seeds'] and 7 not in rep['paired_seeds']
    assert res['models']['tasgl']['complete'] is False


def test_window_illicit_count_is_seed_independent(tmp_path):
    """子窗正例数是数据属性, 必须逐 seed 一致; 不一致就不得报数。"""
    wins = ['35-39', '40-44', '45-49']
    d = _mk(n=20, windows=wins)
    p = tmp_path / 'v.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    res = bh.build(str(p))
    for w in res['cross_window']['mlp']:
        # 造数时每个 seed 用同一个 n_labeled, 但没有 n_pos 字段
        assert w['n_illicit_in_window'] is None
        assert w['n_illicit_consistent'] is True


def test_power_design_exposes_effect_over_mde(tmp_path):
    """功效设计必须给出 观测效应/MDE 与所需种子数, 否则正文只能硬编码。"""
    d = _mk(n=20)
    p = tmp_path / 'u.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    res = bh.build(str(p))
    pd = res['power_design']['mlp']
    assert pd['effect_over_mde_at_n20'] is not None
    assert pd['effect_over_mde_at_n20'] > 0
    assert pd['mde_at_n20'] == pytest.approx(
        (2.0930 + 0.8416) * pd['sd_of_paired_diff'] / 20 ** 0.5, rel=1e-6)
    n_need = pd['seeds_needed_for_observed_effect']
    assert isinstance(n_need, int) and n_need >= 1


def test_required_seeds_formula():
    """目标效应量越小, 需要的种子越多; 且不低于 2。"""
    a = bh.required_seeds_for_mde(0.08, 0.05)
    b = bh.required_seeds_for_mde(0.08, 0.02)
    assert a is not None and b is not None
    assert b > a
    assert bh.required_seeds_for_mde(0.0, 0.05) is None


def test_tex_has_no_chinese_and_balanced_columns(tmp_path):
    d = _mk(n=20, windows=['35-39', '40-44', '45-49'])
    p = tmp_path / 't.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    res = bh.build(str(p))
    out = bh.tex(res, str(tmp_path))
    s = open(out, encoding='utf-8').read()
    assert not any('一' <= ch <= '鿿' for ch in s)
    assert '\\label{tab:bithe20}' in s
    assert '\\label{tab:bithe20win}' in s
    # 列数必须与 tabularx 的列声明一致, 表头与数据行也必须同宽
    spec = None
    for line in s.splitlines():
        m = re.match(r'\\begin\{tabularx\}\{[^}]*\}\{([lcrX]+)\}', line)
        if m:
            spec = len(m.group(1))
            continue
        if not (line.rstrip().endswith(r'\\') and '&' in line):
            continue
        cols = line.count('&') + 1
        assert spec is not None, 'row before any column spec'
        assert cols == spec, (cols, spec, line[:70])
    assert spec is not None
    # \times 之类只能出现在数学模式里, 否则正文编译时报错。
    # 按 $ 开关把文本切成 数学段 / 文本段, 只在文本段里搜。
    segs = re.split(r'(?<!\\)\$', s)
    assert len(segs) % 2 == 1, 'unbalanced $ in generated table'
    text_parts = [seg for i, seg in enumerate(segs) if i % 2 == 0]
    for seg in text_parts:
        assert '\\times' not in seg, 'text-mode \\times: ' + seg[:80]
