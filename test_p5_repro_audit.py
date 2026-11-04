"""test_p5_repro_audit.py — 跨执行可复现性审计的防自欺

守的底线:
  1. 只配对两边都有的种子, 缺失种子不得补零也不得截断凑数。
  2. 没有可配对种子时必须报错, 不许写出"不复用"这种无来源的结论。
  3. 归档分歧只数归档来源, 本次执行的日志不得计入。
  4. 比较精度必须显式给定: 日志印 4 位, 全精度 JSON 不能用更高精度去比,
     否则舍入差会被算成分歧。
"""
import json
import os
import re

import pytest

import p5_repro_audit as ra


def _m(rows, seeds):
    return {'schema': 'p4_second_dataset_v1', 'seeds': seeds,
            'models': {'tasgl': {'rows': rows}}}


def _rows(pairs, drop=()):
    return [({'seed': s, 'status': 'oom', 'f1': None} if s in drop else
             {'seed': s, 'status': 'ok', 'f1': f}) for s, f in pairs]


def test_only_shared_seeds_are_paired():
    cur = _m(_rows([(0, 0.50), (1, 0.60), (2, 0.70)]), [0, 1, 2])
    arc = _m(_rows([(0, 0.51), (1, 0.58)], drop=(2,)), [0, 1, 2])
    out = ra.cross_execution(cur, arc)
    assert out['tasgl']['seeds'] == [0, 1]
    assert out['tasgl']['n_seeds'] == 2
    assert out['tasgl']['mean_abs_diff'] == pytest.approx((0.01 + 0.02) / 2)
    assert out['tasgl']['max_abs_diff_seed'] == 1


def test_no_overlap_raises_instead_of_claiming(tmp_path):
    """完全没有可配对种子时, build 必须失败, 不许产出空结论。"""
    cur = _m(_rows([(0, 0.5)]), [0])
    arc = _m(_rows([(5, 0.5)]), [5])
    p1 = tmp_path / 'c.json'
    p2 = tmp_path / 'a.json'
    p1.write_text(json.dumps(cur), encoding='utf-8')
    p2.write_text(json.dumps(arc), encoding='utf-8')
    with pytest.raises(AssertionError):
        ra.build(str(p1), str(p2), str(tmp_path), str(tmp_path / 'none.json'))


def test_log_scan_skips_current_execution_log(tmp_path):
    """默认只扫归档日志; 本次执行日志(wave2.log)不得被算进归档分歧。"""
    (tmp_path / 'second_dataset.log').write_text(
        '[P4][2nd] tasgl seed 0: F1=0.5484 thr=0.82\n', encoding='utf-8')
    (tmp_path / 'second_dataset_tasgl.log').write_text(
        '[P4][2nd] tasgl seed 0: F1=0.5373 thr=0.82\n', encoding='utf-8')
    (tmp_path / 'wave2.log').write_text(
        '[P4][2nd] tasgl seed 0: F1=0.5383 thr=0.82\n', encoding='utf-8')
    lv = ra.log_disagreement(str(tmp_path))
    assert set(lv) == {'tasgl|0'}
    assert set(lv['tasgl|0']) == {'second_dataset.log',
                                  'second_dataset_tasgl.log'}


def test_rounding_difference_is_not_a_disagreement():
    """0.5254(日志) 与 0.525401(JSON) 是舍入差, 不是分歧。"""
    arc = _m(_rows([(0, 0.525401)]), [0])
    lv = {'tasgl|0': {'a.log': 0.5254}}
    out = ra.combine(lv, arc)
    assert len(out) == 1
    assert out[0]['n_sources'] == 2
    assert out[0]['n_distinct'] == 1
    assert out[0]['disagrees'] is False
    assert out[0]['compared_at_decimals'] == 4


def test_real_disagreement_is_reported():
    arc = _m(_rows([(0, 0.538342)]), [0])
    lv = {'tasgl|0': {'a.log': 0.5484, 'b.log': 0.5373}}
    out = ra.combine(lv, arc)
    assert out[0]['disagrees'] is True
    assert out[0]['n_distinct'] == 3
    assert out[0]['distinct_values'] == [0.5373, 0.5383, 0.5484]


def test_sci_format():
    assert ra._sci(1.7881393432617188e-07) == ('1.79', '-7')
    assert ra._sci(None) is None


def test_tex_shape(tmp_path):
    cur = _m(_rows([(0, 0.50), (1, 0.60)]), [0, 1])
    arc = _m(_rows([(0, 0.51), (1, 0.58)]), [0, 1])
    p1 = tmp_path / 'c.json'
    p2 = tmp_path / 'a.json'
    p1.write_text(json.dumps(cur), encoding='utf-8')
    p2.write_text(json.dumps(arc), encoding='utf-8')
    res = ra.build(str(p1), str(p2), str(tmp_path), str(tmp_path / 'no.json'))
    out = ra.tex(res, str(tmp_path))
    s = open(out, encoding='utf-8').read()
    assert not any('一' <= ch <= '鿿' for ch in s)
    assert '\\label{tab:repro}' in s
    segs = re.split(r'(?<!\\)\$', s)
    assert len(segs) % 2 == 1, 'unbalanced $'
    for i, seg in enumerate(segs):
        if i % 2 == 0:
            assert '\\times' not in seg, 'text-mode \\times'
    spec = None
    for line in s.splitlines():
        m = re.match(r'\\begin\{tabular[x]?\}\{[^}]*\}\{([lcrX]+)\}', line)
        if m:
            spec = len(m.group(1))
            continue
        if line.rstrip().endswith(r'\\') and '&' in line:
            assert line.count('&') + 1 == spec, (line[:60], spec)


def test_end_to_end_on_real_artifacts():
    """真实产物上的端到端校验: 归档十种子与 20 种子执行的 seeds 0-9。"""
    cur = 'results/p4/second_dataset_20seeds.json'
    arc = 'results/p4/second_dataset_confirm.json'
    if not (os.path.exists(cur) and os.path.exists(arc)):
        pytest.skip('artifacts not present')
    res = ra.build(cur, arc)
    t = res['cross_execution']['tasgl']
    assert t['n_seeds'] == 10
    assert t['mean_abs_diff'] == pytest.approx(0.0453, abs=5e-4)
    assert t['max_abs_diff'] == pytest.approx(0.1326, abs=5e-4)
    # 归档内部自相矛盾的种子必须真的存在, 且只来自归档来源
    dis = [d for d in res['log_disagreement'] if d['disagrees']]
    assert dis, 'archived disagreement should be detectable'
    for d in dis:
        assert all('wave2' not in k for k in d['per_source'])
