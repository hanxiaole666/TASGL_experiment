"""test_p6_gate_eval.py — 门控判决器与阳性对照

守的底线:
  1. 阳性对照必须真的被放行。一个只会拒绝的判决器会让论文里所有
     falsified 都失去意义, 而这在表格上和真门限长得一模一样。
  2. 每个子条件都要有只违反它的阴性对照, 否则无法定位判决来自哪一条。
  3. 缺指标必须降级为 incomplete, 绝不能变成 pass。
  4. 门限数值必须来自登记表, 不得在判决器里硬编码。
"""
import json
import os
import re

import pytest

import p6_gate_eval as ge


def _rules(f1=0.40, params=1000, lat=100.0, ece=0.02):
    return {'registration_version': 'test_rules_v1', 'hypotheses': {
        'H1': {'basis': 'derived', 'gates': [
            {'metric': 'mean_f1', 'pass_op': '>=', 'pass_value': f1,
             'on_fail': 'falsified'},
            {'metric': 'deploy_params', 'pass_op': '<=',
             'pass_value': params, 'on_fail': 'falsified'}]},
        'H2': {'basis': 'operational', 'gates': [
            {'metric': 'mean_latency_ms', 'pass_op': '<=',
             'pass_value': lat, 'on_fail': 'falsified'}]},
        'H3': {'basis': 'derived', 'gates': []},
        'H4': {'basis': 'derived', 'gates': []},
        'H5': {'basis': 'literature', 'gates': [
            {'metric': 'positive_class_ece', 'pass_op': '<=',
             'pass_value': ece,
             'on_fail': 'inconclusive_above_absolute_ceiling'},
            {'metric': 'positive_class_ece', 'pass_op': '<',
             'compare_to_metric': 'baseline_positive_class_ece',
             'on_fail': 'falsified'}]},
        'H6': {'basis': 'operational', 'gates': [], 'status': 'not_testable'},
    }}


def test_positive_control_is_admitted():
    """合格候选必须被放行, 否则整套门限只是一条拒绝规则。"""
    r = _rules()
    res = ge.run_controls(r)
    assert res['n_ok'] == res['n_controls'], [
        (c['name'], c['expected'], c['observed']) for c in res['controls']
        if not c['control_ok']]
    first = [c for c in res['controls'] if c['name'] == 'control_all_pass'][0]
    assert first['all_pass'] is True
    assert first['observed'] == {'H1': 'pass', 'H2': 'pass', 'H5': 'pass'}


def test_thresholds_follow_the_register_not_literals():
    """换一套登记数值, 判决必须跟着变; 证明没有硬编码。"""
    big = ge.control_candidates(_rules(f1=0.90, params=10, lat=1.0, ece=0.5))
    small = ge.control_candidates(_rules(f1=0.10, params=10 ** 6,
                                         lat=10 ** 5, ece=0.001))
    assert big != small
    # 严门限下, 宽松门限里的"全通过"候选必须被拒
    strict = _rules(f1=0.90, params=10, lat=1.0, ece=0.5)
    loose = ge.control_candidates(_rules(f1=0.10, params=10 ** 6,
                                         lat=10 ** 5, ece=0.001))
    m = [c for c in loose if c['name'] == 'control_all_pass'][0]['metrics']
    got = ge.evaluate(m, strict)
    assert got['all_pass'] is False
    assert got['hypotheses']['H1']['status'] == 'falsified'


def test_each_subcondition_is_located_by_its_own_control():
    res = ge.run_controls(_rules())
    names = {c['name']: c for c in res['controls']}
    assert names['control_h1_f1_below_floor']['observed']['H1'] == 'falsified'
    assert names['control_h1_params_over_budget']['observed']['H1'] == 'falsified'
    assert names['control_h2_too_slow']['observed']['H2'] == 'falsified'
    # 只违反子条件, 不得波及其它门限
    for n in ('control_h1_f1_below_floor', 'control_h1_params_over_budget'):
        assert names[n]['observed']['H2'] == 'pass'
        assert names[n]['observed']['H5'] == 'pass'
    assert names['control_h2_too_slow']['observed']['H1'] == 'pass'


def test_missing_metric_is_incomplete_never_pass():
    res = ge.run_controls(_rules())
    c = [x for x in res['controls']
         if x['name'] == 'control_incomplete_missing_latency'][0]
    assert c['observed']['H2'] == 'incomplete'
    assert c['all_pass'] is False
    # 逐子条件的状态也必须是 incomplete, 不能是 pass
    got = ge.evaluate(c['metrics'], _rules())
    subs = got['hypotheses']['H2']['subconditions']
    assert all(s['passed'] is None for s in subs)


def test_over_ceiling_but_better_than_baseline_is_inconclusive():
    """越过绝对上限但优于同骨干基线: 登记标签是 inconclusive, 不是 falsified。"""
    r = _rules(ece=0.02)
    m = {'mean_f1': 0.5, 'deploy_params': 100, 'mean_latency_ms': 10.0,
         'positive_class_ece': 0.05, 'baseline_positive_class_ece': 0.20}
    got = ge.evaluate(m, r)
    assert got['hypotheses']['H5']['status'] == 'inconclusive'


def test_under_ceiling_but_worse_than_baseline_is_falsified():
    r = _rules(ece=0.02)
    m = {'mean_f1': 0.5, 'deploy_params': 100, 'mean_latency_ms': 10.0,
         'positive_class_ece': 0.01, 'baseline_positive_class_ece': 0.005}
    got = ge.evaluate(m, r)
    assert got['hypotheses']['H5']['status'] == 'falsified'


def test_ungated_hypotheses_do_not_enter_all_pass():
    r = _rules()
    got = ge.evaluate(ge.control_candidates(r)[0]['metrics'], r)
    assert got['gated_hypotheses'] == ['H1', 'H2', 'H5']
    assert 'H6' not in got['gated_hypotheses']
    assert got['all_pass'] is True


def test_real_candidates_from_artifacts():
    """端到端: 真实产物上的判决; TASGL 必须在 H1 上被证伪。"""
    import p6_gate_paper as gp
    if not os.path.exists('results/p4/final_comparison.json'):
        pytest.skip('artifacts not present')
    res = gp.build('results')
    assert res['controls']['n_ok'] == res['controls']['n_controls']
    t = res['candidates']['elliptic/tasgl']
    assert t['hypotheses']['H1']['status'] == 'falsified'
    assert t['hypotheses']['H2']['status'] == 'pass'
    assert t['all_pass'] is False
    # 实证对照: MLP 在 H1/H2 上真的通过
    m = res['candidates']['elliptic/mlp']
    assert m['hypotheses']['H1']['status'] == 'pass'
    assert m['hypotheses']['H2']['status'] == 'pass'
    assert m['hypotheses']['H5']['status'] == 'incomplete'


def test_tex_shape(tmp_path):
    import p6_gate_paper as gp
    res = gp.build('results') if os.path.exists(
        'results/p4/final_comparison.json') else None
    if res is None:
        pytest.skip('artifacts not present')
    out = gp.tex(res, str(tmp_path))
    s = open(out, encoding='utf-8').read()
    assert not any('一' <= ch <= '鿿' for ch in s)
    assert '\\label{tab:gatectrl}' in s and '\\label{tab:gateverdict}' in s
    segs = re.split(r'(?<!\\)\$', s)
    assert len(segs) % 2 == 1, 'unbalanced $'
    for i, seg in enumerate(segs):
        if i % 2 == 0:
            assert '\\times' not in seg
    spec = None
    for line in s.splitlines():
        m = re.match(r'\\begin\{tabular[x]?\}\{[^}]*\}\{([lcrX ]+)\}', line)
        if m:
            spec = len([c for c in m.group(1) if c != ' '])
            continue
        if line.rstrip().endswith(r'\\') and '&' in line:
            assert line.count('&') + 1 == spec, (line[:70], spec)
