"""test_p5_protocol_map.py — 门限版本映射表的完整性与防伪造

用户的原始要求: "加门限版本映射表: 逐 protocol 标注所用阈值集
(withdrawn / Table III / dev-proxy), 重写摘要中 passes calibration 一句,
避免旧规则 pass 与新规则 falsified 并列。"

本测试守三条底线:
  1. 映射表里的每个门限都能在产物里找到出处, 空目录不产出"看起来完整"的表。
  2. 每个协议都必须被标注门限集, 不允许出现未标注的行。
  3. 生成的 tex 必须是纯英文 (论文正文为英文), 且不含中文字符。
"""
import json
import os
import re
import shutil
import tempfile

import pytest

import p5_protocol_map as pm


def test_build_from_real_results():
    m = pm.build('results')
    assert m['schema'] == pm.SCHEMA
    # 每个协议都要有门限集标注
    for r in m['rows']:
        assert r['threshold_set'], r
        assert r['consequence'], r
        assert r['rule_source'], r
    names = [r['protocol'] for r in m['rows']]
    for need in ('3', '4', '5', '6'):
        assert need in names, f'协议 {need} 没有出现在映射表里'


def test_v0_items_come_from_artifact():
    """V0 (被撤回集合) 的每一项都必须来自门限依据表, 不能手写。"""
    m = pm.build('results')
    v0 = m['sets']['V0']['items']
    assert v0, 'V0 为空: 说明没有从产物里读出被撤回的门限'
    keys = {it['key'] for it in v0}
    assert 'H1_f1' in keys
    f1 = [it for it in v0 if it['key'] == 'H1_f1'][0]
    assert f1['value'] == 0.73
    assert 'withdrawn' in f1['verdict']


def test_v2_lock_is_read_not_hardcoded():
    m = pm.build('results')
    reg = pm.load_json(os.path.join('results', 'p4', 'thresholds',
                                    'register.json'))
    assert m['sets']['V2']['lock_id'] == reg['lock_id']
    # 门限数值必须与登记表一致
    hyp = reg['table']['hypotheses']
    h1 = [g for g in hyp['H1']['gates'] if g['metric'] == 'mean_f1'][0]
    assert h1['pass_value'] == pytest.approx(0.431972, abs=1e-6)


def test_h6_separated_from_h4():
    """H4 是时间外推, 攻击项必须单列为 H6 且不参与 all_pass。"""
    reg = pm.load_json(os.path.join('results', 'p4', 'thresholds',
                                    'register.json'))
    hyp = reg['table']['hypotheses']
    assert 'temporal' in json.dumps(hyp['H4'], ensure_ascii=False).lower() \
        or '区间估计' in hyp['H4']['statement']
    assert hyp['H4']['gates'] == []
    assert 'H6' in hyp
    # H6 已从"不可测的开放项"升级为"预登记测量协议 + 区间报告", 但无论
    # 哪个状态都不能有 gate, 也不能进入 all_pass。
    assert hyp['H6']['status'] in ('not_testable', 'measured_no_gate')
    assert hyp['H6']['gates'] == []
    assert 'H6' in reg['table']['all_pass_rule']
    if hyp['H6']['status'] == 'measured_no_gate':
        proto = hyp['H6']['measurement_protocol']
        assert proto['oracle_labels'] is False
        assert proto['paired_unit'] == 'model seed'
        assert proto['declared_attacks']


def test_no_fabrication_on_empty_dir():
    """空 results 目录不许产出"看起来完整"的映射表。"""
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, 'p4'))
        with pytest.raises(AssertionError):
            pm.build(tmp)
    finally:
        shutil.rmtree(tmp)


def test_tex_is_english_and_wellformed(tmp_path):
    m = pm.build('results')
    p = pm.tex(m, str(tmp_path))
    s = open(p, encoding='utf-8').read()
    # 正文为英文: 不允许出现 CJK
    assert not re.search(r'[一-鿿]', s), '生成的表格里出现了中文'
    assert s.startswith('\\begin{table*}')
    assert '\\label{tab:protomap}' in s
    assert s.rstrip().endswith('\\end{table*}')
    # 列数一致
    ncol = 4
    for line in s.splitlines():
        if line.endswith(r'\\') and '&' in line:
            assert line.count('&') == ncol - 1, line[:60]
    # 门限集标注必须真的出现在表体里
    assert 'withdrawn V0' in s
    # 标签跟着锁定表里真实的注册版本走, 不写死 V2/V3
    tag = pm.version_tag(m['sets']['V2'].get('version'))
    assert 'borrowed %s' % tag in s
    assert 'registered %s' % tag in s


def test_rule_text_present(tmp_path):
    m = pm.build('results')
    assert 'withdrawn' in m['rule'] or 'V0' in json.dumps(m['sets'])
