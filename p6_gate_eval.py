"""p6_gate_eval.py — 把登记门限表变成可执行判决, 并给出阳性对照

为什么需要它
------------
论文里的门限此前只以"表的形式"存在: 表格写着 H1 的 F1 下界与参数上限,
但没有任何程序读它并给出判决, 判决是人工写在正文里的。这有两个后果。

第一, 无法回答"这个门控是不是只会拒绝"。一个只输出 falsified 的判决器
和一条真的门限在论文里长得一模一样, 读者无从区分。

第二, 判决无法随数据重算。换一次执行, 正文里的 pass/falsified 只能靠人
再读一遍, 这正是本项目一直在防的"声明与产物脱钩"。

本模块做两件事:
  1. evaluate(): 从 register.json 读出机器可读的 gate 定义, 对一组指标
     给出逐假设、逐子条件的判决。规则全部来自数据, 不在这里硬编码任何
     门限数值。
  2. positive control: 用同一把判决器跑一组"预先知道答案"的控制候选,
     其中包括一个应当全部通过的正例。若正例被拒, 说明判决器只会被动
     拒绝, 论文里任何 falsified 都失去意义。

判决标签一律取自登记表里的 on_fail / else, 不在这里改名。因此 H5 越过
绝对上限时得到的是 inconclusive_above_absolute_ceiling, 而不是 falsified;
只有当它同时劣于同骨干基线时才判 falsified。

用法
----
  python p6_gate_eval.py eval --metrics metrics.json
  python p6_gate_eval.py control
"""
import json
import operator
import os

SCHEMA = 'p6_gate_eval_v1'

OPS = {'>=': operator.ge, '<=': operator.le,
       '>': operator.gt, '<': operator.lt, '==': operator.eq}

#: 子条件合并规则。这是"每个门限都必须通过"的实现: 任一子条件判
#: falsified 则整体 falsified; 否则若有 inconclusive 则整体 inconclusive;
#: 否则 pass; 指标缺失一律 incomplete, 不猜。
SEVERITY = {'falsified': 2, 'inconclusive': 1, 'pass': 0}


def load_rules(path='results/p4/thresholds/register.json'):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    t = d.get('table') or d
    return t


def _sev(label, on_fail):
    """把登记里的标签映射到严重度; 未知的失败标签按 inconclusive 处理。"""
    lab = str(label or on_fail or 'inconclusive')
    low = lab.lower()
    if 'falsif' in low:
        return 2
    return 1


def eval_subcondition(g, metrics):
    m = g.get('metric')
    rhs_metric = g.get('compare_to_metric')
    if rhs_metric is not None:
        observed = metrics.get(m)
        rhs = metrics.get(rhs_metric)
    else:
        observed = metrics.get(m)
        rhs = g.get('pass_value')
    if observed is None or rhs is None:
        return {'metric': m, 'op': g.get('pass_op'),
                'threshold': g.get('pass_value'),
                'compare_to_metric': rhs_metric, 'rhs': rhs,
                'observed': observed, 'passed': None,
                'status': 'incomplete_missing_metric',
                'label_on_fail': g.get('on_fail')}
    op = OPS.get(g.get('pass_op'))
    if op is None:
        raise ValueError('未知的 pass_op: %r' % g.get('pass_op'))
    ok = bool(op(observed, rhs))
    return {'metric': m, 'op': g.get('pass_op'),
            'threshold': g.get('pass_value'),
            'compare_to_metric': rhs_metric, 'rhs': rhs,
            'observed': observed, 'passed': ok,
            'status': 'pass' if ok else str(g.get('on_fail')),
            'label_on_fail': g.get('on_fail')}


def eval_hypothesis(name, hyp, metrics):
    gates = hyp.get('gates') or []
    if not gates:
        return {'hypothesis': name, 'status': 'not_gated',
                'n_subconditions': 0, 'subconditions': [],
                'note': 'carries no pass/fail gate; reported, not judged'}
    subs = [eval_subcondition(g, metrics) for g in gates]
    if any(s['passed'] is None for s in subs):
        return {'hypothesis': name, 'status': 'incomplete',
                'n_subconditions': len(subs), 'subconditions': subs}
    if all(s['passed'] for s in subs):
        return {'hypothesis': name, 'status': 'pass',
                'n_subconditions': len(subs), 'subconditions': subs}
    sev = max(_sev(s['status'], s['label_on_fail'])
              for s in subs if not s['passed'])
    status = 'falsified' if sev == 2 else 'inconclusive'
    return {'hypothesis': name, 'status': status,
            'n_subconditions': len(subs), 'subconditions': subs}


def evaluate(metrics, rules=None, rules_path=None):
    rules = rules or load_rules(rules_path or
                                'results/p4/thresholds/register.json')
    hyps = rules.get('hypotheses') or {}
    out = {'schema': SCHEMA, 'hypotheses': {},
           'registration_version': rules.get('registration_version')}
    gated, decidable = [], []
    for name in sorted(hyps):
        r = eval_hypothesis(name, hyps[name], metrics)
        out['hypotheses'][name] = r
        if r['n_subconditions']:
            gated.append(name)
            decidable.append(name)
    # all_pass 只对"有门限的假设"成立; H3/H4/H6 无门限, 不参与。
    # 任一门限状态不是 pass 就不算通过, incomplete 也不算。
    out['gated_hypotheses'] = gated
    out['decidable_gates'] = decidable
    out['all_pass'] = bool(gated) and all(
        out['hypotheses'][h]['status'] == 'pass' for h in gated)
    out['all_pass_rule'] = ('all_pass is true only when every gated '
                            'hypothesis is pass; not_gated hypotheses do not '
                            'enter the conjunction; incomplete is not pass')
    return out


# ------------------------------------------------------------------
# 阳性对照
# ------------------------------------------------------------------

#: 控制候选。expected 是"这把判决器应当给出什么", 不是实验结果。
#: 数值刻意取在门限两侧, 便于读者自己验算。
def control_candidates(rules):
    """从登记表读出各门限, 在门限两侧构造控制候选。"""
    h = rules.get('hypotheses') or {}
    h1 = (h.get('H1') or {}).get('gates') or []
    h2 = (h.get('H2') or {}).get('gates') or []
    h5 = (h.get('H5') or {}).get('gates') or []

    def pick(gates, metric, default):
        for g in gates:
            if g.get('metric') == metric:
                return g.get('pass_value'), g.get('pass_op')
        return default, '<='

    f1_floor, _ = pick(h1, 'mean_f1', 0.4320)
    params_cap, _ = pick(h1, 'deploy_params', 20480)
    lat_cap, _ = pick(h2, 'mean_latency_ms', 142.857)
    ece_cap, _ = pick(h5, 'positive_class_ece', 0.05)

    # 基准: 全通过。取门限内的值(略有余量), 并让 H5 的第二子条件也成立。
    base = {'mean_f1': float(f1_floor) + 0.01,
            'deploy_params': int(float(params_cap)) - 100,
            'mean_latency_ms': float(lat_cap) * 0.5,
            'positive_class_ece': float(ece_cap) * 0.5,
            'baseline_positive_class_ece': float(ece_cap) * 0.9}
    ctrls = []

    def add(name, mutate, expect, why):
        m = dict(base)
        m.update(mutate)
        ctrls.append({'name': name, 'metrics': m, 'expected': expect,
                      'why': why})

    add('control_all_pass', {}, {'H1': 'pass', 'H2': 'pass', 'H5': 'pass'},
        'every sub-condition is satisfied; a gate that only rejects would '
        'fail this case')
    add('control_h1_f1_below_floor',
        {'mean_f1': float(f1_floor) - 0.01},
        {'H1': 'falsified', 'H2': 'pass', 'H5': 'pass'},
        'accuracy below the non-inferiority anchor')
    add('control_h1_params_over_budget',
        {'deploy_params': int(float(params_cap)) + 1},
        {'H1': 'falsified', 'H2': 'pass', 'H5': 'pass'},
        'size over the memory-derived budget')
    add('control_h2_too_slow',
        {'mean_latency_ms': float(lat_cap) * 1.5},
        {'H1': 'pass', 'H2': 'falsified', 'H5': 'pass'},
        'per-transaction latency over the throughput budget')
    add('control_h5_over_ceiling_better_than_baseline',
        {'positive_class_ece': float(ece_cap) * 1.5,
         'baseline_positive_class_ece': float(ece_cap) * 4.0},
        {'H1': 'pass', 'H2': 'pass', 'H5': 'inconclusive'},
        'over the absolute ceiling but still better than the same-backbone '
        'baseline: the ceiling alone cannot separate a bad model from a '
        'hard task, so the registered label is inconclusive rather than '
        'falsified')
    add('control_h5_worse_than_baseline',
        {'positive_class_ece': float(ece_cap) * 0.9,
         'baseline_positive_class_ece': float(ece_cap) * 0.5},
        {'H1': 'pass', 'H2': 'pass', 'H5': 'falsified'},
        'under the ceiling but worse than the same-backbone baseline')
    add('control_incomplete_missing_latency',
        {'mean_latency_ms': None},
        {'H1': 'pass', 'H2': 'incomplete', 'H5': 'pass'},
        'a missing measurement must degrade to incomplete, never to pass')
    return ctrls


def run_controls(rules=None, rules_path=None):
    rules = rules or load_rules(rules_path or
                                'results/p4/thresholds/register.json')
    rows = []
    for c in control_candidates(rules):
        got = evaluate(c['metrics'], rules)
        per = {h: got['hypotheses'][h]['status'] for h in c['expected']}
        ok = per == c['expected']
        rows.append({'name': c['name'], 'expected': c['expected'],
                     'observed': per, 'control_ok': ok,
                     'all_pass': got['all_pass'], 'why': c['why'],
                     'metrics': c['metrics']})
    return {'schema': 'p6_positive_control_v1',
            'registration_version': rules.get('registration_version'),
            'controls': rows,
            'n_controls': len(rows),
            'n_ok': sum(1 for r in rows if r['control_ok']),
            'verdict': ('positive control passes: the gate admits a '
                        'qualified candidate' if all(r['control_ok']
                                                     for r in rows)
                        else 'CONTROL FAILED: do not report any verdict')}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['eval', 'control'])
    ap.add_argument('--metrics', default=None)
    ap.add_argument('--rules', default='results/p4/thresholds/register.json')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.cmd == 'eval':
        with open(a.metrics, encoding='utf-8') as f:
            m = json.load(f)
        print(json.dumps(evaluate(m, rules_path=a.rules),
                         ensure_ascii=False, indent=2))
        return 0
    res = run_controls(rules_path=a.rules)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print('saved', a.out)
    print(res['verdict'], '(%d/%d)' % (res['n_ok'], res['n_controls']))
    for r in res['controls']:
        flag = 'ok ' if r['control_ok'] else 'BAD'
        print('  %s %-34s expected=%s observed=%s all_pass=%s'
              % (flag, r['name'], r['expected'], r['observed'],
                 r['all_pass']))
    return 0 if res['n_ok'] == res['n_controls'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
