"""p5_protocol_map.py — 逐协议门限版本映射表

问题
----
论文里出现两种门限: 早期设计文档里的 V0 (F1>=0.73, latency<=8 ms,
params<=20000, 整体 ECE<=0.05) 和后登记的有依据集合 V2 (Table III)。
协议 3/4/5/6 全部在 V0 下判定, P4 主结果在 V2 下判定, BitHeNet 用的是
从 Elliptic 借来的 V2 数值。把"旧规则下的 PASS"和"新规则下的 FALSIFIED"
并排写在一起会让读者误以为二者可比。本模块把映射关系显式化。

做法
----
不手写映射, 而是从产物里读出门限, 并对读到的值做断言:
  * 协议 6 的 development_targets 直接写在 results/v4/select/summary.json
  * 协议 5 的产物没有记录门限, 改为哈希 paper_evidence.py (判定规则的
    代码来源) 并在表里注明"规则未随产物落盘"
  * V2 读 results/p4/thresholds/register.json 及其 lock_id
  * BitHeNet 读 results/p4/second_dataset*.json, 标记 borrowed
  * 第三数据集读 results/p5/dgraphfin_probe.json

产出
----
  results/p5/protocol_threshold_map.json
  generated_tables/protocol_map.tex   (tab:protomap)

用法
----
  python p5_protocol_map.py build [--results DIR] [--paper DIR]
  python p5_protocol_map.py tex    [--results DIR] [--paper DIR]
"""
import hashlib
import json
import os
import re

SCHEMA = 'p5_protocol_threshold_map_v1'

#: V0 (被撤回集合)。这些值来自早期设计文档, 审计判定为 arbitrary。
#: 来源: results/p4/threshold_justification.json 中 basis='arbitrary' 的条目。
V0_LABEL = 'withdrawn V0'
#: 注册集合的名字跟着锁定表里真实的版本号走, 不写死。v3 只给 H6 加了测量
#: 协议, 门限数值一个没动, 但表里既然写了 v3, 正文再叫它 V2 就会自相矛盾。
V2_LABEL = 'registered V2 (Table~\\ref{tab:thr_new})'
BORROWED_LABEL = 'borrowed V2'


def version_tag(version):
    return ((version or '').replace('p4_threshold_table_', '').upper()
            or 'V2')


def sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:24]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _v0_from_justification(results_dir):
    """从门限依据表读出被撤回的 V0 数值, 不手写。"""
    j = load_json(os.path.join(results_dir, 'p4',
                               'threshold_justification.json')) or {}
    hyp = j.get('hypotheses') or {}
    out = []
    for key, r in hyp.items():
        if not isinstance(r, dict):
            continue
        if r.get('original_basis') != 'arbitrary':
            continue
        out.append({'key': key, 'value': r.get('original_value'),
                    'verdict': r.get('verdict') or 'unjustified_withdrawn',
                    'why': (r.get('why_arbitrary') or '')[:160]})
    # 断言: 被撤回集合必须真的被读出来, 否则这张表就是在空口断言
    assert out, ('threshold_justification.json 里没有 basis=arbitrary 的条目; '
                 'V0 集合无法从产物证实')
    return out


def _v2_from_register(results_dir):
    reg = load_json(os.path.join(results_dir, 'p4', 'thresholds',
                                 'register.json')) or {}
    t = reg.get('table') or {}
    hyp = t.get('hypotheses') or {}
    vals = {}
    for h in ('H1', 'H2', 'H5'):
        gates = (hyp.get(h) or {}).get('gates') or []
        vals[h] = [{'metric': g.get('metric'), 'op': g.get('pass_op'),
                    'value': g.get('pass_value')} for g in gates]
    return {'lock_id': reg.get('lock_id'),
            'version': t.get('registration_version'),
            'hypotheses': vals,
            'h6_status': (hyp.get('H6') or {}).get('status'),
            'changelog': t.get('registration_changelog')}


def _protocol6(results_dir):
    s = load_json(os.path.join(results_dir, 'v4', 'select', 'summary.json'))
    if not s:
        return None
    dt = s.get('development_targets') or {}
    return {'thresholds': dt,
            'winner': s.get('winner'),
            'confirmation_authorized': s.get('confirmation_authorized'),
            'n_rows': len(s.get('rows') or [])}


def _protocol5(results_dir):
    p = os.path.join(results_dir, 'v3', 'confirm_91f1beb52cbad975',
                     'hypothesis_summary.json')
    s = load_json(p)
    if not s:
        return None
    return {'lock_id': s.get('lock_id'),
            'all_pass': s.get('all_pass'),
            'thresholds_recorded_in_artifact': False,
            'statuses': {k: (v or {}).get('status')
                         for k, v in (s.get('hypotheses') or {}).items()}}


def _bitheteronet(results_dir):
    out = {'runs': []}
    for name in ('second_dataset_20seeds.json', 'second_dataset_confirm.json',
                 'second_dataset.json'):
        d = load_json(os.path.join(results_dir, 'p4', name))
        if not d:
            continue
        out['runs'].append({
            'file': name,
            'n_seeds_requested': len(d.get('seeds') or []),
            'models': {k: v.get('n_complete', len(v.get('rows') or []))
                       for k, v in (d.get('models') or {}).items()}})
    return out


def _third(results_dir):
    d = load_json(os.path.join(results_dir, 'p5', 'dgraphfin_probe.json'))
    if not d:
        return {'status': 'not_run'}
    return {'status': d.get('status', 'unknown'),
            'temporal': d.get('what_this_can_test'),
            'rows': len(d.get('rows') or [])}


def build(results_dir='results', paper_dir=None):
    v0 = _v0_from_justification(results_dir)
    v2 = _v2_from_register(results_dir)
    tag = version_tag(v2.get('version'))
    v2_label = 'registered %s (Table~\\ref{tab:thr_new})' % tag
    borrowed_label = 'borrowed %s' % tag
    p5 = _protocol5(results_dir)
    p6 = _protocol6(results_dir)
    bh = _bitheteronet(results_dir)
    th = _third(results_dir)

    rows = []

    def row(proto, scope, tset, source, consequence, verdict_use):
        rows.append({'protocol': proto, 'scope': scope,
                     'threshold_set': tset, 'rule_source': source,
                     'consequence': consequence,
                     'verdict_reuse': verdict_use})

    row('3', 'Main result / ablation, 5 seeds', V0_LABEL,
        'early design document; audited as arbitrary '
        '(results/p4/threshold_justification.json)',
        'Kept for historical contrast only. V0 has no source, so no gate '
        'verdict from this protocol is reused.', 'none')

    row('4', 'Temporal / cross-chain diagnostics', V0_LABEL,
        'same source as protocol 3',
        'Descriptive only. Oracle edge injection means the relative drop '
        'cannot be read as adversarial robustness.', 'none')

    row('5', 'Locked exploratory confirmation (5 seeds)', V0_LABEL,
        ('thresholds were never written to the artifact; the decision code is '
         f'paper_evidence.py sha256={sha256_file("paper_evidence.py")}'),
        'The H2, H3 and H5 passes are passes under the withdrawn rule: H2 used '
        'a full-graph 8 ms budget and H5 used overall ECE. They are not '
        'comparable with the Table III verdicts.', 'none')

    if p6:
        dt = p6['thresholds']
        row('6', 'Development gate audit (28 runs)', V0_LABEL,
            ('results/v4/select/summary.json:development_targets '
             f"(f1_min={dt.get('f1_min')}, latency_ms_max="
             f"{dt.get('latency_ms_max')}, params_max={dt.get('params_max')})"),
            ('"No eligible candidate" is rule-dependent: the candidates failed '
             f"f1_min={dt.get('f1_min')}, a withdrawn threshold, and the V0 "
             'full-graph 8 ms latency semantics. It must not be read as '
             '"no candidate is eligible under the registered rule".'), 'none')

    row('P4 main', 'Elliptic, 10 seeds, equal budget', v2_label,
        (f"results/p4/thresholds/register.json lock_id={v2.get('lock_id')} "
         f"({v2.get('version')})"),
        'The only result set in this paper judged under sourced thresholds. '
        'H1, H2 and H5 are gates; H3 and H4 are interval estimates.',
        'primary')

    row('BitHeNet', 'Second dataset, independent', borrowed_label,
        (f"threshold values taken from the Elliptic anchor "
         f"({v2.get('lock_id')}); not re-derived on this dataset"),
        'Any pass must be written as "under a borrowed rule". It is not a gate '
        'confirmation on this dataset, because the non-inferiority anchor came '
        'from baselines on another dataset.', 'labelled_borrowed')

    row('DGraph-Fin', 'Third-dataset probe', 'n.a.',
        ('results/p5/dgraphfin_probe.json; edge_timestamp_only, '
         'no per-transaction time step'),
        'With no per-transaction time step the TASGL temporal component is '
        'degenerate and no temporal split exists, so H4 does not apply. Only '
        'the non-temporal components can be tested here.', 'none')

    out = {'schema': SCHEMA,
           'sets': {
               'V0': {'label': V0_LABEL, 'status': 'withdrawn',
                      'arbitrary_count': len(v0),
                      'items': v0},
               'V2': {'label': v2_label, 'status': 'registered',
                      'lock_id': v2.get('lock_id'),
                      'version': v2.get('version'),
                      'gates': v2.get('hypotheses'),
                      'H6_status': v2.get('h6_status'),
                      'changelog': v2.get('changelog')},
           },
           'protocol5': p5, 'protocol6': p6,
           'bitheteronet': bh, 'third_dataset': th,
           'rows': rows,
           'rule': ('同一篇论文里不得出现"旧规则 PASS"与"新规则 FALSIFIED"'
                    '并列而不标注门限版本的情况')}
    return out


def tex(m, paper_dir):
    """生成 generated_tables/protocol_map.tex (tab:protomap)。"""
    #: 保护 \ref{} / \texttt{} 之类的命令参数不被转义拆坏:
    #: 在 \ref{tab:thr\allowbreak\_new} 里插 \allowbreak 会触发
    #: "Missing \endcsname inserted"。
    _CMD = re.compile(r'(\\[a-zA-Z]+\{[^}]*\})')

    def esc(s):
        return (str(s).replace('\\', '\\allowbreak\\')
                .replace('_', '\\_').replace('%', '\\%')
                .replace('&', '\\&').replace('#', '\\#'))

    def brk(s):
        """转义文本, 但保留 \\ref{...} / \\texttt{...} 参数原样。"""
        out = []
        for part in _CMD.split(str(s)):
            if not part:
                continue
            if _CMD.fullmatch(part):
                out.append(part)
            else:
                out.append(' '.join(
                    esc(t).replace('\\_', '\\allowbreak\\_')
                    for t in part.split(' ')))
        return ''.join(out)

    body = []
    for r in m['rows']:
        body.append(' & '.join([
            esc(r['protocol']), brk(r['scope']), brk(r['threshold_set']),
            brk(r['consequence'])]) + r' \\')
    tbl = (
        '\\begin{table*}[!t]\n'
        '\\caption{Threshold-set version map. Each protocol is labelled with '
        'the threshold set it was actually judged under. A pass obtained under '
        'the withdrawn set is not comparable with a falsification obtained '
        'under the registered set, and the two are never placed side by side '
        'without this label.}\n'
        '\\label{tab:protomap}\n'
        '\\centering\n'
        '\\footnotesize\n'
        '\\begin{tabularx}{\\textwidth}{'
        '>{\\bfseries}l >{\\raggedright\\arraybackslash}p{0.16\\textwidth} '
        '>{\\raggedright\\arraybackslash}p{0.15\\textwidth} '
        '>{\\raggedright\\arraybackslash}X}\n'
        '\\toprule\n'
        'Protocol & Scope & Threshold set & Consequence for reading the result '
        '\\\\\n\\midrule\n'
        + '\n'.join(body) + '\n'
        '\\bottomrule\n'
        '\\end{tabularx}\n'
        '\\begin{flushleft}\\footnotesize\n'
        'V0 is the early design-document set: F1 $\\geq0.73$, full-graph '
        'latency $\\leq8$ ms, parameters $\\leq20{,}000$, overall ECE '
        '$\\leq0.05$. Every value in it was audited as having no source and is '
        f'withdrawn. {version_tag(m["sets"]["V2"].get("version"))} is the '
        'registered set in Table~\\ref{tab:thr_new}, '
        f'lock \\texttt{{{m["sets"]["V2"].get("lock_id")}}}. '
        f'The BitHeNet row uses {version_tag(m["sets"]["V2"].get("version"))} '
        'values derived on Elliptic; that is a '
        'borrowed rule, not a re-derived gate. '
        'The later registration version adds the H6 measurement protocol and '
        'changes no threshold value.\n'
        '\\end{flushleft}\n'
        '\\end{table*}\n')
    d = os.path.join(paper_dir, 'generated_tables')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'protocol_map.tex')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(tbl)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'tex'])
    ap.add_argument('--results', default='results')
    ap.add_argument('--paper', default=None)
    a = ap.parse_args()
    m = build(a.results)
    if a.cmd == 'build':
        os.makedirs(os.path.join(a.results, 'p5'), exist_ok=True)
        p = os.path.join(a.results, 'p5', 'protocol_threshold_map.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        print('saved', p)
        for r in m['rows']:
            print(f"  {r['protocol']:>10s} | {r['threshold_set']}")
        return 0
    paper = a.paper or os.path.join(
        'Doc', 'Kimi_Agent_CCF-A实验脚本-25', 'TASGL_TDSC投稿')
    print('saved', tex(m, paper))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
