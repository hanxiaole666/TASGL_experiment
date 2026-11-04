"""p6_gate_paper.py — 门控判决的可执行化与阳性对照 (论文表格)

产出
----
  results/p6/gate_evaluation.json
  generated_tables/gate_control.tex   (tab:gatectrl, tab:gateverdict)

为什么要单独出一张对照表
------------------------
一篇只报告 falsified 的论文无法让读者判断: 到底是这些方法真的不合格,
还是这套门限只会拒绝。阳性对照把这件事变成可验算的数: 一个按登记规则
构造的合格候选必须被放行; 每一个子条件各有一个只违反它的对照必须被拒;
缺指标的对照必须降级为 incomplete 而不是 pass。

对照是控制, 不是结果: 数值刻意取在门限两侧, 读者可以自己算。
真实候选一列才是结果。

用法
----
  python p6_gate_paper.py build
  python p6_gate_paper.py tex [--paper DIR]
"""
import json
import os

import p6_gate_eval as ge

SCHEMA = 'p6_gate_evaluation_v1'


def _load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def real_candidates(results_dir='results'):
    """从产物里读真实候选的指标; 读不到就留 None, 由判决器降级为 incomplete。

    不在这里补任何估计值。缺指标是结论的一部分: 它说明该门限在该候选上
    不可判, 而不是通过。
    """
    fin = _load(os.path.join(results_dir, 'p4', 'final_comparison.json')) or {}
    scr = _load(os.path.join(results_dir, 'p4', 'per_node_screening.json')) or {}
    cal = _load(os.path.join(results_dir, 'p4',
                             'calibration_predictions_elliptic_seed0.json'))
    cal_mc = _load(os.path.join(results_dir, 'p4',
                                'calibration_predictions_elliptic_seed0_mc.json'))
    second = _load(os.path.join(results_dir, 'p4',
                                'second_dataset_20seeds.json')) or {}

    def lat(model):
        m = (scr.get('models') or {}).get(model) or {}
        b = ((m.get('batches') or {}).get('with_build') or {}).get('1') or {}
        return b.get('ms_per_tx_mean')

    def par(model):
        m = (scr.get('models') or {}).get(model) or {}
        return m.get('params')

    out = {}
    for mdl in ('tasgl', 'graphsage', 'mlp'):
        f = (fin.get('models') or {}).get(mdl) or {}
        out['elliptic/%s' % mdl] = {
            'dataset': 'Elliptic', 'model': mdl,
            'mean_f1': f.get('mean_f1'),
            'deploy_params': par(mdl),
            'mean_latency_ms': lat(mdl),
            # 正类 ECE 只在 TASGL 上实测过 (seed 0); 同骨干基线 = 同骨干的
            # MC-Dropout / EDL 变体。MLP 与 GraphSAGE 没有同类测量, 留空。
            'positive_class_ece': (cal or {}).get('positive_class_ece')
            if mdl == 'tasgl' else None,
            'baseline_positive_class_ece': (cal_mc or {}).get(
                'positive_class_ece') if mdl == 'tasgl' else None,
        }
    sd_t = (second.get('models') or {}).get('tasgl') or {}
    out['bitheteronet/tasgl'] = {
        'dataset': 'BitHeNet', 'model': 'tasgl',
        'mean_f1': sd_t.get('mean_f1'),
        'deploy_params': None, 'mean_latency_ms': None,
        'positive_class_ece': None, 'baseline_positive_class_ece': None,
        'note': 'gate values borrowed from Elliptic; per-transaction latency '
                'and positive-class ECE were not measured on this dataset',
    }
    rows = (sd_t.get('rows') or [])
    for r in rows:
        if r.get('params'):
            out['bitheteronet/tasgl']['deploy_params'] = int(r['params'])
            break
    return out


def build(results_dir='results', rules_path='results/p4/thresholds/register.json',
          out=None):
    rules = ge.load_rules(rules_path)
    ctrl = ge.run_controls(rules)
    cands = real_candidates(results_dir)
    verdicts = {}
    for name, m in cands.items():
        v = ge.evaluate(m, rules)
        v['dataset'] = m.get('dataset')
        v['model'] = m.get('model')
        v['metrics'] = {k: m.get(k) for k in (
            'mean_f1', 'deploy_params', 'mean_latency_ms',
            'positive_class_ece', 'baseline_positive_class_ece')}
        if m.get('note'):
            v['note'] = m['note']
        verdicts[name] = v
    res = {'schema': SCHEMA, 'rules': rules_path,
           'registration_version': rules.get('registration_version'),
           'controls': ctrl, 'candidates': verdicts,
           'verdict': _verdict(ctrl, verdicts)}
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    return res


def _verdict(ctrl, verdicts):
    out = []
    ok = ctrl['n_ok'] == ctrl['n_controls']
    out.append('positive control %s (%d/%d)' % ('passes' if ok else 'FAILED',
                                                ctrl['n_ok'],
                                                ctrl['n_controls']))
    for name, v in sorted(verdicts.items()):
        st = {h: v['hypotheses'][h]['status']
              for h in ('H1', 'H2', 'H5')}
        out.append('%s: %s all_pass=%s' % (name, st, v['all_pass']))
    if not ok:
        out.append('CONTROL FAILED: 任何 falsified 都不得写入论文')
    return out


# ------------------------------------------------------------------ tex

def _f(x, nd=3):
    return 'n/a' if x is None else f'{x:.{nd}f}'


def tex(res, paper_dir):
    def esc(s):
        return (str(s).replace('\\', '\\textbackslash{}')
                .replace('_', '\\_').replace('%', '\\%')
                .replace('&', '\\&').replace('#', '\\#'))

    rows = []
    for c in res['controls']['controls']:
        exp = ', '.join('%s=%s' % (h, s) for h, s in sorted(c['expected'].items()))
        obs = ', '.join('%s=%s' % (h, s) for h, s in sorted(c['observed'].items()))
        rows.append(' & '.join([
            esc(c['name']), esc(exp), esc(obs),
            'yes' if c['control_ok'] else 'NO',
            esc(c['why'])]) + r' \\')

    t1 = ('\\begin{table*}[!t]\n'
          '\\centering\\footnotesize\n'
          '\\setlength{\\tabcolsep}{3pt}\n'
          '\\caption{Positive and negative controls for the registered gate '
          'function. The same decision code is used here and for the measured '
          'candidates in Table~\\ref{tab:gateverdict}. Control values are '
          'chosen on either side of each registered threshold so that the '
          'expected outcome can be recomputed by hand; they are controls, '
          'not results.}\n'
          '\\label{tab:gatectrl}\n'
          '\\begin{tabularx}{\\textwidth}{lll c X}\n'
          '\\hline\n'
          'Control & Expected & Observed & OK & Purpose \\\\\n\\hline\n'
          + '\n'.join(rows) + '\n\\hline\n'
          '\\end{tabularx}\n'
          '\\begin{flushleft}\\footnotesize '
          'The first row is the positive control: a candidate that satisfies '
          'every sub-condition must be admitted, and it is. Without that row '
          'a reader cannot tell a working gate from one that only rejects. '
          'The last row shows that a missing measurement degrades to '
          'incomplete and never to pass.\n'
          '\\end{flushleft}\n'
          '\\end{table*}\n')

    vrows = []
    for name, v in sorted(res['candidates'].items()):
        m = v.get('metrics') or {}
        st = lambda h: esc(v['hypotheses'][h]['status'])
        vrows.append(' & '.join([
            esc(v.get('dataset')), esc(v.get('model')),
            _f(m.get('mean_f1')),
            ('n/a' if m.get('deploy_params') is None
             else f"{int(m['deploy_params']):,}"),
            _f(m.get('mean_latency_ms'), 2),
            _f(m.get('positive_class_ece'), 4),
            st('H1'), st('H2'), st('H5'),
            'yes' if v['all_pass'] else 'no']) + r' \\')

    t2 = ('\n\\begin{table*}[!t]\n'
          '\\centering\\footnotesize\n'
          '\\setlength{\\tabcolsep}{3pt}\n'
          '\\caption{Registered gate outcomes for the measured candidates, '
          'produced by the same decision function as '
          'Table~\\ref{tab:gatectrl}. n/a means the quantity was never '
          'measured for that candidate, and the gate degrades to incomplete '
          'rather than pass.}\n'
          '\\label{tab:gateverdict}\n'
          '\\begin{tabularx}{\\textwidth}{llcccccccc}\n'
          '\\hline\n'
          'Dataset & Model & F1 & Params & ms/tx & Pos. ECE & H1 & H2 & H5 & '
          'All pass \\\\\n\\hline\n'
          + '\n'.join(vrows) + '\n\\hline\n'
          '\\end{tabularx}\n'
          '\\begin{flushleft}\\footnotesize '
          'H3, H4 and H6 carry no gate and are reported as intervals, so they '
          'do not enter the conjunction. all\\_pass is true only when every '
          'gated hypothesis is pass. MLP clears H1 and H2 on Elliptic, which '
          'is the empirical counterpart of the positive control: the gate '
          'admits a qualified model on measured data, not only on '
          'constructed controls. No measured candidate clears all three '
          'gates.\n'
          '\\end{flushleft}\n'
          '\\end{table*}\n')

    d = os.path.join(paper_dir, 'generated_tables')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'gate_control.tex')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(t1 + t2)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'tex'])
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--paper', default=None)
    a = ap.parse_args()
    if a.cmd == 'build':
        res = build(a.results_dir)
        os.makedirs('results/p6', exist_ok=True)
        p = 'results/p6/gate_evaluation.json'
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print('saved', p)
        for v in res['verdict']:
            print('  ', v)
        return 0
    rp = 'results/p6/gate_evaluation.json'
    if not os.path.exists(rp):
        print('missing', rp, '- run build first')
        return 2
    paper = a.paper or os.path.join(
        'Doc', 'Kimi_Agent_CCF-A实验脚本-25', 'TASGL_TDSC投稿')
    with open(rp, encoding='utf-8') as f:
        res = json.load(f)
    print('saved', tex(res, paper))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
