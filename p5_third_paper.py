"""p5_third_paper.py — 第三数据集 (DGraph-Fin) 探针表

用户的原始要求: "第三数据集 + 实测 power/energy; 哪怕 50--100 例外部专家
标注, 也能把 FP/FN 队列从'计划'变成'结果'。"

第三数据集跑出来了, 但结论是否定的, 必须照写:
  * DGraph-Fin 只有边时间戳, 没有逐交易时间步 => 无法构造时间间隙,
    无法做严格 rolling 划分, TASGL 的时间分量在该数据上退化。
  * TASGL 在 6 GB 设备上放不下 (两个种子都是 OOM)。
  * 能跑的两个基线在 30 epoch 下 F1 只有 0.05 左右 (正类率 1.27%),
    与 Elliptic / BitHeNet 不可比。
所以它不能承担"第三个数据集上的门限确认"。外部专家标注仍未做, 保持开放。

产出: generated_tables/third_dataset.tex (tab:third)

用法:
  python p5_third_paper.py build --results results
  python p5_third_paper.py tex --paper DIR
"""
import json
import os

SCHEMA = 'p5_third_dataset_table_v1'

DISPLAY = {'tasgl': 'TASGL', 'graphsage': 'GraphSAGE', 'mlp': 'MLP'}


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def build(results_dir='results', out=None):
    d = load(os.path.join(results_dir, 'p5', 'dgraphfin_probe.json'))
    rows = []
    for name, blk in (d.get('models') or {}).items():
        for r in (blk.get('rows') or []):
            st = r.get('status')
            oom = isinstance(st, str) and st.startswith('oom')
            rows.append({
                'model': DISPLAY.get(name, name), 'seed': r.get('seed'),
                'status': 'oom' if oom else str(st),
                'f1': None if oom else r.get('f1'),
                'precision': None if oom else r.get('precision'),
                'recall': None if oom else r.get('recall'),
                'params': r.get('params'),
                'epochs': r.get('epochs'),
                'oom_alloc_gib': _gib(st) if oom else None})
    res = {'schema': SCHEMA,
           'dataset': d.get('dataset'),
           'n_nodes': d.get('n_nodes'), 'n_features': d.get('n_features'),
           'n_edges': d.get('n_edges'),
           'split': d.get('split'),
           'time_semantics': d.get('time_semantics'),
           'temporal_component': d.get('temporal_component'),
           'what_this_can_test': d.get('what_this_can_test'),
           'rows': rows,
           'positive_rate': _pos_rate(d),
           'carries_no_gate_result': True,
           'why': ('无逐交易时间步 => 时间分量退化且无法做时间划分; '
                   'TASGL 在该规模上 OOM。因此本表不承载任何门限判决。')}
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    return res


def _gib(status):
    import re
    m = re.search(r'Tried to allocate ([0-9.]+) GiB', status or '')
    return float(m.group(1)) if m else None


def _pos_rate(d):
    """探针里记录的正类率; 取不到就返回 None, 不猜。"""
    v = d.get('positive_rate')
    if v is not None:
        return v
    sa = d.get('split_audit') or {}
    per = (sa.get('per_split') or {}).get('train') or {}
    return per.get('positive_rate')


def _n(x, nd=4):
    return 'n.a.' if x is None else f'{x:.{nd}f}'


def tex(res, paper_dir):
    body = []
    for r in res['rows']:
        body.append(' & '.join([
            r['model'], str(r['seed']), r['status'],
            _n(r['f1']), _n(r['precision']), _n(r['recall']),
            'n.a.' if r['params'] is None else f"{r['params']:,}",
            'n.a.' if r['epochs'] is None else str(r['epochs'])]) + r' \\')

    def esc(s):
        # time_semantics 之类的值带下划线, 直接进正文会被当成下标
        return (str(s).replace('\\', '\\textbackslash{}')
                .replace('_', '\\_').replace('%', '\\%')
                .replace('&', '\\&').replace('#', '\\#')
                .replace('$', '\\$'))

    pr = res.get('positive_rate')
    gibs = [r.get('oom_alloc_gib') for r in res['rows']]
    gibs = [g for g in gibs if g]
    oom_note = (f'TASGL stopped with CUDA out of memory at a {max(gibs):.2f} '
                'GiB allocation on a 6 GiB device. '
                if gibs else '')
    cap = ('DGraph-Fin probe: %s nodes, %s features, %s edges, two seeds.'
           % (f"{res['n_nodes']:,}", res['n_features'],
              f"{res['n_edges']:,}"))
    tbl = (
        '\\begin{table}[!t]\n'
        '\\centering\\footnotesize\n'
        '\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{' + cap + '}\n'
        '\\label{tab:third}\n'
        '\\begin{tabularx}{\\columnwidth}{llclllcc}\n'
        '\\hline\n'
        'Model & Seed & Status & F1 & Precision & Recall & Params & Epochs '
        '\\\\\n\\hline\n'
        + '\n'.join(body) + '\n\\hline\n'
        '\\end{tabularx}\n'
        '\\begin{flushleft}\\footnotesize '
        f"Split: {esc(res['split'])}. "
        f"Time semantics: {esc(res['time_semantics'])}. "
        f"Temporal component: {esc(res['temporal_component'])}. "
        f"What this can test: {esc(res['what_this_can_test'])}. "
        + (f'Training positive rate {100 * pr:.2f}\\%. ' if pr else '')
        + oom_note
        + 'This table carries no gate verdict for any hypothesis.\n'
        '\\end{flushleft}\n'
        '\\end{table}\n')
    dd = os.path.join(paper_dir, 'generated_tables')
    os.makedirs(dd, exist_ok=True)
    p = os.path.join(dd, 'third_dataset.tex')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(tbl)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'tex'])
    ap.add_argument('--results', default='results')
    ap.add_argument('--paper', default=os.path.join(
        'Doc', 'Kimi_Agent_CCF-A实验脚本-25', 'TASGL_TDSC投稿'))
    a = ap.parse_args()
    if a.cmd == 'build':
        r = build(a.results, out=os.path.join(a.results, 'p5',
                                              'third_dataset_table.json'))
        print('saved; nodes=%d edges=%d rows=%d'
              % (r['n_nodes'], r['n_edges'], len(r['rows'])))
        return 0
    p = os.path.join(a.results, 'p5', 'third_dataset_table.json')
    if not os.path.exists(p):
        print('missing', p, '- run build first')
        return 2
    print('saved', tex(load(p), a.paper))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
