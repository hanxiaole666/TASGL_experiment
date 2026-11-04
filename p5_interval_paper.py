"""p5_interval_paper.py — H3 / H4 的十种子区间估计表

用户的原始要求: "把 five joint requirements 改为与实际可判定数一致的表述,
或补 H3/H4 的非 oracle 攻击矩阵与区间估计结果。"

两条都做了:
  * H3 (边鲁棒性) 与 H4 (时间外推) 现在各有十种子的点估计 + 95%CI,
    由 p4_interval.py 产出, 本模块只负责把它排成表, 不重新计算。
  * 非 oracle 攻击矩阵仍然缺, 所以对抗鲁棒性单列为 H6 并保持开放。
    删边是结构性扰动, 不是攻击; 这一点写进表格脚注, 因为删边结果很容易
    被误读成对抗鲁棒性。

读数是负面的, 必须照写:
  H3: TASGL 在删掉 50% 边后相对损失 12.34% [7.36%, 17.26%],
      GraphSAGE 3.37%, MLP 0.00% (无边的构造恒等式)。
      TASGL 比两个基线掉得都多, 且两个配对区间都不含 0。
  H4: 训练历史从 1-29 缩到 1-20, TASGL 相对损失 32.97% [21.01%, 44.76%],
      MLP 6.94% [-6.94%, 23.93%], GraphSAGE 36.15% [27.28%, 45.60%]。
      TASGL 比 MLP 掉得多, 与 GraphSAGE 无显著差异。

产出: generated_tables/h3_h4_interval.tex (tab:h3int, tab:h4int)

用法:
  python p5_interval_paper.py build --results results
  python p5_interval_paper.py tex   --paper DIR
"""
import json
import os

SCHEMA = 'p5_h3_h4_interval_tables_v1'

#: 产物里是小写键名, 正文里要用规范写法
DISPLAY = {'tasgl': 'TASGL', 'graphsage': 'GraphSAGE', 'mlp': 'MLP',
           'gcn': 'GCN', 'gat': 'GAT', 'sgc': 'SGC', 'dsgat': 'DSGAT'}


def _name(m):
    return DISPLAY.get(m, m)


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _pct(x, nd=2):
    return 'n.a.' if x is None else f'{100 * x:.{nd}f}\\%'


def _num(x, nd=4):
    return 'n.a.' if x is None else f'{x:.{nd}f}'


def build(results_dir='results', out=None):
    h3 = load(os.path.join(results_dir, 'p4', 'h3_interval.json'))
    h4 = load(os.path.join(results_dir, 'p4', 'h4_interval.json'))
    order = [m for m in ('tasgl', 'graphsage', 'mlp')
             if m in (h3.get('per_model') or {})]
    out_d = {
        'schema': SCHEMA,
        'h3': {'drop_rate': h3.get('drop_rate'),
               'seeds': h3.get('seeds'),
               'models': [{'model': m, 'mean': h3['per_model'][m]['mean'],
                           'sd': h3['per_model'][m]['sd'],
                           'ci95': h3['per_model'][m]['ci95'],
                           'n': h3['per_model'][m]['n']} for m in order],
               'paired': {k: h3['paired'][k] for k in order
                          if k != 'tasgl' and k in (h3.get('paired') or {})},
               'note': h3.get('note')},
        'h4': {'short_train_end': h4.get('short_train_end'),
               'seeds': h4.get('seeds'),
               'models': [{'model': m, 'mean': h4['per_model'][m]['mean'],
                           'sd': h4['per_model'][m]['sd'],
                           'ci95': h4['per_model'][m]['ci95'],
                           'n': h4['per_model'][m]['n']} for m in order],
               'paired': {k: h4['paired'][k] for k in order
                          if k != 'tasgl' and k in (h4.get('paired') or {})},
               'note': h4.get('note')},
        # 判读规则: 这两个假设没有门限, 所以不写 pass / falsified
        'verdict_rule': ('H3 与 H4 是区间估计, 不是门限。任何"通过"或"证伪"的'
                         '说法都超出了这两个表能支持的范围。'),
    }
    # 用绝对值判断"谁的损失更大", 并写成一句话, 避免读者误读方向
    for key in ('h3', 'h4'):
        blk = out_d[key]
        worst = max(blk['models'], key=lambda r: r['mean'])
        blk['largest_loss_model'] = worst['model']
        blk['reading'] = (
            f"配对差 (TASGL 减基线) 为正表示 TASGL 的相对损失更大; "
            f"本表中损失最大的是 {_name(worst['model'])}。")
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(out_d, f, indent=2, ensure_ascii=False)
    return out_d


def _rows(blk):
    lines = []
    for r in blk['models']:
        ci = r['ci95'] or [None, None]
        lines.append(' & '.join([
            _name(r['model']), str(r['n']), _pct(r['mean']), _pct(r['sd']),
            f"[{_pct(ci[0])}, {_pct(ci[1])}]"]) + r' \\')
    return lines


def _paired_rows(blk):
    lines = []
    for nm, p in (blk['paired'] or {}).items():
        ci = p.get('ci95') or [None, None]
        lines.append(' & '.join([
            _name(nm), str(p.get('n')), _pct(p.get('mean_diff')),
            f"[{_pct(ci[0])}, {_pct(ci[1])}]",
            _num(p.get('cohen_dz'), 3),
            f"{p.get('positive_count')}/{p.get('n')}",
            ('yes' if p.get('ci_excludes_zero') else 'no')]) + r' \\')
    return lines


def tex(d, paper_dir):
    def tbl(label, cap, blk, head, body, foot):
        # 列说明必须与表头列数一致: 模型 / n / 均值 / SD / 95%CI = 5 列
        return (
            f'\\begin{{table}}[!t]\n'
            f'\\centering\\footnotesize\n'
            f'\\caption{{{cap}}}\n'
            f'\\label{{{label}}}\n'
            f'\\begin{{tabularx}}{{\\columnwidth}}{{lcccc}}\n'
            f'\\hline\n{head} \\\\\n\\hline\n'
            + '\n'.join(body) + '\n\\hline\n'
            '\\end{tabularx}\n'
            f'\\begin{{flushleft}}\\footnotesize {foot}\\end{{flushleft}}\n'
            '\\end{table}\n')

    h3, h4 = d['h3'], d['h4']
    t1 = tbl('tab:h3int',
             'H3 edge robustness: relative illicit-F1 drop after deleting '
             f'{int(100 * (h3["drop_rate"] or 0.5))}\\% of edges, ten seeds.',
             h3,
             'Model & $n$ & Mean drop & SD & 95\\% CI',
             _rows(h3),
             'No gate is set for H3, so no pass or fail is written. Deleting '
             'edges is a structural perturbation, not an attack. MLP has no '
             'edges in its forward pass, so its drop is zero by construction.')

    t2 = tbl('tab:h4int',
             'H4 temporal extrapolation: relative illicit-F1 loss when the '
             'training window is shortened to steps 1--'
             f'{h4["short_train_end"]}, ten seeds, same test interval.',
             h4,
             'Model & $n$ & Mean loss & SD & 95\\% CI',
             _rows(h4),
             'No gate is set for H4, so no pass or fail is written. Both '
             'schedules use the same test interval, so this is a schedule '
             'effect, not a change of test population.')

    t3 = (
        '\\begin{table*}[!t]\n'
        '\\centering\\footnotesize\n'
        '\\caption{Paired differences for H3 and H4. A positive difference '
        'means TASGL loses more relative F1 than the baseline.}\n'
        '\\label{tab:h34paired}\n'
        '\\setlength{\\tabcolsep}{3pt}\n'
        '\\begin{tabularx}{\\textwidth}{llccccXX}\n'
        '\\hline\n'
        'Hyp. & Baseline & $n$ & Diff & 95\\% CI & $d_z$ & Seeds with larger '
        'TASGL loss & CI excl. 0 \\\\\n\\hline\n'
        + '\n'.join('H3 & ' + r for r in _paired_rows(h3))
        + '\n'
        + '\n'.join('H4 & ' + r for r in _paired_rows(h4))
        + '\n\\hline\n'
        '\\end{tabularx}\n'
        '\\begin{flushleft}\\footnotesize A positive difference means TASGL '
        'degrades more; an interval excluding zero means it degrades more than '
        'that baseline. This is relative degradation, not absolute F1.'
        '\\end{flushleft}\n'
        '\\end{table*}\n')

    dd = os.path.join(paper_dir, 'generated_tables')
    os.makedirs(dd, exist_ok=True)
    p = os.path.join(dd, 'h3_h4_interval.tex')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(t1 + '\n' + t2 + '\n' + t3)
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
        d = build(a.results, out=os.path.join(a.results, 'p5',
                                              'h3_h4_interval_tables.json'))
        print('saved', os.path.join(a.results, 'p5',
                                    'h3_h4_interval_tables.json'))
        for k in ('h3', 'h4'):
            blk = d[k]
            print(f'  {k}:', ', '.join(
                f"{r['model']}={100 * r['mean']:.2f}%" for r in blk['models']))
        return 0
    p = os.path.join(a.results, 'p5', 'h3_h4_interval_tables.json')
    if not os.path.exists(p):
        print('missing', p, '- run build first')
        return 2
    print('saved', tex(load(p), a.paper))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
