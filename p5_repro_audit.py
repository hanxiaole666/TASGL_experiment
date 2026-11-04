"""p5_repro_audit.py — 跨执行可复现性审计 (把"不可复现"变成可复算的数)

为什么需要它
------------
论文里有一句话: "本次执行重跑的 seeds 0-9 不复用归档的十种子产物, TASGL
平均 |ΔF1| = 0.0453, 最大 0.1326"。这句话里的三个数此前是手算的, 手算的
数不能进论文: 读者无法复算, 我们自己也无法在换一次执行后重新生成。本模块
把它们固定成"读两份结果 JSON, 逐 seed 配对相减"的产物。

它同时回答两个不同层次的问题, 这两个问题不能混为一谈:
  * 同一次执行内, 同一种子训练两次是否一致 —— 由 p5_determinism.py 回答;
  * 归档产物(由多次部分运行拼成)与本次完整执行是否一致 —— 由本模块回答。
前者是软件栈的性质, 后者是产物管线的性质。论文里的结论是: 前者稳定到
fp32, 后者不成立, 所以缺陷在管线而不在模型。

产出
----
  results/p5/repro_audit.json
  generated_tables/repro_audit.tex   (tab:repro, tab:reprologs)

用法
----
  python p5_repro_audit.py build [--current ...] [--archive ...]
  python p5_repro_audit.py tex    [--paper DIR]
"""
import json
import os
import re
import glob

SCHEMA = 'p5_repro_audit_v1'

#: 归档日志里逐 seed 的一行形如:
#:   [P4][2nd] tasgl seed 0: F1=0.5484 thr=0.82
LOG_RE = re.compile(r'\[P4\]\[2nd\]\s+(\w+)\s+seed\s+(\d+):\s*F1=([0-9.]+)')


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _f1map(d, model):
    """取 {seed: f1}; 状态不是 ok 或 f1 缺失的种子一律排除, 不补零。"""
    out = {}
    for r in ((d.get('models') or {}).get(model) or {}).get('rows') or []:
        if r.get('status', 'ok') != 'ok' or r.get('f1') is None:
            continue
        out[int(r['seed'])] = float(r['f1'])
    return out


def cross_execution(current, archive, models=None):
    """归档产物 vs 本次执行: 逐 seed 配对相减, 只取两边都有的种子。"""
    models = models or sorted(set(current.get('models') or {})
                              & set(archive.get('models') or {}))
    out = {}
    for m in models:
        a = _f1map(archive, m)
        b = _f1map(current, m)
        both = sorted(set(a) & set(b))
        if len(both) < 2:
            continue
        diffs = {s: abs(b[s] - a[s]) for s in both}
        vals = [diffs[s] for s in both]
        out[m] = {'n_seeds': len(both), 'seeds': both,
                  'per_seed_abs_diff': diffs,
                  'mean_abs_diff': sum(vals) / len(vals),
                  'max_abs_diff': max(vals),
                  'max_abs_diff_seed': max(both, key=lambda s: diffs[s]),
                  'archive_mean_f1': sum(a[s] for s in both) / len(both),
                  'current_mean_f1': sum(b[s] for s in both) / len(both)}
    return out


def log_disagreement(log_dir, pattern='second_dataset*.log'):
    """同一 (model, seed) 在归档日志与归档 JSON 里出现过几个不同的值。

    只在"至少两个来源且值不同"时才报, 单一来源不算分歧。

    默认只扫归档日志, 不包含本次执行的日志: 本次执行属于"跨执行对比",
    不属于"归档内部自相矛盾"。把两者混在一起会把两个不同的问题合成一个,
    也会让分歧数被凑大。
    """
    seen = {}
    for p in sorted(glob.glob(os.path.join(log_dir, pattern))):
        try:
            with open(p, encoding='utf-8') as f:
                txt = f.read()
        except OSError:
            continue
        for m in LOG_RE.finditer(txt):
            key = (m.group(1), int(m.group(2)))
            seen.setdefault(key, {}).setdefault(os.path.basename(p),
                                                float(m.group(3)))
    return {f'{k[0]}|{k[1]}': v for k, v in seen.items()}


def combine(log_vals, archive, models=('tasgl',), nd=4):
    """把日志里的值与归档 JSON 的值放在一起数分歧。

    nd 是比较精度。日志只印 4 位小数, 归档 JSON 存全精度, 直接比会把
    "0.5254 与 0.525401" 算成两个不同的值, 那是舍入差不是分歧。因此统一
    在 nd 位上比, 并在表注里写明精度。
    """
    out = []
    for m in models:
        a = _f1map(archive, m)
        for key, src in sorted(log_vals.items()):
            mm, _, seed_s = str(key).partition('|')
            if mm != m:
                continue
            try:
                seed = int(seed_s)
            except ValueError:
                continue
            vals = dict(src)
            if seed in a:
                vals['(archived json)'] = a[seed]
            if len(vals) < 2:
                continue
            uniq = sorted(set(round(v, nd) for v in vals.values()))
            out.append({'model': m, 'seed': seed,
                        'per_source': {k: round(v, nd)
                                       for k, v in sorted(vals.items())},
                        'n_sources': len(vals), 'n_distinct': len(uniq),
                        'distinct_values': uniq,
                        'spread': (uniq[-1] - uniq[0]) if uniq else None,
                        'compared_at_decimals': nd,
                        'disagrees': len(uniq) > 1})
    return out


def build(current_path='results/p4/second_dataset_20seeds.json',
          archive_path='results/p4/second_dataset_confirm.json',
          log_dir='results/p4',
          determinism_path='results/p5/determinism.json'):
    cur, arc = load(current_path), load(archive_path)
    ce = cross_execution(cur, arc)
    lv = log_disagreement(log_dir)
    dis = combine(lv, arc)
    det = None
    if os.path.exists(determinism_path):
        det = load(determinism_path)
    res = {'schema': SCHEMA,
           'current': current_path, 'archive': archive_path,
           'current_seeds': (cur.get('seeds') or []),
           'archive_seeds': (arc.get('seeds') or []),
           'cross_execution': ce,
           'log_disagreement': dis,
           'determinism': det,
           'verdict': _verdict(ce, det)}
    assert ce, '归档与本次执行没有可配对的种子, 不得写成"不复用"'
    return res


def _verdict(ce, det):
    """结论分层: 执行内是否稳定, 跨执行是否稳定。两者不可互换。"""
    out = []
    for m, d in sorted(ce.items()):
        out.append(f'{m}: 跨执行 seeds 0-9 平均 |ΔF1| {d["mean_abs_diff"]:.4f}, '
                   f'最大 {d["max_abs_diff"]:.4f} (seed {d["max_abs_diff_seed"]})')
    if det:
        out.append('执行内同种子两次: 最大概率偏差 '
                   f'{det.get("max_prob_abs_diff_same_seed")!r}, '
                   f'F1 差 {det.get("f1_abs_diff_same_seed")!r}, '
                   f'判定 {det.get("verdict")}')
    else:
        out.append('执行内确定性探针缺失, 不做该层结论')
    return out


# ------------------------------------------------------------------ tex

def _sci(v, nd=2):
    """1.788e-07 -> ('1.79', '-7'); 用于写成 $1.79\\times10^{-7}$。"""
    if v is None:
        return None
    s = f'{v:.{nd}e}'
    mant, _, exp = s.partition('e')
    return mant, str(int(exp))


def tex(res, paper_dir):
    rows = []
    for m, d in sorted((res.get('cross_execution') or {}).items()):
        rows.append(' & '.join([
            m, str(d['n_seeds']),
            f"{d['archive_mean_f1']:.4f}", f"{d['current_mean_f1']:.4f}",
            f"{d['mean_abs_diff']:.4f}", f"{d['max_abs_diff']:.4f}",
            str(d['max_abs_diff_seed']),
        ]) + r' \\')

    log_rows = []
    for d in (res.get('log_disagreement') or []):
        if not d['disagrees']:
            continue
        log_rows.append(' & '.join([
            d['model'], str(d['seed']), str(d['n_sources']),
            str(d['n_distinct']),
            ', '.join(f'{v:.4f}' for v in d['distinct_values']),
            f"{d['spread']:.4f}" if d['spread'] is not None else '0.0000',
        ]) + r' \\')

    det = res.get('determinism') or {}
    det_bit = ''
    sci = _sci(det.get('max_prob_abs_diff_same_seed'))

    def esc(s):
        # verdict 之类的取值带下划线, 直接进正文会被当成下标
        return str(s).replace('_', '\\_').replace('%', '\\%')

    if sci:
        det_bit = (f" Within one execution, training the same seed twice "
                   f"gives test probabilities that differ by at most "
                   f"${sci[0]}\\times10^{{{sci[1]}}}$ and an F1 differing by "
                   f"${det.get('f1_abs_diff_same_seed'):.1f}$, so the stack is "
                   f"stable to fp32 within an execution "
                   f"(verdict: {esc(det.get('verdict'))}).")
    note = ('Both columns use the same seeds, the same configuration and the '
            'same code; only the execution differs. '
            'The archived artifact was assembled from several partial runs.'
            + det_bit)

    tbl = ('\\begin{table}[!t]\n'
           '\\centering\\footnotesize\n'
           '\\setlength{\\tabcolsep}{2.5pt}\n'
           '\\caption{Cross-execution reproducibility audit on BitHeNet. '
           'The archived ten-seed artifact is compared with the same seeds '
           're-run inside the twenty-seed execution. Worst seed is the seed '
           'at which the largest absolute difference occurs.}\n'
           '\\label{tab:repro}\n'
           '\\begin{tabularx}{\\columnwidth}{lcccccc}\n'
           '\\hline\n'
           'Model & Seeds & Arch. F1 & New F1 & Mean $|\\Delta$| & '
           'Max $|\\Delta$| & Worst seed \\\\\n\\hline\n'
           + '\n'.join(rows) + '\n\\hline\n'
           '\\end{tabularx}\n'
           '\\begin{flushleft}\\footnotesize ' + note + '\n'
           '\\end{flushleft}\n'
           '\\end{table}\n')

    tbl2 = ''
    if log_rows:
        tbl2 = ('\n\\begin{table}[!t]\n'
                '\\centering\\footnotesize\n'
                '\\setlength{\\tabcolsep}{3pt}\n'
                '\\caption{Archived sources that report different F1 for the '
                'same model and seed. Only archived sources are counted; the '
                'twenty-seed execution is excluded, because it belongs to the '
                'cross-execution comparison in Table~\\ref{tab:repro} and not '
                'to a contradiction inside the archive. Values are compared at '
                'four decimals, the precision at which the logs print them. '
                'A single source is not a disagreement and is not listed.}\n'
                '\\label{tab:reprologs}\n'
                '\\begin{tabular}{lccccc}\n'
                '\\hline\n'
                'Model & Seed & Sources & Distinct & Values & Spread '
                '\\\\\n\\hline\n'
                + '\n'.join(log_rows) + '\n\\hline\n'
                '\\end{tabular}\n'
                '\\end{table}\n')

    d = os.path.join(paper_dir, 'generated_tables')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'repro_audit.tex')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(tbl + tbl2)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'tex'])
    ap.add_argument('--current',
                    default='results/p4/second_dataset_20seeds.json')
    ap.add_argument('--archive',
                    default='results/p4/second_dataset_confirm.json')
    ap.add_argument('--log_dir', default='results/p4')
    ap.add_argument('--determinism', default='results/p5/determinism.json')
    ap.add_argument('--paper', default=None)
    a = ap.parse_args()
    if a.cmd == 'build':
        res = build(a.current, a.archive, a.log_dir, a.determinism)
        os.makedirs('results/p5', exist_ok=True)
        p = 'results/p5/repro_audit.json'
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print('saved', p)
        for v in res['verdict']:
            print('  ', v)
        return 0
    rp = 'results/p5/repro_audit.json'
    if not os.path.exists(rp):
        print('missing', rp, '- run build first')
        return 2
    paper = a.paper or os.path.join(
        'Doc', 'Kimi_Agent_CCF-A实验脚本-25', 'TASGL_TDSC投稿')
    print('saved', tex(load(rp), paper))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
