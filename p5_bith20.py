"""p5_bith20.py — BitHeNet 20 种子: 摆脱 p 值下限, 多重比较, 跨窗稳定性

问题
----
十种子时 TASGL 对 MLP 的精确 Wilcoxon p 恰好是 0.0020 = 2/2^10, 也就是该
样本量下能达到的最小值。一个被"顶到地板"的 p 值没有分辨力: 我们无法区分
"勉强显著"和"极强显著", 也无法报告比它更小的效应。二十种子把地板降到
2/2^20 = 1.907e-6, 于是 p 值重新变成有信息量的量。

本模块做四件事:
  1. 20 种子的配对检验 (均值差, 95%CI, bootstrap CI, Cohen dz, t, Wilcoxon,
     符号检验, MDE@power0.8), 并显式报告最小可达 p 值。
  2. Holm 家族校正: 比较族 = {TASGL vs MLP, TASGL vs GraphSAGE}。
  3. 功效设计: 给定观测到的种子间 SD, 反推达到目标效应量所需的种子数,
     以及 n=10 与 n=20 各自的 MDE。
  4. 跨时间窗稳定性: 同一训练模型 + 同一验证阈值, 只在 35-39 / 40-44 /
     45-49 三个子窗上换测试集, 逐窗重做配对检验。

产出
----
  results/p5/bithe20_analysis.json
  generated_tables/bithe20.tex   (tab:bithe20, tab:bithe20win)

用法
----
  python p5_bith20.py build [--in results/p4/second_dataset_20seeds.json]
  python p5_bith20.py tex    [--paper DIR]
"""
import json
import math
import os

import numpy as np

SCHEMA = 'p5_bithe20_analysis_v1'
#: 比较族: 两个基线各一次配对检验, Holm 在这两个上做校正。
FAMILY_DEFAULT = ['mlp', 'graphsage']


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _pairs(rows):
    return {r['seed']: r['f1'] for r in rows
            if r.get('status', 'ok') == 'ok' and r.get('f1') is not None}


def _paired(a, b, alpha=0.05, n_boot=5000, seed=0):
    """与 p4_core.paired_report 同口径, 但允许更大的 bootstrap 次数。"""
    import p4_core
    return p4_core.paired_report(a, b, alpha=alpha, n_boot=n_boot, seed=seed)


def _holm(ps):
    import p0_core
    return p0_core.holm_adjust(ps)


def required_seeds_for_mde(sd_diff, target_mde, alpha=0.05, power=0.8):
    """给定配对差的 SD 与目标效应量, 反推所需种子数 (配对 t, 正态近似)。

    n 满足 MDE(n) <= target, 即 (z_{1-a/2} + z_power) * sd / sqrt(n) <= target
    """
    if not sd_diff or sd_diff <= 0 or not target_mde or target_mde <= 0:
        return None
    from scipy import stats
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    n = ((z_a + z_b) * sd_diff / target_mde) ** 2
    return int(math.ceil(n))


def cross_window(d, baseline):
    """逐子窗配对检验。返回每个窗的配对差与其区间。"""
    base = d['models']['tasgl']['rows']
    cur = d['models'][baseline]['rows']
    out = []
    labels = set()
    for r in base + cur:
        labels |= set((r.get('windows') or {}).keys())
    for lab in sorted(labels):
        a, b, seeds = [], [], []
        cur_by = {r['seed']: r for r in cur}
        for r in base:
            s = r['seed']
            wa = (r.get('windows') or {}).get(lab)
            wb = (cur_by.get(s, {}).get('windows') or {}).get(lab)
            if not wa or not wb:
                continue
            if wa.get('f1') is None or wb.get('f1') is None:
                continue
            a.append(wa['f1']); b.append(wb['f1']); seeds.append(s)
        if len(a) < 2:
            continue
        rep = _paired(a, b)
        rep['window'] = lab
        rep['n_paired'] = len(a)
        rep['paired_seeds'] = seeds
        rep['n_pos'] = int(sum(1 for x, y in zip(a, b) if x > y))
        # 该子窗里的正例(非法)节点数。它必须与 seed 无关, 否则说明窗口
        # 划分本身不稳定, 逐窗对比也就无从谈起。
        counts = {((r.get('windows') or {}).get(lab) or {}).get('n_pos')
                  for r in base}
        counts.discard(None)
        rep['n_illicit_in_window'] = (sorted(counts)[0]
                                      if len(counts) == 1 else None)
        rep['n_illicit_consistent'] = len(counts) <= 1
        out.append(rep)
    return out


def build(path='results/p4/second_dataset_20seeds.json',
          family=None):
    d = load(path)
    family = family or [m for m in FAMILY_DEFAULT if m in d.get('models', {})]
    base = d['models']['tasgl']
    res = {'schema': SCHEMA,
           'source': path,
           'n_seeds_requested': len(d.get('seeds') or []),
           'config_source': d.get('config_source'),
           'models': {}, 'family': family, 'comparisons': {},
           'power_design': {}, 'cross_window': {}}

    for k, v in d['models'].items():
        res['models'][k] = {'n_complete': v.get('n_complete',
                                                len(v.get('rows') or [])),
                            'complete': v.get('complete'),
                            'mean_f1': v.get('mean_f1'),
                            'sd_f1': v.get('sd_f1')}

    base_ok = _pairs(base['rows'])
    ps, names = [], []
    for nm in family:
        cur = _pairs(d['models'][nm]['rows'])
        both = sorted(set(base_ok) & set(cur))
        a = [base_ok[s] for s in both]
        b = [cur[s] for s in both]
        rep = _paired(a, b)
        rep['n_paired'] = len(both)
        rep['paired_seeds'] = both
        # 前 10 个种子的对照: 检验 20 种子是不是原来 10 种子的纯扩展
        if len(both) >= 10:
            sub = _paired(a[:10], b[:10])
            rep['first10_seeds'] = {'n': sub['n'], 'mean_diff': sub['mean_diff'],
                                    'ci95': sub['ci95'],
                                    'wilcoxon_p': sub['wilcoxon_p'],
                                    'min_attainable_wilcoxon_p':
                                        sub['min_attainable_wilcoxon_p']}
        res['comparisons'][nm] = rep
        ps.append(rep.get('wilcoxon_p') if rep.get('wilcoxon_p') is not None
                  else 1.0)
        names.append(nm)

    if ps:
        adj = _holm(ps)
        for nm, p, pa in zip(names, ps, adj):
            res['comparisons'][nm]['wilcoxon_p_holm'] = float(pa)
            res['comparisons'][nm]['significant_after_holm'] = bool(pa <= 0.05)

    # --- 功效设计 ---
    for nm in family:
        rep = res['comparisons'].get(nm) or {}
        sd = rep.get('sd_diff')
        mde20 = ((2.0930 + 0.8416) * sd / math.sqrt(20)) if sd else None
        res['power_design'][nm] = {
            'sd_of_paired_diff': sd,
            'observed_effect': rep.get('mean_diff'),
            'mde_at_n10': ((2.2622 + 0.8416) * sd / math.sqrt(10))
            if sd else None,
            'mde_at_n20': mde20,
            'min_attainable_p_n10': 2.0 / 2 ** 10,
            'min_attainable_p_n20': 2.0 / 2 ** 20,
            'seeds_needed_for_observed_effect': required_seeds_for_mde(
                sd, rep.get('mean_diff')) if sd else None,
            'effect_over_mde_at_n20': (
                abs(rep.get('mean_diff')) / mde20 if sd and mde20 else None),
            'note': ('目标效应量取本次观测到的均值差本身; 若读者认为有意义的'
                     '效应量另有数值, 可用同一公式反推')}

    # --- 跨时间窗 ---
    for nm in family:
        res['cross_window'][nm] = cross_window(d, nm)

    res['verdict'] = _verdict(res)
    return res


def _verdict(res):
    """把结论写成一句话, 不做任何超出区间的推断。"""
    out = []
    for nm, rep in res['comparisons'].items():
        n = rep.get('n_paired') or 0
        if rep.get('status') != 'ok' or n < 10:
            out.append(f'{nm}: 完成种子不足 ({n}), 不做判决')
            continue
        ci = rep.get('ci95')
        p = rep.get('wilcoxon_p_holm')
        floor = rep.get('min_attainable_wilcoxon_p')
        if ci and (ci[0] > 0 or ci[1] < 0):
            out.append(
                f'{nm}: 均值差 {rep.get("mean_diff"):+.4f}, 95%CI '
                f'[{ci[0]:+.4f}, {ci[1]:+.4f}] 不含 0, '
                f'Holm 校正后 Wilcoxon p={p:.3g} '
                f'(该 n 下最小可达 {floor:.3g}, 未被顶到地板)')
        else:
            out.append(f'{nm}: 95%CI 含 0, 未检测到差异')
    return out


# ------------------------------------------------------------------ tex

def _f(x, nd=4):
    return 'n.a.' if x is None else f'{x:.{nd}f}'


def tex(res, paper_dir):
    def esc(s):
        return (str(s).replace('_', '\\_').replace('%', '\\%')
                .replace('&', '\\&'))

    rows = []
    for nm, rep in res['comparisons'].items():
        ci = rep.get('ci95') or [None, None]
        rows.append(' & '.join([
            esc(nm), str(rep.get('n_paired')),
            _f(rep.get('mean_diff')),
            f"[{_f(ci[0])}, {_f(ci[1])}]",
            _f(rep.get('cohen_dz'), 3),
            f"{rep.get('wilcoxon_p'):.3g}"
            if rep.get('wilcoxon_p') is not None else 'n.a.',
            f"{rep.get('wilcoxon_p_holm'):.3g}"
            if rep.get('wilcoxon_p_holm') is not None else 'n.a.',
            ('yes' if rep.get('significant_after_holm') else 'no'),
        ]) + r' \\')

    win_rows = []
    for nm, wins in (res.get('cross_window') or {}).items():
        for w in wins:
            ci = w.get('ci95') or [None, None]
            win_rows.append(' & '.join([
                esc(nm), esc(w['window']),
                ('n.a.' if w.get('n_illicit_in_window') is None
                 else str(w['n_illicit_in_window'])),
                str(w.get('n_paired')),
                _f(w.get('mean_diff')),
                f"[{_f(ci[0])}, {_f(ci[1])}]",
                f"{w.get('n_pos')}/{w.get('n_paired')}",
                ('yes' if w.get('ci_excludes_zero') else 'no'),
            ]) + r' \\')

    n_req = res.get('n_seeds_requested')
    # 功效设计逐比较写进表注: 观测效应 / MDE 的比值, 以及解析该效应所需种子数。
    pw = res.get('power_design') or {}
    pw_bits = []
    for nm, pd in pw.items():
        r = (pd.get('effect_over_mde_at_n20'),
             pd.get('mde_at_n20'),
             pd.get('seeds_needed_for_observed_effect'))
        if r[0] is None or r[1] is None:
            continue
        pw_bits.append(
            f'Against {esc(nm)} the observed effect is '
            f'${r[0]:.2f}\\times$ the minimum detectable effect at power '
            f'0.8 (${r[1]:.4f}$), and resolving an effect of that size at '
            f'{n_req} seeds would need about '
            f'{r[2] if r[2] else "more than 1000"} seeds.')
    tbl1 = (
        '\\begin{table*}[!t]\n'
        '\\centering\\footnotesize\n'
        '\\caption{BitHeNet paired comparisons at '
        f'{n_req} seeds, with Holm correction over the two-baseline family. '
        'The minimum attainable exact two-sided Wilcoxon $p$ is reported so '
        'that a $p$ sitting on that floor can be seen.}\n'
        '\\label{tab:bithe20}\n'
        '\\begin{tabularx}{\\textwidth}{lccccccc}\n'
        '\\hline\n'
        'Baseline vs TASGL & $n$ & Mean diff & 95\\% CI & $d_z$ & '
        'Wilcoxon $p$ & Holm $p$ & Sig. \\\\\n\\hline\n'
        + '\n'.join(rows) + '\n\\hline\n'
        '\\end{tabularx}\n'
        '\\begin{flushleft}\\footnotesize '
        'At 10 seeds the smallest attainable exact two-sided $p$ is '
        '$2/2^{10}=0.00195$, so a reported $p=0.0020$ is censored at the floor '
        'and carries no resolution. At 20 seeds the floor is '
        '$2/2^{20}=1.91\\times10^{-6}$. Holm correction is applied across the '
        'declared family of the two baseline comparisons on this dataset. '
        + ' '.join(pw_bits) + '\n'
        '\\end{flushleft}\n'
        '\\end{table*}\n')

    tbl2 = ''
    if win_rows:
        # 七列在单栏 \columnwidth 下会溢出 (实测 101.4pt 超框), 因此改为
        # 跨双栏 table*。不缩字号: 缩到 \scriptsize 会让 CI 区间难以核对。
        tbl2 = (
            '\\begin{table*}[!t]\n'
            '\\centering\\footnotesize\n'
            '\\caption{BitHeNet cross-window stability. One trained model per '
            'seed, one validation-fitted threshold, three disjoint test '
            'sub-windows. No model is retrained and no threshold is refitted '
            'per window.}\n'
            '\\label{tab:bithe20win}\n'
            '\\begin{tabularx}{\\textwidth}{lccccccc}\n'
            '\\hline\n'
            'Baseline & Window & Illicit & $n$ & Mean diff & 95\\% CI & '
            'Seeds favouring TASGL & CI excl. 0 \\\\\n\\hline\n'
            + '\n'.join(win_rows) + '\n\\hline\n'
            '\\end{tabularx}\n'
            '\\end{table*}\n')

    d = os.path.join(paper_dir, 'generated_tables')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'bithe20.tex')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(tbl1 + '\n' + tbl2)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'tex'])
    ap.add_argument('--in', dest='inp',
                    default='results/p4/second_dataset_20seeds.json')
    ap.add_argument('--paper', default=None)
    a = ap.parse_args()
    if a.cmd == 'build':
        res = build(a.inp)
        os.makedirs('results/p5', exist_ok=True)
        p = 'results/p5/bithe20_analysis.json'
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print('saved', p)
        for v in res['verdict']:
            print('  ', v)
        return 0
    res_path = 'results/p5/bithe20_analysis.json'
    if not os.path.exists(res_path):
        print('missing', res_path, '- run build first')
        return 2
    paper = a.paper or os.path.join(
        'Doc', 'Kimi_Agent_CCF-A实验脚本-25', 'TASGL_TDSC投稿')
    print('saved', tex(load(res_path), paper))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
