"""p5_tex_conformance.py — 按官方 IEEE TDSC 模板校验投稿目录

对照样本: Computer_Society_LaTeX_template-IEEE Transactions on Dependable
and Secure Computing/bare_jrnl_new_sample4.tex

检查项 (每一项都给出样本里的依据行):
  C1  \\documentclass[lettersize,journal]{IEEEtran}
  C2  \\markboth{...}{...} 出现在 \\maketitle 之前
  C3  \\IEEEpubid{...} 存在 (期刊稿必需)
  C4  使用了 \\IEEEpubid 就必须调用 \\IEEEpubidadjcol (否则第二栏文字压标识)
  C5  \\begin{abstract} / \\end{abstract}
  C6  \\begin{IEEEkeywords} / \\end{IEEEkeywords}
  C7  首个 \\section 之后出现 \\IEEEPARstart
  C8  表格: \\caption 在 \\begin{tabular*} 之前, \\label 在 \\caption 之后
  C9  图: \\includegraphics 在 \\caption 之前, \\label 在 \\caption 之后
  C10 \\bibliographystyle{IEEEtran}
  C11 \\appendices 之后用 \\section 起附录
  C12 正文不得出现 CJK 字符
  C13 章节层级: 一级罗马数字, 二级大写字母, 三级阿拉伯数字
  C14 参考文献节之前应有未编号的声明/致谢节 (模板建议)

用法:
  python p5_tex_conformance.py check [--paper DIR]
  退出码 0 表示无 FAIL。
"""
import os
import re
import sys

CJK = re.compile(r'[一-鿿　-〿＀-￯]')


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _flatten(paper_dir):
    """把 main.tex 的 \\input 展开成一个字符串 (不递归编译)。"""
    main = _read(os.path.join(paper_dir, 'main.tex'))
    out = []
    for line in main.splitlines():
        m = re.match(r'\s*\\input\{([^}]*)\}', line)
        if m:
            rel = m.group(1)
            if not rel.endswith('.tex'):
                rel += '.tex'
            p = os.path.join(paper_dir, rel)
            if os.path.exists(p):
                out.append(_read(p))
            else:
                out.append(f'% MISSING INPUT {rel}\n')
        else:
            out.append(line)
    return '\n'.join(out)


def check(paper_dir):
    s = _flatten(paper_dir)
    res = []

    def add(cid, name, ok, detail=''):
        res.append({'id': cid, 'name': name,
                    'status': 'PASS' if ok else 'FAIL', 'detail': detail})

    add('C1', 'documentclass lettersize,journal',
        bool(re.search(r'\\documentclass\[[^\]]*lettersize[^\]]*journal',
                       s)), s.splitlines()[0] if s else '')

    mk = s.find('\\maketitle')
    mb = s.find('\\markboth')
    add('C2', 'markboth before maketitle', mk > 0 and 0 <= mb < mk)

    has_pubid = '\\IEEEpubid' in s and '\\IEEEpubidadjcol' not in s[:0]
    add('C3', 'IEEEpubid present',
        bool(re.search(r'\\IEEEpubid\{', s)),
        '期刊稿必需; 样本第 27 行')

    has_pubid_macro = bool(re.search(r'\\IEEEpubid\{', s))
    has_adj = '\\IEEEpubidadjcol' in s
    add('C4', 'IEEEpubidadjcol called when IEEEpubid used',
        (not has_pubid_macro) or has_adj,
        '样本第 30 行说明: 用了 IEEEpubid 就必须调用 IEEEpubidadjcol')

    add('C5', 'abstract environment',
        '\\begin{abstract}' in s and '\\end{abstract}' in s)
    add('C6', 'IEEEkeywords environment',
        '\\begin{IEEEkeywords}' in s and '\\end{IEEEkeywords}' in s)

    first_sec = s.find('\\section{')
    seg = s[first_sec:first_sec + 4000] if first_sec >= 0 else ''
    add('C7', 'IEEEPARstart in the first section',
        '\\IEEEPARstart' in seg)

    # --- C8 表格 ---
    bad_tbl = []
    for m in re.finditer(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}',
                         s, re.S):
        blk = m.group(1)
        cap = blk.find('\\caption')
        lab = blk.find('\\label')
        tab = re.search(r'\\begin\{tabular', blk)
        tabpos = tab.start() if tab else None
        if cap < 0:
            bad_tbl.append('no caption')
        else:
            if tabpos is not None and cap > tabpos:
                bad_tbl.append('caption after tabular')
            if lab >= 0 and lab < cap:
                bad_tbl.append('label before caption')
    add('C8', 'tables: caption before tabular, label after caption',
        not bad_tbl, '; '.join(bad_tbl[:5]))

    # --- C9 图 ---
    bad_fig = []
    for m in re.finditer(r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}',
                         s, re.S):
        blk = m.group(1)
        cap = blk.find('\\caption')
        lab = blk.find('\\label')
        inc = blk.find('\\includegraphics')
        if cap < 0:
            bad_fig.append('no caption')
            continue
        if inc >= 0 and inc > cap:
            bad_fig.append('includegraphics after caption')
        if lab >= 0 and lab < cap:
            bad_fig.append('label before caption')
    add('C9', 'figures: graphics before caption, label after caption',
        not bad_fig, '; '.join(bad_fig[:5]))

    add('C10', 'bibliographystyle IEEEtran',
        '\\bibliographystyle{IEEEtran}' in s)

    ap = s.find('\\appendices')
    after = s[ap:] if ap >= 0 else ''
    add('C11', 'appendices followed by \\section',
        ap < 0 or bool(re.search(r'\\section\{', after)))

    hits = []
    for i, line in enumerate(s.splitlines(), 1):
        # LaTeX 注释不参与排版, 只检查会被排出来的字符
        code = re.sub(r'(?<!\\\\)%.*$', '', line)
        if CJK.search(code):
            hits.append(f'line {i}: {line.strip()[:60]}')
    add('C12', 'no CJK characters in the typeset text', not hits,
        '; '.join(hits[:3]))

    # --- C13 章节层级 ---
    secs = re.findall(r'\\(section|subsection|subsubsection)\*?\{', s)
    order = {'section': 0, 'subsection': 1, 'subsubsection': 2}
    jump = []
    prev = None
    for k in secs:
        if prev is not None and order[k] > order[prev] + 1:
            jump.append(f'{prev}->{k}')
        prev = k
    add('C13', 'heading levels do not skip a level', not jump,
        '; '.join(jump[:3]))

    # 必须是 \bibliography{...}; 用 \bibliography 去 find 会先命中
    # \bibliographystyle, 把后面的声明节误判成"在参考文献之后"。
    bibpos = s.find('\\bibliography{')
    pre = s[:bibpos] if bibpos >= 0 else s
    add('C14', 'unnumbered declaration section before references',
        bool(re.search(r'\\section\*\{', pre)))

    miss = re.findall(r'% MISSING INPUT (\S+)', s)
    add('C15', 'all \\input files exist', not miss, '; '.join(miss))

    # --- C16/C17 排版引擎: 摘要与关键词整段加粗是 IEEEtran 期刊模式的设计 ---
    # 曾有反馈认为 "Abstract 后面的内容被加粗" 是格式错误。这里不靠口头
    # 解释, 而是直接读 IEEEtran.cls: journal 模式下 \abstract 与
    # \IEEEkeywords 的定义自带 \bfseries, 所以粗体正文是这个模板的正确输出。
    cls_path = os.path.join(paper_dir, 'IEEEtran.cls')
    if not os.path.exists(cls_path):
        add('C16', 'IEEEtran.cls present next to the manuscript', False,
            cls_path)
        add('C17', 'bold abstract/keywords is defined by IEEEtran', False,
            'no class file to read')
        return res
    add('C16', 'IEEEtran.cls present next to the manuscript', True,
        os.path.basename(cls_path))
    cls = _read(cls_path)
    # IEEEtran 在两种期刊分支里都是把 \bfseries 打在 abstract 开头、由
    # \endabstract 的 \normalfont 收回, 所以整段正文是粗体。
    branch_base = ('\\@IEEEabskeysecsize\\bfseries\\textit{\\abstractname}'
                   '---\\relax')
    branch_compsoc = '\\@IEEEtweakunitybaselinestretch{1.15}\\bfseries'
    hit_base = branch_base in cls
    hit_compsoc = branch_compsoc in cls
    ends_normal = bool(re.search(r'\\def\\endabstract\{[^%]*?\\normalfont',
                                 cls, re.S))
    ok = (hit_base or hit_compsoc) and ends_normal
    which = ('two-column base branch' if hit_base else '') + \
            (' + ' if hit_base and hit_compsoc else '') + \
            ('compsoc journal branch' if hit_compsoc else '')
    add('C17', 'bold abstract/keywords body is IEEEtran journal style', ok,
        ('IEEEtran.cls issues \\bfseries at the start of \\abstract (%s) and '
         'restores \\normalfont in \\endabstract, so a bold abstract body is '
         'the template output and not a formatting defect' % which)
        if ok else 'class definitions changed; do not assume bold is correct '
        'until this is re-checked')

    return res


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['check'])
    ap.add_argument('--paper', default=os.path.join(
        'Doc', 'Kimi_Agent_CCF-A实验脚本-25', 'TASGL_TDSC投稿'))
    a = ap.parse_args()
    res = check(a.paper)
    n_fail = 0
    for r in res:
        if r['status'] == 'FAIL':
            n_fail += 1
        print(f"{r['id']:>4s} {r['status']:<4s} {r['name']}"
              + (f"  [{r['detail']}]" if r['detail'] else ''))
    print(f'\n{len(res) - n_fail}/{len(res)} pass, {n_fail} fail')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
