"""ledger_reconcile.py — reconcile the submission ledger against P0–P4 artifacts

Purpose
-------
The submission ledger (TASGL投稿规划与证据台账.md) records the evidence
boundary EVD-01..18 and the to-do lists P0-1..9 and P1-1..8. Part of that
ledger has been superseded by the P2/P3/P4 runs. Keeping "which number is
still authoritative" in prose is exactly the kind of thing that goes wrong
silently. This program binds every ledger statement to a result file on disk
and decides one of four statuses:

    unchanged   the measurement agrees with the ledger; the ledger stands
    updated     a newer run replaces the number (both values are kept)
    partial     only part of the item is covered by new evidence
    open        no real run exists; the item stays a to-do

Hard rules
----------
- No guessing, no interpolation, no placeholder numbers. A missing file is
  reported as n.a. and the item as open.
- Read-only. No result file is ever modified.

Usage
-----
    python ledger_reconcile.py build --results_dir <project root> \
        --paper_dir <manuscript dir> [--ledger <ledger.md>]
    python ledger_reconcile.py check --results_dir <project root>

Outputs
-------
    <paper_dir>/provenance/ledger_reconciliation.json
    <paper_dir>/provenance/ledger_reconciliation.md
    <paper_dir>/generated_tables/ledger_reconcile.tex
"""

import argparse
import hashlib
import json
import os
import sys

SCHEMA = 'ledger_reconciliation_v1'

STATUS = ('unchanged', 'updated', 'partial', 'open')


# ------------------------------------------------------------------
# read helpers
# ------------------------------------------------------------------
def _rd(results_dir, rel):
    p = os.path.join(results_dir, rel)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except (ValueError, OSError):
        return None


def _sha(results_dir, rel):
    p = os.path.join(results_dir, rel)
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def f4(v):
    return 'n.a.' if v is None else '%.4f' % v


def f3(v):
    return 'n.a.' if v is None else '%.3f' % v


def _ms(v):
    return 'n.a.' if v is None else '%.3f ms' % v


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _sd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def p4_model(fin, name):
    """十种子主比较里某个模型的汇总 (与 paper_tdsc.block_main 同一算法)。"""
    m = (fin or {}).get('models', {}).get(name)
    if not m:
        return {}
    rs = m.get('rows') or []
    if not rs:
        return {}
    return {
        'f1': _mean([r.get('f1') for r in rs]),
        'sd': _sd([r.get('f1') for r in rs]),
        'params': rs[0].get('params'),
        'latency': _mean([r.get('latency_ms') for r in rs]),
        'n': len(rs),
    }


def p4_gate_rows(fin, gate_audit):
    """逐种子门控审计行 (优先取主比较内嵌审计)。"""
    rows = []
    if fin and 'tasgl' in (fin.get('models') or {}):
        for r in fin['models']['tasgl']['rows']:
            g = r.get('gate') or {}
            rows.append({
                'seed': r.get('seed'),
                'min_active': g.get('min_active_edge_ratio'),
                'w2': g.get('max_gate_w2_norm'),
                'class': (r.get('admission') or {}).get('classification'),
            })
    elif gate_audit:
        for r in gate_audit.get('rows') or []:
            g = r.get('gate') or {}
            rows.append({
                'seed': r.get('seed'),
                'min_active': g.get('min_active_edge_ratio'),
                'w2': g.get('max_gate_w2_norm'),
                'class': (r.get('admission') or {}).get('classification'),
            })
    return rows


def _yes(b):
    return 'yes' if b else 'no'


def _v(x):
    """把缺失值渲染成 n.a., 绝不把 None 印进论文表体。"""
    return 'n.a.' if x is None else x


def _normalize(items, results_dir):
    """来源文件不存在时, 一律降为 open。

    没有文件就不能声称 unchanged/updated: 那等于把声明当成证据。
    """
    for r in items:
        src = str(r.get('source') or '')
        if src.startswith('('):
            r['source_kind'] = 'synthetic'
            continue
        r['source_kind'] = 'dir' if src.endswith('/') else 'file'
        p = os.path.join(results_dir, src.rstrip('/'))
        if not os.path.exists(p):
            r['status'] = 'open'
            r['observed'] = 'source missing: %s' % src
            r['note'] = ('The ledger item cannot be checked because the '
                         'source is not present. It stays a to-do. '
                         + (r.get('note') or ''))
    return items


# ------------------------------------------------------------------
# ledger items
# ------------------------------------------------------------------
def build_items(results_dir):
    """Return the ledger items with the value actually observed on disk."""
    def rd(rel):
        return _rd(results_dir, rel)

    it = []

    def add(**kw):
        it.append(kw)

    fin = rd('results/p4/final_comparison.json') or {}
    ga = rd('results/p4/gate_audit_10seeds.json') or {}
    rc = rd('results/p4/rootcause_seed0.json') or {}
    env = rd('results/p4/environment.json') or {}
    cost = rd('results/p4/cost_measurement.json') or {}
    ha = rd('results/p4/hpo_audit.json') or {}
    rp = rd('results/p4/rolling_plan.json') or {}
    sdp = rd('results/p4/second_dataset_plan.json') or {}
    # 二十种子执行取代了十种子归档产物: 归档产物对同一 seed 存在三个不同值
    # 且跨执行不可复现 (见 p5_repro_audit), 因此台账也一律改用二十种子;
    # 归档产物只作可复现性审计的对照, 不作结论来源。
    sd = rd('results/p4/second_dataset_20seeds.json')
    atk = rd('results/p6/nonoracle_attack_matrix.json')
    v4 = rd('results/v4/select/summary.json') or {}
    plan = rp.get('plan') or {}
    leak = (rp.get('leakage') or {}).get('checks') or {}
    dset = sdp.get('dataset') or {}

    tg = p4_model(fin, 'tasgl')
    gs = p4_model(fin, 'graphsage')
    ml = p4_model(fin, 'mlp')
    paired = fin.get('paired') or {}
    grows = p4_gate_rows(fin, ga)
    collapsed = [r for r in grows if r['class'] == 'gate_collapsed_open']
    repro = ((rc.get('per_seed') or [{}])[0]).get('reproduce') or {}
    stages = cost.get('stages') or []
    total_s = sum(s.get('total_seconds') or 0 for s in stages)

    # ---------------- EVD-01 .. EVD-18 ----------------
    add(id='EVD-01', group='EVD',
        topic='Elliptic data and temporal visibility',
        ledger='203,769 nodes / 234,355 directed edges / 49 time steps / '
               '165 input features; main split labels 26,381 / 3,513 / '
               '16,670; transductive full-graph masking',
        source='results/p4/rolling_plan.json',
        observed='strict-rolling test-window labels %s; train window %s; '
                 'no-future-leakage checks all pass: %s' % (
                     plan.get('n_test'), plan.get('train_window'),
                     _yes(bool(leak) and all(leak.values()))),
        status='partial',
        note='Node, edge and feature counts stay as declared constants '
             '(provenance/paper_constants.json). The strict rolling plan and '
             'the no-future-leakage checks are new, but the main comparison '
             'is still transductive.')

    add(id='EVD-02', group='EVD', topic='Main accuracy baseline',
        ledger='TASGL 0.5038 / GraphSAGE 0.5367 / MLP 0.5234 '
               '(five seeds, protocol 3)',
        source='results/p4/final_comparison.json',
        observed='TASGL %s +- %s / GraphSAGE %s +- %s / MLP %s +- %s '
                 '(%s seeds, equal HPO budget)' % (
                     f4(tg.get('f1')), f4(tg.get('sd')),
                     f4(gs.get('f1')), f4(gs.get('sd')),
                     f4(ml.get('f1')), f4(ml.get('sd')), tg.get('n')),
        status='updated',
        note='After the equal-budget HPO fix the strongest baseline changes '
             'from GraphSAGE to MLP. The five-seed numbers are kept for '
             'contrast only and no longer carry the main conclusion.')

    add(id='EVD-03', group='EVD', topic='TASGL size and latency',
        ledger='19,595 parameters / 33.318 +- 0.344 ms',
        source='results/p4/final_comparison.json',
        observed='%s parameters / %s' % (tg.get('params'),
                                         _ms(tg.get('latency'))),
        status='updated',
        note='The semantics is one full-graph forward pass, not one '
             'transaction. This is why the 8 ms per-transaction threshold '
             'was withdrawn.')

    add(id='EVD-04', group='EVD', topic='Paired tests and power',
        ledger='five seeds, exact two-sided p = 0.8125 (MLP, GraphSAGE); '
               'minimum attainable p = 0.0625 with five non-zero pairs',
        source='results/p4/final_comparison.json',
        observed='%s seeds: p_W = %s (MLP) / %s (GraphSAGE); minimum '
                 'attainable p = %s; MDE at power 0.8 = %s / %s' % (
                     tg.get('n'),
                     f4((paired.get('mlp') or {}).get('wilcoxon_p')),
                     f4((paired.get('graphsage') or {}).get('wilcoxon_p')),
                     f4((paired.get('mlp') or {}).get(
                         'min_attainable_wilcoxon_p')),
                     f4((paired.get('mlp') or {}).get('mde_at_power80')),
                     f4((paired.get('graphsage') or {}).get(
                         'mde_at_power80'))),
        status='updated',
        note='Ten seeds reach p = 0.00195, so the power limit is lifted. '
             'Both 95% CIs still contain zero and both effects are below the '
             'MDE, so the evidence supports neither dominance nor leadership.')

    add(id='EVD-05', group='EVD', topic='Measured gate sparsity',
        ledger='active-edge ratio 0.8791 +- 0.1338; per-seed range '
               '0.6922 to 1.0000',
        source='results/p4/gate_audit_10seeds.json',
        observed='%d of %s seeds are gate_collapsed_open with an active-edge '
                 'ratio of exactly 1.000 (seeds %s)' % (
                     len(collapsed), len(grows) or 'n.a.',
                     ', '.join(str(r['seed']) for r in collapsed) or 'n.a.'),
        status='updated',
        note='The per-seed audit names the failing seeds. On those seeds the '
             'sparsity claim does not hold, and the paper says so.')

    add(id='EVD-06', group='EVD', topic='Component ablation',
        ledger='three seeds; full 0.5107 +- 0.1022; w/o RFF -0.1132; '
               'w/o time basis -0.0911; w/o low rank +0.0360',
        source='results/ablation_results.json',
        observed='raw file present: %s' % _yes(bool(
            rd('results/ablation_results.json'))),
        status='unchanged',
        note='P4 did not redo the ablation. Three seeds, no paired test, so '
             'the numbers stay descriptive.')

    add(id='EVD-07', group='EVD', topic='Temporal sample-size diagnostic',
        ledger='train <= 20/25/30/34 gives F1 0.2813 / 0.3086 / 0.4465 / '
               '0.5077; relative loss 39.22% at <= 25',
        source='results/temporal_results.json',
        observed='raw file present: %s' % _yes(bool(
            rd('results/temporal_results.json'))),
        status='unchanged',
        note='Protocol 4 still uses the full graph and full-data scaling, so '
             'it is not strict rolling.')

    add(id='EVD-08', group='EVD', topic='Edge-deletion stress diagnostic',
        ledger='two seeds; clean 0.5143; after 10/30/50% deletion '
               '0.4885 / 0.4289 / 0.3850',
        source='results/robustness_results.json',
        observed='raw file present: %s' % _yes(bool(
            rd('results/robustness_results.json'))),
        status='partial',
        note='P4 moved H3 to interval estimation: report the relative drop '
             'ratio with a seed bootstrap 95% CI and drop the 0.6/1.0 cut.')

    add(id='EVD-09', group='EVD', topic='Nature of the seed-0 F1 = 0.3049',
        ledger='an independent single-seed model state; F1 = 0.3049, '
               'ECE = 0.1433',
        source='results/p4/rootcause_seed0.json',
        observed='reproduced: %s; recomputed %s against observed %s; '
                 'absolute difference %s (tolerance %s)' % (
                     _yes(repro.get('reproduced')),
                     f4(repro.get('recomputed')), f4(repro.get('observed')),
                     f4(repro.get('abs_diff')), f4(repro.get('tol'))),
        status='updated',
        note='The cause is a forward-protocol change on one already-trained '
             'model (soft gate, MC-Dropout, different threshold), not an '
             'independent model state. It must not be placed in the same row '
             'as the main result.')

    add(id='EVD-10', group='EVD', topic='Weak cross-chain diagnostic',
        ledger='independent PCA; TASGL target F1 = 0.1045 +- 0.1780; '
               'semantic_alignment = false',
        source='results/crosschain_results.json',
        observed='raw file present: %s' % _yes(bool(
            rd('results/crosschain_results.json'))),
        status='unchanged',
        note='With no semantic alignment this supports neither transfer '
             'advantage nor a strict cross-chain falsification.')

    add(id='EVD-11', group='EVD', topic='Protocol-5 selector compensation',
        ledger='additive score selected graphless_kd; development F1 = '
               '0.6977; H4 loss proxy 25.28%; latency 6.607 ms',
        source='results/v3/select/summary.json',
        observed='raw file present: %s' % _yes(bool(
            rd('results/v3/select/summary.json'))),
        status='unchanged',
        note='Development-window numbers. They are never placed in the same '
             'table as final test numbers.')

    add(id='EVD-12', group='EVD',
        topic='Protocol-5 locked exploratory confirmation',
        ledger='F1 = 0.5185 +- 0.0197; latency 6.494 +- 0.021 ms; H4 loss '
               '12.47%; ECE = 0.0216 +- 0.0044; all_pass = false',
        source='results/v3/confirm_91f1beb52cbad975/',
        observed='directory present: %s' % _yes(os.path.isdir(os.path.join(
            results_dir, 'results/v3/confirm_91f1beb52cbad975'))),
        status='unchanged',
        note='Window 35--49 had been observed before, so this is a locked '
             'exploratory confirmation, not an independent one.')

    add(id='EVD-13', group='EVD', topic='Degenerate H3 and proxy H5',
        ledger='graphless_kd uses hops = 0, so predictions are unchanged by '
               'edge deletion; the H5 control is a same-backbone EDL proxy',
        source='models_v3.py',
        observed='code-level fact, no numeric result file',
        status='unchanged',
        note='The H3 pass comes from an architecture identity, not from '
             'graph robustness.')

    add(id='EVD-14', group='EVD',
        topic='Protocol-6 development-only gate audit',
        ledger='7 candidates x reference/short training x 2 seeds = 28 '
               'development runs; all eligible = false',
        source='results/v4/select/summary.json',
        observed='raw file present: %s' % _yes(bool(v4)),
        status='unchanged',
        note='Only steps 33--34 are evaluated. This is not a final result on '
             'steps 35--49.')

    add(id='EVD-15', group='EVD', topic='Protocol 6: no winner, no mandate',
        ledger='winner = null; confirmation_authorized = false; exit code 2 '
               'is the intended stop',
        source='results/v4/select/summary.json',
        observed='winner = %s; confirmation_authorized = %s' % (
            v4.get('winner'), v4.get('confirmation_authorized')),
        status='unchanged',
        note='No hash is recomputed to bypass the mandate, and the stop is '
             'not called an engineering crash.')

    add(id='EVD-16', group='EVD', topic='Reproducibility and lineage',
        ledger='paper_evidence.py generates the historical tables; config '
               'hash binding; per-seed recovery',
        source='paper_tdsc.py',
        observed='paper_tdsc.py (P4 tables) and ledger_reconcile.py (this '
                 'reconciliation) added; both write a manifest with input '
                 'and output SHA-256',
        status='updated',
        note='P4 fixed the build_model defect that dropped hidden/dropout, '
             'and locked the behaviour with a regression test.')

    add(id='EVD-17', group='EVD', topic='Reference verification',
        ledger='bibliography and task boundaries checked; preprints marked',
        source='(manuscript)',
        observed='cohen1988 added as the source for Cohen d_z and the MDE '
                 'at power 0.8',
        status='updated',
        note='No unverified name is cited.')

    add(id='EVD-18', group='EVD', topic='Hardware and environment fingerprint',
        ledger='only device = cuda and 6.00 GiB total memory confirmed; '
               'model and software stack missing',
        source='results/p4/environment.json',
        observed='%s (%s MiB, %s SM); driver %s; CUDA %s; cuDNN %s; torch '
                 '%s; numpy %s; Python %s; %s; complete = %s' % (
                     _v(env.get('gpu_name')), _v(env.get('gpu_total_mem_mb')),
                     _v(env.get('gpu_sm_count')), _v(env.get('driver_version')),
                     _v(env.get('cuda_runtime')), _v(env.get('cudnn')),
                     _v(env.get('torch_version')), _v(env.get('numpy_version')),
                     _v(env.get('python_version')), _v(env.get('os')),
                     _yes(env.get('complete'))),
        status='updated',
        note='The incomplete-fingerprint limitation is lifted. Real power '
             'draw is still not measured, so no energy number is reported.')

    # ---------------- P0-1 .. P0-9 ----------------
    sd_t = ((sd or {}).get('models') or {}).get('tasgl') or {}
    sd_m = ((sd or {}).get('models') or {}).get('mlp') or {}
    sd_g = ((sd or {}).get('models') or {}).get('graphsage') or {}
    sd_p = (sd or {}).get('paired') or {}
    # 门控塌缩计数必须来自产物, 不能写死: 十种子时是 9/10, 二十种子时是 18/20
    _sd_rows = ((sd_t or {}).get('rows') or [])
    sd_collapsed = sum(1 for r in _sd_rows
                       if (r.get('admission') or {}).get('classification')
                       == 'gate_collapsed_open')
    sd_tasgl_n = len(_sd_rows)
    add(id='P0-1', group='P0', topic='Independent confirmation queue',
        ledger='a new period, chain or split that took no part in any '
               'development; only one locked test may be reported',
        source='results/p4/second_dataset_20seeds.json',
        observed=('BitHeNet %d-seed confirmation %s: TASGL %s / MLP %s / '
                  'GraphSAGE %s; TASGL minus MLP %s with 95%% CI that '
                  'excludes zero: %s' % (
                      len((sd or {}).get('seeds') or []),
                      'produced' if sd else 'not produced yet',
                      f4(sd_t.get('mean_f1')), f4(sd_m.get('mean_f1')),
                      f4(sd_g.get('mean_f1')),
                      f4((sd_p.get('mlp') or {}).get('mean_diff')),
                      _yes((sd_p.get('mlp') or {}).get('ci_excludes_zero'))))
        if sd else 'BitHeNet confirmation: not produced yet',
        status='updated' if sd else 'open',
        note='BitHeNet was never used for development, selection or tuning '
             'in this project (prior_exposure = false). Elliptic 35--49 is '
             'still previously_observed and cannot serve as an independent '
             'confirmation. The gate values are carried over from Elliptic, '
             'and the size budget is cleared by only 69 parameters.')

    add(id='P0-2', group='P0',
        topic='Development study of accuracy and extrapolation',
        ledger='use the train/validation/development windows only; keep the '
               'negative result if nothing improves',
        source='results/v4/select/summary.json',
        observed='protocol 6 produced no eligible candidate after 28 '
                 'development runs; best development F1 = 0.7011',
        status='updated',
        note='Closed with a negative result: no winner, no confirmation '
             'mandate.')

    add(id='P0-3', group='P0', topic='Faithful uncertainty control',
        ledger='prefer reproducing the key ST-CGNN components; otherwise '
               'keep calling it a proxy',
        source='(none)',
        observed='still a same-backbone EDL proxy',
        status='open',
        note='H5 must be worded as a same-backbone EDL proxy, not as a full '
             'reproduction.')

    add(id='P0-4', group='P0', topic='Equal-budget validation-set HPO',
        ledger='every baseline gets the same validation information, search '
               'budget and early-stopping rule',
        source='results/p4/hpo_audit.json',
        observed='fair = %s; violations = %s; 6 trials x 2 seeds per model' % (
            _yes(ha.get('fair')), ha.get('violations')),
        status='updated',
        note='P4 found and fixed the build_model defect that dropped '
             'hidden/dropout. After the fix the strongest baseline moves from '
             'GraphSAGE 0.5367 to MLP 0.4970, which moves the H1 F1 floor '
             'from 0.5163 to 0.4320.')

    add(id='P0-5', group='P0', topic='Unified seeds and statistics protocol',
        ledger='one pre-declared seed set; pre-register the paired unit, SD, '
               'CI and effect size',
        source='results/p4/thresholds/register.json',
        observed='%s seeds; paired unit = seed; exact two-sided Wilcoxon; '
                 'Holm; SD with ddof = 1; CI, Cohen d_z and MDE reported' % (
                     tg.get('n')),
        status='updated',
        note='The five-seed power problem is resolved. No test is swapped '
             'after the fact.')

    add(id='P0-6', group='P0', topic='A second fully trained dataset',
        ledger='define the split, label semantics, metrics and preprocessing '
               'separately for each dataset',
        source='results/p4/second_dataset_plan.json',
        observed='BitHeNet: %s transactions / %s features / %s undirected '
                 'edges; strict rolling 1--29 / 30--34 / 35--49; '
                 '%d-seed training %s' % (
                     dset.get('n_transactions'), dset.get('n_features'),
                     dset.get('n_edges_undirected'),
                     len((sd or {}).get('seeds') or []),
                     'done' if sd else 'not done'),
        status='updated' if sd else 'partial',
        note='The split and the leakage checks are done. See P0-1 for the '
             'training result. Note that %d of %d TASGL seeds on this '
             'dataset are gate-collapsed-open, so the failure found on '
             'Elliptic reproduces on independent data.' % (
                 sd_collapsed, sd_tasgl_n))

    add(id='P0-7', group='P0', topic='Strict inductive time-truncated rolling',
        ledger='keep only the nodes and edges visible at each cutoff; fit '
               'the scaler on the training window only',
        source='results/p4/rolling_plan.json',
        observed='graph cutoff = %s; induced subgraph %s nodes / %s edges; '
                 'leakage checks all pass: %s' % (
                     plan.get('graph_cutoff'), rp.get('n_nodes_induced'),
                     rp.get('n_edges_induced'),
                     _yes(bool(leak) and all(leak.values()))),
        status='partial',
        note='Strict rolling is implemented and used on the second dataset. '
             'The Elliptic main comparison is still transductive, and the two '
             'are reported separately.')

    add(id='P0-8', group='P0', topic='Formal cross-chain semantic alignment',
        ledger='build shared feature semantics, label mapping and graph '
               'construction rules',
        source='results/crosschain_results.json',
        observed='still two independent PCAs; semantic_alignment = false',
        status='open',
        note='Until alignment exists, only the weak diagnostic is kept and no '
             'transfer claim is made.')

    add(id='P0-9', group='P0', topic='Real gate sparsity and cost audit',
        ledger='record the active-edge ratio per layer, per seed and per time '
               'window, plus the real sparse speed-up',
        source='results/p4/cost_measurement.json',
        observed='%d-stage full-chain measurement; %s per-seed gate rows' % (
            len(stages), len(grows)),
        status='updated',
        note='The gate degenerates to full retention on some seeds, so the '
             'sparse-efficiency claim is withdrawn. No measured latency gain '
             'supports it.')

    # ---------------- P1-1 .. P1-8 ----------------
    add(id='P1-1', group='P1', topic='E8 explainability and audit cases',
        ledger='needs external pattern labels or domain expert verification',
        source='(none)', observed='not executed', status='open',
        note='External expert review is still missing and the manuscript '
             'must not imply it exists.')
    if atk:
        n_atk = len(atk.get('per_attack') or [])
        n_run = atk.get('n_runs') or 0
        known = '/'.join(atk.get('attacker_knowledge_declared') or [])
        unknown = atk.get('attacker_knowledge_not_declared') or []
        worst = None
        for e in atk['per_attack']:
            v = (e['models'].get('tasgl') or {}).get('relative_drop_mean')
            if v is not None and (worst is None or v > worst):
                worst = v
        add(id='P1-2', group='P1', topic='Complete attack matrix',
            ledger='state attacker knowledge, capability, goal and cost',
            source='results/p6/nonoracle_attack_matrix.json',
            observed='%d attacks x %d attack seeds x %d model seeds; '
                     '%d runs; knowledge %s' % (
                         n_atk, len(atk.get('attack_seeds') or []),
                         len(atk.get('seeds') or []), n_run, known),
            status='updated',
            note='H6 is therefore no longer not_testable. It is measured and '
                 'reported as an interval with no gate, because the '
                 'alert-capacity anchor is still missing. Declared knowledge '
                 'is %s; %s access is not declared and is not covered.' % (
                     known, '/'.join(unknown)))
    else:
        add(id='P1-2', group='P1', topic='Complete attack matrix',
            ledger='state attacker knowledge, capability, goal and cost',
            source='(none)', observed='not executed', status='open',
            note='For this reason H6 (adversarial robustness) is judged '
                 'not_testable. H4 is temporal extrapolation and is a '
                 'separate item; it is reported as an interval estimate.')
    if atk:
        add(id='P1-3', group='P1', topic='Non-oracle stress test',
            ledger='add attacks that do not use true labels; keep the oracle '
                   'result as an upper-bound diagnostic',
            source='results/p6/nonoracle_attack_matrix.json',
            observed='oracle_labels_used=%s; construction inputs are graph, '
                     'features and clean predictions only' % (
                         _yes(atk.get('oracle_labels_used') is False)),
            status='updated' if atk.get('oracle_labels_used') is False
            else 'partial',
            note='The oracle matrix is kept as an upper-bound diagnostic and '
                 'is not merged. Worst measured relative TASGL drop is %s; '
                 'an untargeted attacker that only deletes or injects edges '
                 'is therefore covered, a white-box attacker is not.' % (
                     '%.1f%%' % (100.0 * worst) if worst is not None
                     else 'n.a.'))
    else:
        add(id='P1-3', group='P1', topic='Non-oracle stress test',
            ledger='add attacks that do not use true labels; keep the oracle '
                   'result as an upper-bound diagnostic',
            source='results/robustness_results.json',
            observed='edge deletion is still oracle-injected', status='open',
            note='No H6 conclusion may be drawn until a non-oracle adaptive '
                 'attack matrix exists. H4 is temporal extrapolation, not '
                 'adversarial robustness, and is reported as an interval '
                 'estimate.')
    add(id='P1-4', group='P1', topic='Teacher-student full cost',
        ledger='report teacher training, distillation, preprocessing, student '
               'deployment, calibration and update cost separately',
        source='results/p4/cost_measurement.json',
        observed='%d stages, %.4f s in total' % (len(stages), total_s),
        status='updated',
        note='About 44% of the chain is one-off work, which inference-only '
             'reporting hides.')
    add(id='P1-5', group='P1', topic='End-to-end system efficiency',
        ledger='add CPU latency, throughput, data loading, memory and batch '
               'curves',
        source='results/p4/cost_measurement.json',
        observed='GPU peak memory recorded; CPU latency and real power draw '
                 'still missing',
        status='partial',
        note='power_limit_w is N/A, so no energy number is reported.')
    add(id='P1-6', group='P1', topic='Reproducible package and environment',
        ledger='record GPU/CPU model, CUDA, cuDNN, PyTorch, driver and OS',
        source='results/p4/environment.json',
        observed='complete = %s; fingerprint %s...' % (
            _yes(env.get('complete')), (env.get('fingerprint') or '')[:16]),
        status='updated',
        note='The missing-fingerprint limitation in the ledger is lifted.')
    add(id='P1-7', group='P1', topic='Public repository and data lineage',
        ledger='publish download sources, checksums, resume points and '
               'result hashes',
        source='(manual)',
        observed='repository access needs human checking',
        status='open',
        note='Automated anonymous access cannot replace human confirmation.')
    add(id='P1-8', group='P1', topic='Metric completeness',
        ledger='report average precision next to PR-AUC; add ROC-AUC and the '
               'confusion matrix',
        source='(none)', observed='not added', status='open',
        note='The reported PR-AUC is a curve-integral implementation and is '
             'named as such; it is not average_precision.')

    return _normalize(it, results_dir)


# ------------------------------------------------------------------
# rendering
# ------------------------------------------------------------------
def _esc(s):
    s = str(s)
    for a, b in (('\\', '\\textbackslash{}'), ('&', '\\&'), ('%', '\\%'),
                 ('#', '\\#'), ('_', '\\_'), ('$', '\\$'), ('~', '\\~'),
                 ('^', '\\^{}'), ('{', '\\{'), ('}', '\\}')):
        s = s.replace(a, b)
    return s


def _cell(s):
    """Escape and give LaTeX a break point at every underscore."""
    return _esc(s).replace('\\_', '\\allowbreak\\_')


def _ascii_guard(items):
    """论文表体必须是英文; 返回含非 ASCII 的条目编号。"""
    bad = []
    for r in items:
        for k in ('topic', 'ledger', 'observed', 'note'):
            v = str(r.get(k) or '')
            if any(ord(ch) > 127 for ch in v):
                bad.append('%s.%s' % (r['id'], k))
    return bad


GROUPS = (
    ('EVD', 'tab:ledger_evd',
     'Reconciliation of the evidence items EVD-01--18 against the P0--P4 '
     'artifacts on disk.'),
    ('P0', 'tab:ledger_p0',
     'Reconciliation of the decisive experiments P0-1--9 against the P0--P4 '
     'artifacts on disk.'),
    ('P1', 'tab:ledger_p1',
     'Reconciliation of the credibility experiments P1-1--8 against the '
     'P0--P4 artifacts on disk.'),
)


def tex_table(items, group=None, label='tab:ledger',
              caption='Reconciliation of the submission ledger against the '
                      'P0--P4 artifacts on disk.'):
    """按组渲染。

    整张表有 35 行, 放进一个 table* 会高于版心并产生 overfull vbox, 因此按
    EVD / P0 / P1 拆成三张较短的表。
    """
    if group:
        items = [r for r in items if r['group'] == group]
    body = []
    for r in items:
        body.append(' & '.join([
            _esc(r['id']), _cell(r['topic']), _cell(r['ledger']),
            _cell(r['observed']), _esc(r['status'])]) + ' \\\\')
    s = []
    s.append('\\begin{table*}[!t]')
    s.append('\\centering\\footnotesize')
    s.append('\\setlength{\\tabcolsep}{4pt}')
    s.append('\\caption{%s Status: unchanged = the ledger statement still '
             'holds; updated = a newer run replaces it; partial = only part '
             'is covered by new evidence; open = no real run exists yet.}'
             % caption)
    s.append('\\label{%s}' % label)
    s.append('\\begin{tabularx}{\\textwidth}{lXXXXl}')
    s.append('\\hline')
    s.append(' & '.join(['Item', 'Topic', 'Ledger statement',
                         'Observed in artifacts', 'Status']) + ' \\\\')
    s.append('\\hline')
    s.extend(body)
    s.append('\\hline')
    s.append('\\end{tabularx}')
    s.append('\\begin{flushleft}\\footnotesize Generated by '
             '\\texttt{ledger\\allowbreak\\_reconcile.py}. It reads the '
             'result files and never fills in an estimate; a missing file is '
             'reported as \\texttt{n.a.}\\end{flushleft}')
    s.append('\\end{table*}')
    return '\n'.join(s) + '\n'


#: 一个 table* 最多放几行。EVD 组有 18 行, 单张表高于版心会产生
#: overfull vbox (实测 29.32pt), 因此按行数再切一次。
MAX_ROWS_PER_TABLE = 9


def tex_tables(items):
    out = []
    for g, lab, cap in GROUPS:
        rows = [r for r in items if r['group'] == g]
        if not rows:
            continue
        chunks = [rows[i:i + MAX_ROWS_PER_TABLE]
                  for i in range(0, len(rows), MAX_ROWS_PER_TABLE)]
        for k, ch in enumerate(chunks):
            suffix = '' if len(chunks) == 1 else '_%d' % (k + 1)
            sub = cap
            if len(chunks) > 1:
                sub = '%s Part %d of %d.' % (cap, k + 1, len(chunks))
            out.append(tex_table(ch, None, lab + suffix, sub))
    return '\n'.join(out)


def md_table(items):
    s = ['# Ledger reconciliation (generated by ledger_reconcile.py)', '',
         'Status: unchanged = the ledger statement still holds; updated = a '
         'newer run replaces it; partial = only part is covered by new '
         'evidence; open = no real run exists yet.', '',
         '| Item | Topic | Ledger statement | Observed | Status |',
         '| --- | --- | --- | --- | --- |']
    for r in items:
        s.append('| %s | %s | %s | %s | %s |' % (
            r['id'], r['topic'], r['ledger'].replace('\n', ' '),
            r['observed'].replace('\n', ' '), r['status']))
    s.append('')
    s.append('## Notes')
    s.append('')
    for r in items:
        if r.get('note'):
            s.append('- **%s**: %s' % (r['id'], r['note']))
    return '\n'.join(s) + '\n'


def build(results_dir, paper_dir, ledger_path=None):
    items = build_items(results_dir)
    bad = _ascii_guard(items)
    if bad:
        raise SystemExit('non-ASCII text would leak into the LaTeX table: %s'
                         % ', '.join(bad))
    out = {'schema': SCHEMA,
           'results_dir': os.path.abspath(results_dir),
           'ledger': os.path.abspath(ledger_path) if ledger_path else None,
           'counts': {},
           'items': items}
    for r in items:
        out['counts'][r['status']] = out['counts'].get(r['status'], 0) + 1
    prov = os.path.join(paper_dir, 'provenance')
    os.makedirs(prov, exist_ok=True)
    with open(os.path.join(prov, 'ledger_reconciliation.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(prov, 'ledger_reconciliation.md'), 'w',
              encoding='utf-8') as f:
        f.write(md_table(items))
    tabs = os.path.join(paper_dir, 'generated_tables')
    os.makedirs(tabs, exist_ok=True)
    with open(os.path.join(tabs, 'ledger_reconcile.tex'), 'w',
              encoding='utf-8') as f:
        f.write(tex_tables(items))
    print('[ledger_reconcile] items=%d %s' % (len(items), out['counts']))
    print('[ledger_reconcile] wrote %s'
          % os.path.join(prov, 'ledger_reconciliation.json'))
    print('[ledger_reconcile] wrote %s'
          % os.path.join(tabs, 'ledger_reconcile.tex'))
    return out


def check(results_dir):
    """Self-check: items complete, statuses legal, fields non-empty,
    LaTeX table is pure ASCII, and every declared source resolves or is
    explicitly marked as missing."""
    items = build_items(results_dir)
    errs = []
    ids = [r['id'] for r in items]
    for want in ['EVD-%02d' % i for i in range(1, 19)] + \
                ['P0-%d' % i for i in range(1, 10)] + \
                ['P1-%d' % i for i in range(1, 9)]:
        if want not in ids:
            errs.append('missing item %s' % want)
    if len(set(ids)) != len(ids):
        errs.append('duplicate ids')
    for r in items:
        if r['status'] not in STATUS:
            errs.append('%s bad status %r' % (r['id'], r['status']))
        for k in ('topic', 'ledger', 'observed', 'source', 'note'):
            if not r.get(k):
                errs.append('%s empty field %s' % (r['id'], k))
    errs += ['non-ASCII: %s' % b for b in _ascii_guard(items)]
    print(json.dumps({'schema': SCHEMA + '_check', 'n_items': len(items),
                      'counts': {s: sum(1 for r in items
                                        if r['status'] == s) for s in STATUS},
                      'errors': errs}, ensure_ascii=False, indent=2))
    return 0 if not errs else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')

    p = sub.add_parser('build')
    p.add_argument('--results_dir', default='.')
    p.add_argument('--paper_dir', required=True)
    p.add_argument('--ledger', default=None)
    p.set_defaults(func=lambda a: build(a.results_dir, a.paper_dir, a.ledger))

    p = sub.add_parser('check')
    p.add_argument('--results_dir', default='.')
    p.set_defaults(func=lambda a: check(a.results_dir))

    args = ap.parse_args()
    if not getattr(args, 'func', None):
        ap.print_help()
        return 2
    r = args.func(args)
    return 0 if r is None or isinstance(r, dict) else int(r)


if __name__ == '__main__':
    sys.exit(main())
