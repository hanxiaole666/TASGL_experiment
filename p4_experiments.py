"""p4_experiments.py — P4 终版证据层命令行入口

子命令
------
rootcause  run / gate-audit / dual-track   种子失败根因与门控收敛审计
rolling    plan / run                      严格 rolling-inductive 独立窗口
hpo        search / audit                  每个基线的等预算公平超参搜索
final      run                             预注册多种子最终比较 (含效应量)
calibration run                            正类 / 分类别校准
thresholds justify / register / audit      H1-H5 门限依据
env        fingerprint                     环境指纹
cost       chain / audit                   师生全链路成本
claims     adjudicate                      强断言降级裁决
stats      paired                          配对比较 + 效应量
report     all                             汇总

约定: 所有输出写入 results/p4/, 缺失数据一律写 *_missing 而非合成值。
"""
import argparse
import gc
import json
import os
import sys

import numpy as np

import p4_core
import p4_measure
import p4_thresholds

P4_TRACEABILITY = [
    {'gap': 'seed-0 0.3049 根因',
     'core': 'p4_core.decompose_forward_semantics / attribute_path / '
             'gate_convergence_report / classify_seed_failure',
     'cli': 'rootcause run', 'test': 'test_p4.py::TestRootCause'},
    {'gap': '门控未收敛种子的处理',
     'core': 'p4_core.run_admission / dual_track_analysis',
     'cli': 'rootcause gate-audit / dual-track',
     'test': 'test_p4.py::TestGateAdmission'},
    {'gap': '严格 rolling-inductive 独立窗口',
     'core': 'p4_core.strict_rolling_plan / induced_subgraph / '
             'fit_scaler_train_only / audit_no_future_leakage',
     'cli': 'rolling plan / rolling run', 'test': 'test_p4.py::TestRolling'},
    {'gap': '每基线等预算公平 HPO',
     'core': 'p2_core.audit_hpo_fairness', 'cli': 'hpo search / hpo audit',
     'test': 'test_p4.py::TestFairHPO'},
    {'gap': '效应量与显著性',
     'core': 'p4_core.paired_report', 'cli': 'stats paired / final run',
     'test': 'test_p4.py::TestPairedReport'},
    {'gap': '强断言降级',
     'core': 'p4_core.adjudicate_claim / weaken_claims',
     'cli': 'claims adjudicate', 'test': 'test_p4.py::TestClaims'},
    {'gap': 'H1-H5 门限依据',
     'core': 'p4_thresholds.build_justification_table / '
             'register_justified_thresholds',
     'cli': 'thresholds justify / register / audit',
     'test': 'test_p4.py::TestThresholdJustification'},
    {'gap': '师生全链路成本与环境指纹',
     'core': 'p4_measure.environment_fingerprint / full_chain_cost / '
             'audit_cost_completeness',
     'cli': 'env fingerprint / cost chain', 'test': 'test_p4.py::TestMeasure'},
    {'gap': '正类 / 分类别校准',
     'core': 'p4_core.classwise_calibration',
     'cli': 'calibration run', 'test': 'test_p4.py::TestCalibration'},
]

DEFAULT_HPO_SPACE = {
    'tasgl': [{'hidden': 48, 'dropout': 0.3, 'rank': 8},
              {'hidden': 32, 'dropout': 0.2, 'rank': 8},
              {'hidden': 64, 'dropout': 0.4, 'rank': 8},
              {'hidden': 48, 'dropout': 0.3, 'rank': 4},
              {'hidden': 48, 'dropout': 0.1, 'rank': 16},
              {'hidden': 32, 'dropout': 0.4, 'rank': 4}],
    'mlp': [{'hidden': 64, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.2},
            {'hidden': 128, 'dropout': 0.4}, {'hidden': 64, 'dropout': 0.1},
            {'hidden': 96, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.4}],
    'gcn': [{'hidden': 64, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.2},
            {'hidden': 128, 'dropout': 0.4}, {'hidden': 64, 'dropout': 0.1},
            {'hidden': 96, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.4}],
    'gat': [{'hidden': 64, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.2},
            {'hidden': 128, 'dropout': 0.4}, {'hidden': 64, 'dropout': 0.1},
            {'hidden': 96, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.4}],
    'graphsage': [{'hidden': 64, 'dropout': 0.3},
                  {'hidden': 32, 'dropout': 0.2},
                  {'hidden': 128, 'dropout': 0.4},
                  {'hidden': 64, 'dropout': 0.1},
                  {'hidden': 96, 'dropout': 0.3},
                  {'hidden': 32, 'dropout': 0.4}],
    'sgc': [{'hidden': 64, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.2},
            {'hidden': 128, 'dropout': 0.4}, {'hidden': 64, 'dropout': 0.1},
            {'hidden': 96, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.4}],
    'dsgat': [{'hidden': 64, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.2},
              {'hidden': 128, 'dropout': 0.4}, {'hidden': 64, 'dropout': 0.1},
              {'hidden': 96, 'dropout': 0.3}, {'hidden': 32, 'dropout': 0.4}],
}


# ------------------------------------------------------------------
# 基础设施
# ------------------------------------------------------------------
def _device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_elliptic(data_dir='./data/elliptic', device=None,
                   strict_scaler=False, visible_idx=None):
    """加载 Elliptic。strict_scaler=True 时只在 visible_idx 上拟合标准化器
    (严格 inductive 要求), 否则沿用现有全量拟合 (会泄漏未来节点的统计量)。"""
    import torch
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from data_loader import (load_elliptic as _le, compute_time_gaps,
                             make_graph_data, standard_split)
    if not strict_scaler:
        X, labels, ts, ei, _ = _le(data_dir)
    else:
        from download_data import ensure_elliptic, ELLIPTIC_FILES
        if not all(os.path.exists(os.path.join(data_dir, f))
                   for f in ELLIPTIC_FILES):
            ensure_elliptic(data_dir)
        feats = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv',
                            header=None)
        classes = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
        edges = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')
        tx_ids = feats.iloc[:, 0].values
        ts = feats.iloc[:, 1].values.astype(np.int64)
        Xr = feats.iloc[:, 2:].values.astype(np.float32)
        idx = visible_idx if visible_idx is not None else np.arange(len(ts))
        sc = StandardScaler().fit(Xr[idx])
        X = sc.transform(Xr).astype(np.float32)
        id2idx = {t: i for i, t in enumerate(tx_ids)}
        lm = {'1': 1, '2': 0, 'unknown': -1}
        labels = (classes['class'].astype(str).map(lm).fillna(-1)
                  .values.astype(np.int64))
        if 'txId' in classes.columns and \
                not (classes['txId'].values == tx_ids).all():
            al = np.full(len(tx_ids), -1, dtype=np.int64)
            for r in classes.itertuples():
                i = id2idx.get(getattr(r, 'txId', None))
                if i is not None:
                    al[i] = lm.get(str(getattr(r, 'class')), -1)
            labels = al
        src = edges['txId1'].map(id2idx).values.astype(np.float64)
        dst = edges['txId2'].map(id2idx).values.astype(np.float64)
        v = ~(np.isnan(src) | np.isnan(dst))
        ei = torch.tensor(np.stack([src[v].astype(np.int64),
                                    dst[v].astype(np.int64)]), dtype=torch.long)
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    data = make_graph_data(X, labels, ts, ei, device or _device())
    data['scaler_fit_scope'] = ('train_visible_only' if strict_scaler
                                else 'all_nodes_leaky')
    return data, ts, labels, ei


def _load_ledger(path=None):
    """加载已声明窗口台账。缺省使用 p2_window_ledger.json。"""
    p = path or 'p2_window_ledger.json'
    if not os.path.exists(p):
        return []
    try:
        obj = json.load(open(p, encoding='utf-8'))
    except Exception:
        return []
    if isinstance(obj, dict):
        w = obj.get('windows') or obj.get('ledger') or []
        if isinstance(w, dict):          # {name: {range: [a,b], ...}}
            obj = [dict(v, name=k) for k, v in w.items()]
        else:
            obj = w
    out = []
    for w in obj:
        if not isinstance(w, dict):
            continue
        win = w.get('window') or w.get('range') or \
            [w.get('start'), w.get('end')]
        if win and win[0] is not None:
            out.append({'name': w.get('name'), 'window': list(win),
                        'usage': w.get('used_for') or w.get('usage'),
                        'prior_exposure': w.get('prior_exposure')})
    return out


def _masks(ts, labels, train_end=34, val_start=30):
    import torch
    tr = (ts <= train_end) & (labels >= 0)
    va = (ts > 29) & (ts <= 34) & (labels >= 0)
    tr = tr & ~va
    te = (ts > 34) & (labels >= 0)
    dev = _device()
    return [torch.tensor(m).to(dev) for m in (tr, va, te)]


def _load_second(data_dir='./data/bitheteronet', train_end=34,
                 val_start=30, device=None, max_group=20):
    """加载 BitHeNet 并构建严格 inductive 数据字典。

    标准化器只在训练窗可见节点上拟合; 图不做时间裁剪 (该数据集的
    时间步是交易自身属性, 与 Elliptic 同构), 但训练/评估掩码按时间划分。
    """
    import torch
    import p4_bitheteronet as bh
    X, labels, ts, ei, meta = bh.load_bitheteronet(
        data_dir, max_group=max_group, verbose=True)
    vis = np.where((ts <= train_end) & (labels >= 0))[0]
    sc = p4_core.fit_scaler_train_only(X, vis)
    Xs = sc.transform(X).astype(np.float32)
    dev = device or _device()
    from data_loader import compute_time_gaps, make_graph_data
    data = make_graph_data(Xs, labels, ts,
                           torch.tensor(ei, dtype=torch.long), dev)
    data['scaler_fit_scope'] = 'train_visible_only'
    data['dataset_meta'] = meta
    return data, ts, labels, meta


def cmd_second_dataset(args):
    import torch
    from models import build_model
    from common import set_seed
    from metrics import eval_with_threshold, measure_efficiency
    if args.second_command == 'plan':
        data, ts, labels, meta = _load_second(args.data_dir, args.train_end)
        rec, (tr, va, te) = p4_core.strict_rolling_plan(
            ts, labels, args.train_end, args.val_end, args.test_end)
        from data_loader import load_elliptic
        X, elab, ets, _, _ = load_elliptic('./data/elliptic')
        al = p4_core_patch_alignment(meta, elab, ets)
        out = {'dataset': meta, 'split': rec, 'alignment': al}
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        p4_core.save_json(out, args.out or
                          'results/p4/second_dataset_plan.json')
        return 0
    if args.second_command == 'run':
        data, ts, labels, meta = _load_second(args.data_dir, args.train_end)
        masks = _masks(ts, labels, train_end=args.train_end)
        rec, (trm, vam, tem) = p4_core.strict_rolling_plan(
            ts, labels, args.train_end, args.val_end, args.test_end)
        res = {'schema': 'p4_second_dataset_v1', 'dataset': meta,
               'split': rec,
               'seeds': args.seeds, 'models': {}}
        # 配置来源必须随结果落盘, 否则读者无法判断"等预算"指什么。
        _sf = getattr(args, 'search', None)
        _exists = bool(_sf and os.path.exists(_sf))
        search = json.load(open(_sf, encoding='utf-8')) if _exists else {}
        res['config_source'] = {
            'search_file': _sf,
            'search_file_exists': _exists,
            'hpo_configs_applied': bool(search),
            'basis': ('等预算 = 三个模型使用同一份配置; 该数据集上未运行任何'
                      '超参搜索, 以免把它用于开发选择')}
        for name in args.models:
            rows = []
            kw = {}
            if search.get('models', {}).get(name):
                kw = search['models'][name]['best']['config']
            for s in args.seeds:
                set_seed(s)
                m = build_model(name, data['in_dim'], **kw)
                lf = None if name == 'tasgl' else False
                try:
                    m, tr = train_with_audit(m, data, masks,
                                             epochs=args.epochs, loss_fn=lf,
                                             min_epochs=args.min_epochs)
                    acc = eval_with_threshold(m, data, masks[1], masks[2])
                    eff = measure_efficiency(m, data, _device())
                    row = {'seed': s, 'status': 'ok',
                           'f1': float(acc['f1']),
                           'precision': float(acc['precision']),
                           'recall': float(acc['recall']),
                           'pr_auc': float(acc['pr_auc']),
                           'threshold': float(acc['threshold']),
                           'ece': float(acc.get('ece')) if acc.get('ece')
                           is not None else None,
                           'params': int(eff['params']),
                           'latency_ms': float(eff['latency_ms']),
                           'epochs': tr['epochs']}
                    if name == 'tasgl':
                        g = p4_core.gate_convergence_report(m, data)
                        row['gate'] = g
                        row['admission'] = p4_core.run_admission(g)
                    # 跨时间窗稳定性: 同一个已训练模型, 同一个验证集阈值,
                    # 只换测试窗。不重新训练, 也不逐窗重调阈值, 否则"稳定性"
                    # 会变成"每窗挑一个好阈值"。
                    if getattr(args, 'sub_windows', None):
                        win = {}
                        for spec in str(args.sub_windows).split(','):
                            try:
                                lo, hi = (int(x) for x in spec.split('-'))
                            except ValueError:
                                continue
                            wm = torch.tensor((ts > lo - 1) & (ts <= hi)
                                              & (labels >= 0)).to(_device())
                            if int(wm.sum()) == 0:
                                continue
                            w = eval_with_threshold(m, data, masks[1], wm)
                            win[f'{lo}-{hi}'] = {
                                'f1': float(w['f1']),
                                'precision': float(w['precision']),
                                'recall': float(w['recall']),
                                'threshold': float(w['threshold']),
                                'n_labeled': int(wm.sum()),
                                'n_pos': int(
                                    (labels[(ts > lo - 1) & (ts <= hi)]
                                     == 1).sum())}
                        if win:
                            row['windows'] = win
                except torch.cuda.OutOfMemoryError as e:
                    # 显存不足不是结果, 也不是"用更小的图重跑一次"的理由。
                    # 这里如实记录缺失种子, 由 complete 字段决定该模型能否
                    # 进入配对检验。
                    row = {'seed': s, 'status': 'oom', 'f1': None,
                           'error': str(e)[:300]}
                    print(f'[P4][2nd] {name} seed {s}: OOM, 记为缺失种子',
                          flush=True)
                rows.append(row)
                if row.get('status') == 'ok':
                    print(f'[P4][2nd] {name} seed {s}: '
                          f'F1={row["f1"]:.4f} thr={row["threshold"]:.2f}',
                          flush=True)
                # 逐种子释放显存。BitHeNet 有 202,804 个节点与 1,342,830 条
                # 无向边, 在 6 GB 设备上连续训练多个种子时, 缓存块累积会触发
                # 对比损失里的 OOM。
                m = None
                gc.collect()
                torch.cuda.empty_cache()
            ok = [r for r in rows if r.get('status', 'ok') == 'ok'
                  and r.get('f1') is not None]
            vals = [r['f1'] for r in ok]
            res['models'][name] = {
                'config': kw, 'rows': rows,
                'n_requested': len(args.seeds), 'n_complete': len(ok),
                'complete': len(ok) == len(args.seeds),
                'mean_f1': float(np.mean(vals)) if vals else None,
                'sd_f1': float(np.std(vals, ddof=1)) if len(vals) > 1
                else None}
        base = res['models'].get('tasgl')
        if base:
            res['paired'] = {}
            # 逐种子配对: 只取两个模型都完成的种子, 并按种子号排序。
            # 缺失种子不得用截断或补零的方式凑够样本量。
            base_ok = {r['seed']: r['f1'] for r in base['rows']
                       if r.get('status', 'ok') == 'ok'
                       and r.get('f1') is not None}
            for nm, mr in res['models'].items():
                if nm == 'tasgl':
                    continue
                cur = {r['seed']: r['f1'] for r in mr['rows']
                       if r.get('status', 'ok') == 'ok'
                       and r.get('f1') is not None}
                both = sorted(set(base_ok) & set(cur))
                a = [base_ok[s] for s in both]
                b = [cur[s] for s in both]
                if len(both) < 2:
                    res['paired'][nm] = {
                        'n_paired': len(both),
                        'status': 'incomplete',
                        'reason': '完成种子不足, 不做配对检验'}
                    continue
                rep = p4_core.paired_report(a, b)
                rep['n_paired'] = len(both)
                rep['paired_seeds'] = both
                rep['n_dropped'] = len(set(base_ok) | set(cur)) - len(both)
                res['paired'][nm] = rep
        res['evidence_class'] = 'confirmatory_independent'
        res['independence_basis'] = (
            'BitHeNet 在本项目中从未用于任何开发/选择/调参 '
            '(prior_exposure=false); 本结果为一次性锁定评估')
        p4_core.save_json(res, args.out or
                          'results/p4/second_dataset.json')
        print('saved', args.out or 'results/p4/second_dataset.json')
        return 0
    return 2


def p4_core_patch_alignment(meta, ell_labels, ell_ts):
    """BitHeNet 与 Elliptic 的语义对齐审计。"""
    import p4_bitheteronet as bh
    a = bh.elliptic_meta(ell_labels, ell_ts)
    b = bh.bitheteronet_meta(meta)
    rep = bh.semantic_alignment_report(a, b)
    rep['note'] = ('对齐发生在任务/标签/时间协议层面; 特征空间不做投影对齐, '
                   '因此该结果只能支持"同一方法在第二个语义对齐数据集上的'
                   '独立确认", 不能支持"跨数据集迁移"')
    return rep


def train_with_audit(model, data, masks, epochs=200, patience=30,
                     loss_fn=None, lr=1e-3, wd=1e-4, min_epochs=0):
    """复制 common.train_model 的逻辑, 但额外返回实际训练轮数
    (用于判断"是否在门控收敛前早停")。

    min_epochs>0 时, 早停在该轮数之前不生效 (P4 修复项 F1: 门控初始化使
    训练初期全部边活跃, 过早早停会留下"门控未收敛"的模型)。
    """
    import torch
    import torch.nn.functional as F
    from common import class_weights_from, temperature_schedule
    from metrics import evaluate
    from losses import TASGLLoss
    dev = _device()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    cw = class_weights_from(data['y'].cpu(), masks[0].cpu()).to(dev)
    is_t = loss_fn is None
    lf = TASGLLoss(class_weights=cw).to(dev) if is_t else None
    best, bad, n_ep, bs, hist = -1.0, 0, 0, None, []
    for ep in range(epochs):
        model.train()
        temp = temperature_schedule(ep, epochs) if is_t else 10.0
        if is_t:
            with torch.no_grad():
                _, ent = model.mc_uncertainty(
                    data['x'], data['edge_index'], data['time_gaps'],
                    data['num_nodes'], T=5)
            x_in = lf.dro.feature_perturb(data['x'], ent, masks[0])
            logits, aux = model(x_in, data['edge_index'], data['time_gaps'],
                                data['num_nodes'], temperature=temp)
            loss, parts = lf(logits, aux['h'], data['y'], ent, masks[0])
        else:
            logits, _ = model(data['x'], data['edge_index'],
                              data['time_gaps'], data['num_nodes'])
            loss = F.cross_entropy(logits[masks[0]], data['y'][masks[0]],
                                   weight=cw)
            parts = {'cls': float(loss.item()), 'con': 0.0}
        opt.zero_grad(); loss.backward(); opt.step()
        val = evaluate(model, data, masks[1])['f1']
        hist.append(float(val))
        if val > best:
            best, bad = val, 0
            bs = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}
        else:
            bad += 1
        n_ep = ep + 1
        if bad >= patience and n_ep >= min_epochs:
            break
    if bs is not None:
        model.load_state_dict(bs)
    return model, {'epochs': n_ep, 'best_val_f1': float(best),
                   'early_stopped': n_ep < epochs,
                   'min_epochs': min_epochs,
                   'val_history': hist}


# ------------------------------------------------------------------
# rootcause
# ------------------------------------------------------------------
def cmd_rootcause(args):
    import torch
    from models import TASGL
    from common import set_seed
    out = {'schema': p4_core.ROOTCAUSE_SCHEMA, 'seeds': args.seeds,
           'epochs': args.epochs, 'per_seed': []}
    data, ts, labels, ei = _load_elliptic(args.data_dir)
    masks = _masks(ts, labels)
    for s in args.seeds:
        set_seed(s)
        m = TASGL(data['in_dim']).to(_device())
        m, tr = train_with_audit(m, data, masks, epochs=args.epochs)
        gate = p4_core.gate_convergence_report(m, data)
        adm = p4_core.run_admission(gate)
        cells = p4_core.decompose_forward_semantics(
            m, data, masks, mc_T=args.mc_T,
            fixed_threshold=args.fixed_threshold)
        # 归因路径: 主协议(det+hard+验证调阈) -> 固定0.5 -> MC -> 软门控
        p_main = 'g=hard|s=deterministic|T=1|t=tuned_same'
        path = [p_main,
                'g=hard|s=deterministic|T=1|t=tuned_same',
                'g=hard|s=mc_dropout|T=%d|t=tuned_same' % args.mc_T,
                'g=soft|s=mc_dropout|T=%d|t=tuned_same' % args.mc_T]
        # 逐步路径需要在 fixed 阈值下比较, 因此重建路径
        fpath = ['g=hard|s=deterministic|T=1|t=tuned_same',
                 'g=hard|s=mc_dropout|T=%d|t=tuned_same' % args.mc_T,
                 'g=soft|s=mc_dropout|T=%d|t=tuned_same' % args.mc_T]
        attr = p4_core.attribute_path(
            {k: {'fixed': v['tuned_same']} for k, v in cells.items()},
            fpath, metric='f1')
        fixed_attr = p4_core.attribute_path(
            {k: {'fixed': v['fixed']} for k, v in cells.items()},
            fpath, metric='f1')
        rec = {'seed': s, 'train': tr, 'gate': gate, 'admission': adm,
               'cells': {k: {'tuned_threshold': v['tuned_threshold'],
                             'tuned_same': v['tuned_same'],
                             'fixed': v['fixed']}
                         for k, v in cells.items()},
               'attribution_tuned': attr,
               'attribution_fixed_0.5': fixed_attr}
        # 阈值因子: 同一语义下"验证集调阈"与"固定 0.5"之差
        det_key = 'g=hard|s=deterministic|T=1|t=tuned_same'
        rec['threshold_factor'] = {
            'tuned_f1': cells[det_key]['tuned_same']['f1'],
            'fixed_f1': cells[det_key]['fixed']['f1'],
            'delta': cells[det_key]['fixed']['f1']
            - cells[det_key]['tuned_same']['f1'],
            'tuned_threshold': cells[det_key]['tuned_threshold'],
            'note': '在确定性/硬掩码语义下单独分离阈值口径的贡献'},
        if args.reproduce is not None:
            rec['reproduce'] = p4_core.reproduce_check(
                cells, args.reproduce, tol=args.tol)
        out['per_seed'].append(rec)
        print(f'[P4][rootcause] seed {s}: epochs={tr["epochs"]} '
              f'early_stop={tr["early_stopped"]} '
              f'active={gate.get("min_active_edge_ratio")} '
              f'converged={gate.get("converged")} '
              f'F1(det/hard/tuned)='
              f'{cells[p_main]["tuned_same"]["f1"]:.4f}')
    out['dual_track'] = p4_core.dual_track_analysis(
        [{'f1': r['cells'][p4_core.semantics_key(
            p4_core.semantics_spec())]['tuned_same']['f1'],
          'admission': r['admission']} for r in out['per_seed']])
    p = args.out or 'results/p4/rootcause.json'
    p4_core.save_json(out, p)
    print('saved', p)
    return 0


def cmd_gate_audit(args):
    import torch
    from models import TASGL
    from common import set_seed
    data, ts, labels, ei = _load_elliptic(args.data_dir)
    masks = _masks(ts, labels)
    rows = []
    for s in args.seeds:
        set_seed(s)
        m = TASGL(data['in_dim']).to(_device())
        m, tr = train_with_audit(m, data, masks, epochs=args.epochs)
        g = p4_core.gate_convergence_report(m, data)
        rows.append({'seed': s, 'epochs': tr['epochs'],
                     'early_stopped': tr['early_stopped'],
                     'best_val_f1': tr['best_val_f1'],
                     'gate': g, 'admission': p4_core.run_admission(g)})
        print(f'[P4][gate] seed {s}: epochs={tr["epochs"]} '
              f'active={g.get("min_active_edge_ratio")} '
              f'w2={g.get("max_gate_w2_norm")} -> '
              f'{rows[-1]["admission"]["classification"]}')
    res = {'schema': 'p4_gate_audit_v1', 'rows': rows,
           'dual_track': None}
    p4_core.save_json(res, args.out or 'results/p4/gate_audit.json')
    return 0


# ------------------------------------------------------------------
# rolling
# ------------------------------------------------------------------
def cmd_rolling(args):
    if args.rolling_command == 'plan':
        data, ts, labels, ei = _load_elliptic(args.data_dir)
        rec, (tr, va, te) = p4_core.strict_rolling_plan(
            ts, labels, args.train_end, args.val_end, args.test_end)
        node_idx, sub_ei = p4_core.induced_subgraph(ei, ts, args.val_end)
        leak = p4_core.audit_no_future_leakage(
            node_idx, sub_ei, ts[node_idx], np.where(tr)[0], args.val_end,
            args.train_end, time_steps_global=ts)
        ledger = _load_ledger(args.ledger)
        expo = p4_core.window_exposure_audit(
            [args.val_end + 1, args.test_end], ledger)
        out = {'plan': rec, 'leakage': leak, 'exposure': expo,
               'n_nodes_induced': int(len(node_idx)),
               'n_edges_induced': int(sub_ei.shape[1])}
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        p4_core.save_json(out, args.out or 'results/p4/rolling_plan.json')
        return 0
    if args.rolling_command == 'run':
        import torch
        from models import TASGL
        from common import set_seed
        data, ts, labels, ei = _load_elliptic(
            args.data_dir, strict_scaler=True,
            visible_idx=np.where((ts <= args.train_end) & (labels >= 0))[0])
        rec, (tr, va, te) = p4_core.strict_rolling_plan(
            ts, labels, args.train_end, args.val_end, args.test_end)
        node_idx, sub_ei = p4_core.induced_subgraph(ei, ts, args.val_end)
        leak = p4_core.audit_no_future_leakage(
            node_idx, sub_ei, ts[node_idx],
            np.where(tr)[0], args.val_end, args.train_end)
        # 在诱导子图上重排索引
        remap = -np.ones(len(ts), dtype=np.int64)
        remap[node_idx] = np.arange(len(node_idx))
        l_ts = ts[node_idx]
        lX = data['x'].cpu().numpy()[node_idx]
        ly = labels[node_idx]
        ltr = (l_ts <= args.train_end) & (ly >= 0)
        lva = (l_ts > args.train_end) & (l_ts <= args.val_end) & (ly >= 0)
        lte = (l_ts > args.val_end) & (ly >= 0)
        ldata = {
            'x': torch.tensor(lX, dtype=torch.float32).to(_device()),
            'y': torch.tensor(ly, dtype=torch.long).to(_device()),
            'edge_index': torch.tensor(sub_ei, dtype=torch.long).to(_device()),
            'time_gaps': torch.tensor(
                np.abs(l_ts[sub_ei[0]] - l_ts[sub_ei[1]]).astype(np.float32)
            ).to(_device()),
            'time_steps': torch.tensor(l_ts, dtype=torch.long).to(_device()),
            'num_nodes': int(len(node_idx)), 'in_dim': int(lX.shape[1]),
        }
        lm = [torch.tensor(m).to(_device()) for m in (ltr, lva, lte)]
        rows = []
        for s in args.seeds:
            set_seed(s)
            m = TASGL(ldata['in_dim']).to(_device())
            m, trn = train_with_audit(m, ldata, lm, epochs=args.epochs)
            gate = p4_core.gate_convergence_report(m, ldata)
            cells = p4_core.decompose_forward_semantics(m, ldata, lm,
                                                        mc_T=args.mc_T)
            key = p4_core.semantics_key(p4_core.semantics_spec())
            rows.append({'seed': s, 'train': trn, 'gate': gate,
                         'admission': p4_core.run_admission(gate),
                         'metrics': cells[key]['tuned_same']})
            print(f'[P4][rolling] seed {s}: '
                  f'F1={cells[key]["tuned_same"]["f1"]:.4f} '
                  f'active={gate.get("min_active_edge_ratio")}')
        out = {'schema': p4_core.ROLLING_SCHEMA, 'plan': rec,
               'leakage': leak, 'scaler': data.get('scaler_fit_scope'),
               'seeds': args.seeds, 'rows': rows,
               'aggregate': p4_core.paired_report(
                   [r['metrics']['f1'] for r in rows],
                   [0.0] * len(rows)) if len(rows) >= 2 else None}
        p4_core.save_json(out, args.out or 'results/p4/rolling_run.json')
        print('saved', args.out or 'results/p4/rolling_run.json')
        return 0
    print('未知 rolling 子命令')
    return 2


# ------------------------------------------------------------------
# hpo
# ------------------------------------------------------------------
def cmd_hpo(args):
    import torch
    from models import build_model
    from common import set_seed
    from metrics import eval_with_threshold
    if args.hpo_command == 'search':
        data, ts, labels, ei = _load_elliptic(args.data_dir)
        masks = _masks(ts, labels)
        space = DEFAULT_HPO_SPACE
        res = {'schema': 'p4_fair_hpo_v1', 'seeds': args.seeds,
               'epochs': args.epochs, 'models': {}}
        for model_name in args.models:
            cand = space.get(model_name, [{}])[:args.trials]
            rows = []
            for ci, kw in enumerate(cand):
                vals, params = [], None
                for s in args.seeds:
                    set_seed(s)
                    m = build_model(model_name, data['in_dim'], **kw)
                    lf = None if model_name == 'tasgl' else False
                    m, tr = train_with_audit(m, data, masks,
                                             epochs=args.epochs,
                                             loss_fn=lf)
                    acc = eval_with_threshold(m, data, masks[1], masks[2])
                    vals.append(float(acc['f1']))
                    params = sum(p.numel() for p in m.parameters()
                                 if p.requires_grad)
                rows.append({'trial': ci, 'config': kw,
                             'val_f1_seeds': vals,
                             'mean_val_f1': float(np.mean(vals)),
                             'params': int(params or 0)})
                print(f'[P4][hpo] {model_name} trial {ci} {kw} '
                      f'val_f1={np.mean(vals):.4f}')
            rows.sort(key=lambda r: (-r['mean_val_f1'], r['params']))
            res['models'][model_name] = {
                'n_trials': len(rows), 'n_seeds': len(args.seeds),
                'seeds': [int(s) for s in args.seeds],
                'epochs': int(args.epochs),
                'selection_metric': 'validation_f1_tuned_on_validation',
                'trajectory': rows, 'best': rows[0]}
        p = args.out or 'results/p4/hpo_search.json'
        p4_core.save_json(res, p)
        print('saved', p)
        return 0
    if args.hpo_command == 'audit':
        from p2_core import audit_hpo_fairness
        spec = json.load(open(args.budget, encoding='utf-8'))
        srch = json.load(open(args.search, encoding='utf-8'))
        # 把搜索结果整理成 p2_core.audit_hpo_fairness 期望的形状
        models = {}
        for name, blk in srch.get('models', {}).items():
            seeds = blk.get('seeds', []) or \
                list(range(int(blk.get('n_seeds', 0))))
            models[name] = {
                'trials_per_seed': int(blk.get('n_trials', -1)),
                'seeds': [int(s) for s in seeds],
                'selection': spec.get('selection'),
                'by_seed': {str(s): {
                    'train_window': spec.get('train_window'),
                    'validation_window': spec.get('validation_window'),
                    'test_labels_used': False} for s in seeds},
                'best_config': blk.get('best', {}).get('config'),
                'best_mean_val_f1': blk.get('best', {}).get('mean_val_f1'),
                'trajectory': blk.get('trajectory', []),
            }
        rep = audit_hpo_fairness({'models': models}, spec)
        rep['declared_models_missing'] = [
            m for m in spec.get('models', [])
            if m not in models]
        print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
        p4_core.save_json(rep, args.out or 'results/p4/hpo_audit.json')
        return 0 if rep.get('fair') else 2
    return 2


# ------------------------------------------------------------------
# final (预注册多种子比较)
# ------------------------------------------------------------------
def cmd_final(args):
    import torch
    from models import build_model
    from common import set_seed
    from metrics import eval_with_threshold, measure_efficiency
    data, ts, labels, ei = _load_elliptic(args.data_dir)
    masks = _masks(ts, labels)
    search = json.load(open(args.search, encoding='utf-8')) \
        if args.search and os.path.exists(args.search) else {}
    res = {'schema': 'p4_final_comparison_v1', 'seeds': args.seeds,
           'models': {}}
    for model_name in args.models:
        kw = {}
        if search.get('models', {}).get(model_name):
            kw = search['models'][model_name]['best']['config']
        rows = []
        for s in args.seeds:
            set_seed(s)
            m = build_model(model_name, data['in_dim'], **kw)
            lf = None if model_name == 'tasgl' else False
            m, tr = train_with_audit(m, data, masks, epochs=args.epochs,
                                     loss_fn=lf,
                                     min_epochs=args.min_epochs)
            acc = eval_with_threshold(m, data, masks[1], masks[2])
            eff = measure_efficiency(m, data, _device())
            row = {'seed': s, 'f1': float(acc['f1']),
                   'precision': float(acc['precision']),
                   'recall': float(acc['recall']),
                   'threshold': float(acc['threshold']),
                   'params': int(eff['params']),
                   'latency_ms': float(eff['latency_ms']),
                   'epochs': tr['epochs'],
                   'early_stopped': tr['early_stopped']}
            if model_name == 'tasgl':
                g = p4_core.gate_convergence_report(m, data)
                row['gate'] = g
                row['admission'] = p4_core.run_admission(g)
            rows.append(row)
        res['models'][model_name] = {'config': kw, 'rows': rows,
                                     'mean_f1': float(
                                         np.mean([r['f1'] for r in rows])),
                                     'sd_f1': float(np.std(
                                         [r['f1'] for r in rows], ddof=1))
                                     if len(rows) > 1 else None}
        print(f'[P4][final] {model_name}: F1='
              f'{res["models"][model_name]["mean_f1"]:.4f}'
              f'±{res["models"][model_name]["sd_f1"] or 0:.4f}')
    # TASGL vs 每个基线的配对比较 (含效应量)
    base = res['models'].get('tasgl')
    if base:
        res['paired'] = {}
        for name, mres in res['models'].items():
            if name == 'tasgl':
                continue
            a = [r['f1'] for r in base['rows']]
            b = [r['f1'] for r in mres['rows']]
            n = min(len(a), len(b))
            res['paired'][name] = p4_core.paired_report(a[:n], b[:n])
    p4_core.save_json(res, args.out or 'results/p4/final_comparison.json')
    print('saved', args.out or 'results/p4/final_comparison.json')
    return 0


# ------------------------------------------------------------------
# 其余子命令
# ------------------------------------------------------------------
def cmd_calibration(args):
    rec = json.load(open(args.predictions, encoding='utf-8'))
    prob = np.asarray(rec['prob'], dtype=float)
    y = np.asarray(rec['labels'], dtype=int)
    out = p4_core.classwise_calibration(prob, y, args.bins)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    p4_core.save_json(out, args.out or 'results/p4/calibration.json')
    return 0


def cmd_thresholds(args):
    if args.thr_command == 'justify':
        a = json.load(open(args.anchors, encoding='utf-8')) \
            if args.anchors else None
        t = p4_thresholds.build_justification_table(a)
        print(json.dumps(t, ensure_ascii=False, indent=2))
        p4_core.save_json(t, args.out or
                          'results/p4/threshold_justification.json')
        return 0
    if args.thr_command == 'register':
        a = json.load(open(args.anchors, encoding='utf-8')) \
            if args.anchors else {}
        r = p4_thresholds.register_justified_thresholds(
            a, out_dir=args.out_dir, declared_seeds=list(range(args.n_seeds)))
        print('lock_id', r['lock_id'])
        p4_core.save_json(r, os.path.join(args.out_dir, 'register.json'))
        return 0
    if args.thr_command == 'audit':
        t = json.load(open(args.table, encoding='utf-8'))
        rep = p4_thresholds.audit_justification(t)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        p4_core.save_json(rep, args.out or 'results/p4/threshold_audit.json')
        return 0
    return 2


def cmd_env(args):
    e = p4_measure.environment_fingerprint()
    print(json.dumps(e, ensure_ascii=False, indent=2))
    p4_measure.save_json(e, args.out or 'results/p4/environment.json')
    return 0


def _cost_measure(args):
    """真实测量师生全链路八阶段成本 (不做任何合成填充)。"""
    import time
    import torch
    from models import build_model
    from common import set_seed
    from metrics import eval_with_threshold, measure_efficiency
    from v3_engine import get_teacher_logits

    env = p4_measure.environment_fingerprint()
    sample = p4_measure.sample_gpu_power_mw() if args.energy else None
    prot = p4_measure.measurement_protocol(
        semantics=args.semantics or 'full_graph_forward',
        n_warmup=args.warmup, n_repeat=args.repeat,
        note='batch=整图单次前向; 与单笔交易口径需换算')
    stages = []

    def _timed(stage, fn):
        """计时 (并按需积分功耗) 的一次性阶段。

        旧实现把前三个阶段的 energy_j 硬编码为 None, 于是"能耗未实测"这条
        阻断理由永远消不掉。这里改成与后续阶段同一条测量路径。
        """
        r = p4_measure.measure_stage(stage, fn, repeats=1, n_warmup=0,
                                     protocol=prot, sample_power=sample,
                                     device=_device())
        return r

    _box = {}

    def _prep():
        _box['d'] = _load_elliptic(args.data_dir)
        return _box['d']
    stages.append(_timed('preprocessing', _prep))
    data, ts, labels, ei = _box['d']
    masks = _masks(ts, labels)

    def _build():
        return p4_core.induced_subgraph(ei, ts, 34)
    stages.append(_timed('graph_construction', _build))

    # --- 教师训练 (真实: v3_engine.get_teacher_logits) ---
    set_seed(0)
    os.makedirs('results/p4/cache', exist_ok=True)

    def _teacher():
        return get_teacher_logits(
            args.teacher, data, masks, _device(), 0, args.epochs,
            'results/p4/cache/teacher_logits.pt',
            'results/p4/cache/teacher.ckpt',
            {'teacher': args.teacher, 'epochs': args.epochs, 'seed': 0},
            overwrite=True)
    st_teacher = _timed('teacher_training', _teacher)
    teach = st_teacher.pop('out', None)
    st_teacher['teacher_model'] = args.teacher
    stages.append(st_teacher)

    # --- 学生训练 (含蒸馏) ---
    set_seed(0)
    student = build_model('tasgl', data['in_dim'])
    tl = teach.to(_device()) if torch.is_tensor(teach) else teach

    def _student_train():
        set_seed(0)
        m = build_model('tasgl', data['in_dim']).to(_device())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        import torch.nn.functional as F
        from losses import TASGLLoss
        from common import class_weights_from
        cw = class_weights_from(data['y'].cpu(), masks[0].cpu()).to(_device())
        lf = TASGLLoss(class_weights=cw).to(_device())
        for ep in range(args.epochs):
            m.train()
            with torch.no_grad():
                _, ent = m.mc_uncertainty(data['x'], data['edge_index'],
                                          data['time_gaps'],
                                          data['num_nodes'], T=5)
            x_in = lf.dro.feature_perturb(data['x'], ent, masks[0])
            logits, aux = m(x_in, data['edge_index'], data['time_gaps'],
                            data['num_nodes'])
            loss, _ = lf(logits, aux['h'], data['y'], ent, masks[0])
            loss = loss + args.kd_weight * F.kl_div(
                F.log_softmax(logits / 2.0, dim=-1),
                torch.softmax(tl.to(_device()) / 2.0, dim=-1),
                reduction='batchmean') * 4.0
            opt.zero_grad(); loss.backward(); opt.step()
        return m
    st = p4_measure.measure_stage('distillation', _student_train, repeats=1,
                                  n_warmup=0, protocol=prot,
                                  sample_power=sample)
    stages.append(st)
    student = _student_train()

    # --- 推理 ---
    inf = p4_measure.measure_stage(
        'inference',
        lambda: student(data['x'], data['edge_index'], data['time_gaps'],
                        data['num_nodes'], hard_mask=True),
        repeats=args.repeat, n_warmup=args.warmup, protocol=prot,
        sample_power=sample)
    stages.append(inf)

    # --- 阈值选择 ---
    def _thr():
        from metrics import tune_threshold
        tune_threshold(student, data, masks[1])
    stages.append(p4_measure.measure_stage('threshold_selection', _thr,
                                           repeats=1, n_warmup=0,
                                           protocol=prot,
                                           sample_power=sample))

    # --- 校准拟合 (温度缩放, 真实拟合) ---
    def _cal():
        from p2_core import fit_stratified_temperature
        with torch.no_grad():
            lg, _ = student(data['x'], data['edge_index'], data['time_gaps'],
                            data['num_nodes'], hard_mask=True)
            p = torch.softmax(lg, -1)[masks[1]].cpu().numpy()
        fit_stratified_temperature(
            p, data['y'][masks[1]].cpu().numpy(),
            np.zeros(len(p), dtype=int))
    try:
        stages.append(p4_measure.measure_stage(
            'calibration', _cal, repeats=1, n_warmup=0, protocol=prot,
            sample_power=sample))
    except Exception as e:
        stages.append({'stage': 'calibration', 'seconds': None,
                       'total_seconds': 0.0, 'energy_j': None,
                       'energy_measured': False, 'measured': False,
                       'reason': f'校准阶段测量失败: {e}',
                       'label': p4_measure.STAGE_LABEL['calibration']})

    # --- 增量更新 (用一半轮数近似一次再训练) ---
    def _upd():
        m = build_model('tasgl', data['in_dim']).to(_device())
        train_with_audit(m, data, masks, epochs=max(args.epochs // 2, 1))
    stages.append(p4_measure.measure_stage('update', _upd, repeats=1,
                                           n_warmup=0, protocol=prot,
                                           sample_power=sample))

    # 阶段字典里的 'out' 是被测函数的返回值 (可能是张量), 不写进结果文件
    for s in stages:
        s.pop('out', None)

    eff = measure_efficiency(student, data, _device())
    chain = p4_measure.full_chain_cost(
        stages, amortize_nodes=data['num_nodes'],
        teacher_lifetime_updates=args.updates)
    audit = p4_measure.audit_cost_completeness(chain, prot, env)
    out = {'schema': 'p4_cost_measurement_v1', 'environment': env,
           'protocol': prot, 'stages': stages, 'chain': chain,
           'audit': audit, 'deploy_efficiency': eff,
           'kd_weight': args.kd_weight, 'teacher_model': args.teacher}
    p4_measure.save_json(out, args.out or 'results/p4/cost_measurement.json')
    print(json.dumps({'chain': chain, 'audit': audit}, ensure_ascii=False,
                     indent=2, default=str))
    return 0


def cmd_cost(args):
    if args.cost_command == 'measure':
        return _cost_measure(args)
    if args.cost_command == 'chain':
        if not args.stages:
            print('cost chain 需要 --stages <阶段列表 JSON>')
            return 2
        st = json.load(open(args.stages, encoding='utf-8'))
        chain = p4_measure.full_chain_cost(
            st, amortize_nodes=args.nodes,
            teacher_lifetime_updates=args.updates)
        prot = p4_measure.measurement_protocol(
            semantics=args.semantics, n_warmup=args.warmup,
            n_repeat=args.repeat) if args.semantics else None
        env = None
        if os.path.exists(args.env or ''):
            env = json.load(open(args.env, encoding='utf-8'))
        audit = p4_measure.audit_cost_completeness(chain, prot, env)
        print(json.dumps({'chain': chain, 'protocol': prot, 'audit': audit},
                         ensure_ascii=False, indent=2))
        p4_measure.save_json({'chain': chain, 'protocol': prot,
                              'audit': audit},
                             args.out or 'results/p4/cost_chain.json')
        return 0
    return 2


def cmd_claims(args):
    claims = json.load(open(args.claims, encoding='utf-8'))
    rep = p4_core.weaken_claims(claims)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    p4_core.save_json(rep, args.out or 'results/p4/claims.json')
    return 0


def cmd_stats(args):
    d = json.load(open(args.input, encoding='utf-8'))
    rep = p4_core.paired_report(d['a'], d['b'], alpha=args.alpha)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    p4_core.save_json(rep, args.out or 'results/p4/paired.json')
    return 0


def cmd_report(args):
    rep = {'schema': 'p4_report_v1', 'traceability': P4_TRACEABILITY}
    p4_core.save_json(rep, args.out or 'results/p4/report.json')
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog='p4_experiments')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('rootcause')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--mc_T', type=int, default=10)
    p.add_argument('--fixed_threshold', type=float, default=0.5)
    p.add_argument('--reproduce', type=float, default=None,
                   help='待复现的历史观测值, 例如 0.3049')
    p.add_argument('--tol', type=float, default=0.01)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_rootcause)

    p = sub.add_parser('gate-audit')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_gate_audit)

    p = sub.add_parser('second-dataset')
    p.add_argument('second_command', choices=['plan', 'run'])
    p.add_argument('--data_dir', default='./data/bitheteronet')
    p.add_argument('--models', nargs='+', default=['tasgl', 'graphsage', 'mlp'])
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--train_end', type=int, default=29)
    p.add_argument('--val_end', type=int, default=34)
    p.add_argument('--test_end', type=int, default=49)
    p.add_argument('--min_epochs', type=int, default=0)
    p.add_argument('--search', default='results/p4/hpo_search.json')
    p.add_argument('--sub_windows', default=None,
                   help=('额外测试窗, 形如 35-39,40-44,45-49; 用于跨时间窗稳定性。'
                         '同一训练模型与同一验证阈值, 只换测试窗'))
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_second_dataset)

    p = sub.add_parser('rolling')
    p.add_argument('rolling_command', choices=['plan', 'run'])
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--train_end', type=int, default=25)
    p.add_argument('--val_end', type=int, default=34)
    p.add_argument('--test_end', type=int, default=49)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--mc_T', type=int, default=10)
    p.add_argument('--ledger', default=None)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_rolling)

    p = sub.add_parser('hpo')
    p.add_argument('hpo_command', choices=['search', 'audit'])
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--models', nargs='+',
                   default=['tasgl', 'mlp', 'gcn', 'gat', 'graphsage'])
    p.add_argument('--trials', type=int, default=6)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--search', default='results/p4/hpo_search.json')
    p.add_argument('--budget', default='p2_hpo_budget.json')
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_hpo)

    p = sub.add_parser('final')
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--models', nargs='+', default=['tasgl', 'graphsage', 'mlp'])
    p.add_argument('--seeds', type=int, nargs='+',
                   default=list(range(10)))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--min_epochs', type=int, default=0,
                   help='早停保护: 早停在该轮数前不生效 (修复项 F1)')
    p.add_argument('--search', default='results/p4/hpo_search.json')
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_final)

    p = sub.add_parser('calibration')
    p.add_argument('--predictions', required=True)
    p.add_argument('--bins', type=int, default=15)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser('thresholds')
    p.add_argument('thr_command', choices=['justify', 'register', 'audit'])
    p.add_argument('--anchors', default=None)
    p.add_argument('--table', default=None)
    p.add_argument('--out_dir', default='results/p4/thresholds')
    p.add_argument('--n_seeds', type=int, default=20)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_thresholds)

    p = sub.add_parser('env')
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_env)

    p = sub.add_parser('cost')
    p.add_argument('cost_command', choices=['chain', 'measure'])
    p.add_argument('--data_dir', default='./data/elliptic')
    p.add_argument('--teacher', default='graphsage')
    p.add_argument('--kd_weight', type=float, default=0.5)
    p.add_argument('--energy', action='store_true')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--stages', default=None,
                   help='仅 cost chain 需要: 已实测阶段列表的 JSON')
    p.add_argument('--nodes', type=int, default=None)
    p.add_argument('--updates', type=int, default=1)
    p.add_argument('--semantics', default=None,
                   choices=list(p4_measure.MEASUREMENT_SEMANTICS))
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeat', type=int, default=100)
    p.add_argument('--env', default=None)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser('claims')
    p.add_argument('--claims', required=True)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_claims)

    p = sub.add_parser('stats')
    p.add_argument('--input', required=True)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser('report')
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
