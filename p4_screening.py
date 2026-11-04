"""p4_screening.py — H2 语义匹配的逐交易筛查延迟与能耗实测

背景
----
已注册的 H2 门限是 "per_node_screening 口径下延迟 <= 142.857 ms (目标 QPS=7)"。
但既有测量只有 `full_graph_forward`（整图一次前向），与门限不是一个口径，
因此 H2 一直只能判为 not_decidable。本模块补齐缺失的那一半：真正的
"一次来一笔/一批交易，为它建局部子图并前向"的测量。

测量口径（全部写进结果文件，缺一不可）
------------------------------------
- scope        : 逐交易筛查（不是整图一次前向）
- batch_size   : 同一批一起筛查的交易数
- hops         : 建子图时向外扩展的跳数（K=0 的模型不需要子图）
- include_build: 计时是否包含子图构建（两档都报，因为部署形态不同）
- n_warmup     : 预热次数（不计入）
- n_repeat     : 计时重复次数
- sync         : 是否每次 torch.cuda.synchronize()
- precision    : fp32
- power        : nvidia-smi power.draw 采样积分得到的能耗（测不到就写 None）

禁止行为
--------
- 不把整图延迟除以节点数冒充逐交易延迟（那个数另行标注为 amortized）。
- 测不到能耗时写 None，不填合成值。
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

SCHEMA = 'p4_per_node_screening_v1'
H2_BUDGET_MS = 1000.0 / 7.0          # 目标 QPS=7 -> 142.857 ms/笔
K_HOPS = {'tasgl': 2, 'graphsage': 2, 'gcn': 2, 'gat': 2, 'mlp': 0, 'sgc': 2,
          'dsgat': 2}


# ------------------------------------------------------------------
# 子图构建
# ------------------------------------------------------------------
def _hop_expand(edge_index, seeds, hops, num_nodes):
    """从 seeds 向外扩展 hops 跳, 返回涉及到的节点集合。"""
    row, col = edge_index
    cur = seeds
    for _ in range(int(hops)):
        if cur.numel() == 0:
            break
        mask = torch.isin(row, cur)
        nxt = torch.unique(col[mask])
        cur = torch.unique(torch.cat([cur, nxt]))
    return cur


def ego_subgraph(data, targets, hops):
    """为目标节点构建 ego 子图。

    返回 (x_sub, edge_index_sub, dt_sub, n_sub, build_seconds)。
    hops=0 时直接返回目标节点自身, 无边。
    """
    x = data['x']
    edge_index = data['edge_index']
    tg = data.get('time_gaps')
    num_nodes = x.size(0)
    targets = torch.as_tensor(targets, dtype=torch.long, device=x.device)
    if int(hops) <= 0:
        return (x[targets],
                torch.empty((2, 0), dtype=torch.long, device=x.device),
                (tg.new_zeros(0) if tg is not None else None),
                int(targets.numel()), 0.0)
    t0 = time.perf_counter()
    node_set = _hop_expand(edge_index, targets, hops, num_nodes)
    # 节点重编号
    idx = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
    idx[node_set] = torch.arange(node_set.numel(), device=x.device)
    row, col = edge_index
    m = idx[row] >= 0
    m = m & (idx[col] >= 0)
    sub = torch.stack([idx[row[m]], idx[col[m]]], dim=0)
    dt_sub = tg[m] if tg is not None else None
    elapsed = time.perf_counter() - t0
    return x[node_set], sub, dt_sub, int(node_set.numel()), elapsed


def _forward(model, x_sub, edge_index_sub, dt_sub=None):
    """调用模型前向。

    基线签名是 (x, edge_index, dt=None, num_nodes=None);
    TASGL 需要 (x, edge_index, dt, num_nodes)。统一按最长签名调用。
    """
    return model(x_sub, edge_index_sub, dt_sub, x_sub.size(0))


# ------------------------------------------------------------------
# 单次筛查计时
# ------------------------------------------------------------------
def screen_once(model, data, targets, hops, include_build):
    """筛查一批目标节点, 返回 (耗时秒, 子图构建秒, 子图节点数)。

    include_build=True 时计时包含子图构建, 模拟"来一笔就现建子图"的部署;
    False 时只计前向, 模拟子图已经物化的部署。两种都报。
    """
    if include_build:
        t0 = time.perf_counter()
        x_sub, ei, dt_sub, n_sub, build_s = ego_subgraph(data, targets, hops)
        if x_sub.is_cuda:
            torch.cuda.synchronize()
        with torch.no_grad():
            _forward(model, x_sub, ei, dt_sub)
        if x_sub.is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - t0, build_s, n_sub
    x_sub, ei, dt_sub, n_sub, build_s = ego_subgraph(data, targets, hops)
    if x_sub.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _forward(model, x_sub, ei, dt_sub)
    if x_sub.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0, build_s, n_sub


def measure_screening(model, data, node_pool, hops, batch_sizes,
                      n_warmup=20, n_repeat=200, include_build=False,
                      sample_power=None, seed=0):
    """对每个 batch_size 测一次逐交易筛查延迟。

    返回 {batch_size: {ms_per_tx, ms_per_batch, subgraph_nodes_mean, ...}}。
    """
    rng = np.random.default_rng(seed)
    model.eval()
    out = {}
    for bs in batch_sizes:
        # 预热 + 计时都用同一批抽样方式, 保证口径一致
        def _draw(i):
            idx = rng.choice(node_pool, size=int(bs), replace=False)
            return torch.as_tensor(idx, dtype=torch.long,
                                   device=data['x'].device)
        for i in range(n_warmup):
            screen_once(model, data, _draw(i), hops, include_build)
        times = []
        builds = []
        sizes = []
        power_w = []
        if sample_power is not None:
            def _poll():
                v = sample_power()
                if v is not None:
                    power_w.append(v)
        for i in range(n_repeat):
            t, b, n_sub = screen_once(model, data, _draw(n_warmup + i), hops,
                                      include_build)
            times.append(t)
            builds.append(b)
            sizes.append(n_sub)
            if sample_power is not None:
                _poll()
        times = np.asarray(times, dtype=float) * 1000.0     # ms
        builds = np.asarray(builds, dtype=float) * 1000.0
        out[int(bs)] = {
            'batch_size': int(bs),
            'ms_per_batch_mean': float(times.mean()),
            'ms_per_batch_sd': float(times.std(ddof=1)) if len(times) > 1
            else None,
            'ms_per_batch_p50': float(np.percentile(times, 50)),
            'ms_per_batch_p95': float(np.percentile(times, 95)),
            'ms_per_tx_mean': float(times.mean() / bs),
            'ms_per_tx_p95': float(np.percentile(times, 95) / bs),
            'subgraph_build_ms_mean': float(builds.mean()),
            'subgraph_nodes_mean': (float(np.mean(sizes)) if sizes else None),
            'n_repeat': int(n_repeat),
            'n_warmup': int(n_warmup),
            'include_build': bool(include_build),
            'mean_power_w': (sum(power_w) / len(power_w)) if power_w else None,
            'energy_j_per_batch': (
                (sum(power_w) / len(power_w)) * (times.mean() / 1000.0))
            if power_w else None,
        }
    return out


# ------------------------------------------------------------------
# 汇总与 H2 判定
# ------------------------------------------------------------------
def h2_verdict(per_model, budget_ms=H2_BUDGET_MS, primary_bs=1):
    """用逐交易口径判 H2。

    primary_bs 是判定所用的批大小。默认 1, 理由写在结果里: 门限来自
    "每秒 7 笔"的到达率假设, 一笔一笔来就一笔一笔筛; 用大批量的摊薄值
    去凑门限等于换了一个更宽松的问题。其余批大小照常报告, 供读者看曲线。

    只有当该批大小真的被测到时才判; 否则返回 not_decidable。
    """
    rows = []
    for name, d in per_model.items():
        variants = d.get('batches') or {}
        # 判定用"含子图构建"的口径: 那才是逐交易筛查的真实部署成本
        primary = variants.get('with_build') or variants.get('forward_only')
        if not primary:
            rows.append({'model': name, 'verdict': 'not_decidable',
                         'reason': 'no per-transaction measurement'})
            continue
        chosen = None
        for bs, v in primary.items():
            if int(bs) == int(primary_bs) and v.get('ms_per_tx_mean') \
                    is not None:
                chosen = dict(v)
        if chosen is None:
            rows.append({'model': name, 'verdict': 'not_decidable',
                         'reason': 'batch size %s not measured' % primary_bs})
            continue
        curve = {int(bs): v.get('ms_per_tx_mean')
                 for bs, v in primary.items()}
        rows.append({
            'model': name,
            'primary_batch_size': int(primary_bs),
            'ms_per_tx': chosen['ms_per_tx_mean'],
            'ms_per_tx_p95': chosen['ms_per_tx_p95'],
            'joule_per_tx': chosen.get('energy_j_per_batch'),
            'budget_ms': budget_ms,
            'margin_factor': (budget_ms / chosen['ms_per_tx_mean'])
            if chosen['ms_per_tx_mean'] else None,
            'batch_curve_ms_per_tx': curve,
            # 批大小带来的摊薄倍数, 由测量值算出而不是手写。
            # 这个数是提醒: 用大批量的摊薄值去凑门限等于换了个更宽松的问题。
            'batch_amortisation_factor': (
                chosen['ms_per_tx_mean'] / min(
                    v for v in curve.values() if v)
                if chosen.get('ms_per_tx_mean') and any(curve.values())
                else None),
            'pass': bool(chosen['ms_per_tx_mean'] <= budget_ms),
            'verdict': ('pass' if chosen['ms_per_tx_mean'] <= budget_ms
                        else 'falsified'),
        })
    return rows


def run(args):
    import p4_core
    import p4_measure
    from common import set_seed
    from data_loader import load_elliptic, standard_split, make_graph_data

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, labels, ts, edge_index, _ = load_elliptic(args.data_dir)
    tr_np, te_np = standard_split(ts, labels, 34)
    va_np = (ts > 29) & (ts <= 34) & (labels >= 0)
    tr_np = tr_np & ~va_np
    sc = p4_core.fit_scaler_train_only(X, np.where(tr_np)[0])
    Xs = sc.transform(X).astype(np.float32)
    data = make_graph_data(Xs, labels, ts, torch.as_tensor(edge_index,
                                                           dtype=torch.long),
                           dev)

    from models import build_model
    search = {}
    if args.search and os.path.exists(args.search):
        search = json.load(open(args.search, encoding='utf-8'))
    sample_power = p4_measure.sample_gpu_power_mw() if args.measure_power \
        else None
    node_pool = np.where(te_np)[0]

    res = {'schema': SCHEMA,
           'semantics': 'per_node_screening',
           'h2_budget_ms': H2_BUDGET_MS,
           'target_qps': 7,
           'n_warmup': args.n_warmup, 'n_repeat': args.n_repeat,
           'sync': 'torch.cuda.synchronize per iteration',
           'precision': 'fp32',
           'batch_sizes': args.batch_sizes,
           'include_build_variants': args.include_build,
           'power_measured': sample_power is not None,
           'environment': p4_measure.environment_fingerprint(),
           'models': {}}

    for name in args.models:
        set_seed(0)
        kw = {}
        if search.get('models', {}).get(name):
            kw = search['models'][name]['best']['config']
        model = build_model(name, data['in_dim'], **kw).to(dev)
        model.eval()
        hops = K_HOPS.get(name, 2)
        batches = {}
        for inc in args.include_build:
            batches['with_build' if inc else 'forward_only'] = \
                measure_screening(model, data, node_pool, hops,
                                  args.batch_sizes, args.n_warmup,
                                  args.n_repeat, inc, sample_power, seed=0)

        # 整图一次前向, 作为对照口径 (不用于 H2 判定)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.n_repeat):
                model(data['x'], data['edge_index'], data['time_gaps'],
                      data['num_nodes'])
        torch.cuda.synchronize()
        full_ms = (time.perf_counter() - t0) / args.n_repeat * 1000.0
        res['models'][name] = {
            'config': kw, 'hops': hops,
            'params': int(sum(p.numel() for p in model.parameters())),
            'batches': batches,
            'full_graph_forward_ms': float(full_ms),
            'full_graph_amortized_us_per_node': float(
                full_ms * 1000.0 / data['num_nodes']),
        }
        print('[%s] hops=%d params=%d full=%.3f ms amortized=%.4f us/node'
              % (name, hops, res['models'][name]['params'], full_ms,
                 res['models'][name]['full_graph_amortized_us_per_node']))
        for variant, bs_dict in batches.items():
            for bs, v in bs_dict.items():
                print('    %-12s bs=%-4d %.4f ms/tx (p95 %.4f) build=%.4f ms'
                      % (variant, bs, v['ms_per_tx_mean'],
                         v['ms_per_tx_p95'], v['subgraph_build_ms_mean']))

    res['h2'] = h2_verdict(res['models'], primary_bs=args.primary_bs)
    res['h2_primary_batch_size'] = args.primary_bs
    res['h2_primary_batch_rationale'] = (
        '门限来自每秒 7 笔的到达率; 一笔一笔到达就一笔一笔筛, 因此判定用 '
        'batch=1 (含子图构建)。大批量的摊薄值另行报告, 不用于判门限。')
    p4_core.save_json(res, args.out)
    print(json.dumps(res['h2'], ensure_ascii=False, indent=2))
    print('saved', args.out)
    return res


def cmd_verdict(args):
    """从已存结果重算 H2 判定 (不重跑测量)。"""
    import p4_core
    d = json.load(open(args.input, encoding='utf-8'))
    rows = h2_verdict(d['models'], primary_bs=args.primary_bs)
    d['h2'] = rows
    d['h2_primary_batch_size'] = args.primary_bs
    p4_core.save_json(d, args.out or args.input)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')

    p = sub.add_parser('run')
    p.add_argument('--data_dir', default='./data/elliptic')
    ap.add_argument('--models', nargs='+',
                    default=['tasgl', 'graphsage', 'mlp'])
    ap.add_argument('--batch_sizes', type=int, nargs='+',
                    default=[1, 8, 32, 128])
    ap.add_argument('--include_build', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--n_warmup', type=int, default=20)
    ap.add_argument('--n_repeat', type=int, default=100)
    ap.add_argument('--measure_power', type=int, default=0,
                    help='Opt-in only. The final manuscript reports NO energy '
                         'figure: the development GPU exposes no power limit, '
                         'so a power.draw-derived joule value is not a valid '
                         'measurement. Enable only on a device with a '
                         'verified, readable power rail. Default 0 keeps '
                         'regenerated results consistent with the paper.')
    p.add_argument('--search', default='results/p4/hpo_search.json')
    p.add_argument('--primary_bs', type=int, default=1)
    p.add_argument('--out', default='results/p4/per_node_screening.json')
    p.set_defaults(func=lambda a: run(a))

    p = sub.add_parser('verdict')
    p.add_argument('--input', default='results/p4/per_node_screening.json')
    p.add_argument('--primary_bs', type=int, default=1)
    p.add_argument('--out', default=None)
    p.set_defaults(func=lambda a: cmd_verdict(a))

    a = ap.parse_args()
    if not getattr(a, 'func', None):
        ap.print_help()
        return 2
    a.func(a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
