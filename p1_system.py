"""P1-4/P1-5：分阶段成本、端到端效率与真实批量曲线。"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from data_loader import compute_time_gaps
from models import MLPBaseline
from models_v3 import TASGLite
from p0_data import load_elliptic_raw
from p0_models import FeatureMLP


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def load_development_graph_timed(data_dir, device, label_ceiling=34):
    """分离数据读取、标准化、图构建和设备传输的计时。

    该路径用于开发成本审计，沿用全特征标准化的 transductive 设定；
    严格 rolling 成本必须另用 P0-7 窗口拟合视图，不能混写。
    """
    device = torch.device(device)
    t0 = time.perf_counter()
    raw = load_elliptic_raw(data_dir, label_ceiling=label_ceiling)
    data_loading = time.perf_counter() - t0
    t0 = time.perf_counter()
    scaler = StandardScaler()
    x = scaler.fit_transform(raw['x']).astype(np.float32)
    preprocessing = time.perf_counter() - t0
    t0 = time.perf_counter()
    edge = torch.as_tensor(raw['edge_index'], dtype=torch.long)
    gaps = compute_time_gaps(edge, raw['node_time'])
    data = {
        'x': torch.as_tensor(x, dtype=torch.float32, device=device),
        'y': torch.as_tensor(raw['y'], dtype=torch.long, device=device),
        'edge_index': edge.to(device), 'time_gaps': gaps.to(device),
        'time_steps': torch.as_tensor(raw['node_time'], dtype=torch.long,
                                      device=device),
        'num_nodes': len(x), 'in_dim': x.shape[1],
    }
    sync(device)
    graph_construction = time.perf_counter() - t0
    timings = {'data_loading': float(data_loading),
               'preprocessing': float(preprocessing),
               'graph_construction': float(graph_construction)}
    return data, raw, scaler, timings


def _rss_mib():
    """跨平台当前进程 RSS； psutil 不可用时明确返回 None。"""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / 1024 ** 2)
    except (ImportError, OSError):
        return None


def _is_batch_capable(model):
    base = model.model if hasattr(model, 'model') else model
    if isinstance(base, (MLPBaseline, FeatureMLP)):
        return True
    return isinstance(base, TASGLite) and base.hops == 0


def _forward(model, x, edge, gaps, n):
    if isinstance(model, FeatureMLP):
        return model(x), {}
    return model(x, edge, gaps, n, hard_mask=True)


def benchmark_inference(model, data, device, batch_sizes=(1, 32, 128, 1024),
                        warmup=3, repeats=10, eval_mask=None):
    """测 steady-state 推理；批量曲线只对语义可分批的无图学生/MLP 启用。

    图模型使用全图前向，不把无邻居采样的节点切片伪装成 mini-batch GNN。
    """
    device = torch.device(device)
    model = model.to(device).eval()
    batch_capable = _is_batch_capable(model)
    sizes = list(batch_sizes) if batch_capable else ['full_graph']
    if repeats < 1 or warmup < 0:
        raise ValueError('repeats 必须为正，warmup 不得为负')
    x, edge, gaps = data['x'], data['edge_index'], data['time_gaps']
    n_total = data['num_nodes']
    selected = (np.arange(n_total) if eval_mask is None else
                np.flatnonzero(np.asarray(eval_mask, dtype=bool)))
    if not len(selected):
        raise ValueError('效率评估节点为空')
    rows = []
    with torch.no_grad():
        for size in sizes:
            batches = ([selected[i:i + int(size)] for i in range(
                0, len(selected), int(size))] if batch_capable else [None])
            n_scored = len(selected) if batch_capable else n_total

            def one_pass():
                for nodes in batches:
                    if nodes is None:
                        out = _forward(model, x, edge, gaps, n_total)
                    else:
                        index = torch.as_tensor(nodes, dtype=torch.long, device=device)
                        xb = x[index]
                        empty = torch.empty((2, 0), dtype=torch.long, device=device)
                        out = _forward(model, xb, empty,
                                       torch.empty(0, device=device), len(nodes))
                    # 强制保留输出，避免死代码消除造成虚假零延迟。
                    logits = out[0] if isinstance(out, tuple) else out
                    if logits.numel() == 0:
                        raise RuntimeError('推理返回空 logits')

            for _ in range(warmup):
                one_pass()
            sync(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            rss_before = _rss_mib()
            t0 = time.perf_counter()
            for _ in range(repeats):
                one_pass()
            sync(device)
            pass_ms = (time.perf_counter() - t0) * 1000.0 / repeats
            rss_after = _rss_mib()
            rss_growth = (None if rss_before is None or rss_after is None
                          else max(0.0, rss_after - rss_before))
            rows.append({
                'device': device.type, 'batch_size': size,
                'batch_semantics': ('independent_node_batches' if batch_capable
                                    else 'full_graph_forward'),
                'n_scored_nodes': int(n_scored), 'n_batches': len(batches),
                'mean_pass_latency_ms': float(pass_ms),
                'mean_batch_latency_ms': float(pass_ms / len(batches)),
                'mean_node_latency_ms': float(pass_ms / max(n_scored, 1)),
                'throughput_nodes_per_second': float(n_scored / (pass_ms / 1000.0)),
                'peak_gpu_memory_mib': (float(torch.cuda.max_memory_allocated(device)
                                               / 1024 ** 2) if device.type == 'cuda'
                                        else None),
                'cpu_rss_after_mib': rss_after,
                'cpu_rss_growth_mib': rss_growth,
                'warmup': int(warmup), 'repeats': int(repeats),
            })
    return {
        'protocol': 'p1_end_to_end_inference_v1', 'batch_curve_supported': batch_capable,
        'limitation': ('Full-graph models require graph-aware mini-batching; '
                       'node slicing is not used as a fake batch curve.')
            if not batch_capable else
            'Independent node batches are valid because this model performs no graph propagation.',
        'rows': rows,
    }


def cold_start_summary(load_timings, artifact_loading_seconds, first_pass_ms):
    return {
        'data_loading_seconds': float(load_timings['data_loading']),
        'preprocessing_seconds': float(load_timings['preprocessing']),
        'graph_construction_seconds': float(load_timings['graph_construction']),
        'artifact_loading_seconds': float(artifact_loading_seconds),
        'first_inference_ms': float(first_pass_ms),
        'cold_start_total_seconds': float(sum(load_timings.values()) +
                                          artifact_loading_seconds + first_pass_ms / 1000),
        'boundary': 'cold start and steady-state inference are separate; teacher training is not hidden in either',
    }
