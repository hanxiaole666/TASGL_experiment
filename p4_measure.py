"""p4_measure.py — P4 环境指纹、测量口径与师生全链路成本

审稿意见 7 (完整系统成本 + 环境指纹) 要求:
  教师训练 / 预处理 / 图构建 / 蒸馏 / 校准 / 阈值选择 / 推理 / 更新
  八个阶段的成本都要记录, 并给出 GPU/CPU/驱动/CUDA/cuDNN/PyTorch/OS 指纹;
  同时必须声明测量口径 (warmup / 重复次数 / 同步方式 / 批量语义)。

本模块的原则:
  1. 测不到就写 measured=False + reason, 绝不填合成值。
  2. 测量口径 (measurement_semantics) 是必填项; 缺少口径的延迟数字
     不得与门限比较 (见 audit_cost_completeness)。
  3. 师生全链路必须把教师训练与蒸馏按"摊销到每个部署时点/每千节点"
     的方式计入, 否则"轻量"结论不成立。
"""
import hashlib
import json
import os
import platform
import subprocess
import time

# ------------------------------------------------------------------
# 阶段定义
# ------------------------------------------------------------------
REQUIRED_STAGES = ('teacher_training', 'preprocessing', 'graph_construction',
                   'distillation', 'calibration', 'threshold_selection',
                   'inference', 'update')

STAGE_LABEL = {
    'teacher_training': '教师模型训练 (一次性, 但必须摊销)',
    'preprocessing': '特征预处理 / 标准化',
    'graph_construction': '图构建 (边表 -> 邻接, 时间差计算)',
    'distillation': '蒸馏 / 学生训练',
    'calibration': '概率校准 (温度/分层校准) 拟合',
    'threshold_selection': '验证集阈值网格搜索',
    'inference': '部署期推理',
    'update': '增量更新 / 再训练',
}

# 测量语义: 现有 metrics.measure_efficiency 测的是"整图一次前向",
# 与"单笔交易筛查延迟"不是一个口径, 不能与 8ms 门限直接比较。
MEASUREMENT_SEMANTICS = ('full_graph_forward', 'per_node_screening',
                         'batch_1_subgraph')

REQUIRED_ENV_KEYS = ('gpu_name', 'gpu_count', 'driver_version',
                     'cuda_runtime', 'cudnn', 'torch_version',
                     'python_version', 'os', 'cpu', 'numpy_version')


def _sha(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False,
                   default=str).encode('utf-8')).hexdigest()


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=20)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def environment_fingerprint():
    """采集完整环境指纹; 采不到的一律 None 并在 missing 中列出。"""
    env = {
        'schema': 'p4_environment_fingerprint_v1',
        'python_version': platform.python_version(),
        'os': f'{platform.system()} {platform.release()}',
        'platform': platform.platform(),
        'cpu': platform.processor() or None,
        'machine': platform.machine(),
    }
    try:
        import numpy
        env['numpy_version'] = numpy.__version__
    except Exception:
        env['numpy_version'] = None
    try:
        import torch
        env['torch_version'] = torch.__version__
        env['cuda_runtime'] = getattr(torch.version, 'cuda', None)
        env['cudnn'] = str(torch.backends.cudnn.version())
        env['cuda_available'] = bool(torch.cuda.is_available())
        env['gpu_count'] = torch.cuda.device_count()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env['gpu_name'] = p.name
            env['gpu_total_mem_mb'] = round(p.total_memory / 1024 ** 2, 1)
            env['gpu_sm_count'] = p.multi_processor_count
    except Exception:
        env['torch_version'] = None

    smi = _run(['nvidia-smi',
                '--query-gpu=name,driver_version,memory.total,power.limit',
                '--format=csv,noheader'])
    if smi:
        parts = [s.strip() for s in smi.split(',')]
        env['gpu_name'] = env.get('gpu_name') or parts[0]
        env['driver_version'] = parts[1] if len(parts) > 1 else None
        env['power_limit_w'] = parts[3] if len(parts) > 3 else None
    else:
        env['driver_version'] = env.get('driver_version')
    env['missing'] = [k for k in REQUIRED_ENV_KEYS if not env.get(k)]
    env['complete'] = not env['missing']
    env['fingerprint'] = _sha({k: env.get(k) for k in REQUIRED_ENV_KEYS})
    return env


def measurement_protocol(semantics='full_graph_forward', n_warmup=10,
                         n_repeat=100, sync='cuda.synchronize',
                         batch=None, precision='fp32', note=''):
    """声明测量口径。缺口径的延迟不得与门限比较。"""
    if semantics not in MEASUREMENT_SEMANTICS:
        raise ValueError(f'semantics 必须是 {MEASUREMENT_SEMANTICS}')
    return {
        'schema': 'p4_measurement_protocol_v1',
        'semantics': semantics,
        'n_warmup': int(n_warmup), 'n_repeat': int(n_repeat),
        'sync': sync, 'batch': batch, 'precision': precision, 'note': note,
        'warning': ('full_graph_forward 的 latency_ms 是"整图一次前向", '
                    '与单笔交易筛查延迟不是一个口径; '
                    '若门限按单笔交易定义, 必须换算为 per_node'),
    }


def sample_gpu_power_mw(device=0, interval_s=0.05):
    """返回采样函数与是否可用。不可用时返回 None (不得伪造能耗)。"""
    ok = _run(['nvidia-smi',
               f'--query-gpu=power.draw', '--format=csv,noheader,nounits'])
    if ok is None:
        return None

    def read():
        v = _run(['nvidia-smi', '--query-gpu=power.draw',
                  '--format=csv,noheader,nounits'])
        try:
            return float(v)          # 瓦
        except Exception:
            return None
    return read


def measure_stage(name, fn, repeats=1, n_warmup=1,
                  protocol=None, sample_power=None, device='cuda'):
    """测量一个阶段: 墙钟时间 + 峰值显存 + (可选)能耗积分。

    返回 dict, 未测到的字段为 None 且 measured=False。
    """
    import torch
    if name not in REQUIRED_STAGES:
        raise ValueError(f'未知阶段 {name}, 必须是 {REQUIRED_STAGES}')
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(n_warmup):
        fn()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()

    power_w = []
    stop = [False]
    if sample_power is not None:
        import threading

        def _poll():
            while not stop[0]:
                v = sample_power()
                if v is not None:
                    power_w.append(v)
                time.sleep(0.05)
        th = threading.Thread(target=_poll, daemon=True)
        th.start()
    else:
        th = None

    t0 = time.time()
    for _ in range(repeats):
        out = fn()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / max(repeats, 1)
    stop[0] = True
    if th is not None:
        th.join(timeout=1.0)

    peak = None
    if device == 'cuda' and torch.cuda.is_available():
        peak = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    energy_j = None
    if power_w:
        mean_w = sum(power_w) / len(power_w)
        energy_j = round(mean_w * dt * repeats, 3)
    return {
        'stage': name, 'label': STAGE_LABEL[name],
        'seconds': round(dt, 6), 'repeats': repeats,
        'total_seconds': round(dt * repeats, 6),
        'peak_mem_mb': peak,
        'energy_j': energy_j,
        'energy_measured': bool(power_w),
        'mean_power_w': (sum(power_w) / len(power_w)) if power_w else None,
        'power_samples': len(power_w),
        'protocol': protocol,
        'measured': True,
        # 被调用函数的返回值 (调用方需要"测量完还要拿到产物"时用)。
        # 不写进 JSON 的场景由调用方自行弹出。
        'out': out,
    }


def full_chain_cost(stages, amortize_nodes=None, teacher_lifetime_updates=1):
    """汇总师生全链路成本, 并把一次性成本 (教师训练/图构建) 摊销。

    stages: [{stage, seconds, energy_j, ...}]
    返回 total / one_off / recurring / 摊销后每千节点成本 / 缺失阶段。
    """
    by = {s.get('stage'): s for s in stages if s.get('stage')}
    one_off = ('teacher_training', 'graph_construction', 'distillation',
               'calibration', 'threshold_selection')
    recurring = ('preprocessing', 'inference', 'update')
    tot_t = sum(float(s.get('total_seconds') or 0) for s in stages)
    tot_e = sum(float(s.get('energy_j') or 0) for s in stages
                if s.get('energy_j') is not None)
    one_t = sum(float(by[k].get('total_seconds') or 0) for k in one_off
                if k in by)
    rec_t = sum(float(by[k].get('total_seconds') or 0) for k in recurring
                if k in by)
    amort = one_t / max(int(teacher_lifetime_updates), 1)
    per_1k = None
    if amortize_nodes:
        per_1k = (rec_t + amort) / (amortize_nodes / 1000.0)
    missing = [k for k in REQUIRED_STAGES if k not in by]
    return {
        'schema': 'p4_full_chain_cost_v1',
        'stages_present': sorted(by),
        'stages_missing': missing,
        'complete': not missing,
        'total_seconds': round(tot_t, 4),
        'total_energy_j': round(tot_e, 3) if tot_e else None,
        'one_off_seconds': round(one_t, 4),
        'recurring_seconds': round(rec_t, 4),
        'amortized_one_off_seconds': round(amort, 4),
        'teacher_lifetime_updates': int(teacher_lifetime_updates),
        'amortized_seconds_per_1k_nodes': (round(per_1k, 6)
                                           if per_1k is not None else None),
        'energy_all_measured': all(
            s.get('energy_measured') for s in stages
            if s.get('stage') in by),
        'note': ('未摊销的一次性成本在论文中必须与部署期成本并列给出; '
                 '只报推理延迟而不报教师训练/蒸馏, 属于成本陈述不完整'),
    }


def audit_cost_completeness(chain, protocol=None, env=None):
    """成本陈述是否可用于"轻量/更快"类主张。"""
    reasons = []
    if not chain.get('complete'):
        reasons.append(f"缺失阶段: {chain.get('stages_missing')}")
    if protocol is None:
        reasons.append('未声明测量口径 (semantics/warmup/repeat/sync)')
    elif protocol.get('semantics') == 'full_graph_forward':
        reasons.append('口径为整图前向, 与"单笔交易延迟"不可直接比较')
    if env is None or not env.get('complete'):
        reasons.append(f"环境指纹不完整: "
                       f"{(env or {}).get('missing')}")
    if not chain.get('energy_all_measured', False):
        reasons.append('能耗未实测 (不得填合成值)')
    return {'schema': 'p4_cost_completeness_v1', 'usable_for_claims':
            not reasons, 'blocking_reasons': reasons}


def save_json(obj, path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path
