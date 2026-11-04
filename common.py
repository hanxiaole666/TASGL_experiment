"""common.py — 全实验共用的训练例程与工具"""
import json
import os
import random
import re
import numpy as np
import torch
import torch.nn.functional as F

from losses import TASGLLoss
from metrics import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_weights_from(labels, mask):
    """类别频率反比权重 (应对 9.8% 不平衡)。"""
    y = labels[mask]
    n0 = (y == 0).sum().item()
    n1 = (y == 1).sum().item()
    tot = max(n0 + n1, 1)
    return torch.tensor([tot / max(2 * n0, 1), tot / max(2 * n1, 1)],
                        dtype=torch.float32)


def temperature_schedule(epoch, total, t_start=10.0, t_end=50.0,
                         warm=0.25, full=0.75):
    """门控软掩码温度退火: 10 -> 50 (epoch 25%–75% 线性)。"""
    lo, hi = int(total * warm), int(total * full)
    if epoch < lo:
        return t_start
    if epoch > hi:
        return t_end
    return t_start + (t_end - t_start) * (epoch - lo) / max(hi - lo, 1)


def _save_ckpt(ckpt_path, payload):
    """原子写入 checkpoint, 避免闪退产生半个文件。"""
    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    tmp = ckpt_path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, ckpt_path)


def train_model(model, data, train_mask, val_mask, device, epochs=200,
                lr=1e-3, wd=1e-4, patience=30, loss_fn=None, verbose=False,
                ckpt_path=None, protocol=None):
    """统一训练循环。TASGL 走 TASGLLoss (MC 不确定性 + 温度退火),
    基线走类别加权 CE。返回最佳验证 F1 时的状态。

    ckpt_path: 若指定, 每 epoch 保存训练断点 (模型/优化器/最优状态),
    闪退后重跑自动从最近 epoch 继续; 训练完成后自动删除断点文件。
    """
    protocol = PROTOCOL_VERSION if protocol is None else protocol
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    cw = class_weights_from(data['y'].cpu(), train_mask.cpu()).to(device)
    if loss_fn is None:
        loss_fn = TASGLLoss(class_weights=cw).to(device)
        is_tasgl = True
    else:
        is_tasgl = False

    best_val, best_state, bad, start_epoch = -1.0, None, 0, 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device,
                            weights_only=False)
            if ck.get('protocol') != protocol:
                print('  [断点恢复] checkpoint 属于旧协议, 已作废并从头训练')
            else:
                model.load_state_dict(ck['model'])
                opt.load_state_dict(ck['opt'])
                best_val, best_state = ck['best_val'], ck['best_state']
                bad, start_epoch = ck['bad'], ck['epoch'] + 1
                print(f'  [断点恢复] 从 epoch {start_epoch} 继续 '
                      f'(best_val={best_val:.4f})')
        except Exception as e:
            print(f'  [断点恢复] checkpoint 损坏({e}), 从头训练')

    for epoch in range(start_epoch, epochs):
        model.train()
        temp = temperature_schedule(epoch, epochs) if is_tasgl else 10.0
        if is_tasgl:
            # MC 不确定性 (训练期用少量样本 T=5 控制开销)
            _, ent = model.mc_uncertainty(
                data['x'], data['edge_index'], data['time_gaps'],
                data['num_nodes'], T=5)
            x_in = loss_fn.dro.feature_perturb(data['x'], ent, train_mask)
            logits, aux = model(x_in, data['edge_index'], data['time_gaps'],
                                data['num_nodes'], temperature=temp)
            loss, parts = loss_fn(logits, aux['h'], data['y'], ent,
                                  train_mask)
        else:
            logits, _ = model(data['x'], data['edge_index'],
                              data['time_gaps'], data['num_nodes'])
            loss = F.cross_entropy(logits[train_mask],
                                   data['y'][train_mask], weight=cw)
            parts = {'cls': loss.item(), 'con': 0.0}

        opt.zero_grad()
        loss.backward()
        opt.step()

        val = evaluate(model, data, val_mask)['f1']
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and (epoch % 20 == 0 or bad == 0):
            print(f'  epoch {epoch:3d} loss={loss.item():.4f} '
                  f'(cls={parts["cls"]:.4f} con={parts["con"]:.4f}) '
                  f'val_f1={val:.4f}')
        if ckpt_path:
            _save_ckpt(ckpt_path, {
                'model': model.state_dict(), 'opt': opt.state_dict(),
                'best_val': best_val, 'best_state': best_state,
                'bad': bad, 'epoch': epoch,
                'protocol': protocol})
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return model, best_val


def _json_default(o):
    """numpy 标量/数组 -> 原生 Python 类型, 保证结果可直接 JSON 序列化。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable')


def resume_paths(exp_name, key, base='results/runs'):
    """返回 (结果 json 路径, 训练 ckpt 路径)。key 中的 / : 等字符会被转义。"""
    d = os.path.join(base, exp_name)
    os.makedirs(d, exist_ok=True)
    safe = re.sub(r'[^A-Za-z0-9_.=-]+', '_', str(key))
    return os.path.join(d, safe + '.json'), os.path.join(d, safe + '.ckpt')


# 全局评估协议: 3 = 验证集阈值调优 + 标签翻转修复。
# 通用协议变化时递增此号; 单项实验修复可给 load_if_done/train_model
# 传入独立 protocol, 只作废对应缓存, 无需手动删除 results/。
PROTOCOL_VERSION = 3


def load_if_done(result_path, overwrite=False, protocol=None):
    """运行级断点: 仅接受指定协议版本的完整字典结果。

    protocol=None 使用全局 PROTOCOL_VERSION; 单项实验修复可传入自己的
    协议号, 避免无辜地作废其他实验。旧结果、旧缓存结构和损坏文件均会
    自动重跑; overwrite=True 时始终返回 None。
    """
    protocol = PROTOCOL_VERSION if protocol is None else protocol
    if result_path and os.path.exists(result_path) and not overwrite:
        try:
            obj = load_json(result_path)
        except Exception:
            return None   # 结果文件损坏 -> 重跑该组合
        if not isinstance(obj, dict) or \
                obj.get('protocol') != protocol:
            return None   # 旧协议/旧缓存结构 -> 自动重跑
        return obj
    return None


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)
