"""TASGL-Lite v3 训练、校准、评估与断点工具。"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from common import set_seed, class_weights_from
from metrics import (accuracy_metrics, expected_calibration_error,
                     selective_prediction_risk, _trapz)
from models_v3 import (LogitCalibrator, EDLHead, symmetric_kl,
                       distillation_loss, temporal_alignment_loss,
                       dirichlet_kl_to_uniform)

# 协议 6：开发目标改为不可互相补偿的硬门槛；协议 5 历史结果保留在 results/v3。
V3_PROTOCOL = 6


def make_masks(time_steps, labels, train_end, val_start, val_end,
               test_start, test_end=49, device=None):
    tr = (time_steps <= train_end) & (labels >= 0)
    va = ((time_steps >= val_start) & (time_steps <= val_end)
          & (labels >= 0))
    tr = tr & ~va
    te = ((time_steps >= test_start) & (time_steps <= test_end)
          & (labels >= 0))
    masks = [torch.tensor(m, dtype=torch.bool) for m in (tr, va, te)]
    if device is not None:
        masks = [m.to(device) for m in masks]
    return masks


def _torch_save_atomic(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available()
            else None}


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def drop_edges(data, rate):
    if rate <= 0:
        return data
    keep = torch.rand(data['edge_index'].size(1),
                      device=data['edge_index'].device) >= rate
    out = dict(data)
    out['edge_index'] = data['edge_index'][:, keep]
    out['time_gaps'] = data['time_gaps'][keep]
    return out


@torch.no_grad()
def predict_logits(model, data):
    model.eval()
    logits, _ = model(data['x'], data['edge_index'], data['time_gaps'],
                      data['num_nodes'])
    return logits


def best_f1_threshold_from_probs(probs, labels, grid=None):
    """在指定验证概率上选择 F1 阈值；只接收开发方提供的数据。"""
    p = torch.as_tensor(probs).reshape(-1)
    y = torch.as_tensor(labels).reshape(-1)
    if grid is None:
        grid = torch.linspace(0.02, 0.98, 49, device=p.device)
    else:
        grid = torch.as_tensor(grid, device=p.device)
    pred = p.unsqueeze(1) >= grid.unsqueeze(0)
    yb = y.to(torch.bool).unsqueeze(1)
    tp = (pred & yb).sum(0).float()
    fp = (pred & ~yb).sum(0).float()
    fn = (~pred & yb).sum(0).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-12)
    k = int(f1.argmax().item())
    return float(grid[k].item()), float(f1[k].item())


def fit_calibrator_and_threshold(logits, labels, val_mask, method):
    cal = LogitCalibrator(method).fit(logits[val_mask].detach().cpu(),
                                     labels[val_mask].detach().cpu())
    pv = cal.probabilities(logits[val_mask].detach().cpu())[:, 1]
    yv = labels[val_mask].detach().cpu()
    threshold, val_f1 = best_f1_threshold_from_probs(pv, yv)
    return cal, threshold, val_f1


def evaluate_logits(logits, labels, mask, calibrator, threshold):
    p = calibrator.probabilities(logits.detach().cpu())
    y = labels.detach().cpu()
    pm, ym = p[mask.cpu()], y[mask.cpu()]
    p1 = pm[:, 1].numpy()
    yn = ym.numpy()
    result = accuracy_metrics(p1, yn, threshold=threshold)
    result['ece'] = expected_calibration_error(pm.numpy(), yn, n_bins=15)
    result['brier'] = float(np.mean((p1 - yn) ** 2))
    true_prob = pm.gather(1, ym.view(-1, 1)).clamp(min=1e-12)
    result['nll'] = float(-true_prob.log().mean().item())
    entropy = -(pm * pm.clamp(min=1e-12).log()).sum(-1).numpy()
    cov, risk = selective_prediction_risk(p1, yn, entropy,
                                          threshold=threshold)
    result['selective_auc'] = float(_trapz(risk, cov))
    result['threshold'] = float(threshold)
    return result


def evaluate_model(model, data, mask, calibrator, threshold):
    return evaluate_logits(predict_logits(model, data), data['y'], mask,
                           calibrator, threshold)


def train_lite(model, data, masks, config, device, epochs, patience,
               ckpt_path, metadata, teacher_logits=None, verbose=False):
    """训练轻量学生并保存 epoch 级断点。所有辅助损失仅看训练截止窗口。"""
    train_mask, val_mask = masks[:2]
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config['lr'],
                            weight_decay=config['wd'])
    cw = class_weights_from(data['y'].cpu(), train_mask.cpu()).to(device)
    cw = cw.pow(config.get('class_weight_power', 0.5))
    train_end = int(metadata['train_end'] if 'train_end' in metadata
                    else metadata['schedule']['train_end'])
    auxiliary_mask = data['time_steps'] <= train_end
    if config.get('kd_weight', 0) > 0 and teacher_logits is None:
        raise ValueError('kd_weight>0 时必须提供图教师 logits')

    best_val, best_state, bad, start_epoch = -1.0, None, 0, 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device,
                            weights_only=False)
            if ck.get('protocol') != V3_PROTOCOL or \
                    ck.get('metadata') != metadata:
                print('  [v3断点] 协议或配置不匹配，从头训练')
            else:
                model.load_state_dict(ck['model'])
                opt.load_state_dict(ck['opt'])
                best_val, best_state = ck['best_val'], ck['best_state']
                bad, start_epoch = ck['bad'], ck['epoch'] + 1
                if ck.get('rng'):
                    _restore_rng(ck['rng'])
                print(f'  [v3断点] 从 epoch {start_epoch} 继续')
        except Exception as exc:
            print(f'  [v3断点] 文件损坏({exc})，从头训练')

    for epoch in range(start_epoch, epochs):
        model.train()
        clean = data
        noisy = drop_edges(data, config.get('edge_drop', 0.0))
        logits, aux = model(clean['x'], clean['edge_index'],
                            clean['time_gaps'], clean['num_nodes'])
        y = data['y'][train_mask]
        z = logits[train_mask]
        ce = F.cross_entropy(z, y, weight=cw, reduction='none')
        gamma = config.get('focal_gamma', 0.0)
        if gamma > 0:
            pt = torch.softmax(z, dim=-1).gather(1, y.view(-1, 1)).squeeze(1)
            ce = ce * (1 - pt).clamp(min=0).pow(gamma)
        loss = ce.mean()
        parts = {'cls': float(loss.detach()), 'robust': 0.0,
                 'consistency': 0.0, 'kd': 0.0, 'temporal': 0.0}

        if config.get('edge_drop', 0.0) > 0:
            noisy_logits, _ = model(noisy['x'], noisy['edge_index'],
                                    noisy['time_gaps'], noisy['num_nodes'])
            robust_ce = F.cross_entropy(noisy_logits[train_mask], y,
                                        weight=cw)
            rw = config.get('robust_ce_weight', 0.5)
            loss = loss + rw * robust_ce
            parts['robust'] = float(robust_ce.detach())
            cw_cons = config.get('consistency_weight', 0.0)
            if cw_cons > 0:
                cons = symmetric_kl(logits[auxiliary_mask],
                                    noisy_logits[auxiliary_mask])
                loss = loss + cw_cons * cons
                parts['consistency'] = float(cons.detach())

        kw = config.get('kd_weight', 0.0)
        if kw > 0:
            kd = distillation_loss(logits[auxiliary_mask],
                                   teacher_logits[auxiliary_mask])
            loss = loss + kw * kd
            parts['kd'] = float(kd.detach())

        tw = config.get('temporal_weight', 0.0)
        if tw > 0:
            temporal = temporal_alignment_loss(aux['h'], data['time_steps'],
                                               train_end)
            loss = loss + tw * temporal
            parts['temporal'] = float(temporal.detach())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        with torch.no_grad():
            val_logits = predict_logits(model, data)
            pv = torch.softmax(val_logits[val_mask], dim=-1)[:, 1]
            _, val_f1 = best_f1_threshold_from_probs(
                pv, data['y'][val_mask])
        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if verbose and (epoch % 20 == 0 or bad == 0):
            print(f'  epoch={epoch:3d} loss={float(loss):.4f} '
                  f'val_f1={val_f1:.4f} parts={parts}')
        if ckpt_path:
            _torch_save_atomic({
                'model': model.state_dict(), 'opt': opt.state_dict(),
                'best_val': best_val, 'best_state': best_state,
                'bad': bad, 'epoch': epoch, 'protocol': V3_PROTOCOL,
                'metadata': metadata, 'rng': _rng_state()}, ckpt_path)
        if bad >= patience:
            break
    if best_state is None:
        raise RuntimeError('训练未产生有效验证状态')
    model.load_state_dict(best_state)
    # 最优状态可能早于最后一次验证，不能让静态图缓存保留旧时间衰减参数。
    if hasattr(model, 'prop') and hasattr(model.prop, '_cached_key'):
        model.prop._cached_key = None
        model.prop._cached_adj = None
        model.prop._cached_inputs = None
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return model, best_val


def get_teacher_logits(teacher_name, data, masks, device, seed, epochs,
                       cache_path, ckpt_path, metadata, overwrite=False):
    """训练一次图教师并缓存软标签；选择阶段的测试标签已在外部屏蔽。"""
    if os.path.exists(cache_path) and not overwrite:
        try:
            obj = torch.load(cache_path, map_location='cpu',
                             weights_only=False)
            if obj.get('protocol') == V3_PROTOCOL and \
                    obj.get('metadata') == metadata and \
                    tuple(obj['logits'].shape) == (data['num_nodes'], 2):
                return obj['logits'].to(device)
        except Exception:
            pass
    from models import build_model
    from common import train_model
    if overwrite and os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    set_seed(seed)
    teacher = build_model(teacher_name, data['in_dim'])
    teacher, _ = train_model(teacher, data, masks[0], masks[1], device,
                             epochs=epochs, loss_fn=False, ckpt_path=ckpt_path,
                             protocol=V3_PROTOCOL)
    logits = predict_logits(teacher, data).detach().cpu()
    _torch_save_atomic({'protocol': V3_PROTOCOL, 'metadata': metadata,
                        'logits': logits}, cache_path)
    return logits.to(device)


def fit_edl_head(backbone, data, masks, config, device, epochs=100):
    """同一骨干上的 EDL 代理基线；用验证 NLL 早停，不使用测试标签。"""
    model = backbone.to(device)
    model.eval()
    with torch.no_grad():
        h = model.encode(data['x'], data['edge_index'], data['time_gaps'],
                         data['num_nodes']).detach()
    head = EDLHead(config['hidden']).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    train_mask, val_mask = masks[:2]
    cw = class_weights_from(data['y'].cpu(), train_mask.cpu()).to(device)
    cw = cw.pow(config.get('class_weight_power', 0.5))
    best, best_state, bad = float('inf'), None, 0
    for epoch in range(epochs):
        head.train()
        evidence, alpha, probs, _ = head(h)
        nll = F.nll_loss(probs[train_mask].clamp(min=1e-9).log(),
                         data['y'][train_mask], weight=cw)
        y1 = F.one_hot(data['y'][train_mask], num_classes=2).float()
        tilde = (1 - y1) * alpha[train_mask] + y1
        kl = dirichlet_kl_to_uniform(tilde).mean()
        loss = nll + min(1.0, epoch / 20.0) * 0.01 * kl
        opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            _, _, pv, _ = head(h)
            vnll = F.nll_loss(pv[val_mask].clamp(min=1e-9).log(),
                              data['y'][val_mask]).item()
        if vnll < best:
            best = vnll
            best_state = {k: v.detach().cpu().clone()
                          for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 20:
            break
    if best_state is None:
        raise RuntimeError('EDL 头训练失败')
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        _, _, probs, vacuity = head(h)
    return probs.detach(), vacuity.detach(), best


def f1_from_scores(labels, scores, threshold):
    return f1_score(labels, scores >= threshold, zero_division=0)
