"""P1-2/P1-3：预注册攻击矩阵与无 oracle 标签的图扰动。

攻击构造函数故意不接收 y。自适应攻击只使用干净模型预测、图结构和
攻击种子；评估标签须在攻击图冻结之后才用于指标计算。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from p0_core import canonical_json, sha256_bytes


CAPABILITIES = {'edge_delete', 'edge_inject', 'feature_noise'}
SELECTIONS = {'random', 'degree_targeted', 'confidence_targeted',
              'all_evaluation'}
KNOWLEDGE = {'black_box', 'gray_box', 'white_box'}


def validate_attack_matrix(config):
    if config.get('protocol') != 'p1_nonoracle_attack_matrix_v1':
        raise ValueError('攻击矩阵协议必须为 p1_nonoracle_attack_matrix_v1')
    if config.get('feature_noise_scope') != 'evaluation_nodes_only':
        raise ValueError('P1-2 要求特征噪声只作用于评估节点')
    if config.get('nonpositive_baseline_rule') != 'inconclusive':
        raise ValueError('必须预声明基线非正下降时为 inconclusive')
    attacks = config.get('attacks', [])
    if not attacks:
        raise ValueError('攻击矩阵不能为空')
    names = []
    for a in attacks:
        required = {'name', 'capability', 'selection', 'knowledge', 'goal',
                    'budget_type', 'budget', 'oracle_labels', 'target_scope'}
        missing = required - set(a)
        if missing:
            raise ValueError(f'攻击缺字段 {a.get("name")}: {sorted(missing)}')
        if a['capability'] not in CAPABILITIES or a['selection'] not in SELECTIONS:
            raise ValueError(f'未知攻击能力或选择规则: {a}')
        if a['knowledge'] not in KNOWLEDGE:
            raise ValueError('攻击者知识必须为 black_box/gray_box/white_box')
        if a['oracle_labels'] is not False:
            raise ValueError('P1-3 非 oracle 路径禁止攻击构造使用真实标签')
        if a['target_scope'] != 'evaluation_incident':
            raise ValueError('图攻击范围必须为 evaluation_incident')
        budget = float(a['budget'])
        if budget <= 0:
            raise ValueError('攻击预算必须为正')
        if a['budget_type'] in {'edge_fraction', 'evaluation_node_fraction'} \
                and budget > 1:
            raise ValueError('比例预算必须位于 (0,1]')
        if a['capability'] == 'feature_noise' and a['budget_type'] != 'feature_std':
            raise ValueError('特征噪声预算必须为 feature_std')
        if a['capability'].startswith('edge_') and a['budget_type'] != 'edge_fraction':
            raise ValueError('边攻击预算必须为 edge_fraction')
        names.append(a['name'])
    if len(set(names)) != len(names):
        raise ValueError('攻击名称不得重复')
    seeds = config.get('attack_seeds', [0])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('attack_seeds 必须非空且不重复')
    return True


def _stable_rng(name, attack_seed):
    value = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return np.random.RandomState((value + int(attack_seed)) % (2 ** 32))


#: 无向边对集合的缓存。np.unique 与 46 万条元组的 Python set 构造每次约 1 s,
#: 而一次攻击矩阵要调用数百次; 同一个 edge_index 的规范边对是不变量,
#: 因此按内容哈希缓存。只读缓存, 不改变任何返回值。
_EDGE_PAIR_CACHE = {}


def _edge_pairs(edge_index):
    edge = np.asarray(edge_index, dtype=np.int64)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError('edge_index 必须为 [2,E]')
    key = sha256_bytes(np.ascontiguousarray(edge).tobytes())
    hit = _EDGE_PAIR_CACHE.get(key)
    if hit is not None:
        return hit[0], hit[1]
    canonical = np.stack([np.minimum(edge[0], edge[1]),
                          np.maximum(edge[0], edge[1])])
    pairs = np.unique(canonical, axis=1)
    existing = set(map(tuple, canonical.T.tolist()))
    if len(_EDGE_PAIR_CACHE) > 4:
        _EDGE_PAIR_CACHE.clear()
    _EDGE_PAIR_CACHE[key] = (pairs, existing)
    return pairs, existing


def _evaluation_targets(p_prob, eval_mask, selection, quantile=0.5):
    p = np.asarray(p_prob, dtype=np.float64)
    candidates = np.flatnonzero(eval_mask)
    if not len(candidates):
        raise ValueError('评估节点为空，无法构造攻击')
    if selection != 'confidence_targeted':
        return candidates
    # 只看模型自己的高置信 illicit 预测，不看真实标签。
    positive = candidates[p[candidates] >= 0.5]
    pool = positive if len(positive) else candidates
    cutoff = np.quantile(p[pool], quantile)
    selected = pool[p[pool] >= cutoff]
    return selected if len(selected) else pool[np.argsort(-p[pool])[:1]]


def _delete_edges(edge_index, eval_mask, p_prob, spec, rng):
    edge = np.asarray(edge_index, dtype=np.int64)
    pairs, _ = _edge_pairs(edge)
    targets = _evaluation_targets(p_prob, eval_mask, spec['selection'],
                                  spec.get('target_quantile', 0.5))
    target_mask = np.zeros(len(p_prob), dtype=bool)
    target_mask[targets] = True
    eligible = target_mask[pairs[0]] | target_mask[pairs[1]]
    candidates = pairs[:, eligible]
    if not candidates.shape[1]:
        raise ValueError('没有符合评估范围的边对可删除')
    if spec['selection'] == 'random':
        order = rng.permutation(candidates.shape[1])
    else:
        # 攻击者按模型置信度与目标连接度排序；标签不参与。
        degree = np.bincount(edge.reshape(-1), minlength=len(p_prob))
        score = np.maximum(p_prob[candidates[0]], p_prob[candidates[1]]) \
            * np.log1p(degree[candidates[0]] + degree[candidates[1]])
        jitter = rng.random(len(score)) * 1e-12
        order = np.argsort(-(score + jitter), kind='stable')
    requested_pairs = max(1, int(round(edge.shape[1] * float(spec['budget']) / 2)))
    n_pairs = min(requested_pairs, candidates.shape[1])
    removed = candidates[:, order[:n_pairs]]
    remove_set = set(map(tuple, removed.T.tolist()))
    canonical = np.stack([np.minimum(edge[0], edge[1]),
                          np.maximum(edge[0], edge[1])])
    keep = np.array([tuple(e) not in remove_set for e in canonical.T], dtype=bool)
    new_edge = edge[:, keep]
    return new_edge, {'removed_undirected_pairs': int(n_pairs),
                      'removed_directed_edges': int((~keep).sum())}


def _inject_edges(edge_index, node_time, eval_mask, p_prob, spec, rng):
    edge = np.asarray(edge_index, dtype=np.int64)
    n = len(p_prob)
    _, existing = _edge_pairs(edge)
    targets = _evaluation_targets(p_prob, eval_mask, spec['selection'],
                                  spec.get('target_quantile', 0.5))
    degree = np.bincount(edge.reshape(-1), minlength=n)
    if spec['selection'] == 'degree_targeted':
        ranked = np.argsort(-degree, kind='stable')
        hubs = ranked[:max(2, min(n, int(0.05 * n) + 1))]
    else:
        hubs = np.arange(n)
    requested = max(2, 2 * int(round(edge.shape[1] * float(spec['budget']) / 2)))
    max_time_gap = spec.get('max_time_gap')
    new_pairs, new_set, attempts = [], set(), 0
    max_attempts = max(1000, requested * 100)
    while len(new_pairs) < requested // 2 and attempts < max_attempts:
        attempts += 1
        u = int(targets[rng.randint(len(targets))])
        v = int(hubs[rng.randint(len(hubs))])
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair in existing or pair in new_set:
            continue
        if max_time_gap is not None and abs(int(node_time[u]) - int(node_time[v])) \
                > int(max_time_gap):
            continue
        new_pairs.append(pair); new_set.add(pair)
    if not new_pairs:
        raise RuntimeError('在预算与约束内未找到可注入的非 oracle 边')
    pairs = np.asarray(new_pairs, dtype=np.int64).T
    additions = np.concatenate([pairs, pairs[::-1]], axis=1)
    new_edge = np.concatenate([edge, additions], axis=1)
    return new_edge, {'injected_undirected_pairs': int(pairs.shape[1]),
                      'injected_directed_edges': int(additions.shape[1]),
                      'candidate_attempts': int(attempts),
                      'requested_directed_edges': int(requested)}


def apply_nonoracle_attack(x, edge_index, node_time, eval_mask, clean_p_prob,
                           spec, attack_seed):
    """返回扰动输入与成本清单；函数签名没有标签入口。"""
    x = np.asarray(x, dtype=np.float32)
    edge = np.asarray(edge_index, dtype=np.int64)
    node_time = np.asarray(node_time, dtype=np.int64)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    p = np.asarray(clean_p_prob, dtype=np.float64).reshape(-1)
    if len(x) != len(node_time) or len(x) != len(p) or len(x) != len(eval_mask):
        raise ValueError('攻击输入的节点数不一致')
    if spec.get('oracle_labels') is not False:
        raise ValueError('禁止 oracle 标签攻击进入 P1-3 路径')
    rng = _stable_rng(spec['name'], attack_seed)
    capability = spec['capability']
    if capability == 'edge_delete':
        new_edge, cost = _delete_edges(edge, eval_mask, p, spec, rng)
        new_x = x
    elif capability == 'edge_inject':
        new_edge, cost = _inject_edges(edge, node_time, eval_mask, p, spec, rng)
        new_x = x
    elif capability == 'feature_noise':
        targets = _evaluation_targets(p, eval_mask, spec['selection'],
                                      spec.get('target_quantile', 0.5))
        new_x = x.copy()
        noise = rng.normal(0, float(spec['budget']),
                           size=(len(targets), x.shape[1])).astype(np.float32)
        new_x[targets] += noise
        new_edge = edge
        cost = {'noisy_evaluation_nodes': int(len(targets)),
                'feature_l2_cost': float(np.linalg.norm(noise)),
                'feature_changed_elements': int(noise.size)}
    else:
        raise ValueError(f'未知攻击能力: {capability}')
    manifest = {
        'protocol': 'p1_nonoracle_attack_v1', 'attack': dict(spec),
        'attack_seed': int(attack_seed), 'oracle_labels_used': False,
        'attack_construction_inputs': ['graph', 'features', 'clean_model_predictions'],
        'original_nodes': int(len(x)), 'original_directed_edges': int(edge.shape[1]),
        'perturbed_directed_edges': int(new_edge.shape[1]),
        'cost': cost,
        'x_sha256': sha256_bytes(np.ascontiguousarray(new_x).tobytes()),
        'edge_sha256': sha256_bytes(np.ascontiguousarray(new_edge).tobytes()),
    }
    return new_x, new_edge, manifest
