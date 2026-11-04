"""p4_thresholds.py — H1–H5 预先固定门限表及其设定依据 (H6 为开放项)

审稿意见 6: "门限 0.73 / 8ms 是任意设定, 且所有候选都未通过 (协议 6
28 次运行无 winner)"。本模块做三件事:

  1. 对每个门限标注 basis: derived(可推导) / operational(运维要求)
     / literature(文献惯例) / arbitrary(无依据)。
  2. basis='arbitrary' 的门限一律判为 unjustified 并撤回, 替换为可推导锚点。
  3. 生成与 p2 同 schema 的门限表并哈希锁定, 保证"预先固定"可追溯。

可推导锚点的公式
----------------
H1 F1 下限  = 等预算最强基线均值 F1 - 非劣性余量
              非劣性余量 = max(统计可分辨下限 MDE, 运维允许损失)
              其中 MDE 由种子数与种子间 SD 反推 (见 required_seeds_for_mde)。
H1 参数上限 = 设备权重内存预算 / 每参数字节数
H2 延迟上限 = 1 / 目标筛查吞吐 (QPS)  -> 换算到 per_node 口径
H3 改为区间估计: 报告比值点估计 + 95%CI, 不用 0.6/1.0 硬切
H4 时间外推 = 短训练窗 -> 参考训练窗的 F1 相对损失, 区间估计, 不设硬门限
H5 校准上限 = 保留 0.05 文献惯例, 但强制作用于"正类 ECE"而非整体 ECE
H6 攻击容限 = (告警处理容量 - 基线告警量) / 基线告警量 折算的 F1 损失上限;
              非 oracle 自适应攻击矩阵补齐前为开放项, 不参与 all_pass

所有替换门限必须在读取最终测试指标之前注册。
"""
import hashlib
import json
import math
import os

THRESHOLD_JUSTIFICATION_SCHEMA = 'p4_threshold_justification_v1'

BASIS_KINDS = ('derived', 'operational', 'literature', 'arbitrary')

# ------------------------------------------------------------------
# 可推导锚点
# ------------------------------------------------------------------
def required_seeds_for_mde(sd, mde, alpha=0.05, power=0.8):
    """给定种子间 SD 与希望分辨的最小效应, 反推所需种子数 (配对 t)。"""
    from scipy import stats
    if not sd or not mde:
        return None
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    n = ((z_a + z_b) * sd / mde) ** 2
    return int(math.ceil(n))


def mde_for_seeds(sd, n, alpha=0.05, power=0.8):
    """给定种子数与 SD, 反推在 80% 功效下可分辨的最小效应。"""
    from scipy import stats
    if not sd or n < 2:
        return None
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    return float((z_a + z_b) * sd / math.sqrt(n))


def derive_f1_floor(baseline_f1, baseline_sd=None, n_seeds=10,
                    alpha=0.05, power=0.8, operational_margin=0.0):
    """H1 的 F1 下限: 非劣性锚点。

    floor = baseline_f1 - max(MDE(n_seeds, sd), operational_margin)
    """
    mde = mde_for_seeds(baseline_sd, n_seeds, alpha, power) \
        if baseline_sd else None
    margin = max([x for x in (mde, operational_margin) if x is not None] or
                 [0.0])
    return {
        'value': float(baseline_f1 - margin),
        'baseline_f1': float(baseline_f1),
        'margin': float(margin),
        'mde_component': mde, 'operational_margin': float(operational_margin),
        'formula': 'floor = baseline_f1 - max(MDE(n, sd), operational_margin)',
        'basis': 'derived',
    }


def derive_params_budget(memory_kb, dtype_bytes=4):
    """H1 参数上限: 由设备权重内存预算反推。"""
    return {'value': int(memory_kb * 1024 / dtype_bytes),
            'formula': 'params_max = memory_budget_bytes / bytes_per_param',
            'memory_kb': memory_kb, 'dtype_bytes': dtype_bytes,
            'basis': 'operational',
            'requires': '必须写明目标设备的内存预算来源'}


def derive_latency_budget(target_qps, semantics='per_node_screening'):
    """H2 延迟上限: 由目标筛查吞吐反推。

    semantics='per_node_screening' -> 单笔预算 = 1/QPS 秒
    semantics='full_graph_forward' -> 整图预算 = 节点数/QPS 秒
    """
    budget_s = 1.0 / float(target_qps)
    return {
        'value_ms': budget_s * 1000.0,
        'target_qps': float(target_qps),
        'semantics': semantics,
        'formula': 'budget = 1 / target_qps (per_node) '
                   '或 N / target_qps (full_graph)',
        'basis': 'operational',
        'requires': '必须写明目标 QPS 的业务来源, 且口径必须与测量口径一致',
    }


#: H6 的攻击矩阵来自 P1 已声明并校验过的非 oracle 协议文件。登记时只引用
#: 它的哈希, 不在门限表里复制一份攻击清单, 避免两份定义漂移。
H6_ATTACK_MATRIX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'p1_attack_matrix.json')


def h6_measurement_protocol(attack_matrix_path=None, n_model_seeds=10,
                            eval_window=(35, 49), dataset='elliptic'):
    """H6 的预登记测量协议 (v3)。

    H1/H2/H5 登记的是 pass/fail 门限; H6 没有可推导的容限锚点
    (告警处理容量未知, derive_attack_tolerance 返回 None), 所以这里登记
    的是"测什么、以什么为配对单位、怎么聚合", 而不是一个数字。

    必须在跑攻击矩阵之前写进锁定表: 否则先看到结果再挑攻击/挑口径,
    就等于事后定门限, 与本项目撤回 V0 任意门限的理由自相矛盾。
    """
    import p1_attacks
    path = attack_matrix_path or H6_ATTACK_MATRIX_PATH
    with open(path, 'rb') as f:
        cfg_bytes = f.read()
    cfg = json.loads(cfg_bytes.decode('utf-8'))
    p1_attacks.validate_attack_matrix(cfg)
    sha = hashlib.sha256(cfg_bytes).hexdigest()
    declared = [{'name': a['name'], 'capability': a['capability'],
                 'selection': a['selection'], 'knowledge': a['knowledge'],
                 'goal': a['goal'], 'budget_type': a['budget_type'],
                 'budget': a['budget'], 'oracle_labels': False}
                for a in cfg['attacks']]
    knowledge = sorted({a['knowledge'] for a in cfg['attacks']})
    return {
        'schema': 'p6_h6_measurement_protocol_v1',
        'dataset': dataset,
        'evaluation_window': list(eval_window),
        'evaluation_window_role': ('same test split as the main results; '
                                   'labels enter only the scoring step'),
        'attack_construction_inputs': ['graph structure', 'node features',
                                       'clean model predictions'],
        'oracle_labels': False,
        'threshold_protocol': ('decision threshold tuned once on the clean '
                               'validation split and frozen; no re-tuning on '
                               'attacked data'),
        'paired_unit': 'model seed',
        'n_model_seeds': int(n_model_seeds),
        'attack_seeds': list(cfg['attack_seeds']),
        'attack_seed_role': ('nested repeat inside one model seed; not an '
                             'independent unit and not counted as extra seeds'),
        'aggregation': ('mean over attack seeds within a model seed, then '
                        'bootstrap over model seeds'),
        'quantities': ['clean_f1', 'attacked_f1', 'delta_f1',
                       'relative_drop_ratio'],
        'relative_drop_ratio': ('(clean_f1 - attacked_f1) / clean_f1; '
                                'inconclusive when clean_f1 <= 0'),
        'interval': 'bootstrap 95% CI over model seeds',
        'declared_attacks': declared,
        'attacker_knowledge_declared': knowledge,
        'attacker_knowledge_not_declared': sorted(
            set(p1_attacks.KNOWLEDGE) - set(knowledge)),
        'cost_declaration': ('per attack: budget_type, budget, and the '
                             'realised cost manifest (perturbed edge pairs '
                             'or feature L2 norm)'),
        'gate': None,
        'gate_reason': ('alert_capacity_per_day is unknown, so '
                        'derive_attack_tolerance returns no value; any pass '
                        'or fail cut-off here would be an arbitrary number'),
        'attack_matrix_sha256': sha,
        'attack_matrix_protocol': cfg['protocol'],
    }


def derive_attack_tolerance(alert_capacity_per_day, base_alerts_per_day):
    """H4 攻击容限: 由告警处理容量反推可容忍的 F1 相对损失上限。"""
    if not alert_capacity_per_day or not base_alerts_per_day:
        return {'value': None, 'basis': 'operational',
                'requires': '缺少告警容量与基线告警量, 无法推导; '
                            '必须改为报告区间估计'}
    slack = (alert_capacity_per_day - base_alerts_per_day) / \
        float(base_alerts_per_day)
    return {'value': float(max(slack, 0.0)),
            'formula': 'tolerance = (capacity - base_alerts) / base_alerts',
            'alert_capacity_per_day': alert_capacity_per_day,
            'base_alerts_per_day': base_alerts_per_day,
            'basis': 'operational'}


def derive_ece_ceiling(bin_count=15, convention='guo2017',
                       apply_to='positive_class_ece'):
    """H5 校准上限: 文献惯例 (Guo et al. 2017, 15 等宽分箱)。"""
    return {
        'value': 0.05, 'bin_count': bin_count, 'convention': convention,
        'apply_to': apply_to, 'basis': 'literature',
        'citation': 'Guo, Pleiss, Sun & Weinberger, ICML 2017, §5',
        'caveat': ('整体 ECE 在正类占比 ~6.5% 时会退化到接近 0 '
                   '(绝大多数样本落在"预测≈0 / 真实≈0"的箱内), '
                   '因此 0.05 必须作用于类别质量加权的正类 ECE'),
    }


# ------------------------------------------------------------------
# 门限依据表
# ------------------------------------------------------------------
def build_justification_table(anchors=None):
    """返回 H1–H5 每个门限的设定依据与裁定。

    anchors 给出时, H1_f1 的"为何任意"说明用实测锚点数值生成, 避免文案里的
    基线数字与最终等预算结果脱节 (终版曾残留 HPO 修复前的 0.5367)。
    """
    a = anchors or {}
    bf1 = a.get('best_baseline_f1')
    bname = a.get('best_baseline_name') or '最强等预算基线'
    if bf1 is None:
        why_h1 = ('设计方案未给出 0.73 的任何来源: 既非基线锚定, 也非运维要求, '
                  '也非文献惯例; 0.73 明显高于实测等预算基线, 在本任务上不可达。')
    else:
        gap = 0.73 - float(bf1)
        why_h1 = ('设计方案未给出 0.73 的任何来源: 既非基线锚定, 也非运维要求, '
                  '也非文献惯例; 等预算 HPO 修复后实测最强基线 %s F1=%.4f, '
                  '0.73 高出 %.4f, 在本任务上不可达。' % (bname, float(bf1), gap))
    return {
        'schema': THRESHOLD_JUSTIFICATION_SCHEMA,
        'rule': ('basis=arbitrary 的门限一律撤回; 撤回后若无可推导替代, '
                 '该假设改为"报告区间估计, 不设通过/证伪硬门限"'),
        'hypotheses': {
            'H1_f1': {
                'original_value': 0.73,
                'original_basis': 'arbitrary',
                'why_arbitrary': why_h1,
                'verdict': 'unjustified_withdrawn',
                'replacement': 'derive_f1_floor(best_equal_budget_baseline_f1, sd, n_seeds)',
                'replacement_basis': 'derived',
                'replacement_note': ('非劣性锚点: TASGL 只需"不劣于等预算最强基线 '
                                     '超过一个统计可分辨的余量", 而不是达到一个绝对数'),
            },
            'H1_params': {
                'original_value': 20000,
                'original_basis': 'operational',
                'why': '80 KB fp32 权重预算反推 (=20000×4B), 但目标设备未写明',
                'verdict': 'justified_conditional',
                'replacement': 'derive_params_budget(memory_kb)',
                'condition': '必须在论文中写明目标设备与其内存预算来源',
            },
            'H2_latency': {
                'original_value': 8.0,
                'original_basis': 'arbitrary',
                'why_arbitrary': ('8 ms 无业务来源; 更严重的是口径未定义: '
                                  '现有 measure_efficiency 测的是"整图一次前向" '
                                  '(实测 33.3 ms / 203769 节点 = 0.16 µs/节点), '
                                  '与"单笔交易 8 ms"不是一个量, 无法比较。'),
                'verdict': 'unjustified_withdrawn',
                'replacement': 'derive_latency_budget(target_qps, semantics)',
                'replacement_basis': 'operational',
                'replacement_note': '先声明目标 QPS 与口径, 再换算门限',
            },
            'H3_ratio': {
                'original_value': 0.6,
                'original_basis': 'arbitrary',
                'why_arbitrary': '0.60 / 1.00 的切点无推导, 且存在灰区, 硬切会把噪声判成结论',
                'verdict': 'unjustified_withdrawn',
                'replacement': '报告 relative_drop_ratio 的点估计 + 95%CI '
                               '(bootstrap over seeds), 不设通过/证伪硬门限',
                'replacement_basis': 'derived',
            },
            'H4_attack': {
                'original_value': 0.05,
                'original_basis': 'arbitrary',
                'why_arbitrary': ('5% / 8% 无告警成本模型支撑; 且当前攻击实验是 '
                                  'oracle 注入式, 不是真实自适应攻击'),
                'verdict': 'unjustified_withdrawn',
                'replacement': 'derive_attack_tolerance(capacity, base_alerts)',
                'replacement_basis': 'operational',
                'replacement_note': ('在补齐非 oracle 自适应攻击矩阵之前, '
                                     'H4 判为 not_testable'),
            },
            'H5_ece': {
                'original_value': 0.05,
                'original_basis': 'literature',
                'why': 'Guo et al. 2017 的 15 分箱 ECE 惯例, 有出处但作用对象错了',
                'verdict': 'justified_after_repair',
                'replacement': 'derive_ece_ceiling(apply_to="positive_class_ece")',
                'replacement_basis': 'literature',
                'repair': ('0.05 必须作用于类别质量加权的正类 ECE; '
                           '整体 ECE 在不平衡下退化, 不得单独使用'),
            },
        },
    }


def register_justified_thresholds(anchors, out_dir='results/p4/thresholds',
                                  declared_seeds=None, meta=None,
                                  h6_protocol=None):
    """把可推导锚点固化为 p2 同 schema 的门限表并哈希锁定。

    anchors: {'best_baseline_f1':.., 'baseline_sd':.., 'n_seeds':..,
              'target_qps':.., 'memory_kb':.., 'alert_capacity_per_day':..,
              'base_alerts_per_day':..}

    h6_protocol: h6_measurement_protocol() 的返回值。给出它就把 H6 从
    "不可测的开放项"升级为"已预登记测量协议、实测后只报区间"的条目,
    登记版本随之升到 v3。不给出它时保持 v2 语义, 旧调用与旧测试不受影响。
    """
    a = anchors or {}
    seeds = declared_seeds or list(range(20))
    table = {
        'schema': 'p2_hypothesis_threshold_table_v1',
        'declared_seeds': seeds,
        'statistics_protocol': {
            'paired_unit': 'seed', 'test': 'wilcoxon_exact_two_sided',
            'alpha': 0.05, 'sample_sd_ddof': 1,
            'multiple_comparison': 'Holm', 'seed_pool_sha256_required': True,
        },
        'registration_rule': '本表必须在读取任何最终测试指标之前注册',
        'justification_schema': THRESHOLD_JUSTIFICATION_SCHEMA,
        'hypotheses': {},
    }

    f1 = a.get('best_baseline_f1')
    if f1 is None:
        table['hypotheses']['H1'] = {
            'statement': 'H1 缺少等预算基线 F1 锚点, 无法推导门限',
            'status': 'not_registered_missing_anchor', 'gates': []}
    else:
        fl = derive_f1_floor(f1, a.get('baseline_sd'),
                             a.get('n_seeds', len(seeds)))
        pm = derive_params_budget(a.get('memory_kb', 80))
        table['hypotheses']['H1'] = {
            'statement': (f"均值 F1 >= {fl['value']:.4f} (等预算最强基线非劣性锚点) "
                          f"且部署参数量 <= {pm['value']}"),
            'basis': 'derived',
            'derivation': fl, 'params_budget': pm,
            'requires_seed_completeness': True,
            'gates': [
                {'metric': 'mean_f1', 'pass_op': '>=',
                 'pass_value': round(fl['value'], 6),
                 'falsify_op': '<', 'falsify_value': round(fl['value'], 6),
                 'on_fail': 'falsified', 'else': 'falsified'},
                {'metric': 'deploy_params', 'pass_op': '<=',
                 'pass_value': pm['value'], 'falsify_op': '>',
                 'falsify_value': pm['value'], 'on_fail': 'falsified',
                 'else': 'falsified'},
            ],
            'report_only': ['sample_sd_f1', 'ci95_f1', 'cohen_dz',
                            'non_inferiority_margin'],
        }

    qps = a.get('target_qps')
    if qps is None:
        table['hypotheses']['H2'] = {
            'statement': 'H2 缺少目标 QPS, 无法推导延迟门限',
            'status': 'not_registered_missing_anchor', 'gates': []}
    else:
        lb = derive_latency_budget(qps, a.get('latency_semantics',
                                              'per_node_screening'))
        table['hypotheses']['H2'] = {
            'statement': (f"{lb['semantics']} 口径下延迟 <= "
                          f"{lb['value_ms']:.6g} ms (目标 QPS={qps})"),
            'basis': 'operational', 'derivation': lb,
            'requires_seed_completeness': True,
            'gates': [{'metric': 'mean_latency_ms', 'pass_op': '<=',
                       'pass_value': round(lb['value_ms'], 9),
                       'falsify_op': '>',
                       'falsify_value': round(lb['value_ms'], 9),
                       'on_fail': 'falsified', 'else': 'falsified'}],
            'report_only': ['measurement_semantics', 'n_warmup', 'n_repeat',
                            'sync', 'environment_fingerprint',
                            'full_chain_seconds'],
        }

    table['hypotheses']['H3'] = {
        'statement': ('改为区间估计: 报告 relative_drop_ratio 点估计与 '
                      '种子 bootstrap 95%CI, 不设 0.6/1.0 硬门限'),
        'basis': 'derived', 'gate_removed': True,
        'requires_seed_completeness': True, 'gates': [],
        'report_only': ['relative_drop_ratio', 'ci95', 'per_baseline'],
    }
    # --- H4: 时间外推 (短训练窗 -> 参考训练窗, 同一评测区间) ---
    # 历史登记把"攻击容限"误写进了 H4。攻击不属于时间外推, 单列为 H6。
    # 这次拆分不改动任何已登记的门限数值: H4 原本就没有 gate, H1/H2/H5 的
    # 数值一字未动, 因此任何 pass/falsified 判决都不受影响。
    table['hypotheses']['H4'] = {
        'statement': ('区间估计: 报告短训练窗相对参考训练窗的 F1 相对损失点估计'
                      '与种子 bootstrap 95%CI, 不设硬门限'),
        'basis': 'derived', 'gate_removed': True,
        'quantity': ('relative_loss = (f1_reference - f1_short) / f1_reference, '
                     '两套 schedule 使用同一评测区间'),
        'requires_seed_completeness': True, 'gates': [],
        'report_only': ['f1_reference', 'f1_short', 'relative_loss', 'ci95',
                        'train_schedule_short', 'train_schedule_reference',
                        'eval_interval'],
    }

    # --- H6: 对抗鲁棒性 ---
    # v3: 攻击矩阵补齐前先登记"怎么测", 再执行; 没有容限锚点就不设门限。
    # v2 及更早: 判为 not_testable 的开放项。两条路径都不给 H6 任何 pass。
    tol = derive_attack_tolerance(a.get('alert_capacity_per_day'),
                                  a.get('base_alerts_per_day'))
    if h6_protocol is not None:
        n_atk = len(h6_protocol.get('declared_attacks') or [])
        n_aseed = len(h6_protocol.get('attack_seeds') or [])
        table['hypotheses']['H6'] = {
            'statement': ('区间报告: 在预登记的非 oracle 攻击矩阵 (%d 个攻击 x '
                          '%d 个攻击种子) 下报告 clean 与 attacked 的 illicit F1 '
                          '及相对下降的点估计与 %d-seed bootstrap 95%%CI; '
                          '不设硬门限') % (n_atk, n_aseed, len(seeds)),
            'basis': 'operational', 'status': 'measured_no_gate',
            'gates': [], 'open_item': False,
            'measurement_protocol': h6_protocol,
            'why_no_gate': h6_protocol.get('gate_reason'),
            'report_only': ['clean_f1', 'attack_f1', 'relative_drop_ratio',
                            'ci95', 'attacker_model', 'oracle_free',
                            'attack_cost']}
    elif tol['value'] is None:
        table['hypotheses']['H6'] = {
            'statement': ('H6 (对抗鲁棒性) 在非 oracle 自适应攻击矩阵补齐前判为 '
                          'not_testable; 它不在 H1–H5 的联合门限集内'),
            'basis': 'operational', 'status': 'not_testable', 'gates': [],
            'open_item': True,
            'missing': ('非 oracle 自适应攻击者模型: 现有 E5 边扰动由已知标签'
                        '构造, 属于 oracle, 增益不能解释为对抗鲁棒性'),
            'report_only': ['clean_f1', 'attack_f1', 'attacker_model',
                            'oracle_free']}
    else:
        table['hypotheses']['H6'] = {
            'statement': f"攻击下 F1 相对损失 <= {tol['value']:.4f}",
            'basis': 'operational', 'derivation': tol,
            'requires_seed_completeness': True,
            'gates': [{'metric': 'relative_f1_loss', 'pass_op': '<=',
                       'pass_value': round(tol['value'], 6),
                       'falsify_op': '>', 'falsify_value': round(tol['value'], 6),
                       'on_fail': 'falsified', 'else': 'falsified'}],
            'report_only': ['clean_f1', 'attack_f1', 'attacker_model',
                            'oracle_free']}

    ec = derive_ece_ceiling()
    table['hypotheses']['H5'] = {
        'statement': (f"{ec['apply_to']} <= {ec['value']} 且低于同骨干基线 "
                      f"的同类指标"),
        'basis': 'literature', 'derivation': ec,
        'requires_seed_completeness': True,
        'gates': [
            {'metric': 'positive_class_ece', 'pass_op': '<=',
             'pass_value': ec['value'], 'falsify_op': '>',
             'falsify_value': ec['value'],
             'on_fail': 'inconclusive_above_absolute_ceiling',
             'else': 'inconclusive_above_absolute_ceiling'},
            {'metric': 'positive_class_ece', 'pass_op': '<',
             'compare_to_metric': 'baseline_positive_class_ece',
             'on_fail': 'falsified', 'else': 'falsified'},
        ],
        'report_only': ['negative_class_ece', 'overall_ece', 'brier', 'nll',
                        'bins', 'weighting', 'calibration_window'],
    }
    h6_note = ('H6 已实测但无容限锚点, 只报区间, 不参与 all_pass 判定'
               if h6_protocol is not None else
               'H6 为开放项, 不参与 all_pass 判定')
    table['all_pass_rule'] = ('仅当 H1–H5 全部为 pass 时 all_pass=true; '
                              'inconclusive / falsified / incomplete_* / '
                              'not_testable 均不得写成达标; ' + h6_note)
    table['registration_version'] = ('p4_threshold_table_v3'
                                     if h6_protocol is not None else
                                     'p4_threshold_table_v2')
    table['registration_changelog'] = [
        {'version': 'v1',
         'change': ('初版登记: H1–H5, 其中 H4 的判定规则栏误写入攻击描述'),
         'threshold_values': ('H1 F1>=0.431972, params<=20480; '
                              'H2 <=142.857142857 ms; H5 ECE<=0.05')},
        {'version': 'v2',
         'change': ('H4 重述为时间外推(区间估计); 攻击项拆出单列为 H6, '
                    '标为开放项, 不参与 all_pass'),
         'threshold_values_changed': False,
         'note': ('拆分只是修正误归档。H4 在 v1 与 v2 中都没有 gate, '
                  'H1/H2/H5 的门限数值完全不变, 因此不改变任何已有判决')},
    ]
    if h6_protocol is not None:
        table['registration_changelog'].append({
            'version': 'v3',
            'change': ('H6 从 not_testable 升级为"预登记测量协议 + 区间报告": '
                       '在执行攻击矩阵之前锁定攻击清单、攻击者知识、配对单位与'
                       '聚合方式'),
            'threshold_values_changed': False,
            'note': ('H6 仍然没有 gate: 告警容量锚点缺失, 任何通过线都会是任意值。'
                     'H1/H2/H5 的门限数值一字未动, 因此不改变任何已有判决'),
            'registered_before_execution': True,
        })
    body = {'schema': table['schema'], 'hypotheses': table['hypotheses']}
    lock = {'schema': 'p4_threshold_lock_v1',
            'table_sha256': hashlib.sha256(
                json.dumps(body, sort_keys=True, ensure_ascii=False,
                           default=str).encode('utf-8')).hexdigest()[:24],
            'declared_seeds_sha256': hashlib.sha256(
                json.dumps(seeds).encode('utf-8')).hexdigest()[:24],
            'meta': meta or {}}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'thresholds.json'), 'w',
              encoding='utf-8') as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, 'thresholds.lock.json'), 'w',
              encoding='utf-8') as f:
        json.dump(lock, f, indent=2, ensure_ascii=False)
    return {'table': table, 'lock': lock, 'lock_id': lock['table_sha256'],
            'table_path': os.path.join(out_dir, 'thresholds.json'),
            'lock_path': os.path.join(out_dir, 'thresholds.lock.json')}


def audit_justification(table):
    """逐个假设裁定门限是否带依据。"""
    out = {}
    for hid, h in (table.get('hypotheses') or {}).items():
        basis = h.get('basis')
        if h.get('status') in ('not_testable', 'not_registered_missing_anchor'):
            verdict = 'not_registered'
        elif basis == 'arbitrary':
            verdict = 'unjustified'
        elif basis in ('derived', 'operational', 'literature'):
            verdict = 'justified'
        else:
            verdict = 'unknown'
        out[hid] = {'basis': basis, 'verdict': verdict,
                    'has_gates': bool(h.get('gates')),
                    'derivation_present': bool(h.get('derivation'))}
    return {'schema': 'p4_threshold_audit_v1', 'per_hypothesis': out,
            'all_justified': all(v['verdict'] in ('justified', 'not_registered')
                                 for v in out.values())}
