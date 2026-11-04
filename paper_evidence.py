#!/usr/bin/env python3
"""从只读原始证据包确定性重建 TASGL 论文表格、绘图输入和血缘清单。

本程序不训练模型、不读取测试标签、不修改 ZIP 或历史 results，也不把不同协议
合并为同一性能结果。所有统计均从包内逐种子 JSON 重算，并与原汇总交叉校验。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, BadZipFile

SCHEMA_VERSION = 1
GENERATOR = "paper_evidence.py"
MODEL_SPECS = [
    ("tasgl", "TASGL"), ("mlp", "MLP"), ("gcn", "GCN"),
    ("gat", "GAT"), ("graphsage", "GraphSAGE"), ("sgc", "SGC"),
    ("dsgat", "DSGAT-proxy"),
]
ABLATION_SPECS = [
    ("full", "full"), ("w/o_rff", "w/o cosine propagation"),
    ("w/o_sparse", "w/o sparse gating"),
    ("w/o_time_basis", "w/o Fourier time"),
    ("w/o_lowrank", "w/o low-rank"),
    ("w/o_dro", "w/o DRO-inspired reg."),
    ("w/o_contrastive", "w/o contrastive"),
]
TEMPORAL_MODELS = [("tasgl", "TASGL"), ("gcn", "GCN"), ("gat", "GAT")]
ROBUST_CONDITIONS = [
    ("clean", "0"), ("feature_noise", "0.1"), ("feature_noise", "0.3"),
    ("feature_noise", "0.5"), ("edge_drop", "0.1"), ("edge_drop", "0.3"),
    ("edge_drop", "0.5"), ("edge_inject", "0.1"), ("edge_inject", "0.3"),
    ("label_flip", "0.05"), ("label_flip", "0.10"),
]
PROTOCOL6_CANDIDATES = [
    "lite_h48", "lite_h32", "lite_kd", "graphless_kd", "lite_robust",
    "lite_unweighted", "lite_temp",
]
GATE_SPECS = [
    ("H1_f1", "H1 F1"), ("H1_params", "H1 size"),
    ("H2_latency", "H2 latency"), ("H3_absolute_drop_proxy", "H3 proxy"),
    ("H4_extrapolation_proxy", "H4 proxy"), ("H5_ece_proxy", "H5 ECE"),
]
EVD_BOUNDARIES = {
    "main": ("报告五种子协议 3 的记录配置、少数类指标与实测成本。",
             "不得声称精度领先、公平 HPO 已完成或独立确认。"),
    "ablation": ("报告三种子均值差，作为模块修订诊断。",
                "不得写成显著因果效应或所有模块均必要。"),
    "temporal": ("报告固定测试 35–49 的训练历史敏感性。",
                "不得称为严格 inductive rolling；协议 4 不与协议 3 混池。"),
    "robust": ("保留两种子历史压力诊断及 oracle 边注入限定。",
               "不得与非 oracle P1 攻击混合，也不得把 F1 上升写成防御收益。"),
    "e7": ("报告独立 seed-0 校准和选择性风险描述。",
           "不得拼入主五种子或协议 5，也不得宣称单调风险控制。"),
    "crosschain": ("仅作独立 PCA、无语义对齐的弱跨域诊断。",
                   "不得声称迁移优势或严格跨链证伪。"),
    "protocol5": ("报告锁定后的探索性确认及退化 H3、EDL proxy 边界。",
                  "不得称为独立确认、图鲁棒性或完整 ST-CGNN 复现。"),
    "protocol6": ("仅报告 33–34 开发门控与授权停止状态。",
                  "不得作为 35–49 最终性能，不得绕过无 winner 的停止规则。"),
    "package": ("记录证据包、成员和生成产物哈希。",
                "哈希证明字节身份，不证明模型性能、外部标注或硬件型号。"),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def finite_number(value, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: 需要有限数值，得到 {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: 非有限数值")
    return result


def mean_sd(values, ddof: int = 1):
    vals = [finite_number(v, "statistic input") for v in values]
    if not vals:
        raise ValueError("空数组不能汇总")
    if ddof not in (0, 1):
        raise ValueError("ddof 仅允许 0 或 1")
    if len(vals) <= ddof:
        raise ValueError(f"n={len(vals)} 无法计算 ddof={ddof} 的标准差")
    mean = statistics.fmean(vals)
    variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
    return mean, math.sqrt(max(0.0, variance))


def summarize(values, ddof: int = 1):
    mean, sd = mean_sd(values, ddof)
    return {"n": len(values), "mean": mean, "sd": sd, "ddof": ddof}


def assert_close(actual, expected, context: str, tolerance: float = 1e-10):
    a = finite_number(actual, context)
    b = finite_number(expected, context)
    if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{context}: 重算 {a!r} 与原汇总 {b!r} 不一致")


def assert_equivalent(actual, expected, context: str):
    """递归比较原始记录与汇总内嵌记录；浮点仅容许序列化级误差。"""
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{context}: 键集合不一致")
        for key in sorted(actual):
            assert_equivalent(actual[key], expected[key], f"{context}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{context}: 数组长度不一致")
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_equivalent(a, b, f"{context}[{i}]")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) \
            and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert_close(actual, expected, context)
    elif actual != expected:
        raise ValueError(f"{context}: 原始值 {actual!r} != 汇总值 {expected!r}")


def require_seed_records(records, seeds, protocol: int, context: str):
    if len(records) != len(seeds):
        raise ValueError(f"{context}: 记录数不等于声明种子数")
    actual = [r.get("seed") for r in records]
    if actual != list(seeds):
        raise ValueError(f"{context}: 种子顺序 {actual} != {list(seeds)}")
    for record in records:
        if record.get("protocol") != protocol:
            raise ValueError(f"{context}: 协议字段不等于 {protocol}")
        if record.get("complete") is False:
            raise ValueError(f"{context}: 含未完成记录")


def exact_wilcoxon_signed_rank(x, y):
    """精确双侧 Wilcoxon；零差值剔除，符号翻转穷举，支持并列平均秩。"""
    if len(x) != len(y) or not x:
        raise ValueError("Wilcoxon 需要等长非空配对")
    pairs = [(finite_number(a, "x"), finite_number(b, "y")) for a, b in zip(x, y)]
    differences = [a - b for a, b in pairs if a != b]
    n = len(differences)
    if n == 0:
        raise ValueError("全部配对差为零，精确 Wilcoxon 无定义")
    ordered = sorted(enumerate(differences), key=lambda item: abs(item[1]))
    ranks = [0.0] * n
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and abs(ordered[end][1]) == abs(ordered[pos][1]):
            end += 1
        average_rank = (pos + 1 + end) / 2.0
        for k in range(pos, end):
            ranks[ordered[k][0]] = average_rank
        pos = end
    w_plus = sum(rank for rank, diff in zip(ranks, differences) if diff > 0)
    total_rank = sum(ranks)
    statistic = min(w_plus, total_rank - w_plus)
    distribution = []
    for signs in itertools.product((0, 1), repeat=n):
        distribution.append(sum(rank for rank, sign in zip(ranks, signs) if sign))
    lower = sum(value <= w_plus + 1e-12 for value in distribution) / len(distribution)
    upper = sum(value >= w_plus - 1e-12 for value in distribution) / len(distribution)
    p_value = min(1.0, 2.0 * min(lower, upper))
    return {"statistic": statistic, "p_value": p_value, "n_nonzero": n,
            "minimum_exact_two_sided_p": min(1.0, 2.0 / (2 ** n))}


def check_selective_curve(f1: float, coverage, risk, context: str):
    if len(coverage) != len(risk) or not coverage:
        raise ValueError(f"{context}: 覆盖率与风险曲线长度不一致")
    if not math.isclose(float(coverage[-1]), 1.0, abs_tol=1e-12):
        raise ValueError(f"{context}: 最后一点不是全覆盖")
    for c, r in zip(coverage, risk):
        finite_number(c, f"{context}.coverage")
        finite_number(r, f"{context}.risk")
        if not 0 < float(c) <= 1:
            raise ValueError(f"{context}: 覆盖率越界")
    assert_close(float(risk[-1]), 1.0 - float(f1), f"{context}: 全覆盖风险必须等于 1-F1")


class EvidencePackage:
    """只读 ZIP 访问、成员哈希登记与独立协议包等价校验。"""

    def __init__(self, path: Path, expected_sha256: str | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"证据包不存在: {self.path}")
        self.sha256 = file_sha256(self.path)
        if expected_sha256 and self.sha256 != expected_sha256.lower():
            raise ValueError(
                f"证据包 SHA-256 不匹配: 实际 {self.sha256}, 预期 {expected_sha256.lower()}")
        try:
            self.zip = ZipFile(self.path)
        except BadZipFile as exc:
            raise ValueError(f"无法读取证据 ZIP: {self.path}") from exc
        names = [i.filename for i in self.zip.infolist() if not i.is_dir()]
        duplicates = [n for n, count in Counter(names).items() if count > 1]
        if duplicates:
            raise ValueError(f"ZIP 含重复成员: {duplicates[:3]}")
        self.names = set(names)
        self.sources = {}
        self.cache = {}
        bad = self.zip.testzip()
        if bad:
            raise ValueError(f"ZIP CRC 校验失败: {bad}")

    def close(self):
        self.zip.close()

    def read_bytes(self, member: str) -> bytes:
        path = PurePosixPath(member)
        if path.is_absolute() or ".." in path.parts or member not in self.names:
            raise ValueError(f"非法或缺失的 ZIP 成员: {member}")
        payload = self.zip.read(member)
        previous = self.sources.get(member)
        digest = sha256_bytes(payload)
        if previous and previous != digest:
            raise ValueError(f"成员读取期间哈希变化: {member}")
        self.sources[member] = digest
        return payload

    def json(self, member: str):
        if member not in self.cache:
            try:
                self.cache[member] = json.loads(self.read_bytes(member).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"JSON 无法解析: {member}") from exc
        return self.cache[member]

    def package_record(self):
        return {"filename": self.path.name, "sha256": self.sha256,
                "role": "combined_v2_protocol5_protocol6", "member_count": len(self.names)}


def validate_nested_package(combined: EvidencePackage, nested_path: Path, prefix: str):
    nested_path = Path(nested_path)
    digest = file_sha256(nested_path)
    with ZipFile(nested_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"{nested_path.name}: CRC 失败 {bad}")
        standalone = {n for n in archive.namelist() if not n.endswith("/")}
        embedded = {n[len(prefix):] for n in combined.names if n.startswith(prefix)}
        if standalone != embedded:
            raise ValueError(f"{nested_path.name}: 与合并包 {prefix} 成员集合不一致")
        for name in sorted(standalone):
            if sha256_bytes(archive.read(name)) != sha256_bytes(combined.read_bytes(prefix + name)):
                raise ValueError(f"{nested_path.name}: 成员 {name} 与合并包内容不一致")
            combined.json(prefix + name) if name.endswith(".json") else combined.read_bytes(prefix + name)
    return {"filename": nested_path.name, "sha256": digest,
            "role": f"standalone_{prefix.rstrip('/')}",
            "member_count": len(standalone),
            "embedded_members_byte_identical": True}


def build_main(ev: EvidencePackage):
    summary = ev.json("main_results.json")
    records_by_model = {}
    rows = []
    per_seed_rows = []
    checks = []
    for model, label in MODEL_SPECS:
        records = [ev.json(f"runs/{model}_seed{seed}.json") for seed in range(5)]
        require_seed_records(records, range(5), 3, f"main.{model}")
        if model not in summary or len(summary[model]) != 5:
            raise ValueError(f"main_results.json 缺少 {model} 五种子汇总")
        for raw, cached in zip(records, summary[model]):
            assert_equivalent(raw, cached, f"main_results.{model}.seed{raw['seed']}")
        records_by_model[model] = records
        row = {"model_key": model, "model": label, "protocol": 3,
               "seeds": [r["seed"] for r in records]}
        for metric in ("f1", "pr_auc", "recall_at_fpr_1pct", "latency_ms",
                       "peak_mem_mb"):
            stats = summarize([r[metric] for r in records], ddof=1)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_sample_sd"] = stats["sd"]
        if model == "tasgl":
            stats = summarize([r["ece"] for r in records], ddof=1)
            row["ece_mean"] = stats["mean"]
            row["ece_sample_sd"] = stats["sd"]
        params = {int(r["params"]) for r in records}
        if len(params) != 1:
            raise ValueError(f"main.{model}: 参数量随种子变化")
        row["parameters"] = params.pop()
        if model == "tasgl":
            active = [r["active_edge_ratio"] for r in records]
            stats = summarize(active, ddof=1)
            row["active_edge_ratio_mean"] = stats["mean"]
            row["active_edge_ratio_sample_sd"] = stats["sd"]
            row["active_edge_ratio_min"] = min(active)
            row["active_edge_ratio_max"] = max(active)
        rows.append(row)
        for r in records:
            per_seed_rows.append({
                "model_key": model, "model": label, "protocol": 3, "seed": r["seed"],
                "threshold": r["threshold"], "precision": r["precision"],
                "recall": r["recall"], "f1": r["f1"], "pr_auc": r["pr_auc"],
                "recall_at_fpr_1pct": r["recall_at_fpr_1pct"], "ece": r.get("ece"),
                "latency_ms": r["latency_ms"], "parameters": r["params"],
                "peak_mem_mb": r["peak_mem_mb"],
                "active_edge_ratio": r.get("active_edge_ratio"),
            })
    checks.append("7 个模型 × 5 个逐种子记录与 main_results.json 全字段一致")
    return {"rows": rows, "per_seed_rows": per_seed_rows,
            "records": records_by_model, "checks": checks}


def build_significance(ev: EvidencePackage, main):
    cached = ev.json("main_results_significance.json")
    rows = []
    tasgl = main["records"]["tasgl"]
    for model, label in MODEL_SPECS[1:]:
        baseline = main["records"][model]
        result = exact_wilcoxon_signed_rank(
            [r["f1"] for r in tasgl], [r["f1"] for r in baseline])
        key = f"tasgl_vs_{model}"
        if key not in cached:
            raise ValueError(f"显著性汇总缺少 {key}")
        assert_close(result["statistic"], cached[key]["statistic"], f"{key}.statistic")
        assert_close(result["p_value"], cached[key]["p_value"], f"{key}.p_value")
        if bool(cached[key]["significant"]) != bool(result["p_value"] < 0.05):
            raise ValueError(f"{key}: significant 字段与 p 值不一致")
        rows.append({"comparison": f"TASGL vs. {label}", "baseline_key": model,
                     "seeds": [0, 1, 2, 3, 4], **result,
                     "significant": result["p_value"] < 0.05,
                     "test": "exact_two_sided_wilcoxon_signed_rank"})
    return {"rows": rows,
            "checks": ["6 个精确双侧 Wilcoxon 均由逐种子 F1 重算并与原汇总一致"]}


def build_pareto(ev: EvidencePackage, main):
    cached = ev.json("pareto_frontier_check.json")
    target = next(r for r in main["rows"] if r["model_key"] == "tasgl")
    dominators = {}
    for cost, flag_key in (("parameters", "params"), ("latency_ms_mean", "latency_ms")):
        dominating = []
        for row in main["rows"]:
            if row["model_key"] == "tasgl":
                continue
            no_worse_cost = row[cost] <= target[cost]
            no_worse_f1 = row["f1_mean"] >= target["f1_mean"]
            strictly_better = row[cost] < target[cost] or row["f1_mean"] > target["f1_mean"]
            if no_worse_cost and no_worse_f1 and strictly_better:
                dominating.append(row["model"])
        expected_flag = not dominating
        if bool(cached[flag_key]) != expected_flag:
            raise ValueError(f"pareto_frontier_check.{flag_key} 与向量支配重算不一致")
        dominators[flag_key] = dominating
    return {"tasgl_on_frontier": {"params": not dominators["params"],
                                  "latency_ms": not dominators["latency_ms"]},
            "dominators": dominators,
            "checks": ["参数与延迟两个视图均按成本—精度向量重算 Pareto 支配"]}


def build_ablation(ev: EvidencePackage):
    cached = ev.json("ablation_results.json")
    records = {}
    aggregate = {}
    per_seed_deltas = []
    for variant, label in ABLATION_SPECS:
        file_variant = variant.replace('/', '_')
        variant_records = [ev.json(f"runs/ablation/{file_variant}_seed{seed}.json")
                           for seed in range(3)]
        require_seed_records(variant_records, range(3), 3, f"ablation.{variant}")
        for raw, saved in zip(variant_records, cached[variant]):
            assert_equivalent(raw, saved, f"ablation_results.{variant}.seed{raw['seed']}")
        records[variant] = variant_records
        stats = summarize([r["f1"] for r in variant_records], ddof=1)
        aggregate[variant] = {"variant_key": variant, "variant": label,
                              "protocol": 3, "seeds": [0, 1, 2], **stats}
    full_mean = aggregate["full"]["mean"]
    rows = []
    for variant, _ in ABLATION_SPECS:
        row = dict(aggregate[variant])
        row["delta_f1_vs_full_difference_of_means"] = row["mean"] - full_mean
        rows.append(row)
        if variant != "full":
            for full, changed in zip(records["full"], records[variant]):
                per_seed_deltas.append({"variant_key": variant, "seed": full["seed"],
                                        "full_f1": full["f1"], "variant_f1": changed["f1"],
                                        "paired_delta_f1": changed["f1"] - full["f1"]})
    return {"rows": rows, "per_seed_deltas": per_seed_deltas, "records": records,
            "checks": ["7 个消融状态 × 3 个逐种子记录与原汇总全字段一致"]}


def build_temporal(ev: EvidencePackage):
    cached = ev.json("temporal_results.json")
    if cached.get("rolling_protocol") != 4 or cached.get("fixed_test_start") != 35:
        raise ValueError("时序协议必须是协议 4 且固定 test_start=35")
    rows = []
    records = {}
    for model, label in TEMPORAL_MODELS:
        records[model] = {}
        for cutoff in (20, 25, 30, 34):
            key = f"{model}_trainle{cutoff}"
            raw_records = [ev.json(f"runs/temporal/{key}_seed{seed}.json")
                           for seed in range(3)]
            require_seed_records(raw_records, range(3), 4, f"temporal.{key}")
            if any(r.get("train_end") != cutoff or r.get("test_start") != 35
                   for r in raw_records):
                raise ValueError(f"temporal.{key}: 训练截止或固定测试区间不一致")
            for raw, saved in zip(raw_records, cached["rolling"][key]):
                assert_equivalent(raw, saved, f"temporal_results.{key}.seed{raw['seed']}")
            records[model][cutoff] = raw_records
        reference = statistics.fmean(r["f1"] for r in records[model][34])
        for cutoff in (20, 25, 30, 34):
            stats = summarize([r["f1"] for r in records[model][cutoff]], ddof=1)
            rows.append({"model_key": model, "model": label, "protocol": 4,
                         "train_end": cutoff, "test_start": 35, "test_end": 49,
                         "seeds": [0, 1, 2], **stats,
                         "reference_cutoff": 34, "reference_f1_mean": reference,
                         "relative_loss_vs_cutoff34": (reference - stats["mean"]) / reference,
                         "visibility": "transductive_full_graph_and_full_data_normalization"})
    return {"rows": rows, "records": records,
            "checks": ["3 个模型 × 4 个截止点 × 3 个种子均核验协议、固定测试区间和原汇总"]}


def build_robustness(ev: EvidencePackage):
    cached = ev.json("robustness_results.json")
    records = {model: {} for model, _ in TEMPORAL_MODELS}
    rows = []
    for model, label in TEMPORAL_MODELS:
        clean_values = None
        for condition, level in ROBUST_CONDITIONS:
            key = f"{condition}:{level}"
            raw_records = [ev.json(f"runs/robustness/{model}_seed{seed}_{condition}_{level}.json")
                           for seed in range(2)]
            if len(raw_records) != 2 or any(r.get("protocol") != 3 for r in raw_records):
                raise ValueError(f"robustness.{model}.{key}: 种子数或协议不正确")
            values = [r["f1"] for r in raw_records]
            if len(cached[model][key]) != 2:
                raise ValueError(f"robustness_results.{model}.{key}: 不是两种子")
            for seed, (value, saved) in enumerate(zip(values, cached[model][key])):
                assert_close(value, saved, f"robustness.{model}.{key}.seed{seed}")
            if condition == "clean":
                clean_values = values
            if clean_values is None:
                raise ValueError("clean 必须先于攻击条件计算")
            stats = summarize(values, ddof=0)
            clean_mean = statistics.fmean(clean_values)
            records[model][key] = raw_records
            rows.append({"model_key": model, "model": label, "protocol": 3,
                         "condition": condition, "level": level, "seeds": [0, 1],
                         **stats, "clean_f1_mean": clean_mean,
                         "relative_drop_vs_clean": (clean_mean - stats["mean"]) / clean_mean,
                         "attack_visibility": ("label_guided_oracle" if condition == "edge_inject"
                                                else "historical_E5_diagnostic"),
                         "sd_definition": "population_ddof_0"})
    return {"rows": rows, "records": records,
            "checks": ["E5 的 3 模型 × 11 条件 × 2 种子均由原始记录重算，SD 为 ddof=0"]}


def build_uncertainty(ev: EvidencePackage):
    cached = ev.json("uncertainty_results.json")
    if cached.get("seed") != 0 or cached.get("protocol") != 3:
        raise ValueError("E7 必须是独立 seed 0、协议 3 记录")
    rows = []
    curve_rows = []
    for model, label in (("tasgl", "TASGL"), ("edl", "Same-backbone EDL proxy")):
        record = cached[model]
        rows.append({"model_key": model, "model": label, "protocol": 3,
                     "seed": 0, "evidence_role": "independent_E7_state",
                     **{k: record[k] for k in ("f1", "precision", "recall", "pr_auc",
                                               "recall_at_fpr_1pct", "threshold", "ece")}})
        coverage, risk = cached["curves"][model]
        check_selective_curve(record["f1"], coverage, risk, f"E7.{model}")
        for c, r in zip(coverage, risk):
            curve_rows.append({"model_key": model, "model": label, "seed": 0,
                               "coverage": c, "selective_risk_1_minus_f1": r,
                               "full_coverage": bool(c == 1.0)})
    timeline_rows = []
    for phase in ("early", "late"):
        item = cached["timeline"][phase]
        if len(item["ts"]) != len(item["entropy"]):
            raise ValueError(f"E7.timeline.{phase}: 时间与熵长度不一致")
        for ts, entropy in zip(item["ts"], item["entropy"]):
            timeline_rows.append({"phase": phase, "time_step": ts, "entropy": entropy,
                                  "seed": 0, "novelty_label_assigned": False})
    return {"rows": rows, "curve_rows": curve_rows, "timeline_rows": timeline_rows,
            "raw": cached,
            "checks": ["E7 两条 20 点曲线的全覆盖风险均精确校验为 1-F1"]}


def build_crosschain(ev: EvidencePackage):
    cached = ev.json("crosschain_results.json")
    meta = cached.get("_meta", {})
    if meta.get("protocol") != 4 or meta.get("semantic_alignment") is not False:
        raise ValueError("E6 必须为协议 4 且显式 semantic_alignment=false")
    rows = []
    per_seed = []
    for model, label in (("tasgl", "TASGL"), ("gcn", "GCN")):
        raw_records = [ev.json(f"runs/crosschain/{model}_seed{seed}.json") for seed in range(3)]
        require_seed_records(raw_records, range(3), 4, f"crosschain.{model}")
        for raw, saved in zip(raw_records, cached[model]):
            assert_equivalent(raw, saved, f"crosschain_results.{model}.seed{raw['seed']}")
        target = summarize([r["target_f1"] for r in raw_records], ddof=1)
        source = summarize([r["source_f1"] for r in raw_records], ddof=1)
        pr_auc = summarize([r["target_pr_auc"] for r in raw_records], ddof=1)
        rows.append({"model_key": model, "model": label, "protocol": 4,
                     "seeds": [0, 1, 2], "source_f1_mean": source["mean"],
                     "source_f1_sample_sd": source["sd"], "target_f1_mean": target["mean"],
                     "target_f1_sample_sd": target["sd"], "target_pr_auc_mean": pr_auc["mean"],
                     "target_pr_auc_sample_sd": pr_auc["sd"],
                     "alignment": "independent_pca", "semantic_alignment": False,
                     "evidence_role": "weak_diagnostic_only"})
        for r in raw_records:
            per_seed.append({"model_key": model, "model": label, **r})
    return {"rows": rows, "per_seed_rows": per_seed, "meta": meta,
            "checks": ["E6 两种模型三种子与原汇总一致，且保留 semantic_alignment=false"]}


def _metric_mean(records, section, metric):
    return statistics.fmean(finite_number(r[section][metric], f"{section}.{metric}")
                            for r in records)


def _ratio_of_means_drop(reference_records, short_records):
    reference = _metric_mean(reference_records, "clean", "f1")
    short = _metric_mean(short_records, "clean", "f1")
    if reference == 0:
        raise ValueError("参考 F1 为 0，不能计算相对损失")
    return reference, short, (reference - short) / reference


def build_protocol5(ev: EvidencePackage):
    lock = ev.json("v3/lock.json")
    if lock.get("protocol") != 5:
        raise ValueError("v3/lock.json 不是协议 5")
    lock_id = lock["lock_id"]
    root = f"v3/confirm_{lock_id}"
    raw_cached = ev.json(f"{root}/raw_results.json")
    hypothesis_cached = ev.json(f"{root}/hypothesis_summary.json")
    if hypothesis_cached.get("lock_id") != lock_id or hypothesis_cached.get("protocol") != 5:
        raise ValueError("协议 5 锁标识或协议字段不一致")
    seeds = lock["final"]["seeds"]
    if seeds != [0, 1, 2, 3, 4] or hypothesis_cached.get("seeds") != seeds:
        raise ValueError("当前论文协议 5 生成器要求预声明种子 0–4")
    model_name = lock["config"]["name"]
    reference = [ev.json(f"{root}/runs/{model_name}_seed{s}_train34.json") for s in seeds]
    short = [ev.json(f"{root}/runs/{model_name}_seed{s}_train25.json") for s in seeds]
    require_seed_records(reference, seeds, 5, "protocol5.reference")
    require_seed_records(short, seeds, 5, "protocol5.short")
    for records, key in ((reference, "reference"), (short, "short")):
        if len(raw_cached[key]) != len(seeds):
            raise ValueError(f"protocol5.raw_results.{key}: 种子数不一致")
        for raw, saved in zip(records, raw_cached[key]):
            assert_equivalent(raw, saved, f"protocol5.raw_results.{key}.seed{raw['seed']}")
            schedule = raw["metadata"]["schedule"]
            expected_test = (35, 49)
            if (schedule["test_start"], schedule["test_end"]) != expected_test:
                raise ValueError("protocol5: 最终评估区间必须为 35–49")
            if raw["metadata"].get("lock_id") != lock_id:
                raise ValueError("protocol5: 逐种子记录未绑定同一锁")
    baselines = {}
    for model in ("gcn", "gat"):
        records = [ev.json(f"{root}/runs/{model}_seed{s}_train34.json") for s in seeds]
        require_seed_records(records, seeds, 5, f"protocol5.baseline.{model}")
        for raw, saved in zip(records, raw_cached["baselines"][model]):
            assert_equivalent(raw, saved, f"protocol5.baseline.{model}.seed{raw['seed']}")
        baselines[model] = records

    f1_stats = summarize([r["clean"]["f1"] for r in reference], ddof=1)
    latency_stats = summarize([r["efficiency"]["latency_ms"] for r in reference], ddof=1)
    params = {int(r["params"]) for r in reference}
    if len(params) != 1:
        raise ValueError("protocol5: 学生参数量随种子变化")
    parameters = params.pop()
    targets = lock["final"]["targets"]
    h1_status = "pass" if f1_stats["mean"] >= targets["H1_f1"] and parameters <= targets["H1_params"] else "falsified"
    h2_status = "pass" if latency_stats["mean"] <= targets["H2_latency_ms"] else "falsified"
    reference_f1, short_f1, h4_loss = _ratio_of_means_drop(reference, short)
    short_stats = summarize([r["clean"]["f1"] for r in short], ddof=1)
    assert_close(reference_f1, f1_stats["mean"], "protocol5.H4.reference_f1")
    assert_close(short_f1, short_stats["mean"], "protocol5.H4.short_f1")
    if h4_loss <= targets["H4_loss_pass"]:
        h4_status = "pass"
    elif h4_loss > targets["H4_loss_falsify"]:
        h4_status = "falsified"
    else:
        h4_status = "inconclusive_between_bounds"

    student_drop = (f1_stats["mean"] - _metric_mean(reference, "edge_drop50", "f1")) / f1_stats["mean"]
    baseline_rows = []
    baseline_statuses = []
    for model, label in (("gcn", "GCN"), ("gat", "GAT")):
        records = baselines[model]
        clean = _metric_mean(records, "clean", "f1")
        dropped = _metric_mean(records, "edge_drop50", "f1")
        drop = (clean - dropped) / clean
        if drop <= 0:
            status = "inconclusive_nonpositive_baseline_drop"
            allowed = None
        else:
            allowed = targets["H3_baseline_fraction"] * drop
            if student_drop <= allowed:
                status = "pass"
            elif student_drop >= drop:
                status = "falsified"
            else:
                status = "inconclusive_between_bounds"
        baseline_statuses.append(status)
        baseline_rows.append({"baseline": label, "baseline_key": model,
                              "clean_f1_mean": clean, "edge_drop50_f1_mean": dropped,
                              "baseline_relative_drop": drop,
                              "allowed_student_drop_60pct": allowed,
                              "literal_h3_comparison": status, "seeds": seeds})
    h3_status = "pass" if all(s == "pass" for s in baseline_statuses) else (
        "falsified" if any(s == "falsified" for s in baseline_statuses) else "inconclusive")

    ece_stats = summarize([r["clean"]["ece"] for r in reference], ddof=1)
    uncalibrated_ece = summarize([r["clean_uncalibrated"]["ece"] for r in reference], ddof=1)
    edl_ece = summarize([r["edl"]["calibrated"]["ece"] for r in reference], ddof=1)
    brier = summarize([r["clean"]["brier"] for r in reference], ddof=1)
    nll = summarize([r["clean"]["nll"] for r in reference], ddof=1)
    h5_status = "pass" if (ece_stats["mean"] <= targets["H5_ece"] and
                           ece_stats["mean"] < edl_ece["mean"]) else "falsified"
    hypotheses = {
        "H1": {"f1_mean": f1_stats["mean"], "f1_sample_sd": f1_stats["sd"],
               "params": parameters, "status": h1_status, "evidence_complete": True},
        "H2": {"latency_ms_mean": latency_stats["mean"],
               "latency_ms_sample_sd": latency_stats["sd"], "status": h2_status,
               "evidence_complete": True},
        "H3": {"tasgl_clean_f1": f1_stats["mean"],
               "tasgl_edge_drop50_f1": _metric_mean(reference, "edge_drop50", "f1"),
               "tasgl_relative_drop": student_drop, "status": h3_status,
               "degenerate_zero_hop_pass": bool(lock["config"]["hops"] == 0 and h3_status == "pass"),
               "evidence_complete": True},
        "H4": {"reference_f1": reference_f1, "short_f1": short_f1,
               "short_f1_sample_sd": short_stats["sd"],
               "relative_loss_ratio_of_means": h4_loss, "status": h4_status,
               "evidence_complete": True},
        "H5": {"ece_mean": ece_stats["mean"], "ece_sample_sd": ece_stats["sd"],
               "uncalibrated_ece_mean": uncalibrated_ece["mean"],
               "uncalibrated_ece_sample_sd": uncalibrated_ece["sd"],
               "edl_same_backbone_ece_mean": edl_ece["mean"],
               "edl_same_backbone_ece_sample_sd": edl_ece["sd"],
               "brier_mean": brier["mean"], "brier_sample_sd": brier["sd"],
               "nll_mean": nll["mean"], "nll_sample_sd": nll["sd"],
               "status": h5_status, "evidence_complete": True},
    }
    cached_h = hypothesis_cached["hypotheses"]
    comparisons = [
        ("H1", "f1_mean"), ("H1", "params"), ("H1", "status"),
        ("H2", "latency_ms_mean"), ("H2", "status"),
        ("H3", "tasgl_relative_drop"), ("H3", "status"),
        ("H4", "reference_f1"), ("H4", "short_f1"),
        ("H4", "relative_loss_ratio_of_means"), ("H4", "status"),
        ("H5", "ece_mean"), ("H5", "edl_same_backbone_ece_mean"),
        ("H5", "brier_mean"), ("H5", "nll_mean"), ("H5", "status"),
    ]
    for h, key in comparisons:
        actual, expected = hypotheses[h][key], cached_h[h][key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert_close(actual, expected, f"protocol5.{h}.{key}")
        elif actual != expected:
            raise ValueError(f"protocol5.{h}.{key}: {actual} != {expected}")
    for row in baseline_rows:
        saved = cached_h["H3"]["per_baseline"][row["baseline_key"]]
        for actual_key, saved_key in (("clean_f1_mean", "clean_f1"),
                                      ("edge_drop50_f1_mean", "edge_drop50_f1"),
                                      ("baseline_relative_drop", "relative_drop"),
                                      ("allowed_student_drop_60pct", "target_tasgl_drop")):
            assert_close(row[actual_key], saved[saved_key],
                         f"protocol5.H3.{row['baseline_key']}.{saved_key}")
        if row["literal_h3_comparison"] != saved["status"]:
            raise ValueError(f"protocol5.H3.{row['baseline_key']}: 状态不一致")
    all_pass = all(item["status"] == "pass" for item in hypotheses.values())
    if bool(hypothesis_cached.get("all_pass")) != all_pass:
        raise ValueError("protocol5.all_pass 与五项状态重算不一致")

    rows = [
        {"hypothesis": "H1", "observed": "F1 and deployable parameters",
         "f1_mean": f1_stats["mean"], "f1_sample_sd": f1_stats["sd"],
         "parameters": parameters, "criterion": "F1>=0.73; parameters<=20000",
         "status": h1_status, "interpretation": "size and accuracy are non-compensatory"},
        {"hypothesis": "H2", "observed": "Student inference plus calibration latency",
         "latency_ms_mean": latency_stats["mean"], "latency_ms_sample_sd": latency_stats["sd"],
         "criterion": "mean latency<=8ms", "status": h2_status,
         "interpretation": "offline teacher training excluded from deployed student latency"},
        {"hypothesis": "H3", "observed": "50% edge-drop ratio of means",
         "student_relative_drop": student_drop,
         "criterion": "student<=60% of each positive baseline drop",
         "status": h3_status,
         "interpretation": "degenerate edge invariance; hops=0" if lock["config"]["hops"] == 0 else "graph-using student"},
        {"hypothesis": "H4", "observed": "Short-history loss relative to reference",
         "reference_f1": reference_f1, "short_f1_mean": short_f1,
         "short_f1_sample_sd": short_stats["sd"], "relative_loss": h4_loss,
         "criterion": "<=5% pass; >8% falsified; otherwise inconclusive",
         "status": h4_status, "interpretation": "ratio-of-means loss"},
        {"hypothesis": "H5", "observed": "Calibrated student versus same-backbone EDL proxy",
         "ece_mean": ece_stats["mean"], "ece_sample_sd": ece_stats["sd"],
         "edl_ece_mean": edl_ece["mean"], "edl_ece_sample_sd": edl_ece["sd"],
         "criterion": "ECE<=0.05 and below same-backbone EDL proxy",
         "status": h5_status, "interpretation": "proxy-bounded; not full ST-CGNN"},
    ]
    calibration = [
        {"state": "student_before_calibration", **uncalibrated_ece},
        {"state": "student_after_calibration", **ece_stats},
        {"state": "same_backbone_edl_proxy", **edl_ece},
    ]
    selection = ev.json("v3/select/summary.json")
    if selection.get("protocol") != 5 or selection.get("test_labels_35_49_used") is not False:
        raise ValueError("协议 5 选择阶段必须屏蔽 35–49 标签")
    winner = next((r for r in selection["rows"] if r["name"] == selection.get("winner")), None)
    if winner is None or selection["winner"] != model_name:
        raise ValueError("协议 5 选择 winner 与锁配置不一致")
    return {"rows": rows, "baseline_rows": baseline_rows, "calibration": calibration,
            "hypotheses": hypotheses, "all_pass": all_pass, "lock": lock,
            "selection_winner": winner, "reference_records": reference,
            "short_records": short,
            "checks": ["协议 5 的 20 份学生／基线逐种子记录、锁、H1–H5 和 all_pass 均交叉校验"]}


def build_protocol6(ev: EvidencePackage):
    summary = ev.json("v4/select/summary.json")
    raw_cached = ev.json("v4/select/raw_results.json")
    if summary.get("protocol") != 6 or summary.get("test_labels_35_49_used") is not False:
        raise ValueError("协议 6 必须为 development-only 且屏蔽 35–49 标签")
    seeds = summary["selection_seeds"]
    if seeds != [0, 1]:
        raise ValueError("当前论文协议 6 生成器要求开发种子 0、1")
    targets = summary["development_targets"]
    rows = []
    gate_rows = []
    run_count = 0
    cached_by_name = {r["name"]: r for r in summary["rows"]}
    for name in PROTOCOL6_CANDIDATES:
        reference = [ev.json(f"v4/select/runs/{name}_seed{s}_train29.json") for s in seeds]
        short = [ev.json(f"v4/select/runs/{name}_seed{s}_train24.json") for s in seeds]
        require_seed_records(reference, seeds, 6, f"protocol6.{name}.reference")
        require_seed_records(short, seeds, 6, f"protocol6.{name}.short")
        run_count += len(reference) + len(short)
        for records, key in ((reference, "reference"), (short, "short")):
            for raw, saved in zip(records, raw_cached[name][key]):
                assert_equivalent(raw, saved, f"protocol6.raw.{name}.{key}.seed{raw['seed']}")
                schedule = raw["metadata"]["schedule"]
                if (schedule["test_start"], schedule["test_end"]) != (33, 34):
                    raise ValueError("protocol6: 开发评分区间必须为 33–34")
                if raw["metadata"].get("lock_id") is not None:
                    raise ValueError("protocol6: 开发记录不得伪造锁标识")
        f1 = _metric_mean(reference, "clean", "f1")
        pr_auc = _metric_mean(reference, "clean", "pr_auc")
        ece = _metric_mean(reference, "clean", "ece")
        brier = _metric_mean(reference, "clean", "brier")
        drop_f1 = _metric_mean(reference, "edge_drop50", "f1")
        edge_drop = (f1 - drop_f1) / f1
        latency = statistics.fmean(r["efficiency"]["latency_ms"] for r in reference)
        params_values = {int(r["params"]) for r in reference}
        if len(params_values) != 1:
            raise ValueError(f"protocol6.{name}: 参数量随种子变化")
        params = params_values.pop()
        reference_f1, short_f1, h4_loss = _ratio_of_means_drop(reference, short)
        assert_close(reference_f1, f1, f"protocol6.{name}.reference_f1")
        gates = {
            "H1_f1": f1 >= targets["f1_min"],
            "H1_params": params <= targets["params_max"],
            "H2_latency": latency <= targets["latency_ms_max"],
            "H3_absolute_drop_proxy": edge_drop <= targets["edge_drop50_rel_drop_max"],
            "H4_extrapolation_proxy": h4_loss <= targets["extrapolation_loss_pass"],
            "H5_ece_proxy": ece <= targets["ece_max"],
        }
        h4_status = ("pass" if h4_loss <= targets["extrapolation_loss_pass"] else
                     "falsified" if h4_loss > targets["extrapolation_loss_falsify"] else
                     "inconclusive_between_bounds")
        failed = [key for key, _ in GATE_SPECS if not gates[key]]
        eligible = all(gates.values())
        row = {"name": name, "protocol": 6, "seeds": seeds, "f1": f1,
               "pr_auc": pr_auc, "ece": ece, "brier": brier, "latency_ms": latency,
               "params": params, "edge_drop50_f1": drop_f1,
               "edge_drop50_rel_drop": edge_drop, "short_f1": short_f1,
               "extrapolation_loss": h4_loss, "development_gates": gates,
               "failed_gates": failed, "h4_development_status": h4_status,
               "eligible": eligible, "evaluation_window": "33-34",
               "test_labels_35_49_used": False}
        saved = cached_by_name.get(name)
        if saved is None:
            raise ValueError(f"protocol6.summary 缺少 {name}")
        for key in ("f1", "pr_auc", "ece", "brier", "latency_ms", "params",
                    "edge_drop50_f1", "edge_drop50_rel_drop", "short_f1",
                    "extrapolation_loss"):
            assert_close(row[key], saved[key], f"protocol6.summary.{name}.{key}")
        if gates != saved["development_gates"] or failed != saved["failed_gates"]:
            raise ValueError(f"protocol6.summary.{name}: 门控重算不一致")
        if eligible != bool(saved["eligible"]) or h4_status != saved["h4_development_status"]:
            raise ValueError(f"protocol6.summary.{name}: eligible/H4 状态不一致")
        rows.append(row)
        for gate, label in GATE_SPECS:
            gate_rows.append({"candidate": name, "gate": gate, "gate_label": label,
                              "passed": gates[gate], "eligible": eligible,
                              "evaluation_window": "33-34"})
    eligible_names = [r["name"] for r in rows if r["eligible"]]
    expected_winner = summary.get("winner")
    authorized = bool(summary.get("confirmation_authorized"))
    if not eligible_names and (expected_winner is not None or authorized):
        raise ValueError("protocol6: 无合格候选却存在 winner 或确认授权")
    if expected_winner is None and authorized:
        raise ValueError("protocol6: winner=null 不得授权确认")
    if expected_winner is None and "v4/lock.json" in ev.names:
        raise ValueError("protocol6: winner=null 却存在锁文件")
    return {"rows": rows, "gate_rows": gate_rows, "winner": expected_winner,
            "confirmation_authorized": authorized, "run_count": run_count,
            "selection_policy": summary.get("selection_policy"),
            "targets": targets,
            "checks": ["协议 6 的 28 份开发记录、六门控、winner 和确认授权均重算核验"]}


def build_figure_inputs(main, ablation, temporal, robustness, uncertainty, p5, p6):
    main_rows = []
    for row in main["rows"]:
        main_rows.append({
            "key": row["model_key"], "label": row["model"],
            "f1_mean": row["f1_mean"], "f1_sample_sd": row["f1_sample_sd"],
            "pr_auc_mean": row["pr_auc_mean"], "pr_auc_sample_sd": row["pr_auc_sample_sd"],
            "recall_fpr1_mean": row["recall_at_fpr_1pct_mean"],
            "recall_fpr1_sample_sd": row["recall_at_fpr_1pct_sample_sd"],
            "latency_ms_mean": row["latency_ms_mean"],
            "latency_ms_sample_sd": row["latency_ms_sample_sd"],
            "parameters": row["parameters"], "seeds": row["seeds"],
        })
    ablation_rows = [{"key": r["variant_key"], "label": r["variant"],
                      "delta_f1": r["delta_f1_vs_full_difference_of_means"]}
                     for r in ablation["rows"] if r["variant_key"] != "full"]
    temporal_rows = []
    for model, label in TEMPORAL_MODELS:
        selected = [r for r in temporal["rows"] if r["model_key"] == model]
        temporal_rows.append({"key": model, "label": label,
                              "cutoffs": [r["train_end"] for r in selected],
                              "f1_mean": [r["mean"] for r in selected],
                              "f1_sample_sd": [r["sd"] for r in selected]})
    edge_rows = []
    for model, label in TEMPORAL_MODELS:
        selected = [r for r in robustness["rows"]
                    if r["model_key"] == model and (r["condition"] == "clean" or
                                                    r["condition"] == "edge_drop")]
        selected.sort(key=lambda r: 0.0 if r["condition"] == "clean" else float(r["level"]))
        edge_rows.append({"key": model, "label": label,
                          "drop_percent": [0 if r["condition"] == "clean" else int(100 * float(r["level"]))
                                           for r in selected],
                          "f1_mean": [r["mean"] for r in selected],
                          "f1_population_sd": [r["sd"] for r in selected]})
    gate_matrix = [[1 if r["development_gates"][gate] else 0 for gate, _ in GATE_SPECS]
                   for r in p6["rows"]]
    tdsc = {
        "schema_version": SCHEMA_VERSION,
        "figure_data_role": "deterministic_aggregation_not_new_experiment",
        "main_protocol3": {"models": main_rows, "sd": "sample_ddof_1"},
        "ablation_protocol3": {"variants": ablation_rows,
                               "delta_definition": "variant_mean_minus_full_mean"},
        "temporal_protocol4": {"models": temporal_rows, "fixed_test": "35-49",
                               "sd": "sample_ddof_1"},
        "edge_drop_historical_E5": {"models": edge_rows, "sd": "population_ddof_0",
                                    "seeds": [0, 1]},
        "protocol6_development": {
            "candidates": [r["name"] for r in p6["rows"]],
            "gates": [label for _, label in GATE_SPECS], "status": gate_matrix,
            "status_encoding": {"0": "FAIL", "1": "PASS"},
            "winner": p6["winner"], "confirmation_authorized": p6["confirmation_authorized"],
            "evaluation_window": "33-34", "test_labels_35_49_used": False,
        },
    }
    e7 = uncertainty["raw"]
    tifs = {
        "schema_version": SCHEMA_VERSION,
        "figure_data_role": "deterministic_aggregation_not_new_experiment",
        "main_protocol3": {"models": main_rows, "sd": "sample_ddof_1"},
        "protocol5_calibration": {
            "states": p5["calibration"], "ece_gate": p5["lock"]["final"]["targets"]["H5_ece"],
            "evidence_role": "locked_exploratory", "comparison": "same_backbone_EDL_proxy",
        },
        "independent_E7": {
            "seed": 0, "coverage": e7["curves"]["tasgl"][0],
            "tasgl_risk": e7["curves"]["tasgl"][1], "edl_risk": e7["curves"]["edl"][1],
            "tasgl_full_coverage_f1": e7["tasgl"]["f1"],
            "edl_full_coverage_f1": e7["edl"]["f1"],
            "risk_definition": "1-F1_on_retained_set",
        },
    }
    return tdsc, tifs


def pm(mean, sd, digits=4):
    return f"${mean:.{digits}f}\\pm{sd:.{digits}f}$"


def tex_escape(value: str):
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def split_hash(value: str):
    if len(value) != 64:
        raise ValueError("SHA-256 必须为 64 个十六进制字符")
    return value[:32], value[32:]


def tex_main(rows):
    lines = [r"\begin{table*}[!t]",
             r"\caption{Five-seed V2 results on Elliptic (protocol 3; mean $\pm$ sample SD). PR-AUC is curve area, not average precision.}",
             r"\label{tab:main_results}", r"\centering", r"\scriptsize",
             r"\begin{tabularx}{\textwidth}{l >{\raggedleft\arraybackslash}X >{\raggedleft\arraybackslash}X >{\raggedleft\arraybackslash}X >{\raggedleft\arraybackslash}X >{\raggedleft\arraybackslash}p{0.09\textwidth} >{\raggedleft\arraybackslash}X}",
             r"\toprule",
             r"Model & Illicit F1 & PR-AUC & Recall@FPR1\% & Latency (ms) & Parameters & Peak memory (MB) \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{r['model']} & {pm(r['f1_mean'], r['f1_sample_sd'])} & "
                     f"{pm(r['pr_auc_mean'], r['pr_auc_sample_sd'])} & "
                     f"{pm(r['recall_at_fpr_1pct_mean'], r['recall_at_fpr_1pct_sample_sd'])} & "
                     f"{pm(r['latency_ms_mean'], r['latency_ms_sample_sd'], 3)} & "
                     f"{r['parameters']:,} & {pm(r['peak_mem_mb_mean'], r['peak_mem_mb_sample_sd'], 1)} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def tex_wilcoxon(rows):
    lines = [r"\begin{table}[!t]",
             r"\caption{Exact two-sided paired Wilcoxon tests for five-seed illicit F1.}",
             r"\label{tab:wilcoxon}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Comparison & Statistic & Exact $p$ & $p<0.05$? \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['comparison']} & {r['statistic']:.1f} & {r['p_value']:.4f} & "
                     f"{'Yes' if r['significant'] else 'No'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tex_protocol5(p5):
    h = p5["hypotheses"]
    lock_id = p5["lock"]["lock_id"]
    lines = [r"\begin{table*}[!t]",
             f"\\caption{{Protocol-5 five-seed locked exploratory evaluation (lock \\texttt{{{lock_id}}}). The test interval was previously observed; these are not independent confirmatory decisions.}}",
             r"\label{tab:protocol5}", r"\centering", r"\footnotesize",
             r"\begin{tabularx}{\textwidth}{>{\bfseries}l >{\raggedright\arraybackslash}p{0.31\textwidth} >{\raggedright\arraybackslash}p{0.23\textwidth} >{\raggedright\arraybackslash}X}",
             r"\toprule", r"Hypothesis & Observed result & Criterion & Status and interpretation \\",
             r"\midrule",
             f"H1 & F1 {pm(h['H1']['f1_mean'], h['H1']['f1_sample_sd'])}; {h['H1']['params']:,} parameters & F1 $\\geq0.73$; parameters $\\leq20$K & \\textbf{{{h['H1']['status'].upper()}}}: size passes, accuracy fails. \\\\",
             f"H2 & {pm(h['H2']['latency_ms_mean'], h['H2']['latency_ms_sample_sd'], 3)} ms & Mean latency $\\leq8$ ms & \\textbf{{{h['H2']['status'].upper()}}}: student and calibration included; teacher excluded. \\\\",
             f"H3 & Student relative drop ${100*h['H3']['tasgl_relative_drop']:.2f}\\%$ & Student drop $\\leq60\\%$ of each positive baseline drop & \\textbf{{{h['H3']['status'].upper()}, DEGENERATE}}: $K=0$ causes edge invariance, not graph robustness. \\\\",
             f"H4 & Reference F1 ${h['H4']['reference_f1']:.4f}$; short F1 {pm(h['H4']['short_f1'], h['H4']['short_f1_sample_sd'])}; loss ${100*h['H4']['relative_loss_ratio_of_means']:.2f}\\%$ & $\\leq5\\%$ pass; $>8\\%$ falsified & \\textbf{{{h['H4']['status'].upper()}}}: exceeds the 8\\% boundary. \\\\",
             f"H5 & ECE {pm(h['H5']['ece_mean'], h['H5']['ece_sample_sd'])}; EDL proxy {pm(h['H5']['edl_same_backbone_ece_mean'], h['H5']['edl_same_backbone_ece_sample_sd'])} & ECE $\\leq0.05$ and below same-backbone proxy & \\textbf{{{h['H5']['status'].upper()}}}: proxy-bounded, not full ST-CGNN. \\\\", 
             r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def _failed_gate_text(failed):
    failed = set(failed)
    parts = []
    if "H1_f1" in failed:
        parts.append("H1 accuracy")
    if "H2_latency" in failed:
        parts.append("H2")
    proxies = []
    if "H3_absolute_drop_proxy" in failed:
        proxies.append("H3")
    if "H4_extrapolation_proxy" in failed:
        proxies.append("H4")
    if proxies:
        parts.append("/".join(proxies) + (" proxies" if len(proxies) > 1 else " proxy"))
    if "H1_params" in failed:
        parts.append("H1 size")
    if "H5_ece_proxy" in failed:
        parts.append("H5 ECE")
    return ", ".join(parts) if parts else "None"


def tex_protocol6(rows):
    lines = [r"\begin{table*}[!t]",
             r"\caption{Protocol-6 development-only gate audit (two seeds; evaluation on steps 33--34). No test labels from steps 35--49 are used.}",
             r"\label{tab:protocol6}", r"\centering", r"\scriptsize",
             r"\begin{tabularx}{\textwidth}{l >{\raggedleft\arraybackslash}X >{\raggedleft\arraybackslash}X >{\raggedleft\arraybackslash}X >{\raggedleft\arraybackslash}X >{\raggedleft\arraybackslash}X >{\raggedleft\arraybackslash}X >{\raggedright\arraybackslash}p{0.21\textwidth}}",
             r"\toprule",
             r"Candidate & Dev. F1 & H4 loss proxy & Latency (ms) & Parameters & Edge-drop proxy & ECE & Failed gate(s) \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"\\texttt{{{tex_escape(r['name'])}}} & {r['f1']:.4f} & "
                     f"{100*r['extrapolation_loss']:.2f}\\% & {r['latency_ms']:.3f} & "
                     f"{r['params']:,} & {100*r['edge_drop50_rel_drop']:.2f}\\% & "
                     f"{r['ece']:.4f} & {_failed_gate_text(r['failed_gates'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def tex_operating_points(main):
    lines = [r"\begin{table}[!t]",
             r"\caption{TASGL v2 per-seed operating points (protocol 3). Thresholds come from validation; metrics are evaluated on steps 35--49.}",
             r"\label{tab:tasgl_operating_points}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{cccccc}", r"\toprule",
             r"Seed & Threshold & Precision & Recall & Illicit F1 \\", r"\midrule"]
    for r in main["per_seed_rows"]:
        if r["model_key"] == "tasgl":
            lines.append(f"{r['seed']} & {r['threshold']:.2f} & {r['precision']:.4f} & "
                         f"{r['recall']:.4f} & {r['f1']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tex_e7(rows):
    by_key = {r["model_key"]: r for r in rows}
    lines = [r"\begin{table}[!t]",
             r"\caption{Independent seed-0 E7 uncertainty diagnostic. This trained state is separate from main seed 0 and protocol 5.}",
             r"\label{tab:e7_forensic}", r"\centering", r"\footnotesize",
             r"\begin{tabular}{lcc}", r"\toprule", r"Metric & TASGL & EDL proxy \\", r"\midrule"]
    specs = [("Illicit F1", "f1", 4), ("Precision", "precision", 4),
             ("Recall", "recall", 4), ("PR-AUC", "pr_auc", 4),
             (r"Recall@FPR $=1\%$", "recall_at_fpr_1pct", 4),
             ("Threshold", "threshold", 2), ("ECE", "ece", 4)]
    for label, key, digits in specs:
        lines.append(f"{label} & {by_key['tasgl'][key]:.{digits}f} & "
                     f"{by_key['edl'][key]:.{digits}f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tex_packages(packages, manuscript):
    combined = next(p for p in packages if p["role"] == "combined_v2_protocol5_protocol6")
    ordered = [combined]
    for role in ("standalone_v3", "standalone_v4"):
        item = next((p for p in packages if p["role"] == role), None)
        if item:
            ordered.append(item)
    if manuscript == "tdsc":
        caption = "Read-only evidence packages. Each SHA-256 value is split across two lines solely for layout."
        label = "tab:packages"
        roles = ["Protocol-3 main and ablation; protocol-4 temporal and cross-domain; robustness and uncertainty diagnostics.",
                 "Protocol-5 selection, configuration lock, and five-seed exploratory evaluation.",
                 "Protocol-6 development-only gate audit; no lock or confirmation."]
    else:
        caption = "Read-only evidence packages. Each SHA-256 value is split across two lines solely for layout."
        label = "tab:tifs_packages"
        roles = ["Main and ablation results; temporal, robustness, cross-domain, uncertainty, and selective-risk diagnostics.",
                 "Protocol-5 selection, configuration lock, calibration, and five-seed exploratory evaluation.",
                 "Protocol-6 development-only gate audit; no lock or final confirmation."]
    lines = [r"\begin{table*}[!t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             r"\centering", r"\footnotesize",
             r"\begin{tabularx}{\textwidth}{l >{\raggedright\arraybackslash}p{0.36\textwidth} >{\raggedright\arraybackslash}X}",
             r"\toprule", r"Package & SHA-256 & Numerical role \\", r"\midrule"]
    for i, package in enumerate(ordered):
        first, second = split_hash(package["sha256"])
        name = tex_escape(package["filename"])
        lines.append(f"\\texttt{{{name}}} & \\texttt{{{first}}}\\newline\\texttt{{{second}}} & {roles[i]} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


class OutputWriter:
    def __init__(self, root: Path):
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError(f"输出目录非空，禁止覆盖历史派生证据: {self.root}")
        self.files = {}

    def write_bytes(self, relative: str, payload: bytes):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digest = sha256_bytes(payload)
        self.files[relative] = digest
        return relative, digest

    def write_text(self, relative: str, text: str):
        return self.write_bytes(relative, text.encode("utf-8"))

    def write_json(self, relative: str, value):
        return self.write_bytes(relative, json_bytes(value))

    def table(self, name: str, rows):
        if not rows:
            raise ValueError(f"空表格: {name}")
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        csv_path = f"tables/{name}.csv"
        json_path = f"tables/{name}.json"
        self.write_json(json_path, rows)
        with (self.root / csv_path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                formatted = {}
                for field in fields:
                    value = row.get(field)
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":"))
                    elif isinstance(value, bool):
                        value = "true" if value else "false"
                    formatted[field] = value
                writer.writerow(formatted)
        payload = (self.root / csv_path).read_bytes()
        self.files[csv_path] = sha256_bytes(payload)
        return [csv_path, json_path]


def capture_sources(ev: EvidencePackage, function):
    before = set(ev.sources)
    value = function()
    return value, sorted(set(ev.sources) - before)


def lineage_entry(manuscript, item, evd, domain, protocol, seeds, sources,
                  statistic, outputs, generator=GENERATOR):
    allowed, forbidden = EVD_BOUNDARIES[domain]
    return {"manuscript": manuscript, "item": item, "evd": evd,
            "protocol": protocol, "seeds": seeds, "source_members": sorted(set(sources)),
            "statistical_definition": statistic, "generator": generator,
            "outputs": sorted(set(outputs)), "allowed_conclusion": allowed,
            "forbidden_inference": forbidden}


def build_lineage(main_sources, significance_sources, ablation_sources, temporal_sources,
                  robustness_sources, uncertainty_sources, crosschain_sources,
                  p5_sources, p6_sources, table_outputs, tdsc_tex, tifs_tex):
    entries = []
    def add(*args, **kwargs):
        entries.append(lineage_entry(*args, **kwargs))
    main_out = table_outputs["table_02_main_results"]
    per_seed_out = table_outputs["table_02_per_seed_operating_points"]
    add("TDSC", "Table tab:main_results; Fig. fig:accuracy_cost", "EVD-02;EVD-03",
        "main", 3, [0, 1, 2, 3, 4], main_sources,
        "均值与样本 SD(ddof=1)；参数和延迟 Pareto 按向量支配。",
        main_out + ["tdsc/generated_data/figure_inputs.json", tdsc_tex["main"]],
        GENERATOR + " + TASGL_TDSC投稿/make_figures.py")
    add("TIFS", "Fig. fig:forensic_metrics; Table tab:tasgl_operating_points",
        "EVD-02;EVD-03", "main", 3, [0, 1, 2, 3, 4], main_sources,
        "五种子均值与样本 SD；逐种子阈值、precision、recall、F1 不聚合替代。",
        main_out + per_seed_out + ["tifs/generated_data/figure_inputs.json", tifs_tex["operating"]],
        GENERATOR + " + TASGL_TIFS投稿/make_figures.py")
    add("TDSC/TIFS", "Table 3 / Wilcoxon paragraph", "EVD-04", "main", 3,
        [0, 1, 2, 3, 4], main_sources + significance_sources,
        "逐种子 F1 配对；零差值剔除；精确双侧符号翻转穷举；不事后换检验。",
        table_outputs["table_03_wilcoxon"] + [tdsc_tex["wilcoxon"]])
    add("TDSC/TIFS", "Table 4; Fig. fig:ablation; ablation paragraph", "EVD-06",
        "ablation", 3, [0, 1, 2], ablation_sources,
        "三种子样本 SD；图中 Δ=变体均值−full 均值；另存逐种子配对差。",
        table_outputs["table_04_ablation"] + table_outputs["table_04_paired_deltas"] +
        ["tdsc/generated_data/figure_inputs.json"])
    add("TDSC/TIFS", "Table 5; Fig. fig:temporal_edge(a); temporal paragraph", "EVD-07",
        "temporal", 4, [0, 1, 2], temporal_sources,
        "固定测试 35–49；样本 SD；相对损失=(cutoff34 均值−当前均值)/cutoff34 均值。",
        table_outputs["table_05_temporal"] + ["tdsc/generated_data/figure_inputs.json"])
    add("TDSC/TIFS", "Table 6; Fig. fig:temporal_edge(b); E5 paragraphs", "EVD-08",
        "robust", 3, [0, 1], robustness_sources,
        "两种子总体 SD(ddof=0)；相对下降=(clean 均值−条件均值)/clean 均值。",
        table_outputs["table_06_robustness"] + ["tdsc/generated_data/figure_inputs.json"])
    add("TDSC/TIFS", "Table 7a; Table tab:e7_forensic; Fig. fig:selective_risk; E7 paragraph",
        "EVD-09", "e7", "3 (independent E7 state)", [0], uncertainty_sources,
        "独立 seed 0；选择性风险=保留集合 1−F1；全覆盖点与表中 F1 一致性断言。",
        table_outputs["table_07a_uncertainty_e7"] + table_outputs["table_07a_selective_risk"] +
        table_outputs["table_07a_uncertainty_timeline"] +
        ["tifs/generated_data/figure_inputs.json", tifs_tex["e7"]],
        GENERATOR + " + TASGL_TIFS投稿/make_figures.py")
    add("TDSC/TIFS", "Table 7b; E6 paragraph", "EVD-10", "crosschain", 4,
        [0, 1, 2], crosschain_sources,
        "独立 PCA；目标 F1 三种子均值与样本 SD；semantic_alignment=false。",
        table_outputs["table_07b_crosschain"] + table_outputs["table_07b_crosschain_per_seed"])
    add("TDSC/TIFS", "Table 8; Table tab:protocol5; Fig. fig:calibration; H1--H5 paragraphs",
        "EVD-11;EVD-12;EVD-13", "protocol5", 5, [0, 1, 2, 3, 4], p5_sources,
        "五种子样本 SD；H4 为 ratio-of-means；H3 逐正基线比较；H5 为同骨干 EDL proxy。",
        table_outputs["table_08_protocol5_hypotheses"] +
        table_outputs["table_08_protocol5_h3_baselines"] +
        table_outputs["table_08_protocol5_calibration"] +
        ["tifs/generated_data/figure_inputs.json", tdsc_tex["protocol5"]],
        GENERATOR + " + TASGL_TIFS投稿/make_figures.py")
    add("TDSC/TIFS", "Table 9; Table tab:protocol6; Fig. fig:protocol6_gates; stopping paragraph",
        "EVD-14;EVD-15", "protocol6", 6, [0, 1], p6_sources,
        "33–34 开发均值；六个不可补偿门槛；无候选时 winner=null 且不授权。",
        table_outputs["table_09_protocol6_gates"] + table_outputs["table_09_protocol6_gate_matrix"] +
        ["tdsc/generated_data/figure_inputs.json", tdsc_tex["protocol6"]],
        GENERATOR + " + TASGL_TDSC投稿/make_figures.py")
    add("TDSC", "Fig. fig:audit_pipeline", "EVD-01;EVD-11;EVD-14;EVD-15",
        "package", "conceptual protocol diagram", [], [],
        "无性能数值；仅绘制开发—锁—授权逻辑。", ["tdsc/generated_data/figure_inputs.json"],
        "TASGL_TDSC投稿/make_figures.py")
    for figure in ("fig_temporal_evidence_map", "fig_forensic_chain", "fig_attack_visibility"):
        add("TIFS", f"Fig. {figure}", "EVD-01;EVD-08;EVD-16;EVD-18", "package",
            "conceptual protocol diagram", [], [], "无性能数值；不生成案例、攻击结果或外部标注。",
            ["tifs/generated_data/figure_inputs.json"], "TASGL_TIFS投稿/make_figures.py")
    add("TDSC/TIFS", "Appendix Table A1; package tables", "EVD-16;EVD-18", "package",
        "v2/protocol5/protocol6 separated", "see per-entry seed sets", [],
        "ZIP 与成员 SHA-256；输出哈希见 manifest.json；代码哈希不替代科学核验。",
        ["appendix_a1_artifacts.csv", "appendix_a1_artifacts.json",
         tdsc_tex["packages"], tifs_tex["packages"], "manifest.json", "manifest.sha256"])
    return entries


def lineage_markdown(entries, packages):
    lines = ["# 论文表图—原始证据追溯清单", "",
             "本清单由 `paper_evidence.py` 从只读 ZIP 自动生成。完整成员哈希、输出哈希和逐表允许／禁止推断见 `manifest.json`。", "",
             "## 证据包", "", "| 文件 | SHA-256 | 角色 |", "|---|---|---|"]
    for p in packages:
        lines.append(f"| `{p['filename']}` | `{p['sha256']}` | `{p['role']}` |")
    lines += ["", "## 表图映射", "",
              "| 稿件 | 表／图 | EVD | 协议 | 种子 | 统计口径 | 生成程序 | 派生输出 |",
              "|---|---|---|---|---|---|---|---|"]
    for e in entries:
        outputs = "<br>".join(f"`{x}`" for x in e["outputs"] if not x.startswith("manifest"))
        lines.append("| {manuscript} | {item} | {evd} | {protocol} | {seeds} | {stat} | `{gen}` | {out} |".format(
            manuscript=e["manuscript"], item=e["item"], evd=e["evd"], protocol=e["protocol"],
            seeds=json.dumps(e["seeds"], ensure_ascii=False), stat=e["statistical_definition"],
            gen=e["generator"], out=outputs))
    lines += ["", "## 使用边界", "",
              "- v2、独立 E7、协议 5、协议 6 和 P0／P1 不合并为同一性能表。",
              "- 本流程不执行训练，不重选种子、阈值、测试集或统计检验。",
              "- 原始 JSON 的哈希和重算一致性不等于独立确认、外部专家标注或真实硬件测量已完成。", ""]
    return "\n".join(lines)


def readme_text(packages):
    combined = packages[0]
    optional = []
    for p in packages[1:]:
        option = "--v3_zip" if p["role"] == "standalone_v3" else "--v4_zip"
        optional.append(f' {option} "D:\\证据\\{p["filename"]}"')
    optional_text = "".join(optional)
    return f"""# TASGL 论文证据确定性生成说明

本目录由 `paper_evidence.py` 生成，内容是**统计聚合、LaTeX 表体、绘图输入与血缘清单**，不是新实验，也不改变 H1–H5 的历史结论。

## 权威输入

- 主证据包：`{combined['filename']}`
- SHA-256：`{combined['sha256']}`
- ZIP 仅以只读方式访问；不上传、不解压执行、不反序列化 `data.pkl`。

## 复现（Reproduction）

在包含 `paper_evidence.py` 和 `log_runner.py` 的脚本目录执行；把证据包放到本机任意路径后，用实际路径替换下面的占位符：

```sh
python3 log_runner.py --program paper_evidence --python python3 -- paper_evidence.py --evidence_zip "<path-to>/{combined['filename']}"{optional_text} --expect_sha256 {combined['sha256']} --output_dir "./paper_evidence_rebuilt"
```

Windows 上把 `python3` 换成 PATH 中可用的解释器（例如 `python` 或 `py -3` 对应的可执行文件）。
输出目录必须不存在或为空，防止覆盖历史派生证据。

## 输出用途

- `tables/`：Table 2–9、逐种子操作点、E7 曲线及附录哈希的 CSV／JSON。
- `tdsc/generated_tables/`、`tifs/generated_tables/`：由同一数据生成的 LaTeX 表体。
- `tdsc/generated_data/figure_inputs.json`、`tifs/generated_data/figure_inputs.json`：两份稿件 `make_figures.py` 的唯一数值输入。
- `manifest.json`：稿件表图—EVD—协议—种子—统计口径—源成员哈希—生成程序—输出哈希映射。
- `validation_report.json`：逐种子与原汇总的一致性、协议隔离和曲线一致性检查。
- `manifest.sha256`：清单自身的 SHA-256。

将对应 `generated_tables/` 与 `generated_data/` 复制到稿件根目录后，运行该稿件的 `make_figures.py` 即可重绘图件；概念流程图不含实验数值。重复复现时请使用新的空目录并比较全部文件哈希。

## 科学边界

本程序不会生成真实 P0／P1 长实验、RTX 2060／CPU 测量、第二数据集结果、独立确认、非 oracle 攻击指标或外部专家标注。v2、E7、协议 5 与协议 6 始终分开，协议 6 的 development-only 结果不得写成最终测试性能。
"""


def generate(args):
    output_dir = Path(args.output_dir)
    writer = OutputWriter(output_dir)  # 先检查目录，验证失败时不创建输出。
    ev = EvidencePackage(Path(args.evidence_zip), args.expect_sha256)
    try:
        packages = [ev.package_record()]
        if args.v3_zip:
            packages.append(validate_nested_package(ev, Path(args.v3_zip), "v3/"))
        if args.v4_zip:
            packages.append(validate_nested_package(ev, Path(args.v4_zip), "v4/"))

        main, main_sources = capture_sources(ev, lambda: build_main(ev))
        significance, significance_sources = capture_sources(ev, lambda: build_significance(ev, main))
        pareto, pareto_sources = capture_sources(ev, lambda: build_pareto(ev, main))
        ablation, ablation_sources = capture_sources(ev, lambda: build_ablation(ev))
        temporal, temporal_sources = capture_sources(ev, lambda: build_temporal(ev))
        robustness, robustness_sources = capture_sources(ev, lambda: build_robustness(ev))
        uncertainty, uncertainty_sources = capture_sources(ev, lambda: build_uncertainty(ev))
        crosschain, crosschain_sources = capture_sources(ev, lambda: build_crosschain(ev))
        p5, p5_sources = capture_sources(ev, lambda: build_protocol5(ev))
        p6, p6_sources = capture_sources(ev, lambda: build_protocol6(ev))
        tdsc_figures, tifs_figures = build_figure_inputs(
            main, ablation, temporal, robustness, uncertainty, p5, p6)

        tables = {
            "table_02_main_results": main["rows"],
            "table_02_per_seed_operating_points": main["per_seed_rows"],
            "table_03_wilcoxon": significance["rows"],
            "table_04_ablation": ablation["rows"],
            "table_04_paired_deltas": ablation["per_seed_deltas"],
            "table_05_temporal": temporal["rows"],
            "table_06_robustness": robustness["rows"],
            "table_07a_uncertainty_e7": uncertainty["rows"],
            "table_07a_selective_risk": uncertainty["curve_rows"],
            "table_07a_uncertainty_timeline": uncertainty["timeline_rows"],
            "table_07b_crosschain": crosschain["rows"],
            "table_07b_crosschain_per_seed": crosschain["per_seed_rows"],
            "table_08_protocol5_hypotheses": p5["rows"],
            "table_08_protocol5_h3_baselines": p5["baseline_rows"],
            "table_08_protocol5_calibration": p5["calibration"],
            "table_09_protocol6_gates": p6["rows"],
            "table_09_protocol6_gate_matrix": p6["gate_rows"],
        }
        table_outputs = {name: writer.table(name, rows) for name, rows in tables.items()}
        writer.write_json("tdsc/generated_data/figure_inputs.json", tdsc_figures)
        writer.write_json("tifs/generated_data/figure_inputs.json", tifs_figures)
        writer.write_json("derived_metrics.json", {
            "schema_version": SCHEMA_VERSION,
            "main": main["rows"], "pareto": pareto,
            "protocol5": {"hypotheses": p5["hypotheses"], "all_pass": p5["all_pass"],
                          "lock_id": p5["lock"]["lock_id"], "selection_winner": p5["selection_winner"]},
            "protocol6": {"winner": p6["winner"],
                          "confirmation_authorized": p6["confirmation_authorized"],
                          "run_count": p6["run_count"], "rows": p6["rows"]},
        })
        tdsc_tex = {
            "main": "tdsc/generated_tables/main_results.tex",
            "wilcoxon": "tdsc/generated_tables/wilcoxon.tex",
            "protocol5": "tdsc/generated_tables/protocol5.tex",
            "protocol6": "tdsc/generated_tables/protocol6.tex",
            "packages": "tdsc/generated_tables/packages.tex",
        }
        tifs_tex = {
            "operating": "tifs/generated_tables/tasgl_operating_points.tex",
            "e7": "tifs/generated_tables/e7_forensic.tex",
            "packages": "tifs/generated_tables/packages.tex",
        }
        writer.write_text(tdsc_tex["main"], tex_main(main["rows"]))
        writer.write_text(tdsc_tex["wilcoxon"], tex_wilcoxon(significance["rows"]))
        writer.write_text(tdsc_tex["protocol5"], tex_protocol5(p5))
        writer.write_text(tdsc_tex["protocol6"], tex_protocol6(p6["rows"]))
        writer.write_text(tdsc_tex["packages"], tex_packages(packages, "tdsc"))
        writer.write_text(tifs_tex["operating"], tex_operating_points(main))
        writer.write_text(tifs_tex["e7"], tex_e7(uncertainty["rows"]))
        writer.write_text(tifs_tex["packages"], tex_packages(packages, "tifs"))

        artifact_rows = []
        for package in packages:
            artifact_rows.append({"artifact_type": "evidence_zip", "artifact": package["filename"],
                                  "sha256": package["sha256"], "role": package["role"],
                                  "member_count": package["member_count"]})
        key_members = ["main_results.json", "main_results_significance.json",
                       "ablation_results.json", "temporal_results.json", "robustness_results.json",
                       "uncertainty_results.json", "crosschain_results.json", "v3/lock.json",
                       "v3/confirm_91f1beb52cbad975/hypothesis_summary.json", "v4/select/summary.json"]
        for member in key_members:
            if member not in ev.sources:
                ev.read_bytes(member)
            artifact_rows.append({"artifact_type": "source_member",
                                  "artifact": f"{packages[0]['filename']}!/{member}",
                                  "sha256": ev.sources[member], "role": "paper_source_json",
                                  "member_count": 1})
        generator_hash = file_sha256(Path(__file__).resolve())
        artifact_rows.append({"artifact_type": "generator", "artifact": GENERATOR,
                              "sha256": generator_hash, "role": "deterministic_aggregation",
                              "member_count": 1})
        table_outputs["appendix_a1_artifacts"] = writer.table("appendix_a1_artifacts", artifact_rows)

        entries = build_lineage(main_sources, significance_sources, ablation_sources,
                                temporal_sources, robustness_sources, uncertainty_sources,
                                crosschain_sources, p5_sources, p6_sources,
                                table_outputs, tdsc_tex, tifs_tex)
        checks = []
        for group in (main, significance, pareto, ablation, temporal, robustness,
                      uncertainty, crosschain, p5, p6):
            checks.extend(group["checks"])
        validation = {
            "schema_version": SCHEMA_VERSION, "valid": True,
            "new_model_training_performed": False, "new_experiment_result_created": False,
            "source_zip_modified": False, "checks": checks,
            "counts": {"main_model_seed_records": 35, "ablation_records": 21,
                       "temporal_records": 36, "robustness_records": 66,
                       "protocol5_student_and_baseline_records": 20,
                       "protocol6_development_records": p6["run_count"]},
            "protocol5_all_pass": p5["all_pass"], "protocol6_winner": p6["winner"],
            "protocol6_confirmation_authorized": p6["confirmation_authorized"],
            "protocol_separation": "v2, independent E7, protocol 5 and protocol 6 are not pooled",
            "p1_real_performance_included": False,
        }
        writer.write_json("validation_report.json", validation)
        writer.write_text("lineage.md", lineage_markdown(entries, packages))
        writer.write_text("README.md", readme_text(packages))

        # 在写清单前，将每项源成员与派生输出的 SHA-256 内联到血缘条目。
        for entry in entries:
            entry["source_member_sha256"] = {
                member: ev.sources[member] for member in entry["source_members"]}
            entry["output_sha256"] = {
                path: writer.files[path] for path in entry["outputs"] if path in writer.files}
            entry["source_package"] = packages[0]["filename"]
            entry["source_package_sha256"] = packages[0]["sha256"]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generator": {"program": GENERATOR, "sha256": generator_hash,
                          "mode": "read_only_deterministic_aggregation"},
            "deterministic": True, "timestamp_included": False,
            "evidence_packages": packages,
            "source_members": dict(sorted(ev.sources.items())),
            "lineage": entries, "outputs": dict(sorted(writer.files.items())),
            "validation_report": "validation_report.json",
            "manifest_self_hash": "manifest.sha256",
            "notes": ["manifest.json 不包含自身哈希；其哈希见 manifest.sha256。",
                      "最终 PDF 的哈希由稿件编译流程另行记录，不由统计聚合器伪造。"],
        }
        manifest_payload = json_bytes(manifest)
        writer.write_bytes("manifest.json", manifest_payload)
        writer.write_text("manifest.sha256", sha256_bytes(manifest_payload) + "\n")
        return {"output_dir": str(output_dir), "files": len(writer.files),
                "tables": len(tables) + 1, "lineage_entries": len(entries),
                "evidence_sha256": ev.sha256, "protocol5_all_pass": p5["all_pass"],
                "protocol6_winner": p6["winner"],
                "confirmation_authorized": p6["confirmation_authorized"]}
    finally:
        ev.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_zip", required=True,
                        help="包含 v2、v3、v4 的只读合并证据 ZIP")
    parser.add_argument("--v3_zip", help="可选独立协议 5 ZIP；用于逐成员等价校验")
    parser.add_argument("--v4_zip", help="可选独立协议 6 ZIP；用于逐成员等价校验")
    parser.add_argument("--expect_sha256", help="主证据 ZIP 的预期 SHA-256；不匹配则拒绝")
    parser.add_argument("--output_dir", required=True, help="不存在或为空的派生输出目录")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = generate(args)
    except Exception as exc:
        print(f"[paper_evidence][ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
