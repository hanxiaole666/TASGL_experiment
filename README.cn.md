# TASGL 实验套件 — 公开源码与数据发布版

> **状态：** 已与 2026-11-04 ***Journal 终稿对齐。
> **本文件为中文版**，逐节对应的英文版为 [`README.md`](README.md)；两份文件的
> 章节编号、数值、命令、限制与口径完全一致。
> **范围：** 本仓库发布的是**实验源码与实测结果数据**，不含论文 LaTeX 源码树、
> 不含第三方原始数据集、不含任何模型权重。

---

## 1. 本仓库是什么

TASGL（Temporal-Aware Sparse Graph Learning，时序感知稀疏图学习）在本项目中
被当作一个**联合可靠性问题**来研究，而不是单指标竞赛。仓库中的代码实现了
论文使用的完整审计链：

- 等预算超参搜索与 Elliptic 十种子主比较；
- 在开发期从未使用过的 BitHeNet 上做二十种子独立确认；
- 已登记的门限集合、带正/负对照的非补偿式 assurance gate；
- 逐交易延迟筛查、八阶段教师–学生成本链、门控收敛诊断、校准与根因审计；
- 预先登记的非 oracle 攻击矩阵（H6）；
- 可复现性、确定性、第三数据集与协议/门限版本映射审计。

论文的核心结论是**负向的**：现有证据既不能确立 TASGL 的性能领先，也不能
确立其劣势。本仓库提供的产物，是让读者能够核验这一判断，而不是只能选择
相信。

---

## 2. 版本锚点

P3 时代的旧 README 写的是「BitHeNet 十种子运行中」「八阶段实测能耗」以及
0.73 / 8 ms 的门限值。**这些口径均已撤回**，不得用来描述本次发布。

论文 LaTeX 源码与两份终稿 PDF **不在**本次公开代码发布范围内（见第 6 节），
它们属于独立的 期刊 投稿包。因此 `paper_tdsc.py check --paper_dir <paper>`
需要该源码树；第 10 节给出的 `build` 路径在本仓库内即可独立运行。

---

## 3. 核心结论

以下数值均逐字取自终稿。

**Elliptic 十种子主比较（等调优预算）。** TASGL 非法类 F1
`0.5081 ± 0.1088`，MLP `0.4970 ± 0.0734`，GraphSAGE `0.4394 ± 0.0820`。
两个配对 95% 置信区间都包含 0，且两个观测效应都低于 power 0.8 下的最小
可检测效应，因此早期「被支配」结论已撤回，**且不以「领先」结论替代**。

**成本与规模是单边的。** TASGL 为 `30,171` 参数、整图单次前向 `45.48 ms`；
MLP 为 `6,434` 参数、`2.17 ms`。因此 TASGL 在**规模而非精度**上未通过已登记的
`20,480` 参数 H1 预算。

**BitHeNet 二十种子独立确认。** TASGL `0.5593 ± 0.1101`，GraphSAGE
`0.5079 ± 0.1252`，MLP `0.4112 ± 0.0394`。TASGL − MLP = `+0.1481`，
95% CI `[0.0887, 0.2074]`，20 个种子中 17 个占优，Holm 校正后精确
Wilcoxon `p = 5.23 × 10⁻⁴`，`dz = 1.168`。与 GraphSAGE 的差异**不显著**——
这是关于统计效力的陈述，不是等效性结论。门控失效在 **18/20** 个种子上重现。
BitHeNet 上适用的是**借用的 Elliptic 规则**，不是该数据集上的门限确认；
规模预算仅以 69 个参数之差通过。

**H6 非 oracle 攻击矩阵。** 720 次运行，10 个评估种子 × 3 个攻击种子、3 个模型，
未使用 oracle 标签，协议在执行前登记。半边删除时 TASGL 损失
`45.5% [32.7%, 59.4%]` 非法类 F1，GraphSAGE `14.7%`，MLP `0.0%`；
在特征噪声下排序反转，因此 TASGL **并非在所有维度上都最脆弱**。表中任何一行
都不带通过/不通过判定。

**正向对照。** `7/7` 个对照行为符合预期：满足 H1/H2/H5 全部子条件的候选被接纳
（`all_pass = true`），每违反一个子条件即被拒绝。因此负向结果来自有区分力的
门控，而非橡皮图章。

**可复现性提醒。** 在该技术栈上，同种子重复**并非逐位一致**（预测概率最大差
`1.788 × 10⁻⁷`，在 `1e-6` 意义下等价，F1 差 `0.0`）。逐种子数值不能在跨运行
的逐位意义上复用；见 `results/p5/determinism.json`。

---

## 4. 依赖与安装

```bash
git clone <your-repo-url>
cd TASGL_experiment
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` 只给出下界：`torch>=2.0.0`、`numpy>=1.24`、
`pandas>=2.0`、`scikit-learn>=1.3`、`matplotlib>=3.7`、`scipy>=1.10`、
`gdown>=5.0`。`torch-geometric` 为可选依赖，未安装时会回退到内置的原生
GCN/GAT/SAGE 实现。`gdown` 仅用于 BitHeteroNet 下载的第四备用通道，前三个
HTTP 直链不依赖它。

实测环境：Windows 10 / Python 3.11.5 / NumPy 1.26.4 / PyTorch 2.2.1 /
CUDA 11.8 / cuDNN 8700 / 单卡 NVIDIA GeForce RTX 2060（6143.6 MB、30 SM、
驱动 522.06）。完整环境指纹见 `results/p4/environment.json`，各结果文件内
也各自记录。**计时数字与这台机器绑定。**

Shell 入口支持 `TASGL_PYTHON`（或旧变量 `PY`）覆盖解释器：

```bash
TASGL_PYTHON=/path/to/python bash run_p4_final_wave1.sh
```

随包发布的 `.py` / `.sh` 中不含任何指向特定用户主目录的路径。Windows `.bat`
启动器是仅用于本机的历史包装脚本，**不随包发布**；请使用可移植的 `.sh` 版本。

---

## 5. 仓库结构

```
TASGL_experiment/
├── README.md                  # 本文件对应的英文版
├── README.cn.md               # 中文版，章节编号与英文版一致
├── OPENSOURCE_MANIFEST.md     # 逐文件哈希 + 收录/排除理由
├── requirements.txt
├── pytest.ini                 # 把测试收集限制在根目录源码树
├── .gitignore                 # 排除原始数据、历史稿件、权重、缓存
│
├── common.py, models.py, losses.py, metrics.py, data_loader.py, ...
├── download_data.py           # 第三方数据集获取（无需账号）
├── p0_*.py … p6_*.py          # P0 → P6 各阶段实验
├── paper_tdsc.py              # 生成表、图输入、溯源清单
├── paper_evidence.py          # 证据包读取与重建
├── ledger_reconcile.py        # 投稿台账对账
├── test_*.py                  # 单元 + 端到端测试（249 通过）
├── run_*.sh                   # 可移植复现入口
│
├── results/                   # 实测 JSON / 小 CSV / 日志（发布子集）
│   ├── p2/ p3/ p4/ p5/ p6/   # 按实验波次分阶段结果
│   ├── runs/ v3/ v4/          # 历史运行，仅作对照
│   └── p4/thresholds/         # V1/V3 门限登记表与锁文件
├── Test-log/                  # 44 份带时间戳的运行日志（0.44 MB）
├── figs/                      # 早期阶段的 5 张诊断 PDF
├── reference_checks/          # 3 份小型文献核对 CSV
└── archive/
    └── superseded/            # 2 份已撤回的含能耗结果变体
```

**权威源码树是仓库根目录**（69 个 `.py`，包含完整的 P5/P6 代码）。`Doc/`
工作区里的是用于 LaTeX 投稿的 47 文件旧快照，属于历史副本，不是应当引用
或继续扩展的代码。

---

## 6. 收录与排除清单

可上传快照为 [`TASGL_github_release/`](TASGL_github_release/)，逐文件说明见
`OPENSOURCE_MANIFEST.md`。规则如下：

| 收录 | 排除 | 理由 |
| --- | --- | --- |
| 根目录全部 `*.py`、`test_*.py`、可移植 `*.sh` | `*.bat` | `.bat` 内含本机专用解释器路径 |
| 根配置 `*.json`、`requirements.txt`、`pytest.ini`、`.gitignore`、两份 README、清单 | 根目录 `*.txt` 运行日志、草稿 `*.md` / `*.pdf` / `*.docx` | 历史性、内部性或已撤回的行文 |
| `results/**/*.json`、小型 `*.csv`、`*.log` | `*.pt`、`*.model`、`*.ckpt`、`*.npz` | 约 43 MB 权重/检查点；**没有任何已报告数字**依赖它们 |
| `Test-log/`、`figs/`、`reference_checks/` | `results/p3/independent_dgraph/**/case_audit_queue.csv` | 10 份队列 CSV 约 84 MB，为可再生的中间案件清单 |
| `archive/superseded/`（2 个 JSON + README） | `Data/`、`data/` | **2.8 GB 第三方原始数据**，不可再分发；用 `download_data.py` 获取 |
| — | `Doc/` | 280 MB / 2779 文件，含草稿、期刊 模板、旧源码快照与内部工作记录 |
| — | `TASGL论文证据追溯/` | 早期稿件阶段的内部证据追溯树 |
| — | `__pycache__/`、`.pytest_cache/`、`.idea/`、`.venv/`、`.workbuddy/` | 缓存、IDE 状态、私有工作记录 |
| — | 论文 LaTeX 源码树、`main.pdf`、`supplemental_material.pdf` | 属独立 期刊 投稿包，且涉及 期刊 版权决定（第 16 节） |

**GitHub 硬限制：** 单文件不得超过 100 MB。多个原始数据文件超过 400 MB
（`elliptic_txs_features.csv` 689 MB、`dgraphfin.npz` 680 MB、若干 BitHeNet
`.npy` 各约 362 MB），这是排除 `Data/` 的决定性原因。随包文件全部小于
1 MB，最大的三个是
`results/p4/predictions_elliptic_seed0_mc.json`（755,186 B）、
`results/p6/nonoracle_attack_matrix.json`（753,677 B）、
`results/p4/predictions_elliptic_seed0.json`（730,620 B）。

`pytest.ini` 设置了 `norecursedirs = Doc Data results archive figs
reference_checks TASGL_github_release …`。若不设置，`pytest` 会递归进入
`Doc/` 下的同名历史模块副本（以及嵌套的发布快照），产生重复模块名收集错误。

**请以 `TASGL_github_release/` 目录内的内容作为仓库根目录发布。** 该目录是
工作区的构建产物，`.gitignore` 已将其排除，避免被当作嵌套副本一起推送。
上传前用 `sha256sum -c RELEASE_FILELIST.sha256` 校验整包。

---

## 7. 快速开始：不下载任何数据也能验证发布包

```bash
python3 -m pytest -q
```

期望计数取决于所在目录：

| 所在目录 | 期望结果 |
| --- | --- |
| 公开发布快照（`TASGL_github_release/`，或克隆后的本仓库） | `244 passed, 6 skipped, 2 warnings` |
| 完整作者工作区（存在 `data/` 与 `Doc/`） | `249 passed, 1 skipped, 2 warnings` |

完整工作区里唯一的 skip 是 `test_paper_evidence.py` 的真实证据包集成用例，
需要设置 `TASGL_EVIDENCE_ZIP`。公开发布快照另有五个**预期内**的 skip，原因是
被排除的产物不存在：`test_p4.py` 一个用例需要 `./data/elliptic`，
`test_p5_tex_conformance.py` 三个用例需要投稿 LaTeX 源码树，
`test_paper_tdsc.py` 一个用例需要投稿目录。**没有任何用例失败。**
`pytest.ini` 已把 `TASGL_github_release/` 排除出测试收集，避免嵌套快照造成
重复模块名导入错误。

然后把全部派生表重生成到临时目录：

```bash
rm -rf /tmp/tasgl_rebuild
python3 paper_tdsc.py build --results_dir . --paper_dir /tmp/tasgl_rebuild
```

期望产出：`generated_tables/` 下 13 张表、
`generated_data/figure_inputs.json`、`provenance/manifest.json`。若数据源
宣称「已测能耗」，`build` 会拒绝执行（第 12 节）。

最后直接从 JSON 核对两个标题级成本数字：

```bash
python3 - <<'PY'
import json
c = json.load(open('results/p4/cost_measurement.json'))['chain']
b = json.load(open('results/p5/bithe20_analysis.json'))
print(c['total_seconds'], c['one_off_seconds'], c['total_energy_j'])
print(b['comparisons']['mlp']['mean_diff'],
      b['comparisons']['mlp']['wilcoxon_p_holm'])
PY
# 25.9254 11.5117 None
# 0.14805480525764353 0.000522613525390625
```

---

## 8. 获取数据集

**本仓库不再分发任何数据集。** 五个数据集均为公开第三方基准；
`download_data.py` 免账号获取，并支持 HTTP Range 断点续传：

```bash
python3 download_data.py --dataset elliptic      --data_root ./data
python3 download_data.py --dataset bitheteronet  --data_root ./data
python3 download_data.py --dataset dgraphfin    --data_root ./data
python3 download_data.py --dataset ethereum     --data_root ./data
python3 download_data.py --dataset dexposure    --data_root ./data
# 或：python3 download_data.py --dataset all --data_root ./data
python3 download_data.py --validate_only --dataset bitheteronet \
    --data_root ./data --validation_report Test-log/bitheteronet-validation.json
```

可用的数据集键：`elliptic`、`ethereum`、`dgraphfin`、`dexposure`、
`bitheteronet`。网络下载的序列化数据**只做静态检查**，代码不会对网络产物
执行不安全的 `pickle.load`。同一作者官方数据的多个访问路径并不是相互独立的
学术镜像，不能保证在每种网络环境下都能下载成功。

`DGraph-Fin` 仅做了探测，**不承载任何门限结论**：`3,700,550` 节点、
`17` 维特征、`4,300,999` 条边，只有边时间戳、没有逐交易时间步，所有时间差
均为 0，时序分量退化；见 `results/p5/third_dataset_table.json`。它无法检验 H4。

---

## 9. 复现实验（P0 → P6）

可移植的 shell 入口（各自 `cd` 回自身目录，并支持 `TASGL_PYTHON`）：

```bash
bash run_p0.sh            # P0 基线实验
bash run_p1.sh            # P1 攻击 / 血缘原语
bash run_p2.sh            # P2 可审计原语 + 等预算 HPO
bash run_p3.sh            # P3 DGraph-Fin 独立证据
bash run_p4.sh            # P4 测量、门限、筛查、成本
bash run_all.sh           # 全链路（数据下载 → 各阶段）
bash run_hypotheses.sh    # H1–H5 开发硬门槛 → 锁定 → 确认
bash run_p4_final_wave1.sh  # HPO + Elliptic 十种子 + BitHeNet
bash run_p4_final_wave2.sh  # H2/H3/H4/H5 区间、筛查、成本
bash run_p5_wave1.sh        # P5 二十种子分析、协议映射、确定性
bash run_p5_wave2.sh        # P5 第三数据集 / 可复现性审计
```

P6 没有独立 shell 包装，直接调用模块即可，例如 `python3 p6_gate_eval.py`
与 `python3 p6_attacks.py`（见 `--help`）。每个阶段都会把 JSON 写入
`results/`，把带时间戳的日志写入 `Test-log/`；脚本支持断点续跑并跳过已完成
阶段。

**不要期待逐种子数值的逐位复现。** 原技术栈上同种子重复的预测概率最大差
`1.788 × 10⁻⁷`，而 F1 完全相同（`results/p5/determinism.json`）。重跑应能
复现已报告的聚合量、区间与判定，而不是相同的浮点字节。
`results/p5/repro_audit.json` 记录了跨执行对比与日志不一致项。

---

## 10. 重生成表格与图件

```bash
python3 paper_tdsc.py build --results_dir . --paper_dir /tmp/tasgl_rebuild
python3 paper_tdsc.py check --paper_dir <期刊-submission-tree>   # 需要 LaTeX 源码
python3 paper_tdsc.py ledger --results_dir .
```

`paper_tdsc.py` 是 13 张派生表、图输入 JSON 与溯源清单的唯一生成器。两道
发布护栏保证它与终稿一致：

1. `assert_energy_not_measured(src)` 拒绝任何含非 `null` `energy_j` /
   `total_energy_j` 的成本源，也拒绝任何含非 `null` `joule_per_tx` 或
   `power_measured = true` 的 H2 筛查源。
2. `tex_h2()` 不再输出 `J/tx` 列，caption 与脚注与已提交的
   `h2_screening.tex` 一致；`tex_cost()` 在 Energy 列写 `not measured` /
   `n/a`，并给出负向能量脚注。

`tex_cost()` 中一次性/经常性分类与 Table S-XXII 一致：一次性 = 图构建 +
教师训练 + 蒸馏 + 门限选择 + 校准（`11.51 s`）；经常性 = 预处理 + 推理 +
更新（`14.41 s`）。

`check` 核验正文每个数字是否出现在生成产物中。它是**数字存在性**检查，
并不校验表格与源 JSON 是否逐字节相等；后一性质由清单哈希与第 11 节的显式
映射覆盖。

**存在两份生成器快照。** 本仓库根目录发布的 `paper_tdsc.py` 是完整源码树
（也驱动 P5/P6）；把它指向投稿 LaTeX 源码树时扫描 `550` 个数字、
`0` 个不可追溯。投稿工作区保留了自己的生成器快照，对终稿正文扫描结果为
`564 numbers, 0 not traceable`。两者在「不存在不可追溯数字」这一结论上一致；
`564` 是稿件权威口径。差异来自同一模块两份历史副本的扫描覆盖度不同，
不是任何已报告数值上的分歧。

---

## 11. 终稿数字 → 数据文件映射

| 终稿数字 / 结论 | 源文件 |
| --- | --- |
| `0.5081 ± 0.1088`、`0.4970 ± 0.0734`、`0.4394 ± 0.0820`、`30,171`、`45.48 ms` | `results/p4/final_comparison.json` |
| `0.5593 ± 0.1101`、`0.5079 ± 0.1252`、`0.4112 ± 0.0394`、20 个原始种子 | `results/p4/second_dataset_20seeds.json` |
| `+0.1481`、CI `[0.0887, 0.2074]`、`p = 5.23 × 10⁻⁴`、`dz = 1.168`、`17/20` | `results/p5/bithe20_analysis.json` |
| H2 `7.362 / 11.258 ms`、`142.857 ms` 预算、批大小曲线 | `results/p4/per_node_screening.json` |
| `25.9254 s`、`11.5117 s`、`44.4%`、8 阶段、能量 `null` | `results/p4/cost_measurement.json` |
| 已登记 V3 门限（`0.4320`、`20,480`、`142.857 ms`、`0.05`） | `results/p4/thresholds/register.json`、`results/p4/threshold_justification.json` |
| 已撤回 V0 值（`0.73`、`20000`、`8.0`、`0.6`、`0.05`）及版本映射 | `results/p5/protocol_threshold_map.json` |
| H3 `12.34%` / `3.37%` / `0.00%`，H4 `32.97%` / `6.94%` 区间 | `results/p4/h3_interval.json`、`results/p4/h4_interval.json` |
| 门控活跃边比例 `1.000`、早停种子、`18/20` 重现 | `results/p4/gate_audit_10seeds.json` |
| 正类 ECE `0.3818`、整体 ECE `0.1222`、seed-0 `0.3049` 根因 | `results/p4/rootcause_seed0.json`、`results/p4/calibration_predictions_elliptic_seed0.json` |
| H6 矩阵、`720` 次运行、`45.5% [32.7%, 59.4%]`、`77,512 / 234,355` 边对 | `results/p6/nonoracle_attack_matrix.json` |
| 正向对照 `7/7`、`all_pass` 判定 | `results/p6/positive_control.json`、`results/p6/gate_evaluation.json` |
| 已撤回结论台账 | `results/p4/claims.json` |
| DGraph-Fin 探测（不承载门限结论） | `results/p5/third_dataset_table.json` |
| 非确定性 `1.788 × 10⁻⁷` | `results/p5/determinism.json` |
| 跨执行 / 日志不一致审计 | `results/p5/repro_audit.json` |
| 环境指纹 | `results/p4/environment.json` |

关键哈希（SHA-256）：

```
c7291bffb0b08040b8d42e43aefbabdc163f95b7348153de9970a57fa7d11199  results/p4/cost_measurement.json
0986b8ed53345a6dbcf0459856abc94b9a2b3879b47cfa986895327ed6885eac  results/p4/per_node_screening.json
b015532799bffda041a3e80926267491fa3f93445ec56eeb6338813b0f24d30f  results/p5/bithe20_analysis.json
d111dd70fbe8e96e3c7276f86489342389ea44ba915d6dbea42b99f26822460c  results/p6/nonoracle_attack_matrix.json
```

---

## 12. 能量口径与已弃用成本变体

**能量未实测。** 开发用 GPU 不暴露功率上限
（`environment.power_limit_w = "[N/A]"`），因此论文**不报告任何焦耳、功耗
采样或能效数字**，成本链只报告墙钟秒数。该陈述出现在主文 VI-E、Table XVIII、
S-II、S-III、S-V、Table S-XXII 以及 Cover Letter 中。

有两份历史结果文件曾含有由功率推导的焦耳值。二者均已**逐字节**归档到
`archive/superseded/` 仅供审计，均不得用于重生成表格或图件：

| 归档文件（SHA-256） | 已撤回内容 | 终稿取值 |
| --- | --- | --- |
| `cost_measurement.2026-11-03.superseded-measured-energy.json`（`bd803489…`） | `35.7963 s`、一次性 `16.5181 s`、`total_energy_j = 2214.953`、`energy_all_measured = true` | `25.9254 s`、`11.5117 s`、`total_energy_j = null`、`energy_all_measured = false` |
| `per_node_screening.2026-11-03.superseded-measured-energy.json`（`acc585ae…`） | 三个 `joule_per_tx` 值、`power_measured = true` | 全部 `joule_per_tx = null`、`power_measured = false` |

已发布的 `results/p4/cost_measurement.json` 是权威 B 变体。已发布的
`results/p4/per_node_screening.json` 与归档运行**保留完全相同的延迟、预算、
裕度、批大小曲线与判定值**，只把撤回的能耗字段置空，并用
`energy_measurement` 记录指回归档条目。`p4_screening.py --measure_power`
现默认 `0`（仅可选启用）。完整理由见 `archive/superseded/README.md`。

> **不要用归档变体运行 `build`。** `paper_tdsc.py` 中的护栏会按设计直接报错。

---

## 13. H1–H6 与门限集合版本映射

| 编号 | 要求 | 终判 |
| --- | --- | --- |
| H1 | 非法类 F1 ≥ `0.4320` **且** 参数量 ≤ `20,480` | 在 Elliptic 上因规模（`30,171`）**被证伪**，非精度问题 |
| H2 | 逐交易筛查延迟 ≤ `142.857 ms`（每秒 7 笔） | **通过**，`7.362 ms`（p95 `11.258 ms`）；TASGL 唯一通过的门 |
| H3 | 删除 50% 边后的相对 F1 降幅 | **仅报区间、不设门**：TASGL `12.34%` vs GraphSAGE `3.37%`、MLP `0.00%` |
| H4 | 缩短训练计划后的相对 F1 损失 | **仅报区间、不设门**：TASGL `32.97%` vs MLP `6.94%` |
| H5 | 正类 ECE ≤ `0.05` 且低于同骨干基线 | 在实测种子上**被证伪**：`0.3818`（整体 ECE `0.1222` 是错误的汇总口径） |
| H6 | 预先登记的非 oracle 矩阵下的干净/被攻击 F1 | **实测但不设门**；缺少告警容量锚点，无法推导容差 |

V0 中五个预登记门限有四个无来源，已**撤回**：F1 `0.73`、参数 `20000`、
延迟 `8.0 ms`（整图语义）、H3 比值 `0.6`、攻击容差 `0.05`（历史上挂在 H4
名下）。V3 是 Table III 的已登记集合，锁 `e10ba4b3df3a9392a943365a`。
撤回 V0 下的 protocol-5 通过，绝不能与已登记 V3 下的证伪并列阅读；
`results/p5/protocol_threshold_map.json` 为每个协议标注了实际判定所用规则。
H6 位于 H1–H5 门集之外，不进入 `all_pass`。

---

## 14. 验证状态与命令

2026-11-04 开源一致性治理中，在根目录权威源码树上实测：

```
python3 -m pytest -q（完整工作区）           → 249 passed, 1 skipped, 2 warnings
python3 -m pytest -q（发布快照）             → 244 passed, 6 skipped, 2 warnings
python3 paper_tdsc.py build --results_dir . → 13 张表 + 图输入 + 清单
paper_tdsc.py check（投稿工作区）           → scanned 564 numbers, 0 not traceable
bash -n run_*.sh                            → 全部可移植入口语法 OK
sha256sum -c RELEASE_FILELIST.sha256        → 1245 / 1245 OK, 0 FAILED
随包最大单文件                              → 755,186 字节（GitHub 上限 100 MB）
```

两条 warning 均无害：PyTorch `index_reduce_` beta API 提示，以及 `unittest`
关于测试方法返回值的弃用提示。

任何改动后建议再跑以下一致性检查：

```bash
# 生成表中不得出现已撤回的能耗主张
grep -RniE "energy is measured|board-level energy|power.draw|J/tx" generated_tables/
# 随包代码中不得出现本机专用路径
grep -RniE "C:\\\\Users\\\\apple|/d/TASGL_experiment|/c/Users" --include='*.py' --include='*.sh' .
```

第一条只应命中「Energy was not measured」这类**负向**陈述；第二条应无输出。

---

## 15. 已知限制与未结项

以下各项在论文中明写，而非隐藏，**且均未被当作结果报告**：

- **四项限制**（S-III / 结论）：能真正检验时序分量的第三数据集；更宽的调优
  网格；非 oracle 矩阵之外的白盒攻击者与可重训防御模型的攻击者；对假阳性
  与假阴性的外部专家复核。
- **五项台账未结项**（S-II，单独跟踪）：审计案件的外部专家复核、正式的跨链
  语义对齐、超出同骨干代理的忠实不确定性控制、仓库与数据血缘的公开发布、
  PR-AUC 之外的指标完整性。
- **能量测量仍未完成**，并明确列为测量限制，而非待报告的结果。
- 单机计时；无板级能耗；无白盒攻击者；无可重训防御模型的攻击者；跨数据集
  特征空间未对齐；Elliptic 主实验 10 种子（精确双侧 Wilcoxon 下限
  `0.00195`），BitHeNet 确认 20 种子（下限 `1.91 × 10⁻⁶`）。
- `Test-log/` 的正式日志停在 2026-11-02 的波次。P4/P5/P6 的结果 JSON 晚于
  最后一批正式日志；补录正式日志台账是独立的整理工作，不改变任何已报告
  数字。
- 投稿工作区 `paper_constants.json` 仍记 `0.2232 µs/node` 的整图摊薄值，
  而正文使用 `0.2112 µs/node`。两者均为描述性数值、不参与门限判定；此差异
  **尚未解决**，此处如实标注，不作静默改动。

---

## 16. 引用、作者声明与上传前检查

若使用本代码或数据，请引用对应的 ***Journal 稿件及本仓库（DOI / 仓库 URL
待定后补充）。稿件中的作者声明为：全文由韩小乐独立撰写；未接受外部资助；
无利益冲突声明；研究仅分析公开基准交易记录，不引入私有用户数据。

**上传前仍有三项需作者本人决定：**

1. **LICENSE** —— 当前未包含任何许可证。选择许可证属于法律决定，请明确
   选定（例如代码 MIT / Apache-2.0、数据 CC-BY-4.0）并添加文件。在此之前，
   本仓库**未被授权复用**。
2. **论文 PDF / LaTeX 源码** —— 当前已排除。发布预印本或源码树之前，请确认
   适用于您投稿的 期刊 政策。
3. **生成式 AI 使用披露** —— 稿件 Author Statements 中无 AI 使用披露。若写作
   过程中存在工具实质性辅助，请在公开前核对 期刊政策，並使披露
   与独著声明保持一致。

`OPENSOURCE_MANIFEST.md` 列出每个随包文件的 SHA-256 及收录/排除理由；
若文件集合发生变化，请重新生成该清单。
