# TASGL Experiment Suite — Public Code and Data Release

> **Status:** aligned with the 2026-11-04 camera-ready ***Journal submission.
> **This README is the English version.** The section-by-section Chinese
> counterpart is [`README.cn.md`](README.cn.md); both carry the same section
> numbers, numbers, commands and caveats.
> **Scope:** this repository publishes the *experiment code and the measured
> result data*, not the manuscript LaTeX tree, not the third-party raw
> datasets, and not any model weights.

---

## 1. What this repository is

TASGL (Temporal-Aware Sparse Graph Learning) is studied here as a **joint
dependability problem**, not as a single-metric contest. The code in this
repository implements the full audit chain used by the manuscript:

- equal-budget hyper-parameter search and a ten-seed Elliptic comparison;
- a twenty-seed independent confirmation on BitHeNet, a dataset never used
  during development;
- registered threshold sets, a non-compensatory assurance gate with positive
  and negative controls;
- per-transaction latency screening, an eight-stage teacher–student cost
  chain, gate-convergence diagnostics, calibration and root-cause audits;
- a pre-registered non-oracle adversarial attack matrix (H6);
- reproducibility, determinism, third-dataset and protocol/version-map audits.

The paper's central result is **negative**: the current evidence does not
establish performance leadership for TASGL, and it does not establish
inferiority either. This repository ships the artifacts that let a reader
verify that statement rather than take it on trust.

**Author:** Xiaole Han, Business School, Qilu Institute of Technology, Jinan,
Shandong 250200, China (`hanxiaoleop@163.com`), ORCID [0009-0009-9611-2368](https://orcid.org/0009-0009-9611-2368).

---

## 2. Version anchor
Earlier P3-era READMEs described a ten-seed BitHeNet run, an "eight-stage
measured energy" chain, and threshold values of 0.73 / 8 ms. **All of those
are withdrawn.** Do not use them to describe this release.

The LaTeX sources and the two final PDFs are *not* included in this public
code release (see Section 6). They live in the separate IEEE submission
package. Consequently `paper_tdsc.py check --paper_dir <paper>` needs that
tree; the `build` path described in Section 10 is self-contained here.

---

## 3. Headline results

All values below are reproduced verbatim from the camera-ready text.

**Ten-seed Elliptic main comparison (equal tuning budget).**
TASGL `0.5081 ± 0.1088` illicit F1, MLP `0.4970 ± 0.0734`, GraphSAGE
`0.4394 ± 0.0820`. Both paired 95% confidence intervals contain zero and
both observed effects are below the minimum detectable effect at power 0.8,
so the earlier "dominance" claim was withdrawn and **no leadership claim
replaces it**.

**Cost and size are one-sided.** TASGL uses `30,171` parameters and
`45.48 ms` per full-graph forward pass; MLP uses `6,434` parameters and
`2.17 ms`. TASGL therefore **fails the registered 20,480-parameter H1 size
budget on size alone**, not on accuracy.

**Twenty-seed independent BitHeNet confirmation.** TASGL
`0.5593 ± 0.1101`, GraphSAGE `0.5079 ± 0.1252`, MLP `0.4112 ± 0.0394`.
TASGL − MLP = `+0.1481`, 95% CI `[0.0887, 0.2074]`, 17 of 20 seeds in
favour, Holm-corrected exact Wilcoxon `p = 5.23 × 10⁻⁴`, `dz = 1.168`.
The difference against GraphSAGE is **not significant** — a statement about
statistical power, not about equivalence. The gate failure repeats on
**18 of 20** seeds. The rule applied on BitHeNet is a **borrowed Elliptic
rule**, not a gate confirmation on that dataset; the size budget is cleared
by only 69 parameters.

**H6 non-oracle attack matrix.** 720 runs, 10 evaluation seeds × 3 attack
seeds, 3 models, no oracle labels used, protocol registered before
execution. TASGL loses `45.5% [32.7%, 59.4%]` of illicit F1 at half-edge
removal against `14.7%` for GraphSAGE and `0.0%` for MLP; under feature
noise the ordering reverses, so TASGL is *not* uniformly most fragile. No
row carries a pass/fail mark.

**Positive control.** `7/7` controls behave as expected: a candidate
satisfying every H1/H2/H5 sub-condition is admitted (`all_pass = true`)
and each single-condition violation is rejected. The negative outcomes are
therefore produced by a discriminating gate, not a rubber stamp.

**Reproducibility caveat.** Same-seed repeats are **not bitwise identical**
on this stack (max probability difference `1.788 × 10⁻⁷`, equivalent at
`1e-6`, F1 difference `0.0`). Per-seed numbers must not be reused in a
cross-execution bitwise sense; see `results/p5/determinism.json`.

---

## 4. Requirements and installation

```bash
git clone <your-repo-url>
cd TASGL_experiment
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

`requirements.txt` pins lower bounds only: `torch>=2.0.0`, `numpy>=1.24`,
`pandas>=2.0`, `scikit-learn>=1.3`, `matplotlib>=3.7`, `scipy>=1.10`,
`gdown>=5.0`. `torch-geometric` is optional — the code falls back to the
built-in native GCN/GAT/SAGE implementations. `gdown` is only needed for the
fourth fallback channel of the BitHeteroNet download; the first three HTTP
direct links do not need it.

The measurements were produced on Windows 10 / Python 3.11.5 / NumPy 1.26.4
/ PyTorch 2.2.1 / CUDA 11.8 / cuDNN 8700 / one NVIDIA GeForce RTX 2060
(6143.6 MB, 30 SMs, driver 522.06). The full environment fingerprint is
recorded in `results/p4/environment.json` and inside each result file.
**Timing numbers are specific to that machine.**

Shell entry points honour `TASGL_PYTHON` (or the legacy `PY` variable) as
an interpreter override:

```bash
TASGL_PYTHON=/path/to/python bash run_p4_final_wave1.sh
```

No path in the shipped `.py` or `.sh` tree points at a specific user's home
directory. Windows `.bat` launchers are local-only historical wrappers and
are **not** shipped; use the portable `.sh` equivalents.

---

## 5. Repository layout

```
TASGL_experiment/
├── README.md                  # this file (English)
├── README.cn.md               # Chinese counterpart, same section numbers
├── OPENSOURCE_MANIFEST.md     # per-file hashes + inclusion/exclusion reasons
├── requirements.txt
├── pytest.ini                 # limits collection to the root source tree
├── .gitignore                 # excludes raw data, drafts, weights, caches
│
├── common.py, models.py, losses.py, metrics.py, data_loader.py, ...
├── download_data.py           # third-party dataset acquisition (no credentials)
├── p0_*.py … p6_*.py          # experiment stages P0 → P6
├── paper_tdsc.py              # generated tables / figure inputs / provenance
├── paper_evidence.py          # evidence-package reader and reconstructor
├── ledger_reconcile.py        # submission-ledger reconciliation
├── test_*.py                  # unit + end-to-end tests (249 passing)
├── run_*.sh                   # portable reproduction entry points
│
├── results/                   # measured JSON / small CSV / logs (published subset)
│   ├── p2/ p3/ p4/ p5/ p6/   # staged results, by experiment wave
│   ├── runs/ v3/ v4/          # historical runs kept for contrast
│   └── p4/thresholds/         # V1/V3 threshold registers and lock file
├── Test-log/                  # 44 timestamped run logs (0.44 MB)
├── figs/                      # 5 diagnostic PDFs from the early stages
├── reference_checks/          # 3 small literature-check CSVs
└── archive/
    └── superseded/            # 2 withdrawn energy-bearing result variants
```

The **authoritative source tree is the repository root** (69 `.py` files,
including the complete P5/P6 code). The `Doc/` working area contains an
older 47-file snapshot used for the LaTeX submission; it is a historical
copy, not the code to cite or extend.

---

## 6. What is published, and what is excluded

The uploadable snapshot is [`TASGL_github_release/`](TASGL_github_release/)
and is described file-by-file in `OPENSOURCE_MANIFEST.md`. Rules:

| Include | Exclude | Reason |
| --- | --- | --- |
| All root `*.py`, `test_*.py`, portable `*.sh` | `*.bat` | `.bat` wrappers embed a machine-specific interpreter path |
| Root config `*.json`, `requirements.txt`, `pytest.ini`, `.gitignore`, both READMEs, manifest | Root `*.txt` run logs, draft `*.md`, draft `*.pdf`, `*.docx` | historical, internal, or superseded prose |
| `results/**/*.json`, small `*.csv`, `*.log` | `*.pt`, `*.model`, `*.ckpt`, `*.npz` | ~43 MB of weights/checkpoints; **no reported number** depends on them |
| `Test-log/`, `figs/`, `reference_checks/` | `results/p3/independent_dgraph/**/case_audit_queue.csv` | 10 queues, ~84 MB, regenerable intermediate case lists |
| `archive/superseded/` (2 JSON + README) | `Data/`, `data/` | **2.8 GB of third-party raw data**, not redistributable; obtain via `download_data.py` |
| — | `Doc/` | 280 MB / 2779 files of drafts, IEEE templates, prior source snapshots and internal working notes |
| — | `TASGL论文证据追溯/` | internal evidence-traceability tree from earlier manuscript drafts |
| — | `__pycache__/`, `.pytest_cache/`, `.idea/`, `.venv/`, `.workbuddy/` | caches, IDE state, private working records |
| — | manuscript LaTeX tree, `main.pdf`, `supplemental_material.pdf` | separate IEEE submission package; also an IEEE copyright decision (Section 16) |

**GitHub hard limit:** no single file may exceed 100 MB. Several raw data
files exceed 400 MB (`elliptic_txs_features.csv` 689 MB, `dgraphfin.npz`
680 MB, several BitHeNet `.npy` ~362 MB each), which is the decisive reason
`Data/` is excluded. Every shipped file is under 1 MB; the three largest are
`results/p4/predictions_elliptic_seed0_mc.json` (755,186 B),
`results/p6/nonoracle_attack_matrix.json` (753,677 B) and
`results/p4/predictions_elliptic_seed0.json` (730,620 B).

`pytest.ini` sets `norecursedirs = Doc Data results archive figs
reference_checks TASGL_github_release …`. Without it, `pytest` recurses into
historical snapshots of the same modules under `Doc/` (and into the nested
release snapshot) and produces duplicate-module collection errors.

**Publish the *contents* of `TASGL_github_release/` as the repository root.**
That directory is a build artifact of the working tree, and `.gitignore`
excludes it so it is never pushed as a nested duplicate. Verify the whole
package with `sha256sum -c RELEASE_FILELIST.sha256` before uploading.

---

## 7. Quick start: verify the release without any dataset

```bash
python3 -m pytest -q
```

Expected counts depend on which tree you are in:

| Tree | Expected |
| --- | --- |
| Public release snapshot (`TASGL_github_release/`, or this repo after cloning) | `244 passed, 6 skipped, 2 warnings` |
| Full author workspace (with `data/` and `Doc/` present) | `249 passed, 1 skipped, 2 warnings` |

The single skip in the full workspace is `test_paper_evidence.py`'s real
evidence-package integration test, which needs `TASGL_EVIDENCE_ZIP`. The
public snapshot has five additional, **expected** skips because the excluded
artifacts are absent: one `test_p4.py` case needs `./data/elliptic`, three
`test_p5_tex_conformance.py` cases need the submission LaTeX tree, and one
`test_paper_tdsc.py` case needs the submission directory. No test fails.
`pytest.ini` excludes `TASGL_github_release/` from collection so that the
nested snapshot cannot produce duplicate-module import errors.

Then regenerate every derived table into a scratch directory:

```bash
rm -rf /tmp/tasgl_rebuild
python3 paper_tdsc.py build --results_dir . --paper_dir /tmp/tasgl_rebuild
```

Expected: 13 tables under `generated_tables/`, `generated_data/figure_inputs.json`,
and `provenance/manifest.json`. The build refuses to run from a source that
claims measured energy (Section 12).

Finally check the two headline cost numbers straight from the JSON:

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

## 8. Obtaining the datasets

**No dataset is redistributed here.** All five are public third-party
benchmarks; `download_data.py` fetches them without an account and supports
HTTP Range resume:

```bash
python3 download_data.py --dataset elliptic      --data_root ./data
python3 download_data.py --dataset bitheteronet  --data_root ./data
python3 download_data.py --dataset dgraphfin    --data_root ./data
python3 download_data.py --dataset ethereum     --data_root ./data
python3 download_data.py --dataset dexposure    --data_root ./data
# or: python3 download_data.py --dataset all --data_root ./data
python3 download_data.py --validate_only --dataset bitheteronet \
    --data_root ./data --validation_report Test-log/bitheteronet-validation.json
```

Available dataset keys: `elliptic`, `ethereum`, `dgraphfin`, `dexposure`,
`bitheteronet`. Downloaded serialized data is subjected to **static checks
only** — the code never calls unsafe `pickle.load` on a network artifact.
Multiple access paths to the same author's official data are not independent
academic mirrors and cannot guarantee download success on every network.

`DGraph-Fin` was probed but **carries no gate result**: `3,700,550` nodes,
`17` features, `4,300,999` edges, edge timestamps only, no per-transaction
time step. All time gaps are therefore zero and the temporal component is
degenerate; see `results/p5/third_dataset_table.json`. It cannot exercise
H4.

---

## 9. Reproducing the experiments (P0 → P6)

Portable shell entry points (they `cd` to their own directory and honour
`TASGL_PYTHON`):

```bash
bash run_p0.sh            # P0 baseline experiments
bash run_p1.sh            # P1 attack / lineage primitives
bash run_p2.sh            # P2 auditable primitives + equal-budget HPO
bash run_p3.sh            # P3 DGraph-Fin independent evidence
bash run_p4.sh            # P4 measurement, thresholds, screening, cost
bash run_all.sh           # full chain (dataset download → all stages)
bash run_hypotheses.sh    # H1–H5 development gate → lock → confirmation
bash run_p4_final_wave1.sh  # HPO + Elliptic 10 seeds + BitHeNet
bash run_p4_final_wave2.sh  # H2/H3/H4/H5 intervals, screening, cost
bash run_p5_wave1.sh        # P5 20-seed analysis, protocol map, determinism
bash run_p5_wave2.sh        # P5 third-dataset / reproducibility audits
```

P6 has no dedicated shell wrapper; invoke the modules directly, e.g.
`python3 p6_gate_eval.py` and `python3 p6_attacks.py` (see `--help`).
Every stage writes JSON under `results/` and a timestamped log under
`Test-log/`; scripts are resumable and skip completed stages.

**Do not expect bitwise reproduction of per-seed numbers.** Same-seed
repeats on the original stack differ by up to `1.788 × 10⁻⁷` in predicted
probability while F1 is identical (`results/p5/determinism.json`). Re-running
should reproduce the reported aggregates, intervals and verdicts, not
identical floating-point bytes. `results/p5/repro_audit.json` records the
cross-execution comparison and its log disagreements.

---

## 10. Regenerating tables and figures

```bash
python3 paper_tdsc.py build --results_dir . --paper_dir /tmp/tasgl_rebuild
python3 paper_tdsc.py check --paper_dir <IEEE-submission-tree>   # needs LaTeX sources
python3 paper_tdsc.py ledger --results_dir .
```

`paper_tdsc.py` is the single generator for the 13 derived tables, the
figure-input JSON and the provenance manifest. Two release guards keep it
consistent with the camera-ready text:

1. `assert_energy_not_measured(src)` rejects any cost source with a non-null
   `energy_j` / `total_energy_j`, and any H2 screening source with a non-null
   `joule_per_tx` or `power_measured = true`.
2. `tex_h2()` no longer emits a `J/tx` column; its caption and footer match
   the submitted `h2_screening.tex`. `tex_cost()` writes `not measured` /
   `n/a` in the Energy column and a negative energy footer.

The one-off/recurring classification in `tex_cost()` matches Table S-XXII:
one-off = graph construction + teacher training + distillation + threshold
selection + calibration (`11.51 s`); recurring = preprocessing + inference +
update (`14.41 s`).

`check` verifies that every number in the manuscript body appears in the
generated outputs. It is a **number-presence** check, not a byte-equality
check between a table and its source JSON; the manifest hashes and the
explicit mapping in Section 11 cover that second property.

**Two generator snapshots exist.** The root `paper_tdsc.py` published here is
the complete source tree (it also drives P5/P6); pointed at the submission
LaTeX tree it scans `550` numbers with `0` unknown. The submission workspace
keeps its own generator snapshot, which scans the final body as
`564 numbers, 0 not traceable`. Both agree on the *absence* of untraceable
numbers; the 564 figure is the manuscript-authoritative one. The difference
is a scanner-coverage difference between two historical copies of the same
module, not a disagreement about any reported value.

---

## 11. Number → source-file traceability

| Manuscript number / claim | Source file |
| --- | --- |
| `0.5081 ± 0.1088`, `0.4970 ± 0.0734`, `0.4394 ± 0.0820`, `30,171`, `45.48 ms` | `results/p4/final_comparison.json` |
| `0.5593 ± 0.1101`, `0.5079 ± 0.1252`, `0.4112 ± 0.0394`, 20 raw seeds | `results/p4/second_dataset_20seeds.json` |
| `+0.1481`, CI `[0.0887, 0.2074]`, `p = 5.23 × 10⁻⁴`, `dz = 1.168`, `17/20` | `results/p5/bithe20_analysis.json` |
| H2 `7.362 / 11.258 ms`, `142.857 ms` budget, batch curve | `results/p4/per_node_screening.json` |
| `25.9254 s`, `11.5117 s`, `44.4%`, 8 stages, energy `null` | `results/p4/cost_measurement.json` |
| Registered V3 thresholds (`0.4320`, `20,480`, `142.857 ms`, `0.05`) | `results/p4/thresholds/register.json`, `results/p4/threshold_justification.json` |
| Withdrawn V0 values (`0.73`, `20000`, `8.0`, `0.6`, `0.05`) and version map | `results/p5/protocol_threshold_map.json` |
| H3 `12.34%` / `3.37%` / `0.00%`, H4 `32.97%` / `6.94%` intervals | `results/p4/h3_interval.json`, `results/p4/h4_interval.json` |
| Gate active-edge ratio `1.000`, early-stopped seeds, `18/20` repeats | `results/p4/gate_audit_10seeds.json` |
| Positive-class ECE `0.3818`, overall ECE `0.1222`, seed-0 `0.3049` root cause | `results/p4/rootcause_seed0.json`, `results/p4/calibration_predictions_elliptic_seed0.json` |
| H6 matrix, `720` runs, `45.5% [32.7%, 59.4%]`, `77,512 / 234,355` edge pairs | `results/p6/nonoracle_attack_matrix.json` |
| Positive control `7/7`, `all_pass` verdicts | `results/p6/positive_control.json`, `results/p6/gate_evaluation.json` |
| Withdrawn claims ledger | `results/p4/claims.json` |
| DGraph-Fin probe (no gate result) | `results/p5/third_dataset_table.json` |
| Non-determinism `1.788 × 10⁻⁷` | `results/p5/determinism.json` |
| Cross-execution / log-disagreement audit | `results/p5/repro_audit.json` |
| Environment fingerprint | `results/p4/environment.json` |

Key hashes (SHA-256):

```
c7291bffb0b08040b8d42e43aefbabdc163f95b7348153de9970a57fa7d11199  results/p4/cost_measurement.json
0986b8ed53345a6dbcf0459856abc94b9a2b3879b47cfa986895327ed6885eac  results/p4/per_node_screening.json
b015532799bffda041a3e80926267491fa3f93445ec56eeb6338813b0f24d30f  results/p5/bithe20_analysis.json
d111dd70fbe8e96e3c7276f86489342389ea44ba915d6dbea42b99f26822460c  results/p6/nonoracle_attack_matrix.json
```

---

## 12. Energy stance and superseded cost variants

**Energy was not measured.** The development GPU exposes no power limit
(`environment.power_limit_w = "[N/A]"`), so the paper reports **no joule,
no power draw, and no energy-efficiency figure**. The cost chain reports
wall-clock seconds only. This statement appears in Section VI-E, Table XVIII,
S-II, S-III, S-V and Table S-XXII, and in the Cover Letter.

Two historical result files did contain power-derived joule values. Both are
archived **byte-for-byte** under `archive/superseded/` for audit only, and
neither may be used to regenerate tables or figures:

| Archived file (SHA-256) | Withdrawn content | Final-manuscript value |
| --- | --- | --- |
| `cost_measurement.2026-11-03.superseded-measured-energy.json` (`bd803489…`) | `35.7963 s`, `16.5181 s` one-off, `total_energy_j = 2214.953`, `energy_all_measured = true` | `25.9254 s`, `11.5117 s`, `total_energy_j = null`, `energy_all_measured = false` |
| `per_node_screening.2026-11-03.superseded-measured-energy.json` (`acc585ae…`) | three `joule_per_tx` values, `power_measured = true` | all `joule_per_tx = null`, `power_measured = false` |

The published `results/p4/cost_measurement.json` is the authoritative B
variant. The published `results/p4/per_node_screening.json` keeps **exactly
the same latency, budget, margin, batch-curve and verdict values** as the
archived run; only the withdrawn energy fields were nulled, and an
`energy_measurement` record points back to the archive. `p4_screening.py
--measure_power` now defaults to `0` (opt-in only). See
`archive/superseded/README.md` for the full rationale.

> **Do not run `build` with the archived variants.** The guard in
> `paper_tdsc.py` will raise, by design.

---

## 13. H1–H6 and the threshold-set version map

| ID | Requirement | Final verdict |
| --- | --- | --- |
| H1 | Illicit F1 ≥ `0.4320` **and** parameters ≤ `20,480` | **Falsified** on Elliptic for size (`30,171`), not accuracy |
| H2 | Per-transaction screening latency ≤ `142.857 ms` (7 queries/s) | **Pass**, `7.362 ms` (p95 `11.258 ms`); the only gate TASGL clears |
| H3 | Relative F1 drop after 50% edge deletion | **Interval only, no gate**: TASGL `12.34%` vs GraphSAGE `3.37%`, MLP `0.00%` |
| H4 | Relative F1 loss from a short training schedule | **Interval only, no gate**: TASGL `32.97%` vs MLP `6.94%` |
| H5 | Positive-class ECE ≤ `0.05` and below the same-backbone baseline | **Falsified** on the measured seed: `0.3818` (overall ECE `0.1222` is the wrong summary) |
| H6 | Clean / attacked F1 under a pre-registered non-oracle matrix | **Measured, not gated**; no alert-capacity anchor exists, so no tolerance is derivable |

Four of the five pre-registered V0 thresholds had no source and were
**withdrawn**: F1 `0.73`, parameters `20000`, latency `8.0 ms` (full-graph
semantics), H3 ratio `0.6`, and attack tolerance `0.05` (historically filed
under H4). V3 is the registered set in Table III, lock
`e10ba4b3df3a9392a943365a`. A protocol-5 pass under withdrawn V0 must never
be read beside a falsification under registered V3; `results/p5/protocol_threshold_map.json`
labels every protocol with the rule it was judged under. H6 is outside the
H1–H5 gate set and does not enter `all_pass`.

---

## 14. Verification status and commands

Executed on the authoritative root tree during the 2026-11-04 release pass:

```
python3 -m pytest -q (full workspace)       → 249 passed, 1 skipped, 2 warnings
python3 -m pytest -q (release snapshot)     → 244 passed, 6 skipped, 2 warnings
python3 paper_tdsc.py build --results_dir . → 13 tables + figure inputs + manifest
paper_tdsc.py check (manuscript workspace)  → scanned 564 numbers, 0 not traceable
bash -n run_*.sh                            → all portable entry points OK
sha256sum -c RELEASE_FILELIST.sha256        → 1245 / 1245 OK, 0 FAILED
max shipped file size                       → 755,186 bytes (GitHub limit 100 MB)
```

The two warnings are benign: a PyTorch `index_reduce_` beta API notice and a
`unittest` deprecation warning about a test method returning a value.

Additional consistency checks worth re-running after any change:

```bash
# no withdrawn energy claims in generated tables
grep -RniE "energy is measured|board-level energy|power.draw|J/tx" generated_tables/
# no machine-specific paths in shipped code
grep -RniE "C:\\\\Users\\\\apple|/d/TASGL_experiment|/c/Users" --include='*.py' --include='*.sh' .
```

The first command should only match *negative* statements such as "Energy was
not measured"; the second should return nothing.

---

## 15. Known limitations and open items

These are stated in the paper, not hidden, and **none is reported as a
result**:

- **Four limitations** (Section S-III / conclusion): a third dataset that
  can actually exercise the temporal component; wider tuning grids;
  white-box and model-retraining attackers beyond the non-oracle matrix;
  external expert review of false positives and false negatives.
- **Five ledger items remain open** (Section S-II), tracked separately:
  external expert review of audit cases, formal cross-chain semantic
  alignment, faithful uncertainty control beyond the same-backbone proxy,
  publication of the repository and data lineage, and metric completeness
  beyond PR-AUC.
- **Energy measurement remains outstanding** and is deliberately listed as
  a measurement limitation, not as an open result.
- Single-machine timings; no board-level energy; no white-box attacker; no
  attacker that can retrain the defender; feature spaces not aligned across
  datasets; main Elliptic run is 10 seeds (exact two-sided Wilcoxon floor
  `0.00195`), BitHeNet confirmation is 20 seeds (floor `1.91 × 10⁻⁶`).
- `Test-log/` ends at the 2026-11-02 formal wave. P4/P5/P6 result JSONs are
  newer than the last formal log entries; updating the formal log ledger is
  a separate housekeeping task and does not change any reported number.
- `paper_constants.json` in the manuscript workspace still records a
  `0.2232 µs/node` full-graph amortisation while the body text uses
  `0.2112 µs/node`. Both are descriptive, not gate values; this discrepancy
  is **unresolved** and is flagged here rather than silently changed.

---

## 16. Citation, author statements, and before you push

If you use this code or data, cite the associated ***Journal manuscript and
this repository (add the DOI / repository URL once available). The author
statements in the manuscript are: the manuscript was written by Xiaole Han
alone; no external funding was received; no competing interests are
declared; the study analyses public benchmark transaction records only and
introduces no private user data.

**Before uploading, three decisions are still the author's to make:**

1. **LICENSE** — none is included. Adding a license is a legal decision;
   choose one explicitly (e.g. MIT / Apache-2.0 / CC-BY-4.0 for data) and
   add the file. Until then the repository is *not* licensed for reuse.
2. **Manuscript PDFs / LaTeX sources** — currently excluded. Confirm the
   IEEE policy that applies to your submission before posting a preprint or
   the source tree.
3. **Generative-AI disclosure** — the manuscript's Author Statements contain
   no AI-usage disclosure. If any tooling materially assisted the writing,
   review IEEE PSPB 8.2.1.B.10 before publication and keep the disclosure
   consistent with the sole-authorship statement.

`OPENSOURCE_MANIFEST.md` lists every shipped file with its SHA-256 and the
reason for inclusion or exclusion; regenerate it if you change the file set.
