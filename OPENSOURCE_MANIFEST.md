# Open-Source Release Manifest — TASGL Experiment Suite

**Release anchor:** 2026-11-04 camera-ready ***Journal submission (17-page main
text + 12-page Supplemental Material).
**Uploadable snapshot:** `TASGL_github_release/`
**Per-file SHA-256 ledger:** `RELEASE_FILELIST.sha256` (in both the repository
root and the snapshot; the ledger covers every file in the snapshot except
itself).

This document is the version-management record of *what is published, why,
and how it was verified*. It is written for a reviewer who wants to check that
the public code and data do not contradict the final manuscript.

---

## 1. Release statistics

| Item | Value |
| --- | --- |
| Files in snapshot (before this manifest and the hash ledger) | **1244** |
| Total size | **5,884,801 bytes ≈ 5.88 MB** |
| Largest single file | `results/p4/predictions_elliptic_seed0_mc.json`, 755,186 bytes |
| GitHub 100 MB single-file limit | satisfied with a >130× margin |
| Root files | 102 |
| `results/` | 1087 |
| `Test-log/` | 44 |
| `figs/` | 5 |
| `reference_checks/` | 3 |
| `archive/superseded/` | 3 |

Breakdown of root files: 69 `.py` modules (including the complete P5/P6
code), 18 `test_*.py` modules, 11 portable `*.sh` entry points, plus
`README.md`, `README.cn.md`, `requirements.txt`, `pytest.ini`, `.gitignore`
and the root configuration `*.json` files.

After `OPENSOURCE_MANIFEST.md` and `RELEASE_FILELIST.sha256` are added, the
snapshot contains **1246 files**.

**Publish the *contents* of `TASGL_github_release/` as the repository root.**
The snapshot is a build artifact of the working tree; `.gitignore` excludes
`TASGL_github_release/` (and `pytest.ini` excludes it from test collection)
so it is never pushed as a nested duplicate of the same modules.

---

## 2. Inclusion / exclusion decisions

| Decision | Path / pattern | Reason |
| --- | --- | --- |
| INCLUDE | root `*.py`, `test_*.py` | authoritative experiment source tree (P0–P6) |
| INCLUDE | root portable `*.sh` | reproduction entry points; all use `TASGL_PYTHON` and `cd` to their own directory |
| INCLUDE | root `*.json` config, `requirements.txt`, `pytest.ini`, `.gitignore` | reproducible configuration |
| INCLUDE | `README.md`, `README.cn.md`, this manifest | reviewer-facing documentation, section-by-section bilingual |
| INCLUDE | `results/**/*.json`, small `*.csv`, `*.log`, `*.txt` | every reported number traces to one of these files |
| INCLUDE | `Test-log/`, `figs/`, `reference_checks/` | 44 timestamped logs, 5 diagnostic PDFs, 3 literature-check CSVs (0.54 MB total) |
| INCLUDE | `archive/superseded/` (2 JSON + README) | audit trail for the two withdrawn energy-bearing variants |
| EXCLUDE | `Data/`, `data/` | **2.8 GB third-party raw data**, not redistributable; several files >400 MB. Obtain with `download_data.py` |
| EXCLUDE | `Doc/` | 280 MB / 2779 files: historical drafts, Journal templates, prior source snapshots, internal working notes. Contains an older 47-file copy of the same code |
| EXCLUDE | `TASGL论文证据追溯/` | internal evidence-traceability tree from earlier (pre-final) manuscript drafts |
| EXCLUDE | `*.pt`, `*.model`, `*.ckpt`, `*.npz` | ~43 MB of weights/checkpoints; no reported number depends on them |
| EXCLUDE | `results/p3/independent_dgraph/**/case_audit_queue.csv` | 10 files, ~84 MB, regenerable intermediate case queues |
| EXCLUDE | `*.bat` | legacy Windows-only wrappers containing a machine-specific interpreter path; portable `.sh` equivalents are shipped |
| EXCLUDE | root `*.txt` run logs, draft `*.md`, draft `*.pdf`, `*.docx` | historical or superseded prose, not part of the code release |
| EXCLUDE | manuscript LaTeX tree, `main.pdf`, `supplemental_material.pdf` | separate Journal submission package; also an Journal copyright/self-archiving decision |
| EXCLUDE | `__pycache__/`, `.pytest_cache/`, `.idea/`, `.venv/`, `.git/`, `.workbuddy/` | caches, IDE state, private working records |

**Log redaction (release-only).** 41 `Test-log/*.txt` files and 9
`results/**/*.log` files originally recorded the development machine's
interpreter and work directory. In the *snapshot copies only*, those
occurrences were replaced with portable placeholders (`python3`,
`<home>`, `<tmp>`, `<site-packages>`, `<python-env>`, and relative paths).
No timing, seed, metric, verdict or timestamp value was altered. The root
originals are untouched; the environment fingerprint in
`results/p4/environment.json` remains the authoritative hardware record.

---

## 3. Key file hashes (SHA-256)

```
c7291bffb0b08040b8d42e43aefbabdc163f95b7348153de9970a57fa7d11199  results/p4/cost_measurement.json
0986b8ed53345a6dbcf0459856abc94b9a2b3879b47cfa986895327ed6885eac  results/p4/per_node_screening.json
b015532799bffda041a3e80926267491fa3f93445ec56eeb6338813b0f24d30f  results/p5/bithe20_analysis.json
d111dd70fbe8e96e3c7276f86489342389ea44ba915d6dbea42b99f26822460c  results/p6/nonoracle_attack_matrix.json
bd803489253848a7371cc81c394cfd2c1917f194a7f983b46a2f9c3b04fb522e  archive/superseded/cost_measurement.2026-11-03.superseded-measured-energy.json
acc585ae515e44fa933cc3885985d972c2c23838bc56783551392a15fda39ec8  archive/superseded/per_node_screening.2026-11-03.superseded-measured-energy.json
```

Verification:

```bash
cd TASGL_github_release
sha256sum -c RELEASE_FILELIST.sha256     # all files OK
```

Windows PowerShell alternative:

```powershell
Get-Content RELEASE_FILELIST.sha256 | ForEach-Object {
  $h, $p = $_ -split '  ', 2
  if ((Get-FileHash $p -Algorithm SHA256).Hash.ToLower() -ne $h) { "MISMATCH: $p" }
}
```

---

## 4. Final-manuscript number → data file map

| Manuscript value | File in snapshot |
| --- | --- |
| `0.5081 ± 0.1088`, `0.4970 ± 0.0734`, `0.4394 ± 0.0820`, `30,171`, `45.48 ms` | `results/p4/final_comparison.json` |
| `0.5593 ± 0.1101`, `0.5079 ± 0.1252`, `0.4112 ± 0.0394` (20 raw seeds) | `results/p4/second_dataset_20seeds.json` |
| `+0.1481`, CI `[0.0887, 0.2074]`, Holm Wilcoxon `p = 5.23 × 10⁻⁴`, `dz = 1.168`, `17/20` | `results/p5/bithe20_analysis.json` |
| H2 `7.362 / 11.258 ms` against `142.857 ms` | `results/p4/per_node_screening.json` |
| `25.9254 s` total, `11.5117 s` one-off (`44.4%`), 8 stages, energy `null` | `results/p4/cost_measurement.json` |
| Registered V3 thresholds (`0.4320`, `20,480`, `142.857 ms`, `0.05`) | `results/p4/thresholds/register.json`, `results/p4/threshold_justification.json` |
| Withdrawn V0 values and protocol→rule version map | `results/p5/protocol_threshold_map.json` |
| H3 `12.34% / 3.37% / 0.00%`, H4 `32.97% / 6.94%` | `results/p4/h3_interval.json`, `results/p4/h4_interval.json` |
| Gate ratio `1.000`, early-stopped seeds, `18/20` repeats | `results/p4/gate_audit_10seeds.json` |
| Positive-class ECE `0.3818`, overall ECE `0.1222`, seed-0 `0.3049` root cause | `results/p4/rootcause_seed0.json`, `results/p4/calibration_predictions_elliptic_seed0.json` |
| H6 matrix, 720 runs, `45.5% [32.7%, 59.4%]` | `results/p6/nonoracle_attack_matrix.json` |
| Positive control `7/7`, `all_pass` verdicts | `results/p6/positive_control.json`, `results/p6/gate_evaluation.json` |
| Withdrawn claims ledger | `results/p4/claims.json` |
| DGraph-Fin probe (no gate result) | `results/p5/third_dataset_table.json` |
| Non-determinism `1.788 × 10⁻⁷` | `results/p5/determinism.json` |
| Cross-execution / log-disagreement audit | `results/p5/repro_audit.json` |
| Environment fingerprint | `results/p4/environment.json` |

---

## 5. Consistency governance applied before this release

1. **A/B cost-source decision.** The root `results/p4/cost_measurement.json`
   was the superseded A variant (`35.7963 s`, `total_energy_j = 2214.953`,
   `energy_all_measured = true`). It was archived byte-for-byte and replaced
   with the final-manuscript B variant (`25.9254 s`, `11.5117 s`, energy
   `null`). See `archive/superseded/README.md`.
2. **H2 energy-field withdrawal.** `results/p4/per_node_screening.json`
   contained three `joule_per_tx` values and `power_measured = true` while the
   same file recorded `power_limit_w = "[N/A]"`. The original was archived and
   the published copy nulls those fields only; every latency, budget, margin,
   batch-curve and verdict value is unchanged. An `energy_measurement`
   record points back to the archive.
3. **Regeneration guard.** `paper_tdsc.py::assert_energy_not_measured()`
   rejects any cost source with non-null `energy_j` / `total_energy_j` and any
   screening source with non-null `joule_per_tx` or `power_measured = true`.
   `tex_h2()` no longer emits a `J/tx` column; its caption/footer match the
   submitted `h2_screening.tex`. `tex_cost()` one-off/recurring classification
   matches Table S-XXII (11.51 s / 14.41 s).
4. **Measurement default.** `p4_screening.py --measure_power` now defaults to
   `0`; power measurement is opt-in and only meaningful on a device with a
   readable power rail.
5. **Portability.** `run_all.sh`, `run_hypotheses.sh` and the four P4/P5 wave
   scripts use `TASGL_PYTHON` and `cd` to their own directory.
   `test_p0.py` / `test_p1.py` use `sys.executable` and only assert on `.bat`
   wrappers when they are present (they are not shipped). `p1_release.py`'s
   redaction fixture now uses a generic example path. Launcher tests remain
   green on a clean checkout.
6. **Test isolation.** `pytest.ini` limits collection to the root tree and
   excludes `Doc/` and `TASGL_github_release/`, so neither historical module
   copies nor the nested release snapshot can produce duplicate-module
   collection errors.
7. **Scientific-notation traceability.** `_scientific_mantissas()` lets the
   checker match `5.23 × 10⁻⁴` in the body against `0.000523` in a generated
   table. Both generator snapshots now report zero unknown numbers.

---

## 6. Verification actually executed

```
python3 -m pytest -q (full workspace)                 → 249 passed, 1 skipped, 2 warnings  (54.48 s)
python3 -m pytest -q (isolated snapshot copy)         → 244 passed, 6 skipped, 2 warnings  (36.78 s)
python3 paper_tdsc.py build --results_dir .
        --paper_dir /tmp/tasgl_build_check            → 13 tables + figure inputs + manifest, exit 0
tasgl_experiment/paper_tdsc.py check
        --paper_dir TASGL_TDSC投稿                    → scanned 564 numbers, 0 not traceable
bash -n run_all.sh run_hypotheses.sh
        run_p4_final_wave1.sh run_p4_final_wave2.sh
        run_p5_wave1.sh run_p5_wave2.sh               → OK
grep on regenerated tables for joule / power.draw /
        J/tx / "energy is measured"                  → only the negative "Energy was not measured" statements
sha256sum -c RELEASE_FILELIST.sha256                  → 1245 / 1245 OK, 0 FAILED
max snapshot file                                     → 755,186 bytes (< 100 MB)
machine-specific path scan on snapshot                → 0 hits outside the two READMEs' own verification examples
```

The five extra skips in the isolated snapshot are expected and correspond
exactly to the excluded artifacts: one `test_p4.py` case needs
`./data/elliptic`, three `test_p5_tex_conformance.py` cases need the
submission LaTeX tree, and one `test_paper_tdsc.py` case needs the submission
directory. The sixth skip (`test_paper_evidence.py`) also occurs in the full
workspace because `TASGL_EVIDENCE_ZIP` is not set. **No test fails in either
tree.**

Note: running the root `pytest` writes `traceability_check.json` in the
manuscript workspace with the root scanner's `550 / 0`. The manuscript's own
generator was re-run afterwards to restore the authoritative `564 / 0`.
Both agree on zero untraceable numbers; see Section 10 of `README.md` /
`README.cn.md`.

---

## 7. Known unresolved items (disclosed, not silently changed)

- No `LICENSE` file: choosing one is the author's legal decision. Do not
  publish as "open source" without it.
- No decision yet on posting the manuscript PDFs / LaTeX sources (Journal policy).
- No generative-AI usage disclosure in the manuscript's Author Statements;
  review Journal policy if tooling materially assisted the writing.
- `paper_constants.json` (manuscript workspace) still records
  `0.2232 µs/node` while the body says `0.2112 µs/node`; both descriptive,
  not gate values, discrepancy unresolved.
- Formal `Test-log/` entries stop at 2026-11-02 while P4/P5/P6 result JSONs
  are newer; log backfill is housekeeping and changes no reported number.
- `fig_gate.pdf` visual inspection, supplemental `\vbox` Overfull warning,
  and reference [32] verification remain on the manuscript-side P1/P2 list.

---

## 8. Regenerating this manifest

The snapshot is a *copy*, not a symlink. If the root source tree changes,
rebuild the snapshot, re-copy both READMEs, re-run the path-redaction pass on
`Test-log/` and `results/**/*.log`, and regenerate `RELEASE_FILELIST.sha256`
over the final file set. Update the statistics in Section 1 and the hashes in
Section 3 before uploading.
