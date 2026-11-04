# Superseded artifacts — retained for audit only

## `cost_measurement.2026-11-03.superseded-measured-energy.json`

- **Original path:** `results/p4/cost_measurement.json`
- **SHA-256:** `bd803489253848a7371cc81c394cfd2c1917f194a7f983b46a2f9c3b04fb522e`
- **Bytes:** 9696

This is the *superseded* cost variant. It reports:

| Field | Superseded value | Final-manuscript value |
| --- | --- | --- |
| `chain.total_seconds` | 35.7963 | 25.9254 |
| `chain.one_off_seconds` | 16.5181 | 11.5117 |
| `chain.total_energy_j` | 2214.953 | `null` |
| `chain.energy_all_measured` | `true` | `false` |
| per-stage `energy_j` | 8 non-null values | 8 × `null` |

It also carries `mean_power_w`, `power_samples=78`, and `energy_measured=true`
per stage, while the same file records `power_limit_w = "[N/A]"`. The final
manuscript states that **energy was not measured** because the device exposes
no power limit; no joule, power, or efficiency figure is claimed anywhere in
the paper. Publishing this variant as the headline cost source would directly
contradict the camera-ready text (Section VI-E, Table XVIII, S-II, S-III, S-V,
and Table S-XXII).

The authoritative final-manuscript measurement now occupies
`results/p4/cost_measurement.json`:

- **SHA-256:** `c7291bffb0b08040b8d42e43aefbabdc163f95b7348153de9970a57fa7d11199`
- **Bytes:** 9465
- `audit.blocking_reasons` includes "energy not measured (do not fill in synthetic values)".

`paper_tdsc.py` refuses to build from a source with any non-null `energy_j`,
`energy_measured=true`, or a non-null `chain.total_energy_j`. The identical
file is also the `--cost_source` recorded in the manuscript's
`provenance/lineage.md`.

The superseded file is kept byte-for-byte so that the version decision is
auditable. **Do not use it to regenerate tables or figures.**

Archived on 2026-11-04 during the open-source release consistency pass.

## `per_node_screening.2026-11-03.superseded-measured-energy.json`

- **Original path:** `results/p4/per_node_screening.json`
- **SHA-256:** `acc585ae515e44fa933cc3885985d972c2c23838bc56783551392a15fda39ec8`
- **Bytes:** 24318

This is the pre-final H2 screening run. Its **latency** numbers are the ones
used in the camera-ready paper (TASGL 7.362 ms / p95 11.258 ms against a
142.857 ms budget), but it also carries three withdrawn `joule_per_tx`
values and `power_measured = true`, while the same file records
`environment.power_limit_w = "[N/A]"`. Those joule values were derived from
power samples that the final manuscript explicitly disclaims:

> Energy was not measured: the device exposes no power limit, so no joule
> figure is reported and the screening verdict is decided by latency only.

The published `results/p4/per_node_screening.json` is the same measurement
with those fields withdrawn: every `joule_per_tx` is `null`,
`power_measured` is `false`, and an `energy_measurement` record points back
to this archive entry. **No latency, budget, margin, batch curve or verdict
value was altered.**

- **Published SHA-256:** `0986b8ed53345a6dbcf0459856abc94b9a2b3879b47cfa986895327ed6885eac`

Two further guards keep regenerated output consistent with the paper:

- `paper_tdsc.py::assert_energy_not_measured()` rejects any cost source with
  a non-null `energy_j` / `total_energy_j` and any screening source with a
  non-null `joule_per_tx` or `power_measured = true`.
- `paper_tdsc.py::tex_h2()` no longer emits a `J/tx` column; its caption and
  footer now match the submitted `h2_screening.tex`.
- `p4_screening.py --measure_power` now defaults to `0`; measuring power is
  opt-in and only meaningful on a device with a readable power rail.

**Do not use the archived file to regenerate tables or figures.**
