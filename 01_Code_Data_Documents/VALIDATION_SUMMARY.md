# Phase 3 Revision-2 Validation Summary

Validation was executed on 20 August 2026 with Python 3.12.13 and fixed `random_state=42`. The four-domain analytical suite and package output contract passed.

## Dataset and split integrity

| Dataset | Evidence domain | Valid rows | Development / holdout | Held-out groups | Overlap |
|---|---|---:|---:|---:|---:|
| CSGB v1.0 | Controlled synthetic | 720 | 540 / 180 | 3 benchmark batteries | 0 |
| NASA PCoE | Real laboratory | 579 | 298 / 281 | 2 physical batteries | 0 |
| CALCE CS2 | Real laboratory | 3,544 | 1,665 / 1,879 | 2 physical batteries | 0 |
| TRI ResVal | Real laboratory | 750 | 558 / 192 | 24 physical cells | 0 |

Totals: **5,593 records**, including **4,873 real laboratory records**, and **2,532 held-out predictions**.

## Main held-out results

| Dataset | M5 capacity MAE (Ah) | Capacity MAPE | SOH MAE (pp) | M4 runtime MAE (min) | M6 runtime MAE (min) | Lower runtime MAE |
|---|---:|---:|---:|---:|---:|---|
| CSGB | 0.0560 | 3.29% | 2.60 | 8.756 | 1.519 | M6 |
| NASA | 0.0737 | 4.67% | 3.68 | 2.347 | 2.826 | M4 |
| CALCE | 0.0287 | 3.47% | 2.61 | 1.601 | 1.721 | M4 |
| TRI | 0.1222 | 3.56% | 2.53 | 33.681 | 34.638 | M4 |

The result is intentionally not overclaimed. M6 reduces MAE by 82.66% on varied-load CSGB. M4 has lower aggregate MAE on NASA, CALCE, and TRI. For TRI, M6 still has lower aggregate MAPE (3.34% versus 3.56%) and lower rate-level MAE at C/80, C/40, C/10, 1C, and 2C; M4 is lower at 0.05 A, C/7, and C/5. TRI's very long low-rate tests make absolute-minute MAE scale-sensitive.

## Explainability, persistence, and decision support

- Global/local Shapley status: pass for CSGB (180 rows), NASA (200 sampled rows), CALCE (200 sampled rows), and TRI (192 rows).
- The supplied run used the deterministic 24-permutation Shapley fallback because the optional `shap` package was unavailable; exact TreeSHAP remains enabled when the pinned dependency is installed.
- Dominant M5 built-in features: CSGB internal resistance; NASA/CALCE cycle count; TRI internal resistance.
- Dominant M6 built-in features: CSGB C-rate; NASA/CALCE cycle count; TRI load power.
- All eight persisted models reloaded below the declared `1e-9` tolerance; the largest observed difference was `1.82e-12` minutes.
- M9 health/recommendation exact-match results: CSGB 86.11% / 88.33%; NASA 77.58% / 77.58%; CALCE 85.47% / 85.47%; TRI 74.48% / 77.60%.
- NASA/CALCE/TRI M9 references are deterministic threshold labels derived from measured targets, not expert maintenance annotations.

## Verification evidence

- Python compilation and shared M1-M9 regression checks: pass.
- Complete CSGB/NASA/CALCE/TRI pipeline: pass.
- Output contract: pass (2,532 held-out rows, 22 graphs, eight models).
- Automated tests: 20 executed; 19 passed; one Streamlit AppTest explicitly skipped because Streamlit was unavailable in the build runtime.
- Dashboard backend, input guards, syntax, output inventory, model persistence, graph readability, and package integrity: pass.

Machine-readable evidence is in `Phase3_Implementation/outputs/phase3_metrics.json`, `phase3_metrics_table.csv`, `phase3_test_report.json`, the four `heldout_predictions.csv` files, and `outputs/tri/rate_level_metrics.csv`.

## Evidence boundary

NASA, CALCE, and TRI are genuine public laboratory datasets. TRI adds 94 aged cells and eight characterization rates, but none of these sources constitutes dynamic field telemetry from UPS, inverter, or telecom installations. The results therefore support an academic decision-support prototype, not a safety, certification, warranty, or deployment claim.
