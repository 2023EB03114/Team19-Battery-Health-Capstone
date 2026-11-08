# Phase 3 Output Data Dictionary

Each `outputs/<dataset>/heldout_predictions.csv` contains one row per held-out record. CSGB, NASA, and CALCE use the common 39-column contract. TRI uses the same contract plus three source-rate fields.

## Identity and inputs

| Column | Meaning |
|---|---|
| `record_id` | Unique aligned record key used by metrics and graphs |
| `source_dataset` | Source/provenance label |
| `battery_id` | Physical cell/battery or benchmark group |
| `cycle_count` | Cycle/order index; TRI uses a documented zero sentinel because prior aging count is absent |
| `temperature_c` | Temperature in °C; TRI uses a documented constant schema placeholder, not a measurement |
| `discharge_current_a` | Representative discharge current in A |
| `avg_voltage_v`, `min_voltage_v` | Representative and minimum discharge voltage in V |
| `internal_resistance_ohm` | Resistance/EIS or documented apparent-sag proxy |
| `load_power_w` | Representative load power in W |
| `rated_capacity_ah` | Reference/rated capacity in Ah |

## TRI-only source-rate fields

| Column | Meaning |
|---|---|
| `source_rate_index` | Verified characterization sequence number 1-8 |
| `source_rate_label` | C/80, C/40, 0.05 A, C/10, C/7, C/5, 1C, or 2C |
| `source_nominal_c_rate` | Numeric nominal C-rate used for stratified analysis |

## Actual and predicted outputs

| Column | Meaning |
|---|---|
| `measured_capacity_ah` | Actual held-out capacity target |
| `predicted_capacity_ah` | M5 predicted capacity |
| `actual_soh_pct` | Actual capacity / rated capacity × 100 |
| `predicted_soh_pct` | Predicted capacity / rated capacity × 100 |
| `backup_time_min` | Actual held-out discharge/backup-time target |
| `formula_reference_min` | M4 consistency diagnostic using measured capacity |
| `formula_operational_min` | M4 operational comparator using M5-predicted capacity |
| `ml_backup_time_min` | M6 predicted backup time |

The M4 reference calculation is an internal known-answer/energy-consistency check. It is not compared as an independent predictive model because it uses the measured target capacity.

## Error fields

For capacity, SOH, M4 operational runtime, and M6 runtime, the export contains signed error, absolute error, and absolute percentage error. Signed error is `prediction - actual`.

## M9 fields

| Column | Meaning |
|---|---|
| `expected_health_class` | Benchmark or deterministic measured-target reference |
| `predicted_health_class` | M9 class from predicted outputs |
| `health_class_match` | Exact-match Boolean |
| `expected_recommendation_label` | Reference recommendation label |
| `predicted_recommendation` | M9 predicted recommendation |
| `recommendation_match` | Exact-match Boolean |
| `decision_level`, `decision_message` | Final secondary guidance output |

NASA/CALCE/TRI M9 references are deterministic threshold labels, not expert annotations.

## TRI rate-level file

`outputs/tri/rate_level_metrics.csv` contains one row for each of the eight source rates, with record/cell counts, capacity and SOH summaries, M4/M6 errors, and the lower-MAE method. All eight rows use the same 24 held-out physical cells.
