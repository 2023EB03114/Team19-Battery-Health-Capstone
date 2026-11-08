# Revision 2 Changelog

Revision 2 expands the Phase-3 empirical evidence without changing the project title or safety boundary.

## Added

- Official TRI/Matr.io ResVal real laboratory dataset integration.
- Streaming raw-archive adapter in `prepare_tri_resval_dataset.py`.
- Compact 750-row cycle/rate summary from 94 aged 4.84 Ah Panasonic 21700 cells.
- Eight-rate held-out analysis from C/80 through 2C.
- TRI quality/provenance JSON, 192 aligned held-out predictions, rate-level metrics, feature importance, global/local Shapley outputs, M9 comparisons, and two persisted models.
- Two TRI-specific explanation graphs; common cross-domain graphs expanded to four datasets.

## Corrected totals

- Validated records: 4,843 -> **5,593**.
- Real laboratory records: 4,123 -> **4,873**.
- Held-out aligned predictions: 2,340 -> **2,532**.
- Graphs: 20 -> **22**.
- Persisted models: 6 -> **8**.
- Automated tests: 18 -> **20 executed** (19 pass, one dependency-aware Streamlit AppTest skip in the build runtime).

## Interpretation changes

- M4 retains lower aggregate runtime MAE on TRI (33.681 versus 34.638 minutes).
- M6 has lower TRI aggregate MAPE (3.34% versus 3.56%) and lower MAE at five of eight discharge rates.
- TRI's low-rate tests last hundreds to thousands of minutes; absolute-minute MAE and percentage error are therefore reported together.
- The report explicitly identifies TRI's missing temperature and prior aging-cycle metadata and does not infer an aging trajectory from rate order.
- TRI is classified as controlled laboratory evidence, not field UPS/inverter telemetry.

## Verification changes

- Output and package verifiers now require four datasets, 2,532 held-out rows, 22 graphs, and eight models.
- Model reload tolerance is `1e-9`, still far below practical measurement resolution; the largest observed difference is `1.82e-12` minutes.
- A deterministic 24-permutation Shapley fallback preserves global/local explanation exports when the optional `shap` package is unavailable; exact TreeSHAP remains enabled when installed.
