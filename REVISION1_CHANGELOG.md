# Phase 3 Revision 1 Changelog

This final submission edition replaces the supplied Phase‑3 working draft.

## Evidence corrections

- Replaced unsupported synthetic-only Phase‑3 claims with an executable three-domain validation.
- Retained CSGB as a controlled synthetic continuity benchmark with 720 records.
- Added verified NASA PCoE processing and grouped evaluation on 579 real discharge-cycle records.
- Added verified CALCE CS2 processing and grouped evaluation on 3,544 real discharge-cycle records.
- Implemented and fixture-tested the Oxford source-native MAT adapter contract; explicitly omitted Oxford metrics because the official file was not processed.
- Corrected the evaluation to battery-group holdouts with zero physical-group overlap.
- Reported the real finding that M4 slightly outperforms M6 on NASA/CALCE instead of assuming ML superiority.

## Implementation additions

- Added `source_adapters.py` and reproducible public-data preparation code.
- Added complete `run_phase3_validation.py` workflow.
- Added global/local SHAP exports and figures for all evaluated domains.
- Added six persisted models with reload consistency checks.
- Added a six-tab Streamlit dashboard and pure backend prediction function.
- Added input guards, power-consistency warning and conservative M9 runtime basis.
- Added 20 comparison, residual, ageing, explainability, persistence and M9 graphs.
- Added aligned held-out prediction exports and complete metric JSON/CSV files.

## Verification and submission corrections

- Added output-contract verification and 18 automated tests.
- Added Windows launchers that automatically use `.venv` when present.
- Added final 27-page DOCX/PDF report, accessibility fixes and visual QA.
- Added data-provenance documentation, source hashes, validation summary, readiness checklist and SHA‑256 manifest.
- Excluded large raw public archives and the obsolete working-draft PDF.

