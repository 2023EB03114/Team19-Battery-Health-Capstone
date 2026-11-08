# Team 19 - Phase 3 Complete Submission, Revision 2

Project: **Explainable Machine Learning Framework for Battery Health Assessment and Dynamic Backup-Time Prediction**  
Course: **BCSZC241T**  
Team: **Axiom AI (Team ID 19)**

This Revision-2 package supersedes the Phase-3 Revision-1 submission. It adds the official TRI/Matr.io multi-rate laboratory dataset and updates the code, report, graphs, models, tests, provenance, and verification counts consistently.

## Start here

1. Open `Report/Team19_Phase3_Implementation_Validation_Final.pdf`.
2. Read `VALIDATION_SUMMARY.md` for the main results and evidence boundary.
3. Read `SUBMISSION_READINESS.md` before uploading.
4. To reproduce the outputs on Windows, follow `Phase3_Implementation/WINDOWS_QUICK_START.txt`.
5. After extracting the ZIP, run `python verify_submission_package.py` to verify every checksum and required output count.

## Package contents

- `Report/` - corrected final report in DOCX and PDF.
- `Phase3_Implementation/` - M1-M10 code, prepared inputs, aligned outputs, 22 graphs, eight persisted models, tests, batch launchers, and execution evidence.
- `Prior_Phases/` - final Phase-1 and Phase-2 reports for continuity.
- `Reference/` - supplied Phase-3 deliverables checklist.
- `VALIDATION_SUMMARY.md` - concise numerical result summary.
- `SUBMISSION_READINESS.md` - completed checks and limited team-side actions.
- `DATA_PROVENANCE_AND_LICENSE.md` - source URLs, evidence classes, licensing, and redistribution boundary.
- `REVISION2_CHANGELOG.md` - additions and corrections in this revision.
- `MANIFEST_SHA256.csv` - SHA-256 digest and byte size for every other included file.

## Verified output totals

- 5,593 validated records: 720 CSGB controlled records and 4,873 real laboratory records.
- Real records: 579 NASA, 3,544 CALCE, and 750 TRI records from 94 physical cells.
- 2,532 aligned held-out predictions: 180 CSGB, 281 NASA, 1,879 CALCE, and 192 TRI.
- Zero physical-battery overlap in all four development/holdout splits.
- 22 readable PNG comparison/explanation graphs.
- Eight persisted Joblib models.
- 20 automated tests executed: 19 passed and one dependency-aware Streamlit AppTest was skipped because Streamlit was unavailable in the build runtime.
- Full analytical pipeline, output contract, dashboard backend, input guards, model reload, and submission verification: PASS.

## Evidence boundary

CSGB is a controlled synthetic continuity benchmark. NASA PCoE, CALCE, and TRI are real public laboratory datasets. TRI expands the evidence to 94 aged 4.84 Ah Panasonic 21700 cells characterized across C/80 to 2C, but it is controlled constant-current laboratory data rather than field UPS/inverter telemetry. The Oxford native-MAT adapter remains fixture-tested without a quantitative Oxford claim.

The raw public archives are not redistributed. Compact derived cycle summaries, quality metadata, official URLs, hashes, and preparation code are included so the validation pipeline runs offline and the source preparation can be independently reproduced.
