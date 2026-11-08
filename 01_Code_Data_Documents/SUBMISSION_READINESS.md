# Submission Readiness - Revision 2

## Technical status

**Ready for academic Phase-3 submission after the team performs the limited identity/signature checks below.** The code, prepared data, record-level outputs, graphs, models, report, and verification evidence are internally aligned and protected by SHA-256 checksums.

Completed checks:

- [x] Project title, team details, and supervisor details remain consistent with Phase 1.
- [x] Controlled synthetic and real laboratory evidence are separated.
- [x] NASA, CALCE, and TRI use physical-battery/cell grouped holdouts with zero overlap.
- [x] TRI's 750 records, 94 cells, eight rates, source exception, and missing-metadata limitations are documented.
- [x] Oxford remains stated as adapter/fixture only; no Oxford metric is claimed.
- [x] M4, M5, M6, M7, M8, M9, and dashboard evidence are documented.
- [x] Formula-versus-ML results are reported even where M4 has lower error.
- [x] Twenty-two graphs and eight models are present and readable.
- [x] Twenty tests were executed; 19 passed and the Streamlit AppTest was explicitly skipped because its optional runtime dependency was unavailable.
- [x] Dashboard backend, guards, syntax, output contract, analytical pipeline, and package verifier pass.
- [x] Raw public archives are excluded; processed summaries, source URLs, hashes, quality files, and reproduction scripts are included.
- [x] DOCX/PDF render, accessibility, comments, metadata, watermark, and blank-page checks completed.
- [x] Manifest and ZIP integrity verified.

## Items to check from the team

These are administrative, not analytical changes:

1. Confirm whether the institution issued a separate cover-page template or file-naming rule.
2. Add physical or digital signatures to the declaration table only if required.
3. If the portal requires an official submission date, update only that date; do not alter the verified metrics or output files.
4. Install `requirements.txt` before locally rerunning the Streamlit AppTest/dashboard. The supplied analytical evidence is already generated and does not require a rerun for review.
5. Upload the final ZIP intact. Do not remove the manifest, report, prepared data, outputs, graphs, models, or provenance files.

## Viva/presentation wording

Use this precise description: NASA, CALCE, and TRI provide real **controlled laboratory** evidence. TRI substantially increases physical-cell and rate coverage, but it is not field validation. M4 has slightly lower aggregate runtime MAE on TRI, while M6 has lower MAPE and wins five of eight rate-level MAE comparisons. This mixed result supports retaining both the transparent formula comparator and the learned model.
