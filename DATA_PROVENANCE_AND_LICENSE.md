# Data Provenance, Licensing, and Redistribution Boundary

## CSGB v1.0

- Role: controlled synthetic continuity and known-answer benchmark retained from Phase 2.
- Included file: `Phase3_Implementation/data/Battery_Base_Data_Benchmark_v1.0.xlsx`.
- Evidence category: controlled synthetic; never presented as real experimental accuracy.

## NASA PCoE battery data

- Official repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Processed batteries: B0005, B0006, B0007, and B0018.
- Included prepared output: `nasa_cycle_summary.csv` (579 discharge-cycle records).
- Archive and MAT-file SHA-256 hashes are recorded in `public_data_provenance.json`.

## CALCE battery data

- Official source: https://calce.umd.edu/battery-data
- Processed batteries: CS2_35, CS2_36, CS2_37, and CS2_38.
- Included prepared output: `calce_cycle_summary.csv` (3,544 discharge-cycle records).
- Official archive URLs and SHA-256 hashes are recorded in `public_data_provenance.json`.

## TRI diagnostic-free battery health data

- Official dataset page: https://data.matr.io/10/
- Official archive: https://data.matr.io/10/ResValData.zip
- Associated paper DOI: https://doi.org/10.1016/j.joule.2025.102010
- License stated by the provider: **CC BY-NC 4.0**.
- Raw archive SHA-256: `5898b06bf34550c2b57ea345c488d71fb24db4fbe6030225e260b062e22e840c`.
- Raw archive size: 250,042,781 bytes; raw archive is not redistributed in this submission.
- Included prepared output: `tri_resval_cycle_summary.csv` (750 observed rate-characterization records from 94 aged Panasonic 21700 cells).
- Conditions: C/80, C/40, 0.05 A, C/10, C/7, C/5, 1C, and 2C, with discharge summarized to a 2.7 V cutoff.
- Source exception: 93 cells supply all eight records; cell `0000ED` supplies six low-rate records and no usable 1C/2C record. The authors' released processing code also skips that cell.
- Missing metadata: no measured temperature or prior aging-cycle count is present in the raw ResVal files. The adapter's constant temperature and zero cycle-count values are explicitly marked schema placeholders, not observations.

## Oxford Battery Degradation Dataset 1

- Official source: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac
- DOI: https://doi.org/10.5287/bodleian:KO2kdmYGg
- Status: native MAT contract implemented in `source_adapters.py` and fixture-tested; official source file not processed in this package.
- Quantitative evaluation: none claimed.

## Redistribution position

Large raw NASA, CALCE, TRI, and Oxford archives are not included. The ZIP contains compact project-derived cycle summaries, the controlled benchmark, source URLs, hashes, quality metadata, and processing code. Source ownership, citation, and reuse conditions remain with the providers. Reproduction requires obtaining the files from the official sources and complying with their applicable terms.
