# Reproducing the Public-Data Preparation

The included validation pipeline runs from compact prepared CSV files and does not require the large raw archives. To reproduce those CSVs from official files, obtain the source archives under the providers' terms and verify their SHA-256 digests against `data/processed_public/public_data_provenance.json`.

## NASA and CALCE

1. Download the NASA battery archive and four CALCE CS2 archives from the official URLs in the provenance JSON.
2. Extract the NASA MAT files and CALCE workbooks into separate folders.
3. Run:

```text
python prepare_public_datasets.py \
  --nasa-dir <NASA_MAT_FOLDER> \
  --calce-dir <CALCE_EXTRACTED_FOLDER> \
  --nasa-archive <NASA_ARCHIVE_ZIP> \
  --calce-raw-dir <CALCE_ZIP_FOLDER> \
  --output-dir data/processed_public
```

## TRI ResVal multi-rate data

Official page: https://data.matr.io/10/  
Archive: https://data.matr.io/10/ResValData.zip  
Expected SHA-256: `5898b06bf34550c2b57ea345c488d71fb24db4fbe6030225e260b062e22e840c`

Run directly against the ZIP; full extraction is unnecessary:

```text
python prepare_tri_resval_dataset.py \
  --archive <TRI_ResValData.zip> \
  --output data/processed_public/tri_resval_cycle_summary.csv \
  --quality-json data/processed_public/tri_resval_quality.json
```

Expected result: 750 records from 94 cells. Ninety-three cells have eight rates; `0000ED` has six low-rate records. The script stops if those verified source characteristics change.

## Oxford

The Oxford adapter is separate in `source_adapters.py`. Its native `Cell*/cyc*/C1dc` contract is fixture-tested, but no Oxford result is included until the authorized official MAT file is processed.

These scripts do not bypass provider access terms and do not download files automatically.
