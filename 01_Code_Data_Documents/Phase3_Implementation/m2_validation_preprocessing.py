import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "battery_id",
    "cycle_count",
    "temperature_c",
    "discharge_current_a",
    "avg_voltage_v",
    "min_voltage_v",
    "internal_resistance_ohm",
    "load_power_w",
    "rated_capacity_ah",
    "measured_capacity_ah",
    "backup_time_min",
]

NUMERIC_COLUMNS = [
    c for c in REQUIRED_COLUMNS
    if c != "battery_id"
]

RANGES = {
    "cycle_count": (0, None),
    "temperature_c": (-30, 80),
    "discharge_current_a": (0, None),
    "avg_voltage_v": (2.0, 5.0),
    "min_voltage_v": (1.5, 5.0),
    "internal_resistance_ohm": (0, 2.0),
    "load_power_w": (0, None),
    "rated_capacity_ah": (0, None),
    "measured_capacity_ah": (0, None),
    "backup_time_min": (0, None),
}


def validate_and_clean(df, impute=True, drop_duplicates=True):
    """
    Validate schema, numeric types and physical ranges.

    Returns:
        cleaned_dataframe
        quality_report_dictionary
    """
    out = df.copy()

    report = {
        "rows_input": int(len(out)),
        "missing_required_columns": [],
        "duplicates_removed": 0,
        "duplicate_key": [],
        "numeric_coercion_nulls": {},
        "range_violations": {},
        "imputed_values": {},
    }

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in out.columns
    ]
    report["missing_required_columns"] = missing

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if drop_duplicates:
        before = len(out)
        if "record_id" in out.columns:
            subset = ["record_id"]
        elif "timestamp" in out.columns:
            subset = [
                column
                for column in ["battery_id", "cycle_count", "timestamp"]
                if column in out.columns
            ]
        else:
            subset = [
                column
                for column in ["battery_id", "cycle_count"]
                if column in out.columns
            ]
        report["duplicate_key"] = subset
        out = out.drop_duplicates(
            subset=subset or None,
            keep="first"
        )
        report["duplicates_removed"] = int(before - len(out))

    for col in NUMERIC_COLUMNS:
        before_null = out[col].isna().sum()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        after_null = out[col].isna().sum()
        report["numeric_coercion_nulls"][col] = int(
            max(0, after_null - before_null)
        )

    for col, (lo, hi) in RANGES.items():
        bad = pd.Series(False, index=out.index)
        if lo is not None:
            bad |= out[col] < lo
        if hi is not None:
            bad |= out[col] > hi

        report["range_violations"][col] = int(bad.sum())
        out.loc[bad, col] = np.nan

    if impute:
        for col in NUMERIC_COLUMNS:
            n = int(out[col].isna().sum())
            if n:
                median = out[col].median()
                if pd.isna(median):
                    raise ValueError(f"{col} has no valid values.")
                out[col] = out[col].fillna(median)
            report["imputed_values"][col] = n
    else:
        out = out.dropna(subset=NUMERIC_COLUMNS)

    report["rows_output"] = int(len(out))
    report["fatal_errors"] = 0
    report["valid"] = (
        len(report["missing_required_columns"]) == 0
        and len(out) > 0
    )

    return out.reset_index(drop=True), report




# ============================================================
# STANDALONE REGRESSION CHECK FOR SHARED MODULE M2
# VALIDATION AND PREPROCESSING
# ============================================================

def main():

    print("=" * 70)
    print("SHARED MODULE M2: VALIDATION & PREPROCESSING (PHASE 3 REGRESSION CHECK)")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1 - Create demonstration dataset
    # --------------------------------------------------------

    print("\nSTEP 1: Creating demonstration dataset...")

    demo_data = pd.DataFrame({

        "battery_id": [
            "BAT_01",
            "BAT_01",     # duplicate battery/cycle
            "BAT_02",
            "BAT_03",
            "BAT_04"
        ],

        "cycle_count": [
            10,
            10,           # duplicate
            30,
            50,
            70
        ],

        "temperature_c": [
            25.0,
            25.0,
            np.nan,       # missing value
            120.0,        # invalid temperature
            30.0
        ],

        "discharge_current_a": [
            1.0,
            1.0,
            1.5,
            2.0,
            1.2
        ],

        "avg_voltage_v": [
            3.95,
            3.95,
            3.88,
            3.80,
            3.91
        ],

        "min_voltage_v": [
            3.55,
            3.55,
            3.40,
            3.30,
            3.48
        ],

        "internal_resistance_ohm": [
            0.045,
            0.045,
            0.052,
            0.060,
            -0.010       # invalid resistance
        ],

        "load_power_w": [
            20.0,
            20.0,
            30.0,
            40.0,
            25.0
        ],

        "rated_capacity_ah": [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0
        ],

        "measured_capacity_ah": [
            1.94,
            1.94,
            1.82,
            1.70,
            1.87
        ],

        "backup_time_min": [
            20.7,
            20.7,
            12.7,
            9.5,
            16.5
        ]
    })

    print("Raw records:", len(demo_data))

    print("\nRaw demonstration data:")

    print(
        demo_data.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # STEP 2 - Run validation and preprocessing
    # --------------------------------------------------------

    print("\nSTEP 2: Running validation and preprocessing...")

    cleaned_data, quality_report = validate_and_clean(
        demo_data,
        impute=True,
        drop_duplicates=True
    )

    print("Validation and preprocessing completed.")

    # --------------------------------------------------------
    # STEP 3 - Display quality report
    # --------------------------------------------------------

    print("\nSTEP 3: DATA QUALITY REPORT")

    print(
        "Input rows             :",
        quality_report["rows_input"]
    )

    print(
        "Output rows            :",
        quality_report["rows_output"]
    )

    print(
        "Missing required fields:",
        quality_report["missing_required_columns"]
    )

    print(
        "Duplicates removed     :",
        quality_report["duplicates_removed"]
    )

    print("\nRange violations detected:")

    for column, count in quality_report[
        "range_violations"
    ].items():

        if count > 0:
            print(
                f"  {column}: {count}"
            )

    print("\nValues imputed:")

    for column, count in quality_report[
        "imputed_values"
    ].items():

        if count > 0:
            print(
                f"  {column}: {count}"
            )

    # --------------------------------------------------------
    # STEP 4 - Display cleaned dataset
    # --------------------------------------------------------

    print("\nSTEP 4: CLEANED DATASET")

    print(
        cleaned_data.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # STEP 5 - Check remaining null values
    # --------------------------------------------------------

    remaining_nulls = int(
        cleaned_data[
            NUMERIC_COLUMNS
        ].isna().sum().sum()
    )

    print("\nSTEP 5: FINAL VALIDATION")

    print(
        "Remaining numeric null values:",
        remaining_nulls
    )

    print(
        "Dataset valid                 :",
        quality_report["valid"]
    )

    # --------------------------------------------------------
    # Determine PASS / FAIL
    # --------------------------------------------------------

    if (
        quality_report["valid"]
        and
        remaining_nulls == 0
        and
        quality_report["duplicates_removed"] >= 1
    ):

        status = "PASS"

    else:

        status = "FAIL"

    print("\n" + "=" * 70)

    print(
        "M2 VALIDATION & PREPROCESSING STATUS:",
        status
    )

    print("=" * 70)

    if status == "PASS":

        print("\nM2 successfully demonstrated:")

        print(
            "1. Required-column validation"
        )

        print(
            "2. Duplicate detection and removal"
        )

        print(
            "3. Missing-value detection"
        )

        print(
            "4. Physical/range validation"
        )

        print(
            "5. Median imputation"
        )

        print(
            "6. Clean dataset generation"
        )


if __name__ == "__main__":
    main()
