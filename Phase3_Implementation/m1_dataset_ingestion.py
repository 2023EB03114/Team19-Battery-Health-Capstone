from pathlib import Path
import pandas as pd

DEFAULT_ALIASES = {
    "voltage": "avg_voltage_v",
    "avg_voltage": "avg_voltage_v",
    "minimum_voltage": "min_voltage_v",
    "min_voltage": "min_voltage_v",
    "current": "discharge_current_a",
    "discharge_current": "discharge_current_a",
    "temperature": "temperature_c",
    "temp": "temperature_c",
    "resistance": "internal_resistance_ohm",
    "internal_resistance": "internal_resistance_ohm",
    "load": "load_power_w",
    "load_power": "load_power_w",
    "rated_capacity": "rated_capacity_ah",
    "capacity": "measured_capacity_ah",
    "measured_capacity": "measured_capacity_ah",
    "backup_time": "backup_time_min",
    "runtime": "backup_time_min",
    "cycle": "cycle_count",
    "cell_id": "battery_id",
}


def load_dataset(path, source_name=None, column_aliases=None):
    """Load CSV/XLS/XLSX battery data and standardize column names."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Use CSV, XLS or XLSX."
        )

    aliases = DEFAULT_ALIASES.copy()
    if column_aliases:
        aliases.update(column_aliases)

    normalized = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        normalized[col] = aliases.get(key, key)

    df = df.rename(columns=normalized)

    if "source_dataset" not in df.columns:
        df["source_dataset"] = source_name or path.stem
    if "battery_id" not in df.columns:
        df["battery_id"] = "unknown"

    return df


def standardize_dataframe(df, source_name="in_memory", column_aliases=None):
    """Standardize an already-loaded pandas DataFrame."""
    aliases = DEFAULT_ALIASES.copy()
    if column_aliases:
        aliases.update(column_aliases)

    out = df.copy()
    normalized = {}
    for col in out.columns:
        key = str(col).strip().lower().replace(" ", "_")
        normalized[col] = aliases.get(key, key)

    out = out.rename(columns=normalized)

    if "source_dataset" not in out.columns:
        out["source_dataset"] = source_name
    if "battery_id" not in out.columns:
        out["battery_id"] = "unknown"

    return out


# ============================================================
# STANDALONE REGRESSION CHECK FOR SHARED MODULE M1
# ============================================================

def main():

    print("=" * 70)
    print("SHARED MODULE M1: DATASET INGESTION (PHASE 3 REGRESSION CHECK)")
    print("=" * 70)

    # Small demonstration dataset using common/raw column names
    demo_data = pd.DataFrame({
        "cell_id": ["CELL_01", "CELL_01", "CELL_02"],
        "cycle": [10, 50, 20],
        "temperature": [25.0, 28.0, 24.5],
        "current": [1.0, 1.5, 0.8],
        "voltage": [3.95, 3.88, 4.00],
        "minimum_voltage": [3.55, 3.42, 3.61],
        "resistance": [0.045, 0.052, 0.041],
        "load": [20.0, 30.0, 18.0],
        "rated_capacity": [2.0, 2.0, 2.1],
        "capacity": [1.94, 1.82, 2.03],
        "backup_time": [20.7, 12.7, 24.4]
    })

    print("\nSTEP 1: Raw demonstration data created.")
    print("Records:", len(demo_data))

    print("\nSTEP 2: Standardizing dataset columns...")

    standardized_data = standardize_dataframe(
        demo_data,
        source_name="Phase2_M1_Demo"
    )

    print("Dataset standardized successfully.")

    print("\nSTEP 3: Standardized column names:")

    for i, column in enumerate(
        standardized_data.columns,
        start=1
    ):
        print(f"{i:02d}. {column}")

    print("\nSTEP 4: Standardized records:")

    print(
        standardized_data.to_string(
            index=False
        )
    )

    required_columns = [
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
        "source_dataset"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in standardized_data.columns
    ]

    print("\nSTEP 5: Schema verification")
    print("Required columns :", len(required_columns))
    print("Missing columns  :", missing_columns)

    if len(missing_columns) == 0:
        status = "PASS"
    else:
        status = "FAIL"

    print("\n" + "=" * 70)
    print("M1 DATASET INGESTION STATUS:", status)
    print("=" * 70)


if __name__ == "__main__":
    main()
