"""Create cycle-level NASA and CALCE Phase-3 validation tables.

The raw archives are intentionally not redistributed. This script expects the
official source files in the paths used by the Phase-3 preparation workspace
and writes compact, auditable CSV summaries suitable for grouped evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


NASA_SOURCE_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
)
CALCE_SOURCE_ROOT = "https://web.calce.umd.edu/batteries/data"
OXFORD_SOURCE_PAGE = (
    "https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac"
)
OXFORD_DOI = "10.5287/bodleian:KO2kdmYGg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def process_nasa(nasa_dir: Path) -> pd.DataFrame:
    """Extract discharge-cycle summaries from NASA B0005/B0006/B0007/B0018."""
    rows: list[dict] = []
    for mat_path in sorted(nasa_dir.glob("B*.mat")):
        battery_id = mat_path.stem
        payload = loadmat(mat_path, simplify_cells=True)[battery_id]
        cycles = payload["cycle"]
        latest_resistance = None
        discharge_number = 0

        for source_index, cycle in enumerate(cycles):
            cycle_type = str(cycle["type"]).strip().lower()
            data = cycle["data"]
            if cycle_type == "impedance":
                electrolyte = float(data.get("Re", np.nan))
                transfer = float(data.get("Rct", np.nan))
                candidate = electrolyte + transfer
                if np.isfinite(candidate) and candidate > 0:
                    latest_resistance = candidate
                continue
            if cycle_type != "discharge":
                continue

            discharge_number += 1
            voltage = _array(data["Voltage_measured"])
            current = np.abs(_array(data["Current_measured"]))
            temperature = _array(data["Temperature_measured"])
            time_seconds = _array(data["Time"])
            capacity = float(data["Capacity"])
            if (
                latest_resistance is None
                or len(voltage) < 5
                or len(time_seconds) < 5
                or not np.isfinite(capacity)
            ):
                # A preceding impedance reading is required so resistance is
                # genuinely available at inference time rather than backfilled
                # from a future measurement.
                continue

            positive = current > 0.05
            if positive.sum() < 5:
                continue
            voltage = voltage[positive]
            current = current[positive]
            temperature = temperature[positive]
            time_seconds = time_seconds[positive]
            duration_min = float((time_seconds.max() - time_seconds.min()) / 60.0)
            load_power = float(np.mean(voltage * current))

            rows.append(
                {
                    "record_id": f"NASA_{battery_id}_{discharge_number:04d}",
                    "source_dataset": "NASA_PCoE_Battery_Aging_ARC_FY08Q4",
                    "source_file": mat_path.name,
                    "source_cycle_index": int(source_index),
                    "battery_id": battery_id,
                    "cycle_count": discharge_number,
                    "temperature_c": float(np.mean(temperature)),
                    "discharge_current_a": float(np.mean(current)),
                    "avg_voltage_v": float(np.mean(voltage)),
                    "min_voltage_v": float(np.min(voltage)),
                    "internal_resistance_ohm": float(latest_resistance),
                    "load_power_w": load_power,
                    "rated_capacity_ah": 2.0,
                    "measured_capacity_ah": capacity,
                    "backup_time_min": duration_min,
                    "efficiency_fraction": 1.0,
                    "temperature_provenance": "measured cell temperature",
                    "resistance_provenance": "latest preceding EIS Re + Rct",
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No NASA discharge records were extracted.")
    return result.sort_values(["battery_id", "cycle_count"]).reset_index(drop=True)


def _read_calce_channel(path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path, engine="openpyxl")
    sheets = [name for name in workbook.sheet_names if name.lower().startswith("channel")]
    if not sheets:
        return pd.DataFrame()
    return pd.read_excel(workbook, sheet_name=sheets[0])


def _calce_cycle_rows(path: Path, battery_id: str) -> list[dict]:
    frame = _read_calce_channel(path)
    required = {
        "Test_Time(s)",
        "Date_Time",
        "Step_Time(s)",
        "Cycle_Index",
        "Current(A)",
        "Voltage(V)",
        "Discharge_Capacity(Ah)",
    }
    if frame.empty or not required.issubset(frame.columns):
        return []

    frame = frame.copy()
    for column in required - {"Date_Time"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Date_Time"] = pd.to_datetime(frame["Date_Time"], errors="coerce")

    records: list[dict] = []
    for local_cycle, group in frame.groupby("Cycle_Index", sort=True):
        group = group.sort_values("Test_Time(s)")
        discharge = group.loc[group["Current(A)"] < -0.5].copy()
        if len(discharge) < 5:
            continue

        current = discharge["Current(A)"].abs()
        voltage = discharge["Voltage(V)"]
        valid = current.notna() & voltage.notna() & (current > 0.05)
        discharge = discharge.loc[valid]
        current = current.loc[valid]
        voltage = voltage.loc[valid]
        if len(discharge) < 5:
            continue

        capacity = float(
            group["Discharge_Capacity(Ah)"].max()
            - group["Discharge_Capacity(Ah)"].min()
        )
        duration_seconds = float(
            discharge["Test_Time(s)"].max() - discharge["Test_Time(s)"].min()
        )
        sample_spacing = discharge["Test_Time(s)"].diff().median()
        if np.isfinite(sample_spacing) and sample_spacing > 0:
            duration_seconds += float(sample_spacing)

        first_discharge_index = discharge.index.min()
        prior = group.loc[group.index < first_discharge_index]
        prior_rest = prior.loc[prior["Current(A)"].abs() < 0.01]
        open_circuit_voltage = (
            float(prior_rest["Voltage(V)"].dropna().iloc[-1])
            if not prior_rest["Voltage(V)"].dropna().empty
            else np.nan
        )
        initial_load_voltage = float(voltage.iloc[0])
        initial_current = float(current.iloc[0])
        inferred_resistance = (
            (open_circuit_voltage - initial_load_voltage) / initial_current
            if np.isfinite(open_circuit_voltage) and initial_current > 0
            else np.nan
        )

        start_time = group["Date_Time"].dropna().min()
        if (
            not np.isfinite(capacity)
            or not 0.45 <= capacity <= 1.30
            or not np.isfinite(duration_seconds)
            or duration_seconds <= 0
            or not np.isfinite(inferred_resistance)
            or not 0.001 <= inferred_resistance <= 1.0
        ):
            continue

        records.append(
            {
                "source_dataset": "CALCE_CS2_1C",
                "source_file": path.name,
                "source_cycle_index": int(local_cycle),
                "source_start_time": start_time,
                "battery_id": battery_id,
                "temperature_c": 25.0,
                "discharge_current_a": float(current.mean()),
                "avg_voltage_v": float(voltage.mean()),
                "min_voltage_v": float(voltage.min()),
                "internal_resistance_ohm": float(inferred_resistance),
                "load_power_w": float(np.mean(voltage.to_numpy() * current.to_numpy())),
                "rated_capacity_ah": 1.1,
                "measured_capacity_ah": capacity,
                "backup_time_min": duration_seconds / 60.0,
                "efficiency_fraction": 1.0,
                "temperature_provenance": "25 C dataset-level test assumption",
                "resistance_provenance": "pre-load voltage sag / initial current",
            }
        )
    return records


def process_calce(calce_dir: Path) -> pd.DataFrame:
    """Extract full-discharge cycle summaries from CALCE CS2_35-CS2_38."""
    rows: list[dict] = []
    for battery_dir in sorted(calce_dir.glob("CS2_*")):
        battery_id = battery_dir.name
        for path in sorted(battery_dir.rglob("*.xlsx")):
            rows.extend(_calce_cycle_rows(path, battery_id))

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No CALCE discharge records were extracted.")
    result = result.sort_values(
        ["battery_id", "source_start_time", "source_file", "source_cycle_index"]
    ).reset_index(drop=True)
    result["cycle_count"] = result.groupby("battery_id").cumcount() + 1
    result["record_id"] = result.apply(
        lambda row: f"CALCE_{row['battery_id']}_{int(row['cycle_count']):04d}",
        axis=1,
    )
    return result[
        [
            "record_id",
            "source_dataset",
            "source_file",
            "source_cycle_index",
            "source_start_time",
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
            "efficiency_fraction",
            "temperature_provenance",
            "resistance_provenance",
        ]
    ]


def write_provenance(
    output_path: Path,
    nasa_archive: Path,
    nasa_dir: Path,
    calce_raw_dir: Path,
) -> None:
    existing = {}
    if output_path.is_file():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    payload = {
        "NASA": {
            "official_source_url": NASA_SOURCE_URL,
            "archive_file": nasa_archive.name,
            "archive_sha256": sha256(nasa_archive),
            "mat_files": {
                path.name: sha256(path) for path in sorted(nasa_dir.glob("B*.mat"))
            },
        },
        "CALCE": {
            "official_source_page": "https://calce.umd.edu/battery-data",
            "archives": {
                path.name: {
                    "url": f"{CALCE_SOURCE_ROOT}/{path.name}",
                    "sha256": sha256(path),
                }
                for path in sorted(calce_raw_dir.glob("CS2_*.zip"))
            },
        },
        "Oxford": {
            "official_source_page": OXFORD_SOURCE_PAGE,
            "doi": OXFORD_DOI,
            "quantitative_evaluation": False,
            "status": (
                "Native MAT adapter contract implemented and fixture-tested; "
                "official source file not processed in this package."
            ),
        },
        "redistribution_note": (
            "Raw NASA, CALCE, TRI, and Oxford source archives are not included. "
            "Only compact derived cycle-level summaries, quality metadata, "
            "official URLs, and provenance hashes are distributed."
        ),
    }
    if "TRI" in existing:
        payload["TRI"] = existing["TRI"]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nasa-dir", type=Path, required=True)
    parser.add_argument("--calce-dir", type=Path, required=True)
    parser.add_argument("--nasa-archive", type=Path, required=True)
    parser.add_argument("--calce-raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nasa = process_nasa(args.nasa_dir)
    nasa.to_csv(args.output_dir / "nasa_cycle_summary.csv", index=False)
    print("NASA rows by battery:")
    print(nasa.groupby("battery_id").size().to_string())

    calce = process_calce(args.calce_dir)
    calce.to_csv(args.output_dir / "calce_cycle_summary.csv", index=False)
    print("CALCE rows by battery:")
    print(calce.groupby("battery_id").size().to_string())

    write_provenance(
        args.output_dir / "public_data_provenance.json",
        args.nasa_archive,
        args.nasa_dir,
        args.calce_raw_dir,
    )


if __name__ == "__main__":
    main()
