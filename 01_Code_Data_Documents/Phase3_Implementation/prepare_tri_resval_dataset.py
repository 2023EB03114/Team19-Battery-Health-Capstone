"""Prepare compact TRI rate-validation records from the official raw archive.

The official ``ResValData.zip`` archive contains Maccor text files for 94
physical 21700 cells.  Each cell was tested at eight constant-current rates
from C/80 through 2C.  This script streams the archive (no 2+ GB extraction is
required), stops each discharge at the first 2.7 V cutoff, and writes one
auditable common-schema row per complete cell/rate test.

The raw archive is not redistributed with the submission.  Reproduction:

    python prepare_tri_resval_dataset.py \
        --archive C:\\path\\to\\ResValData.zip \
        --output data\\processed_public\\tri_resval_cycle_summary.csv

Source: https://data.matr.io/10/
License: CC BY-NC 4.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_PAGE = "https://data.matr.io/10/"
SOURCE_ARCHIVE_URL = "https://data.matr.io/10/ResValData.zip"
SOURCE_DOI = "10.1016/j.joule.2025.102010"
SOURCE_LICENSE = "CC BY-NC 4.0"
SOURCE_DATASET = "TRI_ResVal_94Cell_MultiRate"
NOMINAL_CAPACITY_AH = 4.84
VOLTAGE_CUTOFF_V = 2.7005

# Cycle indices and names follow the authors' released func_resval_data.py.
RATE_CYCLES = {
    1: ("C/80", 1.0 / 80.0),
    3: ("C/40", 1.0 / 40.0),
    4: ("0.05 A", 0.05 / NOMINAL_CAPACITY_AH),
    5: ("C/10", 1.0 / 10.0),
    6: ("C/7", 1.0 / 7.0),
    7: ("C/5", 1.0 / 5.0),
    8: ("1C", 1.0),
    9: ("2C", 2.0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _barcode(metadata: str, member: str) -> str:
    match = re.search(r"Comment/Barcode:\s*([^\t\r\n ]+)", metadata)
    if match:
        return match.group(1).strip()
    stem = Path(member).name
    parts = stem.split("_")
    return parts[2].split(".")[0] if len(parts) >= 3 else Path(member).stem


def _first_cutoff_segment(rows: list[dict[str, str]]) -> dict[str, float] | None:
    """Return the main discharge segment ending at the first 2.7 V cutoff."""
    values = []
    for row in rows:
        if str(row.get("State", "")).strip().upper() != "D":
            continue
        item = {
            "test_s": _number(row, "Test (Sec)"),
            "step_s": _number(row, "Step (Sec)"),
            "capacity_ah": _number(row, "Amp-hr"),
            "energy_wh": _number(row, "Watt-hr"),
            "current_a": _number(row, "Amps"),
            "voltage_v": _number(row, "Volts"),
        }
        if all(np.isfinite(list(item.values()))):
            values.append(item)
    if len(values) < 5:
        return None

    start = next(
        (
            index
            for index, item in enumerate(values)
            if item["current_a"] < -0.01 and item["voltage_v"] > VOLTAGE_CUTOFF_V
        ),
        None,
    )
    if start is None:
        return None
    cutoff = next(
        (
            index
            for index in range(start + 1, len(values))
            if values[index]["voltage_v"] <= VOLTAGE_CUTOFF_V
            and values[index]["capacity_ah"] - values[start]["capacity_ah"] > 0.25
        ),
        None,
    )
    if cutoff is None:
        return None

    segment = values[start : cutoff + 1]
    duration_s = segment[-1]["test_s"] - segment[0]["test_s"]
    capacity_ah = segment[-1]["capacity_ah"] - segment[0]["capacity_ah"]
    energy_wh = segment[-1]["energy_wh"] - segment[0]["energy_wh"]
    if duration_s <= 0 or capacity_ah <= 0 or energy_wh <= 0:
        return None

    duration_h = duration_s / 3600.0
    current_a = capacity_ah / duration_h
    avg_voltage_v = energy_wh / capacity_ah
    load_power_w = energy_wh / duration_h
    loaded = [
        item["voltage_v"]
        for item in segment
        if item["test_s"] - segment[0]["test_s"] <= 10.0
    ]
    if not loaded:
        loaded = [segment[0]["voltage_v"]]
    return {
        "start_test_s": segment[0]["test_s"],
        "duration_s": duration_s,
        "capacity_ah": capacity_ah,
        "energy_wh": energy_wh,
        "current_a": current_a,
        "avg_voltage_v": avg_voltage_v,
        "min_voltage_v": min(item["voltage_v"] for item in segment),
        "initial_loaded_voltage_v": float(np.median(loaded)),
        "load_power_w": load_power_w,
        "sample_count": len(segment),
    }


def _parse_member(archive: zipfile.ZipFile, member: str) -> list[dict]:
    with archive.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        metadata = next(text).strip()
        reader = csv.DictReader(text, delimiter="\t")
        by_cycle: dict[int, list[dict[str, str]]] = defaultdict(list)
        preceding_voltage: dict[int, float] = {}
        last_voltage_by_cycle: dict[int, float] = {}
        for row in reader:
            try:
                cycle = int(float(row["Cyc#"]))
            except (KeyError, TypeError, ValueError):
                continue
            voltage = _number(row, "Volts")
            current = _number(row, "Amps")
            if np.isfinite(voltage):
                last_voltage_by_cycle[cycle] = voltage
            if cycle in RATE_CYCLES:
                by_cycle[cycle].append(row)
                if np.isfinite(voltage) and np.isfinite(current) and current > 0:
                    preceding_voltage[cycle] = voltage

    barcode = _barcode(metadata, member)
    rows = []
    for rate_index, cycle in enumerate(RATE_CYCLES, start=1):
        segment = _first_cutoff_segment(by_cycle.get(cycle, []))
        if segment is None:
            continue
        rate_label, nominal_c_rate = RATE_CYCLES[cycle]
        ocv = preceding_voltage.get(cycle, float("nan"))
        resistance = (
            (ocv - segment["initial_loaded_voltage_v"]) / segment["current_a"]
            if np.isfinite(ocv) and segment["current_a"] > 0
            else float("nan")
        )
        if not np.isfinite(resistance) or resistance <= 0 or resistance > 1.0:
            resistance = float("nan")
        rows.append(
            {
                "record_id": f"TRI_{barcode}_{rate_index:02d}",
                "source_dataset": SOURCE_DATASET,
                "source_file": Path(member).name,
                "source_cycle_index": cycle,
                "source_rate_index": rate_index,
                "source_rate_label": rate_label,
                "source_nominal_c_rate": nominal_c_rate,
                "battery_id": barcode,
                # The raw validation file does not report the cell's prior aging
                # cycle count.  Zero is a declared unavailable-value sentinel,
                # not a claim that these degraded cells are fresh.
                "cycle_count": 0,
                "temperature_c": 25.0,
                "discharge_current_a": segment["current_a"],
                "avg_voltage_v": segment["avg_voltage_v"],
                "min_voltage_v": segment["min_voltage_v"],
                "internal_resistance_ohm": resistance,
                "load_power_w": segment["load_power_w"],
                "rated_capacity_ah": NOMINAL_CAPACITY_AH,
                "measured_capacity_ah": segment["capacity_ah"],
                "backup_time_min": segment["duration_s"] / 60.0,
                "efficiency_fraction": 1.0,
                "measured_energy_wh": segment["energy_wh"],
                "source_sample_count": segment["sample_count"],
                "temperature_provenance": (
                    "25 C schema placeholder; temperature is not present in the raw file "
                    "and the feature is constant/excluded from interpretation"
                ),
                "cycle_count_provenance": (
                    "aging cycle count unavailable in ResVal file; zero sentinel"
                ),
                "resistance_provenance": (
                    "charge-end voltage minus initial loaded voltage, divided by mean current"
                ),
                "target_provenance": "first complete discharge to 2.7 V cutoff",
            }
        )
    return rows


def process_archive(path: Path) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ResValData/")
            and not name.endswith("/")
            and not Path(name).name.startswith("._")
        )
        for number, member in enumerate(members, start=1):
            print(f"[{number:02d}/{len(members):02d}] {Path(member).name}")
            rows.extend(_parse_member(archive, member))

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No complete TRI rate-validation discharges were extracted.")

    # Use a within-cell median only when the low-rate start transient is too
    # small/noisy to yield a positive apparent resistance.  The flag remains
    # in the compact output so this is auditable.
    missing_resistance = frame["internal_resistance_ohm"].isna()
    frame["resistance_imputed"] = missing_resistance
    cell_median = frame.groupby("battery_id")["internal_resistance_ohm"].transform("median")
    global_median = float(frame["internal_resistance_ohm"].median())
    frame["internal_resistance_ohm"] = (
        frame["internal_resistance_ohm"].fillna(cell_median).fillna(global_median)
    )

    frame = frame.sort_values(["battery_id", "source_rate_index"]).reset_index(drop=True)
    duplicate_count = int(frame["record_id"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"Duplicate TRI record_id values: {duplicate_count}")

    rate_counts = frame.groupby("battery_id")["source_rate_index"].nunique()
    quality = {
        "archive_file": path.name,
        "archive_sha256": sha256(path),
        "archive_size_bytes": path.stat().st_size,
        "physical_cells": int(frame["battery_id"].nunique()),
        "records": int(len(frame)),
        "expected_rates_per_cell": len(RATE_CYCLES),
        "complete_eight_rate_cells": int((rate_counts == len(RATE_CYCLES)).sum()),
        "incomplete_cells": {
            str(cell): int(count)
            for cell, count in rate_counts[rate_counts != len(RATE_CYCLES)].items()
        },
        "known_source_exception": (
            "Cell 0000ED has six low-rate records and no usable 1C/2C record; "
            "the authors' released func_resval_data.py also skips this cell."
        ),
        "resistance_values_imputed": int(missing_resistance.sum()),
        "temperature_values_measured": 0,
        "temperature_placeholder_values": int(len(frame)),
        "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
        "cutoff_voltage_v": 2.7,
        "source_page": SOURCE_PAGE,
        "source_archive_url": SOURCE_ARCHIVE_URL,
        "paper_doi": SOURCE_DOI,
        "license": SOURCE_LICENSE,
    }
    return frame, quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, quality = process_archive(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    quality_path = args.quality_json or args.output.with_suffix(".quality.json")
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(json.dumps(quality, indent=2))
    if quality["physical_cells"] != 94:
        raise SystemExit("Expected 94 physical cells in official TRI archive.")
    # The authors' released analysis code explicitly skips cell 0000ED.  Its
    # raw file contains the six low-rate tests but no usable 1C or 2C record,
    # so retaining 750 observed records is more truthful than fabricating two.
    if quality["records"] != 750 or quality["incomplete_cells"] != {"0000ED": 6}:
        raise SystemExit(
            "Expected 750 records: 93 complete cells plus six low-rate records "
            "for the authors' known 0000ED exception."
        )


if __name__ == "__main__":
    main()
