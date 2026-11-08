"""Dataset adapters and contracts for Phase-3 public battery sources.

NASA and CALCE extraction is implemented in ``prepare_public_datasets.py``;
TRI's 94-cell rate-validation archive is streamed by
``prepare_tri_resval_dataset.py``.  This module loads their compact derived
tables and defines the Oxford MAT contract without pretending that an
unavailable source file was validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.io import loadmat


COMMON_COLUMNS = [
    "record_id",
    "source_dataset",
    "source_file",
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
]


def require_common_columns(frame: pd.DataFrame, allow_missing_resistance: bool = False) -> None:
    missing = [column for column in COMMON_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Common-schema columns missing: {missing}")
    if frame["record_id"].duplicated().any():
        raise ValueError("record_id must be unique.")
    numeric = [
        column
        for column in COMMON_COLUMNS
        if column
        not in {"record_id", "source_dataset", "source_file", "battery_id"}
    ]
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() and not (
            allow_missing_resistance and column == "internal_resistance_ohm"
        ):
            raise ValueError(f"{column} contains missing or non-numeric values.")


def load_processed_public_csv(path: Path, expected_source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require_common_columns(frame)
    observed = set(frame["source_dataset"].astype(str))
    if observed != {expected_source}:
        raise ValueError(
            f"Expected source_dataset={expected_source!r}; observed {sorted(observed)}"
        )
    return frame


def load_nasa_processed(path: Path) -> pd.DataFrame:
    return load_processed_public_csv(path, "NASA_PCoE_Battery_Aging_ARC_FY08Q4")


def load_calce_processed(path: Path) -> pd.DataFrame:
    return load_processed_public_csv(path, "CALCE_CS2_1C")


def load_tri_processed(path: Path) -> pd.DataFrame:
    return load_processed_public_csv(path, "TRI_ResVal_94Cell_MultiRate")


def _as_vector(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _oxford_records_from_mapping(
    mapping: Mapping,
    source_file: str,
    rated_capacity_ah: float = 0.740,
) -> pd.DataFrame:
    """Convert a simplified Oxford Cell/cyc/C1dc mapping to common rows.

    ``scipy.io.loadmat(..., simplify_cells=True)`` produces the mapping shape
    expected here. The Oxford characterization file does not directly provide
    internal resistance, so that field remains missing and must be excluded or
    handled explicitly by an Oxford-specific feature contract.
    """
    rows: list[dict] = []
    cell_names = sorted(name for name in mapping if str(name).lower().startswith("cell"))
    for cell_name in cell_names:
        cell = mapping[cell_name]
        if not isinstance(cell, Mapping):
            continue
        cycle_names = sorted(name for name in cell if str(name).lower().startswith("cyc"))
        for cycle_name in cycle_names:
            cycle = cell[cycle_name]
            if not isinstance(cycle, Mapping) or "C1dc" not in cycle:
                continue
            discharge = cycle["C1dc"]
            if not isinstance(discharge, Mapping):
                continue
            time_seconds = _as_vector(discharge["t"])
            voltage = _as_vector(discharge["v"])
            charge_mah = _as_vector(discharge["q"])
            temperature = _as_vector(discharge["T"])
            length = min(len(time_seconds), len(voltage), len(charge_mah), len(temperature))
            if length < 5:
                continue
            time_seconds = time_seconds[:length]
            voltage = voltage[:length]
            charge_mah = charge_mah[:length]
            temperature = temperature[:length]
            capacity_ah = abs(float(charge_mah[-1] - charge_mah[0])) / 1000.0
            current_a = 0.740
            cycle_digits = "".join(character for character in str(cycle_name) if character.isdigit())
            cycle_count = int(cycle_digits or len(rows))
            rows.append(
                {
                    "record_id": f"OXFORD_{cell_name}_{cycle_count:04d}",
                    "source_dataset": "Oxford_Battery_Degradation_Dataset_1",
                    "source_file": source_file,
                    "battery_id": str(cell_name),
                    "cycle_count": cycle_count,
                    "temperature_c": float(np.mean(temperature)),
                    "discharge_current_a": current_a,
                    "avg_voltage_v": float(np.mean(voltage)),
                    "min_voltage_v": float(np.min(voltage)),
                    "internal_resistance_ohm": np.nan,
                    "load_power_w": float(np.mean(voltage) * current_a),
                    "rated_capacity_ah": rated_capacity_ah,
                    "measured_capacity_ah": capacity_ah,
                    "backup_time_min": float((time_seconds.max() - time_seconds.min()) / 60.0),
                    "efficiency_fraction": 1.0,
                    "temperature_provenance": "measured cell temperature",
                    "resistance_provenance": "not supplied by characterization record",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No Oxford C1dc characterization records were extracted.")
    require_common_columns(result, allow_missing_resistance=True)
    return result.sort_values(["battery_id", "cycle_count"]).reset_index(drop=True)


def load_oxford_characterization_mat(path: Path) -> pd.DataFrame:
    """Load Oxford Dataset 1 when the official 262 MB MAT file is available."""
    payload = loadmat(path, simplify_cells=True)
    content = {
        key: value
        for key, value in payload.items()
        if not str(key).startswith("__")
    }
    return _oxford_records_from_mapping(content, Path(path).name)


def oxford_contract_fixture() -> Mapping:
    """Return a tiny structure-only fixture for automated adapter contract tests."""
    return {
        "Cell1": {
            "cyc0000": {
                "C1dc": {
                    "t": np.array([0.0, 60.0, 120.0, 180.0, 240.0]),
                    "v": np.array([4.10, 3.95, 3.80, 3.65, 3.50]),
                    "q": np.array([0.0, 180.0, 360.0, 540.0, 720.0]),
                    "T": np.array([40.0, 40.2, 40.4, 40.5, 40.6]),
                }
            }
        }
    }
