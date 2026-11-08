"""Pure prediction and validation functions used by the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from m3_feature_engineering import engineer_features
from m4_formula_baseline import formula_backup_time_minutes
from m9_decision_support import recommend_action


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

INPUT_FIELDS = [
    "cycle_count",
    "temperature_c",
    "discharge_current_a",
    "avg_voltage_v",
    "min_voltage_v",
    "internal_resistance_ohm",
    "load_power_w",
    "rated_capacity_ah",
]

RANGES = {
    "cycle_count": (0.0, 10000.0),
    "temperature_c": (-30.0, 80.0),
    "discharge_current_a": (0.01, 1000.0),
    "avg_voltage_v": (2.0, 5.0),
    "min_voltage_v": (1.5, 5.0),
    "internal_resistance_ohm": (0.00001, 2.0),
    "load_power_w": (0.01, 100000.0),
    "rated_capacity_ah": (0.01, 10000.0),
}


def load_dashboard_artifacts(output_dir: Path = OUTPUT_DIR):
    metadata_path = output_dir / "dashboard_model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    capacity_model = joblib.load(output_dir / metadata["capacity_model"])
    runtime_model = joblib.load(output_dir / metadata["runtime_model"])
    return capacity_model, runtime_model, metadata


def validate_condition(values: Mapping) -> dict[str, float]:
    missing = [field for field in INPUT_FIELDS if field not in values]
    if missing:
        raise ValueError(f"Required input fields missing: {missing}")
    clean: dict[str, float] = {}
    for field in INPUT_FIELDS:
        try:
            value = float(values[field])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be numeric.") from error
        if not np.isfinite(value):
            raise ValueError(f"{field} must be finite.")
        lower, upper = RANGES[field]
        if not lower <= value <= upper:
            raise ValueError(f"{field} must be between {lower} and {upper}.")
        clean[field] = value
    if clean["min_voltage_v"] > clean["avg_voltage_v"]:
        raise ValueError("min_voltage_v cannot exceed avg_voltage_v.")
    return clean


def predict_condition(
    values: Mapping,
    efficiency_fraction: float = 0.90,
    required_runtime_min: float = 45.0,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    clean = validate_condition(values)
    efficiency = float(efficiency_fraction)
    required_runtime = float(required_runtime_min)
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency_fraction must be in (0, 1].")
    if required_runtime <= 0:
        raise ValueError("required_runtime_min must be positive.")
    capacity_model, runtime_model, metadata = load_dashboard_artifacts(output_dir)
    frame = pd.DataFrame([clean])
    frame["battery_id"] = "DASHBOARD_NEW_CONDITION"
    frame["measured_capacity_ah"] = np.nan
    frame["backup_time_min"] = np.nan
    frame = engineer_features(frame)
    predicted_capacity = float(capacity_model.predict_capacity(frame)[0])
    predicted_soh = predicted_capacity / clean["rated_capacity_ah"] * 100.0
    frame["predicted_soh_percent"] = predicted_soh
    ml_runtime = float(runtime_model.predict(frame)[0])
    formula_runtime = float(
        formula_backup_time_minutes(
            clean["avg_voltage_v"],
            predicted_capacity,
            clean["load_power_w"],
            efficiency=efficiency,
            usable_fraction=1.0,
        )
    )
    power_from_vi = clean["avg_voltage_v"] * clean["discharge_current_a"]
    power_consistency_ratio = clean["load_power_w"] / power_from_vi
    input_warnings = []
    if not 0.5 <= power_consistency_ratio <= 2.0:
        input_warnings.append(
            "Load power differs substantially from average voltage x discharge "
            "current. Confirm pack/cell scaling and converter assumptions."
        )
    decision_runtime = min(ml_runtime, formula_runtime)
    decision = recommend_action(
        predicted_soh,
        decision_runtime,
        required_runtime_min=required_runtime,
        data_quality_ok=True,
    )
    return {
        "predicted_capacity_ah": predicted_capacity,
        "predicted_soh_percent": predicted_soh,
        "formula_backup_time_min": formula_runtime,
        "ml_backup_time_min": ml_runtime,
        "decision_runtime_min": decision_runtime,
        "decision_runtime_basis": "minimum of formula and ML prototype estimates",
        "power_consistency_ratio": power_consistency_ratio,
        "input_warnings": input_warnings,
        "required_runtime_min": required_runtime,
        "decision_level": decision["level"],
        "decision_message": decision["message"],
        "model_domain": metadata["default_model_domain"],
        "warning": metadata["warning"],
    }


def predict_uploaded_frame(
    frame: pd.DataFrame,
    efficiency_fraction: float = 0.90,
    required_runtime_min: float = 45.0,
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("Uploaded table is empty.")
    records = []
    for index, row in frame.iterrows():
        prediction = predict_condition(
            row.to_dict(),
            efficiency_fraction=efficiency_fraction,
            required_runtime_min=required_runtime_min,
            output_dir=output_dir,
        )
        records.append({"input_row": int(index), **prediction})
    return pd.DataFrame(records)


def smoke_prediction(output_dir: Path = OUTPUT_DIR) -> dict:
    return predict_condition(
        {
            "cycle_count": 120,
            "temperature_c": 25.0,
            "discharge_current_a": 1.0,
            "avg_voltage_v": 3.85,
            "min_voltage_v": 3.45,
            "internal_resistance_ohm": 0.055,
            "load_power_w": 4.0,
            "rated_capacity_ah": 2.0,
        },
        output_dir=output_dir,
    )
