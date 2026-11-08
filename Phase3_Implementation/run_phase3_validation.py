"""Run the complete Phase-3 implementation and validation workflow.

Evidence domains are deliberately separated:

* CSGB: controlled 720-row synthetic benchmark retained for regression and
  software-continuity checks against Phase 2.
* NASA: real laboratory discharge cycles, grouped by physical battery.
* CALCE: real laboratory discharge cycles, grouped by physical cell.
* TRI: real multi-rate validation discharges from 94 aged EV cells, grouped by
  physical cell; temperature and prior aging-cycle count are not measured in
  these raw files and are explicitly flagged as unavailable/placeholders.

The script trains M5/M6 models, evaluates the M4 comparator, generates M7/M8
evidence, persists and reloads models, exercises M9, and writes record-aligned
CSV/JSON/PNG outputs. Oxford Dataset 1 has a native adapter contract in
``source_adapters.py`` but is not included in quantitative claims unless the
official source MAT file is actually supplied and processed.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import confusion_matrix

try:
    import shap
except Exception as shap_error:  # pragma: no cover - validated environment has SHAP
    shap = None
    SHAP_IMPORT_ERROR = str(shap_error)
else:
    SHAP_IMPORT_ERROR = None

from m1_dataset_ingestion import standardize_dataframe
from m2_validation_preprocessing import validate_and_clean
from m3_feature_engineering import (
    HEALTH_FEATURES,
    RUNTIME_FEATURES,
    engineer_features,
    get_health_matrix,
    get_runtime_matrix,
)
from m4_formula_baseline import formula_backup_time_minutes
from m5_capacity_soh import CapacitySOHModel
from m6_dynamic_backup_time import BackupTimeModel
from m7_evaluation_comparison import exact_match_accuracy, regression_metrics
from m8_explainability import (
    built_in_feature_importance,
    permutation_feature_importance,
)
from m9_decision_support import (
    classification_report_table,
    confusion_count_table,
    recommend_action,
)
from source_adapters import (
    load_calce_processed,
    load_nasa_processed,
    load_tri_processed,
)


RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"
MODEL_DIR = OUTPUT_DIR / "models"

DATASET_CONFIG = {
    "CSGB": {
        "label": "CSGB controlled benchmark",
        "domain": "controlled synthetic",
        "train_groups": None,
        "test_groups": None,
        "required_runtime_min": None,
        "formula_efficiency": "column",
    },
    "NASA": {
        "label": "NASA PCoE real laboratory data",
        "domain": "real experimental",
        "train_groups": ["B0005", "B0006"],
        "test_groups": ["B0007", "B0018"],
        "required_runtime_min": 45.0,
        "formula_efficiency": 1.0,
    },
    "CALCE": {
        "label": "CALCE CS2 real laboratory data",
        "domain": "real experimental",
        "train_groups": ["CS2_35", "CS2_36"],
        "test_groups": ["CS2_37", "CS2_38"],
        "required_runtime_min": 40.0,
        "formula_efficiency": 1.0,
    },
    "TRI": {
        "label": "TRI 94-cell real multi-rate validation data",
        "domain": "real experimental",
        "train_groups": None,
        "test_groups": None,
        "required_runtime_min": 45.0,
        "formula_efficiency": 1.0,
        "deterministic_test_fraction": 0.25,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def health_class_from_soh(soh_percent: float) -> str:
    value = float(soh_percent)
    if value >= 90.0:
        return "Healthy (>=90%)"
    if value >= 80.0:
        return "Monitor (80-89%)"
    if value >= 65.0:
        return "Inspect/Reduce Load (65-79%)"
    return "Replace Planning (<65%)"


def recommendation_label(level: str) -> str:
    mapping = {
        "normal": "No action needed",
        "monitor": "Monitor more frequently",
        "inspect": "Reduce load or schedule inspection",
        "reduce_load": "Reduce load or schedule inspection",
        "replace": "Plan replacement",
        "inspect_data": "Inspect data quality",
    }
    return mapping[level]


def load_csgb() -> tuple[pd.DataFrame, dict]:
    path = DATA_DIR / "Battery_Base_Data_Benchmark_v1.0.xlsx"
    benchmark = pd.read_excel(path, sheet_name="Benchmark Data")
    required = {
        "record_id",
        "battery_id",
        "cycle_number",
        "rated_capacity_ah",
        "ambient_temp_c",
        "discharge_current_a",
        "internal_resistance_ohm",
        "voltage_sag_v",
        "measured_avg_voltage_v",
        "load_power_w",
        "efficiency_fraction",
        "usable_fraction",
        "actual_capacity_ah",
        "actual_soh_pct",
        "actual_backup_time_min",
        "health_class",
        "required_runtime_min",
        "expected_recommendation",
        "suggested_holdout",
    }
    missing = sorted(required.difference(benchmark.columns))
    if missing:
        raise ValueError(f"CSGB columns missing: {missing}")
    adapted = benchmark.copy()
    adapted["cycle_count"] = adapted["cycle_number"]
    adapted["temperature_c"] = adapted["ambient_temp_c"]
    adapted["avg_voltage_v"] = adapted["measured_avg_voltage_v"]
    adapted["min_voltage_v"] = (
        adapted["measured_avg_voltage_v"] - adapted["voltage_sag_v"]
    )
    adapted["measured_capacity_ah"] = adapted["actual_capacity_ah"]
    adapted["backup_time_min"] = adapted["actual_backup_time_min"]
    adapted["source_dataset"] = "CSGB_v1.0"
    adapted = standardize_dataframe(adapted, source_name="CSGB_v1.0")
    cleaned, quality = validate_and_clean(adapted, impute=False)
    if len(cleaned) != 720 or not quality["valid"]:
        raise ValueError(f"CSGB validation mismatch: rows={len(cleaned)}, {quality}")
    cleaned["actual_soh_pct"] = (
        cleaned["measured_capacity_ah"] / cleaned["rated_capacity_ah"] * 100.0
    )
    return engineer_features(cleaned), {
        **quality,
        "source_file": path.name,
        "source_sha256": file_sha256(path),
    }


def load_public(dataset: str) -> tuple[pd.DataFrame, dict]:
    if dataset == "NASA":
        path = DATA_DIR / "processed_public" / "nasa_cycle_summary.csv"
        raw = load_nasa_processed(path)
    elif dataset == "CALCE":
        path = DATA_DIR / "processed_public" / "calce_cycle_summary.csv"
        raw = load_calce_processed(path)
    elif dataset == "TRI":
        path = DATA_DIR / "processed_public" / "tri_resval_cycle_summary.csv"
        raw = load_tri_processed(path)
    else:
        raise ValueError(dataset)
    adapted = standardize_dataframe(raw, source_name=str(raw["source_dataset"].iloc[0]))
    cleaned, quality = validate_and_clean(adapted, impute=False)
    if len(cleaned) != len(raw) or not quality["valid"]:
        raise ValueError(f"{dataset} validation mismatch: {quality}")
    cleaned["actual_soh_pct"] = (
        cleaned["measured_capacity_ah"] / cleaned["rated_capacity_ah"] * 100.0
    )
    cleaned["usable_fraction"] = 1.0
    source_quality = None
    if dataset == "TRI":
        source_quality_path = DATA_DIR / "processed_public" / "tri_resval_quality.json"
        source_quality = json.loads(source_quality_path.read_text(encoding="utf-8"))
    return engineer_features(cleaned), {
        **quality,
        "source_file": path.name,
        "source_sha256": file_sha256(path),
        "physical_batteries": int(cleaned["battery_id"].nunique()),
        "source_quality": source_quality,
    }


def split_dataset(dataset: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = DATASET_CONFIG[dataset]
    if dataset == "CSGB":
        split = frame["suggested_holdout"].astype(str).str.upper().str.strip()
        train = frame.loc[split == "DEVELOPMENT"].copy()
        test = frame.loc[split == "HOLDOUT"].copy()
    elif config["train_groups"] is not None:
        train = frame.loc[frame["battery_id"].isin(config["train_groups"])].copy()
        test = frame.loc[frame["battery_id"].isin(config["test_groups"])].copy()
    else:
        groups = np.asarray(sorted(frame["battery_id"].astype(str).unique()))
        rng = np.random.default_rng(RANDOM_STATE)
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        test_count = max(
            1,
            int(math.ceil(len(shuffled) * config["deterministic_test_fraction"])),
        )
        test_group_set = set(shuffled[:test_count])
        train = frame.loc[~frame["battery_id"].astype(str).isin(test_group_set)].copy()
        test = frame.loc[frame["battery_id"].astype(str).isin(test_group_set)].copy()
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train_groups = sorted(train["battery_id"].astype(str).unique())
    test_groups = sorted(test["battery_id"].astype(str).unique())
    overlap = sorted(set(train_groups).intersection(test_groups))
    if train.empty or test.empty or overlap:
        raise ValueError(
            f"Invalid {dataset} grouped split: train={len(train)}, "
            f"test={len(test)}, overlap={overlap}"
        )
    return train, test, {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_groups": train_groups,
        "test_groups": test_groups,
        "group_overlap": overlap,
        "overlap_count": len(overlap),
        "pass": len(overlap) == 0,
    }


def run_m9(dataset: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    expected_health: list[str] = []
    predicted_health: list[str] = []
    expected_recommendation: list[str] = []
    predicted_recommendation: list[str] = []
    predicted_level: list[str] = []
    predicted_message: list[str] = []

    for row in frame.itertuples(index=False):
        required_runtime = (
            float(row.required_runtime_min)
            if dataset == "CSGB"
            else float(DATASET_CONFIG[dataset]["required_runtime_min"])
        )
        actual_health = (
            str(row.health_class)
            if dataset == "CSGB"
            else health_class_from_soh(row.actual_soh_pct)
        )
        actual_decision = recommend_action(
            row.actual_soh_pct,
            row.backup_time_min,
            required_runtime_min=required_runtime,
            data_quality_ok=True,
        )
        expected_rec = (
            str(row.expected_recommendation)
            if dataset == "CSGB"
            else recommendation_label(actual_decision["level"])
        )
        predicted_decision = recommend_action(
            row.predicted_soh_pct,
            row.ml_backup_time_min,
            required_runtime_min=required_runtime,
            data_quality_ok=True,
        )
        expected_health.append(actual_health)
        predicted_health.append(health_class_from_soh(row.predicted_soh_pct))
        expected_recommendation.append(expected_rec)
        predicted_recommendation.append(recommendation_label(predicted_decision["level"]))
        predicted_level.append(predicted_decision["level"])
        predicted_message.append(predicted_decision["message"])

    frame = frame.copy()
    frame["expected_health_class"] = expected_health
    frame["predicted_health_class"] = predicted_health
    frame["expected_recommendation_label"] = expected_recommendation
    frame["predicted_recommendation"] = predicted_recommendation
    frame["decision_level"] = predicted_level
    frame["decision_message"] = predicted_message
    frame["health_class_match"] = np.asarray(expected_health) == np.asarray(predicted_health)
    frame["recommendation_match"] = (
        np.asarray(expected_recommendation) == np.asarray(predicted_recommendation)
    )
    metrics = {
        "health_class_accuracy_percent": exact_match_accuracy(
            expected_health, predicted_health
        ),
        "recommendation_accuracy_percent": exact_match_accuracy(
            expected_recommendation, predicted_recommendation
        ),
        "reference_note": (
            "CSGB uses benchmark labels. NASA/CALCE/TRI references are deterministic "
            "threshold labels derived from measured targets, not expert annotations."
        ),
    }
    reports = pd.concat(
        [
            classification_report_table(
                expected_health, predicted_health, f"{dataset} health class"
            ),
            classification_report_table(
                expected_recommendation,
                predicted_recommendation,
                f"{dataset} recommendation",
            ),
        ],
        ignore_index=True,
    )
    confusion = pd.concat(
        [
            confusion_count_table(
                expected_health, predicted_health, f"{dataset} health class"
            ),
            confusion_count_table(
                expected_recommendation,
                predicted_recommendation,
                f"{dataset} recommendation",
            ),
        ],
        ignore_index=True,
    )
    return frame, metrics, reports, confusion


def _monte_carlo_shap(
    model,
    frame: pd.DataFrame,
    *,
    random_state: int,
    permutations: int = 24,
    background: pd.DataFrame | None = None,
) -> tuple[np.ndarray, float]:
    """Approximate Shapley values against a declared median-row baseline.

    A single random feature ordering is applied to all rows per permutation so
    predictions remain vectorized.  For every permutation, the contributions
    telescope exactly from the baseline prediction to each row's prediction.
    Averaging across permutations provides a deterministic model-agnostic
    fallback when the optional ``shap`` package is unavailable.
    """
    values = frame.to_numpy(dtype=float)
    background_values = (
        background.to_numpy(dtype=float) if background is not None else values
    )
    baseline_row = np.median(background_values, axis=0)
    baseline = np.repeat(baseline_row.reshape(1, -1), len(frame), axis=0)
    baseline_frame = pd.DataFrame(baseline, columns=frame.columns)
    base_prediction = float(model.predict(baseline_frame.iloc[[0]])[0])
    rng = np.random.default_rng(random_state)
    orders = np.stack(
        [rng.permutation(values.shape[1]) for _ in range(permutations)]
    )
    current = np.repeat(baseline[None, :, :], permutations, axis=0)
    previous = np.full((permutations, len(frame)), base_prediction, dtype=float)
    contributions = np.zeros(
        (permutations, len(frame), values.shape[1]),
        dtype=float,
    )
    # Batch all permutations into one prediction call at each feature step.
    # This reduces P x F model calls to F while preserving the same estimator.
    for step in range(values.shape[1]):
        for permutation_index, feature_index in enumerate(orders[:, step]):
            current[permutation_index, :, feature_index] = values[:, feature_index]
        updated = model.predict(
            pd.DataFrame(
                current.reshape(-1, values.shape[1]),
                columns=frame.columns,
            )
        ).reshape(permutations, len(frame))
        for permutation_index, feature_index in enumerate(orders[:, step]):
            contributions[permutation_index, :, feature_index] = (
                updated[permutation_index] - previous[permutation_index]
            )
        previous = updated
    return contributions.mean(axis=0), base_prediction


def _explanation_values(
    model,
    frame: pd.DataFrame,
    seed: int,
    *,
    background: pd.DataFrame | None = None,
) -> tuple[np.ndarray, float, str]:
    if shap is not None:
        explanation = shap.TreeExplainer(model)(frame)
        base = float(np.asarray(explanation.base_values).reshape(-1)[0])
        return np.asarray(explanation.values, dtype=float), base, "exact TreeSHAP"
    values, base = _monte_carlo_shap(
        model,
        frame,
        random_state=seed,
        permutations=24,
        background=background,
    )
    return values, base, "deterministic Monte-Carlo Shapley (24 permutations)"


def _plot_shap_summary(
    values: np.ndarray,
    frame: pd.DataFrame,
    *,
    title: str,
    path: Path,
) -> None:
    mean_abs = np.abs(values).mean(axis=0)
    selected = np.argsort(mean_abs)[-min(10, frame.shape[1]) :]
    figure, axis = plt.subplots(figsize=(9.2, 6.4))
    rng = np.random.default_rng(RANDOM_STATE)
    for y_position, feature_index in enumerate(selected):
        feature_values = frame.iloc[:, feature_index].to_numpy(dtype=float)
        spread = np.ptp(feature_values)
        colour = (
            (feature_values - feature_values.min()) / spread
            if spread > 0
            else np.full(len(feature_values), 0.5)
        )
        jitter = rng.uniform(-0.22, 0.22, len(frame))
        axis.scatter(
            values[:, feature_index],
            y_position + jitter,
            c=colour,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=18,
            alpha=0.68,
            edgecolors="none",
        )
    axis.axvline(0, color="#555555", linewidth=0.9)
    axis.set_yticks(np.arange(len(selected)), frame.columns[selected])
    axis.set_xlabel("Shapley contribution to predicted runtime (min)")
    axis.set_title(title, weight="bold")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.99,
        0.01,
        "Colour: low (blue) to high (red) feature value",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    _finish_figure(figure, path)


def _plot_shap_local(
    values: np.ndarray,
    frame: pd.DataFrame,
    *,
    base_value: float,
    prediction: float,
    title: str,
    path: Path,
) -> None:
    row = values[0]
    selected = np.argsort(np.abs(row))[-min(12, len(row)) :]
    labels = [
        f"{frame.columns[index]} = {frame.iloc[0, index]:.4g}"
        for index in selected
    ]
    contributions = row[selected]
    colours = np.where(contributions >= 0, "#D55E00", "#0072B2")
    figure, axis = plt.subplots(figsize=(9.4, 6.2))
    axis.barh(labels, contributions, color=colours, alpha=0.88)
    axis.axvline(0, color="#444444", linewidth=0.9)
    axis.set_xlabel("Shapley contribution to predicted runtime (min)")
    axis.set_title(title, weight="bold")
    axis.grid(axis="x", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.99,
        0.02,
        f"Baseline {base_value:.3f} min  ->  prediction {prediction:.3f} min",
        transform=axis.transAxes,
        ha="right",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#BBBBBB"},
    )
    _finish_figure(figure, path)


def _shap_tables_and_plots(
    dataset: str,
    capacity_model: CapacitySOHModel,
    runtime_model: BackupTimeModel,
    test: pd.DataFrame,
    dataset_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sample_size = min(200, len(test))
    sample = test.sample(n=sample_size, random_state=RANDOM_STATE).sort_index()
    health_x = get_health_matrix(sample)
    runtime_x = get_runtime_matrix(sample)

    health_values, health_base, health_method = _explanation_values(
        capacity_model.model,
        health_x,
        RANDOM_STATE + list(DATASET_CONFIG).index(dataset) * 10,
    )
    runtime_values, runtime_base, runtime_method = _explanation_values(
        runtime_model.model,
        runtime_x,
        RANDOM_STATE + list(DATASET_CONFIG).index(dataset) * 10 + 1,
    )
    health_table = pd.DataFrame(
        {
            "dataset": dataset,
            "model": "M5 capacity",
            "feature": health_x.columns,
            "mean_abs_shap": np.abs(health_values).mean(axis=0),
            "method": health_method,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    runtime_table = pd.DataFrame(
        {
            "dataset": dataset,
            "model": "M6 backup time",
            "feature": runtime_x.columns,
            "mean_abs_shap": np.abs(runtime_values).mean(axis=0),
            "method": runtime_method,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    summary_path = GRAPH_DIR / f"{13 + list(DATASET_CONFIG).index(dataset):02d}_{dataset.lower()}_runtime_shap_summary.png"
    _plot_shap_summary(
        runtime_values,
        runtime_x,
        title=f"{dataset}: M6 global Shapley summary ({sample_size} held-out records)",
        path=summary_path,
    )

    absolute_error = (test["ml_backup_time_min"] - test["backup_time_min"]).abs()
    representative_position = int(np.argsort(absolute_error.to_numpy())[len(test) // 2])
    representative = test.iloc[[representative_position]]
    local_x = get_runtime_matrix(representative)
    record_id = str(representative.iloc[0]["record_id"])
    local_values, local_base, local_method = _explanation_values(
        runtime_model.model,
        local_x,
        RANDOM_STATE + list(DATASET_CONFIG).index(dataset) * 10 + 2,
        background=runtime_x,
    )
    local_path = GRAPH_DIR / f"{17 + list(DATASET_CONFIG).index(dataset):02d}_{dataset.lower()}_runtime_shap_local.png"
    _plot_shap_local(
        local_values,
        local_x,
        base_value=local_base,
        prediction=float(representative.iloc[0]["ml_backup_time_min"]),
        title=f"{dataset}: representative M6 local Shapley explanation - {record_id}",
        path=local_path,
    )

    local_table = pd.DataFrame(
        {
            "dataset": dataset,
            "record_id": record_id,
            "feature": local_x.columns,
            "feature_value": local_x.iloc[0].to_numpy(dtype=float),
            "shap_value_min": local_values[0],
            "method": local_method,
        }
    ).sort_values("shap_value_min", key=lambda values: values.abs(), ascending=False)
    local_table.to_csv(dataset_output / "shap_local_runtime.csv", index=False)
    return health_table, runtime_table, {
        "status": "pass",
        "sample_rows": sample_size,
        "runtime_features": int(runtime_x.shape[1]),
        "capacity_features": int(health_x.shape[1]),
        "runtime_method": runtime_method,
        "capacity_method": health_method,
        "optional_shap_package_available": shap is not None,
        "optional_shap_import_error": SHAP_IMPORT_ERROR,
        "representative_record_id": record_id,
        "runtime_expected_value_min": local_base,
        "runtime_prediction_min": float(representative.iloc[0]["ml_backup_time_min"]),
        "runtime_actual_min": float(representative.iloc[0]["backup_time_min"]),
    }


def evaluate_dataset(dataset: str, frame: pd.DataFrame, quality: dict) -> dict:
    dataset_output = OUTPUT_DIR / dataset.lower()
    dataset_output.mkdir(parents=True, exist_ok=True)
    train, test, split = split_dataset(dataset, frame)

    capacity_model = CapacitySOHModel(random_state=RANDOM_STATE)
    capacity_start = time.perf_counter()
    _, oof_soh = capacity_model.oof_predictions_for_runtime(
        train, group_col="battery_id", max_splits=5
    )
    capacity_model.fit(train)
    capacity_train_seconds = time.perf_counter() - capacity_start
    train["predicted_soh_percent"] = oof_soh

    capacity_inference_start = time.perf_counter()
    test["predicted_capacity_ah"] = capacity_model.predict_capacity(test)
    test["predicted_soh_pct"] = capacity_model.predict_soh(test)
    capacity_inference_seconds = time.perf_counter() - capacity_inference_start
    test["predicted_soh_percent"] = test["predicted_soh_pct"]

    runtime_model = BackupTimeModel(random_state=RANDOM_STATE)
    runtime_start = time.perf_counter()
    runtime_model.fit(train)
    runtime_train_seconds = time.perf_counter() - runtime_start
    runtime_inference_start = time.perf_counter()
    test["ml_backup_time_min"] = runtime_model.predict(test)
    runtime_inference_seconds = time.perf_counter() - runtime_inference_start

    efficiency = (
        test["efficiency_fraction"]
        if DATASET_CONFIG[dataset]["formula_efficiency"] == "column"
        else float(DATASET_CONFIG[dataset]["formula_efficiency"])
    )
    usable_fraction = (
        test["usable_fraction"] if "usable_fraction" in test else 1.0
    )
    test["formula_reference_min"] = formula_backup_time_minutes(
        test["avg_voltage_v"],
        test["measured_capacity_ah"],
        test["load_power_w"],
        efficiency=efficiency,
        usable_fraction=usable_fraction,
    )
    test["formula_operational_min"] = formula_backup_time_minutes(
        test["avg_voltage_v"],
        test["predicted_capacity_ah"],
        test["load_power_w"],
        efficiency=efficiency,
        usable_fraction=usable_fraction,
    )

    capacity_metrics = regression_metrics(
        test["measured_capacity_ah"], test["predicted_capacity_ah"]
    )
    soh_metrics = regression_metrics(test["actual_soh_pct"], test["predicted_soh_pct"])
    formula_reference_metrics = regression_metrics(
        test["backup_time_min"], test["formula_reference_min"]
    )
    formula_operational_metrics = regression_metrics(
        test["backup_time_min"], test["formula_operational_min"]
    )
    runtime_metrics = regression_metrics(
        test["backup_time_min"], test["ml_backup_time_min"]
    )
    error_reduction = {
        metric: (
            (formula_operational_metrics[metric] - runtime_metrics[metric])
            / formula_operational_metrics[metric]
            * 100.0
            if abs(formula_operational_metrics[metric]) > 1e-12
            else None
        )
        for metric in ("MAE", "RMSE", "MAPE_percent")
    }

    for prefix, actual, predicted in (
        ("capacity", "measured_capacity_ah", "predicted_capacity_ah"),
        ("soh", "actual_soh_pct", "predicted_soh_pct"),
        ("formula_runtime", "backup_time_min", "formula_operational_min"),
        ("ml_runtime", "backup_time_min", "ml_backup_time_min"),
    ):
        test[f"{prefix}_error"] = test[predicted] - test[actual]
        test[f"{prefix}_abs_error"] = test[f"{prefix}_error"].abs()
        test[f"{prefix}_ape_pct"] = (
            test[f"{prefix}_abs_error"] / test[actual].abs().clip(lower=1e-9) * 100.0
        )

    test, m9_metrics, m9_report, m9_confusion = run_m9(dataset, test)

    health_x = get_health_matrix(test)
    runtime_x = get_runtime_matrix(test)
    builtin = pd.concat(
        [
            built_in_feature_importance(capacity_model.model, health_x.columns).assign(
                dataset=dataset, model="M5 capacity"
            ),
            built_in_feature_importance(runtime_model.model, runtime_x.columns).assign(
                dataset=dataset, model="M6 backup time"
            ),
        ],
        ignore_index=True,
    )[["dataset", "model", "feature", "importance"]]
    permutation = pd.concat(
        [
            permutation_feature_importance(
                capacity_model.model,
                health_x,
                test["measured_capacity_ah"],
                random_state=RANDOM_STATE,
                n_repeats=10,
            ).assign(dataset=dataset, model="M5 capacity"),
            permutation_feature_importance(
                runtime_model.model,
                runtime_x,
                test["backup_time_min"],
                random_state=RANDOM_STATE,
                n_repeats=10,
            ).assign(dataset=dataset, model="M6 backup time"),
        ],
        ignore_index=True,
    )[["dataset", "model", "feature", "importance_mean", "importance_std"]]

    health_shap, runtime_shap, shap_status = _shap_tables_and_plots(
        dataset, capacity_model, runtime_model, test, dataset_output
    )
    shap_importance = pd.concat([health_shap, runtime_shap], ignore_index=True)

    capacity_path = MODEL_DIR / f"{dataset.lower()}_capacity_model.joblib"
    runtime_path = MODEL_DIR / f"{dataset.lower()}_runtime_model.joblib"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(capacity_model, capacity_path, compress=3)
    joblib.dump(runtime_model, runtime_path, compress=3)
    reloaded_capacity = joblib.load(capacity_path)
    reloaded_runtime = joblib.load(runtime_path)
    capacity_reload_diff = float(
        np.max(
            np.abs(
                reloaded_capacity.predict_capacity(test)
                - test["predicted_capacity_ah"].to_numpy()
            )
        )
    )
    runtime_reload_diff = float(
        np.max(
            np.abs(
                reloaded_runtime.predict(test)
                - test["ml_backup_time_min"].to_numpy()
            )
        )
    )
    persistence = {
        "capacity_model_file": capacity_path.name,
        "capacity_model_sha256": file_sha256(capacity_path),
        "runtime_model_file": runtime_path.name,
        "runtime_model_sha256": file_sha256(runtime_path),
        "capacity_max_abs_reload_difference": capacity_reload_diff,
        "runtime_max_abs_reload_difference": runtime_reload_diff,
        # 1e-9 minutes is far below measurement resolution and accommodates
        # harmless parallel floating-point reduction-order differences.
        "tolerance": 1e-9,
        "pass": capacity_reload_diff <= 1e-9 and runtime_reload_diff <= 1e-9,
    }
    if not persistence["pass"]:
        raise ValueError(f"{dataset} persistence consistency failed: {persistence}")

    per_battery = {}
    for battery_id, group in test.groupby("battery_id"):
        per_battery[str(battery_id)] = {
            "rows": int(len(group)),
            "capacity": regression_metrics(
                group["measured_capacity_ah"], group["predicted_capacity_ah"]
            ),
            "formula_runtime": regression_metrics(
                group["backup_time_min"], group["formula_operational_min"]
            ),
            "ml_runtime": regression_metrics(
                group["backup_time_min"], group["ml_backup_time_min"]
            ),
        }

    per_rate: dict[str, dict] = {}
    if dataset == "TRI":
        rate_rows = []
        for rate_label, group in test.groupby("source_rate_label", sort=False):
            capacity_rate = regression_metrics(
                group["measured_capacity_ah"], group["predicted_capacity_ah"]
            )
            formula_rate = regression_metrics(
                group["backup_time_min"], group["formula_operational_min"]
            )
            ml_rate = regression_metrics(
                group["backup_time_min"], group["ml_backup_time_min"]
            )
            rate_record = {
                "source_rate_index": int(group["source_rate_index"].iloc[0]),
                "source_rate_label": str(rate_label),
                "source_nominal_c_rate": float(group["source_nominal_c_rate"].iloc[0]),
                "records": int(len(group)),
                "physical_cells": int(group["battery_id"].nunique()),
                "capacity_mean_ah": float(group["measured_capacity_ah"].mean()),
                "capacity_min_ah": float(group["measured_capacity_ah"].min()),
                "capacity_max_ah": float(group["measured_capacity_ah"].max()),
                "soh_mean_percent": float(group["actual_soh_pct"].mean()),
                "runtime_mean_min": float(group["backup_time_min"].mean()),
                "runtime_min_min": float(group["backup_time_min"].min()),
                "runtime_max_min": float(group["backup_time_min"].max()),
                "capacity_mae_ah": capacity_rate["MAE"],
                "capacity_mape_percent": capacity_rate["MAPE_percent"],
                "m4_runtime_mae_min": formula_rate["MAE"],
                "m4_runtime_mape_percent": formula_rate["MAPE_percent"],
                "m6_runtime_mae_min": ml_rate["MAE"],
                "m6_runtime_mape_percent": ml_rate["MAPE_percent"],
                "lower_runtime_mae_method": (
                    "M6 ML" if ml_rate["MAE"] < formula_rate["MAE"] else "M4 formula"
                ),
            }
            rate_rows.append(rate_record)
            per_rate[str(rate_label)] = rate_record
        pd.DataFrame(rate_rows).sort_values("source_rate_index").to_csv(
            dataset_output / "rate_level_metrics.csv",
            index=False,
        )

    prediction_columns = [
        "record_id",
        "source_dataset",
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
        "predicted_capacity_ah",
        "actual_soh_pct",
        "predicted_soh_pct",
        "backup_time_min",
        "formula_reference_min",
        "formula_operational_min",
        "ml_backup_time_min",
        "capacity_error",
        "capacity_abs_error",
        "capacity_ape_pct",
        "soh_error",
        "soh_abs_error",
        "soh_ape_pct",
        "formula_runtime_error",
        "formula_runtime_abs_error",
        "formula_runtime_ape_pct",
        "ml_runtime_error",
        "ml_runtime_abs_error",
        "ml_runtime_ape_pct",
        "expected_health_class",
        "predicted_health_class",
        "health_class_match",
        "expected_recommendation_label",
        "predicted_recommendation",
        "recommendation_match",
        "decision_level",
        "decision_message",
    ]
    if dataset == "TRI":
        prediction_columns[4:4] = [
            "source_rate_index",
            "source_rate_label",
            "source_nominal_c_rate",
        ]
    test.loc[:, prediction_columns].to_csv(
        dataset_output / "heldout_predictions.csv", index=False
    )
    builtin.to_csv(dataset_output / "builtin_feature_importance.csv", index=False)
    permutation.to_csv(
        dataset_output / "permutation_feature_importance.csv", index=False
    )
    shap_importance.to_csv(
        dataset_output / "shap_global_importance.csv", index=False
    )
    m9_report.to_csv(dataset_output / "m9_classification_report.csv", index=False)
    m9_confusion.to_csv(dataset_output / "m9_confusion_counts.csv", index=False)

    metrics = {
        "dataset": dataset,
        "label": DATASET_CONFIG[dataset]["label"],
        "evidence_domain": DATASET_CONFIG[dataset]["domain"],
        "quality": quality,
        "split": split,
        "m5_capacity": capacity_metrics,
        "m5_soh": soh_metrics,
        "m4_reference_runtime": formula_reference_metrics,
        "m4_operational_runtime": formula_operational_metrics,
        "m6_ml_runtime": runtime_metrics,
        "m6_vs_m4_error_reduction_percent": error_reduction,
        "m8_shap": shap_status,
        "m9": m9_metrics,
        "persistence": persistence,
        "timing_seconds": {
            "capacity_train_including_grouped_oof": capacity_train_seconds,
            "capacity_inference": capacity_inference_seconds,
            "runtime_train": runtime_train_seconds,
            "runtime_inference": runtime_inference_seconds,
            "capacity_inference_ms_per_row": capacity_inference_seconds / len(test) * 1000,
            "runtime_inference_ms_per_row": runtime_inference_seconds / len(test) * 1000,
        },
        "per_battery": per_battery,
        "per_rate": per_rate,
        "feature_contract": {
            "health_features": HEALTH_FEATURES,
            "runtime_features": RUNTIME_FEATURES,
            "target_leakage_check_pass": (
                {"measured_capacity_ah", "backup_time_min"}.isdisjoint(HEALTH_FEATURES)
                and {"measured_capacity_ah", "backup_time_min"}.isdisjoint(RUNTIME_FEATURES)
            ),
        },
    }
    save_json(dataset_output / "metrics.json", metrics)
    return {
        "dataset": dataset,
        "train": train,
        "test": test,
        "metrics": metrics,
        "builtin": builtin,
        "permutation": permutation,
        "shap": shap_importance,
    }


def _finish_figure(figure, path: Path) -> None:
    if figure._suptitle is not None:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    else:
        figure.tight_layout()
    temporary = path.with_name(f".{path.stem}.writing")
    temporary.unlink(missing_ok=True)
    figure.savefig(
        temporary,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise OSError(f"Graph not written: {temporary}")
    temporary.replace(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"Atomic graph replacement failed: {path}")


def plot_record_coverage(results: dict[str, dict]) -> None:
    datasets = list(results)
    train = [results[name]["metrics"]["split"]["train_rows"] for name in datasets]
    test = [results[name]["metrics"]["split"]["test_rows"] for name in datasets]
    x = np.arange(len(datasets))
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    axis.bar(x, train, color="#0072B2", label="Development/train")
    axis.bar(x, test, bottom=train, color="#E69F00", label="Grouped holdout")
    for index, (a, b) in enumerate(zip(train, test)):
        axis.text(index, a / 2, str(a), ha="center", va="center", color="white", weight="bold")
        axis.text(index, a + b / 2, str(b), ha="center", va="center", color="#222222", weight="bold")
    axis.set_xticks(x, datasets)
    axis.set_ylabel("Cycle-level records")
    axis.set_title("Phase-3 evidence coverage by dataset and grouped split", weight="bold")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, GRAPH_DIR / "01_dataset_record_coverage.png")


def plot_split_integrity(results: dict[str, dict]) -> None:
    datasets = list(results)
    train_groups = [len(results[name]["metrics"]["split"]["train_groups"]) for name in datasets]
    test_groups = [len(results[name]["metrics"]["split"]["test_groups"]) for name in datasets]
    overlaps = [results[name]["metrics"]["split"]["overlap_count"] for name in datasets]
    x = np.arange(len(datasets))
    width = 0.25
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    axis.bar(x - width, train_groups, width, label="Train groups", color="#0072B2")
    axis.bar(x, test_groups, width, label="Holdout groups", color="#009E73")
    axis.bar(x + width, overlaps, width, label="Overlap (must be 0)", color="#D55E00")
    axis.set_xticks(x, datasets)
    axis.set_ylabel("Physical battery/cell groups")
    axis.set_title("Grouped split integrity: no physical group overlap", weight="bold")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, GRAPH_DIR / "02_grouped_split_integrity.png")


def _plot_scatter(results: dict[str, dict], actual: str, predicted: str, title: str, unit: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6.0 * len(results), 5.6), squeeze=False)
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    for index, (dataset, result) in enumerate(results.items()):
        test = result["test"]
        axis = axes[0, index]
        x = test[actual].to_numpy(float)
        y = test[predicted].to_numpy(float)
        low = min(x.min(), y.min())
        high = max(x.max(), y.max())
        pad = max((high - low) * 0.05, 0.02)
        metrics = regression_metrics(x, y)
        axis.scatter(x, y, s=20, alpha=0.58, color=colors[index], edgecolors="none")
        axis.plot([low - pad, high + pad], [low - pad, high + pad], "--", color="#444444")
        axis.set_xlim(low - pad, high + pad)
        axis.set_ylim(low - pad, high + pad)
        axis.set_title(dataset, weight="bold")
        axis.set_xlabel(f"Measured/reference ({unit})")
        axis.set_ylabel(f"Predicted ({unit})")
        axis.grid(alpha=0.2)
        axis.text(
            0.04,
            0.96,
            f"MAE {metrics['MAE']:.3f}\nMAPE {metrics['MAPE_percent']:.2f}%\nR2 {metrics['R2']:.3f}",
            transform=axis.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#BBBBBB"},
        )
    fig.suptitle(title, fontsize=15, weight="bold")
    _finish_figure(fig, path)


def plot_runtime_scatter(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(len(results), 2, figsize=(11.5, 5.2 * len(results)), squeeze=False)
    for row, (dataset, result) in enumerate(results.items()):
        test = result["test"]
        actual = test["backup_time_min"].to_numpy(float)
        for column, (name, prediction, color) in enumerate(
            [
                ("M4 operational formula", "formula_operational_min", "#D55E00"),
                ("M6 Random Forest", "ml_backup_time_min", "#0072B2"),
            ]
        ):
            predicted = test[prediction].to_numpy(float)
            low = min(actual.min(), predicted.min())
            high = max(actual.max(), predicted.max())
            pad = max((high - low) * 0.05, 0.3)
            metrics = regression_metrics(actual, predicted)
            axis = axes[row, column]
            axis.scatter(actual, predicted, s=19, alpha=0.55, color=color, edgecolors="none")
            axis.plot([low - pad, high + pad], [low - pad, high + pad], "--", color="#444444")
            axis.set_title(f"{dataset} - {name}", weight="bold")
            axis.set_xlabel("Measured/reference runtime (min)")
            axis.set_ylabel("Predicted runtime (min)")
            axis.grid(alpha=0.2)
            axis.text(
                0.04,
                0.96,
                f"MAE {metrics['MAE']:.3f} min\nMAPE {metrics['MAPE_percent']:.2f}%\nR2 {metrics['R2']:.3f}",
                transform=axis.transAxes,
                va="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#BBBBBB"},
            )
    fig.suptitle("M4 formula and M6 ML runtime on identical grouped holdout records", fontsize=15, weight="bold")
    _finish_figure(fig, GRAPH_DIR / "05_runtime_actual_vs_predicted.png")


def plot_runtime_metrics(results: dict[str, dict]) -> None:
    datasets = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for axis, metric, label in zip(axes, ["MAE", "RMSE", "MAPE_percent"], ["MAE (min)", "RMSE (min)", "MAPE (%)"]):
        x = np.arange(len(datasets))
        formula = [results[name]["metrics"]["m4_operational_runtime"][metric] for name in datasets]
        ml = [results[name]["metrics"]["m6_ml_runtime"][metric] for name in datasets]
        axis.bar(x - 0.18, formula, 0.36, label="M4 formula", color="#D55E00")
        axis.bar(x + 0.18, ml, 0.36, label="M6 ML", color="#0072B2")
        axis.set_xticks(x, datasets)
        axis.set_ylabel(label)
        axis.set_title(label, weight="bold")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Runtime error comparison across controlled and real experimental domains", fontsize=15, weight="bold")
    _finish_figure(fig, GRAPH_DIR / "06_runtime_error_metrics.png")


def plot_residuals(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6.0 * len(results), 5.2), squeeze=False)
    for index, (dataset, result) in enumerate(results.items()):
        test = result["test"]
        formula = test["formula_operational_min"] - test["backup_time_min"]
        ml = test["ml_backup_time_min"] - test["backup_time_min"]
        axis = axes[0, index]
        axis.hist(formula, bins=25, alpha=0.55, color="#D55E00", label="M4 formula")
        axis.hist(ml, bins=25, alpha=0.6, color="#0072B2", label="M6 ML")
        axis.axvline(0, color="#333333", linestyle="--")
        axis.set_title(dataset, weight="bold")
        axis.set_xlabel("Signed error: predicted - actual (min)")
        axis.set_ylabel("Records")
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Runtime residual distributions on grouped holdout records", fontsize=15, weight="bold")
    _finish_figure(fig, GRAPH_DIR / "07_runtime_residual_distributions.png")


def plot_cycle_quartile_error(results: dict[str, dict]) -> None:
    rows = []
    for dataset, result in results.items():
        test = result["test"].copy()
        if test["cycle_count"].nunique() >= 4:
            test["analysis_quartile"] = pd.qcut(
                test["cycle_count"],
                4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            )
            basis = "Cycle"
        else:
            test["analysis_quartile"] = pd.qcut(
                test["source_nominal_c_rate"],
                4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            )
            basis = "Rate"
        for quartile, group in test.groupby("analysis_quartile", observed=True):
            rows.append(
                {
                    "dataset": dataset,
                    "quartile": str(quartile),
                    "basis": basis,
                    "M4": group["formula_runtime_abs_error"].mean(),
                    "M6": group["ml_runtime_abs_error"].mean(),
                }
            )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(results), figsize=(6.0 * len(results), 5.0), squeeze=False)
    for index, dataset in enumerate(results):
        subset = table.loc[table["dataset"] == dataset]
        x = np.arange(len(subset))
        axis = axes[0, index]
        axis.plot(x, subset["M4"], "o-", color="#D55E00", label="M4 formula")
        axis.plot(x, subset["M6"], "o-", color="#0072B2", label="M6 ML")
        axis.set_xticks(x, subset["quartile"])
        axis.set_title(dataset, weight="bold")
        basis = subset["basis"].iloc[0]
        axis.set_xlabel(f"{basis} quartile")
        axis.set_ylabel("Mean absolute runtime error (min)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.suptitle(
        "Runtime error stability by cycle region (TRI: discharge-rate region)",
        fontsize=15,
        weight="bold",
    )
    _finish_figure(fig, GRAPH_DIR / "08_runtime_error_by_cycle_quartile.png")


def plot_degradation_profiles(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.2), squeeze=False)
    for index, (dataset, result) in enumerate(results.items()):
        combined = pd.concat([result["train"], result["test"]], ignore_index=True)
        axis = axes[0, index]
        for battery_id, group in combined.groupby("battery_id"):
            if dataset == "TRI":
                group = group.sort_values("source_nominal_c_rate")
                x = group["source_nominal_c_rate"]
            else:
                group = group.sort_values("cycle_count")
                x = group["cycle_count"]
            axis.plot(x, group["actual_soh_pct"], linewidth=1.0, alpha=0.58, label=str(battery_id))
        axis.axhline(80, color="#D55E00", linestyle="--", linewidth=1.0)
        axis.set_title(dataset, weight="bold")
        if dataset == "TRI":
            axis.set_xscale("log")
            axis.set_xlabel("Nominal discharge C-rate (log scale)")
            axis.text(
                0.03,
                0.04,
                "94 cells; one line per physical cell\nAging-cycle count unavailable",
                transform=axis.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
            )
        else:
            axis.set_xlabel("Cycle count")
        axis.set_ylabel("Measured/reference SOH (%)")
        axis.grid(alpha=0.2)
        if dataset != "TRI":
            axis.legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle(
        "Capacity/SOH profiles by physical cell (TRI panel shows rate dependence)",
        fontsize=15,
        weight="bold",
    )
    _finish_figure(fig, GRAPH_DIR / "09_capacity_degradation_profiles.png")


def plot_importance(results: dict[str, dict], kind: str, path: Path, title: str) -> None:
    value_column = "importance" if kind == "builtin" else "importance_mean" if kind == "permutation" else "mean_abs_shap"
    fig, axes = plt.subplots(len(results), 2, figsize=(12.5, 4.6 * len(results)), squeeze=False)
    colors = {"M5 capacity": "#009E73", "M6 backup time": "#0072B2"}
    for row, (dataset, result) in enumerate(results.items()):
        table = result[kind]
        for column, model in enumerate(["M5 capacity", "M6 backup time"]):
            subset = table.loc[table["model"] == model].nlargest(8, value_column).sort_values(value_column)
            axis = axes[row, column]
            axis.barh(subset["feature"], subset[value_column], color=colors[model], alpha=0.88)
            axis.set_title(f"{dataset} - {model}", weight="bold")
            axis.set_xlabel(value_column.replace("_", " "))
            axis.grid(axis="x", alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=15, weight="bold")
    _finish_figure(fig, path)


def plot_persistence(results: dict[str, dict]) -> None:
    datasets = list(results)
    capacity = [max(results[name]["metrics"]["persistence"]["capacity_max_abs_reload_difference"], 1e-18) for name in datasets]
    runtime = [max(results[name]["metrics"]["persistence"]["runtime_max_abs_reload_difference"], 1e-18) for name in datasets]
    x = np.arange(len(datasets))
    fig, axis = plt.subplots(figsize=(8.6, 5.4))
    axis.bar(x - 0.18, capacity, 0.36, label="M5 capacity model", color="#009E73")
    axis.bar(x + 0.18, runtime, 0.36, label="M6 runtime model", color="#0072B2")
    axis.axhline(1e-9, color="#D55E00", linestyle="--", label="Pass tolerance 1e-9")
    axis.set_yscale("log")
    axis.set_xticks(x, datasets)
    axis.set_ylabel("Maximum absolute prediction difference after reload")
    axis.set_title("Persisted model reload consistency", weight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, GRAPH_DIR / "21_model_persistence_reload_consistency.png")


def plot_m9(results: dict[str, dict]) -> None:
    datasets = list(results)
    health = [results[name]["metrics"]["m9"]["health_class_accuracy_percent"] for name in datasets]
    rec = [results[name]["metrics"]["m9"]["recommendation_accuracy_percent"] for name in datasets]
    x = np.arange(len(datasets))
    fig, axis = plt.subplots(figsize=(9.0, 5.4))
    axis.bar(x - 0.18, health, 0.36, label="Health class", color="#009E73")
    axis.bar(x + 0.18, rec, 0.36, label="Recommendation", color="#0072B2")
    axis.set_xticks(x, datasets)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Exact-match accuracy (%)")
    axis.set_title("M9 decision-support label agreement", weight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    _finish_figure(fig, GRAPH_DIR / "22_m9_exact_match_accuracy.png")


def generate_graphs(results: dict[str, dict]) -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    plot_record_coverage(results)
    plot_split_integrity(results)
    _plot_scatter(
        results,
        "measured_capacity_ah",
        "predicted_capacity_ah",
        "M5 remaining-capacity predictions on grouped holdout records",
        "Ah",
        GRAPH_DIR / "03_capacity_actual_vs_predicted.png",
    )
    _plot_scatter(
        results,
        "actual_soh_pct",
        "predicted_soh_pct",
        "M5 derived SOH predictions on grouped holdout records",
        "%",
        GRAPH_DIR / "04_soh_actual_vs_predicted.png",
    )
    plot_runtime_scatter(results)
    plot_runtime_metrics(results)
    plot_residuals(results)
    plot_cycle_quartile_error(results)
    plot_degradation_profiles(results)
    plot_importance(results, "builtin", GRAPH_DIR / "10_builtin_feature_importance.png", "Built-in Random Forest feature importance")
    plot_importance(results, "permutation", GRAPH_DIR / "11_permutation_feature_importance.png", "Ten-repeat permutation importance on grouped holdouts")
    plot_importance(results, "shap", GRAPH_DIR / "12_shap_mean_absolute_importance.png", "Global mean absolute SHAP importance")
    plot_persistence(results)
    plot_m9(results)
    graph_count = len(list(GRAPH_DIR.glob("*.png")))
    if graph_count != 22:
        raise ValueError(f"Expected 22 Phase-3 graphs, found {graph_count}.")


def write_summary(results: dict[str, dict], combined_metrics: dict) -> None:
    lines = [
        "PHASE-3 IMPLEMENTATION AND VALIDATION SUMMARY",
        "",
        "Evidence domains",
        "  CSGB: controlled synthetic software/continuity benchmark.",
        "  NASA, CALCE and TRI: real public laboratory discharge measurements.",
        "  TRI adds 750 observed rate-performance records from 94 aged EV cells;",
        "  prior aging-cycle count and temperature are not present in its raw files.",
        "  Oxford: native MAT adapter contract implemented; quantitative source-file",
        "  validation not claimed because the official 262 MB file was not supplied.",
        "",
    ]
    for dataset, result in results.items():
        metrics = result["metrics"]
        lines.extend(
            [
                dataset,
                f"  Split: {metrics['split']['train_rows']} train / {metrics['split']['test_rows']} holdout rows; overlap {metrics['split']['overlap_count']}",
                f"  M5 capacity: MAE {metrics['m5_capacity']['MAE']:.4f} Ah; MAPE {metrics['m5_capacity']['MAPE_percent']:.2f}%; R2 {metrics['m5_capacity']['R2']:.4f}",
                f"  M5 SOH: MAE {metrics['m5_soh']['MAE']:.3f} percentage points; R2 {metrics['m5_soh']['R2']:.4f}",
                f"  M4 operational runtime: MAE {metrics['m4_operational_runtime']['MAE']:.3f} min; MAPE {metrics['m4_operational_runtime']['MAPE_percent']:.2f}%",
                f"  M6 runtime: MAE {metrics['m6_ml_runtime']['MAE']:.3f} min; MAPE {metrics['m6_ml_runtime']['MAPE_percent']:.2f}%; R2 {metrics['m6_ml_runtime']['R2']:.4f}",
                f"  M6 MAE reduction vs M4: {metrics['m6_vs_m4_error_reduction_percent']['MAE']:.2f}%",
                f"  SHAP: {metrics['m8_shap']['status']} ({metrics['m8_shap']['sample_rows']} held-out rows)",
                f"  Persistence: {'PASS' if metrics['persistence']['pass'] else 'FAIL'}",
                "",
            ]
        )
    lines.extend(
        [
            "Interpretation boundary",
            "  NASA, CALCE and TRI are genuine laboratory datasets, but the evaluated",
            "  records use constrained protocols. TRI expands chemistry, cell count,",
            "  health range and discharge-rate coverage; it is not field telemetry.",
            "  These results do not establish general field accuracy for",
            "  UPS/inverter/telecom loads, other chemistries,",
            "  or safety-critical maintenance decisions.",
            "",
            f"Automated graph count: {combined_metrics['inventory']['graph_count']}",
            f"Persisted model count: {combined_metrics['inventory']['model_count']}",
        ]
    )
    (OUTPUT_DIR / "validation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for old_graph in GRAPH_DIR.iterdir():
        if old_graph.is_file() and (
            old_graph.suffix.lower() == ".png" or ".writing" in old_graph.name
        ):
            old_graph.unlink()
    print("=" * 78)
    print("PHASE-3 COMPLETE IMPLEMENTATION AND VALIDATION")
    print("=" * 78)

    csgb, csgb_quality = load_csgb()
    nasa, nasa_quality = load_public("NASA")
    calce, calce_quality = load_public("CALCE")
    tri, tri_quality = load_public("TRI")
    datasets = {
        "CSGB": (csgb, csgb_quality),
        "NASA": (nasa, nasa_quality),
        "CALCE": (calce, calce_quality),
        "TRI": (tri, tri_quality),
    }
    results: dict[str, dict] = {}
    for dataset, (frame, quality) in datasets.items():
        print(f"\n[{dataset}] rows={len(frame)} groups={frame['battery_id'].nunique()}")
        results[dataset] = evaluate_dataset(dataset, frame, quality)
        metrics = results[dataset]["metrics"]
        print(
            f"  split {metrics['split']['train_rows']}/{metrics['split']['test_rows']} "
            f"overlap={metrics['split']['overlap_count']}"
        )
        print(
            f"  capacity MAE={metrics['m5_capacity']['MAE']:.4f} Ah; "
            f"runtime M6 MAE={metrics['m6_ml_runtime']['MAE']:.3f} min"
        )

    generate_graphs(results)

    metric_rows = []
    for dataset, result in results.items():
        metrics = result["metrics"]
        for output, unit in (
            ("m5_capacity", "Ah"),
            ("m5_soh", "percentage points"),
            ("m4_operational_runtime", "min"),
            ("m6_ml_runtime", "min"),
        ):
            for metric, value in metrics[output].items():
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "evidence_domain": metrics["evidence_domain"],
                        "output": output,
                        "metric": metric,
                        "value": value,
                        "unit": "%" if metric == "MAPE_percent" else "unitless" if metric == "R2" else unit,
                    }
                )
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "phase3_metrics_table.csv", index=False)

    environment = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
        "shap": getattr(shap, "__version__", None),
        "random_state": RANDOM_STATE,
    }
    adapter_status = {
        "NASA": {
            "status": "verified on official source-native MAT files",
            "quantitative_evaluation": True,
            "records": int(len(nasa)),
        },
        "CALCE": {
            "status": "verified on official source-native XLSX files",
            "quantitative_evaluation": True,
            "records": int(len(calce)),
        },
        "TRI": {
            "status": "verified on official 94-cell Maccor text archive",
            "quantitative_evaluation": True,
            "records": int(len(tri)),
            "physical_cells": int(tri["battery_id"].nunique()),
            "data_limitations": (
                "Raw files do not contain measured temperature or prior aging-cycle count; "
                "temperature is a declared constant schema placeholder and cycle_count is a zero sentinel."
            ),
        },
        "Oxford": {
            "status": "native MAT contract implemented and fixture-tested; official source file not executed",
            "quantitative_evaluation": False,
            "reason": "Official 262 MB characterization MAT file was not supplied in the project inputs.",
        },
    }
    combined_metrics = {
        "evidence_boundary": {
            "controlled": ["CSGB"],
            "real_experimental": ["NASA", "CALCE", "TRI"],
            "adapter_only": ["Oxford"],
            "field_validation": False,
            "safety_certification": False,
        },
        "datasets": {name: result["metrics"] for name, result in results.items()},
        "adapter_status": adapter_status,
        "environment": environment,
        "inventory": {
            "graph_count": len(list(GRAPH_DIR.glob("*.png"))),
            "model_count": len(list(MODEL_DIR.glob("*.joblib"))),
            "dataset_output_directories": len(results),
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    save_json(OUTPUT_DIR / "phase3_metrics.json", combined_metrics)
    save_json(OUTPUT_DIR / "source_adapter_status.json", adapter_status)
    save_json(OUTPUT_DIR / "environment.json", environment)

    dashboard_metadata = {
        "default_model_domain": "CSGB controlled benchmark",
        "capacity_model": "models/csgb_capacity_model.joblib",
        "runtime_model": "models/csgb_runtime_model.joblib",
        "health_features": HEALTH_FEATURES,
        "runtime_features": RUNTIME_FEATURES,
        "warning": (
            "Dashboard predictions are prototype decision support. The default "
            "models are trained on the controlled CSGB benchmark and are not "
            "validated for field or safety-critical use."
        ),
    }
    save_json(OUTPUT_DIR / "dashboard_model_metadata.json", dashboard_metadata)
    write_summary(results, combined_metrics)
    print("\n" + "=" * 78)
    print("PHASE-3 VALIDATION COMPLETED")
    print(f"Graphs: {combined_metrics['inventory']['graph_count']}")
    print(f"Models: {combined_metrics['inventory']['model_count']}")
    print(f"Elapsed: {combined_metrics['elapsed_seconds']:.2f} s")
    print(f"Outputs: {OUTPUT_DIR.resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
