"""M7 metrics and comparison-graph helpers.

All functions expect predictions that are already aligned to the same record
IDs and held-out split. The Phase-2 comparison runner performs that alignment
before calling these helpers.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """Return the Phase-2 regression metric set."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Metric arrays must have the same shape: "
            f"{y_true.shape} != {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("Metric arrays cannot be empty.")
    if np.any(~np.isfinite(y_true)) or np.any(~np.isfinite(y_pred)):
        raise ValueError("Metric arrays must contain only finite values.")

    epsilon = 1e-9
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = (
        np.mean(
            np.abs(
                (y_true - y_pred)
                / np.maximum(np.abs(y_true), epsilon)
            )
        )
        * 100
    )
    bias = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape),
        "Bias": float(bias),
        "R2": float(r2),
    }


def compare_runtime(actual, formula_prediction, ml_prediction):
    """Compare formula and ML runtime outputs on identical records."""
    return {
        "formula_baseline": regression_metrics(actual, formula_prediction),
        "ml_runtime": regression_metrics(actual, ml_prediction),
    }


def save_metrics(metrics, output_file):
    """Write nested metrics as readable JSON."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)


def exact_match_accuracy(expected, predicted):
    """Return exact-match classification accuracy as a percentage."""
    expected = np.asarray(expected, dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    if expected.shape != predicted.shape or expected.size == 0:
        raise ValueError("Classification arrays must be non-empty and aligned.")
    return float(np.mean(expected == predicted) * 100.0)


def _save_figure(fig, output_file):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _identity_limits(actual, predictions):
    values = [np.asarray(actual, dtype=float)]
    values.extend(np.asarray(item, dtype=float) for item in predictions.values())
    joined = np.concatenate(values)
    lower = float(np.nanmin(joined))
    upper = float(np.nanmax(joined))
    padding = max((upper - lower) * 0.06, 0.05)
    return lower - padding, upper + padding


def plot_actual_vs_predicted(
    actual,
    predictions,
    title,
    unit,
    output_file,
    error_unit=None,
):
    """Plot one or more project series against the same baseline actuals."""
    metric_unit = unit if error_unit is None else error_unit
    actual = np.asarray(actual, dtype=float)
    series = {
        name: np.asarray(values, dtype=float)
        for name, values in predictions.items()
    }
    if not series:
        raise ValueError("At least one prediction series is required.")

    panel_count = len(series)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(6.4 * panel_count, 5.7),
        squeeze=False,
    )
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    lower, upper = _identity_limits(actual, series)

    for index, (name, predicted) in enumerate(series.items()):
        if predicted.shape != actual.shape:
            raise ValueError(f"Prediction series '{name}' is not aligned.")
        axis = axes[0, index]
        axis.scatter(
            actual,
            predicted,
            s=31,
            alpha=0.72,
            color=colors[index % len(colors)],
            edgecolors="white",
            linewidths=0.35,
        )
        axis.plot(
            [lower, upper],
            [lower, upper],
            "--",
            color="#444444",
            linewidth=1.3,
        )
        metrics = regression_metrics(actual, predicted)
        axis.text(
            0.04,
            0.96,
            (
                f"MAE = {metrics['MAE']:.3f} {metric_unit}\n"
                f"RMSE = {metrics['RMSE']:.3f} {metric_unit}\n"
                f"R² = {metrics['R2']:.3f}"
            ),
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#C8C8C8",
                "alpha": 0.92,
            },
        )
        axis.set_title(name, fontsize=12, weight="bold")
        axis.set_xlabel(f"Baseline actual ({unit})")
        axis.set_ylabel(f"Project output ({unit})")
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.22)

    fig.suptitle(title, fontsize=15, weight="bold", y=1.01)
    fig.tight_layout()
    _save_figure(fig, output_file)


def plot_runtime_metric_comparison(runtime_metrics, output_file):
    """Plot MAE/RMSE in minutes and MAPE in percent without mixed axes."""
    display_names = list(runtime_metrics.keys())
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
    x_positions = np.arange(len(display_names))
    width = 0.34
    mae = [runtime_metrics[name]["MAE"] for name in display_names]
    rmse = [runtime_metrics[name]["RMSE"] for name in display_names]
    mape = [runtime_metrics[name]["MAPE_percent"] for name in display_names]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    axes[0].bar(
        x_positions - width / 2,
        mae,
        width,
        label="MAE",
        color="#56B4E9",
    )
    axes[0].bar(
        x_positions + width / 2,
        rmse,
        width,
        label="RMSE",
        color="#0072B2",
    )
    axes[0].set_ylabel("Error (minutes) — lower is better")
    axes[0].set_title("Absolute runtime error")
    axes[0].legend(frameon=False)

    axes[1].bar(
        x_positions,
        mape,
        0.55,
        color=colors[:len(display_names)],
    )
    axes[1].set_ylabel("MAPE (%) — lower is better")
    axes[1].set_title("Relative runtime error")

    for axis in axes:
        axis.set_xticks(x_positions, display_names, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    for bars in axes[0].containers:
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].bar_label(
        axes[1].containers[0],
        fmt="%.2f",
        padding=3,
        fontsize=9,
    )
    fig.suptitle(
        "Phase-2 held-out runtime comparison (same records)",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout()
    _save_figure(fig, output_file)


def plot_residual_distributions(actual, predictions, output_file):
    """Plot signed residuals, where residual equals prediction minus actual."""
    actual = np.asarray(actual, dtype=float)
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
    fig, axis = plt.subplots(figsize=(10.5, 5.7))

    residual_sets = []
    for index, (name, values) in enumerate(predictions.items()):
        predicted = np.asarray(values, dtype=float)
        residual = predicted - actual
        residual_sets.append(residual)
        axis.hist(
            residual,
            bins=22,
            alpha=0.48,
            label=f"{name} (bias {np.mean(residual):+.2f} min)",
            color=colors[index % len(colors)],
            edgecolor="white",
        )

    joined = np.concatenate(residual_sets)
    bound = max(abs(float(np.min(joined))), abs(float(np.max(joined))))
    axis.set_xlim(-bound * 1.05, bound * 1.05)
    axis.axvline(0, color="#333333", linestyle="--", linewidth=1.4)
    axis.set_xlabel("Residual = project prediction − baseline actual (minutes)")
    axis.set_ylabel("Held-out record count")
    axis.set_title("Runtime residual distributions", fontsize=15, weight="bold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, output_file)


def plot_representative_runtime(
    record_ids,
    actual,
    formula,
    ml,
    output_file,
    sample_size=12,
):
    """Plot evenly spaced records after sorting them by actual runtime."""
    record_ids = np.asarray(record_ids, dtype=str)
    actual = np.asarray(actual, dtype=float)
    formula = np.asarray(formula, dtype=float)
    ml = np.asarray(ml, dtype=float)
    order = np.argsort(actual)
    positions = np.linspace(0, len(order) - 1, min(sample_size, len(order)))
    chosen = order[np.unique(np.rint(positions).astype(int))]
    x_positions = np.arange(len(chosen))
    width = 0.26

    fig, axis = plt.subplots(figsize=(15, 6.2))
    bars_actual = axis.bar(
        x_positions - width,
        actual[chosen],
        width,
        label="Baseline actual",
        color="#4D4D4D",
    )
    bars_formula = axis.bar(
        x_positions,
        formula[chosen],
        width,
        label="M4 operational formula",
        color="#D55E00",
    )
    bars_ml = axis.bar(
        x_positions + width,
        ml[chosen],
        width,
        label="M6 ML runtime",
        color="#0072B2",
    )
    axis.set_xticks(x_positions, record_ids[chosen], rotation=42, ha="right")
    axis.set_ylabel("Backup time (minutes)")
    axis.set_title(
        "Representative held-out records: actual vs M4 vs M6",
        fontsize=15,
        weight="bold",
    )
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    if float(np.max(actual[chosen])) < 120:
        for bars in (bars_actual, bars_formula, bars_ml):
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7, rotation=90)
    fig.tight_layout()
    _save_figure(fig, output_file)


def plot_exact_match_accuracy(accuracies, output_file):
    """Plot M9 class and recommendation exact-match accuracy."""
    labels = list(accuracies.keys())
    values = [float(accuracies[label]) for label in labels]
    colors = ["#009E73", "#CC79A7", "#0072B2", "#E69F00"]
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    bars = axis.bar(labels, values, color=colors[:len(labels)])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Exact-match accuracy (%)")
    axis.set_title("M9 output vs baseline labels", fontsize=15, weight="bold")
    axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, output_file)


def main():
    actual = np.array([10.0, 20.0, 30.0, 40.0])
    formula = np.array([11.0, 18.0, 33.0, 39.0])
    ml = np.array([10.5, 20.5, 29.0, 40.5])
    print(json.dumps(compare_runtime(actual, formula, ml), indent=2))


if __name__ == "__main__":
    main()
