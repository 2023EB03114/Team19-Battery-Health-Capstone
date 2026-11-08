from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def recommend_action(
    predicted_soh_percent,
    predicted_backup_time_min,
    required_runtime_min=None,
    data_quality_ok=True
):
    """
    Non-safety-critical rule-based decision-support module.
    """
    soh = float(predicted_soh_percent)
    backup = float(predicted_backup_time_min)

    if not data_quality_ok:
        return {
            "level": "inspect_data",
            "message": (
                "Inspect data quality before acting on the prediction."
            ),
        }

    if soh >= 90:
        level = "normal"
        message = "No immediate action needed."
    elif soh >= 80:
        level = "monitor"
        message = "Monitor battery condition more frequently."
    elif soh >= 65:
        level = "inspect"
        message = "Reduce load or schedule battery inspection."
    else:
        level = "replace"
        message = "Plan battery replacement."

    if required_runtime_min is not None:
        if backup < float(required_runtime_min):
            if soh < 65:
                level = "replace"
                message = (
                    "Backup time is below the required runtime; "
                    "plan replacement."
                )
            else:
                level = "reduce_load"
                message = (
                    "Backup time is below the required runtime; "
                    "reduce load or inspect the battery."
                )

    return {
        "level": level,
        "message": message,
    }


def classification_report_table(expected, predicted, output_name):
    """Return per-class precision/recall/F1 evidence as a tidy table."""
    expected = np.asarray(expected, dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    if expected.shape != predicted.shape or expected.size == 0:
        raise ValueError("Classification arrays must be non-empty and aligned.")
    labels = sorted(set(expected).union(predicted))
    report = classification_report(
        expected,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in labels:
        row = report[label]
        rows.append({
            "output": output_name,
            "class": label,
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1_score": float(row["f1-score"]),
            "support": int(row["support"]),
        })
    for label in ("macro avg", "weighted avg"):
        row = report[label]
        rows.append({
            "output": output_name,
            "class": label,
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1_score": float(row["f1-score"]),
            "support": int(row["support"]),
        })
    rows.append({
        "output": output_name,
        "class": "accuracy",
        "precision": float(report["accuracy"]),
        "recall": float(report["accuracy"]),
        "f1_score": float(report["accuracy"]),
        "support": int(len(expected)),
    })
    return pd.DataFrame(rows)


def confusion_count_table(expected, predicted, output_name):
    """Return a tidy expected-vs-predicted count table."""
    expected = np.asarray(expected, dtype=str)
    predicted = np.asarray(predicted, dtype=str)
    if expected.shape != predicted.shape or expected.size == 0:
        raise ValueError("Classification arrays must be non-empty and aligned.")
    labels = sorted(set(expected).union(predicted))
    counts = confusion_matrix(expected, predicted, labels=labels)
    rows = []
    for row_index, expected_label in enumerate(labels):
        for column_index, predicted_label in enumerate(labels):
            rows.append({
                "output": output_name,
                "expected_class": expected_label,
                "predicted_class": predicted_label,
                "count": int(counts[row_index, column_index]),
            })
    return pd.DataFrame(rows)


def plot_confusion_matrices(comparisons, output_file):
    """Plot class-level M9 error patterns for multiple label outputs."""
    if not comparisons:
        raise ValueError("At least one classification comparison is required.")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        len(comparisons),
        figsize=(8.0 * len(comparisons), 7.0),
        squeeze=False,
    )
    for index, (title, (expected, predicted)) in enumerate(comparisons.items()):
        expected = np.asarray(expected, dtype=str)
        predicted = np.asarray(predicted, dtype=str)
        labels = sorted(set(expected).union(predicted))
        counts = confusion_matrix(expected, predicted, labels=labels)
        axis = axes[0, index]
        image = axis.imshow(counts, cmap="Blues", vmin=0)
        short_labels = [textwrap.fill(label, 18) for label in labels]
        axis.set_xticks(range(len(labels)), short_labels, rotation=35, ha="right")
        axis.set_yticks(range(len(labels)), short_labels)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Benchmark reference label")
        axis.set_title(title, weight="bold")
        threshold = counts.max() / 2.0 if counts.size else 0.0
        for row_index in range(counts.shape[0]):
            for column_index in range(counts.shape[1]):
                count = int(counts[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if count > threshold else "#222222",
                    weight="bold",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        "M9 class-level comparison on 180 held-out benchmark records",
        fontsize=15,
        weight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if not output_file.exists() or output_file.stat().st_size == 0:
        raise OSError(f"Confusion-matrix graph was not written: {output_file}")

# ============================================================
# STANDALONE REGRESSION CHECK FOR SHARED MODULE M9
# ============================================================

def main():
    print("=" * 70)
    print("SHARED MODULE M9: RULE-BASED DECISION SUPPORT (PHASE 3 REGRESSION CHECK)")
    print("=" * 70)

    cases = [
        ("healthy", dict(predicted_soh_percent=95, predicted_backup_time_min=60), "normal"),
        ("monitor", dict(predicted_soh_percent=85, predicted_backup_time_min=60), "monitor"),
        ("inspect", dict(predicted_soh_percent=72, predicted_backup_time_min=60), "inspect"),
        ("replace", dict(predicted_soh_percent=60, predicted_backup_time_min=60), "replace"),
        (
            "runtime shortfall",
            dict(
                predicted_soh_percent=85,
                predicted_backup_time_min=35,
                required_runtime_min=45,
            ),
            "reduce_load",
        ),
        (
            "data-quality block",
            dict(
                predicted_soh_percent=95,
                predicted_backup_time_min=60,
                data_quality_ok=False,
            ),
            "inspect_data",
        ),
    ]
    checks = {}
    print("\nDecision cases:")
    for case_name, inputs, expected_level in cases:
        result = recommend_action(**inputs)
        passed = result["level"] == expected_level
        checks[case_name] = passed
        print(
            f"  {'PASS' if passed else 'FAIL'} - {case_name}: "
            f"{result['level']} | {result['message']}"
        )

    status = "PASS" if all(checks.values()) else "FAIL"

    print("\n" + "=" * 70)
    print("M9 DECISION SUPPORT STATUS:", status)
    print("=" * 70)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
