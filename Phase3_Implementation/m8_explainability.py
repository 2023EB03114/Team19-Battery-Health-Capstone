from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def built_in_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not expose feature_importances_.")

    result = pd.DataFrame({
        "feature": list(feature_names),
        "importance": model.feature_importances_,
    })

    result = result.sort_values(
        "importance",
        ascending=False
    )

    return result.reset_index(drop=True)


def permutation_feature_importance(
    model,
    X,
    y,
    random_state=42,
    n_repeats=10
):
    result = permutation_importance(
        model,
        X,
        y,
        scoring="r2",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    table = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    table = table.sort_values(
        "importance_mean",
        ascending=False
    )

    return table.reset_index(drop=True)


def shap_explanations(
    model,
    X,
    max_rows=200,
    output_dir=None,
    prefix="runtime"
):
    """
    Generate a global SHAP summary and one local
    waterfall explanation when SHAP is available.
    """
    try:
        import shap
    except Exception as error:
        return {
            "status": "skipped",
            "reason": str(error),
        }

    sample = X.iloc[:min(max_rows, len(X))].copy()

    explainer = shap.TreeExplainer(model)
    values = explainer(sample)

    result = {
        "status": "pass",
        "rows": int(len(sample)),
        "features": int(sample.shape[1]),
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        shap.summary_plot(
            values.values,
            sample,
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{prefix}_shap_summary.png",
            dpi=180,
            bbox_inches="tight"
        )
        plt.close()

        shap.plots.waterfall(
            values[0],
            show=False
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / f"{prefix}_shap_local.png",
            dpi=180,
            bbox_inches="tight"
        )
        plt.close()

    return result


def plot_feature_importance_panels(
    model_tables,
    value_column,
    output_file,
    title,
    std_column=None,
    top_n=10,
):
    """Plot comparable horizontal importance panels for one or more models."""
    if not model_tables:
        raise ValueError("At least one model importance table is required.")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1,
        len(model_tables),
        figsize=(7.0 * len(model_tables), 5.7),
        squeeze=False,
    )
    colors = ["#009E73", "#0072B2", "#D55E00", "#CC79A7"]

    for index, (model_name, table) in enumerate(model_tables.items()):
        if value_column not in table or "feature" not in table:
            raise ValueError(
                f"Importance table for {model_name} must contain feature and "
                f"{value_column}."
            )
        shown = table.nlargest(min(top_n, len(table)), value_column).sort_values(
            value_column,
            ascending=True,
        )
        errors = shown[std_column] if std_column and std_column in shown else None
        axis = axes[0, index]
        axis.barh(
            shown["feature"],
            shown[value_column],
            xerr=errors,
            color=colors[index % len(colors)],
            alpha=0.88,
            capsize=2,
        )
        axis.axvline(0.0, color="#444444", linewidth=0.8)
        axis.set_title(model_name, weight="bold")
        axis.set_xlabel(value_column.replace("_", " "))
        axis.grid(axis="x", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(title, fontsize=15, weight="bold")
    figure.tight_layout()
    figure.savefig(output_file, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    if not output_file.exists() or output_file.stat().st_size == 0:
        raise OSError(f"Importance graph was not written correctly: {output_file}")

# ============================================================
# STANDALONE REGRESSION CHECK FOR SHARED MODULE M8
# ============================================================

def main():
    print("=" * 70)
    print("SHARED MODULE M8: EXPLAINABILITY (PHASE 3 REGRESSION CHECK)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "load_power_w": rng.uniform(10.0, 60.0, 80),
        "predicted_soh_percent": rng.uniform(65.0, 100.0, 80),
        "temperature_c": rng.uniform(18.0, 42.0, 80),
        "internal_resistance_ohm": rng.uniform(0.03, 0.12, 80),
    })
    y = (
        250.0 / X["load_power_w"]
        + 0.22 * X["predicted_soh_percent"]
        - 0.08 * np.abs(X["temperature_c"] - 25.0)
        - 30.0 * X["internal_resistance_ohm"]
    )
    model = RandomForestRegressor(
        n_estimators=80,
        random_state=42,
        n_jobs=-1,
    ).fit(X, y)
    built_in = built_in_feature_importance(model, X.columns)
    permutation = permutation_feature_importance(
        model,
        X,
        y,
        random_state=42,
        n_repeats=5,
    )

    checks = {
        "built-in feature coverage": set(built_in["feature"]) == set(X.columns),
        "built-in importance sum": bool(
            np.isclose(built_in["importance"].sum(), 1.0, atol=1e-8)
        ),
        "permutation feature coverage": set(permutation["feature"]) == set(X.columns),
        "finite permutation values": bool(
            np.isfinite(permutation["importance_mean"]).all()
        ),
    }
    print("\nBuilt-in importance:")
    print(built_in.to_string(index=False))
    print("\nPermutation importance:")
    print(permutation.to_string(index=False))
    print("\nVerification checks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} - {name}")

    status = "PASS" if all(checks.values()) else "FAIL"

    print("\n" + "=" * 70)
    print("M8 EXPLAINABILITY STATUS:", status)
    print("=" * 70)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
