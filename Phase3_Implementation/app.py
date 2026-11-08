"""Streamlit user interface for the Phase-3 battery-health prototype."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard_backend import INPUT_FIELDS, predict_condition, predict_uploaded_frame


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"


st.set_page_config(
    page_title="Battery Health and Backup-Time Framework",
    page_icon="🔋",
    layout="wide",
)


@st.cache_data
def load_metrics():
    return json.loads((OUTPUT_DIR / "phase3_metrics.json").read_text(encoding="utf-8"))


def display_graph(filename: str, caption: str) -> None:
    path = GRAPH_DIR / filename
    st.image(str(path), caption=caption, width="stretch")


def main() -> None:
    metrics = load_metrics()
    st.title("Explainable Battery Health and Dynamic Backup-Time Prediction")
    st.warning(
        "Prototype decision support only. CSGB is controlled synthetic evidence; "
        "NASA and CALCE are laboratory data under constrained discharge protocols. "
        "No field validation or safety certification is claimed."
    )
    tabs = st.tabs(
        [
            "Overview",
            "Data Quality",
            "Health and Runtime",
            "Explainability",
            "New Condition",
            "Downloads",
        ]
    )

    with tabs[0]:
        st.subheader("Phase-3 evidence domains")
        columns = st.columns(3)
        for column, dataset in zip(columns, ["CSGB", "NASA", "CALCE"]):
            data = metrics["datasets"][dataset]
            column.metric(
                f"{dataset} grouped holdout",
                f"{data['split']['test_rows']:,} rows",
                f"{len(data['split']['test_groups'])} physical groups",
            )
            column.caption(data["evidence_domain"])
        display_graph(
            "01_dataset_record_coverage.png",
            "Development and grouped-holdout record coverage.",
        )
        display_graph(
            "02_grouped_split_integrity.png",
            "Physical battery/cell overlap is zero in every evaluation.",
        )

    with tabs[1]:
        st.subheader("Upload condition rows for validation and prediction")
        st.write("Required columns: " + ", ".join(INPUT_FIELDS))
        uploaded = st.file_uploader("CSV file", type=["csv"])
        efficiency = st.slider("Formula efficiency", 0.50, 1.00, 0.90, 0.01)
        required_runtime = st.number_input(
            "Required runtime (min)", min_value=1.0, value=45.0, step=1.0
        )
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                st.dataframe(frame.head(20), width="stretch")
                predictions = predict_uploaded_frame(
                    frame,
                    efficiency_fraction=efficiency,
                    required_runtime_min=required_runtime,
                )
            except Exception as error:
                st.error(str(error))
            else:
                st.success(f"Validated and predicted {len(predictions)} rows.")
                st.dataframe(predictions, width="stretch")
                st.download_button(
                    "Download uploaded-row predictions",
                    predictions.to_csv(index=False).encode("utf-8"),
                    "dashboard_uploaded_predictions.csv",
                    "text/csv",
                )

    with tabs[2]:
        st.subheader("Grouped holdout results")
        rows = []
        for dataset, data in metrics["datasets"].items():
            rows.append(
                {
                    "Dataset": dataset,
                    "Evidence": data["evidence_domain"],
                    "Capacity MAE (Ah)": data["m5_capacity"]["MAE"],
                    "SOH MAE (pp)": data["m5_soh"]["MAE"],
                    "M4 runtime MAE (min)": data["m4_operational_runtime"]["MAE"],
                    "M6 runtime MAE (min)": data["m6_ml_runtime"]["MAE"],
                    "M6 runtime R2": data["m6_ml_runtime"]["R2"],
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.info(
            "M6 outperforms M4 on CSGB, while M4 slightly outperforms M6 on "
            "the real NASA and CALCE constant-current protocols."
        )
        display_graph(
            "03_capacity_actual_vs_predicted.png",
            "Remaining-capacity actual-versus-predicted comparison.",
        )
        display_graph(
            "05_runtime_actual_vs_predicted.png",
            "Formula and ML runtime comparison on identical holdout records.",
        )
        display_graph(
            "09_capacity_degradation_profiles.png",
            "Degradation profiles for each physical battery/cell.",
        )

    with tabs[3]:
        st.subheader("Global and local model explanations")
        st.caption(
            "Importance and SHAP values describe fitted model behaviour; they do "
            "not establish causal electrochemical effects."
        )
        display_graph(
            "10_builtin_feature_importance.png",
            "Built-in Random Forest importance for M5 and M6.",
        )
        display_graph(
            "11_permutation_feature_importance.png",
            "Ten-repeat grouped-holdout permutation importance.",
        )
        display_graph(
            "12_shap_mean_absolute_importance.png",
            "Mean absolute SHAP importance.",
        )
        dataset = st.selectbox("SHAP dataset", ["CSGB", "NASA", "CALCE"])
        numbers = {"CSGB": (13, 16), "NASA": (14, 17), "CALCE": (15, 18)}
        global_number, local_number = numbers[dataset]
        display_graph(
            f"{global_number:02d}_{dataset.lower()}_runtime_shap_summary.png",
            f"{dataset} global runtime SHAP summary.",
        )
        display_graph(
            f"{local_number:02d}_{dataset.lower()}_runtime_shap_local.png",
            f"{dataset} representative local runtime explanation.",
        )

    with tabs[4]:
        st.subheader("Predict a new operating condition")
        with st.form("new_condition_form"):
            col1, col2 = st.columns(2)
            cycle_count = col1.number_input("Cycle count", min_value=0, value=120)
            temperature_c = col2.number_input("Temperature (C)", value=25.0)
            discharge_current_a = col1.number_input(
                "Discharge current (A)", min_value=0.01, value=1.0
            )
            avg_voltage_v = col2.number_input(
                "Average voltage (V)", min_value=2.0, max_value=5.0, value=3.85
            )
            min_voltage_v = col1.number_input(
                "Minimum voltage (V)", min_value=1.5, max_value=5.0, value=3.45
            )
            internal_resistance_ohm = col2.number_input(
                "Internal resistance (ohm)", min_value=0.00001, value=0.055, format="%.5f"
            )
            load_power_w = col1.number_input(
                "Load power (W)", min_value=0.01, value=4.0
            )
            rated_capacity_ah = col2.number_input(
                "Rated/reference capacity (Ah)", min_value=0.01, value=2.0
            )
            efficiency_fraction = col1.slider(
                "Formula efficiency", 0.50, 1.00, 0.90, 0.01, key="new_efficiency"
            )
            required_runtime_min = col2.number_input(
                "Required runtime (min)", min_value=1.0, value=45.0, key="new_runtime"
            )
            submitted = st.form_submit_button("Predict")
        if submitted:
            try:
                result = predict_condition(
                    {
                        "cycle_count": cycle_count,
                        "temperature_c": temperature_c,
                        "discharge_current_a": discharge_current_a,
                        "avg_voltage_v": avg_voltage_v,
                        "min_voltage_v": min_voltage_v,
                        "internal_resistance_ohm": internal_resistance_ohm,
                        "load_power_w": load_power_w,
                        "rated_capacity_ah": rated_capacity_ah,
                    },
                    efficiency_fraction=efficiency_fraction,
                    required_runtime_min=required_runtime_min,
                )
            except Exception as error:
                st.error(str(error))
            else:
                a, b, c, d = st.columns(4)
                a.metric("Capacity", f"{result['predicted_capacity_ah']:.3f} Ah")
                b.metric("SOH", f"{result['predicted_soh_percent']:.1f}%")
                c.metric("Formula runtime", f"{result['formula_backup_time_min']:.1f} min")
                d.metric("ML runtime", f"{result['ml_backup_time_min']:.1f} min")
                st.info(result["decision_message"])
                for warning in result["input_warnings"]:
                    st.warning(warning)
                st.warning(result["warning"])

    with tabs[5]:
        st.subheader("Evidence downloads")
        for relative_path in [
            "phase3_metrics.json",
            "phase3_metrics_table.csv",
            "csgb/heldout_predictions.csv",
            "nasa/heldout_predictions.csv",
            "calce/heldout_predictions.csv",
            "source_adapter_status.json",
        ]:
            path = OUTPUT_DIR / relative_path
            mime = "text/csv" if path.suffix == ".csv" else "application/json"
            st.download_button(
                f"Download {relative_path}",
                path.read_bytes(),
                path.name,
                mime,
                key=relative_path,
            )


if __name__ == "__main__":
    main()
