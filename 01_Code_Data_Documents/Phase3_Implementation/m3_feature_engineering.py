import numpy as np
import pandas as pd


HEALTH_FEATURES = [
    "cycle_count",
    "temperature_c",
    "discharge_current_a",
    "avg_voltage_v",
    "min_voltage_v",
    "internal_resistance_ohm",
    "c_rate",
    "thermal_stress",
    "resistance_voltage_drop_v",
    "voltage_sag_v",
]

RUNTIME_FEATURES = [
    "cycle_count",
    "temperature_c",
    "discharge_current_a",
    "avg_voltage_v",
    "min_voltage_v",
    "internal_resistance_ohm",
    "load_power_w",
    "c_rate",
    "thermal_stress",
    "resistance_voltage_drop_v",
    "voltage_sag_v",
    "predicted_soh_percent",
]


def engineer_features(df):
    """
    Generate leakage-safe engineering features.

    Important:
    measured_capacity_ah and backup_time_min
    are NOT used as engineered input features.
    """
    out = df.copy()
    eps = 1e-9

    out["c_rate"] = (
        out["discharge_current_a"]
        / (out["rated_capacity_ah"] + eps)
    )

    out["thermal_stress"] = (
        out["temperature_c"] - 25.0
    ).abs()

    out["resistance_voltage_drop_v"] = (
        out["discharge_current_a"]
        * out["internal_resistance_ohm"]
    )

    out["voltage_sag_v"] = (
        out["avg_voltage_v"]
        - out["min_voltage_v"]
    ).clip(lower=0)

    out["power_from_vi_w"] = (
        out["avg_voltage_v"]
        * out["discharge_current_a"]
    )

    return out


def get_health_matrix(df):
    return df[HEALTH_FEATURES].copy()


def get_runtime_matrix(df):
    missing = [
        c for c in RUNTIME_FEATURES
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Runtime features missing: {missing}")

    return df[RUNTIME_FEATURES].copy()

# ============================================================
# STANDALONE REGRESSION CHECK FOR SHARED MODULE M3
# ============================================================

def main():
    print("=" * 70)
    print("SHARED MODULE M3: FEATURE ENGINEERING (PHASE 3 REGRESSION CHECK)")
    print("=" * 70)

    demo = pd.DataFrame({
        "cycle_count": [10, 50],
        "temperature_c": [25.0, 35.0],
        "discharge_current_a": [1.0, 2.0],
        "avg_voltage_v": [3.90, 3.75],
        "min_voltage_v": [3.55, 3.20],
        "internal_resistance_ohm": [0.05, 0.08],
        "load_power_w": [20.0, 35.0],
        "rated_capacity_ah": [2.0, 2.0],
        "measured_capacity_ah": [1.90, 1.65],
        "backup_time_min": [20.0, 9.0],
        "predicted_soh_percent": [94.0, 81.0],
    })
    engineered = engineer_features(demo)
    health_matrix = get_health_matrix(engineered)
    runtime_matrix = get_runtime_matrix(engineered)

    expected_c_rate = np.array([0.5, 1.0])
    expected_voltage_drop = np.array([0.05, 0.16])
    leakage_targets = {"measured_capacity_ah", "backup_time_min"}
    checks = {
        "c_rate calculation": np.allclose(
            engineered["c_rate"], expected_c_rate, atol=1e-8
        ),
        "resistance drop calculation": np.allclose(
            engineered["resistance_voltage_drop_v"],
            expected_voltage_drop,
        ),
        "non-negative voltage sag": bool((engineered["voltage_sag_v"] >= 0).all()),
        "health target exclusion": leakage_targets.isdisjoint(health_matrix.columns),
        "runtime target exclusion": leakage_targets.isdisjoint(runtime_matrix.columns),
    }

    print("\nEngineered feature preview:")
    print(
        engineered[
            [
                "c_rate",
                "thermal_stress",
                "resistance_voltage_drop_v",
                "voltage_sag_v",
                "power_from_vi_w",
            ]
        ].to_string(index=False)
    )
    print("\nVerification checks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} - {name}")

    status = "PASS" if all(checks.values()) else "FAIL"

    print("\n" + "=" * 70)
    print("M3 FEATURE ENGINEERING STATUS:", status)
    print("=" * 70)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
