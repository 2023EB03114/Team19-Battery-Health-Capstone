import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from m3_feature_engineering import engineer_features, get_runtime_matrix


class BackupTimeModel:

    def __init__(self, random_state=42):
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, df):
        X = get_runtime_matrix(df)
        y = df["backup_time_min"].to_numpy()
        self.model.fit(X, y)
        return self

    def predict(self, df):
        X = get_runtime_matrix(df)
        prediction = self.model.predict(X)
        return prediction


# ============================================================
# STANDALONE REGRESSION CHECK FOR SHARED MODULE M6
# ============================================================

def main():
    print("=" * 70)
    print("SHARED MODULE M6: DYNAMIC BACKUP-TIME ML (PHASE 3 REGRESSION CHECK)")
    print("=" * 70)

    rows = []
    for index in range(36):
        cycle = index * 12
        rated = 2.0
        temperature = 20.0 + (index % 9) * 2.0
        current = 0.7 + (index % 7) * 0.25
        resistance = 0.04 + cycle * 0.00006
        avg_voltage = 4.10 - cycle * 0.00065 - current * 0.02
        load = 10.0 + (index % 8) * 6.0
        predicted_soh = max(58.0, 100.0 - cycle * 0.055)
        predicted_capacity = rated * predicted_soh / 100.0
        runtime = (
            avg_voltage * predicted_capacity * 0.90 / load * 60.0
            * (1.0 - 0.0015 * abs(temperature - 25.0))
            * (1.0 - 0.6 * resistance)
        )
        rows.append({
            "cycle_count": cycle,
            "temperature_c": temperature,
            "discharge_current_a": current,
            "avg_voltage_v": avg_voltage,
            "min_voltage_v": avg_voltage - current * resistance - 0.07,
            "internal_resistance_ohm": resistance,
            "load_power_w": load,
            "rated_capacity_ah": rated,
            "measured_capacity_ah": predicted_capacity,
            "predicted_soh_percent": predicted_soh,
            "backup_time_min": runtime,
        })

    demo = engineer_features(pd.DataFrame(rows))
    model = BackupTimeModel(random_state=42).fit(demo)
    prediction = model.predict(demo)

    missing_feature_pass = False
    try:
        get_runtime_matrix(demo.drop(columns=["load_power_w"]))
    except ValueError:
        missing_feature_pass = True

    checks = {
        "prediction row count": len(prediction) == len(demo),
        "finite predictions": bool(np.isfinite(prediction).all()),
        "positive predictions": bool((prediction > 0).all()),
        "missing-feature rejection": missing_feature_pass,
    }
    print(f"\nDemonstration records: {len(demo)}")
    print("Predicted backup time preview:", prediction[:5].round(3).tolist(), "min")
    print("\nVerification checks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} - {name}")

    status = "PASS" if all(checks.values()) else "FAIL"

    print("\n" + "=" * 70)
    print("M6 DYNAMIC BACKUP-TIME STATUS:", status)
    print("=" * 70)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
