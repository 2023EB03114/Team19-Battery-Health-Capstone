import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from m3_feature_engineering import engineer_features, get_health_matrix


class CapacitySOHModel:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, df):
        X = get_health_matrix(df)
        y = df["measured_capacity_ah"].to_numpy()
        self.model.fit(X, y)
        return self

    def predict_capacity(self, df):
        X = get_health_matrix(df)
        return self.model.predict(X)

    def predict_soh(self, df):
        capacity = self.predict_capacity(df)
        rated_capacity = df["rated_capacity_ah"].to_numpy(dtype=float)
        soh = capacity / rated_capacity * 100.0
        return soh

    def oof_predictions_for_runtime(
        self,
        train_df,
        group_col="battery_id",
        max_splits=5
    ):
        """
        Produce out-of-fold SOH predictions for training M6.
        This avoids using in-sample capacity predictions as
        runtime-model inputs.
        """
        groups = train_df[group_col].astype(str).to_numpy()
        unique_groups = np.unique(groups)

        if len(unique_groups) < 2:
            raise ValueError("At least two battery groups are required.")

        n_splits = min(max_splits, len(unique_groups))
        cv = GroupKFold(n_splits=n_splits)

        X = get_health_matrix(train_df)
        y = train_df["measured_capacity_ah"].to_numpy()

        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )

        oof_capacity = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            groups=groups,
            n_jobs=-1
        )

        rated = train_df["rated_capacity_ah"].to_numpy(dtype=float)
        oof_soh = oof_capacity / rated * 100

        return oof_capacity, oof_soh


# ============================================================
# STANDALONE REGRESSION CHECK FOR SHARED MODULE M5
# ============================================================

def main():
    print("=" * 70)
    print("SHARED MODULE M5: CAPACITY AND SOH ASSESSMENT (PHASE 3 REGRESSION CHECK)")
    print("=" * 70)

    rows = []
    for battery_index in range(4):
        # Keep the demonstration reference capacity fixed so the synthetic
        # example remains physically interpretable (predicted SOH <= 100%).
        rated = 2.0
        for sample_index in range(8):
            cycle = sample_index * 40 + battery_index * 3
            temperature = 23.0 + sample_index * 0.8
            current = 0.8 + sample_index * 0.12
            resistance = 0.04 + 0.00008 * cycle + battery_index * 0.002
            avg_voltage = 4.10 - 0.0007 * cycle - current * 0.015
            capacity = rated * (
                1.0
                - 0.00055 * cycle
                - 0.0012 * abs(temperature - 25.0)
                - 0.005 * battery_index
            )
            rows.append({
                "battery_id": f"M5_DEMO_{battery_index + 1}",
                "cycle_count": cycle,
                "temperature_c": temperature,
                "discharge_current_a": current,
                "avg_voltage_v": avg_voltage,
                "min_voltage_v": avg_voltage - current * resistance - 0.08,
                "internal_resistance_ohm": resistance,
                "load_power_w": avg_voltage * current,
                "rated_capacity_ah": rated,
                "measured_capacity_ah": capacity,
                "backup_time_min": 20.0,
            })

    demo = engineer_features(pd.DataFrame(rows))
    model = CapacitySOHModel(random_state=42)
    oof_capacity, oof_soh = model.oof_predictions_for_runtime(
        demo,
        group_col="battery_id",
        max_splits=4,
    )
    model.fit(demo)
    predicted_capacity = model.predict_capacity(demo)
    predicted_soh = model.predict_soh(demo)
    expected_soh = predicted_capacity / demo["rated_capacity_ah"].to_numpy() * 100.0

    checks = {
        "prediction row count": len(predicted_capacity) == len(demo),
        "finite capacity predictions": bool(np.isfinite(predicted_capacity).all()),
        "SOH equation": bool(np.allclose(predicted_soh, expected_soh)),
        "grouped OOF row count": len(oof_capacity) == len(demo),
        "finite grouped OOF SOH": bool(np.isfinite(oof_soh).all()),
    }

    print(f"\nDemonstration records : {len(demo)}")
    print(f"Battery groups       : {demo['battery_id'].nunique()}")
    print(f"Predicted capacity   : {predicted_capacity[:3].round(4).tolist()} Ah")
    print(f"Predicted SOH        : {predicted_soh[:3].round(2).tolist()} %")
    print("\nVerification checks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} - {name}")

    status = "PASS" if all(checks.values()) else "FAIL"

    print("\n" + "=" * 70)
    print("M5 CAPACITY AND SOH STATUS:", status)
    print("=" * 70)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
