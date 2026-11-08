import numpy as np


def formula_backup_time_minutes(
    voltage_v,
    capacity_ah,
    load_power_w,
    efficiency=0.90,
    usable_fraction=1.0,
):
    """
    Formula-based backup-time baseline.

    T_minutes =
        V * C * usable_fraction * efficiency
        ------------------------------------ * 60
                    Load Power
    """
    voltage = np.asarray(voltage_v, dtype=float)
    capacity = np.asarray(capacity_ah, dtype=float)
    load = np.asarray(load_power_w, dtype=float)
    efficiency_array = np.asarray(efficiency, dtype=float)
    usable_array = np.asarray(usable_fraction, dtype=float)

    try:
        voltage, capacity, load, efficiency_array, usable_array = (
            np.broadcast_arrays(
                voltage,
                capacity,
                load,
                efficiency_array,
                usable_array,
            )
        )
    except ValueError as exc:
        raise ValueError(
            "M4 inputs must be scalars or broadcast-compatible arrays."
        ) from exc

    arrays = {
        "voltage": voltage,
        "capacity": capacity,
        "load power": load,
        "efficiency": efficiency_array,
        "usable fraction": usable_array,
    }
    for name, values in arrays.items():
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{name.title()} must contain finite values.")

    if np.any((efficiency_array <= 0) | (efficiency_array > 1)):
        raise ValueError("Efficiency must be between 0 and 1.")

    if np.any((usable_array <= 0) | (usable_array > 1)):
        raise ValueError("Usable fraction must be between 0 and 1.")

    if np.any(voltage <= 0):
        raise ValueError("Voltage must be positive.")

    if np.any(capacity <= 0):
        raise ValueError("Capacity must be positive.")

    if np.any(load <= 0):
        raise ValueError("Load power must be positive.")

    backup_time_minutes = (
        voltage
        * capacity
        * usable_array
        * efficiency_array
        / load
    ) * 60

    return backup_time_minutes


# ============================================================
# STANDALONE REGRESSION CHECK FOR SHARED MODULE M4
# ============================================================

def main():
    print("=" * 70)
    print("SHARED MODULE M4: FORMULA BASELINE (PHASE 3 REGRESSION CHECK)")
    print("=" * 70)

    predicted = formula_backup_time_minutes(
        voltage_v=12.0,
        capacity_ah=10.0,
        load_power_w=48.0,
        efficiency=0.90,
        usable_fraction=0.80,
    )
    expected = 108.0
    known_answer_pass = bool(np.isclose(float(predicted), expected, atol=1e-10))

    vector = formula_backup_time_minutes(
        voltage_v=[3.8, 3.7],
        capacity_ah=[2.0, 1.8],
        load_power_w=[20.0, 30.0],
        efficiency=0.90,
    )
    vector_pass = vector.shape == (2,) and bool(np.all(vector > 0))

    invalid_input_pass = False
    try:
        formula_backup_time_minutes(3.8, 2.0, 0.0)
    except ValueError:
        invalid_input_pass = True

    checks = {
        "known-answer calculation": known_answer_pass,
        "vector input calculation": vector_pass,
        "zero-load rejection": invalid_input_pass,
    }
    print(f"\nKnown-answer output: {float(predicted):.3f} min")
    print(f"Expected output    : {expected:.3f} min")
    print("Vector outputs     :", np.round(vector, 3).tolist())
    print("\nVerification checks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} - {name}")

    status = "PASS" if all(checks.values()) else "FAIL"

    print("\n" + "=" * 70)
    print("M4 FORMULA BASELINE STATUS:", status)
    print("=" * 70)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
