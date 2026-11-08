from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError:
    AppTest = None

from dashboard_backend import predict_condition, smoke_prediction, validate_condition
from m3_feature_engineering import HEALTH_FEATURES, RUNTIME_FEATURES
from m4_formula_baseline import formula_backup_time_minutes
from run_phase3_validation import load_csgb, load_public, split_dataset
from source_adapters import _oxford_records_from_mapping, oxford_contract_fixture


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"


class Phase3SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads((OUTPUT / "phase3_metrics.json").read_text())

    def test_01_csgb_continuity_dataset(self):
        frame, quality = load_csgb()
        self.assertEqual(len(frame), 720)
        self.assertEqual(frame["battery_id"].nunique(), 12)
        self.assertTrue(quality["valid"])

    def test_02_nasa_real_dataset(self):
        frame, quality = load_public("NASA")
        self.assertEqual(len(frame), 579)
        self.assertEqual(set(frame["battery_id"]), {"B0005", "B0006", "B0007", "B0018"})
        self.assertTrue(quality["valid"])

    def test_03_calce_real_dataset(self):
        frame, quality = load_public("CALCE")
        self.assertEqual(len(frame), 3544)
        self.assertEqual(set(frame["battery_id"]), {"CS2_35", "CS2_36", "CS2_37", "CS2_38"})
        self.assertTrue(quality["valid"])

    def test_04_tri_real_multirate_dataset(self):
        frame, quality = load_public("TRI")
        self.assertEqual(len(frame), 750)
        self.assertEqual(frame["battery_id"].nunique(), 94)
        self.assertEqual(frame["source_rate_label"].nunique(), 8)
        self.assertEqual(quality["source_quality"]["complete_eight_rate_cells"], 93)
        self.assertEqual(quality["source_quality"]["incomplete_cells"], {"0000ED": 6})
        self.assertTrue(quality["valid"])

    def test_05_all_grouped_splits_have_zero_overlap(self):
        for dataset, loader in (
            ("CSGB", load_csgb),
            ("NASA", lambda: load_public("NASA")),
            ("CALCE", lambda: load_public("CALCE")),
            ("TRI", lambda: load_public("TRI")),
        ):
            frame, _ = loader()
            train, test, integrity = split_dataset(dataset, frame)
            self.assertFalse(train.empty)
            self.assertFalse(test.empty)
            self.assertEqual(integrity["overlap_count"], 0)

    def test_06_oxford_native_contract_fixture(self):
        frame = _oxford_records_from_mapping(
            oxford_contract_fixture(), "Oxford_contract_fixture.mat"
        )
        self.assertEqual(len(frame), 1)
        self.assertAlmostEqual(frame.iloc[0]["measured_capacity_ah"], 0.72, places=10)
        self.assertTrue(np.isnan(frame.iloc[0]["internal_resistance_ohm"]))

    def test_07_target_leakage_exclusion(self):
        targets = {"measured_capacity_ah", "backup_time_min", "actual_soh_pct"}
        self.assertTrue(targets.isdisjoint(HEALTH_FEATURES))
        self.assertTrue(targets.isdisjoint(RUNTIME_FEATURES))

    def test_08_formula_known_answer(self):
        result = formula_backup_time_minutes(12.0, 10.0, 48.0, 0.90, 0.80)
        self.assertAlmostEqual(float(result), 108.0, places=10)

    def test_09_phase3_metric_domains_are_separated(self):
        self.assertEqual(
            self.metrics["datasets"]["CSGB"]["evidence_domain"],
            "controlled synthetic",
        )
        self.assertEqual(
            self.metrics["datasets"]["NASA"]["evidence_domain"],
            "real experimental",
        )
        self.assertEqual(
            self.metrics["datasets"]["CALCE"]["evidence_domain"],
            "real experimental",
        )
        self.assertEqual(
            self.metrics["datasets"]["TRI"]["evidence_domain"],
            "real experimental",
        )
        self.assertFalse(self.metrics["evidence_boundary"]["field_validation"])

    def test_10_prediction_export_contract(self):
        expected = {"csgb": 180, "nasa": 281, "calce": 1879, "tri": 192}
        for dataset, rows in expected.items():
            frame = pd.read_csv(OUTPUT / dataset / "heldout_predictions.csv")
            self.assertEqual(len(frame), rows)
            self.assertEqual(frame["record_id"].nunique(), rows)
            self.assertTrue(np.isfinite(frame["predicted_capacity_ah"]).all())
            self.assertTrue(np.isfinite(frame["ml_backup_time_min"]).all())

    def test_11_persistence_reload_contract(self):
        for dataset in ("csgb", "nasa", "calce", "tri"):
            capacity = joblib.load(OUTPUT / "models" / f"{dataset}_capacity_model.joblib")
            runtime = joblib.load(OUTPUT / "models" / f"{dataset}_runtime_model.joblib")
            self.assertTrue(hasattr(capacity, "predict_capacity"))
            self.assertTrue(hasattr(runtime, "predict"))
            self.assertTrue(self.metrics["datasets"][dataset.upper()]["persistence"]["pass"])

    def test_12_shap_outputs_and_dimensions(self):
        for dataset in ("CSGB", "NASA", "CALCE", "TRI"):
            status = self.metrics["datasets"][dataset]["m8_shap"]
            self.assertEqual(status["status"], "pass")
            self.assertEqual(status["runtime_features"], len(RUNTIME_FEATURES))
            table = pd.read_csv(OUTPUT / dataset.lower() / "shap_global_importance.csv")
            self.assertEqual(set(table["model"]), {"M5 capacity", "M6 backup time"})

    def test_13_graph_and_model_inventory(self):
        self.assertEqual(len(list((OUTPUT / "graphs").glob("*.png"))), 22)
        self.assertEqual(len(list((OUTPUT / "models").glob("*.joblib"))), 8)

    def test_14_dashboard_backend_smoke(self):
        result = smoke_prediction()
        self.assertGreater(result["predicted_capacity_ah"], 0)
        self.assertGreater(result["formula_backup_time_min"], 0)
        self.assertGreater(result["ml_backup_time_min"], 0)
        self.assertEqual(result["input_warnings"], [])
        self.assertIn(result["decision_level"], {"normal", "monitor", "inspect", "reduce_load", "replace"})

    def test_15_dashboard_input_guards(self):
        with self.assertRaises(ValueError):
            validate_condition({})
        with self.assertRaises(ValueError):
            predict_condition(
                {
                    "cycle_count": 1,
                    "temperature_c": 25,
                    "discharge_current_a": 1,
                    "avg_voltage_v": 3.5,
                    "min_voltage_v": 4.0,
                    "internal_resistance_ohm": 0.05,
                    "load_power_w": 4,
                    "rated_capacity_ah": 2,
                }
            )

    def test_16_dashboard_power_consistency_warning(self):
        result = predict_condition(
            {
                "cycle_count": 10,
                "temperature_c": 25,
                "discharge_current_a": 1,
                "avg_voltage_v": 3.8,
                "min_voltage_v": 3.4,
                "internal_resistance_ohm": 0.05,
                "load_power_w": 30,
                "rated_capacity_ah": 2,
            }
        )
        self.assertTrue(result["input_warnings"])

    def test_17_streamlit_headless_app(self):
        if AppTest is None:
            requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
            self.assertIn("streamlit==", requirements)
            compile((ROOT / "app.py").read_text(encoding="utf-8"), "app.py", "exec")
            self.skipTest(
                "Streamlit test harness is not installed in this build runtime; "
                "install requirements.txt to execute AppTest."
            )
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 6)
        self.assertIn("Explainable Battery Health", app.title[0].value)

    def test_18_real_runtime_result_is_not_overclaimed(self):
        # The real-data finding is deliberately preserved: M4 is slightly
        # better than M6 on these constrained constant-current protocols.
        for dataset in ("NASA", "CALCE"):
            data = self.metrics["datasets"][dataset]
            self.assertLess(
                data["m4_operational_runtime"]["MAE"],
                data["m6_ml_runtime"]["MAE"],
            )

    def test_19_adapter_status_is_truthful(self):
        status = self.metrics["adapter_status"]
        self.assertTrue(status["NASA"]["quantitative_evaluation"])
        self.assertTrue(status["CALCE"]["quantitative_evaluation"])
        self.assertTrue(status["TRI"]["quantitative_evaluation"])
        self.assertEqual(status["TRI"]["physical_cells"], 94)
        self.assertFalse(status["Oxford"]["quantitative_evaluation"])

    def test_20_tri_rate_level_metrics(self):
        table = pd.read_csv(OUTPUT / "tri" / "rate_level_metrics.csv")
        self.assertEqual(len(table), 8)
        self.assertEqual(table["source_rate_label"].nunique(), 8)
        self.assertEqual(set(table["records"]), {24})
        self.assertEqual(
            set(table["lower_runtime_mae_method"]),
            {"M4 formula", "M6 ML"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
