"""Verify the complete Phase-3 output contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    metrics_path = OUTPUT / "phase3_metrics.json"
    require(metrics_path.is_file(), "phase3_metrics.json missing")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    require(
        set(metrics["datasets"]) == {"CSGB", "NASA", "CALCE", "TRI"},
        "dataset set mismatch",
    )
    expected_rows = {"CSGB": 180, "NASA": 281, "CALCE": 1879, "TRI": 192}
    for dataset, row_count in expected_rows.items():
        data = metrics["datasets"][dataset]
        require(data["split"]["overlap_count"] == 0, f"{dataset} group overlap")
        require(data["split"]["test_rows"] == row_count, f"{dataset} holdout row mismatch")
        require(data["persistence"]["pass"], f"{dataset} persistence failed")
        require(data["m8_shap"]["status"] == "pass", f"{dataset} SHAP failed")
        require(data["feature_contract"]["target_leakage_check_pass"], f"{dataset} leakage check failed")
        prediction_path = OUTPUT / dataset.lower() / "heldout_predictions.csv"
        predictions = pd.read_csv(prediction_path)
        require(len(predictions) == row_count, f"{dataset} prediction row mismatch")
        require(predictions["record_id"].nunique() == row_count, f"{dataset} duplicate record IDs")
        for filename in (
            "metrics.json",
            "builtin_feature_importance.csv",
            "permutation_feature_importance.csv",
            "shap_global_importance.csv",
            "shap_local_runtime.csv",
            "m9_classification_report.csv",
            "m9_confusion_counts.csv",
        ):
            require((OUTPUT / dataset.lower() / filename).is_file(), f"{dataset}/{filename} missing")
        if dataset == "TRI":
            rate_table = pd.read_csv(OUTPUT / "tri" / "rate_level_metrics.csv")
            require(len(rate_table) == 8, "TRI rate-level metric row mismatch")

    graphs = sorted((OUTPUT / "graphs").glob("*.png"))
    require(len(graphs) == 22, f"expected 22 graphs, found {len(graphs)}")
    for graph in graphs:
        with Image.open(graph) as image:
            width, height = image.size
            require(width >= 800 and height >= 500, f"small graph: {graph.name} {image.size}")
            image.verify()

    models = sorted((OUTPUT / "models").glob("*.joblib"))
    require(len(models) == 8, f"expected 8 models, found {len(models)}")
    for model in models:
        require(model.stat().st_size > 10000, f"model appears empty: {model.name}")

    adapters = json.loads((OUTPUT / "source_adapter_status.json").read_text())
    require(adapters["NASA"]["quantitative_evaluation"], "NASA adapter status")
    require(adapters["CALCE"]["quantitative_evaluation"], "CALCE adapter status")
    require(adapters["TRI"]["quantitative_evaluation"], "TRI adapter status")
    require(adapters["TRI"]["physical_cells"] == 94, "TRI physical-cell count")
    require(not adapters["Oxford"]["quantitative_evaluation"], "Oxford status must remain truthful")

    inventory_rows = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():
            inventory_rows.append(
                {
                    "relative_path": path.relative_to(OUTPUT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "category": path.parts[len(OUTPUT.parts)],
                }
            )
    pd.DataFrame(inventory_rows).to_csv(OUTPUT / "output_inventory.csv", index=False)
    print("PHASE-3 OUTPUT VERIFICATION: PASS")
    print(f"Datasets: {len(expected_rows)}")
    print(f"Held-out prediction rows: {sum(expected_rows.values())}")
    print(f"Graphs: {len(graphs)}")
    print(f"Persisted models: {len(models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
