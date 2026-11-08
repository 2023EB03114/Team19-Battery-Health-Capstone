from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST_SHA256.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_files() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        result[relative.as_posix()] = path
    return result


def csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def main() -> int:
    failures: list[str] = []
    if not MANIFEST.exists():
        print("FAIL: MANIFEST_SHA256.csv is missing.")
        return 1

    expected: dict[str, tuple[int, str]] = {}
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected[row["relative_path"]] = (int(row["size_bytes"]), row["sha256"])

    actual = included_files()
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        failures.append(f"Missing files: {missing}")
    if unexpected:
        failures.append(f"Unexpected files: {unexpected}")

    for relative in sorted(set(expected) & set(actual)):
        size, digest = expected[relative]
        path = actual[relative]
        if path.stat().st_size != size:
            failures.append(f"Size mismatch: {relative}")
        elif sha256(path) != digest:
            failures.append(f"SHA-256 mismatch: {relative}")

    graph_count = len(list((ROOT / "Phase3_Implementation/outputs/graphs").glob("*.png")))
    model_count = len(list((ROOT / "Phase3_Implementation/outputs/models").glob("*.joblib")))
    heldout_rows = sum(
        csv_data_rows(ROOT / f"Phase3_Implementation/outputs/{name}/heldout_predictions.csv")
        for name in ("csgb", "nasa", "calce", "tri")
    )
    if graph_count != 22:
        failures.append(f"Graph count is {graph_count}; expected 22.")
    if model_count != 8:
        failures.append(f"Model count is {model_count}; expected 8.")
    if heldout_rows != 2532:
        failures.append(f"Held-out row count is {heldout_rows}; expected 2532.")

    if failures:
        print("PHASE-3 SUBMISSION PACKAGE VERIFICATION: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PHASE-3 SUBMISSION PACKAGE VERIFICATION: PASS")
    print(f"Manifest entries: {len(expected)}")
    print(f"Graphs: {graph_count}")
    print(f"Persisted models: {model_count}")
    print(f"Held-out prediction rows: {heldout_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
