"""Execute the reproducible Phase-3 validation sequence and save its audit log."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "TEST_EXECUTION_LOG.txt"
REPORT_PATH = ROOT / "outputs" / "phase3_test_report.json"


def execute(label, arguments, log_handle, environment):
    command = [sys.executable, *arguments]
    heading = f"\n{'=' * 78}\n{label}\nCOMMAND: {' '.join(command)}\n{'=' * 78}\n"
    print(heading, end="")
    log_handle.write(heading)
    log_handle.flush()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout, end="")
    log_handle.write(result.stdout)
    status = f"\nRETURN CODE: {result.returncode}\n"
    print(status, end="")
    log_handle.write(status)
    log_handle.flush()
    return {
        "label": label,
        "arguments": arguments,
        "return_code": result.returncode,
        "pass": result.returncode == 0,
    }


def main() -> int:
    commands = [
        ("Python syntax compilation", ["-m", "compileall", "-q", "."]),
        *[
            (f"Standalone shared core M{number}", [path.name])
            for number, path in enumerate(sorted(ROOT.glob("m[1-9]_*.py")), start=1)
        ],
        ("Complete Phase-3 pipeline", ["run_phase3_validation.py"]),
        ("Output contract verification", ["verify_outputs.py"]),
        ("Automated Phase-3 tests", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="phase3_mpl_") as mpl_directory:
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = mpl_directory
        environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        with LOG_PATH.open("w", encoding="utf-8") as log_handle:
            log_handle.write("PHASE-3 COMPLETE TEST EXECUTION LOG\n")
            log_handle.write(
                "Generated: "
                + datetime.now().astimezone().isoformat(timespec="seconds")
                + "\n"
            )
            log_handle.write(f"Python executable: {sys.executable}\n")
            log_handle.write(f"Python version: {sys.version.split()[0]}\n")
            for label, arguments in commands:
                result = execute(label, arguments, log_handle, environment)
                results.append(result)
                if not result["pass"]:
                    failure = f"\nVALIDATION SUITE STATUS: FAIL ({label})\n"
                    print(failure)
                    log_handle.write(failure)
                    break
            else:
                success = "\nVALIDATION SUITE STATUS: PASS\n"
                print(success)
                log_handle.write(success)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "commands": results,
        "passed": all(item["pass"] for item in results) and len(results) == len(commands),
        "automated_test_count": 20,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Test report: {REPORT_PATH}")
    print(f"Execution log: {LOG_PATH}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
