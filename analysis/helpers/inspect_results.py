"""Read-only commands for inputs, task summaries, CSV previews, and consistency.

Examples (from the package root, with its virtual environment active):
    python -m analysis.helpers.inspect_results inputs
    python -m analysis.helpers.inspect_results summary 4
    python -m analysis.helpers.inspect_results table 2 top10_volume_od --limit 5
    python -m analysis.helpers.inspect_results check
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from analysis.common import PROJECT_ROOT, RAW_DIR, RESULTS_DIR, expected_raw_files


def task_dir(number: int) -> Path:
    return RESULTS_DIR / f"Task_{number}"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def print_inputs() -> None:
    print(f"Expected raw data directory: {RAW_DIR}")
    for name, path in expected_raw_files().items():
        state = "present" if path.is_file() else "MISSING"
        print(f"  {state:7} {name}")
    print("Raw files are deliberately absent from the submission package.")


def print_summary(number: int, full: bool) -> None:
    path = task_dir(number) / "summary.json"
    data = load_json(path)
    if full:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"Task {number}: {path.relative_to(PROJECT_ROOT)}")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} entries")
            elif isinstance(value, dict):
                print(f"  {key}: {len(value)} keys")
            else:
                print(f"  {key}: {value}")


def print_table(number: int, name: str, limit: int) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError(
            "Table name must contain only letters, numbers, and underscores"
        )
    path = task_dir(number) / "tables" / f"{name}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        print(f"{path.relative_to(PROJECT_ROOT)}")
        print(" | ".join(reader.fieldnames or []))
        for index, row in enumerate(reader):
            if index >= limit:
                break
            print(" | ".join(row.get(key, "") for key in reader.fieldnames or []))


def check() -> None:
    """Check recorded outputs and cross-task count reconciliation without raw data."""
    for number in range(1, 6):
        folder = task_dir(number)
        required = [folder / "summary.json", folder / "report.tex", folder / "run.json"]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise AssertionError(f"Task {number} has a missing or empty required file")
        if not list((folder / "tables").glob("*.csv")):
            raise AssertionError(f"Task {number} has no processed CSV")
        if not list((folder / "figures").glob("*.png")):
            raise AssertionError(f"Task {number} has no figure")
        run = load_json(folder / "run.json")
        if run.get("exit_code") != 0:
            raise AssertionError(f"Task {number} recorded a failed run")

    t1 = load_json(task_dir(1) / "summary.json")
    t3 = load_json(task_dir(3) / "summary.json")
    t4 = load_json(task_dir(4) / "summary.json")
    t5 = load_json(task_dir(5) / "summary.json")
    count_rows = t1["task1_2_counts"]
    statuses = ("N/N", "Y/N", "Y/Y", "N/Y", "Missing/other")
    for month, column in (("2026-04", "april_trips"), ("2026-05", "may_trips")):
        hv = next(int(row[column]) for row in count_rows if row["service"] == "HVFHV")
        company_total = sum(
            int(row[column])
            for row in count_rows
            if row.get("license_num")
            and row["request_match_category"] == "All retained trips"
        )
        if company_total != hv:
            raise AssertionError(f"Task 1 company totals fail for {month}")
        for service in ("Uber", "Lyft"):
            observed = next(
                row
                for row in t3["monthly"]
                if row["month"] == month and row["service"] == service
            )
            actual = {
                row["request_match_category"]: int(row[column])
                for row in count_rows
                if row["service"] == service
            }
            if (
                sum(actual[status] for status in statuses)
                != observed["completed_trips"]
            ):
                raise AssertionError(
                    f"Task 1/3 status counts fail for {month} {service}"
                )
    for row in t4["month_summary"]:
        if row["reference_supported"] > row["shared_eligible"]:
            raise AssertionError("Task 4 supported count exceeds eligible trips")
    for row in t5["comparison"]:
        if row["shared_supported"] > row["shared_eligible"]:
            raise AssertionError("Task 5 supported count exceeds eligible trips")
        if (
            not row["ci95_low_usd"]
            <= row["mean_difference_usd"]
            <= row["ci95_high_usd"]
        ):
            raise AssertionError("Task 5 estimate falls outside its reported interval")
    print("PASS: five task bundles, run records, and Task 1/3/4/5 reconciliations")
    print(
        "Note: this check reads packaged outputs; rerun task scripts to verify raw-data calculations."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inputs", help="show the four expected raw file locations")
    summary = sub.add_parser("summary", help="inspect a task summary.json")
    summary.add_argument("task", type=int, choices=range(1, 6))
    summary.add_argument("--full", action="store_true")
    table = sub.add_parser("table", help="preview a processed task CSV")
    table.add_argument("task", type=int, choices=range(1, 6))
    table.add_argument("name", help="CSV basename without .csv")
    table.add_argument("--limit", type=int, default=5)
    sub.add_parser("check", help="check packaged task bundles and reconciliations")
    args = parser.parse_args()
    if args.command == "inputs":
        print_inputs()
    elif args.command == "summary":
        print_summary(args.task, args.full)
    elif args.command == "table":
        if args.limit < 1:
            parser.error("--limit must be positive")
        print_table(args.task, args.name, args.limit)
    else:
        check()


if __name__ == "__main__":
    main()
