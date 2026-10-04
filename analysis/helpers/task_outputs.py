"""Prepare, run, and verify the required output bundle for each assignment task.

Examples:
  python -m analysis.helpers.task_outputs prepare 1
  python -m analysis.helpers.task_outputs run 1 -- python -m analysis.tasks.task1_analysis
  python -m analysis.helpers.task_outputs check 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from analysis.common import PROJECT_ROOT

ROOT = PROJECT_ROOT
RESULTS = ROOT / "04_Results"
TITLES = {
    1: "Data audit and retained counts",
    2: "Demand and passenger spending",
    3: "Reported sharing",
    4: "Uber excess travel time",
    5: "Shared-trip price comparison and potential shareability",
}


def task_dir(task: int) -> Path:
    return RESULTS / f"Task_{task}"


def prepare(task: int) -> Path:
    folder = task_dir(task)
    (folder / "tables").mkdir(parents=True, exist_ok=True)
    (folder / "figures").mkdir(exist_ok=True)
    guide = folder / "README.md"
    if not guide.exists():
        guide.write_text(
            f"# Task {task}: {TITLES[task]}\n\n"
            "This directory contains generated evidence, not raw Parquet files.\n\n"
            "A completed task bundle includes:\n\n"
            "- `run.log` and `run.json`: command, timestamps, output, and exit code.\n"
            "- `summary.json`: substantive machine-readable results and denominators.\n"
            "- `tables/*.csv`: supporting processed data.\n"
            "- `report.tex`: methods, results, and limitations for the main report.\n"
            "- `figures/`: required maps and charts.\n\n"
            "Before accepting a task, run `python -m analysis.helpers.task_outputs check "
            + str(task)
            + "`. "
            "TeX figure paths start at the package root, for example "
            f"`04_Results/Task_{task}/figures/figure.png`.\n",
            encoding="utf-8",
        )
    return folder


def run(task: int, command: list[str]) -> int:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit(
            "Provide a command after --, e.g. run 1 -- python -m analysis.tasks.task1_analysis"
        )
    folder = prepare(task)
    started = datetime.now().astimezone()
    env = os.environ.copy()
    env["ASSIGNMENT_TASK_OUTPUT_DIR"] = str(folder)
    header = f"\n=== Task {task} run started {started.isoformat(timespec='seconds')} ===\nCommand: {json.dumps(command, ensure_ascii=False)}\n"
    with (folder / "run.log").open("a", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        print(header, end="")
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            exit_code = process.wait()
        except KeyboardInterrupt:
            if process is not None:
                process.terminate()
                exit_code = process.wait()
            else:
                exit_code = 130
            log.write("Interrupted by user.\n")
            print("Interrupted by user.")
        except OSError as exc:
            exit_code = 127
            log.write(f"Could not start command: {exc}\n")
            print(f"Could not start command: {exc}", file=sys.stderr)
        finished = datetime.now().astimezone()
        footer = f"Exit code: {exit_code}; finished: {finished.isoformat(timespec='seconds')}\n"
        log.write(footer)
        print(footer, end="")
    metadata = {
        "task": task,
        "command": command,
        "cwd": str(ROOT),
        "requirements": "requirements.txt",
        "runner_python": sys.version.split()[0],
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "exit_code": exit_code,
        "output_dir": str(folder.relative_to(ROOT)),
    }
    (folder / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return exit_code


def csv_has_data(path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.reader(handle)
            header = next(rows, None)
            first_data = next(rows, None)
            return bool(header) and bool(first_data)
    except (OSError, csv.Error, UnicodeError):
        return False


def check(task: int) -> bool:
    folder = task_dir(task)
    checks = {}
    checks["run.log"] = (folder / "run.log").is_file() and (
        folder / "run.log"
    ).stat().st_size > 0
    run_file = folder / "run.json"
    try:
        run_meta = json.loads(run_file.read_text(encoding="utf-8"))
        checks["run.json: exit 0"] = run_meta.get("exit_code") == 0
    except (OSError, ValueError):
        checks["run.json: exit 0"] = False
    summary_file = folder / "summary.json"
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        checks["summary.json: nonempty"] = isinstance(summary, (dict, list)) and bool(
            summary
        )
    except (OSError, ValueError):
        checks["summary.json: nonempty"] = False
    checks["tables/*.csv: at least one data row"] = any(
        csv_has_data(path) for path in (folder / "tables").glob("*.csv")
    )
    report_file = folder / "report.tex"
    try:
        tex = report_file.read_text(encoding="utf-8").strip()
        checks["report.tex: substantive fragment"] = len(tex) >= 20 and not any(
            marker in tex.lower() for marker in ("todo", "placeholder", "待补充")
        )
    except OSError:
        checks["report.tex: substantive fragment"] = False
    for name, valid in checks.items():
        print(f"{'OK' if valid else 'MISSING'}  {name}")
    return all(checks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "check", "run"):
        command = sub.add_parser(action)
        command.add_argument("task", type=int, choices=sorted(TITLES))
        if action == "run":
            command.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.task).relative_to(ROOT))
    elif args.action == "run":
        raise SystemExit(run(args.task, args.command))
    else:
        raise SystemExit(0 if check(args.task) else 1)


if __name__ == "__main__":
    main()
