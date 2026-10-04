# Task 5: Shared-trip price comparison and potential shareability

This directory contains generated evidence, not raw Parquet files.

A completed task bundle includes:

- `run.log` and `run.json`: command, timestamps, output, and exit code.
- `summary.json`: substantive machine-readable results and denominators.
- `tables/*.csv`: supporting processed data.
- `report.tex`: methods, results, and limitations for the main report.
- `report_task5_2.tex`: bonus Task 5.2 method, table, figure, and limitations.
- `figures/`: required maps and charts.

Task 5.1 compares reported Y/Y passenger spending with supported N/N reference
cells. Task 5.2 screens N/N trips for coarse potential sharing under same
platform, same month, same directed OD, and pickup-time-window rules; the main
window is 10 minutes, with 5- and 15-minute sensitivity checks.

Before accepting a task, run `python -m analysis.helpers.task_outputs check 5`. TeX figure paths start at the package root, for example `04_Results/Task_5/figures/figure.png`.
