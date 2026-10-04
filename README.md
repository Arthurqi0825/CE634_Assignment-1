# CE634 Assignment 1: reproducible code and English report

This folder is a standalone submission package. It contains the final English
LaTeX report, compiled PDF, processed results, PNG/PDF visualizations, reference
files, and the Python code used for Tasks 1–5. The four raw monthly Parquet
files are **not included**. Task 5.1 and the bonus Task 5.2 analysis are complete.

## Layout

| Path | Purpose |
| --- | --- |
| `main.tex`, `main.pdf` | Ten-page English report and compiled copy; maps are in a separate supplementary atlas. |
| `analysis/common.py` | Shared project paths, license codes, status order, zone lookup, numeric conversion, and the eight-component HVFHV passenger-spending calculation. |
| `analysis/tasks/` | Separate implementations for Tasks 1, 2, 3, 4, and 5.1; Task 1 figures have their own script. |
| `analysis/helpers/` | Raw-data overview, initial inspection, Parquet schema/browser, task run recorder, processed-CSV report figures, and read-only result inspection. |
| `04_Results/Task_N/` | Processed `tables/*.csv`, `figures/*.png`, `summary.json`, `report.tex`, `run.json`, and `run.log` for each task. |
| `visualizations/index.html` | Offline gallery of every PNG; click an image to inspect it at full resolution. |
| `supplementary_map_atlas.tex`, `supplementary_map_atlas.pdf` | Standalone atlas with the eight demand maps, directed OD maps, pooling maps, and travel-time geography. |
| `visualizations/PLOT_DATA.md` | Figure-to-CSV input map for drawing without raw Parquet files. |
| `01_Data/Reference/` | TLC dictionaries, zone lookup, and shapefile components needed for a rerun. |
| `03_Reports/raw_data_overview.json` | Separate uncleaned descriptive inspection result. |

The `run.log` files record every command used to regenerate the packaged
outputs from this layout. Each `run.json` records the most recent command for
its task; Task 1 runs its analysis and figure scripts in sequence.

## Requirements and isolated Python environment

- Python 3.8–3.11; the included outputs were checked with Python 3.8.8.
- Python dependencies are pinned in [`requirements.txt`](requirements.txt).
- pdfLaTeX with `graphicx`, `booktabs`, `array`, `tabularx`, `caption`,
  `hyperref`, `geometry`, `microtype`, `amsmath`, `amssymb`, `placeins`, and
  `url` to rebuild the report PDF.
- Optional Tkinter for the graphical Parquet browser. It is not needed for any
  formal analysis or for the command-line inspection tools.

From this folder, create a **project-local** virtual environment and install
the Python dependencies into it:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Every command below uses `./.venv/bin/python`. No global Python installation
or system package is changed. The `.venv/` directory is ignored by Git and is
not part of the ZIP. On an offline machine, use an existing local wheel cache
or an already prepared virtual environment with the pinned packages.

## Required data placement

Put exactly these four monthly TLC files under
`01_Data/Raw/TLC_Trip_Records/` after unpacking the submission:

```text
01_Data/Raw/TLC_Trip_Records/yellow_tripdata_2026-04.parquet
01_Data/Raw/TLC_Trip_Records/yellow_tripdata_2026-05.parquet
01_Data/Raw/TLC_Trip_Records/fhvhv_tripdata_2026-04.parquet
01_Data/Raw/TLC_Trip_Records/fhvhv_tripdata_2026-05.parquet
```

The reference lookup, both data dictionaries, and all shapefile components
are already in `01_Data/Reference/`. Check input placement without reading
the full raw files:

```bash
./.venv/bin/python -m analysis.helpers.inspect_results inputs
./.venv/bin/python -m analysis.helpers.show_parquet_keys
```

For a temporary read-only check against raw files stored elsewhere, set
`ASSIGNMENT_RAW_DIR` to the directory containing the four Parquet files.
That override is optional; normal submission reproduction uses the paths
above. None of the task scripts writes to the raw directory.

Example for the author's local raw-data layout:

```bash
export ASSIGNMENT_RAW_DIR=/Users/zhangziqi/.codex/.chatgpt-projects/g-p-6aa7cf2a7bd081918e32c5e7f08a5dbb/output/UrbanBigData/03_Assignments/Assignment_1/01_Data/Raw/TLC_Trip_Records
```

## Reproduce the analyses in order

Run every command from this folder. `task_outputs run` records the command,
stdout/stderr, timestamps, and exit status with each task. It also supplies
the task's output directory to the analysis script.

```bash
# Descriptive inspection of uncleaned data; separate from formal task samples.
./.venv/bin/python -m analysis.helpers.raw_data_overview

# Task 1: audit all four files, apply analysis-specific rules, count all four
# HVFHV license codes and five sharing statuses, then regenerate audit charts.
./.venv/bin/python -m analysis.helpers.task_outputs run 1 -- ./.venv/bin/python -m analysis.tasks.task1_analysis
./.venv/bin/python -m analysis.helpers.task_outputs run 1 -- ./.venv/bin/python -m analysis.tasks.task1_figures

# Task 2: eight zone demand maps and the separate top-ten directed OD lists
# and maps for trip volume and valid passenger spending.
./.venv/bin/python -m analysis.helpers.task_outputs run 2 -- ./.venv/bin/python -m analysis.tasks.task2_analysis

# Task 3: Uber/Lyft request rate, reported matched rate, and success among
# requests, including denominators, insufficient-sample labels, and maps.
./.venv/bin/python -m analysis.helpers.task_outputs run 3 -- ./.venv/bin/python -m analysis.tasks.task3_analysis

# Task 4: Uber N/N reference cells, supported Y/Y travel-time comparisons,
# coverage, monthly/hourly/zone/within-zone summaries, and figures.
./.venv/bin/python -m analysis.helpers.task_outputs run 4 -- ./.venv/bin/python -m analysis.tasks.task4_analysis

# Task 5.1: Y/Y versus N/N passenger-spending comparison, date-clustered
# uncertainty, reference-threshold sensitivity, and its figure.
./.venv/bin/python -m analysis.helpers.task_outputs run 5 -- ./.venv/bin/python -m analysis.tasks.task5_price_analysis

# Task 5.2 bonus: potential sharing among N/N trips under same-OD
# pickup-time-window rules, including 5/10/15-minute sensitivity.
./.venv/bin/python -m analysis.helpers.task_outputs run 5 -- ./.venv/bin/python -m analysis.tasks.task5_potential_sharing

# Redraw the explanatory Task 2--5 charts from the processed CSV supplements.
./.venv/bin/python -m analysis.helpers.report_figures
```

Tasks 2–4 use the Task 1 outputs for reconciliation, so Task 1 runs first.
Tasks 3–5 concern HVFHV only; Yellow Taxi has no specified sharing status.
Month, day, and hour come from each row's pickup timestamp. OD keys retain
direction and within-zone trips. The common passenger-spending helper requires
all eight HVFHV components and excludes driver pay; it never turns a missing
component into zero.

After each task, run its recorded-output check, replacing `N` with 1–5:

```bash
./.venv/bin/python -m analysis.helpers.task_outputs check N
```

The following read-only check confirms packaged artifacts and cross-task
count reconciliation without scanning raw records:

```bash
./.venv/bin/python -m analysis.helpers.inspect_results check
```

## Inspect and print data without rerunning analyses

```bash
./.venv/bin/python -m analysis.helpers.inspect_results summary 4
./.venv/bin/python -m analysis.helpers.inspect_results summary 5 --full
./.venv/bin/python -m analysis.helpers.inspect_results table 2 top10_volume_od --limit 10
./.venv/bin/python -m analysis.helpers.show_parquet_keys path/to/file.parquet
./.venv/bin/python -m analysis.helpers.browse_parquet --no-gui
```

`browse_parquet` can open an optional Tkinter window when a graphical desktop
and Tkinter are available. `initial_tlc_inspection` is an additional raw-data
diagnostic helper; it is not required to reproduce the formal task results.
The CSV supplements give exact OD counts, service-month denominators,
spending counts and means, sharing numerators/denominators, reference cells,
coverage, exclusion overlaps, Task 5.1 sensitivity estimates, and Task 5.2
potential-shareability screens by pickup-time window, pickup zone, and OD pair.

## Rebuild and review the report

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_map_atlas.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_map_atlas.tex
```

The second pass resolves figure references. The 10-page main report contains the
Task 1--5 explanatory charts; the separate atlas contains the eight required
spatial-demand maps and the detailed OD and zone-level maps. The processed PNGs are
available through `visualizations/index.html`, including non-map charts that
are not repeated in the atlas. Zones 264/265 are valid lookup IDs without
drawable polygons; their counts remain in the CSVs. OD flow lines connect
zone centroids and do not represent exact routes. An uncolored sharing-rate
zone may mean insufficient sample rather than a zero rate.

If raw files or code change, rerun the relevant analyses, recompile the PDF,
and check every numerical statement in `main.tex` against the refreshed CSVs
and summaries before making a new ZIP. The source files are never copied into
the package. Use `git status` to review only intended changes. To rebuild the
ZIP from this folder after the checks, run:

```bash
./.venv/bin/python -m analysis.helpers.package_submission
```

The archive is written to `dist/ce634_taskpaper.zip` and omits Git metadata,
raw Parquet files, the virtual environment, local archives, and LaTeX
intermediate files. The `dist/` folder is ignored by Git.

## Redraw figures without raw Parquet

All submitted plotting inputs are already separated from the monthly Parquet
files. The CSV-to-figure mapping is in `visualizations/PLOT_DATA.md`; the Taxi
Zone lookup and shapefile are under `01_Data/Reference/`. After installing the
project-local dependencies, this command recreates the packaged PNG figures without
opening any raw monthly file:

```bash
./.venv/bin/python -m analysis.helpers.render_all_figures
```

Raw Parquet files are necessary only to recalculate the CSV analyses. The
Task 4 distribution uses saved histogram bins rather than a retained
trip-level extract.

## Publish the visualization dashboard on GitHub Pages

The repository root contains a static dashboard:

```text
index.html
assets/site.css
assets/site.js
.nojekyll
```

It is designed for GitHub Pages with **Deploy from a branch** and the source
set to the repository root. Root publishing is intentional because the page
reads existing files under `04_Results/`, `visualizations/`, and `main.pdf`
without duplicating large figures. The `.nojekyll` file keeps paths such as
`04_Results/Task_1/summary.json` available exactly as written.

Preview locally from the repository root with:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/`. Opening `index.html` directly also displays
the static layout, but browser security may block the JSON fetches used for the
live KPI text.

## Research-paper report build (annotated figures)

The final research-paper version uses `main.tex` and the annotated main figures under:

```text
04_Results/paper_figures/main/
```

Regenerate the annotated main figures from the packaged processed CSV tables and Taxi Zone reference files with:

```bash
python3 -m analysis.render_paper_figures
```

Then compile the paper with:

```bash
pdflatex main.tex
pdflatex main.tex
pdflatex supplementary_map_atlas.tex
pdflatex supplementary_map_atlas.tex
```

The main report follows the assignment tasks and is 10 pages, including its charts and references. The separate map atlas contains the required demand maps and detailed OD, pooling, and travel-time maps.
