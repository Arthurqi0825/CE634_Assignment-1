# CE634 Assignment 1

This repository contains the complete implementation, results, visualizations, and report for **CE634 Urban Big Data – Assignment 1**.

The project analyzes April and May 2026 NYC TLC Yellow Taxi and High Volume For-Hire Vehicle (HVFHV) data, with particular focus on spatial travel demand, OD movements, ride-sharing behavior, travel-time effects, passenger spending, and potential sharing opportunities.

---

## Repository Contents

```text
.
├── analysis/
│   ├── common.py
│   ├── tasks/
│   └── helpers/
│
├── 01_Data/
│   ├── Raw/
│   └── Reference/
│
├── 03_Reports/
│
├── 04_Results/
│   ├── Task_1/
│   ├── Task_2/
│   ├── Task_3/
│   ├── Task_4/
│   └── Task_5/
│
├── visualizations/
│   ├── index.html
│   └── PLOT_DATA.md
│
├── main.tex
├── main.pdf
├── supplementary_map_atlas.tex
├── supplementary_map_atlas.pdf
├── IMPLEMENTATION.md
├── requirements.txt
└── README.md
```

### Main files

- **`main.pdf`** — final 10-page assignment report.
- **`supplementary_map_atlas.pdf`** — supplementary spatial maps.
- **`analysis/tasks/`** — implementation of Tasks 1–5.
- **`04_Results/`** — processed tables, figures, summaries, and task outputs.
- **`visualizations/index.html`** — gallery of generated visualizations.
- **`IMPLEMENTATION.md`** — detailed description of the methodology and implementation of each task.
- **`visualizations/PLOT_DATA.md`** — mapping between figures and their processed data sources.

The original TLC Parquet datasets are not included in the repository.

---

## Assignment Coverage

The repository implements all required tasks and the bonus analysis.

| Task | Analysis |
|---|---|
| **Task 1** | Data inspection, cleaning, sample reconciliation, and sharing-status classification |
| **Task 2** | Spatial demand maps and directed OD analysis by trip volume and passenger spending |
| **Task 3** | Pooling-request, reported-match, and matching-success analysis for Uber and Lyft |
| **Task 4** | Travel-time comparison between successfully shared and comparable non-shared Uber trips |
| **Task 5.1** | Passenger-spending comparison between shared and non-shared trips |
| **Task 5.2** | Bonus potential-sharing analysis with 5-, 10-, and 15-minute temporal windows |

Detailed definitions, filtering rules, formulas, reference-cell construction, minimum sample thresholds, and interpretation assumptions are documented in **`IMPLEMENTATION.md`**.

---

## Data

To reproduce the analysis from the original TLC records, place the following files in:

```text
01_Data/Raw/TLC_Trip_Records/
```

Required files:

```text
yellow_tripdata_2026-04.parquet
yellow_tripdata_2026-05.parquet
fhvhv_tripdata_2026-04.parquet
fhvhv_tripdata_2026-05.parquet
```

Taxi Zone lookup files, shapefiles, and TLC reference material are stored under:

```text
01_Data/Reference/
```

---

## Environment Setup

Python 3.8–3.11 is recommended.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it and install the required packages:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Reproduce the Analysis

Run all commands from the repository root.

### Task 1

```bash
python -m analysis.helpers.task_outputs run 1 -- \
python -m analysis.tasks.task1_analysis

python -m analysis.helpers.task_outputs run 1 -- \
python -m analysis.tasks.task1_figures
```

### Task 2

```bash
python -m analysis.helpers.task_outputs run 2 -- \
python -m analysis.tasks.task2_analysis
```

### Task 3

```bash
python -m analysis.helpers.task_outputs run 3 -- \
python -m analysis.tasks.task3_analysis
```

### Task 4

```bash
python -m analysis.helpers.task_outputs run 4 -- \
python -m analysis.tasks.task4_analysis
```

### Task 5.1

```bash
python -m analysis.helpers.task_outputs run 5 -- \
python -m analysis.tasks.task5_price_analysis
```

### Task 5.2

```bash
python -m analysis.helpers.task_outputs run 5 -- \
python -m analysis.tasks.task5_potential_sharing
```

Tasks 2–4 depend on the reconciled outputs produced in Task 1, so Task 1 should be executed first.

---

## Regenerate Visualizations

Processed plotting inputs are included in the repository.

To regenerate the complete visualization set:

```bash
python -m analysis.helpers.render_all_figures
```

To regenerate explanatory figures used in the report:

```bash
python -m analysis.helpers.report_figures
```

The relationship between each figure and its source CSV file is documented in:

```text
visualizations/PLOT_DATA.md
```

The full visualization gallery can be opened through:

```text
visualizations/index.html
```

---

## Validate Results

Check the recorded output of an individual task using:

```bash
python -m analysis.helpers.task_outputs check N
```

where `N` is the task number.

A repository-wide consistency check can be performed using:

```bash
python -m analysis.helpers.inspect_results check
```

Task summaries can also be inspected directly:

```bash
python -m analysis.helpers.inspect_results summary 4

python -m analysis.helpers.inspect_results summary 5 --full
```

---

## Build the Report

Compile the main report with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Compile the supplementary map atlas with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error supplementary_map_atlas.tex
pdflatex -interaction=nonstopmode -halt-on-error supplementary_map_atlas.tex
```

The final submission outputs are:

```text
main.pdf
supplementary_map_atlas.pdf
```

---

## Detailed Implementation

The complete methodology is documented in:

**[`IMPLEMENTATION.md`](IMPLEMENTATION.md)**

It describes:

- data cleaning and retained-sample rules;
- Uber and Lyft identification;
- sharing-status definitions;
- spatial aggregation;
- directed OD construction;
- pooling request and matching metrics;
- minimum sample thresholds;
- Task 4 reference-cell design;
- travel-time comparison methodology;
- passenger-spending calculation;
- confidence interval calculation;
- Task 5.2 potential-sharing definition and sensitivity analysis.

---

## Output Structure

Each task stores its outputs under:

```text
04_Results/Task_N/
```

Depending on the task, the folder contains:

- processed CSV tables;
- figures;
- task summaries;
- generated report fragments;
- execution records;
- validation information.

The main report contains the key findings, while detailed spatial results are provided in the supplementary atlas and visualization gallery.
