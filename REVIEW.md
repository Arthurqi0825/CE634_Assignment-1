# Submission review

Reviewed on 2026-10-02 from this standalone package.

## Reproducibility checks

- Tasks 1–5.1 were rerun in order against the four local raw TLC Parquet files with `ASSIGNMENT_RAW_DIR` set. The package's `04_Results/Task_N/run.log` and `run.json` record the successful runs. Task 1's log includes its separate figure command.
- A second, isolated rerun wrote to `/tmp`. For all five tasks, `summary.json` matched the packaged result except for `generated_at`.
- `analysis.helpers.task_outputs check N` passed for N = 1, 2, 3, 4, 5. `analysis.helpers.inspect_results check` passed the bundle and cross-task reconciliations.
- Every Python module passed `pyflakes` and Black formatting checks. Five Task 2–5 explanatory figures were generated from the processed CSVs with `analysis.helpers.report_figures`; no raw estimates were changed. `main.tex` compiled with two pdfLaTeX passes to a 41-page PDF (eight narrative pages followed by a map atlas). The atlas captions now explain the sharing sample threshold, especially Lyft's sparse matching-success map.
- All 47 PNGs were recreated with `analysis.helpers.render_all_figures` while `ASSIGNMENT_RAW_DIR` pointed to a nonexistent path. Task 4's exact plotted histogram bin counts are now saved as a CSV. The plotting input map is in `visualizations/PLOT_DATA.md`.
- The ZIP was inspected for the four raw Parquet files, Git metadata, virtual environments, and LaTeX intermediate files; none is included.

## Scope and limits

Task 5.1 evaluates the observed price comparison. Task 5.2 is not included. The package contains processed CSVs, figures, summaries, code, and required reference files, while the four raw monthly Parquet files must be placed as described in `README.md` to rerun the analyses. The virtual environment used for review was already present outside this package; no global packages were installed or changed.
