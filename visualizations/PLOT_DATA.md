# Plot data independent of raw Parquet

All 47 submitted PNGs can be recreated from the processed CSV files in
`04_Results/Task_N/tables/`, the Taxi Zone lookup and shapefile in
`01_Data/Reference/`, and `04_Results/Task_4/summary.json`. No raw monthly
Parquet file is read during this drawing stage.

| Figures | Saved plot data |
| --- | --- |
| Task 1 audit and count charts | `Task_1/tables/task1_2_counts.csv`, `audit_rules.csv` |
| Task 2 eight zone-demand maps | `Task_2/tables/zone_demand.csv`, Taxi Zone lookup and shapefile |
| Task 2 twelve directed OD maps and two comparison charts | `Task_2/tables/top10_volume_od.csv`, `top10_spending_od.csv`, Taxi Zone lookup and shapefile |
| Task 3 twelve rate maps and monthly sharing chart | `Task_3/tables/zone_rates.csv`, `monthly_rates.csv`, Taxi Zone lookup and shapefile |
| Task 4 excess-time maps, hourly charts, summaries and histogram | `Task_4/tables/zone_summary.csv`, `hourly_summary.csv`, `month_summary.csv`, `excess_distribution_bins.csv`, `Task_4/summary.json`, Taxi Zone shapefile |
| Task 5.1 price chart | `Task_5/tables/price_comparison.csv` |

From the package root, after creating the local virtual environment described
in `README.md`, run:

```bash
./.venv/bin/python -m analysis.helpers.render_all_figures
```

The script checks that each task has its expected number of PNGs. It was
verified with `ASSIGNMENT_RAW_DIR` pointing to a nonexistent directory. Raw
Parquet is still required to **recalculate** the CSVs or change the statistical
definitions. The histogram CSV saves the exact 100 bin counts used for the
Task 4 distribution image; individual trip-level observations are not
included in this plotting package.
