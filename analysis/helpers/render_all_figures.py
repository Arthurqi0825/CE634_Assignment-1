"""Recreate every submitted PNG from processed CSVs and the zone shapefile.

This command never opens a raw monthly Parquet file. Run it after the formal
task scripts, or on a remote machine with only the tracked submission files:

    ./.venv/bin/python -m analysis.helpers.render_all_figures

The five explanatory report charts are rendered last so they replace the
less legible initial versions produced during the formal task runs.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.common import RESULTS_DIR, STUDY_MONTHS
from analysis.helpers import report_figures
from analysis.tasks import (
    task1_figures,
    task2_analysis,
    task3_analysis,
    task4_analysis,
)


def rows(task: int, filename: str) -> list[dict[str, str]]:
    """Load a processed table using the common report-figure CSV reader."""
    return report_figures.read_rows(task, filename)


def task2_maps(shapes: dict, lookup: dict) -> None:
    """Reconstruct only the aggregate arrays needed by Task 2's map functions."""
    month_index = {month: index for index, month in enumerate(STUDY_MONTHS)}
    hotspot_service = {"Yellow": 0, "HVFHV": 1}
    flow_service = {"Yellow": 0, "Uber": 1, "Lyft": 2}
    pickup = np.zeros((2, 2, 266), dtype=np.int64)
    dropoff = np.zeros_like(pickup)
    for row in rows(2, "zone_demand.csv"):
        month = month_index[row["month"]]
        service = hotspot_service[row["service"]]
        location = int(row["LocationID"])
        target = pickup if row["measure"] == "pickup" else dropoff
        target[month, service, location] = int(row["trip_count"])
    task2_analysis.draw_hotspots(shapes, lookup, pickup, dropoff)

    volume = rows(2, "top10_volume_od.csv")
    spending = rows(2, "top10_spending_od.csv")
    volume_array = np.zeros((2, 4, 266, 266), dtype=np.int64)
    spending_array = np.zeros((2, 4, 266, 266), dtype=np.float64)
    top_pairs: dict[
        tuple[str, str], tuple[list[tuple[int, int]], list[tuple[int, int]]]
    ] = {}
    for month in STUDY_MONTHS:
        for service in flow_service:
            volume_rows = sorted(
                (
                    row
                    for row in volume
                    if row["month"] == month and row["service"] == service
                ),
                key=lambda row: int(row["rank"]),
            )
            spending_rows = sorted(
                (
                    row
                    for row in spending
                    if row["month"] == month and row["service"] == service
                ),
                key=lambda row: int(row["rank"]),
            )
            if len(volume_rows) != 10 or len(spending_rows) != 10:
                raise ValueError(
                    f"Expected ten volume and spending flows for {month} {service}"
                )
            top_pairs[(month, service)] = (
                [
                    (int(row["PULocationID"]), int(row["DOLocationID"]))
                    for row in volume_rows
                ],
                [
                    (int(row["PULocationID"]), int(row["DOLocationID"]))
                    for row in spending_rows
                ],
            )
            for row in volume_rows:
                origin, destination = int(row["PULocationID"]), int(row["DOLocationID"])
                volume_array[
                    month_index[month], flow_service[service], origin, destination
                ] = int(row["trip_count"])
            for row in spending_rows:
                origin, destination = int(row["PULocationID"]), int(row["DOLocationID"])
                spending_array[
                    month_index[month], flow_service[service], origin, destination
                ] = float(row["total_spending_usd"])
    task2_analysis.draw_flows(shapes, lookup, top_pairs, volume_array, spending_array)
    for row in volume:
        row["also_spending_top10"] = row["also_spending_top10"] == "True"
    task2_analysis.draw_comparison(volume, spending)


def task3_maps(shapes: dict, lookup: dict) -> None:
    """Recreate the twelve zone-rate maps from their exact saved rates."""
    zone_rows = rows(3, "zone_rates.csv")
    metrics = {
        "pooling_request_rate": "Pooling request rate",
        "reported_matched_rate": "Reported matched rate",
        "matching_success": "Matching success among requests",
    }
    for row in zone_rows:
        row["PULocationID"] = int(row["PULocationID"])
        for metric in metrics:
            row[metric] = float(row[metric]) if row[metric] else None
    scale = {}
    for metric in metrics:
        values = [
            row[metric]
            for row in zone_rows
            if row[metric + "_status"] == "reported" and row[metric] is not None
        ]
        scale[metric] = float(np.ceil(max(values) * 100) / 100)
    for month in STUDY_MONTHS:
        for service in ("Uber", "Lyft"):
            by_id = {
                row["PULocationID"]: row
                for row in zone_rows
                if row["month"] == month and row["service"] == service
            }
            for metric, label in metrics.items():
                task3_analysis.draw_rate_map(
                    shapes, lookup, by_id, month, service, metric, label, scale[metric]
                )


def task4_figures(shapes: dict) -> None:
    """Redraw Task 4 maps, the relative-hour chart, and histogram from bins."""
    summary = json.loads((RESULTS_DIR / "Task_4/summary.json").read_text())
    zone_rows = rows(4, "zone_summary.csv")
    for row in zone_rows:
        for field in ("PULocationID", "shared_eligible", "reference_supported"):
            row[field] = int(row[field])
        row["mean_excess_minutes"] = (
            float(row["mean_excess_minutes"]) if row["mean_excess_minutes"] else None
        )
    limit = float(summary["zone_map_color_limit_minutes"])
    for month in STUDY_MONTHS:
        task4_analysis.plot_zone_map(shapes, zone_rows, month, limit)
    task4_analysis.plot_coverage_zone(shapes, zone_rows)

    hourly = rows(4, "hourly_summary.csv")
    for row in hourly:
        row["pickup_hour"] = int(row["pickup_hour"])
        for field in (
            "relative_p25_percent",
            "relative_median_percent",
            "relative_p75_percent",
        ):
            row[field] = float(row[field])
    task4_analysis.plot_hour(
        hourly, "relative_median_percent", "Median R (%)", "hourly_relative_percent.png"
    )

    bins = rows(4, "excess_distribution_bins.csv")
    left = np.array([float(row["bin_left_minutes"]) for row in bins])
    right = np.array([float(row["bin_right_minutes"]) for row in bins])
    counts = np.array([int(row["trip_count"]) for row in bins])
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.bar(left, counts, width=right - left, align="edge", color="#517da5")
    ax.axvline(0, color="#454545", linewidth=1)
    ax.set_xlabel("Observed excess minutes E")
    ax.set_ylabel("Supported Y/Y trips")
    ax.set_title("Uber excess-time distribution (visual range: 1st–99th percentiles)")
    ax.text(
        0.99,
        0.95,
        "Full-data summaries retain tails and negative values.\n"
        f"Displayed range: {float(bins[0]['visual_p01_minutes']):.1f} to "
        f"{float(bins[0]['visual_p99_minutes']):.1f} min",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "Task_4/figures/excess_distribution.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)

    within = summary["within_between"]
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [
        "within zone" if row["within_zone"] else "between zones" for row in within
    ]
    ax.bar(
        labels,
        [row["excess_median_minutes"] for row in within],
        color=["#9c6cad", "#4b8a86"],
    )
    for index, row in enumerate(within):
        ax.text(
            index,
            row["excess_median_minutes"] + 0.15,
            f"n={row['reference_supported']:,}",
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("Median E (minutes)")
    ax.set_title("Trip-weighted within- versus between-zone comparison")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "Task_4/figures/within_between_excess.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    task1_figures.main()
    shapes = task2_analysis.load_shapes()
    lookup = task2_analysis.zone_lookup()
    task2_maps(shapes, lookup)
    task3_maps(shapes, lookup)
    task4_figures(shapes)
    report_figures.main()
    expected = {1: 3, 2: 22, 3: 13, 4: 8, 5: 1}
    for task, count in expected.items():
        actual = len(list((RESULTS_DIR / f"Task_{task}/figures").glob("*.png")))
        if actual != count:
            raise AssertionError(f"Task {task}: expected {count} PNGs, found {actual}")
    print("All 47 PNG figures recreated without opening raw Parquet files.")


if __name__ == "__main__":
    main()
