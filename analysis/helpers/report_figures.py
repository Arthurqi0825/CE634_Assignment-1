"""Redraw the report's explanatory figures from processed CSV supplements.

Run after Tasks 2--5 and before compiling ``main.tex``. This script does not
read raw Parquet files or change statistical estimates.

    ./.venv/bin/python -m analysis.helpers.report_figures
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.common import RESULTS_DIR

BLUE = "#23618a"
ORANGE = "#c1663b"
PURPLE = "#82578f"
GRID = "#dbe2e7"
SERVICE_COLORS = {"Yellow": "#b07d2b", "Uber": BLUE, "Lyft": PURPLE}


def read_rows(task: int, filename: str) -> list[dict[str, str]]:
    """Read one checked, processed table without touching monthly raw data."""
    path = RESULTS_DIR / f"Task_{task}/tables/{filename}"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, task: int, filename: str) -> None:
    path = RESULTS_DIR / f"Task_{task}/figures/{filename}"
    fig.savefig(path, dpi=230, facecolor="white")
    plt.close(fig)
    print(path.relative_to(RESULTS_DIR.parent))


def task2_rankings() -> None:
    """Explain why the flow maps and the two top-ten rankings look different."""
    rows = read_rows(2, "top10_spending_od.csv")
    groups = [
        (month, service)
        for service in ("Yellow", "Uber", "Lyft")
        for month in ("2026-04", "2026-05")
    ]
    fig, (overlap_ax, outside_ax) = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.7),
        gridspec_kw={"width_ratios": [1, 1.45]},
        layout="constrained",
    )
    y_positions = np.arange(len(groups))[::-1]
    for y, (month, service) in zip(y_positions, groups):
        group = [
            row for row in rows if row["month"] == month and row["service"] == service
        ]
        if len(group) != 10:
            raise ValueError(f"Expected ten spending flows for {month} {service}")
        overlap = sum(row["also_volume_top10"] == "True" for row in group)
        spending = sum(float(row["total_spending_usd"]) for row in group)
        outside_rows = [row for row in group if row["DOLocationID"] == "265"]
        outside_share = (
            100
            * sum(float(row["total_spending_usd"]) for row in outside_rows)
            / spending
        )
        color = SERVICE_COLORS[service]
        overlap_ax.barh(y, overlap, height=0.58, color=color)
        overlap_ax.text(
            overlap + 0.12, y, f"{overlap}/10", va="center", fontsize=9, weight="bold"
        )
        outside_ax.barh(y, outside_share, height=0.58, color=color)
        long_bar = outside_share > 60
        outside_ax.text(
            outside_share - 1.5 if long_bar else outside_share + 1,
            y,
            f"{outside_share:.1f}% ({len(outside_rows)}/10 flows)",
            va="center",
            ha="right" if long_bar else "left",
            fontsize=8.4,
            weight="bold",
            color="white" if long_bar else "#1d2931",
        )
    labels = [f"{service} {month[5:]}" for month, service in groups]
    overlap_ax.set_yticks(y_positions, labels)
    outside_ax.set_yticks(y_positions, [""] * len(groups))
    overlap_ax.set_xlim(0, 10)
    overlap_ax.set_xticks(range(0, 11, 2))
    outside_ax.set_xlim(0, 105)
    outside_ax.set_xticks(range(0, 101, 20))
    overlap_ax.set_xlabel("Common OD pairs among two top-ten lists")
    outside_ax.set_xlabel("Share of top-ten spending ending in zone 265 (%)")
    overlap_ax.set_title("Volume vs spending overlap", fontsize=11, weight="bold")
    outside_ax.set_title("Outside-of-NYC destination", fontsize=11, weight="bold")
    for ax in (overlap_ax, outside_ax):
        ax.grid(axis="x", color=GRID)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(
        "High-volume flows differ from high-spending flows", fontsize=14, weight="bold"
    )
    save(fig, 2, "od_ranking_comparison.png")


def task3_rates() -> None:
    """Keep each metric readable while stating every panel's own scale."""
    rows = read_rows(3, "monthly_rates.csv")
    metrics = (
        (
            "pooling_request_rate",
            "Pooling requests / valid trips",
            "valid_status_denominator",
        ),
        (
            "reported_matched_rate",
            "Reported matches / valid trips",
            "valid_status_denominator",
        ),
        ("matching_success", "Y/Y trips / pooling requests", "pooling_requests"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 8.0), layout="constrained")
    for col_index, service in enumerate(("Uber", "Lyft")):
        service_rows = sorted(
            (row for row in rows if row["service"] == service),
            key=lambda row: row["month"],
        )
        for row_index, (field, heading, denominator) in enumerate(metrics):
            ax = axes[row_index, col_index]
            values = [100 * float(row[field]) for row in service_rows]
            color = BLUE if service == "Uber" else PURPLE
            ax.plot(
                [0, 1], values, color=color, linewidth=2.2, marker="o", markersize=6
            )
            margin = max(values) * 0.22 or 0.1
            ax.set_ylim(0, max(values) + margin)
            ax.set_xlim(-0.25, 1.25)
            ax.set_xticks([0, 1], ["Apr", "May"])
            ax.set_title(heading, fontsize=10.5, weight="bold", pad=12)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)
            ax.set_ylabel("Percent", fontsize=9)
            decimals = 3 if service == "Lyft" and row_index < 2 else 2
            for x, value, row in zip((0, 1), values, service_rows):
                ax.annotate(
                    f"{value:.{decimals}f}%",
                    (x, value),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    weight="bold",
                    color=color,
                )
                ax.annotate(
                    f"n={int(row[denominator]):,}",
                    (x, 0),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#4c5c66",
                )
        axes[0, col_index].text(
            0.5,
            1.34,
            service,
            transform=axes[0, col_index].transAxes,
            va="center",
            ha="center",
            color=BLUE if service == "Uber" else PURPLE,
            fontsize=13.5,
            weight="bold",
        )
    fig.suptitle(
        "Reported sharing differs sharply by platform", fontsize=15, weight="bold"
    )
    save(fig, 3, "monthly_sharing_rates.png")


def task4_monthly() -> None:
    """Show the excess-time distribution and missing reference coverage together."""
    rows = read_rows(4, "month_summary.csv")
    fig, (ax, coverage_ax) = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.1),
        gridspec_kw={"width_ratios": [1.6, 1]},
        layout="constrained",
    )
    y = np.array([1, 0])
    colors = [BLUE, ORANGE]
    labels = ["April", "May"]
    for i, row in enumerate(rows):
        q1 = float(row["excess_p25_minutes"])
        median = float(row["excess_median_minutes"])
        q3 = float(row["excess_p75_minutes"])
        ax.plot(
            [q1, q3], [y[i], y[i]], color=colors[i], linewidth=8, solid_capstyle="round"
        )
        ax.scatter(
            median,
            y[i],
            s=105,
            color="white",
            edgecolor=colors[i],
            linewidth=2.5,
            zorder=3,
        )
        ax.text(q3 + 0.25, y[i], f"median {median:.2f} min", va="center", fontsize=10)
    ax.axvline(0, color="#45525b", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlim(-1, 14)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Observed excess E (minutes)")
    ax.set_title("Median and interquartile range", fontsize=11, weight="bold")
    ax.grid(axis="x", color=GRID)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for i, row in enumerate(rows):
        value = 100 * float(row["reference_coverage"])
        coverage_ax.barh(y[i], value, height=0.35, color=colors[i])
        coverage_ax.text(
            value + 1,
            y[i],
            f"{value:.1f}%  ({int(row['reference_supported']):,}/{int(row['shared_eligible']):,})",
            va="center",
            fontsize=9,
        )
    coverage_ax.set_yticks(y, ["", ""])
    coverage_ax.set_xlim(0, 110)
    coverage_ax.set_ylim(-0.6, 1.6)
    coverage_ax.set_xlabel("Eligible Y/Y trips with reference (%)")
    coverage_ax.set_title("Reference coverage", fontsize=11, weight="bold")
    coverage_ax.grid(axis="x", color=GRID)
    coverage_ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(
        "Uber Y/Y time differences are reported only for supported cells",
        fontsize=14,
        weight="bold",
    )
    save(fig, 4, "monthly_coverage_excess.png")


def task4_hourly() -> None:
    """Replace overlapping IQR ribbons with median and coverage trends."""
    rows = read_rows(4, "hourly_summary.csv")
    fig, (time_ax, coverage_ax) = plt.subplots(
        2,
        1,
        figsize=(9.2, 5.7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1]},
        layout="constrained",
    )
    for month, label, color in (("2026-04", "April", BLUE), ("2026-05", "May", ORANGE)):
        month_rows = sorted(
            (row for row in rows if row["month"] == month),
            key=lambda row: int(row["pickup_hour"]),
        )
        hours = [int(row["pickup_hour"]) for row in month_rows]
        medians = [float(row["excess_median_minutes"]) for row in month_rows]
        coverage = [100 * float(row["reference_coverage"]) for row in month_rows]
        time_ax.plot(
            hours,
            medians,
            color=color,
            marker="o",
            markersize=3,
            linewidth=2,
            label=label,
        )
        coverage_ax.plot(
            hours,
            coverage,
            color=color,
            marker="o",
            markersize=3,
            linewidth=2,
            label=label,
        )
    time_ax.set_ylabel("Median E (minutes)")
    time_ax.set_ylim(0, 7)
    time_ax.set_title("Excess time by pickup hour", loc="left", weight="bold")
    time_ax.legend(frameon=False, ncol=2, loc="upper left")
    coverage_ax.set_ylabel("Reference coverage (%)")
    coverage_ax.set_xlabel("Pickup hour (0--23)")
    coverage_ax.set_ylim(0, 80)
    coverage_ax.set_xticks(range(0, 24, 2))
    coverage_ax.set_title(
        "Share of eligible Y/Y trips with a supported N/N reference",
        loc="left",
        fontsize=10,
    )
    for ax in (time_ax, coverage_ax):
        ax.grid(color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Hourly comparisons and their unequal coverage", fontsize=14, weight="bold"
    )
    save(fig, 4, "hourly_excess_minutes.png")


def task5_price() -> None:
    """Display price estimates and intervals beside support and coverage."""
    rows = read_rows(5, "price_comparison.csv")
    fig, ax = plt.subplots(figsize=(9.3, 4.6), layout="constrained")
    positions = [3, 2, 1, 0]
    for y, row in zip(positions, rows):
        estimate = float(row["mean_difference_usd"])
        low = float(row["ci95_low_usd"])
        high = float(row["ci95_high_usd"])
        color = BLUE if row["service"] == "Uber" else PURPLE
        ax.plot([low, high], [y, y], color=color, linewidth=3, solid_capstyle="round")
        ax.plot([low, low], [y - 0.09, y + 0.09], color=color, linewidth=2)
        ax.plot([high, high], [y - 0.09, y + 0.09], color=color, linewidth=2)
        ax.scatter(
            estimate, y, s=95, color=color, edgecolor="white", linewidth=1, zorder=3
        )
        ax.text(
            1,
            y,
            f"{estimate:.2f} USD  |  n={int(row['shared_supported']):,}  |  "
            f"coverage={100 * float(row['shared_reference_coverage']):.1f}%",
            va="center",
            fontsize=9.2,
            color=color,
            weight="bold",
        )
    ax.axvline(0, color="#46545c", linewidth=1.3, linestyle="--")
    ax.set_yticks(positions, [f"{r['service']}  {r['month'][5:]}" for r in rows])
    ax.set_xlim(-39, 38)
    ax.set_xticks([-35, -25, -15, -5, 0])
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlabel("Shared minus cell-comparable non-shared passenger spending (USD)")
    ax.set_title(
        "Observed price differences with date-clustered 95% intervals",
        fontsize=14,
        weight="bold",
    )
    ax.text(
        -38,
        3.43,
        "Lower observed spending for shared trips  ←",
        fontsize=9,
        color="#4c5c66",
    )
    ax.grid(axis="x", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, 5, "price_differences.png")


def main() -> None:
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.axisbelow": True}
    )
    task2_rankings()
    task3_rates()
    task4_monthly()
    task4_hourly()
    task5_price()


if __name__ == "__main__":
    main()
