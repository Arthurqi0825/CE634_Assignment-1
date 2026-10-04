"""Create report-ready Task 1 charts from saved audited CSV tables."""
from __future__ import annotations

import csv

import matplotlib

from analysis.common import PROJECT_ROOT, task_output_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = PROJECT_ROOT
OUT = task_output_dir(1)
FIGURES = OUT / "figures"


def read(name):
    with (OUT / "tables" / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    FIGURES.mkdir(exist_ok=True)
    rows = read("task1_2_counts.csv")
    counts = {
        (r["service"], r["request_match_category"]): (
            int(r["april_trips"]),
            int(r["may_trips"]),
        )
        for r in rows
    }
    colors = {
        "Yellow taxi": "#e2a53b",
        "Juno": "#78909c",
        "Uber": "#416a96",
        "Via": "#8b7f9f",
        "Lyft": "#be5b88",
    }
    monthly = {
        "Yellow taxi": counts[("Yellow taxi", "All retained trips")],
        "Juno": tuple(
            sum(
                counts[("Juno", status)][m]
                for status in ("N/N", "Y/N", "Y/Y", "N/Y", "Missing/other")
            )
            for m in (0, 1)
        ),
        "Uber": tuple(
            sum(
                counts[("Uber", status)][m]
                for status in ("N/N", "Y/N", "Y/Y", "N/Y", "Missing/other")
            )
            for m in (0, 1)
        ),
        "Via": tuple(
            sum(
                counts[("Via", status)][m]
                for status in ("N/N", "Y/N", "Y/Y", "N/Y", "Missing/other")
            )
            for m in (0, 1)
        ),
        "Lyft": tuple(
            sum(
                counts[("Lyft", status)][m]
                for status in ("N/N", "Y/N", "Y/Y", "N/Y", "Missing/other")
            )
            for m in (0, 1)
        ),
    }
    assert all(
        sum(monthly[s][m] for s in ("Juno", "Uber", "Via", "Lyft"))
        == counts[("HVFHV", "All retained trips")][m]
        for m in (0, 1)
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(2)
    for i, service in enumerate(monthly):
        vals = np.array(monthly[service]) / 1e6
        ax.bar(
            x + (i - 2) * 0.17, vals, width=0.15, color=colors[service], label=service
        )
        for j, v in enumerate(vals):
            label = "0" if v == 0 else f"{v:.2f}m"
            ax.text(x[j] + (i - 2) * 0.17, v + 0.18, label, ha="center", fontsize=8)
    ax.set_xticks(x, ["April 2026", "May 2026"])
    ax.set_ylabel("Retained completed trips (millions)")
    ax.set_ylim(0, 18)
    ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURES / "retained_trips_by_service.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    statuses = ("N/N", "Y/N", "Y/Y", "N/Y", "Missing/other")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for i, service in enumerate(("Uber", "Lyft")):
        for j, month in enumerate(("April", "May")):
            ax = axes[i, j]
            total = monthly[service][j]
            vals = [counts[(service, status)][j] / total * 100 for status in statuses]
            bars = ax.barh(statuses[::-1], vals[::-1], color=colors[service])
            ax.set_xscale("symlog", linthresh=0.001)
            ax.set_xlim(0, 200)
            ax.set_title(f"{service}, {month} 2026 (n={total:,})")
            ax.set_xlabel("Share of retained service trips (%)")
            for bar, value in zip(bars, vals[::-1]):
                ax.text(
                    max(value, 0.001) * 1.18,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f}%",
                    va="center",
                    fontsize=8,
                )
            ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Request / match status distribution (request flag first)")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "sharing_status_distribution.png", dpi=170, bbox_inches="tight"
    )
    plt.close(fig)

    rules = read("audit_rules.csv")
    sums = {}
    for r in rules:
        if r["source_file"] == "TOTAL":
            sums[r["rule"]] = int(r["affected_records"])
    if not sums:
        for r in rules:
            sums[r["rule"]] = sums.get(r["rule"], 0) + int(r["affected_records"])
    selected = sorted(
        (
            (k, v)
            for k, v in sums.items()
            if v > 0 and k not in ("pickup_file_month_mismatch",)
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )[:10]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    names, values = zip(*selected)
    ax.barh([n.replace("_", " ") for n in names[::-1]], values[::-1], color="#537b8e")
    ax.set_xscale("log")
    ax.set_xlabel("Flagged records (log scale; flags can overlap)")
    ax.set_title("Most frequent audit flags across 51.04 million source records")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "audit_flags.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    report = OUT / "report.tex"
    tex = report.read_text(encoding="utf-8").replace(".pdf}", ".png}")
    marker = "% Task 1 visualizations"
    if marker not in tex:
        tex += (
            "\n"
            + marker
            + "\n"
            + "\n".join(
                [
                    r"\begin{figure}[htbp]\centering\includegraphics[width=.68\textwidth]{04_Results/Task_1/figures/retained_trips_by_service.png}",
                    r"\caption{Retained completed trip records by month and service. Juno (HV0002) and Via (HV0004) have zero records; all four named HVFHV companies reconcile to the HVFHV total.}\label{fig:t1service}\end{figure}",
                    r"\begin{figure}[htbp]\centering\includegraphics[width=.83\textwidth]{04_Results/Task_1/figures/sharing_status_distribution.png}",
                    r"\caption{Request/match categories for the two companies with retained records, Uber and Lyft. Juno and Via have zero denominator, so shares are undefined. The symmetric log axis preserves visibility of rare statuses and zero values.}\label{fig:t1status}\end{figure}",
                    r"\begin{figure}[htbp]\centering\includegraphics[width=.73\textwidth]{04_Results/Task_1/figures/audit_flags.png}",
                    r"\caption{Most frequent diagnostic flags. Flags overlap, so bar lengths must not be added.}\label{fig:t1audit}\end{figure}",
                ]
            )
            + "\n"
        )
    report.write_text(tex, encoding="utf-8")
    print("Wrote 3 Task 1 charts and updated report.tex")


if __name__ == "__main__":
    main()
