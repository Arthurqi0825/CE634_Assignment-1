"""Task 5.2: potential sharing among non-shared HVFHV trips.

Run from the project root, pointing ASSIGNMENT_RAW_DIR at the directory that
contains the four TLC Parquet files when they are stored outside the package:

  ASSIGNMENT_RAW_DIR=/path/to/Raw/TLC_Trip_Records \
  python -m analysis.helpers.task_outputs run 5 -- \
    python -m analysis.tasks.task5_potential_sharing

The script scans the two HVFHV files only. It does not write to the raw-data
directory.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib

from analysis.common import (
    PROJECT_ROOT,
    RAW_DIR,
    STUDY_MONTHS,
    load_zone_lookup,
    task_output_dir,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = PROJECT_ROOT
RAW = RAW_DIR
OUT = task_output_dir(5)
TABLES, FIGURES = OUT / "tables", OUT / "figures"
MONTHS = list(STUDY_MONTHS)
SERVICES = ["Uber", "Lyft"]
WINDOWS = [5, 10, 15]
MAIN_WINDOW = 10
SIZE = 266
START, MAY, END = map(np.datetime64, ("2026-04-01", "2026-05-01", "2026-06-01"))
START_NS = START.astype("datetime64[ns]").astype("int64")
MAX_TIME_BINS = 20_000


def pack_cell(
    month: np.ndarray,
    service: np.ndarray,
    pu: np.ndarray,
    do: np.ndarray,
    time_bin: np.ndarray,
) -> np.ndarray:
    """Pack a zone-time sharing cell into a sortable int64 key."""
    return (
        (((month.astype(np.int64) * 2 + service.astype(np.int64)) * SIZE + pu) * SIZE + do)
        * MAX_TIME_BINS
        + time_bin
    )


def unpack_cell(key: int) -> tuple[int, int, int, int, int]:
    time_bin = key % MAX_TIME_BINS
    rest = key // MAX_TIME_BINS
    do = rest % SIZE
    rest //= SIZE
    pu = rest % SIZE
    rest //= SIZE
    month = rest // 2
    service = rest % 2
    return int(month), int(service), int(pu), int(do), int(time_bin)


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {name}")
    with (TABLES / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(num: int, den: int) -> float | None:
    return float(num / den) if den else None


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    lookup = load_zone_lookup()
    valid_ids = np.zeros(SIZE, dtype=bool)
    valid_ids[list(lookup)] = True

    counters: dict[int, Counter[int]] = {w: Counter() for w in WINDOWS}
    audit = defaultdict(int)

    cols = [
        "hvfhs_license_num",
        "pickup_datetime",
        "dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_miles",
        "shared_request_flag",
        "shared_match_flag",
    ]

    for file_month in ("04", "05"):
        path = RAW / f"fhvhv_tripdata_2026-{file_month}.parquet"
        pf = pq.ParquetFile(path)
        scanned = 0
        print(f"Scanning {path}: {pf.metadata.num_rows:,} rows", flush=True)
        for batch in pf.iter_batches(batch_size=600_000, columns=cols):
            df = batch.to_pandas()
            scanned += len(df)

            company = df.hvfhs_license_num.to_numpy()
            service = np.where(
                company == "HV0003", 0, np.where(company == "HV0005", 1, -1)
            )

            pickup = pd.to_datetime(df.pickup_datetime, errors="coerce")
            dropoff = pd.to_datetime(df.dropoff_datetime, errors="coerce")
            p = pickup.to_numpy(dtype="datetime64[ns]")
            d = dropoff.to_numpy(dtype="datetime64[ns]")
            month = np.where(
                (p >= START) & (p < MAY), 0, np.where((p >= MAY) & (p < END), 1, -1)
            )
            completed = ~np.isnat(p) & ~np.isnat(d) & (d > p)

            status = (
                df.shared_request_flag.fillna("?").astype(str)
                + "/"
                + df.shared_match_flag.fillna("?").astype(str)
            ).to_numpy()
            nn_status = status == "N/N"

            pu = pd.to_numeric(df.PULocationID, errors="coerce").to_numpy(dtype=float)
            do = pd.to_numeric(df.DOLocationID, errors="coerce").to_numpy(dtype=float)
            pu_i = np.where(np.isfinite(pu), pu, 0).astype(np.int32)
            do_i = np.where(np.isfinite(do), do, 0).astype(np.int32)
            pu_ok = np.isfinite(pu) & (pu == pu_i) & (pu_i > 0) & (pu_i < SIZE)
            do_ok = np.isfinite(do) & (do == do_i) & (do_i > 0) & (do_i < SIZE)
            pu_ok[pu_ok] &= valid_ids[pu_i[pu_ok]]
            do_ok[do_ok] &= valid_ids[do_i[do_ok]]

            miles = pd.to_numeric(df.trip_miles, errors="coerce").to_numpy(dtype=float)
            distance_ok = np.isfinite(miles) & (miles > 0) & (miles <= 100)

            relevant = (service >= 0) & (month >= 0) & completed & nn_status
            eligible = relevant & pu_ok & do_ok & distance_ok

            for m in (0, 1):
                for s in (0, 1):
                    group = relevant & (month == m) & (service == s)
                    audit[(m, s, "completed_nn")] += int(group.sum())
                    audit[(m, s, "invalid_od")] += int((group & ~(pu_ok & do_ok)).sum())
                    audit[(m, s, "invalid_distance")] += int(
                        (group & ~distance_ok).sum()
                    )
                    audit[(m, s, "eligible_nn")] += int((eligible & group).sum())

            if not eligible.any():
                continue

            pickup_ns = pickup.to_numpy(dtype="datetime64[ns]").astype("int64")
            rel_ns = pickup_ns[eligible] - START_NS
            for window in WINDOWS:
                width_ns = window * 60 * 1_000_000_000
                time_bin = (rel_ns // width_ns).astype(np.int64)
                keys = pack_cell(
                    month[eligible],
                    service[eligible],
                    pu_i[eligible].astype(np.int64),
                    do_i[eligible].astype(np.int64),
                    time_bin,
                )
                unique, counts = np.unique(keys, return_counts=True)
                counters[window].update(
                    {int(key): int(count) for key, count in zip(unique, counts)}
                )

        assert scanned == pf.metadata.num_rows
        print(f"  scanned {scanned:,}", flush=True)

    sensitivity_rows = []
    main_cells = counters[MAIN_WINDOW]
    zone_agg = defaultdict(lambda: {"eligible": 0, "potential": 0, "pairs": 0})
    od_agg = defaultdict(lambda: {"eligible": 0, "potential": 0, "pairs": 0})

    for window in WINDOWS:
        by_group = defaultdict(lambda: {"eligible": 0, "potential": 0, "pairs": 0, "multi_cells": 0, "cells": 0})
        for cell_key, n in counters[window].items():
            m, s, pu_id, do_id, _time_bin = unpack_cell(cell_key)
            key = (m, s)
            by_group[key]["eligible"] += n
            by_group[key]["cells"] += 1
            if n >= 2:
                by_group[key]["potential"] += n
                by_group[key]["pairs"] += n * (n - 1) // 2
                by_group[key]["multi_cells"] += 1
            if window == MAIN_WINDOW:
                zkey = (m, s, pu_id)
                odkey = (m, s, pu_id, do_id)
                zone_agg[zkey]["eligible"] += n
                od_agg[odkey]["eligible"] += n
                if n >= 2:
                    zone_agg[zkey]["potential"] += n
                    zone_agg[zkey]["pairs"] += n * (n - 1) // 2
                    od_agg[odkey]["potential"] += n
                    od_agg[odkey]["pairs"] += n * (n - 1) // 2

        for m in (0, 1):
            for s in (0, 1):
                vals = by_group[(m, s)]
                sensitivity_rows.append(
                    {
                        "window_minutes": window,
                        "month": MONTHS[m],
                        "service": SERVICES[s],
                        "eligible_nonshared_trips": vals["eligible"],
                        "potentially_shareable_trips": vals["potential"],
                        "potentially_shareable_share": pct(vals["potential"], vals["eligible"]),
                        "candidate_trip_pairs": vals["pairs"],
                        "nonempty_cells": vals["cells"],
                        "multi_trip_cells": vals["multi_cells"],
                    }
                )

    zone_rows = []
    for (m, s, pu_id), vals in zone_agg.items():
        zone = lookup.get(pu_id, {})
        zone_rows.append(
            {
                "month": MONTHS[m],
                "service": SERVICES[s],
                "PULocationID": pu_id,
                "pickup_zone": zone.get("Zone", ""),
                "borough": zone.get("Borough", ""),
                "eligible_nonshared_trips": vals["eligible"],
                "potentially_shareable_trips": vals["potential"],
                "potentially_shareable_share": pct(vals["potential"], vals["eligible"]),
                "candidate_trip_pairs": vals["pairs"],
            }
        )
    zone_rows.sort(
        key=lambda r: (
            r["month"],
            r["service"],
            -r["potentially_shareable_trips"],
            r["pickup_zone"],
        )
    )

    od_rows = []
    for (m, s, pu_id, do_id), vals in od_agg.items():
        pu_zone = lookup.get(pu_id, {})
        do_zone = lookup.get(do_id, {})
        od_rows.append(
            {
                "month": MONTHS[m],
                "service": SERVICES[s],
                "PULocationID": pu_id,
                "DOLocationID": do_id,
                "origin_zone": pu_zone.get("Zone", ""),
                "destination_zone": do_zone.get("Zone", ""),
                "eligible_nonshared_trips": vals["eligible"],
                "potentially_shareable_trips": vals["potential"],
                "potentially_shareable_share": pct(vals["potential"], vals["eligible"]),
                "candidate_trip_pairs": vals["pairs"],
            }
        )
    od_rows.sort(
        key=lambda r: (
            r["month"],
            r["service"],
            -r["potentially_shareable_trips"],
            r["origin_zone"],
            r["destination_zone"],
        )
    )

    audit_rows = []
    for m in (0, 1):
        for s in (0, 1):
            completed = audit[(m, s, "completed_nn")]
            eligible = audit[(m, s, "eligible_nn")]
            invalid_od = audit[(m, s, "invalid_od")]
            invalid_distance = audit[(m, s, "invalid_distance")]
            audit_rows.append(
                {
                    "month": MONTHS[m],
                    "service": SERVICES[s],
                    "completed_nn": completed,
                    "invalid_od": invalid_od,
                    "invalid_distance": invalid_distance,
                    "eligible_nn": eligible,
                    "excluded_union": completed - eligible,
                    "overlap_excess_flags": invalid_od + invalid_distance - (completed - eligible),
                }
            )

    write_csv("potential_sharing_sensitivity.csv", sensitivity_rows)
    write_csv("potential_sharing_zone_summary.csv", zone_rows)
    write_csv("potential_sharing_top_od.csv", od_rows[:80])
    write_csv("potential_sharing_audit.csv", audit_rows)

    main_rows = [r for r in sensitivity_rows if r["window_minutes"] == MAIN_WINDOW]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), layout="constrained")
    labels = [f"{r['service']}\n{r['month'][-2]}" for r in main_rows]
    x = np.arange(len(main_rows))
    shares = [100 * r["potentially_shareable_share"] for r in main_rows]
    colors = ["#2f6f9f" if r["service"] == "Uber" else "#7b4b8a" for r in main_rows]
    axes[0].bar(x, shares, color=colors)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Potentially shareable N/N trips (%)")
    axes[0].set_title("Same OD and 10-minute pickup window")
    for xi, val in zip(x, shares):
        axes[0].text(xi, val + 0.8, f"{val:.1f}%", ha="center", fontsize=9, weight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", color="#e4e8eb")

    for service, color in [("Uber", "#2f6f9f"), ("Lyft", "#7b4b8a")]:
        for month, style in [("2026-04", "-"), ("2026-05", "--")]:
            rows = [
                r
                for r in sensitivity_rows
                if r["service"] == service and r["month"] == month
            ]
            rows.sort(key=lambda r: r["window_minutes"])
            axes[1].plot(
                [r["window_minutes"] for r in rows],
                [100 * r["potentially_shareable_share"] for r in rows],
                marker="o",
                linestyle=style,
                color=color,
                label=f"{service} {month[-2:]}",
            )
    axes[1].set_xlabel("Pickup-time window (minutes)")
    axes[1].set_ylabel("Potentially shareable N/N trips (%)")
    axes[1].set_title("Threshold sensitivity")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="y", color="#e4e8eb")
    fig.suptitle("Task 5.2 potential sharing among non-shared trips", weight="bold")
    fig.savefig(FIGURES / "potential_sharing.png", dpi=180)
    plt.close(fig)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": "Task 5.2: How many non-shared trips are potentially shareable?",
        "method": "Uber and Lyft N/N completed trips with valid official directed OD zones and 0<trip_miles<=100. A trip is counted as potentially shareable when at least one other non-shared trip from the same platform and pickup month has the same directed OD pair and falls in the same pickup-time window. The main rule uses a 10-minute window; 5- and 15-minute windows are reported as sensitivity checks. This is a coarse zone-time proxy, not a dispatch simulation.",
        "main_window_minutes": MAIN_WINDOW,
        "main_results": main_rows,
        "sensitivity": sensitivity_rows,
        "audit": audit_rows,
        "top_pickup_zones": zone_rows[:40],
        "top_directed_od": od_rows[:40],
        "limitations": [
            "Taxi Zones are coarse and do not provide exact pickup/dropoff coordinates.",
            "The rule does not estimate detour distance, waiting tolerance, vehicle capacity, or platform dispatch feasibility.",
            "Trips near opposite sides of a time-window boundary may be missed; trips in the same bin may still be practically incompatible.",
            "The count is therefore a transparent potential-shareability screen, not a realized matching rate or causal estimate.",
        ],
    }

    summary_path = OUT / "summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    existing["task5_2"] = summary
    (OUT / "summary.json").write_text(
        json.dumps(existing, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report = [
        r"\subsection{Task 5.2: Potentially shareable non-shared trips}",
        r"To estimate how many non-shared trips may have had a plausible sharing partner, we screen Uber and Lyft $N/N$ trips using a transparent zone-time rule. A trip is counted as potentially shareable if at least one other $N/N$ trip from the same platform and pickup month has the same directed OD pair and falls in the same pickup-time window. The main specification uses a 10-minute pickup window; 5- and 15-minute windows are reported as sensitivity checks. Completed trips must have official pickup/dropoff zones and $0<\mathrm{trip\_miles}\leq100$.",
        r"\begin{table}[htbp]\centering\small\caption{Task 5.2 potential shareability under the same-OD, 10-minute pickup-window rule.}\label{tab:t5potential}\begin{tabular}{llrrrr}\hline Month & Platform & Eligible $N/N$ & Potentially shareable & Share & Candidate pairs \\ \hline",
    ]
    for r in main_rows:
        report.append(
            f"{r['month']} & {r['service']} & {r['eligible_nonshared_trips']:,} & {r['potentially_shareable_trips']:,} & {100*r['potentially_shareable_share']:.1f}\\% & {r['candidate_trip_pairs']:,} "
            + r"\\"
        )
    report.extend(
        [
            r"\hline\end{tabular}\end{table}",
            r"\begin{figure}[htbp]\centering\includegraphics[width=.82\textwidth]{04_Results/Task_5/figures/potential_sharing.png}\caption{Potentially shareable non-shared trips under same-OD pickup-window rules. The left panel reports the 10-minute main rule; the right panel shows sensitivity to 5-, 10-, and 15-minute windows.}\label{fig:t5potential}\end{figure}",
            r"This screen suggests that a substantial share of non-shared completed trips had at least one coarse zone-time neighbor. Under the 10-minute rule, Uber has about "
            + f"{100*main_rows[0]['potentially_shareable_share']:.1f}\\% in April and {100*main_rows[2]['potentially_shareable_share']:.1f}\\% in May; Lyft has about {100*main_rows[1]['potentially_shareable_share']:.1f}\\% and {100*main_rows[3]['potentially_shareable_share']:.1f}\\%. "
            + r"The sensitivity table shows the expected monotonic pattern: wider pickup windows increase the potential-shareable count. These values should not be interpreted as trips that definitely could be pooled. Taxi Zones hide exact endpoints, the rule does not model waiting tolerance, detour cost, vehicle capacity, dispatch feasibility, or passenger willingness, and trips close to a time-window boundary may be misclassified. The result is therefore an upper-screening measure of latent co-occurrence in the non-shared trip stream, not a platform matching rate.",
        ]
    )
    (OUT / "report_task5_2.tex").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps(summary["main_results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
