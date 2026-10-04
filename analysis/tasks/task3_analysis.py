"""Task 3 HVFHV sharing rates by pickup month, company and TLC pickup zone.

Run: python -m analysis.helpers.task_outputs run 3 -- python -m analysis.tasks.task3_analysis
"""
from __future__ import annotations

import csv
import json
from datetime import datetime

import matplotlib

from analysis.common import PROJECT_ROOT, RAW_DIR, STUDY_MONTHS, task_output_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm
from matplotlib.patches import Patch, Polygon

from analysis.tasks.task2_analysis import load_shapes, zone_lookup

ROOT = PROJECT_ROOT
RAW = RAW_DIR
OUT = task_output_dir(3)
TABLES, FIGURES = OUT / "tables", OUT / "figures"
MONTHS = STUDY_MONTHS
SERVICES = ("Uber", "Lyft")
STATUSES = ("N/N", "Y/N", "Y/Y", "N/Y", "Missing/other")
START, MAY, END = (np.datetime64(x) for x in ("2026-04-01", "2026-05-01", "2026-06-01"))


def write_csv(name, rows):
    if not rows:
        raise ValueError(name)
    with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {name}: {len(rows)} rows", flush=True)


def rates(values):
    nn, yn, yy, ny, other = (int(x) for x in values)
    v, requests, matched = nn + yn + yy + ny, yn + yy, ny + yy
    return {
        "n_nn": nn,
        "n_yn": yn,
        "n_yy": yy,
        "n_ny": ny,
        "missing_other": other,
        "valid_status_denominator": v,
        "pooling_requests": requests,
        "reported_matches": matched,
        "pooling_request_rate": requests / v if v else None,
        "reported_matched_rate": matched / v if v else None,
        "matching_success": yy / requests if requests else None,
    }


def draw_rate_map(shapes, lookup, row_by_id, month, service, metric, label, maximum):
    fig, ax = plt.subplots(figsize=(7.7, 7.2))
    # One scale for a metric across both services and months.
    norm = PowerNorm(gamma=0.6, vmin=0, vmax=maximum)
    cmap = plt.get_cmap("viridis")
    for zone, parts in shapes.items():
        row = row_by_id.get(zone)
        if row is None or row[metric + "_status"] == "undefined":
            color = "#e8edf0"
        elif row[metric + "_status"] == "insufficient sample":
            color = "#bfc7ce"
        else:
            color = cmap(norm(min(row[metric], maximum)))
        for part in parts:
            ax.add_patch(
                Polygon(
                    part,
                    closed=True,
                    facecolor=color,
                    edgecolor="#aeb9bf",
                    linewidth=0.16,
                )
            )
    ax.set_xlim(900000, 1070000)
    ax.set_ylim(110000, 280000)
    ax.set_aspect("equal")
    ax.axis("off")
    ticks = np.linspace(0, maximum, 6)
    cb = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.68,
        label=label,
        ticks=ticks,
    )
    cb.ax.set_yticklabels([f"{100*t:.1f}%" for t in ticks])
    ax.set_title(f"{month} {service}: {label} by pickup zone", loc="left", fontsize=12)
    ax.legend(
        handles=[
            Patch(facecolor="#bfc7ce", label="insufficient sample"),
            Patch(facecolor="#e8edf0", label="undefined / no trips"),
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
    )
    fig.text(
        0.035,
        0.025,
        "Shared nonlinear color scale per rate; denominators: zone_rates.csv. Zones 264/265 have no polygon.",
        fontsize=7.5,
    )
    path = FIGURES / f"{metric}_{month}_{service.lower()}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_overview(monthly):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), sharey=True)
    items = [
        ("pooling_request_rate", "Pooling request rate"),
        ("reported_matched_rate", "Reported matched rate"),
        ("matching_success", "Matching success among requests"),
    ]
    colors = {"Uber": "#356c9d", "Lyft": "#bb5b84"}
    for ax, (metric, title) in zip(axes, items):
        for i, month in enumerate(MONTHS):
            for j, service in enumerate(SERVICES):
                row = next(
                    r
                    for r in monthly
                    if r["month"] == month and r["service"] == service
                )
                value = row[metric] * 100 if row[metric] is not None else 0
                x = i * 2.5 + j * 0.8
                ax.bar(x, value, width=0.7, color=colors[service])
                ax.text(x, value + 0.6, f"{value:.2f}%", ha="center", fontsize=7.5)
        ax.set_xticks([0.4, 2.9], ["April", "May"])
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Percent of relevant denominator")
    fig.legend(
        handles=[Patch(color=colors[s], label=s) for s in SERVICES],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "monthly_sharing_rates.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(summary):
    lines = [
        r"\subsection{Task 3: HVFHV sharing and reporting consistency}",
        r"Only Uber (HV0003) and Lyft (HV0005) are used. Pickup time assigns month; a completed trip requires valid pickup/dropoff timestamps and a positive timestamp duration. For each service-month and pickup zone, $V=n_{NN}+n_{YN}+n_{YY}+n_{NY}$. We calculate pooling request rate $(n_{YN}+n_{YY})/V$, reported matched rate $(n_{NY}+n_{YY})/V$, and matching success among requests $n_{YY}/(n_{YN}+n_{YY})$. Missing/other flags are separate from $V$; a zero denominator is undefined. The $N/Y$ category remains in $V$ because it is a required reporting-consistency category.",
        r"\begin{table}[htbp]\centering\small\caption{HVFHV sharing rates; denominators are valid-status trips $V$ and pooling requests $Q$.}\label{tab:t3rates}\begin{tabular}{llrrrrr}\hline Month & Service & $V$ & $Q$ & Request \% & Matched \% & Success \% \\ \hline",
    ]
    for r in summary["monthly"]:
        lines.append(
            f"{r['month']} & {r['service']} & {r['valid_status_denominator']:,} & {r['pooling_requests']:,} & {100*r['pooling_request_rate']:.2f} & {100*r['reported_matched_rate']:.2f} & {100*r['matching_success']:.2f} "
            + r"\\"
        )
    lines += [
        r"\hline\end{tabular}\end{table}",
        r"The twelve PNG maps cover all three rates by service and month. For zone comparisons, pooling request and reported matched rates need at least 100 valid-status trips; matching success needs at least 100 requests. Below-threshold zones are marked insufficient sample in gray, not shown as zero. The CSV gives every numerator, denominator, missing/other count, and status. Zones 264/265 remain in the tables but have no drawable polygon.",
        r"\begin{figure}[htbp]\centering\includegraphics[width=.9\textwidth]{04_Results/Task_3/figures/monthly_sharing_rates.png}\caption{Monthly platform sharing rates. The three panels use their stated denominators.}\label{fig:t3overview}\end{figure}",
        r"\begin{figure}[htbp]\centering\includegraphics[width=.46\textwidth]{04_Results/Task_3/figures/reported_matched_rate_2026-04_uber.png}\hfill\includegraphics[width=.46\textwidth]{04_Results/Task_3/figures/matching_success_2026-04_uber.png}\caption{April Uber reported matched rate and matching success by pickup zone, with sample thresholds.}\label{fig:t3maps}\end{figure}",
    ]
    for finding in summary["findings"]:
        lines.append(finding)
    for finding in summary["geographic_findings"]:
        lines.append(finding)
    lines.append(
        r"No Uber or Lyft records in either month had missing/other request-match statuses after the completed-trip filter. Lyft has $N/Y$ records (783 in April, 1,177 in May); Uber has none. These are counted in reported matches but are not pooling requests."
    )
    lines.append(
        r"These are rates among observed completed trips, not among every app request. A reported match does not verify partner identity, overlap duration, or whether the full ride was pooled; $N/Y$ records indicate a flag inconsistency and may inflate the reported matched rate, particularly for Lyft. Differences between services can reflect customer, geography, and product mix rather than platform performance alone."
    )
    (OUT / "report.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    lookup = zone_lookup()
    zone_ok = np.zeros(266, dtype=bool)
    zone_ok[list(lookup)] = True
    totals = np.zeros((2, 2, 5), dtype=np.int64)
    zones = np.zeros((2, 2, 266, 5), dtype=np.int64)
    invalid_pu = np.zeros((2, 2), dtype=np.int64)
    for month_file in ("04", "05"):
        path = RAW / f"fhvhv_tripdata_2026-{month_file}.parquet"
        pf = pq.ParquetFile(path)
        print(f"Scanning {path.name}: {pf.metadata.num_rows:,} rows", flush=True)
        scanned = 0
        cols = [
            "hvfhs_license_num",
            "pickup_datetime",
            "dropoff_datetime",
            "PULocationID",
            "shared_request_flag",
            "shared_match_flag",
        ]
        for batch in pf.iter_batches(batch_size=300_000, columns=cols):
            f = batch.to_pandas()
            n = len(f)
            scanned += n
            pickup = pd.to_datetime(f["pickup_datetime"], errors="coerce").to_numpy(
                dtype="datetime64[ns]"
            )
            dropoff = pd.to_datetime(f["dropoff_datetime"], errors="coerce").to_numpy(
                dtype="datetime64[ns]"
            )
            month = np.where(
                (pickup >= START) & (pickup < MAY),
                0,
                np.where((pickup >= MAY) & (pickup < END), 1, -1),
            )
            core = (month >= 0) & ~np.isnat(dropoff) & (dropoff > pickup)
            company = f["hvfhs_license_num"].fillna("Missing").to_numpy()
            service = np.where(
                company == "HV0003", 0, np.where(company == "HV0005", 1, -1)
            )
            req = f["shared_request_flag"].fillna("Missing").to_numpy()
            match = f["shared_match_flag"].fillna("Missing").to_numpy()
            status = np.full(n, 4, dtype=np.int8)
            status[(req == "N") & (match == "N")] = 0
            status[(req == "Y") & (match == "N")] = 1
            status[(req == "Y") & (match == "Y")] = 2
            status[(req == "N") & (match == "Y")] = 3
            pu_float = pd.to_numeric(f["PULocationID"], errors="coerce").to_numpy(
                dtype=float
            )
            pu = np.where(np.isfinite(pu_float), pu_float, 0).astype(np.int16)
            valid_pu = (pu > 0) & (pu < 266) & np.isfinite(pu_float) & (pu_float == pu)
            valid_pu[valid_pu] &= zone_ok[pu[valid_pu]]
            for m in (0, 1):
                for s in (0, 1):
                    base = core & (month == m) & (service == s)
                    totals[m, s] += np.bincount(status[base], minlength=5)
                    invalid_pu[m, s] += int((base & ~valid_pu).sum())
                    good = base & valid_pu
                    packed = pu[good].astype(np.int32) * 5 + status[good]
                    zones[m, s] += np.bincount(packed, minlength=266 * 5).reshape(
                        266, 5
                    )
        if scanned != pf.metadata.num_rows:
            raise AssertionError((path, scanned, pf.metadata.num_rows))
        print(f"  scanned {scanned:,}", flush=True)
    task1_rows = list(
        csv.DictReader((ROOT / "04_Results/Task_1/tables/task1_2_counts.csv").open())
    )
    expected = {
        (r["service"], r["request_match_category"]): (
            int(r["april_trips"]),
            int(r["may_trips"]),
        )
        for r in task1_rows
    }
    monthly, zone_rows = [], []
    for m, month in enumerate(MONTHS):
        for s, service in enumerate(SERVICES):
            for i, status in enumerate(STATUSES):
                if totals[m, s, i] != expected[(service, status)][m]:
                    raise AssertionError(
                        f"Task 1 status mismatch: {month} {service} {status}"
                    )
            if zones[m, s].sum() + invalid_pu[m, s] != totals[m, s].sum():
                raise AssertionError("Pickup zone reconciliation failure")
            row = {
                "month": month,
                "service": service,
                **rates(totals[m, s]),
                "completed_trips": int(totals[m, s].sum()),
                "invalid_pickup_zone": int(invalid_pu[m, s]),
            }
            monthly.append(row)
            for z in sorted(lookup):
                rr = {
                    "month": month,
                    "service": service,
                    "PULocationID": z,
                    "borough": lookup[z]["Borough"],
                    "zone": lookup[z]["Zone"],
                    **rates(zones[m, s, z]),
                }
                for metric in (
                    "pooling_request_rate",
                    "reported_matched_rate",
                    "matching_success",
                ):
                    denom = (
                        rr["pooling_requests"]
                        if metric == "matching_success"
                        else rr["valid_status_denominator"]
                    )
                    rr[metric + "_status"] = (
                        "undefined"
                        if denom == 0
                        else ("reported" if denom >= 100 else "insufficient sample")
                    )
                zone_rows.append(rr)
    write_csv("monthly_rates.csv", monthly)
    write_csv("zone_rates.csv", zone_rows)
    draw_overview(monthly)
    shapes = load_shapes()
    metric_labels = {
        "pooling_request_rate": "Pooling request rate",
        "reported_matched_rate": "Reported matched rate",
        "matching_success": "Matching success among requests",
    }
    shared_scale = {}
    for metric in metric_labels:
        values = [r[metric] for r in zone_rows if r[metric + "_status"] == "reported"]
        shared_scale[metric] = float(np.ceil(max(values) * 100) / 100)
    for month in MONTHS:
        for service in SERVICES:
            by_id = {
                r["PULocationID"]: r
                for r in zone_rows
                if r["month"] == month and r["service"] == service
            }
            for metric, label in metric_labels.items():
                draw_rate_map(
                    shapes,
                    lookup,
                    by_id,
                    month,
                    service,
                    metric,
                    label,
                    shared_scale[metric],
                )
    findings = []
    for month in MONTHS:
        uber = next(
            r for r in monthly if r["month"] == month and r["service"] == "Uber"
        )
        lyft = next(
            r for r in monthly if r["month"] == month and r["service"] == "Lyft"
        )
        findings.append(
            f"In {month}, Uber's pooling request rate was {100*uber['pooling_request_rate']:.2f}\\% and reported matched rate {100*uber['reported_matched_rate']:.2f}\\%, versus Lyft's {100*lyft['pooling_request_rate']:.2f}\\% and {100*lyft['reported_matched_rate']:.2f}\\%. Uber's matching success among {uber['pooling_requests']:,} requests was {100*uber['matching_success']:.2f}\\%; Lyft's among {lyft['pooling_requests']:,} requests was {100*lyft['matching_success']:.2f}\\%."
        )
    threshold_counts = []
    geographic_findings = []
    for month in MONTHS:
        for service in SERVICES:
            z = [
                r for r in zone_rows if r["month"] == month and r["service"] == service
            ]
            threshold_counts.append(
                {
                    "month": month,
                    "service": service,
                    "zones_matched_reported": sum(
                        r["reported_matched_rate_status"] == "reported" for r in z
                    ),
                    "zones_success_reported": sum(
                        r["matching_success_status"] == "reported" for r in z
                    ),
                }
            )
            eligible = sorted(
                (r for r in z if r["matching_success_status"] == "reported"),
                key=lambda r: (-r["matching_success"], r["PULocationID"]),
            )
            if eligible:
                lead = eligible[0]
                geographic_findings.append(
                    f"For {service} in {month}, {len(eligible)} pickup zones meet the 100-request matching-success threshold. "
                    f"The highest reported matching success among these is {lead['zone']} "
                    f"({100*lead['matching_success']:.1f}\\% of {lead['pooling_requests']:,} requests)."
                )
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "Pickup timestamp month; completed-trip mask from Task 1. Valid statuses N/N,Y/N,Y/Y,N/Y. Missing/other excluded from V. Zone rates require a valid lookup pickup ID; thresholds 100 V for pooling/matched and 100 requests for success.",
        "monthly": monthly,
        "zone_threshold_counts": threshold_counts,
        "findings": findings,
        "geographic_findings": geographic_findings,
        "figure_count": 13,
        "map_rate_scale_maximum": shared_scale,
        "all_status_totals_reconcile_to_task1": True,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(summary)
    print(
        "Task 3 complete: 13 PNG figures; all status totals reconcile with Task 1",
        flush=True,
    )


if __name__ == "__main__":
    main()
