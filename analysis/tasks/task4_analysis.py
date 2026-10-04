"""Task 4: Uber Y/Y observed travel time versus supported N/N OD/hour/day-type medians.

Run: python -m analysis.helpers.task_outputs run 4 -- python -m analysis.tasks.task4_analysis
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime

import matplotlib

from analysis.common import PROJECT_ROOT, RAW_DIR, STUDY_MONTHS, task_output_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch, Polygon

from analysis.tasks.task2_analysis import load_shapes, zone_lookup

ROOT = PROJECT_ROOT
RAW = RAW_DIR
OUT = task_output_dir(4)
TABLES, FIGURES = OUT / "tables", OUT / "figures"
MONTHS = STUDY_MONTHS
START, MAY, END = (np.datetime64(x) for x in ("2026-04-01", "2026-05-01", "2026-06-01"))
ZONE_BASE = 266


def write_csv(name, rows):
    if not rows:
        raise ValueError(name)
    with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: ("" if isinstance(v, float) and not np.isfinite(v) else v)
                    for k, v in row.items()
                }
            )
    print(f"Wrote {name}: {len(rows):,} rows", flush=True)


def key_of(pu, do, hour, weekend):
    return (
        ((pu.astype(np.int64) * ZONE_BASE + do.astype(np.int64)) * 24 + hour) * 2
        + weekend
    ).astype(np.int32)


def quantiles(series):
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return (None, None, None)
    return tuple(float(x) for x in np.quantile(values, [0.25, 0.5, 0.75]))


def grouped_summaries(frame, fields, lookup=None):
    rows = []
    for key, group in frame.groupby(fields, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(fields, key))
        if lookup is not None and "PULocationID" in row:
            zid = int(row["PULocationID"])
            row.update({"borough": lookup[zid]["Borough"], "zone": lookup[zid]["Zone"]})
        e25, e50, e75 = quantiles(group["excess_minutes"])
        r25, r50, r75 = quantiles(group["relative_excess_percent"])
        valid = group["excess_minutes"].notna()
        row.update(
            {
                "shared_eligible": len(group),
                "reference_supported": int(valid.sum()),
                "reference_coverage": float(valid.mean()),
                "excess_p25_minutes": e25,
                "excess_median_minutes": e50,
                "excess_p75_minutes": e75,
                "relative_p25_percent": r25,
                "relative_median_percent": r50,
                "relative_p75_percent": r75,
                "mean_excess_minutes": float(group.loc[valid, "excess_minutes"].mean())
                if valid.any()
                else None,
            }
        )
        rows.append(row)
    return rows


def plot_hour(hour_rows, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"2026-04": "#356c9d", "2026-05": "#ba5a83"}
    for month in MONTHS:
        rows = sorted(
            (r for r in hour_rows if r["month"] == month),
            key=lambda x: x["pickup_hour"],
        )
        x = np.array([r["pickup_hour"] for r in rows])
        med = np.array([r[metric] for r in rows], dtype=float)
        low = np.array([r[metric.replace("median", "p25")] for r in rows], dtype=float)
        high = np.array([r[metric.replace("median", "p75")] for r in rows], dtype=float)
        ax.plot(x, med, marker="o", markersize=3, color=colors[month], label=month)
        ax.fill_between(x, low, high, color=colors[month], alpha=0.13)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    ax.set_xlabel("Pickup hour")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_zone_map(shapes, zone_rows, month, limit):
    rows = {int(r["PULocationID"]): r for r in zone_rows if r["month"] == month}
    fig, ax = plt.subplots(figsize=(7.7, 7.2))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = plt.get_cmap("RdBu_r")
    for zid, parts in shapes.items():
        r = rows.get(zid)
        value = None if r is None else r["mean_excess_minutes"]
        color = (
            "#c8d0d6"
            if r is None or r["reference_supported"] < 10 or value is None
            else cmap(norm(np.clip(value, -limit, limit)))
        )
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
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.68,
        label="Mean excess minutes (supported shared trips)",
    )
    ax.set_title(
        f"{month} Uber Y/Y: mean excess time by pickup zone", loc="left", fontsize=12
    )
    ax.legend(
        handles=[
            Patch(facecolor="#c8d0d6", label="fewer than 10 supported trips / no data")
        ],
        loc="lower left",
        fontsize=8,
    )
    fig.text(
        0.04,
        0.025,
        "Per-zone counts and coverage: zone_summary.csv. Zones 264/265 have no polygon.",
        fontsize=7.5,
    )
    fig.savefig(FIGURES / f"mean_excess_zone_{month}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_zone(shapes, zone_rows):
    pool = {}
    for r in zone_rows:
        z = int(r["PULocationID"])
        x = pool.setdefault(z, [0, 0])
        x[0] += r["shared_eligible"]
        x[1] += r["reference_supported"]
    fig, ax = plt.subplots(figsize=(7.7, 7.2))
    cmap = plt.get_cmap("YlGnBu")
    norm = Normalize(0, 1)
    for zid, parts in shapes.items():
        total, supported = pool.get(zid, [0, 0])
        color = "#e8edf0" if total == 0 else cmap(norm(supported / total))
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
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.68,
        label="Shared trips with supported reference / eligible shared trips",
    )
    ax.set_title(
        "April–May Uber reference coverage by pickup zone", loc="left", fontsize=12
    )
    fig.text(
        0.04,
        0.025,
        "Coverage uses each shared trip once. Zones 264/265 have no polygon.",
        fontsize=7.5,
    )
    fig.savefig(FIGURES / "reference_coverage_zone.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def other_plots(frame, months, within):
    valid = frame["excess_minutes"].dropna().to_numpy()
    lo, hi = np.quantile(valid, [0.01, 0.99])
    histogram_counts, histogram_edges = np.histogram(
        valid[(valid >= lo) & (valid <= hi)], bins=100
    )
    write_csv(
        "excess_distribution_bins.csv",
        [
            {
                "bin_left_minutes": float(histogram_edges[i]),
                "bin_right_minutes": float(histogram_edges[i + 1]),
                "trip_count": int(count),
                "visual_p01_minutes": float(lo),
                "visual_p99_minutes": float(hi),
            }
            for i, count in enumerate(histogram_counts)
        ],
    )
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.hist(valid[(valid >= lo) & (valid <= hi)], bins=100, color="#517da5")
    ax.axvline(0, color="#454545", lw=1)
    ax.set_xlabel("Observed excess minutes E")
    ax.set_ylabel("Supported Y/Y trips")
    ax.set_title("Uber excess-time distribution (visual range: 1st–99th percentiles)")
    ax.text(
        0.99,
        0.95,
        f"Full-data summaries retain tails and negative values.\nDisplayed range: {lo:.1f} to {hi:.1f} min",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "excess_distribution.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(2)
    axes[0].bar(
        x, [r["reference_coverage"] * 100 for r in months], color=["#356c9d", "#ba5a83"]
    )
    axes[0].set_xticks(x, ["April", "May"])
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Reference coverage (%)")
    axes[0].set_title("Supported among eligible Y/Y trips")
    axes[1].bar(
        x, [r["excess_median_minutes"] for r in months], color=["#356c9d", "#ba5a83"]
    )
    axes[1].set_xticks(x, ["April", "May"])
    axes[1].set_ylabel("Median E (minutes)")
    axes[1].set_title("Trip-weighted median excess")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "monthly_coverage_excess.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["within zone" if r["within_zone"] else "between zones" for r in within]
    ax.bar(
        labels,
        [r["excess_median_minutes"] for r in within],
        color=["#9c6cad", "#4b8a86"],
    )
    for i, r in enumerate(within):
        ax.text(
            i,
            r["excess_median_minutes"] + 0.15,
            f"n={r['reference_supported']:,}",
            ha="center",
            fontsize=8,
        )
    ax.set_ylabel("Median E (minutes)")
    ax.set_title("Trip-weighted within- versus between-zone comparison")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "within_between_excess.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(summary):
    a, b = summary["month_summary"]
    lines = [
        r"\subsection{Task 4: Uber relative excess travel time}",
        r"We use only Uber HV0003 trips. The reference is N/N with valid directed OD, pickup and dropoff timestamps, positive timestamp duration, and recorded \texttt{trip\_time} in $(0,24]$ hours. Reference trips from April and May are pooled by ordered OD $\times$ pickup hour $\times$ weekday/weekend (Monday--Friday versus Saturday--Sunday); month is not part of a reference cell. A cell is supported if it contains at least 30 N/N trips. Its median \texttt{trip\_time}/60 is the reference minutes. Each eligible Y/Y trip is matched to its cell; unsupported comparisons have null $E$ and $R$.",
        r"For trip $i$, $E_i=T_i-T_{ref,i}$ minutes and $R_i=100E_i/T_{ref,i}$ percent. Negative values are retained. Each trip receives equal weight in all medians, quartiles, and zone means; cells do not receive equal weight. Relative differences are calculated for individual trips before aggregation.",
        r"\begin{table}[htbp]\centering\scriptsize\caption{Uber Y/Y reference coverage and trip-weighted excess-time summaries. IQR endpoints are the 25th and 75th percentiles.}\label{tab:t4summary}\begin{tabular}{lrrrrrr}\hline Month & Eligible Y/Y & Supported & Coverage \% & Median $E$ & $E$ IQR & Median $R$ \% \\ \hline",
    ]
    for r in summary["month_summary"]:
        lines.append(
            f"{r['month']} & {r['shared_eligible']:,} & {r['reference_supported']:,} & {100*r['reference_coverage']:.1f} & {r['excess_median_minutes']:.2f} & [{r['excess_p25_minutes']:.2f}, {r['excess_p75_minutes']:.2f}] & {r['relative_median_percent']:.1f} "
            + r"\\"
        )
    lines += [
        r"\hline\end{tabular}\end{table}",
        f"For Uber, {summary['audit']['reference_status_nn']:,} N/N and {summary['audit']['shared_status_yy']:,} Y/Y completed OD trips pass the timestamp and zone checks; {summary['audit']['excluded_nonpositive_or_over24h_trip_time_nn']:,} N/N and {summary['audit']['excluded_nonpositive_or_over24h_trip_time_yy']:,} Y/Y records are then excluded for invalid recorded trip time. No completed Uber trip has a missing or invalid OD zone in these files.",
        f"The supported reference contains {summary['supported_reference_cells']:,} cells, covering {summary['reference_trips_in_supported_cells']:,} of {summary['reference_trip_count']:,} eligible N/N records. Among shared trips without a supported comparison, {summary['unsupported_reference_reasons']['no_reference_cell']:,} have no N/N cell at all and {summary['unsupported_reference_reasons']['reference_cell_under_30']:,} have a cell with fewer than 30 N/N trips. April's relative-excess IQR is [{a['relative_p25_percent']:.1f}, {a['relative_p75_percent']:.1f}]\\%; May's is [{b['relative_p25_percent']:.1f}, {b['relative_p75_percent']:.1f}]\\%. The hourly and zone CSV supplements give both full IQRs and coverage denominators.",
        r"\begin{figure}[htbp]\centering\includegraphics[width=.48\textwidth]{04_Results/Task_4/figures/monthly_coverage_excess.png}\hfill\includegraphics[width=.48\textwidth]{04_Results/Task_4/figures/excess_distribution.png}\caption{Month coverage and excess-time distribution. Histogram tails are trimmed for display only; statistical summaries use all supported trips.}\label{fig:t4summary}\end{figure}",
        r"\begin{figure}[htbp]\centering\includegraphics[width=.48\textwidth]{04_Results/Task_4/figures/hourly_excess_minutes.png}\hfill\includegraphics[width=.48\textwidth]{04_Results/Task_4/figures/hourly_relative_percent.png}\caption{Trip-weighted hourly medians and interquartile bands by pickup month.}\label{fig:t4hour}\end{figure}",
        r"\begin{figure}[htbp]\centering\includegraphics[width=.47\textwidth]{04_Results/Task_4/figures/mean_excess_zone_2026-04.png}\hfill\includegraphics[width=.47\textwidth]{04_Results/Task_4/figures/mean_excess_zone_2026-05.png}\caption{Mean non-null excess minutes by pickup zone, displayed only with at least 10 supported shared trips per month. Both maps share one color scale.}\label{fig:t4zone}\end{figure}",
    ]
    w = summary["within_between"]
    for r in w:
        kind = "Within-zone" if r["within_zone"] else "Between-zone"
        lines.append(
            f"{kind} Y/Y trips have median $E={r['excess_median_minutes']:.2f}$ min and median $R={r['relative_median_percent']:.1f}\\%$ among {r['reference_supported']:,} supported trips; reference coverage is {100*r['reference_coverage']:.1f}\\%."
        )
    for item in summary["hour_findings"]:
        lines.append(
            f"In {item['month']}, the lowest hourly median $E$ was {item['lowest_median_minutes']:.2f} min at {item['lowest_hour']:02d}:00 and the highest was {item['highest_median_minutes']:.2f} min at {item['highest_hour']:02d}:00; hourly reference coverage varies, so these are comparisons among supported trips only."
        )
    for item in summary["zone_findings"]:
        lines.append(
            f"Among {item['zones_with_100_supported']} {item['month']} pickup zones with at least 100 supported shared trips, the lowest mean $E$ was {item['lowest_zone']} ({item['lowest_mean_minutes']:.2f} min) and the highest was {item['highest_zone']} ({item['highest_mean_minutes']:.2f} min). The map includes zones with at least 10 supported trips and clips its color at {summary['zone_map_color_limit_minutes']:.0f} min for readability; underlying CSV means are untrimmed."
        )
    lines.append(
        r"This is an observed comparison, not a causal pooling effect. Trips sharing a TLC zone pair can have different exact endpoints; traffic varies by date even within the same hour/day-type cell; routes and passenger selection differ. The flags do not verify a specific sharing partner or overlap time. Reference coverage also varies by location, so mapped averages reflect a selected subset. The N/N median is an observed reference, not an optimal or congestion-free travel time. The zone coverage map and CSV make that selection visible."
    )
    (OUT / "report.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    lookup = zone_lookup()
    zone_valid = np.zeros(266, dtype=bool)
    zone_valid[list(lookup)] = True
    ref_keys = []
    ref_seconds = []
    shared = []
    audit = {
        "raw_uber_rows": 0,
        "completed_uber": 0,
        "od_eligible_uber": 0,
        "reference_status_nn": 0,
        "shared_status_yy": 0,
        "reference_duration_valid": 0,
        "shared_duration_valid": 0,
        "excluded_missing_or_invalid_od": 0,
        "excluded_nonpositive_or_over24h_trip_time_nn": 0,
        "excluded_nonpositive_or_over24h_trip_time_yy": 0,
    }
    monthly_status = np.zeros((2, 2), dtype=np.int64)  # N/N, Y/Y after core
    for file_month in ("04", "05"):
        path = RAW / f"fhvhv_tripdata_2026-{file_month}.parquet"
        pf = pq.ParquetFile(path)
        scanned = 0
        print(f"Scanning {path.name}: {pf.metadata.num_rows:,} rows", flush=True)
        cols = [
            "hvfhs_license_num",
            "pickup_datetime",
            "dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_time",
            "shared_request_flag",
            "shared_match_flag",
        ]
        for batch in pf.iter_batches(batch_size=300_000, columns=cols):
            f = batch.to_pandas()
            n = len(f)
            scanned += n
            uber = f["hvfhs_license_num"].eq("HV0003").to_numpy()
            audit["raw_uber_rows"] += int(uber.sum())
            pickup = pd.to_datetime(f["pickup_datetime"], errors="coerce")
            dropoff = pd.to_datetime(f["dropoff_datetime"], errors="coerce")
            pk = pickup.to_numpy(dtype="datetime64[ns]")
            dk = dropoff.to_numpy(dtype="datetime64[ns]")
            month = np.where(
                (pk >= START) & (pk < MAY), 0, np.where((pk >= MAY) & (pk < END), 1, -1)
            )
            core = uber & (month >= 0) & ~np.isnat(dk) & (dk > pk)
            audit["completed_uber"] += int(core.sum())
            pu_f = pd.to_numeric(f["PULocationID"], errors="coerce").to_numpy(
                dtype=float
            )
            do_f = pd.to_numeric(f["DOLocationID"], errors="coerce").to_numpy(
                dtype=float
            )
            pu = np.where(np.isfinite(pu_f), pu_f, 0).astype(np.int16)
            do = np.where(np.isfinite(do_f), do_f, 0).astype(np.int16)
            pu_ok = (pu > 0) & (pu < 266) & np.isfinite(pu_f) & (pu_f == pu)
            do_ok = (do > 0) & (do < 266) & np.isfinite(do_f) & (do_f == do)
            pu_ok[pu_ok] &= zone_valid[pu[pu_ok]]
            do_ok[do_ok] &= zone_valid[do[do_ok]]
            od = core & pu_ok & do_ok
            audit["od_eligible_uber"] += int(od.sum())
            audit["excluded_missing_or_invalid_od"] += int(
                (core & ~(pu_ok & do_ok)).sum()
            )
            req = f["shared_request_flag"].fillna("Missing").to_numpy()
            match = f["shared_match_flag"].fillna("Missing").to_numpy()
            nn = (req == "N") & (match == "N")
            yy = (req == "Y") & (match == "Y")
            tt = pd.to_numeric(f["trip_time"], errors="coerce").to_numpy(dtype=float)
            time_ok = np.isfinite(tt) & (tt > 0) & (tt <= 86400)
            audit["reference_status_nn"] += int((od & nn).sum())
            audit["shared_status_yy"] += int((od & yy).sum())
            audit["excluded_nonpositive_or_over24h_trip_time_nn"] += int(
                (od & nn & ~time_ok).sum()
            )
            audit["excluded_nonpositive_or_over24h_trip_time_yy"] += int(
                (od & yy & ~time_ok).sum()
            )
            hour = pickup.dt.hour.fillna(0).to_numpy(dtype=np.int8)
            weekend = (
                pickup.dt.dayofweek.fillna(0).to_numpy(dtype=np.int8) >= 5
            ).astype(np.int8)
            for m in (0, 1):
                monthly_status[m, 0] += int((od & nn & (month == m) & time_ok).sum())
                monthly_status[m, 1] += int((od & yy & (month == m) & time_ok).sum())
            sel = od & nn & time_ok
            audit["reference_duration_valid"] += int(sel.sum())
            ref_keys.append(key_of(pu[sel], do[sel], hour[sel], weekend[sel]))
            ref_seconds.append(tt[sel].astype(np.float32))
            sel = od & yy & time_ok
            audit["shared_duration_valid"] += int(sel.sum())
            if sel.any():
                shared.append(
                    pd.DataFrame(
                        {
                            "month_index": month[sel].astype(np.int8),
                            "PULocationID": pu[sel],
                            "DOLocationID": do[sel],
                            "pickup_hour": hour[sel],
                            "weekend": weekend[sel],
                            "observed_minutes": tt[sel] / 60,
                            "cell_key": key_of(
                                pu[sel], do[sel], hour[sel], weekend[sel]
                            ),
                        }
                    )
                )
        if scanned != pf.metadata.num_rows:
            raise AssertionError((path, scanned, pf.metadata.num_rows))
        print(f"  scanned {scanned:,}", flush=True)
    task1 = json.loads((ROOT / "04_Results/Task_1/summary.json").read_text())
    retention = {
        (r["pickup_month"], r["analysis"]): r["retained_records"]
        for r in task1["retention_by_analysis"]
        if r["service"] == "fhvhv"
    }
    for m, month in enumerate(MONTHS):
        if monthly_status[m, 0] != retention[(month, "uber_duration_reference")]:
            raise AssertionError("N/N Task 1 mismatch")
        if monthly_status[m, 1] != retention[(month, "uber_duration_shared")]:
            raise AssertionError("Y/Y Task 1 mismatch")
    keys = np.concatenate(ref_keys)
    seconds = np.concatenate(ref_seconds)
    print(f"Sorting {len(keys):,} reference trips by cell and duration", flush=True)
    order = np.lexsort((seconds, keys))
    sorted_keys = keys[order]
    sorted_seconds = seconds[order]
    unique, starts, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )
    left = starts + (counts - 1) // 2
    right = starts + counts // 2
    medians = (
        sorted_seconds[left].astype(float) + sorted_seconds[right].astype(float)
    ) / 120
    supported = counts >= 30
    print(
        f"Reference cells: {len(unique):,}; supported: {supported.sum():,}", flush=True
    )
    ref_rows = []
    for i in range(len(unique)):
        key = int(unique[i])
        wk = key % 2
        key //= 2
        hr = key % 24
        key //= 24
        do = key % ZONE_BASE
        pu = key // ZONE_BASE
        ref_rows.append(
            {
                "PULocationID": pu,
                "DOLocationID": do,
                "pickup_hour": hr,
                "weekend": wk,
                "reference_trip_count": int(counts[i]),
                "reference_median_minutes": float(medians[i]),
                "supported_n_ge_30": bool(supported[i]),
            }
        )
    write_csv("reference_cells.csv", ref_rows)
    frame = pd.concat(shared, ignore_index=True)
    trip_keys = frame["cell_key"].to_numpy(dtype=np.int32)
    positions = np.searchsorted(unique, trip_keys)
    any_cell = positions < len(unique)
    any_cell[any_cell] &= unique[positions[any_cell]] == trip_keys[any_cell]
    ref = np.full(len(frame), np.nan)
    nref = np.zeros(len(frame), dtype=np.int32)
    nref[any_cell] = counts[positions[any_cell]]
    found = any_cell & (nref >= 30)
    ref[found] = medians[positions[found]]
    frame["reference_median_minutes"] = ref
    frame["reference_cell_n"] = nref
    frame["excess_minutes"] = frame["observed_minutes"] - ref
    frame["relative_excess_percent"] = 100 * frame["excess_minutes"] / ref
    frame["month"] = np.array(MONTHS)[frame["month_index"].to_numpy()]
    frame["within_zone"] = frame["PULocationID"] == frame["DOLocationID"]
    frame["day_type"] = np.where(frame["weekend"], "weekend", "weekday")
    if len(frame) != audit["shared_duration_valid"]:
        raise AssertionError("Shared eligible count mismatch")
    if int(found.sum()) + int((~found).sum()) != len(frame):
        raise AssertionError("Coverage mismatch")
    unsupported_reasons = {
        "no_reference_cell": int((~any_cell).sum()),
        "reference_cell_under_30": int((any_cell & ~found).sum()),
    }
    month_rows = grouped_summaries(frame, ["month"])
    hour_rows = grouped_summaries(frame, ["month", "pickup_hour"])
    zone_rows = grouped_summaries(frame, ["month", "PULocationID"], lookup)
    within_rows = grouped_summaries(frame, ["within_zone"])
    month_within = grouped_summaries(frame, ["month", "within_zone"])
    daytype = grouped_summaries(frame, ["month", "day_type"])
    write_csv("month_summary.csv", month_rows)
    write_csv("hourly_summary.csv", hour_rows)
    write_csv("zone_summary.csv", zone_rows)
    write_csv("within_between_summary.csv", month_within)
    write_csv("day_type_summary.csv", daytype)
    write_csv(
        "audit_reconciliation.csv",
        [{"rule_or_set": k, "records": v} for k, v in audit.items()],
    )
    write_csv(
        "unsupported_reference_reasons.csv",
        [{"reason": k, "shared_trips": v} for k, v in unsupported_reasons.items()],
    )
    # A compact trip-level supplement explicitly carries null E/R for unsupported cells.
    sample = pd.concat(
        [
            frame[frame["excess_minutes"].notna()].head(25),
            frame[frame["excess_minutes"].isna()].head(25),
        ]
    )
    write_csv(
        "shared_trip_examples.csv",
        sample[
            [
                "month",
                "PULocationID",
                "DOLocationID",
                "pickup_hour",
                "day_type",
                "observed_minutes",
                "reference_cell_n",
                "reference_median_minutes",
                "excess_minutes",
                "relative_excess_percent",
            ]
        ].to_dict("records"),
    )
    shapes = load_shapes()
    valid_means = [
        abs(r["mean_excess_minutes"])
        for r in zone_rows
        if r["reference_supported"] >= 10 and r["mean_excess_minutes"] is not None
    ]
    limit = float(np.ceil(np.quantile(valid_means, 0.98))) if valid_means else 1.0
    limit = max(limit, 1.0)
    plot_hour(
        hour_rows,
        "excess_median_minutes",
        "Median E (minutes)",
        "hourly_excess_minutes.png",
    )
    plot_hour(
        hour_rows,
        "relative_median_percent",
        "Median R (%)",
        "hourly_relative_percent.png",
    )
    for month in MONTHS:
        plot_zone_map(shapes, zone_rows, month, limit)
    plot_coverage_zone(shapes, zone_rows)
    other_plots(frame, month_rows, within_rows)
    hour_findings = []
    zone_findings = []
    for month in MONTHS:
        hourly = [r for r in hour_rows if r["month"] == month]
        low = min(hourly, key=lambda r: r["excess_median_minutes"])
        high = max(hourly, key=lambda r: r["excess_median_minutes"])
        hour_findings.append(
            {
                "month": month,
                "lowest_hour": int(low["pickup_hour"]),
                "lowest_median_minutes": low["excess_median_minutes"],
                "highest_hour": int(high["pickup_hour"]),
                "highest_median_minutes": high["excess_median_minutes"],
            }
        )
        eligible = [
            r
            for r in zone_rows
            if r["month"] == month and r["reference_supported"] >= 100
        ]
        low = min(eligible, key=lambda r: r["mean_excess_minutes"])
        high = max(eligible, key=lambda r: r["mean_excess_minutes"])
        zone_findings.append(
            {
                "month": month,
                "zones_with_100_supported": len(eligible),
                "lowest_zone": low["zone"],
                "lowest_mean_minutes": low["mean_excess_minutes"],
                "highest_zone": high["zone"],
                "highest_mean_minutes": high["mean_excess_minutes"],
            }
        )
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "Uber N/N reference pooled across April-May by ordered PU/DO, pickup hour, weekday/weekend. At least 30 reference trips per cell. Y/Y trip_time in (0,24h] compared individually; E and R null without supported cell. All reported aggregations weight trips equally.",
        "audit": audit,
        "reference_cell_count": int(len(unique)),
        "supported_reference_cells": int(supported.sum()),
        "reference_trip_count": int(len(keys)),
        "reference_trips_in_supported_cells": int(counts[supported].sum()),
        "unsupported_reference_reasons": unsupported_reasons,
        "month_summary": month_rows,
        "within_between": within_rows,
        "hour_findings": hour_findings,
        "zone_findings": zone_findings,
        "mean_zone_map_threshold": 10,
        "zone_map_color_limit_minutes": limit,
        "figure_count": 8,
        "task1_reconciliation": True,
        "limitations": [
            "Same zone pair can hide different exact endpoints.",
            "Traffic varies across dates inside a cell.",
            "Route and passenger selection differ.",
            "Sharing partners and overlap times are unverified.",
            "Reference coverage differs by place.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    write_report(summary)
    print("Task 4 complete: eight PNG figures and trip-weighted summaries", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--report-only"]:
        write_report(json.loads((OUT / "summary.json").read_text()))
        print("Regenerated Task 4 report.tex from saved summary.json")
    else:
        main()
