"""Task 2: batch aggregation, directed OD tables, and reproducible maps.

Run from the project root:
  python -m analysis.helpers.task_outputs run 2 -- python -m analysis.tasks.task2_analysis
The declared package requirements are in requirements.txt.
"""
from __future__ import annotations

import csv
import json
import struct
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib

from analysis.common import (
    HVFHV_SPENDING_COMPONENTS,
    PROJECT_ROOT,
    RAW_DIR,
    REFERENCE_DIR,
    STUDY_MONTHS,
    display_path,
    hvfhv_passenger_spending,
    load_zone_lookup,
    task_output_dir,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

ROOT = PROJECT_ROOT
RAW = RAW_DIR
REF = REFERENCE_DIR
OUT = task_output_dir(2)
TABLES, FIGURES = OUT / "tables", OUT / "figures"
MONTHS = list(STUDY_MONTHS)
SERVICES = ["Yellow", "Uber", "Lyft", "Other HVFHV"]
COMPONENTS = list(HVFHV_SPENDING_COMPONENTS)
SIZE = 266
START = np.datetime64("2026-04-01")
MAY = np.datetime64("2026-05-01")
END = np.datetime64("2026-06-01")


def csv_write(name, rows):
    path = TABLES / name
    if not rows:
        raise ValueError(f"No rows for {name}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {display_path(path)}: {len(rows)} rows", flush=True)


def zone_lookup():
    """Return official lookup rows; zones 264/265 remain valid for counts."""
    return load_zone_lookup()


def load_shapes():
    """Read polygon parts and LocationID from the supplied SHP/DBF without GIS extras."""
    base = REF / "taxi_zones/taxi_zones"
    with base.with_suffix(".dbf").open("rb") as f:
        header = f.read(32)
        n = struct.unpack("<I", header[4:8])[0]
        head_len = struct.unpack("<H", header[8:10])[0]
        rec_len = struct.unpack("<H", header[10:12])[0]
        fields = []
        for _ in range((head_len - 33) // 32):
            field = f.read(32)
            fields.append((field[:11].split(b"\0")[0].decode("ascii"), field[16]))
        f.seek(head_len)
        ids = []
        for _ in range(n):
            rec = f.read(rec_len)
            offset = 1
            values = {}
            for name, width in fields:
                values[name] = rec[offset : offset + width].decode("latin1").strip()
                offset += width
            ids.append(int(values["LocationID"]))
    shapes = {}
    with base.with_suffix(".shp").open("rb") as f:
        f.seek(100)
        for zone_id in ids:
            rec_head = f.read(8)
            if len(rec_head) != 8:
                raise ValueError("Shapefile record count does not match DBF")
            length = struct.unpack(">I", rec_head[4:])[0] * 2
            data = f.read(length)
            shape_type = struct.unpack("<I", data[:4])[0]
            if shape_type != 5:
                raise ValueError(f"Expected Polygon, got {shape_type}")
            n_parts, n_points = struct.unpack("<II", data[36:44])
            starts = list(
                struct.unpack("<" + "I" * n_parts, data[44 : 44 + 4 * n_parts])
            ) + [n_points]
            point_data = data[44 + 4 * n_parts :]
            points = np.frombuffer(point_data, dtype="<f8", count=2 * n_points).reshape(
                -1, 2
            )
            shapes[zone_id] = [
                points[starts[i] : starts[i + 1]] for i in range(n_parts)
            ]
    return shapes


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    lookup = zone_lookup()
    valid_ids = np.zeros(SIZE, dtype=bool)
    valid_ids[list(lookup)] = True
    # Ordered key is (month, service, pickup ID, dropoff ID); reverse flows differ.
    od_count = np.zeros((2, 4, SIZE, SIZE), dtype=np.int64)
    spend_count = np.zeros_like(od_count)
    spend_total = np.zeros((2, 4, SIZE, SIZE), dtype=np.float64)
    pickup_count = np.zeros((2, 2, SIZE), dtype=np.int64)
    dropoff_count = np.zeros_like(pickup_count)
    stats = defaultdict(int)
    component_missing = defaultdict(int)
    files = [
        ("yellow", RAW / f"yellow_tripdata_2026-{m}.parquet") for m in ("04", "05")
    ]
    files += [("fhvhv", RAW / f"fhvhv_tripdata_2026-{m}.parquet") for m in ("04", "05")]
    for source, path in files:
        cols = (
            [
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "total_amount",
            ]
            if source == "yellow"
            else [
                "pickup_datetime",
                "dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "hvfhs_license_num",
            ]
            + COMPONENTS
        )
        pf = pq.ParquetFile(path)
        scanned = 0
        print(f"Scanning {path.name} ({pf.metadata.num_rows:,} rows)", flush=True)
        for batch in pf.iter_batches(batch_size=250_000, columns=cols):
            frame = batch.to_pandas()
            n = len(frame)
            scanned += n
            pu = pd.to_numeric(frame["PULocationID"], errors="coerce").to_numpy(
                dtype=float
            )
            do = pd.to_numeric(frame["DOLocationID"], errors="coerce").to_numpy(
                dtype=float
            )
            pu_int = np.where(np.isfinite(pu), pu, 0).astype(np.int32)
            do_int = np.where(np.isfinite(do), do, 0).astype(np.int32)
            pu_ok = (pu_int > 0) & (pu_int < SIZE) & np.isfinite(pu) & (pu == pu_int)
            do_ok = (do_int > 0) & (do_int < SIZE) & np.isfinite(do) & (do == do_int)
            pu_ok[pu_ok] &= valid_ids[pu_int[pu_ok]]
            do_ok[do_ok] &= valid_ids[do_int[do_ok]]
            pu_int[~pu_ok] = 0
            do_int[~do_ok] = 0
            pk = "tpep_pickup_datetime" if source == "yellow" else "pickup_datetime"
            dk = "tpep_dropoff_datetime" if source == "yellow" else "dropoff_datetime"
            pickup = pd.to_datetime(frame[pk], errors="coerce").to_numpy(
                dtype="datetime64[ns]"
            )
            dropoff = pd.to_datetime(frame[dk], errors="coerce").to_numpy(
                dtype="datetime64[ns]"
            )
            month = np.where(
                (pickup >= START) & (pickup < MAY),
                0,
                np.where((pickup >= MAY) & (pickup < END), 1, -1),
            )
            core = (month >= 0) & ~np.isnat(dropoff) & (dropoff > pickup)
            if source == "yellow":
                service = np.zeros(n, dtype=np.int8)
                values = pd.to_numeric(frame["total_amount"], errors="coerce").to_numpy(
                    dtype=float
                )
                complete = np.isfinite(values)
            else:
                company = frame["hvfhs_license_num"].fillna("Missing").to_numpy()
                service = np.where(
                    company == "HV0003", 1, np.where(company == "HV0005", 2, 3)
                ).astype(np.int8)
                values, complete, parts = hvfhv_passenger_spending(frame)
            for m in (0, 1):
                in_month = core & (month == m)
                for s in (0,) if source == "yellow" else (1, 2, 3):
                    base = in_month & (service == s)
                    if not base.any():
                        continue
                    key = (MONTHS[m], SERVICES[s])
                    k = int(base.sum())
                    stats[key + ("completed",)] += k
                    stats[key + ("pu_invalid",)] += int((base & ~pu_ok).sum())
                    stats[key + ("do_invalid",)] += int((base & ~do_ok).sum())
                    stats[key + ("either_invalid",)] += int(
                        (base & ~(pu_ok & do_ok)).sum()
                    )
                    if s == 0:
                        hot_service = 0
                    else:
                        hot_service = 1
                    pick_sel = base & pu_ok
                    drop_sel = base & do_ok
                    pickup_count[m, hot_service] += np.bincount(
                        pu_int[pick_sel], minlength=SIZE
                    )
                    dropoff_count[m, hot_service] += np.bincount(
                        do_int[drop_sel], minlength=SIZE
                    )
                    od = base & pu_ok & do_ok
                    stats[key + ("od_eligible",)] += int(od.sum())
                    if not od.any():
                        continue
                    flat = pu_int[od] * SIZE + do_int[od]
                    od_count[m, s] += np.bincount(flat, minlength=SIZE * SIZE).reshape(
                        SIZE, SIZE
                    )
                    bad_component = od & ~complete
                    for component in (
                        ["total_amount"] if source == "yellow" else COMPONENTS
                    ):
                        if source == "yellow":
                            count = int((od & ~np.isfinite(values)).sum())
                        else:
                            idx = COMPONENTS.index(component)
                            count = int((od & ~np.isfinite(parts[:, idx])).sum())
                        component_missing[key + (component,)] += count
                    negative = od & complete & (values < 0)
                    valid_spending = od & complete & (values >= 0)
                    stats[key + ("spending_missing",)] += int(bad_component.sum())
                    stats[key + ("spending_negative",)] += int(negative.sum())
                    stats[key + ("spending_eligible",)] += int(valid_spending.sum())
                    flat_spend = pu_int[valid_spending] * SIZE + do_int[valid_spending]
                    spend_count[m, s] += np.bincount(
                        flat_spend, minlength=SIZE * SIZE
                    ).reshape(SIZE, SIZE)
                    spend_total[m, s] += np.bincount(
                        flat_spend,
                        weights=values[valid_spending],
                        minlength=SIZE * SIZE,
                    ).reshape(SIZE, SIZE)
        if scanned != pf.metadata.num_rows:
            raise AssertionError((path.name, scanned, pf.metadata.num_rows))
        print(f"  Scanned {scanned:,}", flush=True)
    # Exact reconciliation against Task 1 masks and trip categories.
    task1 = json.loads((ROOT / "04_Results/Task_1/summary.json").read_text())
    for m in MONTHS:
        for s in SERVICES:
            key = (m, s)
            if (
                stats[key + ("completed",)]
                != stats[key + ("od_eligible",)] + stats[key + ("either_invalid",)]
            ):
                raise AssertionError(f"OD eligibility mismatch: {key}")
    for m_idx, month in enumerate(MONTHS):
        for s_idx, service in enumerate(SERVICES):
            key = (month, service)
            if od_count[m_idx, s_idx].sum() != stats[key + ("od_eligible",)]:
                raise AssertionError(f"Grouped OD volume mismatch: {key}")
            if spend_count[m_idx, s_idx].sum() != stats[key + ("spending_eligible",)]:
                raise AssertionError(f"Grouped OD spending mismatch: {key}")
            if (
                stats[key + ("spending_eligible",)]
                + stats[key + ("spending_missing",)]
                + stats[key + ("spending_negative",)]
                != stats[key + ("od_eligible",)]
            ):
                raise AssertionError(f"Spending eligibility mismatch: {key}")
        if pickup_count[m_idx, 1].sum() != sum(
            stats[(month, s, "completed")] - stats[(month, s, "pu_invalid")]
            for s in SERVICES[1:]
        ):
            raise AssertionError("HVFHV pickup total mismatch")
    retention = {
        (r["pickup_month"], r["service"], r["analysis"]): r["retained_records"]
        for r in task1["retention_by_analysis"]
    }
    for m_idx, month in enumerate(MONTHS):
        for source, indices in (("yellow", [0]), ("fhvhv", [1, 2, 3])):
            for analysis, field in (
                ("completed_trip_count", "completed"),
                ("directed_od_volume", "od_eligible"),
                ("directed_od_spending", "spending_eligible"),
            ):
                observed = sum(stats[(month, SERVICES[i], field)] for i in indices)
                expected = retention[(month, source, analysis)]
                if observed != expected:
                    raise AssertionError(
                        f"Task 1 mismatch {month} {source} {analysis}: {observed} != {expected}"
                    )
    print("Reconciled all month/service OD and spending totals with Task 1", flush=True)
    zone_rows, volume_rows, spending_rows, reconciliation = [], [], [], []
    tops = {}
    for m_idx, month in enumerate(MONTHS):
        for h_idx, service in enumerate(("Yellow", "HVFHV")):
            for measure, matrix in (
                ("pickup", pickup_count),
                ("dropoff", dropoff_count),
            ):
                for zone_id in sorted(lookup):
                    zone_rows.append(
                        {
                            "month": month,
                            "service": service,
                            "measure": measure,
                            "LocationID": zone_id,
                            "borough": lookup[zone_id]["Borough"],
                            "zone": lookup[zone_id]["Zone"],
                            "trip_count": int(matrix[m_idx, h_idx, zone_id]),
                            "drawable": zone_id not in (264, 265),
                        }
                    )
        for s_idx, service in enumerate(SERVICES):
            key = (month, service)
            rec = {"month": month, "service": service}
            rec.update(
                {
                    field: stats[key + (field,)]
                    for field in (
                        "completed",
                        "pu_invalid",
                        "do_invalid",
                        "either_invalid",
                        "od_eligible",
                        "spending_missing",
                        "spending_negative",
                        "spending_eligible",
                    )
                }
            )
            rec["od_grouped_count"] = int(od_count[m_idx, s_idx].sum())
            rec["spending_grouped_count"] = int(spend_count[m_idx, s_idx].sum())
            rec["within_zone_trips"] = int(np.trace(od_count[m_idx, s_idx]))
            reconciliation.append(rec)
            if s_idx == 3:
                continue
            vol = od_count[m_idx, s_idx]
            spend = spend_total[m_idx, s_idx]
            spend_n = spend_count[m_idx, s_idx]
            # Stable rank: descending measure, then origin and destination ID.
            pairs = [(a, b) for a in range(SIZE) for b in range(SIZE) if vol[a, b] > 0]
            vol_top = sorted(pairs, key=lambda p: (-vol[p], p[0], p[1]))[:10]
            spend_top = sorted(pairs, key=lambda p: (-spend[p], p[0], p[1]))[:10]
            tops[(month, service)] = (vol_top, spend_top)
            for rank, (a, b) in enumerate(vol_top, 1):
                volume_rows.append(
                    {
                        "month": month,
                        "service": service,
                        "rank": rank,
                        "PULocationID": a,
                        "origin_borough": lookup[a]["Borough"],
                        "origin_zone": lookup[a]["Zone"],
                        "DOLocationID": b,
                        "destination_borough": lookup[b]["Borough"],
                        "destination_zone": lookup[b]["Zone"],
                        "trip_count": int(vol[a, b]),
                        "share_of_service_month": float(vol[a, b] / rec["completed"]),
                        "service_month_denominator": rec["completed"],
                        "within_zone": a == b,
                        "also_spending_top10": (a, b) in spend_top,
                    }
                )
            for rank, (a, b) in enumerate(spend_top, 1):
                spending_rows.append(
                    {
                        "month": month,
                        "service": service,
                        "rank": rank,
                        "PULocationID": a,
                        "origin_borough": lookup[a]["Borough"],
                        "origin_zone": lookup[a]["Zone"],
                        "DOLocationID": b,
                        "destination_borough": lookup[b]["Borough"],
                        "destination_zone": lookup[b]["Zone"],
                        "valid_spending_trip_count": int(spend_n[a, b]),
                        "total_spending_usd": round(float(spend[a, b]), 2),
                        "mean_payment_usd": round(
                            float(spend[a, b] / spend_n[a, b]), 2
                        ),
                        "service_month_valid_spending_denominator": rec[
                            "spending_eligible"
                        ],
                        "also_volume_top10": (a, b) in vol_top,
                        "within_zone": a == b,
                    }
                )
    csv_write("zone_demand.csv", zone_rows)
    csv_write("top10_volume_od.csv", volume_rows)
    csv_write("top10_spending_od.csv", spending_rows)
    csv_write("reconciliation.csv", reconciliation)
    csv_write(
        "missing_spending_components.csv",
        [
            {
                "month": m,
                "service": s,
                "component": c,
                "missing_or_nonfinite_records": component_missing[(m, s, c)],
            }
            for m in MONTHS
            for s in SERVICES
            for c in (["total_amount"] if s == "Yellow" else COMPONENTS)
        ],
    )
    shapes = load_shapes()
    print(f"Loaded {len(shapes)} drawable taxi-zone polygons", flush=True)
    draw_hotspots(shapes, lookup, pickup_count, dropoff_count)
    draw_flows(shapes, lookup, tops, od_count, spend_total)
    draw_comparison(volume_rows, spending_rows)
    summary = build_summary(
        reconciliation, zone_rows, volume_rows, spending_rows, component_missing
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    write_report(summary, volume_rows, spending_rows)
    print("Task 2 outputs complete", flush=True)


def map_base(ax, shapes, values=None, norm=None, cmap="YlOrRd"):
    colors = plt.get_cmap(cmap)
    for zone_id, parts in shapes.items():
        value = None if values is None else values[zone_id]
        face = "#f2f4f6" if value is None or value <= 0 else colors(norm(value))
        for part in parts:
            ax.add_patch(
                Polygon(
                    part,
                    closed=True,
                    facecolor=face,
                    edgecolor="#bbc5cb",
                    linewidth=0.16,
                    zorder=1,
                )
            )
    ax.set_xlim(900000, 1070000)
    ax.set_ylim(110000, 280000)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_hotspots(shapes, lookup, pickup, dropoff):
    # One shared scale across all eight count maps makes colors directly comparable.
    maximum = max(int(pickup.max()), int(dropoff.max()))
    norm = LogNorm(vmin=1, vmax=maximum)
    for m_idx, month in enumerate(MONTHS):
        for h_idx, service in enumerate(("Yellow", "HVFHV")):
            for measure, matrix in (("pickup", pickup), ("dropoff", dropoff)):
                values = matrix[m_idx, h_idx]
                fig, ax = plt.subplots(figsize=(7.3, 7.2))
                map_base(ax, shapes, values, norm)
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap="YlOrRd"),
                    ax=ax,
                    shrink=0.70,
                    label="Completed trips per zone (shared log scale)",
                )
                ax.set_title(
                    f"{month} {service} {measure} zone demand", loc="left", fontsize=13
                )
                top = np.argsort(values)[::-1][:3]
                caption = "Top zones: " + "; ".join(
                    f"{i} {lookup[i]['Zone']} ({int(values[i]):,})" for i in top
                )
                fig.text(0.04, 0.04, caption, fontsize=8)
                fig.text(
                    0.04,
                    0.018,
                    "Zones 264/265 have no boundary; counts remain in CSV. Same legend across all 8 maps.",
                    fontsize=7,
                )
                fig.savefig(
                    FIGURES / f"hotspot_{month}_{service.lower()}_{measure}.png",
                    dpi=170,
                    bbox_inches="tight",
                )
                plt.close(fig)


def draw_flows(shapes, lookup, tops, counts, spending):
    centers = {k: np.concatenate(v).mean(axis=0) for k, v in shapes.items()}
    # Valid lookup IDs 264/265 have no polygons. Show them as explicitly
    # labelled off-map nodes so every top-ten OD remains visible.
    centers.update(
        {264: np.array([1085000.0, 150000.0]), 265: np.array([1085000.0, 128000.0])}
    )
    for m_idx, month in enumerate(MONTHS):
        for s_idx, service in enumerate(SERVICES[:3]):
            volume_top, spending_top = tops[(month, service)]
            for kind, pairs, matrix, unit in (
                ("volume", volume_top, counts, "trips"),
                ("spending", spending_top, spending, "USD"),
            ):
                fig, ax = plt.subplots(figsize=(9, 7.2))
                map_base(ax, shapes)
                ax.set_xlim(900000, 1110000)
                for special in (264, 265):
                    if any(special in pair for pair in pairs):
                        x, y = centers[special]
                        ax.plot(x, y, "o", markersize=6, color="#45535c", zorder=5)
                        ax.text(
                            x + 3500,
                            y,
                            f"{special}: {lookup[special]['Zone']}",
                            fontsize=7,
                            va="center",
                            color="#303a40",
                        )
                vmax = max(float(matrix[m_idx, s_idx, p[0], p[1]]) for p in pairs)
                for rank, (a, b) in enumerate(reversed(pairs), 1):
                    value = float(matrix[m_idx, s_idx, a, b])
                    width = 1.2 + 6.5 * value / vmax
                    color = plt.get_cmap("plasma")((11 - rank) / 11)
                    x1, y1 = centers[a]
                    x2, y2 = centers[b]
                    if a == b:
                        ax.add_patch(
                            Circle(
                                (x1, y1),
                                radius=2100 + 2800 * value / vmax,
                                fill=False,
                                edgecolor=color,
                                linewidth=width,
                                zorder=3,
                            )
                        )
                    else:
                        ax.add_patch(
                            FancyArrowPatch(
                                (x1, y1),
                                (x2, y2),
                                arrowstyle="-|>",
                                mutation_scale=11,
                                linewidth=width,
                                color=color,
                                alpha=0.8,
                                zorder=3,
                                shrinkA=2,
                                shrinkB=2,
                            )
                        )
                ax.set_title(
                    f"{month} {service}: top 10 directed OD flows by {kind}",
                    loc="left",
                    fontsize=12,
                )
                # Ranked sidebar makes all flows legible even where arrows overlap.
                lines = []
                for rank, (a, b) in enumerate(pairs, 1):
                    value = float(matrix[m_idx, s_idx, a, b])
                    shown = (
                        f"{value:,.0f}" if kind == "volume" else f"${value/1e6:,.2f}m"
                    )
                    lines.append(f"{rank:2d}. {a} → {b}   {shown}")
                fig.text(
                    0.76,
                    0.36,
                    "\n".join(lines),
                    family="monospace",
                    fontsize=8,
                    va="center",
                    bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
                )
                fig.text(
                    0.04,
                    0.025,
                    "Arrows show direction; rings are within-zone trips. 264/265 are labelled off-map nodes. Exact values: CSV.",
                    fontsize=8,
                )
                fig.savefig(
                    FIGURES / f"od_{kind}_{month}_{service.lower()}.png",
                    dpi=170,
                    bbox_inches="tight",
                )
                plt.close(fig)


def draw_comparison(volume_rows, spending_rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    colors = {"Yellow": "#e7a332", "Uber": "#446a9b", "Lyft": "#bd5684"}
    for i, month in enumerate(MONTHS):
        ax = axes[i]
        for j, service in enumerate(SERVICES[:3]):
            overlap = sum(
                r["also_spending_top10"]
                for r in volume_rows
                if r["month"] == month and r["service"] == service
            )
            ax.bar(j, overlap, color=colors[service], width=0.65)
            ax.text(j, overlap + 0.2, f"{overlap}/10", ha="center", fontsize=10)
        ax.set_xticks(range(3), SERVICES[:3])
        ax.set_ylim(0, 11)
        ax.set_title(month)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("OD pairs common to volume and spending top 10")
    fig.suptitle("How often high demand is also high total spending")
    fig.tight_layout()
    fig.savefig(FIGURES / "volume_spending_overlap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_summary(reconciliation, zone_rows, volume_rows, spending_rows, missing):
    leading = []
    for month in MONTHS:
        for service in ("Yellow", "HVFHV"):
            for measure in ("pickup", "dropoff"):
                rows = sorted(
                    (
                        r
                        for r in zone_rows
                        if r["month"] == month
                        and r["service"] == service
                        and r["measure"] == measure
                    ),
                    key=lambda r: -r["trip_count"],
                )
                leading.append(
                    {
                        "month": month,
                        "service": service,
                        "measure": measure,
                        "top3": rows[:3],
                        "unmapped_264_265_trips": sum(
                            r["trip_count"]
                            for r in rows
                            if r["LocationID"] in (264, 265)
                        ),
                    }
                )
    comparisons = []
    for month in MONTHS:
        for service in SERVICES[:3]:
            v = [
                r
                for r in volume_rows
                if r["month"] == month and r["service"] == service
            ]
            s = [
                r
                for r in spending_rows
                if r["month"] == month and r["service"] == service
            ]
            comparisons.append(
                {
                    "month": month,
                    "service": service,
                    "top10_overlap": sum(r["also_spending_top10"] for r in v),
                    "volume_top1": v[0],
                    "spending_top1": s[0],
                    "volume_top10_total_share": sum(
                        r["share_of_service_month"] for r in v
                    ),
                }
            )
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "method": "Pickup timestamp assigns month across all four source files. Completed trip requires valid pickup/dropoff, April/May pickup, and dropoff after pickup. Hotspots require a valid endpoint; directed OD requires both official lookup IDs and retains A→A. Spending requires finite, nonnegative Yellow total_amount or all eight finite HVFHV components with a nonnegative sum; driver_pay excluded. Recorded completed trips are demand, not all requests.",
        "reconciliation": reconciliation,
        "leading_zones": leading,
        "volume_spending_comparison": comparisons,
        "missing_component_counts": [
            {
                "month": m,
                "service": s,
                "component": c,
                "missing_or_nonfinite": missing[(m, s, c)],
            }
            for m in MONTHS
            for s in SERVICES
            for c in (["total_amount"] if s == "Yellow" else COMPONENTS)
        ],
        "figure_count": 21,
        "limitations": [
            "Yellow total_amount excludes unrecorded cash tips.",
            "Zone IDs 264/265 are valid in the official lookup but have no shapefile polygon; their counts appear in CSV.",
            "Flow maps use zone centroids or labelled off-map nodes, not actual road routes or exact trip endpoints.",
        ],
    }


def write_report(summary, volume_rows, spending_rows):
    lines = [
        r"\subsection{Task 2: Spatial demand and passenger spending}",
        r"Month is determined from each pickup timestamp across both source files. We count completed recorded trips with positive timestamp duration; zone maps require the relevant valid TLC zone, while directed OD analysis requires both. Reverse OD pairs remain distinct and within-zone trips are retained. The sum of all ordered OD groups equals the eligible-trip denominator in Table~\ref{tab:t2rec}.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Task 2 denominator reconciliation. Spending counts require valid payment.}\label{tab:t2rec}",
        r"\begin{tabular}{llrrrr}\hline Month & Service & Completed & OD eligible & Spend eligible & Invalid OD \\ \hline",
    ]
    for r in summary["reconciliation"]:
        if r["service"] == "Other HVFHV" and r["completed"] == 0:
            continue
        lines.append(
            f"{r['month']} & {r['service']} & {r['completed']:,} & {r['od_eligible']:,} & {r['spending_eligible']:,} & {r['either_invalid']:,} "
            + r"\\"
        )
    lines += [
        r"\hline\end{tabular}\end{table}",
        r"The eight hotspot maps use one logarithmic trip-count legend. Zone IDs 264 and 265 are valid lookup categories without drawable boundaries; their counts are preserved in the CSV. The top pickup and dropoff zones by service and month are:",
        r"\begin{itemize}",
    ]
    for x in summary["leading_zones"]:
        names = ", ".join(f"{z['zone']} ({z['trip_count']:,})" for z in x["top3"])
        lines.append(f"\\item {x['month']} {x['service']} {x['measure']}: {names}.")
    lines.append(r"\end{itemize}")
    lines.append(
        r"Figure~\ref{fig:t2hotyellow} shows the April pickup maps; the remaining six comparable maps are in the figure supplement. The twelve OD flow maps show the ten largest directed flows for each service and month, separately by trip count and passenger spending. Arrows indicate OD direction and rings show within-zone flows. The ranked CSV supplements give zone names, counts, shares, service-month denominators, valid-payment counts, totals, and means."
    )
    lines += [
        r"\begin{figure}[htbp]\centering\includegraphics[width=.46\textwidth]{04_Results/Task_2/figures/hotspot_2026-04_yellow_pickup.png}\hfill\includegraphics[width=.46\textwidth]{04_Results/Task_2/figures/hotspot_2026-04_hvfhv_pickup.png}",
        r"\caption{April pickup trip counts, Yellow and HVFHV, with a common logarithmic legend.}\label{fig:t2hotyellow}\end{figure}",
        r"\begin{figure}[htbp]\centering\includegraphics[width=.48\textwidth]{04_Results/Task_2/figures/od_volume_2026-04_yellow.png}\hfill\includegraphics[width=.48\textwidth]{04_Results/Task_2/figures/od_spending_2026-04_yellow.png}",
        r"\caption{April Yellow directed flows ranked by completed trips and total recorded passenger spending.}\label{fig:t2flows}\end{figure}",
    ]
    lines.append(
        r"\begin{table}[htbp]\centering\small\caption{Top-ten OD overlap between trip volume and total spending.}\label{tab:t2overlap}\begin{tabular}{llr}\hline Month & Service & Common OD pairs / 10 \\ \hline"
    )
    for x in summary["volume_spending_comparison"]:
        lines.append(f"{x['month']} & {x['service']} & {x['top10_overlap']} " + r"\\")
    lines.append(r"\hline\end{tabular}\end{table}")
    lines.append(
        r"\begin{table}[htbp]\centering\small\caption{Leading directed OD in each service-month: volume versus passenger spending. The full top-ten lists are in the CSV supplement.}\label{tab:t2leaders}\begin{tabular}{llrrr}\hline Month & Service & Volume leader (trips) & Spending leader (USD) & Mean USD \\ \hline"
    )
    for x in summary["volume_spending_comparison"]:
        v, p = x["volume_top1"], x["spending_top1"]
        lines.append(
            f"{x['month']} & {x['service']} & {v['PULocationID']} $\\to$ {v['DOLocationID']} ({v['trip_count']:,}) & {p['PULocationID']} $\\to$ {p['DOLocationID']} ({p['total_spending_usd']:,.0f}) & {p['mean_payment_usd']:.2f} "
            + r"\\"
        )
    lines.append(r"\hline\end{tabular}\end{table}")
    lines.append(
        r"Yellow pickup and dropoff peaks remain concentrated in Upper East Side and Midtown zones in both months. HVFHV pickup peaks are LaGuardia and JFK airports, whereas its largest dropoff category is Outside of NYC (zone 265): 930,707 trips in April and 1,018,509 in May. This valid lookup category has no polygon, so the choropleth cannot color it; the count is explicit in the zone CSV and caption. Yellow's highest-volume OD is Upper East Side South $\to$ Upper East Side North (237$\to$236), while Uber and Lyft are led by an intra-zone trip (76$\to$76). Spending leaders instead connect airports to Outside of NYC, illustrating why payment ranking differs from trip ranking. Across both months, only three of Yellow's ten volume leaders and two of each platform's ten volume leaders also appear in the corresponding spending top ten. All three service-month completed-trip totals rise from April to May except Uber, which declines slightly. These comparisons describe observed completed trips and recorded payments, not total travel requests or the exact routes taken."
    )
    lines.append(
        r"Yellow spending uses only \texttt{total\_amount}; HVFHV spending sums base fare, tolls, BCF, sales tax, congestion surcharge, airport fee, tips, and CBD congestion fee, excluding driver pay. A missing/nonfinite component makes the trip's spending undefined; we never replace it with zero. Negative total spending is excluded as a likely refund or correction. Values above USD 1,000 are audited but retained because they can represent long trips. Yellow cash tips may not be recorded. The flow maps show zone-centroid associations or labelled off-map nodes, not actual routes; demand represents completed trip records rather than all requests."
    )
    (OUT / "report.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["--report-only"]:
        summary = json.loads((OUT / "summary.json").read_text())
        write_report(summary, [], [])
        print("Regenerated Task 2 report.tex from saved summary.json")
    else:
        main()
