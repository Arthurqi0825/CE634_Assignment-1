"""Render report-ready figures from processed CSV tables only.

This module deliberately avoids the raw TLC Parquet files. It reads the
packaged task CSV outputs and the Taxi Zone reference files, then writes a
separate figure tree under ``04_Results/report_figures`` with PNG and PDF
versions for LaTeX use.

Run from the project root:

    python3 -m analysis.render_report_figures
"""

from __future__ import annotations

import csv
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "ce634_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "ce634_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from PIL import Image, ImageDraw, ImageFont, ImageStat

from analysis.common import PROJECT_ROOT, REFERENCE_DIR, RESULTS_DIR, load_zone_lookup

OUT = RESULTS_DIR / "report_figures"
MAIN = OUT / "main"
APPENDIX = OUT / "appendix"
PDF_DIR = OUT / "pdf"
PNG_DIR = OUT / "png"
INSTRUCTION = PROJECT_ROOT / "instruction"

BLUE = "#2f6690"
TEAL = "#3f7f7b"
GOLD = "#b8872d"
CLAY = "#b65d3b"
PLUM = "#7a5c98"
GREEN = "#5f8f4e"
INK = "#263238"
MUTED = "#687984"
GRID = "#d8e0e5"
LAND = "#f4f1ea"
BOUNDARY = "#c6c2b8"
MISSING = "#eeeeee"

SERVICE_COLORS = {
    "Yellow": GOLD,
    "HVFHV": TEAL,
    "Uber": BLUE,
    "Lyft": PLUM,
    "Juno": "#a7a7a7",
    "Via": "#a7a7a7",
}


@dataclass
class ManifestRow:
    figure_id: str
    task: str
    title: str
    png_path: Path
    pdf_path: Path
    input_files: list[Path]
    main_or_appendix: str
    key_message: str
    caveat: str
    purpose: str
    key_variables: str
    caption: str
    script_function: str


MANIFEST: list[ManifestRow] = []


def load_shapes() -> dict[int, list[np.ndarray]]:
    """Read Taxi Zone polygon parts directly from the packaged SHP/DBF.

    This local reader keeps the figure-only pipeline independent of pyarrow and
    the raw-data Task 2 module.
    """
    base = REFERENCE_DIR / "taxi_zones" / "taxi_zones"
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
    shapes: dict[int, list[np.ndarray]] = {}
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
            starts = list(struct.unpack("<" + "I" * n_parts, data[44 : 44 + 4 * n_parts])) + [n_points]
            point_data = data[44 + 4 * n_parts :]
            points = np.frombuffer(point_data, dtype="<f8", count=2 * n_points).reshape(-1, 2)
            shapes[zone_id] = [points[starts[i] : starts[i + 1]] for i in range(n_parts)]
    return shapes


def table_path(task: int, name: str) -> Path:
    path = RESULTS_DIR / f"Task_{task}" / "tables" / name
    if not path.is_file():
        raise FileNotFoundError(f"Required processed input is missing: {path}")
    return path


def read_csv(task: int, name: str) -> list[dict[str, str]]:
    path = table_path(task, name)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ensure_dirs() -> None:
    for path in (MAIN, APPENDIX, PDF_DIR, PNG_DIR, INSTRUCTION):
        path.mkdir(parents=True, exist_ok=True)


def choose_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"):
        if name in available:
            return name
    return "DejaVu Serif"


def setup_style() -> None:
    font = choose_font()
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#5c6970",
            "axes.linewidth": 0.8,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(
    fig: plt.Figure,
    filename: str,
    *,
    figure_id: str,
    task: str,
    title: str,
    input_files: list[Path],
    main: bool,
    key_message: str,
    caveat: str,
    purpose: str,
    key_variables: str,
    caption: str,
    script_function: str,
) -> None:
    target_dir = MAIN if main else APPENDIX
    png = target_dir / f"{filename}.png"
    pdf = target_dir / f"{filename}.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    mirror_png = PNG_DIR / png.name
    mirror_pdf = PDF_DIR / pdf.name
    mirror_png.write_bytes(png.read_bytes())
    mirror_pdf.write_bytes(pdf.read_bytes())
    MANIFEST.append(
        ManifestRow(
            figure_id=figure_id,
            task=task,
            title=title,
            png_path=png,
            pdf_path=pdf,
            input_files=input_files,
            main_or_appendix="main" if main else "appendix",
            key_message=key_message,
            caveat=caveat,
            purpose=purpose,
            key_variables=key_variables,
            caption=caption,
            script_function=script_function,
        )
    )


def fmt_count(value: float) -> str:
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:,.0f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def zone_label(row: dict[str, str], prefix: str) -> str:
    if prefix == "PU":
        zone = row["origin_zone"]
        zone_id = row["PULocationID"]
    else:
        zone = row["destination_zone"]
        zone_id = row["DOLocationID"]
    return f"{zone} ({zone_id})"


def flow_label(row: dict[str, str]) -> str:
    return f"{zone_label(row, 'PU')} -> {zone_label(row, 'DO')}"


def shape_bounds(shapes: dict[int, list[np.ndarray]]) -> tuple[float, float, float, float]:
    points = np.vstack([part for parts in shapes.values() for part in parts])
    return (
        float(points[:, 0].min()),
        float(points[:, 0].max()),
        float(points[:, 1].min()),
        float(points[:, 1].max()),
    )


def centroids(shapes: dict[int, list[np.ndarray]]) -> dict[int, tuple[float, float]]:
    centers: dict[int, tuple[float, float]] = {}
    for zone_id, parts in shapes.items():
        points = np.vstack(parts)
        centers[zone_id] = (float(points[:, 0].mean()), float(points[:, 1].mean()))
    return centers


def draw_base_map(ax: plt.Axes, shapes: dict[int, list[np.ndarray]]) -> None:
    for parts in shapes.values():
        for part in parts:
            ax.add_patch(
                Polygon(
                    part,
                    closed=True,
                    facecolor=LAND,
                    edgecolor=BOUNDARY,
                    linewidth=0.25,
                    zorder=0,
                )
            )
    xmin, xmax, ymin, ymax = shape_bounds(shapes)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_choropleth(
    ax: plt.Axes,
    shapes: dict[int, list[np.ndarray]],
    values: dict[int, float],
    *,
    cmap: str,
    norm,
) -> None:
    for zone_id, parts in shapes.items():
        value = values.get(zone_id)
        face = MISSING if value is None or not np.isfinite(value) or value <= 0 else plt.get_cmap(cmap)(norm(value))
        for part in parts:
            ax.add_patch(
                Polygon(
                    part,
                    closed=True,
                    facecolor=face,
                    edgecolor="white",
                    linewidth=0.16,
                    zorder=1,
                )
            )
    xmin, xmax, ymin, ymax = shape_bounds(shapes)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_zone_metric(
    ax: plt.Axes,
    shapes: dict[int, list[np.ndarray]],
    values: dict[int, float | None],
    *,
    cmap: str,
    norm,
    insufficient: set[int] | None = None,
) -> None:
    insufficient = insufficient or set()
    for zone_id, parts in shapes.items():
        value = values.get(zone_id)
        if zone_id in insufficient:
            face = "#cfcfcf"
        elif value is None or not np.isfinite(value):
            face = MISSING
        else:
            face = plt.get_cmap(cmap)(norm(value))
        for part in parts:
            ax.add_patch(
                Polygon(part, closed=True, facecolor=face, edgecolor="white", linewidth=0.16)
            )
    xmin, xmax, ymin, ymax = shape_bounds(shapes)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


def figure_task1() -> None:
    counts = read_csv(1, "task1_2_counts.csv")
    input_files = [table_path(1, "task1_2_counts.csv")]
    service_totals: dict[tuple[str, str], int] = {}
    status_rows = []
    for row in counts:
        service = row["service"]
        if row["request_match_category"] == "All retained trips":
            service_totals[(service, "April")] = int(row["april_trips"])
            service_totals[(service, "May")] = int(row["may_trips"])
        elif service in {"Uber", "Lyft"}:
            status_rows.append(row)
    yellow = next(r for r in counts if r["service"].lower() == "yellow taxi")
    service_totals[("Yellow", "April")] = int(yellow["april_trips"])
    service_totals[("Yellow", "May")] = int(yellow["may_trips"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.1), layout="constrained")
    services = ["Yellow", "Uber", "Lyft"]
    months = ["April", "May"]
    x = np.arange(len(services))
    width = 0.34
    for offset, month, color in [(-width / 2, "April", BLUE), (width / 2, "May", CLAY)]:
        values = [service_totals[(s, month)] for s in services]
        ax1.bar(x + offset, values, width=width, color=color, label=month)
        for xi, value in zip(x + offset, values):
            ax1.text(xi, value * 1.02, fmt_count(value), ha="center", va="bottom", fontsize=10)
    ax1.set_xticks(x, services)
    ax1.set_ylabel("Retained completed trips")
    ax1.set_title("(a) Retained Trips")
    ax1.grid(axis="y", color=GRID)
    ax1.legend(frameon=False, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)

    # Plot only the non-N/N statuses. N/N is the large remainder (about 97--100%)
    # and including it in a 0--3.2% axis clips the stack and hides the categories
    # that Task 1 is intended to compare.
    status_order = ["Y/N", "Y/Y", "N/Y"]
    labels = []
    bottoms = np.zeros(4)
    colors = {"Y/N": GOLD, "Y/Y": BLUE, "N/Y": CLAY}
    row_map = {(r["service"], r["request_match_category"]): r for r in status_rows}
    positions = np.arange(4)
    combo = [("Uber", "April"), ("Uber", "May"), ("Lyft", "April"), ("Lyft", "May")]
    labels = [f"{s}\n{m[:3]}" for s, m in combo]
    for status in status_order:
        values = []
        for service, month in combo:
            field = "april_trips" if month == "April" else "may_trips"
            total = service_totals[(service, month)]
            values.append(100 * int(row_map[(service, status)][field]) / total)
        ax2.bar(positions, values, bottom=bottoms, color=colors[status], label=status)
        bottoms += np.array(values)
    ax2.set_xticks(positions, labels)
    ax2.set_ylabel("Share of retained trips (%)")
    ax2.set_title("(b) Non-N/N Reported Status Share")
    ax2.set_ylim(0, 3.2)
    ax2.text(0.01, 0.98, "N/N is the remainder to 100%", transform=ax2.transAxes, ha="left", va="top", fontsize=9, color=MUTED)
    ax2.grid(axis="y", color=GRID)
    ax2.legend(title="Status", frameon=False, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Task 1 Service and Status Overview", fontsize=16, weight="bold")
    save_figure(
        fig,
        "task1_service_status_overview",
        figure_id="T1-F1",
        task="1",
        title="Task 1 Service and Status Overview",
        input_files=input_files,
        main=True,
        key_message="Uber and Lyft account for all retained HVFHV trips; reported shared statuses are a small share of valid trips.",
        caveat="Yellow has no reported sharing status; Juno and Via have zero retained records.",
        purpose="Combine retained service volumes with the Task 1 request/match status distribution.",
        key_variables="service, month, retained trips, N/N, Y/N, Y/Y, N/Y",
        caption="Retained completed trips by service and month, with the non-N/N reported request/match status shares for Uber and Lyft. N/N is the remainder to 100%; Juno and Via have zero retained observations and Yellow has no sharing status field.",
        script_function="figure_task1",
    )


def figure_task2_demand(shapes: dict[int, list[np.ndarray]]) -> None:
    rows = read_csv(2, "zone_demand.csv")
    input_files = [table_path(2, "zone_demand.csv")]
    max_by_service_measure: dict[tuple[str, str], float] = {}
    for service in ("Yellow", "HVFHV"):
        for measure in ("pickup", "dropoff"):
            vals = [
                int(r["trip_count"])
                for r in rows
                if r["service"] == service and r["measure"] == measure and r["drawable"] == "True"
            ]
            max_by_service_measure[(service, measure)] = max(vals)
    for month in ("2026-04", "2026-05"):
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2), layout="constrained")
        specs = [
            ("Yellow", "pickup", "Yellow pickup"),
            ("Yellow", "dropoff", "Yellow dropoff"),
            ("HVFHV", "pickup", "HVFHV pickup"),
            ("HVFHV", "dropoff", "HVFHV dropoff"),
        ]
        for ax, (service, measure, title) in zip(axes.ravel(), specs):
            vals = {
                int(r["LocationID"]): int(r["trip_count"])
                for r in rows
                if r["month"] == month and r["service"] == service and r["measure"] == measure
            }
            norm = LogNorm(vmin=1, vmax=max_by_service_measure[(service, measure)])
            draw_choropleth(ax, shapes, vals, cmap="YlGnBu", norm=norm)
            ax.set_title(title, fontsize=13, weight="bold")
            sm = ScalarMappable(norm=norm, cmap="YlGnBu")
            cbar = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.01)
            cbar.set_label("Completed trips (log scale)", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        outside = [
            r
            for r in rows
            if r["month"] == month
            and r["LocationID"] == "265"
            and r["service"] == "HVFHV"
            and r["measure"] == "dropoff"
        ][0]
        fig.suptitle(f"Task 2 Spatial Demand Overview, {month}", fontsize=16, weight="bold")
        fig.text(
            0.5,
            0.015,
            f"Note: zone 265 (Outside of NYC) has no polygon; HVFHV dropoffs to zone 265 total {int(outside['trip_count']):,} and are not colored.",
            fontsize=9.5,
            color=MUTED,
            ha="center",
        )
        month_name = "april" if month.endswith("04") else "may"
        save_figure(
            fig,
            f"task2_demand_{month_name}",
            figure_id=f"T2-F{'1' if month.endswith('04') else '2'}",
            task="2",
            title=f"Task 2 Spatial Demand Overview, {month}",
            input_files=input_files,
            main=True,
            key_message="Yellow demand is concentrated in Manhattan zones, while HVFHV airport and outside-city patterns are prominent.",
            caveat="Zone 265 is counted in the table but cannot be displayed spatially because it has no polygon.",
            purpose="Compare Yellow and HVFHV pickup/dropoff demand in one compact monthly panel.",
            key_variables="month, service, pickup/dropoff, LocationID, trip_count",
            caption=f"Pickup and dropoff completed-trip counts by TLC zone in {month}. Color bars use service-measure scales shared across April and May; zone 265 is omitted from the map because no polygon exists.",
            script_function="figure_task2_demand",
        )


def figure_task2_od_summary() -> None:
    volume = read_csv(2, "top10_volume_od.csv")
    spending = read_csv(2, "top10_spending_od.csv")
    inputs = [table_path(2, "top10_volume_od.csv"), table_path(2, "top10_spending_od.csv")]
    groups = [(m, s) for m in ("2026-04", "2026-05") for s in ("Yellow", "Uber", "Lyft")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.3), gridspec_kw={"width_ratios": [1, 1.55]}, layout="constrained")
    y = np.arange(len(groups))[::-1]
    labels = [f"{s} {m[-2:]}" for m, s in groups]
    for yi, (month, service) in zip(y, groups):
        srows = [r for r in spending if r["month"] == month and r["service"] == service]
        overlap = sum(r["also_volume_top10"] == "True" for r in srows)
        outside_rows = [r for r in srows if r["DOLocationID"] == "265"]
        outside_share = 100 * sum(float(r["total_spending_usd"]) for r in outside_rows) / sum(float(r["total_spending_usd"]) for r in srows)
        color = SERVICE_COLORS[service]
        ax1.barh(yi, overlap, color=color, height=0.58)
        ax1.text(overlap + 0.14, yi, f"{overlap}/10", va="center", fontsize=11, weight="bold")
        ax2.barh(yi, outside_share, color=color, height=0.58)
        ax2.text(outside_share + 1, yi, f"{outside_share:.1f}% ({len(outside_rows)}/10)", va="center", fontsize=10)
    ax1.set_yticks(y, labels)
    ax2.set_yticks(y, [""] * len(groups))
    ax1.set_xlim(0, 10)
    ax2.set_xlim(0, 100)
    ax1.set_xlabel("Common OD pairs in both top-ten lists")
    ax2.set_xlabel("Share of top-ten spending ending in zone 265 (%)")
    ax1.set_title("(a) Top-Volume vs Top-Spending Overlap")
    ax2.set_title("(b) Outside-of-NYC in Spending Top Ten")
    for ax in (ax1, ax2):
        ax.grid(axis="x", color=GRID)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Task 2 OD-Flow Comparison", fontsize=16, weight="bold")
    save_figure(
        fig,
        "task2_od_flow_comparison",
        figure_id="T2-F3",
        task="2",
        title="Task 2 OD-Flow Comparison",
        input_files=inputs,
        main=True,
        key_message="Only two or three top-volume OD pairs also appear among the top-spending OD pairs.",
        caveat="The right panel is a share within each spending top-ten list, not a share of all service-month spending.",
        purpose="Summarize the difference between OD rankings by volume and by passenger spending.",
        key_variables="top10 volume OD, top10 spending OD, overlap, destination zone 265",
        caption="Comparison of independently ranked top-ten directed OD lists. The overlap counts common pairs between volume and spending rankings; the spending share reports how much of top-ten spending ends in Outside of NYC.",
        script_function="figure_task2_od_summary",
    )


def figure_od_maps(shapes: dict[int, list[np.ndarray]]) -> None:
    volume = read_csv(2, "top10_volume_od.csv")
    spending = read_csv(2, "top10_spending_od.csv")
    centers = centroids(shapes)
    xmin, xmax, ymin, ymax = shape_bounds(shapes)
    offmap = {
        264: (xmax - 0.02 * (xmax - xmin), ymin + 0.08 * (ymax - ymin)),
        265: (xmax + 0.08 * (xmax - xmin), ymax - 0.12 * (ymax - ymin)),
    }
    specs = [("volume", volume, "trip_count", "Top Directed OD Flows by Trip Volume"), ("spending", spending, "total_spending_usd", "Top Directed OD Flows by Passenger Spending")]
    for kind, rows, value_field, title in specs:
        for month in ("2026-04", "2026-05"):
            for service in ("Yellow", "Uber", "Lyft"):
                subset = sorted([r for r in rows if r["month"] == month and r["service"] == service], key=lambda r: int(r["rank"]))
                fig = plt.figure(figsize=(12.2, 7.0), layout="constrained")
                gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1])
                ax = fig.add_subplot(gs[0, 0])
                tbl = fig.add_subplot(gs[0, 1])
                draw_base_map(ax, shapes)
                ax.set_xlim(xmin, xmax + 0.14 * (xmax - xmin))
                values = [float(r[value_field]) for r in subset]
                max_value = max(values)
                for r, value in zip(subset, values):
                    origin = int(r["PULocationID"])
                    dest = int(r["DOLocationID"])
                    start = centers.get(origin, offmap.get(origin))
                    end = centers.get(dest, offmap.get(dest))
                    if start is None or end is None:
                        continue
                    rank = int(r["rank"])
                    width = 1.6 + 4.0 * math.sqrt(value / max_value)
                    alpha = 0.35 + 0.6 * (11 - rank) / 10
                    color = SERVICE_COLORS[service]
                    if origin == dest:
                        ax.add_patch(Circle(start, 5200 + 900 * (11 - rank), fill=False, edgecolor=color, linewidth=width, alpha=alpha, zorder=3))
                    else:
                        ax.add_patch(
                            FancyArrowPatch(
                                start,
                                end,
                                arrowstyle="-|>",
                                mutation_scale=12 + 1.6 * (11 - rank),
                                color=color,
                                linewidth=width,
                                alpha=alpha,
                                connectionstyle=f"arc3,rad={0.04 if rank % 2 else -0.04}",
                                zorder=3,
                            )
                        )
                    for point, marker, edge in [(start, "o", "#1b3b52"), (end, "s", "#7a331f")]:
                        ax.scatter(*point, marker=marker, s=42, facecolor="white", edgecolor=edge, linewidth=1.2, zorder=4)
                ax.text(offmap[265][0], offmap[265][1], "Outside of NYC (265)", fontsize=10, ha="center", va="bottom", color=INK)
                ax.set_title(f"{service}, {month}", fontsize=14, weight="bold")
                tbl.axis("off")
                tbl.set_title("Ranking", loc="left", fontsize=13, weight="bold", pad=8)
                y = 0.98
                for r in subset:
                    val = fmt_count(float(r[value_field])) if kind == "volume" else f"${float(r[value_field]) / 1_000_000:.2f}M"
                    label = "\n".join(wrap(flow_label(r), 48))
                    tbl.text(0.0, y, f"{r['rank']}. {label}", fontsize=9.2, weight="bold", va="top", color=INK)
                    tbl.text(0.0, y - 0.075, val, fontsize=8.7, va="top", color=MUTED)
                    y -= 0.098
                fig.suptitle(title, fontsize=16, weight="bold")
                save_figure(
                    fig,
                    f"task2_od_{kind}_{month}_{service.lower()}",
                    figure_id=f"T2-A-{kind[:1].upper()}{month[-2:]}-{service}",
                    task="2",
                    title=f"{title}: {service} {month}",
                    input_files=[table_path(2, "top10_volume_od.csv" if kind == "volume" else "top10_spending_od.csv")],
                    main=False,
                    key_message="Top OD flows are labelled with official zone names and IDs.",
                    caveat="Arrows connect zone centroids and do not represent actual routes; zone 265 is shown as an off-map lookup category.",
                    purpose="Appendix map and ranking for the top ten directed OD pairs.",
                    key_variables="rank, PULocationID, DOLocationID, trip count or spending",
                    caption=f"Top ten directed {service} OD flows in {month}, ranked by {'trip volume' if kind == 'volume' else 'aggregate passenger spending'}. Zone names come from the official Taxi Zone lookup.",
                    script_function="figure_od_maps",
                )


def figure_task3_metrics() -> None:
    rows = read_csv(3, "monthly_rates.csv")
    inputs = [table_path(3, "monthly_rates.csv")]
    metrics = [
        ("pooling_request_rate", "Pooling requests / valid trips", "valid_status_denominator"),
        ("reported_matched_rate", "Reported matches / valid trips", "valid_status_denominator"),
        ("matching_success", "Y/Y trips / pooling requests", "pooling_requests"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), layout="constrained")
    for ax, (field, title, denom) in zip(axes, metrics):
        x = np.arange(2)
        width = 0.34
        for offset, service in [(-width / 2, "Uber"), (width / 2, "Lyft")]:
            service_rows = sorted([r for r in rows if r["service"] == service], key=lambda r: r["month"])
            values = [100 * float(r[field]) for r in service_rows]
            ax.bar(x + offset, values, width=width, color=SERVICE_COLORS[service], label=service)
            for xi, value, row in zip(x + offset, values, service_rows):
                decimals = 3 if service == "Lyft" and field != "matching_success" else 2
                ax.text(xi, value + max(values + [0.01]) * 0.04, f"{value:.{decimals}f}%", ha="center", fontsize=9.5, weight="bold")
        ax.set_xticks(x, ["April", "May"])
        ax.set_title(title)
        ax.set_ylabel("Percent")
        ax.grid(axis="y", color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Reported Ride-Pooling Metrics by Platform and Month", fontsize=16, weight="bold")
    fig.text(0.5, 0.012, "Note: y-axis scales differ across panels so the small Lyft rates remain visible.", fontsize=9.5, color=MUTED, ha="center")
    save_figure(
        fig,
        "task3_ride_pooling_metrics",
        figure_id="T3-F1",
        task="3",
        title="Reported Ride-Pooling Metrics by Platform and Month",
        input_files=inputs,
        main=True,
        key_message="Uber has substantially higher reported request and match rates than Lyft; Uber success among requests declines from April to May.",
        caveat="Metrics are based on reported completed-trip flags and do not verify actual shared overlap.",
        purpose="Separate request, match, and success measures without combining them into a generic sharing rate.",
        key_variables="pooling_request_rate, reported_matched_rate, matching_success",
        caption="Reported ride-pooling request, match, and success measures for Uber and Lyft. Y-axis scales differ across panels; denominators are reported in the accompanying text and tables.",
        script_function="figure_task3_metrics",
    )


def figure_task3_appendix_maps(shapes: dict[int, list[np.ndarray]]) -> None:
    rows = read_csv(3, "zone_rates.csv")
    metrics = [
        ("pooling_request_rate", "Request / valid trips"),
        ("reported_matched_rate", "Match / valid trips"),
        ("matching_success", "Y/Y / requests"),
    ]
    max_metric = {}
    for field, _ in metrics:
        vals = [float(r[field]) for r in rows if r[field] and r[f"{field}_status"] == "reported"]
        max_metric[field] = max(vals)
    for month in ("2026-04", "2026-05"):
        for service in ("Uber", "Lyft"):
            fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4), layout="constrained")
            subset = [r for r in rows if r["month"] == month and r["service"] == service]
            for ax, (field, title) in zip(axes, metrics):
                values = {int(r["PULocationID"]): (float(r[field]) if r[field] and r[f"{field}_status"] == "reported" else None) for r in subset}
                insufficient = {int(r["PULocationID"]) for r in subset if r[f"{field}_status"] == "insufficient_sample"}
                norm = Normalize(vmin=0, vmax=max_metric[field])
                draw_zone_metric(ax, shapes, values, cmap="PuBuGn", norm=norm, insufficient=insufficient)
                ax.set_title(title, fontsize=12, weight="bold")
                sm = ScalarMappable(norm=norm, cmap="PuBuGn")
                cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.01)
                cbar.set_label("Rate", fontsize=9)
                cbar.ax.tick_params(labelsize=8)
            fig.suptitle(f"Task 3 Zone-Level Reported Ride-Pooling Metrics: {service} {month}", fontsize=15, weight="bold")
            save_figure(
                fig,
                f"task3_zone_metrics_{month}_{service.lower()}",
                figure_id=f"T3-A-{month[-2:]}-{service}",
                task="3",
                title=f"Task 3 Zone-Level Metrics: {service} {month}",
                input_files=[table_path(3, "zone_rates.csv")],
                main=False,
                key_message="Zone-level rates are reportable only where denominator thresholds are met.",
                caveat="Gray zones have insufficient sample and should not be read as zero.",
                purpose="Appendix maps for request, match, and success metrics by pickup zone.",
                key_variables="PULocationID, valid_status_denominator, pooling_requests, rates",
                caption=f"Zone-level reported ride-pooling metrics for {service} in {month}. Gray zones have insufficient sample; zones 264/265 are in the table but have no polygon.",
                script_function="figure_task3_appendix_maps",
            )


def figure_task4_monthly() -> None:
    rows = read_csv(4, "month_summary.csv")
    inputs = [table_path(4, "month_summary.csv")]
    fig, (ax, cov) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
    y = np.arange(len(rows))[::-1]
    for yi, row, color, label in zip(y, rows, [BLUE, CLAY], ["April", "May"]):
        q1 = float(row["excess_p25_minutes"])
        med = float(row["excess_median_minutes"])
        q3 = float(row["excess_p75_minutes"])
        ax.plot([q1, q3], [yi, yi], color=color, linewidth=10, solid_capstyle="round")
        ax.scatter(med, yi, s=115, facecolor="white", edgecolor=color, linewidth=2.5, zorder=3)
        ax.text(q3 + 0.3, yi, f"median {med:.2f} min", va="center", fontsize=11)
        coverage = 100 * float(row["reference_coverage"])
        cov.barh(yi, coverage, color=color, height=0.35)
        cov.text(coverage + 1, yi, f"{coverage:.1f}% ({int(row['reference_supported']):,}/{int(row['shared_eligible']):,})", va="center", fontsize=10)
    ax.axvline(0, color=INK, linestyle="--", linewidth=1)
    ax.set_yticks(y, ["April", "May"])
    cov.set_yticks(y, ["", ""])
    ax.set_xlabel("Observed travel-time difference (min)")
    cov.set_xlabel("Reference coverage (%)")
    ax.set_title("Shared - non-shared reference travel time")
    cov.set_title("Supported reference coverage")
    cov.set_xlim(0, 100)
    ax.grid(axis="x", color=GRID)
    cov.grid(axis="x", color=GRID)
    for a in (ax, cov):
        a.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Observed Travel-Time Difference", fontsize=16, weight="bold")
    save_figure(
        fig,
        "task4_travel_time_difference",
        figure_id="T4-F1",
        task="4",
        title="Observed Travel-Time Difference",
        input_files=inputs,
        main=True,
        key_message="Supported Uber Y/Y trips have positive median observed travel-time differences in both months.",
        caveat="Results are observational and cover only Y/Y trips with supported non-shared reference cells.",
        purpose="Show median, IQR, and reference coverage for the central Task 4 comparison.",
        key_variables="excess_p25_minutes, excess_median_minutes, excess_p75_minutes, reference_coverage",
        caption="Observed travel-time differences between reported shared Uber trips and non-shared reference trips in supported cells. Thick segments show the IQR; circles show medians; bars show reference coverage.",
        script_function="figure_task4_monthly",
    )


def figure_task4_hourly() -> None:
    rows = read_csv(4, "hourly_summary.csv")
    fig, (ax, cov) = plt.subplots(2, 1, figsize=(11.4, 6.5), sharex=True, layout="constrained")
    for month, label, color in [("2026-04", "April", BLUE), ("2026-05", "May", CLAY)]:
        subset = sorted([r for r in rows if r["month"] == month], key=lambda r: int(r["pickup_hour"]))
        hours = [int(r["pickup_hour"]) for r in subset]
        med = [float(r["excess_median_minutes"]) for r in subset]
        coverage = [100 * float(r["reference_coverage"]) for r in subset]
        ax.plot(hours, med, marker="o", markersize=4, color=color, linewidth=2.2, label=label)
        cov.plot(hours, coverage, marker="o", markersize=4, color=color, linewidth=2.2, label=label)
    ax.set_ylabel("Median observed difference (min)")
    cov.set_ylabel("Reference coverage (%)")
    cov.set_xlabel("Pickup hour")
    cov.set_xticks(range(0, 24, 2))
    ax.set_title("(a) Observed travel-time difference by pickup hour", loc="left")
    cov.set_title("(b) Supported-reference coverage by pickup hour", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for a in (ax, cov):
        a.grid(color=GRID)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Task 4 Hourly Pattern and Coverage", fontsize=16, weight="bold")
    save_figure(
        fig,
        "task4_hourly_pattern_coverage",
        figure_id="T4-F2",
        task="4",
        title="Task 4 Hourly Pattern and Coverage",
        input_files=[table_path(4, "hourly_summary.csv")],
        main=True,
        key_message="Hourly medians vary modestly, while reference coverage changes by hour and affects interpretation.",
        caveat="Low-coverage hours should be interpreted cautiously.",
        purpose="Pair the hourly observed difference with its supported-reference coverage.",
        key_variables="pickup_hour, excess_median_minutes, reference_coverage",
        caption="Hourly median observed travel-time difference for supported Uber Y/Y trips, with the corresponding share of eligible trips having a supported non-shared reference.",
        script_function="figure_task4_hourly",
    )


def figure_task4_appendix_maps(shapes: dict[int, list[np.ndarray]]) -> None:
    rows = read_csv(4, "zone_summary.csv")
    for month in ("2026-04", "2026-05"):
        subset = [r for r in rows if r["month"] == month]
        values = {int(r["PULocationID"]): (float(r["mean_excess_minutes"]) if r["mean_excess_minutes"] else None) for r in subset if int(r["reference_supported"]) >= 10}
        all_values = [v for v in values.values() if v is not None]
        norm = TwoSlopeNorm(vmin=min(-2, min(all_values)), vcenter=0, vmax=max(8, max(all_values)))
        fig, ax = plt.subplots(figsize=(7.5, 7.0), layout="constrained")
        draw_zone_metric(ax, shapes, values, cmap="RdYlBu_r", norm=norm)
        sm = ScalarMappable(norm=norm, cmap="RdYlBu_r")
        cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.01)
        cbar.set_label("Mean observed difference (min)")
        fig.suptitle(f"Task 4 Mean Observed Travel-Time Difference by Pickup Zone, {month}", fontsize=15, weight="bold")
        save_figure(
            fig,
            f"task4_zone_mean_difference_{month}",
            figure_id=f"T4-A-{month[-2:]}",
            task="4",
            title=f"Task 4 Zone Mean Difference, {month}",
            input_files=[table_path(4, "zone_summary.csv")],
            main=False,
            key_message="Zone means vary geographically among supported shared trips.",
            caveat="Only zones with at least ten supported shared trips are colored; this is observational.",
            purpose="Appendix map of pickup-zone mean observed travel-time difference.",
            key_variables="PULocationID, reference_supported, mean_excess_minutes",
            caption=f"Mean observed travel-time difference by pickup zone for Uber Y/Y trips in {month}. Colored zones have at least ten supported shared trips.",
            script_function="figure_task4_appendix_maps",
        )
    combined: dict[int, list[float]] = {}
    for r in rows:
        combined.setdefault(int(r["PULocationID"]), []).append(float(r["reference_coverage"]))
    values = {k: float(np.mean(v)) for k, v in combined.items()}
    fig, ax = plt.subplots(figsize=(7.5, 7.0), layout="constrained")
    norm = Normalize(vmin=0, vmax=1)
    draw_zone_metric(ax, shapes, values, cmap="YlGnBu", norm=norm)
    sm = ScalarMappable(norm=norm, cmap="YlGnBu")
    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.01)
    cbar.set_label("Reference coverage")
    fig.suptitle("Task 4 Reference Coverage by Pickup Zone", fontsize=15, weight="bold")
    save_figure(
        fig,
        "task4_zone_reference_coverage",
        figure_id="T4-A-COV",
        task="4",
        title="Task 4 Reference Coverage by Pickup Zone",
        input_files=[table_path(4, "zone_summary.csv")],
        main=False,
        key_message="Reference support is geographically uneven.",
        caveat="Coverage is the mean of April and May zone coverage values.",
        purpose="Appendix map documenting spatial comparability for Task 4.",
        key_variables="PULocationID, reference_coverage",
        caption="Pickup-zone reference coverage for Uber Y/Y travel-time comparisons, averaged over April and May zone summaries.",
        script_function="figure_task4_appendix_maps",
    )


def figure_task5_price() -> None:
    rows = read_csv(5, "price_comparison.csv")
    fig, ax = plt.subplots(figsize=(11.3, 5.0), layout="constrained")
    positions = np.arange(len(rows))[::-1]
    for y, row in zip(positions, rows):
        est = float(row["mean_difference_usd"])
        low = float(row["ci95_low_usd"])
        high = float(row["ci95_high_usd"])
        color = SERVICE_COLORS[row["service"]]
        ax.plot([low, high], [y, y], color=color, linewidth=3.2, solid_capstyle="round")
        ax.scatter(est, y, s=110, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(high + 0.9, y, f"{est:.2f} USD; n={int(row['shared_supported']):,}; coverage={100 * float(row['shared_reference_coverage']):.1f}%", va="center", fontsize=10, color=color)
    labels = [f"{r['service']} - {'Apr' if r['month'].endswith('04') else 'May'}" for r in rows]
    ax.set_yticks(positions, labels)
    ax.axvline(0, color=INK, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Shared - non-shared reference passenger spending (USD)")
    ax.set_title("Observed Passenger-Spending Difference")
    ax.text(-37, positions[0] + 0.4, "Negative values indicate lower observed spending for reported shared trips.", fontsize=10, color=MUTED)
    ax.grid(axis="x", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(
        fig,
        "task5_passenger_spending_difference",
        figure_id="T5-F1",
        task="5",
        title="Observed Passenger-Spending Difference",
        input_files=[table_path(5, "price_comparison.csv")],
        main=True,
        key_message="Supported reported shared trips have negative shared-minus-reference spending differences in both months.",
        caveat="The comparison is observational, conditional on supported cells, and Lyft sample sizes are very small.",
        purpose="Show Task 5.1 point estimates, 95% CIs, sample sizes, and reference coverage.",
        key_variables="mean_difference_usd, ci95_low_usd, ci95_high_usd, shared_supported, shared_reference_coverage",
        caption="Observed passenger-spending differences between reported shared trips and comparable non-shared reference trips. Error bars show date-clustered 95% confidence intervals; negative values indicate lower observed spending for reported shared trips.",
        script_function="figure_task5_price",
    )


def write_manifest_files() -> None:
    md = INSTRUCTION / "FIGURE_MANIFEST.md"
    csv_path = INSTRUCTION / "figure_manifest.csv"
    lines = ["# Figure Manifest", "", "Generated from `python3 -m analysis.render_report_figures`.", ""]
    for row in MANIFEST:
        lines.extend(
            [
                f"## {row.figure_id}",
                "",
                f"Figure ID: {row.figure_id}",
                f"Task: {row.task}",
                f"Filename: {row.png_path.name}",
                f"PDF filename: {row.pdf_path.as_posix()}",
                f"PNG filename: {row.png_path.as_posix()}",
                "Input table(s):",
            ]
        )
        lines.extend([f"- {p.as_posix()}" for p in row.input_files])
        lines.extend(
            [
                f"Script/function: `analysis.render_report_figures::{row.script_function}`",
                f"Purpose: {row.purpose}",
                f"Main report or appendix: {row.main_or_appendix}",
                f"Key variables: {row.key_variables}",
                f"Important caveat: {row.caveat}",
                f"Suggested report caption: {row.caption}",
                "",
            ]
        )
    md.write_text("\n".join(lines), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "figure_id",
                "task",
                "title",
                "png_path",
                "pdf_path",
                "input_files",
                "main_or_appendix",
                "key_message",
                "caveat",
            ],
        )
        writer.writeheader()
        for row in MANIFEST:
            writer.writerow(
                {
                    "figure_id": row.figure_id,
                    "task": row.task,
                    "title": row.title,
                    "png_path": row.png_path.as_posix(),
                    "pdf_path": row.pdf_path.as_posix(),
                    "input_files": "; ".join(p.as_posix() for p in row.input_files),
                    "main_or_appendix": row.main_or_appendix,
                    "key_message": row.key_message,
                    "caveat": row.caveat,
                }
            )


def validate_outputs() -> list[str]:
    issues: list[str] = []
    for row in MANIFEST:
        for path in (row.png_path, row.pdf_path):
            if not path.is_file() or path.stat().st_size == 0:
                issues.append(f"Missing or empty output: {path}")
        try:
            image = Image.open(row.png_path).convert("RGB")
            width, height = image.size
            if width < 1200 or height < 900:
                issues.append(f"Small PNG dimensions for {row.png_path}: {width}x{height}")
            stat = ImageStat.Stat(image)
            if sum(stat.var) < 1:
                issues.append(f"Possibly blank image: {row.png_path}")
        except Exception as exc:  # pragma: no cover - validation path
            issues.append(f"Cannot open {row.png_path}: {exc}")
    if issues:
        raise RuntimeError("\n".join(issues))
    return issues


def make_contact_sheet() -> None:
    thumbs = []
    for row in MANIFEST:
        img = Image.open(row.png_path).convert("RGB")
        img.thumbnail((360, 260))
        canvas = Image.new("RGB", (380, 310), "white")
        canvas.paste(img, ((380 - img.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        draw.text((12, 275), row.figure_id, fill=(30, 45, 55), font=font)
        draw.text((82, 275), row.png_path.name[:34], fill=(80, 90, 95), font=font)
        thumbs.append(canvas)
    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 380, rows * 310), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 380, (idx // cols) * 310))
    sheet.save(INSTRUCTION / "figure_contact_sheet.png", dpi=(180, 180))
    sheet.save(INSTRUCTION / "figure_contact_sheet.pdf", "PDF", resolution=180)


def main() -> None:
    ensure_dirs()
    setup_style()
    shapes = load_shapes()
    lookup = load_zone_lookup()
    missing_lookup = [zone_id for zone_id in (264, 265) if zone_id not in lookup]
    if missing_lookup:
        raise ValueError(f"Taxi Zone lookup is missing required IDs: {missing_lookup}")
    figure_task1()
    figure_task2_demand(shapes)
    figure_task2_od_summary()
    figure_od_maps(shapes)
    figure_task3_metrics()
    figure_task3_appendix_maps(shapes)
    figure_task4_monthly()
    figure_task4_hourly()
    figure_task4_appendix_maps(shapes)
    figure_task5_price()
    write_manifest_files()
    validate_outputs()
    make_contact_sheet()
    print(f"Generated {len(MANIFEST)} report-ready figures.")
    print(f"Main figures: {sum(1 for r in MANIFEST if r.main_or_appendix == 'main')}")
    print(f"Appendix figures: {sum(1 for r in MANIFEST if r.main_or_appendix == 'appendix')}")
    print("Validated PNG/PDF outputs and wrote instruction/figure_contact_sheet.pdf.")


if __name__ == "__main__":
    main()
