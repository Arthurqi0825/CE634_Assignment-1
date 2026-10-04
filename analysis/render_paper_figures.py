"""Render annotated research-paper figures from processed CE634 result tables.

The script reads only packaged processed CSV tables and Taxi Zone reference files.
It does not require the raw TLC Parquet files.

Run from the project root:
    python3 -m analysis.render_paper_figures
"""
from __future__ import annotations

import csv
from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Polygon

from analysis.common import RESULTS_DIR, load_zone_lookup
import analysis.render_report_figures as rr

OUT = RESULTS_DIR / "paper_figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse the established report palette and geometry helpers.
BLUE = rr.BLUE
CLAY = rr.CLAY
GOLD = rr.GOLD
PLUM = rr.PLUM
TEAL = rr.TEAL
INK = rr.INK
MUTED = rr.MUTED
GRID = rr.GRID
LAND = rr.LAND
BOUNDARY = rr.BOUNDARY
MISSING = rr.MISSING
SERVICE_COLORS = rr.SERVICE_COLORS


def setup_style() -> None:
    rr.setup_style()
    plt.rcParams.update({
        "font.size": 11.5,
        "axes.titlesize": 13.5,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.titlesize": 16,
    })


def read_csv(task: int, name: str) -> list[dict[str, str]]:
    return rr.read_csv(task, name)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{name}.png", dpi=360, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def short_zone(name: str) -> str:
    repl = {
        "Upper East Side South": "Upper East Side S.",
        "Upper East Side North": "Upper East Side N.",
        "Times Sq/Theatre District": "Times Sq./Theatre",
        "Outside of NYC": "Outside NYC",
        "Crown Heights North": "Crown Heights N.",
        "Schuylerville/Edgewater Park": "Schuylerville/Edgewater",
    }
    return repl.get(name, name)


def draw_map(ax, shapes, values, norm, cmap="YlGnBu") -> None:
    cmap_obj = plt.get_cmap(cmap)
    for zid, parts in shapes.items():
        value = values.get(zid)
        if value is None or not np.isfinite(value) or value <= 0:
            face = MISSING
        else:
            face = cmap_obj(norm(value))
        for part in parts:
            ax.add_patch(Polygon(part, closed=True, facecolor=face, edgecolor="white", linewidth=0.18))
    xmin, xmax, ymin, ymax = rr.shape_bounds(shapes)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


def annotate_ranked_zones(ax, shapes, ranked_rows, max_items=3) -> None:
    centers = rr.centroids(shapes)
    # Highlight the leading zone on the map.
    if ranked_rows:
        top = ranked_rows[0]
        zid = int(top["LocationID"])
        if zid in centers:
            x, y = centers[zid]
            ax.scatter(x, y, s=90, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=5)
            ax.text(x, y, "1", ha="center", va="center", fontsize=8.5, weight="bold", color=INK, zorder=6)
    lines = ["Top zones"]
    for idx, row in enumerate(ranked_rows[:max_items], 1):
        lines.append(f"{idx}. {short_zone(row['zone'])} ({int(row['trip_count']):,})")
    ax.text(
        0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
        ha="left", va="bottom", fontsize=7.8, color=INK,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#c7d0d5", alpha=0.93),
        zorder=10,
    )


def figure1_dataset_overview() -> None:
    rows = read_csv(1, "task1_2_counts.csv")
    totals = {}
    yy = {}
    for r in rows:
        service = r["service"]
        cat = r["request_match_category"]
        if cat == "All retained trips":
            if service == "Yellow taxi":
                s = "Yellow"
            else:
                s = service
            totals[(s, "April")] = int(r["april_trips"])
            totals[(s, "May")] = int(r["may_trips"])
        if cat == "Y/Y" and service in {"Uber", "Lyft"}:
            yy[(service, "April")] = int(r["april_trips"])
            yy[(service, "May")] = int(r["may_trips"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"width_ratios": [1.25, 1]}, layout="constrained")
    services = ["Yellow", "Uber", "Lyft"]
    x = np.arange(3)
    w = 0.34
    for off, month, color in [(-w/2, "April", BLUE), (w/2, "May", CLAY)]:
        vals = [totals[(s, month)] / 1e6 for s in services]
        bars = ax1.bar(x + off, vals, width=w, color=color, label=month)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.25, f"{v:.2f}M", ha="center", va="bottom", fontsize=10, weight="bold")
    ax1.set_xticks(x, services)
    ax1.set_ylabel("Retained completed trips (millions)")
    ax1.set_title("(a) Analysis sample by service")
    ax1.legend(frameon=False, ncol=2, loc="upper left")
    ax1.grid(axis="y", color=GRID)
    ax1.spines[["top", "right"]].set_visible(False)
    total = sum(totals[(s,m)] for s in services for m in ("April","May"))
    ax1.text(0.98, 0.95, f"Total retained\n{total/1e6:.2f} million trips", transform=ax1.transAxes,
             ha="right", va="top", fontsize=11, weight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#c7d0d5"))

    labels = ["Uber Apr", "Uber May", "Lyft Apr", "Lyft May"]
    vals = [yy[("Uber","April")], yy[("Uber","May")], yy[("Lyft","April")], yy[("Lyft","May")]]
    ypos = np.arange(4)[::-1]
    colors = [BLUE, BLUE, PLUM, PLUM]
    ax2.barh(ypos, vals, color=colors, height=0.55)
    ax2.set_xscale("log")
    ax2.set_yticks(ypos, labels)
    ax2.set_xlabel("Reported matched shared trips, Y/Y (log scale)")
    ax2.set_title("(b) Size of the matched shared-trip sample")
    for y, v in zip(ypos, vals):
        ax2.text(v*1.15, y, f"{v:,}", va="center", fontsize=10.5, weight="bold")
    ax2.grid(axis="x", color=GRID, which="both")
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.text(0.03, 0.05, "Uber contributes >99.9% of recorded Y/Y trips,\nso later matched comparisons are driven by Uber.",
             transform=ax2.transAxes, fontsize=9.7, color=INK,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f9fa", edgecolor="#c7d0d5"))

    fig.suptitle("Dataset scale and the effective shared-trip sample", weight="bold")
    save(fig, "fig1_dataset_overview")


def demand_figure(month: str, filename: str) -> None:
    rows = read_csv(2, "zone_demand.csv")
    shapes = rr.load_shapes()
    specs = [
        ("Yellow", "pickup", "Yellow pickup"),
        ("Yellow", "dropoff", "Yellow dropoff"),
        ("HVFHV", "pickup", "HVFHV pickup"),
        ("HVFHV", "dropoff", "HVFHV dropoff"),
    ]
    max_by = {}
    for service, measure, _ in specs:
        vals = [int(r["trip_count"]) for r in rows if r["service"]==service and r["measure"]==measure and r["drawable"]=="True"]
        max_by[(service, measure)] = max(vals)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.9), layout="constrained")
    for ax, (service, measure, title) in zip(axes.ravel(), specs):
        ss = [r for r in rows if r["month"]==month and r["service"]==service and r["measure"]==measure and r["drawable"]=="True"]
        values = {int(r["LocationID"]): int(r["trip_count"]) for r in ss}
        norm = LogNorm(vmin=1, vmax=max_by[(service, measure)])
        draw_map(ax, shapes, values, norm)
        ranked = sorted(ss, key=lambda r:int(r["trip_count"]), reverse=True)
        annotate_ranked_zones(ax, shapes, ranked)
        ax.set_title(title, fontsize=13, weight="bold")
        sm = ScalarMappable(norm=norm, cmap="YlGnBu")
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
        cb.set_label("Trips (log scale)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    outside = [r for r in rows if r["month"]==month and r["service"]=="HVFHV" and r["measure"]=="dropoff" and r["LocationID"]=="265"][0]
    month_name = "April 2026" if month.endswith("04") else "May 2026"
    fig.suptitle(f"Spatial distribution of completed-trip demand - {month_name}", weight="bold")
    fig.text(0.5, 0.012,
             f"The numbered marker identifies the leading drawable zone in each panel. Zone 265 (Outside NYC) has no polygon; HVFHV dropoffs to zone 265 = {int(outside['trip_count']):,}.",
             ha="center", fontsize=9.2, color=MUTED)
    save(fig, filename)


def figure4_od_summary() -> None:
    volume = read_csv(2, "top10_volume_od.csv")
    spending = read_csv(2, "top10_spending_od.csv")
    groups = [(m,s) for m in ("2026-04","2026-05") for s in ("Yellow","Uber","Lyft")]
    fig = plt.figure(figsize=(13.4, 7.2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 2.25])
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    y = np.arange(len(groups))[::-1]
    for yi,(month,service) in zip(y,groups):
        srows=[r for r in spending if r["month"]==month and r["service"]==service]
        overlap=sum(r["also_volume_top10"]=="True" for r in srows)
        ax1.barh(yi, overlap, color=SERVICE_COLORS[service], height=0.55)
        ax1.text(overlap+0.15, yi, f"{overlap}/10", va="center", fontsize=10.5, weight="bold")
    ax1.set_yticks(y,[f"{s}\n{'Apr' if m.endswith('04') else 'May'}" for m,s in groups])
    ax1.set_xlim(0,10)
    ax1.set_xlabel("Common OD pairs")
    ax1.set_title("(a) Top-ten overlap", fontsize=12.5)
    ax1.grid(axis="x", color=GRID)
    ax1.spines[["top","right","left"]].set_visible(False)

    ax2.axis("off")
    ax2.text(0.00, 0.965, "(b) Leading OD pair under each ranking criterion", transform=ax2.transAxes, fontsize=12.5, weight="bold", ha="left", va="top")
    ax2.text(0.20, 0.905, "Highest trip volume", transform=ax2.transAxes, fontsize=11.5, weight="bold", ha="center")
    ax2.text(0.70, 0.905, "Highest aggregate passenger spending", transform=ax2.transAxes, fontsize=11.5, weight="bold", ha="center")
    ax2.plot([0.45,0.45],[0.05,0.95], transform=ax2.transAxes, color="#d9e0e4", lw=1)
    y0=0.82; step=0.125
    for idx,(month,service) in enumerate(groups):
        vr=next(r for r in volume if r["month"]==month and r["service"]==service and r["rank"]=="1")
        sr=next(r for r in spending if r["month"]==month and r["service"]==service and r["rank"]=="1")
        yy=y0-idx*step
        ax2.scatter(0.02, yy, s=65, color=SERVICE_COLORS[service], transform=ax2.transAxes)
        ax2.text(0.055, yy, f"{service} {'Apr' if month.endswith('04') else 'May'}", transform=ax2.transAxes,
                 ha="left", va="center", fontsize=10.5, weight="bold")
        vflow=f"{short_zone(vr['origin_zone'])} ({vr['PULocationID']})\n→ {short_zone(vr['destination_zone'])} ({vr['DOLocationID']})"
        sflow=f"{short_zone(sr['origin_zone'])} ({sr['PULocationID']})\n→ {short_zone(sr['destination_zone'])} ({sr['DOLocationID']})"
        ax2.text(0.20, yy, vflow, transform=ax2.transAxes, ha="center", va="center", fontsize=9.5)
        ax2.text(0.70, yy, sflow, transform=ax2.transAxes, ha="center", va="center", fontsize=9.5)
        ax2.text(0.38, yy, f"{int(vr['trip_count']):,} trips", transform=ax2.transAxes, ha="center", va="center", fontsize=8.7, color=MUTED)
        ax2.text(0.94, yy, f"${float(sr['total_spending_usd'])/1e6:.2f}M", transform=ax2.transAxes, ha="center", va="center", fontsize=8.7, color=MUTED)
        if idx < len(groups)-1:
            ax2.plot([0.0,0.98],[yy-step/2,yy-step/2], transform=ax2.transAxes, color="#edf0f2", lw=1)
    ax2.text(0.02, 0.02,
             "Key result: busiest corridors and highest-spending corridors are not interchangeable; airport/outside-NYC flows dominate the spending leaders.",
             transform=ax2.transAxes, fontsize=9.6, color=INK,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f9fa", edgecolor="#c7d0d5"))
    fig.suptitle("Volume and spending rankings identify different OD leaders", weight="bold", y=1.01)
    save(fig, "fig4_od_ranking_summary")


def figure5_pooling_metrics() -> None:
    rows=read_csv(3,"monthly_rates.csv")
    metrics=[
        ("pooling_request_rate","Pooling request rate","valid trips"),
        ("reported_matched_rate","Reported matched rate","valid trips"),
        ("matching_success","Match success among requests","pooling requests"),
    ]
    fig,axes=plt.subplots(2,3,figsize=(13.0,7.5),layout="constrained")
    for ri,service in enumerate(("Uber","Lyft")):
        srows=sorted([r for r in rows if r["service"]==service],key=lambda r:r["month"])
        for ci,(field,title,denom) in enumerate(metrics):
            ax=axes[ri,ci]
            vals=[100*float(r[field]) for r in srows]
            x=np.array([0,1])
            ax.plot(x,vals,marker="o",markersize=8,lw=2.4,color=SERVICE_COLORS[service])
            ax.set_xticks(x,["Apr","May"])
            ymax=max(vals)*1.28 if max(vals)>0 else 1
            ymin=0
            ax.set_ylim(ymin,ymax)
            decimals=3 if service=="Lyft" and field!="matching_success" else 2
            for xx,v in zip(x,vals):
                ax.text(xx,v+ymax*0.055,f"{v:.{decimals}f}%",ha="center",va="bottom",fontsize=10,weight="bold")
            if ri==0: ax.set_title(title,fontsize=12.3,weight="bold")
            if ci==0: ax.set_ylabel(f"{service}\nPercent",weight="bold")
            else: ax.set_ylabel("Percent")
            ax.grid(axis="y",color=GRID)
            ax.spines[["top","right"]].set_visible(False)
            ax.text(0.02,0.04,f"Denominator: {denom}",transform=ax.transAxes,fontsize=8.5,color=MUTED)
    uber_apr=next(r for r in rows if r["service"]=="Uber" and r["month"]=="2026-04")
    lyft_apr=next(r for r in rows if r["service"]=="Lyft" and r["month"]=="2026-04")
    ratio=float(uber_apr["pooling_request_rate"])/float(lyft_apr["pooling_request_rate"])
    fig.suptitle("Reported ride-pooling activity differs strongly by platform", weight="bold")
    save(fig,"fig5_pooling_metrics")


def figure6_travel_time_monthly() -> None:
    rows=read_csv(4,"month_summary.csv")
    fig,(ax,cov)=plt.subplots(1,2,figsize=(12.4,6.0),gridspec_kw={"width_ratios":[1.55,1]},layout="constrained")
    y=np.arange(len(rows))[::-1]
    labels=["April","May"]
    colors=[BLUE,CLAY]
    ax.axvspan(0,12,facecolor="#eef5f8",zorder=0)
    ax.axvline(0,color=INK,ls="--",lw=1.1)
    for yi,row,label,color in zip(y,rows,labels,colors):
        q1=float(row["excess_p25_minutes"]); med=float(row["excess_median_minutes"]); q3=float(row["excess_p75_minutes"])
        rel=float(row["relative_median_percent"])
        ax.plot([q1,q3],[yi,yi],color=color,lw=11,solid_capstyle="round")
        ax.scatter(med,yi,s=140,facecolor="white",edgecolor=color,lw=2.7,zorder=4)
        label_y = yi - 0.19 if label == "April" else yi + 0.19
        va = "top" if label == "April" else "bottom"
        ax.text(med,label_y,f"median +{med:.2f} min",ha="center",va=va,fontsize=10.5,weight="bold",color=color)
        ax.text(q3+0.25,yi,f"IQR {q1:.2f} to {q3:.2f}\nmedian relative +{rel:.1f}%",ha="left",va="center",fontsize=9.2)
        coverage=100*float(row["reference_coverage"])
        b=cov.barh(yi,coverage,color=color,height=.38)
        cov.text(coverage+1.2,yi,f"{coverage:.1f}%",va="center",fontsize=10.5,weight="bold")
        cov.text(coverage/2,yi,f"{int(row['reference_supported']):,}\n supported",ha="center",va="center",fontsize=8.8,color="white",weight="bold")
    ax.set_yticks(y,labels); cov.set_yticks(y,["",""])
    ax.set_xlim(-1,12.5); cov.set_xlim(0,100)
    ax.set_xlabel("Shared - non-shared reference travel time (min)")
    cov.set_xlabel("Eligible shared trips with a supported reference (%)")
    ax.set_title("(a) Distribution of observed differences")
    cov.set_title("(b) Reference coverage")
    ax.text(0.02,0.04,"Positive values mean the shared trip took longer\nthan the median non-shared trip in the same OD × hour × day-type cell.",
            transform=ax.transAxes,fontsize=9.1,color=INK,
            bbox=dict(boxstyle="round,pad=.4",facecolor="white",edgecolor="#c7d0d5"))
    for a in (ax,cov):
        a.grid(axis="x",color=GRID)
        a.spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Supported Uber shared trips show a positive observed travel-time difference",weight="bold")
    save(fig,"fig6_travel_time_monthly")


def figure7_travel_time_hourly() -> None:
    rows=read_csv(4,"hourly_summary.csv")
    fig,(ax,cov)=plt.subplots(2,1,figsize=(11.8,8.0),sharex=True,layout="constrained")
    extrema={"2026-04":{},"2026-05":{}}
    for month,label,color in [("2026-04","April",BLUE),("2026-05","May",CLAY)]:
        ss=sorted([r for r in rows if r["month"]==month],key=lambda r:int(r["pickup_hour"]))
        hours=np.array([int(r["pickup_hour"]) for r in ss])
        med=np.array([float(r["excess_median_minutes"]) for r in ss])
        cover=np.array([100*float(r["reference_coverage"]) for r in ss])
        ax.plot(hours,med,marker="o",ms=4.5,lw=2.3,color=color,label=label)
        cov.plot(hours,cover,marker="o",ms=4.5,lw=2.3,color=color,label=label)
        imin=int(np.argmin(med)); imax=int(np.argmax(med))
        for idx,tag in [(imin,"min"),(imax,"max")]:
            ax.scatter(hours[idx],med[idx],s=65,facecolor="white",edgecolor=color,lw=2,zorder=4)
            dy=-24 if tag=="min" else 12
            ax.annotate(f"{label} {hours[idx]:02d}:00\n{med[idx]:.2f} min",(hours[idx],med[idx]),xytext=(0,dy),textcoords="offset points",ha="center",fontsize=8.8,color=color,
                        arrowprops=dict(arrowstyle="-",color=color,lw=1))
        low=int(np.argmin(cover))
        cov.scatter(hours[low],cover[low],s=65,facecolor="white",edgecolor=color,lw=2,zorder=4)
        cov.annotate(f"{label} lowest support\n{hours[low]:02d}:00: {cover[low]:.1f}%",(hours[low],cover[low]),xytext=(10,-30 if label=="May" else 12),textcoords="offset points",fontsize=8.8,color=color,
                     arrowprops=dict(arrowstyle="-",color=color,lw=1))
    ax.set_ylabel("Median observed difference (min)")
    cov.set_ylabel("Reference coverage (%)")
    cov.set_xlabel("Pickup hour")
    cov.set_xticks(range(0,24,2))
    cov.axhspan(0,30,facecolor="#f3efef",zorder=0)
    cov.text(23.7,28,"<30% support",ha="right",va="top",fontsize=8.8,color=MUTED)
    ax.set_title("(a) Hourly median shared-minus-reference travel time",loc="left")
    cov.set_title("(b) Hourly share of eligible shared trips with a supported reference",loc="left")
    ax.legend(frameon=False,ncol=2,loc="upper left")
    for a in (ax,cov):
        a.grid(color=GRID)
        a.spines[["top","right"]].set_visible(False)
    fig.suptitle("Hourly differences are modest relative to the variation in reference coverage",weight="bold")
    save(fig,"fig7_travel_time_hourly")


def figure8_spending() -> None:
    rows=read_csv(5,"price_comparison.csv")
    order=[
        next(r for r in rows if r["service"]=="Uber" and r["month"]=="2026-04"),
        next(r for r in rows if r["service"]=="Uber" and r["month"]=="2026-05"),
        next(r for r in rows if r["service"]=="Lyft" and r["month"]=="2026-04"),
        next(r for r in rows if r["service"]=="Lyft" and r["month"]=="2026-05"),
    ]
    fig,ax=plt.subplots(figsize=(11.8,6.2),layout="constrained")
    ypos=np.arange(4)[::-1]
    ax.axvspan(-40,0,facecolor="#eef5f8",zorder=0)
    ax.axvline(0,color=INK,ls="--",lw=1.2)
    for y,row in zip(ypos,order):
        est=float(row["mean_difference_usd"]); lo=float(row["ci95_low_usd"]); hi=float(row["ci95_high_usd"])
        color=SERVICE_COLORS[row["service"]]
        ax.plot([lo,hi],[y,y],color=color,lw=4,solid_capstyle="round")
        ax.scatter(est,y,s=125,color=color,edgecolor="white",lw=1.2,zorder=3)
        label_y = y - 0.22 if y == ypos[0] else y + 0.22
        va = "top" if y == ypos[0] else "bottom"
        ax.text(est,label_y,f"{est:.2f} USD",ha="center",va=va,fontsize=10.3,weight="bold",color=color)
        ax.text(hi+0.9,y,f"95% CI [{lo:.2f}, {hi:.2f}]\nn={int(row['shared_supported']):,}, coverage={100*float(row['shared_reference_coverage']):.1f}%",va="center",fontsize=9.2)
    labels=[f"{r['service']} {'Apr' if r['month'].endswith('04') else 'May'}" for r in order]
    ax.set_yticks(ypos,labels)
    ax.set_xlim(-40,8)
    ax.set_xlabel("Shared - non-shared reference passenger spending (USD)")
    ax.text(0.03,0.94,"Negative side = lower observed spending for reported shared trips",transform=ax.transAxes,ha="left",va="top",fontsize=9.7,weight="bold",color=INK)
    ax.grid(axis="x",color=GRID)
    ax.spines[["top","right","left"]].set_visible(False)
    fig.suptitle("Supported shared trips have lower observed passenger spending than their cell references",weight="bold", y=1.01)
    save(fig,"fig8_spending_difference")


def main() -> None:
    setup_style()
    figure1_dataset_overview()
    demand_figure("2026-04","fig2_spatial_demand_april")
    demand_figure("2026-05","fig3_spatial_demand_may")
    figure4_od_summary()
    figure5_pooling_metrics()
    figure6_travel_time_monthly()
    figure7_travel_time_hourly()
    figure8_spending()
    print(f"Wrote 8 annotated paper figures to {OUT}")


if __name__ == "__main__":
    main()
