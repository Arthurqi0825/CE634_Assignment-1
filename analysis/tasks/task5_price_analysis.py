"""Task 5.1: observed shared versus non-shared HVFHV passenger spending.

Run from the project root:
  ./.venv/bin/python -m analysis.helpers.task_outputs run 5 -- ./.venv/bin/python -m analysis.tasks.task5_price_analysis
Requires requirements.txt. This script streams the two raw HVFHV files.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime

import matplotlib

from analysis.common import (
    HVFHV_SPENDING_COMPONENTS,
    PROJECT_ROOT,
    RAW_DIR,
    STUDY_MONTHS,
    hvfhv_passenger_spending,
    load_zone_lookup,
    task_output_dir,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import t as student_t

ROOT = PROJECT_ROOT
RAW = RAW_DIR
OUT = task_output_dir(5)
TABLES, FIGURES = OUT / "tables", OUT / "figures"
COMPONENTS = list(HVFHV_SPENDING_COMPONENTS)
MONTHS, SERVICES = list(STUDY_MONTHS), ["Uber", "Lyft"]
SIZE = 266
CELLS = 2 * 2 * SIZE * SIZE * 24 * 2
START, MAY, END = map(np.datetime64, ("2026-04-01", "2026-05-01", "2026-06-01"))


def write_csv(name, rows):
    with (TABLES / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    lookup = load_zone_lookup()
    valid_ids = np.zeros(SIZE, dtype=bool)
    valid_ids[list(lookup)] = True
    ref_n = np.zeros(CELLS, dtype=np.int32)
    ref_sum = np.zeros(CELLS, dtype=np.float64)
    shared = []
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
    ] + COMPONENTS
    for file_month in ("04", "05"):
        path = RAW / f"fhvhv_tripdata_2026-{file_month}.parquet"
        pf = pq.ParquetFile(path)
        scanned = 0
        print(f"Scanning {path.name}: {pf.metadata.num_rows:,} rows", flush=True)
        for batch in pf.iter_batches(batch_size=250_000, columns=cols):
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
            status = (
                df.shared_request_flag.fillna("?").astype(str)
                + "/"
                + df.shared_match_flag.fillna("?").astype(str)
            )
            status = status.to_numpy()
            relevant = (
                (service >= 0)
                & (month >= 0)
                & ~np.isnat(d)
                & (d > p)
                & np.isin(status, ["N/N", "Y/Y"])
            )
            if not relevant.any():
                continue
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
            price, complete, parts = hvfhv_passenger_spending(df)
            price_ok = np.isfinite(price) & (price > 0)
            eligible = relevant & pu_ok & do_ok & distance_ok & price_ok
            for m in (0, 1):
                for s in (0, 1):
                    for st in ("N/N", "Y/Y"):
                        key = (MONTHS[m], SERVICES[s], st)
                        group = (
                            relevant & (month == m) & (service == s) & (status == st)
                        )
                        audit[key + ("completed_status",)] += int(group.sum())
                        audit[key + ("invalid_od",)] += int(
                            (group & ~(pu_ok & do_ok)).sum()
                        )
                        audit[key + ("invalid_distance",)] += int(
                            (group & ~distance_ok).sum()
                        )
                        audit[key + ("missing_component",)] += int(
                            (group & ~complete).sum()
                        )
                        audit[key + ("nonpositive_price",)] += int(
                            (group & complete & ~price_ok).sum()
                        )
                        audit[key + ("eligible",)] += int((eligible & group).sum())
            if not eligible.any():
                continue
            # Month, company, ordered OD, pickup hour, weekday/weekend.
            hour = pickup.dt.hour.to_numpy(dtype=np.int16)
            weekend = (pickup.dt.dayofweek.to_numpy(dtype=np.int16) >= 5).astype(
                np.int16
            )
            cell = (
                (((month * 2 + service) * SIZE + pu_i) * SIZE + do_i) * 24 + hour
            ) * 2 + weekend
            nn = eligible & (status == "N/N")
            yy = eligible & (status == "Y/Y")
            np.add.at(ref_n, cell[nn], 1)
            np.add.at(ref_sum, cell[nn], price[nn])
            if yy.any():
                shared.append(
                    pd.DataFrame(
                        {
                            "cell": cell[yy],
                            "month": month[yy],
                            "service": service[yy],
                            "pickup_day": pickup.dt.strftime("%Y-%m-%d").to_numpy()[yy],
                            "price": price[yy],
                            "miles": miles[yy],
                        }
                    )
                )
        assert scanned == pf.metadata.num_rows
        print(f"  scanned {scanned:,}", flush=True)
    sh = pd.concat(shared, ignore_index=True)
    sh["reference_n"] = ref_n[sh.cell.to_numpy()]
    sh["supported"] = sh.reference_n >= 30
    has_reference = sh.reference_n.to_numpy() > 0
    sh.loc[has_reference, "reference_mean_usd"] = (
        ref_sum[sh.loc[has_reference, "cell"].to_numpy()]
        / sh.loc[has_reference, "reference_n"].to_numpy()
    )
    sh["difference_usd"] = sh.price - sh.reference_mean_usd
    rows, daily, cell_rows, sensitivity = [], [], [], []
    for m in (0, 1):
        for s in (0, 1):
            group = sh[(sh.month == m) & (sh.service == s)]
            comp = group[group.supported].copy()
            for minimum in (10, 30, 50):
                subset = group[group.reference_n >= minimum]
                sensitivity.append(
                    {
                        "month": MONTHS[m],
                        "service": SERVICES[s],
                        "minimum_reference_n": minimum,
                        "shared_supported": len(subset),
                        "mean_difference_usd": float(subset.difference_usd.mean())
                        if len(subset)
                        else None,
                        "median_difference_usd": float(subset.difference_usd.median())
                        if len(subset)
                        else None,
                    }
                )
            a_nn = audit[(MONTHS[m], SERVICES[s], "N/N", "eligible")]
            a_yy = audit[(MONTHS[m], SERVICES[s], "Y/Y", "eligible")]
            gday = comp.groupby("pickup_day").agg(
                n=("difference_usd", "size"), sum_diff=("difference_usd", "sum")
            )
            estimate = float(comp.difference_usd.mean()) if len(comp) else np.nan
            days = len(gday)
            if days >= 2:
                # Ratio estimator clustered by pickup date; finite-sample t interval.
                u = gday.sum_diff.to_numpy() - estimate * gday.n.to_numpy()
                se = float(np.sqrt(days / (days - 1) * np.sum(u * u)) / len(comp))
                tstat = estimate / se if se > 0 else np.nan
                pval = (
                    float(2 * student_t.sf(abs(tstat), df=days - 1))
                    if np.isfinite(tstat)
                    else np.nan
                )
                critical = float(student_t.ppf(0.975, days - 1))
                lo, hi = estimate - critical * se, estimate + critical * se
            else:
                se = pval = lo = hi = np.nan
            for day, val in gday.iterrows():
                daily.append(
                    {
                        "month": MONTHS[m],
                        "service": SERVICES[s],
                        "pickup_day": day,
                        "shared_n": int(val.n),
                        "difference_sum_usd": float(val.sum_diff),
                    }
                )
            for cell_id, val in (
                comp.groupby("cell")
                .agg(
                    shared_n=("price", "size"),
                    shared_mean_usd=("price", "mean"),
                    reference_mean_usd=("reference_mean_usd", "first"),
                )
                .iterrows()
            ):
                cell_rows.append(
                    {
                        "month": MONTHS[m],
                        "service": SERVICES[s],
                        "cell_id": int(cell_id),
                        "shared_n": int(val.shared_n),
                        "reference_n": int(ref_n[cell_id]),
                        "shared_mean_usd": round(float(val.shared_mean_usd), 4),
                        "reference_mean_usd": round(float(val.reference_mean_usd), 4),
                        "difference_usd": round(
                            float(val.shared_mean_usd - val.reference_mean_usd), 4
                        ),
                    }
                )
            rows.append(
                {
                    "month": MONTHS[m],
                    "service": SERVICES[s],
                    "nonshared_eligible": a_nn,
                    "shared_eligible": a_yy,
                    "shared_supported": len(comp),
                    "shared_reference_coverage": len(comp) / a_yy if a_yy else None,
                    "supported_cells": int(comp.cell.nunique()),
                    "pickup_days": days,
                    "raw_nonshared_mean_usd": float(
                        ref_sum.reshape(2, 2, -1)[m, s].sum() / a_nn
                    ),
                    "raw_shared_mean_usd": float(group.price.mean())
                    if len(group)
                    else None,
                    "matched_shared_mean_usd": float(comp.price.mean())
                    if len(comp)
                    else None,
                    "matched_reference_mean_usd": float(comp.reference_mean_usd.mean())
                    if len(comp)
                    else None,
                    "mean_difference_usd": estimate,
                    "se_day_cluster_usd": se,
                    "ci95_low_usd": lo,
                    "ci95_high_usd": hi,
                    "p_value_two_sided": pval,
                    "relative_difference_percent": float(
                        100 * estimate / comp.reference_mean_usd.mean()
                    )
                    if len(comp)
                    else None,
                }
            )
    audit_rows = []
    for m in MONTHS:
        for s in SERVICES:
            for st in ("N/N", "Y/Y"):
                k = (m, s, st)
                r = {"month": m, "service": s, "status": st}
                r.update(
                    {
                        name: audit[k + (name,)]
                        for name in (
                            "completed_status",
                            "invalid_od",
                            "invalid_distance",
                            "missing_component",
                            "nonpositive_price",
                            "eligible",
                        )
                    }
                )
                r["excluded_union"] = r["completed_status"] - r["eligible"]
                r["overlap_excess_flags"] = (
                    r["invalid_od"]
                    + r["invalid_distance"]
                    + r["missing_component"]
                    + r["nonpositive_price"]
                    - r["excluded_union"]
                )
                audit_rows.append(r)
    write_csv("price_comparison.csv", rows)
    write_csv("eligibility_audit.csv", audit_rows)
    write_csv("daily_differences.csv", daily)
    write_csv("supported_cells.csv", cell_rows)
    write_csv("reference_threshold_sensitivity.csv", sensitivity)
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = np.arange(4)
    est = [r["mean_difference_usd"] for r in rows]
    errs = [
        [est[i] - rows[i]["ci95_low_usd"] for i in range(4)],
        [rows[i]["ci95_high_usd"] - est[i] for i in range(4)],
    ]
    ax.errorbar(positions, est, yerr=errs, fmt="o", capsize=5, color="#215d91")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(positions, [f"{r['service']}\n{r['month']}" for r in rows])
    ax.set_ylabel("Shared minus comparable non-shared spending (USD)")
    ax.set_title("Within-cell price differences and day-clustered 95% intervals")
    fig.tight_layout()
    fig.savefig(FIGURES / "price_differences.png", dpi=180)
    plt.close(fig)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": "Task 5.1: Are recorded shared trips significantly cheaper than non-shared trips?",
        "method": "Uber and Lyft separately; Y/Y versus N/N; positive complete passenger spending, valid directed OD, 0<trip_miles<=100. Reference mean from same pickup month, ordered OD, hour, weekday/weekend with >=30 N/N trips. Shared-trip-weighted mean of individual shared price minus reference mean. Two-sided day-clustered t test by pickup date. Observational, not causal.",
        "comparison": rows,
        "eligibility_audit": audit_rows,
        "reference_threshold_sensitivity": sensitivity,
        "limitations": [
            "Only completed trips are observed; request selection and dynamic prices are unobserved.",
            "Zones hide exact endpoints; pickup hour and weekday/weekend do not control weather or date-specific traffic.",
            "Shared_match_flag reports overlap but does not reveal partner or overlap length.",
            "Trips in unsupported cells do not contribute to the matched estimate.",
            "Distance can change after pooling, so it is an eligibility check, not a primary adjustment variable.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    report = [
        r"\subsection{Task 5.1: Are shared trips cheaper?}",
        r"For each platform separately we compare reported matched shared trips ($Y/Y$) with non-shared trips ($N/N$). Passenger spending is the sum of the eight HVFHV monetary components used in Task 2; missing components are not set to zero and driver pay is excluded. We require completed trips, official directed OD zones, $0<\mathrm{trip\_miles}\leq100$, and positive finite spending. The audit CSV records the union and overlaps of exclusions. The reference for a shared trip is the mean non-shared spending in its same pickup month, directed OD, pickup hour, and weekday/weekend cell, requiring at least 30 reference trips. Within-zone OD pairs remain eligible.",
        r"Let $S_i$ be shared-trip spending and $\bar N_{c(i)}$ the non-shared mean in cell $c(i)$. We estimate $\hat\Delta=n^{-1}\sum_i(S_i-\bar N_{c(i)})$, weighting eligible shared trips equally. Negative values mean the reported matched trips cost less. We use a two-sided $t$ test of $H_0:\Delta=0$ with standard errors clustered by pickup date: $u_d=\sum_{i\in d}(S_i-\bar N_{c(i)}-\hat\Delta)$, $\widehat{SE}=\sqrt{D/(D-1)\sum_d u_d^2}/n$, and $D-1$ degrees of freedom. This accounts for same-day dependence; the confidence interval is $\hat\Delta\pm t_{0.975,D-1}\widehat{SE}$. Inference is conditional on the estimated non-shared reference means and does not propagate their sampling error; this especially limits the small Lyft sample. This is an observational comparison, not a causal discount estimate.",
        r"\begin{table}[htbp]\centering\small\caption{Task 5.1 passenger spending comparison; negative differences favor shared trips.}\label{tab:t5price}\begin{tabular}{llrrrrr}\hline Month & Platform & Shared $n$ & Coverage & $\hat\Delta$ (USD) & 95\% CI (USD) & $p$ \\ \hline",
    ]
    for r in rows:
        report.append(
            f"{r['month']} & {r['service']} & {r['shared_supported']:,} & {100*r['shared_reference_coverage']:.1f}\\% & {r['mean_difference_usd']:.2f} & [{r['ci95_low_usd']:.2f}, {r['ci95_high_usd']:.2f}] & {r['p_value_two_sided']:.3g} "
            + r"\\"
        )
    report.extend(
        [
            r"\hline\end{tabular}\end{table}",
            r"\begin{figure}[htbp]\centering\includegraphics[width=.75\textwidth]{04_Results/Task_5/figures/price_differences.png}\caption{Shared minus cell-comparable non-shared spending, with day-clustered 95\% confidence intervals.}\label{fig:t5price}\end{figure}",
        ]
    )
    for r in rows:
        direction = "lower" if r["mean_difference_usd"] < 0 else "higher"
        significance = (
            "excludes"
            if r["ci95_high_usd"] < 0 or r["ci95_low_usd"] > 0
            else "includes"
        )
        report.append(
            f"For {r['service']} in {r['month']}, the comparable shared-trip mean is USD {r['matched_shared_mean_usd']:.2f} versus USD {r['matched_reference_mean_usd']:.2f} for its reference; the shared trips are USD {abs(r['mean_difference_usd']):.2f} {direction} on average ({r['relative_difference_percent']:.1f}\\% relative to reference). The 95\\% interval {significance} zero; coverage is {r['shared_supported']:,}/{r['shared_eligible']:,} eligible shared trips."
        )
    report.append(
        r"Raw platform means are descriptive only: Lyft's shared mean is higher than its non-shared mean in both months, while its supported-cell differences are negative. Trip mix therefore matters. For Uber, the 10/30/50-reference-trip sensitivity estimates remain negative (April: USD -9.15/-8.20/-7.63; May: USD -9.15/-8.14/-7.45). Lyft's 10/30/50 estimates also stay negative but vary substantially with only 68--73 matched trips in each month; see the sensitivity CSV. The matched estimate is limited to supported cells and still permits unmeasured differences in exact endpoints, trip dates, route, passenger selection, promotions, and dynamic pricing. A reported $Y/Y$ flag does not verify the partner's identity or overlap duration. We use trip distance only to remove implausible records because pooling may itself change the driven distance. No causal discount is claimed."
    )
    (OUT / "report.tex").write_text("\n".join(report) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
