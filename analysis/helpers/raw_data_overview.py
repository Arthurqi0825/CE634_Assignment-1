"""Aggregate uncleaned monthly TLC records for the pre-task dashboard.

This is descriptive exploration. It does not filter anomalous records or produce
the retained samples required by Tasks 1-5.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from analysis.common import PROJECT_ROOT, RAW_DIR, REFERENCE_DIR

ROOT = PROJECT_ROOT
RAW = RAW_DIR
OUT = ROOT / "03_Reports" / "raw_data_overview.json"
INITIAL = ROOT / "03_Reports" / "tlc_initial_inspection.json"
LOOKUP = REFERENCE_DIR / "taxi_zone_lookup.csv"
BATCH_SIZE = 250_000

FILES = [
    ("yellow_2026_04", "yellow", "2026-04", RAW / "yellow_tripdata_2026-04.parquet"),
    ("yellow_2026_05", "yellow", "2026-05", RAW / "yellow_tripdata_2026-05.parquet"),
    ("fhvhv_2026_04", "fhvhv", "2026-04", RAW / "fhvhv_tripdata_2026-04.parquet"),
    ("fhvhv_2026_05", "fhvhv", "2026-05", RAW / "fhvhv_tripdata_2026-05.parquet"),
]
DISTANCE_LABELS = ["≤0", "0–1", "1–3", "3–5", "5–10", "10–20", "20–50", ">50", "缺失"]
FARE_LABELS = ["<0", "=0", "0–10", "10–25", "25–50", "50–100", "100–250", ">250", "缺失"]


def names_by_id() -> dict[int, dict[str, str]]:
    with LOOKUP.open(newline="", encoding="utf-8-sig") as handle:
        return {
            int(row["LocationID"]): {"zone": row["Zone"], "borough": row["Borough"]}
            for row in csv.DictReader(handle)
        }


def numeric_bins(values: np.ndarray, kind: str) -> np.ndarray:
    result = np.zeros(9, dtype=np.int64)
    valid = np.isfinite(values)
    result[-1] = int((~valid).sum())
    x = values[valid]
    if kind == "distance":
        result[:8] = np.bincount(
            np.digitize(x, [0, 1, 3, 5, 10, 20, 50], right=True), minlength=8
        )[:8]
    else:
        result[0] = int((x < 0).sum())
        result[1] = int((x == 0).sum())
        positive = x[x > 0]
        result[2:8] = np.bincount(
            np.digitize(positive, [10, 25, 50, 100, 250], right=True), minlength=6
        )[:6]
    return result


def summarize_file(
    label: str, service: str, month: str, path: Path, lookup: dict, initial: dict
) -> dict:
    pickup_col = "tpep_pickup_datetime" if service == "yellow" else "pickup_datetime"
    distance_col = "trip_distance" if service == "yellow" else "trip_miles"
    amount_col = "total_amount" if service == "yellow" else "base_passenger_fare"
    columns = [pickup_col, "PULocationID", "DOLocationID", distance_col, amount_col]
    parquet = pq.ParquetFile(path)
    hours = np.zeros(24, dtype=np.int64)
    weekdays = np.zeros(7, dtype=np.int64)
    zones = Counter()
    pickup_months = Counter()
    directed_od = np.zeros((266, 266), dtype=np.int64)
    invalid_od = 0
    distance = np.zeros(9, dtype=np.int64)
    amount = np.zeros(9, dtype=np.int64)
    missing_pickup = 0
    outside_file_month = 0
    scanned = 0
    month_start = np.datetime64(month + "-01")
    next_month = np.datetime64("2026-05-01" if month == "2026-04" else "2026-06-01")

    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns):
        frame = batch.to_pandas()
        scanned += len(frame)
        times = frame[pickup_col].to_numpy(dtype="datetime64[ns]")
        valid_times = ~np.isnat(times)
        missing_pickup += int((~valid_times).sum())
        outside_file_month += int(
            (valid_times & ((times < month_start) | (times >= next_month))).sum()
        )
        if valid_times.any():
            selected = times[valid_times]
            hours += np.bincount(
                selected.astype("datetime64[h]").astype("int64") % 24, minlength=24
            )[:24]
            weekdays += np.bincount(
                (selected.astype("datetime64[D]").astype("int64") + 3) % 7, minlength=7
            )[:7]
            values, counts = np.unique(
                selected.astype("datetime64[M]"), return_counts=True
            )
            pickup_months.update(
                {str(value): int(count) for value, count in zip(values, counts)}
            )
        zone_values = frame["PULocationID"].dropna().to_numpy(dtype=np.int64)
        if len(zone_values):
            zone_counts = np.bincount(zone_values)
            zones.update(
                {int(i): int(count) for i, count in enumerate(zone_counts) if count}
            )
        valid_od = frame["PULocationID"].between(1, 265) & frame[
            "DOLocationID"
        ].between(1, 265)
        invalid_od += int((~valid_od).sum())
        if valid_od.any():
            origin = frame.loc[valid_od, "PULocationID"].to_numpy(dtype=np.int64)
            destination = frame.loc[valid_od, "DOLocationID"].to_numpy(dtype=np.int64)
            directed_od += np.bincount(
                origin * 266 + destination, minlength=266 * 266
            ).reshape(266, 266)
        distance += numeric_bins(frame[distance_col].to_numpy(dtype=float), "distance")
        amount += numeric_bins(frame[amount_col].to_numpy(dtype=float), "fare")

    if scanned != parquet.metadata.num_rows:
        raise RuntimeError(
            f"Row-count mismatch for {label}: {scanned} versus {parquet.metadata.num_rows}"
        )
    if int(directed_od.sum()) + invalid_od != scanned:
        raise RuntimeError(f"Directed OD counts do not reconcile for {label}")
    if sum(pickup_months.values()) + missing_pickup != scanned:
        raise RuntimeError(f"Pickup-month counts do not reconcile for {label}")
    top_zones = []
    for zone_id, count in zones.most_common(10):
        zone = lookup.get(zone_id, {"zone": "未在 lookup 中", "borough": "Unknown"})
        top_zones.append(
            {
                "id": zone_id,
                "name": zone["zone"],
                "borough": zone["borough"],
                "count": count,
            }
        )
    boroughs = Counter()
    for zone_id, count in zones.items():
        boroughs[lookup.get(zone_id, {"borough": "Unknown"})["borough"]] += count
    top_directed_od = []
    for flat_index in np.argsort(-directed_od.ravel(), kind="stable")[:10]:
        origin, destination = divmod(int(flat_index), 266)
        count = int(directed_od[origin, destination])
        if not count:
            break
        top_directed_od.append(
            {
                "origin_id": origin,
                "origin_name": lookup.get(origin, {"zone": "Unknown"})["zone"],
                "destination_id": destination,
                "destination_name": lookup.get(destination, {"zone": "Unknown"})[
                    "zone"
                ],
                "count": count,
                "reverse_count": int(directed_od[destination, origin]),
                "within_zone": origin == destination,
            }
        )
    source = initial[label]
    return {
        "label": label,
        "service": service,
        "file_month": month,
        "rows": scanned,
        "hourly_pickups": hours.tolist(),
        "weekday_pickups": weekdays.tolist(),
        "pickup_month_counts": dict(sorted(pickup_months.items())),
        "missing_pickup": missing_pickup,
        "pickup_outside_file_month": outside_file_month,
        "top_pickup_zones": top_zones,
        "pickup_boroughs": dict(boroughs.most_common()),
        "directed_od_top10": top_directed_od,
        "directed_od_valid_trips": int(directed_od.sum()),
        "within_zone_trips": int(np.trace(directed_od)),
        "invalid_od_zone_trips": invalid_od,
        "distance_field": distance_col,
        "distance_bins": dict(zip(DISTANCE_LABELS, map(int, distance))),
        "amount_field": amount_col,
        "amount_bins": dict(zip(FARE_LABELS, map(int, amount))),
        "payment_types": source["value_counts"].get("payment_type", {}),
        "platforms": source["value_counts"].get("hvfhs_license_num", {}),
        "sharing_statuses": source["value_counts"].get(
            "shared_request_match_combo", {}
        ),
    }


def main() -> None:
    lookup = names_by_id()
    initial_data = json.loads(INITIAL.read_text(encoding="utf-8"))
    initial = {item["label"]: item for item in initial_data["files"]}
    summaries = []
    for label, service, month, path in FILES:
        print(f"Scanning {label} ...", flush=True)
        summaries.append(summarize_file(label, service, month, path, lookup, initial))
    by_pickup_month = {"yellow": Counter(), "fhvhv": Counter()}
    for item in summaries:
        by_pickup_month[item["service"]].update(item["pickup_month_counts"])
    output = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "Four uncleaned local Parquet files; platform, sharing and payment categories from tlc_initial_inspection.json",
        "method": "Full-file batch scan; all rows retained, including out-of-month and implausible values. Pickup timestamps drive month, hour and weekday charts. OD keys are ordered (origin,destination); reverse pairs differ and same-zone pairs remain. The OD preview includes records with valid lookup-range IDs and counts invalid OD IDs separately. Monetary histograms are univariate and must not be compared as equivalent passenger spending.",
        "pickup_month_counts": {
            service: dict(sorted(counts.items()))
            for service, counts in by_pickup_month.items()
        },
        "files": summaries,
    }
    OUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
