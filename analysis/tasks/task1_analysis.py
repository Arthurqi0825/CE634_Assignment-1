"""Task 1.1 audit and analysis-specific cleaning; Task 1.2 monthly counts.

Run through task_outputs.py so stdout, metadata, and results stay together:
  python -m analysis.helpers.task_outputs run 1 -- python -m analysis.tasks.task1_analysis
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.common import (
    HVFHV_LICENSES,
    HVFHV_SPENDING_COMPONENTS,
    PROJECT_ROOT,
    RAW_DIR,
    REFERENCE_DIR,
    display_path,
    hvfhv_passenger_spending,
    numeric_array,
    official_zone_ids,
    task_output_dir,
)

ROOT = PROJECT_ROOT
RAW = RAW_DIR
LOOKUP = REFERENCE_DIR / "taxi_zone_lookup.csv"
OUT = task_output_dir(1)
TABLES = OUT / "tables"
BATCH_SIZE = 250_000
START = pd.Timestamp("2026-04-01")
END = pd.Timestamp("2026-06-01")
MONTHS = {
    "2026-04": (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01")),
    "2026-05": (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01")),
}
SPEND_COMPONENTS = list(HVFHV_SPENDING_COMPONENTS)
PLATFORMS = HVFHV_LICENSES
FILES = [
    ("yellow_2026_04", "yellow", "2026-04", RAW / "yellow_tripdata_2026-04.parquet"),
    ("yellow_2026_05", "yellow", "2026-05", RAW / "yellow_tripdata_2026-05.parquet"),
    ("fhvhv_2026_04", "fhvhv", "2026-04", RAW / "fhvhv_tripdata_2026-04.parquet"),
    ("fhvhv_2026_05", "fhvhv", "2026-05", RAW / "fhvhv_tripdata_2026-05.parquet"),
]
RULES = {
    "pickup_missing": "Pickup timestamp missing",
    "dropoff_missing": "Dropoff timestamp missing",
    "pickup_outside_study": "Pickup outside April–May 2026",
    "pickup_file_month_mismatch": "Pickup month differs from source filename month",
    "duration_nonpositive": "Dropoff is at/before pickup",
    "duration_gt24h": "Timestamp duration exceeds 24 hours",
    "pu_zone_missing": "Pickup zone ID missing",
    "pu_zone_invalid": "Pickup zone ID outside official lookup",
    "do_zone_missing": "Dropoff zone ID missing",
    "do_zone_invalid": "Dropoff zone ID outside official lookup",
    "distance_missing": "Distance missing or nonfinite",
    "distance_nonpositive": "Distance is zero or negative",
    "distance_gt100mi": "Distance exceeds 100 miles",
    "payment_missing": "Passenger spending missing or nonfinite",
    "payment_negative": "Passenger spending negative",
    "payment_gt1000": "Passenger spending exceeds $1,000",
    "fare_base_negative": "Base fare negative",
    "request_missing": "HVFHV request timestamp missing",
    "request_after_pickup": "HVFHV request after pickup",
    "on_scene_order_invalid": "HVFHV on-scene time outside request-to-pickup order",
    "trip_time_missing": "HVFHV trip_time missing or nonfinite",
    "trip_time_nonpositive": "HVFHV trip_time nonpositive",
    "trip_time_gt24h": "HVFHV trip_time exceeds 24 hours",
    "shared_flag_other": "HVFHV request/match flag missing or outside Y/N",
    "shared_ny": "HVFHV request N / match Y",
    "company_other": "HVFHV company code outside HV0002/HV0003/HV0004/HV0005",
    "payment_type_voided": "Yellow payment_type 6 (voided)",
}
RULE_KEYS = list(RULES)
RULE_BITS = {name: 1 << i for i, name in enumerate(RULE_KEYS)}
SAMPLE_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "request_datetime",
    "on_scene_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "trip_miles",
    "trip_time",
    "total_amount",
    "base_passenger_fare",
    "hvfhs_license_num",
    "shared_request_flag",
    "shared_match_flag",
    "payment_type",
]


def evaluate(
    frame: pd.DataFrame, service: str, file_month: str, zones: set[int]
) -> tuple[dict, dict, dict]:
    n = len(frame)
    false = np.zeros(n, dtype=bool)
    flags = {name: false.copy() for name in RULE_KEYS}
    pickup_key = "tpep_pickup_datetime" if service == "yellow" else "pickup_datetime"
    dropoff_key = "tpep_dropoff_datetime" if service == "yellow" else "dropoff_datetime"
    pickup = pd.to_datetime(frame[pickup_key], errors="coerce")
    dropoff = pd.to_datetime(frame[dropoff_key], errors="coerce")
    duration_seconds = (dropoff - pickup).dt.total_seconds().to_numpy(dtype=float)
    flags["pickup_missing"] = pickup.isna().to_numpy()
    flags["dropoff_missing"] = dropoff.isna().to_numpy()
    flags["pickup_outside_study"] = (
        pickup.notna() & ((pickup < START) | (pickup >= END))
    ).to_numpy()
    file_start, file_end = MONTHS[file_month]
    flags["pickup_file_month_mismatch"] = (
        pickup.notna() & ((pickup < file_start) | (pickup >= file_end))
    ).to_numpy()
    flags["duration_nonpositive"] = np.isfinite(duration_seconds) & (
        duration_seconds <= 0
    )
    flags["duration_gt24h"] = np.isfinite(duration_seconds) & (duration_seconds > 86400)

    pu = frame["PULocationID"]
    do = frame["DOLocationID"]
    flags["pu_zone_missing"] = pu.isna().to_numpy()
    flags["do_zone_missing"] = do.isna().to_numpy()
    flags["pu_zone_invalid"] = (pu.notna() & ~pu.isin(zones)).to_numpy()
    flags["do_zone_invalid"] = (do.notna() & ~do.isin(zones)).to_numpy()
    pu_ok = ~(flags["pu_zone_missing"] | flags["pu_zone_invalid"])
    do_ok = ~(flags["do_zone_missing"] | flags["do_zone_invalid"])

    distance = numeric_array(
        frame["trip_distance" if service == "yellow" else "trip_miles"]
    )
    flags["distance_missing"] = ~np.isfinite(distance)
    flags["distance_nonpositive"] = np.isfinite(distance) & (distance <= 0)
    flags["distance_gt100mi"] = np.isfinite(distance) & (distance > 100)

    if service == "yellow":
        spending = numeric_array(frame["total_amount"])
        fare_base = numeric_array(frame["fare_amount"])
        flags["payment_type_voided"] = (
            frame["payment_type"].eq(6).fillna(False).to_numpy(dtype=bool)
        )
    else:
        spending, complete, components = hvfhv_passenger_spending(frame)
        fare_base = numeric_array(frame["base_passenger_fare"])
        request = pd.to_datetime(frame["request_datetime"], errors="coerce")
        on_scene = pd.to_datetime(frame["on_scene_datetime"], errors="coerce")
        flags["request_missing"] = request.isna().to_numpy()
        flags["request_after_pickup"] = (
            request.notna() & pickup.notna() & (request > pickup)
        ).to_numpy()
        flags["on_scene_order_invalid"] = (
            (on_scene.notna() & request.notna() & (on_scene < request))
            | (on_scene.notna() & pickup.notna() & (on_scene > pickup))
        ).to_numpy()
        trip_time = numeric_array(frame["trip_time"])
        flags["trip_time_missing"] = ~np.isfinite(trip_time)
        flags["trip_time_nonpositive"] = np.isfinite(trip_time) & (trip_time <= 0)
        flags["trip_time_gt24h"] = np.isfinite(trip_time) & (trip_time > 86400)
        req = frame["shared_request_flag"].fillna("Missing").astype(str)
        match = frame["shared_match_flag"].fillna("Missing").astype(str)
        valid_flags = (
            req.isin(["Y", "N"]).to_numpy() & match.isin(["Y", "N"]).to_numpy()
        )
        flags["shared_flag_other"] = ~valid_flags
        flags["shared_ny"] = (req.eq("N") & match.eq("Y")).to_numpy()
        company = frame["hvfhs_license_num"].fillna("Missing").astype(str)
        flags["company_other"] = ~company.isin(
            [code for code, _ in PLATFORMS]
        ).to_numpy()

    flags["payment_missing"] = ~np.isfinite(spending)
    flags["payment_negative"] = np.isfinite(spending) & (spending < 0)
    flags["payment_gt1000"] = np.isfinite(spending) & (spending > 1000)
    flags["fare_base_negative"] = np.isfinite(fare_base) & (fare_base < 0)

    core = ~(
        flags["pickup_missing"]
        | flags["dropoff_missing"]
        | flags["pickup_outside_study"]
        | flags["duration_nonpositive"]
    )
    od = core & pu_ok & do_ok
    payment_ok = ~(flags["payment_missing"] | flags["payment_negative"])
    distance_ok = ~(
        flags["distance_missing"]
        | flags["distance_nonpositive"]
        | flags["distance_gt100mi"]
    )
    masks = {
        "completed_trip_count": core,
        "pickup_hotspot": core & pu_ok,
        "dropoff_hotspot": core & do_ok,
        "directed_od_volume": od,
        "directed_od_spending": od & payment_ok,
        "distance_based": od & distance_ok,
    }
    if service == "fhvhv":
        company = frame["hvfhs_license_num"].fillna("Missing").astype(str)
        uber = company.eq("HV0003").to_numpy()
        uber_lyft = company.isin(["HV0003", "HV0005"]).to_numpy()
        req = frame["shared_request_flag"].fillna("Missing").astype(str)
        match = frame["shared_match_flag"].fillna("Missing").astype(str)
        status = (req + "/" + match).to_numpy()
        valid_flags = ~flags["shared_flag_other"]
        duration_ok = ~(
            flags["trip_time_missing"]
            | flags["trip_time_nonpositive"]
            | flags["trip_time_gt24h"]
        )
        masks.update(
            {
                "sharing_rate": core & uber_lyft & valid_flags,
                "sharing_zone_rate": core & uber_lyft & valid_flags & pu_ok,
                "uber_duration_reference": od & uber & (status == "N/N") & duration_ok,
                "uber_duration_shared": od & uber & (status == "Y/Y") & duration_ok,
            }
        )
    else:
        status = None
    extras = {"pickup": pickup, "status": status, "spending": spending}
    return flags, masks, extras


def write_csv(name: str, columns: list[str], rows: list[dict]) -> None:
    path = TABLES / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {display_path(path)} ({len(rows)} rows)", flush=True)


def overlap_rows(
    file_name: str, combos: Counter, raw_rows: int
) -> tuple[list[dict], dict]:
    pairs = Counter()
    any_flag = 0
    multiple = 0
    for code, count in combos.items():
        if code == 0:
            continue
        any_flag += count
        bits = [name for i, name in enumerate(RULE_KEYS) if code & (1 << i)]
        if len(bits) >= 2:
            multiple += count
            for left, right in combinations(bits, 2):
                pairs[(left, right)] += count
    rows = [
        {
            "source_file": file_name,
            "rule_a": left,
            "rule_b": right,
            "overlap_records": count,
            "share_of_file": count / raw_rows,
        }
        for (left, right), count in sorted(pairs.items())
        if count
    ]
    return (
        rows,
        {
            "any_flag": any_flag,
            "two_or_more_flags": multiple,
            "no_flags": raw_rows - any_flag,
        },
    )


def sample_value(value) -> str:
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    zones = official_zone_ids()
    file_results = []
    category_counts = Counter()
    raw_month_counts = Counter()
    retained = Counter()
    rules_all = Counter()
    combos_all = Counter()
    samples: dict[str, list[dict]] = {name: [] for name in RULE_KEYS}
    overlap_csv = []
    rule_csv = []

    for label, service, file_month, path in FILES:
        print(f"Auditing {label}: {path.name}", flush=True)
        parquet = pq.ParquetFile(path)
        columns = (
            [
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "fare_amount",
                "total_amount",
                "payment_type",
            ]
            if service == "yellow"
            else [
                "hvfhs_license_num",
                "request_datetime",
                "on_scene_datetime",
                "pickup_datetime",
                "dropoff_datetime",
                "PULocationID",
                "DOLocationID",
                "trip_miles",
                "trip_time",
                "shared_request_flag",
                "shared_match_flag",
            ]
            + SPEND_COMPONENTS
        )
        source_rules = Counter()
        source_combos = Counter()
        source_core = 0
        scanned = 0
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns):
            frame = batch.to_pandas()
            n = len(frame)
            flags, masks, extras = evaluate(frame, service, file_month, zones)
            bits = np.zeros(n, dtype=np.uint32)
            for i, name in enumerate(RULE_KEYS):
                selected = flags[name]
                count = int(np.count_nonzero(selected))
                source_rules[name] += count
                rules_all[name] += count
                if count:
                    bits |= selected.astype(np.uint32) * np.uint32(1 << i)
                    if len(samples[name]) < 3:
                        for row_index in np.flatnonzero(selected)[
                            : 3 - len(samples[name])
                        ]:
                            row = {
                                "rule": name,
                                "source_file": label,
                                "source_row_number": scanned + int(row_index) + 1,
                            }
                            row["completed_trip_count_retained"] = bool(
                                masks["completed_trip_count"][row_index]
                            )
                            row["directed_od_spending_retained"] = bool(
                                masks["directed_od_spending"][row_index]
                            )
                            for column in SAMPLE_COLUMNS:
                                row[column] = (
                                    sample_value(frame.iloc[row_index][column])
                                    if column in frame.columns
                                    else ""
                                )
                            samples[name].append(row)
            values, counts = np.unique(bits, return_counts=True)
            source_combos.update(
                {int(value): int(count) for value, count in zip(values, counts)}
            )
            combos_all.update(
                {int(value): int(count) for value, count in zip(values, counts)}
            )
            source_core += int(np.count_nonzero(masks["completed_trip_count"]))

            pickup = extras["pickup"]
            for month, (start, end) in MONTHS.items():
                month_mask = ((pickup >= start) & (pickup < end)).to_numpy()
                if not month_mask.any():
                    continue
                raw_month_counts[(service, month)] += int(np.count_nonzero(month_mask))
                for analysis, mask in masks.items():
                    retained[(service, month, analysis)] += int(
                        np.count_nonzero(month_mask & mask)
                    )
                base_month = month_mask & masks["completed_trip_count"]
                if service == "yellow":
                    category_counts[
                        (month, "Yellow taxi", "All retained trips")
                    ] += int(np.count_nonzero(base_month))
                else:
                    category_counts[(month, "HVFHV", "All retained trips")] += int(
                        np.count_nonzero(base_month)
                    )
                    if base_month.any():
                        company = (
                            frame.loc[base_month, "hvfhs_license_num"]
                            .fillna("Missing")
                            .astype(str)
                        )
                        req = (
                            frame.loc[base_month, "shared_request_flag"]
                            .fillna("Missing")
                            .astype(str)
                        )
                        match = (
                            frame.loc[base_month, "shared_match_flag"]
                            .fillna("Missing")
                            .astype(str)
                        )
                        status = req + "/" + match
                        status = status.where(
                            req.isin(["Y", "N"]) & match.isin(["Y", "N"]),
                            "Missing/other",
                        )
                        grouped = pd.DataFrame(
                            {"company": company, "status": status}
                        ).value_counts()
                        for (company_code, category), count in grouped.items():
                            category_counts[(month, company_code, category)] += int(
                                count
                            )
            scanned += n
        if scanned != parquet.metadata.num_rows:
            raise RuntimeError(
                f"Source row count mismatch: {label}: {scanned} vs {parquet.metadata.num_rows}"
            )
        overlap_rows_file, overlap_stats = overlap_rows(label, source_combos, scanned)
        overlap_csv.extend(overlap_rows_file)
        for name in RULE_KEYS:
            rule_csv.append(
                {
                    "source_file": label,
                    "service": service,
                    "rule": name,
                    "description": RULES[name],
                    "affected_records": source_rules[name],
                    "raw_records": scanned,
                    "share_of_raw": source_rules[name] / scanned,
                }
            )
        file_results.append(
            {
                "source_file": label,
                "service": service,
                "filename_month": file_month,
                "raw_records": scanned,
                "completed_trip_count_retained": source_core,
                "rule_counts": dict(source_rules),
                "overlap": overlap_stats,
            }
        )
        print(
            f"  rows={scanned:,}; completed-trip mask retained={source_core:,}; multiple flags={overlap_stats['two_or_more_flags']:,}",
            flush=True,
        )

    total_raw = sum(item["raw_records"] for item in file_results)
    overall_overlaps, overall_overlap_stats = overlap_rows(
        "TOTAL", combos_all, total_raw
    )
    overlap_csv.extend(overall_overlaps)
    for name in RULE_KEYS:
        rule_csv.append(
            {
                "source_file": "TOTAL",
                "service": "all",
                "rule": name,
                "description": RULES[name],
                "affected_records": rules_all[name],
                "raw_records": total_raw,
                "share_of_raw": rules_all[name] / total_raw,
            }
        )

    retention_rows = []
    for month in MONTHS:
        for service in ("yellow", "fhvhv"):
            raw = raw_month_counts[(service, month)]
            relevant = [
                "completed_trip_count",
                "pickup_hotspot",
                "dropoff_hotspot",
                "directed_od_volume",
                "directed_od_spending",
                "distance_based",
            ]
            if service == "fhvhv":
                relevant += [
                    "sharing_rate",
                    "sharing_zone_rate",
                    "uber_duration_reference",
                    "uber_duration_shared",
                ]
            for analysis in relevant:
                kept = retained[(service, month, analysis)]
                retention_rows.append(
                    {
                        "pickup_month": month,
                        "service": service,
                        "analysis": analysis,
                        "raw_pickup_records": raw,
                        "retained_records": kept,
                        "excluded_records": raw - kept,
                        "retained_share": kept / raw if raw else None,
                    }
                )

    status_order = ["N/N", "Y/N", "Y/Y", "N/Y", "Missing/other"]
    count_rows = []
    for service_name, category in [
        ("Yellow taxi", "All retained trips"),
        ("HVFHV", "All retained trips"),
    ]:
        count_rows.append(
            {
                "license_num": "",
                "service": service_name,
                "request_match_category": category,
                "april_trips": category_counts[("2026-04", service_name, category)],
                "may_trips": category_counts[("2026-05", service_name, category)],
            }
        )
    for company_code, service_name in PLATFORMS:
        for category in status_order:
            count_rows.append(
                {
                    "license_num": company_code,
                    "service": service_name,
                    "request_match_category": category,
                    "april_trips": category_counts[("2026-04", company_code, category)],
                    "may_trips": category_counts[("2026-05", company_code, category)],
                }
            )
        count_rows.append(
            {
                "license_num": company_code,
                "service": service_name,
                "request_match_category": "All retained trips",
                "april_trips": sum(
                    category_counts[("2026-04", company_code, category)]
                    for category in status_order
                ),
                "may_trips": sum(
                    category_counts[("2026-05", company_code, category)]
                    for category in status_order
                ),
            }
        )
    other_codes = sorted(
        {
            key[1]
            for key in category_counts
            if key[1] not in ("Yellow taxi", "HVFHV")
            and key[1] not in {code for code, _ in PLATFORMS}
        }
    )
    for code in other_codes:
        count_rows.append(
            {
                "license_num": code,
                "service": "Other HVFHV",
                "request_match_category": "All retained trips",
                "april_trips": sum(
                    category_counts[("2026-04", code, status)]
                    for status in status_order
                ),
                "may_trips": sum(
                    category_counts[("2026-05", code, status)]
                    for status in status_order
                ),
            }
        )

    for month, column in [("2026-04", "april_trips"), ("2026-05", "may_trips")]:
        hv_total = category_counts[(month, "HVFHV", "All retained trips")]
        company_total = sum(
            row[column]
            for row in count_rows
            if row["license_num"]
            and row["request_match_category"] == "All retained trips"
        )
        if hv_total != company_total:
            raise RuntimeError(
                f"Company/status counts do not reconcile for {month}: {company_total} vs {hv_total}"
            )
        yellow_total = category_counts[(month, "Yellow taxi", "All retained trips")]
        if yellow_total != retained[("yellow", month, "completed_trip_count")]:
            raise RuntimeError(f"Yellow counts do not reconcile for {month}")
        if hv_total != retained[("fhvhv", month, "completed_trip_count")]:
            raise RuntimeError(f"HVFHV counts do not reconcile for {month}")

    sample_rows = [row for rows in samples.values() for row in rows]
    write_csv(
        "task1_2_counts.csv",
        [
            "license_num",
            "service",
            "request_match_category",
            "april_trips",
            "may_trips",
        ],
        count_rows,
    )
    write_csv(
        "audit_rules.csv",
        [
            "source_file",
            "service",
            "rule",
            "description",
            "affected_records",
            "raw_records",
            "share_of_raw",
        ],
        rule_csv,
    )
    write_csv(
        "rule_overlap.csv",
        ["source_file", "rule_a", "rule_b", "overlap_records", "share_of_file"],
        overlap_csv,
    )
    write_csv(
        "retention_by_analysis.csv",
        [
            "pickup_month",
            "service",
            "analysis",
            "raw_pickup_records",
            "retained_records",
            "excluded_records",
            "retained_share",
        ],
        retention_rows,
    )
    write_csv(
        "anomaly_examples.csv",
        [
            "rule",
            "source_file",
            "source_row_number",
            "completed_trip_count_retained",
            "directed_od_spending_retained",
        ]
        + SAMPLE_COLUMNS,
        sample_rows,
    )

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "Four original April/May 2026 TLC Parquet files; official zone lookup and data dictionaries",
        "raw_records": total_raw,
        "rules": RULES,
        "files": file_results,
        "all_rule_counts": dict(rules_all),
        "overlap": overall_overlap_stats,
        "top_rule_pairs": sorted(
            overall_overlaps, key=lambda row: row["overlap_records"], reverse=True
        )[:20],
        "raw_pickup_month_counts": {
            f"{service}|{month}": raw_month_counts[(service, month)]
            for service in ("yellow", "fhvhv")
            for month in MONTHS
        },
        "retention_by_analysis": retention_rows,
        "task1_2_counts": count_rows,
        "anomaly_examples": {name: rows for name, rows in samples.items() if rows},
        "reported_company_codes": {code: name for code, name in PLATFORMS},
        "other_company_codes": other_codes,
        "notes": [
            "Monthly assignment uses each row's pickup timestamp across source files.",
            "Counts represent recorded completed trips, not unique people, vehicles, or all requests.",
            "Completed-trip count excludes only missing/inconsistent pickup/dropoff time and out-of-study pickup.",
            "Zone, distance, payment, and sharing flags are applied only where those fields affect an analysis.",
            "N/Y is retained as a required reporting-consistency category; other flag statuses remain in Task 1.2 but not sharing-rate denominators.",
            "Negative or missing passenger spending is excluded from spending OD analysis; high amounts >$1,000 are reviewed but not blanket-excluded.",
            "No raw Parquet file is modified or copied to the result package.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def fmt(value: int) -> str:
        return f"{value:,}"

    lines = [
        r"\section{Task 1: Data audit and retained trip counts}",
        r"\label{sec:task1}",
        "We audited all "
        + fmt(total_raw)
        + " recorded trip rows from the four TLC files using the official data dictionaries and zone lookup. "
        "Pickup timestamps determine month membership across source files. The completed-trip count set requires pickup and dropoff timestamps, "
        "pickup in April--May 2026, and a positive timestamp duration. Zone, distance, payment, and sharing rules are applied only to analyses that require those fields. "
        "This avoids treating missing monetary components as zero or removing valid trip records from a count solely because a non-count field is anomalous.",
        "",
        "The audit found "
        + fmt(overall_overlap_stats["any_flag"])
        + " records with at least one reported flag and "
        + fmt(overall_overlap_stats["two_or_more_flags"])
        + " with two or more flags. These are diagnostic flags rather than a universal exclusion count. "
        "Detailed rule counts, pairwise overlaps, retained samples, and example records are supplied in the Task 1 CSV supplements. "
        "Extreme distance above 100 miles is retained in trip counts and OD volumes but excluded from distance-dependent analyses. "
        "Passenger spending above \\$1,000 is reviewed but not automatically removed from spending summaries.",
        "",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Selected audit flags across all four raw files. Counts overlap and are not additive.}",
        r"\begin{tabular}{lr}",
        r"\hline",
        r"Audit flag & Records \\",
        r"\hline",
    ]
    for name in (
        "on_scene_order_invalid",
        "request_after_pickup",
        "distance_nonpositive",
        "duration_nonpositive",
        "payment_negative",
        "distance_gt100mi",
        "shared_ny",
    ):
        lines.append(
            f"{latex_escape(name.replace('_', ' '))} & {fmt(rules_all[name])} " + r"\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
        "The request-after-pickup and on-scene-order flags overlap for "
        + fmt(
            next(
                row["overlap_records"]
                for row in overall_overlaps
                if {row["rule_a"], row["rule_b"]}
                == {"request_after_pickup", "on_scene_order_invalid"}
            )
        )
        + " records. For Task 1.2, "
        + fmt(
            sum(
                row["retained_records"]
                for row in retention_rows
                if row["analysis"] == "completed_trip_count"
            )
        )
        + " trips are retained. Field-specific retention for later analyses is reported separately in the CSV supplement.",
        "",
        r"\begin{table}[htbp]",
        r"\centering\small",
        r"\caption{Retained recorded trips by all four named HVFHV licenses and request/match status. Flags are ordered request first, match second; zeros are explicit.}",
        r"\begin{tabular}{lllrr}",
        r"\hline",
        r"License & Service & Request/match & April & May \\",
        r"\hline",
    ]
    for row in count_rows:
        lines.append(
            f"{latex_escape(row['license_num'] or '-')} & {latex_escape(row['service'])} & {latex_escape(row['request_match_category'])} & {fmt(row['april_trips'])} & {fmt(row['may_trips'])} "
            + r"\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
        "For Juno (HV0002), Uber (HV0003), Via (HV0004), and Lyft (HV0005), each company subtotal equals its five status rows. The four subtotals, plus any separately reported other company codes, reconcile exactly to the HVFHV retained total in each month. Juno and Via have zero retained records in both study months. "
        "The N/Y status is retained as a reporting-inconsistency category, while missing or other flag statuses are listed separately. "
        "Counts refer to records of observed completed trips, not distinct people, vehicles, or all travel requests. "
        "Yellow taxi has no specified sharing status, so it is not subdivided by sharing flags.",
    ]
    (OUT / "report.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "Task 1 summary and report.tex written; category totals reconciled.", flush=True
    )


if __name__ == "__main__":
    main()
