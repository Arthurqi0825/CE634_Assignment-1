from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from analysis.common import PROJECT_ROOT, RAW_DIR, REFERENCE_DIR

ROOT = PROJECT_ROOT
TRIP_DIR = RAW_DIR
REF_DIR = REFERENCE_DIR
REPORT_DIR = ROOT / "03_Reports"


FILES = [
    (
        "yellow_2026_04",
        "yellow",
        "2026-04",
        TRIP_DIR / "yellow_tripdata_2026-04.parquet",
    ),
    (
        "yellow_2026_05",
        "yellow",
        "2026-05",
        TRIP_DIR / "yellow_tripdata_2026-05.parquet",
    ),
    ("fhvhv_2026_04", "fhvhv", "2026-04", TRIP_DIR / "fhvhv_tripdata_2026-04.parquet"),
    ("fhvhv_2026_05", "fhvhv", "2026-05", TRIP_DIR / "fhvhv_tripdata_2026-05.parquet"),
]


YELLOW_KEYS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "payment_type",
]
FHVHV_KEYS = [
    "hvfhs_license_num",
    "request_datetime",
    "on_scene_datetime",
    "pickup_datetime",
    "dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "cbd_congestion_fee",
    "shared_request_flag",
    "shared_match_flag",
]


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pa.Scalar):
        value = value.as_py()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def int_scalar(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, pa.Scalar):
        value = value.as_py()
    return 0 if value is None else int(value)


def numeric_min_max(arr: pa.Array) -> tuple[Any, Any]:
    clean = pc.drop_null(arr)
    if len(clean) == 0:
        return None, None
    return scalar(pc.min(clean)), scalar(pc.max(clean))


def update_min_max(stats: dict[str, Any], prefix: str, arr: pa.Array) -> None:
    amin, amax = numeric_min_max(arr)
    if amin is not None:
        stats[f"{prefix}_min"] = (
            amin
            if stats.get(f"{prefix}_min") is None
            else min(stats[f"{prefix}_min"], amin)
        )
    if amax is not None:
        stats[f"{prefix}_max"] = (
            amax
            if stats.get(f"{prefix}_max") is None
            else max(stats[f"{prefix}_max"], amax)
        )


def add_counter_from_array(counter: Counter, arr: pa.Array) -> None:
    vc = pc.value_counts(arr).to_pylist()
    for item in vc:
        key = item["values"]
        counter[str(key) if key is not None else "Missing"] += int(item["counts"])


def true_count(mask: pa.Array) -> int:
    return int_scalar(pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())))


def inspect_file(
    label: str, service: str, month: str, path: Path, valid_zone_ids: set[int]
) -> dict[str, Any]:
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    columns = [field.name for field in schema]
    dtypes = {field.name: str(field.type) for field in schema}
    key_cols = YELLOW_KEYS if service == "yellow" else FHVHV_KEYS
    available_keys = [col for col in key_cols if col in columns]

    head = pf.read_row_group(0).slice(0, 5).to_pandas().astype(object)
    head_records = json.loads(head.to_json(orient="records", date_format="iso"))

    stats: dict[str, Any] = {
        "label": label,
        "service": service,
        "month": month,
        "path": str(path.relative_to(ROOT)),
        "file_size_bytes": path.stat().st_size,
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "num_rows": pf.metadata.num_rows,
        "num_row_groups": pf.metadata.num_row_groups,
        "columns": columns,
        "dtypes": dtypes,
        "head": head_records,
        "missing": {col: 0 for col in available_keys},
        "time": {},
        "zones": {
            "PULocationID": {"null": 0, "outside_lookup": 0, "min": None, "max": None},
            "DOLocationID": {"null": 0, "outside_lookup": 0, "min": None, "max": None},
        },
        "anomalies": {},
        "value_counts": {},
    }

    if service == "yellow":
        stats["time"] = {
            "pickup_min": None,
            "pickup_max": None,
            "dropoff_min": None,
            "dropoff_max": None,
            "duration_minutes_min": None,
            "duration_minutes_max": None,
        }
        anomalies = {
            "duration_minutes_le_0": 0,
            "duration_minutes_gt_24h": 0,
            "trip_distance_lt_0": 0,
            "trip_distance_eq_0": 0,
            "trip_distance_gt_100": 0,
            "fare_amount_lt_0": 0,
            "total_amount_lt_0": 0,
            "total_amount_eq_0": 0,
            "total_amount_gt_1000": 0,
        }
        payment_type_counts: Counter = Counter()
    else:
        stats["time"] = {
            "request_min": None,
            "request_max": None,
            "pickup_min": None,
            "pickup_max": None,
            "dropoff_min": None,
            "dropoff_max": None,
            "duration_minutes_min": None,
            "duration_minutes_max": None,
            "trip_time_minutes_min": None,
            "trip_time_minutes_max": None,
        }
        anomalies = {
            "duration_minutes_le_0": 0,
            "duration_minutes_gt_24h": 0,
            "trip_time_seconds_le_0": 0,
            "trip_time_seconds_gt_24h": 0,
            "trip_miles_lt_0": 0,
            "trip_miles_eq_0": 0,
            "trip_miles_gt_100": 0,
            "base_passenger_fare_lt_0": 0,
            "passenger_spending_component_missing": 0,
            "passenger_spending_lt_0": 0,
            "passenger_spending_eq_0": 0,
            "passenger_spending_gt_1000": 0,
        }
        hvfhs_counts: Counter = Counter()
        shared_request_counts: Counter = Counter()
        shared_match_counts: Counter = Counter()
        combo_counts: Counter = Counter()

    valid_zone_arr = pa.array(sorted(valid_zone_ids), type=pa.int64())

    for batch in pf.iter_batches(batch_size=500_000, columns=available_keys):
        tbl = pa.Table.from_batches([batch])
        for col in available_keys:
            stats["missing"][col] += int_scalar(
                pc.sum(pc.cast(pc.is_null(tbl[col]), pa.int64()))
            )

        for zone_col in ["PULocationID", "DOLocationID"]:
            if zone_col in tbl.column_names:
                arr = tbl[zone_col].combine_chunks()
                zstats = stats["zones"][zone_col]
                zstats["null"] += int_scalar(
                    pc.sum(pc.cast(pc.is_null(arr), pa.int64()))
                )
                update_min_max(zstats, "", arr)
                zstats["min"] = zstats.pop("_min", zstats.get("min"))
                zstats["max"] = zstats.pop("_max", zstats.get("max"))
                non_null = pc.drop_null(arr).cast(pa.int64())
                outside = pc.invert(pc.is_in(non_null, value_set=valid_zone_arr))
                zstats["outside_lookup"] += true_count(outside)

        if service == "yellow":
            pickup = tbl["tpep_pickup_datetime"].combine_chunks()
            dropoff = tbl["tpep_dropoff_datetime"].combine_chunks()
            update_min_max(stats["time"], "pickup", pickup)
            update_min_max(stats["time"], "dropoff", dropoff)
            duration_min = pc.divide(
                pc.cast(pc.subtract(dropoff, pickup).cast(pa.int64()), pa.float64()),
                60_000_000,
            )
            update_min_max(stats["time"], "duration_minutes", duration_min)
            anomalies["duration_minutes_le_0"] += true_count(
                pc.less_equal(duration_min, 0)
            )
            anomalies["duration_minutes_gt_24h"] += true_count(
                pc.greater(duration_min, 24 * 60)
            )

            dist = tbl["trip_distance"].combine_chunks()
            anomalies["trip_distance_lt_0"] += true_count(pc.less(dist, 0))
            anomalies["trip_distance_eq_0"] += true_count(pc.equal(dist, 0))
            anomalies["trip_distance_gt_100"] += true_count(pc.greater(dist, 100))

            fare = tbl["fare_amount"].combine_chunks()
            total = tbl["total_amount"].combine_chunks()
            anomalies["fare_amount_lt_0"] += true_count(pc.less(fare, 0))
            anomalies["total_amount_lt_0"] += true_count(pc.less(total, 0))
            anomalies["total_amount_eq_0"] += true_count(pc.equal(total, 0))
            anomalies["total_amount_gt_1000"] += true_count(pc.greater(total, 1000))
            add_counter_from_array(
                payment_type_counts, tbl["payment_type"].combine_chunks()
            )
        else:
            request = tbl["request_datetime"].combine_chunks()
            pickup = tbl["pickup_datetime"].combine_chunks()
            dropoff = tbl["dropoff_datetime"].combine_chunks()
            update_min_max(stats["time"], "request", request)
            update_min_max(stats["time"], "pickup", pickup)
            update_min_max(stats["time"], "dropoff", dropoff)
            duration_min = pc.divide(
                pc.cast(pc.subtract(dropoff, pickup).cast(pa.int64()), pa.float64()),
                60_000_000,
            )
            update_min_max(stats["time"], "duration_minutes", duration_min)
            anomalies["duration_minutes_le_0"] += true_count(
                pc.less_equal(duration_min, 0)
            )
            anomalies["duration_minutes_gt_24h"] += true_count(
                pc.greater(duration_min, 24 * 60)
            )

            trip_time = tbl["trip_time"].combine_chunks()
            trip_time_min = pc.divide(pc.cast(trip_time, pa.float64()), 60)
            update_min_max(stats["time"], "trip_time_minutes", trip_time_min)
            anomalies["trip_time_seconds_le_0"] += true_count(
                pc.less_equal(trip_time, 0)
            )
            anomalies["trip_time_seconds_gt_24h"] += true_count(
                pc.greater(trip_time, 24 * 60 * 60)
            )

            miles = tbl["trip_miles"].combine_chunks()
            anomalies["trip_miles_lt_0"] += true_count(pc.less(miles, 0))
            anomalies["trip_miles_eq_0"] += true_count(pc.equal(miles, 0))
            anomalies["trip_miles_gt_100"] += true_count(pc.greater(miles, 100))
            base_fare = tbl["base_passenger_fare"].combine_chunks()
            anomalies["base_passenger_fare_lt_0"] += true_count(pc.less(base_fare, 0))

            spend_cols = [
                "base_passenger_fare",
                "tolls",
                "bcf",
                "sales_tax",
                "congestion_surcharge",
                "airport_fee",
                "tips",
                "cbd_congestion_fee",
            ]
            component_missing = None
            spending = None
            for col in spend_cols:
                arr = tbl[col].combine_chunks()
                missing = pc.is_null(arr)
                component_missing = (
                    missing
                    if component_missing is None
                    else pc.or_(component_missing, missing)
                )
                spending = arr if spending is None else pc.add(spending, arr)
            anomalies["passenger_spending_component_missing"] += true_count(
                component_missing
            )
            valid_spending = pc.if_else(component_missing, None, spending)
            anomalies["passenger_spending_lt_0"] += true_count(
                pc.less(valid_spending, 0)
            )
            anomalies["passenger_spending_eq_0"] += true_count(
                pc.equal(valid_spending, 0)
            )
            anomalies["passenger_spending_gt_1000"] += true_count(
                pc.greater(valid_spending, 1000)
            )

            hv = tbl["hvfhs_license_num"].combine_chunks()
            req = tbl["shared_request_flag"].combine_chunks()
            match = tbl["shared_match_flag"].combine_chunks()
            add_counter_from_array(hvfhs_counts, hv)
            add_counter_from_array(shared_request_counts, req)
            add_counter_from_array(shared_match_counts, match)
            req_py = req.to_pylist()
            match_py = match.to_pylist()
            for r, m in zip(req_py, match_py):
                combo_counts[
                    f"{r if r is not None else 'Missing'}/{m if m is not None else 'Missing'}"
                ] += 1

    stats["anomalies"] = anomalies
    if service == "yellow":
        stats["value_counts"]["payment_type"] = dict(payment_type_counts.most_common())
    else:
        stats["value_counts"]["hvfhs_license_num"] = dict(hvfhs_counts.most_common())
        stats["value_counts"]["shared_request_flag"] = dict(
            shared_request_counts.most_common()
        )
        stats["value_counts"]["shared_match_flag"] = dict(
            shared_match_counts.most_common()
        )
        stats["value_counts"]["shared_request_match_combo"] = dict(
            combo_counts.most_common()
        )

    for key, value in list(stats["time"].items()):
        stats["time"][key] = scalar(value)
    return stats


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return ""
    return f"{int(value):,}"


def fmt_pct(num: int, den: int) -> str:
    return f"{num / den:.3%}" if den else ""


def records_to_markdown(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    cols = list(records[0].keys())

    def clean(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).replace("\n", " ").replace("|", "\\|")
        return text if len(text) <= 120 else text[:117] + "..."

    out = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in records:
        out.append("| " + " | ".join(clean(row.get(col)) for col in cols) + " |")
    return "\n".join(out)


def make_markdown(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# NYC TLC Assignment Data Initial Inspection")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Downloaded Inventory")
    lines.append("")
    lines.append("| File | Purpose | Size | Rows | Columns |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in results["files"]:
        lines.append(
            f"| `{item['path']}` | {item['service']} {item['month']} | "
            f"{item['file_size_mb']:.2f} MB | {fmt_int(item['num_rows'])} | {len(item['columns'])} |"
        )
    for ref in results["reference_files"]:
        lines.append(
            f"| `{ref['path']}` | reference | {ref['file_size_mb']:.2f} MB |  |  |"
        )
    lines.append("")
    lines.append("## Parquet Schemas")
    for item in results["files"]:
        lines.append(f"### {item['label']}")
        lines.append(
            ", ".join([f"`{c}` ({item['dtypes'][c]})" for c in item["columns"]])
        )
        lines.append("")
    lines.append("## Key Field Missingness")
    lines.append("")
    lines.append("| File | Field | Missing | Share |")
    lines.append("|---|---|---:|---:|")
    for item in results["files"]:
        rows = item["num_rows"]
        for col, miss in item["missing"].items():
            if miss:
                lines.append(
                    f"| {item['label']} | `{col}` | {fmt_int(miss)} | {fmt_pct(miss, rows)} |"
                )
    if not any(miss for item in results["files"] for miss in item["missing"].values()):
        lines.append("| all files | key fields | 0 | 0.000% |")
    lines.append("")
    lines.append("## Time And Zone Checks")
    lines.append("")
    lines.append(
        "| File | Pickup range | Dropoff range | PU zone min-max / outside lookup | DO zone min-max / outside lookup |"
    )
    lines.append("|---|---|---|---|---|")
    for item in results["files"]:
        t = item["time"]
        zpu = item["zones"]["PULocationID"]
        zdo = item["zones"]["DOLocationID"]
        lines.append(
            f"| {item['label']} | {t.get('pickup_min')} to {t.get('pickup_max')} | "
            f"{t.get('dropoff_min')} to {t.get('dropoff_max')} | "
            f"{zpu.get('min')}-{zpu.get('max')} / {fmt_int(zpu.get('outside_lookup'))} | "
            f"{zdo.get('min')}-{zdo.get('max')} / {fmt_int(zdo.get('outside_lookup'))} |"
        )
    lines.append("")
    lines.append("## Obvious Anomaly Counts")
    lines.append("")
    lines.append("| File | Check | Count | Share |")
    lines.append("|---|---|---:|---:|")
    for item in results["files"]:
        rows = item["num_rows"]
        for check, count in item["anomalies"].items():
            if count:
                lines.append(
                    f"| {item['label']} | `{check}` | {fmt_int(count)} | {fmt_pct(count, rows)} |"
                )
    lines.append("")
    lines.append("## HVFHV Flag Distributions")
    for item in results["files"]:
        if item["service"] != "fhvhv":
            continue
        lines.append(f"### {item['label']}")
        for name in [
            "hvfhs_license_num",
            "shared_request_flag",
            "shared_match_flag",
            "shared_request_match_combo",
        ]:
            lines.append(f"**{name}**")
            lines.append("")
            lines.append("| Value | Count | Share |")
            lines.append("|---|---:|---:|")
            for val, count in item["value_counts"][name].items():
                lines.append(
                    f"| `{val}` | {fmt_int(count)} | {fmt_pct(count, item['num_rows'])} |"
                )
            lines.append("")
    lines.append("## Sample Rows")
    for item in results["files"]:
        lines.append(f"### {item['label']}")
        lines.append(records_to_markdown(item["head"]))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lookup = pd.read_csv(REF_DIR / "taxi_zone_lookup.csv")
    valid_zone_ids = set(lookup["LocationID"].dropna().astype(int).tolist())
    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "valid_zone_id_count": len(valid_zone_ids),
        "valid_zone_id_min": min(valid_zone_ids),
        "valid_zone_id_max": max(valid_zone_ids),
        "files": [],
        "reference_files": [],
    }

    for label, service, month, path in FILES:
        results["files"].append(
            inspect_file(label, service, month, path, valid_zone_ids)
        )

    for path in sorted(REF_DIR.iterdir()):
        if path.is_file():
            results["reference_files"].append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "file_size_bytes": path.stat().st_size,
                    "file_size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                }
            )

    json_path = REPORT_DIR / "tlc_initial_inspection.json"
    md_path = REPORT_DIR / "tlc_initial_inspection.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(make_markdown(results), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
