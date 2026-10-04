"""Shared paths, TLC definitions, and field-level helpers for Tasks 1--5.

The raw data directory may be overridden with ``ASSIGNMENT_RAW_DIR`` for a
read-only reproducibility check. By default it is inside this package.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(
    os.environ.get("ASSIGNMENT_RAW_DIR", PROJECT_ROOT / "01_Data/Raw/TLC_Trip_Records")
).resolve()
REFERENCE_DIR = PROJECT_ROOT / "01_Data/Reference"
RESULTS_DIR = PROJECT_ROOT / "04_Results"
STUDY_MONTHS = ("2026-04", "2026-05")
HVFHV_LICENSES = (
    ("HV0002", "Juno"),
    ("HV0003", "Uber"),
    ("HV0004", "Via"),
    ("HV0005", "Lyft"),
)
SHARING_STATUSES = ("N/N", "Y/N", "Y/Y", "N/Y", "Missing/other")
HVFHV_SPENDING_COMPONENTS = (
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "tips",
    "cbd_congestion_fee",
)
RAW_FILENAMES = (
    "yellow_tripdata_2026-04.parquet",
    "yellow_tripdata_2026-05.parquet",
    "fhvhv_tripdata_2026-04.parquet",
    "fhvhv_tripdata_2026-05.parquet",
)


def task_output_dir(task: int) -> Path:
    """Return the output directory selected by the task runner, if any."""
    return Path(
        os.environ.get("ASSIGNMENT_TASK_OUTPUT_DIR", RESULTS_DIR / f"Task_{task}")
    )


def display_path(path: Path) -> str:
    """Show project-relative paths when possible, including for dry-run logs."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def expected_raw_files() -> dict[str, Path]:
    """Map the four required filenames to their expected locations."""
    return {name: RAW_DIR / name for name in RAW_FILENAMES}


def missing_raw_files() -> list[Path]:
    """List absent inputs without opening or modifying any data file."""
    return [path for path in expected_raw_files().values() if not path.is_file()]


def load_zone_lookup() -> dict[int, dict[str, str]]:
    """Load the official Taxi Zone lookup, retaining 264/265 without polygons."""
    path = REFERENCE_DIR / "taxi_zone_lookup.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {int(row["LocationID"]): row for row in csv.DictReader(handle)}


def official_zone_ids() -> set[int]:
    """Return IDs valid for counting even when a polygon is unavailable."""
    return set(load_zone_lookup())


def numeric_array(series: pd.Series) -> np.ndarray:
    """Convert a numeric field to float, marking unparsable values as NaN."""
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def hvfhv_passenger_spending(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sum, all-components-present mask, component array).

    The eight TLC passenger components are required for a valid sum. Missing
    values remain NaN; driver_pay is deliberately excluded. The function does
    not decide whether a negative or zero total is valid for a later analysis.
    """
    components = (
        frame[list(HVFHV_SPENDING_COMPONENTS)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    complete = np.isfinite(components).all(axis=1)
    spending = np.full(len(frame), np.nan, dtype=float)
    spending[complete] = components[complete].sum(axis=1)
    return spending, complete, components
