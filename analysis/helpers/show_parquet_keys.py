"""Print every column name and Arrow type from Parquet file metadata.

Examples:
    python -m analysis.helpers.show_parquet_keys
    python -m analysis.helpers.show_parquet_keys 01_Data/Raw/TLC_Trip_Records/yellow_tripdata_2026-04.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from analysis.common import PROJECT_ROOT, RAW_DIR

ROOT = PROJECT_ROOT
DEFAULT_DATA_DIR = RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Parquet files to inspect; defaults to all four raw trip files",
    )
    args = parser.parse_args()
    files = args.files or sorted(DEFAULT_DATA_DIR.glob("*.parquet"))
    if not files:
        parser.error(f"No Parquet files found in {DEFAULT_DATA_DIR}")

    for file in files:
        if not file.is_file():
            parser.error(f"File does not exist: {file}")
        if file.suffix.lower() != ".parquet":
            parser.error(f"Expected a .parquet file: {file}")
        try:
            parquet = pq.ParquetFile(file)
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot read Parquet metadata from {file}: {exc}")
        schema = parquet.schema_arrow
        print(f"\n{file}")
        print(f"Rows: {parquet.metadata.num_rows:,} | Columns: {len(schema)}")
        print(" No.  Key                              Type")
        print("----  -------------------------------  ------------------------------")
        for number, field in enumerate(schema, start=1):
            print(f"{number:>4}  {field.name:<31}  {field.type}")


if __name__ == "__main__":
    main()
