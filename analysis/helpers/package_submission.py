"""Create and verify a ZIP of this standalone submission folder.

Run from the package root:
    ./.venv/bin/python -m analysis.helpers.package_submission
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from analysis.common import PROJECT_ROOT

ARCHIVE_DIR = PROJECT_ROOT / "dist"
ARCHIVE = ARCHIVE_DIR / f"{PROJECT_ROOT.name}.zip"
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "Raw", "dist",
    ".codex", ".agents", "instruction", "instructions", "prompts",
    ".pytest_cache", ".mypy_cache",
}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".fdb_latexmk",
    ".fls",
    ".out",
    ".parquet",
    ".pyc",
    ".stdout",
    ".synctex.gz",
    ".toc",
}


def should_include(path: Path) -> bool:
    """Keep review evidence while excluding raw data and local build state."""
    relative = path.relative_to(PROJECT_ROOT)
    if any(part in EXCLUDED_DIRS for part in relative.parts[:-1]):
        return False
    if path.name in {".DS_Store"} or path.name.endswith("~"):
        return False
    if "prompt" in path.name.lower() and path.suffix.lower() == ".md":
        return False
    if path.name.endswith(tuple(EXCLUDED_SUFFIXES)):
        return False
    if path.suffix == ".log" and path.name != "run.log":
        return False
    return True


def required_files() -> list[Path]:
    """Return essential files that must be present before packaging."""
    paths = [
        "README.md",
        "REVIEW.md",
        "index.html",
        "assets/site.css",
        "assets/site.js",
        ".nojekyll",
        "requirements.txt",
        "main.tex",
        "main.pdf",
        "manifest.json",
        "visualizations/index.html",
        "visualizations/map_atlas.tex",
        "visualizations/PLOT_DATA.md",
        "analysis/helpers/render_all_figures.py",
        "analysis/helpers/report_figures.py",
        "01_Data/Reference/data_dictionary_trip_records_hvfhs.pdf",
        "01_Data/Reference/data_dictionary_trip_records_yellow.pdf",
        "01_Data/Reference/taxi_zone_lookup.csv",
    ]
    paths.extend(
        f"01_Data/Reference/taxi_zones/taxi_zones.{suffix}"
        for suffix in ("cpg", "dbf", "prj", "shp", "shx")
    )
    for task in range(1, 6):
        paths.extend(
            f"04_Results/Task_{task}/{name}"
            for name in ("run.log", "run.json", "summary.json", "report.tex")
        )
    return [PROJECT_ROOT / path for path in paths]


def main() -> None:
    missing = [
        path.relative_to(PROJECT_ROOT)
        for path in required_files()
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(f"Required submission files are missing: {missing}")
    manifest = json.loads((PROJECT_ROOT / "manifest.json").read_text())
    for task in range(1, 6):
        folder = PROJECT_ROOT / f"04_Results/Task_{task}"
        for kind, subfolder, suffix in (
            ("tables", "tables", ".csv"),
            ("figures", "figures", ".png"),
        ):
            actual = len(list((folder / subfolder).glob(f"*{suffix}")))
            expected = manifest[f"Task_{task}"][kind]
            if actual != expected:
                raise SystemExit(
                    f"Task {task} {kind}: expected {expected}, found {actual}"
                )
    files = sorted(
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file() and should_include(path)
    )
    ARCHIVE_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(
        ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in files:
            archive.write(
                path, Path(PROJECT_ROOT.name) / path.relative_to(PROJECT_ROOT)
            )
    with zipfile.ZipFile(ARCHIVE) as archive:
        corrupt = archive.testzip()
    if corrupt:
        raise SystemExit(f"ZIP integrity check failed: {corrupt}")
    print(f"Wrote {ARCHIVE.name}: {len(files)} files, {ARCHIVE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
