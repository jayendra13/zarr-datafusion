"""
Compute rolling 3-month Niño 3.4 SST from local ARCO-ERA5 Zarr chunks.

Produces a table matching NOAA CPC oni.ascii.txt format:

    SEAS  YR    TOTAL    ANOM
    DJF  1950   24.72   -1.53
    JFM  1950   25.17   -1.34
    ...
    NDJ  1950   25.34   -0.80
    DJF  1951   25.42   -0.82

Season / year labelling follows NOAA convention: the **middle month**
of the 3-month window determines the year label.
    DJF year Y  = Dec(Y-1), Jan(Y), Feb(Y)   — middle: Jan
    NDJ year Y  = Nov(Y),   Dec(Y), Jan(Y+1) — middle: Dec

Anomaly is computed against a per-season climatology (default 1991–2020).

Note: the local store currently holds only December chunks, so non-December
months will show as NaN until those months are downloaded.

Usage
-----
    uv run --with numpy --with numcodecs --with tqdm \\
           scripts/compute_nino34_sst.py

    uv run --with numpy --with numcodecs --with tqdm \\
           scripts/compute_nino34_sst.py --years 1980-2025 --workers 8 \\
                                         --baseline 1991-2020 --csv out.csv
"""

from __future__ import annotations

import argparse
import calendar
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numcodecs
import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

STORE        = Path("data/era5_sst_local.zarr")
CF_EPOCH     = datetime(1900, 1, 1, tzinfo=timezone.utc)
TIME_TOTAL   = 1_323_648
KELVIN       = 273.15

# Niño 3.4: lat −5°–+5°, lon 190°–240°  (ERA5: lat 90→−90, lon 0→359.75)
NINO34_LAT = slice(340, 381)   # 41 cells
NINO34_LON = slice(760, 961)   # 201 cells

# 12 rolling seasons.  Each tuple: (name, (m1, m2, m3)) where the middle
# month m2 determines the year label (NOAA convention).
SEASONS = [
    ("DJF", (12,  1,  2)),
    ("JFM", ( 1,  2,  3)),
    ("FMA", ( 2,  3,  4)),
    ("MAM", ( 3,  4,  5)),
    ("AMJ", ( 4,  5,  6)),
    ("MJJ", ( 5,  6,  7)),
    ("JJA", ( 6,  7,  8)),
    ("JAS", ( 7,  8,  9)),
    ("ASO", ( 8,  9, 10)),
    ("SON", ( 9, 10, 11)),
    ("OND", (10, 11, 12)),
    ("NDJ", (11, 12,  1)),
]

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("nino34_compute")

# ──────────────────────────────────────────────────────────────────────────────
# Season helpers
# ──────────────────────────────────────────────────────────────────────────────

def season_month_years(months: tuple[int,int,int], label_year: int) -> list[tuple[int,int]]:
    """
    Resolve a season's (m1, m2, m3) tuple into actual (year, month) pairs.

    DJF year 1950 → [(1949,12), (1950,1), (1950,2)]
    NDJ year 1950 → [(1950,11), (1950,12), (1951,1)]
    """
    m1, m2, m3 = months
    y1 = label_year - 1 if m1 > m2 else label_year   # wraps back  (DJF: Dec)
    y3 = label_year + 1 if m3 < m2 else label_year   # wraps forward (NDJ: Jan)
    return [(y1, m1), (label_year, m2), (y3, m3)]


def all_month_keys(first_year: int, last_year: int) -> set[tuple[int,int]]:
    """
    Return every (year, month) pair needed to compute all seasons for
    [first_year, last_year].  Includes one extra year on each side for
    boundary seasons (DJF needs Dec of prev year, NDJ needs Jan of next).
    """
    keys: set[tuple[int,int]] = set()
    for year in range(first_year, last_year + 1):
        for _, months in SEASONS:
            for y, m in season_month_years(months, year):
                keys.add((y, m))
    return keys


# ──────────────────────────────────────────────────────────────────────────────
# Chunk I/O
# ──────────────────────────────────────────────────────────────────────────────

_blosc = numcodecs.Blosc()


def month_chunk_indices(year: int, month: int) -> list[int]:
    """
    ERA5 time-step indices for every hour in year/month.
    Uses CF-epoch arithmetic — no file reads needed.
    """
    days  = calendar.monthrange(year, month)[1]
    start = int((datetime(year, month,    1,  0, tzinfo=timezone.utc) - CF_EPOCH).total_seconds() // 3600)
    end   = int((datetime(year, month, days, 23, tzinfo=timezone.utc) - CF_EPOCH).total_seconds() // 3600)
    return list(range(start, min(end, TIME_TOTAL - 1) + 1))


def read_nino34_snapshot(chunk_idx: int) -> float | None:
    """
    Read one SST chunk and return the Niño 3.4 spatial mean in Kelvin.
    Returns None if the file is not present (chunk not downloaded).
    """
    path = STORE / "sea_surface_temperature" / f"{chunk_idx}.0.0"
    if not path.exists():
        return None
    raw    = path.read_bytes()
    data   = _blosc.decode(raw)
    globe  = np.frombuffer(data, dtype="<f4").reshape(721, 1440)
    region = globe[NINO34_LAT, NINO34_LON]
    mean   = np.nanmean(region)
    return None if np.isnan(mean) else float(mean)


def compute_monthly_mean(year: int, month: int) -> tuple[int, int, float]:
    """
    Compute Niño 3.4 spatial mean SST (°C) averaged over all hours in
    year/month.  Returns (year, month, sst_celsius) where sst is NaN
    if no chunks are present locally.
    """
    indices    = month_chunk_indices(year, month)
    snapshots  = [v for idx in indices if (v := read_nino34_snapshot(idx)) is not None]

    if not snapshots:
        return year, month, float("nan")

    return year, month, float(np.mean(snapshots)) - KELVIN


# ──────────────────────────────────────────────────────────────────────────────
# Rolling 3-month seasons
# ──────────────────────────────────────────────────────────────────────────────

def build_seasonal_table(
    monthly: dict[tuple[int,int], float],
    first_year: int,
    last_year:  int,
) -> list[tuple[str, int, float]]:
    """
    Build list of (season_name, label_year, mean_sst) rows.
    A season is NaN if any of its 3 months has no data.
    """
    rows: list[tuple[str, int, float]] = []
    for year in range(first_year, last_year + 1):
        for name, months in SEASONS:
            month_values = [
                monthly.get((y, m), float("nan"))
                for y, m in season_month_years(months, year)
            ]
            if any(np.isnan(v) for v in month_values):
                sst = float("nan")
            else:
                sst = float(np.mean(month_values))
            rows.append((name, year, sst))
    return rows


def compute_climatology(
    rows:     list[tuple[str, int, float]],
    baseline: tuple[int, int],
) -> dict[str, float]:
    """Per-season climatological mean over baseline years."""
    base_start, base_end = baseline
    buckets: dict[str, list[float]] = {name: [] for name, _ in SEASONS}
    for name, year, sst in rows:
        if base_start <= year <= base_end and not np.isnan(sst):
            buckets[name].append(sst)

    climo: dict[str, float] = {}
    for name, values in buckets.items():
        climo[name] = float(np.mean(values)) if values else float("nan")
    return climo


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────

HEADER  = f"{'SEAS':<5} {'YR':>4}   {'TOTAL':>7}   {'ANOM':>7}"
DIVIDER = "-" * len(HEADER)


def print_table(
    rows:   list[tuple[str, int, float]],
    climo:  dict[str, float],
    baseline: tuple[int, int],
) -> None:
    print()
    print("Niño 3.4 Rolling 3-Month SST  (ERA5, spatially averaged)")
    print(f"Region  : lat −5°–+5°, lon 190°–240°")
    print(f"Baseline: {baseline[0]}–{baseline[1]}")
    print(f"Note    : ERA5 vs ERSSTv5 offset ~0.1–0.2 °C is expected")
    print(f"Note    : [NaN] = month not yet downloaded")
    print()
    print(HEADER)
    print(DIVIDER)

    for name, year, sst in rows:
        c = climo.get(name, float("nan"))
        if np.isnan(sst) or np.isnan(c):
            print(f"{name:<5} {year:>4}   {'NaN':>7}   {'---':>7}")
        else:
            anom = sst - c
            print(f"{name:<5} {year:>4}   {sst:>7.2f}   {anom:>+7.2f}")

    print(DIVIDER)


def write_csv(
    path:   Path,
    rows:   list[tuple[str, int, float]],
    climo:  dict[str, float],
) -> None:
    lines = ["seas,year,sst_total,sst_anom\n"]
    for name, year, sst in rows:
        if np.isnan(sst):
            continue
        anom = sst - climo.get(name, float("nan"))
        lines.append(f"{name},{year},{sst:.4f},{anom:+.4f}\n")
    path.write_text("".join(lines))
    log.info("CSV written: %s  (%d rows)", path, len(lines) - 1)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute rolling 3-month Niño 3.4 SST from local ERA5 Zarr chunks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--years",    default="1940-2025",
                   help="Year range for output, e.g. 1980-2025")
    p.add_argument("--baseline", default="1991-2020",
                   help="Climatology baseline years")
    p.add_argument("--workers",  type=int, default=4,
                   help="Parallel chunk-reading threads")
    p.add_argument("--store",    type=Path, default=STORE,
                   help="Local Zarr store path")
    p.add_argument("--csv",      type=Path, default=None,
                   help="Optional CSV output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global STORE
    STORE = args.store

    first_year, last_year = (int(y) for y in args.years.split("-"))
    baseline = tuple(int(y) for y in args.baseline.split("-"))

    log.info(
        "Computing Niño 3.4 SST  years=%d–%d  baseline=%d–%d  workers=%d",
        first_year, last_year, baseline[0], baseline[1], args.workers,
    )

    # ── Step 1: identify all (year, month) pairs needed ────────────────────
    keys = all_month_keys(first_year, last_year)
    log.info("Need %d unique (year, month) pairs", len(keys))

    # ── Step 2: compute monthly means in parallel ───────────────────────────
    monthly: dict[tuple[int,int], float] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compute_monthly_mean, y, m): (y, m) for y, m in keys}

        with tqdm(total=len(keys), unit="month",
                  desc="Monthly means", ncols=80) as bar:
            for fut in as_completed(futures):
                yr, mo, sst = fut.result()
                monthly[(yr, mo)] = sst
                bar.update(1)

    valid = sum(1 for v in monthly.values() if not np.isnan(v))
    log.info("Monthly means computed: %d valid / %d total", valid, len(monthly))

    # ── Step 3: rolling 3-month seasons ────────────────────────────────────
    rows = build_seasonal_table(monthly, first_year, last_year)

    # ── Step 4: per-season climatology ─────────────────────────────────────
    climo = compute_climatology(rows, baseline)
    valid_seasons = sum(1 for n, v in climo.items() if not np.isnan(v))
    log.info(
        "Climatology: %d/%d seasons have baseline data  (%d–%d)",
        valid_seasons, len(SEASONS), baseline[0], baseline[1],
    )

    # ── Step 5: output ──────────────────────────────────────────────────────
    print_table(rows, climo, baseline)

    if args.csv:
        write_csv(args.csv, rows, climo)

    # Summary
    valid_rows = [(n, y, s) for n, y, s in rows if not np.isnan(s)]
    if valid_rows:
        totals = [s for _, _, s in valid_rows]
        log.info(
            "Valid rows: %d / %d   min=%.2f  max=%.2f  mean=%.2f °C",
            len(valid_rows), len(rows),
            min(totals), max(totals), float(np.mean(totals)),
        )


if __name__ == "__main__":
    main()
