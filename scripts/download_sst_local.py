"""
Download ARCO-ERA5 sea_surface_temperature locally for offline analysis.

Builds a valid local Zarr v2 store that zarr-datafusion can query directly:
  CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION 'data/era5_sst_local.zarr'

Two modes:
  --snapshot  Dec 15 12:00 UTC per year  (86 chunks,    ~110 MB,  ~2 min)
  --december  All December hours         (61752 chunks,  ~79 GB,  hours)

Always downloads alongside SST:
  - Zarr metadata (.zmetadata, .zarray, .zattrs)
  - latitude  : 1 chunk  (~3 KB compressed)
  - longitude : 1 chunk  (~5 KB compressed)
  - time      : 12 chunks covering 1940-2025  (~78 MB compressed)

Run:
    uv run --with aiohttp --with tqdm --with requests \
           scripts/download_sst_local.py --snapshot

    uv run --with aiohttp --with tqdm --with requests \
           scripts/download_sst_local.py --december --concurrency 32
"""

import argparse
import asyncio
import calendar
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET    = "gcp-public-data-arco-era5"
ZARR_PATH = "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BASE_URL  = f"https://storage.googleapis.com/{BUCKET}/{ZARR_PATH}"
GCS_EPOCH = datetime(1900, 1, 1, tzinfo=timezone.utc)

FIRST_YEAR = 1940
LAST_YEAR  = 2025

# time coordinate chunking (from .zarray)
TIME_CHUNK_SIZE = 67706
TIME_TOTAL      = 1323648
TIME_NUM_CHUNKS = (TIME_TOTAL + TIME_CHUNK_SIZE - 1) // TIME_CHUNK_SIZE  # = 20


def hours_from_epoch(year: int, month: int, day: int, hour: int) -> int:
    dt = datetime(year, month, day, hour, tzinfo=timezone.utc)
    return int((dt - GCS_EPOCH).total_seconds() // 3600)


# ---------------------------------------------------------------------------
# Chunk index helpers
# ---------------------------------------------------------------------------

def snapshot_sst_indices() -> list[int]:
    """Dec 15 12:00 UTC for each year 1940-2025 — one chunk per year."""
    return [hours_from_epoch(y, 12, 15, 12) for y in range(FIRST_YEAR, LAST_YEAR + 1)]


def december_sst_indices() -> list[int]:
    """All hourly chunks for December of every year 1940-2025."""
    indices = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        start = hours_from_epoch(year, 12, 1, 0)
        end   = hours_from_epoch(year, 12, 31, 23)
        indices.extend(range(start, end + 1))
    return indices


def months_sst_indices(year_months: list[tuple[int, int]]) -> list[int]:
    """All hourly chunks for each (year, month) pair, in chronological order."""
    indices = []
    for year, month in year_months:
        last_day = calendar.monthrange(year, month)[1]
        start = hours_from_epoch(year, month, 1, 0)
        end   = hours_from_epoch(year, month, last_day, 23)
        indices.extend(range(start, end + 1))
    return indices


def parse_year_months(spec: str) -> list[tuple[int, int]]:
    """Parse 'YYYY-MM,YYYY-MM' into [(year, month), ...]."""
    pairs = []
    for token in spec.split(","):
        y, m = token.strip().split("-")
        pairs.append((int(y), int(m)))
    return pairs


def time_coord_chunks_needed(sst_indices: list[int]) -> list[int]:
    """Return time coordinate chunk numbers that cover the given SST indices."""
    chunks = set()
    for idx in sst_indices:
        chunks.add(idx // TIME_CHUNK_SIZE)
    return sorted(chunks)


# ---------------------------------------------------------------------------
# File lists to download
# ---------------------------------------------------------------------------

METADATA_FILES = [
    ".zmetadata",
    ".zattrs",
    "sea_surface_temperature/.zarray",
    "sea_surface_temperature/.zattrs",
    "latitude/.zarray",
    "latitude/.zattrs",
    "longitude/.zarray",
    "longitude/.zattrs",
    "time/.zarray",
    "time/.zattrs",
    "level/.zarray",       # pressure levels coordinate (shape [37])
    "level/.zattrs",
]

COORD_DATA_FILES = [
    "latitude/0",
    "longitude/0",
    "level/0",             # 37 pressure levels, single chunk
]


def build_file_list(sst_indices: list[int]) -> list[tuple[str, int]]:
    """
    Returns list of (remote_path, expected_bytes) to download.
    expected_bytes = 0 means unknown (metadata/coord files).
    """
    files: list[tuple[str, int]] = []

    # Metadata (small text files)
    for f in METADATA_FILES:
        files.append((f, 0))

    # Coordinate data
    for f in COORD_DATA_FILES:
        files.append((f, 0))

    # Time coordinate chunks covering our SST range
    for chunk_num in time_coord_chunks_needed(sst_indices):
        files.append((f"time/{chunk_num}", 0))

    # SST chunks (1.28 MB compressed each)
    for idx in sst_indices:
        files.append((f"sea_surface_temperature/{idx}.0.0", 1_346_490))

    return files


# ---------------------------------------------------------------------------
# Async downloader
# ---------------------------------------------------------------------------

async def download_one(
    session:      aiohttp.ClientSession,
    sem:          asyncio.Semaphore,
    remote_path:  str,
    local_path:   Path,
    progress:     tqdm,
    bytes_counter: list,
) -> tuple[str, bool, str]:
    """Download one file. Returns (path, skipped, error_msg)."""
    if local_path.exists():
        progress.update(1)
        return remote_path, True, ""

    url = f"{BASE_URL}/{remote_path}"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    async with sem:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return remote_path, False, f"HTTP {resp.status}"
                data = await resp.read()
                local_path.write_bytes(data)
                bytes_counter[0] += len(data)
                progress.update(1)
                return remote_path, False, ""
        except Exception as e:
            return remote_path, False, str(e)


async def download_all(
    files:       list[tuple[str, int]],
    output_dir:  Path,
    concurrency: int,
) -> None:
    sem           = asyncio.Semaphore(concurrency)
    bytes_counter = [0]
    errors        = []
    skipped       = 0

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    timeout   = aiohttp.ClientTimeout(total=300, connect=10, sock_read=120)

    t0 = time.monotonic()

    with tqdm(total=len(files), unit="chunk", desc="Downloading", ncols=90) as progress:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                download_one(
                    session, sem,
                    remote_path,
                    output_dir / remote_path,
                    progress,
                    bytes_counter,
                )
                for remote_path, _ in files
            ]
            results = await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0

    for path, was_skipped, err in results:
        if was_skipped:
            skipped += 1
        elif err:
            errors.append((path, err))

    downloaded = len(files) - skipped - len(errors)
    mb         = bytes_counter[0] / 1_048_576
    mbps       = mb / elapsed if elapsed > 0 else 0

    print()
    print(f"Downloaded : {downloaded:>6} files  ({mb:.1f} MB)")
    print(f"Skipped    : {skipped:>6} files  (already present)")
    print(f"Errors     : {len(errors):>6} files")
    print(f"Elapsed    : {elapsed:.1f}s   ({mbps:.2f} MB/s)")

    if errors:
        print("\nFailed files:")
        for path, err in errors[:10]:
            print(f"  {path}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download ARCO-ERA5 SST locally")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--snapshot", action="store_true", default=True,
        help="Dec 15 12:00 UTC per year (86 chunks, ~110 MB) [default]",
    )
    mode.add_argument(
        "--december", action="store_true",
        help="All December hours per year (61752 chunks, ~79 GB)",
    )
    mode.add_argument(
        "--year-months", type=str, default=None,
        help="All hours for explicit months, e.g. '2026-01,2026-02'",
    )
    parser.add_argument(
        "--concurrency", type=int, default=16,
        help="Parallel downloads (default: 16)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/era5_sst_local.zarr"),
        help="Local output directory (default: data/era5_sst_local.zarr)",
    )
    parser.add_argument(
        "--years", type=str, default=None,
        help="Year range e.g. 1990-2025 (default: 1940-2025)",
    )
    args = parser.parse_args()

    # Parse year range
    global FIRST_YEAR, LAST_YEAR
    if args.years:
        parts = args.years.split("-")
        FIRST_YEAR, LAST_YEAR = int(parts[0]), int(parts[1])

    # Select mode (year-months takes precedence; --snapshot defaults to True)
    if args.year_months:
        year_months = parse_year_months(args.year_months)
        sst_indices = months_sst_indices(year_months)
        mode_label  = "Explicit months: " + ", ".join(f"{y}-{m:02d}" for y, m in year_months)
    elif args.december:
        sst_indices = december_sst_indices()
        mode_label  = "Full December monthly"
    else:
        sst_indices = snapshot_sst_indices()
        mode_label  = "Dec 15 12:00 snapshot"

    files       = build_file_list(sst_indices)
    sst_chunks  = sum(1 for f, _ in files if "sea_surface_temperature" in f)
    meta_chunks = len(files) - sst_chunks
    total_mb    = sst_chunks * 1.28 + meta_chunks * 0.01

    print("ARCO-ERA5 SST Download")
    print("======================")
    print(f"Mode        : {mode_label}")
    print(f"Years       : {FIRST_YEAR}–{LAST_YEAR}")
    print(f"Output      : {args.output}")
    print(f"Concurrency : {args.concurrency} parallel downloads")
    print()
    print(f"Files to download:")
    print(f"  Metadata + coordinates : {meta_chunks}")
    print(f"  SST chunks             : {sst_chunks}  (~1.28 MB each)")
    print(f"  Total estimate         : ~{total_mb:.0f} MB")
    print()

    # Quick connectivity check
    print("Checking GCS connectivity ...", end=" ", flush=True)
    r = requests.head(f"{BASE_URL}/.zmetadata", timeout=10)
    if r.status_code != 200:
        print(f"FAILED (HTTP {r.status_code})")
        return
    print("OK")
    print()

    args.output.mkdir(parents=True, exist_ok=True)

    asyncio.run(download_all(files, args.output, args.concurrency))

    print()
    print("Local store ready at:", args.output)
    print()
    print("Verify with xarray:")
    print(f"  import xarray as xr")
    print(f"  ds = xr.open_zarr('{args.output}', consolidated=True)")
    print(f"  print(ds)")
    print()
    print("Query with zarr-datafusion:")
    print(f"  CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION '{args.output}'")


if __name__ == "__main__":
    main()
