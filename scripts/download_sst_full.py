"""
Download the full ARCO-ERA5 sea_surface_temperature dataset (1940-2025).

Adds to the existing local Zarr store at data/era5_sst_local.zarr,
skipping any chunks already present (safe to resume after interruption).

Stats:
  Total chunks : 753,888  (one per hour, 1940-01-01 00:00 → 2025-12-31 23:00)
  Chunk size   : ~1.28 MB compressed  (721 × 1440 float32, Blosc lz4)
  Total size   : ~942 GB compressed

Time estimate at observed bandwidth:
  16 workers : ~4 MB/s  → ~66 hours
  32 workers : ~6 MB/s  → ~44 hours
  64 workers : ~8 MB/s  → ~33 hours

Run:
    uv run --with aiohttp --with tqdm --with requests \\
           scripts/download_sst_full.py

    # Adjust concurrency for your connection:
    uv run --with aiohttp --with tqdm --with requests \\
           scripts/download_sst_full.py --concurrency 32

    # Skip confirmation prompt:
    uv run --with aiohttp --with tqdm --with requests \\
           scripts/download_sst_full.py --yes
"""

import argparse
import asyncio
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

# ERA5 actual data range (validated chunk existence)
ERA5_START = datetime(1940, 1, 1,  0, tzinfo=timezone.utc)
ERA5_END   = datetime(2025, 12, 31, 23, tzinfo=timezone.utc)

OUTPUT_DIR = Path("data/era5_sst_local.zarr")


def hours_from_epoch(dt: datetime) -> int:
    return int((dt - GCS_EPOCH).total_seconds() // 3600)


START_IDX = hours_from_epoch(ERA5_START)   # 350616
END_IDX   = hours_from_epoch(ERA5_END)     # 1104503
ALL_INDICES = list(range(START_IDX, END_IDX + 1))


# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

async def download_one(
    session:      aiohttp.ClientSession,
    sem:          asyncio.Semaphore,
    chunk_idx:    int,
    local_path:   Path,
    bytes_done:   list,
    errors:       list,
) -> bool:
    """Download one SST chunk. Returns True if skipped (already exists)."""
    if local_path.exists() and local_path.stat().st_size > 0:
        return True  # already present

    url = f"{BASE_URL}/sea_surface_temperature/{chunk_idx}.0.0"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    async with sem:
        for attempt in range(3):
            try:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        # Chunk doesn't exist (sparse store — normal for unfilled years)
                        return False
                    if resp.status != 200:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    data = await resp.read()
                    local_path.write_bytes(data)
                    bytes_done[0] += len(data)
                    return False
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if attempt == 2:
                    errors.append((chunk_idx, str(e)))
                    return False
                await asyncio.sleep(2 ** attempt)  # backoff: 1s, 2s

    return False


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------

async def download_all(concurrency: int) -> None:
    sst_dir = OUTPUT_DIR / "sea_surface_temperature"

    # Count already-downloaded chunks
    existing = sum(1 for f in sst_dir.glob("*.0.0") if f.stat().st_size > 0) \
               if sst_dir.exists() else 0
    remaining = [i for i in ALL_INDICES
                 if not (sst_dir / f"{i}.0.0").exists()]

    print(f"Chunks total    : {len(ALL_INDICES):>10,}")
    print(f"Already present : {existing:>10,}  ({existing * 1.28 / 1024:.1f} GB)")
    print(f"To download     : {len(remaining):>10,}  ({len(remaining) * 1.28 / 1024:.1f} GB)")
    print()

    if not remaining:
        print("Nothing to download — store is complete.")
        return

    sem           = asyncio.Semaphore(concurrency)
    bytes_done    = [0]
    errors        = []
    skipped_count = [0]

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    timeout   = aiohttp.ClientTimeout(total=300, connect=10, sock_read=120)

    t_start = time.monotonic()

    with tqdm(
        total=len(remaining),
        unit="chunk",
        desc="Downloading",
        ncols=100,
        dynamic_ncols=False,
    ) as bar:

        async def worker(idx: int) -> None:
            local = sst_dir / f"{idx}.0.0"
            skipped = await download_one(
                session, sem, idx, local, bytes_done, errors
            )
            if skipped:
                skipped_count[0] += 1
            bar.update(1)

            # Update postfix every 100 chunks
            done = bar.n
            if done % 100 == 0 and done > 0:
                elapsed   = time.monotonic() - t_start
                mb_done   = bytes_done[0] / 1_048_576
                mb_s      = mb_done / elapsed
                remaining_chunks = len(remaining) - done
                eta_s     = remaining_chunks * 1.28 / mb_s if mb_s > 0 else 0
                eta_h     = eta_s / 3600
                bar.set_postfix(
                    MB_s=f"{mb_s:.1f}",
                    ETA=f"{eta_h:.1f}h",
                    errors=len(errors),
                    refresh=False,
                )

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Process in batches to avoid holding millions of coroutines in memory
            batch_size = concurrency * 10
            for batch_start in range(0, len(remaining), batch_size):
                batch = remaining[batch_start: batch_start + batch_size]
                await asyncio.gather(*[worker(idx) for idx in batch])

    elapsed  = time.monotonic() - t_start
    mb_total = bytes_done[0] / 1_048_576
    gb_total = mb_total / 1024
    mb_s     = mb_total / elapsed if elapsed > 0 else 0

    print()
    print(f"Downloaded : {len(remaining) - len(errors) - skipped_count[0]:>8,} chunks  ({gb_total:.2f} GB)")
    print(f"Skipped    : {skipped_count[0]:>8,} chunks  (already present)")
    print(f"Errors     : {len(errors):>8,} chunks")
    print(f"Elapsed    : {elapsed/3600:.2f} hours  ({mb_s:.2f} MB/s)")

    if errors:
        err_path = OUTPUT_DIR / "download_errors.txt"
        with open(err_path, "w") as f:
            for idx, msg in errors:
                f.write(f"{idx}\t{msg}\n")
        print(f"\n{len(errors)} errors written to {err_path}")
        print("Re-run the script to retry failed chunks.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download full ERA5 SST (1940-2025)")
    parser.add_argument("--concurrency", type=int, default=16,
                        help="Parallel downloads (default: 16)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help=f"Local Zarr store path (default: {OUTPUT_DIR})")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt")
    args = parser.parse_args()

    print("ARCO-ERA5 Full SST Download")
    print("===========================")
    print(f"Source      : {BASE_URL}/sea_surface_temperature/")
    print(f"Output      : {args.output}")
    print(f"Range       : {ERA5_START.date()} → {ERA5_END.date()}")
    print(f"Concurrency : {args.concurrency} workers")
    print()
    total_mb = len(ALL_INDICES) * 1.28
    print(f"Estimated size : ~{total_mb/1024:.0f} GB compressed")
    print(f"Estimated time : ~{total_mb/4/3600:.0f}h at 4 MB/s  "
          f"(~{total_mb/8/3600:.0f}h at 8 MB/s,  "
          f"~{total_mb/16/3600:.0f}h at 16 MB/s)")
    print()

    # Connectivity check
    print("Checking GCS connectivity ...", end=" ", flush=True)
    try:
        r = requests.head(
            f"{BASE_URL}/sea_surface_temperature/{START_IDX}.0.0", timeout=10
        )
        print(f"OK  (HTTP {r.status_code})")
    except Exception as e:
        print(f"FAILED: {e}")
        return
    print()

    if not args.yes:
        answer = input(
            "This will download up to ~942 GB. Continue? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborted.")
            return
    print()

    asyncio.run(download_all(args.concurrency))

    print()
    print("Store location :", args.output)
    print("Query locally  :")
    print(f"  CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION '{args.output}'")


if __name__ == "__main__":
    main()
