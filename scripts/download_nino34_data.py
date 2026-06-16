"""
Download ARCO-ERA5 December SST chunks for Niño 3.4 analysis.

Design
------
Phase 1 — Probe:    Read .zmetadata *remotely* from GCS to verify
                    array geometry (shape, dtype, chunk layout).
Phase 2 — Identify: Derive December time-step indices via CF-epoch
                    arithmetic — no full time-coordinate download.
Phase 3 — Download: Fetch December SST chunks concurrently (aiohttp).
                    A JSON checkpoint file records every completed chunk
                    so the run can be interrupted and resumed cleanly.
Phase 4 — Reference: Save NOAA ONI table as a clean CSV ready for:
                      CREATE EXTERNAL TABLE oni STORED AS CSV …

Later, zarr-datafusion reads .zmetadata *remotely* (GCS) and SST
chunks from the *local* store — no full mirror is required.

Usage
-----
    uv run --with aiohttp --with tqdm \\
           scripts/download_nino34_data.py

    # Custom range, higher concurrency
    uv run --with aiohttp --with tqdm \\
           scripts/download_nino34_data.py --years 1980-2025 --concurrency 32

    # Check what would be downloaded without fetching
    uv run --with aiohttp --with tqdm \\
           scripts/download_nino34_data.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BUCKET    = "gcp-public-data-arco-era5"
ZARR_PATH = "ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
BASE_URL  = f"https://storage.googleapis.com/{BUCKET}/{ZARR_PATH}"

CF_EPOCH = datetime(1900, 1, 1, tzinfo=timezone.utc)   # "hours since 1900-01-01"

ONI_URL  = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

DEFAULT_OUTPUT      = Path("data/era5_sst_local.zarr")
DEFAULT_ONI_OUTPUT  = Path("data/oni_reference.csv")
CHECKPOINT_FILENAME = ".download_checkpoint.json"

# Metadata + coordinate files that must be present for a valid local Zarr store
METADATA_FILES = [
    ".zmetadata",
    ".zattrs",
    "sea_surface_temperature/.zarray",
    "sea_surface_temperature/.zattrs",
    "latitude/.zarray",   "latitude/.zattrs",   "latitude/0",
    "longitude/.zarray",  "longitude/.zattrs",  "longitude/0",
    "level/.zarray",      "level/.zattrs",      "level/0",
    "time/.zarray",       "time/.zattrs",
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
log = logging.getLogger("nino34_download")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Probe remote metadata
# ──────────────────────────────────────────────────────────────────────────────

async def probe_remote_metadata(session: aiohttp.ClientSession) -> dict:
    """
    Fetch .zmetadata from GCS and extract time + SST array geometry.

    Returns
    -------
    dict with keys: time_total, time_chunk_size, sst_shape,
                    sst_dtype, sst_chunk_shape, fill_value
    """
    url = f"{BASE_URL}/.zmetadata"
    log.info("Phase 1 — probing remote metadata: %s", url)

    async with session.get(url) as resp:
        resp.raise_for_status()
        raw = await resp.text()

    meta  = json.loads(raw)["metadata"]
    time_ = meta["time/.zarray"]
    sst   = meta["sea_surface_temperature/.zarray"]

    result = {
        "time_total":      time_["shape"][0],
        "time_chunk_size": time_["chunks"][0],
        "sst_shape":       sst["shape"],
        "sst_dtype":       sst["dtype"],
        "sst_chunk_shape": sst["chunks"],
        "fill_value":      sst.get("fill_value"),
    }

    log.info(
        "Remote: time=%d steps (chunk=%d), SST=%s %s, fill=%s",
        result["time_total"], result["time_chunk_size"],
        result["sst_shape"],  result["sst_dtype"],
        result["fill_value"],
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Identify chunks
# ──────────────────────────────────────────────────────────────────────────────

def identify_chunks_for_months(
    first_year: int,
    last_year:  int,
    time_total: int,
    months:     list[int],
) -> list[int]:
    """
    Return ERA5 time-step indices whose timestamp falls in any of the
    given calendar months (1–12) for years [first_year, last_year].

    Each index maps directly to the SST chunk file:
        sea_surface_temperature/{index}.0.0

    Uses CF-epoch arithmetic — no network call needed.

    months=[12]        → December only  (~64K chunks, ~86 GB)
    months=list(1..12) → full year      (~1.3M chunks, ~1.7 TB)
    months=[10,11,12,1]→ OND+NDJ window (~256K chunks, ~344 GB)
    """
    import calendar as _cal

    indices: list[int] = []

    # Extend range by one year on each side so boundary seasons
    # (DJF needs Dec of prev year, NDJ needs Jan of next) are covered.
    for year in range(first_year - 1, last_year + 2):
        for month in months:
            days      = _cal.monthrange(year, month)[1]
            start_dt  = datetime(year, month,    1,  0, tzinfo=timezone.utc)
            end_dt    = datetime(year, month, days, 23, tzinfo=timezone.utc)
            start_idx = int((start_dt - CF_EPOCH).total_seconds() // 3600)
            end_idx   = int((end_dt   - CF_EPOCH).total_seconds() // 3600)
            end_idx   = min(end_idx, time_total - 1)

            if start_idx >= time_total:
                continue

            indices.extend(range(start_idx, end_idx + 1))

    indices = sorted(set(indices))
    log.info(
        "Phase 2 — identified %d time steps  "
        "(months=%s, %d–%d, with boundary years)",
        len(indices), months, first_year, last_year,
    )
    return indices


def time_coord_chunks_for(indices: list[int], chunk_size: int) -> list[int]:
    """Return sorted unique time-coordinate chunk numbers covering indices."""
    return sorted({idx // chunk_size for idx in indices})


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────

class Checkpoint:
    """
    JSON-backed download state.

    Completed and failed chunk indices are persisted so that an
    interrupted run can resume without re-fetching successful chunks.
    Writes to disk every `save_every` completions and on shutdown.
    """

    def __init__(self, path: Path, save_every: int = 250):
        self.path       = path
        self.save_every = save_every
        self._lock      = asyncio.Lock()
        self.completed: set[int] = set()
        self.failed:    set[int] = set()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._dirty     = 0

    @classmethod
    def load(cls, path: Path, save_every: int = 250) -> "Checkpoint":
        cp = cls(path, save_every)
        if path.exists():
            data = json.loads(path.read_text())
            cp.completed  = set(data.get("completed", []))
            cp.failed     = set(data.get("failed",    []))
            cp.started_at = data.get("started_at", cp.started_at)
            log.info(
                "Checkpoint resumed: %d completed, %d failed",
                len(cp.completed), len(cp.failed),
            )
        else:
            log.info("No checkpoint found — starting fresh")
        return cp

    async def mark_done(self, idx: int) -> None:
        async with self._lock:
            self.completed.add(idx)
            self.failed.discard(idx)
            self._dirty += 1
            if self._dirty >= self.save_every:
                self._write()

    async def mark_failed(self, idx: int, reason: str) -> None:
        async with self._lock:
            self.failed.add(idx)
            log.warning("Chunk %d failed: %s", idx, reason)

    def _write(self) -> None:
        self.path.write_text(json.dumps({
            "completed":  sorted(self.completed),
            "failed":     sorted(self.failed),
            "started_at": self.started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stats": {
                "completed": len(self.completed),
                "failed":    len(self.failed),
            },
        }, indent=2))
        self._dirty = 0

    def save(self) -> None:
        """Force flush to disk — call on shutdown."""
        self._write()
        log.info(
            "Checkpoint saved → %s  (%d completed, %d failed)",
            self.path, len(self.completed), len(self.failed),
        )

    def pending(self, all_indices: list[int]) -> list[int]:
        """Indices not yet successfully downloaded."""
        return [i for i in all_indices if i not in self.completed]


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Concurrent download
# ──────────────────────────────────────────────────────────────────────────────

async def fetch_one(
    session:     aiohttp.ClientSession,
    sem:         asyncio.Semaphore,
    remote_path: str,
    local_path:  Path,
) -> tuple[bool, int]:
    """
    Fetch one file into local_path.
    Returns (success, bytes_written).
    Skips silently if the local file already exists.
    """
    if local_path.exists():
        return True, 0

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{remote_path}"

    async with sem:
        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    return True, 0   # fill-value region — not an error
                resp.raise_for_status()
                data = await resp.read()
            local_path.write_bytes(data)
            return True, len(data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.debug("Fetch failed %s: %s", remote_path, e)
            return False, 0


async def download_metadata_and_coords(
    session:         aiohttp.ClientSession,
    output_dir:      Path,
    time_chunk_size: int,
    dec_indices:     list[int],
) -> None:
    """Download .zarray/.zattrs + coordinate data (lat, lon, level, time chunks)."""
    time_chunks = time_coord_chunks_for(dec_indices, time_chunk_size)
    time_files  = [f"time/{c}" for c in time_chunks]
    all_files   = METADATA_FILES + time_files

    log.info("Downloading %d metadata/coordinate files", len(all_files))
    sem = asyncio.Semaphore(8)

    async def one(path: str) -> None:
        ok, _ = await fetch_one(session, sem, path, output_dir / path)
        if not ok:
            log.warning("Metadata fetch failed: %s", path)

    await asyncio.gather(*[one(f) for f in all_files])
    log.info("Metadata and coordinates ready")


async def download_sst_chunks(
    dec_indices: list[int],
    output_dir:  Path,
    checkpoint:  Checkpoint,
    concurrency: int,
) -> tuple[int, int, float]:
    """
    Concurrently fetch all pending December SST chunks.

    Returns (downloaded, failed, elapsed_seconds).
    """
    pending = checkpoint.pending(dec_indices)
    skipped = len(dec_indices) - len(pending)

    if skipped:
        log.info("Skipping %d already-downloaded chunks", skipped)
    if not pending:
        log.info("All chunks already present — nothing to download")
        return 0, 0, 0.0

    log.info("Phase 3 — downloading %d SST chunks (concurrency=%d)",
             len(pending), concurrency)

    sem         = asyncio.Semaphore(concurrency)
    bytes_total = [0]
    t0          = time.monotonic()

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    timeout   = aiohttp.ClientTimeout(total=300, connect=10, sock_read=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with tqdm(total=len(pending), unit="chunk",
                  desc="SST chunks", ncols=90, dynamic_ncols=True) as bar:

            async def worker(idx: int) -> None:
                remote = f"sea_surface_temperature/{idx}.0.0"
                local  = output_dir / remote
                ok, nb = await fetch_one(session, sem, remote, local)
                if ok:
                    bytes_total[0] += nb
                    await checkpoint.mark_done(idx)
                else:
                    await checkpoint.mark_failed(idx, "fetch error")
                bar.update(1)
                bar.set_postfix(
                    mb=f"{bytes_total[0]/1e6:.0f}",
                    failed=len(checkpoint.failed),
                )

            await asyncio.gather(*[worker(i) for i in pending])

    elapsed    = time.monotonic() - t0
    n_failed   = len(checkpoint.failed)
    n_done     = len(pending) - n_failed
    throughput = bytes_total[0] / 1e6 / elapsed if elapsed > 0 else 0

    log.info(
        "Download complete: %d chunks, %.1f MB in %.1fs (%.1f MB/s)",
        n_done, bytes_total[0] / 1e6, elapsed, throughput,
    )
    if n_failed:
        log.warning("%d chunks failed — re-run to retry", n_failed)

    return n_done, n_failed, elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — NOAA ONI reference CSV
# ──────────────────────────────────────────────────────────────────────────────

async def download_oni_reference(
    session:     aiohttp.ClientSession,
    output_path: Path,
) -> None:
    """
    Fetch NOAA CPC ONI ASCII table and save as a normalised CSV.

    Output columns: seas, year, sst_total, sst_climo, sst_anom

    DataFusion usage:
        CREATE EXTERNAL TABLE oni
        STORED AS CSV WITH HEADER ROW
        LOCATION 'data/oni_reference.csv'

    Then join to ERA5 output:
        SELECT e.year, e.sst_celsius, o.sst_total, o.sst_anom
        FROM era5_dec e JOIN oni o
          ON e.year = o.year AND o.seas = 'OND'
        ORDER BY e.year
    """
    log.info("Phase 4 — fetching NOAA ONI reference: %s", ONI_URL)

    async with session.get(ONI_URL) as resp:
        resp.raise_for_status()
        raw = await resp.text()

    rows: list[str] = []
    for line in raw.strip().splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] == "SEAS":
            continue
        seas, yr, total, climo, anom = parts[:5]
        rows.append(f"{seas},{yr},{total},{climo},{anom}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "seas,year,sst_total,sst_climo,sst_anom\n" + "".join(rows)
    )
    log.info("ONI reference saved: %d rows → %s", len(rows), output_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download ARCO-ERA5 SST chunks for Niño 3.4 ONI computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--years",       default="1940-2025",
                   help="Inclusive year range, e.g. 1980-2025")
    p.add_argument("--months",      default="all",
                   help=(
                       "Months to download. Presets: "
                       "'december' (month 12, ~86 GB), "
                       "'oni-window' (Oct–Jan, ~344 GB, enough for all 12 seasons), "
                       "'all' (full year, ~1.7 TB). "
                       "Or comma-separated list: '10,11,12,1'"
                   ))
    p.add_argument("--concurrency", type=int, default=16,
                   help="Parallel download workers")
    p.add_argument("--output",      type=Path, default=DEFAULT_OUTPUT,
                   help="Local Zarr store directory")
    p.add_argument("--oni-output",  type=Path, default=DEFAULT_ONI_OUTPUT,
                   help="ONI CSV output path")
    p.add_argument("--skip-oni",    action="store_true",
                   help="Skip NOAA ONI reference download")
    p.add_argument("--dry-run",     action="store_true",
                   help="Print plan and exit without downloading")
    p.add_argument("--debug",       action="store_true",
                   help="Enable debug logging")
    return p.parse_args()


def resolve_months(spec: str) -> list[int]:
    """Parse --months argument into a list of calendar month numbers."""
    if spec == "december":
        return [12]
    if spec == "oni-window":
        # Oct, Nov, Dec, Jan — covers all 12 seasons when combined with
        # the boundary-year extension in identify_chunks_for_months
        return [10, 11, 12, 1]
    if spec == "all":
        return list(range(1, 13))
    # Custom comma-separated list
    return [int(m.strip()) for m in spec.split(",")]


async def run(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    first_year, last_year = (int(y) for y in args.years.split("-"))
    checkpoint_path = args.output / CHECKPOINT_FILENAME

    # Shared session for probing + ONI download
    probe_timeout = aiohttp.ClientTimeout(total=60, connect=10)
    async with aiohttp.ClientSession(timeout=probe_timeout) as session:

        # ── Phase 1: probe ──────────────────────────────────────────────────
        remote = await probe_remote_metadata(session)

        # ── Phase 2: identify ───────────────────────────────────────────────
        months  = resolve_months(args.months)
        indices = identify_chunks_for_months(
            first_year, last_year, remote["time_total"], months
        )

        est_gb = len(indices) * 1_346_490 / 1e9
        log.info(
            "Plan: %d chunks, ~%.1f GB compressed  "
            "(months=%s, chunk shape %s, dtype %s, fill=%s)",
            len(indices), est_gb, months,
            remote["sst_chunk_shape"], remote["sst_dtype"],
            remote["fill_value"],
        )

        if args.dry_run:
            log.info("Dry run — exiting without downloading")
            return

        # ── Phase 4: ONI reference (do early, it's tiny) ───────────────────
        if not args.skip_oni:
            await download_oni_reference(session, args.oni_output)

    # ── Phase 3: SST chunks (separate session, higher concurrency) ─────────
    checkpoint = Checkpoint.load(checkpoint_path)

    # Save checkpoint on Ctrl+C or SIGTERM
    loop = asyncio.get_running_loop()

    def _on_signal(sig: signal.Signals) -> None:
        log.warning("Received %s — saving checkpoint and exiting", sig.name)
        checkpoint.save()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal, sig)

    # Metadata + coordinates first (needed for a valid Zarr store)
    meta_session_timeout = aiohttp.ClientTimeout(total=60, connect=10)
    async with aiohttp.ClientSession(timeout=meta_session_timeout) as meta_session:
        await download_metadata_and_coords(
            meta_session, args.output,
            remote["time_chunk_size"], indices,
        )

    # SST data chunks
    downloaded, failed, elapsed = await download_sst_chunks(
        indices, args.output, checkpoint, args.concurrency
    )
    checkpoint.save()

    # ── Summary ─────────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Done")
    log.info("  Years          : %d–%d", first_year, last_year)
    log.info("  December steps : %d", len(dec_indices))
    log.info("  Months         : %s", args.months)
    log.info("  Downloaded     : %d new chunks", downloaded)
    log.info("  Failed         : %d  (re-run to retry)", failed)
    log.info("  Zarr store     : %s", args.output)
    if not args.skip_oni:
        log.info("  ONI reference  : %s", args.oni_output)
    log.info("─" * 60)
    log.info("Next — run the SQL query:")
    log.info(
        "  cargo run --example nino34_query"
        "  # or: zarr-cli with CREATE EXTERNAL TABLE"
    )


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
