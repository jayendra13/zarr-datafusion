"""Shared plumbing for the Python (xarray) side of the benchmark.

Each workload script computes exactly what its SQL counterpart computes, then
emits a single machine-readable line that `run_bench.py` parses:

    BENCH_JSON {"rows": 535704, "mean": 18.412345, "t_open": 0.9, ...}

Two conventions matter for the comparison to be honest, and both are easy to get
wrong:

**Stop the clock before printing.** The Rust CLI captures its elapsed time before
rendering results, so the Python side must not include serialization either.
`t_exec` ends the moment the result is materialized in memory.

**Match DataFusion's NaN semantics.** xarray's reductions default to
`skipna=True`, silently dropping NaN cells; DataFusion's `AVG` propagates NaN
instead. Left alone, the two engines quietly compute different things over any
variable with missing cells (SST over land, satellite nodata). Every reduction
here passes `skipna=False`, and the workload drops NaN explicitly where its SQL
does — mirroring the `HAVING`/`WHERE NOT isnan(...)` clauses rather than relying
on a library default.
"""

import json
import time


class Phase:
    """Wall-clock timer for one phase of the run."""

    def __init__(self):
        self.t0 = time.perf_counter()

    def done(self):
        return time.perf_counter() - self.t0


def emit(workload, rows, mean, t_open, t_exec, notes=""):
    """Print the one line `run_bench.py` looks for.

    `mean` is the fingerprint used to check that both engines computed the same
    thing; it is the mean of the result's value column (or the scalar itself for
    a global aggregate). `rows` is cross-checked against the row count the SQL
    CLI reports, which is what catches "the two sides benchmarked different
    queries" — the failure mode that makes a comparison table worthless.
    """
    payload = {
        "workload": workload,
        "engine": "python-xarray",
        "rows": int(rows),
        "mean": None if mean is None else round(float(mean), 6),
        "t_open": round(t_open, 4),
        "t_exec": round(t_exec, 4),
        "t_total": round(t_open + t_exec, 4),
        "notes": notes,
    }
    print("BENCH_JSON " + json.dumps(payload), flush=True)


# ARCO-ERA5 on GCS: the same store the SQL side registers.
ERA5_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Anonymous access — the bucket is public.
ERA5_OPTS = {"token": "anon"}


# The store's dimension coordinates — never dropped, or label-based selection
# (`.sel(time=..., latitude=...)`) stops working and the comparison changes shape.
ERA5_COORDS = {"time", "latitude", "longitude", "level"}


def open_era5(xr, keep):
    """Open ARCO-ERA5 lazily, keeping only the variables `keep` names.

    `chunks={}` keeps it dask-backed, so nothing is fetched until the selection is
    narrowed and `.compute()` is called.

    `drop_variables` is the part that matters, and it is not a micro-optimisation.
    ARCO-ERA5 holds 273 data variables, and a plain `xr.open_zarr()` acquires
    metadata for every one of them before returning — measured at ~97 s from a
    home connection and ~49 s in-region, regardless of which variable you then
    query. Dropping the ones the workload never touches cuts that by roughly 16x.

    Two things this deliberately does NOT do, because both would make the
    comparison dishonest rather than faster:

      * It keeps the dimension coordinates, so `.sel()` still selects by label —
        the same thing the SQL's WHERE clause does. Dropping them would be
        comparing SQL-with-labels against Python-without.
      * It does not use `decode_times=False`. That was measured and saves nothing
        (95.7 s vs 96.9 s): the cost is per-variable metadata, not CF decoding.

    The enumeration needed to build the drop list is included in the caller's
    `t_open`, since a user has to pay it too.
    """
    import zarr

    group = zarr.open_group(ERA5_STORE, mode="r", storage_options=ERA5_OPTS)
    names = [k for k, _ in group.arrays()]
    wanted = set(keep) | ERA5_COORDS
    drop = [n for n in names if n not in wanted]

    return xr.open_zarr(
        ERA5_STORE,
        chunks={},
        storage_options=ERA5_OPTS,
        drop_variables=drop,
    )


def coord_between(ds, name, lo, hi):
    """Select a coordinate range by mask rather than `slice`.

    ERA5's latitude axis descends (90 → -90), so `slice(-5, 5)` silently returns
    nothing while `slice(5, -5)` works. A boolean mask is correct regardless of
    axis direction, which matters because the SQL side expresses these as
    `BETWEEN` and has no such ordering trap.
    """
    c = ds[name]
    return c[(c >= lo) & (c <= hi)]
