"""Q4 — NDVI per pixel over a Sentinel-2 scene. Counterpart of q4_ndvi.sql.

Shape under test: a pure per-pixel projection with no reduction at all. Two
co-registered bands on the same grid, one arithmetic expression, one row out per
input pixel. This is the complement of the aggregation workloads, and it is
numpy's home ground — expect the two engines to be close, and do not be surprised
if xarray wins.

Equivalent SQL:

    SELECT x, y, ROUND((b08 - b04) / (b08 + b04), 4) AS ndvi
    FROM scene
    WHERE NOT isnan(b08 - b04)
    ORDER BY y, x;

A note on fairness. The SQL says ORDER BY y, x; here the same ordering falls out
of transposing to (y, x) and raveling, because the scene is already a grid. That
is a genuine advantage of the array layout rather than a step we skipped, but it
is worth stating plainly when the numbers are published: the two engines are not
doing identical work for that clause.

The 80 dropped pixels are nodata (NaN in either band), so 1024*1024 - 80 rows.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from _common import Phase, emit  # noqa: E402

STORE = "data/s2_ndvi_scene.zarr"


def main():
    p = Phase()
    ds = xr.open_zarr(STORE, chunks={})
    t_open = p.done()

    p = Phase()
    b04 = ds["b04"]
    b08 = ds["b08"]
    ndvi = ((b08 - b04) / (b08 + b04)).compute()

    # Bands are stored (x, y); transposing to (y, x) then raveling gives image
    # raster order, matching the SQL's ORDER BY y, x.
    flat = ndvi.transpose("y", "x").values.ravel()

    # ROUND(..., 4) — applied so the fingerprint matches the SQL's rounded output
    # rather than differing in the sixth decimal for uninteresting reasons.
    flat = np.round(flat, 4)

    # WHERE NOT isnan(b08 - b04): a NaN in either band makes the difference NaN,
    # so one finite-check covers both. DataFusion treats NaN as equal to NaN and
    # greater than everything, so an unfiltered NaN would poison the mean here in
    # the same way it poisons MAX/AVG there.
    flat = flat[np.isfinite(flat)]
    t_exec = p.done()

    emit(
        "q4_ndvi",
        rows=len(flat),
        mean=flat.mean(),
        t_open=t_open,
        t_exec=t_exec,
    )


if __name__ == "__main__":
    main()
