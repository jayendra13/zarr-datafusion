"""Q2 — diurnal climatology over a CONUS box. Counterpart of q2_diurnal.sql.

Shape under test: a grouped aggregation — "the average day" per grid cell, which
is `da.groupby('time.hour').mean()` in xarray and `GROUP BY lat, lon, hour` in SQL.
This is the workload the whole flatten-nD-to-a-table model is built for, and the
one where a `GROUP BY` replaces a rechunk.

Equivalent SQL (see bench/sql/q2_diurnal.sql): a 72-timestep window over a CONUS
box, averaged by hour-of-day, giving 101 lat x 221 lon x 24 hours = 535,704 rows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from _common import Phase, coord_between, emit, open_era5  # noqa: E402

TIME_LO = "2020-06-01T00:00:00"
TIME_HI = "2020-06-03T23:00:00"
LAT_LO, LAT_HI = 25.0, 50.0
LON_LO, LON_HI = 235.0, 290.0


def main():
    p = Phase()
    ds = open_era5(xr, ["2m_temperature"])
    t_open = p.done()

    p = Phase()
    t2m = ds["2m_temperature"].sel(
        time=slice(TIME_LO, TIME_HI),
        latitude=coord_between(ds, "latitude", LAT_LO, LAT_HI),
        longitude=coord_between(ds, "longitude", LON_LO, LON_HI),
    )
    # GROUP BY lat, lon, hour  ->  groupby over hour-of-day, averaging the time axis.
    clim = (t2m - 273.15).groupby("time.hour").mean("time", skipna=False)

    # Flatten the (hour, lat, lon) cube into the same tabular result the SQL
    # returns, so both sides pay for materializing 535,704 rows rather than one
    # side stopping at a compact nD array.
    clim = clim.compute()
    stacked = clim.stack(cell=("hour", "latitude", "longitude"))
    values = stacked.values
    idx = stacked.indexes["cell"]
    df = pd.DataFrame(
        {
            "hour": idx.get_level_values("hour").astype("int32"),
            "lat": idx.get_level_values("latitude").astype("float64"),
            "lon": idx.get_level_values("longitude").astype("float64"),
            # ROUND(AVG(t2m_c), 3) on the SQL side — matched here so the two
            # fingerprints are comparable rather than differing in the 4th decimal.
            "t2m_mean_c": np.round(values, 3),
        }
    )
    # HAVING AVG(t2m_c) BETWEEN -100 AND 100 — drops any all-NaN group.
    df = df[np.isfinite(df["t2m_mean_c"])]
    df = df[(df["t2m_mean_c"] >= -100) & (df["t2m_mean_c"] <= 100)]
    t_exec = p.done()

    emit(
        "q2_diurnal",
        rows=len(df),
        mean=df["t2m_mean_c"].mean(),
        t_open=t_open,
        t_exec=t_exec,
    )


if __name__ == "__main__":
    main()
