"""Q3 — single-cell temperature timeseries, one month. Counterpart of q3_point.sql.

Shape under test: the needle-in-a-haystack access pattern, and deliberately our
worst case. ARCO-ERA5 is chunked one timestep per full lat x lon plane, so asking
for one grid cell across 744 hours means fetching 744 chunks of roughly 4 MB and
discarding all but one cell of each.

That cost is a property of how the data was written, not of the engine reading it
— which is exactly why this workload belongs in the comparison. xarray must fetch
the same planes, so if our number looks bad here, the honest conclusion is about
chunk layout rather than about either tool.

Equivalent SQL:

    SELECT time, "2m_temperature" - 273.15 AS t2m_c
    FROM era5
    WHERE time BETWEEN TIMESTAMP '2023-07-01T00:00:00Z'
                   AND TIMESTAMP '2023-07-31T23:00:00Z'
      AND latitude = 40.0 AND longitude = 280.0
    ORDER BY time;
"""

import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from _common import Phase, emit, open_era5  # noqa: E402

TIME_LO = "2023-07-01T00:00:00"
TIME_HI = "2023-07-31T23:00:00"
LAT = 40.0
LON = 280.0


def main():
    p = Phase()
    ds = open_era5(xr, ["2m_temperature"])
    t_open = p.done()

    p = Phase()
    # Exact coordinate equality, matching the SQL's `latitude = 40.0`. Both values
    # are exact multiples of 0.25, so they are representable and `method=None`
    # (exact) is safe — no nearest-neighbour snapping that the SQL does not do.
    series = ds["2m_temperature"].sel(
        time=slice(TIME_LO, TIME_HI),
        latitude=LAT,
        longitude=LON,
    )
    series = (series - 273.15).compute()
    # The time axis is already ascending, so the SQL's ORDER BY time is a no-op
    # here rather than a sort we skipped.
    values = series.values
    t_exec = p.done()

    emit(
        "q3_point",
        rows=len(values),
        mean=values.mean(),
        t_open=t_open,
        t_exec=t_exec,
    )


if __name__ == "__main__":
    main()
