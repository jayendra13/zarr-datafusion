"""Q1 — Niño-3.4 box mean SST, 24 hours. Python/xarray counterpart of q1_nino34.sql.

Shape under test: a narrow, highly selective filter over a huge archive. The whole
cost is deciding which chunks to touch and fetching only those; the arithmetic is
trivial. This is the case filter pushdown exists for.

Equivalent SQL:

    SELECT AVG(sea_surface_temperature - 273.15) AS sst_c
    FROM era5
    WHERE time      BETWEEN TIMESTAMP '2023-12-01T00:00:00Z'
                        AND TIMESTAMP '2023-12-01T23:00:00Z'
      AND latitude  BETWEEN  -5.0 AND   5.0
      AND longitude BETWEEN 190.0 AND 240.0;

Expect ~28.65 °C — a strong El Niño December.
"""

import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from _common import Phase, coord_between, emit, open_era5  # noqa: E402

TIME_LO = "2023-12-01T00:00:00"
TIME_HI = "2023-12-01T23:00:00"
LAT_LO, LAT_HI = -5.0, 5.0
LON_LO, LON_HI = 190.0, 240.0


def main():
    p = Phase()
    ds = open_era5(xr, ["sea_surface_temperature"])
    t_open = p.done()

    p = Phase()
    sst = ds["sea_surface_temperature"].sel(
        time=slice(TIME_LO, TIME_HI),
        latitude=coord_between(ds, "latitude", LAT_LO, LAT_HI),
        longitude=coord_between(ds, "longitude", LON_LO, LON_HI),
    )
    # skipna=False mirrors DataFusion's AVG, which propagates NaN rather than
    # skipping it. The Niño-3.4 box is all ocean, so no cell is masked and the
    # two agree — but only because that was checked, not by luck.
    value = float((sst - 273.15).mean(skipna=False).compute())
    t_exec = p.done()

    emit("q1_nino34", rows=1, mean=value, t_open=t_open, t_exec=t_exec)


if __name__ == "__main__":
    main()
