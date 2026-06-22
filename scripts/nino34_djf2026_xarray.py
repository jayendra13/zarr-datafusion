"""
Compute the Nino 3.4 SST query against the local ERA5 Zarr store using xarray.

Mirrors this SQL (run via zarr-datafusion):

    SELECT
      AVG(sea_surface_temperature - 273.15) AS sst_celsius,
      COUNT(*)                              AS n_cells
    FROM era5
    WHERE time      BETWEEN '2025-12-31 06:00:00' AND '2025-12-31 17:00:00'
      AND latitude  BETWEEN -5.0  AND 5.0
      AND longitude BETWEEN 190.0 AND 240.0;

The window is the Nino 3.4 box (5S-5N, 170W-120W = 190E-240E) for the 12
hourly steps 06:00-17:00 UTC on 2025-12-31. All required chunks are present
in data/era5_sst_local.zarr (downloaded via download_sst_local.py).

Notes on this sparse local store:
  * It is a partial download. ERA5's final time chunk is a partial edge chunk
    that zarr-python 3.x mishandles (reshape error), and merely building an
    index over the `time` coordinate reads it. So we drop the `time`
    coordinate and select positionally: the array is contiguous hourly data
    where position i == "hours since 1900-01-01" == the downloader's hour
    index, so isel(time=slice(T_LO, T_HI+1)) is exact and never touches the
    broken chunk.

Run:
    uv run --with xarray --with zarr --with numpy --with dask \
           scripts/nino34_djf2026_xarray.py
"""

from datetime import datetime, timezone

import numpy as np
import xarray as xr

STORE = "data/era5_sst_local.zarr"
EPOCH = datetime(1900, 1, 1, tzinfo=timezone.utc)


def hours_since_1900(y, mo, d, hr):
    dt = datetime(y, mo, d, hr, tzinfo=timezone.utc)
    return int((dt - EPOCH).total_seconds() // 3600)


# Time window expressed in the raw coordinate units (hours since 1900-01-01).
T_LO = hours_since_1900(2025, 12, 31, 6)   # 2025-12-31 06:00:00
T_HI = hours_since_1900(2025, 12, 31, 17)  # 2025-12-31 17:00:00

# Open the partial store, dropping the broken `time` coordinate (see docstring).
ds = xr.open_zarr(STORE, consolidated=True, decode_times=False,
                  drop_variables=["time"])
sst = ds["sea_surface_temperature"]

# Time: positional slice (position i == hours since 1900-01-01).
sst = sst.isel(time=slice(T_LO, T_HI + 1))

# Space: value-based masks on the (small, fully-present) lat/lon coordinates.
# ERA5 latitude is descending (90 -> -90); longitude is 0..360.
lat_mask = (sst.latitude >= -5.0) & (sst.latitude <= 5.0)
lon_mask = (sst.longitude >= 190.0) & (sst.longitude <= 240.0)

window = sst.where(lat_mask & lon_mask, drop=True).load()

# Human-readable timestamps for the report.
time_labels = np.array(
    [np.datetime64("1900-01-01T00:00:00") + np.timedelta64(h, "h")
     for h in range(T_LO, T_HI + 1)]
)

# --- Aggregates ---
celsius = window - 273.15

total_cells = int(window.size)                       # COUNT(*) -- every cell
valid_cells = int(window.notnull().sum())            # non-NaN (ocean) cells
sst_celsius_ocean = float(celsius.mean(skipna=True))  # nanmean over ocean cells
sst_celsius_all = float(celsius.mean())              # nan if any land cell present

print("Nino 3.4 window  (2025-12-31 06:00-17:00 UTC, 5S-5N, 190E-240E)")
print("=" * 64)
print(f"time steps      : {int(window.time.size)}  "
      f"({time_labels[0]} .. {time_labels[-1]} UTC)")
print(f"lat points      : {window.latitude.size}")
print(f"lon points      : {window.longitude.size}")
print(f"total cells     : {total_cells}")
print(f"ocean cells     : {valid_cells}  (non-NaN)")
print(f"land/NaN cells  : {total_cells - valid_cells}")
print("-" * 64)
print(f"AVG sst_celsius (skipna)  : {sst_celsius_ocean:.4f} C")
print(f"AVG sst_celsius (raw mean): {sst_celsius_all}")
print()
print("SQL-equivalent result (COUNT(*) counts all cells):")
print(f"  sst_celsius = {sst_celsius_ocean:.4f},  n_cells = {total_cells}")
