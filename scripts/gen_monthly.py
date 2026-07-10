# Tiny multi-month timestamp fixture for periodic-GROUP-BY tests.
# time = 6 monthly noon-of-the-15th timestamps (2020-01..2020-06), lat/lon = 1 each,
# temperature[t] = t*10 (deterministic). Written as Zarr v3 with CF time so the
# reader decodes `time` to an Arrow Timestamp.
import numpy as np, pandas as pd, xarray as xr
time = pd.to_datetime([f"2020-{m:02d}-15T12:00:00" for m in range(1, 7)])
ds = xr.Dataset(
    {"temperature": (("time", "lat", "lon"), (np.arange(6) * 10).reshape(6, 1, 1).astype("float64"))},
    coords={"time": time, "lat": [0.0], "lon": [0.0]},
)
import shutil, os
out = "data/monthly_v3.zarr"
if os.path.exists(out):
    shutil.rmtree(out)
ds.to_zarr(out, zarr_format=3, mode="w")
print("wrote", out, "months", [t.month for t in time])
