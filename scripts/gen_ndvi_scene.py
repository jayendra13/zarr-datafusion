# /// script
# requires-python = ">=3.11"
# dependencies = ["xarray", "zarr>=3", "numpy", "aiohttp", "fsspec", "dask", "requests", "pystac-client"]
# ///
"""Fetch one Sentinel-2 L2A window and write a tidy local Zarr for the NDVI recipe.

Mirrors xarray-sql's benchmarks/geospatial/01_ndvi.py source resolution: search
the EOPF STAC catalog for a Sentinel-2 L2A scene, open the 10 m reflectance group,
and cut a 1024x1024 pixel window (red b04 + NIR b08).

The output store is shaped for zarr-datafusion's assumptions (1D coords + nD data
vars, coords sorted alphabetically, data dims in that same order). The source
bands are laid out [y, x]; we TRANSPOSE to [x, y] so the alphabetical coord order
(x, y) matches the data-variable dim order -- otherwise the reader would swap the
x/y column labels. NDVI itself is per-pixel and unaffected, but the coordinates
must be honest.

Run:  uv run scripts/gen_ndvi_scene.py
Out:  data/s2_ndvi_scene.zarr   (x[1024], y[1024], b04[1024,1024], b08[1024,1024])
"""
import numpy as np
import xarray as xr
from pystac_client import Client

STAC = "https://stac.core.eopf.eodc.eu"
BBOX = [7.2, 44.5, 7.4, 44.7]              # agricultural area near Turin, Italy
DATETIME = "2025-04-25/2025-05-05"
GROUP = "measurements/reflectance/r10m"
# Pixel window (matches the xarray-sql benchmark): 1024x1024 at (y=4000, x=6000).
Y0, X0, N = 4000, 6000, 1024
OUT = "data/s2_ndvi_scene.zarr"


def main() -> None:
    print(f"searching {STAC} for a sentinel-2-l2a scene over {BBOX} in {DATETIME}")
    catalog = Client.open(STAC)
    item = next(
        catalog.search(
            collections=["sentinel-2-l2a"], bbox=BBOX, datetime=DATETIME, max_items=1
        ).items()
    )
    print(f"resolved item: {item.id}")

    href = item.assets["product"].href
    print(f"opening group {GROUP!r} from consolidated store (no listing)")
    tree = xr.open_datatree(href, engine="zarr", chunks={})
    r10m = tree[GROUP].to_dataset()

    ys, xs = slice(Y0, Y0 + N), slice(X0, X0 + N)
    red = r10m["b04"].isel(y=ys, x=xs)   # dims (y, x)
    nir = r10m["b08"].isel(y=ys, x=xs)

    # Transpose y,x -> x,y so data dims follow alphabetical coord order (x, y).
    # xarray applies CF mask/scale, so bands are surface reflectance floats with
    # NaN over nodata (fill_value 0); keep them as float32 -- NDVI wants reflectance
    # and the reader maps NaN -> null the same way the climatology recipes rely on.
    red_xy = red.transpose("x", "y").values.astype("float32")
    nir_xy = nir.transpose("x", "y").values.astype("float32")
    x_vals = red["x"].values.astype("int64")   # UTM easting (m)
    y_vals = red["y"].values.astype("int64")   # UTM northing (m)

    print(f"window: b04/b08 {red_xy.shape} (x, y); loading {red_xy.nbytes/1e6:.0f} MB/band")
    ds = xr.Dataset(
        {
            "b04": (("x", "y"), red_xy),
            "b08": (("x", "y"), nir_xy),
        },
        coords={"x": ("x", x_vals), "y": ("y", y_vals)},
        attrs={
            "source_item": item.id,
            "source_group": GROUP,
            "window": f"y[{Y0}:{Y0+N}] x[{X0}:{X0+N}]",
            "note": "Sentinel-2 L2A surface reflectance (b04=red, b08=NIR), "
            "CF-scaled floats with NaN nodata; transposed to (x, y) for zarr-datafusion.",
        },
    )
    print(f"writing {OUT} (zarr v2)")
    ds.to_zarr(OUT, mode="w", zarr_format=2, consolidated=True)
    print("done")


if __name__ == "__main__":
    main()
