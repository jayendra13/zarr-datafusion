#!/usr/bin/env python3
"""Phase 1 exit check: can xarray open a skeleton written by the Rust writer?

This is the half of the criterion our own tests cannot supply. `cargo test
--test integration_writer` proves the skeleton is readable by *us*, which is
circular: a writer and reader that share a mistaken axis convention would agree
with each other perfectly. xarray is the outside party that does not share our
conventions, so it is what makes "correct" mean something.

xarray requires dimension metadata, which the skeleton always writes as v3
`dimension_names` — so unlike the v2 fixtures (see scripts/compare_zarr.py), an
xarray-based check works here by construction.

Asserts:
  - the store opens at all
  - dims, coord values, and shapes match what was requested
  - every data variable reads back as its fill_value (no chunks were written)

Usage:
  uv run --with 'zarr>=3' --with xarray --with numpy scripts/check_skeleton.py <store.zarr>

Exit 0 = skeleton is well-formed, 1 = it is not (reasons printed).
"""

import sys

import numpy as np
import xarray as xr

EXPECTED_DIMS = {"time": 7, "lat": 10, "lon": 12}
EXPECTED_VARS = {"temperature": np.int64, "reflectance": np.float32}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]

    try:
        ds = xr.open_zarr(path, consolidated=False)
    except Exception as e:  # noqa: BLE001 - any failure to open is the failure
        print(f"FAIL: xarray could not open {path}: {e}")
        return 1

    problems: list[str] = []

    if dict(ds.sizes) != EXPECTED_DIMS:
        problems.append(f"dims {dict(ds.sizes)} != {EXPECTED_DIMS}")

    for name, size in EXPECTED_DIMS.items():
        if name not in ds.coords:
            problems.append(f"missing coordinate '{name}'")
            continue
        # Coordinate arrays are written in full, so they must hold real values.
        expected = np.arange(size)
        if not np.array_equal(ds.coords[name].values, expected):
            problems.append(f"coord '{name}' values != arange({size})")

    for name, dtype in EXPECTED_VARS.items():
        if name not in ds.data_vars:
            problems.append(f"missing data variable '{name}'")
            continue
        var = ds[name]
        if var.dims != ("time", "lat", "lon"):
            problems.append(f"'{name}' dims {var.dims} != ('time', 'lat', 'lon')")
        if var.dtype != dtype:
            problems.append(f"'{name}' dtype {var.dtype} != {dtype.__name__}")

        # No chunks were written, so everything must be the fill value.
        values = var.values
        if np.issubdtype(var.dtype, np.floating):
            if not np.isnan(values).all():
                problems.append(f"'{name}' should be all-NaN fill, got real values")
        elif (values != 0).any():
            problems.append(f"'{name}' should be all-zero fill, got nonzero values")

    if problems:
        print(f"FAIL: {path}")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"PASS: xarray opened {path}")
    print(f"       dims={dict(ds.sizes)}")
    print(f"       coords={list(ds.coords)}  data_vars={list(ds.data_vars)}")
    print("       all data variables read as fill_value")
    return 0


if __name__ == "__main__":
    sys.exit(main())
