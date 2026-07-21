#!/usr/bin/env python3
"""Phase 2 exit check: did the Rust sink write the values it was given?

This is the outside-consumer half of the sink's validation. `cargo test --test
integration_writer` reads the store back through *our* reader, which is circular:
a writer and reader that share a mistaken axis or stride convention would agree
with each other perfectly. zarr-python does not share those conventions, and the
values are a *position-encoding formula* — so if the sink placed any value at the
wrong cell (an axis transpose, a stride error, a ragged-edge-chunk miscount), the
cell's value will not match the formula for that position and this check fails.

Expected (written by examples/write_filled.rs over time(7) x lat(10) x lon(12)):
    temperature[t,y,x] = 1000*t + 100*y + x        (int64, hole fill 0)
    reflectance[t,y,x] = temperature as f32,
                         except NaN at (t,y,x) = (1,2,3)   (float32, hole fill NaN)

Usage:
  uv run --with 'zarr>=3' --with numpy scripts/check_sink.py <store.zarr>

Exit 0 = every cell matches the formula, 1 = a mismatch (details printed).
"""

import sys

import numpy as np
import zarr

NT, NLAT, NLON = 7, 10, 12


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]

    try:
        z = zarr.open(path, mode="r")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not open {path}: {e}")
        return 1

    problems: list[str] = []

    t, y, x = np.indices((NT, NLAT, NLON))
    formula = (1000 * t + 100 * y + x).astype(np.int64)

    # temperature: exact match everywhere.
    temp = z["temperature"][:]
    if temp.shape != (NT, NLAT, NLON):
        problems.append(f"temperature shape {temp.shape} != {(NT, NLAT, NLON)}")
    elif not np.array_equal(temp, formula):
        n = int((temp != formula).sum())
        where = np.argwhere(temp != formula)[0]
        problems.append(
            f"temperature: {n} cells off-formula; first at {tuple(where)} "
            f"= {temp[tuple(where)]}, expected {formula[tuple(where)]}"
        )

    # reflectance: formula as float32, with a single NaN hole at (1,2,3).
    refl = z["reflectance"][:]
    expected = formula.astype(np.float32)
    expected[1, 2, 3] = np.nan
    # NaN-aware comparison: both NaN, or both equal.
    both_nan = np.isnan(refl) & np.isnan(expected)
    equal = refl == expected
    ok = both_nan | equal
    if not ok.all():
        n = int((~ok).sum())
        where = np.argwhere(~ok)[0]
        problems.append(
            f"reflectance: {n} cells off-formula; first at {tuple(where)} "
            f"= {refl[tuple(where)]}, expected {expected[tuple(where)]}"
        )
    if not np.isnan(refl[1, 2, 3]):
        problems.append(f"reflectance NaN hole missing at (1,2,3): {refl[1, 2, 3]}")

    if problems:
        print(f"FAIL: {path}")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"PASS: sink output matches the formula at every cell ({path})")
    print(f"       temperature and reflectance over {(NT, NLAT, NLON)}, NaN hole at (1,2,3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
