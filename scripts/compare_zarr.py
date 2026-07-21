#!/usr/bin/env python3
"""External oracle for the Zarr write round trip.

Compares two Zarr stores WITHOUT sharing zarr-datafusion's reader assumptions.
This is the point: a round-trip test (`read(write(x)) == x`) is blind to
symmetric bugs — if the reader mislabels an axis and the writer mislabels it
back, the test goes green and the library is wrong. An oracle that re-derives
the mapping the same way our reader does would be equally blind, so this one
compares structure and bytes.

WHY zarr-python AND NOT xarray:
  xarray cannot open a Zarr v2 store that lacks `_ARRAY_DIMENSIONS` — it raises
  `KeyError: Zarr object is missing the attribute _ARRAY_DIMENSIONS ... which are
  required for xarray to determine variable dimensions`. Our v2 fixtures have no
  dimension metadata by design (that is the inference path we need to test), so
  an xarray-based oracle would only ever cover v3 — the easy path.

  The deeper reason: for a store with no dimension metadata there IS no in-store
  ground truth about which axis is lat and which is lon. That mapping is a
  convention, not data. The truth lives in the generator, so the oracle's job is
  to assert that B is structurally and bytewise identical to A — not to
  independently rediscover labels that were never written down.

What is compared by default (dataset identity):
  - the set of arrays
  - per array: shape, dtype, values (NaN-aware), fill_value, dimension names
    (v3 `dimension_names` / v2 `_ARRAY_DIMENSIONS`) when present in either store

Storage details (chunk grid, codecs) are NOT compared by default, because a
rechunk legitimately changes them while preserving dataset identity. Pass
--check-chunks to require an exact chunk-grid match (i.e. a true round trip).

Usage:
  uv run --with zarr --with numpy scripts/compare_zarr.py A.zarr B.zarr
  uv run --with zarr --with numpy scripts/compare_zarr.py --check-chunks A.zarr B.zarr
  uv run --with zarr --with numpy scripts/compare_zarr.py --self-test

Exit code 0 = identical, 1 = differences found (printed to stdout).
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr


def array_names(group) -> set[str]:
    """Names of every array in a group."""
    return {name for name, _ in group.arrays()}


def dim_names(arr) -> tuple[str, ...] | None:
    """Dimension names for a v3 (`dimension_names`) or v2 (`_ARRAY_DIMENSIONS`)
    array, or None when the store records none (the v2 inference case)."""
    v3 = getattr(getattr(arr, "metadata", None), "dimension_names", None)
    if v3 is not None:
        return tuple(v3)
    v2 = arr.attrs.get("_ARRAY_DIMENSIONS")
    if v2 is not None:
        return tuple(v2)
    return None


def values_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bytewise value equality, treating NaN as equal to NaN.

    `==` does not: NaN != NaN, so a plain comparison would report every NaN hole
    as a difference. `equal_nan` is float-only, hence the dtype branch.
    """
    if a.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def fill_value_equal(a, b) -> bool:
    """fill_value equality, NaN-aware (a NaN fill is the normal float case)."""
    fa, fb = a.fill_value, b.fill_value
    try:
        if fa is not None and fb is not None and np.isnan(fa) and np.isnan(fb):
            return True
    except (TypeError, ValueError):
        pass
    return bool(fa == fb) if fa is not None or fb is not None else True


def compare(
    a_path: str,
    b_path: str,
    check_chunks: bool = False,
    allow_added_dim_names: bool = False,
) -> list[str]:
    """Compare two Zarr stores. Returns a list of human-readable differences.

    `allow_added_dim_names` permits B to carry dimension names where A had none —
    the legitimate case of a v2 -> v3 copy, which always adds `dimension_names`.
    Names that are present on *both* sides must still match exactly.
    """
    errors: list[str] = []
    a_root = zarr.open_group(a_path, mode="r")
    b_root = zarr.open_group(b_path, mode="r")

    a_names, b_names = array_names(a_root), array_names(b_root)
    if a_names != b_names:
        if missing := a_names - b_names:
            errors.append(f"arrays missing from B: {sorted(missing)}")
        if extra := b_names - a_names:
            errors.append(f"arrays unexpected in B: {sorted(extra)}")

    for name in sorted(a_names & b_names):
        a, b = a_root[name], b_root[name]

        if a.shape != b.shape:
            errors.append(f"{name}: shape {a.shape} != {b.shape}")
            continue  # everything below is meaningless once shapes diverge

        if a.dtype != b.dtype:
            errors.append(f"{name}: dtype {a.dtype} != {b.dtype}")
            continue

        a_dims, b_dims = dim_names(a), dim_names(b)
        # A v2 -> v3 copy adds dimension names A never had; allow that enrichment,
        # but still require names to match when both stores carry them.
        added_names = allow_added_dim_names and a_dims is None and b_dims is not None
        if a_dims != b_dims and not added_names:
            errors.append(f"{name}: dimension names {a_dims} != {b_dims}")

        if check_chunks and a.chunks != b.chunks:
            errors.append(f"{name}: chunks {a.chunks} != {b.chunks}")

        if not fill_value_equal(a, b):
            errors.append(f"{name}: fill_value {a.fill_value!r} != {b.fill_value!r}")

        a_vals, b_vals = a[...], b[...]
        if not values_equal(a_vals, b_vals):
            if a_vals.dtype.kind == "f":
                diff = ~(
                    (a_vals == b_vals)
                    | (np.isnan(a_vals) & np.isnan(b_vals))
                )
            else:
                diff = a_vals != b_vals
            n = int(diff.sum())
            first = tuple(int(i) for i in np.argwhere(diff)[0])
            errors.append(
                f"{name}: {n}/{a_vals.size} values differ; "
                f"first at {first}: {a_vals[first]!r} != {b_vals[first]!r}"
            )

    return errors


def self_test() -> int:
    """Prove the oracle has teeth.

    An oracle that cannot fail is not an oracle. This builds the exact bug the
    round trip is blind to — a writer that swaps the two spatial axes — and
    asserts we reject it. It also asserts we accept an identical store, so the
    check is not merely rejecting everything.

    The swap is only detectable because lat (10) and lon (12) have DIFFERENT
    lengths. On the shared synthetic fixture (lat = lon = 10) this same swap
    produces matching shapes and would sail through. That is the whole reason
    this fixture exists.
    """
    tmp = Path(tempfile.mkdtemp(prefix="zarr_oracle_selftest_"))
    try:
        nt, nlat, nlon = 7, 10, 12
        rng = np.random.default_rng(0)
        data = rng.random((nt, nlat, nlon), dtype=np.float32)
        data[0, 0, 0] = np.nan  # a NaN must not by itself count as a difference

        def write(path: Path, values, dims):
            root = zarr.group(store=zarr.storage.LocalStore(str(path)),
                              overwrite=True, zarr_format=3)
            root.create_array("lat", data=np.arange(nlat), dimension_names=("lat",))
            root.create_array("lon", data=np.arange(nlon), dimension_names=("lon",))
            root.create_array("time", data=np.arange(nt), dimension_names=("time",))
            root.create_array("temperature", data=values,
                              dimension_names=dims, fill_value=np.nan)

        good = tmp / "good.zarr"
        swapped = tmp / "swapped.zarr"
        write(good, data, ("time", "lat", "lon"))
        # The bug: writer emits lon where lat belongs.
        write(swapped, data.transpose(0, 2, 1), ("time", "lon", "lat"))

        failures = []

        identical = compare(str(good), str(good))
        if identical:
            failures.append(f"FAIL: oracle rejected an identical store: {identical}")
        else:
            print("PASS: identical store accepted (NaN holes not flagged)")

        caught = compare(str(good), str(swapped))
        if not caught:
            failures.append("FAIL: oracle ACCEPTED a lat/lon-swapped store")
        else:
            print("PASS: lat/lon-swapped store rejected:")
            for e in caught:
                print(f"       - {e}")

        for f in failures:
            print(f)
        return 1 if failures else 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare two Zarr stores (external oracle for the write round trip).",
    )
    p.add_argument("a", nargs="?", help="reference store")
    p.add_argument("b", nargs="?", help="store under test")
    p.add_argument("--check-chunks", action="store_true",
                   help="also require an exact chunk-grid match (a true round trip; "
                        "omit when comparing across a deliberate rechunk)")
    p.add_argument("--allow-added-dim-names", action="store_true",
                   help="permit B to add dimension names A lacked (a v2 -> v3 copy); "
                        "names present on both sides must still match")
    p.add_argument("--self-test", action="store_true",
                   help="verify the oracle rejects a lat/lon-swapped store")
    args = p.parse_args()

    if args.self_test:
        return self_test()

    if not args.a or not args.b:
        p.error("two store paths are required (or --self-test)")

    errors = compare(
        args.a,
        args.b,
        check_chunks=args.check_chunks,
        allow_added_dim_names=args.allow_added_dim_names,
    )
    if errors:
        print(f"DIFFER: {args.a} vs {args.b}")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"IDENTICAL: {args.a} vs {args.b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
