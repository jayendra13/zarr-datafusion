#!/usr/bin/env python3
"""Generate test Zarr datasets for zarr-datafusion.

Creates 8 dataset variations:
- Synthetic: v2, v2_blosc, v3, v3_blosc
- ERA5: v2, v2_blosc, v3, v3_blosc

Codecs use Blosc with LZ4 compression.
"""

import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc

# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("data")

# Blosc compressor with LZ4
BLOSC_LZ4 = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)

# ERA5 settings
ERA5_GCS_URL = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
ERA5_VARIABLES = ["geopotential", "temperature"]
ERA5_DATE = "2025-01-01"
ERA5_NUM_TIMESTAMPS = 3
ERA5_NUM_LEVELS = 2

ZarrVersion = Literal[2, 3]


def get_store_path(
    dataset: Literal["synthetic", "era5"],
    version: ZarrVersion,
    with_codecs: bool,
) -> Path:
    """Get the output path for a dataset variation."""
    suffix = "_blosc" if with_codecs else ""
    return DATA_DIR / f"{dataset}_v{version}{suffix}.zarr"


# -----------------------------
# Synthetic Data Generation
# -----------------------------
def generate_synthetic(
    version: ZarrVersion,
    with_codecs: bool,
    nlat: int = 10,
    nlon: int = 10,
    ntime: int = 7,
    seed: int = 42,
) -> None:
    """Generate synthetic weather data.

    Args:
        version: Zarr format version (2 or 3)
        with_codecs: Whether to use Blosc/LZ4 compression
        nlat: Number of latitude points
        nlon: Number of longitude points
        ntime: Number of time steps
        seed: Random seed for reproducibility
    """
    store_path = get_store_path("synthetic", version, with_codecs)
    np.random.seed(seed)

    compressor = BLOSC_LZ4 if with_codecs else None
    codec_desc = "Blosc/LZ4" if with_codecs else "no codecs"

    if version == 3:
        store = zarr.storage.LocalStore(str(store_path))
        root = zarr.group(store=store, overwrite=True, zarr_format=3)

        if with_codecs:
            # Zarr v3 uses compressors parameter
            from zarr.codecs import BloscCodec

            compressors = BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")
        else:
            compressors = None

        # Create coordinate arrays
        root.create_array("lat", data=np.arange(nlat), compressors=compressors)
        root.create_array("lon", data=np.arange(nlon), compressors=compressors)
        root.create_array("time", data=np.arange(ntime), compressors=compressors)

        # Create data variables
        temperature = root.create_array(
            "temperature",
            chunks=(1, nlat, nlon),
            data=np.random.randint(-50, 60, (ntime, nlat, nlon)),
            compressors=compressors,
        )
        humidity = root.create_array(
            "humidity",
            chunks=(1, nlat, nlon),
            data=np.random.randint(10, 80, (ntime, nlat, nlon)),
            compressors=compressors,
        )
    else:
        # Zarr v2
        root = zarr.open_group(str(store_path), mode="w", zarr_format=2)

        root.create_array("lat", data=np.arange(nlat), compressor=compressor)
        root.create_array("lon", data=np.arange(nlon), compressor=compressor)
        root.create_array("time", data=np.arange(ntime), compressor=compressor)

        temperature = root.create_array(
            "temperature",
            chunks=(1, nlat, nlon),
            data=np.random.randint(-50, 60, (ntime, nlat, nlon)),
            compressor=compressor,
        )
        humidity = root.create_array(
            "humidity",
            chunks=(1, nlat, nlon),
            data=np.random.randint(10, 80, (ntime, nlat, nlon)),
            compressor=compressor,
        )

    # Add metadata
    root.attrs["title"] = "Weekly Weather Sample"
    root.attrs["conventions"] = f"Zarr v{version}"
    temperature.attrs.update({"units": "K", "long_name": "Air Temperature"})
    humidity.attrs.update({"units": "%", "long_name": "Relative Humidity"})

    print(f"  Written: {store_path} (v{version}, {codec_desc})")


# -----------------------------
# ERA5 Data Download
# -----------------------------
def generate_era5_all_variations(
    gcs_url: str = ERA5_GCS_URL,
    variables: list[str] = ERA5_VARIABLES,
    date: str = ERA5_DATE,
    num_timestamps: int = ERA5_NUM_TIMESTAMPS,
    num_levels: int = ERA5_NUM_LEVELS,
) -> None:
    """Download ERA5 data once and create all 4 variations.

    Downloads to temp store in native format, loads into memory,
    then writes all variations (v2, v2_blosc, v3, v3_blosc).

    Args:
        gcs_url: GCS URL for ERA5 Zarr store
        variables: List of variables to download
        date: Date to select (YYYY-MM-DD)
        num_timestamps: Number of hourly timestamps to download
        num_levels: Number of hybrid levels (from surface)
    """
    from zarr.codecs import BloscCodec

    temp_path = DATA_DIR / "era5_temp.zarr"

    # Step 1: Download once in native format
    print("  Opening ERA5 store...")
    ds = xr.open_zarr(
        gcs_url,
        chunks="auto",
        storage_options={"token": "anon"},
    )

    time_slice = ds.time.sel(time=date)[:num_timestamps]
    subset = ds[variables].sel(
        time=time_slice,
        hybrid=ds.hybrid[-num_levels:],
    )
    print(f"  Subset shape: {dict(subset.sizes)}")

    print("  Downloading to temp store...")
    with ProgressBar():
        subset.to_zarr(str(temp_path), mode="w", zarr_format=2)

    # Step 2: Load into memory
    print("  Loading into memory...")
    data = xr.open_zarr(str(temp_path)).load()

    all_vars = list(data.data_vars) + list(data.coords)

    # Step 3: Write all 4 variations
    # v2 without codecs
    path = get_store_path("era5", 2, False)
    encoding = {var: {"compressor": None} for var in all_vars}
    data.to_zarr(str(path), mode="w", zarr_format=2, encoding=encoding)
    print(f"  Written: {path} (v2, no codecs)")

    # v2 with Blosc/LZ4
    path = get_store_path("era5", 2, True)
    encoding = {var: {"compressor": BLOSC_LZ4} for var in all_vars}
    data.to_zarr(str(path), mode="w", zarr_format=2, encoding=encoding)
    print(f"  Written: {path} (v2, Blosc/LZ4)")

    # v3 without codecs
    path = get_store_path("era5", 3, False)
    encoding = {var: {} for var in all_vars}
    data.to_zarr(str(path), mode="w", zarr_format=3, encoding=encoding)
    print(f"  Written: {path} (v3, no codecs)")

    # v3 with Blosc/LZ4
    path = get_store_path("era5", 3, True)
    encoding = {
        var: {"compressors": BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")}
        for var in all_vars
    }
    data.to_zarr(str(path), mode="w", zarr_format=3, encoding=encoding)
    print(f"  Written: {path} (v3, Blosc/LZ4)")

    # Step 4: Cleanup
    print("  Cleaning up temp data...")
    shutil.rmtree(temp_path)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """Generate all test datasets."""
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Zarr version: {zarr.__version__}")
    print()

    # Generate all synthetic variations
    print("=" * 60)
    print("SYNTHETIC DATASETS")
    print("=" * 60)
    for version in [2, 3]:
        for with_codecs in [False, True]:
            generate_synthetic(version, with_codecs)
    print()

    # Download ERA5 once and create all variations
    print("=" * 60)
    print("ERA5 DATASETS")
    print("=" * 60)
    generate_era5_all_variations()
    print()

    print("=" * 60)
    print("DONE - Generated 8 dataset variations")
    print("=" * 60)
    print()
    print("Datasets created:")
    for dataset in ["synthetic", "era5"]:
        for version in [2, 3]:
            for with_codecs in [False, True]:
                path = get_store_path(dataset, version, with_codecs)
                print(f"  {path}")


if __name__ == "__main__":
    main()
