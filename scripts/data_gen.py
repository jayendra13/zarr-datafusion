#!/usr/bin/env python3
"""Generate test Zarr v3 datasets for zarr-datafusion.

Creates two datasets:
1. Synthetic weather data (small, for quick testing)
2. ERA5 climate data subset (real data from GCS)
"""

import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar


# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = Path("data")
SYNTHETIC_STORE = DATA_DIR / "synthetic.zarr"
ERA5_STORE = DATA_DIR / "era5.zarr"
ERA5_V2_TEMP = DATA_DIR / "era5_v2_temp.zarr"

ERA5_GCS_URL = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
ERA5_VARIABLES = ["geopotential", "temperature"]
ERA5_DATE = "2025-01-01"
ERA5_NUM_TIMESTAMPS = 3
ERA5_NUM_LEVELS = 2  # Near-surface hybrid levels


def generate_synthetic_data(
    store_path: Path,
    nlat: int = 10,
    nlon: int = 10,
    ntime: int = 7,
    seed: int = 42,
) -> None:
    """Generate synthetic weather data in Zarr v3 format.

    Args:
        store_path: Output path for Zarr store
        nlat: Number of latitude points
        nlon: Number of longitude points
        ntime: Number of time steps
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Create Zarr v3 store
    store = zarr.storage.LocalStore(str(store_path))
    root = zarr.group(store=store, overwrite=True, zarr_format=3)

    # Create coordinate arrays
    root.create_array("lat", data=np.arange(nlat))
    root.create_array("lon", data=np.arange(nlon))
    root.create_array("time", data=np.arange(ntime))

    # Create data variables
    temperature = root.create_array(
        "temperature",
        chunks=(1, nlat, nlon),
        data=np.random.randint(-50, 60, (ntime, nlat, nlon)),
    )

    humidity = root.create_array(
        "humidity",
        chunks=(1, nlat, nlon),
        data=np.random.randint(10, 80, (ntime, nlat, nlon)),
    )

    # Add metadata
    root.attrs["title"] = "Weekly Weather Sample"
    root.attrs["conventions"] = "Zarr v3"

    temperature.attrs.update({"units": "K", "long_name": "Air Temperature"})
    humidity.attrs.update({"units": "%", "long_name": "Relative Humidity"})

    print(f"Synthetic data written to: {store_path}")


def download_era5_subset(
    output_path: Path,
    temp_path: Path,
    gcs_url: str = ERA5_GCS_URL,
    variables: list[str] = ERA5_VARIABLES,
    date: str = ERA5_DATE,
    num_timestamps: int = ERA5_NUM_TIMESTAMPS,
    num_levels: int = ERA5_NUM_LEVELS,
) -> None:
    """Download ERA5 data subset from GCS and convert to Zarr v3.

    Downloads via Zarr v2 (for codec compatibility), then converts to v3.

    Args:
        output_path: Output path for Zarr v3 store
        temp_path: Temporary path for Zarr v2 intermediate
        gcs_url: GCS URL for ERA5 Zarr store
        variables: List of variables to download
        date: Date to select (YYYY-MM-DD)
        num_timestamps: Number of hourly timestamps to download
        num_levels: Number of hybrid levels (from surface)
    """
    print("Opening ERA5 store (metadata only)...")
    ds = xr.open_zarr(
        gcs_url,
        chunks="auto",
        storage_options={"token": "anon"},
    )

    print(f"Full dataset: {dict(ds.sizes)}")
    print(f"Selecting: {variables}, {date}, {num_timestamps} timestamps, {num_levels} levels")

    # Select subset
    time_slice = ds.time.sel(time=date)[:num_timestamps]
    subset = ds[variables].sel(
        time=time_slice,
        hybrid=ds.hybrid[-num_levels:],  # Near-surface levels
    )

    print(f"Subset shape: {dict(subset.sizes)}")

    # Step 1: Download to Zarr v2 (codec compatible)
    print("\nDownloading to temp Zarr v2...")
    with ProgressBar():
        subset.to_zarr(str(temp_path), mode="w", zarr_format=2)

    # Step 2: Load into memory
    print("Loading into memory...")
    data = xr.open_zarr(str(temp_path)).load()

    # Step 3: Write as Zarr v3 (clear encoding)
    print("Writing as Zarr v3...")
    encoding = {var: {} for var in list(data.data_vars) + list(data.coords)}
    data.to_zarr(str(output_path), mode="w", zarr_format=3, encoding=encoding)

    # Step 4: Cleanup
    print("Cleaning up temp files...")
    shutil.rmtree(temp_path)

    print(f"ERA5 data written to: {output_path}")


def main() -> None:
    """Generate all test datasets."""
    DATA_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("Generating synthetic weather data")
    print("=" * 50)
    generate_synthetic_data(SYNTHETIC_STORE)
    print(f"Zarr version: {zarr.__version__}")

    print()
    print("=" * 50)
    print("Downloading ERA5 data from GCS")
    print("=" * 50)
    download_era5_subset(ERA5_STORE, ERA5_V2_TEMP)

    print()
    print("=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
