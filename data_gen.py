import zarr
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
store_path = "data/weather.zarr"
nlat = 10
nlon = 10
ntime = 7  # one week
seed = 42  # Random seed for reproducibility

np.random.seed(seed)

# -----------------------------
# Create Zarr v3 store + group
# -----------------------------
store = zarr.storage.LocalStore(store_path)
root = zarr.group(store=store, overwrite=True, zarr_format=3)

# -----------------------------
# Coordinates
# -----------------------------
lat = np.array(range(nlat))
lon = np.array(range(nlon))
timestamps = np.array(range(ntime))

# -----------------------------
# Create coordinate arrays
# (Zarr v3: data OR shape, not both)
# -----------------------------
root.create_array("lat", data=lat)
root.create_array("lon", data=lon)
root.create_array("time", data=timestamps)

# -----------------------------
# Create data variables
# -----------------------------
temperature = root.create_array(
    "temperature",
    chunks=(1, nlat, nlon),
    data = np.random.randint(-50,60, (ntime, nlat, nlon))
)

humidity = root.create_array(
    "humidity",
    chunks=(1, nlat, nlon),
    data= np.random.randint(10, 80, (ntime, nlat, nlon))
)

# -----------------------------
# Add minimal metadata
# -----------------------------
root.attrs["title"] = "Weekly Weather Sample"
root.attrs["conventions"] = "Zarr v3"

temperature.attrs.update({
    "units": "K",
    "long_name": "Air Temperature"
})

humidity.attrs.update({
    "units": "%",
    "long_name": "Relative Humidity"
})

print("Zarr v3 dataset written to:", store_path)
print("Zarr version:", zarr.__version__)
