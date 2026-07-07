# Generate a small, fully-local, native-chunk icechunk store for Milestone 1.
#
# No virtual chunks / no S3 — this isolates "can Rust read the icechunk format"
# from the harder virtual-HDF5 (M2) and remote-GCS (M3) paths. Mirrors the
# existing synthetic Zarr shape: time(7), lat(10), lon(10), temperature(7,10,10).
#
# Values are deterministic so the Rust reader can verify exactly:
#   temperature[t, y, x] = t*100 + y*10 + x
#
# Run:
#   uv run --with 'icechunk==2.1.0' --with 'zarr>=3' --with numpy \
#     scripts/gen_synthetic_icechunk.py data/synthetic.icechunk
import sys
import numpy as np
import icechunk
import zarr

out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic.icechunk"

storage = icechunk.local_filesystem_storage(out)
repo = icechunk.Repository.open_or_create(storage)
session = repo.writable_session("main")
store = session.store

root = zarr.group(store=store, overwrite=True)

nt, ny, nx = 7, 10, 10

time = root.create_array("time", shape=(nt,), dtype="int64", chunks=(nt,))
time[:] = np.arange(nt, dtype="int64")

lat = root.create_array("lat", shape=(ny,), dtype="float64", chunks=(ny,))
lat[:] = np.linspace(-90.0, 90.0, ny)

lon = root.create_array("lon", shape=(nx,), dtype="float64", chunks=(nx,))
lon[:] = np.linspace(0.0, 360.0, nx, endpoint=False)

# chunked along time so the store has multiple native chunks (7 chunks)
temp = root.create_array(
    "temperature", shape=(nt, ny, nx), dtype="float64", chunks=(1, ny, nx)
)
t_idx, y_idx, x_idx = np.meshgrid(
    np.arange(nt), np.arange(ny), np.arange(nx), indexing="ij"
)
temp[:] = (t_idx * 100 + y_idx * 10 + x_idx).astype("float64")

snapshot = session.commit("initial synthetic data")
print(f"committed snapshot {snapshot} to {out}")
print(f"  time={nt} lat={ny} lon={nx} temperature shape=({nt},{ny},{nx})")
print(f"  temperature[3,4,5] should be {3*100 + 4*10 + 5} = 345.0")
