# Milestone 2: generate a LOCAL icechunk store whose chunks are VIRTUAL
# references into a local NetCDF4/HDF5 file. Proves the virtual-HDF5 decode path
# before the 7.5 GB remote CIRA files (M3). Mirrors zarrs_icechunk's own
# `virtualizarr_netcdf` example, but everything is local (file://, no network).
#
# Deterministic values so the Rust reader can verify exactly:
#   data[z, y, x] = z*100 + y*10 + x
#
# Run:
#   uv run --with 'icechunk==2.1.0' --with 'virtualizarr[hdf,icechunk]==2.7.0' \
#     --with xarray --with h5netcdf --with obstore \
#     scripts/gen_virtual_icechunk.py
import os
import numpy as np
import xarray as xr
import icechunk
from obstore.store import LocalStore
from obspec_utils.registry import ObjectStoreRegistry
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser

data_dir = os.path.abspath("data")
src = os.path.join(data_dir, "m2_source.nc")
store_path = os.path.join(data_dir, "m2.icechunk")

# 1. Write a small source NetCDF4/HDF5 file.
nz, ny, nx = 4, 5, 6
z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
vals = (z * 100 + y * 10 + x).astype("float64")
xr.Dataset({"data": (("z", "y", "x"), vals)}).to_netcdf(src, engine="h5netcdf")
print(f"wrote source HDF5 {src}  data[3,4,5]={3*100+4*10+5}")

# 2. Virtualize it: build a manifest of chunk -> (file, offset, length) refs.
registry = ObjectStoreRegistry({"file://": LocalStore()})
url = f"file://{src}"
vds = open_virtual_dataset(url, parser=HDFParser(), registry=registry)
print("virtual dataset:", dict(vds.sizes))

# 3. Write an icechunk store that keeps those refs virtual, with a local
#    (file://) virtual-chunk container so the bytes are fetched from the .nc.
storage = icechunk.local_filesystem_storage(store_path)
config = icechunk.RepositoryConfig.default()
# file:// container prefix must be a canonical path ending in '/'. virtualizarr
# writes absolute refs (file:///abs/path/m2_source.nc), so the container that owns
# them is the source directory; the store roots at '/'.
container_prefix = f"file://{data_dir}/"
print(f"virtual chunk container prefix: {container_prefix}")
config.set_virtual_chunk_container(
    icechunk.VirtualChunkContainer(container_prefix, icechunk.local_filesystem_store("/"))
)
repo = icechunk.Repository.create(storage=storage, config=config)
session = repo.writable_session(branch="main")
vds.virtualize.to_icechunk(session.store)
snapshot = session.commit("virtual netcdf")
print(f"committed {snapshot} to {store_path}")
