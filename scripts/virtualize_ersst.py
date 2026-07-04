#!/usr/bin/env python3
# Build a VirtualiZarr / kerchunk *Parquet* reference over the local ERSST v5
# NetCDF granules, in the layout our VirtualStoreAdapter reads:
#   <out>/.zmetadata           (consolidated Zarr v2 metadata)
#   <out>/<var>/refs.N.parq    (columns: path, offset, size, raw)
#
# Each monthly granule is one timestep; we concatenate along `time`. ERSST `sst`
# is contiguous + uncompressed, so each chunk is a raw byte range into the .nc.
#
# Usage:
#   uv run --with kerchunk --with h5py --with fsspec --with ujson \
#          --with fastparquet --with pandas --with numpy \
#          scripts/virtualize_ersst.py [IN_DIR] [OUT.parq] [LIMIT]
#
#   LIMIT (optional): only the first N granules — for quick format validation.
import sys, glob, os, json, shutil
from kerchunk.hdf import SingleHdf5ToZarr
from kerchunk.combine import MultiZarrToZarr
from fsspec.implementations.reference import LazyReferenceMapper
import fsspec

# Variables not needed downstream (and PSL's time_bnds virtualizes with a wrong
# time length). Dropped from the finished store so the reader sees only the cube.
DROP_VARS = {"time_bnds"}


def drop_vars(store_dir):
    for name in DROP_VARS:
        d = os.path.join(store_dir, name)
        if os.path.isdir(d):
            shutil.rmtree(d)
    zp = os.path.join(store_dir, ".zmetadata")
    if os.path.exists(zp):
        zm = json.load(open(zp))
        md = zm.get("metadata", zm)
        for k in [k for k in md if k.split("/")[0] in DROP_VARS]:
            del md[k]
        json.dump(zm, open(zp, "w"))

IN = sys.argv[1] if len(sys.argv) > 1 else "data/ersst_v5_psl.nc"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/ersst_v5.parq"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
# If set, record THIS path in the manifest instead of the local file — read the
# local copy for speed but stamp the remote URL so the reference is portable and
# reads straight from the source (e.g. NOAA over HTTPS). Byte offsets are internal
# to the file, so a local scan and the remote file agree.
SOURCE_URL = os.environ.get("ERSST_SOURCE_URL")

os.makedirs(OUT, exist_ok=True)
lfs = fsspec.filesystem("file")
out = LazyReferenceMapper.create(root=OUT, fs=lfs, record_size=100_000)

if os.path.isfile(IN):
    # Single aggregated HDF5 file (PSL): already a complete time series — scan
    # once, write refs straight to the Parquet store. Absolute local path so the
    # manifest points at this file (Stage 1); later stages rewrite the prefix.
    ap = os.path.abspath(IN)
    recorded = SOURCE_URL or ap
    print(f"virtualizing single aggregated file: read {ap}, record {recorded}")
    # Pass a file OBJECT (not a path) so kerchunk records `url` instead of the
    # local path — this is what lets us stamp the remote source URL.
    with open(ap, "rb") as f:
        SingleHdf5ToZarr(f, url=recorded, out=out, inline_threshold=5000).translate()
    out.flush()
    drop_vars(OUT)
else:
    # Directory of per-month granules: scan each, concatenate along time.
    files = sorted(glob.glob(os.path.join(IN, "ersst.v5.*.nc")))
    if LIMIT:
        files = files[:LIMIT]
    if not files:
        sys.exit(f"no granules found in {IN}")
    print(f"scanning {len(files)} granules from {IN}")
    singles = []
    for i, f in enumerate(files):
        singles.append(SingleHdf5ToZarr(os.path.abspath(f), inline_threshold=5000).translate())
        if (i + 1) % 250 == 0:
            print(f"  scanned {i+1}/{len(files)}")
    MultiZarrToZarr(
        singles,
        concat_dims=["time"],
        identical_dims=["lev", "lat", "lon"],
        out=out,
    ).translate()
    out.flush()

# 3) Report what landed.
zmeta = os.path.join(OUT, ".zmetadata")
n_refs = len(glob.glob(os.path.join(OUT, "*", "refs.*.parq")))
print(f"wrote {OUT}: .zmetadata={'yes' if os.path.exists(zmeta) else 'MISSING'}, "
      f"{n_refs} refs.parq files across arrays")
print("arrays:", sorted(d for d in os.listdir(OUT) if os.path.isdir(os.path.join(OUT, d))))
