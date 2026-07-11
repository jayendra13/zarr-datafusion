# Cookbook Adaptation Assessment

Assessment of external cookbooks/tutorials for expanding the zarr-datafusion
`cookbook/`. The bulk of this doc evaluates the [NASA Earthdata Cloud Cookbook](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/tutorials/)
(candidates, not yet shipped; captured 2026-07-10); the
[Shipped adaptations](#shipped-adaptations) section at the end tracks recipes
already adapted from other sources.

# NASA Openscapes Cookbook

Assessment of the NASA Earthdata Cloud Cookbook tutorials.

## Key structural fact

Almost all NASA data in the cookbook is **NetCDF-4/HDF5, not native Zarr**.
Cloud-native access is achieved by *virtualizing* those files into a
Zarr-compatible reference store via **Kerchunk** (the predecessor to
VirtualiZarr / icechunk). This is the same pattern we already use for CIRA
(virtual HDF5 references) and ERSST — so the most relevant tutorials map
directly onto infrastructure we have shipped.

## Adoptability tiers

### Tier 1 — direct fit, high value (Kerchunk / virtual-Zarr examples)

| Tutorial | Why it maps onto zarr-datafusion |
|---|---|
| **PO.DAAC ECCO SSH + Kerchunk** | NetCDF→virtual Zarr reference store on S3; same shape as the icechunk/CIRA path. `SELECT ssh WHERE time=… AND lat=… AND lon=…` is a near drop-in for the remote-icechunk flow. |
| **GES DISC MERRA-2 + Kerchunk** | Multi-file `MultiZarrToZarr` concat along time → large 4-D cube. Good stress test for coordinate filter pushdown + the OOM/streaming-by-variable NEXT item. |
| **ORNL DAYMET + Kerchunk** | Gridded daily climate cube; clean coordinate-subsetting demo. |
| **LP DAAC ECOSTRESS + Kerchunk** | Same mechanism, swath/LST data. |

These exercise the differentiator: pushing lat/lon/time equality filters down
into a virtualized remote store and reading only touched chunks — vs. what
xarray+dask do in these notebooks.

### Tier 2 — adoptable as SQL-rewrite demos (native gridded subsetting)

- **Earthdata Cloud Clinic**, **Hurricane Wind+SST** — both do
  `.sel(lat=slice, lon=slice, time=…)` on gridded nD arrays (SSH, SST,
  salinity, MERRA-2 winds). Each `.sel()` → a `WHERE` clause; spatial-mean →
  `AVG(...) GROUP BY`. Strong "xarray → SQL" side-by-side material.
- **Sea Level Rise** — time-series over a gridded SSH cube; spatial-mean-per-
  timestep → `GROUP BY time`. Showcases MIN/MAX/COUNT + cardinality optimizer.

### Tier 3 — informative, not directly adoptable

- **Zarr-EOSDIS-Store / COF Zarr Reformat** — reference for how NASA exposes
  *actual* Zarr endpoints; a genuine native-Zarr store would be a good
  non-icechunk test fixture.
- **Xarray Fundamentals** — conceptual; mirror examples as SQL equivalents in docs.

### Tier 4 — out of scope

Harmony subsetting, OPeNDAP, Pygeoweaver, MATLAB, ICESat-2 point-cloud/viz,
CMR / CMR-STAC discovery, authentication. Point clouds and server-side
subsetting don't fit the "flatten a Cartesian nD grid to 2D table" model.

## Recommendation

Highest-leverage adoption: a **cookbook recipe built on ECCO SSH or MERRA-2
Kerchunk**. Virtualize the NetCDF collection (VirtualiZarr/icechunk — already
supported), then run coordinate-filtered SQL and show it reads only the
selected chunks. It (a) reuses shipped infrastructure, (b) directly exercises
the current NEXT priority (OOM → streaming-by-variable; MERRA-2's
time-concatenated cube is large), and (c) gives an apples-to-apples "SQL vs
xarray+dask" story on a real NASA dataset.

**Caveat:** these tutorials authenticate to NASA Earthdata (netrc / EDL
tokens) and assume in-region `us-west-2` S3 access. Our remote reader currently
opens stores anonymously — EDL-credential support (or pre-staged public
mirrors) is needed before these run end-to-end.

# Shipped adaptations

Recipes already adapted from cookbooks *other* than NASA Openscapes.

## NDVI — xarray-sql geospatial benchmark

[`cookbook/ndvi/`](../cookbook/ndvi/) adapts
[xarray-sql's `benchmarks/geospatial/01_ndvi.py`](https://github.com/xqlsystems/xarray-sql/blob/main/benchmarks/geospatial/01_ndvi.py)
(sibling to the `02_climatology.py` we already mirror in `cookbook/climatology/`).
It computes NDVI `(b08 - b04) / (b08 + b04)` as one **per-pixel** SQL expression
over a 1024×1024 Sentinel-2 L2A window — the *projection* counterpart to the
aggregation recipes, matching xarray to 4 decimals.

**Relevance to the NASA caveat above.** The remote-read spike for this recipe
concretely reproduced the [Caveat](#recommendation)'s "our remote reader isn't
ready for these stores" warning — two distinct blockers any NASA Tier-1
(virtual-Zarr) or Tier-3 (native-Zarr endpoint) recipe will hit:

1. **HTTP object_store can't list a Ceph/S3 endpoint.** Pointing at a group with
   no group-level consolidated metadata forces a directory listing; the EODC
   endpoint answered `PROPFIND` with `405 Method Not Allowed`.
2. **Root-level consolidated `.zmetadata` spans the whole hierarchical product.**
   The EOPF product's only consolidated metadata sits at the root and mixes
   10/20/60 m arrays (different shapes, multiple `x`/`y`); the reader folds them
   into one Cartesian coord cube and overflows the shape product.

Resolution shipped: a local sample store written by
`scripts/gen_ndvi_scene.py`, which resolves the scene via the same STAC path as
the benchmark. Making these remote stores work directly (prefix-filtering a
consolidated `.zmetadata` down to one subgroup; an S3 backend with custom
endpoint + tenant + anonymous auth) is the reader work that also unblocks the
NASA Tier-1 recipes.

## Sources

- [xarray-sql geospatial benchmarks](https://github.com/xqlsystems/xarray-sql/tree/main/benchmarks/geospatial) — source of the shipped NDVI (`01_ndvi.py`) and climatology (`02_climatology.py`) recipes
- [Earthdata Cloud Clinic](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/tutorials/Earthdata-cloud-clinic.html)
- [GES DISC MERRA-2 Kerchunk](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/examples/GESDISC/GESDISC_MERRA2_tavg1_2d_flx_Nx__Kerchunk.html)
- [PO.DAAC ECCO SSH Kerchunk](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/examples/PODAAC/PODAAC_ECCO_SSH__Kerchunk.html)
- [LP DAAC ECOSTRESS Kerchunk](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/examples/LPDAAC/LPDAAC_ECOSTRESS_LSTE__Kerchunk.html)
- [COF Zarr Access via Reformat](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/external/cof-zarr-reformat.html)
- [earthdata-cloud-cookbook repo](https://github.com/NASA-Openscapes/earthdata-cloud-cookbook)
