# NASA Openscapes Cookbook — Adoption Assessment

Assessment of the [NASA Earthdata Cloud Cookbook](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/tutorials/)
tutorials for expanding the zarr-datafusion `cookbook/`. Captured 2026-07-10.

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

## Sources

- [Earthdata Cloud Clinic](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/tutorials/Earthdata-cloud-clinic.html)
- [GES DISC MERRA-2 Kerchunk](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/examples/GESDISC/GESDISC_MERRA2_tavg1_2d_flx_Nx__Kerchunk.html)
- [PO.DAAC ECCO SSH Kerchunk](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/examples/PODAAC/PODAAC_ECCO_SSH__Kerchunk.html)
- [LP DAAC ECOSTRESS Kerchunk](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/examples/LPDAAC/LPDAAC_ECOSTRESS_LSTE__Kerchunk.html)
- [COF Zarr Access via Reformat](https://nasa-openscapes.github.io/earthdata-cloud-cookbook/external/cof-zarr-reformat.html)
- [earthdata-cloud-cookbook repo](https://github.com/NASA-Openscapes/earthdata-cloud-cookbook)
