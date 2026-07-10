# GeoTIFF / COG Support — Design Note

Assessment of adding GeoTIFF / Cloud-Optimized GeoTIFF (COG) support to
zarr-datafusion. Two routes are considered: a **native Rust reader** vs.
**routing through VirtualiZarr**. Captured 2026-07-10.

## How a COG maps onto the current model

zarr-datafusion flattens an nD Cartesian grid of 1-D coordinates into a 2-D
table, prunes by chunk, and reads only touched chunks (`touched_tiles` in
`optimizer/cardinality/backend/product.rs`; `ZarrArrayMeta.chunks`). A COG is
structurally the same shape:

| Zarr concept | COG equivalent |
|---|---|
| nD data variable | raster band (2-D `y × x`) |
| chunk grid (`chunks`) | internal **tiles** (`TileOffsets` / `TileByteCounts` in the IFD) |
| 1-D coordinate arrays | `x` / `y` derived from the affine geotransform (origin + pixel scale) |
| multiple data vars | multiple bands |
| `touched_chunks` pruning | touched-tile pruning — here it's *exact* (coordinate → pixel is an inverse-affine, not a search) |

So the optimizer, filter/projection pushdown, and streaming machinery largely
carry over. Tiles = chunks.

---

## Route A — Native Rust reader

### What we'd build
1. **Format detection** — branch alongside `detect_zarr_version` / icechunk
   detection (`datasource/zarr.rs`): `.tif`/`.tiff`, TIFF magic
   (`II*\0`/`MM\0*`), GeoTIFF GeoKeys, BigTIFF variant.
2. **A TIFF decoder** — the read path is built entirely on `zarrs` +
   `zarrs_object_store`; none of it decodes TIFF. `tiff`/`image-tiff` (pure
   Rust, tiled TIFF + LZW/DEFLATE/PackBits + subset of predictors) vs. GDAL
   bindings (full coverage, heavy C dependency). Must be **feature-flagged**,
   mirroring the `icechunk` feature, to keep the default build lean.
3. **Metadata → `ZarrStoreMeta`** — parse IFDs (width/height, tile size, band
   count, sample format, nodata) + GeoKeys (CRS, affine). Synthesize `x`/`y`
   coordinate arrays from the transform (new path; current inference assumes
   stored 1-D arrays with `coord_min_max`).
4. **Tile read path** — parallel to `zarr_reader.rs`: given a tile window,
   issue `object_store` byte-range reads (existing S3/GCS/HTTP + `tracked_store`
   plumbing reuses directly), decompress per TIFF codec, undo predictor,
   materialize Arrow.
5. **Filter → tile mapping** — inverse-affine turns a coordinate predicate into
   a pixel/tile range; slots into existing touched-tile logic.

### Barriers (native route), worst first
1. **Single COG = one 2-D scene, no time.** Real value is *collections* — HLS,
   Landsat, Sentinel-2 (see [NASA cookbook note](nasa-cookbook-adoption.md)) are
   thousands of COGs. A time cube needs a **STAC/multi-file stacking layer** —
   a substantial new build.
2. **Projected CRS ≠ lat/lon.** COGs are usually UTM / Web Mercator; `WHERE lat
   = …` needs **reprojection (PROJ)** — new native dependency + correctness
   surface. v0 could expose native `x`/`y` only.
3. **Rotated / full-affine rasters break the model.** The 1-D-coordinate-per-
   axis assumption only holds for **north-up** transforms; rotation/skew makes
   `x`/`y` non-separable → 2-D coordinate columns the schema doesn't support.
4. **Codec/predictor coverage.** Pure-Rust `tiff` covers a subset; JPEG / WebP /
   LERC / ZSTD tiles and float/horizontal predictors are spotty. GDAL fixes it
   at the cost of the heavy dependency.
5. **Overviews/pyramids** have no model concept — default to full-res, ignore.
6. **Minor:** nodata + mask bands → Arrow nulls (easy); `STATISTICS` tags could
   feed the count/minmax optimizer (nice-to-have).

---

## Route B — via VirtualiZarr (preferred)

Routing through VirtualiZarr re-splits the work: the hard container/geospatial
parts move to a **Python authoring step** (mature tooling: rasterio / rioxarray
/ tifffile + PROJ), and the Rust side just consumes a virtual Zarr store — a
path this repo **already ships** (`virtual_store.rs`, `discover_arrays_from_json`,
`infer_schema_from_zmetadata_json`, icechunk, `parquet_refs.rs`). Almost every
native-route barrier stops being a Rust problem.

### What each barrier becomes

| Native-route barrier | VirtualiZarr verdict |
|---|---|
| Need a TIFF decoder (IFD parsing, tile grid, tags) | **Relieved.** tifffile/rasterio parse the container once at authoring → chunk manifest `(uri, offset, length)`. No IFD code in Rust. |
| Schema inference / new `ZarrStoreMeta` path | **Relieved.** VirtualiZarr emits standard `.zarray`/`.zattrs`; existing `discover_arrays_from_json` ingests it unchanged. No new detection branch. |
| Coordinate synthesis from affine | **Relieved.** rioxarray materializes real 1-D `x`/`y` coordinate arrays; the "coords are 1-D arrays" model just works. |
| Projected CRS ≠ lat/lon (PROJ) | **Relieved / pushed to authoring.** `rioxarray.reproject` before writing, or expose native `x`/`y`. No PROJ in Rust. |
| Rotated / full-affine rasters | **Mostly relieved.** Authoring resamples to north-up or refuses; Rust only sees separable 1-D coords. |
| Single COG = no time; STAC mosaic | **Relieved — headline win.** Concatenating thousands of COGs along time/band into one cube *is* VirtualiZarr's core job (`xr.concat` → icechunk); the exact CIRA/MERRA-2 pattern already read. No Rust mosaic layer. |
| Overviews/pyramids | **Relieved.** Authoring references only the chosen level. |
| BigTIFF, nodata/masks | **Relieved.** tifffile handles BigTIFF; nodata → `_FillValue` → existing fill-value handling. |

### The one residual: tile decompression codec

VirtualiZarr records byte ranges but **does not decode pixels** — chunk bytes
are still the raw *TIFF-compressed* tile. So the tile's compression must map to
a codec in the existing pipeline (`physical_plan/codec.rs`):

- **DEFLATE / ZSTD / uncompressed COGs** → map straight to zarr `gzip`/`zstd`
  codecs → **work end-to-end today, zero new Rust code.** Large fraction of real
  COGs.
- **LZW / PackBits / JPEG / WebP / LERC COGs** → no standard zarr codec → need a
  decode codec added. But this is now a *narrow extension point* (the composite-
  codec seam salvaged from the Ballista work), not a whole TIFF reader.
- **Predictor** (horizontal / float differencing) → a zarr `delta`-style filter;
  also a codec-pipeline addition.

Residual Rust work collapses from "TIFF reader + STAC mosaic + CRS" down to
"maybe add one or two decode codecs, only for non-DEFLATE COGs."

### The honest catch

Maturity gradient is inverted from HDF5/NetCDF: VirtualiZarr's HDF5/GRIB parsers
are battle-tested, but its **TIFF/COG authoring path is the least mature** part
of that ecosystem (the `async-tiff`/obstore COG work is newer). Risk shifts to
the Python authoring side — may need a small `tifffile → chunk-manifest` script
rather than turnkey. But that risk sits in throwaway authoring code, not the
shipped Rust library.

---

## Recommendation

Prefer **Route B (VirtualiZarr)**. It converts COG support from a new-reader
project (TIFF decoder + PROJ + affine coords + STAC mosaic, all feature-flagged
Rust) into, at most, a **codec-pipeline extension** — and for DEFLATE/ZSTD COGs,
into *nothing* on the Rust side. It also folds COGs into the same virtual-cube
story as the NASA HLS/Landsat cookbook datasets, giving one code path for
icechunk-virtualized HDF5, NetCDF, *and* GeoTIFF.

Suggested sequencing:
1. Spike: author a virtual Zarr / icechunk store from a single DEFLATE COG
   (tifffile/VirtualiZarr) and query it via the existing reader — likely works
   with no Rust changes.
2. Stack a STAC collection of COGs (Landsat/HLS) into a time cube; query with
   coordinate pushdown — the real milestone.
3. Only if needed: add LZW/JPEG decode codecs to `physical_plan/codec.rs` for
   COGs whose compression isn't DEFLATE/ZSTD.
