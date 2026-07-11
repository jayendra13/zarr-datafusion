# STAC Support — Design Assessment

Should zarr-datafusion add STAC support — i.e. **registering a table via STAC**
(`CREATE EXTERNAL TABLE … LOCATION 'stac://…'`)? Captured 2026-07-11, prompted by
the NDVI cookbook recipe, whose remote source (EOPF Sentinel-2) is discovered
through a STAC catalog.

## TL;DR

**Don't put STAC in the core / as a table-registration mechanism.** A thin
resolver is cheap but low-value; the version that *would* be valuable
(registering a whole collection as one cube) duplicates the virtual-Zarr
infrastructure we already ship and drags reprojection/mosaicking — out of scope
for a "flatten one grid to a table" engine — into the core. Keep STAC in the
**discovery / preprocessing layer** (next to `scripts/gen_ndvi_scene.py`), not
in the storage format path.

## Background: STAC vs. the long URL

STAC is a **discovery** API: a search (collection + bbox + datetime) returns
*items*, each with *assets* (hrefs). It is where `gen_ndvi_scene.py` gets the
Sentinel-2 store URL — but STAC only *hands back* the provider's real object key;
it doesn't shorten or lengthen it. The long EOPF path is EODC's Ceph bucket/key
layout plus the standard Sentinel-2 product name, not a STAC artifact. So
discovery (STAC) and storage (the Zarr store) are cleanly separable, which is the
whole reason STAC needn't live in the reader.

## Two very different "STAC supports"

### Level A — resolver: one STAC query → one href → open

```sql
CREATE EXTERNAL TABLE s STORED AS ZARR
  LOCATION 'stac://stac.core.eopf.eodc.eu/sentinel-2-l2a?bbox=...&datetime=...'
  OPTIONS ('asset' 'product', 'group' 'measurements/reflectance/r10m');
```

Essentially a URL-shortener. The entire benefit is avoiding a copy-paste of one
long href. It buys real problems in exchange:

- A **network catalog search at DDL/plan time** — latency, catalog auth, and a
  heavy STAC + HTTP-JSON dependency pulled into the core.
- **Non-determinism.** A search returns *N* items; `max_items=1` picks an
  arbitrary scene, and the catalog can return different results later — quietly
  breaking reproducibility, which is the property the local-sample cookbook
  recipes exist to guarantee.

Low value, awkward semantics.

### Level B — collection as a table: one STAC query → *many* items → one cube

The genuinely powerful idea: "every Sentinel-2 scene over this bbox across this
date range" as a single queryable table, with `time` / `tile` as added
dimensions. This is what STAC is actually *for*.

But it is a poor fit for **this** engine, for a structural reason: those items do
**not** form one Cartesian grid. Different scenes are different **UTM zones/tiles**,
different footprints, different times. Unioning them into a cube requires
**mosaicking, reprojection, and resampling** (warping across CRSs) — squarely the
"server-side subsetting / reprojection doesn't fit the flatten-a-grid model"
category already classified as **Tier 4 (out of scope)** in
[`cookbook-adaptation.md`](cookbook-adaptation.md). It would turn a Zarr↔SQL
bridge into a geospatial warp engine.

## The valuable version already has a home

"Many scattered files → one queryable cube" is exactly what **VirtualiZarr /
icechunk / kerchunk** do — which we already support (CIRA, ERSST). The idiomatic
pipeline is:

> STAC search → virtualize the collection into a reference store
> (`virtualizarr`, `odc-stac`, `stac-geoparquet`) → point the **existing** reader
> at that one store.

So STAC belongs beside the preprocessing generators, not in the reader. Building
Level B into the core would reimplement the concat/alignment the virtual-Zarr
path gives for free — and still leave the CRS/mosaic problem unsolved.

## Recommendation

- **Keep STAC as an optional helper, not a `LOCATION` scheme.** The cookbook
  pattern — Python resolves the href, the reader opens the store — is the right
  seam (see `scripts/gen_ndvi_scene.py`).
- If ergonomics matter, the most to build is a tiny **`zarr-cli stac-resolve
  <query>`** that just **prints the asset href** to paste into `LOCATION`. That
  captures ~all of Level A's value with none of the DDL-time coupling or
  non-determinism. ~30 lines, optional, no core dependency.
- For the **NASA Tier-1 recipes** (which also start from discovery), reach for
  STAC → virtual-Zarr, then the existing filter-pushdown read — the reusable story.
- Revisit an in-core `stac://` scheme **only if** the project scope explicitly
  broadens from "SQL over a Zarr store" to "SQL over a geospatial catalog" —
  which, per `CLAUDE.md`, it currently is not.

## Where STAC *would* still help

Discovery ergonomics for cookbook/examples and for any recipe that starts from a
catalog query rather than a known store. That value is fully captured by an
optional resolver helper; it does not require — and is not improved by — putting
STAC in the storage core.

## Related

- [`cookbook-adaptation.md`](cookbook-adaptation.md) — remote-read caveat; NASA
  Tier-1 (virtual-Zarr) recipes share the discovery→virtualize→read shape; Tier-4
  scope boundary (reprojection/mosaic).
- [`../cookbook/ndvi/`](../cookbook/ndvi/) and `scripts/gen_ndvi_scene.py` — the
  recipe whose STAC-resolved remote source prompted this assessment.
