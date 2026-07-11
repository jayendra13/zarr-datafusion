# NDVI cookbook — per-pixel band math via SQL

Computes the **Normalized Difference Vegetation Index** for every pixel of a
Sentinel-2 scene with one SQL expression:

```sql
SELECT x, y, (b08 - b04) / (b08 + b04) AS ndvi   -- (NIR - Red) / (NIR + Red)
FROM scene
ORDER BY y, x;
```

| file | scope | output rows | cost |
|------|-------|------------|------|
| [`ndvi.sql`](ndvi.sql) | 1024×1024 window, 10 m bands | 1024 × 1024 = **1.05 M** | ~8 MB (local sample) |

Adapted from
[xarray-sql's `01_ndvi.py`](https://github.com/xqlsystems/xarray-sql/blob/main/benchmarks/geospatial/01_ndvi.py).

## Why this recipe exists

Every other recipe in this cookbook is an **aggregation** — the payoff is
`GROUP BY … AVG` (see [`../climatology`](../climatology)). NDVI is the
complement: a pure **per-pixel expression** across two co-registered variables.
`b04` (red) and `b08` (NIR) are both 10 m bands on the same `x`/`y` grid, so the
flatten-nD→2D model puts them on the **same rows** and the index reduces to
column arithmetic — no reduction, no rechunk, no join. It's the cleanest possible
demonstration that a Zarr scene *is* a table once you stop thinking in arrays.

The Sentinel-2 scene maps onto the reader's assumptions exactly:

| Zarr array | shape | role |
|---|---|---|
| `x` | `[1024]` | coordinate (UTM easting, m) |
| `y` | `[1024]` | coordinate (UTM northing, m) |
| `b04` | `[1024, 1024]` | data var — red |
| `b08` | `[1024, 1024]` | data var — NIR |

## The data (and why it's local)

`scripts/gen_ndvi_scene.py` resolves the scene **the same way the benchmark
does** — search the [EOPF STAC catalog](https://stac.core.eopf.eodc.eu) for a
`sentinel-2-l2a` item over an agricultural area near Turin, Italy
(2025-04-25 … 2025-05-05), open the `measurements/reflectance/r10m` group, and cut
the 1024×1024 pixel window at `(y=4000, x=6000)` — then writes a tidy 4-array
Zarr locally:

```bash
uv run scripts/gen_ndvi_scene.py     # -> data/s2_ndvi_scene.zarr  (~8 MB)
```

**Why a local sample and not the remote store directly?** The EOPF product can't
be read remotely by this reader, for two independent reasons a spike confirmed:

1. **Pointing at the `r10m` group** forces a directory listing (there's no
   group-level consolidated metadata). The EODC Ceph/S3 endpoint answers a
   listing (`PROPFIND`) with `405 Method Not Allowed`, so the HTTP object-store
   backend can't enumerate the arrays.
2. **Pointing at the product root** *does* read — it has a consolidated
   `.zmetadata` (no listing needed) — but that metadata covers the **whole
   hierarchical product**: 10/20/60 m bands with different shapes and multiple
   `x`/`y`. The reader tries to fold them into one Cartesian coord cube and
   overflows the shape product. It's the mixed-dimensionality conflict from
   `docs/design-decisions.md` §14b, without native V3 `dimension_names` to save it.

Reading the remote store would need reader changes (prefix-filter a consolidated
`.zmetadata` down to one subgroup, or an S3 backend with a custom endpoint +
tenant + anonymous auth) — out of scope for a recipe. The generator sidesteps
both: it reads through xarray (which streams individual chunks by key, no listing)
and lands a clean single-group store.

### Transpose to (x, y) — keeping the coordinates honest

The source bands are laid out `[y, x]`, but the reader assumes **coordinates
sorted alphabetically with data dims in that same order** — i.e. `x` before `y`.
The generator therefore **transposes the bands to `[x, y]`** before writing.
NDVI is symmetric per pixel so the index value is unaffected either way, but the
transpose is what keeps the `x`/`y` **column labels** correct instead of swapped.

## Gotchas

- **NaN nodata.** Bands are CF-scaled reflectance floats with `NaN` where the
  source fill was 0. DataFusion treats `NaN` as **equal to `NaN`** and **greater
  than everything**, so a NaN pixel poisons `MAX`/`AVG` *and* slips past a
  `b04 = b04` test. Filter with `NOT isnan(b08 - b04)` — one test covers both
  bands, since a NaN in either makes the difference NaN. (`ndvi.sql` does this.)
- **Row count is fixed** by the window (`n_x × n_y`), never by the filter — this
  is a projection, not a reduction.

## Run

```bash
zarr-cli cookbook/ndvi/ndvi.sql        # 1.05M rows — redirect to a file
```

## Validating

The reader reproduces xarray's NDVI over the valid pixels **to 4 decimals**:

| metric | value |
|---|---|
| valid pixels | 1,048,496 (80 nodata dropped, 0.008%) |
| NDVI min | −0.2395 |
| NDVI max | 0.8391 |
| NDVI mean | 0.2354 |

Reproduce the summary directly:

```sql
SELECT COUNT(*) AS n,
       ROUND(MIN((b08-b04)/(b08+b04)),4) AS ndvi_min,
       ROUND(MAX((b08-b04)/(b08+b04)),4) AS ndvi_max,
       ROUND(AVG((b08-b04)/(b08+b04)),4) AS ndvi_mean
FROM scene
WHERE NOT isnan(b08 - b04);
```

## Plots

[`plots.py`](plots.py) displays the recipe output — fully offline, reading the
frozen `ndvi.csv.gz` (the `ndvi.sql` result; regenerate with the `COPY … STORED
AS CSV` shown at the top of the script, then `gzip -9`):

```bash
uv run --with pandas --with numpy --with matplotlib cookbook/ndvi/plots.py
```

| output | what it shows |
|---|---|
| `ndvi_map.png` | the NDVI raster — the flat `(x, y, ndvi)` table **pivoted back into the 2D scene**. The whole point: a Zarr scene *is* a table, and the table *is* the scene. |
| `ndvi_hist.png` | NDVI distribution over the valid pixels, with the mean and the standard land-cover thresholds. |
| `ndvi_landcover.png` | NDVI binned into land-cover classes (water/snow, bare soil, sparse/moderate/dense vegetation) — the per-pixel expression turned into a classified map. |

The window sits in the Alpine foothills near Turin: green agricultural valleys
and field parcels, bare/rocky slopes, and snow on the high peaks (which reads
NDVI < 0, same as water).

## Writing results to Parquet

No code change needed — wrap the final `SELECT` (drop `ORDER BY` for the write):

```sql
COPY (
  SELECT x, y, ROUND((b08 - b04) / (b08 + b04), 4) AS ndvi
  FROM scene WHERE NOT isnan(b08 - b04)
) TO 'cookbook/ndvi/ndvi.parquet' STORED AS PARQUET;
```
