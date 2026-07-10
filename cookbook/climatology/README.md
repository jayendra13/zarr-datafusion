# Climatology cookbook — gridded normals via SQL

Builds **monthly temperature normals** from ARCO-ERA5 with a single SQL query.
Two variants:

| file | scope | output rows | cost |
|------|-------|------------|------|
| [`city_climatology.sql`](city_climatology.sql) | 10 largest cities, nearest grid cell | 10 × 12 = **120** | ~1–2 GB (mid-month sample) |
| [`monthly_climatology.sql`](monthly_climatology.sql) | whole 0.25° grid, **all** timesteps | 12 × 721 × 1440 = **12.4 M** | hundreds of GB–~1 TB (stress test) |

Both use the same base period (default **1991–2020**, the WMO standard normal),
the same `era5` GCS table, and the same dict-encoding `arrow_cast` gotchas.

## City variant (start here)

`city_climatology.sql` computes a monthly normal for each of the 10 largest
urban agglomerations using an **in-query `VALUES` lookup table**.

### Nearest-cell snapping (why the join is exact equality)

A city centroid almost never lands on the 0.25° grid, and the Zarr filter
pushdown matches coordinates by **exact equality** — it has no nearest-neighbour
mode (see `src/reader/filter.rs`). So we pre-snap each centroid to its nearest
grid point *in the lookup table*:

```
glat = round(lat / 0.25) * 0.25
glon = round(((lon + 360) mod 360) / 0.25) * 0.25    # ERA5 lon is 0–360
```

Multiples of 0.25° are exact in f32/f64, so `g.lat = c.glat AND g.lon = c.glon`
is a safe exact-equality join and the `IN`-lists push down cleanly. To change the
city list, recompute the `glat/glon` literals:

```bash
python3 - <<'PY'
cities = [("Tokyo", 35.6895, 139.6917), ...]   # name, lat, lon
snap = lambda v: round(v*4)/4
for n,la,lo in cities:
    print(n, snap(la), snap(lo % 360))
PY
```

### Cities don't reduce I/O here (important)

ARCO-ERA5 stores **one timestep per chunk over the full lat×lon plane**, so a
spatial filter does *not* cut chunk reads — only the time predicate does.
Restricting to 10 cities costs the same I/O as the whole grid; it only shrinks
the output and the group-by. That's why the city query keeps the **mid-month
sample** (noon on the 15th, averaged across 30 years → 360 planes). Dropping the
`day`/`hour` predicates turns it into the full ~1 TB stress test.

## Whole-grid stress variant

`monthly_climatology.sql` removes all sub-sampling: every hourly timestep in the
base period contributes to each cell's mean (~262,800 chunk reads). It exists to
push the reader/aggregation path at full scale. Bound a trial by narrowing the
year range and/or uncommenting the spatial sub-grid `BETWEEN`.

## Run

```bash
zarr-cli cookbook/climatology/city_climatology.sql      # 120 rows
zarr-cli cookbook/climatology/monthly_climatology.sql   # 12.4M rows — redirect to a file
```

## Writing results to Parquet

No code change needed — DataFusion 54 supports Parquet writes natively. Wrap the
final `SELECT` (drop `ORDER BY` for the write):

```sql
COPY ( <the WITH ... SELECT ... GROUP BY ...> )
TO 'cookbook/climatology/city_climatology_1991_2020.parquet'
STORED AS PARQUET;
```

## Validating the normals

Each value is a **grid-cell** number, so compare references the right way:

- **Pipeline consistency:** diff against ERA5 monthly-mean products
  (WeatherBench 2 / CDS monthly) — should match to rounding.
- **Physical truth:** diff each city against **station normals** (NOAA NCEI
  1991–2020 CSV, or Meteostat `Normals` for global cities). Expect legitimate
  station-vs-cell offsets (~1–2 °C, more on coasts/terrain) and the
  (Tmax+Tmin)/2 vs true-mean gap — not bugs.
