-- ============================================================================
--  Diurnal temperature climatology — GROUP BY lat, lon, hour-of-day
--  (bounded window: a few summer days over a CONUS-ish box)
-- ============================================================================
--  Goal: the "average day". For every grid cell and every hour-of-day (0-23),
--  average the 2 m air temperature across the days in a short window. The result
--  is the typical diurnal cycle at each location -- "how warm is it here at
--  06:00 vs 18:00?" -- flattened to a table:
--
--        rows = n_lat x n_lon x 24 hours
--
--  For the CONUS box below (lat 25..50, lon 235..290 at 0.25 deg) that is
--  101 x 221 x 24 = 535,704 rows.
--
--  WHY THIS RECIPE EXISTS:
--    This is the canonical "climatology = a GROUP BY, not a rechunk" workload.
--    In the array paradigm (xarray/dask) you load native Zarr chunks, *rechunk*
--    so all of `time` lands in one chunk ("pencils"), run a grouped reduction
--    over the calendar, then rechunk back ("pancakes") to write. The rechunk
--    serves only the array layout; the *operation* is just:
--
--        SELECT latitude, longitude, hour, AVG("2m_temperature")
--        GROUP BY latitude, longitude, hour
--
--    i.e. exactly da.groupby("time.hour").mean(). We register the multi-decade
--    ARCO-ERA5 archive as a lazy table and let the WHERE prune the read: the
--    query touches only 2m_temperature over the window and never scans the rest.
--    (Mirrors xarray-sql benchmarks/geospatial/02_climatology.py.)
--
--  WINDOW: 2020-06-01 00:00 .. 2020-06-03 23:00 UTC (72 hourly timesteps, so
--    each hour-of-day bin averages 3 samples). Widen the date range to average
--    a whole season; the row count is fixed by grid x 24 and never changes.
--
--  AREA: a CONUS-ish box. ERA5 latitude DESCENDS and longitude is 0-360 deg E,
--    so 235..290 deg E == 125..70 deg W. Edit the BETWEEN bounds to retarget.
--
--  *** Read volume ***
--    ARCO-ERA5 is hourly with time-chunk = 1: one chunk per timestep over the
--    FULL lat x lon plane (chunk [1, 721, 1440], ~4 MB/plane for f4). A SPATIAL
--    filter does NOT cut chunk reads -- only the time predicate does. So the cost
--    is set by the 72-timestep window: ~72 remote chunk reads, ~300 MB egress.
--    Widening the dates is what makes this expensive (a full 30-yr diurnal normal
--    is the ~1 TB class -- see monthly_climatology.sql).
--
--  VARIABLE: 2m_temperature (land + sea, defined everywhere). Swap in
--    sea_surface_temperature for an ocean-only cycle (NaN over land).
--
--  Gotchas (same as the rest of this cookbook):
--    * `time` is dict-encoded; date funcs crash on it -> arrow_cast() first.
--    * WHERE filters ONLY coordinates. All three predicates here are RANGES
--      (time/lat/lon BETWEEN) -> contiguous per-axis reads, no scattered-Indices
--      conflict. `hour` is derived AFTER the read, in the GROUP BY, not a filter.
--    * latitude/longitude are dict-encoded f32 -> arrow_cast() to Float64 for
--      clean grouping and output.
--    * Absent/masked cells read back as NaN; dropped post-aggregate via HAVING.
--
--  Run:  zarr-cli cookbook/climatology/diurnal_climatology.sql
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

WITH samples AS (                               -- every hourly sample in the window, per cell
  SELECT
    CAST(extract(hour FROM ts) AS INT)    AS hour,
    arrow_cast(latitude,  'Float64')      AS lat,
    arrow_cast(longitude, 'Float64')      AS lon,
    t2m_c
  FROM (
    SELECT
      arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') AS ts,
      "2m_temperature" - 273.15                               AS t2m_c,
      latitude,
      longitude
    FROM era5
    WHERE time      BETWEEN TIMESTAMP '2020-06-01T00:00:00Z'   -- <-- window (widen for a seasonal normal)
                        AND TIMESTAMP '2020-06-03T23:00:00Z'
      AND latitude  BETWEEN 25.0  AND 50.0                     -- <-- CONUS-ish box (ERA5 lat descends)
      AND longitude BETWEEN 235.0 AND 290.0                    -- <-- 235..290 E == 125..70 W
  ) AS g
)
SELECT
  lat,
  lon,
  hour,
  ROUND(AVG(t2m_c), 3) AS t2m_mean_c            -- typical temperature at this cell + hour-of-day
FROM samples
GROUP BY lat, lon, hour
HAVING AVG(t2m_c) BETWEEN -100 AND 100          -- drop NaN (absent/masked) cells
ORDER BY lat DESC, lon, hour;                   -- lat DESC matches ERA5's native order
