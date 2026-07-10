-- ============================================================================
--  Monthly temperature climatology for the 10 largest cities
--  via an in-query (VALUES) lookup table + nearest-grid-cell snapping
-- ============================================================================
--  Goal: monthly "normal" 2 m temperature for each of the 10 largest urban
--  agglomerations, over the 1991-2020 base period. Output:
--
--        rows = 10 cities x 12 months = 120
--
--  NEAREST-CELL TRICK (no engine change needed):
--    A city centroid almost never lands exactly on the 0.25 deg grid, and the
--    Zarr filter pushdown matches coordinates by EXACT equality (it has no
--    nearest-neighbour mode). So we pre-snap each centroid to its nearest grid
--    point IN THE LOOKUP TABLE itself: glat/glon = round(centroid / 0.25) * 0.25.
--    Multiples of 0.25 are exact in f32/f64, so the join below is exact equality
--    and the IN-lists push down cleanly.
--
--    ERA5 longitude is 0-360 (not -180..180), so western-hemisphere centroids
--    are converted: glon = round(((lon + 360) mod 360) / 0.25) * 0.25
--    (e.g. Sao Paulo -46.63 -> 313.25, Mexico City -99.13 -> 260.75).
--    The glat/glon columns below are precomputed; see the snapping script in
--    the README if you change the city list.
--
--  BASE PERIOD: 1991-2020 (WMO standard normal). Edit the year BETWEEN below.
--
--  SAMPLING: one sample per month — noon (12:00 UTC) on the 15th — averaged
--  across the 30 years. NOTE: ARCO-ERA5 stores one timestep per chunk over the
--  FULL lat x lon plane, so a spatial filter does NOT cut chunk reads — only the
--  time predicate does. Restricting to 10 cities therefore costs the same I/O as
--  the whole grid; it only shrinks the output and the group-by. "All hours"
--  (drop the day/hour predicates) is the ~1 TB stress test in
--  monthly_climatology.sql, not a city-scale job — keep the mid-month sample
--  here unless you specifically want that.
--
--  Gotchas (same as the ONI cookbook):
--    * `time` is dict-encoded; date funcs crash on it -> arrow_cast() first.
--    * WHERE filters ONLY coordinates (time via extract, latitude, longitude).
--    * latitude/longitude are dict-encoded f32 -> arrow_cast() to Float64.
--
--  Run:  zarr-cli cookbook/climatology/city_climatology.sql
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

WITH cities(name, lat, lon, glat, glon) AS (   -- in-query lookup; glat/glon = nearest 0.25 deg cell (lon in 0-360)
  VALUES
    ('Tokyo',        35.6895,  139.6917,  35.75, 139.75),
    ('Delhi',        28.7041,   77.1025,  28.75,  77.00),
    ('Shanghai',     31.2304,  121.4737,  31.25, 121.50),
    ('Sao Paulo',   -23.5505,  -46.6333, -23.50, 313.25),
    ('Mexico City',  19.4326,  -99.1332,  19.50, 260.75),
    ('Cairo',        30.0444,   31.2357,  30.00,  31.25),
    ('Mumbai',       19.0760,   72.8777,  19.00,  73.00),
    ('Beijing',      39.9042,  116.4074,  40.00, 116.50),
    ('Dhaka',        23.8103,   90.4125,  23.75,  90.50),
    ('Osaka',        34.6937,  135.5023,  34.75, 135.50)
),
grid AS (                                      -- bounding box (Range), exact cells picked by the JOIN below
  SELECT
    arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') AS ts,
    arrow_cast(latitude,  'Float64')                        AS lat,
    arrow_cast(longitude, 'Float64')                        AS lon,
    "2m_temperature" - 273.15                               AS t2m_c
  FROM era5
  -- NOTE: spatial filters MUST be ranges (BETWEEN), not IN-lists. The reader only
  -- supports ONE scattered-Indices coordinate, and `time` already is one here
  -- (EXTRACT day/hour). Two IN-lists would add lat+lon Indices and panic in
  -- build_read_plans (zarr_reader.rs) -> match_ranges_to_data_var (filter.rs:1574).
  -- BETWEEN yields a contiguous Range per axis; the JOIN does the exact pairing.
  WHERE latitude  BETWEEN -23.5  AND 40.0       -- bbox of the 10 snapped lats
    AND longitude BETWEEN  31.25 AND 313.25     -- bbox of the 10 snapped lons (0-360)
),
samples AS (                                   -- attach city name via exact nearest-cell match
  SELECT
    c.name,
    CAST(extract(month FROM g.ts) AS INT) AS month,
    g.t2m_c
  FROM grid g
  JOIN cities c
    ON g.lat = c.glat AND g.lon = c.glon       -- exact equality: glat/glon are on-grid
  WHERE extract(day  FROM g.ts) = 15           -- mid-month representative sample
    AND extract(hour FROM g.ts) = 12           -- noon UTC
    AND extract(year FROM g.ts) BETWEEN 1991 AND 2020   -- <-- base period
)
SELECT
  name,
  month,
  ROUND(AVG(t2m_c), 3) AS t2m_mean_c           -- monthly normal at the city's grid cell
FROM samples
GROUP BY name, month
ORDER BY name, month;
