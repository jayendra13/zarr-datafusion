-- ============================================================================
--  Monthly temperature climatology — whole grid, ALL valid timesteps
--  (stress-test variant: no sub-sampling — every hour of every day is read)
-- ============================================================================
--  Goal: a gridded monthly "normal". For every month (1-12) and every grid
--  cell, average the 2 m air temperature over EVERY valid timestep in a
--  reference base period. The result is a climatology cube flattened to a table:
--
--        rows = 12 (months) x n_lat x n_lon
--
--  For ARCO-ERA5's 0.25 deg grid that is 12 x 721 x 1440 = 12,454,560 rows.
--  The base period changes WHAT each mean averages over, never the row count —
--  that is fixed by the grid x 12 months.
--
--  BASE PERIOD: 1991-2020, the current WMO standard 30-year normal. Edit the
--  `extract(year ...) BETWEEN 1991 AND 2020` bounds below to retarget.
--
--  TRUE MONTHLY MEAN — no sub-sampling.  Unlike the ONI cookbook (and the
--  earlier draft of this query) there is NO day=15 / hour=12 predicate: every
--  hourly timestep that falls in the base period contributes to the mean. Each
--  cell-month normal is therefore averaged over ~all hours of that month across
--  all 30 years (e.g. January ~ 31 days x 24 hr x 30 yrs ~ 22,300 samples/cell).
--
--  *** STRESS TEST — read volume ***
--    ARCO-ERA5 is hourly with time-chunk = 1, so one chunk per timestep per
--    variable (chunk [1, 721, 1440], ~4 MB/plane for f4). 30 yrs of hourly data
--    is ~30 x 8760 = 262,800 timesteps -> ~262,800 remote chunk reads, on the
--    order of HUNDREDS OF GB to ~1 TB of egress and HOURS over a home link.
--    This query exists to push the reader / aggregation path at full scale.
--    To bound a trial run, narrow the YEAR range (e.g. a single year) and/or
--    add a spatial `latitude/longitude BETWEEN ...` (see commented lines).
--
--  VARIABLE: 2m_temperature (land + sea, defined everywhere). Swap in
--  sea_surface_temperature for an ocean-only normal (NaN over land).
--
--  Gotchas (same as the ONI cookbook):
--    * `time` is dict-encoded; date funcs crash on it -> arrow_cast() first.
--    * WHERE filters ONLY coordinates (time via extract, latitude, longitude).
--    * latitude/longitude are dict-encoded f32 -> arrow_cast() to Float64 for
--      clean grouping and output.
--    * Absent/masked cells read back as NaN; dropped post-aggregate via HAVING.
--
--  Run:  zarr-cli cookbook/climatology/monthly_climatology.sql
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

WITH samples AS (                               -- every valid hourly sample per cell
  SELECT
    CAST(extract(month FROM ts) AS INT)   AS month,
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
    -- Prototype knob: uncomment to stress a sub-grid instead of the whole globe
    -- WHERE latitude  BETWEEN -10.0 AND 10.0
    --   AND longitude BETWEEN 190.0 AND 240.0
  ) AS g
  WHERE extract(year FROM ts) BETWEEN 1991 AND 2020   -- <-- base period (ALL hours/days within)
)
SELECT
  month,
  lat,
  lon,
  ROUND(AVG(t2m_c), 3) AS t2m_mean_c            -- true monthly normal for this cell
FROM samples
GROUP BY month, lat, lon
HAVING AVG(t2m_c) BETWEEN -100 AND 100          -- drop NaN (absent/masked) cells
ORDER BY month, lat, lon;
