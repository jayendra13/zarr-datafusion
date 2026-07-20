-- Q2 — diurnal climatology over a CONUS box (grouped aggregation).
--
-- Shape: "the average day" per grid cell — GROUP BY lat, lon, hour with AVG,
-- i.e. exactly `da.groupby("time.hour").mean()`. This is the workload the
-- flatten-nD-to-a-table model exists for: a climatology is a GROUP BY, not a
-- rechunk. Counterpart: bench/python/q2_diurnal.py.
--
-- 101 lat x 221 lon x 24 hours = 535,704 rows out of a 72-timestep window.
--
-- Derived from cookbook/climatology/diurnal_climatology.sql. `hour` is computed
-- after the read (not a filter), so all three coordinate predicates stay
-- contiguous ranges and push down cleanly.
CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

WITH samples AS (
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
    WHERE time      BETWEEN TIMESTAMP '2020-06-01T00:00:00Z'
                        AND TIMESTAMP '2020-06-03T23:00:00Z'
      AND latitude  BETWEEN 25.0  AND 50.0
      AND longitude BETWEEN 235.0 AND 290.0
  ) AS g
)
SELECT
  lat,
  lon,
  hour,
  ROUND(AVG(t2m_c), 3) AS t2m_mean_c
FROM samples
GROUP BY lat, lon, hour
HAVING AVG(t2m_c) BETWEEN -100 AND 100    -- drop NaN (absent/masked) cells
ORDER BY lat DESC, lon, hour;             -- lat DESC matches ERA5's native order
