-- Q3 — single-cell temperature timeseries, one month (744 hourly steps).
-- Shape: needle-in-a-haystack; deliberately our weak case, since ARCO-ERA5 is
-- chunked one timestep per full lat/lon plane. Counterpart: bench/python/q3_point.py.
CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

SELECT
  arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') AS ts,
  "2m_temperature" - 273.15                               AS t2m_c
FROM era5
WHERE time      BETWEEN TIMESTAMP '2023-07-01T00:00:00Z'
                    AND TIMESTAMP '2023-07-31T23:00:00Z'
  AND latitude  = 40.0
  AND longitude = 280.0
ORDER BY ts;
