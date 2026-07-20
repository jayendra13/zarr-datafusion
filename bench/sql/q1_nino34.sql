-- Q1 — Nino-3.4 box mean SST, 24 hours.
-- Shape: narrow, highly selective filter over a huge archive. Counterpart:
-- bench/python/q1_nino34.py. Expect ~28.65 degC (strong El Nino December).
CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

SELECT AVG(sea_surface_temperature - 273.15) AS sst_c
FROM era5
WHERE time      BETWEEN TIMESTAMP '2023-12-01T00:00:00Z'
                    AND TIMESTAMP '2023-12-01T23:00:00Z'
  AND latitude  BETWEEN  -5.0 AND   5.0
  AND longitude BETWEEN 190.0 AND 240.0;
