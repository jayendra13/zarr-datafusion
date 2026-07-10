-- ============================================================================
-- Case 30 — 2021 Texas Freeze: grid dimensions (rows x columns) in the box
-- ============================================================================
-- "rows"    = number of latitude points inside the box   (COUNT DISTINCT latitude)
-- "columns" = number of longitude points inside the box  (COUNT DISTINCT longitude)
-- grid cells = rows * columns.  Verified result: 124 x 116 in BOTH datasets.
--
-- Case 30 box: lat 24.0 .. 54.75 N,  lon 250.0 .. 278.75 E (0-360 convention)
--
-- Notes:
--   * We count each 1-D coordinate with COUNT(DISTINCT ...) rather than COUNT(*).
--     COUNT(*) is unreliable on these mixed-dimensionality stores (it reflects the
--     flattened cube, not the lat x lon grid) — compute cells = rows * cols instead.
--   * We pin ONE field (a single init_time+lead_time for the forecast, a single
--     time for ERA5) purely to keep the read bounded; pinning does NOT change the
--     coordinate counts. Do NOT use MIN()/MAX() on a remote coordinate — with no
--     cached stats it falls back to scanning the full cube and hangs.
-- ----------------------------------------------------------------------------

-- FORECAST: CIRA FourCastNetv2 (icechunk). Pin one init_time + lead_time.
CREATE EXTERNAL TABLE cira STORED AS ZARR
    LOCATION 'gs://extremeweatherbench/cira-icechunk'
    OPTIONS ('group' 'FOUR_v200_GFS');

SELECT COUNT(DISTINCT latitude)  AS rows_lat,
       COUNT(DISTINCT longitude) AS cols_lon
FROM cira
WHERE init_time = TIMESTAMP '2021-02-14 12:00:00'
  AND lead_time = 24
  AND latitude  BETWEEN 24.0  AND 54.75
  AND longitude BETWEEN 250.0 AND 278.75;
-- -> rows_lat = 124, cols_lon = 116  (14,384 grid cells)

-- TARGET: ARCO-ERA5 (Zarr v3, GCS). Pin one valid time.
CREATE EXTERNAL TABLE era5 STORED AS ZARR
    LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

SELECT COUNT(DISTINCT latitude)  AS rows_lat,
       COUNT(DISTINCT longitude) AS cols_lon
FROM era5
WHERE time = TIMESTAMP '2021-02-15 12:00:00'
  AND latitude  BETWEEN 24.0  AND 54.75
  AND longitude BETWEEN 250.0 AND 278.75;
-- -> rows_lat = 124, cols_lon = 116  (14,384 grid cells)
