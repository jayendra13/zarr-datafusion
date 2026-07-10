-- ============================================================================
-- Case 30 — 2021 Texas Freeze: simplest skill metric (single-scalar RMSE)
-- ============================================================================
-- Forecast 2 m air temperature (CIRA FourCastNetv2) vs ERA5 truth, over the
-- Case 30 bounding box, for ONE forecast valid time.
--
-- "Simplest" choices, on purpose:
--   * One scalar RMSE (no GROUP BY, no view).
--   * A single pinned forecast field (one init_time + one lead_time), so its
--     valid time is a single known timestamp — no valid_time = init + lead
--     arithmetic, and no join on time (only on the shared 0.25° lat/lon grid).
--   * Pinning init+lead also keeps the read to one (1,1,721,1440) chunk, so it
--     stays well under the flatten-to-one-batch OOM limit.
--
-- Case 30 window : 2021-02-10 12:00 .. 2021-02-22 00:00 UTC
-- Case 30 box    : lat 24.0 .. 54.75 N,  lon 250.0 .. 278.75 E (0-360)
-- Chosen field   : init 2021-02-14 12Z + lead 24 h  ->  valid 2021-02-15 12Z
--                  (near the peak cold of the event)
-- Requires: cargo run --features icechunk --bin zarr-cli
-- ----------------------------------------------------------------------------

-- Forecast: CIRA FourCastNetv2 (icechunk, anonymous GCS). t2 in Kelvin,
-- dims (init_time, lead_time, latitude, longitude).
CREATE EXTERNAL TABLE cira STORED AS ZARR
    LOCATION 'gs://extremeweatherbench/cira-icechunk'
    OPTIONS ('group' 'FOUR_v200_GFS');

-- Target: ARCO-ERA5 (public GCS, Zarr v3). 2m_temperature in Kelvin,
-- same 721x1440 0.25-deg grid as the forecast (no regridding needed).
CREATE EXTERNAL TABLE era5 STORED AS ZARR
    LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

-- Single-scalar RMSE over the Case 30 box at the chosen valid time.
-- CAST to DOUBLE: the rmse() UDAF is exact Float64/Float64, both stores are Float32.
-- NOTE: BOTH sides must be boxed. Filtering only the forecast still returns the
-- right answer (the join restricts ERA5 to matching cells), but ERA5 would then
-- scan its full global field — box `e` too so each side reads only ~14k cells.
-- Verified result: rmse_kelvin = 2.68, n_points = 14384 (= 124 lat x 116 lon).
SELECT
    rmse(CAST(f.t2 AS DOUBLE), CAST(e."2m_temperature" AS DOUBLE)) AS rmse_kelvin,
    COUNT(*)                                                        AS n_points
FROM cira f
JOIN era5 e
    ON  f.latitude  = e.latitude
    AND f.longitude = e.longitude
WHERE f.init_time = TIMESTAMP '2021-02-14 12:00:00'   -- one forecast cycle
  AND f.lead_time = 24                                 -- +24 h (lead_time is hours)
  AND e.time      = TIMESTAMP '2021-02-15 12:00:00'   -- = valid time of that field
  AND f.latitude  BETWEEN 24.0  AND 54.75              -- Case 30 box (forecast)
  AND f.longitude BETWEEN 250.0 AND 278.75
  AND e.latitude  BETWEEN 24.0  AND 54.75              -- Case 30 box (ERA5)
  AND e.longitude BETWEEN 250.0 AND 278.75;
