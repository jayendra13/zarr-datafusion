-- Niño 3.4 SST — single-day average (remote bring-up, intentionally small).
--
-- `time` is hourly ("hours since 1900-01-01"), one chunk per hour. A one-day
-- window is therefore exactly 24 contiguous chunks. We average SST over the
-- Niño 3.4 box for that day — small enough to iterate on while we get the
-- REMOTE (GCS) read path working; scope expands later (a month, then the cycle).
--
-- Range filter on `time` => a contiguous `CoordSelection::Range` of 24 chunks.
SELECT
    AVG(sea_surface_temperature - 273.15) AS sst_celsius
FROM era5
WHERE time BETWEEN '2023-12-15 00:00:00' AND '2023-12-15 23:00:00'
  AND latitude  BETWEEN  -5.0 AND   5.0
  AND longitude BETWEEN 190.0 AND 240.0;
