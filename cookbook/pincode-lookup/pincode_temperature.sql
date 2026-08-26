-- ============================================================================
--  Pincode -> grid extraction: a postcode list joined straight onto ERA5
-- ============================================================================
--  Snap each postcode centroid to the nearest ERA5 0.25 deg cell, then read a
--  day of hourly 2 m temperature straight from the public cloud store. No
--  download, no conversion, no staging step.
--
--  Output: 8 rows (one per pincode) x mean/max/min degC over 24 hourly steps.
--
--  THE SNAP:  latitude  = ROUND(lat / 0.25) * 0.25
--             longitude = ROUND(lon / 0.25) * 0.25
--  A centroid almost never lands exactly on the grid, and coordinate matching
--  is exact equality -- so the rounding happens on the LOOKUP side of the join
--  and the join itself is a plain equality. Multiples of 0.25 are exact in
--  f32/f64, so this is safe.
--
--  All eight cities here are east of the prime meridian, so longitude needs no
--  0-360 wrap. For a western-hemisphere point use ((lon + 360) % 360) first.
--
--  COST: ARCO-ERA5 chunk-1 stores one timestep per chunk over the full lat x lon
--  plane, so the pincode filter does NOT cut chunk reads -- the `time` predicate
--  does. 24 timesteps of one variable is roughly 100 MB read. Adding pincodes is
--  free; adding days is not.
--
--  Run:  zarr-cli cookbook/pincode-lookup/pincode_temperature.sql
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

WITH pins(code, city, lat, lon) AS (VALUES
  ('560001','Bengaluru',12.9716,77.5946),
  ('400001','Mumbai',   18.9388,72.8354),
  ('110001','Delhi',    28.6139,77.2090),
  ('700001','Kolkata',  22.5726,88.3639),
  ('600001','Chennai',  13.0827,80.2707),
  ('500001','Hyderabad',17.3850,78.4867),
  ('380001','Ahmedabad',23.0225,72.5714),
  ('781001','Guwahati', 26.1445,91.7362))
SELECT
  p.code,
  p.city,
  COUNT(*) AS hours,
  ROUND(AVG(e."2m_temperature") - 273.15, 1) AS temp_c_mean,
  ROUND(MAX(e."2m_temperature") - 273.15, 1) AS temp_c_max,
  ROUND(MIN(e."2m_temperature") - 273.15, 1) AS temp_c_min
FROM era5 e
JOIN pins p
  ON e.latitude  = ROUND(p.lat/0.25)*0.25
 AND e.longitude = ROUND(p.lon/0.25)*0.25
WHERE e.time >= '2021-06-01 00:00:00'
  AND e.time <  '2021-06-02 00:00:00'
GROUP BY p.code, p.city
ORDER BY temp_c_max DESC;
