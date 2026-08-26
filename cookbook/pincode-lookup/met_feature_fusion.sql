-- ============================================================================
--  Meteorological feature fusion: 6 ERA5 variables -> 6 model-ready features
-- ============================================================================
--  One statement reads six variables from the cloud Zarr store, derives the
--  features a model actually wants, and aggregates them per city for one day.
--  There is no feature-engineering job in front of this -- the derivation IS
--  the query.
--
--  Raw ->  2m_temperature, 2m_dewpoint_temperature, 10m_u/v_component_of_wind,
--          boundary_layer_height, total_precipitation
--  Out ->  temp_c_mean/max, relative humidity (Magnus), wind speed (from the
--          u/v vector), boundary-layer height, daily precipitation, and the
--          ventilation coefficient.
--
--  VENTILATION COEFFICIENT = boundary layer height x wind speed (m^2/s).
--  A standard operational air-quality metric: the lid on the mixing volume
--  times the rate that volume is flushed. Met agencies issue burn bans and
--  pollution advisories off it. LOW = whatever is emitted stays put.
--
--  It is the ORDER BY on purpose. Ranking on boundary-layer height alone is
--  misleading -- Mumbai's lid (431 m) sits BELOW Delhi's (439 m), yet Mumbai
--  ventilates at 3.89 m/s against Delhi's 1.80 and washes out with 8.5 mm of
--  rain. A low lid only matters if the air is also still, which is precisely
--  why this needs six variables in one query rather than one.
--
--  NOTE: this is dispersion POTENTIAL, not pollution. There is no PM2.5 here.
--
--  Magnus RH:  100 * exp(17.625*Td/(243.04+Td)) / exp(17.625*T/(243.04+T))
--  total_precipitation is metres per hour -> x1000 for mm, SUM over the day.
--
--  COST: 6 variables x 24 timesteps, roughly 600 MB read. See the cost note in
--  pincode_temperature.sql -- the city list is free, the time window is not.
--
--  Run:  zarr-cli cookbook/pincode-lookup/met_feature_fusion.sql
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
  ('500001','Hyderabad',17.3850,78.4867)),
fused AS (
  SELECT p.city, e.time,
    e."2m_temperature" - 273.15 AS temp_c,
    e."2m_dewpoint_temperature" - 273.15 AS dew_c,
    SQRT(POWER(e."10m_u_component_of_wind",2) + POWER(e."10m_v_component_of_wind",2)) AS wind_ms,
    e.boundary_layer_height AS blh_m,
    e.total_precipitation * 1000 AS precip_mm
  FROM era5 e
  JOIN pins p
    ON e.latitude  = ROUND(p.lat/0.25)*0.25
   AND e.longitude = ROUND(p.lon/0.25)*0.25
  WHERE e.time >= '2021-06-01 00:00:00'
    AND e.time <  '2021-06-02 00:00:00')
SELECT city,
  ROUND(AVG(temp_c),1) AS temp_c_mean,
  ROUND(MAX(temp_c),1) AS temp_c_max,
  ROUND(AVG(100*EXP(17.625*dew_c/(243.04+dew_c))/EXP(17.625*temp_c/(243.04+temp_c))),0) AS rh_pct,
  ROUND(AVG(wind_ms),2) AS wind_ms,
  ROUND(AVG(blh_m),0)   AS blh_m,
  ROUND(SUM(precip_mm),2) AS precip_mm,
  ROUND(AVG(blh_m) * AVG(wind_ms), 0) AS vent_coef_m2s   -- lid x flushing rate
FROM fused
GROUP BY city
ORDER BY vent_coef_m2s;                                  -- worst dispersion first
