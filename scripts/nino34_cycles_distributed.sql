-- Niño 3.4 December SST cycle — DISTRIBUTED single-scan form (1940–2025).
--
-- This is the distribution-friendly rewrite of scripts/nino34_cycles.sql.
--
-- Why a rewrite was needed
-- ------------------------
-- The original query is 86 `UNION ALL` sub-queries, each pinning
-- `time = '<year>-12-15 12:00'`. Filter pushdown collapses each sub-query to a
-- SINGLE time chunk, so the scan has one partition and CANNOT fan out across
-- workers — the per-year form gets no benefit from distribution.
--
-- This form is ONE scan with a date-part filter, so the scan partitions the
-- time axis and each worker handles a disjoint share of the surviving December
-- chunks. `EXTRACT(MONTH FROM time) = 12` pushes down to a scattered
-- `CoordSelection::Indices`; `plan_partitions` slices the time grid and
-- `restrict_to_partition` intersects each partition with those indices.
--
-- What it computes
-- ----------------
-- The December MONTHLY-MEAN SST over the Niño 3.4 box, one value per year —
-- averaging every December hour rather than a single Dec-15 12:00 snapshot.
-- (The snapshot can't be a single scan: the filter engine allows only one
-- date-part per coordinate, so MONTH=12 AND DAY=15 AND HOUR=12 would collide.
-- The monthly mean is a stronger ENSO signal anyway.)
--
-- Data volume: all December hours 1940–2025 (~68k time chunks on the local
-- era5_sst_local.zarr mirror). This is the workload distribution is meant to
-- parallelize — run it across the cluster, not single-node.
--
-- Reference (NOAA CPC ERSSTv5): El Niño Decembers ~28–30 °C, La Niña ~24–26 °C.

SELECT
    EXTRACT(YEAR FROM time)                       AS year,
    AVG(sea_surface_temperature - 273.15)         AS sst_celsius
FROM era5
WHERE EXTRACT(MONTH FROM time) = 12
  AND latitude  BETWEEN  -5.0 AND   5.0
  AND longitude BETWEEN 190.0 AND 240.0
GROUP BY year
ORDER BY year;
