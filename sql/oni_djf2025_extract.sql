-- ============================================================================
--  ONI for DJF 2025 — extract()/GROUP BY form (no UNION ALL)
-- ============================================================================
--  Same result as sql/oni_djf2025.sql, but the ~80 hand-written per-year
--  branches collapse into a single scan + GROUP BY. We pull the calendar
--  fields out of `time` with extract() and keep only the noon-of-the-15th
--  samples (day=15, hour=12, month in Dec/Jan/Feb).
--
--  ONI = AVG over the 3 months of ( monthly Niño-3.4 SST − monthly climatology )
--
--  ----------------------------------------------------------------------------
--  THIS NOW RUNS ON THE LARGE LOCAL MIRROR (verified -> oni_djf2025_c ≈ -0.42):
--    All three time predicates push down and AND-compose on `time`:
--      * day=15 AND hour=12 -> two CoordFilterKind::DatePart filters intersected
--        (CoordFilters keeps a Vec per coord, not a single Option), so BOTH
--        apply at scan time instead of one winning and the other going post-scan.
--      * extract(month) IN (12,1,2) -> CoordFilterKind::DatePartSet, also pushed.
--    The intersected selection (~noon-of-the-15th DJF samples) is a small
--    CoordSelection::Indices, and the coordinate dictionary now picks an adaptive
--    key width (Int16 -> Int32 -> Int64), so the old `as i16` overflow panic
--    ("Invalid dictionary key -32768 ...") is gone.
--    Net: only the surviving chunks are read (~200 MB here, not the full axis).
--    The UNION-ALL form in oni_djf2025.sql is an equivalent alternative; this
--    extract()/GROUP BY shape is the more readable one and is now preferred.
--
--  Notes:
--    * `time` is dictionary-encoded; date funcs crash on it, so cast first
--      with arrow_cast(...) before extract().
--    * WHERE filters ONLY coordinates (time via extract, latitude, longitude).
--      Absent SST chunks read back as NaN fill; we drop those months with a
--      post-aggregate HAVING (not by filtering the data variable in WHERE).
--
--  zarr-cli now parses multi-line, `;`-separated statements with `--` comments,
--  so this file is self-contained and runs directly:
--      zarr-cli sql/oni_djf2025_extract.sql
--      zarr-cli < sql/oni_djf2025_extract.sql
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'data/era5_sst_local.zarr';

WITH samples AS (
  SELECT
    CAST(extract(year  FROM ts) AS INT) AS yr,
    CAST(extract(month FROM ts) AS INT) AS mo,
    sst_c
  FROM (
    SELECT
      arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') AS ts,
      sea_surface_temperature - 273.15                        AS sst_c
    FROM era5
    WHERE latitude  BETWEEN  -5.0 AND   5.0   -- coordinates only in WHERE
      AND longitude BETWEEN 190.0 AND 240.0
  ) AS box
  WHERE extract(day   FROM ts) = 15           -- time coordinate (via extract)
    AND extract(hour  FROM ts) = 12
    AND extract(month FROM ts) IN (12, 1, 2)
),
monthly AS (                                  -- spatial mean per (year, month)
  SELECT yr, mo, AVG(sst_c) AS sst_c
  FROM samples
  GROUP BY yr, mo
  HAVING AVG(sst_c) BETWEEN 0 AND 50          -- drop absent-chunk (NaN) months; post-scan, not on the data var in WHERE
),
clim AS (                                     -- per-month climatology 1991-2020
  SELECT mo, AVG(sst_c) AS sst_c
  FROM monthly
  WHERE yr BETWEEN 1991 AND 2020
  GROUP BY mo
),
target AS (                                   -- DJF 2025 = Dec 2025 / Jan-Feb 2026
  SELECT mo, sst_c
  FROM monthly
  WHERE (yr = 2025 AND mo = 12)
     OR (yr = 2026 AND mo IN (1, 2))
)
SELECT AVG(target.sst_c - clim.sst_c) AS oni_djf2025_c
FROM target JOIN clim USING (mo);
