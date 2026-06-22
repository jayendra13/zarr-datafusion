-- ============================================================================
--  ONI — Oceanic Niño Index for DJF 2025  (Dec 2025 + Jan 2026 + Feb 2026)
-- ============================================================================
--  ONI = mean over the 3 months of (monthly Niño-3.4 SST anomaly)
--        anomaly = monthly SST  -  monthly climatology
--
--  "Noon-of-the-15th" sampling for BOTH the season and the climatology:
--  each monthly mean is approximated by a single sample at 12:00 UTC on the
--  15th, so every term below reads exactly ONE Zarr chunk (time = literal,
--  equality pushdown). Climatology base period: 1991-2020.
--
--  Niño 3.4 box : latitude [-5, 5], longitude [190, 240]  (170W-120W, 0-360)
--  ERA5 SST     : Kelvin -> Celsius  (subtract 273.15)
--
--  Only base years whose SST chunk is present in the local mirror are included
--  (absent chunks read back as NaN fill, which would poison the average):
--      Dec : 30 years   Jan : 25 years   Feb : 25 years
--
--  zarr-cli reads ONE statement per line, so to run interactively paste this
--  whole query as a single line (newlines stripped). Register the table first:
--      CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION 'data/era5_sst_local.zarr';
-- ============================================================================

WITH
dec_t AS (
  SELECT AVG(sea_surface_temperature - 273.15) AS v FROM era5 WHERE time = '2025-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
),
jan_t AS (
  SELECT AVG(sea_surface_temperature - 273.15) AS v FROM era5 WHERE time = '2026-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
),
feb_t AS (
  SELECT AVG(sea_surface_temperature - 273.15) AS v FROM era5 WHERE time = '2026-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
),
dec_clim AS (
  SELECT AVG(v) AS v FROM (
    SELECT AVG(sea_surface_temperature - 273.15) AS v FROM era5 WHERE time = '1991-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1992-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1993-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1994-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1995-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1996-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1997-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1998-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1999-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2000-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2001-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2002-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2003-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2004-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2005-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2006-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2007-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2008-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2009-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2010-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2011-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2012-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2013-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2014-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2015-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2016-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2017-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2018-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2019-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2020-12-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
  ) AS u_12
),
jan_clim AS (
  SELECT AVG(v) AS v FROM (
    SELECT AVG(sea_surface_temperature - 273.15) AS v FROM era5 WHERE time = '1996-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1997-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1998-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1999-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2000-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2001-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2002-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2003-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2004-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2005-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2006-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2007-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2008-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2009-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2010-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2011-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2012-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2013-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2014-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2015-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2016-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2017-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2018-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2019-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2020-01-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
  ) AS u_1
),
feb_clim AS (
  SELECT AVG(v) AS v FROM (
    SELECT AVG(sea_surface_temperature - 273.15) AS v FROM era5 WHERE time = '1996-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1997-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1998-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '1999-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2000-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2001-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2002-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2003-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2004-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2005-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2006-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2007-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2008-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2009-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2010-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2011-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2012-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2013-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2014-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2015-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2016-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2017-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2018-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2019-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
    UNION ALL
    SELECT AVG(sea_surface_temperature - 273.15) FROM era5 WHERE time = '2020-02-15 12:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0
  ) AS u_2
)
SELECT
    ((dec_t.v - dec_clim.v)
   + (jan_t.v - jan_clim.v)
   + (feb_t.v - feb_clim.v)) / 3.0   AS oni_djf2025_c
FROM dec_t, jan_t, feb_t, dec_clim, jan_clim, feb_clim;
