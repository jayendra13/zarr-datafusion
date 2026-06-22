-- ============================================================================
--  ONI time series — every overlapping 3-month season from 1950 to now
--  with ENSO phase (El Niño / La Niña / Neutral)
-- ============================================================================
--  ONI = 3-month running mean of monthly Niño-3.4 SST anomalies, where the
--  anomaly is (monthly SST − 1991-2020 monthly climatology). Instead of
--  enumerating a fixed season set, we build the monthly anomaly series and take
--  a centred 3-month moving average, so one row is emitted per centre month:
--
--      centre Jan -> DJF   centre Feb -> JFM   ...   centre Dec -> NDJ
--
--  A season is labelled by the YEAR OF ITS CENTRE MONTH (NOAA convention):
--      DJF 1998 = Dec 1997, Jan 1998, Feb 1998.
--
--  ENSO phase from the ONI value (NOAA single-season thresholds):
--      ONI >= +0.5  -> El Niño      ONI <= -0.5 -> La Niña      else Neutral
--  (NOAA only *declares* an event after 5 consecutive seasons past threshold;
--   this column is the per-season classification, not that 5-season rule.)
--
--  Niño-3.4 box: 5°S..5°N, 170°W..120°W  (longitude 190..240 in 0..360).
--
--  ----------------------------------------------------------------------------
--  DATA SOURCE: public ARCO-ERA5 on GCS (anonymous). Full record (from 1940),
--  so every month 1949..now is present and ~900 seasons compute.
--  This is a BIG remote scan: time is one step per chunk, so the
--  day-15/hour-12 pushdown fetches ~900 timesteps (≈78 yrs x 12 mo), each a full
--  lat×lon plane -> expect a ~20-30 min run and several GB of remote reads.
--    * Quick local alternative: swap LOCATION to 'data/era5_sst_local.zarr',
--      but that mirror only has Jan/Feb/Dec fully, so most seasons drop out.
--
--  ----------------------------------------------------------------------------
--  Base period: a single fixed 1991-2020 climatology is used for the whole
--  series. NOAA's official historical ONI instead uses centred 30-year base
--  periods that shift every 5 years, so values for the deep past will differ
--  from NOAA's published table (most by < 0.1-0.2°C).
--
--  Gotchas (same engine behaviour as the DJF file):
--    * `time` is dict-encoded; date funcs crash on it -> arrow_cast() first.
--    * WHERE filters ONLY coordinates (time via extract, latitude, longitude).
--    * Absent SST chunks read back as NaN; dropped post-aggregate via HAVING,
--      not by filtering the data var in WHERE.
--    * sea_surface_temperature is (time, latitude, longitude) — no `level` dim.
--
--  Run:  zarr-cli sql/oni_2025_all_seasons.sql
--        zarr-cli < sql/oni_2025_all_seasons.sql
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

WITH samples AS (                               -- one noon-of-the-15th sample per month
  SELECT
    CAST(extract(year  FROM ts) AS INT) AS yr,
    CAST(extract(month FROM ts) AS INT) AS mo,
    sst_c
  FROM (
    SELECT
      arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') AS ts,
      sea_surface_temperature - 273.15                        AS sst_c
    FROM era5
    WHERE latitude  BETWEEN  -5.0 AND   5.0     -- coordinates only in WHERE
      AND longitude BETWEEN 190.0 AND 240.0
  ) AS box
  WHERE extract(day  FROM ts) = 15              -- time coordinate (via extract)
    AND extract(hour FROM ts) = 12
    AND extract(year FROM ts) BETWEEN 1949 AND 2026   -- Dec 1949 enables DJF 1950
),
monthly AS (                                    -- spatial mean per (year, month)
  SELECT yr, mo, AVG(sst_c) AS sst_c
  FROM samples
  GROUP BY yr, mo
  HAVING AVG(sst_c) BETWEEN 0 AND 50            -- drop absent-chunk (NaN) months
),
clim AS (                                       -- per-month climatology 1991-2020
  SELECT mo, AVG(sst_c) AS sst_c
  FROM monthly
  WHERE yr BETWEEN 1991 AND 2020
  GROUP BY mo
),
anom AS (                                       -- monthly anomaly on a month index t
  SELECT
    m.yr,
    m.mo,
    m.yr * 12 + (m.mo - 1) AS t,                -- contiguous month counter
    m.sst_c - c.sst_c      AS anom
  FROM monthly m
  JOIN clim   c USING (mo)
),
oni AS (                                        -- centred 3-month moving average
  SELECT
    c.yr AS yr,
    c.mo AS mo,
    (p.anom + c.anom + n.anom) / 3.0 AS oni_raw
  FROM anom c
  JOIN anom p ON p.t = c.t - 1                  -- previous month
  JOIN anom n ON n.t = c.t + 1                  -- next month (both must exist)
  WHERE c.yr >= 1950                            -- centre year from 1950 on
),
season_label(mo, season) AS (                   -- centre month -> season code
  VALUES (1,'DJF'), (2,'JFM'), (3,'FMA'),  (4,'MAM'),  (5,'AMJ'),  (6,'MJJ'),
         (7,'JJA'), (8,'JAS'), (9,'ASO'), (10,'SON'), (11,'OND'), (12,'NDJ')
)
SELECT
  o.yr                       AS year,
  sl.season,
  ROUND(o.oni_raw, 2)        AS oni_c,
  CASE WHEN o.oni_raw >=  0.5 THEN 'El Niño'
       WHEN o.oni_raw <= -0.5 THEN 'La Niña'
       ELSE 'Neutral' END    AS enso_phase
FROM oni o
JOIN season_label sl USING (mo)
ORDER BY o.yr, o.mo;
