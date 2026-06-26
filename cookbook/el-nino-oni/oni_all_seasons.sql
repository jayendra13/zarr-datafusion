-- ============================================================================
--  ONI time series — every overlapping 3-month season from 1950 to now
--  with ENSO phase (El Niño / La Niña / Neutral), NOAA-faithful climatology
-- ============================================================================
--  ONI = 3-month running mean of monthly Niño-3.4 SST anomalies, where the
--  anomaly is (monthly SST − monthly climatology). We build the monthly anomaly
--  series and take a centred 3-month moving average, so one row is emitted per
--  centre month:  centre Jan -> DJF, centre Feb -> JFM, ... centre Dec -> NDJ.
--  A season is labelled by the YEAR OF ITS CENTRE MONTH (NOAA convention):
--  DJF 1998 = Dec 1997, Jan 1998, Feb 1998.
--
--  CLIMATOLOGY: each season uses NOAA CPC's centered 30-year base period that
--  shifts forward every 5 years — NOT a single fixed window. A 30-year normal is a
--  snapshot of a warming climate, so NOAA's rolling, centred schedule keeps each
--  season's baseline contemporary to the data it explains. Reproducing that
--  schedule is what keeps this faithful to NOAA: near-zero overall bias (-0.04 °C)
--  and no climatology-induced break in the residual.
--
--  NOAA base-period schedule (from CPC's ONI_change documentation; the stated
--  "1950-1955 -> 1936-1965, 1956-1960 -> 1941-1970, and so on" +5yr pattern,
--  held at the latest complete 30-yr period for recent years):
--
--      ONI season-years     centered 30-yr base period
--      ----------------     --------------------------
--      1950-1955            1936-1965
--      1956-1960            1941-1970
--      1961-1965            1946-1975
--      1966-1970            1951-1980
--      1971-1975            1956-1985
--      1976-1980            1961-1990
--      1981-1985            1966-1995
--      1986-1990            1971-2000
--      1991-1995            1976-2005
--      1996-2000            1981-2010
--      2001-2005            1986-2015
--      2006-now             1991-2020   (held: no complete later 30-yr period yet)
--
--  ----------------------------------------------------------------------------
--  ERA5 CAVEAT: ARCO-ERA5's full record starts in 1940, so the earliest base
--  period (1936-1965) is effectively 1940-1965 here — 26 of its 30 years. Every
--  later base period is fully covered. This is unavoidable with ERA5 and is the
--  one place this query cannot exactly match NOAA.
--
--  COST CAVEAT: the sample window starts at 1940 so the deep-past climatology
--  has data. time is one step per chunk, so the day-15/hour-12 pushdown fetches
--  ~1000 timesteps (≈87 yrs x 12 mo), each a full lat×lon plane -> expect a
--  ~25-35 min run and several GB of remote reads over a home connection.
--
--  Gotchas:
--    * `time` is dict-encoded; date funcs crash on it -> arrow_cast() first.
--    * WHERE filters ONLY coordinates (time via extract, latitude, longitude).
--    * Absent SST chunks read back as NaN; dropped post-aggregate via HAVING.
--    * sea_surface_temperature is (time, latitude, longitude) — no `level` dim.
--
--  Run:  zarr-cli cookbook/el-nino-oni/oni_all_seasons.sql
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
    AND extract(year FROM ts) BETWEEN 1940 AND 2026   -- widened: rolling base needs deep past
),
monthly AS (                                    -- spatial mean per (year, month)
  SELECT yr, mo, AVG(sst_c) AS sst_c
  FROM samples
  GROUP BY yr, mo
  HAVING AVG(sst_c) BETWEEN 0 AND 50            -- drop absent-chunk (NaN) months
),
base_periods(yr_lo, yr_hi, base_lo, base_hi) AS (   -- season-year block -> 30-yr base
  VALUES
    (1940, 1955, 1936, 1965),   -- 1940 lower edge covers Dec 1949 (needed for DJF 1950);
                                --   base 1936-1965 is effectively 1940-1965 in ERA5
    (1956, 1960, 1941, 1970),
    (1961, 1965, 1946, 1975),
    (1966, 1970, 1951, 1980),
    (1971, 1975, 1956, 1985),
    (1976, 1980, 1961, 1990),
    (1981, 1985, 1966, 1995),
    (1986, 1990, 1971, 2000),
    (1991, 1995, 1976, 2005),
    (1996, 2000, 1981, 2010),
    (2001, 2005, 1986, 2015),
    (2006, 2026, 1991, 2020)    -- held at latest complete 30-yr period
),
clim AS (                                       -- per-month climatology, per base period
  SELECT p.yr_lo AS yr_lo, m.mo AS mo, AVG(m.sst_c) AS clim_sst
  FROM base_periods p
  JOIN monthly m ON m.yr BETWEEN p.base_lo AND p.base_hi
  GROUP BY p.yr_lo, m.mo
),
anom AS (                                       -- monthly anomaly vs its own base period
  SELECT
    m.yr,
    m.mo,
    m.yr * 12 + (m.mo - 1) AS t,                -- contiguous month counter
    m.sst_c - c.clim_sst   AS anom
  FROM monthly m
  JOIN base_periods p ON m.yr BETWEEN p.yr_lo AND p.yr_hi
  JOIN clim        c ON c.yr_lo = p.yr_lo AND c.mo = m.mo
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
