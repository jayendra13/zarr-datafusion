-- ============================================================================
--  ONI from ERSST v5 — the authoritative, cheap path
-- ============================================================================
--  Same ONI methodology as oni_all_seasons.sql (3-month running mean of monthly
--  Niño-3.4 SST anomalies, NOAA's rolling centred 30-year base periods), but the
--  source is ERSST v5 — the dataset NOAA CPC actually builds the ONI from. So the
--  expensive ERA5 machinery disappears:
--    * ERSST is ALREADY monthly    -> no day-15 / noon-12 sampling
--    * `sst` is ALREADY in degC    -> no `- 273.15`
--    * record starts 1854          -> the earliest base period (1936-1965) is
--                                     FULLY covered (ERA5's one caveat is gone)
--    * dims are (time, lat, lon)   -> no `level`; Niño box is all ocean (no NaN)
--
--  Source: a VirtualiZarr reference over the NOAA PSL aggregated ERSST file,
--  produced by scripts/virtualize_ersst.py. Reads chunk byte-ranges on demand.
--
--  Run:  target/debug/zarr-cli < cookbook/el-nino-oni/oni_ersst.sql
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS ersst
  STORED AS ZARR
  LOCATION 'data/ersst_v5.parq';

WITH samples AS (                               -- monthly Niño-3.4 samples (already monthly)
  SELECT
    CAST(extract(year  FROM ts) AS INT) AS yr,
    CAST(extract(month FROM ts) AS INT) AS mo,
    sst                                   AS sst_c   -- ERSST sst is already degC
  FROM (
    SELECT
      arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') AS ts,   -- dict time -> date funcs
      sst
    FROM ersst
    WHERE lat BETWEEN  -5.0 AND   5.0            -- coordinates only in WHERE
      AND lon BETWEEN 190.0 AND 240.0
  ) AS box
  WHERE extract(year FROM ts) BETWEEN 1936 AND 2026   -- 1936 => earliest base period is complete
),
monthly AS (                                    -- spatial mean per (year, month)
  SELECT yr, mo, AVG(sst_c) AS sst_c
  FROM samples
  GROUP BY yr, mo
  HAVING AVG(sst_c) BETWEEN 0 AND 50            -- guard against any fill/masked cell
),
base_periods(yr_lo, yr_hi, base_lo, base_hi) AS (   -- season-year block -> 30-yr base
  VALUES
    (1936, 1955, 1936, 1965),   -- ERSST covers 1936 fully (unlike ERA5)
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
    (2006, 2010, 1991, 2020),
    (2011, 2026, 1996, 2025)    -- NOAA advances to the 1996-2025 centred period for 2011+
                                --   (held here for 2016+, whose centred periods need future data)
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
    m.yr * 12 + (m.mo - 1) AS t,
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
  JOIN anom p ON p.t = c.t - 1
  JOIN anom n ON n.t = c.t + 1
  WHERE c.yr >= 1950
),
season_label(mo, season) AS (
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
