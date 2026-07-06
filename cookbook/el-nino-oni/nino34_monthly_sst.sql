-- ============================================================================
--  Niño-3.4 monthly-mean SST — two sources, one aligned time series
-- ============================================================================
--  Emits ONE row per month with the Niño-3.4 (5°S-5°N, 170°W-120°W ->
--  lon 190-240) area-mean SST from BOTH sources, side by side:
--
--      time        sst_ersst      sst_era5
--      ----        ---------      --------
--      YYYY-MM-01  degC (ERSST)   degC (ERA5)
--
--  This is the input for a power-spectral-density comparison of the two signals:
--    * sst_ersst — ERSST v5, an ACTUAL monthly mean (already monthly, already degC)
--    * sst_era5  — ERA5, a single representative hour per month (noon of the 15th),
--                  Kelvin -> degC, so it is a monthly PROXY, not a true monthly mean
--
--  Both are the SAME spatial box and the SAME (year, month) grid; an INNER JOIN
--  aligns them and restricts the output to the overlapping, contiguous span
--  (ERA5's record starts 1940, ERSST 1854), which is what a clean PSD needs.
--
--  Cost: dominated by ERA5. `time` is one step per chunk, so the day-15/hour-12
--  pushdown still fetches ~1000 timesteps, each a full lat×lon plane -> expect a
--  ~25-35 min run and several GB of remote reads over a home connection. ERSST is
--  fetched by byte-range from the VirtualiZarr reference and costs seconds.
--
--  Output is written straight to CSV (header row, one row per month) via COPY,
--  ready for the PSD step — no need to redirect stdout.
--
--  Run:  zarr-cli cookbook/el-nino-oni/nino34_monthly_sst.sql
--        -> writes cookbook/el-nino-oni/nino34_monthly_sst.csv
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS ersst
  STORED AS ZARR
  LOCATION 'data/ersst_v5.parq';

CREATE EXTERNAL TABLE IF NOT EXISTS era5
  STORED AS ZARR
  LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

COPY (
WITH ersst_monthly AS (                           -- true monthly mean, already degC
  SELECT yr, mo, AVG(sst_c) AS sst_c
  FROM (
    SELECT
      CAST(extract(year  FROM ts) AS INT) AS yr,
      CAST(extract(month FROM ts) AS INT) AS mo,
      sst                                 AS sst_c
    FROM (
      SELECT
        arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') AS ts,   -- dict time -> date funcs
        sst
      FROM ersst
      WHERE lat BETWEEN  -5.0 AND   5.0           -- coordinates only in WHERE
        AND lon BETWEEN 190.0 AND 240.0
    ) AS box
    WHERE extract(year FROM ts) BETWEEN 1940 AND 2026
  ) AS s
  GROUP BY yr, mo
  HAVING AVG(sst_c) BETWEEN 0 AND 50              -- guard any fill/masked cell
),
era5_monthly AS (                                 -- monthly proxy: noon of the 15th
  SELECT yr, mo, AVG(sst_c) AS sst_c
  FROM (
    SELECT
      CAST(extract(year  FROM ts) AS INT) AS yr,
      CAST(extract(month FROM ts) AS INT) AS mo,
      sst_c
    FROM (
      SELECT
        arrow_cast(time, 'Timestamp(Microsecond, Some("UTC"))') AS ts,
        sea_surface_temperature - 273.15                        AS sst_c   -- K -> degC
      FROM era5
      WHERE latitude  BETWEEN  -5.0 AND   5.0     -- coordinates only in WHERE
        AND longitude BETWEEN 190.0 AND 240.0
    ) AS box
    WHERE extract(day  FROM ts) = 15              -- one representative hour per month
      AND extract(hour FROM ts) = 12
      AND extract(year FROM ts) BETWEEN 1940 AND 2026
  ) AS s
  GROUP BY yr, mo
  HAVING AVG(sst_c) BETWEEN 0 AND 50              -- drop absent-chunk (NaN) months
)
SELECT
  make_date(e.yr, e.mo, 1)  AS time,             -- first-of-month timestamp
  ROUND(e.sst_c, 4)         AS sst_ersst,        -- ERSST v5 true monthly mean (degC)
  ROUND(a.sst_c, 4)         AS sst_era5          -- ERA5 noon-of-15th proxy    (degC)
FROM ersst_monthly e
JOIN era5_monthly  a ON a.yr = e.yr AND a.mo = e.mo
ORDER BY e.yr, e.mo
)
TO 'cookbook/el-nino-oni/nino34_monthly_sst.csv'
STORED AS CSV
OPTIONS ('format.has_header' 'true');
