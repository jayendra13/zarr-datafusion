-- Niño 3.4 SST — average per hour, two adjacent hours only (scoped down to debug fast)
--
-- Region : lat [-5, 5], lon [190, 240]   (Niño 3.4 box)
-- SST    : ERA5 sea_surface_temperature is Kelvin -> subtract 273.15 for °C
--
-- Two consecutive hourly timesteps. A BETWEEN on time is a contiguous range,
-- so ZarrExec fetches exactly two time chunks (2 of 1,323,648) — no UNION.
-- GROUP BY time then yields one average row per hour.

SELECT
    time,
    AVG(sea_surface_temperature - 273.15)      AS sst_celsius,
    COUNT(*)                                   AS n_cells
FROM era5
WHERE time      BETWEEN '1997-12-15 00:00:00' AND '1997-12-15 01:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
GROUP BY time
ORDER BY time;
