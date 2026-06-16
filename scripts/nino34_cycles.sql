-- Niño 3.4 December SST cycle — all ERA5 years (1940–2025)
-- Snapshot: December 15 12:00:00 UTC (mid-month representative value)
-- Each sub-query fetches exactly 1 chunk (~1.28 MB from GCS)
-- Total: 86 chunks × ~592 ms ≈ 51 seconds  (sequential)
-- Filter pushdown: time equality → ZarrExec reads 1 of 1,323,648 chunks per sub-query
-- Reference (NOAA CPC ERSSTv5): El Niño peaks ~28–30°C, La Niña troughs ~24–26°C

SELECT
    1940                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1940-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1941                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1941-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1942                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1942-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1943                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1943-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1944                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1944-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1945                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1945-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1946                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1946-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1947                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1947-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1948                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1948-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1949                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1949-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1950                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1950-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1951                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1951-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1952                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1952-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1953                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1953-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1954                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1954-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1955                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1955-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1956                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1956-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1957                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1957-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1958                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1958-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1959                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1959-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1960                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1960-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1961                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1961-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1962                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1962-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1963                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1963-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1964                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1964-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1965                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1965-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1966                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1966-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1967                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1967-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1968                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1968-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1969                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1969-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1970                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1970-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1971                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1971-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1972                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1972-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1973                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1973-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1974                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1974-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1975                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1975-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1976                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1976-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1977                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1977-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1978                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1978-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1979                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1979-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1980                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1980-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1981                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1981-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1982                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1982-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1983                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1983-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1984                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1984-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1985                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1985-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1986                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1986-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1987                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1987-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1988                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1988-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1989                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1989-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1990                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1990-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1991                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1991-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1992                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1992-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1993                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1993-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1994                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1994-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1995                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1995-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1996                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1996-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1997                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1997-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1998                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1998-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    1999                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '1999-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2000                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2000-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2001                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2001-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2002                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2002-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2003                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2003-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2004                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2004-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2005                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2005-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2006                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2006-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2007                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2007-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2008                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2008-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2009                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2009-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2010                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2010-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2011                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2011-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2012                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2012-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2013                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2013-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2014                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2014-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2015                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2015-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2016                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2016-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2017                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2017-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2018                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2018-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2019                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2019-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2020                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2020-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2021                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2021-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2022                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2022-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2023                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2023-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2024                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2024-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0
UNION ALL
SELECT
    2025                                         AS year,
    AVG(sea_surface_temperature - 273.15)          AS sst_celsius
FROM era5
WHERE time      = '2025-12-15 12:00:00'
  AND latitude  BETWEEN -5.0 AND 5.0
  AND longitude BETWEEN 190.0 AND 240.0

ORDER BY year
