-- Q4 — NDVI per pixel over a Sentinel-2 scene (pure projection, no reduction).
-- Counterpart: bench/python/q4_ndvi.py. 1024*1024 - 80 nodata = 1,048,496 rows.
-- Needs data/s2_ndvi_scene.zarr (uv run scripts/gen_ndvi_scene.py).
CREATE EXTERNAL TABLE IF NOT EXISTS scene
  STORED AS ZARR
  LOCATION 'data/s2_ndvi_scene.zarr';

SELECT
  x,
  y,
  ROUND((b08 - b04) / (b08 + b04), 4) AS ndvi
FROM scene
WHERE NOT isnan(b08 - b04)
ORDER BY y, x;
