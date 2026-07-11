-- ============================================================================
--  NDVI — per-pixel band math over a Sentinel-2 scene (projection, not GROUP BY)
-- ============================================================================
--  Goal: the Normalized Difference Vegetation Index for every pixel of a
--  1024 x 1024 window of Sentinel-2 L2A surface reflectance:
--
--        NDVI = (NIR - Red) / (NIR + Red) = (b08 - b04) / (b08 + b04)
--
--  Output is one row per pixel (x, y, ndvi):
--
--        rows = n_x * n_y = 1024 * 1024 = 1,048,576
--
--  WHY THIS RECIPE EXISTS:
--    Every other recipe in this cookbook is an AGGREGATION (climatology, ONI,
--    freeze-RMSE) whose payoff is `GROUP BY ... AVG`. NDVI is the complement: a
--    pure PER-PIXEL EXPRESSION across two co-registered variables. It is the
--    cleanest demonstration of the flatten-nD->2D model -- b04 (red) and b08
--    (NIR) share the same x/y grid, so they land on the same rows and the index
--    is just column arithmetic. No reduction, no rechunk, no join. This mirrors
--    xarray-sql's benchmarks/geospatial/01_ndvi.py, whose whole SQL is one line:
--        SELECT x, y, (nir - red) / (nir + red) AS ndvi FROM scene ORDER BY y, x
--
--  DATA: data/s2_ndvi_scene.zarr, written by scripts/gen_ndvi_scene.py -- one
--    1024^2 window (10 m bands) of a real Sentinel-2 L2A scene near Turin, Italy
--    (2025-05-05), resolved from the EOPF STAC catalog exactly like the benchmark.
--    Regenerate/retarget with:  uv run scripts/gen_ndvi_scene.py
--    (We ship a LOCAL sample because the EOPF store can't be read remotely: its
--    Ceph/S3 endpoint rejects directory listing, and its only consolidated
--    metadata is at the product root, which mixes 10/20/60 m arrays into one
--    hierarchy we can't fold into a single coord cube. See README.)
--
--  Gotchas:
--    * b04/b08 are surface-reflectance floats with NaN over nodata (fill 0).
--      DataFusion treats NaN as EQUAL to NaN and GREATER than everything, so a
--      NaN pixel would poison MAX/AVG and survive a `b04 = b04` test. Filter with
--      `NOT isnan(...)` -- one test on (b08 - b04) covers both bands, since a NaN
--      in either makes the difference NaN.
--    * The bands are stored [x, y] (transposed from the source [y, x]) so the
--      alphabetical coord order (x, y) matches the data-variable dim order; the
--      reader would otherwise swap the x/y labels. NDVI is symmetric per pixel,
--      but the coordinates stay honest.
--    * x/y are UTM easting/northing in metres (int64); no dict-cast needed here.
--
--  Run:  zarr-cli cookbook/ndvi/ndvi.sql            # 1.05M rows -- redirect to a file
-- ============================================================================

CREATE EXTERNAL TABLE IF NOT EXISTS scene
  STORED AS ZARR
  LOCATION 'data/s2_ndvi_scene.zarr';

SELECT
  x,                                              -- UTM easting  (m)
  y,                                              -- UTM northing (m)
  ROUND((b08 - b04) / (b08 + b04), 4) AS ndvi     -- (NIR - Red) / (NIR + Red)
FROM scene
WHERE NOT isnan(b08 - b04)                        -- drop nodata (NaN in either band)
ORDER BY y, x;                                    -- image raster order (row-major)
