# Freeze Evaluation Implementation Plan

## Goal

Replicate ExtremeWeatherBench's freeze evaluation (Case 30: 2021 Texas Freeze) using zarr-datafusion SQL queries with custom UDFs for metric calculations.

---

## Target Case: 2021 Texas Freeze (Case 30)

| Attribute | Value |
|-----------|-------|
| **Case ID** | 30 |
| **Event Type** | `freeze` |
| **Start Date** | `2021-02-10 12:00:00 UTC` |
| **End Date** | `2021-02-22 00:00:00 UTC` |
| **Latitude Range** | 24.0° to 54.75° N |
| **Longitude Range** | 250.0° to 278.75° E (or -110.0° to -81.25° W) |

---

## Data Sources

| Dataset | Location | Variable | Dimensions | Resolution |
|---------|----------|----------|------------|------------|
| **Forecast (FourCastNetv2)** | `data/FOUR_v200_GFS.parq` (VirtualiZarr) | `t2` (Kelvin) | (init_time, time, level, lat, lon) | 0.25°, 6h lead times |
| **Target (ERA5)** | `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` | `2m_temperature` (Kelvin) | (time, level, lat, lon) | 0.25°, hourly |

**Key:** Both datasets share the same 721×1440 spatial grid (0.25° resolution) - no regridding needed!

---

## Current State

- [x] VirtualiZarr support for forecast (FOUR_v200_GFS.parq)
- [x] GCS Zarr v3 support for ERA5
- [x] Same spatial grid (721×1440, 0.25° resolution)
- [x] Custom metric UDFs (mae, bias, rmse, spatial_mean, etc.)
- [ ] Time alignment & JOIN implementation
- [ ] Evaluation pipeline

---

## Join Strategy

### Alignment Overview

The alignment follows ExtremeWeatherBench's approach:

1. **Temporal Alignment**: Inner join on `time` (valid_time)
   - Forecast `time` = ERA5 `time`
   - Only keep overlapping timestamps

2. **Spatial Alignment**: Direct join (same grid)
   - `forecast.latitude = target.latitude`
   - `forecast.longitude = target.longitude`
   - No interpolation needed (both 0.25°)

3. **Bounding Box Filter**: Case 30 specific bounds
   - Time: `2021-02-10 12:00:00` to `2021-02-22 00:00:00`
   - Lat: `24.0` to `54.75`
   - Lon: `250.0` to `278.75` (0-360 format)

### SQL Join Pattern

```sql
-- Core alignment join for Case 30
SELECT
    f.init_time,
    f.time AS valid_time,
    f.latitude,
    f.longitude,
    f.t2 AS forecast_temp,
    e."2m_temperature" AS target_temp
FROM gfs f
INNER JOIN era5 e
    ON f.time = e.time
   AND f.latitude = e.latitude
   AND f.longitude = e.longitude
WHERE f.time >= TIMESTAMP '2021-02-10 12:00:00'
  AND f.time <= TIMESTAMP '2021-02-22 00:00:00'
  AND f.latitude BETWEEN 24.0 AND 54.75
  AND f.longitude BETWEEN 250.0 AND 278.75
```

---

## UDF Library (Implemented)

### Scalar Functions (`src/udfs/scalar.rs`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `mae(a, b)` | `(Float64, Float64) → Float64` | Absolute error \|a - b\| |
| `bias(a, b)` | `(Float64, Float64) → Float64` | Signed error (a - b) |
| `squared_error(a, b)` | `(Float64, Float64) → Float64` | (a - b)² |
| `grid_round(coord, res)` | `(Float64, Float64) → Float64` | Round to nearest grid |
| `kelvin_to_celsius(k)` | `(Float64) → Float64` | k - 273.15 |
| `is_freezing(k)` | `(Float64) → Boolean` | k < 273.15 |
| `within_window(t, center, hours)` | `(Int64, Int64, Int64) → Boolean` | Time window filter |

### Aggregate Functions (`src/udfs/aggregate.rs`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `rmse(a, b)` | `(Float64, Float64) → Float64` | Root Mean Squared Error |
| `mean_mae(a, b)` | `(Float64, Float64) → Float64` | Mean Absolute Error |
| `spatial_mean(val, lat)` | `(Float64, Float64) → Float64` | Area-weighted mean (cos lat) |

---

## Complete Evaluation SQL

### Step 1: Register Tables

```sql
-- Forecast (VirtualiZarr reference to S3)
CREATE EXTERNAL TABLE gfs STORED AS ZARR LOCATION 'data/FOUR_v200_GFS.parq';

-- Target (ERA5 from GCS)
CREATE EXTERNAL TABLE era5 STORED AS ZARR
    LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';
```

### Step 2: Create Aligned View for Case 30

```sql
CREATE VIEW case30_aligned AS
SELECT
    f.init_time,
    f.time AS valid_time,
    f.latitude,
    f.longitude,
    f.t2 AS forecast_temp,
    e."2m_temperature" AS target_temp
FROM gfs f
INNER JOIN era5 e
    ON f.time = e.time
   AND f.latitude = e.latitude
   AND f.longitude = e.longitude
WHERE f.time >= TIMESTAMP '2021-02-10 12:00:00'
  AND f.time <= TIMESTAMP '2021-02-22 00:00:00'
  AND f.latitude BETWEEN 24.0 AND 54.75
  AND f.longitude BETWEEN 250.0 AND 278.75;
```

### Step 3: Compute Standard Metrics

```sql
-- RMSE and MAE by init_time (lead time analysis)
SELECT
    init_time,
    rmse(forecast_temp, target_temp) AS rmse_k,
    mean_mae(forecast_temp, target_temp) AS mae_k,
    AVG(bias(forecast_temp, target_temp)) AS bias_k,
    COUNT(*) AS n_points
FROM case30_aligned
GROUP BY init_time
ORDER BY init_time;
```

### Step 4: Compute MinimumMeanAbsoluteError (Freeze Metric)

```sql
-- Daily minimum temperatures (spatial average first)
WITH spatial_ts AS (
    SELECT
        init_time,
        valid_time,
        DATE(valid_time) AS valid_date,
        spatial_mean(forecast_temp, latitude) AS forecast_spatial_mean,
        spatial_mean(target_temp, latitude) AS target_spatial_mean
    FROM case30_aligned
    GROUP BY init_time, valid_time, DATE(valid_time)
),

daily_mins AS (
    SELECT
        init_time,
        valid_date,
        MIN(forecast_spatial_mean) AS forecast_daily_min,
        MIN(target_spatial_mean) AS target_daily_min
    FROM spatial_ts
    GROUP BY init_time, valid_date
)

SELECT
    init_time,
    mean_mae(forecast_daily_min, target_daily_min) AS minimum_mae_k,
    kelvin_to_celsius(AVG(target_daily_min)) AS avg_target_min_c,
    kelvin_to_celsius(AVG(forecast_daily_min)) AS avg_forecast_min_c
FROM daily_mins
GROUP BY init_time
ORDER BY init_time;
```

### Step 5: Freezing Statistics

```sql
-- Count freezing hours and points
SELECT
    init_time,
    COUNT(*) AS total_points,
    SUM(CASE WHEN is_freezing(target_temp) THEN 1 ELSE 0 END) AS target_freezing_points,
    SUM(CASE WHEN is_freezing(forecast_temp) THEN 1 ELSE 0 END) AS forecast_freezing_points,
    100.0 * SUM(CASE WHEN is_freezing(target_temp) THEN 1 ELSE 0 END) / COUNT(*) AS target_freeze_pct
FROM case30_aligned
GROUP BY init_time
ORDER BY init_time;
```

---

## Implementation Tasks

### Phase 1: UDF Infrastructure ✅
- [x] Create `src/udfs/mod.rs` module structure
- [x] Implement scalar UDFs: `mae`, `bias`, `squared_error`, `grid_round`
- [x] Implement aggregate UDFs: `rmse`, `mean_mae`, `spatial_mean`
- [x] Add `kelvin_to_celsius`, `is_freezing`, `within_window`
- [x] Register UDFs in CLI and library context
- [x] Add unit tests for each UDF

### Phase 2: Join Implementation 🔄
- [ ] Test basic forecast-target join with small subset
- [ ] Verify temporal alignment (matching timestamps)
- [ ] Verify spatial alignment (matching coordinates)
- [ ] Handle level dimension (surface variables only)
- [ ] Optimize join performance with filter pushdown

### Phase 3: Metric Validation
- [ ] Compute RMSE and compare to ExtremeWeatherBench
- [ ] Compute MAE and compare to ExtremeWeatherBench
- [ ] Compute MinimumMAE and compare to ExtremeWeatherBench
- [ ] Document any discrepancies

### Phase 4: Extended Metrics
- [ ] Duration metrics (consecutive freezing hours)
- [ ] Percentile-based metrics
- [ ] Lead-time breakdown analysis

---

## Expected Output

| Column | Example Value |
|--------|---------------|
| case_id | 30 |
| event_type | freeze |
| metric | MinimumMeanAbsoluteError |
| init_time | 2021-02-10T00:00:00 |
| value | 2.3 K |
| forecast_source | FourCastNetv2 |
| target_source | ERA5 |

**Output rows:** ~82 (41 lead times × 2 metrics)

---

## Notes

- Both datasets use Kelvin for temperature
- ERA5 longitude is 0-360 format (not -180 to 180)
- The `level` dimension exists but is singleton for surface variables (t2, 2m_temperature)
- UDFs are registered automatically when using the CLI
- `spatial_mean` uses cos(latitude) weighting for proper area averaging
