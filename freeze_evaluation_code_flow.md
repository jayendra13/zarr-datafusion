# Freeze Event Evaluation - Code Flow

This document maps the key code locations for freeze event evaluation in ExtremeWeatherBench, focused on the **first freeze event (Case 30: 2021 Texas Freeze)**.

---

## Target Case: 2021 Texas Freeze

| Attribute | Value |
|-----------|-------|
| **Case ID** | 30 |
| **Title** | 2021 Texas |
| **Start Date** | `2021-02-10 12:00:00` |
| **End Date** | `2021-02-22 00:00:00` |
| **Latitude Min** | 24.0° |
| **Latitude Max** | 54.75° |
| **Longitude Min** | 250.0° (or -110.0° in -180/180) |
| **Longitude Max** | 278.75° (or -81.25° in -180/180) |
| **Event Type** | `freeze` |

---

## Data Sources

### Forecast: FourCastNetv2

| Attribute | Value |
|-----------|-------|
| **Kerchunk Reference** | `gs://extremeweatherbench/FOUR_v200_GFS.parq` |
| **Actual Data** | `s3://noaa-oar-mlwp-data/FOUR_v200_GFS/2021/02*/FOUR_v200_GFS_202102*.nc` |
| **Variable** | `t2` → `surface_air_temperature` (Kelvin) |
| **Dimensions** | `(init_time, lead_time, latitude, longitude)` |
| **Resolution** | 0.25° global, 6-hourly lead times (0h-240h) |

**Subset Query:**
```python
# Forecast subset for Case 30
forecast.sel(
    init_time=slice("2021-01-31", "2021-02-22"),  # buffer for lead times
    latitude=slice(54.75, 24.0),                   # descending order
    longitude=slice(250.0, 278.75),
)
```

### Target: ERA5

| Attribute | Value |
|-----------|-------|
| **Source** | `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` |
| **Variable** | `2m_temperature` → `surface_air_temperature` (Kelvin) |
| **Dimensions** | `(time, latitude, longitude)` |
| **Resolution** | 0.25°, hourly |

**Subset Query:**
```python
# ERA5 subset for Case 30
era5.sel(
    time=slice("2021-02-10T12:00:00", "2021-02-22T00:00:00"),
    latitude=slice(54.75, 24.0),
    longitude=slice(250.0, 278.75),
)
```

### Target: GHCN (Point Observations)

| Attribute | Value |
|-----------|-------|
| **Source** | `gs://extremeweatherbench/datasets/ghcnh_all_2020_2024.parq` |
| **Variable** | `TMIN` → `surface_air_temperature` (converted to Kelvin) |
| **Format** | Parquet with columns: `date`, `station_id`, `latitude`, `longitude`, `TMIN` |

**Subset Query:**
```sql
SELECT * FROM ghcn
WHERE date BETWEEN '2021-02-10' AND '2021-02-22'
  AND latitude BETWEEN 24.0 AND 54.75
  AND longitude BETWEEN -110.0 AND -81.25  -- GHCN uses -180/180 format
```

---

## Key Code Locations

### 1. Reading Forecast Data

**File:** `src/extremeweatherbench/inputs.py`

| Line | Function | Description |
|------|----------|-------------|
| **406-411** | `KerchunkForecast._open_data_from_source()` | Opens kerchunk parquet reference |
| **950-992** | `open_kerchunk_reference()` | Calls `xr.open_dataset(engine="kerchunk")` |
| **307-370** | `ForecastBase.subset_data_to_case()` | Subsets forecast to case time/space bounds |

```python
# inputs.py:970-975 - Opening forecast
kerchunk_ds = xr.open_dataset(
    "gs://extremeweatherbench/FOUR_v200_GFS.parq",
    engine="kerchunk",
    storage_options={"remote_protocol": "s3", "remote_options": {"anon": True}},
    chunks="auto",
)
```

---

### 2. Reading Target Data (ERA5)

**File:** `src/extremeweatherbench/inputs.py`

| Line | Function | Description |
|------|----------|-------------|
| **475-479** | `ERA5._open_data_from_source()` | Opens ERA5 Zarr from GCS |
| **483-492** | `ERA5.subset_data_to_case()` | Subsets ERA5 to case bounds |
| **995-1037** | `zarr_target_subsetter()` | Applies time/space filters |

```python
# inputs.py:476-479 - Opening ERA5
data = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    storage_options={"anon": True},
    chunks=None,
)
```

---

### 3. Spatio-Temporal Alignment

**File:** `src/extremeweatherbench/inputs.py`

| Line | Function | Description |
|------|----------|-------------|
| **1040-1087** | `align_forecast_to_target()` | Main alignment function |
| **1061-1066** | - | Temporal alignment (`xr.align(join="inner")`) |
| **1072-1083** | - | Spatial alignment (`xr.interp(method="nearest")`) |

```python
# inputs.py:1061-1066 - Temporal alignment (inner join)
time_aligned_target, time_aligned_forecast = xr.align(
    target_data,
    forecast_data,
    join="inner",
    exclude={"latitude", "longitude"},
)

# inputs.py:1077-1083 - Spatial alignment (regrid forecast to target grid)
time_space_aligned_forecast = time_aligned_forecast.interp(
    latitude=target_data.latitude,
    longitude=target_data.longitude,
    method="nearest",
    kwargs={"fill_value": "extrapolate"},
)
```

---

### 4. Computing Metrics

**File:** `src/extremeweatherbench/metrics.py`

| Line | Function | Description |
|------|----------|-------------|
| **263-310** | `MinimumMeanAbsoluteError.compute_metric()` | MAE of daily minimum temps |
| **213-240** | `RootMeanSquaredError.compute_metric()` | RMSE of all temps |
| **312-360** | `DurationMeanError.compute_metric()` | Error in freeze duration |

```python
# metrics.py - MinimumMeanAbsoluteError for freeze
class MinimumMeanAbsoluteError(BaseMetric):
    def compute_metric(self, forecast, target, ...):
        forecast_min = forecast.resample(valid_time="1D").min()
        target_min = target.resample(valid_time="1D").min()
        return np.abs(forecast_min - target_min).mean(...)
```

---

### 5. Orchestration

**File:** `src/extremeweatherbench/evaluate.py`

| Line | Function | Description |
|------|----------|-------------|
| **259-340** | `compute_case_operator()` | Main evaluation entry point |
| **676-780** | `_build_datasets()` | Builds forecast & target datasets |

---

## Complete Call Flow

```
evaluate_cli.py:184  →  ewb.run()
         │
         ▼
evaluate.py:142      →  _run_serial()
         │
         ▼
evaluate.py:259      →  compute_case_operator(case_30)
         │
         ├───────────────────────────────────────────────────────────┐
         ▼                                                           ▼
evaluate.py:676      →  _build_datasets()                   evaluate.py:305
         │                                                           │
         ├── Open forecast (kerchunk)                                │
         │   └── Subset: init_time 2021-01-31 to 2021-02-22         │
         │   └── Subset: lat 24.0-54.75, lon 250.0-278.75           │
         │                                                           │
         └── Open ERA5 (zarr)                                        │
             └── Subset: time 2021-02-10 to 2021-02-22              │
             └── Subset: lat 24.0-54.75, lon 250.0-278.75           │
                                                                     │
         ┌───────────────────────────────────────────────────────────┘
         ▼
inputs.py:1040       →  align_forecast_to_target()
         │
         ├── Temporal: xr.align(join="inner")
         │   └── Keep only matching valid_times
         │
         └── Spatial: xr.interp(method="nearest")
             └── Regrid forecast to ERA5 0.25° grid
         │
         ▼
metrics.py:263       →  MinimumMeanAbsoluteError.compute_metric()
metrics.py:213       →  RootMeanSquaredError.compute_metric()
         │
         ▼
DataFrame: {case_id: 30, metric: "MinimumMAE", lead_time: 24h, value: 2.3, ...}
```

---

## SQL Equivalent for Case 30

```sql
-- Register data sources
CREATE EXTERNAL TABLE forecast STORED AS ZARR
LOCATION 'freeze_forecast.zarr';

CREATE EXTERNAL TABLE era5 STORED AS ZARR
LOCATION 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3';

-- Case 30 constraints
WITH case_bounds AS (
    SELECT
        TIMESTAMP '2021-02-10 12:00:00' AS start_date,
        TIMESTAMP '2021-02-22 00:00:00' AS end_date,
        24.0 AS lat_min,
        54.75 AS lat_max,
        250.0 AS lon_min,
        278.75 AS lon_max
),

-- Subset and align
aligned_data AS (
    SELECT
        f.init_time,
        f.lead_time,
        f.latitude,
        f.longitude,
        f.surface_air_temperature AS forecast_temp,
        e.surface_air_temperature AS target_temp,
        f.init_time + f.lead_time AS valid_time
    FROM forecast f
    CROSS JOIN case_bounds c
    INNER JOIN era5 e
        ON (f.init_time + f.lead_time) = e.time
       AND f.latitude = e.latitude
       AND f.longitude = e.longitude
    WHERE e.time BETWEEN c.start_date AND c.end_date
      AND e.latitude BETWEEN c.lat_min AND c.lat_max
      AND e.longitude BETWEEN c.lon_min AND c.lon_max
),

-- Compute daily minimums for MinimumMAE
daily_mins AS (
    SELECT
        lead_time,
        DATE(valid_time) AS date,
        MIN(forecast_temp) AS forecast_daily_min,
        MIN(target_temp) AS target_daily_min
    FROM aligned_data
    GROUP BY lead_time, DATE(valid_time)
)

-- Final metrics
SELECT
    30 AS case_id,
    'freeze' AS event_type,
    lead_time,
    'MinimumMeanAbsoluteError' AS metric,
    AVG(ABS(forecast_daily_min - target_daily_min)) AS value,
    'FourCastNetv2' AS forecast_source,
    'ERA5' AS target_source
FROM daily_mins
GROUP BY lead_time

UNION ALL

SELECT
    30 AS case_id,
    'freeze' AS event_type,
    lead_time,
    'RootMeanSquaredError' AS metric,
    SQRT(AVG(POWER(forecast_temp - target_temp, 2))) AS value,
    'FourCastNetv2' AS forecast_source,
    'ERA5' AS target_source
FROM aligned_data
GROUP BY lead_time

ORDER BY metric, lead_time;
```

---

## Expected Output Shape

| Column | Example Value |
|--------|---------------|
| case_id | 30 |
| event_type | freeze |
| metric | MinimumMeanAbsoluteError |
| lead_time | 24h |
| value | 2.3 |
| forecast_source | FourCastNetv2 |
| target_source | ERA5 |

**Output rows:** ~82 (41 lead times × 2 metrics)
