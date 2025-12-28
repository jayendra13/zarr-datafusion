# Zarr-DataFusion CLI Reference

Interactive SQL shell for querying Zarr stores using DataFusion.

```bash
cargo run --bin zarr-cli
```

---

## Table of Contents

- [CLI Commands](#cli-commands)
- [Loading Data](#loading-data)
- [Basic Queries](#basic-queries)
- [Filtering](#filtering)
- [Aggregations](#aggregations)
- [Grouping and Ordering](#grouping-and-ordering)
- [Window Functions](#window-functions)
- [Common Table Expressions (CTEs)](#common-table-expressions-ctes)
- [Joins](#joins)
- [Table Inspection](#table-inspection)
- [Query Analysis](#query-analysis)

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `help` | Show help message |
| `quit` / `exit` | Exit the CLI |
| `show tables` / `\d` | List all registered tables |

---

## Loading Data

### Create External Table

Load a Zarr store (v2 or v3) as a table:

```sql
CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION 'data/synthetic_v3.zarr';
```

Load multiple tables:

```sql
CREATE EXTERNAL TABLE synthetic_v2 STORED AS ZARR LOCATION 'data/synthetic_v2.zarr';
CREATE EXTERNAL TABLE synthetic_v3 STORED AS ZARR LOCATION 'data/synthetic_v3.zarr';
CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION 'data/era5_v3.zarr';
```

### Drop Table

Remove a registered table:

```sql
DROP TABLE weather;
```

---

## Basic Queries

### Select All Columns

```sql
SELECT * FROM weather LIMIT 10;
```

### Select Specific Columns

```sql
SELECT time, lat, lon, temperature FROM weather LIMIT 10;
```

### Aliasing Columns

```sql
SELECT
    temperature AS temp_kelvin,
    humidity AS relative_humidity
FROM weather
LIMIT 5;
```

### Arithmetic Operations

```sql
SELECT
    temperature,
    temperature - 273.15 AS temp_celsius,
    (temperature - 273.15) * 9/5 + 32 AS temp_fahrenheit
FROM weather
LIMIT 5;
```

### Distinct Values

```sql
SELECT DISTINCT lat FROM weather ORDER BY lat;
SELECT DISTINCT lon FROM weather ORDER BY lon;
SELECT DISTINCT time FROM weather ORDER BY time;
```

---

## Filtering

### Simple WHERE Clause

```sql
SELECT * FROM weather WHERE temperature > 20 LIMIT 10;
```

### Multiple Conditions (AND)

```sql
SELECT * FROM weather
WHERE temperature > 10
  AND humidity < 50
LIMIT 10;
```

### Multiple Conditions (OR)

```sql
SELECT * FROM weather
WHERE temperature > 40
   OR temperature < -30
LIMIT 10;
```

### Range Filter (BETWEEN)

```sql
SELECT * FROM weather
WHERE temperature BETWEEN 0 AND 25
LIMIT 10;
```

### IN Clause

```sql
SELECT * FROM weather
WHERE time IN (0, 1, 2)
LIMIT 10;
```

### NOT IN Clause

```sql
SELECT * FROM weather
WHERE lat NOT IN (0, 9)
LIMIT 10;
```

### NULL Handling

```sql
SELECT * FROM weather WHERE temperature IS NOT NULL LIMIT 10;
```

---

## Aggregations

### Count

```sql
SELECT COUNT(*) AS total_rows FROM weather;
SELECT COUNT(temperature) AS non_null_temps FROM weather;
```

### Sum

```sql
SELECT SUM(temperature) AS total_temp FROM weather;
```

### Average

```sql
SELECT AVG(temperature) AS avg_temp FROM weather;
SELECT AVG(humidity) AS avg_humidity FROM weather;
```

### Min / Max

```sql
SELECT
    MIN(temperature) AS min_temp,
    MAX(temperature) AS max_temp
FROM weather;
```

### Multiple Aggregations

```sql
SELECT
    COUNT(*) AS count,
    AVG(temperature) AS avg_temp,
    MIN(temperature) AS min_temp,
    MAX(temperature) AS max_temp,
    AVG(humidity) AS avg_humidity
FROM weather;
```

---

## Grouping and Ordering

### GROUP BY Single Column

```sql
SELECT
    time,
    AVG(temperature) AS avg_temp
FROM weather
GROUP BY time
ORDER BY time;
```

### GROUP BY Multiple Columns

```sql
SELECT
    lat,
    lon,
    AVG(temperature) AS avg_temp,
    AVG(humidity) AS avg_humidity
FROM weather
GROUP BY lat, lon
ORDER BY lat, lon;
```

### HAVING Clause

Filter groups after aggregation:

```sql
SELECT
    lat,
    lon,
    MAX(temperature) AS max_temp
FROM weather
GROUP BY lat, lon
HAVING MAX(temperature) < 10
ORDER BY max_temp;
```

### ORDER BY with ASC/DESC

```sql
SELECT
    lat,
    lon,
    AVG(temperature) AS avg_temp
FROM weather
GROUP BY lat, lon
ORDER BY avg_temp DESC
LIMIT 10;
```

### LIMIT and OFFSET

```sql
SELECT * FROM weather LIMIT 10 OFFSET 20;
```

---

## Window Functions

### Row Number

```sql
SELECT
    time,
    lat,
    lon,
    temperature,
    ROW_NUMBER() OVER (PARTITION BY time ORDER BY temperature DESC) AS rank
FROM weather
LIMIT 20;
```

### Running Average

```sql
SELECT
    time,
    temperature,
    AVG(temperature) OVER (ORDER BY time ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
FROM weather
WHERE lat = 0 AND lon = 0
ORDER BY time;
```

### Rank and Dense Rank

```sql
SELECT
    lat,
    lon,
    temperature,
    RANK() OVER (ORDER BY temperature DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY temperature DESC) AS dense_rank
FROM weather
WHERE time = 0
LIMIT 20;
```

### Lead and Lag

```sql
SELECT
    time,
    temperature,
    LAG(temperature, 1) OVER (ORDER BY time) AS prev_temp,
    LEAD(temperature, 1) OVER (ORDER BY time) AS next_temp
FROM weather
WHERE lat = 0 AND lon = 0
ORDER BY time;
```

---

## Common Table Expressions (CTEs)

### Simple CTE

```sql
WITH daily_avg AS (
    SELECT time, AVG(temperature) AS avg_temp
    FROM weather
    GROUP BY time
)
SELECT * FROM daily_avg ORDER BY time;
```

### Multiple CTEs

```sql
WITH
hot_spots AS (
    SELECT lat, lon, AVG(temperature) AS avg_temp
    FROM weather
    GROUP BY lat, lon
    HAVING AVG(temperature) > 10
),
cold_spots AS (
    SELECT lat, lon, AVG(temperature) AS avg_temp
    FROM weather
    GROUP BY lat, lon
    HAVING AVG(temperature) < 0
)
SELECT 'hot' AS type, COUNT(*) AS count FROM hot_spots
UNION ALL
SELECT 'cold' AS type, COUNT(*) AS count FROM cold_spots;
```

### Recursive CTE (if supported)

```sql
WITH RECURSIVE numbers AS (
    SELECT 0 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 5
)
SELECT * FROM numbers;
```

---

## Joins

### Self Join

Compare data across different time steps:

```sql
SELECT
    a.lat,
    a.lon,
    a.temperature AS temp_t0,
    b.temperature AS temp_t1,
    b.temperature - a.temperature AS temp_change
FROM weather a
JOIN weather b ON a.lat = b.lat AND a.lon = b.lon
WHERE a.time = 0 AND b.time = 1
LIMIT 10;
```

### Join Multiple Tables

```sql
CREATE EXTERNAL TABLE v2_data STORED AS ZARR LOCATION 'data/synthetic_v2.zarr';
CREATE EXTERNAL TABLE v3_data STORED AS ZARR LOCATION 'data/synthetic_v3.zarr';

SELECT
    v2.lat,
    v2.lon,
    v2.time,
    v2.temperature AS v2_temp,
    v3.temperature AS v3_temp
FROM v2_data v2
JOIN v3_data v3
    ON v2.lat = v3.lat
    AND v2.lon = v3.lon
    AND v2.time = v3.time
LIMIT 10;
```

### Left Join

```sql
SELECT a.*, b.temperature AS other_temp
FROM weather a
LEFT JOIN weather b
    ON a.lat = b.lat + 1
    AND a.lon = b.lon
    AND a.time = b.time
LIMIT 10;
```

---

## Table Inspection

### Describe Table Schema

```sql
DESCRIBE weather;
```

### Show All Tables

```sql
SHOW TABLES;
```

### Show Table Columns (Alternative)

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'weather';
```

---

## Query Analysis

### EXPLAIN Query Plan

```sql
EXPLAIN SELECT AVG(temperature) FROM weather GROUP BY time;
```

### EXPLAIN ANALYZE (with execution stats)

```sql
EXPLAIN ANALYZE SELECT * FROM weather WHERE temperature > 20 LIMIT 100;
```

### EXPLAIN VERBOSE

```sql
EXPLAIN VERBOSE SELECT lat, lon, AVG(temperature) FROM weather GROUP BY lat, lon;
```

---

## Example Session

```
$ cargo run --bin zarr-cli

Zarr-DataFusion CLI

Type SQL queries or 'help' for commands.

zarr> CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION 'data/synthetic_v3.zarr';
zarr> SHOW TABLES;
+---------------+
| table_name    |
+---------------+
| weather       |
+---------------+

zarr> DESCRIBE weather;
+-------------+-----------------------+
| column_name | data_type             |
+-------------+-----------------------+
| lat         | Dictionary(Int16,Int64)|
| lon         | Dictionary(Int16,Int64)|
| time        | Dictionary(Int16,Int64)|
| humidity    | Int64                 |
| temperature | Int64                 |
+-------------+-----------------------+

zarr> SELECT time, AVG(temperature) as avg_temp FROM weather GROUP BY time ORDER BY time;
+------+----------+
| time | avg_temp |
+------+----------+
| 0    | 7.07     |
| 1    | 3.69     |
| 2    | 1.38     |
| 3    | 4.49     |
| 4    | 9.37     |
| 5    | 0.16     |
| 6    | 6.36     |
+------+----------+

zarr> DROP TABLE weather;
zarr> quit
Goodbye!
```

---

## Notes

- Schema is automatically inferred from Zarr metadata (v2 or v3)
- Coordinate arrays (1D) become `Dictionary(Int16, T)` columns for memory efficiency
- Data arrays (nD) are flattened to their native Arrow types
- Projection pushdown: only requested columns are read from disk
- Both compressed (Blosc/LZ4) and uncompressed Zarr stores are supported
