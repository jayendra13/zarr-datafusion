# zarr-datafusion

[![CI](https://github.com/jayendra13/zarr-datafusion/actions/workflows/test.yml/badge.svg)](https://github.com/jayendra13/zarr-datafusion/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/jayendra13/zarr-datafusion/graph/badge.svg)](https://codecov.io/gh/jayendra13/zarr-datafusion)

A Rust library that integrates [Zarr](https://zarr.dev/) (v2 and v3) array storage with [Apache DataFusion](https://datafusion.apache.org/) for querying multidimensional scientific data using SQL.

## Overview

This library allows you to query Zarr stores (commonly used for weather, climate, and scientific datasets) using DataFusion's SQL engine. It flattens multidimensional arrays into a tabular format, enabling SQL queries over gridded data. Both Zarr v2 and v3 formats are supported.

### How It Works

Zarr stores multidimensional data in chunked arrays. For example, a weather dataset might have:
- **Coordinate arrays**: `time`, `lat`, `lon` (1D)
- **Data arrays**: `temperature`, `humidity` (3D: time × lat × lon)

This library flattens the 3D structure into rows where each row represents one grid cell:

```
┌───────────────────────────────────────────────────────────────────────┐
│  Zarr Store (3D)           →    SQL Table (2D)                        │
├───────────────────────────────────────────────────────────────────────┤
│  temperature[t, lat, lon]  →    | timestamp | lat | lon | temperature |
│  humidity[t, lat, lon]     →    | 0         | 0   | 0   | 43          |
│                            →    | 0         | 0   | 1   | 51          |
│                            →    | ...       | ... | ... | ...         |
└───────────────────────────────────────────────────────────────────────┘
```

### Assumptions

> **Note:** This library currently assumes a specific Zarr store structure:

1. **Coordinates are 1D arrays** — Any array with a single dimension is treated as a coordinate (e.g., `time(7)`, `lat(10)`, `lon(10)`)

2. **Data variables are nD arrays** — Arrays with multiple dimensions are treated as data variables. Their dimensionality must equal the number of coordinate arrays.

3. **Cartesian product structure** — Data variables are assumed to represent the Cartesian product of all coordinates. For coordinates `[lat(10), lon(10), time(7)]`, data variables must have shape `[10, 10, 7]`.

4. **Dimension ordering** — Coordinates are sorted alphabetically, and data variable dimensions are assumed to follow this same order.

```
# Zarr v3 structure
synthetic_v3.zarr/
├── zarr.json                 → group metadata
├── lat/zarr.json             → array metadata (shape: [10])
├── lon/zarr.json             → array metadata (shape: [10])
├── time/zarr.json            → array metadata (shape: [7])
├── temperature/zarr.json     → array metadata (shape: [10, 10, 7])
└── humidity/zarr.json        → array metadata (shape: [10, 10, 7])

# Zarr v2 structure
synthetic_v2.zarr/
├── .zgroup                   → group metadata
├── .zattrs                   → group attributes
├── lat/.zarray               → array metadata (shape: [10])
├── lon/.zarray               → array metadata (shape: [10])
├── time/.zarray              → array metadata (shape: [7])
├── temperature/.zarray       → array metadata (shape: [10, 10, 7])
└── humidity/.zarray          → array metadata (shape: [10, 10, 7])
```

## Features

- **Zarr v2 and v3 support** via the [zarrs](https://crates.io/crates/zarrs) crate
- **Schema inference**: Automatically infers Arrow schema from Zarr metadata
- **Projection pushdown**: Only reads arrays that are needed for the query
- **Memory efficient coordinates**: Uses Arrow DictionaryArray for coordinate columns (~75% memory savings)
- **SQL interface**: Full DataFusion SQL support (filtering, aggregation, joins, etc.)

## Usage

```rust
use std::sync::Arc;
use datafusion::prelude::SessionContext;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();

    // Schema is automatically inferred from Zarr metadata (v2 or v3)
    let schema = infer_schema("data/synthetic_v3.zarr")
        .expect("Failed to infer schema");
    let table = ZarrTable::new(Arc::new(schema), "data/synthetic_v3.zarr");

    ctx.register_table("synthetic", Arc::new(table))?;

    // Query with SQL
    let df = ctx.sql("SELECT time, lat, lon, temperature
                      FROM synthetic
                      WHERE temperature > 5
                      LIMIT 10").await?;
    df.show().await?;

    Ok(())
}
```

## Example Output

```
Sample data (first 10 rows):
+-----+-----+------+----------+-------------+
| lat | lon | time | humidity | temperature |
+-----+-----+------+----------+-------------+
| 0   | 0   | 0    | 21       | 52          |
| 0   | 0   | 1    | 34       | 1           |
| 0   | 0   | 2    | 61       | 42          |
| ... | ... | ...  | ...      | ...         |
+-----+-----+------+----------+-------------+

Average temperature per day:
+------+----------+
| time | avg_temp |
+------+----------+
| 0    | 7.07     |
| 1    | 3.69     |
| 2    | 1.38     |
| ...  | ...      |
+------+----------+
```

## Building

```bash
cargo build
```

## Running the Example

First, generate sample Zarr data (creates 8 dataset variations):

```bash
./scripts/generate_data.sh
```

This generates:
```
data/
├── synthetic_v2.zarr       # Zarr v2, no compression
├── synthetic_v2_blosc.zarr # Zarr v2, Blosc/LZ4 compression
├── synthetic_v3.zarr       # Zarr v3, no compression
├── synthetic_v3_blosc.zarr # Zarr v3, Blosc/LZ4 compression
├── era5_v2.zarr            # ERA5 climate data, Zarr v2
├── era5_v2_blosc.zarr      # ERA5 climate data, Zarr v2 + Blosc
├── era5_v3.zarr            # ERA5 climate data, Zarr v3
└── era5_v3_blosc.zarr      # ERA5 climate data, Zarr v3 + Blosc
```

Then run the examples:

```bash
cargo run --example query_synthetic  # Query synthetic data
cargo run --example query_era5       # Query ERA5 climate data
```

## Interactive CLI

An interactive SQL shell is included for exploring Zarr data:

```bash
cargo run --bin zarr-cli
```

### Loading Zarr Data

Use standard SQL `CREATE EXTERNAL TABLE` syntax to load Zarr stores (v2 or v3):

```sql
zarr> CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION 'data/synthetic_v3.zarr';
zarr> SHOW TABLES;
zarr> SELECT * FROM weather LIMIT 5;
zarr> DROP TABLE weather;
```

The schema is automatically inferred from Zarr metadata (v2 or v3). Coordinate arrays (1D) become columns, and data arrays (nD) are flattened.

```
Zarr-DataFusion CLI

Type SQL queries or 'help' for commands.

zarr> CREATE EXTERNAL TABLE synthetic_v2 STORED AS ZARR LOCATION 'data/synthetic_v2.zarr';
zarr> CREATE EXTERNAL TABLE synthetic_v3 STORED AS ZARR LOCATION 'data/synthetic_v3.zarr';
zarr> CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION 'data/era5_v3.zarr';
zarr> SHOW TABLES;
zarr> SELECT * FROM synthetic_v2 LIMIT 5;
zarr> help
zarr> quit
```

### Example Queries

**Sample data:**
```sql
SELECT * FROM synthetic_v3 LIMIT 10;
```

**Filter by temperature:**
```sql
SELECT time, lat, lon, temperature
FROM synthetic_v3
WHERE temperature > 20
LIMIT 10;
```

**Average temperature per time step:**
```sql
SELECT time, AVG(temperature) as avg_temp
FROM synthetic_v3
GROUP BY time
ORDER BY time;
```

**Find locations where temperature is always below 10:**
```sql
SELECT lat, lon, MAX(temperature) as max_temp
FROM synthetic_v3
GROUP BY lat, lon
HAVING MAX(temperature) < 10
ORDER BY max_temp;
```

**Temperature statistics by location:**
```sql
SELECT lat, lon,
       MIN(temperature) as min_temp,
       MAX(temperature) as max_temp,
       AVG(temperature) as avg_temp
FROM synthetic_v3
GROUP BY lat, lon
ORDER BY avg_temp DESC
LIMIT 10;
```

## Architecture

```
src/
├── bin/
│   └── zarr_cli.rs       # Interactive SQL shell (REPL)
├── reader/
│   ├── schema_inference.rs  # Infer Arrow schema from Zarr metadata
│   └── zarr_reader.rs       # Low-level Zarr reading and Arrow conversion
├── datasource/
│   ├── zarr.rs           # DataFusion TableProvider implementation
│   └── factory.rs        # TableProviderFactory for CREATE EXTERNAL TABLE
└── physical_plan/
    └── zarr_exec.rs      # DataFusion ExecutionPlan for scanning
```

## Dependencies

- `arrow` - Apache Arrow for columnar data
- `datafusion` - SQL query engine
- `zarrs` - Zarr v2/v3 file format support
- `tokio` - Async runtime

## Roadmap

### Completed
- [x] Add REPL for quick queries
- [x] Support Schema Inference
- [x] Memory efficient co-ord expansion (DictionaryArray)
- [x] Read ERA5 climate dataset from local disk
- [x] Zarr v2 support (without codecs)
- [x] Zarr Codecs (Blosc, etc.)

### Pushdown Optimizations
- [x] Projection pushdown (only read requested columns)
- [x] Limit pushdown (slice results to limit)
- [ ] Filter pushdown
  - [ ] Coordinate equality (`WHERE lat = 5`)
  - [ ] Coordinate range (`WHERE time BETWEEN 2 AND 4`)
  - [ ] Partition pruning (skip chunks based on coordinate ranges)
  - [ ] Data variable filter (`WHERE temperature > 20`)
- [ ] Aggregate pushdown (push `SUM/AVG/COUNT` to chunk level)
- [ ] Top-K optimization (`ORDER BY x LIMIT k` without full sort)

### REPL Experience
- [ ] Tab completion (tables, columns, SQL keywords)
- [x] Syntax highlighting
- [ ] Multi-line query editing
- [x] Query history persistence (~/.zarr_cli_history)
- [ ] Output formats (table, csv, json, parquet)
- [x] Timing statistics (`5 rows returned in 0.012s`)
- [ ] Progress bar for long-running queries
- [ ] `.schema <table>` command for quick schema view
- [ ] Pager support for large results (less/more)

### Performance
- [ ] Chunk-level parallelism (read chunks concurrently)
- [ ] Streaming RecordBatch output (multiple batches instead of one)
- [ ] Zero-copy reads with memory-mapped I/O
- [ ] Statistics-based chunk pruning

### Data Types
- [ ] Additional numeric types (uint8/16/32, int8/16/32, float16)
- [ ] String/datetime coordinates
- [ ] Handle fill_value as Arrow nulls
- [ ] Expose Zarr attributes in Arrow schema metadata

### Cloud & Storage
- [ ] Read from cloud storage (S3/GCS/Azure) via `object_store` crate
- [ ] HTTP/HTTPS Zarr backend
- [ ] Async chunk prefetching
- [ ] LRU cache for frequently accessed chunks

### Interoperability
- [ ] Integrate [icechunk](https://github.com/earth-mover/icechunk) for transactional Zarr reads
- [ ] [Kerchunk](https://github.com/fsspec/kerchunk)/VirtualiZarr support (virtual references to NetCDF/HDF5)
- [ ] Integrate with [xarray-sql](https://github.com/xarray-contrib/xarray-sql)
- [ ] Python bindings via PyO3
- [ ] Arrow Flight server

### Challenges
- [ ] Tackle the [One Trillion Row Challenge](https://github.com/coiled/1trc) with Zarr + DataFusion

