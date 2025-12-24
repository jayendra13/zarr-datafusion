# zarr-datafusion

A Rust library that integrates [Zarr v3](https://zarr.dev/) array storage with [Apache DataFusion](https://datafusion.apache.org/) for querying multidimensional scientific data using SQL.

## Overview

This library allows you to query Zarr stores (commonly used for weather, climate, and scientific datasets) using DataFusion's SQL engine. It flattens multidimensional arrays into a tabular format, enabling SQL queries over gridded data.

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
synthetic.zarr/
├── lat/          shape: [10]          → coordinate
├── lon/          shape: [10]          → coordinate
├── time/         shape: [7]           → coordinate
├── temperature/  shape: [10, 10, 7]   → data variable (lat × lon × time)
└── humidity/     shape: [10, 10, 7]   → data variable (lat × lon × time)
```

## Features

- **Zarr v3 support** via the [zarrs](https://crates.io/crates/zarrs) crate
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

    // Schema is automatically inferred from Zarr metadata
    let schema = Arc::new(infer_schema("data/synthetic.zarr").expect("Failed to infer schema"));
    let table = Arc::new(ZarrTable::new(schema, "data/synthetic.zarr"));

    ctx.register_table("synthetic", table)?;

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

First, generate sample Zarr data:

```bash
./scripts/generate_data.sh
```

Then run the examples:

```bash
cargo run --example query_synthetic  # Query synthetic data
cargo run --example query_era5       # Query ERA5 climate data
```

## Interactive CLI

An interactive SQL shell is included for exploring Zarr data:

The CLI automatically infers the schema from Zarr v3 metadata. Coordinate arrays (1D) become columns, and data arrays (nD) are flattened.

```bash
cargo run --bin zarr-cli
```

```
Zarr-DataFusion CLI
Registered tables:
  synthetic (data/synthetic.zarr) - columns: lat, lon, time, humidity, temperature
  era5 (data/era5.zarr) - columns: hybrid, latitude, longitude, time, geopotential, temperature

Type SQL queries or 'help' for commands.

zarr> show tables
zarr> SELECT * FROM synthetic LIMIT 5;
zarr> SELECT * FROM era5 LIMIT 5;
zarr> help
zarr> quit
```

### Example Queries

**Sample data:**
```sql
SELECT * FROM synthetic LIMIT 10;
```

**Filter by temperature:**
```sql
SELECT time, lat, lon, temperature
FROM synthetic
WHERE temperature > 20
LIMIT 10;
```

**Average temperature per time step:**
```sql
SELECT time, AVG(temperature) as avg_temp
FROM synthetic
GROUP BY time
ORDER BY time;
```

**Find locations where temperature is always below 10:**
```sql
SELECT lat, lon, MAX(temperature) as max_temp
FROM synthetic
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
FROM synthetic
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
│   └── zarr_reader.rs    # Low-level Zarr reading and Arrow conversion
├── datasource/
│   └── zarr.rs           # DataFusion TableProvider implementation
└── physical_plan/
    └── zarr_exec.rs      # DataFusion ExecutionPlan for scanning
```

## Dependencies

- `arrow` - Apache Arrow for columnar data
- `datafusion` - SQL query engine
- `zarrs` - Zarr v3 file format support
- `tokio` - Async runtime

## Roadmap

- [x] Add REPL for quick quries.
- [x] Support Schema Inference.
- [x] Projection Push down
- [ ] Filter push down
- [x] Memory efficient co-ord expansion (DictionaryArray)
- [ ] Zarr Codecs
- [ ] Zero Copy data
- [x] Read ERA5 climate dataset from local disk.
- [ ] Read ERA5 dataset from cloud storage (S3/GCS buckets).
- [ ] DMA while reading from Cloud
- [ ] Integrate [icechunk](https://github.com/earth-mover/icechunk) for transactional Zarr reads
- [ ] Tackle the [One Trillion Row Challenge](https://github.com/coiled/1trc) with Zarr + DataFusion
- [ ] Integrate with [xarray-sql](https://github.com/xarray-contrib/xarray-sql) for xarray interoperability

