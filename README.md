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
┌─────────────────────────────────────────────────────────┐
│  Zarr Store (3D)           →    SQL Table (2D)                        │
├─────────────────────────────────────────────────────────              ┤
│  temperature[t, lat, lon]  →    | timestamp | lat | lon | temperature |
│  humidity[t, lat, lon]     →    | 0         | 0   | 0   | 43          |
│                            →    | 0         | 0   | 1   | 51          |
│                            →    | ...       | ... | ... | ...         |
└─────────────────────────────────────────────────────────┘
```

## Features

- **Zarr v3 support** via the [zarrs](https://crates.io/crates/zarrs) crate
- **Projection pushdown**: Only reads arrays that are needed for the query
- **SQL interface**: Full DataFusion SQL support (filtering, aggregation, joins, etc.)

## Usage

```rust
use std::sync::Arc;
use datafusion::prelude::SessionContext;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::zarr_reader::zarr_weather_schema;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();

    let schema = Arc::new(zarr_weather_schema());
    let table = Arc::new(ZarrTable::new(schema, "data/weather.zarr"));

    ctx.register_table("weather", table)?;

    // Query with SQL
    let df = ctx.sql("SELECT timestamp, lat, lon, temperature
                      FROM weather
                      WHERE temperature > 5
                      LIMIT 10").await?;
    df.show().await?;

    Ok(())
}
```

## Example Output

```
Sample data (first 10 rows):
+-----------+-----+-----+-------------+----------+
| timestamp | lat | lon | temperature | humidity |
+-----------+-----+-----+-------------+----------+
| 0         | 0   | 0   | 43          | 55       |
| 0         | 0   | 1   | 51          | 62       |
| 0         | 0   | 2   | 39          | 70       |
| ...       | ... | ... | ...         | ...      |
+-----------+-----+-----+-------------+----------+

Average temperature per day:
+-----------+----------+
| timestamp | avg_temp |
+-----------+----------+
| 0         | 5.35     |
| 1         | 3.08     |
| 2         | 0.88     |
| ...       | ...      |
+-----------+----------+
```

## Building

```bash
cargo build
```

## Running the Example

First, generate sample Zarr data:

```bash
uv run --with zarr --with numpy data_gen.py
```

Then run the example:

```bash
cargo run --example query_zarr
```

## Architecture

```
src/
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

- [ ] Read ERA5 climate dataset from local disk
- [ ] Read ERA5 dataset from cloud storage (S3/GCS buckets)
- [ ] Integrate [icechunk](https://github.com/earth-mover/icechunk) for transactional Zarr reads
- [ ] Tackle the [One Trillion Row Challenge](https://github.com/coiled/1trc) with Zarr + DataFusion
- [ ] Integrate with [xarray-sql](https://github.com/xarray-contrib/xarray-sql) for xarray interoperability
