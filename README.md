# zarr-datafusion

[![Crates.io](https://img.shields.io/crates/v/zarr-datafusion.svg)](https://crates.io/crates/zarr-datafusion)
[![docs.rs](https://docs.rs/zarr-datafusion/badge.svg)](https://docs.rs/zarr-datafusion)
[![CI](https://github.com/jayendra13/zarr-datafusion/actions/workflows/test.yml/badge.svg)](https://github.com/jayendra13/zarr-datafusion/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/jayendra13/zarr-datafusion/graph/badge.svg)](https://codecov.io/gh/jayendra13/zarr-datafusion)
![MSRV](https://img.shields.io/badge/rust-1.88+-orange.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/jayendra13/zarr-datafusion)](https://github.com/jayendra13/zarr-datafusion/commits/main)

A Rust library that integrates [Zarr](https://zarr.dev/) (v2 and v3) array storage with [Apache DataFusion](https://datafusion.apache.org/) for querying multidimensional scientific data using SQL.

## Quick Start

```bash
# Install the prebuilt CLI from a GitHub release (no Rust toolchain needed).
# The Linux build is a static musl binary, so it runs on any x86_64 Linux
# regardless of the system glibc version — ideal for deploying to a VM.
# The current build is a preview prerelease, so pin the version explicitly:
curl -fsSL https://raw.githubusercontent.com/jayendra13/zarr-datafusion/main/install.sh \
  | VERSION=v0.1.0-test bash
zarr-cli --version

# Or run from source (requires the Rust toolchain)
cargo run --bin zarr-cli
```

> Release: <https://github.com/jayendra13/zarr-datafusion/releases/tag/v0.1.0-test>
>
> `install.sh` honours `VERSION=<tag>` to pin a release and `INSTALL_DIR=/usr/local/bin`
> to change the install location (defaults to `~/.local/bin`). A `cargo install
> zarr-datafusion` path will be available once the crate is published to crates.io.

```sql
-- Load a Zarr store (local or cloud)
zarr> CREATE EXTERNAL TABLE era5 STORED AS ZARR
      LOCATION 'gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1';

-- Explore schema with extended metadata
zarr> DESCRIBE era5;

-- Query with SQL
zarr> SELECT latitude, longitude, AVG(2m_temperature)
      FROM era5
      WHERE time > '2020-01-01'
      GROUP BY latitude, longitude
      LIMIT 10;
```

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
- **Projection & limit pushdown**: Only reads the arrays a query needs, and stops early on `LIMIT`
- **Filter pushdown**: Coordinate equality, range (`BETWEEN`, `>=`), and date-part filters (e.g. `day`, `month`) prune chunks before reading
- **Optimizer rules**: `MIN`/`MAX`/`COUNT` answered from statistics, skipping the data scan
- **Chunk-level parallelism**: The scan is split into partitions read concurrently
- **Distributed execution** (optional): Multi-node querying via [datafusion-distributed](https://crates.io/crates/datafusion-distributed) (`worker`/`head` binaries, `distributed` feature)
- **VirtualiZarr support**: Reads virtual reference stores (Parquet chunk refs → NetCDF/GRIB byte ranges)
- **Memory efficient coordinates**: Uses Arrow DictionaryArray for coordinate columns (~75% memory savings)
- **SQL interface**: Full DataFusion SQL support (filtering, aggregation, joins, etc.)
- **Cloud storage**: Read directly from GCS and S3 buckets
- **Interactive CLI**: SQL shell with syntax highlighting, history, and I/O statistics
- **Prebuilt binaries**: Static musl `zarr-cli` from GitHub releases (no toolchain needed)

## Prerequisites

### Rust

Install Rust via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### uv (for test data generation)

[uv](https://docs.astral.sh/uv/) is used to run Python scripts for generating test data. Install it:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

The Python scripts use zarr, numpy, xarray, and other scientific packages. These are installed automatically by uv when running the scripts.

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

## Pre-commit hooks

To enable a Git pre-commit hook that runs `cargo fmt`, `cargo clippy`, and `cargo test`, run:

```sh
./scripts/install-hooks.sh
```

This sets `git config core.hooksPath .githooks` and makes the pre-commit hook executable. The hook runs checks and prevents commits if formatting, lints, or tests fail.

---

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

### Features

- **SQL syntax highlighting** for better readability
- **Command history** persisted to `~/.zarr_cli_history`
- **Live I/O statistics** during query execution
- **Extended DESCRIBE** with Zarr metadata (type, dimensions, chunk sizes)

### Loading Zarr Data

Use standard SQL `CREATE EXTERNAL TABLE` syntax to load Zarr stores:

```sql
zarr> CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION 'data/synthetic_v3.zarr';
zarr> SHOW TABLES;
zarr> SELECT * FROM weather LIMIT 5;
zarr> DROP TABLE weather;
```

Load data from cloud storage (GCS, S3):

```sql
zarr> CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION 'gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1';
zarr> SELECT * FROM era5 LIMIT 10;
```

### Extended DESCRIBE

The `DESCRIBE` command shows extended Zarr metadata including variable type, dimensions, sizes, and chunk configuration:

```sql
zarr> DESCRIBE era5;
```

Output:
```
+-------------------------+------------------------------------+-------------+----------+------------------------------------------+-------------+-----------------+
| column_name             | data_type                          | is_nullable | type     | dimension                                | size        | chunks          |
+-------------------------+------------------------------------+-------------+----------+------------------------------------------+-------------+-----------------+
| time                    | Dictionary(Int16, Timestamp(...))  | NO          | coord    | (time)                                   | 82920       | (82920)         |
| latitude                | Dictionary(Int16, Float64)         | NO          | coord    | (latitude)                               | 721         | (721)           |
| longitude               | Dictionary(Int16, Float64)         | NO          | coord    | (longitude)                              | 1440        | (1440)          |
| 2m_temperature          | Float32                            | YES         | data_var | (time: 82920, latitude: 721, ...)        | 86090860800 | (160, 145, 144) |
+-------------------------+------------------------------------+-------------+----------+------------------------------------------+-------------+-----------------+
```

You can also use `zarr_describe()` directly in SQL:

```sql
zarr> SELECT column_name, type, dimension FROM zarr_describe('era5') WHERE type = 'coord';
```

### I/O Statistics

After each query, statistics are displayed:

```
5 rows · 3 arrays · 6.70 KB disk · 13.92 KB mem · 0.013s
```

This shows rows returned, arrays read, compressed bytes from disk, uncompressed bytes in memory, and execution time.

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

## Cookbook

End-to-end, reproducible recipes live under [`cookbook/`](cookbook/):

- **[El Niño / La Niña — ONI from ERA5](cookbook/el-nino-oni/)** — compute the
  Oceanic Niño Index for every overlapping 3-month season from 1950 to present
  with a single SQL query over public ERA5 on GCS, classify each season as El Niño
  / La Niña / Neutral, and validate against NOAA's official ONI table (MAE 0.14 °C
  in the satellite era, Pearson r 0.96, class-agreement weighted κ 0.85).

## Distributed Execution

For larger scans, `zarr-datafusion` can run as a multi-node cluster via
[datafusion-distributed](https://crates.io/crates/datafusion-distributed): a `head`
node plans the query and runs the top stage locally, fanning out scan stages to
`worker` nodes. Both binaries are gated behind the `distributed` feature.

The quickest way to try it locally is the helper script, which builds and runs a
cluster of plain cargo processes:

```bash
scripts/cluster.sh up                                  # build + start workers
scripts/cluster.sh query "SELECT COUNT(*) FROM weather"
scripts/cluster.sh query "SELECT ..." --show-plan      # print the distributed plan
scripts/cluster.sh down                                # stop workers

# Tunables (env vars): WORKERS=3  BASE_PORT=9090  STORE_PATH=...  TABLE=weather
```

Or run the binaries directly:

```bash
# Worker — repeat on each node (PORT defaults to 8080)
PORT=8080 cargo run --features distributed --bin worker

# Head — point it at the workers + a store, then run SQL
WORKER_URLS=http://localhost:8080,http://localhost:8081 \
  cargo run --features distributed --bin head -- \
  --store data/synthetic_v3.zarr --table weather \
  "SELECT time, AVG(temperature) FROM weather GROUP BY time"
```

## Architecture

```
src/
├── lib.rs                       # Crate root; module exports
├── bin/
│   ├── zarr_cli/
│   │   ├── main.rs              # Interactive SQL shell (REPL) + script runner
│   │   └── highlight.rs         # SQL syntax highlighting
│   ├── worker.rs                # Distributed worker node   (feature = "distributed")
│   └── head.rs                  # Distributed head/coordinator (feature = "distributed")
├── reader/
│   ├── schema_inference.rs      # Infer Arrow schema from Zarr metadata
│   ├── zarr_reader.rs           # Zarr reading, nD→2D flattening, Arrow conversion
│   ├── filter.rs                # Coordinate filter pushdown (equality, range, date-part)
│   ├── coord.rs                 # Coordinate handling
│   ├── dtype.rs                 # Zarr → Arrow dtype conversion
│   ├── cf_time.rs               # CF time-units decoding ("hours since 1900-01-01")
│   ├── storage.rs               # Storage backends (local, GCS, S3)
│   ├── tracked_store.rs         # I/O-tracking store wrapper (sync)
│   ├── async_tracked_store.rs   # I/O-tracking store wrapper (async)
│   ├── virtual_store.rs         # VirtualiZarr reference-store adapter
│   ├── parquet_refs.rs          # VirtualiZarr Parquet chunk-reference reader
│   └── stats.rs                 # I/O statistics tracking
├── datasource/
│   ├── zarr.rs                  # ZarrTable: DataFusion TableProvider
│   └── factory.rs               # TableProviderFactory for CREATE EXTERNAL TABLE
├── optimizer/
│   ├── minmax_optimization.rs   # MIN/MAX → constant folding from stats
│   ├── count_optimization.rs    # COUNT(*) → constant folding from stats
│   └── limit_pushdown.rs        # Push LIMIT into the Zarr scan
├── physical_plan/
│   ├── zarr_exec.rs             # ZarrExec: partitioned scan ExecutionPlan
│   ├── partition.rs             # Scan partitioning / chunk selection
│   └── codec.rs                 # Physical-plan codec for distributed execution
├── distributed.rs               # datafusion-distributed wiring (worker/head)
├── udfs/                        # Scalar + aggregate UDFs (rmse, mae, ...)
└── udtf.rs                      # zarr_describe() table function
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
- [x] Read from cloud storage (GCS/S3) via `object_store` crate
- [x] Extended DESCRIBE with Zarr metadata (`zarr_describe()` UDTF)
- [x] MIN/MAX/COUNT optimizer rules (use statistics, skip data scan)
- [x] Distributed execution across `worker`/`head` nodes (datafusion-distributed)
- [x] VirtualiZarr reference stores (Parquet chunk references)
- [x] Prebuilt static-musl `zarr-cli` binaries via GitHub releases

### Pushdown Optimizations
- [x] Projection pushdown (only read requested columns)
- [x] Limit pushdown (slice results to limit)
- [x] Filter pushdown for coordinate equality (`WHERE lat = 5`)
- [x] Coordinate range filters (`WHERE time BETWEEN 2 AND 4`, `>=`)
- [x] Date-part coordinate filters (`WHERE extract(month FROM time) IN (...)`)
- [x] Partition pruning (skip chunks outside the coordinate selection)
- [ ] Data variable filter pushdown (`WHERE temperature > 20`)
- [ ] Aggregate pushdown (push `SUM/AVG/COUNT` to chunk level)
- [ ] Top-K optimization (`ORDER BY x LIMIT k` without full sort)

### REPL Experience
- [x] Syntax highlighting
- [x] Query history persistence (~/.zarr_cli_history)
- [x] Timing and I/O statistics
- [x] Live progress during query execution
- [x] Extended DESCRIBE with Zarr metadata
- [ ] Tab completion (tables, columns, SQL keywords)
- [ ] Multi-line query editing
- [ ] Output formats (table, csv, json, parquet)
- [ ] Pager support for large results (less/more)

### Performance
- [x] Chunk-level parallelism (partitioned scan, chunks read concurrently)
- [ ] Streaming RecordBatch output (multiple batches instead of one)
- [ ] Zero-copy reads with memory-mapped I/O
- [ ] Statistics-based chunk pruning (data-variable min/max)

### Data Types
- [ ] Additional numeric types (uint8/16/32, int8/16/32, float16)
- [ ] String/datetime coordinates
- [ ] Handle fill_value as Arrow nulls
- [ ] Expose Zarr attributes in Arrow schema metadata

### Cloud & Storage
- [x] Read from cloud storage (GCS/S3) via `object_store` crate
- [ ] HTTP/HTTPS Zarr backend
- [ ] Async chunk prefetching
- [ ] LRU cache for frequently accessed chunks

### Interoperability
- [ ] Integrate [icechunk](https://github.com/earth-mover/icechunk) for transactional Zarr reads
- [x] [Kerchunk](https://github.com/fsspec/kerchunk)/VirtualiZarr support — VirtualiZarr Parquet references to NetCDF/GRIB
- [ ] Integrate with [xarray-sql](https://github.com/xarray-contrib/xarray-sql)
- [ ] Python bindings via PyO3
- [ ] Arrow Flight server

### Challenges
- [ ] Tackle the [One Trillion Row Challenge](https://github.com/coiled/1trc) with Zarr + DataFusion

