# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

zarr-datafusion is a Rust library integrating Zarr v3 array storage with Apache DataFusion for querying multidimensional scientific data using SQL. It flattens 3D gridded data (time × lat × lon) into a 2D tabular format.

## Build Commands

```bash
cargo build                      # Build the library
cargo run --example query_zarr   # Run the example
cargo test                       # Run all tests
cargo clippy                     # Run linter
cargo fmt                        # Format code
```

## Architecture

```
src/
├── reader/zarr_reader.rs    # Zarr reading, flattening 3D→2D, Arrow conversion
├── datasource/zarr.rs       # ZarrTable: DataFusion TableProvider
└── physical_plan/zarr_exec.rs # ZarrExec: DataFusion ExecutionPlan
```

**Data flow**: SQL query → ZarrTable.scan() → ZarrExec.execute() → read_zarr() → RecordBatch

**Projection pushdown**: The reader only loads arrays that are needed for the query (e.g., `SELECT temperature` won't read humidity data).

## Test Data

```bash
uv run --with zarr --with numpy data_gen.py
```

Generates `data/weather.zarr` with 5 arrays: time(7), lat(10), lon(10), temperature(7×10×10), humidity(7×10×10). Uses seed=42 for reproducible random data.
