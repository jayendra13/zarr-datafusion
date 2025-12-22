# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

zarr-datafusion is a Rust library integrating Zarr v3 array storage with Apache DataFusion for querying multidimensional scientific data using SQL. It flattens nD gridded data into a 2D tabular format.

## Build Commands

```bash
cargo build                      # Build the library
cargo run --example query_zarr   # Run the example
cargo run --bin zarr-cli         # Run interactive SQL CLI
cargo test                       # Run all tests
cargo clippy                     # Run linter
cargo fmt                        # Format code
```

## Architecture

```
src/
├── bin/zarr_cli.rs              # Interactive SQL REPL
├── reader/
│   ├── schema_inference.rs      # Infer Arrow schema from Zarr metadata
│   └── zarr_reader.rs           # Zarr reading, flattening nD→2D, Arrow conversion
├── datasource/zarr.rs           # ZarrTable: DataFusion TableProvider
└── physical_plan/zarr_exec.rs   # ZarrExec: DataFusion ExecutionPlan
```

**Data flow**: SQL query → ZarrTable.scan() → ZarrExec.execute() → read_zarr() → RecordBatch

**Key features**:
- **Schema inference**: Automatically detects coordinates (1D) and data variables (nD) from Zarr metadata
- **Projection pushdown**: Only loads arrays needed for the query
- **DictionaryArray for coordinates**: Memory-efficient encoding (~75% savings)

## Assumptions

The library assumes a specific Zarr store structure:

1. **Coordinates are 1D arrays**: Any `shape.len() == 1` array is a coordinate
2. **Data variables are nD arrays**: Dimensionality equals number of coordinates
3. **Cartesian product**: Data variables represent `coord1 × coord2 × coord3`
4. **Alphabetical ordering**: Coordinates sorted by name, data dimensions follow same order

```
weather.zarr/
├── lat/          shape: [10]          → coordinate
├── lon/          shape: [10]          → coordinate
├── time/         shape: [7]           → coordinate
├── temperature/  shape: [10, 10, 7]   → data variable (lat × lon × time)
└── humidity/     shape: [10, 10, 7]   → data variable (lat × lon × time)
```

## Test Data

```bash
uv run --with zarr --with numpy data_gen.py
```

Generates `data/weather.zarr` with 5 arrays: time(7), lat(10), lon(10), temperature(7×10×10), humidity(7×10×10). Uses seed=42 for reproducible random data.

## Key Types

- `ZarrTable` — DataFusion TableProvider for Zarr stores
- `ZarrExec` — Physical execution plan
- `infer_schema()` — Infers Arrow schema from Zarr v3 metadata
- `read_zarr()` — Reads Zarr arrays into Arrow RecordBatch
- `DictionaryArray<Int16Type>` — Used for coordinate columns (keys=indices, values=unique coords)
