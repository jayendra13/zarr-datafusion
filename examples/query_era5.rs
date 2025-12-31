//! Example: Query ERA5 climate data using SQL
//!
//! Run with tracing enabled:
//!   RUST_LOG=info cargo run --example query_era5
//!   RUST_LOG=debug cargo run --example query_era5

mod common;

use std::sync::Arc;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema_with_meta;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    common::init_tracing();
    let ctx = common::create_local_context();

    // Load ERA5 data from Zarr v3 store with metadata for statistics
    let store_path = "data/era5_v3.zarr";
    let (schema, metadata) = infer_schema_with_meta(store_path).expect("Failed to infer schema");
    let schema = Arc::new(schema);

    println!("ERA5 Schema:");
    for field in schema.fields() {
        println!("  {}: {:?}", field.name(), field.data_type());
    }
    println!("Total rows: {}", metadata.total_rows);

    let table = Arc::new(ZarrTable::with_metadata(schema, store_path, metadata));
    ctx.register_table("era5", table)?;

    // Query 1: Sample data overview
    common::run_query(
        &ctx,
        "Sample ERA5 data (first 10 rows):",
        "SELECT * FROM era5 LIMIT 10",
    )
    .await?;

    // Query 2: Average temperature by hybrid level (pressure level)
    common::run_query(
        &ctx,
        "Average temperature by hybrid level:",
        "SELECT hybrid,
                AVG(temperature) as avg_temp,
                MIN(temperature) as min_temp,
                MAX(temperature) as max_temp
         FROM era5
         GROUP BY hybrid
         ORDER BY hybrid",
    )
    .await?;

    // Query 3: Count (optimized - uses statistics)
    common::run_query(
        &ctx,
        "Total rows (optimized - uses statistics, no data scan):",
        "SELECT COUNT(*) as total FROM era5",
    )
    .await?;

    Ok(())
}
