//! Example: Query Zarr data from Google Cloud Storage
//!
//! This example demonstrates querying the ERA5 climate reanalysis dataset
//! stored as Zarr on Google Cloud Storage using SQL.
//!
//! Dataset: ARCO-ERA5 (Analysis-Ready, Cloud Optimized ERA5)
//! URL: gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1
//!
//! Run with:
//!   cargo run --example query_gcs
//!
//! Run with tracing enabled:
//!   RUST_LOG=info cargo run --example query_gcs
//!   RUST_LOG=debug cargo run --example query_gcs
//!   RUST_LOG=zarr_datafusion=debug cargo run --example query_gcs

mod common;

use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::init_tracing();
    let ctx = common::create_remote_context();

    println!("Zarr-DataFusion GCS Example");
    println!("============================\n");
    info!("Starting GCS example");

    // Register ERA5 dataset from GCS (public bucket, no credentials needed)
    let gcs_url = "gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1";
    println!("Registering ERA5 dataset from GCS...");
    println!("URL: {}\n", gcs_url);

    let start = std::time::Instant::now();
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION '{}'",
        gcs_url
    ))
    .await?
    .collect()
    .await?;
    println!("Table registered in {:?}\n", start.elapsed());

    // Show schema with extended Zarr metadata
    println!("Schema:");
    println!("-------");
    let df = ctx.sql("SELECT * FROM zarr_describe('era5')").await?;
    df.show().await?;

    // Sample query with LIMIT
    let query = "SELECT * FROM era5 LIMIT 10;";
    println!("\nExecuting query (optimized - uses statistics, no data scan):");
    println!("{}", query);
    println!("------------------------------------------------------------------------");
    let start = std::time::Instant::now();
    let df = ctx.sql(query).await?;
    df.show().await?;
    println!("Query completed in {:?}\n", start.elapsed());

    Ok(())
}
