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

use datafusion::prelude::*;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::{fmt, EnvFilter};
use zarr_datafusion::datasource::factory::ZarrTableFactory;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber with env filter
    // Use RUST_LOG env var to control log level:
    //   RUST_LOG=info   - show info and above
    //   RUST_LOG=debug  - show debug and above
    //   RUST_LOG=zarr_datafusion=debug - only zarr_datafusion debug logs
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(true)
        .with_thread_ids(false)
        .with_line_number(true)
        .init();

    // Create DataFusion context with Zarr support
    let ctx = SessionContext::new();
    ctx.state_ref()
        .write()
        .table_factories_mut()
        .insert("ZARR".to_string(), Arc::new(ZarrTableFactory));

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

    // Show schema
    println!("Schema:");
    println!("-------");
    let df = ctx.sql("DESCRIBE era5").await?;
    df.show().await?;

    // Query with LIMIT (efficient - only reads necessary data)
    println!("\nQuery: SELECT hybrid, latitude, longitude, time, temperature FROM era5 LIMIT 5");
    println!("------------------------------------------------------------------------");
    let start = std::time::Instant::now();
    let df = ctx
        .sql("SELECT hybrid, latitude, longitude, time, temperature FROM era5 LIMIT 5")
        .await?;
    df.show().await?;
    println!("Query completed in {:?}\n", start.elapsed());

    // Show dataset info
    // println!("Dataset Info:");
    // println!("-------------");
    // println!("- Source: ECMWF ERA5 Reanalysis");
    // println!("- Resolution: 0.25° x 0.25° global grid");
    // println!("- Levels: 137 hybrid sigma-pressure levels");
    // println!("- Time: Hourly from 1940 to present (~1.3M timesteps)");
    // println!("- Variables: temperature, humidity, wind, pressure, etc.");
    // println!("\nNote: First rows show NaN because ERA5 has missing data at poles (lat=90°)");

    Ok(())
}
