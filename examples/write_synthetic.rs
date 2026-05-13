//! Example: Write a tiny Zarr v3 dataset, then query it with SQL.
//!
//! Demonstrates the POC writer end-to-end:
//!   1. Build coords + data vars in memory.
//!   2. write_zarr_v3 → on-disk Zarr v3 store.
//!   3. Register the store as a DataFusion table and run a few queries.
//!
//! Run with:
//!   cargo run --example write_synthetic
//!   RUST_LOG=zarr_datafusion=debug cargo run --example write_synthetic

mod common;

use std::path::Path;
use std::sync::Arc;

use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema_with_meta;
use zarr_datafusion::writer::{write_zarr_v3, CoordSpec, DataVarSpec, WriteValues};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    common::init_tracing();

    // Use a fresh temp dir each run; the POC writer errors if the path exists.
    let mut path = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    path.push(format!("zarr_write_example_{nanos}.zarr"));

    write_demo_dataset(&path)?;
    println!("Wrote synthetic Zarr v3 store at: {}", path.display());

    // === Query it back via the existing reader ===
    let path_str = path.to_str().unwrap();
    let (schema, metadata) = infer_schema_with_meta(path_str)?;
    let schema = Arc::new(schema);

    println!("\nSchema:");
    for field in schema.fields() {
        println!("  {}: {:?}", field.name(), field.data_type());
    }
    println!("Total rows: {}", metadata.total_rows);

    let ctx = common::create_local_context();
    ctx.register_table(
        "written",
        Arc::new(ZarrTable::with_metadata(schema, path_str, metadata)),
    )?;

    common::run_query(
        &ctx,
        "First 5 rows:",
        "SELECT * FROM written ORDER BY lat, lon, time LIMIT 5",
    )
    .await?;

    common::run_query(
        &ctx,
        "Per-time averages:",
        "SELECT time, AVG(temperature) AS avg_temp, AVG(humidity) AS avg_hum \
         FROM written GROUP BY time ORDER BY time",
    )
    .await?;

    common::run_query(
        &ctx,
        "Coord bounds (optimized: served from statistics):",
        "SELECT MIN(lat) AS lat_min, MAX(lat) AS lat_max, \
                MIN(lon) AS lon_min, MAX(lon) AS lon_max FROM written",
    )
    .await?;

    // Tidy up so reruns succeed and we don't litter /tmp.
    std::fs::remove_dir_all(&path).ok();
    Ok(())
}

fn write_demo_dataset(path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // lat(4) × lon(3) × time(2) = 24 rows
    let lat: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
    let lon: Vec<f64> = vec![100.0, 110.0, 120.0];
    let time: Vec<i64> = vec![1_700_000_000, 1_700_003_600]; // epoch seconds

    let mut temperature: Vec<f32> = Vec::with_capacity(lat.len() * lon.len() * time.len());
    let mut humidity: Vec<f64> = Vec::with_capacity(lat.len() * lon.len() * time.len());
    for (li, &la) in lat.iter().enumerate() {
        for (oi, &lo) in lon.iter().enumerate() {
            for (ti, _) in time.iter().enumerate() {
                temperature.push((la + 0.1 * lo + ti as f64) as f32);
                humidity.push(50.0 + (li + oi + ti) as f64);
            }
        }
    }

    let coords = vec![
        CoordSpec {
            name: "lat".into(),
            values: WriteValues::F64(lat),
        },
        CoordSpec {
            name: "lon".into(),
            values: WriteValues::F64(lon),
        },
        CoordSpec {
            name: "time".into(),
            values: WriteValues::I64(time),
        },
    ];
    let data_vars = vec![
        DataVarSpec {
            name: "temperature".into(),
            values: WriteValues::F32(temperature),
            shape: vec![4, 3, 2],
        },
        DataVarSpec {
            name: "humidity".into(),
            values: WriteValues::F64(humidity),
            shape: vec![4, 3, 2],
        },
    ];
    write_zarr_v3(path, &coords, &data_vars)?;
    Ok(())
}
