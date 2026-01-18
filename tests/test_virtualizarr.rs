//! Test VirtualiZarr integration

mod common;

use arrow::util::pretty::print_batches;
use datafusion::prelude::SessionContext;
use std::sync::Arc;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema_with_meta;

const VIRTUALIZARR_PATH: &str = "data/FOUR_v200_GFS.parq";

fn setup_virtualizarr_table(ctx: &SessionContext) -> datafusion::error::Result<()> {
    let (schema, metadata) =
        infer_schema_with_meta(VIRTUALIZARR_PATH).expect("Failed to infer schema");

    let schema = Arc::new(schema);
    let table = Arc::new(ZarrTable::with_metadata(
        schema.clone(),
        VIRTUALIZARR_PATH,
        metadata,
    ));
    ctx.register_table("gfs", table)?;
    Ok(())
}

#[tokio::test]
async fn test_virtualizarr_schema_inference() -> datafusion::error::Result<()> {
    let (schema, metadata) =
        infer_schema_with_meta(VIRTUALIZARR_PATH).expect("Failed to infer schema");

    // Verify we detected the expected coordinates and data variables
    let coord_names: Vec<_> = metadata.coords.iter().map(|c| c.name.as_str()).collect();
    let data_var_names: Vec<_> = metadata.data_vars.iter().map(|d| d.name.as_str()).collect();

    println!("Coordinates: {:?}", coord_names);
    println!("Data vars: {:?}", data_var_names);

    assert!(coord_names.contains(&"init_time"));
    assert!(coord_names.contains(&"time"));
    assert!(coord_names.contains(&"latitude"));
    assert!(coord_names.contains(&"longitude"));
    assert!(data_var_names.contains(&"t2"));

    // Verify schema has expected fields
    assert!(schema.field_with_name("t2").is_ok());
    assert!(schema.field_with_name("init_time").is_ok());

    Ok(())
}

#[tokio::test]
async fn test_virtualizarr_simple_query() -> datafusion::error::Result<()> {
    let ctx = common::create_test_context();
    setup_virtualizarr_table(&ctx)?;

    // Simple query with just t2 (the main variable)
    let df = ctx.sql("SELECT t2 FROM gfs LIMIT 5").await?;
    let results = df.collect().await?;

    assert!(!results.is_empty());
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 5);

    println!("Got {} rows of t2 data", total_rows);
    print_batches(&results)?;

    Ok(())
}

#[tokio::test]
async fn test_virtualizarr_with_coordinates() -> datafusion::error::Result<()> {
    let ctx = common::create_test_context();
    setup_virtualizarr_table(&ctx)?;

    // Query with coordinates
    let df = ctx
        .sql("SELECT init_time, time, latitude, longitude, t2 FROM gfs LIMIT 10")
        .await?;
    let results = df.collect().await?;

    assert!(!results.is_empty());
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 10);

    println!("Got {} rows with coordinates", total_rows);
    print_batches(&results)?;

    Ok(())
}

#[tokio::test]
async fn test_virtualizarr_init_time_as_timestamp() -> datafusion::error::Result<()> {
    let ctx = common::create_test_context();
    setup_virtualizarr_table(&ctx)?;

    // init_time should now be automatically detected as a nanosecond epoch timestamp
    // via the heuristic (column name contains "time" + int64 dtype)
    let df = ctx.sql("SELECT init_time FROM gfs LIMIT 5").await?;
    let results = df.collect().await?;

    assert!(!results.is_empty());
    println!("init_time (auto-detected as timestamp):");
    print_batches(&results)?;

    // Verify the first value is a reasonable timestamp (2020-09-30)
    let batch = &results[0];
    let col = batch.column(0);
    println!("Column type: {:?}", col.data_type());

    Ok(())
}
