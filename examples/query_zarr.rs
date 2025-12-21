use std::sync::Arc;

use datafusion::prelude::SessionContext;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::zarr_reader::zarr_weather_schema;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();

    // Use the schema from the library
    let schema = Arc::new(zarr_weather_schema());

    // Point to the Zarr v3 store created by data_gen.py
    let table = Arc::new(ZarrTable::new(schema, "data/weather.zarr"));

    ctx.register_table("weather", table)?;

    // Query: select all columns
    println!("Sample data (first 10 rows):");
    let df = ctx
        .sql("SELECT * FROM weather LIMIT 10")
        .await?;
    df.show().await?;

    // Query: select temperature readings with coordinates
    println!("\nFiltered data (temperature > 5):");
    let df = ctx
        .sql("SELECT timestamp, lat, lon, temperature FROM weather WHERE temperature > 5 LIMIT 10")
        .await?;
    df.show().await?;

    // Query: average temperature per timestamp
    println!("\nAverage temperature per day:");
    let df = ctx
        .sql("SELECT timestamp, AVG(temperature) as avg_temp FROM weather GROUP BY timestamp ORDER BY timestamp")
        .await?;
    df.show().await?;

    // Query: count using a column (avoids empty projection issue)
    println!("\nTotal rows:");
    let df = ctx.sql("SELECT COUNT(temperature) as total FROM weather").await?;
    df.show().await?;

    Ok(())
}