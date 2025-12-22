use std::sync::Arc;

use datafusion::prelude::SessionContext;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema;

async fn run_query(ctx: &SessionContext, description: &str, sql: &str) -> datafusion::error::Result<()> {
    println!("\n{description}");
    println!("SQL: {sql}");
    println!();
    let df = ctx.sql(sql).await?;
    df.show().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();

    // Infer schema from Zarr metadata
    let store_path = "data/weather.zarr";
    let schema = Arc::new(infer_schema(store_path).expect("Failed to infer schema"));

    // Point to the Zarr v3 store created by data_gen.py
    let table = Arc::new(ZarrTable::new(schema, store_path));

    ctx.register_table("weather", table)?;

    run_query(
        &ctx,
        "Sample data (first 10 rows):",
        "SELECT * FROM weather LIMIT 10",
    )
    .await?;

    run_query(
        &ctx,
        "Filtered data (temperature > 5):",
        "SELECT time, lat, lon, temperature FROM weather WHERE temperature > 5 LIMIT 10",
    )
    .await?;

    run_query(
        &ctx,
        "Average temperature per day:",
        "SELECT time, AVG(temperature) as avg_temp FROM weather GROUP BY time ORDER BY time",
    )
    .await?;

    run_query(
        &ctx,
        "Total rows:",
        "SELECT COUNT(temperature) as total FROM weather",
    )
    .await?;

    Ok(())
}
