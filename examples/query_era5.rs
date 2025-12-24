use std::sync::Arc;

use datafusion::prelude::SessionContext;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::reader::schema_inference::infer_schema;

async fn run_query(
    ctx: &SessionContext,
    description: &str,
    sql: &str,
) -> datafusion::error::Result<()> {
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

    // Load ERA5 data from Zarr v3 store
    let store_path = "data/era5.zarr";
    let schema = Arc::new(infer_schema(store_path).expect("Failed to infer schema"));

    println!("ERA5 Schema:");
    for field in schema.fields() {
        println!("  {}: {:?}", field.name(), field.data_type());
    }

    let table = Arc::new(ZarrTable::new(schema, store_path));
    ctx.register_table("era5", table)?;

    // Query 1: Sample data overview
    run_query(
        &ctx,
        "Sample ERA5 data (first 10 rows):",
        "SELECT * FROM era5 LIMIT 10",
    )
    .await?;

    // Query 2: Average temperature by hybrid level (pressure level)
    run_query(
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

    Ok(())
}
