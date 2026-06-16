mod common;

use std::time::Instant;

const LOCAL_STORE: &str = "data/era5_sst_local.zarr";

const LAT_MIN: f64 = -5.0;
const LAT_MAX: f64 = 5.0;
const LON_MIN: f64 = 190.0;
const LON_MAX: f64 = 240.0;

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    common::init_tracing();
    let ctx = common::create_remote_context();

    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION '{LOCAL_STORE}'"
    ))
    .await?
    .collect()
    .await?;

    // Single query using EXTRACT — replaces the 86-sub-query UNION ALL
    let sql = format!(
        r#"
        SELECT AVG(sea_surface_temperature - 273.15) AS sst_celsius
        FROM era5
        WHERE EXTRACT(MONTH FROM time) = 12
          AND latitude  BETWEEN {LAT_MIN} AND {LAT_MAX}
          AND longitude BETWEEN {LON_MIN} AND {LON_MAX}
        "#
    );

    println!("Niño 3.4 December SST — EXTRACT approach");
    println!("=========================================");
    println!("Store  : {LOCAL_STORE}");
    println!("Region : lat [{LAT_MIN}, {LAT_MAX}]  lon [{LON_MIN}, {LON_MAX}]");
    println!("Filter : EXTRACT(MONTH FROM time) = 12");
    println!();
    println!("SQL:{sql}");

    let t = Instant::now();
    ctx.sql(&sql).await?.show().await?;
    println!("completed in {:?}", t.elapsed());

    Ok(())
}
