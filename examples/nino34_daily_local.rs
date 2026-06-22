//! Example: Niño 3.4 daily-mean SST — December 2025, fully local
//!
//! Computes the daily mean Sea Surface Temperature over the Niño 3.4 region
//! for December 2025 from the *local* ARCO-ERA5 Zarr store
//! (`data/era5_sst_local.zarr`). No network access — all 744 hourly chunks
//! for Dec 2025 are present on disk.
//!
//! Populate the local store first if needed:
//!   uv run --with aiohttp --with tqdm --with requests \
//!          scripts/download_sst_local.py --year-months '2025-12' --concurrency 32
//!
//! ## Niño 3.4 region
//!   Latitude  :  5°S – 5°N    (latitude  BETWEEN  -5.0 AND   5.0)
//!   Longitude : 170°W – 120°W (longitude BETWEEN 190.0 AND 240.0, 0-360 system)
//!
//! ## Expected result (verified against the local store)
//!   31 rows (one per December day), each aggregating
//!   24 hours × 41 lat × 201 lon = 197,784 cells.
//!   Daily means run ~25.8–26.2 °C; whole-month mean ≈ 25.93 °C.
//!   (ERA5 SST for the Niño 3.4 box — a different product from NOAA ERSSTv5.)
//!
//! ## I/O note
//!   Chunks are full global snapshots (721×1440), so the lat/lon filter does
//!   *not* reduce bytes read — all 744 chunks are still decompressed (~40×
//!   amplification vs the ~24 MB of cells actually used).
//!
//! Run:
//!   cargo run --example nino34_daily_local
//!   RUST_LOG=info cargo run --example nino34_daily_local

mod common;

use std::time::Instant;

const LOCAL_STORE: &str = "data/era5_sst_local.zarr";

// Niño 3.4 box
const LAT_MIN: f64 = -5.0;
const LAT_MAX: f64 = 5.0;
const LON_MIN: f64 = 190.0; // 170°W in 0-360
const LON_MAX: f64 = 240.0; // 120°W in 0-360

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    common::init_tracing();
    let ctx = common::create_remote_context();

    println!("Niño 3.4 Daily-Mean SST — December 2025 (local)");
    println!("===============================================");
    println!("Store  : {LOCAL_STORE}");
    println!("Region : lat [{LAT_MIN}, {LAT_MAX}]  lon [{LON_MIN}, {LON_MAX}]  (Niño 3.4 box)");
    println!("Period : 2025-12-01 00:00:00 → 2025-12-31 23:00:00");
    println!();

    // -----------------------------------------------------------------------
    // Register table from the local Zarr store
    // -----------------------------------------------------------------------
    println!("Registering ERA5 table from local store ...");
    let t0 = Instant::now();
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION '{LOCAL_STORE}'"
    ))
    .await?
    .collect()
    .await?;
    println!("  registered in {:?}\n", t0.elapsed());

    // -----------------------------------------------------------------------
    // Query 1 — Row count (validates filter pushdown scope)
    //
    // Expected: 744 time steps × 41 lat points × 201 lon points = 6,131,304
    // -----------------------------------------------------------------------
    let count_sql = format!(
        r#"
        SELECT COUNT(*) AS row_count
        FROM era5
        WHERE time      BETWEEN '2025-12-01 00:00:00' AND '2025-12-31 23:00:00'
          AND latitude  BETWEEN {LAT_MIN} AND {LAT_MAX}
          AND longitude BETWEEN {LON_MIN} AND {LON_MAX}
        "#
    );

    println!("Query 1 — Row count (expected: 744 × 41 × 201 = 6,131,304)");
    println!("SQL:{count_sql}");
    let t1 = Instant::now();
    ctx.sql(&count_sql).await?.show().await?;
    println!("  completed in {:?}\n", t1.elapsed());

    // -----------------------------------------------------------------------
    // Query 2 — Daily mean SST (the Niño 3.4 index, per day)
    //
    // ERA5 stores SST in Kelvin — subtract 273.15 for Celsius.
    // GROUP BY date_trunc('day', time) → 31 rows for December 2025.
    // -----------------------------------------------------------------------
    // `time` is a DictionaryArray; date/time functions on a dictionary column
    // trip a DataFusion type assertion, so cast it to a plain timestamp first.
    // The bare-column `time BETWEEN ...` in WHERE still drives filter pushdown.
    let day_expr = "date_trunc('day', arrow_cast(time, 'Timestamp(Microsecond, Some(\"UTC\"))'))";
    let daily_sql = format!(
        r#"
        SELECT
            {day_expr}                            AS day,
            AVG(sea_surface_temperature - 273.15) AS sst_celsius,
            COUNT(*)                              AS n
        FROM era5
        WHERE time      BETWEEN '2025-12-01 00:00:00' AND '2025-12-31 23:00:00'
          AND latitude  BETWEEN {LAT_MIN} AND {LAT_MAX}
          AND longitude BETWEEN {LON_MIN} AND {LON_MAX}
        GROUP BY {day_expr}
        ORDER BY day
        "#
    );

    println!("Query 2 — Daily mean SST, Dec 2025  (Niño 3.4 index, per day)");
    println!("SQL:{daily_sql}");
    let t2 = Instant::now();
    ctx.sql(&daily_sql).await?.show_limit(31).await?;
    println!("  completed in {:?}\n", t2.elapsed());

    // -----------------------------------------------------------------------
    // Query 3 — Monthly mean SST  (single Niño 3.4 index value for the month)
    // -----------------------------------------------------------------------
    let monthly_sql = format!(
        r#"
        SELECT
            AVG(sea_surface_temperature - 273.15) AS sst_celsius_dec2025
        FROM era5
        WHERE time      BETWEEN '2025-12-01 00:00:00' AND '2025-12-31 23:00:00'
          AND latitude  BETWEEN {LAT_MIN} AND {LAT_MAX}
          AND longitude BETWEEN {LON_MIN} AND {LON_MAX}
        "#
    );

    println!("Query 3 — Monthly mean SST, Dec 2025  (Niño 3.4 index)");
    println!("SQL:{monthly_sql}");
    let t3 = Instant::now();
    ctx.sql(&monthly_sql).await?.show().await?;
    println!("  completed in {:?}\n", t3.elapsed());

    Ok(())
}
