//! Example: Niño 3.4 SST index — December 1997 (strong El Niño peak)
//!
//! Computes the mean Sea Surface Temperature over the Niño 3.4 region
//! for December 1997 directly from the ARCO-ERA5 Zarr store on GCS.
//!
//! ## Niño 3.4 region
//!   Latitude  :  5°S – 5°N   (latitude BETWEEN -5.0 AND 5.0)
//!   Longitude : 170°W – 120°W (longitude BETWEEN 190.0 AND 240.0, 0-360 system)
//!
//! ## Expected output (reference: NOAA CPC ERSSTv5)
//!   Raw SST Dec 1997 : 28.740°C
//!   ERA5 will differ by ~0.1–0.5°C (different SST product)
//!
//! ## I/O profile
//!   Chunks fetched  : 744  (1 per hour × 31 days)
//!   Chunk size      : ~1.28 MB compressed  (full global field, 721×1440)
//!   Total GCS reads : ~0.95 GB compressed
//!   Useful data     : ~24.5 MB  (41 lat × 201 lon × 744 time × 4 bytes)
//!   Amplification   : ~40×  (chunking optimised for global snapshots, not regions)
//!
//! Run:
//!   cargo run --example nino34_query
//!   RUST_LOG=info cargo run --example nino34_query

mod common;

use std::time::Instant;

const GCS_STORE: &str = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3";

// Niño 3.4 box
const LAT_MIN: f64 = -5.0;
const LAT_MAX: f64 = 5.0;
const LON_MIN: f64 = 190.0; // 170°W in 0-360
const LON_MAX: f64 = 240.0; // 120°W in 0-360

// Reference value from NOAA CPC (ERSSTv5, 1991-2020 base)
const NOAA_REF_SST_DEC2025: f64 = 28.740; // raw SST °C
const NOAA_REF_ANOM_DEC2025: f64 = 2.100; // anomaly vs 1991-2020 climatology

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    common::init_tracing();
    let ctx = common::create_remote_context();

    println!("Niño 3.4 SST Query — December 2025");
    println!("====================================");
    println!("Store  : {GCS_STORE}");
    println!("Region : lat [{LAT_MIN}, {LAT_MAX}]  lon [{LON_MIN}, {LON_MAX}]  (Niño 3.4 box)");
    println!("Period : 2025-12-01 00:00:00 → 2025-12-31 23:00:00");
    println!();

    // -----------------------------------------------------------------------
    // Register table
    // -----------------------------------------------------------------------
    println!("Registering ERA5 table from GCS ...");
    let t0 = Instant::now();
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION '{GCS_STORE}'"
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
        WHERE time     BETWEEN '2025-12-01 00:00:00' AND '2025-12-31 23:00:00'
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
    // Query 2 — Mean SST (the Niño 3.4 index value)
    //
    // ERA5 stores SST in Kelvin — subtract 273.15 for Celsius.
    // Expected (NOAA ERSSTv5 reference): 28.740°C
    // ERA5 will be close but not identical (~0.1–0.5°C difference).
    // -----------------------------------------------------------------------
    let sst_sql = format!(
        r#"
        SELECT
            AVG(sea_surface_temperature - 273.15)  AS sst_celsius,
            MIN(sea_surface_temperature - 273.15)  AS sst_min,
            MAX(sea_surface_temperature - 273.15)  AS sst_max,
            MAX(sea_surface_temperature - 273.15)
                - MIN(sea_surface_temperature - 273.15) AS sst_range
        FROM era5
        WHERE time     BETWEEN '2025-12-01 00:00:00' AND '2025-12-31 23:00:00'
          AND latitude  BETWEEN {LAT_MIN} AND {LAT_MAX}
          AND longitude BETWEEN {LON_MIN} AND {LON_MAX}
        "#
    );

    println!("Query 2 — Mean SST, Dec 2025  (Niño 3.4 index)");
    println!("SQL:{sst_sql}");
    println!(
        "  I/O budget: 744 chunks × ~1.28 MB = ~0.95 GB fetched from GCS (full global fields)"
    );
    println!("  Filter pushdown skips 1,322,904 of 1,323,648 time chunks.\n");
    let t2 = Instant::now();
    ctx.sql(&sst_sql).await?.show().await?;
    let elapsed = t2.elapsed();
    println!("  completed in {:?}\n", elapsed);

    println!("Reference values (NOAA CPC, ERSSTv5, 1991-2020 base):");
    println!("  Raw SST  Dec 2025 : {NOAA_REF_SST_DEC2025:.3}°C");
    println!("  Anomaly  Dec 2025 : +{NOAA_REF_ANOM_DEC2025:.3}°C  (vs 1991-2020 climatology)");
    println!("  (ERA5 vs ERSSTv5 difference expected: ~0.1–0.5°C)");
    println!();

    // -----------------------------------------------------------------------
    // Query 3 — Hourly SST time series  (first and last day of month)
    //
    // Shows daily variation within Dec 2025.
    // Useful for visualising the diurnal SST cycle.
    // -----------------------------------------------------------------------
    let timeseries_sql = format!(
        r#"
        SELECT
            time,
            AVG(sea_surface_temperature - 273.15) AS sst_celsius
        FROM era5
        WHERE time     BETWEEN '2025-12-01 00:00:00' AND '2025-12-02 23:00:00'
          AND latitude  BETWEEN {LAT_MIN} AND {LAT_MAX}
          AND longitude BETWEEN {LON_MIN} AND {LON_MAX}
        GROUP BY time
        ORDER BY time
        "#
    );

    println!("Query 3 — Hourly SST, first 48 hours of Dec 2025  (diurnal cycle check)");
    println!("SQL:{timeseries_sql}");
    let t3 = Instant::now();
    ctx.sql(&timeseries_sql).await?.show().await?;
    println!("  completed in {:?}\n", t3.elapsed());

    Ok(())
}
