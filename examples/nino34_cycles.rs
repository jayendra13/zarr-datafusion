//! Example: Niño 3.4 December SST cycle — all ERA5 years (1940–2025)
//!
//! Fetches a mid-month December SST snapshot (Dec 15 12:00:00 UTC) for every
//! year in the ERA5 record and prints the full ENSO cycle.
//!
//! ## Why UNION ALL and not IN or date_part()
//!
//! `time IN (...)` — DataFusion translates to OR expressions; ZarrExec's filter
//! parser skips OR, so all 1.3M chunks would be scanned.
//!
//! `WHERE date_part('month', time) = 12` — function on a coordinate column;
//! not pushable to ZarrExec, causes a full scan.
//!
//! UNION ALL — each sub-query gets its own ZarrExec with `time = X` equality
//! pushdown: exactly 1 chunk fetched per year.
//!
//! ## I/O profile
//!   Years        : 1940–2025  (86 years)
//!   Chunks/year  : 1  (Dec 15 12:00:00 UTC — one hourly snapshot)
//!   Total chunks : 86
//!   Data/chunk   : ~1.28 MB compressed  (full global SST field)
//!   Total GCS    : ~110 MB
//!   Est. time    : ~51 s  (sequential, ~592 ms/chunk)
//!
//! Run:
//!   cargo run --example nino34_cycles
//!   RUST_LOG=info cargo run --example nino34_cycles

mod common;

use std::time::Instant;

const GCS_STORE: &str = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3";

const LOCAL_STORE: &str = "data/era5_sst_local.zarr";

const LAT_MIN: f64 = -5.0;
const LAT_MAX: f64 = 5.0;
const LON_MIN: f64 = 190.0;
const LON_MAX: f64 = 240.0;

const FIRST_YEAR: u32 = 1940;
const LAST_YEAR: u32 = 2025;

fn build_cycle_sql() -> String {
    let mut parts: Vec<String> = Vec::new();

    for year in FIRST_YEAR..=LAST_YEAR {
        parts.push(format!(
            r#"SELECT
    {year}                                       AS year,
    AVG(sea_surface_temperature - 273.15)        AS sst_celsius
FROM era5
WHERE time      = '{year}-12-15 12:00:00'
  AND latitude  BETWEEN {LAT_MIN} AND {LAT_MAX}
  AND longitude BETWEEN {LON_MIN} AND {LON_MAX}"#
        ));
    }

    format!("{}\nORDER BY year", parts.join("\nUNION ALL\n"))
}

fn enso_phase(sst: f64, dec_climatology: f64) -> &'static str {
    let anom = sst - dec_climatology;
    if anom >= 0.5 {
        "El Niño ▲"
    } else if anom <= -0.5 {
        "La Niña ▼"
    } else {
        "Neutral"
    }
}

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    common::init_tracing();
    let ctx = common::create_remote_context();

    let store = LOCAL_STORE;

    println!("Niño 3.4 December SST Cycle — ERA5 1940–2025");
    println!("==============================================");
    println!("Store    : {store}");
    println!("Snapshot : December 15  12:00:00 UTC  (mid-month)");
    println!("Region   : lat [{LAT_MIN}, {LAT_MAX}]  lon [{LON_MIN}, {LON_MAX}]");
    println!(
        "I/O      : {} chunks × ~1.28 MB = ~{:.0} MB (local disk)",
        LAST_YEAR - FIRST_YEAR + 1,
        (LAST_YEAR - FIRST_YEAR + 1) as f64 * 1.28
    );
    println!();

    println!("Registering ERA5 table ...");
    let t0 = Instant::now();
    ctx.sql(&format!(
        "CREATE EXTERNAL TABLE era5 STORED AS ZARR LOCATION '{store}'"
    ))
    .await?
    .collect()
    .await?;
    println!("  registered in {:?}\n", t0.elapsed());

    let sql = build_cycle_sql();

    println!(
        "Running UNION ALL query ({} sub-queries) ...",
        LAST_YEAR - FIRST_YEAR + 1
    );
    println!("Each sub-query: time = 'YYYY-12-15 12:00:00' → 1 chunk fetched\n");

    let t1 = Instant::now();
    let df = ctx.sql(&sql).await?;
    let batches = df.collect().await?;
    let elapsed = t1.elapsed();

    // ERA5 Dec 15 12:00 snapshot climatology (1991-2020), computed from this dataset: 26.559°C
    // NOAA ERSSTv5 December climatology (1991-2020): 26.645°C  — difference: -0.086°C
    let era5_dec_climatology = 26.559_f64;

    println!(
        "{:<6} {:>12}  {:>10}  Phase",
        "Year", "SST (°C)", "Anom (°C)"
    );
    println!("{}", "-".repeat(50));

    for batch in &batches {
        use arrow::array::{Float64Array, Int64Array};

        let years = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("year column should be Int64");
        let ssts = batch
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("sst_celsius column should be Float64");

        for i in 0..batch.num_rows() {
            let year = years.value(i) as i32;
            let sst = ssts.value(i);
            let anom = sst - era5_dec_climatology;
            let phase = enso_phase(sst, era5_dec_climatology);

            // Bar chart: scale ±3°C → 20 chars
            let bar_len = ((anom.abs() / 3.0) * 20.0).min(20.0) as usize;
            let bar = if anom >= 0.0 {
                format!("+{}", "█".repeat(bar_len))
            } else {
                format!("-{}", "█".repeat(bar_len))
            };

            println!(
                "{year:<6} {:>10.3}°C  {:>+8.3}°C  {phase:<12} {bar}",
                sst, anom
            );
        }
    }

    println!("{}", "-".repeat(50));
    println!(
        "\nCompleted {} years in {:?}",
        LAST_YEAR - FIRST_YEAR + 1,
        elapsed
    );
    println!(
        "Throughput: {:.1} chunks/s  ({:.0} ms/chunk)",
        (LAST_YEAR - FIRST_YEAR + 1) as f64 / elapsed.as_secs_f64(),
        elapsed.as_millis() as f64 / (LAST_YEAR - FIRST_YEAR + 1) as f64
    );
    println!();
    println!("ERA5 Dec climatology used : {era5_dec_climatology}°C  (Dec 15 12:00 snapshot mean, 1991-2020)");
    println!(
        "NOAA ERSSTv5 climatology  : 26.645°C  (monthly mean, 1991-2020) — ERA5 bias: -0.086°C"
    );
    println!("NOAA ONI threshold        : ±0.5°C anomaly for 5+ consecutive overlapping seasons");
    println!(
        "Note: ERA5 anomalies are larger than NOAA ONI (~+0.4–0.8°C El Niño, ~-0.2–0.5°C La Niña)"
    );
    println!(
        "      due to ERA5 diurnal cycle + higher resolution vs ERSSTv5 monthly in-situ analysis"
    );

    Ok(())
}
