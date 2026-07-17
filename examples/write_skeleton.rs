//! Write a Zarr skeleton (Phase 1 of docs/zarr-write-roundtrip-plan.md).
//!
//! Emits the round-trip fixture's grid — time(7) x lat(10) x lon(12), ragged
//! chunks (1,4,5) — with coordinate arrays populated and no data chunks, so
//! every data variable reads back as fill_value.
//!
//! Pair with the outside-consumer check:
//!
//! ```bash
//! cargo run --example write_skeleton
//! uv run --with 'zarr>=3' --with xarray --with numpy \
//!   scripts/check_skeleton.py data/skeleton_demo.zarr
//! ```

use serde_json::json;
use zarr_datafusion::writer::{
    create_skeleton, CoordSpec, CoordValues, DataVarSpec, SkeletonSpec, WriteDataType,
};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/skeleton_demo.zarr".to_string());
    let _ = std::fs::remove_dir_all(&path);

    let coords = vec![
        CoordSpec::new("time", CoordValues::Int64((0..7).collect())),
        CoordSpec::new("lat", CoordValues::Int64((0..10).collect())),
        CoordSpec::new("lon", CoordValues::Int64((0..12).collect())),
    ];

    let mut temperature = DataVarSpec::new("temperature", WriteDataType::Int64);
    temperature.attributes.insert("units".into(), json!("K"));
    temperature
        .attributes
        .insert("long_name".into(), json!("Air Temperature"));

    let mut reflectance = DataVarSpec::new("reflectance", WriteDataType::Float32);
    reflectance.attributes.insert("units".into(), json!("1"));
    reflectance
        .attributes
        .insert("long_name".into(), json!("Surface Reflectance"));

    let mut spec = SkeletonSpec::new(coords, vec![temperature, reflectance], vec![1, 4, 5]);
    spec.attributes
        .insert("title".into(), json!("Write Skeleton Demo"));

    create_skeleton(&path, &spec)?;

    println!("Skeleton written: {path}");
    println!("  grid   : time(7) x lat(10) x lon(12), chunks (1,4,5)");
    println!("  coords : populated");
    println!("  vars   : temperature (int64, fill 0), reflectance (float32, fill NaN)");
    println!("\nVerify with an outside consumer:");
    println!(
        "  uv run --with 'zarr>=3' --with xarray --with numpy \\\n    scripts/check_skeleton.py {path}"
    );
    Ok(())
}
