//! Integration tests for the Zarr write path (Phase 1: skeleton creation).
//!
//! See docs/zarr-write-roundtrip-plan.md. Phase 1's exit criterion is that a
//! skeleton opens with correct dims/coords/shape and reads all-fill_value. Here
//! we assert that through *our own* reader; the xarray half of the criterion is
//! checked by `scripts/check_skeleton.py`, which is what proves the store is
//! legible to an outside consumer rather than only to us.

mod common;

use std::path::PathBuf;

use common::{create_test_context, execute_query, register_zarr_table};
use zarr_datafusion::writer::{
    create_skeleton, CoordSpec, CoordValues, DataVarSpec, SkeletonSpec, WriteDataType,
};

/// Mirrors the round-trip fixture's grid: time(7) x lat(10) x lon(12), ragged
/// chunks. Distinct axis lengths are what make an axis swap detectable.
fn rt_spec() -> SkeletonSpec {
    let coords = vec![
        CoordSpec::new("time", CoordValues::Int64((0..7).collect())),
        CoordSpec::new("lat", CoordValues::Int64((0..10).collect())),
        CoordSpec::new("lon", CoordValues::Int64((0..12).collect())),
    ];
    let data_vars = vec![
        DataVarSpec::new("temperature", WriteDataType::Int64),
        DataVarSpec::new("reflectance", WriteDataType::Float32),
    ];
    SkeletonSpec::new(coords, data_vars, vec![1, 4, 5])
}

/// A scratch store path, removed if a previous run left one behind.
fn scratch(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    p.push(name);
    let _ = std::fs::remove_dir_all(&p);
    p.to_string_lossy().into_owned()
}

#[test]
fn skeleton_creates_readable_store() {
    let path = scratch("skeleton_readable.zarr");
    create_skeleton(&path, &rt_spec()).expect("create skeleton");

    let ctx = create_test_context();
    let (schema, meta) = register_zarr_table(&ctx, "sk", &path);

    // Coordinates come back in *dimension* order, not alphabetical: the reader
    // honours the v3 `dimension_names` the skeleton wrote, so it never reaches
    // the alphabetical fallback (schema_inference.rs). This is the writer and
    // reader agreeing on which axis is which — the property the whole round-trip
    // plan is built to protect.
    let coord_names: Vec<&str> = meta.coords.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(coord_names, vec!["time", "lat", "lon"]);

    // The cartesian product of the grid we asked for.
    assert_eq!(meta.total_rows, 7 * 10 * 12);

    let var_names: Vec<&str> = meta.data_vars.iter().map(|v| v.name.as_str()).collect();
    assert!(var_names.contains(&"temperature"));
    assert!(var_names.contains(&"reflectance"));

    for field in ["time", "lat", "lon", "temperature", "reflectance"] {
        assert!(
            schema.field_with_name(field).is_ok(),
            "schema should expose '{field}'"
        );
    }
}

#[tokio::test]
async fn skeleton_data_vars_read_as_fill_value() {
    let path = scratch("skeleton_fill.zarr");
    create_skeleton(&path, &rt_spec()).expect("create skeleton");

    let ctx = create_test_context();
    register_zarr_table(&ctx, "sk", &path);

    // No chunks were written, so every data cell must come back as fill_value:
    // NaN for the float variable, 0 for the int one. This is the "allocated but
    // empty" property Phase 2 will rely on to leave holes in filtered writes.
    let batches = execute_query(
        &ctx,
        "SELECT COUNT(*) AS n, \
                COUNT(reflectance) FILTER (WHERE NOT isnan(reflectance)) AS non_nan, \
                SUM(ABS(temperature)) AS abs_sum \
         FROM sk",
    )
    .await;

    let batch = &batches[0];
    let n = batch.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
    let non_nan = batch.column(1).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
    let abs_sum = batch.column(2).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();

    assert_eq!(n.value(0), 7 * 10 * 12, "full cartesian product of the grid");
    assert_eq!(non_nan.value(0), 0, "float var must be entirely NaN fill");
    assert_eq!(abs_sum.value(0), 0, "int var must be entirely 0 fill");
}

#[tokio::test]
async fn skeleton_coords_hold_their_values() {
    let path = scratch("skeleton_coords.zarr");
    create_skeleton(&path, &rt_spec()).expect("create skeleton");

    let ctx = create_test_context();
    register_zarr_table(&ctx, "sk", &path);

    // Coordinate arrays are written in full, so they are real data, not fill.
    // Distinct maxima per axis also catch a lat/lon transposition.
    let batches = execute_query(
        &ctx,
        "SELECT MAX(time) AS t, MAX(lat) AS y, MAX(lon) AS x FROM sk",
    )
    .await;
    let batch = &batches[0];
    let col = |i: usize| {
        batch
            .column(i)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0)
    };
    assert_eq!((col(0), col(1), col(2)), (6, 9, 11));
}

#[test]
fn skeleton_rejects_specs_that_would_be_unreadable() {
    // A chunk shape must cover every dimension.
    let mut spec = rt_spec();
    spec.chunks = vec![1, 4];
    let err = create_skeleton(&scratch("bad_chunks.zarr"), &spec).unwrap_err();
    assert!(err.to_string().contains("chunk shape is required"), "{err}");

    // A data variable named like a coordinate would be read back as a 1-D
    // coordinate, silently losing the variable.
    let mut spec = rt_spec();
    spec.data_vars.push(DataVarSpec::new("lat", WriteDataType::Float32));
    let err = create_skeleton(&scratch("bad_collide.zarr"), &spec).unwrap_err();
    assert!(err.to_string().contains("collides with a coordinate"), "{err}");
}
