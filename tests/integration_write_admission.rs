//! Phase 3 of the Zarr write path (docs/zarr-write-roundtrip-plan.md): admission +
//! grid derivation from the physical plan feeding the sink.
//!
//! `derive_write_shape` is a fallible constructor — a query is admissible because
//! we can derive its target grid, and a query that cannot be lowered is exactly the
//! one rejected. These tests exercise it over *real* physical plans built from SQL,
//! using the **baseline** context: with the cardinality rule active, a `GROUP BY`
//! would be rewritten into a `ZarrAggregateExec` before we see it, whereas admission
//! reasons about the plain `AggregateExec <- ZarrExec` shape.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

use common::{create_baseline_context, register_zarr_table};
use zarr_datafusion::writer::{
    create_skeleton, derive_write_shape, CoordSpec, CoordValues, DataVarSpec, RejectReason,
    SkeletonSpec, WriteDataType, WriteShape,
};

fn rt_spec() -> SkeletonSpec {
    SkeletonSpec::new(
        vec![
            CoordSpec::new("time", CoordValues::Int64((0..7).collect())),
            CoordSpec::new("lat", CoordValues::Int64((0..10).collect())),
            CoordSpec::new("lon", CoordValues::Int64((0..12).collect())),
        ],
        vec![
            DataVarSpec::new("temperature", WriteDataType::Int64),
            DataVarSpec::new("reflectance", WriteDataType::Float32),
        ],
        vec![1, 4, 5],
    )
}

fn scratch(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    p.push(name);
    let _ = std::fs::remove_dir_all(&p);
    p.to_string_lossy().into_owned()
}

/// Register a skeleton store and return a context that can plan over it.
fn ctx_over_skeleton(name: &str) -> SessionContext {
    let path = scratch(name);
    create_skeleton(&path, &rt_spec()).unwrap();
    let ctx = create_baseline_context();
    register_zarr_table(&ctx, "t", &path);
    ctx
}

async fn shape_of(ctx: &SessionContext, sql: &str) -> Result<WriteShape, RejectReason> {
    let plan: Arc<dyn ExecutionPlan> = ctx
        .sql(sql)
        .await
        .expect("plan sql")
        .create_physical_plan()
        .await
        .expect("physical plan");
    derive_write_shape(&plan)
}

// --- admits ---------------------------------------------------------------

#[tokio::test]
async fn admits_full_projection() {
    let ctx = ctx_over_skeleton("adm_full.zarr");
    let shape = shape_of(&ctx, "SELECT time, lat, lon, temperature, reflectance FROM t")
        .await
        .expect("admissible");
    assert_eq!(shape.grid_axes, vec!["time", "lat", "lon"]);
    assert!(!shape.is_reduce);
    let vars: Vec<(&str, WriteDataType)> = shape
        .data_vars
        .iter()
        .map(|v| (v.name.as_str(), v.data_type))
        .collect();
    assert!(vars.contains(&("temperature", WriteDataType::Int64)));
    assert!(vars.contains(&("reflectance", WriteDataType::Float32)));
}

#[tokio::test]
async fn admits_transform_projection() {
    // A derived variable is a new data var; the grid is unchanged (§5.7).
    let ctx = ctx_over_skeleton("adm_transform.zarr");
    let shape = shape_of(
        &ctx,
        "SELECT time, lat, lon, temperature * 2 AS t2 FROM t",
    )
    .await
    .expect("admissible");
    assert_eq!(shape.grid_axes, vec!["time", "lat", "lon"]);
    assert!(!shape.is_reduce);
    assert_eq!(shape.data_vars.len(), 1);
    assert_eq!(shape.data_vars[0].name, "t2");
    assert_eq!(shape.data_vars[0].data_type, WriteDataType::Int64);
}

#[tokio::test]
async fn admits_reduce_grouped_on_coordinates() {
    // GROUP BY lat, lon reduces the time axis away; the target grid is the group
    // keys, the variable is the aggregate output (§5.2 — a climatology).
    let ctx = ctx_over_skeleton("adm_reduce.zarr");
    let shape = shape_of(
        &ctx,
        "SELECT lat, lon, AVG(temperature) AS tmean FROM t GROUP BY lat, lon",
    )
    .await
    .expect("admissible");
    assert!(shape.is_reduce);
    assert_eq!(shape.grid_axes, vec!["lat", "lon"]); // time reduced away, dim order
    assert_eq!(shape.data_vars.len(), 1);
    assert_eq!(shape.data_vars[0].name, "tmean");
}

// --- rejects --------------------------------------------------------------

#[tokio::test]
async fn rejects_dropped_axis() {
    // Dropping `time` without a GROUP BY collapses 7 rows onto each (lat,lon) cell.
    let ctx = ctx_over_skeleton("rej_drop.zarr");
    let err = shape_of(&ctx, "SELECT lat, lon, temperature FROM t")
        .await
        .unwrap_err();
    assert_eq!(err, RejectReason::DroppedAxis("time".to_string()));
}

#[tokio::test]
async fn rejects_scalar_aggregate() {
    // A global aggregate is a scalar, not a grid.
    let ctx = ctx_over_skeleton("rej_scalar.zarr");
    let err = shape_of(&ctx, "SELECT AVG(temperature) AS m FROM t")
        .await
        .unwrap_err();
    assert_eq!(err, RejectReason::ScalarAggregate);
}

#[tokio::test]
async fn rejects_group_by_data_variable() {
    // Grouping by a data variable is not a coordinate reduce; recognize() declines.
    let ctx = ctx_over_skeleton("rej_gbdata.zarr");
    let err = shape_of(
        &ctx,
        "SELECT temperature, AVG(reflectance) AS m FROM t GROUP BY temperature",
    )
    .await
    .unwrap_err();
    assert_eq!(err, RejectReason::UnpushableAggregate);
}

#[tokio::test]
async fn rejects_computed_aggregate_argument() {
    // AVG(temperature - 5) cannot be pushed (the arithmetic would be dropped, the
    // exact bug class of issue #24 / commit e7c871b), so it is not an admissible
    // reduce shape.
    let ctx = ctx_over_skeleton("rej_computedagg.zarr");
    let err = shape_of(
        &ctx,
        "SELECT lat, AVG(temperature - 5) AS m FROM t GROUP BY lat",
    )
    .await
    .unwrap_err();
    assert_eq!(err, RejectReason::UnpushableAggregate);
}

#[tokio::test]
async fn rejects_coordinates_only() {
    // No data variable to write.
    let ctx = ctx_over_skeleton("rej_coordsonly.zarr");
    let err = shape_of(&ctx, "SELECT time, lat, lon FROM t")
        .await
        .unwrap_err();
    assert_eq!(err, RejectReason::NoDataVariables);
}
