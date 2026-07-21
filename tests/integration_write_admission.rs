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

// ---------------------------------------------------------------------------
// Materialisation seam: WriteShape + source store -> SkeletonSpec, end to end
// ---------------------------------------------------------------------------

use datafusion::physical_plan::collect;
use zarr_datafusion::writer::{derive_skeleton_spec, write_batches};

async fn plan_of(ctx: &SessionContext, sql: &str) -> Arc<dyn ExecutionPlan> {
    ctx.sql(sql)
        .await
        .expect("plan sql")
        .create_physical_plan()
        .await
        .expect("physical plan")
}

async fn scalar_i64(ctx: &SessionContext, sql: &str) -> i64 {
    let b = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    b[0].column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0)
}

/// The committed round-trip fixture (int64 coords, int64 + float32-with-NaN vars).
const SRC: &str = "data/synthetic_rt_v3.zarr";

#[tokio::test]
async fn materialise_full_copy_spec() {
    let ctx = create_baseline_context();
    register_zarr_table(&ctx, "src", SRC);
    let plan = plan_of(&ctx, "SELECT time, lat, lon, temperature, reflectance FROM src").await;

    let spec = derive_skeleton_spec(&plan, vec![1, 4, 5]).expect("materialise");

    // Grid axes carry the source coordinate values, in dimension order.
    let grid: Vec<(String, usize)> = spec
        .coords
        .iter()
        .map(|c| (c.name.clone(), c.values.len()))
        .collect();
    assert_eq!(
        grid,
        vec![
            ("time".to_string(), 7),
            ("lat".to_string(), 10),
            ("lon".to_string(), 12)
        ]
    );
    let vars: Vec<(String, WriteDataType)> = spec
        .data_vars
        .iter()
        .map(|v| (v.name.clone(), v.data_type))
        .collect();
    assert_eq!(
        vars,
        vec![
            ("temperature".to_string(), WriteDataType::Int64),
            ("reflectance".to_string(), WriteDataType::Float32),
        ]
    );
}

#[tokio::test]
async fn materialise_subset_gathers_coordinates() {
    let ctx = create_baseline_context();
    register_zarr_table(&ctx, "src", SRC);
    // lat is arange(10); `lat < 5` narrows the grid to five points, other axes full.
    let plan = plan_of(&ctx, "SELECT time, lat, lon, temperature FROM src WHERE lat < 5").await;

    let spec = derive_skeleton_spec(&plan, vec![1, 4, 5]).expect("materialise");
    let axis = |n: &str| spec.coords.iter().find(|c| c.name == n).unwrap().values.len();
    assert_eq!(axis("lat"), 5, "lat narrowed to the subset");
    assert_eq!(axis("time"), 7, "time unfiltered");
    assert_eq!(axis("lon"), 12, "lon unfiltered");
}

/// A full `zarr -> (Arrow) -> zarr` copy driven only by a SELECT, asserting the
/// target reads back identical to `src`. Runs for both v2 and v3 sources — the v2
/// path exercises the reader's shape-inference (no `dimension_names`), which is
/// where the round trip could go wrong. (The external `compare_zarr.py` oracle
/// checks the same copy against the original; see examples/write_copy.rs.)
async fn copy_roundtrips(src: &str, tag: &str) {
    let ctx = create_baseline_context();
    register_zarr_table(&ctx, "src", src);
    let plan = plan_of(&ctx, "SELECT time, lat, lon, temperature, reflectance FROM src").await;

    let spec = derive_skeleton_spec(&plan, vec![1, 4, 5]).expect("materialise");
    let target = scratch(tag);
    create_skeleton(&target, &spec).expect("create target");

    let batches = collect(plan, ctx.task_ctx()).await.expect("execute");
    let n = write_batches(&target, &spec, batches).expect("write");
    assert_eq!(n, 7 * 10 * 12);

    // Target must read back identical: same value sum and same NaN structure in the
    // float variable (the fixture has real NaN holes).
    let tgt = create_baseline_context();
    register_zarr_table(&tgt, "t", &target);
    assert_eq!(
        scalar_i64(&ctx, "SELECT SUM(temperature) FROM src").await,
        scalar_i64(&tgt, "SELECT SUM(temperature) FROM t").await,
        "temperature sum differs for {src}",
    );
    let nan = "SELECT SUM(CASE WHEN isnan(reflectance) THEN 1 ELSE 0 END)";
    let src_nan = scalar_i64(&ctx, &format!("{nan} FROM src")).await;
    assert!(src_nan > 0, "fixture should have NaN holes to preserve");
    assert_eq!(
        src_nan,
        scalar_i64(&tgt, &format!("{nan} FROM t")).await,
        "NaN structure differs for {src}",
    );
}

#[tokio::test]
async fn end_to_end_copy_roundtrips_v3() {
    copy_roundtrips("data/synthetic_rt_v3.zarr", "mat_copy_v3.zarr").await;
}

#[tokio::test]
async fn end_to_end_copy_roundtrips_v2() {
    // v2 has no dimension_names — the reader infers axes from shape, and the copy
    // upgrades it to a v3 target with dimension_names.
    copy_roundtrips("data/synthetic_rt_v2.zarr", "mat_copy_v2.zarr").await;
}

// ---------------------------------------------------------------------------
// DataSink driver: writing a Zarr store as a DataFusion ExecutionPlan
// ---------------------------------------------------------------------------

use zarr_datafusion::writer::zarr_write_exec;

#[tokio::test]
async fn data_sink_writes_a_store_as_an_execution_plan() {
    let ctx = create_baseline_context();
    register_zarr_table(&ctx, "src", SRC);
    let plan = plan_of(&ctx, "SELECT time, lat, lon, temperature, reflectance FROM src").await;

    // Derive the spec at plan time (not from the stream), then wrap the input in a
    // sink node. Executing the node creates the skeleton and writes the store.
    let spec = derive_skeleton_spec(&plan, vec![1, 4, 5]).expect("materialise");
    let target = scratch("datasink_copy.zarr");
    let exec = zarr_write_exec(plan, target.clone(), spec, 3); // 3 write partitions

    let out = collect(exec, ctx.task_ctx()).await.expect("execute sink");
    // The sink yields a single `count` row (UInt64) of rows written.
    let count = out[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .expect("count column")
        .value(0);
    assert_eq!(count, 7 * 10 * 12);

    // The written store must match the source.
    let tgt = create_baseline_context();
    register_zarr_table(&tgt, "t", &target);
    assert_eq!(
        scalar_i64(&ctx, "SELECT SUM(temperature) FROM src").await,
        scalar_i64(&tgt, "SELECT SUM(temperature) FROM t").await,
    );
    let nan = "SELECT SUM(CASE WHEN isnan(reflectance) THEN 1 ELSE 0 END)";
    assert_eq!(
        scalar_i64(&ctx, &format!("{nan} FROM src")).await,
        scalar_i64(&tgt, &format!("{nan} FROM t")).await,
    );
}
