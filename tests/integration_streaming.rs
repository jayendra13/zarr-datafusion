//! Integration tests for the streaming (windowed) Zarr scan — Block 0 Phase 2.
//!
//! The dominant invariant: a scan run with a small `batch_size` emits the SAME
//! rows as the un-windowed reference, just split across multiple `RecordBatch`es.
//! Tests execute the `ZarrExec` node directly with a custom-`batch_size`
//! `TaskContext` so they observe the scan's true batch boundaries (DataFusion's
//! `CoalesceBatchesExec` would otherwise re-batch the output).

mod common;

use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use futures::TryStreamExt;

use common::*;
use zarr_datafusion::optimizer::ZarrLimitPushdownRule;

// Synthetic store layout: coords time(7), lat(10), lon(10), so the outer
// (most-significant) axis is `time`, inner_rows = lat*lon = 100, total = 700.
// The scan also splits into several partitions along the same (time) axis;
// windowing subdivides each partition further, so the batch *count* depends on
// the partition layout — tests assert on rows/contents, not raw batch counts.
const INNER_ROWS: usize = 100;
const TOTAL_ROWS: usize = 700;

/// SessionContext whose scans emit at most `batch_size` rows per window.
fn ctx_with_batch_size(batch_size: usize) -> SessionContext {
    let cfg = SessionConfig::new().with_batch_size(batch_size);
    let state = SessionStateBuilder::new()
        .with_config(cfg)
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(ZarrLimitPushdownRule::new()))
        .build();
    SessionContext::new_with_state(state)
}

/// Execute the `ZarrExec` leaf of `sql` directly (all partitions) and return the
/// raw batches it emits — i.e. exactly what the streaming scan produced.
async fn zarr_exec_batches(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    let plan = get_physical_plan(ctx, sql).await;
    let zarr_exec = find_zarr_exec(&plan).expect("plan should contain a ZarrExec");
    let n = zarr_exec
        .properties()
        .output_partitioning()
        .partition_count();
    let task_ctx = ctx.task_ctx();
    let mut all = Vec::new();
    for p in 0..n {
        let stream = zarr_exec
            .execute(p, task_ctx.clone())
            .expect("ZarrExec::execute");
        let batches: Vec<RecordBatch> = stream.try_collect().await.expect("collect stream");
        all.extend(batches);
    }
    all
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Logical contents of a batch sequence, independent of how it was batched or how
/// coordinate dictionaries happen to be encoded per window.
fn rendered(batches: &[RecordBatch]) -> String {
    arrow::util::pretty::pretty_format_batches(batches)
        .expect("pretty format")
        .to_string()
}

/// Number of partitions the scan for `sql` exposes (each is an independent stream).
async fn partition_count(ctx: &SessionContext, sql: &str) -> usize {
    let plan = get_physical_plan(ctx, sql).await;
    find_zarr_exec(&plan)
        .expect("ZarrExec")
        .properties()
        .output_partitioning()
        .partition_count()
}

/// Reference = the same query read in one shot (huge batch_size => one window per
/// partition).
async fn reference_batches(path: &str, sql: &str) -> Vec<RecordBatch> {
    let ctx = ctx_with_batch_size(10_000_000);
    register_zarr_table(&ctx, "data", path);
    zarr_exec_batches(&ctx, sql).await
}

/// Core transparency assertion: streamed result == reference result, and the
/// stream actually split into more than one batch.
async fn assert_streams_transparently(
    path: &str,
    sql: &str,
    batch_size: usize,
) -> Vec<RecordBatch> {
    let ctx = ctx_with_batch_size(batch_size);
    register_zarr_table(&ctx, "data", path);
    let streamed = zarr_exec_batches(&ctx, sql).await;
    let reference = reference_batches(path, sql).await;

    assert!(
        streamed.len() > 1,
        "expected multiple batches for `{sql}` at batch_size={batch_size}, got {}",
        streamed.len()
    );
    assert_eq!(
        rendered(&streamed),
        rendered(&reference),
        "streamed rows must equal reference for `{sql}`"
    );
    streamed
}

#[tokio::test]
async fn streaming_select_star_is_transparent() {
    // 140 rows/window, inner_rows=100 => 1 time step per window (100 rows).
    let batches = assert_streams_transparently(SYNTHETIC_V3, "SELECT * FROM data", 140).await;
    assert_eq!(total_rows(&batches), TOTAL_ROWS);
    for b in &batches {
        // One time step is the smallest divisible unit here (100 rows <= 140).
        assert!(
            b.num_rows() <= 140,
            "batch exceeded batch_size: {}",
            b.num_rows()
        );
    }
}

#[tokio::test]
async fn streaming_projection_is_transparent() {
    let batches =
        assert_streams_transparently(SYNTHETIC_V3, "SELECT temperature FROM data", 140).await;
    assert_eq!(total_rows(&batches), TOTAL_ROWS);
    // Only the projected column is present.
    assert_eq!(batches[0].num_columns(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "temperature");
}

#[tokio::test]
async fn streaming_indivisible_plane_still_streams() {
    // batch_size (32) < inner_rows (100): each window is a single outer (time) step
    // = one lat*lon plane = 100 rows, so every batch necessarily exceeds batch_size
    // — the Block 0 granularity bound. Still streams, still transparent.
    let batches = assert_streams_transparently(SYNTHETIC_V3, "SELECT * FROM data", 32).await;
    assert_eq!(total_rows(&batches), TOTAL_ROWS);
    // Each plane (100 rows) exceeds batch_size (32): the indivisible-plane bound.
    for b in &batches {
        assert_eq!(b.num_rows(), INNER_ROWS, "one indivisible plane per window");
    }
}

#[tokio::test]
async fn streaming_with_range_filter_is_transparent() {
    // A filter that still leaves more than one window's worth of rows.
    assert_streams_transparently(SYNTHETIC_V3, "SELECT * FROM data WHERE lat > 2.0", 140).await;
}

#[tokio::test]
async fn streaming_coordinate_only_is_transparent() {
    // Coordinate-only projection windows along the coord axis too.
    assert_streams_transparently(SYNTHETIC_V3, "SELECT lat FROM data", 4).await;
}

#[tokio::test]
async fn single_batch_per_partition_when_batch_size_large() {
    // batch_size >= per-partition rows => no windowing: exactly one batch per
    // partition (degenerate = old behavior).
    let ctx = ctx_with_batch_size(10_000_000);
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);
    let sql = "SELECT * FROM data";
    let pc = partition_count(&ctx, sql).await;
    let batches = zarr_exec_batches(&ctx, sql).await;
    assert_eq!(
        batches.len(),
        pc,
        "large batch_size => one batch per partition"
    );
    assert_eq!(total_rows(&batches), TOTAL_ROWS);
}

#[tokio::test]
async fn streaming_limit_stops_mid_window() {
    // LIMIT 100 with 140-row windows: first window is sliced to 100 and the scan
    // stops — exactly 100 rows, matching the reference.
    let ctx = ctx_with_batch_size(140);
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);
    let sql = "SELECT temperature FROM data LIMIT 100";
    let streamed = zarr_exec_batches(&ctx, sql).await;
    let reference = reference_batches(SYNTHETIC_V3, sql).await;

    assert_eq!(total_rows(&streamed), 100, "LIMIT must cap total rows");
    assert_eq!(rendered(&streamed), rendered(&reference));
}

#[tokio::test]
async fn streaming_empty_result_yields_empty_batches_with_schema() {
    // A filter that matches nothing: defined behavior is each partition stream
    // emitting one empty batch carrying the projected schema (no rows, no panics).
    let ctx = ctx_with_batch_size(140);
    let (schema, _) = register_zarr_table(&ctx, "data", SYNTHETIC_V3);
    let batches = zarr_exec_batches(&ctx, "SELECT * FROM data WHERE lat = 999.0").await;

    assert_eq!(total_rows(&batches), 0, "no rows should match");
    assert!(!batches.is_empty(), "expected at least one (empty) batch");
    for b in &batches {
        assert_eq!(b.num_rows(), 0);
        assert_eq!(b.num_columns(), schema.fields().len());
    }
}

#[tokio::test]
async fn streaming_v2_v3_parity() {
    // v2 and v3 stream identically (to each other and to their references).
    let v2 = {
        let ctx = ctx_with_batch_size(140);
        register_zarr_table(&ctx, "data", SYNTHETIC_V2);
        zarr_exec_batches(&ctx, "SELECT * FROM data").await
    };
    let v3 = {
        let ctx = ctx_with_batch_size(140);
        register_zarr_table(&ctx, "data", SYNTHETIC_V3);
        zarr_exec_batches(&ctx, "SELECT * FROM data").await
    };
    assert!(v2.len() > 1 && v3.len() > 1, "both should stream");
    assert_eq!(rendered(&v2), rendered(&v3), "v2 and v3 must agree");
}

#[tokio::test]
async fn streaming_preserves_optimizer_shortcircuit() {
    // The MIN/MAX/COUNT rules fold to constants from statistics and bypass the
    // scan entirely; the streaming rework must not change that.
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);
    for sql in [
        "SELECT COUNT(*) FROM data",
        "SELECT MIN(lat) FROM data",
        "SELECT MAX(lon) FROM data",
    ] {
        let plan = get_physical_plan(&ctx, sql).await;
        assert_no_zarr_exec(&plan);
    }
}

// The ships-value proof: on a large cube a full scan must NOT build one giant
// batch. Ignored because it reads the whole ERA5 cube (~6M rows); run manually:
//   cargo test --test integration_streaming -- --ignored --nocapture
#[tokio::test]
#[ignore = "reads the full ERA5 cube; run manually to see peak-batch bounding"]
async fn streaming_bounds_peak_batch_on_large_cube() {
    // batch_size well under one outer plane, so each window is a single plane.
    let ctx = ctx_with_batch_size(1_000_000);
    register_zarr_table(&ctx, "data", ERA5_V3);
    let batches = zarr_exec_batches(&ctx, "SELECT temperature FROM data").await;

    let total = total_rows(&batches);
    let max_batch = batches.iter().map(|b| b.num_rows()).max().unwrap_or(0);
    println!(
        "ERA5 cube: total={total} rows across {} batches, largest batch={max_batch} rows \
         ({:.1}% of the cube)",
        batches.len(),
        100.0 * max_batch as f64 / total as f64
    );

    assert!(
        batches.len() > 1,
        "large cube must stream into multiple batches"
    );
    // The whole point: the biggest batch is bounded to about one plane, not the
    // whole cube. Un-windowed, this would be a single `total`-row batch.
    assert!(
        max_batch * 2 <= total,
        "largest batch ({max_batch}) should be at most ~half the cube ({total})"
    );
}
