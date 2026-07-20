//! Disk-byte accounting: what the scan reports having actually fetched.
//!
//! These pin the distinction between "read no bytes" and "nobody counted the bytes".
//! The remote (object-store) path regressed on exactly this: the tracking wrapper was
//! installed with a `None` stats handle, so every GCS/S3/HTTP query reported `0 B`.
//! Because zero is a plausible-looking measurement, nothing surfaced the gap until a
//! benchmark tried to quote bytes-fetched and got zero for a query that had clearly
//! moved megabytes.

mod common;

use common::*;

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;
use zarr_datafusion::reader::stats::SharedIoStats;

/// Run `sql` to completion and return the scan's I/O stats.
///
/// The stats must be read from the *executed* plan: `ZarrExec` accumulates into the
/// handle during execution, so collecting the results is what populates it.
async fn stats_after_running(ctx: &SessionContext, sql: &str) -> SharedIoStats {
    let plan = get_physical_plan(ctx, sql).await;
    let io_stats = find_stats(&plan).expect("a ZarrExec in the plan");
    let task_ctx = ctx.task_ctx();
    datafusion::physical_plan::collect(plan, task_ctx)
        .await
        .expect("query executes");
    io_stats
}

fn find_stats(plan: &Arc<dyn ExecutionPlan>) -> Option<SharedIoStats> {
    if let Some(z) = plan.downcast_ref::<ZarrExec>() {
        return Some(z.io_stats());
    }
    plan.children().iter().find_map(|c| find_stats(c))
}

#[tokio::test]
async fn local_scan_reports_bytes_actually_read() {
    let ctx = SessionContext::new();
    register_zarr_table(&ctx, "t", SYNTHETIC_V3);
    let stats = stats_after_running(&ctx, "SELECT temperature FROM t WHERE lat = -10.0").await;

    let disk = stats
        .disk_bytes_tracked()
        .expect("a local filesystem scan is tracked, so bytes must be reported");
    assert!(
        disk > 0,
        "a scan that returned data must report having read something, got {disk} B"
    );
}

#[tokio::test]
async fn reported_bytes_are_not_wildly_larger_than_the_store() {
    // A loose upper bound, aimed at double-counting: the earlier remote wrapper
    // collects a byte-range stream to measure it, which is the kind of place a chunk
    // gets counted twice. The synthetic store is tiny, so any large multiple of it
    // means the accounting is summing the same reads more than once.
    let ctx = SessionContext::new();
    register_zarr_table(&ctx, "t", SYNTHETIC_V3);
    let stats = stats_after_running(&ctx, "SELECT temperature, humidity FROM t").await;

    let disk = stats.disk_bytes_tracked().expect("tracked");
    let store_bytes = dir_size(std::path::Path::new(SYNTHETIC_V3));
    assert!(
        disk <= store_bytes * 3,
        "read {disk} B from a {store_bytes} B store — suspicious double-counting"
    );
}

#[tokio::test]
async fn a_full_scan_reads_the_store_about_once() {
    // The read-amplification regression, pinned end to end. Before the read window
    // was decoupled from `batch_size`, a full scan of this store read 207 MB from
    // 6.5 MB on disk — each chunk fetched and decompressed 32 times, because a
    // chunk's cells are scattered across 256 row-stripes while the window covered 8.
    //
    // The bound is deliberately loose (2x, not 1.05x): coordinates and metadata are
    // read on top of the chunk data, and the point is to catch a return to 30x+, not
    // to pin an exact byte count.
    let ctx = SessionContext::new();
    register_zarr_table(&ctx, "t", NDVI_SCENE);
    let stats = stats_after_running(&ctx, "SELECT x, y, b04, b08 FROM t").await;

    let disk = stats.disk_bytes_tracked().expect("tracked");
    let store_bytes = dir_size(std::path::Path::new(NDVI_SCENE));
    let ratio = disk as f64 / store_bytes as f64;
    assert!(
        ratio < 2.0,
        "read {disk} B from a {store_bytes} B store ({ratio:.1}x) — \
         the read window is narrower than a chunk row again"
    );
}

#[tokio::test]
async fn output_batches_respect_batch_size() {
    // Reading wide must not mean emitting wide. The window is now sized from chunk
    // geometry (far larger than batch_size), so without slicing this query would
    // hand downstream operators one million-row RecordBatch and break the batching
    // contract they pipeline against.
    let ctx = SessionContext::new_with_config(
        datafusion::prelude::SessionConfig::new().with_batch_size(8192),
    );
    register_zarr_table(&ctx, "t", NDVI_SCENE);
    let batches = execute_query(&ctx, "SELECT x, y, b04 FROM t").await;

    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 1024 * 1024, "every row still arrives");
    assert!(
        batches.len() > 1,
        "expected the window to be sliced into several batches, got {}",
        batches.len()
    );
    for b in &batches {
        assert!(
            b.num_rows() <= 8192,
            "batch of {} rows exceeds batch_size 8192",
            b.num_rows()
        );
    }
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for e in entries.flatten() {
            let p = e.path();
            total += if p.is_dir() {
                dir_size(&p)
            } else {
                p.metadata().map(|m| m.len()).unwrap_or(0)
            };
        }
    }
    total
}
