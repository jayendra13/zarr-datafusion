//! Shared test utilities for integration tests
//!
//! Provides helper functions for creating test contexts, registering tables,
//! executing queries, and introspecting physical plans.

#![allow(dead_code)]

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::optimizer::{CountStatisticsRule, MinMaxStatisticsRule};
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;
use zarr_datafusion::reader::schema_inference::{infer_schema_with_meta, ZarrStoreMeta};

// Test data paths
pub const SYNTHETIC_V2: &str = "data/synthetic_v2.zarr";
pub const SYNTHETIC_V2_BLOSC: &str = "data/synthetic_v2_blosc.zarr";
pub const SYNTHETIC_V3: &str = "data/synthetic_v3.zarr";
pub const SYNTHETIC_V3_BLOSC: &str = "data/synthetic_v3_blosc.zarr";
pub const ERA5_V2: &str = "data/era5_v2.zarr";
pub const ERA5_V2_BLOSC: &str = "data/era5_v2_blosc.zarr";
pub const ERA5_V3: &str = "data/era5_v3.zarr";
pub const ERA5_V3_BLOSC: &str = "data/era5_v3_blosc.zarr";

/// All synthetic dataset variants for parameterized tests
pub const ALL_SYNTHETIC: &[&str] = &[
    SYNTHETIC_V2,
    SYNTHETIC_V2_BLOSC,
    SYNTHETIC_V3,
    SYNTHETIC_V3_BLOSC,
];

/// All ERA5 dataset variants
pub const ALL_ERA5: &[&str] = &[ERA5_V2, ERA5_V2_BLOSC, ERA5_V3, ERA5_V3_BLOSC];

/// Create a SessionContext with custom optimizer rules (CountStatisticsRule, MinMaxStatisticsRule)
pub fn create_test_context() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_optimizer_rule(Arc::new(CountStatisticsRule::new()))
        .with_optimizer_rule(Arc::new(MinMaxStatisticsRule::new()))
        .build();
    SessionContext::new_with_state(state)
}

/// Create a SessionContext without custom optimizer rules (baseline for comparison)
pub fn create_baseline_context() -> SessionContext {
    SessionContext::new()
}

/// Register a Zarr table with the context and return schema + metadata
pub fn register_zarr_table(
    ctx: &SessionContext,
    table_name: &str,
    path: &str,
) -> (SchemaRef, ZarrStoreMeta) {
    let (schema, metadata) = infer_schema_with_meta(path).expect("Failed to infer schema");
    let schema = Arc::new(schema);
    let table = Arc::new(ZarrTable::with_metadata(
        schema.clone(),
        path,
        metadata.clone(),
    ));
    ctx.register_table(table_name, table)
        .expect("Failed to register table");
    (schema, metadata)
}

/// Execute a SQL query and collect all RecordBatches
pub async fn execute_query(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    let df = ctx.sql(sql).await.expect("Query failed");
    df.collect().await.expect("Failed to collect results")
}

/// Execute a query expecting a single RecordBatch
pub async fn execute_query_single(ctx: &SessionContext, sql: &str) -> RecordBatch {
    let batches = execute_query(ctx, sql).await;
    assert!(!batches.is_empty(), "Expected at least one batch");
    // If multiple batches, concatenate them
    if batches.len() == 1 {
        batches.into_iter().next().unwrap()
    } else {
        arrow::compute::concat_batches(&batches[0].schema(), &batches)
            .expect("Failed to concat batches")
    }
}

/// Get the physical plan for a query
pub async fn get_physical_plan(
    ctx: &SessionContext,
    sql: &str,
) -> Arc<dyn ExecutionPlan> {
    let df = ctx.sql(sql).await.expect("Query failed");
    df.create_physical_plan()
        .await
        .expect("Failed to create physical plan")
}

/// Find ZarrExec node in a physical plan (recursive search)
pub fn find_zarr_exec(plan: &Arc<dyn ExecutionPlan>) -> Option<&ZarrExec> {
    // Check if current node is ZarrExec
    if let Some(zarr_exec) = plan.as_any().downcast_ref::<ZarrExec>() {
        return Some(zarr_exec);
    }

    // Recursively search children
    for child in plan.children() {
        if let Some(found) = find_zarr_exec(child) {
            return Some(found);
        }
    }

    None
}

/// Assert that a plan does NOT contain ZarrExec (i.e., query was optimized away)
pub fn assert_no_zarr_exec(plan: &Arc<dyn ExecutionPlan>) {
    assert!(
        find_zarr_exec(plan).is_none(),
        "Expected plan to NOT contain ZarrExec (should be optimized away)"
    );
}

/// Assert that a plan DOES contain ZarrExec
pub fn assert_has_zarr_exec(plan: &Arc<dyn ExecutionPlan>) {
    assert!(
        find_zarr_exec(plan).is_some(),
        "Expected plan to contain ZarrExec"
    );
}

/// Get a scalar i64 value from the first column/row of a batch
pub fn get_scalar_i64(batch: &RecordBatch) -> i64 {
    use arrow::array::Int64Array;
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Expected Int64Array")
        .value(0)
}
