//! Integration tests for optimizer rule effectiveness
//!
//! Tests that CountStatisticsRule and MinMaxStatisticsRule correctly
//! optimize queries to use statistics without scanning data.

mod common;

use common::*;

// ============== COUNT Optimization Tests ==============

#[tokio::test]
async fn test_optimizer_count_star_uses_statistics() {
    let ctx = create_test_context();
    let (_, meta) = register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT COUNT(*) FROM data").await;

    // Optimized plan should NOT contain ZarrExec
    assert_no_zarr_exec(&plan);

    // Verify result is correct
    let batch = execute_query_single(&ctx, "SELECT COUNT(*) FROM data").await;
    let count = get_scalar_i64(&batch);

    assert_eq!(count as usize, meta.total_rows);
}

#[tokio::test]
async fn test_optimizer_count_column_uses_statistics() {
    let ctx = create_test_context();
    let (_, meta) = register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // COUNT(column) with null_count=0 should also be optimized
    let batch = execute_query_single(&ctx, "SELECT COUNT(temperature) FROM data").await;
    let count = get_scalar_i64(&batch);

    // With null_count = 0, COUNT(col) = total_rows
    assert_eq!(count as usize, meta.total_rows);
}

#[tokio::test]
async fn test_optimizer_count_with_filter_not_optimized() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT COUNT(*) FROM data WHERE temperature > 0").await;

    // Filter prevents optimization, so ZarrExec should be present
    assert_has_zarr_exec(&plan);
}

#[tokio::test]
async fn test_optimizer_count_with_group_by_not_optimized() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT time, COUNT(*) FROM data GROUP BY time").await;

    // GROUP BY prevents optimization
    assert_has_zarr_exec(&plan);
}

// ============== MIN/MAX Optimization Tests ==============

#[tokio::test]
async fn test_optimizer_min_coordinate_uses_statistics() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT MIN(lat) FROM data").await;

    // Should be optimized - no ZarrExec
    assert_no_zarr_exec(&plan);
}

#[tokio::test]
async fn test_optimizer_max_coordinate_uses_statistics() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT MAX(lon) FROM data").await;

    assert_no_zarr_exec(&plan);
}

#[tokio::test]
async fn test_optimizer_minmax_multiple_coords() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan =
        get_physical_plan(&ctx, "SELECT MIN(lat), MAX(lat), MIN(lon), MAX(lon) FROM data").await;

    assert_no_zarr_exec(&plan);

    // Verify values are returned
    let batch = execute_query_single(
        &ctx,
        "SELECT MIN(lat), MAX(lat), MIN(lon), MAX(lon) FROM data",
    )
    .await;

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 4);
}

#[tokio::test]
async fn test_optimizer_minmax_data_var_not_optimized() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // MIN/MAX on data variables (not coordinates) should NOT be optimized
    // because data variables don't have pre-computed statistics
    let plan =
        get_physical_plan(&ctx, "SELECT MIN(temperature), MAX(temperature) FROM data").await;

    // Data vars don't have statistics, so ZarrExec should be present
    assert_has_zarr_exec(&plan);
}

#[tokio::test]
async fn test_optimizer_minmax_with_filter_not_optimized() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT MIN(lat) FROM data WHERE temperature > 0").await;

    // Filter prevents optimization
    assert_has_zarr_exec(&plan);
}

#[tokio::test]
async fn test_optimizer_minmax_with_group_by_not_optimized() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT time, MIN(lat) FROM data GROUP BY time").await;

    // GROUP BY prevents optimization
    assert_has_zarr_exec(&plan);
}

// ============== Compare Optimized vs Unoptimized ==============

#[tokio::test]
async fn test_optimizer_count_result_matches_scan() {
    // Use context with optimizer
    let opt_ctx = create_test_context();
    register_zarr_table(&opt_ctx, "data", SYNTHETIC_V3);

    // Use context without custom optimizer
    let base_ctx = create_baseline_context();
    register_zarr_table(&base_ctx, "data", SYNTHETIC_V3);

    let opt_result = execute_query_single(&opt_ctx, "SELECT COUNT(*) as cnt FROM data").await;

    let base_result = execute_query_single(&base_ctx, "SELECT COUNT(*) as cnt FROM data").await;

    // Both should return same value
    let opt_count = get_scalar_i64(&opt_result);
    let base_count = get_scalar_i64(&base_result);

    assert_eq!(opt_count, base_count);
}

#[tokio::test]
async fn test_optimizer_minmax_result_matches_scan() {
    use arrow::array::Int64Array;

    // Use context with optimizer
    let opt_ctx = create_test_context();
    register_zarr_table(&opt_ctx, "data", SYNTHETIC_V3);

    // Use context without custom optimizer
    let base_ctx = create_baseline_context();
    register_zarr_table(&base_ctx, "data", SYNTHETIC_V3);

    let opt_result = execute_query_single(&opt_ctx, "SELECT MIN(lat), MAX(lat) FROM data").await;
    let base_result = execute_query_single(&base_ctx, "SELECT MIN(lat), MAX(lat) FROM data").await;

    // Compare actual values (not string representation which includes schema metadata)
    let opt_min = opt_result.column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    let opt_max = opt_result.column(1).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    let base_min = base_result.column(0).as_any().downcast_ref::<Int64Array>().unwrap().value(0);
    let base_max = base_result.column(1).as_any().downcast_ref::<Int64Array>().unwrap().value(0);

    assert_eq!(opt_min, base_min, "MIN values should match");
    assert_eq!(opt_max, base_max, "MAX values should match");
}

#[tokio::test]
async fn test_optimizer_mixed_aggregate_not_fully_optimized() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Mix of count and sum - sum requires data scan
    let plan = get_physical_plan(&ctx, "SELECT COUNT(*), SUM(temperature) FROM data").await;

    // SUM requires data scan, so ZarrExec should be present
    assert_has_zarr_exec(&plan);
}

#[tokio::test]
async fn test_optimizer_count_and_minmax_combined() {
    let ctx = create_test_context();
    let (_, meta) = register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Verify values returned are correct
    let batch =
        execute_query_single(&ctx, "SELECT COUNT(*), MIN(lat), MAX(lon) FROM data").await;

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 3);

    // Verify count is correct
    let count = get_scalar_i64(&batch);
    assert_eq!(count as usize, meta.total_rows);
}
