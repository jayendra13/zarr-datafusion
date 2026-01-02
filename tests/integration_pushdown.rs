//! Integration tests for projection and limit pushdown verification
//!
//! Verifies that DataFusion correctly pushes down projections and limits
//! to the ZarrExec plan.

mod common;

use common::*;
use datafusion::physical_plan::ExecutionPlan;

#[tokio::test]
async fn test_pushdown_projection_single_column() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT temperature FROM data").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    // Verify only temperature column is in projected schema
    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(
        projected_schema.fields().len(),
        1,
        "Should project only 1 column"
    );
    assert_eq!(projected_schema.field(0).name(), "temperature");
}

#[tokio::test]
async fn test_pushdown_projection_multiple_columns() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT lat, lon FROM data").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(
        projected_schema.fields().len(),
        2,
        "Should project 2 columns"
    );
}

#[tokio::test]
async fn test_pushdown_projection_all_columns() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT * FROM data").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(
        projected_schema.fields().len(),
        5,
        "Should project all 5 columns"
    );
}

#[tokio::test]
async fn test_pushdown_limit_small() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 10").await;

    assert_eq!(batch.num_rows(), 10, "Should return exactly 10 rows");
}

#[tokio::test]
async fn test_pushdown_limit_larger_than_data() {
    let ctx = create_test_context();
    let (_, meta) = register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 10000").await;

    // Should return all rows, not 10000
    assert_eq!(
        batch.num_rows(),
        meta.total_rows,
        "Should return all available rows"
    );
}

#[tokio::test]
async fn test_pushdown_limit_with_projection() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT lat, lon FROM data LIMIT 50").await;

    assert_eq!(batch.num_rows(), 50);
    assert_eq!(batch.num_columns(), 2);
}

#[tokio::test]
async fn test_pushdown_projection_data_variable_only() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT humidity FROM data LIMIT 10").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(projected_schema.fields().len(), 1);
    assert_eq!(projected_schema.field(0).name(), "humidity");
}

#[tokio::test]
async fn test_pushdown_projection_coords_only() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT lat, lon, time FROM data LIMIT 10").await;
    let zarr_exec = find_zarr_exec(&plan).expect("Should have ZarrExec");

    let projected_schema = zarr_exec.properties().equivalence_properties().schema();
    assert_eq!(projected_schema.fields().len(), 3);
}

#[tokio::test]
async fn test_pushdown_projection_preserves_data() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Query single column
    let batch_single = execute_query_single(
        &ctx,
        "SELECT temperature FROM data ORDER BY lat, lon, time LIMIT 50",
    )
    .await;

    // Query all columns
    let batch_all = execute_query_single(
        &ctx,
        "SELECT temperature FROM (SELECT * FROM data ORDER BY lat, lon, time LIMIT 50)",
    )
    .await;

    // Both should have same temperature values
    let temp_single = format!("{:?}", batch_single.column(0));
    let temp_all = format!("{:?}", batch_all.column(0));
    assert_eq!(
        temp_single, temp_all,
        "Projection should not affect data values"
    );
}

#[tokio::test]
async fn test_pushdown_limit_with_order_by() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(
        &ctx,
        "SELECT temperature FROM data ORDER BY temperature LIMIT 5",
    )
    .await;

    assert_eq!(batch.num_rows(), 5);
}

#[tokio::test]
async fn test_pushdown_limit_one() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 1").await;

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 5);
}

// =============================================================================
// Filter pushdown tests
// =============================================================================

#[tokio::test]
async fn test_pushdown_filter_single_coordinate() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on time coordinate (first coordinate in synthetic data)
    // synthetic data has time = [0, 1, 2, 3, 4, 5, 6]
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time = 0").await;

    // Should return rows where time = 0 (all lat × lon combinations)
    // With time fixed, rows = lat(10) × lon(10) = 100
    assert_eq!(
        batch.num_rows(),
        100,
        "Should return lat × lon rows for time=0"
    );
}

#[tokio::test]
async fn test_pushdown_filter_multiple_coordinates() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on time and lat coordinates
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time = 0 AND lat = 0").await;

    // With time and lat fixed, rows = lon(10) = 10
    assert_eq!(
        batch.num_rows(),
        10,
        "Should return lon rows for time=0 AND lat=0"
    );
}

#[tokio::test]
async fn test_pushdown_filter_all_coordinates() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on all coordinates - should return single row
    let batch = execute_query_single(
        &ctx,
        "SELECT * FROM data WHERE time = 0 AND lat = 0 AND lon = 0",
    )
    .await;

    assert_eq!(
        batch.num_rows(),
        1,
        "Should return single row when all coordinates specified"
    );
}

#[tokio::test]
async fn test_pushdown_filter_with_projection() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter with specific columns
    let batch = execute_query_single(
        &ctx,
        "SELECT temperature, humidity FROM data WHERE time = 0 AND lat = 0",
    )
    .await;

    assert_eq!(batch.num_columns(), 2, "Should project only 2 columns");
    assert_eq!(batch.num_rows(), 10, "Should return filtered rows");
}

#[tokio::test]
async fn test_pushdown_filter_with_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter + limit combination
    let batch = execute_query_single(&ctx, "SELECT * FROM data WHERE time = 0 LIMIT 5").await;

    assert_eq!(batch.num_rows(), 5, "Should return limited rows");
}

#[tokio::test]
async fn test_pushdown_filter_preserves_data_correctness() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on time coordinate (which is Int64)
    let batch = execute_query_single(&ctx, "SELECT time FROM data WHERE time = 3").await;

    // All returned time values should be 3
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::Int16Type;
    let time_col = batch.column(0);
    let time_dict = time_col.as_dictionary::<Int16Type>();

    // Check that all dictionary keys resolve to value 3
    for i in 0..batch.num_rows() {
        if !time_dict.is_null(i) {
            let key = time_dict.keys().value(i);
            let values = time_dict.values();
            let value = values
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .expect("time should be Int64")
                .value(key as usize);
            assert_eq!(value, 3, "All time values should be 3");
        }
    }
}

#[tokio::test]
async fn test_pushdown_filter_nonexistent_value_returns_empty() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on value that doesn't exist
    let batches = execute_query(&ctx, "SELECT * FROM data WHERE time = 9999").await;

    // Should return 0 rows (filter not found in coordinates)
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 0,
        "Should return no rows for non-existent filter value"
    );
}

#[tokio::test]
async fn test_pushdown_filter_on_data_variable_not_pushed() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // Filter on data variable (temperature) - should NOT be pushed down
    // but should still work via DataFusion's filter
    let batches = execute_query(&ctx, "SELECT * FROM data WHERE temperature > 290 LIMIT 10").await;

    // Should still return results (filter applied by DataFusion)
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows <= 10, "Should respect limit");
}
