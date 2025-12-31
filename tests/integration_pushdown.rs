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

    let batch =
        execute_query_single(&ctx, "SELECT temperature FROM data ORDER BY temperature LIMIT 5")
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
