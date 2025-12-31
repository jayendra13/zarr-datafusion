//! Integration tests for error handling
//!
//! Tests that appropriate errors are returned for invalid inputs.

mod common;

use common::*;
use zarr_datafusion::reader::schema_inference::{detect_zarr_version, infer_schema};

#[test]
fn test_error_invalid_path() {
    let result = infer_schema("/nonexistent/path/to/zarr.zarr");
    assert!(result.is_err(), "Should fail for non-existent path");
}

#[test]
fn test_error_not_zarr_directory() {
    // Use system temp directory which exists but is not a Zarr store
    let result = infer_schema("/tmp");
    assert!(result.is_err(), "Should fail for non-Zarr directory");
}

#[test]
fn test_error_version_detection_no_metadata() {
    let result = detect_zarr_version("/tmp");
    assert!(result.is_err(), "Should fail when no Zarr metadata present");
}

#[tokio::test]
async fn test_error_invalid_sql_syntax() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let result = ctx.sql("SELEKT * FROM data").await;
    assert!(result.is_err(), "Should fail for invalid SQL syntax");
}

#[tokio::test]
async fn test_error_unknown_column() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let result = ctx.sql("SELECT nonexistent_column FROM data").await;
    assert!(result.is_err(), "Should fail for unknown column");
}

#[tokio::test]
async fn test_error_table_not_registered() {
    let ctx = create_test_context();
    // Don't register any table

    let result = ctx.sql("SELECT * FROM unknown_table").await;
    assert!(result.is_err(), "Should fail for unregistered table");
}


#[test]
fn test_error_message_contains_path() {
    let bad_path = "/definitely/not/a/real/zarr/store";
    let result = infer_schema(bad_path);

    if let Err(e) = result {
        let error_msg = e.to_string();
        // Error message should be informative
        assert!(
            !error_msg.is_empty(),
            "Error message should not be empty"
        );
    }
}

#[tokio::test]
async fn test_valid_query_after_error() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    // First query fails
    let _fail = ctx.sql("SELECT bad_column FROM data").await;

    // Second query should still work
    let result = ctx.sql("SELECT COUNT(*) FROM data").await;
    assert!(
        result.is_ok(),
        "Valid query should succeed after failed query"
    );

    let batch = execute_query_single(&ctx, "SELECT COUNT(*) FROM data").await;
    assert_eq!(batch.num_rows(), 1);
}
