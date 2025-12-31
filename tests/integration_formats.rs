//! Integration tests for Zarr v2/v3 format compatibility and compression variants
//!
//! Verifies that both Zarr versions produce identical results and that
//! compressed data is read correctly.

mod common;

use common::*;
use zarr_datafusion::reader::schema_inference::infer_schema_with_meta;

#[test]
fn test_format_v2_v3_same_schema() {
    let (schema_v2, _) = infer_schema_with_meta(SYNTHETIC_V2).unwrap();
    let (schema_v3, _) = infer_schema_with_meta(SYNTHETIC_V3).unwrap();

    assert_eq!(
        schema_v2.fields().len(),
        schema_v3.fields().len(),
        "V2 and V3 should have same number of fields"
    );

    for (f2, f3) in schema_v2.fields().iter().zip(schema_v3.fields().iter()) {
        assert_eq!(f2.name(), f3.name(), "Field names should match");
        assert_eq!(f2.data_type(), f3.data_type(), "Field types should match");
        assert_eq!(
            f2.is_nullable(),
            f3.is_nullable(),
            "Nullability should match"
        );
    }
}

#[test]
fn test_format_v2_v3_same_metadata() {
    let (_, meta_v2) = infer_schema_with_meta(SYNTHETIC_V2).unwrap();
    let (_, meta_v3) = infer_schema_with_meta(SYNTHETIC_V3).unwrap();

    assert_eq!(
        meta_v2.total_rows, meta_v3.total_rows,
        "Total rows should match"
    );
    assert_eq!(
        meta_v2.coords.len(),
        meta_v3.coords.len(),
        "Number of coordinates should match"
    );
    assert_eq!(
        meta_v2.data_vars.len(),
        meta_v3.data_vars.len(),
        "Number of data variables should match"
    );
}

#[tokio::test]
async fn test_format_v2_v3_same_data() {
    let ctx_v2 = create_test_context();
    let ctx_v3 = create_test_context();

    register_zarr_table(&ctx_v2, "data", SYNTHETIC_V2);
    register_zarr_table(&ctx_v3, "data", SYNTHETIC_V3);

    // Query ordered data for consistent comparison
    let batch_v2 = execute_query_single(
        &ctx_v2,
        "SELECT lat, lon, time, temperature, humidity FROM data ORDER BY lat, lon, time LIMIT 100",
    )
    .await;

    let batch_v3 = execute_query_single(
        &ctx_v3,
        "SELECT lat, lon, time, temperature, humidity FROM data ORDER BY lat, lon, time LIMIT 100",
    )
    .await;

    assert_eq!(batch_v2.num_rows(), batch_v3.num_rows());
    assert_eq!(batch_v2.num_columns(), batch_v3.num_columns());

    // Compare column values (using string representation for simplicity)
    for i in 0..batch_v2.num_columns() {
        let col_v2 = format!("{:?}", batch_v2.column(i));
        let col_v3 = format!("{:?}", batch_v3.column(i));
        assert_eq!(
            col_v2, col_v3,
            "Column {} should have identical values",
            batch_v2.schema().field(i).name()
        );
    }
}

#[tokio::test]
async fn test_format_blosc_reads_correctly() {
    let ctx = create_test_context();
    let (_, meta) = register_zarr_table(&ctx, "data", SYNTHETIC_V2_BLOSC);

    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 100").await;

    assert_eq!(batch.num_rows(), 100);
    assert_eq!(batch.num_columns(), 5);
    assert_eq!(meta.total_rows, 700);
}

#[tokio::test]
async fn test_format_blosc_same_as_uncompressed() {
    let ctx_plain = create_test_context();
    let ctx_blosc = create_test_context();

    register_zarr_table(&ctx_plain, "data", SYNTHETIC_V3);
    register_zarr_table(&ctx_blosc, "data", SYNTHETIC_V3_BLOSC);

    let batch_plain = execute_query_single(
        &ctx_plain,
        "SELECT temperature, humidity FROM data ORDER BY lat, lon, time LIMIT 50",
    )
    .await;

    let batch_blosc = execute_query_single(
        &ctx_blosc,
        "SELECT temperature, humidity FROM data ORDER BY lat, lon, time LIMIT 50",
    )
    .await;

    // Data should be identical
    assert_eq!(batch_plain.num_rows(), batch_blosc.num_rows());

    for i in 0..batch_plain.num_columns() {
        let col_plain = format!("{:?}", batch_plain.column(i));
        let col_blosc = format!("{:?}", batch_blosc.column(i));
        assert_eq!(
            col_plain, col_blosc,
            "Column {} should be identical between compressed and uncompressed",
            batch_plain.schema().field(i).name()
        );
    }
}

#[tokio::test]
async fn test_format_all_synthetic_variants_queryable() {
    for path in ALL_SYNTHETIC {
        let ctx = create_test_context();
        register_zarr_table(&ctx, "data", path);

        let batch = execute_query_single(&ctx, "SELECT COUNT(*) FROM data").await;
        let count = get_scalar_i64(&batch);

        assert_eq!(count, 700, "Dataset {} should have 700 rows", path);
    }
}

#[tokio::test]
async fn test_format_all_era5_variants_queryable() {
    for path in ALL_ERA5 {
        let ctx = create_test_context();
        register_zarr_table(&ctx, "data", path);

        let batch = execute_query_single(&ctx, "SELECT COUNT(*) FROM data").await;
        let count = get_scalar_i64(&batch);

        assert!(count > 0, "Dataset {} should have rows", path);
    }
}

#[test]
fn test_format_era5_schema_structure() {
    let (schema, meta) = infer_schema_with_meta(ERA5_V3).unwrap();

    // ERA5 should have coordinates and data variables
    assert!(!meta.coords.is_empty(), "Should have coordinates");
    assert!(!meta.data_vars.is_empty(), "Should have data variables");
    assert!(meta.total_rows > 0, "Should have data");

    // All coordinates should have min/max values
    for coord in &meta.coords {
        assert!(
            coord.coord_min_max.is_some(),
            "Coordinate {} should have min/max",
            coord.name
        );
    }

    // Schema fields should match coords + data_vars
    assert_eq!(
        schema.fields().len(),
        meta.coords.len() + meta.data_vars.len()
    );
}
