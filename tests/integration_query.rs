//! Integration tests for full SQL-to-RecordBatch query pipeline
//!
//! Tests basic SQL operations against Zarr data stores.

mod common;

use arrow::array::{DictionaryArray, Int64Array};
use arrow::datatypes::Int16Type;
use arrow::datatypes::DataType;
use common::*;

#[tokio::test]
async fn test_query_select_star() {
    let ctx = create_test_context();
    let (_, meta) = register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT * FROM data").await;

    // 5 columns: lat, lon, time (coords) + humidity, temperature (data vars)
    assert_eq!(batch.num_columns(), 5);
    // 10 * 10 * 7 = 700 rows
    assert_eq!(batch.num_rows(), meta.total_rows);
}

#[tokio::test]
async fn test_query_select_specific_columns() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT lat, temperature FROM data LIMIT 100").await;

    assert_eq!(batch.num_columns(), 2);
    assert_eq!(batch.schema().field(0).name(), "lat");
    assert_eq!(batch.schema().field(1).name(), "temperature");
    assert_eq!(batch.num_rows(), 100);
}

#[tokio::test]
async fn test_query_with_where_clause() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(
        &ctx,
        "SELECT temperature FROM data WHERE temperature > 30 LIMIT 50",
    )
    .await;

    // Verify all returned values satisfy filter
    let temp_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Expected Int64Array");

    for i in 0..batch.num_rows() {
        assert!(
            temp_col.value(i) > 30,
            "Row {} has temperature {} which should be > 30",
            i,
            temp_col.value(i)
        );
    }
}

#[tokio::test]
async fn test_query_with_order_by() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch =
        execute_query_single(&ctx, "SELECT temperature FROM data ORDER BY temperature LIMIT 20")
            .await;

    // Verify ascending order
    let temp_col = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Expected Int64Array");

    for i in 1..batch.num_rows() {
        assert!(
            temp_col.value(i - 1) <= temp_col.value(i),
            "Row {} ({}) should be <= row {} ({})",
            i - 1,
            temp_col.value(i - 1),
            i,
            temp_col.value(i)
        );
    }
}

#[tokio::test]
async fn test_query_with_group_by() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(
        &ctx,
        "SELECT time, AVG(temperature) as avg_temp FROM data GROUP BY time ORDER BY time",
    )
    .await;

    // 7 unique time values
    assert_eq!(batch.num_rows(), 7);
    assert_eq!(batch.num_columns(), 2);
}

#[tokio::test]
async fn test_query_with_limit() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT * FROM data LIMIT 100").await;

    assert_eq!(batch.num_rows(), 100);
}

#[tokio::test]
async fn test_query_aggregate_functions() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(
        &ctx,
        "SELECT
            SUM(temperature) as sum_temp,
            COUNT(*) as total
         FROM data",
    )
    .await;

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 2);

    // Verify count matches expected row count
    let count = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Expected Int64Array")
        .value(0);
    assert_eq!(count, 700);
}

#[tokio::test]
async fn test_dictionary_encoding_coordinates() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT lat, lon, time FROM data LIMIT 10").await;

    // All coordinate columns should be Dictionary type
    for i in 0..3 {
        let dtype = batch.column(i).data_type();
        assert!(
            matches!(dtype, DataType::Dictionary(_, _)),
            "Column '{}' should be Dictionary type, got {:?}",
            batch.schema().field(i).name(),
            dtype
        );
    }
}

#[tokio::test]
async fn test_dictionary_values_correct() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT lat FROM data").await;

    let dict_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<DictionaryArray<Int16Type>>()
        .expect("Should be DictionaryArray<Int16Type>");

    // Values array should have 10 unique lat values (0-9)
    let values = dict_array
        .values()
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Values should be Int64Array");

    assert_eq!(values.len(), 10);
}

#[tokio::test]
async fn test_query_count_star() {
    let ctx = create_test_context();
    let (_, meta) = register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(&ctx, "SELECT COUNT(*) as cnt FROM data").await;

    let count = get_scalar_i64(&batch);
    assert_eq!(count as usize, meta.total_rows);
}

#[tokio::test]
async fn test_query_multiple_aggregates() {
    let ctx = create_test_context();
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let batch = execute_query_single(
        &ctx,
        "SELECT
            MIN(lat) as min_lat,
            MAX(lat) as max_lat,
            MIN(lon) as min_lon,
            MAX(lon) as max_lon
         FROM data",
    )
    .await;

    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 4);
}
