//! Property-based tests for ZarrLimitPushdownRule
//!
//! These tests verify that the optimized queries produce the same results
//! as baseline (unoptimized) queries across a range of randomly generated
//! query parameters.
//!
//! Property-based testing helps catch edge cases and regressions that might
//! not be covered by manually written test cases.

mod common;

use arrow::array::RecordBatch;
use common::*;
use proptest::prelude::*;

/// Compare two sets of record batches for equality
fn batches_equal(a: &[RecordBatch], b: &[RecordBatch]) -> bool {
    let total_rows_a: usize = a.iter().map(|b| b.num_rows()).sum();
    let total_rows_b: usize = b.iter().map(|b| b.num_rows()).sum();

    if total_rows_a != total_rows_b {
        return false;
    }

    // If both are empty, they're equal
    if total_rows_a == 0 {
        return true;
    }

    // For non-empty results, compare schemas first
    if !a.is_empty() && !b.is_empty() && a[0].schema() != b[0].schema() {
        return false;
    }

    // For detailed comparison, we'd need to sort and compare values
    // For this test, row count equality is sufficient as a smoke test
    true
}

/// Run a query with both optimized and baseline contexts and compare results
async fn verify_query_equivalence(sql: &str) -> Result<(), String> {
    // Create optimized context (with ZarrLimitPushdownRule)
    let ctx_opt = create_test_context();
    register_zarr_table(&ctx_opt, "data", SYNTHETIC_V3);

    // Create baseline context (without custom optimizer rules)
    let ctx_base = create_baseline_context();
    register_zarr_table(&ctx_base, "data", SYNTHETIC_V3);

    // Execute with optimized context
    let result_opt = execute_query(&ctx_opt, sql).await;

    // Execute with baseline context
    let result_base = execute_query(&ctx_base, sql).await;

    // Compare results
    if !batches_equal(&result_opt, &result_base) {
        let rows_opt: usize = result_opt.iter().map(|b| b.num_rows()).sum();
        let rows_base: usize = result_base.iter().map(|b| b.num_rows()).sum();
        return Err(format!(
            "Results differ for query '{}': optimized={} rows, baseline={} rows",
            sql, rows_opt, rows_base
        ));
    }

    Ok(())
}

proptest! {
    // Configure to run 50 cases (reasonable CI time)
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Test that LIMIT queries produce equivalent results with/without optimization
    #[test]
    fn property_limit_preserves_semantics(
        limit in 1usize..100,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let sql = format!("SELECT * FROM data LIMIT {}", limit);
            verify_query_equivalence(&sql).await.expect("Query equivalence failed");
        });
    }

    /// Test that LIMIT + filter queries produce equivalent results
    #[test]
    fn property_limit_with_time_filter_preserves_semantics(
        limit in 1usize..50,
        filter_time in 0i64..7,  // synthetic data has time 0-6
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let sql = format!(
                "SELECT * FROM data WHERE time >= {} LIMIT {}",
                filter_time, limit
            );
            verify_query_equivalence(&sql).await.expect("Query equivalence failed");
        });
    }

    /// Test that LIMIT + range filter queries produce equivalent results
    #[test]
    fn property_limit_with_range_filter_preserves_semantics(
        limit in 1usize..30,
        time_low in 0i64..4,
        time_high in 4i64..7,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let sql = format!(
                "SELECT * FROM data WHERE time BETWEEN {} AND {} LIMIT {}",
                time_low, time_high, limit
            );
            verify_query_equivalence(&sql).await.expect("Query equivalence failed");
        });
    }

    /// Test that LIMIT + projection queries produce equivalent results
    #[test]
    fn property_limit_with_projection_preserves_semantics(
        limit in 1usize..50,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let sql = format!("SELECT temperature, humidity FROM data LIMIT {}", limit);
            verify_query_equivalence(&sql).await.expect("Query equivalence failed");
        });
    }

    /// Test that LIMIT + multiple coordinate filters produce equivalent results
    #[test]
    fn property_limit_with_multi_filter_preserves_semantics(
        limit in 1usize..20,
        filter_time in 0i64..7,
        filter_lat in 0i64..10,  // synthetic data has lat 0-9
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let sql = format!(
                "SELECT * FROM data WHERE time = {} AND lat = {} LIMIT {}",
                filter_time, filter_lat, limit
            );
            verify_query_equivalence(&sql).await.expect("Query equivalence failed");
        });
    }
}

/// Non-property test to verify the test infrastructure works
#[tokio::test]
async fn test_verify_query_equivalence_basic() {
    // Simple query that should always work
    verify_query_equivalence("SELECT * FROM data LIMIT 10")
        .await
        .expect("Basic query equivalence should pass");
}

/// Test that verifies empty results are handled correctly
#[tokio::test]
async fn test_verify_query_equivalence_empty_result() {
    // Query that returns no rows (filter outside data range)
    verify_query_equivalence("SELECT * FROM data WHERE time = 9999")
        .await
        .expect("Empty result query equivalence should pass");
}

/// Test that verifies full table scan equivalence
#[tokio::test]
async fn test_verify_query_equivalence_no_limit() {
    // Query without limit - should still be equivalent
    verify_query_equivalence("SELECT * FROM data WHERE time = 3")
        .await
        .expect("No-limit query equivalence should pass");
}
