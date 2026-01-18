//! User-Defined Functions (UDFs) for weather metric calculations
//!
//! Provides scalar and aggregate functions for forecast verification:
//! - Scalar: mae, bias, squared_error, grid_round, kelvin_to_celsius, is_freezing, within_window
//! - Aggregate: rmse, mean_mae, spatial_mean, correlation

mod aggregate;
mod scalar;

pub use aggregate::*;
pub use scalar::*;

use datafusion::prelude::SessionContext;

/// Register all metric UDFs with a DataFusion SessionContext
pub fn register_metric_udfs(ctx: &SessionContext) {
    // Scalar functions
    ctx.register_udf(scalar::mae_udf());
    ctx.register_udf(scalar::bias_udf());
    ctx.register_udf(scalar::squared_error_udf());
    ctx.register_udf(scalar::grid_round_udf());
    ctx.register_udf(scalar::kelvin_to_celsius_udf());
    ctx.register_udf(scalar::is_freezing_udf());
    ctx.register_udf(scalar::within_window_udf());

    // Aggregate functions
    ctx.register_udaf(aggregate::rmse_udaf());
    ctx.register_udaf(aggregate::mean_mae_udaf());
    ctx.register_udaf(aggregate::spatial_mean_udaf());
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionContext;
    use std::sync::Arc;

    async fn create_test_context() -> SessionContext {
        let ctx = SessionContext::new();
        register_metric_udfs(&ctx);
        ctx
    }

    #[tokio::test]
    async fn test_scalar_udfs_in_sql() {
        let ctx = create_test_context().await;

        // Create test data
        let forecast = Float64Array::from(vec![300.0, 290.0, 280.0, 270.0]);
        let target = Float64Array::from(vec![298.0, 292.0, 282.0, 268.0]);
        let batch = RecordBatch::try_from_iter(vec![
            ("forecast", Arc::new(forecast) as _),
            ("target", Arc::new(target) as _),
        ])
        .unwrap();

        ctx.register_batch("temps", batch).unwrap();

        // Test mae
        let result = ctx
            .sql("SELECT mae(forecast, target) as error FROM temps")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let errors = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(errors.len(), 4);
        assert!((errors.value(0) - 2.0).abs() < 0.001); // |300 - 298| = 2
    }

    #[tokio::test]
    async fn test_aggregate_udfs_in_sql() {
        let ctx = create_test_context().await;

        let forecast = Float64Array::from(vec![300.0, 290.0, 280.0, 270.0]);
        let target = Float64Array::from(vec![298.0, 292.0, 282.0, 268.0]);
        let batch = RecordBatch::try_from_iter(vec![
            ("forecast", Arc::new(forecast) as _),
            ("target", Arc::new(target) as _),
        ])
        .unwrap();

        ctx.register_batch("temps", batch).unwrap();

        // Test rmse aggregate
        let result = ctx
            .sql("SELECT rmse(forecast, target) as rmse_val FROM temps")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let rmse_val = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);

        // Expected: sqrt(mean([4, 4, 4, 4])) = sqrt(4) = 2
        assert!((rmse_val - 2.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_grid_round_udf() {
        let ctx = create_test_context().await;

        let coords = Float64Array::from(vec![45.12, 45.13, 45.24, 45.26]);
        let batch = RecordBatch::try_from_iter(vec![("lat", Arc::new(coords) as _)]).unwrap();

        ctx.register_batch("coords", batch).unwrap();

        let result = ctx
            .sql("SELECT grid_round(lat, 0.25) as rounded FROM coords")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let rounded = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert!((rounded.value(0) - 45.0).abs() < 0.001);
        assert!((rounded.value(1) - 45.25).abs() < 0.001); // 45.13 rounds to 45.25
        assert!((rounded.value(2) - 45.25).abs() < 0.001);
        assert!((rounded.value(3) - 45.25).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_within_window_udf() {
        let ctx = create_test_context().await;

        // Times in nanoseconds: center at 100, check ±12 hours
        let hour_ns: i64 = 3600 * 1_000_000_000;
        let times = Int64Array::from(vec![
            100 * hour_ns,                // center
            100 * hour_ns + 6 * hour_ns,  // +6h (in window)
            100 * hour_ns - 12 * hour_ns, // -12h (edge, in window)
            100 * hour_ns + 13 * hour_ns, // +13h (out of window)
        ]);
        let batch = RecordBatch::try_from_iter(vec![("time_ns", Arc::new(times) as _)]).unwrap();

        ctx.register_batch("times", batch).unwrap();

        let result = ctx
            .sql(&format!(
                "SELECT within_window(time_ns, {}, 12) as in_window FROM times",
                100 * hour_ns
            ))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let in_window = result[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .unwrap();

        assert!(in_window.value(0)); // center
        assert!(in_window.value(1)); // +6h
        assert!(in_window.value(2)); // -12h
        assert!(!in_window.value(3)); // +13h
    }
}
