//! Aggregate UDFs for weather metric calculations
//!
//! Group-level functions that aggregate across multiple rows:
//! - rmse: Root Mean Squared Error
//! - mean_mae: Mean Absolute Error (aggregate)
//! - spatial_mean: Area-weighted spatial mean

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::datatypes::{DataType, Field};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Signature, Volatility,
};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// ============================================================================
// RMSE (Root Mean Squared Error) Aggregate
// ============================================================================

#[derive(Debug)]
struct RmseUdaf {
    signature: Signature,
}

impl RmseUdaf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for RmseUdaf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for RmseUdaf {}

impl Hash for RmseUdaf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl AggregateUDFImpl for RmseUdaf {
    fn name(&self) -> &str {
        "rmse"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn accumulator(
        &self,
        _acc_args: datafusion::logical_expr::function::AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(RmseAccumulator::new()))
    }

    fn state_fields(
        &self,
        _args: datafusion::logical_expr::function::StateFieldsArgs,
    ) -> Result<Vec<Arc<Field>>> {
        Ok(vec![
            Arc::new(Field::new("sum_squared_error", DataType::Float64, true)),
            Arc::new(Field::new("count", DataType::Int64, true)),
        ])
    }
}

#[derive(Debug, Default)]
struct RmseAccumulator {
    sum_squared_error: f64,
    count: u64,
}

impl RmseAccumulator {
    fn new() -> Self {
        Self::default()
    }
}

impl Accumulator for RmseAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let forecast = values[0].as_primitive::<arrow::datatypes::Float64Type>();
        let target = values[1].as_primitive::<arrow::datatypes::Float64Type>();

        for i in 0..forecast.len() {
            if !forecast.is_null(i) && !target.is_null(i) {
                let diff = forecast.value(i) - target.value(i);
                self.sum_squared_error += diff * diff;
                self.count += 1;
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.count == 0 {
            Ok(ScalarValue::Float64(None))
        } else {
            let mse = self.sum_squared_error / self.count as f64;
            Ok(ScalarValue::Float64(Some(mse.sqrt())))
        }
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            ScalarValue::Float64(Some(self.sum_squared_error)),
            ScalarValue::Int64(Some(self.count as i64)),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let sum_arr = states[0].as_primitive::<arrow::datatypes::Float64Type>();
        let count_arr = states[1].as_primitive::<arrow::datatypes::Int64Type>();

        for i in 0..sum_arr.len() {
            if !sum_arr.is_null(i) && !count_arr.is_null(i) {
                self.sum_squared_error += sum_arr.value(i);
                self.count += count_arr.value(i) as u64;
            }
        }
        Ok(())
    }
}

pub fn rmse_udaf() -> AggregateUDF {
    AggregateUDF::from(RmseUdaf::new())
}

// ============================================================================
// MEAN_MAE (Mean Absolute Error) Aggregate
// ============================================================================

#[derive(Debug)]
struct MeanMaeUdaf {
    signature: Signature,
}

impl MeanMaeUdaf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for MeanMaeUdaf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for MeanMaeUdaf {}

impl Hash for MeanMaeUdaf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl AggregateUDFImpl for MeanMaeUdaf {
    fn name(&self) -> &str {
        "mean_mae"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn accumulator(
        &self,
        _acc_args: datafusion::logical_expr::function::AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(MeanMaeAccumulator::new()))
    }

    fn state_fields(
        &self,
        _args: datafusion::logical_expr::function::StateFieldsArgs,
    ) -> Result<Vec<Arc<Field>>> {
        Ok(vec![
            Arc::new(Field::new("sum_abs_error", DataType::Float64, true)),
            Arc::new(Field::new("count", DataType::Int64, true)),
        ])
    }
}

#[derive(Debug, Default)]
struct MeanMaeAccumulator {
    sum_abs_error: f64,
    count: u64,
}

impl MeanMaeAccumulator {
    fn new() -> Self {
        Self::default()
    }
}

impl Accumulator for MeanMaeAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let forecast = values[0].as_primitive::<arrow::datatypes::Float64Type>();
        let target = values[1].as_primitive::<arrow::datatypes::Float64Type>();

        for i in 0..forecast.len() {
            if !forecast.is_null(i) && !target.is_null(i) {
                let diff = (forecast.value(i) - target.value(i)).abs();
                self.sum_abs_error += diff;
                self.count += 1;
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.count == 0 {
            Ok(ScalarValue::Float64(None))
        } else {
            Ok(ScalarValue::Float64(Some(
                self.sum_abs_error / self.count as f64,
            )))
        }
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            ScalarValue::Float64(Some(self.sum_abs_error)),
            ScalarValue::Int64(Some(self.count as i64)),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let sum_arr = states[0].as_primitive::<arrow::datatypes::Float64Type>();
        let count_arr = states[1].as_primitive::<arrow::datatypes::Int64Type>();

        for i in 0..sum_arr.len() {
            if !sum_arr.is_null(i) && !count_arr.is_null(i) {
                self.sum_abs_error += sum_arr.value(i);
                self.count += count_arr.value(i) as u64;
            }
        }
        Ok(())
    }
}

pub fn mean_mae_udaf() -> AggregateUDF {
    AggregateUDF::from(MeanMaeUdaf::new())
}

// ============================================================================
// SPATIAL_MEAN (Area-weighted mean using cos(latitude))
// ============================================================================

#[derive(Debug)]
struct SpatialMeanUdaf {
    signature: Signature,
}

impl SpatialMeanUdaf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for SpatialMeanUdaf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for SpatialMeanUdaf {}

impl Hash for SpatialMeanUdaf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl AggregateUDFImpl for SpatialMeanUdaf {
    fn name(&self) -> &str {
        "spatial_mean"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn accumulator(
        &self,
        _acc_args: datafusion::logical_expr::function::AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(SpatialMeanAccumulator::new()))
    }

    fn state_fields(
        &self,
        _args: datafusion::logical_expr::function::StateFieldsArgs,
    ) -> Result<Vec<Arc<Field>>> {
        Ok(vec![
            Arc::new(Field::new("weighted_sum", DataType::Float64, true)),
            Arc::new(Field::new("weight_sum", DataType::Float64, true)),
        ])
    }
}

#[derive(Debug, Default)]
struct SpatialMeanAccumulator {
    weighted_sum: f64,
    weight_sum: f64,
}

impl SpatialMeanAccumulator {
    fn new() -> Self {
        Self::default()
    }
}

impl Accumulator for SpatialMeanAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let value = values[0].as_primitive::<arrow::datatypes::Float64Type>();
        let latitude = values[1].as_primitive::<arrow::datatypes::Float64Type>();

        for i in 0..value.len() {
            if !value.is_null(i) && !latitude.is_null(i) {
                // Area weight: cos(latitude in radians)
                let lat_rad = latitude.value(i).to_radians();
                let weight = lat_rad.cos().abs(); // abs() to handle southern hemisphere

                self.weighted_sum += value.value(i) * weight;
                self.weight_sum += weight;
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.weight_sum == 0.0 {
            Ok(ScalarValue::Float64(None))
        } else {
            Ok(ScalarValue::Float64(Some(
                self.weighted_sum / self.weight_sum,
            )))
        }
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            ScalarValue::Float64(Some(self.weighted_sum)),
            ScalarValue::Float64(Some(self.weight_sum)),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let weighted_sum_arr = states[0].as_primitive::<arrow::datatypes::Float64Type>();
        let weight_sum_arr = states[1].as_primitive::<arrow::datatypes::Float64Type>();

        for i in 0..weighted_sum_arr.len() {
            if !weighted_sum_arr.is_null(i) && !weight_sum_arr.is_null(i) {
                self.weighted_sum += weighted_sum_arr.value(i);
                self.weight_sum += weight_sum_arr.value(i);
            }
        }
        Ok(())
    }
}

pub fn spatial_mean_udaf() -> AggregateUDF {
    AggregateUDF::from(SpatialMeanUdaf::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    #[test]
    fn test_rmse_accumulator() {
        let mut acc = RmseAccumulator::new();

        let forecast = Arc::new(Float64Array::from(vec![300.0, 290.0, 280.0, 270.0])) as ArrayRef;
        let target = Arc::new(Float64Array::from(vec![298.0, 292.0, 282.0, 268.0])) as ArrayRef;

        acc.update_batch(&[forecast, target]).unwrap();

        let result = acc.evaluate().unwrap();
        if let ScalarValue::Float64(Some(rmse)) = result {
            // All errors are 2.0, so RMSE = sqrt(mean([4,4,4,4])) = sqrt(4) = 2.0
            assert!((rmse - 2.0).abs() < 0.001);
        } else {
            panic!("Expected Float64 result");
        }
    }

    #[test]
    fn test_mean_mae_accumulator() {
        let mut acc = MeanMaeAccumulator::new();

        let forecast = Arc::new(Float64Array::from(vec![300.0, 290.0, 280.0, 270.0])) as ArrayRef;
        let target = Arc::new(Float64Array::from(vec![298.0, 292.0, 282.0, 268.0])) as ArrayRef;

        acc.update_batch(&[forecast, target]).unwrap();

        let result = acc.evaluate().unwrap();
        if let ScalarValue::Float64(Some(mae)) = result {
            // All absolute errors are 2.0, so MAE = mean([2,2,2,2]) = 2.0
            assert!((mae - 2.0).abs() < 0.001);
        } else {
            panic!("Expected Float64 result");
        }
    }

    #[test]
    fn test_spatial_mean_accumulator() {
        let mut acc = SpatialMeanAccumulator::new();

        // Temperature values at different latitudes
        let values = Arc::new(Float64Array::from(vec![300.0, 300.0, 300.0])) as ArrayRef;
        // Equator (0°), mid-latitude (45°), high-latitude (60°)
        let latitudes = Arc::new(Float64Array::from(vec![0.0, 45.0, 60.0])) as ArrayRef;

        acc.update_batch(&[values, latitudes]).unwrap();

        let result = acc.evaluate().unwrap();
        if let ScalarValue::Float64(Some(mean)) = result {
            // With equal values, the weighted mean should still be 300
            assert!((mean - 300.0).abs() < 0.001);
        } else {
            panic!("Expected Float64 result");
        }
    }

    #[test]
    fn test_spatial_mean_weighting() {
        let mut acc = SpatialMeanAccumulator::new();

        // Different temps at equator vs pole
        // Equator should have more weight (cos(0) = 1) than pole (cos(90) ≈ 0)
        let values = Arc::new(Float64Array::from(vec![300.0, 200.0])) as ArrayRef;
        let latitudes = Arc::new(Float64Array::from(vec![0.0, 89.0])) as ArrayRef;

        acc.update_batch(&[values, latitudes]).unwrap();

        let result = acc.evaluate().unwrap();
        if let ScalarValue::Float64(Some(mean)) = result {
            // Equator weight >> pole weight, so mean should be close to 300
            assert!(
                mean > 290.0,
                "Mean {} should be closer to equator value 300",
                mean
            );
        } else {
            panic!("Expected Float64 result");
        }
    }
}
