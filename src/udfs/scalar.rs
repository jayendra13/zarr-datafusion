//! Scalar UDFs for weather metric calculations
//!
//! Point-wise functions that operate on individual values.

use arrow::array::{Array, BooleanArray, Float64Array, Int64Array};
use arrow::datatypes::DataType;
use datafusion::common::Result;
use datafusion::logical_expr::{ColumnarValue, ScalarUDF, ScalarUDFImpl, Signature, Volatility};
use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// ============================================================================
// MAE (Mean Absolute Error) - scalar version
// ============================================================================

#[derive(Debug)]
struct MaeUdf {
    signature: Signature,
}

impl MaeUdf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for MaeUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for MaeUdf {}

impl Hash for MaeUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl ScalarUDFImpl for MaeUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "mae"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let forecast = args.args[0].clone().into_array(args.number_rows)?;
        let target = args.args[1].clone().into_array(args.number_rows)?;

        let forecast = forecast.as_any().downcast_ref::<Float64Array>().unwrap();
        let target = target.as_any().downcast_ref::<Float64Array>().unwrap();

        let result: Float64Array = forecast
            .iter()
            .zip(target.iter())
            .map(|(f, t)| match (f, t) {
                (Some(f), Some(t)) => Some((f - t).abs()),
                _ => None,
            })
            .collect();

        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

pub fn mae_udf() -> ScalarUDF {
    ScalarUDF::from(MaeUdf::new())
}

// ============================================================================
// BIAS (signed error)
// ============================================================================

#[derive(Debug)]
struct BiasUdf {
    signature: Signature,
}

impl BiasUdf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for BiasUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for BiasUdf {}

impl Hash for BiasUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl ScalarUDFImpl for BiasUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "bias"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let forecast = args.args[0].clone().into_array(args.number_rows)?;
        let target = args.args[1].clone().into_array(args.number_rows)?;

        let forecast = forecast.as_any().downcast_ref::<Float64Array>().unwrap();
        let target = target.as_any().downcast_ref::<Float64Array>().unwrap();

        let result: Float64Array = forecast
            .iter()
            .zip(target.iter())
            .map(|(f, t)| match (f, t) {
                (Some(f), Some(t)) => Some(f - t),
                _ => None,
            })
            .collect();

        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

pub fn bias_udf() -> ScalarUDF {
    ScalarUDF::from(BiasUdf::new())
}

// ============================================================================
// SQUARED_ERROR (for RMSE calculation)
// ============================================================================

#[derive(Debug)]
struct SquaredErrorUdf {
    signature: Signature,
}

impl SquaredErrorUdf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for SquaredErrorUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for SquaredErrorUdf {}

impl Hash for SquaredErrorUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl ScalarUDFImpl for SquaredErrorUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "squared_error"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let forecast = args.args[0].clone().into_array(args.number_rows)?;
        let target = args.args[1].clone().into_array(args.number_rows)?;

        let forecast = forecast.as_any().downcast_ref::<Float64Array>().unwrap();
        let target = target.as_any().downcast_ref::<Float64Array>().unwrap();

        let result: Float64Array = forecast
            .iter()
            .zip(target.iter())
            .map(|(f, t)| match (f, t) {
                (Some(f), Some(t)) => Some((f - t).powi(2)),
                _ => None,
            })
            .collect();

        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

pub fn squared_error_udf() -> ScalarUDF {
    ScalarUDF::from(SquaredErrorUdf::new())
}

// ============================================================================
// GRID_ROUND (round coordinate to nearest grid resolution)
// ============================================================================

#[derive(Debug)]
struct GridRoundUdf {
    signature: Signature,
}

impl GridRoundUdf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for GridRoundUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for GridRoundUdf {}

impl Hash for GridRoundUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl ScalarUDFImpl for GridRoundUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "grid_round"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let coord = args.args[0].clone().into_array(args.number_rows)?;
        let resolution = args.args[1].clone().into_array(args.number_rows)?;

        let coord = coord.as_any().downcast_ref::<Float64Array>().unwrap();
        let resolution = resolution.as_any().downcast_ref::<Float64Array>().unwrap();

        // Resolution is typically a scalar, replicated for all rows
        let res = resolution.value(0);

        let result: Float64Array = coord
            .iter()
            .map(|c| c.map(|c| (c / res).round() * res))
            .collect();

        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

pub fn grid_round_udf() -> ScalarUDF {
    ScalarUDF::from(GridRoundUdf::new())
}

// ============================================================================
// KELVIN_TO_CELSIUS
// ============================================================================

#[derive(Debug)]
struct KelvinToCelsiusUdf {
    signature: Signature,
}

impl KelvinToCelsiusUdf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl PartialEq for KelvinToCelsiusUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for KelvinToCelsiusUdf {}

impl Hash for KelvinToCelsiusUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl ScalarUDFImpl for KelvinToCelsiusUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "kelvin_to_celsius"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let kelvin = args.args[0].clone().into_array(args.number_rows)?;
        let kelvin = kelvin.as_any().downcast_ref::<Float64Array>().unwrap();

        let result: Float64Array = kelvin.iter().map(|k| k.map(|k| k - 273.15)).collect();

        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

pub fn kelvin_to_celsius_udf() -> ScalarUDF {
    ScalarUDF::from(KelvinToCelsiusUdf::new())
}

// ============================================================================
// IS_FREEZING (temperature below 273.15K / 0°C)
// ============================================================================

#[derive(Debug)]
struct IsFreezingUdf {
    signature: Signature,
}

impl IsFreezingUdf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl PartialEq for IsFreezingUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for IsFreezingUdf {}

impl Hash for IsFreezingUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl ScalarUDFImpl for IsFreezingUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "is_freezing"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let temp = args.args[0].clone().into_array(args.number_rows)?;
        let temp = temp.as_any().downcast_ref::<Float64Array>().unwrap();

        let result: BooleanArray = temp.iter().map(|t| t.map(|t| t < 273.15)).collect();

        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

pub fn is_freezing_udf() -> ScalarUDF {
    ScalarUDF::from(IsFreezingUdf::new())
}

// ============================================================================
// WITHIN_WINDOW (check if time is within ±hours of center)
// ============================================================================

#[derive(Debug)]
struct WithinWindowUdf {
    signature: Signature,
}

impl WithinWindowUdf {
    fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Int64, DataType::Int64, DataType::Int64],
                Volatility::Immutable,
            ),
        }
    }
}

impl PartialEq for WithinWindowUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for WithinWindowUdf {}

impl Hash for WithinWindowUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl ScalarUDFImpl for WithinWindowUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "within_window"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let time = args.args[0].clone().into_array(args.number_rows)?;
        let center = args.args[1].clone().into_array(args.number_rows)?;
        let hours = args.args[2].clone().into_array(args.number_rows)?;

        let time = time.as_any().downcast_ref::<Int64Array>().unwrap();
        let center = center.as_any().downcast_ref::<Int64Array>().unwrap();
        let hours = hours.as_any().downcast_ref::<Int64Array>().unwrap();

        // Center and hours are typically scalars
        let center_val = center.value(0);
        let hours_val = hours.value(0);
        let window_ns = hours_val * 3600 * 1_000_000_000; // hours to nanoseconds

        let result: BooleanArray = time
            .iter()
            .map(|t| t.map(|t| t >= (center_val - window_ns) && t <= (center_val + window_ns)))
            .collect();

        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

pub fn within_window_udf() -> ScalarUDF {
    ScalarUDF::from(WithinWindowUdf::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mae_calculation() {
        let forecast = Float64Array::from(vec![Some(300.0), Some(290.0), None]);
        let target = Float64Array::from(vec![Some(298.0), Some(292.0), Some(280.0)]);

        let udf = MaeUdf::new();

        // Create minimal args for testing
        let args = datafusion::logical_expr::ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(forecast)),
                ColumnarValue::Array(Arc::new(target)),
            ],
            arg_fields: vec![],
            number_rows: 3,
            return_field: Arc::new(arrow::datatypes::Field::new(
                "result",
                DataType::Float64,
                true,
            )),
            config_options: Arc::new(datafusion::config::ConfigOptions::default()),
        };

        let result = udf.invoke_with_args(args).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let result = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            assert!((result.value(0) - 2.0).abs() < 0.001);
            assert!((result.value(1) - 2.0).abs() < 0.001);
            assert!(result.is_null(2));
        }
    }

    #[test]
    fn test_grid_round_calculation() {
        // 45.12 / 0.25 = 180.48 -> rounds to 180 -> 45.0
        // 45.13 / 0.25 = 180.52 -> rounds to 181 -> 45.25
        // 45.24 / 0.25 = 180.96 -> rounds to 181 -> 45.25
        // 45.26 / 0.25 = 181.04 -> rounds to 181 -> 45.25
        let coord = Float64Array::from(vec![45.12, 45.13, 45.24, 45.26]);
        let resolution = Float64Array::from(vec![0.25, 0.25, 0.25, 0.25]);

        let udf = GridRoundUdf::new();

        let args = datafusion::logical_expr::ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(coord)),
                ColumnarValue::Array(Arc::new(resolution)),
            ],
            arg_fields: vec![],
            number_rows: 4,
            return_field: Arc::new(arrow::datatypes::Field::new(
                "result",
                DataType::Float64,
                true,
            )),
            config_options: Arc::new(datafusion::config::ConfigOptions::default()),
        };

        let result = udf.invoke_with_args(args).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let result = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            assert!((result.value(0) - 45.0).abs() < 0.001);
            assert!((result.value(1) - 45.25).abs() < 0.001); // 45.13 rounds to 45.25
            assert!((result.value(2) - 45.25).abs() < 0.001);
            assert!((result.value(3) - 45.25).abs() < 0.001);
        }
    }

    #[test]
    fn test_kelvin_to_celsius() {
        let kelvin = Float64Array::from(vec![273.15, 300.0, 250.0]);

        let udf = KelvinToCelsiusUdf::new();

        let args = datafusion::logical_expr::ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(kelvin))],
            arg_fields: vec![],
            number_rows: 3,
            return_field: Arc::new(arrow::datatypes::Field::new(
                "result",
                DataType::Float64,
                true,
            )),
            config_options: Arc::new(datafusion::config::ConfigOptions::default()),
        };

        let result = udf.invoke_with_args(args).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let result = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            assert!((result.value(0) - 0.0).abs() < 0.001);
            assert!((result.value(1) - 26.85).abs() < 0.001);
            assert!((result.value(2) - (-23.15)).abs() < 0.001);
        }
    }

    #[test]
    fn test_is_freezing() {
        let temp = Float64Array::from(vec![273.15, 273.14, 280.0, 260.0]);

        let udf = IsFreezingUdf::new();

        let args = datafusion::logical_expr::ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(temp))],
            arg_fields: vec![],
            number_rows: 4,
            return_field: Arc::new(arrow::datatypes::Field::new(
                "result",
                DataType::Boolean,
                true,
            )),
            config_options: Arc::new(datafusion::config::ConfigOptions::default()),
        };

        let result = udf.invoke_with_args(args).unwrap();

        if let ColumnarValue::Array(arr) = result {
            let result = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(!result.value(0)); // 273.15 is not freezing (exactly 0°C)
            assert!(result.value(1)); // 273.14 is freezing
            assert!(!result.value(2)); // 280 is not freezing
            assert!(result.value(3)); // 260 is freezing
        }
    }
}
