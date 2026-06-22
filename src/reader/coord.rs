//! Coordinate utilities for Zarr array flattening
//!
//! Provides utilities for building DictionaryArrays from coordinate values,
//! calculating subset ranges for limit optimization, and building coordinate keys.
//!
//! # Compact Coordinate Encoding
//!
//! For regularly-spaced coordinates (common in scientific data), we use compact
//! encodings that store O(1) parameters instead of O(N) values:
//!
//! - **Arithmetic**: `value[i] = first + i * step` (lat/lon grids, uniform time series)
//! - **Gaussian**: (future) Gaussian quadrature points for climate models
//! - **Logarithmic**: (future) Log-spaced pressure levels
//!
//! This reduces memory usage and enables lazy value generation at query time.

use arrow::array::{
    ArrayRef, DictionaryArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Int16Type, Int32Type, Int64Type};
use std::sync::Arc;
use tracing::debug;

/// Tolerance for detecting arithmetic sequences in floating-point coordinates
const ARITHMETIC_TOLERANCE_F64: f64 = 1e-10;

// ============================================================================
// CompactCoord: Extensible enum for O(1) coordinate encodings
// ============================================================================

/// Compact coordinate encoding - stores O(1) parameters instead of O(N) values.
///
/// Extensible design: add new variants for different grid types (Gaussian, etc.)
/// without changing the CoordValues API.
#[derive(Debug, Clone, Copy)]
pub enum CompactCoord {
    /// Arithmetic sequence: value[i] = first + i * step
    /// Used for regular lat/lon grids, uniform time series.
    Arithmetic { first: f64, step: f64, len: usize },

    /// Arithmetic sequence for integer coordinates
    ArithmeticInt { first: i64, step: i64, len: usize },
    // Future variants:
    // /// Gaussian quadrature points: latitudes for spectral climate models
    // Gaussian { n_points: usize },
    //
    // /// Logarithmic spacing: value[i] = base^(start + i)
    // Logarithmic { base: f64, start: f64, len: usize },
}

impl CompactCoord {
    /// Number of coordinate values
    pub fn len(&self) -> usize {
        match self {
            CompactCoord::Arithmetic { len, .. } => *len,
            CompactCoord::ArithmeticInt { len, .. } => *len,
        }
    }

    /// Check if there are no coordinate values
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Generate value at index i as f64
    #[inline]
    pub fn value_at_f64(&self, i: usize) -> f64 {
        match self {
            CompactCoord::Arithmetic { first, step, .. } => first + (i as f64) * step,
            CompactCoord::ArithmeticInt { first, step, .. } => (*first + (i as i64) * *step) as f64,
        }
    }

    /// Generate value at index i as i64 (for integer coords)
    #[inline]
    pub fn value_at_i64(&self, i: usize) -> i64 {
        match self {
            CompactCoord::ArithmeticInt { first, step, .. } => first + (i as i64) * step,
            CompactCoord::Arithmetic { first, step, .. } => {
                (first + (i as f64) * step).round() as i64
            }
        }
    }

    /// Generate all values as Vec<f64>
    pub fn to_vec_f64(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.value_at_f64(i)).collect()
    }

    /// Generate all values as Vec<i64>
    pub fn to_vec_i64(&self) -> Vec<i64> {
        (0..self.len()).map(|i| self.value_at_i64(i)).collect()
    }

    /// Returns a summary string for debugging
    pub fn summary(&self) -> String {
        match self {
            CompactCoord::Arithmetic { first, step, len } => {
                format!("[{}, +{}, len={}] (arithmetic f64)", first, step, len)
            }
            CompactCoord::ArithmeticInt { first, step, len } => {
                format!("[{}, +{}, len={}] (arithmetic i64)", first, step, len)
            }
        }
    }

    /// Check if this is an integer-based encoding
    pub fn is_integer(&self) -> bool {
        matches!(self, CompactCoord::ArithmeticInt { .. })
    }
}

// ============================================================================
// Detection functions: check if values form a compact pattern
// ============================================================================

/// Try to detect if i64 values form an arithmetic sequence
pub fn try_as_compact_i64(values: &[i64]) -> Option<CompactCoord> {
    if values.is_empty() {
        return None;
    }
    if values.len() == 1 {
        return Some(CompactCoord::ArithmeticInt {
            first: values[0],
            step: 0,
            len: 1,
        });
    }

    let first = values[0];
    let step = values[1] - values[0];

    // Verify all values follow the pattern
    for (i, &v) in values.iter().enumerate() {
        let expected = first + (i as i64) * step;
        if v != expected {
            return None;
        }
    }

    Some(CompactCoord::ArithmeticInt {
        first,
        step,
        len: values.len(),
    })
}

/// Try to detect if f64 values form an arithmetic sequence
pub fn try_as_compact_f64(values: &[f64]) -> Option<CompactCoord> {
    if values.is_empty() {
        return None;
    }
    if values.len() == 1 {
        return Some(CompactCoord::Arithmetic {
            first: values[0],
            step: 0.0,
            len: 1,
        });
    }

    let first = values[0];
    let step = values[1] - values[0];

    // Verify all values follow the pattern within tolerance
    for (i, &v) in values.iter().enumerate() {
        let expected = first + (i as f64) * step;
        if (v - expected).abs() > ARITHMETIC_TOLERANCE_F64 {
            return None;
        }
    }

    Some(CompactCoord::Arithmetic {
        first,
        step,
        len: values.len(),
    })
}

/// Try to detect if f32 values form an arithmetic sequence (converts to f64 internally)
pub fn try_as_compact_f32(values: &[f32]) -> Option<CompactCoord> {
    if values.is_empty() {
        return None;
    }
    if values.len() == 1 {
        return Some(CompactCoord::Arithmetic {
            first: values[0] as f64,
            step: 0.0,
            len: 1,
        });
    }

    let first = values[0] as f64;
    let step = (values[1] - values[0]) as f64;

    // Use looser tolerance for f32 source data
    let tolerance = 1e-5;
    for (i, &v) in values.iter().enumerate() {
        let expected = first + (i as f64) * step;
        if ((v as f64) - expected).abs() > tolerance {
            return None;
        }
    }

    Some(CompactCoord::Arithmetic {
        first,
        step,
        len: values.len(),
    })
}

// ============================================================================
// CoordValues: unified representation for coordinate data
// ============================================================================

/// Coordinate values - either compact encoding or explicit values.
///
/// For regularly-spaced coordinates, uses O(1) memory via CompactCoord.
/// Falls back to explicit Vec storage for irregular coordinates.
#[derive(Debug, Clone)]
pub enum CoordValues {
    /// Compact encoding (arithmetic, gaussian, etc.) - O(1) memory
    Compact {
        encoding: CompactCoord,
        /// Whether this represents timestamp microseconds
        is_timestamp: bool,
    },

    /// Explicit i64 values (irregular spacing)
    Int64(Vec<i64>),
    /// Explicit f32 values (irregular spacing)
    Float32(Vec<f32>),
    /// Explicit f64 values (irregular spacing)
    Float64(Vec<f64>),
    /// Timestamps as microseconds since Unix epoch
    TimestampMicros(Vec<i64>),
}

impl CoordValues {
    /// Create CoordValues from i64 values, using compact encoding if possible
    pub fn from_i64(values: Vec<i64>) -> Self {
        if let Some(encoding) = try_as_compact_i64(&values) {
            debug!("Detected compact i64 coord: {}", encoding.summary());
            CoordValues::Compact {
                encoding,
                is_timestamp: false,
            }
        } else {
            CoordValues::Int64(values)
        }
    }

    /// Create CoordValues from f32 values
    ///
    /// Note: f32 coordinates are kept as explicit values to preserve the data type.
    /// Compact encoding uses f64 internally, which would cause type mismatches.
    pub fn from_f32(values: Vec<f32>) -> Self {
        // Keep f32 as explicit to preserve type - compact encoding uses f64
        CoordValues::Float32(values)
    }

    /// Create CoordValues from f64 values, using compact encoding if possible
    pub fn from_f64(values: Vec<f64>) -> Self {
        if let Some(encoding) = try_as_compact_f64(&values) {
            debug!("Detected compact f64 coord: {}", encoding.summary());
            CoordValues::Compact {
                encoding,
                is_timestamp: false,
            }
        } else {
            CoordValues::Float64(values)
        }
    }

    /// Create CoordValues for timestamps, using compact encoding if possible
    pub fn from_timestamp_micros(values: Vec<i64>) -> Self {
        if let Some(encoding) = try_as_compact_i64(&values) {
            debug!("Detected compact timestamp coord: {}", encoding.summary());
            CoordValues::Compact {
                encoding,
                is_timestamp: true,
            }
        } else {
            CoordValues::TimestampMicros(values)
        }
    }

    /// Number of coordinate values
    pub fn len(&self) -> usize {
        match self {
            CoordValues::Compact { encoding, .. } => encoding.len(),
            CoordValues::Int64(v) => v.len(),
            CoordValues::Float32(v) => v.len(),
            CoordValues::Float64(v) => v.len(),
            CoordValues::TimestampMicros(v) => v.len(),
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if this coordinate uses compact encoding
    pub fn is_compact(&self) -> bool {
        matches!(self, CoordValues::Compact { .. })
    }

    /// Returns a summary string for debugging
    pub fn summary(&self) -> String {
        match self {
            CoordValues::Compact {
                encoding,
                is_timestamp,
            } => {
                let suffix = if *is_timestamp { " (timestamp)" } else { "" };
                format!("{}{}", encoding.summary(), suffix)
            }
            CoordValues::Int64(v) if v.len() > 2 => {
                format!(
                    "[{}, ..., {}] (len={}, i64)",
                    v.first().unwrap(),
                    v.last().unwrap(),
                    v.len()
                )
            }
            CoordValues::Float32(v) if v.len() > 2 => {
                format!(
                    "[{}, ..., {}] (len={}, f32)",
                    v.first().unwrap(),
                    v.last().unwrap(),
                    v.len()
                )
            }
            CoordValues::Float64(v) if v.len() > 2 => {
                format!(
                    "[{}, ..., {}] (len={}, f64)",
                    v.first().unwrap(),
                    v.last().unwrap(),
                    v.len()
                )
            }
            CoordValues::TimestampMicros(v) if v.len() > 2 => {
                format!(
                    "[{}, ..., {}] (len={}, timestamp)",
                    v.first().unwrap(),
                    v.last().unwrap(),
                    v.len()
                )
            }
            _ => format!("{:?}", self),
        }
    }

    /// Slice the coordinate values for a given range
    pub fn slice(&self, start: usize, end: usize) -> CoordValues {
        match self {
            CoordValues::Compact {
                encoding,
                is_timestamp,
            } => {
                // For compact encoding, adjust the parameters
                let new_len = end - start;
                let new_encoding = match encoding {
                    CompactCoord::Arithmetic { first, step, .. } => CompactCoord::Arithmetic {
                        first: first + (start as f64) * step,
                        step: *step,
                        len: new_len,
                    },
                    CompactCoord::ArithmeticInt { first, step, .. } => {
                        CompactCoord::ArithmeticInt {
                            first: first + (start as i64) * step,
                            step: *step,
                            len: new_len,
                        }
                    }
                };
                CoordValues::Compact {
                    encoding: new_encoding,
                    is_timestamp: *is_timestamp,
                }
            }
            CoordValues::Int64(v) => CoordValues::Int64(v[start..end].to_vec()),
            CoordValues::Float32(v) => CoordValues::Float32(v[start..end].to_vec()),
            CoordValues::Float64(v) => CoordValues::Float64(v[start..end].to_vec()),
            CoordValues::TimestampMicros(v) => CoordValues::TimestampMicros(v[start..end].to_vec()),
        }
    }

    /// Gather values at the given scattered indices into a new CoordValues.
    ///
    /// Used when a DatePart filter produces non-contiguous index sets.
    /// Compact encodings are expanded to explicit values since the result
    /// is no longer an arithmetic sequence.
    pub fn gather(&self, indices: &[usize]) -> CoordValues {
        match self {
            CoordValues::Compact {
                encoding,
                is_timestamp,
            } => {
                let vals: Vec<i64> = indices.iter().map(|&i| encoding.value_at_i64(i)).collect();
                if *is_timestamp {
                    CoordValues::TimestampMicros(vals)
                } else {
                    CoordValues::Int64(vals)
                }
            }
            CoordValues::Int64(v) => CoordValues::Int64(indices.iter().map(|&i| v[i]).collect()),
            CoordValues::Float32(v) => {
                CoordValues::Float32(indices.iter().map(|&i| v[i]).collect())
            }
            CoordValues::Float64(v) => {
                CoordValues::Float64(indices.iter().map(|&i| v[i]).collect())
            }
            CoordValues::TimestampMicros(v) => {
                CoordValues::TimestampMicros(indices.iter().map(|&i| v[i]).collect())
            }
        }
    }

    /// Get values as i64 Vec (for filtering). Generates from compact encoding if needed.
    pub fn as_i64_vec(&self) -> Vec<i64> {
        match self {
            CoordValues::Compact { encoding, .. } => encoding.to_vec_i64(),
            CoordValues::Int64(v) => v.clone(),
            CoordValues::Float32(v) => v.iter().map(|x| *x as i64).collect(),
            CoordValues::Float64(v) => v.iter().map(|x| *x as i64).collect(),
            CoordValues::TimestampMicros(v) => v.clone(),
        }
    }

    /// Get values as f64 Vec (for filtering). Generates from compact encoding if needed.
    pub fn as_f64_vec(&self) -> Vec<f64> {
        match self {
            CoordValues::Compact { encoding, .. } => encoding.to_vec_f64(),
            CoordValues::Int64(v) => v.iter().map(|x| *x as f64).collect(),
            CoordValues::Float32(v) => v.iter().map(|x| *x as f64).collect(),
            CoordValues::Float64(v) => v.clone(),
            CoordValues::TimestampMicros(v) => v.iter().map(|x| *x as f64).collect(),
        }
    }

    /// Check if this is a timestamp coordinate
    pub fn is_timestamp(&self) -> bool {
        match self {
            CoordValues::Compact { is_timestamp, .. } => *is_timestamp,
            CoordValues::TimestampMicros(_) => true,
            _ => false,
        }
    }

    /// Check if this uses integer values (i64 or compact int)
    pub fn is_integer(&self) -> bool {
        match self {
            CoordValues::Compact { encoding, .. } => encoding.is_integer(),
            CoordValues::Int64(_) | CoordValues::TimestampMicros(_) => true,
            CoordValues::Float32(_) | CoordValues::Float64(_) => false,
        }
    }

    /// Check if this uses f32 values (only explicit f32)
    pub fn is_f32(&self) -> bool {
        matches!(self, CoordValues::Float32(_))
    }
}

// ============================================================================
// DictionaryArray creation
// ============================================================================

/// Create a DictionaryArray for a coordinate column with proper type.
///
/// For compact encodings, generates values lazily at query time.
/// For explicit values, uses them directly.
///
/// `key_type` is the dictionary key width declared in the schema for this
/// coordinate (`Int16`, `Int32`, or `Int64`, chosen from the coordinate's
/// cardinality). The keys built here must match it so the produced array agrees
/// with the table schema. Picking the width from the full coordinate cardinality
/// guarantees every key (`< selection size <= cardinality`) fits without the
/// silent `as i16` wraparound that used to panic on coordinates larger than
/// 32,767 distinct values.
pub fn create_coord_dictionary_typed(
    values: &CoordValues,
    coord_idx: usize,
    coord_sizes: &[usize],
    total_rows: usize,
    key_type: &DataType,
) -> ArrayRef {
    debug!(
        "Creating coord dictionary array: values={}, coord_idx={}, coord_sizes={:?}, total_rows={}, key_type={:?}",
        values.summary(),
        coord_idx,
        coord_sizes,
        total_rows,
        key_type,
    );

    let values_array = coord_values_to_array(values);
    build_dictionary_with_key_type(key_type, coord_idx, coord_sizes, total_rows, values_array)
}

/// Materialize the dictionary *value* array (the distinct coordinate values),
/// independent of the key width.
fn coord_values_to_array(values: &CoordValues) -> ArrayRef {
    match values {
        CoordValues::Compact {
            encoding,
            is_timestamp,
        } => {
            if *is_timestamp {
                Arc::new(
                    TimestampMicrosecondArray::from(encoding.to_vec_i64())
                        .with_timezone("UTC".to_string()),
                )
            } else if encoding.is_integer() {
                Arc::new(Int64Array::from(encoding.to_vec_i64()))
            } else {
                Arc::new(Float64Array::from(encoding.to_vec_f64()))
            }
        }
        CoordValues::Int64(vals) => Arc::new(Int64Array::from(vals.clone())),
        CoordValues::Float32(vals) => Arc::new(Float32Array::from(vals.clone())),
        CoordValues::Float64(vals) => Arc::new(Float64Array::from(vals.clone())),
        CoordValues::TimestampMicros(vals) => {
            Arc::new(TimestampMicrosecondArray::from(vals.clone()).with_timezone("UTC".to_string()))
        }
    }
}

/// Build the dictionary keys at the requested width and wrap them with `values_array`.
///
/// The raw key for each row is `(row_idx / inner_size) % coord_size`, always in
/// `0..coord_size`. We assert the width can hold `coord_size - 1` so an undersized
/// `key_type` fails loudly with a clear message instead of wrapping silently.
fn build_dictionary_with_key_type(
    key_type: &DataType,
    coord_idx: usize,
    coord_sizes: &[usize],
    total_rows: usize,
    values_array: ArrayRef,
) -> ArrayRef {
    let raw_keys = build_coord_keys_range(coord_idx, coord_sizes, 0, total_rows);
    let coord_size = coord_sizes[coord_idx];
    let max_key = coord_size.saturating_sub(1);

    match key_type {
        DataType::Int16 => {
            assert!(
                max_key <= i16::MAX as usize,
                "coordinate too large for Int16 dictionary keys: {coord_size} distinct values \
                 (max {})",
                i16::MAX
            );
            let keys: Vec<i16> = raw_keys.into_iter().map(|k| k as i16).collect();
            Arc::new(DictionaryArray::<Int16Type>::new(
                Int16Array::from(keys),
                values_array,
            ))
        }
        DataType::Int32 => {
            assert!(
                max_key <= i32::MAX as usize,
                "coordinate too large for Int32 dictionary keys: {coord_size} distinct values \
                 (max {})",
                i32::MAX
            );
            let keys: Vec<i32> = raw_keys.into_iter().map(|k| k as i32).collect();
            Arc::new(DictionaryArray::<Int32Type>::new(
                Int32Array::from(keys),
                values_array,
            ))
        }
        DataType::Int64 => {
            let keys: Vec<i64> = raw_keys.into_iter().map(|k| k as i64).collect();
            Arc::new(DictionaryArray::<Int64Type>::new(
                Int64Array::from(keys),
                values_array,
            ))
        }
        other => panic!("unsupported dictionary key type for coordinate: {other:?}"),
    }
}

// ============================================================================
// Coordinate key computation (unchanged)
// ============================================================================

/// Compute the raw coordinate index for a single row.
///
/// For a Cartesian product of coordinates with sizes [a, b, c, d], row index maps to:
///   key[0] = (row_idx / (b*c*d)) % a
///   key[1] = (row_idx / (c*d)) % b
///   key[2] = (row_idx / d) % c
///   key[3] = row_idx % d
///
/// This is the mathematical property of row-major (C) order arrays. The result is
/// the raw index (`0..coord_size`); callers narrow it to the schema's key width.
#[inline]
pub fn compute_coord_key(row_idx: usize, coord_idx: usize, coord_sizes: &[usize]) -> usize {
    let inner_size: usize = coord_sizes[coord_idx + 1..].iter().product();
    let inner_size = if inner_size == 0 { 1 } else { inner_size };

    let num_values = coord_sizes[coord_idx];
    (row_idx / inner_size) % num_values
}

/// Build raw key indices for DictionaryArray using on-demand computation
///
/// Uses the Cartesian product formula to compute keys without nested loops.
pub fn build_coord_keys(coord_idx: usize, coord_sizes: &[usize], total_rows: usize) -> Vec<usize> {
    build_coord_keys_range(coord_idx, coord_sizes, 0, total_rows)
}

/// Build raw key indices for a range of rows [start_row, start_row + num_rows)
pub fn build_coord_keys_range(
    coord_idx: usize,
    coord_sizes: &[usize],
    start_row: usize,
    num_rows: usize,
) -> Vec<usize> {
    let inner_size: usize = coord_sizes[coord_idx + 1..].iter().product();
    let inner_size = if inner_size == 0 { 1 } else { inner_size };
    let coord_size = coord_sizes[coord_idx];

    (start_row..start_row + num_rows)
        .map(|row_idx| (row_idx / inner_size) % coord_size)
        .collect()
}

// ============================================================================
// Limit/subset calculation utilities (unchanged)
// ============================================================================

/// Calculate the subset ranges needed for a limited number of rows
///
/// For row-major (C) order, the last dimension varies fastest.
pub fn calculate_limited_subset(shape: &[u64], limit: usize) -> Vec<std::ops::Range<u64>> {
    let limit = limit as u64;
    let mut ranges = Vec::with_capacity(shape.len());

    for (i, &dim_size) in shape.iter().enumerate().rev() {
        if i == shape.len() - 1 {
            let take = limit.min(dim_size);
            ranges.push(0..take);
        } else {
            let inner_size: u64 = shape[i + 1..].iter().product();
            let slices_needed = limit.div_ceil(inner_size);
            let take = slices_needed.min(dim_size);
            ranges.push(0..take);
        }
    }

    ranges.reverse();
    ranges
}

/// Calculate how many values we need from each coordinate for a given row limit
pub fn calculate_coord_limits(coord_sizes: &[usize], limit: usize) -> Vec<usize> {
    let mut limits = Vec::with_capacity(coord_sizes.len());
    let n = coord_sizes.len();

    for i in 0..n {
        let inner_size: usize = coord_sizes[i + 1..].iter().product();
        let inner_size = if inner_size == 0 { 1 } else { inner_size };

        let needed = limit.div_ceil(inner_size);
        let take = needed.min(coord_sizes[i]);
        limits.push(take);
    }

    limits
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compact_arithmetic_i64() {
        let values = vec![0, 10, 20, 30, 40];
        let coord = CoordValues::from_i64(values);
        assert!(coord.is_compact());
        assert_eq!(coord.len(), 5);

        if let CoordValues::Compact { encoding, .. } = coord {
            assert_eq!(encoding.value_at_i64(0), 0);
            assert_eq!(encoding.value_at_i64(2), 20);
            assert_eq!(encoding.value_at_i64(4), 40);
        }
    }

    #[test]
    fn test_compact_arithmetic_f64() {
        let values = vec![-90.0, -89.5, -89.0, -88.5, -88.0];
        let coord = CoordValues::from_f64(values);
        assert!(coord.is_compact());
        assert_eq!(coord.len(), 5);

        if let CoordValues::Compact { encoding, .. } = coord {
            assert!((encoding.value_at_f64(0) - (-90.0)).abs() < 1e-10);
            assert!((encoding.value_at_f64(2) - (-89.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_irregular_values_not_compact() {
        let values = vec![1, 2, 4, 8, 16]; // Not arithmetic
        let coord = CoordValues::from_i64(values.clone());
        assert!(!coord.is_compact());
        assert_eq!(coord.len(), 5);

        if let CoordValues::Int64(v) = coord {
            assert_eq!(v, values);
        }
    }

    #[test]
    fn test_compact_slice() {
        let values = vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
        let coord = CoordValues::from_i64(values);
        let sliced = coord.slice(2, 5);

        assert!(sliced.is_compact());
        assert_eq!(sliced.len(), 3);

        if let CoordValues::Compact { encoding, .. } = sliced {
            assert_eq!(encoding.value_at_i64(0), 20);
            assert_eq!(encoding.value_at_i64(1), 30);
            assert_eq!(encoding.value_at_i64(2), 40);
        }
    }

    #[test]
    fn test_timestamp_compact() {
        // Hourly timestamps
        let hour_micros = 3600 * 1_000_000i64;
        let values = vec![0, hour_micros, 2 * hour_micros, 3 * hour_micros];
        let coord = CoordValues::from_timestamp_micros(values);

        assert!(coord.is_compact());
        if let CoordValues::Compact { is_timestamp, .. } = coord {
            assert!(is_timestamp);
        }
    }

    // ------------------------------------------------------------------
    // Adaptive dictionary key width (Int16 -> Int32 -> Int64)
    // ------------------------------------------------------------------

    use arrow::array::Array;

    /// A coordinate of `n` distinct integer values laid out as a single axis.
    fn single_axis_coord(n: usize) -> CoordValues {
        CoordValues::Int64((0..n as i64).collect())
    }

    #[test]
    fn test_create_dictionary_int16_keys() {
        let coord = single_axis_coord(5);
        let dict = create_coord_dictionary_typed(&coord, 0, &[5], 5, &DataType::Int16);

        let dict = dict
            .as_any()
            .downcast_ref::<DictionaryArray<Int16Type>>()
            .expect("expected Int16-keyed dictionary");
        assert_eq!(dict.len(), 5);
        // Single axis: key[row] == row
        assert_eq!(dict.keys().values(), &[0i16, 1, 2, 3, 4]);
    }

    #[test]
    fn test_create_dictionary_int32_keys_large_coordinate() {
        // 40,000 distinct values exceeds the old Int16 ceiling (32,767) and used
        // to panic in DictionaryArray::new via a silent `as i16` wrap.
        let n = 40_000;
        let coord = single_axis_coord(n);
        let dict = create_coord_dictionary_typed(&coord, 0, &[n], n, &DataType::Int32);

        let dict = dict
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .expect("expected Int32-keyed dictionary");
        assert_eq!(dict.len(), n);
        // First and last keys index past the Int16 range without wrapping.
        assert_eq!(dict.keys().value(0), 0);
        assert_eq!(dict.keys().value(n - 1), (n - 1) as i32);
        assert_eq!(dict.values().len(), n);
    }

    #[test]
    fn test_create_dictionary_int64_keys() {
        let coord = single_axis_coord(8);
        let dict = create_coord_dictionary_typed(&coord, 0, &[8], 8, &DataType::Int64);

        let dict = dict
            .as_any()
            .downcast_ref::<DictionaryArray<Int64Type>>()
            .expect("expected Int64-keyed dictionary");
        assert_eq!(dict.len(), 8);
        assert_eq!(dict.keys().value(7), 7i64);
    }

    #[test]
    #[should_panic(expected = "too large for Int16")]
    fn test_int16_key_overflow_panics_clearly() {
        // Defense in depth: an undersized key type fails loudly instead of
        // wrapping silently (the old behavior).
        let n = 40_000;
        let coord = single_axis_coord(n);
        let _ = create_coord_dictionary_typed(&coord, 0, &[n], n, &DataType::Int16);
    }

    #[test]
    fn test_cartesian_product_keys_multi_axis() {
        // sizes [2, 3]: row-major keys for axis 0 and axis 1 over 6 rows.
        // axis 0 key = row / 3 ; axis 1 key = row % 3
        assert_eq!(build_coord_keys(0, &[2, 3], 6), vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(build_coord_keys(1, &[2, 3], 6), vec![0, 1, 2, 0, 1, 2]);
    }
}
