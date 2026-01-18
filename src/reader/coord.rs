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
    ArrayRef, DictionaryArray, Float32Array, Float64Array, Int16Array, Int64Array,
    TimestampMicrosecondArray,
};
use arrow::datatypes::Int16Type;
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
pub fn create_coord_dictionary_typed(
    values: &CoordValues,
    coord_idx: usize,
    coord_sizes: &[usize],
    total_rows: usize,
) -> ArrayRef {
    debug!(
        "Creating coord dictionary array: values={}, coord_idx={}, coord_sizes={:?}, total_rows={}",
        values.summary(),
        coord_idx,
        coord_sizes,
        total_rows
    );

    let keys = build_coord_keys(values.len(), coord_idx, coord_sizes, total_rows);
    let keys_array = Int16Array::from(keys);

    match values {
        CoordValues::Compact {
            encoding,
            is_timestamp,
        } => {
            if *is_timestamp {
                // Generate timestamp values from compact encoding
                let vals = encoding.to_vec_i64();
                let values_array =
                    TimestampMicrosecondArray::from(vals).with_timezone("UTC".to_string());
                Arc::new(DictionaryArray::<Int16Type>::new(
                    keys_array,
                    Arc::new(values_array),
                ))
            } else if encoding.is_integer() {
                // Generate i64 values
                let vals = encoding.to_vec_i64();
                let values_array = Int64Array::from(vals);
                Arc::new(DictionaryArray::<Int16Type>::new(
                    keys_array,
                    Arc::new(values_array),
                ))
            } else {
                // Generate f64 values
                let vals = encoding.to_vec_f64();
                let values_array = Float64Array::from(vals);
                Arc::new(DictionaryArray::<Int16Type>::new(
                    keys_array,
                    Arc::new(values_array),
                ))
            }
        }
        CoordValues::Int64(vals) => {
            let values_array = Int64Array::from(vals.clone());
            Arc::new(DictionaryArray::<Int16Type>::new(
                keys_array,
                Arc::new(values_array),
            ))
        }
        CoordValues::Float32(vals) => {
            let values_array = Float32Array::from(vals.clone());
            Arc::new(DictionaryArray::<Int16Type>::new(
                keys_array,
                Arc::new(values_array),
            ))
        }
        CoordValues::Float64(vals) => {
            let values_array = Float64Array::from(vals.clone());
            Arc::new(DictionaryArray::<Int16Type>::new(
                keys_array,
                Arc::new(values_array),
            ))
        }
        CoordValues::TimestampMicros(vals) => {
            let values_array =
                TimestampMicrosecondArray::from(vals.clone()).with_timezone("UTC".to_string());
            Arc::new(DictionaryArray::<Int16Type>::new(
                keys_array,
                Arc::new(values_array),
            ))
        }
    }
}

// ============================================================================
// Coordinate key computation (unchanged)
// ============================================================================

/// Compute coordinate key for a single row index
///
/// For a Cartesian product of coordinates with sizes [a, b, c, d], row index maps to:
///   key[0] = (row_idx / (b*c*d)) % a
///   key[1] = (row_idx / (c*d)) % b
///   key[2] = (row_idx / d) % c
///   key[3] = row_idx % d
///
/// This is the mathematical property of row-major (C) order arrays.
#[inline]
pub fn compute_coord_key(row_idx: usize, coord_idx: usize, coord_sizes: &[usize]) -> i16 {
    let inner_size: usize = coord_sizes[coord_idx + 1..].iter().product();
    let inner_size = if inner_size == 0 { 1 } else { inner_size };

    let num_values = coord_sizes[coord_idx];
    ((row_idx / inner_size) % num_values) as i16
}

/// Build keys array for DictionaryArray using on-demand computation
///
/// Uses the Cartesian product formula to compute keys without nested loops.
pub fn build_coord_keys(
    num_values: usize,
    coord_idx: usize,
    coord_sizes: &[usize],
    total_rows: usize,
) -> Vec<i16> {
    build_coord_keys_range(num_values, coord_idx, coord_sizes, 0, total_rows)
}

/// Build keys array for a range of rows [start_row, start_row + num_rows)
pub fn build_coord_keys_range(
    _num_values: usize,
    coord_idx: usize,
    coord_sizes: &[usize],
    start_row: usize,
    num_rows: usize,
) -> Vec<i16> {
    let inner_size: usize = coord_sizes[coord_idx + 1..].iter().product();
    let inner_size = if inner_size == 0 { 1 } else { inner_size };
    let coord_size = coord_sizes[coord_idx];

    (start_row..start_row + num_rows)
        .map(|row_idx| ((row_idx / inner_size) % coord_size) as i16)
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
}
