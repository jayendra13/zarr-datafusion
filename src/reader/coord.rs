//! Coordinate utilities for Zarr array flattening
//!
//! Provides utilities for building DictionaryArrays from coordinate values,
//! calculating subset ranges for limit optimization, and building coordinate keys.

use arrow::array::{ArrayRef, DictionaryArray, Float32Array, Float64Array, Int16Array, Int64Array};
use arrow::datatypes::Int16Type;
use std::sync::Arc;

/// Coordinate values that can be either i64 or f32/f64
pub enum CoordValues {
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

impl CoordValues {
    pub fn len(&self) -> usize {
        match self {
            CoordValues::Int64(v) => v.len(),
            CoordValues::Float32(v) => v.len(),
            CoordValues::Float64(v) => v.len(),
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Create a DictionaryArray for a coordinate column with proper type
pub fn create_coord_dictionary_typed(
    values: &CoordValues,
    coord_idx: usize,
    coord_sizes: &[usize],
    total_rows: usize,
) -> ArrayRef {
    let keys = build_coord_keys(values.len(), coord_idx, coord_sizes, total_rows);
    let keys_array = Int16Array::from(keys);

    match values {
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
    }
}

/// Build keys array for DictionaryArray
///
/// Instead of expanding [0,1,2] to [0,0,0,1,1,1,2,2,2,...] (700 i64 values = 5600 bytes),
/// we store:
///   - values: [0,1,2] (the unique coordinate values)
///   - keys: [0,0,0,1,1,1,2,2,2,...] (indices into values, as i16)
///
/// Memory: 700 i16 keys (1400 bytes) + 3 i64 values (24 bytes) = 1424 bytes (~75% savings)
///
/// References:
/// - Arrow DictionaryArray: https://docs.rs/arrow/latest/arrow/array/struct.DictionaryArray.html
/// - DataFusion Dictionary support: https://datafusion.apache.org/user-guide/sql/data_types.html
pub fn build_coord_keys(
    num_values: usize,
    coord_idx: usize,
    coord_sizes: &[usize],
    total_rows: usize,
) -> Vec<i16> {
    let mut keys: Vec<i16> = Vec::with_capacity(total_rows);

    // Elements after this coordinate (inner loop size)
    let inner_size: usize = coord_sizes[coord_idx + 1..].iter().product();
    let inner_size = if inner_size == 0 { 1 } else { inner_size };

    // Elements before this coordinate (outer loop count)
    let outer_count: usize = coord_sizes[..coord_idx].iter().product();
    let outer_count = if outer_count == 0 { 1 } else { outer_count };

    for _ in 0..outer_count {
        for i in 0..num_values {
            for _ in 0..inner_size {
                keys.push(i as i16);
            }
        }
    }

    keys
}

/// Calculate the subset ranges needed for a limited number of rows
///
/// For row-major (C) order, the last dimension varies fastest.
/// Given shape [a, b, c, d] and limit N, we need to figure out
/// which ranges to read to get exactly N elements.
pub fn calculate_limited_subset(shape: &[u64], limit: usize) -> Vec<std::ops::Range<u64>> {
    let limit = limit as u64;
    let mut ranges = Vec::with_capacity(shape.len());

    // Work backwards from the last dimension
    for (i, &dim_size) in shape.iter().enumerate().rev() {
        if i == shape.len() - 1 {
            // Last dimension: take min(limit, dim_size)
            let take = limit.min(dim_size);
            ranges.push(0..take);
        } else {
            // Earlier dimensions: calculate how many complete "slices" we need
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
///
/// For coords with sizes [a, b, c, d] and limit N rows:
/// - coord d (last): need min(N, d) values
/// - coord c: need ceil(N / d) values, capped at c
/// - coord b: need ceil(N / (c*d)) values, capped at b
/// - coord a: need ceil(N / (b*c*d)) values, capped at a
pub fn calculate_coord_limits(coord_sizes: &[usize], limit: usize) -> Vec<usize> {
    let mut limits = Vec::with_capacity(coord_sizes.len());
    let n = coord_sizes.len();

    for i in 0..n {
        // Product of all coords after this one
        let inner_size: usize = coord_sizes[i + 1..].iter().product();
        let inner_size = if inner_size == 0 { 1 } else { inner_size };

        // How many values do we need from this coord?
        let needed = limit.div_ceil(inner_size);
        let take = needed.min(coord_sizes[i]);
        limits.push(take);
    }

    limits
}
