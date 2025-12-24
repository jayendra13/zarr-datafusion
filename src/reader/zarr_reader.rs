//! Zarr array reader that flattens nD data into Arrow RecordBatches
//!
//! See [`super::schema_inference`] for assumptions about Zarr store structure
//! (1D coordinates, nD data variables as Cartesian product of coordinates).

use arrow::{
    array::{ArrayRef, DictionaryArray, Float32Array, Float64Array, Int16Array, Int64Array, RecordBatch},
    datatypes::{DataType, Int16Type, Schema, SchemaRef},
};
use datafusion::{
    common::DataFusionError, error::Result, execution::SendableRecordBatchStream,
    physical_plan::stream::RecordBatchStreamAdapter,
};
use futures::stream;
use std::sync::Arc;
use zarrs::{array::Array, array_subset::ArraySubset, filesystem::FilesystemStore};

use super::schema_inference::discover_arrays;

fn zarr_err(e: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Coordinate values that can be either i64 or f32/f64
enum CoordValues {
    Int64(Vec<i64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
}

pub fn read_zarr(
    store_path: &str,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
) -> Result<SendableRecordBatchStream> {
    let store = Arc::new(FilesystemStore::new(store_path).map_err(zarr_err)?);

    // Discover store structure
    let store_meta =
        discover_arrays(store_path).map_err(|e| DataFusionError::External(e))?;

    let coord_names: Vec<_> = store_meta.coords.iter().map(|c| c.name.clone()).collect();
    let coord_types: Vec<_> = store_meta.coords.iter().map(|c| c.data_type.clone()).collect();

    // Load coordinate arrays and get their sizes
    let mut coord_sizes: Vec<usize> = Vec::new();
    let mut coord_values: Vec<CoordValues> = Vec::new();

    for (coord, dtype) in store_meta.coords.iter().zip(coord_types.iter()) {
        let arr = Array::open(store.clone(), &format!("/{}", coord.name)).map_err(zarr_err)?;
        let size = arr.shape()[0] as usize;
        coord_sizes.push(size);

        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        let values = match dtype.as_str() {
            "float32" => {
                let (vals, _) = arr
                    .retrieve_array_subset_ndarray::<f32>(&subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Float32(vals)
            }
            "float64" => {
                let (vals, _) = arr
                    .retrieve_array_subset_ndarray::<f64>(&subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Float64(vals)
            }
            _ => {
                let (vals, _) = arr
                    .retrieve_array_subset_ndarray::<i64>(&subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Int64(vals)
            }
        };
        coord_values.push(values);
    }

    // Total rows = product of all coordinate sizes
    let total_rows: usize = coord_sizes.iter().product();

    let projected_indices = projection.unwrap_or_else(|| (0..schema.fields().len()).collect());

    let mut result_arrays: Vec<ArrayRef> = Vec::new();

    for idx in &projected_indices {
        let field = schema.field(*idx);
        let field_name = field.name();

        // Check if this is a coordinate
        if let Some(coord_idx) = coord_names.iter().position(|n| n == field_name) {
            // Create DictionaryArray for coordinate (memory efficient)
            let dict_array = create_coord_dictionary_typed(
                &coord_values[coord_idx],
                coord_idx,
                &coord_sizes,
                total_rows,
            );
            result_arrays.push(dict_array);
        } else {
            // Data variable - read and flatten based on schema type
            let arr =
                Array::open(store.clone(), &format!("/{}", field_name)).map_err(zarr_err)?;
            let subset = ArraySubset::new_with_shape(arr.shape().to_vec());

            let array: ArrayRef = match field.data_type() {
                DataType::Float32 => {
                    let (vals, _) = arr
                        .retrieve_array_subset_ndarray::<f32>(&subset)
                        .map_err(zarr_err)?
                        .into_raw_vec_and_offset();
                    Arc::new(Float32Array::from(vals))
                }
                DataType::Float64 => {
                    let (vals, _) = arr
                        .retrieve_array_subset_ndarray::<f64>(&subset)
                        .map_err(zarr_err)?
                        .into_raw_vec_and_offset();
                    Arc::new(Float64Array::from(vals))
                }
                _ => {
                    let (vals, _) = arr
                        .retrieve_array_subset_ndarray::<i64>(&subset)
                        .map_err(zarr_err)?
                        .into_raw_vec_and_offset();
                    Arc::new(Int64Array::from(vals))
                }
            };
            result_arrays.push(array);
        }
    }

    let projected_schema = Arc::new(Schema::new(
        projected_indices
            .iter()
            .map(|&i| schema.field(i).clone())
            .collect::<Vec<_>>(),
    ));

    let batch = RecordBatch::try_new(projected_schema.clone(), result_arrays)?;
    let stream = stream::iter(vec![Ok(batch)]);

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

/// Create a DictionaryArray for a coordinate column with proper type
fn create_coord_dictionary_typed(
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
            Arc::new(DictionaryArray::<Int16Type>::new(keys_array, Arc::new(values_array)))
        }
        CoordValues::Float32(vals) => {
            let values_array = Float32Array::from(vals.clone());
            Arc::new(DictionaryArray::<Int16Type>::new(keys_array, Arc::new(values_array)))
        }
        CoordValues::Float64(vals) => {
            let values_array = Float64Array::from(vals.clone());
            Arc::new(DictionaryArray::<Int16Type>::new(keys_array, Arc::new(values_array)))
        }
    }
}

impl CoordValues {
    fn len(&self) -> usize {
        match self {
            CoordValues::Int64(v) => v.len(),
            CoordValues::Float32(v) => v.len(),
            CoordValues::Float64(v) => v.len(),
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
fn build_coord_keys(
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
