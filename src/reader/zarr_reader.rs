//! Zarr array reader that flattens nD data into Arrow RecordBatches
//!
//! See [`super::schema_inference`] for assumptions about Zarr store structure
//! (1D coordinates, nD data variables as Cartesian product of coordinates).

use tracing::{debug, info, instrument};

use arrow::{
    array::{
        ArrayRef, DictionaryArray, Float32Array, Float64Array, Int16Array, Int64Array, RecordBatch,
    },
    datatypes::{DataType, Int16Type, Schema, SchemaRef},
};
use datafusion::{
    common::DataFusionError, error::Result, execution::SendableRecordBatchStream,
    physical_plan::stream::RecordBatchStreamAdapter,
};
use futures::stream;
use std::sync::Arc;
use std::time::Instant;
use zarrs::{array::Array, array_subset::ArraySubset, filesystem::FilesystemStore};

use super::schema_inference::discover_arrays;
use super::stats::SharedIoStats;
use super::tracked_store::TrackedStore;

fn zarr_err(e: impl std::error::Error + Send + Sync + 'static) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Get element size in bytes for a Zarr data type string
fn dtype_to_bytes(dtype: &str) -> u64 {
    match dtype {
        "float32" | "int32" | "uint32" => 4,
        "float64" | "int64" | "uint64" => 8,
        "int16" | "uint16" => 2,
        "int8" | "uint8" => 1,
        _ => 8, // Default assumption
    }
}

/// Get element size in bytes for an Arrow DataType
fn arrow_dtype_to_bytes(dtype: &DataType) -> u64 {
    match dtype {
        DataType::Float32 | DataType::Int32 | DataType::UInt32 => 4,
        DataType::Float64 | DataType::Int64 | DataType::UInt64 => 8,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int8 | DataType::UInt8 => 1,
        _ => 8, // Default assumption
    }
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
    limit: Option<usize>,
    stats: Option<SharedIoStats>,
) -> Result<SendableRecordBatchStream> {
    let fs_store = Arc::new(FilesystemStore::new(store_path).map_err(zarr_err)?);

    // Wrap with TrackedStore if stats are provided
    let store: Arc<TrackedStore<FilesystemStore>> = Arc::new(TrackedStore::new(
        fs_store,
        stats.clone().unwrap_or_default(),
    ));

    // Discover store structure (with timing)
    let meta_start = Instant::now();
    let store_meta = discover_arrays(store_path).map_err(DataFusionError::External)?;
    if let Some(ref s) = stats {
        // TODO: Track actual metadata bytes read in discover_arrays() instead of estimating
        let meta_bytes = (store_meta.coords.len() + store_meta.data_vars.len()) as u64 * 500;
        s.record_metadata(meta_bytes, meta_start.elapsed());
    }

    let coord_names: Vec<_> = store_meta.coords.iter().map(|c| c.name.clone()).collect();
    let coord_types: Vec<_> = store_meta
        .coords
        .iter()
        .map(|c| c.data_type.clone())
        .collect();

    // Load coordinate arrays and get their sizes
    let mut coord_sizes: Vec<usize> = Vec::new();
    let mut coord_values: Vec<CoordValues> = Vec::new();

    for (coord, dtype) in store_meta.coords.iter().zip(coord_types.iter()) {
        let read_start = Instant::now();
        let arr = Array::open(store.clone(), &format!("/{}", coord.name)).map_err(zarr_err)?;
        let size = arr.shape()[0] as usize;
        coord_sizes.push(size);

        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        let element_bytes = dtype_to_bytes(dtype);
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

        if let Some(ref s) = stats {
            let bytes = size as u64 * element_bytes;
            s.record_coord(bytes, read_start.elapsed());
        }
        coord_values.push(values);
    }

    // Total rows = product of all coordinate sizes
    let total_rows: usize = coord_sizes.iter().product();

    let total_columns = schema.fields().len();
    let projected_indices = projection.unwrap_or_else(|| (0..total_columns).collect());

    // Log projection optimization effect
    let skipped_columns = total_columns - projected_indices.len();
    if skipped_columns > 0 {
        let projected_names: Vec<_> = projected_indices
            .iter()
            .map(|&i| schema.field(i).name().as_str())
            .collect();
        info!(
            reading = projected_indices.len(),
            skipping = skipped_columns,
            columns = ?projected_names,
            "Projection optimization"
        );
    } else {
        info!(columns = total_columns, "No projection optimization (all columns)");
    }

    // Log limit info (sync path applies limit via slicing at the end)
    if let Some(limit) = limit {
        let effective = limit.min(total_rows);
        let reduction_pct = 100.0 * (1.0 - (effective as f64 / total_rows as f64));
        info!(
            total_rows,
            effective_rows = effective,
            reduction_pct = format!("{:.2}%", reduction_pct),
            "Limit will be applied via slicing"
        );
    }

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
            let read_start = Instant::now();
            let arr = Array::open(store.clone(), &format!("/{}", field_name)).map_err(zarr_err)?;
            let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
            let num_elements: u64 = arr.shape().iter().product();

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

            if let Some(ref s) = stats {
                let bytes = num_elements * arrow_dtype_to_bytes(field.data_type());
                s.record_data(bytes, read_start.elapsed());
            }
            result_arrays.push(array);
        }
    }

    let projected_schema = Arc::new(Schema::new(
        projected_indices
            .iter()
            .map(|&i| schema.field(i).clone())
            .collect::<Vec<_>>(),
    ));

    // Apply limit if specified
    let result_arrays = if let Some(limit) = limit {
        let limit = limit.min(total_rows);
        result_arrays
            .into_iter()
            .map(|arr| arr.slice(0, limit))
            .collect()
    } else {
        result_arrays
    };

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

impl CoordValues {
    fn len(&self) -> usize {
        match self {
            CoordValues::Int64(v) => v.len(),
            CoordValues::Float32(v) => v.len(),
            CoordValues::Float64(v) => v.len(),
        }
    }
}

/// Calculate the subset ranges needed for a limited number of rows
///
/// For row-major (C) order, the last dimension varies fastest.
/// Given shape [a, b, c, d] and limit N, we need to figure out
/// which ranges to read to get exactly N elements.
fn calculate_limited_subset(shape: &[u64], limit: usize) -> Vec<std::ops::Range<u64>> {
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
            let slices_needed = (limit + inner_size - 1) / inner_size; // ceil
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
fn calculate_coord_limits(coord_sizes: &[usize], limit: usize) -> Vec<usize> {
    let mut limits = Vec::with_capacity(coord_sizes.len());
    let n = coord_sizes.len();

    for i in 0..n {
        // Product of all coords after this one
        let inner_size: usize = coord_sizes[i + 1..].iter().product();
        let inner_size = if inner_size == 0 { 1 } else { inner_size };

        // How many values do we need from this coord?
        let needed = (limit + inner_size - 1) / inner_size; // ceil
        let take = needed.min(coord_sizes[i]);
        limits.push(take);
    }

    limits
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

// =============================================================================
// Async version for remote object stores
// =============================================================================

use super::schema_inference::{discover_arrays_async, ZarrStoreMeta};
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

/// Async version of read_zarr for remote object stores
#[instrument(level = "info", skip_all)]
pub async fn read_zarr_async(
    store: AsyncReadableListableStorage,
    prefix: &ObjectPath,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: Option<SharedIoStats>,
    cached_meta: Option<ZarrStoreMeta>,
) -> Result<SendableRecordBatchStream> {
    info!("Starting async Zarr read");

    // Use cached metadata if available, otherwise discover
    let store_meta = if let Some(meta) = cached_meta {
        info!("Using cached metadata");
        meta
    } else {
        debug!("Discovering store metadata");
        let meta_start = Instant::now();
        let meta = discover_arrays_async(&store, prefix)
            .await
            .map_err(DataFusionError::External)?;
        debug!(elapsed = ?meta_start.elapsed(), "Metadata discovery complete");

        if let Some(ref s) = stats {
            // TODO: Track actual metadata bytes read
            let meta_bytes = (meta.coords.len() + meta.data_vars.len()) as u64 * 500;
            s.record_metadata(meta_bytes, meta_start.elapsed());
        }
        meta
    };

    let coord_names: Vec<_> = store_meta.coords.iter().map(|c| c.name.clone()).collect();
    let coord_types: Vec<_> = store_meta
        .coords
        .iter()
        .map(|c| c.data_type.clone())
        .collect();

    // Get coordinate sizes from metadata (already discovered)
    let coord_sizes: Vec<usize> = store_meta
        .coords
        .iter()
        .map(|c| c.shape[0] as usize)
        .collect();
    debug!(?coord_names, ?coord_sizes, "Coordinate info");

    // Total rows = product of all coordinate sizes
    let total_rows: usize = coord_sizes.iter().product();

    // Apply limit early to avoid allocating memory for rows we won't use
    let effective_rows = limit.map(|l| l.min(total_rows)).unwrap_or(total_rows);

    // Log limit optimization effect
    if effective_rows < total_rows {
        let reduction_pct = 100.0 * (1.0 - (effective_rows as f64 / total_rows as f64));
        info!(
            total_rows,
            effective_rows,
            reduction_pct = format!("{:.2}%", reduction_pct),
            "Limit optimization applied"
        );
    } else {
        info!(total_rows, "No limit optimization (reading all rows)");
    }

    // Calculate how many values we need from each coordinate
    let coord_value_limits = calculate_coord_limits(&coord_sizes, effective_rows);

    // Log coordinate reduction details
    let coord_reduction: Vec<String> = coord_names
        .iter()
        .zip(coord_value_limits.iter())
        .zip(coord_sizes.iter())
        .map(|((name, &limit), &full)| format!("{}={}/{}", name, limit, full))
        .collect();
    debug!(limits = ?coord_reduction, "Coordinate value limits");

    // Load only the coordinate values we need for the limit
    debug!("Loading coordinate values");
    let mut coord_values: Vec<CoordValues> = Vec::new();

    // TODO:: This can happen in parallel for all coordinates
    for (i, (coord, dtype)) in store_meta.coords.iter().zip(coord_types.iter()).enumerate() {
        debug!(coord_name = %coord.name, "Reading coordinate");
        let read_start = Instant::now();
        let array_path = format!("/{}/{}", prefix, coord.name);
        debug!(path = %array_path, "Opening coordinate array");

        let arr = Array::async_open(store.clone(), &array_path)
            .await
            .map_err(zarr_err)?;

        let values_needed = coord_value_limits[i];
        debug!(values_needed, full_size = coord_sizes[i], "Reading coordinate subset");

        // Read only the subset of values we need for this limit
        let subset = ArraySubset::new_with_ranges(&[0..values_needed as u64]);
        let element_bytes = dtype_to_bytes(dtype);
        let values = match dtype.as_str() {
            "float32" => {
                let (vals, _) = arr
                    .async_retrieve_array_subset_ndarray::<f32>(&subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Float32(vals)
            }
            "float64" => {
                let (vals, _) = arr
                    .async_retrieve_array_subset_ndarray::<f64>(&subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Float64(vals)
            }
            _ => {
                let (vals, _) = arr
                    .async_retrieve_array_subset_ndarray::<i64>(&subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Int64(vals)
            }
        };

        debug!(elapsed = ?read_start.elapsed(), "Coordinate read complete");
        if let Some(ref s) = stats {
            let bytes = values_needed as u64 * element_bytes;
            s.record_coord(bytes, read_start.elapsed());
        }
        coord_values.push(values);
    }
    info!("All coordinates loaded");

    let total_columns = schema.fields().len();
    let projected_indices = projection.unwrap_or_else(|| (0..total_columns).collect());

    // Log projection optimization effect
    let skipped_columns = total_columns - projected_indices.len();
    if skipped_columns > 0 {
        let projected_names: Vec<_> = projected_indices
            .iter()
            .map(|&i| schema.field(i).name().as_str())
            .collect();
        info!(
            reading = projected_indices.len(),
            skipping = skipped_columns,
            columns = ?projected_names,
            "Projection optimization"
        );
    } else {
        info!(columns = total_columns, "No projection optimization (all columns)")
    }

    let mut result_arrays: Vec<ArrayRef> = Vec::new();

    for idx in &projected_indices {
        let field = schema.field(*idx);
        let field_name = field.name();

        // Check if this is a coordinate
        if let Some(coord_idx) = coord_names.iter().position(|n| n == field_name) {
            debug!(field = %field_name, "Building dictionary array for coordinate");
            // Create DictionaryArray for coordinate (memory efficient)
            // Use coord_value_limits for key generation pattern (respects limit)
            let dict_array = create_coord_dictionary_typed(
                &coord_values[coord_idx],
                coord_idx,
                &coord_value_limits,
                effective_rows,
            );
            result_arrays.push(dict_array);
        } else {
            // Data variable - read and flatten based on schema type
            debug!(field_name = %field_name, "Reading data variable");
            let read_start = Instant::now();
            let array_path = format!("/{}/{}", prefix, field_name);
            debug!(path = %array_path, "Opening data variable array");

            let arr = Array::async_open(store.clone(), &array_path)
                .await
                .map_err(zarr_err)?;
            debug!(shape = ?arr.shape(), "Data variable shape");

            // Use limited subset when limit is applied to avoid reading entire array
            let full_elements: u64 = arr.shape().iter().product();
            let subset = if effective_rows < total_rows {
                let ranges = calculate_limited_subset(arr.shape(), effective_rows);
                let limited_subset = ArraySubset::new_with_ranges(&ranges);
                let subset_elements = limited_subset.num_elements();
                let reduction_pct = 100.0 * (1.0 - (subset_elements as f64 / full_elements as f64));
                info!(
                    field = %field_name,
                    subset_elements,
                    full_elements,
                    reduction_pct = format!("{:.2}%", reduction_pct),
                    "Data subset optimization"
                );
                limited_subset
            } else {
                debug!(field = %field_name, full_elements, "Reading full array");
                ArraySubset::new_with_shape(arr.shape().to_vec())
            };
            let num_elements: u64 = subset.num_elements();

            let array: ArrayRef = match field.data_type() {
                DataType::Float32 => {
                    let (vals, _) = arr
                        .async_retrieve_array_subset_ndarray::<f32>(&subset)
                        .await
                        .map_err(zarr_err)?
                        .into_raw_vec_and_offset();
                    Arc::new(Float32Array::from(vals))
                }
                DataType::Float64 => {
                    let (vals, _) = arr
                        .async_retrieve_array_subset_ndarray::<f64>(&subset)
                        .await
                        .map_err(zarr_err)?
                        .into_raw_vec_and_offset();
                    Arc::new(Float64Array::from(vals))
                }
                _ => {
                    let (vals, _) = arr
                        .async_retrieve_array_subset_ndarray::<i64>(&subset)
                        .await
                        .map_err(zarr_err)?
                        .into_raw_vec_and_offset();
                    Arc::new(Int64Array::from(vals))
                }
            };

            debug!(elapsed = ?read_start.elapsed(), "Data variable read complete");
            if let Some(ref s) = stats {
                let bytes = num_elements * arrow_dtype_to_bytes(field.data_type());
                s.record_data(bytes, read_start.elapsed());
            }
            result_arrays.push(array);
        }
    }

    debug!("Building projected schema");
    let projected_schema = Arc::new(Schema::new(
        projected_indices
            .iter()
            .map(|&i| schema.field(i).clone())
            .collect::<Vec<_>>(),
    ));

    // Apply limit if specified
    let result_arrays = if let Some(limit) = limit {
        let limit = limit.min(total_rows);
        debug!(limit, "Applying final limit slice");
        result_arrays
            .into_iter()
            .map(|arr| arr.slice(0, limit))
            .collect()
    } else {
        result_arrays
    };

    let batch = RecordBatch::try_new(projected_schema.clone(), result_arrays)?;
    info!(
        num_rows = batch.num_rows(),
        num_columns = batch.num_columns(),
        "RecordBatch created successfully"
    );

    let stream = stream::iter(vec![Ok(batch)]);

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}
