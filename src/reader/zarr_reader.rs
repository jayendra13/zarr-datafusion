//! Zarr array reader that flattens nD data into Arrow RecordBatches
//!
//! See [`super::schema_inference`] for assumptions about Zarr store structure
//! (1D coordinates, nD data variables as Cartesian product of coordinates).

use tracing::{debug, info, instrument, warn};

use arrow::{
    array::{ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch, RecordBatchOptions},
    datatypes::{DataType, Schema, SchemaRef},
};
use datafusion::{
    common::DataFusionError, error::Result, execution::SendableRecordBatchStream,
    physical_plan::stream::RecordBatchStreamAdapter,
};
use futures::stream;
use std::sync::Arc;
use std::time::Instant;
use zarrs::{array::Array, array_subset::ArraySubset, filesystem::FilesystemStore};

use super::coord::{
    calculate_coord_limits, calculate_limited_subset, create_coord_dictionary_typed, CoordValues,
};
use super::filter::{
    calculate_coord_ranges, calculate_filtered_rows, coord_ranges_to_array_ranges, CoordFilters,
    CoordValuesRef,
};
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

// =============================================================================
// Macros for type-dispatched array reading (reduces ~90 lines of duplication)
//
// We maintain both sync and async paths because Tokio uses thread pools for
// file I/O rather than io_uring, adding ~1-5μs overhead per operation. For
// Zarr's many-chunk workloads, sync is faster for local files.
// =============================================================================

/// Macro to read coordinate array values with type dispatch.
/// Handles both sync and async variants of zarrs array retrieval.
macro_rules! read_coord_values {
    // Sync version - uses retrieve_array_subset_ndarray
    (sync, $arr:expr, $subset:expr, $dtype:expr) => {
        match $dtype {
            "float32" => {
                let (vals, _) = $arr
                    .retrieve_array_subset_ndarray::<f32>($subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Float32(vals)
            }
            "float64" => {
                let (vals, _) = $arr
                    .retrieve_array_subset_ndarray::<f64>($subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Float64(vals)
            }
            _ => {
                let (vals, _) = $arr
                    .retrieve_array_subset_ndarray::<i64>($subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Int64(vals)
            }
        }
    };
    // Async version - uses async_retrieve_array_subset_ndarray
    (async, $arr:expr, $subset:expr, $dtype:expr) => {
        match $dtype {
            "float32" => {
                let (vals, _) = $arr
                    .async_retrieve_array_subset_ndarray::<f32>($subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Float32(vals)
            }
            "float64" => {
                let (vals, _) = $arr
                    .async_retrieve_array_subset_ndarray::<f64>($subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Float64(vals)
            }
            _ => {
                let (vals, _) = $arr
                    .async_retrieve_array_subset_ndarray::<i64>($subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Int64(vals)
            }
        }
    };
}

/// Macro to read data variable array with type dispatch.
/// Returns an ArrayRef based on the Arrow DataType.
macro_rules! read_data_array {
    // Sync version
    (sync, $arr:expr, $subset:expr, $data_type:expr) => {
        match $data_type {
            DataType::Float32 => {
                let (vals, _) = $arr
                    .retrieve_array_subset_ndarray::<f32>($subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                Arc::new(Float32Array::from(vals)) as ArrayRef
            }
            DataType::Float64 => {
                let (vals, _) = $arr
                    .retrieve_array_subset_ndarray::<f64>($subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                Arc::new(Float64Array::from(vals)) as ArrayRef
            }
            _ => {
                let (vals, _) = $arr
                    .retrieve_array_subset_ndarray::<i64>($subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                Arc::new(Int64Array::from(vals)) as ArrayRef
            }
        }
    };
    // Async version
    (async, $arr:expr, $subset:expr, $data_type:expr) => {
        match $data_type {
            DataType::Float32 => {
                let (vals, _) = $arr
                    .async_retrieve_array_subset_ndarray::<f32>($subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                Arc::new(Float32Array::from(vals)) as ArrayRef
            }
            DataType::Float64 => {
                let (vals, _) = $arr
                    .async_retrieve_array_subset_ndarray::<f64>($subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                Arc::new(Float64Array::from(vals)) as ArrayRef
            }
            _ => {
                let (vals, _) = $arr
                    .async_retrieve_array_subset_ndarray::<i64>($subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                Arc::new(Int64Array::from(vals)) as ArrayRef
            }
        }
    };
}

pub fn read_zarr(
    store_path: &str,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: Option<SharedIoStats>,
    coord_filters: Option<CoordFilters>,
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
        let values = read_coord_values!(sync, arr, &subset, dtype.as_str());

        if let Some(ref s) = stats {
            let bytes = size as u64 * element_bytes;
            s.record_coord(bytes, read_start.elapsed());
        }
        coord_values.push(values);
    }

    // Total rows = product of all coordinate sizes (before filtering)
    let total_rows: usize = coord_sizes.iter().product();

    // Calculate coordinate ranges based on filters
    let coord_ranges = if let Some(ref filters) = coord_filters {
        // Convert coord_values to refs for filtering
        let coord_refs: Vec<CoordValuesRef> = coord_values
            .iter()
            .map(|v| match v {
                CoordValues::Int64(vals) => CoordValuesRef::Int64(vals),
                CoordValues::Float32(vals) => CoordValuesRef::Float32(vals),
                CoordValues::Float64(vals) => CoordValuesRef::Float64(vals),
            })
            .collect();

        match calculate_coord_ranges(filters, &coord_names, &coord_refs) {
            Some(ranges) => {
                let filtered_rows = calculate_filtered_rows(&ranges);
                let reduction_pct = 100.0 * (1.0 - (filtered_rows as f64 / total_rows as f64));
                info!(
                    total_rows,
                    filtered_rows,
                    reduction_pct = format!("{:.2}%", reduction_pct),
                    filters = ?filters.filters.keys().collect::<Vec<_>>(),
                    "Filter pushdown optimization"
                );
                Some(ranges)
            }
            None => {
                // Filter value not found - return empty result
                warn!("Filter value not found in coordinates - returning empty result");
                let projected_schema = Arc::new(Schema::new(
                    projection
                        .as_ref()
                        .map(|indices| {
                            indices
                                .iter()
                                .map(|&i| schema.field(i).as_ref().clone())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_else(|| {
                            schema.fields().iter().map(|f| f.as_ref().clone()).collect()
                        }),
                ));
                let batch = RecordBatch::new_empty(projected_schema.clone());
                let stream = stream::iter(vec![Ok(batch)]);
                return Ok(Box::pin(RecordBatchStreamAdapter::new(
                    projected_schema,
                    stream,
                )));
            }
        }
    } else {
        None
    };

    // Calculate effective sizes based on filters
    let (effective_coord_sizes, effective_rows) = if let Some(ref ranges) = coord_ranges {
        let sizes: Vec<usize> = ranges.iter().map(|(start, end)| end - start).collect();
        let rows = calculate_filtered_rows(ranges);
        (sizes, rows)
    } else {
        (coord_sizes.clone(), total_rows)
    };

    // Extract filtered coordinate values
    let filtered_coord_values: Vec<CoordValues> = if let Some(ref ranges) = coord_ranges {
        coord_values
            .iter()
            .zip(ranges.iter())
            .map(|(values, (start, end))| match values {
                CoordValues::Int64(vals) => CoordValues::Int64(vals[*start..*end].to_vec()),
                CoordValues::Float32(vals) => CoordValues::Float32(vals[*start..*end].to_vec()),
                CoordValues::Float64(vals) => CoordValues::Float64(vals[*start..*end].to_vec()),
            })
            .collect()
    } else {
        coord_values
    };

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
        info!(
            columns = total_columns,
            "No projection optimization (all columns)"
        );
    }

    // Apply limit (after filter reduction)
    let final_rows = limit
        .map(|l| l.min(effective_rows))
        .unwrap_or(effective_rows);
    if let Some(limit) = limit {
        if limit < effective_rows {
            let reduction_pct = 100.0 * (1.0 - (final_rows as f64 / effective_rows as f64));
            info!(
                effective_rows,
                final_rows,
                reduction_pct = format!("{:.2}%", reduction_pct),
                "Limit optimization"
            );
        }
    }

    let mut result_arrays: Vec<ArrayRef> = Vec::new();

    for idx in &projected_indices {
        let field = schema.field(*idx);
        let field_name = field.name();

        // Check if this is a coordinate
        if let Some(coord_idx) = coord_names.iter().position(|n| n == field_name) {
            // Create DictionaryArray for coordinate (memory efficient)
            let dict_array = create_coord_dictionary_typed(
                &filtered_coord_values[coord_idx],
                coord_idx,
                &effective_coord_sizes,
                effective_rows,
            );
            result_arrays.push(dict_array);
        } else {
            // Data variable - read filtered subset
            let read_start = Instant::now();
            let arr = Array::open(store.clone(), &format!("/{}", field_name)).map_err(zarr_err)?;

            // Calculate the subset to read based on coordinate filters
            let subset = if let Some(ref ranges) = coord_ranges {
                let array_ranges = coord_ranges_to_array_ranges(ranges);
                ArraySubset::new_with_ranges(&array_ranges)
            } else {
                ArraySubset::new_with_shape(arr.shape().to_vec())
            };
            let num_elements: u64 = subset.num_elements();

            let array: ArrayRef = read_data_array!(sync, arr, &subset, field.data_type());

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

    // Apply limit if specified (slice the already-filtered arrays)
    let result_arrays = if let Some(limit) = limit {
        let limit = limit.min(effective_rows);
        result_arrays
            .into_iter()
            .map(|arr| arr.slice(0, limit))
            .collect()
    } else {
        result_arrays
    };

    // Handle empty projection (e.g., count(*)) - need to set row count explicitly
    let batch = if result_arrays.is_empty() {
        info!(final_rows, "Empty projection - returning row count only");
        RecordBatch::try_new_with_options(
            projected_schema.clone(),
            result_arrays,
            &RecordBatchOptions::new().with_row_count(Some(final_rows)),
        )?
    } else {
        RecordBatch::try_new(projected_schema.clone(), result_arrays)?
    };
    let stream = stream::iter(vec![Ok(batch)]);

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

// =============================================================================
// Async version for remote object stores
// =============================================================================

use super::schema_inference::{discover_arrays_async, ZarrStoreMeta};
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

/// Async version of read_zarr for remote object stores
#[allow(clippy::too_many_arguments)]
#[instrument(level = "info", skip_all)]
pub async fn read_zarr_async(
    store: AsyncReadableListableStorage,
    prefix: &ObjectPath,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: Option<SharedIoStats>,
    cached_meta: Option<ZarrStoreMeta>,
    coord_filters: Option<CoordFilters>,
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

    // Total rows = product of all coordinate sizes (before filtering)
    let total_rows: usize = coord_sizes.iter().product();

    // First, load all coordinate values (needed for filter matching)
    debug!("Loading coordinate values for filter matching");
    let mut all_coord_values: Vec<CoordValues> = Vec::new();

    for (coord, dtype) in store_meta.coords.iter().zip(coord_types.iter()) {
        let read_start = Instant::now();
        let array_path = format!("/{}/{}", prefix, coord.name);

        let arr = Array::async_open(store.clone(), &array_path)
            .await
            .map_err(zarr_err)?;

        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        let element_bytes = dtype_to_bytes(dtype);
        let values = read_coord_values!(async, arr, &subset, dtype.as_str());

        if let Some(ref s) = stats {
            let bytes = coord.shape[0] * element_bytes;
            s.record_coord(bytes, read_start.elapsed());
        }
        all_coord_values.push(values);
    }

    // Calculate coordinate ranges based on filters
    let coord_ranges = if let Some(ref filters) = coord_filters {
        let coord_refs: Vec<CoordValuesRef> = all_coord_values
            .iter()
            .map(|v| match v {
                CoordValues::Int64(vals) => CoordValuesRef::Int64(vals),
                CoordValues::Float32(vals) => CoordValuesRef::Float32(vals),
                CoordValues::Float64(vals) => CoordValuesRef::Float64(vals),
            })
            .collect();

        match calculate_coord_ranges(filters, &coord_names, &coord_refs) {
            Some(ranges) => {
                let filtered_rows = calculate_filtered_rows(&ranges);
                let reduction_pct = 100.0 * (1.0 - (filtered_rows as f64 / total_rows as f64));
                info!(
                    total_rows,
                    filtered_rows,
                    reduction_pct = format!("{:.2}%", reduction_pct),
                    filters = ?filters.filters.keys().collect::<Vec<_>>(),
                    "Filter pushdown optimization"
                );
                Some(ranges)
            }
            None => {
                // Filter value not found - return empty result
                warn!("Filter value not found in coordinates - returning empty result");
                let projected_schema = Arc::new(Schema::new(
                    projection
                        .as_ref()
                        .map(|indices| {
                            indices
                                .iter()
                                .map(|&i| schema.field(i).as_ref().clone())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_else(|| {
                            schema.fields().iter().map(|f| f.as_ref().clone()).collect()
                        }),
                ));
                let batch = RecordBatch::new_empty(projected_schema.clone());
                let stream = stream::iter(vec![Ok(batch)]);
                return Ok(Box::pin(RecordBatchStreamAdapter::new(
                    projected_schema,
                    stream,
                )));
            }
        }
    } else {
        None
    };
    debug!(?coord_ranges, "Coordinate ranges calculated");

    // Calculate effective sizes based on filters
    let (effective_coord_sizes, rows_after_filter) = if let Some(ref ranges) = coord_ranges {
        let sizes: Vec<usize> = ranges.iter().map(|(start, end)| end - start).collect();
        let rows = calculate_filtered_rows(ranges);
        (sizes, rows)
    } else {
        (coord_sizes.clone(), total_rows)
    };

    // Extract filtered coordinate values
    let filtered_coord_values: Vec<CoordValues> = if let Some(ref ranges) = coord_ranges {
        all_coord_values
            .iter()
            .zip(ranges.iter())
            .map(|(values, (start, end))| match values {
                CoordValues::Int64(vals) => CoordValues::Int64(vals[*start..*end].to_vec()),
                CoordValues::Float32(vals) => CoordValues::Float32(vals[*start..*end].to_vec()),
                CoordValues::Float64(vals) => CoordValues::Float64(vals[*start..*end].to_vec()),
            })
            .collect()
    } else {
        all_coord_values
    };

    // Apply limit (after filter reduction)
    let effective_rows = limit
        .map(|l| l.min(rows_after_filter))
        .unwrap_or(rows_after_filter);

    // Log limit optimization effect
    if effective_rows < rows_after_filter {
        let reduction_pct = 100.0 * (1.0 - (effective_rows as f64 / rows_after_filter as f64));
        info!(
            rows_after_filter,
            effective_rows,
            reduction_pct = format!("{:.2}%", reduction_pct),
            "Limit optimization applied"
        );
    }

    // Calculate how many values we need from each coordinate (for limit optimization on top of filter)
    let coord_value_limits = if effective_rows < rows_after_filter {
        calculate_coord_limits(&effective_coord_sizes, effective_rows)
    } else {
        effective_coord_sizes.clone()
    };

    info!("Coordinates loaded and filtered");

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
        info!(
            columns = total_columns,
            "No projection optimization (all columns)"
        )
    }

    let mut result_arrays: Vec<ArrayRef> = Vec::new();

    for idx in &projected_indices {
        let field = schema.field(*idx);
        let field_name = field.name();

        // Check if this is a coordinate
        if let Some(coord_idx) = coord_names.iter().position(|n| n == field_name) {
            debug!(field = %field_name, "Building dictionary array for coordinate");
            // Create DictionaryArray for coordinate (memory efficient)
            let dict_array = create_coord_dictionary_typed(
                &filtered_coord_values[coord_idx],
                coord_idx,
                &coord_value_limits,
                effective_rows,
            );
            result_arrays.push(dict_array);
        } else {
            // Data variable - read filtered subset
            debug!(field_name = %field_name, "Reading data variable");
            let read_start = Instant::now();
            let array_path = format!("/{}/{}", prefix, field_name);
            debug!(path = %array_path, "Opening data variable array");

            let arr = Array::async_open(store.clone(), &array_path)
                .await
                .map_err(zarr_err)?;
            debug!(shape = ?arr.shape(), "Data variable shape");

            // Calculate the subset to read based on coordinate filters
            let full_elements: u64 = arr.shape().iter().product();
            let subset = if let Some(ref ranges) = coord_ranges {
                let array_ranges = coord_ranges_to_array_ranges(ranges);
                let filtered_subset = ArraySubset::new_with_ranges(&array_ranges);
                let subset_elements = filtered_subset.num_elements();
                let reduction_pct = 100.0 * (1.0 - (subset_elements as f64 / full_elements as f64));
                info!(
                    field = %field_name,
                    subset_elements,
                    full_elements,
                    reduction_pct = format!("{:.2}%", reduction_pct),
                    "Filter-based data subset optimization"
                );
                filtered_subset
            } else if effective_rows < total_rows {
                let ranges = calculate_limited_subset(arr.shape(), effective_rows);
                let limited_subset = ArraySubset::new_with_ranges(&ranges);
                let subset_elements = limited_subset.num_elements();
                let reduction_pct = 100.0 * (1.0 - (subset_elements as f64 / full_elements as f64));
                info!(
                    field = %field_name,
                    subset_elements,
                    full_elements,
                    reduction_pct = format!("{:.2}%", reduction_pct),
                    "Limit-based data subset optimization"
                );
                limited_subset
            } else {
                debug!(field = %field_name, full_elements, "Reading full array");
                ArraySubset::new_with_shape(arr.shape().to_vec())
            };
            let num_elements: u64 = subset.num_elements();

            let array: ArrayRef = read_data_array!(async, arr, &subset, field.data_type());

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

    // Apply final limit slice if needed
    let final_rows = limit
        .map(|l| l.min(rows_after_filter))
        .unwrap_or(rows_after_filter);
    let result_arrays = if let Some(limit) = limit {
        let limit = limit.min(rows_after_filter);
        debug!(limit, "Applying final limit slice");
        result_arrays
            .into_iter()
            .map(|arr| arr.slice(0, limit))
            .collect()
    } else {
        result_arrays
    };

    // Handle empty projection (e.g., count(*)) - need to set row count explicitly
    let batch = if result_arrays.is_empty() {
        info!(final_rows, "Empty projection - returning row count only");
        RecordBatch::try_new_with_options(
            projected_schema.clone(),
            result_arrays,
            &RecordBatchOptions::new().with_row_count(Some(final_rows)),
        )?
    } else {
        RecordBatch::try_new(projected_schema.clone(), result_arrays)?
    };
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
