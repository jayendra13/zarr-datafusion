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

use super::cf_time::apply_cf_time_conversion;
use super::coord::{
    calculate_coord_limits, calculate_limited_subset, create_coord_dictionary_typed, CoordValues,
};
use super::filter::{
    calculate_coord_ranges, calculate_filtered_rows, determine_effective_coords,
    match_ranges_to_data_var, CoordFilters, CoordSelection, CoordValuesRef,
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
            "int32" => {
                let (vals, _) = $arr
                    .retrieve_array_subset_ndarray::<i32>($subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                // Convert to i64 for uniform handling
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            "uint32" => {
                let (vals, _) = $arr
                    .retrieve_array_subset_ndarray::<u32>($subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            "int16" => {
                let (vals, _) = $arr
                    .retrieve_array_subset_ndarray::<i16>($subset)
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
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
            "int32" => {
                let (vals, _) = $arr
                    .async_retrieve_array_subset_ndarray::<i32>($subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                // Convert to i64 for uniform handling
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            "uint32" => {
                let (vals, _) = $arr
                    .async_retrieve_array_subset_ndarray::<u32>($subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
            }
            "int16" => {
                let (vals, _) = $arr
                    .async_retrieve_array_subset_ndarray::<i16>($subset)
                    .await
                    .map_err(zarr_err)?
                    .into_raw_vec_and_offset();
                CoordValues::Int64(vals.into_iter().map(|v| v as i64).collect())
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

// =============================================================================
// Helper functions to reduce sync/async duplication
// =============================================================================

/// Create an empty result stream when filter values are not found.
///
/// This consolidates the identical logic used in both sync and async paths when
/// a filter value doesn't match any coordinate values, requiring an empty result.
fn create_empty_result_stream(
    schema: &SchemaRef,
    projection: Option<&Vec<usize>>,
) -> Result<SendableRecordBatchStream> {
    let projected_schema = Arc::new(Schema::new(
        projection
            .map(|indices| {
                indices
                    .iter()
                    .map(|&i| schema.field(i).as_ref().clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| schema.fields().iter().map(|f| f.as_ref().clone()).collect()),
    ));
    let batch = RecordBatch::new_empty(projected_schema.clone());
    let stream = stream::iter(vec![Ok(batch)]);
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

/// Build projected schema, excluding coordinates that were skipped due to
/// mixed-dimensionality optimization.
///
/// This consolidates the identical schema building logic used in both sync and async paths.
fn build_projected_schema(
    schema: &SchemaRef,
    projected_indices: &[usize],
    coord_names: &[String],
    effective_coord_indices: &[usize],
) -> SchemaRef {
    let projected_fields: Vec<_> = projected_indices
        .iter()
        .filter_map(|&i| {
            let field = schema.field(i);
            let field_name = field.name();
            // Check if this is a coordinate that was skipped
            if let Some(coord_idx) = coord_names.iter().position(|n| n == field_name) {
                if !effective_coord_indices.contains(&coord_idx) {
                    return None; // Skip coordinates not in effective set
                }
            }
            Some(field.clone())
        })
        .collect();
    Arc::new(Schema::new(projected_fields))
}

/// Convert CoordValues to CoordValuesRef for filtering.
///
/// This consolidates the repeated conversion logic in both sync and async paths.
fn coord_values_to_refs(coord_values: &[CoordValues]) -> Vec<CoordValuesRef<'_>> {
    coord_values
        .iter()
        .map(|v| match v {
            CoordValues::Int64(vals) => CoordValuesRef::Int64(vals),
            CoordValues::Float32(vals) => CoordValuesRef::Float32(vals),
            CoordValues::Float64(vals) => CoordValuesRef::Float64(vals),
            CoordValues::TimestampMicros(vals) => CoordValuesRef::TimestampMicros(vals),
            CoordValues::Compact {
                encoding,
                is_timestamp,
            } => CoordValuesRef::Compact {
                encoding: *encoding,
                is_timestamp: *is_timestamp,
            },
        })
        .collect()
}

/// Calculate effective sizes and filtered rows based on coordinate selections.
///
/// Returns (effective_coord_sizes, effective_rows).
fn calculate_effective_sizes(
    coord_sizes: &[usize],
    selections: &Option<Vec<CoordSelection>>,
) -> (Vec<usize>, usize) {
    if let Some(ref sels) = selections {
        let sizes: Vec<usize> = sels.iter().map(|s| s.len()).collect();
        let rows = calculate_filtered_rows(sels);
        (sizes, rows)
    } else {
        (coord_sizes.to_vec(), coord_sizes.iter().product())
    }
}

/// Extract filtered coordinate values based on selections.
///
/// `Range` selections slice contiguously; `Indices` selections gather scattered positions.
fn extract_filtered_coords(
    coord_values: Vec<CoordValues>,
    selections: &Option<Vec<CoordSelection>>,
) -> Vec<CoordValues> {
    if let Some(ref sels) = selections {
        coord_values
            .iter()
            .zip(sels.iter())
            .map(|(values, sel)| match sel {
                CoordSelection::Range(start, end) => values.slice(*start, *end),
                CoordSelection::Indices(indices) => values.gather(indices),
            })
            .collect()
    } else {
        coord_values
    }
}

/// Expand coordinate selections into `ArraySubset`s for zarr chunk reading.
///
/// All-Range selections produce a single subset (one read, existing behavior).
/// A selection containing `Indices` produces one subset per index — each a
/// single-step Range in that dimension — so scattered chunks are read individually
/// and their results concatenated by the caller.
fn build_read_subsets(
    selections: &[CoordSelection],
    coord_sizes: &[usize],
    data_var_shape: &[u64],
) -> Vec<ArraySubset> {
    let indices_pos = selections
        .iter()
        .position(|s| matches!(s, CoordSelection::Indices(_)));

    match indices_pos {
        None => {
            let array_ranges = match_ranges_to_data_var(coord_sizes, selections, data_var_shape)
                .unwrap_or_else(|| data_var_shape.iter().map(|&s| 0..s).collect());
            vec![ArraySubset::new_with_ranges(&array_ranges)]
        }
        Some(pos) => {
            let CoordSelection::Indices(ref indices) = selections[pos] else {
                unreachable!()
            };
            indices
                .iter()
                .map(|&idx| {
                    let expanded: Vec<CoordSelection> = selections
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            if i == pos {
                                CoordSelection::Range(idx, idx + 1)
                            } else {
                                s.clone()
                            }
                        })
                        .collect();
                    let array_ranges =
                        match_ranges_to_data_var(coord_sizes, &expanded, data_var_shape)
                            .unwrap_or_else(|| data_var_shape.iter().map(|&s| 0..s).collect());
                    ArraySubset::new_with_ranges(&array_ranges)
                })
                .collect()
        }
    }
}

/// Recalculate query coordinate sizes and rows for effective coordinates only.
///
/// This handles mixed-dimensionality optimization where only a subset of coordinates
/// are needed for the projected columns.
fn calculate_query_coord_sizes(
    coord_sizes: &[usize],
    effective_coord_sizes: &[usize],
    effective_coord_indices: &[usize],
    effective_rows: usize,
) -> (Vec<usize>, usize) {
    if effective_coord_indices.len() < coord_sizes.len() {
        let sizes: Vec<usize> = effective_coord_indices
            .iter()
            .map(|&i| effective_coord_sizes[i])
            .collect();
        let rows: usize = sizes.iter().product();
        info!(
            all_coords = coord_sizes.len(),
            effective_coords = effective_coord_indices.len(),
            query_rows = rows,
            original_rows = effective_rows,
            "Using reduced coordinate set for variable dimensionality"
        );
        (sizes, rows)
    } else {
        (effective_coord_sizes.to_vec(), effective_rows)
    }
}

/// Create the final result batch, handling empty projections (e.g., COUNT(*)).
fn create_result_batch(
    projected_schema: SchemaRef,
    result_arrays: Vec<ArrayRef>,
    final_rows: usize,
) -> Result<RecordBatch> {
    if result_arrays.is_empty() {
        info!(final_rows, "Empty projection - returning row count only");
        Ok(RecordBatch::try_new_with_options(
            projected_schema,
            result_arrays,
            &RecordBatchOptions::new().with_row_count(Some(final_rows)),
        )?)
    } else {
        Ok(RecordBatch::try_new(projected_schema, result_arrays)?)
    }
}

/// Apply limit to result arrays by slicing them.
fn apply_limit_to_arrays(
    result_arrays: Vec<ArrayRef>,
    limit: Option<usize>,
    max_rows: usize,
) -> Vec<ArrayRef> {
    if let Some(limit) = limit {
        let limit = limit.min(max_rows);
        result_arrays
            .into_iter()
            .map(|arr| arr.slice(0, limit))
            .collect()
    } else {
        result_arrays
    }
}

/// Intersect a partition's outer-dimension slice into the per-coordinate
/// selections that the rest of `read_zarr` consumes (effective sizes, filtered
/// coord values, and data-var subsets all derive from these).
///
/// - `coord_ranges = None`  => no filters yet; start from full-range selections
///   so we can still restrict the outer coordinate.
/// - `coord_ranges = Some`  => existing per-coordinate selections (one per coord).
/// - `outer_coord_idx`      => index (in coord order) of the coord to restrict.
/// - `partition_range`      => half-open `[start, end)` index slice on that coord.
/// - `coord_sizes`          => full length of each coordinate (to build defaults).
fn restrict_to_partition(
    coord_ranges: Option<Vec<CoordSelection>>,
    outer_coord_idx: usize,
    partition_range: std::ops::Range<u64>,
    coord_sizes: &[usize],
) -> Option<Vec<CoordSelection>> {
    // FILL 5b: existing selections, or one full-range selection per coordinate.
    let mut sels: Vec<CoordSelection> = coord_ranges.unwrap_or_else(|| {
        coord_sizes
            .iter()
            .map(|&size| CoordSelection::Range(0, size))
            .collect()
    });

    let p_start = partition_range.start as usize;
    let p_end = partition_range.end as usize;

    // FILL 5c: intersect the outer coordinate's selection with [p_start, p_end).
    let restricted = match &sels[outer_coord_idx] {
        // Contiguous slice: clamp both ends into the partition window. When the
        // filter's selection and this partition's window don't overlap, `start`
        // would exceed `end`; clamp to an EMPTY range `[start, start)` so this
        // partition simply yields no rows (other partitions cover those rows).
        CoordSelection::Range(s, e) => {
            let start = (*s).max(p_start);
            let end = (*e).min(p_end).max(start);
            CoordSelection::Range(start, end)
        }
        // Scattered positions: keep only those that fall in the window.
        CoordSelection::Indices(v) => CoordSelection::Indices(
            v.iter()
                .copied()
                .filter(|&i| i >= p_start && i < p_end)
                .collect(),
        ),
    };
    sels[outer_coord_idx] = restricted;

    Some(sels)
}

/// Narrow `coord_ranges` to a partition's outer-dimension slice, if any.
///
/// Maps the data var's axis-0 to its coordinate by size (coords are ordered to
/// match the axes, so the outer coordinate is the first whose size == data var
/// `shape[0]`) and intersects the slice in via [`restrict_to_partition`]. Shared
/// by the sync ([`read_zarr`]) and async ([`read_zarr_async`]) readers.
fn apply_partition_range(
    coord_ranges: Option<Vec<CoordSelection>>,
    partition_range: Option<std::ops::Range<u64>>,
    store_meta: &super::schema_inference::ZarrStoreMeta,
    coord_sizes: &[usize],
) -> Option<Vec<CoordSelection>> {
    let Some(range) = partition_range else {
        return coord_ranges;
    };
    let outer_axis_len = store_meta
        .data_vars
        .first()
        .and_then(|dv| dv.shape.first())
        .copied();
    let outer_coord_idx =
        outer_axis_len.and_then(|len| coord_sizes.iter().position(|&s| s as u64 == len));
    match outer_coord_idx {
        Some(idx) => restrict_to_partition(coord_ranges, idx, range, coord_sizes),
        // Can't identify the outer coordinate => can't safely slice. scan() only
        // partitions when the dim is known, so this shouldn't happen; stay
        // defensive and fall back to no slicing.
        None => coord_ranges,
    }
}

pub fn read_zarr(
    store_path: &str,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: Option<SharedIoStats>,
    coord_filters: Option<CoordFilters>,
    // Outer-dimension index slice for this partition (half-open `[start, end)`).
    // `None` => read the whole (filtered) store, the legacy single-partition path.
    // (Plain `//` not `///`: doc comments aren't allowed on fn parameters.)
    partition_range: Option<std::ops::Range<u64>>,
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
        let raw_values = read_coord_values!(sync, arr, &subset, dtype.as_str());

        // Apply CF time conversion using helper function
        let values =
            apply_cf_time_conversion(raw_values, coord.cf_time_attrs.as_ref(), &coord.name);

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
        let coord_refs = coord_values_to_refs(&coord_values);

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
                return create_empty_result_stream(&schema, projection.as_ref());
            }
        }
    } else {
        None
    };

    // Narrow the read to this partition's outer-dimension slice, if any.
    let coord_ranges =
        apply_partition_range(coord_ranges, partition_range, &store_meta, &coord_sizes);

    // Calculate effective sizes based on filters
    let (effective_coord_sizes, effective_rows) =
        calculate_effective_sizes(&coord_sizes, &coord_ranges);

    // Extract filtered coordinate values
    let filtered_coord_values = extract_filtered_coords(coord_values, &coord_ranges);

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

    // ==========================================================================
    // Mixed-dimensionality handling: determine which coordinates are actually
    // needed for the projected columns (data variables or coordinates only)
    // ==========================================================================
    // Separate projected columns into coordinates and data variables
    let (projected_coord_names, projected_var_names): (Vec<&str>, Vec<&str>) = projected_indices
        .iter()
        .map(|&i| schema.field(i).name().as_str())
        .partition(|name| coord_names.contains(&name.to_string()));

    // Determine effective coordinates for the projected columns
    // Pass limit to enable coordinate-only optimization when LIMIT is present
    let effective_coord_indices = determine_effective_coords(
        &projected_var_names,
        &projected_coord_names,
        &store_meta.data_vars,
        &coord_names,
        &coord_sizes,
        limit,
    )
    .map_err(DataFusionError::Plan)?;

    // Recalculate effective sizes and rows for the relevant coordinates only
    let (query_coord_sizes, query_rows) = calculate_query_coord_sizes(
        &coord_sizes,
        &effective_coord_sizes,
        &effective_coord_indices,
        effective_rows,
    );

    // Apply limit (after filter and dimensionality reduction)
    let final_rows = limit.map(|l| l.min(query_rows)).unwrap_or(query_rows);
    if let Some(limit) = limit {
        if limit < query_rows {
            let reduction_pct = 100.0 * (1.0 - (final_rows as f64 / query_rows as f64));
            info!(
                query_rows,
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
            // Skip coordinates not relevant to the projected variables
            if !effective_coord_indices.contains(&coord_idx) {
                debug!(field = %field_name, "Skipping coordinate not used by projected variables");
                continue;
            }

            // Find position of this coordinate in the effective set
            let query_coord_idx = effective_coord_indices
                .iter()
                .position(|&i| i == coord_idx)
                .unwrap();

            // Create DictionaryArray for coordinate (memory efficient)
            let dict_array = create_coord_dictionary_typed(
                &filtered_coord_values[coord_idx],
                query_coord_idx,
                &query_coord_sizes,
                final_rows,
            );
            result_arrays.push(dict_array);
        } else {
            // Data variable - read filtered subset
            let read_start = Instant::now();
            let arr = Array::open(store.clone(), &format!("/{}", field_name)).map_err(zarr_err)?;
            let data_var_shape = arr.shape();

            // Build subsets: one for all-Range filters, N for Indices (scattered chunks)
            let subsets = if let Some(ref sels) = coord_ranges {
                build_read_subsets(sels, &coord_sizes, data_var_shape)
            } else {
                vec![ArraySubset::new_with_shape(arr.shape().to_vec())]
            };
            let num_elements: u64 = subsets.iter().map(|s| s.num_elements()).sum();

            let mut parts: Vec<ArrayRef> = Vec::with_capacity(subsets.len());
            for subset in &subsets {
                parts.push(read_data_array!(sync, arr, subset, field.data_type()));
            }
            let array: ArrayRef = if parts.len() == 1 {
                parts.remove(0)
            } else {
                let refs: Vec<&dyn arrow::array::Array> =
                    parts.iter().map(|a| a.as_ref()).collect();
                arrow::compute::concat(&refs).map_err(DataFusionError::from)?
            };

            if let Some(ref s) = stats {
                let bytes = num_elements * arrow_dtype_to_bytes(field.data_type());
                s.record_data(bytes, read_start.elapsed());
            }
            result_arrays.push(array);
        }
    }

    // Build projected schema, excluding coordinates that were skipped
    let projected_schema = build_projected_schema(
        &schema,
        &projected_indices,
        &coord_names,
        &effective_coord_indices,
    );

    // Apply limit if specified (slice the already-filtered arrays)
    let result_arrays = apply_limit_to_arrays(result_arrays, limit, query_rows);

    // Create result batch and wrap in stream
    let batch = create_result_batch(projected_schema.clone(), result_arrays, final_rows)?;
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
    // Outer-dimension index slice for this partition; `None` => whole store.
    partition_range: Option<std::ops::Range<u64>>,
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
        let array_path = if prefix.as_ref().is_empty() {
            format!("/{}", coord.name)
        } else {
            format!("/{}/{}", prefix, coord.name)
        };

        let arr = Array::async_open(store.clone(), &array_path)
            .await
            .map_err(zarr_err)?;

        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        let element_bytes = dtype_to_bytes(dtype);
        let raw_values = read_coord_values!(async, arr, &subset, dtype.as_str());

        // Apply CF time conversion using helper function
        let values =
            apply_cf_time_conversion(raw_values, coord.cf_time_attrs.as_ref(), &coord.name);

        debug!(path = %array_path, "Coordinate values loaded");
        if let Some(ref s) = stats {
            let bytes = coord.shape[0] * element_bytes;
            s.record_coord(bytes, read_start.elapsed());
        }
        all_coord_values.push(values);
    }

    // Calculate coordinate ranges based on filters
    let coord_ranges = if let Some(ref filters) = coord_filters {
        let coord_refs = coord_values_to_refs(&all_coord_values);

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
                return create_empty_result_stream(&schema, projection.as_ref());
            }
        }
    } else {
        None
    };
    debug!(?coord_ranges, "Coordinate ranges calculated");

    // Narrow the read to this partition's outer-dimension slice, if any.
    let coord_ranges =
        apply_partition_range(coord_ranges, partition_range, &store_meta, &coord_sizes);

    // Calculate effective sizes based on filters
    let (effective_coord_sizes, rows_after_filter) =
        calculate_effective_sizes(&coord_sizes, &coord_ranges);

    // Extract filtered coordinate values
    let filtered_coord_values = extract_filtered_coords(all_coord_values, &coord_ranges);

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
    // TODO: Use this for optimized coordinate loading in the future
    let _coord_value_limits = if effective_rows < rows_after_filter {
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

    // ==========================================================================
    // Mixed-dimensionality handling: determine which coordinates are actually
    // needed for the projected columns (data variables or coordinates only)
    // ==========================================================================
    // Separate projected columns into coordinates and data variables
    let (projected_coord_names, projected_var_names): (Vec<&str>, Vec<&str>) = projected_indices
        .iter()
        .map(|&i| schema.field(i).name().as_str())
        .partition(|name| coord_names.contains(&name.to_string()));

    // Determine effective coordinates for the projected columns
    // Pass limit to enable coordinate-only optimization when LIMIT is present
    let effective_coord_indices = determine_effective_coords(
        &projected_var_names,
        &projected_coord_names,
        &store_meta.data_vars,
        &coord_names,
        &coord_sizes,
        limit,
    )
    .map_err(DataFusionError::Plan)?;

    // Recalculate effective sizes and rows for the relevant coordinates only
    let (query_coord_sizes, query_rows) = calculate_query_coord_sizes(
        &coord_sizes,
        &effective_coord_sizes,
        &effective_coord_indices,
        rows_after_filter,
    );

    // Recalculate effective_rows with limit applied to query_rows
    let effective_rows = limit.map(|l| l.min(query_rows)).unwrap_or(query_rows);

    let mut result_arrays: Vec<ArrayRef> = Vec::new();

    for idx in &projected_indices {
        let field = schema.field(*idx);
        let field_name = field.name();

        // Check if this is a coordinate
        if let Some(coord_idx) = coord_names.iter().position(|n| n == field_name) {
            // Skip coordinates not relevant to the projected variables
            if !effective_coord_indices.contains(&coord_idx) {
                debug!(field = %field_name, "Skipping coordinate not used by projected variables");
                continue;
            }

            // Find position of this coordinate in the effective set
            let query_coord_idx = effective_coord_indices
                .iter()
                .position(|&i| i == coord_idx)
                .unwrap();

            debug!(field = %field_name, coord_idx, query_coord_idx, "Building dictionary array for coordinate");
            // Create DictionaryArray for coordinate (memory efficient)
            // Use query_coord_idx for position within the effective coordinate set
            let dict_array = create_coord_dictionary_typed(
                &filtered_coord_values[coord_idx],
                query_coord_idx,
                &query_coord_sizes,
                effective_rows,
            );
            result_arrays.push(dict_array);
        } else {
            // Data variable - read filtered subset
            debug!(field_name = %field_name, "Reading data variable");
            let read_start = Instant::now();
            let array_path = if prefix.as_ref().is_empty() {
                format!("/{}", field_name)
            } else {
                format!("/{}/{}", prefix, field_name)
            };
            debug!(path = %array_path, "Opening data variable array");

            let arr = Array::async_open(store.clone(), &array_path)
                .await
                .map_err(zarr_err)?;
            debug!(shape = ?arr.shape(), "Data variable shape");

            // Build subsets: one for all-Range filters, N for Indices (scattered chunks)
            let full_elements: u64 = arr.shape().iter().product();
            let data_var_shape = arr.shape();
            let subsets = if let Some(ref sels) = coord_ranges {
                let subsets = build_read_subsets(sels, &coord_sizes, data_var_shape);
                let subset_elements: u64 = subsets.iter().map(|s| s.num_elements()).sum();
                let reduction_pct = 100.0 * (1.0 - (subset_elements as f64 / full_elements as f64));
                info!(
                    field = %field_name,
                    subset_elements,
                    full_elements,
                    num_subsets = subsets.len(),
                    reduction_pct = format!("{:.2}%", reduction_pct),
                    "Filter-based data subset optimization"
                );
                subsets
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
                vec![limited_subset]
            } else {
                debug!(field = %field_name, full_elements, "Reading full array");
                vec![ArraySubset::new_with_shape(arr.shape().to_vec())]
            };
            let num_elements: u64 = subsets.iter().map(|s| s.num_elements()).sum();

            let mut parts: Vec<ArrayRef> = Vec::with_capacity(subsets.len());
            for subset in &subsets {
                parts.push(read_data_array!(async, arr, subset, field.data_type()));
            }
            let array: ArrayRef = if parts.len() == 1 {
                parts.remove(0)
            } else {
                let refs: Vec<&dyn arrow::array::Array> =
                    parts.iter().map(|a| a.as_ref()).collect();
                arrow::compute::concat(&refs).map_err(DataFusionError::from)?
            };

            debug!(elapsed = ?read_start.elapsed(), "Data variable read complete");
            if let Some(ref s) = stats {
                let bytes = num_elements * arrow_dtype_to_bytes(field.data_type());
                s.record_data(bytes, read_start.elapsed());
            }
            result_arrays.push(array);
        }
    }

    // Build projected schema, excluding coordinates that were skipped
    debug!("Building projected schema");
    let projected_schema = build_projected_schema(
        &schema,
        &projected_indices,
        &coord_names,
        &effective_coord_indices,
    );

    // Apply final limit slice if needed (use query_rows, not rows_after_filter)
    let final_rows = limit.map(|l| l.min(query_rows)).unwrap_or(query_rows);
    let result_arrays = apply_limit_to_arrays(result_arrays, limit, query_rows);
    if limit.is_some() {
        debug!(final_rows, "Applied final limit slice");
    }

    // Create result batch and wrap in stream
    let batch = create_result_batch(projected_schema.clone(), result_arrays, final_rows)?;
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
