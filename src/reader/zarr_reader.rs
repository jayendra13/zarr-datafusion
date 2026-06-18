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
    match_ranges_to_data_var, resolve_coord_selection, CoordFilters, CoordSelection,
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
    partition_sel: &CoordSelection,
    coord_sizes: &[usize],
) -> Option<Vec<CoordSelection>> {
    // FILL 5b: existing selections, or one full-range selection per coordinate.
    let mut sels: Vec<CoordSelection> = coord_ranges.unwrap_or_else(|| {
        coord_sizes
            .iter()
            .map(|&size| CoordSelection::Range(0, size))
            .collect()
    });

    // REPLACE the outer coordinate's selection with this partition's selection.
    // The partition selection is the head-resolved surviving sub-set for this
    // partition (a subset of the full filter result), so it is authoritative —
    // there is a single resolution site (the head), and the worker's own
    // re-resolution of the outer filter, if any, is discarded here. For the
    // unfiltered geometry case the partition selection is the partition's axis
    // slice, which is likewise exactly what this partition should read.
    sels[outer_coord_idx] = partition_sel.clone();

    Some(sels)
}

/// Narrow `coord_ranges` to a partition's outer-dimension slice, if any.
///
/// Maps the data var's axis-0 to its coordinate by size (coords are ordered to
/// match the axes, so the outer coordinate is the first whose size == data var
/// `shape[0]`) and applies the partition selection via [`restrict_to_partition`].
/// Shared by the sync ([`read_zarr`]) and async ([`read_zarr_async`]) readers.
fn apply_partition_selection(
    coord_ranges: Option<Vec<CoordSelection>>,
    partition_selection: Option<CoordSelection>,
    store_meta: &super::schema_inference::ZarrStoreMeta,
    coord_sizes: &[usize],
) -> Option<Vec<CoordSelection>> {
    let Some(sel) = partition_selection else {
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
        Some(idx) => restrict_to_partition(coord_ranges, idx, &sel, coord_sizes),
        // Can't identify the outer coordinate => can't safely slice. scan() only
        // partitions when the dim is known, so this shouldn't happen; stay
        // defensive and fall back to no slicing.
        None => coord_ranges,
    }
}

/// Identify the OUTER coordinate (axis-0 of the first data var) by size-match.
/// Returns its index into `store_meta.coords`, or `None` if no coord matches.
fn identify_outer_coord(store_meta: &ZarrStoreMeta) -> Option<usize> {
    let outer_axis_len = store_meta
        .data_vars
        .first()
        .and_then(|dv| dv.shape.first())
        .copied()?;
    store_meta
        .coords
        .iter()
        .position(|c| c.shape.first().copied() == Some(outer_axis_len))
}

/// Plan a worker-side read of *only* this partition's slice of the outer coord.
///
/// The head ships the surviving outer-axis selection, so a worker never needs the
/// full (~10 MB) outer coordinate: it reads one contiguous window and gathers its
/// slice from that. Returns `(read_start, read_end, extract_sel)` where
/// `[read_start, read_end)` is the half-open window to fetch on the outer axis and
/// `extract_sel` is the selection *relative to that window* used to pull out the
/// surviving values.
///
/// - `Range(s, e)` => read exactly `[s, e)`; extract is the identity `Range(0, e-s)`.
/// - `Indices(v)`  => read the bounding window `[min, max+1)` in one request, then
///   gather with offsets shifted into the window. One bounding read is never worse
///   than the old full-axis read and is far cheaper when the indices cluster;
///   reading each scattered index separately would be thousands of tiny requests.
///
/// An empty selection reads nothing (`read_start == read_end`).
fn plan_outer_coord_read(sel: &CoordSelection) -> (u64, u64, CoordSelection) {
    match sel {
        CoordSelection::Range(s, e) => (
            *s as u64,
            *e as u64,
            CoordSelection::Range(0, e.saturating_sub(*s)),
        ),
        CoordSelection::Indices(v) => {
            let (Some(&lo), Some(&hi)) = (v.iter().min(), v.iter().max()) else {
                return (0, 0, CoordSelection::Indices(Vec::new()));
            };
            let rel: Vec<usize> = v.iter().map(|&i| i - lo).collect();
            (lo as u64, (hi + 1) as u64, CoordSelection::Indices(rel))
        }
    }
}

/// A view of `filters` with the outer coord's filter removed, when that coord's
/// selection is supplied by the partition (and so resolved on the head).
///
/// Re-resolving the outer filter on the worker would be both wasteful and unsound:
/// the worker only holds a partial slice of the outer coord, so `calculate_coord_ranges`
/// could mis-resolve (or wrongly return "no match") against it. Its selection is
/// REPLACED by [`apply_partition_selection`] regardless, so we drop it here.
/// Returns a borrow when there is nothing to strip (the common path).
fn filters_without_outer<'a>(
    filters: &'a CoordFilters,
    outer_idx: Option<usize>,
    coord_names: &[String],
) -> std::borrow::Cow<'a, CoordFilters> {
    match outer_idx {
        Some(i) if filters.filters.contains_key(&coord_names[i]) => {
            let mut f = filters.clone();
            f.filters.remove(&coord_names[i]);
            std::borrow::Cow::Owned(f)
        }
        _ => std::borrow::Cow::Borrowed(filters),
    }
}

/// Build the selections used to extract coordinate *values* from what was read.
///
/// Identical to `coord_ranges` (the absolute, data-var-read selections) except at
/// the outer coord, where we only read a window and so must extract with the
/// window-relative selection. Returns `coord_ranges` unchanged when the outer
/// coord was not sliced (no partitioning, or unidentifiable outer coord).
fn make_extract_selections(
    coord_ranges: &Option<Vec<CoordSelection>>,
    outer_idx: Option<usize>,
    outer_extract_sel: &Option<CoordSelection>,
) -> Option<Vec<CoordSelection>> {
    match (coord_ranges, outer_idx, outer_extract_sel) {
        (Some(sels), Some(oi), Some(esel)) => {
            let mut e = sels.clone();
            e[oi] = esel.clone();
            Some(e)
        }
        _ => coord_ranges.clone(),
    }
}

/// Outcome of resolving the outer-axis filter on the head, used to decide how to
/// partition the scan (see `split_selection`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OuterSelection {
    /// The outer filter resolved to this selection; partition by splitting it.
    Resolved(CoordSelection),
    /// A *present* outer filter matched no values — the scan yields nothing.
    Empty,
    /// No identifiable outer coordinate, or no filter on it. Fall back to
    /// whole-axis geometry partitioning; NO coordinate is read in this case.
    Unfiltered,
}

/// Read ONLY the outer coordinate from a LOCAL store and resolve its filter to a
/// selection on the outer axis. Run on the head at plan time so partitions can be
/// drawn around the *surviving* set instead of the full axis.
///
/// Reads nothing when there is no outer filter (returns `Unfiltered`), so
/// unfiltered scans keep paying zero plan-time I/O.
pub fn resolve_outer_selection(
    store_path: &str,
    store_meta: &ZarrStoreMeta,
    coord_filters: Option<&CoordFilters>,
) -> Result<OuterSelection> {
    let Some(idx) = identify_outer_coord(store_meta) else {
        return Ok(OuterSelection::Unfiltered);
    };
    let coord = &store_meta.coords[idx];
    let Some(filter) = coord_filters.and_then(|f| f.get(&coord.name)) else {
        return Ok(OuterSelection::Unfiltered);
    };

    // Read just this one coordinate's values (the whole axis).
    let store = Arc::new(FilesystemStore::new(store_path).map_err(zarr_err)?);
    let arr = Array::open(store, &format!("/{}", coord.name)).map_err(zarr_err)?;
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let raw = read_coord_values!(sync, arr, &subset, coord.data_type.as_str());
    let values = apply_cf_time_conversion(raw, coord.cf_time_attrs.as_ref(), &coord.name);

    let refs = coord_values_to_refs(std::slice::from_ref(&values));
    Ok(
        match resolve_coord_selection(&coord.name, Some(filter), &refs[0]) {
            Some(sel) => OuterSelection::Resolved(sel),
            None => OuterSelection::Empty,
        },
    )
}

/// Async (remote-store) variant of [`resolve_outer_selection`]. Reads only the
/// outer coordinate over the network when an outer filter is present.
pub async fn resolve_outer_selection_async(
    store: AsyncReadableListableStorage,
    prefix: &ObjectPath,
    store_meta: &ZarrStoreMeta,
    coord_filters: Option<&CoordFilters>,
) -> Result<OuterSelection> {
    let Some(idx) = identify_outer_coord(store_meta) else {
        return Ok(OuterSelection::Unfiltered);
    };
    let coord = &store_meta.coords[idx];
    let Some(filter) = coord_filters.and_then(|f| f.get(&coord.name)) else {
        return Ok(OuterSelection::Unfiltered);
    };

    let array_path = if prefix.as_ref().is_empty() {
        format!("/{}", coord.name)
    } else {
        format!("/{}/{}", prefix, coord.name)
    };
    let arr = Array::async_open(store, &array_path)
        .await
        .map_err(zarr_err)?;
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let raw = read_coord_values!(async, arr, &subset, coord.data_type.as_str());
    let values = apply_cf_time_conversion(raw, coord.cf_time_attrs.as_ref(), &coord.name);

    let refs = coord_values_to_refs(std::slice::from_ref(&values));
    Ok(
        match resolve_coord_selection(&coord.name, Some(filter), &refs[0]) {
            Some(sel) => OuterSelection::Resolved(sel),
            None => OuterSelection::Empty,
        },
    )
}

pub fn read_zarr(
    store_path: &str,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: Option<SharedIoStats>,
    coord_filters: Option<CoordFilters>,
    // Outer-axis selection for this partition (intersected with the filter
    // selection). `None` => read the whole (filtered) store, the legacy
    // single-partition path. (Plain `//`: doc comments aren't allowed on params.)
    partition_selection: Option<CoordSelection>,
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

    // When this partition supplies the outer-axis selection, we read ONLY that
    // slice of the outer coord (not the whole axis). `None` => no partitioning,
    // or the outer coord can't be identified => read everything as before.
    let outer_slice_idx = partition_selection
        .as_ref()
        .and_then(|_| identify_outer_coord(&store_meta));
    // The selection (relative to the slice we read) used to extract the outer
    // coord's surviving values; set while reading the outer coord below.
    let mut outer_extract_sel: Option<CoordSelection> = None;

    // Load coordinate arrays and get their sizes
    let mut coord_sizes: Vec<usize> = Vec::new();
    let mut coord_values: Vec<CoordValues> = Vec::new();

    for (i, (coord, dtype)) in store_meta.coords.iter().zip(coord_types.iter()).enumerate() {
        let read_start = Instant::now();
        let arr = Array::open(store.clone(), &format!("/{}", coord.name)).map_err(zarr_err)?;
        let size = arr.shape()[0] as usize;
        coord_sizes.push(size);

        let element_bytes = dtype_to_bytes(dtype);
        let (raw_values, read_len) = if Some(i) == outer_slice_idx {
            // Read only this partition's window of the outer coord.
            let sel = partition_selection
                .as_ref()
                .expect("outer_slice_idx implies Some");
            let (rs, re, extract_sel) = plan_outer_coord_read(sel);
            outer_extract_sel = Some(extract_sel);
            let subset = ArraySubset::new_with_ranges(std::slice::from_ref(&(rs..re)));
            (
                read_coord_values!(sync, arr, &subset, dtype.as_str()),
                (re - rs) as usize,
            )
        } else {
            let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
            (read_coord_values!(sync, arr, &subset, dtype.as_str()), size)
        };

        // Apply CF time conversion using helper function
        let values =
            apply_cf_time_conversion(raw_values, coord.cf_time_attrs.as_ref(), &coord.name);

        if let Some(ref s) = stats {
            let bytes = read_len as u64 * element_bytes;
            s.record_coord(bytes, read_start.elapsed());
        }
        coord_values.push(values);
    }

    // Total rows = product of all coordinate sizes (before filtering)
    let total_rows: usize = coord_sizes.iter().product();

    // Calculate coordinate ranges based on filters
    let coord_ranges = if let Some(ref filters) = coord_filters {
        let coord_refs = coord_values_to_refs(&coord_values);
        // Skip re-resolving the outer filter when this partition supplied its
        // (head-resolved) selection — we only hold a slice of that coord here.
        let filters = filters_without_outer(filters, outer_slice_idx, &coord_names);

        match calculate_coord_ranges(&filters, &coord_names, &coord_refs) {
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

    // Narrow the read to this partition's outer-dimension slice, if any. These
    // selections carry ABSOLUTE outer-axis indices (they drive the data-var
    // reads), so keep them distinct from the relative `extract_selections` below.
    let coord_ranges =
        apply_partition_selection(coord_ranges, partition_selection, &store_meta, &coord_sizes);

    // Calculate effective sizes based on filters
    let (effective_coord_sizes, effective_rows) =
        calculate_effective_sizes(&coord_sizes, &coord_ranges);

    // Extract filtered coordinate values. The outer coord was read as a partial
    // window, so it is extracted with a window-relative selection rather than the
    // absolute one used for the data-var reads.
    let extract_selections =
        make_extract_selections(&coord_ranges, outer_slice_idx, &outer_extract_sel);
    let filtered_coord_values = extract_filtered_coords(coord_values, &extract_selections);

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
    // Outer-axis selection for this partition; `None` => whole store.
    partition_selection: Option<CoordSelection>,
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

    // When this partition supplies the outer-axis selection, read ONLY that slice
    // of the outer coord over the network instead of the whole axis (see the sync
    // path for the rationale). `None` => no partitioning / unidentifiable outer.
    let outer_slice_idx = partition_selection
        .as_ref()
        .and_then(|_| identify_outer_coord(&store_meta));
    let mut outer_extract_sel: Option<CoordSelection> = None;

    // First, load all coordinate values (needed for filter matching)
    debug!("Loading coordinate values for filter matching");
    let mut all_coord_values: Vec<CoordValues> = Vec::new();

    for (i, (coord, dtype)) in store_meta.coords.iter().zip(coord_types.iter()).enumerate() {
        let read_start = Instant::now();
        let array_path = if prefix.as_ref().is_empty() {
            format!("/{}", coord.name)
        } else {
            format!("/{}/{}", prefix, coord.name)
        };

        let arr = Array::async_open(store.clone(), &array_path)
            .await
            .map_err(zarr_err)?;

        let element_bytes = dtype_to_bytes(dtype);
        let (raw_values, read_len) = if Some(i) == outer_slice_idx {
            let sel = partition_selection
                .as_ref()
                .expect("outer_slice_idx implies Some");
            let (rs, re, extract_sel) = plan_outer_coord_read(sel);
            outer_extract_sel = Some(extract_sel);
            let subset = ArraySubset::new_with_ranges(std::slice::from_ref(&(rs..re)));
            (
                read_coord_values!(async, arr, &subset, dtype.as_str()),
                (re - rs) as usize,
            )
        } else {
            let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
            (
                read_coord_values!(async, arr, &subset, dtype.as_str()),
                coord.shape[0] as usize,
            )
        };

        // Apply CF time conversion using helper function
        let values =
            apply_cf_time_conversion(raw_values, coord.cf_time_attrs.as_ref(), &coord.name);

        debug!(path = %array_path, "Coordinate values loaded");
        if let Some(ref s) = stats {
            let bytes = read_len as u64 * element_bytes;
            s.record_coord(bytes, read_start.elapsed());
        }
        all_coord_values.push(values);
    }

    // Calculate coordinate ranges based on filters
    let coord_ranges = if let Some(ref filters) = coord_filters {
        let coord_refs = coord_values_to_refs(&all_coord_values);
        // Skip re-resolving the outer filter when this partition supplied its
        // (head-resolved) selection — we only hold a slice of that coord here.
        let filters = filters_without_outer(filters, outer_slice_idx, &coord_names);

        match calculate_coord_ranges(&filters, &coord_names, &coord_refs) {
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

    // Narrow the read to this partition's outer-dimension slice, if any. These
    // selections carry ABSOLUTE outer-axis indices (they drive the data-var
    // reads), so keep them distinct from the relative `extract_selections` below.
    let coord_ranges =
        apply_partition_selection(coord_ranges, partition_selection, &store_meta, &coord_sizes);

    // Calculate effective sizes based on filters
    let (effective_coord_sizes, rows_after_filter) =
        calculate_effective_sizes(&coord_sizes, &coord_ranges);

    // Extract filtered coordinate values. The outer coord was read as a partial
    // window, so it is extracted with a window-relative selection rather than the
    // absolute one used for the data-var reads.
    let extract_selections =
        make_extract_selections(&coord_ranges, outer_slice_idx, &outer_extract_sel);
    let filtered_coord_values = extract_filtered_coords(all_coord_values, &extract_selections);

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

#[cfg(test)]
mod resolve_outer_tests {
    use super::*;
    use crate::reader::filter::{CoordFilterKind, CoordFilters};
    use datafusion::common::ScalarValue;

    // Synthetic v3 store: time(7), lat(10), lon(10); temperature/humidity shape
    // [7,10,10]. The outer axis (7) size-matches `time`, so it is the outer coord.
    const STORE: &str = "data/synthetic_v3.zarr";

    fn one_filter(coord: &str, value: i64) -> CoordFilters {
        let mut f = CoordFilters::new();
        f.filters.insert(
            coord.to_string(),
            CoordFilterKind::Eq(ScalarValue::Int64(Some(value))),
        );
        f
    }

    #[test]
    fn unfiltered_when_filter_is_only_on_inner_coord() {
        let meta = discover_arrays(STORE).unwrap();
        // A filter on lat (inner) leaves the outer (time) unconstrained -> no read.
        let filters = one_filter("lat", 3);
        let out = resolve_outer_selection(STORE, &meta, Some(&filters)).unwrap();
        assert_eq!(out, OuterSelection::Unfiltered);
    }

    #[test]
    fn unfiltered_when_no_filters_at_all() {
        let meta = discover_arrays(STORE).unwrap();
        let out = resolve_outer_selection(STORE, &meta, None).unwrap();
        assert_eq!(out, OuterSelection::Unfiltered);
    }

    #[test]
    fn resolves_outer_time_equality_to_range() {
        let meta = discover_arrays(STORE).unwrap();
        // time values are 0..=6; time = 3 -> the single index range [3, 4).
        let filters = one_filter("time", 3);
        let out = resolve_outer_selection(STORE, &meta, Some(&filters)).unwrap();
        assert_eq!(out, OuterSelection::Resolved(CoordSelection::Range(3, 4)));
    }

    #[test]
    fn empty_when_outer_filter_matches_nothing() {
        let meta = discover_arrays(STORE).unwrap();
        let filters = one_filter("time", 999);
        let out = resolve_outer_selection(STORE, &meta, Some(&filters)).unwrap();
        assert_eq!(out, OuterSelection::Empty);
    }
}

#[cfg(test)]
mod outer_read_tests {
    use super::*;

    // ── plan_outer_coord_read (pure window/offset math) ──────────────────────

    #[test]
    fn range_reads_exact_window_identity_extract() {
        // A contiguous slice is read verbatim; extraction is the identity range.
        let (rs, re, ext) = plan_outer_coord_read(&CoordSelection::Range(8, 16));
        assert_eq!((rs, re), (8, 16));
        assert_eq!(ext, CoordSelection::Range(0, 8));
    }

    #[test]
    fn empty_range_reads_nothing() {
        let (rs, re, ext) = plan_outer_coord_read(&CoordSelection::Range(5, 5));
        assert_eq!((rs, re), (5, 5));
        assert_eq!(ext, CoordSelection::Range(0, 0));
    }

    #[test]
    fn indices_read_bounding_window_with_shifted_offsets() {
        // Scattered indices => one bounding read [min, max+1) and offsets shifted
        // into that window, so a later gather reconstructs the exact positions.
        let (rs, re, ext) = plan_outer_coord_read(&CoordSelection::Indices(vec![3, 5, 8]));
        assert_eq!((rs, re), (3, 9));
        assert_eq!(ext, CoordSelection::Indices(vec![0, 2, 5]));
    }

    #[test]
    fn empty_indices_read_nothing() {
        let (rs, re, ext) = plan_outer_coord_read(&CoordSelection::Indices(vec![]));
        assert_eq!((rs, re), (0, 0));
        assert_eq!(ext, CoordSelection::Indices(vec![]));
    }

    // ── end-to-end: partitioned read == full read (against real data) ────────
    //
    // Synthetic v3: time(7) × lat(10) × lon(10) = 700 rows, time is the outer
    // coord. The flattening is Cartesian with time most-significant, so time `t`
    // occupies rows [t*100, (t+1)*100).

    use crate::reader::schema_inference::infer_schema;
    use arrow::compute::concat_batches;
    use arrow::record_batch::RecordBatch;
    use futures::executor::block_on;
    use futures::TryStreamExt;

    const STORE: &str = "data/synthetic_v3.zarr";

    fn read(partition: Option<CoordSelection>) -> RecordBatch {
        let schema = Arc::new(infer_schema(STORE).unwrap());
        let stream = read_zarr(STORE, schema.clone(), None, None, None, None, partition).unwrap();
        let batches: Vec<RecordBatch> = block_on(stream.try_collect()).unwrap();
        concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    #[test]
    fn full_axis_partition_matches_unpartitioned() {
        let full = read(None);
        // A single partition spanning the whole outer axis must be identical.
        let whole = read(Some(CoordSelection::Range(0, 7)));
        assert_eq!(full, whole);
    }

    #[test]
    fn complementary_range_partitions_reconstruct_full() {
        let full = read(None);
        let p0 = read(Some(CoordSelection::Range(0, 3))); // time 0..3 -> 300 rows
        let p1 = read(Some(CoordSelection::Range(3, 7))); // time 3..7 -> 400 rows
        assert_eq!(p0.num_rows(), 300);
        assert_eq!(p1.num_rows(), 400);
        let rebuilt = concat_batches(&full.schema(), &[p0, p1]).unwrap();
        assert_eq!(full, rebuilt);
    }

    #[test]
    fn scattered_indices_partition_matches_those_time_blocks() {
        let full = read(None);
        // Read only time {1, 3, 5} via the bounding-window + gather path.
        let got = read(Some(CoordSelection::Indices(vec![1, 3, 5])));
        assert_eq!(got.num_rows(), 300);
        // Expected: full rows for time 1, then 3, then 5 (100 rows each).
        let expected = concat_batches(
            &full.schema(),
            &[
                full.slice(100, 100),
                full.slice(300, 100),
                full.slice(500, 100),
            ],
        )
        .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn empty_partition_yields_no_rows() {
        assert_eq!(read(Some(CoordSelection::Range(2, 2))).num_rows(), 0);
    }
}
