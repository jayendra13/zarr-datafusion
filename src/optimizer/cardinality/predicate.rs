//! Bridge from the scan's pushed-down filters into the cardinality module's
//! [`IndexSet`] world. Phase 2 lowers a query's already-resolved coordinate
//! selections into a [`ProductSet`] so exact cardinality can be observed at plan
//! time; it changes no execution behaviour.
//!
//! [`IndexSet`]: super::IndexSet
//! [`ProductSet`]: super::ProductSet

use std::collections::HashMap;
use std::fmt;

use arrow::datatypes::Schema;

use crate::reader::filter::{calculate_coord_ranges, CoordFilters, CoordSelection, CoordValuesRef};
use crate::reader::schema_inference::ZarrArrayMeta;

use super::{AxisSet, CubeShape, IndexSet, ProductSet};

/// One axis's resolved selection becomes one [`AxisSet`].
///
/// A contiguous `Range(s, e)` is an interval; scattered `Indices` (e.g. all
/// December timestamps, or an irregular date-part match) become an explicit index
/// set — which stays exact on irregular calendars where no clean stride exists.
impl From<&CoordSelection> for AxisSet {
    fn from(sel: &CoordSelection) -> Self {
        match sel {
            CoordSelection::Range(s, e) => AxisSet::interval(*s as u64, *e as u64),
            CoordSelection::Indices(v) => AxisSet::indices(v.iter().map(|&i| i as u64).collect()),
        }
    }
}

/// Assemble a single-box [`ProductSet`] from per-axis resolved selections.
///
/// Axis order follows `selections`, i.e. the scan's coordinate order (axis 0 is the
/// outer/streaming axis). Each selection becomes one [`AxisSet`] via the `From`
/// bridge; an empty selection makes the whole set empty (cardinality 0).
///
/// The scan produces `Indices` sorted and de-duplicated, so this set's cardinality
/// equals the scan's `calculate_filtered_rows` (product of selection lengths).
/// `AxisSet::indices` de-dups defensively, so even a hypothetical duplicate index
/// yields the true distinct-point count.
pub fn product_from_selections(selections: &[CoordSelection]) -> ProductSet {
    ProductSet::single(selections.iter().map(AxisSet::from).collect())
}

/// Lower a query's coordinate filters into a [`ProductSet`] over the cube's index
/// space — the Phase-2 observation entry point.
///
/// Resolves each coordinate's AND-composed filters against its values (reusing the
/// scan's [`calculate_coord_ranges`]) and maps the result. Returns `None` exactly
/// when the query is provably empty (some filter matches no value).
pub fn selection_from_filters(
    filters: &CoordFilters,
    coord_names: &[String],
    coord_values: &[CoordValuesRef<'_>],
) -> Option<ProductSet> {
    let selections = calculate_coord_ranges(filters, coord_names, coord_values)?;
    Some(product_from_selections(&selections))
}

// --- Phase 2.4: derived shape quantities ---------------------------------------

/// Exact plan-time shape statistics for a scan's coordinate selection — the
/// numbers the diagnostic reports and Phase 3's cost model will build on. All are
/// computed, not estimated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SelectionStats {
    /// Total surviving rows = the selection's exact cardinality.
    pub rows: u128,
    /// Rows forced resident by a single outer-axis step: the product of the
    /// *non-outer* axis sizes. This is the floor the current streaming scan cannot
    /// tile below (design-note Gap 1).
    pub inner_rows: u128,
    /// Distinct chunks the selection touches (`touched_tiles(chunk_shape)`) — the
    /// exact I/O-cost driver.
    pub touched_chunks: u128,
}

/// Compute [`SelectionStats`] for `selections` over `shape`. Axis order and arity
/// of `shape` and `selections` must match (both in the scan's coordinate order,
/// axis 0 = outer/streaming axis).
pub fn selection_stats(shape: &CubeShape, selections: &[CoordSelection]) -> SelectionStats {
    debug_assert_eq!(
        shape.ndim(),
        selections.len(),
        "shape/selection arity mismatch"
    );
    let set = product_from_selections(selections);
    let inner_rows = selections.iter().skip(1).map(|s| s.len() as u128).product();
    SelectionStats {
        rows: set.cardinality(),
        inner_rows,
        touched_chunks: set.touched_tiles(&shape.chunk_shape()),
    }
}

// --- Phase 2.5: mixed-dimensionality (Gap 2) detection -------------------------

/// True iff the projection hits the mixed-dimensionality single-batch fallback
/// (design-note Gap 2): some projected data variable does not span the full
/// effective coordinate set, so the scan cannot window it and emits one batch
/// (correct, but not memory-bounded).
///
/// Mirrors `all_full_cube` in `zarr_reader`: a projected column is "full cube" if
/// it is a coordinate, or a data var whose dimensionality (`shape.len()`) equals
/// `effective_ndim`.
pub fn hits_single_batch_fallback(
    projected: &[String],
    coord_names: &[String],
    data_var_ndims: &HashMap<String, usize>,
    effective_ndim: usize,
) -> bool {
    !projected.iter().all(|name| {
        coord_names.contains(name)
            || data_var_ndims
                .get(name)
                .is_some_and(|&d| d == effective_ndim)
    })
}

// --- Phase 2.7: observe-only scan diagnostics ----------------------------------

/// A bundle of the exact plan-time facts about a scan: its selection statistics,
/// its dimensionality, and whether it hits the single-batch fallback. Purely
/// observational — produced for logging/`EXPLAIN`, drives no execution decision in
/// Phase 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanDiagnostics {
    pub stats: SelectionStats,
    pub ndim: usize,
    pub single_batch_fallback: bool,
}

impl fmt::Display for ScanDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "rows={} inner_rows={} touched_chunks={} ndim={} single_batch_fallback={}",
            self.stats.rows,
            self.stats.inner_rows,
            self.stats.touched_chunks,
            self.ndim,
            self.single_batch_fallback,
        )
    }
}

/// Build [`ScanDiagnostics`] from the scan's raw artifacts, doing the
/// effective-coordinate subsetting internally so call sites stay one line.
///
/// `effective_coord_indices` maps effective (outer-first) position -> index into
/// `coords` / `coord_sizes` (store coordinate order). `coord_ranges`, when present,
/// is the scan's per-store-coord resolved selections; when absent, every effective
/// coordinate is taken as its full range.
#[allow(clippy::too_many_arguments)]
pub fn scan_diagnostics(
    coords: &[ZarrArrayMeta],
    effective_coord_indices: &[usize],
    coord_ranges: Option<&[CoordSelection]>,
    coord_sizes: &[usize],
    projected_names: &[String],
    coord_names: &[String],
    data_vars: &[ZarrArrayMeta],
) -> ScanDiagnostics {
    let effective_ndim = effective_coord_indices.len();

    let eff_metas: Vec<ZarrArrayMeta> = effective_coord_indices
        .iter()
        .map(|&i| coords[i].clone())
        .collect();
    let shape = CubeShape::from_coord_metas(&eff_metas);

    let eff_selections: Vec<CoordSelection> = effective_coord_indices
        .iter()
        .map(|&i| match coord_ranges {
            Some(sels) => sels[i].clone(),
            None => CoordSelection::Range(0, coord_sizes[i]),
        })
        .collect();

    let stats = selection_stats(&shape, &eff_selections);

    let data_var_ndims: HashMap<String, usize> = data_vars
        .iter()
        .map(|v| (v.name.clone(), v.shape.len()))
        .collect();
    let single_batch_fallback = hits_single_batch_fallback(
        projected_names,
        coord_names,
        &data_var_ndims,
        effective_ndim,
    );

    ScanDiagnostics {
        stats,
        ndim: effective_ndim,
        single_batch_fallback,
    }
}

/// Observe-only scan hook (Phases 2 + 3): at debug level, log the exact
/// cardinality diagnostics; and, when a memory budget is configured
/// (`ZARR_MEM_BUDGET_BYTES`), warn if the predicted peak footprint exceeds it.
/// Drives no execution decision — a single call the scan makes at both read sites.
///
/// `schema` is the full store schema (for resolving projected field names);
/// `projected_schema` is the projected batch schema (for `row_width`). Does nothing
/// — not even the diagnostics computation — when debug logging is off and no budget
/// is set.
#[allow(clippy::too_many_arguments)]
pub fn observe_scan(
    coords: &[ZarrArrayMeta],
    effective_coord_indices: &[usize],
    coord_ranges: Option<&[CoordSelection]>,
    coord_sizes: &[usize],
    schema: &Schema,
    projected_indices: &[usize],
    coord_names: &[String],
    data_vars: &[ZarrArrayMeta],
    projected_schema: &Schema,
    batch_size: usize,
) {
    let debug_on = tracing::enabled!(tracing::Level::DEBUG);
    let budget = super::budget::MemoryBudget::from_env();
    if !debug_on && budget.is_none() {
        return;
    }

    let projected_names: Vec<String> = projected_indices
        .iter()
        .map(|&i| schema.field(i).name().to_string())
        .collect();
    let diag = scan_diagnostics(
        coords,
        effective_coord_indices,
        coord_ranges,
        coord_sizes,
        &projected_names,
        coord_names,
        data_vars,
    );
    if debug_on {
        tracing::debug!(%diag, "cardinality diagnostics (observe-only)");
    }

    if let Some(budget) = budget {
        let row_width = super::cost::row_width(projected_schema);
        let peak = super::cost::peak_bytes(
            diag.stats.rows,
            diag.stats.inner_rows,
            batch_size,
            diag.single_batch_fallback,
            row_width,
        );
        // Peak-based admission needs no I/O accounting; leave bytes_read at 0.
        let cost = super::cost::ScanCost {
            rows: diag.stats.rows,
            touched_chunks: diag.stats.touched_chunks,
            bytes_read: 0,
            peak_bytes: peak,
        };
        if let Err(infeasible) = super::budget::admit(&cost, &budget) {
            tracing::warn!(%infeasible, "scan predicted to exceed memory budget (advisory; not enforced)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_becomes_interval() {
        let sel = CoordSelection::Range(2, 7);
        assert_eq!(AxisSet::from(&sel), AxisSet::interval(2, 7));
        assert_eq!(AxisSet::from(&sel).len(), 5);
    }

    #[test]
    fn indices_become_index_set() {
        let sel = CoordSelection::Indices(vec![9, 1, 4, 1]); // unsorted, dup
                                                             // `AxisSet::indices` sorts + dedups.
        assert_eq!(AxisSet::from(&sel), AxisSet::indices(vec![1, 4, 9]));
    }

    #[test]
    fn cardinality_matches_calculate_filtered_rows() {
        use crate::optimizer::cardinality::IndexSet;
        use crate::reader::filter::calculate_filtered_rows;
        let sels = vec![
            CoordSelection::Range(0, 100),
            CoordSelection::Indices(vec![3, 5, 9, 12]),
            CoordSelection::Range(10, 20),
        ];
        let ps = product_from_selections(&sels);
        // Same number the scan sizes its batch to: 100 * 4 * 10.
        assert_eq!(ps.cardinality(), calculate_filtered_rows(&sels) as u128);
        assert_eq!(ps.cardinality(), 4000);
    }

    #[test]
    fn empty_axis_gives_empty_set() {
        use crate::optimizer::cardinality::IndexSet;
        let sels = vec![CoordSelection::Range(5, 5), CoordSelection::Range(0, 3)];
        let ps = product_from_selections(&sels);
        assert!(ps.is_empty());
        assert_eq!(ps.cardinality(), 0);
    }

    fn coord_meta(name: &str, extent: u64, chunk: u64) -> ZarrArrayMeta {
        ZarrArrayMeta {
            name: name.into(),
            data_type: "float64".into(),
            shape: vec![extent],
            chunks: Some(vec![chunk]),
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: None,
        }
    }

    fn data_var_meta(name: &str, shape: Vec<u64>) -> ZarrArrayMeta {
        ZarrArrayMeta {
            name: name.into(),
            data_type: "float64".into(),
            shape,
            chunks: None,
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: None,
        }
    }

    #[test]
    fn selection_stats_rows_inner_and_chunks() {
        // time: 0..100 selected, chunk 24 -> tiles {0,1,2,3,4} = 5
        // lat:  {2,4,6} (indices), chunk 16 -> tile 0 only = 1
        // lon:  10..20 selected, chunk 4  -> tiles {2,3,4} = 3
        let shape = CubeShape::new(vec![
            crate::optimizer::cardinality::Axis::new("time", 100, 24),
            crate::optimizer::cardinality::Axis::new("lat", 50, 16),
            crate::optimizer::cardinality::Axis::new("lon", 60, 4),
        ]);
        let sels = vec![
            CoordSelection::Range(0, 100),
            CoordSelection::Indices(vec![2, 4, 6]),
            CoordSelection::Range(10, 20),
        ];
        let stats = selection_stats(&shape, &sels);
        assert_eq!(stats.rows, 100 * 3 * 10); // 3000
        assert_eq!(stats.inner_rows, 3 * 10); // non-outer axes: 30
        assert_eq!(stats.touched_chunks, 15); // time=5 * lat=1 * lon=3
    }

    #[test]
    fn single_batch_fallback_detects_mixed_dim() {
        let coord_names = vec!["time".to_string(), "lat".to_string(), "lon".to_string()];
        let ndims: HashMap<String, usize> = [("temp".to_string(), 3), ("elevation".to_string(), 2)]
            .into_iter()
            .collect();

        // Full-cube projection: a 3-D var + a coordinate -> streams (no fallback).
        assert!(!hits_single_batch_fallback(
            &["temp".to_string(), "lat".to_string()],
            &coord_names,
            &ndims,
            3,
        ));
        // Mixed dim: a 2-D static field next to the effective 3-D set -> fallback.
        assert!(hits_single_batch_fallback(
            &["temp".to_string(), "elevation".to_string()],
            &coord_names,
            &ndims,
            3,
        ));
    }

    #[test]
    fn scan_diagnostics_end_to_end() {
        let coords = vec![
            coord_meta("time", 100, 24),
            coord_meta("lat", 50, 16),
            coord_meta("lon", 60, 4),
        ];
        let coord_names = vec!["time".to_string(), "lat".to_string(), "lon".to_string()];
        let coord_sizes = vec![100usize, 50, 60];
        let data_vars = vec![
            data_var_meta("temp", vec![100, 50, 60]),
            data_var_meta("elevation", vec![50, 60]),
        ];
        // All three coords effective, in order; explicit selections.
        let sels = vec![
            CoordSelection::Range(0, 100),
            CoordSelection::Indices(vec![2, 4, 6]),
            CoordSelection::Range(10, 20),
        ];
        let diag = scan_diagnostics(
            &coords,
            &[0, 1, 2],
            Some(&sels),
            &coord_sizes,
            &["temp".to_string()],
            &coord_names,
            &data_vars,
        );
        assert_eq!(diag.ndim, 3);
        assert_eq!(diag.stats.rows, 3000);
        assert_eq!(diag.stats.inner_rows, 30);
        assert_eq!(diag.stats.touched_chunks, 15);
        assert!(!diag.single_batch_fallback); // temp is full-cube

        // Projecting the 2-D static field flips the fallback flag.
        let diag2 = scan_diagnostics(
            &coords,
            &[0, 1, 2],
            Some(&sels),
            &coord_sizes,
            &["elevation".to_string()],
            &coord_names,
            &data_vars,
        );
        assert!(diag2.single_batch_fallback);
    }
}
