//! Filter pushdown support for Zarr queries
//!
//! This module parses DataFusion filter expressions to extract coordinate
//! equality filters (e.g., `time = 1323647`), which can be used to read
//! only the relevant subset of Zarr arrays.
//!
//! For a Zarr store with coordinates [time, hybrid, lat, lon], a filter like
//! `time = X AND hybrid = Y` allows us to read only the slice of data where
//! those coordinates match, dramatically reducing memory usage.

use crate::reader::coord::CompactCoord;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{Between, Expr, Operator};
use std::collections::HashMap;
use tracing::{debug, info, trace, warn};

/// Represents a parsed coordinate filter
///
/// For a filter like `time = 1323647`, this stores:
/// - coord_name: "time"
/// - value: ScalarValue::Int64(1323647)
#[derive(Debug, Clone)]
pub struct CoordFilter {
    /// Name of the coordinate column
    pub coord_name: String,
    /// Value to match (must be equality filter)
    pub value: ScalarValue,
}

/// Kind of filter applied to a coordinate
///
/// Supports both exact equality (`coord = value`) and range filters
/// (`coord BETWEEN low AND high`, `coord >= value`, etc.)
#[derive(Debug, Clone)]
pub enum CoordFilterKind {
    /// Exact equality: coord = value
    Eq(ScalarValue),
    /// Range filter with optional bounds
    Range {
        /// Lower bound (None = unbounded)
        low: Option<ScalarValue>,
        /// Upper bound (None = unbounded)
        high: Option<ScalarValue>,
        /// Whether lower bound is inclusive (true for >=, false for >)
        low_inclusive: bool,
        /// Whether upper bound is inclusive (true for <=, false for <)
        high_inclusive: bool,
    },
}

impl std::fmt::Display for CoordFilterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoordFilterKind::Eq(val) => write!(f, "={}", val),
            CoordFilterKind::Range {
                low,
                high,
                low_inclusive,
                high_inclusive,
            } => match (low, high) {
                (Some(l), Some(h)) => write!(f, " BETWEEN {} AND {}", l, h),
                (Some(l), None) => {
                    write!(f, "{}{}", if *low_inclusive { ">=" } else { ">" }, l)
                }
                (None, Some(h)) => {
                    write!(f, "{}{}", if *high_inclusive { "<=" } else { "<" }, h)
                }
                (None, None) => write!(f, " (unbounded)"),
            },
        }
    }
}

/// Collection of coordinate filters extracted from a WHERE clause
#[derive(Debug, Clone, Default)]
pub struct CoordFilters {
    /// Map from coordinate name to filter kind (equality or range)
    pub filters: HashMap<String, CoordFilterKind>,
}

impl CoordFilters {
    pub fn new() -> Self {
        Self {
            filters: HashMap::new(),
        }
    }

    /// Check if any filters were extracted
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Get the filter for a coordinate, if any
    pub fn get(&self, coord_name: &str) -> Option<&CoordFilterKind> {
        self.filters.get(coord_name)
    }

    /// Number of coordinate filters
    pub fn len(&self) -> usize {
        self.filters.len()
    }
}

/// Check if filters can possibly be satisfied based on coordinate min/max bounds
///
/// This is an early rejection optimization that avoids loading coordinate data
/// when filters are clearly outside the coordinate's value range.
///
/// Returns `true` if filters could potentially be satisfied (no early rejection),
/// `false` if filters are definitely unsatisfiable (early rejection).
///
/// # Arguments
/// * `filters` - The parsed coordinate filters from the query
/// * `coord_meta` - Metadata for coordinate arrays, including min/max bounds
///
/// # Example
/// For a latitude coordinate with bounds (24.0, 54.75), a filter like
/// `latitude = 100.0` would return `false` (impossible).
pub fn filter_satisfiable_by_bounds(
    filters: &CoordFilters,
    coord_meta: &[super::schema_inference::ZarrArrayMeta],
) -> bool {
    for (coord_name, filter_kind) in &filters.filters {
        // Find the coordinate metadata
        let coord = match coord_meta.iter().find(|c| &c.name == coord_name) {
            Some(c) => c,
            None => continue, // Coordinate not found, skip (shouldn't happen)
        };

        // Check if coordinate has min/max bounds
        let (coord_min, coord_max) = match coord.coord_min_max {
            Some(bounds) => bounds,
            None => continue, // No bounds available, can't early-reject
        };

        // Check if filter overlaps with coordinate bounds
        let satisfiable = match filter_kind {
            CoordFilterKind::Eq(value) => {
                // Equality filter: value must be within [min, max]
                match scalar_to_f64(value) {
                    Some(v) => v >= coord_min && v <= coord_max,
                    None => true, // Can't check, assume satisfiable
                }
            }
            CoordFilterKind::Range {
                low,
                high,
                low_inclusive: _,
                high_inclusive: _,
            } => {
                // Range filter: ranges must overlap
                // Filter range [filter_low, filter_high] overlaps coord range [coord_min, coord_max]
                // if filter_high >= coord_min AND filter_low <= coord_max
                let filter_low = low.as_ref().and_then(scalar_to_f64);
                let filter_high = high.as_ref().and_then(scalar_to_f64);

                match (filter_low, filter_high) {
                    (Some(fl), Some(fh)) => fh >= coord_min && fl <= coord_max,
                    (Some(fl), None) => fl <= coord_max, // >= fl, check if fl <= max
                    (None, Some(fh)) => fh >= coord_min, // <= fh, check if fh >= min
                    (None, None) => true,                // Unbounded, always satisfiable
                }
            }
        };

        if !satisfiable {
            info!(
                coord = %coord_name,
                filter = %filter_kind,
                coord_min,
                coord_max,
                "Early rejection: filter outside coordinate bounds"
            );
            return false;
        }
    }

    true
}

/// Intermediate structure for collecting partial bounds during filter parsing
///
/// Used to accumulate separate `>=` and `<=` expressions that may combine
/// into a single range filter.
#[derive(Debug, Default)]
struct PartialBounds {
    /// Equality filter value (takes precedence over range)
    eq: Option<ScalarValue>,
    /// Lower bound with inclusivity flag (value, inclusive)
    low: Option<(ScalarValue, bool)>,
    /// Upper bound with inclusivity flag (value, inclusive)
    high: Option<(ScalarValue, bool)>,
}

impl PartialBounds {
    /// Convert partial bounds into a CoordFilterKind
    fn into_filter(self) -> Option<CoordFilterKind> {
        // Equality takes precedence
        if let Some(value) = self.eq {
            return Some(CoordFilterKind::Eq(value));
        }

        // Convert bounds to range filter if any bounds exist
        if self.low.is_some() || self.high.is_some() {
            let (low, low_inclusive) = self.low.map(|(v, i)| (Some(v), i)).unwrap_or((None, true));
            let (high, high_inclusive) =
                self.high.map(|(v, i)| (Some(v), i)).unwrap_or((None, true));
            return Some(CoordFilterKind::Range {
                low,
                high,
                low_inclusive,
                high_inclusive,
            });
        }

        None
    }
}

/// Parse DataFusion filter expressions to extract coordinate filters
///
/// Extracts the following filter types for coordinate columns:
/// - Equality: `coord = value`
/// - BETWEEN: `coord BETWEEN low AND high`
/// - Range comparisons: `coord >= value`, `coord > value`, `coord <= value`, `coord < value`
/// - Combined ranges: `coord >= low AND coord <= high`
///
/// Combined with AND:
/// - `coord1 = value1 AND coord2 BETWEEN low2 AND high2`
///
/// Other filter types (OR, LIKE, etc.) are ignored and left for
/// DataFusion to handle post-scan.
pub fn parse_coord_filters(filters: &[Expr], coord_names: &[String]) -> CoordFilters {
    // Two-pass approach:
    // 1. Collect partial bounds for each coordinate
    // 2. Merge into final CoordFilterKind values

    let mut partial_bounds: HashMap<String, PartialBounds> = HashMap::new();

    for filter in filters {
        extract_filters(filter, coord_names, &mut partial_bounds);
    }

    // Convert partial bounds to final filter kinds
    let mut result = CoordFilters::new();
    for (coord_name, bounds) in partial_bounds {
        if let Some(filter) = bounds.into_filter() {
            result.filters.insert(coord_name, filter);
        }
    }

    if !result.is_empty() {
        let filter_info: Vec<_> = result
            .filters
            .iter()
            .map(|(k, v)| format!("{}{}", k, v))
            .collect();
        info!(
            num_filters = result.len(),
            filters = ?filter_info,
            "Extracted coordinate filters for pushdown"
        );
    } else {
        debug!("No coordinate filters found for pushdown");
    }

    result
}

/// Recursively extract filters from an expression
fn extract_filters(
    expr: &Expr,
    coord_names: &[String],
    partial_bounds: &mut HashMap<String, PartialBounds>,
) {
    match expr {
        // Handle AND: recurse into both sides
        Expr::BinaryExpr(binary) if binary.op == Operator::And => {
            extract_filters(&binary.left, coord_names, partial_bounds);
            extract_filters(&binary.right, coord_names, partial_bounds);
        }

        // Handle equality: Column = Literal or Literal = Column
        Expr::BinaryExpr(binary) if binary.op == Operator::Eq => {
            if let Some((col_name, value)) = extract_column_literal_eq(&binary.left, &binary.right)
            {
                if coord_names.contains(&col_name) {
                    debug!(
                        coord = %col_name,
                        value = %value,
                        "Found coordinate equality filter"
                    );
                    partial_bounds.entry(col_name).or_default().eq = Some(value);
                } else {
                    trace!(
                        column = %col_name,
                        "Equality filter on non-coordinate column, skipping"
                    );
                }
            }
        }

        // Handle >= : Column >= Literal or Literal <= Column
        Expr::BinaryExpr(binary) if binary.op == Operator::GtEq => {
            if let Some((col_name, value)) =
                extract_column_literal_comparison(&binary.left, &binary.right, true)
            {
                if coord_names.contains(&col_name) {
                    debug!(coord = %col_name, value = %value, "Found >= filter");
                    partial_bounds.entry(col_name).or_default().low = Some((value, true));
                }
            }
        }

        // Handle > : Column > Literal or Literal < Column
        Expr::BinaryExpr(binary) if binary.op == Operator::Gt => {
            if let Some((col_name, value)) =
                extract_column_literal_comparison(&binary.left, &binary.right, true)
            {
                if coord_names.contains(&col_name) {
                    debug!(coord = %col_name, value = %value, "Found > filter");
                    partial_bounds.entry(col_name).or_default().low = Some((value, false));
                }
            }
        }

        // Handle <= : Column <= Literal or Literal >= Column
        Expr::BinaryExpr(binary) if binary.op == Operator::LtEq => {
            if let Some((col_name, value)) =
                extract_column_literal_comparison(&binary.left, &binary.right, false)
            {
                if coord_names.contains(&col_name) {
                    debug!(coord = %col_name, value = %value, "Found <= filter");
                    partial_bounds.entry(col_name).or_default().high = Some((value, true));
                }
            }
        }

        // Handle < : Column < Literal or Literal > Column
        Expr::BinaryExpr(binary) if binary.op == Operator::Lt => {
            if let Some((col_name, value)) =
                extract_column_literal_comparison(&binary.left, &binary.right, false)
            {
                if coord_names.contains(&col_name) {
                    debug!(coord = %col_name, value = %value, "Found < filter");
                    partial_bounds.entry(col_name).or_default().high = Some((value, false));
                }
            }
        }

        // Handle BETWEEN: Column BETWEEN low AND high
        Expr::Between(Between {
            expr,
            negated,
            low,
            high,
        }) if !negated => {
            if let Some(col_name) = extract_column_name(expr) {
                if coord_names.contains(&col_name) {
                    if let (Some(low_val), Some(high_val)) =
                        (extract_literal(low), extract_literal(high))
                    {
                        debug!(
                            coord = %col_name,
                            low = %low_val,
                            high = %high_val,
                            "Found BETWEEN filter"
                        );
                        let bounds = partial_bounds.entry(col_name).or_default();
                        bounds.low = Some((low_val, true));
                        bounds.high = Some((high_val, true));
                    }
                }
            }
        }

        // Handle CAST expressions that wrap the filter
        Expr::Cast(cast) => {
            extract_filters(&cast.expr, coord_names, partial_bounds);
        }

        // Other expressions: OR, LIKE, etc. - skip
        other => {
            trace!(expr_type = %other.variant_name(), "Skipping unsupported filter expression");
        }
    }
}

/// Extract column name and literal value from a comparison expression
///
/// For `is_lower_bound=true`: returns (column, value) for `Column >= Value` or `Value <= Column`
/// For `is_lower_bound=false`: returns (column, value) for `Column <= Value` or `Value >= Column`
fn extract_column_literal_comparison(
    left: &Expr,
    right: &Expr,
    is_lower_bound: bool,
) -> Option<(String, ScalarValue)> {
    // For lower bound (>=, >): Column OP Literal means column is the subject
    // For upper bound (<=, <): Column OP Literal means column is the subject
    if let (Some(col_name), Some(value)) = (extract_column_name(left), extract_literal(right)) {
        return Some((col_name, value));
    }

    // Handle reversed: Literal OP Column
    // For >= : if Literal >= Column, then Column <= Literal (upper bound)
    // For <= : if Literal <= Column, then Column >= Literal (lower bound)
    // So for reversed operands, we flip the bound type
    if let (Some(value), Some(col_name)) = (extract_literal(left), extract_column_name(right)) {
        // When reversed, the meaning flips:
        // "5 >= col" means "col <= 5" (upper bound)
        // "5 <= col" means "col >= 5" (lower bound)
        // So we only return if the flipped meaning matches what we're looking for
        if !is_lower_bound {
            // Looking for upper bound: "5 >= col" means col <= 5
            return Some((col_name, value));
        } else {
            // Looking for lower bound: "5 <= col" means col >= 5
            return Some((col_name, value));
        }
    }

    None
}

/// Extract column name and literal value from an equality expression
///
/// Returns Some((column_name, value)) for patterns like:
/// - Column = Literal
/// - Literal = Column
/// - Cast(Column) = Literal
fn extract_column_literal_eq(left: &Expr, right: &Expr) -> Option<(String, ScalarValue)> {
    // Try Column = Literal
    if let (Some(col_name), Some(value)) = (extract_column_name(left), extract_literal(right)) {
        return Some((col_name, value));
    }

    // Try Literal = Column
    if let (Some(value), Some(col_name)) = (extract_literal(left), extract_column_name(right)) {
        return Some((col_name, value));
    }

    None
}

/// Extract column name from expression, handling Cast wrappers
fn extract_column_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(col) => Some(col.name.clone()),
        Expr::Cast(cast) => extract_column_name(&cast.expr),
        Expr::TryCast(cast) => extract_column_name(&cast.expr),
        _ => None,
    }
}

/// Extract literal value from expression
///
/// Unwraps Dictionary scalar values to get the underlying value,
/// since coordinate filters compare against raw values, not dictionary indices.
fn extract_literal(expr: &Expr) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(value, _) => Some(unwrap_dictionary_value(value.clone())),
        Expr::Cast(cast) => {
            // Handle cast of literal
            if let Expr::Literal(value, _) = cast.expr.as_ref() {
                // Try to cast the value to the target type
                value
                    .cast_to(&cast.data_type)
                    .ok()
                    .map(unwrap_dictionary_value)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Unwrap Dictionary scalar values to get the inner value
///
/// DataFusion wraps literal values in Dictionary type when comparing against
/// Dictionary columns. We need the raw value for coordinate lookup.
fn unwrap_dictionary_value(value: ScalarValue) -> ScalarValue {
    match value {
        ScalarValue::Dictionary(_, inner) => unwrap_dictionary_value(*inner),
        other => other,
    }
}

/// Calculate which indices to read from each coordinate based on filters
///
/// For each coordinate:
/// - If filtered with equality (e.g., `time = X`), find the index of X
/// - If filtered with range (e.g., `time BETWEEN 2 AND 5`), find the range of matching indices
/// - If not filtered, read all values
///
/// Returns a vector of (start_idx, end_idx) ranges for each coordinate.
/// If a filter value is not found in the coordinate, returns None (no matches).
pub fn calculate_coord_ranges(
    filters: &CoordFilters,
    coord_names: &[String],
    coord_values: &[CoordValuesRef<'_>],
) -> Option<Vec<(usize, usize)>> {
    let mut ranges = Vec::with_capacity(coord_names.len());

    for (i, name) in coord_names.iter().enumerate() {
        let values = &coord_values[i];
        let range = if let Some(filter) = filters.get(name) {
            // Find the range of indices matching the filter
            if let Some((start, end)) = find_filter_range(values, filter) {
                debug!(
                    coord = %name,
                    filter = %filter,
                    start,
                    end,
                    "Found filter range"
                );
                (start, end)
            } else {
                warn!(
                    coord = %name,
                    filter = %filter,
                    "Filter did not match any values - query will return no results"
                );
                return None; // No matches possible
            }
        } else {
            // No filter on this coordinate - read all values
            (0, values.len())
        };
        ranges.push(range);
    }

    Some(ranges)
}

/// Reference to coordinate values for searching
pub enum CoordValuesRef<'a> {
    Int64(&'a [i64]),
    Float32(&'a [f32]),
    Float64(&'a [f64]),
    /// Timestamps as microseconds since Unix epoch
    /// Note: Filter pushdown for timestamps is limited - we compare raw microsecond values.
    /// Full timestamp string parsing (e.g., `time = '2020-01-01'`) is not yet supported.
    TimestampMicros(&'a [i64]),
    /// Compact encoding (arithmetic sequence, etc.) - O(1) lookup
    Compact {
        encoding: CompactCoord,
        is_timestamp: bool,
    },
}

impl<'a> CoordValuesRef<'a> {
    pub fn len(&self) -> usize {
        match self {
            CoordValuesRef::Int64(v) => v.len(),
            CoordValuesRef::Float32(v) => v.len(),
            CoordValuesRef::Float64(v) => v.len(),
            CoordValuesRef::TimestampMicros(v) => v.len(),
            CoordValuesRef::Compact { encoding, .. } => encoding.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Find the index of a scalar value in coordinate values
fn find_value_index(values: &CoordValuesRef<'_>, target: &ScalarValue) -> Option<usize> {
    match (values, target) {
        (CoordValuesRef::Int64(vals), ScalarValue::Int64(Some(v))) => {
            vals.iter().position(|x| x == v)
        }
        (CoordValuesRef::Int64(vals), ScalarValue::Int32(Some(v))) => {
            let v64 = *v as i64;
            vals.iter().position(|x| *x == v64)
        }
        (CoordValuesRef::Float32(vals), ScalarValue::Float32(Some(v))) => {
            vals.iter().position(|x| (x - v).abs() < f32::EPSILON)
        }
        (CoordValuesRef::Float32(vals), ScalarValue::Float64(Some(v))) => {
            let v32 = *v as f32;
            vals.iter().position(|x| (x - v32).abs() < f32::EPSILON)
        }
        (CoordValuesRef::Float64(vals), ScalarValue::Float64(Some(v))) => {
            vals.iter().position(|x| (x - v).abs() < f64::EPSILON)
        }
        (CoordValuesRef::Float64(vals), ScalarValue::Float32(Some(v))) => {
            let v64 = *v as f64;
            vals.iter().position(|x| (x - v64).abs() < f64::EPSILON)
        }
        // Handle integer to float comparisons
        (CoordValuesRef::Float32(vals), ScalarValue::Int64(Some(v))) => {
            let vf = *v as f32;
            vals.iter().position(|x| (x - vf).abs() < f32::EPSILON)
        }
        (CoordValuesRef::Float64(vals), ScalarValue::Int64(Some(v))) => {
            let vf = *v as f64;
            vals.iter().position(|x| (x - vf).abs() < f64::EPSILON)
        }
        // Timestamp comparisons (microseconds since Unix epoch)
        (CoordValuesRef::TimestampMicros(vals), ScalarValue::TimestampMicrosecond(Some(v), _)) => {
            vals.iter().position(|x| x == v)
        }
        (CoordValuesRef::TimestampMicros(vals), ScalarValue::TimestampNanosecond(Some(v), _)) => {
            // Convert nanoseconds to microseconds for comparison
            let v_micros = v / 1000;
            vals.iter().position(|x| *x == v_micros)
        }
        (CoordValuesRef::TimestampMicros(vals), ScalarValue::Int64(Some(v))) => {
            // Allow comparing timestamps with raw i64 microsecond values
            vals.iter().position(|x| x == v)
        }
        // Compact coordinate O(1) lookup for arithmetic sequences
        (
            CoordValuesRef::Compact {
                encoding,
                is_timestamp,
            },
            target,
        ) => find_compact_index(encoding, *is_timestamp, target),
        _ => {
            debug!(
                target_type = ?std::mem::discriminant(target),
                "Unsupported filter value type for coordinate lookup"
            );
            None
        }
    }
}

/// O(1) index lookup for compact (arithmetic) coordinates
///
/// For arithmetic sequence `value[i] = first + i * step`, we compute:
/// `index = (target - first) / step` and verify it's valid.
fn find_compact_index(
    encoding: &CompactCoord,
    is_timestamp: bool,
    target: &ScalarValue,
) -> Option<usize> {
    // Convert target to appropriate numeric type
    let target_f64 = match target {
        ScalarValue::Int64(Some(v)) => *v as f64,
        ScalarValue::Int32(Some(v)) => *v as f64,
        ScalarValue::Float32(Some(v)) => *v as f64,
        ScalarValue::Float64(Some(v)) => *v,
        ScalarValue::TimestampMicrosecond(Some(v), _) if is_timestamp => *v as f64,
        ScalarValue::TimestampNanosecond(Some(v), _) if is_timestamp => (*v / 1000) as f64,
        _ => {
            debug!(
                target_type = ?std::mem::discriminant(target),
                "Unsupported target type for compact coordinate lookup"
            );
            return None;
        }
    };

    match encoding {
        CompactCoord::ArithmeticInt { first, step, len } => {
            if *step == 0 {
                // Constant sequence - only index 0 if target matches
                return if (target_f64 - *first as f64).abs() < f64::EPSILON {
                    Some(0)
                } else {
                    None
                };
            }
            let target_i64 = target_f64.round() as i64;
            let diff = target_i64 - first;
            if diff % step != 0 {
                return None; // Not evenly divisible
            }
            let index = (diff / step) as usize;
            if index < *len {
                Some(index)
            } else {
                None
            }
        }
        CompactCoord::Arithmetic { first, step, len } => {
            if step.abs() < f64::EPSILON {
                // Constant sequence
                return if (target_f64 - first).abs() < f64::EPSILON {
                    Some(0)
                } else {
                    None
                };
            }
            let index_f64 = (target_f64 - first) / step;
            let index = index_f64.round() as usize;
            // Verify this index produces the target value (within tolerance)
            if index < *len {
                let computed = first + (index as f64) * step;
                if (computed - target_f64).abs() < 1e-9 {
                    return Some(index);
                }
            }
            None
        }
    }
}

/// Convert a ScalarValue to f64 for comparison purposes
///
/// Returns None if the type is not supported for comparison.
/// Note: Timestamps are converted to microseconds for uniform comparison.
fn scalar_to_f64(value: &ScalarValue) -> Option<f64> {
    match value {
        ScalarValue::Int64(Some(v)) => Some(*v as f64),
        ScalarValue::Int32(Some(v)) => Some(*v as f64),
        ScalarValue::Float32(Some(v)) => Some(*v as f64),
        ScalarValue::Float64(Some(v)) => Some(*v),
        ScalarValue::TimestampMicrosecond(Some(v), _) => Some(*v as f64),
        ScalarValue::TimestampNanosecond(Some(v), _) => Some((*v / 1000) as f64), // Convert ns to µs
        _ => None,
    }
}

// ============================================================================
// Compact coordinate helper functions for O(1) range queries
// ============================================================================

/// For compact ascending: find first index where value >= target
fn compact_lower_bound(encoding: &CompactCoord, target_f64: f64) -> usize {
    match encoding {
        CompactCoord::ArithmeticInt { first, step, len } => {
            if *step == 0 {
                return if (*first as f64) >= target_f64 {
                    0
                } else {
                    *len
                };
            }
            // index = ceil((target - first) / step)
            let diff = target_f64 - (*first as f64);
            let idx = (diff / (*step as f64)).ceil() as i64;
            idx.clamp(0, *len as i64) as usize
        }
        CompactCoord::Arithmetic { first, step, len } => {
            if step.abs() < f64::EPSILON {
                return if *first >= target_f64 { 0 } else { *len };
            }
            let diff = target_f64 - first;
            let idx = (diff / step).ceil() as i64;
            idx.clamp(0, *len as i64) as usize
        }
    }
}

/// For compact ascending: find count of values <= target (exclusive end index)
fn compact_upper_bound(encoding: &CompactCoord, target_f64: f64) -> usize {
    match encoding {
        CompactCoord::ArithmeticInt { first, step, len } => {
            if *step == 0 {
                return if (*first as f64) <= target_f64 {
                    *len
                } else {
                    0
                };
            }
            // Count of values <= target = floor((target - first) / step) + 1
            let diff = target_f64 - (*first as f64);
            let idx = (diff / (*step as f64)).floor() as i64 + 1;
            idx.clamp(0, *len as i64) as usize
        }
        CompactCoord::Arithmetic { first, step, len } => {
            if step.abs() < f64::EPSILON {
                return if *first <= target_f64 { *len } else { 0 };
            }
            let diff = target_f64 - first;
            let idx = (diff / step).floor() as i64 + 1;
            idx.clamp(0, *len as i64) as usize
        }
    }
}

// =============================================================================
// Descending range search helpers (unified to reduce duplication)
// =============================================================================

/// Type of bound for descending range search.
///
/// Used to parameterize the unified descending search functions.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DescendingBoundType {
    /// Find first index where value <= target (start of range)
    FirstLeq,
    /// Find first index where value < target (end of range)
    FirstLt,
}

/// Unified function for compact descending range search.
///
/// Consolidates `compact_first_leq_descending` and `compact_first_lt_descending`.
fn compact_descending_bound(
    encoding: &CompactCoord,
    target_f64: f64,
    bound_type: DescendingBoundType,
    inclusive: bool,
) -> usize {
    match encoding {
        CompactCoord::ArithmeticInt { first, step, len } => {
            if *step == 0 {
                let first_f64 = *first as f64;
                return match (bound_type, inclusive) {
                    (DescendingBoundType::FirstLeq, true) => {
                        if first_f64 <= target_f64 {
                            0
                        } else {
                            *len
                        }
                    }
                    (DescendingBoundType::FirstLeq, false) => {
                        if first_f64 < target_f64 {
                            0
                        } else {
                            *len
                        }
                    }
                    (DescendingBoundType::FirstLt, true) => {
                        if first_f64 >= target_f64 {
                            *len
                        } else {
                            0
                        }
                    }
                    (DescendingBoundType::FirstLt, false) => {
                        if first_f64 > target_f64 {
                            *len
                        } else {
                            0
                        }
                    }
                };
            }
            let step_f64 = *step as f64;
            let first_f64 = *first as f64;
            let diff = (first_f64 - target_f64) / (-step_f64);
            let idx = match (bound_type, inclusive) {
                (DescendingBoundType::FirstLeq, true) => diff.ceil() as i64,
                (DescendingBoundType::FirstLeq, false) => (diff + f64::EPSILON).ceil() as i64,
                (DescendingBoundType::FirstLt, true) => diff.floor() as i64 + 1,
                (DescendingBoundType::FirstLt, false) => diff.ceil() as i64,
            };
            idx.clamp(0, *len as i64) as usize
        }
        CompactCoord::Arithmetic { first, step, len } => {
            if step.abs() < f64::EPSILON {
                return match (bound_type, inclusive) {
                    (DescendingBoundType::FirstLeq, true) => {
                        if *first <= target_f64 {
                            0
                        } else {
                            *len
                        }
                    }
                    (DescendingBoundType::FirstLeq, false) => {
                        if *first < target_f64 {
                            0
                        } else {
                            *len
                        }
                    }
                    (DescendingBoundType::FirstLt, true) => {
                        if *first >= target_f64 {
                            *len
                        } else {
                            0
                        }
                    }
                    (DescendingBoundType::FirstLt, false) => {
                        if *first > target_f64 {
                            *len
                        } else {
                            0
                        }
                    }
                };
            }
            let diff = (first - target_f64) / (-step);
            let idx = match (bound_type, inclusive) {
                (DescendingBoundType::FirstLeq, true) => diff.ceil() as i64,
                (DescendingBoundType::FirstLeq, false) => (diff + f64::EPSILON).ceil() as i64,
                (DescendingBoundType::FirstLt, true) => diff.floor() as i64 + 1,
                (DescendingBoundType::FirstLt, false) => diff.ceil() as i64,
            };
            idx.clamp(0, *len as i64) as usize
        }
    }
}

/// Check if coordinate values are sorted in descending order
fn is_descending(values: &CoordValuesRef<'_>) -> bool {
    if values.len() < 2 {
        return false;
    }
    match values {
        CoordValuesRef::Int64(vals) => vals[0] > vals[vals.len() - 1],
        CoordValuesRef::Float32(vals) => vals[0] > vals[vals.len() - 1],
        CoordValuesRef::Float64(vals) => vals[0] > vals[vals.len() - 1],
        CoordValuesRef::TimestampMicros(vals) => vals[0] > vals[vals.len() - 1],
        CoordValuesRef::Compact { encoding, .. } => {
            // For arithmetic sequences, descending if step < 0
            match encoding {
                CompactCoord::Arithmetic { step, .. } => *step < 0.0,
                CompactCoord::ArithmeticInt { step, .. } => *step < 0,
            }
        }
    }
}

/// Binary search for lower bound index (first value >= target)
///
/// Returns the index of the first element >= target, or values.len() if all are less.
/// Assumes the coordinate values are sorted in ascending order.
fn binary_search_lower_bound(values: &CoordValuesRef<'_>, target: &ScalarValue) -> Option<usize> {
    let target_f64 = scalar_to_f64(target)?;

    let result = match values {
        CoordValuesRef::Int64(vals) => vals.partition_point(|&x| (x as f64) < target_f64),
        CoordValuesRef::Float32(vals) => vals.partition_point(|&x| (x as f64) < target_f64),
        CoordValuesRef::Float64(vals) => vals.partition_point(|&x| x < target_f64),
        CoordValuesRef::TimestampMicros(vals) => vals.partition_point(|&x| (x as f64) < target_f64),
        CoordValuesRef::Compact { encoding, .. } => {
            // For arithmetic: index = ceil((target - first) / step)
            compact_lower_bound(encoding, target_f64)
        }
    };

    Some(result)
}

/// Binary search for upper bound index (last value <= target)
///
/// Returns the index of the last element <= target, or None if all are greater.
/// Assumes the coordinate values are sorted in ascending order.
fn binary_search_upper_bound(values: &CoordValuesRef<'_>, target: &ScalarValue) -> Option<usize> {
    let target_f64 = scalar_to_f64(target)?;

    let result = match values {
        CoordValuesRef::Int64(vals) => {
            // partition_point returns first element > target, so subtract 1
            vals.partition_point(|&x| (x as f64) <= target_f64)
        }
        CoordValuesRef::Float32(vals) => vals.partition_point(|&x| (x as f64) <= target_f64),
        CoordValuesRef::Float64(vals) => vals.partition_point(|&x| x <= target_f64),
        CoordValuesRef::TimestampMicros(vals) => {
            vals.partition_point(|&x| (x as f64) <= target_f64)
        }
        CoordValuesRef::Compact { encoding, .. } => {
            // For arithmetic: index = floor((target - first) / step) + 1
            compact_upper_bound(encoding, target_f64)
        }
    };

    // Result is the number of elements <= target, which is also the exclusive end index
    Some(result)
}

/// Find the range of indices matching a filter
///
/// For equality filters, returns a single-element range (idx, idx+1).
/// For range filters, returns (start_idx, end_idx) covering all matching values.
/// Handles both ascending and descending coordinate arrays.
/// Returns None if no values match (empty result).
fn find_filter_range(
    values: &CoordValuesRef<'_>,
    filter: &CoordFilterKind,
) -> Option<(usize, usize)> {
    match filter {
        CoordFilterKind::Eq(value) => {
            // Use existing exact match logic
            find_value_index(values, value).map(|idx| (idx, idx + 1))
        }
        CoordFilterKind::Range {
            low,
            high,
            low_inclusive,
            high_inclusive,
        } => {
            let len = values.len();
            if len == 0 {
                return None;
            }

            let descending = is_descending(values);
            debug!(
                descending,
                ?low,
                ?high,
                low_inclusive,
                high_inclusive,
                "Finding range filter indices"
            );

            let (start, end) = if descending {
                // For descending arrays (e.g., latitude 90 to -90):
                // - Higher values come first
                // - BETWEEN low AND high means: find all x where low <= x <= high
                // - Start is first index where value <= high
                // - End is first index where value < low

                // Find start: first index where value <= high
                let start = if let Some(high_val) = high {
                    find_first_leq_descending(values, high_val, *high_inclusive)?
                } else {
                    0 // No upper bound, start from beginning
                };

                // Find end: first index where value < low (exclusive)
                let end = if let Some(low_val) = low {
                    find_first_lt_descending(values, low_val, *low_inclusive)?
                } else {
                    len // No lower bound, go to end
                };

                (start, end)
            } else {
                // For ascending arrays (standard case):
                // - Lower values come first
                // - BETWEEN low AND high means: find all x where low <= x <= high
                // - Start is first index where value >= low
                // - End is first index where value > high (exclusive end)

                // Calculate start index from lower bound
                let start = if let Some(low_val) = low {
                    let idx = binary_search_lower_bound(values, low_val)?;
                    if *low_inclusive {
                        idx
                    } else {
                        // For exclusive (>), find first value > target
                        let upper_idx = binary_search_upper_bound(values, low_val)?;
                        if upper_idx > idx {
                            upper_idx
                        } else {
                            idx
                        }
                    }
                } else {
                    0
                };

                // Calculate end index from upper bound
                let end = if let Some(high_val) = high {
                    let idx = binary_search_upper_bound(values, high_val)?;
                    if *high_inclusive {
                        idx
                    } else {
                        binary_search_lower_bound(values, high_val)?
                    }
                } else {
                    len
                };

                (start, end)
            };

            if start < end {
                debug!(
                    start,
                    end,
                    total_len = len,
                    descending,
                    "Calculated range filter indices"
                );
                Some((start, end))
            } else {
                debug!(
                    start,
                    end, descending, "Range filter resulted in empty range"
                );
                None
            }
        }
    }
}

/// Unified function for descending array bound search.
///
/// Consolidates `find_first_leq_descending` and `find_first_lt_descending`.
fn find_descending_bound(
    values: &CoordValuesRef<'_>,
    target: &ScalarValue,
    bound_type: DescendingBoundType,
    inclusive: bool,
) -> Option<usize> {
    let target_f64 = scalar_to_f64(target)?;

    let result = match values {
        CoordValuesRef::Int64(vals) => descending_partition_point(
            vals.iter().map(|&x| x as f64),
            target_f64,
            bound_type,
            inclusive,
        ),
        CoordValuesRef::Float32(vals) => descending_partition_point(
            vals.iter().map(|&x| x as f64),
            target_f64,
            bound_type,
            inclusive,
        ),
        CoordValuesRef::Float64(vals) => {
            descending_partition_point(vals.iter().copied(), target_f64, bound_type, inclusive)
        }
        CoordValuesRef::TimestampMicros(vals) => descending_partition_point(
            vals.iter().map(|&x| x as f64),
            target_f64,
            bound_type,
            inclusive,
        ),
        CoordValuesRef::Compact { encoding, .. } => {
            compact_descending_bound(encoding, target_f64, bound_type, inclusive)
        }
    };

    Some(result)
}

/// Helper to compute partition point for descending arrays.
///
/// Returns the index based on bound type and inclusivity.
fn descending_partition_point(
    values: impl Iterator<Item = f64> + Clone,
    target: f64,
    bound_type: DescendingBoundType,
    inclusive: bool,
) -> usize {
    let vals: Vec<f64> = values.collect();
    match (bound_type, inclusive) {
        // FirstLeq: find first index where value <= target
        (DescendingBoundType::FirstLeq, true) => vals.partition_point(|&x| x > target),
        (DescendingBoundType::FirstLeq, false) => vals.partition_point(|&x| x >= target),
        // FirstLt: find first index where value < target
        (DescendingBoundType::FirstLt, true) => vals.partition_point(|&x| x >= target),
        (DescendingBoundType::FirstLt, false) => vals.partition_point(|&x| x > target),
    }
}

/// For descending arrays: find first index where value <= target (or < if not inclusive)
fn find_first_leq_descending(
    values: &CoordValuesRef<'_>,
    target: &ScalarValue,
    inclusive: bool,
) -> Option<usize> {
    find_descending_bound(values, target, DescendingBoundType::FirstLeq, inclusive)
}

/// For descending arrays: find first index where value < target (or <= if not inclusive)
fn find_first_lt_descending(
    values: &CoordValuesRef<'_>,
    target: &ScalarValue,
    inclusive: bool,
) -> Option<usize> {
    find_descending_bound(values, target, DescendingBoundType::FirstLt, inclusive)
}

/// Calculate the total number of rows after applying coordinate filters
pub fn calculate_filtered_rows(coord_ranges: &[(usize, usize)]) -> usize {
    coord_ranges
        .iter()
        .map(|(start, end)| end - start)
        .product()
}

/// Calculate Zarr array subset ranges from coordinate filter ranges
///
/// Converts coordinate ranges to ArraySubset ranges for reading
/// a specific slice of an nD Zarr array.
pub fn coord_ranges_to_array_ranges(coord_ranges: &[(usize, usize)]) -> Vec<std::ops::Range<u64>> {
    coord_ranges
        .iter()
        .map(|(start, end)| (*start as u64)..(*end as u64))
        .collect()
}

/// Match coordinate ranges to a data variable's actual dimensions
///
/// Some data variables have fewer dimensions than the total number of coordinates.
/// For example, ERA5 has coordinates [time, level, latitude, longitude] but surface
/// variables like `2m_temperature` only have [time, latitude, longitude] (no level).
///
/// This function matches coordinate dimensions to the data variable by size,
/// returning only the ranges that apply to this specific variable.
///
/// # Arguments
/// * `coord_sizes` - Full size of each coordinate (e.g., [1323648, 37, 721, 1440])
/// * `coord_ranges` - Filtered ranges for each coordinate (e.g., [(0, 277), (0, 37), (141, 265), (1000, 1116)])
/// * `data_var_shape` - Shape of the data variable (e.g., [1323648, 721, 1440] for 3D)
///
/// # Returns
/// Ranges matching the data variable's dimensions, or None if dimensions can't be matched
pub fn match_ranges_to_data_var(
    coord_sizes: &[usize],
    coord_ranges: &[(usize, usize)],
    data_var_shape: &[u64],
) -> Option<Vec<std::ops::Range<u64>>> {
    use tracing::debug;

    // If dimensions match exactly, use all ranges
    if coord_ranges.len() == data_var_shape.len() {
        return Some(coord_ranges_to_array_ranges(coord_ranges));
    }

    // Data variable has fewer dimensions - match by size
    let mut matched_ranges = Vec::with_capacity(data_var_shape.len());

    for dim_size in data_var_shape {
        let dim_size_usize = *dim_size as usize;

        // Find the coordinate that matches this dimension size
        let mut found = false;
        for (coord_idx, &coord_size) in coord_sizes.iter().enumerate() {
            if coord_size == dim_size_usize {
                // Check if we haven't already used this coordinate
                // (handles case where multiple coords have same size)
                if matched_ranges.len() == matched_ranges.iter().filter(|_| true).count() {
                    let (start, end) = coord_ranges[coord_idx];
                    matched_ranges.push((start as u64)..(end as u64));
                    found = true;
                    break;
                }
            }
        }

        if !found {
            // If no matching coordinate found, use full range for this dimension
            debug!(
                dim_size = dim_size_usize,
                "No matching coordinate for dimension, using full range"
            );
            matched_ranges.push(0..*dim_size);
        }
    }

    if matched_ranges.len() == data_var_shape.len() {
        Some(matched_ranges)
    } else {
        debug!(
            expected = data_var_shape.len(),
            got = matched_ranges.len(),
            "Failed to match all dimensions"
        );
        None
    }
}

// =============================================================================
// Variable-to-coordinate mapping for mixed-dimensionality datasets
// =============================================================================

use super::schema_inference::ZarrArrayMeta;

/// Get the coordinate indices that apply to a specific data variable
///
/// This is essential for mixed-dimensionality datasets like ERA5, where some
/// variables are 3D (time × lat × lon) and others are 4D (time × level × lat × lon).
///
/// Priority:
/// 1. If explicit `_ARRAY_DIMENSIONS` metadata exists, match by name
/// 2. Otherwise, fall back to matching by shape (ambiguous if multiple coords share size)
pub fn get_variable_coords(
    var_shape: &[u64],
    var_dimensions: Option<&[String]>,
    coord_names: &[String],
    coord_sizes: &[usize],
) -> Vec<usize> {
    // If explicit dimension names exist, match by name
    if let Some(dims) = var_dimensions {
        let indices: Vec<usize> = dims
            .iter()
            .filter_map(|dim_name| coord_names.iter().position(|c| c == dim_name))
            .collect();

        if indices.len() == dims.len() {
            debug!(
                dims = ?dims,
                indices = ?indices,
                "Matched variable dimensions by name"
            );
            return indices;
        }
        // If name matching failed (some dims not found), fall through to shape matching
        warn!(
            dims = ?dims,
            found = indices.len(),
            "Partial dimension name match, falling back to shape matching"
        );
    }

    // Fallback: match by size (greedy, preserves order)
    match_shape_to_coords(var_shape, coord_sizes)
}

/// Match a variable's shape to coordinate indices by size
///
/// For a variable with shape [277, 124, 116] and coordinates with sizes
/// [277, 37, 124, 116] (time, level, lat, lon), this returns [0, 2, 3]
/// (time, lat, lon - skipping level).
fn match_shape_to_coords(var_shape: &[u64], coord_sizes: &[usize]) -> Vec<usize> {
    let mut matched_indices = Vec::with_capacity(var_shape.len());
    let mut used = vec![false; coord_sizes.len()];

    for &dim_size in var_shape {
        let dim_size_usize = dim_size as usize;

        // Find the first unused coordinate with matching size
        for (coord_idx, &coord_size) in coord_sizes.iter().enumerate() {
            if !used[coord_idx] && coord_size == dim_size_usize {
                matched_indices.push(coord_idx);
                used[coord_idx] = true;
                break;
            }
        }
    }

    if matched_indices.len() != var_shape.len() {
        warn!(
            var_shape = ?var_shape,
            coord_sizes = ?coord_sizes,
            matched = matched_indices.len(),
            "Could not match all variable dimensions to coordinates"
        );
    } else {
        debug!(
            var_shape = ?var_shape,
            matched_indices = ?matched_indices,
            "Matched variable dimensions by shape"
        );
    }

    matched_indices
}

/// Determine the effective coordinates for a set of projected columns
///
/// For a query that projects only 3D variables, this returns only the 3 coordinate
/// indices used by those variables. For coordinate-only queries with LIMIT (no data
/// variables), returns only the indices of selected coordinates to avoid unnecessary
/// Cartesian expansion. If the projection is empty (e.g., COUNT(*)), returns all
/// coordinates.
///
/// The `limit` parameter enables the coordinate-only optimization. Without a LIMIT,
/// coordinate-only queries return all coordinates to preserve correct semantics for
/// aggregate queries like `SELECT COUNT(*), MIN(lat) FROM data`.
///
/// Returns `Err` with a helpful message if the projected variables have mixed
/// dimensionality (e.g., both 3D and 4D variables in the same query).
pub fn determine_effective_coords(
    projected_var_names: &[&str],
    projected_coord_names: &[&str],
    all_data_vars: &[ZarrArrayMeta],
    coord_names: &[String],
    coord_sizes: &[usize],
    limit: Option<usize>,
) -> Result<Vec<usize>, String> {
    // Case 1: Data variables projected - determine coordinates from data variable shapes
    if !projected_var_names.is_empty() {
        // Fall through to existing logic below
    } else if !projected_coord_names.is_empty() && limit.is_some() {
        // Case 2: Only coordinates projected with LIMIT (e.g., SELECT time LIMIT 10)
        // Return indices of only the selected coordinates, preserving alphabetical order
        // This optimization avoids Cartesian expansion when user only wants coordinate values
        let mut indices: Vec<usize> = projected_coord_names
            .iter()
            .filter_map(|&name| coord_names.iter().position(|c| c == name))
            .collect();
        indices.sort();
        indices.dedup();

        if indices.is_empty() {
            // No valid coordinates found - this shouldn't happen
            return Ok((0..coord_sizes.len()).collect());
        }

        debug!(
            projected_coords = ?projected_coord_names,
            indices = ?indices,
            limit = ?limit,
            "Coordinate-only query with LIMIT: using selected coordinates only"
        );

        return Ok(indices);
    } else {
        // Case 3: Empty projection (e.g., COUNT(*)) or coord-only without LIMIT
        // Use all coordinates to preserve correct row count for aggregates
        return Ok((0..coord_sizes.len()).collect());
    }

    // Collect coordinate indices for each projected variable
    let mut all_var_coord_sets: Vec<(String, Vec<usize>)> = Vec::new();

    for &var_name in projected_var_names {
        // Find the data variable metadata
        if let Some(var_meta) = all_data_vars.iter().find(|v| v.name == var_name) {
            let var_coords = get_variable_coords(
                &var_meta.shape,
                var_meta.dimensions.as_deref(),
                coord_names,
                coord_sizes,
            );
            all_var_coord_sets.push((var_name.to_string(), var_coords));
        } else {
            // Variable not found in metadata - should not happen
            warn!(var_name = %var_name, "Data variable not found in metadata");
        }
    }

    if all_var_coord_sets.is_empty() {
        // All projected columns were coordinates, not data variables
        return Ok((0..coord_sizes.len()).collect());
    }

    // Check if all variables use the same set of coordinates
    let first_coords = &all_var_coord_sets[0].1;
    let all_same = all_var_coord_sets
        .iter()
        .all(|(_, coords)| coords == first_coords);

    if all_same {
        return Ok(first_coords.clone());
    }

    // Mixed dimensionality - build helpful error message
    let mut dim_groups: HashMap<usize, Vec<&str>> = HashMap::new();
    for (name, coords) in &all_var_coord_sets {
        dim_groups.entry(coords.len()).or_default().push(name);
    }

    let detail: Vec<String> = dim_groups
        .iter()
        .map(|(dims, vars)| format!("{}D: {}", dims, vars.join(", ")))
        .collect();

    Err(format!(
        "Cannot project variables with different dimensions in the same query. \
         Found: {}. Query these variable types separately.",
        detail.join("; ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    #[test]
    fn test_parse_simple_equality() {
        let coord_names = vec!["time".to_string(), "lat".to_string()];

        // time = 100
        let filter = col("time").eq(lit(100i64));
        let filters = parse_coord_filters(&[filter], &coord_names);

        assert_eq!(filters.len(), 1);
        assert!(filters.get("time").is_some());
    }

    #[test]
    fn test_parse_and_filters() {
        let coord_names = vec!["time".to_string(), "hybrid".to_string(), "lat".to_string()];

        // time = 100 AND hybrid = 50
        let filter = col("time")
            .eq(lit(100i64))
            .and(col("hybrid").eq(lit(50i64)));
        let filters = parse_coord_filters(&[filter], &coord_names);

        assert_eq!(filters.len(), 2);
        assert!(filters.get("time").is_some());
        assert!(filters.get("hybrid").is_some());
    }

    #[test]
    fn test_ignore_non_coord_columns() {
        let coord_names = vec!["time".to_string()];

        // temperature = 20 (not a coordinate)
        let filter = col("temperature").eq(lit(20i64));
        let filters = parse_coord_filters(&[filter], &coord_names);

        assert!(filters.is_empty());
    }

    #[test]
    fn test_find_value_index() {
        let vals = vec![10i64, 20, 30, 40, 50];
        let values_ref = CoordValuesRef::Int64(&vals);

        assert_eq!(
            find_value_index(&values_ref, &ScalarValue::Int64(Some(30))),
            Some(2)
        );
        assert_eq!(
            find_value_index(&values_ref, &ScalarValue::Int64(Some(100))),
            None
        );
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_calculate_filtered_rows() {
        // time: 1 value, hybrid: 1 value, lat: 721, lon: 1440
        let ranges = vec![(5, 6), (10, 11), (0, 721), (0, 1440)];
        let rows = calculate_filtered_rows(&ranges);
        assert_eq!(rows, 1 * 1 * 721 * 1440);
    }

    // ==================== Binary search tests ====================

    #[test]
    fn test_binary_search_lower_bound_int64() {
        let vals = vec![10i64, 20, 30, 40, 50];
        let values_ref = CoordValuesRef::Int64(&vals);

        // Exact match
        assert_eq!(
            binary_search_lower_bound(&values_ref, &ScalarValue::Int64(Some(30))),
            Some(2)
        );

        // Between values - returns first >= target
        assert_eq!(
            binary_search_lower_bound(&values_ref, &ScalarValue::Int64(Some(25))),
            Some(2) // index of 30
        );

        // Below all values
        assert_eq!(
            binary_search_lower_bound(&values_ref, &ScalarValue::Int64(Some(5))),
            Some(0)
        );

        // Above all values
        assert_eq!(
            binary_search_lower_bound(&values_ref, &ScalarValue::Int64(Some(100))),
            Some(5) // len()
        );
    }

    #[test]
    fn test_binary_search_upper_bound_int64() {
        let vals = vec![10i64, 20, 30, 40, 50];
        let values_ref = CoordValuesRef::Int64(&vals);

        // Exact match - returns index after last <=
        assert_eq!(
            binary_search_upper_bound(&values_ref, &ScalarValue::Int64(Some(30))),
            Some(3) // exclusive end index
        );

        // Between values - returns first > target position
        assert_eq!(
            binary_search_upper_bound(&values_ref, &ScalarValue::Int64(Some(25))),
            Some(2) // index after 20
        );

        // Below all values
        assert_eq!(
            binary_search_upper_bound(&values_ref, &ScalarValue::Int64(Some(5))),
            Some(0) // no elements <= 5
        );

        // Above all values
        assert_eq!(
            binary_search_upper_bound(&values_ref, &ScalarValue::Int64(Some(100))),
            Some(5) // all elements included
        );
    }

    #[test]
    fn test_binary_search_float64() {
        let vals = vec![1.0f64, 2.5, 5.0, 7.5, 10.0];
        let values_ref = CoordValuesRef::Float64(&vals);

        // Exact match
        assert_eq!(
            binary_search_lower_bound(&values_ref, &ScalarValue::Float64(Some(5.0))),
            Some(2)
        );

        // Between values
        assert_eq!(
            binary_search_lower_bound(&values_ref, &ScalarValue::Float64(Some(3.0))),
            Some(2) // first >= 3.0 is 5.0 at index 2
        );

        assert_eq!(
            binary_search_upper_bound(&values_ref, &ScalarValue::Float64(Some(6.0))),
            Some(3) // elements <= 6.0: [1.0, 2.5, 5.0], exclusive end = 3
        );
    }

    // ==================== find_filter_range tests ====================

    #[test]
    fn test_find_filter_range_equality() {
        let vals = vec![10i64, 20, 30, 40, 50];
        let values_ref = CoordValuesRef::Int64(&vals);

        let filter = CoordFilterKind::Eq(ScalarValue::Int64(Some(30)));
        let range = find_filter_range(&values_ref, &filter);

        assert_eq!(range, Some((2, 3))); // single element at index 2
    }

    #[test]
    fn test_find_filter_range_between() {
        let vals = vec![10i64, 20, 30, 40, 50];
        let values_ref = CoordValuesRef::Int64(&vals);

        // BETWEEN 20 AND 40 (inclusive)
        let filter = CoordFilterKind::Range {
            low: Some(ScalarValue::Int64(Some(20))),
            high: Some(ScalarValue::Int64(Some(40))),
            low_inclusive: true,
            high_inclusive: true,
        };
        let range = find_filter_range(&values_ref, &filter);

        assert_eq!(range, Some((1, 4))); // indices 1, 2, 3 (values 20, 30, 40)
    }

    #[test]
    fn test_find_filter_range_exclusive() {
        let vals = vec![10i64, 20, 30, 40, 50];
        let values_ref = CoordValuesRef::Int64(&vals);

        // > 20 AND < 40 (exclusive on both ends)
        let filter = CoordFilterKind::Range {
            low: Some(ScalarValue::Int64(Some(20))),
            high: Some(ScalarValue::Int64(Some(40))),
            low_inclusive: false,
            high_inclusive: false,
        };
        let range = find_filter_range(&values_ref, &filter);

        assert_eq!(range, Some((2, 3))); // only index 2 (value 30)
    }

    #[test]
    fn test_find_filter_range_half_open_low() {
        let vals = vec![10i64, 20, 30, 40, 50];
        let values_ref = CoordValuesRef::Int64(&vals);

        // >= 30 (no upper bound)
        let filter = CoordFilterKind::Range {
            low: Some(ScalarValue::Int64(Some(30))),
            high: None,
            low_inclusive: true,
            high_inclusive: true,
        };
        let range = find_filter_range(&values_ref, &filter);

        assert_eq!(range, Some((2, 5))); // indices 2, 3, 4 (values 30, 40, 50)
    }

    #[test]
    fn test_find_filter_range_half_open_high() {
        let vals = vec![10i64, 20, 30, 40, 50];
        let values_ref = CoordValuesRef::Int64(&vals);

        // <= 30 (no lower bound)
        let filter = CoordFilterKind::Range {
            low: None,
            high: Some(ScalarValue::Int64(Some(30))),
            low_inclusive: true,
            high_inclusive: true,
        };
        let range = find_filter_range(&values_ref, &filter);

        assert_eq!(range, Some((0, 3))); // indices 0, 1, 2 (values 10, 20, 30)
    }

    #[test]
    fn test_find_filter_range_no_match() {
        let vals = vec![10i64, 20, 30, 40, 50];
        let values_ref = CoordValuesRef::Int64(&vals);

        // BETWEEN 100 AND 200 (no values in range)
        let filter = CoordFilterKind::Range {
            low: Some(ScalarValue::Int64(Some(100))),
            high: Some(ScalarValue::Int64(Some(200))),
            low_inclusive: true,
            high_inclusive: true,
        };
        let range = find_filter_range(&values_ref, &filter);

        assert_eq!(range, None); // empty result
    }

    // ==================== Variable coordinate mapping tests ====================

    #[test]
    fn test_get_variable_coords_with_explicit_dims() {
        let coord_names = vec![
            "time".to_string(),
            "level".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ];
        let coord_sizes = vec![100, 37, 721, 1440];

        // 3D variable with explicit dimensions (no level)
        let var_shape = vec![100u64, 721, 1440];
        let var_dims = Some(vec![
            "time".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ]);

        let result =
            get_variable_coords(&var_shape, var_dims.as_deref(), &coord_names, &coord_sizes);

        // Should return indices for time, latitude, longitude (skipping level)
        assert_eq!(result, vec![0, 2, 3]);
    }

    #[test]
    fn test_get_variable_coords_shape_fallback() {
        let coord_names = vec![
            "time".to_string(),
            "level".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ];
        let coord_sizes = vec![100, 37, 721, 1440];

        // 3D variable without explicit dimensions - must match by shape
        let var_shape = vec![100u64, 721, 1440];
        let var_dims: Option<&[String]> = None;

        let result = get_variable_coords(&var_shape, var_dims, &coord_names, &coord_sizes);

        // Should match time (100), latitude (721), longitude (1440) by size
        assert_eq!(result, vec![0, 2, 3]);
    }

    #[test]
    fn test_get_variable_coords_4d() {
        let coord_names = vec![
            "time".to_string(),
            "level".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ];
        let coord_sizes = vec![100, 37, 721, 1440];

        // 4D variable uses all coordinates
        let var_shape = vec![100u64, 37, 721, 1440];
        let var_dims: Option<&[String]> = None;

        let result = get_variable_coords(&var_shape, var_dims, &coord_names, &coord_sizes);

        assert_eq!(result, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_determine_effective_coords_same_dims() {
        use super::super::schema_inference::ZarrArrayMeta;

        let coord_names = vec![
            "time".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ];
        let coord_sizes = vec![100, 721, 1440];

        let data_vars = vec![
            ZarrArrayMeta {
                name: "temperature".to_string(),
                data_type: "float32".to_string(),
                shape: vec![100, 721, 1440],
                chunks: None,
                coord_min_max: None,
                cf_time_attrs: None,
                dimensions: Some(vec![
                    "time".to_string(),
                    "latitude".to_string(),
                    "longitude".to_string(),
                ]),
            },
            ZarrArrayMeta {
                name: "humidity".to_string(),
                data_type: "float32".to_string(),
                shape: vec![100, 721, 1440],
                chunks: None,
                coord_min_max: None,
                cf_time_attrs: None,
                dimensions: Some(vec![
                    "time".to_string(),
                    "latitude".to_string(),
                    "longitude".to_string(),
                ]),
            },
        ];

        let projected_vars = vec!["temperature", "humidity"];
        let projected_coords: Vec<&str> = vec![]; // No coordinates directly projected

        let result = determine_effective_coords(
            &projected_vars,
            &projected_coords,
            &data_vars,
            &coord_names,
            &coord_sizes,
            None,
        );

        // Both variables use the same 3 coordinates
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_determine_effective_coords_mixed_error() {
        use super::super::schema_inference::ZarrArrayMeta;

        let coord_names = vec![
            "time".to_string(),
            "level".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ];
        let coord_sizes = vec![100, 37, 721, 1440];

        let data_vars = vec![
            ZarrArrayMeta {
                name: "temperature".to_string(), // 4D
                data_type: "float32".to_string(),
                shape: vec![100, 37, 721, 1440],
                chunks: None,
                coord_min_max: None,
                cf_time_attrs: None,
                dimensions: Some(vec![
                    "time".to_string(),
                    "level".to_string(),
                    "latitude".to_string(),
                    "longitude".to_string(),
                ]),
            },
            ZarrArrayMeta {
                name: "surface_temp".to_string(), // 3D
                data_type: "float32".to_string(),
                shape: vec![100, 721, 1440],
                chunks: None,
                coord_min_max: None,
                cf_time_attrs: None,
                dimensions: Some(vec![
                    "time".to_string(),
                    "latitude".to_string(),
                    "longitude".to_string(),
                ]),
            },
        ];

        let projected_vars = vec!["temperature", "surface_temp"];
        let projected_coords: Vec<&str> = vec![]; // No coordinates directly projected

        let result = determine_effective_coords(
            &projected_vars,
            &projected_coords,
            &data_vars,
            &coord_names,
            &coord_sizes,
            None,
        );

        // Should error because variables have different dimensionality
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("Cannot project variables with different dimensions"));
    }

    #[test]
    fn test_determine_effective_coords_empty_projection() {
        use super::super::schema_inference::ZarrArrayMeta;

        let coord_names = vec!["time".to_string(), "latitude".to_string()];
        let coord_sizes = vec![100, 721];
        let data_vars: Vec<ZarrArrayMeta> = vec![];
        let projected_vars: Vec<&str> = vec![]; // e.g., COUNT(*)
        let projected_coords: Vec<&str> = vec![]; // No coordinates directly projected

        let result = determine_effective_coords(
            &projected_vars,
            &projected_coords,
            &data_vars,
            &coord_names,
            &coord_sizes,
            None,
        );

        // Empty projection (e.g. COUNT(*)) should return all coordinates
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0, 1]);
    }

    #[test]
    fn test_determine_effective_coords_coord_only_with_limit() {
        // Test coordinate-only queries with LIMIT like SELECT time, latitude LIMIT 10
        let coord_names = vec![
            "time".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ];
        let coord_sizes = vec![100, 721, 1440];
        let data_vars: Vec<super::super::schema_inference::ZarrArrayMeta> = vec![];
        let projected_vars: Vec<&str> = vec![]; // No data variables
        let projected_coords = vec!["time", "latitude"]; // Only coordinates projected

        let result = determine_effective_coords(
            &projected_vars,
            &projected_coords,
            &data_vars,
            &coord_names,
            &coord_sizes,
            Some(10),
        );

        // Coordinate-only projection WITH LIMIT should return only the selected coordinates
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0, 1]); // time=0, latitude=1
    }

    #[test]
    fn test_determine_effective_coords_coord_only_no_limit() {
        // Test coordinate-only queries WITHOUT LIMIT like SELECT time, latitude
        // Should return all coordinates (no optimization) to preserve aggregate semantics
        let coord_names = vec![
            "time".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ];
        let coord_sizes = vec![100, 721, 1440];
        let data_vars: Vec<super::super::schema_inference::ZarrArrayMeta> = vec![];
        let projected_vars: Vec<&str> = vec![]; // No data variables
        let projected_coords = vec!["time", "latitude"]; // Only coordinates projected

        let result = determine_effective_coords(
            &projected_vars,
            &projected_coords,
            &data_vars,
            &coord_names,
            &coord_sizes,
            None,
        );

        // Coordinate-only projection WITHOUT LIMIT should return ALL coordinates
        // This preserves correct behavior for aggregate queries like SELECT COUNT(*), MIN(lat)
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0, 1, 2]); // All coordinates
    }

    #[test]
    fn test_determine_effective_coords_single_coord_with_limit() {
        // Test single coordinate query with LIMIT like SELECT time LIMIT 10
        let coord_names = vec![
            "time".to_string(),
            "latitude".to_string(),
            "longitude".to_string(),
        ];
        let coord_sizes = vec![100, 721, 1440];
        let data_vars: Vec<super::super::schema_inference::ZarrArrayMeta> = vec![];
        let projected_vars: Vec<&str> = vec![]; // No data variables
        let projected_coords = vec!["time"]; // Only time projected

        let result = determine_effective_coords(
            &projected_vars,
            &projected_coords,
            &data_vars,
            &coord_names,
            &coord_sizes,
            Some(10),
        );

        // Single coordinate projection should return only that coordinate
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0]); // time=0
    }

    // ==================== filter_satisfiable_by_bounds tests ====================

    #[test]
    fn test_filter_satisfiable_by_bounds_equality_within() {
        use super::super::schema_inference::ZarrArrayMeta;

        let coords = vec![ZarrArrayMeta {
            name: "lat".to_string(),
            data_type: "float64".to_string(),
            shape: vec![100],
            chunks: None,
            coord_min_max: Some((0.0, 90.0)),
            cf_time_attrs: None,
            dimensions: None,
        }];

        let mut filters = CoordFilters::new();
        filters.filters.insert(
            "lat".to_string(),
            CoordFilterKind::Eq(ScalarValue::Float64(Some(45.0))),
        );

        assert!(filter_satisfiable_by_bounds(&filters, &coords));
    }

    #[test]
    fn test_filter_satisfiable_by_bounds_equality_outside() {
        use super::super::schema_inference::ZarrArrayMeta;

        let coords = vec![ZarrArrayMeta {
            name: "lat".to_string(),
            data_type: "float64".to_string(),
            shape: vec![100],
            chunks: None,
            coord_min_max: Some((0.0, 90.0)),
            cf_time_attrs: None,
            dimensions: None,
        }];

        let mut filters = CoordFilters::new();
        filters.filters.insert(
            "lat".to_string(),
            CoordFilterKind::Eq(ScalarValue::Float64(Some(100.0))),
        );

        assert!(!filter_satisfiable_by_bounds(&filters, &coords));
    }

    #[test]
    fn test_filter_satisfiable_by_bounds_range_overlapping() {
        use super::super::schema_inference::ZarrArrayMeta;

        let coords = vec![ZarrArrayMeta {
            name: "time".to_string(),
            data_type: "int64".to_string(),
            shape: vec![100],
            chunks: None,
            coord_min_max: Some((0.0, 100.0)),
            cf_time_attrs: None,
            dimensions: None,
        }];

        let mut filters = CoordFilters::new();
        filters.filters.insert(
            "time".to_string(),
            CoordFilterKind::Range {
                low: Some(ScalarValue::Int64(Some(50))),
                high: Some(ScalarValue::Int64(Some(150))),
                low_inclusive: true,
                high_inclusive: true,
            },
        );

        // Range [50, 150] overlaps with coord bounds [0, 100]
        assert!(filter_satisfiable_by_bounds(&filters, &coords));
    }

    #[test]
    fn test_filter_satisfiable_by_bounds_range_completely_outside() {
        use super::super::schema_inference::ZarrArrayMeta;

        let coords = vec![ZarrArrayMeta {
            name: "time".to_string(),
            data_type: "int64".to_string(),
            shape: vec![100],
            chunks: None,
            coord_min_max: Some((0.0, 100.0)),
            cf_time_attrs: None,
            dimensions: None,
        }];

        let mut filters = CoordFilters::new();
        filters.filters.insert(
            "time".to_string(),
            CoordFilterKind::Range {
                low: Some(ScalarValue::Int64(Some(200))),
                high: Some(ScalarValue::Int64(Some(300))),
                low_inclusive: true,
                high_inclusive: true,
            },
        );

        // Range [200, 300] is completely outside coord bounds [0, 100]
        assert!(!filter_satisfiable_by_bounds(&filters, &coords));
    }

    #[test]
    fn test_filter_satisfiable_by_bounds_no_bounds() {
        use super::super::schema_inference::ZarrArrayMeta;

        let coords = vec![ZarrArrayMeta {
            name: "lat".to_string(),
            data_type: "float64".to_string(),
            shape: vec![100],
            chunks: None,
            coord_min_max: None, // No bounds available
            cf_time_attrs: None,
            dimensions: None,
        }];

        let mut filters = CoordFilters::new();
        filters.filters.insert(
            "lat".to_string(),
            CoordFilterKind::Eq(ScalarValue::Float64(Some(999.0))),
        );

        // Without bounds, we can't early-reject, so return true
        assert!(filter_satisfiable_by_bounds(&filters, &coords));
    }

    #[test]
    fn test_filter_satisfiable_by_bounds_half_open_range() {
        use super::super::schema_inference::ZarrArrayMeta;

        let coords = vec![ZarrArrayMeta {
            name: "lon".to_string(),
            data_type: "float64".to_string(),
            shape: vec![360],
            chunks: None,
            coord_min_max: Some((-180.0, 180.0)),
            cf_time_attrs: None,
            dimensions: None,
        }];

        // Test >= 170 (overlaps with coord max 180)
        let mut filters = CoordFilters::new();
        filters.filters.insert(
            "lon".to_string(),
            CoordFilterKind::Range {
                low: Some(ScalarValue::Float64(Some(170.0))),
                high: None,
                low_inclusive: true,
                high_inclusive: true,
            },
        );
        assert!(filter_satisfiable_by_bounds(&filters, &coords));

        // Test >= 200 (completely outside)
        let mut filters = CoordFilters::new();
        filters.filters.insert(
            "lon".to_string(),
            CoordFilterKind::Range {
                low: Some(ScalarValue::Float64(Some(200.0))),
                high: None,
                low_inclusive: true,
                high_inclusive: true,
            },
        );
        assert!(!filter_satisfiable_by_bounds(&filters, &coords));
    }
}
