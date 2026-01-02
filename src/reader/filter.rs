//! Filter pushdown support for Zarr queries
//!
//! This module parses DataFusion filter expressions to extract coordinate
//! equality filters (e.g., `time = 1323647`), which can be used to read
//! only the relevant subset of Zarr arrays.
//!
//! For a Zarr store with coordinates [time, hybrid, lat, lon], a filter like
//! `time = X AND hybrid = Y` allows us to read only the slice of data where
//! those coordinates match, dramatically reducing memory usage.

use datafusion::common::ScalarValue;
use datafusion::logical_expr::Expr;
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

/// Collection of coordinate filters extracted from a WHERE clause
#[derive(Debug, Clone, Default)]
pub struct CoordFilters {
    /// Map from coordinate name to filter value
    pub filters: HashMap<String, ScalarValue>,
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

    /// Get the filter value for a coordinate, if any
    pub fn get(&self, coord_name: &str) -> Option<&ScalarValue> {
        self.filters.get(coord_name)
    }

    /// Number of coordinate filters
    pub fn len(&self) -> usize {
        self.filters.len()
    }
}

/// Parse DataFusion filter expressions to extract coordinate equality filters
///
/// Only extracts simple equality filters of the form:
/// - `coord = value` (Column = Literal)
/// - `value = coord` (Literal = Column)
///
/// Combined with AND:
/// - `coord1 = value1 AND coord2 = value2`
///
/// Other filter types (OR, >, <, LIKE, etc.) are ignored and left for
/// DataFusion to handle post-scan.
pub fn parse_coord_filters(filters: &[Expr], coord_names: &[String]) -> CoordFilters {
    let mut result = CoordFilters::new();

    for filter in filters {
        extract_equality_filters(filter, coord_names, &mut result);
    }

    if !result.is_empty() {
        info!(
            num_filters = result.len(),
            filters = ?result.filters.keys().collect::<Vec<_>>(),
            "Extracted coordinate filters for pushdown"
        );
    } else {
        debug!("No coordinate equality filters found for pushdown");
    }

    result
}

/// Recursively extract equality filters from an expression
fn extract_equality_filters(expr: &Expr, coord_names: &[String], result: &mut CoordFilters) {
    match expr {
        // Handle AND: recurse into both sides
        Expr::BinaryExpr(binary) if binary.op == datafusion::logical_expr::Operator::And => {
            extract_equality_filters(&binary.left, coord_names, result);
            extract_equality_filters(&binary.right, coord_names, result);
        }

        // Handle equality: Column = Literal or Literal = Column
        Expr::BinaryExpr(binary) if binary.op == datafusion::logical_expr::Operator::Eq => {
            if let Some((col_name, value)) = extract_column_literal_eq(&binary.left, &binary.right)
            {
                if coord_names.contains(&col_name) {
                    debug!(
                        coord = %col_name,
                        value = %value,
                        "Found coordinate equality filter"
                    );
                    result.filters.insert(col_name, value);
                } else {
                    trace!(
                        column = %col_name,
                        "Equality filter on non-coordinate column, skipping"
                    );
                }
            }
        }

        // Handle CAST expressions that wrap the filter
        Expr::Cast(cast) => {
            extract_equality_filters(&cast.expr, coord_names, result);
        }

        // Other expressions: OR, >, <, etc. - skip for now
        other => {
            trace!(expr_type = %other.variant_name(), "Skipping non-equality filter expression");
        }
    }
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
/// - If filtered (e.g., `time = X`), find the index of X in the coordinate values
/// - If not filtered, read all values
///
/// Returns a map from coordinate name to (start_idx, end_idx) range.
/// If a filter value is not found in the coordinate, returns None (no matches).
pub fn calculate_coord_ranges(
    filters: &CoordFilters,
    coord_names: &[String],
    coord_values: &[CoordValuesRef<'_>],
) -> Option<Vec<(usize, usize)>> {
    let mut ranges = Vec::with_capacity(coord_names.len());

    for (i, name) in coord_names.iter().enumerate() {
        let values = &coord_values[i];
        let range = if let Some(filter_value) = filters.get(name) {
            // Find the index of the filter value in this coordinate
            if let Some(idx) = find_value_index(values, filter_value) {
                debug!(
                    coord = %name,
                    filter_value = %filter_value,
                    index = idx,
                    "Found filter value at index"
                );
                (idx, idx + 1) // Single value range
            } else {
                warn!(
                    coord = %name,
                    filter_value = %filter_value,
                    "Filter value not found in coordinate - query will return no results"
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
}

impl<'a> CoordValuesRef<'a> {
    pub fn len(&self) -> usize {
        match self {
            CoordValuesRef::Int64(v) => v.len(),
            CoordValuesRef::Float32(v) => v.len(),
            CoordValuesRef::Float64(v) => v.len(),
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
        _ => {
            debug!(
                target_type = ?std::mem::discriminant(target),
                "Unsupported filter value type for coordinate lookup"
            );
            None
        }
    }
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
    fn test_calculate_filtered_rows() {
        // time: 1 value, hybrid: 1 value, lat: 721, lon: 1440
        let ranges = vec![(5, 6), (10, 11), (0, 721), (0, 1440)];
        let rows = calculate_filtered_rows(&ranges);
        assert_eq!(rows, 1 * 1 * 721 * 1440);
    }
}
