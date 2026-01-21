//! E-graph analysis for tracking types, statistics, and cardinality
//!
//! This module implements the `Analysis` trait from egg to maintain
//! domain-specific information during equality saturation.

use std::collections::HashMap;

use arrow::datatypes::DataType;
use datafusion::common::ScalarValue;
use egg::{Analysis, DidMerge, EGraph, Id};

use super::cost::CostParameters;
use super::language::{parse_literal, ZarrPlan};
use crate::reader::schema_inference::ZarrStoreMeta;

/// Analysis data computed for each e-class
#[derive(Debug, Clone, Default)]
pub struct ZarrAnalysisData {
    /// Data type of expressions, or schema info for relations
    pub data_type: Option<DataType>,

    /// Known constant value (for constant folding)
    pub constant: Option<ScalarValue>,

    /// Estimated cardinality (number of rows)
    pub cardinality: Option<usize>,

    /// Whether this expression can contain nulls
    pub nullable: Option<bool>,

    /// Estimated byte size
    pub byte_size: Option<usize>,

    /// For MIN aggregate: known minimum value from statistics
    pub min_value: Option<ScalarValue>,

    /// For MAX aggregate: known maximum value from statistics
    pub max_value: Option<ScalarValue>,

    /// For COUNT: known row count from statistics
    pub row_count: Option<usize>,

    /// Column name (for column references)
    pub column_name: Option<String>,

    /// Table name (for table references)
    pub table_name: Option<String>,
}

impl ZarrAnalysisData {
    /// Create analysis data for a known constant
    pub fn constant(value: ScalarValue) -> Self {
        let is_null = value.is_null();
        let data_type = value.data_type();
        Self {
            data_type: Some(data_type),
            constant: Some(value),
            nullable: Some(is_null),
            ..Default::default()
        }
    }

    /// Create analysis data for a column reference
    pub fn column(name: String, data_type: Option<DataType>) -> Self {
        Self {
            column_name: Some(name),
            data_type,
            nullable: Some(true), // Assume nullable unless we know otherwise
            ..Default::default()
        }
    }

    /// Create analysis data for a table reference
    pub fn table(name: String, cardinality: Option<usize>) -> Self {
        Self {
            table_name: Some(name),
            cardinality,
            ..Default::default()
        }
    }

    /// Check if this is a known constant
    pub fn is_constant(&self) -> bool {
        self.constant.is_some()
    }

    /// Get the constant value if known
    pub fn get_constant(&self) -> Option<&ScalarValue> {
        self.constant.as_ref()
    }
}

/// Analysis for the Zarr query optimizer
///
/// This tracks:
/// - Table statistics from Zarr metadata
/// - Data types for type checking
/// - Cardinality estimates for cost computation
/// - Constant values for constant folding
#[derive(Default)]
pub struct ZarrAnalysis {
    /// Statistics cache for Zarr tables (table name -> metadata)
    pub table_stats: HashMap<String, ZarrStoreMeta>,
    /// Tunable cost parameters
    pub cost_params: CostParameters,
}

impl ZarrAnalysis {
    /// Create a new analysis with table statistics
    pub fn new(table_stats: HashMap<String, ZarrStoreMeta>, cost_params: CostParameters) -> Self {
        Self {
            table_stats,
            cost_params,
        }
    }
}

impl Analysis<ZarrPlan> for ZarrAnalysis {
    type Data = ZarrAnalysisData;

    fn make(egraph: &mut EGraph<ZarrPlan, Self>, enode: &ZarrPlan, _id: Id) -> Self::Data {
        match enode {
            // === Universal Symbol - can be literal, column, table, or type ===
            // Format detection: type:value = literal, otherwise column/table/type
            ZarrPlan::Symbol(s) => {
                let s_str = s.as_str();
                // Check if it's a literal (format: type:value)
                if let Some((type_str, value_str)) = parse_literal(s) {
                    match type_str.as_str() {
                        "i64" => {
                            if let Ok(v) = value_str.parse::<i64>() {
                                return ZarrAnalysisData::constant(ScalarValue::Int64(Some(v)));
                            }
                        }
                        "f64" => {
                            if let Ok(v) = value_str.parse::<f64>() {
                                return ZarrAnalysisData::constant(ScalarValue::Float64(Some(v)));
                            }
                        }
                        "str" => {
                            return ZarrAnalysisData::constant(ScalarValue::Utf8(Some(
                                value_str,
                            )));
                        }
                        "bool" => {
                            let v = value_str == "true";
                            return ZarrAnalysisData::constant(ScalarValue::Boolean(Some(v)));
                        }
                        "usize" | "u64" => {
                            if let Ok(v) = value_str.parse::<u64>() {
                                return ZarrAnalysisData::constant(ScalarValue::UInt64(Some(v)));
                            }
                        }
                        _ => {}
                    }
                }
                // Otherwise treat as column/identifier reference
                ZarrAnalysisData::column(s_str.to_string(), None)
            }

            ZarrPlan::True => ZarrAnalysisData::constant(ScalarValue::Boolean(Some(true))),
            ZarrPlan::False => ZarrAnalysisData::constant(ScalarValue::Boolean(Some(false))),
            ZarrPlan::Null => ZarrAnalysisData {
                constant: Some(ScalarValue::Null),
                nullable: Some(true),
                ..Default::default()
            },

            // === Arithmetic expressions ===
            ZarrPlan::Add([l, r])
            | ZarrPlan::Sub([l, r])
            | ZarrPlan::Mul([l, r])
            | ZarrPlan::Div([l, r])
            | ZarrPlan::Mod([l, r]) => {
                let left = &egraph[*l].data;
                let right = &egraph[*r].data;

                // Try constant folding
                if let (Some(lv), Some(rv)) = (&left.constant, &right.constant) {
                    if let Some(result) = fold_arithmetic(enode, lv, rv) {
                        return ZarrAnalysisData::constant(result);
                    }
                }

                // Propagate type information
                ZarrAnalysisData {
                    data_type: merge_numeric_types(&left.data_type, &right.data_type),
                    nullable: merge_nullable(left.nullable, right.nullable),
                    ..Default::default()
                }
            }

            ZarrPlan::Neg([e]) => {
                let inner = &egraph[*e].data;
                if let Some(v) = &inner.constant {
                    if let Some(result) = negate_scalar(v) {
                        return ZarrAnalysisData::constant(result);
                    }
                }
                inner.clone()
            }

            // === Comparison expressions ===
            ZarrPlan::Eq([l, r])
            | ZarrPlan::Neq([l, r])
            | ZarrPlan::Lt([l, r])
            | ZarrPlan::Le([l, r])
            | ZarrPlan::Gt([l, r])
            | ZarrPlan::Ge([l, r]) => {
                let left = &egraph[*l].data;
                let right = &egraph[*r].data;

                // Try constant folding
                if let (Some(lv), Some(rv)) = (&left.constant, &right.constant) {
                    if let Some(result) = fold_comparison(enode, lv, rv) {
                        return ZarrAnalysisData::constant(ScalarValue::Boolean(Some(result)));
                    }
                }

                ZarrAnalysisData {
                    data_type: Some(DataType::Boolean),
                    nullable: merge_nullable(left.nullable, right.nullable),
                    ..Default::default()
                }
            }

            // === Logical expressions ===
            ZarrPlan::And([l, r]) => {
                let left = &egraph[*l].data;
                let right = &egraph[*r].data;

                if let (Some(lv), Some(rv)) = (&left.constant, &right.constant) {
                    if let (ScalarValue::Boolean(Some(lb)), ScalarValue::Boolean(Some(rb))) =
                        (lv, rv)
                    {
                        return ZarrAnalysisData::constant(ScalarValue::Boolean(Some(
                            *lb && *rb,
                        )));
                    }
                }

                ZarrAnalysisData {
                    data_type: Some(DataType::Boolean),
                    nullable: merge_nullable(left.nullable, right.nullable),
                    ..Default::default()
                }
            }

            ZarrPlan::Or([l, r]) => {
                let left = &egraph[*l].data;
                let right = &egraph[*r].data;

                if let (Some(lv), Some(rv)) = (&left.constant, &right.constant) {
                    if let (ScalarValue::Boolean(Some(lb)), ScalarValue::Boolean(Some(rb))) =
                        (lv, rv)
                    {
                        return ZarrAnalysisData::constant(ScalarValue::Boolean(Some(
                            *lb || *rb,
                        )));
                    }
                }

                ZarrAnalysisData {
                    data_type: Some(DataType::Boolean),
                    nullable: merge_nullable(left.nullable, right.nullable),
                    ..Default::default()
                }
            }

            ZarrPlan::Not([e]) => {
                let inner = &egraph[*e].data;
                if let Some(ScalarValue::Boolean(Some(b))) = &inner.constant {
                    return ZarrAnalysisData::constant(ScalarValue::Boolean(Some(!b)));
                }

                ZarrAnalysisData {
                    data_type: Some(DataType::Boolean),
                    nullable: inner.nullable,
                    ..Default::default()
                }
            }

            // === Aggregate functions ===
            ZarrPlan::Count(_) | ZarrPlan::CountDistinct(_) => ZarrAnalysisData {
                data_type: Some(DataType::Int64),
                nullable: Some(false), // COUNT never returns null
                ..Default::default()
            },

            ZarrPlan::Sum([e]) | ZarrPlan::Avg([e]) => {
                let inner = &egraph[*e].data;
                ZarrAnalysisData {
                    data_type: inner.data_type.clone(),
                    nullable: Some(true), // Could be null if no rows
                    ..Default::default()
                }
            }

            ZarrPlan::Min([e]) | ZarrPlan::Max([e]) => {
                let inner = &egraph[*e].data;
                // MIN/MAX preserve the type of their argument
                ZarrAnalysisData {
                    data_type: inner.data_type.clone(),
                    nullable: Some(true),
                    // Statistics-based values will be set via custom analysis
                    ..Default::default()
                }
            }

            // === Relational operators ===
            ZarrPlan::Scan([table, _proj]) => {
                let table_data = &egraph[*table].data;
                ZarrAnalysisData {
                    table_name: table_data.table_name.clone(),
                    cardinality: table_data.cardinality,
                    ..Default::default()
                }
            }

            ZarrPlan::Filter([input, _pred]) => {
                let input_data = &egraph[*input].data;
                ZarrAnalysisData {
                    // Estimate 50% selectivity (very rough)
                    cardinality: input_data.cardinality.map(|c| c / 2),
                    ..Default::default()
                }
            }

            ZarrPlan::Project([input, _exprs]) => {
                let input_data = &egraph[*input].data;
                ZarrAnalysisData {
                    cardinality: input_data.cardinality,
                    ..Default::default()
                }
            }

            ZarrPlan::Aggregate([input, _group, _aggs]) => {
                let input_data = &egraph[*input].data;
                // Without GROUP BY, aggregate produces 1 row
                // With GROUP BY, estimate based on distinct values (rough)
                ZarrAnalysisData {
                    cardinality: Some(input_data.cardinality.map(|c| c / 10).unwrap_or(1)),
                    ..Default::default()
                }
            }

            ZarrPlan::InnerJoin([l, r, _cond])
            | ZarrPlan::LeftJoin([l, r, _cond])
            | ZarrPlan::RightJoin([l, r, _cond])
            | ZarrPlan::FullJoin([l, r, _cond]) => {
                let left = &egraph[*l].data;
                let right = &egraph[*r].data;
                ZarrAnalysisData {
                    // Very rough: assume smaller side determines result
                    cardinality: match (left.cardinality, right.cardinality) {
                        (Some(l), Some(r)) => Some(l.min(r)),
                        (Some(c), None) | (None, Some(c)) => Some(c),
                        (None, None) => None,
                    },
                    ..Default::default()
                }
            }

            ZarrPlan::CrossJoin([l, r]) => {
                let left = &egraph[*l].data;
                let right = &egraph[*r].data;
                ZarrAnalysisData {
                    cardinality: match (left.cardinality, right.cardinality) {
                        (Some(l), Some(r)) => Some(l * r),
                        _ => None,
                    },
                    ..Default::default()
                }
            }

            ZarrPlan::Limit([input, count]) => {
                let input_data = &egraph[*input].data;
                let count_data = &egraph[*count].data;

                let limit = count_data
                    .constant
                    .as_ref()
                    .and_then(|v| match v {
                        ScalarValue::Int64(Some(n)) => Some(*n as usize),
                        _ => None,
                    })
                    .unwrap_or(usize::MAX);

                ZarrAnalysisData {
                    cardinality: input_data.cardinality.map(|c| c.min(limit)),
                    ..Default::default()
                }
            }

            ZarrPlan::Sort([input, _keys]) => {
                let input_data = &egraph[*input].data;
                ZarrAnalysisData {
                    cardinality: input_data.cardinality,
                    ..Default::default()
                }
            }

            ZarrPlan::Distinct([input]) => {
                let input_data = &egraph[*input].data;
                ZarrAnalysisData {
                    cardinality: input_data.cardinality,
                    ..Default::default()
                }
            }

            ZarrPlan::Union([l, r]) => {
                let left = &egraph[*l].data;
                let right = &egraph[*r].data;
                ZarrAnalysisData {
                    cardinality: match (left.cardinality, right.cardinality) {
                        (Some(l), Some(r)) => Some(l + r),
                        _ => None,
                    },
                    ..Default::default()
                }
            }

            // === Empty and lists ===
            ZarrPlan::Empty => ZarrAnalysisData {
                cardinality: Some(0),
                ..Default::default()
            },

            ZarrPlan::List(children) => {
                // List is just a container, aggregate child properties
                let mut result = ZarrAnalysisData::default();
                for &child in children.iter() {
                    let child_data = &egraph[child].data;
                    // Merge nullability
                    result.nullable = merge_nullable(result.nullable, child_data.nullable);
                }
                result
            }

            // Default for other nodes
            _ => ZarrAnalysisData::default(),
        }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        // Merge analysis data when e-classes are unified
        let mut changed = false;

        // Merge data type (prefer known type)
        if a.data_type.is_none() && b.data_type.is_some() {
            a.data_type = b.data_type;
            changed = true;
        }

        // Merge constant (prefer known constant)
        if a.constant.is_none() && b.constant.is_some() {
            a.constant = b.constant;
            changed = true;
        }

        // Merge cardinality (prefer smaller estimate for safety)
        match (&a.cardinality, &b.cardinality) {
            (None, Some(c)) => {
                a.cardinality = Some(*c);
                changed = true;
            }
            (Some(ac), Some(bc)) if bc < ac => {
                a.cardinality = Some(*bc);
                changed = true;
            }
            _ => {}
        }

        // Merge nullable (prefer known value)
        if a.nullable.is_none() && b.nullable.is_some() {
            a.nullable = b.nullable;
            changed = true;
        }

        // Merge statistics
        if a.min_value.is_none() && b.min_value.is_some() {
            a.min_value = b.min_value;
            changed = true;
        }
        if a.max_value.is_none() && b.max_value.is_some() {
            a.max_value = b.max_value;
            changed = true;
        }
        if a.row_count.is_none() && b.row_count.is_some() {
            a.row_count = b.row_count;
            changed = true;
        }

        // Merge names
        if a.column_name.is_none() && b.column_name.is_some() {
            a.column_name = b.column_name;
            changed = true;
        }
        if a.table_name.is_none() && b.table_name.is_some() {
            a.table_name = b.table_name;
            changed = true;
        }

        if changed {
            DidMerge(true, true)
        } else {
            DidMerge(false, false)
        }
    }
}

/// Fold arithmetic operations on constants
fn fold_arithmetic(op: &ZarrPlan, left: &ScalarValue, right: &ScalarValue) -> Option<ScalarValue> {
    match (left, right) {
        (ScalarValue::Int64(Some(l)), ScalarValue::Int64(Some(r))) => {
            let result = match op {
                ZarrPlan::Add(_) => l.checked_add(*r)?,
                ZarrPlan::Sub(_) => l.checked_sub(*r)?,
                ZarrPlan::Mul(_) => l.checked_mul(*r)?,
                ZarrPlan::Div(_) => {
                    if *r == 0 {
                        return None;
                    }
                    l.checked_div(*r)?
                }
                ZarrPlan::Mod(_) => {
                    if *r == 0 {
                        return None;
                    }
                    l.checked_rem(*r)?
                }
                _ => return None,
            };
            Some(ScalarValue::Int64(Some(result)))
        }
        (ScalarValue::Float64(Some(l)), ScalarValue::Float64(Some(r))) => {
            let result = match op {
                ZarrPlan::Add(_) => l + r,
                ZarrPlan::Sub(_) => l - r,
                ZarrPlan::Mul(_) => l * r,
                ZarrPlan::Div(_) => {
                    if *r == 0.0 {
                        return None;
                    }
                    l / r
                }
                ZarrPlan::Mod(_) => l % r,
                _ => return None,
            };
            Some(ScalarValue::Float64(Some(result)))
        }
        _ => None,
    }
}

/// Fold comparison operations on constants
fn fold_comparison(op: &ZarrPlan, left: &ScalarValue, right: &ScalarValue) -> Option<bool> {
    match (left, right) {
        (ScalarValue::Int64(Some(l)), ScalarValue::Int64(Some(r))) => match op {
            ZarrPlan::Eq(_) => Some(l == r),
            ZarrPlan::Neq(_) => Some(l != r),
            ZarrPlan::Lt(_) => Some(l < r),
            ZarrPlan::Le(_) => Some(l <= r),
            ZarrPlan::Gt(_) => Some(l > r),
            ZarrPlan::Ge(_) => Some(l >= r),
            _ => None,
        },
        (ScalarValue::Float64(Some(l)), ScalarValue::Float64(Some(r))) => match op {
            ZarrPlan::Eq(_) => Some(l == r),
            ZarrPlan::Neq(_) => Some(l != r),
            ZarrPlan::Lt(_) => Some(l < r),
            ZarrPlan::Le(_) => Some(l <= r),
            ZarrPlan::Gt(_) => Some(l > r),
            ZarrPlan::Ge(_) => Some(l >= r),
            _ => None,
        },
        (ScalarValue::Utf8(Some(l)), ScalarValue::Utf8(Some(r))) => match op {
            ZarrPlan::Eq(_) => Some(l == r),
            ZarrPlan::Neq(_) => Some(l != r),
            ZarrPlan::Lt(_) => Some(l < r),
            ZarrPlan::Le(_) => Some(l <= r),
            ZarrPlan::Gt(_) => Some(l > r),
            ZarrPlan::Ge(_) => Some(l >= r),
            _ => None,
        },
        _ => None,
    }
}

/// Negate a scalar value
fn negate_scalar(value: &ScalarValue) -> Option<ScalarValue> {
    match value {
        ScalarValue::Int64(Some(v)) => Some(ScalarValue::Int64(Some(-v))),
        ScalarValue::Float64(Some(v)) => Some(ScalarValue::Float64(Some(-v))),
        _ => None,
    }
}

/// Merge two numeric types to find the result type
fn merge_numeric_types(left: &Option<DataType>, right: &Option<DataType>) -> Option<DataType> {
    match (left, right) {
        (Some(DataType::Float64), _) | (_, Some(DataType::Float64)) => Some(DataType::Float64),
        (Some(DataType::Int64), _) | (_, Some(DataType::Int64)) => Some(DataType::Int64),
        (Some(dt), _) => Some(dt.clone()),
        (_, Some(dt)) => Some(dt.clone()),
        (None, None) => None,
    }
}

/// Merge nullable flags
fn merge_nullable(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        // If either side can be null, result can be null
        (Some(true), _) | (_, Some(true)) => Some(true),
        (Some(false), Some(false)) => Some(false),
        (Some(b), None) | (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding_int() {
        let left = ScalarValue::Int64(Some(5));
        let right = ScalarValue::Int64(Some(3));
        let add = ZarrPlan::Add([Id::from(0), Id::from(1)]);
        let result = fold_arithmetic(&add, &left, &right);
        assert_eq!(result, Some(ScalarValue::Int64(Some(8))));
    }

    #[test]
    fn test_constant_folding_float() {
        let left = ScalarValue::Float64(Some(2.5));
        let right = ScalarValue::Float64(Some(1.5));
        let mul = ZarrPlan::Mul([Id::from(0), Id::from(1)]);
        let result = fold_arithmetic(&mul, &left, &right);
        assert_eq!(result, Some(ScalarValue::Float64(Some(3.75))));
    }

    #[test]
    fn test_comparison_folding() {
        let left = ScalarValue::Int64(Some(5));
        let right = ScalarValue::Int64(Some(3));
        let lt = ZarrPlan::Lt([Id::from(0), Id::from(1)]);
        assert_eq!(fold_comparison(&lt, &left, &right), Some(false));
        assert_eq!(fold_comparison(&lt, &right, &left), Some(true));
    }

    #[test]
    fn test_analysis_data_constant() {
        let data = ZarrAnalysisData::constant(ScalarValue::Int64(Some(42)));
        assert!(data.is_constant());
        assert_eq!(
            data.get_constant(),
            Some(&ScalarValue::Int64(Some(42)))
        );
    }
}
