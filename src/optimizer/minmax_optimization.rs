//! Custom optimizer rule to replace MIN/MAX aggregates with constants from statistics
//!
//! This rule transforms:
//!   `SELECT MIN(coord) FROM table` → `SELECT 0` (from statistics)
//!   `SELECT MAX(coord) FROM table` → `SELECT 90` (from statistics)
//!   `SELECT MIN(lat), MAX(lat), MIN(lon) FROM table` → `SELECT 0, 90, 0`
//!
//! The optimization is only applied when:
//! 1. All aggregate expressions are MIN() or MAX() functions
//! 2. No GROUP BY clause
//! 3. No WHERE filters
//! 4. Statistics provide exact min_value/max_value for the column

use datafusion::common::stats::Precision;
use datafusion::common::tree_node::Transformed;
use datafusion::common::{Result, ScalarValue};
use datafusion::datasource::source_as_provider;
use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::logical_expr::{EmptyRelation, Expr, LogicalPlan, Projection, TableScan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::OptimizerRule;
use datafusion::prelude::lit;
use std::sync::Arc;
use tracing::{debug, info, trace, warn};

/// Optimizer rule that replaces MIN()/MAX() with constants from table statistics
#[derive(Debug, Default)]
pub struct MinMaxStatisticsRule;

impl MinMaxStatisticsRule {
    pub fn new() -> Self {
        Self
    }
}

/// Type of aggregate function (MIN or MAX)
#[derive(Debug, Clone, Copy)]
enum MinMaxType {
    Min,
    Max,
}

impl OptimizerRule for MinMaxStatisticsRule {
    fn name(&self) -> &str {
        "minmax_statistics"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn datafusion::optimizer::OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Only optimize Aggregate nodes
        let LogicalPlan::Aggregate(aggregate) = &plan else {
            trace!("Skipping non-Aggregate node");
            return Ok(Transformed::no(plan));
        };

        debug!(
            group_by_count = aggregate.group_expr.len(),
            aggr_count = aggregate.aggr_expr.len(),
            "Evaluating Aggregate node for MIN/MAX optimization"
        );

        // Must have no GROUP BY (simple aggregate)
        if !aggregate.group_expr.is_empty() {
            debug!(
                group_by_count = aggregate.group_expr.len(),
                "Skipping: has GROUP BY clause"
            );
            return Ok(Transformed::no(plan));
        }

        // Must have at least one aggregate expression
        if aggregate.aggr_expr.is_empty() {
            debug!("Skipping: no aggregate expressions");
            return Ok(Transformed::no(plan));
        }

        // Input must be a simple TableScan with no filters
        let input = aggregate.input.as_ref();
        let table_scan = match unwrap_to_table_scan(input) {
            Some(scan) => scan,
            None => {
                debug!("Skipping: could not find TableScan in input");
                return Ok(Transformed::no(plan));
            }
        };

        let table_name = table_scan.table_name.to_string();
        debug!(table = %table_name, "Found TableScan");

        // Must have no filters
        if !table_scan.filters.is_empty() {
            debug!(
                table = %table_name,
                filter_count = table_scan.filters.len(),
                "Skipping: query has WHERE filters"
            );
            return Ok(Transformed::no(plan));
        }

        // Get TableProvider from TableSource to access statistics
        let provider = match source_as_provider(&table_scan.source) {
            Ok(p) => p,
            Err(e) => {
                warn!(
                    table = %table_name,
                    error = %e,
                    "Could not get TableProvider from TableSource"
                );
                return Ok(Transformed::no(plan));
            }
        };

        // Get statistics from the table
        let statistics = match provider.statistics() {
            Some(stats) => stats,
            None => {
                debug!(
                    table = %table_name,
                    "Skipping: table does not provide statistics"
                );
                return Ok(Transformed::no(plan));
            }
        };

        let schema = provider.schema();

        // Process all aggregate expressions - all must be MIN() or MAX() functions
        let mut minmax_values: Vec<(String, ScalarValue)> = Vec::new();

        for aggr_expr in &aggregate.aggr_expr {
            match extract_minmax_info(aggr_expr, input) {
                Some((minmax_type, column_name)) => {
                    // Get the alias name for this expression
                    let alias_name = get_expr_alias(aggr_expr, &aggregate.schema);

                    // Find column index to get statistics
                    let col_idx = schema.fields().iter().position(|f| f.name() == &column_name);

                    if let Some(idx) = col_idx {
                        let col_stats = &statistics.column_statistics;
                        if idx < col_stats.len() {
                            let value = match minmax_type {
                                MinMaxType::Min => match &col_stats[idx].min_value {
                                    Precision::Exact(v) => {
                                        debug!(
                                            alias = %alias_name,
                                            column = %column_name,
                                            value = %v,
                                            "Found exact MIN value from statistics"
                                        );
                                        v.clone()
                                    }
                                    Precision::Inexact(_) => {
                                        debug!(
                                            column = %column_name,
                                            "Skipping: min value is inexact"
                                        );
                                        return Ok(Transformed::no(plan));
                                    }
                                    Precision::Absent => {
                                        debug!(
                                            column = %column_name,
                                            "Skipping: min value not available"
                                        );
                                        return Ok(Transformed::no(plan));
                                    }
                                },
                                MinMaxType::Max => match &col_stats[idx].max_value {
                                    Precision::Exact(v) => {
                                        debug!(
                                            alias = %alias_name,
                                            column = %column_name,
                                            value = %v,
                                            "Found exact MAX value from statistics"
                                        );
                                        v.clone()
                                    }
                                    Precision::Inexact(_) => {
                                        debug!(
                                            column = %column_name,
                                            "Skipping: max value is inexact"
                                        );
                                        return Ok(Transformed::no(plan));
                                    }
                                    Precision::Absent => {
                                        debug!(
                                            column = %column_name,
                                            "Skipping: max value not available"
                                        );
                                        return Ok(Transformed::no(plan));
                                    }
                                },
                            };

                            minmax_values.push((alias_name, value));
                        } else {
                            debug!(
                                column = %column_name,
                                "Skipping: column statistics not available"
                            );
                            return Ok(Transformed::no(plan));
                        }
                    } else {
                        debug!(column = %column_name, "Skipping: column not found in schema");
                        return Ok(Transformed::no(plan));
                    }
                }
                None => {
                    // Not a MIN/MAX function - can't optimize this aggregate
                    debug!("Skipping: found non-MIN/MAX aggregate function");
                    return Ok(Transformed::no(plan));
                }
            }
        }

        info!(
            table = %table_name,
            minmax_expressions = minmax_values.len(),
            values = ?minmax_values,
            "MIN/MAX optimization applied - replacing with constants"
        );

        create_minmax_plan(&minmax_values)
    }
}

/// Extract MIN/MAX function info from an expression
/// Returns Some((MinMaxType, column_name)) if it's a MIN or MAX function, None otherwise
fn extract_minmax_info(expr: &Expr, input_plan: &LogicalPlan) -> Option<(MinMaxType, String)> {
    let inner = unwrap_alias(expr);
    match inner {
        Expr::AggregateFunction(AggregateFunction { func, params }) => {
            let func_name = func.name().to_lowercase();
            let minmax_type = match func_name.as_str() {
                "min" => MinMaxType::Min,
                "max" => MinMaxType::Max,
                _ => {
                    trace!(func_name = %func_name, "Not a MIN/MAX function");
                    return None;
                }
            };

            // Extract column name from arguments
            // Need to handle: Column, Alias(Column), Cast(Column), Alias(Cast(Column))
            // Also handle __common_expr_N which needs to be traced back
            let arg = params.args.first()?;
            let inner_arg = unwrap_alias(arg);

            match inner_arg {
                Expr::Column(col) => {
                    // Check if this is a __common_expr reference that needs tracing
                    if col.name.starts_with("__common_expr") {
                        trace_column_in_plan(&col.name, input_plan, minmax_type)
                    } else {
                        Some((minmax_type, col.name.clone()))
                    }
                }
                Expr::Cast(cast) => {
                    // Dictionary columns get cast to their value type
                    if let Expr::Column(col) = unwrap_alias(cast.expr.as_ref()) {
                        return Some((minmax_type, col.name.clone()));
                    }
                    debug!(cast_expr_type = %cast.expr.variant_name(), "MIN/MAX cast argument is not a column");
                    None
                }
                other => {
                    debug!(expr_type = %other.variant_name(), "MIN/MAX argument is not a column or cast");
                    None
                }
            }
        }
        _ => {
            trace!("Not an AggregateFunction expression");
            None
        }
    }
}

/// Trace a __common_expr column back to its original column through projections
fn trace_column_in_plan(
    expr_name: &str,
    plan: &LogicalPlan,
    minmax_type: MinMaxType,
) -> Option<(MinMaxType, String)> {
    match plan {
        LogicalPlan::Projection(proj) => {
            // Find the expression that defines this common expr
            for proj_expr in &proj.expr {
                if let Expr::Alias(alias) = proj_expr {
                    if alias.name == expr_name {
                        // Found the definition, extract column from the inner expression
                        let inner = unwrap_alias(&alias.expr);
                        if let Expr::Cast(cast) = inner {
                            if let Expr::Column(col) = unwrap_alias(cast.expr.as_ref()) {
                                debug!(
                                    common_expr = %expr_name,
                                    original_col = %col.name,
                                    "Traced common expression to original column"
                                );
                                return Some((minmax_type, col.name.clone()));
                            }
                        } else if let Expr::Column(col) = inner {
                            debug!(
                                common_expr = %expr_name,
                                original_col = %col.name,
                                "Traced common expression to original column"
                            );
                            return Some((minmax_type, col.name.clone()));
                        }
                    }
                }
            }
            // Try in input plan
            trace_column_in_plan(expr_name, &proj.input, minmax_type)
        }
        LogicalPlan::TableScan(_) => None,
        LogicalPlan::SubqueryAlias(alias) => {
            trace_column_in_plan(expr_name, &alias.input, minmax_type)
        }
        other => {
            // Try to trace through other plan types
            for input in other.inputs() {
                if let Some(result) = trace_column_in_plan(expr_name, input, minmax_type) {
                    return Some(result);
                }
            }
            None
        }
    }
}

/// Get the alias name for an expression
fn get_expr_alias(expr: &Expr, _schema: &Arc<datafusion::common::DFSchema>) -> String {
    // Try to get alias from the expression itself
    if let Expr::Alias(alias) = expr {
        return alias.name.clone();
    }

    // Use the expression's schema name (e.g., "min(lat)" or "max(lon)")
    expr.schema_name().to_string()
}

/// Unwrap Alias expressions to get the inner expression
fn unwrap_alias(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(alias) => unwrap_alias(&alias.expr),
        other => other,
    }
}

/// Unwrap projections to find the underlying TableScan
/// Returns None if there's a Filter (min/max would change after filtering)
fn unwrap_to_table_scan(plan: &LogicalPlan) -> Option<&TableScan> {
    match plan {
        LogicalPlan::TableScan(scan) => Some(scan),
        LogicalPlan::Projection(Projection { input, .. }) => unwrap_to_table_scan(input),
        LogicalPlan::SubqueryAlias(alias) => unwrap_to_table_scan(&alias.input),
        LogicalPlan::Filter(_) => {
            // Filter changes the min/max - can't optimize
            debug!("Found Filter node - min/max would change after filtering");
            None
        }
        other => {
            debug!(node_type = %other.display(), "Unsupported node type in input");
            None
        }
    }
}

/// Create a new plan that returns MIN/MAX values as constants
fn create_minmax_plan(
    minmax_values: &[(String, ScalarValue)],
) -> Result<Transformed<LogicalPlan>> {
    // Create literal expressions for each value with their aliases
    let exprs: Vec<Expr> = minmax_values
        .iter()
        .map(|(alias, value)| lit(value.clone()).alias(alias))
        .collect();

    // Create an empty relation (no input needed)
    let empty = LogicalPlan::EmptyRelation(EmptyRelation {
        produce_one_row: true,
        schema: Arc::new(arrow::datatypes::Schema::empty().try_into()?),
    });

    // Create projection with all the constants
    let projection = LogicalPlan::Projection(Projection::try_new(exprs, Arc::new(empty))?);

    Ok(Transformed::yes(projection))
}
