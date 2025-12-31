//! Custom optimizer rule to replace count aggregates with constants from statistics
//!
//! This rule transforms:
//!   `SELECT count(*) FROM table` → `SELECT 700`
//!   `SELECT count(col) FROM table` → `SELECT 700` (when null_count = 0)
//!   `SELECT count(*), count(col1), count(col2) FROM table` → `SELECT 700, 700, 700`
//!
//! The optimization is only applied when:
//! 1. All aggregate expressions are count() functions (no sum, avg, etc.)
//! 2. No GROUP BY clause
//! 3. No WHERE filters
//! 4. Statistics provide exact num_rows
//! 5. For count(col), the column has exact null_count

use datafusion::common::stats::Precision;
use datafusion::common::tree_node::Transformed;
use datafusion::common::Result;
use datafusion::datasource::source_as_provider;
use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::logical_expr::{EmptyRelation, Expr, LogicalPlan, Projection, TableScan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::OptimizerRule;
use datafusion::prelude::lit;
use std::sync::Arc;
use tracing::{debug, info, trace, warn};

/// Optimizer rule that replaces count() with constants from table statistics
#[derive(Debug, Default)]
pub struct CountStatisticsRule;

impl CountStatisticsRule {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizerRule for CountStatisticsRule {
    fn name(&self) -> &str {
        "count_statistics"
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
            "Evaluating Aggregate node for count optimization"
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

        // num_rows must be exact
        let num_rows: usize = match statistics.num_rows {
            Precision::Exact(n) => {
                debug!(table = %table_name, num_rows = n, "Got exact row count from statistics");
                n
            }
            Precision::Inexact(n) => {
                debug!(
                    table = %table_name,
                    approx_rows = n,
                    "Skipping: row count is inexact"
                );
                return Ok(Transformed::no(plan));
            }
            Precision::Absent => {
                debug!(table = %table_name, "Skipping: row count not available");
                return Ok(Transformed::no(plan));
            }
        };

        let schema = provider.schema();

        // Process all aggregate expressions - all must be count() functions
        let mut count_values: Vec<(String, i64)> = Vec::new();

        for aggr_expr in &aggregate.aggr_expr {
            match extract_count_info(aggr_expr) {
                Some((is_count_star, column_name)) => {
                    // Get the alias name for this expression
                    let alias_name = get_expr_alias(aggr_expr, &aggregate.schema);

                    let count_value = if is_count_star {
                        debug!(alias = %alias_name, "Found count(*)");
                        num_rows as i64
                    } else if let Some(ref col_name) = column_name {
                        // Find column index and get null count
                        let col_idx = schema.fields().iter().position(|f| f.name() == col_name);

                        if let Some(idx) = col_idx {
                            let col_stats = &statistics.column_statistics;
                            if idx < col_stats.len() {
                                match col_stats[idx].null_count {
                                    Precision::Exact(0) => {
                                        debug!(
                                            alias = %alias_name,
                                            column = %col_name,
                                            "Column has no nulls, count = num_rows"
                                        );
                                        num_rows as i64
                                    }
                                    Precision::Exact(nulls) => {
                                        let count = num_rows.saturating_sub(nulls) as i64;
                                        debug!(
                                            alias = %alias_name,
                                            column = %col_name,
                                            null_count = nulls,
                                            count,
                                            "Column has nulls, count = num_rows - null_count"
                                        );
                                        count
                                    }
                                    Precision::Inexact(nulls) => {
                                        debug!(
                                            column = %col_name,
                                            approx_nulls = nulls,
                                            "Skipping: null count is inexact"
                                        );
                                        return Ok(Transformed::no(plan));
                                    }
                                    Precision::Absent => {
                                        debug!(
                                            column = %col_name,
                                            "Skipping: null count not available"
                                        );
                                        return Ok(Transformed::no(plan));
                                    }
                                }
                            } else {
                                debug!(
                                    column = %col_name,
                                    "Skipping: column statistics not available"
                                );
                                return Ok(Transformed::no(plan));
                            }
                        } else {
                            debug!(column = %col_name, "Skipping: column not found in schema");
                            return Ok(Transformed::no(plan));
                        }
                    } else {
                        debug!("Skipping: column name not available");
                        return Ok(Transformed::no(plan));
                    };

                    count_values.push((alias_name, count_value));
                }
                None => {
                    // Not a count function - can't optimize this aggregate
                    debug!("Skipping: found non-count aggregate function");
                    return Ok(Transformed::no(plan));
                }
            }
        }

        info!(
            table = %table_name,
            count_expressions = count_values.len(),
            values = ?count_values,
            "Count optimization applied - replacing with constants"
        );

        create_multi_count_plan(&count_values)
    }
}

/// Extract count function info from an expression
/// Returns Some((is_count_star, column_name)) if it's a count function, None otherwise
fn extract_count_info(expr: &Expr) -> Option<(bool, Option<String>)> {
    let inner = unwrap_alias(expr);
    match inner {
        Expr::AggregateFunction(AggregateFunction { func, params }) => {
            if func.name() != "count" {
                trace!(func_name = %func.name(), "Not a count function");
                return None;
            }
            // Check if count(*) or count(column)
            match params.args.first() {
                Some(Expr::Literal(_, _)) => Some((true, None)), // count(1) or count(null)
                Some(Expr::Column(col)) => Some((false, Some(col.name.clone()))),
                None => Some((true, None)), // count() with no args = count(*)
                _ => {
                    debug!("Unsupported count argument type");
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

/// Get the alias name for an expression
fn get_expr_alias(expr: &Expr, _schema: &Arc<datafusion::common::DFSchema>) -> String {
    // Try to get alias from the expression itself
    if let Expr::Alias(alias) = expr {
        return alias.name.clone();
    }

    // Use the expression's schema name (e.g., "count(*)" or "count(col)")
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
/// Returns None if there's a Filter (count would change after filtering)
fn unwrap_to_table_scan(plan: &LogicalPlan) -> Option<&TableScan> {
    match plan {
        LogicalPlan::TableScan(scan) => Some(scan),
        LogicalPlan::Projection(Projection { input, .. }) => unwrap_to_table_scan(input),
        LogicalPlan::SubqueryAlias(alias) => unwrap_to_table_scan(&alias.input),
        LogicalPlan::Filter(_) => {
            // Filter changes the count - can't optimize
            debug!("Found Filter node - count would change after filtering");
            None
        }
        other => {
            debug!(node_type = %other.display(), "Unsupported node type in input");
            None
        }
    }
}

/// Create a new plan that returns multiple count values as constants
fn create_multi_count_plan(count_values: &[(String, i64)]) -> Result<Transformed<LogicalPlan>> {
    // Create literal expressions for each count value with their aliases
    let exprs: Vec<Expr> = count_values
        .iter()
        .map(|(alias, value)| lit(*value).alias(alias))
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
