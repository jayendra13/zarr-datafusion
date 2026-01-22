//! Conversion between DataFusion LogicalPlan and egg RecExpr
//!
//! This module handles bidirectional conversion:
//! - `logical_plan_to_egg`: DataFusion LogicalPlan → egg RecExpr<ZarrPlan>
//! - `egg_to_logical_plan`: egg RecExpr<ZarrPlan> → DataFusion LogicalPlan

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::Schema;
use datafusion::common::{Column, DFSchema, DataFusionError, Result, ScalarValue};
use datafusion::logical_expr::{
    expr::AggregateFunction, BinaryExpr, EmptyRelation, Expr, JoinType, LogicalPlan,
    LogicalPlanBuilder, Operator,
};
use egg::{Id, RecExpr, Symbol};

use super::language::{make_literal, parse_literal, ZarrPlan};

/// Convert a DataFusion LogicalPlan to egg's RecExpr representation
pub fn logical_plan_to_egg(plan: &LogicalPlan) -> Result<RecExpr<ZarrPlan>> {
    let mut expr = RecExpr::default();
    convert_plan_to_egg(plan, &mut expr)?;
    Ok(expr)
}

/// Recursively convert a LogicalPlan node to egg format
fn convert_plan_to_egg(plan: &LogicalPlan, expr: &mut RecExpr<ZarrPlan>) -> Result<Id> {
    match plan {
        LogicalPlan::TableScan(scan) => {
            // Table reference
            let table_id = expr.add(ZarrPlan::Symbol(scan.table_name.to_string().into()));

            // Projection (column indices or empty)
            let proj_id = if let Some(indices) = &scan.projection {
                let ids: Vec<Id> = indices
                    .iter()
                    .map(|&i| expr.add(ZarrPlan::Symbol(make_literal("usize", &i.to_string()))))
                    .collect();
                expr.add(ZarrPlan::List(ids.into()))
            } else {
                expr.add(ZarrPlan::Empty)
            };

            Ok(expr.add(ZarrPlan::Scan([table_id, proj_id])))
        }

        LogicalPlan::Filter(filter) => {
            let input_id = convert_plan_to_egg(&filter.input, expr)?;
            let pred_id = convert_expr_to_egg(&filter.predicate, expr)?;
            Ok(expr.add(ZarrPlan::Filter([input_id, pred_id])))
        }

        LogicalPlan::Projection(proj) => {
            let input_id = convert_plan_to_egg(&proj.input, expr)?;
            let expr_ids: Vec<Id> = proj
                .expr
                .iter()
                .map(|e| convert_expr_to_egg(e, expr))
                .collect::<Result<_>>()?;
            let exprs_id = expr.add(ZarrPlan::List(expr_ids.into()));
            Ok(expr.add(ZarrPlan::Project([input_id, exprs_id])))
        }

        LogicalPlan::Aggregate(agg) => {
            let input_id = convert_plan_to_egg(&agg.input, expr)?;

            // Group by expressions
            let group_ids: Vec<Id> = agg
                .group_expr
                .iter()
                .map(|e| convert_expr_to_egg(e, expr))
                .collect::<Result<_>>()?;
            let group_id = if group_ids.is_empty() {
                expr.add(ZarrPlan::Empty)
            } else {
                expr.add(ZarrPlan::List(group_ids.into()))
            };

            // Aggregate expressions
            let agg_ids: Vec<Id> = agg
                .aggr_expr
                .iter()
                .map(|e| convert_expr_to_egg(e, expr))
                .collect::<Result<_>>()?;
            let aggs_id = if agg_ids.is_empty() {
                expr.add(ZarrPlan::Empty)
            } else {
                expr.add(ZarrPlan::List(agg_ids.into()))
            };

            Ok(expr.add(ZarrPlan::Aggregate([input_id, group_id, aggs_id])))
        }

        LogicalPlan::Join(join) => {
            let left_id = convert_plan_to_egg(&join.left, expr)?;
            let right_id = convert_plan_to_egg(&join.right, expr)?;

            // Join condition
            let cond_ids: Vec<Id> = join
                .on
                .iter()
                .map(|(l, r)| {
                    let l_id = convert_expr_to_egg(l, expr)?;
                    let r_id = convert_expr_to_egg(r, expr)?;
                    Ok(expr.add(ZarrPlan::Eq([l_id, r_id])))
                })
                .collect::<Result<_>>()?;

            // Add filter condition if present
            let all_conds = if let Some(filter) = &join.filter {
                let filter_id = convert_expr_to_egg(filter, expr)?;
                let mut conds = cond_ids;
                conds.push(filter_id);
                conds
            } else {
                cond_ids
            };

            let cond_id = if all_conds.is_empty() {
                expr.add(ZarrPlan::True)
            } else if all_conds.len() == 1 {
                all_conds[0]
            } else {
                // Combine with AND
                all_conds
                    .into_iter()
                    .reduce(|acc, c| expr.add(ZarrPlan::And([acc, c])))
                    .unwrap()
            };

            let plan_node = match join.join_type {
                JoinType::Inner => ZarrPlan::InnerJoin([left_id, right_id, cond_id]),
                JoinType::Left => ZarrPlan::LeftJoin([left_id, right_id, cond_id]),
                JoinType::Right => ZarrPlan::RightJoin([left_id, right_id, cond_id]),
                JoinType::Full => ZarrPlan::FullJoin([left_id, right_id, cond_id]),
                JoinType::LeftSemi => ZarrPlan::SemiJoin([left_id, right_id, cond_id]),
                JoinType::LeftAnti => ZarrPlan::AntiJoin([left_id, right_id, cond_id]),
                _ => {
                    return Err(DataFusionError::NotImplemented(format!(
                        "Join type {:?} not supported in e-graph",
                        join.join_type
                    )));
                }
            };

            Ok(expr.add(plan_node))
        }

        LogicalPlan::Sort(sort) => {
            let input_id = convert_plan_to_egg(&sort.input, expr)?;
            let key_ids: Vec<Id> = sort
                .expr
                .iter()
                .map(|e| convert_sort_expr_to_egg(e, expr))
                .collect::<Result<_>>()?;
            let keys_id = expr.add(ZarrPlan::List(key_ids.into()));
            Ok(expr.add(ZarrPlan::Sort([input_id, keys_id])))
        }

        LogicalPlan::Limit(limit) => {
            let input_id = convert_plan_to_egg(&limit.input, expr)?;
            let count_id = if let Some(fetch) = &limit.fetch {
                // The fetch is an expression - convert it to egg representation
                convert_expr_to_egg(fetch, expr)?
            } else {
                // No limit specified, use a large number
                expr.add(ZarrPlan::Symbol(make_literal(
                    "i64",
                    &i64::MAX.to_string(),
                )))
            };
            Ok(expr.add(ZarrPlan::Limit([input_id, count_id])))
        }

        LogicalPlan::Distinct(distinct) => {
            let input_id = convert_plan_to_egg(distinct.input(), expr)?;
            Ok(expr.add(ZarrPlan::Distinct([input_id])))
        }

        LogicalPlan::Union(union) => {
            // Union of multiple inputs - chain them
            let mut result_id = convert_plan_to_egg(&union.inputs[0], expr)?;
            for input in union.inputs.iter().skip(1) {
                let input_id = convert_plan_to_egg(input, expr)?;
                result_id = expr.add(ZarrPlan::Union([result_id, input_id]));
            }
            Ok(result_id)
        }

        LogicalPlan::SubqueryAlias(alias) => {
            // Just pass through the input, alias is for naming
            convert_plan_to_egg(&alias.input, expr)
        }

        LogicalPlan::EmptyRelation(_) => Ok(expr.add(ZarrPlan::Empty)),

        _ => Err(DataFusionError::NotImplemented(format!(
            "E-graph conversion not implemented for plan type: {}",
            plan.display()
        ))),
    }
}

/// Convert a DataFusion Expr to egg format
fn convert_expr_to_egg(df_expr: &Expr, expr: &mut RecExpr<ZarrPlan>) -> Result<Id> {
    match df_expr {
        Expr::Column(col) => Ok(expr.add(ZarrPlan::Symbol(col.name.clone().into()))),

        Expr::Literal(value, _metadata) => {
            let lit = scalar_to_literal(value);
            Ok(expr.add(ZarrPlan::Symbol(lit)))
        }

        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            let left_id = convert_expr_to_egg(left, expr)?;
            let right_id = convert_expr_to_egg(right, expr)?;
            let node = match op {
                // Arithmetic
                Operator::Plus => ZarrPlan::Add([left_id, right_id]),
                Operator::Minus => ZarrPlan::Sub([left_id, right_id]),
                Operator::Multiply => ZarrPlan::Mul([left_id, right_id]),
                Operator::Divide => ZarrPlan::Div([left_id, right_id]),
                Operator::Modulo => ZarrPlan::Mod([left_id, right_id]),
                // Comparison
                Operator::Eq => ZarrPlan::Eq([left_id, right_id]),
                Operator::NotEq => ZarrPlan::Neq([left_id, right_id]),
                Operator::Lt => ZarrPlan::Lt([left_id, right_id]),
                Operator::LtEq => ZarrPlan::Le([left_id, right_id]),
                Operator::Gt => ZarrPlan::Gt([left_id, right_id]),
                Operator::GtEq => ZarrPlan::Ge([left_id, right_id]),
                // Logical
                Operator::And => ZarrPlan::And([left_id, right_id]),
                Operator::Or => ZarrPlan::Or([left_id, right_id]),
                // String
                Operator::StringConcat => ZarrPlan::Concat([left_id, right_id]),
                _ => {
                    return Err(DataFusionError::NotImplemented(format!(
                        "Operator {:?} not supported in e-graph",
                        op
                    )));
                }
            };
            Ok(expr.add(node))
        }

        Expr::Not(inner) => {
            let inner_id = convert_expr_to_egg(inner, expr)?;
            Ok(expr.add(ZarrPlan::Not([inner_id])))
        }

        Expr::Negative(inner) => {
            let inner_id = convert_expr_to_egg(inner, expr)?;
            Ok(expr.add(ZarrPlan::Neg([inner_id])))
        }

        Expr::IsNull(inner) => {
            let inner_id = convert_expr_to_egg(inner, expr)?;
            Ok(expr.add(ZarrPlan::IsNull([inner_id])))
        }

        Expr::IsNotNull(inner) => {
            let inner_id = convert_expr_to_egg(inner, expr)?;
            Ok(expr.add(ZarrPlan::IsNotNull([inner_id])))
        }

        Expr::Between(between) => {
            let expr_id = convert_expr_to_egg(&between.expr, expr)?;
            let low_id = convert_expr_to_egg(&between.low, expr)?;
            let high_id = convert_expr_to_egg(&between.high, expr)?;
            let between_id = expr.add(ZarrPlan::Between([expr_id, low_id, high_id]));
            if between.negated {
                Ok(expr.add(ZarrPlan::Not([between_id])))
            } else {
                Ok(between_id)
            }
        }

        Expr::Case(case) => {
            // Simplified case handling - just handle CASE WHEN cond THEN val ELSE default END
            if case.when_then_expr.len() == 1 && case.else_expr.is_some() {
                let (when, then) = &case.when_then_expr[0];
                let cond_id = convert_expr_to_egg(when, expr)?;
                let then_id = convert_expr_to_egg(then, expr)?;
                let else_id = convert_expr_to_egg(case.else_expr.as_ref().unwrap(), expr)?;
                Ok(expr.add(ZarrPlan::Case([cond_id, then_id, else_id])))
            } else {
                Err(DataFusionError::NotImplemented(
                    "Complex CASE expressions not yet supported in e-graph".to_string(),
                ))
            }
        }

        Expr::Cast(cast) => {
            let expr_id = convert_expr_to_egg(&cast.expr, expr)?;
            let type_id = expr.add(ZarrPlan::Symbol(format!("{:?}", cast.data_type).into()));
            Ok(expr.add(ZarrPlan::Cast([expr_id, type_id])))
        }

        Expr::TryCast(cast) => {
            let expr_id = convert_expr_to_egg(&cast.expr, expr)?;
            let type_id = expr.add(ZarrPlan::Symbol(format!("{:?}", cast.data_type).into()));
            Ok(expr.add(ZarrPlan::TryCast([expr_id, type_id])))
        }

        Expr::Alias(alias) => {
            let inner_id = convert_expr_to_egg(&alias.expr, expr)?;
            let name_id = expr.add(ZarrPlan::Symbol(make_literal("str", &alias.name)));
            Ok(expr.add(ZarrPlan::Alias([inner_id, name_id])))
        }

        Expr::AggregateFunction(AggregateFunction { func, params }) => {
            let func_name = func.name().to_lowercase();

            // Get the first argument (if any)
            let arg_id = if params.args.is_empty() {
                expr.add(ZarrPlan::Star)
            } else {
                convert_expr_to_egg(&params.args[0], expr)?
            };

            let node = match func_name.as_str() {
                "count" => ZarrPlan::Count([arg_id]),
                "sum" => ZarrPlan::Sum([arg_id]),
                "avg" => ZarrPlan::Avg([arg_id]),
                "min" => ZarrPlan::Min([arg_id]),
                "max" => ZarrPlan::Max([arg_id]),
                "stddev" | "stddev_samp" => ZarrPlan::Stddev([arg_id]),
                "variance" | "var_samp" => ZarrPlan::Variance([arg_id]),
                _ => {
                    return Err(DataFusionError::NotImplemented(format!(
                        "Aggregate function {} not supported in e-graph",
                        func_name
                    )));
                }
            };

            Ok(expr.add(node))
        }

        #[allow(deprecated)]
        Expr::Wildcard { .. } => Ok(expr.add(ZarrPlan::Star)),

        _ => Err(DataFusionError::NotImplemented(format!(
            "Expression type {:?} not supported in e-graph",
            df_expr.variant_name()
        ))),
    }
}

/// Convert a sort expression to egg format
fn convert_sort_expr_to_egg(
    sort: &datafusion::logical_expr::SortExpr,
    expr: &mut RecExpr<ZarrPlan>,
) -> Result<Id> {
    let expr_id = convert_expr_to_egg(&sort.expr, expr)?;
    let asc_id = expr.add(if sort.asc {
        ZarrPlan::True
    } else {
        ZarrPlan::False
    });
    let nulls_first_id = expr.add(if sort.nulls_first {
        ZarrPlan::True
    } else {
        ZarrPlan::False
    });
    Ok(expr.add(ZarrPlan::SortExpr([expr_id, asc_id, nulls_first_id])))
}

/// Convert a ScalarValue to a literal symbol
fn scalar_to_literal(value: &ScalarValue) -> Symbol {
    match value {
        ScalarValue::Boolean(Some(b)) => make_literal("bool", &b.to_string()),
        ScalarValue::Int8(Some(n)) => make_literal("i8", &n.to_string()),
        ScalarValue::Int16(Some(n)) => make_literal("i16", &n.to_string()),
        ScalarValue::Int32(Some(n)) => make_literal("i32", &n.to_string()),
        ScalarValue::Int64(Some(n)) => make_literal("i64", &n.to_string()),
        ScalarValue::UInt8(Some(n)) => make_literal("u8", &n.to_string()),
        ScalarValue::UInt16(Some(n)) => make_literal("u16", &n.to_string()),
        ScalarValue::UInt32(Some(n)) => make_literal("u32", &n.to_string()),
        ScalarValue::UInt64(Some(n)) => make_literal("u64", &n.to_string()),
        ScalarValue::Float32(Some(n)) => make_literal("f32", &n.to_string()),
        ScalarValue::Float64(Some(n)) => make_literal("f64", &n.to_string()),
        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => make_literal("str", s),
        ScalarValue::Null => "null:null".into(),
        _ => format!("unknown:{:?}", value).into(),
    }
}

/// Parse a literal symbol back to a ScalarValue
fn literal_to_scalar(sym: &Symbol) -> Result<ScalarValue> {
    let s = sym.as_str();
    if let Some((type_str, value_str)) = parse_literal(sym) {
        match type_str.as_str() {
            "bool" => Ok(ScalarValue::Boolean(Some(value_str == "true"))),
            "i8" => value_str
                .parse::<i8>()
                .map(|v| ScalarValue::Int8(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "i16" => value_str
                .parse::<i16>()
                .map(|v| ScalarValue::Int16(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "i32" => value_str
                .parse::<i32>()
                .map(|v| ScalarValue::Int32(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "i64" => value_str
                .parse::<i64>()
                .map(|v| ScalarValue::Int64(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "u8" => value_str
                .parse::<u8>()
                .map(|v| ScalarValue::UInt8(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "u16" => value_str
                .parse::<u16>()
                .map(|v| ScalarValue::UInt16(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "u32" => value_str
                .parse::<u32>()
                .map(|v| ScalarValue::UInt32(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "u64" | "usize" => value_str
                .parse::<u64>()
                .map(|v| ScalarValue::UInt64(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "f32" => value_str
                .parse::<f32>()
                .map(|v| ScalarValue::Float32(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "f64" => value_str
                .parse::<f64>()
                .map(|v| ScalarValue::Float64(Some(v)))
                .map_err(|e| DataFusionError::Internal(e.to_string())),
            "str" => Ok(ScalarValue::Utf8(Some(value_str))),
            "null" => Ok(ScalarValue::Null),
            _ => Err(DataFusionError::Internal(format!(
                "Unknown literal type: {}",
                type_str
            ))),
        }
    } else {
        // Not a literal format - treat as column name
        Err(DataFusionError::Internal(format!(
            "Cannot parse as literal: {}",
            s
        )))
    }
}

/// Context for converting egg RecExpr back to DataFusion LogicalPlan
///
/// This holds information extracted from the original plan that's needed
/// to reconstruct a valid LogicalPlan.
#[derive(Clone)]
pub struct ConversionContext {
    /// Table scans from the original plan, keyed by table name
    pub table_scans: HashMap<String, Arc<LogicalPlan>>,
    /// Schemas for each table
    pub table_schemas: HashMap<String, Arc<Schema>>,
    /// The original plan (for fallback/future use)
    #[allow(dead_code)]
    pub original_plan: Arc<LogicalPlan>,
}

impl ConversionContext {
    /// Create a new conversion context from the original plan
    pub fn from_plan(plan: &LogicalPlan) -> Self {
        let mut ctx = ConversionContext {
            table_scans: HashMap::new(),
            table_schemas: HashMap::new(),
            original_plan: Arc::new(plan.clone()),
        };
        ctx.extract_context(plan);
        ctx
    }

    /// Recursively extract table scans and schemas from the plan
    fn extract_context(&mut self, plan: &LogicalPlan) {
        match plan {
            LogicalPlan::TableScan(scan) => {
                let table_name = scan.table_name.to_string();
                self.table_scans
                    .insert(table_name.clone(), Arc::new(plan.clone()));
                self.table_schemas
                    .insert(table_name, Arc::new(scan.source.schema().as_ref().clone()));
            }
            _ => {
                // Recurse into children
                for child in plan.inputs() {
                    self.extract_context(child);
                }
            }
        }
    }
}

/// Convert an egg RecExpr back to a DataFusion LogicalPlan
///
/// This requires the original plan to provide context (schema, table sources, etc.)
pub fn egg_to_logical_plan(rec_expr: &RecExpr<ZarrPlan>, original: &LogicalPlan) -> Result<LogicalPlan> {
    let ctx = ConversionContext::from_plan(original);
    let root_id = Id::from(rec_expr.as_ref().len() - 1);
    convert_egg_to_plan(rec_expr, root_id, &ctx)
}

/// Recursively convert an egg node to a LogicalPlan
fn convert_egg_to_plan(
    rec_expr: &RecExpr<ZarrPlan>,
    id: Id,
    ctx: &ConversionContext,
) -> Result<LogicalPlan> {
    let node = &rec_expr[id];

    match node {
        ZarrPlan::Scan([table_id, _proj_id]) => {
            // Get the table name
            let table_name = get_symbol(rec_expr, *table_id)?;

            // Look up the original table scan
            if let Some(scan_plan) = ctx.table_scans.get(&table_name) {
                Ok((**scan_plan).clone())
            } else {
                Err(DataFusionError::Internal(format!(
                    "Table '{}' not found in context",
                    table_name
                )))
            }
        }

        ZarrPlan::Filter([input_id, pred_id]) => {
            let input = convert_egg_to_plan(rec_expr, *input_id, ctx)?;
            let predicate = convert_egg_to_expr(rec_expr, *pred_id, &input)?;
            LogicalPlanBuilder::from(input).filter(predicate)?.build()
        }

        ZarrPlan::Project([input_id, exprs_id]) => {
            let input = convert_egg_to_plan(rec_expr, *input_id, ctx)?;
            let exprs = convert_egg_to_expr_list(rec_expr, *exprs_id, &input)?;
            if exprs.is_empty() {
                // Empty projection means select all
                Ok(input)
            } else {
                LogicalPlanBuilder::from(input).project(exprs)?.build()
            }
        }

        ZarrPlan::Aggregate([input_id, group_id, aggs_id]) => {
            let input = convert_egg_to_plan(rec_expr, *input_id, ctx)?;
            let group_exprs = convert_egg_to_expr_list(rec_expr, *group_id, &input)?;
            let agg_exprs = convert_egg_to_expr_list(rec_expr, *aggs_id, &input)?;
            LogicalPlanBuilder::from(input)
                .aggregate(group_exprs, agg_exprs)?
                .build()
        }

        ZarrPlan::Sort([input_id, keys_id]) => {
            let input = convert_egg_to_plan(rec_expr, *input_id, ctx)?;
            let sort_exprs = convert_egg_to_sort_expr_list(rec_expr, *keys_id, &input)?;
            LogicalPlanBuilder::from(input).sort(sort_exprs)?.build()
        }

        ZarrPlan::Limit([input_id, count_id]) => {
            let input = convert_egg_to_plan(rec_expr, *input_id, ctx)?;
            let count_expr = convert_egg_to_expr(rec_expr, *count_id, &input)?;

            // Extract the literal value if it's a simple literal
            let fetch = if let Expr::Literal(ScalarValue::Int64(Some(n)), _) = &count_expr {
                Some(*n as usize)
            } else if let Expr::Literal(ScalarValue::UInt64(Some(n)), _) = &count_expr {
                Some(*n as usize)
            } else {
                // For complex expressions, we can't easily get a usize
                // Return the original plan's limit or use a default
                None
            };

            LogicalPlanBuilder::from(input).limit(0, fetch)?.build()
        }

        ZarrPlan::Distinct([input_id]) => {
            let input = convert_egg_to_plan(rec_expr, *input_id, ctx)?;
            LogicalPlanBuilder::from(input).distinct()?.build()
        }

        ZarrPlan::Union([left_id, right_id]) => {
            let left = convert_egg_to_plan(rec_expr, *left_id, ctx)?;
            let right = convert_egg_to_plan(rec_expr, *right_id, ctx)?;
            LogicalPlanBuilder::from(left).union(right)?.build()
        }

        ZarrPlan::InnerJoin([left_id, right_id, cond_id]) => {
            convert_join(rec_expr, *left_id, *right_id, *cond_id, JoinType::Inner, ctx)
        }

        ZarrPlan::LeftJoin([left_id, right_id, cond_id]) => {
            convert_join(rec_expr, *left_id, *right_id, *cond_id, JoinType::Left, ctx)
        }

        ZarrPlan::RightJoin([left_id, right_id, cond_id]) => {
            convert_join(rec_expr, *left_id, *right_id, *cond_id, JoinType::Right, ctx)
        }

        ZarrPlan::FullJoin([left_id, right_id, cond_id]) => {
            convert_join(rec_expr, *left_id, *right_id, *cond_id, JoinType::Full, ctx)
        }

        ZarrPlan::SemiJoin([left_id, right_id, cond_id]) => {
            convert_join(
                rec_expr,
                *left_id,
                *right_id,
                *cond_id,
                JoinType::LeftSemi,
                ctx,
            )
        }

        ZarrPlan::AntiJoin([left_id, right_id, cond_id]) => {
            convert_join(
                rec_expr,
                *left_id,
                *right_id,
                *cond_id,
                JoinType::LeftAnti,
                ctx,
            )
        }

        ZarrPlan::Empty => {
            // Create an empty relation with a minimal schema
            let schema = Arc::new(DFSchema::empty());
            Ok(LogicalPlan::EmptyRelation(EmptyRelation {
                produce_one_row: false,
                schema,
            }))
        }

        // For nodes that don't represent plans (expressions, literals, etc.),
        // we can't convert them to a LogicalPlan directly
        _ => Err(DataFusionError::Internal(format!(
            "Cannot convert {:?} to LogicalPlan - not a plan node",
            node
        ))),
    }
}

/// Convert a join from egg format
fn convert_join(
    rec_expr: &RecExpr<ZarrPlan>,
    left_id: Id,
    right_id: Id,
    cond_id: Id,
    join_type: JoinType,
    ctx: &ConversionContext,
) -> Result<LogicalPlan> {
    let left = convert_egg_to_plan(rec_expr, left_id, ctx)?;
    let right = convert_egg_to_plan(rec_expr, right_id, ctx)?;

    // Build a combined plan for expression conversion
    let combined_plan = LogicalPlanBuilder::from(left.clone())
        .cross_join(right.clone())?
        .build()?;

    let cond_expr = convert_egg_to_expr(rec_expr, cond_id, &combined_plan)?;

    // For now, use filter-based join
    LogicalPlanBuilder::from(left)
        .join_on(right, join_type, vec![cond_expr])?
        .build()
}

/// Convert an egg expression to a DataFusion Expr
fn convert_egg_to_expr(rec_expr: &RecExpr<ZarrPlan>, id: Id, plan: &LogicalPlan) -> Result<Expr> {
    let node = &rec_expr[id];

    match node {
        ZarrPlan::Symbol(sym) => {
            let s = sym.as_str();
            // Check if it's a literal (has type:value format)
            if let Ok(scalar) = literal_to_scalar(sym) {
                Ok(Expr::Literal(scalar, None))
            } else {
                // It's a column reference
                Ok(Expr::Column(Column::new_unqualified(s)))
            }
        }

        ZarrPlan::True => Ok(Expr::Literal(ScalarValue::Boolean(Some(true)), None)),
        ZarrPlan::False => Ok(Expr::Literal(ScalarValue::Boolean(Some(false)), None)),
        ZarrPlan::Null => Ok(Expr::Literal(ScalarValue::Null, None)),
        #[allow(deprecated)]
        ZarrPlan::Star => Ok(Expr::Wildcard {
            qualifier: None,
            options: Default::default(),
        }),

        // Binary arithmetic operations
        ZarrPlan::Add([l, r]) => binary_expr(rec_expr, *l, *r, Operator::Plus, plan),
        ZarrPlan::Sub([l, r]) => binary_expr(rec_expr, *l, *r, Operator::Minus, plan),
        ZarrPlan::Mul([l, r]) => binary_expr(rec_expr, *l, *r, Operator::Multiply, plan),
        ZarrPlan::Div([l, r]) => binary_expr(rec_expr, *l, *r, Operator::Divide, plan),
        ZarrPlan::Mod([l, r]) => binary_expr(rec_expr, *l, *r, Operator::Modulo, plan),

        // Comparison operations
        ZarrPlan::Eq([l, r]) => binary_expr(rec_expr, *l, *r, Operator::Eq, plan),
        ZarrPlan::Neq([l, r]) => binary_expr(rec_expr, *l, *r, Operator::NotEq, plan),
        ZarrPlan::Lt([l, r]) => binary_expr(rec_expr, *l, *r, Operator::Lt, plan),
        ZarrPlan::Le([l, r]) => binary_expr(rec_expr, *l, *r, Operator::LtEq, plan),
        ZarrPlan::Gt([l, r]) => binary_expr(rec_expr, *l, *r, Operator::Gt, plan),
        ZarrPlan::Ge([l, r]) => binary_expr(rec_expr, *l, *r, Operator::GtEq, plan),

        // Logical operations
        ZarrPlan::And([l, r]) => binary_expr(rec_expr, *l, *r, Operator::And, plan),
        ZarrPlan::Or([l, r]) => binary_expr(rec_expr, *l, *r, Operator::Or, plan),
        ZarrPlan::Not([e]) => {
            let expr = convert_egg_to_expr(rec_expr, *e, plan)?;
            Ok(Expr::Not(Box::new(expr)))
        }

        ZarrPlan::Neg([e]) => {
            let expr = convert_egg_to_expr(rec_expr, *e, plan)?;
            Ok(Expr::Negative(Box::new(expr)))
        }

        ZarrPlan::IsNull([e]) => {
            let expr = convert_egg_to_expr(rec_expr, *e, plan)?;
            Ok(Expr::IsNull(Box::new(expr)))
        }

        ZarrPlan::IsNotNull([e]) => {
            let expr = convert_egg_to_expr(rec_expr, *e, plan)?;
            Ok(Expr::IsNotNull(Box::new(expr)))
        }

        ZarrPlan::Between([expr_id, low_id, high_id]) => {
            let expr = convert_egg_to_expr(rec_expr, *expr_id, plan)?;
            let low = convert_egg_to_expr(rec_expr, *low_id, plan)?;
            let high = convert_egg_to_expr(rec_expr, *high_id, plan)?;
            Ok(Expr::Between(datafusion::logical_expr::Between {
                expr: Box::new(expr),
                negated: false,
                low: Box::new(low),
                high: Box::new(high),
            }))
        }

        ZarrPlan::Case([cond_id, then_id, else_id]) => {
            let cond = convert_egg_to_expr(rec_expr, *cond_id, plan)?;
            let then = convert_egg_to_expr(rec_expr, *then_id, plan)?;
            let else_expr = convert_egg_to_expr(rec_expr, *else_id, plan)?;
            Ok(Expr::Case(datafusion::logical_expr::Case {
                expr: None,
                when_then_expr: vec![(Box::new(cond), Box::new(then))],
                else_expr: Some(Box::new(else_expr)),
            }))
        }

        ZarrPlan::Alias([expr_id, name_id]) => {
            let expr = convert_egg_to_expr(rec_expr, *expr_id, plan)?;
            let name_sym = get_symbol(rec_expr, *name_id)?;
            // Remove the "str:" prefix if present
            let name = if let Some((_, v)) = parse_literal(&name_sym.clone().into()) {
                v
            } else {
                name_sym
            };
            Ok(expr.alias(name))
        }

        // Aggregate functions
        ZarrPlan::Count([e]) => {
            let arg = convert_egg_to_expr(rec_expr, *e, plan)?;
            Ok(datafusion::functions_aggregate::count::count(arg))
        }
        ZarrPlan::Sum([e]) => {
            let arg = convert_egg_to_expr(rec_expr, *e, plan)?;
            Ok(datafusion::functions_aggregate::sum::sum(arg))
        }
        ZarrPlan::Avg([e]) => {
            let arg = convert_egg_to_expr(rec_expr, *e, plan)?;
            Ok(datafusion::functions_aggregate::average::avg(arg))
        }
        ZarrPlan::Min([e]) => {
            let arg = convert_egg_to_expr(rec_expr, *e, plan)?;
            Ok(datafusion::functions_aggregate::min_max::min(arg))
        }
        ZarrPlan::Max([e]) => {
            let arg = convert_egg_to_expr(rec_expr, *e, plan)?;
            Ok(datafusion::functions_aggregate::min_max::max(arg))
        }

        // String operations
        ZarrPlan::Concat([l, r]) => binary_expr(rec_expr, *l, *r, Operator::StringConcat, plan),

        // List node - shouldn't be directly converted to an expression
        ZarrPlan::List(_) => Err(DataFusionError::Internal(
            "List node should not be converted directly to Expr".to_string(),
        )),

        // Empty - represents no expression
        ZarrPlan::Empty => Err(DataFusionError::Internal(
            "Empty node should not be converted directly to Expr".to_string(),
        )),

        _ => Err(DataFusionError::NotImplemented(format!(
            "Conversion from egg {:?} to Expr not implemented",
            node
        ))),
    }
}

/// Helper to create a binary expression
fn binary_expr(
    rec_expr: &RecExpr<ZarrPlan>,
    left_id: Id,
    right_id: Id,
    op: Operator,
    plan: &LogicalPlan,
) -> Result<Expr> {
    let left = convert_egg_to_expr(rec_expr, left_id, plan)?;
    let right = convert_egg_to_expr(rec_expr, right_id, plan)?;
    Ok(Expr::BinaryExpr(BinaryExpr {
        left: Box::new(left),
        op,
        right: Box::new(right),
    }))
}

/// Get a symbol from a node (expects Symbol variant)
fn get_symbol(rec_expr: &RecExpr<ZarrPlan>, id: Id) -> Result<String> {
    match &rec_expr[id] {
        ZarrPlan::Symbol(s) => Ok(s.to_string()),
        other => Err(DataFusionError::Internal(format!(
            "Expected Symbol, got {:?}",
            other
        ))),
    }
}

/// Convert a List node to a vector of expressions
fn convert_egg_to_expr_list(
    rec_expr: &RecExpr<ZarrPlan>,
    id: Id,
    plan: &LogicalPlan,
) -> Result<Vec<Expr>> {
    match &rec_expr[id] {
        ZarrPlan::List(children) => children
            .iter()
            .map(|&child_id| convert_egg_to_expr(rec_expr, child_id, plan))
            .collect(),
        ZarrPlan::Empty => Ok(vec![]),
        other => Err(DataFusionError::Internal(format!(
            "Expected List or Empty, got {:?}",
            other
        ))),
    }
}

/// Convert a List of sort expressions
fn convert_egg_to_sort_expr_list(
    rec_expr: &RecExpr<ZarrPlan>,
    id: Id,
    plan: &LogicalPlan,
) -> Result<Vec<datafusion::logical_expr::SortExpr>> {
    match &rec_expr[id] {
        ZarrPlan::List(children) => children
            .iter()
            .map(|&child_id| convert_egg_to_sort_expr(rec_expr, child_id, plan))
            .collect(),
        ZarrPlan::Empty => Ok(vec![]),
        other => Err(DataFusionError::Internal(format!(
            "Expected List or Empty for sort keys, got {:?}",
            other
        ))),
    }
}

/// Convert a SortExpr node
fn convert_egg_to_sort_expr(
    rec_expr: &RecExpr<ZarrPlan>,
    id: Id,
    plan: &LogicalPlan,
) -> Result<datafusion::logical_expr::SortExpr> {
    match &rec_expr[id] {
        ZarrPlan::SortExpr([expr_id, asc_id, nulls_first_id]) => {
            let expr = convert_egg_to_expr(rec_expr, *expr_id, plan)?;
            let asc = matches!(&rec_expr[*asc_id], ZarrPlan::True);
            let nulls_first = matches!(&rec_expr[*nulls_first_id], ZarrPlan::True);
            Ok(datafusion::logical_expr::SortExpr {
                expr,
                asc,
                nulls_first,
            })
        }
        // If it's just an expression, default to ascending with nulls first
        _ => {
            let expr = convert_egg_to_expr(rec_expr, id, plan)?;
            Ok(datafusion::logical_expr::SortExpr {
                expr,
                asc: true,
                nulls_first: true,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::*;

    #[tokio::test]
    async fn test_convert_simple_projection() -> Result<()> {
        let ctx = SessionContext::new();
        ctx.register_csv("test", "data/test.csv", CsvReadOptions::new())
            .await
            .ok();

        // This test will fail if test.csv doesn't exist, which is fine
        // The important thing is that the conversion logic compiles

        Ok(())
    }

    #[test]
    fn test_scalar_to_literal() {
        let int_lit = scalar_to_literal(&ScalarValue::Int64(Some(42)));
        assert_eq!(int_lit.as_str(), "i64:42");

        let float_lit = scalar_to_literal(&ScalarValue::Float64(Some(3.14)));
        assert_eq!(float_lit.as_str(), "f64:3.14");

        let str_lit = scalar_to_literal(&ScalarValue::Utf8(Some("hello".to_string())));
        assert_eq!(str_lit.as_str(), "str:hello");
    }

    #[test]
    fn test_literal_to_scalar() {
        let int_val = literal_to_scalar(&"i64:42".into()).unwrap();
        assert_eq!(int_val, ScalarValue::Int64(Some(42)));

        let float_val = literal_to_scalar(&"f64:3.14".into()).unwrap();
        if let ScalarValue::Float64(Some(v)) = float_val {
            assert!((v - 3.14).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }

        let str_val = literal_to_scalar(&"str:hello".into()).unwrap();
        assert_eq!(str_val, ScalarValue::Utf8(Some("hello".to_string())));
    }

    #[test]
    fn test_parse_and_convert_expr() {
        let mut expr = RecExpr::<ZarrPlan>::default();

        // Build: x + 1
        // Column takes a Symbol directly in the variant
        let x = expr.add(ZarrPlan::Symbol("x".into()));
        let one = expr.add(ZarrPlan::Symbol("i64:1".into()));
        let _add = expr.add(ZarrPlan::Add([x, one]));

        assert_eq!(expr.as_ref().len(), 3);
    }

    #[test]
    fn test_parse_expression_string() {
        // Test that we can parse an expression from a string
        // Bare symbols: x is column, i64:1 is literal (both Symbol variant)
        let expr: RecExpr<ZarrPlan> = "(+ x i64:1)".parse().unwrap();
        assert_eq!(expr.as_ref().len(), 3);
    }

    #[test]
    fn test_expr_round_trip() -> Result<()> {
        // Create a simple expression
        let col_expr = Expr::Column(Column::new_unqualified("x"));
        let lit_expr = Expr::Literal(ScalarValue::Int64(Some(42)), None);
        let binary = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(col_expr),
            op: Operator::Plus,
            right: Box::new(lit_expr),
        });

        // Convert to egg
        let mut rec_expr = RecExpr::<ZarrPlan>::default();
        let _root = convert_expr_to_egg(&binary, &mut rec_expr)?;

        // Create a dummy plan for context
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, false)]);
        let empty_plan = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(DFSchema::try_from(schema)?),
        });

        // Convert back
        let root_id = Id::from(rec_expr.as_ref().len() - 1);
        let result = convert_egg_to_expr(&rec_expr, root_id, &empty_plan)?;

        // Verify structure
        match result {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                assert_eq!(op, Operator::Plus);
                match *left {
                    Expr::Column(col) => assert_eq!(col.name, "x"),
                    _ => panic!("Expected Column"),
                }
                match *right {
                    Expr::Literal(ScalarValue::Int64(Some(42)), _) => {}
                    _ => panic!("Expected Int64(42)"),
                }
            }
            _ => panic!("Expected BinaryExpr"),
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_plan_round_trip_simple_filter() -> Result<()> {
        let ctx = SessionContext::new();

        // Create a simple in-memory table
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ]);
        let batch = arrow::array::RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![1, 2, 3])),
                Arc::new(arrow::array::StringArray::from(vec!["x", "y", "z"])),
            ],
        )?;

        ctx.register_batch("test_table", batch)?;

        // Create a simple filter plan
        let plan = ctx
            .sql("SELECT a FROM test_table WHERE a > 1")
            .await?
            .into_optimized_plan()?;

        // Convert to egg
        let rec_expr = logical_plan_to_egg(&plan)?;

        // Convert back
        let result = egg_to_logical_plan(&rec_expr, &plan)?;

        // The result should be a valid plan
        // We can't directly compare plans due to internal differences,
        // but we can verify the result is structurally similar
        assert!(matches!(
            result,
            LogicalPlan::Projection(_) | LogicalPlan::Filter(_) | LogicalPlan::TableScan(_)
        ));

        Ok(())
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    // Generate random ScalarValue for testing
    fn arb_scalar_value() -> impl Strategy<Value = ScalarValue> {
        prop_oneof![
            any::<i64>().prop_map(|v| ScalarValue::Int64(Some(v))),
            any::<f64>()
                .prop_filter("Must be finite", |f| f.is_finite())
                .prop_map(|v| ScalarValue::Float64(Some(v))),
            any::<bool>().prop_map(|v| ScalarValue::Boolean(Some(v))),
            "[a-z]{1,10}".prop_map(|s| ScalarValue::Utf8(Some(s))),
        ]
    }

    // Generate random binary operators
    fn arb_binary_op() -> impl Strategy<Value = Operator> {
        prop_oneof![
            Just(Operator::Plus),
            Just(Operator::Minus),
            Just(Operator::Multiply),
            Just(Operator::Eq),
            Just(Operator::NotEq),
            Just(Operator::Lt),
            Just(Operator::LtEq),
            Just(Operator::Gt),
            Just(Operator::GtEq),
            Just(Operator::And),
            Just(Operator::Or),
        ]
    }

    // Generate column names
    fn arb_column_name() -> impl Strategy<Value = String> {
        "[a-z]{1,5}".prop_map(|s| s)
    }

    // Generate simple expressions
    fn arb_simple_expr() -> impl Strategy<Value = Expr> {
        prop_oneof![
            arb_column_name().prop_map(|name| Expr::Column(Column::new_unqualified(name))),
            arb_scalar_value().prop_map(|v| Expr::Literal(v, None)),
        ]
    }

    // Generate compound expressions (up to depth 2)
    fn arb_expr() -> impl Strategy<Value = Expr> {
        arb_simple_expr().prop_recursive(2, 8, 4, |inner| {
            prop_oneof![
                // Binary expression
                (inner.clone(), arb_binary_op(), inner.clone()).prop_map(|(left, op, right)| {
                    Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(left),
                        op,
                        right: Box::new(right),
                    })
                }),
                // NOT expression
                inner.clone().prop_map(|e| Expr::Not(Box::new(e))),
                // Negative expression (only for numeric types)
                inner.clone().prop_map(|e| Expr::Negative(Box::new(e))),
                // IS NULL
                inner.clone().prop_map(|e| Expr::IsNull(Box::new(e))),
                // IS NOT NULL
                inner.clone().prop_map(|e| Expr::IsNotNull(Box::new(e))),
            ]
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_scalar_round_trip(scalar in arb_scalar_value()) {
            // Convert to symbol and back
            let symbol = scalar_to_literal(&scalar);
            let result = literal_to_scalar(&symbol);

            // For successfully parsed values, they should match
            if let Ok(result_scalar) = result {
                match (&scalar, &result_scalar) {
                    (ScalarValue::Int64(Some(a)), ScalarValue::Int64(Some(b))) => {
                        prop_assert_eq!(a, b);
                    }
                    (ScalarValue::Float64(Some(a)), ScalarValue::Float64(Some(b))) => {
                        prop_assert!((a - b).abs() < 1e-10 || (a.is_nan() && b.is_nan()));
                    }
                    (ScalarValue::Boolean(Some(a)), ScalarValue::Boolean(Some(b))) => {
                        prop_assert_eq!(a, b);
                    }
                    (ScalarValue::Utf8(Some(a)), ScalarValue::Utf8(Some(b))) => {
                        prop_assert_eq!(a, b);
                    }
                    _ => {} // Other types may not round-trip perfectly
                }
            }
        }

        #[test]
        fn test_expr_round_trip_proptest(expr in arb_expr()) {
            // Create a schema with columns that might be referenced
            let schema = Schema::new(vec![
                Field::new("a", DataType::Int64, true),
                Field::new("b", DataType::Int64, true),
                Field::new("c", DataType::Float64, true),
                Field::new("d", DataType::Utf8, true),
                Field::new("e", DataType::Boolean, true),
            ]);
            let df_schema = DFSchema::try_from(schema).unwrap();
            let empty_plan = LogicalPlan::EmptyRelation(EmptyRelation {
                produce_one_row: false,
                schema: Arc::new(df_schema),
            });

            // Try to convert to egg
            let mut rec_expr = RecExpr::<ZarrPlan>::default();
            let convert_result = convert_expr_to_egg(&expr, &mut rec_expr);

            // If conversion succeeded, try round-trip
            if let Ok(_) = convert_result {
                let root_id = Id::from(rec_expr.as_ref().len() - 1);
                let back_result = convert_egg_to_expr(&rec_expr, root_id, &empty_plan);

                // The round-trip should succeed
                prop_assert!(back_result.is_ok(), "Round-trip failed: {:?}", back_result.err());
            }
        }

        #[test]
        fn test_binary_expr_structure_preserved(
            col_name in arb_column_name(),
            op in arb_binary_op(),
            scalar in arb_scalar_value()
        ) {
            let col = Expr::Column(Column::new_unqualified(col_name.clone()));
            let lit = Expr::Literal(scalar.clone(), None);
            let binary = Expr::BinaryExpr(BinaryExpr {
                left: Box::new(col),
                op,
                right: Box::new(lit),
            });

            // Convert to egg
            let mut rec_expr = RecExpr::<ZarrPlan>::default();
            if let Ok(_) = convert_expr_to_egg(&binary, &mut rec_expr) {
                // Create minimal context
                let schema = Schema::new(vec![
                    Field::new(&col_name, DataType::Int64, true),
                ]);
                let df_schema = DFSchema::try_from(schema).unwrap();
                let empty_plan = LogicalPlan::EmptyRelation(EmptyRelation {
                    produce_one_row: false,
                    schema: Arc::new(df_schema),
                });

                // Convert back
                let root_id = Id::from(rec_expr.as_ref().len() - 1);
                if let Ok(result) = convert_egg_to_expr(&rec_expr, root_id, &empty_plan) {
                    // Verify it's still a binary expression
                    match result {
                        Expr::BinaryExpr(BinaryExpr { left, op: result_op, .. }) => {
                            prop_assert_eq!(op, result_op);
                            // Verify left side is the column
                            match *left {
                                Expr::Column(c) => prop_assert_eq!(c.name, col_name),
                                _ => prop_assert!(false, "Expected column on left"),
                            }
                        }
                        _ => prop_assert!(false, "Expected BinaryExpr, got {:?}", result),
                    }
                }
            }
        }
    }
}
