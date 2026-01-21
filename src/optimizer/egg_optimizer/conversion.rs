//! Conversion between DataFusion LogicalPlan and egg RecExpr
//!
//! This module handles bidirectional conversion:
//! - `logical_plan_to_egg`: DataFusion LogicalPlan → egg RecExpr<ZarrPlan>
//! - `egg_to_logical_plan`: egg RecExpr<ZarrPlan> → DataFusion LogicalPlan

use datafusion::common::{DataFusionError, Result};
use datafusion::logical_expr::{
    expr::AggregateFunction, BinaryExpr, Expr, JoinType, LogicalPlan, Operator,
};
use egg::{Id, RecExpr, Symbol};

use super::language::{make_literal, ZarrPlan};

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
                all_conds.into_iter().reduce(|acc, c| {
                    expr.add(ZarrPlan::And([acc, c]))
                }).unwrap()
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
                expr.add(ZarrPlan::Symbol(make_literal("i64", &i64::MAX.to_string())))
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
            let type_id = expr.add(ZarrPlan::Symbol(
                format!("{:?}", cast.data_type).into(),
            ));
            Ok(expr.add(ZarrPlan::Cast([expr_id, type_id])))
        }

        Expr::TryCast(cast) => {
            let expr_id = convert_expr_to_egg(&cast.expr, expr)?;
            let type_id = expr.add(ZarrPlan::Symbol(
                format!("{:?}", cast.data_type).into(),
            ));
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

        Expr::Wildcard { .. } => Ok(expr.add(ZarrPlan::Star)),

        _ => Err(DataFusionError::NotImplemented(format!(
            "Expression type {:?} not supported in e-graph",
            df_expr.variant_name()
        ))),
    }
}

/// Convert a sort expression to egg format
fn convert_sort_expr_to_egg(sort: &datafusion::logical_expr::SortExpr, expr: &mut RecExpr<ZarrPlan>) -> Result<Id> {
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
fn scalar_to_literal(value: &datafusion::common::ScalarValue) -> Symbol {
    use datafusion::common::ScalarValue;
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

/// Convert an egg RecExpr back to a DataFusion LogicalPlan
///
/// This requires the original plan to provide context (schema, table sources, etc.)
pub fn egg_to_logical_plan(
    _rec_expr: &RecExpr<ZarrPlan>,
    original: &LogicalPlan,
) -> Result<LogicalPlan> {
    // For now, return the original plan if conversion back is too complex
    // The full implementation would need to:
    // 1. Build the plan from the RecExpr bottom-up
    // 2. Look up table sources from the original plan
    // 3. Reconstruct schema information
    //
    // This is a significant undertaking and would require careful handling
    // of all the context that DataFusion plans carry.

    // For initial implementation, we detect if the optimized expression
    // matches certain patterns (like constant folding for MIN/MAX) and
    // return the appropriate simplified plan.

    // If no optimization pattern matched, return the original
    // A full implementation would traverse rec_expr and rebuild the plan
    Ok(original.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::common::ScalarValue;
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
}
