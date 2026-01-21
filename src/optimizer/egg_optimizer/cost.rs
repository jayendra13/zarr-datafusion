//! Cost function for extracting optimal plans from e-graphs
//!
//! The cost function considers:
//! - I/O cost (bytes read from Zarr stores)
//! - Computation cost (CPU operations)
//! - Memory usage
//! - Network penalty for remote data access

use egg::{CostFunction, EGraph, Id};

use super::analysis::ZarrAnalysis;
use super::language::ZarrPlan;

/// Tunable cost parameters
///
/// These parameters control the relative importance of different
/// cost factors in the optimization process.
#[derive(Debug, Clone)]
pub struct CostParameters {
    /// Weight for I/O cost (bytes read)
    pub io_weight: f64,
    /// Weight for computation cost
    pub compute_weight: f64,
    /// Weight for memory usage
    pub memory_weight: f64,
    /// Penalty multiplier for network I/O (remote Zarr)
    pub remote_penalty: f64,
}

impl Default for CostParameters {
    fn default() -> Self {
        Self {
            io_weight: 1.0,
            compute_weight: 0.1,
            memory_weight: 0.5,
            remote_penalty: 10.0,
        }
    }
}

/// Cost function for Zarr query optimization
pub struct ZarrCostFunction<'a> {
    egraph: &'a EGraph<ZarrPlan, ZarrAnalysis>,
    params: &'a CostParameters,
}

impl<'a> ZarrCostFunction<'a> {
    pub fn new(egraph: &'a EGraph<ZarrPlan, ZarrAnalysis>, params: &'a CostParameters) -> Self {
        Self { egraph, params }
    }
}

impl<'a> CostFunction<ZarrPlan> for ZarrCostFunction<'a> {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &ZarrPlan, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        // Get analysis data for this node's e-class
        // Note: enode.children() gives us the child Ids

        let base_cost: f64 = match enode {
            // === I/O Operations ===
            ZarrPlan::Scan([table, _proj]) => {
                let table_data = &self.egraph[*table].data;
                // Estimate I/O based on cardinality
                let rows = table_data.cardinality.unwrap_or(10_000) as f64;
                let bytes = table_data.byte_size.unwrap_or(rows as usize * 100) as f64;
                self.params.io_weight * bytes
            }

            ZarrPlan::ZarrScan([_path, _coords, _vars]) => {
                // Zarr scan - similar to regular scan but potentially remote
                // Base cost is high due to I/O
                self.params.io_weight * 100_000.0
            }

            // Coordinate filter is very cheap - it reduces I/O significantly
            ZarrPlan::CoordFilter([input, _coord, _range]) => {
                let input_cost = costs(*input);
                // Coordinate filter reduces cost by pushing down to scan
                input_cost * 0.1 + 1.0
            }

            // === Filter Operations ===
            ZarrPlan::Filter([input, _pred]) => {
                let input_cost = costs(*input);
                let input_data = &self.egraph[*input].data;
                let rows = input_data.cardinality.unwrap_or(10_000) as f64;
                // Filter cost: scan input + evaluate predicate
                input_cost + self.params.compute_weight * rows * 0.01
            }

            // === Projection ===
            ZarrPlan::Project([input, exprs]) => {
                let input_cost = costs(*input);
                let exprs_cost = costs(*exprs);
                // Projection is cheap
                input_cost + exprs_cost + 5.0
            }

            // === Aggregation ===
            ZarrPlan::Aggregate([input, group, aggs]) => {
                let input_cost = costs(*input);
                let group_cost = costs(*group);
                let aggs_cost = costs(*aggs);
                let input_data = &self.egraph[*input].data;
                let rows = input_data.cardinality.unwrap_or(10_000) as f64;
                // Aggregation cost depends on input size and whether grouped
                let group_data = &self.egraph[*group].data;
                let is_grouped = group_data.cardinality.map(|c| c > 0).unwrap_or(false);
                let agg_cost = if is_grouped {
                    // Hash aggregation
                    self.params.compute_weight * rows * 0.1
                        + self.params.memory_weight * rows * 0.01
                } else {
                    // Simple aggregation (single pass)
                    self.params.compute_weight * rows * 0.01
                };
                input_cost + group_cost + aggs_cost + agg_cost
            }

            // === Join Operations ===
            ZarrPlan::InnerJoin([left, right, cond]) => {
                let left_cost = costs(*left);
                let right_cost = costs(*right);
                let cond_cost = costs(*cond);
                let left_data = &self.egraph[*left].data;
                let right_data = &self.egraph[*right].data;
                let l_rows = left_data.cardinality.unwrap_or(1000) as f64;
                let r_rows = right_data.cardinality.unwrap_or(1000) as f64;
                // Hash join cost
                let join_cost = self.params.compute_weight * (l_rows + r_rows) * 0.1
                    + self.params.memory_weight * l_rows.min(r_rows) * 0.1;
                left_cost + right_cost + cond_cost + join_cost
            }

            ZarrPlan::LeftJoin([left, right, cond])
            | ZarrPlan::RightJoin([left, right, cond])
            | ZarrPlan::FullJoin([left, right, cond]) => {
                let left_cost = costs(*left);
                let right_cost = costs(*right);
                let cond_cost = costs(*cond);
                // Outer joins are slightly more expensive
                left_cost + right_cost + cond_cost + 100.0
            }

            ZarrPlan::CrossJoin([left, right]) => {
                let left_cost = costs(*left);
                let right_cost = costs(*right);
                let left_data = &self.egraph[*left].data;
                let right_data = &self.egraph[*right].data;
                let l_rows = left_data.cardinality.unwrap_or(1000) as f64;
                let r_rows = right_data.cardinality.unwrap_or(1000) as f64;
                // Cross join is very expensive (O(n*m))
                left_cost + right_cost + self.params.compute_weight * l_rows * r_rows * 0.001
            }

            ZarrPlan::SemiJoin([left, right, cond]) | ZarrPlan::AntiJoin([left, right, cond]) => {
                let left_cost = costs(*left);
                let right_cost = costs(*right);
                let cond_cost = costs(*cond);
                // Semi/anti joins can be more efficient than full joins
                left_cost + right_cost + cond_cost + 50.0
            }

            // === Sort Operations ===
            ZarrPlan::Sort([input, keys]) => {
                let input_cost = costs(*input);
                let keys_cost = costs(*keys);
                let input_data = &self.egraph[*input].data;
                let rows = input_data.cardinality.unwrap_or(10_000) as f64;
                // Sort is O(n log n)
                let sort_cost =
                    self.params.compute_weight * rows * rows.ln().max(1.0) * 0.01
                        + self.params.memory_weight * rows * 0.1;
                input_cost + keys_cost + sort_cost
            }

            // === Limit ===
            ZarrPlan::Limit([input, _count]) => {
                let input_cost = costs(*input);
                // Limit can significantly reduce downstream work
                input_cost + 1.0
            }

            // === Distinct ===
            ZarrPlan::Distinct([input]) => {
                let input_cost = costs(*input);
                let input_data = &self.egraph[*input].data;
                let rows = input_data.cardinality.unwrap_or(10_000) as f64;
                // Hash-based distinct
                input_cost + self.params.memory_weight * rows * 0.1
            }

            // === Union ===
            ZarrPlan::Union([left, right]) => {
                let left_cost = costs(*left);
                let right_cost = costs(*right);
                // Union is cheap
                left_cost + right_cost + 10.0
            }

            // === Expressions - very cheap ===
            ZarrPlan::Add([l, r])
            | ZarrPlan::Sub([l, r])
            | ZarrPlan::Mul([l, r])
            | ZarrPlan::Div([l, r])
            | ZarrPlan::Mod([l, r]) => {
                let left_cost = costs(*l);
                let right_cost = costs(*r);
                left_cost + right_cost + 1.0
            }

            ZarrPlan::Neg([e]) => costs(*e) + 0.5,

            // === Comparison expressions ===
            ZarrPlan::Eq([l, r])
            | ZarrPlan::Neq([l, r])
            | ZarrPlan::Lt([l, r])
            | ZarrPlan::Le([l, r])
            | ZarrPlan::Gt([l, r])
            | ZarrPlan::Ge([l, r]) => {
                let left_cost = costs(*l);
                let right_cost = costs(*r);
                left_cost + right_cost + 1.0
            }

            ZarrPlan::Between([e, low, high]) => {
                costs(*e) + costs(*low) + costs(*high) + 2.0
            }

            ZarrPlan::In([e, list]) => costs(*e) + costs(*list) + 2.0,

            // === Logical expressions ===
            ZarrPlan::And([l, r]) | ZarrPlan::Or([l, r]) => {
                let left_cost = costs(*l);
                let right_cost = costs(*r);
                left_cost + right_cost + 0.5
            }

            ZarrPlan::Not([e])
            | ZarrPlan::IsNull([e])
            | ZarrPlan::IsNotNull([e]) => costs(*e) + 0.5,

            // === Aggregate functions (per-row cost) ===
            ZarrPlan::Count([e])
            | ZarrPlan::CountDistinct([e])
            | ZarrPlan::Sum([e])
            | ZarrPlan::Avg([e])
            | ZarrPlan::Min([e])
            | ZarrPlan::Max([e])
            | ZarrPlan::Stddev([e])
            | ZarrPlan::Variance([e]) => costs(*e) + 1.0,

            // === Scalar functions ===
            ZarrPlan::Abs([e])
            | ZarrPlan::Sqrt([e])
            | ZarrPlan::Floor([e])
            | ZarrPlan::Ceil([e])
            | ZarrPlan::Upper([e])
            | ZarrPlan::Lower([e])
            | ZarrPlan::Trim([e])
            | ZarrPlan::Length([e]) => costs(*e) + 1.0,

            ZarrPlan::Pow([base, exp]) => costs(*base) + costs(*exp) + 2.0,
            ZarrPlan::Round([e, digits]) => costs(*e) + costs(*digits) + 1.0,
            ZarrPlan::Concat([l, r]) => costs(*l) + costs(*r) + 2.0,
            ZarrPlan::Substring([e, start, len]) => {
                costs(*e) + costs(*start) + costs(*len) + 2.0
            }

            // === Type casting ===
            ZarrPlan::Cast([e, _t]) | ZarrPlan::TryCast([e, _t]) => costs(*e) + 1.0,

            // === Conditional ===
            ZarrPlan::Case([cond, then, else_]) => {
                costs(*cond) + costs(*then) + costs(*else_) + 2.0
            }

            ZarrPlan::Coalesce(children) => {
                children.iter().map(|&c| costs(c)).sum::<f64>() + 1.0
            }

            ZarrPlan::NullIf([l, r]) => costs(*l) + costs(*r) + 1.0,

            // === Date/Time ===
            ZarrPlan::DatePart([_part, e]) | ZarrPlan::DateTrunc([_part, e]) => costs(*e) + 2.0,
            ZarrPlan::Now => 0.1,

            // === Window functions ===
            ZarrPlan::Window([func, part, order, frame]) => {
                costs(*func) + costs(*part) + costs(*order) + costs(*frame) + 50.0
            }
            ZarrPlan::RowNumber | ZarrPlan::Rank | ZarrPlan::DenseRank => 1.0,
            ZarrPlan::Lag([e, offset]) | ZarrPlan::Lead([e, offset]) => {
                costs(*e) + costs(*offset) + 2.0
            }
            ZarrPlan::FirstValue([e]) | ZarrPlan::LastValue([e]) => costs(*e) + 2.0,

            // === Lists and structure ===
            ZarrPlan::List(children) => {
                children.iter().map(|&c| costs(c)).sum::<f64>()
            }

            ZarrPlan::Alias([e, _name]) => costs(*e) + 0.1,
            ZarrPlan::SortExpr([e, _asc, _nulls]) => costs(*e) + 0.5,

            // === Literals and references - essentially free ===
            ZarrPlan::Symbol(_)
            | ZarrPlan::True
            | ZarrPlan::False
            | ZarrPlan::Null
            | ZarrPlan::Star
            | ZarrPlan::Empty => 0.0,

            // === Zarr-specific ===
            ZarrPlan::Resample([input, _dim, _freq]) => {
                let input_cost = costs(*input);
                // Resampling requires sorting and grouping
                input_cost + 100.0
            }

            // Catch-all for any variants we might have missed
            ZarrPlan::Like([e, pattern]) => costs(*e) + costs(*pattern) + 3.0,
        };

        base_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egg::RecExpr;

    #[test]
    fn test_cost_params_default() {
        let params = CostParameters::default();
        assert_eq!(params.io_weight, 1.0);
        assert_eq!(params.compute_weight, 0.1);
    }

    #[test]
    fn test_simple_expression_cost() {
        // Bare symbols are parsed as Symbol
        let expr: RecExpr<ZarrPlan> = "(+ x y)".parse().unwrap();
        let mut egraph = egg::EGraph::<ZarrPlan, ZarrAnalysis>::default();
        let root = egraph.add_expr(&expr);
        egraph.rebuild();

        let params = CostParameters::default();
        let cost_fn = ZarrCostFunction::new(&egraph, &params);
        let extractor = egg::Extractor::new(&egraph, cost_fn);
        let (cost, _) = extractor.find_best(root);

        // Simple expression should have low cost
        assert!(cost < 10.0);
    }

    #[test]
    fn test_literal_zero_cost() {
        // Bare symbol i64:42 is parsed as Symbol(Symbol("i64:42"))
        let expr: RecExpr<ZarrPlan> = "i64:42".parse().unwrap();
        let mut egraph = egg::EGraph::<ZarrPlan, ZarrAnalysis>::default();
        let root = egraph.add_expr(&expr);
        egraph.rebuild();

        let params = CostParameters::default();
        let cost_fn = ZarrCostFunction::new(&egraph, &params);
        let extractor = egg::Extractor::new(&egraph, cost_fn);
        let (cost, _) = extractor.find_best(root);

        // Literal should have zero cost
        assert_eq!(cost, 0.0);
    }
}
