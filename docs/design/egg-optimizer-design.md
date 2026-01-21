# Zarr-DataFusion E-Graph Query Optimizer Design

**Author:** Claude
**Created:** January 21, 2026
**Status:** Draft

## Executive Summary

This document proposes an architecture for integrating the `egg` library (equality saturation via e-graphs) with zarr-datafusion to build a sophisticated query optimizer. The optimizer will leverage Xarray/Zarr metadata and statistics to make intelligent optimization decisions while delegating to DataFusion's built-in optimizations where appropriate.

## Background

### Problem Statement

The zarr-datafusion project needs a query optimizer capable of:

1. **Expression simplification** - Algebraic transformations (e.g., `a * 2 / 2` → `a`)
2. **Subquery rewrites** - Restructuring nested queries for efficiency
3. **Statistics-driven optimization** - Using Zarr/Xarray metadata to choose optimal plans
4. **Efficient scientific data access** - Optimizing for the Cartesian product structure of Zarr stores

Current optimizer rules (`MinMaxStatisticsRule`, `CountStatisticsRule`, `ZarrLimitPushdownRule`) use pattern matching but cannot explore the full space of equivalent query plans.

### Why E-Graphs and Equality Saturation?

Traditional rewrite systems are destructive - applying one rewrite prevents exploring alternatives. E-graphs solve this by:

1. **Compact representation** - Store exponentially many equivalent expressions efficiently
2. **Non-destructive rewrites** - All equivalent forms coexist in the same structure
3. **Optimal extraction** - Cost functions select the best plan after saturation

The `egg` library provides:
- High-performance e-graph implementation
- Flexible `Language` trait for custom ASTs
- `Analysis` trait for domain-specific information (e.g., statistics)
- `Runner` for equality saturation with resource limits
- `Extractor` for cost-based plan selection

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Query Optimization Pipeline                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  SQL Query                                                          │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────┐                       │
│  │  DataFusion Parser + Analyzer           │                       │
│  │  (SQL → LogicalPlan)                    │                       │
│  └─────────────────────────────────────────┘                       │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────┐                       │
│  │  DataFusion Built-in Optimizers         │                       │
│  │  - Predicate pushdown                   │                       │
│  │  - Projection pushdown                  │                       │
│  │  - Common subexpression elimination     │                       │
│  └─────────────────────────────────────────┘                       │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────┐                       │
│  │  Zarr E-Graph Optimizer (NEW)           │                       │
│  │  ┌─────────────────────────────────┐    │                       │
│  │  │ 1. Convert: LogicalPlan → EGraph│    │                       │
│  │  └─────────────────────────────────┘    │                       │
│  │  ┌─────────────────────────────────┐    │                       │
│  │  │ 2. Saturate with rewrite rules  │    │                       │
│  │  │    - Expression simplification  │    │                       │
│  │  │    - Join reordering            │    │                       │
│  │  │    - Aggregate pushdown         │    │                       │
│  │  │    - Zarr-specific rewrites     │    │                       │
│  │  └─────────────────────────────────┘    │                       │
│  │  ┌─────────────────────────────────┐    │                       │
│  │  │ 3. Extract: EGraph → LogicalPlan│    │                       │
│  │  │    (cost = f(stats, I/O, etc.)) │    │                       │
│  │  └─────────────────────────────────┘    │                       │
│  └─────────────────────────────────────────┘                       │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────┐                       │
│  │  Physical Planning                      │                       │
│  │  (LogicalPlan → ExecutionPlan)          │                       │
│  └─────────────────────────────────────────┘                       │
│      │                                                              │
│      ▼                                                              │
│  Optimized Query Execution                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Detailed Design

### 1. Language Definition

We define a custom `Language` for DataFusion logical plans:

```rust
use egg::{define_language, Id, Symbol};

define_language! {
    pub enum ZarrPlan {
        // === Relational Operators ===
        "scan" = Scan([Id; 2]),           // [table_ref, projection]
        "filter" = Filter([Id; 2]),       // [input, predicate]
        "project" = Project([Id; 2]),     // [input, expressions]
        "aggregate" = Aggregate([Id; 3]), // [input, group_by, aggregates]
        "join" = Join([Id; 4]),           // [left, right, condition, type]
        "limit" = Limit([Id; 2]),         // [input, count]
        "sort" = Sort([Id; 2]),           // [input, keys]
        "union" = Union([Id; 2]),         // [left, right]

        // === Expressions ===
        "col" = Column(Symbol),           // Column reference
        "lit" = Literal(Symbol),          // Literal value (serialized)
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Mod([Id; 2]),
        "and" = And([Id; 2]),
        "or" = Or([Id; 2]),
        "not" = Not([Id; 1]),
        "=" = Eq([Id; 2]),
        "<>" = Neq([Id; 2]),
        "<" = Lt([Id; 2]),
        "<=" = Le([Id; 2]),
        ">" = Gt([Id; 2]),
        ">=" = Ge([Id; 2]),
        "between" = Between([Id; 3]),     // [expr, low, high]
        "in" = In([Id; 2]),               // [expr, list]
        "is_null" = IsNull([Id; 1]),
        "is_not_null" = IsNotNull([Id; 1]),
        "cast" = Cast([Id; 2]),           // [expr, type]
        "case" = Case([Id; 3]),           // [condition, then, else]

        // === Aggregate Functions ===
        "count" = Count([Id; 1]),
        "sum" = Sum([Id; 1]),
        "avg" = Avg([Id; 1]),
        "min" = Min([Id; 1]),
        "max" = Max([Id; 1]),

        // === Window Functions ===
        "window" = Window([Id; 4]),       // [func, partition, order, frame]
        "row_number" = RowNumber,
        "rank" = Rank,
        "dense_rank" = DenseRank,
        "lag" = Lag([Id; 2]),
        "lead" = Lead([Id; 2]),

        // === Zarr-Specific ===
        "zarr_scan" = ZarrScan([Id; 3]),  // [store_path, coords, vars]
        "coord_filter" = CoordFilter([Id; 3]), // [input, coord, range]
        "resample" = Resample([Id; 3]),   // [input, dim, freq]

        // === Lists and Metadata ===
        "list" = List(Box<[Id]>),         // Variable-length list
        "empty" = Empty,                   // Empty list/relation

        // Join types (leaf nodes)
        "inner" = Inner,
        "left" = Left,
        "right" = Right,
        "full" = Full,
        "cross" = Cross,
        "semi" = Semi,
        "anti" = Anti,

        // Table reference (serialized metadata)
        Table(Symbol),
        // Type reference
        Type(Symbol),
    }
}
```

### 2. Analysis for Statistics and Types

The `Analysis` trait integrates domain-specific information:

```rust
use egg::{Analysis, DidMerge, EGraph, Id};
use datafusion::common::Statistics;
use arrow::datatypes::DataType;

/// Analysis data attached to each e-class
#[derive(Debug, Clone)]
pub struct ZarrAnalysisData {
    /// Data type of expressions, schema for relations
    pub data_type: Option<TypeInfo>,

    /// Statistics from Zarr metadata
    pub statistics: Option<Statistics>,

    /// Estimated row count (from Zarr coordinate sizes)
    pub cardinality: Option<usize>,

    /// Known constant value (for constant folding)
    pub constant: Option<ScalarValue>,

    /// Zarr-specific: coordinate dimension info
    pub coord_info: Option<CoordInfo>,
}

#[derive(Debug, Clone)]
pub enum TypeInfo {
    /// Scalar expression type
    Scalar(DataType),
    /// Relation schema (list of field names and types)
    Relation(Vec<(String, DataType)>),
}

pub struct ZarrAnalysis {
    /// Cache of table statistics by name
    pub table_stats: HashMap<String, ZarrStoreMeta>,
    /// Tunable cost parameters
    pub cost_params: CostParameters,
}

impl Analysis<ZarrPlan> for ZarrAnalysis {
    type Data = ZarrAnalysisData;

    fn make(egraph: &EGraph<ZarrPlan, Self>, enode: &ZarrPlan) -> Self::Data {
        match enode {
            ZarrPlan::ZarrScan([store, coords, vars]) => {
                // Lookup statistics from Zarr metadata
                let store_name = extract_symbol(egraph, *store);
                let stats = self.table_stats.get(&store_name);
                ZarrAnalysisData {
                    statistics: stats.map(|s| s.to_datafusion_statistics()),
                    cardinality: stats.map(|s| s.total_rows),
                    ..Default::default()
                }
            }
            ZarrPlan::Filter([input, _pred]) => {
                // Selectivity estimation from predicate analysis
                let input_data = &egraph[*input].data;
                ZarrAnalysisData {
                    cardinality: input_data.cardinality.map(|c| c / 2), // Estimate
                    ..input_data.clone()
                }
            }
            ZarrPlan::Min([arg]) | ZarrPlan::Max([arg]) => {
                // Can use min/max from statistics
                let arg_data = &egraph[*arg].data;
                if let Some(stats) = &arg_data.statistics {
                    // Extract constant from column statistics
                }
                Default::default()
            }
            // ... other nodes
            _ => Default::default(),
        }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        // Merge analysis data when e-classes are unified
        // Prefer more precise information
        let changed = merge_option(&mut a.statistics, b.statistics)
            | merge_option(&mut a.cardinality, b.cardinality)
            | merge_option(&mut a.constant, b.constant);
        if changed { DidMerge(true, true) } else { DidMerge(false, false) }
    }
}
```

### 3. Rewrite Rules

Rewrite rules are defined using `egg`'s pattern syntax:

```rust
use egg::{rewrite as rw, Rewrite};

pub fn expression_simplification_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    vec![
        // Arithmetic simplification
        rw!("add-zero"; "(+ ?x 0)" => "?x"),
        rw!("add-zero-rev"; "(+ 0 ?x)" => "?x"),
        rw!("mul-one"; "(* ?x 1)" => "?x"),
        rw!("mul-one-rev"; "(* 1 ?x)" => "?x"),
        rw!("mul-zero"; "(* ?x 0)" => "0"),
        rw!("mul-zero-rev"; "(* 0 ?x)" => "0"),
        rw!("div-one"; "(/ ?x 1)" => "?x"),
        rw!("div-self"; "(/ ?x ?x)" => "1"),
        rw!("sub-self"; "(- ?x ?x)" => "0"),

        // Algebraic identities
        rw!("add-comm"; "(+ ?x ?y)" => "(+ ?y ?x)"),
        rw!("mul-comm"; "(* ?x ?y)" => "(* ?y ?x)"),
        rw!("add-assoc"; "(+ (+ ?x ?y) ?z)" => "(+ ?x (+ ?y ?z))"),
        rw!("mul-assoc"; "(* (* ?x ?y) ?z)" => "(* ?x (* ?y ?z))"),
        rw!("mul-div-cancel"; "(/ (* ?x ?y) ?y)" => "?x"),

        // Boolean simplification
        rw!("and-true"; "(and ?x true)" => "?x"),
        rw!("and-false"; "(and ?x false)" => "false"),
        rw!("or-true"; "(or ?x true)" => "true"),
        rw!("or-false"; "(or ?x false)" => "?x"),
        rw!("not-not"; "(not (not ?x))" => "?x"),
        rw!("de-morgan-and"; "(not (and ?x ?y))" => "(or (not ?x) (not ?y))"),
        rw!("de-morgan-or"; "(not (or ?x ?y))" => "(and (not ?x) (not ?y))"),

        // Comparison simplification
        rw!("eq-self"; "(= ?x ?x)" => "true"),
        rw!("neq-self"; "(<> ?x ?x)" => "false"),
    ]
}

pub fn relational_rewrite_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    vec![
        // Filter pushdown through projection
        rw!("filter-project-commute";
            "(filter (project ?input ?exprs) ?pred)" =>
            "(project (filter ?input ?pred) ?exprs)"
            // Condition: pred only references columns from input
        ),

        // Filter combination
        rw!("filter-merge";
            "(filter (filter ?input ?p1) ?p2)" =>
            "(filter ?input (and ?p1 ?p2))"
        ),

        // Projection elimination
        rw!("project-project";
            "(project (project ?input ?e1) ?e2)" =>
            "(project ?input ?e2)"
            // When e2 is a subset of e1
        ),

        // Join commutativity
        rw!("join-comm";
            "(join ?l ?r ?cond inner)" =>
            "(join ?r ?l ?cond inner)"
        ),

        // Join associativity
        rw!("join-assoc";
            "(join (join ?a ?b ?c1 inner) ?d ?c2 inner)" =>
            "(join ?a (join ?b ?d ?c2 inner) ?c1 inner)"
        ),

        // Aggregate pushdown through join
        rw!("agg-pushdown";
            "(aggregate (join ?l ?r ?cond inner) ?gb ?aggs)" =>
            "(join (aggregate ?l ?gb ?aggs) ?r ?cond inner)"
            // When aggregates only reference left side
        ),
    ]
}

pub fn zarr_specific_rules() -> Vec<Rewrite<ZarrPlan, ZarrAnalysis>> {
    vec![
        // Coordinate filter pushdown to scan
        rw!("coord-filter-to-scan";
            "(filter (zarr_scan ?path ?coords ?vars) (= (col ?dim) ?val))" =>
            "(coord_filter (zarr_scan ?path ?coords ?vars) ?dim ?val)"
            // When ?dim is a coordinate dimension
        ),

        // Range filter pushdown
        rw!("range-filter-to-scan";
            "(filter (zarr_scan ?path ?coords ?vars) (between (col ?dim) ?low ?high))" =>
            "(coord_filter (zarr_scan ?path ?coords ?vars) ?dim (list ?low ?high))"
        ),

        // MIN/MAX from statistics
        rw!("min-from-stats";
            "(aggregate (zarr_scan ?path ?coords ?vars) empty (list (min (col ?c))))" =>
            "(project empty (list (lit ?min_val)))"
            // Condition: ?min_val extracted from statistics
        ),

        // COUNT from statistics
        rw!("count-from-stats";
            "(aggregate (zarr_scan ?path ?coords ?vars) empty (list (count ?x)))" =>
            "(project empty (list (lit ?count_val)))"
            // Condition: ?count_val = product of coordinate sizes
        ),

        // Resample optimization for temporal queries
        rw!("resample-pushdown";
            "(aggregate (filter (zarr_scan ?path ?coords ?vars) ?pred) ?gb (list (min ?col)))" =>
            "(resample (filter (zarr_scan ?path ?coords ?vars) ?pred) time ?freq)"
            // For temporal aggregations
        ),
    ]
}
```

### 4. Cost Function

The cost function determines which equivalent plan to extract:

```rust
use egg::{CostFunction, Language};

/// Cost parameters (tunable)
#[derive(Debug, Clone)]
pub struct CostParameters {
    /// Weight for I/O cost (bytes read)
    pub io_weight: f64,
    /// Weight for computation cost
    pub compute_weight: f64,
    /// Weight for memory usage
    pub memory_weight: f64,
    /// Penalty for network I/O (remote Zarr)
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

pub struct ZarrCostFunction<'a> {
    egraph: &'a EGraph<ZarrPlan, ZarrAnalysis>,
    params: &'a CostParameters,
}

impl<'a> CostFunction<ZarrPlan> for ZarrCostFunction<'a> {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &ZarrPlan, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let base_cost: f64 = match enode {
            // I/O operations
            ZarrPlan::ZarrScan(_) | ZarrPlan::Scan(_) => {
                let data = &self.egraph[enode.children()[0]].data;
                let io_cost = data.statistics
                    .as_ref()
                    .map(|s| s.total_byte_size.get_value().unwrap_or(1_000_000) as f64)
                    .unwrap_or(1_000_000.0);
                self.params.io_weight * io_cost
            }

            // Filter is cheap if it pushes down
            ZarrPlan::Filter(_) => 10.0,
            ZarrPlan::CoordFilter(_) => 1.0, // Pushed filter is very cheap

            // Projection is cheap
            ZarrPlan::Project(_) => 5.0,

            // Aggregations depend on input size
            ZarrPlan::Aggregate(_) => {
                let data = &self.egraph[enode.children()[0]].data;
                data.cardinality.unwrap_or(10000) as f64 * 0.01
            }

            // Joins are expensive
            ZarrPlan::Join(_) => {
                let left = &self.egraph[enode.children()[0]].data;
                let right = &self.egraph[enode.children()[1]].data;
                let l_card = left.cardinality.unwrap_or(1000) as f64;
                let r_card = right.cardinality.unwrap_or(1000) as f64;
                l_card * r_card * 0.001 // Assuming good join algorithm
            }

            // Constants are free
            ZarrPlan::Literal(_) | ZarrPlan::Empty => 0.0,

            // Expression operators are cheap
            ZarrPlan::Add(_) | ZarrPlan::Sub(_) | ZarrPlan::Mul(_) | ZarrPlan::Div(_) => 1.0,
            ZarrPlan::And(_) | ZarrPlan::Or(_) | ZarrPlan::Not(_) => 1.0,
            ZarrPlan::Eq(_) | ZarrPlan::Neq(_) | ZarrPlan::Lt(_) | ZarrPlan::Le(_) |
            ZarrPlan::Gt(_) | ZarrPlan::Ge(_) => 1.0,

            // Aggregate functions
            ZarrPlan::Count(_) | ZarrPlan::Sum(_) | ZarrPlan::Avg(_) |
            ZarrPlan::Min(_) | ZarrPlan::Max(_) => 1.0,

            _ => 10.0, // Default cost
        };

        // Add children costs
        enode.children().iter().map(|&c| costs(c)).sum::<f64>() + base_cost
    }
}
```

### 5. Integration with DataFusion

The optimizer integrates as a custom `OptimizerRule`:

```rust
use datafusion::optimizer::{OptimizerRule, OptimizerConfig};
use datafusion::logical_expr::LogicalPlan;
use datafusion::common::Result;
use egg::{Runner, Extractor};

pub struct EggOptimizerRule {
    /// Statistics cache for Zarr tables
    table_stats: Arc<RwLock<HashMap<String, ZarrStoreMeta>>>,
    /// Tunable parameters
    cost_params: CostParameters,
    /// Resource limits for saturation
    runner_limits: RunnerLimits,
}

#[derive(Debug, Clone)]
pub struct RunnerLimits {
    pub iter_limit: usize,
    pub node_limit: usize,
    pub time_limit: Duration,
}

impl Default for RunnerLimits {
    fn default() -> Self {
        Self {
            iter_limit: 30,
            node_limit: 10_000,
            time_limit: Duration::from_secs(5),
        }
    }
}

impl OptimizerRule for EggOptimizerRule {
    fn name(&self) -> &str {
        "egg_optimizer"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // 1. Convert LogicalPlan to RecExpr<ZarrPlan>
        let rec_expr = logical_plan_to_egg(&plan)?;

        // 2. Create e-graph with analysis
        let analysis = ZarrAnalysis {
            table_stats: self.table_stats.read().clone(),
            cost_params: self.cost_params.clone(),
        };
        let runner = Runner::new(analysis)
            .with_expr(&rec_expr)
            .with_iter_limit(self.runner_limits.iter_limit)
            .with_node_limit(self.runner_limits.node_limit)
            .with_time_limit(self.runner_limits.time_limit)
            .run(&all_rules());

        // 3. Extract best plan
        let cost_fn = ZarrCostFunction {
            egraph: &runner.egraph,
            params: &self.cost_params,
        };
        let extractor = Extractor::new(&runner.egraph, cost_fn);
        let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);

        tracing::info!(
            original_nodes = rec_expr.as_ref().len(),
            final_nodes = best_expr.as_ref().len(),
            egraph_classes = runner.egraph.number_of_classes(),
            egraph_nodes = runner.egraph.total_number_of_nodes(),
            iterations = runner.iterations.len(),
            best_cost = best_cost,
            "E-graph optimization complete"
        );

        // 4. Convert back to LogicalPlan
        let optimized = egg_to_logical_plan(&best_expr)?;

        if optimized != plan {
            Ok(Transformed::yes(optimized))
        } else {
            Ok(Transformed::no(plan))
        }
    }
}
```

### 6. Conversion Functions

Bidirectional conversion between DataFusion and egg representations:

```rust
/// Convert DataFusion LogicalPlan to egg RecExpr
pub fn logical_plan_to_egg(plan: &LogicalPlan) -> Result<RecExpr<ZarrPlan>> {
    let mut expr = RecExpr::default();
    convert_plan_recursive(plan, &mut expr)?;
    Ok(expr)
}

fn convert_plan_recursive(plan: &LogicalPlan, expr: &mut RecExpr<ZarrPlan>) -> Result<Id> {
    match plan {
        LogicalPlan::TableScan(scan) => {
            // Check if this is a Zarr table
            let table_id = expr.add(ZarrPlan::Table(scan.table_name.to_string().into()));

            // Convert projection
            let proj_id = if let Some(indices) = &scan.projection {
                let ids: Vec<Id> = indices.iter()
                    .map(|&i| expr.add(ZarrPlan::Literal(i.to_string().into())))
                    .collect();
                expr.add(ZarrPlan::List(ids.into()))
            } else {
                expr.add(ZarrPlan::Empty)
            };

            Ok(expr.add(ZarrPlan::Scan([table_id, proj_id])))
        }

        LogicalPlan::Filter(filter) => {
            let input_id = convert_plan_recursive(&filter.input, expr)?;
            let pred_id = convert_expr_recursive(&filter.predicate, expr)?;
            Ok(expr.add(ZarrPlan::Filter([input_id, pred_id])))
        }

        LogicalPlan::Projection(proj) => {
            let input_id = convert_plan_recursive(&proj.input, expr)?;
            let expr_ids: Vec<Id> = proj.expr.iter()
                .map(|e| convert_expr_recursive(e, expr))
                .collect::<Result<_>>()?;
            let exprs_id = expr.add(ZarrPlan::List(expr_ids.into()));
            Ok(expr.add(ZarrPlan::Project([input_id, exprs_id])))
        }

        LogicalPlan::Aggregate(agg) => {
            let input_id = convert_plan_recursive(&agg.input, expr)?;

            let group_ids: Vec<Id> = agg.group_expr.iter()
                .map(|e| convert_expr_recursive(e, expr))
                .collect::<Result<_>>()?;
            let group_id = if group_ids.is_empty() {
                expr.add(ZarrPlan::Empty)
            } else {
                expr.add(ZarrPlan::List(group_ids.into()))
            };

            let agg_ids: Vec<Id> = agg.aggr_expr.iter()
                .map(|e| convert_expr_recursive(e, expr))
                .collect::<Result<_>>()?;
            let aggs_id = expr.add(ZarrPlan::List(agg_ids.into()));

            Ok(expr.add(ZarrPlan::Aggregate([input_id, group_id, aggs_id])))
        }

        LogicalPlan::Join(join) => {
            let left_id = convert_plan_recursive(&join.left, expr)?;
            let right_id = convert_plan_recursive(&join.right, expr)?;
            let cond_id = join.on.iter()
                .map(|(l, r)| {
                    let l_id = convert_expr_recursive(l, expr)?;
                    let r_id = convert_expr_recursive(r, expr)?;
                    Ok(expr.add(ZarrPlan::Eq([l_id, r_id])))
                })
                .collect::<Result<Vec<_>>>()?;
            let cond_id = if cond_id.is_empty() {
                expr.add(ZarrPlan::Empty)
            } else {
                expr.add(ZarrPlan::List(cond_id.into()))
            };

            let join_type = match join.join_type {
                JoinType::Inner => expr.add(ZarrPlan::Inner),
                JoinType::Left => expr.add(ZarrPlan::Left),
                JoinType::Right => expr.add(ZarrPlan::Right),
                JoinType::Full => expr.add(ZarrPlan::Full),
                _ => expr.add(ZarrPlan::Inner),
            };

            Ok(expr.add(ZarrPlan::Join([left_id, right_id, cond_id, join_type])))
        }

        // ... other plan types
        _ => Err(DataFusionError::NotImplemented(
            format!("E-graph conversion not implemented for: {}", plan.display())
        ))
    }
}

/// Convert egg RecExpr back to DataFusion LogicalPlan
pub fn egg_to_logical_plan(expr: &RecExpr<ZarrPlan>) -> Result<LogicalPlan> {
    // Implementation mirrors logical_plan_to_egg in reverse
    // ...
}
```

## Implementation Plan

### Phase 1: Foundation
1. Define `ZarrPlan` language with core operators
2. Implement `ZarrAnalysis` with basic type tracking
3. Create conversion functions (LogicalPlan ↔ RecExpr)
4. Add expression simplification rules
5. Write unit tests for conversion roundtrip

### Phase 2: Statistics Integration
1. Implement statistics extraction from `ZarrStoreMeta`
2. Add cardinality estimation to analysis
3. Implement MIN/MAX/COUNT constant folding via analysis
4. Create cost function with I/O awareness
5. Write tests for statistics-driven optimization

### Phase 3: Relational Rewrites
1. Add filter pushdown rules
2. Add projection pushdown rules
3. Add join reordering rules (for future multi-table queries)
4. Implement Zarr-specific coordinate filter pushdown

### Phase 4: Extreme Weather Bench Integration
1. Create integration tests from freeze_evaluation_code_flow.md
2. Benchmark query performance with/without e-graph optimizer
3. Tune cost parameters for climate data workloads
4. Add resample/temporal aggregation optimizations

### Phase 5: Caching and Performance
1. Implement e-graph caching for common query patterns
2. Add parallel saturation support
3. Profile and optimize conversion overhead
4. Integrate with DataFusion's optimizer pipeline

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_expression_simplification() {
    let rules = expression_simplification_rules();
    let start: RecExpr<ZarrPlan> = "(* (+ a 0) 1)".parse().unwrap();
    let runner = Runner::default().with_expr(&start).run(&rules);
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(runner.roots[0]);
    assert_eq!(best.to_string(), "a");
}

#[test]
fn test_filter_pushdown() {
    let rules = relational_rewrite_rules();
    // (filter (project scan exprs) pred) => (project (filter scan pred) exprs)
    let start: RecExpr<ZarrPlan> =
        "(filter (project (scan t empty) (list (col a))) (= (col b) (lit 1)))".parse().unwrap();
    // ...
}
```

### Integration Tests
Based on the freeze evaluation queries:

```rust
#[tokio::test]
async fn test_extreme_weather_bench_case30() {
    // Register Zarr tables
    let ctx = SessionContext::new()
        .with_optimizer_rule(Arc::new(EggOptimizerRule::default()));

    ctx.register_table("era5", era5_table).await?;
    ctx.register_table("forecast", forecast_table).await?;

    // Case 30: 2021 Texas Freeze
    let sql = r#"
        WITH case_bounds AS (
            SELECT
                TIMESTAMP '2021-02-10 12:00:00' AS start_date,
                TIMESTAMP '2021-02-22 00:00:00' AS end_date,
                24.0 AS lat_min, 54.75 AS lat_max,
                250.0 AS lon_min, 278.75 AS lon_max
        ),
        aligned_data AS (
            SELECT
                f.lead_time,
                f.surface_air_temperature AS forecast_temp,
                e.surface_air_temperature AS target_temp
            FROM forecast f
            CROSS JOIN case_bounds c
            INNER JOIN era5 e
                ON f.valid_time = e.time
               AND f.latitude = e.latitude
               AND f.longitude = e.longitude
            WHERE e.time BETWEEN c.start_date AND c.end_date
              AND e.latitude BETWEEN c.lat_min AND c.lat_max
              AND e.longitude BETWEEN c.lon_min AND c.lon_max
        )
        SELECT
            lead_time,
            SQRT(AVG(POWER(forecast_temp - target_temp, 2))) AS rmse
        FROM aligned_data
        GROUP BY lead_time
    "#;

    let result = ctx.sql(sql).await?.collect().await?;
    // Verify optimization applied coordinate filters
    // Verify correct results
}
```

## Metrics and Observability

The optimizer will emit metrics via tracing:

```rust
#[derive(Debug)]
pub struct OptimizationMetrics {
    /// Time spent in saturation
    pub saturation_time: Duration,
    /// Time spent in extraction
    pub extraction_time: Duration,
    /// Number of e-graph nodes
    pub egraph_nodes: usize,
    /// Number of e-classes
    pub egraph_classes: usize,
    /// Number of rewrites applied
    pub rewrites_applied: usize,
    /// Cost reduction ratio
    pub cost_reduction: f64,
}
```

## Open Questions

1. **Handling remote statistics**: Should we fetch min/max for remote Zarr stores?
   - Trade-off: Extra I/O vs. better optimization
   - Proposal: Make it configurable via `CostParameters`

2. **E-graph caching**: What's the appropriate cache key?
   - Query template (with placeholders for literals)?
   - Hash of LogicalPlan structure?

3. **Integration point**: Should this run before or after DataFusion's built-in optimizers?
   - Proposal: After, to benefit from their normalization

4. **Join optimization scope**: How much join reordering to support initially?
   - Proposal: Start with 2-way joins, expand later

## References

1. [DataFusion Optimizer Guide](https://datafusion.apache.org/library-user-guide/query-optimizer.html)
2. [egg: Fast and Extensible Equality Saturation](https://egraphs-good.github.io/)
3. [Database Theory in Action: Search-Based Program Optimization](https://drops.dagstuhl.de/storage/00lipics/lipics-vol328-icdt2025/html/LIPIcs.ICDT.2025.34/)
4. [Readings in Database Systems, Ch. 7: Query Optimization](http://www.redbook.io/ch7-queryoptimization.html)
5. [sql-optimizer-labs](https://github.com/risinglightdb/sql-optimizer-labs)
6. [DataFusion-Tokomak](https://github.com/datafusion-contrib/datafusion-tokomak)
