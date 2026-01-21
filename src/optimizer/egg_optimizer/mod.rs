//! E-graph based query optimizer using the egg library
//!
//! This module implements query optimization using equality saturation,
//! which explores the space of equivalent query plans and extracts the
//! lowest-cost plan based on Zarr-specific statistics.
//!
//! # Architecture
//!
//! The optimizer consists of:
//! - `ZarrPlan`: A language defining query plan nodes for e-graphs
//! - `ZarrAnalysis`: Analysis tracking types, statistics, and cardinality
//! - Rewrite rules: Expression simplification and relational rewrites
//! - `ZarrCostFunction`: Cost-based plan extraction using I/O statistics
//!
//! # Usage
//!
//! ```ignore
//! let optimizer = EggOptimizerRule::new();
//! let ctx = SessionContext::new()
//!     .with_optimizer_rule(Arc::new(optimizer));
//! ```

mod analysis;
mod conversion;
mod cost;
mod language;
mod rules;

pub use analysis::{ZarrAnalysis, ZarrAnalysisData};
pub use conversion::{egg_to_logical_plan, logical_plan_to_egg};
pub use cost::{CostParameters, ZarrCostFunction};
pub use language::ZarrPlan;
pub use rules::{all_rules, expression_simplification_rules, relational_rewrite_rules};

use datafusion::common::tree_node::Transformed;
use datafusion::common::Result;
use datafusion::logical_expr::LogicalPlan;
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use egg::{Extractor, Runner};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::{debug, info, trace, warn};

use crate::reader::schema_inference::ZarrStoreMeta;

/// Resource limits for equality saturation
#[derive(Debug, Clone)]
pub struct RunnerLimits {
    /// Maximum number of iterations
    pub iter_limit: usize,
    /// Maximum number of e-graph nodes
    pub node_limit: usize,
    /// Maximum time for saturation
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

/// E-graph based optimizer rule for DataFusion
///
/// This rule converts DataFusion's LogicalPlan to an e-graph representation,
/// applies equality saturation with rewrite rules, and extracts the lowest-cost
/// equivalent plan.
pub struct EggOptimizerRule {
    /// Statistics cache for Zarr tables (table name -> metadata)
    table_stats: Arc<RwLock<HashMap<String, ZarrStoreMeta>>>,
    /// Tunable cost parameters
    cost_params: CostParameters,
    /// Resource limits for saturation
    runner_limits: RunnerLimits,
    /// Whether the optimizer is enabled
    enabled: bool,
}

impl Default for EggOptimizerRule {
    fn default() -> Self {
        Self::new()
    }
}

impl EggOptimizerRule {
    /// Create a new e-graph optimizer with default settings
    pub fn new() -> Self {
        Self {
            table_stats: Arc::new(RwLock::new(HashMap::new())),
            cost_params: CostParameters::default(),
            runner_limits: RunnerLimits::default(),
            enabled: true,
        }
    }

    /// Set cost parameters
    pub fn with_cost_params(mut self, params: CostParameters) -> Self {
        self.cost_params = params;
        self
    }

    /// Set runner limits
    pub fn with_runner_limits(mut self, limits: RunnerLimits) -> Self {
        self.runner_limits = limits;
        self
    }

    /// Register statistics for a table
    pub fn register_table_stats(&self, table_name: &str, stats: ZarrStoreMeta) {
        let mut cache = self.table_stats.write().unwrap();
        cache.insert(table_name.to_string(), stats);
    }

    /// Enable or disable the optimizer
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl std::fmt::Debug for EggOptimizerRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EggOptimizerRule")
            .field("cost_params", &self.cost_params)
            .field("runner_limits", &self.runner_limits)
            .field("enabled", &self.enabled)
            .finish()
    }
}

impl OptimizerRule for EggOptimizerRule {
    fn name(&self) -> &str {
        "egg_optimizer"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        // Run top-down to optimize the full plan at once
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
        if !self.enabled {
            trace!("E-graph optimizer disabled, skipping");
            return Ok(Transformed::no(plan));
        }

        // Try to convert to e-graph representation
        let rec_expr = match logical_plan_to_egg(&plan) {
            Ok(expr) => expr,
            Err(e) => {
                debug!(error = %e, "Could not convert plan to e-graph, skipping optimization");
                return Ok(Transformed::no(plan));
            }
        };

        let original_nodes = rec_expr.as_ref().len();
        debug!(
            nodes = original_nodes,
            "Converted LogicalPlan to e-graph representation"
        );

        // Create analysis with table statistics
        let table_stats = self.table_stats.read().unwrap().clone();
        let analysis = ZarrAnalysis::new(table_stats, self.cost_params.clone());

        // Run equality saturation
        let rules = all_rules();
        let runner: Runner<ZarrPlan, ZarrAnalysis, ()> = Runner::new(analysis)
            .with_expr(&rec_expr)
            .with_iter_limit(self.runner_limits.iter_limit)
            .with_node_limit(self.runner_limits.node_limit)
            .with_time_limit(self.runner_limits.time_limit)
            .run(&rules);

        let stop_reason = runner.stop_reason.as_ref().map(|r| format!("{:?}", r));
        let iterations = runner.iterations.len();
        let egraph_classes = runner.egraph.number_of_classes();
        let egraph_nodes = runner.egraph.total_number_of_nodes();

        debug!(
            stop_reason = ?stop_reason,
            iterations,
            egraph_classes,
            egraph_nodes,
            "Equality saturation completed"
        );

        // Extract the best plan
        let cost_fn = ZarrCostFunction::new(&runner.egraph, &self.cost_params);
        let extractor = Extractor::new(&runner.egraph, cost_fn);
        let (best_cost, best_expr) = extractor.find_best(runner.roots[0]);

        let final_nodes = best_expr.as_ref().len();

        info!(
            original_nodes,
            final_nodes,
            egraph_classes,
            egraph_nodes,
            iterations,
            best_cost,
            "E-graph optimization complete"
        );

        // Convert back to LogicalPlan
        let optimized = match egg_to_logical_plan(&best_expr, &plan) {
            Ok(p) => p,
            Err(e) => {
                warn!(error = %e, "Could not convert optimized e-graph back to LogicalPlan");
                return Ok(Transformed::no(plan));
            }
        };

        // Only report a transformation if the plan actually changed
        if format!("{:?}", optimized) != format!("{:?}", plan) {
            debug!("Plan was transformed by e-graph optimizer");
            Ok(Transformed::yes(optimized))
        } else {
            trace!("Plan unchanged by e-graph optimizer");
            Ok(Transformed::no(plan))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = EggOptimizerRule::new();
        assert!(optimizer.enabled);
        assert_eq!(optimizer.name(), "egg_optimizer");
    }

    #[test]
    fn test_optimizer_with_custom_params() {
        let params = CostParameters {
            io_weight: 2.0,
            compute_weight: 0.5,
            memory_weight: 1.0,
            remote_penalty: 5.0,
        };
        let optimizer = EggOptimizerRule::new().with_cost_params(params.clone());
        assert_eq!(optimizer.cost_params.io_weight, 2.0);
    }
}
