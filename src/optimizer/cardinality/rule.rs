//! The cardinality physical-optimizer rule (Phase 5).
//!
//! Walks the physical plan and annotates each [`ZarrExec`] with the memory budget
//! (in bytes) — lifting the policy the reader used to read from the environment
//! inline into a single, discoverable optimizer entry point. The reader still does
//! the mechanics (bytes ÷ its own `row_width` → row target → tiling), so the budget
//! has one authoritative division site (see the Phase-5 A/B decision).
//!
//! Because the rule walks the whole plan tree it can also *see* a scan's parent
//! operators — e.g. an aggregate above a `ZarrExec`. Phase 5 does not act on that;
//! it is the seam Phase 7 (aggregate pushdown) will build on.

use std::sync::Arc;

use datafusion::common::Result;
use datafusion::config::ConfigOptions;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::ExecutionPlan;

use super::budget::{max_groups, MemoryBudget};
use super::pushdown::{max_group_count, recognize};
use crate::physical_plan::zarr_exec::ZarrExec;

/// Physical rule that stamps each `ZarrExec` with the streaming memory budget.
#[derive(Debug)]
pub struct CardinalityRule {
    budget: Option<MemoryBudget>,
}

impl CardinalityRule {
    /// Budget from `ZARR_MEM_BUDGET_BYTES` — matches the reader's env fallback, so
    /// registering the rule is behaviour-neutral until a budget is configured.
    pub fn new() -> Self {
        Self {
            budget: MemoryBudget::from_env(),
        }
    }

    /// Construct with an explicit budget (for tests / programmatic sessions).
    pub fn with_budget(budget: Option<MemoryBudget>) -> Self {
        Self { budget }
    }

    /// Observe-only (Phase 7.2): if this node is an `AggregateExec` directly over a
    /// `ZarrExec` with a pushable shape, log the recognized candidate and whether its
    /// (upper-bound) group count fits the group budget. Drives no rewrite yet — this
    /// is the seam Phase 7.3 will act on. Costs nothing when debug logging is off.
    fn observe_pushdown(&self, plan: &Arc<dyn ExecutionPlan>) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let Some(agg) = plan.downcast_ref::<AggregateExec>() else {
            return;
        };
        let Some(zarr) = agg.input().downcast_ref::<ZarrExec>() else {
            return;
        };
        let Some(cand) = recognize(agg, zarr) else {
            return;
        };
        let Some(meta) = zarr.store_meta() else {
            return;
        };
        let groups = max_group_count(&cand, meta);
        let cap = max_groups();
        tracing::debug!(
            aggs = ?cand.aggs,
            group_by = ?cand.group_names,
            max_groups = %groups,
            cap = %cap,
            pushable = groups <= cap,
            "aggregate pushdown candidate (observe-only, Phase 7.2)"
        );
    }

    /// Recursively stamp every `ZarrExec` with the budget; leave all else untouched.
    fn annotate(&self, plan: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        self.observe_pushdown(&plan);
        if let Some(zarr) = plan.downcast_ref::<ZarrExec>() {
            // No budget configured => leave the node as-is (reader uses batch_size).
            let Some(budget) = self.budget else {
                return Ok(plan);
            };
            return Ok(Arc::new(
                zarr.clone().with_stream_budget_bytes(Some(budget.bytes)),
            ));
        }
        let children = plan.children();
        if children.is_empty() {
            return Ok(plan);
        }
        let new_children = children
            .into_iter()
            .map(|c| self.annotate(Arc::clone(c)))
            .collect::<Result<Vec<_>>>()?;
        plan.with_new_children(new_children)
    }
}

impl Default for CardinalityRule {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizerRule for CardinalityRule {
    fn name(&self) -> &str {
        "cardinality"
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.annotate(plan)
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::empty::EmptyExec;
    use std::sync::Arc;

    fn zarr_exec() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("lat", DataType::Float64, false),
            Field::new("temperature", DataType::Float64, true),
        ]));
        Arc::new(ZarrExec::new(
            schema,
            "dummy".into(),
            None,
            None,
            None,
            None,
            None,
        ))
    }

    fn budget_of(plan: &Arc<dyn ExecutionPlan>) -> Option<u64> {
        plan.downcast_ref::<ZarrExec>()
            .expect("ZarrExec")
            .stream_budget_bytes()
    }

    #[test]
    fn stamps_budget_on_zarr_exec() {
        let rule = CardinalityRule::with_budget(Some(MemoryBudget::new(4096)));
        let out = rule.annotate(zarr_exec()).unwrap();
        assert_eq!(budget_of(&out), Some(4096));
    }

    #[test]
    fn idempotent() {
        let rule = CardinalityRule::with_budget(Some(MemoryBudget::new(4096)));
        let once = rule.annotate(zarr_exec()).unwrap();
        let twice = rule.annotate(once).unwrap();
        assert_eq!(budget_of(&twice), Some(4096));
    }

    #[test]
    fn no_op_without_budget() {
        let rule = CardinalityRule::with_budget(None);
        let out = rule.annotate(zarr_exec()).unwrap();
        assert_eq!(budget_of(&out), None);
    }

    #[test]
    fn no_op_on_non_zarr_plan() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(schema));
        let rule = CardinalityRule::with_budget(Some(MemoryBudget::new(4096)));
        let out = rule.annotate(plan).unwrap();
        assert!(out.downcast_ref::<EmptyExec>().is_some());
    }
}
