//! Physical optimizer rule to push LIMIT down into ZarrExec
//!
//! This rule enables limit pushdown past FilterExec when the filter is already
//! handled by ZarrExec's coord_filters. DataFusion normally can't push limits
//! past filters because filters might remove rows, but for Zarr with sorted
//! scientific data, the filter is internalized in coord_filters, so we can
//! safely push the limit.
//!
//! ## Problem
//!
//! Without this rule, a query like:
//! ```sql
//! SELECT time, latitude, longitude, temperature
//! FROM era5
//! WHERE latitude BETWEEN 24.0 AND 54.75
//! LIMIT 5
//! ```
//!
//! Results in a plan where the limit is NOT pushed to ZarrExec:
//! ```text
//! CoalescePartitionsExec: fetch=5
//!   FilterExec: latitude >= 24 AND ...
//!     ZarrExec: limit=None          <-- limit NOT pushed!
//! ```
//!
//! ## Solution
//!
//! This rule finds limits in the plan (from CoalescePartitionsExec or GlobalLimitExec)
//! and pushes them into ZarrExec when coord_filters already handle the filtering:
//! ```text
//! CoalescePartitionsExec: fetch=5
//!   FilterExec: latitude >= 24 AND ...
//!     ZarrExec: limit=5, filters=[latitude BETWEEN 24 AND 54.75]  <-- limit PUSHED!
//! ```

use crate::physical_plan::zarr_exec::ZarrExec;
use datafusion::common::Result;
use datafusion::config::ConfigOptions;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::ExecutionPlan;
use std::sync::Arc;
use tracing::{debug, info};

/// Physical optimizer rule that pushes LIMIT into ZarrExec
///
/// This optimization is safe because:
/// 1. ZarrExec already internalizes coordinate filters via coord_filters
/// 2. Coordinate filters on sorted scientific data produce contiguous ranges
/// 3. The limit can be applied during Zarr array reading
#[derive(Debug, Default)]
pub struct ZarrLimitPushdownRule;

impl ZarrLimitPushdownRule {
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for ZarrLimitPushdownRule {
    fn name(&self) -> &str {
        "zarr_limit_pushdown"
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        debug!(
            plan_name = plan.name(),
            "ZarrLimitPushdownRule::optimize called"
        );

        // Extract limit from the top of the plan
        let limit = extract_limit(&plan);

        if let Some(limit_value) = limit {
            debug!(
                limit = limit_value,
                "Found limit at top of plan, attempting pushdown"
            );
            push_limit_to_zarr(plan, Some(limit_value))
        } else {
            debug!("No limit found at top, recursing to find nested limits");
            // No limit found, just recurse to optimize children
            push_limit_to_zarr(plan, None)
        }
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Extract limit value from plan node (CoalescePartitionsExec or GlobalLimitExec)
fn extract_limit(plan: &Arc<dyn ExecutionPlan>) -> Option<usize> {
    // Check for CoalescePartitionsExec with fetch
    if let Some(coalesce) = plan.as_any().downcast_ref::<CoalescePartitionsExec>() {
        if let Some(fetch) = coalesce.fetch() {
            return Some(fetch);
        }
    }

    // Check for GlobalLimitExec
    if let Some(global_limit) = plan.as_any().downcast_ref::<GlobalLimitExec>() {
        // GlobalLimitExec.fetch() returns Option<usize>
        if let Some(fetch) = global_limit.fetch() {
            return Some(fetch);
        }
    }

    None
}

/// Recursively traverse the plan and push limit into ZarrExec
fn push_limit_to_zarr(
    plan: Arc<dyn ExecutionPlan>,
    limit: Option<usize>,
) -> Result<Arc<dyn ExecutionPlan>> {
    debug!(
        plan_name = plan.name(),
        limit = ?limit,
        "push_limit_to_zarr visiting node"
    );

    // Check if current node is ZarrExec
    if let Some(zarr_exec) = plan.as_any().downcast_ref::<ZarrExec>() {
        debug!(
            current_limit = ?zarr_exec.limit(),
            incoming_limit = ?limit,
            "Found ZarrExec"
        );
        // Only push limit if ZarrExec doesn't already have one (or has a larger one)
        let should_push = match (zarr_exec.limit(), limit) {
            (None, Some(_)) => true,
            (Some(existing), Some(new)) if new < existing => true,
            _ => false,
        };

        if should_push {
            let new_limit = limit.unwrap();
            info!(
                limit = new_limit,
                has_coord_filters = zarr_exec.coord_filters().is_some(),
                "Pushing limit into ZarrExec"
            );
            return Ok(Arc::new(zarr_exec.with_limit(limit)));
        }

        debug!("Not pushing limit (already has one or no incoming limit)");
        return Ok(plan);
    }

    // Extract limit from this node if it has one
    let limit_from_node = extract_limit(&plan);
    let effective_limit = match (limit, limit_from_node) {
        (Some(l1), Some(l2)) => Some(l1.min(l2)),
        (Some(l), None) | (None, Some(l)) => Some(l),
        (None, None) => None,
    };

    if limit_from_node.is_some() {
        debug!(
            limit_from_node = ?limit_from_node,
            effective_limit = ?effective_limit,
            "Found limit in current node"
        );
    }

    // Recursively process children
    let children = plan.children();
    debug!(num_children = children.len(), "Processing children");
    if children.is_empty() {
        return Ok(plan);
    }

    let new_children: Result<Vec<Arc<dyn ExecutionPlan>>> = children
        .into_iter()
        .map(|child| push_limit_to_zarr(child.clone(), effective_limit))
        .collect();

    plan.with_new_children(new_children?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_name() {
        let rule = ZarrLimitPushdownRule::new();
        assert_eq!(rule.name(), "zarr_limit_pushdown");
    }
}
