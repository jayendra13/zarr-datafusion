//! Phase 7.2 — aggregate-pushdown *recognition* (observe-only).
//!
//! These drive real SQL through DataFusion to a physical plan, locate the
//! `AggregateExec` sitting directly over the `ZarrExec`, and assert what the
//! recognizer makes of it. No rewrite happens yet (7.3); this pins the matcher's
//! accept/decline behavior.

mod common;

use common::*;

use std::sync::Arc;

use datafusion::physical_plan::aggregates::AggregateExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;

use zarr_datafusion::optimizer::cardinality::pushdown::{recognize, AggKind};
use zarr_datafusion::optimizer::cardinality::GroupKey;
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;

/// Find the first `AggregateExec` whose input is a `ZarrExec`.
fn find_agg_over_zarr(plan: &Arc<dyn ExecutionPlan>) -> Option<(&AggregateExec, &ZarrExec)> {
    if let Some(agg) = plan.downcast_ref::<AggregateExec>() {
        if let Some(z) = agg.input().downcast_ref::<ZarrExec>() {
            return Some((agg, z));
        }
    }
    for c in plan.children() {
        if let Some(found) = find_agg_over_zarr(c) {
            return Some(found);
        }
    }
    None
}

async fn plan_for(sql: &str) -> Arc<dyn ExecutionPlan> {
    // Plain context (no stats rules) so MIN/MAX/COUNT aren't answered logically and
    // a real AggregateExec survives into the physical plan.
    let ctx = SessionContext::new();
    register_zarr_table(&ctx, "t", SYNTHETIC_V3);
    get_physical_plan(&ctx, sql).await
}

#[tokio::test]
async fn recognizes_global_sum_over_data_var() {
    let plan = plan_for("SELECT SUM(temperature) AS s FROM t").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    let cand = recognize(agg, z).expect("recognized as pushable");
    assert_eq!(cand.aggs.len(), 1);
    assert_eq!(cand.aggs[0].kind, AggKind::Sum);
    assert_eq!(cand.aggs[0].column.as_deref(), Some("temperature"));
    assert!(cand.group_keys.is_empty(), "no GROUP BY => global aggregate");
}

#[tokio::test]
async fn recognizes_group_by_coordinate() {
    // Synthetic cube axes are [time, lat, lon]; GROUP BY lat => axis 1.
    let plan = plan_for("SELECT lat, AVG(temperature) AS a FROM t GROUP BY lat").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    let cand = recognize(agg, z).expect("recognized as pushable");
    assert_eq!(cand.aggs[0].kind, AggKind::Avg);
    assert_eq!(cand.group_keys, vec![GroupKey::Axis(1)]);
    assert_eq!(cand.group_names, vec!["lat".to_string()]);
}

#[tokio::test]
async fn declines_distinct_aggregate() {
    let plan = plan_for("SELECT COUNT(DISTINCT temperature) AS c FROM t").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    assert!(recognize(agg, z).is_none(), "DISTINCT is not pushable");
}

#[tokio::test]
async fn declines_non_count_aggregate_over_coordinate() {
    // MAX over a coordinate (not a data variable) isn't a pushable data aggregate.
    let plan = plan_for("SELECT MAX(lat) AS m FROM t").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    assert!(recognize(agg, z).is_none());
}

#[tokio::test]
async fn declines_group_by_data_variable() {
    // Grouping by a data variable has no cube-axis mapping.
    let plan = plan_for("SELECT temperature, COUNT(*) AS c FROM t GROUP BY temperature").await;
    let (agg, z) = find_agg_over_zarr(&plan).expect("AggregateExec over ZarrExec");
    assert!(recognize(agg, z).is_none());
}
