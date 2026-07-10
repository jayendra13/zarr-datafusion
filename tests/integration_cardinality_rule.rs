//! Phase 5.3: `CardinalityRule` end-to-end — the rule stamps the budget onto the
//! `ZarrExec` at plan time and leaves results unchanged.
//!
//! Uses an *explicit* budget (`with_budget`), not `ZARR_MEM_BUDGET_BYTES`, so the
//! test is deterministic and doesn't race other tests via the environment.

mod common;

use common::*;

use std::sync::Arc;

use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;

use zarr_datafusion::optimizer::cardinality::budget::MemoryBudget;
use zarr_datafusion::optimizer::CardinalityRule;

/// A session with only the cardinality rule registered, carrying `budget`.
fn ctx_with_rule(budget: Option<MemoryBudget>) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_physical_optimizer_rule(Arc::new(CardinalityRule::with_budget(budget)))
        .build();
    SessionContext::new_with_state(state)
}

#[tokio::test]
async fn rule_stamps_budget_on_zarr_exec() {
    let ctx = ctx_with_rule(Some(MemoryBudget::new(256)));
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);

    let plan = get_physical_plan(&ctx, "SELECT temperature FROM data").await;
    let zarr = find_zarr_exec(&plan).expect("ZarrExec present in plan");
    assert_eq!(zarr.stream_budget_bytes(), Some(256));
}

#[tokio::test]
async fn rule_preserves_row_count() {
    // A tiny budget drives fine tiling in the scan; the result's row count must be
    // unchanged (tiling changes batch granularity, never the rows). Value-level
    // transparency of tiling is covered exhaustively by integration_streaming.
    let with_budget = ctx_with_rule(Some(MemoryBudget::new(256)));
    register_zarr_table(&with_budget, "data", SYNTHETIC_V3);
    let tiled = execute_query(&with_budget, "SELECT temperature FROM data").await;
    let tiled_rows: usize = tiled.iter().map(|b| b.num_rows()).sum();

    let no_budget = ctx_with_rule(None);
    register_zarr_table(&no_budget, "data", SYNTHETIC_V3);
    let base = execute_query(&no_budget, "SELECT temperature FROM data").await;
    let base_rows: usize = base.iter().map(|b| b.num_rows()).sum();

    assert_eq!(tiled_rows, 700);
    assert_eq!(
        tiled_rows, base_rows,
        "budget-driven tiling must preserve all rows"
    );
}

#[tokio::test]
async fn no_budget_leaves_exec_unstamped() {
    let ctx = ctx_with_rule(None);
    register_zarr_table(&ctx, "data", SYNTHETIC_V3);
    let plan = get_physical_plan(&ctx, "SELECT temperature FROM data").await;
    let zarr = find_zarr_exec(&plan).expect("ZarrExec present in plan");
    assert_eq!(zarr.stream_budget_bytes(), None);
}
