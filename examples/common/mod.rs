//! Common utilities shared across zarr-datafusion examples
//!
//! This module provides helper functions for initializing tracing,
//! creating DataFusion session contexts, and running queries.
//!
//! # Usage
//!
//! ```rust,ignore
//! mod common;
//!
//! fn main() {
//!     common::init_tracing();
//!     let ctx = common::create_local_context();
//!     // ... use ctx ...
//! }
//! ```

// Allow unused functions - this is a shared utility module and not all
// functions are used by every example
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::error::Result;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use tracing_subscriber::EnvFilter;
use zarr_datafusion::datasource::factory::ZarrTableFactory;
use zarr_datafusion::optimizer::{CountStatisticsRule, MinMaxStatisticsRule};

/// Initialize the tracing subscriber with environment-based filtering.
///
/// Control log level via RUST_LOG environment variable:
/// - `RUST_LOG=info` - info and above
/// - `RUST_LOG=debug` - debug and above
/// - `RUST_LOG=zarr_datafusion=debug` - only zarr_datafusion debug logs
pub fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(true)
        .with_line_number(true)
        .init();
}

/// Create a SessionContext configured for local Zarr file access.
///
/// Includes the CountStatisticsRule and MinMaxStatisticsRule optimizers
/// for efficient count(*) and min/max queries on coordinates.
pub fn create_local_context() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_optimizer_rule(Arc::new(CountStatisticsRule::new()))
        .with_optimizer_rule(Arc::new(MinMaxStatisticsRule::new()))
        .build();
    SessionContext::new_with_state(state)
}

/// Create a SessionContext configured for remote Zarr access (GCS, S3, etc).
///
/// Includes ZarrTableFactory for `CREATE EXTERNAL TABLE ... STORED AS ZARR`,
/// plus CountStatisticsRule and MinMaxStatisticsRule optimizers.
pub fn create_remote_context() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_table_factories(HashMap::from([(
            "ZARR".to_string(),
            Arc::new(ZarrTableFactory) as _,
        )]))
        .with_optimizer_rule(Arc::new(CountStatisticsRule::new()))
        .with_optimizer_rule(Arc::new(MinMaxStatisticsRule::new()))
        .build();
    SessionContext::new_with_state(state)
}

/// Execute a SQL query and display results with a description.
///
/// Prints the description, SQL statement, and tabular results.
pub async fn run_query(ctx: &SessionContext, description: &str, sql: &str) -> Result<()> {
    println!("\n{description}");
    println!("SQL: {sql}");
    println!();
    let df = ctx.sql(sql).await?;
    df.show().await?;
    Ok(())
}
