//! datafusion-distributed head / runner.
//!
//! Plans a SQL query against a Zarr store, distributes the lower stages across
//! the workers listed in `WORKER_URLS`, and runs the top stage locally. Use
//! `--show-plan` to print the distributed plan instead of executing.
//!
//! Usage:
//!   WORKER_URLS=http://worker1:8080,http://worker2:8080 \
//!     head "SELECT lat, AVG(temperature) FROM weather GROUP BY lat" \
//!     [--store /data/synthetic_v3.zarr] [--table weather] [--show-plan]

use std::error::Error;
use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_distributed::{display_plan_ascii, DistributedExt, SessionStateBuilderExt};
use futures::TryStreamExt;

use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::distributed::{configure_distributed_builder, StaticWorkerResolver};
use zarr_datafusion::reader::schema_inference::{
    infer_schema_with_meta, infer_schema_with_meta_async,
};
use zarr_datafusion::reader::storage::{create_async_store, is_remote_url};
use zarr_datafusion::udfs::register_metric_udfs;

struct Args {
    query: String,
    store: String,
    table: String,
    show_plan: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut query: Option<String> = None;
    let mut store =
        std::env::var("STORE_PATH").unwrap_or_else(|_| "/data/synthetic_v3.zarr".into());
    let mut table = "weather".to_string();
    let mut show_plan = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--store" => store = it.next().ok_or("--store needs a value")?,
            "--table" => table = it.next().ok_or("--table needs a value")?,
            "--show-plan" => show_plan = true,
            _ if query.is_none() => query = Some(arg),
            other => return Err(format!("unexpected argument: {other}")),
        }
    }

    Ok(Args {
        query: query.ok_or("missing SQL query (first positional argument)")?,
        store,
        table,
        show_plan,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = parse_args()?;

    let worker_urls = std::env::var("WORKER_URLS")
        .map_err(|_| "WORKER_URLS env var is required (comma-separated worker URLs)")?;
    let resolver = StaticWorkerResolver::from_csv(&worker_urls)?;
    tracing::info!(workers = resolver.urls().len(), "resolved cluster");

    // Head session: discovery + distributed planner + our codec/UDFs.
    let builder = SessionStateBuilder::new()
        .with_default_features()
        .with_distributed_worker_resolver(resolver)
        .with_distributed_planner();
    let state = configure_distributed_builder(builder).build();
    let ctx = SessionContext::from(state);
    // Append the metric UDFs to the context; built-in aggregates (avg/sum/count)
    // stay registered because this appends rather than replacing the function list.
    register_metric_udfs(&ctx);

    // Register the Zarr store as a table. The TableProvider stays on the head;
    // only the physical ZarrExec travels to workers (which rebuild the store
    // from `path` — for a public GCS bucket that's anonymous access).
    let table: Arc<ZarrTable> = if is_remote_url(&args.store) {
        let (store, prefix) = create_async_store(&args.store).await?;
        let (schema, metadata) = infer_schema_with_meta_async(&store, &prefix).await?;
        Arc::new(ZarrTable::with_cached_remote(
            Arc::new(schema),
            args.store.clone(),
            store,
            prefix,
            metadata,
        ))
    } else {
        let (schema, metadata) = infer_schema_with_meta(&args.store)?;
        Arc::new(ZarrTable::with_metadata(
            Arc::new(schema),
            args.store.clone(),
            metadata,
        ))
    };
    ctx.register_table(&args.table, table)?;

    let df = ctx.sql(&args.query).await?;

    if args.show_plan {
        let plan = df.create_physical_plan().await?;
        println!("{}", display_plan_ascii(plan.as_ref(), false));
    } else {
        let batches = df.execute_stream().await?.try_collect::<Vec<_>>().await?;
        println!("{}", pretty_format_batches(&batches)?);
    }

    Ok(())
}
