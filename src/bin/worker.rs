//! datafusion-distributed worker.
//!
//! A long-running gRPC server that receives serialized physical stages from a
//! head node, runs them with the local DataFusion runtime, and streams Arrow
//! batches back. Each worker registers the same [`ZarrPhysicalCodec`] and metric
//! UDFs as the head (via [`configure_distributed_builder`]) so it can decode and
//! execute `ZarrExec`.
//!
//! It needs neither a `WorkerResolver` nor the distributed planner — only the
//! head plans and discovers workers. The store is rebuilt from the `ZarrExec`
//! path on execution, so every worker must resolve that path identically (a
//! shared volume locally, or the same `gs://`/`s3://` URL in the cloud).
//!
//! Config via env: `PORT` (default 8080), `COMMIT_HASH` (worker version tag).

use std::error::Error;
use std::net::SocketAddr;

use datafusion::common::DataFusionError;
use datafusion::execution::SessionState;
use datafusion::prelude::SessionContext;
use datafusion_distributed::{Worker, WorkerQueryContext};
use tonic::transport::Server;

use zarr_datafusion::distributed::configure_distributed_builder;
use zarr_datafusion::udfs::register_metric_udfs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let version = std::env::var("COMMIT_HASH").unwrap_or_default();

    let worker = Worker::from_session_builder(build_session).with_version(version);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!(%addr, "starting zarr-datafusion worker");

    Server::builder()
        .add_service(worker.into_worker_server())
        .serve(addr)
        .await?;

    Ok(())
}

/// Build the per-query session state on the worker: the pre-populated builder
/// from datafusion-distributed plus our codec and UDFs. The UDFs are registered
/// on the context (which appends) rather than the builder (which would replace
/// and drop the built-in aggregates a partial-aggregate stage relies on).
async fn build_session(ctx: WorkerQueryContext) -> Result<SessionState, DataFusionError> {
    let session_ctx = SessionContext::from(configure_distributed_builder(ctx.builder).build());
    register_metric_udfs(&session_ctx);
    Ok(session_ctx.state())
}
