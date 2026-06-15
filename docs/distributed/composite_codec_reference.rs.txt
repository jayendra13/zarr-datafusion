//! End-to-end test: run a SQL query over a Zarr store on an **in-process**
//! (standalone) Ballista cluster.
//!
//! This boots a real Ballista scheduler + executor inside the test process and
//! drives a query over Arrow Flight / gRPC, so it genuinely exercises both
//! serialization codecs:
//!   * client -> scheduler ships the *logical* plan  -> `ZarrLogicalCodec`
//!   * scheduler -> executor ships the *physical* plan -> `ZarrPhysicalCodec`
//!
//! ## Why composite codecs?
//! Ballista serializes its own shuffle operators (`ShuffleWriterExec`,
//! `ShuffleReaderExec`) *through the same extension codec*, and every Ballista
//! query has at least one shuffle stage. So a bare `ZarrPhysicalCodec` would
//! fail to serialize Ballista's operators. The composite codecs below route
//! Zarr nodes to our codec and delegate everything else to Ballista's default.
//! A one-byte tag prefix makes the decode side unambiguous.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use ballista::prelude::{SessionConfigExt, SessionContextExt};
use ballista_core::serde::{BallistaLogicalExtensionCodec, BallistaPhysicalExtensionCodec};
use datafusion::common::{DataFusionError, Result, TableReference};
use datafusion::datasource::TableProvider;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{AggregateUDF, Extension, LogicalPlan, ScalarUDF, WindowUDF};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::SessionContext;
use datafusion_proto::logical_plan::LogicalExtensionCodec;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;

use zarr_datafusion::datasource::zarr::ZarrTable;
use zarr_datafusion::physical_plan::codec::{ZarrLogicalCodec, ZarrPhysicalCodec};
use zarr_datafusion::physical_plan::zarr_exec::ZarrExec;
use zarr_datafusion::reader::schema_inference::infer_schema_with_meta;

const TAG_ZARR: u8 = 0;
const TAG_OTHER: u8 = 1;

// =============================================================================
// Composite physical codec: ZarrExec -> ours, everything else -> Ballista
// =============================================================================

#[derive(Debug)]
struct CompositePhysicalCodec {
    zarr: ZarrPhysicalCodec,
    inner: BallistaPhysicalExtensionCodec,
}

impl Default for CompositePhysicalCodec {
    fn default() -> Self {
        Self {
            zarr: ZarrPhysicalCodec,
            inner: BallistaPhysicalExtensionCodec::default(),
        }
    }
}

impl PhysicalExtensionCodec for CompositePhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (tag, rest) = buf
            .split_first()
            .ok_or_else(|| DataFusionError::Internal("empty physical codec buffer".into()))?;
        match *tag {
            TAG_ZARR => self.zarr.try_decode(rest, inputs, ctx),
            _ => self.inner.try_decode(rest, inputs, ctx),
        }
    }

    fn try_encode(&self, node: Arc<dyn ExecutionPlan>, buf: &mut Vec<u8>) -> Result<()> {
        if node.as_any().is::<ZarrExec>() {
            buf.push(TAG_ZARR);
            self.zarr.try_encode(node, buf)
        } else {
            buf.push(TAG_OTHER);
            self.inner.try_encode(node, buf)
        }
    }

    // Delegate the rest to Ballista's codec (used for shuffle exprs, UDFs, etc.)
    fn try_decode_expr(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn PhysicalExpr>],
    ) -> Result<Arc<dyn PhysicalExpr>> {
        self.inner.try_decode_expr(buf, inputs)
    }
    fn try_encode_expr(&self, node: &Arc<dyn PhysicalExpr>, buf: &mut Vec<u8>) -> Result<()> {
        self.inner.try_encode_expr(node, buf)
    }
    fn try_decode_udf(&self, name: &str, buf: &[u8]) -> Result<Arc<ScalarUDF>> {
        self.inner.try_decode_udf(name, buf)
    }
    fn try_encode_udf(&self, node: &ScalarUDF, buf: &mut Vec<u8>) -> Result<()> {
        self.inner.try_encode_udf(node, buf)
    }
    fn try_decode_udaf(&self, name: &str, buf: &[u8]) -> Result<Arc<AggregateUDF>> {
        self.inner.try_decode_udaf(name, buf)
    }
    fn try_encode_udaf(&self, node: &AggregateUDF, buf: &mut Vec<u8>) -> Result<()> {
        self.inner.try_encode_udaf(node, buf)
    }
    fn try_decode_udwf(&self, name: &str, buf: &[u8]) -> Result<Arc<WindowUDF>> {
        self.inner.try_decode_udwf(name, buf)
    }
    fn try_encode_udwf(&self, node: &WindowUDF, buf: &mut Vec<u8>) -> Result<()> {
        self.inner.try_encode_udwf(node, buf)
    }
}

// =============================================================================
// Composite logical codec: ZarrTable -> ours, everything else -> Ballista
// =============================================================================

#[derive(Debug)]
struct CompositeLogicalCodec {
    zarr: ZarrLogicalCodec,
    inner: BallistaLogicalExtensionCodec,
}

impl Default for CompositeLogicalCodec {
    fn default() -> Self {
        Self {
            zarr: ZarrLogicalCodec,
            inner: BallistaLogicalExtensionCodec::default(),
        }
    }
}

impl LogicalExtensionCodec for CompositeLogicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[LogicalPlan],
        ctx: &TaskContext,
    ) -> Result<Extension> {
        self.inner.try_decode(buf, inputs, ctx)
    }
    fn try_encode(&self, node: &Extension, buf: &mut Vec<u8>) -> Result<()> {
        self.inner.try_encode(node, buf)
    }

    fn try_decode_table_provider(
        &self,
        buf: &[u8],
        table_ref: &TableReference,
        schema: SchemaRef,
        ctx: &TaskContext,
    ) -> Result<Arc<dyn TableProvider>> {
        let (tag, rest) = buf
            .split_first()
            .ok_or_else(|| DataFusionError::Internal("empty table provider buffer".into()))?;
        match *tag {
            TAG_ZARR => self
                .zarr
                .try_decode_table_provider(rest, table_ref, schema, ctx),
            _ => self
                .inner
                .try_decode_table_provider(rest, table_ref, schema, ctx),
        }
    }

    fn try_encode_table_provider(
        &self,
        table_ref: &TableReference,
        node: Arc<dyn TableProvider>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        if node.as_any().is::<ZarrTable>() {
            buf.push(TAG_ZARR);
            self.zarr.try_encode_table_provider(table_ref, node, buf)
        } else {
            buf.push(TAG_OTHER);
            self.inner.try_encode_table_provider(table_ref, node, buf)
        }
    }

    fn try_decode_file_format(
        &self,
        buf: &[u8],
        ctx: &TaskContext,
    ) -> Result<Arc<dyn datafusion::datasource::file_format::FileFormatFactory>> {
        self.inner.try_decode_file_format(buf, ctx)
    }
    fn try_encode_file_format(
        &self,
        buf: &mut Vec<u8>,
        node: Arc<dyn datafusion::datasource::file_format::FileFormatFactory>,
    ) -> Result<()> {
        self.inner.try_encode_file_format(buf, node)
    }
    fn try_decode_udf(&self, name: &str, buf: &[u8]) -> Result<Arc<ScalarUDF>> {
        self.inner.try_decode_udf(name, buf)
    }
    fn try_encode_udf(&self, node: &ScalarUDF, buf: &mut Vec<u8>) -> Result<()> {
        self.inner.try_encode_udf(node, buf)
    }
    fn try_decode_udaf(&self, name: &str, buf: &[u8]) -> Result<Arc<AggregateUDF>> {
        self.inner.try_decode_udaf(name, buf)
    }
    fn try_encode_udaf(&self, node: &AggregateUDF, buf: &mut Vec<u8>) -> Result<()> {
        self.inner.try_encode_udaf(node, buf)
    }
    fn try_decode_udwf(&self, name: &str, buf: &[u8]) -> Result<Arc<WindowUDF>> {
        self.inner.try_decode_udwf(name, buf)
    }
    fn try_encode_udwf(&self, node: &WindowUDF, buf: &mut Vec<u8>) -> Result<()> {
        self.inner.try_encode_udwf(node, buf)
    }
}

fn assert_send_sync<T: Send + Sync>() {}

// Booting an in-process Ballista cluster pulls in a large dependency tree and
// real gRPC services, so this is excluded from the default `cargo test` run.
// Run it explicitly with:
//     cargo test --test integration_ballista -- --ignored
#[ignore = "boots an in-process Ballista cluster; run with --ignored"]
#[tokio::test(flavor = "multi_thread")]
async fn ballista_standalone_select_star() {
    // Codecs must be Send + Sync to live in the session config.
    assert_send_sync::<CompositePhysicalCodec>();
    assert_send_sync::<CompositeLogicalCodec>();

    let path = "data/synthetic_v3.zarr";

    // Build a Ballista session config carrying both composite codecs.
    let config = datafusion::prelude::SessionConfig::new_with_ballista()
        .with_ballista_logical_extension_codec(Arc::new(CompositeLogicalCodec::default()))
        .with_ballista_physical_extension_codec(Arc::new(CompositePhysicalCodec::default()));

    let state = datafusion::execution::session_state::SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .build();

    // Boot the in-process scheduler + executor from this state.
    let ctx = SessionContext::standalone_with_state(state)
        .await
        .expect("failed to start standalone Ballista");

    // Register the Zarr table (client side).
    let (schema, meta) = infer_schema_with_meta(path).expect("schema inference");
    let table = ZarrTable::with_metadata(Arc::new(schema), path, meta.clone());
    ctx.register_table("data", Arc::new(table))
        .expect("register table");

    // Run the query over the cluster.
    let df = ctx.sql("SELECT * FROM data").await.expect("plan query");
    let batches = df.collect().await.expect("collect results");

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, meta.total_rows, "row count mismatch");
    assert_eq!(batches[0].num_columns(), 5, "expected 5 columns");
}
