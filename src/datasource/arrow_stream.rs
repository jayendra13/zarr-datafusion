//! Arrow stream table provider for lazy evaluation from external sources.
//!
//! This module provides `ArrowStreamTable`, a DataFusion TableProvider that wraps
//! a factory function producing Arrow record batch streams. This enables lazy
//! evaluation where data is not read until query execution time.
//!
//! The primary use case is integrating external data sources like Xarray Datasets
//! via the Arrow C Data Interface, allowing efficient streaming without
//! materializing entire datasets upfront.

use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::TableProvider;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::memory::MemoryStream;
use datafusion::physical_plan::streaming::PartitionStream;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::SendableRecordBatchStream;
use tracing::info;

use crate::physical_plan::arrow_stream_exec::ArrowStreamExec;

/// A type alias for a factory function that creates record batch iterators.
///
/// The factory is called on each query execution to create a fresh stream,
/// allowing the same table to be queried multiple times.
pub type RecordBatchFactory = Arc<dyn Fn() -> Box<dyn Iterator<Item = RecordBatch> + Send> + Send + Sync>;

/// A partition stream that wraps a factory function for creating record batch iterators.
///
/// The factory is called lazily on each `execute()` invocation, allowing
/// the same table to be queried multiple times.
pub struct ArrowStreamPartition {
    schema: SchemaRef,
    /// A factory that returns a fresh iterator of RecordBatches.
    /// Called on each execute() to create a new stream.
    stream_factory: RecordBatchFactory,
}

impl ArrowStreamPartition {
    /// Create a new ArrowStreamPartition with the given schema and factory.
    pub fn new(schema: SchemaRef, stream_factory: RecordBatchFactory) -> Self {
        Self {
            schema,
            stream_factory,
        }
    }
}

impl Debug for ArrowStreamPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowStreamPartition")
            .field("schema", &self.schema)
            .finish()
    }
}

impl PartitionStream for ArrowStreamPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        // Call the factory to get a fresh iterator for this execution
        let iterator = (self.stream_factory)();

        // Collect batches into a Vec (for MemoryStream compatibility)
        // In a more advanced implementation, this could be truly streaming
        let batches: Vec<RecordBatch> = iterator.collect();

        info!(
            num_batches = batches.len(),
            "ArrowStreamPartition executed"
        );

        Box::pin(
            MemoryStream::try_new(batches, Arc::clone(&self.schema), None)
                .expect("MemoryStream creation should not fail with valid schema"),
        )
    }
}

/// A lazy table provider that wraps a record batch factory function.
///
/// This table provider enables true lazy evaluation: data is NOT read until
/// query execution time. The factory function is called on each query execution
/// to create a fresh stream, allowing the same table to be queried multiple times.
///
/// # Example (Rust)
///
/// ```ignore
/// use arrow::array::{Int64Array, RecordBatch};
/// use arrow::datatypes::{DataType, Field, Schema};
/// use std::sync::Arc;
/// use datafusion::prelude::SessionContext;
/// use zarr_datafusion::datasource::arrow_stream::ArrowStreamTable;
///
/// // Create a factory that produces record batches
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("x", DataType::Int64, false),
/// ]));
/// let schema_clone = schema.clone();
///
/// let factory = Arc::new(move || {
///     let batch = RecordBatch::try_new(
///         schema_clone.clone(),
///         vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
///     ).unwrap();
///     Box::new(std::iter::once(batch)) as Box<dyn Iterator<Item = RecordBatch> + Send>
/// });
///
/// // Create the lazy table - NO DATA LOADED YET
/// let table = ArrowStreamTable::new(schema, factory);
///
/// // Register with DataFusion
/// let ctx = SessionContext::new();
/// ctx.register_table("data", Arc::new(table)).unwrap();
///
/// // Data only loaded HERE during collect()
/// // let result = ctx.sql("SELECT * FROM data").await?.collect().await?;
/// ```
pub struct ArrowStreamTable {
    schema: SchemaRef,
    /// Factory function that creates fresh record batch iterators
    stream_factory: RecordBatchFactory,
}

impl ArrowStreamTable {
    /// Create a new ArrowStreamTable with the given schema and factory.
    ///
    /// # Arguments
    ///
    /// * `schema` - The Arrow schema for the table
    /// * `stream_factory` - A factory function that returns iterators of RecordBatches.
    ///   This is called on each query execution to create a fresh stream.
    pub fn new(schema: SchemaRef, stream_factory: RecordBatchFactory) -> Self {
        Self {
            schema,
            stream_factory,
        }
    }
}

impl Debug for ArrowStreamTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowStreamTable")
            .field("schema", &self.schema)
            .finish()
    }
}

#[async_trait]
impl TableProvider for ArrowStreamTable {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> datafusion::datasource::TableType {
        datafusion::datasource::TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        info!(
            projection = ?projection,
            limit = ?limit,
            "ArrowStreamTable::scan called"
        );

        Ok(Arc::new(ArrowStreamExec::new(
            self.schema.clone(),
            self.stream_factory.clone(),
            projection.cloned(),
            limit,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]))
    }

    fn create_test_factory(schema: SchemaRef) -> RecordBatchFactory {
        Arc::new(move || {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                    Arc::new(Int64Array::from(vec![10, 20, 30])),
                ],
            )
            .unwrap();
            Box::new(std::iter::once(batch)) as Box<dyn Iterator<Item = RecordBatch> + Send>
        })
    }

    #[test]
    fn test_arrow_stream_table_creation() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone());

        let table = ArrowStreamTable::new(schema.clone(), factory);

        assert_eq!(table.schema().fields().len(), 2);
        assert_eq!(table.schema().field(0).name(), "id");
        assert_eq!(table.schema().field(1).name(), "value");
    }

    #[tokio::test]
    async fn test_arrow_stream_table_query() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone());
        let table = ArrowStreamTable::new(schema, factory);

        let ctx = SessionContext::new();
        ctx.register_table("data", Arc::new(table)).unwrap();

        let df = ctx.sql("SELECT * FROM data").await.unwrap();
        let batches = df.collect().await.unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(batches[0].num_columns(), 2);
    }

    #[tokio::test]
    async fn test_arrow_stream_table_lazy_evaluation() {
        // Track when the factory is called
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let schema = create_test_schema();
        let schema_clone = schema.clone();

        let factory: RecordBatchFactory = Arc::new(move || {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            let batch = RecordBatch::try_new(
                schema_clone.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                    Arc::new(Int64Array::from(vec![10, 20, 30])),
                ],
            )
            .unwrap();
            Box::new(std::iter::once(batch)) as Box<dyn Iterator<Item = RecordBatch> + Send>
        });

        // Create table - factory should NOT be called yet
        let table = ArrowStreamTable::new(schema, factory);
        assert_eq!(call_count.load(Ordering::SeqCst), 0, "Factory called during table creation");

        // Register with context - factory should NOT be called yet
        let ctx = SessionContext::new();
        ctx.register_table("data", Arc::new(table)).unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 0, "Factory called during registration");

        // Execute query - factory should be called NOW
        let df = ctx.sql("SELECT * FROM data").await.unwrap();
        let _batches = df.collect().await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1, "Factory should be called during execution");
    }

    #[tokio::test]
    async fn test_arrow_stream_table_multiple_queries() {
        // Track factory calls
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let schema = create_test_schema();
        let schema_clone = schema.clone();

        let factory: RecordBatchFactory = Arc::new(move || {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            let batch = RecordBatch::try_new(
                schema_clone.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                    Arc::new(Int64Array::from(vec![10, 20, 30])),
                ],
            )
            .unwrap();
            Box::new(std::iter::once(batch)) as Box<dyn Iterator<Item = RecordBatch> + Send>
        });

        let table = ArrowStreamTable::new(schema, factory);
        let ctx = SessionContext::new();
        ctx.register_table("data", Arc::new(table)).unwrap();

        // First query
        let df = ctx.sql("SELECT * FROM data").await.unwrap();
        let _batches = df.collect().await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Second query - factory should be called again
        let df = ctx.sql("SELECT * FROM data LIMIT 1").await.unwrap();
        let _batches = df.collect().await.unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 2, "Factory should be called for each query");
    }

    #[tokio::test]
    async fn test_arrow_stream_table_with_projection() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone());
        let table = ArrowStreamTable::new(schema, factory);

        let ctx = SessionContext::new();
        ctx.register_table("data", Arc::new(table)).unwrap();

        let df = ctx.sql("SELECT value FROM data").await.unwrap();
        let batches = df.collect().await.unwrap();

        // Should only have 1 column due to projection
        assert_eq!(batches[0].num_columns(), 1);
        assert_eq!(batches[0].schema().field(0).name(), "value");
    }

    #[tokio::test]
    async fn test_arrow_stream_table_with_limit() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone());
        let table = ArrowStreamTable::new(schema, factory);

        let ctx = SessionContext::new();
        ctx.register_table("data", Arc::new(table)).unwrap();

        let df = ctx.sql("SELECT * FROM data LIMIT 2").await.unwrap();
        let batches = df.collect().await.unwrap();

        // Concatenate all batches to count total rows
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }

    #[tokio::test]
    async fn test_arrow_stream_table_aggregation() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone());
        let table = ArrowStreamTable::new(schema, factory);

        let ctx = SessionContext::new();
        ctx.register_table("data", Arc::new(table)).unwrap();

        let df = ctx.sql("SELECT SUM(value) as total FROM data").await.unwrap();
        let batches = df.collect().await.unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);

        let total = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(total, 60); // 10 + 20 + 30
    }
}
