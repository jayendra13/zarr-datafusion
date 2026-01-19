//! Execution plan for Arrow stream data sources.
//!
//! This module provides `ArrowStreamExec`, a DataFusion ExecutionPlan that
//! lazily executes a factory function to produce record batch streams.
//! This supports the lazy evaluation pattern where data is not read until
//! query execution time.

use std::fmt::Debug;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::memory::MemoryStream;
use datafusion::physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties};
use datafusion::physical_plan::SendableRecordBatchStream;
use tracing::info;

/// A type alias for a factory function that creates record batch iterators.
pub type RecordBatchFactory = Arc<dyn Fn() -> Box<dyn Iterator<Item = RecordBatch> + Send> + Send + Sync>;

/// Execution plan for Arrow stream data sources.
///
/// This plan lazily executes a factory function to produce record batches
/// during query execution. The factory is called each time `execute()` is
/// called, allowing for multiple queries against the same data source.
pub struct ArrowStreamExec {
    /// Full schema of the data source
    schema: SchemaRef,
    /// Factory function that creates fresh record batch iterators
    stream_factory: RecordBatchFactory,
    /// Column indices to project (None = all columns)
    projection: Option<Vec<usize>>,
    /// Maximum number of rows to return
    limit: Option<usize>,
    /// Plan properties (cached)
    properties: PlanProperties,
}

impl ArrowStreamExec {
    /// Create a new ArrowStreamExec.
    ///
    /// # Arguments
    ///
    /// * `schema` - The full schema of the data source
    /// * `stream_factory` - Factory function that creates record batch iterators
    /// * `projection` - Optional column indices to project
    /// * `limit` - Optional maximum number of rows to return
    pub fn new(
        schema: SchemaRef,
        stream_factory: RecordBatchFactory,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
    ) -> Self {
        // Compute projected schema for plan properties
        let projected_schema = if let Some(ref indices) = projection {
            Arc::new(Schema::new(
                indices
                    .iter()
                    .map(|&i| schema.field(i).clone())
                    .collect::<Vec<_>>(),
            ))
        } else {
            schema.clone()
        };

        let properties = PlanProperties::new(
            EquivalenceProperties::new(projected_schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Self {
            schema,
            stream_factory,
            projection,
            limit,
            properties,
        }
    }

    /// Get the current limit
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Create a new ArrowStreamExec with the given limit
    pub fn with_limit(&self, limit: Option<usize>) -> Self {
        Self::new(
            self.schema.clone(),
            self.stream_factory.clone(),
            self.projection.clone(),
            limit,
        )
    }
}

impl Debug for ArrowStreamExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrowStreamExec")
            .field("schema", &self.schema)
            .field("projection", &self.projection)
            .field("limit", &self.limit)
            .finish()
    }
}

impl DisplayAs for ArrowStreamExec {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let mut parts = vec!["ArrowStreamExec".to_string()];
        if let Some(limit) = self.limit {
            parts.push(format!("limit={}", limit));
        }
        if let Some(ref proj) = self.projection {
            parts.push(format!("projection={:?}", proj));
        }
        write!(f, "{}", parts.join(", "))
    }
}

impl ExecutionPlan for ArrowStreamExec {
    fn name(&self) -> &str {
        "ArrowStreamExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion::error::Result<SendableRecordBatchStream> {
        info!(
            limit = ?self.limit,
            projection = ?self.projection,
            "ArrowStreamExec::execute called"
        );

        // Call the factory to get a fresh iterator
        let iterator = (self.stream_factory)();

        // Collect batches, applying limit if specified
        let batches: Vec<RecordBatch> = if let Some(limit) = self.limit {
            let mut result = Vec::new();
            let mut total_rows = 0;

            for batch in iterator {
                let rows_to_take = (limit - total_rows).min(batch.num_rows());
                if rows_to_take > 0 {
                    let sliced = batch.slice(0, rows_to_take);
                    result.push(sliced);
                    total_rows += rows_to_take;
                }
                if total_rows >= limit {
                    break;
                }
            }
            result
        } else {
            iterator.collect()
        };

        // Apply projection if specified
        let batches: Vec<RecordBatch> = if let Some(ref projection) = self.projection {
            batches
                .into_iter()
                .map(|batch| {
                    let columns: Vec<_> = projection
                        .iter()
                        .map(|&i| batch.column(i).clone())
                        .collect();
                    let projected_schema = Arc::new(Schema::new(
                        projection
                            .iter()
                            .map(|&i| self.schema.field(i).clone())
                            .collect::<Vec<_>>(),
                    ));
                    RecordBatch::try_new(projected_schema, columns)
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?
        } else {
            batches
        };

        info!(
            num_batches = batches.len(),
            total_rows = batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            "ArrowStreamExec execution complete"
        );

        // Get the schema for the output stream from equivalence properties
        let output_schema = self.properties.equivalence_properties().schema().clone();

        Ok(Box::pin(MemoryStream::try_new(batches, output_schema, None)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn create_test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]))
    }

    fn create_test_factory(schema: SchemaRef, num_batches: usize, rows_per_batch: usize) -> RecordBatchFactory {
        Arc::new(move || {
            let batches: Vec<RecordBatch> = (0..num_batches)
                .map(|batch_idx| {
                    let start = (batch_idx * rows_per_batch) as i64;
                    let ids: Vec<i64> = (start..start + rows_per_batch as i64).collect();
                    let values: Vec<i64> = ids.iter().map(|&x| x * 10).collect();

                    RecordBatch::try_new(
                        schema.clone(),
                        vec![
                            Arc::new(Int64Array::from(ids)),
                            Arc::new(Int64Array::from(values)),
                        ],
                    )
                    .unwrap()
                })
                .collect();

            Box::new(batches.into_iter()) as Box<dyn Iterator<Item = RecordBatch> + Send>
        })
    }

    #[test]
    fn test_arrow_stream_exec_creation() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone(), 1, 3);

        let exec = ArrowStreamExec::new(schema.clone(), factory, None, None);

        assert_eq!(
            exec.properties()
                .equivalence_properties()
                .schema()
                .fields()
                .len(),
            2
        );
        assert!(exec.limit().is_none());
    }

    #[test]
    fn test_arrow_stream_exec_with_limit() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone(), 1, 10);

        let exec = ArrowStreamExec::new(schema, factory, None, Some(5));

        assert_eq!(exec.limit(), Some(5));
    }

    #[tokio::test]
    async fn test_arrow_stream_exec_execute() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone(), 2, 5);

        let exec = ArrowStreamExec::new(schema, factory, None, None);
        let ctx = Arc::new(TaskContext::default());

        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 10); // 2 batches * 5 rows
    }

    #[tokio::test]
    async fn test_arrow_stream_exec_with_limit_execution() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone(), 3, 10);

        let exec = ArrowStreamExec::new(schema, factory, None, Some(7));
        let ctx = Arc::new(TaskContext::default());

        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 7);
    }

    #[tokio::test]
    async fn test_arrow_stream_exec_with_projection() {
        let schema = create_test_schema();
        let factory = create_test_factory(schema.clone(), 1, 5);

        // Project only the "value" column (index 1)
        let exec = ArrowStreamExec::new(schema, factory, Some(vec![1]), None);
        let ctx = Arc::new(TaskContext::default());

        let stream = exec.execute(0, ctx).unwrap();
        let batches: Vec<RecordBatch> = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();

        assert_eq!(batches[0].num_columns(), 1);
        assert_eq!(batches[0].schema().field(0).name(), "value");
    }

    #[tokio::test]
    async fn test_arrow_stream_exec_factory_called_on_execute() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let schema = create_test_schema();
        let schema_clone = schema.clone();

        let factory: RecordBatchFactory = Arc::new(move || {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            let batch = RecordBatch::try_new(
                schema_clone.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![1])),
                    Arc::new(Int64Array::from(vec![10])),
                ],
            )
            .unwrap();
            Box::new(std::iter::once(batch)) as Box<dyn Iterator<Item = RecordBatch> + Send>
        });

        let exec = ArrowStreamExec::new(schema, factory, None, None);

        // Factory should not be called yet
        assert_eq!(call_count.load(Ordering::SeqCst), 0);

        // Execute - factory should be called
        let ctx = Arc::new(TaskContext::default());
        let _stream = exec.execute(0, ctx.clone()).unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Execute again - factory should be called again
        let _stream2 = exec.execute(0, ctx).unwrap();
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }
}
