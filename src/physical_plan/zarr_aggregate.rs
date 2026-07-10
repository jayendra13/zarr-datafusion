//! `ZarrAggregateExec` — global aggregate pushdown (Phase 7.3).
//!
//! A recognized `SUM`/`COUNT`/`AVG`/`MIN`/`MAX` over a `ZarrExec` with no `GROUP BY`
//! is rewritten (by [`crate::optimizer::CardinalityRule`]) into this operator, which
//! **replaces the whole `AggregateExec ← ZarrExec` subtree**. It drives the
//! (data-variable-projected, streaming) `ZarrExec` child and folds its batches into
//! accumulators instead of letting DataFusion hash-aggregate flattened rows — the
//! group count is exactly 1, so the fold is trivially memory-bounded. Emits a single
//! final row in the aggregate's output schema.
//!
//! This is the self-contained "approach A" foundation; `GROUP BY` (per-axis and
//! periodic) builds on it in Phases 7.4/7.5.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use futures::TryStreamExt;

use crate::optimizer::cardinality::pushdown::{AggKind, AggSpec};

/// One accumulator, enough to finalize any supported aggregate.
#[derive(Clone, Default)]
struct Acc {
    count: u128,
    sum: f64,
    min: Option<f64>,
    max: Option<f64>,
}

impl Acc {
    fn push(&mut self, v: f64) {
        self.count += 1;
        self.sum += v;
        self.min = Some(self.min.map_or(v, |m| m.min(v)));
        self.max = Some(self.max.map_or(v, |m| m.max(v)));
    }
}

/// Physical operator: fold a `ZarrExec` scan into one row of global aggregates.
pub struct ZarrAggregateExec {
    /// The scan to read (already projected to the aggregates' argument columns).
    input: Arc<dyn ExecutionPlan>,
    /// Aggregates to compute, in the output schema's column order.
    aggs: Vec<AggSpec>,
    /// The aggregate's output schema (one row).
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl ZarrAggregateExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, aggs: Vec<AggSpec>, schema: SchemaRef) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            input,
            aggs,
            schema,
            properties,
        }
    }

    /// Fold one input batch into `accs` (parallel to `self.aggs`).
    fn fold_batch(accs: &mut [Acc], aggs: &[AggSpec], batch: &RecordBatch) -> Result<()> {
        for (i, spec) in aggs.iter().enumerate() {
            match (spec.kind, spec.column.as_deref()) {
                // COUNT(*) counts every row.
                (AggKind::Count, None) => accs[i].count += batch.num_rows() as u128,
                (kind, Some(name)) => {
                    let array = batch.column_by_name(name).ok_or_else(|| {
                        DataFusionError::Internal(format!("aggregate column {name} not in scan"))
                    })?;
                    let f64arr = arrow::compute::cast(array, &DataType::Float64)?;
                    let vals = f64arr
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .expect("cast to Float64");
                    match kind {
                        // COUNT(col) counts non-null values only.
                        AggKind::Count => accs[i].count += (vals.len() - vals.null_count()) as u128,
                        _ => {
                            for j in 0..vals.len() {
                                if vals.is_valid(j) {
                                    accs[i].push(vals.value(j));
                                }
                            }
                        }
                    }
                }
                // Non-count aggregate with no column — recognizer never emits this.
                (_, None) => {
                    return Err(DataFusionError::Internal(
                        "non-count aggregate without a column".into(),
                    ))
                }
            }
        }
        Ok(())
    }

    /// Build the single output row from finalized accumulators.
    fn finalize(schema: &SchemaRef, aggs: &[AggSpec], accs: &[Acc]) -> Result<RecordBatch> {
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(aggs.len());
        for (i, spec) in aggs.iter().enumerate() {
            let acc = &accs[i];
            let natural: ArrayRef = match spec.kind {
                AggKind::Count => Arc::new(Int64Array::from(vec![acc.count as i64])),
                AggKind::Sum => Arc::new(Float64Array::from(vec![(acc.count > 0).then_some(acc.sum)])),
                AggKind::Avg => Arc::new(Float64Array::from(vec![
                    (acc.count > 0).then(|| acc.sum / acc.count as f64),
                ])),
                AggKind::Min => Arc::new(Float64Array::from(vec![acc.min])),
                AggKind::Max => Arc::new(Float64Array::from(vec![acc.max])),
            };
            // Match DataFusion's declared output type (e.g. SUM(int64) -> Int64).
            let field_ty = schema.field(i).data_type();
            let col = if natural.data_type() == field_ty {
                natural
            } else {
                arrow::compute::cast(&natural, field_ty)?
            };
            columns.push(col);
        }
        RecordBatch::try_new(schema.clone(), columns)
            .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}

impl std::fmt::Debug for ZarrAggregateExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ZarrAggregateExec(aggs={:?})", self.aggs)
    }
}

impl DisplayAs for ZarrAggregateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let names: Vec<String> = self
            .aggs
            .iter()
            .map(|a| format!("{:?}({})", a.kind, a.column.as_deref().unwrap_or("*")))
            .collect();
        write!(f, "ZarrAggregateExec: {}", names.join(", "))
    }
}

impl ExecutionPlan for ZarrAggregateExec {
    fn name(&self) -> &str {
        "ZarrAggregateExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(ZarrAggregateExec::new(
            children[0].clone(),
            self.aggs.clone(),
            self.schema.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "ZarrAggregateExec has one output partition, got {partition}"
            )));
        }
        let input = self.input.clone();
        let aggs = self.aggs.clone();
        let schema = self.schema.clone();
        let n_in = input.properties().output_partitioning().partition_count();

        let fut = async move {
            let mut accs = vec![Acc::default(); aggs.len()];
            // Fold every input partition into the single group.
            for p in 0..n_in {
                let mut stream = input.execute(p, context.clone())?;
                while let Some(batch) = stream.try_next().await? {
                    Self::fold_batch(&mut accs, &aggs, &batch)?;
                }
            }
            Self::finalize(&schema, &aggs, &accs)
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            stream,
        )))
    }
}
