use crate::reader::zarr_reader::read_zarr;
use arrow::datatypes::{Schema, SchemaRef};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::{
    physical_expr::EquivalenceProperties,
    physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties},
};
use std::sync::Arc;

#[derive(Debug)]
pub struct ZarrExec {
    schema: SchemaRef,
    path: String,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    properties: PlanProperties,
}

impl DisplayAs for ZarrExec {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match self.limit {
            Some(limit) => write!(f, "ZarrExec: path={}, limit={}", self.path, limit),
            None => write!(f, "ZarrExec: path={}", self.path),
        }
    }
}

impl ZarrExec {
    pub fn new(
        schema: SchemaRef,
        path: String,
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
            path,
            projection,
            limit,
            properties,
        }
    }
}
impl ExecutionPlan for ZarrExec {
    fn name(&self) -> &str {
        "ZarrExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn children(&self) -> Vec<&std::sync::Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        &self.properties
    }

    fn with_new_children(
        self: std::sync::Arc<Self>,
        _children: Vec<std::sync::Arc<dyn ExecutionPlan>>,
    ) -> datafusion::error::Result<std::sync::Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: std::sync::Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
        read_zarr(
            &self.path,
            self.schema.clone(),
            self.projection.clone(),
            self.limit,
        )
    }
}
