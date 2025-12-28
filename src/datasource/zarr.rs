use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::{datasource::TableProvider, error::Result, physical_plan::ExecutionPlan};
use std::sync::Arc;

use crate::physical_plan::zarr_exec::ZarrExec;

#[derive(Debug)]
pub struct ZarrTable {
    schema: SchemaRef,
    path: String,
}

impl ZarrTable {
    pub fn new(schema: SchemaRef, path: impl Into<String>) -> Self {
        Self {
            schema,
            path: path.into(),
        }
    }
}

#[async_trait]
impl TableProvider for ZarrTable {
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
        _filters: &[datafusion::logical_expr::Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(ZarrExec::new(
            self.schema.clone(),
            self.path.clone(),
            projection.cloned(),
            limit,
        )))
    }
}
