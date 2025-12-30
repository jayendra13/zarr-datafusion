use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::{datasource::TableProvider, error::Result, physical_plan::ExecutionPlan};
use std::sync::Arc;
use tracing::info;
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

use crate::physical_plan::zarr_exec::ZarrExec;
use crate::reader::schema_inference::ZarrStoreMeta;

/// Cached remote store info (store, prefix, metadata)
pub type CachedRemoteStore = Option<(AsyncReadableListableStorage, ObjectPath, ZarrStoreMeta)>;

pub struct ZarrTable {
    schema: SchemaRef,
    path: String,
    /// Cached async store and metadata for remote URLs (avoids recreating on each query)
    cached_remote: CachedRemoteStore,
}

impl std::fmt::Debug for ZarrTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrTable")
            .field("schema", &self.schema)
            .field("path", &self.path)
            .field("cached_remote", &self.cached_remote.as_ref().map(|(_, p, _)| p))
            .finish()
    }
}

impl ZarrTable {
    pub fn new(schema: SchemaRef, path: impl Into<String>) -> Self {
        Self {
            schema,
            path: path.into(),
            cached_remote: None,
        }
    }

    /// Create a ZarrTable with a cached async store and metadata
    pub fn with_cached_remote(
        schema: SchemaRef,
        path: impl Into<String>,
        store: AsyncReadableListableStorage,
        prefix: ObjectPath,
        metadata: ZarrStoreMeta,
    ) -> Self {
        Self {
            schema,
            path: path.into(),
            cached_remote: Some((store, prefix, metadata)),
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
        // Log projection pushdown
        let total_columns = self.schema.fields().len();
        if let Some(indices) = projection {
            let projected_names: Vec<_> = indices
                .iter()
                .map(|&i| self.schema.field(i).name().as_str())
                .collect();
            info!(
                projected = indices.len(),
                total = total_columns,
                columns = ?projected_names,
                "Projection pushdown"
            );
        } else {
            info!(projected = total_columns, total = total_columns, "No projection pushdown (all columns)");
        }

        // Log limit pushdown
        if let Some(limit) = limit {
            info!(limit, "Limit pushdown");
        }

        Ok(Arc::new(ZarrExec::new(
            self.schema.clone(),
            self.path.clone(),
            projection.cloned(),
            limit,
            self.cached_remote.clone(),
        )))
    }
}
