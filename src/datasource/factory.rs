use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::{Session, TableProviderFactory};
use datafusion::common::DataFusionError;
use datafusion::datasource::TableProvider;
use datafusion::error::Result;
use datafusion::logical_expr::CreateExternalTable;
use tracing::{debug, info, instrument};

use crate::datasource::zarr::ZarrTable;
use crate::reader::schema_inference::{infer_schema_with_meta, infer_schema_with_meta_async};
use crate::reader::storage::{create_async_store, is_remote_url};

#[derive(Debug, Default)]
pub struct ZarrTableFactory;

#[async_trait]
impl TableProviderFactory for ZarrTableFactory {
    #[instrument(level = "info", skip_all)]
    async fn create(
        &self,
        _state: &dyn Session,
        cmd: &CreateExternalTable,
    ) -> Result<Arc<dyn TableProvider>> {
        info!("Creating Zarr table");

        if is_remote_url(&cmd.location) {
            info!("Remote URL detected - using async schema inference");
            let (store, prefix) = create_async_store(&cmd.location)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            debug!("Store created, inferring schema");
            let (schema, metadata) = infer_schema_with_meta_async(&store, &prefix)
                .await
                .map_err(|e| DataFusionError::External(e))?;
            let schema = Arc::new(schema);
            info!(num_fields = schema.fields().len(), "Table created successfully (with cached store and metadata)");
            Ok(Arc::new(ZarrTable::with_cached_remote(
                schema,
                &cmd.location,
                store,
                prefix,
                metadata,
            )))
        } else {
            info!("Local path detected - using sync schema inference");
            let (schema, metadata) = infer_schema_with_meta(&cmd.location)?;
            let schema = Arc::new(schema);
            info!(
                num_fields = schema.fields().len(),
                total_rows = metadata.total_rows,
                "Table created successfully (with metadata for statistics)"
            );
            Ok(Arc::new(ZarrTable::with_metadata(
                schema,
                &cmd.location,
                metadata,
            )))
        }
    }
}
