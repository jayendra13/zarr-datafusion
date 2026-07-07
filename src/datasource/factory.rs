use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::{Session, TableProviderFactory};
use datafusion::common::DataFusionError;
use datafusion::datasource::TableProvider;
use datafusion::error::Result;
use datafusion::logical_expr::CreateExternalTable;
use tracing::{debug, info, instrument};

use crate::datasource::zarr::ZarrTable;
use crate::reader::schema_inference::{
    infer_schema_from_zmetadata_json, infer_schema_with_meta, infer_schema_with_meta_async,
};
use crate::reader::storage::{create_async_store, is_remote_url};
use crate::reader::virtual_store::{is_virtualizarr_store_async, VirtualStoreAdapter};
#[cfg(feature = "icechunk")]
use zarrs_object_store::object_store::path::Path as ObjectPath;

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

        // Icechunk stores are versioned Zarr repos with their own on-disk layout.
        // Detect them up front and open via the icechunk repository API, then
        // reuse the async schema-inference + read path over the resulting store
        // (parked in ZarrTable's cached_remote slot, keyed at the root prefix).
        #[cfg(feature = "icechunk")]
        if crate::reader::icechunk_store::is_icechunk_store(&cmd.location) {
            info!("Icechunk store detected");
            let store = crate::reader::icechunk_store::open_icechunk_async(&cmd.location)
                .await
                .map_err(DataFusionError::External)?;
            // The synthetic store is flat at the repo root, so arrays live at
            // `<name>/zarr.json` with no prefix.
            let prefix = ObjectPath::default();
            let (schema, metadata) = infer_schema_with_meta_async(&store, &prefix)
                .await
                .map_err(DataFusionError::External)?;
            let schema = Arc::new(schema);

            if schema.fields().is_empty() {
                return Err(DataFusionError::Plan(format!(
                    "No arrays found in icechunk store at '{}'. \
                     Check that the path is a valid icechunk repository.",
                    cmd.location
                )));
            }

            info!(
                num_fields = schema.fields().len(),
                "Icechunk table created successfully"
            );
            return Ok(Arc::new(ZarrTable::with_cached_remote(
                schema,
                &cmd.location,
                store,
                prefix,
                metadata,
            )));
        }

        if is_remote_url(&cmd.location) {
            info!("Remote URL detected - using async schema inference");
            let (store, prefix) = create_async_store(&cmd.location)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            // Check if this is a remote VirtualiZarr store
            if is_virtualizarr_store_async(&store, &prefix).await {
                info!("Remote VirtualiZarr store detected");

                // Create the VirtualStoreAdapter asynchronously
                let adapter = VirtualStoreAdapter::new_async(&store, &prefix, &cmd.location)
                    .await
                    .map_err(DataFusionError::External)?;

                // Infer schema from the pre-loaded metadata (no additional I/O needed)
                debug!("Inferring schema from VirtualiZarr adapter metadata");
                let (schema, metadata) = infer_schema_from_zmetadata_json(adapter.raw_metadata())
                    .map_err(DataFusionError::External)?;
                let schema = Arc::new(schema);
                let adapter = Arc::new(adapter);

                // Validate that we found at least one array
                if schema.fields().is_empty() {
                    return Err(DataFusionError::Plan(format!(
                        "No arrays found in VirtualiZarr store at '{}'. \
                         Check that the URL is correct and the store is accessible.",
                        cmd.location
                    )));
                }

                info!(
                    num_fields = schema.fields().len(),
                    "Remote VirtualiZarr table created successfully"
                );
                return Ok(Arc::new(ZarrTable::with_remote_virtualizarr(
                    schema,
                    &cmd.location,
                    adapter,
                    metadata,
                )));
            }

            debug!("Store created, inferring schema");
            let (schema, metadata) = infer_schema_with_meta_async(&store, &prefix)
                .await
                .map_err(DataFusionError::External)?;
            let schema = Arc::new(schema);

            // Validate that we found at least one array
            if schema.fields().is_empty() {
                return Err(DataFusionError::Plan(format!(
                    "No arrays found in Zarr store at '{}'. \
                     Check that the URL is correct and the store is accessible.",
                    cmd.location
                )));
            }

            info!(
                num_fields = schema.fields().len(),
                "Table created successfully (with cached store and metadata)"
            );
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

            // Validate that we found at least one array
            if schema.fields().is_empty() {
                return Err(DataFusionError::Plan(format!(
                    "No arrays found in Zarr store at '{}'. \
                     Check that the path is correct and contains valid Zarr data.",
                    cmd.location
                )));
            }

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
