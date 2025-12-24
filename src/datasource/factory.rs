use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::{Session, TableProviderFactory};
use datafusion::datasource::TableProvider;
use datafusion::error::Result;
use datafusion::logical_expr::CreateExternalTable;

use crate::datasource::zarr::ZarrTable;
use crate::reader::schema_inference::infer_schema;

#[derive(Debug, Default)]
pub struct ZarrTableFactory;

#[async_trait]
impl TableProviderFactory for ZarrTableFactory {
    async fn create(
        &self,
        _state: &dyn Session,
        cmd: &CreateExternalTable,
    ) -> Result<Arc<dyn TableProvider>> {
        let schema = Arc::new(infer_schema(&cmd.location)?);
        Ok(Arc::new(ZarrTable::new(schema, &cmd.location)))
    }
}
