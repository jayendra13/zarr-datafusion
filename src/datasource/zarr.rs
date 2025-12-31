use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::stats::{ColumnStatistics, Precision, Statistics};
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
    /// Store metadata for statistics (used for count optimization)
    store_meta: Option<ZarrStoreMeta>,
}

impl std::fmt::Debug for ZarrTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrTable")
            .field("schema", &self.schema)
            .field("path", &self.path)
            .field("cached_remote", &self.cached_remote.as_ref().map(|(_, p, _)| p))
            .field("total_rows", &self.store_meta.as_ref().map(|m| m.total_rows))
            .finish()
    }
}

impl ZarrTable {
    pub fn new(schema: SchemaRef, path: impl Into<String>) -> Self {
        Self {
            schema,
            path: path.into(),
            cached_remote: None,
            store_meta: None,
        }
    }

    /// Create a ZarrTable with store metadata (for local paths)
    pub fn with_metadata(
        schema: SchemaRef,
        path: impl Into<String>,
        metadata: ZarrStoreMeta,
    ) -> Self {
        Self {
            schema,
            path: path.into(),
            cached_remote: None,
            store_meta: Some(metadata),
        }
    }

    /// Create a ZarrTable with a cached async store and metadata (for remote URLs)
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
            cached_remote: Some((store, prefix, metadata.clone())),
            store_meta: Some(metadata),
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

    /// Return statistics for this table
    ///
    /// This enables DataFusion's optimizer to convert count(*) and count(column)
    /// queries into constant values without scanning the data.
    ///
    /// For coordinate columns, we also provide:
    /// - min_value/max_value: Enables MIN(coord)/MAX(coord) optimization
    /// - distinct_count: Number of unique coordinate values
    fn statistics(&self) -> Option<Statistics> {
        let meta = self.store_meta.as_ref()?;

        // Build column statistics
        let column_statistics: Vec<ColumnStatistics> = self
            .schema
            .fields()
            .iter()
            .map(|field| {
                let field_name = field.name();

                // Check if this is a coordinate column with min/max
                if let Some(coord) = meta.coords.iter().find(|c| &c.name == field_name) {
                    if let Some((min, max)) = coord.coord_min_max {
                        // Coordinates have distinct_count = shape[0] (number of unique values)
                        let distinct_count = coord.shape[0] as usize;

                        // Convert min/max to ScalarValue based on the underlying type
                        // Dictionary types have a value type inside
                        let (min_value, max_value) = match field.data_type() {
                            arrow::datatypes::DataType::Dictionary(_, value_type) => {
                                scalar_values_from_f64(min, max, value_type.as_ref())
                            }
                            dt => scalar_values_from_f64(min, max, dt),
                        };

                        info!(
                            coord = %field_name,
                            min = %min_value,
                            max = %max_value,
                            distinct = distinct_count,
                            "Coordinate statistics"
                        );

                        return ColumnStatistics {
                            null_count: Precision::Exact(0),
                            min_value: Precision::Exact(min_value),
                            max_value: Precision::Exact(max_value),
                            distinct_count: Precision::Exact(distinct_count),
                            ..Default::default()
                        };
                    }
                }

                // Default: only null_count for data variables
                ColumnStatistics {
                    null_count: Precision::Exact(0),
                    ..Default::default()
                }
            })
            .collect();

        info!(
            total_rows = meta.total_rows,
            num_columns = column_statistics.len(),
            "Providing statistics for query optimization"
        );

        Some(Statistics {
            num_rows: Precision::Exact(meta.total_rows),
            total_byte_size: Precision::Absent,
            column_statistics,
        })
    }
}

/// Convert f64 min/max values to appropriate ScalarValue based on Arrow data type
fn scalar_values_from_f64(
    min: f64,
    max: f64,
    data_type: &arrow::datatypes::DataType,
) -> (datafusion::common::ScalarValue, datafusion::common::ScalarValue) {
    use arrow::datatypes::DataType;
    use datafusion::common::ScalarValue;

    match data_type {
        DataType::Float64 => (ScalarValue::Float64(Some(min)), ScalarValue::Float64(Some(max))),
        DataType::Float32 => (
            ScalarValue::Float32(Some(min as f32)),
            ScalarValue::Float32(Some(max as f32)),
        ),
        DataType::Int64 => (
            ScalarValue::Int64(Some(min as i64)),
            ScalarValue::Int64(Some(max as i64)),
        ),
        DataType::Int32 => (
            ScalarValue::Int32(Some(min as i32)),
            ScalarValue::Int32(Some(max as i32)),
        ),
        DataType::Int16 => (
            ScalarValue::Int16(Some(min as i16)),
            ScalarValue::Int16(Some(max as i16)),
        ),
        DataType::UInt64 => (
            ScalarValue::UInt64(Some(min as u64)),
            ScalarValue::UInt64(Some(max as u64)),
        ),
        DataType::UInt32 => (
            ScalarValue::UInt32(Some(min as u32)),
            ScalarValue::UInt32(Some(max as u32)),
        ),
        // Fallback to Float64
        _ => (ScalarValue::Float64(Some(min)), ScalarValue::Float64(Some(max))),
    }
}
