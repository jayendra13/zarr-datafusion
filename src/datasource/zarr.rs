use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::stats::{ColumnStatistics, Precision, Statistics};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::{datasource::TableProvider, error::Result, physical_plan::ExecutionPlan};
use std::sync::Arc;
use tracing::{debug, info};
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

use crate::physical_plan::partition::{plan_partitions, split_selection, PartitionSpec};
use crate::physical_plan::zarr_exec::ZarrExec;
use crate::reader::filter::{is_date_part_filter, parse_coord_filters, CoordFilters};
use crate::reader::schema_inference::ZarrStoreMeta;
use crate::reader::storage::is_remote_url;
use crate::reader::virtual_store::{is_virtualizarr_store, VirtualStoreAdapter};
use crate::reader::zarr_reader::{
    resolve_outer_selection, resolve_outer_selection_async, OuterSelection,
};

/// Cached remote store info (store, prefix, metadata)
pub type CachedRemoteStore = Option<(AsyncReadableListableStorage, ObjectPath, ZarrStoreMeta)>;

/// Cached VirtualiZarr adapter (pre-loaded refs and metadata)
pub type CachedVirtualiZarrAdapter = Option<Arc<VirtualStoreAdapter>>;

pub struct ZarrTable {
    schema: SchemaRef,
    path: String,
    /// Cached async store and metadata for remote URLs (avoids recreating on each query)
    cached_remote: CachedRemoteStore,
    /// Store metadata for statistics (used for count optimization)
    store_meta: Option<ZarrStoreMeta>,
    /// Cached VirtualiZarr adapter for remote VirtualiZarr stores
    cached_virtualizarr: CachedVirtualiZarrAdapter,
    /// True when the cached store is an icechunk store (local or remote). Icechunk
    /// reads go through the async path (via `cached_remote`) but the scan stays
    /// single-partition — the outer-selection resolver can't drive the cached
    /// async store yet, and for a local icechunk path it would wrongly reopen the
    /// path as a plain Zarr store.
    icechunk: bool,
}

impl std::fmt::Debug for ZarrTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrTable")
            .field("schema", &self.schema)
            .field("path", &self.path)
            .field(
                "cached_remote",
                &self.cached_remote.as_ref().map(|(_, p, _)| p),
            )
            .field(
                "total_rows",
                &self.store_meta.as_ref().map(|m| m.total_rows),
            )
            .field(
                "has_virtualizarr_adapter",
                &self.cached_virtualizarr.is_some(),
            )
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
            cached_virtualizarr: None,
            icechunk: false,
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
            cached_virtualizarr: None,
            icechunk: false,
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
            cached_virtualizarr: None,
            icechunk: false,
        }
    }

    /// Create a ZarrTable for a remote VirtualiZarr store with cached adapter
    pub fn with_remote_virtualizarr(
        schema: SchemaRef,
        path: impl Into<String>,
        adapter: Arc<VirtualStoreAdapter>,
        metadata: ZarrStoreMeta,
    ) -> Self {
        Self {
            schema,
            path: path.into(),
            cached_remote: None,
            store_meta: Some(metadata),
            cached_virtualizarr: Some(adapter),
            icechunk: false,
        }
    }

    /// Create a ZarrTable backed by an icechunk store (local or remote).
    ///
    /// The icechunk store is parked in the `cached_remote` slot (keyed at
    /// `prefix`, which is the group path within the repo, or root) so schema
    /// inference and the async read path treat it like any other remote store.
    /// The `icechunk` marker keeps the scan single-partition.
    pub fn with_cached_icechunk(
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
            cached_virtualizarr: None,
            icechunk: true,
        }
    }

    /// Get the store metadata (for dimension info display)
    pub fn store_meta(&self) -> Option<&ZarrStoreMeta> {
        self.store_meta.as_ref()
    }

    /// Get the store path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Decide this scan's output partitions.
    ///
    /// When an outer-axis filter is present, this resolves it to the *surviving*
    /// index set (reading only the outer coordinate, on the head, at plan time)
    /// and splits THAT across partitions — so a narrow filter still fans out
    /// across the cluster. With no outer filter it falls back to splitting the
    /// full axis geometrically (no coordinate read).
    ///
    /// Returns an EMPTY vec to mean "single unpartitioned read" — used for
    /// VirtualiZarr stores (their read path doesn't take a range yet) and when we
    /// lack the metadata to slice safely.
    async fn plan_scan_partitions(
        &self,
        state: &dyn Session,
        limit: Option<usize>,
        coord_filters: Option<&CoordFilters>,
    ) -> Vec<PartitionSpec> {
        // Guard 0: a LIMIT is a GLOBAL row cap, but it gets pushed into each
        // partition's read independently — which is unsound across multiple
        // partitions (each would return its own first-N rows). Until per-
        // partition fetch is handled by a global limit above the scan, keep
        // limited queries single-partition (where pushdown stays correct).
        if limit.is_some() {
            return Vec::new();
        }

        // Guard 1: local and remote stores partition; VirtualiZarr does not yet
        // (its async read path ignores the partition selection).
        if is_virtualizarr_store(&self.path) {
            return Vec::new();
        }

        // Guard 1b: icechunk stores (local or remote) stay single-partition for
        // now — the outer-selection resolver can't drive the cached async store
        // yet, and for a local icechunk path the sync branch below would wrongly
        // reopen `self.path` as a plain Zarr store.
        if self.icechunk {
            return Vec::new();
        }

        // Guard 2: we need metadata and a data variable to size the outer axis.
        let Some(meta) = &self.store_meta else {
            return Vec::new();
        };
        let Some(data_var) = meta.data_vars.first() else {
            return Vec::new();
        };
        // The reader maps the partition selection to the coordinate whose size
        // equals the data var's outer-axis length (`shape[0]`). If no such
        // coordinate exists we can't slice safely, so don't partition.
        let outer_len: u64 = data_var.shape.first().copied().unwrap_or(0);
        let outer_coord_exists = meta
            .coords
            .iter()
            .any(|c| c.shape.first().copied() == Some(outer_len));
        if !outer_coord_exists {
            return Vec::new();
        }

        let target_partitions: usize = state.config().target_partitions();
        // Unknown chunking => treat the whole axis as one chunk (=> one partition).
        let chunk_len: u64 = data_var
            .chunks
            .as_ref()
            .and_then(|c| c.first().copied())
            .unwrap_or(outer_len);

        // Resolve the outer filter on the head (reads only the outer coord, and
        // only when an outer filter exists). Remote stores resolve via the cached
        // async store the head set up; workers never reach here.
        let outer = if is_remote_url(&self.path) {
            match &self.cached_remote {
                Some((store, prefix, _)) => {
                    resolve_outer_selection_async(store.clone(), prefix, meta, coord_filters).await
                }
                // No cached async store on the head (unexpected): fall back to
                // geometry partitioning rather than building a store here.
                None => Ok(OuterSelection::Unfiltered),
            }
        } else {
            resolve_outer_selection(&self.path, meta, coord_filters)
        };

        match outer {
            // Partition the SURVIVING set so a narrow filter still fans out.
            Ok(OuterSelection::Resolved(sel)) => {
                split_selection(&sel, chunk_len, target_partitions)
                    .into_iter()
                    .map(|outer| PartitionSpec { outer })
                    .collect()
            }
            // A present outer filter matched nothing: one empty partition.
            Ok(OuterSelection::Empty) => vec![PartitionSpec::range(0, 0)],
            // No outer filter: split the full axis geometrically (no coord read).
            Ok(OuterSelection::Unfiltered) => {
                plan_partitions(outer_len, chunk_len, target_partitions)
            }
            // A resolution read error shouldn't abort the query — fall back to
            // geometry partitioning (still correct, just not load-balanced).
            Err(e) => {
                debug!(error = %e, "outer-selection resolution failed; using geometry partitioning");
                plan_partitions(outer_len, chunk_len, target_partitions)
            }
        }
    }
}

#[async_trait]
impl TableProvider for ZarrTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> datafusion::datasource::TableType {
        datafusion::datasource::TableType::Base
    }

    /// Indicate which filters can be pushed down to the scan
    ///
    /// Returns `Inexact` for all filters - we'll handle coordinate equality
    /// filters during scan, but DataFusion should still apply filters post-scan
    /// for correctness (in case we miss any).
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|f| {
                // DatePart filters are handled exactly by ZarrExec (via index gather),
                // so tell DataFusion to drop the FilterExec — otherwise it re-evaluates
                // date_part() on Dictionary-encoded columns and hits a type error.
                if is_date_part_filter(f) {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Inexact
                }
            })
            .collect())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[datafusion::logical_expr::Expr],
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
            info!(
                projected = total_columns,
                total = total_columns,
                "No projection pushdown (all columns)"
            );
        }

        // Log limit pushdown
        if let Some(limit) = limit {
            info!(limit, "Limit pushdown");
        }

        // Parse coordinate filters for filter pushdown
        debug!(
            num_filters = filters.len(),
            filters = ?filters,
            "Filters passed to scan()"
        );
        let coord_filters = if let Some(meta) = &self.store_meta {
            let coord_names: Vec<String> = meta.coords.iter().map(|c| c.name.clone()).collect();
            debug!(?coord_names, "Coordinate names from metadata");
            let parsed = parse_coord_filters(filters, &coord_names);
            if !parsed.is_empty() {
                info!(
                    num_filters = parsed.len(),
                    coords = ?parsed.filters.keys().collect::<Vec<_>>(),
                    "Filter pushdown"
                );
                Some(parsed)
            } else {
                None
            }
        } else {
            // No metadata available - can't do filter pushdown
            None
        };

        // ── Plan output partitions ────────────────────────────────────────
        // Resolve any outer-axis filter to the surviving set and split THAT
        // across partitions (local + remote); VirtualiZarr stays single-partition.
        let partitions = self
            .plan_scan_partitions(state, limit, coord_filters.as_ref())
            .await;

        let exec = ZarrExec::new(
            self.schema.clone(),
            self.path.clone(),
            projection.cloned(),
            limit,
            self.cached_remote.clone(),
            coord_filters,
            self.cached_virtualizarr.clone(),
        );

        // FILL 6d: attach the planned slices (builder consumes & returns exec).
        let exec = exec.with_partitions(partitions);

        Ok(Arc::new(exec))
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
) -> (
    datafusion::common::ScalarValue,
    datafusion::common::ScalarValue,
) {
    use arrow::datatypes::DataType;
    use datafusion::common::ScalarValue;

    match data_type {
        DataType::Float64 => (
            ScalarValue::Float64(Some(min)),
            ScalarValue::Float64(Some(max)),
        ),
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
        _ => (
            ScalarValue::Float64(Some(min)),
            ScalarValue::Float64(Some(max)),
        ),
    }
}
