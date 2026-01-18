use crate::reader::filter::CoordFilters;
use crate::reader::schema_inference::ZarrStoreMeta;
use crate::reader::stats::{SharedIoStats, ZarrIoStats};
use crate::reader::storage::is_remote_url;
use crate::reader::virtual_store::{is_virtualizarr_store, VirtualStoreAdapter};
use crate::reader::zarr_reader::{read_zarr, read_zarr_async};
use arrow::datatypes::{Schema, SchemaRef};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::{
    common::DataFusionError,
    physical_expr::EquivalenceProperties,
    physical_plan::{DisplayAs, ExecutionPlan, Partitioning, PlanProperties},
};
use std::sync::Arc;
use tracing::{debug, info};
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

/// Cached remote store info (store, prefix, metadata)
pub type CachedRemoteStore = Option<(AsyncReadableListableStorage, ObjectPath, ZarrStoreMeta)>;

/// Cached VirtualiZarr adapter (pre-loaded refs and metadata)
pub type CachedVirtualiZarrAdapter = Option<Arc<VirtualStoreAdapter>>;

pub struct ZarrExec {
    schema: SchemaRef,
    path: String,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    properties: PlanProperties,
    io_stats: SharedIoStats,
    /// Cached remote store and metadata (avoids recreating on each query)
    cached_remote: CachedRemoteStore,
    /// Coordinate filters for filter pushdown (e.g., time = X, hybrid = Y)
    coord_filters: Option<CoordFilters>,
    /// Cached VirtualiZarr adapter for remote VirtualiZarr stores
    cached_virtualizarr: CachedVirtualiZarrAdapter,
}

impl std::fmt::Debug for ZarrExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrExec")
            .field("schema", &self.schema)
            .field("path", &self.path)
            .field("projection", &self.projection)
            .field("limit", &self.limit)
            .field("has_cached_remote", &self.cached_remote.is_some())
            .field(
                "has_cached_virtualizarr",
                &self.cached_virtualizarr.is_some(),
            )
            .field(
                "coord_filters",
                &self
                    .coord_filters
                    .as_ref()
                    .map(|f| f.filters.keys().collect::<Vec<_>>()),
            )
            .finish()
    }
}

impl DisplayAs for ZarrExec {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let mut parts = vec![format!("path={}", self.path)];
        if let Some(limit) = self.limit {
            parts.push(format!("limit={}", limit));
        }
        if let Some(ref filters) = self.coord_filters {
            if !filters.is_empty() {
                let filter_strs: Vec<_> = filters
                    .filters
                    .iter()
                    .map(|(k, v)| format!("{}{}", k, v))
                    .collect();
                parts.push(format!("filters=[{}]", filter_strs.join(", ")));
            }
        }
        write!(f, "ZarrExec: {}", parts.join(", "))
    }
}

impl ZarrExec {
    pub fn new(
        schema: SchemaRef,
        path: String,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
        cached_remote: CachedRemoteStore,
        coord_filters: Option<CoordFilters>,
        cached_virtualizarr: CachedVirtualiZarrAdapter,
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
            io_stats: Arc::new(ZarrIoStats::new()),
            cached_remote,
            coord_filters,
            cached_virtualizarr,
        }
    }

    /// Get I/O statistics collected during execution
    pub fn io_stats(&self) -> SharedIoStats {
        self.io_stats.clone()
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
        info!(
            path = %self.path,
            limit = ?self.limit,
            projection = ?self.projection,
            has_cached_remote = self.cached_remote.is_some(),
            has_cached_virtualizarr = self.cached_virtualizarr.is_some(),
            has_coord_filters = self.coord_filters.is_some(),
            "ZarrExec::execute called"
        );

        // Check for cached VirtualiZarr adapter first (remote VirtualiZarr)
        if let Some(adapter) = &self.cached_virtualizarr {
            info!("Using cached VirtualiZarr adapter (remote) execution path");
            return execute_virtualizarr_with_adapter(
                adapter.clone(),
                self.schema.clone(),
                self.projection.clone(),
                self.limit,
                self.io_stats.clone(),
                self.coord_filters.clone(),
            );
        }

        if is_virtualizarr_store(&self.path) {
            info!("Using VirtualiZarr (local) execution path");
            execute_virtualizarr(
                self.path.clone(),
                self.schema.clone(),
                self.projection.clone(),
                self.limit,
                self.io_stats.clone(),
                self.coord_filters.clone(),
            )
        } else if is_remote_url(&self.path) {
            info!("Using remote (async) execution path");
            execute_remote(
                self.path.clone(),
                self.schema.clone(),
                self.projection.clone(),
                self.limit,
                self.io_stats.clone(),
                self.cached_remote.clone(),
                self.coord_filters.clone(),
            )
        } else {
            info!("Using local (sync) execution path");
            read_zarr(
                &self.path,
                self.schema.clone(),
                self.projection.clone(),
                self.limit,
                Some(self.io_stats.clone()),
                self.coord_filters.clone(),
            )
        }
    }
}

/// Execute read from remote object store
fn execute_remote(
    path: String,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: SharedIoStats,
    cached_remote: CachedRemoteStore,
    coord_filters: Option<CoordFilters>,
) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
    use crate::reader::storage::create_async_store;
    use arrow::record_batch::RecordBatch;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream::{self, TryStreamExt};

    debug!(path = %path, has_cached_remote = cached_remote.is_some(), "Setting up remote execution stream");

    // Create a stream that will perform the async read when polled
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

    // Use try_flatten to handle the Result<Stream, Error> pattern
    let stream = stream::once(async move {
        debug!("Remote stream polled - starting async execution");

        // Use cached store and metadata if available
        let (store, prefix, cached_meta) = if let Some((store, prefix, meta)) = cached_remote {
            info!("Using cached async store and metadata");
            (store, prefix, Some(meta))
        } else {
            debug!("Creating async store (no cache)");
            let (store, prefix) = create_async_store(&path)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            debug!(prefix = %prefix, "Async store created");
            (store, prefix, None)
        };

        // Read the data
        debug!("Starting read_zarr_async");
        let result_stream = read_zarr_async(
            store,
            &prefix,
            schema,
            projection,
            limit,
            Some(stats),
            cached_meta,
            coord_filters,
        )
        .await?;

        // Collect into batches and return as stream
        debug!("Collecting batches");
        let batches: Vec<RecordBatch> = result_stream.try_collect().await?;
        info!(num_batches = batches.len(), "Remote read complete");

        Ok::<_, DataFusionError>(stream::iter(batches.into_iter().map(Ok)))
    })
    .try_flatten();

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

/// Execute read from VirtualiZarr Parquet reference store
fn execute_virtualizarr(
    path: String,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: SharedIoStats,
    coord_filters: Option<CoordFilters>,
) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
    use crate::reader::schema_inference::discover_arrays;
    use arrow::record_batch::RecordBatch;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream::{self, TryStreamExt};
    use zarrs::storage::AsyncReadableListableStorage;

    debug!(path = %path, "Setting up VirtualiZarr execution stream");

    // Pre-discover metadata from local .zmetadata (avoids async discovery issues)
    let cached_meta = discover_arrays(&path).map_err(DataFusionError::External)?;

    // Create projected schema
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

    // Create the stream that performs async read when polled
    let stream = stream::once(async move {
        debug!("VirtualiZarr stream polled - starting async execution");

        // Create VirtualStoreAdapter
        let adapter = VirtualStoreAdapter::new(&path).map_err(DataFusionError::External)?;

        // Wrap in Arc for zarrs
        let store: AsyncReadableListableStorage = Arc::new(adapter);

        // Empty prefix - VirtualiZarr stores use relative paths internally
        let prefix = ObjectPath::from("");

        // Read the data with cached metadata (avoids async discovery)
        debug!("Starting read_zarr_async for VirtualiZarr");
        let result_stream = read_zarr_async(
            store,
            &prefix,
            schema,
            projection,
            limit,
            Some(stats),
            Some(cached_meta), // Use pre-loaded metadata
            coord_filters,
        )
        .await?;

        // Collect into batches and return as stream
        debug!("Collecting batches from VirtualiZarr");
        let batches: Vec<RecordBatch> = result_stream.try_collect().await?;
        info!(num_batches = batches.len(), "VirtualiZarr read complete");

        Ok::<_, DataFusionError>(stream::iter(batches.into_iter().map(Ok)))
    })
    .try_flatten();

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

/// Execute read from a cached VirtualiZarr adapter (for remote VirtualiZarr stores)
fn execute_virtualizarr_with_adapter(
    adapter: Arc<VirtualStoreAdapter>,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: SharedIoStats,
    coord_filters: Option<CoordFilters>,
) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
    use arrow::record_batch::RecordBatch;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream::{self, TryStreamExt};

    debug!("Setting up remote VirtualiZarr execution with cached adapter");

    // Create projected schema
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

    // Create the stream that performs async read when polled
    let stream = stream::once(async move {
        debug!("Remote VirtualiZarr stream polled - starting async execution");

        // Use the pre-loaded adapter directly
        let store: AsyncReadableListableStorage = adapter;

        // Empty prefix - VirtualiZarr stores use relative paths internally
        let prefix = ObjectPath::from("");

        // Read the data (metadata will be discovered from the adapter)
        debug!("Starting read_zarr_async for remote VirtualiZarr");
        let result_stream = read_zarr_async(
            store,
            &prefix,
            schema,
            projection,
            limit,
            Some(stats),
            None, // Let it discover metadata from the adapter
            coord_filters,
        )
        .await?;

        // Collect into batches and return as stream
        debug!("Collecting batches from remote VirtualiZarr");
        let batches: Vec<RecordBatch> = result_stream.try_collect().await?;
        info!(
            num_batches = batches.len(),
            "Remote VirtualiZarr read complete"
        );

        Ok::<_, DataFusionError>(stream::iter(batches.into_iter().map(Ok)))
    })
    .try_flatten();

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}
