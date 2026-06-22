use crate::physical_plan::partition::PartitionSpec;
use crate::reader::filter::{CoordFilters, CoordSelection};
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
    properties: Arc<PlanProperties>,
    io_stats: SharedIoStats,
    /// Cached remote store and metadata (avoids recreating on each query)
    cached_remote: CachedRemoteStore,
    /// Coordinate filters for filter pushdown (e.g., time = X, hybrid = Y)
    coord_filters: Option<CoordFilters>,
    /// Cached VirtualiZarr adapter for remote VirtualiZarr stores
    cached_virtualizarr: CachedVirtualiZarrAdapter,
    /// Outer-dimension slices, one per output partition.
    /// EMPTY means "single unpartitioned read" (legacy behavior) — that keeps
    /// every existing `ZarrExec::new` caller (codec, tests) working unchanged.
    /// Populated by `with_partitions` from the planner in `ZarrTable::scan`.
    partitions: Vec<PartitionSpec>,
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
                    .map(|(k, v)| {
                        let joined: Vec<String> = v.iter().map(|f| f.to_string()).collect();
                        format!("{}{}", k, joined.join(" AND "))
                    })
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
        // Start with a single, unpartitioned read (empty `partitions`).
        let properties = Self::build_properties(&schema, projection.as_ref(), 1);
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
            partitions: Vec::new(),
        }
    }

    /// Build the `PlanProperties` for a given output partition count.
    ///
    /// Shared by `new` (count = 1) and `with_partitions` (count = number of
    /// slices). `&SchemaRef` / `Option<&Vec<usize>>` are *borrows* — this only
    /// reads them to derive the projected schema, it doesn't take ownership.
    fn build_properties(
        schema: &SchemaRef,
        projection: Option<&Vec<usize>>,
        num_partitions: usize,
    ) -> Arc<PlanProperties> {
        // Compute projected schema for plan properties
        let projected_schema = if let Some(indices) = projection {
            Arc::new(Schema::new(
                indices
                    .iter()
                    .map(|&i| schema.field(i).clone())
                    .collect::<Vec<_>>(),
            ))
        } else {
            schema.clone()
        };

        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(projected_schema),
            Partitioning::UnknownPartitioning(num_partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ))
    }

    /// Attach the planner's partition slices and re-derive `PlanProperties` so
    /// the plan advertises `partitions.len()` output partitions.
    ///
    /// Takes `self` by value (`mut self`) and returns it — the builder idiom:
    /// `ZarrExec::new(..).with_partitions(specs)`.
    pub fn with_partitions(mut self, partitions: Vec<PartitionSpec>) -> Self {
        // ▢ FILL 2: how many output partitions should we advertise?
        // An empty `partitions` must still mean ONE partition (legacy single
        // read), so floor the count at 1. Hint: `.len().max(1)`.
        let num_partitions = partitions.len().max(1);
        self.properties =
            Self::build_properties(&self.schema, self.projection.as_ref(), num_partitions);
        self.partitions = partitions;
        self
    }

    /// Get I/O statistics collected during execution
    pub fn io_stats(&self) -> SharedIoStats {
        self.io_stats.clone()
    }

    /// Get the current limit
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Get the full (unprojected) schema
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Get the store path
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Get the projection indices, if any
    pub fn projection(&self) -> Option<&Vec<usize>> {
        self.projection.as_ref()
    }

    /// Get the coordinate filters
    pub fn coord_filters(&self) -> Option<&CoordFilters> {
        self.coord_filters.as_ref()
    }

    /// Get this scan's partition slices (empty == single unpartitioned read).
    /// Used by the distributed codec (to ship them) and the `TaskEstimator`
    /// (to split them across worker tasks).
    pub fn partitions(&self) -> &[PartitionSpec] {
        &self.partitions
    }

    /// Create a new ZarrExec with the given limit
    pub fn with_limit(&self, limit: Option<usize>) -> Self {
        Self::new(
            self.schema.clone(),
            self.path.clone(),
            self.projection.clone(),
            limit,
            self.cached_remote.clone(),
            self.coord_filters.clone(),
            self.cached_virtualizarr.clone(),
        )
        // `new` resets partitions to empty; re-attach ours so a limited copy
        // keeps the same output partitioning. `.clone()` because `self` is
        // borrowed (`&self`) and we can't move the Vec out of it.
        .with_partitions(self.partitions.clone())
    }
}
impl ExecutionPlan for ZarrExec {
    fn name(&self) -> &str {
        "ZarrExec"
    }

    fn children(&self) -> Vec<&std::sync::Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn properties(&self) -> &Arc<datafusion::physical_plan::PlanProperties> {
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
        partition: usize,
        _context: std::sync::Arc<datafusion::execution::TaskContext>,
    ) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
        // Look up this partition's outer-axis selection. Empty `partitions` =>
        // legacy whole-store read (`None`). The selection is intersected with the
        // filter-derived ranges inside the reader.
        let partition_selection: Option<CoordSelection> = if self.partitions.is_empty() {
            None
        } else {
            Some(self.partitions[partition].outer.clone())
        };

        info!(
            path = %self.path,
            partition,
            partition_selection = ?partition_selection,
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
                partition_selection,
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
                partition_selection,
            )
        }
    }
}

// =============================================================================
// Helper functions to reduce duplication across execute_* functions
// =============================================================================

/// Build projected schema from full schema and projection indices.
///
/// This consolidates the identical schema projection logic used across all execute functions.
fn build_exec_projected_schema(schema: &SchemaRef, projection: Option<&Vec<usize>>) -> SchemaRef {
    if let Some(indices) = projection {
        Arc::new(Schema::new(
            indices
                .iter()
                .map(|&i| schema.field(i).clone())
                .collect::<Vec<_>>(),
        ))
    } else {
        schema.clone()
    }
}

/// Parameters for async read execution.
///
/// Encapsulates the common parameters passed to all async execute functions.
struct AsyncReadParams {
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: SharedIoStats,
    coord_filters: Option<CoordFilters>,
    /// Outer-axis selection for this partition; `None` => whole store.
    partition_selection: Option<CoordSelection>,
}

/// Execute an async read with the given store setup function.
///
/// This consolidates the common stream creation pattern used across all async execute functions:
/// 1. Create projected schema
/// 2. Create async stream that sets up store and calls read_zarr_async
/// 3. Collect batches and wrap in RecordBatchStreamAdapter
fn execute_async_read<F, Fut>(
    params: AsyncReadParams,
    store_setup: F,
    debug_name: &'static str,
) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<
            Output = datafusion::error::Result<(
                AsyncReadableListableStorage,
                ObjectPath,
                Option<ZarrStoreMeta>,
            )>,
        > + Send,
{
    use arrow::record_batch::RecordBatch;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream::{self, TryStreamExt};

    let projected_schema = build_exec_projected_schema(&params.schema, params.projection.as_ref());

    let stream = stream::once(async move {
        debug!("{} stream polled - starting async execution", debug_name);

        // Set up the store using the provided function
        let (store, prefix, cached_meta) = store_setup().await?;

        // Read the data
        debug!("Starting read_zarr_async for {}", debug_name);
        let result_stream = read_zarr_async(
            store,
            &prefix,
            params.schema,
            params.projection,
            params.limit,
            Some(params.stats),
            cached_meta,
            params.coord_filters,
            params.partition_selection,
        )
        .await?;

        // Collect into batches and return as stream
        debug!("Collecting batches from {}", debug_name);
        let batches: Vec<RecordBatch> = result_stream.try_collect().await?;
        info!(num_batches = batches.len(), "{} read complete", debug_name);

        Ok::<_, DataFusionError>(stream::iter(batches.into_iter().map(Ok)))
    })
    .try_flatten();

    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

/// Execute read from remote object store
#[allow(clippy::too_many_arguments)]
fn execute_remote(
    path: String,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,
    limit: Option<usize>,
    stats: SharedIoStats,
    cached_remote: CachedRemoteStore,
    coord_filters: Option<CoordFilters>,
    partition_selection: Option<CoordSelection>,
) -> datafusion::error::Result<datafusion::execution::SendableRecordBatchStream> {
    use crate::reader::storage::create_async_store;

    debug!(path = %path, has_cached_remote = cached_remote.is_some(), "Setting up remote execution stream");

    let params = AsyncReadParams {
        schema,
        projection,
        limit,
        stats,
        coord_filters,
        partition_selection,
    };

    execute_async_read(
        params,
        move || async move {
            if let Some((store, prefix, meta)) = cached_remote {
                info!("Using cached async store and metadata");
                Ok((store, prefix, Some(meta)))
            } else {
                debug!("Creating async store (no cache)");
                let (store, prefix) = create_async_store(&path)
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                debug!(prefix = %prefix, "Async store created");
                Ok((store, prefix, None))
            }
        },
        "Remote",
    )
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

    debug!(path = %path, "Setting up VirtualiZarr execution stream");

    // Pre-discover metadata from local .zmetadata (avoids async discovery issues)
    let cached_meta = discover_arrays(&path).map_err(DataFusionError::External)?;

    let params = AsyncReadParams {
        schema,
        projection,
        limit,
        stats,
        coord_filters,
        // VirtualiZarr stays single-partition for now (scan() doesn't partition it).
        partition_selection: None,
    };

    execute_async_read(
        params,
        move || async move {
            // Create VirtualStoreAdapter
            let adapter = VirtualStoreAdapter::new(&path).map_err(DataFusionError::External)?;
            let store: AsyncReadableListableStorage = Arc::new(adapter);
            let prefix = ObjectPath::from("");
            Ok((store, prefix, Some(cached_meta)))
        },
        "VirtualiZarr",
    )
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
    debug!("Setting up remote VirtualiZarr execution with cached adapter");

    let params = AsyncReadParams {
        schema,
        projection,
        limit,
        stats,
        coord_filters,
        partition_selection: None,
    };

    execute_async_read(
        params,
        move || async move {
            let store: AsyncReadableListableStorage = adapter;
            let prefix = ObjectPath::from("");
            Ok((store, prefix, None))
        },
        "Remote VirtualiZarr",
    )
}
