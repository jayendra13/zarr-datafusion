# Design Decisions — Code Map

Companion to [`design-decisions.md`](design-decisions.md). For each decision it
points at the concrete file, trait, type, or function that implements it and
shows the **key lines** of that logic — the rest of each body is elided with
`...`. Snippets are anchored on symbol names (not line numbers) so they stay
valid as the code moves; search for the symbol if a snippet has drifted.

## Orientation: where things live

| Layer | Module | Role |
| --- | --- | --- |
| Crate root | `src/lib.rs` | Module exports (`reader`, `datasource`, `optimizer`, `physical_plan`, `udfs`, `udtf`, `distributed`) |
| Table provider | `src/datasource/` | `ZarrTable` (`TableProvider`) + `ZarrTableFactory` for DDL |
| Physical plan | `src/physical_plan/` | `ZarrExec` (`ExecutionPlan`), partitioning, distributed codec |
| Reader | `src/reader/` | Schema inference, nD→2D flattening, filters, coords, storage, stats |
| Optimizer | `src/optimizer/` | Limit pushdown + MIN/MAX/COUNT statistics rules |
| UDFs / UDTF | `src/udfs/`, `src/udtf.rs` | `rmse`/`mae` etc., `zarr_describe()` |
| Distributed | `src/distributed.rs` | datafusion-distributed wiring (`distributed` feature) |
| CLI | `src/bin/zarr_cli/` | REPL; **wires the factory, optimizer rules, and UDTF into the session** |

**Data flow:** SQL → `ZarrTable::scan` → `ZarrExec::execute` → `read_zarr` /
`read_zarr_async` → `RecordBatch`. The reader entry point carries every
pushdown the planner resolved:

```rust
// src/reader/zarr_reader.rs — read_zarr (sync; read_zarr_async mirrors it)
pub fn read_zarr(
    store_path: &str,
    schema: SchemaRef,
    projection: Option<Vec<usize>>,        // projection pushdown
    limit: Option<usize>,                  // limit pushdown
    stats: Option<SharedIoStats>,          // I/O stats
    coord_filters: Option<CoordFilters>,   // filter pushdown
    partition_selection: Option<CoordSelection>, // this partition's outer slice
) -> Result<SendableRecordBatchStream> { ... }
```

---

## Foundational data model

### 1. Flatten nD → 2D (Cartesian product), as an Arrow `RecordBatch` stream
`src/reader/zarr_reader.rs` — `read_zarr` / `read_zarr_async` do the flattening
and the **Apache Arrow** conversion; `build_read_plans` turns a coordinate
selection into the chunk reads that materialize the rows (a row per grid cell).

```rust
// src/reader/zarr_reader.rs — build_read_plans
fn build_read_plans(
    selections: &[CoordSelection],   // one selection per coordinate
    coord_sizes: &[usize],
    data_var_shape: &[u64],
    data_var_chunks: Option<&[u64]>,
) -> Vec<ReadPlan> {
    ...
    // Map the coordinate selections onto the data variable's nD ArraySubset
    let array_ranges = match_ranges_to_data_var(coord_sizes, selections, data_var_shape) ...;
    return vec![ReadPlan { subset: ArraySubset::new_with_ranges(&array_ranges), keep: Keep::All }];
    ...
}
```

**Apache Arrow is the output contract.** Each flattened column is an Arrow
`ArrayRef` (coordinates as `DictionaryArray`, decision 3; data variables as
primitive arrays), assembled into a single `RecordBatch` — the columnar,
zero-copy unit DataFusion operates on. `RecordBatchOptions::with_row_count`
carries the row count even for an empty projection (e.g. `COUNT(*)`), where there
are no columns to infer it from:

```rust
// src/reader/zarr_reader.rs — create_result_batch (nD values -> Arrow columns -> one RecordBatch)
fn create_result_batch(projected_schema: SchemaRef, result_arrays: Vec<ArrayRef>,
                       final_rows: usize) -> Result<RecordBatch> {
    if result_arrays.is_empty() {                       // empty projection (COUNT(*))
        Ok(RecordBatch::try_new_with_options(projected_schema, result_arrays,
            &RecordBatchOptions::new().with_row_count(Some(final_rows)))?)
    } else {
        Ok(RecordBatch::try_new(projected_schema, result_arrays)?)
    }
}
```

**Done lazily — deferred, pull-based execution.** `read_zarr` returns a
`SendableRecordBatchStream`, not eager rows: `ZarrTable::scan` only *plans*, and
no bytes are read until DataFusion polls the stream returned by
`ZarrExec::execute`. Combined with projection / filter / limit pushdown (decisions
5–9) and partition slicing (decision 16), the cube is never fully materialized —
only the chunks a query needs are read, per partition.

```rust
// src/reader/zarr_reader.rs — wrap the batch in a pull-based stream (one per partition)
let batch = create_result_batch(projected_schema.clone(), result_arrays, final_rows)?;
let stream = stream::iter(vec![Ok(batch)]);
Ok(Box::pin(RecordBatchStreamAdapter::new(projected_schema, stream)))
```

> Caveat: each partition currently emits a **single** `RecordBatch` (one
> `stream::iter(vec![Ok(batch)])`). True multi-batch streaming — yielding batches
> incrementally instead of building one per partition — is a roadmap item
> ("Streaming RecordBatch output" in the README), so laziness today comes from
> deferred execution + pushdown + partitioning, not from chunk-by-chunk emission.

### 2. Structural convention (1D = coord, nD = data var)
`src/reader/schema_inference.rs` — role inference from array shape, and v2/v3
detection.

```rust
// src/reader/schema_inference.rs — ZarrArrayMeta::is_coordinate
impl ZarrArrayMeta {
    pub fn is_coordinate(&self) -> bool {
        self.shape.len() == 1          // any 1-D array is a coordinate
    }
    ...
}

// detect_zarr_version: zarr.json => V3, .zgroup/.zarray => V2
pub fn detect_zarr_version(store_path: &str) -> Result<ZarrVersion, ...> {
    if root.join("zarr.json").exists() { return Ok(ZarrVersion::V3); }
    if root.join(".zgroup").exists() || root.join(".zarray").exists() { return Ok(ZarrVersion::V2); }
    ...
}
```

### 3. Coordinates as `DictionaryArray`
`src/reader/coord.rs` — `create_coord_dictionary_typed` builds the
dictionary-encoded coordinate columns (keys = indices, values = unique coords).

### 4. Adaptive dictionary key width (Int16 → Int32 → Int64)
`src/reader/dtype.rs` — the key type is chosen from cardinality, fixing the
silent `as i16` wraparound that used to panic `DictionaryArray::new`.

```rust
// src/reader/dtype.rs — dictionary_key_type_for_cardinality
pub fn dictionary_key_type_for_cardinality(cardinality: usize) -> DataType {
    if cardinality <= (i16::MAX as usize) + 1 {        // <= 32,768
        DataType::Int16
    } else if cardinality <= (i32::MAX as usize) + 1 { // <= ~2.1 billion
        DataType::Int32
    } else {
        DataType::Int64
    }
}
```

The schema declares the key type from `coord.shape[0]`, and
`create_coord_dictionary_typed` builds keys at that same width (failing loudly if
undersized) so batches always match the declared type.

---

## Pushdown and optimization

### 5. Internalize coordinate filters into the scan
`src/datasource/zarr.rs::scan` parses WHERE-clause exprs into `CoordFilters` and
hands them to `ZarrExec`, instead of leaving them to a post-scan `FilterExec`.

```rust
// src/datasource/zarr.rs — TableProvider::scan
async fn scan(&self, state: &dyn Session, projection: Option<&Vec<usize>>,
              filters: &[Expr], limit: Option<usize>) -> Result<Arc<dyn ExecutionPlan>> {
    ...
    let coord_names: Vec<String> = meta.coords.iter().map(|c| c.name.clone()).collect();
    let parsed = parse_coord_filters(filters, &coord_names);   // WHERE -> CoordFilters
    ...
}
```

`ZarrExec` carries them and surfaces them via `coord_filters()`.

### 6. LIMIT pushdown past `FilterExec`
`src/optimizer/limit_pushdown.rs` — `ZarrLimitPushdownRule` (a
`PhysicalOptimizerRule`) finds a limit anywhere in the tree and pushes it into
the scan; sound only because the coordinate filters are internalized (decision 5).

```rust
// src/optimizer/limit_pushdown.rs
pub struct ZarrLimitPushdownRule;

impl PhysicalOptimizerRule for ZarrLimitPushdownRule {
    fn name(&self) -> &str { "zarr_limit_pushdown" }
    fn optimize(&self, plan: Arc<dyn ExecutionPlan>, _cfg: &ConfigOptions)
        -> Result<Arc<dyn ExecutionPlan>> {
        let limit = find_limit_anywhere(&plan);     // search the whole tree, not just root
        push_limit_to_zarr(plan, limit)             // -> ZarrExec::with_limit
    }
}
```

```rust
// src/physical_plan/zarr_exec.rs — how the rule injects the limit
pub fn with_limit(&self, limit: Option<usize>) -> Self {
    Self::new(self.schema.clone(), self.path.clone(), self.projection.clone(),
              limit, ...).with_partitions(self.partitions.clone())
}
```

### 7. MIN/MAX/COUNT from statistics
`src/optimizer/count_optimization.rs` and `minmax_optimization.rs` — logical
`OptimizerRule`s that constant-fold aggregates from metadata, skipping the scan.

```rust
// src/optimizer/count_optimization.rs
pub struct CountStatisticsRule;

impl OptimizerRule for CountStatisticsRule {
    fn name(&self) -> &str { "count_statistics" }
    fn rewrite(&self, plan: LogicalPlan, _cfg: &dyn OptimizerConfig)
        -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Aggregate(aggregate) = &plan else { return Ok(Transformed::no(plan)); };
        if !aggregate.group_expr.is_empty() { ... }   // only simple (no GROUP BY) aggregates
        ...                                            // replace COUNT(*) with a constant
    }
}
```

`MinMaxStatisticsRule` is the analogous rule for `MIN`/`MAX` on coordinates.

### 8. Vec-of-filters per coordinate (AND-composed)
`src/reader/filter.rs` — the core of filter pushdown. Each coordinate maps to a
**list** of `CoordFilterKind`, each resolved to a position set and intersected.

```rust
// src/reader/filter.rs — the filter vocabulary
pub enum CoordFilterKind {
    Eq(ScalarValue),
    Range { low: Option<ScalarValue>, high: Option<ScalarValue>,
            low_inclusive: bool, high_inclusive: bool },
    DatePart { field: String, value: i32 },            // EXTRACT(field) = v
    InList(Vec<ScalarValue>),                           // coord IN (...)
    DatePartSet { field: String, values: Vec<i32> },   // EXTRACT(field) IN (...)
}

pub struct CoordFilters {
    pub filters: HashMap<String, Vec<CoordFilterKind>>, // AND-composed per coord
}
```

```rust
// src/reader/filter.rs — resolve each filter to positions, then intersect
pub fn resolve_coord_selection(name: &str, filters: Option<&[CoordFilterKind]>,
                               values: &CoordValuesRef<'_>) -> Option<CoordSelection> {
    ...
    let mut acc: Option<CoordSelection> = None;
    for filter in filters {
        let sel = resolve_single_filter(name, filter, values)?;
        acc = Some(match acc { None => sel, Some(prev) => prev.intersect(&sel) });
        if acc.as_ref().is_some_and(CoordSelection::is_empty) { return None; } // empty => no rows
    }
    acc
}
```

```rust
// src/reader/filter.rs — Range ∩ Range stays a Range; anything with Indices collapses to Indices
pub fn intersect(&self, other: &CoordSelection) -> CoordSelection {
    match (self, other) {
        (Range(s1, e1), Range(s2, e2)) => Range((*s1).max(*s2), (*e1).min(*e2).max((*s1).max(*s2))),
        (Range(s, e), Indices(v)) | (Indices(v), Range(s, e)) =>
            Indices(v.iter().copied().filter(|&i| i >= *s && i < *e).collect()),
        (Indices(a), Indices(b)) => { /* set intersection */ ... }
    }
}
```

Note: `PartialBounds::into_filters` emits **every** collected predicate (the old
code dropped all but one via a fixed `eq > range > date_part` precedence). The
mirror `CoordFilterKind` in `src/physical_plan/codec.rs` round-trips these
through the distributed codec.

### 9. Coordinate-only queries skip Cartesian expansion (gated on LIMIT)
`src/reader/filter.rs::determine_effective_coords` — returns only the projected
coordinates' dimensions, but only when a `LIMIT` is present.

```rust
// src/reader/filter.rs — determine_effective_coords
} else if !projected_coord_names.is_empty() && limit.is_some() {
    // SELECT time ... LIMIT 10 -> use only the selected coords, no Cartesian product
    let mut indices: Vec<usize> = projected_coord_names.iter()
        .filter_map(|&name| coord_names.iter().position(|c| c == name)).collect();
    indices.sort(); indices.dedup();
    return Ok(indices);
} else {
    // COUNT(*) / coord-only without LIMIT -> all coords, to keep the row count correct
    return Ok((0..coord_sizes.len()).collect());
}
```

### 10. Compact coordinate encoding for arithmetic sequences
`src/reader/coord.rs::CompactCoord` — O(1) `(first, step, len)` representation;
`src/reader/filter.rs::filter_satisfiable_by_bounds` rejects impossible filters
before any read.

```rust
// src/reader/coord.rs — CompactCoord
pub enum CompactCoord {
    Arithmetic { first: f64, step: f64, len: usize },     // value[i] = first + i*step
    ArithmeticInt { first: i64, step: i64, len: usize },
}
impl CompactCoord {
    pub fn value_at_f64(&self, i: usize) -> f64 { ... first + (i as f64) * step ... }
}
```

```rust
// src/reader/filter.rs — filter_satisfiable_by_bounds (early rejection)
let (coord_min, coord_max) = match coord.coord_min_max { Some(b) => b, None => continue };
CoordFilterKind::Eq(value) => match scalar_to_f64(value) {
    Some(v) => v >= coord_min && v <= coord_max,   // latitude = 100 with bounds [0,90] => false
    None => true,
},
// ... if any kind is unsatisfiable, return false and skip the read.
```

---

## Domain correctness

### 11. CF-time decoding
`src/reader/cf_time.rs` — decode CF time units into microseconds since the Unix
epoch (Arrow timestamps), matching xarray.

```rust
// src/reader/cf_time.rs
pub struct CFTimeUnit { pub multiplier_us: i64, pub epoch_offset_us: i64, pub is_nanoseconds: bool }

pub fn decode_cf_time(values: &[i64], unit: &CFTimeUnit) -> Vec<i64> {
    ...
    values.iter().map(|v| unit.epoch_offset_us + v * unit.multiplier_us).collect()
}
// CFTimeAttrs::is_time_coordinate() = units.contains(" since ")  // e.g. "hours since 1900-01-01"
```

### 12. Mixed-dimensionality variables
`src/reader/schema_inference.rs` stores per-variable `dimensions` parsed from
`_ARRAY_DIMENSIONS`; `src/reader/filter.rs` uses them so a projection only pulls
the coordinates a variable actually has.

```rust
// src/reader/schema_inference.rs — ZarrArrayMeta
pub struct ZarrArrayMeta {
    ...
    /// Parsed from `_ARRAY_DIMENSIONS` (xarray/CF convention) or inferred from shape.
    pub dimensions: Option<Vec<String>>,
}
```

```rust
// src/reader/filter.rs — determine_effective_coords maps a var's dims (not all coords)
let var_coords = get_variable_coords(&var_meta.shape, var_meta.dimensions.as_deref(),
                                     coord_names, coord_sizes);
```

---

## Storage and I/O

### 13. `object_store` backends + tracked stores + I/O stats
`src/reader/storage.rs::StorageLocation::parse` classifies local vs remote;
`src/reader/tracked_store.rs` / `async_tracked_store.rs` wrap the store to count
bytes; `src/reader/stats.rs::ZarrIoStats` holds the atomic counters.

```rust
// src/reader/storage.rs — StorageLocation::parse
if location.starts_with("s3://") || location.starts_with("gs://")
   || location.starts_with("http://") || location.starts_with("https://") {
    ... is_remote: true ...
} else if location.starts_with("file://") { ... } else { /* plain local path */ }
```

```rust
// src/reader/stats.rs — ZarrIoStats (lock-free atomics)
pub struct ZarrIoStats {
    pub coord_bytes: AtomicU64, pub data_bytes: AtomicU64,
    pub disk_bytes: AtomicU64,            // compressed bytes actually read
    pub coord_arrays: AtomicU64, pub data_arrays: AtomicU64,
    pub data_nanos: AtomicU64, ...        // timing
}
```

```rust
// src/reader/zarr_reader.rs — TrackedStore wraps the real store when stats are requested
let store = Arc::new(TrackedStore::new(fs_store, stats.clone().unwrap_or_default()));
```

### 14. VirtualiZarr support
`src/reader/virtual_store.rs::VirtualStoreAdapter` adapts a reference store;
`src/reader/parquet_refs.rs::ParquetRefs` loads `refs.N.parq` chunk references.

```rust
// src/reader/parquet_refs.rs — ParquetRefs::load (sync; load_async parallelizes with join_all)
pub fn load(store_path: &str, array_name: &str) -> Result<Self, ...> {
    let pattern = format!("{}/{}/refs.*.parq", store_path, array_name);
    let mut files: Vec<_> = glob::glob(&pattern)?.filter_map(Result::ok).collect();
    files.sort();   // refs.0.parq, refs.1.parq, ...
    ...
}
```

The factory detects these stores and builds the adapter (see decision 19's
`ZarrTableFactory::create`).

---

## Distributed execution

### 15. datafusion-distributed wiring
`src/distributed.rs` — feature-gated (`distributed`); `src/bin/worker.rs` and
`src/bin/head.rs` are the cluster binaries. Both sides must agree on the codec
and the metric UDFs.

```rust
// src/distributed.rs — module doc + codec registration
//! Both sides must agree on ... the ZarrPhysicalCodec (so a ZarrExec serialized
//! on the head decodes on the worker) and the metric UDFs ...
use crate::physical_plan::codec::ZarrPhysicalCodec;
```

### 16. Outer-axis chunk-grid partitioning
`src/physical_plan/partition.rs::split_selection` slices the outer selection into
chunk-aligned pieces (never splitting a chunk across two partitions).

```rust
// src/physical_plan/partition.rs — split_selection / split_range
pub fn split_selection(sel: &CoordSelection, chunk_len: u64, target: usize) -> Vec<CoordSelection> {
    match sel {
        CoordSelection::Range(s, e) => split_range(*s, *e, chunk_len, target) ...,
        CoordSelection::Indices(v)  => split_indices(v, chunk_len, target) ...,
    }
}
// split_range breaks only on chunk boundaries (multiples of chunk_len) so a chunk
// is never read by two workers; first/last pieces are clipped to [s, e).
```

```rust
// src/physical_plan/zarr_exec.rs — each output partition reads its own outer slice
let partition_selection: Option<CoordSelection> = if self.partitions.is_empty() {
    None                                              // legacy whole-store read
} else {
    Some(self.partitions[partition].outer.clone())
};
```

### 17. Head-side resolution; workers replace their selection
`src/reader/zarr_reader.rs::resolve_outer_selection` resolves the outer-axis
filter once (on the head); `split_selection` then splits that resolved set.

### 18. Keep a chunk's survivors on one worker
`src/physical_plan/partition.rs::split_indices` groups indices sharing a chunk so
the chunk is read once; `src/reader/zarr_reader.rs::bucket_outer_indices` does the
same bucketing inside a single read.

```rust
// src/physical_plan/partition.rs — split_indices
fn split_indices(v: &[usize], chunk_len: u64, target: usize) -> Vec<Vec<usize>> {
    ...
    // Consecutive runs of indices sharing a chunk form an indivisible group:
    for i in 1..=v.len() {
        if i == v.len() || v[i] / chunk_len != v[start] / chunk_len {
            groups.push(&v[start..i]); start = i;
        }
    }
    // ... then pack groups greedily into <= target balanced partitions.
}
```

```rust
// src/reader/zarr_reader.rs — bucket_outer_indices: one read per chunk, record offsets
Some((start, end, offsets)) if idx / chunk_len == *start / chunk_len => {
    *end = idx + 1; offsets.push(idx - *start);   // same chunk: extend window
}
_ => buckets.push((idx, idx + 1, vec![0])),       // new chunk: open a bucket
```

### Distributed plumbing (supports 15–18)
`src/physical_plan/codec.rs::ZarrPhysicalCodec` serializes `ZarrExec` (incl.
partitions + coord filters) for the worker hop, via datafusion-proto;
`src/distributed.rs` provides worker discovery and task spreading.

```rust
// src/distributed.rs — StaticWorkerResolver (one entry per worker, not a VIP)
impl WorkerResolver for StaticWorkerResolver {
    fn get_urls(&self) -> Result<Vec<Url>, DataFusionError> { Ok(self.urls.clone()) }
}

// ZarrTaskEstimator: advertise one task per partition slice so the scan spreads
impl TaskEstimator for ZarrTaskEstimator {
    fn task_estimation(&self, plan: &Arc<dyn ExecutionPlan>, _: &ConfigOptions) -> Option<TaskEstimation> {
        let exec = plan.downcast_ref::<ZarrExec>()?;
        let n = exec.partitions().len();
        (n >= 2).then(|| TaskEstimation::desired(n))   // else default Maximum(1)
    }
    ...
}
```

---

## Interface and ergonomics

### 19. `CREATE EXTERNAL TABLE ... STORED AS ZARR`
`src/datasource/factory.rs::ZarrTableFactory` implements
`TableProviderFactory`; it also handles remote + VirtualiZarr detection here.

```rust
// src/datasource/factory.rs
pub struct ZarrTableFactory;

#[async_trait]
impl TableProviderFactory for ZarrTableFactory {
    async fn create(&self, _state: &dyn Session, cmd: &CreateExternalTable)
        -> Result<Arc<dyn TableProvider>> {
        if is_remote_url(&cmd.location) {
            let (store, prefix) = create_async_store(&cmd.location).await?;
            if is_virtualizarr_store_async(&store, &prefix).await {
                let adapter = VirtualStoreAdapter::new_async(&store, &prefix, &cmd.location).await?;
                ...
            }
            ...
        }
        ...
    }
}
```

### 20. `zarr_describe()` UDTF
`src/udtf.rs` — registers a table function returning xarray-style metadata.

```rust
// src/udtf.rs
pub fn register_zarr_functions(ctx: &SessionContext) {
    ctx.register_udtf("zarr_describe", Arc::new(ZarrDescribeFunc::new(ctx.clone())));
}
// Returns: column_name, type ("coord"/"data_var"), dimension, size, chunks
```

### 21. Static musl `zarr-cli` binary
`install.sh`, `build.rs`, and the release workflow under `.github/` /
`cloudbuild/`; entrypoint `src/bin/zarr_cli/main.rs`.

---

## Where it all comes together

`src/bin/zarr_cli/main.rs` is the single place the factory, both logical
optimizer rules, the physical limit-pushdown rule, and the UDTFs are registered
on the session:

```rust
// src/bin/zarr_cli/main.rs
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_config(config)
    .with_table_factory("ZARR".to_string(), Arc::new(ZarrTableFactory) as _)   // decision 19
    .with_optimizer_rule(Arc::new(CountStatisticsRule::new()))                 // decision 7
    .with_optimizer_rule(Arc::new(MinMaxStatisticsRule::new()))                // decision 7
    .with_physical_optimizer_rule(Arc::new(ZarrLimitPushdownRule::new()))      // decision 6
    .build();
let ctx = SessionContext::new_with_state(state);
register_zarr_functions(&ctx);   // decision 20  (zarr_describe)
register_metric_udfs(&ctx);      // rmse / mae / ...
```

## Reading suggestions

- **Trace one query end to end:** `datasource/zarr.rs::scan` →
  `physical_plan/zarr_exec.rs::execute` → `reader/zarr_reader.rs::read_zarr`.
- **Understand filter pushdown:** read `reader/filter.rs` top-to-bottom — the
  `CoordFilterKind` / `CoordSelection` / `CoordFilters` types, then
  `resolve_coord_selection`.
- **Understand partitioning:** `physical_plan/partition.rs`
  (`split_selection`, `split_indices`) with `reader/zarr_reader.rs`
  (`resolve_outer_selection`, `bucket_outer_indices`).
- **See the wiring:** `src/bin/zarr_cli/main.rs` (snippet above).
