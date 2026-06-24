# Design Decisions — Code Map

Companion to [`design-decisions.md`](design-decisions.md). For each decision it
points at the concrete files, traits, types, and functions that implement it, so
you can jump from the *why* to the *where*. Line numbers are approximate
(`file:line`) and drift as the code changes — search by symbol name if a number
is stale.

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
| CLI | `src/bin/zarr_cli/` | REPL; **wires the optimizer rules + UDTF into the session** |

**Data flow:** SQL → `ZarrTable::scan` (`datasource/zarr.rs:262`) → `ZarrExec::execute`
(`physical_plan/zarr_exec.rs:242`) → `read_zarr` / `read_zarr_async`
(`reader/zarr_reader.rs:854` / `:1174`) → `RecordBatch`.

---

## Foundational data model

### 1. Flatten nD → 2D (Cartesian product)
- `src/reader/zarr_reader.rs` — `read_zarr` (`:854`), `read_zarr_async` (`:1174`):
  the core flattening + Arrow conversion.
- `src/reader/zarr_reader.rs` — `build_read_plans` (`:380`): turns a coordinate
  selection into the chunk reads that materialize the rows.

### 2. Structural convention (1D = coord, nD = data var)
- `src/reader/schema_inference.rs` — `infer_schema` (`:202`), `detect_zarr_version`
  (`:104`), `enum ZarrVersion` (`:45`): role inference from array shape and v2/v3
  detection.

### 3. Coordinates as `DictionaryArray`
- `src/reader/coord.rs` — `create_coord_dictionary` / `create_coord_dictionary_typed`
  (`:483`): builds the dictionary-encoded coordinate columns.

### 4. Adaptive dictionary key width (Int16 → Int32 → Int64)
- `src/reader/dtype.rs` — `dictionary_key_type_for_cardinality` (`:97`): picks the
  key type from cardinality.
- `src/reader/schema_inference.rs` — declares the key type from `coord.shape[0]`;
  used at `dtype.rs:110`/`:114`.
- `src/reader/coord.rs` — `create_coord_dictionary_typed` (`:483`) builds keys at
  the schema-declared width (fails loudly if undersized).

---

## Pushdown and optimization

### 5. Internalize coordinate filters into the scan
- `src/reader/filter.rs` — `struct CoordFilters` (`:171`),
  `enum CoordFilterKind` (`:36`): parsed coordinate predicates.
- `src/physical_plan/zarr_exec.rs` — `coord_filters` accessor (`:194`): how the
  scan carries them.
- `src/datasource/zarr.rs` — `scan` (`:262`): where WHERE-clause exprs are turned
  into `CoordFilters` and handed to `ZarrExec`.

### 6. LIMIT pushdown past `FilterExec`
- `src/optimizer/limit_pushdown.rs` — `struct ZarrLimitPushdownRule` (`:95`): the
  physical optimizer rule.
- `src/physical_plan/zarr_exec.rs` — `with_limit` (`:206`): how the rule injects
  the limit into the scan.
- Registered in `src/bin/zarr_cli/main.rs` (`with_physical_optimizer`, `:226`).

### 7. MIN/MAX/COUNT from statistics
- `src/optimizer/count_optimization.rs` — `CountStatisticsRule` (`:29`).
- `src/optimizer/minmax_optimization.rs` — `MinMaxStatisticsRule` (`:28`).
- `src/optimizer/mod.rs` — re-exports (`:5`–`:7`).
- Registered in `src/bin/zarr_cli/main.rs` (`with_optimizer`, `:224`–`:225`).

### 8. Vec-of-filters per coordinate (AND-composed)
- `src/reader/filter.rs` — `struct CoordFilters` now holds a `Vec<CoordFilterKind>`
  per coord (`:171`); `enum CoordFilterKind` variants incl. `InList` (`:59`),
  `DatePartSet` (`:62`), `DatePart` (`:51`).
- `src/reader/filter.rs` — `struct PartialBounds` (`:300`), `into_filters` (`:322`):
  emit every collected predicate instead of dropping losers.
- `src/reader/filter.rs` — `enum CoordSelection` (`:109`), `intersect` (`:145`),
  `resolve_coord_selection` (`:782`): resolve each predicate to a position set and
  intersect.
- `src/physical_plan/codec.rs` — `enum CoordFilterKind` mirror (`:74`): round-trips
  the filters through the distributed codec.

### 9. Coordinate-only queries skip Cartesian expansion (gated on LIMIT)
- `src/reader/filter.rs` — `determine_effective_coords` (`:1721`): picks only the
  projected coords' dimensions when a `LIMIT` is present.

### 10. Compact coordinate encoding for arithmetic sequences
- `src/reader/coord.rs` — `enum CompactCoord` (`:37`): O(1) `(first, step, len)`
  representation with index/range lookups.
- `src/reader/filter.rs` — `filter_satisfiable_by_bounds` (`:222`): early rejection
  of impossible filters from coordinate min/max.

---

## Domain correctness

### 11. CF-time decoding
- `src/reader/cf_time.rs` — `struct CFTimeAttrs` (`:24`), `struct CFTimeUnit` (`:33`),
  `decode_cf_time` (`:159`), `decode_cf_time_f64` (`:177`).

### 12. Mixed-dimensionality variables
- `src/reader/schema_inference.rs` — `_ARRAY_DIMENSIONS` parsing (`:150`, `:409`):
  stores per-variable dimension names.
- `src/reader/filter.rs` — `determine_effective_coords` (`:1721`),
  `match_ranges_to_data_var` (`:1562`): map filters/coords onto variables with
  fewer dims than the full coordinate set.

---

## Storage and I/O

### 13. `object_store` backends + tracked stores
- `src/reader/storage.rs` — `parse` (`:67`), `is_remote_url` (`:247`): builds
  local / GCS (`GoogleCloudStorageBuilder`) / S3 (`AmazonS3Builder`) stores.
- `src/reader/tracked_store.rs` — `struct TrackedStore` (`:21`): sync read
  byte-counting wrapper.
- `src/reader/async_tracked_store.rs` — `struct AsyncTrackedStore` (`:23`): async
  equivalent.
- `src/reader/stats.rs` — `struct ZarrIoStats` (`:15`): atomic counters surfaced in
  the CLI stats line.

### 14. VirtualiZarr support
- `src/reader/virtual_store.rs` — `VirtualStoreAdapter` (used from
  `physical_plan/zarr_exec.rs:6`/`:24`): reference-store adapter, cached on
  `ZarrTable`.
- `src/reader/parquet_refs.rs` — `struct ParquetRefs` (`:31`): loads
  `refs.N.parq` chunk references (sync + async/parallel).

---

## Distributed execution

### 15. datafusion-distributed pivot
- `src/distributed.rs` — feature-gated wiring (`distributed`).
- `src/bin/worker.rs`, `src/bin/head.rs` — the two cluster binaries.

### 16. Outer-axis chunk-grid partitioning
- `src/physical_plan/partition.rs` — `split_selection` (`:82`): slices the outer
  chunk grid into N balanced partitions.
- `src/physical_plan/zarr_exec.rs` — `execute` (`:242`) exposes the partitions.

### 17. Head-side resolution; workers replace their selection
- `src/reader/zarr_reader.rs` — `resolve_outer_selection` (`:788`): single
  resolution site on the head.
- `src/physical_plan/partition.rs` — `split_selection` (`:82`): splits the
  resolved set across partitions.

### 18. Keep a chunk's survivors on one worker
- `src/physical_plan/partition.rs` — `split_indices` (`:141`): groups indices that
  share a chunk before packing into balanced partitions.
- `src/reader/zarr_reader.rs` — `bucket_outer_indices` (`:475`): buckets scattered
  outer indices per chunk so a chunk is read once.

### Distributed plumbing (supports 15–18)
- `src/distributed.rs` — `ZarrPhysicalCodec` (`:5`/`:29`): serializes `ZarrExec`
  (incl. partitions, coord filters) for the worker hop.
- `src/distributed.rs` — `StaticWorkerResolver` (`:39`), `ZarrTaskEstimator` (`:78`):
  worker discovery + task spreading.
- `src/physical_plan/codec.rs` — proto mirrors of `ZarrExec` / `ZarrTable` /
  `CoordFilterKind` (`:157`, `:283`, `:74`).

---

## Interface and ergonomics

### 19. `CREATE EXTERNAL TABLE ... STORED AS ZARR`
- `src/datasource/factory.rs` — `struct ZarrTableFactory` (`:19`),
  `impl TableProviderFactory` (`:22`).
- `src/datasource/zarr.rs` — `struct ZarrTable` (`:28`),
  `impl TableProvider for ZarrTable` (`:229`).
- Registered in `src/bin/zarr_cli/main.rs:17`.

### 20. `zarr_describe()` UDTF
- `src/udtf.rs` — `zarr_describe` (`:24`), `register_udtf` (`:23`).

### 21. Static musl `zarr-cli` binary
- `install.sh`, `build.rs`, and the release workflow under `.github/` /
  `cloudbuild/`; entrypoint `src/bin/zarr_cli/main.rs`.

---

## Reading suggestions

- **Trace one query end to end:** start at `datasource/zarr.rs:262` (`scan`),
  follow into `physical_plan/zarr_exec.rs:242` (`execute`), then
  `reader/zarr_reader.rs:854` (`read_zarr`).
- **Understand filter pushdown:** read `reader/filter.rs` top-to-bottom — the type
  definitions (`:36`–`:171`) then `resolve_coord_selection` (`:782`).
- **Understand partitioning:** `physical_plan/partition.rs` (`split_selection`,
  `split_indices`) paired with `reader/zarr_reader.rs` (`resolve_outer_selection`,
  `bucket_outer_indices`).
- **See the wiring:** `src/bin/zarr_cli/main.rs` registers the factory, optimizer
  rules, and UDTF — the single place the pieces come together.
</content>
