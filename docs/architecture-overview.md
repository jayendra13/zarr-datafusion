# Architecture Overview — A Bird's-Eye View

How the modules fit together and the **key data structures and methods that link
them**, traced through the one thing that exercises the whole library: the
`zarr-cli` REPL running a SQL query.

Companions: [`design-decisions.md`](design-decisions.md) (the *why*) and
[`design-decisions-code-map.md`](design-decisions-code-map.md) (decision → code).

---

## The 10,000-foot picture

```
                          ┌──────────────────────────────────────────────┐
   user types SQL  ─────▶ │  src/bin/zarr_cli/main.rs   (the REPL)         │
                          │  builds a SessionState, owns the SessionContext│
                          └───────────────┬──────────────────────────────┘
                                          │ ctx.sql(...) / create_physical_plan / collect
                                          ▼
                  ┌────────────────────────────────────────────────────────┐
                  │              DataFusion  (SessionContext)                │
                  │  parses SQL · plans · applies optimizer rules · executes │
                  └───┬───────────────┬───────────────────────┬────────────┘
        register      │ CREATE TABLE  │ SELECT (scan)          │ optimize
        the engine    ▼               ▼                        ▼
   ┌───────────────────────┐  ┌──────────────────┐   ┌────────────────────────┐
   │ datasource/factory.rs │  │ datasource/zarr.rs│   │ optimizer/*.rs          │
   │ ZarrTableFactory      │  │ ZarrTable         │   │ Count/MinMax (logical)  │
   │  → infer schema       │  │  ::scan()         │   │ ZarrLimitPushdown (phys)│
   │  → build ZarrTable    │  │  → build ZarrExec │   └───────────┬────────────┘
   └──────────┬────────────┘  └─────────┬────────┘               │ rewrites
              │ ZarrStoreMeta            │ ZarrExec               │ ZarrExec
              ▼                          ▼                        ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                     physical_plan/zarr_exec.rs  (ZarrExec)                 │
   │   ExecutionPlan: per-partition execute() → SendableRecordBatchStream       │
   │   carries: schema · path · projection · limit · CoordFilters · partitions  │
   └───────────────────────────────┬───────────────────────────────────────────┘
                                    │ read_zarr / read_zarr_async
                                    ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                         reader/  (the engine room)                         │
   │  schema_inference · filter · coord · dtype · cf_time · zarr_reader         │
   │  storage · tracked_store · virtual_store · parquet_refs · stats            │
   │  nD Zarr chunks ──▶ flatten ──▶ Arrow columns ──▶ RecordBatch              │
   └─────────────────────────────────────────────────────────────────────────┘

   optional: physical_plan/codec.rs + distributed.rs ship ZarrExec to remote
             worker/head nodes (feature = "distributed").
```

---

## The CLI request, end to end

`src/bin/zarr_cli/main.rs` is the conductor. Everything else is a library the
DataFusion `SessionContext` calls into.

### Step 0 — wire the engine into a session (once, at startup)
```rust
// src/bin/zarr_cli/main.rs
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_table_factory("ZARR".into(), Arc::new(ZarrTableFactory) as _) // CREATE ... STORED AS ZARR
    .with_optimizer_rule(Arc::new(CountStatisticsRule::new()))          // logical rules
    .with_optimizer_rule(Arc::new(MinMaxStatisticsRule::new()))
    .with_physical_optimizer_rule(Arc::new(ZarrLimitPushdownRule::new()))// physical rule
    .build();
let ctx = SessionContext::new_with_state(state);
register_zarr_functions(&ctx);   // zarr_describe() UDTF
register_metric_udfs(&ctx);      // rmse / mae / ...
```
**Link:** the `SessionContext` is the hub. The factory, both logical rules, the
physical rule, and the UDTFs all register here — this is the only place the
library's pieces are bolted onto DataFusion.

### Step 1 — `CREATE EXTERNAL TABLE weather STORED AS ZARR LOCATION '...'`
DataFusion routes the DDL to the registered factory.
```
ctx.sql("CREATE EXTERNAL TABLE ...")
   → ZarrTableFactory::create(cmd)               // datasource/factory.rs
       → infer_schema* / detect_zarr_version      // reader/schema_inference.rs
       → ZarrTable { schema, path, store_meta, io_stats, caches }
   → registered as a TableProvider in the SessionContext
```
**Linking structures:**
- `ZarrStoreMeta` / `ZarrArrayMeta` (`reader/schema_inference.rs`) — the metadata
  the factory infers once and hands to `ZarrTable`; later reused by `scan`,
  `filter`, and `zarr_describe` (no re-reading metadata).
- `SchemaRef` (Arrow) — the schema contract shared by table, exec, and reader.

### Step 2 — `SELECT ... FROM weather WHERE ... LIMIT n`
DataFusion plans the query and calls `ZarrTable::scan`.
```
ctx.sql("SELECT ...")  →  TableProvider::scan(projection, filters, limit)   // datasource/zarr.rs
   ├─ parse_coord_filters(filters, coord_names)  → CoordFilters             // reader/filter.rs
   ├─ plan_partitions(... target_partitions)     → Vec<PartitionSpec>       // physical_plan/partition.rs
   └─ ZarrExec::new(schema, path, projection, limit, …, Some(CoordFilters)) // coord_filters is a ctor arg
          .with_partitions(Vec<PartitionSpec>)                              // physical_plan/zarr_exec.rs
```
**Linking structures (the "currencies" passed down):**
- `CoordFilters` — parsed WHERE predicates per coordinate (decision 5/8).
- `PartitionSpec { outer: CoordSelection }` — one per output partition (decision 16).
- `Option<Vec<usize>>` projection + `Option<usize>` limit — the other pushdowns.

### Step 3 — optimizer rules rewrite the plan
- Logical: `CountStatisticsRule` / `MinMaxStatisticsRule` may replace an aggregate
  with a constant from `ZarrStoreMeta` (no scan at all).
- Physical: `ZarrLimitPushdownRule` finds a limit and folds it into the scan via
  `ZarrExec::with_limit(...)`.

### Step 4 — execute → Arrow stream
DataFusion polls the physical plan; the CLI uses `collect(plan, task_ctx)`.
```
ZarrExec::execute(partition, ctx)                       // physical_plan/zarr_exec.rs
   → partition_selection = self.partitions[partition].outer  (a CoordSelection)
   → read_zarr / read_zarr_async(path, schema, projection, limit,
                                 io_stats, coord_filters, partition_selection)
   → SendableRecordBatchStream  (one RecordBatch per partition today)
```
Inside the reader (`reader/zarr_reader.rs`), the rest of the module collaborates:
```
resolve_coord_selection(filters, coord values)   // filter.rs  → CoordSelection (∩)
build_read_plans(selections, shapes, chunks)      // → ReadPlan { ArraySubset, Keep }
storage.rs + tracked_store.rs                     // read chunk bytes, count into io_stats
dtype.rs / cf_time.rs / coord.rs                  // decode dtypes, CF time, build DictionaryArrays
create_result_batch(schema, arrays, rows)         // ArrayRef columns → Arrow RecordBatch
```

### Step 5 — results + I/O stats back to the user
```rust
// src/bin/zarr_cli/main.rs — execute_statement
let plan = df.create_physical_plan().await?;
let io_stats = find_zarr_exec_stats(&plan);     // pull SharedIoStats out of the ZarrExec node
let batches = collect(plan, ctx.task_ctx()).await?;
print_batches(&batches)?;                        // Arrow's pretty-printer
print_stats_line(row_count, elapsed, io_stats);  // "5 rows · 3 arrays · 6.70 KB disk · ..."
```
**Link:** `SharedIoStats` (`Arc<ZarrIoStats>`, `reader/stats.rs`) is shared
between the `TrackedStore` (which increments it during reads) and the `ZarrExec`
(which holds it). The CLI fishes it back out of the plan tree to print the
stats line — the same `Arc`, no copying.

---

## The data structures that stitch modules together

| Structure / type | Defined in | Links … | Role |
| --- | --- | --- | --- |
| `SessionContext` / `SessionState` | DataFusion | CLI ↔ everything | The hub; factory, rules, UDTFs register here |
| `ZarrStoreMeta` / `ZarrArrayMeta` | `reader/schema_inference.rs` | factory → table → scan → filter → udtf | Inferred-once Zarr metadata (shapes, chunks, dims, CF time, bounds) |
| `SchemaRef` (Arrow) | arrow | table ↔ exec ↔ reader | The agreed column schema |
| `ZarrTable` | `datasource/zarr.rs` | DataFusion `TableProvider` ↔ engine | Holds schema + meta + caches; builds `ZarrExec` in `scan` |
| `ZarrExec` | `physical_plan/zarr_exec.rs` | planner ↔ reader ↔ distributed | The `ExecutionPlan`; carries every pushdown to `read_zarr` |
| `CoordFilters` / `CoordFilterKind` | `reader/filter.rs` | scan → exec → reader | Filter-pushdown currency (WHERE on coordinates) |
| `CoordSelection` | `reader/filter.rs` | filter ↔ partition ↔ reader | Resolved positions (`Range`/`Indices`); intersected & sliced |
| `PartitionSpec` | `physical_plan/partition.rs` | scan → exec → execute | One output partition's outer-axis `CoordSelection` |
| `RecordBatch` / `SendableRecordBatchStream` | arrow / datafusion | reader → exec → DataFusion → CLI | The columnar result + its pull-based stream |
| `SharedIoStats` (`Arc<ZarrIoStats>`) | `reader/stats.rs` | tracked_store → exec → CLI | Lock-free I/O counters surfaced as the stats line |
| `ZarrPhysicalCodec` | `physical_plan/codec.rs` | head ↔ worker | Serializes `ZarrExec` (incl. filters + partitions) for the network hop |

---

## The methods that hand work across module boundaries

| Boundary | Method | What crosses |
| --- | --- | --- |
| SQL DDL → engine | `ZarrTableFactory::create` | `CreateExternalTable` → `Arc<dyn TableProvider>` (`ZarrTable`) |
| store → schema | `infer_schema*` / `detect_zarr_version` | store path → `SchemaRef` + `ZarrStoreMeta` |
| query plan → scan | `ZarrTable::scan` | projection/filters/limit → `Arc<dyn ExecutionPlan>` (`ZarrExec`) |
| WHERE → pushdown | `parse_coord_filters` | `&[Expr]` → `CoordFilters` |
| scan → partitions | `plan_partitions` / `split_selection` | `CoordSelection` → `Vec<PartitionSpec>` |
| optimize → scan | `ZarrExec::with_limit` | folds `LIMIT` into the scan |
| exec → reader | `read_zarr` / `read_zarr_async` | all pushdowns → `SendableRecordBatchStream` |
| filters → positions | `resolve_coord_selection` | `CoordFilters` + coord values → `CoordSelection` |
| selection → chunks | `build_read_plans` | `CoordSelection`s → `ReadPlan { ArraySubset, Keep }` |
| values → Arrow | `create_result_batch` | `Vec<ArrayRef>` → `RecordBatch` |
| reads → stats | `TrackedStore` (wraps store) | chunk bytes → `SharedIoStats` |
| plan → CLI stats | `find_zarr_exec_stats` | walks the plan tree → `SharedIoStats` |
| local → distributed | `ZarrPhysicalCodec` (encode/decode) | `ZarrExec` ⇄ bytes across worker/head |

---

## Module dependency map (who calls whom)

```
bin/zarr_cli ──▶ datasource ──▶ physical_plan ──▶ reader
      │              │                │              ▲
      │              └────────────────┼──────────────┘   (datasource & physical_plan
      ▼                               │                    both lean on reader types)
   optimizer ◀───────────────────────┘
      ▲
      └── (registered on the SessionContext by the CLI)

distributed ──▶ physical_plan (ZarrExec, codec) + udfs   [feature = "distributed"]
udtf / udfs ──▶ reader (schema meta) ; registered by the CLI
```

- **`reader/`** is the foundation — it depends on nothing else in the crate and
  everyone depends on it (its types are the shared vocabulary).
- **`physical_plan/`** wraps the reader as a DataFusion `ExecutionPlan` and adds
  partitioning + the serialization codec.
- **`datasource/`** is the DataFusion entry surface (`TableProvider` +
  `TableProviderFactory`) that builds `physical_plan` nodes.
- **`optimizer/`**, **`udtf`/`udfs`**, and **`distributed`** are cross-cutting
  add-ons the CLI (or the worker/head) registers on the session.

---

## TL;DR — one sentence per module

- **`bin/zarr_cli`** — builds the session, runs the REPL, prints batches + I/O stats.
- **`datasource`** — `ZarrTable` (`TableProvider`) and the `CREATE TABLE` factory.
- **`physical_plan`** — `ZarrExec` (`ExecutionPlan`), partition planning, distributed codec.
- **`reader`** — reads Zarr, flattens nD → Arrow `RecordBatch`, resolves filters, tracks I/O.
- **`optimizer`** — short-circuits `COUNT`/`MIN`/`MAX` and pushes `LIMIT` into the scan.
- **`udtf` / `udfs`** — `zarr_describe()` and metric functions (`rmse`, `mae`).
- **`distributed`** — ships `ZarrExec` to worker nodes over Arrow Flight.
</content>
