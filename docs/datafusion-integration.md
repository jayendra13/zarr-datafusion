# DataFusion Integration — What We Own vs. What We Borrow

zarr-datafusion is not a query engine. It is a **DataFusion data source plus a
small bundle of domain-specific optimizer rules**. This doc names the DataFusion
concepts we plug into and draws a clean line between the logic we own and the
machinery we borrow.

Companions: [`architecture-overview.md`](architecture-overview.md) (how modules
connect) and [`design-decisions.md`](design-decisions.md) (the *why*).

---

## The DataFusion extension points we implement

DataFusion is built to be extended at specific seams. We plug into exactly these:

| Seam (DataFusion trait / API) | Our implementation | What it buys us |
| --- | --- | --- |
| `TableProvider` | `ZarrTable` (`datasource/zarr.rs`) | Makes a Zarr store look like a SQL table; its `scan()` is our hook for projection / filters / limit |
| `TableProviderFactory` | `ZarrTableFactory` (`datasource/factory.rs`) | Wires `CREATE EXTERNAL TABLE ... STORED AS ZARR` into SQL |
| `ExecutionPlan` | `ZarrExec` (`physical_plan/zarr_exec.rs`) | A leaf node in the physical plan; `execute(partition)` returns our stream |
| `PhysicalOptimizerRule` | `ZarrLimitPushdownRule` (`optimizer/limit_pushdown.rs`) | Folds `LIMIT` into the scan |
| `OptimizerRule` (logical) | `CountStatisticsRule`, `MinMaxStatisticsRule` (`optimizer/`) | Replaces aggregates with constants before planning |
| `TableFunctionImpl` (UDTF) | `zarr_describe()` (`udtf.rs`) | Custom table function in SQL |
| `ScalarUDF` / `AggregateUDF` | `rmse`, `mae`, … (`udfs/`) | Domain functions usable in SQL |
| `TaskEstimator` / `WorkerResolver` (datafusion-distributed) | `ZarrTaskEstimator`, `StaticWorkerResolver` (`distributed.rs`) | Spread our scan across worker nodes |
| `PhysicalExtensionCodec` (datafusion-proto) | `ZarrPhysicalCodec` (`physical_plan/codec.rs`) | Serialize our custom `ZarrExec` over the wire |

### Vocabulary we consume (speak, but don't implement)

- `Expr` — the logical expression tree we read in `parse_coord_filters` to find
  coordinate predicates.
- `ScalarValue` — the literal type carried inside `CoordFilterKind` and serialized
  by the codec.
- `SchemaRef`, `RecordBatch`, `ArrayRef`, `DictionaryArray` — Arrow, strictly, but
  the columnar substrate DataFusion mandates.
- `SendableRecordBatchStream` / `RecordBatchStreamAdapter` — the pull-based output
  contract.
- `PlanProperties` (`Partitioning`, `EquivalenceProperties`, `Boundedness`,
  `EmissionType`) — metadata an `ExecutionPlan` must advertise so the planner can
  reason about us.
- `SessionContext` / `SessionStateBuilder` — the registry we bolt everything onto.
- `TableProviderFilterPushDown` (`Exact` / `Inexact` / `Unsupported`) — how `scan`
  tells DataFusion which filters it handled.

---

## Where the ownership line falls

The cleanest mental model: **DataFusion owns the trunk and branches of the query;
we own one leaf and a few pruning shears.**

```
        SQL text
           │   ◀── DataFusion: parser, binder
        LogicalPlan
           │   ◀── DataFusion: optimizer framework
           │   ◀── OURS: Count / MinMax logical rules   (a few leaves)
        PhysicalPlan
           │   ◀── DataFusion: HashAggregate, Sort, Filter, Repartition,
           │                   CoalescePartitions, GlobalLimit, joins, windows…
           │   ◀── OURS: ZarrLimitPushdown physical rule
        ┌──┴──────────────────────────────┐
        │ everything above the scan         │  ← borrowed, in full
        ├───────────────────────────────────┤
        │ ZarrExec (the leaf)               │  ← OURS
        │   read_zarr → RecordBatch stream  │
        └───────────────────────────────────┘
```

### Borrowed from DataFusion (we write zero of this)

- SQL parsing, name/type binding, type coercion.
- The entire relational algebra above the scan: `JOIN`, `GROUP BY`, `HAVING`,
  `ORDER BY`, window functions, subqueries, `DISTINCT`, set ops.
- The physical operators that run them (hash aggregate, sort-merge, repartition,
  coalesce, the generic limit / filter execs).
- Arrow compute kernels (comparisons, arithmetic, casting).
- The streaming, partition-parallel execution driver (`collect`, `task_ctx`).
- The optimizer *framework* and most generic rules (projection-pushdown plumbing,
  constant folding, common-subexpression elimination, …).
- datafusion-distributed's Arrow Flight transport and stage planning.

### Owned by zarr-datafusion (the genuinely new logic)

Everything we own is plugged into DataFusion through one of its extension traits —
that *is* the ownership boundary. Each owned type below is annotated with the
DataFusion seam it implements (full list in the table up top):

- **The data-source seam** — `ZarrTable` *implements* `TableProvider`,
  constructed by `ZarrTableFactory` *implementing* `TableProviderFactory`. This is
  how a Zarr store enters DataFusion as a SQL table.
- **The execution seam** — `ZarrExec` *implements* `ExecutionPlan`: the leaf node
  DataFusion executes, advertising `PlanProperties` and returning our stream from
  `execute(partition)`.
- **The leaf logic** — everything in `reader/`: Zarr decoding, the nD→2D flattening
  model, CF-time, dtype mapping, dictionary coordinate encoding. Reached *through*
  `ZarrExec` via `read_zarr` / `read_zarr_async`, but implements no DataFusion
  trait itself — DataFusion has no concept of Zarr; this is 100% ours.
- **Our own filter IR** — `CoordFilters` / `CoordFilterKind` / `CoordSelection`.
  We *read* DataFusion's `Expr` (and answer `TableProviderFilterPushDown` in
  `supports_filters_pushdown`) but translate it into our own structures and resolve
  them to array positions (`resolve_coord_selection`, set intersection).
- **Domain-aware optimizations DataFusion can't make** — `ZarrLimitPushdownRule`
  *implements* `PhysicalOptimizerRule` (pushing `LIMIT` *past* a filter, which the
  generic planner refuses but which is provably safe here); `CountStatisticsRule` /
  `MinMaxStatisticsRule` *implement* the logical `OptimizerRule` (constant-folding
  aggregates from `ZarrStoreMeta`).
- **Partitioning strategy** — `PartitionSpec`, `split_selection` / `split_indices`:
  chunk-aware slicing of a scan, surfaced through `ZarrExec`'s `Partitioning`.
- **Distributed wiring** — `ZarrTaskEstimator` / `StaticWorkerResolver` *implement*
  datafusion-distributed's `TaskEstimator` / `WorkerResolver`; `ZarrPhysicalCodec`
  *implements* `PhysicalExtensionCodec` to serialize our `ZarrExec`.
- **Functions** — `zarr_describe()` *implements* `TableFunctionImpl` (UDTF);
  `rmse` / `mae` *implement* `ScalarUDF` / `AggregateUDF`.
- **Observability** — `ZarrIoStats` + `TrackedStore` (compressed-vs-uncompressed
  byte accounting); pure internal logic, no DataFusion trait.

---

## The interesting boundary cases (where the line blurs)

Three places are worth understanding, because that is where "borrow vs. own" gets
subtle.

1. **Filter pushdown — shared.** DataFusion owns the *mechanism* (it offers
   filters to `scan` and honors our `Exact` / `Inexact` verdict); we own the
   *semantics* (which predicates we can resolve to positions, and the promise that
   we actually applied them). If we return `Exact` but fail to filter, DataFusion
   produces wrong results — the contract is ours to keep.

2. **Limit pushdown — we deliberately stepped outside DataFusion's rules.** The
   generic planner will *not* push a limit below a filter. Our custom physical rule
   does, justified entirely by a domain invariant: sorted coordinate filters are
   resolved in-scan, so the surviving rows are known up front. This is the clearest
   case of owning logic that *contradicts* DataFusion's defaults.

3. **Statistics — we bypass execution entirely.** The `Count` / `MinMax` rules
   answer from `ZarrStoreMeta` so the scan never runs. DataFusion also has a
   statistics framework (`Statistics` on `ExecutionPlan`); we chose explicit
   rewrite rules instead. *Follow-up worth considering:* implementing
   `ExecutionPlan::statistics()` might let DataFusion's own aggregate rules do this,
   shrinking our owned surface.

---

## One-line summary

We are a **data source plus a small bundle of domain-specific optimizer rules**.
We own the leaf of the plan (reading and flattening Zarr) and a few rewrites that
exploit scientific-data invariants; DataFusion owns the entire query engine above
the scan.
