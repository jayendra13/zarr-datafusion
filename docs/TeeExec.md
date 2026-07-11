# Query Debugging via Intermediate-Output Dumps (`TeeExec`)

> Status: **design / parked** — not yet implemented. Captured here so we can come
> back to it. See "Recommended design" for the intended approach and
> "Implementation plan (phased)" for the concrete build order. Ships **gated**:
> compiled out by default (Cargo feature `tee-debug`) and inert at runtime unless
> explicitly asked for a dump.

## Goal

A debugging tool that dumps the output of each query-execution step to Parquet on
disk, so we can examine intermediate results and understand how a query (e.g.
`sql/oni_djf2025_extract.sql`) actually runs.

## The one reframe: logical steps have no output to dump

A **logical plan is not executable** — its nodes (`Filter`, `Aggregate`, `Join`)
describe intent, not data. There is no `RecordBatch` to capture at a logical node.
Only the **physical plan** produces data: every `ExecutionPlan` node exposes
`execute(partition) -> SendableRecordBatchStream`.

So "dump each logical step" isn't literally possible. What *is* possible and just
as useful: dump each **physical node's** output, then map it back to the logical
operator it lowered from (most are close to 1:1 — `Filter -> FilterExec`,
`Aggregate -> AggregateExec partial+final`, `Join -> HashJoinExec`). That mapping
is the "understand execution" payoff.

## Why we already have the prerequisites

The two hard pieces exist in this codebase:

- A custom `ExecutionPlan` — `ZarrExec` (`src/physical_plan/zarr_exec.rs`).
- A registered physical optimizer rule — `ZarrLimitPushdownRule`, wired in
  `src/bin/zarr_cli/main.rs` via `.with_physical_optimizer_rule(...)`.

A new `TeeExec` and `InstrumentDumpRule` slot into the same architecture
(`src/optimizer/` + custom exec) with low risk.

## Recommended design: a pass-through "tee" operator + a rule that inserts it

Avoid the naive approach — walking the plan and calling `collect()` on each
subtree — because it **re-executes overlapping work N times** (for the ONI query
that re-reads the ~200 MB scan once per node), and stateful operators (partial
aggregates) behave differently when executed in isolation.

Instead, **one execution, tap everything**:

1. **`TeeExec`** — a custom `ExecutionPlan` that wraps a child. Its `execute()`
   calls `child.execute()` and wraps the returned stream so each batch is
   (a) forwarded downstream unchanged and (b) written to a per-node Parquet file
   as a side effect. It is an identity operator — zero semantic change. (Same
   shape as `ZarrExec`, so low risk.)
2. **`InstrumentDumpRule`** — a `PhysicalOptimizerRule` that runs **last** (after
   all real optimizations, so it can't perturb plan choices), walks the tree,
   assigns each node a stable pre-order id, and wraps it in `TeeExec`.
3. **Output layout** on disk:

   ```
   debug_dump/
     manifest.txt              # EXPLAIN ANALYZE text + node-id -> file map + logical-op mapping
     00_ProjectionExec/part-0.parquet
     01_HashJoinExec/part-0.parquet
     ...
     14_ZarrExec/part-0.parquet
   ```

Because the tee is pass-through and inserted after optimization, a single normal
run dumps every level correctly, with right partitioning and no extra scans.

## Gating strategy (two independent gates)

This is a debugging-only tool. It must add **zero cost and zero surface** to
normal builds and normal runs. Two gates, matching conventions already in the
repo:

1. **Compile-time — Cargo feature `tee-debug` (off by default).** All new code
   (`TeeExec`, `InstrumentDumpRule`, the Parquet sink, the `--inspect` reader)
   lives behind `#[cfg(feature = "tee-debug")]`, exactly like the `icechunk` and
   `distributed` features in `Cargo.toml`. A default `cargo build` never compiles
   the tee, never pulls it into the plan, and pays nothing. `parquet = "58"` is
   already a dependency (features `arrow`, `async`), so no new crate is needed —
   but we still gate the *code* so the operator can't accidentally be constructed
   in a release build.
2. **Runtime — opt-in per statement (off by default even when compiled in).**
   Building with `--features tee-debug` only makes the capability *available*.
   The `InstrumentDumpRule` is installed but produces an **identity plan
   transform** (no tees inserted) unless a dump was explicitly requested for the
   current statement via the CLI `--dump-dir <path>` flag / `\dump` meta-command /
   `ZARR_TEE_DUMP_DIR` env var (see Phase 3). No request ⇒ the plan is
   byte-for-byte the un-instrumented plan.

Net effect of the two gates:

| Build | Runtime request | Behavior |
|-------|-----------------|----------|
| default (no feature) | n/a | tee code absent; normal plan; zero overhead |
| `--features tee-debug` | no `--dump-dir` | rule present but inert; identity plan |
| `--features tee-debug` | `--dump-dir DIR` | tees inserted; Parquet dump + manifest |

## Implementation plan (phased)

Each phase is independently landable, leaves `main` green, and (from Phase 1 on)
is fully behind the `tee-debug` feature. Phase 0 is unconditional (it uses only
DataFusion built-ins) and is worth doing first regardless of whether we ever
finish the tee.

### Phase 0 — `EXPLAIN ANALYZE` baseline (ungated, no new deps)

Free wins that also become the manifest backbone. DataFusion already threads
per-operator row counts + wall time through `MetricsSet`; `EXPLAIN ANALYZE`
renders it.

- **Deliverable:** confirm `EXPLAIN ANALYZE <sql>` works through `zarr-cli`
  (batch + REPL) and surfaces `ZarrExec`'s metrics. Today `ZarrExec` reports I/O
  via `ZarrIoStats` but does **not** implement `ExecutionPlan::metrics()` — add a
  `MetricsSet` (baseline `output_rows` + `elapsed_compute` via
  `BaselineMetrics`) so the leaf shows up in `ANALYZE` output like the built-in
  operators.
- **Files:** `src/physical_plan/zarr_exec.rs` (add `metrics()` + wire
  `BaselineMetrics` into each `execute_*` stream).
- **Acceptance:** `EXPLAIN ANALYZE SELECT …` shows a non-zero `output_rows` /
  `elapsed_compute` on the `ZarrExec` line; a golden test asserts the metric
  fields are present.
- **Why first:** answers "what ran, how many rows, how long" with no Parquet, and
  the emitted text is dropped verbatim into `manifest.txt` in Phase 2.

### Phase 1 — `TeeExec` operator (gated)

A pass-through `ExecutionPlan` that wraps a single child. Identity semantics; the
only side effect is writing each forwarded batch to Parquet.

- **New file:** `src/physical_plan/tee_exec.rs` (module gated
  `#[cfg(feature = "tee-debug")]` from `physical_plan/mod.rs`).
- **Shape:** mirror `ZarrExec` — `PlanProperties` **delegated from the child**
  (same schema, same `Partitioning`, same emission/boundedness), `children()`
  returns the one child, `with_new_children()` rebuilds around the replacement.
- **`execute(partition, ctx)`:** call `child.execute(partition, ctx)`, wrap the
  returned `SendableRecordBatchStream` in an adapter that, for each `Ok(batch)`,
  (a) writes it to `<dump_dir>/<id>_<OpName>/part-<partition>.parquet` via
  `parquet::arrow::AsyncArrowWriter` and (b) yields the **same batch** unchanged.
  The writer is created lazily on first batch (so empty partitions produce no
  file) and finalized when the stream ends.
- **Config carried on the node:** `dump_dir: PathBuf`, `node_id: usize`,
  `op_label: String`, `row_cap: Option<usize>` (stop writing — not stop
  forwarding — after N rows per `(node, partition)`).
- **Acceptance / tests** (`tests/integration_tee.rs`, gated):
  - byte-identical result: run a query with and without a hand-wrapped `TeeExec`,
    assert `RecordBatch` streams are equal (identity property).
  - round-trip: the dumped Parquet reads back to the same rows, **including
    `DictionaryArray` coord columns** (see Gotchas).
  - `row_cap` truncates the file but not the downstream stream.
- **Not yet wired into planning** — constructed directly in tests only.

### Phase 2 — `InstrumentDumpRule` (gated)

A `PhysicalOptimizerRule` that inserts the tees. Runs **last** so it can't
perturb plan choices (repartition/coalesce adjacency).

- **New file:** `src/optimizer/instrument_dump.rs` (gated), exported from
  `optimizer/mod.rs` under `#[cfg(feature = "tee-debug")]`.
- **Activation:** the rule reads a per-statement request (a `DumpRequest`:
  `dump_dir`, node-selection predicate, `row_cap`). **No request ⇒ return the
  plan unchanged** (this is the runtime gate). Plumb the request via a thread/
  task-local or a field on the rule set per statement (see Phase 3 for how the
  CLI sets it).
- **Transform:** pre-order walk assigning each node a **stable id**; for every
  node that passes the selection predicate, wrap it: `node → TeeExec(node)` (via
  `with_new_children` on the parent). Leave `ZarrExec` and everything else
  otherwise untouched.
- **Manifest:** after the walk, write `<dump_dir>/manifest.txt` containing the
  `EXPLAIN ANALYZE` text (Phase 0), the `node-id → file` map, and the
  **physical→logical op mapping** (`FilterExec→Filter`, `AggregateExec
  mode=Partial/Final→Aggregate`, `HashJoinExec→Join`, `ZarrExec→TableScan`).
- **Acceptance:** with the rule active over the ONI query, a single run produces
  one Parquet dir per selected node + a manifest; with no request, `EXPLAIN`
  output is identical to the rule being absent (golden diff).

### Phase 3 — CLI surface + gating wiring (gated)

Make it usable from `zarr-cli`, and this is where the runtime gate is actually
set.

- **Cargo:** add `tee-debug = ["dep:...none..."]` feature (no new dep; it just
  toggles `cfg`). Register the rule in `main.rs` **only** under
  `#[cfg(feature = "tee-debug")]`, appended **after** `CardinalityRule` so it runs
  last:
  ```rust
  let builder = /* existing rules … */;
  #[cfg(feature = "tee-debug")]
  let builder = builder.with_physical_optimizer_rule(Arc::new(InstrumentDumpRule::new()));
  ```
- **Flag:** extend `parse_cli_args` (`src/bin/zarr_cli/main.rs`) with
  `--dump-dir <PATH>` (also honor `ZARR_TEE_DUMP_DIR`). Under the feature it sets
  the `DumpRequest` for statements in that invocation; **without** the feature the
  flag errors with "rebuild with `--features tee-debug`" so behavior is explicit.
- **REPL:** a `\dump <SQL>` meta-command that runs exactly one statement with the
  request set, then clears it.
- **Selection + safety:** `--dump-nodes all|<OpType>|/regex/` (default `all`) and
  `--dump-row-cap <N>` (**default 10_000** — see Data-volume gotcha; `full` opt-in
  via `--dump-row-cap 0`).
- **Acceptance:** `cargo build` (no feature) unaffected and `--dump-dir` errors
  cleanly; `cargo run --features tee-debug -- --dump-dir /tmp/d -c "…"` writes the
  tree; REPL `\dump` works and does not leak the request into the next statement.

### Phase 4 — inspect ergonomics (gated)

Read dumps without opening Parquet by hand.

- **Deliverable:** `zarr-cli --inspect <dump_dir>` prints, per node in id order:
  the logical op it lowered from, the Arrow schema, row count, and `head(N)`
  (default 10). Reads `manifest.txt` for the mapping + the per-node Parquet for
  the preview.
- **Files:** small reader module (gated) + `--inspect` branch in `main.rs`.
- **Acceptance:** `--inspect` on a Phase-3 dump reproduces the manifest mapping
  and shows previews; partial-aggregate nodes are labeled `(partial state)` so the
  weird intermediate schema isn't mistaken for the final result.

### Phase 5 — docs + cookbook (ungated docs, gated example)

- Update this file's status to **implemented (gated)**, add a "How to use"
  snippet, and add one worked example over `sql/oni_djf2025_extract.sql`.
- Add a gated example `examples/tee_dump_oni.rs`
  (`required-features = ["tee-debug"]`) mirroring the icechunk example entries.
- Save a project memory note that the tool exists and is feature-gated.

### Suggested first slice (highest value / lowest effort)

**Phase 0 + a stripped Phase 1/2/3 that only tees the leaf `ZarrExec` and the top
node**, activated by `--dump-dir` under `--features tee-debug`. That already lets
us compare "what came off disk" vs "final result" with a row cap — ~80% of the
debugging value — while proving the feature gate, the identity property, and the
Parquet round-trip. Expand `--dump-nodes` to all-nodes and add `--inspect`
(Phase 4) once the tee is proven.

## Gotchas to decide up front

- **Data volume.** Dumping every node of the ONI query writes the scan output
  (~200 MB) plus every intermediate. A **row cap per node** (e.g. first 10k rows)
  should be the default; "full" is opt-in.
- **Partial-aggregate schemas look weird.** `AggregateExec mode=Partial` emits
  *intermediate state* (e.g. avg = `count` + `sum`), not the final average. The
  manifest should label partial vs final so we're not confused.
- **Insert the rule LAST.** If tees go in before other physical rules, they can
  change adjacency-based decisions (repartition/coalesce). Running last keeps the
  plan we're debugging identical to the real one.
- **Dictionary-encoded coords.** This crate's coord columns are `DictionaryArray`;
  make sure the Parquet writer handles them (it does, but worth a test) so dumped
  files round-trip.

## Suggested first slice

Highest-value, lowest-effort: **Phase 0 + a stripped Phase 1/2 that only tees the
leaf `ZarrExec` and the top node.** That already lets us compare "what came off
disk" vs "final result" with a row cap — ~80% of the debugging value for a
fraction of the work. Expand to all-nodes once the tee is proven.
