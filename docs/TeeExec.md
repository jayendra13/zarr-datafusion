# Query Debugging via Intermediate-Output Dumps (`TeeExec`)

> Status: **design / parked** — not yet implemented. Captured here so we can come
> back to it. See "Recommended design" for the intended approach.

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

## High-level plan (phased)

- **Phase 0 — free wins first.** Wire up `EXPLAIN ANALYZE` (DataFusion already
  gives per-operator row counts + timing via `MetricsSet`). That alone answers
  "what ran, how many rows, how long" without dumping any data — and it's the
  backbone of the manifest. Do this before writing any Parquet.
- **Phase 1 — `TeeExec`.** Pass-through wrapper writing batches to Parquet via
  `ArrowWriter`/`ParquetSink`, one file per `(node, partition)`. Unit-test that
  output is byte-identical with/without the wrapper.
- **Phase 2 — `InstrumentDumpRule`.** Insert tees, assign ids, emit the manifest.
  Register it like the other rules in `main.rs`, gated so it's off by default.
- **Phase 3 — CLI surface.** A `--dump-dir <path>` flag (or `\dump <SQL>`
  meta-command) that installs the rule for that one statement. Add **node
  selection** (`all` / by operator type / regex) and a **per-node row cap** so we
  don't write hundreds of MB per node.
- **Phase 4 — inspect ergonomics.** `zarr-cli --inspect debug_dump/` that prints,
  per node, the logical op it came from, schema, and `head(N)` — so we read
  results without opening Parquet by hand.

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
