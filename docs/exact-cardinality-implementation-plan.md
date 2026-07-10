# Exact-Cardinality Optimizer — Phase-wise Implementation Plan

> Companion to [`exact-cardinality-optimizer.md`](./exact-cardinality-optimizer.md)
> (design + §9 module sketch) and [`isl-theory-primer.md`](./isl-theory-primer.md)
> (Tier-B theory). This plan is re-anchored to the **current** codebase, in which
> the streaming actuator (design-note Block 0) has **already landed**
> (`zarr_reader.rs` windowing, `WindowedScan`).

## Current state (what exists today)

- **Streaming scan — DONE.** `read_zarr` windows the outer effective coordinate:
  `inner_rows = product(query_coord_sizes[1..])`,
  `max_steps = (batch_size / inner_rows).max(1)`, one batch per window
  (`zarr_reader.rs:1360–1372`, `WindowedScan` at `:1412`). Guarantee: peak ≈
  **max(batch_size, inner_rows) rows**, tiling **only the outer axis**, sized by
  DataFusion's row-count `batch_size`.
- **Exact cardinality already computed, but only reactively.**
  `calculate_filtered_rows = selections.iter().map(len).product()`
  (`filter.rs:1534`) runs inside the read path to size allocations — never as a
  planning oracle.
- **Filters + selections exist.** `CoordFilters` (`filter.rs:173`),
  `CoordFilterKind::{Eq, Range, DatePart, InList, DatePartSet}` (`:36`),
  `CoordSelection::{Range, Indices}` (`:109`) with `intersect`/`len`.
- **Existing metadata-only rules** to emulate: `MinMaxStatisticsRule`,
  `CountStatisticsRule` (`src/optimizer/`).
- **Partition plumbing exists:** `PartitionSpec` / `with_partitions`
  (`physical_plan/zarr_exec.rs`).

### The three residual gaps streaming does NOT close (this plan's targets)

1. **`inner_rows` floor.** Windowing tiles only axis 0, so peak is bounded by
   `inner_rows`, not the budget. A 4D reduce-over-time query has
   `inner_rows = level×lat×lon` and cannot be tiled below one outer step.
2. **Mixed-dimensionality fallback.** When a projected var doesn't span the full
   effective coord set (`all_full_cube == false`, `:1351`), windowing opts out →
   single batch of `final_rows` (`:1376–1401`) → the OOM class returns for that
   shape.
3. **No pre-flight / bytes-not-rows.** Streaming bounds row *count* to
   `batch_size`, not *bytes* to a memory budget; nothing predicts footprint or
   rejects a query before it runs.

## Guiding principles

- **Observe before act.** Compute exact cardinality read-only and verify it against
  reality before any plan changes behavior.
- **Additive, reusing existing artifacts.** The module *consumes* `CoordFilters` /
  `CoordSelection` / `ZarrArrayMeta` and *emits* decisions; the read path is
  unchanged until Phase 4.
- **Tier A (pure Rust) is the whole everyday path.** isl/`barvinok` (Tier B) is
  gated, optional, and last.
- **Each phase compiles, is testable in isolation, and ideally ships value alone.**

## Dependency graph

```
[Phase 0 DONE: streaming]
        │
Phase 1 core ─► Phase 2 lowering(observe) ─► Phase 3 cost+admission(advisory)
        │                                              │
        │                                              ▼
        │                                     Phase 4 drive streaming ──► Phase 5 rule ──► Phase 6 partitions
        │                                     (closes gaps 1 & 2)              │
        └──────────────────────────────────────────────────────────► Phase 7 agg pushdown
                                                                               │
                                                       Phase 8 Tier-B isl (optional, gated)
```

### A cross-cutting helper needed early: `row_width`

Several phases need bytes-per-row. Compute from the **projected** Arrow schema:
sum of dtype byte widths, with coordinate columns counted at their
`DictionaryArray<Int16Type>` key width (2 bytes), not the value width. Put this in
`cost.rs` as `fn row_width(projected_schema: &SchemaRef) -> usize` and unit-test it
against a couple of known projections.

---

## Phase 1 — The pure core: `CubeShape` + `IndexSet` + Tier-A backend

**Objective.** A dependency-free integer-set abstraction with a pure-Rust backend.
No DataFusion, no FFI.

**Work.**
- New module `src/optimizer/cardinality/` (`mod.rs`, `axis.rs`, `index_set.rs`,
  `backend/product.rs`).
- `axis.rs`: `Axis { name, extent, chunk }`, `CubeShape { axes: Vec<Axis> }`.
- `index_set.rs`: the `IndexSet` trait — `ndim`, `is_empty`, `cardinality() -> u128`,
  `intersect`, `union`, `project_out(dim)`, `apply(&AffineMap)` (may be
  `unimplemented!` in Tier A / return a conservative bound), `touched_tiles(&[u64]) -> u128`.
- `backend/product.rs`: Tier-A representation = a per-axis selection that is a
  union of intervals + strided cosets (mirrors `CoordSelection` but richer).
  Implement:
  - `cardinality` — per-axis size via interval length / coset count, multiplied
    across independent axes; unions via inclusion–exclusion.
  - `touched_tiles(tile)` — per axis, distinct `floor(x/chunk)` over the selection
    (for `Range(s,e)`: `floor((e-1)/c) - floor(s/c) + 1`; for scattered indices:
    distinct chunk ids), multiplied across axes.
  - `project_out` — drop an axis (its factor becomes 1).

**What we achieve.** A verified, exact counter for boxes, strides/cosets, and
unions — the axis-aligned and periodic cases that cover essentially every real
query. Reusable, zero-dependency, fully unit-testable.

**How to test.**
- **Property test vs brute force:** for random boxes, strided selections, and
  unions in ≤3 dims with small extents, assert `cardinality()` equals explicit
  `HashSet` enumeration of the integer points.
- **`touched_tiles` property test:** enumerate points, map each through
  `floor(x/chunk)`, count distinct tiles, assert equality.
- **Algebra laws:** `intersect` with universe is identity; `project_out` reduces
  `ndim` by 1 and multiplies out the dropped factor; empty ∩ anything is empty.

---

## Phase 2 — Lowering + observation: `CoordFilters` → `IndexSet` (read-only)

**Objective.** Bridge the scan's existing pushed-down filters into an `IndexSet`
and *observe* exact cardinality on live queries — changing nothing.

**Work.**
- `predicate.rs`: `selection_from_filters(shape, filters, coords) -> Option<BoxedIndexSet>`.
  Reuse the existing lowering: `CoordFilterKind::Range`/`Eq` → interval;
  `DatePart`/`DatePartSet` → strided coset(s); `InList` → union of points. This
  parallels how `filter.rs` already turns filters into `CoordSelection`; lift that
  into index-set form. Return `None` when provably empty.
- `axis.rs`: build `CubeShape` from `ZarrArrayMeta` / `schema_inference` (extents
  and chunk sizes per coordinate).
- Add a **read-only diagnostic path**: behind a flag / `EXPLAIN`-style hook, print
  `rows = sel.cardinality()`, `inner_rows` (product of non-outer axes),
  `touched_chunks = sel.touched_tiles(chunk_shape)`, and whether the query hits the
  **mixed-dim single-batch fallback** (`all_full_cube == false`).

**What we achieve.** The engine can now **state exact cardinality, inner_rows, and
touched-chunk counts for real queries at plan time, at zero behavior risk** — and
flag which queries will single-batch (Gap 2). This is the first tangible artifact.

**How to test.**
- **Golden cardinalities:** pin `rows` / `touched_chunks` for a set of
  synthetic + ERA5/ONI selections.
- **Cross-check against reality (the key test):** assert
  `sel.cardinality()` **equals the total row count the streaming scan actually
  produces** (sum over all emitted windows) for the same query. This proves the
  oracle matches the executor.
- **Gap-2 detection:** assert the mixed-dimensionality query (a static field
  selected next to a time-varying one) is correctly identified as single-batch.
- **Empty selection:** a contradictory filter lowers to `None`.

*Checkpoint: exact cardinality is observable and verified on live queries.*

---

## Phase 3 — Cost model + advisory admission

**Objective.** Turn exact counts into a deterministic cost + a *warning-level*
feasibility verdict. Still no plan changes.

**Work.**
- `cost.rs`: `ScanCost { rows, touched_chunks, bytes_read, peak_bytes }`;
  `estimate_scan_cost(sel, shape, row_width)`.
  - `bytes_read = touched_chunks × chunk_bytes` (chunk_bytes from dtype × chunk
    volume).
  - `peak_bytes = max_window_rows × row_width`, where `max_window_rows =
    max(batch_size, inner_rows)` for the streaming path, or `final_rows` for the
    Gap-2 single-batch fallback.
  - `row_width` from the cross-cutting helper.
- `budget.rs`: `MemoryBudget { bytes }`; `admit(cost, budget) -> Result<(), Infeasible>`
  returning the **exact predicted footprint** in the error/warning.
- Wire `admit` as a **log warning** (or an error behind an off-by-default config
  flag) inside `ZarrTable::scan()`.

**What we achieve.** Shippable diagnostics: the Gap-1 query reports *"single window
≈ 760 MB"*, the Gap-2 query reports *"single-batch fallback ≈ 120 GB"*, **before
execution** — instead of discovering it by running (or dying).

**How to test.**
- **Golden costs:** pin `bytes_read` / `peak_bytes` for the benchmark selections.
- **`row_width` unit tests** across projections (including dictionary-encoded
  coords at 2 B keys).
- **Admission asserts:** a query exceeding a small budget yields `Infeasible` with
  a footprint within a tolerance of the measured peak; a small query is admitted.

*Checkpoint: shippable pre-flight diagnostics.*

---

## Phase 4 — First behavior change: drive streaming from cost (closes Gaps 1 & 2)

**Objective.** Replace the fixed row-count window with a budget-derived one, and
handle the two shapes streaming currently mishandles.

**Work.**
- `budget.rs`: `plan_streaming(sel, shape, row_width, budget) -> BatchPlan`.
  - `budget_rows = budget.bytes / row_width`.
  - **Gap 1 (inner_rows floor):** if `inner_rows ≤ budget_rows`, behave as today
    (outer-axis window of `budget_rows / inner_rows` steps). If
    `inner_rows > budget_rows`, escalate to **multi-axis tiling** — tile the next
    inner axis too, so a single outer step no longer forces `inner_rows` resident.
    `BatchPlan` grows a per-axis tile vector.
  - **Gap 2 (mixed-dim fallback):** when `all_full_cube == false`, either warn +
    proceed, or (config) reject — using the exact `final_rows × row_width`.
- Surgically edit the window computation at `zarr_reader.rs:1364` to consume the
  `BatchPlan` instead of `max_steps = batch_size / inner_rows`. The `WindowedScan`
  plumbing already exists; this feeds it a budget-derived tile schedule.

**What we achieve.** Peak memory is bounded by an actual **byte budget**, not a
row-count proxy; the 4D reduce-over-time query (Gap 1) tiles a second axis and
stays bounded; the mixed-dim query (Gap 2) no longer silently OOMs. This is the
first place cardinality *changes execution* — and the real prize.

**How to test.**
- **OOM proof, Gap 1:** the 4D reduce-over-time query runs with peak ≤ budget
  (extend the existing OOM-proof test from commit `bea8084`).
- **Regression:** the day-15 sample and the small synthetic queries are byte-
  identical / unaffected.
- **Multi-axis tiling correctness:** the concatenation of all emitted windows
  equals the un-tiled result (row-set equality), for a query where
  `inner_rows > budget_rows`.
- **Gap 2:** mixed-dim query either streams (if made windowable) or is rejected
  with the exact footprint — no silent single-batch OOM.

---

## Phase 5 — Formalize as a `PhysicalOptimizerRule`

**Objective.** Lift the Phase 2–4 calls out of `ZarrTable::scan()` into a rule, so
it gains plan context (e.g. the aggregate sitting above the scan). Refactor, not
new math.

**Work.**
- `rule.rs`: `CardinalityRule { budget }` implementing `PhysicalOptimizerRule`;
  walk the plan, find each `ZarrExec`, compute `sel` / `cost`, run `admit`, set the
  batch plan via a `ZarrExec::set_batch_plan` knob. Register on the session
  alongside `MinMaxStatisticsRule` / `CountStatisticsRule`.

**What we achieve.** A single, discoverable optimizer entry point; the rule can now
see parent operators (prerequisite for Phase 7). Low risk — the math is proven in
Phases 1–4.

**How to test.**
- **Optimizer regression** (extend `integration_optimizer.rs`): the rule annotates
  `ZarrExec` and produces the same batch plans Phase 4 computed inline.
- **Idempotence:** running the rule twice yields the same plan.
- **No-op safety:** plans without a `ZarrExec` pass through untouched.

---

## Phase 6 — Partition count from `touched_chunks`

**Objective.** Feed the existing parallelism path a cardinality-derived partition
count (orthogonal to memory).

**Work.**
- In `rule.rs`, `n = partitions_from(cost.touched_chunks, target)`; call the
  existing `with_partitions` / `PartitionSpec` path (`zarr_exec.rs`).

**What we achieve.** Parallelism sized to real work (touched chunks), not a fixed
default — deterministically.

**How to test.**
- **Golden partition counts** for benchmark selections.
- **Correctness under partitioning:** result row-set identical to single-partition;
  partition boundaries align to chunk boundaries.
- **Bounds:** never more partitions than touched chunks; never zero.

---

## Phase 7 — Group cardinality + partial-aggregate pushdown

**Objective.** The biggest new executor capability — last in Tier A. Use exact
group counts to push `SUM`/`COUNT` into the chunk reader.

**Work.**
- `budget.rs`: `GroupKey { Axis(usize) | Periodic { axis, field } }`;
  `group_cardinality(sel, shape, keys)` = cardinality of `sel` projected onto the
  key images (uses `project_out` / `apply`).
- `rule.rs`: when a parent `AggregateExec` groups on a coordinate (or a function of
  one, e.g. month-of-time) and `group_cardinality ≤ budget.groups()`, call a new
  `ZarrExec::enable_partial_aggregate(keys)` hook that computes partial aggregates
  per chunk and combines.

**What we achieve.** `AVG`/`SUM`/`COUNT` over a coordinate axis no longer
materializes the flattened selection — a genuinely new physical operator, chosen
deterministically because the group count is known to fit.

**How to test.**
- **Group-count goldens:** exact group counts for coordinate and periodic keys.
- **End-to-end equivalence:** pushed-down aggregate result equals the non-pushed
  result (bit-exact for `SUM`/`COUNT`, within fp tolerance for `AVG`).
- **Viability gate:** a high-cardinality group-by (exceeds `budget.groups()`) is
  *not* pushed down and falls back cleanly.

*Checkpoint: full Tier-A optimizer — exact costs and fast aggregates.*

---

## Phase 8 — Tier B: `isl` / `barvinok` behind the `polyhedral` feature

**Objective.** Add the FFI backend for genuinely coupled predicates (joins relating
two axes, diagonals, regrids) once Tier A is trusted. Deliberately last; must never
gate the core.

**Work.**
- `backend/isl.rs`: implement `IndexSet` over `isl` sets; `cardinality` via
  `isl_set_count_val` for bounded sets (or `isl_set_card` → evaluate the piecewise
  quasi-polynomial at the known extents for parametric families). `apply` becomes
  real (affine maps via `isl_map`).
- Gate behind cargo feature `polyhedral` (off by default). Without it, a coupled
  predicate falls back to a conservative upper bound (product of axis extents) and
  the optimizer skips the exact-count-dependent decision rather than failing.
- **FFI/linking de-risked (spike done, glibc):** a one-function C shim + `build.rs`
  (`static=barvinok,isl` + `dylib=gmp,ntl,stdc++`) links and counts a coupled band
  exactly from Rust. Prefer this shim over `isl-rs` (dodges isl-version skew).
  **Corrections from the spike:** modern barvinok **requires NTL** (C++) — there is
  no `--without-ntl`; and **static-musl (the CLI release target) is not viable as-is**
  (no musl toolchain + glibc-built chain; NTL/C++ is the blocker) → ship `polyhedral`
  glibc-only, or drop NTL via an older barvinok first. Full recipe + verdicts:
  [`isl-theory-primer.md`](isl-theory-primer.md).

**What we achieve.** Exact cardinality for the non-separable cases where
product-of-extents is meaningless — coupled/diagonal/join predicates — without any
cost to the default build.

**How to test.**
- **Tier A vs Tier B agreement (the licensing test):** on separable sets,
  `product` and `barvinok` must agree **exactly** — this cross-check is what
  justifies trusting Tier B where Tier A can't reach.
- **Coupled goldens:** a diagonal predicate (`lat_index + lon_index ≤ K`) counts to
  the exact triangular-region count, not the box product.
- **Feature-off safety:** default build compiles and runs with the conservative
  fallback; coupled queries still execute (just without the exact-count decision).

**Motivating queries.** Five recurring, *affine*-coupled climate queries justify the
backend (full write-up, per-query shape and expected q-error in
[`exact-cardinality-tier-b-use-cases.md`](exact-cardinality-tier-b-use-cases.md)):
1. **Forecast verification** — `valid = init + lead`, windowed → diagonal band in the
   (init, lead) plane (the freeze / forecast-vs-ERA5 cookbook).
2. **ONI 3-month running seasons** — `|member − center| ≤ 1` → band with month wrap
   (the el-niño ONI cookbook).
3. **Heatwave / spell detection** — `a.time ≤ b.time ≤ a.time + N` → band self-join.
4. **Coarsening / regrid** — `coarse = ⌊fine / k⌋` → floor-division map.
5. **Cumulative accumulation** (GDD, cumulative precip) — `b.time ≤ a.time` → triangle
   (windowed variant → band).
The band and floor-map cases (1, 3, 4, 5-windowed) have **large** q-error vs the naïve
product-of-extents and are the ones worth building for; the triangle (5-full) and the
3-wide band (2) are only ~2–4× and likely change no decision.

**Measuring the improvement.** Three levels, measured in order:
- **Level 1 — q-error** (`max(est/truth, truth/est)`) against a brute-force oracle:
  the plan-independent headline, and the framing the thesis wants (computed → q-error
  ≈ 1). Needs no DataFusion plumbing.
- **Level 2 — plan sensitivity:** `EXPLAIN VERBOSE` tree-diff (join order, join
  algorithm, partition count, `ZarrAggregateExec` presence) — the estimate only matters
  where it *crosses a decision threshold*.
- **Level 3 — runtime:** `EXPLAIN ANALYZE` (time, peak memory, bytes, spills), only for
  queries whose plan flipped.
- **Plumbing caveat:** pushdown-admission and partition fan-out already consume *our*
  cardinality (a Tier-B estimate flips them immediately, visible in `EXPLAIN`); but
  **join order/algorithm** need the exact count *published into DataFusion's
  `Statistics`* first — until that small bridge exists, the join plan is byte-identical
  and only Level 1 differs.

**Effort deconstruction.** W1 — `barvinok → isl → GMP (+ NTL, C++)` build/FFI —
**glibc de-risked** (spike: builds, links, counts a coupled band from Rust); the
remaining W1 risk is **static-musl** (not viable as-is — see the primer verdicts). W2 — `backend/isl.rs` (M;
`touched_tiles` under floor-division is the fiddly method). W3 — *the consumer gap*
(M–L): nothing produces coupled predicates today (filters are `coord op value`) and the
reader can't read non-box selections, so a value-delivering Phase 8 also needs a
coupled-predicate surface + reader fallback, or the "publish into `Statistics`" bridge
for joins. W4 — fallback/gating/docs (S–M; mostly *already there* via `apply → None`).
W5 — testing (M; needs the C libs in CI). **Current value ≈ 0** (no live query needs
it); this is the research tail — build a bounded `iscc` spike (licensing test + one
coupled golden) before committing to the FFI/consumer work.

---

## Where to stop

- **After Phase 2:** exact cardinality is observable and verified — zero risk.
- **After Phase 3:** shippable pre-flight diagnostics (footprint before running).
- **After Phase 4:** cardinality-driven streaming that closes Gaps 1 & 2 — the real
  prize; ~80% of practical value at near-zero research risk.
- **After Phase 7:** full Tier-A optimizer (exact costs + fast aggregates).
- **Phase 8** is optional and gated; everything genuinely novel (lattice counting,
  Ehrhart) lives there and must never block the core.
