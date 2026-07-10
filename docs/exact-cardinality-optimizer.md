# Exact-Cardinality Optimization for Cube Queries

> **Status: design note / research direction, not implemented.** This document
> captures a line of thinking about where the engine's optimizer could go. It
> starts from a concrete failure (a query that OOMs), extracts the generic fix,
> and then follows the idea to its conclusion: because gridded data has a known,
> regular structure, *cardinality is computed exactly rather than estimated*, and
> that changes what kind of optimizer is possible.

> **Cross-cutting theme.** A dense rectilinear cube is a function over a
> coordinate lattice. We always know the extents (how many timestamps, how many
> latitudes, …), so the *shape* of any intermediate result is knowable in closed
> form before reading a byte. The relational world's hardest problem —
> cardinality estimation — is, for the coordinate-structured part of a query,
> not an estimate at all.

---

## 1. The triggering problem: unbounded single-batch materialization

The ONI cookbook query (`cookbook/el-nino-oni/oni_all_seasons.sql`) samples one
timestep per month (`day=15 AND hour=12`) as a cheap proxy for a monthly mean.
Replacing the sample with a *true* monthly mean — averaging every hourly
timestep — OOMs. The reason is architectural, not arithmetic:

1. **The day/hour predicates are the only thing shrinking the time axis.** They
   parse to `CoordFilterKind::DatePart` filters on `time`
   (`src/reader/filter.rs`) and collapse the axis to ~1 step/month (~1000 over 87
   years). Dropping them to average the whole month restores the full hourly axis
   — ~760,000 steps, a ~760× explosion.

2. **The scan materializes the entire flattened selection as one in-memory
   RecordBatch.** `read_zarr` builds every column to length `final_rows`
   (= product of per-coordinate selection sizes) and emits a single-element
   stream (`src/reader/zarr_reader.rs`, around the
   `stream::iter(vec![Ok(batch)])` emission). The `AVG`/`GROUP BY` runs *on top
   of* that fully resident batch, so the ~6.2-billion-row Cartesian product
   (~760k timesteps × ~8,200 Niño-box cells) must all live in memory at once.

The OOM is generic: any query whose **selection** is large will fail, regardless
of how small the **result** is.

## 2. The immediate generic fix: stream bounded batches

The single-batch emission violates the DataFusion `ExecutionPlan` contract. Every
well-behaved source (Parquet, CSV) yields a *sequence* of batches sized to
`session.config().batch_size()` (default 8192). The fix is to make the scan a
true streaming operator: **iterate the outer-coordinate selection in windows and
yield one batch per window.**

Why this is the right generic solution and not a band-aid:

- **It decouples scan memory from selection size.** Peak memory becomes
  `O(batch_size + downstream operator state)`. The monthly mean then works
  untouched — `AVG`/`GROUP BY` is already a streaming consumer and its state
  (~1000 month groups) is tiny; the 6-billion-row selection never has to be
  resident.
- **It fixes the whole class** — large `SELECT *`, `ORDER BY … LIMIT`, windows,
  joins — at the source.
- **It is the idiomatic DataFusion contract**, so it composes with the optimizer,
  spill-to-disk, and backpressure for free.

It maps cleanly onto the existing architecture: the Cartesian flattening is
outer-axis-most-significant, so an outer-axis window is a *row-contiguous* batch;
`build_read_plans` already buckets data-var reads per chunk; the
`RecordBatchStreamAdapter` is already multi-batch capable (it is just being fed a
one-element iterator today); and `execute()` already has the `TaskContext` to
read `batch_size`.

This is **orthogonal** to two things already in the codebase:

- **`PartitionSpec` partitioning** (`src/physical_plan/zarr_exec.rs`,
  `with_partitions`) is about *parallelism*, not memory — partitioning a 6B-row
  scan still OOMs if each partition single-batches. Streaming is the
  prerequisite; partitioning sits on top.
- **`MinMaxStatisticsRule` / `CountStatisticsRule`** and a future partial-aggregate
  pushdown make specific shapes *fast*; streaming makes *every* shape survive
  memory. Pushdown is the cherry, not the foundation.

## 3. The bigger idea: cardinality is computed, not estimated

A relational optimizer's hardest job — and its dominant source of bad plans — is
cardinality estimation. Leis et al., *How Good Are Query Optimizers, Really?*
(VLDB 2015), showed that estimation error, compounding multiplicatively through
joins, is *the* reason plans go wrong; the cost model and plan search are
secondary. Histograms, sampling, and independence assumptions all carry error
that explodes through a plan.

For a dense cube, that problem largely evaporates. Coordinates are sorted 1D
arrays, so a predicate on a dimension resolves to an **exact** surviving index
count via binary search; independent axes multiply *exactly* (it is structurally
true, not an independence *assumption*). The engine already computes this:
`calculate_filtered_rows` = `selections.iter().map(len).product()`
(`src/reader/filter.rs`) is, literally, an exact cardinality propagator — it is
just used reactively at scan time, not as a planning-time cost oracle.

**The proposal is to lift that into a logical cost model that runs before
physical planning.** The part that is hard relationally (cardinality) becomes
free; the residual research questions move to where exactness *breaks* and to how
exact shapes propagate through a richer algebra.

## 4. What an exact-cardinality optimizer enables

- **Deterministic physical planning.** Streaming granularity, partition count,
  spill-or-not, join order — every decision a relational planner gambles on
  becomes a closed-form computation over known extents. The cost model is
  `bytes_read = touched_chunks × chunk_bytes`, `peak_mem = max_materialized_shape
  × dtype_width`, `compute = product of reduction extents`. No statistics
  objects, no `ANALYZE`.
- **Admission control / feasibility pre-flight.** The direct answer to §1: know
  the exact peak footprint *before* executing and pick batch size, partition
  count, or reject/warn — a priori, not after dying.
- **Group-cardinality-driven aggregation pushdown.** A `GROUP BY` on a coordinate
  (or a function of one, like month-of-time) has an exactly known group count, so
  you know the aggregation state fits and can choose partition-local-then-combine
  deterministically.
- **Axis-reduction recognition.** `AVG` over a full dimension *is* a tensor
  reduction along an axis. Reasoning at the array level lets the optimizer pick a
  strided reduction operator that never materializes the flattened table — a
  rewrite invisible to a planner staring at a 2D scan.

The natural architecture is two levels: a **logical array-algebra IR** above the
flatten layer where every operator carries an exact shape (per-axis extent +
selection set) — "cardinality estimation" becomes "shape inference," exact and
compositional — feeding a **structural cost model** that drives ordinary
Selinger-style search with a *perfect* cardinality oracle in place of a
histogram.

## 5. Where exactness breaks (the real frontier)

Naming the boundary precisely matters, because that is where the work is:

- **Predicates on data *values*, not coordinates.** `WHERE temperature > 300` is
  data-dependent — back to estimation. The honest design is a **hybrid**: exact
  structural reasoning for the coordinate-selection part, and chunk-level min/max
  **zone maps** (exact pruning, not estimates) plus classical estimation for the
  value-dependent residual. This is the same boundary the engine already
  acknowledges as an open gap (data-variable filters are post-scan).
- **Sparsity / missing chunks.** The product model assumes density; the ONI query
  already hits this (absent SST chunks read back as NaN). Real cubes need
  per-chunk occupancy metadata, and cardinality becomes "product minus the
  holes."
- **Non-axis-aligned predicates.** A diagonal selection, or a join condition
  coupling two axes, is not a per-axis product. The clean generalization is in
  §6.

## 6. The polyhedral / compiler connection

The polyhedral model is a compiler framework for representing and transforming
loop nests, and it is almost exactly the right algebra for cube queries.

**What it is.** To optimize a loop nest, a compiler represents it geometrically:
the **iteration domain** is the set of index tuples the loops visit
(`for t,y,x` → `{ (t,y,x) : 0≤t<T, 0≤y<Y, 0≤x<X }`, the integer points in a
parametric polytope); **access functions** are affine maps from iteration points
to array elements; a **schedule** is an affine map giving each point an execution
order, and optimizations (tiling, interchange, skewing, fusion) are schedule
transformations that preserve dependences. This is the machinery under LLVM's
Polly, Pluto, PPCG, and `isl` (the Integer Set Library).

**The isomorphism.** A query selection over a cube *is* an iteration domain. Once
coordinate predicates are mapped to index space,
`WHERE lat BETWEEN −5 AND 5 AND lon BETWEEN 190 AND 240` becomes
`{ (t,y,x) : y_lo≤y≤y_hi, x_lo≤x≤x_hi, 0≤t<T }` — an integer polytope. **The
number of rows the query touches = the number of integer points in that
polytope = its cardinality.** Counting them is a solved problem:

- **Ehrhart polynomials** — the count is a quasi-polynomial in the size
  parameters, so you get a *symbolic* cardinality formula valid before the exact
  extents are known.
- **Barvinok's algorithm** — counts lattice points in polynomial time for fixed
  dimension; the `barvinok` library (built on `isl`) does exactly this.

The "trivial product of extents" of §3 is just the **axis-aligned box** special
case. Polyhedral counting is the general version, and it does not break when the
predicate is not a clean box:

- **Coupled axes / joins** — a join relating two index spaces (`i = j`, or any
  affine relation) is a polytope spanning both; product cardinality is
  meaningless, lattice counting is exact, and composition of affine relations
  stays exact through a chain.
- **Periodic / strided predicates** — `day = 15`, `EXTRACT(month) = 12`,
  `hour = 12` are not convex but are quasi-affine (affine + modulo). `isl` works
  over Presburger sets, so a periodic selection is a union of lattice cosets,
  still counted exactly. The ONI `day=15 AND hour=12` is precisely such a set.
- **Strides / regridding** (every Nth point) are affine lattices — native.

**The two reuses that are the real payoff:**

1. **Chunk-touch = tiling intersection.** Zarr chunking *is* polyhedral tiling.
   "Which chunks does this query read?" = "which tiles does the iteration domain
   intersect?", computable exactly → exact I/O cost and read-coalescing plans,
   symbolically, before reading a byte. `build_read_plans` is a hand-rolled,
   special-case version of this.
2. **Streaming = tiling for a memory budget.** The §2 "stream the outer axis in
   `batch_size` windows" idea is, formally, a *tiling of the scan's iteration
   domain*. Compilers choose tile sizes to fit a cache; here you choose them to
   fit a memory budget — same math, different constraint. The optimizer could
   *derive* streaming granularity rather than hardcode it.

A bonus: the ONI centered 3-month running mean is a **stencil**, and stencils are
the canonical polyhedral workload (time-tiling, skewing). A rolling-window
operator over a cube is the textbook example, not a special case.

**The boundary is the same as §5:** the polyhedral model requires affine (or
quasi-affine) constraints over the *index* space. It reasons about coordinate
structure, not values, so `WHERE temperature > 300` falls outside. The saving
grace: cubes have few axes (3–5), and Barvinok is polynomial in fixed dimension,
so the counting is genuinely cheap here.

## 7. Prior art

- **Array DBMSs** — the most direct lineage. **SciDB** (Stonebraker et al.):
  chunked multidimensional arrays, a real array algebra (AQL/AFL), cost-based
  optimization over array ops. **rasdaman** (Peter Baumann): array algebra
  formalized, tile-based, basis of the OGC WCPS standard. **TileDB**: modern
  dense/sparse multidimensional storage with dimension-based slicing.
  **SciQL** (MonetDB): arrays as first-class in SQL.
- **OLAP / data-cube theory** — Gray et al.'s *Data Cube* operator (1997), and
  especially Harinarayan–Rajaraman–Ullman, *Implementing Data Cubes Efficiently*
  (1996), which is exactly about reasoning over the group-by lattice (the
  dimension powerset) when cardinalities are known — materialized-aggregate
  selection. MOLAP engines plan over dimension cardinalities and sparsity
  precisely.
- **Lazy chunked-array graphs** — Dask/Xarray build task graphs over chunked
  arrays with structural (not statistical) shape knowledge and do graph-level
  optimization (fusion, culling). Not cost-based, but close in spirit.
- **Polyhedral / compiler** — `isl`, Pluto, Polly, PPCG for transformation;
  `barvinok` (Verdoolaege) for lattice-point counting / Ehrhart polynomials.

**Why it looks underused.** The array-DB community built bespoke chunk/tile cost
reasoning but largely did not adopt the polyhedral lattice-counting machinery as
their cardinality oracle; the compiler community built that machinery for
register/cache optimization without pointing it at cloud-object-store query
planning. Wiring `isl`/`barvinok` in as the exact cardinality-and-cost engine for
a cube-query optimizer — with **tiling as the unifying primitive** for both chunk
I/O and memory-bounded streaming — is the cross-pollination that, as far as we
know, nobody has done in the cloud-native ARCO/Zarr setting this engine targets.

## 8. Synthesis and direction

The existing `MinMaxStatisticsRule` / `CountStatisticsRule` are point instances of
one principle: *answer from metadata without scanning*. The direction here is to
make that principle the optimizer's foundation rather than two special cases:

1. **Now (generic, low-risk):** convert the scan to stream `batch_size`-bounded
   batches along the outer axis (§2). Retires the OOM class for every query shape.
2. **Next:** a logical array-algebra layer whose shape inference is an exact
   cardinality oracle, feeding a deterministic structural cost model (§3–4), with
   admission control derived from exact peak-memory.
3. **Hybrid at the boundary:** zone maps + classical estimation exactly where a
   predicate touches data values rather than coordinates (§5).
4. **Research reach:** the lattice-point / Ehrhart treatment of non-axis-aligned
   and join predicates, and tiling-as-one-primitive for both I/O and streaming
   (§6).

---

## 9. Module sketch: `optimizer::cardinality`

A self-contained module, separate from the scan and the existing optimizer rules.
It *consumes* what the scan already produces (`CoordFilters`, `CoordSelection`,
`ZarrArrayMeta`) and *emits* decisions (batch size, partition count, pushdown
flags, an admission verdict). The read path is unchanged except that it honors
the chosen batch/window size and pushdown flags.

```
src/optimizer/cardinality/
├── mod.rs          # public API + re-exports
├── axis.rs         # Axis, CubeShape — built from schema_inference metadata
├── index_set.rs    # IndexSet trait: a Presburger set over the index space
├── backend/
│   ├── product.rs  # Tier A: pure-Rust boxes / strides / unions-of-boxes
│   └── isl.rs      # Tier B: FFI to isl + barvinok for coupled affine sets
├── predicate.rs    # lower CoordFilters -> IndexSet (reuses reader::filter)
├── cost.rs         # ScanCost + cost formulas
├── budget.rs       # MemoryBudget, BatchPlan, admission control
└── rule.rs         # DataFusion PhysicalOptimizerRule
```

### 9.1 Library choice — two tiers behind one trait

| Tier | Backend | Handles | Dep |
|------|---------|---------|-----|
| **A** | pure Rust (`product.rs`) | axis-aligned boxes, strided/periodic (date-part) selections, unions thereof via inclusion–exclusion | none |
| **B** | `isl` + `barvinok` via FFI (`isl.rs`) | genuinely coupled affine sets — joins relating two axes, diagonals, regrids | C libs, feature-gated |

Tier A covers essentially every query the engine sees today. Tier B is engaged
only when the set is non-separable: `isl_set_card` (barvinok) returns a piecewise
quasi-polynomial, evaluated at the known extents for the exact count. Both
backends implement the same `IndexSet` trait, so callers never branch on tier.

isl/barvinok lives behind a cargo feature `polyhedral` (off by default). Without
it, a coupled predicate falls back to a conservative upper bound (product of axis
extents) and the optimizer simply skips the exact-count-dependent decision rather
than failing. The default build stays dependency-free.

### 9.2 Core types and signatures

```rust
// axis.rs — the logical model, straight from ZarrArrayMeta
pub struct Axis  { pub name: String, pub extent: u64, pub chunk: u64 }
pub struct CubeShape { pub axes: Vec<Axis> }

// index_set.rs — the heart of the module
/// An integer (Presburger) set over a cube's index space.
pub trait IndexSet: Clone {
    fn ndim(&self) -> usize;
    fn is_empty(&self) -> bool;

    /// Exact number of integer points.
    /// Tier A: closed form (product / inclusion–exclusion). Tier B: barvinok.
    fn cardinality(&self) -> u128;

    fn intersect(&self, other: &Self) -> Self;   // AND of two predicates
    fn union(&self, other: &Self) -> Self;        // OR / multi-coset

    /// Existential projection — drop an axis (GROUP BY removes a reduced axis).
    fn project_out(&self, dim: usize) -> Self;

    /// Image under an affine map A·x + b — joins, regrids, periodic keys.
    fn apply(&self, map: &AffineMap) -> Self;

    /// Distinct tiles touched under floor-div by `tile`
    /// = cardinality of the image of  x -> floor(x / tile).  Drives I/O cost.
    fn touched_tiles(&self, tile: &[u64]) -> u128;
}

pub type BoxedIndexSet = Box<dyn IndexSetDyn>;   // object-safe wrapper over the trait
```

```rust
// predicate.rs — bridge from the existing scan filters
/// Lower the scan's coordinate filters into an index set over `shape`.
/// Reuses CoordSelection: Range -> interval, Indices -> union of points/cosets.
/// Returns None when the selection is provably empty.
pub fn selection_from_filters(
    shape: &CubeShape,
    filters: &CoordFilters,
    coords: &[CoordValuesRef<'_>],
) -> Option<BoxedIndexSet>;
```

```rust
// cost.rs — a deterministic cost model, no statistics objects
pub struct ScanCost {
    pub rows: u128,            // sel.cardinality()
    pub touched_chunks: u128,  // sel.touched_tiles(chunk_shape)
    pub bytes_read: u128,      // touched_chunks * chunk_bytes
    pub peak_bytes: u128,      // rows_in_largest_batch * row_width
}
pub fn estimate_scan_cost(
    sel: &BoxedIndexSet, shape: &CubeShape, row_width: usize,
) -> ScanCost;
```

```rust
// budget.rs — streaming granularity and admission
pub struct MemoryBudget { pub bytes: u64 }
pub struct BatchPlan { pub outer_axis: usize, pub rows_per_batch: u64, pub n_batches: u64 }

/// Choose the outer-axis tile size so each emitted batch fits the budget.
/// This is loop-tiling for a memory budget instead of a cache (see §6).
pub fn plan_streaming(
    sel: &BoxedIndexSet, shape: &CubeShape, row_width: usize, budget: &MemoryBudget,
) -> BatchPlan;

/// Pre-flight admission: reject (or justify) before a byte is read.
pub fn admit(cost: &ScanCost, budget: &MemoryBudget) -> Result<(), Infeasible>;

// group cardinality — viability of partial-aggregate pushdown
pub enum GroupKey { Axis(usize), Periodic { axis: usize, field: DateField } }
/// Exact group count = cardinality of `sel` projected onto the key images.
pub fn group_cardinality(
    sel: &BoxedIndexSet, shape: &CubeShape, keys: &[GroupKey],
) -> u128;
```

### 9.3 Algorithm catalog

| # | Algorithm | Purpose | Tier | Cost |
|---|-----------|---------|------|------|
| 1 | Sorted-coordinate binary search | value predicate → index interval | A | O(log n) — *already in `reader::filter`* |
| 2 | Per-axis product | box cardinality | A | O(d) |
| 3 | Inclusion–exclusion over a box union | periodic / date-part / `OR` | A | O(2ᵏ), k = #boxes (small) |
| 4 | Stride / coset counting | `day=15`, `hour=12` style | A | O(1) per coset |
| 5 | Barvinok lattice-point count | coupled / diagonal / join | B | poly in fixed dim |
| 6 | Ehrhart quasi-polynomial | symbolic count vs extent (partition-size formulas) | B | — |
| 7 | Tile-image counting | touched chunks → I/O cost | A (boxes) / B (general) | O(d) / poly |
| 8 | Outer-axis tiling for a budget | streaming granularity | A | O(1) |
| 9 | Projection counting | group cardinality | A (separable) / B (coupled) | O(d) / poly |

Algorithms 1–4, 7–8 are the everyday path and need no external library. 5–6 and
the coupled cases of 7, 9 are the `polyhedral`-feature path.

### 9.4 DataFusion integration

A single `PhysicalOptimizerRule` that walks the plan, finds each `ZarrExec`, and
annotates it:

```rust
pub struct CardinalityRule { budget: MemoryBudget }

impl PhysicalOptimizerRule for CardinalityRule {
    fn optimize(
        &self, plan: Arc<dyn ExecutionPlan>, _: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // for each ZarrExec node:
        //   sel  = selection_from_filters(shape, exec.filters(), coords)?;
        //   cost = estimate_scan_cost(&sel, &shape, row_width);
        //   admit(&cost, &self.budget)?;                       // exact footprint in the error
        //   let plan = plan_streaming(&sel, &shape, row_width, &self.budget);
        //   exec.set_batch_plan(plan);                          // the §2 streaming knob
        //   if let Some(agg) = parent_aggregate_on_coords(...) {
        //       if group_cardinality(&sel, &shape, &agg.keys) <= self.budget.groups() {
        //           exec.enable_partial_aggregate(agg.keys);    // push SUM/COUNT to chunks
        //       }
        //   }
        //   let n = partitions_from(cost.touched_chunks, target);
        //   exec.set_partitions(n);                             // feeds PartitionSpec
    }
}
```

Every output is something the scan can already act on: `set_batch_plan` is the
streaming work from §2; `set_partitions` feeds the existing
`with_partitions`/`PartitionSpec` path (`physical_plan/zarr_exec.rs`);
`enable_partial_aggregate` is the new aggregate-pushdown hook. The rule reuses the
same `CoordFilters`/`CoordSelection` the scan builds, so it is purely additive.

### 9.5 Testing strategy

- **Tier A vs brute force** — property test: closed-form count == explicit
  enumeration for random boxes, strides, and unions.
- **Tier A vs Tier B** — on separable sets, `product` and `barvinok` must agree
  *exactly*; this is the cross-check that justifies trusting Tier B where Tier A
  can't go.
- **Golden costs** — pin `rows` and `touched_chunks` for the ERA5/ONI selections,
  so a regression in lowering or tiling is caught.
- **Admission** — assert the monthly-mean-over-hourly query is rejected (or
  auto-streamed) with the exact predicted footprint, and the day-15 sample is
  admitted.

---

## 10. Build order — one logical block at a time

To cope with the complexity, build so that each block **compiles, is testable in
isolation, and ideally ships value on its own**, with dependencies pointing only
backward. Two principles do the work: *actuator before controller* (the scan must
be able to stream before the optimizer can choose how), and *observe before act*
(compute exact cardinality read-only before any plan changes behavior).

```
0 streaming ──────────────► 4 drive streaming ──► 5 rule ──► 6 partitions
                          ▲                          │
1 core ─► 2 lowering ─► 3 cost                       └──► 7 agg pushdown
                          │                                      ▲
                          └──────────────────────────────────────┘
                                                     8 Tier-B backend (optional, gated)
```

### Block 0 — Streaming scan (the actuator). *Prerequisite; ships value alone.*
Make the scan emit multiple `batch_size`-bounded batches instead of one (§2).
Independent of all cardinality math; on its own it retires the OOM class. Build
it first because every later block drives it.
*Test:* a large query returns N batches with bounded peak memory.

### Block 1 — The pure core: `CubeShape` + `IndexSet` + Tier-A product backend.
*No DataFusion, no FFI.* The integer-set abstraction with the pure-Rust backend
(boxes, strides/cosets, unions, `cardinality`, `intersect`/`union`,
`project_out`, `touched_tiles`). Zero deps, fully unit-testable.
*Test:* property test — closed form == brute-force enumeration.

### Block 2 — Lowering: `selection_from_filters` (CoordFilters → IndexSet).
*Real queries, still read-only.* Bridge the scan's existing pushed-down filters
into an `IndexSet`; compute exact rows and touched chunks, change nothing.
*Test:* golden cardinalities for ERA5/ONI, cross-checked against the row count
the scan actually produces. *Checkpoint:* you can now **observe** exact
cardinality on live queries at zero behavior risk.

### Block 3 — Cost model + advisory admission.
*Pure functions on Blocks 1–2; no plan changes.* `ScanCost`,
`estimate_scan_cost`, `MemoryBudget`, `admit`, wired as a warning (or error
behind a flag). The monthly-mean query now reports "~60 GB" instead of dying.
*Test:* golden cost numbers. *Checkpoint:* shippable diagnostics.

### Block 4 — First behavior change: drive streaming from cost.
`plan_streaming` picks the outer-axis window from the budget instead of a fixed
`batch_size`, feeding Block 0. The first place cardinality *changes execution* —
the monthly mean auto-streams.
*Test:* OOM query runs bounded; day-15 sample unaffected.

### Block 5 — Formalize as a `PhysicalOptimizerRule`.
*Refactor, not new math.* Lift the calls out of `ZarrTable::scan()` into a rule
registered on the session that walks the plan, annotates `ZarrExec`, and gains
the context to see the aggregate on top. Low risk — the math is already proven.

### Block 6 — Partition count from `touched_chunks`.
*Parallelism, orthogonal to memory.* Feed the existing
`with_partitions`/`PartitionSpec` path a cardinality-derived partition count.

### Block 7 — Group cardinality + partial-aggregate pushdown.
*Biggest new executor capability — last in Tier A.* `group_cardinality` via
projection, plus the `enable_partial_aggregate` hook that pushes `SUM`/`COUNT`
into the chunk reader. Introduces a genuinely new physical operator.
*Checkpoint:* full Tier-A optimizer — exact costs *and* fast aggregates.

### Block 8 — Tier B: `isl`/`barvinok` behind the `polyhedral` feature.
*Deliberately last.* Add the FFI backend for coupled/diagonal/join predicates
only once Tier A is trusted. Highest build friction, lowest query frequency, so
it must never gate the core.
*Test:* on separable sets, `barvinok` must agree **exactly** with Tier A — the
cross-check that licenses trusting it where Tier A cannot reach.

### Where to stop
Blocks 0–4 are ~80% of the practical value at almost no research risk — plumbing
plus arithmetic. Everything genuinely novel (lattice counting, Ehrhart) lives in
Block 8, which is optional. Natural ship points: after **0** (OOM gone), after
**3** (exact diagnostics), after **4** (cardinality-driven streaming — the real
prize), after **7** (full optimizer).
