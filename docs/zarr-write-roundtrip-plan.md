# Zarr write path: design & implementation plan

**Status:** Phase 0 done. Phase 1 core done (skeleton writer landed, exit criterion
met). Phase 2+ unstarted; the design below supersedes the original proposal.

**Scope:** add a Zarr **write** path, validated by a `zarr -> parquet -> zarr` round
trip on synthetic data, then generalised. Read-side behaviour is unchanged
throughout.

**Read §3 first.** Three assumptions the original plan rested on have been
falsified — two by measurement, one by reading code already in this repo. They are
load-bearing for Phases 2-5.

---

## 1. Executive summary

The library was read-only: `ZarrTable` implements `TableProvider::scan` and nothing
wrote arrays. This plan adds a writer, driven by one principle:

> **Establish an oracle before writing any writer code.**

A round-trip test (`read(write(x)) == x`) needs no judgement about what "correct"
means — but it is **blind to symmetric bugs**. If the reader mislabels an axis and
the writer mislabels it back, the test passes and the library is wrong. This is not
hypothetical: `cookbook/ndvi/ndvi.sql` documents having been bitten by exactly this
(bands stored `[x, y]`, transposed from the source `[y, x]`, so alphabetical coord
order matches the data dim order — "the reader would otherwise swap the x/y labels").

**The write that justifies this work is derived-variable materialisation** — compute
NDVI, anomalies, or a climatology in SQL and hand back a Zarr an xarray user can
open. Rechunking is *not* the goal and the tabular layer is a bad rechunker (§9).

**The core insight** (§5.3-5.5): the axis that governs the design is not the source
format and not the use case, but **alignment** — whether each target chunk has
exactly one writer. Misalignment is silent data loss, not a slow path. And the
misaligned case is the *general* one; the aligned case is a narrow optimisation.

---

## 2. What is done

### Phase 0 — Fixture & oracle (done)

The old synthetic fixture could not validate a writer (§4). Phase 0 added a
dedicated one plus an external oracle:

- `scripts/data_gen.py::generate_roundtrip` -> `data/synthetic_rt_v{2,3}.zarr`.
  Distinct axis lengths `time(7) x lat(10) x lon(12)`; a float32 `reflectance` with
  ~49 NaN holes; ragged chunks `(1,4,5)`. Kept **separate** from the shared
  synthetic store, whose 100-row lat*lon plane the streaming/aggregate tests are
  tuned to — reshaping it would silently move which branch they cover.
- `scripts/compare_zarr.py` — zarr-python based (not xarray: xarray cannot open a v2
  store lacking `_ARRAY_DIMENSIONS`, and v2 is where the bugs are). Compares
  structure and bytes.

**Exit criterion met:** `--self-test` shows an identical store accepted (NaN holes
not falsely flagged) and a lat/lon-swapped copy **rejected** on shape. The oracle
has teeth.

### Phase 1 — Skeleton (core done; DDL surface deferred)

- `src/writer/skeleton.rs` — `SkeletonSpec` (coords in dimension order, data vars,
  chunk shape) -> `create_skeleton(path, spec)`. v3 only, Blosc/LZ4.
- `tests/integration_writer.rs` — 4 tests via our own reader.
- `scripts/check_skeleton.py` — the xarray (outside-consumer) half.
- `examples/write_skeleton.rs` — emits the fixture grid.

**Exit criterion met on both halves.** xarray reports
`dims={'time': 7, 'lat': 10, 'lon': 12}` with every data variable reading as
`fill_value`.

Decisions pinned:

- **Empty chunks are free.** zarrs never writes a chunk that is entirely
  `fill_value` (`array_sync_writable.rs`), so "metadata but no chunks" *is* the
  skeleton. Unwritten regions read as `fill_value` with no code of ours involved —
  Phase 2's hole-filling inherits this.
- **`dimension_names` is always written.** Measured: the reader then returns coords
  in dimension order (`time, lat, lon`) and never reaches the alphabetical fallback.
  Writer and reader agree on axes *by construction*, closing the writer's half of
  the §4 axis-swap hazard.
- **Attributes round-trip** (`units`, `long_name`, root `title`) — nearly free via
  the builder. The oracle does not assert on them.
- **v3 write only**; v2 read stays supported.
- **Validation rejects unreadable specs**: chunk rank mismatch, duplicate coord
  names, and a data variable colliding with a coord name (which would be
  re-discovered as a 1-D coordinate, silently losing the variable).

---

## 3. Findings that reshape the plan

These were measured or found in existing code. Each killed an assumption the
original plan was built on.

### 3.1 The dict does not carry the grid

The original §5 proposed coordinate arrays come from "the dict values, not the
rows", flagged as needing verification. `examples/probe_dict_grid.rs` verifies it
against the known `time(7) x lat(10) x lon(12)` fixture:

```
SELECT * FROM rt
  batch 0: rows=240   dict_values: time=2 lat=10 lon=12     <- time=2, not 7
batch_size=32:
  batch 0: rows=24    dict_values: time=1 lat=2  lon=12     <- lat=2, not 10
```

A batch's dict carries the **streaming scan's window**, not the axis. The
assumption was true when written; **the streaming scan invalidated it**.

Two ways it still *looks* true, both traps:

- `LIMIT 5` reports the full `time=7 lat=10 lon=12` — it takes the coordinate-only
  fast path. Spot-checking with a `LIMIT` query confirms nothing.
- After a parquet round trip it also reports the full grid — but that is a
  *value-presence* dict that equals the axis only because all 840 rows sat in one
  unfiltered batch. Filter or split it and it drifts silently.

**Consequences.** Coordinate arrays cannot come from dict values: recovering the
grid from batches needs a union across *all* of them, and since row->chunk mapping
needs the grid, that means buffering the data too — defeating the streaming scan.
**The grid must come from metadata or a declaration, never from the data stream.**
Phase 2's "dict key == axis index" fast path also dies: keys are window-relative
(see §5.6 for what replaces it).

### 3.2 `DataSink` is not file-oriented

The original §5 argued for `insert_into` because "`COPY TO` routes through
`FileFormatFactory`/`DataSink`, which assume a *file*". False.
`datafusion-datasource-54.0.0/src/sink.rs:48`:

```rust
async fn write_all(&self, data: SendableRecordBatchStream, context: &Arc<TaskContext>) -> Result<u64>;
```

Nothing file-shaped. It is `FileSink`/`FileFormat` that are file-oriented, not
`DataSink`. The stated reason to prefer `insert_into` does not exist. Both verbs
route to a sink; the sink is identical either way, so **the verb is a late,
reversible decision** (§6).

### 3.3 `optimizer::cardinality` is already the admission engine

The largest finding. The cardinality module was built for the read path — its docs
never mention `write`, `sink`, or `rechunk`. But it is not really a cardinality
optimizer: it is a **general exact-reasoning engine over a cube's index space**, and
every primitive the sink's admission rule needs already exists there, unit-tested
against brute-force enumeration:

| What the sink needs | What already exists |
|---|---|
| Derive the target grid from the query's predicate | `predicate::selection_from_filters()` — `CoordFilters` -> `ProductSet` |
| Reduce: target grid = the group keys' axes | `group::project_onto(sel, keep_axes)` |
| Reduce admission: group keys must be coordinates | `GroupKey::Axis` / `Periodic`, `group_cardinality()` |
| Alignment: which chunks does a selection touch? | `IndexSet::touched_tiles(&self, tile) -> u128` |
| Walk the plan to the scan, through an aggregate | `rule::descend_to_zarr()` — bails on `children.len() != 1`, i.e. joins |
| Recognise coord group keys vs data vars | `pushdown::recognize()` |
| Bound the sink's buffer memory | `budget.rs` / `cost.rs` — exact `peak_bytes` |

It is Tier A — "pure and dependency-free, no DataFusion here" — so the sink can
reuse it without coupling.

**This falsified a rule I had already written into this doc.** An earlier revision
proposed "coordinate predicates must form a single box per axis; gathers deferred to
a slow path". But `AxisSet` is:

```rust
pub enum AxisSet {
    Ap { first: u64, stride: u64, count: u64 },   // interval AND strided coset
    Indices(Vec<u64>),                            // arbitrary gather
}
```

with `ProductSet` a union of these, counted by inclusion–exclusion. `predicate.rs`
calls out the exact case the box rule proposed deferring: *"scattered `Indices`
(e.g. all December timestamps, or an irregular date-part match) become an explicit
index set — which stays exact on irregular calendars where no clean stride exists."*

So `EXTRACT(MONTH FROM time) = 12` (the ONI/niño recipe) is **already exact**. Both
justifications for the box rule dissolve: the grid is derivable for gathers
(`selection_from_filters`), and alignment is decidable for gathers
(`touched_tiles`). The box rule invented a limitation the machinery does not have,
and is **struck**. The real admission line is §5.8: *what fails to lower*.

---

## 4. Why the shared fixture cannot validate a writer

`scripts/data_gen.py` synthetic defaults are `nlat=10, nlon=10, ntime=7`, both
variables `(ntime, nlat, nlon)`, chunked `(1, nlat, nlon)`:

1. **`lat` and `lon` are both length 10.** `schema_inference.rs:619`: *"multiple
   coordinates share the same size — we fall back to alphabetical ordering."* On v2
   (no dimension names) the lat/lon assignment is unverifiable by construction. Swap
   them and every shape matches, the round trip passes, the data compares equal.
2. **v2 and v3 take different paths.** v3 carries `dimension_names`; v2 carries
   neither those nor `_ARRAY_DIMENSIONS`, so it lands in the shape-inference
   fallback. **v2 is where the bugs will be.**
3. **The data is integer.** `randint` -> int64, so it exercises no float, NaN, or
   `fill_value` behaviour — precisely the machinery Phase 2 depends on and what NDVI
   needs.

Hence the separate round-trip fixture (§2), rather than reshaping the shared one.

---

## 5. The design space

### 5.1 Sources: all that matters is where the grid lives

```
 SOURCE                          GRID LIVES IN     SKELETON BUILDABLE
 ------------------------------  ----------------  ---------------------
 Zarr                            metadata          at plan time, no pass
   (+ GRIB/NetCDF/HDF5 arriving  (coord arrays)
    as virtual Zarr)
 ------------------------------  ----------------  ---------------------
 Parquet                         data only         needs a pass...
 CSV / Avro                      data only          ...or a declaration
 VALUES / generate_series        nowhere           must declare
```

- **Columnar vs row-oriented is not a distinction for us.** By the time DataFusion
  hands the sink a stream, Parquet and CSV are both Arrow batches. The only residue
  is opportunistic dict encoding (a speedup only — §5.6) and Parquet row-group stats
  (bounds, not axis values, so they cannot reconstruct a grid).
- **"Gridded source" means `ZarrTable` in practice.** GRIB/NetCDF/HDF5 arrive *as
  Zarr* via virtualizarr / icechunk virtual chunk refs.

### 5.2 Five use cases, one operation

```
   copy    subset   transform   rechunk   reduce
     |        |         |          |        |
     +--------+----+----+----------+--------+
                   |
                   v
      write rows into a declared target grid
           with a declared chunk shape
```

The sink never learns the source's chunking, so **rechunk is not a mode — it is a
different value in the `chunks` option**. These are user-facing descriptions of one
mechanism: no modes, no branches named after use cases.

- `copy` is not a feature — zarr -> zarr with everything identical is a file copy.
  Its only role is the Phase 7 round-trip oracle.
- **`reduce` was missing from the original taxonomy, and it dominates.** Of the
  cookbook recipes, the three climatology, four ONI/niño34, and freeze-RMSE recipes
  are all `GROUP BY`. Only two are not: `ndvi.sql` (a pure per-pixel projection —
  its own header calls it "the complement" to everything else) and
  `freeze/case30_box_dims.sql` (a global aggregate, no `GROUP BY`). A design that
  cannot write a `GROUP BY` result cannot materialise a climatology — arguably the
  most valuable write there is. It fits §5.7 cleanly: **grid = the group keys'
  axes, variables = the aggregate outputs.**

### 5.3 The axis that matters is alignment

**Does the input stream's partitioning map onto target chunks such that each target
chunk has exactly one writer?** An 8x8 `lat` x `lon` plane; letters are the *source
partition* owning each cell; heavy rules are *target chunk* boundaries.

```
 ALIGNED - target chunks (4,4) == source chunks (4,4)

         0 1 2 3   4 5 6 7          lon ->
       +---------+---------+
     0 | A A A A | B B B B |
     1 | A A A A | B B B B |
     2 | A A A A | B B B B |
     3 | A A A A | B B B B |
       +---------+---------+        every target chunk holds
     4 | C C C C | D D D D |        exactly ONE letter
     5 | C C C C | D D D D |        -> one writer -> safe
     6 | C C C C | D D D D |
     7 | C C C C | D D D D |
       +---------+---------+
  lat v

 MISALIGNED - target chunks (2,8), i.e. full-width row strips

         0 1 2 3   4 5 6 7
       +---------------------+
     0 | A A A A   B B B B   |  <- target chunk 0: writers A + B
     1 | A A A A   B B B B   |
       +---------------------+
     2 | A A A A   B B B B   |  <- target chunk 1: writers A + B
     3 | A A A A   B B B B   |
       +---------------------+
     4 | C C C C   D D D D   |  <- target chunk 2: writers C + D
     5 | C C C C   D D D D   |
       +---------------------+
     6 | C C C C   D D D D   |  <- target chunk 3: writers C + D
     7 | C C C C   D D D D   |
       +---------------------+
```

### 5.4 Misalignment is a correctness bug, not a slow path

A Zarr chunk is **one object on disk**, written whole or not at all. Two partitions
each holding part of one target chunk do not merge — they race:

```
  target chunk 0, if A and B both write it:

    A's buffer:  [ A A A A . . . . ]   (its half; rest = fill_value)
    B's buffer:  [ . . . . B B B B ]   (its half; rest = fill_value)

    A calls set() -> disk: [ A A A A _ _ _ _ ]
    B calls set() -> disk: [ _ _ _ _ B B B B ]
                            ^^^^^^^
                            A's half is gone.

    Last writer wins. No error. Silent data loss.
```

This is *why* one-chunk-per-partition is enforced rather than preferred, and why
repartition-by-target-chunk is a **correctness requirement**:

```
    scan A -+
    scan B -+-> shuffle -> one writer owns target chunk 0 -> one set()
    scan C -+
    scan D -+
```

**A second route to the same corruption: a dropped coordinate.** `SELECT lat, lon,
temp FROM era5` against a 3-D store collapses seven `time` values onto each
`(lat, lon)` index; the scatter overwrites and the last row wins. Same silent loss,
different cause. The rule that closes it falls out of `reduce` (§5.8): every target
axis must be projected and unique per row — drop a coord *without* `GROUP BY` and we
reject; drop it *with* `GROUP BY` and it is a reduce, which is well defined.

### 5.5 The inversion: the shuffle path is the general case

```
              do target chunks align with source partitions?
                            |
                +-----------+------------+
               yes                       no
                |                         |
      scan partition == target     target chunk spans
      chunk, 1:1                   several partitions
                |                         |
      shuffle provably a no-op      shuffle REQUIRED
      -> elide it (fast path)       -> for correctness
                |                         |
      Zarr copy / transform         rechunk
      with matching chunks          AND every Parquet / CSV /
                                    generated source - their
                                    partitioning bears no relation
                                    to the target grid at all
```

The original phasing had this backwards: it treated zero-shuffle as the norm and
rechunk's shuffle as a Phase 5 extra. In fact a Parquet/CSV/generated source is
*always* in the rechunk regime — row groups are not chunks — so the shuffle is
needed from the first non-Zarr source onward. `IndexSet::touched_tiles` (§3.3)
decides alignment exactly, at plan time.

### 5.6 Row -> index: name matching, dict as an optimisation only

The sink's job per row is an **index tuple into the target grid**. That needs:

1. **The grid** — axis names, order, values, chunk shape. From metadata or a
   declaration, never from the stream (§3.1).
2. **Which column is which axis** — match by name against the target schema. A
   column matching no axis must match a declared variable, or it is an error.
3. **value -> index**, per coord column:
   - **Dict column**: map each *dict value* to a global axis index once per batch
     (the dict is small), building `LUT[key] = global_index`. Each row is then one
     array lookup: `O(dict_len * log axis)` per batch instead of
     `O(rows * log axis)`. This *rescues* the fast path in a form that survives
     windowing — `key == axis index` is false, but `LUT[key] -> axis index` holds.
   - **Plain column**: per-row binary search into the axis. Correct, just slower.
     Precedent: `filter.rs:1249` `binary_search_lower_bound` already backs `BETWEEN`
     pushdown.

**Build and test the sink against plain columns first** — where nothing clever can
hide a bug — then add the dict LUT with the slow path as its oracle.

Two wrinkles:

- **Sortedness.** Binary search needs monotonic axes. Zarr coords usually are and
  `filter.rs` already bets on it, but it is not guaranteed. Check once at sink
  construction; fall back to a hash map on unsorted axes — noting float hashing needs
  care (bit patterns, `-0.0`, NaN), so sorted + binary search stays primary.
- **Values outside the grid.** A row whose `lat` is not in the axis has no index.
  **Loud error, never a silent drop**: it signals that source and target grids
  disagree.

### 5.7 Grid and variables are independent inputs

```
    source metadata            the query's output schema
    or declaration                        |
          |                               |
          v                               v
     +---------+                     +---------+
     |  GRID   |                     |  VARS   |
     | axis    |                     | names + |
     | names   |                     | dtypes  |
     | values  |                     +---------+
     | order   |                          |
     +---------+                          |
          +---------------+---------------+
                          v
                      SKELETON

    subset    : changes GRID, leaves VARS
    transform : changes VARS, leaves GRID
    reduce    : GRID = group keys' axes; VARS = aggregate outputs
```

Conflating them breaks subset and transform in *opposite* directions.

### 5.8 Admission: express it in `optimizer::cardinality`, do not reinvent

The governing question for Phase 2 is **what is the admissible shape of the query
feeding the sink?** — which projections and aggregations are legal, what must appear
in the output, what is rejected at plan time. Getting it wrong in the permissive
direction produces exactly the silent corruption of §5.4.

Per §3.3 this is not a new problem. The rule should be *expressed in terms of the
existing engine*, roughly:

| Question | Answer |
|---|---|
| Is the query analysable? | `descend_to_zarr()` reaches a `ZarrExec` through single-child nodes; bails on joins |
| What is the target grid? | `selection_from_filters()` -> `ProductSet` (exact for intervals, cosets, gathers, unions) |
| ...under a `GROUP BY`? | `project_onto(sel, key_axes)`; keys admissible iff `GroupKey::Axis`/`Periodic` via `recognize()` |
| Is the write aligned? | `touched_tiles(target_chunk_shape)` |
| Does it fit memory? | `budget.rs` exact `peak_bytes` |
| Otherwise | **reject at plan time with a clear error** |

The admission line is *what fails to lower into an `IndexSet`* — narrower and more
principled than any hand-drawn rule, and already tested.

**Coordinate and data predicates are different and must not be conflated:**

| Predicate kind | Example | Effect on target grid |
|---|---|---|
| coordinate | `lat BETWEEN -5.0 AND 5.0`, `EXTRACT(MONTH FROM time) = 12` | shrinks the grid; lowers to an `AxisSet` |
| data | `NOT isnan(b08 - b04)` | **none**; excluded cells stay `fill_value` |

**Do not ban data predicates.** `cookbook/ndvi/ndvi.sql` filters
`NOT isnan(b08 - b04)`; banning them makes the recipe that justifies this work
inexpressible. Silent fill is correct there — nodata lands as NaN, which is what an
xarray user expects. (`cookbook/el-nino-oni/oni_ersst.sql:35` already carries a
`coordinates only in WHERE` convention comment.)

### 5.9 Tiered execution: bypass the tabular layer when values are a passthrough

§9 says a pure rechunk is the case where the tabular layer contributes nothing —
values are byte-identical to the source, so SQL, Arrow, and coordinate
materialisation are all overhead. The original conclusion was "then it should be a
separate binary with no DataFusion dependency". **That is the wrong split.** The
right one keeps the *planning* and specialises the *execution*.

**The dispatch key** is a plan-level predicate: does each output data column carry
**new** values or **source** values? A bare `Column` reference to a source data
variable, under coordinate `WHERE` filters only, is a passthrough. Any arithmetic,
`CASE`, UDF, or aggregate makes it computed. This is decided by the same
`descend_to_zarr` + `recognize` walk aggregate pushdown already uses (§3.3),
including its join rejection (`children.len() != 1`).

**Three tiers, dispatched on that key:**

| Tier | When | Per-chunk inner loop | zarrs primitive |
|---|---|---|---|
| **0 — compressed passthrough** | identity values **and** target chunk = whole source chunks (no re-tiling) **and** identical codec | copy raw bytes `store.set(key, source_bytes)` — no decode/encode | store-level `get`/`set` on the chunk key |
| **1 — array-native rechunk** | identity values, chunk grids differ (e.g. ERA5 source bundles all levels in one chunk, target splits `level:1`) | `retrieve_array_subset_ndarray` -> scatter in n-D -> `store_array_subset_ndarray` | `array_sync_readable.rs` / `array_sync_readable_writable.rs` |
| **2 — tabular** (the rest of this plan) | values are **new** (transform, reduce) | flatten -> SQL -> row-scatter (§5.6) | the Phase 2 sink |

**What is shared across all three — this is the point.** The bypass does *not*
duplicate plan analysis. Grid derivation (`selection_from_filters` -> `ProductSet`),
alignment / source-target chunk incidence (`touched_tiles`), and
repartition-so-each-target-chunk-has-one-writer (§5.4) are **identical** in every
tier. Only the fill differs — byte-copy vs array-copy vs row-scatter. So the tabular
flatten stops being "the pipeline" and becomes *one implementation of "fill this
target chunk"*, chosen by a plan predicate.

**Mechanism: a plan-rewrite rule, not a fork.** Exactly the move `CardinalityRule`
already makes — `try_pushdown_aggregate` detects `AggregateExec <- ZarrExec` and
swaps in a `ZarrAggregateExec` that never emits the normal row stream. The bypass
detects `ZarrSink <- [coord filter / bare projection only] <- ZarrExec` and swaps in
a `ZarrRechunkExec` that works in ndarray space and **emits only a row count** — so
Arrow is never in the loop. It must fire at plan-rewrite time, because routing
through *any* DataFusion `ExecutionPlan` commits to `RecordBatch`.

**The scheduling insight (Tier 1).** A level-split target chunk requires
decompressing the all-levels source chunk that holds it. The optimal schedule is
therefore *decompress each source chunk once, then write every target chunk it
feeds* — a `GROUP BY source-chunk` on the write side. `touched_tiles` already
computes the source-target incidence, so we can generate that schedule; it is the
difference between a per-hour memory blow-up and one source chunk in flight per
worker.

**Honest limits:**

- **The memory floor is the source's, not ours.** Array-native streaming reaches
  "one source chunk decompressed" and no lower. For an ERA5 model-level source chunk
  (~137 x 721 x 1440 x 4 ≈ 570 MB/variable) that floor is high *because the source
  bundles all levels* — no engine beats it without first re-chunking the source.
- **Mixed projections.** `SELECT lat, lon, temperature, (b08-b04) AS ndvi` mixes a
  passthrough and a computed column. v1: whole-query dispatch — any computed data
  column sends the whole query to Tier 2. Per-variable tiering (passthrough vars
  array-native, computed vars tabular, shared skeleton) is a real refinement, not a
  first cut.
- **Tier 0 assumes identical codecs** source-to-target, and we **write v3 only** —
  so a v2 source or a re-compression drops to Tier 1.

This re-scopes Phase 7 (§7): from "rechunk demo (slow, don't bother)" to "Tier 1
array-native rechunk", which is both faster and the more honest thing to ship. It
reuses Phases 3 and 5 wholesale and adds only the `ZarrRechunkExec` inner loop.

---

## 6. The SQL surface

**Decision: build the sink first as a plain `DataSink`; defer the verb.** The sink
is identical either way.

**Lean: `COPY TO`** — three independent arguments, all found by writing the SQL out
rather than by reasoning about traits:

1. **Subset's filter would be written twice** under `insert_into` (once in
   `OPTIONS`, once in the `SELECT`); drift yields an out-of-grid error or a silently
   under-filled store. `COPY TO` has the query inline, so the grid derives from
   *this* plan's pushed filters — one source of truth.
2. **Variables would be declared twice.** A `TableProvider` must return a schema at
   registration, but under §5.7 variables are known only once the `SELECT` is
   planned — so `insert_into` needs a column list restating the `SELECT`.
3. **A `like` option would copy the grid, not the schema** (§5.7), contradicting
   `CREATE TABLE ... LIKE` everywhere else. `COPY TO` never introduces the option.

| Surface | Cost | UX |
|---|---|---|
| `COPY TO` | a `FileFormatFactory` impl | one statement; matches the `COPY TO ... STORED AS PARQUET` pattern in CLAUDE.md |
| `insert_into` | ~10 lines once the sink exists | two statements; duplicates (1) and (2) |
| Rust/DataFrame API only | least | **disqualifying** — cookbook recipes are `.sql` via `zarr-cli` |

Cost of `COPY TO`: grid derivation must walk to the `ZarrExec`. That is
`descend_to_zarr()`, which already exists and already declines joins — so the
"fragility" is a shipped, tested rejection rule, not a new risk.

### Zarr -> Zarr

| Use case | Grid from | Chunks | Status |
|---|---|---|---|
| copy | derived from plan | = src | doable |
| rechunk | derived from plan | `chunks` option | doable; needs §5.4 shuffle |
| transform | derived from plan | `chunks` option | doable |
| subset | derived from predicate | `chunks` option | doable |
| reduce | `project_onto(sel, keys)` | `chunks` option | doable; needs §5.8 wiring |

```sql
-- subset + transform in one statement; grid derives from the pushed predicate
COPY (SELECT time, lat, lon, (b08 - b04) / (b08 + b04) AS ndvi
      FROM scene
      WHERE lat BETWEEN -5.0 AND 5.0        -- coord  -> shrinks the grid
        AND NOT isnan(b08 - b04))           -- data   -> holes, fill_value
  TO 'data/out.zarr' STORED AS ZARR
  OPTIONS ('chunks' '1,4,5');

-- reduce: grid = the group keys' axes
COPY (SELECT lat, lon, AVG(sea_surface_temperature) AS sst_mean
      FROM era5 GROUP BY lat, lon)
  TO 'data/sst_mean.zarr' STORED AS ZARR
  OPTIONS ('chunks' '64,64');
```

### Non-Zarr sources

```sql
COPY (SELECT time, lat, lon, temperature FROM parquet_src)
  TO 'data/out.zarr' STORED AS ZARR
  OPTIONS ('chunks' '1,4,5',
           'grid' ???);   -- coord VALUES are unspellable in a string
```

No source store to derive from, and a float64 axis cannot be typed into an option.
See §8 gaps 1-2.

---

## 7. Phases

| Phase | Deliverable | Oracle | Risk | Status |
|---|---|---|---|---|
| 0 | Fixture + external oracle | — | low | **done** |
| 1 | Skeleton (metadata, coord arrays, no data) | xarray opens it; all-fill read | low | **core done** |
| 2 | Chunk-aligned sink, single partition, v3, plain columns | round trip + xarray | **high** — core design | **done** |
| 3 | Admission rule wired to `optimizer::cardinality` | plans over the fixture | medium | **structural decision done**; value materialisation deferred |
| 4 | Round trip green, v3 and v2 | round trip + compare_zarr | **high** — first misaligned source | not started |
| 5 | Repartition by target chunk (the shuffle) | round trip + xarray | **high** — correctness (§5.4) | not started |
| 6 | Dict LUT fast path; shuffle elision via `touched_tiles` | slow path as oracle | low | not started |
| 7 | Tier 1 array-native rechunk (`ZarrRechunkExec`) | round trip + xarray | medium; see §5.9, §9 | not started |

Changes from the original phasing: **admission is now its own phase (3)** rather
than an afterthought; **the shuffle is promoted to a first-class phase (5)** because
it is a correctness requirement the parquet round trip already needs (§5.5); and
**the dict fast path is demoted to Phase 6**, behind a working plain-column sink.

### Phase 2 — the sink

> A Zarr chunk must be written as a complete n-D tile, but rows arrive in scan order.

Design: **make the write partitioning be the target chunk grid.** One partition owns
one chunk, so each is independent:

1. Allocate a dense buffer for its chunk, initialised to `fill_value`.
2. **Scatter** each row into the buffer by index — no ordering requirement, so no
   `SortExec` and no `ORDER BY`. This falls out for free, and it is why a
   row-oriented source arriving in arbitrary order costs nothing.
3. On end-of-partition: encode and `store.set(chunk_key, bytes)`.

Single-partition and v3-only. **Do not chase parallelism here** — but note that
single-partition *hides* §5.4 entirely: with one writer no chunk can have two
owners, so Phase 2 green proves nothing about alignment.

**Done** (`src/writer/sink.rs`, `write_batches`). Two simplifications taken for this
cut, both because it is single-partition:

- **The whole array is materialised per variable** and handed to zarrs in one
  `store_array_subset_elements` call, which owns all chunk decomposition — ragged
  edge chunks included, so none of that math is ours to get wrong. Bounded
  per-chunk-per-partition memory (plan gap 3) arrives with partitioning in Phase 5.
- **Default fills only.** A custom `fill_value` is refused, because a full-array
  write must write the holes too and the sink only knows the defaults (0 / NaN).

Row -> index is a binary search of each coordinate value into its axis (plain
columns, not the dict fast path — Phase 6); an off-axis value is a loud error
(§5.6). Validated by `tests/integration_writer.rs` through our own reader **and** by
`scripts/check_sink.py`, which re-derives a position-encoding formula
(`temp[t,y,x] = 1000t+100y+x`) with zarr-python — so an axis transpose or stride
bug that our reader would echo back is caught by a party that shares none of our
conventions. `examples/write_filled.rs` emits the store the script checks.

> **Aside — a real optimiser bug this surfaced (not a sink bug).** A first cut of
> the fill-hole test used `COUNT(x) FILTER (WHERE isnan(x))`; it returned the grid
> cardinality regardless of the predicate. The count optimiser folds `COUNT` to the
> exact cardinality and **ignores an attached `FILTER` clause**. zarr-python
> confirmed the store itself is byte-correct. Worked around in the test with
> `SUM(CASE ...)`; the optimiser bug is logged separately as out of Phase 2 scope.

### Phase 3 — admission & grid derivation

**Structural decision done** (`src/writer/plan.rs`, `derive_write_shape`).

Admission and derivation are one fallible constructor, not a boolean gate: a query
is admissible *because* we can derive its target grid, so `derive_write_shape(plan)
-> Result<WriteShape, RejectReason>` returns the derived shape or a reason. This
settles the plan-discussion Q1 — nothing else builds the spec, because building it
and admitting it are the same analysis; a separate builder would re-derive and the
two would drift (the recurring failure mode — falsified assumptions, #24).

The rule that closes the §5.4 corruption hole is structural: **every source axis
must be projected (kept as an output coordinate) or reduced (a `GROUP BY` key).**
A pure projection that drops an axis is rejected (`DroppedAxis`) — otherwise many
rows collapse onto one cell, last-write-wins. A reduce's target grid *is* the group
keys; admissibility of the aggregate is `recognize()`'s existing decision, **reused
not reinvented** (§5.8) — so `AVG(temperature - 5)` and `GROUP BY <data var>` reject
for free, via the same code that guards aggregate pushdown.

Two scope calls:

- **No store I/O, no cardinality visibility change.** The decision needs only axis
  *names* + the plan's output schema + `recognize()` (already `pub`); the plan-walk
  is local to `writer/`. So `optimizer::cardinality` was not touched — cleaner than
  the "promote `descend_to_zarr`/`project_onto` to pub" the design anticipated.
- **Value materialisation deferred.** `WriteShape` is the grid *structure* (axes in
  dim order, data vars + widened dtypes, `is_reduce`). Turning it into a concrete
  `SkeletonSpec` — loading the source coord arrays and gathering them by any subset
  filter via `selection_from_filters` — needs store I/O that is not cleanly exposed
  yet. It *consumes* a `WriteShape` (does not re-derive it), so Q1's single-source
  rule holds. This is the seam into Phase 4.

Validated by `tests/integration_write_admission.rs` over real physical plans
(baseline context, so a `GROUP BY` stays `AggregateExec <- ZarrExec` rather than
being pushed to `ZarrAggregateExec` first): full projection, transform, coordinate
reduce; and rejections for dropped axis, scalar aggregate, group-by-data-variable,
computed aggregate argument, coordinates-only.

### Phase 4 — round trip

`zarr -> parquet -> zarr`, both formats. Parquet is a deliberate intermediate:
`COPY TO ... STORED AS PARQUET` already works, and it gives an inspectable
checkpoint (the debugging trick in CLAUDE.md). Assert with **both** oracles:
byte-equality of the round trip, *and* `compare_zarr.py` against the original — the
second is the one that can fail interestingly.

Its parquet source is the first input whose partitioning bears no relation to the
target grid, so Phase 5's shuffle is a prerequisite for parallelising it.

---

## 8. What is still missing

| # | Gap | Blocks | Notes |
|---|---|---|---|
| 1 | **Grid provenance, non-Zarr** | Phase 4 | No metadata, no predicate to derive from, float64 axes unspellable in `OPTIONS`. Needs a discovery pass (`SELECT DISTINCT ax ORDER BY ax` per axis) or a reference-store escape hatch. |
| 2 | **Coord/var classification, non-Zarr** | Phase 4 | Parquet carries no marker saying `lat` is an axis. **Distinct from gap 1** — axis *values* and *which columns are axes* are different problems. |
| 3 | **Shuffle memory bound** | Phase 5 | Phase 2 bounds memory at one buffer per partition, true only when a partition owns one chunk. After a repartition each partition owns *many* → N × buffers, not a bound. Needs sort-by-chunk-index within partition (stream one at a time) or a bounded pool. `budget.rs` may supply the accounting (§3.3). |
| 4 | **Coord identity through projection** | Phase 3 | `SELECT time AS t` — under `COPY TO` the output schema *is* the target schema, so "which columns are axes" needs tracing coords through aliases and expressions. Expression analysis, not name matching. |
| 5 | ~~`fill_value` / dtype declaration~~ **(settled)** | — | `WriteDataType::from_arrow` maps a query's Arrow output type by **widen-or-reject, never narrow** (ints -> Int64, Float16/32 -> Float32, Float64 stays; UInt64 and non-numerics refused). `DataVarSpec::from_arrow_field` derives the var. `(b08-b04)/(b08+b04)` over Float32 bands stays Float32. Fill defaults NaN/0, overridable via `DataVarSpec::fill_value`. A *declared* target dtype disagreeing with the query output remains a `COPY TO ... OPTIONS` concern (deferred with the verb, §6). |
| 6 | **Atomicity** | later | A failed write leaves a half-written store. No transaction, no cleanup. icechunk would give this free; plain Zarr will not. |

Gaps 1 and 2 both land on **Phase 4**. Give its first cut a reference-store escape
hatch so real discovery can be deferred rather than blocking the round trip.

---

## 9. What this is not

Rechunking through the tabular layer is **correct but structurally slow**, and this
plan should not pretend otherwise. An ERA5 `[1, 37, 721, 1440]` source chunk is
38.4M cells: 153.7 MB native becomes roughly 460 MB of rows (4 B value + ~8 B of
dict keys per cell) plus Arrow overhead, and we pay flatten/unflatten CPU to
reconstruct indices we already had. A chunk-native tool does a decompress, a slice,
and 37 writes.

A pure rechunk is the case where the tabular layer contributes **nothing**: the
values are byte-identical to the source, so SQL, Arrow, and coordinate
materialisation are all overhead. The write that *justifies* the **tabular** engine
is one where the values are **new** — NDVI, anomalies, climatologies — because there
the tabular layer did the work that produced them.

But "the tabular layer adds nothing here" is an argument for **bypassing it**, not
for abandoning the engine. §5.9 resolves this: the chunk-native path
(decompress -> slice -> write) lives *inside* the sink as Tier 1, gated by a plan
predicate, reusing the same grid-derivation, alignment, and repartition machinery —
it is a specialised `ExecutionPlan` (`ZarrRechunkExec`), not a separate binary. What
this section rules out is doing a pure rechunk *through Arrow rows*; it does not rule
out doing it in this crate.

---

## 10. Sequencing

Phase 0 gates everything (**done**). Phase 1 core is **done**.

**Next: Phase 2** — the sink, against plain columns, grid from an explicit
declaration, single partition, v3, no verb. It is the smallest thing that is
definitely on the critical path under every finding above, and gap 5 is its only
blocker (small: a `fill_value` option and a dtype mapping table).

Then Phase 3 (admission, wired to `optimizer::cardinality`) before Phase 4, so the
round trip is built against a decided rule rather than an implied one.

The smallest complete real instance is the **NDVI recipe**: one variable, 2-D, same
grid as the source, no shuffle, real NaN nodata — which is why Phase 0 put a NaN
variable in the fixture rather than deferring float handling. The **climatology
recipes** are the first real `reduce`, and the first thing that needs §5.8 wired.
