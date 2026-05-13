# Zarr Write Support — Phased Plan

This document lays out a phase-wise plan for adding Zarr **write** support to
`zarr-datafusion`. Each phase has a single, human-verifiable goal and can be
landed independently. Phases are ordered by dependency.

## Goals

End-state we are building toward:

1. Write a Zarr v3 store from a Rust API (POC writer). **— done**
2. Write a Zarr v3 store from a `zarr-cli rechunk` subcommand (specialized fast
   path, bypasses DataFusion).
3. Write a Zarr v3 store from SQL:
   `COPY src TO 'out.zarr' STORED AS ZARR OPTIONS ('chunks.<coord>' '...')`.
4. Support coord-aligned filters and projection through the COPY path
   (subselection: time range, lat range, drop a variable).
5. Stream-write large datasets without buffering the entire array in RAM.
6. Round-trip fidelity: stores written by this library must be readable by
   xarray/zarr-python without losing CF time metadata, dimension names, or
   coord attributes.

## Non-goals

- Zarr v2 writes (read-side already supports v2; write side stays v3-only).
- Remote object-store writes (GCS/S3) — local filesystem only until the local
  path is solid.
- Append/update mode — writes always create a fresh store; existing path is an
  error.
- Value-based filters (`WHERE temperature > 30`) in the COPY path — these
  break Zarr's rectangularity invariant. The sink rejects them with a clear
  error.
- Writing arbitrary tabular sources (Parquet, CSV) into Zarr — the COPY source
  must currently resolve to a `ZarrTable`.

## Architectural context

The central problem of any `COPY → ZARR` sink is mapping a 2D
`SendableRecordBatchStream` (Arrow rows = expanded Cartesian product) back to
chunked nD Zarr arrays. The mapping decomposes into four sub-problems:

1. **Cardinality flip.** One batch contributes to many chunks; one chunk
   needs values from many batches.
2. **Coord deduplication.** Coord columns in the batch repeat values; coord
   arrays in Zarr store each unique value exactly once.
3. **Index reconstruction.** Per-row: `(coord values) → (i, j, k)` nD position.
4. **Chunk-boundary alignment.** Each row lands in chunk
   `(⌊i/cl⌋, ⌊j/co⌋, ⌊k/ct⌋)` at intra-chunk position `(i%cl, j%co, k%ct)`.

Two execution models satisfy this:

- **Buffered:** allocate the full nD array(s) in RAM, place rows at their
  nD positions, then write chunks. Simple. Order-independent.
  Bounded by RAM.
- **Streaming-ordered:** require input to arrive in row-major coord order
  (declared via `required_input_ordering`), keep only one "chunk row" of
  buffers active at a time. Bounded memory; needs the ordering contract.

The plan lands the buffered variant first, then layers streaming on top.

## Key data structures we will lean on

| Structure | Source | Role |
|---|---|---|
| `ZarrStoreMeta` | `reader/schema_inference.rs` (existing) | Target shape description: coords list, data_vars list with shapes/chunks. Sink uses this to plan output before any row arrives. |
| `CoordValues` | `reader/coord.rs` (existing) | Per-coord realized values (`Vec<f32>` / `Vec<f64>` / `Vec<i64>`). |
| **`CoordArray`** | new (Phase 1) | Bundles `ZarrArrayMeta` + `CoordValues` + a fast `value_to_index` lookup. Used by both filter pushdown (read) and sink index reconstruction (write). |
| `SendableRecordBatchStream` | DataFusion | Source of rows feeding the sink. |
| `WriteValues`, `CoordSpec`, `DataVarSpec` | `writer/zarr_writer.rs` (Phase 0) | Existing writer's pre-shaped Rust API. The sink ultimately calls down into this layer. |

---

## Phase 0 — Writer POC (Rust API) — **DONE**

**Goal.** A Rust function `write_zarr_v3(path, coords, data_vars)` that creates
a Zarr v3 store on local disk from pre-shaped inputs.

**Deliverables (landed):**
- `src/writer/{mod.rs, zarr_writer.rs}`
- `tests/integration_writer.rs` (round-trip: writer → existing reader)
- `examples/write_synthetic.rs`

**Verification (landed):**
- Round-trip integration test asserts cell-level equality after read-back.
- Example end-to-end runs SQL against a freshly-written store.

**Limitations carried into later phases:**
- Single chunk per array (= full shape). Phases 4+ replace this.
- No CF time round-trip. Phase 2 fixes.
- No coord attrs (`_ARRAY_DIMENSIONS`, units, calendar) on output. Phase 2.
- Errors if target path exists. Stays that way.

---

## Phase 1 — `CoordArray` abstraction

**Goal.** Lift the per-coord triple (metadata + realized values + value→index
lookup) into one struct, reusable by both the read-side filter pushdown and
the future write-side sink.

**Why.**
- Filter pushdown currently rebuilds value→index lookups inline (`coord.rs`,
  `filter.rs`). The sink will need the same lookups. One struct, one place.
- Makes the `ZarrExec` output-ordering work in Phase 3 cleaner: ordering
  references concrete coord positions.

**Deliverables.**
- New struct `CoordArray { meta: ZarrArrayMeta, values: CoordValues, value_to_index: ValueIndex }`
  in `src/reader/coord.rs` (or a new `src/reader/coord_array.rs`).
- `ValueIndex` is an enum chosen per dtype (e.g., `HashMap<OrderedFloat<f64>, usize>` for
  floats, `HashMap<i64, usize>` for ints; can later add a sorted-array variant
  for compact representations).
- Refactor `reader/filter.rs` coord-lookup sites to use `CoordArray`.
- No public API change; refactor only.

**Verification.**
- Existing test suite passes unchanged.
- Unit tests on `CoordArray`: build, lookup, miss behavior.
- Filter-pushdown integration tests (`integration_pushdown.rs`) unchanged.

**Size.** Small (~200 LOC + test).

**Dependencies.** None.

---

## Phase 2 — Writer fidelity: CF time round-trip + coord/var attrs

**Goal.** `write_zarr_v3` faithfully round-trips CF time coordinates and
dimension-name metadata, so output stores are readable by xarray/zarr-python
with semantic equivalence to the source.

**Why.**
- The reader decodes `time` from `int64 (seconds since 1970)` to
  `Timestamp(Microsecond, UTC)` via CF attrs (`units`, `calendar`). On the
  write side, we need the inverse: re-encode timestamps to integers and emit
  the same attrs.
- Without `_ARRAY_DIMENSIONS` attrs on data vars, xarray cannot reconstruct
  the dim ordering.
- Required by the COPY path (Phase 5+) to deliver useful output.

**Deliverables.**
- Extend `WriteValues` (or add a sibling) to represent CF-attributed coord
  arrays: carry `CFTimeAttrs` alongside the underlying `i64` epoch values.
- `write_zarr_v3` writes the proper attrs onto each array's `zarr.json`:
  - Data var: `_ARRAY_DIMENSIONS = [coord1, coord2, ...]`.
  - Time coord: `units`, `calendar` from `CFTimeAttrs`.
- Optional: pass through arbitrary user-supplied attrs via a new
  `CoordSpec::attrs: Option<JsonValue>` field.

**Verification.**
- New integration test: write a dataset with a CF time coord, then read it via
  Python (`uv run --with xarray --with zarr python -c "..."`) and assert
  `ds.time.dtype == datetime64`, `ds.temperature.dims == ('time','lat','lon')`.
- Existing `write_zarr_v3` round-trip test extended to check
  `_ARRAY_DIMENSIONS` attr presence.

**Size.** Medium (~400 LOC + tests + a Python verification script).

**Dependencies.** None (Phase 1 helpful but not required).

---

## Phase 3 — `ZarrExec` advertises output ordering

**Goal.** Teach `ZarrExec` to declare its `output_ordering` via
`EquivalenceProperties`. The read-side iteration order is already
deterministic — we just don't tell the planner.

**Why.**
- Independently useful: queries like
  `SELECT * FROM zarr ORDER BY lat, lon, time LIMIT 10` no longer insert a
  redundant `SortExec`.
- Critical for the streaming sink (Phase 7) to skip the sort when the source
  is already in the right order.

**Deliverables.**
- In `physical_plan/zarr_exec.rs`, build `EquivalenceProperties` with an
  `output_ordering` of `(coord1 ASC, coord2 ASC, ..., coordN ASC)` reflecting
  the reader's iteration order (alphabetical coord order, innermost-fastest).
- Handle projection: if a coord is projected out, drop it from the ordering.

**Verification.**
- New test: `EXPLAIN SELECT * FROM synth ORDER BY lat, lon, time LIMIT 10`
  produces a plan with no `SortExec`.
- New test: `EXPLAIN SELECT * FROM synth ORDER BY time LIMIT 10` *does*
  include a `SortExec` (innermost coord is not the prefix of native order).
- Existing test suite passes.

**Size.** Small (~80 LOC + 2 tests).

**Dependencies.** None.

---

## Phase 4 — CLI `rechunk` subcommand (specialized fast path)

**Goal.** A standalone subcommand
`zarr-cli rechunk --in src.zarr --out out.zarr --chunks time=7,lat=64,lon=64`
that copies a Zarr store with a new chunk shape, **without going through
DataFusion**.

**Why.**
- The most common write use-case (rechunking) doesn't need SQL.
- Bypasses the entire `SendableRecordBatchStream` mapping problem: this
  reads chunks from source and writes chunks to dest directly. No flatten,
  no unflatten.
- Ships independently of any FileFormat/DataSink plumbing.
- Validates the writer-side chunk-buffering logic before we use it inside a
  more complex sink.

**Deliverables.**
- New module `src/bin/zarr_cli/rechunk.rs` (or a top-level `src/rechunk.rs`)
  with `rechunk(src_path, dst_path, target_chunks)`.
- Reads `ZarrStoreMeta` from src, builds output meta with overridden chunks.
- Streams data var values chunk-by-chunk: iterate over output chunks; for
  each, read the corresponding subset from source via `retrieve_array_subset`
  and write via `store_chunk_elements`.
- Coord arrays: byte-faithful copy (same values, may use different chunking
  for coords or keep single-chunk).
- Preserves all attrs (CF, `_ARRAY_DIMENSIONS`, user attrs).
- CLI subcommand in `zarr-cli`: `zarr-cli rechunk --in --out --chunks <kv>`.

**Verification.**
- Integration test: rechunk synthetic dataset with new chunks; read back via
  existing reader; assert all cells equal element-wise to source.
- Same dataset, rechunk through the CLI; verify on-disk chunk shape via
  inspecting `zarr.json` files.
- xarray check: `xr.open_zarr(out).equals(xr.open_zarr(src))`.

**Size.** Medium (~500 LOC + tests).

**Dependencies.** Phase 2 (preserves CF attrs).

---

## Phase 5 — Buffered DataSink + `COPY ... TO 'path' STORED AS ZARR` wiring

**Goal.** SQL-level write support: a DataFusion `FileFormat` + `DataSink`
that accepts `COPY <ZarrTable> TO 'out.zarr' STORED AS ZARR` and produces
a working Zarr v3 store. **Buffered variant only.**

**Why.**
- Unlocks SQL-driven workflows: dataset transformation pipelines.
- The buffered variant is simple enough to ship as a first cut.
- Establishes the FileFormat plumbing pattern that the streaming variant
  (Phase 7) will reuse.

**Deliverables.**
- `src/datasource/zarr_format.rs`: `ZarrFileFormat` and `ZarrFileFormatFactory`
  implementing DataFusion's `FileFormat` trait on the write side. Parses
  `OPTIONS` (chunks per coord, optional codec choice).
- `src/datasource/zarr_sink.rs`: `ZarrDataSink` implementing
  DataFusion's `DataSink`. Receives `SendableRecordBatchStream`, runs the
  buffered mapping:
  1. Read source `ZarrStoreMeta` (from the plan's `TableProvider`).
  2. Build `Vec<CoordArray>` for output coords (initially same as source,
     unless filtered/projected — Phase 6 extends this).
  3. Allocate full nD `ndarray::ArrayD<f32>` / `ArrayD<f64>` per data var.
  4. Per batch, per row: look up `(i, j, k)` from `CoordArray.value_to_index`;
     write to ndarray at that position.
  5. On end-of-stream: slice ndarrays into chunks; call existing writer.
- Register the format in `zarr-cli`.

**Verification.**
- Integration test: `COPY synth TO '/tmp/out.zarr' STORED AS ZARR
  OPTIONS ('chunks.time' '7','chunks.lat' '5','chunks.lon' '5')` succeeds;
  read-back equals source element-wise.
- xarray cross-check on the written store.
- Error test: `COPY src TO existing.zarr` errors with PathExists.
- Performance baseline: time to rechunk synthetic dataset (small).

**Size.** Large (~800 LOC + tests).

**Dependencies.** Phase 1 (CoordArray), Phase 2 (CF round-trip).

---

## Phase 6 — Coord-aligned filter and projection support in the sink

**Goal.** Support `COPY (SELECT cols FROM src WHERE coord-filters) TO ...`.
Filters must be coord-aligned (e.g., `lat BETWEEN -10 AND 10`, `time = X`,
`time IN (...)`). Projection drops data variables by omission.

**Why.**
- Use case 2 ("subselection"): read, filter on time+location, drop a variable,
  write a new zarr.
- Most-requested combination with Phase 4 rechunking.

**Deliverables.**
- Sink computes output `Vec<CoordArray>` from the *filtered* source coords,
  not source coords directly. Two approaches; pick one:
  - (a) Pre-scan: run `SELECT DISTINCT coord FROM <filtered_query> ORDER BY
    coord` for each coord column. Small queries.
  - (b) Push the filter through `CoordFilters` (already exists in
    `reader/filter.rs`) and read the filtered coord arrays from the source.
- Projection: sink inspects batch schema, only writes data vars present
  in the schema.
- Reject non-coord-aligned filters at plan time: detect by checking if the
  query's filter expressions reference any non-coord column.
- On end-of-stream, assert `sum(cells_filled) == prod(output_shape)`. If not,
  error with a clear message (the filter was not rectangular).

**Verification.**
- `COPY (SELECT lat, lon, time, temperature FROM src WHERE time BETWEEN ...
  AND lat BETWEEN ...) TO 'out.zarr' STORED AS ZARR` produces correctly-shaped
  output; humidity is absent.
- `COPY (SELECT * FROM src WHERE temperature > 30) TO 'out.zarr' ...` errors
  with "non-rectangular filter".
- Verify output xarray-loadable.

**Size.** Medium (~400 LOC + tests).

**Dependencies.** Phase 5.

---

## Phase 7 — Streaming-ordered sink

**Goal.** Replace the buffered variant's full-array allocation with chunk-row
buffering, enabling writes of datasets that don't fit in RAM.

**Why.**
- Lifts the RAM ceiling on the COPY path.
- Required for production-scale rechunking.

**Deliverables.**
- `ZarrDataSink` declares `required_input_ordering = (coord1 ASC, coord2 ASC,
  ..., coordN ASC)`. DataFusion's planner auto-inserts `SortExec` or
  `SortPreservingMergeExec` as needed; Phase 3 ensures the scan can satisfy
  the ordering without sorting in the rechunking case.
- New chunk-buffer manager: keys are tuples of chunk indices; each entry holds
  per-data-var buffers and a fill-count. Flushed when full or on stream end.
- Sink chooses variant: buffered (default for small expected output) or
  streaming (option, or auto-selected by data-size estimate).
- Memory ceiling parameter: `OPTIONS ('max_buffered_chunks' '8')`.

**Verification.**
- Generate a synthetic ~1 GB dataset; rechunk via streaming sink; assert
  peak RSS bounded by `< source size / 4` (or some configurable bound).
- Output equals buffered-sink output element-wise.
- `EXPLAIN COPY ... STORED AS ZARR` shows expected `SortExec` only when
  needed.

**Size.** Large (~600 LOC + tests including a memory-bounded perf test).

**Dependencies.** Phase 3 (output ordering), Phase 5 (buffered baseline).

---

## Phase 8 — Multi-partition / parallel sink

**Goal.** Allow the COPY path to run with multiple scan partitions in
parallel, merging through `SortPreservingMergeExec`.

**Why.**
- The default DataFusion scan can run in parallel; today we'd be forcing
  single-partition for the sink. This phase unlocks throughput.

**Deliverables.**
- Sink declares it can handle merged-ordered input (multiple partitions ->
  one merged stream).
- Verify with multi-partition source: outputs identical to single-partition
  sink.
- Tune target partition count based on chunk grid alignment (chunks should
  not span partition boundaries to avoid spilling between writers).

**Verification.**
- Same correctness tests as Phase 5/6 but with `target_partitions > 1`.
- Throughput benchmark: parallel COPY measurably faster than serial on
  multi-core machines.

**Size.** Medium (~300 LOC + tests).

**Dependencies.** Phase 7.

---

## Decision log / open questions to resolve as phases land

These choices materially affect the design. Marking them now so they don't
get rediscovered in each phase.

1. **Chunk-shape option syntax.**
   `OPTIONS ('chunks.time' '7', 'chunks.lat' '5')` (named, by coord) or
   `OPTIONS ('chunks' '7,5,5')` (positional)?
   *Recommendation:* named. More robust to coord-order changes.

2. **What if the user gives chunks for a coord that doesn't exist?**
   Error. Strict validation in Phase 5.

3. **Coord-array chunking on output.**
   Coord arrays are 1D and typically small. Default: single-chunk. Allow
   override via `OPTIONS ('chunks.<coord>' '...')` applies to both the
   coord and any data vars indexed by it. *Recommendation:* coord arrays
   stay single-chunk regardless; option applies to data vars only.

4. **Codec defaults.**
   Phase 0 uses zarrs default (raw bytes). Production likely wants blosc.
   *Recommendation:* expose `OPTIONS ('codec' 'blosc')` in Phase 5; default
   stays raw until benchmarks justify a change.

5. **Numeric coord equality during index lookup.**
   `f64` exact-match can fail under arithmetic drift. Sink uses strict equality
   and errors on miss. Document this.

6. **Rectangularity check granularity.**
   The end-of-stream check (`cells_filled == expected`) catches missing cells
   but doesn't tell the user which. Phase 6 adds per-chunk fill-rate logging
   under `RUST_LOG=zarr_datafusion::writer=debug`.

7. **Variable renaming in SELECT.**
   `SELECT lat AS la, ...` — disallow at the sink (Phase 5), since the
   output coord name would no longer match the source. Re-evaluate if real
   need surfaces.

## Glossary

- **Coord (coordinate)** — a 1D array in the Zarr store representing one
  dimension (e.g., `lat`, `lon`, `time`).
- **Data variable** — an nD array indexed by the coords (e.g.,
  `temperature(lat, lon, time)`).
- **Cartesian product** — the row-major flattening of all `(c1, c2, ..., cn)`
  combinations into a 2D row stream.
- **Chunk** — the unit of on-disk storage in Zarr; a contiguous slab of the
  array of shape `chunk_shape`.
- **Rectangularity** — the property that the set of rows in a stream forms
  the full Cartesian product of unique coord values per column. Required for
  Zarr output.
- **CoordArray** — the new struct introduced in Phase 1: `(metadata, values,
  value_to_index)` for one coord.
- **Buffered sink / streaming sink** — the two execution models for the
  RecordBatch → Zarr mapping, differing in memory and ordering requirements.
