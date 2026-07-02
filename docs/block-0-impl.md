# Block 0 — Streaming Scan: Implementation Plan

> Detailed plan for **Block 0** of the
> [exact-cardinality optimizer roadmap](exact-cardinality-optimizer.md) (§10).
> Block 0 is the *actuator*: make the scan emit multiple `batch_size`-bounded
> RecordBatches instead of one. It is independent of all cardinality math, and on
> its own it retires the OOM class (see §1 of that doc). Every later block drives
> the lever this block installs.

## Terminology (settled)

- **`batch`** — a DataFusion/Arrow `RecordBatch`. Their word; the output unit. Used
  verbatim.
- **`batch_size`** — DataFusion's session config (`datafusion.execution.batch_size`,
  default 8192). **DataFusion decides it, not us.** It arrives via the `TaskContext`
  in `execute(partition, context)` (currently ignored as `_context`,
  `zarr_exec.rs:245`). We read it and use it as the *target* rows per batch.
- **No new "window"/"slab" type.** The input slice that produces one batch is just a
  `CoordSelection` along the outer axis — the *same* type and the *same* splitter
  (`split_selection`) we already use for partitioning. Streaming is fine-grained
  partitioning applied lazily *within* one partition. "Outer-axis sub-selection" is
  prose only; the value is a `CoordSelection`, the helper is `build_batch`. (Note:
  "window" is avoided deliberately — it collides with DataFusion window functions.)

### The mathematical relation

Let `N_out` = outer-axis steps in the (filtered) selection, `R_in` = `inner_rows` =
product of the non-outer effective dimension sizes. Total rows
`query_rows = N_out × R_in` (Cartesian, outer-axis most-significant). Split the outer
axis into sub-selections covering `w_i` steps each (`Σ wᵢ = N_out`):

```
rows(batch_i)   = w_i × R_in
Σ rows(batch_i) = (Σ w_i) × R_in = N_out × R_in = query_rows
```

A batch is an outer sub-selection scaled by the constant `R_in`. To target
`batch_size = B`, pick steps-per-slice `w = max(1, ⌊B / R_in⌋)` ⇒ `⌈N_out / w⌉`
batches. The `max(1, …)` is exactly why a batch can exceed `B` when `R_in > B` (one
plane is indivisible in Block 0 — see the granularity bound below). Because the
flattening is outer-most-significant, each sub-selection is a row-contiguous range, so
`concat(batches) == old_single_batch`. That equality + the sum identity is the entire
correctness proof.

## Phased build order (Block 0 internal)

Each phase compiles and passes tests on its own → one clean commit each.

- **Phase 0 — Plumb `batch_size` (zero behavior change).** Read
  `context.session_config().batch_size()` in `execute()`; thread `batch_size: usize`
  into `read_zarr`, `read_zarr_async`, `AsyncReadParams`, `execute_remote`,
  `execute_virtualizarr{,_with_adapter}`. Don't use it yet. *Verify:* build + existing
  tests green.
- **Phase 1 — Extract `build_batch(sub_sel, …) -> RecordBatch` (zero behavior
  change).** Pull the column loop + `create_result_batch` out of `read_zarr`; call it
  once with the full selection. Output byte-identical. Sync only. *Verify:* existing
  tests green.
- **Phase 2 — Windowed lazy stream, sync path.** `split_selection(outer, chunk_len,
  batch_size).map(build_batch)`, emitted lazily; coord dicts rebuilt at sub-selection
  row count. Settle the two decisions below. *Verify:* new
  `tests/integration_streaming.rs`, `concat(streamed) == reference` (cases 1–8).
- **Phase 3 — Mirror to async + kill `try_collect`.** Same for `read_zarr_async`;
  replace `try_collect().await?` + re-stream (`zarr_exec.rs:401-404`) with lazy
  `try_flatten`. *Verify:* laziness test (case 13), v2/v3 parity (case 12).
- **Phase 4 — LIMIT as per-window stop.** Replace post-hoc slice with early stop;
  slice only the final straddling batch. *Verify:* cases 9–11.
- **Phase 5 — Regression + proof.** Optimizer short-circuit unchanged (14); `#[ignore]`'d
  large/remote peak-memory test (15).

## The core change in one sentence

Both `read_zarr` (sync) and `read_zarr_async` build **one** giant batch and emit
`stream::iter(vec![Ok(batch)])` (`src/reader/zarr_reader.rs:1564`, and the sync
twin near `:1154`). Replace that with: **split the outer-axis selection into
windows, run the existing column-building loop per window, and emit one batch per
window** — lazily.

Because the Cartesian flattening is outer-axis-most-significant, a window of outer
steps is a row-contiguous slice, so `concat(batches) == old_single_batch`. That
equality is the entire correctness story.

**Key reuse:** `src/physical_plan/partition.rs` already has
`split_selection`/`split_indices`, which chop a `CoordSelection` along the outer
axis for partitioning. **Streaming is just fine-grained partitioning applied
lazily within one partition** — the same primitive, so no new splitter is needed.

## Files touched

| File | Change |
|------|--------|
| `src/reader/zarr_reader.rs` | Extract the per-result column-building block into a `build_batch(sub_sel: &CoordSelection) -> RecordBatch` helper; wrap it in a lazy per-sub-selection stream. Both sync + async. The bulk of the work. |
| `src/physical_plan/zarr_exec.rs` | `execute()` currently ignores `_context` (`:245`) — read `session_config().batch_size()` and thread it to the readers. **Critically**, fix `execute_async_read` (`:401-404`): it does `try_collect()` into a `Vec<RecordBatch>` then re-streams, which materializes everything and defeats streaming on the remote path. Replace with passing the lazy inner stream through `try_flatten`. |
| `src/physical_plan/partition.rs` | Reuse `split_selection`/`split_indices` for window splitting; optionally add a thin `windows(sel, outer_steps_per_window)` wrapper. Minimal/no change. |
| `src/reader/*` signatures | Add `batch_size: usize` to `read_zarr`, `read_zarr_async`, `AsyncReadParams`, `execute_remote`, `execute_virtualizarr`. Mechanical ripple. |
| `tests/integration_streaming.rs` | New test file (below). |

**Not touched:** filter parsing, schema inference, the optimizer rules,
`CoordSelection` semantics, `build_read_plans` (called as-is per window).

## Old logic redacted

- **R1** — `stream::iter(vec![Ok(batch)])` single-batch emission (both readers) →
  windowed lazy stream.
- **R2** — `execute_async_read`'s `try_collect().await?` + re-stream
  (`zarr_exec.rs:401-404`) → lazy `try_flatten`. *This is the one that silently
  keeps the remote path non-streaming even after R1.*
- **R3** — `apply_limit_to_arrays` + the single post-hoc `final_rows` slice
  (`zarr_reader.rs:1551`) → limit becomes a **per-window stopping condition**
  (stop after `limit` rows; slice only the final straddling window). The up-front
  `calculate_limited_subset` optimization stays.
- **R4** — the monolithic `result_arrays` / `create_result_batch(..., final_rows)`
  block → moves inside the per-window helper, with `create_coord_dictionary_typed`
  rebuilt at *window* row count, not `final_rows`.

## Test plan — ~15 cases

Grouped by what they protect. The dominant invariant everywhere:
**streamed result == reference single-batch result.**

**Transparency / correctness (the core):**

1. Result > `batch_size` → multiple batches whose concatenation equals the
   reference rows.
2. Result ≤ `batch_size` → exactly 1 batch (no empty trailing batch).
3. Every batch (incl. last) carries the projected schema and ≤ `batch_size` rows
   (modulo the outer-step caveat below).
4. Range filter that still exceeds `batch_size` → streams, equals reference.
5. Scattered `Indices` selection (`day=15` date-part) → exercises the
   `split_indices` window path, equals reference.
6. Projection + streaming → projected columns correct across batches.
7. Coordinate-only query (no data var) + streaming.
8. Mixed-dimensionality variable + streaming.

**Edges:**

9. `LIMIT` < total, landing mid-window → exactly `limit` rows, early stop (no
   extra windows read).
10. Empty result (filter matches nothing) → defined behavior (one empty batch
    with correct schema) and tested.
11. `batch_size` ≥ total rows → single batch (degenerate = old behavior).

**Path / format coverage (parametrize over the existing harness):**

12. v2 and v3 both stream identically.
13. Async/remote path is **lazy** — assert via instrumented store or
    `ZarrIoStats` that batches arrive incrementally and the path no longer
    collects-all (guards R2). Local `object_store` so it runs in CI.

**Regression:**

14. MIN/MAX/COUNT optimizer short-circuit still bypasses the scan (unchanged).
15. A real OOM-regression / peak-memory test on large/remote data — `#[ignore]`'d
    (needs big input), run manually; it's the actual "ships value" proof.

Plus 1–2 **unit tests** for the windowing splitter (a `Range` of N outer steps
with window W → ⌈N/W⌉ contiguous sub-selections covering exactly `[0, N)`; same
for `Indices`).

## Two decisions to make up front

- **Granularity bound.** Block 0 windows *whole outer steps*, so the per-batch
  bound is `max(batch_size, inner_rows)`, where `inner_rows` = product of
  non-outer effective sizes. For the Niño box `inner_rows ≈ 8200` (~one batch) —
  fine. But a single global timestep (721×1440 ≈ 1M rows) won't be sub-split.
  That is acceptable: it still bounds memory to **one plane** instead of the whole
  time series (the actual OOM cause). Sub-plane (inner-axis) windowing is the
  existing "inner-axis partitioning" open gap — explicitly **out of scope** for
  Block 0. Document it.
- **Empty-result contract.** Decide now: zero batches vs one empty batch.
  DataFusion consumers prefer one empty batch with the schema — pick that and test
  it (case 10).

## Definition of done

`concat(streamed) == reference` across cases 1–12; the remote path proven lazy
(13); LIMIT / empty / degenerate handled (9–11); v2/v3 parity (12); optimizer
rules green (14); peak memory bounded on the ignored large test (15). At that
point the monthly-mean query runs bounded with a fixed `batch_size`, and Block 4
later just swaps the fixed size for a cardinality-derived one.

## Known limitations (as built)

These are the deliberate scope boundaries of the shipped Block 0. Both keep the
result **correct**; they only limit *where the memory-streaming optimization
applies*.

- **Granularity bound (whole outer steps).** A window is a whole number of outer
  steps, so the per-batch bound is `max(batch_size, inner_rows)` where `inner_rows`
  = product of the non-outer effective sizes. If one inner plane already exceeds
  `batch_size` (e.g. a single global timestep, 721×1440 ≈ 1M rows), that plane is
  one batch and exceeds `batch_size`. Acceptable: memory is bounded to **one plane**
  instead of the whole time series (the actual OOM cause). Sub-plane (inner-axis)
  windowing is out of scope. Tested by `streaming_indivisible_plane_still_streams`.

- **Broadcast (missing-coord) projections fall back to a single batch.** Windowing
  slices the outer coordinate, which is only transparent when every projected data
  var spans the full **effective** coordinate set — the coords *this projection*
  uses. The gate is `all_full_cube` in `read_zarr` / `read_zarr_async`: a projected
  data var qualifies when `shape.len() == effective_coord_indices.len()` (coordinate
  columns are always fine). The comparison is against the *effective* set, not every
  store coordinate — so a var that merely doesn't use some coordinate (e.g. a
  surface field `sst[time,lat,lon]` in a store that also has a `level` axis) still
  streams. Only a var missing one of the *effective* coords — a genuine broadcast,
  like a static field selected alongside a time-varying one — makes `windows` empty
  and takes the single-batch path.

  *Why:* consider coords `time(100) × lat(721) × lon(1440)` with
  `temperature[time,lat,lon]` (full cube) and a static `elevation[lat,lon]` (no
  time axis). The flattened table is the Cartesian product, so `elevation` is
  *broadcast* — the same lat×lon plane repeated for every timestep. Windowing along
  time works for `temperature` (its read scales with the window) but not for
  `elevation`: it has no time axis to slice, so a per-window read yields a
  `lat×lon` block that would need tiling `(t1−t0)×` to line up. The window path
  doesn't broadcast-tile, so it would mismatch row counts. Rather than special-case
  that, we fall back to the original single-batch path, which already handles the
  broadcast correctly.

  *Consequence:* `SELECT temperature` streams; `SELECT temperature, elevation` (or
  `SELECT elevation`) does not — it can still materialize a large single batch. The
  fallback is driven by the *projection*, not the store. Broadcast-aware /
  inner-axis windowing is future work.
