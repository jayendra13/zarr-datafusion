# TODO — Distributed zarr-datafusion

**Goal:** run zarr-datafusion SQL queries distributed across multiple nodes.

**Approach:** [datafusion-distributed](https://datafusion-contrib.github.io/datafusion-distributed/)
— an embeddable Arrow Flight / Tonic library, not a separate scheduler daemon.
The `worker` binary links this crate and registers our codec directly, so it can
run `ZarrExec`. (We pivoted away from Ballista, whose stock Docker images could
not run our physical plan.)

## Done

- [x] **DataFusion 51 → 54 bump** (arrow/parquet 57 → 58).
- [x] **`ZarrPhysicalCodec`** — serializes `ZarrExec` (schema, path, projection,
      limit, coord_filters, **partitions**) for the worker hop. Workers rebuild
      the store from `path`.
- [x] **`worker` / `head` binaries** + `StaticWorkerResolver` (`src/distributed.rs`,
      `distributed` feature). `scripts/cluster.sh {up,down,query,status}` runs a
      local multi-process cluster; `docker/` runs the containerized equivalent.
- [x] **Scan parallelism** — `ZarrExec` exposes N output partitions by slicing the
      outer (axis-0) chunk grid (`src/physical_plan/partition.rs`). Outer coord
      identified by size-match (not `dimension_names`).
- [x] **Distributed fanout** — `ZarrTaskEstimator` spreads the scan across tasks;
      each worker reads a disjoint `partition_range`. Verified end-to-end on a
      3-worker cluster (local `GROUP BY` and remote GCS Niño 3.4), results match
      the xarray reference.
- [x] **Remote (GCS) partitioning** — partition selection threaded through the async
      path; `head` registers `gs://`/`s3://` stores and workers rebuild them.
- [x] **Surviving-set partitioning** — the head resolves the outer-axis filter to the
      surviving index set (`resolve_outer_selection`, reads only the outer coord) and
      splits THAT across partitions (`split_selection`); workers REPLACE their outer
      selection with the head-resolved slice (single resolution site, no divergence).
      Partitions carry a `CoordSelection` (`Range` or scattered `Indices`). Verified
      e2e: `nino34_day.sql` (single day = 24 chunks) now spreads across all 3 workers
      (was: one worker, rest idle), result unchanged (28.55 °C).
- [x] **No redundant outer-coord read on workers** — when a partition supplies the
      outer selection, the worker reads ONLY that window of the outer coord (a
      `Range` window verbatim, scattered `Indices` via one bounding read + gather)
      and skips re-resolving the outer filter (`filters_without_outer`), instead of
      fetching the full (~10 MB) axis and discarding it. Absolute selections still
      drive the data-var reads; a separate window-relative selection extracts the
      coord values (`make_extract_selections`). Unit + e2e equivalence tests in
      `zarr_reader::outer_read_tests` (partitioned read == full read on real data).

## Next

- [ ] **LIMIT + partitioning soundness** — currently falls back to a single
      partition (per-partition limit is unsound).
- [ ] **VirtualiZarr partitioning** — excluded for now (passes no partition selection).
- [ ] **Inner-axis / multi-axis partitioning** — only axis 0 is split, so a single
      timestep over the full global grid stays on one worker.

## Filter pushdown / coordinate limitations

Surfaced while building the DJF-2025 ONI query (`sql/oni_djf2025_extract.sql`):
an `extract()`-based query panicked instead of pruning to the ~86 noon-of-the-15th
snapshots. Two independent root causes, which compounded.

- [x] **One pushed-down filter per coordinate.** `CoordFilters` was
      `HashMap<coord, CoordFilterKind>` (`src/reader/filter.rs`), and `PartialBounds`
      held single `eq` / `low` / `high` / `date_part` slots merged with a fixed
      precedence **eq > range > date_part** (`into_filter`). So a coordinate could carry
      only ONE of {equality, range, single date-part}; the rest fell back to a
      post-scan `FilterExec`. Effects:
      - `extract(day)=15 AND extract(hour)=12` on `time` kept only one (last wins).
      - a range + a date-part on the same coord couldn't coexist (`year BETWEEN … AND month=12`).
      - `IN` / `OR` weren't parsed at all (only `Eq`/`Gt[Eq]`/`Lt[Eq]`/`Between`/`date_part=`).
      *Fixed:* `CoordFilters` now holds a `Vec<CoordFilterKind>` per coord (AND-composed).
      `PartialBounds::into_filters` emits every collected predicate instead of dropping
      losers; `resolve_coord_selection` resolves each to a position set and intersects
      them (`CoordSelection::intersect`). Added `InList` (`coord IN (...)` → `Indices`
      union) and `DatePartSet` (`EXTRACT(field) IN (...)` → `Indices` union), both
      parsed from `Expr::InList` and round-tripped through the distributed codec. So
      `extract(day)=15 AND extract(hour)=12` and `range AND date-part` now both push
      down fully. (`OR` is still left to a post-scan `FilterExec` — cross-coordinate
      disjunction doesn't fit the per-coord selection model.) Unit tests cover the
      intersect helper, two-date-part composition, range+date-part coexistence, empty
      intersection, IN lists, and EXTRACT-IN sets.

- [x] **`Int16` dictionary keys cap a coordinate at 32,767 distinct values.**
      Coordinates are emitted as `DictionaryArray<Int16Type>`; keys are built with
      `((row_idx / inner_size) % coord_size) as i16` (`src/reader/coord.rs:603`). When
      `coord_size > 32_767` the `as i16` **wraps silently** (index 32_768 → -32_768),
      and arrow's `DictionaryArray::new` then panics:
      `Invalid dictionary key -32768 … expected 0 <= key < 65748`. The ceiling is on
      distinct coordinate values in the *selection*, not rows-per-batch, so batching
      doesn't help; it bites both a scattered `Indices` set (e.g. `hour=12` ≈ 65,748
      timestamps across the full ERA5 hourly axis) and any contiguous `Range` slice
      longer than 32,767 steps. *Fixed:* the key width is now chosen from the
      coordinate cardinality (`dictionary_key_type_for_cardinality`, `src/reader/dtype.rs`),
      stepping `Int16`→`Int32`→`Int64`. The schema declares it from `coord.shape[0]`
      and the reader builds keys at that same width (`create_coord_dictionary_typed`
      takes the schema key type), so batches always match the declared type. An
      undersized key type now fails loudly with a clear "coordinate too large" message
      instead of wrapping. Unit tests cover the Int16/Int32/Int64 ladder and a
      40,000-value coordinate (the old panic case).

      *Interaction:* limitation #1 inflated the selection — only `hour=12` survived
      instead of the `month∈DJF ∧ day=15 ∧ hour=12` intersection (~86 positions) — which
      is what pushed `coord_size` past the `Int16` ceiling and triggered the panic.

## Open questions / risks

- [ ] Per-partition coordinate `DictionaryArray`s are redundantly encoded after
      concat (logically correct, just not deduped).
- [ ] Latent reader underflow when v3 `dimensions` is populated — we avoid it by
      using size-match for the outer coord; worth root-causing.
- [ ] GCP deployment: swap `StaticWorkerResolver` for a Compute API resolver and
      point the store at a `gs://` URL (see `docker/README.md` for the layout).

## Notes

- Project memory: `distributed-execution.md` has the full status, DF-54 breaking
  changes, and gotchas. `ballista-integration.md` documents the abandoned approach.
