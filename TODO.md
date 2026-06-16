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

## Next

- [ ] **Eliminate the redundant outer-coord read on workers** — workers still read the
      full (~10 MB) outer coord and re-resolve the filter, only for `restrict_to_partition`
      to discard it (REPLACE). Read only the shipped slice instead.
- [ ] **LIMIT + partitioning soundness** — currently falls back to a single
      partition (per-partition limit is unsound).
- [ ] **VirtualiZarr partitioning** — excluded for now (passes no partition selection).
- [ ] **Inner-axis / multi-axis partitioning** — only axis 0 is split, so a single
      timestep over the full global grid stays on one worker.

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
