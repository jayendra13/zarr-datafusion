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
- [x] **Remote (GCS) partitioning** — `partition_range` threaded through the async
      path; `head` registers `gs://`/`s3://` stores and workers rebuild them.

## Next

- [ ] **Load balance narrow filters** — a single-day filter clusters all surviving
      chunks into one partition of the full axis, so only one worker does real
      work. Partition the *surviving* chunk set, not the full axis.
- [ ] **Resolve filter indices on the head** and ship them — workers each read the
      full (~10 MB) time coord to resolve date-part filters independently.
- [ ] **LIMIT + partitioning soundness** — currently falls back to a single
      partition (per-partition limit is unsound).
- [ ] **VirtualiZarr partitioning** — excluded for now (passes `partition_range: None`).

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
