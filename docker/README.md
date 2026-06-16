# Local distributed cluster (Docker Compose)

Simulates a multi-node [datafusion-distributed](https://datafusion-contrib.github.io/datafusion-distributed/)
cluster on one machine: three `worker` containers plus an on-demand `head`
runner, all sharing the local `data/` directory as a read-only volume.

This mirrors the eventual GCP layout — only the `WorkerResolver` (a static list
here, the Compute API on GCP) and the store location (a shared volume here, a
`gs://` URL on GCP) differ.

## Prerequisites

Generate the test data first (creates `data/synthetic_v3.zarr`, etc.):

```bash
./scripts/generate_data.sh
```

## Run

```bash
# Build the image and start 3 workers.
docker compose -f docker/docker-compose.yml up -d --build

# Show the distributed plan (no execution) — should show staged DistributedExec
# with NetworkShuffleExec and ZarrExec leaves spread across tasks.
docker compose -f docker/docker-compose.yml run --rm head \
  "SELECT lat, AVG(temperature) FROM weather GROUP BY lat" --show-plan

# Execute the query for real.
docker compose -f docker/docker-compose.yml run --rm head \
  "SELECT lat, AVG(temperature) FROM weather GROUP BY lat"

# Tear down.
docker compose -f docker/docker-compose.yml down
```

`head` accepts `--store <path>` (default `/data/synthetic_v3.zarr`) and
`--table <name>` (default `weather`).

## What to look for

A correctly distributed query prints a staged plan like:

```
┌── DistributedExec ── Tasks: t0:[p0]
│ ...
│   [Stage N] => NetworkCoalesceExec / NetworkShuffleExec
  ┌── Stage ... Tasks: t0 t1 t2
  │   ...
  │     ZarrExec: path=/data/synthetic_v3.zarr, ...
```

If you instead see a single non-staged plan, the input is too small to
distribute — the planner only fans out past a per-partition byte threshold.

## How it maps to the code

- `src/bin/worker.rs` — the worker gRPC server (registers `ZarrPhysicalCodec` + UDFs).
- `src/bin/head.rs` — plans/runs the query; `StaticWorkerResolver` from `WORKER_URLS`.
- `src/distributed.rs` — shared codec/UDF registration and the resolver.

Both binaries are behind the `distributed` cargo feature:

```bash
cargo build --features distributed --bin worker --bin head
```
