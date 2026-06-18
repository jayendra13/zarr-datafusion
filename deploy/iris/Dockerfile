# Task image for running zarr-datafusion's distributed head/worker under Iris.
#
# Contains: the Rust `worker` + `head` binaries (built with --features distributed)
# and a thin Python layer (the iris client + the two supervisor wrappers) so the
# worker can register its endpoint and the head can discover the pool.
#
# Build & push (from repo root):
#   docker build -f deploy/iris/Dockerfile -t YOUR_REGISTRY/zarr-dist:latest .
#   docker push YOUR_REGISTRY/zarr-dist:latest
#
# This is the image referenced by `default_task_image` in my-cluster.yaml.

# ---- Stage 1: build the Rust binaries ----------------------------------------
# Pinned, not `rust:1` (a floating tag): DataFusion 54 sets the crate MSRV to
# 1.88 (Cargo.toml `rust-version`), and an older base image is exactly the build
# break we hit on the DF-54 bump. 1.95 matches the toolchain the build is
# verified against; bump deliberately, in lockstep with the local toolchain.
FROM rust:1.95-bookworm AS build
WORKDIR /src

# Cache deps first, then build the two distributed binaries.
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --features distributed --bin worker --bin head \
    && cp target/release/worker target/release/head /usr/local/bin/

# ---- Stage 2: runtime image --------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

# Iris client (pulls marin-rigging/finelog from PyPI). marin-iris ships only
# dev/prerelease wheels (no final release), so --pre is required and we pin an
# exact snapshot for reproducibility. Bump this in lockstep with your cluster's
# controller/worker image versions.
RUN pip install --no-cache-dir --pre "marin-iris==0.2.19.dev202606171019"

# Rust binaries.
COPY --from=build /usr/local/bin/worker /app/worker
COPY --from=build /usr/local/bin/head   /app/head

# Supervisor wrappers.
COPY deploy/iris/worker_supervisor.py /app/worker_supervisor.py
COPY deploy/iris/run_query.py         /app/run_query.py

# Default port the worker binds; Iris injects IRIS_ADVERTISE_HOST at runtime.
ENV PORT=8080 \
    WORKER_ACTOR_NAME=zarr-worker \
    WORKER_BIN=/app/worker \
    HEAD_BIN=/app/head

# No CMD: the job entrypoint (worker_supervisor.py or run_query.py) is supplied
# by `iris job run`.
