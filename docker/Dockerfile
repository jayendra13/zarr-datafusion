# Multi-stage build for the datafusion-distributed `worker` and `head` binaries.
# Both binaries live in the same image; docker-compose picks which one to run.
#
# Build from the repo root:
#   docker build -f docker/Dockerfile -t zarr-datafusion-dist .

# datafusion-distributed 2.0 uses Duration::from_mins(), stabilized in Rust 1.92.
# Track the local toolchain (1.95) to keep container builds at parity.
FROM rust:1.95-bookworm AS builder
WORKDIR /build

# Cache dependencies: copy manifests first, then sources.
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY examples ./examples
COPY tests ./tests

RUN cargo build --release --features distributed --bin worker --bin head

FROM debian:bookworm-slim AS runtime
# zarrs/object_store reach out over TLS for remote stores; ca-certificates covers that.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/worker /usr/local/bin/worker
COPY --from=builder /build/target/release/head   /usr/local/bin/head

# Workers listen here; overridable via PORT.
EXPOSE 8080
ENV RUST_LOG=info

# Default to the worker; the head service overrides the command in compose.
CMD ["worker"]
