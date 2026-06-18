#!/usr/bin/env python3
"""Iris supervisor for a datafusion-distributed worker.

The Rust `worker` binary is a plain tonic gRPC server; it cannot speak Iris's
endpoint-registry API. This wrapper bridges that gap: it registers the worker's
reachable address with the Iris controller (so the head can discover it), then
runs the Rust binary as a child and holds the registration for the task's life.

Discovery contract (must match run_query.py):
  - registered under the actor name  WORKER_ACTOR_NAME  (default "zarr-worker")
  - all replicas register the SAME name; the head's resolve() returns all of them

Env consumed:
  IRIS_ADVERTISE_HOST  injected by Iris = this task's VPC-reachable IP
  PORT                 port the Rust worker binds (default 8080)
  WORKER_ACTOR_NAME    discovery name (default "zarr-worker")
  COMMIT_HASH          passed through to the worker for version matching
  WORKER_BIN           path to the Rust binary (default /app/worker)
"""

import os
import signal
import subprocess
import sys

from iris.client import iris_ctx


def main() -> int:
    host = os.environ.get("IRIS_ADVERTISE_HOST")
    if not host:
        # Not running under Iris (or pre-0.2 runtime). Fail loud rather than
        # registering an unreachable address.
        print("IRIS_ADVERTISE_HOST not set — not running inside an Iris task?", file=sys.stderr)
        return 2

    port = int(os.environ.get("PORT", "8080"))
    actor_name = os.environ.get("WORKER_ACTOR_NAME", "zarr-worker")
    worker_bin = os.environ.get("WORKER_BIN", "/app/worker")
    address = f"{host}:{port}"

    ctx = iris_ctx()
    endpoint_id = ctx.registry.register(actor_name, address)
    print(f"registered {actor_name} -> {address} (endpoint {endpoint_id})", flush=True)

    # PORT is already in the environment; the Rust worker reads it directly.
    proc = subprocess.Popen([worker_bin], env=os.environ.copy())

    # Forward termination so the worker shuts down cleanly on preemption/stop.
    def _forward(signum, _frame):
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)

    try:
        return proc.wait()
    finally:
        # Best-effort: the controller also TTL-cleans endpoints on task death,
        # but unregister promptly so a fast-restarting replica doesn't leave a
        # stale address in the registry.
        try:
            ctx.registry.unregister(endpoint_id)
        except Exception as e:  # noqa: BLE001 - cleanup must not mask exit code
            print(f"unregister failed (controller will TTL-clean): {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
