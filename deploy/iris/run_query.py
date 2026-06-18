#!/usr/bin/env python3
"""Iris head wrapper: resolve worker endpoints, then run the Rust `head` binary.

The Rust `head` takes the cluster as a static CSV in WORKER_URLS. This wrapper
asks the Iris controller which worker replicas are currently registered, builds
that CSV, and execs the binary. Because the head is one-shot and re-resolves on
every invocation, this transparently tracks a dynamic (autoscaled) worker pool.

Usage (as an Iris job entrypoint):
  python run_query.py "SELECT lat, AVG(sst) FROM era5 GROUP BY lat" [head args...]

The controller address is taken from the Iris context (`iris_ctx().client`),
which Iris wires up automatically for any in-cluster task — no controller URL
needs to be passed in.

Env consumed:
  WORKER_JOB_ID        full job id of the worker pool (e.g. /<user>/zarr-workers)
  WORKER_ACTOR_NAME    discovery name, must match worker_supervisor.py ("zarr-worker")
  STORE_PATH           gs:// URL of the Zarr store (head reads it via --store)
  HEAD_BIN             path to the Rust binary (default /app/head)

Everything after argv[0] is forwarded verbatim to the head binary (the SQL plus
any flags like --table / --show-plan). --store is appended from STORE_PATH if
not already present in the forwarded args.
"""

import os
import sys

from iris.client import iris_ctx
from iris.cluster.types import JobName  # full hierarchical worker job id


def _resolve_worker_urls() -> str:
    worker_job_id = JobName.from_string(os.environ["WORKER_JOB_ID"])
    actor_name = os.environ.get("WORKER_ACTOR_NAME", "zarr-worker")

    # iris_ctx() is initialized from the task's job_info; its client already
    # points at the controller (IRIS_CONTROLLER_ADDRESS is injected by Iris).
    client = iris_ctx().client
    result = client.resolver_for_job(worker_job_id).resolve(actor_name)

    urls = []
    for ep in result.endpoints:
        url = ep.url
        # register() stored "host:port"; the Rust StaticWorkerResolver wants a
        # full http URL. Add the scheme if the registry didn't.
        if "://" not in url:
            url = f"http://{url}"
        urls.append(url)

    if not urls:
        raise SystemExit(
            f"no '{actor_name}' workers registered under {worker_job_id} — "
            "is the worker pool running and healthy?"
        )
    return ",".join(urls)


def main() -> int:
    forwarded = sys.argv[1:]
    if not forwarded:
        raise SystemExit('usage: run_query.py "<SQL>" [head args...]')

    env = os.environ.copy()
    env["WORKER_URLS"] = _resolve_worker_urls()
    print(f"resolved {env['WORKER_URLS'].count(',') + 1} workers", flush=True)

    head_bin = os.environ.get("HEAD_BIN", "/app/head")
    args = [head_bin, *forwarded]
    if "--store" not in forwarded and os.environ.get("STORE_PATH"):
        args += ["--store", os.environ["STORE_PATH"]]

    # exec: replace this process so the head's exit code is the job's exit code.
    os.execve(head_bin, args, env)


if __name__ == "__main__":
    raise SystemExit(main())
