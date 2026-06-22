# Running zarr-datafusion distributed on GCP via Iris

Iris orchestrates the **lifecycle + discovery + GCP VMs**. It does **not** carry
query traffic — the Arrow Flight / tonic RPC flows head ↔ worker directly, so a
GCP firewall rule for the worker port is required (step 0).

```
                 Iris controller (GCE VM)            registry: zarr-worker -> ip:8080 (xN)
                   ^ lifecycle/discovery                  ^ register          | resolve
   iris CLI --------|                                     |                   v
                    +--> worker pool (N CPU VMs) ----- worker_supervisor.py + /app/worker
                    |         ^ tonic :8080  <----- direct query stages -----+
                    +--> head job (1 short VM) ------ run_query.py + /app/head
                                                  (reads gs:// store; same on all nodes)
```

## Files

| File | Role |
|------|------|
| `my-cluster.yaml` | Standalone GCP cluster config (CPU scale group, no TPU). Edit the `# <-- EDIT` lines. |
| `worker_supervisor.py` | Registers a worker's `ip:port` with Iris, then runs `/app/worker`. |
| `run_query.py` | Resolves worker endpoints, sets `WORKER_URLS`, execs `/app/head <SQL>`. |
| `Dockerfile` | Builds the `worker`+`head` Rust binaries + Python iris client into the task image. |

## Prerequisites

- A GCP project with the **Compute Engine** and **Artifact Registry** APIs enabled.
- Two service accounts (names are conventions; match them in `my-cluster.yaml`):
  - `iris-controller@…` — runs the controller VM, impersonated for SSH/bootstrap.
  - `iris-worker@…` — runs worker/head VMs; needs **read** on your Zarr bucket.
- Local tools: `gcloud` (authenticated via `gcloud auth login` + `gcloud auth application-default login`), `docker`, and the Iris CLI:
  ```bash
  uv pip install "marin-iris[controller]>=0.2.0"   # pulls marin-rigging/finelog from PyPI
  iris --help                                       # sanity check
  ```
- The data store is the **public** ARCO-ERA5 archive — no upload or copy needed:
  `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
  (the same source the `scripts/download_*` tools pull from). It is anonymously
  readable, so workers need no credentials for it.

Throughout, replace `YOUR_PROJECT`, `YOUR_BUCKET`, `YOUR_REGISTRY`, and `<user>`
(your Iris user prefix — `iris whoami` or the `--user` you pass) with real values.

## 0. Edit the cluster config

Open `my-cluster.yaml` and replace every line tagged `# <-- EDIT`:
`platform.gcp.project_id`, both `service_account`s, the controller `zone`, the
`scale_groups.*.zones`, `storage.remote_state_dir`, and `default_task_image`.
Keep the zone/region close to your bucket's region to avoid egress cost and latency.

## 1. One-time GCP setup

```bash
# Firewall: allow the worker tonic port between VMs in the VPC (tag-scoped).
gcloud compute firewall-rules create iris-zarr-worker \
  --project=YOUR_PROJECT --network=default \
  --allow=tcp:8080 --source-tags=iris-worker --target-tags=iris-worker

# The ERA5 data store is public — no grant needed for it. The only bucket that
# needs credentials is the Iris *state* bucket (storage.remote_state_dir); grant
# both cluster SAs access to it.
gsutil iam ch serviceAccount:iris-controller@YOUR_PROJECT.iam.gserviceaccount.com:objectAdmin gs://YOUR_BUCKET
gsutil iam ch serviceAccount:iris-worker@YOUR_PROJECT.iam.gserviceaccount.com:objectViewer  gs://YOUR_BUCKET
```

## 2. Build & push the task image

From the repo root (the Dockerfile expects `Cargo.toml`/`src` in the build context):

```bash
docker build -f deploy/iris/Dockerfile -t YOUR_REGISTRY/zarr-dist:latest .
docker push YOUR_REGISTRY/zarr-dist:latest
```

**Verify:** the binaries are linked and the iris client imports —
```bash
docker run --rm YOUR_REGISTRY/zarr-dist:latest sh -c \
  'python -c "import iris" && /app/head 2>&1 | grep -q "missing SQL query" && echo OK'
```
(`/app/head` has no `--help`; with no args it exits with `missing SQL query`, which
confirms it runs.) Verified locally: image ~1.4 GB on `rust:1.95` + `python:3.12-slim`.

### Where to push: GAR vs GHCR

For cluster runs, **Google Artifact Registry (GAR)** is the lowest-friction target:
the worker SAs pull via existing IAM (no credentials to inject), and same-region
pulls avoid egress cost and cold-start latency on the ~1.4 GB image. `YOUR_REGISTRY`
then looks like `REGION-docker.pkg.dev/YOUR_PROJECT/zarr`.

To publish a **public, reproducible** image instead (e.g. so others can run this
README without a GCP project), push to **GHCR**. `gh` supplies the auth token; the
push itself is still `docker push`:

```bash
# 1. Add the packages scope to your gh session, then log docker into GHCR.
gh auth refresh --scopes write:packages,read:packages
echo "$(gh auth token)" | docker login ghcr.io -u <github-user> --password-stdin

# 2. Tag + push (owner = your GitHub username, lowercase).
docker build -f deploy/iris/Dockerfile -t ghcr.io/<github-user>/zarr-dist:latest .
docker push ghcr.io/<github-user>/zarr-dist:latest

# 3. Make it public (one-time) in the web UI — there is no supported REST API
#    to change an existing package's visibility:
#    https://github.com/users/<github-user>/packages/container/package/zarr-dist/settings
#    -> Danger Zone -> Change visibility -> Public.
```

The `org.opencontainers.image.source` label in the `Dockerfile` links the package
to this repo automatically. A public image lets the GCP VMs pull anonymously (no
credential injection), at the cost of internet egress into GCP — fine for a demo,
but for production runs mirror it into GAR and set that as `YOUR_REGISTRY`.

## 3. Start the cluster

```bash
iris --config=deploy/iris/my-cluster.yaml cluster start
iris --config=deploy/iris/my-cluster.yaml cluster status
```

**Verify:** `cluster status` reports the controller as healthy and prints its
address. Note the controller IP — you need it for step 5 as `IRIS_CONTROLLER_URL`
(`http://<controller-ip>:10000`). First start also builds/pins images, so it can
take a few minutes.

## 4. Launch the long-lived worker pool

```bash
iris --config=deploy/iris/my-cluster.yaml job run \
  --job-name zarr-workers --replicas 8 --no-wait \
  --cpu 8 --memory 32GB --enable-extra-resources \
  -e PORT 8080 -e COMMIT_HASH "$(git rev-parse --short HEAD)" \
  -- python /app/worker_supervisor.py
```

`--replicas 8` gives 8 workers, each registering under `zarr-worker`. The full
job id is `/<user>/zarr-workers`.

**Verify:**
```bash
iris --config=deploy/iris/my-cluster.yaml job list            # zarr-workers RUNNING
iris --config=deploy/iris/my-cluster.yaml job logs zarr-workers | grep registered
# expect one "registered zarr-worker -> <ip>:8080" line per replica
```
The autoscaler may take 1–3 min to provision VMs the first time (scale group
starts at `num_vms: 0`). Wait until all replicas are RUNNING before step 5.

## 5. Run a SQL query (head job)

Test query: **mean SST over the Niño 3.4 box for the daytime hours of 31 Dec 2025**.
Small enough to finish quickly, but non-trivial enough to prove distribution works.

The store is chunked one hour per time step (`…chunk-1…`), and each SST chunk is a
full global field — so the lat/lon box does *not* change how many chunks are read;
the **time window does**. The head resolves the time filter to a contiguous set of
hourly indices and the surviving-set partitioner splits those across the workers:

- `06:00–17:00 UTC` → **12 time chunks** (the daytime window below)
- narrow to `09:00–14:00 UTC` → **6 time chunks**

(These are UTC hours. The Niño 3.4 box sits ~9–10 h west of UTC, so true *local*
daytime would straddle into 1 Jan; we use a UTC daytime window to keep it on one
calendar day and make the chunk count exact.)

```bash
iris --config=deploy/iris/my-cluster.yaml job run \
  --job-name nino34-daytime-q --cpu 4 --memory 16GB \
  -e WORKER_JOB_ID /<user>/zarr-workers \
  -e STORE_PATH gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3 \
  -- python /app/run_query.py "SELECT AVG(sea_surface_temperature - 273.15) AS sst_celsius, COUNT(*) AS n_cells FROM era5 WHERE time BETWEEN '2025-12-31 06:00:00' AND '2025-12-31 17:00:00' AND latitude BETWEEN -5.0 AND 5.0 AND longitude BETWEEN 190.0 AND 240.0" --table era5
```

Expect a single row: `sst_celsius` ≈ 26 °C and `n_cells` = 12 h × 41 lat × 201 lon
= 98,892. No `GROUP BY time`, so we avoid the DictionaryArray date-function crash;
the bare `time BETWEEN …` still drives filter pushdown and the per-worker time split.

**Confirm distribution worked:** `--show-plan` (append after the SQL) prints the
distributed plan — you should see the scan partitioned into stages dispatched to
workers. Cross-check the worker logs (`job logs zarr-workers`) for read activity
across multiple replicas, not just one.

The head runs as an in-cluster job, so it reaches the controller via the
Iris-injected context — no controller URL is passed. (Only set `IRIS_CONTROLLER_URL`
yourself if you ever run the head *outside* the cluster.)

Without `--no-wait`, `job run` streams the head's stdout — the query result table
prints when it finishes. Add `--show-plan` after the SQL to print the distributed
plan instead of executing it (a fast end-to-end wiring check). Re-run this step as
many times as you like against the same worker pool.

## 6. Tear down

```bash
iris --config=deploy/iris/my-cluster.yaml job kill zarr-workers   # stops workers, frees VMs
iris --config=deploy/iris/my-cluster.yaml cluster stop            # tears down the controller
```

**Verify:** `gcloud compute instances list --project YOUR_PROJECT` shows no
lingering `zarr-dist`/`iris-worker` VMs.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| Head: `no 'zarr-worker' workers registered` | Worker pool not RUNNING yet, or `WORKER_JOB_ID` mismatch. Check `job list` and that the id is `/<user>/zarr-workers` exactly. |
| Head hangs / `transport error` connecting to a worker | Firewall rule missing or wrong tags — step 1. Confirm worker VMs carry the `iris-worker` network tag and `tcp:8080` is allowed between them. |
| Worker logs: `IRIS_ADVERTISE_HOST not set` | Job wasn't launched through Iris, or an older `marin-iris`. Confirm the runtime injects `IRIS_ADVERTISE_HOST` (it does on `>=0.2.0`). |
| Workers mis-decode plans / odd codec errors | Workers built from a different commit than the head (incompatible `ZarrPhysicalCodec`). Rebuild head + workers from the same commit. Note: `COMMIT_HASH` only *tags* workers (visible via `GetWorkerInfo`) — datafusion-distributed v2 does **not** enforce it. |
| Worker: `403`/`AccessDenied` on the state bucket | SAs lack access to `storage.remote_state_dir` — step 1. (The ERA5 data store is public, so it won't 403.) |
| Workers never leave PENDING | Scale group can't provision: check quota in the zone, `capacity_type` (preemptible availability), and that `machine_type`/`zones` are valid for the project. |
| Head can't reach the controller | Only an issue if you run the head *outside* the cluster. As an in-cluster job (step 5) the controller address is injected automatically via `iris_ctx()`. |

## Notes & gotchas

- **Version tag (`COMMIT_HASH`):** read only by the *worker* (`worker.rs` →
  `Worker::with_version`); it stamps each worker's build, surfaced via the
  `GetWorkerInfo` RPC for inspection. datafusion-distributed v2 does **not**
  compare or enforce it — it's observability, not a safety gate. Still, build
  head + workers from the same commit: the head ships serialized plans + the
  custom `ZarrPhysicalCodec`, so a worker on a different build could mis-decode.
  (The head ignores `COMMIT_HASH`, so it's set only on the worker job.)
- **One worker per VM** keeps the fixed port 8080 collision-free. If you co-schedule
  multiple workers per VM, switch to Iris named ports (`ports=["flight"]` via the
  programmatic `IrisClient.submit`) and read `iris_ctx().get_port("flight")` in
  `worker_supervisor.py` instead of the fixed `PORT`.
- **Scale-to-zero:** `num_vms: 0` + `scale_down_delay` reap idle workers; the head
  re-resolves on each run, so a fresh pool is picked up automatically (cold-start
  latency per query). For interactive use, set `num_vms` to keep a warm pool.
- **Store access:** every worker rebuilds the store from the same `gs://` URL, so
  the bucket must be reachable by all worker SAs (step 0). No data is shipped
  through the head.

## Verified against marin@main (lib/iris)

Confirmed in the cloned source:
- `JobName` lives in `iris.cluster.types`, parsed via `JobName.from_string`;
- `iris_ctx()` is exported from `iris.client`; `ctx.registry.register(name, "host:port")`;
- `ResolveResult.endpoints[*].url` is the registered address;
- `iris job run -e KEY VALUE` injects task env (per the CLI's own `-e WANDB_API_KEY` example);
- the Iris worker agent injects `IRIS_ADVERTISE_HOST` (= the host's routable IP,
  `task_attempt.py:171`) and `IRIS_CONTROLLER_ADDRESS` / `IRIS_CONTROLLER_URL`
  (`runtime/env.py:124`) into every in-cluster task — you set neither.

Still pin/confirm against *your* installed `marin-iris` version (these files target
`>=0.2.0`): image tags in `my-cluster.yaml`, and that your controller exposes port
10000 to wherever you run the head job (or run the head as a child job inside the
cluster so `IRIS_CONTROLLER_URL` is reachable on the VPC).
