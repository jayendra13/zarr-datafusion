#!/usr/bin/env bash
#
# Local datafusion-distributed cluster as plain cargo-built processes — no
# Docker, so iterating on the worker/head binaries is just a rebuild away.
#
# Usage:
#   scripts/cluster.sh up                      # build + start workers
#   scripts/cluster.sh status                  # show running workers
#   scripts/cluster.sh query "SELECT ..."      # run a query via the head
#   scripts/cluster.sh query "SELECT ..." --show-plan
#   scripts/cluster.sh down                    # stop workers
#   scripts/cluster.sh restart                 # down + up
#
# Tunables (env vars):
#   WORKERS=3            number of workers
#   BASE_PORT=8080       first worker port; workers use BASE_PORT..BASE_PORT+N-1
#   STORE_PATH=...       Zarr store the head queries (default data/synthetic_v3.zarr)
#   TABLE=weather        table name registered by the head
#   PROFILE=release      cargo profile: release|debug (debug compiles faster)
#   RUST_LOG=info        log level for workers and head
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/.cluster"

WORKERS="${WORKERS:-3}"
BASE_PORT="${BASE_PORT:-9090}"
STORE_PATH="${STORE_PATH:-$ROOT/data/synthetic_v3.zarr}"
TABLE="${TABLE:-weather}"
PROFILE="${PROFILE:-release}"
export RUST_LOG="${RUST_LOG:-info}"

CARGO_FLAGS=(--features distributed --bin worker --bin head)
[[ "$PROFILE" == "release" ]] && CARGO_FLAGS+=(--release)
BIN_DIR="$ROOT/target/$PROFILE"

# Comma-separated http URLs the head uses to reach the workers.
worker_urls() {
  local urls=() i
  for ((i = 0; i < WORKERS; i++)); do
    urls+=("http://localhost:$((BASE_PORT + i))")
  done
  local IFS=,
  echo "${urls[*]}"
}

build() {
  echo ">> building (profile=$PROFILE)" >&2
  (cd "$ROOT" && cargo build "${CARGO_FLAGS[@]}")
}

# Block until a TCP port accepts connections, or give up.
wait_port() {
  local port=$1 tries=100
  while ((tries-- > 0)); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      exec 3>&- 3<&-
      return 0
    fi
    sleep 0.1
  done
  return 1
}

up() {
  build
  mkdir -p "$RUN_DIR"
  local i port pidfile
  for ((i = 0; i < WORKERS; i++)); do
    port=$((BASE_PORT + i))
    pidfile="$RUN_DIR/worker-$port.pid"
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
      echo ">> worker on :$port already running (pid $(cat "$pidfile"))" >&2
      continue
    fi
    PORT="$port" nohup "$BIN_DIR/worker" >"$RUN_DIR/worker-$port.log" 2>&1 &
    echo $! >"$pidfile"
    echo ">> started worker on :$port (pid $!), log: $RUN_DIR/worker-$port.log" >&2
  done
  for ((i = 0; i < WORKERS; i++)); do
    port=$((BASE_PORT + i))
    if wait_port "$port"; then
      echo ">> worker on :$port is accepting connections" >&2
    else
      echo "!! worker on :$port never came up — see $RUN_DIR/worker-$port.log" >&2
      exit 1
    fi
  done
  echo ">> cluster up: $(worker_urls)" >&2
}

down() {
  if [[ ! -d "$RUN_DIR" ]]; then
    echo ">> no cluster state; nothing to stop" >&2
    return 0
  fi
  local pidfile pid
  shopt -s nullglob
  for pidfile in "$RUN_DIR"/worker-*.pid; do
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo ">> stopped worker pid $pid" >&2
    fi
    rm -f "$pidfile"
  done
  shopt -u nullglob
}

status() {
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "no cluster state ($RUN_DIR absent)"
    return 0
  fi
  local pidfile pid port any=0
  shopt -s nullglob
  for pidfile in "$RUN_DIR"/worker-*.pid; do
    any=1
    pid="$(cat "$pidfile")"
    port="$(basename "$pidfile" .pid)"
    port="${port#worker-}"
    if kill -0 "$pid" 2>/dev/null; then
      echo "worker :$port  pid $pid  RUNNING"
    else
      echo "worker :$port  pid $pid  DEAD (stale pidfile)"
    fi
  done
  shopt -u nullglob
  ((any)) || echo "no workers tracked"
}

query() {
  if [[ $# -lt 1 ]]; then
    echo "usage: $0 query \"SELECT ...\" [--show-plan] [--store PATH] [--table NAME]" >&2
    exit 2
  fi
  if [[ ! -x "$BIN_DIR/head" ]]; then
    echo "!! head binary not built; run '$0 up' first" >&2
    exit 1
  fi
  # Defaults go first (STORE_PATH via env, table via flag); the caller's own
  # args follow so an explicit --store/--table wins (the head takes last-wins).
  WORKER_URLS="$(worker_urls)" STORE_PATH="$STORE_PATH" \
    "$BIN_DIR/head" --table "$TABLE" "$@"
}

case "${1:-}" in
up) up ;;
down) down ;;
restart)
  down
  up
  ;;
status) status ;;
query)
  shift
  query "$@"
  ;;
*)
  echo "usage: $0 {up|down|restart|status|query \"SQL\" [head args...]}" >&2
  exit 2
  ;;
esac
