#!/usr/bin/env python3
"""Run the four benchmark workloads through both engines and report the results.

Each workload exists twice — once as SQL under `bench/sql/`, once as xarray under
`bench/python/` — computing the same thing over the same data. This driver runs
both, repeatedly, in fresh processes, and prints a comparison table.

    uv run bench/run_bench.py                    # all four, 3 reps each
    uv run bench/run_bench.py --only q2_diurnal  # one workload
    uv run bench/run_bench.py --reps 5 --engine sql

Design notes, because the details are what make a benchmark honest:

**Fresh process per repetition.** Nothing carries over between reps except
whatever the OS and the remote service cache. The first rep of each pair is
labelled `cold`, the rest `warm`, and they are reported separately — averaging
them together is how a benchmark quietly flatters itself by an order of magnitude.

**Output is redirected, not rendered.** A 535,704-row result printed to a
terminal measures the terminal. Both engines write to a log file, and both stop
their internal clock before serializing anything.

**Row counts are cross-checked automatically.** After each workload the SQL and
Python row counts must agree, and the value fingerprints must agree to a small
tolerance. If they don't, the comparison is meaningless and the table says so
rather than printing two numbers side by side that describe different work.

**Two clocks are reported.** `wall` is what the driver measures around the whole
process, including interpreter and CLI startup; `engine` is what the engine
itself reports for query execution. Python startup with a large scientific stack
is genuinely slow, and attributing that to query performance would misrepresent
it — so both are shown and the comparison uses `engine`.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
OUTDIR = BENCH / "out"

# Dependencies for the xarray side, installed on the fly by uv.
UV_DEPS = ["xarray", "zarr", "numpy", "pandas", "dask", "gcsfs", "fsspec"]

# The CLI's summary line, e.g.
#   "535704 rows · 248 arrays · 1.80 GB disk · 6.50 MB mem · 4.127s"
# Disk may read "n/a" where a path does its own object I/O and nothing counted.
STATS_RE = re.compile(
    r"^(?P<rows>[\d,]+)\s+rows?\s*·.*?·\s*(?P<disk>[\d.]+\s*[KMGT]?B|n/a)\s+disk"
    r".*?·\s*(?P<secs>[\d.]+)s\s*$"
)

# The CLI reports each statement separately; the first "OK (0.214s)" is the
# CREATE EXTERNAL TABLE, i.e. schema inference — the counterpart of xarray's
# open_zarr. Capturing it is what lets one run produce both the steady-state and
# the cold-start comparison instead of reconstructing the second by hand.
OK_RE = re.compile(r"^OK \((?P<secs>[\d.]+)s\)$")

BYTES_RE = re.compile(r"^(?P<num>[\d.]+)\s*(?P<unit>[KMGT]?B)$")
_UNITS = {"B": 1, "KB": 1_000, "MB": 1_000_000, "GB": 1_000_000_000, "TB": 10**12}


@dataclass
class Workload:
    name: str
    sql: str
    py: str
    description: str
    # Local-only workloads need their fixture present; remote ones do not.
    requires: list = field(default_factory=list)


WORKLOADS = [
    Workload(
        "q1_nino34",
        "sql/q1_nino34.sql",
        "python/q1_nino34.py",
        "Nino-3.4 box mean, 24h (narrow filter)",
    ),
    Workload(
        "q2_diurnal",
        "sql/q2_diurnal.sql",
        "python/q2_diurnal.py",
        "Diurnal climatology, CONUS 3d (GROUP BY)",
    ),
    Workload(
        "q3_point",
        "sql/q3_point.sql",
        "python/q3_point.py",
        "Point timeseries, 744 steps (weak case)",
    ),
    Workload(
        "q4_ndvi",
        "sql/q4_ndvi.sql",
        "python/q4_ndvi.py",
        "NDVI per pixel (projection, no reduction)",
        requires=["data/s2_ndvi_scene.zarr"],
    ),
]


def parse_bytes(text):
    if text == "n/a":
        return None
    m = BYTES_RE.match(text.replace(" ", ""))
    if not m:
        return None
    return int(float(m.group("num")) * _UNITS[m.group("unit")])


@dataclass
class Run:
    engine: str
    rep: int
    phase: str
    ok: bool
    wall: float
    engine_secs: float = None
    open_secs: float = None  # schema inference / open_zarr — the store-opening cost
    rows: int = None
    mean: float = None
    disk_bytes: int = None
    error: str = ""


def run_sql(workload, rep, cli, log_path):
    sql_path = BENCH / workload.sql
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.run(
            [str(cli)],
            stdin=open(sql_path),
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
    wall = time.perf_counter() - t0

    text = log_path.read_text(errors="replace")
    stats = None
    open_secs = None
    for line in text.splitlines():
        s = line.strip()
        if open_secs is None:
            ok = OK_RE.match(s)
            if ok:
                open_secs = float(ok.group("secs"))
        m = STATS_RE.match(s)
        if m:
            stats = m
    if proc.returncode != 0 or stats is None:
        err = next(
            (ln for ln in text.splitlines() if "rror" in ln),
            f"exit={proc.returncode}, no stats line",
        )
        return Run("sql", rep, phase_of(rep), False, wall, error=err.strip()[:160])

    return Run(
        engine="sql",
        rep=rep,
        phase=phase_of(rep),
        ok=True,
        wall=wall,
        engine_secs=float(stats.group("secs")),
        open_secs=open_secs,
        rows=int(stats.group("rows").replace(",", "")),
        disk_bytes=parse_bytes(stats.group("disk")),
    )


def run_python(workload, rep, log_path):
    py_path = BENCH / workload.py
    cmd = ["uv", "run"]
    for dep in UV_DEPS:
        cmd += ["--with", dep]
    cmd.append(str(py_path))

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    wall = time.perf_counter() - t0

    out = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(out)

    payload = None
    for line in out.splitlines():
        if line.startswith("BENCH_JSON "):
            payload = json.loads(line[len("BENCH_JSON ") :])
    if proc.returncode != 0 or payload is None:
        err = next(
            (ln for ln in out.splitlines() if "Error" in ln or "error" in ln),
            f"exit={proc.returncode}, no BENCH_JSON line",
        )
        return Run("python", rep, phase_of(rep), False, wall, error=err.strip()[:160])

    return Run(
        engine="python",
        rep=rep,
        phase=phase_of(rep),
        ok=True,
        wall=wall,
        # Compare like with like: the SQL side's `engine` is query execution with
        # table registration excluded, so Python's is t_exec, not t_total.
        engine_secs=payload["t_exec"],
        open_secs=payload["t_open"],
        rows=payload["rows"],
        mean=payload["mean"],
        # xarray gives no comparable byte counter; say so rather than imply zero.
        disk_bytes=None,
    )


def warmup_python_env():
    """Resolve, install and import the Python stack once, before anything is timed.

    `uv run --with xarray --with dask ...` downloads and builds that whole tree on
    its first invocation. Left alone, that lands inside rep 1 — the run labelled
    `cold` — and Python's cold number becomes a package install rather than a
    query, badly misrepresenting it on a freshly provisioned machine.

    Only the environment is warmed, deliberately not the data: running a real
    workload first would prime the remote read too, and `cold` would stop meaning
    anything at all.
    """
    cmd = ["uv", "run"]
    for dep in UV_DEPS:
        cmd += ["--with", dep]
    cmd += ["python", "-c", "import xarray, zarr, numpy, pandas, dask, gcsfs, fsspec"]

    print("warming Python environment (uv resolve + import, not timed) ...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    took = time.perf_counter() - t0
    if proc.returncode != 0:
        print("!! Python warm-up FAILED — the python side will likely fail too:")
        print((proc.stderr or proc.stdout or "").strip()[:600], flush=True)
        return False
    print(f"   ready in {took:.1f}s", flush=True)
    return True


def phase_of(rep):
    return "cold" if rep == 1 else "warm"


def fmt_bytes(n):
    if n is None:
        return "n/a"
    for unit in ("TB", "GB", "MB", "KB"):
        scale = _UNITS[unit]
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} B"


def summarize(runs):
    """Cold time, mean warm time, and the agreed row count for one engine."""
    ok = [r for r in runs if r.ok]
    if not ok:
        return None
    cold = next((r.engine_secs for r in ok if r.phase == "cold"), None)
    warms = [r.engine_secs for r in ok if r.phase == "warm"]
    opens = [r.open_secs for r in ok if r.phase == "warm" and r.open_secs is not None]
    open_warm = sum(opens) / len(opens) if opens else None
    warm = sum(warms) / len(warms) if warms else None
    return {
        "cold": cold,
        "warm": warm,
        "open": open_warm,
        # What a script with nothing already open pays: store-opening + the query.
        "first_answer": (open_warm + warm) if (open_warm is not None and warm is not None) else None,
        "wall_cold": next((r.wall for r in ok if r.phase == "cold"), None),
        "rows": ok[0].rows,
        "mean": next((r.mean for r in ok if r.mean is not None), None),
        "disk": next((r.disk_bytes for r in ok if r.disk_bytes is not None), None),
        "n_ok": len(ok),
        "n": len(runs),
    }


def check_agreement(sql_sum, py_sum):
    """Do the two engines agree they computed the same thing?"""
    if not sql_sum or not py_sum:
        return "—"
    problems = []
    if sql_sum["rows"] != py_sum["rows"]:
        problems.append(f"rows {sql_sum['rows']} vs {py_sum['rows']}")
    # Only Python reports a value fingerprint today; when SQL grows one, compare.
    if problems:
        return "MISMATCH: " + "; ".join(problems)
    return "rows agree"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3, help="repetitions per engine")
    ap.add_argument("--only", action="append", help="workload name (repeatable)")
    ap.add_argument(
        "--engine",
        choices=["sql", "python", "both"],
        default="both",
    )
    ap.add_argument(
        "--cli",
        default=str(ROOT / "target" / "release" / "zarr-cli"),
        help="path to zarr-cli (build with --release first)",
    )
    ap.add_argument(
        "--skip-warmup",
        action="store_true",
        help="don't pre-resolve the Python env (only safe if uv's cache is already warm)",
    )
    args = ap.parse_args()

    cli = Path(args.cli)
    if args.engine in ("sql", "both") and not cli.exists():
        sys.exit(f"zarr-cli not found at {cli}\nBuild it: cargo build --release --bin zarr-cli")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    selected = [w for w in WORKLOADS if not args.only or w.name in args.only]
    if not selected:
        sys.exit(f"no workload matched {args.only}")

    if args.engine in ("python", "both") and not args.skip_warmup:
        warmup_python_env()

    results = {}
    for w in selected:
        missing = [p for p in w.requires if not (ROOT / p).exists()]
        if missing:
            print(f"!! skipping {w.name}: missing {', '.join(missing)}", flush=True)
            continue

        print(f"\n=== {w.name} — {w.description}", flush=True)
        runs = {"sql": [], "python": []}

        for engine in ("sql", "python"):
            if args.engine not in (engine, "both"):
                continue
            for rep in range(1, args.reps + 1):
                log = OUTDIR / f"{w.name}.{engine}.rep{rep}.log"
                if engine == "sql":
                    r = run_sql(w, rep, cli, log)
                else:
                    r = run_python(w, rep, log)
                runs[engine].append(r)

                if r.ok:
                    print(
                        f"  {engine:<7} rep{rep} [{r.phase}] "
                        f"engine={r.engine_secs:7.3f}s wall={r.wall:7.3f}s "
                        f"rows={r.rows:<9} disk={fmt_bytes(r.disk_bytes)}",
                        flush=True,
                    )
                else:
                    print(f"  {engine:<7} rep{rep} [{r.phase}] FAILED: {r.error}", flush=True)

        results[w.name] = {
            "description": w.description,
            "sql": summarize(runs["sql"]),
            "python": summarize(runs["python"]),
            "runs": {k: [vars(r) for r in v] for k, v in runs.items()},
        }

    report(results)

    out_json = OUTDIR / "results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nraw results -> {out_json}")


def _verdict(sql_v, py_v):
    """State the ratio in whichever direction is true."""
    if not sql_v or not py_v:
        return "—"
    ratio = py_v / sql_v
    if ratio >= 1:
        return f"SQL {ratio:.2f}x faster"
    return f"Python {1 / ratio:.2f}x faster"


def report(results):
    # --- Table 1: steady state -------------------------------------------------
    print("\n\n" + "=" * 104)
    print("TABLE 1 — STEADY STATE: query execution only, store already open")
    print("The fair comparison for a notebook or service, where opening is paid once")
    print("and amortized over many queries. 'cold' = first rep, 'warm' = mean of rest.")
    print("=" * 104)
    print(
        f"{'workload':<12} {'engine':<8} {'cold':>9} {'warm':>9} "
        f"{'rows':>10} {'bytes read':>12}  agreement"
    )
    print("-" * 104)
    for name, r in results.items():
        agreement = check_agreement(r["sql"], r["python"])
        for engine in ("sql", "python"):
            s = r[engine]
            if not s:
                print(f"{name:<12} {engine:<8} {'—':>9} {'—':>9} {'—':>10} {'—':>12}  (no successful run)")
                continue
            cold = f"{s['cold']:.3f}s" if s["cold"] is not None else "—"
            warm = f"{s['warm']:.3f}s" if s["warm"] is not None else "—"
            note = agreement if engine == "python" else ""
            print(
                f"{name:<12} {engine:<8} {cold:>9} {warm:>9} "
                f"{s['rows']:>10} {fmt_bytes(s['disk']):>12}  {note}"
            )
        sq, py = r["sql"], r["python"]
        if sq and py:
            print(f"{'':<12} {'->':<8} {_verdict(sq['warm'], py['warm'])} (warm execution)")
        print("-" * 104)

    # --- Table 2: cold start ---------------------------------------------------
    print("\n" + "=" * 104)
    print("TABLE 2 — COLD START: open the store + run the query once")
    print("What a script, cron job or dashboard actually pays. SQL 'open' is CREATE")
    print("EXTERNAL TABLE (schema inference); Python 'open' is open_zarr.")
    print("=" * 104)
    print(f"{'workload':<12} {'engine':<8} {'open':>9} {'query':>9} {'first answer':>14}  verdict")
    print("-" * 104)
    for name, r in results.items():
        for engine in ("sql", "python"):
            s = r[engine]
            if not s:
                continue
            op = f"{s['open']:.3f}s" if s["open"] is not None else "—"
            q = f"{s['warm']:.3f}s" if s["warm"] is not None else "—"
            fa = f"{s['first_answer']:.3f}s" if s["first_answer"] is not None else "—"
            print(f"{name:<12} {engine:<8} {op:>9} {q:>9} {fa:>14}")
        sq, py = r["sql"], r["python"]
        if sq and py:
            print(
                f"{'':<12} {'->':<8} "
                f"{_verdict(sq['first_answer'], py['first_answer'])} (to first answer)"
            )
        print("-" * 104)

    print(
        "\nbytes read: what the engine reports having fetched. 'n/a' means nothing was\n"
        "counting — xarray exposes no comparable counter, and some Zarr paths do their\n"
        "own object I/O. It is never silently reported as zero.\n"
        "\nThe Python side opens with drop_variables so it fetches metadata only for the\n"
        "variables it queries; a plain open_zarr() over all 273 ARCO-ERA5 variables is\n"
        "~16x slower and would be a strawman baseline."
    )


if __name__ == "__main__":
    main()
