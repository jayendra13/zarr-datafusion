# Tier B (polyhedral) use cases and how to measure the win

**Purpose.** Motivate the optional `polyhedral` backend (Phase 8: `isl` + `barvinok`)
with queries a climate analyst actually runs, and pin down **how we would measure**
whether exact lattice-point counting improves anything over the current engine. This
is a design/justification note, not an implemented feature — see
[`exact-cardinality-implementation-plan.md`](exact-cardinality-implementation-plan.md)
§8 and [`exact-cardinality-optimizer.md`](exact-cardinality-optimizer.md).

## The gap in one paragraph

The engine flattens an nD Zarr cube into a 2D table (coordinate columns + data
variables) and reasons about a query's selection as an integer set over the cube
lattice. **Tier A** (`optimizer/cardinality/backend/product.rs`, `ProductSet`)
represents *separable* sets exactly — axis-aligned boxes, per-axis strided cosets,
and unions of them — which covers essentially every single-table box+aggregate query.
It **cannot** represent a set that *couples* two axes (a diagonal, a band, a triangle,
an L1 ball, a floor-division map). For those, Tier A's honest answer is "product of the
axis extents," which overcounts — sometimes by orders of magnitude. Tier B counts the
integer points of the actual polytope exactly (Ehrhart/Barvinok). The `IndexSet` trait
was designed for this split: `apply()` returns `Option`, so Tier A returns `None` for a
coupling map and the optimizer *already* falls back to the conservative bound rather
than failing — Tier B just makes those cases exact.

**Where the win lands:** not (mostly) in reading data — coupled joins are executed by
DataFusion after the scan — but in the **optimizer's cardinality**: join order,
join-algorithm choice, memory admission, partition fan-out, and aggregate-pushdown
viability. The queries below are chosen because they *recur* and because their coupling
is **affine** (Barvinok's domain); a Euclidean "within N km" neighborhood is quadratic
and out of scope without a linear relaxation.

---

## Five practical use cases

Two of these (1, 2) are workloads this repository already runs by hand in its cookbooks.

### 1. Forecast verification — skill vs. lead time for a valid-time window
*Domain: forecast evaluation. In-repo: the EWB "freeze" / forecast-vs-ERA5 cookbook.*

```sql
SELECT f.lead_time, AVG((f.t2 - o.t2) * (f.t2 - o.t2)) AS mse
FROM forecast f JOIN era5 o
  ON o.time = f.init_time + f.lead_time                 -- valid_time coupling
 AND o.time BETWEEN TIMESTAMP '2020-09-01' AND TIMESTAMP '2020-09-30'
GROUP BY f.lead_time;
```

- **Coupling / shape:** `valid = init + lead`; pinning the valid-time window makes
  `init + lead ∈ [lo, hi]` — a **diagonal band** in the (init, lead) plane.
- **Tier A / naïve:** the full `|init| × |lead|` box.
- **Tier B:** exact band count — the join-cardinality that should drive the plan.

### 2. Overlapping running seasons — the ONI 3-month mean
*Domain: climate indices. In-repo: the el-niño ONI cookbook.*

```sql
SELECT s.center_month, AVG(m.sst_anom) AS oni
FROM seasons s JOIN monthly m
  ON ABS(m.month - s.center_month) <= 1        -- DJF, JFM, FMA … (Dec→Jan wraps)
GROUP BY s.center_month;
```

- **Coupling / shape:** `|member − center| ≤ 1` — a **band around the diagonal**,
  *piecewise* because months wrap (Presburger, not a plain polytope).
- **Tier A / naïve:** `12 × 12`.
- **Tier B:** the exact banded count with wrap-around handled.

### 3. Consecutive-day extremes — heatwave / dry-spell detection
*Domain: WMO heatwave, CDD/CWD indices.*

```sql
SELECT a.time AS spell_start
FROM daily a JOIN daily b
  ON b.time BETWEEN a.time AND a.time + INTERVAL '4 days'   -- 5-day window
WHERE a.tmax > 35 AND b.tmax > 35
GROUP BY a.time
HAVING COUNT(*) = 5;
```

- **Coupling / shape:** `a.time ≤ b.time ≤ a.time + 4` — a **band** relating each start
  day to its window (the threshold is a separable data filter).
- **Tier A / naïve:** the `n × n` time self-join box.
- **Tier B:** exact band cardinality (~`n × 5`), so the windowed self-join isn't costed
  as an `n²` blowup.

### 4. Coarsening / regrid — fine cells into coarse boxes
*Domain: preprocessing (0.25°→1°, model-to-obs grid alignment). One of the most common
gridded operations.*

```sql
SELECT c.lat, c.lon, AVG(f.temperature) AS coarse_mean
FROM fine f JOIN coarse c
  ON c.lat_idx = f.lat_idx / 4                 -- integer (floor) division
 AND c.lon_idx = f.lon_idx / 4;
```

- **Coupling / shape:** `coarse = floor(fine / 4)` — a **floor-division map**
  (piecewise-affine; the same operation as `touched_tiles`).
- **Tier A / naïve:** relates two grids only through their extents → `|fine| × |coarse|`.
- **Tier B:** exact fine-cells-per-coarse-cell (join output `= |fine|`, not the product).

### 5. Cumulative accumulation — growing-degree-days / cumulative precip
*Domain: agricultural & hydrological climate.*

```sql
SELECT a.time,
       SUM(GREATEST(b.tmean - 10, 0)) AS gdd_to_date
FROM daily a JOIN daily b
  ON b.time <= a.time                          -- prefix per row
GROUP BY a.time;
-- windowed variant: ON b.time BETWEEN a.time - INTERVAL '90 days' AND a.time
```

- **Coupling / shape:** `b.time ≤ a.time` — a **lower-triangular** region; the windowed
  variant is a **band**.
- **Tier A / naïve:** `n × n`.
- **Tier B:** exact `n(n+1)/2` (triangle) or `~n × 90` (window).

---

## Measurement methodology

"Improvement" has three levels; only some are visible in the physical plan. Measure
them in order — the first needs nothing but the two estimators and a brute-force
oracle; the later two depend on the estimate actually reaching a decision.

### Level 1 — estimate accuracy (the primary, plan-independent artifact)

For each query, compute three numbers and report the **q-error**, the standard metric
for cardinality estimation:

```
q_error = max(estimate / truth, truth / estimate)     # 1.0 is perfect
```

- **truth** — brute-force enumerate the coupled region (small fixtures) or take the
  actual join-output row count from `EXPLAIN ANALYZE`.
- **Tier A / naïve** — the product-of-extents bound the engine falls back to today.
- **Tier B** — the Barvinok/Ehrhart count (prototype via `iscc`; production via the
  `polyhedral` backend).

This table *is* the headline result and matches the thesis framing ("computed, not
estimated → q-error ≈ 1"). It requires **no plumbing into DataFusion** and no plan
change — it isolates the estimator quality itself.

| # | Query | shape | naïve | Tier B (exact) | expected q-error (Tier A) |
|---|-------|-------|-------|----------------|---------------------------|
| 1 | verification (valid window `W`) | band | \|init\|·\|lead\| | band count | **high** (narrow `W` ⇒ 10×–1000×) |
| 2 | ONI 3-month | band (wrap) | 144 | ~36 | ~4× (modest) |
| 3 | heatwave 5-day | band | n² | ~5n | **high** (~n/5) |
| 4 | regrid floor-div | floor map | \|fine\|·\|coarse\| | \|fine\| | **very high** (~\|coarse\|) |
| 5 | cumulative (full) | triangle | n² | n(n+1)/2 | 2× (modest) |
| 5w | cumulative (90-day) | band | n² | ~90n | **high** |

Note the split: **band/floor-map cases (1, 3, 4, 5w) have large q-error** and are the
ones worth building for; the **triangle (5-full) and the 3-way band (2) are only ~2–4×**
and probably don't change any decision.

### Level 2 — plan sensitivity (does the estimate flip a decision?)

A better estimate only matters if it **crosses a threshold** that changes the plan.
`EXPLAIN VERBOSE` prints estimated `Statistics` (row counts) on each operator, so:

1. Produce the plan with the naïve estimate and again with the exact estimate (behind
   `polyhedral`).
2. **Diff the plan trees.** Look for: join order swap, hash-join ↔ nested-loop,
   repartition/partition count, and — for decisions this engine owns — whether
   `ZarrAggregateExec` appears (pushdown admitted) or the partition fan-out changes.

Rule of thumb from the table: expect a plan change on #1/#4 (and #3, #5w), not on #2/#5.

### Level 3 — runtime (only where the plan flipped)

`EXPLAIN ANALYZE` reports **actual** rows and time per operator. For queries whose plan
changed, compare the two variants on wall time, **peak memory** (the memory-pool
high-water mark / process RSS), bytes read, and spill count, and compute the
estimated-vs-actual error per operator to confirm the flipped plan is the better one.

### The plumbing prerequisite (be honest about this)

- **Decisions this engine already owns** — aggregate-pushdown admission (the
  `group_cardinality` gate in `CardinalityRule`) and partition fan-out — consume *our*
  cardinality directly. A Tier-B estimate feeding those flips the decision immediately
  and is **visible in `EXPLAIN`** (operator appears/disappears, partition count moves).
- **Join order / join algorithm** are chosen by DataFusion from each input's
  `Statistics` plus its own join-selectivity guess — it does **not** call
  `group_cardinality`. Until the exact count is *published into the cost model* (e.g. a
  custom optimizer rule that annotates the join's estimated statistics, or overriding
  the scan's reported `Statistics`), the join plan is **byte-identical** and Level 1
  (q-error) is the only observable difference. This bridge is a prerequisite for Levels
  2–3 on the join cases, and is itself a small feature.

### Concrete recipe

Per documented query:
1. **Always:** the q-error row — brute-force truth vs naïve vs Tier B.
2. **If a decision could flip:** `EXPLAIN VERBOSE` with naïve vs exact; diff the trees.
3. **If the plan flipped:** `EXPLAIN ANALYZE` both; compare time / peak memory / bytes /
   spills and estimated-vs-actual error.

---

## Scope & caveats (so the doc doesn't oversell)

- **Coupled predicates are niche.** The dominant SQL-on-Zarr workload is box selection +
  aggregate (pure Tier A). Coupling shows up almost only in **self-joins / cross-grid
  joins**, and even then Tier B *refines an estimate* rather than *enabling a new
  capability* — the join still runs in DataFusion after the scan.
- **Affine only.** Diagonals, bands, triangles, L1/L∞ balls, cosets, floor-division, and
  simplices are in Barvinok's domain; a Euclidean/great-circle neighborhood
  (`Δlat² + Δlon² ≤ r²`) is quadratic and needs an L1/L∞ relaxation first.
- **Build cost is the real tax.** `barvinok → isl → GMP` is cross-platform fragile,
  especially for the static-musl CLI binary; prototype by shelling out to `iscc` before
  committing to FFI. See the Phase-8 effort breakdown in the implementation plan.

## Appendix — polytope-shape taxonomy

The abstract shapes behind both these five and the earlier geometric examples
(diagonal transect, ordered self-join, L1 stencil, characteristic line):

| shape | example predicate | Tier A verdict |
|-------|-------------------|----------------|
| box | `lat BETWEEN a AND b` | exact (Tier A) |
| per-axis coset | `time % 6 = 0` | exact (Tier A) |
| diagonal band / parallelogram | `x − y ∈ [lo, hi]` | **Tier B** |
| triangle / simplex | `x < y` | **Tier B** |
| L1 ball (diamond) | `\|Δx\| + \|Δy\| ≤ r` | **Tier B** |
| sheared line / coset | `x − v·t = c` | **Tier B** |
| floor-division map | `y = ⌊x / k⌋` | **Tier B** |

Rule: **the moment a predicate relates one coordinate axis to another and the region
isn't an axis-aligned box, it's a Tier-B set** — and "product of extents" is the wrong
number.
