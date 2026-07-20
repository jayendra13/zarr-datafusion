//! Memory budget and advisory admission control (Phase 3).
//!
//! A [`MemoryBudget`] is a ceiling on a scan's peak resident footprint; [`admit`]
//! compares a [`ScanCost`]'s exact `peak_bytes` against it and reports the exact
//! predicted overshoot when it doesn't fit — *before* the scan runs, instead of
//! discovering it by dying.
//!
//! In Phase 3 this is **advisory**: the scan logs an [`Infeasible`] verdict as a
//! warning and proceeds unchanged. Phase 4 is where the budget starts *driving*
//! execution (choosing window/tile sizes so `peak_bytes` stays under it).

use std::error::Error;
use std::fmt;

use super::cost::ScanCost;

/// A ceiling on the memory a scan may hold resident at its peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    pub bytes: u64,
}

impl MemoryBudget {
    pub fn new(bytes: u64) -> Self {
        Self { bytes }
    }

    /// Convenience: a budget expressed in gibibytes.
    pub fn gib(n: u64) -> Self {
        Self {
            bytes: n * 1024 * 1024 * 1024,
        }
    }

    /// Read a budget from the `ZARR_MEM_BUDGET_BYTES` environment variable, or
    /// `None` when it is unset or unparseable. This is what makes Phase-3 admission
    /// off-by-default: no env var, no check.
    pub fn from_env() -> Option<Self> {
        std::env::var("ZARR_MEM_BUDGET_BYTES")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(Self::new)
    }
}

/// Default ceiling on one scan partition's resident read window: 256 MiB.
///
/// Deliberately a fixed number rather than a fraction of system RAM. Beyond the
/// point where the window covers one *chunk row* there is nothing left to win —
/// amplification is already 1.0 and a larger window reads exactly the same bytes —
/// so the useful size is a property of the data's chunk geometry, not of the
/// machine. Sizing from RAM would also read the *host's* memory inside a container
/// (cgroup limits live elsewhere) and get the process OOM-killed, and it would have
/// a library unilaterally claiming a share of a host application's memory.
///
/// 256 MiB clears one chunk row for typical geometries — an ARCO-ERA5 timestep slab
/// is roughly 20–40 MB — while staying small enough to be safe anywhere the process
/// can run at all.
pub const DEFAULT_SCAN_MEMORY_LIMIT: u64 = 256 * 1024 * 1024;

/// Resolve the ceiling for a single scan partition's read window.
///
/// Layered, most specific first:
///
/// 1. `explicit` — a budget already resolved upstream (the `CardinalityRule` stamps
///    one onto each `ZarrExec`; `ZARR_MEM_BUDGET_BYTES` feeds that).
/// 2. [`DEFAULT_SCAN_MEMORY_LIMIT`].
///
/// then divided by `partitions`, because each partition streams its own window
/// concurrently and the ceiling describes the whole scan, not one worker. Without
/// this division a scan on an 8-core box would hold 8x the intended memory.
///
/// Returns at least 1 byte so callers can divide by it without guarding.
pub fn resolve_scan_ceiling(explicit: Option<u64>, partitions: usize) -> u64 {
    let total = explicit.unwrap_or(DEFAULT_SCAN_MEMORY_LIMIT);
    (total / partitions.max(1) as u64).max(1)
}

/// Bytes needed to hold one whole **chunk row** — the window at which every chunk is
/// read exactly once, and past which extra memory buys nothing.
///
/// `align[0]` is the outer-axis chunk thickness; the inner axes are spanned whole,
/// because rows are emitted in flatten order and a chunk's cells are scattered
/// across every inner index. Comparing this against the resolved ceiling is what
/// turns an invisible re-read cliff into an actionable number: when it doesn't fit,
/// the caller can say by how much.
pub fn chunk_row_bytes(sizes: &[u64], align: &[u64], row_width: usize) -> u128 {
    if sizes.is_empty() {
        return 0;
    }
    let outer = align.first().copied().unwrap_or(1).max(1).min(sizes[0]) as u128;
    let inner: u128 = sizes[1..].iter().map(|&s| (s as u128).max(1)).product();
    outer * inner * (row_width as u128).max(1)
}

/// The maximum number of output groups an aggregate may produce and still be a
/// pushdown candidate (Phase 7): pushing an aggregate keeps one accumulator per
/// group resident, so this caps that group table. Overridable via
/// `ZARR_MAX_GROUPS`; defaults to 1,048,576 — comfortably covers coordinate and
/// periodic group-bys while ruling out a group-per-cell blow-up.
pub fn max_groups() -> u128 {
    std::env::var("ZARR_MAX_GROUPS")
        .ok()
        .and_then(|s| s.trim().parse::<u128>().ok())
        .unwrap_or(1 << 20)
}

/// Verdict for a scan whose predicted peak footprint exceeds the budget. Carries
/// the exact numbers so the warning/error is actionable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Infeasible {
    pub peak_bytes: u128,
    pub budget_bytes: u64,
}

impl fmt::Display for Infeasible {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "scan peak memory ~{} ({} bytes) exceeds budget {} ({} bytes)",
            human_bytes(self.peak_bytes),
            self.peak_bytes,
            human_bytes(self.budget_bytes as u128),
            self.budget_bytes,
        )
    }
}

impl Error for Infeasible {}

/// Advisory admission: `Ok` when the scan's exact `peak_bytes` fits the budget,
/// else [`Infeasible`] with the predicted footprint. In Phase 3 the caller logs
/// this; it rejects nothing.
pub fn admit(cost: &ScanCost, budget: &MemoryBudget) -> Result<(), Infeasible> {
    if cost.peak_bytes <= budget.bytes as u128 {
        Ok(())
    } else {
        Err(Infeasible {
            peak_bytes: cost.peak_bytes,
            budget_bytes: budget.bytes,
        })
    }
}

// --- Phase 4.1: streaming tile schedule (the controller) -----------------------

/// A per-axis tile schedule: `tiles[i]` is how many indices of axis `i` a single
/// emitted batch spans (axis 0 = outer/streaming axis, in the scan's coordinate
/// order). One batch is the Cartesian product of these tiles, so its row count is
/// `∏ tiles[i]`.
///
/// This generalizes the scan's current single-knob `max_steps` (which tiles only
/// axis 0): when the inner axes alone exceed the target, the plan tiles an inner
/// axis too, so no single outer step forces the whole inner block resident (the
/// design-note Gap-1 fix).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchPlan {
    pub tiles: Vec<u64>,
}

impl BatchPlan {
    /// Rows in one batch: the product of the per-axis tiles.
    pub fn rows_per_batch(&self) -> u128 {
        self.tiles.iter().map(|&t| t as u128).product()
    }

    /// Number of batches this plan emits over a selection of the given per-axis
    /// sizes: the product of `ceil(size / tile)` across axes.
    pub fn n_batches(&self, sizes: &[u64]) -> u128 {
        sizes
            .iter()
            .zip(&self.tiles)
            .map(|(&s, &t)| s.div_ceil(t.max(1)) as u128)
            .product()
    }
}

/// Choose the largest batch whose row count stays within `target_rows`, tiling
/// from the outer axis inward.
///
/// `sizes` are the per-axis *selected* sizes (axis 0 = outer). The rule: keep the
/// deepest axes whole and give the outermost axis that still fits as many steps as
/// possible; every axis outside that "pivot" is single-stepped (tile 1). When the
/// inner axes already exceed the target (`inner_rows > target`), the pivot moves
/// inward — tiling an inner axis — which is exactly what the current outer-only
/// windowing cannot do. When the whole selection fits, the plan is one batch
/// (`tiles == sizes`).
///
/// This is loop-tiling for a memory budget instead of a cache: the same geometric
/// operation as chunk-touch counting, run with a different constraint.
pub fn plan_streaming(sizes: &[u64], target_rows: u64) -> BatchPlan {
    let target = (target_rows as u128).max(1);
    let d = sizes.len();
    if d == 0 {
        return BatchPlan { tiles: Vec::new() };
    }
    for i in 0..d {
        // Rows resident for one step of axis `i` with every deeper axis whole.
        let inner: u128 = sizes[i + 1..].iter().map(|&s| s as u128).product();
        if inner <= target {
            // Axis `i` is the pivot: as many steps as fit; outer axes single-stepped;
            // deeper axes whole (tiles start as `sizes`).
            let mut tiles = sizes.to_vec();
            let steps = (target / inner).max(1).min(sizes[i] as u128);
            tiles[i] = steps as u64;
            for t in tiles.iter_mut().take(i) {
                *t = 1;
            }
            return BatchPlan { tiles };
        }
    }
    // The innermost axis has an empty deeper-product (= 1 <= target), so the loop
    // always returns above.
    unreachable!("innermost axis always fits the target")
}

/// Like [`plan_streaming`], but snapping the tiled axis to whole **chunk rows**.
///
/// A chunk is the unit of I/O: reading one cell fetches and decompresses the whole
/// chunk. Because rows are emitted in flatten order (inner axes sweeping fastest), a
/// chunk's cells are *not* a contiguous run of rows — they are scattered across as
/// many stripes as the chunk is thick along the tiled axis. So a tile thinner than
/// the chunk makes every window re-fetch the same chunks:
///
/// ```text
///   tile = 8 rows of an axis chunked every 256  ->  each chunk fetched 32 times
///   tile = 256 (one whole chunk row)            ->  each chunk fetched once
/// ```
///
/// Measured on `data/era5_v3.zarr`, that is the difference between 882 chunk reads
/// (714 MB, 1.61 s) and 15 reads (19.17 MB, 0.065 s) for the same query.
///
/// The budget still wins: this never returns a larger batch than `plan_streaming`
/// would, it only rounds the tiled axis *down* to a whole number of chunk rows. When
/// the target cannot hold even one chunk row, it returns the unaligned plan rather
/// than blowing the budget — re-reads are a performance problem, an OOM is a
/// correctness one. Callers can detect that case via
/// [`BatchPlan::chunk_amplification`] and report it instead of degrading silently.
///
/// `chunks` is the per-axis chunk extent in the same axis order as `sizes`; a
/// missing or zero entry is treated as 1 (no alignment constraint).
pub fn plan_streaming_chunked(sizes: &[u64], target_rows: u64, chunks: &[u64]) -> BatchPlan {
    let plan = plan_streaming(sizes, target_rows);
    let d = sizes.len();
    if d == 0 {
        return plan;
    }

    let mut tiles = plan.tiles;
    for i in 0..d {
        let size = sizes[i];
        let tile = tiles[i];
        // Only the partially-tiled axis can be misaligned: a whole axis (tile ==
        // size) already consumes each of its chunks once, and a single-stepped
        // outer axis (tile == 1) is only there because deeper axes filled the
        // budget — widening it would exceed the target.
        if tile >= size || tile <= 1 {
            continue;
        }
        let chunk = chunks.get(i).copied().unwrap_or(1).max(1);
        if chunk <= 1 || tile < chunk {
            // Either no constraint, or the budget cannot hold one chunk row. Leave
            // the unaligned tile; `chunk_amplification` will report the cost.
            continue;
        }
        // Largest whole number of chunk rows that still fits the original tile.
        let aligned = (tile / chunk) * chunk;
        if aligned >= 1 {
            tiles[i] = aligned.min(size);
        }
    }

    BatchPlan { tiles }
}

impl BatchPlan {
    /// Predicted redundant-read factor: how many times the average chunk is fetched
    /// and decompressed under this plan.
    ///
    /// `1.0` means each chunk is read exactly once (the goal). The count is exact
    /// rather than approximate — per axis it enumerates the overlapping
    /// (tile, chunk) pairs and divides by the chunk count — because two distinct
    /// effects both cost reads and a cruder model misses one of them:
    ///
    /// ```text
    ///   tile < chunk            -> ceil(chunk/tile) windows each re-read the chunk
    ///   tile > chunk, unaligned -> boundary chunks are read by two adjacent tiles
    /// ```
    ///
    /// The second is why rounding a tile down to a whole number of chunk rows is
    /// worth doing even when it makes the batch smaller. Geometry only, no I/O, so
    /// it is directly assertable in tests against measured chunk-read counts.
    pub fn chunk_amplification(&self, sizes: &[u64], chunks: &[u64]) -> f64 {
        let mut factor = 1.0_f64;
        for (i, &tile) in self.tiles.iter().enumerate() {
            let size = sizes.get(i).copied().unwrap_or(1).max(1);
            let chunk = chunks.get(i).copied().unwrap_or(1).max(1);
            let tile = tile.max(1);

            // Overlapping (tile, chunk) pairs along this axis = the number of chunk
            // reads it costs.
            let mut pairs: u128 = 0;
            let mut start = 0u64;
            while start < size {
                let end = (start + tile).min(size) - 1; // inclusive
                pairs += (end / chunk - start / chunk + 1) as u128;
                start += tile;
            }
            let n_chunks = size.div_ceil(chunk).max(1) as u128;
            factor *= pairs as f64 / n_chunks as f64;
        }
        factor
    }
}

/// Greatest common divisor, for combining chunk extents across variables.
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Per-axis chunk extent that satisfies **every** projected variable at once.
///
/// A projection can span variables with different chunk shapes, and a tile aligned
/// to one is not necessarily aligned to another. The least common multiple is the
/// smallest tile that is a whole number of chunk rows for all of them.
///
/// Capped at the axis size, since a tile can never exceed the selection, and an lcm
/// that overflows the axis is no more useful than the axis itself. Variables that
/// don't report a chunk shape are skipped rather than defaulting to 1, which would
/// otherwise silently disable alignment for everyone else.
pub fn chunk_alignment(per_var: &[&[u64]], sizes: &[u64]) -> Vec<u64> {
    (0..sizes.len())
        .map(|axis| {
            let mut acc: u64 = 1;
            for chunks in per_var {
                let c = chunks.get(axis).copied().unwrap_or(1).max(1);
                // lcm(a, c) = a / gcd(a, c) * c, computed to avoid overflow.
                acc = (acc / gcd(acc, c)).saturating_mul(c);
                if acc >= sizes[axis] {
                    return sizes[axis].max(1);
                }
            }
            acc.min(sizes[axis].max(1))
        })
        .collect()
}

// --- Phase 4.4: enforcement for the untileable single-batch path ----------------

/// Whether admission is *enforced* (reject) rather than advisory (warn), from the
/// `ZARR_MEM_ENFORCE` environment variable. Any non-empty value other than `0` /
/// `false` enables it. Off by default, so nothing rejects unless opted in.
pub fn enforce() -> bool {
    std::env::var("ZARR_MEM_ENFORCE")
        .map(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

/// Pure admission decision for the single-batch (untileable) path: a `final_rows`-row
/// batch of `row_width` bytes/row. `Ok` unless enforcement is on, a budget is set,
/// and the batch's peak exceeds it. Split out from [`admit_single_batch`] so the
/// logic is testable without touching the environment.
fn decide_single_batch(
    enforce: bool,
    budget: Option<MemoryBudget>,
    final_rows: usize,
    row_width: usize,
) -> Result<(), Infeasible> {
    if !enforce {
        return Ok(());
    }
    match budget {
        Some(budget) => {
            let peak = final_rows as u128 * row_width as u128;
            if peak <= budget.bytes as u128 {
                Ok(())
            } else {
                Err(Infeasible {
                    peak_bytes: peak,
                    budget_bytes: budget.bytes,
                })
            }
        }
        None => Ok(()),
    }
}

/// Enforce the memory budget on the single-batch path (the small-result or mixed-
/// dimensionality fallback that streaming cannot tile). Reads the budget and the
/// enforcement flag from the environment; returns [`Infeasible`] only when
/// enforcement is on and the batch would exceed the budget — the design-note Gap-2
/// case, which otherwise silently OOMs. Advisory-by-default: a no-op unless
/// `ZARR_MEM_ENFORCE` is set.
pub fn admit_single_batch(final_rows: usize, row_width: usize) -> Result<(), Infeasible> {
    decide_single_batch(enforce(), MemoryBudget::from_env(), final_rows, row_width)
}

/// Render a byte count in binary units, for readable diagnostics.
/// Format a byte count for human-facing diagnostics (KB/MB/GB).
pub fn human_bytes(n: u128) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", UNITS[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost_with_peak(peak_bytes: u128) -> ScanCost {
        ScanCost {
            rows: 0,
            touched_chunks: 0,
            bytes_read: 0,
            peak_bytes,
        }
    }

    #[test]
    fn under_budget_admits() {
        let budget = MemoryBudget::gib(1);
        assert!(admit(&cost_with_peak(500 * 1024 * 1024), &budget).is_ok());
    }

    #[test]
    fn at_budget_admits() {
        let budget = MemoryBudget::new(1000);
        assert!(admit(&cost_with_peak(1000), &budget).is_ok()); // <= is admitted
    }

    #[test]
    fn over_budget_is_infeasible_with_footprint() {
        let budget = MemoryBudget::gib(4);
        let peak = 60u128 * 1024 * 1024 * 1024; // 60 GiB
        let err = admit(&cost_with_peak(peak), &budget).unwrap_err();
        assert_eq!(err.peak_bytes, peak);
        assert_eq!(err.budget_bytes, 4 * 1024 * 1024 * 1024);
        let msg = err.to_string();
        assert!(msg.contains("60.0 GiB"), "message was: {msg}");
        assert!(msg.contains("exceeds budget"));
    }

    #[test]
    fn human_bytes_units() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1536), "1.5 KiB");
        assert_eq!(human_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }

    #[test]
    fn single_batch_advisory_by_default() {
        // Enforcement off: never rejects, even wildly over budget.
        assert!(decide_single_batch(false, Some(MemoryBudget::new(1)), 1_000_000, 8).is_ok());
    }

    #[test]
    fn single_batch_no_budget_admits() {
        assert!(decide_single_batch(true, None, 1_000_000, 8).is_ok());
    }

    #[test]
    fn single_batch_enforced_rejects_over_budget() {
        // Enforced + budget + over -> Infeasible with the exact footprint.
        let err = decide_single_batch(true, Some(MemoryBudget::new(1000)), 500, 8).unwrap_err();
        assert_eq!(err.peak_bytes, 500 * 8); // 4000 > 1000
        assert_eq!(err.budget_bytes, 1000);
    }

    #[test]
    fn single_batch_enforced_admits_within_budget() {
        assert!(decide_single_batch(true, Some(MemoryBudget::new(1_000_000)), 500, 8).is_ok());
    }

    #[test]
    fn plan_whole_selection_is_one_batch() {
        // 100 * 5 * 10 = 5000 <= 8192 target -> one batch, tiles == sizes.
        let plan = plan_streaming(&[100, 5, 10], 8192);
        assert_eq!(plan.tiles, vec![100, 5, 10]);
        assert_eq!(plan.rows_per_batch(), 5000);
        assert_eq!(plan.n_batches(&[100, 5, 10]), 1);
    }

    #[test]
    fn plan_tiles_outer_axis_like_max_steps() {
        // inner_rows = 5*10 = 50 <= target. Today's behavior: max_steps =
        // 8192/50 = 163 outer steps; inner axes whole.
        let plan = plan_streaming(&[1000, 5, 10], 8192);
        assert_eq!(plan.tiles, vec![163, 5, 10]);
        assert!(plan.rows_per_batch() <= 8192);
        assert_eq!(plan.rows_per_batch(), 163 * 50);
    }

    #[test]
    fn plan_escalates_to_inner_axis_when_inner_exceeds_target() {
        // inner_rows = 100*100 = 10000 > 8192: outer cannot help. Pivot moves to
        // axis 1 -> outer single-stepped, axis 1 tiled, axis 2 whole.
        let plan = plan_streaming(&[1000, 100, 100], 8192);
        assert_eq!(plan.tiles, vec![1, 81, 100]); // 8192/100 = 81
        assert!(plan.rows_per_batch() <= 8192);
        assert_eq!(plan.rows_per_batch(), 81 * 100);
    }

    #[test]
    fn plan_tiles_innermost_axis_when_it_alone_exceeds_target() {
        // Even one row of the two outer axes plus a whole inner axis blows the
        // target; the innermost axis itself must be tiled.
        let plan = plan_streaming(&[10, 10, 20_000], 8192);
        assert_eq!(plan.tiles, vec![1, 1, 8192]);
        assert_eq!(plan.rows_per_batch(), 8192);
    }

    // --- chunk-aligned windows ---------------------------------------------------

    #[test]
    fn aligned_plan_snaps_the_tiled_axis_to_whole_chunk_rows() {
        // 1000 outer steps, inner 50 rows, target 8192 -> unaligned tile is 163.
        // With chunks of 40 along the outer axis, 163 rounds down to 160 (4 chunk
        // rows), which still fits the target and reads each chunk exactly once.
        let plan = plan_streaming_chunked(&[1000, 5, 10], 8192, &[40, 5, 10]);
        assert_eq!(plan.tiles, vec![160, 5, 10]);
        assert!(plan.rows_per_batch() <= 8192);
        assert_eq!(plan.chunk_amplification(&[1000, 5, 10], &[40, 5, 10]), 1.0);
    }

    #[test]
    fn aligned_plan_never_exceeds_the_budget() {
        // Rounding is always downward: an aligned plan is never bigger than the
        // unaligned one, because an OOM is worse than a re-read.
        for (sizes, target, chunks) in [
            (vec![1000u64, 5, 10], 8192u64, vec![40u64, 5, 10]),
            (vec![1024, 1024], 8192, vec![256, 512]),
            (vec![3, 2, 721, 1440], 8192, vec![1, 1, 181, 720]),
            (vec![500, 3, 40], 256, vec![64, 3, 40]),
        ] {
            let base = plan_streaming(&sizes, target);
            let aligned = plan_streaming_chunked(&sizes, target, &chunks);
            assert!(
                aligned.rows_per_batch() <= base.rows_per_batch(),
                "aligned {:?} exceeded unaligned {:?}",
                aligned.tiles,
                base.tiles
            );
        }
    }

    #[test]
    fn amplification_matches_the_measured_ndvi_geometry() {
        // data/s2_ndvi_scene.zarr: [1024, 1024] chunked [256, 512], default
        // batch_size 8192. Measured: 207.02 MB read from a 6.47 MB store = 32x.
        let chunks = [256u64, 512];
        let plan = plan_streaming_chunked(&[1024, 1024], 8192, &chunks);
        assert_eq!(plan.tiles, vec![8, 1024], "budget holds < one chunk row");
        assert_eq!(plan.chunk_amplification(&[1024, 1024], &chunks), 32.0);
    }

    #[test]
    fn amplification_matches_the_measured_era5_geometry() {
        // data/era5_v3.zarr: (3, 2, 721, 1440) chunked (1, 1, 181, 720). The inner
        // axes exceed 8192 rows, so the pivot lands on `latitude` with a tile of 5
        // against a chunk of 181. Measured: 714.16 MB vs a 19.17 MB floor = 37x.
        let chunks = [1u64, 1, 181, 720];
        let plan = plan_streaming_chunked(&[3, 2, 721, 1440], 8192, &chunks);
        assert_eq!(plan.tiles, vec![1, 1, 5, 1440]);
        assert_eq!(plan.chunk_amplification(&[3, 2, 721, 1440], &chunks), 37.0);
    }

    #[test]
    fn a_budget_that_holds_a_chunk_row_removes_the_amplification() {
        // The same ERA5 geometry with room for one whole latitude chunk row
        // (181 * 1440 = 260,640 rows) drops the factor to 1.
        let chunks = [1u64, 1, 181, 720];
        let plan = plan_streaming_chunked(&[3, 2, 721, 1440], 260_640, &chunks);
        assert_eq!(plan.tiles, vec![1, 1, 181, 1440]);
        assert_eq!(plan.chunk_amplification(&[3, 2, 721, 1440], &chunks), 1.0);
    }

    #[test]
    fn ceiling_defaults_and_is_split_across_partitions() {
        // No explicit budget -> the 256 MiB default, so the pathological
        // batch_size-as-budget behaviour cannot come back by omission.
        assert_eq!(resolve_scan_ceiling(None, 1), DEFAULT_SCAN_MEMORY_LIMIT);
        // Every partition streams its own window concurrently, so the scan-wide
        // ceiling is shared, not granted to each worker.
        assert_eq!(resolve_scan_ceiling(None, 8), DEFAULT_SCAN_MEMORY_LIMIT / 8);
        // An explicit budget wins over the default, and is split the same way.
        assert_eq!(resolve_scan_ceiling(Some(1_000_000), 4), 250_000);
        // Never zero: callers divide by this.
        assert_eq!(resolve_scan_ceiling(Some(1), 8), 1);
        assert_eq!(resolve_scan_ceiling(Some(64), 0), 64);
    }

    #[test]
    fn chunk_row_bytes_is_the_knee_of_the_curve() {
        // NDVI: outer chunk 256 x full inner 1024 x 24 B/row = 6.29 MB. This is the
        // window past which more memory buys nothing — measured budgets of 4, 8 and
        // 32 MB all produced an identical 6.48 MB read.
        assert_eq!(chunk_row_bytes(&[1024, 1024], &[256, 512], 24), 6_291_456);

        // A whole-axis alignment on the outer axis needs the entire selection.
        assert_eq!(chunk_row_bytes(&[10, 20], &[10, 20], 4), 800);

        // Degenerate shapes stay sane rather than panicking or returning 0.
        assert_eq!(chunk_row_bytes(&[], &[], 8), 0);
        assert_eq!(chunk_row_bytes(&[100], &[7], 8), 56);
    }

    #[test]
    fn the_default_ceiling_clears_realistic_geometries() {
        // The claim the default rests on: 256 MiB covers one chunk row for the
        // geometries we actually read, so amplification is 1.0 out of the box.
        let ceiling = resolve_scan_ceiling(None, 1);

        // NDVI scene: [1024, 1024] chunked [256, 512], ~24 B/row.
        let need = chunk_row_bytes(&[1024, 1024], &[256, 512], 24);
        assert!(
            need < ceiling as u128,
            "NDVI needs {need}, ceiling {ceiling}"
        );

        // ARCO-ERA5 surface slab: one timestep over the full 721x1440 grid, ~20 B/row.
        let need = chunk_row_bytes(&[3, 721, 1440], &[1, 721, 1440], 20);
        assert!(
            need < ceiling as u128,
            "ERA5 needs {need}, ceiling {ceiling}"
        );
    }

    #[test]
    fn alignment_removes_boundary_straddling() {
        // The subtler half of the problem. With room for ~1M rows the unaligned plan
        // picks a 976-row tile on an axis chunked every 256 — wider than a chunk, so
        // the naive "ceil(chunk/tile)" model sees no cost at all. But 976 is not a
        // multiple of 256, so the last chunk is read by both tiles.
        let sizes = [1024u64, 1024];
        let chunks = [256u64, 512];
        let target = 1_000_000;

        let unaligned = plan_streaming(&sizes, target);
        assert_eq!(unaligned.tiles, vec![976, 1024]);
        assert_eq!(unaligned.chunk_amplification(&sizes, &chunks), 1.25);

        let aligned = plan_streaming_chunked(&sizes, target, &chunks);
        assert_eq!(
            aligned.tiles,
            vec![768, 1024],
            "976 rounds down to 3 chunk rows"
        );
        assert_eq!(aligned.chunk_amplification(&sizes, &chunks), 1.0);
    }

    #[test]
    fn alignment_never_makes_amplification_worse() {
        // The property that matters: snapping down can shrink the batch, but it must
        // never cost more reads than leaving the tile where it was.
        for (sizes, target, chunks) in [
            (vec![1024u64, 1024], 1_000_000u64, vec![256u64, 512]),
            (vec![1024, 1024], 8192, vec![256, 512]),
            (vec![3, 2, 721, 1440], 300_000, vec![1, 1, 181, 720]),
            (vec![1000, 5, 10], 8192, vec![40, 5, 10]),
            (vec![721, 1440], 50_000, vec![181, 720]),
            (vec![100], 37, vec![7]),
        ] {
            let base = plan_streaming(&sizes, target).chunk_amplification(&sizes, &chunks);
            let aligned = plan_streaming_chunked(&sizes, target, &chunks)
                .chunk_amplification(&sizes, &chunks);
            assert!(
                aligned <= base + 1e-9,
                "alignment worsened amplification: {aligned} > {base} for {sizes:?} / {chunks:?}"
            );
        }
    }

    #[test]
    fn whole_selection_needs_no_alignment() {
        // Everything fits: tiles == sizes, each chunk touched once regardless of
        // how the axes are chunked.
        let chunks = [256u64, 512];
        let plan = plan_streaming_chunked(&[1024, 1024], 8_000_000, &chunks);
        assert_eq!(plan.tiles, vec![1024, 1024]);
        assert_eq!(plan.chunk_amplification(&[1024, 1024], &chunks), 1.0);
    }

    #[test]
    fn chunk_alignment_is_the_lcm_across_variables() {
        // Two vars chunked differently on the outer axis: a tile must be a multiple
        // of both, so lcm(256, 128) = 256 and lcm(100, 150) = 300.
        let a: &[u64] = &[256, 512];
        let b: &[u64] = &[128, 512];
        assert_eq!(chunk_alignment(&[a, b], &[1024, 1024]), vec![256, 512]);

        let c: &[u64] = &[100, 4];
        let d: &[u64] = &[150, 4];
        assert_eq!(chunk_alignment(&[c, d], &[1000, 4]), vec![300, 4]);
    }

    #[test]
    fn chunk_alignment_caps_at_the_axis_size() {
        // An lcm larger than the selection is useless — a tile can never exceed the
        // axis — and capping keeps the value from overflowing on adversarial shapes.
        let a: &[u64] = &[7, 5];
        let b: &[u64] = &[11, 5];
        assert_eq!(chunk_alignment(&[a, b], &[50, 5]), vec![50, 5]);
    }

    #[test]
    fn plan_respects_target_across_shapes() {
        // Property-ish: for a spread of shapes/targets, a batch never exceeds the
        // target (unless a single innermost step already does), and tiles are
        // within extents.
        let cases = [
            (vec![7u64, 10, 10], 100u64),
            (vec![500, 3, 40], 256),
            (vec![24, 24, 24, 24], 1000),
            (vec![1, 1, 1], 8192),
        ];
        for (sizes, target) in cases {
            let plan = plan_streaming(&sizes, target);
            assert_eq!(plan.tiles.len(), sizes.len());
            for (t, s) in plan.tiles.iter().zip(&sizes) {
                assert!(*t >= 1 && *t <= *s, "tile {t} out of range for size {s}");
            }
            // rows_per_batch <= target, except the unavoidable case where the
            // innermost tile is already 1 and still... (here targets are >= 1 so ok).
            assert!(
                plan.rows_per_batch() <= target as u128 || plan.tiles.iter().all(|&t| t == 1),
                "batch {} exceeds target {} for {:?}",
                plan.rows_per_batch(),
                target,
                sizes
            );
        }
    }
}
