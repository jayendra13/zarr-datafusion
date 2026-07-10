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
fn human_bytes(n: u128) -> String {
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
