//! Exact group cardinality for `GROUP BY` over cube coordinates (Phase 7.1).
//!
//! Aggregate pushdown is only sound when the number of output groups is known to
//! fit in memory. Because a query's selection is an exact [`IndexSet`], the group
//! count is *computed*, not estimated: for a `GROUP BY` on coordinate axes it is
//! the selection projected onto those axes; for a periodic key (e.g. month of a
//! time axis) it is bounded by the key's period. This drives the deterministic
//! "is this aggregate pushable?" decision — never a heuristic.
//!
//! Pure and dependency-free (like the rest of Tier A) — no DataFusion here.

use super::{IndexSet, ProductSet};

/// A `GROUP BY` key expressed over the cube's index space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupKey {
    /// Group by the coordinate on cube axis `axis` — one group per distinct index
    /// selected on that axis (e.g. `GROUP BY latitude`).
    Axis(usize),
    /// Group by a periodic function of the coordinate on `axis` — at most `period`
    /// distinct groups regardless of how many indices are selected (e.g. month of a
    /// time axis is `period = 12`, hour-of-day `24`). The exact count needs the
    /// coordinate values, so structurally we bound it by `min(selected, period)`.
    Periodic { axis: usize, period: u64 },
}

impl GroupKey {
    /// The cube axis this key groups over.
    pub fn axis(&self) -> usize {
        match self {
            GroupKey::Axis(a) => *a,
            GroupKey::Periodic { axis, .. } => *axis,
        }
    }
}

/// The number of output groups a selection `sel` produces under `keys`.
///
/// - **No keys** → `1` (a global aggregate is a single group).
/// - **Empty selection** → `0`.
/// - **All-`Axis` keys** → *exact*: the selection projected onto the key axes,
///   counted with the same inclusion–exclusion the rest of Tier A uses (so it is
///   exact even for a union of boxes, not just a single Cartesian box).
/// - **Any `Periodic` key** → a safe *upper bound*: the product of each key's group
///   count, with periodic axes capped at their `period`. (Exact periodic counts need
///   the coordinate values; the bound is what the viability gate needs, and it is
///   tight for the common single-box, full-period case.)
///
/// An upper bound is exactly right for the admission decision: if the bound fits the
/// group budget, the true count does too.
pub fn group_cardinality(sel: &ProductSet, keys: &[GroupKey]) -> u128 {
    if keys.is_empty() {
        return 1;
    }
    if sel.is_empty() {
        return 0;
    }
    if keys.iter().all(|k| matches!(k, GroupKey::Axis(_))) {
        let axes: Vec<usize> = keys.iter().map(GroupKey::axis).collect();
        return project_onto(sel, &axes).cardinality();
    }
    keys.iter()
        .map(|k| match k {
            GroupKey::Axis(a) => axis_selected_count(sel, *a),
            GroupKey::Periodic { axis, period } => {
                axis_selected_count(sel, *axis).min(u128::from(*period))
            }
        })
        .product()
}

/// Distinct indices selected on a single axis = the selection projected onto it.
fn axis_selected_count(sel: &ProductSet, axis: usize) -> u128 {
    project_onto(sel, &[axis]).cardinality()
}

/// Project `sel` onto `keep_axes`, dropping every other axis. Projecting out from
/// the highest axis down keeps the lower axis indices valid as dimensions collapse.
fn project_onto(sel: &ProductSet, keep_axes: &[usize]) -> ProductSet {
    let mut cur = sel.clone();
    for axis in (0..sel.ndim()).rev() {
        if !keep_axes.contains(&axis) {
            cur = cur.project_out(axis);
        }
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::cardinality::backend::product::AxisSet;
    use std::collections::HashSet;

    // ---- brute-force reference over enumerated cells ----

    fn axis_indices(a: &AxisSet) -> Vec<u64> {
        match a {
            AxisSet::Ap {
                first,
                stride,
                count,
            } => (0..*count).map(|i| first + stride * i).collect(),
            AxisSet::Indices(v) => v.clone(),
        }
    }

    /// Every index tuple in the set (union of boxes, de-duplicated).
    fn cells(sel: &ProductSet) -> HashSet<Vec<u64>> {
        let mut out = HashSet::new();
        for bx in sel.boxes() {
            let per_axis: Vec<Vec<u64>> = bx.iter().map(axis_indices).collect();
            let mut acc: Vec<Vec<u64>> = vec![Vec::new()];
            for idxs in &per_axis {
                let mut next = Vec::new();
                for prefix in &acc {
                    for &i in idxs {
                        let mut t = prefix.clone();
                        t.push(i);
                        next.push(t);
                    }
                }
                acc = next;
            }
            out.extend(acc);
        }
        out
    }

    /// Brute-force distinct group tuples: `Axis` -> the index, `Periodic` -> index %
    /// period (a stand-in periodic function; the structural bound must dominate it).
    fn brute_groups(sel: &ProductSet, keys: &[GroupKey]) -> u128 {
        let groups: HashSet<Vec<u64>> = cells(sel)
            .iter()
            .map(|cell| {
                keys.iter()
                    .map(|k| match k {
                        GroupKey::Axis(a) => cell[*a],
                        GroupKey::Periodic { axis, period } => cell[*axis] % period,
                    })
                    .collect()
            })
            .collect();
        groups.len() as u128
    }

    fn box_of(axes: Vec<AxisSet>) -> ProductSet {
        ProductSet::single(axes)
    }

    #[test]
    fn no_keys_is_one_group() {
        let s = box_of(vec![AxisSet::interval(0, 5), AxisSet::interval(0, 3)]);
        assert_eq!(group_cardinality(&s, &[]), 1);
    }

    #[test]
    fn empty_selection_is_zero_groups() {
        let s = ProductSet::empty(2);
        assert_eq!(group_cardinality(&s, &[GroupKey::Axis(0)]), 0);
    }

    #[test]
    fn single_axis_key_counts_distinct_indices() {
        // 3-D box; group by axis 1 -> distinct indices on axis 1.
        let s = box_of(vec![
            AxisSet::interval(0, 4),
            AxisSet::interval(2, 9), // 7 distinct
            AxisSet::interval(0, 3),
        ]);
        let g = group_cardinality(&s, &[GroupKey::Axis(1)]);
        assert_eq!(g, 7);
        assert_eq!(g, brute_groups(&s, &[GroupKey::Axis(1)]));
    }

    #[test]
    fn multi_axis_key_single_box_is_product() {
        let s = box_of(vec![
            AxisSet::interval(0, 5),
            AxisSet::interval(0, 3),
            AxisSet::interval(0, 4),
        ]);
        let keys = [GroupKey::Axis(0), GroupKey::Axis(2)];
        assert_eq!(group_cardinality(&s, &keys), 5 * 4);
        assert_eq!(group_cardinality(&s, &keys), brute_groups(&s, &keys));
    }

    #[test]
    fn multi_axis_key_union_is_exact_not_overcounted() {
        // Two overlapping boxes; the projected group set is their UNION, which the
        // product-of-per-axis-counts would overcount but inclusion-exclusion nails.
        let s = ProductSet::from_boxes(
            2,
            vec![
                vec![AxisSet::interval(0, 4), AxisSet::interval(0, 4)],
                vec![AxisSet::interval(2, 6), AxisSet::interval(2, 6)],
            ],
        );
        let keys = [GroupKey::Axis(0), GroupKey::Axis(1)];
        assert_eq!(group_cardinality(&s, &keys), brute_groups(&s, &keys));
    }

    #[test]
    fn periodic_key_is_a_valid_upper_bound() {
        // 20 selected indices on axis 0, period 12 -> bound 12; brute distinct
        // residues must not exceed it (and here equals it).
        let s = box_of(vec![AxisSet::interval(0, 20), AxisSet::interval(0, 3)]);
        let keys = [GroupKey::Periodic {
            axis: 0,
            period: 12,
        }];
        let bound = group_cardinality(&s, &keys);
        assert_eq!(bound, 12);
        assert!(brute_groups(&s, &keys) <= bound);
    }

    #[test]
    fn periodic_bound_caps_at_selected_count() {
        // Only 5 indices selected but period 12 -> at most 5 groups.
        let s = box_of(vec![AxisSet::interval(0, 5)]);
        let keys = [GroupKey::Periodic {
            axis: 0,
            period: 12,
        }];
        assert_eq!(group_cardinality(&s, &keys), 5);
        assert!(brute_groups(&s, &keys) <= 5);
    }

    #[test]
    fn mixed_axis_and_periodic_bound_dominates_brute() {
        let s = box_of(vec![AxisSet::interval(0, 24), AxisSet::interval(0, 5)]);
        let keys = [
            GroupKey::Periodic {
                axis: 0,
                period: 12,
            },
            GroupKey::Axis(1),
        ];
        let bound = group_cardinality(&s, &keys);
        assert_eq!(bound, 12 * 5);
        assert!(brute_groups(&s, &keys) <= bound);
    }
}
