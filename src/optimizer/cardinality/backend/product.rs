//! Tier-A backend: separable product sets over the index space.
//!
//! This is the pure-Rust, zero-dependency counter that covers essentially every
//! query the engine sees. The representation mirrors the scan's `CoordSelection`
//! but is richer along two axes:
//!
//! - Per axis, an [`AxisSet`] is either an arithmetic progression (which subsumes
//!   a contiguous interval as stride 1, and a strided/periodic coset as stride > 1)
//!   or an explicit list of indices. AP∩AP is closed under intersection via the
//!   Chinese Remainder Theorem, so `day = 15 AND hour = 12` style coset
//!   intersections stay exact and closed-form.
//! - A [`ProductSet`] is a **union of boxes**, each box a per-axis `AxisSet`. A
//!   single box is a Cartesian product (fully separable); a union of boxes models
//!   `OR` and axis-wise unions. Cardinality and touched-tile counts over a union
//!   are computed exactly by inclusion–exclusion, which stays correct regardless of
//!   how the boxes overlap.
//!
//! What this backend deliberately cannot do is a *genuinely coupled* set (a join
//! `i = j`, a diagonal) that is not a union of products — that is Tier B's job
//! (`isl`/`barvinok`, Phase 8). [`ProductSet::apply`] returns `None` for such
//! maps rather than approximating.

use super::super::axis::CubeShape;
use super::super::index_set::{AffineMap, IndexSet};

/// A subset of one axis's indices.
///
/// `Ap { first, stride, count }` is the arithmetic progression
/// `{ first + stride*i : 0 <= i < count }` with `stride >= 1`; a contiguous
/// interval `[s, e)` is `Ap { first: s, stride: 1, count: e - s }`. `Indices` is an
/// explicit sorted, de-duplicated list for irregular sets (e.g. `IN (…)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisSet {
    Ap { first: u64, stride: u64, count: u64 },
    Indices(Vec<u64>),
}

impl AxisSet {
    /// The contiguous interval `[start, end)`. Empty if `start >= end`.
    pub fn interval(start: u64, end: u64) -> Self {
        AxisSet::Ap {
            first: start,
            stride: 1,
            count: end.saturating_sub(start),
        }
    }

    /// The arithmetic progression `{ first + stride*i : 0 <= i < count }`.
    /// `stride` is clamped to at least 1.
    pub fn coset(first: u64, stride: u64, count: u64) -> Self {
        AxisSet::Ap {
            first,
            stride: stride.max(1),
            count,
        }
    }

    /// A single index.
    pub fn point(x: u64) -> Self {
        AxisSet::Ap {
            first: x,
            stride: 1,
            count: 1,
        }
    }

    /// The empty axis set.
    pub fn empty() -> Self {
        AxisSet::Ap {
            first: 0,
            stride: 1,
            count: 0,
        }
    }

    /// An explicit set of indices; sorted and de-duplicated on construction.
    pub fn indices(mut v: Vec<u64>) -> Self {
        v.sort_unstable();
        v.dedup();
        AxisSet::Indices(v)
    }

    /// Number of indices in the set.
    pub fn len(&self) -> u64 {
        match self {
            AxisSet::Ap { count, .. } => *count,
            AxisSet::Indices(v) => v.len() as u64,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True iff `x` is a member.
    fn contains(&self, x: u64) -> bool {
        match self {
            AxisSet::Ap {
                first,
                stride,
                count,
            } => {
                x >= *first && (x - first).is_multiple_of(*stride) && (x - first) / stride < *count
            }
            AxisSet::Indices(v) => v.binary_search(&x).is_ok(),
        }
    }

    /// Intersection of two axis sets. AP∩AP stays an AP (via CRT); anything
    /// involving `Indices` collapses to `Indices`.
    fn intersect(&self, other: &AxisSet) -> AxisSet {
        use AxisSet::*;
        match (self, other) {
            (
                Ap {
                    first: af,
                    stride: as_,
                    count: an,
                },
                Ap {
                    first: bf,
                    stride: bs,
                    count: bn,
                },
            ) => ap_intersect(*af, *as_, *an, *bf, *bs, *bn),
            (Ap { .. }, Indices(v)) | (Indices(v), Ap { .. }) => {
                let ap = if matches!(self, Ap { .. }) {
                    self
                } else {
                    other
                };
                AxisSet::Indices(v.iter().copied().filter(|&x| ap.contains(x)).collect())
            }
            (Indices(a), Indices(b)) => {
                // Both are sorted; walk them together.
                let mut out = Vec::new();
                let (mut i, mut j) = (0, 0);
                while i < a.len() && j < b.len() {
                    match a[i].cmp(&b[j]) {
                        std::cmp::Ordering::Less => i += 1,
                        std::cmp::Ordering::Greater => j += 1,
                        std::cmp::Ordering::Equal => {
                            out.push(a[i]);
                            i += 1;
                            j += 1;
                        }
                    }
                }
                AxisSet::Indices(out)
            }
        }
    }

    /// Image of this set under `x -> floor(x / tile)`: the distinct tile indices it
    /// touches, as an `AxisSet`. For a plain interval this is closed-form (another
    /// interval); otherwise it is materialised into distinct tile ids (their count
    /// is bounded by the number of chunks, which is small).
    fn tile_image(&self, tile: u64) -> AxisSet {
        let tile = tile.max(1);
        if self.is_empty() {
            return AxisSet::empty();
        }
        match self {
            // Contiguous interval -> contiguous run of tiles.
            AxisSet::Ap {
                first,
                stride: 1,
                count,
            } => {
                let last = first + (count - 1);
                let t_lo = first / tile;
                let t_hi = last / tile;
                AxisSet::interval(t_lo, t_hi + 1)
            }
            // Strided AP: sweep the tile range, keep tiles the AP actually hits.
            AxisSet::Ap {
                first,
                stride,
                count,
            } => {
                let last = first + stride * (count - 1);
                let mut tiles = Vec::new();
                for t in (first / tile)..=(last / tile) {
                    if ap_hits_interval(*first, *stride, *count, t * tile, t * tile + tile) {
                        tiles.push(t);
                    }
                }
                AxisSet::Indices(tiles)
            }
            // Explicit indices: distinct floor(x/tile), single pass (input sorted).
            AxisSet::Indices(v) => {
                let mut tiles = Vec::new();
                for &x in v {
                    let t = x / tile;
                    if tiles.last() != Some(&t) {
                        tiles.push(t);
                    }
                }
                AxisSet::Indices(tiles)
            }
        }
    }
}

/// Does the AP `{ first + stride*i : 0 <= i < count }` have a member in the
/// half-open interval `[lo, hi)`?
fn ap_hits_interval(first: u64, stride: u64, count: u64, lo: u64, hi: u64) -> bool {
    if count == 0 || first >= hi {
        return false;
    }
    // Smallest i with first + stride*i >= lo.
    let i = if lo <= first {
        0
    } else {
        (lo - first).div_ceil(stride)
    };
    if i >= count {
        return false;
    }
    first + stride * i < hi
}

/// Intersection of two arithmetic progressions, returned as an `AxisSet`
/// (an AP, possibly empty). Combines the modular condition (via CRT on the two
/// strides) with the overlap of the two value ranges.
fn ap_intersect(af: u64, as_: u64, an: u64, bf: u64, bs: u64, bn: u64) -> AxisSet {
    if an == 0 || bn == 0 {
        return AxisSet::empty();
    }
    let amax = af + as_ * (an - 1);
    let bmax = bf + bs * (bn - 1);
    let lo = af.max(bf);
    let hi = amax.min(bmax);
    if lo > hi {
        return AxisSet::empty();
    }
    // Solve x ≡ af (mod as_) and x ≡ bf (mod bs).
    let (r, l) = match crt(af as i128, as_ as i128, bf as i128, bs as i128) {
        Some(v) => v,
        None => return AxisSet::empty(),
    };
    let l = l as u64;
    let r = r as u64; // in [0, l)
                      // Smallest x >= lo with x ≡ r (mod l).
    let offset = (r + l - (lo % l)) % l;
    let x0 = lo + offset;
    if x0 > hi {
        return AxisSet::empty();
    }
    let count = (hi - x0) / l + 1;
    AxisSet::Ap {
        first: x0,
        stride: l,
        count,
    }
}

/// Extended Euclid: returns `(g, x, y)` with `a*x + b*y = g = gcd(a, b)`.
fn ext_gcd(a: i128, b: i128) -> (i128, i128, i128) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (g, x, y) = ext_gcd(b, a % b);
        (g, y, x - (a / b) * y)
    }
}

/// Chinese Remainder Theorem for two congruences: solve `x ≡ a1 (mod m1)` and
/// `x ≡ a2 (mod m2)`. Returns `(residue, modulus)` with `modulus = lcm(m1, m2)`
/// and `0 <= residue < modulus`, or `None` when the congruences are incompatible.
fn crt(a1: i128, m1: i128, a2: i128, m2: i128) -> Option<(i128, i128)> {
    let (g, p, _) = ext_gcd(m1, m2);
    if (a2 - a1) % g != 0 {
        return None;
    }
    let lcm = m1 / g * m2;
    let md = m2 / g;
    let t = imod(imod((a2 - a1) / g, md) * imod(p, md), md);
    let r = imod(a1 + m1 * t, lcm);
    Some((r, lcm))
}

/// Non-negative modulo.
fn imod(a: i128, m: i128) -> i128 {
    ((a % m) + m) % m
}

/// A union of boxes over a `ndim`-dimensional index space. Each box is one
/// [`AxisSet`] per axis (`box.len() == ndim`). The empty `boxes` vector is the
/// empty set. Boxes containing an empty axis carry no points and are dropped on
/// construction, so a stored box is always non-empty and `is_empty` reduces to
/// "no boxes".
#[derive(Debug, Clone)]
pub struct ProductSet {
    ndim: usize,
    boxes: Vec<Vec<AxisSet>>,
}

impl ProductSet {
    /// The empty set over `ndim` dimensions.
    pub fn empty(ndim: usize) -> Self {
        Self {
            ndim,
            boxes: Vec::new(),
        }
    }

    /// A single box (Cartesian product of per-axis sets). Its `ndim` is the number
    /// of axis sets. Empty if any axis is empty.
    pub fn single(axes: Vec<AxisSet>) -> Self {
        let ndim = axes.len();
        Self::from_boxes(ndim, vec![axes])
    }

    /// A union of boxes over `ndim` dimensions. Every box must have `ndim` axes.
    /// Boxes with an empty axis are dropped.
    pub fn from_boxes(ndim: usize, boxes: Vec<Vec<AxisSet>>) -> Self {
        let boxes = boxes
            .into_iter()
            .filter(|b| {
                debug_assert_eq!(b.len(), ndim, "box arity must match ndim");
                !b.iter().any(AxisSet::is_empty)
            })
            .collect();
        Self { ndim, boxes }
    }

    /// The full cube: one box of full intervals `[0, extent)` per axis.
    pub fn universe(shape: &CubeShape) -> Self {
        let axes = shape
            .axes
            .iter()
            .map(|a| AxisSet::interval(0, a.extent))
            .collect();
        Self::single(axes)
    }

    /// The disjunct boxes, for diagnostics / Phase 2 lowering.
    pub fn boxes(&self) -> &[Vec<AxisSet>] {
        &self.boxes
    }
}

impl IndexSet for ProductSet {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }

    fn cardinality(&self) -> u128 {
        inclusion_exclusion(&self.boxes)
    }

    fn intersect(&self, other: &Self) -> Self {
        assert_eq!(self.ndim, other.ndim, "intersect: ndim mismatch");
        // (∪ Aᵢ) ∩ (∪ Bⱼ) = ∪ (Aᵢ ∩ Bⱼ); each box-pair intersect is per-axis.
        let mut boxes = Vec::new();
        for a in &self.boxes {
            for b in &other.boxes {
                boxes.push(box_intersect(a, b));
            }
        }
        Self::from_boxes(self.ndim, boxes)
    }

    fn union(&self, other: &Self) -> Self {
        assert_eq!(self.ndim, other.ndim, "union: ndim mismatch");
        let mut boxes = self.boxes.clone();
        boxes.extend(other.boxes.clone());
        Self::from_boxes(self.ndim, boxes)
    }

    fn project_out(&self, dim: usize) -> Self {
        assert!(dim < self.ndim, "project_out: dim out of range");
        let boxes = self
            .boxes
            .iter()
            .map(|b| {
                let mut b = b.clone();
                b.remove(dim);
                b
            })
            .collect();
        Self::from_boxes(self.ndim - 1, boxes)
    }

    fn apply(&self, map: &AffineMap) -> Option<Self> {
        // Tier A honours only maps that keep the set separable: identity and pure
        // axis permutations. A coupling map (a row referencing two inputs, a
        // non-unit coefficient, or a non-zero offset) is not representable here.
        if map.in_dim() != self.ndim {
            return None;
        }
        if map.is_identity() {
            return Some(self.clone());
        }
        let perm = as_permutation(map)?;
        let boxes = self
            .boxes
            .iter()
            .map(|b| perm.iter().map(|&src| b[src].clone()).collect())
            .collect();
        Some(Self::from_boxes(map.out_dim(), boxes))
    }

    fn touched_tiles(&self, tile: &[u64]) -> u128 {
        assert_eq!(tile.len(), self.ndim, "touched_tiles: tile arity mismatch");
        let tboxes: Vec<Vec<AxisSet>> = self
            .boxes
            .iter()
            .map(|b| b.iter().zip(tile).map(|(s, &t)| s.tile_image(t)).collect())
            .collect();
        inclusion_exclusion(&tboxes)
    }
}

/// If `map` is a pure axis permutation (each output row selects exactly one input
/// axis with coefficient 1, zero offset, and the selection is a bijection), return
/// the permutation as `out[r] = input[perm[r]]`. Otherwise `None`.
fn as_permutation(map: &AffineMap) -> Option<Vec<usize>> {
    let n = map.in_dim();
    if map.out_dim() != n || map.offset.iter().any(|&b| b != 0) {
        return None;
    }
    let mut perm = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    for row in &map.matrix {
        let mut src = None;
        for (c, &v) in row.iter().enumerate() {
            match v {
                0 => {}
                1 if src.is_none() => src = Some(c),
                _ => return None,
            }
        }
        let src = src?;
        if seen[src] {
            return None;
        }
        seen[src] = true;
        perm.push(src);
    }
    Some(perm)
}

/// Per-axis intersection of two boxes of equal arity.
fn box_intersect(a: &[AxisSet], b: &[AxisSet]) -> Vec<AxisSet> {
    a.iter().zip(b).map(|(x, y)| x.intersect(y)).collect()
}

/// Cardinality of one box: the product of its per-axis sizes. A box with an empty
/// axis has cardinality 0; a zero-dimensional box (no axes, from `project_out` down
/// to 0-D) has cardinality 1 (the single empty tuple).
fn box_card(b: &[AxisSet]) -> u128 {
    b.iter().map(|s| s.len() as u128).product()
}

/// Exact cardinality of a union of boxes by inclusion–exclusion:
/// `|∪ Bᵢ| = Σ_{∅≠S⊆boxes} (-1)^{|S|+1} |∩_{i∈S} Bᵢ|`. The intersection of any
/// subset of boxes is itself a box (per-axis AP/index intersection), so every term
/// is exact. Cost is `O(2^k)` in the number of boxes `k`, which stays small for
/// real predicates.
fn inclusion_exclusion(boxes: &[Vec<AxisSet>]) -> u128 {
    let n = boxes.len();
    if n == 0 {
        return 0;
    }
    assert!(
        n < 32,
        "inclusion-exclusion over {n} disjuncts is too expensive; \
         a query with this many OR-boxes should be handled upstream"
    );
    let mut total: i128 = 0;
    for mask in 1u32..(1u32 << n) {
        // Intersect the boxes selected by `mask`.
        let mut acc: Option<Vec<AxisSet>> = None;
        for (i, b) in boxes.iter().enumerate() {
            if mask & (1 << i) == 0 {
                continue;
            }
            acc = Some(match acc {
                None => b.clone(),
                Some(cur) => box_intersect(&cur, b),
            });
        }
        let card = box_card(acc.as_deref().unwrap_or(&[])) as i128;
        if card == 0 {
            continue;
        }
        if mask.count_ones() % 2 == 1 {
            total += card;
        } else {
            total -= card;
        }
    }
    debug_assert!(total >= 0, "inclusion-exclusion produced a negative count");
    total as u128
}

#[cfg(test)]
impl AxisSet {
    /// Explicit members, for brute-force cross-checks in tests.
    fn members(&self) -> Vec<u64> {
        match self {
            AxisSet::Ap {
                first,
                stride,
                count,
            } => (0..*count).map(|i| first + stride * i).collect(),
            AxisSet::Indices(v) => v.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::cardinality::axis::Axis;
    use std::collections::HashSet;

    /// Tiny deterministic xorshift PRNG so property tests need no external crate.
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }

    /// All integer points of a product set, by brute-force enumeration of each box
    /// (a `HashSet` so overlapping boxes are de-duplicated — the ground truth for
    /// cardinality of a union).
    fn brute_points(ps: &ProductSet) -> HashSet<Vec<u64>> {
        let mut out = HashSet::new();
        for b in ps.boxes() {
            let members: Vec<Vec<u64>> = b.iter().map(|a| a.members()).collect();
            let mut point = vec![0u64; members.len()];
            enumerate(&members, 0, &mut point, &mut out);
        }
        out
    }

    fn enumerate(
        members: &[Vec<u64>],
        axis: usize,
        point: &mut Vec<u64>,
        out: &mut HashSet<Vec<u64>>,
    ) {
        if axis == members.len() {
            out.insert(point.clone());
            return;
        }
        for &m in &members[axis] {
            point[axis] = m;
            enumerate(members, axis + 1, point, out);
        }
    }

    /// A random AxisSet within `[0, extent)`: interval, strided coset, or indices.
    fn rand_axis(rng: &mut Rng, extent: u64) -> AxisSet {
        match rng.below(3) {
            0 => {
                let a = rng.below(extent);
                let b = rng.below(extent);
                AxisSet::interval(a.min(b), a.max(b) + 1)
            }
            1 => {
                let first = rng.below(extent);
                let stride = 1 + rng.below(4);
                let max_count = (extent - first).div_ceil(stride);
                let count = rng.below(max_count) + 1;
                AxisSet::coset(first, stride, count)
            }
            _ => {
                let k = rng.below(extent) + 1;
                let v = (0..k).map(|_| rng.below(extent)).collect();
                AxisSet::indices(v)
            }
        }
    }

    fn rand_set(rng: &mut Rng, ndim: usize, extent: u64, n_boxes: usize) -> ProductSet {
        let boxes = (0..n_boxes)
            .map(|_| (0..ndim).map(|_| rand_axis(rng, extent)).collect())
            .collect();
        ProductSet::from_boxes(ndim, boxes)
    }

    #[test]
    fn cardinality_matches_bruteforce() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..4000 {
            let ndim = 1 + (rng.below(3) as usize); // 1..=3
            let extent = 4 + rng.below(9); // 4..=12
            let n_boxes = 1 + (rng.below(3) as usize); // 1..=3
            let ps = rand_set(&mut rng, ndim, extent, n_boxes);
            let expected = brute_points(&ps).len() as u128;
            assert_eq!(
                ps.cardinality(),
                expected,
                "cardinality mismatch for {:?}",
                ps.boxes()
            );
        }
    }

    #[test]
    fn touched_tiles_matches_bruteforce() {
        let mut rng = Rng::new(0xBEEF);
        for _ in 0..4000 {
            let ndim = 1 + (rng.below(3) as usize);
            let extent = 4 + rng.below(9);
            let n_boxes = 1 + (rng.below(2) as usize);
            let ps = rand_set(&mut rng, ndim, extent, n_boxes);
            let tile: Vec<u64> = (0..ndim).map(|_| 1 + rng.below(4)).collect();

            let expected: HashSet<Vec<u64>> = brute_points(&ps)
                .iter()
                .map(|p| p.iter().zip(&tile).map(|(&x, &t)| x / t).collect())
                .collect();
            assert_eq!(
                ps.touched_tiles(&tile),
                expected.len() as u128,
                "touched_tiles mismatch for {:?} tile {:?}",
                ps.boxes(),
                tile
            );
        }
    }

    #[test]
    fn intersect_matches_bruteforce() {
        let mut rng = Rng::new(0x1234);
        for _ in 0..3000 {
            let ndim = 1 + (rng.below(3) as usize);
            let extent = 4 + rng.below(9);
            let (na, nb) = (1 + rng.below(2) as usize, 1 + rng.below(2) as usize);
            let a = rand_set(&mut rng, ndim, extent, na);
            let b = rand_set(&mut rng, ndim, extent, nb);
            let inter = a.intersect(&b);

            let pa = brute_points(&a);
            let pb = brute_points(&b);
            let expected: HashSet<_> = pa.intersection(&pb).cloned().collect();
            assert_eq!(
                inter.cardinality(),
                expected.len() as u128,
                "intersect mismatch:\n a={:?}\n b={:?}",
                a.boxes(),
                b.boxes()
            );
        }
    }

    #[test]
    fn union_matches_bruteforce() {
        let mut rng = Rng::new(0x5678);
        for _ in 0..3000 {
            let ndim = 1 + (rng.below(3) as usize);
            let extent = 4 + rng.below(9);
            let (na, nb) = (1 + rng.below(2) as usize, 1 + rng.below(2) as usize);
            let a = rand_set(&mut rng, ndim, extent, na);
            let b = rand_set(&mut rng, ndim, extent, nb);
            let u = a.union(&b);

            let expected: HashSet<_> = brute_points(&a).union(&brute_points(&b)).cloned().collect();
            assert_eq!(u.cardinality(), expected.len() as u128);
        }
    }

    #[test]
    fn project_out_matches_bruteforce() {
        let mut rng = Rng::new(0x9abc);
        for _ in 0..3000 {
            let ndim = 1 + (rng.below(3) as usize);
            let extent = 4 + rng.below(9);
            let nb = 1 + rng.below(2) as usize;
            let ps = rand_set(&mut rng, ndim, extent, nb);
            let dim = rng.below(ndim as u64) as usize;
            let projected = ps.project_out(dim);
            assert_eq!(projected.ndim(), ndim - 1);

            let expected: HashSet<Vec<u64>> = brute_points(&ps)
                .iter()
                .map(|p| {
                    let mut q = p.clone();
                    q.remove(dim);
                    q
                })
                .collect();
            assert_eq!(projected.cardinality(), expected.len() as u128);
        }
    }

    #[test]
    fn intersect_with_universe_is_identity() {
        let shape = CubeShape::new(vec![
            Axis::new("t", 7, 3),
            Axis::new("y", 10, 4),
            Axis::new("x", 12, 5),
        ]);
        let universe = ProductSet::universe(&shape);
        let mut rng = Rng::new(0xDEAD);
        for _ in 0..500 {
            let ps = rand_set(&mut rng, 3, 7, 2); // within all extents
            let inter = ps.intersect(&universe);
            assert_eq!(inter.cardinality(), ps.cardinality());
        }
    }

    #[test]
    fn empty_is_absorbing() {
        let empty = ProductSet::empty(2);
        assert_eq!(empty.cardinality(), 0);
        assert!(empty.is_empty());

        let ps = ProductSet::single(vec![AxisSet::interval(0, 5), AxisSet::interval(0, 5)]);
        assert_eq!(ps.intersect(&empty).cardinality(), 0);
        assert_eq!(empty.intersect(&ps).cardinality(), 0);
        // Union with empty is identity.
        assert_eq!(ps.union(&empty).cardinality(), ps.cardinality());
    }

    #[test]
    fn contradictory_coset_intersection_is_empty() {
        // x ≡ 0 (mod 2) and x ≡ 1 (mod 2) on [0, 100): no common member.
        let evens = AxisSet::coset(0, 2, 50);
        let odds = AxisSet::coset(1, 2, 50);
        let both = ProductSet::single(vec![evens]).intersect(&ProductSet::single(vec![odds]));
        assert_eq!(both.cardinality(), 0);
    }

    #[test]
    fn coprime_coset_intersection_is_crt() {
        // x ≡ 0 (mod 3) and x ≡ 0 (mod 5) on a wide range -> multiples of 15.
        let by3 = AxisSet::coset(0, 3, 40); // 0..=117
        let by5 = AxisSet::coset(0, 5, 24); // 0..=115
        let both = ProductSet::single(vec![by3]).intersect(&ProductSet::single(vec![by5]));
        // Multiples of 15 in [0, 115]: 0,15,...,105 -> 8 values.
        assert_eq!(both.cardinality(), 8);
    }

    #[test]
    fn apply_identity_and_permutation() {
        let ps = ProductSet::single(vec![
            AxisSet::interval(0, 3),
            AxisSet::coset(1, 2, 4),
            AxisSet::indices(vec![0, 5, 9]),
        ]);
        // Identity preserves cardinality.
        let id = AffineMap::identity(3);
        assert_eq!(ps.apply(&id).unwrap().cardinality(), ps.cardinality());
        // A permutation (reverse axis order) preserves cardinality.
        let rev = AffineMap {
            matrix: vec![vec![0, 0, 1], vec![0, 1, 0], vec![1, 0, 0]],
            offset: vec![0, 0, 0],
        };
        assert_eq!(ps.apply(&rev).unwrap().cardinality(), ps.cardinality());
        // A coupling map i0 = i0 + i1 is not separable -> None.
        let coupled = AffineMap {
            matrix: vec![vec![1, 1, 0], vec![0, 1, 0], vec![0, 0, 1]],
            offset: vec![0, 0, 0],
        };
        assert!(ps.apply(&coupled).is_none());
    }

    #[test]
    fn project_to_zero_dim() {
        // A non-empty 1-D set projected to 0-D is the single empty tuple (card 1).
        let ps = ProductSet::single(vec![AxisSet::interval(2, 9)]);
        let zero = ps.project_out(0);
        assert_eq!(zero.ndim(), 0);
        assert_eq!(zero.cardinality(), 1);
        // Empty stays empty.
        let empty = ProductSet::empty(1).project_out(0);
        assert_eq!(empty.cardinality(), 0);
    }
}
