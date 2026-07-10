//! The `IndexSet` abstraction: an integer (Presburger) set over a cube's index
//! space.
//!
//! A query's coordinate selection *is* a set of integer index tuples over the
//! cube lattice. This trait is the interface for reasoning about such sets
//! exactly: how many points they contain (`cardinality`), how they compose
//! (`intersect`/`union`/`project_out`/`apply`), and which storage tiles they touch
//! (`touched_tiles`). "Cardinality estimation" becomes "shape inference" — exact
//! and compositional.
//!
//! Two backends implement this behind one interface:
//! - **Tier A** ([`super::backend::product::ProductSet`], pure Rust): axis-aligned
//!   boxes, strided/periodic cosets, and unions thereof. Covers essentially every
//!   query the engine sees today.
//! - **Tier B** (`isl`/`barvinok`, feature-gated, not built yet): genuinely
//!   axis-coupled sets — joins relating two axes, diagonals, regrids — counted by
//!   Barvinok's algorithm. Deferred to Phase 8.
//!
//! Callers program against the trait and never branch on tier.

/// An integer (Presburger) set over a cube's index space.
///
/// Coordinates are index tuples `(i0, i1, …, i_{d-1})` with `0 <= i_k < extent_k`.
/// All methods are *exact* in Tier A for the sets it can represent; the one
/// partial operation, [`IndexSet::apply`], reports its partiality via `Option`
/// rather than approximating.
pub trait IndexSet: Clone {
    /// Number of dimensions of the index space this set lives in.
    fn ndim(&self) -> usize;

    /// True iff the set contains no points.
    fn is_empty(&self) -> bool;

    /// Exact number of integer points in the set.
    ///
    /// Tier A computes this in closed form (per-axis product, with
    /// inclusion–exclusion across a union of boxes); Tier B via Barvinok
    /// lattice-point counting. `u128` because a dense cube's full cardinality can
    /// exceed `u64` (e.g. a ~173-billion-row surface cube is fine, but products of
    /// several large extents are not).
    fn cardinality(&self) -> u128;

    /// Set intersection — the AND-composition of two predicates over the same
    /// index space. Both operands must have the same `ndim`.
    fn intersect(&self, other: &Self) -> Self;

    /// Set union — the OR-composition / multi-coset. Both operands must have the
    /// same `ndim`.
    fn union(&self, other: &Self) -> Self;

    /// Existential projection: drop axis `dim`, collapsing the set onto the
    /// remaining axes. Used to model a `GROUP BY` that reduces away an axis, or to
    /// count distinct values along the surviving axes. The result has `ndim - 1`
    /// dimensions.
    fn project_out(&self, dim: usize) -> Self;

    /// Image under an affine map `A·x + b` — the operation joins, regrids, and
    /// periodic-key remaps need.
    ///
    /// Returns `None` when the backend cannot represent the image exactly. Tier A
    /// is separable (a product/union of per-axis sets) and so can only honour maps
    /// that keep the set separable — in practice the identity and axis permutation;
    /// a genuinely axis-coupling map (`i = j`) returns `None`, signalling the caller
    /// to skip the exact-count-dependent decision rather than approximate. Tier B
    /// returns `Some` for any affine map.
    fn apply(&self, map: &AffineMap) -> Option<Self>
    where
        Self: Sized;

    /// Number of distinct storage tiles the set touches, under floor-division by
    /// `tile` (one tile size per axis). Equivalently, the cardinality of the image
    /// of `x -> (i0/tile0, …, i_{d-1}/tile_{d-1})`. This is the exact chunk-touch
    /// count that drives I/O cost.
    ///
    /// `tile.len()` must equal `ndim()`.
    fn touched_tiles(&self, tile: &[u64]) -> u128;
}

/// An affine map `x -> A·x + b` over the integer index space.
///
/// `matrix` is row-major: `matrix[r]` are the coefficients of output row `r`, and
/// `offset[r]` its constant term, so `out[r] = offset[r] + Σ_c matrix[r][c]·x[c]`.
/// The map goes from `matrix[0].len()`-dimensional input to `matrix.len()`-
/// dimensional output.
///
/// Only Tier B evaluates this in general; Tier A inspects it for the separable
/// special cases (identity, permutation) and returns `None` otherwise. It is
/// defined here so the trait signature is stable across both tiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffineMap {
    pub matrix: Vec<Vec<i64>>,
    pub offset: Vec<i64>,
}

impl AffineMap {
    /// The `n`-dimensional identity map `x -> x`.
    pub fn identity(n: usize) -> Self {
        let matrix = (0..n)
            .map(|r| (0..n).map(|c| if r == c { 1 } else { 0 }).collect())
            .collect();
        Self {
            matrix,
            offset: vec![0; n],
        }
    }

    /// Input dimensionality (number of columns).
    pub fn in_dim(&self) -> usize {
        self.matrix.first().map(|row| row.len()).unwrap_or(0)
    }

    /// Output dimensionality (number of rows).
    pub fn out_dim(&self) -> usize {
        self.matrix.len()
    }

    /// True iff this is the identity map on its input dimension.
    pub fn is_identity(&self) -> bool {
        let n = self.in_dim();
        self.out_dim() == n
            && self.offset.iter().all(|&b| b == 0)
            && self
                .matrix
                .iter()
                .enumerate()
                .all(|(r, row)| row.iter().enumerate().all(|(c, &v)| v == i64::from(r == c)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_identity() {
        let id = AffineMap::identity(3);
        assert_eq!(id.in_dim(), 3);
        assert_eq!(id.out_dim(), 3);
        assert!(id.is_identity());
    }

    #[test]
    fn non_identity_detected() {
        let mut m = AffineMap::identity(2);
        m.offset[0] = 1;
        assert!(!m.is_identity());

        // A coupling map i = j is not identity.
        let coupled = AffineMap {
            matrix: vec![vec![1, 1]],
            offset: vec![0],
        };
        assert!(!coupled.is_identity());
    }
}
