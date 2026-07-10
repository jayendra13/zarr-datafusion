//! Exact-cardinality reasoning for cube queries.
//!
//! A dense rectilinear cube is a function over a coordinate lattice whose extents
//! are always known, so the *shape* (cardinality) of any coordinate-structured
//! intermediate is knowable in closed form before reading a byte. This module
//! turns that observation into machinery: a query's coordinate selection becomes an
//! [`IndexSet`] over the cube's index space, and its exact cardinality, touched-tile
//! count, and (later) streaming/partition plan follow by counting — not estimating.
//!
//! **Phase 1 (this module today): the pure core.** [`CubeShape`] + the [`IndexSet`]
//! trait + a Tier-A [`ProductSet`] backend (boxes, strided cosets, unions via
//! inclusion–exclusion). No DataFusion, no FFI, fully unit-tested against
//! brute-force enumeration. Nothing here changes query behaviour yet — later phases
//! lower the scan's `CoordFilters` into an `IndexSet` (Phase 2), build a cost model
//! and admission control (Phase 3), and drive streaming/partitioning from it
//! (Phase 4+). See `docs/exact-cardinality-implementation-plan.md`.

pub mod axis;
pub mod backend;
pub mod budget;
pub mod cost;
pub mod index_set;
pub mod predicate;
pub mod rule;

pub use axis::{Axis, CubeShape};
pub use backend::product::{AxisSet, ProductSet};
pub use index_set::{AffineMap, IndexSet};
