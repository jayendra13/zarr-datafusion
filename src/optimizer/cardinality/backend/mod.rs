//! Counting backends behind the [`super::IndexSet`] trait.
//!
//! - [`product`] — Tier A, pure Rust: separable product/union sets. The everyday
//!   path; no dependencies.
//! - Tier B (`isl`/`barvinok`, feature `polyhedral`) is deferred to Phase 8 and
//!   not present yet.

pub mod product;
