//! Zarr write path.
//!
//! Phase 1 of `docs/zarr-write-roundtrip-plan.md`: create a target store's
//! metadata and coordinate arrays, but no data-variable chunks. This mirrors the
//! `da.empty(...)` + `to_zarr(compute=False)` step xarray users already do by
//! hand before filling regions.
//!
//! Writing is v3-only by design; reading v2 stays supported.

pub mod skeleton;
pub mod sink;

pub use skeleton::{
    create_skeleton, CoordSpec, CoordValues, DataVarSpec, SkeletonSpec, WriteDataType,
};
pub use sink::write_batches;
