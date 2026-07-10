//! The logical shape of a cube: its axes, their extents, and their chunking.
//!
//! A dense rectilinear array is a function over a coordinate lattice. [`CubeShape`]
//! captures that lattice's *structure* — how many points along each axis and how
//! the axis is tiled into chunks — independent of any query. Everything the
//! cardinality module reasons about (row counts, touched chunks, streaming window
//! sizes) is derived from this shape plus a query's [`crate::optimizer::cardinality::IndexSet`].
//!
//! In Phase 1 these are constructed directly in tests. Phase 2 builds a
//! [`CubeShape`] from `ZarrArrayMeta` (extents from `shape`, chunk sizes from
//! `chunks`), so the field names here mirror that metadata.

use crate::reader::schema_inference::ZarrArrayMeta;

/// One dimension of a cube.
///
/// `extent` is the number of coordinate values along the axis (its `shape`), and
/// `chunk` is the storage tile size along the axis (its `chunks`). Both are in
/// units of array indices, not coordinate values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Axis {
    pub name: String,
    /// Number of indices along the axis, `0..extent`.
    pub extent: u64,
    /// Chunk (tile) size along the axis. Must be `>= 1`.
    pub chunk: u64,
}

impl Axis {
    /// Build an axis. `chunk` is clamped to at least 1 so tile arithmetic never
    /// divides by zero (a chunk size of 0 is meaningless for a non-empty axis).
    pub fn new(name: impl Into<String>, extent: u64, chunk: u64) -> Self {
        Self {
            name: name.into(),
            extent,
            chunk: chunk.max(1),
        }
    }

    /// Number of chunks covering the full extent (`ceil(extent / chunk)`).
    pub fn n_chunks(&self) -> u64 {
        self.extent.div_ceil(self.chunk)
    }
}

/// The full lattice: an ordered list of axes.
///
/// Axis order is significant — it is the same order the scan flattens in
/// (outer-axis-most-significant), so axis 0 is the outer / streaming axis.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CubeShape {
    pub axes: Vec<Axis>,
}

impl CubeShape {
    pub fn new(axes: Vec<Axis>) -> Self {
        Self { axes }
    }

    /// Build a shape from the 1-D coordinate arrays' metadata, in the order given.
    ///
    /// Each coordinate contributes one axis: `extent` from its (1-D) `shape`, and
    /// `chunk` from its `chunks` — falling back to the full extent (a single chunk)
    /// when chunk metadata is absent. The caller supplies the coordinate metas
    /// already ordered to match the order the scan flattens in, so axis 0 here is
    /// the scan's outer/streaming axis.
    pub fn from_coord_metas(coords: &[ZarrArrayMeta]) -> Self {
        let axes = coords
            .iter()
            .map(|c| {
                let extent = c.shape.first().copied().unwrap_or(0);
                let chunk = c
                    .chunks
                    .as_ref()
                    .and_then(|ch| ch.first().copied())
                    .unwrap_or(extent);
                Axis::new(c.name.clone(), extent, chunk)
            })
            .collect();
        Self { axes }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.axes.len()
    }

    /// Per-axis extents, in axis order.
    pub fn extents(&self) -> Vec<u64> {
        self.axes.iter().map(|a| a.extent).collect()
    }

    /// Per-axis chunk sizes, in axis order — the `tile` argument for
    /// [`crate::optimizer::cardinality::IndexSet::touched_tiles`].
    pub fn chunk_shape(&self) -> Vec<u64> {
        self.axes.iter().map(|a| a.chunk).collect()
    }

    /// Total points in the full (unfiltered) cube: the product of all extents.
    /// This is the cardinality of a `SELECT *` with no predicate.
    pub fn full_cardinality(&self) -> u128 {
        self.axes.iter().map(|a| a.extent as u128).product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_chunks_rounds_up() {
        assert_eq!(Axis::new("t", 10, 4).n_chunks(), 3);
        assert_eq!(Axis::new("t", 8, 4).n_chunks(), 2);
        assert_eq!(Axis::new("t", 1, 4).n_chunks(), 1);
        assert_eq!(Axis::new("t", 0, 4).n_chunks(), 0);
    }

    #[test]
    fn chunk_clamped_to_one() {
        assert_eq!(Axis::new("t", 5, 0).chunk, 1);
        assert_eq!(Axis::new("t", 5, 0).n_chunks(), 5);
    }

    #[test]
    fn from_coord_metas_builds_axes() {
        let meta = |name: &str, extent: u64, chunk: Option<u64>| ZarrArrayMeta {
            name: name.into(),
            data_type: "float64".into(),
            shape: vec![extent],
            chunks: chunk.map(|c| vec![c]),
            coord_min_max: None,
            cf_time_attrs: None,
            dimensions: None,
        };
        let shape = CubeShape::from_coord_metas(&[
            meta("time", 100, Some(24)),
            meta("lat", 50, Some(16)),
            meta("lon", 60, None), // no chunk info -> single chunk of full extent
        ]);
        assert_eq!(shape.extents(), vec![100, 50, 60]);
        assert_eq!(shape.chunk_shape(), vec![24, 16, 60]);
        assert_eq!(shape.axes[0].name, "time");
    }

    #[test]
    fn shape_aggregates() {
        let shape = CubeShape::new(vec![
            Axis::new("time", 7, 3),
            Axis::new("lat", 10, 5),
            Axis::new("lon", 10, 5),
        ]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.extents(), vec![7, 10, 10]);
        assert_eq!(shape.chunk_shape(), vec![3, 5, 5]);
        assert_eq!(shape.full_cardinality(), 700);
    }
}
