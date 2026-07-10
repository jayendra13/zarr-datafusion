//! Deterministic scan cost model (Phase 3).
//!
//! Turns the exact shape facts (a selection's cardinality and touched-chunk count,
//! from Phases 1–2) into the concrete quantities a planner needs: how many bytes a
//! scan reads, and — the memory-safety number — how many bytes it holds resident at
//! its peak. Every value is *computed* from known extents, chunking, and dtype
//! widths; nothing is estimated.
//!
//! Phase 3 is advisory: these numbers feed a warning-level admission check
//! ([`super::budget`], not built yet), not any plan change. This file starts with
//! the cross-cutting `row_width` helper the peak-memory estimate depends on.

use arrow::datatypes::{DataType, Schema};

use super::{IndexSet, ProductSet};

/// Per-row resident byte width of a projected batch.
///
/// Models the memory one flattened row occupies. Coordinate columns are
/// dictionary-encoded, so a row costs only the *key* width — the keys buffer is
/// row-length, while the unique-values dictionary is small and shared, not paid
/// per row. Data columns cost their value dtype width. Because the key type is
/// promoted Int16 → Int32 → Int64 for large axes (see `schema_inference`), we read
/// the actual key width rather than assuming 2 bytes.
pub fn row_width(schema: &Schema) -> usize {
    schema
        .fields()
        .iter()
        .map(|f| dtype_width(f.data_type()))
        .sum()
}

/// Resident byte width of one value of `dt`. For a dictionary the per-row cost is
/// the key width (the value width is amortised across the shared dictionary).
/// Fixed-width types use their natural size; anything else falls back to a
/// conservative 8 bytes.
fn dtype_width(dt: &DataType) -> usize {
    use DataType::*;
    match dt {
        Boolean | Int8 | UInt8 => 1,
        Int16 | UInt16 | Float16 => 2,
        Int32 | UInt32 | Float32 | Date32 | Time32(_) => 4,
        Int64 | UInt64 | Float64 | Date64 | Time64(_) | Timestamp(..) | Duration(_) => 8,
        Dictionary(key, _) => dtype_width(key),
        _ => 8,
    }
}

// --- Phase 3.2: I/O bytes -------------------------------------------------------

/// Bytes in one stored chunk of a data variable: element width times the number of
/// elements per chunk. Because Zarr reads whole chunks (a query that clips a chunk
/// still fetches and decompresses all of it), this is the true I/O granule — which
/// is why I/O cost is counted in chunks, not rows.
pub fn chunk_bytes(chunk_shape: &[u64], elem_width: usize) -> u128 {
    chunk_shape.iter().map(|&c| c as u128).product::<u128>() * elem_width as u128
}

/// Bytes a scan reads from storage: whole chunks touched × per-chunk bytes.
pub fn bytes_read(touched_chunks: u128, chunk_bytes: u128) -> u128 {
    touched_chunks * chunk_bytes
}

// --- Phase 3.3: the assembled scan cost ----------------------------------------

/// Deterministic, fully-computed cost of one scan — no statistics objects, no
/// sampling. Every field follows in closed form from known extents, chunking, and
/// dtype widths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScanCost {
    /// Logical result size = the selection's cardinality.
    pub rows: u128,
    /// Distinct data-variable chunks read.
    pub touched_chunks: u128,
    /// Bytes pulled from storage (whole chunks, even when clipped).
    pub bytes_read: u128,
    /// Memory high-water mark: the widest single resident batch × `row_width`.
    /// This — not `rows` — is the number that predicts an OOM.
    pub peak_bytes: u128,
}

/// Cost a scan of `sel` that reads one data variable chunked as `chunk_shape`
/// (element width `elem_width` bytes), producing rows of `row_width` bytes.
///
/// `chunk_shape` is the **data variable's** per-axis chunking (not the coordinate
/// arrays'), aligned to `sel`'s axis order — it drives both the touched-chunk count
/// and per-chunk bytes. `inner_rows` is the product of the non-outer axis sizes (the
/// streaming floor, from [`super::predicate::SelectionStats`]). `single_batch_fallback`
/// (Phase 2.5) selects the peak model: a windowed scan's widest batch is
/// `max(batch_size, inner_rows)` rows, whereas a mixed-dimensionality fallback
/// materializes the whole selection (`rows`) in a single batch.
///
/// For a projection of several data variables, call this per variable and sum
/// `bytes_read` / `touched_chunks`; `peak_bytes` already reflects all columns via
/// `row_width`.
#[allow(clippy::too_many_arguments)]
pub fn estimate_scan_cost(
    sel: &ProductSet,
    chunk_shape: &[u64],
    elem_width: usize,
    row_width: usize,
    inner_rows: u128,
    batch_size: usize,
    single_batch_fallback: bool,
) -> ScanCost {
    let rows = sel.cardinality();
    let touched_chunks = sel.touched_tiles(chunk_shape);
    let bytes = bytes_read(touched_chunks, chunk_bytes(chunk_shape, elem_width));
    let peak = peak_bytes(
        rows,
        inner_rows,
        batch_size,
        single_batch_fallback,
        row_width,
    );

    ScanCost {
        rows,
        touched_chunks,
        bytes_read: bytes,
        peak_bytes: peak,
    }
}

/// The memory high-water mark, in bytes, independent of I/O accounting.
///
/// A windowed scan's widest resident batch is `max(batch_size, inner_rows)` rows
/// (it cannot tile below the `inner_rows` floor); a mixed-dimensionality fallback
/// materializes the whole selection (`rows`) in one batch. Times `row_width` gives
/// the peak. This is the number admission control checks, and it needs no
/// data-variable chunk geometry — only the row width and the batch model.
pub fn peak_bytes(
    rows: u128,
    inner_rows: u128,
    batch_size: usize,
    single_batch_fallback: bool,
    row_width: usize,
) -> u128 {
    let max_batch_rows = if single_batch_fallback {
        rows
    } else {
        (batch_size as u128).max(inner_rows)
    };
    max_batch_rows * row_width as u128
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, TimeUnit};

    fn dict(key: DataType) -> DataType {
        DataType::Dictionary(Box::new(key), Box::new(DataType::Float64))
    }

    #[test]
    fn row_width_dict_coords_plus_data() {
        // two Int16-key dict coords (2 + 2) + one f64 data var (8) = 12
        let schema = Schema::new(vec![
            Field::new("lat", dict(DataType::Int16), false),
            Field::new("lon", dict(DataType::Int16), false),
            Field::new("temp", DataType::Float64, true),
        ]);
        assert_eq!(row_width(&schema), 2 + 2 + 8);
    }

    #[test]
    fn row_width_promoted_key_widths() {
        // Int32-key coord (4) + Int64-key coord (8) + f32 data (4) = 16
        let schema = Schema::new(vec![
            Field::new("a", dict(DataType::Int32), false),
            Field::new("b", dict(DataType::Int64), false),
            Field::new("v", DataType::Float32, true),
        ]);
        assert_eq!(row_width(&schema), 4 + 8 + 4);
    }

    #[test]
    fn row_width_timestamp_dict_counts_key_only() {
        // CF time coord: dict with Int16 key and Timestamp values -> per row = 2.
        let ts = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let schema = Schema::new(vec![Field::new(
            "time",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(ts)),
            false,
        )]);
        assert_eq!(row_width(&schema), 2);
    }

    #[test]
    fn chunk_bytes_and_bytes_read() {
        // A [24,16,16] f64 chunk holds 6144 elements -> 49152 bytes.
        let cb = chunk_bytes(&[24, 16, 16], 8);
        assert_eq!(cb, 24 * 16 * 16 * 8);
        assert_eq!(bytes_read(10, cb), 10 * cb);
    }

    use crate::optimizer::cardinality::{AxisSet, ProductSet};

    /// A box selection over a `[24,16,16]`-chunked f64 var: time 0..48 (2 chunks),
    /// lat 0..16 (1 chunk), lon 0..32 (2 chunks).
    fn era5_selection() -> ProductSet {
        ProductSet::single(vec![
            AxisSet::interval(0, 48),
            AxisSet::interval(0, 16),
            AxisSet::interval(0, 32),
        ])
    }

    #[test]
    fn golden_cost_windowed() {
        let sel = era5_selection();
        let inner_rows = 16 * 32; // lat * lon
        let row_width = 2 + 2 + 2 + 8; // 3 dict coords + f64 data
        let cost = estimate_scan_cost(&sel, &[24, 16, 16], 8, row_width, inner_rows, 8192, false);

        assert_eq!(cost.rows, 48 * 16 * 32); // 24576
        assert_eq!(cost.touched_chunks, 4); // time=2 * lat=1 * lon=2 chunks
        assert_eq!(cost.bytes_read, 4 * 24 * 16 * 16 * 8); // 4 chunks * 49152
                                                           // windowed: max(batch_size, inner_rows) = max(8192, 512) = 8192
        assert_eq!(cost.peak_bytes, 8192 * row_width as u128);
    }

    #[test]
    fn fallback_uses_full_selection_for_peak() {
        let sel = era5_selection();
        let row_width = 14u128;
        let cost = estimate_scan_cost(&sel, &[24, 16, 16], 8, row_width as usize, 512, 8192, true);
        // Single-batch fallback: the whole selection is resident at once.
        assert_eq!(cost.peak_bytes, cost.rows * row_width);
        assert_eq!(cost.peak_bytes, 24576 * 14);
    }

    #[test]
    fn inner_rows_floor_beats_batch_size() {
        let sel = era5_selection();
        // A 4-D reduce-over-time shape: inner axes huge -> inner_rows > batch_size.
        let inner_rows = 20_000u128;
        let cost = estimate_scan_cost(&sel, &[24, 16, 16], 8, 10, inner_rows, 8192, false);
        // Windowing cannot tile below inner_rows, so the peak is inner_rows-bound.
        assert_eq!(cost.peak_bytes, inner_rows * 10);
    }
}
