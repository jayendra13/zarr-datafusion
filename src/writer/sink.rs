//! Chunk-writing sink: scatter Arrow rows into a skeleton's data-variable chunks.
//!
//! Phase 2 of docs/zarr-write-roundtrip-plan.md. The central constraint:
//!
//! > A Zarr chunk must be written as a complete n-D tile, but rows arrive in scan
//! > order.
//!
//! The resolution is to **scatter each row into a dense buffer by its grid index**
//! — no ordering requirement, so no `SortExec` and no `ORDER BY`, and a
//! row-oriented source in arbitrary order costs nothing (§5.6). Row -> index is a
//! binary search of each coordinate value into its axis; a value that is not on the
//! axis is a **loud error**, never a silent drop, because it means the source and
//! target grids disagree.
//!
//! **Phase 2 is deliberately the simplest correct cut, and these limits are known:**
//!
//! - **Single "partition": the whole array is materialised in memory, per
//!   variable.** One dense `Vec` of `product(shape)` elements is built and handed to
//!   zarrs in one `store_array_subset` call, which owns all chunk decomposition
//!   (including ragged edge chunks). Bounded, per-chunk-per-partition memory is the
//!   Phase 5 refinement (plan gap 3), not this.
//! - **No alignment concern.** With one writer, no chunk can have two owners, so the
//!   §5.4 corruption hazard cannot arise here — and Phase 2 passing proves nothing
//!   about it. That is Phase 5's job.
//! - **Plain columns only.** Coordinate columns are read as plain typed arrays, not
//!   the `DictionaryArray` fast path (§5.6, Phase 6). Build and test where nothing
//!   clever can hide a bug first.
//! - **Default fills only.** A variable with a custom `fill_value` is refused: this
//!   sink writes the *whole* array including the holes, so the hole value it writes
//!   must equal the array's declared fill, and it only knows the defaults (0 / NaN).

use std::sync::Arc;

use arrow::array::{Array as _, Float32Array, Float64Array, Int64Array};
use arrow::compute::cast;
use arrow::datatypes::DataType as ArrowType;
use arrow::record_batch::RecordBatch;
use tracing::{debug, info};
use zarrs::array::Array;
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;

use super::skeleton::{CoordValues, SkeletonSpec, WriteDataType};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A dense, row-major target buffer for one data variable, initialised to the
/// variable's fill value.
enum Buffer {
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Buffer {
    fn filled(dtype: WriteDataType, total: usize) -> Self {
        match dtype {
            WriteDataType::Int64 => Buffer::I64(vec![0i64; total]),
            WriteDataType::Float32 => Buffer::F32(vec![f32::NAN; total]),
            WriteDataType::Float64 => Buffer::F64(vec![f64::NAN; total]),
        }
    }
}

/// One coordinate column of a batch, cast to its axis's element type, paired with
/// the axis values it indexes into.
enum AxisCol<'a> {
    I64 { col: Int64Array, axis: &'a [i64] },
    F64 { col: Float64Array, axis: &'a [f64] },
}

impl AxisCol<'_> {
    /// The global axis index of the value in row `r`. `Err` if the value is null or
    /// not present on the axis — both mean the row does not belong on this grid.
    fn index_at(&self, name: &str, r: usize) -> Result<usize, BoxError> {
        match self {
            AxisCol::I64 { col, axis } => {
                if col.is_null(r) {
                    return Err(format!("null value in coordinate column '{name}'").into());
                }
                let v = col.value(r);
                axis.binary_search(&v)
                    .map_err(|_| format!("value {v} in column '{name}' is not on axis '{name}'").into())
            }
            AxisCol::F64 { col, axis } => {
                if col.is_null(r) {
                    return Err(format!("null value in coordinate column '{name}'").into());
                }
                let v = col.value(r);
                axis.binary_search_by(|a| {
                    a.partial_cmp(&v)
                        .expect("axis and coordinate values are non-NaN")
                })
                .map_err(|_| format!("value {v} in column '{name}' is not on axis '{name}'").into())
            }
        }
    }
}

/// One data-variable column of a batch, cast to the variable's element type.
enum VarCol {
    I64(Int64Array),
    F32(Float32Array),
    F64(Float64Array),
}

impl VarCol {
    /// Write the value in row `r` into `buf` at `pos`. A null leaves the fill value
    /// in place (null == missing == hole).
    fn scatter(&self, r: usize, buf: &mut Buffer, pos: usize) {
        match (self, buf) {
            (VarCol::I64(c), Buffer::I64(b)) => {
                if !c.is_null(r) {
                    b[pos] = c.value(r);
                }
            }
            (VarCol::F32(c), Buffer::F32(b)) => {
                if !c.is_null(r) {
                    b[pos] = c.value(r);
                }
            }
            (VarCol::F64(c), Buffer::F64(b)) => {
                if !c.is_null(r) {
                    b[pos] = c.value(r);
                }
            }
            // Buffer and column are built from the same WriteDataType, so the
            // variants always agree.
            _ => unreachable!("buffer and column element types diverged"),
        }
    }
}

/// Cast `col` to `target` and downcast to a concrete Arrow array.
fn cast_to(
    batch: &RecordBatch,
    name: &str,
    target: &ArrowType,
) -> Result<arrow::array::ArrayRef, BoxError> {
    let idx = batch
        .schema()
        .index_of(name)
        .map_err(|_| format!("column '{name}' not found in input"))?;
    cast(batch.column(idx), target).map_err(|e| format!("cast '{name}' to {target:?}: {e}").into())
}

/// Row-major strides for `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Write data-variable chunks into an existing skeleton at `store_path` by
/// scattering the rows of `batches` into the target grid.
///
/// `spec` must be the same spec the skeleton was created from — it supplies the
/// coordinate axis values used to map each row to its grid index. Returns the
/// number of rows written. See the module docs for the Phase 2 limits.
pub fn write_batches(
    store_path: &str,
    spec: &SkeletonSpec,
    batches: impl IntoIterator<Item = RecordBatch>,
) -> Result<u64, BoxError> {
    // A custom fill would disagree with the holes this sink writes (see module docs).
    if let Some(v) = spec.data_vars.iter().find(|v| v.fill_value.is_some()) {
        return Err(format!(
            "data variable '{}' has a custom fill_value; the Phase 2 sink writes \
             the whole array and only knows the default fills (0 / NaN)",
            v.name
        )
        .into());
    }

    let shape: Vec<usize> = spec.coords.iter().map(|c| c.values.len()).collect();
    let strides = row_major_strides(&shape);
    let total: usize = shape.iter().product();

    let store = Arc::new(FilesystemStore::new(store_path)?);

    // One dense buffer per data variable, plus the opened array to flush it into.
    let mut buffers: Vec<Buffer> = spec
        .data_vars
        .iter()
        .map(|v| Buffer::filled(v.data_type, total))
        .collect();
    let arrays: Vec<Array<FilesystemStore>> = spec
        .data_vars
        .iter()
        .map(|v| Array::open(store.clone(), &format!("/{}", v.name)))
        .collect::<Result<_, _>>()?;

    let mut rows_written: u64 = 0;

    for batch in batches {
        let n = batch.num_rows();

        // Coordinate columns, cast to their axis element type.
        let axis_cols: Vec<AxisCol> = spec
            .coords
            .iter()
            .map(|c| match &c.values {
                CoordValues::Int64(axis) => {
                    let arr = cast_to(&batch, &c.name, &ArrowType::Int64)?;
                    let col = arr
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or("cast to Int64 did not yield Int64Array")?
                        .clone();
                    Ok(AxisCol::I64 { col, axis })
                }
                CoordValues::Float64(axis) => {
                    let arr = cast_to(&batch, &c.name, &ArrowType::Float64)?;
                    let col = arr
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or("cast to Float64 did not yield Float64Array")?
                        .clone();
                    Ok(AxisCol::F64 { col, axis })
                }
            })
            .collect::<Result<_, BoxError>>()?;

        // Data-variable columns, cast to the variable's element type.
        let var_cols: Vec<VarCol> = spec
            .data_vars
            .iter()
            .map(|v| match v.data_type {
                WriteDataType::Int64 => {
                    let arr = cast_to(&batch, &v.name, &ArrowType::Int64)?;
                    Ok(VarCol::I64(
                        arr.as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or("cast to Int64 did not yield Int64Array")?
                            .clone(),
                    ))
                }
                WriteDataType::Float32 => {
                    let arr = cast_to(&batch, &v.name, &ArrowType::Float32)?;
                    Ok(VarCol::F32(
                        arr.as_any()
                            .downcast_ref::<Float32Array>()
                            .ok_or("cast to Float32 did not yield Float32Array")?
                            .clone(),
                    ))
                }
                WriteDataType::Float64 => {
                    let arr = cast_to(&batch, &v.name, &ArrowType::Float64)?;
                    Ok(VarCol::F64(
                        arr.as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or("cast to Float64 did not yield Float64Array")?
                            .clone(),
                    ))
                }
            })
            .collect::<Result<_, BoxError>>()?;

        for r in 0..n {
            let mut pos = 0usize;
            for (axis_idx, ac) in axis_cols.iter().enumerate() {
                let i = ac.index_at(&spec.coords[axis_idx].name, r)?;
                pos += i * strides[axis_idx];
            }
            for (v, vc) in var_cols.iter().enumerate() {
                vc.scatter(r, &mut buffers[v], pos);
            }
        }

        rows_written += n as u64;
    }

    // Flush each buffer as one full-array subset write; zarrs decomposes it into
    // chunks (edge chunks included).
    let full = ArraySubset::new_with_shape(shape.iter().map(|&s| s as u64).collect());
    for (v, (array, buf)) in arrays.iter().zip(buffers).enumerate() {
        match buf {
            Buffer::I64(b) => array.store_array_subset_elements(&full, &b),
            Buffer::F32(b) => array.store_array_subset_elements(&full, &b),
            Buffer::F64(b) => array.store_array_subset_elements(&full, &b),
        }
        .map_err(|e| format!("store '{}': {e}", spec.data_vars[v].name))?;
        debug!(var = %spec.data_vars[v].name, "data variable chunks written");
    }

    info!(
        path = store_path,
        rows = rows_written,
        data_vars = spec.data_vars.len(),
        "Sink wrote data variables"
    );
    Ok(rows_written)
}
