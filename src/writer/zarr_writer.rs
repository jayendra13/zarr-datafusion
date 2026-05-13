//! Zarr v3 writer (POC).
//!
//! Writes a Zarr v3 group on the local filesystem from pre-shaped inputs:
//! - 1D coordinate arrays (f32 / f64 / i64)
//! - nD data variable arrays (f32 or f64), row-major, single-chunk
//!
//! Assumes the same store structure the reader expects (see
//! [`crate::reader::schema_inference`]): coordinates are 1D, data variables
//! are nD with `dim_i == coords[i].len()` in coord order.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tracing::{debug, info, instrument};
use zarrs::array::{ArrayBuilder, DataType};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::GroupBuilder;

/// Values for a coordinate or data variable.
///
/// POC scope:
/// - Coordinates: F32, F64, I64
/// - Data variables: F32, F64 only (I64 is rejected with `UnsupportedDataVarType`)
#[derive(Debug, Clone)]
pub enum WriteValues {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I64(Vec<i64>),
}

impl WriteValues {
    fn len(&self) -> usize {
        match self {
            WriteValues::F32(v) => v.len(),
            WriteValues::F64(v) => v.len(),
            WriteValues::I64(v) => v.len(),
        }
    }

    fn dtype_name(&self) -> &'static str {
        match self {
            WriteValues::F32(_) => "f32",
            WriteValues::F64(_) => "f64",
            WriteValues::I64(_) => "i64",
        }
    }
}

/// A 1D coordinate array.
#[derive(Debug, Clone)]
pub struct CoordSpec {
    pub name: String,
    pub values: WriteValues,
}

/// An nD data variable array. `shape` is row-major and must match
/// the coord lengths in coord-order. `values.len()` must equal `product(shape)`.
#[derive(Debug, Clone)]
pub struct DataVarSpec {
    pub name: String,
    pub values: WriteValues,
    pub shape: Vec<u64>,
}

#[derive(Debug)]
pub enum WriteError {
    PathExists(PathBuf),
    EmptyCoords,
    DuplicateName(String),
    CoordRank(String),
    DataVarRankMismatch {
        name: String,
        expected: usize,
        got: usize,
    },
    DataVarShapeMismatch {
        name: String,
        dim: usize,
        coord_name: String,
        expected: u64,
        got: u64,
    },
    DataVarLengthMismatch {
        name: String,
        expected: u64,
        got: usize,
    },
    UnsupportedDataVarType {
        name: String,
        dtype: &'static str,
    },
    Io(std::io::Error),
    Zarrs(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::PathExists(p) => write!(f, "path already exists: {}", p.display()),
            WriteError::EmptyCoords => write!(f, "at least one coordinate is required"),
            WriteError::DuplicateName(n) => write!(f, "duplicate array name: {n}"),
            WriteError::CoordRank(n) => {
                write!(f, "coordinate '{n}' must be non-empty (1D, len >= 1)")
            }
            WriteError::DataVarRankMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "data var '{name}' rank mismatch: expected {expected} dims (= num coords), got {got}"
            ),
            WriteError::DataVarShapeMismatch {
                name,
                dim,
                coord_name,
                expected,
                got,
            } => write!(
                f,
                "data var '{name}' shape[{dim}] = {got}, but coord '{coord_name}' has length {expected}"
            ),
            WriteError::DataVarLengthMismatch {
                name,
                expected,
                got,
            } => write!(
                f,
                "data var '{name}' values.len() = {got}, expected {expected} (= product(shape))"
            ),
            WriteError::UnsupportedDataVarType { name, dtype } => write!(
                f,
                "data var '{name}' has dtype {dtype}; POC supports f32 and f64 only"
            ),
            WriteError::Io(e) => write!(f, "io error: {e}"),
            WriteError::Zarrs(e) => write!(f, "zarrs error: {e}"),
        }
    }
}

impl std::error::Error for WriteError {}

impl From<std::io::Error> for WriteError {
    fn from(e: std::io::Error) -> Self {
        WriteError::Io(e)
    }
}

fn zarrs_err<E: std::error::Error + Send + Sync + 'static>(e: E) -> WriteError {
    WriteError::Zarrs(Box::new(e))
}

/// Write a Zarr v3 group containing the given coords and data variables.
///
/// - `path` must not already exist (POC behavior).
/// - Chunk shape = full array shape (single chunk per array).
/// - Codec: zarrs default (raw bytes for fixed-width types).
/// - Fill values: NaN for f32/f64, 0 for i64.
#[instrument(skip(coords, data_vars), fields(n_coords = coords.len(), n_data_vars = data_vars.len()))]
pub fn write_zarr_v3(
    path: &Path,
    coords: &[CoordSpec],
    data_vars: &[DataVarSpec],
) -> Result<(), WriteError> {
    validate(path, coords, data_vars)?;

    std::fs::create_dir_all(path)?;
    let store = Arc::new(FilesystemStore::new(path).map_err(zarrs_err)?);

    // Root group metadata (zarr.json at the root).
    let group = GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(zarrs_err)?;
    group.store_metadata().map_err(zarrs_err)?;
    info!(path = %path.display(), "wrote zarr v3 group");

    for coord in coords {
        write_array_single_chunk(
            store.clone(),
            &coord.name,
            vec![coord.values.len() as u64],
            &coord.values,
        )?;
    }

    for var in data_vars {
        write_array_single_chunk(store.clone(), &var.name, var.shape.clone(), &var.values)?;
    }

    Ok(())
}

fn validate(
    path: &Path,
    coords: &[CoordSpec],
    data_vars: &[DataVarSpec],
) -> Result<(), WriteError> {
    if path.exists() {
        return Err(WriteError::PathExists(path.to_path_buf()));
    }
    if coords.is_empty() {
        return Err(WriteError::EmptyCoords);
    }

    let mut seen = std::collections::HashSet::new();
    for c in coords {
        if !seen.insert(c.name.as_str()) {
            return Err(WriteError::DuplicateName(c.name.clone()));
        }
        if c.values.len() == 0 {
            return Err(WriteError::CoordRank(c.name.clone()));
        }
    }
    for v in data_vars {
        if !seen.insert(v.name.as_str()) {
            return Err(WriteError::DuplicateName(v.name.clone()));
        }
    }

    let coord_lens: Vec<u64> = coords.iter().map(|c| c.values.len() as u64).collect();
    for var in data_vars {
        match &var.values {
            WriteValues::F32(_) | WriteValues::F64(_) => {}
            other => {
                return Err(WriteError::UnsupportedDataVarType {
                    name: var.name.clone(),
                    dtype: other.dtype_name(),
                });
            }
        }
        if var.shape.len() != coords.len() {
            return Err(WriteError::DataVarRankMismatch {
                name: var.name.clone(),
                expected: coords.len(),
                got: var.shape.len(),
            });
        }
        for (dim, (&got, &expected)) in var.shape.iter().zip(coord_lens.iter()).enumerate() {
            if got != expected {
                return Err(WriteError::DataVarShapeMismatch {
                    name: var.name.clone(),
                    dim,
                    coord_name: coords[dim].name.clone(),
                    expected,
                    got,
                });
            }
        }
        let expected_elems: u64 = var.shape.iter().product();
        if var.values.len() as u64 != expected_elems {
            return Err(WriteError::DataVarLengthMismatch {
                name: var.name.clone(),
                expected: expected_elems,
                got: var.values.len(),
            });
        }
    }
    Ok(())
}

fn write_array_single_chunk(
    store: Arc<FilesystemStore>,
    name: &str,
    shape: Vec<u64>,
    values: &WriteValues,
) -> Result<(), WriteError> {
    let chunk_shape = shape.clone();
    let zero_indices = vec![0_u64; shape.len()];
    let array_path = format!("/{name}");
    debug!(
        name,
        ?shape,
        dtype = values.dtype_name(),
        "writing zarr array"
    );

    match values {
        WriteValues::F32(data) => {
            let array = ArrayBuilder::new(shape, chunk_shape, DataType::Float32, f32::NAN)
                .build(store, &array_path)
                .map_err(zarrs_err)?;
            array.store_metadata().map_err(zarrs_err)?;
            array
                .store_chunk_elements::<f32>(&zero_indices, data)
                .map_err(zarrs_err)?;
        }
        WriteValues::F64(data) => {
            let array = ArrayBuilder::new(shape, chunk_shape, DataType::Float64, f64::NAN)
                .build(store, &array_path)
                .map_err(zarrs_err)?;
            array.store_metadata().map_err(zarrs_err)?;
            array
                .store_chunk_elements::<f64>(&zero_indices, data)
                .map_err(zarrs_err)?;
        }
        WriteValues::I64(data) => {
            let array = ArrayBuilder::new(shape, chunk_shape, DataType::Int64, 0_i64)
                .build(store, &array_path)
                .map_err(zarrs_err)?;
            array.store_metadata().map_err(zarrs_err)?;
            array
                .store_chunk_elements::<i64>(&zero_indices, data)
                .map_err(zarrs_err)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("zarr_writer_test_{name}_{pid}_{nanos}.zarr"));
        p
    }

    #[test]
    fn rejects_existing_path() {
        let p = tmp_path("existing");
        std::fs::create_dir_all(&p).unwrap();
        let err = write_zarr_v3(&p, &[], &[]).unwrap_err();
        assert!(matches!(err, WriteError::PathExists(_)));
        std::fs::remove_dir_all(&p).ok();
    }

    #[test]
    fn rejects_shape_mismatch() {
        let p = tmp_path("shape_mismatch");
        let coords = vec![CoordSpec {
            name: "x".into(),
            values: WriteValues::I64(vec![0, 1, 2]),
        }];
        let bad = vec![DataVarSpec {
            name: "v".into(),
            values: WriteValues::F32(vec![1.0, 2.0]),
            shape: vec![2],
        }];
        let err = write_zarr_v3(&p, &coords, &bad).unwrap_err();
        assert!(
            matches!(
                err,
                WriteError::DataVarShapeMismatch {
                    dim: 0,
                    expected: 3,
                    got: 2,
                    ..
                }
            ),
            "got {err:?}"
        );
        assert!(
            !p.exists(),
            "should not have created path on validation failure"
        );
    }

    #[test]
    fn rejects_i64_data_var() {
        let p = tmp_path("i64_dv");
        let coords = vec![CoordSpec {
            name: "x".into(),
            values: WriteValues::I64(vec![0, 1]),
        }];
        let bad = vec![DataVarSpec {
            name: "v".into(),
            values: WriteValues::I64(vec![10, 20]),
            shape: vec![2],
        }];
        let err = write_zarr_v3(&p, &coords, &bad).unwrap_err();
        assert!(
            matches!(err, WriteError::UnsupportedDataVarType { .. }),
            "got {err:?}"
        );
    }
}
