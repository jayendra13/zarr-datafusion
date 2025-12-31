//! Schema inference for Zarr v2 and v3 stores
//!
//! # Assumptions
//!
//! This module assumes a specific Zarr store structure:
//!
//! 1. **Coordinates are 1D arrays**: Any array with `shape.len() == 1` is treated as a coordinate.
//!    Examples: `time(7)`, `lat(10)`, `lon(10)`
//!
//! 2. **Data variables are nD arrays**: Arrays with `shape.len() > 1` are treated as data variables.
//!    Their dimensionality must equal the number of coordinate arrays.
//!
//! 3. **Cartesian product structure**: Data variables are assumed to be the Cartesian product
//!    of all coordinates. For coordinates `[time(7), lat(10), lon(10)]`, data variables must
//!    have shape `[7, 10, 10]` (i.e., `time × lat × lon`).
//!
//! 4. **Dimension ordering**: Coordinates are sorted alphabetically, and data variable dimensions
//!    are assumed to follow this same order.
//!
//! # Example
//!
//! ```text
//! weather.zarr/
//! ├── time/       shape: [7]           → coordinate
//! ├── lat/        shape: [10]          → coordinate
//! ├── lon/        shape: [10]          → coordinate
//! ├── temperature/ shape: [7, 10, 10]  → data variable (time × lat × lon)
//! └── humidity/    shape: [7, 10, 10]  → data variable (time × lat × lon)
//! ```

use arrow::datatypes::{Field, Schema};
use std::fs;
use std::path::Path;
use tracing::{debug, info, instrument};

use super::dtype::{parse_v2_dtype, zarr_dtype_to_arrow, zarr_dtype_to_arrow_dictionary};

/// Zarr format version
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZarrVersion {
    V2,
    V3,
}

/// Detect Zarr version by checking metadata files
pub fn detect_zarr_version(
    store_path: &str,
) -> Result<ZarrVersion, Box<dyn std::error::Error + Send + Sync>> {
    let root = Path::new(store_path);

    // Check for zarr.json (V3)
    if root.join("zarr.json").exists() {
        return Ok(ZarrVersion::V3);
    }

    // Check for .zgroup or .zarray (V2)
    if root.join(".zgroup").exists() || root.join(".zarray").exists() {
        return Ok(ZarrVersion::V2);
    }

    // Try to detect by looking at subdirectories
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if path.join("zarr.json").exists() {
                return Ok(ZarrVersion::V3);
            }
            if path.join(".zarray").exists() {
                return Ok(ZarrVersion::V2);
            }
        }
    }

    Err("Could not detect Zarr version: no metadata files found".into())
}

#[derive(Debug, Clone)]
pub struct ZarrArrayMeta {
    pub name: String,
    pub data_type: String,
    pub shape: Vec<u64>,
    /// Min/max bounds for coordinate arrays (None for data variables)
    /// Stored as (min, max) in f64 for simplicity
    pub coord_min_max: Option<(f64, f64)>,
}

impl ZarrArrayMeta {
    pub fn is_coordinate(&self) -> bool {
        self.shape.len() == 1
    }
}

/// Discovered Zarr store structure
#[derive(Debug, Clone)]
pub struct ZarrStoreMeta {
    pub coords: Vec<ZarrArrayMeta>,    // 1D arrays (sorted by name)
    pub data_vars: Vec<ZarrArrayMeta>, // nD arrays
    pub total_rows: usize,             // Product of all coordinate sizes
}

/// Discover all arrays in a Zarr store (v2 or v3)
pub fn discover_arrays(
    store_path: &str,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    let version = detect_zarr_version(store_path)?;

    match version {
        ZarrVersion::V2 => discover_arrays_v2(store_path),
        ZarrVersion::V3 => discover_arrays_v3(store_path),
    }
}

/// Discover arrays in a Zarr v2 store
fn discover_arrays_v2(
    store_path: &str,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    let root = Path::new(store_path);
    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let zarray = path.join(".zarray");
            if zarray.exists() {
                let content = fs::read_to_string(&zarray)?;
                let meta: serde_json::Value = serde_json::from_str(&content)?;

                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                let shape: Vec<u64> = meta
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                    .unwrap_or_default();

                // V2 uses numpy dtype format like "<i8", "<f4"
                let dtype_raw = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("<f8");

                let data_type = parse_v2_dtype(dtype_raw);

                arrays.push(ZarrArrayMeta {
                    name,
                    data_type,
                    shape,
                    coord_min_max: None, // Will be computed in separate_and_sort_arrays
                });
            }
        }
    }

    separate_and_sort_arrays(arrays, store_path)
}

/// Discover arrays in a Zarr v3 store
fn discover_arrays_v3(
    store_path: &str,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    let root = Path::new(store_path);
    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let zarr_json = path.join("zarr.json");
            if zarr_json.exists() {
                let content = fs::read_to_string(&zarr_json)?;
                let meta: serde_json::Value = serde_json::from_str(&content)?;

                if meta.get("node_type").and_then(|v| v.as_str()) == Some("array") {
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let shape: Vec<u64> = meta
                        .get("shape")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                        .unwrap_or_default();

                    let data_type = meta
                        .get("data_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("float64")
                        .to_string();

                    arrays.push(ZarrArrayMeta {
                        name,
                        data_type,
                        shape,
                        coord_min_max: None, // Will be computed in separate_and_sort_arrays
                    });
                }
            }
        }
    }

    separate_and_sort_arrays(arrays, store_path)
}

/// Compute min/max for a coordinate array by reading its data
/// Returns None if the computation fails (e.g., unsupported dtype, read error)
fn compute_coord_min_max(store_path: &str, coord_name: &str, data_type: &str) -> Option<(f64, f64)> {
    use zarrs::array::Array;
    use zarrs::array_subset::ArraySubset;
    use zarrs::filesystem::FilesystemStore;

    // Open store and array
    let store = FilesystemStore::new(store_path).ok()?;
    let array_path = format!("/{}", coord_name);
    let array = Array::open(store.into(), &array_path).ok()?;

    // Get the full subset (entire 1D array)
    let shape = array.shape();
    let subset = ArraySubset::new_with_start_shape(vec![0], shape.to_vec()).ok()?;

    // Read based on data type and compute min/max
    match data_type {
        "float64" => {
            let data: Vec<f64> = array.retrieve_array_subset_elements(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            Some((min, max))
        }
        "float32" => {
            let data: Vec<f32> = array.retrieve_array_subset_elements(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = data.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            Some((min, max))
        }
        "int64" => {
            let data: Vec<i64> = array.retrieve_array_subset_elements(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        "int32" => {
            let data: Vec<i32> = array.retrieve_array_subset_elements(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        "int16" => {
            let data: Vec<i16> = array.retrieve_array_subset_elements(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        "uint64" => {
            let data: Vec<u64> = array.retrieve_array_subset_elements(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        "uint32" => {
            let data: Vec<u32> = array.retrieve_array_subset_elements(&subset).ok()?;
            if data.is_empty() {
                return None;
            }
            let min = *data.iter().min()? as f64;
            let max = *data.iter().max()? as f64;
            Some((min, max))
        }
        _ => {
            debug!(data_type = %data_type, "Unsupported data type for min/max computation");
            None
        }
    }
}

/// Separate arrays into coordinates and data variables, then sort
/// Also computes min/max for coordinate arrays by reading their data
fn separate_and_sort_arrays(
    arrays: Vec<ZarrArrayMeta>,
    store_path: &str,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    // Use into_iter + partition for single-pass, zero-clone separation
    let (mut coords, mut data_vars): (Vec<_>, Vec<_>) = arrays
        .into_iter()
        .partition(|a| a.is_coordinate());

    coords.sort_by(|a, b| a.name.cmp(&b.name));
    data_vars.sort_by(|a, b| a.name.cmp(&b.name));

    // Compute min/max for each coordinate by reading the data
    for coord in &mut coords {
        if let Some(min_max) = compute_coord_min_max(store_path, &coord.name, &coord.data_type) {
            debug!(
                coord = %coord.name,
                min = min_max.0,
                max = min_max.1,
                "Computed coordinate min/max"
            );
            coord.coord_min_max = Some(min_max);
        }
    }

    // Compute total_rows = product of all coordinate sizes
    let total_rows: usize = coords
        .iter()
        .map(|c| c.shape[0] as usize)
        .product();

    Ok(ZarrStoreMeta { coords, data_vars, total_rows })
}

/// Infer Arrow schema from Zarr store metadata (v2 or v3)
/// Coordinates use DictionaryArray for memory efficiency (stores unique values once)
pub fn infer_schema(store_path: &str) -> Result<Schema, Box<dyn std::error::Error + Send + Sync>> {
    let (schema, _meta) = infer_schema_with_meta(store_path)?;
    Ok(schema)
}

/// Infer Arrow schema and return the store metadata for statistics
/// This allows caching the metadata for later use during query execution
pub fn infer_schema_with_meta(store_path: &str) -> Result<(Schema, ZarrStoreMeta), Box<dyn std::error::Error + Send + Sync>> {
    let meta = discover_arrays(store_path)?;

    let mut fields: Vec<Field> = Vec::new();

    // Coordinates use Dictionary encoding for memory efficiency
    // Instead of [0,0,0,1,1,1,2,2,2] we store {values: [0,1,2], indices: [0,0,0,1,1,1,2,2,2]}
    for coord in &meta.coords {
        fields.push(Field::new(
            &coord.name,
            zarr_dtype_to_arrow_dictionary(&coord.data_type),
            false,
        ));
    }

    // Data variables use regular arrays
    for var in &meta.data_vars {
        fields.push(Field::new(
            &var.name,
            zarr_dtype_to_arrow(&var.data_type),
            true,
        ));
    }

    Ok((Schema::new(fields), meta))
}

// =============================================================================
// Async versions for remote object stores
// =============================================================================

use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::path::Path as ObjectPath;

/// Async version of discover_arrays for remote object stores
#[instrument(level = "debug", skip_all)]
pub async fn discover_arrays_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    debug!("Detecting Zarr version");
    let version = detect_zarr_version_async(store, prefix).await?;
    info!(?version, "Zarr version detected");

    let result = match version {
        ZarrVersion::V2 => discover_arrays_v2_async(store, prefix).await,
        ZarrVersion::V3 => discover_arrays_v3_async(store, prefix).await,
    };

    if let Ok(ref meta) = result {
        info!(
            coords = meta.coords.len(),
            data_vars = meta.data_vars.len(),
            "Arrays discovered"
        );
        for coord in &meta.coords {
            debug!(name = %coord.name, shape = ?coord.shape, dtype = %coord.data_type, "Coordinate");
        }
        for var in &meta.data_vars {
            debug!(name = %var.name, shape = ?var.shape, dtype = %var.data_type, "Data variable");
        }
    }

    result
}

/// Async version of detect_zarr_version for remote object stores
pub async fn detect_zarr_version_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<ZarrVersion, Box<dyn std::error::Error + Send + Sync>> {
    use zarrs::storage::AsyncListableStorageTraits;
    use zarrs::storage::StorePrefix;

    // Check for root zarr.json (V3)
    let zarr_json_path = format!("{}/zarr.json", prefix);
    if store_key_exists(store, &zarr_json_path).await {
        return Ok(ZarrVersion::V3);
    }

    // Check for root .zgroup (V2)
    let zgroup_path = format!("{}/.zgroup", prefix);
    if store_key_exists(store, &zgroup_path).await {
        return Ok(ZarrVersion::V2);
    }

    // List directories and check first one for version detection
    // StorePrefix requires trailing slash
    let prefix_str = if prefix.as_ref().is_empty() {
        "/".to_string()
    } else {
        format!("{}/", prefix.as_ref().trim_end_matches('/'))
    };
    let store_prefix = StorePrefix::new(&prefix_str)
        .map_err(|e| format!("Invalid prefix '{}': {}", prefix_str, e))?;
    let entries = store.list_dir(&store_prefix).await
        .map_err(|e| format!("Failed to list directory: {}", e))?;

    for subdir in entries.prefixes() {
        let subdir_str = subdir.as_str().trim_end_matches('/');
        // Check for zarr.json in subdirectory (V3)
        let v3_path = format!("{}/zarr.json", subdir_str);
        if store_key_exists(store, &v3_path).await {
            return Ok(ZarrVersion::V3);
        }

        // Check for .zarray in subdirectory (V2)
        let v2_path = format!("{}/.zarray", subdir_str);
        if store_key_exists(store, &v2_path).await {
            return Ok(ZarrVersion::V2);
        }
    }

    Err("Could not detect Zarr version: no metadata files found".into())
}

/// Check if a key exists in the async store
async fn store_key_exists(store: &AsyncReadableListableStorage, key: &str) -> bool {
    use zarrs::storage::{AsyncReadableStorageTraits, StoreKey};

    let store_key = match StoreKey::new(key) {
        Ok(k) => k,
        Err(_) => return false,
    };

    match store.get(&store_key).await {
        Ok(Some(_)) => true,
        _ => false,
    }
}

/// Read a key from the async store as string
async fn store_get_string(
    store: &AsyncReadableListableStorage,
    key: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    use zarrs::storage::{AsyncReadableStorageTraits, StoreKey};

    let store_key = StoreKey::new(key)
        .map_err(|e| format!("Invalid key '{}': {}", key, e))?;

    let bytes = store
        .get(&store_key)
        .await
        .map_err(|e| format!("Failed to read '{}': {}", key, e))?
        .ok_or_else(|| format!("Key not found: {}", key))?;

    String::from_utf8(bytes.to_vec())
        .map_err(|e| format!("Invalid UTF-8 in '{}': {}", key, e).into())
}

/// Async version of discover_arrays_v2 for remote stores
async fn discover_arrays_v2_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    use zarrs::storage::{AsyncListableStorageTraits, StorePrefix};

    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    // StorePrefix requires trailing slash
    let prefix_str = if prefix.as_ref().is_empty() {
        "/".to_string()
    } else {
        format!("{}/", prefix.as_ref().trim_end_matches('/'))
    };
    let store_prefix = StorePrefix::new(&prefix_str)
        .map_err(|e| format!("Invalid prefix '{}': {}", prefix_str, e))?;
    let entries = store.list_dir(&store_prefix).await
        .map_err(|e| format!("Failed to list directory: {}", e))?;

    for subdir in entries.prefixes() {
        let subdir_str = subdir.as_str().trim_end_matches('/');
        let zarray_path = format!("{}/.zarray", subdir_str);

        // Try to read .zarray metadata
        if let Ok(content) = store_get_string(store, &zarray_path).await {
            let meta: serde_json::Value = serde_json::from_str(&content)?;

            // Extract array name from path (last component)
            let name = subdir_str
                .trim_end_matches('/')
                .rsplit('/')
                .next()
                .unwrap_or("unknown")
                .to_string();

            let shape: Vec<u64> = meta
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                .unwrap_or_default();

            let dtype_raw = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("<f8");
            let data_type = parse_v2_dtype(dtype_raw);

            arrays.push(ZarrArrayMeta {
                name,
                data_type,
                shape,
                coord_min_max: None, // Not computed for async/remote stores yet
            });
        }
    }

    separate_and_sort_arrays_async(arrays)
}

/// Async version of discover_arrays_v3 for remote stores
async fn discover_arrays_v3_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    use zarrs::storage::{AsyncListableStorageTraits, StorePrefix};

    let mut arrays: Vec<ZarrArrayMeta> = Vec::new();

    // StorePrefix requires trailing slash
    let prefix_str = if prefix.as_ref().is_empty() {
        "/".to_string()
    } else {
        format!("{}/", prefix.as_ref().trim_end_matches('/'))
    };
    let store_prefix = StorePrefix::new(&prefix_str)
        .map_err(|e| format!("Invalid prefix '{}': {}", prefix_str, e))?;
    let entries = store.list_dir(&store_prefix).await
        .map_err(|e| format!("Failed to list directory: {}", e))?;

    for subdir in entries.prefixes() {
        let subdir_str = subdir.as_str().trim_end_matches('/');
        let zarr_json_path = format!("{}/zarr.json", subdir_str);

        // Try to read zarr.json metadata
        if let Ok(content) = store_get_string(store, &zarr_json_path).await {
            let meta: serde_json::Value = serde_json::from_str(&content)?;

            // Only process arrays (not groups)
            if meta.get("node_type").and_then(|v| v.as_str()) == Some("array") {
                let name = subdir_str
                    .trim_end_matches('/')
                    .rsplit('/')
                    .next()
                    .unwrap_or("unknown")
                    .to_string();

                let shape: Vec<u64> = meta
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                    .unwrap_or_default();

                let data_type = meta
                    .get("data_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("float64")
                    .to_string();

                arrays.push(ZarrArrayMeta {
                    name,
                    data_type,
                    shape,
                    coord_min_max: None, // Not computed for async/remote stores yet
                });
            }
        }
    }

    separate_and_sort_arrays_async(arrays)
}

/// Separate arrays into coordinates and data variables (async version, no min/max computation)
fn separate_and_sort_arrays_async(
    arrays: Vec<ZarrArrayMeta>,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    // Use into_iter + partition for single-pass, zero-clone separation
    let (mut coords, mut data_vars): (Vec<_>, Vec<_>) = arrays
        .into_iter()
        .partition(|a| a.is_coordinate());

    coords.sort_by(|a, b| a.name.cmp(&b.name));
    data_vars.sort_by(|a, b| a.name.cmp(&b.name));

    // Compute total_rows = product of all coordinate sizes
    let total_rows: usize = coords
        .iter()
        .map(|c| c.shape[0] as usize)
        .product();

    Ok(ZarrStoreMeta { coords, data_vars, total_rows })
}

/// Async version of infer_schema for remote object stores
#[instrument(level = "debug", skip_all)]
pub async fn infer_schema_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<Schema, Box<dyn std::error::Error + Send + Sync>> {
    let (schema, _meta) = infer_schema_with_meta_async(store, prefix).await?;
    Ok(schema)
}

/// Async version of infer_schema that also returns the store metadata
/// This allows caching the metadata for later use during query execution
#[instrument(level = "debug", skip_all)]
pub async fn infer_schema_with_meta_async(
    store: &AsyncReadableListableStorage,
    prefix: &ObjectPath,
) -> Result<(Schema, ZarrStoreMeta), Box<dyn std::error::Error + Send + Sync>> {
    debug!("Starting async schema inference");
    let meta = discover_arrays_async(store, prefix).await?;

    let mut fields: Vec<Field> = Vec::new();

    for coord in &meta.coords {
        fields.push(Field::new(
            &coord.name,
            zarr_dtype_to_arrow_dictionary(&coord.data_type),
            false,
        ));
    }

    for var in &meta.data_vars {
        fields.push(Field::new(
            &var.name,
            zarr_dtype_to_arrow(&var.data_type),
            true,
        ));
    }

    info!(num_fields = fields.len(), "Schema inferred");
    Ok((Schema::new(fields), meta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::DataType;

    // ==================== detect_zarr_version tests ====================

    #[test]
    fn test_detect_zarr_version_v2() {
        assert_eq!(
            detect_zarr_version("data/synthetic_v2.zarr").unwrap(),
            ZarrVersion::V2
        );
        assert_eq!(
            detect_zarr_version("data/synthetic_v2_blosc.zarr").unwrap(),
            ZarrVersion::V2
        );
    }

    #[test]
    fn test_detect_zarr_version_v3() {
        assert_eq!(
            detect_zarr_version("data/synthetic_v3.zarr").unwrap(),
            ZarrVersion::V3
        );
        assert_eq!(
            detect_zarr_version("data/synthetic_v3_blosc.zarr").unwrap(),
            ZarrVersion::V3
        );
    }

    #[test]
    fn test_detect_zarr_version_error() {
        assert!(detect_zarr_version("data/nonexistent.zarr").is_err());
    }

    // ==================== ZarrArrayMeta tests ====================

    #[test]
    fn test_array_meta_is_coordinate() {
        // 1D arrays are coordinates
        let coord = ZarrArrayMeta {
            name: "lat".to_string(),
            data_type: "float64".to_string(),
            shape: vec![10],
            coord_min_max: Some((0.0, 90.0)),
        };
        assert!(coord.is_coordinate());

        // 2D and 3D arrays are NOT coordinates
        let data_2d = ZarrArrayMeta {
            name: "temp".to_string(),
            data_type: "float64".to_string(),
            shape: vec![10, 10],
            coord_min_max: None,
        };
        assert!(!data_2d.is_coordinate());

        let data_3d = ZarrArrayMeta {
            name: "temp".to_string(),
            data_type: "float64".to_string(),
            shape: vec![7, 10, 10],
            coord_min_max: None,
        };
        assert!(!data_3d.is_coordinate());
    }

    // ==================== discover_arrays tests ====================

    #[test]
    fn test_discover_arrays_v2() {
        let meta = discover_arrays("data/synthetic_v2.zarr").unwrap();

        // 3 coordinates (sorted): lat, lon, time
        assert_eq!(meta.coords.len(), 3);
        let coord_names: Vec<_> = meta.coords.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(coord_names, vec!["lat", "lon", "time"]);

        // 2 data variables (sorted): humidity, temperature
        assert_eq!(meta.data_vars.len(), 2);
        let var_names: Vec<_> = meta.data_vars.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(var_names, vec!["humidity", "temperature"]);

        // Shapes
        assert_eq!(meta.coords[0].shape, vec![10]); // lat
        assert_eq!(meta.coords[1].shape, vec![10]); // lon
        assert_eq!(meta.coords[2].shape, vec![7]); // time
        assert_eq!(meta.data_vars[0].shape, vec![7, 10, 10]); // humidity
        assert_eq!(meta.data_vars[1].shape, vec![7, 10, 10]); // temperature

        // All dtypes should be int64 (from <i8)
        for arr in meta.coords.iter().chain(meta.data_vars.iter()) {
            assert_eq!(arr.data_type, "int64");
        }
    }

    #[test]
    fn test_discover_arrays_v3() {
        let meta = discover_arrays("data/synthetic_v3.zarr").unwrap();

        // Same structure as v2
        assert_eq!(meta.coords.len(), 3);
        assert_eq!(meta.data_vars.len(), 2);

        let coord_names: Vec<_> = meta.coords.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(coord_names, vec!["lat", "lon", "time"]);

        let var_names: Vec<_> = meta.data_vars.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(var_names, vec!["humidity", "temperature"]);
    }

    // ==================== infer_schema tests ====================

    #[test]
    fn test_infer_schema_structure() {
        let schema = infer_schema("data/synthetic_v2.zarr").unwrap();

        // 5 fields: 3 coords + 2 data vars
        assert_eq!(schema.fields().len(), 5);

        let names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["lat", "lon", "time", "humidity", "temperature"]);
    }

    #[test]
    fn test_infer_schema_coord_types() {
        let schema = infer_schema("data/synthetic_v2.zarr").unwrap();

        // First 3 fields (coordinates) should be Dictionary type, non-nullable
        for i in 0..3 {
            let field = schema.field(i);
            assert!(
                matches!(field.data_type(), DataType::Dictionary(_, _)),
                "Coordinate {} should be Dictionary type",
                field.name()
            );
            assert!(
                !field.is_nullable(),
                "Coordinate {} should not be nullable",
                field.name()
            );
        }
    }

    #[test]
    fn test_infer_schema_data_var_types() {
        let schema = infer_schema("data/synthetic_v2.zarr").unwrap();

        // Last 2 fields (data vars) should be regular Int64, nullable
        for i in 3..5 {
            let field = schema.field(i);
            assert_eq!(
                field.data_type(),
                &DataType::Int64,
                "Data var {} should be Int64",
                field.name()
            );
            assert!(
                field.is_nullable(),
                "Data var {} should be nullable",
                field.name()
            );
        }
    }

    #[test]
    fn test_infer_schema_v2_v3_parity() {
        let schema_v2 = infer_schema("data/synthetic_v2.zarr").unwrap();
        let schema_v3 = infer_schema("data/synthetic_v3.zarr").unwrap();

        // Both should produce identical schemas
        assert_eq!(schema_v2.fields().len(), schema_v3.fields().len());

        for (f2, f3) in schema_v2.fields().iter().zip(schema_v3.fields().iter()) {
            assert_eq!(f2.name(), f3.name(), "Field names should match");
            assert_eq!(
                f2.data_type(),
                f3.data_type(),
                "Data types should match for {}",
                f2.name()
            );
            assert_eq!(
                f2.is_nullable(),
                f3.is_nullable(),
                "Nullability should match for {}",
                f2.name()
            );
        }
    }
}
