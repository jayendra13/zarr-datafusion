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

use arrow::datatypes::{DataType, Field, Schema};
use std::fs;
use std::path::Path;

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

/// Parse Zarr v2 numpy dtype string to normalized type name
/// Examples: "<i8" -> "int64", "<f4" -> "float32", "|b1" -> "bool"
fn parse_v2_dtype(dtype: &str) -> String {
    // V2 dtype format: [<>|][type_char][byte_size]
    // < = little-endian, > = big-endian, | = not applicable
    // Type chars: i=int, u=uint, f=float, b=bool, S=string, U=unicode

    let chars: Vec<char> = dtype.chars().collect();
    if chars.len() < 2 {
        return "float64".to_string();
    }

    // Skip endianness prefix if present
    let (type_char, size_str) = if chars[0] == '<' || chars[0] == '>' || chars[0] == '|' {
        if chars.len() < 3 {
            return "float64".to_string();
        }
        (chars[1], &dtype[2..])
    } else {
        (chars[0], &dtype[1..])
    };

    let size: u32 = size_str.parse().unwrap_or(8);

    match type_char {
        'i' => match size {
            1 => "int8",
            2 => "int16",
            4 => "int32",
            8 => "int64",
            _ => "int64",
        },
        'u' => match size {
            1 => "uint8",
            2 => "uint16",
            4 => "uint32",
            8 => "uint64",
            _ => "uint64",
        },
        'f' => match size {
            2 => "float16",
            4 => "float32",
            8 => "float64",
            _ => "float64",
        },
        'b' => "bool",
        _ => "float64",
    }
    .to_string()
}

fn zarr_dtype_to_arrow(dtype: &str) -> DataType {
    match dtype {
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        "float16" => DataType::Float16,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "bool" => DataType::Boolean,
        _ => DataType::Utf8,
    }
}

/// Convert Zarr dtype to Arrow Dictionary type for coordinates
/// Uses Int16 keys (supports up to 32K unique values) with the value type from Zarr
fn zarr_dtype_to_arrow_dictionary(dtype: &str) -> DataType {
    let value_type = zarr_dtype_to_arrow(dtype);
    DataType::Dictionary(Box::new(DataType::Int16), Box::new(value_type))
}

#[derive(Debug, Clone)]
pub struct ZarrArrayMeta {
    pub name: String,
    pub data_type: String,
    pub shape: Vec<u64>,
}

impl ZarrArrayMeta {
    pub fn is_coordinate(&self) -> bool {
        self.shape.len() == 1
    }
}

/// Discovered Zarr store structure
#[derive(Debug)]
pub struct ZarrStoreMeta {
    pub coords: Vec<ZarrArrayMeta>,    // 1D arrays (sorted by name)
    pub data_vars: Vec<ZarrArrayMeta>, // nD arrays
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
                let dtype_raw = meta
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("<f8");

                let data_type = parse_v2_dtype(dtype_raw);

                arrays.push(ZarrArrayMeta {
                    name,
                    data_type,
                    shape,
                });
            }
        }
    }

    separate_and_sort_arrays(arrays)
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
                    });
                }
            }
        }
    }

    separate_and_sort_arrays(arrays)
}

/// Separate arrays into coordinates and data variables, then sort
fn separate_and_sort_arrays(
    arrays: Vec<ZarrArrayMeta>,
) -> Result<ZarrStoreMeta, Box<dyn std::error::Error + Send + Sync>> {
    let mut coords: Vec<_> = arrays
        .iter()
        .filter(|a| a.is_coordinate())
        .cloned()
        .collect();
    let mut data_vars: Vec<_> = arrays
        .iter()
        .filter(|a| !a.is_coordinate())
        .cloned()
        .collect();

    coords.sort_by(|a, b| a.name.cmp(&b.name));
    data_vars.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(ZarrStoreMeta { coords, data_vars })
}

/// Infer Arrow schema from Zarr store metadata (v2 or v3)
/// Coordinates use DictionaryArray for memory efficiency (stores unique values once)
pub fn infer_schema(store_path: &str) -> Result<Schema, Box<dyn std::error::Error + Send + Sync>> {
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

    Ok(Schema::new(fields))
}
