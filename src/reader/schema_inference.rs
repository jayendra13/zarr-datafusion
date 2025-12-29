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
                let dtype_raw = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("<f8");

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

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== parse_v2_dtype tests ====================

    #[test]
    fn test_parse_v2_dtype_all_types() {
        let cases = [
            // Integers
            ("<i8", "int64"),
            ("<i4", "int32"),
            ("<i2", "int16"),
            ("<i1", "int8"),
            // Unsigned integers
            ("<u8", "uint64"),
            ("<u4", "uint32"),
            ("<u2", "uint16"),
            ("<u1", "uint8"),
            // Floats
            ("<f8", "float64"),
            ("<f4", "float32"),
            ("<f2", "float16"),
            // Bool
            ("|b1", "bool"),
        ];
        for (input, expected) in cases {
            assert_eq!(parse_v2_dtype(input), expected, "Failed for {}", input);
        }
    }

    #[test]
    fn test_parse_v2_dtype_big_endian() {
        assert_eq!(parse_v2_dtype(">i8"), "int64");
        assert_eq!(parse_v2_dtype(">f4"), "float32");
    }

    #[test]
    fn test_parse_v2_dtype_edge_cases() {
        // Malformed/short strings default to float64
        assert_eq!(parse_v2_dtype("x"), "float64");
        assert_eq!(parse_v2_dtype(""), "float64");
        // Unknown type char defaults to float64
        assert_eq!(parse_v2_dtype("<x8"), "float64");
    }

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

    // ==================== zarr_dtype_to_arrow tests ====================

    #[test]
    fn test_zarr_dtype_to_arrow_all_types() {
        let cases = [
            ("int8", DataType::Int8),
            ("int16", DataType::Int16),
            ("int32", DataType::Int32),
            ("int64", DataType::Int64),
            ("uint8", DataType::UInt8),
            ("uint16", DataType::UInt16),
            ("uint32", DataType::UInt32),
            ("uint64", DataType::UInt64),
            ("float16", DataType::Float16),
            ("float32", DataType::Float32),
            ("float64", DataType::Float64),
            ("bool", DataType::Boolean),
            ("unknown", DataType::Utf8), // fallback
        ];
        for (input, expected) in cases {
            assert_eq!(zarr_dtype_to_arrow(input), expected, "Failed for {}", input);
        }
    }

    // ==================== ZarrArrayMeta tests ====================

    #[test]
    fn test_array_meta_is_coordinate() {
        // 1D arrays are coordinates
        let coord = ZarrArrayMeta {
            name: "lat".to_string(),
            data_type: "float64".to_string(),
            shape: vec![10],
        };
        assert!(coord.is_coordinate());

        // 2D and 3D arrays are NOT coordinates
        let data_2d = ZarrArrayMeta {
            name: "temp".to_string(),
            data_type: "float64".to_string(),
            shape: vec![10, 10],
        };
        assert!(!data_2d.is_coordinate());

        let data_3d = ZarrArrayMeta {
            name: "temp".to_string(),
            data_type: "float64".to_string(),
            shape: vec![7, 10, 10],
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
