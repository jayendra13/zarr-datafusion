//! Data type parsing and conversion utilities for Zarr arrays
//!
//! Handles conversion between Zarr dtype strings and Arrow DataTypes.

use arrow::datatypes::DataType;

/// Parse Zarr v2 numpy dtype string to normalized type name
/// Examples: "<i8" -> "int64", "<f4" -> "float32", "|b1" -> "bool"
pub fn parse_v2_dtype(dtype: &str) -> String {
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

/// Convert Zarr dtype string to Arrow DataType
pub fn zarr_dtype_to_arrow(dtype: &str) -> DataType {
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
pub fn zarr_dtype_to_arrow_dictionary(dtype: &str) -> DataType {
    let value_type = zarr_dtype_to_arrow(dtype);
    DataType::Dictionary(Box::new(DataType::Int16), Box::new(value_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_v2_dtype_all_types() {
        assert_eq!(parse_v2_dtype("<i1"), "int8");
        assert_eq!(parse_v2_dtype("<i2"), "int16");
        assert_eq!(parse_v2_dtype("<i4"), "int32");
        assert_eq!(parse_v2_dtype("<i8"), "int64");
        assert_eq!(parse_v2_dtype("<u1"), "uint8");
        assert_eq!(parse_v2_dtype("<u2"), "uint16");
        assert_eq!(parse_v2_dtype("<u4"), "uint32");
        assert_eq!(parse_v2_dtype("<u8"), "uint64");
        assert_eq!(parse_v2_dtype("<f2"), "float16");
        assert_eq!(parse_v2_dtype("<f4"), "float32");
        assert_eq!(parse_v2_dtype("<f8"), "float64");
        assert_eq!(parse_v2_dtype("|b1"), "bool");
    }

    #[test]
    fn test_parse_v2_dtype_big_endian() {
        assert_eq!(parse_v2_dtype(">i4"), "int32");
        assert_eq!(parse_v2_dtype(">f8"), "float64");
    }

    #[test]
    fn test_parse_v2_dtype_edge_cases() {
        assert_eq!(parse_v2_dtype(""), "float64");
        assert_eq!(parse_v2_dtype("x"), "float64");
        assert_eq!(parse_v2_dtype("<"), "float64");
        assert_eq!(parse_v2_dtype("<i"), "float64");
    }

    #[test]
    fn test_zarr_dtype_to_arrow_all_types() {
        assert_eq!(zarr_dtype_to_arrow("int8"), DataType::Int8);
        assert_eq!(zarr_dtype_to_arrow("int16"), DataType::Int16);
        assert_eq!(zarr_dtype_to_arrow("int32"), DataType::Int32);
        assert_eq!(zarr_dtype_to_arrow("int64"), DataType::Int64);
        assert_eq!(zarr_dtype_to_arrow("uint8"), DataType::UInt8);
        assert_eq!(zarr_dtype_to_arrow("uint16"), DataType::UInt16);
        assert_eq!(zarr_dtype_to_arrow("uint32"), DataType::UInt32);
        assert_eq!(zarr_dtype_to_arrow("uint64"), DataType::UInt64);
        assert_eq!(zarr_dtype_to_arrow("float16"), DataType::Float16);
        assert_eq!(zarr_dtype_to_arrow("float32"), DataType::Float32);
        assert_eq!(zarr_dtype_to_arrow("float64"), DataType::Float64);
        assert_eq!(zarr_dtype_to_arrow("bool"), DataType::Boolean);
        assert_eq!(zarr_dtype_to_arrow("unknown"), DataType::Utf8);
    }
}
