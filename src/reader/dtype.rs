//! Data type parsing and conversion utilities for Zarr arrays
//!
//! Handles conversion between Zarr dtype strings and Arrow DataTypes.

use arrow::datatypes::DataType;

/// Parse Zarr v2 numpy dtype string to normalized type name
/// Examples: "<i8" -> "int64", "<f4" -> "float32", "|b1" -> "bool", "<M8[ns]" -> "int64"
pub fn parse_v2_dtype(dtype: &str) -> String {
    // V2 dtype format: [<>|][type_char][byte_size] or [<>|]M8[unit] for datetime
    // < = little-endian, > = big-endian, | = not applicable
    // Type chars: i=int, u=uint, f=float, b=bool, S=string, U=unicode, M=datetime64, m=timedelta64

    // Handle datetime64 and timedelta64 (stored as int64 nanoseconds)
    // Format: <M8[ns], <M8[us], <m8[ns], etc.
    if dtype.contains("M8") || dtype.contains("m8") {
        return "int64".to_string();
    }

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

    // Handle size string (might have [unit] suffix for datetime types)
    let size_only = size_str.split('[').next().unwrap_or("8");
    let size: u32 = size_only.parse().unwrap_or(8);

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

/// Pick the narrowest signed-integer dictionary key type that can index a
/// coordinate of `cardinality` distinct values without overflow.
///
/// Dictionary keys are non-negative indices `0..cardinality`, so the largest key
/// is `cardinality - 1`. We step up Int16 → Int32 → Int64 as the coordinate grows:
///
/// - `Int16` for up to `i16::MAX + 1` (32,768) values
/// - `Int32` for up to `i32::MAX + 1` (~2.1 billion) values
/// - `Int64` beyond that
///
/// This avoids the silent `as i16` wraparound that previously panicked Arrow's
/// `DictionaryArray::new` once a coordinate exceeded 32,767 distinct values.
pub fn dictionary_key_type_for_cardinality(cardinality: usize) -> DataType {
    if cardinality <= (i16::MAX as usize) + 1 {
        DataType::Int16
    } else if cardinality <= (i32::MAX as usize) + 1 {
        DataType::Int32
    } else {
        DataType::Int64
    }
}

/// Convert Zarr dtype to Arrow Dictionary type for coordinates.
///
/// The key width is chosen from `cardinality` (the number of distinct coordinate
/// values) via [`dictionary_key_type_for_cardinality`]; the value type comes from
/// the Zarr dtype.
pub fn zarr_dtype_to_arrow_dictionary(dtype: &str, cardinality: usize) -> DataType {
    let value_type = zarr_dtype_to_arrow(dtype);
    let key_type = dictionary_key_type_for_cardinality(cardinality);
    DataType::Dictionary(Box::new(key_type), Box::new(value_type))
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

    #[test]
    fn test_dictionary_key_type_steps_int16_int32_int64() {
        // Int16 range: 0..=i16::MAX (max key 32767, i.e. 32768 distinct values)
        assert_eq!(dictionary_key_type_for_cardinality(0), DataType::Int16);
        assert_eq!(dictionary_key_type_for_cardinality(1), DataType::Int16);
        assert_eq!(dictionary_key_type_for_cardinality(32_767), DataType::Int16);
        assert_eq!(dictionary_key_type_for_cardinality(32_768), DataType::Int16);

        // Just past the Int16 ceiling -> Int32 (this is the old panic case)
        assert_eq!(dictionary_key_type_for_cardinality(32_769), DataType::Int32);
        assert_eq!(dictionary_key_type_for_cardinality(65_748), DataType::Int32);
        assert_eq!(
            dictionary_key_type_for_cardinality(i32::MAX as usize + 1),
            DataType::Int32
        );

        // Past the Int32 ceiling -> Int64
        assert_eq!(
            dictionary_key_type_for_cardinality(i32::MAX as usize + 2),
            DataType::Int64
        );
    }

    #[test]
    fn test_zarr_dtype_to_arrow_dictionary_picks_key_width() {
        // Small coordinate -> Int16 keys
        assert_eq!(
            zarr_dtype_to_arrow_dictionary("float64", 100),
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Float64))
        );
        // Large coordinate -> Int32 keys, value type preserved
        assert_eq!(
            zarr_dtype_to_arrow_dictionary("int64", 100_000),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Int64))
        );
    }
}
