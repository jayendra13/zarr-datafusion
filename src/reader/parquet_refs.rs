//! Parquet reference file reader for VirtualiZarr stores.
//!
//! VirtualiZarr stores use Parquet files to store chunk references that point
//! to byte ranges in source files (NetCDF, GRIB, etc.) on cloud storage.

use arrow::array::{Array, BinaryArray, Int64Array, StringArray};
use arrow::datatypes::Int32Type;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;

/// A reference to a chunk's data location.
#[derive(Debug, Clone)]
pub struct ChunkRef {
    /// URL/path to source file (e.g., s3://bucket/path/file.nc)
    pub path: String,
    /// Byte offset within the source file
    pub offset: u64,
    /// Number of bytes to read
    pub size: u64,
    /// Inline data (for small arrays like coordinates)
    pub raw: Option<Vec<u8>>,
}

/// Collection of chunk references loaded from Parquet files.
#[derive(Debug)]
pub struct ParquetRefs {
    /// References indexed by flattened chunk index
    refs: Vec<ChunkRef>,
}

impl ParquetRefs {
    /// Load chunk references from all refs.*.parq files for an array.
    ///
    /// # Arguments
    /// * `store_path` - Path to the VirtualiZarr store root
    /// * `array_name` - Name of the array (e.g., "t2", "latitude")
    pub fn load(store_path: &str, array_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let pattern = format!("{}/{}/refs.*.parq", store_path, array_name);
        let mut refs = Vec::new();

        // Collect and sort files to ensure correct order (refs.0.parq, refs.1.parq, ...)
        let mut files: Vec<_> = glob::glob(&pattern)?.filter_map(Result::ok).collect();
        files.sort();

        if files.is_empty() {
            return Err(format!(
                "No parquet reference files found for array '{}' at {}",
                array_name, store_path
            )
            .into());
        }

        for file_path in files {
            Self::load_parquet_file(&file_path, &mut refs)?;
        }

        Ok(Self { refs })
    }

    /// Load references from a single Parquet file.
    fn load_parquet_file(
        path: &Path,
        refs: &mut Vec<ChunkRef>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        for batch_result in reader {
            let batch = batch_result?;

            // Expected columns: path, offset, size, raw
            let paths = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .or_else(|| {
                    // Handle dictionary-encoded categorical columns
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<arrow::array::DictionaryArray<Int32Type>>()
                        .and_then(|dict| dict.values().as_any().downcast_ref::<StringArray>())
                });

            let offsets = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or("Expected Int64 for offset column")?;

            let sizes = batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or("Expected Int64 for size column")?;

            let raws = batch.column(3).as_any().downcast_ref::<BinaryArray>();

            // Handle dictionary-encoded path column
            let path_col = batch.column(0);
            let is_dict = path_col
                .as_any()
                .is::<arrow::array::DictionaryArray<Int32Type>>();

            for i in 0..batch.num_rows() {
                let path_str = if is_dict {
                    let dict = path_col
                        .as_any()
                        .downcast_ref::<arrow::array::DictionaryArray<Int32Type>>()
                        .unwrap();
                    let values = dict
                        .values()
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap();
                    let key = dict.keys().value(i);
                    values.value(key as usize).to_string()
                } else if let Some(paths) = paths {
                    paths.value(i).to_string()
                } else {
                    return Err("Could not read path column".into());
                };

                let raw_data = raws.and_then(|r| {
                    if r.is_valid(i) {
                        Some(r.value(i).to_vec())
                    } else {
                        None
                    }
                });

                refs.push(ChunkRef {
                    path: path_str,
                    offset: offsets.value(i) as u64,
                    size: sizes.value(i) as u64,
                    raw: raw_data,
                });
            }
        }

        Ok(())
    }

    /// Get a chunk reference by its flattened index.
    pub fn get(&self, chunk_index: usize) -> Option<&ChunkRef> {
        self.refs.get(chunk_index)
    }

    /// Total number of chunk references.
    pub fn len(&self) -> usize {
        self.refs.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }
}

/// Convert chunk indices to a flat index (C-order/row-major).
///
/// # Arguments
/// * `indices` - Chunk indices for each dimension (e.g., [0, 5, 0, 0])
/// * `chunks_per_dim` - Number of chunks in each dimension (e.g., [3914, 41, 1, 1])
pub fn indices_to_flat_index(indices: &[u64], chunks_per_dim: &[u64]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;

    // Iterate in reverse for C-order (row-major)
    for i in (0..indices.len()).rev() {
        flat += indices[i] as usize * stride;
        stride *= chunks_per_dim[i] as usize;
    }

    flat
}

/// Parse a Zarr v2 chunk key into array name and chunk indices.
///
/// # Example
/// ```ignore
/// parse_chunk_key("t2/0.5.0.0") -> ("t2", [0, 5, 0, 0])
/// ```
pub fn parse_chunk_key(key: &str) -> Result<(String, Vec<u64>), Box<dyn std::error::Error>> {
    let key = key.trim_start_matches('/');
    let parts: Vec<&str> = key.split('/').collect();

    if parts.len() < 2 {
        return Err(format!("Invalid chunk key format: {}", key).into());
    }

    let array_name = parts[0].to_string();
    let indices: Result<Vec<u64>, _> = parts[1].split('.').map(|s| s.parse::<u64>()).collect();

    Ok((array_name, indices?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indices_to_flat_index() {
        // For t2 with chunks_per_dim [3914, 41, 1, 1]
        let chunks_per_dim = vec![3914, 41, 1, 1];

        // First chunk
        assert_eq!(indices_to_flat_index(&[0, 0, 0, 0], &chunks_per_dim), 0);

        // Second time step
        assert_eq!(indices_to_flat_index(&[0, 1, 0, 0], &chunks_per_dim), 1);

        // Fifth time step
        assert_eq!(indices_to_flat_index(&[0, 5, 0, 0], &chunks_per_dim), 5);

        // First chunk of second init_time
        assert_eq!(indices_to_flat_index(&[1, 0, 0, 0], &chunks_per_dim), 41);
    }

    #[test]
    fn test_parse_chunk_key() {
        let (name, indices) = parse_chunk_key("t2/0.5.0.0").unwrap();
        assert_eq!(name, "t2");
        assert_eq!(indices, vec![0, 5, 0, 0]);

        let (name, indices) = parse_chunk_key("/latitude/0").unwrap();
        assert_eq!(name, "latitude");
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_load_parquet_refs() {
        // Test loading refs from local VirtualiZarr store
        let refs = ParquetRefs::load("data/FOUR_v200_GFS.parq", "t2").unwrap();

        // Should have loaded refs (t2 has 160,474 chunks = 3914 * 41)
        assert!(!refs.is_empty());
        println!("Loaded {} refs for t2", refs.len());

        // First ref should point to S3
        let first = refs.get(0).unwrap();
        assert!(first.path.starts_with("s3://"));
        assert!(first.size > 0);
        println!(
            "First ref: path={}, offset={}, size={}",
            &first.path[..80],
            first.offset,
            first.size
        );
    }
}
