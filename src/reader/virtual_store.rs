//! Virtual store adapter for VirtualiZarr Parquet reference stores.
//!
//! Implements zarrs storage traits to enable reading virtual Zarr datasets
//! where chunks are stored as byte ranges in remote files (NetCDF, GRIB, etc.).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use tracing::{debug, instrument};
use url::Url;

use zarrs::storage::{
    byte_range::ByteRange, AsyncListableStorageTraits, AsyncMaybeBytesIterator,
    AsyncReadableStorageTraits, MaybeBytes, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes,
    StorePrefix,
};
use zarrs_object_store::object_store::aws::AmazonS3Builder;
use zarrs_object_store::object_store::path::Path as ObjectPath;
use zarrs_object_store::object_store::{GetRange, ObjectStore};

use super::parquet_refs::{indices_to_flat_index, parse_chunk_key, ChunkRef, ParquetRefs};

/// Virtual store adapter that reads from VirtualiZarr Parquet reference files.
///
/// This adapter translates Zarr chunk requests into byte-range fetches from
/// the source files (typically NetCDF on S3).
pub struct VirtualStoreAdapter {
    /// Path to the VirtualiZarr store root (kept for debugging/logging)
    #[allow(dead_code)]
    store_path: String,
    /// Consolidated metadata from .zmetadata
    metadata: serde_json::Value,
    /// Loaded parquet refs, keyed by array name
    refs: HashMap<String, ParquetRefs>,
    /// S3 client for fetching chunks from source files
    s3_client: Option<Arc<dyn ObjectStore>>,
    /// Array metadata cache (shape, chunks) from .zmetadata
    array_meta: HashMap<String, ArrayMetaCache>,
}

/// Cached array metadata for chunk index calculations
#[derive(Debug, Clone)]
struct ArrayMetaCache {
    /// Array shape (kept for future use in range validation)
    #[allow(dead_code)]
    shape: Vec<u64>,
    /// Chunk sizes (kept for future use in partial read optimization)
    #[allow(dead_code)]
    chunks: Vec<u64>,
    /// Number of chunks per dimension (used for flat index calculation)
    chunks_per_dim: Vec<u64>,
}

impl VirtualStoreAdapter {
    /// Create a new VirtualStoreAdapter from a local VirtualiZarr store path.
    pub fn new(store_path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let zmetadata_path = Path::new(store_path).join(".zmetadata");
        let content = std::fs::read_to_string(&zmetadata_path).map_err(|e| {
            format!(
                "Failed to read .zmetadata at {}: {}",
                zmetadata_path.display(),
                e
            )
        })?;

        // Handle non-standard JSON values (NaN, Infinity) that Zarr uses
        let content = content
            .replace(":NaN", ":null")
            .replace(": NaN", ": null")
            .replace(":Infinity", ":null")
            .replace(": Infinity", ": null")
            .replace(":-Infinity", ":null")
            .replace(": -Infinity", ": null");

        let metadata: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse .zmetadata: {}", e))?;

        // Pre-parse array metadata for chunk calculations
        let array_meta = Self::parse_array_metadata(&metadata)?;

        // Load refs for all arrays upfront (could be made lazy)
        let refs = Self::load_all_refs(store_path, &array_meta)?;

        // Create S3 client if any refs point to S3
        let s3_client = Self::create_s3_client_if_needed(&refs)?;

        Ok(Self {
            store_path: store_path.to_string(),
            metadata,
            refs,
            s3_client,
            array_meta,
        })
    }

    /// Create a new VirtualStoreAdapter from a remote store asynchronously.
    ///
    /// # Arguments
    /// * `store` - The async storage backend (GCS, S3, etc.)
    /// * `prefix` - Path prefix to the VirtualiZarr store root
    /// * `location` - Original location string (for debugging/logging)
    pub async fn new_async<S>(
        store: &S,
        prefix: &ObjectPath,
        location: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        S: AsyncReadableStorageTraits + AsyncListableStorageTraits,
    {
        debug!(location = %location, prefix = %prefix, "Creating async VirtualStoreAdapter");

        // Read .zmetadata from remote store
        let zmetadata_key = zarrs::storage::StoreKey::new(format!("{}/.zmetadata", prefix))
            .map_err(|e| format!("Invalid store key: {}", e))?;

        let content_bytes = store
            .get(&zmetadata_key)
            .await?
            .ok_or_else(|| format!("Missing .zmetadata at {}", prefix))?;

        let content = String::from_utf8(content_bytes.to_vec())
            .map_err(|e| format!("Invalid UTF-8 in .zmetadata: {}", e))?;

        // Handle non-standard JSON values (NaN, Infinity) that Zarr uses
        let content = content
            .replace(":NaN", ":null")
            .replace(": NaN", ": null")
            .replace(":Infinity", ":null")
            .replace(": Infinity", ": null")
            .replace(":-Infinity", ":null")
            .replace(": -Infinity", ": null");

        let metadata: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse .zmetadata: {}", e))?;

        // Pre-parse array metadata for chunk calculations
        let array_meta = Self::parse_array_metadata(&metadata)?;

        // Load refs for all arrays asynchronously
        let refs = Self::load_all_refs_async(store, prefix, &array_meta).await?;

        // Create S3 client if any refs point to S3
        let s3_client = Self::create_s3_client_if_needed(&refs)?;

        debug!(
            location = %location,
            arrays = refs.len(),
            "Created async VirtualStoreAdapter"
        );

        Ok(Self {
            store_path: location.to_string(),
            metadata,
            refs,
            s3_client,
            array_meta,
        })
    }

    /// Load parquet refs for all arrays from a remote store asynchronously.
    /// Uses parallel fetching to minimize total latency.
    async fn load_all_refs_async<S>(
        store: &S,
        prefix: &ObjectPath,
        array_meta: &HashMap<String, ArrayMetaCache>,
    ) -> Result<HashMap<String, ParquetRefs>, Box<dyn std::error::Error + Send + Sync>>
    where
        S: AsyncReadableStorageTraits + AsyncListableStorageTraits,
    {
        use futures::future::join_all;
        use tracing::{debug, info};

        let array_names: Vec<String> = array_meta.keys().cloned().collect();

        info!(
            num_arrays = array_names.len(),
            "Loading parquet refs in parallel"
        );

        // Create futures for all arrays
        let fetch_futures: Vec<_> = array_names
            .iter()
            .map(|array_name| {
                let name = array_name.clone();
                async move {
                    match ParquetRefs::load_async(store, prefix, &name).await {
                        Ok(r) => {
                            debug!(array = %name, count = r.len(), "Loaded refs");
                            Some((name, r))
                        }
                        Err(e) => {
                            debug!(array = %name, error = %e, "No refs found (may be inline)");
                            None
                        }
                    }
                }
            })
            .collect();

        // Execute all in parallel
        let results = join_all(fetch_futures).await;

        // Collect successful results
        let refs: HashMap<String, ParquetRefs> = results.into_iter().flatten().collect();

        info!(loaded = refs.len(), "Loaded parquet refs");
        Ok(refs)
    }

    /// Parse array metadata from .zmetadata for all arrays
    fn parse_array_metadata(
        metadata: &serde_json::Value,
    ) -> Result<HashMap<String, ArrayMetaCache>, Box<dyn std::error::Error + Send + Sync>> {
        let mut result = HashMap::new();

        let meta_obj = metadata
            .get("metadata")
            .and_then(|m| m.as_object())
            .ok_or("Missing 'metadata' object in .zmetadata")?;

        for (key, value) in meta_obj {
            if key.ends_with("/.zarray") {
                let array_name = key.trim_end_matches("/.zarray").to_string();

                let shape: Vec<u64> = value
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                    .unwrap_or_default();

                let chunks: Vec<u64> = value
                    .get("chunks")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_u64()).collect())
                    .unwrap_or_default();

                // Calculate chunks per dimension
                let chunks_per_dim: Vec<u64> = shape
                    .iter()
                    .zip(chunks.iter())
                    .map(|(&s, &c)| s.div_ceil(c))
                    .collect();

                result.insert(
                    array_name,
                    ArrayMetaCache {
                        shape,
                        chunks,
                        chunks_per_dim,
                    },
                );
            }
        }

        Ok(result)
    }

    /// Load parquet refs for all arrays in the store
    fn load_all_refs(
        store_path: &str,
        array_meta: &HashMap<String, ArrayMetaCache>,
    ) -> Result<HashMap<String, ParquetRefs>, Box<dyn std::error::Error + Send + Sync>> {
        let mut refs = HashMap::new();

        for array_name in array_meta.keys() {
            // Check if refs exist for this array
            let refs_pattern = format!("{}/{}/refs.*.parq", store_path, array_name);
            if glob::glob(&refs_pattern)?.next().is_some() {
                debug!(array = %array_name, "Loading parquet refs");
                match ParquetRefs::load(store_path, array_name) {
                    Ok(r) => {
                        debug!(array = %array_name, count = r.len(), "Loaded refs");
                        refs.insert(array_name.clone(), r);
                    }
                    Err(e) => {
                        debug!(array = %array_name, error = %e, "No refs found (may be inline)");
                    }
                }
            }
        }

        Ok(refs)
    }

    /// Create S3 client if any refs point to S3 URLs
    fn create_s3_client_if_needed(
        refs: &HashMap<String, ParquetRefs>,
    ) -> Result<Option<Arc<dyn ObjectStore>>, Box<dyn std::error::Error + Send + Sync>> {
        // Check if any refs point to S3
        for parquet_refs in refs.values() {
            if let Some(first_ref) = parquet_refs.get(0) {
                if first_ref.path.starts_with("s3://") {
                    // Extract bucket from first S3 URL
                    let url = Url::parse(&first_ref.path)?;
                    let bucket = url
                        .host_str()
                        .ok_or("Missing bucket in S3 URL")?
                        .to_string();

                    debug!(bucket = %bucket, "Creating S3 client for VirtualiZarr refs");

                    // Try environment credentials first, fall back to anonymous for public buckets
                    let store = AmazonS3Builder::from_env()
                        .with_bucket_name(&bucket)
                        .with_skip_signature(true) // Allow anonymous access for public buckets
                        .with_region("us-east-1") // NOAA data is typically in us-east-1
                        .build()?;

                    return Ok(Some(Arc::new(store)));
                }
            }
        }

        Ok(None)
    }

    /// Get the raw .zmetadata JSON for schema inference
    pub fn raw_metadata(&self) -> &serde_json::Value {
        &self.metadata
    }

    /// Get metadata value for a key from .zmetadata
    fn get_metadata(&self, key: &str) -> Result<MaybeBytes, StorageError> {
        let key = key.trim_start_matches('/');

        // Handle .zmetadata - return the full consolidated metadata
        if key == ".zmetadata" {
            let json = serde_json::to_vec(&self.metadata)
                .map_err(|e| StorageError::Other(format!("JSON error: {}", e)))?;
            return Ok(Some(json.into()));
        }

        // Handle root-level keys
        if key == ".zgroup" {
            if let Some(zgroup) = self.metadata.get("metadata").and_then(|m| m.get(".zgroup")) {
                let json = serde_json::to_vec(zgroup)
                    .map_err(|e| StorageError::Other(format!("JSON error: {}", e)))?;
                return Ok(Some(json.into()));
            }
        }

        if key == ".zattrs" {
            if let Some(zattrs) = self.metadata.get("metadata").and_then(|m| m.get(".zattrs")) {
                let json = serde_json::to_vec(zattrs)
                    .map_err(|e| StorageError::Other(format!("JSON error: {}", e)))?;
                return Ok(Some(json.into()));
            }
            // Return empty object if no root attrs
            return Ok(Some(b"{}".to_vec().into()));
        }

        // Handle array-level metadata (.zarray, .zattrs)
        let is_metadata_key = key.contains("/.zarray")
            || key.contains("/.zattrs")
            || key.ends_with(".zarray")
            || key.ends_with(".zattrs");

        if !is_metadata_key {
            return Ok(None);
        }
        let meta_key = key.to_string();

        if let Some(value) = self.metadata.get("metadata").and_then(|m| m.get(&meta_key)) {
            // For .zarray metadata, convert unsupported dtypes
            let value = if meta_key.ends_with(".zarray") {
                Self::transform_zarray_metadata(value)
            } else {
                value.clone()
            };

            let json = serde_json::to_vec(&value)
                .map_err(|e| StorageError::Other(format!("JSON error: {}", e)))?;
            return Ok(Some(json.into()));
        }

        Ok(None)
    }

    /// Transform .zarray metadata to handle unsupported dtypes
    /// zarrs doesn't support datetime64/timedelta64, convert to int64
    fn transform_zarray_metadata(zarray: &serde_json::Value) -> serde_json::Value {
        let mut zarray = zarray.clone();
        if let Some(obj) = zarray.as_object_mut() {
            if let Some(dtype) = obj.get("dtype").and_then(|d| d.as_str()) {
                // Convert datetime64 and timedelta64 to int64 (underlying storage)
                // Format: <M8[ns], <M8[us], <m8[ns], etc.
                if dtype.contains("M8") || dtype.contains("m8") {
                    obj.insert("dtype".to_string(), serde_json::json!("<i8"));
                }
            }
        }
        zarray
    }

    /// Fetch a chunk from S3 using the reference info
    #[instrument(level = "debug", skip(self))]
    async fn fetch_chunk(&self, chunk_ref: &ChunkRef) -> Result<MaybeBytes, StorageError> {
        // Handle inline data
        if let Some(ref raw) = chunk_ref.raw {
            return Ok(Some(raw.clone().into()));
        }

        // Parse S3 URL
        let url = Url::parse(&chunk_ref.path)
            .map_err(|e| StorageError::Other(format!("Invalid URL: {}", e)))?;

        let s3_client = self
            .s3_client
            .as_ref()
            .ok_or_else(|| StorageError::Other("No S3 client configured".to_string()))?;

        // Extract path from URL (everything after bucket)
        let path = url.path().trim_start_matches('/');
        let object_path = ObjectPath::from(path);

        // Fetch byte range using get_opts with range
        debug!(
            path = %chunk_ref.path,
            offset = chunk_ref.offset,
            size = chunk_ref.size,
            "Fetching chunk from S3"
        );

        let opts = zarrs_object_store::object_store::GetOptions {
            range: Some(GetRange::Bounded(
                chunk_ref.offset..(chunk_ref.offset + chunk_ref.size),
            )),
            ..Default::default()
        };

        let result = s3_client
            .get_opts(&object_path, opts)
            .await
            .map_err(|e| StorageError::Other(format!("S3 fetch error: {}", e)))?;

        let bytes = result
            .bytes()
            .await
            .map_err(|e| StorageError::Other(format!("S3 read error: {}", e)))?;

        Ok(Some(bytes))
    }
}

#[async_trait]
impl AsyncReadableStorageTraits for VirtualStoreAdapter {
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let key_str = key.to_string();
        debug!(key = %key_str, "VirtualStoreAdapter::get called");
        let key_str = key_str.trim_start_matches('/');

        // Check if this is a metadata request
        if key_str.ends_with(".zarray")
            || key_str.ends_with(".zattrs")
            || key_str == ".zgroup"
            || key_str.ends_with(".zgroup")
            || key_str == ".zmetadata"
        {
            debug!(key = %key_str, "Returning metadata");
            return self.get_metadata(key_str);
        }

        // zarrs may try to probe for Zarr v3 metadata (zarr.json) - return None
        // to signal it doesn't exist (this is a v2 store)
        if key_str.ends_with("zarr.json") {
            debug!(key = %key_str, "V3 metadata requested, returning None (V2 store)");
            return Ok(None);
        }

        // This is a chunk request - parse the key
        debug!(key = %key_str, "Parsing as chunk key");
        let (array_name, chunk_indices) =
            parse_chunk_key(key_str).map_err(|e| StorageError::Other(e.to_string()))?;

        // Get array metadata for chunk index calculation
        let array_meta = self
            .array_meta
            .get(&array_name)
            .ok_or_else(|| StorageError::Other(format!("Unknown array: {}", array_name)))?;

        // Calculate flat chunk index
        let flat_index = indices_to_flat_index(&chunk_indices, &array_meta.chunks_per_dim);

        // Look up the chunk reference
        let refs = self.refs.get(&array_name).ok_or_else(|| {
            StorageError::Other(format!("No refs loaded for array: {}", array_name))
        })?;

        let chunk_ref = refs.get(flat_index).ok_or_else(|| {
            StorageError::Other(format!("Chunk {} not found in refs", flat_index))
        })?;

        // Fetch the chunk data
        self.fetch_chunk(chunk_ref).await
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: zarrs::storage::byte_range::ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        // For now, just get the full chunk and slice
        // TODO: Optimize with actual partial reads
        let full_data = self.get(key).await?;

        if let Some(data) = full_data {
            let data_len = data.len() as u64;
            let ranges: Vec<ByteRange> = byte_ranges.collect();
            let mut results = Vec::new();

            for range in ranges {
                let start = range.start(data_len) as usize;
                let end = range.end(data_len) as usize;
                results.push(data.slice(start..end));
            }

            Ok(Some(Box::pin(stream::iter(results.into_iter().map(Ok)))))
        } else {
            Ok(None)
        }
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let key_str = key.to_string();
        let key_str = key_str.trim_start_matches('/');

        // For metadata, get the size from memory
        if key_str.ends_with(".zarray")
            || key_str.ends_with(".zattrs")
            || key_str == ".zgroup"
            || key_str.ends_with(".zgroup")
        {
            if let Ok(Some(data)) = self.get_metadata(key_str) {
                return Ok(Some(data.len() as u64));
            }
        }

        // For chunks, get size from refs
        if let Ok((array_name, chunk_indices)) = parse_chunk_key(key_str) {
            if let Some(array_meta) = self.array_meta.get(&array_name) {
                let flat_index = indices_to_flat_index(&chunk_indices, &array_meta.chunks_per_dim);
                if let Some(refs) = self.refs.get(&array_name) {
                    if let Some(chunk_ref) = refs.get(flat_index) {
                        return Ok(Some(chunk_ref.size));
                    }
                }
            }
        }

        Ok(None)
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

#[async_trait]
impl AsyncListableStorageTraits for VirtualStoreAdapter {
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        // List all keys from metadata
        let mut keys = Vec::new();

        if let Some(meta) = self.metadata.get("metadata").and_then(|m| m.as_object()) {
            for key in meta.keys() {
                if let Ok(store_key) = StoreKey::new(key) {
                    keys.push(store_key);
                }
            }
        }

        Ok(keys)
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let prefix_str = prefix.as_str();
        let all_keys = self.list().await?;

        Ok(all_keys
            .into_iter()
            .filter(|k| k.as_str().starts_with(prefix_str))
            .collect())
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let prefix_str = prefix.as_str().trim_matches('/');

        let mut keys = Vec::new();
        let mut prefixes = std::collections::HashSet::new();

        if let Some(meta) = self.metadata.get("metadata").and_then(|m| m.as_object()) {
            for key in meta.keys() {
                let key = key.trim_start_matches('/');

                // Check if key is under this prefix
                if prefix_str.is_empty() || key.starts_with(prefix_str) {
                    let remainder = if prefix_str.is_empty() {
                        key.to_string()
                    } else {
                        key.trim_start_matches(prefix_str)
                            .trim_start_matches('/')
                            .to_string()
                    };

                    // Is this a direct child or a subdirectory?
                    if let Some(slash_pos) = remainder.find('/') {
                        // Subdirectory - add as prefix
                        let subdir = &remainder[..slash_pos];
                        let full_prefix = if prefix_str.is_empty() {
                            format!("{}/", subdir)
                        } else {
                            format!("{}/{}/", prefix_str, subdir)
                        };
                        prefixes.insert(full_prefix);
                    } else {
                        // Direct child - add as key
                        if let Ok(store_key) = StoreKey::new(key) {
                            keys.push(store_key);
                        }
                    }
                }
            }
        }

        let prefix_vec: Vec<StorePrefix> = prefixes
            .into_iter()
            .filter_map(|p| StorePrefix::new(&p).ok())
            .collect();

        Ok(StoreKeysPrefixes::new(keys, prefix_vec))
    }

    async fn size(&self) -> Result<u64, StorageError> {
        // Not implemented - would need to sum all chunk sizes
        Ok(0)
    }

    async fn size_prefix(&self, _prefix: &StorePrefix) -> Result<u64, StorageError> {
        // Not implemented
        Ok(0)
    }
}

/// Check if a path is a VirtualiZarr Parquet reference store
pub fn is_virtualizarr_store(path: &str) -> bool {
    let root = Path::new(path);

    // Must have .zmetadata
    if !root.join(".zmetadata").exists() {
        return false;
    }

    // Must have at least one refs.*.parq file in a subdirectory
    let pattern = format!("{}/*/refs.*.parq", path);
    if let Ok(mut matches) = glob::glob(&pattern) {
        return matches.next().is_some();
    }

    false
}

/// Asynchronously check if a remote path is a VirtualiZarr Parquet reference store.
///
/// # Arguments
/// * `store` - The async storage backend (GCS, S3, etc.)
/// * `prefix` - Path prefix to check (e.g., "bucket/path/to/store")
///
/// Note: This function first tries to check via file existence (works for buckets
/// without list permissions), then falls back to listing if needed.
pub async fn is_virtualizarr_store_async<S>(store: &S, prefix: &ObjectPath) -> bool
where
    S: AsyncReadableStorageTraits + AsyncListableStorageTraits,
{
    // Check for .zmetadata - this is required for VirtualiZarr
    let zmetadata_key =
        zarrs::storage::StoreKey::new(format!("{}/.zmetadata", prefix)).expect("Invalid store key");

    // Try to get the .zmetadata file
    let zmetadata_content = match store.get(&zmetadata_key).await {
        Ok(Some(content)) => content,
        _ => {
            debug!(prefix = %prefix, "No .zmetadata found - not a VirtualiZarr store");
            return false;
        }
    };

    // Parse .zmetadata to find array names, then check for refs.0.parq
    let content_str = match String::from_utf8(zmetadata_content.to_vec()) {
        Ok(s) => s,
        Err(_) => {
            debug!(prefix = %prefix, "Invalid UTF-8 in .zmetadata");
            return false;
        }
    };

    // Sanitize JSON (handle NaN, Infinity)
    let content_str = content_str
        .replace(":NaN", ":null")
        .replace(": NaN", ": null")
        .replace(":Infinity", ":null")
        .replace(": Infinity", ": null")
        .replace(":-Infinity", ":null")
        .replace(": -Infinity", ": null");

    let metadata: serde_json::Value = match serde_json::from_str(&content_str) {
        Ok(v) => v,
        Err(_) => {
            debug!(prefix = %prefix, "Failed to parse .zmetadata JSON");
            return false;
        }
    };

    // Extract array names from .zmetadata
    let meta_obj = match metadata.get("metadata").and_then(|m| m.as_object()) {
        Some(obj) => obj,
        None => {
            debug!(prefix = %prefix, "No 'metadata' object in .zmetadata");
            return false;
        }
    };

    // Find array names (keys ending with /.zarray)
    let array_names: Vec<String> = meta_obj
        .keys()
        .filter_map(|k| {
            if k.ends_with("/.zarray") {
                Some(k.trim_end_matches("/.zarray").to_string())
            } else {
                None
            }
        })
        .collect();

    // Check if any array has refs.0.parq (the first refs file)
    // Only check the first few arrays to avoid slow sequential HTTP requests
    // for large stores (ERA5 has 277 arrays). If it's truly a VirtualiZarr store,
    // the first few arrays should have refs.
    let arrays_to_check: Vec<_> = array_names.into_iter().take(5).collect();

    debug!(
        prefix = %prefix,
        num_arrays = arrays_to_check.len(),
        "Checking for VirtualiZarr refs"
    );

    for array_name in arrays_to_check {
        let refs_key =
            zarrs::storage::StoreKey::new(format!("{}/{}/refs.0.parq", prefix, array_name));
        if let Ok(refs_key) = refs_key {
            if store.size_key(&refs_key).await.ok().flatten().is_some() {
                debug!(
                    prefix = %prefix,
                    array = %array_name,
                    "Found refs.0.parq - is a VirtualiZarr store"
                );
                return true;
            }
        }
    }

    debug!(prefix = %prefix, "No refs.*.parq found - not a VirtualiZarr store");
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_virtualizarr_store() {
        // This should be detected as VirtualiZarr
        assert!(is_virtualizarr_store("data/FOUR_v200_GFS.parq"));

        // Regular Zarr stores should not be detected
        assert!(!is_virtualizarr_store("data/synthetic_v2.zarr"));
    }

    #[test]
    fn test_virtual_store_adapter_creation() {
        let adapter = VirtualStoreAdapter::new("data/FOUR_v200_GFS.parq").unwrap();

        // Should have loaded refs for multiple arrays
        assert!(!adapter.refs.is_empty());
        println!("Loaded refs for {} arrays", adapter.refs.len());

        // Should have array metadata
        assert!(!adapter.array_meta.is_empty());

        // t2 should be in the refs
        assert!(adapter.refs.contains_key("t2"));

        // Should have S3 client since refs point to S3
        assert!(adapter.s3_client.is_some());
    }

    #[tokio::test]
    async fn test_virtual_store_metadata_access() {
        use zarrs::storage::AsyncReadableStorageTraits;

        let adapter = VirtualStoreAdapter::new("data/FOUR_v200_GFS.parq").unwrap();

        // Should be able to read .zgroup
        let zgroup_key = StoreKey::new(".zgroup").unwrap();
        let zgroup = adapter.get(&zgroup_key).await.unwrap();
        assert!(zgroup.is_some());

        // Should be able to read t2/.zarray
        let zarray_key = StoreKey::new("t2/.zarray").unwrap();
        let zarray = adapter.get(&zarray_key).await.unwrap();
        assert!(zarray.is_some());

        let zarray_content: serde_json::Value = serde_json::from_slice(&zarray.unwrap()).unwrap();
        println!("t2/.zarray: {:?}", zarray_content);

        // Verify shape
        let shape = zarray_content.get("shape").unwrap().as_array().unwrap();
        assert_eq!(shape.len(), 4); // [init_time, time, lat, lon]
    }
}
