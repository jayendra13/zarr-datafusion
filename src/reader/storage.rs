//! Storage backend factory for local and cloud object stores.
//!
//! Supports:
//! - Local filesystem: plain paths or `file:///path`
//! - AWS S3: `s3://bucket/path`
//! - Google Cloud Storage: `gs://bucket/path`

use std::sync::Arc;
use tracing::{debug, info, instrument};
use url::Url;
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::object_store::aws::AmazonS3Builder;
use zarrs_object_store::object_store::gcp::GoogleCloudStorageBuilder;
use zarrs_object_store::object_store::path::Path as ObjectPath;
use zarrs_object_store::object_store::Error as ObjectStoreError;
use zarrs_object_store::AsyncObjectStore;

/// Error type for storage operations
#[derive(Debug)]
pub enum StorageError {
    InvalidUrl(String),
    ObjectStore(ObjectStoreError),
    Filesystem(std::io::Error),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::InvalidUrl(msg) => write!(f, "Invalid URL: {}", msg),
            StorageError::ObjectStore(e) => write!(f, "Object store error: {}", e),
            StorageError::Filesystem(e) => write!(f, "Filesystem error: {}", e),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<ObjectStoreError> for StorageError {
    fn from(e: ObjectStoreError) -> Self {
        StorageError::ObjectStore(e)
    }
}

impl From<std::io::Error> for StorageError {
    fn from(e: std::io::Error) -> Self {
        StorageError::Filesystem(e)
    }
}

/// Storage location with backend type and path information
#[derive(Debug, Clone)]
pub struct StorageLocation {
    /// The original URL/path string
    pub url: String,
    /// Path within the store (for object stores, this is the prefix)
    pub path: String,
    /// Whether this is a remote (cloud) store
    pub is_remote: bool,
}

impl StorageLocation {
    /// Parse a URL or path into a StorageLocation
    pub fn parse(location: &str) -> Result<Self, StorageError> {
        if location.starts_with("s3://") || location.starts_with("gs://") {
            // Cloud URL - extract path from URL
            let url = Url::parse(location).map_err(|e| StorageError::InvalidUrl(e.to_string()))?;
            let path = url.path().trim_start_matches('/').to_string();
            Ok(StorageLocation {
                url: location.to_string(),
                path,
                is_remote: true,
            })
        } else if location.starts_with("file://") {
            // Local file URL
            let path = location.trim_start_matches("file://").to_string();
            Ok(StorageLocation {
                url: location.to_string(),
                path,
                is_remote: false,
            })
        } else {
            // Plain local path
            Ok(StorageLocation {
                url: location.to_string(),
                path: location.to_string(),
                is_remote: false,
            })
        }
    }
}

/// Create an async storage backend from a URL or path.
///
/// # Supported URL schemes
///
/// - `s3://bucket/path` - AWS S3 (credentials from environment)
/// - `gs://bucket/path` - Google Cloud Storage (credentials from environment or anonymous)
/// - `file:///path` or plain path - Local filesystem
///
/// # Environment Variables
///
/// ## S3
/// - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - Credentials
/// - `AWS_DEFAULT_REGION` - Region
/// - `AWS_ENDPOINT` - Custom endpoint (for S3-compatible services)
///
/// ## GCS
/// - `GOOGLE_SERVICE_ACCOUNT` - Path to service account JSON
/// - Or uses Application Default Credentials
/// - Falls back to anonymous access for public buckets
#[instrument(level = "debug")]
pub async fn create_async_store(
    location: &str,
) -> Result<(AsyncReadableListableStorage, ObjectPath), StorageError> {
    debug!("Parsing storage location");
    let _loc = StorageLocation::parse(location)?;
    debug!(is_remote = _loc.is_remote, path = %_loc.path, "Location parsed");

    if location.starts_with("s3://") {
        info!("Creating S3 store");
        create_s3_store(location).await
    } else if location.starts_with("gs://") {
        info!("Creating GCS store");
        create_gcs_store(location).await
    } else {
        debug!("Local filesystem - not supported for async");
        Err(StorageError::InvalidUrl(
            "Local filesystem should use synchronous FilesystemStore".to_string(),
        ))
    }
}

/// Create an S3 storage backend
async fn create_s3_store(
    url: &str,
) -> Result<(AsyncReadableListableStorage, ObjectPath), StorageError> {
    let parsed = Url::parse(url).map_err(|e| StorageError::InvalidUrl(e.to_string()))?;
    let bucket = parsed
        .host_str()
        .ok_or_else(|| StorageError::InvalidUrl("Missing bucket in S3 URL".to_string()))?;
    let path = parsed.path().trim_start_matches('/');

    let store = AmazonS3Builder::from_env()
        .with_bucket_name(bucket)
        .build()?;

    let async_store: AsyncReadableListableStorage = Arc::new(AsyncObjectStore::new(store));
    let object_path = ObjectPath::from(path);

    Ok((async_store, object_path))
}

/// Create a GCS storage backend
#[instrument(level = "debug")]
async fn create_gcs_store(
    url: &str,
) -> Result<(AsyncReadableListableStorage, ObjectPath), StorageError> {
    let parsed = Url::parse(url).map_err(|e| StorageError::InvalidUrl(e.to_string()))?;
    let bucket = parsed
        .host_str()
        .ok_or_else(|| StorageError::InvalidUrl("Missing bucket in GCS URL".to_string()))?;
    let path = parsed.path().trim_start_matches('/');
    debug!(bucket = bucket, path = path, "Parsed GCS URL");

    // For public buckets, try anonymous access first (faster)
    // Then fall back to credentials from environment
    debug!("Trying anonymous access for public bucket");
    let store = GoogleCloudStorageBuilder::new()
        .with_bucket_name(bucket)
        .with_skip_signature(true)
        .build()
        .or_else(|_| {
            debug!("Anonymous access failed, trying credentials from environment");
            GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(bucket)
                .build()
        })?;

    let async_store: AsyncReadableListableStorage = Arc::new(AsyncObjectStore::new(store));
    let object_path = ObjectPath::from(path);
    info!(
        bucket = bucket,
        path = path,
        "GCS store created successfully"
    );

    Ok((async_store, object_path))
}

/// Check if a location is a remote (cloud) URL
pub fn is_remote_url(location: &str) -> bool {
    location.starts_with("s3://") || location.starts_with("gs://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_url() {
        let loc = StorageLocation::parse("s3://my-bucket/path/to/data.zarr").unwrap();
        assert!(loc.is_remote);
        assert_eq!(loc.path, "path/to/data.zarr");
    }

    #[test]
    fn test_parse_gcs_url() {
        let loc = StorageLocation::parse("gs://my-bucket/path/to/data.zarr").unwrap();
        assert!(loc.is_remote);
        assert_eq!(loc.path, "path/to/data.zarr");
    }

    #[test]
    fn test_parse_local_path() {
        let loc = StorageLocation::parse("/data/synthetic.zarr").unwrap();
        assert!(!loc.is_remote);
        assert_eq!(loc.path, "/data/synthetic.zarr");
    }

    #[test]
    fn test_parse_file_url() {
        let loc = StorageLocation::parse("file:///data/synthetic.zarr").unwrap();
        assert!(!loc.is_remote);
        assert_eq!(loc.path, "/data/synthetic.zarr");
    }

    #[test]
    fn test_is_remote_url() {
        assert!(is_remote_url("s3://bucket/path"));
        assert!(is_remote_url("gs://bucket/path"));
        assert!(!is_remote_url("/local/path"));
        assert!(!is_remote_url("file:///local/path"));
    }
}
