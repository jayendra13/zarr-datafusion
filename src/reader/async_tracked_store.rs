//! An async storage adapter that tracks I/O and logs chunk keys using tracing.
//!
//! Wraps any async storage backend to log chunk URLs and track bytes read.

use std::sync::Arc;

use async_trait::async_trait;
use futures::{stream, StreamExt, TryStreamExt};
use tracing::debug;

use zarrs::storage::{
    byte_range::{ByteRange, ByteRangeIterator},
    AsyncListableStorageTraits, AsyncMaybeBytesIterator, AsyncReadableStorageTraits, MaybeBytes,
    StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
};

use super::stats::SharedIoStats;

/// Async storage adapter that logs chunk keys via tracing and optionally tracks bytes read.
///
/// Wraps an inner async storage and logs every `get` operation with `tracing::debug!`.
#[derive(Debug)]
pub struct AsyncTrackedStore<S: ?Sized> {
    inner: Arc<S>,
    stats: Option<SharedIoStats>,
}

impl<S: ?Sized> AsyncTrackedStore<S> {
    /// Create a new async tracked store wrapping the given storage.
    pub fn new(inner: Arc<S>, stats: Option<SharedIoStats>) -> Self {
        Self { inner, stats }
    }
}

#[async_trait]
impl<S: ?Sized + AsyncReadableStorageTraits> AsyncReadableStorageTraits for AsyncTrackedStore<S> {
    async fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        // debug!(chunk_key = %key, "Fetching chunk from remote store");

        let result = self.inner.get(key).await?;

        if let Some(ref bytes) = result {
            let len = bytes.len() as u64;
            debug!(chunk_key = %key, bytes = len, "Chunk fetched successfully");
            if let Some(ref s) = self.stats {
                s.record_disk_read(len);
            }
        }

        Ok(result)
    }

    async fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<AsyncMaybeBytesIterator<'a>, StorageError> {
        let byte_ranges_vec: Vec<ByteRange> = byte_ranges.collect();
        debug!(
            chunk_key = %key,
            num_ranges = byte_ranges_vec.len(),
            "Fetching partial chunk from remote store"
        );

        let result = self
            .inner
            .get_partial_many(key, Box::new(byte_ranges_vec.into_iter()))
            .await?;

        if let Some(result_stream) = result {
            let bytes_vec: Vec<_> = result_stream.try_collect().await?;
            let total_bytes: u64 = bytes_vec.iter().map(|b| b.len() as u64).sum();

            debug!(
                chunk_key = %key,
                bytes = total_bytes,
                "Partial chunk fetched successfully"
            );

            if let Some(ref s) = self.stats {
                s.record_disk_read(total_bytes);
            }

            Ok(Some(stream::iter(bytes_vec.into_iter().map(Ok)).boxed()))
        } else {
            Ok(None)
        }
    }

    async fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key).await
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

#[async_trait]
impl<S: ?Sized + AsyncListableStorageTraits> AsyncListableStorageTraits for AsyncTrackedStore<S> {
    async fn list(&self) -> Result<StoreKeys, StorageError> {
        self.inner.list().await
    }

    async fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.inner.list_prefix(prefix).await
    }

    async fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.inner.list_dir(prefix).await
    }

    async fn size(&self) -> Result<u64, StorageError> {
        self.inner.size().await
    }

    async fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.inner.size_prefix(prefix).await
    }
}
