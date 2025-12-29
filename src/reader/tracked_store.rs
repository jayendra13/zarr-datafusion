//! A storage adapter that tracks disk I/O bytes.
//!
//! Wraps any storage backend and counts actual bytes read from disk.
//! Based on zarrs_storage's UsageLogStorageAdapter pattern.

use std::sync::Arc;

use zarrs::storage::{
    byte_range::{ByteRange, ByteRangeIterator},
    ListableStorageTraits, MaybeBytes, MaybeBytesIterator, ReadableStorageTraits, StorageError,
    StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
};

use super::stats::SharedIoStats;

/// Storage adapter that tracks bytes read from disk.
///
/// Wraps an inner storage and accumulates byte counts into shared stats.
#[derive(Debug)]
pub struct TrackedStore<S> {
    inner: Arc<S>,
    stats: SharedIoStats,
}

impl<S> TrackedStore<S> {
    /// Create a new tracked store wrapping the given storage.
    pub fn new(inner: Arc<S>, stats: SharedIoStats) -> Self {
        Self { inner, stats }
    }
}

impl<S: ReadableStorageTraits> ReadableStorageTraits for TrackedStore<S> {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let result = self.inner.get(key)?;

        // Track actual bytes read from disk
        if let Some(ref bytes) = result {
            self.stats.record_disk_read(bytes.len() as u64);
        }

        Ok(result)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        // Collect ranges to allow reuse
        let ranges: Vec<ByteRange> = byte_ranges.collect();

        let result = self
            .inner
            .get_partial_many(key, Box::new(ranges.into_iter()))?;

        // Track bytes - we need to consume the iterator to count, then recreate it
        if let Some(iter) = result {
            let bytes_vec: Vec<_> = iter.collect::<Result<Vec<_>, _>>()?;
            let total_bytes: u64 = bytes_vec.iter().map(|b| b.len() as u64).sum();
            self.stats.record_disk_read(total_bytes);

            // Return a new iterator over the collected bytes
            Ok(Some(Box::new(bytes_vec.into_iter().map(Ok))))
        } else {
            Ok(None)
        }
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

impl<S: ListableStorageTraits> ListableStorageTraits for TrackedStore<S> {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.inner.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.inner.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.inner.list_dir(prefix)
    }

    fn size(&self) -> Result<u64, StorageError> {
        self.inner.size()
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.inner.size_prefix(prefix)
    }
}
