//! I/O statistics for Zarr reads
//!
//! Tracks bytes read, arrays accessed, and timing breakdown for metadata,
//! coordinates, and data variables.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// I/O statistics collected during Zarr reads.
///
/// Uses atomic counters for thread-safety without locks, which is important
/// for async/multi-threaded execution in DataFusion.
#[derive(Debug, Default)]
pub struct ZarrIoStats {
    // Byte counts (in-memory/uncompressed)
    pub metadata_bytes: AtomicU64,
    pub coord_bytes: AtomicU64,
    pub data_bytes: AtomicU64,

    // Disk bytes (actual I/O, compressed)
    pub disk_bytes: AtomicU64,

    // Array counts
    pub coord_arrays: AtomicU64,
    pub data_arrays: AtomicU64,

    // Timing (stored as nanoseconds)
    pub metadata_nanos: AtomicU64,
    pub coord_nanos: AtomicU64,
    pub data_nanos: AtomicU64,
}

impl ZarrIoStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn total_bytes(&self) -> u64 {
        self.metadata_bytes.load(Ordering::Relaxed)
            + self.coord_bytes.load(Ordering::Relaxed)
            + self.data_bytes.load(Ordering::Relaxed)
    }

    pub fn total_arrays(&self) -> u64 {
        self.coord_arrays.load(Ordering::Relaxed) + self.data_arrays.load(Ordering::Relaxed)
    }

    pub fn metadata_time(&self) -> Duration {
        Duration::from_nanos(self.metadata_nanos.load(Ordering::Relaxed))
    }

    pub fn coord_time(&self) -> Duration {
        Duration::from_nanos(self.coord_nanos.load(Ordering::Relaxed))
    }

    pub fn data_time(&self) -> Duration {
        Duration::from_nanos(self.data_nanos.load(Ordering::Relaxed))
    }

    /// Record metadata read stats
    pub fn record_metadata(&self, bytes: u64, duration: Duration) {
        self.metadata_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.metadata_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Record coordinate array read stats
    pub fn record_coord(&self, bytes: u64, duration: Duration) {
        self.coord_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.coord_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.coord_arrays.fetch_add(1, Ordering::Relaxed);
    }

    /// Record data variable read stats
    pub fn record_data(&self, bytes: u64, duration: Duration) {
        self.data_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.data_nanos
            .fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        self.data_arrays.fetch_add(1, Ordering::Relaxed);
    }

    /// Record disk bytes read (actual I/O)
    pub fn record_disk_read(&self, bytes: u64) {
        self.disk_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get total disk bytes read
    pub fn total_disk_bytes(&self) -> u64 {
        self.disk_bytes.load(Ordering::Relaxed)
    }
}

/// Thread-safe handle for sharing stats across async boundaries
pub type SharedIoStats = Arc<ZarrIoStats>;

/// Format bytes in human-readable form (KB, MB, GB)
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}
