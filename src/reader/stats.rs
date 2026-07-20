//! I/O statistics for Zarr reads
//!
//! Tracks bytes read, arrays accessed, and timing breakdown for metadata,
//! coordinates, and data variables.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

    /// Whether a byte-counting store was actually installed for this read.
    ///
    /// Without this we cannot tell "read nothing" from "never measured": both leave
    /// `disk_bytes` at 0, and a reported `0 B` on a path that does its own I/O
    /// (icechunk, VirtualiZarr) reads as a measurement when it is an absence of one.
    /// Set by the tracking store wrappers at construction; see [`disk_bytes_tracked`].
    ///
    /// [`disk_bytes_tracked`]: ZarrIoStats::disk_bytes_tracked
    pub disk_tracked: AtomicBool,

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

    /// Declare that a byte-counting store is in place for this read.
    ///
    /// Called by the tracking store wrappers when they are constructed with a stats
    /// handle. Paths that bypass those wrappers (icechunk and VirtualiZarr do their
    /// own object I/O) never call this, so they report "not measured" rather than 0.
    pub fn mark_disk_tracked(&self) {
        self.disk_tracked.store(true, Ordering::Relaxed);
    }

    /// Total disk bytes read, or `None` when nothing was counting.
    ///
    /// Prefer this over [`total_disk_bytes`] for anything user-facing: a bare `0`
    /// is indistinguishable from an unmeasured read, and reporting an absent
    /// measurement as zero bytes overstates how little I/O a query did.
    ///
    /// [`total_disk_bytes`]: ZarrIoStats::total_disk_bytes
    pub fn disk_bytes_tracked(&self) -> Option<u64> {
        self.disk_tracked
            .load(Ordering::Relaxed)
            .then(|| self.disk_bytes.load(Ordering::Relaxed))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untracked_reads_are_not_reported_as_zero_bytes() {
        // The distinction that matters for anything user-facing: a path that never
        // installed a counting store (icechunk, VirtualiZarr) must report "unknown",
        // not "0 bytes" — the latter reads as a measurement showing no I/O.
        let stats = ZarrIoStats::new();
        assert_eq!(stats.disk_bytes_tracked(), None);
        assert_eq!(stats.total_disk_bytes(), 0);
    }

    #[test]
    fn a_tracked_read_of_nothing_reports_zero_not_unknown() {
        // The converse: once a counting store is installed, 0 is a real answer.
        let stats = ZarrIoStats::new();
        stats.mark_disk_tracked();
        assert_eq!(stats.disk_bytes_tracked(), Some(0));
    }

    #[test]
    fn tracked_reads_accumulate() {
        let stats = ZarrIoStats::new();
        stats.mark_disk_tracked();
        stats.record_disk_read(1_000);
        stats.record_disk_read(2_500);
        assert_eq!(stats.disk_bytes_tracked(), Some(3_500));
    }

    #[test]
    fn recording_without_marking_still_reports_unknown() {
        // Guards the wiring: `record_disk_read` alone must not imply tracking, so a
        // partially-instrumented path can't masquerade as a complete measurement.
        let stats = ZarrIoStats::new();
        stats.record_disk_read(42);
        assert_eq!(stats.disk_bytes_tracked(), None);
    }
}
