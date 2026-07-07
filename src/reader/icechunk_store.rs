//! Icechunk store support (feature `icechunk`).
//!
//! Icechunk repos are versioned Zarr stores. We open one read-only at its `main`
//! branch tip and expose it as a [`zarrs`] async storage backend
//! ([`AsyncIcechunkStore`], which implements the same
//! `AsyncReadable*`/`AsyncListable*` traits our object-store backends do). That
//! lets the existing async schema-inference and read paths treat an icechunk
//! store exactly like any other remote Zarr store — it drops straight into
//! `ZarrTable`'s `cached_remote` slot.
//!
//! This module is deliberately narrow: detection + open. It does not know about
//! virtual chunks (icechunk resolves those internally) or groups (the first
//! integration target, `data/synthetic.icechunk`, is flat at the store root).
//!
//! Known trade-off: icechunk performs its own object I/O, so our
//! `AsyncTrackedStore` byte statistics do not cover reads that go through here.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use icechunk::repository::VersionInfo;
use icechunk::{new_local_filesystem_storage, Repository, RepositoryConfig};
use tracing::{debug, info};
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_icechunk::AsyncIcechunkStore;

/// Error opening an icechunk store.
type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Detect a local icechunk repository by its on-disk layout.
///
/// A local-filesystem icechunk repo has a `snapshots/` directory and a top-level
/// `repo` marker file. (Unlike a remote/object-store icechunk repo it has **no**
/// `refs/` directory — those refs are inlined — so we key on `snapshots/`, which
/// is present in both layouts, plus the `repo` marker to avoid matching an
/// arbitrary directory that merely happens to contain a `snapshots` subdir.)
pub fn is_icechunk_store(path: &str) -> bool {
    let root = Path::new(path);
    let has_snapshots = root.join("snapshots").is_dir();
    let has_repo_marker = root.join("repo").is_file();
    let detected = has_snapshots && has_repo_marker;
    if detected {
        debug!(
            path,
            "Detected local icechunk store (snapshots/ + repo marker)"
        );
    }
    detected
}

/// Open a local icechunk repository read-only at its `main` branch tip and return
/// it as a [`zarrs`] async storage backend.
///
/// The returned store can be handed to `infer_schema_with_meta_async` and the
/// async read path unchanged.
pub async fn open_icechunk_async(path: &str) -> Result<AsyncReadableListableStorage, BoxError> {
    info!(path, "Opening local icechunk store");
    let storage = new_local_filesystem_storage(Path::new(path)).await?;
    // Default config: local native + inlined chunks need no virtual-chunk
    // containers or credentials. (Virtual/remote containers arrive in a later phase.)
    let repo = Repository::open(Some(RepositoryConfig::default()), storage, HashMap::new()).await?;
    let session = repo
        .readonly_session(&VersionInfo::BranchTipRef("main".to_string()))
        .await?;
    // `Arc<AsyncIcechunkStore>` coerces to `AsyncReadableListableStorage` via the
    // blanket `AsyncReadableListableStorageTraits` impl.
    let store: AsyncReadableListableStorage = Arc::new(AsyncIcechunkStore::new(session));
    debug!(
        path,
        "Opened icechunk readonly session at branch tip 'main'"
    );
    Ok(store)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zarrs::storage::AsyncListableStorageTraits;

    const SYNTHETIC: &str = "data/synthetic.icechunk";

    #[test]
    fn detects_local_icechunk_layout() {
        assert!(
            is_icechunk_store(SYNTHETIC),
            "data/synthetic.icechunk should be detected as an icechunk store"
        );
        // A plain directory (the repo root) is not an icechunk store.
        assert!(!is_icechunk_store("."));
        assert!(!is_icechunk_store("data"));
    }

    #[tokio::test]
    async fn opens_and_lists_root_arrays() {
        let store = open_icechunk_async(SYNTHETIC)
            .await
            .expect("open synthetic.icechunk");
        // The store must list the four arrays the M1 generator wrote at the root.
        let keys = store.list().await.expect("list store keys");
        let joined = keys
            .iter()
            .map(|k| k.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for name in ["time", "lat", "lon", "temperature"] {
            assert!(
                joined.contains(&format!("{name}/zarr.json")),
                "expected {name}/zarr.json in store listing, got:\n{joined}"
            );
        }
    }
}
