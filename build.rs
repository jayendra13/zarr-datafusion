//! Build script: bake a full version string into the binaries.
//!
//! The package version (`CARGO_PKG_VERSION`, e.g. `0.1.0`) is combined with the
//! short hash of the current commit as SemVer 2.0 *build metadata*:
//!
//!     0.1.0+g2d7f1be          clean tree
//!     0.1.0+g2d7f1be.dirty    uncommitted changes present
//!
//! A git short hash is not a number, so it cannot be the SemVer *patch*; build
//! metadata (everything after `+`) is the correct place for it. The result is
//! exposed to the crate as the `ZARR_BUILD_VERSION` env var via `env!()`.
//!
//! CI can override the embedded hash by setting `ZARR_GIT_SHA` (useful when the
//! `.git` dir isn't present in the build context); otherwise we shell out to git.

use std::process::Command;

fn main() {
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());

    let sha = std::env::var("ZARR_GIT_SHA")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .or_else(git_short_hash);

    let version = match sha {
        Some(sha) => {
            let dirty = if git_is_dirty() { ".dirty" } else { "" };
            format!("{pkg_version}+g{sha}{dirty}")
        }
        // No git info available (e.g. building from a published crate tarball):
        // fall back to the bare package version.
        None => pkg_version,
    };

    println!("cargo:rustc-env=ZARR_BUILD_VERSION={version}");

    // Re-run if the commit changes so the embedded hash stays current.
    println!("cargo:rerun-if-env-changed=ZARR_GIT_SHA");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs");
}

fn git_short_hash() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn git_is_dirty() -> bool {
    Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false)
}
