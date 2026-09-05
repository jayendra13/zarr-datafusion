#!/usr/bin/env bash
# Build zarr-cli locally as a static musl binary and upload it to a GitHub
# release, so the curl installer serves a fresh binary without waiting on CI.
#
#   scripts/release-local.sh [TAG]
#
# TAG defaults to "v<version-in-Cargo.toml>". The release is created if missing;
# existing assets are replaced (--clobber). After it runs, this works anywhere:
#
#   curl -fsSL https://raw.githubusercontent.com/stratoscale-io/zarr-datafusion/main/install.sh | bash
#
# The x86_64 musl target is fully static, so it runs on any x86_64 Linux
# regardless of the system glibc. For a full multi-platform release (macOS,
# Windows, gnu), push a "v*" git tag instead and let .github/workflows/publish.yml
# build everything.
set -euo pipefail

REPO="stratoscale-io/zarr-datafusion"
BIN="zarr-cli"
TARGET="x86_64-unknown-linux-musl"

# Run from the repo root regardless of where the script is invoked from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

err() { echo "error: $*" >&2; exit 1; }

command -v gh >/dev/null 2>&1 || err "the GitHub CLI ('gh') is required"
gh auth status >/dev/null 2>&1 || err "run 'gh auth login' first"
command -v cargo-zigbuild >/dev/null 2>&1 \
  || err "cargo-zigbuild is required for the musl build (cargo install cargo-zigbuild; and install 'zig')"

VERSION="$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')"
TAG="${1:-v$VERSION}"

echo "==> Building ${BIN} (${TARGET}) in release mode"
rustup target add "$TARGET" >/dev/null 2>&1 || true
cargo zigbuild --release --bin "$BIN" --target "$TARGET"

echo "==> Packaging"
asset="${BIN}-${TARGET}.tar.gz"
tar czf "$asset" -C "target/${TARGET}/release" "$BIN"
sha256sum "$asset" > "${asset}.sha256"
echo "    $asset ($(du -h "$asset" | cut -f1))"

echo "==> Publishing to release ${TAG}"
if gh release view "$TAG" >/dev/null 2>&1; then
  gh release upload "$TAG" "$asset" "${asset}.sha256" --clobber
else
  gh release create "$TAG" \
    --title "$TAG" \
    --notes "Local static musl build of ${BIN} (${TARGET})." \
    "$asset" "${asset}.sha256"
fi

rm -f "$asset" "${asset}.sha256"

echo
echo "Done. Verify from any x86_64 Linux machine with:"
echo "    curl -fsSL https://raw.githubusercontent.com/${REPO}/main/install.sh | VERSION=${TAG} bash"
