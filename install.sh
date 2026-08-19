#!/usr/bin/env bash
# Install the zarr-cli binary from a GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/jayendra13/zarr-datafusion/main/install.sh | bash
#
# Options (env vars):
#   VERSION       release tag to install (default: the pinned DEFAULT_VERSION below)
#   INSTALL_DIR   where to put the binary  (default: ~/.local/bin)
#
# LINUX ONLY for now. The Linux build is a fully static musl binary, so it runs
# on any x86_64 Linux regardless of the system glibc version. macOS binaries are
# not published yet -- the script aborts there with build-from-source
# instructions rather than failing later on a 404.
#
# The version is PINNED rather than tracking `latest`: releases have shipped
# assets inconsistently, so a pinned tag is the only one known to be installable.
# Bump DEFAULT_VERSION when a new release is verified to carry its assets.
set -euo pipefail

REPO="jayendra13/zarr-datafusion"
DEFAULT_VERSION="v0.1.1"
VERSION="${VERSION:-$DEFAULT_VERSION}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BIN="zarr-cli"

err() { echo "error: $*" >&2; exit 1; }

# --- gate on platform, then map to the release asset target triple ---
# Only Linux is published today. Anything else aborts here with an explanation,
# instead of downloading a URL we know does not exist.
os="$(uname -s)"
arch="$(uname -m)"

unsupported() {
  cat >&2 <<MSG
error: no prebuilt zarr-cli for $1.

Only Linux binaries are published at the moment; a macOS build is planned but
not available yet. To use zarr-cli on this machine, build it from source
(requires the Rust toolchain):

    git clone https://github.com/${REPO}.git
    cd zarr-datafusion
    cargo build --release --bin ${BIN}
    # binary at target/release/${BIN}
MSG
  exit 1
}

case "$os" in
  Linux)
    case "$arch" in
      x86_64|amd64) target="x86_64-unknown-linux-musl" ;;
      aarch64|arm64) unsupported "Linux $arch (only x86_64 is published)" ;;
      *) unsupported "Linux $arch" ;;
    esac ;;
  Darwin)  unsupported "macOS ($arch)" ;;
  *)       unsupported "$os" ;;
esac

asset="${BIN}-${target}.tar.gz"
# `latest` stays supported as an explicit opt-in (VERSION=latest), but is not
# the default -- see the note at the top of this file.
if [ "$VERSION" = "latest" ]; then
  base="https://github.com/${REPO}/releases/latest/download"
else
  base="https://github.com/${REPO}/releases/download/${VERSION}"
fi
url="${base}/${asset}"

# --- download + verify + install ---
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading ${url}"
curl -fSL --proto '=https' --tlsv1.2 -o "${tmp}/${asset}" "$url" \
  || err "download failed (does release '${VERSION}' have asset '${asset}'?)"

# Verify checksum if the release ships one.
if curl -fsSL -o "${tmp}/${asset}.sha256" "${url}.sha256" 2>/dev/null; then
  echo "Verifying checksum"
  ( cd "$tmp" && sha256sum -c "${asset}.sha256" ) || err "checksum verification failed"
else
  echo "warning: no .sha256 published for ${asset}; skipping verification" >&2
fi

tar -xzf "${tmp}/${asset}" -C "$tmp"
mkdir -p "$INSTALL_DIR"
install -m 0755 "${tmp}/${BIN}" "${INSTALL_DIR}/${BIN}"

echo "Installed ${BIN} -> ${INSTALL_DIR}/${BIN}"
"${INSTALL_DIR}/${BIN}" --version || true

# --- PATH hint ---
case ":$PATH:" in
  *":${INSTALL_DIR}:"*) ;;
  *) echo
     echo "Note: ${INSTALL_DIR} is not on your PATH. Add this to your shell profile:"
     echo "    export PATH=\"${INSTALL_DIR}:\$PATH\"" ;;
esac
