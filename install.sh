#!/usr/bin/env bash
# Install the zarr-cli binary from a GitHub release.
#
#   curl -fsSL https://raw.githubusercontent.com/jayendra13/zarr-datafusion/main/install.sh | bash
#
# Options (env vars):
#   VERSION       release tag to install (default: latest)         e.g. VERSION=v0.1.0
#   INSTALL_DIR   where to put the binary  (default: ~/.local/bin)
#
# The linux build is a fully static musl binary, so it runs on any x86_64 Linux
# regardless of the system glibc version.
set -euo pipefail

REPO="jayendra13/zarr-datafusion"
VERSION="${VERSION:-latest}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BIN="zarr-cli"

err() { echo "error: $*" >&2; exit 1; }

# --- detect platform -> release asset target triple ---
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
  Linux)
    case "$arch" in
      x86_64|amd64) target="x86_64-unknown-linux-musl" ;;
      aarch64|arm64) target="aarch64-unknown-linux-musl" ;;
      *) err "unsupported Linux arch: $arch" ;;
    esac ;;
  Darwin)
    case "$arch" in
      x86_64) target="x86_64-apple-darwin" ;;
      arm64)  target="aarch64-apple-darwin" ;;
      *) err "unsupported macOS arch: $arch" ;;
    esac ;;
  *) err "unsupported OS: $os (use the Windows .zip asset manually)" ;;
esac

asset="${BIN}-${target}.tar.gz"
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
