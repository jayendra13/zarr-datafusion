#!/usr/bin/env sh
set -euo pipefail

# Configure git to use the repository's .githooks directory for hooks
git config core.hooksPath .githooks
# Ensure the hook is executable
chmod +x .githooks/pre-commit

echo "Installed pre-commit hook (git config core.hooksPath .githooks)."