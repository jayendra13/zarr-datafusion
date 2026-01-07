# Contributing

Thanks for contributing! To keep the repository consistent and avoid regressions we provide a Git pre-commit hook that runs:

- `cargo fmt --all -- --check` (formatting)
- `cargo clippy --all-targets -- -D warnings` (lints)
- `cargo test --workspace` (tests)

How to enable the hook (recommended):

1. Run the install script from the repository root:

   ```sh
   ./scripts/install-hooks.sh
   ```

   This sets `git config core.hooksPath .githooks` and makes the pre-commit hook executable.

2. Verify with a test commit or run the checks manually:

   ```sh
   cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings && cargo test --workspace
   ```

If you prefer not to enable the hook globally for the repository, you can still run the checks locally before committing.

Thanks for keeping the codebase healthy!

## Releasing to crates.io

Releases are automated via GitHub Actions. To publish a new version:

1. **Update the version** in `Cargo.toml`:
   ```toml
   version = "0.2.0"
   ```

2. **Commit and push** the version bump:
   ```sh
   git add Cargo.toml
   git commit -m "Bump version to 0.2.0"
   git push origin main
   ```

3. **Create and push a tag** matching the version:
   ```sh
   git tag v0.2.0
   git push origin v0.2.0
   ```

The `publish.yml` workflow will:
- Run the full CI test suite
- Verify the tag version matches `Cargo.toml`
- Publish to crates.io

### Prerequisites

A `CRATES_TOKEN` secret must be configured in the repository settings:
1. Generate a token at https://crates.io/settings/tokens
2. Add it as a repository secret: Settings → Secrets → Actions → New repository secret

### Manual/Dry-run Publishing

You can trigger a dry-run publish manually via the Actions tab:
1. Go to Actions → "Publish to crates.io"
2. Click "Run workflow"
3. Check "Dry run" to test without publishing