# Agents

This file records AI agent contributions to the project (kept here instead of
`Co-Authored-By` trailers in commit messages).

## Contributors

- **Claude Opus 4.8** (Anthropic, via Claude Code)

## Contributions

- Ballista serialization support: `ZarrPhysicalCodec` and `ZarrLogicalCodec`
  (`src/physical_plan/codec.rs`), serde derives on the Zarr metadata types, and
  the embedded standalone Ballista end-to-end test
  (`tests/integration_ballista.rs`).
