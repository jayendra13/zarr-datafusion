# Ballista Cluster Support for zarr-datafusion

## Overview

This document identifies the changes needed to run zarr-datafusion on an [Apache DataFusion Ballista](https://datafusion.apache.org/ballista/user-guide/introduction.html) cluster for distributed query execution.

## Key Insight: Ballista Requirements

Ballista distributes physical execution plans across nodes. Each custom `ExecutionPlan` must:
1. Be serializable to protobuf bytes via `PhysicalExtensionCodec`
2. Be reconstructable on any worker node from those bytes
3. Have all non-serializable state (stores, connections) recreated at execution time

## Current State Analysis

### Non-Serializable Components in ZarrExec

| Field | Type | Serializable | Solution |
|-------|------|--------------|----------|
| `schema` | `SchemaRef` | Yes | Use DataFusion's schema serde |
| `path` | `String` | Yes | Direct |
| `projection` | `Option<Vec<usize>>` | Yes | Direct |
| `limit` | `Option<usize>` | Yes | Direct |
| `properties` | `PlanProperties` | Yes | Reconstruct from schema |
| `io_stats` | `SharedIoStats` | **No** | Create fresh on each node |
| `cached_remote` | `CachedRemoteStore` | **No** | Recreate from path on execute() |
| `coord_filters` | `Option<CoordFilters>` | **No** (needs serde) | Add serde, serialize |

### Non-Serializable in ZarrTable

`ZarrTable` is not serialized by Ballista - only `ZarrExec` is distributed. The scheduler creates the physical plan (including `ZarrExec`) and distributes it to workers.

---

## Implementation Plan

### 1. Add Dependencies (Feature-Gated)

**File: `Cargo.toml`**

```toml
[features]
default = []
ballista = ["dep:ballista", "dep:ballista-core", "dep:prost", "dep:datafusion-proto"]

[dependencies]
serde = { version = "1.0", features = ["derive"] }  # Always needed for serde on types
prost = { version = "0.13", optional = true }
datafusion-proto = { version = "51.0.0", optional = true }
ballista = { version = "0.13", optional = true }
ballista-core = { version = "0.13", optional = true }

[build-dependencies]
prost-build = "0.13"  # Only runs when proto files exist
```

**Usage:**
```bash
cargo build --features ballista  # Enable Ballista support
```

### 2. Add Serde to Core Types

**File: `src/reader/filter.rs`**

Add `Serialize`/`Deserialize` to:
- `CoordFilter`
- `CoordFilters`

Note: `ScalarValue` from DataFusion already has serde support in `datafusion-proto`.

**File: `src/reader/schema_inference.rs`**

Add `Serialize`/`Deserialize` to:
- `ZarrArrayMeta`
- `ZarrStoreMeta`

### 3. Create Protobuf Definitions

**New file: `proto/zarr_exec.proto`**

```protobuf
syntax = "proto3";
package zarr_datafusion;

message ZarrExecNode {
  bytes schema = 1;  // Arrow schema serialized
  string path = 2;
  repeated uint32 projection = 3;
  optional uint64 limit = 4;
  optional ZarrStoreMeta store_meta = 5;
  repeated CoordFilterEntry coord_filters = 6;
}

message ZarrStoreMeta {
  repeated ZarrArrayMeta coords = 1;
  repeated ZarrArrayMeta data_vars = 2;
  uint64 total_rows = 3;
}

message ZarrArrayMeta {
  string name = 1;
  string data_type = 2;
  repeated uint64 shape = 3;
  optional double min_value = 4;
  optional double max_value = 5;
}

message CoordFilterEntry {
  string coord_name = 1;
  bytes scalar_value = 2;  // Serialized ScalarValue
}
```

### 4. Create Build Script

**New file: `build.rs`**

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::compile_protos(&["proto/zarr_exec.proto"], &["proto/"])?;
    Ok(())
}
```

### 5. Implement PhysicalExtensionCodec (Feature-Gated)

**New file: `src/codec.rs`**

```rust
#[cfg(feature = "ballista")]
pub struct ZarrPhysicalCodec;

impl PhysicalExtensionCodec for ZarrPhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        registry: &dyn FunctionRegistry,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // 1. Decode protobuf message
        // 2. Reconstruct schema from bytes
        // 3. Create ZarrExec with:
        //    - schema, path, projection, limit from proto
        //    - cached_remote = None (will be created in execute())
        //    - io_stats = fresh Arc::new(ZarrIoStats::new())
        //    - coord_filters from proto
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        // 1. Downcast to ZarrExec
        // 2. Serialize schema to bytes
        // 3. Build ZarrExecNode protobuf
        // 4. Encode to buf
    }
}
```

### 6. Modify ZarrExec for Lazy Store Creation

**File: `src/physical_plan/zarr_exec.rs`**

The `execute()` method already handles the case where `cached_remote` is `None` - it creates the store from the path. This means:
- When serialized, we don't serialize `cached_remote`
- When deserialized, `cached_remote = None`
- On `execute()`, the worker creates the store from the path

This is the existing code path (lines 221-227):
```rust
} else {
    debug!("Creating async store (no cache)");
    let (store, prefix) = create_async_store(&path)
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    (store, prefix, None)
}
```

**Enhancement needed**: Pass `ZarrStoreMeta` through serialization to avoid re-discovering metadata on each worker.

### 7. Serialize Store Metadata

To avoid each worker re-discovering array metadata (expensive for remote stores), serialize `ZarrStoreMeta` in the plan:

**File: `src/physical_plan/zarr_exec.rs`**

Add a field:
```rust
pub struct ZarrExec {
    // ... existing fields ...
    store_meta: Option<ZarrStoreMeta>,  // NEW: serialized metadata
}
```

Then in `execute()` for remote stores, use the pre-computed metadata instead of re-discovering.

### 8. Credential Handling for Distributed Execution

**Worker Node Requirements:**

Each Ballista executor node needs environment-based credentials:

| Store | Environment Variables |
|-------|----------------------|
| S3 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` |
| GCS | `GOOGLE_SERVICE_ACCOUNT` or Application Default Credentials |

**No code changes needed** - the existing `create_async_store()` function reads from environment.

**Deployment note**: Configure these in Ballista executor deployment (e.g., Kubernetes secrets, Docker env, etc.)

### 9. Register Codec with Ballista

**Example usage in application code:**

```rust
use ballista::prelude::*;
use zarr_datafusion::codec::ZarrPhysicalCodec;

let config = SessionConfig::new()
    .with_ballista_physical_extension_codec(Arc::new(ZarrPhysicalCodec::default()));

let ctx = SessionContext::new_with_config(config);
// or connect to Ballista scheduler
```

---

## Files to Create/Modify

| File | Action | Feature-Gated | Description |
|------|--------|---------------|-------------|
| `Cargo.toml` | Modify | - | Add feature flags, optional deps |
| `build.rs` | Create | - | Protobuf compilation |
| `proto/zarr_exec.proto` | Create | - | Protobuf message definitions |
| `src/lib.rs` | Modify | `#[cfg(feature = "ballista")]` | Conditionally export codec module |
| `src/codec.rs` | Create | `#[cfg(feature = "ballista")]` | PhysicalExtensionCodec implementation |
| `src/reader/filter.rs` | Modify | - | Add serde derives to CoordFilters |
| `src/reader/schema_inference.rs` | Modify | - | Add serde derives to ZarrStoreMeta |
| `src/physical_plan/zarr_exec.rs` | Modify | - | Add store_meta field, accessor methods |

---

## Testing Strategy

1. **Unit tests**: Round-trip serialization of ZarrExec
2. **Integration test**: Single-node Ballista execution
3. **Distributed test**: Multi-node Ballista cluster (requires infra setup)

---

## What Stays the Same

- `ZarrTable` (TableProvider) - not serialized by Ballista
- Optimizer rules (MinMaxStatisticsRule, CountStatisticsRule) - run on scheduler
- Filter pushdown logic - already in ZarrExec
- Storage backends - recreated on each worker from path

---

## Summary of Architectural Approach

**Reconstruct, don't serialize** non-serializable state:
- `io_stats`: Fresh counter per worker (natural for distributed metrics)
- `cached_remote`: Recreate from path string + environment credentials
- `store_meta`: Serialize in plan to avoid re-discovery

This aligns with Ballista's distributed execution model where each executor is independent.
