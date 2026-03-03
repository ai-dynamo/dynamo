# Object Storage Module

The object module provides traits and implementations for storing KV cache
blocks in object storage systems (S3, MinIO). This corresponds to the G4
(object store) tier in the storage hierarchy.

## ObjectBlockOps Trait

The primary trait for block-level object storage operations:

| Method | Purpose |
|--------|---------|
| `has_blocks(keys)` | Check existence and size of blocks |
| `put_blocks(keys, src_layout, block_ids)` | Upload blocks using logical layout handle |
| `get_blocks(keys, dst_layout, block_ids)` | Download blocks using logical layout handle |
| `put_blocks_with_layout(keys, layout, block_ids)` | Upload using resolved physical layout |
| `get_blocks_with_layout(keys, layout, block_ids)` | Download using resolved physical layout |

### Logical vs Physical Layout

The trait offers two APIs for put/get:

- **Logical** (`put_blocks` / `get_blocks`): Takes a `LogicalLayoutHandle` (G1, G2, G3).
  Workers resolve this to their own physical layout internally. Used by the leader
  (which doesn't have physical layouts) and by `CoordinatedWorker`.
- **Physical** (`put_blocks_with_layout` / `get_blocks_with_layout`): Takes a resolved
  `PhysicalLayout` directly. Used by `PhysicalWorker` after resolving its handles, and
  by `S3ObjectBlockClient` which performs the actual I/O.

## Key Formatting

Keys map `SequenceHash` values to object storage paths:

- **`DefaultKeyFormatter`**: Uses the hash's Display representation
  (e.g., `0:abc123`). Suitable for single-worker scenarios.
- **`RankPrefixedKeyFormatter`**: Prefixes with worker rank
  (e.g., `0/0:abc123`). Required for SPMD workers where multiple workers
  store the same logical block with different physical data.

The `create_key_formatter(rank)` factory returns the appropriate formatter.

## ObjectLockManager

Distributed locking protocol for coordinated offloads to prevent duplicate
uploads:

```text
has_meta(hash)
  → true  → skip (already offloaded)
  → false → try_acquire_lock(hash)
              → true  → transfer → create_meta(hash) → release_lock(hash)
              → false → skip (another instance owns it)
```

Uses conditional PUT (`If-None-Match: *`) for lock acquisition with deadline-based
expiry for stale lock recovery.

## S3 Implementation

The `s3` submodule (feature-gated behind `s3`) provides:

- **`S3ObjectBlockClient`**: Implements `ObjectBlockOps` for S3-compatible storage.
  Supports concurrent uploads/downloads via `rayon` thread pool and contiguous
  memory fast paths for aligned block data.
- **`S3LockManager`**: Implements `ObjectLockManager` using S3 conditional writes.

## Factory Functions

- **`create_object_client(config, rank)`**: Creates an `Arc<dyn ObjectBlockOps>`
  from configuration. Selects the backend (S3 or future alternatives) based on
  `ObjectClientConfig`.
- **`create_lock_manager(config, instance_id)`**: Creates an
  `Arc<dyn ObjectLockManager>` for distributed lock coordination.
