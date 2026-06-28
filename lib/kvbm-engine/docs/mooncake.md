# Mooncake Store Integration

Mooncake Store is a distributed KV cache storage backend that integrates with the
KVBM G4 tier via the `ObjectBlockOps` trait.

## Module Structure

- **`client`** — `MooncakeObjectBlockClient` implementing `ObjectBlockOps` for
  put/get/has block operations via Mooncake Store. Supports both TCP copy and
  RDMA zero-copy transfer paths. Batch operations use Mooncake's native
  `batch_put_from`/`batch_get_into`/`batch_is_exist` APIs for reduced RPC
  overhead.

- **`lock`** — `MooncakeLockManager` providing optimistic (no-op) locking for
  Mooncake Store. Since KVBM block writes are content-deterministically
  idempotent (same hash = same content), concurrent writes don't require
  distributed locking. The lock manager always grants locks and ignores release
  requests.

## Transport Protocols

| Protocol | Feature | Description |
|----------|---------|-------------|
| TCP | Default | Standard network copy path |
| RDMA | `use_zero_copy` flag | Zero-copy via `put_from`/`get_into` + `register_buffer` |

The RDMA path requires calling `register_layout()` before transfers and
falls back to the copy path for non-contiguous layouts.

## Configuration

See `kvbm_config::MooncakeObjectConfig` for configuration options including
metadata server address, master server gRPC address, transport protocol,
segment sizes, and namespace isolation.

## Requirements

- Mooncake Store C library with Rust bindings (`mooncake_store` crate)
- For RDMA: NIXL environment with appropriate RDMA device drivers
- Feature flag: `mooncake` in `kvbm-engine`