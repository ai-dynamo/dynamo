# velo-discovery

Peer discovery abstraction and a file-based implementation for Velo distributed systems.

## PeerDiscovery trait

`velo_messenger::PeerDiscovery` provides a lookup interface over cluster membership:

| Input | Output |
|---|---|
| `WorkerId` | `PeerInfo` (contains `WorkerAddress`) |
| `InstanceId` | `PeerInfo` (contains `WorkerAddress`) |

Implement this trait to integrate with any backend (etcd, consul, a database, etc.).

```rust
pub trait PeerDiscovery: Send + Sync {
    fn discover_by_worker_id(&self, worker_id: WorkerId) -> BoxFuture<'_, Result<PeerInfo>>;
    fn discover_by_instance_id(&self, instance_id: InstanceId) -> BoxFuture<'_, Result<PeerInfo>>;
}
```

## FilesystemPeerDiscovery

A bundled implementation that stores peer records in a JSON file on disk.
**Suitable for development, testing, and single-host deployments.
Not recommended for production or multi-host deployments** — use an external
service (etcd, consul) and a matching `PeerDiscovery` implementation instead.

### Usage

```rust,no_run
use velo_discovery::FilesystemPeerDiscovery;
use velo_messenger::Messenger;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let discovery = FilesystemPeerDiscovery::new("/tmp/peers.json")?;

    let messenger = Messenger::builder()
        .add_transport(tcp_transport)
        .discovery(Arc::new(discovery))
        .build()
        .await?;

    Ok(())
}
```

For tests that need throwaway storage, use `new_temp()`:

```rust,no_run
let discovery = FilesystemPeerDiscovery::new_temp()?;
```

### File format

```json
{
  "peers": [
    {
      "instance_id": "uuid-string",
      "worker_id": 123,
      "worker_address": "<msgpack bytes>",
      "address_checksum": 12345678
    }
  ]
}
```

### Concurrency

- **Cross-process**: `fs4` file locking (shared for reads, exclusive for writes) with atomic rename on write.
- **Within-process**: `RwLock` protects the in-memory cache; an `AsyncMutex` serializes write operations.
- Cache is invalidated on writes and lazily reloaded on the next read.

### Manual peer management

```rust,no_run
use velo_discovery::FilesystemPeerDiscovery;

let discovery = FilesystemPeerDiscovery::new("/tmp/peers.json")?;

// Register — works from sync or async contexts
discovery.register_peer_info(&peer_info)?;

// Unregister
discovery.unregister_instance(instance_id)?;
```

## Tests

```sh
cargo test -p velo-discovery
```
