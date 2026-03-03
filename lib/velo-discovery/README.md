# velo-discovery

Filesystem-based peer discovery for Velo distributed systems.

Provides `FilesystemPeerDiscovery`, an implementation of the
`velo_messenger::PeerDiscovery` trait that stores peer information in a JSON
file on disk. Suitable for development, testing, and single-host deployments
where an external service (etcd, consul) is not desired.

## Usage

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

## File format

The discovery file is a JSON object with a `peers` array:

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

## Concurrency

- **Cross-process**: uses `fs4` file locking (shared for reads, exclusive for
  writes) with atomic rename on write.
- **Within-process**: `RwLock` protects the in-memory cache; an `AsyncMutex`
  serializes write operations (register/unregister) to prevent interleaving.
- Cache is invalidated on writes and lazily reloaded on the next read.

## Manual peer management

Peers can be registered and unregistered directly without going through the
`PeerDiscovery` trait:

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
