# velo

Top-level facade for the Velo distributed systems stack. Re-exports the public
API from the underlying crates so consumers only need a single dependency.

## Crate map

```
velo              ← you are here (facade + re-exports)
├── velo-messenger   active messaging, handlers, distributed events
├── velo-events      generational event system
├── velo-common      shared types (InstanceId, PeerInfo, WorkerId)
└── velo-backend     transport abstraction (TCP, gRPC, NATS, UCX)
```

Peer discovery backends (e.g. `velo-discovery`) are separate crates that
implement the `PeerDiscovery` trait and are passed in at build time.

## Quick start

```rust,no_run
use std::sync::Arc;
use velo::{Velo, Handler, Context};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let transport = /* create a transport (e.g. TcpTransport) */;

    let node = Velo::builder()
        .add_transport(transport)
        .build()
        .await?;

    // Register a handler
    let handler = Handler::unary_handler("ping", |_ctx: Context| {
        Ok(Some(bytes::Bytes::from("pong")))
    }).build();
    node.register_handler(handler)?;

    // Send a request to a peer
    let peer = node.instance_id(); // (normally a different instance)
    let resp = node.unary("ping")?
        .raw_payload(bytes::Bytes::new())
        .instance(peer)
        .send()
        .await?;

    Ok(())
}
```

## What's re-exported

| Source crate | Key types |
|---|---|
| `velo-messenger` | `Messenger`, `Handler`, `Context`, `TypedContext`, send/unary/typed-unary builders, `VeloEvents`, `PeerDiscovery` |
| `velo-events` | `Event`, `EventManager`, `EventHandle`, `EventAwaiter`, `EventStatus`, `EventPoison` |
| `velo-common` | `InstanceId`, `PeerInfo`, `WorkerId`, `WorkerAddress` |

The `Velo` struct itself is a thin wrapper around `Arc<Messenger>` that
delegates every method. Use `velo.messenger()` if you need the inner
`Messenger` directly.

## Tests

Integration tests live in `tests/` and exercise messaging and distributed
events across two `Velo` instances connected over TCP:

```sh
cargo test -p velo
```
