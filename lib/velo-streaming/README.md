# velo-streaming

Typed exclusive-attachment streaming abstraction over the velo transport.
One producer (`StreamSender`) pushes serialized frames to one consumer
(`StreamAnchor`) through an `AnchorManager` registry. Frames are
seven-variant enums (`StreamFrame<T>`) carrying items, sentinels, errors,
and heartbeats over a flume channel backed by pluggable transport.

## Concepts

### Streaming model

Typed exclusive-attachment streaming over velo transport. One producer
(`StreamSender<T>`) pushes frames to one consumer (`StreamAnchor<T>`)
through an `AnchorManager` registry. Items are serialized via `rmp_serde`
and travel as `Vec<u8>` through a `flume::bounded(256)` channel.

### Anchors

`StreamAnchor<T>` is created via `AnchorManager::create_anchor`. The
`.handle()` method returns a `StreamAnchorHandle` (a `u128` encoding
`WorkerId` in the upper 64 bits + local ID in the lower 64 bits) that can
be sent to any worker for attachment.

### Exclusive attachment

Only one `StreamSender` may attach to an anchor at a time. Enforced
atomically via `DashMap::entry` under the shard lock. After `detach()`, a
new sender can reattach to the same anchor.

### Sentinels

Terminal frames (`Finalized`, `Detached`, `Dropped`) travel the same
channel as data items, providing ordering guarantees. `Heartbeat` frames
are internal-only and never yielded to the consumer by `StreamAnchor`.

### Cancellation

Consumer-initiated upstream cancel via `StreamAnchor::cancel()` or
`StreamController::cancel()`. Fires a `_stream_cancel` active message to
the sender's worker. The sender observes cancellation through
`StreamSender::cancellation_token()`.

## Quick start

```rust,no_run
use futures::StreamExt;
use velo_streaming::{AnchorManager, StreamFrame};

async fn example(mgr: &AnchorManager) -> anyhow::Result<()> {
    // Consumer creates an anchor
    let mut anchor = mgr.create_anchor::<String>();
    let handle = anchor.handle();

    // Producer attaches (could be on a different worker)
    let sender = mgr.attach_stream_anchor::<String>(handle).await?;

    // Send items
    sender.send("hello".into()).await?;
    sender.send("world".into()).await?;
    sender.finalize()?;

    // Consume the stream
    while let Some(frame) = anchor.next().await {
        match frame {
            Ok(StreamFrame::Item(s)) => println!("{s}"),
            Ok(StreamFrame::Finalized) => break,
            Err(e) => eprintln!("stream error: {e}"),
            _ => {}
        }
    }

    Ok(())
}
```

## Key types

| Type | Role |
|------|------|
| `AnchorManager` | Central registry: creates anchors, attaches senders |
| `StreamAnchor<T>` | Consumer-side `Stream` impl yielding `Result<StreamFrame<T>, StreamError>` |
| `StreamSender<T>` | Producer-side sender with heartbeat and drop safety |
| `StreamAnchorHandle` | Compact `u128` wire handle for cross-worker routing |
| `StreamController` | Cloneable cancel handle for consumer-side upstream cancel |
| `StreamFrame<T>` | Seven-variant wire frame enum (Item, Finalized, Detached, Dropped, Heartbeat, SenderError, TransportError) |
| `AttachError` | Attach failures: not found, already attached, transport error |
| `SendError` | Send failures: channel closed, serialization error |
| `StreamError` | Consumer-side errors: sender dropped, transport error, deserialization error |

## Crate map

```text
velo-streaming
├── anchor      AnchorManager, StreamAnchor, AnchorEntry, AttachError, StreamController
├── sender      StreamSender with heartbeat task and drop safety
├── frame       StreamFrame<T>, StreamError, SendError
├── handle      StreamAnchorHandle (u128 wire encoding)
├── control     Control-plane AM handlers and wire types
├── transport   FrameTransport trait (pluggable ordered-delivery)
└── velo_transport  VeloFrameTransport (AM-backed impl)
```
