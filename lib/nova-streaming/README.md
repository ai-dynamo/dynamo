# nova-streaming

Streaming abstractions over Nova's transport layer.

## Features

- **Attachable streams** - Receivers create anchors, senders attach and stream data
- **Multiple writers** - Multiple senders can attach/detach from same anchor
- **Control plane** - Attach/Cancel via Nova RPC
- **Data plane** - Data/Heartbeat/Finalize frames via Nova AM
- **Backpressure** - Bounded channels with `try_send()` for hot paths
- **Timeouts** - Configurable attach and message timeouts
- **Cancellation** - Receivers can cancel streams, notifying senders

## Quick Start

### Receiver Side

```rust,ignore
use dynamo_nova_streaming::{NovaStreaming, StreamFrame};

let streaming = NovaStreaming::builder(nova.clone()).build()?;

// Create stream and get handle
let (receiver, handle) = streaming.create_stream::<MyData>();

// Send handle to sender (e.g., in RPC response)
send_to_sender(handle);

// Consume frames
while let Some(frame) = receiver.recv().await {
    match frame {
        StreamFrame::Data(data) => process(data),
        StreamFrame::Finalized => break,
        StreamFrame::Cancelled(reason) => break,
        _ => {}
    }
}
```

### Sender Side

```rust,ignore
use dynamo_nova_streaming::{NovaStreaming, StreamHandle};

let handle: StreamHandle = receive_from_receiver();

let sender = streaming.arm::<MyData>(handle).attach().await?;

for item in items {
    sender.send(item).await?;
}

sender.finalize().await?;
```

### Hot Path (Zero-Copy)

```rust,ignore
let mode = sender.send_mode();

loop {
    match mode.try_send(item) {
        Ok(()) => continue,
        Err(TrySendError::Full(item)) => {
            mode.progress()?;
            tokio::task::yield_now().await;
        }
        Err(e) => return Err(e.into()),
    }
}
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `NovaStreaming` | Entry point wrapping Nova |
| `StreamReceiver<T>` | Receives frames from senders |
| `StreamSender<T>` | Sends frames to receiver |
| `StreamHandle` | Wire-transportable anchor ID (16 bytes) |
| `StreamFrame<T>` | Frame enum: Data, Heartbeat, Finalized, etc. |

### Nova Handlers

| Handler | Type | Purpose |
|---------|------|---------|
| `stream_attach` | RPC | Sender attach request |
| `stream_cancel` | RPC | Receiver cancellation |
| `stream_data` | AM | Frame delivery |
