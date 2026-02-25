# TCP Transport Design

## Overview

The TCP transport provides high-performance, zero-copy message delivery over raw TCP connections. It uses a custom 11-byte frame preamble for minimal overhead and supports CPU pinning for predictable latency.

## Connection Management

### DashMap Connection Pool

Peer connections are managed via two `DashMap` instances:

- `peers: DashMap<InstanceId, SocketAddr>` — registered peer addresses
- `connections: DashMap<InstanceId, ConnectionHandle>` — active writer task handles

Connections are established lazily on first `send_message()` call. Each connection spawns a dedicated writer task that owns the TCP stream.

### Writer Tasks

Each `ConnectionHandle` wraps a bounded `flume::Sender<SendTask>` (default capacity: 256). The send path:

1. **Fast path**: `try_send()` on existing connection — non-blocking, no allocation
2. **Slow path (full)**: `send_async()` via spawned task — applies backpressure
3. **Slow path (new)**: `get_or_create_connection()` — establishes TCP connection and spawns writer

The writer task loop:
```
recv_async(SendTask) → encode_frame(&mut stream, ...) → loop
```

## TcpFrameCodec

### Wire Format

```
[u16 BE: schema_version(1)] [u8: frame_type] [u32 BE: header_len] [u32 BE: payload_len] [header] [payload]
```

Total preamble: 11 bytes. Maximum frame: 16 MB.

### Decoder State Machine

The codec uses a two-state decoder for streaming TCP:

```
AwaitingHeader ──(11 bytes available)──→ AwaitingData ──(data available)──→ emit frame, reset
```

Zero-copy is achieved via `BytesMut::split_to().freeze()` — the output `Bytes` share the underlying receive buffer.

### Encoder

`encode_frame()` writes three segments via `write_all()`:
1. Preamble (11 bytes)
2. Header bytes
3. Payload bytes

`write_vectored()` is intentionally not used because it doesn't guarantee writing all bytes for payloads exceeding the kernel send buffer (~128KB).

## TCP Listener

### Frame Routing

Incoming frames are routed based on `MessageType`:

| MessageType | Target Stream |
|------------|---------------|
| Message | `message_stream` |
| Response | `response_stream` |
| Ack, Event | `event_stream` |
| ShuttingDown | `response_stream` (for correlation) |

### Drain Behavior

During drain (`ShutdownState::is_draining()`):
- **Message** frames are rejected: a `ShuttingDown` frame is sent back with the original header for correlation
- **Response/Ack/Event** frames pass through normally

### CPU Pinning (Linux)

`RuntimeConfig::CpuPin(cpu_id)` creates a single-threaded tokio runtime with the thread pinned to the specified CPU core via `nix::sched::sched_setaffinity`. This provides predictable latency by avoiding context switches.

On non-Linux platforms, `CpuPin` falls back to a regular single-threaded runtime with a warning.

## Socket Configuration

Both listener and writer sides configure:

- `TCP_NODELAY` — disable Nagle's algorithm for low-latency framing
- `SO_LINGER(1s)` — ensure clean socket shutdown
- `TCP_KEEPALIVE` — 60s idle time, 10s probe interval
- **Buffer sizes** — 1 MB send/receive buffers for high throughput
