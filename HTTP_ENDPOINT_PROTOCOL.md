# HTTP Request Plane Protocol Specification

## Overview

The HTTP request plane uses the same message encoding as NATS for compatibility, but delivers it over HTTP/2 POST requests.

## Message Format

All HTTP POST requests to the endpoint must contain a **TwoPartCodec-encoded** payload:

```
┌──────────────────────────────────┐
│   Length Prefix (u64, 8 bytes)   │  ← Control header length
├──────────────────────────────────┤
│   Control Header (JSON bytes)    │  ← RequestControlMessage
├──────────────────────────────────┤
│   Length Prefix (u64, 8 bytes)   │  ← Request data length
├──────────────────────────────────┤
│   Request Data (JSON bytes)      │  ← Actual request payload
└──────────────────────────────────┘
```

### Control Header (RequestControlMessage)

```json
{
  "id": "unique-request-id",
  "request_type": "single_in",
  "response_type": "many_out",
  "connection_info": {
    "transport": "tcp",
    "info": "{\"address\":\"127.0.0.1:9000\",\"subject\":\"uuid-xyz\",\"context\":\"ctx-123\",\"stream_type\":\"response\"}"
  }
}
```

### Request Data

The actual request payload, JSON-serialized:

```json
{
  "message": "Hello, world!"
}
```

## HTTP Request

### Endpoint URL

```
POST http://{host}:{port}{root_path}/{subject}
```

Example:
```
POST http://localhost:8081/v1/rpc/namespace.example.component.echo-worker.endpoint.echo.instance.0
```

### Headers

- `Content-Type: application/octet-stream`
- `traceparent: 00-{trace-id}-{span-id}-{flags}` (optional, for distributed tracing)
- `tracestate: {state}` (optional)
- `x-request-id: {id}` (optional)
- `x-dynamo-request-id: {id}` (optional)

### Body

Binary data encoded with TwoPartCodec (see Message Format above).

## HTTP Response

The endpoint returns **202 Accepted** immediately, indicating the request was received and will be processed asynchronously.

```http
HTTP/1.1 202 Accepted
Content-Length: 0
```

The actual response data is streamed back over TCP (not HTTP) using the connection information provided in the control header.

## Why Not Simple JSON?

You might wonder why we don't just accept simple JSON POST requests. The reasons are:

1. **Protocol Compatibility**: The HTTP and NATS request planes must be interchangeable
2. **Connection Info**: We need to tell the worker where to stream responses (TCP endpoint)
3. **Request Metadata**: Control headers contain request type, ID, and routing information
4. **Response Streaming**: Responses stream over TCP for better performance with large/long-running responses

## How to Send Requests

### Using PushRouter (Recommended)

```rust
use dynamo_runtime::pipeline::network::egress::push_router::{PushRouter, RouterMode};

let router = PushRouter::<MyRequest, MyResponse>::from_client(
    client,
    RouterMode::RoundRobin,
).await?;

let mut stream = router.generate(request.into()).await?;
while let Some(response) = stream.next().await {
    // Handle response
}
```

The PushRouter automatically:
- Encodes messages with TwoPartCodec
- Manages TCP response streams
- Handles service discovery
- Provides load balancing

### Using Raw HTTP (Advanced)

If you need to send raw HTTP requests:

```rust
use dynamo_runtime::pipeline::network::codec::{TwoPartCodec, TwoPartMessage};
use dynamo_runtime::pipeline::network::ConnectionInfo;

// 1. Register TCP response stream
let tcp_server = /* get TCP server */;
let options = StreamOptions::builder()
    .context(ctx)
    .enable_response_stream(true)
    .build()?;
let pending = tcp_server.register(options).await;
let (connection_info, stream_provider) = pending.recv_stream.unwrap().into_parts();

// 2. Build control message
let control_msg = RequestControlMessage {
    id: ctx.id().to_string(),
    request_type: RequestType::SingleIn,
    response_type: ResponseType::ManyOut,
    connection_info,
};

// 3. Encode with TwoPartCodec
let ctrl_bytes = serde_json::to_vec(&control_msg)?;
let data_bytes = serde_json::to_vec(&my_request)?;
let msg = TwoPartMessage::from_parts(ctrl_bytes.into(), data_bytes.into());
let codec = TwoPartCodec::default();
let encoded = codec.encode_message(msg)?;

// 4. Send HTTP POST
let response = reqwest::Client::new()
    .post("http://localhost:8081/v1/rpc/namespace.example.component.worker.endpoint.echo.instance.0")
    .header("Content-Type", "application/octet-stream")
    .body(encoded)
    .send()
    .await?;

// 5. Await TCP response stream
let tcp_stream = stream_provider.await??;
// Handle responses from TCP stream
```

## Response Flow

```
[Client]                   [HTTP Server]              [Worker]
   │                            │                         │
   │─1. Register TCP stream─────┤                         │
   │                            │                         │
   │─2. HTTP POST (encoded)─────>                         │
   │                            │                         │
   │<─3. 202 Accepted───────────┤                         │
   │                            │                         │
   │                            │─4. Handle payload──────>│
   │                            │                         │
   │<─5. TCP connect (call-home)─────────────────────────┤
   │                            │                         │
   │<─6. Stream responses────────────────────────────────┤
   │                            │                         │
   │<─7. Final marker───────────────────────────────────┤
   │                            │                         │
```

## Testing

### With the Example Client

```bash
# Terminal 1 - Server
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- server

# Terminal 2 - Client
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- client
```

### With Your Own Code

See `lib/runtime/examples/http_request_plane_demo.rs` for a complete working example.

## Monitoring

### HTTP Metrics

Standard HTTP server metrics are available:
- Request count
- Request duration
- Inflight requests
- Status codes (202 for success, 4xx/5xx for errors)
- Request/response bytes

### Distributed Tracing

The endpoint extracts and propagates W3C TraceContext headers:
- `traceparent`
- `tracestate`
- `x-request-id`
- `x-dynamo-request-id`

These are attached to the processing span for end-to-end visibility.

## Security Considerations

### Current State
- No authentication (suitable for internal networks only)
- No encryption (use behind VPN or service mesh)

### Future Enhancements
- mTLS for mutual authentication
- JWT/API key authentication
- Rate limiting per client
- Network policies

## Performance

### Latency
- HTTP POST ack: ~1-5ms
- TCP handshake: ~1-5ms
- First response: depends on processing time
- Subsequent responses: TCP streaming (minimal overhead)

### Throughput
- HTTP/2 multiplexing allows many concurrent requests per connection
- Connection pool: 50 connections per host
- No theoretical limit on request rate (limited by worker capacity)

## FAQ

**Q: Why do responses go over TCP instead of HTTP?**
A: TCP streaming provides better performance for long-running requests and large response streams. HTTP SSE/WebSocket support may be added in the future.

**Q: Can I use this with standard HTTP load balancers?**
A: Yes! nginx, envoy, and cloud load balancers work great for the request plane (HTTP). The response plane (TCP) needs direct routing to workers.

**Q: Is this compatible with NATS mode?**
A: Yes, workers and clients can switch between HTTP and NATS modes via `DYN_REQUEST_PLANE` with no code changes.

**Q: What about JetStream/persistence?**
A: HTTP mode doesn't provide built-in queuing. For persistence, use NATS mode or implement retries at the application level.

