# HTTP Request Plane - Usage Guide

## Understanding the Error

If you see this error:
```
WARN Failed to handle request: Two Part Codec Error: Invalid message: No message decoded
```

This happens because you're sending raw data to the endpoint, but it expects **TwoPartCodec-encoded** messages (control header + payload). This is the same format used by NATS, which allows HTTP and NATS modes to be interchangeable.

**Solution:** Use the provided client or PushRouter, which handles encoding automatically. Direct curl testing is not supported.

For protocol details, see [HTTP_ENDPOINT_PROTOCOL.md](./HTTP_ENDPOINT_PROTOCOL.md).

## Quick Start

### 1. Start etcd (Required)

```bash
etcd
```

### 2. Run the Demo

#### Option A: Automated Test (Recommended)
```bash
./test_http_request_plane.sh
```

#### Option B: Manual Setup

**Terminal 1 - Start the Server:**
```bash
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- server
```

**Terminal 2 - Run the Client:**
```bash
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- client
```

### 3. Understanding the Protocol

The HTTP endpoint accepts POST requests with TwoPartCodec-encoded payloads:
- **Control Header**: JSON with request ID, type, and TCP connection info
- **Request Data**: JSON-serialized request payload

**Note:** Direct curl testing is not straightforward because the payload must be encoded with TwoPartCodec. Use the provided client instead, which handles encoding automatically.

## Expected Output

### Server Output

```
INFO http_request_plane_demo: Request plane mode: http
INFO http_request_plane_demo: ✓ Running in HTTP/2 mode
INFO http_request_plane_demo: HTTP RPC endpoint: http://0.0.0.0:8081/v1/dynamo

📋 To test the HTTP endpoint:
   Run the client: cargo run --example http_request_plane_demo -- client

   Note: Direct curl testing requires encoding the request with TwoPartCodec,
   which includes control headers + request payload. Use the client instead.

INFO dynamo_runtime::pipeline::network::ingress::http_endpoint: Starting HTTP endpoint server on 0.0.0.0:8081 at path /v1/dynamo/:endpoint
```

### Client Output

```
INFO http_request_plane_demo: Client starting with request plane mode: http
INFO http_request_plane_demo: Waiting for worker instances...
INFO http_request_plane_demo: ✓ Worker instances available

🚀 Sending requests...

INFO http_request_plane_demo: → Sending: "Hello from request #1"
INFO http_request_plane_demo: ← Received: message='Hello from request #1' from worker='worker-1'
INFO http_request_plane_demo: → Sending: "Hello from request #2"
INFO http_request_plane_demo: ← Received: message='Hello from request #2' from worker='worker-1'
INFO http_request_plane_demo: → Sending: "Hello from request #3"
INFO http_request_plane_demo: ← Received: message='Hello from request #3' from worker='worker-1'
INFO http_request_plane_demo: → Sending: "Hello from request #4"
INFO http_request_plane_demo: ← Received: message='Hello from request #4' from worker='worker-1'
INFO http_request_plane_demo: → Sending: "Hello from request #5"
INFO http_request_plane_demo: ← Received: message='Hello from request #5' from worker='worker-1'

✅ All requests completed successfully!
```

## Configuration

Set these environment variables to customize the HTTP server:

```bash
# Request plane mode (default: nats)
export DYN_REQUEST_PLANE=http

# HTTP server bind address (default: 0.0.0.0)
export DYN_HTTP_RPC_HOST=0.0.0.0

# HTTP server port (default: 8081)
export DYN_HTTP_RPC_PORT=8081

# HTTP RPC root path (default: /v1/dynamo)
export DYN_HTTP_RPC_ROOT_PATH=/v1/dynamo

# HTTP request timeout in seconds (default: 5)
export DYN_HTTP_REQUEST_TIMEOUT=10
```

## Architecture

```
┌─────────┐                    ┌─────────┐
│ Client  │ ─── HTTP/2 ───────>│ Worker  │
│ (Router)│    POST Request     │(Handler)│
└─────────┘                    └─────────┘
     │                              │
     │<──────── TCP Stream ─────────┘
     │         Responses
     └───────────────────────────────┘
```

**Request Flow:**
1. Client discovers worker via etcd
2. Client registers TCP response stream
3. Client sends HTTP/2 POST with encoded request
4. Worker receives 202 Accepted immediately
5. Worker processes request asynchronously
6. Worker streams responses back over TCP
7. Client receives streaming responses

## Troubleshooting

### "Two Part Codec Error: Invalid message"
- This error occurs when sending raw data directly to the HTTP endpoint
- The endpoint expects TwoPartCodec-encoded messages (control header + payload)
- **Solution:** Use the provided client or PushRouter instead of raw curl

### "No instances found"
- Make sure the server is running and has registered in etcd
- Check etcd is accessible: `etcdctl get --prefix v1/instances/`
- Wait a few seconds after server startup for registration

### "Connection refused"
- Verify the HTTP port is not in use: `lsof -i :8081`
- Check firewall settings
- Ensure DYN_HTTP_RPC_PORT matches on client and server

### "Route not found"
- Verify the endpoint path format matches
- Check server logs for registered routes
- Ensure DYN_HTTP_RPC_ROOT_PATH is consistent

## Comparison: HTTP vs NATS

| Feature | HTTP Mode | NATS Mode |
|---------|-----------|-----------|
| Dependencies | etcd only | etcd + NATS + JetStream |
| Request Latency | Low | Low |
| Load Balancing | HTTP LB (nginx, envoy) | NATS built-in |
| Observability | Standard HTTP metrics | NATS stats |
| Debugging | curl, Postman, browsers | NATS CLI |
| Production Ready | ✅ Yes | ✅ Yes (default) |

## Next Steps

- Integrate with your existing services
- Add authentication (mTLS, API keys)
- Configure HTTP load balancers (nginx, envoy)
- Monitor with Prometheus HTTP metrics
- Scale workers horizontally

For more information, see [HTTP_REQUEST_PLANE_IMPLEMENTATION.md](./HTTP_REQUEST_PLANE_IMPLEMENTATION.md)

