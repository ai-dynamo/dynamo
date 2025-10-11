# How to Use HTTP Mode Instead of NATS

## TL;DR - Quick Start

**For Frontend and Backend workers:**

```bash
export DYN_REQUEST_PLANE=http
# Restart your frontend and backend services
```

That's it! No code changes needed.

---

## What Changes?

When you set `DYN_REQUEST_PLANE=http`:

### Backend (Request Receiver)
- **Before (NATS)**: Uses `PushEndpoint` to receive requests from NATS queue
- **After (HTTP)**: Uses `HttpEndpoint` to receive HTTP/2 POST requests on port 8081

### Frontend (Request Sender)
- **Before (NATS)**: `PushRouter` sends requests via NATS client
- **After (HTTP)**: `PushRouter` sends requests via HTTP/2 client (reqwest)

### What Stays the Same?
- **Response streaming**: Still uses TCP with call-home pattern
- **Service discovery**: Still uses etcd
- **Routing logic**: Round-robin, random, KV routing - all unchanged
- **Tracing**: Still propagates W3C TraceContext headers
- **Your application code**: Zero changes!

---

## Step-by-Step Migration

### 1. For Local Development/Testing

```bash
# Terminal 1 - Start etcd
etcd

# Terminal 2 - Start backend with HTTP mode
export DYN_REQUEST_PLANE=http
export DYN_HTTP_RPC_HOST=0.0.0.0
export DYN_HTTP_RPC_PORT=8081
# Start your backend (e.g., python -m dynamo.backend.trtllm)
cargo run --bin your-backend

# Terminal 3 - Start frontend with HTTP mode
export DYN_REQUEST_PLANE=http
# Start your frontend (e.g., python -m dynamo.frontend)
cargo run --bin your-frontend
```

### 2. For Docker/Container Deployments

Add to your Dockerfile or docker-compose.yml:

```yaml
services:
  backend:
    environment:
      - DYN_REQUEST_PLANE=http
      - DYN_HTTP_RPC_HOST=0.0.0.0
      - DYN_HTTP_RPC_PORT=8081
      - DYN_HTTP_RPC_ROOT_PATH=/v1/dynamo
    ports:
      - "8081:8081"  # HTTP RPC port
      - "9000:9000"  # TCP streaming port (if needed)

  frontend:
    environment:
      - DYN_REQUEST_PLANE=http
      - DYN_HTTP_RPC_HOST=backend  # Container name for DNS
      - DYN_HTTP_RPC_PORT=8081
```

### 3. For Kubernetes

Update your deployment YAML:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: dynamo-http-config
data:
  DYN_REQUEST_PLANE: "http"
  DYN_HTTP_RPC_HOST: "0.0.0.0"
  DYN_HTTP_RPC_PORT: "8081"
  DYN_HTTP_RPC_ROOT_PATH: "/v1/dynamo"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dynamo-backend
spec:
  template:
    spec:
      containers:
      - name: backend
        envFrom:
        - configMapRef:
            name: dynamo-http-config
        ports:
        - containerPort: 8081
          name: http-rpc
          protocol: TCP
        - containerPort: 9000  # TCP streaming
          name: tcp-stream
          protocol: TCP
```

Add a Service for HTTP load balancing:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: dynamo-backend-http
spec:
  selector:
    app: dynamo-backend
  ports:
  - port: 8081
    targetPort: 8081
    protocol: TCP
    name: http-rpc
  type: ClusterIP  # Or LoadBalancer for external access
```

### 4. For Python Components

If you're using Python frontends/backends, set environment variables before starting:

```python
import os

os.environ["DYN_REQUEST_PLANE"] = "http"
os.environ["DYN_HTTP_RPC_HOST"] = "0.0.0.0"
os.environ["DYN_HTTP_RPC_PORT"] = "8081"

# Then start your dynamo service
from dynamo.frontend import run
run()
```

Or via command line:

```bash
DYN_REQUEST_PLANE=http python -m dynamo.frontend --http-port=8000
DYN_REQUEST_PLANE=http python -m dynamo.backend.trtllm
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_REQUEST_PLANE` | `nats` | Request plane mode: `nats` or `http` |
| `DYN_HTTP_RPC_HOST` | `0.0.0.0` | HTTP server bind address |
| `DYN_HTTP_RPC_PORT` | `8081` | HTTP server port |
| `DYN_HTTP_RPC_ROOT_PATH` | `/v1/dynamo` | API path prefix |
| `DYN_HTTP_REQUEST_TIMEOUT` | `5` | HTTP request timeout (seconds) |

### Port Requirements

**HTTP Mode:**
- Port 8081: HTTP/2 RPC endpoint (configurable)
- Port 9000+: TCP response streaming (auto-assigned)
- Port 2379: etcd client port

**NATS Mode (for comparison):**
- Port 4222: NATS client port
- Port 9000+: TCP response streaming (auto-assigned)
- Port 2379: etcd client port

---

## Verification

### Check Backend is Listening

```bash
# Check HTTP endpoint
curl -X POST http://localhost:8081/v1/dynamo/health
# Or
lsof -i :8081
```

### Check Service Registration in etcd

```bash
# List all instances
etcdctl get --prefix v1/instances/

# Check specific backend
etcdctl get v1/instances/namespace.default.component.backend.endpoint.generate.instance.0
```

You should see HTTP endpoint info in the transport field:

```json
{
  "transport": {
    "http_tcp": {
      "http_endpoint": "http://0.0.0.0:8081/v1/dynamo/namespace.default.component.backend.endpoint.generate.instance.0",
      "tcp_endpoint": "0.0.0.0:9000"
    }
  }
}
```

### Monitor HTTP Traffic

```bash
# Watch HTTP logs
# Backends will log: "Starting HTTP endpoint server on 0.0.0.0:8081"

# Check Prometheus metrics (if enabled)
curl http://localhost:8000/metrics | grep http_requests
```

---

## Load Balancing

### With HTTP Load Balancer (nginx example)

```nginx
upstream dynamo_backends {
    # Round-robin by default
    server backend-1:8081;
    server backend-2:8081;
    server backend-3:8081;
}

server {
    listen 80;

    location /v1/dynamo/ {
        proxy_pass http://dynamo_backends;
        proxy_http_version 1.1;

        # Forward tracing headers
        proxy_set_header traceparent $http_traceparent;
        proxy_set_header tracestate $http_tracestate;
        proxy_set_header x-request-id $http_x_request_id;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

### With Envoy

```yaml
static_resources:
  listeners:
  - address:
      socket_address:
        address: 0.0.0.0
        port_value: 80
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: backend
              domains: ["*"]
              routes:
              - match:
                  prefix: "/v1/dynamo/"
                route:
                  cluster: dynamo_backends
          http_filters:
          - name: envoy.filters.http.router

  clusters:
  - name: dynamo_backends
    connect_timeout: 5s
    type: STRICT_DNS
    lb_policy: ROUND_ROBIN
    load_assignment:
      cluster_name: dynamo_backends
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: backend
                port_value: 8081
```

---

## Troubleshooting

### Error: "builder error"

**Symptom**: Client fails with "builder error"

**Cause**: Missing HTTP client initialization in `AddressedPushRouter`

**Solution**: Make sure you're on the latest code. The error was fixed by properly initializing the HTTP client in `addressed_router()` function.

### Error: "Connection refused"

**Symptom**: Frontend can't connect to backend

**Check:**
1. Is backend running? `lsof -i :8081`
2. Is backend in HTTP mode? Check logs for "Starting HTTP endpoint server"
3. Firewall blocking port 8081?
4. Is `DYN_HTTP_RPC_HOST` set correctly? (use `0.0.0.0` for all interfaces)

### Error: "No instances found"

**Symptom**: Frontend can't discover backend instances

**Check:**
1. Is etcd running? `etcdctl endpoint health`
2. Is backend registered? `etcdctl get --prefix v1/instances/`
3. Wait a few seconds after backend startup
4. Check both frontend and backend have same `DYN_REQUEST_PLANE` setting

### Error: "Two Part Codec Error: Invalid message"

**Symptom**: HTTP endpoint returns this error

**Cause**: You're sending raw data instead of TwoPartCodec-encoded payload

**Solution**: Don't use raw curl. Use the `PushRouter` from your code, which handles encoding automatically.

### Performance Issues

**Symptom**: Higher latency compared to NATS

**Check:**
1. HTTP/2 enabled? (Should be automatic with reqwest)
2. Connection pooling working? Check `pool_max_idle_per_host` setting
3. Consider putting HTTP load balancer in front
4. Adjust `DYN_HTTP_REQUEST_TIMEOUT` if needed

---

## Rollback to NATS

If you need to switch back:

```bash
# Option 1: Unset the variable (defaults to NATS)
unset DYN_REQUEST_PLANE

# Option 2: Explicitly set to NATS
export DYN_REQUEST_PLANE=nats

# Restart your services
```

No code changes needed!

---

## Production Checklist

Before deploying HTTP mode in production:

- [ ] Set `DYN_REQUEST_PLANE=http` on all workers (frontend and backend)
- [ ] Configure HTTP port (`DYN_HTTP_RPC_PORT`) to avoid conflicts
- [ ] Open firewall rules for HTTP port (default 8081)
- [ ] Set up HTTP load balancer (nginx, envoy, cloud LB)
- [ ] Configure health checks on HTTP endpoint
- [ ] Set up monitoring/metrics for HTTP requests
- [ ] Test failover and scaling
- [ ] Configure distributed tracing (if using)
- [ ] Document the HTTP endpoints for your team
- [ ] Set appropriate `DYN_HTTP_REQUEST_TIMEOUT` for your workload
- [ ] Consider TLS/mTLS for security (if needed)

---

## When to Use HTTP vs NATS

### Use HTTP Mode When:
- ✅ You want simpler deployment (no NATS dependency)
- ✅ You need standard HTTP observability tools
- ✅ You want to use HTTP load balancers
- ✅ You prefer familiar HTTP debugging (curl, Postman)
- ✅ You're deploying in environments that limit services

### Use NATS Mode When:
- ✅ You need built-in message persistence (JetStream)
- ✅ You want pub/sub beyond request/response
- ✅ You need advanced routing (subject-based, wildcard)
- ✅ You're already running NATS infrastructure
- ✅ You need NATS-specific features (streams, KV store)

### Can Use Either:
- Request/response patterns
- Round-robin / random routing
- Distributed tracing
- Service discovery via etcd
- Response streaming (always TCP)
- Horizontal scaling

---

## Examples

### Example 1: Simple Test

```bash
# Start the HTTP demo
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- server  # Terminal 1
DYN_REQUEST_PLANE=http cargo run --example http_request_plane_demo -- client  # Terminal 2
```

### Example 2: Frontend-Backend Communication

See `lib/llm/src/entrypoint/input/common.rs` for how frontends create routers:

```rust
// This code already supports HTTP mode automatically!
let router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
    client.clone(),
    router_mode,
    busy_threshold,
).await?;
```

When `DYN_REQUEST_PLANE=http`, the router automatically uses HTTP transport.

### Example 3: Python Frontend

```bash
# Python frontend connecting to Rust backend
export DYN_REQUEST_PLANE=http
python -m dynamo.frontend --http-port=8000

# Backend in HTTP mode
DYN_REQUEST_PLANE=http cargo run --bin trtllm-backend
```

---

## Additional Resources

- [HTTP_REQUEST_PLANE_IMPLEMENTATION.md](./HTTP_REQUEST_PLANE_IMPLEMENTATION.md) - Technical implementation details
- [HTTP_REQUEST_PLANE_USAGE.md](./HTTP_REQUEST_PLANE_USAGE.md) - Usage guide with examples
- [HTTP_ENDPOINT_PROTOCOL.md](./HTTP_ENDPOINT_PROTOCOL.md) - Protocol specification
- [HTTP_REQUEST_PLANE_SUMMARY.md](./HTTP_REQUEST_PLANE_SUMMARY.md) - Complete summary

---

## Questions?

If you encounter issues:

1. Check the troubleshooting section above
2. Verify environment variables are set correctly
3. Check logs for "Starting HTTP endpoint server" message
4. Verify etcd registration with `etcdctl get --prefix v1/instances/`
5. Test with the provided example: `http_request_plane_demo.rs`

For more help, see the documentation files listed above.

