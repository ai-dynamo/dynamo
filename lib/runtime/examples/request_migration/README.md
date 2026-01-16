# Request Migration Example

This example demonstrates Dynamo's fault tolerance and automatic request retry capabilities when streaming responses encounter errors, with immediate failover to healthy instances.

## What It Demonstrates

1. **Server Error Injection**: Configurable server failure modes (never, first request, always)
2. **Client-Side Retry Logic**: Automatic error detection and immediate retry
3. **Instant Failover**: Client routes to healthy instances without waiting
4. **Server Identification**: Each response includes the server name to show which instance handled the request
5. **Instance Management**: Two-tier architecture for tracking available instances

## Architecture

### Server
- **Configurable Failure Modes** via `FAILURE_MODE` environment variable:
  - `never` - Normal operation, no failures
  - `first` - Fails only on the first request (default)
  - `always` - Fails every request (demonstrates continuous failover)
- Simulates stream errors after sending 2 characters
- Includes server name in all responses for identification

### Client
- Detects stream errors using the `MaybeError` trait
- Implements retry logic with up to 3 attempts
- **Retries immediately** without waiting between attempts
- Relies on multiple servers for instant failover

### Instance Management

The system uses a two-tier architecture:
- **`instance_source`**: Source of truth from discovery backend (etcd/k8s)
- **`instance_avail`**: Filtered list that temporarily removes failing instances

When an error occurs:
1. `PushRouter` calls `report_instance_down()` → instance removed from `instance_avail`
2. Client immediately retries → routes to a different healthy instance
3. After 5 seconds, reconcile task restores the failed instance from `instance_source`

## Prerequisites

This example requires **at least 2 servers** running simultaneously to properly demonstrate failover:
- When a server fails, the instance is marked down
- The client immediately retries to a different healthy instance
- With only 1 server, retry attempts will fail with "no instances found"

## Configuration

### Environment Variables

**Server:**
- **`SERVER_NAME`**: Identifies the server in responses (default: `"server"`)
- **`FAILURE_MODE`**: Controls when the server fails (default: `"first"`)
  - `never` - Never inject errors (normal operation)
  - `first` - Fail only on the first request
  - `always` - Fail on every request (useful for demonstrating continuous failover)
- **`DYN_STORE_KV`**: Discovery backend (`file`, `etcd`, `mem`)

**Client:**
- **`DYN_STORE_KV`**: Must match server's discovery backend

## Running the Example

### Basic Failover Demo (Recommended)

Shows how the client automatically fails over to a healthy instance when one server has a temporary failure.

```bash
# Terminal 1 - Server 1 (fails on first request only)
cd lib/runtime/examples/request_migration
SERVER_NAME=server1 FAILURE_MODE=always DYN_STORE_KV=file cargo run --bin server

# Terminal 2 - Server 2 (never fails)
cd lib/runtime/examples/request_migration
SERVER_NAME=server2 FAILURE_MODE=never DYN_STORE_KV=file cargo run --bin server

# Terminal 3 - Client
cd lib/runtime/examples/request_migration
DYN_STORE_KV=file cargo run --bin client
```

**Expected Output:**
```
=== Attempt 1 ===
Annotated { data: Some("from server: server1, data: h"), id: None, event: None, comment: None }
Annotated { data: Some("from server: server1, data: e"), id: None, event: None, comment: None }
Error: Stream ended before generation completed
Retrying request...

=== Attempt 2 ===
Annotated { data: Some("from server: server2, data: h"), id: None, event: None, comment: None }
Annotated { data: Some("from server: server2, data: e"), id: None, event: None, comment: None }
Annotated { data: Some("from server: server2, data: l"), id: None, event: None, comment: None }
...rest of characters from server2...

=== Success! ===
```

**What happened:** Server1 failed on attempt 1 → PushRouter marked it as down → Attempt 2 immediately routed to server2 ✅

### Continuous Failover Demo (One Server Always Fails)

Demonstrates behavior when one instance is permanently unhealthy and always fails.

```bash
# Terminal 1 - Server 1 (always fails every request)
cd lib/runtime/examples/request_migration
SERVER_NAME=server1 FAILURE_MODE=always DYN_STORE_KV=file cargo run --bin server

# Terminal 2 - Server 2 (never fails)
cd lib/runtime/examples/request_migration
SERVER_NAME=server2 FAILURE_MODE=never DYN_STORE_KV=file cargo run --bin server

# Terminal 3 - Client
cd lib/runtime/examples/request_migration
DYN_STORE_KV=file cargo run --bin client
```

**Expected Output:**
```
=== Attempt 1 ===
Annotated { data: Some("from server: server1, data: h"), id: None, event: None, comment: None }
Annotated { data: Some("from server: server1, data: e"), id: None, event: None, comment: None }
Error: Stream ended before generation completed
Retrying request...

=== Attempt 2 ===
Annotated { data: Some("from server: server2, data: h"), id: None, event: None, comment: None }
Annotated { data: Some("from server: server2, data: e"), id: None, event: None, comment: None }
...rest of characters from server2...

=== Success! ===
```

**What happened:** With `FAILURE_MODE=always`, server1 fails every request. The system continuously routes around it to server2. After 5 seconds (reconcile interval), server1 is restored to the available pool, but will be marked down again on the next failed request.

## Key Concepts

### STREAM_ERR_MSG
Special error message (`"Stream ended before generation completed"`) that signals stream disconnection. When `PushRouter` detects this error in a response, it automatically calls `report_instance_down()` to mark the instance as unavailable.

### Immediate Failover
The client retries immediately after detecting an error, routing the request to a different healthy instance. This provides instant failover without waiting for reconciliation.

### Reconcile Interval
A background task runs every 5 seconds to restore instances that were temporarily marked down. This ensures:
- Transient failures don't permanently blacklist servers
- Instances are given a "cooldown" period before being retried
- The system automatically recovers from temporary network issues

### Two-Tier Instance Management
- **`instance_source`**: Authoritative list from discovery backend (etcd/k8s), updated in real-time
- **`instance_avail`**: Filtered subset excluding recently failed instances
- Reconcile syncs `instance_avail` back to `instance_source` periodically

### Discovery Backend
The authoritative source (etcd/k8s) removes truly dead servers:
- **etcd**: Server's lease expires (10s TTL) when process dies
- **k8s**: Pod deletion removes the instance from discovery
The reconcile mechanism only restores instances that are still alive in the discovery backend.

## Debug Logging

Enable detailed logs to see the internal behavior:

**Server logs** (see failure injection and discovery registration):
```bash
RUST_LOG=request_migration=info,dynamo_runtime::discovery=debug,dynamo_runtime::component=debug \
  SERVER_NAME=server1 FAILURE_MODE=first DYN_STORE_KV=file cargo run --bin server
```

**Client logs** (see instance management and routing decisions):
```bash
RUST_LOG=dynamo_runtime::component::client=debug,dynamo_runtime::pipeline::network::egress::push_router=debug \
  DYN_STORE_KV=file cargo run --bin client
```

You'll see logs like:
- `Server initialized: name=server1, failure_mode=first request only`
- `Simulating stream error after 2 characters (server: server1)`
- `Reporting instance {id} down due to stream error`
- `inhibiting instance {id}`
- `periodic reconciliation for endpoint=...`
