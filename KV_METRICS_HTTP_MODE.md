# KV Metrics HTTP Mode

## Overview

When using `DYN_REQUEST_PLANE=http`, the NATS-based KV metrics publishing is automatically disabled to eliminate unnecessary NATS traffic. This document explains what was changed and how the system behaves in HTTP mode.

## Changes Made

### 1. Service-Level NATS Stats Collection Disabled

**File:** `/home/ubuntu/dynamo/lib/runtime/src/component/service.rs`

When `DYN_REQUEST_PLANE=http` is set, the automatic NATS service stats collection (which generates `$SRV.STATS.*` requests) is now disabled. This eliminates the periodic NATS traffic:

```
[SUB _INBOX.* ...]
[PUB $SRV.STATS.dynamo_backend ...]
```

### 2. KV Metrics Publishing Disabled

**File:** `/home/ubuntu/dynamo/lib/llm/src/kv_router/publisher.rs`

When `DYN_REQUEST_PLANE=http` is set, the background task that publishes KV metrics to NATS is disabled. This eliminates the NATS traffic:

```
[PUB namespace.dynamo.kv_metrics ...]
```

### 3. KV Events Publishing Disabled

**File:** `/home/ubuntu/dynamo/lib/llm/src/kv_router/publisher.rs`

When `DYN_REQUEST_PLANE=http` is set, the KV event publishing to NATS is also disabled. This eliminates the NATS traffic:

```
[PUB namespace-dynamo-component-backend-kv-events.queue ...]
```

Instead of publishing to NATS, KV events are consumed by a dummy processor that discards them.

### 4. KV Router Background Task Disabled

**File:** `/home/ubuntu/dynamo/lib/llm/src/kv_router/subscriber.rs`

When `DYN_REQUEST_PLANE=http` is set, the KV router background task that establishes NATS connections for event consumption and state management is completely disabled. This eliminates all remaining NATS traffic including:

```
[NATS connection establishment]
[NATS queue creation and consumption]
[NATS object store operations]
[PING/PONG keep-alive messages]
```

Instead, a minimal HTTP-mode background task is started that only handles etcd-based operations like worker instance monitoring.

## Behavior by Mode

### NATS Mode (default: `DYN_REQUEST_PLANE=nats` or unset)

- ✅ NATS service stats collection enabled
- ✅ KV metrics published to NATS at `namespace.dynamo.kv_metrics`
- ✅ KV events published to NATS at `namespace-dynamo-component-backend-kv-events.queue`
- ✅ Service discovery via NATS microservices API
- ✅ Request routing via NATS subjects

**NATS Traffic:**
```
[TRC] - [SUB _INBOX.shcjpujnveuspZM7cDhSwx 33]
[TRC] - [PUB $SRV.STATS.dynamo_backend _INBOX.shcjpujnveuspZM7cDhSwx 0]
[TRC] - [PUB namespace.dynamo.kv_metrics 310]
[TRC] - [PUB namespace-dynamo-component-backend-kv-events.queue 191]
[TRC] - [PING]
[TRC] - [PONG]
```

### HTTP Mode (`DYN_REQUEST_PLANE=http`)

- 🚫 NATS service stats collection disabled
- 🚫 KV metrics publishing to NATS disabled
- 🚫 KV events publishing to NATS disabled
- ✅ Service discovery still via etcd
- ✅ Request routing via HTTP/2

**NATS Traffic:**
```
# No NATS traffic at all - completely eliminated
```

## What Still Works in HTTP Mode

### KV Metrics Endpoint

The KV metrics endpoint remains available for pull-based metrics collection:

```rust
component
    .endpoint(KV_METRICS_ENDPOINT)
    .endpoint_builder()
    .stats_handler(move |_| {
        let metrics = metrics_rx.borrow_and_update().clone();
        serde_json::to_value(&*metrics).unwrap()
    })
    .handler(handler)
    .start()
    .await
```

This endpoint can be queried directly for current KV metrics without NATS.

### Prometheus Metrics

All Prometheus metrics continue to work in HTTP mode. The `WorkerMetricsPublisher` still updates Prometheus gauges:

```rust
pub fn publish(&self, metrics: Arc<ForwardPassMetrics>) -> Result<...> {
    // Update Prometheus gauges (works in both modes)
    if let Some(gauges) = self.prometheus_gauges.get() {
        gauges.update_from_kvstats(&metrics.kv_stats);
    }

    self.tx.send(metrics)
}
```

## Migration Considerations

### If You're Using KV Metrics Subscriber

The KV metrics subscriber in HTTP mode will no longer receive push-based updates from NATS. You have these options:

1. **Poll the KV metrics endpoint**: Use the existing `KV_METRICS_ENDPOINT` to periodically query for metrics
2. **Use Prometheus**: Scrape Prometheus metrics instead of subscribing to NATS events
3. **Keep NATS mode**: If you need push-based KV metrics, use `DYN_REQUEST_PLANE=nats`

### Example: Polling KV Metrics Endpoint

```rust
// In HTTP mode, poll the KV metrics endpoint instead of subscribing to NATS
let client = /* your HTTP client or router */;
let mut interval = tokio::time::interval(Duration::from_millis(100));

loop {
    interval.tick().await;

    // Make request to KV_METRICS_ENDPOINT
    let response = router.call(/* KV metrics request */).await?;

    // Process metrics
    let metrics: ForwardPassMetrics = /* deserialize response */;
    handle_metrics(metrics);
}
```

## Benefits of HTTP Mode

When using HTTP mode for request routing, disabling NATS-based metrics publishing provides:

1. **Reduced NATS Load**: No periodic stats requests or KV metrics publications
2. **Simpler Architecture**: Fewer moving parts when NATS is not the primary transport
3. **Cleaner Logs**: No NATS trace logs cluttering your monitoring
4. **Resource Efficiency**: Fewer network operations and background tasks

## Debugging

### Check Current Mode

```bash
echo $DYN_REQUEST_PLANE
# Output: http (or nats, or empty for default)
```

### Verify NATS Metrics Disabled

Look for these log messages when starting your service:

```
[DEBUG] Skipping NATS service metrics collection for 'dynamo_backend' - request plane mode is 'http'
[DEBUG] Skipping NATS metrics publishing for KV metrics - request plane mode is 'http' (worker_id: 123)
```

### NATS Server Logs

In HTTP mode, you should see no NATS traffic at all:

```
# No NATS traffic - completely eliminated
```

No `PUB $SRV.STATS.*` or `PUB namespace.dynamo.kv_metrics` messages should appear.

## Implementation Details

### Code Changes

1. **Service Stats**: `lib/runtime/src/component/service.rs:91-108`
   ```rust
   let request_plane_mode = RequestPlaneMode::from_env();
   if request_plane_mode.is_nats() {
       component.start_scraping_nats_service_component_metrics()?;
   } else {
       tracing::debug!("Skipping NATS service metrics...");
   }
   ```

2. **KV Metrics**: `lib/llm/src/kv_router/publisher.rs:805-816`
   ```rust
   let request_plane_mode = RequestPlaneMode::from_env();
   if request_plane_mode.is_nats() {
       self.start_nats_metrics_publishing(namespace, worker_id);
   } else {
       tracing::debug!("Skipping NATS metrics publishing...");
   }
   ```

### No Code Changes Required

These changes are environment-based. Simply set:

```bash
export DYN_REQUEST_PLANE=http
```

All NATS metrics publishing will be automatically disabled.

## Summary

Setting `DYN_REQUEST_PLANE=http` now:

- ✅ Disables NATS service stats collection
- ✅ Disables NATS KV metrics publishing
- ✅ Eliminates unnecessary NATS traffic
- ✅ Maintains Prometheus metrics
- ✅ Keeps KV metrics endpoint available for polling
- ✅ No application code changes needed

This provides a cleaner, more efficient HTTP-based deployment without NATS overhead.

