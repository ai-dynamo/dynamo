---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Graceful Shutdown Architecture
---

This document describes the internals of how Dynamo components handle shutdown signals to ensure in-flight requests complete successfully and resources are properly cleaned up.

This is an architecture reference. For how to tune graceful shutdown for a deployment — grace periods, drain windows, and enabling migration — see the [Graceful Shutdown](../fault-tolerance/graceful-shutdown.md) use-case guide.

## Overview

Graceful shutdown in Dynamo ensures that:

1. **Routing stops quickly** - Endpoints are unregistered from discovery first
2. **In-flight requests can finish** - Workers keep serving during a short grace period
3. **Endpoints drain** - After the grace period, endpoints are invalidated and optionally wait for in-flight work
4. **Resources are cleaned up** - Engines, connections, and temporary files are released
5. **Pods restart cleanly** - Exit codes signal Kubernetes for proper restart behavior

## Signal Handling

All Dynamo components handle Unix signals for graceful shutdown:

| Signal | Trigger | Behavior |
|--------|---------|----------|
| `SIGTERM` | Kubernetes pod termination | Graceful shutdown initiated |
| `SIGINT` | Ctrl+C / manual interrupt | Graceful shutdown initiated |

### Implementation

Each component registers signal handlers at startup:

```python
def signal_handler():
    asyncio.create_task(graceful_shutdown(runtime, endpoints))

for sig in (signal.SIGTERM, signal.SIGINT):
    loop.add_signal_handler(sig, signal_handler)
```

The `graceful_shutdown()` function:
1. Logs the shutdown signal
2. Unregisters all endpoints from discovery
3. Waits for a configurable grace period (`DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`, default 5s)
4. Calls `runtime.shutdown()` to invalidate endpoints and stop accepting new requests
5. Waits for in-flight requests (based on `graceful_shutdown` per endpoint)
6. Returns to allow cleanup to proceed

The aggregate wait in `runtime.shutdown()` is bounded by
`DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`, which defaults to 900 seconds
(15 minutes). If endpoint draining exceeds this timeout, Dynamo logs the
remaining graceful endpoint count and proceeds with runtime teardown.

## Endpoint Draining

After the grace period, `runtime.shutdown()` invalidates endpoints so no new requests are accepted. The behavior for in-flight requests depends on the `graceful_shutdown` parameter when serving the endpoint.

### Configuration

When registering an endpoint, the `graceful_shutdown` parameter controls draining behavior:

```python
generate_endpoint.serve_endpoint(
    handler.generate,
    graceful_shutdown=True,  # Wait for all requests to finish
    metrics_labels=[("model", model_name)],
    health_check_payload=health_check_payload,
)
```

| `graceful_shutdown` | Behavior |
|---------------------|----------|
| `True` | Wait for all in-flight requests to complete, bounded by `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` |
| `False` | Return immediately without waiting for requests |

### Component-Specific Behavior

| Component | Default Behavior | Rationale |
|-----------|------------------|-----------|
| **Frontend** | N/A (HTTP server) | HTTP server handles its own shutdown |
| **Prefill Workers** | `graceful_shutdown=True` | Prefill operations must complete to avoid wasted computation |
| **Decode Workers** | `graceful_shutdown=True` | Decode operations should complete to avoid wasted computation |
| **Router** | `graceful_shutdown=True` | Ensure routing decisions complete |

### Migration Integration

Backend workers always use `graceful_shutdown=True`, meaning they wait for in-flight requests to complete until the engine is stopped. Request migration is configured at the **frontend** level via `--migration-limit`:

- When migration is enabled at the frontend, disconnected streams from failed workers are automatically retried on healthy workers
- Workers don't need to know about migration configuration - they simply complete their work or signal incomplete streams
- See [Request Migration Architecture](./request-migration.md) for details on how migration works

## Resource Cleanup

After endpoint draining, components clean up their resources in `finally` blocks:

### vLLM Worker Cleanup

```python
finally:
    logger.debug("Cleaning up worker")
    handler.cleanup()
```

The handler's `cleanup()` method:
- Removes temporary directories (LoRA adapters, etc.)
- Releases engine resources

### SGLang Worker Cleanup

```python
def cleanup(self) -> None:
    # Cancel pending consume tasks
    for task in self._consume_tasks:
        if not task.done():
            task.cancel()
    self._consume_tasks.clear()

    # Shutdown engine
    self.engine.shutdown()
```

### TensorRT-LLM Worker Cleanup

```python
async def cleanup(self):
    if self._llm:
        try:
            self._llm.shutdown()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
        finally:
            self._llm = None
```

## Error-Initiated Shutdown

Workers can initiate graceful shutdown when fatal errors occur:

### Engine Health Monitoring (vLLM)

The `VllmEngineMonitor` continuously checks engine health:

```python
async def _check_engine_health(self):
    while True:
        try:
            await self.engine_client.check_health()
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)  # 2 seconds
        except EngineDeadError as e:
            logger.error(f"Health check failed: {e}")
            self._shutdown_engine()
            self.runtime.shutdown()
            os._exit(1)
```

Configuration:
- `HEALTH_CHECK_INTERVAL`: 2 seconds between checks
- `ENGINE_SHUTDOWN_TIMEOUT`: 30 seconds max for engine shutdown

### Fatal Error Handling (TensorRT-LLM)

```python
async def _initiate_shutdown(self, error: Exception):
    logging.warning(f"Initiating graceful shutdown due to: {error}")

    try:
        if self.runtime:
            self.runtime.shutdown()
        if self.engine:
            await self.engine.cleanup()
    except Exception as cleanup_error:
        logging.error(f"Error during graceful shutdown: {cleanup_error}")
    finally:
        logging.critical("Forcing process exit for restart")
        os._exit(1)
```

## Kubernetes Integration

### Pod Termination Flow

1. Kubernetes sends `SIGTERM` to the pod
2. Dynamo initiates graceful shutdown
3. Pod has `terminationGracePeriodSeconds` to complete (default: 30s)
4. If not terminated, Kubernetes sends `SIGKILL`

### Health Check Integration

Kubernetes uses health endpoints to determine pod readiness:

- **During shutdown**: Endpoints become unavailable
- **Readiness probe fails**: Traffic stops routing to the pod
- **Graceful draining**: Existing requests complete

## Related Documentation

- [Graceful Shutdown](../fault-tolerance/graceful-shutdown.md) - How to tune grace periods and drain windows (use-case guide)
- [Request Migration Architecture](request-migration.md) - How requests migrate during shutdown
- [Request Cancellation Architecture](request-cancellation.md) - Canceling in-flight requests
- [Health Checks](../observability/health-checks.md) - Liveness and readiness probes
