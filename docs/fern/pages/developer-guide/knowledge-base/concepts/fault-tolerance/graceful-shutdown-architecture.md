---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Graceful Shutdown Architecture
---

NVIDIA Dynamo gives admitted requests time to finish before releasing engine resources. Shutdown deadlines bound this wait: expiry can interrupt requests and force termination.

This is an architecture reference. For how to tune graceful shutdown for a deployment — grace periods, drain windows, and enabling migration — see the [Graceful Shutdown](../../../../kubernetes/fault-tolerance/graceful-shutdown.md) use-case guide.

## Overview

Backend workers follow this order within one total deadline:

1. Withdraw endpoints from discovery.
2. Keep serving during router propagation grace.
3. Close admission and drain tracked requests.
4. On prefill workers, wait for KV transfers or apply the declared fallback.
5. Clean up the engine and await runtime teardown.

## Signal Handling

All Dynamo components handle Unix signals for graceful shutdown:

| Signal | Trigger | Behavior |
|--------|---------|----------|
| `SIGTERM` | Kubernetes pod termination | Graceful shutdown initiated |
| `SIGINT` | Ctrl+C / manual interrupt | Graceful shutdown initiated |

### Implementation

Rust backends use the `dynamo-backend-common` worker lifecycle. Python vLLM, SGLang, and TensorRT-LLM use `WorkerShutdown`, which owns admission tracking and joins the engine-owning task during cleanup. Both follow the sequence above. SGLang defers engine-installed signal callbacks until cleanup and runtime teardown complete.

`DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS` sets the SIGTERM-origin total. Pre-cleanup stages withhold the cleanup reserve; engine cleanup and runtime teardown share that reserve inside the total. An OS-thread watchdog bounds stalled teardown, and a second SIGTERM or SIGINT forces termination with exit code 70.

> [!WARNING]
> A successful request drain establishes that tracked requests have finished. A timed-out drain does not: engine cleanup must handle unfinished execution. Likewise, waiting a fixed KV-transfer fallback allowance is not proof that remote reads have completed.

`runtime.shutdown()` starts teardown without waiting. `runtime.shutdown_and_wait()` joins the runtime-owned teardown task, including bounded lease revocation. Its optional timeout bounds endpoint draining only; worker coordinators additionally bound the entire wait by their remaining shutdown allowance. Outside worker coordination, the default endpoint-drain timeout is `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` (900 seconds).

## Endpoint Draining

Worker admission closes after router grace, before engine cleanup. Runtime endpoint invalidation happens later during transport teardown. For endpoints without a worker coordinator, `graceful_shutdown` controls whether runtime invalidation waits for in-flight work.

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
| **Frontend** | HTTP request draining | Stop admission while response bodies and upgraded WebSocket tasks complete |
| **Prefill Workers** | `graceful_shutdown=True` | Prefill operations must complete to avoid wasted computation |
| **Decode Workers** | `graceful_shutdown=True` | Decode operations should complete to avoid wasted computation |
| **Router** | `graceful_shutdown=True` | Ensure routing decisions complete |

### Frontend HTTP Draining

The Frontend uses a separate HTTP drain state before it cancels the distributed runtime:

1. Mark the server as draining.
2. Return `503 Service Unavailable` from `/health` and reject new OpenAI-compatible requests with 503.
3. Keep `/live` at `200 OK` so liveness probes do not restart the process while requests drain.
4. Wait for admitted response bodies, including streaming responses, to complete.
5. Continue tracking accepted `/v1/realtime` WebSocket sessions until their tasks exit, even though the HTTP upgrade response has completed.
6. After `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` expires (default `5` seconds), enter the stopping state and cancel runtime state.

This separates readiness from liveness: traffic is removed promptly without turning an intentional
drain into a restart loop.

### Migration Integration

Worker admission rejects stale-routed requests with a typed unavailable response before entering the engine. Draining errors use the established worker-unavailable wire identity so older frontends can apply their existing migration policy. Request migration is configured at the frontend via `--migration-limit`:

- When migration is enabled at the frontend, requests interrupted by worker failure or graceful shutdown after grace expires are retried on healthy workers, subject to the retry budget and request limits
- Workers don't need to know about migration configuration - they simply complete their work or signal incomplete streams
- See [Request Migration Architecture](request-migration-architecture.md) for details on how migration works

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
3. The pod's `terminationGracePeriodSeconds` bounds termination, including any `preStop` hook; explicit worker budgets must fit within this grace period with the required margin.
4. If not terminated, Kubernetes sends `SIGKILL`

### Health Check Integration

Kubernetes uses health endpoints to determine pod readiness:

- **During shutdown**: Endpoints become unavailable
- **Readiness probe fails**: Traffic stops routing to the pod
- **Graceful draining**: Existing requests have time to complete

## Related Documentation

- [Graceful Shutdown](../../../../kubernetes/fault-tolerance/graceful-shutdown.md) - How to tune grace periods and drain windows (use-case guide)
- [Request Migration Architecture](request-migration-architecture.md) - How requests migrate during shutdown
- [Request Cancellation Architecture](request-cancellation-architecture.md) - Canceling in-flight requests
- [Health Check Reference](../../../../reference/observability/health-checks.mdx) - Liveness and readiness endpoints
