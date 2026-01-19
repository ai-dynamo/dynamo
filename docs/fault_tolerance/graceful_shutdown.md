<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Graceful Shutdown

This document describes how Dynamo components handle shutdown signals to ensure in-flight requests complete successfully and resources are properly cleaned up.

## Overview

Graceful shutdown in Dynamo ensures that:

1. **No new requests are accepted** - Endpoints are immediately invalidated
2. **In-flight requests complete** - Existing requests finish processing (configurable)
3. **Resources are cleaned up** - Engines, connections, and temporary files are released
4. **Pods restart cleanly** - Exit codes signal Kubernetes for proper restart behavior

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
    asyncio.create_task(graceful_shutdown(runtime))

for sig in (signal.SIGTERM, signal.SIGINT):
    loop.add_signal_handler(sig, signal_handler)
```

The `graceful_shutdown()` function:
1. Logs the shutdown signal
2. Calls `runtime.shutdown()` to invalidate endpoints
3. Waits for in-flight requests (based on configuration)
4. Returns to allow cleanup to proceed

## Endpoint Draining

When `runtime.shutdown()` is called, endpoints are immediately invalidated so no new requests are accepted. The behavior for in-flight requests depends on the `graceful_shutdown` parameter when serving the endpoint.

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
| `True` | Wait for all in-flight requests to complete before returning |
| `False` | Return immediately without waiting for requests |

### Component-Specific Behavior

| Component | Default Behavior | Rationale |
|-----------|------------------|-----------|
| **Frontend** | N/A (HTTP server) | HTTP server handles its own shutdown |
| **Prefill Workers** | `graceful_shutdown=True` | Prefill operations must complete to avoid wasted computation |
| **Decode Workers** | Conditional | If migration is enabled (`migration_limit > 0`), shutdown immediately to allow migration; otherwise wait |
| **Router** | `graceful_shutdown=True` | Ensure routing decisions complete |

### Decode Worker Migration Integration

Decode workers use conditional draining based on whether request migration is supported:

```python
generate_endpoint.serve_endpoint(
    handler.generate,
    graceful_shutdown=config.migration_limit <= 0,  # If no migration, wait for requests
    ...
)
```

When `migration_limit > 0`:
- Worker shuts down immediately (`graceful_shutdown=False`)
- In-flight requests are migrated to healthy workers
- No request loss occurs

When `migration_limit <= 0`:
- Worker waits for in-flight requests (`graceful_shutdown=True`)
- Migration is not available
- Requests complete on the shutting-down worker

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

### Recommended Configuration

For production deployments, configure adequate termination grace period:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
spec:
  services:
    VllmWorker:
      extraPodSpec:
        terminationGracePeriodSeconds: 60  # Allow time for request draining
```

### Health Check Integration

Kubernetes uses health endpoints to determine pod readiness:

- **During shutdown**: Endpoints become unavailable
- **Readiness probe fails**: Traffic stops routing to the pod
- **Graceful draining**: Existing requests complete

## Best Practices

### 1. Set Appropriate Grace Periods

Match `terminationGracePeriodSeconds` to your expected request completion time:
- Short requests (< 10s): 30s grace period
- Long generation (> 30s): 120s+ grace period

### 2. Enable Request Migration for Decode Workers

If using disaggregated serving, enable migration for decode workers:

```python
--migration-limit 3  # Allow up to 3 migration attempts
```

This allows immediate shutdown while preserving request state.

### 3. Monitor Shutdown Metrics

Track shutdown behavior via logs:

```
INFO  Received shutdown signal, shutting down DistributedRuntime
INFO  DistributedRuntime shutdown complete
DEBUG Cleaning up worker
```

### 4. Handle Cleanup Errors

Ensure cleanup methods handle errors gracefully:

```python
def cleanup(self):
    for resource in self.resources:
        try:
            resource.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
            # Continue with other resources
```

## Related Documentation

- [Request Migration](request_migration.md) - How requests migrate during shutdown
- [Request Cancellation](request_cancellation.md) - Canceling in-flight requests
- [Health Checks](../observability/health-checks.md) - Liveness and readiness probes
