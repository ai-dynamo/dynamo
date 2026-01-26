---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Fault Tolerance"
---

Dynamo provides comprehensive fault tolerance mechanisms to ensure reliable LLM inference in production deployments. This section covers the various strategies and features that enable Dynamo to handle failures gracefully and maintain service availability.

## Overview

Fault tolerance in Dynamo operates at multiple levels:

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| **Request** | Migration, Cancellation | Handle in-flight request failures |
| **Worker** | Health Checks, Graceful Shutdown | Detect and recover from worker failures |
| **System** | Load Shedding, Request Rejection | Prevent system overload |
| **Infrastructure** | etcd HA, NATS resilience | Handle infrastructure component failures |

## Key Features

### Request Migration

When a worker fails during request processing, Dynamo can migrate in-progress requests to healthy workers. The migration system:

- Preserves partial generation state (accumulated tokens)
- Transparently continues generation on a new worker
- Maintains seamless token flow to clients

See [Request Migration](request-migration.md) for details.

### Request Cancellation

Dynamo supports canceling in-flight requests to free computational resources:

- Graceful stop signals for clean termination
- Kill signals for immediate termination
- Hierarchical cancellation propagation through request chains

See [Request Cancellation](request-cancellation.md) for details.

### Graceful Shutdown

Workers handle shutdown signals (SIGTERM/SIGINT) gracefully:

- Immediately stop accepting new requests
- Optionally drain in-flight requests before terminating
- Clean up resources (engines, connections, temp files)

See [Graceful Shutdown](graceful-shutdown.md) for details.

### Request Rejection (Load Shedding)

When workers are overloaded, Dynamo rejects new requests to prevent cascading failures:

- Configurable busy thresholds based on KV cache utilization
- Real-time worker load monitoring
- HTTP 503 responses with retry guidance

See [Request Rejection](request-rejection.md) for details.

### Health Checks

Dynamo provides multiple health check mechanisms:

- **HTTP Endpoints**: `/health` and `/live` endpoints for orchestration
- **Canary Health Checks**: Active monitoring via periodic test requests
- **Engine Monitoring**: Automatic shutdown on engine failure detection

See [Health Checks](../observability/health-checks.md) for details.

## Configuration Quick Reference

| Feature | Environment Variable | Default |
|---------|---------------------|---------|
| Worker health port | `DYN_SYSTEM_PORT` | `9090` |
| Canary health checks | `DYN_HEALTH_CHECK_ENABLED` | `false` (K8s: `true`) |
| Canary wait time | `DYN_CANARY_WAIT_TIME` | `10` seconds |
| Health check timeout | `DYN_HEALTH_CHECK_REQUEST_TIMEOUT` | `3` seconds |
| Decode blocks threshold | `--active-decode-blocks-threshold` | None (disabled) |
| Prefill tokens threshold | `--active-prefill-tokens-threshold` | None (disabled) |

## Failure Scenarios and Recovery

### Worker Pod Restart

1. Worker receives SIGTERM from Kubernetes
2. Endpoints are immediately invalidated (no new requests)
3. In-flight requests complete or migrate (based on configuration)
4. Resources are cleaned up
5. Pod restarts with fresh state

### Worker Crash (Unexpected)

1. etcd lease expires (TTL-based detection)
2. Client discovers endpoint removal via etcd watch
3. New requests route to remaining healthy workers
4. In-flight requests on crashed worker are migrated (if enabled)

### Network Partition

1. Worker loses connectivity to etcd/NATS
2. Lease keep-alive fails, lease eventually expires
3. Worker is removed from service discovery
4. Traffic reroutes to reachable workers

### GPU Failure

1. Engine health check detects GPU error (XID, OOM, etc.)
2. Worker initiates graceful shutdown
3. Runtime is shut down, engine cleaned up
4. Process exits with code 1 for pod restart

## Testing Fault Tolerance

Dynamo includes a comprehensive testing framework for validating fault tolerance:

- Request cancellation tests
- Migration tests with worker failures
- etcd HA failover tests
- Hardware fault injection (GPU XID, network partitions)

See [Fault Tolerance Testing](testing.md) for details.

## Related Documentation

- [Observability](../observability/README.md) - Metrics and monitoring
- [Distributed Runtime](../design-docs/distributed-runtime.md) - Service discovery architecture
- [Event Plane](../design-docs/event-plane.md) - etcd and NATS coordination
