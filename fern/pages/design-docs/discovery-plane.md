---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Discovery Plane
---

# Discovery Plane

Dynamo's service discovery layer lets components find each other at runtime. Workers register their endpoints when they start, and frontends discover them automatically.
The discovery backend adapts to the deployment environment.

![Discovery plane architecture showing Kubernetes and etcd backends](../../assets/img/discovery-plane.svg)

## Discovery Backends

| Deployment | Discovery Backend | Configuration |
|------------|-------------------|---------------|
| **Kubernetes** (with Dynamo operator) | Native K8s (CRDs, EndpointSlices) | Operator sets `DYN_DISCOVERY_BACKEND=kubernetes` |
| **Bare metal / Local** (default) | etcd | `ETCD_ENDPOINTS` (defaults to `http://localhost:2379`) |

> **Note:** The runtime always defaults to etcd. Kubernetes discovery must be explicitly enabled -- the Dynamo operator handles this automatically.

## Kubernetes Discovery

When running on Kubernetes with the Dynamo operator, service discovery uses native Kubernetes resources instead of etcd.

### How It Works

1. Workers register their endpoints by creating **DynamoWorkerMetadata** custom resources.
2. **EndpointSlices** signal pod readiness to the system.
3. Components watch for CRD changes to discover available workers.

### Benefits

- No external etcd cluster required.
- Native integration with Kubernetes pod lifecycle.
- Automatic cleanup when pods terminate.
- Works with standard Kubernetes RBAC.

### Environment Variables (Injected by Operator)

| Variable | Description |
|----------|-------------|
| `DYN_DISCOVERY_BACKEND` | Set to `kubernetes` |
| `POD_NAME` | Current pod name |
| `POD_NAMESPACE` | Current namespace |
| `POD_UID` | Pod unique identifier |

## etcd Discovery (Default)

When `DYN_DISCOVERY_BACKEND` is not set (or set to `etcd`), etcd is used for service discovery.

### Connection Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ETCD_ENDPOINTS` | Comma-separated etcd URLs | `http://localhost:2379` |
| `ETCD_AUTH_USERNAME` | Basic auth username | None |
| `ETCD_AUTH_PASSWORD` | Basic auth password | None |
| `ETCD_AUTH_CA` | CA certificate path (TLS) | None |
| `ETCD_AUTH_CLIENT_CERT` | Client certificate path | None |
| `ETCD_AUTH_CLIENT_KEY` | Client key path | None |

Example:

```bash
export ETCD_ENDPOINTS=http://etcd-0:2379,http://etcd-1:2379,http://etcd-2:2379
```

### Service Registration

Workers register their endpoints in etcd with a key hierarchy:

```
/services/{namespace}/{component}/{endpoint}/{instance_id}
```

For example:

```
/services/vllm-agg/backend/generate/694d98147d54be25
```

Frontends and routers discover available workers by watching the relevant prefix and receiving real-time updates when workers join or leave.

### Lease-Based Cleanup

Each runtime maintains a lease with etcd (default TTL: 10 seconds). If a worker crashes or loses connectivity:

![Lease lifecycle showing DistributedRuntime keep-alive heartbeat to etcd](../../assets/img/discovery-plane-lease.svg)

1. Keep-alive heartbeats stop.
2. The lease expires after the TTL.
3. All registered endpoints are automatically deleted.
4. Clients receive removal events and reroute traffic to healthy workers.

This ensures stale endpoints are cleaned up without manual intervention.

## KV Store

Dynamo provides a KV store abstraction for storing metadata (endpoint instances, model deployment cards, event channels). Multiple backends are supported:

| Backend | Use Case |
|---------|----------|
| etcd | Production deployments |
| Memory | Testing and development |
| NATS | NATS-only deployments |
| File | Local persistence |

## Operational Guidance

### Use Kubernetes Discovery on K8s

The Dynamo operator automatically sets `DYN_DISCOVERY_BACKEND=kubernetes` for pods. No additional setup required.

### Deploy an etcd Cluster for Bare Metal

For bare-metal production deployments, deploy a 3-node etcd cluster for high availability.

### Tune Lease TTLs

Balance between failure detection speed and overhead:

- **Short TTL (5s)** -- Faster failure detection, more keep-alive traffic.
- **Long TTL (30s)** -- Less overhead, slower detection.

The default (10s) is a reasonable starting point for most deployments.

## Related Documentation

- [Event Plane](event-plane.md) -- Pub/sub for KV cache events and worker metrics
- [Distributed Runtime](distributed-runtime.md) -- Runtime architecture
- [Request Plane](request-plane.md) -- Request transport configuration
- [Fault Tolerance](../fault-tolerance/README.md) -- Failure handling
