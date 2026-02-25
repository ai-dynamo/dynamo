---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Event Plane
---

# Dynamo Event Plane

The event plane provides Dynamo with a pub/sub layer for near real-time event exchange between components. It delivers KV cache updates, worker load metrics, and sequence tracking events, enabling features like KV-aware routing and disaggregated serving.

## When Is the Event Plane Used?

Key use cases:

- **KV cache events** -- Workers publish cache state so the router can make cache-aware scheduling decisions.
- **Worker load metrics** -- Workers report utilization so the router can balance load.
- **Sequence tracking** -- Coordinates active sequences across router replicas for fault-tolerant routing.

![Event plane architecture showing NATS and ZMQ transport options connecting Frontend, Planner, and Worker](../../assets/img/event-plane-transport.svg)

## Choosing a Transport

The event plane supports two transports:

| | NATS (default) | ZMQ |
|---|---|---|
| **External infrastructure** | Requires a NATS server | None (peer-to-peer) |
| **Setup complexity** | Simple -- point at a NATS server | Automatic -- workers bind sockets and register via discovery |
| **Best for** | Large-scale deployments | Low operational overhead |

## Configuration

### Transport Selection

Set the `DYN_EVENT_PLANE` environment variable to choose a transport:

```bash
# Use NATS (default -- no need to set explicitly)
export DYN_EVENT_PLANE=nats

# Use ZMQ
export DYN_EVENT_PLANE=zmq
```

Python components also accept this as a CLI flag:

```bash
# vLLM backend
python3 -m dynamo.vllm --event-plane zmq --model Qwen/Qwen3-0.6B

# SGLang backend
python3 -m dynamo.sglang --event-plane zmq --model Qwen/Qwen3-0.6B
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DYN_EVENT_PLANE` | Transport: `nats` or `zmq` | `nats` |
| `NATS_SERVER` | NATS server URL (NATS transport only) | `nats://localhost:4222` |

## NATS Transport

When using NATS (`DYN_EVENT_PLANE=nats` or unset):

- Requires a running NATS server. Set `NATS_SERVER` if it is not on `localhost:4222`.
- Events are published to NATS subjects scoped by namespace and component.
- Built-in reconnection and message buffering during brief disconnections.

Example setup:

```bash
export NATS_SERVER=nats://nats-server:4222
export DYN_EVENT_PLANE=nats

# Start workers -- explicitly enable KV event publishing
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B \
    --kv-events-config '{"publisher":"nats","topic":"kv-events","enable_kv_cache_events":true}'

# Start frontend -- it subscribes to events from NATS automatically
python3 -m dynamo.frontend --router-mode kv
```

## ZMQ Transport

When using ZMQ (`DYN_EVENT_PLANE=zmq`):

- No external server required. Each worker binds a ZMQ PUB socket and advertises its address through the discovery system.
- Subscribers automatically discover and connect to all active publishers.
- When publishers come and go (e.g., workers scaling up/down), subscribers dynamically adjust their connections.

Example setup:

```bash
export DYN_EVENT_PLANE=zmq

# Start workers -- each binds a ZMQ socket, registers with discovery
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'

# Start frontend -- discovers workers and connects directly
python3 -m dynamo.frontend --router-mode kv
```

## Disabling the Event Plane

If you do not need KV-aware routing, you can disable the event plane entirely:

```bash
python3 -m dynamo.frontend --router-mode kv --no-router-kv-events
```

With `--no-router-kv-events`:

- The router falls back to prediction-based cache-aware routing (estimates cache state from routing decisions).
- No NATS server or ZMQ sockets are needed.
- TTL-based expiration and LRU pruning keep predicted state from growing stale.

## Deployment Modes

### Bare Metal / Local

Both transports work out of the box:

```bash
# NATS (requires nats-server running)
export NATS_SERVER=nats://localhost:4222

# OR ZMQ (no extra infrastructure)
export DYN_EVENT_PLANE=zmq
```

### Kubernetes (with Dynamo Operator)

The operator can inject `DYN_EVENT_PLANE` into pods. The same transport options apply. If using NATS, deploy a NATS server in the cluster and set `NATS_SERVER` accordingly.

## Related Documentation

- [Discovery Plane](discovery-plane.md) -- Service discovery and coordination (etcd, Kubernetes)
- [Distributed Runtime](distributed-runtime.md) -- Runtime architecture
- [Request Plane](request-plane.md) -- Request transport configuration
- [Fault Tolerance](../fault-tolerance/README.md) -- Failure handling
