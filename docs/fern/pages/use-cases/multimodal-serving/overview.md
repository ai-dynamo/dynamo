---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal Model Serving
subtitle: Deploy multimodal models with image, video, and audio support in Dynamo
---

Dynamo supports multimodal inference across multiple LLM backends, enabling models to process images, video, and audio alongside text.

## Which Feature to Use

Dynamo provides support for improving latency and throughput for multimodal workloads, with image and video inputs, through the following features. Use them together or separately, depending on your workload characteristics:

| Workload | Feature | Benefit |
|----------|---------|---------|
| Workload where significant time is spent preprocessing. | [Parallel media decoding](parallel-media-decoding.md) | Move image fetching and decompression off the worker's critical path. |
| Workload includes repeated multimodal content across requests. | [Embedding cache](embedding-cache.md) | Skip re-encoding repeated multimodal content. |
| Workload includes repeated multimodal content across requests, and multiple backend workers serve multimodal requests. | [Multimodal KV routing](multimodal-kv-routing.md) | Maximize KV cache hit rates for multimodal content. |
| Workload where media encoding is a bottleneck. | [Encoder disaggregation](encoder-disaggregation.md) | Scale encoders independently of LLM workers. |


> [!IMPORTANT]
> These features currently support image and video inputs only. Support for audio modalities will be added in upcoming releases.

## Multimodal Performance Optimization Features

<CardGroup cols={2}>
  <Card title="Parallel Media Decoding" icon="regular image" href="parallel-media-decoding.md">
    Decode image inputs concurrently in the Rust frontend and transfer pixels through NIXL
  </Card>
  <Card title="Embedding Cache" icon="regular database" href="embedding-cache.md">
    Cache vision encoder embeddings to skip re-encoding repeated multimodal content
  </Card>
  <Card title="Multimodal KV Routing" icon="regular arrows-split-up-and-left" href="multimodal-kv-routing.md">
    Include multimodal identity in cache-aware, load-balanced worker selection
  </Card>
  <Card title="Encoder Disaggregation" icon="regular microchip" href="encoder-disaggregation.md">
    Separate vision encoding into a dedicated worker for independent scaling
  </Card>
</CardGroup>

## Example Workflows

Reference implementations for deploying multimodal models for each backend:

- [SGLang Multimodal](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md)
- [TensorRT-LLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md)
- [vLLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md)

To use an author-provided custom vision tower or projector, see [Custom Vision Encoders](custom-vision-encoders.md).

## Shared Image Download Cache

The optional shared cache stores encoded bytes from HTTP and HTTPS image URLs
before image decoding. Workers can reuse those bytes across processes while
keeping their local decoded-image caches independent. Data URLs,
frontend-decoded NIXL inputs, video, and audio do not use this cache.

### Backend Support

| Backend | Mode | Shared encoded-image cache |
| --- | --- | --- |
| vLLM | Aggregated and disaggregated paths that load URLs through Dynamo `ImageLoader` | Supported |
| TensorRT-LLM | Aggregated and disaggregated paths that load URLs through Dynamo `ImageLoader` | Supported |
| SGLang | Standard URL loading | Not supported. Dynamo passes URL strings directly to SGLang. |
| SGLang | `--frontend-decoding` | Only URL items that still reach Dynamo `ImageLoader` can use the cache. NIXL-decoded items bypass it. |

Dynamo does not provision or discover a cache service. Deploy Redis Cluster or
Dragonfly separately and provide a cluster endpoint that every participating
URL-loading worker can reach. The connection URL carries the endpoint,
authentication, and TLS choice. See [Cache Service Deployment](#cache-service-deployment)
for single-node and multi-node layouts.

Set the following environment variables on every participating worker:

| Environment variable | Default | Description |
| --- | --- | --- |
| `DYN_MM_SHARED_IMAGE_CACHE_ENABLED` | `0` | Set to `1` to enable the shared cache. |
| `DYN_MM_SHARED_IMAGE_CACHE_URL` | None | Redis Cluster or Dragonfly connection URL. Required when the cache is enabled. |
| `DYN_MM_SHARED_IMAGE_CACHE_TTL_SECS` | `3600` | Positive cache-entry lifetime in seconds. |
| `DYN_MM_SHARED_IMAGE_CACHE_CONNECT_TIMEOUT_SECS` | `0.1` | Positive connection timeout in seconds. |
| `DYN_MM_SHARED_IMAGE_CACHE_IO_TIMEOUT_SECS` | `2.0` | Positive Redis operation timeout in seconds. |
| `DYN_MM_MAX_FILE_SIZE_MB` | `64` | Maximum encoded image size downloaded from the origin, in MiB. The limit applies whether or not the shared cache is enabled, so oversized images are never stored in it. |
| `DYN_MM_IMAGE_CACHE_SESSION_SCOPED` | `0` | Set to `1` to partition Dynamo image and image-embedding caches by session affinity. |

Store the full connection URL in a Kubernetes Secret and inject it into each
worker rather than placing credentials in a manifest:

```yaml
env:
  - name: DYN_MM_SHARED_IMAGE_CACHE_ENABLED
    value: "1"
  - name: DYN_MM_SHARED_IMAGE_CACHE_URL
    valueFrom:
      secretKeyRef:
        name: multimodal-image-cache
        key: connection-url
  - name: DYN_MM_IMAGE_CACHE_SESSION_SCOPED
    value: "1"
```

> [!WARNING]
> Use a `rediss://` URL unless Redis traffic is protected by equivalent
> transport encryption, such as an encrypted service mesh. A `redis://` URL
> sends cache data and any URL credentials without TLS and should be used only
> on an appropriately secured in-cluster network.

Cache reads and fills run on the request path. A miss waits for Redis `SET` so
the fill completes before the request returns. Redis errors fail open, and each
cache operation is attempted once. While a cache node is unreachable, an image
load pays one failed `GET` and one failed `SET`: each is bounded by
`DYN_MM_SHARED_IMAGE_CACHE_CONNECT_TIMEOUT_SECS` when a new connection is
refused or times out, and by `DYN_MM_SHARED_IMAGE_CACHE_IO_TIMEOUT_SECS` when
an open connection stops responding.
Warnings are emitted initially and at most once per minute; individual
operation failures remain available at debug level. Dynamo does not currently
use a bounded asynchronous write queue.

### Cache Service Deployment

Dynamo uses the Redis Cluster protocol for both backends. The client uses
`DYN_MM_SHARED_IMAGE_CACHE_URL` only to discover the cluster: it reads the slot
map from that endpoint and then connects to each node at the address the node
announces. When a node stops responding, the client rediscovers through the same
URL, so point it at an address that survives cache restarts, such as a
Kubernetes Service that selects every cache node, not at an individual pod.
After a restart, the nodes announce their new addresses and workers pick them
up without restarting. Dynamo requires redis-py 6.2 or later.

**Single-node Dragonfly.** Run Dragonfly with `--cluster_mode=emulated` so that
it answers Redis Cluster commands, and keep its default announced address (the
pod IP). Do not set `--cluster_announce_ip` to the host in
`DYN_MM_SHARED_IMAGE_CACHE_URL`: with redis-py 6.2 through 7.1, a node that
announces the same host and port as the URL removes the URL from the client's
rediscovery list after one failed operation.

**Redis Cluster.** Run the nodes as a StatefulSet with a headless Service for
per-pod DNS names, plus a regular Service that selects every node for the URL.
Configure each node with:

- `cluster-announce-ip` set to the pod IP, or `cluster-announce-hostname` set to
  the pod's DNS name together with `cluster-preferred-endpoint-type hostname`
  (Redis 7.0 or later).
- `cluster-require-full-coverage no`, so the remaining nodes keep serving while
  one node is down. Images that hash to the unavailable node fall back to the
  origin.
- A `nodes.conf` that persists across pod restarts, so a restarted node keeps its
  identity and slots. A rolling restart then keeps the cluster available. If
  every node restarts at once, all peer addresses change and the nodes need
  their peers' new addresses before they can re-form the cluster.

**Multi-node Dragonfly.** With `--cluster_mode=yes`, Dragonfly nodes do not
discover each other. A cluster orchestrator must push the slot map with
`DFLYCLUSTER CONFIG` and push it again whenever a node restarts, because the
map is kept in memory. Point the URL at a Service that selects every node, as
for Redis Cluster.

### Session Scoping

Enable `DYN_MM_IMAGE_CACHE_SESSION_SCOPED=1` when the same URL can resolve to
different content for different sessions. Dynamo derives the
scope from these headers, in precedence order:

1. `x-dynamo-session-id`
2. Recognized agent headers: Claude Code `x-claude-code-session-id` (or
   `x-claude-code-agent-id` for a child agent), Codex `thread-id`, then OpenCode
   `x-session-id`

Use `x-dynamo-session-id` to provide an explicit Dynamo session identity.
Dynamo normalizes it and the recognized agent headers into both an
`AgentContext` and the session affinity used for routing and image-cache
scoping.

> [!WARNING]
> Session scoping partitions cache entries; it is not an authentication
> boundary. Unless a trusted proxy sets or overwrites `x-dynamo-session-id`,
> the caller controls the scope. Configure the proxy to derive this header
> from the authenticated identity on every request, and treat scope identifiers
> as secrets. A caller that learns another identifier can reuse cached content
> stored under that scope for the same URL.

When session scoping is enabled, a missing or blank scope bypasses the local
decoded-image cache, in-flight request deduplication, the shared encoded-image
cache, and Dynamo-owned image embedding caches. A valid scope partitions all of
those caches. The scope does not partition backend KV cache keys; affinity can
route a session back to the same worker, but KV identity remains content-based.
