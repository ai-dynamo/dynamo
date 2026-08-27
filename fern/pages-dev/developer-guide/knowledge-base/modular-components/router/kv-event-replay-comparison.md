---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KV Event Replay — Dynamo vs vLLM
subtitle: How the two systems handle gap detection, replay, and recovery for KV cache events
---

## Overview

Both Dynamo and vLLM publish KV cache events (block stored, block removed, etc.) over a fire-and-forget transport (ZMQ PUB/SUB). Because PUB/SUB is lossy, both systems need a mechanism for consumers to detect missed messages and recover. This document compares the two approaches.

## The Problem

A KV event consumer (router, cache coordinator) subscribes to a live stream of block events from workers. Events carry monotonically increasing sequence numbers. When the consumer detects a gap in the sequence (e.g., received seq 42 then seq 45), it needs to recover the missed events or it will have a stale, incorrect view of the worker's KV cache state.

## Architecture Comparison

| | vLLM Replay Buffer | Dynamo Local Indexer |
|---|---|---|
| **Core buffer** | `collections.deque[tuple[int, bytes]]` with `maxlen` | `VecDeque<RouterEvent>` with `max_buffer_size` |
| **Buffer semantics** | FIFO ring, old entries silently dropped | FIFO ring, old entries silently dropped |
| **Event ordering** | Monotonic sequence number (8-byte int) | Monotonic `event_id` with consecutive-ID validation |
| **Lookup** | Linear scan (`for seq, buf in buffer`) | Binary search (`binary_search_by_key`) |
| **Serialization** | Pre-serialized msgpack bytes stored in buffer | Structured events stored; serialized on demand |
| **Fallback when buffer too old** | Consumer must rebuild externally | Full RadixTree snapshot |
| **Initial sync** | Not built in — consumer starts from live stream | Tree dump (request with `start_event_id=None`) |
| **Recoverable state** | Buffer only | RadixTree snapshot (buffer is an optimization layer) |
| **Compression / dedup** | Events stored as-is (pre-serialized) | RadixTree compresses shared prefixes across sequences |
| **Expiration** | Replay history expires through `maxlen` eviction | Replay history expires through buffer eviction; event-backed tree state changes through worker events, not router TTL pruning |
| **Transport** | ZMQ PUB/SUB + ROUTER/REQ | Dynamo service RPC (request/response) |
| **Multi-rank** | Port offset per DP rank | Separate query endpoint per DP rank |
| **Thread model** | Background thread with queue | Single-threaded tokio runtime on dedicated OS thread |
| **Delivery guarantee** | Fire-and-forget live delivery; replay is bounded by retained history | Fire-and-forget live delivery; recovery can return retained events or a snapshot |
| **Duplicate/stale events** | Consumer filters by sequence number | Router filters stale event IDs and coordinates per-rank recovery |

## How Each System Works

### vLLM: Buffer-Only Replay

vLLM's `ZmqEventPublisher` (in `vllm/distributed/kv_events.py`) runs two ZMQ sockets in a background thread:

1. **PUB socket** (default `tcp://*:5557`): Streams `KVEventBatch` messages tagged with a monotonic sequence number.
2. **ROUTER socket** (optional, e.g., `tcp://*:5558`): Handles replay requests from consumers.

The publisher keeps a `deque` of the last `buffer_steps` (default 10,000) serialized batches. When a consumer detects a gap, it sends the missing start sequence number to the ROUTER socket. The publisher linearly scans the buffer and streams back all batches from that sequence onward, ending with a sentinel (`seq=-1, payload=empty`).

**Trade-offs:**
- Lightweight — no additional state beyond the buffer itself; easy to reason about and deploy.
- If the gap is older than the buffer window, the consumer must rebuild state through other means (e.g., restart and re-discover).
- No built-in initial state sync — a consumer that connects after events have already been published starts with an empty view.
- Linear scan on every replay request (no indexing into the buffer).
- Consumer handles dedup by checking `replay_seq > last_seq`.

### Dynamo: Buffer + Indexer with Tree Dump Fallback

Dynamo's `LocalKvIndexer` (in `lib/kv-router/src/indexer/local.rs`) wraps a `KvIndexer` (backed by a `RadixTree`) with a circular event buffer:

```text
LocalKvIndexer
├── indexer: KvIndexer          // Current state and snapshot source (RadixTree)
├── event_buffer: VecDeque      // Circular buffer for fast replay
└── max_buffer_size: usize
```

When the router queries a worker, the local indexer can return six response variants:

| Response | When | What happens |
|----------|------|--------------|
| `Events` | Requested start is available in the buffer | Returns retained events and a real-event watermark |
| `TreeDump` | Initial/full recovery or retained events cannot cover the request | Returns a full RadixTree snapshot as synthetic events plus the latest real-event watermark |
| `TreeDumpFailed` | The worker cannot construct an exact snapshot and the client opted into explicit failure | Returns the failure and watermark so the router can reset the rank and continue in degraded mode |
| `TooNew` | Requested range begins after the newest available event | Reports the available watermark without applying state |
| `InvalidRange` | The requested end precedes the start | Rejects the malformed range |
| `Error` | The worker query itself fails | Returns a serialized query error |

The snapshot fallback makes an evicted replay range recoverable while the worker-local indexer is available. A successful tree dump transactionally replaces that worker rank in the router's index. It is not a transport delivery guarantee: both the live stream and the query can fail, and router state can remain temporarily degraded.

## Gap Detection

Both systems detect gaps the same way: the consumer tracks the last sequence/event ID it processed and compares it against the next one received.

**vLLM** (from `examples/online_serving/kv_events_subscriber.py`):
```python
if last_seq >= 0 and seq > last_seq + 1:
    missed = seq - last_seq - 1
    replay.send((last_seq + 1).to_bytes(8, "big"))
    # ... receive and process replayed events
```

**Dynamo** (from `lib/llm/src/kv_router/indexer/recovery/worker_query_state.rs`):
The router tracks an admission cursor per worker and data-parallel rank. Discovering and activating a source with a recovery target starts an initial full recovery immediately; live events arriving during recovery are admitted or buffered according to the rank state. A later gap buffers the live event, resets that rank, and requests a full snapshot with both range bounds unset. This deliberately favors a current, self-contained snapshot over trying to splice a bounded missing range into potentially stale state.

On success, the router transactionally replaces the rank from `TreeDump`, advances to the worker's real-event watermark, then drains buffered live events. If snapshot construction or transport fails, the router resets or fences the affected rank as appropriate and continues with degraded live-event processing. A later gap or source change can trigger another recovery.

## When to Use Which

**vLLM's built-in replay** is a good fit when:
- You are running vLLM standalone and want basic gap recovery without additional infrastructure.
- Your consumer is long-lived and rarely disconnects — transient gaps are the main concern.
- You are building a custom external router or cache coordinator and want to consume KV events directly from vLLM without wrapping it in another framework.

**Dynamo's local indexer** is a good fit when:
- You need snapshot-based recovery, including initial state sync for newly joined routers or consumers that were offline for extended periods.
- You are running multiple router replicas that may start at different times and should independently rebuild cache state from workers.
- You want dedup and recovery handled by the infrastructure rather than implementing it in each consumer.

The two approaches share the same core idea — a FIFO ring buffer for catching up on small, transient gaps. Dynamo adds a RadixTree underneath, which enables a current-state snapshot fallback at the cost of additional memory and complexity. vLLM keeps replay history in the buffer, which is sufficient when consumers are stable and gaps remain inside the retained window.

For deployments using Dynamo's KV-aware routing, the local indexer is used automatically. For standalone vLLM deployments where you want to build your own event consumer, vLLM's replay buffer provides a lightweight starting point.

## See Also

- **[KV Router Index Data Structures](https://github.com/ai-dynamo/dynamo/blob/main/lib/kv-router/src/indexer/README.md)**: `RadixTree`, `ConcurrentRadixTree`, and `PositionalIndexer` internals
- **[Router Guide](router-guide.md)**: Deployment topologies and worker-set configuration
- **[Configuration and Tuning](configuration-and-tuning.md)**: Router flags and tuning details
- **[Router Design](router-design.md)**: Architecture details and event transport modes
