---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: KV Event Replay — Dynamo vs vLLM
subtitle: How vLLM replay and Dynamo worker-local recovery protect different KV event hops
---

## Overview

vLLM's replay buffer and Dynamo's worker-local indexer protect different parts of the KV event
path. vLLM can offer replay to a consumer of its raw ZMQ engine stream. Dynamo consumes and
normalizes that engine stream inside the worker. It then uses its worker-local indexer to recover
routers that miss events on Dynamo's separate worker-to-router event plane.

Dynamo's vLLM integration subscribes to the raw PUB stream. It does not use the vLLM replay socket.
A worker-local snapshot restores all cache state that the Dynamo worker indexed. It cannot
reconstruct a raw engine event that did not reach the worker.

## The Problem

There are two possible loss boundaries:

1. **Engine to Dynamo worker:** The engine publishes raw events over ZMQ. Dynamo normalizes accepted
   events and assigns its own event IDs. The listener reads the engine sequence number for logs.
   It does not use the engine replay socket. Therefore, the Dynamo worker-local indexer cannot
   recover a lost raw message.
2. **Dynamo worker to router:** The worker updates its worker-local indexer before it publishes
   normalized events over Dynamo's event plane. If the router detects a sequence gap, it can query
   the worker.
   The worker returns retained events or a current tree snapshot.

## Architecture Comparison

| | vLLM Replay Buffer | Dynamo Local Indexer |
|---|---|---|
| **Recovery boundary** | Raw vLLM publisher to a replay-aware consumer | Dynamo worker to router |
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
| **Delivery guarantee** | Live delivery is fire-and-forget. Replay retains a bounded history. | Recovery can return retained normalized events or a snapshot. It does not repair engine-to-worker loss. |
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

After the Dynamo worker accepts and normalizes an engine event, it applies the resulting state
update to `LocalKvIndexer`. It does this before publication over Dynamo's event plane.
`LocalKvIndexer` (in `lib/kv-router/src/indexer/local.rs`) wraps a `KvIndexer` (backed by a
`RadixTree`) with a circular event buffer:

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

Both recovery mechanisms use increasing IDs, but they operate on different sequences. A
replay-aware vLLM consumer tracks the engine publisher sequence. The Dynamo router tracks IDs that
the worker assigns after it accepts and normalizes raw events.

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

The Dynamo worker assigns this sequence after raw-event filtering. Therefore, its gap detection
does not reveal an engine ZMQ message that was lost before ingestion.

## When to Use Which

**vLLM's built-in replay** is a good fit when:
- You are running vLLM standalone and want basic gap recovery without additional infrastructure.
- Your consumer is long-lived and rarely disconnects — transient gaps are the main concern.
- You are building a custom external router or cache coordinator and want to consume KV events directly from vLLM without wrapping it in another framework.

**Dynamo's local indexer** is a good fit when:
- You need snapshot-based recovery, including initial state sync for newly joined routers or consumers that were offline for extended periods.
- You are running multiple router replicas that may start at different times and should independently rebuild cache state from workers.
- You want dedup and recovery handled by the infrastructure rather than implementing it in each consumer.

Both approaches use a FIFO ring buffer to recover small, temporary gaps. They are not
interchangeable. vLLM replay can protect the raw engine hop for consumers that use its replay
socket. Dynamo adds a RadixTree after engine ingestion. This tree provides current-state snapshots
for routers on the second hop.

For Dynamo KV-aware routing, Dynamo enables the worker-local indexer after the worker has a KV event
publisher. vLLM and SGLang still require `--kv-events-config` to feed that publisher. For standalone
vLLM deployments, the replay buffer provides a lightweight base for a custom event consumer.

## See Also

- **[KV Router Index Data Structures](https://github.com/ai-dynamo/dynamo/blob/main/lib/kv-router/src/indexer/README.md)**: `RadixTree`, `ConcurrentRadixTree`, and `PositionalIndexer` internals
- **[Router Guide](router-guide.md)**: Deployment modes and quick start for KV-aware routing
- **[Configuration and Tuning](configuration-and-tuning.md)**: Router flags and tuning details
- **[Router Design](router-design.md#event-flow-and-recovery)**: Engine-to-worker ingestion,
  worker-local indexing, and router recovery
