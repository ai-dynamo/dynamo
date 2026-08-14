# KV Publisher Event Pipeline

This module turns engine KV-cache events into Dynamo router events. The ZMQ
source is the common production path for vLLM and SGLang, while the event
processor publishes normalized events through the configured event plane.

## Files

- `zmq_listener.rs`: receives engine ZMQ batches and converts wire events into
  placement-aware events.
- `event_processor.rs`: owns the async receive loop and preserves event ordering.
- `batching.rs`: coalesces adjacent store/remove events before publish.
- `dedup.rs`: gates duplicate removes with per-rank, per-tier refcounts.
- `sinks.rs`: applies events to the local worker indexer and publishes them
  externally.
- `worker_metrics.rs`: emits worker-local runtime metrics.

## Stage Notes

`zmq_listener.rs` skips ignored raw events before assigning the publisher-local
event id. This preserves the current contract that filtered ZMQ events do not
advance the core-router event id. The accepted events from one native
`KvEventBatch` are forwarded to the processor as one ordered
`Vec<PlacementEvent>`.

`event_processor.rs` checks for gaps in the raw input event id stream before
batching. That metric is about events dropped before this processor receives
them; batching and dedup do not affect it.

`batching.rs` merges compatible adjacent removes or stores. It flushes when the
event kind changes, the DP rank changes, the storage tier changes, the store
parent chain breaks, the timeout expires, or the pending block cap is reached.
The cap is checked after each complete source event, so it bounds coalescing
without splitting an individual source event.

With `batching_timeout_ms` set to `None` or `0`, the processor flushes the
compatible tail at each native-list boundary. Compatible events inside that
native list are still coalesced. With a positive timeout, the tail may remain
pending across native lists until the deadline or another flush condition.

`dedup.rs` is applied during flush. Stores update refcounts and still pass
through. Removes only pass through when the refcount for a block reaches zero;
unknown removes pass through defensively.

`sinks.rs` applies the emitted router event to the optional local indexer before
publishing it externally. Both side effects receive the same `RouterEvent`.

## Ordering

```mermaid
sequenceDiagram
    participant L as zmq_listener
    participant P as event_processor
    participant B as batching
    participant D as dedup
    participant S as sinks

    L->>P: Vec<PlacementEvent> (one native batch)
    loop each source event, without returning to select!
        P->>P: record raw event-id gap metric
        P->>B: append to pending batch
        B->>B: flush at structural boundary or block cap
    end
    Note over P,B: None/0 flushes the tail here; a timeout may retain it across lists
    B->>D: filter pending removes / track stores
    D-->>B: filtered event data
    B->>S: emit RouterEvent
    S->>S: apply local indexer
    S->>S: publish external event
```
