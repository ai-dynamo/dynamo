# vLLM RFC: Per-Iteration Forward Pass Metrics via ZMQ

> For submission to https://github.com/vllm-project/vllm/issues/new?template=750-RFC.yml

---

## Title

`[RFC]: Per-iteration forward pass metrics with accurate engine-level timing`

---

## Motivation

**Problem: orchestration systems need per-iteration scheduler telemetry, but vLLM only exposes aggregated Prometheus metrics.**

Inference orchestrators (autoscalers, routers, disaggregated serving planners) need to understand the *per-iteration* cost structure of a running vLLM engine:

- How many prefill vs decode requests were in each batch?
- What was the KV cache depth distribution across decode requests?
- How many tokens were computed vs cache-hit?
- How long did the GPU forward pass actually take?
- How many requests are queued and waiting?

Today, vLLM exposes Prometheus gauge/histogram metrics that are **scraped asynchronously** by an external collector. This has fundamental limitations for per-iteration telemetry:

1. **Lossy**: Prometheus scraping is pull-based at a configurable interval. With iteration times of 10-100ms, the scraper can miss 90%+ of iterations. Gauge values reflect only the most recent state at scrape time, not the full distribution. Aggregated metrics inevitably lose information.

2. **Unsynchronized**: The scraper runs on a separate timer from the engine loop. Metrics from different gauges may reflect different iterations, making it impossible to correlate prefill/decode counts with wall time for the same batch.

3. **No per-iteration history**: There is no way to reconstruct the sequence of batch compositions over time. An autoscaler cannot build a cost model from Prometheus data because it only sees snapshots.

4. **Latency**: Push-based Prometheus (Pushgateway) uses HTTP, adding latency and overhead proportional to scrape frequency. For per-iteration emission at 100+ iterations/second, this is prohibitive.

**Why this matters for the ecosystem:**

- **NVIDIA Dynamo** currently implements this as an out-of-tree `--scheduler-cls` subclass ([InstrumentedScheduler](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/vllm/instrumented_scheduler.py)), but measuring wall time from the scheduler is inherently imprecise because the scheduler cannot observe the GPU forward pass boundary (see Proposed Change).
- **Autoscalers** (Kubernetes HPA, custom planners) need per-iteration throughput signals to make scaling decisions within seconds, not minutes.

---

## Proposed Change

### 1. Add `wall_time` measurement in EngineCore

Measure the GPU forward pass time at the exact boundary -- around `future.result()` in `EngineCore.step()` / `step_with_batch_queue()`:

```python
# In EngineCore.step():
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
...
t_start = time.monotonic()
model_output = future.result()   # blocks until GPU finishes
wall_time = time.monotonic() - t_start
...
self.scheduler.update_from_output(scheduler_output, model_output, wall_time=wall_time)
```

This is the **only** place in the codebase with direct access to both the GPU wait boundary and the scheduler output. The scheduler cannot measure this accurately because:
- In sync mode: `schedule()` returns before `execute_model` runs
- In async mode: `schedule(N+1)` runs concurrently with GPU batch N, so scheduler-side timestamps include overlap from adjacent batches

Pass `wall_time` to `update_from_output()` as a new optional kwarg so the scheduler can include it in metrics.

### 2. Define a per-iteration metrics struct

A compact, versioned struct emitted once per forward pass:

```python
class ForwardPassMetrics(msgspec.Struct, frozen=True):
    version: int = 1             # can include more info in later versions

    # Identity
    worker_id: str = ""          # unique engine instance identifier
    dp_rank: int = 0             # data parallel rank
    counter_id: int = 0          # monotonic sequence number

    # Timing (measured in EngineCore)
    wall_time: float = 0.0       # seconds, GPU forward pass time

    # Scheduled batch composition
    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0       # tokens being computed this iteration
    var_prefill_length: float = 0.0   # variance of total prompt lengths
    sum_prefill_kv_tokens: int = 0    # KV tokens read (cache hits + prior chunks)
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0     # total KV depth across decode requests
    var_decode_kv_tokens: float = 0.0

    # Queue state
    num_queued_prefill: int = 0
    sum_queued_prefill_tokens: int = 0
    num_queued_decode: int = 0        # preempted requests waiting
    sum_queued_decode_kv_tokens: int = 0
```

**Why these specific fields:**
- An autoscaler needs `wall_time` + `num_prefill_requests` + `num_decode_requests` + token counts to build a cost model of the form `latency = f(prefill_tokens, decode_batch_size, kv_depth)`.
- Variance fields enable detecting heterogeneous batches (mix of short and long sequences) which affect padding overhead and CUDA graph efficiency.
- Queue metrics enable load-aware routing and backpressure signals.
- `msgspec.Struct` is zero-copy serializable and already used by vLLM for KV cache events.

### 3. Emit via ZMQ PUB/SUB (not Prometheus)

Publish the struct over a ZMQ PUB socket bound to a configurable localhost port, using msgpack serialization:

```
ZMQ message: [topic_bytes, sequence_bytes, msgpack_payload]
```

**Why ZMQ over Prometheus:**

| | ZMQ PUB/SUB | Prometheus |
|---|---|---|
| **Delivery** | Push, every iteration | Pull, scraper interval |
| **Completeness** | Every iteration captured | 90%+ iterations missed |
| **Correlation** | All fields from same iteration in one message | Gauges may reflect different iterations |
| **Latency** | ~10us per message (IPC) | HTTP round-trip per scrape |
| **CPU overhead** | Background thread, non-blocking send | Metric registry lock contention |
| **Consumers** | Multiple SUB sockets, zero-copy | One scraper endpoint |
| **Format** | Versioned, typed, extensible (msgspec) | Flat key-value gauges |

The ZMQ publisher runs in a background daemon thread (same pattern as vLLM's existing `ZmqEventPublisher` for KV cache events). The scheduler hot path only pays for `queue.put_nowait()` on a bounded queue -- no serialization, no I/O.

**Backward compatibility: Prometheus "most recent" gauges.** For users who only need approximate metrics via existing Prometheus infrastructure, we can optionally expose the most recent `ForwardPassMetrics` as Prometheus gauges (updated in-place each iteration, scraped at whatever interval the collector uses). This is strictly less capable than the ZMQ stream but maintains compatibility with existing monitoring dashboards.

### 4. Data parallel support

Each DP rank runs its own EngineCore with its own scheduler. Each rank binds its own ZMQ PUB socket on `base_port + dp_rank`, emitting independent FPM streams tagged with `dp_rank`.

**Attention DP (non-MoE):** Each rank is fully independent (`dp_size=1` locally). Each rank emits its own FPM stream. No cross-rank coordination needed -- the consumer (autoscaler, planner) subscribes to each rank's ZMQ port independently and aggregates as needed.

**DP+EP (MoE):** Each rank has its own scheduler and emits its own FPM. Although the GPU forward pass is synchronized across ranks via collectives (`coordinate_batch_across_dp`), each rank's `wall_time` is measured locally at its own `future.result()` boundary. The measurements are nearly identical across ranks (collectives force sync), so any rank's data is representative. Consumers can average or use rank 0's data.

This is the **same approach used by KV cache events** today: each DP rank publishes to its own ZMQ port, and the relay/consumer layer handles multi-rank aggregation outside the engine.

### 5. Activation

Controlled by a new engine argument:

```
--forward-pass-metrics-port PORT   # 0 = disabled (default), >0 = ZMQ PUB base port
```

For DP deployments, rank N binds on `PORT + N`. When enabled, the scheduler base class (or a thin mixin) handles metric extraction and ZMQ publishing. No subclass override needed -- this should work with any scheduler implementation.

### 6. Wire format and versioning

- **Serialization**: msgpack via `msgspec.msgpack.Encoder` (same as KV cache events)
- **ZMQ multipart**: `[b"", seq.to_bytes(8, "big"), msgpack_payload]`
  - Empty topic allows future topic-based filtering
  - 8-byte big-endian sequence number for ordering / gap detection
  - msgpack payload is the serialized `ForwardPassMetrics`
- **Versioning**: `version` field in the struct. Consumers must check version before interpreting fields. Bump on incompatible changes.

### 7. Implementation scope

| Component | Change |
|-----------|--------|
| `EngineCore.step()` / `step_with_batch_queue()` | Measure `wall_time` around `future.result()`, pass to `update_from_output()` |
| `Scheduler.update_from_output()` | Accept optional `wall_time` kwarg |
| `SchedulerInterface` | New optional method `get_forward_pass_metrics()` or mixin |
| New: `ForwardPassMetrics` struct | In `vllm/v1/metrics/` or `vllm/v1/core/sched/` |
| New: `FpmPublisher` (ZMQ background thread) | Modeled after existing `ZmqEventPublisher` |
| `AsyncEngineArgs` | New `--forward-pass-metrics-port` argument |
| Optional: Prometheus stat logger | Expose most-recent FPM fields as gauges |

---

## Feedback Period

2 weeks.

---

## CC List

@simon-mo @youkaichao @WoosukKwon @robertgshaw2-redhat

---

## Any Other Things

**Reference implementation:** NVIDIA Dynamo's [InstrumentedScheduler](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/vllm/instrumented_scheduler.py) implements this as an out-of-tree scheduler subclass with scheduler-side timing. Moving the timing into EngineCore and the ZMQ publisher into vLLM core would:

1. Eliminate the need for `--scheduler-cls` overrides for metrics
2. Provide accurate GPU timing (not scheduler-approximate)
3. Allow any orchestration system (not just Dynamo) to consume per-iteration metrics
4. Reuse existing ZMQ infrastructure from KV cache events

**Existing ZMQ precedent in vLLM:** The KV cache event system (`KVEventsConfig`, `ZmqEventPublisher`) already uses this exact pattern -- ZMQ PUB on localhost, msgpack serialization, background thread. Forward pass metrics would follow the same architecture.

**Not in scope:** How consumers (Dynamo, custom autoscalers, etc.) subscribe, relay, or aggregate these metrics. That is consumer-side logic. This RFC only covers emission from vLLM.
