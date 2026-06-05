# KV Router Memory Failure-Mode Campaign

Date: June 5, 2026

Branch source: `f68be6d409efb3ac3eb6874bebcd3077a9c131e6`

Merged change under test: [fix(kv-router): harden cancellation and replica sync (#10331)](https://github.com/ai-dynamo/dynamo/commit/f68be6d409efb3ac3eb6874bebcd3077a9c131e6)

## Executive Summary

The observed frontend RSS growth is not one bug. The campaign isolated five
material failure modes:

1. **HTTP cancellation cleanup leak, fixed by `f68be6d`.** Before the fix,
   cancelled requests could remain in the scheduler queue or active tracker.
   In the matched forced-cancellation A/B, scheduler-pending state fell from
   792,165 to 923 at peak, a 99.88% reduction.
2. **Legitimate overload state.** When clients keep more requests live than
   workers can complete, queued and booked requests legitimately consume
   memory. Fixed concurrency eventually plateaus. It is still operationally
   unsafe if the configured concurrency or arrival population is enormous.
3. **KV index/cache cardinality.** With replica sync off, unique 4k-token
   prompts grew 84 frontends to 1,030,676 MiB aggregate RSS, about 1.0 TiB.
   A matched high-reuse run peaked at 18,274 MiB. Both drained active and
   pending requests to zero. This was the largest independent memory driver.
4. **Replica event loss under burst pressure.** The side-effect batching work
   improves consumer throughput and removes an avoidable drop point, but it
   does not make delivery reliable. Natural runs still logged subscriber lag
   and skipped events when event production exceeded one subscriber loop.
5. **Allocator and container high-water retention.** Logical cleanup can be
   correct while RSS remains high. Tries, maps, vectors, and glibc arenas keep
   capacity for reuse. This is not the same as live logical state leaking.

The mitigated combined run used bounded router queues, token-capacity
admission, the cancellation fix, moderate prompt reuse, KV events, and replica
sync. It handled 21,134 forced HTTP cancellations, issued 363,557 backpressure
rejections, logged no replica lag or input gaps, and drained active and pending
state to zero.

## Production Conclusions

- Deploying `f68be6d` is necessary. It closes a severe cancellation-driven
  scheduler leak, but it cannot prevent legitimate overload growth.
- Configure an actual queue cap. `DYN_ROUTER_QUEUE_THRESHOLD=0.0` controls
  queueing sensitivity; it does not bound queue length.
- Enable token-capacity admission when worker capacity is known.
- Treat KV prompt/index cardinality as a first-class memory budget. Replica
  sync is not required for KV-event workloads to reach OOM-level RSS.
- Replica sync dropped events are still possible. Batching raises the loss
  threshold; it does not provide delivery or repair guarantees.
- TTL is a cleanup backstop, not overload control. It also does not force RSS
  to return after data structures and the allocator reached a high-water mark.

## Test Environment

- ComputeLab allocation: four exclusive CPU nodes
  `lego-c2-qs-[25,35,37-38]`
- Per node: 144 CPU cores and approximately 450 GiB allocatable memory
- Standard Jie-scale topology where applicable:
  - 168 frontends
  - 336 prefill mocker workers
  - 168 decode mocker workers
  - 60 AIPerf load generators
- Disaggregated routing with etcd discovery and NATS request plane
- ZMQ event plane
- Source tree: local commit `f68be6d409`; remote build tree had a different
  commit identifier but the identical Git tree
  `86183201d06c6f10076cd3b0d5e9ce43652cfcaa`
- Main prompt shape: 4,096 input tokens; output length varied by direction
- Queue tiers were injected through the existing
  `router_queue_by_incoming_missing_isl` configuration field because this
  commit did not expose it as a frontend CLI option

The raw campaign artifacts remain under:

```text
/home/scratch.rupei_gpu/router-memory-8way-20260605
```

## Direction 1: Overload And Admission

### Normal fixed concurrency

Configuration:

- 168 frontends, 504 workers, 60 load generators
- 256 concurrency per generator, 15,360 total
- KV events off, replica sync off
- No queue cap and no token-capacity admission

Result:

- Peak active requests: 15,360
- Peak scheduler-pending requests: 10
- Final active and pending: 0

This workload did not produce an unbounded queue. The live set plateaued at
the configured client concurrency and drained.

### High fixed concurrency, unbounded queue

Configuration:

- Same topology
- 2,048 concurrency per generator, 122,880 total

Result:

- Peak active requests: 122,880
- Peak scheduler-pending requests: 59,451
- Peak pending ISL tokens: approximately 244 million
- Peak aggregate frontend RSS: 27,447 MiB

This was large but still bounded by fixed client concurrency. Cleanup was
progressing after client shutdown, but a 45-second drain was insufficient to
reach zero.

### High fixed concurrency, bounded queue

Configuration:

- Same high-concurrency workload
- Queue tiers: `[(0,512)]`

Result:

- Peak scheduler-pending requests: 7,050
- Queue backpressure responses: 1,018,646
- Final active: 39
- Final pending: 32
- Peak aggregate frontend RSS: 32,618 MiB

The cap materially bounded logical queued state. RSS high-water was higher
than the unbounded run because the system generated and retained capacity for
an extreme volume of HTTP 503 responses. Queue bounding solves queue growth,
not all overload allocation.

## Direction 2: Forced HTTP Cancellation

The matched five-minute A/B used the Jie-scale topology, streaming 4,096 input
plus 4,096 output tokens, and client timeouts rotated across 50 ms, 100 ms,
250 ms, 500 ms, and 1 second.

| Metric | Pre-fix baseline | Patched candidate |
| --- | ---: | ---: |
| Peak scheduler-pending | 792,165 | 923 |
| Pending after drain | 787,589 | 828 |
| Peak active | 912,321 | 30,876 |
| Aggregate RSS high-water | 58,453 MiB | 26,860 MiB |

The patch materially fixes the cancellation leak. The candidate's remaining
active state was backend/in-flight work, not the old scheduler-pending
accumulation.

The reproduction requires real HTTP disconnects. A load test where requests
complete normally, or where timed-out clients do not close their sockets,
does not exercise this failure mode.

## Direction 3: KV Events And Prompt Cardinality

Both runs used:

- 84 frontends and 504 workers
- KV events on
- Replica sync off
- Prefix caching on
- Approximately 2,048 concurrent requests
- 4,096 input tokens and 64 output tokens
- Peak active requests near 2,040
- Zero subscriber-lag and input-gap warnings
- Final active and pending requests: 0

| Prompt population | Aggregate RSS peak | Approx. RSS/frontend |
| --- | ---: | ---: |
| Unique prompts | 1,030,676 MiB | 12.0 GiB |
| 16-entry high reuse | 18,274 MiB | 0.21 GiB |

Unique prompts consumed about 56 times more aggregate RSS at nearly identical
active concurrency. Because replica sync was off and logical state drained,
this isolates KV prompt/index cardinality and allocator retention.

This result also explains why `USE_KV_EVENTS=true` with replica sync disabled
can still reach OOM-level memory. CRTC or equivalent prefix-index state must
be disabled, approximated, or explicitly bounded for low-reuse populations.

## Direction 4: Replica Sync Steady State

Configuration:

- 84 frontends and 504 workers
- Replica sync on
- KV events off and overlap credit `0.0`
- 2,048 configured concurrency
- Achieved request throughput: approximately 570 requests/second

Result:

- Peak active: 2,116
- Peak pending: 1,407
- Initialized aggregate RSS: approximately 16.3 GiB
- Aggregate RSS high-water: 18,658 MiB
- Subscriber lag: 0
- Input gaps: 0
- Final active and pending: 0

Replica sync itself was stable at this event rate. Its memory overhead was
modest compared with the unique-prompt KV-index run.

## Direction 5: Forced Replica Subscriber Stall

One of 32 frontends was sent `SIGSTOP` for 10 seconds during replica-only
traffic, then resumed.

Result on the paused frontend:

- Subscriber-lag warnings: 8
- Events explicitly reported skipped: 31,290
- Peer lag warnings: 0
- Kernel OOM record: none

The paused frontend subsequently exited, so this run cannot measure its final
mirrored state. It does prove that the event plane drops lifecycle events when
a subscriber cannot drain fast enough.

## Direction 6: Replica Fan-Out And Event Rate

The sweep held configured aggregate concurrency at 2,048 but achieved
different throughput because frontend count changes routing and HTTP
overhead.

| Frontends | Achieved req/s | Lag warnings | Skipped events | Final active | Final pending | RSS high-water |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 668 | 50 | 25,243 | 168 | 168 | 1,442 MiB |
| 32 | 1,645 | 10 | 2,918 | 8 | 8 | 11,146 MiB |
| 84 | 570 | 0 | 0 | 0 | 0 | 18,658 MiB |

Interpretation:

- Replica event loss is not monotonic in frontend count.
- Event production rate and burst concentration relative to the single
  subscriber loop are the controlling variables.
- The residual records in the lagging runs are consistent with missed
  lifecycle cleanup. They are supporting evidence, not a precise orphan
  count, because remaining AIPerf wrapper processes were terminated through
  their recorded PID manifests after the benchmark window.
- Aggregate steady-state replica memory scales with frontend count because
  every frontend keeps a global mirrored view.

The batching patch is still useful: it increases apply throughput and removes
an avoidable intermediate broadcast drop point. These runs show that it does
not make the ZMQ event plane reliable under all burst patterns.

## Direction 7: Allocator Retention

Several runs reached zero logical state while RSS remained at its peak:

- KV unique: active 0, pending 0, RSS approximately 1.0 TiB
- KV reuse: active 0, pending 0, RSS 18,274 MiB
- Bounded overload: active 39, pending 32, RSS 32,618 MiB

The data structures remove records, but their backing allocations do not have
to shrink. Hash maps retain bucket arrays, vectors retain capacity, tries keep
nodes or slabs for reuse, and glibc arenas retain pages. Therefore:

- High RSS after drain does not prove live requests are still present.
- Calling `shrink_to_fit()` on every free path is not the primary correctness
  fix and would add CPU and fragmentation costs.
- Cardinality and admission limits are more important than forcing shrinkage.
- Allocator selection or periodic trimming can be evaluated separately when
  returning RSS to the OS is an operational requirement.

## Direction 8: Combined Mitigated Case

Configuration:

- 32 frontends
- 64 prefill plus 32 decode workers
- Replica sync on
- KV events and prefix indexing on
- 128 prompt entries per load generator
- Queue tiers: `[(0,256)]`
- Token-capacity admission enabled
- 1,024 configured concurrency
- Rotated 50 ms to 1 second HTTP cancellation timeouts
- 60-second load and 60-second drain

Result:

- HTTP cancellation warnings: 21,134
- Peak active: 836
- Peak pending: 128
- Queue backpressure responses: 363,557
- Subscriber lag: 0
- Input gaps: 0
- Final active: 0
- Final pending: 0
- Aggregate RSS high-water: 6,128 MiB

This is the cleanest production-shaped outcome in the campaign. It combines
early rejection, finite queued state, cancellation-safe booking, and bounded
prompt diversity.

## Failure-Mode Matrix

| Failure mode | Trigger | Logical state after drain | RSS behavior | Primary control |
| --- | --- | --- | --- | --- |
| Cancellation leak | HTTP client disconnect races booking/selection | Pre-fix state persists | Grows with cancelled requests | Deploy `f68be6d` |
| Legitimate overload | Arrival/live concurrency exceeds service rate | Drains when clients stop | Tracks live queue high-water | Queue cap and admission |
| KV cardinality | Unique low-reuse prompts with KV events | Can reach zero | Can remain extremely high | Bound/approximate/disable index |
| Replica event loss | Subscriber cannot keep up with event bursts | Remote mirror may become stale | Stale state plus retained capacity | Throughput, reliable delivery/repair |
| Allocator retention | Any large prior live set | Zero is possible | Does not necessarily fall | Cardinality budgets, allocator policy |
| Rejection churn | Very high overload with queue cap | Bounded | Response allocation high-water | Upstream rate limiting |

## Recommended Production Actions

### Immediate

1. Verify every frontend runs the shared library containing `f68be6d`, then
   restart all frontend processes. A Rust-only rebuild is sufficient only if
   no stale wheel or image copy is loaded.
2. Set `router_queue_by_incoming_missing_isl` explicitly. Size it from an
   aggregate memory budget; remember that a per-worker cap is multiplied by
   worker count.
3. Enable `DYN_ADMISSION_CONTROL=token-capacity` where worker capacity
   metadata is trustworthy.
4. Keep HTTP/request-context cancellation tests in release qualification.
5. For low-reuse workloads, disable KV events or use an approximate/bounded
   index until cardinality is controlled.

### Monitoring

Alert on these signals together:

- `router_queue_pending_requests`
- `router_queue_pending_isl_tokens`
- `active_requests`
- queue backpressure/rejection totals
- replica `Subscriber lagged behind` warnings and skipped count
- per-frontend RSS and RSS slope
- active/pending state that remains after traffic drains
- prompt registry, trie node, or index cardinality when those metrics exist

Logical-state metrics distinguish a real cleanup problem from allocator
high-water RSS.

### Follow-Up Engineering

1. Add explicit queue-cap defaults derived from worker and frontend sizing.
2. Add a cardinality budget or approximate mode for KV prompt/index state.
3. Add replica state repair or reliable lifecycle delivery. Consumer batching
   alone cannot recover skipped `Free` events.
4. Consider partial replica views if global load accuracy is not worth the
   `frontends x global state` memory cost.
5. Benchmark allocator alternatives or controlled trimming separately from
   lifecycle correctness.
6. Add a sustained event-rate benchmark that records generated, applied,
   skipped, and repaired lifecycle event counts.

## Limitations

- Mocker workers model router pressure but not every production backend
  timing behavior.
- Fan-out runs had fixed configured concurrency, not fixed achieved request
  throughput. They identify event-rate sensitivity but are not a pure
  frontend-count scaling law.
- Some AIPerf wrapper processes did not exit after their benchmark interval
  and were terminated only by their recorded run-specific PIDs. This can
  affect exact final residual counts.
- The 10-second subscriber pause caused the paused frontend to exit, so only
  event loss, not later convergence, was measured in that direction.
- The campaign accepted current event ordering and loss semantics. It did not
  implement epochs, replay, duplicate repair, or free-before-add buffering.

## Related RCA

- [Memory experiment notes](https://gitlab-master.nvidia.com/jihao/dynamo-writings/-/tree/main/ads/memory?ref_type=heads)
- [KV router unbounded memory experiment](https://gitlab-master.nvidia.com/jihao/dynamo-writings/-/blob/main/ads/memory/KV-ROUTER-UNBOUNDED-MEMORY-EXPERIMENT-2026-06-03.md?ref_type=heads)
- [KV router memory root-cause analysis](https://gitlab-master.nvidia.com/jihao/dynamo-writings/-/blob/main/ads/memory/KV-ROUTER-MEMORY-ROOT-CAUSE-2026-06-03.md?ref_type=heads)

The campaign aligns with the RCA's broad conclusion that overload, global
state replication, event throughput, and high-water allocation compound one
another. It narrows the production picture by separating the now-fixed HTTP
cancellation leak from KV cardinality and from still-lossy replica lifecycle
delivery.
