# KV Routing Locality Metrics — Design Proposal

Status: proposal. No code changes in this PR.

## Summary

When the router selects a worker that holds less KV overlap than another eligible
worker, the blocks that were not reused become extra prefill work. Those blocks are
either recomputed on the selected worker or fetched from the worker that holds them.
Dynamo already has both halves of that story in code — a non-max-overlap observer and
a peer-fetch hint path — but neither is measured in a way that answers the operational
question:

> How much prefill work is currently being recomputed that a peer fetch could have
> recovered instead?

This proposal adds metrics that answer it, splits the lost work into recoverable and
unrecoverable, and fixes an observer dispatch that makes the existing signal unsafe to
run at full request rate.

## Motivation

Two decisions depend on this number:

1. **Is peer KV recovery worth enabling for this deployment?** Today there is no
   measurement that says how much recoverable prefill a fleet is leaving on the floor.
2. **When a deployment is slow, is routing locality a contributor?** During an incident
   an operator needs to know whether prefill is inflated by non-optimal placement, and
   by how much, per request.

Both are diagnostic questions asked on live deployments, not simulation. The intended
operating model is: off by default, enabled on every request while diagnosing, with a
cost low enough that nobody hesitates to turn it on mid-incident.

## Background: what exists today

### The locality signal exists, prefill-only

`NonMaxOverlapSelection` (`scheduling/types.rs`) records that the scheduler chose a
worker with less overlap than the best eligible one, carrying the selected and highest
overlap in effective blocks. `non_max_overlap_selection()` (`scheduling/queue.rs`)
computes it, skipping pinned requests and ineligible ranks. Two metrics are exported,
`router_non_max_overlap_selections_total` and `router_overlap_blocks_lost`.

The observer is installed only for `WORKER_TYPE_PREFILL`
(`lib/llm/src/kv_router/scheduler.rs`). Aggregated deployments and conditional-disagg
decode — which retains a positive overlap credit — produce no signal at all.

### The recovery path exists, and is uninstrumented

`RouterHint` (`router_hint.rs`) hands the selected worker a peer's
`source_control_endpoint` plus the block hashes that peer holds, and
`RouterHintRootCandidates::best_source(prefix_blocks_to_beat, ..)` only returns a peer
holding **more** prefix blocks than the target already has. That delta is exactly the
recoverable quantity.

There are no metrics on this path — no counter for hints issued, no measure of blocks
recovered. Whether peer recovery is firing, and what it is worth, is currently
unobservable.

### What is missing

- No normalization. Absolute blocks lost cannot be compared across workloads, and
  histograms cannot be divided after the fact.
- No recoverability split. Lost blocks that a peer could serve and lost blocks that
  must be recomputed are counted identically, though only the first is addressable.
- No per-request detail. Aggregates cannot identify which prefix or which worker pair
  is responsible.

## Proposal

### The core quantity

For an admitted request that selected a non-maximal-overlap worker:

```text
lost_blocks   = highest_overlap_blocks - selected_overlap_blocks
novel_blocks  = required_blocks - highest_overlap_blocks
regret_ratio  = lost_blocks / (novel_blocks + lost_blocks)
```

`novel_blocks` is prefill no placement could have avoided. `regret_ratio` is therefore
**the fraction of the prefill actually paid that was avoidable**, bounded in `(0, 1]`.

The unbounded alternative `lost / novel` is rejected: it diverges exactly when the
prompt was fully cached somewhere else, which is the worst case and the one that must
not fall off the end of a histogram.

### The recoverability ladder

Lost blocks are classified by where they can be recovered from, because the rungs have
different costs and different fixes:

| Class | Source | Cost model |
|---|---|---|
| `peer` | `RouterHintCandidateSource::Worker` | fabric transfer |
| `cache_owner` | `RouterHintCandidateSource::CacheOwner`, KVBM lower tier | PCIe / SSD |
| `shared` | `SharedCacheHits` (for example SGLang HiCache) | shared-store read |
| `none` | nowhere | recompute, unavoidable |

Collapsing these into one "recoverable" number would hide the case where the shared
store already covers the loss and peer transfer would add little.

### Metric set

| Metric | Type | Labels |
|---|---|---|
| `router_recoverable_prefill_blocks_total` | counter | `worker_type`, `source_class` |
| `router_unrecoverable_prefill_blocks_total` | counter | `worker_type` |
| `router_overlap_regret_ratio` | histogram | `worker_type` |
| `router_hints_issued_total` | counter | `worker_type`, `source_class` |
| `router_hint_prefix_blocks_recovered` | histogram | `worker_type` |

Counters, not only histograms: the investment case is a rate (recoverable prefill
blocks per second), which converts to GPU-seconds and to hardware. Histograms answer
distribution questions; they do not sum.

Where a calibrated `PrefillLoadEstimator` is configured, the same quantity is also
emitted in seconds. Seconds are what make the case; blocks are what make it arguable.

## Design details

### 1. Replace the per-event `spawn_blocking` dispatch

`dispatch_non_max_overlap_selection` currently spawns a blocking task per observed
selection:

```rust
let _observer_task = tokio::task::spawn_blocking(move || {
    observer(&request_id, selection);
});
```

The motivation is sound: `book_and_respond` runs inside the single scheduler actor
(`tokio::spawn(actor.run(admission_rx))`), so every routing decision in the process is
serialized through that task, and `NonMaxOverlapSelectionObserver` is an
`Arc<dyn Fn(..)>` with no contract forbidding blocking. Running an arbitrary closure
inline could stall all routing.

The lever is wrong for the cost, and worse under the intended always-on use:

- A thread hop per event to perform two atomic increments.
- No bound on fan-out. When the blocking pool saturates, tasks queue without limit;
  memory and observer latency both grow unbounded under a burst, with no shedding.
- The `JoinHandle` is dropped, so a panicking observer fails silently.
- Completion order is arbitrary, which is harmless for counters and wrong for a
  per-request diagnostic trace.

Proposed split by cost class:

- **Counters inline.** Prometheus counter and histogram updates are atomics. Call them
  directly from the actor and drop the indirection. The `Arc<dyn Fn>` hook buys nothing
  when there is one implementation in the same workspace.
- **Records over a bounded channel.** For per-request diagnostics, `try_send` a `Copy`
  record to a single writer task; on `Full`, increment a dropped-records counter and
  continue. The scheduler then cannot be stalled by the sink, which is the property
  required of something enabled during an incident.

This matches the existing idiom in this repository —
`ActiveSequenceEventPublisher` (`lib/llm/src/kv_router/sequence.rs`) and
`routing_load.rs` both use bounded `mpsc` plus `try_send` for exactly this problem.

If an extension point is still wanted for out-of-tree consumers, the contract belongs
in the type (a narrow trait documented as non-blocking, or handing the consumer the
`Receiver`) rather than being defended at the call site by every deployment.

Records must stay `Copy`. The existing `request_id: String` clone is already paid on
the booking path for its error branch, so it is not new cost today — but a `String` per
record in a channel at full request rate would be. Use a `u64` sequence number, or
`Arc<str>` where the real ID is required.

### 2. Where recoverability classification lives

`router_hint_metadata_for_dp_rank()` is defined on `ModelRuntimeConfig` in `lib/llm`,
while the scheduler actor lives in `lib/kv-router` and is generic over
`C: WorkerConfigLike`. Two options:

1. Extend `WorkerConfigLike` with a hint-capability accessor, moving classification
   into the actor.
2. Keep classification in the `lib/llm` observer, which already holds the
   `RuntimeConfigWatch`.

**Option 2 is proposed.** It respects the crate boundary and avoids widening a trait
that the scheduling module treats as a versioned contract. Revisit only if
classification is later needed inside selection.

Either way the lookup must take a short synchronous borrow of `workers_with_configs`
and must not be held across an `.await`, per the scheduling module guardrails.

### 3. Peer recoverability is cheap; the lower rungs are not

The `peer` rung needs no new indexer work. `NonMaxOverlapSelection.highest_overlap_worker`
already names the worker holding the lost blocks; asking whether it could serve them is
the same check `router_hint_for_selection` performs — a `workers_with_configs` lookup,
`router_hint_metadata_for_dp_rank()`, and a worker-type match. That is a hash lookup and
two string comparisons, only on requests that actually lost overlap, and it works on
every indexer topology.

The `cache_owner` rung requires the hint candidate chain, which is gated:

```rust
// lib/llm/src/kv_router/indexer/mod.rs
matches!(self, Self::KvIndexer { approx: None, primary_records_routing_decisions: false, .. }
            | Self::Concurrent { approx: None, primary_records_routing_decisions: false, .. })
```

No approximate LRU side indexer, no route recording, and not the remote indexer. Large
fleets are the most likely to run those configurations, so the lower rungs may be
unavailable precisely where the diagnostic is most wanted. This is a documented
limitation, not something to paper over.

Candidate collection is additionally gated on `has_router_hint_capable_workers()`, so
with peer recovery disabled the chain is never built. Measuring the counterfactual
requires decoupling retention from that capability check behind an explicit flag.

### 4. Ungate decode

Remove the `WORKER_TYPE_PREFILL` condition on observer installation. Aggregated
deployments and conditional-disagg decode retain positive overlap credit and currently
emit nothing.

### 5. Extend `NonMaxOverlapSelection`

Add `required_blocks` so `regret_ratio` is computable at the decision site, plus the
recoverability classification. The struct stays `Copy`.

## Non-goals

- **No load-side or logit signal is exposed.** The scheduling module treats selector
  values as a versioned external API and prohibits exposing intermediates whose meaning
  depends on the default policy. A logit delta is exactly such an intermediate. This
  means these metrics deliberately do not say whether a given sacrifice was a *good*
  trade — only what it cost in locality. That is a real limitation and is accepted here
  to keep the public selector contract unchanged.
- No change to routing behavior. Observation only.
- No change to replay parity semantics.

## Limitations

- **Predicted, not realized.** Overlap comes from an eventually-consistent indexer.
  Blocks counted as lost may already have been evicted from the peer. A
  predicted-versus-realized calibration metric is future work and bounds how much the
  number should be trusted.
- **Recoverable is not free.** Peer recovery converts compute into transfer. Net value
  is `prefill_time(lost) - transfer_time(lost) - handshake`, and below some block count
  recovery is a pessimization. Reporting gross recoverable blocks overstates the case;
  this is why seconds are emitted where an estimator is available.
- **Directional bias.** The figure is an upper bound on direct savings — transfer costs
  real bandwidth and the interconnect saturates — and a lower bound on total benefit,
  since with recovery cheap the router could rationally trade more locality for load
  balance, a second-order gain not captured by replaying current decisions. It is
  *gross recoverable prefill under current policy*, not a performance projection.
- **Multi-turn amortization.** Routing to a worker creates the prefix there, so a
  session pays the cost once and amortizes it across turns, while a per-request
  histogram counts it per request. The fleet-wide counters are the honest view for
  capacity questions.

## Phasing

1. **Observer dispatch cleanup.** Inline counters, remove `spawn_blocking`, keep
   behavior otherwise identical. Independently reviewable, and a strict improvement
   even if nothing below lands.
2. **Core metrics.** Ungate decode, extend `NonMaxOverlapSelection`, add the regret
   ratio, recoverable/unrecoverable counters, and peer classification. No new
   configuration.
3. **Diagnostic records.** Per-request records behind a flag, bounded sink, drop
   accounting.
4. **Lower rungs.** Shadow chain retention for `cache_owner`, with the topology
   limitation documented.

## Testing and benchmarks

- Unit coverage for `regret_ratio` boundaries: fully cached elsewhere, zero novel
  blocks, single eligible worker, pinned requests (excluded), and ties.
- Classification tests per ladder rung, including a peer that holds more blocks but is
  hint-ineligible.
- A drop-path test asserting the bounded sink sheds rather than blocking, and that the
  dropped counter advances.
- Because this touches the admission path, before/after runs of
  `lib/kv-router/benches/worker_selection.rs` are included, per the scheduling module
  guardrails, with the diagnostic both off and on.

## Alternatives considered

- **Sampling.** Rejected: the diagnostic use case wants every request, and after the
  dispatch fix the per-event cost no longer justifies sampling.
- **Extending `PlacementCacheSample` in `aisimulate-core`.** That record carries only
  the selected worker's overlap and ISL, and the crate is external and exact-pinned
  (`=0.1.0-dev.1`). A cross-repo schema change is not warranted for a live diagnostic.
- **Keeping `spawn_blocking` and shrinking the payload.** Does not address unbounded
  fan-out, silent panics, or ordering, and leaves a thread hop per request.
- **Prometheus only, no per-request records.** Aggregates cannot identify the
  responsible prefix or worker pair, which is the primary diagnostic need.
