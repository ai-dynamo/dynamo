<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rush Hour interactive replay contract

## Supported kernel

`ReplaySession` is an in-memory, incrementally driven wrapper around Dynamo's
ordinary aggregated offline replay runtime. The supported Wave 2 slice is:

- aggregated vLLM with one DP rank per replica;
- round-robin routing and homogeneous replicas;
- trace-mode admissions with explicit UUIDs and exact planned output token IDs;
- existing Dynamo `Trace` and Mooncake JSONL loading for single-turn sessions;
- prefix caching, including active/inactive ownership and future eviction;
- configurable initial replica count and absolute scale targets;
- replica startup and drain lifecycle transitions; and
- arbitrary busy pause points in this slice.

The public handle is `Send`, while the non-`Send` mocker runtime remains
confined to a dedicated owner thread. Commands are synchronous and
transactional: a failed advance or mutation restores the pre-command deep
memento before returning the error.

Long seeks should use
`advance_sampled(until_ms, telemetry_interval_ms, checkpoint_interval_ms)`.
It executes all absolute-grid sub-boundaries under one rollback transaction,
returns window telemetry on the telemetry cadence, and retains deep mementos
only on the checkpoint cadence. This avoids constructing a rollback memento
for every one-second UI sample. The final target always has telemetry, and the
combined observation count is bounded at 100,000.

Retained-future playback that needs only the selected window can use
`advance_to_with_telemetry(until_ms, window_start_ms)`. The cursor movement and
telemetry capture share one rollback transaction, so callers do not need a
second outer checkpoint to preserve atomic-on-error behavior.

At a pause at time `T`, pre-existing completion, worker-ready, arrival, and
retirement events at `T` settle, but no new engine pass starts. A scale action
therefore runs after ordinary events at `T`; work resumes only on the next
`advance_to` call. Scale targets are absolute. Retargeting while a startup or
drain transition is still in progress is rejected in this bounded API.
Interactive stepping continues to consume pending worker-ready and drain
events through the requested cursor even after request-bearing work is done;
consumers should distinguish request workload completion from full replica
lifecycle quiescence.

## Deep checkpoint and restore

`ReplaySession::checkpoint` captures:

1. the clock, event sequence, and ordered future event heap;
2. pending immutable trace admissions plus their timeline-local cursor;
3. active request phase and scheduler waiting/running state;
4. every active sequence, including generated/planned tokens and allocation
   cursor;
5. logical full-block refcounts, active partial identities, registered prefix
   metadata, and inactive eviction order;
6. worker topology, busy/in-flight accounting, startup/drain sets, and stable
   replica IDs;
7. collector, traffic, and runtime accounting; and
8. the applied scale-action revision.

Restore never replays from time zero. It constructs fresh scheduler cores and
fresh physical KVBM block pools, reserves active slots before rebuilding the
inactive cache, re-registers full blocks, and recreates logical ownership. No
live `BlockManager`, `MutableBlock`, or `ImmutableBlock` handle crosses a
timeline boundary. Tests compare source/restored physical manager IDs, busy
continuations, scale-up startup, scale-down drain, final reports, and a
capacity-constrained continuation whose future behavior depends on eviction
order.

Restoring a checkpoint creates an independent branch. Rush Hour owns the
policy that a mutation truncates its stored future checkpoints and telemetry;
the Dynamo kernel does not retain or merge an old future.

## Checkpoint retention cost

Immutable high-volume payloads are structurally shared:

- raw future requests use `Arc<[DirectRequest]> + next_index`, so the future
  trace is O(1) per checkpoint (ordinary non-interactive replay keeps its
  streaming `VecDeque` behavior and releases consumed requests);
- registered KV block token payloads are shared `Arc<[u32]>`; and
- collector token-timestamp histories are copy-on-write and shared after a
  request becomes immutable.

Timeline-local ownership remains value-owned. Each checkpoint is therefore
O(active scheduler state + resident block metadata + collector request
metadata + pending events), not O(total future trace token volume). Retaining
hundreds of checkpoints can still consume substantial memory because resident
block and per-request metadata are copied at every checkpoint. Wave 2 should
use bounded checkpoint retention. Durable serialization, delta checkpoints,
and an unbounded checkpoint archive are not part of this kernel.

For a local debug-build smoke test, the 23,608-row one-hour Mooncake trace
loaded in about 3.1 seconds. Advancing four replicas directly to minute 23 took
about 10.9 seconds; doing the same seek with one-second telemetry and ten-second
checkpoints took about 17.2 seconds and returned 1,380 observations plus 138
deep checkpoints. A busy checkpoint there took about 50 milliseconds and
restore took about 0.31 seconds. These numbers are evidence of practicality on
the development machine, not a stable performance guarantee.

## Telemetry contract

`telemetry_since(window_start_ms)` is non-destructive and returns:

- current cursor, traffic-arrival horizon, optional known terminal horizon,
  and workload completion;
- current queue/running counts and active/provisioned topology;
- cumulative arrivals, admissions, completions, and visible output tokens;
- true window counts/rates and window p95 TTFT, ITL, and E2E;
- window and cumulative prefix-KV reuse; and
- stable per-replica lifecycle, queue/running/in-flight counts, and the same
  request-derived window metrics where attribution exists.

Scheduler passes can calculate a modeled completion time before the event is
visible. Telemetry filters token and latency samples against the paused cursor,
so it never leaks such future values. Percentiles are optional when a window
has no samples. Speculative decoding is unsupported; accepted tokens per
decode forward is therefore exactly `1.0` once the window contains a visible
non-first/decode token and otherwise unavailable.

## Explicitly unsupported

The constructor returns an error rather than weakening checkpoint semantics
for:

- multi-turn `WorkloadDriver` sessions;
- KV-router or planner-hook state;
- disaggregated replay;
- DP greater than one;
- speculative decoding;
- stateful AI Configurator performance-model callbacks;
- KVBM offload tiers;
- SGLang or TRT-LLM scheduler modes; or
- requests without deterministic UUID/output-token plans.

The bounded limits are 100,000 requests, 4,096 scale actions, and 128 replicas.
Mooncake loading currently materializes each single-turn request's input token
vector, so very large traces still have a one-time construction-memory cost.

The older `checkpoint_quiescent` proof remains for its original narrow test
surface. Rush Hour should use `ReplaySession::checkpoint`, whose kind is
`DeepRuntimeMemento`.
