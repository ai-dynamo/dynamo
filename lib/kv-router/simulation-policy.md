<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native routing policy in a simulation host

An embedded Rust host can call Dynamo's native worker selector and own the
session-affinity lifecycle. `DefaultWorkerSelector` and `WorkerSelector` are
already public. The manual-clock constructor and shared sibling-group key
helper extend these APIs for hosts that advance virtual time.

This interface is independent of AISimulate. Dynamo's published AISimulate
`0.12.0` dependencies remain unchanged. It does not add an AISimulate routing
adapter or enable conversation routing in the current CLI. A downstream CLI
needs a separate integration, an installable dependency pinned to an actual
merged Dynamo commit, and an end-to-end test of that consumer.

## Build a native consumer

The manual-clock and group-key additions are unreleased. Before merge, validate
against the full immutable SHA of the reviewed PR head. After merge, update to
the actual merged Dynamo commit and revalidate. Replace the placeholder below;
it is not a branch name or a published version:

```toml
[package]
name = "native-router-host"
version = "0.1.0"
edition = "2024"

[dependencies]
dynamo-kv-router = { git = "https://github.com/ai-dynamo/dynamo.git", rev = "<FULL_40_CHARACTER_DYNAMO_COMMIT>", features = ["standalone-selection"] }
tokio = { version = "1", features = ["time"] }
```

Use the pinned repository's Rust toolchain and native build prerequisites from
the [contribution guide](../../docs/fern/pages/community/contributing/overview.md).
Commit the consumer's generated `Cargo.lock`, then use `cargo run --locked`.
The `standalone-selection` feature exposes
`dynamo_kv_router::services::selection::affinity`; using these Rust types does
not start the standalone HTTP service. The manual-clock example below does not
require a Tokio runtime.

## Selection and affinity have separate responsibilities

The selector evaluates current worker configuration, eligibility, cache overlap,
and load. `SessionAffinity` owns session bindings, tentative initialization,
active leases, and idle TTL. The host supplies those inputs and manages the
actual dispatch lifecycle. `RoutingEligibility::with_eligible_affinity_target`
exposes the same eligibility narrowing used by Dynamo's scheduler queue:

1. Advance the affinity table to the event's virtual timestamp, then call
   `try_acquire(key, requested_target)`. `AcquireStep::Wait` means another request
   is initializing that key: retain the notification, process other events,
   then retry when notified. An offline event loop must not block waiting for
   the very dispatch event that it needs to process.
2. Keep the returned `Hold` while selecting. Forward `hold.target()` into the
   scheduling request. An existing binding constrains the default selector to
   its eligible target; a fresh key permits normal selection.
3. Commit the hold with the worker and DP rank that actually accepted dispatch.
   A selection result alone does not establish ownership. If dispatch fails,
   drop the tentative hold; do not commit an unaccepted selection.
4. Retain the resulting `AffinityLease` until the host's request ownership ends.
   Advance virtual time before releasing the lease. Dropping the final lease
   starts the idle TTL from that virtual release time.

This selection helper uses only public Dynamo types. The caller provides a
`SchedulingRequest` populated with current overlap and load information:

```rust
use std::collections::HashMap;
use dynamo_kv_router::{
    DefaultWorkerSelector, KvSchedulerError, SchedulingRequest,
    WorkerConfigLike, WorkerId, WorkerSelectionInput, WorkerSelector,
};
use dynamo_kv_router::protocols::WorkerSelectionResult;
use dynamo_kv_router::services::selection::affinity::Hold;

fn select_for_held_session<C: WorkerConfigLike>(
    selector: &DefaultWorkerSelector,
    workers: &HashMap<WorkerId, C>,
    request: &mut SchedulingRequest,
    hold: &Hold,
    block_size: u32,
) -> Result<WorkerSelectionResult, KvSchedulerError> {
    request.affinity_target = hold.target();
    let mut eligibility = request.eligibility();
    if <DefaultWorkerSelector as WorkerSelector<C>>::uses_exclusive_affinity_target(selector)
        && let Some(target) = request.affinity_target
    {
        eligibility = eligibility.with_eligible_affinity_target(workers, target);
    }
    selector.select_worker(WorkerSelectionInput::configured(
        workers, request, eligibility, block_size,
    ))
}
```

The host still owns worker availability, load accounting, dispatch rejection,
completion, cancellation, and any wait queues. For disaggregated serving, use
separate prefill and decode tables and selector inputs: a binding in one pool
does not identify a worker in the other pool.

An ineligible affinity target does not become a hard pin: the helper preserves
the existing eligible candidate set when the target is absent, overloaded, or
excluded by routing constraints. Supply the host's overload and availability
sets through `eligibility_with_overloaded` and `with_available_workers` when
applicable. The acceptance host must handle the resulting selection against its
affinity enforcement policy, including invalidation and retry when a bound
worker disappears. Call `Hold::invalidate` or `AffinityLease::invalidate` to
remove a stale binding. Do not turn a failed hard-affinity commit into success.

## Grouping strategy and enforcement are different settings

| Choice | Meaning |
| --- | --- |
| Session grouping | Use the conversation's session ID as the affinity key. |
| Sibling-group grouping | For a child, use `subagent_group_affinity_id(parent_session_id)`. Siblings share that key; the parent retains its own session key. |
| `SessionAffinityMode::Hard` | A dispatch that violates the bound worker/rank returns an error and invalidates the binding. This is the native default. |
| `SessionAffinityMode::Soft` | The binding follows an accepted dispatch to another target. The host still chooses how selection treats affinity. |

Proposed downstream values such as `affinity.mode: session` or `sibling_group`
describe the grouping strategy, not the native `Hard`/`Soft` enforcement mode.
Those downstream configuration values are planned consumer integration, not
configuration accepted by the current CLI. Setting native `Soft` mode alone
does not change the default selector's exclusive affinity-target behavior.

The group-key helper hashes the immediate parent session ID into Dynamo's
internal key namespace. It does not derive conversation ancestry or alter
request IDs. The host must supply unambiguous session/parent IDs and keep their
namespaces distinct across unrelated runs.

## Run the native affinity lifecycle

Place this example in the consumer's `src/main.rs`. It demonstrates one child
dispatch accepted by worker 7, DP rank 0, followed by release and virtual expiry:

```rust
use std::time::Duration;
use dynamo_kv_router::services::selection::affinity::{
    AcquireStep, AffinityTarget, SessionAffinity, SessionAffinityConfig,
    subagent_group_affinity_id,
};
use tokio::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let epoch = Instant::now();
    let ttl = Duration::from_secs(10);
    let table = SessionAffinity::with_manual_clock(
        SessionAffinityConfig::new(ttl), epoch,
    )?;
    let key = subagent_group_affinity_id("parent-session");
    let AcquireStep::Held(hold) = table.try_acquire(&key, None)? else {
        panic!("a fresh key must not wait");
    };
    assert!(hold.target().is_none());

    // In a routing host, select first and commit only after dispatch succeeds.
    let dispatched = AffinityTarget::new(7, Some(0));
    let lease = table.commit(hold, dispatched)?;
    table.advance_clock(epoch + Duration::from_secs(20))?;
    assert_eq!(table.query_target(&key, None)?, Some(dispatched));
    assert_eq!(table.query_target("parent-session", None)?, None);

    // Active work survives TTL. Idle expiry starts when the last lease ends.
    drop(lease);
    table.advance_clock(epoch + Duration::from_secs(29))?;
    assert_eq!(table.query_target(&key, None)?, Some(dispatched));
    table.advance_clock(epoch + Duration::from_secs(30))?;
    assert_eq!(table.query_target(&key, None)?, None);
    assert!(table.advance_clock(epoch).is_err());
    Ok(())
}
```

Manual time is monotonic: advancing backwards returns an error. No wall-clock
reaper is spawned for a manual-clock table; advancing time removes idle expired
bindings while retaining active leases. Use one consistent event order and
advance before acquisition, commit, and release. Ordinary `with_config` tables
continue to use their runtime clock.
