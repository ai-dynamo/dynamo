# lib/kv-router/src/scheduling

Scheduling owns request admission, queueing, worker selection, and the handoff
from projected load into sequence-state booking.

## Queue hierarchy

There is one `SchedulerQueueActor` per scheduler. It owns one outer `PolicyQueue`, which owns one `PolicyClassQueue` for every class in the resolved profile.

```mermaid
sequenceDiagram
    participant H as SchedulerQueue
    participant A as SchedulerQueueActor
    box PolicyQueue owns all class queues
        participant P as PolicyQueue
        participant C as PolicyClassQueue instances
    end
    participant S as WorkerSelector

    H->>A: Enqueue(request)
    A->>A: Resolve class_index
    alt Immediate path
        A->>S: select_worker(request)
        S-->>A: Selected worker
        A->>A: Book state and respond
    else Queued path
        A->>P: enqueue(class_index, request)
        P->>C: Push into the selected class queue
        H->>A: Update after capacity changes
        A->>P: pop_next()
        P->>C: Get one candidate per dispatchable class
        C-->>P: Dispatch candidate
        P-->>A: DRR winner
        A->>S: select_worker(request)
        S-->>A: Selected worker
        A->>A: Book state and respond
    end
```

- `SchedulerQueue` is the public handle that sends commands to the actor.
- `SchedulerQueueActor` classifies each request and chooses the immediate or queued path.
- `PolicyQueue` does not classify requests. It owns all class queues and runs DRR across their candidates.
- Each `PolicyClassQueue` owns ordering and accounting for one class such as `latency`, `agents`, or `batch`.
- `SchedulerQueueActor::admit_one` performs final worker selection and booking after either path.
- A single-class profile still uses `PolicyQueue`, but DRR has no cross-class effect.

## Per-class ready storage

`PolicyQueueEntry<T>` is one queued payload plus its class index, priority key, enqueue sequence, and token/accounting snapshot. In production, `T` is `QueuedRequest`, which wraps the `SchedulingRequest`, enqueue timestamp, and optional block hashes.

```text
PolicyClassQueue("agents")
├── pending: BinaryHeap<PolicyQueueEntry>                 # WorkerPlacement::Any
├── ready_by_worker: FxHashMap<WorkerWithDpRank, BinaryHeap<PolicyQueueEntry>>
│   ├── Worker(7, dp_rank=0) → heap of requests pinned to that rank
│   └── Worker(9, dp_rank=1) → heap of requests pinned to that rank
├── blocked_workers: FxHashSet<WorkerWithDpRank>
└── candidate_worker_heads: BTreeSet<WorkerLaneHead>       # one head per unblocked lane
```

- `pending` is the shared ready heap for requests where the existing selector may choose any eligible worker.
- `ready_by_worker` is sparse. A worker/rank heap is created only while exact-placement requests exist for it and removed when empty.
- `candidate_worker_heads` indexes one request head per worker lane by the same queue priority. A blocked head is removed until that worker's next capacity update; periodic or topology-wide updates recheck all blocked lanes.
- A `PolicyClassQueue` does not own worker configuration, capacity, or scoring state. `WorkerWithDpRank` is only the exact-placement lane key; the actor and selector retain worker knowledge.
- Every heap uses the class's configured priority ordering. `BinaryHeap::peek()` reads its highest-priority root in O(1); push and pop are O(log n).
- `PolicyClassQueue::next_dispatchable` compares the shared root with the highest indexed worker head. It removes blocked worker heads until it finds a dispatchable one, so each blocked lane is checked once per capacity update rather than once per pop.
- `round_cursor` marks where the next quantum-granting DRR pass begins. `carry_class` lets one class spend residual deficit before that pass; it is revalidated through `next_dispatchable` and never grants another quantum when the carried candidate is blocked or too expensive.
- Worker lanes contain head-of-line blocking to one exact worker. A blocked Worker 7 root cannot hide a ready Worker 9 root or an unpinned root.

## Policy-class admission lifecycle

`SchedulerQueueActor` owns admission state, physical queue accounting, worker booking, and lifecycle delivery. The request path owns only the progress updater and one `AdmissionLease` after selection.

```mermaid
sequenceDiagram
    participant C as Lifecycle owner
    participant H as LocalScheduler / SchedulerQueue
    participant A as SchedulerQueueActor
    participant G as Admission strategy
    participant Q as PolicyQueue
    participant S as Worker slots
    participant R as RequestGuard

    C->>H: schedule(TrackedWithAdmission)
    H->>A: EnqueueLeased(request, lease)
    A->>G: admit(request facts + live progress)
    alt Bypass
        A->>A: Continue without strategy lifecycle
    else Ready(Any or Exact)
        A->>Q: Enter the physical ready class/lane if capacity blocks
    else Defer
        A->>Q: Retain as strategy-deferred
        G-->>A: MakeReady(id, placement)
        A->>Q: Reclassify/migrate into the physical ready class/lane
    end
    Note over H,A: The remaining flow applies only to Ready/Defer, while Bypass disarms the lease
    par Lease handoff after ownership
        A->>A: Arm lease only after generation-scoped ownership exists
        A-->>H: Return armed lease acknowledgement
    and Selection when ready
        Note over A,S: Immediate, or after deferred/blocked work becomes ready
        A->>S: Select worker and book request state
        A-->>H: SchedulingResponse(progress)
    end
    H->>H: Attach the armed lease
    H-->>C: Selected worker + lifecycle capabilities
    C->>R: Move progress and lease before next fallible await
    R->>R: Backend dispatch succeeds
    R->>R: AdmissionLease::mark_dispatched fallback
    R->>H: LocalScheduler::mark_dispatched
    H->>A: Bounded Dispatched command
    A->>G: AdmissionEvent::Dispatched
    loop Response items
        R->>R: Publish monotonic context progress
    end
    alt Successful terminal item
        R->>R: Mark Completed before yielding
        R-->>C: Yield Stop / EoS / Length
    else Natural EOF
        R->>R: Mark Completed
    else Cancellation or error
        R->>R: Mark Aborted
    end
    Note over C,R: Drop after the terminal yield and normal resume converge on the same Completed lease drop
    R->>A: Lease drop via SegQueue + coalesced bounded wake
    A->>S: Release worker booking
    A->>G: Completed or Aborted
```

- `EnqueueLeased` arms the lease only after actor ownership exists. Its lifecycle generation prevents stale cleanup from touching a reused request ID.
- Lease drops use the lock-free `SegQueue` and coalesced bounded actor wakeups. Cleanup drains to quiescence so no wake is lost; do not replace it with an unbounded channel or task-per-drop cleanup. Any bounded/interleaved drain must preserve wake correctness and be benchmarked.
- Strategy ownership and physical queue class are separate. Exact placement may migrate queue storage/accounting while callbacks remain with the strategy that admitted the request.
- Queue limits gate new arrivals; they are not strict occupancy invariants after reclassification or worker-count changes. Strict migration limits require a rejected-`MakeReady` contract.
- Dispatch remains a two-step protocol: `AdmissionLease::mark_dispatched()` records the cancellation fallback before `LocalScheduler::mark_dispatched(request_id)` sends the live actor event. If that send is canceled, terminal cleanup emits `Dispatched` before the terminal event.

## Guardrails

- A request ID identifies at most one active scheduler request. Do not reuse it until the prior request reaches terminal cleanup; cancellation and admission lifecycle state intentionally key by request ID.
- `SchedulerQueueActor::admit_one` is the canonical admission path: compute projected
  load, select worker, skip booking if the response receiver is closed, then
  book state before responding. Failed response delivery must roll back the
  booking. Do not bypass this for normal scheduling.
- Do not remove or weaken `admission_gate` without proving selection plus
  booking remains serialized enough to avoid oversubscription regressions.
- Potential-load projection must go through
  `ActiveSequencesMultiWorker::potential_blocks_and_tokens_at(...)` with
  `SchedulingRequest::prefill_token_deltas()`. Do not scan per-worker
  `ActiveSequences` directly from scheduling.
- `SchedulingRequest` helper methods are the canonical source for effective
  cached tokens, effective overlap, worker allowance, prefill-token defaults,
  and request block count. Do not duplicate this logic in policies or selectors.
- WSPT must remain prefill-hint aware: pinned requests use the pinned worker's
  effective cached tokens; unpinned requests use the best allowed worker. Do not
  silently fall back to raw ISL unless tracking is disabled or cache data is
  absent.
- Pinned-worker and allowed-worker constraints must be validated before
  selection and respected by queue capacity checks, selector candidate
  iteration, and WSPT priority.
- Prefill load hints are computed at scheduler/request boundaries from
  selected-worker `cached_tokens`. Do not move ISL/cache-token math back into
  `ActiveSequences`.
- Selectors should be side-effect free: no booking, no queue mutation, and no
  `PromptRegistry` mutation.
- Do not hold the pending-heap lock while selecting, calling slot state,
  responding, or awaiting. The queue heap is only for parked requests.
- Do not hold `workers_with_configs.borrow()` across `.await`; take a short
  synchronous snapshot or borrow for selection only.
- Any change to queue ordering, WSPT keys, capacity checks, admission
  serialization, or selector scoring should include focused tests and
  before/after routing or queue benchmarks.
- Keep text and external IDs such as request IDs on standard hash collections.
  Use `FxHashMap` / `FxHashSet` for internal numeric hot-path keys only.
