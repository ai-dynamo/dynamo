# lib/router/admission

`dynamo-admission` is the admission stage: it decides whether a request is admitted, held in a per-class queue, or rejected, and it owns the request-classifier plugin API. Hosts outside `lib/router` import through `host` and classifier plugins through `dynamo_kv_router::plugins::prelude`.

## Policy queue: classes, order, and expiry

- `PolicyQueue` does not classify requests. It owns one flat list of class queues, selected by name (`PolicyProfile::resolve_class_index`; an unknown name is `UnknownPolicyClassError`), and gives each class weighted turns by deficit round robin. `extract_next` returns the winning head with a `DispatchCharge`; `settle` spends that cost only for an admitted request. `pop_next` charges at once, for callers without an admission step.
- Within one strict-priority tier, FCFS/LCFS/WSPT order deadline-bearing entries earliest-due-first ahead of all non-deadline entries before the policy score applies; class due times never enter their order. An EDF class, the policy-class YAML default, ignores strict priority and the score: it orders by the explicit deadline, else router ingress plus the class `slo_ttft_ms`, then enqueue order. Due-time keys (`DueKey`) share one origin fixed at queue construction; a due time before it keys as the origin.
- Controlled delay is a per-class setting (`controlled_delay`, EDF only; configuration rejects it elsewhere). `PolicyQueue::with_class_budget_expiry` fixes each class's expiry at construction: an EDF class with controlled delay expires every finite due time at its lane roots (`expiring_roots`, blocked lanes included); every other class expires only explicit deadlines through its own `due_entries` index. `PolicyQueue::new` (offline replay) expires only explicit deadlines. Any removal of a blocked lane's root, by expiry or cancellation, unblocks the lane.
- Lanes: a request whose classifier set a lane key (`LaneKey`, a per-process keyed hash of a string the classifier chose, ThunderAgent's being the session id) waits in that key's lane; a request without one waits in its pinned or exclusively targeted worker's lane, or in the class's shared heap. Without a classifier the lanes are the worker lanes alone. A lane whose head cannot dispatch is blocked with the reason the dispatch predicate gave (`BlockReason::Capacity`, `SharedCapacity`, `Gate`, or `Fallback`) and leaves the candidate index until a recheck that can free it; the Lanes section below has the rules. The predicate returns a `LaneCheck`; a `bool` converts as capacity-only. `take_expired` pops the overdue prefix from `expiring_roots` in due order, O(log n) per entry; a class without head expiry finds its expired explicit deadlines in `due_entries` with one scan and a heap rebuild.
- The classifier API also carries the dispatch gate (`DispatchGate`, `DispatchHead`, `DispatchDecision`, `DispatchRecheck`): the kernel asks an installed gate about each lane head before its own capacity check, keeps a `Hold` queued in its lane, and re-asks held lanes on the gate's recheck handle or its hold deadline, not on worker frees.
- The queue's indexes are exposed to tests only through the `testing` hooks at the end of `queue.rs`; the profile-driven tests live in `lib/router/kv-router/src/scheduling/policy_queue_tests.rs`.

## Lanes

Each class has one shared heap plus one lane per key: a classifier's lane key, a hard pin, a
ranked soft target, or a worker-only soft target. A request with a lane key always waits in
that key's lane, pinned or not; only a request without one takes a worker lane.

```text
PolicyClassQueue
├── pending                                      sessionless requests with no pin or target
├── ready_by_lane[Keyed(hash)]                   one lane per classifier lane key with queued requests
├── ready_by_lane[Worker(Exact(worker, rank))]   sessionless hard pins
├── ready_by_lane[Worker(Affinity(worker, rank))] sessionless ranked soft affinity
├── ready_by_lane[Worker(WorkerAffinity(worker))] sessionless worker-only soft affinity
├── blocked_lanes: lane -> BlockReason           Capacity, Gate, or Fallback
├── shared_blocked, shared_heads                 lanes refused as SharedCapacity, in head order
├── promoted_shared                              the shared lane standing for the group among the candidates
├── candidate_lane_heads                         unblocked lane roots
├── fallback_heads                               roots whose target is gone; compete every pass
├── recheck: RecheckPlan                         which blocked lanes the next dispatch re-asks
└── expiring_roots                               full lane identity, including blocked roots
```

The scheduler supplies `QueueLane` and answers each head with a `LaneCheck`: `Dispatch`, or
`Blocked(reason)`; a `bool` answer blocks on capacity. A head whose eligible workers are the
class's, with no pin, allow-list, LoRA, required taint, or exclusive target, is refused as
`SharedCapacity`:
every such head gets the same answer, so the group keeps its own order and a capacity recheck
asks only its best head. Refused, nothing else is visited. Allowed, that one lane is promoted to
the candidates as the group's stand-in and the rest stay where they are: one free slot admits
one head. When the stand-in dispatches, the next best is promoted in its place, so n free slots
admit n heads in n passes at one ask each; refused for the shared reason, the slot is gone and
the group waits for the next capacity change; refused for its own reason, it is filed there and
the next best stands in. A worker that frees costs one ask per class, as the shared heap had
it, and never a walk of every session. The admission stage owns no worker
capacity or eligibility state. A blocked lane keeps its reason and leaves the candidate index
until a recheck that can free it: `recheck_worker` re-asks the lanes pinned or bound to that
worker and every keyed lane blocked on capacity; `recheck_affinity_lanes` (the scheduler's
timer) re-asks affinity lanes; `recheck_blocked_lanes(RecheckScope::Capacity)` re-asks every
capacity-blocked lane and `RecheckScope::All` adds the lanes the dispatch gate holds, which
only its wake or hold timer can free. Known fallback heads compete on every dispatch pass. A
hard pin cannot hide a soft request that may fall back. Expiry and cancellation update all
indexes and unblock a removed root's successor. Empty lanes leave the affinity index.
