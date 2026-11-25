# lib/router/admission

`dynamo-admission` is the admission stage: it decides whether a request is admitted, held in a per-class queue, or rejected, and it owns the request-classifier plugin API. Hosts outside `lib/router` import through `host` and classifier plugins through `dynamo_kv_router::plugins::request_classifier`. A classifier reads workers through `dynamo_worker_selection::WorkerCandidates`, the view a filter or scorer reads; do not add a second row type here, so that a value added to a worker input reaches both plugin kinds. A method only the kernel or a host calls on a classifier contract type (building a `ClassifyRequest`, reading back what the classifier set) goes on the sealed `host::ClassifyRequestHostExt` trait, and creating a `RequestProgress` is the function `host::request_progress`, not an inherent method; a plugin does not name `host`, so do not re-export it from a plugin module. `RequestClassifierContext::new` stays inherent and public: a classifier's own tests and benches build the context its factory receives, so a plugin calls it too.

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
├── blocked_lanes: lane -> BlockReason           Capacity or Fallback
├── gate_held                                    lanes the dispatch gate holds; only a full recheck asks them
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
the next best stands in. A full recheck asks every lane of the group, since the gate answers
per head, and moves only the lanes whose answer changed: when all that the gate lets go may
go, they stay and the best of them is promoted in the same way. A worker that frees costs one
ask per class, as the shared heap had
it, and never a walk of every session: a capacity recheck walks only `blocked_lanes`, the lanes
blocked on their own capacity or fallback, not the shared group and not the gate's holds. The
admission stage owns no worker
capacity or eligibility state. A blocked lane keeps its reason and leaves the candidate index
until a recheck that can free it: `recheck_worker` re-asks the lanes pinned or bound to that
worker and every keyed lane blocked on capacity; `recheck_affinity_lanes` (the scheduler's
timer) re-asks affinity lanes; `recheck_blocked_lanes(RecheckScope::Capacity)` re-asks every
capacity-blocked lane and `RecheckScope::All` adds the lanes the dispatch gate holds, which
only its wake or hold timer can free. Known fallback heads compete on every dispatch pass. A
hard pin cannot hide a soft request that may fall back. Expiry and cancellation update all
indexes and unblock a removed root's successor. Empty lanes leave the affinity index.

## Performance

The queue runs on the scheduler's actor thread, once per arrival and once per completion. `lib/router/scheduler/AGENTS.md` has the rules for the actor; these are the queue's.

- **Adding a block reason or an index over lanes.** A worker free runs a capacity recheck in every class with a blocked lane, and it must not visit a lane the change cannot free. The shared-capacity group answers with one head and `gate_held` is visited only on a full recheck. Keep a new kind of blocked lane out of the `blocked_lanes` walk unless a capacity change can free it: when gate-held lanes sat in that walk, 10,000 of them cost 20 to 40 µs on every free. Keyed lanes whose head has its own constraint (an allow-list, a LoRA, a required taint) still break the rule: they are blocked as `Capacity`, and `RecheckPlan::covers` re-asks every keyed lane when any worker frees, whichever workers the lane can run on. `admission_core`'s `constrained_lanes` case measures it at 96 µs per free with 1,000 such lanes and 1.8 ms with 10,000. Matching them to the freed worker needs the scheduler to name the ranks a refused head waits on.
- **Editing the dispatch loop.** Measure every edit to `next_dispatchable_lane`, `pop_lane`, and `push_ready`, a removal or a reorder included. Reading the lane head's cost through the reference the dispatch check already held, in place of a second lookup, made `drain_one_per_exact_lane` 20% to 27% slower, and so did two other equivalent forms. The `head_expiry` early-outs look redundant and are worth 2% to 5% on lane drains.
- **Growing a queued entry.** `PolicyQueueEntry`'s header stays within 64 bytes (`queue_entry_stays_compact`); the drain benchmarks lose 50% to 150% past one cache line.
- **Calling `update_queue` or `admission_barrier`.** Both make the actor re-ask every blocked lane, gate-held ones included, which an arrival or a completion does only when a dispatch gate wakes the queue. Do not add a call on a per-request path, and keep them out of a measured loop: one barrier per cycle takes the `session_lanes` bench from 26 µs to 310 µs at 10,000 lanes. `admission_core`'s `gate_wake` case times the same recheck on every completion.
- **Editing a full recheck.** With a gate installed it runs on every wake, so it is a per-request path. Ask each group in place and move only a lane whose answer changed, as `gate_held` and the shared group do: taking every shared lane out and filing it again cost `gate_wake` 4.8 ms a completion against 1.5 ms with 10,000 lanes waiting and 10,000 held. The same holds when the wake arrives with a free slot and every waiting lane may go: they stay in the shared group and its best head is promoted, as on a capacity recheck. Making each of them a candidate, where all but one were blocked again on the next pass, cost `gate_wake_room` 6.4 ms a completion against 1.4 ms at the same sizes. The frontend ends a request's lifecycle before it frees the slot, but the wake goes through the classifier's event delivery and the free goes straight to the actor, so the wake can arrive with the free or after it: `gate_wake` and `gate_wake_room` bound what a wake costs.
- **Before the PR.** `cargo bench -p dynamo-kv-router --bench policy_queue` times the queue alone, and `cargo bench -p dynamo-kv-router --bench admission_core --features bench` one arrival, completion, and admission through the scheduler with a backlog waiting. Compare the 10,000 sizes. On a shared machine, build both arms' bench executables, run them alternately for three rounds, and compare medians: back-to-back runs of the same code have differed by tens of percent.
