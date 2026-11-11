# lib/router/admission

`dynamo-admission` is the admission stage: admit, hold, or reject now. The per-class policy queue and its stats, the resolved policy profiles, the classifier runtime with request lifecycles, the classifier plugin API (`RequestClassifier`, `ClassifyRequest`, `WorkerView`), `AdmissionConfig`, and `AdmissionError`; the YAML policy file, the plugin registry, and the queue's HTTP exposure stay in `dynamo-kv-router`. The kernel is the only caller that admits, holds, or rejects.

- Depends on (Dynamo crates): `dynamo-router-protocol`, `dynamo-session-affinity` (the affinity target), `dynamo-slot-tracker` (the fullness rule), `dynamo-tokens`.
- Must not depend on: other stages except `dynamo-slot-tracker` and `dynamo-session-affinity`, the kernel, kv-router, the runtime, any transport (tokio for the queue's instants and the classifier's delivery task only).
- Hosts import this crate through its `host` module; classifier plugins import `dynamo_kv_router::plugins::prelude`.
- Modules are private except `host`. The root list in `src/lib.rs` is the in-router contract (kernel, composition, other stages); `host` is the subset promised to hosts outside `lib/router`. A new cross-crate name goes on the root list, and on `host` only when a host needs it.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.

## Policy queue: classes, order, and expiry

- `PolicyQueue` does not classify requests. It owns one flat list of class queues, selected by name (`PolicyProfile::resolve_class_index`; an unknown name is `UnknownPolicyClassError`), and gives each class weighted turns by deficit round robin. `extract_next` returns the winning head with a `DispatchCharge`; `settle` spends that cost only for an admitted request. `pop_next` charges at once, for callers without an admission step.
- Within one strict-priority tier, FCFS/LCFS/WSPT order deadline-bearing entries earliest-due-first ahead of all non-deadline entries before the policy score applies; class due times never enter their order. An EDF class, the policy-class YAML default, ignores strict priority and the score: it orders by the explicit deadline, else router ingress plus the class `slo_ttft_ms`, then enqueue order. Due-time keys (`DueKey`) share one origin fixed at queue construction; a due time before it keys as the origin.
- Controlled delay is a per-class setting (`controlled_delay`, EDF only; configuration rejects it elsewhere). `PolicyQueue::with_class_budget_expiry` fixes each class's expiry at construction: an EDF class with controlled delay expires every finite due time at its lane roots (`expiring_roots`, blocked lanes included); every other class expires only explicit deadlines through its own `due_entries` index. `PolicyQueue::new` (offline replay) expires only explicit deadlines. Any removal of a blocked lane's root, by expiry or cancellation, unblocks the lane.
- Per-worker lanes stay: a full worker blocks only its own lane's head. `take_expired` pops the overdue prefix from `expiring_roots` in due order, O(log n) per entry; a class without head expiry finds its expired explicit deadlines in `due_entries` with one scan and a heap rebuild.
- The queue's indexes are exposed to tests only through the `testing` hooks at the end of `queue.rs`; the profile-driven tests live in `lib/router/kv-router/src/scheduling/policy_queue_tests.rs`.
