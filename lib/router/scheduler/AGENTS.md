# lib/router/scheduler

`dynamo-scheduler` is the router's scheduler kernel: it runs the stages (index lookup, slot tracking, session affinity, worker selection, admission) atomically for one request and owns the request and response vocabulary, eligibility, the queue actor and `LocalScheduler`, the overlap and prefill-load analysis, the selection kernel, the validated kernel configuration, and the tracking hash. `protocols` holds the kernel's own types and re-exports nothing.

- Depends on (Dynamo crates): `dynamo-admission`, `dynamo-kv-events`, `dynamo-kv-hashing`, `dynamo-kv-hints`, `dynamo-kv-indexer`, `dynamo-multimodal`, `dynamo-router-protocol`, `dynamo-session-affinity`, `dynamo-slot-tracker`, `dynamo-tokens`, `dynamo-worker-selection`.
- Must not depend on: kv-router, the runtime, any transport (axum, hyper, reqwest, zmq, prometheus); `dynamo-kv-router` composes it with those adapters.
- Hosts import this crate through its `host` module.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`; the kernel's invariants, lock order, and actor rules are in `README.md` under "Working rules".

## Queue actor: due times and controlled delay

- `SchedulerQueueActor::admit_one` is the required admission path: compute projected load, reject a request whose due time has passed, select a worker, skip the capacity reservation if the response receiver is closed or the due time has passed, then reserve capacity before responding. Failed response delivery must release that capacity. `extract_next` hands the actor the DRR winner with its cost unpaid; `settle` charges it only for an admitted request, so expiry, cancellation, and failed selection, booking, or delivery spend no credit.
- Each class queue has its own expiry timer (`ClassTimers`), armed at that class's earliest enforced due time; when it fires the actor rejects that class's overdue entries, spends no credit, and rearms. A due time is also checked before enqueue limits, before each insertion attempt, before each extraction, after an optional overlap refresh, right before worker selection, and right before booking or untracked delivery. A limit refusal stands only after due times are judged again at a fresh instant (`insert_arrival`). A rejection whose response receiver is already closed is dropped uncounted. All deadline math uses `tokio::time::Instant`.
- The optional overlap refresh is the only await inside a drain. Scheduler traffic that arrives during it is handled at once, each event in a new actor turn, and never abandons it; rows prepared before such a turn are not used after it. Never hold a command reply across that await. The candidate has left its class queue, so class expiry does not cover it: the refresh races the candidate's own due time (`refresh_candidate`; `OverlapScoresRefresh` owns the refresh timeout).
- Tests that need a profile build it with `PolicyProfile::from_classes`; the YAML file is the composed router's, and its queue tests live in `lib/router/kv-router/src/scheduling/queue_yaml_tests.rs`.
