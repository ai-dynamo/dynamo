# lib/router/scheduler

`dynamo-scheduler` is the router's kernel: it runs the stages for one request and owns the request and response types, the queue actor, and `SelectionCore`. Hosts outside `lib/router` import through `host`. Modules are private except `host` and `test_utils`: the root list in `src/lib.rs` is the in-router contract, `host` is the subset promised to hosts outside `lib/router`, and a type another crate owns is imported from that crate, not re-exported here.

## Queue actor: due times and controlled delay

- `SchedulerQueueActor::admit_one` is the required admission path: compute projected load, reject a request whose due time has passed, select a worker, skip the capacity reservation if the response receiver is closed or the due time has passed, then reserve capacity before responding. Failed response delivery must release that capacity. `extract_next` hands the actor the DRR winner with its cost unpaid; `settle` charges it only for an admitted request, so expiry, cancellation, and failed selection, booking, or delivery spend no credit.
- Each class queue has its own expiry timer (`ClassTimers`), armed at that class's earliest enforced due time; when it fires the actor rejects that class's overdue entries, spends no credit, and rearms. A due time is also checked before enqueue limits, before each insertion attempt, before each extraction, after an optional overlap refresh, right before worker selection, and right before booking or untracked delivery. A limit refusal stands only after due times are judged again at a fresh instant (`insert_arrival`). A rejection whose response receiver is already closed is dropped uncounted. All deadline math uses `tokio::time::Instant`.
- The optional overlap refresh is the only await inside a drain. Scheduler traffic that arrives during it is handled at once, each event in a new actor turn, and never abandons it; rows prepared before such a turn are not used after it. Never hold a command reply across that await. The candidate has left its class queue, so class expiry does not cover it: the refresh races the candidate's own due time (`refresh_candidate`; `OverlapScoresRefresh` owns the refresh timeout).
- Tests that need a profile build it with `PolicyProfile::from_classes`; the YAML file is the composed router's, and its queue tests live in `lib/router/kv-router/src/scheduling/queue_yaml_tests.rs`.

## Exclusive affinity

Costing, classification, queue placement, readiness and selection share the same
`QueueEligibility` snapshots. An eligible exclusive soft target supplies the cached
work and capacity checks; an ineligible target permits fallback within the original
constraints. Exact pins remain exact. Prepared rows keep main's configuration,
availability and overload freshness checks. Callers may project loads and fill rows;
the actor repairs ranks it wrote since, or projects again. Record every actor-side
write to a rank's load in `ActorWriteLog` after the write, or a prepared projection
misses it. Configured reject thresholds keep the full load and eligibility pass on
the actor.

Affinity lanes wake on target capacity changes, matching arrivals, or a 100 ms
`AFFINITY_RECHECK_INTERVAL` while affinity backlog exists. This is a recheck cadence,
not an admission-delay guarantee. Known fallback heads recheck on every dispatch
pass. Ordinary requests retain `NoEndpoints` when no allowed endpoint is available.
