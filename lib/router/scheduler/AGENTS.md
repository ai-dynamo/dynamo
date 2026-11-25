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
availability and overload freshness checks. Callers may project loads, count the
ranks past configured reject thresholds in the same pass, and fill rows; the actor
repairs and judges again the ranks it wrote since, or projects again. Record every
actor-side write to a rank's load in `ActorWriteLog` after the write, or a prepared
projection misses it. A write from another task after the caller's pass (a release,
a prefill mark, a replica's booking) is not re-read: that selection keeps the
caller's load and fullness for the rank. When adding an await between receiving a
request and admitting it, drop its judged prepared load after the await, as the
overlap refresh does.

Affinity lanes wake on target capacity changes, matching arrivals, or a 100 ms
`AFFINITY_RECHECK_INTERVAL` while affinity backlog exists. This is a recheck cadence,
not an admission-delay guarantee. Known fallback heads recheck on every dispatch
pass. Ordinary requests retain `NoEndpoints` when no allowed endpoint is available.

## SelectionCore

- Before you change booking, release, the session hold or commit, or selection-id claims, read the invariants at the top of `src/selection/core/run.rs`. Each names the test that pins it. Keep those tests passing, and add one when you add a booking owner or an await between booking and install.
- A new lock site takes `entries` before `reservation_index`, never the reverse; the reservation sweep nests them in that order. `indexed_booking` in `src/selection/core/reservations.rs` shows how to release one before taking the other.
- A run dropped at any await after the booking lands must still free the booking and the id claim. Keep the booking in its guard until it is installed.

## Performance

PR numbers point to worked examples; open one only when you need to see how a rule was applied.

One `SchedulerQueueActor` task per partition admits every request, so its time per admission caps that partition's request rate. Check a change against the cases below; the `dynamo-perf-best-practices` skill has the general rules and the evidence a PR needs.

- **Adding work to an admission.** If it depends only on the request and worker configuration, do it on the caller's task in `prepare_candidates`. If it depends on load the actor books, the caller may compute it, but the actor must re-read the ranks it wrote since (`ActorWriteLog`, `adopt_prepared_load`). Either way keep the actor-side path: pinned requests and policies with filters get no prepared rows, a prepared load is dropped when the reject thresholds, a rank's capacity or the set of ranks changed since its pass, and prepared rows last only the actor turn that received the request (#17465, #18540).
- **Adding anything inside a loop over ranks.** It runs once per rank per admission, thousands of times at fleet scale. Don't add an allocation, a lock, or a SipHash probe to it; use what the row already carries, its position, or an Fx map (#18008).
- **Adding a check that can exclude workers.** Drop those rows from the prepared set with `PreparedRows::retain`. Rebuild every row only when the check changes what preparation itself would build (#18009).
- **Adding a check that can stop early.** Read ranks one at a time through `worker_load_reader` and stop at the first answer. Don't project every rank first, and don't reuse load read for an earlier admission (#17480).
- **Adding an input or a per-request computation.** Run it only when a component declares the input or the class can use the result, as `projects_overlap` and `queueing_enabled` do. Work nobody reads still runs for every request (#15423, #18528).
- **Adding a log line or tracing event.** Emit it from the caller after the reply, with the values carried in the response as `SelectionLogFields` does. An event the actor must emit per candidate goes behind one `tracing::enabled!` check outside the loop, as `log_scored_candidates` does (#17700).
- **Changing `ADMISSION_BATCH` or `ADMISSION_CHANNEL_CAPACITY`.** They bound how long the actor holds its thread and how many rows are in flight; their comments give the numbers. Keep a full channel plus one batch well under ranks / `PREPARED_LOAD_MAX_STALE` at fleet scale: with more bookings than that ahead of a caller, the actor projects every rank for each admission and a backlog stops draining. Report admission latency as well as throughput (#18538, #17465, #19236).
- **Replacing a shared snapshot.** `AvailableWorkerIds` caches by `Arc` pointer. A producer that rebuilds an equal set must hand back the same `Arc`, or every admission rebuilds the derived set (#18008).
- **Refactoring the per-rank loops.** Measure even a removal or a reorder. The loops are sensitive to memory access order across threads; see the `NOTE(perf)` in the slot tracker's `reproject_worker_loads` (#18540).
- **Before the PR.** `cargo bench -p dynamo-kv-router --bench selection_core --features ingest-zmq,replica-sync-zmq,peer-recovery,bench` (up to 4,096 workers, with queue-gate and reject-threshold cases) and `cargo bench -p dynamo-router-policies --bench default_lifecycle` give the cost of one admission; `cargo bench -p dynamo-kv-router --bench policy_queue` covers the class queues. Compare the largest sizes, since small fleets hide per-rank costs. All three run the caller and the actor on one thread, so they cannot show a change that moves work between the two or that depends on memory shared across threads. Measure those with many concurrent callers on a multi-thread runtime, and say how in the PR.
- **Don't re-propose without new evidence.** Projecting load on the caller while the actor still fills the rows measured slower than doing both on the caller (#18540). A smaller admission lease gained under 2% with admission off and nothing measurable with it on (#11879).
