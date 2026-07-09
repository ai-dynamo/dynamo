@AGENTS.md

## Policy-Class Admission Control

Policy-class admission is a queue-owned lifecycle, not push-router or strategy-specific routing logic.

Flow:

1. `SchedulerQueueActor` resolves the policy class and calls its `PolicyClassAdmissionStrategy::admit` implementation with an opaque admission ID, optional session ID, full `context_tokens`, and a live `WorkerEligibility` handle.
2. `Bypass` continues through existing scheduling without lifecycle events. `Ready` continues through the existing queue and selector, optionally with an exact-worker constraint. `Defer` remains in the policy queue until the strategy returns `MakeReady`.
3. After selection and booking, the queue records `request_id -> { admission ticket, selected worker }`. This map is the authority for whether a request has an admission lifecycle; do not propagate an admission flag through `SchedulingResponse` or `WorkerSelection`.
4. Once the scheduling response is delivered, `RequestGuard` owns cleanup. It reports generic `Dispatched`, `Completed { context_tokens }`, or `Aborted` transitions through the scheduler. The queue ignores these calls when the request has no active admission ticket.
5. `Completed` commits the final logical context only after normal stream exhaustion. Dispatch failure, cancellation, backend error, or guard drop produces `Aborted`; partial output is not committed.

Keep `LocalScheduler::free` and the raw `MarkFree` protocol as slot cleanup only. Admission completion is an additive request-lifecycle path. Do not add per-output-token admission events or ThunderAgent branches to `lib/llm/src/kv_router/push_router`.

The public strategy contract lives in `lib/kv-router/src/scheduling/queue_admission/`. Concrete strategies and their registration live in `lib/admission-control/`. Add request facts as typed accessors only when a real strategy needs them; expose changing system state through a narrow provider instead of the actor-owned `SchedulingRequest` or an untyped metadata bag.
