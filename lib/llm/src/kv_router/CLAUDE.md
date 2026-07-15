# lib/llm/src/kv_router

This layer connects request routing and response streaming to scheduler-owned state. See the [scheduling lifecycle walkthrough](../../../kv-router/src/scheduling/CLAUDE.md#policy-class-admission-lifecycle) for the actor-owned flow; keep admission policy and queue bookkeeping there.

## Response-stream handoff

- `RequestGuard` takes ownership of admission progress and `AdmissionLease` immediately after selection, before the next fallible await. Do not split cleanup ownership across another guard or spawned task.
- After backend dispatch succeeds, use `RequestGuard::mark_dispatched`; it records the lease fallback before awaiting the live scheduler event.
- A successful `Stop`, `EoS`, or `Length` item is the commit point: mark completion before yielding it, then end the wrapper when that yield resumes. Natural EOF is also completion; cancellation and typed or annotated errors abort.
- Terminal cleanup drops the lease; do not also call the legacy `KvRouter::free` path.
