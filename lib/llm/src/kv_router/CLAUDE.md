# lib/llm/src/kv_router

This layer connects request routing and response streaming to scheduler-owned state. Keep admission policy and queue bookkeeping in `dynamo-kv-router`.

## Admission lifecycle handoff

- `RequestGuard` takes ownership of admission progress and `AdmissionLease` immediately after selection, before the next fallible await. Do not split cleanup ownership across another guard or spawned task.
- After backend dispatch succeeds, use `RequestGuard::mark_dispatched`; it records the lease fallback before awaiting the live scheduler event.
- A successful `Stop`, `EoS`, or `Length` item is the commit point: mark completion before yielding it, then end the wrapper when that yield resumes. Natural EOF is also completion; cancellation and typed or annotated errors abort.
- Admission-managed terminal cleanup drops the lease so the scheduler actor releases booking and emits lifecycle events. Do not also call the legacy `KvRouter::free` path.
- Streamed context progress is monotonic and lock-free. Keep it off the actor command channel and update it only through `RequestProgressUpdater`.
