# lib/llm/src/kv_router

This module chooses a worker, sends the request to that worker, and wraps the response stream. The scheduler decides when work may run and owns queue state; this module owns cleanup after worker selection. See the [scheduling lifecycle walkthrough](../../../kv-router/src/scheduling/CLAUDE.md#policy-class-admission-lifecycle) for the full admission flow.

## Module map

- `kv_router.rs` estimates how much of a request each worker already has cached, then asks the scheduler to choose a worker. It selects a worker but does not send the request.
- `scheduler.rs` connects discovered workers and their current load to the scheduling code in `lib/kv-router`.
- `indexer/` and `route_lookup.rs` track which KV-cache blocks each worker holds and look for reusable blocks before selection.
- `publisher/` receives KV-cache events and worker metrics from inference engines.
- `push_router.rs` sends a request to the selected worker and wraps its response stream. `push_router/selection.rs` chooses the worker, `push_router/request_guard.rs` tracks progress and cleanup, and `push_router/cancellation.rs` stops unfinished work when the client cancels.
- `prefill_router/` optionally runs a request on a prefill worker before sending it to decode.
- `encoder_router.rs` optionally runs multimodal inputs through an encoder worker before token generation.

## Response-stream rules

- Immediately after selection, move the token counter and `AdmissionLease` cleanup handle into `RequestGuard`. The counter tracks the request's current prompt-plus-output length. Do this before sending the request to the worker so failure or cancellation still releases the worker reservation. Do not split cleanup ownership across another guard or spawned task.
- After the worker accepts the request, call `RequestGuard::mark_dispatched`. It records dispatch on the cleanup handle before notifying the scheduler, so cleanup still reports `Dispatched` before `Completed` or `Aborted` if that notification is cancelled.
- Mark `Stop`, `EoS`, or `Length` complete before sending the item to the caller, then stop when the caller asks for the next item. A normal stream close also completes; cancellation and error response items abort.
- For requests with an `AdmissionLease`, dropping the handle performs scheduler cleanup. Do not also call `KvRouter::free`.
