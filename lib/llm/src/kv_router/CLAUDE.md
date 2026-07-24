# lib/llm/src/kv_router

This module chooses a worker, sends the request to that worker, and wraps the response stream. The scheduler owns queue state; this module owns request cleanup after worker selection.

## Module map

- `kv_router.rs` estimates cache overlap and asks the scheduler to choose a worker.
- `scheduler.rs` connects discovered workers and their current load to `lib/kv-router`.
- `indexer/` and `route_lookup.rs` track and query KV-cache blocks.
- `publisher/` receives KV-cache events and worker metrics.
- `push_router.rs` sends the request and wraps its response stream. `push_router/selection.rs` selects the worker, `push_router/request_guard.rs` tracks progress and cleanup, and `push_router/cancellation.rs` stops unfinished work when the client cancels.
- `prefill_router/` and `encoder_router.rs` handle optional prefill and multimodal stages.

## Response-stream rules

- Move the optional progress updater and `RequestLifecycleLease` into `RequestGuard` immediately after selection and before backend dispatch.
- Publish prompt-plus-output progress at output-block boundaries. On a successful terminal item, publish authoritative completion usage before yielding the item; a normal stream close repeats that update idempotently.
- A normal stream close finishes the request. Cancellation, transport errors, and typed error or cancellation finish reasons abort it.
- Dropping `RequestLifecycleLease` performs generation-safe scheduler cleanup. Do not also call `KvRouter::free` for the same request.
