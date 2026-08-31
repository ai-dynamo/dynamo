# lib/llm/src/kv_router

## Module map

- `kv_router.rs` estimates how much of a request each worker already has cached, then asks the scheduler to choose a worker. It selects a worker but does not send the request.
- `scheduler.rs` connects discovered workers and their current load to the scheduling code in `lib/kv-router`.
- `indexer/` and `route_lookup.rs` track which KV-cache blocks each worker holds and look for reusable blocks before selection.
- `publisher/` receives KV-cache events and worker metrics from inference engines.
- `routing_host.rs` owns selection, dispatch, and response cleanup. `routing_host/kv_selection.rs` chooses KV-routed workers, `routing_host/request_guard.rs` tracks progress and cleanup, and `routing_host/cancellation.rs` stops unfinished work when the client cancels.
- `prefill_router/` optionally runs a request on a prefill worker before sending it to decode.
- `encoder_router.rs` optionally runs multimodal inputs through an encoder worker before token generation.
