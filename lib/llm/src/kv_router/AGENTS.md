# lib/llm/src/kv_router

This module is the router's host layer around the `lib/router/kv-router` selection core. Read
[`README.md`](README.md) before changing selection inputs, booking ownership, cancellation,
prefill activation, or the encode and prefill hops; it explains the layer map, per-stage roles,
and the booking lifecycle these rules protect.

## Module map

- `kv_router.rs`: `KvRouter`. Estimates how much of a request each worker already has cached,
  then asks the scheduler to choose a worker. It selects and books, but never sends a request.
- `embedded.rs`: connects discovered workers and their load to one `SelectionService` partition
  from `lib/router/kv-router`, and runs each selection through `SelectionCore`.
- `indexer/`: tracks which KV-cache blocks each worker holds; lookups run in `lib/router/kv-router`.
- `publisher/`: receives KV-cache events and worker metrics from inference engines.
- `active_load.rs`: `ActiveLoad`, the load message workers publish on the event plane. Written
  by `publisher/`, read by `metrics_subscriber.rs`, `discovery/worker_monitor.rs`, and `pool_relay/`.
- `metrics.rs`, `metrics_subscriber.rs`: router Prometheus metrics, and the subscription to
  workers' KV load metrics.
- `sequence.rs`, `sequence/`: runtime glue for active-sequence tracking and its replica sync.
- `request_lease.rs`: one request-liveness coordinator and reaper per `KvRouter`. Holds booking
  leases and releases them on completion or expiry.
- `routing_load.rs`: the load context shared by a worker set's scheduler and its load metrics.
- `routing_host.rs`: `RoutingHost`, the pipeline's terminal engine. Owns selection, dispatch,
  and response cleanup for one worker set.
  - `routing_host/kv.rs`: the KV path: preview, plan, select, track, dispatch.
  - `routing_host/kv_selection.rs`: turns a request's routing inputs into a `KvRouter` call.
  - `routing_host/hosted.rs`, `routing_host/occupancy.rs`: non-KV modes (random, round-robin,
    load-based) and Direct mode.
  - `routing_host/request_guard.rs`: tracks a dispatched request's progress and releases its
    booking.
  - `routing_host/cancellation.rs`: what a client disconnect cancels, and the cleanup budget.
- `prefill_router/`: `PrefillRouter`, the optional prefill hop in front of decode.
  - `activation.rs`: target and binding lifecycle; how the prefill hop's routing mode is chosen.
  - `admission.rs`: consumes the prefill stream, in the foreground or as a background task.
  - `handoff.rs`: keeps prefill running while decode runs and reports prefill failures.
  - `conditional_bypass.rs`: conditional disaggregation's decision to skip remote prefill.
  - `query.rs`: `reserve_prefill_worker`, the prefill-only booking used by the Runtime EPP.
- `encoder_router.rs`: `EncoderRouter`, the optional encode hop for multimodal inputs.
- `plugins.rs`: installs the linked router plugin registry.
- `shared_cache.rs`: the HiCache shared KV cache client (SGLang + Mooncake).

## Guardrails

- `KvRouter` never dispatches. `RoutingHost` dispatches decode, aggregated, and prefill requests
  (the prefill hop through the prefill set's own `RoutingHost`); `EncoderRouter` dispatches
  encode requests through a round-robin `PushRouter`.
- Each booking has exactly one owner at a time (`RoutePlan`, then `RequestGuard`, or a detached
  lease) and is released exactly once.
- A preview never books capacity and never creates an affinity binding.
- Validate request-supplied worker pins only at initial selection. A pinned worker that
  disappears after a preview or at dispatch is a service failure, not a client error.
- Decode and aggregated routing own the classifier lifecycle. The prefill hop must not start
  one.
- Never drop a decode leg that carries staged KV before it reaches its worker, except when its
  cleanup budget runs out.
- Fail closed while the prefill target is `Pending`. Pass through only when `Untargeted`.
- Do not let clients request a conditional bypass.
- When you add a selection input, add it to the inputs table in `README.md` and decide whether
  the Runtime EPP (`deploy/inference-gateway/ext-proc/src/epp.rs`) must supply it too.
- The Runtime EPP builds its own decode scoring override (`decode_router_config_override` in
  `epp.rs`). Change it together with `build_decode_router_override` in
  `prefill_router/mod.rs`, or the two hosts score decode differently.
- Keep `README.md` in step with changes to the layer map, per-stage roles, booking lifecycle,
  cancellation, or prefill activation.

## Performance

PR numbers point to worked examples; open one only when you need to see how a rule was
applied.

The selection path's rules are in `lib/router/scheduler/AGENTS.md` and the index's in
`lib/router/kv-indexer/AGENTS.md`. These cover the host's ingest and sync paths.

- **Adding a stage to KV event ingest.** Keep per-publisher work off the shared reader. On the
  brokered path, give each publisher a bounded lane and drop the newest batch from a full one,
  as `indexer/recovery/broker_zmq.rs` does. On the direct path a full lane makes its socket's
  reader wait instead of dropping, so a slow stage stalls every publisher on that socket, and
  once the stall fills the ZMQ high-water marks the publishers drop. On neither path raise a
  channel capacity or a ZMQ high-water mark to hide a slow stage (#14167, #16568).
- **Adding state that every request or load event touches.** If they write it, shard it by
  worker as `ACTIVE_LOAD_SHARDS` does. If they only read it, publish a copy-on-write snapshot.
  One mutex there is contended by every task on the path (#16118).
- **Reacting to a discovery change.** Do work for the workers that changed, not for all of
  them. Work over the whole fleet on each change is quadratic at startup (#16568).
- **Publishing per-request events to peer routers.** Enqueue synchronously into the bounded
  queue in `sequence.rs`; don't spawn a task per event. Exclude your own publisher where the
  events are received (`excluded_publisher_id`): the `router_id` check at apply still pays to
  receive and decode each one (#11920, #11992, #17114).

## Testing

- Unit tests live beside the code: `routing_host/tests.rs` and the tests in `prefill_router/`.
  On macOS run `cargo test -p dynamo-llm --no-default-features --lib kv_router::`.
- Some ZMQ, event-plane, and replica-sync tests bind local ports and can fail inside a sandbox
  or in a full parallel run. Rerun a failing one alone before treating it as a regression.
- End-to-end router tests are described in the
  [router testing guide](../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/router-testing.md).
