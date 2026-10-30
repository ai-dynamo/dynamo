# lib/router/slot-tracker

- This crate is the slot tracker stage: active sequences, prompt registry, block and prefill trackers, load projection, fullness, the booking descriptor, and the replication wire. The replica mesh, the registry, and the HTTP server stay in `dynamo-kv-router` as adapters.
- Depend only on `dynamo-router-protocol` among router crates. Never add `dynamo-runtime`, `dynamo-kv-router`, `dynamo-kv-indexer`, axum, reqwest, zmq, or prometheus; the gate rejects them. Tokio is allowed for channels and timers, never for sockets.
- The router reads worker config through `WorkerConfigLike`; this crate never does. Capacity arrives as `dynamo_router_protocol::load::WorkerCapacity`, rank ranges as values.
- Files import shared sibling-crate names through `crate::prelude` (crate-private, explicit list).
- Items used by files still in `lib/kv-router` are `pub` with a doc comment, never `#[doc(hidden)] pub`.
- Run `scripts/router_layering_check.sh` before committing.
