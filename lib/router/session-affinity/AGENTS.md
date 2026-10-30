# lib/router/session-affinity

- This crate is the session affinity stage: the binding table, leases, versions, the replica sink port, config, and the replication wire. The mesh that carries the wire and the frontend coordinator stay outside (kv-router adapter, lib/llm host).
- Depend only on `dynamo-router-protocol` among router crates. Never add `dynamo-runtime`, `dynamo-kv-router`, another stage crate, axum, reqwest, zmq, or prometheus; the gate rejects them. Tokio is allowed for `Notify` and timers, never for sockets.
- Items used by files still in `lib/kv-router` are `pub` with a doc comment, never `#[doc(hidden)] pub`.
- Run `scripts/router_layering_check.sh` before committing.
