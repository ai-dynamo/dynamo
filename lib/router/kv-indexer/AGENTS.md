# lib/router/kv-indexer

- This crate is the KV indexer stage: index structures, ingest backend, lookup, the indexer's own error and config slice, and its counters. HTTP servers, the ZMQ listener and worker registry, and anything that needs `dynamo-runtime` stay in `dynamo-kv-router` as adapters.
- Depend only on `dynamo-router-protocol`, `dynamo-kv-events`, `dynamo-kv-hashing` among router crates. Never add `dynamo-runtime`, `dynamo-kv-router`, axum, reqwest, or zmq; the gate rejects them. Tokio is allowed for channels and the indexer actors, never for sockets.
- Files import shared sibling-crate names through `crate::prelude` (crate-private, explicit list). Add a name there rather than a per-file `use` of a sibling crate.
- `export` is transitional. Do not add new Prometheus types; new counters are atomics read by a `stats()` snapshot.
- Items used by files still in `lib/kv-router` are `pub` with a doc comment, never `#[doc(hidden)] pub`.
- Run `scripts/router_layering_check.sh` before committing.
