# lib/router/kv-indexer

`dynamo-kv-indexer` is the KV indexer stage: KV events in, prefix knowledge out. Index structures, the ingest backend, lookup, recording, session updates, the request-plane DTOs, identity material, `IndexerConfig`, the transfer candidate vocabulary and the residency owner key it emits, the cache-owner hint-source metadata (`RouterHintSourceMetadata`) hosts publish and the residency snapshot resolves, and its counters. The HTTP server, the ZMQ listener and worker registry, and Prometheus exposition stay in `dynamo-kv-router` as adapters; the `export` feature is the transitional seam.

- Depends on (Dynamo crates): `dynamo-kv-events`, `dynamo-kv-hashing`, `dynamo-router-protocol`, `dynamo-tokens`.
- Must not depend on: other stages, the kernel, kv-router, `dynamo-kv-hints`, the runtime, any socket transport (tokio for channels and the indexer actors, never for sockets).
- Hosts import this crate through its `host` module. The DC relay (`lib/llm` pool relay, the relay benches) imports the `cuckoo` module, whose root re-export list is the relay contract; its submodules are private and nothing else in this crate uses it.
- Modules are private except `host`, `testing`, and `cuckoo` (the DC relay contract). The root list in `src/lib.rs` is the in-router contract (kernel, composition, other stages); `host` is the subset promised to hosts outside `lib/router`. A new cross-crate name goes on the root list, and on `host` only when a host needs it.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`.
